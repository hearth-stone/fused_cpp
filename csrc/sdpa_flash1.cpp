// ── FlashAttention-1 风格 SDPA 内核 ─────────────────────────────────────
//
// 计算结构（与 FA1 论文一致）：
//   外层：K/V 序列 S 维以 BLOCK_S 分块（j 块）
//   中层：Q  序列 L 维以 BLOCK_L 分块（i 块）
//   对每对 (j, i)：
//     1. S_ij = Q_i @ K_j^T * scale，应用 mask
//     2. 维护每个 i 块的 (m_i, ℓ_i, O_i) 三元组：
//          m_new = max(m_old, m_block)
//          ℓ_new = e^{m_old-m_new} * ℓ_old + sum(e^{S_ij - m_new})
//          O_new = e^{m_old-m_new} * O_old + P_ij @ V_j
//   循环结束后统一执行 O_i /= ℓ_i 完成最终归一化。
//
// 与 FA2 风格的差异：
//   * FA1: 外层 K/V，需要在外层迭代之间持久化 (m_i, ℓ_i, O_i) 状态。
//   * FA2: 外层 Q 行，每行内部完成一次 online softmax 即可（无需块级持久状态）。
//
// 实现要点：
//   * (b, n) 在外层并行（OpenMP）；同一个 (b, n) 内部串行处理 j 与 i 块循环，
//     保证 (m_i, ℓ_i, O_i) 的更新顺序正确。
//   * (m_i, ℓ_i, O_i) 缓冲按 i 块大小（[BLOCK_L, Ev]）分配在堆上（Ev 可达
//     128~192，再乘 BLOCK_L 容易超过栈上限），由 std::vector 管理。
//   * 因果掩码 (`is_causal=True`)：
//       - 块级 early-skip：当 K/V 块的最小可见行 > 块的最大允许行时整块跳过；
//       - 边界块逐元素剪裁：将 -inf 写入 S_ij 越界元素。
//   * BLOCK_S / BLOCK_L 大于实际维度时自动退化为单块（block_len_actual 收敛）。
//   * 严禁全量物化 attn_scores[B, N, L, S]：每线程仅持有 (m, ℓ, O) +
//     scores_buf[BLOCK_L * BLOCK_S]。

#include <torch/extension.h>
#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>

#include "sdpa_common.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// ── BLOCK 默认尺寸 ────────────────────────────────────────────────────
constexpr int FA1_BLOCK_S = 64;
constexpr int FA1_BLOCK_L = 32;

template <typename scalar_t>
inline void sdpa_flash1_kernel_tmpl(const scalar_t* q_ptr, const scalar_t* k_ptr, const scalar_t* v_ptr,
                                    const SdpaParams& p) {
  // ── stride（contiguous [B, N, seq, dim] 布局）──
  const int64_t q_stride_b = p.N * p.L * p.E;
  const int64_t q_stride_n = p.L * p.E;
  const int64_t q_stride_l = p.E;
  const int64_t k_stride_b = p.N * p.S * p.E;
  const int64_t k_stride_n = p.S * p.E;
  const int64_t k_stride_s = p.E;
  const int64_t v_stride_b = p.N * p.S * p.Ev;
  const int64_t v_stride_n = p.S * p.Ev;
  const int64_t v_stride_s = p.Ev;
  const int64_t m_stride_b = p.N * p.L * p.S;
  const int64_t m_stride_n = p.L * p.S;
  const int64_t m_stride_l = p.S;
  const int64_t o_stride_b = p.N * p.L * p.Ev;
  const int64_t o_stride_n = p.L * p.Ev;
  const int64_t o_stride_l = p.Ev;

  const int64_t L = p.L;
  const int64_t S = p.S;
  const int64_t E = p.E;
  const int64_t Ev = p.Ev;
  const int64_t BS = FA1_BLOCK_S;
  const int64_t BL = FA1_BLOCK_L;
  const int64_t num_j_blocks = (S + BS - 1) / BS;
  const int64_t num_i_blocks = (L + BL - 1) / BL;

  const int64_t total_heads = p.B * p.N;

  // ── 每个 (b, n) 独立处理；并行在 batch * num_heads 维度 ──
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t bn = 0; bn < total_heads; ++bn) {
    const int64_t b = bn / p.N;
    const int64_t n = bn % p.N;

    // ── 私有状态：在所有 i 块上保留 (m_i[L], ℓ_i[L], O_i[L*Ev])。
    //
    // 由于 FA1 外层是 K/V (j 块)，所有 i 块的状态需要跨 j 迭代持久化。
    // 因此一次性按完整 L 长度分配，避免在 j 循环里重复 alloc。
    std::vector<float> m_state(static_cast<size_t>(L), p.neg_inf);
    std::vector<float> l_state(static_cast<size_t>(L), 0.0f);
    std::vector<float> o_state(static_cast<size_t>(L) * Ev, 0.0f);

    // scores 缓冲：复用为单块 [BL_actual, BS_actual]
    std::vector<float> scores_buf(static_cast<size_t>(BL) * BS);

    // ── 外层：遍历 K/V 块（j 块）──
    for (int64_t jb = 0; jb < num_j_blocks; ++jb) {
      const int64_t j_start = jb * BS;
      const int64_t j_len = std::min(BS, S - j_start);

      // ── 中层：遍历 Q 块（i 块）──
      for (int64_t ib = 0; ib < num_i_blocks; ++ib) {
        const int64_t i_start = ib * BL;
        const int64_t i_len = std::min(BL, L - i_start);

        // ── 因果掩码块级 early-skip ──
        //
        // 整块 (i, j) 不可见的条件：
        //   block_start_j > causal_limit_for_block_i_max
        // 其中 causal_limit_for_l = l + causal_offset。
        // 块内最大 l 值为 i_start + i_len - 1，因此整块不可见
        // 当且仅当 j_start > (i_start + i_len - 1) + causal_offset。
        if (p.is_causal) {
          const int64_t i_max_l = i_start + i_len - 1;
          const int64_t max_visible_s = i_max_l + p.causal_offset;
          if (j_start > max_visible_s) {
            // 整块不可见，跳过
            continue;
          }
        }

        // ── 步骤 1: 计算 S_ij = Q_i @ K_j^T * scale ──
        for (int64_t ii = 0; ii < i_len; ++ii) {
          const int64_t l = i_start + ii;
          const scalar_t* q_row = q_ptr + b * q_stride_b + n * q_stride_n + l * q_stride_l;
          float* s_row = scores_buf.data() + ii * BS;
          for (int64_t jj = 0; jj < j_len; ++jj) {
            const int64_t s_idx = j_start + jj;
            const scalar_t* k_row = k_ptr + b * k_stride_b + n * k_stride_n + s_idx * k_stride_s;
            float dot = 0.0f;
            for (int64_t e = 0; e < E; ++e) {
              dot += static_cast<float>(q_row[e]) * static_cast<float>(k_row[e]);
            }
            s_row[jj] = dot * p.scale_f;
          }
        }

        // ── 步骤 2: 应用 additive attn_mask ──
        if (p.mask_ptr) {
          for (int64_t ii = 0; ii < i_len; ++ii) {
            const int64_t l = i_start + ii;
            const float* m_row = p.mask_ptr + b * m_stride_b + n * m_stride_n + l * m_stride_l + j_start;
            float* s_row = scores_buf.data() + ii * BS;
            for (int64_t jj = 0; jj < j_len; ++jj) {
              s_row[jj] += m_row[jj];
            }
          }
        }

        // ── 步骤 3: 应用 causal mask（边界块逐元素剪裁）──
        if (p.is_causal) {
          for (int64_t ii = 0; ii < i_len; ++ii) {
            const int64_t l = i_start + ii;
            const int64_t causal_limit = l + p.causal_offset;
            float* s_row = scores_buf.data() + ii * BS;
            for (int64_t jj = 0; jj < j_len; ++jj) {
              const int64_t s_idx = j_start + jj;
              if (s_idx > causal_limit) {
                s_row[jj] = p.neg_inf;
              }
            }
          }
        }

        // ── 步骤 4: 按 i 行更新 (m_i, ℓ_i, O_i) ──
        for (int64_t ii = 0; ii < i_len; ++ii) {
          const int64_t l = i_start + ii;
          float* s_row = scores_buf.data() + ii * BS;

          // (a) 当前块的局部最大值
          float m_block = p.neg_inf;
          for (int64_t jj = 0; jj < j_len; ++jj) {
            if (s_row[jj] > m_block) m_block = s_row[jj];
          }

          // 整块全部 -inf：跳过更新（不会带来贡献）
          if (m_block == p.neg_inf) {
            continue;
          }

          const float m_old = m_state[l];
          const float m_new = std::max(m_old, m_block);
          const float correction = std::exp(m_old - m_new);

          // (b) 累加修正：ℓ 与 O 已有值乘 correction
          l_state[l] *= correction;
          float* o_row = o_state.data() + l * Ev;
          for (int64_t ev = 0; ev < Ev; ++ev) {
            o_row[ev] *= correction;
          }

          // (c) 累加当前块贡献
          for (int64_t jj = 0; jj < j_len; ++jj) {
            const float p_val = std::exp(s_row[jj] - m_new);
            l_state[l] += p_val;
            const int64_t s_idx = j_start + jj;
            const scalar_t* v_row = v_ptr + b * v_stride_b + n * v_stride_n + s_idx * v_stride_s;
            for (int64_t ev = 0; ev < Ev; ++ev) {
              o_row[ev] += p_val * static_cast<float>(v_row[ev]);
            }
          }

          m_state[l] = m_new;
        }
      }  // end i-block loop
    }  // end j-block loop

    // ── 最终归一化：output[b, n, l, :] = O_state[l] / l_state[l] ──
    for (int64_t l = 0; l < L; ++l) {
      float* o_out = p.out_ptr + b * o_stride_b + n * o_stride_n + l * o_stride_l;
      const float ell = l_state[l];
      const float* o_in = o_state.data() + l * Ev;
      if (ell > 0.0f) {
        const float inv = 1.0f / ell;
        for (int64_t ev = 0; ev < Ev; ++ev) {
          o_out[ev] = o_in[ev] * inv;
        }
      } else {
        // 所有 scores 被掩码：输出零向量（与 naive/flash2 一致）
        for (int64_t ev = 0; ev < Ev; ++ev) {
          o_out[ev] = 0.0f;
        }
      }
    }
  }
}

}  // anonymous namespace

// ── dtype-erased 入口（注册到全局表）──
void sdpa_flash1_impl(const SdpaParams& p) {
  if (p.dtype == SdpaDtype::kBFloat16) {
    sdpa_flash1_kernel_tmpl<at::BFloat16>(static_cast<const at::BFloat16*>(p.q_ptr),
                                          static_cast<const at::BFloat16*>(p.k_ptr),
                                          static_cast<const at::BFloat16*>(p.v_ptr), p);
  } else {
    sdpa_flash1_kernel_tmpl<float>(static_cast<const float*>(p.q_ptr), static_cast<const float*>(p.k_ptr),
                                   static_cast<const float*>(p.v_ptr), p);
  }
}

REGISTER_SDPA_VERSION("flash1", sdpa_flash1_impl);
