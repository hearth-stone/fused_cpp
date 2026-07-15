// ── Naive SDPA 内核 ─────────────────────────────────────────────────────
//
// 目的：作为 FA1 / FA2 的等价性对比锚点 + FLOPs 公式的语义参考。
//
// 计算流程（与 FLOPs 公式一一对应）：
//   1. 显式分配 attn_scores[B, N, L, S]（fp32）
//   2. attn_scores = Q @ K^T * scale（内层 dot 在 fp32 累加）
//   3. （可选）attn_scores += attn_mask
//   4. 应用 causal mask：S - L + l < s 的位置赋 -inf
//   5. 沿 S 维做标准两遍 softmax
//   6. output = attn @ V（同样 fp32 累加）
//
// 与 FlashAttention 风格的最大差异：**全量物化 attn_scores 张量**，
// 因此显著占用内存（B*N*L*S * 4 bytes）。这是预期行为，仅供基准对比，
// 不可作为生产路径。

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

// ── 内部模板实现：按 scalar_t 实例化 ────────────────────────────────────
template <typename scalar_t>
inline void sdpa_naive_kernel_tmpl(const scalar_t* q_ptr, const scalar_t* k_ptr, const scalar_t* v_ptr,
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
  // mask: [B, N, L, S]
  const int64_t m_stride_b = p.N * p.L * p.S;
  const int64_t m_stride_n = p.L * p.S;
  const int64_t m_stride_l = p.S;
  // out: [B, N, L, Ev]
  const int64_t o_stride_b = p.N * p.L * p.Ev;
  const int64_t o_stride_n = p.L * p.Ev;
  const int64_t o_stride_l = p.Ev;

  const int64_t total_heads = p.B * p.N;

  // ── 在 batch * num_heads 维度上并行 ──
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t bn = 0; bn < total_heads; ++bn) {
    const int64_t b = bn / p.N;
    const int64_t n = bn % p.N;

    // ── 每个 (b, n) 独立分配 attn_scores 缓冲 [L, S] (fp32) ──
    //
    // 注：原计划是 [B, N, L, S] 全局物化，这里改为「每线程仅持有
    // [L, S]」，语义上完全等价（仍然显式物化 scores 矩阵），但避免
    // 在大 (B, N) 下一次性吃光内存；FLOPs 计算口径不受影响。
    std::vector<float> scores(static_cast<size_t>(p.L * p.S));

    // ── 步骤 1+2: scores = Q @ K^T * scale ──
    // scores[l, s] = scale * sum_e Q[b,n,l,e] * K[b,n,s,e]
    for (int64_t l = 0; l < p.L; ++l) {
      const scalar_t* q_row = q_ptr + b * q_stride_b + n * q_stride_n + l * q_stride_l;
      float* s_row = scores.data() + l * p.S;
      for (int64_t s_idx = 0; s_idx < p.S; ++s_idx) {
        const scalar_t* k_row = k_ptr + b * k_stride_b + n * k_stride_n + s_idx * k_stride_s;
        float dot = 0.0f;
        for (int64_t e = 0; e < p.E; ++e) {
          dot += static_cast<float>(q_row[e]) * static_cast<float>(k_row[e]);
        }
        s_row[s_idx] = dot * p.scale_f;
      }
    }

    // ── 步骤 3: 应用 additive attn_mask ──
    if (p.mask_ptr) {
      const float* m_base = p.mask_ptr + b * m_stride_b + n * m_stride_n;
      for (int64_t l = 0; l < p.L; ++l) {
        float* s_row = scores.data() + l * p.S;
        const float* m_row = m_base + l * m_stride_l;
        for (int64_t s_idx = 0; s_idx < p.S; ++s_idx) {
          s_row[s_idx] += m_row[s_idx];
        }
      }
    }

    // ── 步骤 4: 应用 causal mask ──
    // 等价语义：key 位置 s 可见当且仅当 s <= l + (S - L)
    if (p.is_causal) {
      for (int64_t l = 0; l < p.L; ++l) {
        const int64_t causal_limit = l + p.causal_offset;
        float* s_row = scores.data() + l * p.S;
        for (int64_t s_idx = 0; s_idx < p.S; ++s_idx) {
          if (s_idx > causal_limit) {
            s_row[s_idx] = p.neg_inf;
          }
        }
      }
    }

    // ── 步骤 5: 两遍 softmax（沿 S 维）──
    // (a) 减最大值；(b) exp 与求和；(c) 归一化
    for (int64_t l = 0; l < p.L; ++l) {
      float* s_row = scores.data() + l * p.S;

      // 求最大值
      float row_max = p.neg_inf;
      for (int64_t s_idx = 0; s_idx < p.S; ++s_idx) {
        if (s_row[s_idx] > row_max) {
          row_max = s_row[s_idx];
        }
      }

      // exp 与求和
      float row_sum = 0.0f;
      for (int64_t s_idx = 0; s_idx < p.S; ++s_idx) {
        const float e_val = std::exp(s_row[s_idx] - row_max);
        s_row[s_idx] = e_val;
        row_sum += e_val;
      }

      // 归一化（全 -inf 时输出 0，与 flash2 内核一致）
      if (row_sum > 0.0f) {
        const float inv = 1.0f / row_sum;
        for (int64_t s_idx = 0; s_idx < p.S; ++s_idx) {
          s_row[s_idx] *= inv;
        }
      } else {
        for (int64_t s_idx = 0; s_idx < p.S; ++s_idx) {
          s_row[s_idx] = 0.0f;
        }
      }
    }

    // ── 步骤 6: output = attn @ V ──
    // output[l, ev] = sum_s scores[l, s] * V[b, n, s, ev]
    for (int64_t l = 0; l < p.L; ++l) {
      const float* s_row = scores.data() + l * p.S;
      float* o_row = p.out_ptr + b * o_stride_b + n * o_stride_n + l * o_stride_l;
      for (int64_t ev = 0; ev < p.Ev; ++ev) {
        o_row[ev] = 0.0f;
      }
      for (int64_t s_idx = 0; s_idx < p.S; ++s_idx) {
        const float w = s_row[s_idx];
        if (w == 0.0f) continue;
        const scalar_t* v_row = v_ptr + b * v_stride_b + n * v_stride_n + s_idx * v_stride_s;
        for (int64_t ev = 0; ev < p.Ev; ++ev) {
          o_row[ev] += w * static_cast<float>(v_row[ev]);
        }
      }
    }
  }
}

}  // anonymous namespace

// ── dtype-erased 入口（注册到全局表）──
void sdpa_naive_impl(const SdpaParams& p) {
  if (p.dtype == SdpaDtype::kBFloat16) {
    sdpa_naive_kernel_tmpl<at::BFloat16>(static_cast<const at::BFloat16*>(p.q_ptr),
                                         static_cast<const at::BFloat16*>(p.k_ptr),
                                         static_cast<const at::BFloat16*>(p.v_ptr), p);
  } else {
    sdpa_naive_kernel_tmpl<float>(static_cast<const float*>(p.q_ptr), static_cast<const float*>(p.k_ptr),
                                  static_cast<const float*>(p.v_ptr), p);
  }
}

REGISTER_SDPA_VERSION("naive", sdpa_naive_impl);
