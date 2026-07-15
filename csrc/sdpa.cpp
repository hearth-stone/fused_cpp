#include <torch/extension.h>
#include <cmath>
#include <limits>
#include <algorithm>
#include "utils.h"
#include "sdpa_common.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// ── 旧版 SDPA 参数结构体（保留供历史接口使用）────────────────────────
// 与 SdpaParams 在数据上等价；为避免破坏旧 sdpa_flash_impl 模板代码，
// 这里保留独立的 LegacySdpaParams 结构。新增内核统一使用 sdpa_common.h
// 中的 SdpaParams + dtype-erased 接口。
struct LegacySdpaParams {
  int64_t B;
  int64_t N;
  int64_t L;
  int64_t S;
  int64_t E;
  int64_t Ev;
  float scale_f;
  float neg_inf;
  int64_t causal_offset;
  bool is_causal;
  float* out_ptr;
  const float* mask_ptr;
};

// ── FlashAttention-2 风格 SDPA 分块核心模板函数 ─────────────────────────
//
// 模板参数：
//   scalar_t  - 输入数据类型（float / at::BFloat16）
//   BLOCK_S   - K/V 序列维度分块大小（编译时常量，默认 64）
//
// 核心思想：
//   对每个 query 位置，将 K/V 的序列维度 S 按 BLOCK_S 分块，
//   逐块计算 Q@K_block^T → 应用掩码 → Online Softmax 更新 → 累加 attn@V_block。
//   消除 attn_scores[B, N, L, S] 中间张量的完整物化。
//
template <typename scalar_t, int BLOCK_S = 64>
inline void sdpa_flash2_kernel_impl(const scalar_t* q_ptr, const scalar_t* k_ptr, const scalar_t* v_ptr,
                                    const LegacySdpaParams& p) {
  // Q/K/V 的 stride：[B, N, seq, head_dim] contiguous 布局
  const int64_t q_stride_b = p.N * p.L * p.E;
  const int64_t q_stride_n = p.L * p.E;
  const int64_t q_stride_l = p.E;
  const int64_t k_stride_b = p.N * p.S * p.E;
  const int64_t k_stride_n = p.S * p.E;
  const int64_t k_stride_s = p.E;
  const int64_t v_stride_b = p.N * p.S * p.Ev;
  const int64_t v_stride_n = p.S * p.Ev;
  const int64_t v_stride_s = p.Ev;
  // mask stride：与 [B, N, L, S] 布局一致
  const int64_t m_stride_b = p.N * p.L * p.S;
  const int64_t m_stride_n = p.L * p.S;
  const int64_t m_stride_l = p.S;
  // output stride：[B, N, L, Ev]
  const int64_t o_stride_b = p.N * p.L * p.Ev;
  const int64_t o_stride_n = p.L * p.Ev;
  const int64_t o_stride_l = p.Ev;

  // K/V 序列维度的分块数量（向上取整）
  const int64_t num_blocks = (p.S + BLOCK_S - 1) / BLOCK_S;

  // ── OpenMP 并行：在 batch × head 维度上并行 ──
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (int64_t b = 0; b < p.B; ++b) {
    for (int64_t n = 0; n < p.N; ++n) {
      // ── 遍历每个 query 位置 ──
      for (int64_t l = 0; l < p.L; ++l) {
        // 当前 query 行指针：Q[b, n, l, :]
        const scalar_t* q_row = q_ptr + b * q_stride_b + n * q_stride_n + l * q_stride_l;

        // 输出行指针：output[b, n, l, :]
        float* o_row = p.out_ptr + b * o_stride_b + n * o_stride_n + l * o_stride_l;

        // ── Online Softmax 状态初始化 ──
        float running_max = p.neg_inf;  // 全局最大值
        float running_sum = 0.0f;       // 全局 exp 累加和

        // output 累加器：output_acc[Ev]，float32 累加
        // 使用 alignas 对齐以利于向量化
        alignas(64) float output_acc[4096];  // 足够大的静态数组
        for (int64_t ev = 0; ev < p.Ev; ++ev) {
          output_acc[ev] = 0.0f;
        }

        // 局部 scores 缓冲区：scores_buf[BLOCK_S]
        alignas(64) float scores_buf[BLOCK_S];

        // 因果掩码的可见边界：key 位置 s 可见当且仅当 s <= l + causal_offset
        const int64_t causal_limit = l + p.causal_offset;

        // ── 逐块遍历 K/V 序列维度 ──
        for (int64_t blk = 0; blk < num_blocks; ++blk) {
          const int64_t block_start = blk * BLOCK_S;
          const int64_t block_len = std::min(static_cast<int64_t>(BLOCK_S), p.S - block_start);

          // ── 因果掩码 early exit：整块不可见则跳出 ──
          if (p.is_causal && block_start > causal_limit) {
            break;  // 后续块也不可见（因果掩码单调性）
          }

          // ── 步骤 1: 计算当前块的 scores ──
          // scores_buf[j] = dot(Q_row, K[b,n,block_start+j,:]) * scale
          for (int64_t j = 0; j < block_len; ++j) {
            const scalar_t* k_row = k_ptr + b * k_stride_b + n * k_stride_n + (block_start + j) * k_stride_s;
            float dot = 0.0f;
            for (int64_t e = 0; e < p.E; ++e) {
              dot += static_cast<float>(q_row[e]) * static_cast<float>(k_row[e]);
            }
            scores_buf[j] = dot * p.scale_f;
          }

          // ── 步骤 2: 应用 Additive Attention Mask（先于因果掩码）──
          if (p.mask_ptr) {
            const float* m_row = p.mask_ptr + b * m_stride_b + n * m_stride_n + l * m_stride_l + block_start;
            for (int64_t j = 0; j < block_len; ++j) {
              scores_buf[j] += m_row[j];
            }
          }

          // ── 步骤 3: 应用因果掩码 ──
          if (p.is_causal) {
            // 判断整块是否完全可见
            if (block_start + block_len - 1 > causal_limit) {
              // 部分可见：逐元素检查
              for (int64_t j = 0; j < block_len; ++j) {
                if (block_start + j > causal_limit) {
                  scores_buf[j] = p.neg_inf;
                }
              }
            }
            // 若 block_start + block_len - 1 <= causal_limit，
            // 整块完全可见，无需任何掩码操作
          }

          // ── 步骤 4: Online Softmax 更新 ──
          // (a) 计算当前块的局部最大值
          float block_max = p.neg_inf;
          for (int64_t j = 0; j < block_len; ++j) {
            if (scores_buf[j] > block_max) {
              block_max = scores_buf[j];
            }
          }

          // (b) 计算新的全局最大值
          const float new_max = std::max(running_max, block_max);

          // (c) 计算修正因子：将之前累加的结果从旧 max 修正到新 max
          const float correction = std::exp(running_max - new_max);

          // (d) 更新 running_sum 和 output_acc
          //     先对已有累加值应用修正因子
          running_sum *= correction;
          for (int64_t ev = 0; ev < p.Ev; ++ev) {
            output_acc[ev] *= correction;
          }

          // (e) 计算当前块的 exp(score - new_max)，累加到 running_sum 和 output_acc
          for (int64_t j = 0; j < block_len; ++j) {
            const float exp_val = std::exp(scores_buf[j] - new_max);
            running_sum += exp_val;

            // 累加 exp_val * V[b, n, block_start+j, :]
            const scalar_t* v_row = v_ptr + b * v_stride_b + n * v_stride_n + (block_start + j) * v_stride_s;
            for (int64_t ev = 0; ev < p.Ev; ++ev) {
              output_acc[ev] += exp_val * static_cast<float>(v_row[ev]);
            }
          }

          // (f) 更新全局最大值
          running_max = new_max;
        }  // end block loop

        // ── 步骤 5: 最终归一化 ──
        // output[ev] = output_acc[ev] / running_sum
        if (running_sum > 0.0f) {
          const float inv_sum = 1.0f / running_sum;
          for (int64_t ev = 0; ev < p.Ev; ++ev) {
            o_row[ev] = output_acc[ev] * inv_sum;
          }
        } else {
          // 所有 scores 被掩码为 -inf，输出零向量
          for (int64_t ev = 0; ev < p.Ev; ++ev) {
            o_row[ev] = 0.0f;
          }
        }
      }  // end l loop
    }  // end n loop
  }  // end b loop
}

// ── FlashAttention-2 内核：统一 dtype-erased 入口 ───────────────────────
//
// 通过 SdpaParams::dtype 在内部 dispatch 到 fp32 / bf16 模板实例。
// 该函数被 REGISTER_SDPA_VERSION("flash2", ...) 注册到全局表，
// 同时被旧入口 scaled_dot_product_attention 直接调用。
void sdpa_flash2_impl(const SdpaParams& p) {
  // ── 安全检查：output_acc 栈上缓冲区上限 ──
  constexpr int64_t MAX_EV = 4096;
  TORCH_CHECK(p.Ev <= MAX_EV, "sdpa_flash2_impl: v_head_dim (", p.Ev, ") exceeds maximum supported value (", MAX_EV,
              ")");

  LegacySdpaParams lp{};
  lp.B = p.B;
  lp.N = p.N;
  lp.L = p.L;
  lp.S = p.S;
  lp.E = p.E;
  lp.Ev = p.Ev;
  lp.scale_f = p.scale_f;
  lp.neg_inf = p.neg_inf;
  lp.causal_offset = p.causal_offset;
  lp.is_causal = p.is_causal;
  lp.out_ptr = p.out_ptr;
  lp.mask_ptr = p.mask_ptr;

  if (p.dtype == SdpaDtype::kBFloat16) {
    sdpa_flash2_kernel_impl<at::BFloat16>(static_cast<const at::BFloat16*>(p.q_ptr),
                                          static_cast<const at::BFloat16*>(p.k_ptr),
                                          static_cast<const at::BFloat16*>(p.v_ptr), lp);
  } else {
    sdpa_flash2_kernel_impl<float>(static_cast<const float*>(p.q_ptr), static_cast<const float*>(p.k_ptr),
                                   static_cast<const float*>(p.v_ptr), lp);
  }
}

// ── 注册到全局版本表 ────────────────────────────────────────────────────
REGISTER_SDPA_VERSION("flash2", sdpa_flash2_impl);

// ── 旧版对外接口 ────────────────────────────────────────────────────────
//
// 接口与默认行为完全保留（向后兼容）。内部直接调用 sdpa_flash2_impl，
// 避免 sdpa_dispatch 的字符串查表开销。
at::Tensor scaled_dot_product_attention(at::Tensor query, at::Tensor key, at::Tensor value,
                                        c10::optional<at::Tensor> attn_mask, double dropout_p, bool is_causal,
                                        c10::optional<double> scale, bool enable_gqa) {
  torch::NoGradGuard no_grad;

  // ── 输入校验 ──────────────────────────────────────────────────────────
  TORCH_CHECK(query.dim() == 4,
              "scaled_dot_product_attention: query must be 4-D "
              "[B, N, L, E], got ",
              query.dim(), "-D");
  TORCH_CHECK(key.dim() == 4,
              "scaled_dot_product_attention: key must be 4-D "
              "[B, N, S, E], got ",
              key.dim(), "-D");
  TORCH_CHECK(value.dim() == 4,
              "scaled_dot_product_attention: value must be 4-D "
              "[B, N, S, Ev], got ",
              value.dim(), "-D");

  TORCH_CHECK(query.size(0) == key.size(0) && query.size(0) == value.size(0),
              "scaled_dot_product_attention: batch size mismatch among Q/K/V (", query.size(0), ", ", key.size(0), ", ",
              value.size(0), ")");
  TORCH_CHECK(query.size(1) == key.size(1) && query.size(1) == value.size(1),
              "scaled_dot_product_attention: num_heads mismatch among Q/K/V (", query.size(1), ", ", key.size(1), ", ",
              value.size(1), ")");
  TORCH_CHECK(key.size(2) == value.size(2), "scaled_dot_product_attention: K and V seq_len mismatch (", key.size(2),
              " vs ", value.size(2), ")");
  TORCH_CHECK(query.size(3) == key.size(3), "scaled_dot_product_attention: Q and K head_dim mismatch (", query.size(3),
              " vs ", key.size(3), ")");

  // ── 参数处理 ──────────────────────────────────────────────────────────
  TORCH_CHECK(!enable_gqa, "scaled_dot_product_attention: enable_gqa=true is not supported");

  if (dropout_p != 0.0) {
    TORCH_WARN("scaled_dot_product_attention: dropout_p=", dropout_p, " is ignored (inference only)");
  }

  double scale_val = scale.has_value() ? scale.value() : 1.0 / std::sqrt(static_cast<double>(query.size(-1)));

  // ── 记录原始 dtype ────────────────────────────────────────────────────
  auto orig_dtype = query.scalar_type();

  // ── 确保输入 contiguous ──────────────────────────────────────────────
  auto q = ensure_contiguous(query);
  auto k = ensure_contiguous(key);
  auto v = ensure_contiguous(value);

  // ── 维度信息 ─────────────────────────────────────────────────────────
  const int64_t B = q.size(0);   // batch_size
  const int64_t N = q.size(1);   // num_heads
  const int64_t L = q.size(2);   // query seq_len
  const int64_t S = k.size(2);   // key/value seq_len
  const int64_t E = q.size(3);   // qk_head_dim
  const int64_t Ev = v.size(3);  // v_head_dim（可能与 E 不同）

  const float scale_f = static_cast<float>(scale_val);
  const float neg_inf = -std::numeric_limits<float>::infinity();
  const int64_t causal_offset = S - L;  // 因果掩码偏移量

  // ── 安全检查：output_acc 栈上缓冲区上限 ─────────────────────────────
  constexpr int64_t MAX_EV = 4096;
  TORCH_CHECK(Ev <= MAX_EV, "scaled_dot_product_attention: v_head_dim (", Ev, ") exceeds maximum supported value (",
              MAX_EV, ")");

  // ── 处理 dtype；非 fp32/bf16 输入提升到 fp32 ──
  SdpaDtype kernel_dtype;
  if (orig_dtype == at::kBFloat16) {
    kernel_dtype = SdpaDtype::kBFloat16;
  } else if (orig_dtype == at::kFloat) {
    kernel_dtype = SdpaDtype::kFloat32;
  } else {
    q = q.to(at::kFloat);
    k = k.to(at::kFloat);
    v = v.to(at::kFloat);
    kernel_dtype = SdpaDtype::kFloat32;
  }

  // ── 仅分配输出张量（float32），不再分配 attn_scores 中间张量 ──────────
  auto output_fp32 = at::empty({B, N, L, Ev}, q.options().dtype(at::kFloat));

  // ── 处理可选的 additive attention mask ───────────────────────────────
  const float* mask_ptr = nullptr;
  at::Tensor mask_fp32;
  if (attn_mask.has_value()) {
    mask_fp32 = ensure_contiguous(attn_mask.value().to(at::kFloat));
    mask_ptr = mask_fp32.data_ptr<float>();
  }

  // ── 构建参数结构体并直调 flash2 内核 ───────────────────────────────
  SdpaParams params{};
  params.B = B;
  params.N = N;
  params.L = L;
  params.S = S;
  params.E = E;
  params.Ev = Ev;
  params.scale_f = scale_f;
  params.neg_inf = neg_inf;
  params.causal_offset = causal_offset;
  params.is_causal = is_causal;
  params.dtype = kernel_dtype;
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.mask_ptr = mask_ptr;
  params.out_ptr = output_fp32.data_ptr<float>();

  sdpa_flash2_impl(params);

  // ── 转换回原始 dtype ─────────────────────────────────────────────────
  if (orig_dtype != at::kFloat) {
    return output_fp32.to(orig_dtype);
  }
  return output_fp32;
}
