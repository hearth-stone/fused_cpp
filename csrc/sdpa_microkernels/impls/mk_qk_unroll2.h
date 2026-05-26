#pragma once
// ── 微内核 impl：MK_QkUnroll2 ────────────────────────────────────────────
//
// **评估用 trait，不接入 SDPA 主路径**——测「BFMMLA 主路径 + 2-way k-unroll」
// 相对 baseline 的收益。验证假设：baseline 内层 e+=4 一次 16 BFMMLA，没有
// k-unroll；OoO issue window 可能被 loop overhead 限制（每 iter 1-2 cycle
// cmp+branch / 32 cycle BFMMLA-bound iter ≈ 3-6%）。
//
// 与 baseline BFMMLA 主路径**算法等价**（按位相同的 fp32 输出）；唯一差别
// 是内层 e += 8 一次跑 32 BFMMLA，跨 e-block 累加器复用。
//
// 寄存器账本：16 BFMMLA acc + 段0 4 a-reg + 段0 4 b-reg + 段1 4 a-reg
// + 段1 4 b-reg = **32 NEON reg 正好打满**。如果编译器决定不复用段间
// 寄存器，可能轻微 spill；实测验证。
//
// 只重写 bf16 qkt_8x8；其余 op fall through 到 baseline。
//
// 编译期开关：FUSED_CPP_MK_ENABLE_QK_UNROLL2，默认 1。

#include <torch/extension.h>
#include <cstdint>

#include "../neon_cache_config.h"
#include "../neon_cache_microkernels.h"

#ifndef FUSED_CPP_MK_ENABLE_QK_UNROLL2
#define FUSED_CPP_MK_ENABLE_QK_UNROLL2 1
#endif

namespace fused_cpp::sdpa_microkernels {

#if FUSED_CPP_MK_ENABLE_QK_UNROLL2
struct MK_QkUnroll2 {
  static constexpr const char* kName = "qk_unroll2";
  static constexpr bool kEnabled = true;

  static inline void qkt_8x8(
      const at::BFloat16* Q, int64_t q_row_stride,
      const at::BFloat16* K, int64_t k_row_stride,
      int64_t E, float scale, float* scores_buf) {
#if FUSED_CPP_SDPA_CACHE_HAS_BFMMLA
    gemm_qkt_microkernel_8x8_bf16_unroll2(
        Q, q_row_stride, K, k_row_stride, E, scale, scores_buf);
#else
    gemm_qkt_8x8(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf);
#endif
  }
  static inline void qkt_8x8(
      const float* Q, int64_t q_row_stride,
      const float* K, int64_t k_row_stride,
      int64_t E, float scale, float* scores_buf) {
    gemm_qkt_8x8(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf);
  }

  // 其他 op 全部 fall through。
  static inline void qkt_8x4(
      const at::BFloat16* Q, int64_t q_row_stride,
      const at::BFloat16* K, int64_t k_row_stride,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride) {
    gemm_qkt_8x4(Q, q_row_stride, K, k_row_stride, E, scale,
                 scores_buf, scores_row_stride);
  }
  static inline void qkt_8x4(
      const float* Q, int64_t q_row_stride,
      const float* K, int64_t k_row_stride,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride) {
    gemm_qkt_8x4(Q, q_row_stride, K, k_row_stride, E, scale,
                 scores_buf, scores_row_stride);
  }
  static inline void qkt_tail(
      const at::BFloat16* Q, int64_t q_row_stride,
      const at::BFloat16* K, int64_t k_row_stride,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride,
      int Lq, int Sk) {
    gemm_qkt_tail(Q, q_row_stride, K, k_row_stride, E, scale,
                  scores_buf, scores_row_stride, Lq, Sk);
  }
  static inline void qkt_tail(
      const float* Q, int64_t q_row_stride,
      const float* K, int64_t k_row_stride,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride,
      int Lq, int Sk) {
    gemm_qkt_tail(Q, q_row_stride, K, k_row_stride, E, scale,
                  scores_buf, scores_row_stride, Lq, Sk);
  }
  static inline void pv_8x8(
      const float* P_hat, int64_t P_row_stride,
      const at::BFloat16* V, int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride) {
    gemm_pv_8x8(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
  }
  static inline void pv_8x8(
      const float* P_hat, int64_t P_row_stride,
      const float* V, int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride) {
    gemm_pv_8x8(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
  }
  static inline void pv_tail(
      const float* P_hat, int64_t P_row_stride,
      const at::BFloat16* V, int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride,
      int Lq, int Ev) {
    gemm_pv_tail(P_hat, P_row_stride, V, v_row_stride, Sk,
                 O, o_row_stride, Lq, Ev);
  }
  static inline void pv_tail(
      const float* P_hat, int64_t P_row_stride,
      const float* V, int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride,
      int Lq, int Ev) {
    gemm_pv_tail(P_hat, P_row_stride, V, v_row_stride, Sk,
                 O, o_row_stride, Lq, Ev);
  }
};
#endif  // FUSED_CPP_MK_ENABLE_QK_UNROLL2

}  // namespace fused_cpp::sdpa_microkernels
