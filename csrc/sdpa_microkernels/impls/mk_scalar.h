#pragma once
// ── 微内核 impl：MK_Scalar ───────────────────────────────────────────────
//
// 纯标量参考实现：5 个 op 全部走三层 for 循环 + fp32 累加。可在任何平台
// 编译运行，主要用作：
//   * 跨平台等价性参考（NEON 平台上验证 MK_Baseline 的输出一致）；
//   * 性能下界 baseline（GFLOPS 远低于 MK_Baseline，便于回归）；
//   * 编译没有 NEON / BFMMLA 的最小验证目标。
//
// 直接复用 neon_cache_microkernels.h 中已有的 `gemm_qkt_tail_scalar` /
// `gemm_pv_tail_scalar` 模板函数；它们对任意 Lq/Sk/Ev 都是满秩实现，
// 用 (Lq=8, Sk=8) / (Lq=8, Sk=4) 调用就成了 8x8 / 8x4 的纯标量等价。
//
// 编译期开关：FUSED_CPP_MK_ENABLE_SCALAR，默认 1。

#include <torch/extension.h>
#include <cstdint>

#include "../neon_cache_microkernels.h"

#ifndef FUSED_CPP_MK_ENABLE_SCALAR
#define FUSED_CPP_MK_ENABLE_SCALAR 1
#endif

namespace fused_cpp::sdpa_microkernels {

#if FUSED_CPP_MK_ENABLE_SCALAR
struct MK_Scalar {
  static constexpr const char* kName = "scalar";
  static constexpr bool kEnabled = true;

  // —— QKᵀ 8×8 / 8×4 / 任意尾部都走同一个 gemm_qkt_tail_scalar 模板 ——
  // scores_row_stride 在 8×8 主体里固定为 8（与 baseline 接口一致）。
  static inline void qkt_8x8(
      const at::BFloat16* Q, int64_t q_row_stride,
      const at::BFloat16* K, int64_t k_row_stride,
      int64_t E, float scale, float* scores_buf) {
    gemm_qkt_tail_scalar<at::BFloat16>(
        Q, q_row_stride, K, k_row_stride, E, scale,
        scores_buf, /*scores_row_stride=*/8, 8, 8);
  }
  static inline void qkt_8x8(
      const float* Q, int64_t q_row_stride,
      const float* K, int64_t k_row_stride,
      int64_t E, float scale, float* scores_buf) {
    gemm_qkt_tail_scalar<float>(
        Q, q_row_stride, K, k_row_stride, E, scale,
        scores_buf, /*scores_row_stride=*/8, 8, 8);
  }

  static inline void qkt_8x4(
      const at::BFloat16* Q, int64_t q_row_stride,
      const at::BFloat16* K, int64_t k_row_stride,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride) {
    gemm_qkt_tail_scalar<at::BFloat16>(
        Q, q_row_stride, K, k_row_stride, E, scale,
        scores_buf, scores_row_stride, 8, 4);
  }
  static inline void qkt_8x4(
      const float* Q, int64_t q_row_stride,
      const float* K, int64_t k_row_stride,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride) {
    gemm_qkt_tail_scalar<float>(
        Q, q_row_stride, K, k_row_stride, E, scale,
        scores_buf, scores_row_stride, 8, 4);
  }

  static inline void qkt_tail(
      const at::BFloat16* Q, int64_t q_row_stride,
      const at::BFloat16* K, int64_t k_row_stride,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride,
      int Lq, int Sk) {
    gemm_qkt_tail_scalar<at::BFloat16>(
        Q, q_row_stride, K, k_row_stride, E, scale,
        scores_buf, scores_row_stride, Lq, Sk);
  }
  static inline void qkt_tail(
      const float* Q, int64_t q_row_stride,
      const float* K, int64_t k_row_stride,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride,
      int Lq, int Sk) {
    gemm_qkt_tail_scalar<float>(
        Q, q_row_stride, K, k_row_stride, E, scale,
        scores_buf, scores_row_stride, Lq, Sk);
  }

  // —— P̂·V 全部走 gemm_pv_tail_scalar；O 是 += 累加而非 = 重置 ——
  static inline void pv_8x8(
      const float* P_hat, int64_t P_row_stride,
      const at::BFloat16* V, int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride) {
    gemm_pv_tail_scalar<at::BFloat16>(
        P_hat, P_row_stride, V, v_row_stride, Sk,
        O, o_row_stride, 8, 8);
  }
  static inline void pv_8x8(
      const float* P_hat, int64_t P_row_stride,
      const float* V, int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride) {
    gemm_pv_tail_scalar<float>(
        P_hat, P_row_stride, V, v_row_stride, Sk,
        O, o_row_stride, 8, 8);
  }

  static inline void pv_tail(
      const float* P_hat, int64_t P_row_stride,
      const at::BFloat16* V, int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride,
      int Lq, int Ev) {
    gemm_pv_tail_scalar<at::BFloat16>(
        P_hat, P_row_stride, V, v_row_stride, Sk,
        O, o_row_stride, Lq, Ev);
  }
  static inline void pv_tail(
      const float* P_hat, int64_t P_row_stride,
      const float* V, int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride,
      int Lq, int Ev) {
    gemm_pv_tail_scalar<float>(
        P_hat, P_row_stride, V, v_row_stride, Sk,
        O, o_row_stride, Lq, Ev);
  }
};
#endif  // FUSED_CPP_MK_ENABLE_SCALAR

}  // namespace fused_cpp::sdpa_microkernels
