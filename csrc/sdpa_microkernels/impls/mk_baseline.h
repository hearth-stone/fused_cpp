#pragma once
// ── 微内核 impl：MK_Baseline ────────────────────────────────────────────
//
// 把 csrc/sdpa_microkernels/neon_cache_microkernels.h 中已有的全局自由
// 函数 `gemm_qkt_8x8 / gemm_qkt_8x4 / gemm_qkt_tail / gemm_pv_8x8 /
// gemm_pv_tail`（其内部按 dtype 自动选 NEON/BFMMLA/scalar 路径）包装成
// `MK_Baseline` trait struct。所有方法都是 `static inline`，会被编译器
// 在调用点内联，零开销。
//
// 编译期开关：FUSED_CPP_MK_ENABLE_BASELINE，默认 1。setup.py 不需要改。

#include <torch/extension.h>
#include <cstdint>

#include "../neon_cache_config.h"
#include "../neon_cache_microkernels.h"

#ifndef FUSED_CPP_MK_ENABLE_BASELINE
#define FUSED_CPP_MK_ENABLE_BASELINE 1
#endif

namespace fused_cpp::sdpa_microkernels {

#if FUSED_CPP_MK_ENABLE_BASELINE
struct MK_Baseline {
  static constexpr const char* kName = "baseline";
  static constexpr bool kEnabled = true;

  // —— QKᵀ 主体 8×8 ——
  static inline void qkt_8x8(const at::BFloat16* Q, int64_t q_row_stride, const at::BFloat16* K, int64_t k_row_stride,
                             int64_t E, float scale, float* scores_buf) {
    gemm_qkt_8x8(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf);
  }
  static inline void qkt_8x8(const float* Q, int64_t q_row_stride, const float* K, int64_t k_row_stride, int64_t E,
                             float scale, float* scores_buf) {
    gemm_qkt_8x8(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf);
  }

  // —— QKᵀ 退化 8×4 ——
  static inline void qkt_8x4(const at::BFloat16* Q, int64_t q_row_stride, const at::BFloat16* K, int64_t k_row_stride,
                             int64_t E, float scale, float* scores_buf, int64_t scores_row_stride) {
    gemm_qkt_8x4(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf, scores_row_stride);
  }
  static inline void qkt_8x4(const float* Q, int64_t q_row_stride, const float* K, int64_t k_row_stride, int64_t E,
                             float scale, float* scores_buf, int64_t scores_row_stride) {
    gemm_qkt_8x4(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf, scores_row_stride);
  }

  // —— QKᵀ 任意尾部 ——
  static inline void qkt_tail(const at::BFloat16* Q, int64_t q_row_stride, const at::BFloat16* K, int64_t k_row_stride,
                              int64_t E, float scale, float* scores_buf, int64_t scores_row_stride, int Lq, int Sk) {
    gemm_qkt_tail(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf, scores_row_stride, Lq, Sk);
  }
  static inline void qkt_tail(const float* Q, int64_t q_row_stride, const float* K, int64_t k_row_stride, int64_t E,
                              float scale, float* scores_buf, int64_t scores_row_stride, int Lq, int Sk) {
    gemm_qkt_tail(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf, scores_row_stride, Lq, Sk);
  }

  // —— P̂·V 主体 8×8 ——
  static inline void pv_8x8(const float* P_hat, int64_t P_row_stride, const at::BFloat16* V, int64_t v_row_stride,
                            int64_t Sk, float* O, int64_t o_row_stride) {
    gemm_pv_8x8(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
  }
  static inline void pv_8x8(const float* P_hat, int64_t P_row_stride, const float* V, int64_t v_row_stride, int64_t Sk,
                            float* O, int64_t o_row_stride) {
    gemm_pv_8x8(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
  }

  // —— P̂·V 任意尾部 ——
  static inline void pv_tail(const float* P_hat, int64_t P_row_stride, const at::BFloat16* V, int64_t v_row_stride,
                             int64_t Sk, float* O, int64_t o_row_stride, int Lq, int Ev) {
    gemm_pv_tail(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride, Lq, Ev);
  }
  static inline void pv_tail(const float* P_hat, int64_t P_row_stride, const float* V, int64_t v_row_stride, int64_t Sk,
                             float* O, int64_t o_row_stride, int Lq, int Ev) {
    gemm_pv_tail(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride, Lq, Ev);
  }
};
#endif  // FUSED_CPP_MK_ENABLE_BASELINE

}  // namespace fused_cpp::sdpa_microkernels
