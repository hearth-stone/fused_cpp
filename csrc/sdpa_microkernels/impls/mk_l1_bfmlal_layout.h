#pragma once
// L1-only evaluation impl:
//   * bf16 QKT uses Q row-major + K transposed to K_col[E][8], then BFMLAL.
//   * bf16 PV exposes pv_8x8_pbf16 for P already materialized as bf16.
//
// This impl is intentionally not an SDPA outer-path contract. It measures the
// microkernel ceiling when the producer/packer is allowed to choose a layout
// that keeps the 8x8 working set in L1.

#include <torch/extension.h>
#include <cstdint>
#include <vector>

#include "../neon_cache_config.h"
#include "../neon_cache_microkernels.h"
#include "../../sdpa_profile.h"

#ifndef FUSED_CPP_MK_ENABLE_L1_BFMLAL_LAYOUT
#define FUSED_CPP_MK_ENABLE_L1_BFMLAL_LAYOUT 1
#endif

namespace fused_cpp::sdpa_microkernels {

#if FUSED_CPP_MK_ENABLE_L1_BFMLAL_LAYOUT
struct MK_L1BfmlalLayout {
  static constexpr const char* kName = "l1_bfmlal_layout";
  static constexpr bool kEnabled = true;
  static constexpr bool kHasPvPbf16 = true;
  static constexpr bool kHasQktKcol = true;

  static inline void qkt_8x8(const at::BFloat16* Q, int64_t q_row_stride, const at::BFloat16* K, int64_t k_row_stride,
                             int64_t E, float scale, float* scores_buf) {
#if FUSED_CPP_SDPA_CACHE_HAS_BF16
    static thread_local std::vector<at::BFloat16> k_col_buf;
    static thread_local const at::BFloat16* last_K = nullptr;
    static thread_local int64_t last_k_row_stride = 0;
    static thread_local int64_t last_E = 0;

    if (K != last_K || k_row_stride != last_k_row_stride || E != last_E) {
      FUSED_CPP_SDPA_PROFILE_DEEP_SCOPE(::fused_cpp::sdpa_profile::Slot::kKColPack);
      k_col_buf.resize(static_cast<size_t>(8 * E));
      pack_k_8rows_to_col_bf16(K, k_row_stride, E, k_col_buf.data());
      last_K = K;
      last_k_row_stride = k_row_stride;
      last_E = E;
    }

    {
      FUSED_CPP_SDPA_PROFILE_DEEP_SCOPE(::fused_cpp::sdpa_profile::Slot::kQktMicro);
      gemm_qkt_microkernel_8x8_bf16_qrow_kcol_bfmlal(Q, q_row_stride, k_col_buf.data(), E, scale, scores_buf);
    }
#else
    gemm_qkt_8x8(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf);
#endif
  }

  // —— QKᵀ 主体 8×8 bf16：K_col[E][8] 已由 benchmark / caller 提供 ——
  static inline void qkt_8x8_kcol(const at::BFloat16* Q, int64_t q_row_stride, const at::BFloat16* K_col, int64_t E,
                                  float scale, float* scores_buf) {
    gemm_qkt_microkernel_8x8_bf16_qrow_kcol_bfmlal(Q, q_row_stride, K_col, E, scale, scores_buf);
  }

  static inline void qkt_8x8(const float* Q, int64_t q_row_stride, const float* K, int64_t k_row_stride, int64_t E,
                             float scale, float* scores_buf) {
    gemm_qkt_8x8(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf);
  }

  static inline void qkt_8x4(const at::BFloat16* Q, int64_t q_row_stride, const at::BFloat16* K, int64_t k_row_stride,
                             int64_t E, float scale, float* scores_buf, int64_t scores_row_stride) {
    gemm_qkt_8x4(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf, scores_row_stride);
  }
  static inline void qkt_8x4(const float* Q, int64_t q_row_stride, const float* K, int64_t k_row_stride, int64_t E,
                             float scale, float* scores_buf, int64_t scores_row_stride) {
    gemm_qkt_8x4(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf, scores_row_stride);
  }

  static inline void qkt_tail(const at::BFloat16* Q, int64_t q_row_stride, const at::BFloat16* K, int64_t k_row_stride,
                              int64_t E, float scale, float* scores_buf, int64_t scores_row_stride, int Lq, int Sk) {
    gemm_qkt_tail(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf, scores_row_stride, Lq, Sk);
  }
  static inline void qkt_tail(const float* Q, int64_t q_row_stride, const float* K, int64_t k_row_stride, int64_t E,
                              float scale, float* scores_buf, int64_t scores_row_stride, int Lq, int Sk) {
    gemm_qkt_tail(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf, scores_row_stride, Lq, Sk);
  }

  static inline void pv_8x8(const float* P_hat, int64_t P_row_stride, const at::BFloat16* V, int64_t v_row_stride,
                            int64_t Sk, float* O, int64_t o_row_stride) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
    gemm_pv_microkernel_8x8_bf16_pquad(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
#else
    gemm_pv_8x8(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
#endif
  }
  static inline void pv_8x8(const float* P_hat, int64_t P_row_stride, const float* V, int64_t v_row_stride, int64_t Sk,
                            float* O, int64_t o_row_stride) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
    gemm_pv_microkernel_8x8_fp32_pquad(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
#else
    gemm_pv_8x8(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
#endif
  }

  static inline void pv_8x8_pbf16(const at::BFloat16* P_bf16, int64_t P_row_stride, const at::BFloat16* V,
                                  int64_t v_row_stride, int64_t Sk, float* O, int64_t o_row_stride) {
    gemm_pv_microkernel_8x8_bf16_pbf16_prepacked(P_bf16, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
  }

  static inline void pv_tail(const float* P_hat, int64_t P_row_stride, const at::BFloat16* V, int64_t v_row_stride,
                             int64_t Sk, float* O, int64_t o_row_stride, int Lq, int Ev) {
    gemm_pv_tail(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride, Lq, Ev);
  }
  static inline void pv_tail(const float* P_hat, int64_t P_row_stride, const float* V, int64_t v_row_stride, int64_t Sk,
                             float* O, int64_t o_row_stride, int Lq, int Ev) {
    gemm_pv_tail(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride, Lq, Ev);
  }
};
#endif  // FUSED_CPP_MK_ENABLE_L1_BFMLAL_LAYOUT

}  // namespace fused_cpp::sdpa_microkernels
