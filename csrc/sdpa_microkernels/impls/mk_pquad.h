#pragma once
// ── 微内核 impl：MK_PQuad ─────────────────────────────────────────────────
//
// PV fp32 主体的「P 折成 quad load」改进版。所有 op 默认沿用 MK_Baseline
// 的 NEON 路径（即 neon_cache_microkernels.h 里 gemm_qkt_8x8 / gemm_qkt_8x4
// / gemm_qkt_tail / gemm_pv_8x8(bf16) / gemm_pv_tail），唯一的差别在于
// **fp32 版本的 pv_8x8 改派到 gemm_pv_microkernel_8x8_fp32_pquad**：
//
//   * fp32 主体（baseline）：每 4-k 段 32 个 P 标量 load + 8 个 V quad load
//                            = 40 LSU op；FMA pipe 占主导。
//   * pquad（本 impl）     ：每 4-k 段 8 个 P quad load + 8 个 V quad load
//                            = 16 LSU op（2.5×↓）；FMA pipe 占用与主体相同。
//
// 寄存器账本与主体相同（28 NEON reg 稳态），数值上对 fp32 严格按位等价。
//
// bf16 路径完全不动（仍指向 baseline 的 BFMMLA / 标量兜底），因为本改进
// 只针对 fp32 的 P 来源。
//
// 编译期开关：FUSED_CPP_MK_ENABLE_PQUAD，默认 1。

#include <torch/extension.h>
#include <cstdint>

#include "../neon_cache_config.h"
#include "../neon_cache_microkernels.h"

#ifndef FUSED_CPP_MK_ENABLE_PQUAD
#define FUSED_CPP_MK_ENABLE_PQUAD 1
#endif

namespace fused_cpp::sdpa_microkernels {

#if FUSED_CPP_MK_ENABLE_PQUAD
struct MK_PQuad {
  static constexpr const char* kName = "pquad";
  static constexpr bool kEnabled = true;

  // —— QKᵀ 主体 8×8（与 baseline 同） ——
  static inline void qkt_8x8(
      const at::BFloat16* Q, int64_t q_row_stride,
      const at::BFloat16* K, int64_t k_row_stride,
      int64_t E, float scale, float* scores_buf) {
    gemm_qkt_8x8(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf);
  }
  static inline void qkt_8x8(
      const float* Q, int64_t q_row_stride,
      const float* K, int64_t k_row_stride,
      int64_t E, float scale, float* scores_buf) {
    gemm_qkt_8x8(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf);
  }

  // —— QKᵀ 退化 8×4（与 baseline 同） ——
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

  // —— QKᵀ 任意尾部（与 baseline 同） ——
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

  // —— P̂·V 主体 8×8 ——
  // bf16：仍走 baseline 的 widen + vfmaq_n（V 是 bf16，与本改进无关）。
  static inline void pv_8x8(
      const float* P_hat, int64_t P_row_stride,
      const at::BFloat16* V, int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride) {
    gemm_pv_8x8(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
  }
  // fp32：派到新的 _pquad 微内核。
  static inline void pv_8x8(
      const float* P_hat, int64_t P_row_stride,
      const float* V, int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
    gemm_pv_microkernel_8x8_fp32_pquad(
        P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
#else
    // 没 NEON 时退到标量兜底（与 gemm_pv_8x8(float) 在无 NEON 分支一致）。
    gemm_pv_8x8(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
#endif
  }

  // —— P̂·V 任意尾部（与 baseline 同） ——
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
#endif  // FUSED_CPP_MK_ENABLE_PQUAD

}  // namespace fused_cpp::sdpa_microkernels
