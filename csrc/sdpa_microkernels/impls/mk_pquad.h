#pragma once
// ── 微内核 impl：MK_PQuad ─────────────────────────────────────────────────
//
// PV「P 折成 quad load」改进版，覆盖 **fp32 和 bf16 两条 PV 路径**。
// QKᵀ 全部沿用 MK_Baseline（本 impl 只改 PV）。
//
//   * fp32 baseline：每 4-k 段 32 个 P 标量 load + 8 个 V quad load = 40 LSU。
//   * fp32 pquad   ：每 4-k 段 8 个 P quad load + 8 个 V quad load = 16 LSU。
//   * bf16 baseline：每 4-k 段 32 个 P 标量 load + 4 个 V vld1q_u16 = 36 LSU。
//   * bf16 pquad   ：每 4-k 段 8 个 P quad load + 4 个 V vld1q_bf16，
//                    P 临时 round 到 bf16 后走 BFMLALB/T lane。
//
// bf16 pquad 的 BFMLAL 快路径避开 V widen；无 BF16 arithmetic 时保留
// 旧 widen+FMA pquad fallback。fp32 仍采用 pquad 软件流水（跨迭代 V
// 预取放段 2 中段）。
//
// 寄存器账本与主体相同（~30 NEON reg 稳态），数值上：
//   * fp32：按位等价（累加顺序逐段一致）
//   * bf16：P 临时 fp32→bf16 round，非按位等价；microkernel max_abs
//           约 1.6e-6，SDPA 目标小形状 max_abs 0.0078125
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
  // bf16：派到 _bf16_pquad；BF16 arithmetic 目标上内部走 P->bf16
  //       BFMLALB/T lane 快路径，fallback 是旧的 widen+FMA pquad。
  static inline void pv_8x8(
      const float* P_hat, int64_t P_row_stride,
      const at::BFloat16* V, int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
    gemm_pv_microkernel_8x8_bf16_pquad(
        P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
#else
    gemm_pv_8x8(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
#endif
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
