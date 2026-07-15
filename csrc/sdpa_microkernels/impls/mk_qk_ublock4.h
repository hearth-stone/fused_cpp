#pragma once
// ── 微内核 impl：MK_QkUblock4 ───────────────────────────────────────────
//
// QKᵀ fp32 主体的「4×4 双向分块 + 16 累加器外积扇出」改进版。所有 op
// 默认沿用 MK_Baseline 的 NEON 路径（即 neon_cache_microkernels.h 里
// gemm_qkt_8x4 / gemm_qkt_tail / gemm_pv_8x8 / gemm_pv_tail，以及 bf16
// 版本的 gemm_qkt_8x8），唯一的差别在于 **fp32 版本的 qkt_8x8 改派到
// gemm_qkt_microkernel_8x8_fp32_ublock4**：
//
//   * fp32 baseline：64 个独立 dot product（8×8 个 (i,j) 各跑一遍
//                    `for e: vfmaq_f32; vaddvq_f32`），单累加器 RAW
//                    依赖链 = E/4，OoO 跨 (i,j) iteration 渲染只能
//                    并发 ~3 条 fma 链；实测 ~10 GFLOPS（~9% 109 fmla
//                    peak）。
//   * ublock4（本 impl）：8×8 输出切成 4 个 4×4 子块，每个子块 16 个
//                          独立 float32x4_t 累加器同时沿 e 维累加，
//                          ILP=16 条独立 fma 链；vpaddq tree-reduce
//                          收尾。预期 95–105 GFLOPS（~10× 提升）。
//
// 寄存器账本：16 acc + 4 Q + 4 K + 临时 ≈ 25/32 NEON reg 稳态。数值上
// 与 baseline **接近等价但不按位相同**——fp32 下 vfmaq + vpaddq tree
// reduce 的累加顺序与原 vfmaq + vaddvq 不同，存在 ULP-级误差，已被
// SDPA 等价性测试容忍度覆盖（atol=rtol=1e-4 fp32）。
//
// bf16 路径完全不动（仍指向 baseline 的 BFMMLA / 标量兜底），因为本
// 改进只针对 fp32 的 QKᵀ。
//
// 编译期开关：FUSED_CPP_MK_ENABLE_QK_UBLOCK4，默认 1。

#include <torch/extension.h>
#include <cstdint>

#include "../neon_cache_config.h"
#include "../neon_cache_microkernels.h"

#ifndef FUSED_CPP_MK_ENABLE_QK_UBLOCK4
#define FUSED_CPP_MK_ENABLE_QK_UBLOCK4 1
#endif

namespace fused_cpp::sdpa_microkernels {

#if FUSED_CPP_MK_ENABLE_QK_UBLOCK4
struct MK_QkUblock4 {
  static constexpr const char* kName = "qk_ublock4";
  static constexpr bool kEnabled = true;

  // —— QKᵀ 主体 8×8 ——
  // bf16：仍走 baseline 的 BFMMLA 主路径（QKᵀ-bf16 切 BFMLALB/T 是
  // 另一个 P0 项，在不同的 trait 里实现）。
  static inline void qkt_8x8(const at::BFloat16* Q, int64_t q_row_stride, const at::BFloat16* K, int64_t k_row_stride,
                             int64_t E, float scale, float* scores_buf) {
    gemm_qkt_8x8(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf);
  }
  // fp32：派到新的 _ublock4 微内核。
  static inline void qkt_8x8(const float* Q, int64_t q_row_stride, const float* K, int64_t k_row_stride, int64_t E,
                             float scale, float* scores_buf) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
    gemm_qkt_microkernel_8x8_fp32_ublock4(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf);
#else
    // 没 NEON 时退到标量兜底（与 gemm_qkt_8x8(float) 在无 NEON 分支一致）。
    gemm_qkt_8x8(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf);
#endif
  }

  // —— QKᵀ 退化 8×4（与 baseline 同） ——
  static inline void qkt_8x4(const at::BFloat16* Q, int64_t q_row_stride, const at::BFloat16* K, int64_t k_row_stride,
                             int64_t E, float scale, float* scores_buf, int64_t scores_row_stride) {
    gemm_qkt_8x4(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf, scores_row_stride);
  }
  static inline void qkt_8x4(const float* Q, int64_t q_row_stride, const float* K, int64_t k_row_stride, int64_t E,
                             float scale, float* scores_buf, int64_t scores_row_stride) {
    gemm_qkt_8x4(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf, scores_row_stride);
  }

  // —— QKᵀ 任意尾部（与 baseline 同） ——
  static inline void qkt_tail(const at::BFloat16* Q, int64_t q_row_stride, const at::BFloat16* K, int64_t k_row_stride,
                              int64_t E, float scale, float* scores_buf, int64_t scores_row_stride, int Lq, int Sk) {
    gemm_qkt_tail(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf, scores_row_stride, Lq, Sk);
  }
  static inline void qkt_tail(const float* Q, int64_t q_row_stride, const float* K, int64_t k_row_stride, int64_t E,
                              float scale, float* scores_buf, int64_t scores_row_stride, int Lq, int Sk) {
    gemm_qkt_tail(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf, scores_row_stride, Lq, Sk);
  }

  // —— P̂·V 主体 8×8（与 baseline 同） ——
  static inline void pv_8x8(const float* P_hat, int64_t P_row_stride, const at::BFloat16* V, int64_t v_row_stride,
                            int64_t Sk, float* O, int64_t o_row_stride) {
    gemm_pv_8x8(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
  }
  static inline void pv_8x8(const float* P_hat, int64_t P_row_stride, const float* V, int64_t v_row_stride, int64_t Sk,
                            float* O, int64_t o_row_stride) {
    gemm_pv_8x8(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
  }

  // —— P̂·V 任意尾部（与 baseline 同） ——
  static inline void pv_tail(const float* P_hat, int64_t P_row_stride, const at::BFloat16* V, int64_t v_row_stride,
                             int64_t Sk, float* O, int64_t o_row_stride, int Lq, int Ev) {
    gemm_pv_tail(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride, Lq, Ev);
  }
  static inline void pv_tail(const float* P_hat, int64_t P_row_stride, const float* V, int64_t v_row_stride, int64_t Sk,
                             float* O, int64_t o_row_stride, int Lq, int Ev) {
    gemm_pv_tail(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride, Lq, Ev);
  }
};
#endif  // FUSED_CPP_MK_ENABLE_QK_UBLOCK4

}  // namespace fused_cpp::sdpa_microkernels
