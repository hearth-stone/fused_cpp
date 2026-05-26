#pragma once
// ── 微内核 impl：MK_QkPackkFull ───────────────────────────────────────────
//
// **评估用 trait，不接入 SDPA 主路径**——用于在 microkernel benchmark 框架
// 下测「pack K + packed inner microkernel」的总开销 GFLOPS（pessimistic
// 下界）。每次 trait::qkt_8x8 调用都做一次 pack（B = std::vector + memcpy）
// 然后跑 packed inner，pack overhead 完整计入 timing。
//
// 与 MK_QkPackkInner（thread_local cache 跳过重复 pack）成对使用：
//   * full：每次都 pack——下界，反映「pack 一次只跑一次 8×8」的最坏情况
//   * inner：第一次 pack 后 N-1 次跳过——上界，反映 SDPA 实际中 K 在外层
//             pack 一次给 L/8 个 8-row tile 共用的 amortize 后行为
//
// 只重写 bf16 qkt_8x8（评估目标）；其余 9 个 dtype×op 组合全部 fall through
// 到 baseline。
//
// 编译期开关：FUSED_CPP_MK_ENABLE_QK_PACKK_FULL，默认 1。

#include <torch/extension.h>
#include <cstdint>
#include <vector>

#include "../neon_cache_config.h"
#include "../neon_cache_microkernels.h"

#ifndef FUSED_CPP_MK_ENABLE_QK_PACKK_FULL
#define FUSED_CPP_MK_ENABLE_QK_PACKK_FULL 1
#endif

namespace fused_cpp::sdpa_microkernels {

#if FUSED_CPP_MK_ENABLE_QK_PACKK_FULL
struct MK_QkPackkFull {
  static constexpr const char* kName = "qk_packk_full";
  static constexpr bool kEnabled = true;

  // —— QKᵀ 主体 8×8 ——
  // bf16：在内部 pack K 后调 packed inner microkernel。每次都 pack。
  static inline void qkt_8x8(
      const at::BFloat16* Q, int64_t q_row_stride,
      const at::BFloat16* K, int64_t k_row_stride,
      int64_t E, float scale, float* scores_buf) {
#if FUSED_CPP_SDPA_CACHE_HAS_BFMMLA
    // pack K → 栈外 buffer。std::vector 的 alloc 在第一次调用后会被 malloc
    // 分配器复用（同一线程同样大小）；这是 microkernel 收益评估的 acceptable
    // 噪声源。生产路径要替换成 SDPA 入口处一次性分配的全局 buffer。
    std::vector<uint16_t> K_packed(static_cast<size_t>(8 * E));
    pack_k_8rows_to_pairs_bf16(K, k_row_stride, E, K_packed.data());
    gemm_qkt_microkernel_8x8_bf16_packk_inner(
        Q, q_row_stride, K_packed.data(), K, k_row_stride, E, scale, scores_buf);
#else
    gemm_qkt_8x8(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf);
#endif
  }
  // fp32：fall through 到 baseline。
  static inline void qkt_8x8(
      const float* Q, int64_t q_row_stride,
      const float* K, int64_t k_row_stride,
      int64_t E, float scale, float* scores_buf) {
    gemm_qkt_8x8(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf);
  }

  // —— 其他 op 全部 fall through ——
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
#endif  // FUSED_CPP_MK_ENABLE_QK_PACKK_FULL

}  // namespace fused_cpp::sdpa_microkernels
