#pragma once
// ── 微内核 impl：MK_QkPackkInner ──────────────────────────────────────────
//
// **评估用 trait，不接入 SDPA 主路径**——配套 MK_QkPackkFull，用 thread_local
// cache 跳过重复 pack：第一次 qkt_8x8 调用 pack K 到 thread-local buffer，
// 之后调用如果 (K 指针 + k_row_stride + E) 全没变就直接复用，跳过 pack。
//
// microkernel benchmark 框架的 inner loop 始终用同一个 K vector（地址、
// stride、E 都不变），所以前 N-1 次都跳过 pack，timing 反映的是「纯 packed
// inner microkernel」的速度——是 K-pack 在 SDPA 真实场景中（K 在外层
// pack 一次给 L/8 个 i-tile 共用）的 amortize 上界。
//
// 限制：
//   * 仅 single-thread benchmark 安全。多线程下每线程独立 cache。
//   * 不同 K 张量但同地址（极端 corner case，e.g. arena reuse）会 stale。
//     SDPA 集成时**不要**直接复用此 trait——这是评估原型，生产路径要在
//     SDPA 入口显式 pack。
//   * thread_local std::vector 的 ctor/dtor 在 .so unload 时由 CRT 处理，
//     行为良好但不保证零开销，无伤大雅。
//
// 只重写 bf16 qkt_8x8；其余 9 个 dtype×op 组合 fall through 到 baseline。
//
// 编译期开关：FUSED_CPP_MK_ENABLE_QK_PACKK_INNER，默认 1。

#include <torch/extension.h>
#include <cstdint>
#include <vector>

#include "../neon_cache_config.h"
#include "../neon_cache_microkernels.h"

#ifndef FUSED_CPP_MK_ENABLE_QK_PACKK_INNER
#define FUSED_CPP_MK_ENABLE_QK_PACKK_INNER 1
#endif

namespace fused_cpp::sdpa_microkernels {

#if FUSED_CPP_MK_ENABLE_QK_PACKK_INNER
struct MK_QkPackkInner {
  static constexpr const char* kName = "qk_packk_inner";
  static constexpr bool kEnabled = true;

  static inline void qkt_8x8(
      const at::BFloat16* Q, int64_t q_row_stride,
      const at::BFloat16* K, int64_t k_row_stride,
      int64_t E, float scale, float* scores_buf) {
#if FUSED_CPP_SDPA_CACHE_HAS_BFMMLA
    // thread_local cache：跳过重复 pack。
    static thread_local std::vector<uint16_t> packed_buf;
    static thread_local const at::BFloat16* last_K = nullptr;
    static thread_local int64_t last_k_row_stride = 0;
    static thread_local int64_t last_E = 0;

    if (K != last_K || k_row_stride != last_k_row_stride || E != last_E) {
      packed_buf.assign(static_cast<size_t>(8 * E), 0);
      pack_k_8rows_to_pairs_bf16(K, k_row_stride, E, packed_buf.data());
      last_K = K;
      last_k_row_stride = k_row_stride;
      last_E = E;
    }
    gemm_qkt_microkernel_8x8_bf16_packk_inner(
        Q, q_row_stride, packed_buf.data(), K, k_row_stride, E, scale, scores_buf);
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
#endif  // FUSED_CPP_MK_ENABLE_QK_PACKK_INNER

}  // namespace fused_cpp::sdpa_microkernels
