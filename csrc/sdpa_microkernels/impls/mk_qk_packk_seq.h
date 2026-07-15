#pragma once
// ── 微内核 impl：MK_QkPackkSeq ───────────────────────────────────────────
//
// **评估用 trait，不接入 SDPA 主路径**——测「外层把 K 完全按 kernel 访存
// 顺序排布好」的 inner microkernel 峰值效率，是 K-pack 收益的理论上限。
//
// 与 MK_QkPackkInner 的关键差别：
//   * QkPackkInner：按 row-pair 拼成 [4 pair][E*2] u16，4 个 pair 之间
//     仍跨 stride，内层 4 vld1q_u16 各自独立 base。
//   * QkPackkSeq（本 trait）：按 e_block 摊平成 [E/4][32 u16]，每 e_block
//     64 字节 = 一整 cacheline = 4 个 BFMMLA B 操作数紧密相邻。内层用
//     **单条 vld1q_u16_x4** 拿一整 cacheline → 4 个 uint16x8_t，K-LSU 端从
//     4 vld1q → 1 vld1q_x4。HW prefetcher 看到的是 stride-64 单流，最优。
//
// 同样用 thread_local cache 跳过重复 pack，模拟 SDPA 真实场景下 K 在外层
// pack 一次给 L/8 个 i-tile 共用的 amortize 后行为。
//
// 只重写 bf16 qkt_8x8；其余 9 个 dtype×op 组合 fall through 到 baseline。
//
// 编译期开关：FUSED_CPP_MK_ENABLE_QK_PACKK_SEQ，默认 1。

#include <torch/extension.h>
#include <cstdint>
#include <vector>

#include "../neon_cache_config.h"
#include "../neon_cache_microkernels.h"

#ifndef FUSED_CPP_MK_ENABLE_QK_PACKK_SEQ
#define FUSED_CPP_MK_ENABLE_QK_PACKK_SEQ 1
#endif

namespace fused_cpp::sdpa_microkernels {

#if FUSED_CPP_MK_ENABLE_QK_PACKK_SEQ
struct MK_QkPackkSeq {
  static constexpr const char* kName = "qk_packk_seq";
  static constexpr bool kEnabled = true;

  static inline void qkt_8x8(const at::BFloat16* Q, int64_t q_row_stride, const at::BFloat16* K, int64_t k_row_stride,
                             int64_t E, float scale, float* scores_buf) {
#if FUSED_CPP_SDPA_CACHE_HAS_BFMMLA
    static thread_local std::vector<uint16_t> packed_buf;
    static thread_local const at::BFloat16* last_K = nullptr;
    static thread_local int64_t last_k_row_stride = 0;
    static thread_local int64_t last_E = 0;

    if (K != last_K || k_row_stride != last_k_row_stride || E != last_E) {
      packed_buf.assign(static_cast<size_t>(8 * E), 0);
      pack_k_8rows_to_seq_bf16(K, k_row_stride, E, packed_buf.data());
      last_K = K;
      last_k_row_stride = k_row_stride;
      last_E = E;
    }
    gemm_qkt_microkernel_8x8_bf16_packk_seq_inner(Q, q_row_stride, packed_buf.data(), K, k_row_stride, E, scale,
                                                  scores_buf);
#else
    gemm_qkt_8x8(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf);
#endif
  }
  static inline void qkt_8x8(const float* Q, int64_t q_row_stride, const float* K, int64_t k_row_stride, int64_t E,
                             float scale, float* scores_buf) {
    gemm_qkt_8x8(Q, q_row_stride, K, k_row_stride, E, scale, scores_buf);
  }

  // 其他 op 全部 fall through。
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
    gemm_pv_8x8(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
  }
  static inline void pv_8x8(const float* P_hat, int64_t P_row_stride, const float* V, int64_t v_row_stride, int64_t Sk,
                            float* O, int64_t o_row_stride) {
    gemm_pv_8x8(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
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
#endif  // FUSED_CPP_MK_ENABLE_QK_PACKK_SEQ

}  // namespace fused_cpp::sdpa_microkernels
