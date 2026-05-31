#pragma once
// ── 微内核 impl：MK_QkPackqkSeq4Ptr ──────────────────────────────────────────
//
// **评估用 trait，不接入 SDPA 主路径**——在 MK_QkPackkSeq 已经把 K 完全按
// kernel 访存顺序排成 [E/4][32 u16] / 单条 vld1q_u16_x4 拿一整 cacheline
// 之后，本 trait 把 Q 也按同样布局 pre-pack。inner kernel 的 A/B 双侧都走
// `vld1q_u16`（各 4 条独立 load），验证 seq4 的 ptr 调度变体。
//
// 与 packk_seq 的关键差别：
//   * packk_seq：仅 K 走 packed load；Q 仍是 baseline 的 8 vld1_u16 + 4
//     vcombine_u16，跨 q_row_stride 8 个独立 8-byte load。
//   * packqk_seq（本 trait）：Q 也走 packed load；A 路径从 8 vld1 + 4
//     vcombine → 4 vld1q_u16 + 4 vreinterpret，cacheline 利用率
//     1/8 → 1/4，HW prefetcher 看到「Q + K 两条独立 stride-64 stream」。
//
// 用 thread_local cache 跳过重复 pack：**Q 与 K 各自独立**两份 buffer，
// 缓存键 (Q_ptr, q_row_stride, E) / (K_ptr, k_row_stride, E) 独立判断。
// **不要共用 buffer**——两个键独立失效，共用会让 K-cache 重置时覆盖
// Q-cache，下一次 Q 命中读到脏数据。
//
// ⚠️  Q-cache 仅在 microkernel benchmark 的 inner loop 中命中（同一 Q 重复
// 调用），**这并不模拟真实 SDPA 场景**——SDPA 主循环中 Q 按 8-row 滑窗
// 变化，每 i-tile 必 miss 触发 8*E*2 字节 pack。本 trait 给出的是 'Q 也
// 已经 pre-pack 时' 的理论上限，**不能外推到 SDPA 端到端收益**。
//
// 只重写 bf16 qkt_8x8（评估目标）；其余 9 个 dtype × op 组合全部 fall
// through 到 baseline。
//
// 编译期开关：FUSED_CPP_MK_ENABLE_QK_PACKQK_SEQ4_PTR，默认 1。

#include <torch/extension.h>
#include <cstdint>
#include <vector>

#include "../neon_cache_config.h"
#include "../neon_cache_microkernels.h"

#ifndef FUSED_CPP_MK_ENABLE_QK_PACKQK_SEQ4_PTR
#define FUSED_CPP_MK_ENABLE_QK_PACKQK_SEQ4_PTR 1
#endif

namespace fused_cpp::sdpa_microkernels {

#if FUSED_CPP_MK_ENABLE_QK_PACKQK_SEQ4_PTR
struct MK_QkPackqkSeq4Ptr {
  static constexpr const char* kName = "qk_packqk_seq4_ptr";
  static constexpr bool kEnabled = true;

  static inline void qkt_8x8(
      const at::BFloat16* Q, int64_t q_row_stride,
      const at::BFloat16* K, int64_t k_row_stride,
      int64_t E, float scale, float* scores_buf) {
#if FUSED_CPP_SDPA_CACHE_HAS_BFMMLA
    // ── Q 侧 thread_local cache ──
    static thread_local std::vector<uint16_t> q_packed_buf;
    static thread_local const at::BFloat16* last_Q = nullptr;
    static thread_local int64_t last_q_row_stride = 0;
    static thread_local int64_t last_q_E = 0;

    if (Q != last_Q || q_row_stride != last_q_row_stride || E != last_q_E) {
      q_packed_buf.assign(static_cast<size_t>(8 * E), 0);
      pack_q_8rows_to_seq_bf16(Q, q_row_stride, E, q_packed_buf.data());
      last_Q = Q;
      last_q_row_stride = q_row_stride;
      last_q_E = E;
    }

    // ── K 侧 thread_local cache（与 packk_seq 完全独立的一份）──
    static thread_local std::vector<uint16_t> k_packed_buf;
    static thread_local const at::BFloat16* last_K = nullptr;
    static thread_local int64_t last_k_row_stride = 0;
    static thread_local int64_t last_k_E = 0;

    if (K != last_K || k_row_stride != last_k_row_stride || E != last_k_E) {
      k_packed_buf.assign(static_cast<size_t>(8 * E), 0);
      pack_k_8rows_to_seq_bf16(K, k_row_stride, E, k_packed_buf.data());
      last_K = K;
      last_k_row_stride = k_row_stride;
      last_k_E = E;
    }

    gemm_qkt_microkernel_8x8_bf16_packqk_seq4_ptr_inner(
        q_packed_buf.data(), Q, q_row_stride,
        k_packed_buf.data(), K, k_row_stride,
        E, scale, scores_buf);
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

  // —— 其他 op 全部 fall through 到 baseline ——
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
#endif  // FUSED_CPP_MK_ENABLE_QK_PACKQK_SEQ4_PTR

}  // namespace fused_cpp::sdpa_microkernels
