// ── L3-Resident SDPA with Pre-Packed Q + K + V ──────────────────────────
//
// 本文件实现 `flash2_neon_l3kv_packqkv` SDPA 变体。与 `flash2_neon_l3kv_packv`
// 的区别：
//
//   * V 仍然在 SDPA 入口多线程 pre-pack 成 [B, N, Ev/8, S, 8]（与 packv 一致）；
//   * **K 在 SDPA 入口多线程 pre-pack** 成 [B, N, S/8, E_main/4, 32 u16]，
//     与 packqk_seq4 microkernel 的 K_seq 入参对齐；
//   * Q 在 process_q_tile_lc_packqkv 入口处一次性 pack 当前 q-tile 的所有
//     完整 8-row 子块到 thread-local q_seq_buf；q-tile 内所有 (s_l3, s_l2,
//     qi_inner, s_off) 复用 packed Q，**不重复 pack**。
//
// QKᵀ inner 走 gemm_qkt_microkernel_8x8_bf16_packqk_seq4_bmajor_inner：4 条
// 独立 vld1q_u16 + B-major BFMMLA（microkernel benchmark 上 +23% vs baseline）。
// PV 走 packed V 路径。
//
// 限制：
//   * **bf16-only**：fp32 没 BFMMLA 加速，pack K/Q 纯亏。fp32 输入直接 delegate
//     到 `flash2_neon_l3kv_packv`。
//   * **`S % 8 == 0` 且 `Ev % 8 == 0`**：partial S block 处理复杂，且生产场景
//     S 全是 2 的幂。不满足时建议改用 `flash2_neon_l3kv_packv`。
//   * `E % 4 != 0` 时 partial e_block 不 pack，inner 标量 tail 自动从原始 K
//     指针读 [E_main, E)；与 baseline bit-for-bit 等价。
//
// Path A（KV 装 L3）才启用 K pack；Path B（KV 不装 L3）会引入 3× DRAM 带宽
// （pack 读原 K + pack 写 K_packed + compute 读 K_packed），可能负收益——
// 退化为 packv 行为，跳过 K pack。

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "sdpa_common.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sdpa_microkernels/neon_cache_config.h"
#include "sdpa_pack_utils.h"
#include "sdpa_tile_sizes.h"
#include "sdpa_microkernels/mk_traits.h"
#include "sdpa_microkernels/all_impls.h"
#include "sdpa_microkernels/neon_cache_microkernels.h"
#include "sdpa_flash2_neon_l3kv_impl.h"

namespace {

using ::fused_cpp::sdpa_flash2_neon_l3kv_impl::run_path_collapse3;
using ::fused_cpp::sdpa_flash2_neon_l3kv_impl::run_path_collapse3_packqkv;
using ::fused_cpp::sdpa_flash2_neon_l3kv_impl::run_path_taskloop;
using ::fused_cpp::sdpa_pack_utils::pack_v_to_evblock8;
using ::fused_cpp::sdpa_tile_sizes::compute_tile_sizes_l3kv;
using ::fused_cpp::sdpa_tile_sizes::effective_cache_bytes;
using ::fused_cpp::sdpa_tile_sizes::TileSizes;

// ──────────────────────────────────────────────────────────────────────
// V pack helper is shared with packv through sdpa_pack_utils. SVE targets use
// predicated svld1/svst1 for the 8-element row copy, with a 4-row S unroll and
// an explicit 0..3-row tail.
// ──────────────────────────────────────────────────────────────────────

// fp32 K pack layout for the new QK^T path:
//   K_packed[b, n, s_block, e, lane] = K[b, n, s_block * 8 + lane, e]
//
// This is the fp32 counterpart of the bf16 seq8 K pack, but it keeps Q in its
// original row-major layout.  The QK^T microkernel loads four contiguous Q
// values per row and uses lane FMLA into vector score accumulators.
void pack_k_fp32_to_sblock8(const float* k_src, float* k_dst, int64_t B, int64_t N, int64_t S, int64_t E) {
  TORCH_CHECK(S % 8 == 0, "pack_k_fp32_to_sblock8 requires S % 8 == 0, got S=", S);
  const int64_t S_blocks = S / 8;
  const int64_t src_stride_b = N * S * E;
  const int64_t src_stride_n = S * E;
  const int64_t dst_stride_b = N * S_blocks * E * 8;
  const int64_t dst_stride_n = S_blocks * E * 8;
  const int64_t dst_stride_sb = E * 8;

#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t sb = 0; sb < S_blocks; ++sb) {
        const float* src = k_src + b * src_stride_b + n * src_stride_n + sb * 8 * E;
        float* dst = k_dst + b * dst_stride_b + n * dst_stride_n + sb * dst_stride_sb;
        for (int64_t e = 0; e < E; ++e) {
          float* d = dst + e * 8;
          d[0] = src[0 * E + e];
          d[1] = src[1 * E + e];
          d[2] = src[2 * E + e];
          d[3] = src[3 * E + e];
          d[4] = src[4 * E + e];
          d[5] = src[5 * E + e];
          d[6] = src[6 * E + e];
          d[7] = src[7 * E + e];
        }
      }
    }
  }
}

inline void qkt_packk8_tail_scalar(const float* Q, int64_t q_row_stride, const float* Kp, int64_t E, float scale,
                                   float* scores_buf, int64_t scores_row_stride, int Lq, int Sk) {
  for (int i = 0; i < Lq; ++i) {
    const float* q = Q + i * q_row_stride;
    float* out = scores_buf + i * scores_row_stride;
    for (int j = 0; j < Sk; ++j) {
      float sum = 0.0f;
      for (int64_t e = 0; e < E; ++e) {
        sum += q[e] * Kp[e * 8 + j];
      }
      out[j] = sum * scale;
    }
  }
}

struct MK_Fp32PackK8PQuad {
  static constexpr const char* kName = "fp32_packk8_pquad";

  static inline void qkt_8x8(const float* Q, int64_t q_row_stride, const float* Kp, int64_t /*k_row_stride*/, int64_t E,
                             float scale, float* scores_buf) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
    const float32x4_t scale_v = vdupq_n_f32(scale);
    float32x4_t lo0 = vdupq_n_f32(0.0f), hi0 = vdupq_n_f32(0.0f);
    float32x4_t lo1 = vdupq_n_f32(0.0f), hi1 = vdupq_n_f32(0.0f);
    float32x4_t lo2 = vdupq_n_f32(0.0f), hi2 = vdupq_n_f32(0.0f);
    float32x4_t lo3 = vdupq_n_f32(0.0f), hi3 = vdupq_n_f32(0.0f);
    float32x4_t lo4 = vdupq_n_f32(0.0f), hi4 = vdupq_n_f32(0.0f);
    float32x4_t lo5 = vdupq_n_f32(0.0f), hi5 = vdupq_n_f32(0.0f);
    float32x4_t lo6 = vdupq_n_f32(0.0f), hi6 = vdupq_n_f32(0.0f);
    float32x4_t lo7 = vdupq_n_f32(0.0f), hi7 = vdupq_n_f32(0.0f);

    const float* q0 = Q + 0 * q_row_stride;
    const float* q1 = Q + 1 * q_row_stride;
    const float* q2 = Q + 2 * q_row_stride;
    const float* q3 = Q + 3 * q_row_stride;
    const float* q4 = Q + 4 * q_row_stride;
    const float* q5 = Q + 5 * q_row_stride;
    const float* q6 = Q + 6 * q_row_stride;
    const float* q7 = Q + 7 * q_row_stride;

    int64_t e = 0;
    for (; e + 4 <= E; e += 4) {
      const float32x4_t qv0 = vld1q_f32(q0 + e);
      const float32x4_t qv1 = vld1q_f32(q1 + e);
      const float32x4_t qv2 = vld1q_f32(q2 + e);
      const float32x4_t qv3 = vld1q_f32(q3 + e);
      const float32x4_t qv4 = vld1q_f32(q4 + e);
      const float32x4_t qv5 = vld1q_f32(q5 + e);
      const float32x4_t qv6 = vld1q_f32(q6 + e);
      const float32x4_t qv7 = vld1q_f32(q7 + e);

      const float32x4_t k0l = vld1q_f32(Kp + (e + 0) * 8 + 0);
      const float32x4_t k0h = vld1q_f32(Kp + (e + 0) * 8 + 4);
      const float32x4_t k1l = vld1q_f32(Kp + (e + 1) * 8 + 0);
      const float32x4_t k1h = vld1q_f32(Kp + (e + 1) * 8 + 4);
      const float32x4_t k2l = vld1q_f32(Kp + (e + 2) * 8 + 0);
      const float32x4_t k2h = vld1q_f32(Kp + (e + 2) * 8 + 4);
      const float32x4_t k3l = vld1q_f32(Kp + (e + 3) * 8 + 0);
      const float32x4_t k3h = vld1q_f32(Kp + (e + 3) * 8 + 4);

#define FUSED_CPP_QKT_PACKK8_FMA_ROW(ID, QV)      \
  do {                                            \
    lo##ID = vfmaq_laneq_f32(lo##ID, k0l, QV, 0); \
    hi##ID = vfmaq_laneq_f32(hi##ID, k0h, QV, 0); \
    lo##ID = vfmaq_laneq_f32(lo##ID, k1l, QV, 1); \
    hi##ID = vfmaq_laneq_f32(hi##ID, k1h, QV, 1); \
    lo##ID = vfmaq_laneq_f32(lo##ID, k2l, QV, 2); \
    hi##ID = vfmaq_laneq_f32(hi##ID, k2h, QV, 2); \
    lo##ID = vfmaq_laneq_f32(lo##ID, k3l, QV, 3); \
    hi##ID = vfmaq_laneq_f32(hi##ID, k3h, QV, 3); \
  } while (0)

      FUSED_CPP_QKT_PACKK8_FMA_ROW(0, qv0);
      FUSED_CPP_QKT_PACKK8_FMA_ROW(1, qv1);
      FUSED_CPP_QKT_PACKK8_FMA_ROW(2, qv2);
      FUSED_CPP_QKT_PACKK8_FMA_ROW(3, qv3);
      FUSED_CPP_QKT_PACKK8_FMA_ROW(4, qv4);
      FUSED_CPP_QKT_PACKK8_FMA_ROW(5, qv5);
      FUSED_CPP_QKT_PACKK8_FMA_ROW(6, qv6);
      FUSED_CPP_QKT_PACKK8_FMA_ROW(7, qv7);
#undef FUSED_CPP_QKT_PACKK8_FMA_ROW
    }

    for (; e < E; ++e) {
      const float32x4_t kl = vld1q_f32(Kp + e * 8 + 0);
      const float32x4_t kh = vld1q_f32(Kp + e * 8 + 4);
#define FUSED_CPP_QKT_PACKK8_FMA_TAIL(ID, QROW) \
  do {                                          \
    const float q = QROW[e];                    \
    lo##ID = vfmaq_n_f32(lo##ID, kl, q);        \
    hi##ID = vfmaq_n_f32(hi##ID, kh, q);        \
  } while (0)
      FUSED_CPP_QKT_PACKK8_FMA_TAIL(0, q0);
      FUSED_CPP_QKT_PACKK8_FMA_TAIL(1, q1);
      FUSED_CPP_QKT_PACKK8_FMA_TAIL(2, q2);
      FUSED_CPP_QKT_PACKK8_FMA_TAIL(3, q3);
      FUSED_CPP_QKT_PACKK8_FMA_TAIL(4, q4);
      FUSED_CPP_QKT_PACKK8_FMA_TAIL(5, q5);
      FUSED_CPP_QKT_PACKK8_FMA_TAIL(6, q6);
      FUSED_CPP_QKT_PACKK8_FMA_TAIL(7, q7);
#undef FUSED_CPP_QKT_PACKK8_FMA_TAIL
    }

#define FUSED_CPP_QKT_PACKK8_STORE_ROW(ID)                            \
  do {                                                                \
    vst1q_f32(scores_buf + (ID) * 8 + 0, vmulq_f32(lo##ID, scale_v)); \
    vst1q_f32(scores_buf + (ID) * 8 + 4, vmulq_f32(hi##ID, scale_v)); \
  } while (0)
    FUSED_CPP_QKT_PACKK8_STORE_ROW(0);
    FUSED_CPP_QKT_PACKK8_STORE_ROW(1);
    FUSED_CPP_QKT_PACKK8_STORE_ROW(2);
    FUSED_CPP_QKT_PACKK8_STORE_ROW(3);
    FUSED_CPP_QKT_PACKK8_STORE_ROW(4);
    FUSED_CPP_QKT_PACKK8_STORE_ROW(5);
    FUSED_CPP_QKT_PACKK8_STORE_ROW(6);
    FUSED_CPP_QKT_PACKK8_STORE_ROW(7);
#undef FUSED_CPP_QKT_PACKK8_STORE_ROW
#else
    qkt_packk8_tail_scalar(Q, q_row_stride, Kp, E, scale, scores_buf, 8, 8, 8);
#endif
  }

  static inline void qkt_8x4(const float* Q, int64_t q_row_stride, const float* Kp, int64_t /*k_row_stride*/, int64_t E,
                             float scale, float* scores_buf, int64_t scores_row_stride) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
    const float32x4_t scale_v = vdupq_n_f32(scale);
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    float32x4_t acc4 = vdupq_n_f32(0.0f);
    float32x4_t acc5 = vdupq_n_f32(0.0f);
    float32x4_t acc6 = vdupq_n_f32(0.0f);
    float32x4_t acc7 = vdupq_n_f32(0.0f);

    const float* q0 = Q + 0 * q_row_stride;
    const float* q1 = Q + 1 * q_row_stride;
    const float* q2 = Q + 2 * q_row_stride;
    const float* q3 = Q + 3 * q_row_stride;
    const float* q4 = Q + 4 * q_row_stride;
    const float* q5 = Q + 5 * q_row_stride;
    const float* q6 = Q + 6 * q_row_stride;
    const float* q7 = Q + 7 * q_row_stride;

    int64_t e = 0;
    for (; e + 4 <= E; e += 4) {
      const float32x4_t qv0 = vld1q_f32(q0 + e);
      const float32x4_t qv1 = vld1q_f32(q1 + e);
      const float32x4_t qv2 = vld1q_f32(q2 + e);
      const float32x4_t qv3 = vld1q_f32(q3 + e);
      const float32x4_t qv4 = vld1q_f32(q4 + e);
      const float32x4_t qv5 = vld1q_f32(q5 + e);
      const float32x4_t qv6 = vld1q_f32(q6 + e);
      const float32x4_t qv7 = vld1q_f32(q7 + e);
      const float32x4_t k0 = vld1q_f32(Kp + (e + 0) * 8);
      const float32x4_t k1 = vld1q_f32(Kp + (e + 1) * 8);
      const float32x4_t k2 = vld1q_f32(Kp + (e + 2) * 8);
      const float32x4_t k3 = vld1q_f32(Kp + (e + 3) * 8);
#define FUSED_CPP_QKT_PACKK4_FMA_ROW(ID, QV)       \
  do {                                             \
    acc##ID = vfmaq_laneq_f32(acc##ID, k0, QV, 0); \
    acc##ID = vfmaq_laneq_f32(acc##ID, k1, QV, 1); \
    acc##ID = vfmaq_laneq_f32(acc##ID, k2, QV, 2); \
    acc##ID = vfmaq_laneq_f32(acc##ID, k3, QV, 3); \
  } while (0)
      FUSED_CPP_QKT_PACKK4_FMA_ROW(0, qv0);
      FUSED_CPP_QKT_PACKK4_FMA_ROW(1, qv1);
      FUSED_CPP_QKT_PACKK4_FMA_ROW(2, qv2);
      FUSED_CPP_QKT_PACKK4_FMA_ROW(3, qv3);
      FUSED_CPP_QKT_PACKK4_FMA_ROW(4, qv4);
      FUSED_CPP_QKT_PACKK4_FMA_ROW(5, qv5);
      FUSED_CPP_QKT_PACKK4_FMA_ROW(6, qv6);
      FUSED_CPP_QKT_PACKK4_FMA_ROW(7, qv7);
#undef FUSED_CPP_QKT_PACKK4_FMA_ROW
    }
    for (; e < E; ++e) {
      const float32x4_t k = vld1q_f32(Kp + e * 8);
#define FUSED_CPP_QKT_PACKK4_FMA_TAIL(ID, QROW) \
  do {                                          \
    acc##ID = vfmaq_n_f32(acc##ID, k, QROW[e]); \
  } while (0)
      FUSED_CPP_QKT_PACKK4_FMA_TAIL(0, q0);
      FUSED_CPP_QKT_PACKK4_FMA_TAIL(1, q1);
      FUSED_CPP_QKT_PACKK4_FMA_TAIL(2, q2);
      FUSED_CPP_QKT_PACKK4_FMA_TAIL(3, q3);
      FUSED_CPP_QKT_PACKK4_FMA_TAIL(4, q4);
      FUSED_CPP_QKT_PACKK4_FMA_TAIL(5, q5);
      FUSED_CPP_QKT_PACKK4_FMA_TAIL(6, q6);
      FUSED_CPP_QKT_PACKK4_FMA_TAIL(7, q7);
#undef FUSED_CPP_QKT_PACKK4_FMA_TAIL
    }
#define FUSED_CPP_QKT_PACKK4_STORE_ROW(ID) vst1q_f32(scores_buf + (ID) * scores_row_stride, vmulq_f32(acc##ID, scale_v))
    FUSED_CPP_QKT_PACKK4_STORE_ROW(0);
    FUSED_CPP_QKT_PACKK4_STORE_ROW(1);
    FUSED_CPP_QKT_PACKK4_STORE_ROW(2);
    FUSED_CPP_QKT_PACKK4_STORE_ROW(3);
    FUSED_CPP_QKT_PACKK4_STORE_ROW(4);
    FUSED_CPP_QKT_PACKK4_STORE_ROW(5);
    FUSED_CPP_QKT_PACKK4_STORE_ROW(6);
    FUSED_CPP_QKT_PACKK4_STORE_ROW(7);
#undef FUSED_CPP_QKT_PACKK4_STORE_ROW
#else
    qkt_packk8_tail_scalar(Q, q_row_stride, Kp, E, scale, scores_buf, scores_row_stride, 8, 4);
#endif
  }

  static inline void qkt_tail(const float* Q, int64_t q_row_stride, const float* Kp, int64_t /*k_row_stride*/,
                              int64_t E, float scale, float* scores_buf, int64_t scores_row_stride, int Lq, int Sk) {
    qkt_packk8_tail_scalar(Q, q_row_stride, Kp, E, scale, scores_buf, scores_row_stride, Lq, Sk);
  }

  static inline void pv_8x8(const float* P_hat, int64_t P_row_stride, const float* V, int64_t v_row_stride, int64_t Sk,
                            float* O, int64_t o_row_stride) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
    ::fused_cpp::sdpa_microkernels::gemm_pv_microkernel_8x8_fp32_pquad(P_hat, P_row_stride, V, v_row_stride, Sk, O,
                                                                       o_row_stride);
#else
    ::fused_cpp::sdpa_microkernels::gemm_pv_8x8(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
#endif
  }

  static inline void pv_tail(const float* P_hat, int64_t P_row_stride, const float* V, int64_t v_row_stride, int64_t Sk,
                             float* O, int64_t o_row_stride, int Lq, int Ev) {
    ::fused_cpp::sdpa_microkernels::gemm_pv_tail(P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride, Lq, Ev);
  }
};

void sdpa_flash2_neon_l3kv_packqkv_fp32_packk_pquad_impl(const SdpaParams& p, const char* version_name) {
  TORCH_CHECK(p.dtype == SdpaDtype::kFloat32, "fp32 packK path called with non-fp32 dtype");
  TORCH_CHECK(p.S % 8 == 0, version_name, " fp32 packK path requires S % 8 == 0, got S=", p.S);
  TORCH_CHECK(p.Ev % 8 == 0, version_name, " fp32 packK path requires Ev % 8 == 0, got Ev=", p.Ev);

  const bool profile_on = ::fused_cpp::sdpa_profile::enabled();
  if (profile_on) {
    ::fused_cpp::sdpa_profile::reset();
  }
  const uint64_t profile_total_t0 = profile_on ? ::fused_cpp::sdpa_profile::now_ns() : 0;

  const auto* q_ptr = static_cast<const float*>(p.q_ptr);
  const auto* k_ptr = static_cast<const float*>(p.k_ptr);
  const auto* v_ptr = static_cast<const float*>(p.v_ptr);

  at::Tensor v_packed;
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kVAlloc);
    v_packed = at::empty({p.B, p.N, p.Ev / 8, p.S, 8}, at::TensorOptions().dtype(at::kFloat));
  }
  auto* v_packed_ptr = static_cast<float*>(v_packed.data_ptr());
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kVPack);
    pack_v_to_evblock8<float>(v_ptr, v_packed_ptr, p.B, p.N, p.S, p.Ev);
  }

  at::Tensor k_packed;
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kKAlloc);
    k_packed = at::empty({p.B, p.N, p.S / 8, p.E, 8}, at::TensorOptions().dtype(at::kFloat));
  }
  auto* k_packed_ptr = static_cast<float*>(k_packed.data_ptr());
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kKPack);
    pack_k_fp32_to_sblock8(k_ptr, k_packed_ptr, p.B, p.N, p.S, p.E);
  }

  const int64_t q_stride_b = p.N * p.L * p.E;
  const int64_t q_stride_n = p.L * p.E;
  const int64_t q_stride_l = p.E;

  const int64_t S_blocks = p.S / 8;
  const int64_t k_packed_stride_n = S_blocks * p.E * 8;
  const int64_t k_packed_stride_b = p.N * k_packed_stride_n;
  const int64_t k_packed_stride_s = p.E;

  const int64_t Eb = p.Ev / 8;
  const int64_t v_evblock_stride = p.S * 8;
  const int64_t v_packed_stride_n = Eb * v_evblock_stride;
  const int64_t v_packed_stride_b = p.N * v_packed_stride_n;
  const int64_t v_packed_stride_s = 8;

  const int64_t m_stride_b = p.N * p.L * p.S;
  const int64_t m_stride_n = p.L * p.S;
  const int64_t m_stride_l = p.S;
  const int64_t o_stride_b = p.N * p.L * p.Ev;
  const int64_t o_stride_n = p.L * p.Ev;
  const int64_t o_stride_l = p.Ev;

  TileSizes ts = compute_tile_sizes_l3kv(p.B, p.N, p.S, p.L, p.E, p.Ev, sizeof(float));

  const int64_t kv_bytes_per_bn = p.S * (p.E + p.Ev) * static_cast<int64_t>(sizeof(float));
  const auto& cache_bytes = effective_cache_bytes();
  const int64_t l3_budget = static_cast<int64_t>(cache_bytes[2] * FUSED_CPP_SDPA_L3_RATIO);
  const bool kv_fits_l3 = kv_bytes_per_bn <= l3_budget;
  const int total_threads =
#ifdef _OPENMP
      omp_get_max_threads();
#else
      1;
#endif
  const bool path_a = kv_fits_l3 || total_threads <= 1;

  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kMain);
#define FUSED_CPP_RUN_FP32_PACKK(HAS_MASK, CAUSAL)                                                                     \
  do {                                                                                                                 \
    if (path_a) {                                                                                                      \
      run_path_collapse3<MK_Fp32PackK8PQuad, float, true, HAS_MASK, CAUSAL>(                                           \
          q_ptr, k_packed_ptr, v_packed_ptr, p, ts, q_stride_b, q_stride_n, q_stride_l, k_packed_stride_b,             \
          k_packed_stride_n, k_packed_stride_s, v_packed_stride_b, v_packed_stride_n, v_packed_stride_s,               \
          v_evblock_stride, m_stride_b, m_stride_n, m_stride_l, o_stride_b, o_stride_n, o_stride_l);                   \
    } else {                                                                                                           \
      const int64_t num_groups = std::max<int64_t>(1, total_threads);                                                  \
      run_path_taskloop<MK_Fp32PackK8PQuad, float, true, HAS_MASK, CAUSAL>(                                            \
          q_ptr, k_packed_ptr, v_packed_ptr, p, ts, num_groups, q_stride_b, q_stride_n, q_stride_l, k_packed_stride_b, \
          k_packed_stride_n, k_packed_stride_s, v_packed_stride_b, v_packed_stride_n, v_packed_stride_s,               \
          v_evblock_stride, m_stride_b, m_stride_n, m_stride_l, o_stride_b, o_stride_n, o_stride_l);                   \
    }                                                                                                                  \
  } while (0)

    if (p.mask_ptr != nullptr) {
      if (p.is_causal) {
        FUSED_CPP_RUN_FP32_PACKK(true, true);
      } else {
        FUSED_CPP_RUN_FP32_PACKK(true, false);
      }
    } else {
      if (p.is_causal) {
        FUSED_CPP_RUN_FP32_PACKK(false, true);
      } else {
        FUSED_CPP_RUN_FP32_PACKK(false, false);
      }
    }
#undef FUSED_CPP_RUN_FP32_PACKK
  }

  if (profile_on) {
    ::fused_cpp::sdpa_profile::add(::fused_cpp::sdpa_profile::Slot::kTotal,
                                   ::fused_cpp::sdpa_profile::now_ns() - profile_total_t0);
    ::fused_cpp::sdpa_profile::print_summary(version_name, MK_Fp32PackK8PQuad::kName, p, path_a ? "A" : "B");
  }
}

// ──────────────────────────────────────────────────────────────────────
// 顶层入口：packqkv (bf16 path)
// ──────────────────────────────────────────────────────────────────────

template <bool kPbf16PV, int kExpPolyDegree = 5>
void sdpa_flash2_neon_l3kv_packqkv_bf16_impl_tmpl(const SdpaParams& p, const char* version_name) {
  TORCH_CHECK(p.dtype == SdpaDtype::kBFloat16, "packqkv bf16 path called with non-bf16 dtype");
  TORCH_CHECK(p.S % 8 == 0, version_name, " requires S % 8 == 0, got S=", p.S,
              "; please use flash2_neon_l3kv_packv for non-aligned S.");
  TORCH_CHECK(p.Ev % 8 == 0, version_name, " requires Ev % 8 == 0, got Ev=", p.Ev);
  const bool profile_on = ::fused_cpp::sdpa_profile::enabled();
  if (profile_on) {
    ::fused_cpp::sdpa_profile::reset();
  }
  const uint64_t profile_total_t0 = profile_on ? ::fused_cpp::sdpa_profile::now_ns() : 0;

  const auto* q_ptr = static_cast<const at::BFloat16*>(p.q_ptr);
  const auto* k_ptr = static_cast<const at::BFloat16*>(p.k_ptr);
  const auto* v_ptr = static_cast<const at::BFloat16*>(p.v_ptr);

  // ── 分配 packed V buffer ──
  at::Tensor v_packed;
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kVAlloc);
    v_packed = at::empty({p.B, p.N, p.Ev / 8, p.S, 8}, at::TensorOptions().dtype(at::kBFloat16));
  }
  auto* v_packed_ptr = static_cast<at::BFloat16*>(v_packed.data_ptr());
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kVPack);
    pack_v_to_evblock8<at::BFloat16>(v_ptr, v_packed_ptr, p.B, p.N, p.S, p.Ev);
  }

  // ── strides ──
  const int64_t Eb = p.Ev / 8;
  const int64_t v_evblock_stride = p.S * 8;
  const int64_t v_packed_stride_n = Eb * v_evblock_stride;
  const int64_t v_packed_stride_b = p.N * v_packed_stride_n;
  const int64_t v_packed_stride_s = 8;

  const int64_t q_stride_b = p.N * p.L * p.E;
  const int64_t q_stride_n = p.L * p.E;
  const int64_t q_stride_l = p.E;
  const int64_t k_orig_stride_b = p.N * p.S * p.E;
  const int64_t k_orig_stride_n = p.S * p.E;
  const int64_t k_orig_stride_s = p.E;
  const int64_t m_stride_b = p.N * p.L * p.S;
  const int64_t m_stride_n = p.L * p.S;
  const int64_t m_stride_l = p.S;
  const int64_t o_stride_b = p.N * p.L * p.Ev;
  const int64_t o_stride_n = p.L * p.Ev;
  const int64_t o_stride_l = p.Ev;

  TileSizes ts = compute_tile_sizes_l3kv(p.B, p.N, p.S, p.L, p.E, p.Ev, sizeof(at::BFloat16));

  const int64_t kv_bytes_per_bn = p.S * (p.E + p.Ev) * static_cast<int64_t>(sizeof(at::BFloat16));
  const auto& cache_bytes = effective_cache_bytes();
  const int64_t l3_budget = static_cast<int64_t>(cache_bytes[2] * FUSED_CPP_SDPA_L3_RATIO);
  const bool kv_fits_l3 = kv_bytes_per_bn <= l3_budget;
  const int total_threads =
#ifdef _OPENMP
      omp_get_max_threads();
#else
      1;
#endif

  // ── Path B：不 pack K，退化为 flash2_neon_l3kv_packv 行为 ──
  // 通过 sdpa_dispatch 路由，避免重复实现 path B 全套逻辑。注意：
  // V 已经 pack 过了，但 packv 入口会再 pack 一次（白做一份 V copy）。
  // 这是工程取舍——packv 入口的多线程 pack overhead 与一次 SDPA 计算
  // 相比是小项；为了避免在本文件再写一份完整的 path B taskloop 调用，
  // 接受这点冗余。如果实测显著影响 path B 性能，可以单独优化。
  if (!kv_fits_l3 && total_threads > 1) {
    sdpa_dispatch("flash2_neon_l3kv_packv_pquad", p);
    return;
  }

  // ── Path A：分配 packed K buffer 并 pack ──
  const int64_t E_main = p.E & ~int64_t{3};
  const int64_t e_blocks = E_main / 4;
  const int64_t kblock_u16 = e_blocks * 32;  // 单个 8-row 子块容量
  const int64_t S_blocks = p.S / 8;
  const int64_t k_packed_total_u16 = p.B * p.N * S_blocks * kblock_u16;

  // 用 BFloat16 dtype 分配（u16 兼容），元素数按 u16 算
  // 注意：at::empty 没有 kU16，用 BF16 alias，因为我们只关心字节数与对齐
  at::Tensor k_packed;
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kKAlloc);
    k_packed = at::empty({k_packed_total_u16}, at::TensorOptions().dtype(at::kBFloat16));
  }
  auto* k_packed_ptr = reinterpret_cast<uint16_t*>(k_packed.data_ptr());

  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kKPack);
    ::fused_cpp::sdpa_microkernels::pack_k_to_seq8<at::BFloat16>(k_ptr, k_packed_ptr, p.B, p.N, p.S, p.E);
  }

  const int64_t k_sblock_stride = kblock_u16;  // u16 单位
  const int64_t k_packed_stride_n = S_blocks * kblock_u16;
  const int64_t k_packed_stride_b = p.N * k_packed_stride_n;

  // ── 4 路 (kHasMask, kCausal) 分发 ──
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kMain);
    if (p.mask_ptr != nullptr) {
      if (p.is_causal) {
        run_path_collapse3_packqkv<true, true, kPbf16PV, kExpPolyDegree>(
            q_ptr, k_packed_ptr, k_ptr, v_packed_ptr, p, ts, q_stride_b, q_stride_n, q_stride_l, k_orig_stride_b,
            k_orig_stride_n, k_orig_stride_s, k_packed_stride_b, k_packed_stride_n, k_sblock_stride, v_packed_stride_b,
            v_packed_stride_n, v_packed_stride_s, v_evblock_stride, m_stride_b, m_stride_n, m_stride_l, o_stride_b,
            o_stride_n, o_stride_l);
      } else {
        run_path_collapse3_packqkv<true, false, kPbf16PV, kExpPolyDegree>(
            q_ptr, k_packed_ptr, k_ptr, v_packed_ptr, p, ts, q_stride_b, q_stride_n, q_stride_l, k_orig_stride_b,
            k_orig_stride_n, k_orig_stride_s, k_packed_stride_b, k_packed_stride_n, k_sblock_stride, v_packed_stride_b,
            v_packed_stride_n, v_packed_stride_s, v_evblock_stride, m_stride_b, m_stride_n, m_stride_l, o_stride_b,
            o_stride_n, o_stride_l);
      }
    } else {
      if (p.is_causal) {
        run_path_collapse3_packqkv<false, true, kPbf16PV, kExpPolyDegree>(
            q_ptr, k_packed_ptr, k_ptr, v_packed_ptr, p, ts, q_stride_b, q_stride_n, q_stride_l, k_orig_stride_b,
            k_orig_stride_n, k_orig_stride_s, k_packed_stride_b, k_packed_stride_n, k_sblock_stride, v_packed_stride_b,
            v_packed_stride_n, v_packed_stride_s, v_evblock_stride, m_stride_b, m_stride_n, m_stride_l, o_stride_b,
            o_stride_n, o_stride_l);
      } else {
        run_path_collapse3_packqkv<false, false, kPbf16PV, kExpPolyDegree>(
            q_ptr, k_packed_ptr, k_ptr, v_packed_ptr, p, ts, q_stride_b, q_stride_n, q_stride_l, k_orig_stride_b,
            k_orig_stride_n, k_orig_stride_s, k_packed_stride_b, k_packed_stride_n, k_sblock_stride, v_packed_stride_b,
            v_packed_stride_n, v_packed_stride_s, v_evblock_stride, m_stride_b, m_stride_n, m_stride_l, o_stride_b,
            o_stride_n, o_stride_l);
      }
    }
  }
  if (profile_on) {
    ::fused_cpp::sdpa_profile::add(::fused_cpp::sdpa_profile::Slot::kTotal,
                                   ::fused_cpp::sdpa_profile::now_ns() - profile_total_t0);
    ::fused_cpp::sdpa_profile::print_summary(
        version_name, kPbf16PV ? "qk_packqk_seq4_bmajor_pv_pbf16_prepacked" : "qk_packqk_seq4_bmajor_pv_pquad", p, "A");
  }
}

void sdpa_flash2_neon_l3kv_packqkv_bf16_impl(const SdpaParams& p) {
  sdpa_flash2_neon_l3kv_packqkv_bf16_impl_tmpl<false>(p, "flash2_neon_l3kv_packqkv");
}

void sdpa_flash2_neon_l3kv_packqkv_pbf16pv_bf16_impl(const SdpaParams& p) {
  sdpa_flash2_neon_l3kv_packqkv_bf16_impl_tmpl<true, 5>(p, "flash2_neon_l3kv_packqkv_pbf16pv");
}

void sdpa_flash2_neon_l3kv_packqkv_pbf16pv_exp_poly4_bf16_impl(const SdpaParams& p) {
  sdpa_flash2_neon_l3kv_packqkv_bf16_impl_tmpl<true, 4>(p, "flash2_neon_l3kv_packqkv_pbf16pv_exp_poly4");
}

void sdpa_flash2_neon_l3kv_packqkv_pbf16pv_exp_poly6_bf16_impl(const SdpaParams& p) {
  sdpa_flash2_neon_l3kv_packqkv_bf16_impl_tmpl<true, 6>(p, "flash2_neon_l3kv_packqkv_pbf16pv_exp_poly6");
}

void sdpa_flash2_neon_l3kv_packqkv_entry(const SdpaParams& p) {
  // fp32 输入走 K+V pre-pack + fp32 packed-K QK^T + PV pquad。bf16 走本文件
  // 的 packqkv_bf16_impl，PV 已切到 MK_QkPackqkSeq4BmajorPvPquad::pv_8x8。
  if (p.dtype != SdpaDtype::kBFloat16) {
    sdpa_flash2_neon_l3kv_packqkv_fp32_packk_pquad_impl(p, "flash2_neon_l3kv_packqkv");
    return;
  }
  sdpa_flash2_neon_l3kv_packqkv_bf16_impl(p);
}

void sdpa_flash2_neon_l3kv_packqkv_pbf16pv_entry(const SdpaParams& p) {
  if (p.dtype != SdpaDtype::kBFloat16) {
    sdpa_flash2_neon_l3kv_packqkv_fp32_packk_pquad_impl(p, "flash2_neon_l3kv_packqkv_pbf16pv");
    return;
  }
  sdpa_flash2_neon_l3kv_packqkv_pbf16pv_bf16_impl(p);
}

void sdpa_flash2_neon_l3kv_packqkv_pbf16pv_exp_poly4_entry(const SdpaParams& p) {
  if (p.dtype != SdpaDtype::kBFloat16) {
    sdpa_flash2_neon_l3kv_packqkv_fp32_packk_pquad_impl(p, "flash2_neon_l3kv_packqkv_pbf16pv_exp_poly4");
    return;
  }
  sdpa_flash2_neon_l3kv_packqkv_pbf16pv_exp_poly4_bf16_impl(p);
}

void sdpa_flash2_neon_l3kv_packqkv_pbf16pv_exp_poly6_entry(const SdpaParams& p) {
  if (p.dtype != SdpaDtype::kBFloat16) {
    sdpa_flash2_neon_l3kv_packqkv_fp32_packk_pquad_impl(p, "flash2_neon_l3kv_packqkv_pbf16pv_exp_poly6");
    return;
  }
  sdpa_flash2_neon_l3kv_packqkv_pbf16pv_exp_poly6_bf16_impl(p);
}

}  // anonymous namespace

REGISTER_SDPA_VERSION("flash2_neon_l3kv_packqkv", sdpa_flash2_neon_l3kv_packqkv_entry);
REGISTER_SDPA_VERSION("flash2_neon_l3kv_packqkv_pbf16pv", sdpa_flash2_neon_l3kv_packqkv_pbf16pv_entry);
REGISTER_SDPA_VERSION("flash2_neon_l3kv_packqkv_pbf16pv_exp_poly4",
                      sdpa_flash2_neon_l3kv_packqkv_pbf16pv_exp_poly4_entry);
REGISTER_SDPA_VERSION("flash2_neon_l3kv_packqkv_pbf16pv_exp_poly6",
                      sdpa_flash2_neon_l3kv_packqkv_pbf16pv_exp_poly6_entry);
