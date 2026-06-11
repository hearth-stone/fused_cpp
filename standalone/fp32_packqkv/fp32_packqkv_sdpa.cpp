#include "fp32_packqkv_sdpa.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>

#include "sdpa_common.h"
#include "sdpa_tile_sizes.h"
#include "sdpa_pack_utils.h"
#include "sdpa_flash2_neon_l3kv_impl.h"
#include "sdpa_microkernels/neon_cache_microkernels.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fp32_packqkv_sdpa {
namespace {

using ::fused_cpp::sdpa_flash2_neon_l3kv_impl::run_path_collapse3;
using ::fused_cpp::sdpa_flash2_neon_l3kv_impl::run_path_taskloop;
using ::fused_cpp::sdpa_pack_utils::pack_v_to_evblock8;
using ::fused_cpp::sdpa_tile_sizes::compute_tile_sizes_l3kv;
using ::fused_cpp::sdpa_tile_sizes::effective_cache_bytes;
using ::fused_cpp::sdpa_tile_sizes::TileSizes;

constexpr int64_t kEvBlock = 8;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

inline int64_t clamp_i64(int64_t v, int64_t lo, int64_t hi) {
  return std::max(lo, std::min(v, hi));
}

inline void check_config(const Config& cfg) {
  if (cfg.B <= 0 || cfg.N <= 0 || cfg.L <= 0 || cfg.S <= 0 ||
      cfg.E <= 0 || cfg.Ev <= 0) {
    throw std::invalid_argument("shape dimensions must be positive");
  }
  if (cfg.S % 8 != 0) {
    throw std::invalid_argument("fp32 packqkv path requires S % 8 == 0");
  }
  if (cfg.Ev % kEvBlock != 0) {
    throw std::invalid_argument("fp32 packqkv path requires Ev % 8 == 0");
  }
  if (cfg.s_tile < 0 || cfg.s_tile % 8 != 0) {
    throw std::invalid_argument(
        "fp32 packqkv path requires s_tile == 0 or s_tile % 8 == 0");
  }
}

void pack_k_fp32_to_sblock8(
    const float* k_src,
    float* k_dst,
    int64_t B,
    int64_t N,
    int64_t S,
    int64_t E) {
  TORCH_CHECK(S % 8 == 0,
              "pack_k_fp32_to_sblock8 requires S % 8 == 0, got S=", S);
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
        const float* src = k_src + b * src_stride_b
                                 + n * src_stride_n
                                 + sb * 8 * E;
        float* dst = k_dst + b * dst_stride_b
                           + n * dst_stride_n
                           + sb * dst_stride_sb;
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

inline void qkt_packk8_tail_scalar(
    const float* Q,
    int64_t q_row_stride,
    const float* Kp,
    int64_t E,
    float scale,
    float* scores_buf,
    int64_t scores_row_stride,
    int Lq,
    int Sk) {
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

  static inline void qkt_8x8(
      const float* Q, int64_t q_row_stride,
      const float* Kp, int64_t,
      int64_t E, float scale, float* scores_buf) {
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

#define FUSED_CPP_QKT_PACKK8_FMA_ROW(ID, QV)                       \
      do {                                                         \
        lo##ID = vfmaq_laneq_f32(lo##ID, k0l, QV, 0);              \
        hi##ID = vfmaq_laneq_f32(hi##ID, k0h, QV, 0);              \
        lo##ID = vfmaq_laneq_f32(lo##ID, k1l, QV, 1);              \
        hi##ID = vfmaq_laneq_f32(hi##ID, k1h, QV, 1);              \
        lo##ID = vfmaq_laneq_f32(lo##ID, k2l, QV, 2);              \
        hi##ID = vfmaq_laneq_f32(hi##ID, k2h, QV, 2);              \
        lo##ID = vfmaq_laneq_f32(lo##ID, k3l, QV, 3);              \
        hi##ID = vfmaq_laneq_f32(hi##ID, k3h, QV, 3);              \
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
#define FUSED_CPP_QKT_PACKK8_FMA_TAIL(ID, QROW)                    \
      do {                                                         \
        const float q = QROW[e];                                   \
        lo##ID = vfmaq_n_f32(lo##ID, kl, q);                       \
        hi##ID = vfmaq_n_f32(hi##ID, kh, q);                       \
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

#define FUSED_CPP_QKT_PACKK8_STORE_ROW(ID)                         \
    do {                                                           \
      vst1q_f32(scores_buf + (ID) * 8 + 0,                         \
                vmulq_f32(lo##ID, scale_v));                       \
      vst1q_f32(scores_buf + (ID) * 8 + 4,                         \
                vmulq_f32(hi##ID, scale_v));                       \
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
    qkt_packk8_tail_scalar(
        Q, q_row_stride, Kp, E, scale, scores_buf, 8, 8, 8);
#endif
  }

  static inline void qkt_8x4(
      const float* Q, int64_t q_row_stride,
      const float* Kp, int64_t,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride) {
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
#define FUSED_CPP_QKT_PACKK4_FMA_ROW(ID, QV)                       \
      do {                                                         \
        acc##ID = vfmaq_laneq_f32(acc##ID, k0, QV, 0);             \
        acc##ID = vfmaq_laneq_f32(acc##ID, k1, QV, 1);             \
        acc##ID = vfmaq_laneq_f32(acc##ID, k2, QV, 2);             \
        acc##ID = vfmaq_laneq_f32(acc##ID, k3, QV, 3);             \
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
#define FUSED_CPP_QKT_PACKK4_FMA_TAIL(ID, QROW)                    \
      do { acc##ID = vfmaq_n_f32(acc##ID, k, QROW[e]); } while (0)
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
#define FUSED_CPP_QKT_PACKK4_STORE_ROW(ID)                         \
    vst1q_f32(scores_buf + (ID) * scores_row_stride,               \
              vmulq_f32(acc##ID, scale_v))
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
    qkt_packk8_tail_scalar(
        Q, q_row_stride, Kp, E, scale, scores_buf, scores_row_stride, 8, 4);
#endif
  }

  static inline void qkt_tail(
      const float* Q, int64_t q_row_stride,
      const float* Kp, int64_t,
      int64_t E, float scale,
      float* scores_buf, int64_t scores_row_stride,
      int Lq, int Sk) {
    qkt_packk8_tail_scalar(
        Q, q_row_stride, Kp, E, scale, scores_buf, scores_row_stride, Lq, Sk);
  }

  static inline void pv_8x8(
      const float* P_hat, int64_t P_row_stride,
      const float* V, int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
    ::fused_cpp::sdpa_microkernels::gemm_pv_microkernel_8x8_fp32_pquad(
        P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
#else
    ::fused_cpp::sdpa_microkernels::gemm_pv_8x8(
        P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
#endif
  }

  static inline void pv_tail(
      const float* P_hat, int64_t P_row_stride,
      const float* V, int64_t v_row_stride,
      int64_t Sk,
      float* O, int64_t o_row_stride,
      int Lq, int Ev) {
    ::fused_cpp::sdpa_microkernels::gemm_pv_tail(
        P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride, Lq, Ev);
  }
};

template <bool kCausal>
void run_fp32_packk_path(
    const float* q_ptr,
    const float* k_packed_ptr,
    const float* v_packed_ptr,
    const SdpaParams& p,
    const TileSizes& ts,
    bool path_a,
    int total_threads) {
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

  FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kMain);
  if (path_a) {
    run_path_collapse3<MK_Fp32PackK8PQuad, float, true, false, kCausal>(
        q_ptr, k_packed_ptr, v_packed_ptr, p, ts,
        q_stride_b, q_stride_n, q_stride_l,
        k_packed_stride_b, k_packed_stride_n, k_packed_stride_s,
        v_packed_stride_b, v_packed_stride_n, v_packed_stride_s,
        v_evblock_stride,
        m_stride_b, m_stride_n, m_stride_l,
        o_stride_b, o_stride_n, o_stride_l);
  } else {
    const int64_t num_groups = std::max<int64_t>(1, total_threads);
    run_path_taskloop<MK_Fp32PackK8PQuad, float, true, false, kCausal>(
        q_ptr, k_packed_ptr, v_packed_ptr, p, ts, static_cast<int>(num_groups),
        q_stride_b, q_stride_n, q_stride_l,
        k_packed_stride_b, k_packed_stride_n, k_packed_stride_s,
        v_packed_stride_b, v_packed_stride_n, v_packed_stride_s,
        v_evblock_stride,
        m_stride_b, m_stride_n, m_stride_l,
        o_stride_b, o_stride_n, o_stride_l);
  }
}

const float* ptr_at(
    const float* base,
    int64_t nb0,
    int64_t nb1,
    int64_t nb2,
    int64_t nb3,
    int64_t i0,
    int64_t i1,
    int64_t i2,
    int64_t i3) {
  return reinterpret_cast<const float*>(
      reinterpret_cast<const char*>(base) +
      i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);
}

float* ptr_at(
    float* base,
    int64_t nb0,
    int64_t nb1,
    int64_t nb2,
    int64_t nb3,
    int64_t i0,
    int64_t i1,
    int64_t i2,
    int64_t i3) {
  return reinterpret_cast<float*>(
      reinterpret_cast<char*>(base) +
      i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);
}

}  // namespace

void sdpa_fp32_packqkv_pbf16pv(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const Config& cfg_in) {
  Config cfg = cfg_in;
  check_config(cfg);
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr) {
    throw std::invalid_argument("q/k/v/out must be non-null");
  }
  if (cfg.scale == 0.0f) {
    cfg.scale = 1.0f / std::sqrt(static_cast<float>(cfg.E));
  }

  SdpaParams p;
  p.B = cfg.B;
  p.N = cfg.N;
  p.L = cfg.L;
  p.S = cfg.S;
  p.E = cfg.E;
  p.Ev = cfg.Ev;
  p.scale_f = cfg.scale;
  p.neg_inf = kNegInf;
  p.causal_offset = cfg.causal_offset;
  p.is_causal = cfg.causal;
  p.dtype = SdpaDtype::kFloat32;
  p.q_ptr = q;
  p.k_ptr = k;
  p.v_ptr = v;
  p.mask_ptr = nullptr;
  p.out_ptr = out;

  const bool profile_on = ::fused_cpp::sdpa_profile::enabled();
  if (profile_on) {
    ::fused_cpp::sdpa_profile::reset();
  }
  const uint64_t profile_total_t0 =
      profile_on ? ::fused_cpp::sdpa_profile::now_ns() : 0;

  const int64_t eb = cfg.Ev / 8;
  AlignedVector<float> v_packed;
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kVAlloc);
    v_packed.resize(static_cast<size_t>(cfg.B * cfg.N * eb * cfg.S * 8));
  }
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kVPack);
    pack_v_to_evblock8<float>(v, v_packed.data(), cfg.B, cfg.N, cfg.S, cfg.Ev);
  }

  AlignedVector<float> k_packed;
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kKAlloc);
    k_packed.resize(static_cast<size_t>(cfg.B * cfg.N * (cfg.S / 8) * cfg.E * 8));
  }
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kKPack);
    pack_k_fp32_to_sblock8(k, k_packed.data(), cfg.B, cfg.N, cfg.S, cfg.E);
  }

  TileSizes ts = compute_tile_sizes_l3kv(
      cfg.B, cfg.N, cfg.S, cfg.L, cfg.E, cfg.Ev, sizeof(float));
  if (cfg.s_tile > 0) {
    ts.Sc_l2 = cfg.s_tile;
    ts.Sc_l3 = std::max<int64_t>(ts.Sc_l2, std::min<int64_t>(cfg.S, ts.Sc_l3));
    ts.Sc_l3 = (ts.Sc_l3 / ts.Sc_l2) * ts.Sc_l2;
    if (ts.Sc_l3 < ts.Sc_l2) {
      ts.Sc_l3 = ts.Sc_l2;
    }
    if (ts.Sc_l3 > cfg.S) {
      ts.Sc_l3 = cfg.S;
    }
  }

  const int64_t kv_bytes_per_bn =
      cfg.S * (cfg.E + cfg.Ev) * static_cast<int64_t>(sizeof(float));
  const auto& cache_bytes = effective_cache_bytes();
  const int64_t l3_budget =
      static_cast<int64_t>(cache_bytes[2] * FUSED_CPP_SDPA_L3_RATIO);
  const bool kv_fits_l3 = kv_bytes_per_bn <= l3_budget;
  const int total_threads =
#ifdef _OPENMP
      omp_get_max_threads();
#else
      1;
#endif
  const bool path_a = kv_fits_l3 || total_threads <= 1;

  if (cfg.causal) {
    run_fp32_packk_path<true>(
        q, k_packed.data(), v_packed.data(), p, ts, path_a, total_threads);
  } else {
    run_fp32_packk_path<false>(
        q, k_packed.data(), v_packed.data(), p, ts, path_a, total_threads);
  }

  if (profile_on) {
    ::fused_cpp::sdpa_profile::add(
        ::fused_cpp::sdpa_profile::Slot::kTotal,
        ::fused_cpp::sdpa_profile::now_ns() - profile_total_t0);
    ::fused_cpp::sdpa_profile::print_summary(
        "fp32_packqkv_standalone",
        MK_Fp32PackK8PQuad::kName,
        p,
        path_a ? "A" : "B");
  }
}

void reference_sdpa_fp32(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const Config& cfg_in) {
  Config cfg = cfg_in;
  check_config(cfg);
  if (cfg.scale == 0.0f) {
    cfg.scale = 1.0f / std::sqrt(static_cast<float>(cfg.E));
  }

  const int64_t q_b_stride = cfg.N * cfg.L * cfg.E;
  const int64_t q_n_stride = cfg.L * cfg.E;
  const int64_t k_b_stride = cfg.N * cfg.S * cfg.E;
  const int64_t k_n_stride = cfg.S * cfg.E;
  const int64_t v_b_stride = cfg.N * cfg.S * cfg.Ev;
  const int64_t v_n_stride = cfg.S * cfg.Ev;
  const int64_t out_b_stride = cfg.N * cfg.L * cfg.Ev;
  const int64_t out_n_stride = cfg.L * cfg.Ev;

  AlignedVector<float> scores(static_cast<size_t>(cfg.S));
  for (int64_t b = 0; b < cfg.B; ++b) {
    for (int64_t n = 0; n < cfg.N; ++n) {
      for (int64_t l = 0; l < cfg.L; ++l) {
        const float* q_row = q + b * q_b_stride + n * q_n_stride + l * cfg.E;
        const float* k_bn = k + b * k_b_stride + n * k_n_stride;
        const float* v_bn = v + b * v_b_stride + n * v_n_stride;
        float* out_row = out + b * out_b_stride + n * out_n_stride + l * cfg.Ev;

        const int64_t visible = cfg.causal
            ? clamp_i64(l + cfg.causal_offset + 1, 0, cfg.S)
            : cfg.S;

        float m = kNegInf;
        for (int64_t s = 0; s < visible; ++s) {
          const float* k_row = k_bn + s * cfg.E;
          float acc = 0.0f;
          for (int64_t e = 0; e < cfg.E; ++e) {
            acc += q_row[e] * k_row[e];
          }
          scores[static_cast<size_t>(s)] = acc * cfg.scale;
          m = std::max(m, scores[static_cast<size_t>(s)]);
        }

        std::fill(out_row, out_row + cfg.Ev, 0.0f);
        if (visible <= 0) {
          continue;
        }

        float denom = 0.0f;
        for (int64_t s = 0; s < visible; ++s) {
          const float prob = std::exp(scores[static_cast<size_t>(s)] - m);
          denom += prob;
          const float* v_row = v_bn + s * cfg.Ev;
          for (int64_t ev = 0; ev < cfg.Ev; ++ev) {
            out_row[ev] += prob * v_row[ev];
          }
        }
        const float inv = 1.0f / denom;
        for (int64_t ev = 0; ev < cfg.Ev; ++ev) {
          out_row[ev] *= inv;
        }
      }
    }
  }
}

double counted_gflops(const Config& cfg, double mean_ms) {
  int64_t active_per_head = 0;
  if (!cfg.causal) {
    active_per_head = cfg.L * cfg.S;
  } else {
    for (int64_t l = 0; l < cfg.L; ++l) {
      active_per_head += clamp_i64(l + cfg.causal_offset + 1, 0, cfg.S);
    }
  }
  const double active =
      static_cast<double>(cfg.B) * static_cast<double>(cfg.N) *
      static_cast<double>(active_per_head);
  const double flops_per_score =
      2.0 * static_cast<double>(cfg.E) +
      2.0 * static_cast<double>(cfg.Ev) + 5.0;
  return active * flops_per_score / (mean_ms * 1.0e6);
}

double checksum(const float* data, int64_t size) {
  double sum = 0.0;
  for (int64_t i = 0; i < size; ++i) {
    sum += static_cast<double>(data[i]);
  }
  return sum;
}

double max_abs_diff(const float* a, const float* b, int64_t size) {
  double m = 0.0;
  for (int64_t i = 0; i < size; ++i) {
    m = std::max(m, static_cast<double>(std::abs(a[i] - b[i])));
  }
  return m;
}

}  // namespace fp32_packqkv_sdpa

extern "C" FUSED_CPP_FP32_PACKQKV_API int
fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_contiguous(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int64_t B,
    int64_t N,
    int64_t L,
    int64_t S,
    int64_t E,
    int64_t Ev,
    int causal,
    float scale) {
  try {
    fp32_packqkv_sdpa::Config cfg;
    cfg.B = B;
    cfg.N = N;
    cfg.L = L;
    cfg.S = S;
    cfg.E = E;
    cfg.Ev = Ev;
    cfg.causal = causal != 0;
    cfg.scale = scale;
    cfg.causal_offset = S - L;
    fp32_packqkv_sdpa::sdpa_fp32_packqkv_pbf16pv(q, k, v, out, cfg);
    return 0;
  } catch (const std::invalid_argument&) {
    return 2;
  } catch (const std::exception&) {
    return 100;
  } catch (...) {
    return 101;
  }
}

extern "C" FUSED_CPP_FP32_PACKQKV_API int
fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int64_t B,
    int64_t H,
    int64_t L,
    int64_t S,
    int64_t D,
    int64_t DV,
    int64_t q_nb0,
    int64_t q_nb1,
    int64_t q_nb2,
    int64_t q_nb3,
    int64_t k_nb0,
    int64_t k_nb1,
    int64_t k_nb2,
    int64_t k_nb3,
    int64_t v_nb0,
    int64_t v_nb1,
    int64_t v_nb2,
    int64_t v_nb3,
    int64_t o_nb0,
    int64_t o_nb1,
    int64_t o_nb2,
    int64_t o_nb3,
    float scale) {
  try {
    if (q == nullptr || k == nullptr || v == nullptr || out == nullptr) {
      return 1;
    }
    if (B <= 0 || H <= 0 || L <= 0 || S <= 0 || D <= 0 || DV <= 0) {
      return 2;
    }
    if ((S % 8) != 0 || (DV % 8) != 0) {
      return 3;
    }

    fp32_packqkv_sdpa::AlignedVector<float> q_dense(
        static_cast<size_t>(B * H * L * D));
    fp32_packqkv_sdpa::AlignedVector<float> k_dense(
        static_cast<size_t>(B * H * S * D));
    fp32_packqkv_sdpa::AlignedVector<float> v_dense(
        static_cast<size_t>(B * H * S * DV));
    fp32_packqkv_sdpa::AlignedVector<float> o_dense(
        static_cast<size_t>(B * H * L * DV));

    for (int64_t b = 0; b < B; ++b) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t l = 0; l < L; ++l) {
          float* dst = q_dense.data() + ((b * H + h) * L + l) * D;
          const float* src = fp32_packqkv_sdpa::ptr_at(
              q, q_nb0, q_nb1, q_nb2, q_nb3, 0, l, h, b);
          std::memcpy(dst, src, static_cast<size_t>(D) * sizeof(float));
        }
        for (int64_t s = 0; s < S; ++s) {
          float* kd = k_dense.data() + ((b * H + h) * S + s) * D;
          const float* ks = fp32_packqkv_sdpa::ptr_at(
              k, k_nb0, k_nb1, k_nb2, k_nb3, 0, s, h, b);
          std::memcpy(kd, ks, static_cast<size_t>(D) * sizeof(float));

          float* vd = v_dense.data() + ((b * H + h) * S + s) * DV;
          const float* vs = fp32_packqkv_sdpa::ptr_at(
              v, v_nb0, v_nb1, v_nb2, v_nb3, 0, s, h, b);
          std::memcpy(vd, vs, static_cast<size_t>(DV) * sizeof(float));
        }
      }
    }

    fp32_packqkv_sdpa::Config cfg;
    cfg.B = B;
    cfg.N = H;
    cfg.L = L;
    cfg.S = S;
    cfg.E = D;
    cfg.Ev = DV;
    cfg.causal = false;
    cfg.scale = scale;
    cfg.causal_offset = S - L;
    fp32_packqkv_sdpa::sdpa_fp32_packqkv_pbf16pv(
        q_dense.data(), k_dense.data(), v_dense.data(), o_dense.data(), cfg);

    for (int64_t b = 0; b < B; ++b) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t l = 0; l < L; ++l) {
          const float* src = o_dense.data() + ((b * H + h) * L + l) * DV;
          float* dst = fp32_packqkv_sdpa::ptr_at(
              out, o_nb0, o_nb1, o_nb2, o_nb3, 0, h, l, b);
          std::memcpy(dst, src, static_cast<size_t>(DV) * sizeof(float));
        }
      }
    }

    return 0;
  } catch (const std::invalid_argument&) {
    return 2;
  } catch (const std::exception&) {
    return 100;
  } catch (...) {
    return 101;
  }
}
