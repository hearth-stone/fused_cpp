// SPDX-License-Identifier: Apache-2.0
#include "sparse_mla_tail_microkernels.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

#include "sdpa_microkernels/neon_cache_config.h"

#if FUSED_CPP_SDPA_CACHE_HAS_SVE &&     \
    (defined(__ARM_FEATURE_SVE_BF16) || \
     defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC))
#include <arm_sve.h>
#define FUSED_CPP_SPARSE_MLA_HAS_SVE_BF16 1
#else
#define FUSED_CPP_SPARSE_MLA_HAS_SVE_BF16 0
#endif

namespace fused_cpp::sparse_mla_tail_microkernels {
namespace {

constexpr int64_t kTile = 8;

constexpr uint16_t active_mask(std::array<uint8_t, kTile> valid_len) {
  uint16_t mask = 0;
  for (int64_t row_pair = 0; row_pair < kTile / 2; ++row_pair) {
    const int64_t pair_len =
        valid_len[2 * row_pair] > valid_len[2 * row_pair + 1] ? valid_len[2 * row_pair] : valid_len[2 * row_pair + 1];
    for (int64_t col_pair = 0; col_pair < kTile / 2; ++col_pair) {
      if (pair_len > 2 * col_pair) {
        mask |= static_cast<uint16_t>(uint16_t{1} << (row_pair * 4 + col_pair));
      }
    }
  }
  return mask;
}

constexpr uint64_t valid_mask(std::array<uint8_t, kTile> valid_len) {
  uint64_t mask = 0;
  for (int64_t row = 0; row < kTile; ++row) {
    for (int64_t col = 0; col < valid_len[row]; ++col) {
      mask |= uint64_t{1} << (row * kTile + col);
    }
  }
  return mask;
}

constexpr uint16_t kActiveCausal = active_mask({1, 2, 3, 4, 5, 6, 7, 8});
constexpr uint16_t kActiveCompressed0 = active_mask({0, 0, 0, 1, 1, 1, 1, 2});
constexpr uint16_t kActiveCompressed1 = active_mask({2, 2, 2, 3, 3, 3, 3, 4});
constexpr uint16_t kActiveCompressed2 = active_mask({4, 4, 4, 5, 5, 5, 5, 6});
constexpr uint16_t kActiveCompressed3 = active_mask({6, 6, 6, 7, 7, 7, 7, 8});
constexpr uint16_t kActiveUniform2 = active_mask({2, 2, 2, 2, 2, 2, 2, 2});
constexpr uint16_t kActiveUniform4 = active_mask({4, 4, 4, 4, 4, 4, 4, 4});
constexpr uint16_t kActiveUniform6 = active_mask({6, 6, 6, 6, 6, 6, 6, 6});

constexpr uint64_t kValidCausal = valid_mask({1, 2, 3, 4, 5, 6, 7, 8});
constexpr uint64_t kValidCompressed0 = valid_mask({0, 0, 0, 1, 1, 1, 1, 2});
constexpr uint64_t kValidCompressed1 = valid_mask({2, 2, 2, 3, 3, 3, 3, 4});
constexpr uint64_t kValidCompressed2 = valid_mask({4, 4, 4, 5, 5, 5, 5, 6});
constexpr uint64_t kValidCompressed3 = valid_mask({6, 6, 6, 7, 7, 7, 7, 8});
constexpr uint64_t kValidUniform2 = valid_mask({2, 2, 2, 2, 2, 2, 2, 2});
constexpr uint64_t kValidUniform4 = valid_mask({4, 4, 4, 4, 4, 4, 4, 4});
constexpr uint64_t kValidUniform6 = valid_mask({6, 6, 6, 6, 6, 6, 6, 6});

#if FUSED_CPP_SDPA_CACHE_HAS_NEON && FUSED_CPP_SDPA_CACHE_HAS_BFMMLA

template <uint16_t kMask, int kBit>
inline void qk_update(float32x4_t& acc, bfloat16x8_t q, bfloat16x8_t k) {
  if constexpr ((kMask & (uint16_t{1} << kBit)) != 0) {
    acc = vbfmmlaq_f32(acc, q, k);
  }
}

template <uint16_t kMask, int kBit, int kRowPair, int kColPair>
inline void qk_store(float32x4_t acc, float32x4_t scale, float* scores) {
  if constexpr ((kMask & (uint16_t{1} << kBit)) != 0) {
    const float32x4_t scaled = vmulq_f32(acc, scale);
    vst1_f32(scores + (2 * kRowPair) * kTile + 2 * kColPair, vget_low_f32(scaled));
    vst1_f32(scores + (2 * kRowPair + 1) * kTile + 2 * kColPair, vget_high_f32(scaled));
  }
}

template <uint16_t kMask>
void qkt_pruned_impl(const uint16_t* q_packed, const uint16_t* k_packed, int64_t d_qk, float scale, float* scores) {
  float32x4_t bm00 = vdupq_n_f32(0.0f), bm01 = vdupq_n_f32(0.0f);
  float32x4_t bm02 = vdupq_n_f32(0.0f), bm03 = vdupq_n_f32(0.0f);
  float32x4_t bm10 = vdupq_n_f32(0.0f), bm11 = vdupq_n_f32(0.0f);
  float32x4_t bm12 = vdupq_n_f32(0.0f), bm13 = vdupq_n_f32(0.0f);
  float32x4_t bm20 = vdupq_n_f32(0.0f), bm21 = vdupq_n_f32(0.0f);
  float32x4_t bm22 = vdupq_n_f32(0.0f), bm23 = vdupq_n_f32(0.0f);
  float32x4_t bm30 = vdupq_n_f32(0.0f), bm31 = vdupq_n_f32(0.0f);
  float32x4_t bm32 = vdupq_n_f32(0.0f), bm33 = vdupq_n_f32(0.0f);

  const auto update_block = [&](const uint16_t* q_ptr, const uint16_t* k_ptr) {
    const bfloat16x8_t a01 = vreinterpretq_bf16_u16(vld1q_u16(q_ptr));
    const bfloat16x8_t a23 = vreinterpretq_bf16_u16(vld1q_u16(q_ptr + 8));
    const bfloat16x8_t a45 = vreinterpretq_bf16_u16(vld1q_u16(q_ptr + 16));
    const bfloat16x8_t a67 = vreinterpretq_bf16_u16(vld1q_u16(q_ptr + 24));
    const bfloat16x8_t b01 = vreinterpretq_bf16_u16(vld1q_u16(k_ptr));
    const bfloat16x8_t b23 = vreinterpretq_bf16_u16(vld1q_u16(k_ptr + 8));
    const bfloat16x8_t b45 = vreinterpretq_bf16_u16(vld1q_u16(k_ptr + 16));
    const bfloat16x8_t b67 = vreinterpretq_bf16_u16(vld1q_u16(k_ptr + 24));
    qk_update<kMask, 0>(bm00, a01, b01);
    qk_update<kMask, 1>(bm01, a01, b23);
    qk_update<kMask, 2>(bm02, a01, b45);
    qk_update<kMask, 3>(bm03, a01, b67);
    qk_update<kMask, 4>(bm10, a23, b01);
    qk_update<kMask, 5>(bm11, a23, b23);
    qk_update<kMask, 6>(bm12, a23, b45);
    qk_update<kMask, 7>(bm13, a23, b67);
    qk_update<kMask, 8>(bm20, a45, b01);
    qk_update<kMask, 9>(bm21, a45, b23);
    qk_update<kMask, 10>(bm22, a45, b45);
    qk_update<kMask, 11>(bm23, a45, b67);
    qk_update<kMask, 12>(bm30, a67, b01);
    qk_update<kMask, 13>(bm31, a67, b23);
    qk_update<kMask, 14>(bm32, a67, b45);
    qk_update<kMask, 15>(bm33, a67, b67);
  };

  const uint16_t* q_ptr = q_packed;
  const uint16_t* k_ptr = k_packed;
  int64_t e = 0;
  for (; e + 8 <= d_qk; e += 8, q_ptr += 64, k_ptr += 64) {
    update_block(q_ptr, k_ptr);
    update_block(q_ptr + 32, k_ptr + 32);
  }
  for (; e < d_qk; e += 4, q_ptr += 32, k_ptr += 32) {
    update_block(q_ptr, k_ptr);
  }

  std::fill(scores, scores + kTile * kTile, 0.0f);
  const float32x4_t scale_vec = vdupq_n_f32(scale);
  qk_store<kMask, 0, 0, 0>(bm00, scale_vec, scores);
  qk_store<kMask, 1, 0, 1>(bm01, scale_vec, scores);
  qk_store<kMask, 2, 0, 2>(bm02, scale_vec, scores);
  qk_store<kMask, 3, 0, 3>(bm03, scale_vec, scores);
  qk_store<kMask, 4, 1, 0>(bm10, scale_vec, scores);
  qk_store<kMask, 5, 1, 1>(bm11, scale_vec, scores);
  qk_store<kMask, 6, 1, 2>(bm12, scale_vec, scores);
  qk_store<kMask, 7, 1, 3>(bm13, scale_vec, scores);
  qk_store<kMask, 8, 2, 0>(bm20, scale_vec, scores);
  qk_store<kMask, 9, 2, 1>(bm21, scale_vec, scores);
  qk_store<kMask, 10, 2, 2>(bm22, scale_vec, scores);
  qk_store<kMask, 11, 2, 3>(bm23, scale_vec, scores);
  qk_store<kMask, 12, 3, 0>(bm30, scale_vec, scores);
  qk_store<kMask, 13, 3, 1>(bm31, scale_vec, scores);
  qk_store<kMask, 14, 3, 2>(bm32, scale_vec, scores);
  qk_store<kMask, 15, 3, 3>(bm33, scale_vec, scores);
}

#endif

#if FUSED_CPP_SPARSE_MLA_HAS_SVE_BF16

#if defined(__GNUC__) || defined(__clang__)
#define FUSED_CPP_SPARSE_MLA_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define FUSED_CPP_SPARSE_MLA_ALWAYS_INLINE inline
#endif

template <uint64_t kMask, int kRow>
constexpr int valid_prefix_len() {
  int len = 0;
  while (len < kTile && (kMask & (uint64_t{1} << (kRow * kTile + len))) != 0) {
    ++len;
  }
  return len;
}

template <uint64_t kMask, int kRowPair, int kColPair>
constexpr bool qk_block_is_active() {
  constexpr int kLen0 = valid_prefix_len<kMask, 2 * kRowPair>();
  constexpr int kLen1 = valid_prefix_len<kMask, 2 * kRowPair + 1>();
  return (kLen0 > 2 * kColPair) || (kLen1 > 2 * kColPair);
}

template <bool kActive>
FUSED_CPP_SPARSE_MLA_ALWAYS_INLINE svfloat32_t qk_update_sve(svfloat32_t acc,
                                                             svbfloat16_t q,
                                                             svbfloat16_t k) {
  if constexpr (kActive) {
    return svbfmmla_f32(acc, q, k);
  }
  return acc;
}

FUSED_CPP_SPARSE_MLA_ALWAYS_INLINE svfloat32_t exp_poly5_sve(svbool_t pg,
                                                             svfloat32_t x) {
  x = svmin_n_f32_x(pg, x, 87.0f);
  x = svmax_n_f32_x(pg, x, -87.0f);

  const svfloat32_t fn =
      svrintn_f32_x(pg, svmul_n_f32_x(pg, x, 1.4426950408889634f));
  const svint32_t n = svcvt_s32_f32_x(pg, fn);
  const svfloat32_t r = svmls_n_f32_x(pg, x, fn, 0.6931471805599453f);

  svfloat32_t poly = svdup_f32(0.00833333f);
  poly = svmla_f32_x(pg, svdup_f32(0.04166666f), poly, r);
  poly = svmla_f32_x(pg, svdup_f32(0.16666666f), poly, r);
  poly = svmla_f32_x(pg, svdup_f32(0.5f), poly, r);
  poly = svmla_f32_x(pg, svdup_f32(1.0f), poly, r);
  poly = svmla_f32_x(pg, svdup_f32(1.0f), poly, r);
  return svscale_f32_x(pg, poly, n);
}

template <int kCount, int kLaneOffset>
FUSED_CPP_SPARSE_MLA_ALWAYS_INLINE float row_pair_max_sve(svfloat32_t scores) {
  static_assert(kCount >= 0 && kCount <= 2);
  if constexpr (kCount == 0) {
    return -std::numeric_limits<float>::infinity();
  }
  if constexpr (kLaneOffset != 0) {
    scores = svext_f32(scores, scores, kLaneOffset);
  }
  return svmaxv_f32(svwhilelt_b32(uint64_t{0}, static_cast<uint64_t>(kCount)),
                    scores);
}

template <int kCount0, int kCount1>
FUSED_CPP_SPARSE_MLA_ALWAYS_INLINE svbool_t valid_pair_lanes_sve(svbool_t pg4) {
  const svuint32_t valid =
      svdupq_n_u32(kCount0 > 0, kCount0 > 1, kCount1 > 0, kCount1 > 1);
  return svcmpne_n_u32(pg4, valid, 0);
}

FUSED_CPP_SPARSE_MLA_ALWAYS_INLINE svfloat32_t
load_bf16_f32_sve(svbool_t pg, const at::BFloat16* value) {
  const svuint32_t raw =
      svld1uh_u32(pg, reinterpret_cast<const uint16_t*>(value));
  return svreinterpret_f32_u32(svlsl_n_u32_x(pg, raw, 16));
}

FUSED_CPP_SPARSE_MLA_ALWAYS_INLINE svfloat32_t
round_to_bf16_f32_sve(svbool_t pg, svfloat32_t value) {
  const svbfloat16_t bf16 = svcvt_bf16_f32_x(pg, value);
  return svreinterpret_f32_u32(
      svlsl_n_u32_x(pg, svreinterpret_u32_bf16(bf16), 16));
}

template <uint64_t kMask, bool kPrune2x2, int kRow, int kKey, int kLane>
FUSED_CPP_SPARSE_MLA_ALWAYS_INLINE svfloat32_t
pv_update_fmla_sve(svbool_t pg, svfloat32_t output, svfloat32_t value,
                   svfloat32_t probabilities) {
  if constexpr (!kPrune2x2 ||
                (kMask & (uint64_t{1} << (kRow * kTile + kKey))) != 0) {
    return svmla_f32_x(pg, output, value, svdup_lane_f32(probabilities, kLane));
  }
  return output;
}

template <uint64_t kMask, bool kPrune2x2, int kRowPair>
FUSED_CPP_SPARSE_MLA_ALWAYS_INLINE void pv_row_pair_fmla_sve(
    svfloat32_t p01, svfloat32_t p23, svfloat32_t p45, svfloat32_t p67,
    svfloat32_t correction_pair, const at::BFloat16* v_packed,
    int64_t v_evblock_stride, int64_t d_v, float* output,
    int64_t output_stride) {
  constexpr int kRow0 = 2 * kRowPair;
  constexpr int kRow1 = kRow0 + 1;
  const svfloat32_t correction0 = svdup_lane_f32(correction_pair, 0);
  const svfloat32_t correction1 = svdup_lane_f32(correction_pair, 2);

  const int64_t vector_step = std::min<int64_t>(svcntw(), kTile);
  for (int64_t ev = 0; ev < d_v; ev += vector_step) {
    const int64_t block_offset = ev & (kTile - 1);
    const int64_t active = std::min<int64_t>(vector_step, kTile - block_offset);
    const svbool_t pg =
        svwhilelt_b32(uint64_t{0}, static_cast<uint64_t>(active));
    const at::BFloat16* v =
        v_packed + (ev / kTile) * v_evblock_stride + block_offset;
    svfloat32_t output0 = svmul_f32_x(
        pg, svld1_f32(pg, output + kRow0 * output_stride + ev), correction0);
    svfloat32_t output1 = svmul_f32_x(
        pg, svld1_f32(pg, output + kRow1 * output_stride + ev), correction1);

#define FUSED_CPP_PV_KEY(KEY, PROBS, LANE0, LANE1)                      \
  do {                                                                  \
    const svfloat32_t value = load_bf16_f32_sve(pg, v + (KEY) * kTile); \
    output0 = pv_update_fmla_sve<kMask, kPrune2x2, kRow0, KEY, LANE0>(  \
        pg, output0, value, PROBS);                                     \
    output1 = pv_update_fmla_sve<kMask, kPrune2x2, kRow1, KEY, LANE1>(  \
        pg, output1, value, PROBS);                                     \
  } while (false)
    FUSED_CPP_PV_KEY(0, p01, 0, 2);
    FUSED_CPP_PV_KEY(1, p01, 1, 3);
    FUSED_CPP_PV_KEY(2, p23, 0, 2);
    FUSED_CPP_PV_KEY(3, p23, 1, 3);
    FUSED_CPP_PV_KEY(4, p45, 0, 2);
    FUSED_CPP_PV_KEY(5, p45, 1, 3);
    FUSED_CPP_PV_KEY(6, p67, 0, 2);
    FUSED_CPP_PV_KEY(7, p67, 1, 3);
#undef FUSED_CPP_PV_KEY
    svst1_f32(pg, output + kRow0 * output_stride + ev, output0);
    svst1_f32(pg, output + kRow1 * output_stride + ev, output1);
  }
}

template <uint64_t kMask, bool kPrune2x2, int kRow, int kKey, int kLane,
          bool kOdd>
FUSED_CPP_SPARSE_MLA_ALWAYS_INLINE svfloat32_t pv_update_bfmlal_sve(
    svfloat32_t output, svbfloat16_t value, svbfloat16_t probabilities) {
  if constexpr (!kPrune2x2 ||
                (kMask & (uint64_t{1} << (kRow * kTile + kKey))) != 0) {
    if constexpr (kOdd) {
      return svbfmlalt_lane_f32(output, value, probabilities, kLane);
    }
    return svbfmlalb_lane_f32(output, value, probabilities, kLane);
  }
  return output;
}

template <uint64_t kMask, bool kPrune2x2, int kRowPair>
FUSED_CPP_SPARSE_MLA_ALWAYS_INLINE void pv_row_pair_bfmlal_sve(
    svfloat32_t p01, svfloat32_t p23, svfloat32_t p45, svfloat32_t p67,
    svfloat32_t correction_pair, const at::BFloat16* v_packed,
    int64_t v_evblock_stride, int64_t d_v, float* output,
    int64_t output_stride) {
  constexpr int kRow0 = 2 * kRowPair;
  constexpr int kRow1 = kRow0 + 1;
  const svbool_t pg4 = svwhilelt_b32(uint64_t{0}, uint64_t{4});
  const svbool_t pg8 = svwhilelt_b32(uint64_t{0}, uint64_t{8});
  const svbool_t pg_bf16 = svptrue_b16();
  const svbfloat16_t p01_bf16 = svcvt_bf16_f32_x(pg4, p01);
  const svbfloat16_t p23_bf16 = svcvt_bf16_f32_x(pg4, p23);
  const svbfloat16_t p45_bf16 = svcvt_bf16_f32_x(pg4, p45);
  const svbfloat16_t p67_bf16 = svcvt_bf16_f32_x(pg4, p67);
  const svfloat32_t correction0 = svdup_lane_f32(correction_pair, 0);
  const svfloat32_t correction1 = svdup_lane_f32(correction_pair, 2);

  for (int64_t ev = 0; ev < d_v; ev += kTile) {
    const at::BFloat16* v = v_packed + (ev / kTile) * v_evblock_stride;
    svfloat32_t even0;
    svfloat32_t odd0;
    svfloat32_t even1;
    svfloat32_t odd1;
    if (svcntw() >= kTile) {
      const svfloat32_t o0 =
          svld1_f32(pg8, output + kRow0 * output_stride + ev);
      const svfloat32_t o1 =
          svld1_f32(pg8, output + kRow1 * output_stride + ev);
      even0 = svmul_f32_x(pg8, svuzp1_f32(o0, o0), correction0);
      odd0 = svmul_f32_x(pg8, svuzp2_f32(o0, o0), correction0);
      even1 = svmul_f32_x(pg8, svuzp1_f32(o1, o1), correction1);
      odd1 = svmul_f32_x(pg8, svuzp2_f32(o1, o1), correction1);
    } else {
      const svfloat32_t o0_lo =
          svld1_f32(pg4, output + kRow0 * output_stride + ev);
      const svfloat32_t o0_hi =
          svld1_f32(pg4, output + kRow0 * output_stride + ev + 4);
      const svfloat32_t o1_lo =
          svld1_f32(pg4, output + kRow1 * output_stride + ev);
      const svfloat32_t o1_hi =
          svld1_f32(pg4, output + kRow1 * output_stride + ev + 4);
      even0 = svmul_f32_x(pg4, svuzp1_f32(o0_lo, o0_hi), correction0);
      odd0 = svmul_f32_x(pg4, svuzp2_f32(o0_lo, o0_hi), correction0);
      even1 = svmul_f32_x(pg4, svuzp1_f32(o1_lo, o1_hi), correction1);
      odd1 = svmul_f32_x(pg4, svuzp2_f32(o1_lo, o1_hi), correction1);
    }

#define FUSED_CPP_PV_BFMLAL_KEY(KEY, PROBS, LANE0, LANE1)                     \
  do {                                                                        \
    const svbfloat16_t value = svld1rq_bf16(                                  \
        pg_bf16, reinterpret_cast<const __bf16*>(v + (KEY) * kTile));         \
    even0 = pv_update_bfmlal_sve<kMask, kPrune2x2, kRow0, KEY, LANE0, false>( \
        even0, value, PROBS);                                                 \
    odd0 = pv_update_bfmlal_sve<kMask, kPrune2x2, kRow0, KEY, LANE0, true>(   \
        odd0, value, PROBS);                                                  \
    even1 = pv_update_bfmlal_sve<kMask, kPrune2x2, kRow1, KEY, LANE1, false>( \
        even1, value, PROBS);                                                 \
    odd1 = pv_update_bfmlal_sve<kMask, kPrune2x2, kRow1, KEY, LANE1, true>(   \
        odd1, value, PROBS);                                                  \
  } while (false)
    FUSED_CPP_PV_BFMLAL_KEY(0, p01_bf16, 0, 4);
    FUSED_CPP_PV_BFMLAL_KEY(1, p01_bf16, 2, 6);
    FUSED_CPP_PV_BFMLAL_KEY(2, p23_bf16, 0, 4);
    FUSED_CPP_PV_BFMLAL_KEY(3, p23_bf16, 2, 6);
    FUSED_CPP_PV_BFMLAL_KEY(4, p45_bf16, 0, 4);
    FUSED_CPP_PV_BFMLAL_KEY(5, p45_bf16, 2, 6);
    FUSED_CPP_PV_BFMLAL_KEY(6, p67_bf16, 0, 4);
    FUSED_CPP_PV_BFMLAL_KEY(7, p67_bf16, 2, 6);
#undef FUSED_CPP_PV_BFMLAL_KEY

    if (svcntw() >= kTile) {
      svst1_f32(pg8, output + kRow0 * output_stride + ev,
                svzip1_f32(even0, odd0));
      svst1_f32(pg8, output + kRow1 * output_stride + ev,
                svzip1_f32(even1, odd1));
    } else {
      svst1_f32(pg4, output + kRow0 * output_stride + ev,
                svzip1_f32(even0, odd0));
      svst1_f32(pg4, output + kRow0 * output_stride + ev + 4,
                svzip2_f32(even0, odd0));
      svst1_f32(pg4, output + kRow1 * output_stride + ev,
                svzip1_f32(even1, odd1));
      svst1_f32(pg4, output + kRow1 * output_stride + ev + 4,
                svzip2_f32(even1, odd1));
    }
  }
}

template <int kGroup>
FUSED_CPP_SPARSE_MLA_ALWAYS_INLINE svbfloat16_t pack_probability_k4_sve(
    svfloat32_t p01, svfloat32_t p23, svfloat32_t p45, svfloat32_t p67) {
  const svbool_t pg4 = svwhilelt_b32(uint64_t{0}, uint64_t{4});
  const svbfloat16_t lo = svcvt_bf16_f32_x(pg4, kGroup == 0 ? p01 : p45);
  const svbfloat16_t hi = svcvt_bf16_f32_x(pg4, kGroup == 0 ? p23 : p67);
  const svuint16_t lo_compact =
      svuzp1_u16(svreinterpret_u16_bf16(lo), svreinterpret_u16_bf16(lo));
  const svuint16_t hi_compact =
      svuzp1_u16(svreinterpret_u16_bf16(hi), svreinterpret_u16_bf16(hi));
  const svuint32_t packed = svzip1_u32(svreinterpret_u32_u16(lo_compact),
                                       svreinterpret_u32_u16(hi_compact));
  // BFMMLA operates independently on each 128-bit segment.  The narrowing
  // conversion above forms the required 2x4 probability matrix in the first
  // eight BF16 lanes; repeat that matrix into every scalable-vector segment.
  const svbool_t pg_bf16 = svptrue_b16();
  const svuint16_t repeat_first_segment =
      svand_n_u16_x(pg_bf16, svindex_u16(0, 1), 7);
  return svreinterpret_bf16_u16(
      svtbl_u16(svreinterpret_u16_u32(packed), repeat_first_segment));
}

template <uint64_t kMask, bool kPrune2x2, int kRowPair>
FUSED_CPP_SPARSE_MLA_ALWAYS_INLINE void pv_row_pair_bfmmla_sve(
    svfloat32_t p01, svfloat32_t p23, svfloat32_t p45, svfloat32_t p67,
    svfloat32_t correction_pair, const at::BFloat16* v_bfmmla_packed,
    int64_t v_bfmmla_evblock_stride, int64_t d_v, float* output,
    int64_t output_stride) {
  constexpr int kRow0 = 2 * kRowPair;
  constexpr int kRow1 = kRow0 + 1;
  const svbool_t pg_bf16 = svptrue_b16();
  const svbfloat16_t probability0 =
      pack_probability_k4_sve<0>(p01, p23, p45, p67);
  const svbfloat16_t probability1 =
      pack_probability_k4_sve<1>(p01, p23, p45, p67);
  const svfloat32_t correction0 = svdup_lane_f32(correction_pair, 0);
  const svfloat32_t correction1 = svdup_lane_f32(correction_pair, 2);
  const int64_t segment_count = std::min<int64_t>(svcntw() / 4, 2);

  for (int64_t ev = 0; ev < d_v; ev += kTile) {
    const at::BFloat16* v =
        v_bfmmla_packed + (ev / kTile) * v_bfmmla_evblock_stride;
    for (int64_t col_pair = 0; col_pair < 4; col_pair += segment_count) {
      const svbool_t pg_out =
          svwhilelt_b32(uint64_t{0}, static_cast<uint64_t>(segment_count * 2));
      svfloat32_t output0 = svmul_f32_x(
          pg_out,
          svld1_f32(pg_out, output + kRow0 * output_stride + ev + col_pair * 2),
          correction0);
      svfloat32_t output1 = svmul_f32_x(
          pg_out,
          svld1_f32(pg_out, output + kRow1 * output_stride + ev + col_pair * 2),
          correction1);
      // Each u64 lane is one adjacent FP32 output-column pair.  ZIP1 packs
      // [row0 pair 0, row1 pair 0, row0 pair 1, row1 pair 1], which is the
      // 2x2 accumulator layout expected independently by each 128-bit BFMMLA
      // segment.  TRN1 would skip pair 1 at SVL=256 because it selects even
      // u64 lanes across the full scalable vector.
      svfloat32_t acc = svreinterpret_f32_u64(svzip1_u64(
          svreinterpret_u64_f32(output0), svreinterpret_u64_f32(output1)));
      if constexpr (!kPrune2x2 || qk_block_is_active<kMask, kRowPair, 0>() ||
                    qk_block_is_active<kMask, kRowPair, 1>()) {
        const svbfloat16_t value0 = svld1_bf16(
            pg_bf16, reinterpret_cast<const __bf16*>(v + col_pair * 8));
        acc = svbfmmla_f32(acc, probability0, value0);
      }
      if constexpr (!kPrune2x2 || qk_block_is_active<kMask, kRowPair, 2>() ||
                    qk_block_is_active<kMask, kRowPair, 3>()) {
        const svbfloat16_t value1 = svld1_bf16(
            pg_bf16, reinterpret_cast<const __bf16*>(v + (4 + col_pair) * 8));
        acc = svbfmmla_f32(acc, probability1, value1);
      }
      const svuint64_t acc64 = svreinterpret_u64_f32(acc);
      svst1_f32(pg_out, output + kRow0 * output_stride + ev + col_pair * 2,
                svreinterpret_f32_u64(svuzp1_u64(acc64, acc64)));
      svst1_f32(pg_out, output + kRow1 * output_stride + ev + col_pair * 2,
                svreinterpret_f32_u64(svuzp2_u64(acc64, acc64)));
    }
  }
}

template <uint64_t kMask, bool kPrune2x2, SparseMlaPvBackend kPvBackend>
__attribute__((noinline)) void online_softmax_pv_sve_impl(
    const uint16_t* q_packed, const uint16_t* k_packed,
    const at::BFloat16* v_packed, int64_t v_evblock_stride,
    const at::BFloat16* v_bfmmla_packed, int64_t v_bfmmla_evblock_stride,
    int64_t d_qk, int64_t d_v, float scale, float* running_max,
    float* running_sum, float* output, int64_t output_stride) {
  const svbool_t pg_bf16 = svptrue_b16();
  const svbool_t pg4 = svwhilelt_b32(uint64_t{0}, uint64_t{4});
  const svbool_t pg2 = svwhilelt_b32(uint64_t{0}, uint64_t{2});

  svfloat32_t bm00 = svdup_f32(0.0f), bm01 = svdup_f32(0.0f);
  svfloat32_t bm02 = svdup_f32(0.0f), bm03 = svdup_f32(0.0f);
  svfloat32_t bm10 = svdup_f32(0.0f), bm11 = svdup_f32(0.0f);
  svfloat32_t bm12 = svdup_f32(0.0f), bm13 = svdup_f32(0.0f);
  svfloat32_t bm20 = svdup_f32(0.0f), bm21 = svdup_f32(0.0f);
  svfloat32_t bm22 = svdup_f32(0.0f), bm23 = svdup_f32(0.0f);
  svfloat32_t bm30 = svdup_f32(0.0f), bm31 = svdup_f32(0.0f);
  svfloat32_t bm32 = svdup_f32(0.0f), bm33 = svdup_f32(0.0f);

  const uint16_t* q_ptr = q_packed;
  const uint16_t* k_ptr = k_packed;
  for (int64_t e = 0; e < d_qk; e += 4, q_ptr += 32, k_ptr += 32) {
    const svbfloat16_t q01 =
        svld1rq_bf16(pg_bf16, reinterpret_cast<const __bf16*>(q_ptr));
    const svbfloat16_t q23 =
        svld1rq_bf16(pg_bf16, reinterpret_cast<const __bf16*>(q_ptr + 8));
    const svbfloat16_t q45 =
        svld1rq_bf16(pg_bf16, reinterpret_cast<const __bf16*>(q_ptr + 16));
    const svbfloat16_t q67 =
        svld1rq_bf16(pg_bf16, reinterpret_cast<const __bf16*>(q_ptr + 24));
    const svbfloat16_t k01 =
        svld1rq_bf16(pg_bf16, reinterpret_cast<const __bf16*>(k_ptr));
    const svbfloat16_t k23 =
        svld1rq_bf16(pg_bf16, reinterpret_cast<const __bf16*>(k_ptr + 8));
    const svbfloat16_t k45 =
        svld1rq_bf16(pg_bf16, reinterpret_cast<const __bf16*>(k_ptr + 16));
    const svbfloat16_t k67 =
        svld1rq_bf16(pg_bf16, reinterpret_cast<const __bf16*>(k_ptr + 24));

#define FUSED_CPP_QK_UPDATE(ACC, ROW_PAIR, COL_PAIR, Q, K) \
  ACC = qk_update_sve < !kPrune2x2 ||                      \
        qk_block_is_active<kMask, ROW_PAIR, COL_PAIR>() > (ACC, Q, K)
    FUSED_CPP_QK_UPDATE(bm00, 0, 0, q01, k01);
    FUSED_CPP_QK_UPDATE(bm01, 0, 1, q01, k23);
    FUSED_CPP_QK_UPDATE(bm02, 0, 2, q01, k45);
    FUSED_CPP_QK_UPDATE(bm03, 0, 3, q01, k67);
    FUSED_CPP_QK_UPDATE(bm10, 1, 0, q23, k01);
    FUSED_CPP_QK_UPDATE(bm11, 1, 1, q23, k23);
    FUSED_CPP_QK_UPDATE(bm12, 1, 2, q23, k45);
    FUSED_CPP_QK_UPDATE(bm13, 1, 3, q23, k67);
    FUSED_CPP_QK_UPDATE(bm20, 2, 0, q45, k01);
    FUSED_CPP_QK_UPDATE(bm21, 2, 1, q45, k23);
    FUSED_CPP_QK_UPDATE(bm22, 2, 2, q45, k45);
    FUSED_CPP_QK_UPDATE(bm23, 2, 3, q45, k67);
    FUSED_CPP_QK_UPDATE(bm30, 3, 0, q67, k01);
    FUSED_CPP_QK_UPDATE(bm31, 3, 1, q67, k23);
    FUSED_CPP_QK_UPDATE(bm32, 3, 2, q67, k45);
    FUSED_CPP_QK_UPDATE(bm33, 3, 3, q67, k67);
#undef FUSED_CPP_QK_UPDATE
  }

#define FUSED_CPP_SOFTMAX_ROW_PAIR(ROW_PAIR, S0, S1, S2, S3, CORR)             \
  do {                                                                         \
    constexpr int kRow0 = 2 * (ROW_PAIR);                                      \
    constexpr int kRow1 = kRow0 + 1;                                           \
    constexpr int kLen0 = valid_prefix_len<kMask, kRow0>();                    \
    constexpr int kLen1 = valid_prefix_len<kMask, kRow1>();                    \
    S0 = svmul_n_f32_x(pg4, S0, scale);                                        \
    S1 = svmul_n_f32_x(pg4, S1, scale);                                        \
    S2 = svmul_n_f32_x(pg4, S2, scale);                                        \
    S3 = svmul_n_f32_x(pg4, S3, scale);                                        \
    if constexpr (kLen0 == 0 && kLen1 == 0) {                                  \
      S0 = svdup_f32(0.0f);                                                    \
      S1 = svdup_f32(0.0f);                                                    \
      S2 = svdup_f32(0.0f);                                                    \
      S3 = svdup_f32(0.0f);                                                    \
      CORR = svdup_f32(1.0f);                                                  \
    } else {                                                                   \
      float tile_max0 = -std::numeric_limits<float>::infinity();               \
      float tile_max1 = -std::numeric_limits<float>::infinity();               \
      if constexpr (kLen0 > 0) {                                               \
        tile_max0 = std::max(                                                  \
            tile_max0, row_pair_max_sve<(kLen0 > 2 ? 2 : kLen0), 0>(S0));      \
        tile_max0 = std::max(                                                  \
            tile_max0,                                                         \
            row_pair_max_sve<(kLen0 > 4 ? 2 : (kLen0 > 2 ? kLen0 - 2 : 0)),    \
                             0>(S1));                                          \
        tile_max0 = std::max(                                                  \
            tile_max0,                                                         \
            row_pair_max_sve<(kLen0 > 6 ? 2 : (kLen0 > 4 ? kLen0 - 4 : 0)),    \
                             0>(S2));                                          \
        tile_max0 = std::max(                                                  \
            tile_max0, row_pair_max_sve<(kLen0 > 6 ? kLen0 - 6 : 0), 0>(S3));  \
      }                                                                        \
      if constexpr (kLen1 > 0) {                                               \
        tile_max1 = std::max(                                                  \
            tile_max1, row_pair_max_sve<(kLen1 > 2 ? 2 : kLen1), 2>(S0));      \
        tile_max1 = std::max(                                                  \
            tile_max1,                                                         \
            row_pair_max_sve<(kLen1 > 4 ? 2 : (kLen1 > 2 ? kLen1 - 2 : 0)),    \
                             2>(S1));                                          \
        tile_max1 = std::max(                                                  \
            tile_max1,                                                         \
            row_pair_max_sve<(kLen1 > 6 ? 2 : (kLen1 > 4 ? kLen1 - 4 : 0)),    \
                             2>(S2));                                          \
        tile_max1 = std::max(                                                  \
            tile_max1, row_pair_max_sve<(kLen1 > 6 ? kLen1 - 6 : 0), 2>(S3));  \
      }                                                                        \
      const float old_max0 = kLen0 > 0 ? running_max[kRow0] : 0.0f;            \
      const float old_max1 = kLen1 > 0 ? running_max[kRow1] : 0.0f;            \
      const float new_max0 = kLen0 > 0 ? std::max(old_max0, tile_max0) : 0.0f; \
      const float new_max1 = kLen1 > 0 ? std::max(old_max1, tile_max1) : 0.0f; \
      const svfloat32_t old_max =                                              \
          svdupq_n_f32(old_max0, old_max0, old_max1, old_max1);                \
      const svfloat32_t new_max =                                              \
          svdupq_n_f32(new_max0, new_max0, new_max1, new_max1);                \
      CORR = exp_poly5_sve(pg4, svsub_f32_x(pg4, old_max, new_max));           \
      S0 = exp_poly5_sve(pg4, svsub_f32_x(pg4, S0, new_max));                  \
      S1 = exp_poly5_sve(pg4, svsub_f32_x(pg4, S1, new_max));                  \
      S2 = exp_poly5_sve(pg4, svsub_f32_x(pg4, S2, new_max));                  \
      S3 = exp_poly5_sve(pg4, svsub_f32_x(pg4, S3, new_max));                  \
      S0 = svsel_f32(valid_pair_lanes_sve<(kLen0 > 2 ? 2 : kLen0),             \
                                          (kLen1 > 2 ? 2 : kLen1)>(pg4),       \
                     S0, svdup_f32(0.0f));                                     \
      S1 = svsel_f32(                                                          \
          valid_pair_lanes_sve<(kLen0 > 4 ? 2 : (kLen0 > 2 ? kLen0 - 2 : 0)),  \
                               (kLen1 > 4 ? 2 : (kLen1 > 2 ? kLen1 - 2 : 0))>( \
              pg4),                                                            \
          S1, svdup_f32(0.0f));                                                \
      S2 = svsel_f32(                                                          \
          valid_pair_lanes_sve<(kLen0 > 6 ? 2 : (kLen0 > 4 ? kLen0 - 4 : 0)),  \
                               (kLen1 > 6 ? 2 : (kLen1 > 4 ? kLen1 - 4 : 0))>( \
              pg4),                                                            \
          S2, svdup_f32(0.0f));                                                \
      S3 = svsel_f32(valid_pair_lanes_sve<(kLen0 > 6 ? kLen0 - 6 : 0),         \
                                          (kLen1 > 6 ? kLen1 - 6 : 0)>(pg4),   \
                     S3, svdup_f32(0.0f));                                     \
      const svfloat32_t pair_sum = svadd_f32_x(pg4, svadd_f32_x(pg4, S0, S1),  \
                                               svadd_f32_x(pg4, S2, S3));      \
      S0 = round_to_bf16_f32_sve(pg4, S0);                                     \
      S1 = round_to_bf16_f32_sve(pg4, S1);                                     \
      S2 = round_to_bf16_f32_sve(pg4, S2);                                     \
      S3 = round_to_bf16_f32_sve(pg4, S3);                                     \
      if constexpr (kLen0 > 0) {                                               \
        const float correction =                                               \
            svlastb_f32(svwhilelt_b32(uint64_t{0}, uint64_t{1}), CORR);        \
        running_sum[kRow0] =                                                   \
            running_sum[kRow0] * correction + svaddv_f32(pg2, pair_sum);       \
        running_max[kRow0] = new_max0;                                         \
      }                                                                        \
      if constexpr (kLen1 > 0) {                                               \
        const svfloat32_t shifted_sum = svext_f32(pair_sum, pair_sum, 2);      \
        const svfloat32_t shifted_corr = svext_f32(CORR, CORR, 2);             \
        const float correction = svlastb_f32(                                  \
            svwhilelt_b32(uint64_t{0}, uint64_t{1}), shifted_corr);            \
        running_sum[kRow1] =                                                   \
            running_sum[kRow1] * correction + svaddv_f32(pg2, shifted_sum);    \
        running_max[kRow1] = new_max1;                                         \
      }                                                                        \
    }                                                                          \
  } while (false)

  svfloat32_t correction_pair = svdup_f32(1.0f);
  FUSED_CPP_SOFTMAX_ROW_PAIR(0, bm00, bm01, bm02, bm03, correction_pair);
#define FUSED_CPP_PV_ROW_PAIR(ROW_PAIR, P0, P1, P2, P3, CORR)                  \
  do {                                                                         \
    if constexpr (kPvBackend == SparseMlaPvBackend::kFmla) {                   \
      pv_row_pair_fmla_sve<kMask, kPrune2x2, ROW_PAIR>(                        \
          P0, P1, P2, P3, CORR, v_packed, v_evblock_stride, d_v, output,       \
          output_stride);                                                      \
    } else if constexpr (kPvBackend == SparseMlaPvBackend::kBfmlal) {          \
      pv_row_pair_bfmlal_sve<kMask, kPrune2x2, ROW_PAIR>(                      \
          P0, P1, P2, P3, CORR, v_packed, v_evblock_stride, d_v, output,       \
          output_stride);                                                      \
    } else {                                                                   \
      pv_row_pair_bfmmla_sve<kMask, kPrune2x2, ROW_PAIR>(                      \
          P0, P1, P2, P3, CORR, v_bfmmla_packed, v_bfmmla_evblock_stride, d_v, \
          output, output_stride);                                              \
    }                                                                          \
  } while (false)
  FUSED_CPP_PV_ROW_PAIR(0, bm00, bm01, bm02, bm03, correction_pair);

  correction_pair = svdup_f32(1.0f);
  FUSED_CPP_SOFTMAX_ROW_PAIR(1, bm10, bm11, bm12, bm13, correction_pair);
  FUSED_CPP_PV_ROW_PAIR(1, bm10, bm11, bm12, bm13, correction_pair);

  correction_pair = svdup_f32(1.0f);
  FUSED_CPP_SOFTMAX_ROW_PAIR(2, bm20, bm21, bm22, bm23, correction_pair);
  FUSED_CPP_PV_ROW_PAIR(2, bm20, bm21, bm22, bm23, correction_pair);

  correction_pair = svdup_f32(1.0f);
  FUSED_CPP_SOFTMAX_ROW_PAIR(3, bm30, bm31, bm32, bm33, correction_pair);
  FUSED_CPP_PV_ROW_PAIR(3, bm30, bm31, bm32, bm33, correction_pair);
#undef FUSED_CPP_PV_ROW_PAIR
#undef FUSED_CPP_SOFTMAX_ROW_PAIR
}

template <bool kPrune2x2, SparseMlaPvBackend kPvBackend>
bool dispatch_online_softmax_pv_sve(
    const uint16_t* q_packed, const uint16_t* k_packed,
    const at::BFloat16* v_packed, int64_t v_evblock_stride, int64_t d_qk,
    const at::BFloat16* v_bfmmla_packed, int64_t v_bfmmla_evblock_stride,
    int64_t d_v, float scale, uint64_t mask, float* running_max,
    float* running_sum, float* output, int64_t output_stride) {
#define FUSED_CPP_DISPATCH_MASK(MASK)                                        \
  case MASK:                                                                 \
    online_softmax_pv_sve_impl<MASK, kPrune2x2, kPvBackend>(                 \
        q_packed, k_packed, v_packed, v_evblock_stride, v_bfmmla_packed,     \
        v_bfmmla_evblock_stride, d_qk, d_v, scale, running_max, running_sum, \
        output, output_stride);                                              \
    return true
  switch (mask) {
    FUSED_CPP_DISPATCH_MASK(kValidCausal);
    FUSED_CPP_DISPATCH_MASK(kValidCompressed0);
    FUSED_CPP_DISPATCH_MASK(kValidCompressed1);
    FUSED_CPP_DISPATCH_MASK(kValidCompressed2);
    FUSED_CPP_DISPATCH_MASK(kValidCompressed3);
    FUSED_CPP_DISPATCH_MASK(kValidUniform2);
    FUSED_CPP_DISPATCH_MASK(kValidUniform4);
    FUSED_CPP_DISPATCH_MASK(kValidUniform6);
    default:
      return false;
  }
#undef FUSED_CPP_DISPATCH_MASK
}

#undef FUSED_CPP_SPARSE_MLA_ALWAYS_INLINE

#endif

#if FUSED_CPP_SDPA_CACHE_HAS_NEON && FUSED_CPP_SDPA_CACHE_HAS_BF16

template <uint64_t kMask, int kRow, int kCol, int kLane>
inline void pv_update(float32x4_t& even, float32x4_t& odd, bfloat16x8_t v, bfloat16x4_t p) {
  if constexpr ((kMask & (uint64_t{1} << (kRow * kTile + kCol))) != 0) {
    even = vbfmlalbq_lane_f32(even, v, p, kLane);
    odd = vbfmlaltq_lane_f32(odd, v, p, kLane);
  }
}

template <uint64_t kMask>
void pv_pruned_one(const at::BFloat16* p_bf16, const at::BFloat16* v, float* output, int64_t output_stride) {
  float32x4_t o0_lo = vld1q_f32(output), o0_hi = vld1q_f32(output + 4);
  float32x4_t o1_lo = vld1q_f32(output + output_stride);
  float32x4_t o1_hi = vld1q_f32(output + output_stride + 4);
  float32x4_t o2_lo = vld1q_f32(output + 2 * output_stride);
  float32x4_t o2_hi = vld1q_f32(output + 2 * output_stride + 4);
  float32x4_t o3_lo = vld1q_f32(output + 3 * output_stride);
  float32x4_t o3_hi = vld1q_f32(output + 3 * output_stride + 4);
  float32x4_t o4_lo = vld1q_f32(output + 4 * output_stride);
  float32x4_t o4_hi = vld1q_f32(output + 4 * output_stride + 4);
  float32x4_t o5_lo = vld1q_f32(output + 5 * output_stride);
  float32x4_t o5_hi = vld1q_f32(output + 5 * output_stride + 4);
  float32x4_t o6_lo = vld1q_f32(output + 6 * output_stride);
  float32x4_t o6_hi = vld1q_f32(output + 6 * output_stride + 4);
  float32x4_t o7_lo = vld1q_f32(output + 7 * output_stride);
  float32x4_t o7_hi = vld1q_f32(output + 7 * output_stride + 4);

  float32x4_t e0 = vuzp1q_f32(o0_lo, o0_hi), d0 = vuzp2q_f32(o0_lo, o0_hi);
  float32x4_t e1 = vuzp1q_f32(o1_lo, o1_hi), d1 = vuzp2q_f32(o1_lo, o1_hi);
  float32x4_t e2 = vuzp1q_f32(o2_lo, o2_hi), d2 = vuzp2q_f32(o2_lo, o2_hi);
  float32x4_t e3 = vuzp1q_f32(o3_lo, o3_hi), d3 = vuzp2q_f32(o3_lo, o3_hi);
  float32x4_t e4 = vuzp1q_f32(o4_lo, o4_hi), d4 = vuzp2q_f32(o4_lo, o4_hi);
  float32x4_t e5 = vuzp1q_f32(o5_lo, o5_hi), d5 = vuzp2q_f32(o5_lo, o5_hi);
  float32x4_t e6 = vuzp1q_f32(o6_lo, o6_hi), d6 = vuzp2q_f32(o6_lo, o6_hi);
  float32x4_t e7 = vuzp1q_f32(o7_lo, o7_hi), d7 = vuzp2q_f32(o7_lo, o7_hi);

  const auto* p_ptr = reinterpret_cast<const bfloat16_t*>(p_bf16);
  const auto* v_ptr = reinterpret_cast<const bfloat16_t*>(v);
  {
    const bfloat16x4_t p0 = vld1_bf16(p_ptr);
    const bfloat16x4_t p1 = vld1_bf16(p_ptr + kTile);
    const bfloat16x4_t p2 = vld1_bf16(p_ptr + 2 * kTile);
    const bfloat16x4_t p3 = vld1_bf16(p_ptr + 3 * kTile);
    const bfloat16x4_t p4 = vld1_bf16(p_ptr + 4 * kTile);
    const bfloat16x4_t p5 = vld1_bf16(p_ptr + 5 * kTile);
    const bfloat16x4_t p6 = vld1_bf16(p_ptr + 6 * kTile);
    const bfloat16x4_t p7 = vld1_bf16(p_ptr + 7 * kTile);
    const bfloat16x8_t v0 = vld1q_bf16(v_ptr);
    const bfloat16x8_t v1 = vld1q_bf16(v_ptr + kTile);
    const bfloat16x8_t v2 = vld1q_bf16(v_ptr + 2 * kTile);
    const bfloat16x8_t v3 = vld1q_bf16(v_ptr + 3 * kTile);
#define FUSED_CPP_PV_COL(COL, LANE, VEC)           \
  pv_update<kMask, 0, COL, LANE>(e0, d0, VEC, p0); \
  pv_update<kMask, 1, COL, LANE>(e1, d1, VEC, p1); \
  pv_update<kMask, 2, COL, LANE>(e2, d2, VEC, p2); \
  pv_update<kMask, 3, COL, LANE>(e3, d3, VEC, p3); \
  pv_update<kMask, 4, COL, LANE>(e4, d4, VEC, p4); \
  pv_update<kMask, 5, COL, LANE>(e5, d5, VEC, p5); \
  pv_update<kMask, 6, COL, LANE>(e6, d6, VEC, p6); \
  pv_update<kMask, 7, COL, LANE>(e7, d7, VEC, p7)
    FUSED_CPP_PV_COL(0, 0, v0);
    FUSED_CPP_PV_COL(1, 1, v1);
    FUSED_CPP_PV_COL(2, 2, v2);
    FUSED_CPP_PV_COL(3, 3, v3);
#undef FUSED_CPP_PV_COL
  }
  {
    const bfloat16x4_t p0 = vld1_bf16(p_ptr + 4);
    const bfloat16x4_t p1 = vld1_bf16(p_ptr + kTile + 4);
    const bfloat16x4_t p2 = vld1_bf16(p_ptr + 2 * kTile + 4);
    const bfloat16x4_t p3 = vld1_bf16(p_ptr + 3 * kTile + 4);
    const bfloat16x4_t p4 = vld1_bf16(p_ptr + 4 * kTile + 4);
    const bfloat16x4_t p5 = vld1_bf16(p_ptr + 5 * kTile + 4);
    const bfloat16x4_t p6 = vld1_bf16(p_ptr + 6 * kTile + 4);
    const bfloat16x4_t p7 = vld1_bf16(p_ptr + 7 * kTile + 4);
    const bfloat16x8_t v4 = vld1q_bf16(v_ptr + 4 * kTile);
    const bfloat16x8_t v5 = vld1q_bf16(v_ptr + 5 * kTile);
    const bfloat16x8_t v6 = vld1q_bf16(v_ptr + 6 * kTile);
    const bfloat16x8_t v7 = vld1q_bf16(v_ptr + 7 * kTile);
#define FUSED_CPP_PV_COL(COL, LANE, VEC)           \
  pv_update<kMask, 0, COL, LANE>(e0, d0, VEC, p0); \
  pv_update<kMask, 1, COL, LANE>(e1, d1, VEC, p1); \
  pv_update<kMask, 2, COL, LANE>(e2, d2, VEC, p2); \
  pv_update<kMask, 3, COL, LANE>(e3, d3, VEC, p3); \
  pv_update<kMask, 4, COL, LANE>(e4, d4, VEC, p4); \
  pv_update<kMask, 5, COL, LANE>(e5, d5, VEC, p5); \
  pv_update<kMask, 6, COL, LANE>(e6, d6, VEC, p6); \
  pv_update<kMask, 7, COL, LANE>(e7, d7, VEC, p7)
    FUSED_CPP_PV_COL(4, 0, v4);
    FUSED_CPP_PV_COL(5, 1, v5);
    FUSED_CPP_PV_COL(6, 2, v6);
    FUSED_CPP_PV_COL(7, 3, v7);
#undef FUSED_CPP_PV_COL
  }

  o0_lo = vzip1q_f32(e0, d0);
  o0_hi = vzip2q_f32(e0, d0);
  o1_lo = vzip1q_f32(e1, d1);
  o1_hi = vzip2q_f32(e1, d1);
  o2_lo = vzip1q_f32(e2, d2);
  o2_hi = vzip2q_f32(e2, d2);
  o3_lo = vzip1q_f32(e3, d3);
  o3_hi = vzip2q_f32(e3, d3);
  o4_lo = vzip1q_f32(e4, d4);
  o4_hi = vzip2q_f32(e4, d4);
  o5_lo = vzip1q_f32(e5, d5);
  o5_hi = vzip2q_f32(e5, d5);
  o6_lo = vzip1q_f32(e6, d6);
  o6_hi = vzip2q_f32(e6, d6);
  o7_lo = vzip1q_f32(e7, d7);
  o7_hi = vzip2q_f32(e7, d7);

  vst1q_f32(output, o0_lo);
  vst1q_f32(output + 4, o0_hi);
  vst1q_f32(output + output_stride, o1_lo);
  vst1q_f32(output + output_stride + 4, o1_hi);
  vst1q_f32(output + 2 * output_stride, o2_lo);
  vst1q_f32(output + 2 * output_stride + 4, o2_hi);
  vst1q_f32(output + 3 * output_stride, o3_lo);
  vst1q_f32(output + 3 * output_stride + 4, o3_hi);
  vst1q_f32(output + 4 * output_stride, o4_lo);
  vst1q_f32(output + 4 * output_stride + 4, o4_hi);
  vst1q_f32(output + 5 * output_stride, o5_lo);
  vst1q_f32(output + 5 * output_stride + 4, o5_hi);
  vst1q_f32(output + 6 * output_stride, o6_lo);
  vst1q_f32(output + 6 * output_stride + 4, o6_hi);
  vst1q_f32(output + 7 * output_stride, o7_lo);
  vst1q_f32(output + 7 * output_stride + 4, o7_hi);
}

template <uint64_t kMask>
void pv_pruned_impl(const at::BFloat16* p_bf16, const at::BFloat16* v_packed, int64_t v_evblock_stride, int64_t d_v,
                    float* output, int64_t output_stride) {
  for (int64_t ev = 0; ev < d_v; ev += kTile) {
    const at::BFloat16* v = v_packed + (ev / kTile) * v_evblock_stride;
    pv_pruned_one<kMask>(p_bf16, v, output + ev, output_stride);
  }
}

#endif

}  // namespace

bool qkt_8x8_bf16_2x2_pruned(const uint16_t* q_packed, const uint16_t* k_packed, int64_t d_qk, float scale,
                             uint16_t active_2x2_mask, float* scores) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON && FUSED_CPP_SDPA_CACHE_HAS_BFMMLA
  if (d_qk % 4 != 0) {
    return false;
  }
  switch (active_2x2_mask) {
    case kActiveCausal:
      qkt_pruned_impl<kActiveCausal>(q_packed, k_packed, d_qk, scale, scores);
      return true;
    case kActiveCompressed0:
      qkt_pruned_impl<kActiveCompressed0>(q_packed, k_packed, d_qk, scale, scores);
      return true;
    case kActiveCompressed1:
      qkt_pruned_impl<kActiveCompressed1>(q_packed, k_packed, d_qk, scale, scores);
      return true;
    case kActiveCompressed2:
      qkt_pruned_impl<kActiveCompressed2>(q_packed, k_packed, d_qk, scale, scores);
      return true;
    case kActiveCompressed3:
      qkt_pruned_impl<kActiveCompressed3>(q_packed, k_packed, d_qk, scale, scores);
      return true;
    case kActiveUniform2:
      qkt_pruned_impl<kActiveUniform2>(q_packed, k_packed, d_qk, scale, scores);
      return true;
    case kActiveUniform4:
      qkt_pruned_impl<kActiveUniform4>(q_packed, k_packed, d_qk, scale, scores);
      return true;
    case kActiveUniform6:
      qkt_pruned_impl<kActiveUniform6>(q_packed, k_packed, d_qk, scale, scores);
      return true;
    default:
      return false;
  }
#else
  (void)q_packed;
  (void)k_packed;
  (void)d_qk;
  (void)scale;
  (void)active_2x2_mask;
  (void)scores;
  return false;
#endif
}

bool pv_8x8_bf16_pruned(const at::BFloat16* p_bf16, const at::BFloat16* v_packed, int64_t v_evblock_stride, int64_t d_v,
                        float* output, int64_t output_row_stride, uint64_t mask) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON && FUSED_CPP_SDPA_CACHE_HAS_BF16
  switch (mask) {
    case kValidCausal:
      pv_pruned_impl<kValidCausal>(p_bf16, v_packed, v_evblock_stride, d_v, output, output_row_stride);
      return true;
    case kValidCompressed0:
      pv_pruned_impl<kValidCompressed0>(p_bf16, v_packed, v_evblock_stride, d_v, output, output_row_stride);
      return true;
    case kValidCompressed1:
      pv_pruned_impl<kValidCompressed1>(p_bf16, v_packed, v_evblock_stride, d_v, output, output_row_stride);
      return true;
    case kValidCompressed2:
      pv_pruned_impl<kValidCompressed2>(p_bf16, v_packed, v_evblock_stride, d_v, output, output_row_stride);
      return true;
    case kValidCompressed3:
      pv_pruned_impl<kValidCompressed3>(p_bf16, v_packed, v_evblock_stride, d_v, output, output_row_stride);
      return true;
    case kValidUniform2:
      pv_pruned_impl<kValidUniform2>(p_bf16, v_packed, v_evblock_stride, d_v, output, output_row_stride);
      return true;
    case kValidUniform4:
      pv_pruned_impl<kValidUniform4>(p_bf16, v_packed, v_evblock_stride, d_v, output, output_row_stride);
      return true;
    case kValidUniform6:
      pv_pruned_impl<kValidUniform6>(p_bf16, v_packed, v_evblock_stride, d_v, output, output_row_stride);
      return true;
    default:
      return false;
  }
#else
  (void)p_bf16;
  (void)v_packed;
  (void)v_evblock_stride;
  (void)d_v;
  (void)output;
  (void)output_row_stride;
  (void)mask;
  return false;
#endif
}

bool pack_v_8x8_bfmmla_sve(const at::BFloat16* v, int64_t v_row_stride,
                           int64_t d_v, at::BFloat16* v_bfmmla_packed) {
#if FUSED_CPP_SPARSE_MLA_HAS_SVE_BF16
  if (v == nullptr || v_bfmmla_packed == nullptr || v_row_stride < d_v ||
      d_v <= 0 || d_v % kTile != 0) {
    return false;
  }
  const int64_t vector_words = svcntw();
  if (vector_words != 4 && vector_words != 8) {
    return false;
  }
  const int64_t segment_count = vector_words / 4;
  for (int64_t ev = 0; ev < d_v; ev += kTile) {
    at::BFloat16* dst_ev = v_bfmmla_packed + (ev / kTile) * kTile * kTile;
    for (int64_t key_group = 0; key_group < 2; ++key_group) {
      for (int64_t col_pair = 0; col_pair < 4; col_pair += segment_count) {
        at::BFloat16* dst = dst_ev + (key_group * 4 + col_pair) * 8;
        for (int64_t segment = 0; segment < segment_count; ++segment) {
          for (int64_t key = 0; key < 4; ++key) {
            const at::BFloat16* src = v + (key_group * 4 + key) * v_row_stride +
                                      ev + (col_pair + segment) * 2;
            dst[segment * 8 + key] = src[0];
            dst[segment * 8 + 4 + key] = src[1];
          }
        }
      }
    }
  }
  return true;
#else
  (void)v;
  (void)v_row_stride;
  (void)d_v;
  (void)v_bfmmla_packed;
  return false;
#endif
}

bool online_softmax_pv_8x8_bf16_sve(
    const uint16_t* q_packed, const uint16_t* k_packed,
    const at::BFloat16* v_packed, int64_t v_evblock_stride, int64_t d_qk,
    const at::BFloat16* v_bfmmla_packed, int64_t v_bfmmla_evblock_stride,
    int64_t d_v, float scale, uint64_t valid_mask_value, bool prune_2x2,
    SparseMlaPvBackend pv_backend, float* running_max, float* running_sum,
    float* output, int64_t output_row_stride) {
#if FUSED_CPP_SPARSE_MLA_HAS_SVE_BF16
  const int64_t vector_words = svcntw();
  if ((vector_words != 4 && vector_words != 8) || q_packed == nullptr ||
      k_packed == nullptr || v_packed == nullptr || running_max == nullptr ||
      running_sum == nullptr || output == nullptr || d_qk <= 0 ||
      d_qk % 4 != 0 || d_v <= 0 || d_v % kTile != 0 ||
      v_evblock_stride < kTile * kTile || output_row_stride < d_v) {
    return false;
  }
  if (pv_backend == SparseMlaPvBackend::kBfmmla &&
      (v_bfmmla_packed == nullptr || v_bfmmla_evblock_stride < kTile * kTile)) {
    return false;
  }
#define FUSED_CPP_DISPATCH_PV(BACKEND)                                         \
  do {                                                                         \
    if (prune_2x2) {                                                           \
      return dispatch_online_softmax_pv_sve<true, BACKEND>(                    \
          q_packed, k_packed, v_packed, v_evblock_stride, d_qk,                \
          v_bfmmla_packed, v_bfmmla_evblock_stride, d_v, scale,                \
          valid_mask_value, running_max, running_sum, output,                  \
          output_row_stride);                                                  \
    }                                                                          \
    return dispatch_online_softmax_pv_sve<false, BACKEND>(                     \
        q_packed, k_packed, v_packed, v_evblock_stride, d_qk, v_bfmmla_packed, \
        v_bfmmla_evblock_stride, d_v, scale, valid_mask_value, running_max,    \
        running_sum, output, output_row_stride);                               \
  } while (false)
  switch (pv_backend) {
    case SparseMlaPvBackend::kFmla:
      FUSED_CPP_DISPATCH_PV(SparseMlaPvBackend::kFmla);
    case SparseMlaPvBackend::kBfmlal:
      FUSED_CPP_DISPATCH_PV(SparseMlaPvBackend::kBfmlal);
    case SparseMlaPvBackend::kBfmmla:
      FUSED_CPP_DISPATCH_PV(SparseMlaPvBackend::kBfmmla);
  }
#undef FUSED_CPP_DISPATCH_PV
  return false;
#else
  (void)q_packed;
  (void)k_packed;
  (void)v_packed;
  (void)v_evblock_stride;
  (void)v_bfmmla_packed;
  (void)v_bfmmla_evblock_stride;
  (void)d_qk;
  (void)d_v;
  (void)scale;
  (void)valid_mask_value;
  (void)prune_2x2;
  (void)pv_backend;
  (void)running_max;
  (void)running_sum;
  (void)output;
  (void)output_row_stride;
  return false;
#endif
}

}  // namespace fused_cpp::sparse_mla_tail_microkernels
