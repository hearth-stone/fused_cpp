// SPDX-License-Identifier: Apache-2.0
#include "sparse_mla_tail_microkernels.h"

#include <algorithm>
#include <array>
#include <cstdint>

#include "sdpa_microkernels/neon_cache_config.h"

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

}  // namespace fused_cpp::sparse_mla_tail_microkernels
