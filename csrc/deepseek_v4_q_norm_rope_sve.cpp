#include "deepseek_v4_q_norm_rope_sve.h"

#include <cmath>
#include <cstdint>
#include <type_traits>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

#ifndef FUSED_CPP_STRICT_MODE
#define FUSED_CPP_STRICT_MODE 0
#endif

namespace fused_cpp::deepseek_v4 {
namespace {

#if defined(__ARM_FEATURE_SVE)

#if defined(__GNUC__) || defined(__clang__)
#define FUSED_CPP_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define FUSED_CPP_ALWAYS_INLINE inline
#endif

#if defined(__ARM_FEATURE_SVE_BF16) || \
    (defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC))
#define FUSED_CPP_HAS_SVE_BF16 1
#else
#define FUSED_CPP_HAS_SVE_BF16 0
#endif

inline svfloat32_t bf16_load_f32(svbool_t pg, const uint16_t* ptr) {
  const svuint32_t h = svld1uh_u32(pg, ptr);
  return svreinterpret_f32_u32(svlsl_n_u32_x(pg, h, 16));
}

inline svfloat32_t bf16_u16_to_f32_low(svbool_t pg, svuint16_t h16) {
  const svuint32_t h = svunpklo_u32(h16);
  return svreinterpret_f32_u32(svlsl_n_u32_x(pg, h, 16));
}

inline svfloat32x2_t bf16_load2_evenodd_f32(svbool_t pg,
                                            svbool_t pg_h,
                                            const uint16_t* ptr) {
  const svuint16x2_t pair = svld2_u16(pg_h, ptr);
  return svcreate2_f32(
      bf16_u16_to_f32_low(pg, svget2_u16(pair, 0)),
      bf16_u16_to_f32_low(pg, svget2_u16(pair, 1)));
}

inline svfloat32x2_t bf16_load2_evenodd_f32(svbool_t pg, const uint16_t* ptr) {
  const int64_t active_lanes =
      static_cast<int64_t>(svcntp_b32(svptrue_b32(), pg));
  const svbool_t pg_h = svwhilelt_b16(static_cast<int64_t>(0), active_lanes);
  return bf16_load2_evenodd_f32(pg, pg_h, ptr);
}

inline svuint32_t f32_to_bf16_bits(svbool_t pg, svfloat32_t v) {
  const svuint32_t bits = svreinterpret_u32_f32(v);
  const svuint32_t lsb = svand_n_u32_x(pg, svlsr_n_u32_x(pg, bits, 16), 1);
  const svuint32_t bias = svadd_n_u32_x(pg, lsb, 0x7fff);
  return svlsr_n_u32_x(pg, svadd_u32_x(pg, bits, bias), 16);
}

inline void bf16_store_f32(svbool_t pg, uint16_t* ptr, svfloat32_t v) {
  svst1h_u32(pg, ptr, f32_to_bf16_bits(pg, v));
}

inline void bf16_scatter_evenodd_f32(svbool_t pg, uint16_t* ptr, svfloat32_t v) {
  const svuint32_t idx = svindex_u32(0, 2);
  svst1h_scatter_u32index_u32(pg, ptr, idx, f32_to_bf16_bits(pg, v));
}

inline void prefetch_read_l1(const void* ptr) {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(ptr, 0, 3);
#else
  (void)ptr;
#endif
}

template <typename scalar_t>
FUSED_CPP_ALWAYS_INLINE svfloat32_t load_contiguous_f32(svbool_t pg, const scalar_t* ptr) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return svld1_f32(pg, ptr);
  } else {
    return bf16_load_f32(pg, ptr);
  }
}

template <typename scalar_t>
FUSED_CPP_ALWAYS_INLINE void load_evenodd_contiguous_f32(
    svbool_t pg,
    const scalar_t* ptr,
    svfloat32_t& even,
    svfloat32_t& odd) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    const svfloat32x2_t pair = svld2_f32(pg, ptr);
    even = svget2_f32(pair, 0);
    odd = svget2_f32(pair, 1);
  } else {
    const svfloat32x2_t pair = bf16_load2_evenodd_f32(pg, ptr);
    even = svget2_f32(pair, 0);
    odd = svget2_f32(pair, 1);
  }
}

template <typename scalar_t>
FUSED_CPP_ALWAYS_INLINE void load_evenodd_contiguous_f32(
    svbool_t pg,
    svbool_t pg_h,
    const scalar_t* ptr,
    svfloat32_t& even,
    svfloat32_t& odd) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    (void)pg_h;
    const svfloat32x2_t pair = svld2_f32(pg, ptr);
    even = svget2_f32(pair, 0);
    odd = svget2_f32(pair, 1);
  } else {
    const svfloat32x2_t pair = bf16_load2_evenodd_f32(pg, pg_h, ptr);
    even = svget2_f32(pair, 0);
    odd = svget2_f32(pair, 1);
  }
}

template <typename scalar_t>
FUSED_CPP_ALWAYS_INLINE void store_contiguous_f32(svbool_t pg, scalar_t* ptr, svfloat32_t v) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    svst1_f32(pg, ptr, v);
  } else {
    bf16_store_f32(pg, ptr, v);
  }
}

template <typename scalar_t>
FUSED_CPP_ALWAYS_INLINE void store_evenodd_contiguous_f32(
    svbool_t pg,
    scalar_t* ptr,
    svfloat32_t even,
    svfloat32_t odd) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    svst2_f32(pg, ptr, svcreate2_f32(even, odd));
  } else {
    bf16_scatter_evenodd_f32(pg, ptr, even);
    bf16_scatter_evenodd_f32(pg, ptr + 1, odd);
  }
}

template <typename row_t>
FUSED_CPP_ALWAYS_INLINE void compute_kv_rows4_rope_chunk_sve(
    const row_t* row0,
    const row_t* row1,
    const row_t* row2,
    const row_t* row3,
    const float* cos_row0,
    const float* sin_row0,
    const float* cos_row1,
    const float* sin_row1,
    const float* cos_row2,
    const float* sin_row2,
    const float* cos_row3,
    const float* sin_row3,
    int64_t nope_head_dim,
    int64_t pair,
    svbool_t pg,
    svbool_t pg_h,
    svfloat32x2_t& out0,
    svfloat32x2_t& out1,
    svfloat32x2_t& out2,
    svfloat32x2_t& out3) {
  const int64_t offset = nope_head_dim + 2 * pair;
  svfloat32_t even0;
  svfloat32_t odd0;
  svfloat32_t even1;
  svfloat32_t odd1;
  svfloat32_t even2;
  svfloat32_t odd2;
  svfloat32_t even3;
  svfloat32_t odd3;
  load_evenodd_contiguous_f32(pg, pg_h, row0 + offset, even0, odd0);
  load_evenodd_contiguous_f32(pg, pg_h, row1 + offset, even1, odd1);
  load_evenodd_contiguous_f32(pg, pg_h, row2 + offset, even2, odd2);
  load_evenodd_contiguous_f32(pg, pg_h, row3 + offset, even3, odd3);

  const svfloat32_t c0 = svld1_f32(pg, cos_row0 + pair);
  const svfloat32_t s0 = svld1_f32(pg, sin_row0 + pair);
  const svfloat32_t c1 = svld1_f32(pg, cos_row1 + pair);
  const svfloat32_t s1 = svld1_f32(pg, sin_row1 + pair);
  const svfloat32_t c2 = svld1_f32(pg, cos_row2 + pair);
  const svfloat32_t s2 = svld1_f32(pg, sin_row2 + pair);
  const svfloat32_t c3 = svld1_f32(pg, cos_row3 + pair);
  const svfloat32_t s3 = svld1_f32(pg, sin_row3 + pair);

  out0 = svcreate2_f32(
      svmul_f32_x(pg, even0, c0),
      svmul_f32_x(pg, odd0, c0));
  out1 = svcreate2_f32(
      svmul_f32_x(pg, even1, c1),
      svmul_f32_x(pg, odd1, c1));
  out2 = svcreate2_f32(
      svmul_f32_x(pg, even2, c2),
      svmul_f32_x(pg, odd2, c2));
  out3 = svcreate2_f32(
      svmul_f32_x(pg, even3, c3),
      svmul_f32_x(pg, odd3, c3));

  out0 = svcreate2_f32(
      svmls_f32_x(pg, svget2_f32(out0, 0), odd0, s0),
      svmla_f32_x(pg, svget2_f32(out0, 1), even0, s0));
  out1 = svcreate2_f32(
      svmls_f32_x(pg, svget2_f32(out1, 0), odd1, s1),
      svmla_f32_x(pg, svget2_f32(out1, 1), even1, s1));
  out2 = svcreate2_f32(
      svmls_f32_x(pg, svget2_f32(out2, 0), odd2, s2),
      svmla_f32_x(pg, svget2_f32(out2, 1), even2, s2));
  out3 = svcreate2_f32(
      svmls_f32_x(pg, svget2_f32(out3, 0), odd3, s3),
      svmla_f32_x(pg, svget2_f32(out3, 1), even3, s3));
}

#if !FUSED_CPP_STRICT_MODE
float sum_sq_f32_sve(const float* row, int64_t head_dim) {
  const int64_t vl = static_cast<int64_t>(svcntw());
  const int64_t step = 4 * vl;
  const svbool_t pg_all = svptrue_b32();
  svfloat32_t acc0 = svdup_f32(0.0f);
  svfloat32_t acc1 = svdup_f32(0.0f);
  svfloat32_t acc2 = svdup_f32(0.0f);
  svfloat32_t acc3 = svdup_f32(0.0f);
  int64_t d = 0;
  for (; d + step <= head_dim; d += step) {
    svfloat32_t v = svld1_f32(pg_all, row + d);
    acc0 = svmla_f32_x(pg_all, acc0, v, v);
    v = svld1_f32(pg_all, row + d + vl);
    acc1 = svmla_f32_x(pg_all, acc1, v, v);
    v = svld1_f32(pg_all, row + d + 2 * vl);
    acc2 = svmla_f32_x(pg_all, acc2, v, v);
    v = svld1_f32(pg_all, row + d + 3 * vl);
    acc3 = svmla_f32_x(pg_all, acc3, v, v);
  }
  for (; d < head_dim; d += vl) {
    const svbool_t pg = svwhilelt_b32(d, head_dim);
    const svfloat32_t v = svld1_f32(pg, row + d);
    acc0 = svmla_f32_m(pg, acc0, v, v);
  }
  acc0 = svadd_f32_x(pg_all, acc0, acc1);
  acc2 = svadd_f32_x(pg_all, acc2, acc3);
  const svfloat32_t acc = svadd_f32_x(pg_all, acc0, acc2);
  return svaddv_f32(svptrue_b32(), acc);
}

// TODO(zhangxu): check asm
void sum_sq_f32_sve_4x4(const float* row0,
                        const float* row1,
                        const float* row2,
                        const float* row3,
                        int64_t head_dim,
                        float& sum0,
                        float& sum1,
                        float& sum2,
                        float& sum3) {
  const int64_t vl = static_cast<int64_t>(svcntw());
  const int64_t step = 4 * vl;
  const int64_t unroll_step = 2 * step;
  const svbool_t pg_all = svptrue_b32();

  if (head_dim > 0) {
    prefetch_read_l1(row0);
    prefetch_read_l1(row1);
    prefetch_read_l1(row2);
    prefetch_read_l1(row3);
  }

  svfloat32_t acc00 = svdup_f32(0.0f);
  svfloat32_t acc01 = svdup_f32(0.0f);
  svfloat32_t acc02 = svdup_f32(0.0f);
  svfloat32_t acc03 = svdup_f32(0.0f);
  svfloat32_t acc10 = svdup_f32(0.0f);
  svfloat32_t acc11 = svdup_f32(0.0f);
  svfloat32_t acc12 = svdup_f32(0.0f);
  svfloat32_t acc13 = svdup_f32(0.0f);
  svfloat32_t acc20 = svdup_f32(0.0f);
  svfloat32_t acc21 = svdup_f32(0.0f);
  svfloat32_t acc22 = svdup_f32(0.0f);
  svfloat32_t acc23 = svdup_f32(0.0f);
  svfloat32_t acc30 = svdup_f32(0.0f);
  svfloat32_t acc31 = svdup_f32(0.0f);
  svfloat32_t acc32 = svdup_f32(0.0f);
  svfloat32_t acc33 = svdup_f32(0.0f);

  int64_t d = 0;
  for (; d + unroll_step <= head_dim; d += unroll_step) {
    {
      svfloat32_t a0 = svld1_f32(pg_all, row0 + d);
      svfloat32_t a1 = svld1_f32(pg_all, row1 + d);
      svfloat32_t a2 = svld1_f32(pg_all, row2 + d);
      svfloat32_t a3 = svld1_f32(pg_all, row3 + d);

      svfloat32_t b0 = svld1_f32(pg_all, row0 + d + vl);
      svfloat32_t b1 = svld1_f32(pg_all, row1 + d + vl);
      svfloat32_t b2 = svld1_f32(pg_all, row2 + d + vl);
      svfloat32_t b3 = svld1_f32(pg_all, row3 + d + vl);

      acc00 = svmla_f32_x(pg_all, acc00, a0, a0);
      acc10 = svmla_f32_x(pg_all, acc10, a1, a1);
      acc20 = svmla_f32_x(pg_all, acc20, a2, a2);
      acc30 = svmla_f32_x(pg_all, acc30, a3, a3);

      a0 = svld1_f32(pg_all, row0 + d + 2 * vl);
      a1 = svld1_f32(pg_all, row1 + d + 2 * vl);
      a2 = svld1_f32(pg_all, row2 + d + 2 * vl);
      a3 = svld1_f32(pg_all, row3 + d + 2 * vl);

      acc01 = svmla_f32_x(pg_all, acc01, b0, b0);
      acc11 = svmla_f32_x(pg_all, acc11, b1, b1);
      acc21 = svmla_f32_x(pg_all, acc21, b2, b2);
      acc31 = svmla_f32_x(pg_all, acc31, b3, b3);

      b0 = svld1_f32(pg_all, row0 + d + 3 * vl);
      b1 = svld1_f32(pg_all, row1 + d + 3 * vl);
      b2 = svld1_f32(pg_all, row2 + d + 3 * vl);
      b3 = svld1_f32(pg_all, row3 + d + 3 * vl);

      acc02 = svmla_f32_x(pg_all, acc02, a0, a0);
      acc12 = svmla_f32_x(pg_all, acc12, a1, a1);
      acc22 = svmla_f32_x(pg_all, acc22, a2, a2);
      acc32 = svmla_f32_x(pg_all, acc32, a3, a3);

      a0 = svld1_f32(pg_all, row0 + d + step);
      a1 = svld1_f32(pg_all, row1 + d + step);
      a2 = svld1_f32(pg_all, row2 + d + step);
      a3 = svld1_f32(pg_all, row3 + d + step);

      acc03 = svmla_f32_x(pg_all, acc03, b0, b0);
      acc13 = svmla_f32_x(pg_all, acc13, b1, b1);
      acc23 = svmla_f32_x(pg_all, acc23, b2, b2);
      acc33 = svmla_f32_x(pg_all, acc33, b3, b3);

      b0 = svld1_f32(pg_all, row0 + d + step + vl);
      b1 = svld1_f32(pg_all, row1 + d + step + vl);
      b2 = svld1_f32(pg_all, row2 + d + step + vl);
      b3 = svld1_f32(pg_all, row3 + d + step + vl);

      acc00 = svmla_f32_x(pg_all, acc00, a0, a0);
      acc10 = svmla_f32_x(pg_all, acc10, a1, a1);
      acc20 = svmla_f32_x(pg_all, acc20, a2, a2);
      acc30 = svmla_f32_x(pg_all, acc30, a3, a3);

      a0 = svld1_f32(pg_all, row0 + d + step + 2 * vl);
      a1 = svld1_f32(pg_all, row1 + d + step + 2 * vl);
      a2 = svld1_f32(pg_all, row2 + d + step + 2 * vl);
      a3 = svld1_f32(pg_all, row3 + d + step + 2 * vl);

      acc01 = svmla_f32_x(pg_all, acc01, b0, b0);
      acc11 = svmla_f32_x(pg_all, acc11, b1, b1);
      acc21 = svmla_f32_x(pg_all, acc21, b2, b2);
      acc31 = svmla_f32_x(pg_all, acc31, b3, b3);

      b0 = svld1_f32(pg_all, row0 + d + step + 3 * vl);
      b1 = svld1_f32(pg_all, row1 + d + step + 3 * vl);
      b2 = svld1_f32(pg_all, row2 + d + step + 3 * vl);
      b3 = svld1_f32(pg_all, row3 + d + step + 3 * vl);

      acc02 = svmla_f32_x(pg_all, acc02, a0, a0);
      acc12 = svmla_f32_x(pg_all, acc12, a1, a1);
      acc22 = svmla_f32_x(pg_all, acc22, a2, a2);
      acc32 = svmla_f32_x(pg_all, acc32, a3, a3);

      acc03 = svmla_f32_x(pg_all, acc03, b0, b0);
      acc13 = svmla_f32_x(pg_all, acc13, b1, b1);
      acc23 = svmla_f32_x(pg_all, acc23, b2, b2);
      acc33 = svmla_f32_x(pg_all, acc33, b3, b3);
    }
  }
  for (; d + step <= head_dim; d += step) {
    {
      svfloat32_t a0 = svld1_f32(pg_all, row0 + d);
      svfloat32_t a1 = svld1_f32(pg_all, row1 + d);
      svfloat32_t a2 = svld1_f32(pg_all, row2 + d);
      svfloat32_t a3 = svld1_f32(pg_all, row3 + d);

      svfloat32_t b0 = svld1_f32(pg_all, row0 + d + vl);
      svfloat32_t b1 = svld1_f32(pg_all, row1 + d + vl);
      svfloat32_t b2 = svld1_f32(pg_all, row2 + d + vl);
      svfloat32_t b3 = svld1_f32(pg_all, row3 + d + vl);

      acc00 = svmla_f32_x(pg_all, acc00, a0, a0);
      acc10 = svmla_f32_x(pg_all, acc10, a1, a1);
      acc20 = svmla_f32_x(pg_all, acc20, a2, a2);
      acc30 = svmla_f32_x(pg_all, acc30, a3, a3);

      a0 = svld1_f32(pg_all, row0 + d + 2 * vl);
      a1 = svld1_f32(pg_all, row1 + d + 2 * vl);
      a2 = svld1_f32(pg_all, row2 + d + 2 * vl);
      a3 = svld1_f32(pg_all, row3 + d + 2 * vl);

      acc01 = svmla_f32_x(pg_all, acc01, b0, b0);
      acc11 = svmla_f32_x(pg_all, acc11, b1, b1);
      acc21 = svmla_f32_x(pg_all, acc21, b2, b2);
      acc31 = svmla_f32_x(pg_all, acc31, b3, b3);

      b0 = svld1_f32(pg_all, row0 + d + 3 * vl);
      b1 = svld1_f32(pg_all, row1 + d + 3 * vl);
      b2 = svld1_f32(pg_all, row2 + d + 3 * vl);
      b3 = svld1_f32(pg_all, row3 + d + 3 * vl);

      acc02 = svmla_f32_x(pg_all, acc02, a0, a0);
      acc12 = svmla_f32_x(pg_all, acc12, a1, a1);
      acc22 = svmla_f32_x(pg_all, acc22, a2, a2);
      acc32 = svmla_f32_x(pg_all, acc32, a3, a3);

      acc03 = svmla_f32_x(pg_all, acc03, b0, b0);
      acc13 = svmla_f32_x(pg_all, acc13, b1, b1);
      acc23 = svmla_f32_x(pg_all, acc23, b2, b2);
      acc33 = svmla_f32_x(pg_all, acc33, b3, b3);
    }
  }

  for (; d < head_dim; d += vl) {
    const svbool_t pg = svwhilelt_b32(d, head_dim);
    const svfloat32_t v0 = svld1_f32(pg, row0 + d);
    const svfloat32_t v1 = svld1_f32(pg, row1 + d);
    const svfloat32_t v2 = svld1_f32(pg, row2 + d);
    const svfloat32_t v3 = svld1_f32(pg, row3 + d);
    acc00 = svmla_f32_m(pg, acc00, v0, v0);
    acc10 = svmla_f32_m(pg, acc10, v1, v1);
    acc20 = svmla_f32_m(pg, acc20, v2, v2);
    acc30 = svmla_f32_m(pg, acc30, v3, v3);
  }

  acc00 = svadd_f32_x(pg_all, acc00, acc01);
  acc02 = svadd_f32_x(pg_all, acc02, acc03);
  acc10 = svadd_f32_x(pg_all, acc10, acc11);
  acc12 = svadd_f32_x(pg_all, acc12, acc13);
  acc20 = svadd_f32_x(pg_all, acc20, acc21);
  acc22 = svadd_f32_x(pg_all, acc22, acc23);
  acc30 = svadd_f32_x(pg_all, acc30, acc31);
  acc32 = svadd_f32_x(pg_all, acc32, acc33);

  sum0 = svaddv_f32(pg_all, svadd_f32_x(pg_all, acc00, acc02));
  sum1 = svaddv_f32(pg_all, svadd_f32_x(pg_all, acc10, acc12));
  sum2 = svaddv_f32(pg_all, svadd_f32_x(pg_all, acc20, acc22));
  sum3 = svaddv_f32(pg_all, svadd_f32_x(pg_all, acc30, acc32));
}
#endif

float sum_sq_bf16_sve(const uint16_t* row, int64_t head_dim) {
  svfloat32_t acc = svdup_f32(0.0f);
  for (int64_t d = 0; d < head_dim; d += static_cast<int64_t>(svcntw())) {
    const svbool_t pg = svwhilelt_b32(d, head_dim);
    const svfloat32_t v = bf16_load_f32(pg, row + d);
    acc = svmla_f32_m(pg, acc, v, v);
  }
  return svaddv_f32(svptrue_b32(), acc);
}

void sum_sq_bf16_sve_4x4(const uint16_t* row0,
                         const uint16_t* row1,
                         const uint16_t* row2,
                         const uint16_t* row3,
                         int64_t head_dim,
                         float& sum0,
                         float& sum1,
                         float& sum2,
                         float& sum3) {
#if FUSED_CPP_HAS_SVE_BF16
  const int64_t vlh = static_cast<int64_t>(svcnth());
  const int64_t step = 4 * vlh;
  const int64_t unroll_step = 2 * step;
  const svbool_t pg_all_b16 = svptrue_b16();
  const svbool_t pg_all_b32 = svptrue_b32();

  if (head_dim > 0) {
    prefetch_read_l1(row0);
    prefetch_read_l1(row1);
    prefetch_read_l1(row2);
    prefetch_read_l1(row3);
  }

  svfloat32_t acc00 = svdup_f32(0.0f);
  svfloat32_t acc01 = svdup_f32(0.0f);
  svfloat32_t acc02 = svdup_f32(0.0f);
  svfloat32_t acc03 = svdup_f32(0.0f);
  svfloat32_t acc10 = svdup_f32(0.0f);
  svfloat32_t acc11 = svdup_f32(0.0f);
  svfloat32_t acc12 = svdup_f32(0.0f);
  svfloat32_t acc13 = svdup_f32(0.0f);
  svfloat32_t acc20 = svdup_f32(0.0f);
  svfloat32_t acc21 = svdup_f32(0.0f);
  svfloat32_t acc22 = svdup_f32(0.0f);
  svfloat32_t acc23 = svdup_f32(0.0f);
  svfloat32_t acc30 = svdup_f32(0.0f);
  svfloat32_t acc31 = svdup_f32(0.0f);
  svfloat32_t acc32 = svdup_f32(0.0f);
  svfloat32_t acc33 = svdup_f32(0.0f);

  int64_t d = 0;
  for (; d + unroll_step <= head_dim; d += unroll_step) {
    {
      svbfloat16_t a0 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row0 + d));
      svbfloat16_t a1 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row1 + d));
      svbfloat16_t a2 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row2 + d));
      svbfloat16_t a3 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row3 + d));

      svbfloat16_t b0 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row0 + d + vlh));
      svbfloat16_t b1 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row1 + d + vlh));
      svbfloat16_t b2 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row2 + d + vlh));
      svbfloat16_t b3 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row3 + d + vlh));

      acc00 = svbfdot_f32(acc00, a0, a0);
      acc10 = svbfdot_f32(acc10, a1, a1);
      acc20 = svbfdot_f32(acc20, a2, a2);
      acc30 = svbfdot_f32(acc30, a3, a3);

      a0 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row0 + d + 2 * vlh));
      a1 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row1 + d + 2 * vlh));
      a2 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row2 + d + 2 * vlh));
      a3 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row3 + d + 2 * vlh));

      acc01 = svbfdot_f32(acc01, b0, b0);
      acc11 = svbfdot_f32(acc11, b1, b1);
      acc21 = svbfdot_f32(acc21, b2, b2);
      acc31 = svbfdot_f32(acc31, b3, b3);

      b0 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row0 + d + 3 * vlh));
      b1 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row1 + d + 3 * vlh));
      b2 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row2 + d + 3 * vlh));
      b3 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row3 + d + 3 * vlh));

      acc02 = svbfdot_f32(acc02, a0, a0);
      acc12 = svbfdot_f32(acc12, a1, a1);
      acc22 = svbfdot_f32(acc22, a2, a2);
      acc32 = svbfdot_f32(acc32, a3, a3);

      a0 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row0 + d + step));
      a1 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row1 + d + step));
      a2 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row2 + d + step));
      a3 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row3 + d + step));

      acc03 = svbfdot_f32(acc03, b0, b0);
      acc13 = svbfdot_f32(acc13, b1, b1);
      acc23 = svbfdot_f32(acc23, b2, b2);
      acc33 = svbfdot_f32(acc33, b3, b3);

      b0 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row0 + d + step + vlh));
      b1 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row1 + d + step + vlh));
      b2 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row2 + d + step + vlh));
      b3 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row3 + d + step + vlh));

      acc00 = svbfdot_f32(acc00, a0, a0);
      acc10 = svbfdot_f32(acc10, a1, a1);
      acc20 = svbfdot_f32(acc20, a2, a2);
      acc30 = svbfdot_f32(acc30, a3, a3);

      a0 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row0 + d + step + 2 * vlh));
      a1 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row1 + d + step + 2 * vlh));
      a2 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row2 + d + step + 2 * vlh));
      a3 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row3 + d + step + 2 * vlh));

      acc01 = svbfdot_f32(acc01, b0, b0);
      acc11 = svbfdot_f32(acc11, b1, b1);
      acc21 = svbfdot_f32(acc21, b2, b2);
      acc31 = svbfdot_f32(acc31, b3, b3);

      b0 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row0 + d + step + 3 * vlh));
      b1 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row1 + d + step + 3 * vlh));
      b2 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row2 + d + step + 3 * vlh));
      b3 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row3 + d + step + 3 * vlh));

      acc02 = svbfdot_f32(acc02, a0, a0);
      acc12 = svbfdot_f32(acc12, a1, a1);
      acc22 = svbfdot_f32(acc22, a2, a2);
      acc32 = svbfdot_f32(acc32, a3, a3);

      acc03 = svbfdot_f32(acc03, b0, b0);
      acc13 = svbfdot_f32(acc13, b1, b1);
      acc23 = svbfdot_f32(acc23, b2, b2);
      acc33 = svbfdot_f32(acc33, b3, b3);
    }
  }

  for (; d + step <= head_dim; d += step) {
    {
      svbfloat16_t a0 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row0 + d));
      svbfloat16_t a1 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row1 + d));
      svbfloat16_t a2 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row2 + d));
      svbfloat16_t a3 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row3 + d));

      svbfloat16_t b0 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row0 + d + vlh));
      svbfloat16_t b1 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row1 + d + vlh));
      svbfloat16_t b2 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row2 + d + vlh));
      svbfloat16_t b3 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row3 + d + vlh));

      acc00 = svbfdot_f32(acc00, a0, a0);
      acc10 = svbfdot_f32(acc10, a1, a1);
      acc20 = svbfdot_f32(acc20, a2, a2);
      acc30 = svbfdot_f32(acc30, a3, a3);

      a0 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row0 + d + 2 * vlh));
      a1 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row1 + d + 2 * vlh));
      a2 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row2 + d + 2 * vlh));
      a3 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row3 + d + 2 * vlh));

      acc01 = svbfdot_f32(acc01, b0, b0);
      acc11 = svbfdot_f32(acc11, b1, b1);
      acc21 = svbfdot_f32(acc21, b2, b2);
      acc31 = svbfdot_f32(acc31, b3, b3);

      b0 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row0 + d + 3 * vlh));
      b1 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row1 + d + 3 * vlh));
      b2 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row2 + d + 3 * vlh));
      b3 = svld1_bf16(pg_all_b16, reinterpret_cast<const bfloat16_t*>(row3 + d + 3 * vlh));

      acc02 = svbfdot_f32(acc02, a0, a0);
      acc12 = svbfdot_f32(acc12, a1, a1);
      acc22 = svbfdot_f32(acc22, a2, a2);
      acc32 = svbfdot_f32(acc32, a3, a3);

      acc03 = svbfdot_f32(acc03, b0, b0);
      acc13 = svbfdot_f32(acc13, b1, b1);
      acc23 = svbfdot_f32(acc23, b2, b2);
      acc33 = svbfdot_f32(acc33, b3, b3);
    }
  }

  for (; d < head_dim; d += vlh) {
    const svbool_t pg = svwhilelt_b16(d, head_dim);
    const svbfloat16_t v0 =
        svld1_bf16(pg, reinterpret_cast<const bfloat16_t*>(row0 + d));
    const svbfloat16_t v1 =
        svld1_bf16(pg, reinterpret_cast<const bfloat16_t*>(row1 + d));
    const svbfloat16_t v2 =
        svld1_bf16(pg, reinterpret_cast<const bfloat16_t*>(row2 + d));
    const svbfloat16_t v3 =
        svld1_bf16(pg, reinterpret_cast<const bfloat16_t*>(row3 + d));
    acc00 = svbfdot_f32(acc00, v0, v0);
    acc10 = svbfdot_f32(acc10, v1, v1);
    acc20 = svbfdot_f32(acc20, v2, v2);
    acc30 = svbfdot_f32(acc30, v3, v3);
  }

  acc00 = svadd_f32_x(pg_all_b32, acc00, acc01);
  acc02 = svadd_f32_x(pg_all_b32, acc02, acc03);
  acc10 = svadd_f32_x(pg_all_b32, acc10, acc11);
  acc12 = svadd_f32_x(pg_all_b32, acc12, acc13);
  acc20 = svadd_f32_x(pg_all_b32, acc20, acc21);
  acc22 = svadd_f32_x(pg_all_b32, acc22, acc23);
  acc30 = svadd_f32_x(pg_all_b32, acc30, acc31);
  acc32 = svadd_f32_x(pg_all_b32, acc32, acc33);

  sum0 = svaddv_f32(pg_all_b32, svadd_f32_x(pg_all_b32, acc00, acc02));
  sum1 = svaddv_f32(pg_all_b32, svadd_f32_x(pg_all_b32, acc10, acc12));
  sum2 = svaddv_f32(pg_all_b32, svadd_f32_x(pg_all_b32, acc20, acc22));
  sum3 = svaddv_f32(pg_all_b32, svadd_f32_x(pg_all_b32, acc30, acc32));
#else
#error "sum_sq_bf16_sve_4x4 requires SVE BF16"
#endif
}

void process_f32_rope_row_sve(float* row,
                              const float* cos_row,
                              const float* sin_row,
                              int64_t nope_head_dim,
                              int64_t rope_half,
                              float inv_rms);

void process_bf16_rope_row_sve(uint16_t* row,
                               const float* cos_row,
                               const float* sin_row,
                               int64_t nope_head_dim,
                               int64_t rope_half,
                               float inv_rms);

#if !FUSED_CPP_STRICT_MODE
void process_f32_row_sve(float* row,
                         const float* cos_row,
                         const float* sin_row,
                         int64_t head_dim,
                         int64_t nope_head_dim,
                         int64_t rope_half,
                         float inv_rms) {
  const svfloat32_t inv = svdup_f32(inv_rms);
  for (int64_t d = 0; d < nope_head_dim; d += static_cast<int64_t>(svcntw())) {
    const svbool_t pg = svwhilelt_b32(d, nope_head_dim);
    const svfloat32_t v = svld1_f32(pg, row + d);
    svst1_f32(pg, row + d, svmul_f32_x(pg, v, inv));
  }

  process_f32_rope_row_sve(row, cos_row, sin_row, nope_head_dim, rope_half, inv_rms);
}
#endif

void process_f32_rope_row_sve(float* row,
                              const float* cos_row,
                              const float* sin_row,
                              int64_t nope_head_dim,
                              int64_t rope_half,
                              float inv_rms) {
  const svfloat32_t inv = svdup_f32(inv_rms);
  for (int64_t pair = 0; pair < rope_half; pair += static_cast<int64_t>(svcntw())) {
    const svbool_t pg = svwhilelt_b32(pair, rope_half);
    svfloat32x2_t q_pair = svld2_f32(pg, row + nope_head_dim + 2 * pair);
    const svfloat32_t c = svld1_f32(pg, cos_row + pair);
    const svfloat32_t s = svld1_f32(pg, sin_row + pair);
    const svfloat32_t c_inv = svmul_f32_x(pg, c, inv);
    const svfloat32_t s_inv = svmul_f32_x(pg, s, inv);
    const svfloat32_t even = svget2_f32(q_pair, 0);
    const svfloat32_t odd = svget2_f32(q_pair, 1);
    const svfloat32_t out_even =
        svmls_f32_x(pg, svmul_f32_x(pg, even, c_inv), odd, s_inv);
    const svfloat32_t out_odd =
        svmla_f32_x(pg, svmul_f32_x(pg, odd, c_inv), even, s_inv);
    svst2_f32(pg, row + nope_head_dim + 2 * pair, svcreate2_f32(out_even, out_odd));
  }
}

#if !FUSED_CPP_STRICT_MODE
// TODO(zhangxu): inline asm to avoid movprfx
FUSED_CPP_ALWAYS_INLINE void process_f32_rows4_rope_chunk_sve(
    float* row0,
    float* row1,
    float* row2,
    float* row3,
    const float* cos_row,
    const float* sin_row,
    int64_t nope_head_dim,
    int64_t pair,
    svbool_t pg,
    svfloat32_t inv0,
    svfloat32_t inv1,
    svfloat32_t inv2,
    svfloat32_t inv3) {
  const svfloat32_t c = svld1_f32(pg, cos_row + pair);
  const svfloat32_t s = svld1_f32(pg, sin_row + pair);

  const svfloat32_t c_inv0 = svmul_f32_x(pg, inv0, c);
  const svfloat32_t c_inv1 = svmul_f32_x(pg, inv1, c);
  const svfloat32_t c_inv2 = svmul_f32_x(pg, inv2, c);
  const svfloat32_t c_inv3 = svmul_f32_x(pg, inv3, c);

  const svfloat32_t s_inv0 = svmul_f32_x(pg, inv0, s);
  const svfloat32_t s_inv1 = svmul_f32_x(pg, inv1, s);
  const svfloat32_t s_inv2 = svmul_f32_x(pg, inv2, s);
  const svfloat32_t s_inv3 = svmul_f32_x(pg, inv3, s);

  const svfloat32x2_t q_pair0 = svld2_f32(pg, row0 + nope_head_dim + 2 * pair);
  const svfloat32x2_t q_pair1 = svld2_f32(pg, row1 + nope_head_dim + 2 * pair);
  const svfloat32x2_t q_pair2 = svld2_f32(pg, row2 + nope_head_dim + 2 * pair);
  const svfloat32x2_t q_pair3 = svld2_f32(pg, row3 + nope_head_dim + 2 * pair);

  const svfloat32_t even0 = svget2_f32(q_pair0, 0);
  const svfloat32_t odd0 = svget2_f32(q_pair0, 1);
  const svfloat32_t even1 = svget2_f32(q_pair1, 0);
  const svfloat32_t odd1 = svget2_f32(q_pair1, 1);
  const svfloat32_t even2 = svget2_f32(q_pair2, 0);
  const svfloat32_t odd2 = svget2_f32(q_pair2, 1);
  const svfloat32_t even3 = svget2_f32(q_pair3, 0);
  const svfloat32_t odd3 = svget2_f32(q_pair3, 1);

  svfloat32x2_t out0 = svcreate2_f32(
      svmul_f32_x(pg, even0, c_inv0),
      svmul_f32_x(pg, odd0, c_inv0));
  svfloat32x2_t out1 = svcreate2_f32(
      svmul_f32_x(pg, even1, c_inv1),
      svmul_f32_x(pg, odd1, c_inv1));
  svfloat32x2_t out2 = svcreate2_f32(
      svmul_f32_x(pg, even2, c_inv2),
      svmul_f32_x(pg, odd2, c_inv2));
  svfloat32x2_t out3 = svcreate2_f32(
      svmul_f32_x(pg, even3, c_inv3),
      svmul_f32_x(pg, odd3, c_inv3));

  out0 = svcreate2_f32(
      svmls_f32_x(pg, svget2_f32(out0, 0), odd0, s_inv0),
      svmla_f32_x(pg, svget2_f32(out0, 1), even0, s_inv0));
  out1 = svcreate2_f32(
      svmls_f32_x(pg, svget2_f32(out1, 0), odd1, s_inv1),
      svmla_f32_x(pg, svget2_f32(out1, 1), even1, s_inv1));
  out2 = svcreate2_f32(
      svmls_f32_x(pg, svget2_f32(out2, 0), odd2, s_inv2),
      svmla_f32_x(pg, svget2_f32(out2, 1), even2, s_inv2));
  out3 = svcreate2_f32(
      svmls_f32_x(pg, svget2_f32(out3, 0), odd3, s_inv3),
      svmla_f32_x(pg, svget2_f32(out3, 1), even3, s_inv3));

  svst2_f32(pg, row0 + nope_head_dim + 2 * pair, out0);
  svst2_f32(pg, row1 + nope_head_dim + 2 * pair, out1);
  svst2_f32(pg, row2 + nope_head_dim + 2 * pair, out2);
  svst2_f32(pg, row3 + nope_head_dim + 2 * pair, out3);
}

void process_f32_rows4_sve(float* row0,
                           float* row1,
                           float* row2,
                           float* row3,
                           const float* cos_row,
                           const float* sin_row,
                           int64_t nope_head_dim,
                           int64_t rope_half,
                           float inv_rms0,
                           float inv_rms1,
                           float inv_rms2,
                           float inv_rms3) {
  const svfloat32_t inv0 = svdup_f32(inv_rms0);
  const svfloat32_t inv1 = svdup_f32(inv_rms1);
  const svfloat32_t inv2 = svdup_f32(inv_rms2);
  const svfloat32_t inv3 = svdup_f32(inv_rms3);

  for (int64_t d = 0; d < nope_head_dim; d += static_cast<int64_t>(svcntw())) {
    const svbool_t pg = svwhilelt_b32(d, nope_head_dim);
    const svfloat32_t v0 = svld1_f32(pg, row0 + d);
    const svfloat32_t v1 = svld1_f32(pg, row1 + d);
    const svfloat32_t v2 = svld1_f32(pg, row2 + d);
    const svfloat32_t v3 = svld1_f32(pg, row3 + d);
    svst1_f32(pg, row0 + d, svmul_f32_x(pg, v0, inv0));
    svst1_f32(pg, row1 + d, svmul_f32_x(pg, v1, inv1));
    svst1_f32(pg, row2 + d, svmul_f32_x(pg, v2, inv2));
    svst1_f32(pg, row3 + d, svmul_f32_x(pg, v3, inv3));
  }

  const int64_t vl = static_cast<int64_t>(svcntw());
  const svbool_t pg_all = svptrue_b32();
  int64_t pair = 0;
  for (; pair + vl <= rope_half; pair += vl) {
    process_f32_rows4_rope_chunk_sve(
        row0, row1, row2, row3, cos_row, sin_row, nope_head_dim, pair, pg_all,
        inv0, inv1, inv2, inv3);
  }
  if (pair < rope_half) {
    const svbool_t pg = svwhilelt_b32(pair, rope_half);
    process_f32_rows4_rope_chunk_sve(
        row0, row1, row2, row3, cos_row, sin_row, nope_head_dim, pair, pg,
        inv0, inv1, inv2, inv3);
  }
}
#endif

void process_bf16_row_sve(uint16_t* row,
                          const float* cos_row,
                          const float* sin_row,
                          int64_t head_dim,
                          int64_t nope_head_dim,
                          int64_t rope_half,
                          float inv_rms) {
  const svfloat32_t inv = svdup_f32(inv_rms);
  for (int64_t d = 0; d < nope_head_dim; d += static_cast<int64_t>(svcntw())) {
    const svbool_t pg = svwhilelt_b32(d, nope_head_dim);
    const svfloat32_t v = svmul_f32_x(pg, bf16_load_f32(pg, row + d), inv);
    bf16_store_f32(pg, row + d, v);
  }

  process_bf16_rope_row_sve(row, cos_row, sin_row, nope_head_dim, rope_half, inv_rms);
}

void process_bf16_rope_row_sve(uint16_t* row,
                               const float* cos_row,
                               const float* sin_row,
                               int64_t nope_head_dim,
                               int64_t rope_half,
                               float inv_rms) {
  const svfloat32_t inv = svdup_f32(inv_rms);
  const int64_t vl = static_cast<int64_t>(svcntw());
  for (int64_t pair = 0; pair < rope_half; pair += vl) {
    const int64_t active = pair + vl <= rope_half ? vl : rope_half - pair;
    const svbool_t pg = svwhilelt_b32(static_cast<int64_t>(0), active);
    const svbool_t pg_h = svwhilelt_b16(static_cast<int64_t>(0), active);
    const svfloat32x2_t q_pair =
        bf16_load2_evenodd_f32(pg, pg_h, row + nope_head_dim + 2 * pair);
    const svfloat32_t even = svmul_f32_x(pg, svget2_f32(q_pair, 0), inv);
    const svfloat32_t odd = svmul_f32_x(pg, svget2_f32(q_pair, 1), inv);
    const svfloat32_t c = svld1_f32(pg, cos_row + pair);
    const svfloat32_t s = svld1_f32(pg, sin_row + pair);
    const svfloat32_t out_even =
        svmls_f32_x(pg, svmul_f32_x(pg, even, c), odd, s);
    const svfloat32_t out_odd =
        svmla_f32_x(pg, svmul_f32_x(pg, odd, c), even, s);
    bf16_scatter_evenodd_f32(pg, row + nope_head_dim + 2 * pair, out_even);
    bf16_scatter_evenodd_f32(pg, row + nope_head_dim + 2 * pair + 1, out_odd);
  }
}

FUSED_CPP_ALWAYS_INLINE void process_bf16_rows4_rope_chunk_sve(
    uint16_t* row0,
    uint16_t* row1,
    uint16_t* row2,
    uint16_t* row3,
    const float* cos_row,
    const float* sin_row,
    int64_t nope_head_dim,
    int64_t pair,
    svbool_t pg,
    svbool_t pg_h,
    svfloat32_t inv0,
    svfloat32_t inv1,
    svfloat32_t inv2,
    svfloat32_t inv3) {
  const svfloat32_t c = svld1_f32(pg, cos_row + pair);
  const svfloat32_t s = svld1_f32(pg, sin_row + pair);

  const svfloat32x2_t q_pair0 =
      bf16_load2_evenodd_f32(pg, pg_h, row0 + nope_head_dim + 2 * pair);
  const svfloat32x2_t q_pair1 =
      bf16_load2_evenodd_f32(pg, pg_h, row1 + nope_head_dim + 2 * pair);
  const svfloat32x2_t q_pair2 =
      bf16_load2_evenodd_f32(pg, pg_h, row2 + nope_head_dim + 2 * pair);
  const svfloat32x2_t q_pair3 =
      bf16_load2_evenodd_f32(pg, pg_h, row3 + nope_head_dim + 2 * pair);

  const svfloat32_t even0 = svmul_f32_x(pg, svget2_f32(q_pair0, 0), inv0);
  const svfloat32_t odd0 = svmul_f32_x(pg, svget2_f32(q_pair0, 1), inv0);
  const svfloat32_t even1 = svmul_f32_x(pg, svget2_f32(q_pair1, 0), inv1);
  const svfloat32_t odd1 = svmul_f32_x(pg, svget2_f32(q_pair1, 1), inv1);
  const svfloat32_t even2 = svmul_f32_x(pg, svget2_f32(q_pair2, 0), inv2);
  const svfloat32_t odd2 = svmul_f32_x(pg, svget2_f32(q_pair2, 1), inv2);
  const svfloat32_t even3 = svmul_f32_x(pg, svget2_f32(q_pair3, 0), inv3);
  const svfloat32_t odd3 = svmul_f32_x(pg, svget2_f32(q_pair3, 1), inv3);

  svfloat32x2_t out0 = svcreate2_f32(
      svmul_f32_x(pg, even0, c),
      svmul_f32_x(pg, odd0, c));
  svfloat32x2_t out1 = svcreate2_f32(
      svmul_f32_x(pg, even1, c),
      svmul_f32_x(pg, odd1, c));
  svfloat32x2_t out2 = svcreate2_f32(
      svmul_f32_x(pg, even2, c),
      svmul_f32_x(pg, odd2, c));
  svfloat32x2_t out3 = svcreate2_f32(
      svmul_f32_x(pg, even3, c),
      svmul_f32_x(pg, odd3, c));

  out0 = svcreate2_f32(
      svmls_f32_x(pg, svget2_f32(out0, 0), odd0, s),
      svmla_f32_x(pg, svget2_f32(out0, 1), even0, s));
  out1 = svcreate2_f32(
      svmls_f32_x(pg, svget2_f32(out1, 0), odd1, s),
      svmla_f32_x(pg, svget2_f32(out1, 1), even1, s));
  out2 = svcreate2_f32(
      svmls_f32_x(pg, svget2_f32(out2, 0), odd2, s),
      svmla_f32_x(pg, svget2_f32(out2, 1), even2, s));
  out3 = svcreate2_f32(
      svmls_f32_x(pg, svget2_f32(out3, 0), odd3, s),
      svmla_f32_x(pg, svget2_f32(out3, 1), even3, s));

  bf16_scatter_evenodd_f32(pg, row0 + nope_head_dim + 2 * pair, svget2_f32(out0, 0));
  bf16_scatter_evenodd_f32(pg, row0 + nope_head_dim + 2 * pair + 1, svget2_f32(out0, 1));
  bf16_scatter_evenodd_f32(pg, row1 + nope_head_dim + 2 * pair, svget2_f32(out1, 0));
  bf16_scatter_evenodd_f32(pg, row1 + nope_head_dim + 2 * pair + 1, svget2_f32(out1, 1));
  bf16_scatter_evenodd_f32(pg, row2 + nope_head_dim + 2 * pair, svget2_f32(out2, 0));
  bf16_scatter_evenodd_f32(pg, row2 + nope_head_dim + 2 * pair + 1, svget2_f32(out2, 1));
  bf16_scatter_evenodd_f32(pg, row3 + nope_head_dim + 2 * pair, svget2_f32(out3, 0));
  bf16_scatter_evenodd_f32(pg, row3 + nope_head_dim + 2 * pair + 1, svget2_f32(out3, 1));
}

void process_bf16_rows4_sve(uint16_t* row0,
                            uint16_t* row1,
                            uint16_t* row2,
                            uint16_t* row3,
                            const float* cos_row,
                            const float* sin_row,
                            int64_t nope_head_dim,
                            int64_t rope_half,
                            float inv_rms0,
                            float inv_rms1,
                            float inv_rms2,
                            float inv_rms3) {
  const svfloat32_t inv0 = svdup_f32(inv_rms0);
  const svfloat32_t inv1 = svdup_f32(inv_rms1);
  const svfloat32_t inv2 = svdup_f32(inv_rms2);
  const svfloat32_t inv3 = svdup_f32(inv_rms3);

  for (int64_t d = 0; d < nope_head_dim; d += static_cast<int64_t>(svcntw())) {
    const svbool_t pg = svwhilelt_b32(d, nope_head_dim);
    const svfloat32_t v0 = svmul_f32_x(pg, bf16_load_f32(pg, row0 + d), inv0);
    const svfloat32_t v1 = svmul_f32_x(pg, bf16_load_f32(pg, row1 + d), inv1);
    const svfloat32_t v2 = svmul_f32_x(pg, bf16_load_f32(pg, row2 + d), inv2);
    const svfloat32_t v3 = svmul_f32_x(pg, bf16_load_f32(pg, row3 + d), inv3);
    bf16_store_f32(pg, row0 + d, v0);
    bf16_store_f32(pg, row1 + d, v1);
    bf16_store_f32(pg, row2 + d, v2);
    bf16_store_f32(pg, row3 + d, v3);
  }

  const int64_t vl = static_cast<int64_t>(svcntw());
  const svbool_t pg_all = svptrue_b32();
  const svbool_t pg_all_h = svwhilelt_b16(static_cast<int64_t>(0), vl);
  int64_t pair = 0;
  for (; pair + vl <= rope_half; pair += vl) {
    process_bf16_rows4_rope_chunk_sve(
        row0,
        row1,
        row2,
        row3,
        cos_row,
        sin_row,
        nope_head_dim,
        pair,
        pg_all,
        pg_all_h,
        inv0,
        inv1,
        inv2,
        inv3);
  }
  if (pair < rope_half) {
    const int64_t active = rope_half - pair;
    const svbool_t pg = svwhilelt_b32(static_cast<int64_t>(0), active);
    const svbool_t pg_h = svwhilelt_b16(static_cast<int64_t>(0), active);
    process_bf16_rows4_rope_chunk_sve(
        row0,
        row1,
        row2,
        row3,
        cos_row,
        sin_row,
        nope_head_dim,
        pair,
        pg,
        pg_h,
        inv0,
        inv1,
        inv2,
        inv3);
  }
}

FUSED_CPP_ALWAYS_INLINE void process_f32_kv_rows2_rope_chunk_sve(
    float* row0,
    float* row1,
    const float* cos_row0,
    const float* sin_row0,
    const float* cos_row1,
    const float* sin_row1,
    int64_t nope_head_dim,
    int64_t pair,
    svbool_t pg) {
  const svfloat32_t c0 = svld1_f32(pg, cos_row0 + pair);
  const svfloat32_t s0 = svld1_f32(pg, sin_row0 + pair);
  const svfloat32_t c1 = svld1_f32(pg, cos_row1 + pair);
  const svfloat32_t s1 = svld1_f32(pg, sin_row1 + pair);

  const svfloat32x2_t kv_pair0 = svld2_f32(pg, row0 + nope_head_dim + 2 * pair);
  const svfloat32x2_t kv_pair1 = svld2_f32(pg, row1 + nope_head_dim + 2 * pair);

  const svfloat32_t even0 = svget2_f32(kv_pair0, 0);
  const svfloat32_t odd0 = svget2_f32(kv_pair0, 1);
  const svfloat32_t even1 = svget2_f32(kv_pair1, 0);
  const svfloat32_t odd1 = svget2_f32(kv_pair1, 1);

  svfloat32x2_t out0 = svcreate2_f32(
      svmul_f32_x(pg, even0, c0),
      svmul_f32_x(pg, odd0, c0));
  svfloat32x2_t out1 = svcreate2_f32(
      svmul_f32_x(pg, even1, c1),
      svmul_f32_x(pg, odd1, c1));

  out0 = svcreate2_f32(
      svmls_f32_x(pg, svget2_f32(out0, 0), odd0, s0),
      svmla_f32_x(pg, svget2_f32(out0, 1), even0, s0));
  out1 = svcreate2_f32(
      svmls_f32_x(pg, svget2_f32(out1, 0), odd1, s1),
      svmla_f32_x(pg, svget2_f32(out1, 1), even1, s1));

  svst2_f32(pg, row0 + nope_head_dim + 2 * pair, out0);
  svst2_f32(pg, row1 + nope_head_dim + 2 * pair, out1);
}

void process_f32_kv_rows2_rope_sve(float* row0,
                                   float* row1,
                                   const float* cos_row0,
                                   const float* sin_row0,
                                   const float* cos_row1,
                                   const float* sin_row1,
                                   int64_t nope_head_dim,
                                   int64_t rope_half) {
  const int64_t vl = static_cast<int64_t>(svcntw());
  const svbool_t pg_all = svptrue_b32();
  int64_t pair = 0;
  for (; pair + vl <= rope_half; pair += vl) {
    process_f32_kv_rows2_rope_chunk_sve(
        row0, row1, cos_row0, sin_row0, cos_row1, sin_row1, nope_head_dim, pair, pg_all);
  }
  if (pair < rope_half) {
    const svbool_t pg = svwhilelt_b32(pair, rope_half);
    process_f32_kv_rows2_rope_chunk_sve(
        row0, row1, cos_row0, sin_row0, cos_row1, sin_row1, nope_head_dim, pair, pg);
  }
}

FUSED_CPP_ALWAYS_INLINE void process_f32_kv_rows4_rope_chunk_sve(
    float* row0,
    float* row1,
    float* row2,
    float* row3,
    const float* cos_row0,
    const float* sin_row0,
    const float* cos_row1,
    const float* sin_row1,
    const float* cos_row2,
    const float* sin_row2,
    const float* cos_row3,
    const float* sin_row3,
    int64_t nope_head_dim,
    int64_t pair,
    svbool_t pg) {
  svfloat32x2_t out0;
  svfloat32x2_t out1;
  svfloat32x2_t out2;
  svfloat32x2_t out3;
  compute_kv_rows4_rope_chunk_sve(
      row0,
      row1,
      row2,
      row3,
      cos_row0,
      sin_row0,
      cos_row1,
      sin_row1,
      cos_row2,
      sin_row2,
      cos_row3,
      sin_row3,
      nope_head_dim,
      pair,
      pg,
      pg,
      out0,
      out1,
      out2,
      out3);
  svst2_f32(pg, row0 + nope_head_dim + 2 * pair, out0);
  svst2_f32(pg, row1 + nope_head_dim + 2 * pair, out1);
  svst2_f32(pg, row2 + nope_head_dim + 2 * pair, out2);
  svst2_f32(pg, row3 + nope_head_dim + 2 * pair, out3);
}

void process_f32_kv_rows4_rope_sve(float* row0,
                                   float* row1,
                                   float* row2,
                                   float* row3,
                                   const float* cos_row0,
                                   const float* sin_row0,
                                   const float* cos_row1,
                                   const float* sin_row1,
                                   const float* cos_row2,
                                   const float* sin_row2,
                                   const float* cos_row3,
                                   const float* sin_row3,
                                   int64_t nope_head_dim,
                                   int64_t rope_half) {
  const int64_t vl = static_cast<int64_t>(svcntw());
  const svbool_t pg_all = svptrue_b32();
  int64_t pair = 0;
  for (; pair + vl <= rope_half; pair += vl) {
    process_f32_kv_rows4_rope_chunk_sve(
        row0,
        row1,
        row2,
        row3,
        cos_row0,
        sin_row0,
        cos_row1,
        sin_row1,
        cos_row2,
        sin_row2,
        cos_row3,
        sin_row3,
        nope_head_dim,
        pair,
        pg_all);
  }
  if (pair < rope_half) {
    const svbool_t pg = svwhilelt_b32(pair, rope_half);
    process_f32_kv_rows4_rope_chunk_sve(
        row0,
        row1,
        row2,
        row3,
        cos_row0,
        sin_row0,
        cos_row1,
        sin_row1,
        cos_row2,
        sin_row2,
        cos_row3,
        sin_row3,
        nope_head_dim,
        pair,
        pg);
  }
}

FUSED_CPP_ALWAYS_INLINE void process_bf16_kv_rows2_rope_chunk_sve(
    uint16_t* row0,
    uint16_t* row1,
    const float* cos_row0,
    const float* sin_row0,
    const float* cos_row1,
    const float* sin_row1,
    int64_t nope_head_dim,
    int64_t pair,
    svbool_t pg,
    svbool_t pg_h) {
  const svfloat32x2_t kv_pair0 =
      bf16_load2_evenodd_f32(pg, pg_h, row0 + nope_head_dim + 2 * pair);
  const svfloat32x2_t kv_pair1 =
      bf16_load2_evenodd_f32(pg, pg_h, row1 + nope_head_dim + 2 * pair);
  const svfloat32_t even0 = svget2_f32(kv_pair0, 0);
  const svfloat32_t odd0 = svget2_f32(kv_pair0, 1);
  const svfloat32_t even1 = svget2_f32(kv_pair1, 0);
  const svfloat32_t odd1 = svget2_f32(kv_pair1, 1);
  const svfloat32_t c0 = svld1_f32(pg, cos_row0 + pair);
  const svfloat32_t s0 = svld1_f32(pg, sin_row0 + pair);
  const svfloat32_t c1 = svld1_f32(pg, cos_row1 + pair);
  const svfloat32_t s1 = svld1_f32(pg, sin_row1 + pair);

  svfloat32x2_t out0 = svcreate2_f32(
      svmul_f32_x(pg, even0, c0),
      svmul_f32_x(pg, odd0, c0));
  svfloat32x2_t out1 = svcreate2_f32(
      svmul_f32_x(pg, even1, c1),
      svmul_f32_x(pg, odd1, c1));

  out0 = svcreate2_f32(
      svmls_f32_x(pg, svget2_f32(out0, 0), odd0, s0),
      svmla_f32_x(pg, svget2_f32(out0, 1), even0, s0));
  out1 = svcreate2_f32(
      svmls_f32_x(pg, svget2_f32(out1, 0), odd1, s1),
      svmla_f32_x(pg, svget2_f32(out1, 1), even1, s1));

  bf16_scatter_evenodd_f32(pg, row0 + nope_head_dim + 2 * pair, svget2_f32(out0, 0));
  bf16_scatter_evenodd_f32(pg, row0 + nope_head_dim + 2 * pair + 1, svget2_f32(out0, 1));
  bf16_scatter_evenodd_f32(pg, row1 + nope_head_dim + 2 * pair, svget2_f32(out1, 0));
  bf16_scatter_evenodd_f32(pg, row1 + nope_head_dim + 2 * pair + 1, svget2_f32(out1, 1));
}

void process_bf16_kv_rows2_rope_sve(uint16_t* row0,
                                    uint16_t* row1,
                                    const float* cos_row0,
                                    const float* sin_row0,
                                    const float* cos_row1,
                                    const float* sin_row1,
                                    int64_t nope_head_dim,
                                    int64_t rope_half) {
  const int64_t vl = static_cast<int64_t>(svcntw());
  const svbool_t pg_all = svptrue_b32();
  const svbool_t pg_all_h = svwhilelt_b16(static_cast<int64_t>(0), vl);
  int64_t pair = 0;
  for (; pair + vl <= rope_half; pair += vl) {
    process_bf16_kv_rows2_rope_chunk_sve(
        row0,
        row1,
        cos_row0,
        sin_row0,
        cos_row1,
        sin_row1,
        nope_head_dim,
        pair,
        pg_all,
        pg_all_h);
  }
  if (pair < rope_half) {
    const int64_t active = rope_half - pair;
    const svbool_t pg = svwhilelt_b32(static_cast<int64_t>(0), active);
    const svbool_t pg_h = svwhilelt_b16(static_cast<int64_t>(0), active);
    process_bf16_kv_rows2_rope_chunk_sve(
        row0,
        row1,
        cos_row0,
        sin_row0,
        cos_row1,
        sin_row1,
        nope_head_dim,
        pair,
        pg,
        pg_h);
  }
}

FUSED_CPP_ALWAYS_INLINE void process_bf16_kv_rows4_rope_chunk_sve(
    uint16_t* row0,
    uint16_t* row1,
    uint16_t* row2,
    uint16_t* row3,
    const float* cos_row0,
    const float* sin_row0,
    const float* cos_row1,
    const float* sin_row1,
    const float* cos_row2,
    const float* sin_row2,
    const float* cos_row3,
    const float* sin_row3,
    int64_t nope_head_dim,
    int64_t pair,
    svbool_t pg,
    svbool_t pg_h) {
  svfloat32x2_t out0;
  svfloat32x2_t out1;
  svfloat32x2_t out2;
  svfloat32x2_t out3;
  compute_kv_rows4_rope_chunk_sve(
      row0,
      row1,
      row2,
      row3,
      cos_row0,
      sin_row0,
      cos_row1,
      sin_row1,
      cos_row2,
      sin_row2,
      cos_row3,
      sin_row3,
      nope_head_dim,
      pair,
      pg,
      pg_h,
      out0,
      out1,
      out2,
      out3);
  bf16_scatter_evenodd_f32(pg, row0 + nope_head_dim + 2 * pair, svget2_f32(out0, 0));
  bf16_scatter_evenodd_f32(pg, row0 + nope_head_dim + 2 * pair + 1, svget2_f32(out0, 1));
  bf16_scatter_evenodd_f32(pg, row1 + nope_head_dim + 2 * pair, svget2_f32(out1, 0));
  bf16_scatter_evenodd_f32(pg, row1 + nope_head_dim + 2 * pair + 1, svget2_f32(out1, 1));
  bf16_scatter_evenodd_f32(pg, row2 + nope_head_dim + 2 * pair, svget2_f32(out2, 0));
  bf16_scatter_evenodd_f32(pg, row2 + nope_head_dim + 2 * pair + 1, svget2_f32(out2, 1));
  bf16_scatter_evenodd_f32(pg, row3 + nope_head_dim + 2 * pair, svget2_f32(out3, 0));
  bf16_scatter_evenodd_f32(pg, row3 + nope_head_dim + 2 * pair + 1, svget2_f32(out3, 1));
}

void process_bf16_kv_rows4_rope_sve(uint16_t* row0,
                                    uint16_t* row1,
                                    uint16_t* row2,
                                    uint16_t* row3,
                                    const float* cos_row0,
                                    const float* sin_row0,
                                    const float* cos_row1,
                                    const float* sin_row1,
                                    const float* cos_row2,
                                    const float* sin_row2,
                                    const float* cos_row3,
                                    const float* sin_row3,
                                    int64_t nope_head_dim,
                                    int64_t rope_half) {
  const int64_t vl = static_cast<int64_t>(svcntw());
  const svbool_t pg_all = svptrue_b32();
  const svbool_t pg_all_h = svwhilelt_b16(static_cast<int64_t>(0), vl);
  int64_t pair = 0;
  for (; pair + vl <= rope_half; pair += vl) {
    process_bf16_kv_rows4_rope_chunk_sve(
        row0,
        row1,
        row2,
        row3,
        cos_row0,
        sin_row0,
        cos_row1,
        sin_row1,
        cos_row2,
        sin_row2,
        cos_row3,
        sin_row3,
        nope_head_dim,
        pair,
        pg_all,
        pg_all_h);
  }
  if (pair < rope_half) {
    const int64_t active = rope_half - pair;
    const svbool_t pg = svwhilelt_b32(static_cast<int64_t>(0), active);
    const svbool_t pg_h = svwhilelt_b16(static_cast<int64_t>(0), active);
    process_bf16_kv_rows4_rope_chunk_sve(
        row0,
        row1,
        row2,
        row3,
        cos_row0,
        sin_row0,
        cos_row1,
        sin_row1,
        cos_row2,
        sin_row2,
        cos_row3,
        sin_row3,
        nope_head_dim,
        pair,
        pg,
        pg_h);
  }
}

template <typename row_t, typename cache_t>
void copy_kv_nope_to_cache_sve(const row_t* row, cache_t* cache_row, int64_t nope_head_dim) {
  const int64_t vl = static_cast<int64_t>(svcntw());
  for (int64_t d = 0; d < nope_head_dim; d += vl) {
    const svbool_t pg = svwhilelt_b32(d, nope_head_dim);
    const svfloat32_t v = load_contiguous_f32(pg, row + d);
    store_contiguous_f32(pg, cache_row + d, v);
  }
}

template <typename row_t, typename cache_t>
void process_kv_rope_row_to_cache_sve(const row_t* row,
                                      cache_t* cache_row,
                                      const float* cos_row,
                                      const float* sin_row,
                                      int64_t nope_head_dim,
                                      int64_t rope_half) {
  const int64_t vl = static_cast<int64_t>(svcntw());
  for (int64_t pair = 0; pair < rope_half; pair += vl) {
    const int64_t active = std::min<int64_t>(vl, rope_half - pair);
    const svbool_t pg = svwhilelt_b32(static_cast<int64_t>(0), active);
    const svbool_t pg_h = svwhilelt_b16(static_cast<int64_t>(0), active);
    const int64_t offset = nope_head_dim + 2 * pair;
    svfloat32_t even;
    svfloat32_t odd;
    load_evenodd_contiguous_f32(pg, pg_h, row + offset, even, odd);
    const svfloat32_t c = svld1_f32(pg, cos_row + pair);
    const svfloat32_t s = svld1_f32(pg, sin_row + pair);
    const svfloat32_t out_even =
        svmls_f32_x(pg, svmul_f32_x(pg, even, c), odd, s);
    const svfloat32_t out_odd =
        svmla_f32_x(pg, svmul_f32_x(pg, odd, c), even, s);
    store_evenodd_contiguous_f32(pg, cache_row + offset, out_even, out_odd);
  }
}

template <typename row_t, typename cache_t>
FUSED_CPP_ALWAYS_INLINE void copy_kv_nope_rows2_to_cache_sve(
    const row_t* row0,
    const row_t* row1,
    cache_t* cache_row0,
    cache_t* cache_row1,
    int64_t nope_head_dim) {
  const int64_t vl = static_cast<int64_t>(svcntw());
  for (int64_t d = 0; d < nope_head_dim; d += vl) {
    const svbool_t pg = svwhilelt_b32(d, nope_head_dim);
    const svfloat32_t v0 = load_contiguous_f32(pg, row0 + d);
    const svfloat32_t v1 = load_contiguous_f32(pg, row1 + d);
    store_contiguous_f32(pg, cache_row0 + d, v0);
    store_contiguous_f32(pg, cache_row1 + d, v1);
  }
}

template <typename row_t, typename cache_t>
FUSED_CPP_ALWAYS_INLINE void copy_kv_nope_rows4_to_cache_sve(
    const row_t* row0,
    const row_t* row1,
    const row_t* row2,
    const row_t* row3,
    cache_t* cache_row0,
    cache_t* cache_row1,
    cache_t* cache_row2,
    cache_t* cache_row3,
    int64_t nope_head_dim) {
  const int64_t vl = static_cast<int64_t>(svcntw());
  for (int64_t d = 0; d < nope_head_dim; d += vl) {
    const svbool_t pg = svwhilelt_b32(d, nope_head_dim);
    const svfloat32_t v0 = load_contiguous_f32(pg, row0 + d);
    const svfloat32_t v1 = load_contiguous_f32(pg, row1 + d);
    const svfloat32_t v2 = load_contiguous_f32(pg, row2 + d);
    const svfloat32_t v3 = load_contiguous_f32(pg, row3 + d);
    store_contiguous_f32(pg, cache_row0 + d, v0);
    store_contiguous_f32(pg, cache_row1 + d, v1);
    store_contiguous_f32(pg, cache_row2 + d, v2);
    store_contiguous_f32(pg, cache_row3 + d, v3);
  }
}

template <typename row_t, typename cache_t>
FUSED_CPP_ALWAYS_INLINE void process_kv_rows2_rope_to_cache_chunk_sve(
    const row_t* row0,
    const row_t* row1,
    cache_t* cache_row0,
    cache_t* cache_row1,
    const float* cos_row0,
    const float* sin_row0,
    const float* cos_row1,
    const float* sin_row1,
    int64_t nope_head_dim,
    int64_t pair,
    svbool_t pg,
    svbool_t pg_h) {
  const svfloat32_t c0 = svld1_f32(pg, cos_row0 + pair);
  const svfloat32_t s0 = svld1_f32(pg, sin_row0 + pair);
  const svfloat32_t c1 = svld1_f32(pg, cos_row1 + pair);
  const svfloat32_t s1 = svld1_f32(pg, sin_row1 + pair);

  const int64_t offset = nope_head_dim + 2 * pair;
  svfloat32_t even0;
  svfloat32_t odd0;
  svfloat32_t even1;
  svfloat32_t odd1;
  load_evenodd_contiguous_f32(pg, pg_h, row0 + offset, even0, odd0);
  load_evenodd_contiguous_f32(pg, pg_h, row1 + offset, even1, odd1);

  svfloat32x2_t out0 = svcreate2_f32(
      svmul_f32_x(pg, even0, c0),
      svmul_f32_x(pg, odd0, c0));
  svfloat32x2_t out1 = svcreate2_f32(
      svmul_f32_x(pg, even1, c1),
      svmul_f32_x(pg, odd1, c1));

  out0 = svcreate2_f32(
      svmls_f32_x(pg, svget2_f32(out0, 0), odd0, s0),
      svmla_f32_x(pg, svget2_f32(out0, 1), even0, s0));
  out1 = svcreate2_f32(
      svmls_f32_x(pg, svget2_f32(out1, 0), odd1, s1),
      svmla_f32_x(pg, svget2_f32(out1, 1), even1, s1));

  store_evenodd_contiguous_f32(
      pg, cache_row0 + offset, svget2_f32(out0, 0), svget2_f32(out0, 1));
  store_evenodd_contiguous_f32(
      pg, cache_row1 + offset, svget2_f32(out1, 0), svget2_f32(out1, 1));
}

template <typename row_t, typename cache_t>
void process_kv_rows2_rope_to_cache_sve(const row_t* row0,
                                        const row_t* row1,
                                        cache_t* cache_row0,
                                        cache_t* cache_row1,
                                        const float* cos_row0,
                                        const float* sin_row0,
                                        const float* cos_row1,
                                        const float* sin_row1,
                                        int64_t nope_head_dim,
                                        int64_t rope_half) {
  const int64_t vl = static_cast<int64_t>(svcntw());
  const svbool_t pg_all = svptrue_b32();
  const svbool_t pg_all_h = svwhilelt_b16(static_cast<int64_t>(0), vl);
  int64_t pair = 0;
  for (; pair + vl <= rope_half; pair += vl) {
    process_kv_rows2_rope_to_cache_chunk_sve(
        row0,
        row1,
        cache_row0,
        cache_row1,
        cos_row0,
        sin_row0,
        cos_row1,
        sin_row1,
        nope_head_dim,
        pair,
        pg_all,
        pg_all_h);
  }
  if (pair < rope_half) {
    const int64_t active = rope_half - pair;
    const svbool_t pg = svwhilelt_b32(static_cast<int64_t>(0), active);
    const svbool_t pg_h = svwhilelt_b16(static_cast<int64_t>(0), active);
    process_kv_rows2_rope_to_cache_chunk_sve(
        row0,
        row1,
        cache_row0,
        cache_row1,
        cos_row0,
        sin_row0,
        cos_row1,
        sin_row1,
        nope_head_dim,
        pair,
        pg,
        pg_h);
  }
}

template <typename row_t, typename cache_t>
FUSED_CPP_ALWAYS_INLINE void process_kv_rows4_rope_to_cache_chunk_sve(
    const row_t* row0,
    const row_t* row1,
    const row_t* row2,
    const row_t* row3,
    cache_t* cache_row0,
    cache_t* cache_row1,
    cache_t* cache_row2,
    cache_t* cache_row3,
    const float* cos_row0,
    const float* sin_row0,
    const float* cos_row1,
    const float* sin_row1,
    const float* cos_row2,
    const float* sin_row2,
    const float* cos_row3,
    const float* sin_row3,
    int64_t nope_head_dim,
    int64_t pair,
    svbool_t pg,
    svbool_t pg_h) {
  svfloat32x2_t out0;
  svfloat32x2_t out1;
  svfloat32x2_t out2;
  svfloat32x2_t out3;
  compute_kv_rows4_rope_chunk_sve(
      row0,
      row1,
      row2,
      row3,
      cos_row0,
      sin_row0,
      cos_row1,
      sin_row1,
      cos_row2,
      sin_row2,
      cos_row3,
      sin_row3,
      nope_head_dim,
      pair,
      pg,
      pg_h,
      out0,
      out1,
      out2,
      out3);
  const int64_t offset = nope_head_dim + 2 * pair;
  store_evenodd_contiguous_f32(
      pg, cache_row0 + offset, svget2_f32(out0, 0), svget2_f32(out0, 1));
  store_evenodd_contiguous_f32(
      pg, cache_row1 + offset, svget2_f32(out1, 0), svget2_f32(out1, 1));
  store_evenodd_contiguous_f32(
      pg, cache_row2 + offset, svget2_f32(out2, 0), svget2_f32(out2, 1));
  store_evenodd_contiguous_f32(
      pg, cache_row3 + offset, svget2_f32(out3, 0), svget2_f32(out3, 1));
}

template <typename row_t, typename cache_t>
void process_kv_rows4_rope_to_cache_sve(const row_t* row0,
                                        const row_t* row1,
                                        const row_t* row2,
                                        const row_t* row3,
                                        cache_t* cache_row0,
                                        cache_t* cache_row1,
                                        cache_t* cache_row2,
                                        cache_t* cache_row3,
                                        const float* cos_row0,
                                        const float* sin_row0,
                                        const float* cos_row1,
                                        const float* sin_row1,
                                        const float* cos_row2,
                                        const float* sin_row2,
                                        const float* cos_row3,
                                        const float* sin_row3,
                                        int64_t nope_head_dim,
                                        int64_t rope_half) {
  const int64_t vl = static_cast<int64_t>(svcntw());
  const svbool_t pg_all = svptrue_b32();
  const svbool_t pg_all_h = svwhilelt_b16(static_cast<int64_t>(0), vl);
  int64_t pair = 0;
  for (; pair + vl <= rope_half; pair += vl) {
    process_kv_rows4_rope_to_cache_chunk_sve(
        row0,
        row1,
        row2,
        row3,
        cache_row0,
        cache_row1,
        cache_row2,
        cache_row3,
        cos_row0,
        sin_row0,
        cos_row1,
        sin_row1,
        cos_row2,
        sin_row2,
        cos_row3,
        sin_row3,
        nope_head_dim,
        pair,
        pg_all,
        pg_all_h);
  }
  if (pair < rope_half) {
    const int64_t active = rope_half - pair;
    const svbool_t pg = svwhilelt_b32(static_cast<int64_t>(0), active);
    const svbool_t pg_h = svwhilelt_b16(static_cast<int64_t>(0), active);
    process_kv_rows4_rope_to_cache_chunk_sve(
        row0,
        row1,
        row2,
        row3,
        cache_row0,
        cache_row1,
        cache_row2,
        cache_row3,
        cos_row0,
        sin_row0,
        cos_row1,
        sin_row1,
        cos_row2,
        sin_row2,
        cos_row3,
        sin_row3,
        nope_head_dim,
        pair,
        pg,
        pg_h);
  }
}

template <typename row_t>
void q_norm_rope_fused_sve_impl(row_t* q_data,
                                const int64_t* pos_data,
                                int64_t scalar_pos,
                                bool scalar_position,
                                const float* cos_sin_data,
                                int64_t cos_sin_stride,
                                int64_t num_tokens,
                                int64_t num_heads,
                                int64_t head_dim,
                                int64_t nope_head_dim,
                                int64_t rope_half,
                                int64_t q_stride_t,
                                int64_t q_stride_h,
                                double eps) {
#if !FUSED_CPP_STRICT_MODE
  if constexpr (std::is_same_v<row_t, float>) {
    const int64_t head_groups = num_heads / 4;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int64_t token = 0; token < num_tokens; ++token) {
      for (int64_t head_group = 0; head_group < head_groups; ++head_group) {
        const int64_t head_base = head_group * 4;
        float* row0 = q_data + token * q_stride_t + (head_base + 0) * q_stride_h;
        float* row1 = q_data + token * q_stride_t + (head_base + 1) * q_stride_h;
        float* row2 = q_data + token * q_stride_t + (head_base + 2) * q_stride_h;
        float* row3 = q_data + token * q_stride_t + (head_base + 3) * q_stride_h;
        const float* cos_row = nullptr;
        const float* sin_row = nullptr;
        if (rope_half != 0) {
          const int64_t pos = scalar_position ? scalar_pos : pos_data[token];
          const float* cs_row = cos_sin_data + pos * cos_sin_stride;
          cos_row = cs_row;
          sin_row = cs_row + rope_half;
        }

        float sum0;
        float sum1;
        float sum2;
        float sum3;
        sum_sq_f32_sve_4x4(row0, row1, row2, row3, head_dim, sum0, sum1, sum2, sum3);
        const float scale = 1.0f / static_cast<float>(head_dim);
        const float eps_f = static_cast<float>(eps);
        const float inv0 = 1.0f / std::sqrt(sum0 * scale + eps_f);
        const float inv1 = 1.0f / std::sqrt(sum1 * scale + eps_f);
        const float inv2 = 1.0f / std::sqrt(sum2 * scale + eps_f);
        const float inv3 = 1.0f / std::sqrt(sum3 * scale + eps_f);
        process_f32_rows4_sve(
            row0,
            row1,
            row2,
            row3,
            cos_row,
            sin_row,
            nope_head_dim,
            rope_half,
            inv0,
            inv1,
            inv2,
            inv3);
      }
    }

    const int64_t tail_head_start = head_groups * 4;
    if (tail_head_start < num_heads) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
      for (int64_t token = 0; token < num_tokens; ++token) {
        for (int64_t head = tail_head_start; head < num_heads; ++head) {
          float* row = q_data + token * q_stride_t + head * q_stride_h;
          const float* cos_row = nullptr;
          const float* sin_row = nullptr;
          if (rope_half != 0) {
            const int64_t pos = scalar_position ? scalar_pos : pos_data[token];
            const float* cs_row = cos_sin_data + pos * cos_sin_stride;
            cos_row = cs_row;
            sin_row = cs_row + rope_half;
          }
          const float sum_sq = sum_sq_f32_sve(row, head_dim);
          const float inv_rms = 1.0f / std::sqrt(
              sum_sq / static_cast<float>(head_dim) + static_cast<float>(eps));
          process_f32_row_sve(row, cos_row, sin_row, head_dim, nope_head_dim, rope_half, inv_rms);
        }
      }
    }
  } else
#endif
  {
    const int64_t head_groups = num_heads / 4;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int64_t token = 0; token < num_tokens; ++token) {
      for (int64_t head_group = 0; head_group < head_groups; ++head_group) {
        const int64_t head_base = head_group * 4;
        row_t* row0 = q_data + token * q_stride_t + (head_base + 0) * q_stride_h;
        row_t* row1 = q_data + token * q_stride_t + (head_base + 1) * q_stride_h;
        row_t* row2 = q_data + token * q_stride_t + (head_base + 2) * q_stride_h;
        row_t* row3 = q_data + token * q_stride_t + (head_base + 3) * q_stride_h;
        const float* cos_row = nullptr;
        const float* sin_row = nullptr;
        if (rope_half != 0) {
          const int64_t pos = scalar_position ? scalar_pos : pos_data[token];
          const float* cs_row = cos_sin_data + pos * cos_sin_stride;
          cos_row = cs_row;
          sin_row = cs_row + rope_half;
        }

        float sum0;
        float sum1;
        float sum2;
        float sum3;
        sum_sq_bf16_sve_4x4(row0, row1, row2, row3, head_dim, sum0, sum1, sum2, sum3);
        const float scale = 1.0f / static_cast<float>(head_dim);
        const float eps_f = static_cast<float>(eps);
        const float inv0 = 1.0f / std::sqrt(sum0 * scale + eps_f);
        const float inv1 = 1.0f / std::sqrt(sum1 * scale + eps_f);
        const float inv2 = 1.0f / std::sqrt(sum2 * scale + eps_f);
        const float inv3 = 1.0f / std::sqrt(sum3 * scale + eps_f);
        process_bf16_rows4_sve(
            row0,
            row1,
            row2,
            row3,
            cos_row,
            sin_row,
            nope_head_dim,
            rope_half,
            inv0,
            inv1,
            inv2,
            inv3);
      }
    }

    const int64_t tail_head_start = head_groups * 4;
    if (tail_head_start < num_heads) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
      for (int64_t token = 0; token < num_tokens; ++token) {
        for (int64_t head = tail_head_start; head < num_heads; ++head) {
          row_t* row = q_data + token * q_stride_t + head * q_stride_h;
          const float* cos_row = nullptr;
          const float* sin_row = nullptr;
          if (rope_half != 0) {
            const int64_t pos = scalar_position ? scalar_pos : pos_data[token];
            const float* cs_row = cos_sin_data + pos * cos_sin_stride;
            cos_row = cs_row;
            sin_row = cs_row + rope_half;
          }

          const float sum_sq = sum_sq_bf16_sve(row, head_dim);
          const float inv_rms = 1.0f / std::sqrt(
              sum_sq / static_cast<float>(head_dim) + static_cast<float>(eps));
          process_bf16_row_sve(row, cos_row, sin_row, head_dim, nope_head_dim, rope_half, inv_rms);
        }
      }
    }
  }
}

template <typename row_t>
void kv_rope_fused_sve_impl(row_t* kv_data,
                            const int64_t* pos_data,
                            int64_t scalar_pos,
                            bool scalar_position,
                            const float* cos_sin_data,
                            int64_t cos_sin_stride,
                            int64_t num_tokens,
                            int64_t nope_head_dim,
                            int64_t rope_half,
                            int64_t kv_stride_t) {
  const int64_t token_groups4 = num_tokens / 4;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t token_group = 0; token_group < token_groups4; ++token_group) {
    const int64_t token0 = token_group * 4;
    const int64_t token1 = token0 + 1;
    const int64_t token2 = token0 + 2;
    const int64_t token3 = token0 + 3;
    row_t* row0 = kv_data + token0 * kv_stride_t;
    row_t* row1 = kv_data + token1 * kv_stride_t;
    row_t* row2 = kv_data + token2 * kv_stride_t;
    row_t* row3 = kv_data + token3 * kv_stride_t;
    const int64_t pos0 = scalar_position ? scalar_pos : pos_data[token0];
    const int64_t pos1 = scalar_position ? scalar_pos : pos_data[token1];
    const int64_t pos2 = scalar_position ? scalar_pos : pos_data[token2];
    const int64_t pos3 = scalar_position ? scalar_pos : pos_data[token3];
    const float* cs_row0 = cos_sin_data + pos0 * cos_sin_stride;
    const float* cs_row1 = cos_sin_data + pos1 * cos_sin_stride;
    const float* cs_row2 = cos_sin_data + pos2 * cos_sin_stride;
    const float* cs_row3 = cos_sin_data + pos3 * cos_sin_stride;
    const float* cos_row0 = cs_row0;
    const float* sin_row0 = cs_row0 + rope_half;
    const float* cos_row1 = cs_row1;
    const float* sin_row1 = cs_row1 + rope_half;
    const float* cos_row2 = cs_row2;
    const float* sin_row2 = cs_row2 + rope_half;
    const float* cos_row3 = cs_row3;
    const float* sin_row3 = cs_row3 + rope_half;
    if constexpr (std::is_same_v<row_t, float>) {
      process_f32_kv_rows4_rope_sve(
          row0,
          row1,
          row2,
          row3,
          cos_row0,
          sin_row0,
          cos_row1,
          sin_row1,
          cos_row2,
          sin_row2,
          cos_row3,
          sin_row3,
          nope_head_dim,
          rope_half);
    } else {
      process_bf16_kv_rows4_rope_sve(
          row0,
          row1,
          row2,
          row3,
          cos_row0,
          sin_row0,
          cos_row1,
          sin_row1,
          cos_row2,
          sin_row2,
          cos_row3,
          sin_row3,
          nope_head_dim,
          rope_half);
    }
  }

  int64_t tail_token = token_groups4 * 4;
  if (tail_token + 1 < num_tokens) {
    row_t* row0 = kv_data + tail_token * kv_stride_t;
    row_t* row1 = kv_data + (tail_token + 1) * kv_stride_t;
    const int64_t pos0 = scalar_position ? scalar_pos : pos_data[tail_token];
    const int64_t pos1 = scalar_position ? scalar_pos : pos_data[tail_token + 1];
    const float* cs_row0 = cos_sin_data + pos0 * cos_sin_stride;
    const float* cs_row1 = cos_sin_data + pos1 * cos_sin_stride;
    const float* cos_row0 = cs_row0;
    const float* sin_row0 = cs_row0 + rope_half;
    const float* cos_row1 = cs_row1;
    const float* sin_row1 = cs_row1 + rope_half;
    if constexpr (std::is_same_v<row_t, float>) {
      process_f32_kv_rows2_rope_sve(
          row0, row1, cos_row0, sin_row0, cos_row1, sin_row1, nope_head_dim, rope_half);
    } else {
      process_bf16_kv_rows2_rope_sve(
          row0, row1, cos_row0, sin_row0, cos_row1, sin_row1, nope_head_dim, rope_half);
    }
    tail_token += 2;
  }

  if (tail_token < num_tokens) {
    row_t* row = kv_data + tail_token * kv_stride_t;
    const int64_t pos = scalar_position ? scalar_pos : pos_data[tail_token];
    const float* cs_row = cos_sin_data + pos * cos_sin_stride;
    const float* cos_row = cs_row;
    const float* sin_row = cs_row + rope_half;
    if constexpr (std::is_same_v<row_t, float>) {
      process_f32_rope_row_sve(row, cos_row, sin_row, nope_head_dim, rope_half, 1.0f);
    } else {
      process_bf16_rope_row_sve(row, cos_row, sin_row, nope_head_dim, rope_half, 1.0f);
    }
  }
}

template <typename row_t>
void indexer_q_rope_fused_sve_impl(row_t* q_data,
                                   const int64_t* pos_data,
                                   int64_t scalar_pos,
                                   bool scalar_position,
                                   const float* cos_sin_data,
                                   int64_t cos_sin_stride,
                                   int64_t num_tokens,
                                   int64_t num_heads,
                                   int64_t nope_head_dim,
                                   int64_t rope_half,
                                   int64_t q_stride_t,
                                   int64_t q_stride_h) {
  const int64_t head_groups4 = num_heads / 4;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (int64_t token = 0; token < num_tokens; ++token) {
    for (int64_t head_group = 0; head_group < head_groups4; ++head_group) {
      const int64_t head_base = head_group * 4;
      row_t* row0 = q_data + token * q_stride_t + (head_base + 0) * q_stride_h;
      row_t* row1 = q_data + token * q_stride_t + (head_base + 1) * q_stride_h;
      row_t* row2 = q_data + token * q_stride_t + (head_base + 2) * q_stride_h;
      row_t* row3 = q_data + token * q_stride_t + (head_base + 3) * q_stride_h;
      const int64_t pos = scalar_position ? scalar_pos : pos_data[token];
      const float* cs_row = cos_sin_data + pos * cos_sin_stride;
      if constexpr (std::is_same_v<row_t, float>) {
        process_f32_kv_rows4_rope_sve(
            row0,
            row1,
            row2,
            row3,
            cs_row,
            cs_row + rope_half,
            cs_row,
            cs_row + rope_half,
            cs_row,
            cs_row + rope_half,
            cs_row,
            cs_row + rope_half,
            nope_head_dim,
            rope_half);
      } else {
        process_bf16_kv_rows4_rope_sve(
            row0,
            row1,
            row2,
            row3,
            cs_row,
            cs_row + rope_half,
            cs_row,
            cs_row + rope_half,
            cs_row,
            cs_row + rope_half,
            cs_row,
            cs_row + rope_half,
            nope_head_dim,
            rope_half);
      }
    }
  }

  int64_t tail_head = head_groups4 * 4;
  if (tail_head + 1 < num_heads) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t token = 0; token < num_tokens; ++token) {
      row_t* row0 = q_data + token * q_stride_t + tail_head * q_stride_h;
      row_t* row1 = q_data + token * q_stride_t + (tail_head + 1) * q_stride_h;
      const int64_t pos = scalar_position ? scalar_pos : pos_data[token];
      const float* cs_row = cos_sin_data + pos * cos_sin_stride;
      if constexpr (std::is_same_v<row_t, float>) {
        process_f32_kv_rows2_rope_sve(
            row0, row1, cs_row, cs_row + rope_half, cs_row, cs_row + rope_half,
            nope_head_dim, rope_half);
      } else {
        process_bf16_kv_rows2_rope_sve(
            row0, row1, cs_row, cs_row + rope_half, cs_row, cs_row + rope_half,
            nope_head_dim, rope_half);
      }
    }
    tail_head += 2;
  }

  if (tail_head < num_heads) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t token = 0; token < num_tokens; ++token) {
      row_t* row = q_data + token * q_stride_t + tail_head * q_stride_h;
      const int64_t pos = scalar_position ? scalar_pos : pos_data[token];
      const float* cs_row = cos_sin_data + pos * cos_sin_stride;
      if constexpr (std::is_same_v<row_t, float>) {
        process_f32_rope_row_sve(row, cs_row, cs_row + rope_half, nope_head_dim, rope_half, 1.0f);
      } else {
        process_bf16_rope_row_sve(row, cs_row, cs_row + rope_half, nope_head_dim, rope_half, 1.0f);
      }
    }
  }
}

template <typename cache_t>
FUSED_CPP_ALWAYS_INLINE cache_t* kv_cache_row_from_slot(cache_t* cache_data,
                                                        int64_t slot,
                                                        bool cache_is_2d,
                                                        int64_t cache_stride_slot,
                                                        int64_t cache_block_size,
                                                        int64_t cache_stride_block,
                                                        int64_t cache_stride_offset) {
  if (cache_is_2d) {
    return cache_data + slot * cache_stride_slot;
  }
  return cache_data + (slot / cache_block_size) * cache_stride_block +
      (slot % cache_block_size) * cache_stride_offset;
}

template <typename row_t, typename cache_t>
FUSED_CPP_ALWAYS_INLINE void process_kv_token_to_cache_sve(const row_t* kv_data,
                                                           cache_t* cache_data,
                                                           const int64_t* slot_data,
                                                           const int64_t* pos_data,
                                                           int64_t scalar_pos,
                                                           bool scalar_position,
                                                           const float* cos_sin_data,
                                                           int64_t cos_sin_stride,
                                                           int64_t token,
                                                           int64_t nope_head_dim,
                                                           int64_t rope_half,
                                                           int64_t kv_stride_t,
                                                           bool cache_is_2d,
                                                           int64_t cache_stride_slot,
                                                           int64_t cache_block_size,
                                                           int64_t cache_stride_block,
                                                           int64_t cache_stride_offset) {
  const int64_t slot = slot_data[token];
  if (slot < 0) {
    return;
  }
  const row_t* row = kv_data + token * kv_stride_t;
  cache_t* cache_row = kv_cache_row_from_slot(
      cache_data,
      slot,
      cache_is_2d,
      cache_stride_slot,
      cache_block_size,
      cache_stride_block,
      cache_stride_offset);
  copy_kv_nope_to_cache_sve(row, cache_row, nope_head_dim);
  if (rope_half == 0) {
    return;
  }
  const int64_t pos = scalar_position ? scalar_pos : pos_data[token];
  const float* cs_row = cos_sin_data + pos * cos_sin_stride;
  process_kv_rope_row_to_cache_sve(
      row, cache_row, cs_row, cs_row + rope_half, nope_head_dim, rope_half);
}

template <typename row_t, typename cache_t>
void kv_rope_cache_insert_fused_sve_impl(const row_t* kv_data,
                                         cache_t* cache_data,
                                         const int64_t* slot_data,
                                         const int64_t* pos_data,
                                         int64_t scalar_pos,
                                         bool scalar_position,
                                         const float* cos_sin_data,
                                         int64_t cos_sin_stride,
                                         int64_t num_tokens,
                                         int64_t nope_head_dim,
                                         int64_t rope_half,
                                         int64_t kv_stride_t,
                                         bool cache_is_2d,
                                         int64_t cache_stride_slot,
                                         int64_t cache_block_size,
                                         int64_t cache_stride_block,
                                         int64_t cache_stride_offset) {
  const int64_t token_groups4 = num_tokens / 4;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t token_group = 0; token_group < token_groups4; ++token_group) {
    const int64_t token0 = token_group * 4;
    const int64_t token1 = token0 + 1;
    const int64_t token2 = token0 + 2;
    const int64_t token3 = token0 + 3;
    const int64_t slot0 = slot_data[token0];
    const int64_t slot1 = slot_data[token1];
    const int64_t slot2 = slot_data[token2];
    const int64_t slot3 = slot_data[token3];
    if (slot0 < 0 || slot1 < 0 || slot2 < 0 || slot3 < 0) {
      process_kv_token_to_cache_sve(
          kv_data,
          cache_data,
          slot_data,
          pos_data,
          scalar_pos,
          scalar_position,
          cos_sin_data,
          cos_sin_stride,
          token0,
          nope_head_dim,
          rope_half,
          kv_stride_t,
          cache_is_2d,
          cache_stride_slot,
          cache_block_size,
          cache_stride_block,
          cache_stride_offset);
      process_kv_token_to_cache_sve(
          kv_data,
          cache_data,
          slot_data,
          pos_data,
          scalar_pos,
          scalar_position,
          cos_sin_data,
          cos_sin_stride,
          token1,
          nope_head_dim,
          rope_half,
          kv_stride_t,
          cache_is_2d,
          cache_stride_slot,
          cache_block_size,
          cache_stride_block,
          cache_stride_offset);
      process_kv_token_to_cache_sve(
          kv_data,
          cache_data,
          slot_data,
          pos_data,
          scalar_pos,
          scalar_position,
          cos_sin_data,
          cos_sin_stride,
          token2,
          nope_head_dim,
          rope_half,
          kv_stride_t,
          cache_is_2d,
          cache_stride_slot,
          cache_block_size,
          cache_stride_block,
          cache_stride_offset);
      process_kv_token_to_cache_sve(
          kv_data,
          cache_data,
          slot_data,
          pos_data,
          scalar_pos,
          scalar_position,
          cos_sin_data,
          cos_sin_stride,
          token3,
          nope_head_dim,
          rope_half,
          kv_stride_t,
          cache_is_2d,
          cache_stride_slot,
          cache_block_size,
          cache_stride_block,
          cache_stride_offset);
      continue;
    }

    const row_t* row0 = kv_data + token0 * kv_stride_t;
    const row_t* row1 = kv_data + token1 * kv_stride_t;
    const row_t* row2 = kv_data + token2 * kv_stride_t;
    const row_t* row3 = kv_data + token3 * kv_stride_t;
    cache_t* cache_row0 = kv_cache_row_from_slot(
        cache_data,
        slot0,
        cache_is_2d,
        cache_stride_slot,
        cache_block_size,
        cache_stride_block,
        cache_stride_offset);
    cache_t* cache_row1 = kv_cache_row_from_slot(
        cache_data,
        slot1,
        cache_is_2d,
        cache_stride_slot,
        cache_block_size,
        cache_stride_block,
        cache_stride_offset);
    cache_t* cache_row2 = kv_cache_row_from_slot(
        cache_data,
        slot2,
        cache_is_2d,
        cache_stride_slot,
        cache_block_size,
        cache_stride_block,
        cache_stride_offset);
    cache_t* cache_row3 = kv_cache_row_from_slot(
        cache_data,
        slot3,
        cache_is_2d,
        cache_stride_slot,
        cache_block_size,
        cache_stride_block,
        cache_stride_offset);

    copy_kv_nope_rows4_to_cache_sve(
        row0,
        row1,
        row2,
        row3,
        cache_row0,
        cache_row1,
        cache_row2,
        cache_row3,
        nope_head_dim);
    if (rope_half == 0) {
      continue;
    }

    const int64_t pos0 = scalar_position ? scalar_pos : pos_data[token0];
    const int64_t pos1 = scalar_position ? scalar_pos : pos_data[token1];
    const int64_t pos2 = scalar_position ? scalar_pos : pos_data[token2];
    const int64_t pos3 = scalar_position ? scalar_pos : pos_data[token3];
    const float* cs_row0 = cos_sin_data + pos0 * cos_sin_stride;
    const float* cs_row1 = cos_sin_data + pos1 * cos_sin_stride;
    const float* cs_row2 = cos_sin_data + pos2 * cos_sin_stride;
    const float* cs_row3 = cos_sin_data + pos3 * cos_sin_stride;
    process_kv_rows4_rope_to_cache_sve(
        row0,
        row1,
        row2,
        row3,
        cache_row0,
        cache_row1,
        cache_row2,
        cache_row3,
        cs_row0,
        cs_row0 + rope_half,
        cs_row1,
        cs_row1 + rope_half,
        cs_row2,
        cs_row2 + rope_half,
        cs_row3,
        cs_row3 + rope_half,
        nope_head_dim,
        rope_half);
  }

  int64_t tail_token = token_groups4 * 4;
  if (tail_token + 1 < num_tokens) {
    const int64_t slot0 = slot_data[tail_token];
    const int64_t slot1 = slot_data[tail_token + 1];
    if (slot0 >= 0 && slot1 >= 0) {
      const row_t* row0 = kv_data + tail_token * kv_stride_t;
      const row_t* row1 = kv_data + (tail_token + 1) * kv_stride_t;
      cache_t* cache_row0 = kv_cache_row_from_slot(
          cache_data,
          slot0,
          cache_is_2d,
          cache_stride_slot,
          cache_block_size,
          cache_stride_block,
          cache_stride_offset);
      cache_t* cache_row1 = kv_cache_row_from_slot(
          cache_data,
          slot1,
          cache_is_2d,
          cache_stride_slot,
          cache_block_size,
          cache_stride_block,
          cache_stride_offset);
      copy_kv_nope_rows2_to_cache_sve(row0, row1, cache_row0, cache_row1, nope_head_dim);
      if (rope_half != 0) {
        const int64_t pos0 = scalar_position ? scalar_pos : pos_data[tail_token];
        const int64_t pos1 = scalar_position ? scalar_pos : pos_data[tail_token + 1];
        const float* cs_row0 = cos_sin_data + pos0 * cos_sin_stride;
        const float* cs_row1 = cos_sin_data + pos1 * cos_sin_stride;
        process_kv_rows2_rope_to_cache_sve(
            row0,
            row1,
            cache_row0,
            cache_row1,
            cs_row0,
            cs_row0 + rope_half,
            cs_row1,
            cs_row1 + rope_half,
            nope_head_dim,
            rope_half);
      }
    } else {
      process_kv_token_to_cache_sve(
          kv_data,
          cache_data,
          slot_data,
          pos_data,
          scalar_pos,
          scalar_position,
          cos_sin_data,
          cos_sin_stride,
          tail_token,
          nope_head_dim,
          rope_half,
          kv_stride_t,
          cache_is_2d,
          cache_stride_slot,
          cache_block_size,
          cache_stride_block,
          cache_stride_offset);
      process_kv_token_to_cache_sve(
          kv_data,
          cache_data,
          slot_data,
          pos_data,
          scalar_pos,
          scalar_position,
          cos_sin_data,
          cos_sin_stride,
          tail_token + 1,
          nope_head_dim,
          rope_half,
          kv_stride_t,
          cache_is_2d,
          cache_stride_slot,
          cache_block_size,
          cache_stride_block,
          cache_stride_offset);
    }
    tail_token += 2;
  }

  if (tail_token < num_tokens) {
    process_kv_token_to_cache_sve(
        kv_data,
        cache_data,
        slot_data,
        pos_data,
        scalar_pos,
        scalar_position,
        cos_sin_data,
        cos_sin_stride,
        tail_token,
        nope_head_dim,
        rope_half,
        kv_stride_t,
        cache_is_2d,
        cache_stride_slot,
        cache_block_size,
        cache_stride_block,
        cache_stride_offset);
  }
}

#endif  // defined(__ARM_FEATURE_SVE)

}  // namespace

bool q_norm_rope_fused_sve(const at::Tensor& q,
                           const at::Tensor& positions_long,
                           const at::Tensor& cos_sin_f,
                           double eps) {
#if defined(__ARM_FEATURE_SVE)
#if FUSED_CPP_STRICT_MODE
  TORCH_CHECK(
      q.scalar_type() == at::kBFloat16,
      "FUSED_CPP_STRICT_MODE q_norm_rope_fused_sve only supports bf16 q, got ",
      q.scalar_type());
  TORCH_CHECK(
      q.stride(2) == 1,
      "FUSED_CPP_STRICT_MODE q stride(2) must be 1, got ",
      q.stride(2));
  TORCH_CHECK(
      cos_sin_f.stride(1) == 1,
      "FUSED_CPP_STRICT_MODE cos_sin_f stride(1) must be 1, got ",
      cos_sin_f.stride(1));
#else
  if (q.stride(2) != 1 || cos_sin_f.stride(1) != 1) {
    return false;
  }
#endif
  const int64_t num_tokens = q.size(0);
  const int64_t num_heads = q.size(1);
  const int64_t head_dim = q.size(2);
  const int64_t rope_head_dim = cos_sin_f.size(1);
  const int64_t nope_head_dim = head_dim - rope_head_dim;
  const int64_t rope_half = rope_head_dim / 2;
  const bool scalar_position = positions_long.dim() == 0;
  const int64_t* pos_data = scalar_position ? nullptr : positions_long.data_ptr<int64_t>();
  const int64_t scalar_pos = scalar_position ? positions_long.item<int64_t>() : 0;
  const float* cos_sin_data = cos_sin_f.data_ptr<float>();
  const int64_t cos_sin_stride = cos_sin_f.stride(0);
  const int64_t q_stride_t = q.stride(0);
  const int64_t q_stride_h = q.stride(1);

#if !FUSED_CPP_STRICT_MODE
  if (q.scalar_type() == at::kFloat) {
    q_norm_rope_fused_sve_impl<float>(
        q.data_ptr<float>(),
        pos_data,
        scalar_pos,
        scalar_position,
        cos_sin_data,
        cos_sin_stride,
        num_tokens,
        num_heads,
        head_dim,
        nope_head_dim,
        rope_half,
        q_stride_t,
        q_stride_h,
        eps);
    return true;
  }
#endif
  if (q.scalar_type() == at::kBFloat16) {
    q_norm_rope_fused_sve_impl<uint16_t>(
        reinterpret_cast<uint16_t*>(q.data_ptr<at::BFloat16>()),
        pos_data,
        scalar_pos,
        scalar_position,
        cos_sin_data,
        cos_sin_stride,
        num_tokens,
        num_heads,
        head_dim,
        nope_head_dim,
        rope_half,
        q_stride_t,
        q_stride_h,
        eps);
    return true;
  }
#if FUSED_CPP_STRICT_MODE
  TORCH_CHECK(
      false,
      "FUSED_CPP_STRICT_MODE q_norm_rope_fused_sve only supports bf16 q, got ",
      q.scalar_type());
#endif
  return false;
#else
  (void)q;
  (void)positions_long;
  (void)cos_sin_f;
  (void)eps;
#if FUSED_CPP_STRICT_MODE
  TORCH_CHECK(false, "FUSED_CPP_STRICT_MODE q_norm_rope_fused_sve requires SVE");
#endif
  return false;
#endif
}

bool indexer_q_rope_fused_sve(const at::Tensor& q,
                              const at::Tensor& positions_long,
                              const at::Tensor& cos_sin_f) {
#if defined(__ARM_FEATURE_SVE)
#if FUSED_CPP_STRICT_MODE
  TORCH_CHECK(
      q.scalar_type() == at::kBFloat16,
      "FUSED_CPP_STRICT_MODE indexer_q_rope_fused_sve only supports bf16 q, got ",
      q.scalar_type());
  TORCH_CHECK(
      q.stride(2) == 1,
      "FUSED_CPP_STRICT_MODE indexer q stride(2) must be 1, got ",
      q.stride(2));
  TORCH_CHECK(
      cos_sin_f.stride(1) == 1,
      "FUSED_CPP_STRICT_MODE indexer cos_sin_f stride(1) must be 1, got ",
      cos_sin_f.stride(1));
#else
  if (q.scalar_type() != at::kBFloat16 || q.stride(2) != 1 || cos_sin_f.stride(1) != 1) {
    return false;
  }
#endif
  const int64_t num_tokens = q.size(0);
  const int64_t num_heads = q.size(1);
  const int64_t head_dim = q.size(2);
  const int64_t rope_head_dim = cos_sin_f.size(1);
  const int64_t nope_head_dim = head_dim - rope_head_dim;
  const int64_t rope_half = rope_head_dim / 2;
  if (rope_half == 0) {
    return true;
  }
  const bool scalar_position = positions_long.dim() == 0;
  const int64_t* pos_data = scalar_position ? nullptr : positions_long.data_ptr<int64_t>();
  const int64_t scalar_pos = scalar_position ? positions_long.item<int64_t>() : 0;
  const float* cos_sin_data = cos_sin_f.data_ptr<float>();
  const int64_t cos_sin_stride = cos_sin_f.stride(0);
  const int64_t q_stride_t = q.stride(0);
  const int64_t q_stride_h = q.stride(1);

  indexer_q_rope_fused_sve_impl<uint16_t>(
      reinterpret_cast<uint16_t*>(q.data_ptr<at::BFloat16>()),
      pos_data,
      scalar_pos,
      scalar_position,
      cos_sin_data,
      cos_sin_stride,
      num_tokens,
      num_heads,
      nope_head_dim,
      rope_half,
      q_stride_t,
      q_stride_h);
  return true;
#else
  (void)q;
  (void)positions_long;
  (void)cos_sin_f;
#if FUSED_CPP_STRICT_MODE
  TORCH_CHECK(false, "FUSED_CPP_STRICT_MODE indexer_q_rope_fused_sve requires SVE");
#endif
  return false;
#endif
}

bool kv_rope_fused_sve(const at::Tensor& kv,
                       const at::Tensor& positions_long,
                       const at::Tensor& cos_sin_f) {
#if defined(__ARM_FEATURE_SVE)
#if FUSED_CPP_STRICT_MODE
  TORCH_CHECK(
      kv.scalar_type() == at::kBFloat16,
      "FUSED_CPP_STRICT_MODE kv_rope_fused_sve only supports bf16 kv, got ",
      kv.scalar_type());
  TORCH_CHECK(
      kv.stride(1) == 1,
      "FUSED_CPP_STRICT_MODE kv stride(1) must be 1, got ",
      kv.stride(1));
  TORCH_CHECK(
      cos_sin_f.stride(1) == 1,
      "FUSED_CPP_STRICT_MODE cos_sin_f stride(1) must be 1, got ",
      cos_sin_f.stride(1));
#else
  if (kv.stride(1) != 1 || cos_sin_f.stride(1) != 1) {
    return false;
  }
#endif
  const int64_t num_tokens = kv.size(0);
  const int64_t head_dim = kv.size(1);
  const int64_t rope_head_dim = cos_sin_f.size(1);
  const int64_t nope_head_dim = head_dim - rope_head_dim;
  const int64_t rope_half = rope_head_dim / 2;
#if !FUSED_CPP_STRICT_MODE
  const bool supported_dtype =
      kv.scalar_type() == at::kFloat || kv.scalar_type() == at::kBFloat16;
  if (!supported_dtype) {
    return false;
  }
#endif
  if (rope_half == 0) {
    return true;
  }
  const bool scalar_position = positions_long.dim() == 0;
  const int64_t* pos_data = scalar_position ? nullptr : positions_long.data_ptr<int64_t>();
  const int64_t scalar_pos = scalar_position ? positions_long.item<int64_t>() : 0;
  const float* cos_sin_data = cos_sin_f.data_ptr<float>();
  const int64_t cos_sin_stride = cos_sin_f.stride(0);
  const int64_t kv_stride_t = kv.stride(0);

#if !FUSED_CPP_STRICT_MODE
  if (kv.scalar_type() == at::kFloat) {
    kv_rope_fused_sve_impl<float>(
        kv.data_ptr<float>(),
        pos_data,
        scalar_pos,
        scalar_position,
        cos_sin_data,
        cos_sin_stride,
        num_tokens,
        nope_head_dim,
        rope_half,
        kv_stride_t);
    return true;
  }
#endif
  if (kv.scalar_type() == at::kBFloat16) {
    kv_rope_fused_sve_impl<uint16_t>(
        reinterpret_cast<uint16_t*>(kv.data_ptr<at::BFloat16>()),
        pos_data,
        scalar_pos,
        scalar_position,
        cos_sin_data,
        cos_sin_stride,
        num_tokens,
        nope_head_dim,
        rope_half,
        kv_stride_t);
    return true;
  }
#if FUSED_CPP_STRICT_MODE
  TORCH_CHECK(
      false,
      "FUSED_CPP_STRICT_MODE kv_rope_fused_sve only supports bf16 kv, got ",
      kv.scalar_type());
#endif
  return false;
#else
  (void)kv;
  (void)positions_long;
  (void)cos_sin_f;
#if FUSED_CPP_STRICT_MODE
  TORCH_CHECK(false, "FUSED_CPP_STRICT_MODE kv_rope_fused_sve requires SVE");
#endif
  return false;
#endif
}

bool kv_rope_cache_insert_fused_sve(const at::Tensor& kv,
                                    const at::Tensor& swa_kv_cache,
                                    const at::Tensor& slot_mapping_long,
                                    const at::Tensor& positions_long,
                                    const at::Tensor& cos_sin_f) {
#if defined(__ARM_FEATURE_SVE)
#if FUSED_CPP_STRICT_MODE
  if (swa_kv_cache.numel() == 0) {
    return false;
  }
  TORCH_CHECK(
      kv.scalar_type() == at::kBFloat16,
      "FUSED_CPP_STRICT_MODE kv_rope_cache_insert_fused_sve only supports bf16 kv, got ",
      kv.scalar_type());
  TORCH_CHECK(
      swa_kv_cache.scalar_type() == at::kBFloat16,
      "FUSED_CPP_STRICT_MODE kv_rope_cache_insert_fused_sve only supports bf16 swa_kv_cache, got ",
      swa_kv_cache.scalar_type());
  TORCH_CHECK(
      kv.stride(1) == 1,
      "FUSED_CPP_STRICT_MODE kv stride(1) must be 1, got ",
      kv.stride(1));
  TORCH_CHECK(
      cos_sin_f.stride(1) == 1,
      "FUSED_CPP_STRICT_MODE cos_sin_f stride(1) must be 1, got ",
      cos_sin_f.stride(1));
#else
  if (swa_kv_cache.numel() == 0 || kv.stride(1) != 1 || cos_sin_f.stride(1) != 1) {
    return false;
  }
#endif
#if !FUSED_CPP_STRICT_MODE
  const bool supported_kv_dtype =
      kv.scalar_type() == at::kFloat || kv.scalar_type() == at::kBFloat16;
  const bool supported_cache_dtype =
      swa_kv_cache.scalar_type() == at::kFloat || swa_kv_cache.scalar_type() == at::kBFloat16;
  if (!supported_kv_dtype || !supported_cache_dtype) {
    return false;
  }
#endif

  int64_t cache_stride_slot = 0;
  int64_t cache_block_size = 0;
  int64_t cache_stride_block = 0;
  int64_t cache_stride_offset = 0;
  int64_t cache_stride_d = 0;
  const bool cache_is_2d = swa_kv_cache.dim() == 2;
  if (cache_is_2d) {
    cache_stride_slot = swa_kv_cache.stride(0);
    cache_stride_d = swa_kv_cache.stride(1);
  } else if (swa_kv_cache.dim() == 3) {
    cache_block_size = swa_kv_cache.size(1);
    cache_stride_block = swa_kv_cache.stride(0);
    cache_stride_offset = swa_kv_cache.stride(1);
    cache_stride_d = swa_kv_cache.stride(2);
  } else {
#if FUSED_CPP_STRICT_MODE
    TORCH_CHECK(
        false,
        "FUSED_CPP_STRICT_MODE swa_kv_cache must be 2-D or 3-D, got ",
        swa_kv_cache.dim(),
        "-D");
#endif
    return false;
  }
  if (cache_stride_d != 1) {
#if FUSED_CPP_STRICT_MODE
    TORCH_CHECK(
        false,
        "FUSED_CPP_STRICT_MODE swa_kv_cache last-dim stride must be 1, got ",
        cache_stride_d);
#endif
    return false;
  }

  const int64_t num_tokens = kv.size(0);
  const int64_t head_dim = kv.size(1);
  const int64_t rope_head_dim = cos_sin_f.size(1);
  const int64_t nope_head_dim = head_dim - rope_head_dim;
  const int64_t rope_half = rope_head_dim / 2;
  const bool scalar_position = positions_long.dim() == 0;
  const int64_t* pos_data = scalar_position ? nullptr : positions_long.data_ptr<int64_t>();
  const int64_t scalar_pos = scalar_position ? positions_long.item<int64_t>() : 0;
  const float* cos_sin_data = cos_sin_f.data_ptr<float>();
  const int64_t cos_sin_stride = cos_sin_f.stride(0);
  const int64_t kv_stride_t = kv.stride(0);
  const int64_t* slot_data = slot_mapping_long.data_ptr<int64_t>();

#if !FUSED_CPP_STRICT_MODE
  if (kv.scalar_type() == at::kFloat && swa_kv_cache.scalar_type() == at::kFloat) {
    kv_rope_cache_insert_fused_sve_impl<float, float>(
        kv.data_ptr<float>(),
        swa_kv_cache.data_ptr<float>(),
        slot_data,
        pos_data,
        scalar_pos,
        scalar_position,
        cos_sin_data,
        cos_sin_stride,
        num_tokens,
        nope_head_dim,
        rope_half,
        kv_stride_t,
        cache_is_2d,
        cache_stride_slot,
        cache_block_size,
        cache_stride_block,
        cache_stride_offset);
    return true;
  }
  if (kv.scalar_type() == at::kFloat && swa_kv_cache.scalar_type() == at::kBFloat16) {
    kv_rope_cache_insert_fused_sve_impl<float, uint16_t>(
        kv.data_ptr<float>(),
        reinterpret_cast<uint16_t*>(swa_kv_cache.data_ptr<at::BFloat16>()),
        slot_data,
        pos_data,
        scalar_pos,
        scalar_position,
        cos_sin_data,
        cos_sin_stride,
        num_tokens,
        nope_head_dim,
        rope_half,
        kv_stride_t,
        cache_is_2d,
        cache_stride_slot,
        cache_block_size,
        cache_stride_block,
        cache_stride_offset);
    return true;
  }
  if (kv.scalar_type() == at::kBFloat16 && swa_kv_cache.scalar_type() == at::kFloat) {
    kv_rope_cache_insert_fused_sve_impl<uint16_t, float>(
        reinterpret_cast<const uint16_t*>(kv.data_ptr<at::BFloat16>()),
        swa_kv_cache.data_ptr<float>(),
        slot_data,
        pos_data,
        scalar_pos,
        scalar_position,
        cos_sin_data,
        cos_sin_stride,
        num_tokens,
        nope_head_dim,
        rope_half,
        kv_stride_t,
        cache_is_2d,
        cache_stride_slot,
        cache_block_size,
        cache_stride_block,
        cache_stride_offset);
    return true;
  }
#endif
  if (kv.scalar_type() == at::kBFloat16 && swa_kv_cache.scalar_type() == at::kBFloat16) {
    kv_rope_cache_insert_fused_sve_impl<uint16_t, uint16_t>(
        reinterpret_cast<const uint16_t*>(kv.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint16_t*>(swa_kv_cache.data_ptr<at::BFloat16>()),
        slot_data,
        pos_data,
        scalar_pos,
        scalar_position,
        cos_sin_data,
        cos_sin_stride,
        num_tokens,
        nope_head_dim,
        rope_half,
        kv_stride_t,
        cache_is_2d,
        cache_stride_slot,
        cache_block_size,
        cache_stride_block,
        cache_stride_offset);
    return true;
  }
#if FUSED_CPP_STRICT_MODE
  TORCH_CHECK(
      false,
      "FUSED_CPP_STRICT_MODE kv_rope_cache_insert_fused_sve requires bf16 kv and bf16 swa_kv_cache, got kv=",
      kv.scalar_type(),
      " cache=",
      swa_kv_cache.scalar_type());
#endif
  return false;
#else
  (void)kv;
  (void)swa_kv_cache;
  (void)slot_mapping_long;
  (void)positions_long;
  (void)cos_sin_f;
#if FUSED_CPP_STRICT_MODE
  TORCH_CHECK(false, "FUSED_CPP_STRICT_MODE kv_rope_cache_insert_fused_sve requires SVE");
#endif
  return false;
#endif
}

}  // namespace fused_cpp::deepseek_v4
