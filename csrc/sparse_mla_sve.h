// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <cstring>

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && \
    (defined(__ARM_FEATURE_BF16) || defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC))
#include <arm_sve.h>
#define FUSED_CPP_SPARSE_MLA_HAS_SVE_BFMMLA 1
#else
#define FUSED_CPP_SPARSE_MLA_HAS_SVE_BFMMLA 0
#endif

namespace fused_cpp::sparse_mla_sve {

// The scalable N tile contains eight keys per 128-bit SVE segment. Thus the
// same 16-accumulator BFMMLA topology computes 8x8 at SVL128 and 8x16 at
// SVL256. Callers must keep every worker at the same process SVE VL because
// the packed layout is VL-dependent and shared with the worker-local kernel.
inline int64_t n_tile() {
#if FUSED_CPP_SPARSE_MLA_HAS_SVE_BFMMLA
  return static_cast<int64_t>(svcntb() / 2);
#else
  return 8;
#endif
}

#if FUSED_CPP_SPARSE_MLA_HAS_SVE_BFMMLA

inline int64_t packed_k_tile_elements(int64_t reduction) {
  return reduction * n_tile();
}

// Pack K[N, reduction] into four BFMMLA B vectors per reduction block. Each
// SVE 128-bit segment holds a 4x2 BF16 panel for one pair of key columns.
inline void pack_indexed_k_tile_bf16(const uint16_t* kv, int64_t kv_row_stride,
                                     const int64_t* indices, int64_t reduction,
                                     uint16_t* packed_k) {
  const int64_t segments = static_cast<int64_t>(svcntb() / 16);
  int64_t out = 0;
  for (int64_t reduction_base = 0; reduction_base < reduction;
       reduction_base += 4) {
    for (int64_t column_pair = 0; column_pair < 4; ++column_pair) {
      for (int64_t segment = 0; segment < segments; ++segment) {
        const int64_t column0 = segment * 8 + column_pair * 2;
        const int64_t column1 = column0 + 1;
        std::memcpy(packed_k + out,
                    kv + indices[column0] * kv_row_stride + reduction_base,
                    4 * sizeof(uint16_t));
        out += 4;
        std::memcpy(packed_k + out,
                    kv + indices[column1] * kv_row_stride + reduction_base,
                    4 * sizeof(uint16_t));
        out += 4;
      }
    }
  }
}

inline svbfloat16_t load_bf16(const uint16_t* ptr) {
  return svld1_bf16(svptrue_b16(), reinterpret_cast<const __bf16*>(ptr));
}

inline void store_qkt_row_pair(float* scores, int64_t scores_row_stride,
                               int row_pair, int column_pair, float scale,
                               svfloat32_t accumulator) {
  const svbool_t pg_all = svptrue_b32();
  const svfloat32_t scaled =
      svmul_f32_x(pg_all, accumulator, svdup_f32(scale));
  const svuint64_t scaled_u64 = svreinterpret_u64_f32(scaled);
  const svfloat32_t row0 =
      svreinterpret_f32_u64(svuzp1_u64(scaled_u64, scaled_u64));
  const svfloat32_t row1 =
      svreinterpret_f32_u64(svuzp2_u64(scaled_u64, scaled_u64));

  // UZP packs one two-column result from every 128-bit segment into the low
  // half. Scatter those pairs to their row-major positions in the 2VL tile.
  const uint64_t packed_lanes = static_cast<uint64_t>(svcntw() / 2);
  const svbool_t pg = svwhilelt_b32(uint64_t{0}, packed_lanes);
  const svuint32_t lane = svindex_u32(0, 1);
  const svuint32_t segment = svlsr_n_u32_x(pg, lane, 1);
  const svuint32_t column_in_pair = svand_n_u32_x(pg, lane, 1);
  svuint32_t offsets =
      svadd_u32_x(pg, svlsl_n_u32_x(pg, segment, 3), column_in_pair);
  offsets = svadd_n_u32_x(pg, offsets,
                          static_cast<uint32_t>(2 * column_pair));

  float* row0_out = scores + static_cast<int64_t>(2 * row_pair) *
                                 scores_row_stride;
  float* row1_out = row0_out + scores_row_stride;
  svst1_scatter_u32index_f32(pg, row0_out, offsets, row0);
  svst1_scatter_u32index_f32(pg, row1_out, offsets, row1);
}

// Q is packed as [reduction/4][four 2-row x 4-reduction panels]. K uses the
// scalable B layout above. BFMMLA works independently in each 128-bit segment,
// keeping the instruction/register topology fixed while N scales with SVE VL.
__attribute__((noinline)) inline void qkt_8x2vl_bf16(
    const uint16_t* packed_q, const uint16_t* packed_k, int64_t reduction,
    float scale, float* scores) {
  svfloat32_t c00 = svdup_f32(0.0f), c01 = svdup_f32(0.0f);
  svfloat32_t c02 = svdup_f32(0.0f), c03 = svdup_f32(0.0f);
  svfloat32_t c10 = svdup_f32(0.0f), c11 = svdup_f32(0.0f);
  svfloat32_t c12 = svdup_f32(0.0f), c13 = svdup_f32(0.0f);
  svfloat32_t c20 = svdup_f32(0.0f), c21 = svdup_f32(0.0f);
  svfloat32_t c22 = svdup_f32(0.0f), c23 = svdup_f32(0.0f);
  svfloat32_t c30 = svdup_f32(0.0f), c31 = svdup_f32(0.0f);
  svfloat32_t c32 = svdup_f32(0.0f), c33 = svdup_f32(0.0f);

  const svbool_t pg_bf16 = svptrue_b16();
  const int64_t lanes_h = static_cast<int64_t>(svcnth());
  const uint16_t* q_ptr = packed_q;
  const uint16_t* k_ptr = packed_k;
  for (int64_t reduction_base = 0; reduction_base < reduction;
       reduction_base += 4) {
    const svbfloat16_t b0 = load_bf16(k_ptr + 0 * lanes_h);
    const svbfloat16_t b1 = load_bf16(k_ptr + 1 * lanes_h);
    const svbfloat16_t b2 = load_bf16(k_ptr + 2 * lanes_h);
    const svbfloat16_t b3 = load_bf16(k_ptr + 3 * lanes_h);
    k_ptr += 4 * lanes_h;

    const svbfloat16_t a0 = svld1rq_bf16(
        pg_bf16, reinterpret_cast<const __bf16*>(q_ptr + 0));
    const svbfloat16_t a1 = svld1rq_bf16(
        pg_bf16, reinterpret_cast<const __bf16*>(q_ptr + 8));
    const svbfloat16_t a2 = svld1rq_bf16(
        pg_bf16, reinterpret_cast<const __bf16*>(q_ptr + 16));
    const svbfloat16_t a3 = svld1rq_bf16(
        pg_bf16, reinterpret_cast<const __bf16*>(q_ptr + 24));
    q_ptr += 32;

    c00 = svbfmmla_f32(c00, a0, b0);
    c01 = svbfmmla_f32(c01, a0, b1);
    c02 = svbfmmla_f32(c02, a0, b2);
    c03 = svbfmmla_f32(c03, a0, b3);
    c10 = svbfmmla_f32(c10, a1, b0);
    c11 = svbfmmla_f32(c11, a1, b1);
    c12 = svbfmmla_f32(c12, a1, b2);
    c13 = svbfmmla_f32(c13, a1, b3);
    c20 = svbfmmla_f32(c20, a2, b0);
    c21 = svbfmmla_f32(c21, a2, b1);
    c22 = svbfmmla_f32(c22, a2, b2);
    c23 = svbfmmla_f32(c23, a2, b3);
    c30 = svbfmmla_f32(c30, a3, b0);
    c31 = svbfmmla_f32(c31, a3, b1);
    c32 = svbfmmla_f32(c32, a3, b2);
    c33 = svbfmmla_f32(c33, a3, b3);
  }

  const int64_t tile = n_tile();
  store_qkt_row_pair(scores, tile, 0, 0, scale, c00);
  store_qkt_row_pair(scores, tile, 0, 1, scale, c01);
  store_qkt_row_pair(scores, tile, 0, 2, scale, c02);
  store_qkt_row_pair(scores, tile, 0, 3, scale, c03);
  store_qkt_row_pair(scores, tile, 1, 0, scale, c10);
  store_qkt_row_pair(scores, tile, 1, 1, scale, c11);
  store_qkt_row_pair(scores, tile, 1, 2, scale, c12);
  store_qkt_row_pair(scores, tile, 1, 3, scale, c13);
  store_qkt_row_pair(scores, tile, 2, 0, scale, c20);
  store_qkt_row_pair(scores, tile, 2, 1, scale, c21);
  store_qkt_row_pair(scores, tile, 2, 2, scale, c22);
  store_qkt_row_pair(scores, tile, 2, 3, scale, c23);
  store_qkt_row_pair(scores, tile, 3, 0, scale, c30);
  store_qkt_row_pair(scores, tile, 3, 1, scale, c31);
  store_qkt_row_pair(scores, tile, 3, 2, scale, c32);
  store_qkt_row_pair(scores, tile, 3, 3, scale, c33);
}

#endif  // FUSED_CPP_SPARSE_MLA_HAS_SVE_BFMMLA

}  // namespace fused_cpp::sparse_mla_sve
