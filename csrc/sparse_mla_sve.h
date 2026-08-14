// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <cstring>

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && \
    (defined(__ARM_FEATURE_BF16) || defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC))
#include <arm_neon.h>
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

inline int64_t packed_v_elements(int64_t output_columns,
                                 int64_t reduction_capacity) {
  const int64_t tile = n_tile();
  return ((output_columns + tile - 1) / tile) * reduction_capacity * tile;
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

// Pack a contiguous K tile into the same scalable B layout. valid_columns may
// be smaller than 2VL for the final tile; zero panels keep the fixed BFMMLA
// instruction topology without reading past the shared dense interval.
inline void pack_contiguous_k_tile_bf16(
    const uint16_t* kv, int64_t kv_row_stride, int64_t valid_columns,
    int64_t reduction, uint16_t* packed_k) {
  const int64_t segments = static_cast<int64_t>(svcntb() / 16);
  int64_t out = 0;
  for (int64_t reduction_base = 0; reduction_base < reduction;
       reduction_base += 4) {
    for (int64_t column_pair = 0; column_pair < 4; ++column_pair) {
      for (int64_t segment = 0; segment < segments; ++segment) {
        const int64_t column0 = segment * 8 + column_pair * 2;
        const int64_t column1 = column0 + 1;
        if (column0 < valid_columns) {
          std::memcpy(packed_k + out,
                      kv + column0 * kv_row_stride + reduction_base,
                      4 * sizeof(uint16_t));
        } else {
          std::memset(packed_k + out, 0, 4 * sizeof(uint16_t));
        }
        out += 4;
        if (column1 < valid_columns) {
          std::memcpy(packed_k + out,
                      kv + column1 * kv_row_stride + reduction_base,
                      4 * sizeof(uint16_t));
        } else {
          std::memset(packed_k + out, 0, 4 * sizeof(uint16_t));
        }
        out += 4;
      }
    }
  }
}

inline void store_v_4x8_bfmmla_vectors(
    uint16x8_t row0, uint16x8_t row1, uint16x8_t row2, uint16x8_t row3,
    uint16_t* block, int64_t tile, int64_t segment) {
  const uint16x8x2_t rows01 = vzipq_u16(row0, row1);
  const uint16x8x2_t rows23 = vzipq_u16(row2, row3);
  const uint32x4x2_t columns03 = vzipq_u32(
      vreinterpretq_u32_u16(rows01.val[0]),
      vreinterpretq_u32_u16(rows23.val[0]));
  const uint32x4x2_t columns47 = vzipq_u32(
      vreinterpretq_u32_u16(rows01.val[1]),
      vreinterpretq_u32_u16(rows23.val[1]));
  vst1q_u16(block + 0 * tile + segment * 8,
            vreinterpretq_u16_u32(columns03.val[0]));
  vst1q_u16(block + 1 * tile + segment * 8,
            vreinterpretq_u16_u32(columns03.val[1]));
  vst1q_u16(block + 2 * tile + segment * 8,
            vreinterpretq_u16_u32(columns47.val[0]));
  vst1q_u16(block + 3 * tile + segment * 8,
            vreinterpretq_u16_u32(columns47.val[1]));
}

inline void store_v_4x8_bfmmla_segment(
    const uint16_t* row0_ptr, const uint16_t* row1_ptr,
    const uint16_t* row2_ptr, const uint16_t* row3_ptr, uint16_t* block,
    int64_t tile, int64_t segment) {
  store_v_4x8_bfmmla_vectors(
      vld1q_u16(row0_ptr), vld1q_u16(row1_ptr), vld1q_u16(row2_ptr),
      vld1q_u16(row3_ptr), block, tile, segment);
}

inline void zero_v_bfmmla_segment(uint16_t* block, int64_t tile,
                                  int64_t segment) {
  for (int64_t panel = 0; panel < 4; ++panel) {
    std::memset(block + panel * tile + segment * 8, 0,
                8 * sizeof(uint16_t));
  }
}

// Pack four gathered V rows at a time into the scalable BFMMLA B layout. A
// NEON 4x8 transpose produces four 4x2 panels for every 128-bit SVE segment;
// the panels for equal column pairs are adjacent across all segments and form
// one scalable B operand. output_columns is a multiple of eight on this path,
// so a possible half tile at SVL256 is padded by a zero 8-column segment.
inline void pack_indexed_v_tile_bf16(
    const uint16_t* kv, int64_t kv_row_stride, const int64_t* indices,
    int64_t reduction_rows, int64_t output_columns,
    int64_t reduction_capacity, int64_t reduction_offset,
    uint16_t* packed_v) {
  const int64_t tile = n_tile();
  const int64_t segments = static_cast<int64_t>(svcntb() / 16);
  for (int64_t output_base = 0; output_base < output_columns;
       output_base += tile) {
    uint16_t* output_tile =
        packed_v + (output_base / tile) * reduction_capacity * tile;
    for (int64_t reduction_base = 0; reduction_base < reduction_rows;
         reduction_base += 4) {
      uint16_t* block =
          output_tile + (reduction_offset + reduction_base) * tile;
      for (int64_t segment = 0; segment < segments; ++segment) {
        const int64_t column = output_base + segment * 8;
        if (column + 8 <= output_columns) {
          store_v_4x8_bfmmla_segment(
              kv + indices[reduction_base + 0] * kv_row_stride + column,
              kv + indices[reduction_base + 1] * kv_row_stride + column,
              kv + indices[reduction_base + 2] * kv_row_stride + column,
              kv + indices[reduction_base + 3] * kv_row_stride + column,
              block, tile, segment);
        } else {
          zero_v_bfmmla_segment(block, tile, segment);
        }
      }
    }
  }
}

// Gather each group of four indexed KV rows once and materialize both packed
// operands. V is the leading output_columns slice of each K row, so its 4x8
// transpose reuses the vectors already loaded for two adjacent K=4 panels.
inline void pack_indexed_kv_tile_bf16(
    const uint16_t* kv, int64_t kv_row_stride, const int64_t* indices,
    int64_t reduction, int64_t output_columns, int64_t reduction_capacity,
    int64_t reduction_offset, uint16_t* packed_k, uint16_t* packed_v) {
  const int64_t tile = n_tile();
  const int64_t segments = static_cast<int64_t>(svcntb() / 16);
  for (int64_t row_base = 0; row_base < tile; row_base += 4) {
    const uint16_t* row0 = kv + indices[row_base + 0] * kv_row_stride;
    const uint16_t* row1 = kv + indices[row_base + 1] * kv_row_stride;
    const uint16_t* row2 = kv + indices[row_base + 2] * kv_row_stride;
    const uint16_t* row3 = kv + indices[row_base + 3] * kv_row_stride;
    const int64_t key_segment = row_base / 8;
    const int64_t key_pair = (row_base % 8) / 2;

    int64_t reduction_base = 0;
    for (; reduction_base + 8 <= reduction; reduction_base += 8) {
      const uint16x8_t values0 = vld1q_u16(row0 + reduction_base);
      const uint16x8_t values1 = vld1q_u16(row1 + reduction_base);
      const uint16x8_t values2 = vld1q_u16(row2 + reduction_base);
      const uint16x8_t values3 = vld1q_u16(row3 + reduction_base);

      for (int64_t half = 0; half < 2; ++half) {
        uint16_t* k_block =
            packed_k + ((reduction_base / 4) + half) * 4 * tile;
        uint16_t* k_pair01 =
            k_block + (key_pair * segments + key_segment) * 8;
        uint16_t* k_pair23 =
            k_block + ((key_pair + 1) * segments + key_segment) * 8;
        if (half == 0) {
          vst1q_u16(k_pair01,
                    vcombine_u16(vget_low_u16(values0),
                                 vget_low_u16(values1)));
          vst1q_u16(k_pair23,
                    vcombine_u16(vget_low_u16(values2),
                                 vget_low_u16(values3)));
        } else {
          vst1q_u16(k_pair01,
                    vcombine_u16(vget_high_u16(values0),
                                 vget_high_u16(values1)));
          vst1q_u16(k_pair23,
                    vcombine_u16(vget_high_u16(values2),
                                 vget_high_u16(values3)));
        }
      }

      if (reduction_base < output_columns) {
        const int64_t output_tile = reduction_base / tile;
        const int64_t output_segment = (reduction_base % tile) / 8;
        uint16_t* v_block =
            packed_v + output_tile * reduction_capacity * tile +
            (reduction_offset + row_base) * tile;
        store_v_4x8_bfmmla_vectors(values0, values1, values2, values3,
                                   v_block, tile, output_segment);
      }
    }

    if (reduction_base < reduction) {
      uint16_t* k_block = packed_k + (reduction_base / 4) * 4 * tile;
      std::memcpy(k_block + (key_pair * segments + key_segment) * 8,
                  row0 + reduction_base, 4 * sizeof(uint16_t));
      std::memcpy(k_block + (key_pair * segments + key_segment) * 8 + 4,
                  row1 + reduction_base, 4 * sizeof(uint16_t));
      std::memcpy(k_block + ((key_pair + 1) * segments + key_segment) * 8,
                  row2 + reduction_base, 4 * sizeof(uint16_t));
      std::memcpy(
          k_block + ((key_pair + 1) * segments + key_segment) * 8 + 4,
          row3 + reduction_base, 4 * sizeof(uint16_t));
    }
  }
}

// Pack a contiguous V slab into one scalable B tile. The caller parallelizes
// independent output tiles and supplies valid_columns for an optional final
// eight-column half tile at SVL256.
inline void pack_contiguous_v_tile_bf16(
    const uint16_t* kv, int64_t kv_row_stride, int64_t reduction_rows,
    int64_t output_base, int64_t valid_columns, uint16_t* packed_v) {
  const int64_t tile = n_tile();
  const int64_t segments = static_cast<int64_t>(svcntb() / 16);
  for (int64_t reduction_base = 0; reduction_base < reduction_rows;
       reduction_base += 4) {
    uint16_t* block = packed_v + reduction_base * tile;
    for (int64_t segment = 0; segment < segments; ++segment) {
      const int64_t local_column = segment * 8;
      if (local_column + 8 <= valid_columns) {
        const int64_t column = output_base + local_column;
        store_v_4x8_bfmmla_segment(
            kv + (reduction_base + 0) * kv_row_stride + column,
            kv + (reduction_base + 1) * kv_row_stride + column,
            kv + (reduction_base + 2) * kv_row_stride + column,
            kv + (reduction_base + 3) * kv_row_stride + column, block, tile,
            segment);
      } else {
        zero_v_bfmmla_segment(block, tile, segment);
      }
    }
  }
}

// P is row-major [8, reduction]. Repack it into the same four 2-row A panels
// used by Q. A partial final K=4 block is explicitly zero padded so PV can use
// only BFMMLA instructions for every valid reduction length.
inline void pack_p_8rows_bf16(const uint16_t* probabilities,
                              int64_t probability_row_stride,
                              int64_t reduction,
                              int64_t padded_reduction,
                              uint16_t* packed_p) {
  for (int64_t reduction_base = 0; reduction_base < padded_reduction;
       reduction_base += 4) {
    uint16_t* block = packed_p + (reduction_base / 4) * 32;
    for (int64_t row = 0; row < 8; ++row) {
      uint16_t* destination = block + row * 4;
      if (reduction_base + 4 <= reduction) {
        std::memcpy(destination,
                    probabilities + row * probability_row_stride +
                        reduction_base,
                    4 * sizeof(uint16_t));
      } else {
        for (int64_t lane = 0; lane < 4; ++lane) {
          const int64_t column = reduction_base + lane;
          destination[lane] =
              column < reduction
                  ? probabilities[row * probability_row_stride + column]
                  : uint16_t{0};
        }
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

inline void add_pv_row_pair(float* output, int64_t output_row_stride,
                            int row_pair, int column_pair,
                            int64_t valid_columns,
                            svfloat32_t accumulator) {
  const svuint64_t accumulator_u64 = svreinterpret_u64_f32(accumulator);
  const svfloat32_t row0 =
      svreinterpret_f32_u64(svuzp1_u64(accumulator_u64, accumulator_u64));
  const svfloat32_t row1 =
      svreinterpret_f32_u64(svuzp2_u64(accumulator_u64, accumulator_u64));

  const uint64_t packed_lanes = static_cast<uint64_t>(svcntw() / 2);
  const svbool_t pg_packed = svwhilelt_b32(uint64_t{0}, packed_lanes);
  const svuint32_t lane = svindex_u32(0, 1);
  const svuint32_t segment = svlsr_n_u32_x(pg_packed, lane, 1);
  const svuint32_t column_in_pair = svand_n_u32_x(pg_packed, lane, 1);
  svuint32_t offsets = svadd_u32_x(
      pg_packed, svlsl_n_u32_x(pg_packed, segment, 3), column_in_pair);
  offsets = svadd_n_u32_x(pg_packed, offsets,
                          static_cast<uint32_t>(2 * column_pair));
  const svbool_t pg = svcmplt_n_u32(
      pg_packed, offsets, static_cast<uint32_t>(valid_columns));

  float* row0_out = output + static_cast<int64_t>(2 * row_pair) *
                                 output_row_stride;
  float* row1_out = row0_out + output_row_stride;
  const svfloat32_t old0 =
      svld1_gather_u32index_f32(pg, row0_out, offsets);
  const svfloat32_t old1 =
      svld1_gather_u32index_f32(pg, row1_out, offsets);
  svst1_scatter_u32index_f32(pg, row0_out, offsets,
                             svadd_f32_x(pg, old0, row0));
  svst1_scatter_u32index_f32(pg, row1_out, offsets,
                             svadd_f32_x(pg, old1, row1));
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

// Compute O[8, 2VL] += P[8, K] * V[K, 2VL]. P and V use the scalable A/B
// layouts produced above. The final output tile may contain one zero-padded
// 8-column segment at SVL256; valid_columns predicates those lanes on update.
__attribute__((noinline)) inline void pv_8x2vl_bf16(
    const uint16_t* packed_p, const uint16_t* packed_v, int64_t reduction,
    int64_t valid_columns, float* output, int64_t output_row_stride) {
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
  const uint16_t* p_ptr = packed_p;
  const uint16_t* v_ptr = packed_v;
  for (int64_t reduction_base = 0; reduction_base < reduction;
       reduction_base += 4) {
    const svbfloat16_t b0 = load_bf16(v_ptr + 0 * lanes_h);
    const svbfloat16_t b1 = load_bf16(v_ptr + 1 * lanes_h);
    const svbfloat16_t b2 = load_bf16(v_ptr + 2 * lanes_h);
    const svbfloat16_t b3 = load_bf16(v_ptr + 3 * lanes_h);
    v_ptr += 4 * lanes_h;

    const svbfloat16_t a0 = svld1rq_bf16(
        pg_bf16, reinterpret_cast<const __bf16*>(p_ptr + 0));
    const svbfloat16_t a1 = svld1rq_bf16(
        pg_bf16, reinterpret_cast<const __bf16*>(p_ptr + 8));
    const svbfloat16_t a2 = svld1rq_bf16(
        pg_bf16, reinterpret_cast<const __bf16*>(p_ptr + 16));
    const svbfloat16_t a3 = svld1rq_bf16(
        pg_bf16, reinterpret_cast<const __bf16*>(p_ptr + 24));
    p_ptr += 32;

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

  add_pv_row_pair(output, output_row_stride, 0, 0, valid_columns, c00);
  add_pv_row_pair(output, output_row_stride, 0, 1, valid_columns, c01);
  add_pv_row_pair(output, output_row_stride, 0, 2, valid_columns, c02);
  add_pv_row_pair(output, output_row_stride, 0, 3, valid_columns, c03);
  add_pv_row_pair(output, output_row_stride, 1, 0, valid_columns, c10);
  add_pv_row_pair(output, output_row_stride, 1, 1, valid_columns, c11);
  add_pv_row_pair(output, output_row_stride, 1, 2, valid_columns, c12);
  add_pv_row_pair(output, output_row_stride, 1, 3, valid_columns, c13);
  add_pv_row_pair(output, output_row_stride, 2, 0, valid_columns, c20);
  add_pv_row_pair(output, output_row_stride, 2, 1, valid_columns, c21);
  add_pv_row_pair(output, output_row_stride, 2, 2, valid_columns, c22);
  add_pv_row_pair(output, output_row_stride, 2, 3, valid_columns, c23);
  add_pv_row_pair(output, output_row_stride, 3, 0, valid_columns, c30);
  add_pv_row_pair(output, output_row_stride, 3, 1, valid_columns, c31);
  add_pv_row_pair(output, output_row_stride, 3, 2, valid_columns, c32);
  add_pv_row_pair(output, output_row_stride, 3, 3, valid_columns, c33);
}

#endif  // FUSED_CPP_SPARSE_MLA_HAS_SVE_BFMMLA

}  // namespace fused_cpp::sparse_mla_sve
