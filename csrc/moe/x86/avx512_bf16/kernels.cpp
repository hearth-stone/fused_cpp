// SPDX-License-Identifier: Apache-2.0
#include "kernels.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

#include <immintrin.h>

namespace fused_cpp::moe::x86::avx512_bf16 {
namespace {

inline __m512bh AsBf16(__m512i value) { return reinterpret_cast<__m512bh&>(value); }

inline __m512bh BroadcastBf16Pair(const uint16_t* pointer) {
  uint32_t pair = 0;
  std::memcpy(&pair, pointer, sizeof(pair));
  __m512i broadcast = _mm512_set1_epi32(static_cast<int>(pair));
  return AsBf16(broadcast);
}

inline __m512 ExpNeg(__m512 gate, int degree) {
  const __m512 one = _mm512_set1_ps(1.0f);
  const __m512 x = _mm512_max_ps(_mm512_set1_ps(-87.0f),
                                 _mm512_min_ps(_mm512_set1_ps(87.0f), _mm512_sub_ps(_mm512_setzero_ps(), gate)));
  const __m512 fn = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.4426950408889634f)),
                                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  const __m512 r = _mm512_fnmadd_ps(fn, _mm512_set1_ps(0.6931471805599453f), x);

  __m512 poly;
  if (degree == 4) {
    poly = _mm512_set1_ps(0.04166666f);
  } else if (degree == 5) {
    poly = _mm512_fmadd_ps(_mm512_set1_ps(0.00833333f), r, _mm512_set1_ps(0.04166666f));
  } else {
    const __m512 inner = _mm512_fmadd_ps(_mm512_set1_ps(0.0013888889f), r, _mm512_set1_ps(0.00833333f));
    poly = _mm512_fmadd_ps(inner, r, _mm512_set1_ps(0.04166666f));
  }
  poly = _mm512_fmadd_ps(poly, r, _mm512_set1_ps(0.16666666f));
  poly = _mm512_fmadd_ps(poly, r, _mm512_set1_ps(0.5f));
  poly = _mm512_fmadd_ps(poly, r, one);
  poly = _mm512_fmadd_ps(poly, r, one);

  __m512i exponent = _mm512_cvtps_epi32(fn);
  exponent = _mm512_slli_epi32(_mm512_add_epi32(exponent, _mm512_set1_epi32(127)), 23);
  return _mm512_mul_ps(poly, _mm512_castsi512_ps(exponent));
}

inline __m512 SiluMul(__m512 gate, __m512 up, int degree) {
  const __m512 denominator = _mm512_add_ps(ExpNeg(gate, degree), _mm512_set1_ps(1.0f));
  return _mm512_div_ps(_mm512_mul_ps(gate, up), denominator);
}

inline void StoreBf16(__m512 value, uint16_t* pointer, __mmask16 mask = 0xffff) {
  const __m256bh converted = _mm512_cvtneps_pbh(value);
  const __m256i bits = reinterpret_cast<const __m256i&>(converted);
  _mm256_mask_storeu_epi16(pointer, mask, bits);
}

inline __m256i Bf16Bits(__m512 value) {
  const __m256bh converted = _mm512_cvtneps_pbh(value);
  return reinterpret_cast<const __m256i&>(converted);
}

void StorePackedFeaturePair(__m512 feature0, __m512 feature1, uint16_t* destination) {
  const __m256i value0 = Bf16Bits(feature0);
  const __m256i value1 = Bf16Bits(feature1);
  const __m256i low = _mm256_unpacklo_epi16(value0, value1);
  const __m256i high = _mm256_unpackhi_epi16(value0, value1);
  const __m256i rows0_to7 = _mm256_permute2x128_si256(low, high, 0x20);
  const __m256i rows8_to15 = _mm256_permute2x128_si256(low, high, 0x31);
  __m512i packed = _mm512_castsi256_si512(rows0_to7);
  packed = _mm512_inserti64x4(packed, rows8_to15, 1);
  _mm512_storeu_si512(reinterpret_cast<void*>(destination), packed);
}

void W13SmallM12(const uint16_t* packed_a, const uint16_t* packed_b, uint16_t* packed_c, int k_pad, int feature_block,
                 int degree) {
  __m512 gate[12];
  __m512 up[12];
#pragma GCC unroll 12
  for (int row = 0; row < 12; ++row) {
    gate[row] = _mm512_setzero_ps();
    up[row] = _mm512_setzero_ps();
  }

  const uint16_t* b_block = packed_b + static_cast<int64_t>(feature_block) * k_pad * 32;
  for (int kp = 0; kp < k_pad / 2; ++kp) {
    const uint16_t* b_pointer = b_block + static_cast<int64_t>(kp) * 64;
    const __m512bh b_gate = AsBf16(_mm512_loadu_si512(reinterpret_cast<const void*>(b_pointer)));
    const __m512bh b_up = AsBf16(_mm512_loadu_si512(reinterpret_cast<const void*>(b_pointer + 32)));
    if (kp + 8 < k_pad / 2) {
      _mm_prefetch(reinterpret_cast<const char*>(b_pointer + 8 * 64), _MM_HINT_T0);
    }
#pragma GCC unroll 12
    for (int row = 0; row < 12; ++row) {
      const __m512bh a_pair = BroadcastBf16Pair(packed_a + static_cast<int64_t>(kp) * 32 + row * 2);
      gate[row] = _mm512_dpbf16_ps(gate[row], b_gate, a_pair);
      up[row] = _mm512_dpbf16_ps(up[row], b_up, a_pair);
    }
  }

  alignas(32) uint16_t row_values[12][16];
#pragma GCC unroll 12
  for (int row = 0; row < 12; ++row) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(row_values[row]), Bf16Bits(SiluMul(gate[row], up[row], degree)));
  }
  for (int feature = 0; feature < 16; ++feature) {
    uint16_t* pair_destination = packed_c + static_cast<int64_t>(feature_block * 8 + feature / 2) * 32 + (feature & 1);
#pragma GCC unroll 12
    for (int row = 0; row < 12; ++row) {
      pair_destination[row * 2] = row_values[row][feature];
    }
  }
}

void W2SmallM12(const uint16_t* packed_a, const uint16_t* packed_b, float* route_output, uint16_t* direct_output,
                const int64_t* route_ids, int route_stride, int k_pad, int hidden_size, int output_block,
                bool direct_bf16) {
  __m512 output0[12];
  __m512 output1[12];
#pragma GCC unroll 12
  for (int row = 0; row < 12; ++row) {
    output0[row] = _mm512_setzero_ps();
    output1[row] = _mm512_setzero_ps();
  }

  const uint16_t* b_block = packed_b + static_cast<int64_t>(output_block) * k_pad * 32;
  for (int kp = 0; kp < k_pad / 2; ++kp) {
    const uint16_t* b_pointer = b_block + static_cast<int64_t>(kp) * 64;
    const __m512bh b0 = AsBf16(_mm512_loadu_si512(reinterpret_cast<const void*>(b_pointer)));
    const __m512bh b1 = AsBf16(_mm512_loadu_si512(reinterpret_cast<const void*>(b_pointer + 32)));
    if (kp + 8 < k_pad / 2) {
      _mm_prefetch(reinterpret_cast<const char*>(b_pointer + 8 * 64), _MM_HINT_T0);
    }
#pragma GCC unroll 12
    for (int row = 0; row < 12; ++row) {
      const __m512bh a_pair = BroadcastBf16Pair(packed_a + static_cast<int64_t>(kp) * 32 + row * 2);
      output0[row] = _mm512_dpbf16_ps(output0[row], b0, a_pair);
      output1[row] = _mm512_dpbf16_ps(output1[row], b1, a_pair);
    }
  }

  const int column = output_block * 32;
  const int remaining = hidden_size - column;
  const __mmask16 mask0 =
      remaining >= 16 ? 0xffff : static_cast<__mmask16>((uint32_t{1} << std::max(remaining, 0)) - 1);
  const int remaining1 = remaining - 16;
  const __mmask16 mask1 =
      remaining1 >= 16 ? 0xffff : static_cast<__mmask16>((uint32_t{1} << std::max(remaining1, 0)) - 1);
#pragma GCC unroll 12
  for (int row = 0; row < 12; ++row) {
    const int64_t route = route_ids[row];
    if (direct_bf16) {
      uint16_t* destination = direct_output + route * route_stride + column;
      StoreBf16(output0[row], destination, mask0);
      if (remaining1 > 0) {
        StoreBf16(output1[row], destination + 16, mask1);
      }
    } else {
      float* destination = route_output + route * route_stride + column;
      _mm512_mask_storeu_ps(destination, mask0, output0[row]);
      if (remaining1 > 0) {
        _mm512_mask_storeu_ps(destination + 16, mask1, output1[row]);
      }
    }
  }
}

void W13PackedBlock(const uint16_t* packed_a, const uint16_t* packed_b, uint16_t* packed_c, int k_pad,
                    int feature_block, int feature_offset, int degree) {
  __m512 gate[8];
  __m512 up[8];
#pragma GCC unroll 8
  for (int feature = 0; feature < 8; ++feature) {
    gate[feature] = _mm512_setzero_ps();
    up[feature] = _mm512_setzero_ps();
  }

  const uint16_t* b_block = packed_b + static_cast<int64_t>(feature_block) * k_pad * 32;
  for (int kp = 0; kp < k_pad / 2; ++kp) {
    const __m512bh a_pair =
        AsBf16(_mm512_loadu_si512(reinterpret_cast<const void*>(packed_a + static_cast<int64_t>(kp) * 32)));
    const uint16_t* b_pointer = b_block + static_cast<int64_t>(kp) * 64 + feature_offset * 2;
    if (kp + 8 < k_pad / 2) {
      _mm_prefetch(reinterpret_cast<const char*>(packed_a + static_cast<int64_t>(kp + 8) * 32), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(b_pointer + 8 * 64), _MM_HINT_T0);
    }
#pragma GCC unroll 8
    for (int feature = 0; feature < 8; ++feature) {
      const __m512bh b_gate = BroadcastBf16Pair(b_pointer + feature * 2);
      const __m512bh b_up = BroadcastBf16Pair(b_pointer + 32 + feature * 2);
      gate[feature] = _mm512_dpbf16_ps(gate[feature], a_pair, b_gate);
      up[feature] = _mm512_dpbf16_ps(up[feature], a_pair, b_up);
    }
  }

  const int feature_pair_base = feature_block * 8 + feature_offset / 2;
#pragma GCC unroll 4
  for (int pair = 0; pair < 4; ++pair) {
    const int feature0 = pair * 2;
    const int feature1 = feature0 + 1;
    StorePackedFeaturePair(SiluMul(gate[feature0], up[feature0], degree), SiluMul(gate[feature1], up[feature1], degree),
                           packed_c + static_cast<int64_t>(feature_pair_base + pair) * 32);
  }
}

void W2PackedBlock(const uint16_t* packed_a, const uint16_t* packed_b, float* route_output, uint16_t* direct_output,
                   const int64_t* route_ids, int route_stride, int rows, int k_pad, int hidden_size, int output_block,
                   int output_offset, bool direct_bf16) {
  __m512 output[8];
#pragma GCC unroll 8
  for (int column = 0; column < 8; ++column) {
    output[column] = _mm512_setzero_ps();
  }

  const uint16_t* b_block = packed_b + static_cast<int64_t>(output_block) * k_pad * 32;
  for (int kp = 0; kp < k_pad / 2; ++kp) {
    const __m512bh a_pair =
        AsBf16(_mm512_loadu_si512(reinterpret_cast<const void*>(packed_a + static_cast<int64_t>(kp) * 32)));
    const uint16_t* b_pointer = b_block + static_cast<int64_t>(kp) * 64 + output_offset * 2;
    if (kp + 8 < k_pad / 2) {
      _mm_prefetch(reinterpret_cast<const char*>(packed_a + static_cast<int64_t>(kp + 8) * 32), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(b_pointer + 8 * 64), _MM_HINT_T0);
    }
#pragma GCC unroll 8
    for (int column = 0; column < 8; ++column) {
      const __m512bh b_pair = BroadcastBf16Pair(b_pointer + column * 2);
      output[column] = _mm512_dpbf16_ps(output[column], a_pair, b_pair);
    }
  }

  const int valid_rows = std::min(rows, 16);
  const __mmask16 row_mask = valid_rows == 16 ? 0xffff : static_cast<__mmask16>((uint32_t{1} << valid_rows) - 1);
  const int column_base = output_block * 32 + output_offset;
  const int valid_columns = std::min(8, hidden_size - column_base);
  alignas(64) int32_t route_offsets[16] = {};
  for (int row = 0; row < valid_rows; ++row) {
    route_offsets[row] = static_cast<int32_t>(route_ids[row] * route_stride + column_base);
  }
  const __m512i offsets = _mm512_load_si512(route_offsets);
  for (int column = 0; column < valid_columns; ++column) {
    if (direct_bf16) {
      alignas(32) uint16_t values[16];
      _mm256_store_si256(reinterpret_cast<__m256i*>(values), Bf16Bits(output[column]));
      for (int row = 0; row < valid_rows; ++row) {
        direct_output[static_cast<int64_t>(route_ids[row]) * route_stride + column_base + column] = values[row];
      }
    } else {
      _mm512_mask_i32scatter_ps(route_output + column, row_mask, offsets, output[column], 4);
    }
  }
}

}  // namespace

void ComputeW13(const uint16_t* a, int a_stride, const uint16_t* packed_b, uint16_t* c, int c_stride, int rows,
                int k_pad, int feature_block_begin, int feature_block_end, int silu_poly_degree) {
  const int full_panels = rows / 12;
  const int tail_rows = rows % 12;
  for (int block = feature_block_begin; block < feature_block_end; ++block) {
    for (int panel = 0; panel < full_panels; ++panel) {
      W13SmallM12(a + static_cast<int64_t>(panel) * a_stride * 16, packed_b,
                  c + static_cast<int64_t>(panel) * c_stride * 16, k_pad, block, silu_poly_degree);
    }
    if (tail_rows == 0) {
      continue;
    }
    const uint16_t* tail_a = a + static_cast<int64_t>(full_panels) * a_stride * 16;
    uint16_t* tail_c = c + static_cast<int64_t>(full_panels) * c_stride * 16;
    for (int feature_offset = 0; feature_offset < 16; feature_offset += 8) {
      W13PackedBlock(tail_a, packed_b, tail_c, k_pad, block, feature_offset, silu_poly_degree);
    }
  }
}

void ComputeW2(const uint16_t* a, int a_stride, const uint16_t* packed_b, float* route_output, uint16_t* direct_output,
               const int64_t* route_ids, int route_stride, int rows, int k_pad, int hidden_size, int output_block_begin,
               int output_block_end, bool direct_bf16) {
  const int full_panels = rows / 12;
  const int tail_rows = rows % 12;
  for (int block = output_block_begin; block < output_block_end; ++block) {
    for (int panel = 0; panel < full_panels; ++panel) {
      W2SmallM12(a + static_cast<int64_t>(panel) * a_stride * 16, packed_b, route_output, direct_output,
                 route_ids + panel * 12, route_stride, k_pad, hidden_size, block, direct_bf16);
    }
    if (tail_rows == 0) {
      continue;
    }
    const uint16_t* tail_a = a + static_cast<int64_t>(full_panels) * a_stride * 16;
    const int64_t* tail_routes = route_ids + full_panels * 12;
    for (int output_offset = 0; output_offset < 32; output_offset += 8) {
      if (block * 32 + output_offset >= hidden_size) {
        break;
      }
      W2PackedBlock(tail_a, packed_b, route_output, direct_output, tail_routes, route_stride, tail_rows, k_pad,
                    hidden_size, block, output_offset, direct_bf16);
    }
  }
}

void MergeRoutes(const float* route_output, const float* weights, uint16_t* output, int64_t token_begin,
                 int64_t token_end, int64_t top_k, int64_t hidden_size) {
  for (int64_t token = token_begin; token < token_end; ++token) {
    int64_t hidden = 0;
    for (; hidden + 16 <= hidden_size; hidden += 16) {
      __m512 sum = _mm512_setzero_ps();
      for (int64_t route = 0; route < top_k; ++route) {
        const int64_t flat = token * top_k + route;
        const __m512 value = _mm512_loadu_ps(route_output + flat * hidden_size + hidden);
        sum = _mm512_fmadd_ps(value, _mm512_set1_ps(weights[flat]), sum);
      }
      StoreBf16(sum, output + token * hidden_size + hidden);
    }
    if (hidden < hidden_size) {
      const __mmask16 mask = static_cast<__mmask16>((uint32_t{1} << (hidden_size - hidden)) - 1);
      __m512 sum = _mm512_setzero_ps();
      for (int64_t route = 0; route < top_k; ++route) {
        const int64_t flat = token * top_k + route;
        const __m512 value = _mm512_maskz_loadu_ps(mask, route_output + flat * hidden_size + hidden);
        sum = _mm512_fmadd_ps(value, _mm512_set1_ps(weights[flat]), sum);
      }
      StoreBf16(sum, output + token * hidden_size + hidden, mask);
    }
  }
}

}  // namespace fused_cpp::moe::x86::avx512_bf16
