// SPDX-License-Identifier: Apache-2.0

#include <arm_sve.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Config {
  int64_t rows = 2048;
  int64_t head_dim = 192;
  int64_t rope_dim = 64;
  int64_t heads = 1;
  int threads = 1;
  int warmup = 20;
  int runs = 51;
  int inner = 20;
  int unroll = 1;
  bool qnorm_only = false;
};

struct Accuracy {
  double max_abs = 0.0;
  double relative_l2 = 0.0;
  int64_t bf16_mismatches = 0;
};

struct Timing {
  double baseline_us = 0.0;
  double candidate_us = 0.0;
  double paired_gain = 0.0;
};

uint16_t float_to_bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

float bf16_to_float(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

svfloat32_t bf16_load_contiguous_f32(svbool_t pg, const uint16_t* source) {
  const svuint32_t values = svld1uh_u32(pg, source);
  return svreinterpret_f32_u32(svlsl_n_u32_x(pg, values, 16));
}

svfloat32_t bf16_u16_to_f32_low(svbool_t pg, svuint16_t values) {
  const svuint32_t wide = svunpklo_u32(values);
  return svreinterpret_f32_u32(svlsl_n_u32_x(pg, wide, 16));
}

svfloat32x2_t bf16_load_evenodd_f32(svbool_t pg, svbool_t pg_h, const uint16_t* source) {
  const svuint16x2_t values = svld2_u16(pg_h, source);
  return svcreate2_f32(bf16_u16_to_f32_low(pg, svget2_u16(values, 0)),
                       bf16_u16_to_f32_low(pg, svget2_u16(values, 1)));
}

svuint32_t f32_to_bf16_bits(svbool_t pg, svfloat32_t value) {
  const svuint32_t bits = svreinterpret_u32_f32(value);
  const svuint32_t lsb = svand_n_u32_x(pg, svlsr_n_u32_x(pg, bits, 16), 1);
  const svuint32_t bias = svadd_n_u32_x(pg, lsb, 0x7fff);
  return svlsr_n_u32_x(pg, svadd_u32_x(pg, bits, bias), 16);
}

void bf16_store_contiguous_f32(svbool_t pg, uint16_t* destination, svfloat32_t value) {
  svst1h_u32(pg, destination, f32_to_bf16_bits(pg, value));
}

void bf16_store_evenodd_f32(svbool_t pg, uint16_t* destination, svfloat32_t even, svfloat32_t odd) {
  const svuint32_t indices = svindex_u32(0, 2);
  svst1h_scatter_u32index_u32(pg, destination, indices, f32_to_bf16_bits(pg, even));
  svst1h_scatter_u32index_u32(pg, destination + 1, indices, f32_to_bf16_bits(pg, odd));
}

void rope_split_f32(float* rows, const int32_t* positions, const float* cache, int64_t row_count, int64_t rope_dim,
                    int threads) {
  const int64_t half = rope_dim / 2;
#pragma omp parallel for schedule(static) num_threads(threads)
  for (int64_t row = 0; row < row_count; ++row) {
    float* values = rows + row * rope_dim;
    const float* cache_row = cache + static_cast<int64_t>(positions[row]) * rope_dim;
    const float* cosine = cache_row;
    const float* sine = cache_row + half;
    for (int64_t pair = 0; pair < half; pair += static_cast<int64_t>(svcntw())) {
      const svbool_t pg = svwhilelt_b32(pair, half);
      const svfloat32x2_t input = svld2_f32(pg, values + 2 * pair);
      const svfloat32_t even = svget2_f32(input, 0);
      const svfloat32_t odd = svget2_f32(input, 1);
      const svfloat32_t c = svld1_f32(pg, cosine + pair);
      const svfloat32_t s = svld1_f32(pg, sine + pair);
      const svfloat32_t out_even = svmls_f32_x(pg, svmul_f32_x(pg, even, c), odd, s);
      const svfloat32_t out_odd = svmla_f32_x(pg, svmul_f32_x(pg, odd, c), even, s);
      svst2_f32(pg, values + 2 * pair, svcreate2_f32(out_even, out_odd));
    }
  }
}

template <int Unroll>
void rope_fcmla_f32(float* rows, const int32_t* positions, const float* interleaved_cache, int64_t row_count,
                    int64_t rope_dim, int threads) {
  static_assert(Unroll == 1 || Unroll == 2 || Unroll == 4 || Unroll == 8);
  const int64_t vl = static_cast<int64_t>(svcntw());
  const svfloat32_t zero = svdup_f32(0.0f);
  const svbool_t pg_all = svptrue_b32();
#pragma omp parallel for schedule(static) num_threads(threads)
  for (int64_t row = 0; row < row_count; ++row) {
    float* values = rows + row * rope_dim;
    const float* cache_row = interleaved_cache + static_cast<int64_t>(positions[row]) * rope_dim;
    int64_t offset = 0;
    for (; offset + Unroll * vl <= rope_dim; offset += Unroll * vl) {
#pragma GCC unroll 8
      for (int slot = 0; slot < Unroll; ++slot) {
        const int64_t slot_offset = offset + slot * vl;
        const svfloat32_t input = svld1_f32(pg_all, values + slot_offset);
        const svfloat32_t twiddle = svld1_f32(pg_all, cache_row + slot_offset);
        svfloat32_t output = svcmla_f32_x(pg_all, zero, input, twiddle, 0);
        output = svcmla_f32_x(pg_all, output, input, twiddle, 90);
        svst1_f32(pg_all, values + slot_offset, output);
      }
    }
    for (; offset < rope_dim; offset += vl) {
      const svbool_t pg = svwhilelt_b32(offset, rope_dim);
      const svfloat32_t input = svld1_f32(pg, values + offset);
      const svfloat32_t twiddle = svld1_f32(pg, cache_row + offset);
      svfloat32_t output = svcmla_f32_x(pg, zero, input, twiddle, 0);
      output = svcmla_f32_x(pg, output, input, twiddle, 90);
      svst1_f32(pg, values + offset, output);
    }
  }
}

void rope_split_bf16(uint16_t* rows, const int32_t* positions, const float* cache, int64_t row_count,
                     int64_t rope_dim, int threads) {
  const int64_t half = rope_dim / 2;
  const int64_t vl = static_cast<int64_t>(svcntw());
#pragma omp parallel for schedule(static) num_threads(threads)
  for (int64_t row = 0; row < row_count; ++row) {
    uint16_t* values = rows + row * rope_dim;
    const float* cache_row = cache + static_cast<int64_t>(positions[row]) * rope_dim;
    const float* cosine = cache_row;
    const float* sine = cache_row + half;
    for (int64_t pair = 0; pair < half; pair += vl) {
      const int64_t active = std::min(vl, half - pair);
      const svbool_t pg = svwhilelt_b32(static_cast<int64_t>(0), active);
      const svbool_t pg_h = svwhilelt_b16(static_cast<int64_t>(0), active);
      const svfloat32x2_t input = bf16_load_evenodd_f32(pg, pg_h, values + 2 * pair);
      const svfloat32_t even = svget2_f32(input, 0);
      const svfloat32_t odd = svget2_f32(input, 1);
      const svfloat32_t c = svld1_f32(pg, cosine + pair);
      const svfloat32_t s = svld1_f32(pg, sine + pair);
      const svfloat32_t out_even = svmls_f32_x(pg, svmul_f32_x(pg, even, c), odd, s);
      const svfloat32_t out_odd = svmla_f32_x(pg, svmul_f32_x(pg, odd, c), even, s);
      bf16_store_evenodd_f32(pg, values + 2 * pair, out_even, out_odd);
    }
  }
}

void rope_split_interleaved_bf16(uint16_t* rows, const int32_t* positions, const float* interleaved_cache,
                                 int64_t row_count, int64_t rope_dim, int threads) {
  const int64_t half = rope_dim / 2;
  const int64_t vl = static_cast<int64_t>(svcntw());
#pragma omp parallel for schedule(static) num_threads(threads)
  for (int64_t row = 0; row < row_count; ++row) {
    uint16_t* values = rows + row * rope_dim;
    const float* cache_row = interleaved_cache + static_cast<int64_t>(positions[row]) * rope_dim;
    for (int64_t pair = 0; pair < half; pair += vl) {
      const int64_t active = std::min(vl, half - pair);
      const svbool_t pg = svwhilelt_b32(static_cast<int64_t>(0), active);
      const svbool_t pg_h = svwhilelt_b16(static_cast<int64_t>(0), active);
      const svfloat32x2_t input = bf16_load_evenodd_f32(pg, pg_h, values + 2 * pair);
      const svfloat32x2_t twiddle = svld2_f32(pg, cache_row + 2 * pair);
      const svfloat32_t even = svget2_f32(input, 0);
      const svfloat32_t odd = svget2_f32(input, 1);
      const svfloat32_t c = svget2_f32(twiddle, 0);
      const svfloat32_t s = svget2_f32(twiddle, 1);
      const svfloat32_t out_even = svmls_f32_x(pg, svmul_f32_x(pg, even, c), odd, s);
      const svfloat32_t out_odd = svmla_f32_x(pg, svmul_f32_x(pg, odd, c), even, s);
      bf16_store_evenodd_f32(pg, values + 2 * pair, out_even, out_odd);
    }
  }
}

template <int Unroll>
void rope_fcmla_bf16(uint16_t* rows, const int32_t* positions, const float* interleaved_cache, int64_t row_count,
                     int64_t rope_dim, int threads) {
  static_assert(Unroll == 1 || Unroll == 2 || Unroll == 4 || Unroll == 8);
  const int64_t vl = static_cast<int64_t>(svcntw());
  const svfloat32_t zero = svdup_f32(0.0f);
  const svbool_t pg_all = svptrue_b32();
#pragma omp parallel for schedule(static) num_threads(threads)
  for (int64_t row = 0; row < row_count; ++row) {
    uint16_t* values = rows + row * rope_dim;
    const float* cache_row = interleaved_cache + static_cast<int64_t>(positions[row]) * rope_dim;
    int64_t offset = 0;
    for (; offset + Unroll * vl <= rope_dim; offset += Unroll * vl) {
#pragma GCC unroll 8
      for (int slot = 0; slot < Unroll; ++slot) {
        const int64_t slot_offset = offset + slot * vl;
        const svfloat32_t input = bf16_load_contiguous_f32(pg_all, values + slot_offset);
        const svfloat32_t twiddle = svld1_f32(pg_all, cache_row + slot_offset);
        svfloat32_t output = svcmla_f32_x(pg_all, zero, input, twiddle, 0);
        output = svcmla_f32_x(pg_all, output, input, twiddle, 90);
        bf16_store_contiguous_f32(pg_all, values + slot_offset, output);
      }
    }
    for (; offset < rope_dim; offset += vl) {
      const svbool_t pg = svwhilelt_b32(offset, rope_dim);
      const svfloat32_t input = bf16_load_contiguous_f32(pg, values + offset);
      const svfloat32_t twiddle = svld1_f32(pg, cache_row + offset);
      svfloat32_t output = svcmla_f32_x(pg, zero, input, twiddle, 0);
      output = svcmla_f32_x(pg, output, input, twiddle, 90);
      bf16_store_contiguous_f32(pg, values + offset, output);
    }
  }
}

float inverse_rms_bf16(const uint16_t* row, int64_t head_dim) {
  const svbool_t pg_all = svptrue_b32();
  svfloat32_t sum = svdup_f32(0.0f);
  int64_t dimension = 0;
  for (; dimension + static_cast<int64_t>(svcntw()) <= head_dim; dimension += static_cast<int64_t>(svcntw())) {
    const svfloat32_t value = bf16_load_contiguous_f32(pg_all, row + dimension);
    sum = svmla_f32_x(pg_all, sum, value, value);
  }
  if (dimension < head_dim) {
    const svbool_t pg = svwhilelt_b32(dimension, head_dim);
    const svfloat32_t value = bf16_load_contiguous_f32(pg, row + dimension);
    sum = svmla_f32_x(pg, sum, value, value);
  }
  return 1.0f /
         std::sqrt(svaddv_f32(pg_all, sum) / static_cast<float>(head_dim) + 1.0e-6f);
}

void qnorm_rope_split_bf16(uint16_t* rows, const int32_t* positions, const float* cache, int64_t row_count,
                           int64_t head_dim, int64_t rope_dim, int threads) {
  const int64_t nope_dim = head_dim - rope_dim;
  const int64_t half = rope_dim / 2;
  const int64_t vl = static_cast<int64_t>(svcntw());
#pragma omp parallel for schedule(static) num_threads(threads)
  for (int64_t row = 0; row < row_count; ++row) {
    uint16_t* values = rows + row * head_dim;
    const svfloat32_t inv = svdup_f32(inverse_rms_bf16(values, head_dim));
    for (int64_t dimension = 0; dimension < nope_dim; dimension += vl) {
      const svbool_t pg = svwhilelt_b32(dimension, nope_dim);
      const svfloat32_t value = bf16_load_contiguous_f32(pg, values + dimension);
      bf16_store_contiguous_f32(pg, values + dimension, svmul_f32_x(pg, value, inv));
    }
    const float* cache_row = cache + static_cast<int64_t>(positions[row]) * rope_dim;
    const float* cosine = cache_row;
    const float* sine = cache_row + half;
    for (int64_t pair = 0; pair < half; pair += vl) {
      const int64_t active = std::min(vl, half - pair);
      const svbool_t pg = svwhilelt_b32(static_cast<int64_t>(0), active);
      const svbool_t pg_h = svwhilelt_b16(static_cast<int64_t>(0), active);
      const svfloat32x2_t input = bf16_load_evenodd_f32(pg, pg_h, values + nope_dim + 2 * pair);
      const svfloat32_t even = svmul_f32_x(pg, svget2_f32(input, 0), inv);
      const svfloat32_t odd = svmul_f32_x(pg, svget2_f32(input, 1), inv);
      const svfloat32_t c = svld1_f32(pg, cosine + pair);
      const svfloat32_t s = svld1_f32(pg, sine + pair);
      const svfloat32_t out_even = svmls_f32_x(pg, svmul_f32_x(pg, even, c), odd, s);
      const svfloat32_t out_odd = svmla_f32_x(pg, svmul_f32_x(pg, odd, c), even, s);
      bf16_store_evenodd_f32(pg, values + nope_dim + 2 * pair, out_even, out_odd);
    }
  }
}

template <int Unroll>
void qnorm_rope_fcmla_bf16(uint16_t* rows, const int32_t* positions, const float* interleaved_cache,
                           int64_t row_count, int64_t head_dim, int64_t rope_dim, int threads) {
  static_assert(Unroll == 1 || Unroll == 2 || Unroll == 4 || Unroll == 8);
  const int64_t nope_dim = head_dim - rope_dim;
  const int64_t vl = static_cast<int64_t>(svcntw());
  const svfloat32_t zero = svdup_f32(0.0f);
  const svbool_t pg_all = svptrue_b32();
#pragma omp parallel for schedule(static) num_threads(threads)
  for (int64_t row = 0; row < row_count; ++row) {
    uint16_t* values = rows + row * head_dim;
    const svfloat32_t inv = svdup_f32(inverse_rms_bf16(values, head_dim));
    for (int64_t dimension = 0; dimension < nope_dim; dimension += vl) {
      const svbool_t pg = svwhilelt_b32(dimension, nope_dim);
      const svfloat32_t value = bf16_load_contiguous_f32(pg, values + dimension);
      bf16_store_contiguous_f32(pg, values + dimension, svmul_f32_x(pg, value, inv));
    }
    const float* cache_row = interleaved_cache + static_cast<int64_t>(positions[row]) * rope_dim;
    int64_t offset = 0;
    for (; offset + Unroll * vl <= rope_dim; offset += Unroll * vl) {
#pragma GCC unroll 8
      for (int slot = 0; slot < Unroll; ++slot) {
        const int64_t slot_offset = offset + slot * vl;
        const svfloat32_t input =
            svmul_f32_x(pg_all, bf16_load_contiguous_f32(pg_all, values + nope_dim + slot_offset), inv);
        const svfloat32_t twiddle = svld1_f32(pg_all, cache_row + slot_offset);
        svfloat32_t output = svcmla_f32_x(pg_all, zero, input, twiddle, 0);
        output = svcmla_f32_x(pg_all, output, input, twiddle, 90);
        bf16_store_contiguous_f32(pg_all, values + nope_dim + slot_offset, output);
      }
    }
    for (; offset < rope_dim; offset += vl) {
      const svbool_t pg = svwhilelt_b32(offset, rope_dim);
      const svfloat32_t input =
          svmul_f32_x(pg, bf16_load_contiguous_f32(pg, values + nope_dim + offset), inv);
      const svfloat32_t twiddle = svld1_f32(pg, cache_row + offset);
      svfloat32_t output = svcmla_f32_x(pg, zero, input, twiddle, 0);
      output = svcmla_f32_x(pg, output, input, twiddle, 90);
      bf16_store_contiguous_f32(pg, values + nope_dim + offset, output);
    }
  }
}

template <typename Function>
decltype(auto) dispatch_unroll(int unroll, Function&& function) {
  switch (unroll) {
  case 1:
    return function(std::integral_constant<int, 1>{});
  case 2:
    return function(std::integral_constant<int, 2>{});
  case 4:
    return function(std::integral_constant<int, 4>{});
  case 8:
    return function(std::integral_constant<int, 8>{});
  default:
    throw std::invalid_argument("unroll must be one of 1, 2, 4, or 8");
  }
}

template <typename T>
Accuracy compare(const std::vector<T>& reference, const std::vector<T>& actual) {
  long double reference_l2 = 0.0;
  long double difference_l2 = 0.0;
  Accuracy result;
  for (size_t index = 0; index < reference.size(); ++index) {
    const float ref = [&] {
      if constexpr (std::is_same_v<T, float>) {
        return reference[index];
      } else {
        return bf16_to_float(reference[index]);
      }
    }();
    const float value = [&] {
      if constexpr (std::is_same_v<T, float>) {
        return actual[index];
      } else {
        return bf16_to_float(actual[index]);
      }
    }();
    const double difference = static_cast<double>(value) - ref;
    result.max_abs = std::max(result.max_abs, std::abs(difference));
    reference_l2 += static_cast<long double>(ref) * ref;
    difference_l2 += static_cast<long double>(difference) * difference;
    if constexpr (!std::is_same_v<T, float>) {
      result.bf16_mismatches += actual[index] != reference[index] ? 1 : 0;
    }
  }
  result.relative_l2 = std::sqrt(static_cast<double>(difference_l2 / std::max(reference_l2, 1.0e-30L)));
  return result;
}

double median(std::vector<double> values) {
  const size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle), values.end());
  return values[middle];
}

template <typename Baseline, typename Candidate>
Timing measure_pair(Baseline&& baseline, Candidate&& candidate, int warmup, int runs, int inner) {
  for (int iteration = 0; iteration < warmup; ++iteration) {
    baseline();
    candidate();
  }
  std::vector<double> baseline_samples;
  std::vector<double> candidate_samples;
  std::vector<double> paired_gains;
  baseline_samples.reserve(static_cast<size_t>(runs));
  candidate_samples.reserve(static_cast<size_t>(runs));
  paired_gains.reserve(static_cast<size_t>(runs));
  auto time = [&](auto&& function) {
    const auto begin = Clock::now();
    for (int iteration = 0; iteration < inner; ++iteration) {
      function();
    }
    const auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - begin).count() / inner;
  };
  for (int run = 0; run < runs; ++run) {
    double baseline_time = 0.0;
    double candidate_time = 0.0;
    if ((run & 1) == 0) {
      baseline_time = time(baseline);
      candidate_time = time(candidate);
    } else {
      candidate_time = time(candidate);
      baseline_time = time(baseline);
    }
    baseline_samples.push_back(baseline_time);
    candidate_samples.push_back(candidate_time);
    paired_gains.push_back(baseline_time / candidate_time - 1.0);
  }
  return {median(std::move(baseline_samples)), median(std::move(candidate_samples)), median(std::move(paired_gains))};
}

void fill_data(const Config& config, std::vector<float>& source_f32, std::vector<uint16_t>& source_bf16,
               std::vector<int32_t>& positions, std::vector<float>& split_cache,
               std::vector<float>& interleaved_cache) {
  constexpr int64_t kPositions = 4096;
  std::mt19937 generator(20260820);
  std::uniform_real_distribution<float> values(-2.0f, 2.0f);
  source_f32.resize(static_cast<size_t>(config.rows * config.rope_dim));
  source_bf16.resize(source_f32.size());
  for (size_t index = 0; index < source_f32.size(); ++index) {
    source_f32[index] = values(generator);
    source_bf16[index] = float_to_bf16(source_f32[index]);
  }
  positions.resize(static_cast<size_t>(config.rows));
  for (int64_t row = 0; row < config.rows; ++row) {
    positions[static_cast<size_t>(row)] = static_cast<int32_t>(((row / config.heads) * 37) % kPositions);
  }
  split_cache.resize(static_cast<size_t>(kPositions * config.rope_dim));
  interleaved_cache.resize(split_cache.size());
  const int64_t half = config.rope_dim / 2;
  for (int64_t position = 0; position < kPositions; ++position) {
    for (int64_t pair = 0; pair < half; ++pair) {
      const float angle = static_cast<float>(position) * std::pow(10000.0f, -2.0f * pair / config.rope_dim);
      const float c = std::cos(angle);
      const float s = std::sin(angle);
      split_cache[static_cast<size_t>(position * config.rope_dim + pair)] = c;
      split_cache[static_cast<size_t>(position * config.rope_dim + half + pair)] = s;
      interleaved_cache[static_cast<size_t>(position * config.rope_dim + 2 * pair)] = c;
      interleaved_cache[static_cast<size_t>(position * config.rope_dim + 2 * pair + 1)] = s;
    }
  }
}

template <typename T>
std::vector<T> scalar_reference(const std::vector<T>& source, const std::vector<int32_t>& positions,
                                const std::vector<float>& split_cache, const Config& config) {
  std::vector<T> output = source;
  const int64_t half = config.rope_dim / 2;
  for (int64_t row = 0; row < config.rows; ++row) {
    const float* cache =
        split_cache.data() + static_cast<int64_t>(positions[static_cast<size_t>(row)]) * config.rope_dim;
    for (int64_t pair = 0; pair < half; ++pair) {
      const size_t even_index = static_cast<size_t>(row * config.rope_dim + 2 * pair);
      const size_t odd_index = even_index + 1;
      const float even = [&] {
        if constexpr (std::is_same_v<T, float>) {
          return source[even_index];
        } else {
          return bf16_to_float(source[even_index]);
        }
      }();
      const float odd = [&] {
        if constexpr (std::is_same_v<T, float>) {
          return source[odd_index];
        } else {
          return bf16_to_float(source[odd_index]);
        }
      }();
      const float out_even = even * cache[pair] - odd * cache[half + pair];
      const float out_odd = odd * cache[pair] + even * cache[half + pair];
      if constexpr (std::is_same_v<T, float>) {
        output[even_index] = out_even;
        output[odd_index] = out_odd;
      } else {
        output[even_index] = float_to_bf16(out_even);
        output[odd_index] = float_to_bf16(out_odd);
      }
    }
  }
  return output;
}

Config parse_args(int argc, char** argv) {
  Config config;
  auto require_value = [&](int& index) -> const char* {
    if (++index >= argc) {
      throw std::invalid_argument("missing option value");
    }
    return argv[index];
  };
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (option == "--rows") {
      config.rows = std::stoll(require_value(index));
    } else if (option == "--head-dim") {
      config.head_dim = std::stoll(require_value(index));
    } else if (option == "--rope-dim") {
      config.rope_dim = std::stoll(require_value(index));
    } else if (option == "--heads") {
      config.heads = std::stoll(require_value(index));
    } else if (option == "--threads") {
      config.threads = std::stoi(require_value(index));
    } else if (option == "--warmup") {
      config.warmup = std::stoi(require_value(index));
    } else if (option == "--runs") {
      config.runs = std::stoi(require_value(index));
    } else if (option == "--inner") {
      config.inner = std::stoi(require_value(index));
    } else if (option == "--unroll") {
      config.unroll = std::stoi(require_value(index));
    } else if (option == "--qnorm-only") {
      config.qnorm_only = true;
    } else {
      throw std::invalid_argument("unknown option: " + option);
    }
  }
  if (config.rows <= 0 || config.head_dim < config.rope_dim || config.rope_dim <= 0 || config.rope_dim % 2 != 0 ||
      config.heads <= 0 || config.rows % config.heads != 0 || config.threads <= 0 || config.warmup < 0 ||
      config.runs <= 0 || config.inner <= 0 ||
      (config.unroll != 1 && config.unroll != 2 && config.unroll != 4 && config.unroll != 8)) {
    throw std::invalid_argument("invalid benchmark configuration");
  }
  return config;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Config config = parse_args(argc, argv);
    omp_set_dynamic(0);
    std::vector<float> source_f32;
    std::vector<uint16_t> source_bf16;
    std::vector<int32_t> positions;
    std::vector<float> split_cache;
    std::vector<float> interleaved_cache;
    fill_data(config, source_f32, source_bf16, positions, split_cache, interleaved_cache);

    std::mt19937 q_generator(20260821);
    std::uniform_real_distribution<float> q_values(-2.0f, 2.0f);
    std::vector<uint16_t> source_q_bf16(static_cast<size_t>(config.rows * config.head_dim));
    for (uint16_t& value : source_q_bf16) {
      value = float_to_bf16(q_values(q_generator));
    }

    const std::vector<float> reference_f32 = scalar_reference(source_f32, positions, split_cache, config);
    const std::vector<uint16_t> reference_bf16 = scalar_reference(source_bf16, positions, split_cache, config);
    std::vector<float> check_split_f32 = source_f32;
    std::vector<float> check_fcmla_f32 = source_f32;
    std::vector<uint16_t> check_split_bf16 = source_bf16;
    std::vector<uint16_t> check_split_interleaved_bf16 = source_bf16;
    std::vector<uint16_t> check_fcmla_bf16 = source_bf16;
    rope_split_f32(check_split_f32.data(), positions.data(), split_cache.data(), config.rows, config.rope_dim, 1);
    dispatch_unroll(config.unroll, [&](auto unroll) {
      rope_fcmla_f32<unroll.value>(check_fcmla_f32.data(), positions.data(), interleaved_cache.data(), config.rows,
                                   config.rope_dim, 1);
    });
    rope_split_bf16(check_split_bf16.data(), positions.data(), split_cache.data(), config.rows, config.rope_dim, 1);
    rope_split_interleaved_bf16(check_split_interleaved_bf16.data(), positions.data(), interleaved_cache.data(),
                                config.rows, config.rope_dim, 1);
    dispatch_unroll(config.unroll, [&](auto unroll) {
      rope_fcmla_bf16<unroll.value>(check_fcmla_bf16.data(), positions.data(), interleaved_cache.data(), config.rows,
                                    config.rope_dim, 1);
    });
    std::vector<uint16_t> check_split_q_bf16 = source_q_bf16;
    std::vector<uint16_t> check_fcmla_q_bf16 = source_q_bf16;
    qnorm_rope_split_bf16(check_split_q_bf16.data(), positions.data(), split_cache.data(), config.rows, config.head_dim,
                          config.rope_dim, 1);
    dispatch_unroll(config.unroll, [&](auto unroll) {
      qnorm_rope_fcmla_bf16<unroll.value>(check_fcmla_q_bf16.data(), positions.data(), interleaved_cache.data(),
                                          config.rows, config.head_dim, config.rope_dim, 1);
    });

    const Accuracy split_f32_accuracy = compare(reference_f32, check_split_f32);
    const Accuracy fcmla_f32_accuracy = compare(reference_f32, check_fcmla_f32);
    const Accuracy split_bf16_accuracy = compare(reference_bf16, check_split_bf16);
    const Accuracy split_interleaved_bf16_accuracy = compare(reference_bf16, check_split_interleaved_bf16);
    const Accuracy fcmla_bf16_accuracy = compare(reference_bf16, check_fcmla_bf16);
    const Accuracy qnorm_fcmla_accuracy = compare(check_split_q_bf16, check_fcmla_q_bf16);
    if (split_f32_accuracy.max_abs > 1.0e-5 || fcmla_f32_accuracy.max_abs > 1.0e-5 ||
        split_bf16_accuracy.max_abs > 0.015625 || split_interleaved_bf16_accuracy.max_abs > 0.015625 ||
        fcmla_bf16_accuracy.max_abs > 0.015625 ||
        qnorm_fcmla_accuracy.max_abs > 0.015625) {
      throw std::runtime_error("RoPE correctness threshold failed");
    }

    std::vector<float> bench_split_f32 = source_f32;
    std::vector<float> bench_fcmla_f32 = source_f32;
    std::vector<uint16_t> bench_split_bf16 = source_bf16;
    std::vector<uint16_t> bench_split_interleaved_bf16 = source_bf16;
    std::vector<uint16_t> bench_fcmla_bf16 = source_bf16;
    std::vector<uint16_t> bench_split_q_bf16 = source_q_bf16;
    std::vector<uint16_t> bench_fcmla_q_bf16 = source_q_bf16;
    Timing f32_timing;
    Timing bf16_timing;
    Timing interleaved_control_timing;
    if (!config.qnorm_only) {
      f32_timing = measure_pair(
          [&] { rope_split_f32(bench_split_f32.data(), positions.data(), split_cache.data(), config.rows,
                               config.rope_dim, config.threads); },
          [&] {
            dispatch_unroll(config.unroll, [&](auto unroll) {
              rope_fcmla_f32<unroll.value>(bench_fcmla_f32.data(), positions.data(), interleaved_cache.data(),
                                           config.rows, config.rope_dim, config.threads);
            });
          },
          config.warmup, config.runs, config.inner);
      bf16_timing = measure_pair(
          [&] { rope_split_bf16(bench_split_bf16.data(), positions.data(), split_cache.data(), config.rows,
                                config.rope_dim, config.threads); },
          [&] {
            dispatch_unroll(config.unroll, [&](auto unroll) {
              rope_fcmla_bf16<unroll.value>(bench_fcmla_bf16.data(), positions.data(), interleaved_cache.data(),
                                            config.rows, config.rope_dim, config.threads);
            });
          },
          config.warmup, config.runs, config.inner);
      interleaved_control_timing = measure_pair(
          [&] { rope_split_interleaved_bf16(bench_split_interleaved_bf16.data(), positions.data(),
                                            interleaved_cache.data(), config.rows, config.rope_dim, config.threads); },
          [&] {
            dispatch_unroll(config.unroll, [&](auto unroll) {
              rope_fcmla_bf16<unroll.value>(bench_fcmla_bf16.data(), positions.data(), interleaved_cache.data(),
                                            config.rows, config.rope_dim, config.threads);
            });
          },
          config.warmup, config.runs, config.inner);
    }
    const Timing qnorm_timing = measure_pair(
        [&] { qnorm_rope_split_bf16(bench_split_q_bf16.data(), positions.data(), split_cache.data(), config.rows,
                                    config.head_dim, config.rope_dim, config.threads); },
        [&] {
          dispatch_unroll(config.unroll, [&](auto unroll) {
            qnorm_rope_fcmla_bf16<unroll.value>(bench_fcmla_q_bf16.data(), positions.data(),
                                                interleaved_cache.data(), config.rows, config.head_dim,
                                                config.rope_dim, config.threads);
          });
        },
        config.warmup, config.runs, config.inner);

    std::cout << std::fixed << std::setprecision(6)
              << "{\"rows\":" << config.rows << ",\"rope_dim\":" << config.rope_dim << ",\"heads\":"
              << config.heads << ",\"head_dim\":" << config.head_dim << ",\"threads\":" << config.threads
              << ",\"unroll\":" << config.unroll
              << ",\"f32\":{\"split_us\":" << f32_timing.baseline_us << ",\"fcmla_us\":"
              << f32_timing.candidate_us << ",\"gain\":"
              << (f32_timing.candidate_us > 0.0 ? f32_timing.baseline_us / f32_timing.candidate_us - 1.0 : 0.0)
              << ",\"paired_gain\":" << f32_timing.paired_gain
              << ",\"max_abs\":" << fcmla_f32_accuracy.max_abs << "},\"bf16\":{\"split_us\":"
              << bf16_timing.baseline_us << ",\"fcmla_us\":" << bf16_timing.candidate_us << ",\"gain\":"
              << (bf16_timing.candidate_us > 0.0 ? bf16_timing.baseline_us / bf16_timing.candidate_us - 1.0 : 0.0)
              << ",\"paired_gain\":"
              << bf16_timing.paired_gain << ",\"mismatches\":"
              << fcmla_bf16_accuracy.bf16_mismatches << ",\"max_abs\":" << fcmla_bf16_accuracy.max_abs
              << ",\"split_mismatches\":" << split_bf16_accuracy.bf16_mismatches
              << ",\"split_interleaved_us\":" << interleaved_control_timing.baseline_us
              << ",\"fcmla_control_us\":" << interleaved_control_timing.candidate_us
              << ",\"fcmla_vs_interleaved_gain\":"
              << (interleaved_control_timing.candidate_us > 0.0
                      ? interleaved_control_timing.baseline_us / interleaved_control_timing.candidate_us - 1.0
                      : 0.0)
              << "},\"qnorm_bf16\":{\"split_us\":" << qnorm_timing.baseline_us << ",\"fcmla_us\":"
              << qnorm_timing.candidate_us << ",\"gain\":"
              << qnorm_timing.baseline_us / qnorm_timing.candidate_us - 1.0 << ",\"paired_gain\":"
              << qnorm_timing.paired_gain << ",\"mismatches\":"
              << qnorm_fcmla_accuracy.bf16_mismatches << ",\"max_abs\":" << qnorm_fcmla_accuracy.max_abs << "}}\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
}
