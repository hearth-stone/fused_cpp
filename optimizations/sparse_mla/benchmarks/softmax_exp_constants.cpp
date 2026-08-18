// SPDX-License-Identifier: Apache-2.0
//
// Lab-only AArch64 benchmark for the Sparse MLA online-softmax exp kernel.
// The pressure wrappers deliberately keep the same 16 scalable accumulators as
// the production 8x2VL BFMMLA kernels live across the inlined NEON softmax body.

#include <arm_neon.h>
#include <arm_sve.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace sparse_mla::softmax_exp_bench {
namespace {

constexpr float kLn2 = 0.6931471805599453f;
constexpr float kInvLn2 = 1.4426950408889634f;
constexpr float kExpHi = 87.0f;
constexpr float kExpLo = -87.0f;

enum class Variant {
  kHornerBroadcast,
  kEstrinPackedLane,
};

struct PackedExpConstants {
  float32x4_t range;
  float32x4_t odd;
  float32x4_t even;
};

struct ErrorMetrics {
  double max_abs = 0.0;
  double max_rel = 0.0;
  uint32_t max_ulp = 0;
  double rms = 0.0;
  double sum_abs = 0.0;
  double bf16_max_abs = 0.0;
  int64_t bf16_mismatches = 0;
  int64_t special_mismatches = 0;
  int64_t special_count = 0;
  int64_t count = 0;
};

struct BenchmarkResult {
  double median_ns_per_element = 0.0;
  double min_ns_per_element = 0.0;
  double max_ns_per_element = 0.0;
  double elements_per_second = 0.0;
  float checksum = 0.0f;
};

template <typename To, typename From>
To BitCast(const From& source) {
  static_assert(sizeof(To) == sizeof(From));
  static_assert(std::is_trivially_copyable_v<To>);
  static_assert(std::is_trivially_copyable_v<From>);
  To destination;
  std::memcpy(&destination, &source, sizeof(destination));
  return destination;
}

[[gnu::always_inline]] inline float32x4_t ExpHornerBroadcast(float32x4_t x) {
  const float32x4_t ln2 = vdupq_n_f32(kLn2);
  const float32x4_t inv_ln2 = vdupq_n_f32(kInvLn2);
  x = vminq_f32(x, vdupq_n_f32(kExpHi));
  x = vmaxq_f32(x, vdupq_n_f32(kExpLo));

  const float32x4_t fn = vrndnq_f32(vmulq_f32(x, inv_ln2));
  const int32x4_t n = vcvtq_s32_f32(fn);
  const float32x4_t r = vfmsq_f32(x, fn, ln2);

  float32x4_t poly = vdupq_n_f32(1.0f / 120.0f);
  poly = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), poly, r);
  poly = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), poly, r);
  poly = vfmaq_f32(vdupq_n_f32(0.5f), poly, r);
  poly = vfmaq_f32(vdupq_n_f32(1.0f), poly, r);
  poly = vfmaq_f32(vdupq_n_f32(1.0f), poly, r);

  const int32x4_t exponent = vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(127)), 23);
  return vmulq_f32(poly, vreinterpretq_f32_s32(exponent));
}

[[gnu::always_inline]] inline PackedExpConstants MakePackedExpConstants() {
  PackedExpConstants constants = {{kLn2, kInvLn2, kExpLo, kExpHi},
                                  {1.0f, 1.0f / 6.0f, 1.0f / 120.0f, 0.0f},
                                  {1.0f, 0.5f, 1.0f / 24.0f, 0.0f}};
  // Keep GCC from scalarizing known lanes into separate broadcast constants.
  // The empty constraint emits no instruction and executes once per kernel.
  asm volatile("" : "+w"(constants.range), "+w"(constants.odd), "+w"(constants.even));
  return constants;
}

[[gnu::always_inline]] inline float32x4_t ExpEstrinPackedLane(float32x4_t x,
                                                              const PackedExpConstants& constants) {
  // Packed constants are consumed through lane FMLA/FMLS. The even
  // coefficients are Estrin addends and need one reusable DUP result each.
  x = vminq_f32(x, vdupq_laneq_f32(constants.range, 3));
  x = vmaxq_f32(x, vdupq_laneq_f32(constants.range, 2));
  const float32x4_t fn = vrndnq_f32(vmulq_laneq_f32(x, constants.range, 1));
  const int32x4_t n = vcvtq_s32_f32(fn);
  const float32x4_t r = vfmsq_laneq_f32(x, fn, constants.range, 0);
  const float32x4_t r2 = vmulq_f32(r, r);
  const float32x4_t r4 = vmulq_f32(r2, r2);

  float32x4_t pair01 = vdupq_laneq_f32(constants.even, 0);
  float32x4_t pair23 = vdupq_laneq_f32(constants.even, 1);
  float32x4_t pair45 = vdupq_laneq_f32(constants.even, 2);
  pair01 = vfmaq_laneq_f32(pair01, r, constants.odd, 0);
  pair23 = vfmaq_laneq_f32(pair23, r, constants.odd, 1);
  pair45 = vfmaq_laneq_f32(pair45, r, constants.odd, 2);
  float32x4_t poly = vfmaq_f32(pair01, r2, pair23);
  poly = vfmaq_f32(poly, r4, pair45);

  const int32x4_t exponent = vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(127)), 23);
  return vmulq_f32(poly, vreinterpretq_f32_s32(exponent));
}

template <Variant kVariant>
[[gnu::always_inline]] inline float32x4_t Exp(float32x4_t x, const PackedExpConstants& constants) {
  if constexpr (kVariant == Variant::kHornerBroadcast) {
    return ExpHornerBroadcast(x);
  }
  return ExpEstrinPackedLane(x, constants);
}

template <Variant kVariant, int kUnroll>
[[gnu::always_inline]] inline float RunSoftmaxKernel(const float* src, uint16_t* dst, int64_t length,
                                                     float new_max) {
  static_assert(kUnroll == 1 || kUnroll == 2 || kUnroll == 4 || kUnroll == 8);
  constexpr int64_t kVectorWidth = 4;
  constexpr int64_t kStep = kVectorWidth * kUnroll;
  std::array<float32x4_t, kUnroll> sums;
#pragma GCC unroll 8
  for (int lane = 0; lane < kUnroll; ++lane) {
    sums[lane] = vdupq_n_f32(0.0f);
  }

  const float32x4_t max_vector = vdupq_n_f32(new_max);
  PackedExpConstants constants{};
  if constexpr (kVariant == Variant::kEstrinPackedLane) {
    constants = MakePackedExpConstants();
  }
  int64_t offset = 0;
  for (; offset + kStep <= length; offset += kStep) {
    std::array<float32x4_t, kUnroll> values;
#pragma GCC unroll 8
    for (int lane = 0; lane < kUnroll; ++lane) {
      const int64_t vector_offset = offset + lane * kVectorWidth;
      const float32x4_t score = vld1q_f32(src + vector_offset);
      values[lane] = Exp<kVariant>(vsubq_f32(score, max_vector), constants);
    }
#pragma GCC unroll 8
    for (int lane = 0; lane < kUnroll; ++lane) {
      const int64_t vector_offset = offset + lane * kVectorWidth;
      const bfloat16x4_t bf16 = vcvt_bf16_f32(values[lane]);
      vst1_u16(dst + vector_offset, vreinterpret_u16_bf16(bf16));
      sums[lane] = vaddq_f32(sums[lane], values[lane]);
    }
  }

  float sum = 0.0f;
#pragma GCC unroll 8
  for (int lane = 0; lane < kUnroll; ++lane) {
    sum += vaddvq_f32(sums[lane]);
  }
  for (; offset < length; ++offset) {
    const float value = std::exp(src[offset] - new_max);
    uint32_t bits = BitCast<uint32_t>(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    dst[offset] = static_cast<uint16_t>(bits >> 16);
    sum += value;
  }
  return sum;
}

using PlainKernel = float (*)(const float*, uint16_t*, int64_t, float);

extern "C" [[gnu::noinline]] float HornerU1(const float* src, uint16_t* dst, int64_t length, float new_max) {
  return RunSoftmaxKernel<Variant::kHornerBroadcast, 1>(src, dst, length, new_max);
}

extern "C" [[gnu::noinline]] float HornerU2(const float* src, uint16_t* dst, int64_t length, float new_max) {
  return RunSoftmaxKernel<Variant::kHornerBroadcast, 2>(src, dst, length, new_max);
}

extern "C" [[gnu::noinline]] float HornerU4(const float* src, uint16_t* dst, int64_t length, float new_max) {
  return RunSoftmaxKernel<Variant::kHornerBroadcast, 4>(src, dst, length, new_max);
}

extern "C" [[gnu::noinline]] float HornerU8(const float* src, uint16_t* dst, int64_t length, float new_max) {
  return RunSoftmaxKernel<Variant::kHornerBroadcast, 8>(src, dst, length, new_max);
}

extern "C" [[gnu::noinline]] float EstrinU1(const float* src, uint16_t* dst, int64_t length, float new_max) {
  return RunSoftmaxKernel<Variant::kEstrinPackedLane, 1>(src, dst, length, new_max);
}

extern "C" [[gnu::noinline]] float EstrinU2(const float* src, uint16_t* dst, int64_t length, float new_max) {
  return RunSoftmaxKernel<Variant::kEstrinPackedLane, 2>(src, dst, length, new_max);
}

extern "C" [[gnu::noinline]] float EstrinU4(const float* src, uint16_t* dst, int64_t length, float new_max) {
  return RunSoftmaxKernel<Variant::kEstrinPackedLane, 4>(src, dst, length, new_max);
}

extern "C" [[gnu::noinline]] float EstrinU8(const float* src, uint16_t* dst, int64_t length, float new_max) {
  return RunSoftmaxKernel<Variant::kEstrinPackedLane, 8>(src, dst, length, new_max);
}

[[gnu::always_inline]] inline svbfloat16_t LoadBf16(const uint16_t* ptr) {
  return svreinterpret_bf16_u16(svld1_u16(svptrue_b16(), ptr));
}

template <Variant kVariant, int kUnroll>
[[gnu::always_inline]] inline float RunPressureKernel(const float* src, uint16_t* dst, int64_t length, float new_max,
                                                      const uint16_t* packed_a, const uint16_t* packed_b) {
  const svbool_t pg32 = svptrue_b32();
  const svbool_t pg16 = svptrue_b16();
  const int64_t lanes_h = static_cast<int64_t>(svcnth());
  const svbfloat16_t a0 = svld1rq_bf16(pg16, reinterpret_cast<const __bf16*>(packed_a + 0));
  const svbfloat16_t a1 = svld1rq_bf16(pg16, reinterpret_cast<const __bf16*>(packed_a + 8));
  const svbfloat16_t a2 = svld1rq_bf16(pg16, reinterpret_cast<const __bf16*>(packed_a + 16));
  const svbfloat16_t a3 = svld1rq_bf16(pg16, reinterpret_cast<const __bf16*>(packed_a + 24));
  const svbfloat16_t b0 = LoadBf16(packed_b + 0 * lanes_h);
  const svbfloat16_t b1 = LoadBf16(packed_b + 1 * lanes_h);
  const svbfloat16_t b2 = LoadBf16(packed_b + 2 * lanes_h);
  const svbfloat16_t b3 = LoadBf16(packed_b + 3 * lanes_h);

  const svfloat32_t zero = svdup_f32(0.0f);
  svfloat32_t c00 = svbfmmla_f32(zero, a0, b0);
  svfloat32_t c01 = svbfmmla_f32(zero, a0, b1);
  svfloat32_t c02 = svbfmmla_f32(zero, a0, b2);
  svfloat32_t c03 = svbfmmla_f32(zero, a0, b3);
  svfloat32_t c10 = svbfmmla_f32(zero, a1, b0);
  svfloat32_t c11 = svbfmmla_f32(zero, a1, b1);
  svfloat32_t c12 = svbfmmla_f32(zero, a1, b2);
  svfloat32_t c13 = svbfmmla_f32(zero, a1, b3);
  svfloat32_t c20 = svbfmmla_f32(zero, a2, b0);
  svfloat32_t c21 = svbfmmla_f32(zero, a2, b1);
  svfloat32_t c22 = svbfmmla_f32(zero, a2, b2);
  svfloat32_t c23 = svbfmmla_f32(zero, a2, b3);
  svfloat32_t c30 = svbfmmla_f32(zero, a3, b0);
  svfloat32_t c31 = svbfmmla_f32(zero, a3, b1);
  svfloat32_t c32 = svbfmmla_f32(zero, a3, b2);
  svfloat32_t c33 = svbfmmla_f32(zero, a3, b3);

  // The first barrier prevents sinking BFMMLA into the softmax region. The
  // second extends all 16 live ranges through that region.
  asm volatile(""
               :
               : "w"(c00), "w"(c01), "w"(c02), "w"(c03), "w"(c10), "w"(c11), "w"(c12), "w"(c13),
                 "w"(c20), "w"(c21), "w"(c22), "w"(c23), "w"(c30), "w"(c31), "w"(c32), "w"(c33));
  const float sum = RunSoftmaxKernel<kVariant, kUnroll>(src, dst, length, new_max);
  asm volatile(""
               :
               : "w"(c00), "w"(c01), "w"(c02), "w"(c03), "w"(c10), "w"(c11), "w"(c12), "w"(c13),
                 "w"(c20), "w"(c21), "w"(c22), "w"(c23), "w"(c30), "w"(c31), "w"(c32), "w"(c33));
  const float after = svaddv_f32(pg32, c00) + svaddv_f32(pg32, c01) + svaddv_f32(pg32, c02) +
                      svaddv_f32(pg32, c03) + svaddv_f32(pg32, c10) + svaddv_f32(pg32, c11) +
                      svaddv_f32(pg32, c12) + svaddv_f32(pg32, c13) + svaddv_f32(pg32, c20) +
                      svaddv_f32(pg32, c21) + svaddv_f32(pg32, c22) + svaddv_f32(pg32, c23) +
                      svaddv_f32(pg32, c30) + svaddv_f32(pg32, c31) + svaddv_f32(pg32, c32) +
                      svaddv_f32(pg32, c33);
  return sum + after * 0x1p-60f;
}

using PressureKernel = float (*)(const float*, uint16_t*, int64_t, float, const uint16_t*, const uint16_t*);

extern "C" [[gnu::noinline]] float HornerPressureU4(const float* src, uint16_t* dst, int64_t length, float new_max,
                                                     const uint16_t* packed_a, const uint16_t* packed_b) {
  return RunPressureKernel<Variant::kHornerBroadcast, 4>(src, dst, length, new_max, packed_a, packed_b);
}

extern "C" [[gnu::noinline]] float HornerPressureU8(const float* src, uint16_t* dst, int64_t length, float new_max,
                                                     const uint16_t* packed_a, const uint16_t* packed_b) {
  return RunPressureKernel<Variant::kHornerBroadcast, 8>(src, dst, length, new_max, packed_a, packed_b);
}

extern "C" [[gnu::noinline]] float EstrinPressureU4(const float* src, uint16_t* dst, int64_t length, float new_max,
                                                     const uint16_t* packed_a, const uint16_t* packed_b) {
  return RunPressureKernel<Variant::kEstrinPackedLane, 4>(src, dst, length, new_max, packed_a, packed_b);
}

extern "C" [[gnu::noinline]] float EstrinPressureU8(const float* src, uint16_t* dst, int64_t length, float new_max,
                                                     const uint16_t* packed_a, const uint16_t* packed_b) {
  return RunPressureKernel<Variant::kEstrinPackedLane, 8>(src, dst, length, new_max, packed_a, packed_b);
}

float Bf16ToFloat(uint16_t value) {
  return BitCast<float>(static_cast<uint32_t>(value) << 16);
}

uint16_t FloatToBf16(float value) {
  uint32_t bits = BitCast<uint32_t>(value);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

uint32_t UlpDistance(float actual, float expected) {
  if (!std::isfinite(actual) || !std::isfinite(expected) || actual < 0.0f || expected < 0.0f) {
    return actual == expected ? 0u : std::numeric_limits<uint32_t>::max();
  }
  const uint32_t actual_bits = BitCast<uint32_t>(actual);
  const uint32_t expected_bits = BitCast<uint32_t>(expected);
  return actual_bits > expected_bits ? actual_bits - expected_bits : expected_bits - actual_bits;
}

template <Variant kVariant>
ErrorMetrics CheckAccuracy(const std::vector<float>& input) {
  std::vector<float> output(input.size());
  PackedExpConstants constants{};
  if constexpr (kVariant == Variant::kEstrinPackedLane) {
    constants = MakePackedExpConstants();
  }
  for (size_t offset = 0; offset < input.size(); offset += 4) {
    const float32x4_t x = vld1q_f32(input.data() + offset);
    vst1q_f32(output.data() + offset, Exp<kVariant>(x, constants));
  }

  ErrorMetrics metrics;
  double squared_error = 0.0;
  for (size_t index = 0; index < input.size(); ++index) {
    const float reference = std::exp(std::clamp(input[index], kExpLo, kExpHi));
    if (!std::isfinite(input[index]) || input[index] < kExpLo || input[index] > 0.0f) {
      const bool matches = std::isnan(input[index]) ? std::isnan(output[index])
                                                    : std::isfinite(output[index]) && output[index] > 0.0f;
      metrics.special_mismatches += !matches;
      ++metrics.special_count;
      continue;
    }
    const double absolute = std::abs(static_cast<double>(output[index]) - reference);
    const double relative = reference > 0.0f ? absolute / reference : absolute;
    metrics.max_abs = std::max(metrics.max_abs, absolute);
    metrics.max_rel = std::max(metrics.max_rel, relative);
    metrics.max_ulp = std::max(metrics.max_ulp, UlpDistance(output[index], reference));
    metrics.sum_abs += absolute;
    squared_error += absolute * absolute;
    const uint16_t actual_bf16 = FloatToBf16(output[index]);
    const uint16_t reference_bf16 = FloatToBf16(reference);
    metrics.bf16_mismatches += actual_bf16 != reference_bf16;
    metrics.bf16_max_abs = std::max(metrics.bf16_max_abs,
                                    std::abs(static_cast<double>(Bf16ToFloat(actual_bf16)) -
                                             static_cast<double>(Bf16ToFloat(reference_bf16))));
    ++metrics.count;
  }
  metrics.rms = std::sqrt(squared_error / static_cast<double>(metrics.count));
  return metrics;
}

BenchmarkResult BenchmarkPlain(PlainKernel kernel, const std::vector<float>& input, std::vector<uint16_t>& output,
                               int warmup, int iterations, int samples) {
  float checksum = 0.0f;
  for (int iteration = 0; iteration < warmup; ++iteration) {
    checksum += kernel(input.data(), output.data(), static_cast<int64_t>(input.size()), 0.0f);
  }
  std::vector<double> timings;
  timings.reserve(static_cast<size_t>(samples));
  for (int sample = 0; sample < samples; ++sample) {
    const auto begin = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
      checksum += kernel(input.data(), output.data(), static_cast<int64_t>(input.size()), 0.0f);
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ns = std::chrono::duration<double, std::nano>(end - begin).count();
    timings.push_back(elapsed_ns / (static_cast<double>(iterations) * static_cast<double>(input.size())));
  }
  std::sort(timings.begin(), timings.end());
  const double median = timings[timings.size() / 2];
  return {median, timings.front(), timings.back(), 1.0e9 / median, checksum};
}

BenchmarkResult BenchmarkPressure(PressureKernel kernel, const std::vector<float>& input, std::vector<uint16_t>& output,
                                  const std::vector<uint16_t>& packed_a, const std::vector<uint16_t>& packed_b,
                                  int warmup, int iterations, int samples) {
  float checksum = 0.0f;
  for (int iteration = 0; iteration < warmup; ++iteration) {
    checksum += kernel(input.data(), output.data(), static_cast<int64_t>(input.size()), 0.0f, packed_a.data(),
                       packed_b.data());
  }
  std::vector<double> timings;
  timings.reserve(static_cast<size_t>(samples));
  for (int sample = 0; sample < samples; ++sample) {
    const auto begin = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
      checksum += kernel(input.data(), output.data(), static_cast<int64_t>(input.size()), 0.0f, packed_a.data(),
                         packed_b.data());
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ns = std::chrono::duration<double, std::nano>(end - begin).count();
    timings.push_back(elapsed_ns / (static_cast<double>(iterations) * static_cast<double>(input.size())));
  }
  std::sort(timings.begin(), timings.end());
  const double median = timings[timings.size() / 2];
  return {median, timings.front(), timings.back(), 1.0e9 / median, checksum};
}

void PrintAccuracy(const char* name, const ErrorMetrics& metrics) {
  std::cout << "accuracy,variant=" << name << ",count=" << metrics.count << ",max_abs=" << metrics.max_abs
            << ",max_rel=" << metrics.max_rel << ",max_ulp=" << metrics.max_ulp << ",rms=" << metrics.rms
            << ",mean_abs=" << metrics.sum_abs / static_cast<double>(metrics.count)
            << ",bf16_max_abs=" << metrics.bf16_max_abs << ",bf16_mismatches=" << metrics.bf16_mismatches
            << ",special_count=" << metrics.special_count << ",special_mismatches=" << metrics.special_mismatches
            << '\n';
}

void PrintBenchmark(const char* mode, const char* name, int unroll, const BenchmarkResult& result) {
  std::cout << "benchmark,mode=" << mode << ",variant=" << name << ",unroll=" << unroll
            << ",median_ns_per_element=" << result.median_ns_per_element
            << ",min_ns_per_element=" << result.min_ns_per_element
            << ",max_ns_per_element=" << result.max_ns_per_element
            << ",gelements_per_second=" << result.elements_per_second / 1.0e9 << ",checksum=" << result.checksum
            << '\n';
}

}  // namespace
}  // namespace sparse_mla::softmax_exp_bench

int main(int argc, char** argv) {
  using namespace sparse_mla::softmax_exp_bench;
  int64_t length = 2048;
  int warmup = 200;
  int iterations = 2000;
  int samples = 21;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    const auto parse = [&](const char* prefix, auto& value) {
      const std::string key(prefix);
      if (argument.rfind(key, 0) == 0) {
        value = static_cast<std::decay_t<decltype(value)>>(std::stoll(argument.substr(key.size())));
        return true;
      }
      return false;
    };
    if (!parse("--length=", length) && !parse("--warmup=", warmup) && !parse("--iterations=", iterations) &&
        !parse("--samples=", samples)) {
      throw std::invalid_argument("unknown argument: " + argument);
    }
  }
  if (length <= 0 || length % 32 != 0 || warmup < 0 || iterations <= 0 || samples <= 0 || samples % 2 == 0) {
    throw std::invalid_argument("length must be positive and divisible by 32; samples must be a positive odd value");
  }

  std::mt19937 generator(20260816);
  std::uniform_real_distribution<float> distribution(kExpLo, 0.0f);
  std::vector<float> accuracy_input(1u << 20);
  for (float& value : accuracy_input) {
    value = distribution(generator);
  }
  const std::array<float, 20> boundary = {kExpLo,
                                          std::nextafter(kExpLo, -std::numeric_limits<float>::infinity()),
                                          std::nextafter(kExpLo, 0.0f),
                                          -80.0f,
                                          -32.0f,
                                          -20.0f,
                                          -10.0f,
                                          -4.0f,
                                          -1.0f,
                                          -kLn2,
                                          -0.5f,
                                          -0.25f,
                                          std::nextafter(0.0f, -1.0f),
                                          -0.0f,
                                          0.0f,
                                          std::nextafter(0.0f, 1.0f),
                                          kExpHi,
                                          std::numeric_limits<float>::infinity(),
                                          -std::numeric_limits<float>::infinity(),
                                          std::numeric_limits<float>::quiet_NaN()};
  std::copy(boundary.begin(), boundary.end(), accuracy_input.begin());

  std::cout << std::setprecision(9) << "config,length=" << length << ",warmup=" << warmup
            << ",iterations=" << iterations << ",samples=" << samples << ",sve_bits=" << svcntb() * 8 << '\n';
  PrintAccuracy("horner_broadcast", CheckAccuracy<Variant::kHornerBroadcast>(accuracy_input));
  PrintAccuracy("estrin_packed_lane", CheckAccuracy<Variant::kEstrinPackedLane>(accuracy_input));

  std::vector<float> input(static_cast<size_t>(length));
  std::uniform_real_distribution<float> softmax_distribution(-16.0f, 0.0f);
  for (float& value : input) {
    value = softmax_distribution(generator);
  }
  std::vector<uint16_t> output(static_cast<size_t>(length));

  const std::array<std::tuple<const char*, int, PlainKernel>, 8> plain_kernels = {
      std::tuple{"horner_broadcast", 1, HornerU1}, std::tuple{"horner_broadcast", 2, HornerU2},
      std::tuple{"horner_broadcast", 4, HornerU4}, std::tuple{"horner_broadcast", 8, HornerU8},
      std::tuple{"estrin_packed_lane", 1, EstrinU1}, std::tuple{"estrin_packed_lane", 2, EstrinU2},
      std::tuple{"estrin_packed_lane", 4, EstrinU4}, std::tuple{"estrin_packed_lane", 8, EstrinU8}};
  for (const auto& [name, unroll, kernel] : plain_kernels) {
    PrintBenchmark("plain", name, unroll, BenchmarkPlain(kernel, input, output, warmup, iterations, samples));
  }

  std::vector<uint16_t> packed_a(32);
  std::vector<uint16_t> packed_b(static_cast<size_t>(4 * svcnth()));
  std::uniform_int_distribution<uint16_t> bf16_bits(0x3d00u, 0x3f00u);
  for (uint16_t& value : packed_a) {
    value = bf16_bits(generator);
  }
  for (uint16_t& value : packed_b) {
    value = bf16_bits(generator);
  }
  const std::array<std::tuple<const char*, int, PressureKernel>, 4> pressure_kernels = {
      std::tuple{"horner_broadcast", 4, HornerPressureU4}, std::tuple{"horner_broadcast", 8, HornerPressureU8},
      std::tuple{"estrin_packed_lane", 4, EstrinPressureU4},
      std::tuple{"estrin_packed_lane", 8, EstrinPressureU8}};
  for (const auto& [name, unroll, kernel] : pressure_kernels) {
    PrintBenchmark("8x2vl_pressure", name, unroll,
                   BenchmarkPressure(kernel, input, output, packed_a, packed_b, warmup, iterations, samples));
  }
  return 0;
}
