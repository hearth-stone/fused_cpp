// SPDX-License-Identifier: Apache-2.0

#include <arm_sve.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using KernelFn = void (*)(const float*, float*);

#define DECLARE_KERNEL(rows, variant, scope) \
  extern "C" void sve_fexpa_m##rows##_##variant##_##scope(const float*, float*)
#define DECLARE_SET(rows, variant)     \
  DECLARE_KERNEL(rows, variant, exp);  \
  DECLARE_KERNEL(rows, variant, silu); \
  DECLARE_KERNEL(rows, variant, clamped_silu)
DECLARE_SET(8, poly4);
DECLARE_SET(8, poly5);
DECLARE_SET(8, poly6);
DECLARE_SET(8, fexpa);
DECLARE_SET(12, poly4);
DECLARE_SET(12, poly5);
DECLARE_SET(12, poly6);
DECLARE_SET(12, fexpa);
#undef DECLARE_SET
#undef DECLARE_KERNEL

enum class Scope { kExp, kSilu, kClampedSilu };

struct Variant {
  const char* name;
  KernelFn functions[2][3];
};

constexpr Variant kVariants[] = {
    {"poly4",
     {{sve_fexpa_m8_poly4_exp, sve_fexpa_m8_poly4_silu, sve_fexpa_m8_poly4_clamped_silu},
      {sve_fexpa_m12_poly4_exp, sve_fexpa_m12_poly4_silu, sve_fexpa_m12_poly4_clamped_silu}}},
    {"poly5",
     {{sve_fexpa_m8_poly5_exp, sve_fexpa_m8_poly5_silu, sve_fexpa_m8_poly5_clamped_silu},
      {sve_fexpa_m12_poly5_exp, sve_fexpa_m12_poly5_silu, sve_fexpa_m12_poly5_clamped_silu}}},
    {"poly6",
     {{sve_fexpa_m8_poly6_exp, sve_fexpa_m8_poly6_silu, sve_fexpa_m8_poly6_clamped_silu},
      {sve_fexpa_m12_poly6_exp, sve_fexpa_m12_poly6_silu, sve_fexpa_m12_poly6_clamped_silu}}},
    {"fexpa_poly2",
     {{sve_fexpa_m8_fexpa_exp, sve_fexpa_m8_fexpa_silu, sve_fexpa_m8_fexpa_clamped_silu},
      {sve_fexpa_m12_fexpa_exp, sve_fexpa_m12_fexpa_silu, sve_fexpa_m12_fexpa_clamped_silu}}},
};

struct Accuracy {
  double max_absolute = 0.0;
  double max_relative = 0.0;
  long double squared_error = 0.0;
  uint64_t max_ulp = 0;
  uint64_t bf16_mismatch = 0;
  uint64_t samples = 0;
  bool finite = true;
};

const char* ScopeName(Scope scope) {
  switch (scope) {
    case Scope::kExp:
      return "exp";
    case Scope::kSilu:
      return "silu";
    case Scope::kClampedSilu:
      return "clamped_silu";
  }
  return "unknown";
}

float Clamp(float value, float low, float high) { return std::max(low, std::min(value, high)); }

float Reference(float gate, float up, Scope scope) {
  if (scope == Scope::kClampedSilu) {
    gate = std::min(gate, 10.0f);
    up = Clamp(up, -10.0f, 10.0f);
  }
  const float exponential = std::exp(-gate);
  return scope == Scope::kExp ? exponential : gate * up / (1.0f + exponential);
}

uint16_t ToBf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1U;
  bits += 0x7fffU + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

uint64_t UlpDistancePositive(float lhs, float rhs) {
  uint32_t lhs_bits = 0;
  uint32_t rhs_bits = 0;
  std::memcpy(&lhs_bits, &lhs, sizeof(lhs_bits));
  std::memcpy(&rhs_bits, &rhs, sizeof(rhs_bits));
  return lhs_bits > rhs_bits ? lhs_bits - rhs_bits : rhs_bits - lhs_bits;
}

void UpdateAccuracy(Accuracy* accuracy, float actual, float expected, Scope scope) {
  if (!std::isfinite(actual) || !std::isfinite(expected)) {
    accuracy->finite = false;
    return;
  }
  const double absolute = std::abs(static_cast<double>(actual) - expected);
  const double relative = absolute / std::max(1.0e-6, std::abs(static_cast<double>(expected)));
  accuracy->max_absolute = std::max(accuracy->max_absolute, absolute);
  accuracy->max_relative = std::max(accuracy->max_relative, relative);
  accuracy->squared_error += absolute * absolute;
  if (scope == Scope::kExp) {
    accuracy->max_ulp = std::max(accuracy->max_ulp, UlpDistancePositive(actual, expected));
  }
  accuracy->bf16_mismatch += ToBf16(actual) != ToBf16(expected);
  ++accuracy->samples;
}

int ParsePositive(const char* value, const char* name) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string("invalid ") + name + ": " + value);
  }
  return static_cast<int>(parsed);
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

double Measure(KernelFn function, const float* input, float* output, int threads, int input_stride, int output_stride,
               int inner) {
  double begin = 0.0;
  double end = 0.0;
#pragma omp parallel num_threads(threads) shared(begin, end)
  {
    const int worker = omp_get_thread_num();
    const float* worker_input = input + static_cast<int64_t>(worker) * input_stride;
    float* worker_output = output + static_cast<int64_t>(worker) * output_stride;
#pragma omp barrier
#pragma omp master
    begin = omp_get_wtime();
#pragma omp barrier
    for (int iteration = 0; iteration < inner; ++iteration) {
      function(worker_input, worker_output);
    }
#pragma omp barrier
#pragma omp master
    end = omp_get_wtime();
  }
  return (end - begin) * 1.0e9 / static_cast<double>(inner);
}

Accuracy CheckAccuracy(const Variant& variant, int shape, Scope scope, int lanes, int requested_samples,
                       float bound_override = 0.0f) {
  const int rows = shape == 0 ? 8 : 12;
  const int input_elements = rows * 2 * lanes;
  const int output_elements = rows * lanes;
  std::vector<float> input(input_elements);
  std::vector<float> output(output_elements);
  std::mt19937 generator(20260821 + shape * 31 + static_cast<int>(scope));
  const float bound = bound_override > 0.0f ? bound_override : (scope == Scope::kClampedSilu ? 20.0f : 10.0f);
  std::uniform_real_distribution<float> distribution(-bound, bound);
  Accuracy accuracy;
  while (accuracy.samples < static_cast<uint64_t>(requested_samples)) {
    for (float& value : input) {
      value = distribution(generator);
    }
    variant.functions[shape][static_cast<int>(scope)](input.data(), output.data());
    for (int row = 0; row < rows && accuracy.samples < static_cast<uint64_t>(requested_samples); ++row) {
      const int input_base = row * 2 * lanes;
      for (int lane = 0; lane < lanes && accuracy.samples < static_cast<uint64_t>(requested_samples); ++lane) {
        const float expected = Reference(input[input_base + lane], input[input_base + lanes + lane], scope);
        UpdateAccuracy(&accuracy, output[row * lanes + lane], expected, scope);
      }
    }
  }
  return accuracy;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int warmup = 30;
    int runs = 51;
    int inner = 4000;
    int threads = 1;
    int accuracy_samples = 65536;
    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--warmup" && index + 1 < argc) {
        warmup = ParsePositive(argv[++index], "warmup");
      } else if (argument == "--runs" && index + 1 < argc) {
        runs = ParsePositive(argv[++index], "runs");
      } else if (argument == "--inner" && index + 1 < argc) {
        inner = ParsePositive(argv[++index], "inner");
      } else if (argument == "--threads" && index + 1 < argc) {
        threads = ParsePositive(argv[++index], "threads");
      } else if (argument == "--accuracy-samples" && index + 1 < argc) {
        accuracy_samples = ParsePositive(argv[++index], "accuracy-samples");
      } else {
        std::cerr << "usage: bench_sve_fexpa_exp [--warmup N] [--runs N] [--inner N] [--threads N]"
                     " [--accuracy-samples N]\n";
        return 2;
      }
    }

    constexpr int kExpectedLanes = FUSED_CPP_MOE_SVE_VECTOR_BITS / 32;
    const int lanes = static_cast<int>(svcntw());
    if (lanes != kExpectedLanes) {
      throw std::runtime_error("runtime SVE vector length does not match compiled vector length");
    }
    omp_set_dynamic(0);
    std::cout << "sve_bits=" << FUSED_CPP_MOE_SVE_VECTOR_BITS << " threads=" << threads << " warmup=" << warmup
              << " runs=" << runs << " inner=" << inner << " accuracy_samples=" << accuracy_samples << '\n';

    for (int shape = 0; shape < 2; ++shape) {
      const int rows = shape == 0 ? 8 : 12;
      const int input_stride = rows * 2 * lanes;
      const int output_stride = rows * lanes;
      std::vector<float> input(static_cast<size_t>(threads) * input_stride);
      std::vector<float> output(static_cast<size_t>(threads) * output_stride);
      std::mt19937 generator(20260821 + shape);
      std::uniform_real_distribution<float> distribution(-6.0f, 6.0f);
      for (float& value : input) {
        value = distribution(generator);
      }

      for (int scope_index = 0; scope_index < 3; ++scope_index) {
        const Scope scope = static_cast<Scope>(scope_index);
        for (const Variant& variant : kVariants) {
          const Accuracy accuracy = CheckAccuracy(variant, shape, scope, lanes, accuracy_samples);
          const double rms = std::sqrt(static_cast<double>(accuracy.squared_error / accuracy.samples));
          std::cout << std::scientific << std::setprecision(6) << "accuracy rows=" << rows
                    << " scope=" << ScopeName(scope) << " evaluator=" << variant.name
                    << " max_abs=" << accuracy.max_absolute << " max_rel=" << accuracy.max_relative << " rms=" << rms
                    << " max_ulp=" << accuracy.max_ulp
                    << " bf16_mismatch_pct=" << (100.0 * accuracy.bf16_mismatch / accuracy.samples)
                    << " finite=" << accuracy.finite << '\n';
          if (!accuracy.finite) {
            return 1;
          }
        }

        for (int iteration = 0; iteration < warmup; ++iteration) {
          const Variant& variant = kVariants[iteration % 4];
          Measure(variant.functions[shape][scope_index], input.data(), output.data(), threads, input_stride,
                  output_stride, 1);
        }
        std::vector<std::vector<double>> samples(4);
        for (int sample = 0; sample < runs; ++sample) {
          for (int offset = 0; offset < 4; ++offset) {
            const int variant_index = (sample + offset) % 4;
            samples[variant_index].push_back(Measure(kVariants[variant_index].functions[shape][scope_index],
                                                     input.data(), output.data(), threads, input_stride, output_stride,
                                                     inner));
          }
        }
        const double baseline = Median(samples[1]);
        for (int variant_index = 0; variant_index < 4; ++variant_index) {
          const double median = Median(samples[variant_index]);
          const double values_per_second = static_cast<double>(threads) * rows * lanes / (median * 1.0e-9);
          std::cout << std::fixed << std::setprecision(3) << "performance rows=" << rows
                    << " scope=" << ScopeName(scope) << " evaluator=" << kVariants[variant_index].name
                    << " median_wave_ns=" << median << " gvalues_s=" << values_per_second / 1.0e9
                    << " relative_to_poly5=" << median / baseline
                    << " gain_vs_poly5_pct=" << (baseline / median - 1.0) * 100.0 << '\n';
        }
      }
    }

    for (const Variant& variant : kVariants) {
      const Accuracy accuracy = CheckAccuracy(variant, 1, Scope::kExp, lanes, accuracy_samples, 87.0f);
      std::cout << std::scientific << std::setprecision(6)
                << "wide_accuracy rows=12 scope=exp evaluator=" << variant.name << " max_rel=" << accuracy.max_relative
                << " max_ulp=" << accuracy.max_ulp
                << " bf16_mismatch_pct=" << (100.0 * accuracy.bf16_mismatch / accuracy.samples)
                << " finite=" << accuracy.finite << '\n';
      if (!accuracy.finite) {
        return 1;
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
