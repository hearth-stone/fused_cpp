// SPDX-License-Identifier: Apache-2.0

#include <arm_sve.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct alignas(64) Constants {
  float coefficients0[8] = {1.0f, 1.0f, 0.5f, 0.16666666f, 1.0f, 1.0f, 0.5f, 0.16666666f};
  float coefficients1[8] = {0.04166666f, 0.00833333f, 1.4426950408889634f, 0.6931471805599453f,
                            0.04166666f, 0.00833333f, 1.4426950408889634f, 0.6931471805599453f};
  float swiglu_limit = 10.0f;
};

using KernelFn = void (*)(const float*, float*, const Constants*);

#define DECLARE_KERNEL(rows, evaluator, scope) \
  extern "C" void sve_exp_m##rows##_##evaluator##_##scope(const float*, float*, const Constants*)
#define DECLARE_SET(rows, evaluator)     \
  DECLARE_KERNEL(rows, evaluator, exp);  \
  DECLARE_KERNEL(rows, evaluator, silu); \
  DECLARE_KERNEL(rows, evaluator, clamped_silu)
DECLARE_SET(8, horner);
DECLARE_SET(8, even_odd);
DECLARE_SET(8, estrin);
DECLARE_SET(12, horner);
DECLARE_SET(12, even_odd);
DECLARE_SET(12, estrin);
#undef DECLARE_SET
#undef DECLARE_KERNEL

enum class Scope { kExp, kSilu, kClampedSilu };

struct Variant {
  const char* name;
  KernelFn functions[2][3];
};

constexpr Variant kVariants[] = {
    {"horner",
     {{sve_exp_m8_horner_exp, sve_exp_m8_horner_silu, sve_exp_m8_horner_clamped_silu},
      {sve_exp_m12_horner_exp, sve_exp_m12_horner_silu, sve_exp_m12_horner_clamped_silu}}},
    {"even_odd",
     {{sve_exp_m8_even_odd_exp, sve_exp_m8_even_odd_silu, sve_exp_m8_even_odd_clamped_silu},
      {sve_exp_m12_even_odd_exp, sve_exp_m12_even_odd_silu, sve_exp_m12_even_odd_clamped_silu}}},
    {"estrin",
     {{sve_exp_m8_estrin_exp, sve_exp_m8_estrin_silu, sve_exp_m8_estrin_clamped_silu},
      {sve_exp_m12_estrin_exp, sve_exp_m12_estrin_silu, sve_exp_m12_estrin_clamped_silu}}},
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

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

float Clamp(float value, float low, float high) { return std::max(low, std::min(value, high)); }

float Reference(float gate, float up, Scope scope) {
  if (scope == Scope::kClampedSilu) {
    gate = std::min(gate, 10.0f);
    up = Clamp(up, -10.0f, 10.0f);
  }
  const float x = -gate;
  const float n = std::nearbyint(x * 1.4426950408889634f);
  const float r = std::fma(-n, 0.6931471805599453f, x);
  float polynomial = std::fma(r, 0.00833333f, 0.04166666f);
  polynomial = std::fma(r, polynomial, 0.16666666f);
  polynomial = std::fma(r, polynomial, 0.5f);
  polynomial = std::fma(r, polynomial, 1.0f);
  polynomial = std::fma(r, polynomial, 1.0f);
  const float exponential = std::ldexp(polynomial, static_cast<int>(n));
  return scope == Scope::kExp ? exponential : gate * up / (1.0f + exponential);
}

int ParseInt(const char* value, const char* name) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0) {
    throw std::invalid_argument(std::string("invalid ") + name + ": " + value);
  }
  return static_cast<int>(parsed);
}

double Measure(KernelFn function, const float* input, float* output, const Constants* constants, int threads,
               int input_stride, int output_stride, int inner) {
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
      function(worker_input, worker_output, constants);
    }
#pragma omp barrier
#pragma omp master
    end = omp_get_wtime();
  }
  return (end - begin) * 1.0e9 / static_cast<double>(inner);
}

}  // namespace

int main(int argc, char** argv) {
  int warmup = 1000;
  int runs = 31;
  int inner = 2000;
  int threads = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--warmup" && i + 1 < argc) {
      warmup = ParseInt(argv[++i], "warmup");
    } else if (argument == "--runs" && i + 1 < argc) {
      runs = ParseInt(argv[++i], "runs");
    } else if (argument == "--inner" && i + 1 < argc) {
      inner = ParseInt(argv[++i], "inner");
    } else if (argument == "--threads" && i + 1 < argc) {
      threads = ParseInt(argv[++i], "threads");
    } else {
      std::cerr << "usage: bench_sve_exp_evaluators [--warmup N] [--runs N] [--inner N] [--threads N]\n";
      return 2;
    }
  }

  constexpr int kExpectedLanes = FUSED_CPP_MOE_SVE_VECTOR_BITS / 32;
  const int lanes = static_cast<int>(svcntw());
  if (lanes != kExpectedLanes) {
    std::cerr << "runtime SVE lanes=" << lanes << " do not match compiled lanes=" << kExpectedLanes << '\n';
    return 2;
  }
  omp_set_dynamic(0);
  const Constants constants;
  std::mt19937 generator(20260817);
  std::uniform_real_distribution<float> distribution(-6.0f, 6.0f);
  std::cout << "sve_bits=" << FUSED_CPP_MOE_SVE_VECTOR_BITS << " threads=" << threads << " warmup=" << warmup
            << " runs=" << runs << " inner=" << inner << '\n';

  for (int shape = 0; shape < 2; ++shape) {
    const int rows = shape == 0 ? 8 : 12;
    const int input_stride = rows * 2 * lanes;
    const int output_stride = rows * lanes;
    std::vector<float> input(static_cast<size_t>(threads) * input_stride);
    std::vector<float> output(static_cast<size_t>(threads) * output_stride);
    for (float& value : input) {
      value = distribution(generator);
    }
    for (int scope_index = 0; scope_index < 3; ++scope_index) {
      const Scope scope = static_cast<Scope>(scope_index);
      for (const Variant& variant : kVariants) {
        variant.functions[shape][scope_index](input.data(), output.data(), &constants);
        double max_relative_error = 0.0;
        int max_row = 0;
        int max_lane = 0;
        for (int row = 0; row < rows; ++row) {
          const int input_base = row * 2 * lanes;
          for (int lane = 0; lane < lanes; ++lane) {
            const float expected = Reference(input[input_base + lane], input[input_base + lanes + lane], scope);
            const float actual = output[row * lanes + lane];
            const double absolute = std::abs(static_cast<double>(actual) - expected);
            const double relative = absolute / std::max(1.0e-7, std::abs(static_cast<double>(expected)));
            if (relative > max_relative_error) {
              max_relative_error = relative;
              max_row = row;
              max_lane = lane;
            }
          }
        }
        if (!std::isfinite(max_relative_error) || max_relative_error > 2.0e-5) {
          std::cerr << "correctness failure rows=" << rows << " scope=" << ScopeName(scope)
                    << " evaluator=" << variant.name << " max_rel=" << max_relative_error << " row=" << max_row
                    << " lane=" << max_lane << " actual=" << output[max_row * lanes + max_lane] << " expected="
                    << Reference(input[max_row * 2 * lanes + max_lane], input[max_row * 2 * lanes + lanes + max_lane],
                                 scope)
                    << '\n';
          return 1;
        }
      }

      for (int iteration = 0; iteration < warmup; ++iteration) {
        const Variant& variant = kVariants[iteration % 3];
        Measure(variant.functions[shape][scope_index], input.data(), output.data(), &constants, threads, input_stride,
                output_stride, 1);
      }
      std::vector<std::vector<double>> samples(3);
      for (int sample = 0; sample < runs; ++sample) {
        for (int offset = 0; offset < 3; ++offset) {
          const int variant_index = (sample + offset) % 3;
          samples[variant_index].push_back(Measure(kVariants[variant_index].functions[shape][scope_index], input.data(),
                                                   output.data(), &constants, threads, input_stride, output_stride,
                                                   inner));
        }
      }
      const double baseline = Median(samples[0]);
      for (int variant_index = 0; variant_index < 3; ++variant_index) {
        const double median = Median(samples[variant_index]);
        const double values_per_second = static_cast<double>(threads) * rows * lanes / (median * 1.0e-9);
        std::cout << std::fixed << std::setprecision(3) << "rows=" << rows << " live_z=" << rows * 2
                  << " scope=" << ScopeName(scope) << " evaluator=" << kVariants[variant_index].name
                  << " median_wave_ns=" << median << " gvalues_s=" << values_per_second / 1.0e9
                  << " relative=" << median / baseline << " gain_pct=" << (baseline / median - 1.0) * 100.0 << '\n';
      }
    }
  }
  return 0;
}
