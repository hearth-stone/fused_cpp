// SPDX-License-Identifier: Apache-2.0
//
// Direct, same-process comparison between the standalone SVE JIT GEMM and
// the corresponding low-level kernels in refs/i8gemm/lib/bf16gemm_sve.S.

#include "jit_kernels.h"
#include "vector_length.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
void bf16gemm_k_nld1_f(const uint16_t*, const uint16_t*, float*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_nld2_f(const uint16_t*, const uint16_t*, float*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_nld4_f(const uint16_t*, const uint16_t*, float*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_nld_f(const uint16_t*, const uint16_t*, float*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_nld_f_m12(const uint16_t*, const uint16_t*, float*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_nld_f_m8_ilv(const uint16_t*, const uint16_t*, float*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_nld_f_m12_ilv(const uint16_t*, const uint16_t*, float*, uint16_t*, const gemm_params_t*);
}

namespace {

using Clock = std::chrono::steady_clock;
using I8mmKernelFn = void (*)(const uint16_t*, const uint16_t*, float*, uint16_t*, const gemm_params_t*);

struct alignas(8) ExtendedParams {
  gemm_params_t gemm{};
  int32_t kc = 0;
  int32_t packed_n = 0;
  int32_t n_begin = 0;
  int32_t mode = 0;
  float* partial_c = nullptr;
};

static_assert(sizeof(gemm_params_t) == 24);
static_assert(offsetof(ExtendedParams, n_begin) == 32);

struct Options {
  std::vector<int> rows{1, 2, 4, 8, 12};
  int k = 4096;
  int n = 1024;
  int experts = 1;
  int warmup = 32;
  int runs = 201;
  int inner = 1;
  bool include_ilv = false;
  bool include_column_pipeline = false;
};

uint16_t to_bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1U;
  bits += 0x7fffU + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

std::vector<int> parse_rows(const std::string& text) {
  std::vector<int> rows;
  size_t begin = 0;
  while (begin < text.size()) {
    const size_t end = text.find(',', begin);
    rows.push_back(std::stoi(text.substr(begin, end - begin)));
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  return rows;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    auto value = [&]() -> std::string {
      if (++i >= argc) {
        throw std::invalid_argument("missing value for " + argument);
      }
      return argv[i];
    };
    if (argument == "--rows") {
      options.rows = parse_rows(value());
    } else if (argument == "--k") {
      options.k = std::stoi(value());
    } else if (argument == "--n") {
      options.n = std::stoi(value());
    } else if (argument == "--experts") {
      options.experts = std::stoi(value());
    } else if (argument == "--warmup") {
      options.warmup = std::stoi(value());
    } else if (argument == "--runs") {
      options.runs = std::stoi(value());
    } else if (argument == "--inner") {
      options.inner = std::stoi(value());
    } else if (argument == "--include-ilv") {
      options.include_ilv = true;
    } else if (argument == "--include-column-pipeline") {
      options.include_column_pipeline = true;
    } else if (argument == "--help") {
      std::cout << "usage: bench_jit_vs_i8mm_pure_gemm [--rows 1,2,4,8,12] [--k K] [--n N]"
                   " [--experts E] [--warmup W] [--runs R] [--inner I] [--include-ilv]"
                   " [--include-column-pipeline]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + argument);
    }
  }
  if (options.rows.empty() || options.k <= 0 || options.n <= 0 || options.experts <= 0 || options.warmup < 0 ||
      options.runs <= 0 || options.inner <= 0) {
    throw std::invalid_argument("invalid non-positive benchmark option");
  }
  if (options.k % 8 != 0 || options.n % fused_cpp::moe_sve::kNTile != 0) {
    throw std::invalid_argument("K must be divisible by 8 and N by the compiled SVE N tile");
  }
  for (const int rows : options.rows) {
    if (rows != 1 && rows != 2 && rows != 4 && rows != 8 && rows != 12) {
      throw std::invalid_argument("direct i8mm comparison supports M=1,2,4,8,12");
    }
    if (options.include_ilv && rows != 8 && rows != 12) {
      throw std::invalid_argument("upstream ILV comparison supports only M=8,12");
    }
    if (options.include_column_pipeline && rows != 12) {
      throw std::invalid_argument("column-pipelined comparison supports only M=12");
    }
  }
  return options;
}

void check_runtime_vector_length() {
  uint64_t runtime_bytes = 0;
  asm volatile("cntb %0" : "=r"(runtime_bytes));
  if (runtime_bytes != fused_cpp::moe_sve::kVectorBytes) {
    throw std::runtime_error("runtime SVE vector length does not match the compiled JIT vector length");
  }
}

void fill_inputs(std::vector<uint16_t>* a, std::vector<uint16_t>* b, int rows, int k, int n) {
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < k; ++column) {
      const int value = (row * 37 + column * 17 + 11) % 257 - 128;
      (*a)[static_cast<size_t>(row) * k + column] = to_bf16(static_cast<float>(value) / 128.0f);
    }
  }
  for (int row = 0; row < k; ++row) {
    for (int column = 0; column < n; ++column) {
      const int value = (row * 29 + column * 13 + 7) % 251 - 125;
      (*b)[static_cast<size_t>(row) * n + column] = to_bf16(static_cast<float>(value) / 256.0f);
    }
  }
}

std::vector<uint16_t> pack_a(const std::vector<uint16_t>& source, int rows, int physical_rows, int k) {
  std::vector<uint16_t> packed(static_cast<size_t>(physical_rows) * k, 0);
  for (int kb = 0; kb < k; kb += 4) {
    uint16_t* block = packed.data() + static_cast<size_t>(kb / 4) * physical_rows * 4;
    for (int row = 0; row < rows; ++row) {
      std::copy_n(source.data() + static_cast<size_t>(row) * k + kb, 4, block + row * 4);
    }
  }
  return packed;
}

std::vector<uint16_t> pack_b(const std::vector<uint16_t>& source, int k, int n) {
  constexpr int segments = fused_cpp::moe_sve::kSegments128;
  constexpr int n_tile = fused_cpp::moe_sve::kNTile;
  std::vector<uint16_t> packed(static_cast<size_t>(k) * n);
  size_t index = 0;
  for (int nb = 0; nb < n; nb += n_tile) {
    for (int kb = 0; kb < k; kb += 4) {
      for (int column_pair = 0; column_pair < 4; ++column_pair) {
        for (int segment = 0; segment < segments; ++segment) {
          const int column = nb + segment * 8 + column_pair * 2;
          for (int kk = 0; kk < 4; ++kk) {
            packed[index++] = source[static_cast<size_t>(kb + kk) * n + column];
          }
          for (int kk = 0; kk < 4; ++kk) {
            packed[index++] = source[static_cast<size_t>(kb + kk) * n + column + 1];
          }
        }
      }
    }
  }
  return packed;
}

I8mmKernelFn i8mm_kernel(int rows) {
  switch (rows) {
    case 1:
      return bf16gemm_k_nld1_f;
    case 2:
      return bf16gemm_k_nld2_f;
    case 4:
      return bf16gemm_k_nld4_f;
    case 8:
      return bf16gemm_k_nld_f;
    case 12:
      return bf16gemm_k_nld_f_m12;
    default:
      throw std::invalid_argument("unsupported i8mm M");
  }
}

I8mmKernelFn i8mm_ilv_kernel(int rows) {
  switch (rows) {
    case 8:
      return bf16gemm_k_nld_f_m8_ilv;
    case 12:
      return bf16gemm_k_nld_f_m12_ilv;
    default:
      throw std::invalid_argument("unsupported i8mm ILV M");
  }
}

double median(std::vector<double> samples) {
  const size_t middle = samples.size() / 2;
  std::nth_element(samples.begin(), samples.begin() + middle, samples.end());
  return samples[middle];
}

template <typename Call>
double time_call(Call&& call, int inner) {
  const auto begin = Clock::now();
  for (int iteration = 0; iteration < inner; ++iteration) {
    call(iteration);
  }
  const auto elapsed = std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
  return elapsed / inner;
}

bool run_case(const Options& options, int rows) {
  const int physical_rows = rows <= 8 ? 8 : 12;
  std::vector<uint16_t> a(static_cast<size_t>(rows) * options.k);
  std::vector<uint16_t> b(static_cast<size_t>(options.k) * options.n);
  fill_inputs(&a, &b, rows, options.k, options.n);
  const std::vector<uint16_t> packed_a = pack_a(a, rows, physical_rows, options.k);
  const std::vector<uint16_t> one_packed_b = pack_b(b, options.k, options.n);
  std::vector<uint16_t> packed_b(static_cast<size_t>(options.experts) * one_packed_b.size());
  for (int expert = 0; expert < options.experts; ++expert) {
    std::copy(one_packed_b.begin(), one_packed_b.end(),
              packed_b.begin() + static_cast<size_t>(expert) * one_packed_b.size());
  }

  ExtendedParams params;
  params.gemm = gemm_params_t{rows, options.k, options.n, options.k, options.k, options.n};
  params.kc = options.k;
  params.packed_n = options.n;
  std::string error;
  const fused_cpp::moe_sve::jit::KernelFn jit_kernel = fused_cpp::moe_sve::jit::get_gemm_f32_kernel(rows, &error);
  if (jit_kernel == nullptr) {
    throw std::runtime_error("JIT generation failed: " + error);
  }
  const I8mmKernelFn reference_kernel = i8mm_kernel(rows);
  const I8mmKernelFn ilv_kernel = options.include_ilv ? i8mm_ilv_kernel(rows) : nullptr;
  const fused_cpp::moe_sve::jit::KernelFn column_kernel =
      options.include_column_pipeline
          ? fused_cpp::moe_sve::jit::get_probe_kernel(
                rows, fused_cpp::moe_sve::jit::ProbeMode::kFullWithStoreColumnPipeline, &error)
          : nullptr;
  if (options.include_column_pipeline && column_kernel == nullptr) {
    throw std::runtime_error("column-pipelined JIT generation failed: " + error);
  }
  std::vector<float> jit_output(static_cast<size_t>(physical_rows) * options.n,
                                std::numeric_limits<float>::quiet_NaN());
  std::vector<float> i8mm_output(static_cast<size_t>(physical_rows) * options.n,
                                 std::numeric_limits<float>::quiet_NaN());
  std::vector<float> ilv_output;
  if (options.include_ilv) {
    ilv_output.assign(static_cast<size_t>(physical_rows) * options.n, std::numeric_limits<float>::quiet_NaN());
  }
  std::vector<float> column_output;
  if (options.include_column_pipeline) {
    column_output.assign(static_cast<size_t>(physical_rows) * options.n, std::numeric_limits<float>::quiet_NaN());
  }

  auto call_jit = [&](int64_t iteration) {
    const size_t expert = static_cast<size_t>(iteration % options.experts);
    const uint16_t* weight = packed_b.data() + expert * one_packed_b.size();
    jit_kernel(packed_a.data(), weight, jit_output.data(), nullptr, &params.gemm);
  };
  auto call_i8mm = [&](int64_t iteration) {
    const size_t expert = static_cast<size_t>(iteration % options.experts);
    const uint16_t* weight = packed_b.data() + expert * one_packed_b.size();
    reference_kernel(a.data(), weight, i8mm_output.data(), const_cast<uint16_t*>(packed_a.data()), &params.gemm);
  };
  auto call_ilv = [&](int64_t iteration) {
    const size_t expert = static_cast<size_t>(iteration % options.experts);
    const uint16_t* weight = packed_b.data() + expert * one_packed_b.size();
    ilv_kernel(a.data(), weight, ilv_output.data(), const_cast<uint16_t*>(packed_a.data()), &params.gemm);
  };
  auto call_column = [&](int64_t iteration) {
    const size_t expert = static_cast<size_t>(iteration % options.experts);
    const uint16_t* weight = packed_b.data() + expert * one_packed_b.size();
    column_kernel(packed_a.data(), weight, column_output.data(), nullptr, &params.gemm);
  };

  call_jit(0);
  call_i8mm(0);
  if (options.include_ilv) {
    call_ilv(0);
  }
  if (options.include_column_pipeline) {
    call_column(0);
  }
  const size_t compared_bytes = static_cast<size_t>(rows) * options.n * sizeof(float);
  const bool bitwise_equal = std::memcmp(jit_output.data(), i8mm_output.data(), compared_bytes) == 0;
  const bool ilv_bitwise_equal =
      !options.include_ilv || std::memcmp(i8mm_output.data(), ilv_output.data(), compared_bytes) == 0;
  const bool column_bitwise_equal =
      !options.include_column_pipeline || std::memcmp(i8mm_output.data(), column_output.data(), compared_bytes) == 0;
  bool finite_output = true;
  double max_abs_diff = 0.0;
  double ilv_max_abs_diff = 0.0;
  double column_max_abs_diff = 0.0;
  for (size_t i = 0; i < static_cast<size_t>(rows) * options.n; ++i) {
    finite_output = finite_output && std::isfinite(jit_output[i]) && std::isfinite(i8mm_output[i]);
    max_abs_diff = std::max(max_abs_diff, std::abs(static_cast<double>(jit_output[i] - i8mm_output[i])));
    if (options.include_ilv) {
      finite_output = finite_output && std::isfinite(ilv_output[i]);
      ilv_max_abs_diff = std::max(ilv_max_abs_diff, std::abs(static_cast<double>(ilv_output[i] - i8mm_output[i])));
    }
    if (options.include_column_pipeline) {
      finite_output = finite_output && std::isfinite(column_output[i]);
      column_max_abs_diff =
          std::max(column_max_abs_diff, std::abs(static_cast<double>(column_output[i] - i8mm_output[i])));
    }
  }
  if (!bitwise_equal || !ilv_bitwise_equal || !column_bitwise_equal || !finite_output) {
    std::cerr << "correctness mismatch for M=" << rows << ", finite=" << finite_output
              << ", max_abs_diff=" << max_abs_diff << ", ilv_max_abs_diff=" << ilv_max_abs_diff
              << ", column_max_abs_diff=" << column_max_abs_diff << '\n';
    return false;
  }

  int64_t cursor = 0;
  for (int iteration = 0; iteration < options.warmup; ++iteration) {
    call_jit(cursor++);
    call_i8mm(cursor++);
    if (options.include_ilv) {
      call_ilv(cursor++);
    }
  }
  std::vector<double> jit_samples;
  std::vector<double> i8mm_samples;
  std::vector<double> ilv_samples;
  jit_samples.reserve(options.runs);
  i8mm_samples.reserve(options.runs);
  ilv_samples.reserve(options.runs);
  if (options.include_ilv) {
    constexpr std::array<std::array<int, 3>, 6> kOrders{{
        {{0, 1, 2}},
        {{0, 2, 1}},
        {{1, 0, 2}},
        {{1, 2, 0}},
        {{2, 0, 1}},
        {{2, 1, 0}},
    }};
    auto measure_variant = [&](int variant) {
      switch (variant) {
        case 0:
          jit_samples.push_back(time_call([&](int inner) { call_jit(cursor + inner); }, options.inner));
          break;
        case 1:
          i8mm_samples.push_back(time_call([&](int inner) { call_i8mm(cursor + inner); }, options.inner));
          break;
        case 2:
          ilv_samples.push_back(time_call([&](int inner) { call_ilv(cursor + inner); }, options.inner));
          break;
        default:
          throw std::logic_error("invalid benchmark variant");
      }
      cursor += options.inner;
    };
    for (int sample = 0; sample < options.runs; ++sample) {
      for (const int variant : kOrders[static_cast<size_t>(sample) % kOrders.size()]) {
        measure_variant(variant);
      }
    }
  } else {
    for (int sample = 0; sample < options.runs; ++sample) {
      if ((sample & 1) == 0) {
        jit_samples.push_back(time_call([&](int inner) { call_jit(cursor + inner); }, options.inner));
        cursor += options.inner;
        i8mm_samples.push_back(time_call([&](int inner) { call_i8mm(cursor + inner); }, options.inner));
        cursor += options.inner;
      } else {
        i8mm_samples.push_back(time_call([&](int inner) { call_i8mm(cursor + inner); }, options.inner));
        cursor += options.inner;
        jit_samples.push_back(time_call([&](int inner) { call_jit(cursor + inner); }, options.inner));
        cursor += options.inner;
      }
    }
  }

  const double jit_us = median(jit_samples);
  const double i8mm_us = median(i8mm_samples);
  const double ilv_us = options.include_ilv ? median(ilv_samples) : 0.0;
  const double work = 2.0 * rows * options.k * options.n;
  const double jit_gflops = work / (jit_us * 1.0e3);
  const double i8mm_gflops = work / (i8mm_us * 1.0e3);
  const double ilv_gflops = options.include_ilv ? work / (ilv_us * 1.0e3) : 0.0;
  const double regression_percent = (jit_us / i8mm_us - 1.0) * 100.0;

  std::cout << std::fixed << std::setprecision(6) << "{\"m\":" << rows << ",\"k\":" << options.k
            << ",\"n\":" << options.n << ",\"sve_bits\":" << fused_cpp::moe_sve::kVectorBits
            << ",\"experts\":" << options.experts << ",\"samples\":" << options.runs << ",\"jit_median_us\":" << jit_us
            << ",\"i8mm_median_us\":" << i8mm_us << ",\"jit_gflops\":" << jit_gflops
            << ",\"i8mm_gflops\":" << i8mm_gflops << ",\"jit_time_over_i8mm\":" << (jit_us / i8mm_us)
            << ",\"regression_percent\":" << regression_percent << ",\"max_abs_diff\":" << max_abs_diff
            << ",\"bitwise_equal\":" << (bitwise_equal ? "true" : "false");
  if (options.include_ilv) {
    std::cout << ",\"ilv_median_us\":" << ilv_us << ",\"ilv_gflops\":" << ilv_gflops
              << ",\"ilv_time_over_non_ilv\":" << (ilv_us / i8mm_us)
              << ",\"ilv_speedup_percent\":" << (i8mm_us / ilv_us - 1.0) * 100.0
              << ",\"ilv_max_abs_diff\":" << ilv_max_abs_diff
              << ",\"ilv_bitwise_equal\":" << (ilv_bitwise_equal ? "true" : "false");
  }
  if (options.include_column_pipeline) {
    std::cout << ",\"column_max_abs_diff\":" << column_max_abs_diff
              << ",\"column_bitwise_equal\":" << (column_bitwise_equal ? "true" : "false");
  }
  std::cout << "}\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    check_runtime_vector_length();
    bool passed = true;
    for (const int rows : options.rows) {
      passed = run_case(options, rows) && passed;
    }
    return passed ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
