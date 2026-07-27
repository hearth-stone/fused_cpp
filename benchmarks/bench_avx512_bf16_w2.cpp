// SPDX-License-Identifier: Apache-2.0
#include "../csrc/moe/x86/avx512_bf16/backend.h"
#include "../csrc/moe/x86/avx512_bf16/kernels.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace avx512_moe = ::fused_cpp::moe::x86::avx512_bf16;

uint16_t Bf16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

void PackA(const uint16_t* source, uint16_t* packed, int rows, int k_size, int k_pad) {
  constexpr int kRowsPerPanel = 12;
  constexpr int kPhysicalRowsPerPanel = 16;
  const int panels = (rows + kRowsPerPanel - 1) / kRowsPerPanel;
  std::fill(packed, packed + static_cast<int64_t>(panels) * kPhysicalRowsPerPanel * k_pad, static_cast<uint16_t>(0));
  for (int panel = 0; panel < panels; ++panel) {
    uint16_t* destination = packed + static_cast<int64_t>(panel) * kPhysicalRowsPerPanel * k_pad;
    for (int kp = 0; kp < k_pad / 2; ++kp) {
      for (int row = 0; row < kRowsPerPanel && panel * kRowsPerPanel + row < rows; ++row) {
        for (int half = 0; half < 2 && kp * 2 + half < k_size; ++half) {
          destination[static_cast<int64_t>(kp) * 32 + row * 2 + half] =
              source[static_cast<int64_t>(panel * kRowsPerPanel + row) * k_size + kp * 2 + half];
        }
      }
    }
  }
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

template <typename Function>
std::pair<double, double> Measure(Function&& function, int warmup, int runs) {
  for (int iteration = 0; iteration < warmup; ++iteration) {
    function();
  }
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(runs));
  for (int iteration = 0; iteration < runs; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    function();
    samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
  }
  return {Median(samples), *std::min_element(samples.begin(), samples.end())};
}

int ParsePositive(char** argv, int index, int default_value, const char* name) {
  if (argv[index] == nullptr) {
    return default_value;
  }
  const int value = std::stoi(argv[index]);
  if (value <= 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
  return value;
}

double Gflops(double flops, double milliseconds) { return flops / milliseconds / 1.0e6; }

}  // namespace

int main(int argc, char** argv) {
  try {
    const int m = argc > 1 ? ParsePositive(argv, 1, 12, "M") : 12;
    const int k = argc > 2 ? ParsePositive(argv, 2, 512, "K") : 512;
    const int n = argc > 3 ? ParsePositive(argv, 3, 4096, "N") : 4096;
    const int warmup = argc > 4 ? ParsePositive(argv, 4, 10, "warmup") : 10;
    const int runs = argc > 5 ? ParsePositive(argv, 5, 51, "runs") : 51;
    if (!avx512_moe::RuntimeSupported()) {
      std::cerr << "AVX-512 BF16 is not available at runtime\n";
      return 2;
    }

    std::mt19937 generator(20260726);
    std::normal_distribution<float> distribution(0.0f, 0.01f);
    std::vector<uint16_t> input(static_cast<size_t>(m) * k);
    std::vector<uint16_t> weight(static_cast<size_t>(n) * k);
    for (uint16_t& value : input) {
      value = Bf16Bits(distribution(generator));
    }
    for (uint16_t& value : weight) {
      value = Bf16Bits(distribution(generator));
    }

    const int k_pad = avx512_moe::RoundK(k);
    const int n_pad = avx512_moe::RoundN(n);
    const int panels = (m + 11) / 12;
    std::vector<uint16_t> packed_a(static_cast<size_t>(panels) * 16 * k_pad);
    std::vector<uint16_t> packed_b(static_cast<size_t>(k_pad) * n_pad);
    std::vector<float> output(static_cast<size_t>(m) * n);
    std::vector<int64_t> route_ids(static_cast<size_t>(m));
    for (int row = 0; row < m; ++row) {
      route_ids[static_cast<size_t>(row)] = row;
    }
    PackA(input.data(), packed_a.data(), m, k, k_pad);
    avx512_moe::PackW2(weight.data(), packed_b.data(), n, k, k_pad, n_pad);

    auto kernel = [&]() {
      avx512_moe::ComputeW2(packed_a.data(), k_pad, packed_b.data(), output.data(), nullptr, route_ids.data(), n, m,
                            k_pad, n, 0, n_pad / 32, false);
    };
    kernel();
    const avx512_moe::JitStats jit_stats = avx512_moe::GetJitStats();
    const auto [median_ms, best_ms] = Measure(kernel, warmup, runs);
    double checksum = 0.0;
    for (float value : output) {
      checksum += value;
    }

    const double flops = 2.0 * m * k * n;
    const char* implementation = std::getenv("FUSED_CPP_MOE_AVX512_IMPL");
    const char* k_loop = std::getenv("FUSED_CPP_MOE_AVX512_K_LOOP");
    std::cout << std::setprecision(10) << "{\"m\":" << m << ",\"k\":" << k << ",\"n\":" << n << ",\"warmup\":" << warmup
              << ",\"runs\":" << runs << ",\"custom_impl\":\"" << (implementation == nullptr ? "auto" : implementation)
              << "\",\"k_loop\":\"" << (k_loop == nullptr ? "auto" : k_loop)
              << "\",\"jit\":{\"kernel_count\":" << jit_stats.kernel_count << ",\"code_bytes\":" << jit_stats.code_bytes
              << ",\"generation_ms\":" << (static_cast<double>(jit_stats.generation_nanoseconds) / 1.0e6)
              << "},\"checksum\":" << checksum << ",\"kernel\":{\"median_ms\":" << median_ms
              << ",\"best_ms\":" << best_ms << ",\"median_gflops\":" << Gflops(flops, median_ms)
              << ",\"best_gflops\":" << Gflops(flops, best_ms) << "}}\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
  return 0;
}
