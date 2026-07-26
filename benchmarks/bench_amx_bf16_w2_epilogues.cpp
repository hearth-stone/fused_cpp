// SPDX-License-Identifier: Apache-2.0
#include "../csrc/moe/x86/avx512_bf16/backend.h"
#include "../csrc/moe/x86/avx512_bf16/kernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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

namespace x86_moe = ::fused_cpp::moe::x86::avx512_bf16;

constexpr const char* kEnvironment = "FUSED_CPP_MOE_AMX_W2_EPILOGUE";

uint16_t Bf16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

void PackA(const uint16_t* source, uint16_t* packed, int rows, int k_size, int k_pad) {
  std::fill(packed, packed + static_cast<int64_t>(rows) * k_pad, static_cast<uint16_t>(0));
  for (int row = 0; row < rows; ++row) {
    std::copy(source + static_cast<int64_t>(row) * k_size, source + static_cast<int64_t>(row + 1) * k_size,
              packed + static_cast<int64_t>(row) * k_pad);
  }
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
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

std::vector<std::string> ParseModes(const char* raw) {
  const std::string value = raw == nullptr ? "baseline,combined,tile_store" : raw;
  std::vector<std::string> modes;
  size_t begin = 0;
  while (begin <= value.size()) {
    const size_t end = value.find(',', begin);
    const std::string mode = value.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    if (mode != "auto" && mode != "baseline" && mode != "combined" && mode != "tile_store") {
      throw std::invalid_argument("unknown W2 epilogue mode '" + mode + "'");
    }
    if (std::find(modes.begin(), modes.end(), mode) == modes.end()) {
      modes.push_back(mode);
    }
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  if (modes.empty()) {
    throw std::invalid_argument("at least one W2 epilogue mode is required");
  }
  return modes;
}

void SelectMode(const std::string& mode) {
  if (mode == "auto") {
    unsetenv(kEnvironment);
  } else {
    setenv(kEnvironment, mode.c_str(), 1);
  }
}

double Gflops(double flops, double milliseconds) { return flops / milliseconds / 1.0e6; }

}  // namespace

int main(int argc, char** argv) {
  try {
    const int m = argc > 1 ? ParsePositive(argv, 1, 16, "M") : 16;
    const int k = argc > 2 ? ParsePositive(argv, 2, 512, "K") : 512;
    const int n = argc > 3 ? ParsePositive(argv, 3, 4096, "N") : 4096;
    const int warmup = argc > 4 ? ParsePositive(argv, 4, 20, "warmup") : 20;
    const int runs = argc > 5 ? ParsePositive(argv, 5, 101, "runs") : 101;
    const std::vector<std::string> modes = ParseModes(argc > 6 ? argv[6] : nullptr);
    if (!x86_moe::AmxRuntimeSupported()) {
      std::cerr << "AMX BF16 is not available at runtime\n";
      return 2;
    }

    std::mt19937 generator(20260720);
    std::normal_distribution<float> distribution(0.0f, 0.01f);
    std::vector<uint16_t> input(static_cast<size_t>(m) * k);
    std::vector<uint16_t> weight(static_cast<size_t>(n) * k);
    for (uint16_t& value : input) {
      value = Bf16Bits(distribution(generator));
    }
    for (uint16_t& value : weight) {
      value = Bf16Bits(distribution(generator));
    }

    const int k_pad = x86_moe::AmxRoundK(k);
    const int n_pad = x86_moe::RoundN(n);
    std::vector<uint16_t> packed_a(static_cast<size_t>(m) * k_pad);
    std::vector<uint16_t> packed_b(static_cast<size_t>(k_pad) * n_pad);
    std::vector<int64_t> route_ids(static_cast<size_t>(m));
    for (int row = 0; row < m; ++row) {
      route_ids[static_cast<size_t>(row)] = row;
    }
    PackA(input.data(), packed_a.data(), m, k, k_pad);
    x86_moe::PackW2(weight.data(), packed_b.data(), n, k, k_pad, n_pad);

    std::vector<std::vector<float>> outputs(modes.size(), std::vector<float>(static_cast<size_t>(m) * n));
    auto run = [&](size_t variant) {
      SelectMode(modes[variant]);
      x86_moe::ComputeW2Amx(packed_a.data(), k_pad, packed_b.data(), outputs[variant].data(), nullptr, route_ids.data(),
                            n, m, k_pad, n, 0, n_pad / 32, false);
    };
    for (size_t variant = 0; variant < modes.size(); ++variant) {
      run(variant);
    }
    size_t mismatches = 0;
    float max_abs = 0.0f;
    for (size_t variant = 1; variant < modes.size(); ++variant) {
      for (size_t index = 0; index < outputs[variant].size(); ++index) {
        mismatches += outputs[variant][index] != outputs[0][index];
        max_abs = std::max(max_abs, std::abs(outputs[variant][index] - outputs[0][index]));
      }
    }
    for (int iteration = 0; iteration < warmup; ++iteration) {
      for (size_t variant = 0; variant < modes.size(); ++variant) {
        run(variant);
      }
    }

    std::vector<std::vector<double>> samples(modes.size());
    for (std::vector<double>& variant_samples : samples) {
      variant_samples.reserve(static_cast<size_t>(runs));
    }
    for (int iteration = 0; iteration < runs; ++iteration) {
      const size_t offset = static_cast<size_t>(iteration) % modes.size();
      for (size_t order = 0; order < modes.size(); ++order) {
        const size_t variant = (offset + order) % modes.size();
        const auto start = std::chrono::steady_clock::now();
        run(variant);
        samples[variant].push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
      }
    }

    const double flops = 2.0 * m * k * n;
    const double output_bytes = static_cast<double>(m) * n * sizeof(float);
    const x86_moe::JitStats jit_stats = x86_moe::GetJitStats();
    std::cout << std::setprecision(10) << "{\"m\":" << m << ",\"k\":" << k << ",\"n\":" << n << ",\"warmup\":" << warmup
              << ",\"runs\":" << runs << ",\"mismatches\":" << mismatches << ",\"max_abs\":" << max_abs
              << ",\"jit\":{\"kernel_count\":" << jit_stats.kernel_count << ",\"code_bytes\":" << jit_stats.code_bytes
              << "},\"variants\":{";
    for (size_t variant = 0; variant < modes.size(); ++variant) {
      const double median_ms = Median(samples[variant]);
      const double best_ms = *std::min_element(samples[variant].begin(), samples[variant].end());
      if (variant != 0) {
        std::cout << ',';
      }
      std::cout << '\"' << modes[variant] << "\":{\"median_ms\":" << median_ms << ",\"best_ms\":" << best_ms
                << ",\"median_gflops\":" << Gflops(flops, median_ms) << ",\"best_gflops\":" << Gflops(flops, best_ms)
                << ",\"median_effective_output_gbps\":" << output_bytes / median_ms / 1.0e6 << '}';
    }
    std::cout << "}}\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
  return 0;
}
