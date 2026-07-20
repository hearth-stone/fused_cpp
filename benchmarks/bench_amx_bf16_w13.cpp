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

uint16_t Bf16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

float Bf16Float(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
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

std::vector<std::string> ParseModes(const std::string& csv) {
  std::vector<std::string> modes;
  size_t begin = 0;
  while (begin <= csv.size()) {
    const size_t end = csv.find(',', begin);
    const std::string mode = csv.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    if (mode.empty()) {
      throw std::invalid_argument("SiLU epilogue list contains an empty mode");
    }
    modes.push_back(mode);
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  return modes;
}

void SelectMode(const std::string& mode) {
  if (::setenv("FUSED_CPP_MOE_AMX_SILU_EPILOGUE", mode.c_str(), 1) != 0) {
    throw std::runtime_error("could not set FUSED_CPP_MOE_AMX_SILU_EPILOGUE");
  }
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
    const int m = argc > 1 ? ParsePositive(argv, 1, 16, "M") : 16;
    const int k = argc > 2 ? ParsePositive(argv, 2, 4096, "K") : 4096;
    const int features = argc > 3 ? ParsePositive(argv, 3, 512, "F") : 512;
    const int warmup = argc > 4 ? ParsePositive(argv, 4, 10, "warmup") : 10;
    const int runs = argc > 5 ? ParsePositive(argv, 5, 51, "runs") : 51;
    const int degree = argc > 6 ? ParsePositive(argv, 6, 5, "degree") : 5;
    if (degree != 4 && degree != 5 && degree != 6) {
      throw std::invalid_argument("degree must be 4, 5, or 6");
    }
    const char* current_epilogue = std::getenv("FUSED_CPP_MOE_AMX_SILU_EPILOGUE");
    const std::vector<std::string> modes =
        ParseModes(argc > 7 ? argv[7] : (current_epilogue == nullptr ? "auto" : current_epilogue));
    if (!x86_moe::AmxRuntimeSupported()) {
      std::cerr << "AMX BF16 is not available at runtime\n";
      return 2;
    }

    std::mt19937 generator(20260719);
    std::normal_distribution<float> distribution(0.0f, 0.01f);
    std::vector<uint16_t> input(static_cast<size_t>(m) * k);
    std::vector<uint16_t> weight(static_cast<size_t>(2) * features * k);
    for (uint16_t& value : input) {
      value = Bf16Bits(distribution(generator));
    }
    for (uint16_t& value : weight) {
      value = Bf16Bits(distribution(generator));
    }

    const int k_pad = x86_moe::AmxRoundK(k);
    const int feature_pad = ((features + 15) / 16) * 16;
    std::vector<uint16_t> packed_a(static_cast<size_t>(m) * k_pad);
    std::vector<uint16_t> packed_b(static_cast<size_t>(2) * feature_pad * k_pad);
    const size_t output_elements = static_cast<size_t>(m) * feature_pad;
    std::vector<std::vector<uint16_t>> outputs(modes.size(), std::vector<uint16_t>(output_elements));
    PackA(input.data(), packed_a.data(), m, k, k_pad);
    x86_moe::PackW13(weight.data(), packed_b.data(), features, k, k_pad, feature_pad);

    auto kernel = [&](size_t mode_index) {
      x86_moe::ComputeW13Amx(packed_a.data(), k_pad, packed_b.data(), outputs[mode_index].data(), feature_pad, m, k_pad,
                             0, feature_pad / 16, degree);
    };

    // Generate every specialization before warm-up. Measurement order rotates
    // by run so thermal/frequency drift is shared by all variants.
    for (size_t mode = 0; mode < modes.size(); ++mode) {
      SelectMode(modes[mode]);
      kernel(mode);
    }
    const x86_moe::JitStats jit_stats = x86_moe::GetJitStats();
    for (int iteration = 0; iteration < warmup; ++iteration) {
      for (size_t offset = 0; offset < modes.size(); ++offset) {
        const size_t mode = (static_cast<size_t>(iteration) + offset) % modes.size();
        SelectMode(modes[mode]);
        kernel(mode);
      }
    }

    std::vector<std::vector<double>> samples(modes.size());
    for (std::vector<double>& mode_samples : samples) {
      mode_samples.reserve(static_cast<size_t>(runs));
    }
    for (int iteration = 0; iteration < runs; ++iteration) {
      for (size_t offset = 0; offset < modes.size(); ++offset) {
        const size_t mode = (static_cast<size_t>(iteration) + offset) % modes.size();
        SelectMode(modes[mode]);
        const auto start = std::chrono::steady_clock::now();
        kernel(mode);
        samples[mode].push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
      }
    }

    size_t reference_mode = 0;
    const auto baseline = std::find(modes.begin(), modes.end(), "baseline");
    if (baseline != modes.end()) {
      reference_mode = static_cast<size_t>(baseline - modes.begin());
    }
    std::vector<uint64_t> checksums(modes.size());
    std::vector<uint64_t> mismatches(modes.size());
    std::vector<float> max_abs(modes.size());
    for (size_t mode = 0; mode < modes.size(); ++mode) {
      for (size_t element = 0; element < output_elements; ++element) {
        checksums[mode] += outputs[mode][element];
        if (outputs[mode][element] != outputs[reference_mode][element]) {
          ++mismatches[mode];
        }
        max_abs[mode] = std::max(
            max_abs[mode], std::abs(Bf16Float(outputs[mode][element]) - Bf16Float(outputs[reference_mode][element])));
      }
    }

    const double flops = 4.0 * m * k * features;
    std::cout << std::setprecision(10) << "{\"m\":" << m << ",\"k\":" << k << ",\"features\":" << features
              << ",\"degree\":" << degree << ",\"warmup\":" << warmup << ",\"runs\":" << runs
              << ",\"custom_impl\":\"amx_jit\",\"jit\":{\"kernel_count\":" << jit_stats.kernel_count
              << ",\"code_bytes\":" << jit_stats.code_bytes
              << ",\"generation_ms\":" << (static_cast<double>(jit_stats.generation_nanoseconds) / 1.0e6) << "}";
    if (modes.size() == 1) {
      const double median_ms = Median(samples[0]);
      const double best_ms = *std::min_element(samples[0].begin(), samples[0].end());
      std::cout << ",\"silu_epilogue\":\"" << modes[0] << "\",\"checksum\":" << checksums[0]
                << ",\"kernel\":{\"median_ms\":" << median_ms << ",\"best_ms\":" << best_ms
                << ",\"median_gflops\":" << Gflops(flops, median_ms) << ",\"best_gflops\":" << Gflops(flops, best_ms)
                << "}}\n";
      return 0;
    }
    std::cout << ",\"reference\":\"" << modes[reference_mode] << "\",\"variants\":[";
    for (size_t mode = 0; mode < modes.size(); ++mode) {
      if (mode != 0) {
        std::cout << ',';
      }
      const double median_ms = Median(samples[mode]);
      const double best_ms = *std::min_element(samples[mode].begin(), samples[mode].end());
      std::cout << "{\"silu_epilogue\":\"" << modes[mode] << "\",\"checksum\":" << checksums[mode]
                << ",\"mismatch_count\":" << mismatches[mode] << ",\"max_abs\":" << max_abs[mode]
                << ",\"median_ms\":" << median_ms << ",\"best_ms\":" << best_ms
                << ",\"median_gflops\":" << Gflops(flops, median_ms) << ",\"best_gflops\":" << Gflops(flops, best_ms)
                << '}';
    }
    std::cout << "]}\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
  return 0;
}
