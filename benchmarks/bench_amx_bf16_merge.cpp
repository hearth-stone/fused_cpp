// SPDX-License-Identifier: Apache-2.0
#include "../csrc/moe/x86/avx512_bf16/kernels.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
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

struct Measurement {
  double median_ms = 0.0;
  double best_ms = 0.0;
};

template <typename FlatFunction, typename MappedFunction>
std::pair<Measurement, Measurement> MeasureRotated(FlatFunction&& flat, MappedFunction&& mapped, int warmup, int runs) {
  for (int iteration = 0; iteration < warmup; ++iteration) {
    flat();
    mapped();
  }
  std::vector<double> flat_samples;
  std::vector<double> mapped_samples;
  flat_samples.reserve(static_cast<size_t>(runs));
  mapped_samples.reserve(static_cast<size_t>(runs));
  auto measure = [](auto&& function) {
    const auto start = std::chrono::steady_clock::now();
    function();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  };
  for (int iteration = 0; iteration < runs; ++iteration) {
    if ((iteration & 1) == 0) {
      flat_samples.push_back(measure(flat));
      mapped_samples.push_back(measure(mapped));
    } else {
      mapped_samples.push_back(measure(mapped));
      flat_samples.push_back(measure(flat));
    }
  }
  return {{Median(flat_samples), *std::min_element(flat_samples.begin(), flat_samples.end())},
          {Median(mapped_samples), *std::min_element(mapped_samples.begin(), mapped_samples.end())}};
}

double BandwidthGbPerSecond(double bytes, double milliseconds) { return bytes / milliseconds / 1.0e6; }

}  // namespace

int main(int argc, char** argv) {
  try {
    const int tokens = argc > 1 ? ParsePositive(argv, 1, 256, "tokens") : 256;
    const int hidden = argc > 2 ? ParsePositive(argv, 2, 4096, "hidden") : 4096;
    const int experts = argc > 3 ? ParsePositive(argv, 3, 8, "experts") : 8;
    const int top_k = argc > 4 ? ParsePositive(argv, 4, 2, "top_k") : 2;
    const int warmup = argc > 5 ? ParsePositive(argv, 5, 20, "warmup") : 20;
    const int runs = argc > 6 ? ParsePositive(argv, 6, 101, "runs") : 101;
    const int64_t routes = static_cast<int64_t>(tokens) * top_k;

    std::mt19937 generator(20260720);
    std::normal_distribution<float> value_distribution(0.0f, 0.1f);
    std::uniform_real_distribution<float> weight_distribution(0.0f, 1.0f);
    std::vector<float> flat_output(static_cast<size_t>(routes) * hidden);
    std::vector<float> mapped_output(flat_output.size());
    std::vector<float> weights(static_cast<size_t>(routes));
    std::vector<int64_t> route_rows(static_cast<size_t>(routes));
    std::vector<int64_t> expert_counts(static_cast<size_t>(experts), 0);
    std::vector<int64_t> expert_offsets(static_cast<size_t>(experts), 0);

    for (float& value : flat_output) {
      value = value_distribution(generator);
    }
    for (int64_t flat = 0; flat < routes; ++flat) {
      weights[static_cast<size_t>(flat)] = weight_distribution(generator);
      ++expert_counts[static_cast<size_t>(flat % experts)];
    }
    for (int expert = 1; expert < experts; ++expert) {
      expert_offsets[static_cast<size_t>(expert)] =
          expert_offsets[static_cast<size_t>(expert - 1)] + expert_counts[static_cast<size_t>(expert - 1)];
    }
    std::vector<int64_t> expert_rows = expert_offsets;
    for (int64_t flat = 0; flat < routes; ++flat) {
      const int expert = static_cast<int>(flat % experts);
      const int64_t row = expert_rows[static_cast<size_t>(expert)]++;
      route_rows[static_cast<size_t>(flat)] = row;
      std::copy(flat_output.begin() + flat * hidden, flat_output.begin() + (flat + 1) * hidden,
                mapped_output.begin() + row * hidden);
    }

    std::vector<uint16_t> flat_result(static_cast<size_t>(tokens) * hidden);
    std::vector<uint16_t> mapped_result(flat_result.size());
    auto flat_merge = [&]() {
      x86_moe::MergeRoutes(flat_output.data(), weights.data(), flat_result.data(), 0, tokens, top_k, hidden);
    };
    auto mapped_merge = [&]() {
      x86_moe::MergeRoutesMapped(mapped_output.data(), route_rows.data(), weights.data(), mapped_result.data(), 0,
                                 tokens, top_k, hidden);
    };

    flat_merge();
    mapped_merge();
    size_t mismatches = 0;
    for (size_t index = 0; index < flat_result.size(); ++index) {
      mismatches += flat_result[index] != mapped_result[index];
    }
    const auto [flat_measurement, mapped_measurement] = MeasureRotated(flat_merge, mapped_merge, warmup, runs);
    const double logical_bytes = static_cast<double>(tokens) * hidden * (top_k * sizeof(float) + sizeof(uint16_t));
    std::cout << std::setprecision(10) << "{\"tokens\":" << tokens << ",\"hidden\":" << hidden
              << ",\"experts\":" << experts << ",\"top_k\":" << top_k << ",\"routes\":" << routes
              << ",\"warmup\":" << warmup << ",\"runs\":" << runs << ",\"mismatches\":" << mismatches
              << ",\"flat\":{\"median_ms\":" << flat_measurement.median_ms
              << ",\"best_ms\":" << flat_measurement.best_ms
              << ",\"median_logical_gbps\":" << BandwidthGbPerSecond(logical_bytes, flat_measurement.median_ms)
              << "},\"mapped\":{\"median_ms\":" << mapped_measurement.median_ms
              << ",\"best_ms\":" << mapped_measurement.best_ms
              << ",\"median_logical_gbps\":" << BandwidthGbPerSecond(logical_bytes, mapped_measurement.median_ms)
              << "}}\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
  return 0;
}
