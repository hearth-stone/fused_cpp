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
#include <string>
#include <utility>
#include <vector>

#include <oneapi/dnnl/dnnl.hpp>

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
    uint16_t* panel_destination = packed + static_cast<int64_t>(panel) * kPhysicalRowsPerPanel * k_pad;
    for (int kp = 0; kp < k_pad / 2; ++kp) {
      uint16_t* pair_destination = panel_destination + static_cast<int64_t>(kp) * kPhysicalRowsPerPanel * 2;
      for (int lane = 0; lane < kRowsPerPanel; ++lane) {
        const int row = panel * kRowsPerPanel + lane;
        if (row >= rows) {
          break;
        }
        for (int half = 0; half < 2; ++half) {
          const int k = kp * 2 + half;
          if (k < k_size) {
            pair_destination[lane * 2 + half] = source[static_cast<int64_t>(row) * k_size + k];
          }
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
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
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

    std::mt19937 generator(20260718);
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
    std::vector<float> custom_output(static_cast<size_t>(m) * n);
    std::vector<int64_t> route_ids(static_cast<size_t>(m));
    for (int row = 0; row < m; ++row) {
      route_ids[static_cast<size_t>(row)] = row;
    }
    PackA(input.data(), packed_a.data(), m, k, k_pad);
    avx512_moe::PackW2(weight.data(), packed_b.data(), n, k, k_pad, n_pad);

    auto custom_kernel = [&]() {
      avx512_moe::ComputeW2(packed_a.data(), k_pad, packed_b.data(), custom_output.data(), nullptr, route_ids.data(), n,
                            m, k_pad, n, 0, n_pad / 32, false);
    };
    auto custom_pack_a_and_kernel = [&]() {
      PackA(input.data(), packed_a.data(), m, k, k_pad);
      custom_kernel();
    };

    const dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream stream(engine);
    const auto bf16 = dnnl::memory::data_type::bf16;
    const auto f32 = dnnl::memory::data_type::f32;
    const auto ab = dnnl::memory::format_tag::ab;
    const auto ba = dnnl::memory::format_tag::ba;
    const auto any = dnnl::memory::format_tag::any;
    const auto input_md = dnnl::memory::desc({m, k}, bf16, ab);
    const auto weight_user_md = dnnl::memory::desc({k, n}, bf16, ba);
    const auto weight_any_md = dnnl::memory::desc({k, n}, bf16, any);
    const auto output_md = dnnl::memory::desc({m, n}, f32, ab);
    auto input_memory = dnnl::memory(input_md, engine, input.data());
    auto weight_user_memory = dnnl::memory(weight_user_md, engine, weight.data());
    auto output = std::vector<float>(static_cast<size_t>(m) * n);
    auto output_memory = dnnl::memory(output_md, engine, output.data());
    const auto descriptor = dnnl::matmul::primitive_desc(engine, input_md, weight_any_md, output_md);
    auto weight_memory = dnnl::memory(descriptor.weights_desc(), engine);
    dnnl::reorder(weight_user_memory, weight_memory).execute(stream, weight_user_memory, weight_memory);
    stream.wait();
    const dnnl::matmul matmul(descriptor);
    auto onednn_kernel = [&]() {
      matmul.execute(stream,
                     {{DNNL_ARG_SRC, input_memory}, {DNNL_ARG_WEIGHTS, weight_memory}, {DNNL_ARG_DST, output_memory}});
      stream.wait();
    };

    custom_kernel();
    onednn_kernel();
    float max_abs = 0.0f;
    double checksum = 0.0;
    for (size_t index = 0; index < output.size(); ++index) {
      max_abs = std::max(max_abs, std::abs(custom_output[index] - output[index]));
      checksum += custom_output[index];
    }

    const auto [custom_median, custom_best] = Measure(custom_kernel, warmup, runs);
    const auto [custom_pack_median, custom_pack_best] = Measure(custom_pack_a_and_kernel, warmup, runs);
    const auto [onednn_median, onednn_best] = Measure(onednn_kernel, warmup, runs);
    const double flops = 2.0 * m * k * n;
    const char* isa = std::getenv("ONEDNN_MAX_CPU_ISA");
    std::cout << std::setprecision(10) << "{\"m\":" << m << ",\"k\":" << k << ",\"n\":" << n << ",\"warmup\":" << warmup
              << ",\"runs\":" << runs << ",\"onednn_max_cpu_isa\":\"" << (isa == nullptr ? "" : isa)
              << "\",\"onednn_impl\":\"" << descriptor.impl_info_str() << "\",\"max_abs\":" << max_abs
              << ",\"checksum\":" << checksum << ",\"custom_kernel\":{\"median_ms\":" << custom_median
              << ",\"best_ms\":" << custom_best << ",\"median_gflops\":" << Gflops(flops, custom_median)
              << ",\"best_gflops\":" << Gflops(flops, custom_best)
              << "},\"custom_pack_a_kernel\":{\"median_ms\":" << custom_pack_median
              << ",\"best_ms\":" << custom_pack_best << ",\"median_gflops\":" << Gflops(flops, custom_pack_median)
              << ",\"best_gflops\":" << Gflops(flops, custom_pack_best)
              << "},\"onednn\":{\"median_ms\":" << onednn_median << ",\"best_ms\":" << onednn_best
              << ",\"median_gflops\":" << Gflops(flops, onednn_median)
              << ",\"best_gflops\":" << Gflops(flops, onednn_best) << "}}\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
  return 0;
}
