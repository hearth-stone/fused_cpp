// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <immintrin.h>
#include <oneapi/dnnl/dnnl.hpp>

namespace {

uint16_t Bf16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

__m512 LoadBf16(const uint16_t* pointer) {
  const __m256i values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pointer));
  const __m512i widened = _mm512_slli_epi32(_mm512_cvtepu16_epi32(values), 16);
  return _mm512_castsi512_ps(widened);
}

__m512 ExpNeg(__m512 gate) {
  const __m512 one = _mm512_set1_ps(1.0f);
  const __m512 x = _mm512_max_ps(_mm512_set1_ps(-87.0f),
                                 _mm512_min_ps(_mm512_set1_ps(87.0f), _mm512_sub_ps(_mm512_setzero_ps(), gate)));
  const __m512 fn = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.4426950408889634f)),
                                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  const __m512 r = _mm512_fnmadd_ps(fn, _mm512_set1_ps(0.6931471805599453f), x);
  __m512 poly = _mm512_fmadd_ps(_mm512_set1_ps(0.00833333f), r, _mm512_set1_ps(0.04166666f));
  poly = _mm512_fmadd_ps(poly, r, _mm512_set1_ps(0.16666666f));
  poly = _mm512_fmadd_ps(poly, r, _mm512_set1_ps(0.5f));
  poly = _mm512_fmadd_ps(poly, r, one);
  poly = _mm512_fmadd_ps(poly, r, one);
  __m512i exponent = _mm512_cvtps_epi32(fn);
  exponent = _mm512_slli_epi32(_mm512_add_epi32(exponent, _mm512_set1_epi32(127)), 23);
  return _mm512_mul_ps(poly, _mm512_castsi512_ps(exponent));
}

void SiluMul(const uint16_t* gate_up, uint16_t* intermediate, int64_t rows, int64_t features) {
  for (int64_t row = 0; row < rows; ++row) {
    const uint16_t* gate = gate_up + row * features * 2;
    const uint16_t* up = gate + features;
    uint16_t* output = intermediate + row * features;
    for (int64_t feature = 0; feature < features; feature += 16) {
      const __m512 gate_values = LoadBf16(gate + feature);
      const __m512 up_values = LoadBf16(up + feature);
      const __m512 denominator = _mm512_add_ps(ExpNeg(gate_values), _mm512_set1_ps(1.0f));
      const __m512 result = _mm512_div_ps(_mm512_mul_ps(gate_values, up_values), denominator);
      const __m256bh converted = _mm512_cvtneps_pbh(result);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + feature), reinterpret_cast<const __m256i&>(converted));
    }
  }
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

int ParseInt(char** argv, int index, int default_value) {
  return argv[index] == nullptr ? default_value : std::stoi(argv[index]);
}

}  // namespace

int main(int argc, char** argv) {
  const int m = argc > 1 ? ParseInt(argv, 1, 12) : 12;
  const int hidden = argc > 2 ? ParseInt(argv, 2, 4096) : 4096;
  const int features = argc > 3 ? ParseInt(argv, 3, 512) : 512;
  const int warmup = argc > 4 ? ParseInt(argv, 4, 5) : 5;
  const int runs = argc > 5 ? ParseInt(argv, 5, 21) : 21;
  if (m <= 0 || hidden <= 0 || features <= 0 || features % 16 != 0 || warmup < 0 || runs <= 0) {
    std::cerr << "usage: bench_moe_onednn_avx512 [M H F warmup runs], with F % 16 == 0\n";
    return 2;
  }

  std::mt19937 generator(20260718);
  std::normal_distribution<float> distribution(0.0f, 0.01f);
  std::vector<uint16_t> input(static_cast<size_t>(m) * hidden);
  std::vector<uint16_t> w13(static_cast<size_t>(2) * features * hidden);
  std::vector<uint16_t> w2(static_cast<size_t>(hidden) * features);
  for (uint16_t& value : input) {
    value = Bf16Bits(distribution(generator));
  }
  for (uint16_t& value : w13) {
    value = Bf16Bits(distribution(generator));
  }
  for (uint16_t& value : w2) {
    value = Bf16Bits(distribution(generator));
  }
  std::vector<uint16_t> gate_up(static_cast<size_t>(m) * features * 2);
  std::vector<uint16_t> intermediate(static_cast<size_t>(m) * features);
  std::vector<uint16_t> output(static_cast<size_t>(m) * hidden);

  const dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  dnnl::stream stream(engine);
  const auto bf16 = dnnl::memory::data_type::bf16;
  const auto ab = dnnl::memory::format_tag::ab;
  const auto ba = dnnl::memory::format_tag::ba;
  const auto any = dnnl::memory::format_tag::any;
  const auto input_md = dnnl::memory::desc({m, hidden}, bf16, ab);
  const auto w13_user_md = dnnl::memory::desc({hidden, 2 * features}, bf16, ba);
  const auto w13_any_md = dnnl::memory::desc({hidden, 2 * features}, bf16, any);
  const auto gate_up_md = dnnl::memory::desc({m, 2 * features}, bf16, ab);
  const auto intermediate_md = dnnl::memory::desc({m, features}, bf16, ab);
  const auto w2_user_md = dnnl::memory::desc({features, hidden}, bf16, ba);
  const auto w2_any_md = dnnl::memory::desc({features, hidden}, bf16, any);
  const auto output_md = dnnl::memory::desc({m, hidden}, bf16, ab);

  const auto input_memory = dnnl::memory(input_md, engine, input.data());
  auto w13_user_memory = dnnl::memory(w13_user_md, engine, w13.data());
  const auto gate_up_memory = dnnl::memory(gate_up_md, engine, gate_up.data());
  const auto intermediate_memory = dnnl::memory(intermediate_md, engine, intermediate.data());
  auto w2_user_memory = dnnl::memory(w2_user_md, engine, w2.data());
  const auto output_memory = dnnl::memory(output_md, engine, output.data());
  const auto w13_descriptor = dnnl::matmul::primitive_desc(engine, input_md, w13_any_md, gate_up_md);
  const auto w2_descriptor = dnnl::matmul::primitive_desc(engine, intermediate_md, w2_any_md, output_md);
  auto w13_memory = dnnl::memory(w13_descriptor.weights_desc(), engine);
  auto w2_memory = dnnl::memory(w2_descriptor.weights_desc(), engine);
  dnnl::reorder(w13_user_memory, w13_memory).execute(stream, w13_user_memory, w13_memory);
  dnnl::reorder(w2_user_memory, w2_memory).execute(stream, w2_user_memory, w2_memory);
  stream.wait();
  const dnnl::matmul w13_matmul(w13_descriptor);
  const dnnl::matmul w2_matmul(w2_descriptor);

  auto execute = [&]() {
    w13_matmul.execute(stream,
                       {{DNNL_ARG_SRC, input_memory}, {DNNL_ARG_WEIGHTS, w13_memory}, {DNNL_ARG_DST, gate_up_memory}});
    stream.wait();
    SiluMul(gate_up.data(), intermediate.data(), m, features);
    w2_matmul.execute(
        stream, {{DNNL_ARG_SRC, intermediate_memory}, {DNNL_ARG_WEIGHTS, w2_memory}, {DNNL_ARG_DST, output_memory}});
    stream.wait();
  };
  for (int iteration = 0; iteration < warmup; ++iteration) {
    execute();
  }
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(runs));
  for (int iteration = 0; iteration < runs; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    execute();
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }
  const double median_ms = Median(samples);
  const double best_ms = *std::min_element(samples.begin(), samples.end());
  const double flops = static_cast<double>(6) * m * hidden * features;
  std::cout << "{\"m\":" << m << ",\"hidden\":" << hidden << ",\"features\":" << features << ",\"warmup\":" << warmup
            << ",\"runs\":" << runs << ",\"w13_impl\":\"" << w13_descriptor.impl_info_str() << "\",\"w2_impl\":\""
            << w2_descriptor.impl_info_str() << "\",\"w13_packed_bytes\":" << w13_descriptor.weights_desc().get_size()
            << ",\"w2_packed_bytes\":" << w2_descriptor.weights_desc().get_size() << ",\"median_ms\":" << median_ms
            << ",\"best_ms\":" << best_ms << ",\"median_gflops\":" << flops / median_ms / 1e6
            << ",\"best_gflops\":" << flops / best_ms / 1e6 << "}\n";
  return 0;
}
