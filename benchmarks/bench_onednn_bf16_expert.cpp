// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <oneapi/dnnl/dnnl.hpp>

namespace {

using ArgumentMap = std::unordered_map<int, dnnl::memory>;

constexpr int64_t kMaximumReferenceMhf = 2'000'000;

uint16_t Bf16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

float Bf16Value(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

struct ErrorStats {
  double max_abs = 0.0;
  double root_mean_square = 0.0;
  double cosine_similarity = 1.0;
};

double ComparisonValue(uint16_t value) { return Bf16Value(value); }

template <typename Value>
double ComparisonValue(Value value) {
  return static_cast<double>(value);
}

template <typename ReferenceValue>
ErrorStats Compare(const std::vector<uint16_t>& actual, const std::vector<ReferenceValue>& reference) {
  double squared_error = 0.0;
  double actual_norm = 0.0;
  double reference_norm = 0.0;
  double dot = 0.0;
  ErrorStats stats;
  for (size_t index = 0; index < actual.size(); ++index) {
    const double actual_value = Bf16Value(actual[index]);
    const double reference_value = ComparisonValue(reference[index]);
    const double difference = actual_value - reference_value;
    stats.max_abs = std::max(stats.max_abs, std::abs(difference));
    squared_error += difference * difference;
    actual_norm += actual_value * actual_value;
    reference_norm += reference_value * reference_value;
    dot += actual_value * reference_value;
  }
  stats.root_mean_square = std::sqrt(squared_error / static_cast<double>(actual.size()));
  if (actual_norm > 0.0 && reference_norm > 0.0) {
    stats.cosine_similarity = dot / std::sqrt(actual_norm * reference_norm);
  }
  return stats;
}

dnnl::primitive_attr FusedGateAttributes(const dnnl::memory::desc& up_descriptor) {
  dnnl::post_ops operations;
  operations.append_eltwise(dnnl::algorithm::eltwise_swish, 1.0f, 0.0f);
  operations.append_binary(dnnl::algorithm::binary_mul, up_descriptor);
  dnnl::primitive_attr attributes;
  attributes.set_post_ops(operations);
  return attributes;
}

class OneDnnExpert {
 public:
  OneDnnExpert(int64_t rows, int64_t hidden, int64_t features, uint16_t* input, uint16_t* gate_weight,
               uint16_t* up_weight, uint16_t* down_weight)
      : basic_up_(rows * features),
        basic_gate_(rows * features),
        basic_activated_(rows * features),
        basic_intermediate_(rows * features),
        basic_output_(rows * hidden),
        fused_up_(rows * features),
        fused_intermediate_(rows * features),
        fused_output_(rows * hidden),
        engine_(dnnl::engine::kind::cpu, 0),
        stream_(engine_),
        bf16_(dnnl::memory::data_type::bf16),
        rows_hidden_descriptor_({rows, hidden}, bf16_, dnnl::memory::format_tag::ab),
        rows_features_descriptor_({rows, features}, bf16_, dnnl::memory::format_tag::ab),
        gate_up_user_descriptor_({hidden, features}, bf16_, dnnl::memory::format_tag::ba),
        gate_up_any_descriptor_({hidden, features}, bf16_, dnnl::memory::format_tag::any),
        down_user_descriptor_({features, hidden}, bf16_, dnnl::memory::format_tag::ba),
        down_any_descriptor_({features, hidden}, bf16_, dnnl::memory::format_tag::any),
        input_memory_(rows_hidden_descriptor_, engine_, input),
        gate_weight_user_memory_(gate_up_user_descriptor_, engine_, gate_weight),
        up_weight_user_memory_(gate_up_user_descriptor_, engine_, up_weight),
        down_weight_user_memory_(down_user_descriptor_, engine_, down_weight),
        basic_up_memory_(rows_features_descriptor_, engine_, basic_up_.data()),
        basic_gate_memory_(rows_features_descriptor_, engine_, basic_gate_.data()),
        basic_activated_memory_(rows_features_descriptor_, engine_, basic_activated_.data()),
        basic_intermediate_memory_(rows_features_descriptor_, engine_, basic_intermediate_.data()),
        basic_output_memory_(rows_hidden_descriptor_, engine_, basic_output_.data()),
        fused_up_memory_(rows_features_descriptor_, engine_, fused_up_.data()),
        fused_intermediate_memory_(rows_features_descriptor_, engine_, fused_intermediate_.data()),
        fused_output_memory_(rows_hidden_descriptor_, engine_, fused_output_.data()),
        up_descriptor_(engine_, rows_hidden_descriptor_, gate_up_any_descriptor_, rows_features_descriptor_),
        basic_gate_descriptor_(engine_, rows_hidden_descriptor_, gate_up_any_descriptor_, rows_features_descriptor_),
        fused_gate_descriptor_(engine_, rows_hidden_descriptor_, gate_up_any_descriptor_, rows_features_descriptor_,
                               FusedGateAttributes(rows_features_descriptor_)),
        down_descriptor_(engine_, rows_features_descriptor_, down_any_descriptor_, rows_hidden_descriptor_),
        swish_descriptor_(engine_, dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_swish,
                          rows_features_descriptor_, rows_features_descriptor_, 1.0f, 0.0f),
        multiply_descriptor_(engine_, dnnl::algorithm::binary_mul, rows_features_descriptor_, rows_features_descriptor_,
                             rows_features_descriptor_),
        up_weight_memory_(up_descriptor_.weights_desc(), engine_),
        basic_gate_weight_memory_(basic_gate_descriptor_.weights_desc(), engine_),
        fused_gate_weight_memory_(fused_gate_descriptor_.weights_desc(), engine_),
        down_weight_memory_(down_descriptor_.weights_desc(), engine_),
        up_primitive_(up_descriptor_),
        basic_gate_primitive_(basic_gate_descriptor_),
        fused_gate_primitive_(fused_gate_descriptor_),
        down_primitive_(down_descriptor_),
        swish_primitive_(swish_descriptor_),
        multiply_primitive_(multiply_descriptor_) {
    dnnl::reorder(up_weight_user_memory_, up_weight_memory_)
        .execute(stream_, up_weight_user_memory_, up_weight_memory_);
    dnnl::reorder(gate_weight_user_memory_, basic_gate_weight_memory_)
        .execute(stream_, gate_weight_user_memory_, basic_gate_weight_memory_);
    dnnl::reorder(gate_weight_user_memory_, fused_gate_weight_memory_)
        .execute(stream_, gate_weight_user_memory_, fused_gate_weight_memory_);
    dnnl::reorder(down_weight_user_memory_, down_weight_memory_)
        .execute(stream_, down_weight_user_memory_, down_weight_memory_);
    stream_.wait();

    basic_up_arguments_ = {
        {DNNL_ARG_SRC, input_memory_}, {DNNL_ARG_WEIGHTS, up_weight_memory_}, {DNNL_ARG_DST, basic_up_memory_}};
    basic_gate_arguments_ = {{DNNL_ARG_SRC, input_memory_},
                             {DNNL_ARG_WEIGHTS, basic_gate_weight_memory_},
                             {DNNL_ARG_DST, basic_gate_memory_}};
    basic_swish_arguments_ = {{DNNL_ARG_SRC, basic_gate_memory_}, {DNNL_ARG_DST, basic_activated_memory_}};
    basic_multiply_arguments_ = {{DNNL_ARG_SRC_0, basic_activated_memory_},
                                 {DNNL_ARG_SRC_1, basic_up_memory_},
                                 {DNNL_ARG_DST, basic_intermediate_memory_}};
    basic_down_arguments_ = {{DNNL_ARG_SRC, basic_intermediate_memory_},
                             {DNNL_ARG_WEIGHTS, down_weight_memory_},
                             {DNNL_ARG_DST, basic_output_memory_}};

    fused_up_arguments_ = {
        {DNNL_ARG_SRC, input_memory_}, {DNNL_ARG_WEIGHTS, up_weight_memory_}, {DNNL_ARG_DST, fused_up_memory_}};
    fused_gate_arguments_ = {{DNNL_ARG_SRC, input_memory_},
                             {DNNL_ARG_WEIGHTS, fused_gate_weight_memory_},
                             {DNNL_ARG_DST, fused_intermediate_memory_},
                             {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1, fused_up_memory_}};
    fused_down_arguments_ = {{DNNL_ARG_SRC, fused_intermediate_memory_},
                             {DNNL_ARG_WEIGHTS, down_weight_memory_},
                             {DNNL_ARG_DST, fused_output_memory_}};
  }

  void RunBasic() {
    up_primitive_.execute(stream_, basic_up_arguments_);
    basic_gate_primitive_.execute(stream_, basic_gate_arguments_);
    swish_primitive_.execute(stream_, basic_swish_arguments_);
    multiply_primitive_.execute(stream_, basic_multiply_arguments_);
    down_primitive_.execute(stream_, basic_down_arguments_);
    stream_.wait();
  }

  void RunFused() {
    up_primitive_.execute(stream_, fused_up_arguments_);
    fused_gate_primitive_.execute(stream_, fused_gate_arguments_);
    down_primitive_.execute(stream_, fused_down_arguments_);
    stream_.wait();
  }

  const std::vector<uint16_t>& BasicOutput() const { return basic_output_; }
  const std::vector<uint16_t>& FusedOutput() const { return fused_output_; }

  size_t BasicPackedWeightBytes() const {
    return up_weight_memory_.get_desc().get_size() + basic_gate_weight_memory_.get_desc().get_size() +
           down_weight_memory_.get_desc().get_size();
  }

  size_t FusedPackedWeightBytes() const {
    return up_weight_memory_.get_desc().get_size() + fused_gate_weight_memory_.get_desc().get_size() +
           down_weight_memory_.get_desc().get_size();
  }

  std::string UpImplementation() const { return up_descriptor_.impl_info_str(); }
  std::string BasicGateImplementation() const { return basic_gate_descriptor_.impl_info_str(); }
  std::string BasicSwishImplementation() const { return swish_descriptor_.impl_info_str(); }
  std::string BasicMultiplyImplementation() const { return multiply_descriptor_.impl_info_str(); }
  std::string FusedGateImplementation() const { return fused_gate_descriptor_.impl_info_str(); }
  std::string DownImplementation() const { return down_descriptor_.impl_info_str(); }

 private:
  std::vector<uint16_t> basic_up_;
  std::vector<uint16_t> basic_gate_;
  std::vector<uint16_t> basic_activated_;
  std::vector<uint16_t> basic_intermediate_;
  std::vector<uint16_t> basic_output_;
  std::vector<uint16_t> fused_up_;
  std::vector<uint16_t> fused_intermediate_;
  std::vector<uint16_t> fused_output_;
  dnnl::engine engine_;
  dnnl::stream stream_;
  dnnl::memory::data_type bf16_;
  dnnl::memory::desc rows_hidden_descriptor_;
  dnnl::memory::desc rows_features_descriptor_;
  dnnl::memory::desc gate_up_user_descriptor_;
  dnnl::memory::desc gate_up_any_descriptor_;
  dnnl::memory::desc down_user_descriptor_;
  dnnl::memory::desc down_any_descriptor_;
  dnnl::memory input_memory_;
  dnnl::memory gate_weight_user_memory_;
  dnnl::memory up_weight_user_memory_;
  dnnl::memory down_weight_user_memory_;
  dnnl::memory basic_up_memory_;
  dnnl::memory basic_gate_memory_;
  dnnl::memory basic_activated_memory_;
  dnnl::memory basic_intermediate_memory_;
  dnnl::memory basic_output_memory_;
  dnnl::memory fused_up_memory_;
  dnnl::memory fused_intermediate_memory_;
  dnnl::memory fused_output_memory_;
  dnnl::matmul::primitive_desc up_descriptor_;
  dnnl::matmul::primitive_desc basic_gate_descriptor_;
  dnnl::matmul::primitive_desc fused_gate_descriptor_;
  dnnl::matmul::primitive_desc down_descriptor_;
  dnnl::eltwise_forward::primitive_desc swish_descriptor_;
  dnnl::binary::primitive_desc multiply_descriptor_;
  dnnl::memory up_weight_memory_;
  dnnl::memory basic_gate_weight_memory_;
  dnnl::memory fused_gate_weight_memory_;
  dnnl::memory down_weight_memory_;
  dnnl::matmul up_primitive_;
  dnnl::matmul basic_gate_primitive_;
  dnnl::matmul fused_gate_primitive_;
  dnnl::matmul down_primitive_;
  dnnl::eltwise_forward swish_primitive_;
  dnnl::binary multiply_primitive_;
  ArgumentMap basic_up_arguments_;
  ArgumentMap basic_gate_arguments_;
  ArgumentMap basic_swish_arguments_;
  ArgumentMap basic_multiply_arguments_;
  ArgumentMap basic_down_arguments_;
  ArgumentMap fused_up_arguments_;
  ArgumentMap fused_gate_arguments_;
  ArgumentMap fused_down_arguments_;
};

std::vector<float> ReferenceExpert(int64_t rows, int64_t hidden, int64_t features, const std::vector<uint16_t>& input,
                                   const std::vector<uint16_t>& gate_weight, const std::vector<uint16_t>& up_weight,
                                   const std::vector<uint16_t>& down_weight) {
  std::vector<float> intermediate(rows * features);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t feature = 0; feature < features; ++feature) {
      float gate = 0.0f;
      float up = 0.0f;
      for (int64_t inner = 0; inner < hidden; ++inner) {
        const float input_value = Bf16Value(input[row * hidden + inner]);
        gate += input_value * Bf16Value(gate_weight[feature * hidden + inner]);
        up += input_value * Bf16Value(up_weight[feature * hidden + inner]);
      }
      intermediate[row * features + feature] = gate / (1.0f + std::exp(-gate)) * up;
    }
  }

  std::vector<float> output(rows * hidden);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < hidden; ++column) {
      float value = 0.0f;
      for (int64_t inner = 0; inner < features; ++inner) {
        value += intermediate[row * features + inner] * Bf16Value(down_weight[column * features + inner]);
      }
      output[row * hidden + column] = value;
    }
  }
  return output;
}

void PrintErrorStats(const ErrorStats& stats) {
  std::cout << "{\"max_abs\":" << stats.max_abs << ",\"root_mean_square\":" << stats.root_mean_square
            << ",\"cosine_similarity\":" << stats.cosine_similarity << "}";
}

int ParseInteger(char* argument) { return std::stoi(argument); }

}  // namespace

int main(int argc, char** argv) {
  const int64_t rows = argc > 1 ? ParseInteger(argv[1]) : 64;
  const int64_t hidden = argc > 2 ? ParseInteger(argv[2]) : 4096;
  const int64_t features = argc > 3 ? ParseInteger(argv[3]) : 512;
  const int warmup = argc > 4 ? ParseInteger(argv[4]) : 5;
  const int runs = argc > 5 ? ParseInteger(argv[5]) : 21;
  if (rows <= 0 || hidden <= 0 || features <= 0 || warmup < 0 || runs <= 0) {
    std::cerr << "usage: bench_onednn_bf16_expert [M H F warmup runs]\n";
    return 2;
  }

  std::mt19937 generator(20260720);
  std::normal_distribution<float> distribution(0.0f, 0.01f);
  std::vector<uint16_t> input(rows * hidden);
  std::vector<uint16_t> gate_weight(features * hidden);
  std::vector<uint16_t> up_weight(features * hidden);
  std::vector<uint16_t> down_weight(hidden * features);
  for (uint16_t& value : input) {
    value = Bf16Bits(distribution(generator));
  }
  for (uint16_t& value : gate_weight) {
    value = Bf16Bits(distribution(generator));
  }
  for (uint16_t& value : up_weight) {
    value = Bf16Bits(distribution(generator));
  }
  for (uint16_t& value : down_weight) {
    value = Bf16Bits(distribution(generator));
  }

  const auto prepack_start = std::chrono::steady_clock::now();
  OneDnnExpert expert(rows, hidden, features, input.data(), gate_weight.data(), up_weight.data(), down_weight.data());
  const auto prepack_end = std::chrono::steady_clock::now();
  const double prepack_ms = std::chrono::duration<double, std::milli>(prepack_end - prepack_start).count();

  expert.RunBasic();
  expert.RunFused();
  const ErrorStats fused_vs_basic = Compare(expert.FusedOutput(), expert.BasicOutput());

  const bool reference_computed = rows <= std::numeric_limits<int64_t>::max() / hidden &&
                                  rows * hidden <= std::numeric_limits<int64_t>::max() / features &&
                                  rows * hidden * features <= kMaximumReferenceMhf;
  ErrorStats basic_vs_reference;
  ErrorStats fused_vs_reference;
  if (reference_computed) {
    const std::vector<float> reference =
        ReferenceExpert(rows, hidden, features, input, gate_weight, up_weight, down_weight);
    basic_vs_reference = Compare(expert.BasicOutput(), reference);
    fused_vs_reference = Compare(expert.FusedOutput(), reference);
  }

  for (int iteration = 0; iteration < warmup; ++iteration) {
    expert.RunBasic();
    expert.RunFused();
  }

  std::vector<double> basic_samples;
  std::vector<double> fused_samples;
  basic_samples.reserve(runs);
  fused_samples.reserve(runs);
  auto measure = [](auto&& function) {
    const auto start = std::chrono::steady_clock::now();
    function();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
  };
  for (int iteration = 0; iteration < runs; ++iteration) {
    if ((iteration & 1) == 0) {
      basic_samples.push_back(measure([&]() { expert.RunBasic(); }));
      fused_samples.push_back(measure([&]() { expert.RunFused(); }));
    } else {
      fused_samples.push_back(measure([&]() { expert.RunFused(); }));
      basic_samples.push_back(measure([&]() { expert.RunBasic(); }));
    }
  }

  const double basic_median_ms = Median(basic_samples);
  const double fused_median_ms = Median(fused_samples);
  const double basic_best_ms = *std::min_element(basic_samples.begin(), basic_samples.end());
  const double fused_best_ms = *std::min_element(fused_samples.begin(), fused_samples.end());
  const double flops = 6.0 * static_cast<double>(rows) * static_cast<double>(hidden) * static_cast<double>(features);
  const dnnl_version_t* version = dnnl_version();

  std::cout << std::setprecision(10);
  std::cout << "{\"m\":" << rows << ",\"hidden\":" << hidden << ",\"features\":" << features
            << ",\"dtype\":\"bf16\",\"warmup\":" << warmup << ",\"runs\":" << runs << ",\"onednn_version\":\""
            << version->major << "." << version->minor << "." << version->patch
            << "\",\"prepack_and_create_ms\":" << prepack_ms << ",\"implementations\":{"
            << "\"up\":\"" << expert.UpImplementation() << "\",\"basic_gate\":\"" << expert.BasicGateImplementation()
            << "\",\"basic_swish\":\"" << expert.BasicSwishImplementation() << "\",\"basic_multiply\":\""
            << expert.BasicMultiplyImplementation() << "\",\"fused_gate_postops\":\""
            << expert.FusedGateImplementation() << "\",\"down\":\"" << expert.DownImplementation()
            << "\"},\"correctness\":{\"fused_vs_basic\":";
  PrintErrorStats(fused_vs_basic);
  std::cout << ",\"scalar_reference_computed\":" << (reference_computed ? "true" : "false") << ",\"basic_vs_scalar\":";
  if (reference_computed) {
    PrintErrorStats(basic_vs_reference);
  } else {
    std::cout << "null";
  }
  std::cout << ",\"fused_vs_scalar\":";
  if (reference_computed) {
    PrintErrorStats(fused_vs_reference);
  } else {
    std::cout << "null";
  }
  std::cout << "},\"basic\":{"
            << "\"primitive_count\":5,\"packed_weight_bytes\":" << expert.BasicPackedWeightBytes()
            << ",\"median_ms\":" << basic_median_ms << ",\"best_ms\":" << basic_best_ms
            << ",\"median_gflops\":" << flops / basic_median_ms / 1.0e6
            << ",\"best_gflops\":" << flops / basic_best_ms / 1.0e6 << "},\"postop_fused\":{"
            << "\"primitive_count\":3,\"packed_weight_bytes\":" << expert.FusedPackedWeightBytes()
            << ",\"median_ms\":" << fused_median_ms << ",\"best_ms\":" << fused_best_ms
            << ",\"median_gflops\":" << flops / fused_median_ms / 1.0e6
            << ",\"best_gflops\":" << flops / fused_best_ms / 1.0e6
            << "},\"fused_speedup_vs_basic\":" << basic_median_ms / fused_median_ms << "}\n";

  const bool scalar_reference_failed =
      reference_computed &&
      (!std::isfinite(basic_vs_reference.max_abs) || !std::isfinite(fused_vs_reference.max_abs) ||
       basic_vs_reference.cosine_similarity < 0.999 || fused_vs_reference.cosine_similarity < 0.999);
  if (!std::isfinite(fused_vs_basic.max_abs) || fused_vs_basic.cosine_similarity < 0.999 || scalar_reference_failed) {
    std::cerr << "oneDNN post-op fused output failed the basic-expert equivalence check\n";
    return 1;
  }
  return 0;
}
