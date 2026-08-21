// SPDX-License-Identifier: Apache-2.0

#include <arm_sve.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include "gemm_params.h"
void i8gemm_k_hybrid(const int8_t* a, const int8_t* packed_b, int32_t* c, int8_t* unused,
                     const gemm_params_t* params);
void i8gemm_k_narrow(const int8_t* a, const int8_t* packed_b, int32_t* c, int8_t* unused,
                     const gemm_params_t* params);
void i8gemm_k_narrow2(const int8_t* a, const int8_t* packed_b, int32_t* c, int8_t* unused,
                      const gemm_params_t* params);
void i8gemm_k_nld(const int8_t* a, const int8_t* packed_b, int32_t* c, int8_t* packed_a,
                  const gemm_params_t* params);
void i8gemm_k_nld1(const int8_t* a, const int8_t* packed_b, int32_t* c, int8_t* packed_a,
                   const gemm_params_t* params);
void i8gemm_k_nld2(const int8_t* a, const int8_t* packed_b, int32_t* c, int8_t* packed_a,
                   const gemm_params_t* params);
void i8gemm_k_nld4(const int8_t* a, const int8_t* packed_b, int32_t* c, int8_t* packed_a,
                   const gemm_params_t* params);
void i8gemm_k_nld_m12(const int8_t* a, const int8_t* packed_b, int32_t* c, int8_t* packed_a,
                      const gemm_params_t* params);
void i8_pack_A_neon_m8_asm(const int8_t* a, int8_t* packed_a, int k, int lda);
}

namespace {

using Clock = std::chrono::steady_clock;

struct Config {
  std::string routes_path;
  std::string schedule_path;
  int tokens = 2048;
  int top_k = 6;
  int experts = 256;
  int hidden = 4096;
  int intermediate = 512;
  int threads = 80;
  int team_width = 8;
  int cpu_start = 240;
  int warmup = 7;
  int runs = 31;
  int w13_window_tiles = 0;
  int w2_window_tiles = 0;
  std::vector<std::pair<int, int>> window_pairs;
  float swiglu_limit = 10.0f;
  int seed = 20260821;
  bool check = false;
  bool check_kernels = false;
  bool packed_m12 = true;
};

struct TeamBarrier {
  explicit TeamBarrier(int participants) : participants(participants) {}

  void Wait() {
    const int observed = generation.load(std::memory_order_acquire);
    if (arrivals.fetch_add(1, std::memory_order_acq_rel) + 1 == participants) {
      arrivals.store(0, std::memory_order_relaxed);
      generation.fetch_add(1, std::memory_order_release);
      return;
    }
    while (generation.load(std::memory_order_acquire) == observed) {
      asm volatile("yield" ::: "memory");
    }
  }

  const int participants;
  std::atomic<int> arrivals{0};
  std::atomic<int> generation{0};
};

struct TeamWorkspace {
  TeamWorkspace(int team_width, int max_rows, int hidden, int intermediate)
      : barrier(team_width),
        input_q(static_cast<size_t>(max_rows) * hidden),
        w13_packed_a(static_cast<size_t>(max_rows + 7) * hidden),
        input_scale(max_rows),
        w13_acc(static_cast<size_t>(max_rows) * 2 * intermediate),
        intermediate_bf16(static_cast<size_t>(max_rows) * intermediate),
        intermediate_q(static_cast<size_t>(max_rows) * intermediate),
        w2_packed_a(static_cast<size_t>(max_rows + 7) * intermediate),
        intermediate_scale(max_rows),
        w2_acc(static_cast<size_t>(max_rows) * hidden) {}

  TeamBarrier barrier;
  std::vector<int8_t> input_q;
  std::vector<int8_t> w13_packed_a;
  std::vector<float> input_scale;
  std::vector<int32_t> w13_acc;
  std::vector<uint16_t> intermediate_bf16;
  std::vector<int8_t> intermediate_q;
  std::vector<int8_t> w2_packed_a;
  std::vector<float> intermediate_scale;
  std::vector<int32_t> w2_acc;
};

uint16_t FloatToBf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

float Bf16ToFloat(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

svfloat32_t LoadBf16(svbool_t pg, const uint16_t* source) {
  const svuint32_t values = svld1uh_u32(pg, source);
  return svreinterpret_f32_u32(svlsl_n_u32_x(pg, values, 16));
}

svuint32_t F32ToBf16Bits(svbool_t pg, svfloat32_t value) {
  const svuint32_t bits = svreinterpret_u32_f32(value);
  const svuint32_t lsb = svand_n_u32_x(pg, svlsr_n_u32_x(pg, bits, 16), 1);
  return svlsr_n_u32_x(pg, svadd_u32_x(pg, bits, svadd_n_u32_x(pg, lsb, 0x7fff)), 16);
}

void StoreBf16(svbool_t pg, uint16_t* destination, svfloat32_t value) {
  svst1h_u32(pg, destination, F32ToBf16Bits(pg, value));
}

int SelectM12Blocks(int rows) {
  constexpr int kM12Efficiency = 83;
  constexpr int kM8Efficiency = 70;
  const int maximum = rows / 12;
  int best_blocks = 0;
  int64_t best_cost = std::numeric_limits<int64_t>::max();
  for (int blocks = 0; blocks <= maximum; ++blocks) {
    const int tail = rows - blocks * 12;
    const int tail_blocks = (tail + 7) / 8;
    const int64_t cost = static_cast<int64_t>(blocks) * 12 * kM8Efficiency +
                         static_cast<int64_t>(tail_blocks) * 8 * kM12Efficiency;
    if (cost < best_cost) {
      best_cost = cost;
      best_blocks = blocks;
    }
  }
  return best_blocks;
}

void PackM12Block(const int8_t* source, int8_t* destination, int k) {
  size_t output = 0;
  for (int k_begin = 0; k_begin < k; k_begin += 8) {
    for (int row_pair = 0; row_pair < 6; ++row_pair) {
      const int row = row_pair * 2;
      std::memcpy(destination + output, source + static_cast<size_t>(row) * k + k_begin, 8);
      output += 8;
      std::memcpy(destination + output, source + static_cast<size_t>(row + 1) * k + k_begin, 8);
      output += 8;
    }
  }
}

void PackM8Block(const int8_t* source, int8_t* destination, int rows, int k) {
  if (rows == 8) {
    i8_pack_A_neon_m8_asm(source, destination, k, k);
    return;
  }
  size_t output = 0;
  for (int k_begin = 0; k_begin < k; k_begin += 8) {
    for (int row_pair = 0; row_pair < 4; ++row_pair) {
      for (int row_in_pair = 0; row_in_pair < 2; ++row_in_pair) {
        const int row = row_pair * 2 + row_in_pair;
        if (row < rows) {
          std::memcpy(destination + output, source + static_cast<size_t>(row) * k + k_begin, 8);
        } else {
          std::memset(destination + output, 0, 8);
        }
        output += 8;
      }
    }
  }
}

svfloat32_t ExpFexpaNeg(svbool_t pg, svfloat32_t gate) {
  svfloat32_t x = svneg_f32_x(pg, gate);
  x = svmin_n_f32_x(pg, x, 87.0f);
  x = svmax_n_f32_x(pg, x, -87.0f);
  svfloat32_t encoded = svmla_n_f32_x(pg, svdup_f32(196735.0f), x, 1.4426950216293335f);
  const svfloat32_t k = svsub_n_f32_x(pg, encoded, 196735.0f);
  svfloat32_t residual = svmls_n_f32_x(pg, x, k, 0.693145751953125f);
  residual = svmls_n_f32_x(pg, residual, k, 1.428606765330187e-06f);
  const svfloat32_t scale = svexpa_f32(svreinterpret_u32_f32(encoded));
  svfloat32_t polynomial =
      svmla_n_f32_x(pg, svdup_f32(1.000003695487976f), residual, 0.5000003576278687f);
  polynomial = svmul_f32_x(pg, polynomial, residual);
  return svmla_f32_x(pg, scale, scale, polynomial);
}

void QuantizeBf16Row(const uint16_t* source, int8_t* destination, int columns, float* scale_out) {
  const int64_t vl = static_cast<int64_t>(svcntw());
  svfloat32_t maximum = svdup_f32(0.0f);
  for (int64_t column = 0; column < columns; column += vl) {
    const svbool_t pg = svwhilelt_b32(column, static_cast<int64_t>(columns));
    maximum = svmax_f32_m(pg, maximum, svabs_f32_x(pg, LoadBf16(pg, source + column)));
  }
  const float max_value = svmaxv_f32(svptrue_b32(), maximum);
  const float scale = max_value > 0.0f ? max_value / 127.0f : 1.0f;
  const float inverse_scale = 1.0f / scale;
  *scale_out = scale;
  for (int64_t column = 0; column < columns; column += vl) {
    const svbool_t pg = svwhilelt_b32(column, static_cast<int64_t>(columns));
    svfloat32_t value = svmul_n_f32_x(pg, LoadBf16(pg, source + column), inverse_scale);
    value = svmax_n_f32_x(pg, svmin_n_f32_x(pg, value, 127.0f), -127.0f);
    svst1b_s32(pg, destination + column, svcvt_s32_f32_x(pg, svrinta_f32_x(pg, value)));
  }
}

void ComputeW13Epilogue(const int32_t* accumulator, const float* weight_scales, float activation_scale,
                        uint16_t* output, int intermediate, float swiglu_limit) {
  const int64_t vl = static_cast<int64_t>(svcntw());
  const svfloat32_t activation = svdup_f32(activation_scale);
  for (int64_t column = 0; column < intermediate; column += vl) {
    const svbool_t pg = svwhilelt_b32(column, static_cast<int64_t>(intermediate));
    svfloat32_t gate = svcvt_f32_s32_x(pg, svld1_s32(pg, accumulator + column));
    svfloat32_t up = svcvt_f32_s32_x(pg, svld1_s32(pg, accumulator + intermediate + column));
    gate = svmul_f32_x(pg, gate, svmul_f32_x(pg, activation, svld1_f32(pg, weight_scales + column)));
    up = svmul_f32_x(pg, up, svmul_f32_x(pg, activation, svld1_f32(pg, weight_scales + intermediate + column)));
    gate = svmin_n_f32_x(pg, gate, swiglu_limit);
    up = svmax_n_f32_x(pg, svmin_n_f32_x(pg, up, swiglu_limit), -swiglu_limit);
    const svfloat32_t numerator = svmul_f32_x(pg, gate, up);
    const svfloat32_t denominator = svadd_n_f32_x(pg, ExpFexpaNeg(pg, gate), 1.0f);
    StoreBf16(pg, output + column, svdiv_f32_x(pg, numerator, denominator));
  }
}

void StoreW2RouteRange(const int32_t* accumulator, const float* weight_scales, float activation_scale,
                       float* route_output, int column_begin, int column_end) {
  const int64_t vl = static_cast<int64_t>(svcntw());
  const svfloat32_t activation = svdup_f32(activation_scale);
  for (int64_t column = column_begin; column < column_end; column += vl) {
    const svbool_t pg = svwhilelt_b32(column, static_cast<int64_t>(column_end));
    svfloat32_t value = svcvt_f32_s32_x(pg, svld1_s32(pg, accumulator + column));
    value = svmul_f32_x(pg, value, svmul_f32_x(pg, activation, svld1_f32(pg, weight_scales + column)));
    svst1_f32(pg, route_output + column, value);
  }
}

std::vector<int32_t> ReadRoutes(const Config& config) {
  std::ifstream stream(config.routes_path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("failed to open routes file: " + config.routes_path);
  }
  std::vector<int32_t> routes(static_cast<size_t>(config.tokens) * config.top_k);
  stream.read(reinterpret_cast<char*>(routes.data()), static_cast<std::streamsize>(routes.size() * sizeof(int32_t)));
  if (!stream || stream.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error("routes file has the wrong size");
  }
  return routes;
}

std::vector<std::vector<int>> ReadSchedule(const Config& config) {
  std::ifstream stream(config.schedule_path);
  if (!stream) {
    throw std::runtime_error("failed to open schedule file: " + config.schedule_path);
  }
  std::vector<std::vector<int>> lanes;
  std::string line;
  while (std::getline(stream, line)) {
    std::vector<int> lane;
    std::stringstream parser(line);
    std::string value;
    while (std::getline(parser, value, ',')) {
      if (!value.empty()) {
        lane.push_back(std::stoi(value));
      }
    }
    lanes.push_back(std::move(lane));
  }
  if (static_cast<int>(lanes.size()) != config.threads / config.team_width) {
    throw std::runtime_error("schedule lane count does not match threads/team_width");
  }
  return lanes;
}

void BindCurrentThread(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
    throw std::runtime_error("failed to bind benchmark worker");
  }
}

Config ParseArgs(int argc, char** argv) {
  Config config;
  auto value = [&](int& index) {
    if (++index >= argc) {
      throw std::invalid_argument("missing argument value");
    }
    return std::string(argv[index]);
  };
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (option == "--routes") config.routes_path = value(index);
    else if (option == "--schedule") config.schedule_path = value(index);
    else if (option == "--tokens") config.tokens = std::stoi(value(index));
    else if (option == "--top-k") config.top_k = std::stoi(value(index));
    else if (option == "--experts") config.experts = std::stoi(value(index));
    else if (option == "--hidden") config.hidden = std::stoi(value(index));
    else if (option == "--intermediate") config.intermediate = std::stoi(value(index));
    else if (option == "--threads") config.threads = std::stoi(value(index));
    else if (option == "--team-width") config.team_width = std::stoi(value(index));
    else if (option == "--cpu-start") config.cpu_start = std::stoi(value(index));
    else if (option == "--warmup") config.warmup = std::stoi(value(index));
    else if (option == "--runs") config.runs = std::stoi(value(index));
    else if (option == "--w13-window") config.w13_window_tiles = std::stoi(value(index));
    else if (option == "--w2-window") config.w2_window_tiles = std::stoi(value(index));
    else if (option == "--window-pairs") {
      std::stringstream stream(value(index));
      for (std::string pair; std::getline(stream, pair, ',');) {
        const size_t separator = pair.find(':');
        if (separator == std::string::npos) {
          throw std::invalid_argument("--window-pairs entries must be W13:W2");
        }
        config.window_pairs.emplace_back(std::stoi(pair.substr(0, separator)),
                                         std::stoi(pair.substr(separator + 1)));
      }
    }
    else if (option == "--swiglu-limit") config.swiglu_limit = std::stof(value(index));
    else if (option == "--seed") config.seed = std::stoi(value(index));
    else if (option == "--check") config.check = true;
    else if (option == "--check-kernels") config.check_kernels = true;
    else if (option == "--gemm-kernel") {
      const std::string kernel = value(index);
      if (kernel == "packed_m12") config.packed_m12 = true;
      else if (kernel == "hybrid") config.packed_m12 = false;
      else throw std::invalid_argument("--gemm-kernel must be packed_m12 or hybrid");
    }
    else throw std::invalid_argument("unknown option: " + option);
  }
  if (config.routes_path.empty() || config.schedule_path.empty() || config.tokens <= 0 || config.top_k <= 0 ||
      config.experts <= 0 || config.hidden <= 0 || config.intermediate <= 0 || config.threads <= 0 ||
      config.team_width <= 0 || config.threads % config.team_width != 0 || config.warmup < 0 || config.runs <= 0 ||
      config.w13_window_tiles < 0 || config.w2_window_tiles < 0 ||
      config.hidden % 16 != 0 || config.intermediate % 16 != 0) {
    throw std::invalid_argument("invalid benchmark configuration");
  }
  for (const auto& [w13_window, w2_window] : config.window_pairs) {
    if (w13_window < 0 || w2_window < 0) {
      throw std::invalid_argument("window pairs must be non-negative");
    }
  }
  return config;
}

class W8A8Benchmark {
 public:
  explicit W8A8Benchmark(Config config)
      : config_(std::move(config)),
        routes_(ReadRoutes(config_)),
        lanes_(ReadSchedule(config_)),
        hidden_bf16_(static_cast<size_t>(config_.tokens) * config_.hidden),
        topk_weights_(routes_.size(), 1.0f / config_.top_k),
        w13_packed_(static_cast<size_t>(config_.experts) * config_.hidden * 2 * config_.intermediate),
        w2_packed_(static_cast<size_t>(config_.experts) * config_.intermediate * config_.hidden),
        w13_scales_(static_cast<size_t>(config_.experts) * 2 * config_.intermediate),
        w2_scales_(static_cast<size_t>(config_.experts) * config_.hidden),
        route_output_(static_cast<size_t>(routes_.size()) * config_.hidden),
        output_bf16_(static_cast<size_t>(config_.tokens) * config_.hidden) {
    Initialize();
  }

  double Run() {
    const auto begin = Clock::now();
#pragma omp parallel num_threads(config_.threads)
    {
      const int tid = omp_get_thread_num();
      BindCurrentThread(config_.cpu_start + tid);
      const int lane_index = tid / config_.team_width;
      const int local_tid = tid % config_.team_width;
      TeamWorkspace& workspace = *workspaces_[static_cast<size_t>(lane_index)];
      for (const int expert : lanes_[static_cast<size_t>(lane_index)]) {
        RunExpert(expert, local_tid, workspace);
      }
#pragma omp barrier
      MergeTokens(tid);
    }
    const auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
  }

  void SetWindows(int w13_window_tiles, int w2_window_tiles) {
    config_.w13_window_tiles = w13_window_tiles;
    config_.w2_window_tiles = w2_window_tiles;
  }

  float ValidateKernels() {
    const bool original_kernel = config_.packed_m12;
    config_.packed_m12 = false;
    Run();
    const std::vector<uint16_t> reference = output_bf16_;
    config_.packed_m12 = true;
    Run();
    float maximum = 0.0f;
    for (size_t index = 0; index < output_bf16_.size(); ++index) {
      maximum = std::max(maximum, std::abs(Bf16ToFloat(output_bf16_[index]) - Bf16ToFloat(reference[index])));
    }
    config_.packed_m12 = original_kernel;
    return maximum;
  }

  double Checksum() const {
    long double sum = 0.0;
    for (const uint16_t value : output_bf16_) {
      sum += Bf16ToFloat(value);
    }
    return static_cast<double>(sum);
  }

  size_t WeightBytes() const { return w13_packed_.size() + w2_packed_.size(); }

  size_t W13WindowBytes() const {
    return WindowBytes(config_.hidden, 2 * config_.intermediate, config_.w13_window_tiles);
  }

  size_t W2WindowBytes() const {
    return WindowBytes(config_.intermediate, config_.hidden, config_.w2_window_tiles);
  }

  float Validate() const {
    if (!config_.check) {
      return 0.0f;
    }
    float maximum = 0.0f;
    std::vector<int8_t> quantized(static_cast<size_t>(config_.hidden));
    for (int token = 0; token < config_.tokens; ++token) {
      const uint16_t* input = hidden_bf16_.data() + static_cast<size_t>(token) * config_.hidden;
      float input_max = 0.0f;
      for (int column = 0; column < config_.hidden; ++column) {
        input_max = std::max(input_max, std::abs(Bf16ToFloat(input[column])));
      }
      const float input_scale = input_max > 0.0f ? input_max / 127.0f : 1.0f;
      int32_t w13_sum = 0;
      for (int column = 0; column < config_.hidden; ++column) {
        const float scaled = std::max(-127.0f, std::min(127.0f, Bf16ToFloat(input[column]) / input_scale));
        quantized[static_cast<size_t>(column)] = static_cast<int8_t>(std::nearbyint(scaled));
        w13_sum += quantized[static_cast<size_t>(column)];
      }
      float gate = std::min(static_cast<float>(w13_sum) * input_scale * 0.001f, config_.swiglu_limit);
      float up = static_cast<float>(w13_sum) * input_scale * 0.001f;
      up = std::max(-config_.swiglu_limit, std::min(config_.swiglu_limit, up));
      const float intermediate = Bf16ToFloat(FloatToBf16(gate * up / (1.0f + std::exp(-gate))));
      const float intermediate_scale = std::abs(intermediate) > 0.0f ? std::abs(intermediate) / 127.0f : 1.0f;
      const int32_t intermediate_q = static_cast<int32_t>(std::nearbyint(intermediate / intermediate_scale));
      const float route_value =
          static_cast<float>(config_.intermediate * intermediate_q) * intermediate_scale * 0.001f;
      const float expected = Bf16ToFloat(FloatToBf16(route_value));
      for (int column = 0; column < config_.hidden; ++column) {
        const float actual = Bf16ToFloat(output_bf16_[static_cast<size_t>(token) * config_.hidden + column]);
        maximum = std::max(maximum, std::abs(actual - expected));
      }
    }
    return maximum;
  }

 private:
  size_t WindowBytes(int k, int n, int requested_tiles) const {
    const int n_tile = static_cast<int>(svcntb() / 2);
    const int owner_tiles = (n / n_tile + config_.team_width - 1) / config_.team_width;
    const int window_tiles = requested_tiles > 0 ? std::min(requested_tiles, owner_tiles) : owner_tiles;
    return static_cast<size_t>(window_tiles) * k * n_tile;
  }

  void Initialize() {
    std::mt19937 generator(config_.seed);
    std::uniform_real_distribution<float> activation(-0.2f, 0.2f);
    std::uniform_int_distribution<int> quantized_weight(-16, 16);
    for (uint16_t& value : hidden_bf16_) {
      value = FloatToBf16(activation(generator));
    }
    if (config_.check) {
      std::fill(w13_packed_.begin(), w13_packed_.end(), static_cast<int8_t>(1));
      std::fill(w2_packed_.begin(), w2_packed_.end(), static_cast<int8_t>(1));
      std::fill(w13_scales_.begin(), w13_scales_.end(), 0.001f);
      std::fill(w2_scales_.begin(), w2_scales_.end(), 0.001f);
    } else {
      for (int8_t& value : w13_packed_) {
        value = static_cast<int8_t>(quantized_weight(generator));
      }
      for (int8_t& value : w2_packed_) {
        value = static_cast<int8_t>(quantized_weight(generator));
      }
      for (size_t index = 0; index < w13_scales_.size(); ++index) {
        w13_scales_[index] = 0.001f + static_cast<float>(index % 17) * 0.00001f;
      }
      for (size_t index = 0; index < w2_scales_.size(); ++index) {
        w2_scales_[index] = 0.001f + static_cast<float>(index % 19) * 0.00001f;
      }
    }

    expert_routes_.resize(config_.experts);
    for (size_t flat = 0; flat < routes_.size(); ++flat) {
      const int expert = routes_[flat];
      if (expert < 0 || expert >= config_.experts) {
        throw std::runtime_error("route expert id is out of range");
      }
      expert_routes_[static_cast<size_t>(expert)].push_back(static_cast<int32_t>(flat));
    }
    std::vector<int> seen(config_.experts, 0);
    for (const auto& lane : lanes_) {
      int max_rows = 1;
      for (const int expert : lane) {
        if (expert < 0 || expert >= config_.experts || seen[expert]++) {
          throw std::runtime_error("schedule contains an invalid or duplicate expert");
        }
        max_rows = std::max(max_rows, static_cast<int>(expert_routes_[expert].size()));
      }
      workspaces_.push_back(
          std::make_unique<TeamWorkspace>(config_.team_width, max_rows, config_.hidden, config_.intermediate));
    }
    for (int expert = 0; expert < config_.experts; ++expert) {
      if (!expert_routes_[expert].empty() && seen[expert] != 1) {
        throw std::runtime_error("active expert is missing from schedule");
      }
    }
  }

  void RunExpert(int expert, int local_tid, TeamWorkspace& workspace) {
    const auto& route_ids = expert_routes_[static_cast<size_t>(expert)];
    const int rows = static_cast<int>(route_ids.size());
    const int row_begin = local_tid * rows / config_.team_width;
    const int row_end = (local_tid + 1) * rows / config_.team_width;
    for (int row = row_begin; row < row_end; ++row) {
      const int token = route_ids[static_cast<size_t>(row)] / config_.top_k;
      QuantizeBf16Row(hidden_bf16_.data() + static_cast<size_t>(token) * config_.hidden,
                      workspace.input_q.data() + static_cast<size_t>(row) * config_.hidden, config_.hidden,
                      &workspace.input_scale[static_cast<size_t>(row)]);
    }
    workspace.barrier.Wait();

    RunStageGemm(workspace.input_q.data(), workspace.w13_packed_a.data(),
                 w13_packed_.data() + static_cast<size_t>(expert) * config_.hidden * 2 * config_.intermediate,
                 workspace.w13_acc.data(), rows, config_.hidden, 2 * config_.intermediate, local_tid,
                 config_.w13_window_tiles, workspace.barrier);
    workspace.barrier.Wait();

    const float* w13_scale = w13_scales_.data() + static_cast<size_t>(expert) * 2 * config_.intermediate;
    for (int row = row_begin; row < row_end; ++row) {
      uint16_t* intermediate = workspace.intermediate_bf16.data() + static_cast<size_t>(row) * config_.intermediate;
      ComputeW13Epilogue(workspace.w13_acc.data() + static_cast<size_t>(row) * 2 * config_.intermediate, w13_scale,
                         workspace.input_scale[static_cast<size_t>(row)], intermediate, config_.intermediate,
                         config_.swiglu_limit);
      QuantizeBf16Row(intermediate, workspace.intermediate_q.data() + static_cast<size_t>(row) * config_.intermediate,
                      config_.intermediate, &workspace.intermediate_scale[static_cast<size_t>(row)]);
    }
    workspace.barrier.Wait();

    RunStageGemm(workspace.intermediate_q.data(), workspace.w2_packed_a.data(),
                 w2_packed_.data() + static_cast<size_t>(expert) * config_.intermediate * config_.hidden,
                 workspace.w2_acc.data(), rows, config_.intermediate, config_.hidden, local_tid,
                 config_.w2_window_tiles, workspace.barrier);
    workspace.barrier.Wait();

    const int column_begin = local_tid * config_.hidden / config_.team_width;
    const int column_end = (local_tid + 1) * config_.hidden / config_.team_width;
    const float* w2_scale = w2_scales_.data() + static_cast<size_t>(expert) * config_.hidden;
    for (int row = 0; row < rows; ++row) {
      float* destination =
          route_output_.data() + static_cast<size_t>(route_ids[static_cast<size_t>(row)]) * config_.hidden;
      StoreW2RouteRange(workspace.w2_acc.data() + static_cast<size_t>(row) * config_.hidden, w2_scale,
                        workspace.intermediate_scale[static_cast<size_t>(row)], destination, column_begin, column_end);
    }
    workspace.barrier.Wait();
  }

  bool UsePackedA(int rows, int k) const {
    if (!config_.packed_m12) {
      return false;
    }
    if (rows <= 4) {
      return false;
    }
    if (config_.team_width >= 2 && k <= 256 && rows <= 256) {
      return false;
    }
    if (config_.team_width >= 2 && rows <= 16 && k <= 1024) {
      return false;
    }
    return true;
  }

  void PreparePackedA(const int8_t* input, int8_t* packed_a, int rows, int k, int local_tid) const {
    const int m12_blocks = SelectM12Blocks(rows);
    const int m12_rows = m12_blocks * 12;
    const int tail_blocks = (rows - m12_rows + 7) / 8;
    const int total_blocks = m12_blocks + tail_blocks;
    for (int block = local_tid; block < total_blocks; block += config_.team_width) {
      if (block < m12_blocks) {
        const int row = block * 12;
        PackM12Block(input + static_cast<size_t>(row) * k, packed_a + static_cast<size_t>(row) * k, k);
      } else {
        const int tail_block = block - m12_blocks;
        const int row = m12_rows + tail_block * 8;
        const int block_rows = std::min(8, rows - row);
        PackM8Block(input + static_cast<size_t>(row) * k, packed_a + static_cast<size_t>(row) * k, block_rows, k);
      }
    }
  }

  void RunM8Kernel(const int8_t* input, const int8_t* packed_weight, int32_t* output, int8_t* packed_a,
                   int rows, const gemm_params_t& params) const {
    if (rows <= 1) {
      i8gemm_k_nld1(input, packed_weight, output, packed_a, &params);
    } else if (rows <= 2) {
      i8gemm_k_nld2(input, packed_weight, output, packed_a, &params);
    } else if (rows <= 4) {
      i8gemm_k_nld4(input, packed_weight, output, packed_a, &params);
    } else {
      i8gemm_k_nld(input, packed_weight, output, packed_a, &params);
    }
  }

  void RunDirectWindow(const int8_t* input, const int8_t* packed_weight, int32_t* output, int rows, int k, int n,
                       int n_begin, int columns) const {
    gemm_params_t params{rows, k, columns, k, k, n};
    auto kernel = i8gemm_k_hybrid;
    if (config_.packed_m12 && rows <= 2) {
      kernel = i8gemm_k_narrow;
    } else if (config_.packed_m12 && rows <= 4) {
      kernel = i8gemm_k_narrow2;
    }
    kernel(input, packed_weight + static_cast<size_t>(n_begin) * k, output + n_begin, nullptr, &params);
  }

  void RunPackedWindow(const int8_t* input, int8_t* packed_a, const int8_t* packed_weight, int32_t* output,
                       int rows, int k, int n, int n_begin, int columns) const {
    const int m12_blocks = SelectM12Blocks(rows);
    const int m12_rows = m12_blocks * 12;
    const int8_t* weight = packed_weight + static_cast<size_t>(n_begin) * k;
    for (int block = 0; block < m12_blocks; ++block) {
      const int row = block * 12;
      gemm_params_t params{12, k, columns, k, k, n};
      i8gemm_k_nld_m12(input + static_cast<size_t>(row) * k, weight,
                       output + static_cast<size_t>(row) * n + n_begin,
                       packed_a + static_cast<size_t>(row) * k, &params);
    }
    for (int row = m12_rows; row < rows; row += 8) {
      const int block_rows = std::min(8, rows - row);
      gemm_params_t params{block_rows, k, columns, k, k, n};
      RunM8Kernel(input + static_cast<size_t>(row) * k, weight,
                  output + static_cast<size_t>(row) * n + n_begin,
                  packed_a + static_cast<size_t>(row) * k, block_rows, params);
    }
  }

  void RunStageGemm(const int8_t* input, int8_t* packed_a, const int8_t* packed_weight, int32_t* output, int rows,
                    int k, int n, int local_tid, int window_tiles, TeamBarrier& barrier) const {
    const bool packed = UsePackedA(rows, k);
    if (packed) {
      PreparePackedA(input, packed_a, rows, k, local_tid);
      barrier.Wait();
    }
    const int n_tile = static_cast<int>(svcntb() / 2);
    const int tiles = n / n_tile;
    const int tile_begin = local_tid * tiles / config_.team_width;
    const int tile_end = (local_tid + 1) * tiles / config_.team_width;
    const int owner_tiles = tile_end - tile_begin;
    const int tiles_per_window = window_tiles > 0 ? std::min(window_tiles, owner_tiles) : owner_tiles;
    for (int begin = tile_begin; begin < tile_end; begin += tiles_per_window) {
      const int end = std::min(tile_end, begin + tiles_per_window);
      const int n_begin = begin * n_tile;
      const int columns = (end - begin) * n_tile;
      if (packed) {
        RunPackedWindow(input, packed_a, packed_weight, output, rows, k, n, n_begin, columns);
      } else {
        RunDirectWindow(input, packed_weight, output, rows, k, n, n_begin, columns);
      }
    }
  }

  void MergeTokens(int tid) {
    const int token_begin = tid * config_.tokens / config_.threads;
    const int token_end = (tid + 1) * config_.tokens / config_.threads;
    const int64_t vl = static_cast<int64_t>(svcntw());
    for (int token = token_begin; token < token_end; ++token) {
      for (int64_t column = 0; column < config_.hidden; column += vl) {
        const svbool_t pg = svwhilelt_b32(column, static_cast<int64_t>(config_.hidden));
        svfloat32_t accumulator = svdup_f32(0.0f);
        for (int slot = 0; slot < config_.top_k; ++slot) {
          const size_t flat = static_cast<size_t>(token) * config_.top_k + slot;
          const svfloat32_t value = svld1_f32(pg, route_output_.data() + flat * config_.hidden + column);
          accumulator = svmla_n_f32_x(pg, accumulator, value, topk_weights_[flat]);
        }
        StoreBf16(pg, output_bf16_.data() + static_cast<size_t>(token) * config_.hidden + column, accumulator);
      }
    }
  }

  Config config_;
  std::vector<int32_t> routes_;
  std::vector<std::vector<int>> lanes_;
  std::vector<std::vector<int32_t>> expert_routes_;
  std::vector<uint16_t> hidden_bf16_;
  std::vector<float> topk_weights_;
  std::vector<int8_t> w13_packed_;
  std::vector<int8_t> w2_packed_;
  std::vector<float> w13_scales_;
  std::vector<float> w2_scales_;
  std::vector<float> route_output_;
  std::vector<uint16_t> output_bf16_;
  std::vector<std::unique_ptr<TeamWorkspace>> workspaces_;
};

double Median(std::vector<double> values) {
  const size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle), values.end());
  return values[middle];
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Config config = ParseArgs(argc, argv);
    omp_set_dynamic(0);
    W8A8Benchmark benchmark(config);
    std::vector<std::pair<int, int>> window_pairs = config.window_pairs;
    if (window_pairs.empty()) {
      window_pairs.emplace_back(config.w13_window_tiles, config.w2_window_tiles);
    }
    std::vector<size_t> variant_order(window_pairs.size());
    std::iota(variant_order.begin(), variant_order.end(), size_t{0});
    std::mt19937 order_generator(config.seed ^ 0x51a7e123u);
    for (int iteration = 0; iteration < config.warmup; ++iteration) {
      std::shuffle(variant_order.begin(), variant_order.end(), order_generator);
      for (const size_t variant : variant_order) {
        benchmark.SetWindows(window_pairs[variant].first, window_pairs[variant].second);
        benchmark.Run();
      }
    }
    std::vector<std::vector<double>> samples(window_pairs.size());
    for (auto& variant_samples : samples) {
      variant_samples.reserve(static_cast<size_t>(config.runs));
    }
    for (int iteration = 0; iteration < config.runs; ++iteration) {
      std::shuffle(variant_order.begin(), variant_order.end(), order_generator);
      for (const size_t variant : variant_order) {
        benchmark.SetWindows(window_pairs[variant].first, window_pairs[variant].second);
        samples[variant].push_back(benchmark.Run());
      }
    }
    benchmark.SetWindows(window_pairs.front().first, window_pairs.front().second);
    const float kernel_max_abs = config.check_kernels ? benchmark.ValidateKernels() : 0.0f;
    for (size_t variant = 0; variant < window_pairs.size(); ++variant) {
      const auto [w13_window, w2_window] = window_pairs[variant];
      benchmark.SetWindows(w13_window, w2_window);
      std::cout << std::fixed << std::setprecision(6)
                << "w8a8_median_ms=" << Median(samples[variant]) << " checksum=" << benchmark.Checksum()
                << " max_abs=" << benchmark.Validate() << " weight_bytes=" << benchmark.WeightBytes()
                << " kernel_max_abs=" << kernel_max_abs
                << " threads=" << config.threads
                << " team_width=" << config.team_width
                << " gemm_kernel=" << (config.packed_m12 ? "packed_m12" : "hybrid")
                << " w13_window_tiles=" << w13_window
                << " w13_window_bytes=" << benchmark.W13WindowBytes()
                << " w2_window_tiles=" << w2_window
                << " w2_window_bytes=" << benchmark.W2WindowBytes() << '\n';
    }
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
}
