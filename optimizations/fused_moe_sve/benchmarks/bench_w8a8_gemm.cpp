// SPDX-License-Identifier: Apache-2.0

#include <arm_sve.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include "gemm_params.h"
void i8gemm_k_hybrid(const int8_t* a, const int8_t* packed_b, int32_t* c, int8_t* unused,
                     const gemm_params_t* params);
void i8gemm_mt_dispatch(const int8_t* a, const int8_t* packed_b, int32_t* c, int m, int k, int n, int threads);
}

namespace {

using Clock = std::chrono::steady_clock;

struct Config {
  int m = 918;
  int k = 4096;
  int n = 1024;
  int threads = 80;
  int cpu_start = 240;
  int warmup = 7;
  int runs = 31;
  int seed = 20260821;
};

uint16_t FloatToBf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
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

void QuantizeRow(const uint16_t* source, int8_t* destination, int columns, float* scale_out) {
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

void BindCurrentThread(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
    std::terminate();
  }
}

Config ParseArgs(int argc, char** argv) {
  Config config;
  auto value = [&](int& index) {
    if (++index >= argc) throw std::invalid_argument("missing argument value");
    return std::string(argv[index]);
  };
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (option == "--m") config.m = std::stoi(value(index));
    else if (option == "--k") config.k = std::stoi(value(index));
    else if (option == "--n") config.n = std::stoi(value(index));
    else if (option == "--threads") config.threads = std::stoi(value(index));
    else if (option == "--cpu-start") config.cpu_start = std::stoi(value(index));
    else if (option == "--warmup") config.warmup = std::stoi(value(index));
    else if (option == "--runs") config.runs = std::stoi(value(index));
    else if (option == "--seed") config.seed = std::stoi(value(index));
    else throw std::invalid_argument("unknown option: " + option);
  }
  const int n_tile = static_cast<int>(svcntb() / 2);
  if (config.m <= 0 || config.k <= 0 || config.n <= 0 || config.threads <= 0 || config.warmup < 0 ||
      config.runs <= 0 || config.k % 16 || config.n % n_tile) {
    throw std::invalid_argument("invalid benchmark configuration");
  }
  return config;
}

class Benchmark {
 public:
  explicit Benchmark(Config config)
      : config_(config),
        input_bf16_(static_cast<size_t>(config.m) * config.k),
        input_q_(static_cast<size_t>(config.m) * config.k),
        input_scales_(config.m),
        packed_weight_(static_cast<size_t>(config.k) * config.n),
        weight_scales_(config.n),
        accumulator_(static_cast<size_t>(config.m) * config.n),
        output_bf16_(static_cast<size_t>(config.m) * config.n) {
    std::mt19937 generator(config.seed);
    std::uniform_real_distribution<float> input(-0.2f, 0.2f);
    std::uniform_int_distribution<int> weight(-16, 16);
    for (uint16_t& value : input_bf16_) value = FloatToBf16(input(generator));
    for (int8_t& value : packed_weight_) value = static_cast<int8_t>(weight(generator));
    for (int column = 0; column < config.n; ++column) {
      weight_scales_[static_cast<size_t>(column)] = 0.001f + static_cast<float>(column % 19) * 0.00001f;
    }
    QuantizeAllSerial();
  }

  double RunRaw() {
    const auto begin = Clock::now();
#pragma omp parallel num_threads(config_.threads)
    {
      const int tid = omp_get_thread_num();
      BindCurrentThread(config_.cpu_start + tid);
      RunNRange(tid);
    }
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
  }

  double RunDispatch() {
    const auto begin = Clock::now();
    i8gemm_mt_dispatch(input_q_.data(), packed_weight_.data(), accumulator_.data(), config_.m, config_.k, config_.n,
                       config_.threads);
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
  }

  double RunBf16Io() {
    const auto begin = Clock::now();
#pragma omp parallel num_threads(config_.threads)
    {
      const int tid = omp_get_thread_num();
      BindCurrentThread(config_.cpu_start + tid);
      const int row_begin = tid * config_.m / config_.threads;
      const int row_end = (tid + 1) * config_.m / config_.threads;
      for (int row = row_begin; row < row_end; ++row) {
        QuantizeRow(input_bf16_.data() + static_cast<size_t>(row) * config_.k,
                    input_q_.data() + static_cast<size_t>(row) * config_.k, config_.k,
                    &input_scales_[static_cast<size_t>(row)]);
      }
#pragma omp barrier
      RunNRange(tid);
#pragma omp barrier
      DequantizeRows(row_begin, row_end);
    }
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
  }

  void ValidateDispatch() {
    std::fill(accumulator_.begin(), accumulator_.end(), 0);
    RunRaw();
    const std::vector<int32_t> reference = accumulator_;
    std::fill(accumulator_.begin(), accumulator_.end(), 0);
    RunDispatch();
    for (size_t index = 0; index < accumulator_.size(); ++index) {
      if (accumulator_[index] != reference[index]) {
        throw std::runtime_error("dispatch mismatch at index " + std::to_string(index) + ": expected " +
                                 std::to_string(reference[index]) + ", got " +
                                 std::to_string(accumulator_[index]));
      }
    }
  }

  double Checksum() const {
    uint64_t sum = 0;
    for (const uint16_t value : output_bf16_) sum += value;
    return static_cast<double>(sum);
  }

 private:
  void QuantizeAllSerial() {
    for (int row = 0; row < config_.m; ++row) {
      QuantizeRow(input_bf16_.data() + static_cast<size_t>(row) * config_.k,
                  input_q_.data() + static_cast<size_t>(row) * config_.k, config_.k,
                  &input_scales_[static_cast<size_t>(row)]);
    }
  }

  void RunNRange(int tid) {
    const int n_tile = static_cast<int>(svcntb() / 2);
    const int tiles = config_.n / n_tile;
    const int tile_begin = tid * tiles / config_.threads;
    const int tile_end = (tid + 1) * tiles / config_.threads;
    const int n_begin = tile_begin * n_tile;
    const int columns = (tile_end - tile_begin) * n_tile;
    if (columns == 0) return;
    gemm_params_t params{config_.m, config_.k, columns, config_.k, config_.k, config_.n};
    i8gemm_k_hybrid(input_q_.data(), packed_weight_.data() + static_cast<size_t>(n_begin) * config_.k,
                    accumulator_.data() + n_begin, nullptr, &params);
  }

  void DequantizeRows(int row_begin, int row_end) {
    const int64_t vl = static_cast<int64_t>(svcntw());
    for (int row = row_begin; row < row_end; ++row) {
      const svfloat32_t activation_scale = svdup_f32(input_scales_[static_cast<size_t>(row)]);
      for (int64_t column = 0; column < config_.n; column += vl) {
        const svbool_t pg = svwhilelt_b32(column, static_cast<int64_t>(config_.n));
        svfloat32_t value =
            svcvt_f32_s32_x(pg, svld1_s32(pg, accumulator_.data() + static_cast<size_t>(row) * config_.n + column));
        value = svmul_f32_x(
            pg, value,
            svmul_f32_x(pg, activation_scale, svld1_f32(pg, weight_scales_.data() + column)));
        svst1h_u32(pg, output_bf16_.data() + static_cast<size_t>(row) * config_.n + column, F32ToBf16Bits(pg, value));
      }
    }
  }

  Config config_;
  std::vector<uint16_t> input_bf16_;
  std::vector<int8_t> input_q_;
  std::vector<float> input_scales_;
  std::vector<int8_t> packed_weight_;
  std::vector<float> weight_scales_;
  std::vector<int32_t> accumulator_;
  std::vector<uint16_t> output_bf16_;
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
    Benchmark benchmark(config);
    benchmark.ValidateDispatch();
    for (int iteration = 0; iteration < config.warmup; ++iteration) {
      benchmark.RunRaw();
      benchmark.RunDispatch();
      benchmark.RunBf16Io();
    }
    std::vector<double> raw_samples;
    std::vector<double> dispatch_samples;
    std::vector<double> bf16_samples;
    for (int iteration = 0; iteration < config.runs; ++iteration) {
      if (iteration % 3 == 0) {
        raw_samples.push_back(benchmark.RunRaw());
        dispatch_samples.push_back(benchmark.RunDispatch());
        bf16_samples.push_back(benchmark.RunBf16Io());
      } else if (iteration % 3 == 1) {
        dispatch_samples.push_back(benchmark.RunDispatch());
        bf16_samples.push_back(benchmark.RunBf16Io());
        raw_samples.push_back(benchmark.RunRaw());
      } else {
        bf16_samples.push_back(benchmark.RunBf16Io());
        raw_samples.push_back(benchmark.RunRaw());
        dispatch_samples.push_back(benchmark.RunDispatch());
      }
    }
    const double raw_ms = Median(raw_samples);
    const double dispatch_ms = Median(dispatch_samples);
    const double bf16_ms = Median(bf16_samples);
    const double flops = 2.0 * config.m * config.k * config.n;
    std::cout << std::fixed << std::setprecision(6) << "M=" << config.m << " K=" << config.k << " N=" << config.n
              << " hybrid_ms=" << raw_ms << " hybrid_tops=" << flops / raw_ms / 1.0e9
              << " dispatch_ms=" << dispatch_ms << " dispatch_tops=" << flops / dispatch_ms / 1.0e9
              << " bf16_io_ms=" << bf16_ms << " bf16_io_tflops=" << flops / bf16_ms / 1.0e9
              << " checksum=" << benchmark.Checksum() << '\n';
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
}
