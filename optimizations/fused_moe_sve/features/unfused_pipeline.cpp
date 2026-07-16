#include "unfused_pipeline.h"

#include <algorithm>
#include <arm_sve.h>
#include <atomic>
#include <barrier>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <pthread.h>
#include <sched.h>

#include "gemm_params.h"

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE) || !defined(__ARM_FEATURE_BF16)
#error "unfused_pipeline requires AArch64 SVE BF16"
#endif

extern "C" {
void moe_sve_w13_silu_poly5_packc_m12_rows_opt(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                               const gemm_params_t*);
void moe_sve_w2_packed_m12(const uint16_t*, const uint16_t*, float*, uint16_t*, const gemm_params_t*);
void moe_sve_w2_packed_bf16_m12(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
}

namespace fused_moe_sve::unfused {
namespace {

using Clock = std::chrono::steady_clock;

struct Range {
  int64_t begin = 0;
  int64_t size = 0;
};

Range split_evenly(int64_t items, int lanes, int lane) {
  const int64_t base = items / lanes;
  const int64_t extra = items % lanes;
  if (lane < extra) {
    return {lane * (base + 1), base + 1};
  }
  return {extra * (base + 1) + static_cast<int64_t>(lane - extra) * base, base};
}

template <typename T>
class AlignedBuffer {
 public:
  AlignedBuffer() = default;

  explicit AlignedBuffer(size_t size) : size_(size) {
    void* allocation = nullptr;
    if (size > std::numeric_limits<size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }
    const int error = posix_memalign(&allocation, 64, size * sizeof(T));
    if (error != 0) {
      throw std::runtime_error("posix_memalign failed: " + std::to_string(error));
    }
    data_ = static_cast<T*>(allocation);
  }

  ~AlignedBuffer() { std::free(data_); }

  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  AlignedBuffer(AlignedBuffer&& other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
      std::free(data_);
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  T* data() { return data_; }
  const T* data() const { return data_; }
  size_t size() const { return size_; }
  uint64_t bytes() const { return static_cast<uint64_t>(size_) * sizeof(T); }

 private:
  T* data_ = nullptr;
  size_t size_ = 0;
};

uint16_t float_to_bf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

float bf16_to_float(uint16_t value) { return std::bit_cast<float>(static_cast<uint32_t>(value) << 16); }

uint32_t mix32(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value;
}

const uint16_t* value_table(float scale) {
  static uint16_t input_values[32];
  static uint16_t weight_values[32];
  static std::once_flag input_once;
  static std::once_flag weight_once;
  if (scale == 1.0f / 64.0f) {
    std::call_once(input_once, [&]() {
      for (int code = 0; code < 32; ++code) {
        const int sample = code < 16 ? code - 16 : code - 15;
        input_values[code] = float_to_bf16(static_cast<float>(sample) * scale);
      }
    });
    return input_values;
  }
  std::call_once(weight_once, [&]() {
    for (int code = 0; code < 32; ++code) {
      const int sample = code < 16 ? code - 16 : code - 15;
      weight_values[code] = float_to_bf16(static_cast<float>(sample) * scale);
    }
  });
  return weight_values;
}

uint16_t input_value(int copy, int expert, int route, int column) {
  const uint32_t key = static_cast<uint32_t>(column) * 0x9e3779b9u ^ static_cast<uint32_t>(route) * 0x85ebca6bu ^
                       static_cast<uint32_t>(expert) * 0xc2b2ae35u ^ static_cast<uint32_t>(copy) * 0x27d4eb2fu;
  return value_table(1.0f / 64.0f)[mix32(key) & 31u];
}

uint16_t weight_value(int copy, int expert, int kind, int k, int n) {
  const uint32_t key = static_cast<uint32_t>(k) * 0x9e3779b9u ^ static_cast<uint32_t>(n) * 0x85ebca6bu ^
                       static_cast<uint32_t>(expert) * 0xc2b2ae35u ^ static_cast<uint32_t>(copy) * 0x27d4eb2fu ^
                       static_cast<uint32_t>(kind + 1) * 0x165667b1u;
  return value_table(1.0f / 1024.0f)[mix32(key) & 31u];
}

void fill_source(AlignedBuffer<uint16_t>& source, const Config& config, int copy) {
  for (int route = 0; route < config.routes; ++route) {
    for (int expert = 0; expert < config.experts; ++expert) {
      uint16_t* row =
          source.data() + (static_cast<int64_t>(route) * config.experts + expert) * static_cast<int64_t>(config.hidden);
      for (int column = 0; column < config.hidden; ++column) {
        row[column] = input_value(copy, expert, route, column);
      }
    }
  }
}

template <typename LogicalValue>
void fill_packed_b(uint16_t* packed, int k_size, int n_size, LogicalValue logical_value) {
  const int segments = static_cast<int>(svcntb() / 16);
  const int n_tile = segments * 8;
  int64_t index = 0;
  for (int nb = 0; nb < n_size; nb += n_tile) {
    for (int rb = 0; rb < k_size / 4; ++rb) {
      const int row_base = rb * 4;
      for (int column_pair = 0; column_pair < 4; ++column_pair) {
        for (int segment = 0; segment < segments; ++segment) {
          const int column = nb + segment * 8 + column_pair * 2;
          for (int k = 0; k < 4; ++k) {
            packed[index++] = logical_value(row_base + k, column);
          }
          for (int k = 0; k < 4; ++k) {
            packed[index++] = logical_value(row_base + k, column + 1);
          }
        }
      }
    }
  }
  if (index != static_cast<int64_t>(k_size) * n_size) {
    throw std::runtime_error("packed-B initialization produced the wrong element count");
  }
}

struct PackedWeights {
  AlignedBuffer<uint16_t> w1;
  AlignedBuffer<uint16_t> w3;
  AlignedBuffer<uint16_t> w13;
  AlignedBuffer<uint16_t> w2;

  uint64_t bytes() const { return w1.bytes() + w3.bytes() + w13.bytes() + w2.bytes(); }
};

struct DataCopy {
  AlignedBuffer<uint16_t> source;
  PackedWeights weights;

  uint64_t bytes() const { return source.bytes() + weights.bytes(); }
};

PackedWeights make_weights(const Config& config, int copy) {
  const size_t hf = static_cast<size_t>(config.hidden) * config.intermediate;
  PackedWeights weights{AlignedBuffer<uint16_t>(static_cast<size_t>(config.experts) * hf),
                        AlignedBuffer<uint16_t>(static_cast<size_t>(config.experts) * hf),
                        AlignedBuffer<uint16_t>(static_cast<size_t>(config.experts) * hf * 2),
                        AlignedBuffer<uint16_t>(static_cast<size_t>(config.experts) * hf)};
  for (int expert = 0; expert < config.experts; ++expert) {
    uint16_t* w1 = weights.w1.data() + static_cast<size_t>(expert) * hf;
    uint16_t* w3 = weights.w3.data() + static_cast<size_t>(expert) * hf;
    uint16_t* w13 = weights.w13.data() + static_cast<size_t>(expert) * hf * 2;
    uint16_t* w2 = weights.w2.data() + static_cast<size_t>(expert) * hf;
    fill_packed_b(w1, config.hidden, config.intermediate,
                  [&](int k, int n) { return weight_value(copy, expert, 0, k, n); });
    fill_packed_b(w3, config.hidden, config.intermediate,
                  [&](int k, int n) { return weight_value(copy, expert, 1, k, n); });
    fill_packed_b(w13, config.hidden, config.intermediate * 2, [&](int k, int packed_column) {
      const int block = packed_column / 8;
      const int offset = packed_column % 8;
      const int kind = offset < 4 ? 0 : 1;
      const int feature = block * 4 + (offset & 3);
      return weight_value(copy, expert, kind, k, feature);
    });
    fill_packed_b(w2, config.intermediate, config.hidden,
                  [&](int k, int n) { return weight_value(copy, expert, 2, k, n); });
  }
  return weights;
}

struct Scratch {
  AlignedBuffer<uint16_t> extracted;
  AlignedBuffer<uint16_t> packed_input;
  AlignedBuffer<float> gate;
  AlignedBuffer<float> silu;
  AlignedBuffer<float> up;
  AlignedBuffer<uint16_t> product;
  AlignedBuffer<uint16_t> packed_product;
  AlignedBuffer<uint16_t> fused_intermediate;
  AlignedBuffer<uint16_t> unfused_output;
  AlignedBuffer<uint16_t> fused_output;

  uint64_t bytes() const {
    return extracted.bytes() + packed_input.bytes() + gate.bytes() + silu.bytes() + up.bytes() + product.bytes() +
           packed_product.bytes() + fused_intermediate.bytes() + unfused_output.bytes() + fused_output.bytes();
  }
};

Scratch make_scratch(const Config& config) {
  const size_t input = static_cast<size_t>(config.experts) * config.routes * config.hidden;
  const size_t intermediate = static_cast<size_t>(config.experts) * config.routes * config.intermediate;
  return {AlignedBuffer<uint16_t>(input),        AlignedBuffer<uint16_t>(input),
          AlignedBuffer<float>(intermediate),    AlignedBuffer<float>(intermediate),
          AlignedBuffer<float>(intermediate),    AlignedBuffer<uint16_t>(intermediate),
          AlignedBuffer<uint16_t>(intermediate), AlignedBuffer<uint16_t>(intermediate),
          AlignedBuffer<uint16_t>(input),        AlignedBuffer<uint16_t>(input)};
}

int bind_current_thread(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  return pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

class WorkerPool {
 public:
  WorkerPool(int workers, int cpu_start) : workers_(workers), cpu_start_(cpu_start) {
    threads_.reserve(static_cast<size_t>(workers));
    for (int tid = 0; tid < workers; ++tid) {
      threads_.emplace_back([this, tid]() { worker_loop(tid); });
    }
    std::unique_lock<std::mutex> lock(mutex_);
    ready_cv_.wait(lock, [&]() { return ready_workers_ == workers_; });
    if (startup_error_ != 0) {
      const int error = startup_error_;
      lock.unlock();
      shutdown();
      throw std::runtime_error("pthread_setaffinity_np failed: " + std::to_string(error));
    }
  }

  ~WorkerPool() { shutdown(); }

  WorkerPool(const WorkerPool&) = delete;
  WorkerPool& operator=(const WorkerPool&) = delete;

  void run(const std::function<void(int)>& function) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      current_job_ = &function;
      remaining_workers_ = workers_;
      worker_exception_ = nullptr;
      ++generation_;
    }
    start_cv_.notify_all();

    std::exception_ptr error;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      done_cv_.wait(lock, [&]() { return remaining_workers_ == 0; });
      error = worker_exception_;
      current_job_ = nullptr;
    }
    if (error != nullptr) {
      std::rethrow_exception(error);
    }
  }

 private:
  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_) {
        return;
      }
      stopping_ = true;
      ++generation_;
    }
    start_cv_.notify_all();
    for (std::thread& thread : threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

  void worker_loop(int tid) {
    const int bind_error = bind_current_thread(cpu_start_ + tid);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (bind_error != 0 && startup_error_ == 0) {
        startup_error_ = bind_error;
      }
      ++ready_workers_;
    }
    ready_cv_.notify_one();

    uint64_t seen_generation = 0;
    while (true) {
      const std::function<void(int)>* job = nullptr;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        start_cv_.wait(lock, [&]() { return stopping_ || generation_ != seen_generation; });
        if (stopping_) {
          return;
        }
        seen_generation = generation_;
        job = current_job_;
      }
      try {
        (*job)(tid);
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (worker_exception_ == nullptr) {
          worker_exception_ = std::current_exception();
        }
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        --remaining_workers_;
        if (remaining_workers_ == 0) {
          done_cv_.notify_one();
        }
      }
    }
  }

  int workers_ = 0;
  int cpu_start_ = 0;
  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::condition_variable ready_cv_;
  std::condition_variable start_cv_;
  std::condition_variable done_cv_;
  const std::function<void(int)>* current_job_ = nullptr;
  std::exception_ptr worker_exception_;
  uint64_t generation_ = 0;
  int ready_workers_ = 0;
  int remaining_workers_ = 0;
  int startup_error_ = 0;
  bool stopping_ = false;
};

svfloat32_t exp_poly5_neg(svbool_t predicate, svfloat32_t gate) {
  svfloat32_t x = svneg_f32_x(predicate, gate);
  x = svmin_n_f32_x(predicate, x, 87.0f);
  x = svmax_n_f32_x(predicate, x, -87.0f);
  svfloat32_t rounded = svmul_n_f32_x(predicate, x, 1.4426950408889634f);
  rounded = svrintn_f32_x(predicate, rounded);
  const svint32_t exponent = svcvt_s32_f32_x(predicate, rounded);
  const svfloat32_t reduced = svmls_n_f32_x(predicate, x, rounded, 0.6931471805599453f);
  svfloat32_t polynomial = svmla_n_f32_x(predicate, svdup_f32(0.04166666f), reduced, 0.00833333f);
  polynomial = svmla_f32_x(predicate, svdup_f32(0.16666666f), polynomial, reduced);
  polynomial = svmla_f32_x(predicate, svdup_f32(0.5f), polynomial, reduced);
  polynomial = svmla_f32_x(predicate, svdup_f32(1.0f), polynomial, reduced);
  polynomial = svmla_f32_x(predicate, svdup_f32(1.0f), polynomial, reduced);
  const svint32_t exponent_bits = svlsl_n_s32_x(predicate, svadd_n_s32_x(predicate, exponent, 127), 23);
  return svmul_f32_x(predicate, polynomial, svreinterpret_f32_s32(exponent_bits));
}

svuint32_t f32_to_bf16_bits(svbool_t predicate, svfloat32_t value) {
  const svuint32_t bits = svreinterpret_u32_f32(value);
  const svuint32_t lsb = svand_n_u32_x(predicate, svlsr_n_u32_x(predicate, bits, 16), 1);
  const svuint32_t bias = svadd_n_u32_x(predicate, lsb, 0x7fff);
  return svlsr_n_u32_x(predicate, svadd_u32_x(predicate, bits, bias), 16);
}

void extract_rows(const uint16_t* source, uint16_t* destination, const Config& config, int expert, int lane) {
  const Range rows = split_evenly(config.routes, config.threads_per_expert, lane);
  for (int64_t row = rows.begin; row < rows.begin + rows.size; ++row) {
    const uint16_t* src = source + (row * config.experts + expert) * static_cast<int64_t>(config.hidden);
    uint16_t* dst =
        destination + (static_cast<int64_t>(expert) * config.routes + row) * static_cast<int64_t>(config.hidden);
    std::memcpy(dst, src, static_cast<size_t>(config.hidden) * sizeof(uint16_t));
  }
}

void pack_m12(const uint16_t* source, int64_t source_stride, uint16_t* packed, int rows, int k_size, int lanes,
              int lane) {
  const Range blocks = split_evenly(rows / 12, lanes, lane);
  for (int64_t block = blocks.begin; block < blocks.begin + blocks.size; ++block) {
    uint16_t* block_dst = packed + block * 12 * static_cast<int64_t>(k_size);
    for (int k_block = 0; k_block < k_size; k_block += 4) {
      uint16_t* k_dst = block_dst + static_cast<int64_t>(k_block / 4) * 48;
      for (int row = 0; row < 12; ++row) {
        const uint16_t* src = source + (block * 12 + row) * source_stride + k_block;
        std::memcpy(k_dst + row * 4, src, 4 * sizeof(uint16_t));
      }
    }
  }
}

void run_plain_f32(const uint16_t* packed_a, const uint16_t* packed_b, float* output, int m, int k, int n, int n_tile,
                   int lanes, int lane) {
  const Range tiles = split_evenly(n / n_tile, lanes, lane);
  if (tiles.size == 0) {
    return;
  }
  const int n_begin = static_cast<int>(tiles.begin) * n_tile;
  gemm_params_t params{};
  params.m = 12;
  params.k = k;
  params.n = static_cast<int>(tiles.size) * n_tile;
  params.lda = k;
  params.ldb = k;
  params.ldc = n;
  const uint16_t* b = packed_b + tiles.begin * k * static_cast<int64_t>(n_tile);
  for (int row = 0; row < m; row += 12) {
    moe_sve_w2_packed_m12(packed_a + static_cast<int64_t>(row) * k, b, output + static_cast<int64_t>(row) * n + n_begin,
                          nullptr, &params);
  }
}

void run_w2_bf16(const uint16_t* packed_a, const uint16_t* packed_b, uint16_t* output, int m, int k, int n, int n_tile,
                 int lanes, int lane) {
  const Range tiles = split_evenly(n / n_tile, lanes, lane);
  if (tiles.size == 0) {
    return;
  }
  const int n_begin = static_cast<int>(tiles.begin) * n_tile;
  gemm_params_t params{};
  params.m = 12;
  params.k = k;
  params.n = static_cast<int>(tiles.size) * n_tile;
  params.lda = k;
  params.ldb = k;
  params.ldc = n;
  const uint16_t* b = packed_b + tiles.begin * k * static_cast<int64_t>(n_tile);
  for (int row = 0; row < m; row += 12) {
    moe_sve_w2_packed_bf16_m12(packed_a + static_cast<int64_t>(row) * k, b,
                               output + static_cast<int64_t>(row) * n + n_begin, nullptr, &params);
  }
}

void run_fused_w13_dimensions(const uint16_t* packed_a, const uint16_t* packed_b, uint16_t* packed_c, int routes,
                              int hidden, int intermediate, int w13_ranges, int lanes, int n_tile, int lane) {
  const int total_tiles = 2 * intermediate / n_tile;
  const int range_tiles = total_tiles / w13_ranges;
  for (int range_index = 0; range_index < w13_ranges; ++range_index) {
    const Range lane_tiles = split_evenly(range_tiles, lanes, lane);
    if (lane_tiles.size == 0) {
      continue;
    }
    const int64_t absolute_tile = static_cast<int64_t>(range_index) * range_tiles + lane_tiles.begin;
    const int n_begin = static_cast<int>(absolute_tile) * n_tile;
    gemm_params_t params{};
    params.m = routes;
    params.k = hidden;
    params.n = static_cast<int>(lane_tiles.size) * n_tile;
    params.lda = hidden;
    params.ldb = hidden;
    params.ldc = intermediate;
    moe_sve_w13_silu_poly5_packc_m12_rows_opt(packed_a,
                                              packed_b + absolute_tile * hidden * static_cast<int64_t>(n_tile),
                                              packed_c + static_cast<int64_t>(n_begin) * 6, nullptr, &params);
  }
}

void run_fused_w13(const uint16_t* packed_a, const uint16_t* packed_b, uint16_t* packed_c, const Config& config,
                   int n_tile, int lane) {
  run_fused_w13_dimensions(packed_a, packed_b, packed_c, config.routes, config.hidden, config.intermediate,
                           config.w13_ranges, config.threads_per_expert, n_tile, lane);
}

void run_silu(const float* gate, float* output, int64_t elements, int lanes, int lane) {
  const Range range = split_evenly(elements, lanes, lane);
  const int64_t end = range.begin + range.size;
  const int64_t vector_length = static_cast<int64_t>(svcntw());
  for (int64_t index = range.begin; index < end; index += vector_length) {
    const svbool_t predicate = svwhilelt_b32(index, end);
    const svfloat32_t value = svld1_f32(predicate, gate + index);
    const svfloat32_t denominator = svadd_n_f32_x(predicate, exp_poly5_neg(predicate, value), 1.0f);
    svst1_f32(predicate, output + index, svdiv_f32_x(predicate, value, denominator));
  }
}

void run_multiply(const float* silu, const float* up, uint16_t* output, int64_t elements, int lanes, int lane) {
  const Range range = split_evenly(elements, lanes, lane);
  const int64_t end = range.begin + range.size;
  const int64_t vector_length = static_cast<int64_t>(svcntw());
  for (int64_t index = range.begin; index < end; index += vector_length) {
    const svbool_t predicate = svwhilelt_b32(index, end);
    const svfloat32_t product =
        svmul_f32_x(predicate, svld1_f32(predicate, silu + index), svld1_f32(predicate, up + index));
    svst1h_u32(predicate, output + index, f32_to_bf16_bits(predicate, product));
  }
}

struct TimingCompletion {
  std::vector<Clock::time_point>* boundaries = nullptr;
  size_t* phase = nullptr;

  void operator()() noexcept { (*boundaries)[(*phase)++] = Clock::now(); }
};

using TeamBarrier = std::barrier<TimingCompletion>;

struct ClockCompletion {
  Clock::time_point* timestamp = nullptr;

  void operator()() noexcept { *timestamp = Clock::now(); }
};

using ClockBarrier = std::barrier<ClockCompletion>;

}  // namespace

class Experiment::Impl {
 public:
  explicit Impl(Config config) : config_(config), n_tile_(static_cast<int>(svcntb() / 2)) {
    validate();
    scratch_ = make_scratch(config_);
    copies_.reserve(static_cast<size_t>(config_.copies));
    const size_t source_elements = static_cast<size_t>(config_.experts) * config_.routes * config_.hidden;
    for (int copy = 0; copy < config_.copies; ++copy) {
      DataCopy data{AlignedBuffer<uint16_t>(source_elements), make_weights(config_, copy)};
      fill_source(data.source, config_, copy);
      copies_.push_back(std::move(data));
    }
    pool_ = std::make_unique<WorkerPool>(workers(), config_.cpu_start);
  }

  RunResult run(Variant variant, int copy_index) {
    if (copy_index < 0 || copy_index >= config_.copies) {
      throw std::out_of_range("copy index is out of range");
    }
    DataCopy& data = copies_[static_cast<size_t>(copy_index)];
    const std::vector<std::string> stage_names =
        variant == Variant::kExplicitUnfused
            ? std::vector<std::string>{"extract", "pack_a1",  "w1_gemm", "silu",
                                       "w3_gemm", "multiply", "pack_a2", "w2_gemm"}
            : std::vector<std::string>{"gather_pack_a1", "fused_w13_silu_mul_packc", "w2_gemm"};
    const size_t boundary_count = stage_names.size() + 1;
    std::vector<std::vector<Clock::time_point>> boundaries(static_cast<size_t>(config_.experts),
                                                           std::vector<Clock::time_point>(boundary_count));
    std::vector<size_t> phases(static_cast<size_t>(config_.experts), 0);
    std::vector<std::unique_ptr<TeamBarrier>> barriers;
    barriers.reserve(static_cast<size_t>(config_.experts));
    for (int expert = 0; expert < config_.experts; ++expert) {
      barriers.push_back(std::make_unique<TeamBarrier>(
          config_.threads_per_expert,
          TimingCompletion{&boundaries[static_cast<size_t>(expert)], &phases[static_cast<size_t>(expert)]}));
    }

    const std::function<void(int)> job = [&](int tid) {
      const int expert = tid / config_.threads_per_expert;
      const int lane = tid % config_.threads_per_expert;
      TeamBarrier& barrier = *barriers[static_cast<size_t>(expert)];
      barrier.arrive_and_wait();
      if (variant == Variant::kExplicitUnfused) {
        run_unfused_worker(data, expert, lane, barrier);
      } else {
        run_fused_worker(data, expert, lane, barrier);
      }
    };
    pool_->run(job);

    Clock::time_point first_start = boundaries[0][0];
    Clock::time_point last_finish = boundaries[0].back();
    std::vector<double> stage_seconds(stage_names.size(), 0.0);
    for (int expert = 0; expert < config_.experts; ++expert) {
      const auto& expert_boundaries = boundaries[static_cast<size_t>(expert)];
      first_start = std::min(first_start, expert_boundaries.front());
      last_finish = std::max(last_finish, expert_boundaries.back());
      for (size_t stage = 0; stage < stage_names.size(); ++stage) {
        const double seconds =
            std::chrono::duration<double>(expert_boundaries[stage + 1] - expert_boundaries[stage]).count();
        stage_seconds[stage] = std::max(stage_seconds[stage], seconds);
      }
    }
    return {std::chrono::duration<double>(last_finish - first_start).count(), stage_names, stage_seconds};
  }

  ErrorMetrics check(int copy) {
    run(Variant::kExplicitUnfused, copy);
    run(Variant::kProductionFused, copy);
    const int64_t intermediate_elements = static_cast<int64_t>(config_.experts) * config_.routes * config_.intermediate;
    const int64_t output_elements = static_cast<int64_t>(config_.experts) * config_.routes * config_.hidden;
    ErrorMetrics metrics;
    compare(scratch_.packed_product.data(), scratch_.fused_intermediate.data(), intermediate_elements,
            metrics.intermediate_mismatches, metrics.intermediate_max_abs, metrics.intermediate_relative_l2);
    compare(scratch_.unfused_output.data(), scratch_.fused_output.data(), output_elements, metrics.output_mismatches,
            metrics.output_max_abs, metrics.output_relative_l2);
    metrics.output_elements = output_elements;
    return metrics;
  }

  int n_tile() const { return n_tile_; }
  int workers() const { return config_.experts * config_.threads_per_expert; }

  uint64_t allocated_bytes() const {
    uint64_t bytes = scratch_.bytes();
    for (const DataCopy& copy : copies_) {
      bytes += copy.bytes();
    }
    return bytes;
  }

  double flop_count() const {
    return 6.0 * config_.experts * config_.routes * config_.hidden * static_cast<double>(config_.intermediate);
  }

 private:
  void validate() const {
    if (config_.experts <= 0 || config_.routes <= 0 || config_.hidden <= 0 || config_.intermediate <= 0 ||
        config_.threads_per_expert <= 0 || config_.w13_ranges <= 0 || config_.copies <= 0 || config_.cpu_start < 0) {
      throw std::invalid_argument("all dimensions/counts must be positive and cpu_start must be nonnegative");
    }
    if (config_.routes % 12 != 0) {
      throw std::invalid_argument("routes must be a multiple of 12 for the M12-only comparison");
    }
    if (config_.hidden % 8 != 0 || config_.intermediate % 8 != 0 || config_.intermediate % 4 != 0) {
      throw std::invalid_argument("hidden/intermediate must satisfy the SVE M12 K and W13 interleave alignment");
    }
    if (config_.hidden % n_tile_ != 0 || config_.intermediate % n_tile_ != 0 ||
        (2 * config_.intermediate) % n_tile_ != 0) {
      throw std::invalid_argument("hidden/intermediate N dimensions must be multiples of the runtime SVE n_tile");
    }
    const int w13_tiles = 2 * config_.intermediate / n_tile_;
    if (w13_tiles % config_.w13_ranges != 0) {
      throw std::invalid_argument("w13_ranges must evenly divide the packed W13 N tiles");
    }
    const int max_lanes =
        std::min({config_.hidden / n_tile_, config_.intermediate / n_tile_, w13_tiles / config_.w13_ranges});
    if (config_.threads_per_expert > max_lanes) {
      throw std::invalid_argument("threads_per_expert exceeds an available N-split tile count");
    }
    if (config_.cpu_start + workers() > CPU_SETSIZE) {
      throw std::invalid_argument("requested CPU affinity exceeds CPU_SETSIZE");
    }
  }

  void run_unfused_worker(const DataCopy& data, int expert, int lane, TeamBarrier& barrier) {
    const int64_t input_stride = static_cast<int64_t>(config_.routes) * config_.hidden;
    const int64_t intermediate_stride = static_cast<int64_t>(config_.routes) * config_.intermediate;
    const int64_t weight_stride = static_cast<int64_t>(config_.hidden) * config_.intermediate;
    const uint16_t* extracted = scratch_.extracted.data() + expert * input_stride;
    uint16_t* packed_input = scratch_.packed_input.data() + expert * input_stride;
    float* gate = scratch_.gate.data() + expert * intermediate_stride;
    float* silu = scratch_.silu.data() + expert * intermediate_stride;
    float* up = scratch_.up.data() + expert * intermediate_stride;
    uint16_t* product = scratch_.product.data() + expert * intermediate_stride;
    uint16_t* packed_product = scratch_.packed_product.data() + expert * intermediate_stride;
    uint16_t* output = scratch_.unfused_output.data() + expert * input_stride;
    extract_rows(data.source.data(), scratch_.extracted.data(), config_, expert, lane);
    barrier.arrive_and_wait();
    pack_m12(extracted, config_.hidden, packed_input, config_.routes, config_.hidden, config_.threads_per_expert, lane);
    barrier.arrive_and_wait();
    run_plain_f32(packed_input, data.weights.w1.data() + expert * weight_stride, gate, config_.routes, config_.hidden,
                  config_.intermediate, n_tile_, config_.threads_per_expert, lane);
    barrier.arrive_and_wait();
    run_silu(gate, silu, intermediate_stride, config_.threads_per_expert, lane);
    barrier.arrive_and_wait();
    run_plain_f32(packed_input, data.weights.w3.data() + expert * weight_stride, up, config_.routes, config_.hidden,
                  config_.intermediate, n_tile_, config_.threads_per_expert, lane);
    barrier.arrive_and_wait();
    run_multiply(silu, up, product, intermediate_stride, config_.threads_per_expert, lane);
    barrier.arrive_and_wait();
    pack_m12(product, config_.intermediate, packed_product, config_.routes, config_.intermediate,
             config_.threads_per_expert, lane);
    barrier.arrive_and_wait();
    run_w2_bf16(packed_product, data.weights.w2.data() + expert * weight_stride, output, config_.routes,
                config_.intermediate, config_.hidden, n_tile_, config_.threads_per_expert, lane);
    barrier.arrive_and_wait();
  }

  void run_fused_worker(const DataCopy& data, int expert, int lane, TeamBarrier& barrier) {
    const int64_t input_stride = static_cast<int64_t>(config_.routes) * config_.hidden;
    const int64_t intermediate_stride = static_cast<int64_t>(config_.routes) * config_.intermediate;
    const int64_t weight_stride = static_cast<int64_t>(config_.hidden) * config_.intermediate;
    uint16_t* packed_input = scratch_.packed_input.data() + expert * input_stride;
    uint16_t* intermediate = scratch_.fused_intermediate.data() + expert * intermediate_stride;
    uint16_t* output = scratch_.fused_output.data() + expert * input_stride;
    const uint16_t* expert_source = data.source.data() + static_cast<int64_t>(expert) * config_.hidden;
    pack_m12(expert_source, static_cast<int64_t>(config_.experts) * config_.hidden, packed_input, config_.routes,
             config_.hidden, config_.threads_per_expert, lane);
    barrier.arrive_and_wait();
    run_fused_w13(packed_input, data.weights.w13.data() + expert * weight_stride * 2, intermediate, config_, n_tile_,
                  lane);
    barrier.arrive_and_wait();
    run_w2_bf16(intermediate, data.weights.w2.data() + expert * weight_stride, output, config_.routes,
                config_.intermediate, config_.hidden, n_tile_, config_.threads_per_expert, lane);
    barrier.arrive_and_wait();
  }

  static void compare(const uint16_t* reference, const uint16_t* actual, int64_t elements, int64_t& mismatches,
                      float& max_abs, double& relative_l2) {
    long double reference_squared = 0.0;
    long double difference_squared = 0.0;
    for (int64_t index = 0; index < elements; ++index) {
      mismatches += reference[index] != actual[index] ? 1 : 0;
      const float reference_value = bf16_to_float(reference[index]);
      const float actual_value = bf16_to_float(actual[index]);
      const float difference = actual_value - reference_value;
      max_abs = std::max(max_abs, std::abs(difference));
      reference_squared += static_cast<long double>(reference_value) * reference_value;
      difference_squared += static_cast<long double>(difference) * difference;
    }
    relative_l2 = std::sqrt(static_cast<double>(difference_squared / std::max(reference_squared, 1.0e-30L)));
  }

  Config config_;
  int n_tile_ = 0;
  std::vector<DataCopy> copies_;
  Scratch scratch_;
  std::unique_ptr<WorkerPool> pool_;
};

class FragmentedExperiment::Impl {
 public:
  explicit Impl(FragmentedConfig config) : config_(config), n_tile_(static_cast<int>(svcntb() / 2)) {
    validate();
    build_tasks();
    const size_t input_elements = static_cast<size_t>(total_routes_) * config_.hidden;
    const size_t intermediate_elements = static_cast<size_t>(total_routes_) * config_.intermediate;
    scratch_ = Scratch{AlignedBuffer<uint16_t>(input_elements), AlignedBuffer<uint16_t>(intermediate_elements),
                       AlignedBuffer<uint16_t>(input_elements)};

    copies_.reserve(static_cast<size_t>(config_.copies));
    const size_t hf = static_cast<size_t>(config_.hidden) * config_.intermediate;
    for (int copy = 0; copy < config_.copies; ++copy) {
      DataCopy data{AlignedBuffer<uint16_t>(input_elements),
                    AlignedBuffer<uint16_t>(static_cast<size_t>(task_count()) * hf * 2),
                    AlignedBuffer<uint16_t>(static_cast<size_t>(task_count()) * hf)};
      fill_data(data, copy);
      copies_.push_back(std::move(data));
    }
    pool_ = std::make_unique<WorkerPool>(workers(), config_.cpu_start);
  }

  RunResult run(int copy_index) {
    if (copy_index < 0 || copy_index >= config_.copies) {
      throw std::out_of_range("copy index is out of range");
    }
    DataCopy& data = copies_[static_cast<size_t>(copy_index)];
    const std::vector<std::string> stage_names = {"gather_pack_a1", "fused_w13_silu_mul_packc", "w2_gemm"};
    std::vector<std::vector<Clock::time_point>> boundaries;
    std::vector<size_t> phases(static_cast<size_t>(config_.teams), 0);
    std::vector<size_t> team_task_counts(static_cast<size_t>(config_.teams), 0);
    std::vector<std::unique_ptr<TeamBarrier>> team_barriers;
    std::vector<std::unique_ptr<std::barrier<>>> claim_barriers;
    boundaries.reserve(static_cast<size_t>(config_.teams));
    team_barriers.reserve(static_cast<size_t>(config_.teams));
    claim_barriers.reserve(static_cast<size_t>(config_.teams));
    const bool dynamic = config_.schedule == FragmentedConfig::Schedule::kDynamic;
    for (int team = 0; team < config_.teams; ++team) {
      const size_t max_team_tasks = dynamic ? tasks_.size() : team_tasks_[static_cast<size_t>(team)].size();
      boundaries.emplace_back(1 + 3 * max_team_tasks);
      team_barriers.push_back(std::make_unique<TeamBarrier>(
          config_.threads_per_team,
          TimingCompletion{&boundaries[static_cast<size_t>(team)], &phases[static_cast<size_t>(team)]}));
      claim_barriers.push_back(std::make_unique<std::barrier<>>(config_.threads_per_team));
      if (!dynamic) {
        team_task_counts[static_cast<size_t>(team)] = team_tasks_[static_cast<size_t>(team)].size();
      }
    }

    Clock::time_point start;
    Clock::time_point finish;
    ClockBarrier start_barrier(workers(), ClockCompletion{&start});
    ClockBarrier finish_barrier(workers(), ClockCompletion{&finish});
    std::atomic<int> next_task{0};
    std::vector<int> current_tasks(static_cast<size_t>(config_.teams), -1);
    const std::function<void(int)> job = [&](int tid) {
      const int team = tid / config_.threads_per_team;
      const int lane = tid % config_.threads_per_team;
      TeamBarrier& team_barrier = *team_barriers[static_cast<size_t>(team)];
      start_barrier.arrive_and_wait();
      team_barrier.arrive_and_wait();
      if (dynamic) {
        run_dynamic_team(data, team, lane, next_task, current_tasks, team_task_counts,
                         *claim_barriers[static_cast<size_t>(team)], team_barrier);
      } else {
        run_slot_team(data, team, lane, team_barrier);
      }
      finish_barrier.arrive_and_wait();
    };
    pool_->run(job);

    std::vector<double> stage_seconds(stage_names.size(), 0.0);
    for (int team = 0; team < config_.teams; ++team) {
      std::vector<double> team_stage_seconds(stage_names.size(), 0.0);
      const auto& team_boundaries = boundaries[static_cast<size_t>(team)];
      const size_t team_task_count = team_task_counts[static_cast<size_t>(team)];
      for (size_t task = 0; task < team_task_count; ++task) {
        for (size_t stage = 0; stage < stage_names.size(); ++stage) {
          const size_t boundary = task * stage_names.size() + stage;
          team_stage_seconds[stage] +=
              std::chrono::duration<double>(team_boundaries[boundary + 1] - team_boundaries[boundary]).count();
        }
      }
      for (size_t stage = 0; stage < stage_names.size(); ++stage) {
        stage_seconds[stage] = std::max(stage_seconds[stage], team_stage_seconds[stage]);
      }
    }
    return {std::chrono::duration<double>(finish - start).count(), stage_names, stage_seconds};
  }

  FragmentedCheckMetrics check(int copy_index) {
    run(copy_index);
    const DataCopy& data = copies_[static_cast<size_t>(copy_index)];
    int max_routes = 0;
    for (const Task& task : tasks_) {
      max_routes = std::max(max_routes, task.routes);
    }
    AlignedBuffer<uint16_t> reference_packed(static_cast<size_t>(max_routes) * config_.hidden);
    AlignedBuffer<uint16_t> reference_intermediate(static_cast<size_t>(max_routes) * config_.intermediate);
    AlignedBuffer<uint16_t> reference_output(static_cast<size_t>(max_routes) * config_.hidden);
    FragmentedCheckMetrics metrics;
    const size_t hf = static_cast<size_t>(config_.hidden) * config_.intermediate;
    for (size_t task_index = 0; task_index < tasks_.size(); ++task_index) {
      const Task& task = tasks_[task_index];
      const int64_t input_offset = task.route_offset * static_cast<int64_t>(config_.hidden);
      const int64_t intermediate_offset = task.route_offset * static_cast<int64_t>(config_.intermediate);
      pack_m12(data.source.data() + input_offset, config_.hidden, reference_packed.data(), task.routes, config_.hidden,
               1, 0);
      run_fused_w13_dimensions(reference_packed.data(), data.w13.data() + task_index * hf * 2,
                               reference_intermediate.data(), task.routes, config_.hidden, config_.intermediate,
                               config_.w13_ranges, 1, n_tile_, 0);
      run_w2_bf16(reference_intermediate.data(), data.w2.data() + task_index * hf, reference_output.data(), task.routes,
                  config_.intermediate, config_.hidden, n_tile_, 1, 0);
      compare_exact(scratch_.intermediate.data() + intermediate_offset, reference_intermediate.data(),
                    static_cast<int64_t>(task.routes) * config_.intermediate, metrics.intermediate_mismatches,
                    metrics.intermediate_elements);
      compare_exact(scratch_.output.data() + input_offset, reference_output.data(),
                    static_cast<int64_t>(task.routes) * config_.hidden, metrics.output_mismatches,
                    metrics.output_elements);
    }
    return metrics;
  }

  int n_tile() const { return n_tile_; }
  int workers() const { return config_.teams * config_.threads_per_team; }
  int task_count() const { return static_cast<int>(tasks_.size()); }
  int total_routes() const { return total_routes_; }

  uint64_t active_stage_bytes() const {
    return static_cast<uint64_t>(config_.teams) * 2 * config_.hidden * config_.intermediate;
  }

  uint64_t unique_weight_bytes() const {
    return static_cast<uint64_t>(task_count()) * 6 * config_.hidden * config_.intermediate;
  }

  uint64_t allocated_bytes() const {
    uint64_t bytes = scratch_.bytes();
    for (const DataCopy& copy : copies_) {
      bytes += copy.bytes();
    }
    return bytes;
  }

  double flop_count() const {
    return 6.0 * config_.teams * config_.base_routes * config_.hidden * static_cast<double>(config_.intermediate);
  }

 private:
  struct Task {
    int routes = 0;
    int64_t route_offset = 0;
  };

  struct DataCopy {
    AlignedBuffer<uint16_t> source;
    AlignedBuffer<uint16_t> w13;
    AlignedBuffer<uint16_t> w2;

    uint64_t bytes() const { return source.bytes() + w13.bytes() + w2.bytes(); }
  };

  struct Scratch {
    AlignedBuffer<uint16_t> packed_input;
    AlignedBuffer<uint16_t> intermediate;
    AlignedBuffer<uint16_t> output;

    uint64_t bytes() const { return packed_input.bytes() + intermediate.bytes() + output.bytes(); }
  };

  void validate() const {
    if (config_.teams <= 0 || config_.base_routes <= 0 || config_.hidden <= 0 || config_.intermediate <= 0 ||
        config_.threads_per_team <= 0 || config_.w13_ranges <= 0 || config_.copies <= 0 || config_.cpu_start < 0 ||
        config_.replaced_teams < 0 || config_.replaced_teams > config_.teams || config_.split_factor <= 0) {
      throw std::invalid_argument("fragmented experiment dimensions/counts are invalid");
    }
    if (config_.base_routes % 12 != 0 || config_.base_routes % config_.split_factor != 0 ||
        (config_.base_routes / config_.split_factor) % 12 != 0) {
      throw std::invalid_argument("base and fragmented routes must be multiples of M12");
    }
    if (config_.hidden % 8 != 0 || config_.intermediate % 8 != 0 || config_.intermediate % 4 != 0 ||
        config_.hidden % n_tile_ != 0 || config_.intermediate % n_tile_ != 0 ||
        (2 * config_.intermediate) % n_tile_ != 0) {
      throw std::invalid_argument("hidden/intermediate do not satisfy the SVE kernel alignment");
    }
    const int w13_tiles = 2 * config_.intermediate / n_tile_;
    if (w13_tiles % config_.w13_ranges != 0) {
      throw std::invalid_argument("w13_ranges must evenly divide the packed W13 N tiles");
    }
    const int max_lanes =
        std::min({config_.hidden / n_tile_, config_.intermediate / n_tile_, w13_tiles / config_.w13_ranges});
    if (config_.threads_per_team > max_lanes) {
      throw std::invalid_argument("threads_per_team exceeds an available N-split tile count");
    }
    if (config_.cpu_start + workers() > CPU_SETSIZE) {
      throw std::invalid_argument("requested CPU affinity exceeds CPU_SETSIZE");
    }
    const int64_t total_routes = static_cast<int64_t>(config_.teams) * config_.base_routes;
    if (total_routes > std::numeric_limits<int>::max()) {
      throw std::invalid_argument("total routes exceed the experiment index range");
    }
  }

  bool team_is_replaced(int team) const {
    const int previous = team * config_.replaced_teams / config_.teams;
    const int next = (team + 1) * config_.replaced_teams / config_.teams;
    return next != previous;
  }

  void build_tasks() {
    team_tasks_.resize(static_cast<size_t>(config_.teams));
    int64_t route_offset = 0;
    for (int team = 0; team < config_.teams; ++team) {
      const int fragments = team_is_replaced(team) ? config_.split_factor : 1;
      const int routes = config_.base_routes / fragments;
      for (int fragment = 0; fragment < fragments; ++fragment) {
        const int task = static_cast<int>(tasks_.size());
        tasks_.push_back(Task{routes, route_offset});
        team_tasks_[static_cast<size_t>(team)].push_back(task);
        route_offset += routes;
      }
    }
    total_routes_ = static_cast<int>(route_offset);
    dynamic_task_order_.reserve(tasks_.size());
    for (size_t task = 0; task < tasks_.size(); ++task) {
      dynamic_task_order_.push_back(static_cast<int>(task));
    }
    std::stable_sort(dynamic_task_order_.begin(), dynamic_task_order_.end(), [&](int left, int right) {
      return tasks_[static_cast<size_t>(left)].routes > tasks_[static_cast<size_t>(right)].routes;
    });
  }

  void fill_data(DataCopy& data, int copy) const {
    const size_t hf = static_cast<size_t>(config_.hidden) * config_.intermediate;
    for (size_t task_index = 0; task_index < tasks_.size(); ++task_index) {
      const Task& task = tasks_[task_index];
      uint16_t* source = data.source.data() + task.route_offset * static_cast<int64_t>(config_.hidden);
      for (int route = 0; route < task.routes; ++route) {
        for (int column = 0; column < config_.hidden; ++column) {
          source[static_cast<int64_t>(route) * config_.hidden + column] =
              input_value(copy, static_cast<int>(task_index), route, column);
        }
      }
      uint16_t* w13 = data.w13.data() + task_index * hf * 2;
      uint16_t* w2 = data.w2.data() + task_index * hf;
      fill_packed_b(w13, config_.hidden, config_.intermediate * 2, [&](int k, int packed_column) {
        const int block = packed_column / 8;
        const int offset = packed_column % 8;
        const int kind = offset < 4 ? 0 : 1;
        const int feature = block * 4 + (offset & 3);
        return weight_value(copy, static_cast<int>(task_index), kind, k, feature);
      });
      fill_packed_b(w2, config_.intermediate, config_.hidden,
                    [&](int k, int n) { return weight_value(copy, static_cast<int>(task_index), 2, k, n); });
    }
  }

  void run_task(const DataCopy& data, int task_index, int lane, TeamBarrier& barrier) {
    const size_t hf = static_cast<size_t>(config_.hidden) * config_.intermediate;
    const Task& task = tasks_[static_cast<size_t>(task_index)];
    const int64_t input_offset = task.route_offset * static_cast<int64_t>(config_.hidden);
    const int64_t intermediate_offset = task.route_offset * static_cast<int64_t>(config_.intermediate);
    const uint16_t* source = data.source.data() + input_offset;
    uint16_t* packed_input = scratch_.packed_input.data() + input_offset;
    uint16_t* intermediate = scratch_.intermediate.data() + intermediate_offset;
    uint16_t* output = scratch_.output.data() + input_offset;
    pack_m12(source, config_.hidden, packed_input, task.routes, config_.hidden, config_.threads_per_team, lane);
    barrier.arrive_and_wait();
    run_fused_w13_dimensions(packed_input, data.w13.data() + static_cast<size_t>(task_index) * hf * 2, intermediate,
                             task.routes, config_.hidden, config_.intermediate, config_.w13_ranges,
                             config_.threads_per_team, n_tile_, lane);
    barrier.arrive_and_wait();
    run_w2_bf16(intermediate, data.w2.data() + static_cast<size_t>(task_index) * hf, output, task.routes,
                config_.intermediate, config_.hidden, n_tile_, config_.threads_per_team, lane);
    barrier.arrive_and_wait();
  }

  void run_slot_team(const DataCopy& data, int team, int lane, TeamBarrier& barrier) {
    for (int task_index : team_tasks_[static_cast<size_t>(team)]) {
      run_task(data, task_index, lane, barrier);
    }
  }

  void run_dynamic_team(const DataCopy& data, int team, int lane, std::atomic<int>& next_task,
                        std::vector<int>& current_tasks, std::vector<size_t>& team_task_counts,
                        std::barrier<>& claim_barrier, TeamBarrier& timing_barrier) {
    while (true) {
      if (lane == 0) {
        const int order_index = next_task.fetch_add(1, std::memory_order_relaxed);
        current_tasks[static_cast<size_t>(team)] =
            order_index < task_count() ? dynamic_task_order_[static_cast<size_t>(order_index)] : -1;
        if (current_tasks[static_cast<size_t>(team)] >= 0) {
          ++team_task_counts[static_cast<size_t>(team)];
        }
      }
      claim_barrier.arrive_and_wait();
      const int task_index = current_tasks[static_cast<size_t>(team)];
      if (task_index < 0) {
        return;
      }
      run_task(data, task_index, lane, timing_barrier);
    }
  }

  static void compare_exact(const uint16_t* actual, const uint16_t* reference, int64_t elements, int64_t& mismatches,
                            int64_t& compared_elements) {
    for (int64_t index = 0; index < elements; ++index) {
      mismatches += actual[index] != reference[index] ? 1 : 0;
    }
    compared_elements += elements;
  }

  FragmentedConfig config_;
  int n_tile_ = 0;
  int total_routes_ = 0;
  std::vector<Task> tasks_;
  std::vector<std::vector<int>> team_tasks_;
  std::vector<int> dynamic_task_order_;
  std::vector<DataCopy> copies_;
  Scratch scratch_;
  std::unique_ptr<WorkerPool> pool_;
};

Experiment::Experiment(Config config) : impl_(std::make_unique<Impl>(config)) {}
Experiment::~Experiment() = default;

RunResult Experiment::run(Variant variant, int copy) { return impl_->run(variant, copy); }
ErrorMetrics Experiment::check(int copy) { return impl_->check(copy); }
int Experiment::n_tile() const { return impl_->n_tile(); }
int Experiment::workers() const { return impl_->workers(); }
uint64_t Experiment::allocated_bytes() const { return impl_->allocated_bytes(); }
double Experiment::flop_count() const { return impl_->flop_count(); }

FragmentedExperiment::FragmentedExperiment(FragmentedConfig config) : impl_(std::make_unique<Impl>(config)) {}
FragmentedExperiment::~FragmentedExperiment() = default;
RunResult FragmentedExperiment::run(int copy) { return impl_->run(copy); }
FragmentedCheckMetrics FragmentedExperiment::check(int copy) { return impl_->check(copy); }
int FragmentedExperiment::n_tile() const { return impl_->n_tile(); }
int FragmentedExperiment::workers() const { return impl_->workers(); }
int FragmentedExperiment::task_count() const { return impl_->task_count(); }
int FragmentedExperiment::total_routes() const { return impl_->total_routes(); }
uint64_t FragmentedExperiment::active_stage_bytes() const { return impl_->active_stage_bytes(); }
uint64_t FragmentedExperiment::unique_weight_bytes() const { return impl_->unique_weight_bytes(); }
uint64_t FragmentedExperiment::allocated_bytes() const { return impl_->allocated_bytes(); }
double FragmentedExperiment::flop_count() const { return impl_->flop_count(); }

}  // namespace fused_moe_sve::unfused
