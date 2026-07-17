#include "moe/common/route_merge.h"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Config {
  int64_t tokens = 2048;
  int64_t hidden = 4096;
  int64_t top_k = 6;
  int threads = 96;
  int warmup = 3;
  int runs = 11;
  int copies = 1;
  std::string source = "f32";
  bool check = false;
};

int64_t parse_i64(const char* value, const char* name) {
  char* end = nullptr;
  const long long parsed = std::strtoll(value, &end, 10);
  if (end == value || *end != '\0') {
    throw std::invalid_argument(std::string("invalid ") + name + ": " + value);
  }
  return static_cast<int64_t>(parsed);
}

Config parse_args(int argc, char** argv) {
  Config config;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto take = [&](const char* name) -> const char* {
      if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (arg == "--tokens") {
      config.tokens = parse_i64(take("--tokens"), "tokens");
    } else if (arg == "--hidden") {
      config.hidden = parse_i64(take("--hidden"), "hidden");
    } else if (arg == "--top-k") {
      config.top_k = parse_i64(take("--top-k"), "top-k");
    } else if (arg == "--threads") {
      config.threads = static_cast<int>(parse_i64(take("--threads"), "threads"));
    } else if (arg == "--warmup") {
      config.warmup = static_cast<int>(parse_i64(take("--warmup"), "warmup"));
    } else if (arg == "--runs") {
      config.runs = static_cast<int>(parse_i64(take("--runs"), "runs"));
    } else if (arg == "--copies") {
      config.copies = static_cast<int>(parse_i64(take("--copies"), "copies"));
    } else if (arg == "--source") {
      config.source = take("--source");
    } else if (arg == "--check") {
      config.check = true;
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  if (config.tokens <= 0 || config.hidden <= 0 || config.top_k <= 0 || config.threads <= 0 || config.warmup < 0 ||
      config.runs <= 0 || config.copies <= 0) {
    throw std::invalid_argument(
        "tokens, hidden, top-k, threads, runs, and copies must be positive; warmup must be non-negative");
  }
  if (config.source != "f32" && config.source != "bf16" && config.source != "both") {
    throw std::invalid_argument("source must be f32, bf16, or both");
  }
  return config;
}

uint16_t f32_to_bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<uint16_t>((bits >> 16) | 0x40U);
  }
  const uint32_t rounding_bias = 0x7fffU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

float bf16_to_f32(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void convert_f32_to_bf16(const float* src, uint16_t* dst, int64_t n) {
  int64_t i = 0;
#if defined(__aarch64__) && defined(__ARM_FEATURE_BF16)
  for (; i + 8 <= n; i += 8) {
    const bfloat16x8_t packed = vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(vld1q_f32(src + i)), vld1q_f32(src + i + 4));
    vst1q_u16(dst + i, vreinterpretq_u16_bf16(packed));
  }
#endif
  for (; i < n; ++i) dst[i] = f32_to_bf16(src[i]);
}

void accumulate_f32(float* acc, const float* src, float weight, int64_t n) {
  int64_t i = 0;
#if defined(__aarch64__)
  for (; i + 8 <= n; i += 8) {
    vst1q_f32(acc + i, vfmaq_n_f32(vld1q_f32(acc + i), vld1q_f32(src + i), weight));
    vst1q_f32(acc + i + 4, vfmaq_n_f32(vld1q_f32(acc + i + 4), vld1q_f32(src + i + 4), weight));
  }
#endif
  for (; i < n; ++i) acc[i] += src[i] * weight;
}

void accumulate_bf16(float* acc, const uint16_t* src, float weight, int64_t n) {
  int64_t i = 0;
#if defined(__aarch64__)
  for (; i + 8 <= n; i += 8) {
    const uint16x8_t packed = vld1q_u16(src + i);
    const float32x4_t lo = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(packed)), 16));
    const float32x4_t hi = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_high_u16(packed)), 16));
    vst1q_f32(acc + i, vfmaq_n_f32(vld1q_f32(acc + i), lo, weight));
    vst1q_f32(acc + i + 4, vfmaq_n_f32(vld1q_f32(acc + i + 4), hi, weight));
  }
#endif
  for (; i < n; ++i) acc[i] += bf16_to_f32(src[i]) * weight;
}

template <typename Src>
void merge_sequential(const Src* route, const float* weights, uint16_t* output, int64_t token_begin, int64_t token_end,
                      int64_t top_k, int64_t hidden) {
  std::vector<float> acc(static_cast<size_t>(hidden));
  for (int64_t token = token_begin; token < token_end; ++token) {
    std::fill(acc.begin(), acc.end(), 0.0f);
    for (int64_t slot = 0; slot < top_k; ++slot) {
      const int64_t flat = token * top_k + slot;
      if constexpr (std::is_same_v<Src, float>) {
        accumulate_f32(acc.data(), route + flat * hidden, weights[flat], hidden);
      } else {
        accumulate_bf16(acc.data(), route + flat * hidden, weights[flat], hidden);
      }
    }
    convert_f32_to_bf16(acc.data(), output + token * hidden, hidden);
  }
}

std::vector<int> affinity_cpus() {
  std::vector<int> cpus;
#if defined(__linux__)
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (CPU_ISSET(cpu, &mask)) cpus.push_back(cpu);
    }
  }
#endif
  return cpus;
}

void bind_to_cpu(int cpu) {
#if defined(__linux__)
  if (cpu < 0) return;
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(cpu, &mask);
  const int error = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
  if (error != 0) throw std::runtime_error("pthread_setaffinity_np failed");
#else
  (void)cpu;
#endif
}

class ParallelRunner {
 public:
  explicit ParallelRunner(int threads)
      : start_(static_cast<std::ptrdiff_t>(threads + 1)), done_(static_cast<std::ptrdiff_t>(threads + 1)) {
    const std::vector<int> cpus = affinity_cpus();
    workers_.reserve(static_cast<size_t>(threads));
    for (int tid = 0; tid < threads; ++tid) {
      const int cpu = cpus.empty() ? -1 : cpus[static_cast<size_t>(tid) % cpus.size()];
      workers_.emplace_back([this, tid, cpu]() {
        bind_to_cpu(cpu);
        while (true) {
          start_.arrive_and_wait();
          if (stop_.load(std::memory_order_relaxed)) return;
          job_(tid);
          done_.arrive_and_wait();
        }
      });
    }
  }

  ParallelRunner(const ParallelRunner&) = delete;
  ParallelRunner& operator=(const ParallelRunner&) = delete;

  ~ParallelRunner() {
    stop_.store(true, std::memory_order_relaxed);
    start_.arrive_and_wait();
    for (std::thread& worker : workers_) worker.join();
  }

  double run(const std::function<void(int)>& job) {
    job_ = job;
    const auto begin = Clock::now();
    start_.arrive_and_wait();
    done_.arrive_and_wait();
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
  }

 private:
  std::barrier<> start_;
  std::barrier<> done_;
  std::atomic<bool> stop_{false};
  std::function<void(int)> job_;
  std::vector<std::thread> workers_;
};

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t mid = values.size() / 2;
  return values.size() % 2 == 0 ? 0.5 * (values[mid - 1] + values[mid]) : values[mid];
}

bool uses_fixed_tree(int64_t top_k) { return top_k == 2 || top_k == 4 || top_k == 6 || top_k == 8; }

int largest_power_of_two_less_than(int value) {
  int power = 1;
  while ((power << 1) < value) {
    power <<= 1;
  }
  return power;
}

template <typename Src>
float load_source(const Src* route, int64_t flat) {
  if constexpr (std::is_same_v<Src, float>) {
    return route[flat];
  } else {
    return bf16_to_f32(route[flat]);
  }
}

template <typename Src>
float tree_element(const Src* route, const float* weights, int64_t token, int64_t top_k, int64_t hidden, int64_t h,
                   int begin, int count) {
  if (count == 1) {
    const int64_t flat = (token * top_k + begin) * hidden + h;
    volatile float product = load_source(route, flat) * weights[token * top_k + begin];
    return product;
  }
  const int left_count = largest_power_of_two_less_than(count);
  volatile float lhs = tree_element(route, weights, token, top_k, hidden, h, begin, left_count);
  volatile float rhs = tree_element(route, weights, token, top_k, hidden, h, begin + left_count, count - left_count);
  volatile float sum = lhs + rhs;
  return sum;
}

template <typename Src>
void sve_reference(const Src* route, const float* weights, uint16_t* output, int64_t tokens, int64_t top_k,
                   int64_t hidden) {
  for (int64_t token = 0; token < tokens; ++token) {
    for (int64_t h = 0; h < hidden; ++h) {
      float value = 0.0f;
      if (uses_fixed_tree(top_k)) {
        value = tree_element(route, weights, token, top_k, hidden, h, 0, static_cast<int>(top_k));
      } else {
        for (int64_t slot = 0; slot < top_k; ++slot) {
          const int64_t flat = (token * top_k + slot) * hidden + h;
          value = std::fma(load_source(route, flat), weights[token * top_k + slot], value);
        }
      }
      output[token * hidden + h] = f32_to_bf16(value);
    }
  }
}

template <typename Src>
void run_source(const Config& config, const std::vector<Src>& route, const std::vector<float>& weights,
                const char* source_name) {
  ParallelRunner runner(config.threads);
  const size_t route_stride = static_cast<size_t>(config.tokens * config.top_k * config.hidden);
  const size_t output_stride = static_cast<size_t>(config.tokens * config.hidden);
  std::vector<uint16_t> output(output_stride * static_cast<size_t>(config.copies));
  std::vector<uint16_t> sve_ref;
  if (config.check) {
    sve_ref.resize(output_stride);
    sve_reference(route.data(), weights.data(), sve_ref.data(), config.tokens, config.top_k, config.hidden);
  }

  const double payload_bytes =
      static_cast<double>(config.tokens) *
      (static_cast<double>(config.top_k * config.hidden) * sizeof(Src) +
       static_cast<double>(config.top_k) * sizeof(float) + static_cast<double>(config.hidden) * sizeof(uint16_t));
  double baseline_ms = 0.0;
  int64_t iteration = 0;
  for (int variant : {0, 1, 2, 4}) {
    int copy = 0;
    auto job = [&](int tid) {
      const int64_t rows_per_thread = (config.tokens + config.threads - 1) / config.threads;
      const int64_t begin = static_cast<int64_t>(tid) * rows_per_thread;
      const int64_t end = std::min<int64_t>(config.tokens, begin + rows_per_thread);
      const Src* route_ptr = route.data() + static_cast<size_t>(copy) * route_stride;
      uint16_t* output_ptr = output.data() + static_cast<size_t>(copy) * output_stride;
      if (variant == 0) {
        merge_sequential(route_ptr, weights.data(), output_ptr, begin, end, config.top_k, config.hidden);
      } else if constexpr (std::is_same_v<Src, float>) {
        fused_cpp::moe_route_merge::merge_f32_sve(route_ptr, weights.data(), output_ptr, begin, end, config.top_k,
                                                  config.hidden, variant);
      } else {
        fused_cpp::moe_route_merge::merge_bf16_sve(route_ptr, weights.data(), output_ptr, begin, end, config.top_k,
                                                   config.hidden, variant);
      }
    };
    auto run_once = [&]() {
      copy = static_cast<int>(iteration++ % config.copies);
      return runner.run(job);
    };
    for (int i = 0; i < config.warmup; ++i) run_once();
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(config.runs));
    for (int i = 0; i < config.runs; ++i) samples.push_back(run_once());
    const double elapsed_ms = median(std::move(samples));
    if (variant == 0) baseline_ms = elapsed_ms;

    if (variant != 0 && config.check) {
      const auto output_begin = output.begin() + static_cast<std::ptrdiff_t>(copy * output_stride);
      const auto output_end = output_begin + static_cast<std::ptrdiff_t>(output_stride);
      const auto mismatch = std::mismatch(output_begin, output_end, sve_ref.begin());
      if (mismatch.first != output_end) {
        const size_t index = static_cast<size_t>(mismatch.first - output_begin);
        throw std::runtime_error("SVE output mismatch at element " + std::to_string(index));
      }
    }
    const double gbps = payload_bytes / (elapsed_ms * 1.0e6);
    const double speedup = variant == 0 ? 1.0 : baseline_ms / elapsed_ms;
    std::cout << source_name << ',' << config.tokens << ',' << config.top_k << ',' << config.hidden << ','
              << config.threads << ',' << config.copies << ','
              << (variant == 0 ? "baseline" : "sve_u" + std::to_string(variant)) << ',' << elapsed_ms << ',' << gbps
              << ',' << speedup << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Config config = parse_args(argc, argv);
    if (!fused_cpp::moe_route_merge::sve_available()) {
      throw std::runtime_error("this benchmark requires an SVE build");
    }
    const size_t route_elements = static_cast<size_t>(config.tokens * config.top_k * config.hidden);
    const size_t total_route_elements = route_elements * static_cast<size_t>(config.copies);
    std::vector<float> route_f32(config.source == "f32" || config.source == "both" ? total_route_elements : 0);
    std::vector<uint16_t> route_bf16(config.source == "bf16" || config.source == "both" ? total_route_elements : 0);
    uint32_t state = 0x12345678U;
    for (size_t i = 0; i < route_elements; ++i) {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      const float value = (static_cast<float>(state & 0xffffU) / 65535.0f - 0.5f) * 0.5f;
      if (!route_f32.empty()) route_f32[i] = value;
      if (!route_bf16.empty()) route_bf16[i] = f32_to_bf16(value);
    }
    for (int copy = 1; copy < config.copies; ++copy) {
      const size_t offset = static_cast<size_t>(copy) * route_elements;
      if (!route_f32.empty()) std::copy_n(route_f32.data(), route_elements, route_f32.data() + offset);
      if (!route_bf16.empty()) std::copy_n(route_bf16.data(), route_elements, route_bf16.data() + offset);
    }
    std::vector<float> weights(static_cast<size_t>(config.tokens * config.top_k));
    for (int64_t token = 0; token < config.tokens; ++token) {
      float sum = 0.0f;
      for (int64_t slot = 0; slot < config.top_k; ++slot) {
        const float value = static_cast<float>(((token + 3 * slot) % 11) + 1);
        weights[static_cast<size_t>(token * config.top_k + slot)] = value;
        sum += value;
      }
      for (int64_t slot = 0; slot < config.top_k; ++slot) {
        weights[static_cast<size_t>(token * config.top_k + slot)] /= sum;
      }
    }

    std::cout << "source,tokens,top_k,hidden,threads,copies,variant,median_ms,payload_gbps,speedup\n";
    if (config.source == "f32" || config.source == "both") {
      run_source(config, route_f32, weights, "f32");
    }
    if (config.source == "bf16" || config.source == "both") {
      run_source(config, route_bf16, weights, "bf16");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
