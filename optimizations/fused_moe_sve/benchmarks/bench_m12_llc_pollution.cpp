#include <algorithm>
#include <arm_sve.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <pthread.h>
#include <sched.h>

#include "gemm_params.h"

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE) || !defined(__ARM_FEATURE_BF16)
#error "bench_m12_llc_pollution requires AArch64 SVE BF16"
#endif

extern "C" {
void moe_sve_w13_silu_poly5_packc_m12_rows_opt(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                               const gemm_params_t*);
void moe_sve_w2_packed_bf16_m12(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
}

namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kCacheLineBytes = 64;
constexpr size_t kMiB = size_t{1} << 20;
constexpr int kM = 12;
constexpr int kHidden = 4096;
constexpr int kIntermediate = 512;
constexpr size_t kExpertWeightBytes = 12 * kMiB;

struct Options {
  int victim_mib = 32;
  int polluter_experts = 0;
  int polluter_panels = 1;
  int trials = 15;
  int evict_mib = 192;
  int cpu = 48;
};

int ParseInt(std::string_view text, const char* name, bool allow_zero = false) {
  const std::string value(text);
  size_t consumed = 0;
  const long parsed = std::stol(value, &consumed);
  const long minimum = allow_zero ? 0 : 1;
  if (consumed != value.size() || parsed < minimum || parsed > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string("invalid ") + name + ": " + value);
  }
  return static_cast<int>(parsed);
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    auto value = [&](const char* name) -> std::string_view {
      if (++index >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + name);
      }
      return argv[index];
    };
    if (argument == "--victim-mib") {
      options.victim_mib = ParseInt(value("--victim-mib"), "victim MiB");
    } else if (argument == "--polluter-experts") {
      options.polluter_experts = ParseInt(value("--polluter-experts"), "polluter experts", true);
    } else if (argument == "--polluter-panels") {
      options.polluter_panels = ParseInt(value("--polluter-panels"), "polluter panels");
    } else if (argument == "--trials") {
      options.trials = ParseInt(value("--trials"), "trials");
    } else if (argument == "--evict-mib") {
      options.evict_mib = ParseInt(value("--evict-mib"), "eviction MiB");
    } else if (argument == "--cpu") {
      options.cpu = ParseInt(value("--cpu"), "CPU", true);
    } else if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: bench_m12_llc_pollution [options]\n"
                << "  --victim-mib MiB --polluter-experts N --polluter-panels N --trials N\n"
                << "  --evict-mib MiB --cpu CPU\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }
  return options;
}

size_t CheckedMultiply(size_t left, size_t right, const char* name) {
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
    throw std::overflow_error(std::string(name) + " size overflow");
  }
  return left * right;
}

void PinCurrentThread(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  const int error = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  if (error != 0) {
    throw std::runtime_error("pthread_setaffinity_np failed: " + std::to_string(error));
  }
}

struct FreeDeleter {
  void operator()(uint16_t* pointer) const { std::free(pointer); }
};

using AlignedBf16 = std::unique_ptr<uint16_t, FreeDeleter>;

AlignedBf16 AllocateBf16(size_t elements) {
  void* allocation = nullptr;
  const size_t bytes = CheckedMultiply(elements, sizeof(uint16_t), "BF16 allocation");
  if (posix_memalign(&allocation, kCacheLineBytes, bytes) != 0) {
    throw std::bad_alloc();
  }
  return AlignedBf16(static_cast<uint16_t*>(allocation));
}

void RunM12(const uint16_t* packed_a, const uint16_t* packed_b, uint16_t* output, int k, int n) {
  gemm_params_t params{};
  params.m = kM;
  params.k = k;
  params.n = n;
  params.lda = k;
  params.ldb = k;
  params.ldc = n;
  moe_sve_w2_packed_bf16_m12(packed_a, packed_b, output, nullptr, &params);
}

void RunW13Range(const uint16_t* packed_a, const uint16_t* packed_b, uint16_t* packed_c) {
  gemm_params_t params{};
  params.m = kM;
  params.k = kHidden;
  params.n = kIntermediate;
  params.lda = kHidden;
  params.ldb = kHidden;
  params.ldc = kIntermediate;
  moe_sve_w13_silu_poly5_packc_m12_rows_opt(packed_a, packed_b, packed_c, nullptr, &params);
}

void RunPolluterExpert(const uint16_t* packed_weights, const uint16_t* a13, uint16_t* intermediate, uint16_t* output) {
  constexpr size_t w13_range_elements = static_cast<size_t>(kHidden) * kIntermediate;
  constexpr size_t w2_elements = static_cast<size_t>(kIntermediate) * kHidden;
  static_assert((2 * w13_range_elements + w2_elements) * sizeof(uint16_t) == kExpertWeightBytes);
  RunW13Range(a13, packed_weights, intermediate);
  RunW13Range(a13, packed_weights + w13_range_elements, intermediate + static_cast<size_t>(kIntermediate) * 6);
  RunM12(intermediate, packed_weights + 2 * w13_range_elements, output, kIntermediate, kHidden);
}

uint64_t ScanCacheLines(const uint16_t* data, size_t bytes) {
  constexpr size_t elements_per_line = kCacheLineBytes / sizeof(uint16_t);
  const volatile uint16_t* volatile_data = data;
  const size_t elements = bytes / sizeof(uint16_t);
  uint64_t checksum = 0;
  for (size_t index = 0; index < elements; index += elements_per_line) {
    checksum += volatile_data[index];
  }
  return checksum;
}

void InitializePointerChain(uint16_t* data, size_t bytes, uint32_t seed) {
  constexpr size_t words_per_line = kCacheLineBytes / sizeof(uint32_t);
  const size_t lines = bytes / kCacheLineBytes;
  if (lines == 0 || lines > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("victim line count is outside the pointer-chain range");
  }
  std::vector<uint32_t> order(lines);
  std::iota(order.begin(), order.end(), uint32_t{0});
  std::mt19937 generator(seed);
  std::shuffle(order.begin(), order.end(), generator);
  const auto zero = std::find(order.begin(), order.end(), uint32_t{0});
  std::iter_swap(order.begin(), zero);
  uint32_t* words = reinterpret_cast<uint32_t*>(data);
  for (size_t index = 0; index < lines; ++index) {
    words[static_cast<size_t>(order[index]) * words_per_line] = order[(index + 1) % lines];
  }
}

uint32_t ProbePointerChain(const uint16_t* data, size_t bytes) {
  constexpr size_t words_per_line = kCacheLineBytes / sizeof(uint32_t);
  const size_t lines = bytes / kCacheLineBytes;
  const volatile uint32_t* words = reinterpret_cast<const volatile uint32_t*>(data);
  uint32_t line = 0;
  for (size_t index = 0; index < lines; ++index) {
    line = words[static_cast<size_t>(line) * words_per_line];
  }
  return line;
}

double TimePointerChain(const uint16_t* data, size_t bytes, uint64_t& checksum) {
  const auto begin = Clock::now();
  checksum += static_cast<uint64_t>(ProbePointerChain(data, bytes)) + 1;
  const auto end = Clock::now();
  return std::chrono::duration<double>(end - begin).count();
}

double Median(std::vector<double> values) {
  if (values.empty()) {
    throw std::invalid_argument("cannot summarize empty samples");
  }
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) * 0.5 : values[middle];
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    PinCurrentThread(options.cpu);

    const size_t victim_bytes = static_cast<size_t>(options.victim_mib) * kMiB;
    const size_t victim_elements = victim_bytes / sizeof(uint16_t);
    const size_t all_victim_elements =
        CheckedMultiply(victim_elements, static_cast<size_t>(options.trials), "victim copies");
    const size_t polluter_bytes = CheckedMultiply(
        CheckedMultiply(static_cast<size_t>(options.trials), options.polluter_experts, "polluter trials"),
        kExpertWeightBytes, "polluter weights");
    const size_t polluter_elements = polluter_bytes / sizeof(uint16_t);
    const size_t evict_bytes = static_cast<size_t>(options.evict_mib) * kMiB;
    const size_t evict_elements = evict_bytes / sizeof(uint16_t);

    AlignedBf16 victims = AllocateBf16(all_victim_elements);
    AlignedBf16 polluters = AllocateBf16(std::max<size_t>(1, polluter_elements));
    AlignedBf16 eviction = AllocateBf16(evict_elements);
    std::fill_n(victims.get(), all_victim_elements, 0x3b80u);
    std::fill_n(polluters.get(), std::max<size_t>(1, polluter_elements), 0x3b00u);
    std::fill_n(eviction.get(), evict_elements, 0x3a80u);
    for (int trial = 0; trial < options.trials; ++trial) {
      InitializePointerChain(victims.get() + static_cast<size_t>(trial) * victim_elements, victim_bytes,
                             0x5a17u + static_cast<uint32_t>(trial));
    }

    std::vector<uint16_t> polluter_a13(static_cast<size_t>(kM) * kHidden, 0x3c00u);
    std::vector<uint16_t> polluter_intermediate(static_cast<size_t>(kM) * kIntermediate, 0);
    std::vector<uint16_t> polluter_output(static_cast<size_t>(kM) * kHidden, 0);

    uint64_t checksum = 0;
    std::vector<double> cold_samples;
    std::vector<double> hot_samples;
    std::vector<double> post_samples;
    cold_samples.reserve(static_cast<size_t>(options.trials));
    hot_samples.reserve(static_cast<size_t>(options.trials));
    post_samples.reserve(static_cast<size_t>(options.trials));

    const size_t weights_per_trial = CheckedMultiply(static_cast<size_t>(options.polluter_experts),
                                                     kExpertWeightBytes / sizeof(uint16_t), "trial weights");
    for (int trial = 0; trial < options.trials; ++trial) {
      checksum += ScanCacheLines(eviction.get(), evict_bytes);
      const uint16_t* victim = victims.get() + static_cast<size_t>(trial) * victim_elements;
      cold_samples.push_back(TimePointerChain(victim, victim_bytes, checksum));
      for (int promotion = 0; promotion < 10; ++promotion) {
        checksum += static_cast<uint64_t>(ProbePointerChain(victim, victim_bytes)) + 1;
      }
      hot_samples.push_back(TimePointerChain(victim, victim_bytes, checksum));

      const uint16_t* trial_weights = polluters.get() + static_cast<size_t>(trial) * weights_per_trial;
      for (int expert = 0; expert < options.polluter_experts; ++expert) {
        const uint16_t* expert_weights =
            trial_weights + static_cast<size_t>(expert) * kExpertWeightBytes / sizeof(uint16_t);
        for (int panel = 0; panel < options.polluter_panels; ++panel) {
          RunPolluterExpert(expert_weights, polluter_a13.data(), polluter_intermediate.data(), polluter_output.data());
        }
      }
      post_samples.push_back(TimePointerChain(victim, victim_bytes, checksum));
    }
    if (checksum == 0 || (options.polluter_experts > 0 && polluter_output.front() == 0)) {
      throw std::runtime_error("benchmark outputs were not materialized");
    }

    const double cold = Median(cold_samples);
    const double hot = Median(hot_samples);
    const double post = Median(post_samples);
    const double eviction_fraction = cold > hot ? std::clamp((post - hot) / (cold - hot), 0.0, 1.0) : 0.0;
    std::cout << std::fixed << std::setprecision(4) << "victim_mib=" << options.victim_mib
              << " polluter_experts=" << options.polluter_experts << " polluter_mib=" << options.polluter_experts * 12
              << " polluter_panels=" << options.polluter_panels << " trials=" << options.trials
              << " cold_ms=" << cold * 1.0e3 << " hot_ms=" << hot * 1.0e3 << " post_ms=" << post * 1.0e3
              << " post_over_hot=" << post / hot << " eviction_fraction=" << eviction_fraction << '\n'
              << "RESULT_JSON {\"victim_mib\":" << options.victim_mib
              << ",\"polluter_experts\":" << options.polluter_experts
              << ",\"polluter_mib\":" << options.polluter_experts * 12
              << ",\"polluter_panels\":" << options.polluter_panels << ",\"trials\":" << options.trials
              << ",\"cold_ms\":" << cold * 1.0e3 << ",\"hot_ms\":" << hot * 1.0e3 << ",\"post_ms\":" << post * 1.0e3
              << ",\"post_over_hot\":" << post / hot << ",\"eviction_fraction\":" << eviction_fraction << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
