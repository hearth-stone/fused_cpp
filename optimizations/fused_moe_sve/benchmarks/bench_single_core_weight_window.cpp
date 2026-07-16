#include <algorithm>
#include <arm_sve.h>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include "gemm_params.h"

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE) || !defined(__ARM_FEATURE_BF16)
#error "bench_single_core_weight_window requires AArch64 SVE BF16"
#endif

extern "C" {
void moe_sve_w2_packed_bf16_m12(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
}

namespace {

constexpr size_t kCacheLineBytes = 64;
constexpr uint16_t kOutputSentinel = 0x7fc1;

struct Options {
  int m = 120;
  int k = 4096;
  int n = 12288;
  int warmup = 2;
  int runs = 7;
  int cpu = 0;
  int cold_tail_mib = 192;
  bool stop_before_run = false;
  bool check = false;
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
    if (argument == "--m") {
      options.m = ParseInt(value("--m"), "M");
    } else if (argument == "--k") {
      options.k = ParseInt(value("--k"), "K");
    } else if (argument == "--n") {
      options.n = ParseInt(value("--n"), "N");
    } else if (argument == "--warmup") {
      options.warmup = ParseInt(value("--warmup"), "warmup", true);
    } else if (argument == "--runs") {
      options.runs = ParseInt(value("--runs"), "runs");
    } else if (argument == "--cpu") {
      options.cpu = ParseInt(value("--cpu"), "CPU", true);
    } else if (argument == "--cold-tail-mib") {
      options.cold_tail_mib = ParseInt(value("--cold-tail-mib"), "cold tail MiB", true);
    } else if (argument == "--stop-before-run") {
      options.stop_before_run = true;
    } else if (argument == "--check") {
      options.check = true;
    } else if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: bench_single_core_weight_window [options]\n"
                << "  --m M --k K --n N --warmup N --runs N\n"
                << "  --cpu CPU --cold-tail-mib MiB\n"
                << "  --stop-before-run --check\n";
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

size_t DivideRoundUp(size_t value, size_t divisor) {
  return value / divisor + static_cast<size_t>(value % divisor != 0);
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

AlignedBf16 AllocateAlignedBf16(size_t elements) {
  void* allocation = nullptr;
  const size_t bytes = CheckedMultiply(elements, sizeof(uint16_t), "aligned BF16 allocation");
  if (posix_memalign(&allocation, kCacheLineBytes, bytes) != 0) {
    throw std::bad_alloc();
  }
  return AlignedBf16(static_cast<uint16_t*>(allocation));
}

void RunGemm(const Options& options, const uint16_t* packed_a, const uint16_t* packed_b, uint16_t* output) {
  gemm_params_t params{};
  params.m = 12;
  params.k = options.k;
  params.n = options.n;
  params.lda = options.k;
  params.ldb = options.k;
  params.ldc = options.n;
  for (int row = 0; row < options.m; row += 12) {
    moe_sve_w2_packed_bf16_m12(packed_a + static_cast<int64_t>(row) * options.k, packed_b,
                               output + static_cast<int64_t>(row) * options.n, nullptr, &params);
  }
}

struct Stats {
  double median_seconds = 0.0;
  double minimum_seconds = 0.0;
  double mean_seconds = 0.0;
};

Stats Summarize(std::vector<double> values) {
  if (values.empty()) {
    throw std::invalid_argument("cannot summarize empty samples");
  }
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  const double minimum = *std::min_element(values.begin(), values.end());
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  const double median = values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) * 0.5 : values[middle];
  return {median, minimum, mean};
}

void CheckOutput(const std::vector<uint16_t>& output) {
  bool saw_nonzero = false;
  for (uint16_t value : output) {
    if (value == kOutputSentinel || (value & 0x7f80u) == 0x7f80u) {
      throw std::runtime_error("kernel left a sentinel or non-finite BF16 output");
    }
    saw_nonzero = saw_nonzero || value != 0;
  }
  if (!saw_nonzero) {
    throw std::runtime_error("kernel produced an all-zero output");
  }
}

uint64_t ScanColdTail(const uint16_t* tail, size_t elements) {
  constexpr size_t kElementsPerCacheLine = kCacheLineBytes / sizeof(uint16_t);
  const volatile uint16_t* volatile_tail = tail;
  uint64_t checksum = 0;
  for (size_t index = 0; index < elements; index += kElementsPerCacheLine) {
    checksum += volatile_tail[index];
  }
  return checksum;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    const int n_tile = static_cast<int>(svcnth());
    if (options.m % 12 != 0 || options.k % 8 != 0 || options.n % n_tile != 0) {
      throw std::invalid_argument("M must align to 12, K to 8, and N to the runtime SVE n_tile");
    }

    PinCurrentThread(options.cpu);
    const size_t weight_elements = CheckedMultiply(static_cast<size_t>(options.k), options.n, "weight");
    const size_t weight_bytes = CheckedMultiply(weight_elements, sizeof(uint16_t), "weight");
    const size_t used_copies = static_cast<size_t>(options.warmup) + options.runs;
    const size_t cold_tail_bytes = static_cast<size_t>(options.cold_tail_mib) << 20;
    const size_t cold_tail_copies = std::max<size_t>(1, DivideRoundUp(cold_tail_bytes, weight_bytes));
    const size_t allocated_copies = used_copies + cold_tail_copies;
    const size_t all_weight_elements = CheckedMultiply(weight_elements, allocated_copies, "all weights");

    AlignedBf16 weights = AllocateAlignedBf16(all_weight_elements);
    for (size_t copy = 0; copy < allocated_copies; ++copy) {
      const uint16_t value = static_cast<uint16_t>(0x3b80u + copy % 32);
      std::fill_n(weights.get() + copy * weight_elements, weight_elements, value);
    }

    std::vector<uint16_t> packed_a(CheckedMultiply(static_cast<size_t>(options.m), options.k, "packed A"), 0x3c00u);
    std::vector<uint16_t> output(CheckedMultiply(static_cast<size_t>(options.m), options.n, "output"), kOutputSentinel);
    const size_t cold_tail_elements = CheckedMultiply(cold_tail_copies, weight_elements, "cold tail");
    const uint64_t cold_tail_checksum = ScanColdTail(weights.get() + used_copies * weight_elements, cold_tail_elements);
    if (cold_tail_checksum == 0) {
      throw std::runtime_error("cold-tail cache scan produced a zero checksum");
    }

    std::cout << std::fixed << std::setprecision(4) << "config M=" << options.m << " K=" << options.k
              << " N=" << options.n << " n_tile=" << n_tile << " cpu=" << options.cpu << " panels=" << options.m / 12
              << " weight_mib=" << weight_bytes / static_cast<double>(1ULL << 20)
              << " used_weight_copies=" << used_copies << " allocated_weight_copies=" << allocated_copies
              << " cold_tail_mib=" << cold_tail_copies * weight_bytes / static_cast<double>(1ULL << 20)
              << " cold_tail_checksum=" << cold_tail_checksum << " allocated_gib="
              << CheckedMultiply(all_weight_elements, sizeof(uint16_t), "all weights") / static_cast<double>(1ULL << 30)
              << '\n';

    if (options.stop_before_run) {
      std::cout << "profiler_ready pid=" << getpid() << '\n' << std::flush;
      if (std::raise(SIGSTOP) != 0) {
        throw std::runtime_error("failed to stop before the measured region");
      }
    }

    size_t copy = 0;
    for (int warmup = 0; warmup < options.warmup; ++warmup, ++copy) {
      RunGemm(options, packed_a.data(), weights.get() + copy * weight_elements, output.data());
    }

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(options.runs));
    for (int run = 0; run < options.runs; ++run, ++copy) {
      const auto begin = std::chrono::steady_clock::now();
      RunGemm(options, packed_a.data(), weights.get() + copy * weight_elements, output.data());
      const auto end = std::chrono::steady_clock::now();
      samples.push_back(std::chrono::duration<double>(end - begin).count());
    }
    if (copy != used_copies) {
      throw std::logic_error("weight copy accounting mismatch");
    }

    CheckOutput(output);
    const Stats stats = Summarize(samples);
    const double flops = 2.0 * options.m * options.k * options.n;
    const double gflops = flops / stats.median_seconds / 1.0e9;
    std::cout << "median_ms=" << stats.median_seconds * 1.0e3 << " min_ms=" << stats.minimum_seconds * 1.0e3
              << " mean_ms=" << stats.mean_seconds * 1.0e3 << " gflops=" << gflops << '\n'
              << "RESULT_JSON {\"m\":" << options.m << ",\"k\":" << options.k << ",\"n\":" << options.n
              << ",\"panels\":" << options.m / 12 << ",\"weight_bytes\":" << weight_bytes
              << ",\"used_weight_copies\":" << used_copies << ",\"allocated_weight_copies\":" << allocated_copies
              << ",\"cold_tail_bytes\":" << cold_tail_copies * weight_bytes
              << ",\"median_ms\":" << stats.median_seconds * 1.0e3 << ",\"gflops\":" << gflops << "}\n";
    if (options.check) {
      std::cout << "check: PASS\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
