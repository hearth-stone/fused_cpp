#include <algorithm>
#include <arm_sve.h>
#include <array>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include "gemm_params.h"

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE) || !defined(__ARM_FEATURE_BF16)
#error "bench_msmall_kblock requires AArch64 SVE BF16"
#endif

extern "C" {
void moe_sve_w2_packed_bf16(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w2_packed_bf16_m4(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w2_packed_bf16_m2(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w2_packed_bf16_m1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m8_bf16_kblock_packed(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m4_bf16_kblock_packed(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m2_bf16_kblock_packed(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m1_bf16_kblock_packed(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
}

namespace {

using Clock = std::chrono::steady_clock;
using Kernel = void (*)(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);

constexpr size_t kCacheLineBytes = 64;
constexpr size_t kKiB = size_t{1} << 10;
constexpr size_t kMiB = size_t{1} << 20;
constexpr size_t kWeightGuardBytes = 4096;
constexpr size_t kFixedColorStrideBytes = 16 * kKiB;
constexpr uint16_t kOutputSentinel = 0x7fc1;

struct ExperimentParams {
  gemm_params_t gemm;
  int k_block;
};

static_assert(offsetof(ExperimentParams, k_block) == 24);

struct HeightSpec {
  int m;
  int scratch_rows;
  const char* name;
  Kernel baseline;
  Kernel kblock;
};

constexpr std::array<HeightSpec, 4> kHeightSpecs{{
    {8, 8, "m8", moe_sve_w2_packed_bf16, moe_sve_m8_bf16_kblock_packed},
    {4, 4, "m4", moe_sve_w2_packed_bf16_m4, moe_sve_m4_bf16_kblock_packed},
    {2, 2, "m2", moe_sve_w2_packed_bf16_m2, moe_sve_m2_bf16_kblock_packed},
    {1, 2, "m1", moe_sve_w2_packed_bf16_m1, moe_sve_m1_bf16_kblock_packed},
}};

struct Options {
  std::string m = "all";
  std::string variant = "kblock";
  int k = 4096;
  int n = 512;
  int k_block = 1024;
  int warmup = 5;
  int runs = 51;
  int cpu = 48;
  int cold_tail_mib = 192;
  int weight_color = -1;
  bool check_only = false;
  bool prewarm_a = false;
  bool stop_before_run = false;
  bool unique_scratch = false;
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
      options.m = value("--m");
    } else if (argument == "--variant") {
      options.variant = value("--variant");
    } else if (argument == "--k") {
      options.k = ParseInt(value("--k"), "K");
    } else if (argument == "--n") {
      options.n = ParseInt(value("--n"), "N");
    } else if (argument == "--k-block") {
      options.k_block = ParseInt(value("--k-block"), "Kc");
    } else if (argument == "--warmup") {
      options.warmup = ParseInt(value("--warmup"), "warmup", true);
    } else if (argument == "--runs") {
      options.runs = ParseInt(value("--runs"), "runs");
    } else if (argument == "--cpu") {
      options.cpu = ParseInt(value("--cpu"), "CPU", true);
    } else if (argument == "--cold-tail-mib") {
      options.cold_tail_mib = ParseInt(value("--cold-tail-mib"), "cold tail MiB", true);
    } else if (argument == "--weight-color") {
      options.weight_color = ParseInt(value("--weight-color"), "weight color", true);
      if (options.weight_color > 3) {
        throw std::invalid_argument("weight color must be in [0, 3]");
      }
    } else if (argument == "--check-only") {
      options.check_only = true;
    } else if (argument == "--prewarm-a") {
      options.prewarm_a = true;
    } else if (argument == "--stop-before-run") {
      options.stop_before_run = true;
    } else if (argument == "--unique-scratch") {
      options.unique_scratch = true;
    } else if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: bench_msmall_kblock [options]\n"
                << "  --m all|8|4|2|1 --variant baseline|kblock\n"
                << "  --k K --n N --k-block Kc\n"
                << "  --warmup N --runs N --cpu CPU --cold-tail-mib MiB\n"
                << "  --weight-color 0..3\n"
                << "  --check-only --prewarm-a --stop-before-run --unique-scratch\n";
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
  const size_t allocation_elements = std::max<size_t>(1, elements);
  const size_t bytes = CheckedMultiply(allocation_elements, sizeof(uint16_t), "BF16 allocation");
  if (posix_memalign(&allocation, kCacheLineBytes, bytes) != 0) {
    throw std::bad_alloc();
  }
  return AlignedBf16(static_cast<uint16_t*>(allocation));
}

uint16_t FloatToBf16(float value) {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

void FillPattern(uint16_t* data, size_t elements, uint32_t multiplier, float scale) {
  for (size_t index = 0; index < elements; ++index) {
    const int centered = static_cast<int>((index * multiplier + 11) % 31) - 15;
    data[index] = FloatToBf16(static_cast<float>(centered == 0 ? 1 : centered) * scale);
  }
}

void RepackBKBlocked(const uint16_t* source, uint16_t* destination, int k, int n, int k_block) {
  const size_t vl_bytes = svcntb();
  const size_t n_tile = svcnth();
  const auto* source_bytes = reinterpret_cast<const std::byte*>(source);
  auto* destination_bytes = reinterpret_cast<std::byte*>(destination);
  const size_t n_tiles = static_cast<size_t>(n) / n_tile;
  size_t destination_offset = 0;
  for (int k_start = 0; k_start < k; k_start += k_block) {
    const size_t chunk = static_cast<size_t>(std::min(k_block, k - k_start));
    const size_t chunk_bytes = chunk * vl_bytes;
    for (size_t tile = 0; tile < n_tiles; ++tile) {
      const size_t source_offset = (tile * static_cast<size_t>(k) + static_cast<size_t>(k_start)) * vl_bytes;
      std::memcpy(destination_bytes + destination_offset, source_bytes + source_offset, chunk_bytes);
      destination_offset += chunk_bytes;
    }
  }
  const size_t expected_bytes =
      CheckedMultiply(CheckedMultiply(static_cast<size_t>(k), n, "packed B"), sizeof(uint16_t), "packed B");
  if (destination_offset != expected_bytes) {
    throw std::logic_error("K-blocked B byte accounting mismatch");
  }
}

ExperimentParams MakeParams(const HeightSpec& height, const Options& options) {
  return {{height.m, options.k, options.n, options.k, options.k, options.n}, options.k_block};
}

void Invoke(Kernel kernel, const uint16_t* packed_a, const uint16_t* packed_b, uint16_t* output, uint16_t* scratch,
            const ExperimentParams& params) {
  kernel(packed_a, packed_b, output, scratch, &params.gemm);
}

void CheckFiniteOutput(const std::vector<uint16_t>& output) {
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

void CheckCorrectness(const HeightSpec& height, const Options& options) {
  const size_t a_elements = CheckedMultiply(size_t{8}, static_cast<size_t>(options.k), "packed A");
  const size_t b_elements = CheckedMultiply(static_cast<size_t>(options.k), options.n, "packed B");
  const size_t c_elements = CheckedMultiply(static_cast<size_t>(height.m), options.n, "output");
  const size_t scratch_elements = CheckedMultiply(
      CheckedMultiply(static_cast<size_t>(height.scratch_rows), options.n, "partial C"), size_t{2}, "FP32 partial C");
  AlignedBf16 packed_a = AllocateBf16(a_elements);
  AlignedBf16 packed_b = AllocateBf16(b_elements);
  AlignedBf16 kblocked_b = AllocateBf16(b_elements);
  AlignedBf16 scratch = AllocateBf16(scratch_elements);
  FillPattern(packed_a.get(), a_elements, 17, 1.0f / 256.0f);
  FillPattern(packed_b.get(), b_elements, 13, 1.0f / 512.0f);
  RepackBKBlocked(packed_b.get(), kblocked_b.get(), options.k, options.n, options.k_block);
  std::fill_n(scratch.get(), scratch_elements, uint16_t{0});

  const ExperimentParams params = MakeParams(height, options);
  std::vector<uint16_t> reference(c_elements, kOutputSentinel);
  std::vector<uint16_t> actual(c_elements, kOutputSentinel);
  Invoke(height.baseline, packed_a.get(), packed_b.get(), reference.data(), nullptr, params);
  Invoke(height.kblock, packed_a.get(), kblocked_b.get(), actual.data(), scratch.get(), params);
  CheckFiniteOutput(reference);
  CheckFiniteOutput(actual);
  const auto mismatch = std::mismatch(reference.begin(), reference.end(), actual.begin());
  if (mismatch.first != reference.end()) {
    const size_t index = static_cast<size_t>(mismatch.first - reference.begin());
    throw std::runtime_error(std::string(height.name) + " Kc=" + std::to_string(options.k_block) +
                             " differs from baseline at output element " + std::to_string(index));
  }
  std::cout << "correctness height=" << height.name << " Kc=" << options.k_block << " exact=PASS\n";
}

uint64_t ScanCacheLines(const uint16_t* data, size_t elements) {
  constexpr size_t elements_per_line = kCacheLineBytes / sizeof(uint16_t);
  const volatile uint16_t* volatile_data = data;
  uint64_t checksum = 0;
  for (size_t index = 0; index < elements; index += elements_per_line) {
    checksum += volatile_data[index];
  }
  return checksum;
}

double Quantile(const std::vector<double>& sorted, double quantile) {
  const double position = quantile * static_cast<double>(sorted.size() - 1);
  const size_t lower = static_cast<size_t>(position);
  const size_t upper = std::min(lower + 1, sorted.size() - 1);
  const double fraction = position - static_cast<double>(lower);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
}

struct Stats {
  double median;
  double p10;
  double p90;
};

Stats Summarize(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return {Quantile(values, 0.5), Quantile(values, 0.1), Quantile(values, 0.9)};
}

void RunHeight(const HeightSpec& height, const Options& options) {
  const bool use_kblock = options.variant == "kblock";
  const Kernel kernel = use_kblock ? height.kblock : height.baseline;
  const size_t a_elements = CheckedMultiply(size_t{8}, static_cast<size_t>(options.k), "packed A");
  const size_t weight_elements = CheckedMultiply(static_cast<size_t>(options.k), options.n, "packed B");
  const size_t output_elements = CheckedMultiply(static_cast<size_t>(height.m), options.n, "output");
  const size_t scratch_elements = CheckedMultiply(
      CheckedMultiply(static_cast<size_t>(height.scratch_rows), options.n, "partial C"), size_t{2}, "FP32 partial C");
  const size_t rounds = static_cast<size_t>(options.warmup) + options.runs;
  const size_t weight_guard_bytes = options.weight_color >= 0 ? kFixedColorStrideBytes : kWeightGuardBytes;
  const size_t weight_base_offset_bytes =
      options.weight_color >= 0 ? static_cast<size_t>(options.weight_color) * 4 * kKiB : 0;
  const size_t weight_base_offset_elements = weight_base_offset_bytes / sizeof(uint16_t);
  const size_t weight_stride_elements = weight_elements + weight_guard_bytes / sizeof(uint16_t);
  const size_t used_weight_elements =
      weight_base_offset_elements + CheckedMultiply(rounds, weight_stride_elements, "timed weights");
  const size_t cold_tail_elements =
      CheckedMultiply(static_cast<size_t>(options.cold_tail_mib), kMiB, "cold tail") / sizeof(uint16_t);
  const size_t all_weight_elements = used_weight_elements + cold_tail_elements;

  AlignedBf16 packed_a = AllocateBf16(a_elements);
  AlignedBf16 weights = AllocateBf16(all_weight_elements);
  FillPattern(packed_a.get(), a_elements, 17, 1.0f / 256.0f);
  for (size_t copy = 0; copy < rounds; ++copy) {
    const uint16_t value = static_cast<uint16_t>(0x3a40u + copy % 32);
    uint16_t* const weight = weights.get() + weight_base_offset_elements + copy * weight_stride_elements;
    std::fill_n(weight, weight_elements, value);
    std::fill_n(weight + weight_elements, weight_guard_bytes / sizeof(uint16_t), uint16_t{0});
  }
  std::fill_n(weights.get() + used_weight_elements, cold_tail_elements, uint16_t{0x3b00});

  std::vector<uint16_t> output(output_elements, kOutputSentinel);
  AlignedBf16 scratch;
  AlignedBf16 unique_scratches;
  if (use_kblock) {
    if (options.unique_scratch) {
      unique_scratches = AllocateBf16(CheckedMultiply(rounds, scratch_elements, "all partial C copies"));
      std::fill_n(unique_scratches.get(), rounds * scratch_elements, uint16_t{0});
    } else {
      scratch = AllocateBf16(scratch_elements);
      std::fill_n(scratch.get(), scratch_elements, uint16_t{0});
    }
  }

  const uint64_t cold_checksum = ScanCacheLines(weights.get() + used_weight_elements, cold_tail_elements);
  if (cold_tail_elements != 0 && cold_checksum == 0) {
    throw std::runtime_error("cold-tail scan produced a zero checksum");
  }

  const size_t weight_bytes = CheckedMultiply(weight_elements, sizeof(uint16_t), "weight bytes");
  const size_t a_slice_bytes = CheckedMultiply(static_cast<size_t>(options.k_block), size_t{16}, "A slice");
  const size_t b_slice_bytes =
      CheckedMultiply(static_cast<size_t>(options.k_block), static_cast<size_t>(svcntb()), "B slice");
  const size_t partial_bytes =
      CheckedMultiply(CheckedMultiply(static_cast<size_t>(height.scratch_rows), options.n, "partial bytes"),
                      sizeof(float), "partial bytes");
  const double allocated_gib =
      CheckedMultiply(all_weight_elements, sizeof(uint16_t), "all weight bytes") / static_cast<double>(size_t{1} << 30);
  std::cout << std::fixed << std::setprecision(4) << "config height=" << height.name << " M=" << height.m
            << " K=" << options.k << " N=" << options.n << " Kc=" << options.k_block << " n_tile=" << svcnth()
            << " variant=" << options.variant << " cpu=" << options.cpu
            << " weight_mib=" << weight_bytes / static_cast<double>(kMiB) << " warmup=" << options.warmup
            << " runs=" << options.runs << " cold_tail_mib=" << options.cold_tail_mib
            << " weight_color=" << options.weight_color << " unique_scratch=" << options.unique_scratch
            << " prewarm_a=" << options.prewarm_a << " a_slice_kib=" << a_slice_bytes / static_cast<double>(kKiB)
            << " b_slice_kib=" << b_slice_bytes / static_cast<double>(kKiB)
            << " partial_kib=" << partial_bytes / static_cast<double>(kKiB) << " allocated_gib=" << allocated_gib
            << '\n';

  if (options.stop_before_run) {
    std::cout << "profiler_ready pid=" << getpid() << '\n' << std::flush;
    if (std::raise(SIGSTOP) != 0) {
      throw std::runtime_error("failed to stop for profiler attach");
    }
  }

  const ExperimentParams params = MakeParams(height, options);
  size_t copy = 0;
  uint64_t a_checksum = 0;
  for (int warmup = 0; warmup < options.warmup; ++warmup, ++copy) {
    if (options.prewarm_a) {
      a_checksum += ScanCacheLines(packed_a.get(), a_elements);
    }
    uint16_t* const invocation_scratch =
        use_kblock ? (options.unique_scratch ? unique_scratches.get() + copy * scratch_elements : scratch.get())
                   : nullptr;
    Invoke(kernel, packed_a.get(), weights.get() + weight_base_offset_elements + copy * weight_stride_elements,
           output.data(), invocation_scratch, params);
  }

  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(options.runs));
  for (int run = 0; run < options.runs; ++run, ++copy) {
    if (options.prewarm_a) {
      a_checksum += ScanCacheLines(packed_a.get(), a_elements);
    }
    uint16_t* const invocation_scratch =
        use_kblock ? (options.unique_scratch ? unique_scratches.get() + copy * scratch_elements : scratch.get())
                   : nullptr;
    const auto begin = Clock::now();
    Invoke(kernel, packed_a.get(), weights.get() + weight_base_offset_elements + copy * weight_stride_elements,
           output.data(), invocation_scratch, params);
    const auto end = Clock::now();
    samples.push_back(std::chrono::duration<double>(end - begin).count());
  }
  if (copy != rounds) {
    throw std::logic_error("weight copy accounting mismatch");
  }

  const Stats stats = Summarize(std::move(samples));
  const double flops = 2.0 * height.m * options.k * options.n;
  const double gflops = flops / stats.median / 1.0e9;
  const double b_gbs = static_cast<double>(weight_bytes) / stats.median / 1.0e9;
  const uint64_t output_checksum = ScanCacheLines(output.data(), output.size());
  const std::string variant_name = use_kblock ? "kblock_packed_" + std::to_string(options.k_block) : "baseline_ld1h";
  std::cout << "variant=" << variant_name << " median_ms=" << stats.median * 1.0e3 << " p10_ms=" << stats.p10 * 1.0e3
            << " p90_ms=" << stats.p90 * 1.0e3 << " GFLOP/s=" << gflops << " B_GB/s=" << b_gbs << '\n';
  std::cout << "RESULT_JSON {\"height\":\"" << height.name << "\",\"m\":" << height.m << ",\"variant\":\""
            << variant_name << "\",\"k_block\":" << options.k_block << ",\"median_ms\":" << stats.median * 1.0e3
            << ",\"p10_ms\":" << stats.p10 * 1.0e3 << ",\"p90_ms\":" << stats.p90 * 1.0e3 << ",\"gflops\":" << gflops
            << ",\"b_gbs\":" << b_gbs << ",\"a_slice_kib\":" << a_slice_bytes / static_cast<double>(kKiB)
            << ",\"b_slice_kib\":" << b_slice_bytes / static_cast<double>(kKiB)
            << ",\"partial_kib\":" << partial_bytes / static_cast<double>(kKiB) << "}\n";
  std::cout << "output_checksum=" << output_checksum << " a_checksum=" << a_checksum << '\n';
}

std::vector<const HeightSpec*> SelectHeights(const Options& options) {
  if (options.m == "all") {
    std::vector<const HeightSpec*> heights;
    for (const HeightSpec& height : kHeightSpecs) {
      heights.push_back(&height);
    }
    return heights;
  }
  const int requested_m = ParseInt(options.m, "M");
  const auto height = std::find_if(kHeightSpecs.begin(), kHeightSpecs.end(),
                                   [&](const HeightSpec& spec) { return spec.m == requested_m; });
  if (height == kHeightSpecs.end()) {
    throw std::invalid_argument("--m must be all, 8, 4, 2, or 1");
  }
  return {&*height};
}

void ValidateOptions(const Options& options) {
  if (options.variant != "baseline" && options.variant != "kblock") {
    throw std::invalid_argument("--variant must be baseline or kblock");
  }
  if (options.k % 8 != 0 || options.k_block % 8 != 0) {
    throw std::invalid_argument("K and Kc must be multiples of 8");
  }
  if (options.n % static_cast<int>(svcnth()) != 0) {
    throw std::invalid_argument("N must align to the runtime SVE n_tile");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    ValidateOptions(options);
    PinCurrentThread(options.cpu);
    for (const HeightSpec* height : SelectHeights(options)) {
      CheckCorrectness(*height, options);
      if (!options.check_only) {
        RunHeight(*height, options);
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
