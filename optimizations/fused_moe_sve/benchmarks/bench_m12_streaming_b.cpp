#include <algorithm>
#include <arm_sve.h>
#include <array>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "../cxx17_compat.h"

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include "gemm_params.h"

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE) || !defined(__ARM_FEATURE_BF16)
#error "bench_m12_streaming_b requires AArch64 SVE BF16"
#endif

extern "C" {
void moe_sve_w2_packed_bf16_m12(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_ld1h(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_pldl1strm_256(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_pldl1strm_512(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_pldl1strm_1024(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_pldl1strm_2048(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_pldl2strm_512(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_pldl2strm_1024(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_pldl2strm_2048(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_ldnt1h(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_pldl1strm_2048_x1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_pldl2strm_1024_x1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_pldl3strm_512_x1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                       const gemm_params_t*);
void moe_sve_m12_bf16_pldl3strm_1024_x1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                        const gemm_params_t*);
void moe_sve_m12_bf16_pldl3strm_2048_x1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                        const gemm_params_t*);
void moe_sve_m12_bf16_pldl3strm_4096_x1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                        const gemm_params_t*);
void moe_sve_m12_bf16_kblock_256(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_512(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_768(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_1024(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_256(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_384(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_448(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_512(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_544(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_552(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_560(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_568(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_576(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_640(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_704(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_720(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_736(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_752(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_768(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_784(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_792(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_800(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_816(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_832(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_848(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_864(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_880(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_896(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_960(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_1024(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_1088(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_1152(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_m12_bf16_kblock_packed_1280(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
}

namespace {

using Clock = std::chrono::steady_clock;
using Kernel = void (*)(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);

constexpr int kM = 12;
constexpr size_t kCacheLineBytes = 64;
constexpr size_t kKiB = size_t{1} << 10;
constexpr size_t kMiB = size_t{1} << 20;
constexpr size_t kWeightGuardBytes = 4096;
constexpr size_t kFixedColorStrideBytes = 16 * kKiB;
constexpr uint16_t kOutputSentinel = 0x7fc1;

struct Variant {
  const char* name;
  Kernel kernel;
  bool needs_scratch = false;
  int k_block = 0;
  bool kblocked_b = false;
};

constexpr std::array<Variant, 48> kVariants{{
    {"baseline_ld1h", moe_sve_m12_bf16_ld1h},
    {"pldl1strm_256", moe_sve_m12_bf16_pldl1strm_256},
    {"pldl1strm_512", moe_sve_m12_bf16_pldl1strm_512},
    {"pldl1strm_1024", moe_sve_m12_bf16_pldl1strm_1024},
    {"pldl1strm_2048", moe_sve_m12_bf16_pldl1strm_2048},
    {"pldl2strm_512", moe_sve_m12_bf16_pldl2strm_512},
    {"pldl2strm_1024", moe_sve_m12_bf16_pldl2strm_1024},
    {"pldl2strm_2048", moe_sve_m12_bf16_pldl2strm_2048},
    {"ldnt1h", moe_sve_m12_bf16_ldnt1h},
    {"pldl1strm_2048_x1", moe_sve_m12_bf16_pldl1strm_2048_x1},
    {"pldl2strm_1024_x1", moe_sve_m12_bf16_pldl2strm_1024_x1},
    {"pldl3strm_512_x1", moe_sve_m12_bf16_pldl3strm_512_x1},
    {"pldl3strm_1024_x1", moe_sve_m12_bf16_pldl3strm_1024_x1},
    {"pldl3strm_2048_x1", moe_sve_m12_bf16_pldl3strm_2048_x1},
    {"pldl3strm_4096_x1", moe_sve_m12_bf16_pldl3strm_4096_x1},
    {"kblock_256", moe_sve_m12_bf16_kblock_256, true},
    {"kblock_512", moe_sve_m12_bf16_kblock_512, true},
    {"kblock_768", moe_sve_m12_bf16_kblock_768, true},
    {"kblock_1024", moe_sve_m12_bf16_kblock_1024, true},
    {"kblock_packed_256", moe_sve_m12_bf16_kblock_packed_256, true, 256, true},
    {"kblock_packed_384", moe_sve_m12_bf16_kblock_packed_384, true, 384, true},
    {"kblock_packed_448", moe_sve_m12_bf16_kblock_packed_448, true, 448, true},
    {"kblock_packed_512", moe_sve_m12_bf16_kblock_packed_512, true, 512, true},
    {"kblock_packed_544", moe_sve_m12_bf16_kblock_packed_544, true, 544, true},
    {"kblock_packed_552", moe_sve_m12_bf16_kblock_packed_552, true, 552, true},
    {"kblock_packed_560", moe_sve_m12_bf16_kblock_packed_560, true, 560, true},
    {"kblock_packed_568", moe_sve_m12_bf16_kblock_packed_568, true, 568, true},
    {"kblock_packed_576", moe_sve_m12_bf16_kblock_packed_576, true, 576, true},
    {"kblock_packed_640", moe_sve_m12_bf16_kblock_packed_640, true, 640, true},
    {"kblock_packed_704", moe_sve_m12_bf16_kblock_packed_704, true, 704, true},
    {"kblock_packed_720", moe_sve_m12_bf16_kblock_packed_720, true, 720, true},
    {"kblock_packed_736", moe_sve_m12_bf16_kblock_packed_736, true, 736, true},
    {"kblock_packed_752", moe_sve_m12_bf16_kblock_packed_752, true, 752, true},
    {"kblock_packed_768", moe_sve_m12_bf16_kblock_packed_768, true, 768, true},
    {"kblock_packed_784", moe_sve_m12_bf16_kblock_packed_784, true, 784, true},
    {"kblock_packed_792", moe_sve_m12_bf16_kblock_packed_792, true, 792, true},
    {"kblock_packed_800", moe_sve_m12_bf16_kblock_packed_800, true, 800, true},
    {"kblock_packed_816", moe_sve_m12_bf16_kblock_packed_816, true, 816, true},
    {"kblock_packed_832", moe_sve_m12_bf16_kblock_packed_832, true, 832, true},
    {"kblock_packed_848", moe_sve_m12_bf16_kblock_packed_848, true, 848, true},
    {"kblock_packed_864", moe_sve_m12_bf16_kblock_packed_864, true, 864, true},
    {"kblock_packed_880", moe_sve_m12_bf16_kblock_packed_880, true, 880, true},
    {"kblock_packed_896", moe_sve_m12_bf16_kblock_packed_896, true, 896, true},
    {"kblock_packed_960", moe_sve_m12_bf16_kblock_packed_960, true, 960, true},
    {"kblock_packed_1024", moe_sve_m12_bf16_kblock_packed_1024, true, 1024, true},
    {"kblock_packed_1088", moe_sve_m12_bf16_kblock_packed_1088, true, 1088, true},
    {"kblock_packed_1152", moe_sve_m12_bf16_kblock_packed_1152, true, 1152, true},
    {"kblock_packed_1280", moe_sve_m12_bf16_kblock_packed_1280, true, 1280, true},
}};

struct Shape {
  std::string name;
  int k;
  int n;
};

struct Options {
  std::string shape = "all";
  std::string variants = "all";
  int m = kM;
  int custom_k = 0;
  int custom_n = 0;
  int warmup = 5;
  int runs = 31;
  int cpu = 48;
  int workers = 1;
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
    if (argument == "--shape") {
      options.shape = value("--shape");
    } else if (argument == "--variants") {
      options.variants = value("--variants");
    } else if (argument == "--m") {
      options.m = ParseInt(value("--m"), "M");
    } else if (argument == "--k") {
      options.custom_k = ParseInt(value("--k"), "K");
    } else if (argument == "--n") {
      options.custom_n = ParseInt(value("--n"), "N");
    } else if (argument == "--warmup") {
      options.warmup = ParseInt(value("--warmup"), "warmup", true);
    } else if (argument == "--runs") {
      options.runs = ParseInt(value("--runs"), "runs");
    } else if (argument == "--cpu") {
      options.cpu = ParseInt(value("--cpu"), "CPU", true);
    } else if (argument == "--workers") {
      options.workers = ParseInt(value("--workers"), "workers");
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
      std::cout << "Usage: bench_m12_streaming_b [options]\n"
                << "  --shape all|w13|w2\n"
                << "  --variants all|baseline_ld1h,name...\n"
                << "  --m M                     execute M/12 consecutive M12 panels\n"
                << "  --k K --n N             use one custom shape\n"
                << "  --warmup N --runs N --cpu CPU --workers N --cold-tail-mib MiB\n"
                << "  --weight-color 0..3     hold every cold B at one L1 set phase\n"
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
  if (n % static_cast<int>(n_tile) != 0 || k % 8 != 0 || k_block % 8 != 0) {
    throw std::invalid_argument("K-blocked B requires aligned K, Kc, and N");
  }

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

gemm_params_t MakeParams(const Shape& shape) {
  gemm_params_t params{};
  params.m = kM;
  params.k = shape.k;
  params.n = shape.n;
  params.lda = shape.k;
  params.ldb = shape.k;
  params.ldc = shape.n;
  return params;
}

void Invoke(const Variant& variant, const uint16_t* packed_a, const uint16_t* packed_b, uint16_t* output,
            uint16_t* scratch, const gemm_params_t& params) {
  variant.kernel(packed_a, packed_b, output, variant.needs_scratch ? scratch : nullptr, &params);
}

void InvokeRows(const Variant& variant, const uint16_t* packed_a, const uint16_t* packed_b, uint16_t* output,
                uint16_t* scratch, const gemm_params_t& params, int rows) {
  for (int row = 0; row < rows; row += kM) {
    Invoke(variant, packed_a + static_cast<size_t>(row) * params.k, packed_b,
           output + static_cast<size_t>(row) * params.ldc, scratch, params);
  }
}

void CheckFiniteOutput(const std::vector<uint16_t>& output) {
  bool saw_nonzero = false;
  for (uint16_t value : output) {
    if (value == kOutputSentinel || (value & 0x7f80u) == 0x7f80u) {
      throw std::runtime_error("baseline left a sentinel or non-finite BF16 output");
    }
    saw_nonzero = saw_nonzero || value != 0;
  }
  if (!saw_nonzero) {
    throw std::runtime_error("baseline produced an all-zero output");
  }
}

void CheckCorrectness(const Shape& shape) {
  const size_t a_elements = CheckedMultiply(kM, static_cast<size_t>(shape.k), "packed A");
  const size_t b_elements = CheckedMultiply(static_cast<size_t>(shape.k), shape.n, "packed B");
  const size_t c_elements = CheckedMultiply(kM, static_cast<size_t>(shape.n), "output");
  AlignedBf16 packed_a = AllocateBf16(a_elements);
  AlignedBf16 packed_b = AllocateBf16(b_elements + kWeightGuardBytes / sizeof(uint16_t));
  FillPattern(packed_a.get(), a_elements, 17, 1.0f / 256.0f);
  FillPattern(packed_b.get(), b_elements, 13, 1.0f / 512.0f);
  std::fill_n(packed_b.get() + b_elements, kWeightGuardBytes / sizeof(uint16_t), uint16_t{0});

  const gemm_params_t params = MakeParams(shape);
  std::vector<uint16_t> reference(c_elements, kOutputSentinel);
  const Variant production{"production", moe_sve_w2_packed_bf16_m12};
  Invoke(production, packed_a.get(), packed_b.get(), reference.data(), nullptr, params);
  CheckFiniteOutput(reference);

  for (size_t variant_index = 0; variant_index < kVariants.size(); ++variant_index) {
    std::vector<uint16_t> actual(c_elements, kOutputSentinel);
    AlignedBf16 scratch = AllocateBf16(CheckedMultiply(c_elements, size_t{2}, "FP32 partial C"));
    std::fill_n(scratch.get(), c_elements * 2, kOutputSentinel);
    AlignedBf16 kblocked_b;
    const uint16_t* variant_b = packed_b.get();
    if (kVariants[variant_index].kblocked_b) {
      kblocked_b = AllocateBf16(b_elements + kWeightGuardBytes / sizeof(uint16_t));
      RepackBKBlocked(packed_b.get(), kblocked_b.get(), shape.k, shape.n, kVariants[variant_index].k_block);
      std::fill_n(kblocked_b.get() + b_elements, kWeightGuardBytes / sizeof(uint16_t), uint16_t{0});
      variant_b = kblocked_b.get();
    }
    Invoke(kVariants[variant_index], packed_a.get(), variant_b, actual.data(), scratch.get(), params);
    const auto mismatch = std::mismatch(reference.begin(), reference.end(), actual.begin());
    if (mismatch.first != reference.end()) {
      const size_t index = static_cast<size_t>(mismatch.first - reference.begin());
      throw std::runtime_error(shape.name + ": " + kVariants[variant_index].name +
                               " differs from baseline at output element " + std::to_string(index));
    }
  }
  std::cout << "correctness shape=" << shape.name << " variants=" << kVariants.size() << " exact=PASS\n";
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
  if (sorted.empty()) {
    throw std::invalid_argument("cannot summarize empty samples");
  }
  const double position = quantile * static_cast<double>(sorted.size() - 1);
  const size_t lower = static_cast<size_t>(position);
  const size_t upper = std::min(lower + 1, sorted.size() - 1);
  const double fraction = position - static_cast<double>(lower);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
}

struct Stats {
  double median = 0.0;
  double p10 = 0.0;
  double p90 = 0.0;
};

Stats Summarize(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return {Quantile(values, 0.5), Quantile(values, 0.1), Quantile(values, 0.9)};
}

size_t VariantAtSlot(int round, size_t slot, size_t count) {
  const size_t start = static_cast<size_t>(round) % count;
  if ((round & 1) == 0) {
    return (start + slot) % count;
  }
  return (start + count - slot) % count;
}

void RunShape(const Shape& shape, const Options& options, const std::vector<size_t>& active_variants) {
  const size_t a_elements = CheckedMultiply(options.m, static_cast<size_t>(shape.k), "packed A");
  const size_t weight_elements = CheckedMultiply(static_cast<size_t>(shape.k), shape.n, "packed B");
  const size_t output_elements = CheckedMultiply(options.m, static_cast<size_t>(shape.n), "output");
  const size_t panel_output_elements = CheckedMultiply(kM, static_cast<size_t>(shape.n), "panel output");
  const size_t scratch_elements = CheckedMultiply(panel_output_elements, size_t{2}, "FP32 partial C");
  const size_t rounds = static_cast<size_t>(options.warmup) + options.runs;
  const size_t used_copies = CheckedMultiply(rounds, active_variants.size(), "weight copies");
  const size_t weight_guard_bytes = options.weight_color >= 0 ? kFixedColorStrideBytes : kWeightGuardBytes;
  const size_t weight_base_offset_bytes =
      options.weight_color >= 0 ? static_cast<size_t>(options.weight_color) * 4 * kKiB : 0;
  const size_t weight_base_offset_elements = weight_base_offset_bytes / sizeof(uint16_t);
  const size_t weight_stride_elements = weight_elements + weight_guard_bytes / sizeof(uint16_t);
  const size_t used_weight_elements =
      weight_base_offset_elements + CheckedMultiply(used_copies, weight_stride_elements, "all timed weights");
  const size_t cold_tail_elements =
      CheckedMultiply(static_cast<size_t>(options.cold_tail_mib), kMiB, "cold tail") / sizeof(uint16_t);
  const size_t all_weight_elements = used_weight_elements + cold_tail_elements;

  AlignedBf16 packed_a = AllocateBf16(a_elements);
  AlignedBf16 weights = AllocateBf16(all_weight_elements);
  FillPattern(packed_a.get(), a_elements, 17, 1.0f / 256.0f);
  for (size_t copy = 0; copy < used_copies; ++copy) {
    const uint16_t value = static_cast<uint16_t>(0x3a40u + copy % 32);
    uint16_t* const weight = weights.get() + weight_base_offset_elements + copy * weight_stride_elements;
    std::fill_n(weight, weight_elements, value);
    std::fill_n(weight + weight_elements, weight_guard_bytes / sizeof(uint16_t), uint16_t{0});
  }
  std::fill_n(weights.get() + used_weight_elements, cold_tail_elements, uint16_t{0x3b00});

  std::vector<std::vector<uint16_t>> outputs;
  std::vector<AlignedBf16> scratches;
  AlignedBf16 unique_scratches;
  outputs.reserve(active_variants.size());
  scratches.reserve(active_variants.size());
  for (size_t index = 0; index < active_variants.size(); ++index) {
    outputs.emplace_back(output_elements, kOutputSentinel);
    if (!options.unique_scratch) {
      scratches.emplace_back(AllocateBf16(scratch_elements));
      std::fill_n(scratches.back().get(), scratch_elements, uint16_t{0});
    }
  }
  if (options.unique_scratch) {
    const size_t all_scratch_elements = CheckedMultiply(used_copies, scratch_elements, "all partial C copies");
    unique_scratches = AllocateBf16(all_scratch_elements);
    std::fill_n(unique_scratches.get(), all_scratch_elements, uint16_t{0});
  }

  const uint64_t cold_checksum = ScanCacheLines(weights.get() + used_weight_elements, cold_tail_elements);
  if (cold_tail_elements != 0 && cold_checksum == 0) {
    throw std::runtime_error("cold-tail scan produced a zero checksum");
  }

  const size_t weight_bytes = CheckedMultiply(weight_elements, sizeof(uint16_t), "weight bytes");
  const double allocated_gib =
      CheckedMultiply(all_weight_elements, sizeof(uint16_t), "all weight bytes") / static_cast<double>(size_t{1} << 30);
  std::cout << std::fixed << std::setprecision(4) << "config shape=" << shape.name << " M=" << options.m
            << " K=" << shape.k << " N=" << shape.n << " n_tile=" << svcnth() << " cpu=" << options.cpu
            << " weight_mib=" << weight_bytes / static_cast<double>(kMiB) << " variants=" << active_variants.size()
            << " warmup=" << options.warmup << " runs=" << options.runs << " cold_tail_mib=" << options.cold_tail_mib
            << " weight_color=" << options.weight_color << " unique_scratch=" << options.unique_scratch
            << " prewarm_a=" << options.prewarm_a << " allocated_gib=" << allocated_gib << '\n';

  if (options.stop_before_run) {
    std::cout << "profiler_ready pid=" << getpid() << '\n' << std::flush;
    if (std::raise(SIGSTOP) != 0) {
      throw std::runtime_error("failed to stop for profiler attach");
    }
  }

  const gemm_params_t params = MakeParams(shape);
  size_t copy = 0;
  uint64_t a_checksum = 0;
  for (int round = 0; round < options.warmup; ++round) {
    for (size_t slot = 0; slot < active_variants.size(); ++slot, ++copy) {
      const size_t active_index = VariantAtSlot(round, slot, active_variants.size());
      const size_t variant_index = active_variants[active_index];
      uint16_t* const scratch =
          options.unique_scratch ? unique_scratches.get() + copy * scratch_elements : scratches[active_index].get();
      if (options.prewarm_a) {
        a_checksum += ScanCacheLines(packed_a.get(), a_elements);
      }
      InvokeRows(kVariants[variant_index], packed_a.get(),
                 weights.get() + weight_base_offset_elements + copy * weight_stride_elements,
                 outputs[active_index].data(), scratch, params, options.m);
    }
  }

  std::vector<std::vector<double>> samples(active_variants.size());
  for (auto& variant_samples : samples) {
    variant_samples.reserve(static_cast<size_t>(options.runs));
  }
  for (int run = 0; run < options.runs; ++run) {
    const int order_round = options.warmup + run;
    for (size_t slot = 0; slot < active_variants.size(); ++slot, ++copy) {
      const size_t active_index = VariantAtSlot(order_round, slot, active_variants.size());
      const size_t variant_index = active_variants[active_index];
      uint16_t* const scratch =
          options.unique_scratch ? unique_scratches.get() + copy * scratch_elements : scratches[active_index].get();
      if (options.prewarm_a) {
        a_checksum += ScanCacheLines(packed_a.get(), a_elements);
      }
      const auto begin = Clock::now();
      InvokeRows(kVariants[variant_index], packed_a.get(),
                 weights.get() + weight_base_offset_elements + copy * weight_stride_elements,
                 outputs[active_index].data(), scratch, params, options.m);
      const auto end = Clock::now();
      samples[active_index].push_back(std::chrono::duration<double>(end - begin).count());
    }
  }
  if (copy != used_copies) {
    throw std::logic_error("weight copy accounting mismatch");
  }

  uint64_t output_checksum = 0;
  for (const auto& output : outputs) {
    output_checksum += ScanCacheLines(output.data(), output.size());
  }

  const Stats baseline = Summarize(samples.front());
  const double flops = 2.0 * options.m * shape.k * shape.n;
  const double streamed_weight_bytes = static_cast<double>(weight_bytes) * options.m / kM;
  std::cout << "variant              median_ms    p10_ms    p90_ms   GFLOP/s   B_GB/s  median_gain%  paired_gain%\n";
  for (size_t active_index = 0; active_index < active_variants.size(); ++active_index) {
    const size_t variant_index = active_variants[active_index];
    const Stats stats = Summarize(samples[active_index]);
    std::vector<double> paired_gains;
    paired_gains.reserve(static_cast<size_t>(options.runs));
    for (int run = 0; run < options.runs; ++run) {
      paired_gains.push_back(
          samples.front()[static_cast<size_t>(run)] / samples[active_index][static_cast<size_t>(run)] - 1.0);
    }
    const double median_gain = baseline.median / stats.median - 1.0;
    const double paired_gain = Summarize(std::move(paired_gains)).median;
    const double gflops = flops / stats.median / 1.0e9;
    const double b_gbs = streamed_weight_bytes / stats.median / 1.0e9;
    std::cout << std::left << std::setw(20) << kVariants[variant_index].name << std::right << std::setw(10)
              << stats.median * 1.0e3 << std::setw(10) << stats.p10 * 1.0e3 << std::setw(10) << stats.p90 * 1.0e3
              << std::setw(10) << gflops << std::setw(10) << b_gbs << std::setw(14) << median_gain * 100.0
              << std::setw(14) << paired_gain * 100.0 << '\n';
    std::cout << "RESULT_JSON {\"shape\":\"" << shape.name << "\",\"variant\":\"" << kVariants[variant_index].name
              << "\",\"median_ms\":" << stats.median * 1.0e3 << ",\"p10_ms\":" << stats.p10 * 1.0e3
              << ",\"p90_ms\":" << stats.p90 * 1.0e3 << ",\"gflops\":" << gflops << ",\"b_gbs\":" << b_gbs
              << ",\"median_gain_pct\":" << median_gain * 100.0 << ",\"paired_gain_pct\":" << paired_gain * 100.0
              << "}\n";
  }
  std::cout << "output_checksum=" << output_checksum << " a_checksum=" << a_checksum << "\n";
}

struct WaveWorkItem {
  size_t active_index = 0;
  size_t copy = 0;
  bool measured = false;
};

void RunShapeWave(const Shape& shape, const Options& options, const std::vector<size_t>& active_variants) {
  for (size_t variant_index : active_variants) {
    if (kVariants[variant_index].kblocked_b) {
      throw std::invalid_argument("--workers does not support K-blocked B variants");
    }
  }

  const size_t workers = static_cast<size_t>(options.workers);
  const size_t a_elements = CheckedMultiply(options.m, static_cast<size_t>(shape.k), "packed A");
  const size_t weight_elements = CheckedMultiply(static_cast<size_t>(shape.k), shape.n, "packed B");
  const size_t output_elements = CheckedMultiply(options.m, static_cast<size_t>(shape.n), "output");
  const size_t panel_output_elements = CheckedMultiply(kM, static_cast<size_t>(shape.n), "panel output");
  const size_t scratch_elements = CheckedMultiply(panel_output_elements, size_t{2}, "FP32 partial C");
  const size_t rounds = static_cast<size_t>(options.warmup) + options.runs;
  const size_t used_copies = CheckedMultiply(rounds, active_variants.size(), "weight copies");
  const size_t weight_guard_bytes = options.weight_color >= 0 ? kFixedColorStrideBytes : kWeightGuardBytes;
  const size_t weight_base_offset_bytes =
      options.weight_color >= 0 ? static_cast<size_t>(options.weight_color) * 4 * kKiB : 0;
  const size_t weight_base_offset_elements = weight_base_offset_bytes / sizeof(uint16_t);
  const size_t weight_stride_elements = weight_elements + weight_guard_bytes / sizeof(uint16_t);
  const size_t worker_weight_elements =
      weight_base_offset_elements + CheckedMultiply(used_copies, weight_stride_elements, "worker timed weights");
  const size_t all_worker_weight_elements =
      CheckedMultiply(workers, worker_weight_elements, "all worker timed weights");
  const size_t cold_tail_elements =
      CheckedMultiply(static_cast<size_t>(options.cold_tail_mib), kMiB, "cold tail") / sizeof(uint16_t);
  const size_t all_weight_elements = all_worker_weight_elements + cold_tail_elements;

  AlignedBf16 packed_a = AllocateBf16(CheckedMultiply(workers, a_elements, "all packed A"));
  AlignedBf16 weights = AllocateBf16(all_weight_elements);
  AlignedBf16 outputs = AllocateBf16(CheckedMultiply(workers, output_elements, "all outputs"));
  AlignedBf16 scratches = AllocateBf16(CheckedMultiply(workers, scratch_elements, "all scratches"));
  for (size_t worker = 0; worker < workers; ++worker) {
    FillPattern(packed_a.get() + worker * a_elements, a_elements, 17 + static_cast<uint32_t>(worker % 7),
                1.0f / 256.0f);
    uint16_t* const worker_weights = weights.get() + worker * worker_weight_elements;
    for (size_t copy = 0; copy < used_copies; ++copy) {
      const uint16_t value = static_cast<uint16_t>(0x3a40u + (copy + worker) % 32);
      uint16_t* const weight =
          worker_weights + weight_base_offset_elements + copy * weight_stride_elements;
      std::fill_n(weight, weight_elements, value);
      std::fill_n(weight + weight_elements, weight_guard_bytes / sizeof(uint16_t), uint16_t{0});
    }
  }
  std::fill_n(weights.get() + all_worker_weight_elements, cold_tail_elements, uint16_t{0x3b00});
  std::fill_n(outputs.get(), workers * output_elements, kOutputSentinel);
  std::fill_n(scratches.get(), workers * scratch_elements, uint16_t{0});

  const uint64_t cold_checksum =
      ScanCacheLines(weights.get() + all_worker_weight_elements, cold_tail_elements);
  if (cold_tail_elements != 0 && cold_checksum == 0) {
    throw std::runtime_error("cold-tail scan produced a zero checksum");
  }

  std::vector<WaveWorkItem> work;
  work.reserve(used_copies);
  size_t copy = 0;
  for (size_t round = 0; round < rounds; ++round) {
    for (size_t slot = 0; slot < active_variants.size(); ++slot, ++copy) {
      work.push_back(
          {VariantAtSlot(static_cast<int>(round), slot, active_variants.size()), copy,
           round >= static_cast<size_t>(options.warmup)});
    }
  }

  const size_t weight_bytes = CheckedMultiply(weight_elements, sizeof(uint16_t), "weight bytes");
  const double allocated_gib =
      CheckedMultiply(all_weight_elements, sizeof(uint16_t), "all weight bytes") /
      static_cast<double>(size_t{1} << 30);
  std::cout << std::fixed << std::setprecision(4) << "wave_config shape=" << shape.name << " M=" << options.m
            << " K=" << shape.k << " N=" << shape.n << " n_tile=" << svcnth() << " cpu_start=" << options.cpu
            << " workers=" << options.workers << " weight_mib=" << weight_bytes / static_cast<double>(kMiB)
            << " variants=" << active_variants.size() << " warmup=" << options.warmup << " runs=" << options.runs
            << " cold_tail_mib=" << options.cold_tail_mib << " allocated_gib=" << allocated_gib << '\n';

  if (options.stop_before_run) {
    std::cout << "profiler_ready pid=" << getpid() << '\n' << std::flush;
    if (std::raise(SIGSTOP) != 0) {
      throw std::runtime_error("failed to stop for profiler attach");
    }
  }

  const gemm_params_t params = MakeParams(shape);
  std::vector<double> wave_seconds(work.size(), 0.0);
  std::vector<uint64_t> a_checksums(workers, 0);
  Clock::time_point wave_start;
  size_t completed = 0;
  const auto start_completion = [&]() noexcept { wave_start = Clock::now(); };
  const auto finish_completion = [&]() noexcept {
    wave_seconds[completed++] = std::chrono::duration<double>(Clock::now() - wave_start).count();
  };
  fused_moe_sve::support::PhaseBarrier<decltype(start_completion)> start_barrier(
      options.workers, start_completion);
  fused_moe_sve::support::PhaseBarrier<decltype(finish_completion)> finish_barrier(
      options.workers, finish_completion);
  fused_moe_sve::support::PhaseBarrier<> launch_barrier(options.workers + 1);
  bool abort_workers = false;
  int affinity_error = 0;
  std::vector<std::thread> threads;
  threads.reserve(workers);
  for (size_t worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&, worker]() {
      launch_barrier.arrive_and_wait();
      if (abort_workers) {
        return;
      }
      const uint16_t* const worker_a = packed_a.get() + worker * a_elements;
      const uint16_t* const worker_weights = weights.get() + worker * worker_weight_elements;
      uint16_t* const worker_output = outputs.get() + worker * output_elements;
      uint16_t* const worker_scratch = scratches.get() + worker * scratch_elements;
      uint64_t a_checksum = 0;
      for (const WaveWorkItem& item : work) {
        if (options.prewarm_a) {
          a_checksum += ScanCacheLines(worker_a, a_elements);
        }
        start_barrier.arrive_and_wait();
        const size_t variant_index = active_variants[item.active_index];
        InvokeRows(kVariants[variant_index], worker_a,
                   worker_weights + weight_base_offset_elements + item.copy * weight_stride_elements,
                   worker_output, worker_scratch, params, options.m);
        finish_barrier.arrive_and_wait();
      }
      a_checksums[worker] = a_checksum;
    });

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(options.cpu + static_cast<int>(worker), &set);
    const int error = pthread_setaffinity_np(threads.back().native_handle(), sizeof(set), &set);
    if (error != 0 && affinity_error == 0) {
      affinity_error = error;
    }
  }
  abort_workers = affinity_error != 0;
  launch_barrier.arrive_and_wait();
  for (std::thread& thread : threads) {
    thread.join();
  }
  if (affinity_error != 0) {
    throw std::runtime_error("worker pthread_setaffinity_np failed: " + std::to_string(affinity_error));
  }
  if (completed != work.size()) {
    throw std::logic_error("wave timing accounting mismatch");
  }

  std::vector<std::vector<double>> samples(active_variants.size());
  for (size_t item_index = 0; item_index < work.size(); ++item_index) {
    if (work[item_index].measured) {
      samples[work[item_index].active_index].push_back(wave_seconds[item_index]);
    }
  }
  const Stats baseline = Summarize(samples.front());
  const double flops =
      2.0 * options.m * shape.k * static_cast<double>(shape.n) * options.workers;
  const double streamed_weight_bytes =
      static_cast<double>(weight_bytes) * options.m / kM * options.workers;
  std::cout << "variant              median_ms    p10_ms    p90_ms   GFLOP/s   B_GB/s  median_gain%  paired_gain%\n";
  for (size_t active_index = 0; active_index < active_variants.size(); ++active_index) {
    const size_t variant_index = active_variants[active_index];
    const Stats stats = Summarize(samples[active_index]);
    std::vector<double> paired_gains;
    paired_gains.reserve(static_cast<size_t>(options.runs));
    for (int run = 0; run < options.runs; ++run) {
      paired_gains.push_back(
          samples.front()[static_cast<size_t>(run)] / samples[active_index][static_cast<size_t>(run)] - 1.0);
    }
    const double median_gain = baseline.median / stats.median - 1.0;
    const double paired_gain = Summarize(std::move(paired_gains)).median;
    const double gflops = flops / stats.median / 1.0e9;
    const double b_gbs = streamed_weight_bytes / stats.median / 1.0e9;
    std::cout << std::left << std::setw(20) << kVariants[variant_index].name << std::right << std::setw(10)
              << stats.median * 1.0e3 << std::setw(10) << stats.p10 * 1.0e3 << std::setw(10)
              << stats.p90 * 1.0e3 << std::setw(10) << gflops << std::setw(10) << b_gbs << std::setw(14)
              << median_gain * 100.0 << std::setw(14) << paired_gain * 100.0 << '\n';
    std::cout << "WAVE_RESULT_JSON {\"shape\":\"" << shape.name << "\",\"variant\":\""
              << kVariants[variant_index].name << "\",\"workers\":" << options.workers
              << ",\"median_ms\":" << stats.median * 1.0e3 << ",\"p10_ms\":" << stats.p10 * 1.0e3
              << ",\"p90_ms\":" << stats.p90 * 1.0e3 << ",\"gflops\":" << gflops << ",\"b_gbs\":" << b_gbs
              << ",\"median_gain_pct\":" << median_gain * 100.0 << ",\"paired_gain_pct\":"
              << paired_gain * 100.0 << "}\n";
  }

  uint64_t output_checksum = ScanCacheLines(outputs.get(), workers * output_elements);
  const uint64_t a_checksum = std::accumulate(a_checksums.begin(), a_checksums.end(), uint64_t{0});
  std::cout << "output_checksum=" << output_checksum << " a_checksum=" << a_checksum << "\n";
}

std::vector<Shape> SelectShapes(const Options& options) {
  if ((options.custom_k == 0) != (options.custom_n == 0)) {
    throw std::invalid_argument("--k and --n must be provided together");
  }
  if (options.custom_k != 0) {
    return {{"custom", options.custom_k, options.custom_n}};
  }
  if (options.shape == "all") {
    return {{"w13_range", 4096, 512}, {"w2", 512, 4096}};
  }
  if (options.shape == "w13") {
    return {{"w13_range", 4096, 512}};
  }
  if (options.shape == "w2") {
    return {{"w2", 512, 4096}};
  }
  throw std::invalid_argument("--shape must be all, w13, or w2");
}

std::vector<size_t> SelectVariants(const Options& options) {
  if (options.variants == "all") {
    std::vector<size_t> indices(kVariants.size());
    std::iota(indices.begin(), indices.end(), size_t{0});
    return indices;
  }

  std::vector<size_t> indices;
  size_t begin = 0;
  while (begin <= options.variants.size()) {
    const size_t end = options.variants.find(',', begin);
    const size_t length = (end == std::string::npos ? options.variants.size() : end) - begin;
    const std::string_view name(options.variants.data() + begin, length);
    const auto variant = std::find_if(kVariants.begin(), kVariants.end(),
                                      [&](const Variant& candidate) { return name == candidate.name; });
    if (name.empty() || variant == kVariants.end()) {
      throw std::invalid_argument("unknown or empty variant in --variants: " + std::string(name));
    }
    const size_t index = static_cast<size_t>(variant - kVariants.begin());
    if (std::find(indices.begin(), indices.end(), index) != indices.end()) {
      throw std::invalid_argument("duplicate variant in --variants: " + std::string(name));
    }
    indices.push_back(index);
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }

  const auto baseline = std::find(indices.begin(), indices.end(), size_t{0});
  if (baseline != indices.end()) {
    std::rotate(indices.begin(), baseline, baseline + 1);
  }
  return indices;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    PinCurrentThread(options.cpu);
    const int n_tile = static_cast<int>(svcnth());
    const std::vector<Shape> shapes = SelectShapes(options);
    const std::vector<size_t> active_variants = SelectVariants(options);
    for (const Shape& shape : shapes) {
      if (options.m % kM != 0) {
        throw std::invalid_argument("M must be a positive multiple of 12");
      }
      if (shape.k % 8 != 0 || shape.n % n_tile != 0) {
        throw std::invalid_argument("K must align to 8 and N to the runtime SVE n_tile");
      }
      if (options.cpu + options.workers > CPU_SETSIZE) {
        throw std::invalid_argument("worker CPU range exceeds CPU_SETSIZE");
      }
      CheckCorrectness(shape);
      if (!options.check_only) {
        if (options.workers == 1) {
          RunShape(shape, options, active_variants);
        } else {
          RunShapeWave(shape, options, active_variants);
        }
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
