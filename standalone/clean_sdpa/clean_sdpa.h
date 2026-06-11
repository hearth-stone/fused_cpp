#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <vector>

#include "torch/extension.h"

namespace clean_sdpa {

template <typename T, std::size_t Alignment = 64>
struct AlignedAllocator {
  using value_type = T;

  AlignedAllocator() noexcept = default;

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

  [[nodiscard]] T* allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }
    void* ptr = nullptr;
    if (n != 0 && ::posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
  }

  void deallocate(T* ptr, std::size_t) noexcept {
    ::free(ptr);
  }

  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };
};

template <typename T, typename U, std::size_t Alignment>
inline bool operator==(
    const AlignedAllocator<T, Alignment>&,
    const AlignedAllocator<U, Alignment>&) noexcept {
  return true;
}

template <typename T, typename U, std::size_t Alignment>
inline bool operator!=(
    const AlignedAllocator<T, Alignment>&,
    const AlignedAllocator<U, Alignment>&) noexcept {
  return false;
}

template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T, 64>>;

struct Config {
  int64_t B = 1;
  int64_t N = 8;
  int64_t L = 512;
  int64_t S = 512;
  int64_t E = 64;
  int64_t Ev = 64;
  bool causal = false;
  float scale = 0.0f;
  int64_t causal_offset = 0;
  // 0 means use the same cache-derived Sc_l2/Sc_l3 as the extension path.
  int64_t s_tile = 0;
};

void sdpa_bf16_packqkv_pbf16pv(
    const at::BFloat16* q,
    const at::BFloat16* k,
    const at::BFloat16* v,
    float* out,
    const Config& cfg);

void reference_sdpa_bf16(
    const at::BFloat16* q,
    const at::BFloat16* k,
    const at::BFloat16* v,
    float* out,
    const Config& cfg);

double counted_gflops(const Config& cfg, double mean_ms);
double checksum(const float* data, int64_t size);
double max_abs_diff(const float* a, const float* b, int64_t size);

}  // namespace clean_sdpa
