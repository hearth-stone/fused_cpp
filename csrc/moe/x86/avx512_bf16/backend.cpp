// SPDX-License-Identifier: Apache-2.0
#include "backend.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(__x86_64__)
#include <cpuid.h>
#include <immintrin.h>
#endif

namespace fused_cpp::moe::x86::avx512_bf16 {
namespace {

int RoundUp(int value, int quantum) {
  if (value > std::numeric_limits<int>::max() - (quantum - 1)) {
    throw std::overflow_error("AVX-512 BF16 packed dimension exceeds int32");
  }
  return ((value + quantum - 1) / quantum) * quantum;
}

bool EnvFalse(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  return value[0] == '\0' || value[0] == '0' || std::strcmp(value, "false") == 0 || std::strcmp(value, "False") == 0 ||
         std::strcmp(value, "off") == 0 || std::strcmp(value, "OFF") == 0;
}

#if defined(__x86_64__)
__attribute__((target("xsave"))) uint64_t ReadXcr0() { return _xgetbv(0); }

bool CpuSupportsAvx512Bf16() {
  unsigned int eax = 0;
  unsigned int ebx = 0;
  unsigned int ecx = 0;
  unsigned int edx = 0;
  if (__get_cpuid_max(0, nullptr) < 7 || !__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    return false;
  }
  constexpr unsigned int kOsxsave = 1u << 27;
  constexpr unsigned int kAvx = 1u << 28;
  if ((ecx & (kOsxsave | kAvx)) != (kOsxsave | kAvx)) {
    return false;
  }
  constexpr uint64_t kXcr0Avx512State =
      (uint64_t{1} << 1) | (uint64_t{1} << 2) | (uint64_t{1} << 5) | (uint64_t{1} << 6) | (uint64_t{1} << 7);
  if ((ReadXcr0() & kXcr0Avx512State) != kXcr0Avx512State) {
    return false;
  }
  if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
    return false;
  }
  constexpr unsigned int kAvx512F = 1u << 16;
  constexpr unsigned int kAvx512Bw = 1u << 30;
  constexpr unsigned int kAvx512Vl = 1u << 31;
  if ((ebx & (kAvx512F | kAvx512Bw | kAvx512Vl)) != (kAvx512F | kAvx512Bw | kAvx512Vl)) {
    return false;
  }
  if (eax < 1) {
    return false;
  }
  if (!__get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx)) {
    return false;
  }
  constexpr unsigned int kAvx512Bf16 = 1u << 5;
  return (eax & kAvx512Bf16) != 0;
}
#endif

}  // namespace

bool RuntimeSupported() {
#if defined(__x86_64__) && defined(FUSED_CPP_MOE_HAS_X86_AVX512_BF16)
  static const bool supported = CpuSupportsAvx512Bf16();
  return supported && !EnvFalse("FUSED_CPP_MOE_AVX512_BF16");
#else
  return false;
#endif
}

int NTile() { return 32; }

int RoundK(int value) { return RoundUp(std::max(value, 2), 2); }

int RoundN(int value) { return RoundUp(std::max(value, 32), 32); }

void PackB(const uint16_t* source, uint16_t* packed, int k_size, int n_size) {
  for (int nb = 0; nb < n_size; nb += 32) {
    uint16_t* block = packed + static_cast<int64_t>(nb / 32) * k_size * 32;
    for (int kp = 0; kp < k_size / 2; ++kp) {
      for (int n = 0; n < 32; ++n) {
        block[static_cast<int64_t>(kp) * 64 + n * 2] = source[static_cast<int64_t>(kp * 2) * n_size + nb + n];
        block[static_cast<int64_t>(kp) * 64 + n * 2 + 1] = source[static_cast<int64_t>(kp * 2 + 1) * n_size + nb + n];
      }
    }
  }
}

void PackW13(const uint16_t* weight, uint16_t* packed, int64_t f_size, int64_t h_size, int k_pad, int f_pad) {
  std::fill(packed, packed + static_cast<int64_t>(k_pad) * f_pad * 2, static_cast<uint16_t>(0));
  for (int fb = 0; fb < f_pad; fb += 16) {
    uint16_t* block = packed + static_cast<int64_t>(fb / 16) * k_pad * 32;
    for (int kp = 0; kp < k_pad / 2; ++kp) {
      uint16_t* gate_dst = block + static_cast<int64_t>(kp) * 64;
      uint16_t* up_dst = gate_dst + 32;
      for (int lane = 0; lane < 16; ++lane) {
        const int64_t feature = fb + lane;
        for (int half = 0; half < 2; ++half) {
          const int64_t k = kp * 2 + half;
          if (feature < f_size && k < h_size) {
            gate_dst[lane * 2 + half] = weight[feature * h_size + k];
            up_dst[lane * 2 + half] = weight[(f_size + feature) * h_size + k];
          }
        }
      }
    }
  }
}

void PackW2(const uint16_t* weight, uint16_t* packed, int64_t h_size, int64_t f_size, int k_pad, int n_pad) {
  std::fill(packed, packed + static_cast<int64_t>(k_pad) * n_pad, static_cast<uint16_t>(0));
  for (int nb = 0; nb < n_pad; nb += 32) {
    uint16_t* block = packed + static_cast<int64_t>(nb / 32) * k_pad * 32;
    for (int kp = 0; kp < k_pad / 2; ++kp) {
      uint16_t* dst = block + static_cast<int64_t>(kp) * 64;
      for (int lane = 0; lane < 32; ++lane) {
        const int64_t output = nb + lane;
        for (int half = 0; half < 2; ++half) {
          const int64_t k = kp * 2 + half;
          if (output < h_size && k < f_size) {
            dst[lane * 2 + half] = weight[output * f_size + k];
          }
        }
      }
    }
  }
}

}  // namespace fused_cpp::moe::x86::avx512_bf16
