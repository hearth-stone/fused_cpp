// SPDX-License-Identifier: Apache-2.0
#include "policy.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#if defined(__x86_64__)
#include <cpuid.h>
#endif

namespace fused_cpp::moe::x86::avx512_bf16 {
namespace {

constexpr int kIntelFamily = 6;
constexpr int kC8iModel = 0xad;
constexpr int64_t kTinyCallMacs = 128 * 1024;
constexpr int64_t kTinyExpertMacs = 8 * 1024;
constexpr int64_t kReferenceMacsPerThread = int64_t{64} * 4096 * 512;
constexpr int64_t kC8iL2Bytes = int64_t{2} << 20;
constexpr int kGenericM1N4MinRows = 76;

bool ProductAtMost(int64_t a, int64_t b, int64_t c, int64_t limit) {
  if (a < 0 || b < 0 || c < 0 || limit < 0) {
    return false;
  }
  if (a == 0 || b == 0 || c == 0) {
    return true;
  }
  return a <= limit / b && a * b <= limit / c;
}

bool ProductAtLeast(int a, int b, int64_t threshold) {
  return a > 0 && b > 0 && static_cast<int64_t>(a) * b >= threshold;
}

bool IsC8iHardware(const X86CpuIdentity& identity) {
  return identity.is_intel && identity.family == kIntelFamily && identity.model == kC8iModel;
}

bool UseC8iProfile() {
  const char* raw = std::getenv("FUSED_CPP_MOE_X86_POLICY_PROFILE");
  if (raw == nullptr || raw[0] == '\0' || std::strcmp(raw, "auto") == 0) {
    return IsC8iHardware(GetX86CpuIdentity());
  }
  if (std::strcmp(raw, "generic_v1") == 0 || std::strcmp(raw, "generic") == 0) {
    return false;
  }
  if (std::strcmp(raw, "intel_06_ad_c8i_v1") == 0 || std::strcmp(raw, "c8i") == 0) {
    return true;
  }
  throw std::runtime_error("FUSED_CPP_MOE_X86_POLICY_PROFILE must be auto, generic_v1, or intel_06_ad_c8i_v1; got '" +
                           std::string(raw) + "'");
}

X86ExecutionIsa ResolveIsaOverride(const X86PolicyInput& input, bool c8i_profile) {
  const char* raw = std::getenv("FUSED_CPP_MOE_X86_ISA");
  if (raw != nullptr && raw[0] != '\0' && std::strcmp(raw, "auto") != 0) {
    if (std::strcmp(raw, "avx512") == 0 || std::strcmp(raw, "avx512_bf16") == 0) {
      if (!input.avx512_available) {
        throw std::runtime_error("FUSED_CPP_MOE_X86_ISA=avx512 requested an unavailable AVX-512 BF16 runtime");
      }
      return X86ExecutionIsa::kAvx512Bf16;
    }
    if (std::strcmp(raw, "amx") == 0 || std::strcmp(raw, "amx_bf16") == 0) {
      if (!input.amx_available) {
        throw std::runtime_error("FUSED_CPP_MOE_X86_ISA=amx requested an unavailable AMX BF16 runtime");
      }
      return X86ExecutionIsa::kAmxBf16;
    }
    throw std::runtime_error("FUSED_CPP_MOE_X86_ISA must be auto, avx512, or amx; got '" + std::string(raw) + "'");
  }

  if (c8i_profile && ProductAtMost(input.max_routes, input.hidden_size, input.intermediate_size, kTinyExpertMacs) &&
      input.avx512_available) {
    return X86ExecutionIsa::kAvx512Bf16;
  }
  if (input.amx_available) {
    return X86ExecutionIsa::kAmxBf16;
  }
  if (input.avx512_available) {
    return X86ExecutionIsa::kAvx512Bf16;
  }
  throw std::runtime_error("dimension-aware x86 policy has no available BF16 execution ISA");
}

int64_t C8iTargetRows(int64_t hidden_size, int64_t intermediate_size) {
  if (hidden_size <= 0 || intermediate_size <= 0 ||
      hidden_size > std::numeric_limits<int64_t>::max() / intermediate_size) {
    return 64;
  }
  const int64_t macs_per_row = hidden_size * intermediate_size;
  const int64_t rows = (kReferenceMacsPerThread + macs_per_row - 1) / macs_per_row;
  return std::clamp<int64_t>(rows, 16, 256);
}

}  // namespace

X86CpuIdentity GetX86CpuIdentity() {
  static const X86CpuIdentity identity = [] {
    X86CpuIdentity result;
#if defined(__x86_64__)
    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;
    if (__get_cpuid_max(0, nullptr) == 0 || !__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
      return result;
    }
    result.is_intel = ebx == 0x756e6547u && edx == 0x49656e69u && ecx == 0x6c65746eu;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
      return result;
    }
    const int base_family = static_cast<int>((eax >> 8) & 0xf);
    const int base_model = static_cast<int>((eax >> 4) & 0xf);
    const int extended_family = static_cast<int>((eax >> 20) & 0xff);
    const int extended_model = static_cast<int>((eax >> 16) & 0xf);
    result.family = base_family == 0xf ? base_family + extended_family : base_family;
    result.model = base_family == 0x6 || base_family == 0xf ? base_model | (extended_model << 4) : base_model;
    result.stepping = static_cast<int>(eax & 0xf);
#endif
    return result;
  }();
  return identity;
}

X86PolicyDecision ResolveX86Policy(const X86PolicyInput& input) {
  if (input.hidden_size <= 0 || input.intermediate_size <= 0 || input.total_routes < 0 || input.max_routes < 0 ||
      input.second_max_routes < 0 || input.active_experts < 0 || input.requested_threads <= 0) {
    throw std::invalid_argument("x86 policy dimensions, route counts, and thread count are invalid");
  }

  const bool c8i_profile = UseC8iProfile();
  X86PolicyDecision decision;
  decision.profile = c8i_profile ? "intel_06_ad_c8i_v1" : "generic_v1";
  decision.execution_threads = input.requested_threads;
  decision.nsplit_target_rows = c8i_profile ? C8iTargetRows(input.hidden_size, input.intermediate_size) : int64_t{64};

  if (input.allow_isa_selection) {
    decision.isa = ResolveIsaOverride(input, c8i_profile);
    if (c8i_profile && ProductAtMost(input.total_routes, input.hidden_size, input.intermediate_size, kTinyCallMacs)) {
      decision.execution_threads = 1;
    }
  } else {
    decision.isa = input.requested_isa;
  }

  decision.route_skewed = input.second_max_routes > 0 && input.max_routes >= decision.nsplit_target_rows &&
                          input.second_max_routes <= input.max_routes / 2;
  return decision;
}

bool UseAutomaticAvx512SmallMMultiN(Avx512SmallMMultiNStage stage, int rows, int reduction_size, int output_size,
                                    int cooperative_threads) {
  if (!UseC8iProfile() || rows < 1 || rows > 4 || cooperative_threads <= 0 || cooperative_threads >= 8) {
    return false;
  }
  if (stage == Avx512SmallMMultiNStage::kW13) {
    if (cooperative_threads >= 4 && rows == 4 && output_size < 1024) {
      return false;
    }
    switch (rows) {
      case 1:
        return ProductAtLeast(reduction_size, output_size, 32 * 1024);
      case 2:
        return reduction_size >= 256 && ProductAtLeast(reduction_size, output_size, 64 * 1024);
      case 3:
        return ProductAtLeast(reduction_size, output_size, 64 * 1024);
      case 4:
        return reduction_size >= 512 && ProductAtLeast(reduction_size, output_size, 256 * 1024);
    }
  }
  if (cooperative_threads >= 4 && (reduction_size < 1024 || rows == 4)) {
    return false;
  }
  if (cooperative_threads >= 2 && ((rows == 3 && reduction_size < 1024) || (rows == 4 && reduction_size < 2048))) {
    return false;
  }
  switch (rows) {
    case 1:
    case 2:
      return reduction_size >= 256 && ProductAtLeast(reduction_size, output_size, 256 * 1024);
    case 3:
      return reduction_size >= 512 && ProductAtLeast(reduction_size, output_size, 2 * 1024 * 1024);
    case 4:
      return reduction_size >= 1024 && ProductAtLeast(reduction_size, output_size, 8 * 1024 * 1024);
  }
  return false;
}

bool UseAutomaticAvx512BulkMN(Avx512BulkMNStage stage, int rows, int reduction_size, int output_size,
                              int cooperative_threads) {
  if (!UseC8iProfile() || stage != Avx512BulkMNStage::kW13) {
    return false;
  }
  // C8i calibration: moving both panel loops into the JIT body only pays for
  // many short W13 reductions.  W2 and conventional hidden sizes were neutral
  // within noise, while enabling them broadly could regress by roughly 0.2%.
  return rows >= 48 && reduction_size > 0 && reduction_size <= 64 && output_size >= 1024 && cooperative_threads > 0 &&
         cooperative_threads <= 4;
}

AmxKernelPattern ResolveAutomaticAmxKernelPattern(int rows, int hidden_size, int intermediate_size) {
  if (!UseC8iProfile() || hidden_size <= 0 || intermediate_size <= 0) {
    return rows >= kGenericM1N4MinRows ? AmxKernelPattern::kM1N4 : AmxKernelPattern::kM2N2;
  }
  if (hidden_size < 4096) {
    return AmxKernelPattern::kM2N2;
  }
  const int hidden_half_ceil = hidden_size / 2 + hidden_size % 2;
  const int m1n4_min_rows = intermediate_size >= hidden_half_ceil ? 128 : 96;
  return rows >= m1n4_min_rows ? AmxKernelPattern::kM1N4 : AmxKernelPattern::kM2N2;
}

int ResolveAutomaticAmxCacheBlocks(AmxCacheStage stage, int k_pad, int rows, int hidden_size, int intermediate_size,
                                   AmxKernelPattern pattern) {
  if (k_pad <= 0) {
    throw std::invalid_argument("AMX cache policy requires positive padded K");
  }
  const bool c8i_profile = UseC8iProfile();
  if (c8i_profile && hidden_size > 0 && intermediate_size > 0 && rows <= 16) {
    return 0;
  }
  const int64_t target_bytes = stage == AmxCacheStage::kW13 ? kC8iL2Bytes / 2 : kC8iL2Bytes / 4;
  const int64_t bytes_per_block = static_cast<int64_t>(k_pad) * 64;
  int blocks = static_cast<int>(std::max<int64_t>(1, target_bytes / bytes_per_block));
  if (pattern == AmxKernelPattern::kM1N4 && blocks > 1) {
    blocks -= blocks % 2;
  }
  return blocks;
}

const char* X86ExecutionIsaName(X86ExecutionIsa isa) {
  return isa == X86ExecutionIsa::kAmxBf16 ? "amx_bf16" : "avx512_bf16";
}

const char* AmxKernelPatternName(AmxKernelPattern pattern) {
  switch (pattern) {
    case AmxKernelPattern::kM1N2:
      return "m1n2";
    case AmxKernelPattern::kM2N2:
      return "m2n2";
    case AmxKernelPattern::kM1N4:
      return "m1n4";
  }
  return "unknown";
}

}  // namespace fused_cpp::moe::x86::avx512_bf16
