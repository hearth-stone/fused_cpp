// SPDX-License-Identifier: Apache-2.0
#include "backend.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(__aarch64__) && defined(__linux__)
#include <asm/hwcap.h>
#include <linux/prctl.h>
#include <sys/prctl.h>
#include <sys/auxv.h>
#endif

#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
#include "../arm/sve_bf16/packing.h"
#endif

#if defined(FUSED_CPP_MOE_HAS_X86_AVX512_BF16)
#include "../x86/avx512_bf16/backend.h"
#endif

namespace fused_cpp::moe {
namespace {

#if defined(__aarch64__) && defined(__linux__)
#ifndef HWCAP_SVE
#define HWCAP_SVE (1UL << 22)
#endif
#ifndef HWCAP2_SVEBF16
#define HWCAP2_SVEBF16 (1UL << 12)
#endif
#ifndef HWCAP2_BF16
#define HWCAP2_BF16 (1UL << 14)
#endif
#endif

#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
bool env_false(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  return value[0] == '\0' || value[0] == '0' || std::strcmp(value, "false") == 0 || std::strcmp(value, "False") == 0 ||
         std::strcmp(value, "off") == 0 || std::strcmp(value, "OFF") == 0;
}
#endif

int round_up(int value, int quantum) { return ((value + quantum - 1) / quantum) * quantum; }

int neon_n_tile() { return 8; }

int neon_round_k(int value) { return round_up(value < 8 ? 8 : value, 8); }

int neon_round_n(int value) { return round_up(value < 8 ? 8 : value, 8); }

void neon_pack_b(const uint16_t* source, uint16_t* packed, int k_size, int n_size) {
  int64_t index = 0;
  for (int column_block = 0; column_block < n_size / 8; ++column_block) {
    for (int row_block = 0; row_block < k_size / 4; ++row_block) {
      const int row_base = row_block * 4;
      const int column_base = column_block * 8;
      for (int column_pair = 0; column_pair < 4; ++column_pair) {
        const int column0 = column_base + column_pair * 2;
        const int column1 = column0 + 1;
        for (int row = 0; row < 4; ++row) {
          packed[index++] = source[(row_base + row) * n_size + column0];
        }
        for (int row = 0; row < 4; ++row) {
          packed[index++] = source[(row_base + row) * n_size + column1];
        }
      }
    }
  }
}

bool neon_runtime_supported() {
#if defined(__aarch64__) && defined(__linux__)
  static const bool supported = (getauxval(AT_HWCAP) & HWCAP_ASIMD) != 0 && (getauxval(AT_HWCAP2) & HWCAP2_BF16) != 0;
  return supported;
#elif defined(__aarch64__) && defined(__ARM_FEATURE_BF16)
  return true;
#else
  return false;
#endif
}

bool sve_runtime_supported() {
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE) && defined(__aarch64__) && defined(__linux__)
  static const bool supported = (getauxval(AT_HWCAP) & HWCAP_SVE) != 0 &&
                                (getauxval(AT_HWCAP2) & HWCAP2_SVEBF16) != 0 &&
                                (getauxval(AT_HWCAP2) & HWCAP2_BF16) != 0;
  return supported && !env_false("FUSED_CPP_MOE_SVE");
#else
  return false;
#endif
}

const MoeBackend kArmNeonBackend{
    BackendId::kArmNeonBf16, "arm_neon_bf16", "arm",        "neon_bf16",  kFusedSiluPackC,
    neon_runtime_supported,  neon_n_tile,     neon_round_k, neon_round_n, neon_pack_b,
};

const MoeBackend kArmSveBackend{
    BackendId::kArmSveBf16,
    "arm_sve_bf16",
    "arm",
    "sve_bf16",
    kFusedSiluPackC | kDirectRouteF32 | kDirectRouteBf16 | kRouteMerge,
    sve_runtime_supported,
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
    ::fused_cpp::moe_sve::n_tile,
    ::fused_cpp::moe_sve::round_k,
    ::fused_cpp::moe_sve::round_n,
    ::fused_cpp::moe_sve::pack_b,
#else
    nullptr,
    nullptr,
    nullptr,
    nullptr,
#endif
};

const MoeBackend kX86Avx512Bf16Backend{
    BackendId::kX86Avx512Bf16,
    "x86_avx512_bf16",
    "x86",
    "avx512_bf16",
    kFusedSiluPackC | kDirectRouteF32 | kRouteMerge,
#if defined(FUSED_CPP_MOE_HAS_X86_AVX512_BF16)
    ::fused_cpp::moe::x86::avx512_bf16::RuntimeSupported,
    ::fused_cpp::moe::x86::avx512_bf16::NTile,
    ::fused_cpp::moe::x86::avx512_bf16::RoundK,
    ::fused_cpp::moe::x86::avx512_bf16::RoundN,
    ::fused_cpp::moe::x86::avx512_bf16::PackB,
#else
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
#endif
};

const MoeBackend kX86AmxBf16Backend{
    BackendId::kX86AmxBf16,
    "x86_amx_bf16",
    "x86",
    "amx_bf16",
    kFusedSiluPackC | kDirectRouteF32 | kDirectRouteBf16 | kRouteMerge,
#if defined(FUSED_CPP_MOE_HAS_X86_AVX512_BF16)
    ::fused_cpp::moe::x86::avx512_bf16::AmxRuntimeSupported,
    ::fused_cpp::moe::x86::avx512_bf16::NTile,
    ::fused_cpp::moe::x86::avx512_bf16::AmxRoundK,
    ::fused_cpp::moe::x86::avx512_bf16::RoundN,
    ::fused_cpp::moe::x86::avx512_bf16::PackB,
#else
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
#endif
};

const MoeBackend* known_backend(const std::string& name) {
  if (name == kArmNeonBackend.name || name == "neon") {
    return &kArmNeonBackend;
  }
  if (name == kArmSveBackend.name || name == "sve") {
    return &kArmSveBackend;
  }
  if (name == kX86Avx512Bf16Backend.name || name == "avx512_bf16") {
    return &kX86Avx512Bf16Backend;
  }
  if (name == kX86AmxBf16Backend.name || name == "amx_bf16") {
    return &kX86AmxBf16Backend;
  }
  return nullptr;
}

std::string unavailable_message(const MoeBackend& backend) {
  return "MoE backend '" + std::string(backend.name) + "' is not supported by this build/runtime";
}

}  // namespace

bool backend_runtime_supported(const MoeBackend& backend) {
  return backend.runtime_supported != nullptr && backend.runtime_supported();
}

const MoeBackend& resolve_backend(const std::string& requested, bool fuse_silu) {
  if (requested.empty() || requested == "auto") {
    if (fuse_silu && backend_runtime_supported(kX86AmxBf16Backend)) {
      return kX86AmxBf16Backend;
    }
    if (fuse_silu && backend_runtime_supported(kX86Avx512Bf16Backend)) {
      return kX86Avx512Bf16Backend;
    }
    if (fuse_silu && backend_runtime_supported(kArmSveBackend)) {
      return kArmSveBackend;
    }
    if (backend_runtime_supported(kArmNeonBackend)) {
      return kArmNeonBackend;
    }
    throw std::runtime_error("no BF16 fused MoE backend is supported by this CPU/runtime");
  }

  if (requested == "x86_avx2") {
    throw std::runtime_error("MoE backend '" + requested + "' is known but not implemented");
  }
  const MoeBackend* backend = known_backend(requested);
  if (backend == nullptr) {
    throw std::invalid_argument("unknown MoE backend '" + requested +
                                "'; expected auto, arm_neon_bf16, arm_sve_bf16, x86_avx512_bf16, or "
                                "x86_amx_bf16");
  }
  if ((backend->id == BackendId::kArmSveBf16 || backend->id == BackendId::kX86Avx512Bf16 ||
       backend->id == BackendId::kX86AmxBf16) &&
      !fuse_silu) {
    throw std::invalid_argument(std::string(backend->name) + " currently requires fuse_silu=True");
  }
  if (!backend_runtime_supported(*backend)) {
    throw std::runtime_error(unavailable_message(*backend));
  }
  return *backend;
}

const MoeBackend& backend_from_id(int64_t backend_id) {
  const MoeBackend* backend = nullptr;
  if (backend_id == static_cast<int64_t>(BackendId::kArmNeonBf16)) {
    backend = &kArmNeonBackend;
  } else if (backend_id == static_cast<int64_t>(BackendId::kArmSveBf16)) {
    backend = &kArmSveBackend;
  } else if (backend_id == static_cast<int64_t>(BackendId::kX86Avx512Bf16)) {
    backend = &kX86Avx512Bf16Backend;
  } else if (backend_id == static_cast<int64_t>(BackendId::kX86AmxBf16)) {
    backend = &kX86AmxBf16Backend;
  } else {
    throw std::invalid_argument("unknown packed MoE backend id " + std::to_string(backend_id));
  }
  if (!backend_runtime_supported(*backend)) {
    throw std::runtime_error(unavailable_message(*backend));
  }
  return *backend;
}

std::vector<std::string> available_backend_names() {
  std::vector<std::string> names;
  if (backend_runtime_supported(kArmNeonBackend)) {
    names.emplace_back(kArmNeonBackend.name);
  }
  if (backend_runtime_supported(kArmSveBackend)) {
    names.emplace_back(kArmSveBackend.name);
  }
  if (backend_runtime_supported(kX86Avx512Bf16Backend)) {
    names.emplace_back(kX86Avx512Bf16Backend.name);
  }
  if (backend_runtime_supported(kX86AmxBf16Backend)) {
    names.emplace_back(kX86AmxBf16Backend.name);
  }
  return names;
}

void validate_sve_vector_length_at_import() {
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE) && defined(__aarch64__) && defined(__linux__)
  if ((getauxval(AT_HWCAP) & HWCAP_SVE) == 0) {
    return;
  }
#ifndef PR_SVE_GET_VL
#define PR_SVE_GET_VL 51
#endif
#ifndef PR_SVE_VL_LEN_MASK
#define PR_SVE_VL_LEN_MASK 0xffff
#endif
  const int runtime_vl = prctl(PR_SVE_GET_VL);
  if (runtime_vl < 0) {
    throw std::runtime_error("failed to query the runtime SVE vector length with PR_SVE_GET_VL");
  }
  const int runtime_bytes = runtime_vl & PR_SVE_VL_LEN_MASK;
  constexpr int compiled_bytes = FUSED_CPP_MOE_SVE_VECTOR_BITS / 8;
  if (runtime_bytes != compiled_bytes) {
    throw std::runtime_error("SVE vector length mismatch: fused_cpp._moe_C was built for " +
                             std::to_string(FUSED_CPP_MOE_SVE_VECTOR_BITS) + " bits, but the importing thread uses " +
                             std::to_string(runtime_bytes * 8) +
                             " bits; rebuild with FUSED_CPP_SVE_VECTOR_BITS=" +
                             std::to_string(runtime_bytes * 8));
  }
#endif
}

}  // namespace fused_cpp::moe
