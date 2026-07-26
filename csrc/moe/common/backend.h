// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace fused_cpp::moe {

enum class BackendId : int64_t {
  kArmNeonBf16 = 0,
  kArmSveBf16 = 1,
  kX86Avx2 = 100,
  kX86Avx512Bf16 = 101,
  kX86AmxBf16 = 102,
  kX86AmxBf16N64 = 103,
  // AMX-compatible N32/K32 packing with per-call AVX-512/AMX dispatch.
  kX86AutoBf16 = 104,
};

enum BackendCapability : uint64_t {
  kFusedSiluPackC = uint64_t{1} << 0,
  kDirectRouteF32 = uint64_t{1} << 1,
  kDirectRouteBf16 = uint64_t{1} << 2,
  kRouteMerge = uint64_t{1} << 3,
  kWeightWindows = uint64_t{1} << 4,
};

using RuntimeSupportedFn = bool (*)();
using TileFn = int (*)();
using RoundFn = int (*)(int);
using PackBFn = void (*)(const uint16_t*, uint16_t*, int, int);

// Backend descriptors are immutable for the process lifetime. Function
// pointers cross the ISA boundary; callers must resolve runtime support before
// invoking any target-specific entry.
struct MoeBackend {
  BackendId id;
  const char* name;
  const char* architecture;
  const char* isa;
  uint64_t capabilities;
  RuntimeSupportedFn runtime_supported;
  TileFn n_tile;
  RoundFn round_k;
  RoundFn round_n;
  PackBFn pack_b;
};

const MoeBackend& resolve_backend(const std::string& requested, bool fuse_silu);
const MoeBackend& backend_from_id(int64_t backend_id);
std::vector<std::string> available_backend_names();
bool backend_runtime_supported(const MoeBackend& backend);
void validate_sve_vector_length_at_import();

}  // namespace fused_cpp::moe
