/*
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

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
};

enum BackendCapability : uint64_t {
  kFusedSiluPackC = uint64_t{1} << 0,
  kDirectRouteF32 = uint64_t{1} << 1,
  kDirectRouteBf16 = uint64_t{1} << 2,
  kRouteMerge = uint64_t{1} << 3,
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
