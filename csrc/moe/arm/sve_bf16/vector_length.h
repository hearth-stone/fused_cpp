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

#ifndef FUSED_CPP_MOE_SVE_VECTOR_BITS
#error "FUSED_CPP_MOE_SVE_VECTOR_BITS must be defined for the SVE MoE backend"
#endif

namespace fused_cpp::moe_sve {

inline constexpr int kVectorBits = FUSED_CPP_MOE_SVE_VECTOR_BITS;
static_assert(kVectorBits == 128 || kVectorBits == 256,
              "this delivery supports fixed SVE vector lengths of 128 or 256 bits");

inline constexpr int kVectorBytes = kVectorBits / 8;
inline constexpr int kBf16Lanes = kVectorBits / 16;
inline constexpr int kF32Lanes = kVectorBits / 32;
inline constexpr int kNTile = kVectorBytes / 2;
inline constexpr int kSegments128 = kVectorBits / 128;

#if defined(__ARM_FEATURE_SVE_BITS)
static_assert(__ARM_FEATURE_SVE_BITS == kVectorBits,
              "compiler SVE vector length does not match FUSED_CPP_MOE_SVE_VECTOR_BITS");
#endif

}  // namespace fused_cpp::moe_sve
