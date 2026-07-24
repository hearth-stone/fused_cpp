// SPDX-License-Identifier: Apache-2.0
#pragma once

#ifndef FUSED_CPP_MOE_SVE_VECTOR_BITS
#error "FUSED_CPP_MOE_SVE_VECTOR_BITS must be defined for the SVE MoE backend"
#endif

namespace fused_cpp::moe_sve {

inline constexpr int kVectorBits = FUSED_CPP_MOE_SVE_VECTOR_BITS;
static_assert(kVectorBits == 128 || kVectorBits == 256 || kVectorBits == 512 || kVectorBits == 1024 ||
                  kVectorBits == 2048,
              "fixed SVE vector length must be 128, 256, 512, 1024, or 2048 bits");

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
