// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace fused_cpp::moe::x86::avx512_bf16 {

// These x86 BF16 helpers own AVX-512/AMX runtime detection and the shared
// K-pair packed-B layout. Callers must resolve the matching backend first.
bool RuntimeSupported();
bool AmxRuntimeSupported();
bool EnsureAmxThreadPermission();
int NTile();
int RoundK(int value);
int RoundN(int value);
int AmxRoundK(int value);
void PackB(const uint16_t* source, uint16_t* packed, int k_size, int n_size);

void PackW13(const uint16_t* weight, uint16_t* packed, int64_t f_size, int64_t h_size, int k_pad, int f_pad);
void PackW2(const uint16_t* weight, uint16_t* packed, int64_t h_size, int64_t f_size, int k_pad, int n_pad);

}  // namespace fused_cpp::moe::x86::avx512_bf16
