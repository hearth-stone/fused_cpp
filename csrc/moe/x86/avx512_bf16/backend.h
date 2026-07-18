// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace fused_cpp::moe::x86::avx512_bf16 {

// These baseline-ISA helpers own runtime detection and the VNNI2 packed-B
// layout. Callers must check RuntimeSupported() before entering kernels.cpp.
bool RuntimeSupported();
int NTile();
int RoundK(int value);
int RoundN(int value);
void PackB(const uint16_t* source, uint16_t* packed, int k_size, int n_size);

void PackW13(const uint16_t* weight, uint16_t* packed, int64_t f_size, int64_t h_size, int k_pad, int f_pad);
void PackW2(const uint16_t* weight, uint16_t* packed, int64_t h_size, int64_t f_size, int k_pad, int n_pad);

}  // namespace fused_cpp::moe::x86::avx512_bf16
