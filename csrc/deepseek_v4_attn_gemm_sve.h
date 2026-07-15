#pragma once

#include <cstdint>

namespace fused_cpp::deepseek_v4::attn_sve {

bool available();
bool enabled_by_env();
int round_k(int k);
int round_n(int n);
int n_tile();
int64_t a_scratch_elems(int64_t m, int64_t k);

void pack_b(const uint16_t* B, uint16_t* B_reo, int K, int N);

void dispatch_f32(const uint16_t* A, const uint16_t* B_reo, float* C, uint16_t* A_reorder, int M, int K, int N,
                  int ldc);

void dispatch_bf16(const uint16_t* A, const uint16_t* B_reo, uint16_t* C, uint16_t* A_reorder, int M, int K, int N,
                   int ldc);

}  // namespace fused_cpp::deepseek_v4::attn_sve
