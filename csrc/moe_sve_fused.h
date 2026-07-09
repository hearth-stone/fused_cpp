#pragma once

#include <cstdint>

#include "gemm_params.h"

namespace fused_cpp::moe_sve {

enum class Backend : int64_t {
    kNeon = 0,
    kSve = 1,
};

bool available();
bool enabled_by_env();
int n_tile();
int round_k(int k);
int round_n(int n);

void pack_b(const uint16_t* B, uint16_t* B_reo, int K, int N);

void pack_a_block(const uint16_t* A, uint16_t* packed, int rows, int K);

void gather_pack_a(const uint16_t* input, int64_t H,
                   const int64_t* expert_routes, int64_t top_k,
                   uint16_t* packed, int total_rows, int K_pad,
                   int block_begin, int block_end);

void w13_silu_rowmajor(const uint16_t* A, const uint16_t* B_reo,
                       uint16_t* C, uint16_t* A_reorder,
                       const gemm_params_t* params, int64_t degree);

void w13_silu_packed(const uint16_t* packed_A, const uint16_t* B_reo,
                     uint16_t* C, const gemm_params_t* params,
                     int64_t degree);

void w13_silu_packc(const uint16_t* packed_A, const uint16_t* B_reo,
                    uint16_t* C, const gemm_params_t* params,
                    int64_t rows, int64_t degree);

void w2_packed(const uint16_t* packed_A, const uint16_t* B_reo,
               float* C, const gemm_params_t* params, int64_t rows);

void w2_rowmajor(const uint16_t* A, const uint16_t* B_reo, float* C,
                 uint16_t* A_reorder, const gemm_params_t* params);

}  // namespace fused_cpp::moe_sve
