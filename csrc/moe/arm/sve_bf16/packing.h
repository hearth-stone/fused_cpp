/*
Copyright (c) Huawei Technologies Co., Ltd. 2023-2026. All rights reserved.

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

void gather_pack_a(const uint16_t* input, int64_t H, const int64_t* expert_routes, int64_t top_k, uint16_t* packed,
                   int total_rows, int K_pad, int block_begin, int block_end);

void w13_silu_rowmajor(const uint16_t* A, const uint16_t* B_reo, uint16_t* C, uint16_t* A_reorder,
                       const gemm_params_t* params, int64_t degree);

void w13_silu_packed(const uint16_t* packed_A, const uint16_t* B_reo, uint16_t* C, const gemm_params_t* params,
                     int64_t degree);

void w13_silu_packc(const uint16_t* packed_A, const uint16_t* B_reo, uint16_t* C, const gemm_params_t* params,
                    int64_t rows, int64_t degree);

void w2_packed(const uint16_t* packed_A, const uint16_t* B_reo, float* C, const gemm_params_t* params, int64_t rows);

void w2_rowmajor(const uint16_t* A, const uint16_t* B_reo, float* C, uint16_t* A_reorder, const gemm_params_t* params);

}  // namespace fused_cpp::moe_sve
