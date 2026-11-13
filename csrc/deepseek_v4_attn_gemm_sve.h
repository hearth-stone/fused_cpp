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

#pragma once

#include <cstdint>

namespace fused_cpp::deepseek_v4::attn_sve {

inline constexpr int kMPanelRows = 12;

bool available();
bool enabled_by_env();
int round_k(int k);
int round_n(int n);
int m_panel_rows();
int n_tile();
int64_t a_scratch_elems(int64_t m, int64_t k);
int64_t packed_a_elems(int64_t m, int64_t k);

void pack_b(const uint16_t* B, uint16_t* B_reo, int K, int N);
void pack_a_range(const uint16_t* A, uint16_t* packed, int M, int K, int panel_begin, int panel_end);

void dispatch_packed_f32(const uint16_t* packed_A, const uint16_t* B_reo, float* C, int M, int K, int N, int ldc);

void dispatch_packed_bf16(const uint16_t* packed_A, const uint16_t* B_reo, uint16_t* C, int M, int K, int N, int ldc);

void dispatch_f32(const uint16_t* A, const uint16_t* B_reo, float* C, uint16_t* A_reorder, int M, int K, int N,
                  int ldc);

void dispatch_bf16(const uint16_t* A, const uint16_t* B_reo, uint16_t* C, uint16_t* A_reorder, int M, int K, int N,
                   int ldc);

}  // namespace fused_cpp::deepseek_v4::attn_sve
