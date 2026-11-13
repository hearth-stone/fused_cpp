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

namespace fused_cpp::moe_route_merge {

// SVE weighted route merge for token-major [tokens, top_k, hidden] input.
// Fixed top-k 2/4/6/8 uses a compile-time adjacent binary reduction; other
// positive values use an ordered runtime loop. Output is bf16 bits in
// [tokens, hidden]. Ranges are thread-disjoint.
bool sve_available();

void merge_f32_sve(const float* route_output, const float* weights, uint16_t* output, int64_t token_begin,
                   int64_t token_end, int64_t top_k, int64_t hidden_size);

void merge_bf16_sve(const uint16_t* route_output, const float* weights, uint16_t* output, int64_t token_begin,
                    int64_t token_end, int64_t top_k, int64_t hidden_size);

}  // namespace fused_cpp::moe_route_merge
