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
