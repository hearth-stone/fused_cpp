// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace fused_cpp::moe::x86::avx512_bf16 {

// ISA-owned entry points. A stores each logical M12 panel in a 16-lane VNNI2
// block: every K-pair contains one uint32 BF16 pair for each physical row lane.
// Packed B uses the same K-pair convention along N. The executor validates ISA
// support before entering.
void ComputeW13(const uint16_t* a, int a_stride, const uint16_t* packed_b, uint16_t* c, int c_stride, int rows,
                int k_pad, int feature_block_begin, int feature_block_end, int silu_poly_degree);

void ComputeW2(const uint16_t* a, int a_stride, const uint16_t* packed_b, float* route_output, uint16_t* direct_output,
               const int64_t* route_ids, int route_stride, int rows, int k_pad, int hidden_size, int output_block_begin,
               int output_block_end, bool direct_bf16);

void MergeRoutes(const float* route_output, const float* weights, uint16_t* output, int64_t token_begin,
                 int64_t token_end, int64_t top_k, int64_t hidden_size);

}  // namespace fused_cpp::moe::x86::avx512_bf16
