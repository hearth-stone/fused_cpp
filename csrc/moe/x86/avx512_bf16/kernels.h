// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fused_cpp::moe::x86::avx512_bf16 {

enum class AmxPackedBLayout : uint8_t {
  kN32,
  // Within each N64 superblock, store [K32 x left N32][K32 x right N32]
  // for every K32 chunk. M1N4 therefore streams one contiguous 4 KiB chunk.
  kN64K32,
};

// ISA-owned entry points. A stores each logical M12 panel in a 16-lane VNNI2
// block: every K-pair contains one uint32 BF16 pair for each physical row lane.
// Packed B uses the same K-pair convention along N. The executor validates ISA
// support before entering.
void ComputeW13(const uint16_t* a, int a_stride, const uint16_t* packed_b, uint16_t* c, int c_stride, int rows,
                int k_pad, int feature_block_begin, int feature_block_end, int silu_poly_degree,
                int cooperative_threads = 1);

void ComputeW2(const uint16_t* a, int a_stride, const uint16_t* packed_b, float* route_output, uint16_t* direct_output,
               const int64_t* route_ids, int route_stride, int rows, int k_pad, int hidden_size, int output_block_begin,
               int output_block_end, bool direct_bf16, const float* route_weights = nullptr,
               int cooperative_threads = 1);

// Intrinsic implementations are kept as the per-call fallback for JIT cache
// misses/failures and for FUSED_CPP_MOE_AVX512_IMPL=intrinsic.
void ComputeW13Intrinsic(const uint16_t* a, int a_stride, const uint16_t* packed_b, uint16_t* c, int c_stride, int rows,
                         int k_pad, int feature_block_begin, int feature_block_end, int silu_poly_degree);

void ComputeW2Intrinsic(const uint16_t* a, int a_stride, const uint16_t* packed_b, float* route_output,
                        uint16_t* direct_output, const int64_t* route_ids, int route_stride, int rows, int k_pad,
                        int hidden_size, int output_block_begin, int output_block_end, bool direct_bf16,
                        const float* route_weights = nullptr);

// Resolve all exact-M kernels needed by the current routing plan before worker
// threads start. K is intentionally dynamic and is not part of the cache key.
void PrepareJitKernels(const std::vector<int>& row_counts, int silu_poly_degree, int hidden_size, bool direct_bf16,
                       bool weighted_direct_bf16 = false);

// AMX consumes row-major M16 panels and the same K-pair/N32 packed weights as
// AVX-512. K is padded to 32 BF16 elements by the AMX backend.
void PrepareAmxJitKernels(const std::vector<int>& row_counts, int silu_poly_degree, int hidden_size, bool direct_bf16,
                          bool weighted_direct_bf16 = false, AmxPackedBLayout b_layout = AmxPackedBLayout::kN32,
                          int intermediate_size = 0);

// True when weighted AMX W2 output is stored in expert-contiguous row order
// and the merge must translate flat route ids through a row map.
bool AmxW2UsesContiguousRouteOutput();

void ComputeW13Amx(const uint16_t* a, int a_stride, const uint16_t* packed_b, uint16_t* c, int c_stride, int rows,
                   int k_pad, int feature_block_begin, int feature_block_end, int silu_poly_degree,
                   AmxPackedBLayout b_layout = AmxPackedBLayout::kN32, int hidden_size = 0, int intermediate_size = 0);

void ComputeW2Amx(const uint16_t* a, int a_stride, const uint16_t* packed_b, float* route_output,
                  uint16_t* direct_output, const int64_t* route_ids, int route_stride, int rows, int k_pad,
                  int hidden_size, int output_block_begin, int output_block_end, bool direct_bf16,
                  const float* route_weights = nullptr, AmxPackedBLayout b_layout = AmxPackedBLayout::kN32,
                  int intermediate_size = 0);

struct JitStats {
  uint64_t kernel_count = 0;
  uint64_t code_bytes = 0;
  uint64_t generation_nanoseconds = 0;
};

JitStats GetJitStats();

void MergeRoutes(const float* route_output, const float* weights, uint16_t* output, int64_t token_begin,
                 int64_t token_end, int64_t top_k, int64_t hidden_size);

void MergeRoutesMapped(const float* route_output, const int64_t* route_rows, const float* weights, uint16_t* output,
                       int64_t token_begin, int64_t token_end, int64_t top_k, int64_t hidden_size);

}  // namespace fused_cpp::moe::x86::avx512_bf16
