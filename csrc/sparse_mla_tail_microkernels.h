// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/extension.h>

#include <cstdint>

namespace fused_cpp::sparse_mla_tail_microkernels {

// Thread-safe AArch64 BF16 helpers for packed sparse-MLA 8x8 tails. The Q/K
// layout is [d_qk/4][32 u16], scores is row-major [8][8], and all pointers must
// remain valid for the call. False means the mask or ISA is unsupported and the
// caller must execute its portable/full-tile fallback.
bool qkt_8x8_bf16_2x2_pruned(const uint16_t* q_packed, const uint16_t* k_packed, int64_t d_qk, float scale,
                             uint16_t active_2x2_mask, float* scores);

// P is row-major BF16 [8][8]. V is packed as [d_v/8][8][8], with
// v_evblock_stride BF16 elements between output-dimension blocks. Output is
// fp32 [8][d_v]. No alignment beyond that required by ordinary NEON loads is
// assumed. The implementation skips masked coefficients and is tolerance-
// equivalent to the full BF16 PV path for finite inputs.
bool pv_8x8_bf16_pruned(const at::BFloat16* p_bf16, const at::BFloat16* v_packed, int64_t v_evblock_stride, int64_t d_v,
                        float* output, int64_t output_row_stride, uint64_t valid_mask);

}  // namespace fused_cpp::sparse_mla_tail_microkernels
