#pragma once

#include <cstdint>

namespace fused_cpp::deepseek_v4::indexer_sve {

bool available();
int n_tile();
int round_n(int n);

// Pack K directly from paged-cache row offsets into the BFMMLA B layout.
// key_row_offsets contains element offsets from kv_cache for the N logical keys.
void pack_paged_k(const uint16_t* kv_cache, const int64_t* key_row_offsets, uint16_t* packed_k, int K, int N, int Np);

// Compute sum_h weights[m, h] * relu(dot(q[m, h], k[n])).  Q is row-major
// bf16 [M, H, K], packed_k is produced by pack_paged_k, and scores is fp32
// [M, Np].  Supported fast-path shapes have H % 8 == 0 and K % 4 == 0.
void weighted_relu_scores(const uint16_t* q, const float* weights, const uint16_t* packed_k, float* scores, int M,
                          int H, int K, int Np);

// Select the exact largest topk scores independently for every row.  Scores
// are contiguous within a row; output may be strided.  Indices are local to
// [row_starts[m], row_ends[m]) and are ordered by descending score, with lower
// indices first for ties.  The implementation allocates one reusable candidate
// buffer per OpenMP worker and writes indices only.
void batched_topk_indices(const float* scores, int64_t score_stride, int64_t score_columns, const int64_t* row_starts,
                          const int64_t* row_ends, int32_t* output, int64_t output_stride0, int64_t output_stride1,
                          int64_t M, int64_t topk);

}  // namespace fused_cpp::deepseek_v4::indexer_sve
