#include "deepseek_v4_indexer_sve.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "moe/arm/common/nm_window_schedule.h"

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && \
    (defined(__ARM_FEATURE_BF16) || defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC))
#include <arm_sve.h>
#endif

namespace fused_cpp::deepseek_v4::indexer_sve {
namespace {

inline int round_up(int value, int quantum) { return ((value + quantum - 1) / quantum) * quantum; }

struct TopkCandidate {
  float score;
  int32_t index;
};

struct TopkCandidateGreater {
  bool operator()(const TopkCandidate& lhs, const TopkCandidate& rhs) const {
    const bool lhs_nan = std::isnan(lhs.score);
    const bool rhs_nan = std::isnan(rhs.score);
    if (lhs_nan != rhs_nan) {
      return lhs_nan;
    }
    if (lhs_nan || lhs.score == rhs.score) {
      return lhs.index < rhs.index;
    }
    return lhs.score > rhs.score;
  }
};

void select_topk_row(const float* scores, int64_t valid_len, int64_t topk, TopkCandidate* candidates, int32_t* output,
                     int64_t output_stride) {
  for (int64_t i = 0; i < valid_len; ++i) {
    candidates[i] = TopkCandidate{scores[i], static_cast<int32_t>(i)};
  }

  const int64_t k_take = std::min(topk, valid_len);
  if (k_take == 0) {
    return;
  }
  TopkCandidate* const begin = candidates;
  TopkCandidate* const middle = begin + k_take;
  TopkCandidate* const end = begin + valid_len;
  const TopkCandidateGreater greater;
  if (middle != end) {
    std::nth_element(begin, middle, end, greater);
  }
  std::sort(begin, middle, greater);
  for (int64_t i = 0; i < k_take; ++i) {
    output[i * output_stride] = candidates[i].index;
  }
}

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && \
    (defined(__ARM_FEATURE_BF16) || defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC))

inline svbfloat16_t load_bf16(const uint16_t* ptr) {
  return svld1_bf16(svptrue_b16(), reinterpret_cast<const __bf16*>(ptr));
}

inline void pack_q_head_block(const uint16_t* q, uint16_t* packed_q, int K) {
  for (int kb = 0; kb < K; kb += 4) {
    for (int row_pair = 0; row_pair < 4; ++row_pair) {
      const int row0 = 2 * row_pair;
      const int row1 = row0 + 1;
      uint16_t* dst = packed_q + static_cast<int64_t>(kb / 4) * 32 + row_pair * 8;
      for (int k = 0; k < 4; ++k) {
        dst[k] = q[static_cast<int64_t>(row0) * K + kb + k];
        dst[4 + k] = q[static_cast<int64_t>(row1) * K + kb + k];
      }
    }
  }
}

inline svfloat32_t relu_weight_accumulate(svfloat32_t score, svfloat32_t value, svfloat32_t weight) {
  const svbool_t pg = svptrue_b32();
  return svmla_f32_x(pg, score, svmax_n_f32_x(pg, value, 0.0f), weight);
}

inline svfloat32_t reduce_head_pairs(svfloat32_t score) {
  const svbool_t pg = svptrue_b32();
  const svuint64_t score64 = svreinterpret_u64_f32(score);
  // BFMMLA produces [even_h c0, even_h c1, odd_h c0, odd_h c1]
  // independently in every 128-bit segment.  TRN duplicates the adjacent
  // u64 column pairs, the add reduces the two head parities, and UZP1 packs
  // one copy of every reduced column pair into the low half of the vector.
  const svfloat32_t even = svreinterpret_f32_u64(svtrn1_u64(score64, score64));
  const svfloat32_t odd = svreinterpret_f32_u64(svtrn2_u64(score64, score64));
  const svuint64_t reduced64 = svreinterpret_u64_f32(svadd_f32_x(pg, even, odd));
  return svreinterpret_f32_u64(svuzp1_u64(reduced64, reduced64));
}

inline void scatter_reduced_score(float* output, int column_pair, svfloat32_t score) {
  const uint64_t packed_lanes = static_cast<uint64_t>(svcntw() / 2);
  const svbool_t pg = svwhilelt_b32(uint64_t{0}, packed_lanes);
  const svuint32_t lane = svindex_u32(0, 1);
  const svuint32_t segment = svlsr_n_u32_x(pg, lane, 1);
  const svuint32_t column_in_pair = svand_n_u32_x(pg, lane, 1);
  svuint32_t offsets = svadd_u32_x(pg, svlsl_n_u32_x(pg, segment, 3), column_in_pair);
  offsets = svadd_n_u32_x(pg, offsets, static_cast<uint32_t>(2 * column_pair));
  svst1_scatter_u32index_f32(pg, output, offsets, reduce_head_pairs(score));
}

__attribute__((noinline)) void weighted_relu_tile(const uint16_t* packed_q, const float* weights,
                                                  const uint16_t* packed_k, float* scores, int H, int K) {
  const svbool_t pg_bf16 = svptrue_b16();
  const int lanes_h = static_cast<int>(svcnth());

  svfloat32_t score0 = svdup_f32(0.0f);
  svfloat32_t score1 = svdup_f32(0.0f);
  svfloat32_t score2 = svdup_f32(0.0f);
  svfloat32_t score3 = svdup_f32(0.0f);

  for (int head_base = 0; head_base < H; head_base += 8) {
    svfloat32_t c00 = svdup_f32(0.0f), c01 = svdup_f32(0.0f);
    svfloat32_t c02 = svdup_f32(0.0f), c03 = svdup_f32(0.0f);
    svfloat32_t c10 = svdup_f32(0.0f), c11 = svdup_f32(0.0f);
    svfloat32_t c12 = svdup_f32(0.0f), c13 = svdup_f32(0.0f);
    svfloat32_t c20 = svdup_f32(0.0f), c21 = svdup_f32(0.0f);
    svfloat32_t c22 = svdup_f32(0.0f), c23 = svdup_f32(0.0f);
    svfloat32_t c30 = svdup_f32(0.0f), c31 = svdup_f32(0.0f);
    svfloat32_t c32 = svdup_f32(0.0f), c33 = svdup_f32(0.0f);

    const uint16_t* q_ptr = packed_q + static_cast<int64_t>(head_base) * K;
    const uint16_t* k_ptr = packed_k;
    for (int kb = 0; kb < K; kb += 4) {
      const svbfloat16_t b0 = load_bf16(k_ptr + 0 * lanes_h);
      const svbfloat16_t b1 = load_bf16(k_ptr + 1 * lanes_h);
      const svbfloat16_t b2 = load_bf16(k_ptr + 2 * lanes_h);
      const svbfloat16_t b3 = load_bf16(k_ptr + 3 * lanes_h);
      k_ptr += 4 * lanes_h;

      const svbfloat16_t a0 = svld1rq_bf16(pg_bf16, reinterpret_cast<const __bf16*>(q_ptr + 0));
      const svbfloat16_t a1 = svld1rq_bf16(pg_bf16, reinterpret_cast<const __bf16*>(q_ptr + 8));
      const svbfloat16_t a2 = svld1rq_bf16(pg_bf16, reinterpret_cast<const __bf16*>(q_ptr + 16));
      const svbfloat16_t a3 = svld1rq_bf16(pg_bf16, reinterpret_cast<const __bf16*>(q_ptr + 24));
      q_ptr += 32;

      c00 = svbfmmla_f32(c00, a0, b0);
      c01 = svbfmmla_f32(c01, a0, b1);
      c02 = svbfmmla_f32(c02, a0, b2);
      c03 = svbfmmla_f32(c03, a0, b3);
      c10 = svbfmmla_f32(c10, a1, b0);
      c11 = svbfmmla_f32(c11, a1, b1);
      c12 = svbfmmla_f32(c12, a1, b2);
      c13 = svbfmmla_f32(c13, a1, b3);
      c20 = svbfmmla_f32(c20, a2, b0);
      c21 = svbfmmla_f32(c21, a2, b1);
      c22 = svbfmmla_f32(c22, a2, b2);
      c23 = svbfmmla_f32(c23, a2, b3);
      c30 = svbfmmla_f32(c30, a3, b0);
      c31 = svbfmmla_f32(c31, a3, b1);
      c32 = svbfmmla_f32(c32, a3, b2);
      c33 = svbfmmla_f32(c33, a3, b3);
    }

    const float* head_weights = weights + head_base;
    svfloat32_t weight = svdupq_n_f32(head_weights[0], head_weights[0], head_weights[1], head_weights[1]);
    score0 = relu_weight_accumulate(score0, c00, weight);
    score1 = relu_weight_accumulate(score1, c01, weight);
    score2 = relu_weight_accumulate(score2, c02, weight);
    score3 = relu_weight_accumulate(score3, c03, weight);

    weight = svdupq_n_f32(head_weights[2], head_weights[2], head_weights[3], head_weights[3]);
    score0 = relu_weight_accumulate(score0, c10, weight);
    score1 = relu_weight_accumulate(score1, c11, weight);
    score2 = relu_weight_accumulate(score2, c12, weight);
    score3 = relu_weight_accumulate(score3, c13, weight);

    weight = svdupq_n_f32(head_weights[4], head_weights[4], head_weights[5], head_weights[5]);
    score0 = relu_weight_accumulate(score0, c20, weight);
    score1 = relu_weight_accumulate(score1, c21, weight);
    score2 = relu_weight_accumulate(score2, c22, weight);
    score3 = relu_weight_accumulate(score3, c23, weight);

    weight = svdupq_n_f32(head_weights[6], head_weights[6], head_weights[7], head_weights[7]);
    score0 = relu_weight_accumulate(score0, c30, weight);
    score1 = relu_weight_accumulate(score1, c31, weight);
    score2 = relu_weight_accumulate(score2, c32, weight);
    score3 = relu_weight_accumulate(score3, c33, weight);
  }

  scatter_reduced_score(scores, 0, score0);
  scatter_reduced_score(scores, 1, score1);
  scatter_reduced_score(scores, 2, score2);
  scatter_reduced_score(scores, 3, score3);
}

#endif

}  // namespace

bool available() {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && \
    (defined(__ARM_FEATURE_BF16) || defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC))
  return true;
#else
  return false;
#endif
}

int n_tile() {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && \
    (defined(__ARM_FEATURE_BF16) || defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC))
  return static_cast<int>(svcntb() / 2);
#else
  return 8;
#endif
}

int round_n(int n) {
  const int tile = n_tile();
  if (n < 0 || n > std::numeric_limits<int>::max() - (tile - 1)) {
    throw std::invalid_argument("DeepSeek V4 indexer N cannot be rounded safely");
  }
  return round_up(std::max(n, 1), tile);
}

void pack_paged_k(const uint16_t* kv_cache, const int64_t* key_row_offsets, uint16_t* packed_k, int K, int N, int Np) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && \
    (defined(__ARM_FEATURE_BF16) || defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC))
  const int segments = static_cast<int>(svcntb() / 16);
  const int tile = segments * 8;
  const int tile_count = Np / tile;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (tile_count > 1)
#endif
  for (int tile_index = 0; tile_index < tile_count; ++tile_index) {
    const int column_base = tile_index * tile;
    uint16_t* dst = packed_k + static_cast<int64_t>(tile_index) * K * tile;
    int64_t out = 0;
    for (int row_base = 0; row_base < K; row_base += 4) {
      for (int column_pair = 0; column_pair < 4; ++column_pair) {
        for (int segment = 0; segment < segments; ++segment) {
          const int column0 = column_base + segment * 8 + column_pair * 2;
          const int column1 = column0 + 1;
          const int64_t offset0 = column0 < N ? key_row_offsets[column0] : -1;
          const int64_t offset1 = column1 < N ? key_row_offsets[column1] : -1;
          for (int k = 0; k < 4; ++k) {
            dst[out++] = offset0 >= 0 ? kv_cache[offset0 + row_base + k] : static_cast<uint16_t>(0);
          }
          for (int k = 0; k < 4; ++k) {
            dst[out++] = offset1 >= 0 ? kv_cache[offset1 + row_base + k] : static_cast<uint16_t>(0);
          }
        }
      }
    }
  }
#else
  (void)kv_cache;
  (void)key_row_offsets;
  (void)packed_k;
  (void)K;
  (void)N;
  (void)Np;
  throw std::runtime_error("DeepSeek V4 indexer SVE BF16 pack is unavailable");
#endif
}

void weighted_relu_scores(const uint16_t* q, const float* weights, const uint16_t* packed_k, float* scores, int M,
                          int H, int K, int Np) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && \
    (defined(__ARM_FEATURE_BF16) || defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC))
  const int tile = n_tile();
  const int64_t n_tiles = Np / tile;
  int64_t requested_threads = 1;
#ifdef _OPENMP
  requested_threads = omp_get_max_threads();
#endif
  const fused_cpp::nm_window::Geometry geometry = fused_cpp::nm_window::Choose(
      n_tiles, static_cast<int64_t>(K) * tile * static_cast<int64_t>(sizeof(uint16_t)), M, requested_threads);
  const int64_t task_count = geometry.task_count();

#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(std::min<int64_t>(requested_threads, task_count)))
#endif
  {
    std::vector<uint16_t> packed_q(static_cast<int64_t>(H) * K);
#ifdef _OPENMP
#pragma omp for schedule(static, 1)
#endif
    for (int64_t task = 0; task < task_count; ++task) {
      const int64_t window = task / geometry.m_splits;
      const int64_t m_split = task % geometry.m_splits;
      const int64_t tile_begin = window * n_tiles / geometry.n_windows;
      const int64_t tile_end = (window + 1) * n_tiles / geometry.n_windows;
      const int64_t m_begin = m_split * M / geometry.m_splits;
      const int64_t m_end = (m_split + 1) * M / geometry.m_splits;
      for (int64_t m = m_begin; m < m_end; ++m) {
        const uint16_t* q_row = q + m * H * K;
        for (int head_base = 0; head_base < H; head_base += 8) {
          pack_q_head_block(q_row + static_cast<int64_t>(head_base) * K,
                            packed_q.data() + static_cast<int64_t>(head_base) * K, K);
        }
        const float* weight_row = weights + m * H;
        float* score_row = scores + m * Np;
        for (int64_t n_tile_index = tile_begin; n_tile_index < tile_end; ++n_tile_index) {
          const int64_t column_base = n_tile_index * tile;
          const uint16_t* packed_k_tile = packed_k + n_tile_index * K * tile;
          weighted_relu_tile(packed_q.data(), weight_row, packed_k_tile, score_row + column_base, H, K);
        }
      }
    }
  }
#else
  (void)q;
  (void)weights;
  (void)packed_k;
  (void)scores;
  (void)M;
  (void)H;
  (void)K;
  (void)Np;
  throw std::runtime_error("DeepSeek V4 indexer SVE BF16 kernel is unavailable");
#endif
}

void batched_topk_indices(const float* scores, int64_t score_stride, int64_t score_columns, const int64_t* row_starts,
                          const int64_t* row_ends, int32_t* output, int64_t output_stride0, int64_t output_stride1,
                          int64_t M, int64_t topk) {
  if (score_stride < score_columns || score_columns < 0 || output_stride0 <= 0 || output_stride1 <= 0 || M < 0 ||
      topk < 0) {
    throw std::invalid_argument("DeepSeek V4 indexer batched TopK received invalid shape or stride");
  }
  if (topk > 0 &&
      (output_stride1 > std::numeric_limits<int64_t>::max() / topk || output_stride0 < topk * output_stride1)) {
    throw std::invalid_argument("DeepSeek V4 indexer batched TopK output rows may overlap");
  }

  int64_t max_valid_len = 0;
  for (int64_t m = 0; m < M; ++m) {
    const int64_t row_start = row_starts[m];
    const int64_t row_end = row_ends[m];
    if (row_start < 0 || row_end < row_start || row_end > score_columns ||
        row_end - row_start > std::numeric_limits<int32_t>::max()) {
      throw std::invalid_argument("DeepSeek V4 indexer batched TopK received an invalid row range");
    }
    max_valid_len = std::max(max_valid_len, row_end - row_start);
  }
  if (M == 0 || topk == 0 || max_valid_len == 0) {
    return;
  }

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    std::vector<TopkCandidate> candidates(static_cast<size_t>(max_valid_len));
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int64_t m = 0; m < M; ++m) {
      const int64_t row_start = row_starts[m];
      const int64_t valid_len = row_ends[m] - row_start;
      select_topk_row(scores + m * score_stride + row_start, valid_len, topk, candidates.data(),
                      output + m * output_stride0, output_stride1);
    }
  }
}

}  // namespace fused_cpp::deepseek_v4::indexer_sve
