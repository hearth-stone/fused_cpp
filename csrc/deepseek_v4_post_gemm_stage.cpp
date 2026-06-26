#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <tuple>

#include "deepseek_v4_q_norm_rope_sve.h"
#include "profile_utils.h"

#ifndef FUSED_CPP_STRICT_MODE
#define FUSED_CPP_STRICT_MODE 0
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

at::Tensor bf16_linear_to_dtype(at::Tensor input,
                                at::Tensor weight,
                                bool output_bf16,
                                int64_t nthreads);
at::Tensor bf16_linear_prepacked_to_dtype(at::Tensor input,
                                          at::Tensor packed_weight,
                                          int64_t K,
                                          int64_t N,
                                          int64_t Np,
                                          bool output_bf16,
                                          int64_t nthreads);

namespace {

using torch::indexing::Slice;
using ::fused_cpp::profile::TimePoint;

struct PostGemmStageProfile {
  double input_check_ms = 0.0;
  double main_q_gemm_ms = 0.0;
  double main_q_norm_rope_swa_insert_ms = 0.0;
  double indexer_q_gemm_ms = 0.0;
  double indexer_q_rope_weights_ms = 0.0;
  double mla_save_partial_states_ms = 0.0;
  double mla_compress_norm_rope_insert_ms = 0.0;
  double indexer_save_partial_states_ms = 0.0;
  double indexer_compress_norm_rope_insert_ms = 0.0;
  double sparse_indexer_short_path_ms = 0.0;
  double sparse_indexer_gather_ms = 0.0;
  double sparse_indexer_fold_q_ms = 0.0;
  double sparse_indexer_score_topk_ms = 0.0;
};

#if FUSED_CPP_ENABLE_PROFILING

bool PostGemmProfileEnabled() {
  return ::fused_cpp::profile::env_enabled(
      "FUSED_CPP_DEEPSEEK_V4_POST_GEMM_PROFILE");
}

void PrintPostGemmProfile(const PostGemmStageProfile& profile, double total_ms) {
  const double known_ms = profile.input_check_ms + profile.main_q_gemm_ms +
      profile.main_q_norm_rope_swa_insert_ms + profile.indexer_q_gemm_ms +
      profile.indexer_q_rope_weights_ms + profile.mla_save_partial_states_ms +
      profile.mla_compress_norm_rope_insert_ms + profile.indexer_save_partial_states_ms +
      profile.indexer_compress_norm_rope_insert_ms + profile.sparse_indexer_short_path_ms +
      profile.sparse_indexer_gather_ms + profile.sparse_indexer_fold_q_ms +
      profile.sparse_indexer_score_topk_ms;
  const double other_ms = std::max(0.0, total_ms - known_ms);
  const auto pct = [total_ms](double ms) -> double {
    return total_ms > 0.0 ? (100.0 * ms / total_ms) : 0.0;
  };

  std::cerr << std::fixed << std::setprecision(3)
            << "deepseek_v4_post_gemm_stage_profile"
            << " total_ms=" << total_ms
            << " input_check_ms=" << profile.input_check_ms << "(" << pct(profile.input_check_ms) << "%)"
            << " main_q_gemm_ms=" << profile.main_q_gemm_ms << "(" << pct(profile.main_q_gemm_ms) << "%)"
            << " main_q_norm_rope_swa_insert_ms=" << profile.main_q_norm_rope_swa_insert_ms
            << "(" << pct(profile.main_q_norm_rope_swa_insert_ms) << "%)"
            << " indexer_q_gemm_ms=" << profile.indexer_q_gemm_ms << "(" << pct(profile.indexer_q_gemm_ms) << "%)"
            << " indexer_q_rope_weights_ms=" << profile.indexer_q_rope_weights_ms
            << "(" << pct(profile.indexer_q_rope_weights_ms) << "%)"
            << " mla_save_partial_states_ms=" << profile.mla_save_partial_states_ms
            << "(" << pct(profile.mla_save_partial_states_ms) << "%)"
            << " mla_compress_norm_rope_insert_ms=" << profile.mla_compress_norm_rope_insert_ms
            << "(" << pct(profile.mla_compress_norm_rope_insert_ms) << "%)"
            << " indexer_save_partial_states_ms=" << profile.indexer_save_partial_states_ms
            << "(" << pct(profile.indexer_save_partial_states_ms) << "%)"
            << " indexer_compress_norm_rope_insert_ms=" << profile.indexer_compress_norm_rope_insert_ms
            << "(" << pct(profile.indexer_compress_norm_rope_insert_ms) << "%)"
            << " sparse_indexer_short_path_ms=" << profile.sparse_indexer_short_path_ms
            << "(" << pct(profile.sparse_indexer_short_path_ms) << "%)"
            << " sparse_indexer_gather_ms=" << profile.sparse_indexer_gather_ms
            << "(" << pct(profile.sparse_indexer_gather_ms) << "%)"
            << " sparse_indexer_fold_q_ms=" << profile.sparse_indexer_fold_q_ms
            << "(" << pct(profile.sparse_indexer_fold_q_ms) << "%)"
            << " sparse_indexer_score_topk_ms=" << profile.sparse_indexer_score_topk_ms
            << "(" << pct(profile.sparse_indexer_score_topk_ms) << "%)"
            << " other_ms=" << other_ms << "(" << pct(other_ms) << "%)"
            << '\n';
}

#endif  // FUSED_CPP_ENABLE_PROFILING

bool DeepSeekV4KvRopeWriteKvEnabled() {
  const char* value = std::getenv("FUSED_CPP_DEEPSEEK_V4_KV_ROPE_WRITE_KV");
  if (value == nullptr) {
    return false;
  }
  return !(value[0] == '\0' || std::strcmp(value, "0") == 0 ||
           std::strcmp(value, "false") == 0 || std::strcmp(value, "FALSE") == 0);
}

void CheckCpuTensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
}

void CheckDim(const at::Tensor& tensor, const char* name, int64_t dim) {
  TORCH_CHECK(tensor.dim() == dim, name, " must be ", dim, "-D, got ", tensor.dim(), "-D");
}

at::Tensor ToLongCpu(const at::Tensor& tensor) {
  return tensor.to(at::TensorOptions().dtype(at::kLong).device(at::kCPU));
}

at::Tensor LinearToDtype(const at::Tensor& input, const at::Tensor& weight, at::ScalarType dtype) {
  TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kFloat,
              "LinearToDtype only supports bf16/fp32 output with bf16gemm, got ",
              dtype);
  return ::bf16_linear_to_dtype(input, weight, dtype == at::kBFloat16, 0);
}

at::Tensor LinearPrepackedToDtype(const at::Tensor& input,
                                  const at::Tensor& packed_weight,
                                  int64_t K,
                                  int64_t N,
                                  int64_t Np,
                                  at::ScalarType dtype) {
  TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kFloat,
              "LinearPrepackedToDtype only supports bf16/fp32 output with bf16gemm, got ",
              dtype);
  return ::bf16_linear_prepacked_to_dtype(input, packed_weight, K, N, Np, dtype == at::kBFloat16, 0);
}

at::Tensor GptjRopeApply(const at::Tensor& x,
                         const at::Tensor& cos_sin_cache,
                         const at::Tensor& positions,
                         int64_t rope_head_dim) {
  TORCH_CHECK(rope_head_dim >= 0, "rope_head_dim must be non-negative");
  if (rope_head_dim == 0) {
    return x.to(at::kFloat);
  }
  TORCH_CHECK(rope_head_dim % 2 == 0, "rope_head_dim must be even, got ", rope_head_dim);
  CheckDim(cos_sin_cache, "cos_sin_cache", 2);
  TORCH_CHECK(cos_sin_cache.size(1) >= rope_head_dim,
              "cos_sin_cache last dim must cover rope_head_dim");

  const int64_t head_dim = x.size(-1);
  TORCH_CHECK(head_dim >= rope_head_dim,
              "x last dim must be >= rope_head_dim: ", head_dim, " vs ", rope_head_dim);
  const int64_t nope_dim = head_dim - rope_head_dim;
  const int64_t half = rope_head_dim / 2;

  at::Tensor x_float = x.to(at::kFloat);
  at::Tensor out = x_float.clone();
  at::Tensor rope = x_float.slice(-1, nope_dim, head_dim);
  at::Tensor even = rope.slice(-1, 0, rope_head_dim, 2);
  at::Tensor odd = rope.slice(-1, 1, rope_head_dim, 2);

  at::Tensor pos_long = ToLongCpu(positions).to(x.device());
  at::Tensor cs_rows;
  if (pos_long.dim() == 0) {
    cs_rows = cos_sin_cache.to(at::kFloat).index({pos_long.item<int64_t>()});
  } else {
    cs_rows = cos_sin_cache.to(at::kFloat).index_select(0, pos_long.reshape(-1));
    while (cs_rows.dim() < x.dim()) {
      cs_rows = cs_rows.unsqueeze(-2);
    }
  }
  at::Tensor cos_v = cs_rows.slice(-1, 0, half);
  at::Tensor sin_v = cs_rows.slice(-1, half, 2 * half);

  at::Tensor rotated = at::empty_like(rope);
  rotated.slice(-1, 0, rope_head_dim, 2).copy_(even * cos_v - odd * sin_v);
  rotated.slice(-1, 1, rope_head_dim, 2).copy_(odd * cos_v + even * sin_v);
  out.slice(-1, nope_dim, head_dim).copy_(rotated);
  return out;
}

at::Tensor GptjRopeApplyScalar(const at::Tensor& x,
                               const at::Tensor& cos_sin_cache,
                               int64_t position,
                               int64_t rope_head_dim) {
  at::Tensor pos = at::full({}, position, at::TensorOptions().dtype(at::kLong).device(x.device()));
  return GptjRopeApply(x, cos_sin_cache, pos, rope_head_dim);
}

void CheckPositionsInRange(const at::Tensor& positions_long,
                           int64_t num_tokens,
                           int64_t max_position) {
  if (num_tokens == 0) {
    return;
  }
  if (positions_long.dim() == 0) {
    const int64_t pos = positions_long.item<int64_t>();
    TORCH_CHECK(pos >= 0 && pos < max_position,
                "position out of cos_sin_cache range: ", pos, " vs ", max_position);
    return;
  }
  const int64_t* pos_data = positions_long.data_ptr<int64_t>();
  for (int64_t i = 0; i < num_tokens; ++i) {
    const int64_t pos = pos_data[i];
    TORCH_CHECK(pos >= 0 && pos < max_position,
                "position out of cos_sin_cache range: ", pos, " vs ", max_position);
  }
}

void CheckMainQKvShape(const at::Tensor& q, const at::Tensor& kv) {
  TORCH_CHECK(kv.size(0) == q.size(0), "kv token count must match q");
  TORCH_CHECK(kv.size(1) == q.size(2), "kv last dim must match q head_dim");
}

template <typename scalar_t>
void QNormRopeFusedImpl(const at::Tensor& q,
                        const at::Tensor& positions_long,
                        const at::Tensor& cos_sin_f,
                        double eps) {
  const int64_t num_tokens = q.size(0);
  const int64_t num_heads = q.size(1);
  const int64_t head_dim = q.size(2);
  const int64_t rope_head_dim = cos_sin_f.size(1);
  const int64_t nope_head_dim = head_dim - rope_head_dim;
  const int64_t rope_half = rope_head_dim / 2;
  const bool scalar_position = positions_long.dim() == 0;

  scalar_t* q_data = q.data_ptr<scalar_t>();
  const int64_t q_stride_t = q.stride(0);
  const int64_t q_stride_h = q.stride(1);
  const int64_t q_stride_d = q.stride(2);
  const int64_t* pos_data = scalar_position ? nullptr : positions_long.data_ptr<int64_t>();
  const int64_t scalar_pos = scalar_position ? positions_long.item<int64_t>() : 0;
  const float* cos_sin_data = cos_sin_f.data_ptr<float>();
  const int64_t cos_sin_stride = cos_sin_f.stride(0);

  const int64_t total = num_tokens * num_heads;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t linear_idx = 0; linear_idx < total; ++linear_idx) {
    const int64_t token = linear_idx / num_heads;
    const int64_t head = linear_idx - token * num_heads;
    scalar_t* q_row = q_data + token * q_stride_t + head * q_stride_h;

    float sum_sq = 0.0f;
    for (int64_t d = 0; d < head_dim; ++d) {
      const float v = static_cast<float>(q_row[d * q_stride_d]);
      sum_sq += v * v;
    }
    const float inv_rms =
        1.0f / std::sqrt(sum_sq / static_cast<float>(head_dim) + static_cast<float>(eps));

    for (int64_t d = 0; d < nope_head_dim; ++d) {
      const float v = static_cast<float>(q_row[d * q_stride_d]) * inv_rms;
      q_row[d * q_stride_d] = static_cast<scalar_t>(v);
    }

    if (rope_head_dim == 0) {
      continue;
    }

    const int64_t pos = scalar_position ? scalar_pos : pos_data[token];
    const float* cs_row = cos_sin_data + pos * cos_sin_stride;
    const float* cos_row = cs_row;
    const float* sin_row = cs_row + rope_half;
    for (int64_t pair = 0; pair < rope_half; ++pair) {
      const int64_t even_d = nope_head_dim + 2 * pair;
      const int64_t odd_d = even_d + 1;
      const float even_norm = static_cast<float>(
          static_cast<scalar_t>(static_cast<float>(q_row[even_d * q_stride_d]) * inv_rms));
      const float odd_norm = static_cast<float>(
          static_cast<scalar_t>(static_cast<float>(q_row[odd_d * q_stride_d]) * inv_rms));
      const float c = cos_row[pair];
      const float s = sin_row[pair];
      q_row[even_d * q_stride_d] = static_cast<scalar_t>(even_norm * c - odd_norm * s);
      q_row[odd_d * q_stride_d] = static_cast<scalar_t>(odd_norm * c + even_norm * s);
    }
  }
}

void QNormRopeFused(const at::Tensor& q,
                    const at::Tensor& positions,
                    const at::Tensor& cos_sin_cache,
                    double eps) {
  CheckDim(q, "q", 3);
  const int64_t num_tokens = q.size(0);
  const int64_t head_dim = q.size(2);
  const int64_t rope_head_dim = cos_sin_cache.size(1);
  const int64_t nope_head_dim = head_dim - rope_head_dim;
  TORCH_CHECK(nope_head_dim >= 0, "main q head_dim must be >= rope dim");
  TORCH_CHECK(rope_head_dim >= 0, "rope_head_dim must be non-negative");
  TORCH_CHECK(rope_head_dim % 2 == 0, "rope_head_dim must be even, got ", rope_head_dim);
  CheckDim(cos_sin_cache, "cos_sin_cache", 2);
  at::Tensor positions_long = ToLongCpu(positions).contiguous();
  TORCH_CHECK(
      positions_long.dim() == 0 || positions_long.numel() == num_tokens,
      "positions must be scalar or have num_tokens elements");
  at::Tensor cos_sin_f = cos_sin_cache.to(at::kFloat).contiguous();
  if (rope_head_dim != 0) {
    CheckPositionsInRange(positions_long, num_tokens, cos_sin_f.size(0));
  }
#if FUSED_CPP_STRICT_MODE
  TORCH_CHECK(
      q.scalar_type() == at::kBFloat16,
      "FUSED_CPP_STRICT_MODE QNormRopeFused only supports bf16 q, got ",
      q.scalar_type());
#endif
  if (::fused_cpp::deepseek_v4::q_norm_rope_fused_sve(q, positions_long, cos_sin_f, eps)) {
    return;
  }

#if FUSED_CPP_STRICT_MODE
  TORCH_CHECK(
      false,
      "FUSED_CPP_STRICT_MODE QNormRopeFused requires the SVE bf16 fast path; "
      "unsupported layout or target");
#else
  if (q.scalar_type() == at::kBFloat16) {
    QNormRopeFusedImpl<at::BFloat16>(q, positions_long, cos_sin_f, eps);
  } else if (q.scalar_type() == at::kFloat) {
    QNormRopeFusedImpl<float>(q, positions_long, cos_sin_f, eps);
  } else {
    TORCH_CHECK(false, "QNormRopeFused only supports bf16/fp32 q, got ", q.scalar_type());
  }
#endif
}

template <typename kv_t, typename cache_t>
void KvRopeCacheInsertFusedImpl(const at::Tensor& kv,
                                const at::Tensor& swa_kv_cache,
                                const at::Tensor& slot_mapping_long,
                                const at::Tensor& positions_long,
                                const at::Tensor& cos_sin_f,
                                bool do_rope) {
  const int64_t num_tokens = kv.size(0);
  const int64_t head_dim = kv.size(1);
  const int64_t rope_head_dim = cos_sin_f.size(1);
  const int64_t nope_head_dim = head_dim - rope_head_dim;
  const int64_t rope_half = rope_head_dim / 2;
  const bool scalar_position = positions_long.dim() == 0;

  kv_t* kv_data = kv.data_ptr<kv_t>();
  const int64_t kv_stride_t = kv.stride(0);
  const int64_t kv_stride_d = kv.stride(1);
  const int64_t* pos_data = scalar_position ? nullptr : positions_long.data_ptr<int64_t>();
  const int64_t scalar_pos = scalar_position ? positions_long.item<int64_t>() : 0;
  const float* cos_sin_data = cos_sin_f.data_ptr<float>();
  const int64_t cos_sin_stride = cos_sin_f.stride(0);
  const bool has_cache = swa_kv_cache.numel() != 0;
  const int64_t* slot_data = has_cache ? slot_mapping_long.data_ptr<int64_t>() : nullptr;

  cache_t* cache_data = nullptr;
  int64_t cache_stride_slot = 0;
  int64_t cache_stride_d = 0;
  int64_t cache_block_size = 0;
  int64_t cache_stride_block = 0;
  int64_t cache_stride_offset = 0;
  if (has_cache) {
    cache_data = swa_kv_cache.data_ptr<cache_t>();
    if (swa_kv_cache.dim() == 2) {
      cache_stride_slot = swa_kv_cache.stride(0);
      cache_stride_d = swa_kv_cache.stride(1);
    } else {
      cache_block_size = swa_kv_cache.size(1);
      cache_stride_block = swa_kv_cache.stride(0);
      cache_stride_offset = swa_kv_cache.stride(1);
      cache_stride_d = swa_kv_cache.stride(2);
    }
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t token = 0; token < num_tokens; ++token) {
    kv_t* kv_row = kv_data + token * kv_stride_t;

    if (do_rope && rope_head_dim != 0) {
      const int64_t pos = scalar_position ? scalar_pos : pos_data[token];
      const float* cs_row = cos_sin_data + pos * cos_sin_stride;
      const float* cos_row = cs_row;
      const float* sin_row = cs_row + rope_half;
      for (int64_t pair = 0; pair < rope_half; ++pair) {
        const int64_t even_d = nope_head_dim + 2 * pair;
        const int64_t odd_d = even_d + 1;
        const float even = static_cast<float>(kv_row[even_d * kv_stride_d]);
        const float odd = static_cast<float>(kv_row[odd_d * kv_stride_d]);
        const float c = cos_row[pair];
        const float s = sin_row[pair];
        kv_row[even_d * kv_stride_d] = static_cast<kv_t>(even * c - odd * s);
        kv_row[odd_d * kv_stride_d] = static_cast<kv_t>(odd * c + even * s);
      }
    }

    if (!has_cache) {
      continue;
    }
    const int64_t slot = slot_data[token];
    if (slot < 0) {
      continue;
    }

    cache_t* cache_row = nullptr;
    if (swa_kv_cache.dim() == 2) {
      cache_row = cache_data + slot * cache_stride_slot;
    } else {
      cache_row = cache_data + (slot / cache_block_size) * cache_stride_block +
          (slot % cache_block_size) * cache_stride_offset;
    }
    for (int64_t d = 0; d < head_dim; ++d) {
      cache_row[d * cache_stride_d] = static_cast<cache_t>(static_cast<float>(kv_row[d * kv_stride_d]));
    }
  }
}

template <typename kv_t>
void DispatchKvRopeCacheInsertFusedCacheDtype(const at::Tensor& kv,
                                              const at::Tensor& swa_kv_cache,
                                              const at::Tensor& slot_mapping_long,
                                              const at::Tensor& positions_long,
                                              const at::Tensor& cos_sin_f,
                                              bool do_rope) {
  if (swa_kv_cache.numel() == 0 || swa_kv_cache.scalar_type() == at::kBFloat16) {
    KvRopeCacheInsertFusedImpl<kv_t, at::BFloat16>(
        kv, swa_kv_cache, slot_mapping_long, positions_long, cos_sin_f, do_rope);
  } else if (swa_kv_cache.scalar_type() == at::kFloat) {
    KvRopeCacheInsertFusedImpl<kv_t, float>(
        kv, swa_kv_cache, slot_mapping_long, positions_long, cos_sin_f, do_rope);
  } else {
    TORCH_CHECK(
        false,
        "KvRopeCacheInsertFused only supports bf16/fp32 swa_kv_cache, got ",
        swa_kv_cache.scalar_type());
  }
}

void KvRopeCacheInsertFused(const at::Tensor& kv,
                            const at::Tensor& swa_kv_cache,
                            const at::Tensor& slot_mapping,
                            const at::Tensor& positions,
                            const at::Tensor& cos_sin_cache) {
  CheckDim(kv, "kv", 2);
  CheckDim(cos_sin_cache, "cos_sin_cache", 2);
  const int64_t num_tokens = kv.size(0);
  const int64_t head_dim = kv.size(1);
  const int64_t rope_head_dim = cos_sin_cache.size(1);
  const int64_t nope_head_dim = head_dim - rope_head_dim;
  TORCH_CHECK(nope_head_dim >= 0, "main kv head_dim must be >= rope dim");
  TORCH_CHECK(rope_head_dim >= 0, "rope_head_dim must be non-negative");
  TORCH_CHECK(rope_head_dim % 2 == 0, "rope_head_dim must be even, got ", rope_head_dim);
  at::Tensor positions_long = ToLongCpu(positions).contiguous();
  TORCH_CHECK(
      positions_long.dim() == 0 || positions_long.numel() == num_tokens,
      "positions must be scalar or have num_tokens elements");
  at::Tensor slots_long;
  if (swa_kv_cache.numel() != 0) {
    TORCH_CHECK(
        swa_kv_cache.dim() == 2 || swa_kv_cache.dim() == 3,
        "swa_kv_cache must be 2-D or 3-D");
    TORCH_CHECK(
        swa_kv_cache.size(-1) == head_dim,
        "swa_kv_cache last dim must match kv head_dim");
    slots_long = ToLongCpu(slot_mapping).reshape({-1}).contiguous();
    TORCH_CHECK(slots_long.numel() >= num_tokens, "slot_mapping must cover all kv tokens");
    const int64_t slot_capacity =
        swa_kv_cache.dim() == 2 ? swa_kv_cache.size(0) : swa_kv_cache.size(0) * swa_kv_cache.size(1);
    const int64_t* slot_data = slots_long.data_ptr<int64_t>();
    for (int64_t i = 0; i < num_tokens; ++i) {
      const int64_t slot = slot_data[i];
      TORCH_CHECK(slot < slot_capacity,
                  "slot_mapping slot exceeds swa_kv_cache capacity: ",
                  slot,
                  " vs ",
                  slot_capacity);
    }
  } else {
    slots_long = at::empty({0}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  }
  at::Tensor cos_sin_f = cos_sin_cache.to(at::kFloat).contiguous();
  if (rope_head_dim != 0) {
    CheckPositionsInRange(positions_long, num_tokens, cos_sin_f.size(0));
  }
#if FUSED_CPP_STRICT_MODE
  TORCH_CHECK(
      kv.scalar_type() == at::kBFloat16,
      "FUSED_CPP_STRICT_MODE KvRopeCacheInsertFused only supports bf16 kv, got ",
      kv.scalar_type());
  if (swa_kv_cache.numel() != 0) {
    TORCH_CHECK(
        swa_kv_cache.scalar_type() == at::kBFloat16,
        "FUSED_CPP_STRICT_MODE KvRopeCacheInsertFused only supports bf16 swa_kv_cache, got ",
        swa_kv_cache.scalar_type());
  }
#endif

  if (!DeepSeekV4KvRopeWriteKvEnabled()) {
    const bool cache_insert_done =
        ::fused_cpp::deepseek_v4::kv_rope_cache_insert_fused_sve(
            kv, swa_kv_cache, slots_long, positions_long, cos_sin_f);
    if (cache_insert_done) {
      return;
    }
#if FUSED_CPP_STRICT_MODE
    TORCH_CHECK(
        swa_kv_cache.numel() == 0,
        "FUSED_CPP_STRICT_MODE KvRopeCacheInsertFused requires the SVE bf16 cache-insert fast path");
#endif
  }

  const bool rope_done =
      ::fused_cpp::deepseek_v4::kv_rope_fused_sve(kv, positions_long, cos_sin_f);
  if (rope_done && swa_kv_cache.numel() == 0) {
    return;
  }

#if FUSED_CPP_STRICT_MODE
  TORCH_CHECK(
      false,
      "FUSED_CPP_STRICT_MODE KvRopeCacheInsertFused requires the SVE bf16 fast path; "
      "fallback kv copy/rope path is disabled");
#else
  if (kv.scalar_type() == at::kBFloat16) {
    DispatchKvRopeCacheInsertFusedCacheDtype<at::BFloat16>(
        kv, swa_kv_cache, slots_long, positions_long, cos_sin_f, !rope_done);
  } else if (kv.scalar_type() == at::kFloat) {
    DispatchKvRopeCacheInsertFusedCacheDtype<float>(
        kv, swa_kv_cache, slots_long, positions_long, cos_sin_f, !rope_done);
  } else {
    TORCH_CHECK(false, "KvRopeCacheInsertFused only supports bf16/fp32 kv, got ", kv.scalar_type());
  }
#endif
}

void SavePartialStates(const at::Tensor& kv,
                       const at::Tensor& score,
                       const at::Tensor& ape,
                       const at::Tensor& positions,
                       const at::Tensor& state_cache,
                       const at::Tensor& slot_mapping,
                       int64_t compress_ratio) {
  if (state_cache.numel() == 0) {
    return;
  }
  const int64_t num_tokens = kv.size(0);
  const int64_t block_size = state_cache.size(1);
  const int64_t state_width = state_cache.size(-1) / 2;
  at::Tensor slots = ToLongCpu(slot_mapping).reshape({-1});
  at::Tensor pos_cpu = ToLongCpu(positions).reshape({-1});

  for (int64_t i = 0; i < num_tokens; ++i) {
    const int64_t slot = slots[i].item<int64_t>();
    if (slot < 0) {
      continue;
    }
    const int64_t block = slot / block_size;
    const int64_t offset = slot % block_size;
    const int64_t ape_row = std::max<int64_t>(pos_cpu[i].item<int64_t>() % compress_ratio, 0);
    state_cache.index({block, offset, Slice(0, state_width)}).copy_(kv.index({i}).to(state_cache.scalar_type()));
    state_cache.index({block, offset, Slice(state_width, 2 * state_width)})
        .copy_((score.index({i}) + ape.index({ape_row})).to(state_cache.scalar_type()));
  }
}

void KvCompressNormRopeInsert(const at::Tensor& state_cache,
                              const at::Tensor& token_to_req_indices,
                              const at::Tensor& positions,
                              const at::Tensor& slot_mapping,
                              const at::Tensor& block_table,
                              const at::Tensor& rms_norm_weight,
                              double rms_norm_eps,
                              const at::Tensor& cos_sin_cache,
                              const at::Tensor& kv_cache,
                              const at::Tensor& kv_slot_mapping,
                              int64_t compress_ratio) {
  if (kv_cache.numel() == 0 || state_cache.numel() == 0) {
    return;
  }
  const int64_t num_tokens = positions.size(0);
  const int64_t state_block_size = state_cache.size(1);
  const int64_t state_width = state_cache.size(-1) / 2;
  const int64_t head_dim = rms_norm_weight.size(0);
  TORCH_CHECK(state_width % head_dim == 0, "state width must be a multiple of head_dim");
  const int64_t coff = state_width / head_dim;
  TORCH_CHECK(coff == 1 || coff == 2, "compressor coff must be 1 or 2, got ", coff);
  const int64_t window = coff * compress_ratio;
  const int64_t rope_head_dim = cos_sin_cache.size(1);
  const int64_t kv_cache_block_size = kv_cache.size(1);

  at::Tensor pos_cpu = ToLongCpu(positions).reshape({-1});
  at::Tensor slot_cpu = ToLongCpu(slot_mapping).reshape({-1});
  at::Tensor kv_slot_cpu = ToLongCpu(kv_slot_mapping).reshape({-1});
  at::Tensor req_cpu = ToLongCpu(token_to_req_indices).reshape({-1});
  at::Tensor rms_weight = rms_norm_weight.to(at::kFloat);

  for (int64_t i = 0; i < num_tokens; ++i) {
    const int64_t slot = slot_cpu[i].item<int64_t>();
    if (slot < 0) {
      continue;
    }
    const int64_t position = pos_cpu[i].item<int64_t>();
    if ((position + 1) % compress_ratio != 0) {
      continue;
    }
    const int64_t kv_slot = kv_slot_cpu[i].item<int64_t>();
    if (kv_slot < 0) {
      continue;
    }
    const int64_t req_idx = req_cpu[i].item<int64_t>();
    const int64_t start = position - window + 1;

    std::vector<at::Tensor> kv_rows;
    std::vector<at::Tensor> score_rows;
    kv_rows.reserve(static_cast<size_t>(window));
    score_rows.reserve(static_cast<size_t>(window));
    for (int64_t t = 0; t < window; ++t) {
      const int64_t p = start + t;
      if (p < 0) {
        kv_rows.push_back(at::zeros({head_dim}, state_cache.options().dtype(at::kFloat)));
        score_rows.push_back(at::full({head_dim}, -std::numeric_limits<float>::infinity(),
                                      state_cache.options().dtype(at::kFloat)));
        continue;
      }
      const int64_t logical_block = p / state_block_size;
      const int64_t logical_offset = p % state_block_size;
      const int64_t block = block_table.index({req_idx, logical_block}).item<int64_t>();
      at::Tensor row = state_cache.index({block, logical_offset}).to(at::kFloat);
      if (coff == 2 && t >= compress_ratio) {
        kv_rows.push_back(row.slice(0, head_dim, 2 * head_dim));
        score_rows.push_back(row.slice(0, state_width + head_dim, state_width + 2 * head_dim));
      } else {
        kv_rows.push_back(row.slice(0, 0, head_dim));
        score_rows.push_back(row.slice(0, state_width, state_width + head_dim));
      }
    }

    at::Tensor kv_stack = at::stack(kv_rows, 0);
    at::Tensor score_stack = at::stack(score_rows, 0);
    at::Tensor all_neg_inf = score_stack.eq(-std::numeric_limits<float>::infinity()).all(0, true);
    if (all_neg_inf.any().item<bool>()) {
      score_stack = at::where(all_neg_inf.expand_as(score_stack), at::zeros_like(score_stack), score_stack);
    }
    at::Tensor weights = at::softmax(score_stack, 0);
    at::Tensor compressed = (kv_stack * weights).sum(0);
    at::Tensor var = compressed.pow(2).mean(-1, false);
    at::Tensor normed = compressed * at::rsqrt(var + rms_norm_eps) * rms_weight;
    const int64_t compressed_pos = (position / compress_ratio) * compress_ratio;
    at::Tensor rotated = GptjRopeApplyScalar(normed, cos_sin_cache, compressed_pos, rope_head_dim);

    const int64_t kv_block = kv_slot / kv_cache_block_size;
    const int64_t kv_offset = kv_slot % kv_cache_block_size;
    kv_cache.index({kv_block, kv_offset}).copy_(rotated.to(kv_cache.scalar_type()));
  }
}

std::tuple<at::Tensor, at::Tensor> IndexerQRopeQuant(const at::Tensor& positions,
                                                     const at::Tensor& index_q,
                                                     const at::Tensor& cos_sin_cache,
                                                     const at::Tensor& index_weights) {
  const int64_t rope_dim = cos_sin_cache.size(1);
  at::Tensor q_rot = GptjRopeApply(index_q, cos_sin_cache, positions, rope_dim).to(at::kBFloat16);
  const double softmax_scale = std::pow(static_cast<double>(index_q.size(-1)), -0.5);
  const double head_scale = std::pow(static_cast<double>(index_q.size(1)), -0.5);
  at::Tensor weights = index_weights.to(at::kFloat) * softmax_scale * head_scale;
  return std::make_tuple(q_rot, weights);
}

void SparseAttnIndexerPrefill(const at::Tensor& q_quant,
                              const at::Tensor& weights,
                              const at::Tensor& kv_cache,
                              const at::Tensor& topk_indices_buffer,
                              int64_t topk_tokens,
                              const at::Tensor& cu_seq_lens,
                              const at::Tensor& cu_seqlen_ks,
                              const at::Tensor& cu_seqlen_ke,
                              const at::Tensor& block_table,
                              PostGemmStageProfile* profile) {
  const int64_t num_tokens = q_quant.size(0);
  const int64_t head_dim = q_quant.size(-1);
  const int64_t block_size = kv_cache.size(1);
  topk_indices_buffer.slice(0, 0, num_tokens).fill_(-1);

  at::Tensor ks_cpu = ToLongCpu(cu_seqlen_ks).reshape({-1});
  at::Tensor ke_cpu = ToLongCpu(cu_seqlen_ke).reshape({-1});
  at::Tensor valid_lens = ke_cpu - ks_cpu;
  if (valid_lens.numel() > 0 && valid_lens.max().item<int64_t>() <= topk_tokens) {
    FUSED_CPP_PROFILE_START(phase_start);
    for (int64_t i = 0; i < num_tokens; ++i) {
      const int64_t valid_len = valid_lens[i].item<int64_t>();
      if (valid_len <= 0) {
        continue;
      }
      topk_indices_buffer.index({i, Slice(0, valid_len)})
          .copy_(at::arange(valid_len, topk_indices_buffer.options().dtype(at::kInt)));
    }
    FUSED_CPP_PROFILE_ADD_IF_PTR(
        profile == nullptr ? nullptr : &profile->sparse_indexer_short_path_ms,
        phase_start);
    return;
  }

  at::Tensor cu_cpu = ToLongCpu(cu_seq_lens).reshape({-1});
  const int64_t num_reqs = cu_cpu.numel() - 1;
  const int64_t total_seq_lens = cu_cpu[-1].item<int64_t>();
  FUSED_CPP_PROFILE_START(phase_start);
  at::Tensor k_gathered = at::empty({total_seq_lens, head_dim}, q_quant.options().dtype(at::kFloat));
  for (int64_t req = 0; req < num_reqs; ++req) {
    const int64_t seq_start = cu_cpu[req].item<int64_t>();
    const int64_t seq_end = cu_cpu[req + 1].item<int64_t>();
    const int64_t seq_len = seq_end - seq_start;
    if (seq_len == 0) {
      continue;
    }
    const int64_t num_blocks = (seq_len + block_size - 1) / block_size;
    at::Tensor block_ids = block_table.index({req, Slice(0, num_blocks)}).to(at::kLong);
    at::Tensor gathered = kv_cache.index_select(0, block_ids).reshape({num_blocks * block_size, head_dim});
      k_gathered.index({Slice(seq_start, seq_end)}).copy_(gathered.slice(0, 0, seq_len).to(at::kFloat));
  }
  FUSED_CPP_PROFILE_ADD_IF_PTR(
      profile == nullptr ? nullptr : &profile->sparse_indexer_gather_ms,
      phase_start);

  FUSED_CPP_PROFILE_RESTART(phase_start);
  at::Tensor q_w = (q_quant.to(at::kFloat) * weights.to(at::kFloat).unsqueeze(-1)).sum(1);
  FUSED_CPP_PROFILE_ADD_IF_PTR(
      profile == nullptr ? nullptr : &profile->sparse_indexer_fold_q_ms,
      phase_start);

  FUSED_CPP_PROFILE_RESTART(phase_start);
  at::Tensor logits = at::matmul(q_w, k_gathered.t());
  for (int64_t i = 0; i < num_tokens; ++i) {
    const int64_t row_start = ks_cpu[i].item<int64_t>();
    const int64_t row_end = ke_cpu[i].item<int64_t>();
    const int64_t valid_len = row_end - row_start;
    if (valid_len <= 0) {
      continue;
    }
    const int64_t k_take = std::min<int64_t>(topk_tokens, valid_len);
    at::Tensor row = logits.index({i, Slice(row_start, row_end)});
    auto topk = row.topk(k_take, -1);
    at::Tensor idx = std::get<1>(topk).to(at::kInt);
    topk_indices_buffer.index({i, Slice(0, k_take)}).copy_(idx);
  }
  FUSED_CPP_PROFILE_ADD_IF_PTR(
      profile == nullptr ? nullptr : &profile->sparse_indexer_score_topk_ms,
      phase_start);
}

void RunCompressor(const at::Tensor& kv_score,
                   const at::Tensor& positions,
                   const at::Tensor& ape,
                   const at::Tensor& state_cache,
                   const at::Tensor& state_slot_mapping,
                   const at::Tensor& token_to_req_indices,
                   const at::Tensor& block_table,
                   const at::Tensor& kv_cache,
                   const at::Tensor& kv_slot_mapping,
                   const at::Tensor& norm_weight,
                   const at::Tensor& cos_sin_cache,
                   int64_t compress_ratio,
                   double rms_norm_eps,
                   double* save_partial_states_ms,
                   double* compress_norm_rope_insert_ms) {
  const int64_t state_width = ape.size(1);
  TORCH_CHECK(kv_score.size(1) == 2 * state_width,
              "kv_score last dim must be 2 * ape width");
  at::Tensor kv = kv_score.slice(1, 0, state_width);
  at::Tensor score = kv_score.slice(1, state_width, 2 * state_width);
  FUSED_CPP_PROFILE_START(phase_start);
  SavePartialStates(kv, score, ape, positions, state_cache, state_slot_mapping, compress_ratio);
  FUSED_CPP_PROFILE_ADD_IF_PTR(save_partial_states_ms, phase_start);
  FUSED_CPP_PROFILE_RESTART(phase_start);
  KvCompressNormRopeInsert(state_cache, token_to_req_indices, positions, state_slot_mapping,
                           block_table, norm_weight, rms_norm_eps, cos_sin_cache, kv_cache,
                           kv_slot_mapping, compress_ratio);
  FUSED_CPP_PROFILE_ADD_IF_PTR(compress_norm_rope_insert_ms, phase_start);
}

}  // namespace

// Expected dtype contract for DeepSeek V4 post-GEMM stage on Arm CPU:
// - qr: bf16
// - kv: bf16
// - kv_score / indexer_kv_score: fp32
// - indexer_weights: bf16
// - returned q: bf16
// - topk_indices_buffer: int32
// - swa_kv_cache: bf16
// - compressor state_cache: fp32
// - compressor kv_cache: bf16 on CPU
std::tuple<at::Tensor, at::Tensor> deepseek_v4_post_gemm_parallel_stage(
    at::Tensor qr,
    at::Tensor kv,
    at::Tensor kv_score,
    at::Tensor indexer_kv_score,
    at::Tensor indexer_weights,
    at::Tensor positions,
    at::Tensor main_wq_b_weight,
    at::Tensor indexer_wq_b_weight,
    at::Tensor main_cos_sin_cache,
    at::Tensor indexer_cos_sin_cache,
    at::Tensor swa_kv_cache,
    at::Tensor swa_slot_mapping,
    at::Tensor mla_ape,
    at::Tensor mla_state_cache,
    at::Tensor mla_state_slot_mapping,
    at::Tensor mla_token_to_req_indices,
    at::Tensor mla_block_table,
    at::Tensor mla_kv_cache,
    at::Tensor mla_kv_slot_mapping,
    at::Tensor mla_norm_weight,
    at::Tensor indexer_ape,
    at::Tensor indexer_state_cache,
    at::Tensor indexer_state_slot_mapping,
    at::Tensor indexer_token_to_req_indices,
    at::Tensor indexer_block_table,
    at::Tensor indexer_kv_cache,
    at::Tensor indexer_kv_slot_mapping,
    at::Tensor indexer_norm_weight,
    at::Tensor topk_indices_buffer,
    at::Tensor prefill_cu_seq_lens,
    at::Tensor prefill_cu_seqlen_ks,
    at::Tensor prefill_cu_seqlen_ke,
    at::Tensor prefill_block_table,
    int64_t main_head_dim,
    double q_eps,
    int64_t mla_compress_ratio,
    double mla_rms_norm_eps,
    int64_t indexer_compress_ratio,
    double indexer_rms_norm_eps,
    int64_t topk_tokens) {
#if FUSED_CPP_ENABLE_PROFILING
  const bool profile_enabled = PostGemmProfileEnabled();
  PostGemmStageProfile profile;
  PostGemmStageProfile* profile_ptr = profile_enabled ? &profile : nullptr;
#else
  PostGemmStageProfile* profile_ptr = nullptr;
#endif
  FUSED_CPP_PROFILE_START(total_start);
  FUSED_CPP_PROFILE_START(phase_start);

  CheckCpuTensor(qr, "qr");
  CheckCpuTensor(kv, "kv");
  CheckDim(qr, "qr", 2);
  CheckDim(kv, "kv", 2);
  TORCH_CHECK(main_head_dim > 0, "main_head_dim must be positive");
  TORCH_CHECK(main_wq_b_weight.size(0) % main_head_dim == 0,
              "main_wq_b_weight out features must be divisible by main_head_dim");
  TORCH_CHECK(indexer_wq_b_weight.size(0) % indexer_norm_weight.size(0) == 0,
              "indexer_wq_b_weight out features must be divisible by indexer head_dim");
  FUSED_CPP_PROFILE_ADD_IF_PTR(
      profile_ptr == nullptr ? nullptr : &profile_ptr->input_check_ms,
      phase_start);

  const int64_t main_num_heads = main_wq_b_weight.size(0) / main_head_dim;
  FUSED_CPP_PROFILE_RESTART(phase_start);
  at::Tensor q = LinearToDtype(qr, main_wq_b_weight, qr.scalar_type())
                     .reshape({qr.size(0), main_num_heads, main_head_dim});
  FUSED_CPP_PROFILE_ADD_IF_PTR(
      profile_ptr == nullptr ? nullptr : &profile_ptr->main_q_gemm_ms,
      phase_start);

  FUSED_CPP_PROFILE_RESTART(phase_start);
  CheckMainQKvShape(q, kv);
  QNormRopeFused(q, positions, main_cos_sin_cache, q_eps);
  KvRopeCacheInsertFused(kv, swa_kv_cache, swa_slot_mapping, positions, main_cos_sin_cache);
  FUSED_CPP_PROFILE_ADD_IF_PTR(
      profile_ptr == nullptr ? nullptr : &profile_ptr->main_q_norm_rope_swa_insert_ms,
      phase_start);

  const int64_t indexer_head_dim = indexer_norm_weight.size(0);
  FUSED_CPP_PROFILE_RESTART(phase_start);
  at::Tensor indexer_q_linear = LinearToDtype(qr, indexer_wq_b_weight, qr.scalar_type());
  TORCH_CHECK(indexer_q_linear.size(1) % indexer_head_dim == 0,
              "indexer q linear out features must be divisible by indexer head_dim");
  FUSED_CPP_PROFILE_ADD_IF_PTR(
      profile_ptr == nullptr ? nullptr : &profile_ptr->indexer_q_gemm_ms,
      phase_start);

  const int64_t indexer_num_heads = indexer_q_linear.size(1) / indexer_head_dim;
  at::Tensor indexer_q =
      indexer_q_linear.reshape({indexer_q_linear.size(0), indexer_num_heads, indexer_head_dim});

  FUSED_CPP_PROFILE_RESTART(phase_start);
  auto indexer_q_and_weights = IndexerQRopeQuant(positions, indexer_q, indexer_cos_sin_cache, indexer_weights);
  at::Tensor q_quant = std::get<0>(indexer_q_and_weights);
  at::Tensor scaled_weights = std::get<1>(indexer_q_and_weights);
  FUSED_CPP_PROFILE_ADD_IF_PTR(
      profile_ptr == nullptr ? nullptr : &profile_ptr->indexer_q_rope_weights_ms,
      phase_start);

  RunCompressor(kv_score, positions, mla_ape, mla_state_cache, mla_state_slot_mapping,
                mla_token_to_req_indices, mla_block_table, mla_kv_cache, mla_kv_slot_mapping,
                mla_norm_weight, main_cos_sin_cache, mla_compress_ratio, mla_rms_norm_eps,
                profile_ptr == nullptr ? nullptr : &profile_ptr->mla_save_partial_states_ms,
                profile_ptr == nullptr ? nullptr : &profile_ptr->mla_compress_norm_rope_insert_ms);
  RunCompressor(indexer_kv_score, positions, indexer_ape, indexer_state_cache,
                indexer_state_slot_mapping, indexer_token_to_req_indices, indexer_block_table,
                indexer_kv_cache, indexer_kv_slot_mapping, indexer_norm_weight,
                indexer_cos_sin_cache, indexer_compress_ratio, indexer_rms_norm_eps,
                profile_ptr == nullptr ? nullptr : &profile_ptr->indexer_save_partial_states_ms,
                profile_ptr == nullptr ? nullptr : &profile_ptr->indexer_compress_norm_rope_insert_ms);

  // Keep the migrated native post-GEMM path self-contained.  Do not dispatch
  // through the retired standalone sparse_attn_indexer_prefill_cpp_v0 symbol.
  SparseAttnIndexerPrefill(q_quant, scaled_weights, indexer_kv_cache, topk_indices_buffer,
                           topk_tokens, prefill_cu_seq_lens, prefill_cu_seqlen_ks,
                           prefill_cu_seqlen_ke, prefill_block_table, profile_ptr);
  FUSED_CPP_PROFILE_IF_ENABLED(
      profile_ptr != nullptr,
      PrintPostGemmProfile(
          *profile_ptr,
          ::fused_cpp::profile::elapsed_ms(total_start)));
  return std::make_tuple(q, topk_indices_buffer);
}

std::tuple<at::Tensor, at::Tensor> deepseek_v4_post_gemm_parallel_stage_prepacked(
    at::Tensor qr,
    at::Tensor kv,
    at::Tensor kv_score,
    at::Tensor indexer_kv_score,
    at::Tensor indexer_weights,
    at::Tensor positions,
    at::Tensor main_wq_b_packed,
    int64_t main_wq_b_K,
    int64_t main_wq_b_N,
    int64_t main_wq_b_Np,
    at::Tensor indexer_wq_b_packed,
    int64_t indexer_wq_b_K,
    int64_t indexer_wq_b_N,
    int64_t indexer_wq_b_Np,
    at::Tensor main_cos_sin_cache,
    at::Tensor indexer_cos_sin_cache,
    at::Tensor swa_kv_cache,
    at::Tensor swa_slot_mapping,
    at::Tensor mla_ape,
    at::Tensor mla_state_cache,
    at::Tensor mla_state_slot_mapping,
    at::Tensor mla_token_to_req_indices,
    at::Tensor mla_block_table,
    at::Tensor mla_kv_cache,
    at::Tensor mla_kv_slot_mapping,
    at::Tensor mla_norm_weight,
    at::Tensor indexer_ape,
    at::Tensor indexer_state_cache,
    at::Tensor indexer_state_slot_mapping,
    at::Tensor indexer_token_to_req_indices,
    at::Tensor indexer_block_table,
    at::Tensor indexer_kv_cache,
    at::Tensor indexer_kv_slot_mapping,
    at::Tensor indexer_norm_weight,
    at::Tensor topk_indices_buffer,
    at::Tensor prefill_cu_seq_lens,
    at::Tensor prefill_cu_seqlen_ks,
    at::Tensor prefill_cu_seqlen_ke,
    at::Tensor prefill_block_table,
    int64_t main_head_dim,
    double q_eps,
    int64_t mla_compress_ratio,
    double mla_rms_norm_eps,
    int64_t indexer_compress_ratio,
    double indexer_rms_norm_eps,
    int64_t topk_tokens) {
#if FUSED_CPP_ENABLE_PROFILING
  const bool profile_enabled = PostGemmProfileEnabled();
  PostGemmStageProfile profile;
  PostGemmStageProfile* profile_ptr = profile_enabled ? &profile : nullptr;
#else
  PostGemmStageProfile* profile_ptr = nullptr;
#endif
  FUSED_CPP_PROFILE_START(total_start);
  FUSED_CPP_PROFILE_START(phase_start);

  CheckCpuTensor(qr, "qr");
  CheckCpuTensor(kv, "kv");
  CheckDim(qr, "qr", 2);
  CheckDim(kv, "kv", 2);
  TORCH_CHECK(main_head_dim > 0, "main_head_dim must be positive");
  TORCH_CHECK(main_wq_b_N % main_head_dim == 0,
              "main_wq_b_N must be divisible by main_head_dim");
  TORCH_CHECK(indexer_wq_b_N % indexer_norm_weight.size(0) == 0,
              "indexer_wq_b_N must be divisible by indexer head_dim");
  FUSED_CPP_PROFILE_ADD_IF_PTR(
      profile_ptr == nullptr ? nullptr : &profile_ptr->input_check_ms,
      phase_start);

  const int64_t main_num_heads = main_wq_b_N / main_head_dim;
  FUSED_CPP_PROFILE_RESTART(phase_start);
  at::Tensor q = LinearPrepackedToDtype(
                     qr,
                     main_wq_b_packed,
                     main_wq_b_K,
                     main_wq_b_N,
                     main_wq_b_Np,
                     qr.scalar_type())
                     .reshape({qr.size(0), main_num_heads, main_head_dim});
  FUSED_CPP_PROFILE_ADD_IF_PTR(
      profile_ptr == nullptr ? nullptr : &profile_ptr->main_q_gemm_ms,
      phase_start);

  FUSED_CPP_PROFILE_RESTART(phase_start);
  CheckMainQKvShape(q, kv);
  QNormRopeFused(q, positions, main_cos_sin_cache, q_eps);
  KvRopeCacheInsertFused(kv, swa_kv_cache, swa_slot_mapping, positions, main_cos_sin_cache);
  FUSED_CPP_PROFILE_ADD_IF_PTR(
      profile_ptr == nullptr ? nullptr : &profile_ptr->main_q_norm_rope_swa_insert_ms,
      phase_start);

  const int64_t indexer_head_dim = indexer_norm_weight.size(0);
  FUSED_CPP_PROFILE_RESTART(phase_start);
  at::Tensor indexer_q_linear = LinearPrepackedToDtype(
      qr,
      indexer_wq_b_packed,
      indexer_wq_b_K,
      indexer_wq_b_N,
      indexer_wq_b_Np,
      qr.scalar_type());
  TORCH_CHECK(indexer_q_linear.size(1) % indexer_head_dim == 0,
              "indexer q linear out features must be divisible by indexer head_dim");
  FUSED_CPP_PROFILE_ADD_IF_PTR(
      profile_ptr == nullptr ? nullptr : &profile_ptr->indexer_q_gemm_ms,
      phase_start);

  const int64_t indexer_num_heads = indexer_q_linear.size(1) / indexer_head_dim;
  at::Tensor indexer_q =
      indexer_q_linear.reshape({indexer_q_linear.size(0), indexer_num_heads, indexer_head_dim});

  FUSED_CPP_PROFILE_RESTART(phase_start);
  auto indexer_q_and_weights = IndexerQRopeQuant(positions, indexer_q, indexer_cos_sin_cache, indexer_weights);
  at::Tensor q_quant = std::get<0>(indexer_q_and_weights);
  at::Tensor scaled_weights = std::get<1>(indexer_q_and_weights);
  FUSED_CPP_PROFILE_ADD_IF_PTR(
      profile_ptr == nullptr ? nullptr : &profile_ptr->indexer_q_rope_weights_ms,
      phase_start);

  RunCompressor(kv_score, positions, mla_ape, mla_state_cache, mla_state_slot_mapping,
                mla_token_to_req_indices, mla_block_table, mla_kv_cache, mla_kv_slot_mapping,
                mla_norm_weight, main_cos_sin_cache, mla_compress_ratio, mla_rms_norm_eps,
                profile_ptr == nullptr ? nullptr : &profile_ptr->mla_save_partial_states_ms,
                profile_ptr == nullptr ? nullptr : &profile_ptr->mla_compress_norm_rope_insert_ms);
  RunCompressor(indexer_kv_score, positions, indexer_ape, indexer_state_cache,
                indexer_state_slot_mapping, indexer_token_to_req_indices, indexer_block_table,
                indexer_kv_cache, indexer_kv_slot_mapping, indexer_norm_weight,
                indexer_cos_sin_cache, indexer_compress_ratio, indexer_rms_norm_eps,
                profile_ptr == nullptr ? nullptr : &profile_ptr->indexer_save_partial_states_ms,
                profile_ptr == nullptr ? nullptr : &profile_ptr->indexer_compress_norm_rope_insert_ms);

  // Keep the migrated native post-GEMM path self-contained.  Do not dispatch
  // through the retired standalone sparse_attn_indexer_prefill_cpp_v0 symbol.
  SparseAttnIndexerPrefill(q_quant, scaled_weights, indexer_kv_cache, topk_indices_buffer,
                           topk_tokens, prefill_cu_seq_lens, prefill_cu_seqlen_ks,
                           prefill_cu_seqlen_ke, prefill_block_table, profile_ptr);
  FUSED_CPP_PROFILE_IF_ENABLED(
      profile_ptr != nullptr,
      PrintPostGemmProfile(
          *profile_ptr,
          ::fused_cpp::profile::elapsed_ms(total_start)));
  return std::make_tuple(q, topk_indices_buffer);
}
