#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "sdpa_common.h"
#include "sdpa_flash2_neon_l3kv_impl.h"
#include "sdpa_microkernels/impls/mk_baseline.h"
#include "sdpa_microkernels/neon_cache_microkernels.h"
#include "sdpa_pack_utils.h"
#include "sdpa_tile_sizes.h"

namespace {

constexpr int64_t kQueryBlock = 8;
constexpr int64_t kIndexedKt = 4;
constexpr int64_t kDenseThreshold = 16;

struct DenseSeg {
  int64_t start;
  int64_t length;
};

struct IndexedTile {
  std::array<std::array<int64_t, kIndexedKt>, kQueryBlock> idx{};
  uint64_t valid_mask = 0;
};

struct BlockPlan {
  int64_t token0 = 0;
  int64_t lq_eff = 0;
  std::vector<DenseSeg> dense_segments;
  std::vector<IndexedTile> indexed_tiles;
};

struct Run {
  int64_t pos;
  int64_t start;
  int64_t length;
};

static inline std::vector<Run> contiguous_runs(const std::vector<int64_t>& values) {
  std::vector<Run> runs;
  if (values.empty()) {
    return runs;
  }
  int64_t run_pos = 0;
  int64_t run_start = values[0];
  int64_t run_len = 1;
  for (int64_t pos = 1; pos < static_cast<int64_t>(values.size()); ++pos) {
    if (values[pos] == values[pos - 1] + 1) {
      ++run_len;
      continue;
    }
    runs.push_back({run_pos, run_start, run_len});
    run_pos = pos;
    run_start = values[pos];
    run_len = 1;
  }
  runs.push_back({run_pos, run_start, run_len});
  return runs;
}

static inline BlockPlan build_block_plan(
    int64_t token0,
    const std::vector<std::vector<int64_t>>& rows) {
  BlockPlan plan;
  plan.token0 = token0;
  plan.lq_eff = static_cast<int64_t>(rows.size());

  std::vector<std::vector<uint8_t>> consumed;
  consumed.reserve(rows.size());
  for (const auto& row : rows) {
    consumed.emplace_back(row.size(), uint8_t{0});
  }

  if (plan.lq_eff == kQueryBlock) {
    std::vector<std::vector<Run>> row_runs;
    row_runs.reserve(kQueryBlock);
    for (const auto& row : rows) {
      row_runs.push_back(contiguous_runs(row));
    }

    for (const Run& first_run : row_runs[0]) {
      if (first_run.length < kDenseThreshold) {
        continue;
      }

      std::array<int64_t, kQueryBlock> match_pos{};
      std::array<int64_t, kQueryBlock> match_len{};
      match_pos[0] = first_run.pos;
      match_len[0] = first_run.length;

      bool matched_all = true;
      for (int64_t row_idx = 1; row_idx < kQueryBlock; ++row_idx) {
        bool matched_row = false;
        for (const Run& run : row_runs[row_idx]) {
          if (run.start == first_run.start) {
            match_pos[row_idx] = run.pos;
            match_len[row_idx] = run.length;
            matched_row = true;
            break;
          }
        }
        if (!matched_row) {
          matched_all = false;
          break;
        }
      }
      if (!matched_all) {
        continue;
      }

      int64_t dense_len = match_len[0];
      for (int64_t row_idx = 1; row_idx < kQueryBlock; ++row_idx) {
        dense_len = std::min(dense_len, match_len[row_idx]);
      }
      if (dense_len < kDenseThreshold) {
        continue;
      }
      dense_len = (dense_len / kQueryBlock) * kQueryBlock;
      if (dense_len == 0) {
        continue;
      }

      bool overlaps = false;
      for (int64_t row_idx = 0; row_idx < kQueryBlock && !overlaps; ++row_idx) {
        for (int64_t off = 0; off < dense_len; ++off) {
          if (consumed[row_idx][match_pos[row_idx] + off]) {
            overlaps = true;
            break;
          }
        }
      }
      if (overlaps) {
        continue;
      }

      plan.dense_segments.push_back({first_run.start, dense_len});
      for (int64_t row_idx = 0; row_idx < kQueryBlock; ++row_idx) {
        for (int64_t off = 0; off < dense_len; ++off) {
          consumed[row_idx][match_pos[row_idx] + off] = 1;
        }
      }
    }
  }

  std::vector<std::vector<int64_t>> leftovers;
  leftovers.reserve(rows.size());
  int64_t max_leftover = 0;
  for (int64_t row_idx = 0; row_idx < plan.lq_eff; ++row_idx) {
    std::vector<int64_t> row_leftover;
    const auto& row = rows[row_idx];
    const auto& row_consumed = consumed[row_idx];
    row_leftover.reserve(row.size());
    for (int64_t i = 0; i < static_cast<int64_t>(row.size()); ++i) {
      if (!row_consumed[i]) {
        row_leftover.push_back(row[i]);
      }
    }
    max_leftover = std::max(max_leftover, static_cast<int64_t>(row_leftover.size()));
    leftovers.push_back(std::move(row_leftover));
  }

  for (int64_t start_col = 0; start_col < max_leftover; start_col += kIndexedKt) {
    IndexedTile tile;
    for (int64_t row_idx = 0; row_idx < kQueryBlock; ++row_idx) {
      const std::vector<int64_t>* row_values = nullptr;
      if (row_idx < plan.lq_eff) {
        row_values = &leftovers[row_idx];
      }
      for (int64_t col = 0; col < kIndexedKt; ++col) {
        const int64_t value_pos = start_col + col;
        if (row_values != nullptr &&
            value_pos < static_cast<int64_t>(row_values->size())) {
          tile.idx[row_idx][col] = (*row_values)[value_pos];
          tile.valid_mask |= uint64_t{1} << (row_idx * kIndexedKt + col);
        } else {
          tile.idx[row_idx][col] = 0;
        }
      }
    }
    plan.indexed_tiles.push_back(tile);
  }

  return plan;
}

static inline std::vector<BlockPlan> build_sparse_mla_plans(
    const int64_t* indices,
    int64_t s_q,
    int64_t topk) {
  std::vector<BlockPlan> plans;
  plans.reserve((s_q + kQueryBlock - 1) / kQueryBlock);
  for (int64_t token0 = 0; token0 < s_q; token0 += kQueryBlock) {
    std::vector<std::vector<int64_t>> rows;
    const int64_t token_end = std::min(token0 + kQueryBlock, s_q);
    rows.reserve(token_end - token0);
    for (int64_t token_idx = token0; token_idx < token_end; ++token_idx) {
      std::vector<int64_t> row;
      row.reserve(topk);
      const int64_t* src = indices + token_idx * topk;
      for (int64_t k = 0; k < topk; ++k) {
        if (src[k] >= 0) {
          row.push_back(src[k]);
        }
      }
      rows.push_back(std::move(row));
    }
    plans.push_back(build_block_plan(token0, rows));
  }
  return plans;
}

static inline bool find_full_shared_contiguous_run(
    const int64_t* indices,
    int64_t s_q,
    int64_t topk,
    int64_t s_kv,
    int64_t& run_start) {
  if (s_q <= 0 || topk <= 0) {
    return false;
  }

  run_start = indices[0];
  if (run_start < 0) {
    return false;
  }
  for (int64_t col = 0; col < topk; ++col) {
    if (indices[col] != run_start + col) {
      return false;
    }
  }
  TORCH_CHECK(
      run_start + topk <= s_kv,
      "sparse_mla: dense fast path index range [",
      run_start,
      ", ",
      run_start + topk,
      ") exceeds s_kv=",
      s_kv);

  for (int64_t row = 1; row < s_q; ++row) {
    const int64_t* row_idx = indices + row * topk;
    for (int64_t col = 0; col < topk; ++col) {
      if (row_idx[col] != run_start + col) {
        return false;
      }
    }
  }
  return true;
}

template <typename scalar_t>
static inline void pack_mqa_v_to_evblock8(
    const scalar_t* v_src,
    int64_t v_row_stride,
    scalar_t* v_dst,
    int64_t s,
    int64_t d_v) {
  const int64_t ev_blocks = d_v / 8;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int64_t ev_block = 0; ev_block < ev_blocks; ++ev_block) {
    scalar_t* dst_block = v_dst + ev_block * s * 8;
    const int64_t ev = ev_block * 8;
    for (int64_t row = 0; row < s; ++row) {
      std::memcpy(
          dst_block + row * 8,
          v_src + row * v_row_stride + ev,
          8 * sizeof(scalar_t));
    }
  }
}

static inline bool run_dense_packqkv_mqa_fast_path(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices_2d,
    float scale,
    int64_t d_v,
    const at::Tensor* sink_tensor,
    at::Tensor& output,
    at::Tensor& max_logits,
    at::Tensor& lse) {
  if (q.scalar_type() != at::kBFloat16 || sink_tensor != nullptr) {
    return false;
  }

  const int64_t s_q = q.size(0);
  const int64_t h_q = q.size(1);
  const int64_t d_qk = q.size(2);
  const int64_t s_kv = kv.size(0);
  const int64_t topk = indices_2d.size(1);
  if (topk % 8 != 0 || d_v % 8 != 0) {
    return false;
  }

  int64_t run_start = 0;
  if (!find_full_shared_contiguous_run(
          indices_2d.data_ptr<int64_t>(), s_q, topk, s_kv, run_start)) {
    return false;
  }

  using ::fused_cpp::sdpa_flash2_neon_l3kv_impl::run_path_collapse3_packqkv;
  using ::fused_cpp::sdpa_tile_sizes::compute_tile_sizes_l3kv;

  const auto* q_ptr = q.data_ptr<at::BFloat16>();
  const auto* kv_base = kv.data_ptr<at::BFloat16>() + run_start * d_qk;

  const int64_t e_main = d_qk & ~int64_t{3};
  const int64_t e_blocks = e_main / 4;
  const int64_t kblock_u16 = e_blocks * 32;
  const int64_t s_blocks = topk / 8;

  at::Tensor k_packed = at::empty(
      {s_blocks * kblock_u16},
      q.options());
  auto* k_packed_ptr = reinterpret_cast<uint16_t*>(k_packed.data_ptr());
  ::fused_cpp::sdpa_microkernels::pack_k_to_seq8<at::BFloat16>(
      kv_base,
      k_packed_ptr,
      /*B=*/1,
      /*N=*/1,
      topk,
      d_qk);

  at::Tensor v_packed = at::empty(
      {d_v / 8, topk, 8},
      q.options());
  pack_mqa_v_to_evblock8<at::BFloat16>(
      kv_base,
      d_qk,
      v_packed.data_ptr<at::BFloat16>(),
      topk,
      d_v);

  at::Tensor output_fp32 = at::empty(
      {s_q, h_q, d_v},
      q.options().dtype(at::kFloat));
  at::Tensor max_sdpa = at::empty(
      {1, h_q, s_q},
      q.options().dtype(at::kFloat));
  at::Tensor lse_sdpa = at::empty(
      {1, h_q, s_q},
      q.options().dtype(at::kFloat));

  SdpaParams p{};
  p.B = 1;
  p.N = h_q;
  p.L = s_q;
  p.S = topk;
  p.E = d_qk;
  p.Ev = d_v;
  p.scale_f = scale;
  p.neg_inf = -std::numeric_limits<float>::infinity();
  p.causal_offset = topk - s_q;
  p.is_causal = false;
  p.dtype = SdpaDtype::kBFloat16;
  p.q_ptr = q_ptr;
  p.k_ptr = kv_base;
  p.v_ptr = kv_base;
  p.mask_ptr = nullptr;
  p.out_ptr = output_fp32.data_ptr<float>();
  p.max_logits_ptr = max_sdpa.data_ptr<float>();
  p.lse_ptr = lse_sdpa.data_ptr<float>();

  const auto ts = compute_tile_sizes_l3kv(
      p.B, p.N, p.S, p.L, p.E, p.Ev, sizeof(at::BFloat16));

  const int64_t q_stride_b = s_q * h_q * d_qk;
  const int64_t q_stride_n = d_qk;
  const int64_t q_stride_l = h_q * d_qk;
  const int64_t k_orig_stride_b = topk * d_qk;
  const int64_t k_orig_stride_n = 0;
  const int64_t k_orig_stride_s = d_qk;
  const int64_t k_packed_stride_b = s_blocks * kblock_u16;
  const int64_t k_packed_stride_n = 0;
  const int64_t k_sblock_stride = kblock_u16;
  const int64_t v_packed_stride_b = (d_v / 8) * topk * 8;
  const int64_t v_packed_stride_n = 0;
  const int64_t v_packed_stride_s = 8;
  const int64_t v_evblock_stride = topk * 8;
  const int64_t m_stride_b = 0;
  const int64_t m_stride_n = 0;
  const int64_t m_stride_l = 0;
  const int64_t o_stride_b = s_q * h_q * d_v;
  const int64_t o_stride_n = d_v;
  const int64_t o_stride_l = h_q * d_v;

  run_path_collapse3_packqkv<false, false, false, 5>(
      q_ptr,
      k_packed_ptr,
      kv_base,
      v_packed.data_ptr<at::BFloat16>(),
      p,
      ts,
      q_stride_b,
      q_stride_n,
      q_stride_l,
      k_orig_stride_b,
      k_orig_stride_n,
      k_orig_stride_s,
      k_packed_stride_b,
      k_packed_stride_n,
      k_sblock_stride,
      v_packed_stride_b,
      v_packed_stride_n,
      v_packed_stride_s,
      v_evblock_stride,
      m_stride_b,
      m_stride_n,
      m_stride_l,
      o_stride_b,
      o_stride_n,
      o_stride_l);

  output.copy_(output_fp32.to(q.scalar_type()));
  max_logits.copy_(max_sdpa.squeeze(0).permute({1, 0}));
  lse.copy_(lse_sdpa.squeeze(0).permute({1, 0}));
  return true;
}

template <typename scalar_t>
static inline float scalar_to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <typename scalar_t>
static inline float dot_qk(const scalar_t* q, const scalar_t* k, int64_t dim) {
  float acc = 0.0f;
  for (int64_t e = 0; e < dim; ++e) {
    acc += scalar_to_float(q[e]) * scalar_to_float(k[e]);
  }
  return acc;
}

#if FUSED_CPP_SDPA_CACHE_HAS_NEON

static inline float dot_qk(const float* q, const float* k, int64_t dim) {
  float32x4_t acc = vdupq_n_f32(0.0f);
  int64_t e = 0;
  for (; e + 4 <= dim; e += 4) {
    float32x4_t qv = vld1q_f32(q + e);
    float32x4_t kv = vld1q_f32(k + e);
    acc = vfmaq_f32(acc, qv, kv);
  }
  float sum = vaddvq_f32(acc);
  for (; e < dim; ++e) {
    sum += q[e] * k[e];
  }
  return sum;
}

static inline float dot_qk(
    const at::BFloat16* q,
    const at::BFloat16* k,
    int64_t dim) {
  const uint16_t* qp = reinterpret_cast<const uint16_t*>(q);
  const uint16_t* kp = reinterpret_cast<const uint16_t*>(k);
  float32x4_t acc = vdupq_n_f32(0.0f);
  int64_t e = 0;
  for (; e + 4 <= dim; e += 4) {
    uint16x4_t q_bf = vld1_u16(qp + e);
    uint16x4_t k_bf = vld1_u16(kp + e);
    float32x4_t qv = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(q_bf), 16));
    float32x4_t kv = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(k_bf), 16));
    acc = vfmaq_f32(acc, qv, kv);
  }
  float sum = vaddvq_f32(acc);
  for (; e < dim; ++e) {
    sum += scalar_to_float(q[e]) * scalar_to_float(k[e]);
  }
  return sum;
}

#endif

static inline void scale_row(float* row, float factor, int64_t len) {
  int64_t i = 0;
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  const float32x4_t vf = vdupq_n_f32(factor);
  for (; i + 4 <= len; i += 4) {
    float32x4_t v = vld1q_f32(row + i);
    vst1q_f32(row + i, vmulq_f32(v, vf));
  }
#endif
  for (; i < len; ++i) {
    row[i] *= factor;
  }
}

template <typename scalar_t>
static inline void add_value_scaled(
    float* dst,
    const scalar_t* value,
    float scale,
    int64_t len) {
  for (int64_t i = 0; i < len; ++i) {
    dst[i] += scale * scalar_to_float(value[i]);
  }
}

#if FUSED_CPP_SDPA_CACHE_HAS_NEON

static inline void add_value_scaled(
    float* dst,
    const float* value,
    float scale,
    int64_t len) {
  int64_t i = 0;
  const float32x4_t vs = vdupq_n_f32(scale);
  for (; i + 4 <= len; i += 4) {
    float32x4_t d = vld1q_f32(dst + i);
    float32x4_t v = vld1q_f32(value + i);
    d = vfmaq_f32(d, v, vs);
    vst1q_f32(dst + i, d);
  }
  for (; i < len; ++i) {
    dst[i] += scale * value[i];
  }
}

static inline void add_value_scaled(
    float* dst,
    const at::BFloat16* value,
    float scale,
    int64_t len) {
  const uint16_t* vp = reinterpret_cast<const uint16_t*>(value);
  int64_t i = 0;
  const float32x4_t vs = vdupq_n_f32(scale);
  for (; i + 4 <= len; i += 4) {
    uint16x4_t v_bf = vld1_u16(vp + i);
    float32x4_t v = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(v_bf), 16));
    float32x4_t d = vld1q_f32(dst + i);
    d = vfmaq_f32(d, v, vs);
    vst1q_f32(dst + i, d);
  }
  for (; i < len; ++i) {
    dst[i] += scale * scalar_to_float(value[i]);
  }
}

#endif

static inline float row_max(const float* scores, int64_t len) {
  float m = -std::numeric_limits<float>::infinity();
  for (int64_t i = 0; i < len; ++i) {
    m = std::max(m, scores[i]);
  }
  return m;
}

static inline float exp_sum_to(
    float* dst,
    const float* scores,
    float max_value,
    int64_t len) {
  float sum = 0.0f;
  for (int64_t i = 0; i < len; ++i) {
    const float p = std::exp(scores[i] - max_value);
    dst[i] = p;
    sum += p;
  }
  return sum;
}

template <typename scalar_t>
static inline void update_dense_segment_sdpa_style(
    const scalar_t* q_block,
    int64_t q_row_stride,
    const scalar_t* kv_seg,
    int64_t kv_row_stride,
    int64_t seg_len,
    int64_t d_qk,
    int64_t d_v,
    float scale,
    float* running_max,
    float* running_sum,
    float* output_acc,
    float* real_max,
    float* real_sum,
    std::vector<float>& scores,
    std::vector<float>& p_hat) {
  using ::fused_cpp::sdpa_microkernels::gemm_pv_8x8;
  using ::fused_cpp::sdpa_microkernels::gemm_pv_tail;
  using ::fused_cpp::sdpa_microkernels::gemm_qkt_8x4;
  using ::fused_cpp::sdpa_microkernels::gemm_qkt_8x8;
  using ::fused_cpp::sdpa_microkernels::gemm_qkt_tail;

  scores.assign(static_cast<size_t>(kQueryBlock * seg_len), 0.0f);
  p_hat.assign(static_cast<size_t>(kQueryBlock * seg_len), 0.0f);

  int64_t s_off = 0;
  alignas(64) float tmp_qkt[kQueryBlock * kQueryBlock];
  for (; s_off + kQueryBlock <= seg_len; s_off += kQueryBlock) {
    const scalar_t* k_tile = kv_seg + s_off * kv_row_stride;
    gemm_qkt_8x8(q_block, q_row_stride, k_tile, kv_row_stride,
                 d_qk, scale, tmp_qkt);
    for (int64_t row = 0; row < kQueryBlock; ++row) {
      std::copy_n(
          tmp_qkt + row * kQueryBlock,
          kQueryBlock,
          scores.data() + row * seg_len + s_off);
    }
  }
  if (s_off + kIndexedKt <= seg_len) {
    const scalar_t* k_tile = kv_seg + s_off * kv_row_stride;
    gemm_qkt_8x4(q_block, q_row_stride, k_tile, kv_row_stride,
                 d_qk, scale, scores.data() + s_off, seg_len);
    s_off += kIndexedKt;
  }
  if (s_off < seg_len) {
    const scalar_t* k_tile = kv_seg + s_off * kv_row_stride;
    gemm_qkt_tail(q_block, q_row_stride, k_tile, kv_row_stride,
                  d_qk, scale, scores.data() + s_off, seg_len,
                  kQueryBlock, static_cast<int>(seg_len - s_off));
  }

  for (int64_t row = 0; row < kQueryBlock; ++row) {
    const float* scores_row = scores.data() + row * seg_len;
    float* p_row = p_hat.data() + row * seg_len;
    const float tile_max = row_max(scores_row, seg_len);

    const float real_new_max = std::max(real_max[row], tile_max);
    float real_tile_sum = 0.0f;
    for (int64_t col = 0; col < seg_len; ++col) {
      real_tile_sum += std::exp(scores_row[col] - real_new_max);
    }
    real_sum[row] = real_sum[row] * std::exp(real_max[row] - real_new_max) +
                    real_tile_sum;
    real_max[row] = real_new_max;

    const float new_max = std::max(running_max[row], tile_max);
    const float correction = std::exp(running_max[row] - new_max);
    running_sum[row] *= correction;
    scale_row(output_acc + row * d_v, correction, d_v);
    running_sum[row] += exp_sum_to(p_row, scores_row, new_max, seg_len);
    running_max[row] = new_max;
  }

  for (int64_t ev_off = 0; ev_off < d_v; ev_off += 8) {
    const int64_t ev_cur = std::min<int64_t>(8, d_v - ev_off);
    const scalar_t* v_ptr = kv_seg + ev_off;
    float* o_ptr = output_acc + ev_off;

    int64_t k_off = 0;
    for (; k_off + kQueryBlock <= seg_len; k_off += kQueryBlock) {
      const scalar_t* v_tile = v_ptr + k_off * kv_row_stride;
      if (ev_cur == 8) {
        gemm_pv_8x8(
            p_hat.data() + k_off,
            seg_len,
            v_tile,
            kv_row_stride,
            kQueryBlock,
            o_ptr,
            d_v);
      } else {
        gemm_pv_tail(
            p_hat.data() + k_off,
            seg_len,
            v_tile,
            kv_row_stride,
            kQueryBlock,
            o_ptr,
            d_v,
            kQueryBlock,
            static_cast<int>(ev_cur));
      }
    }
    if (k_off < seg_len) {
      const scalar_t* v_tile = v_ptr + k_off * kv_row_stride;
      gemm_pv_tail(
          p_hat.data() + k_off,
          seg_len,
          v_tile,
          kv_row_stride,
          seg_len - k_off,
          o_ptr,
          d_v,
          kQueryBlock,
          static_cast<int>(ev_cur));
    }
  }
}

template <typename scalar_t>
static inline void update_indexed_row(
    const float* scores,
    const int64_t* idx,
    int64_t count,
    const scalar_t* kv,
    int64_t kv_row_stride,
    int64_t d_v,
    int64_t row,
    float* running_max,
    float* running_sum,
    float* output_acc,
    float* real_max,
    float* real_sum) {
  const float tile_max = row_max(scores, count);

  const float real_new_max = std::max(real_max[row], tile_max);
  float real_tile_sum = 0.0f;
  for (int64_t col = 0; col < count; ++col) {
    real_tile_sum += std::exp(scores[col] - real_new_max);
  }
  real_sum[row] =
      real_sum[row] * std::exp(real_max[row] - real_new_max) + real_tile_sum;
  real_max[row] = real_new_max;

  const float new_max = std::max(running_max[row], tile_max);
  const float correction = std::exp(running_max[row] - new_max);
  running_sum[row] *= correction;
  float* out_row = output_acc + row * d_v;
  scale_row(out_row, correction, d_v);

  for (int64_t col = 0; col < count; ++col) {
    const float p = std::exp(scores[col] - new_max);
    running_sum[row] += p;
    add_value_scaled(out_row, kv + idx[col] * kv_row_stride, p, d_v);
  }
  running_max[row] = new_max;
}

static inline void apply_sink_row(
    float sink_score,
    int64_t row,
    int64_t d_v,
    float* running_max,
    float* running_sum,
    float* output_acc) {
  if (std::isinf(sink_score) && sink_score < 0.0f) {
    return;
  }
  if (std::isinf(sink_score) && sink_score > 0.0f) {
    running_max[row] = sink_score;
    running_sum[row] = 1.0f;
    std::fill(output_acc + row * d_v, output_acc + (row + 1) * d_v, 0.0f);
    return;
  }
  const float new_max = std::max(running_max[row], sink_score);
  const float correction = std::exp(running_max[row] - new_max);
  const float sink_scale = std::exp(sink_score - new_max);
  scale_row(output_acc + row * d_v, correction, d_v);
  running_sum[row] = running_sum[row] * correction + sink_scale;
  running_max[row] = new_max;
}

template <typename scalar_t>
static inline void store_output_row(
    scalar_t* dst,
    const float* src,
    float inv_sum,
    int64_t d_v) {
  for (int64_t i = 0; i < d_v; ++i) {
    dst[i] = static_cast<scalar_t>(src[i] * inv_sum);
  }
}

template <typename scalar_t>
static inline void run_sparse_mla_kernel(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices_2d,
    const at::Tensor* sink_tensor,
    float scale,
    int64_t d_v,
    at::Tensor& output,
    at::Tensor& max_logits,
    at::Tensor& lse) {
  const int64_t s_q = q.size(0);
  const int64_t h_q = q.size(1);
  const int64_t d_qk = q.size(2);
  const int64_t s_kv = kv.size(0);
  const int64_t topk = indices_2d.size(1);

  const scalar_t* q_ptr = q.data_ptr<scalar_t>();
  const scalar_t* kv_ptr = kv.data_ptr<scalar_t>();
  scalar_t* out_ptr = output.data_ptr<scalar_t>();
  float* max_ptr = max_logits.data_ptr<float>();
  float* lse_ptr = lse.data_ptr<float>();
  const int64_t* idx_ptr = indices_2d.data_ptr<int64_t>();
  const float* sink_ptr =
      sink_tensor == nullptr ? nullptr : sink_tensor->data_ptr<float>();

  const std::vector<BlockPlan> plans = build_sparse_mla_plans(idx_ptr, s_q, topk);

  std::vector<float> output_acc(kQueryBlock * d_v);
  std::vector<float> dense_scores;
  std::vector<float> dense_p_hat;

  for (const BlockPlan& plan : plans) {
    for (int64_t head = 0; head < h_q; ++head) {
      std::array<float, kQueryBlock> running_max;
      std::array<float, kQueryBlock> running_sum;
      std::array<float, kQueryBlock> real_max;
      std::array<float, kQueryBlock> real_sum;
      const float neg_inf = -std::numeric_limits<float>::infinity();
      running_max.fill(neg_inf);
      running_sum.fill(0.0f);
      real_max.fill(neg_inf);
      real_sum.fill(0.0f);
      std::fill(output_acc.begin(), output_acc.end(), 0.0f);

      const scalar_t* q_block =
          q_ptr + (plan.token0 * h_q + head) * d_qk;

      for (const DenseSeg& seg : plan.dense_segments) {
        TORCH_CHECK(
            seg.start >= 0 && seg.start + seg.length <= s_kv,
            "sparse_mla: index out of range in dense segment start=",
            seg.start,
            ", length=",
            seg.length,
            ", s_kv=",
            s_kv);
        const scalar_t* k_seg = kv_ptr + seg.start * d_qk;
        update_dense_segment_sdpa_style(
            q_block,
            h_q * d_qk,
            k_seg,
            d_qk,
            seg.length,
            d_qk,
            d_v,
            scale,
            running_max.data(),
            running_sum.data(),
            output_acc.data(),
            real_max.data(),
            real_sum.data(),
            dense_scores,
            dense_p_hat);
      }

      for (const IndexedTile& tile : plan.indexed_tiles) {
        for (int64_t row = 0; row < plan.lq_eff; ++row) {
          int64_t gathered_idx[kIndexedKt];
          float scores[kIndexedKt];
          int64_t count = 0;
          const scalar_t* q_row =
              q_ptr + ((plan.token0 + row) * h_q + head) * d_qk;
          for (int64_t col = 0; col < kIndexedKt; ++col) {
            if ((tile.valid_mask & (uint64_t{1} << (row * kIndexedKt + col))) == 0) {
              continue;
            }
            const int64_t idx = tile.idx[row][col];
            TORCH_CHECK(
                idx >= 0 && idx < s_kv,
                "sparse_mla: index out of range: ",
                idx,
                " for s_kv=",
                s_kv);
            gathered_idx[count] = idx;
            scores[count] = dot_qk(q_row, kv_ptr + idx * d_qk, d_qk) * scale;
            ++count;
          }
          if (count == 0) {
            continue;
          }
          update_indexed_row(
              scores,
              gathered_idx,
              count,
              kv_ptr,
              d_qk,
              d_v,
              row,
              running_max.data(),
              running_sum.data(),
              output_acc.data(),
              real_max.data(),
              real_sum.data());
        }
      }

      if (sink_ptr != nullptr) {
        const float sink_score = sink_ptr[head];
        for (int64_t row = 0; row < plan.lq_eff; ++row) {
          apply_sink_row(
              sink_score,
              row,
              d_v,
              running_max.data(),
              running_sum.data(),
              output_acc.data());
        }
      }

      for (int64_t row = 0; row < plan.lq_eff; ++row) {
        const int64_t token = plan.token0 + row;
        float* max_cell = max_ptr + token * h_q + head;
        float* lse_cell = lse_ptr + token * h_q + head;
        if (real_sum[row] > 0.0f) {
          *max_cell = real_max[row];
          *lse_cell = real_max[row] + std::log(real_sum[row]);
        }

        scalar_t* out_row = out_ptr + (token * h_q + head) * d_v;
        if (running_sum[row] > 0.0f) {
          store_output_row(
              out_row,
              output_acc.data() + row * d_v,
              1.0f / running_sum[row],
              d_v);
        } else {
          std::fill(out_row, out_row + d_v, scalar_t{0});
        }
      }
    }
  }
}

static inline bool is_integer_dtype(at::ScalarType dtype) {
  return dtype == at::kChar || dtype == at::kShort ||
         dtype == at::kInt || dtype == at::kLong;
}

static inline void check_sparse_inputs(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    int64_t d_v,
    const c10::optional<at::Tensor>& attn_sink,
    const c10::optional<at::Tensor>& topk_length,
    const c10::optional<at::Tensor>& out) {
  TORCH_CHECK(q.dim() == 3, "q must be 3-D [s_q, h_q, d_qk], got ", q.sizes());
  TORCH_CHECK(kv.dim() == 3, "kv must be 3-D [s_kv, h_kv, d_qk], got ", kv.sizes());
  TORCH_CHECK(
      indices.dim() == 3,
      "indices must be 3-D [s_q, h_kv, topk], got ",
      indices.sizes());
  TORCH_CHECK(
      is_integer_dtype(indices.scalar_type()),
      "indices must use an integer dtype, got ",
      indices.scalar_type());
  TORCH_CHECK(
      q.scalar_type() == at::kBFloat16 || q.scalar_type() == at::kFloat,
      "sparse_mla C++ supports q dtype bfloat16 or float32, got ",
      q.scalar_type());
  TORCH_CHECK(
      q.scalar_type() == kv.scalar_type(),
      "sparse_mla C++ requires q/kv dtype match, got ",
      q.scalar_type(),
      " vs ",
      kv.scalar_type());
  TORCH_CHECK(q.device().is_cpu(), "sparse_mla C++ currently supports CPU tensors only");
  TORCH_CHECK(kv.device() == q.device(), "q and kv must be on the same device");
  TORCH_CHECK(indices.device() == q.device(), "indices must be on the same device as q");

  const int64_t s_q = q.size(0);
  const int64_t h_q = q.size(1);
  const int64_t d_qk = q.size(2);
  const int64_t s_kv = kv.size(0);
  const int64_t h_kv = kv.size(1);
  const int64_t kv_d = kv.size(2);
  TORCH_CHECK(indices.size(2) != 0, "indices topk dimension must be non-zero");
  TORCH_CHECK(kv_d == d_qk, "q/kv d_qk mismatch: q=", d_qk, ", kv=", kv_d);
  TORCH_CHECK(
      h_kv == 1,
      "sparse_mla C++ currently matches vLLM sparse CPU fallback and requires h_kv == 1");
  TORCH_CHECK(
      indices.size(0) == s_q && indices.size(1) == h_kv,
      "indices shape must be [s_q, h_kv, topk], got ",
      indices.sizes(),
      " for s_q=",
      s_q,
      ", h_kv=",
      h_kv);
  TORCH_CHECK(
      d_v > 0 && d_v <= d_qk,
      "d_v must satisfy 0 < d_v <= d_qk, got d_v=",
      d_v,
      ", d_qk=",
      d_qk);
  TORCH_CHECK(s_kv > 0, "kv must contain at least one row");

  if (attn_sink.has_value()) {
    const at::Tensor& sink = attn_sink.value();
    TORCH_CHECK(
        sink.dim() == 1 && sink.numel() >= h_q,
        "attn_sink must be 1-D with at least h_q entries, got ",
        sink.sizes(),
        " for h_q=",
        h_q);
    TORCH_CHECK(sink.device() == q.device(), "attn_sink must be on the same device as q");
  }
  if (topk_length.has_value()) {
    const at::Tensor& len = topk_length.value();
    TORCH_CHECK(
        is_integer_dtype(len.scalar_type()),
        "topk_length must use an integer dtype, got ",
        len.scalar_type());
    TORCH_CHECK(
        len.dim() == 1 && len.numel() == s_q,
        "topk_length must be [s_q], got ",
        len.sizes(),
        " for s_q=",
        s_q);
    TORCH_CHECK(len.device() == q.device(), "topk_length must be on the same device as q");
  }
  if (out.has_value()) {
    const at::Tensor& out_t = out.value();
    TORCH_CHECK(
        out_t.dim() == 3 &&
            out_t.size(0) == s_q &&
            out_t.size(1) == h_q &&
            out_t.size(2) == d_v,
        "out must have shape [",
        s_q,
        ", ",
        h_q,
        ", ",
        d_v,
        "], got ",
        out_t.sizes());
    TORCH_CHECK(out_t.device() == q.device(), "out must be on the same device as q");
    TORCH_CHECK(out_t.scalar_type() == q.scalar_type(), "out dtype must match q dtype");
    TORCH_CHECK(out_t.stride(2) == 1, "out must be contiguous on the last dimension");
  }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> flash_mla_sparse_fwd(
    at::Tensor q,
    at::Tensor kv,
    at::Tensor indices,
    double sm_scale,
    c10::optional<int64_t> d_v_opt,
    c10::optional<at::Tensor> attn_sink,
    c10::optional<at::Tensor> topk_length,
    c10::optional<at::Tensor> out) {
  const int64_t d_v = d_v_opt.value_or(kv.size(2));
  check_sparse_inputs(q, kv, indices, d_v, attn_sink, topk_length, out);

  at::Tensor q_c = q.contiguous();
  at::Tensor kv_c = kv.contiguous();
  at::Tensor indices_2d = indices.reshape({indices.size(0), -1})
                              .to(at::kLong)
                              .contiguous();

  at::Tensor sink_c;
  at::Tensor* sink_ptr = nullptr;
  if (attn_sink.has_value()) {
    sink_c = attn_sink.value().slice(0, 0, q.size(1)).to(at::kFloat).contiguous();
    sink_ptr = &sink_c;
  }

  at::Tensor output_tmp = at::empty({q.size(0), q.size(1), d_v}, q.options());
  at::Tensor max_logits = at::full(
      {q.size(0), q.size(1)},
      -std::numeric_limits<float>::infinity(),
      q.options().dtype(at::kFloat));
  at::Tensor lse = at::full(
      {q.size(0), q.size(1)},
      std::numeric_limits<float>::infinity(),
      q.options().dtype(at::kFloat));

  if (run_dense_packqkv_mqa_fast_path(
          q_c,
          kv_c,
          indices_2d,
          static_cast<float>(sm_scale),
          d_v,
          sink_ptr,
          output_tmp,
          max_logits,
          lse)) {
    if (out.has_value()) {
      out.value().copy_(output_tmp);
      return std::make_tuple(out.value(), max_logits, lse);
    }
    return std::make_tuple(output_tmp, max_logits, lse);
  }

  if (q_c.scalar_type() == at::kBFloat16) {
    run_sparse_mla_kernel<at::BFloat16>(
        q_c,
        kv_c,
        indices_2d,
        sink_ptr,
        static_cast<float>(sm_scale),
        d_v,
        output_tmp,
        max_logits,
        lse);
  } else {
    run_sparse_mla_kernel<float>(
        q_c,
        kv_c,
        indices_2d,
        sink_ptr,
        static_cast<float>(sm_scale),
        d_v,
        output_tmp,
        max_logits,
        lse);
  }

  if (out.has_value()) {
    out.value().copy_(output_tmp);
    return std::make_tuple(out.value(), max_logits, lse);
  }
  return std::make_tuple(output_tmp, max_logits, lse);
}
