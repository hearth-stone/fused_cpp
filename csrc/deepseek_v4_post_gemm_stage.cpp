#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <type_traits>
#include <tuple>
#include <vector>

#include "deepseek_v4_attn_gemm_sve.h"
#include "deepseek_v4_indexer_sve.h"
#include "deepseek_v4_q_norm_rope_sve.h"
#include "profile_utils.h"
#include "workspace_pool.h"

#ifdef __aarch64__
#include "gemm_params.h"
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

#ifndef FUSED_CPP_STRICT_MODE
#define FUSED_CPP_STRICT_MODE 0
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __aarch64__
extern "C" {
void bf16gemm_k_ld(const uint16_t* A, const uint16_t* B_reo, float* C, uint16_t* A_reorder,
                   const gemm_params_t* params);
void bf16gemm_k_ld1(const uint16_t* A, const uint16_t* B_reo, float* C, uint16_t* A_reorder,
                    const gemm_params_t* params);
void bf16gemm_k_ld2(const uint16_t* A, const uint16_t* B_reo, float* C, uint16_t* A_reorder,
                    const gemm_params_t* params);
void bf16gemm_k_ld4(const uint16_t* A, const uint16_t* B_reo, float* C, uint16_t* A_reorder,
                    const gemm_params_t* params);
void deepseek_v4_attn_gemm_packed_f32(const uint16_t* A, const uint16_t* B_reo, float* C, uint16_t* A_reorder,
                                      const gemm_params_t* params);
void deepseek_v4_attn_gemm_packed_bf16(const uint16_t* A, const uint16_t* B_reo, uint16_t* C, uint16_t* A_reorder,
                                       const gemm_params_t* params);
#ifdef __linux__
void bf16gemm_k_nld_b(const uint16_t* A, const uint16_t* B_reo, uint16_t* C, uint16_t* A_reorder,
                      const gemm_params_t* params);
void bf16gemm_k_nld1_b(const uint16_t* A, const uint16_t* B_reo, uint16_t* C, uint16_t* A_reorder,
                       const gemm_params_t* params);
void bf16gemm_k_nld2_b(const uint16_t* A, const uint16_t* B_reo, uint16_t* C, uint16_t* A_reorder,
                       const gemm_params_t* params);
void bf16gemm_k_nld4_b(const uint16_t* A, const uint16_t* B_reo, uint16_t* C, uint16_t* A_reorder,
                       const gemm_params_t* params);
#endif
}
#endif

at::Tensor bf16_linear_to_dtype(at::Tensor input, at::Tensor weight, bool output_bf16, int64_t nthreads);
at::Tensor bf16_linear_prepacked_to_dtype(at::Tensor input, at::Tensor packed_weight, int64_t K, int64_t N, int64_t Np,
                                          bool output_bf16, int64_t nthreads);
void bf16_linear_prepacked_to_dtype_out(at::Tensor input, at::Tensor packed_weight, int64_t K, int64_t N, int64_t Np,
                                        bool output_bf16, int64_t nthreads, at::Tensor output);
std::tuple<at::Tensor, int64_t, int64_t> deepseek_v4_gemm_prepare_weight_for_backend(at::Tensor weight, bool use_sve);

namespace {

using ::fused_cpp::profile::TimePoint;
using torch::indexing::Slice;

struct PostGemmStageProfile {
  double input_check_ms = 0.0;
  double shared_q_gemm_ms = 0.0;
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
  double sparse_indexer_score_ms = 0.0;
  double sparse_indexer_topk_ms = 0.0;
  int sparse_indexer_sve_n_tile = -1;
  int sparse_indexer_topk_native = -1;
};

#if FUSED_CPP_ENABLE_PROFILING

bool PostGemmProfileEnabled() { return ::fused_cpp::profile::env_enabled("FUSED_CPP_DEEPSEEK_V4_POST_GEMM_PROFILE"); }

void PrintPostGemmProfile(const PostGemmStageProfile& profile, double total_ms) {
  const double known_ms = profile.input_check_ms + profile.shared_q_gemm_ms + profile.main_q_gemm_ms +
                          profile.main_q_norm_rope_swa_insert_ms + profile.indexer_q_gemm_ms +
                          profile.indexer_q_rope_weights_ms + profile.mla_save_partial_states_ms +
                          profile.mla_compress_norm_rope_insert_ms + profile.indexer_save_partial_states_ms +
                          profile.indexer_compress_norm_rope_insert_ms + profile.sparse_indexer_short_path_ms +
                          profile.sparse_indexer_gather_ms + profile.sparse_indexer_fold_q_ms +
                          profile.sparse_indexer_score_ms + profile.sparse_indexer_topk_ms;
  const double other_ms = std::max(0.0, total_ms - known_ms);
  const auto pct = [total_ms](double ms) -> double { return total_ms > 0.0 ? (100.0 * ms / total_ms) : 0.0; };

  std::cerr << std::fixed << std::setprecision(3) << "deepseek_v4_post_gemm_stage_profile"
            << " total_ms=" << total_ms << " input_check_ms=" << profile.input_check_ms << "("
            << pct(profile.input_check_ms) << "%)"
            << " shared_q_gemm_ms=" << profile.shared_q_gemm_ms << "(" << pct(profile.shared_q_gemm_ms) << "%)"
            << " main_q_gemm_ms=" << profile.main_q_gemm_ms << "(" << pct(profile.main_q_gemm_ms) << "%)"
            << " main_q_norm_rope_swa_insert_ms=" << profile.main_q_norm_rope_swa_insert_ms << "("
            << pct(profile.main_q_norm_rope_swa_insert_ms) << "%)"
            << " indexer_q_gemm_ms=" << profile.indexer_q_gemm_ms << "(" << pct(profile.indexer_q_gemm_ms) << "%)"
            << " indexer_q_rope_weights_ms=" << profile.indexer_q_rope_weights_ms << "("
            << pct(profile.indexer_q_rope_weights_ms) << "%)"
            << " mla_save_partial_states_ms=" << profile.mla_save_partial_states_ms << "("
            << pct(profile.mla_save_partial_states_ms) << "%)"
            << " mla_compress_norm_rope_insert_ms=" << profile.mla_compress_norm_rope_insert_ms << "("
            << pct(profile.mla_compress_norm_rope_insert_ms) << "%)"
            << " indexer_save_partial_states_ms=" << profile.indexer_save_partial_states_ms << "("
            << pct(profile.indexer_save_partial_states_ms) << "%)"
            << " indexer_compress_norm_rope_insert_ms=" << profile.indexer_compress_norm_rope_insert_ms << "("
            << pct(profile.indexer_compress_norm_rope_insert_ms) << "%)"
            << " sparse_indexer_short_path_ms=" << profile.sparse_indexer_short_path_ms << "("
            << pct(profile.sparse_indexer_short_path_ms) << "%)"
            << " sparse_indexer_gather_ms=" << profile.sparse_indexer_gather_ms << "("
            << pct(profile.sparse_indexer_gather_ms) << "%)"
            << " sparse_indexer_fold_q_ms=" << profile.sparse_indexer_fold_q_ms << "("
            << pct(profile.sparse_indexer_fold_q_ms) << "%)"
            << " sparse_indexer_score_topk_ms=" << profile.sparse_indexer_score_ms + profile.sparse_indexer_topk_ms
            << "(" << pct(profile.sparse_indexer_score_ms + profile.sparse_indexer_topk_ms) << "%)"
            << " sparse_indexer_score_ms=" << profile.sparse_indexer_score_ms << "("
            << pct(profile.sparse_indexer_score_ms) << "%)"
            << " sparse_indexer_topk_ms=" << profile.sparse_indexer_topk_ms << "("
            << pct(profile.sparse_indexer_topk_ms) << "%)"
            << " sparse_indexer_topk_backend="
            << (profile.sparse_indexer_topk_native > 0
                    ? "native_batch_exact"
                    : (profile.sparse_indexer_topk_native == 0 ? "aten_rowwise" : "short_path"))
            << " sparse_indexer_backend="
            << (profile.sparse_indexer_sve_n_tile > 0
                    ? "sve_8x2vl"
                    : (profile.sparse_indexer_sve_n_tile == 0 ? "fallback" : "short_path"))
            << " sparse_indexer_n_tile=" << profile.sparse_indexer_sve_n_tile << " other_ms=" << other_ms << "("
            << pct(other_ms) << "%)" << '\n';
}

#endif  // FUSED_CPP_ENABLE_PROFILING

bool DeepSeekV4KvRopeWriteKvEnabled() {
  const char* value = std::getenv("FUSED_CPP_DEEPSEEK_V4_KV_ROPE_WRITE_KV");
  if (value == nullptr) {
    return false;
  }
  return !(value[0] == '\0' || std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 ||
           std::strcmp(value, "FALSE") == 0);
}

enum class PostGemmBackend {
  kNeon,
  kSve,
};

constexpr int64_t kPostGemmNeonMPanelRows = 8;
constexpr int64_t kPostGemmTargetBWindowBytes = 1 << 20;
constexpr int kPostGemmSharedQPoolMinThreads = 32;
constexpr int kPostGemmDefaultNGroups = 2;

PostGemmBackend SelectedPostGemmBackend();

int64_t PostGemmMPanelRows(PostGemmBackend backend) {
  return backend == PostGemmBackend::kSve ? ::fused_cpp::deepseek_v4::attn_sve::m_panel_rows()
                                          : kPostGemmNeonMPanelRows;
}

int64_t ChoosePostGemmWindowGroups(int64_t K, int64_t Np, int64_t n_tile, int64_t num_threads) {
  const int64_t n_tiles = Np / n_tile;
  const int64_t bytes_per_tile = K * n_tile * static_cast<int64_t>(sizeof(uint16_t));
  const int64_t min_groups = std::min<int64_t>(
      n_tiles,
      std::max<int64_t>(1, (Np * K * static_cast<int64_t>(sizeof(uint16_t)) + kPostGemmTargetBWindowBytes - 1) /
                               kPostGemmTargetBWindowBytes));
  if (num_threads < min_groups) {
    return min_groups;
  }
  for (int64_t groups = min_groups; groups <= std::min(n_tiles, num_threads); ++groups) {
    if (num_threads % groups == 0 &&
        bytes_per_tile * ((n_tiles + groups - 1) / groups) <= kPostGemmTargetBWindowBytes) {
      return groups;
    }
  }
  return min_groups;
}

void PackPostGemmANeonRange(const uint16_t* input, uint16_t* packed, int64_t M, int64_t K, int64_t panel_begin,
                            int64_t panel_end) {
  const int64_t k_blocks = K / 4;
  for (int64_t panel = panel_begin; panel < panel_end; ++panel) {
    uint16_t* packed_panel = packed + panel * kPostGemmNeonMPanelRows * K;
    for (int64_t k_block = 0; k_block < k_blocks; ++k_block) {
      uint16_t* dst = packed_panel + k_block * kPostGemmNeonMPanelRows * 4;
      for (int64_t row = 0; row < kPostGemmNeonMPanelRows; ++row) {
        const int64_t source_row = panel * kPostGemmNeonMPanelRows + row;
        uint16_t* dst_row = dst + row * 4;
        if (source_row < M) {
          std::memcpy(dst_row, input + source_row * K + k_block * 4, 4 * sizeof(uint16_t));
        } else {
          std::memset(dst_row, 0, 4 * sizeof(uint16_t));
        }
      }
    }
  }
}

bool EnvFalseLocal(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  return value[0] == '\0' || value[0] == '0' || std::strcmp(value, "false") == 0 || std::strcmp(value, "False") == 0 ||
         std::strcmp(value, "off") == 0 || std::strcmp(value, "OFF") == 0;
}

bool PostGemmM8AlignedEnabled() {
  const char* value = std::getenv("FUSED_CPP_POST_GEMM_M8_ALIGNED");
  return value == nullptr || !EnvFalseLocal("FUSED_CPP_POST_GEMM_M8_ALIGNED");
}

bool PostGemmSharedQPoolEnabled(int64_t M) {
#if defined(__aarch64__) && defined(_OPENMP)
  const char* value = std::getenv("FUSED_CPP_POST_GEMM_SHARED_Q_POOL");
  if (!PostGemmM8AlignedEnabled() || (value != nullptr && EnvFalseLocal("FUSED_CPP_POST_GEMM_SHARED_Q_POOL"))) {
    return false;
  }
  if (value != nullptr) {
    return true;
  }
  const int num_threads = omp_get_max_threads();
  const PostGemmBackend backend = SelectedPostGemmBackend();
  const int64_t panel_rows = PostGemmMPanelRows(backend);
  const int64_t m_panels = (M + panel_rows - 1) / panel_rows;
  if (backend == PostGemmBackend::kSve) {
    return m_panels > 0;
  }
  return num_threads >= kPostGemmSharedQPoolMinThreads && m_panels >= num_threads;
#else
  (void)M;
  return false;
#endif
}

int RequestedPostGemmNGroups(PostGemmBackend backend, int num_threads) {
  const char* value = std::getenv("FUSED_CPP_POST_GEMM_N_GROUPS");
  if (value == nullptr || value[0] == '\0') {
    if (backend == PostGemmBackend::kSve) {
      return num_threads <= 48 ? std::max(kPostGemmDefaultNGroups, num_threads)
                               : std::max(kPostGemmDefaultNGroups, num_threads / 2);
    }
    return kPostGemmDefaultNGroups;
  }
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  TORCH_CHECK(
      end != value && *end == '\0' && parsed >= kPostGemmDefaultNGroups && parsed <= std::numeric_limits<int>::max(),
      "FUSED_CPP_POST_GEMM_N_GROUPS must be an integer >= ", kPostGemmDefaultNGroups, ", got ", value);
  return static_cast<int>(parsed);
}

PostGemmBackend SelectedPostGemmBackend() {
  const char* backend = std::getenv("FUSED_CPP_POST_GEMM_BACKEND");
  if (backend != nullptr) {
    if (std::strcmp(backend, "sve") == 0 || std::strcmp(backend, "SVE") == 0) {
      TORCH_CHECK(::fused_cpp::deepseek_v4::attn_sve::available(),
                  "post GEMM backend=sve requested but this build/CPU does not "
                  "support SVE BF16");
      return PostGemmBackend::kSve;
    }
    if (std::strcmp(backend, "neon") == 0 || std::strcmp(backend, "NEON") == 0) {
      return PostGemmBackend::kNeon;
    }
    TORCH_CHECK(
        std::strcmp(backend, "auto") == 0 || std::strcmp(backend, "AUTO") == 0 || std::strcmp(backend, "default") == 0,
        "FUSED_CPP_POST_GEMM_BACKEND must be "
        "one of auto/default/neon/sve, got ",
        backend);
  }
  if (::fused_cpp::deepseek_v4::attn_sve::available()) {
    return PostGemmBackend::kSve;
  }
  return PostGemmBackend::kNeon;
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

bool IsInt32Tensor(const at::Tensor& tensor) { return tensor.scalar_type() == at::kInt; }

#if defined(__ARM_FEATURE_SVE)
inline svuint32_t F32ToBf16BitsSve(svbool_t pg, svfloat32_t v) {
  const svuint32_t bits = svreinterpret_u32_f32(v);
  const svuint32_t lsb = svand_n_u32_x(pg, svlsr_n_u32_x(pg, bits, 16), 1);
  const svuint32_t bias = svadd_n_u32_x(pg, lsb, 0x7fff);
  return svlsr_n_u32_x(pg, svadd_u32_x(pg, bits, bias), 16);
}

inline void StoreBf16F32Sve(svbool_t pg, uint16_t* ptr, svfloat32_t v) { svst1h_u32(pg, ptr, F32ToBf16BitsSve(pg, v)); }

inline void ScatterBf16EvenOddF32Sve(svbool_t pg, uint16_t* ptr, svfloat32_t v) {
  const svuint32_t idx = svindex_u32(0, 2);
  svst1h_scatter_u32index_u32(pg, ptr, idx, F32ToBf16BitsSve(pg, v));
}

template <typename cache_t>
inline void StoreCompressorCacheF32Sve(svbool_t pg, cache_t* ptr, svfloat32_t v) {
  if constexpr (std::is_same_v<cache_t, float>) {
    svst1_f32(pg, ptr, v);
  } else {
    StoreBf16F32Sve(pg, ptr, v);
  }
}

template <typename cache_t>
inline void StoreCompressorCacheEvenOddF32Sve(svbool_t pg, cache_t* ptr, svfloat32_t even, svfloat32_t odd) {
  if constexpr (std::is_same_v<cache_t, float>) {
    svst2_f32(pg, ptr, svcreate2_f32(even, odd));
  } else {
    ScatterBf16EvenOddF32Sve(pg, ptr, even);
    ScatterBf16EvenOddF32Sve(pg, ptr + 1, odd);
  }
}

inline svfloat32_t ExpApproxF32Sve(svbool_t pg, svfloat32_t x, svfloat32_t exp_hi, svfloat32_t exp_lo,
                                   svfloat32_t inv_ln2, svfloat32_t ln2, svfloat32_t c0, svfloat32_t c1, svfloat32_t c2,
                                   svfloat32_t c3, svfloat32_t c4, svfloat32_t c5, svfloat32_t c6) {
  x = svmin_f32_x(pg, x, exp_hi);
  x = svmax_f32_x(pg, x, exp_lo);
  const svfloat32_t y = svmul_f32_x(pg, x, inv_ln2);
  const svfloat32_t fn = svrinta_f32_x(pg, y);
  const svfloat32_t r = svmls_f32_x(pg, x, fn, ln2);
  const svint32_t n = svcvt_s32_f32_x(pg, fn);

  svfloat32_t p = c6;
  p = svmla_f32_x(pg, c5, p, r);
  p = svmla_f32_x(pg, c4, p, r);
  p = svmla_f32_x(pg, c3, p, r);
  p = svmla_f32_x(pg, c2, p, r);
  p = svmla_f32_x(pg, c1, p, r);
  p = svmla_f32_x(pg, c0, p, r);
  return svscale_f32_x(pg, p, n);
}

inline void WriteSparseIndexerShortPathRowSve(int32_t* dst, const int32_t* arange_src, int64_t valid_len,
                                              int64_t row_width) {
  const int64_t take = std::max<int64_t>(0, valid_len);
  const int64_t vl = static_cast<int64_t>(svcntw());
  const svbool_t pg_all = svptrue_b32();

  int64_t col = 0;
  for (; col + vl <= take; col += vl) {
    const svint32_t values = svld1_s32(pg_all, arange_src + col);
    svst1_s32(pg_all, dst + col, values);
  }
  if (col < take) {
    const svbool_t pg = svwhilelt_b32(col, take);
    const svint32_t values = svld1_s32(pg, arange_src + col);
    svst1_s32(pg, dst + col, values);
  }

  const svint32_t neg_one = svdup_s32(-1);
  col = take;
  for (; col + vl <= row_width; col += vl) {
    svst1_s32(pg_all, dst + col, neg_one);
  }
  if (col < row_width) {
    const svbool_t pg = svwhilelt_b32(col, row_width);
    svst1_s32(pg, dst + col, neg_one);
  }
}

inline void SavePartialStateRowSve(const float* kv_row, const float* score_row, const float* ape_row, float* state_row,
                                   int64_t state_width) {
  int64_t d = 0;
  for (; d < state_width; d += static_cast<int64_t>(svcntw())) {
    const svbool_t pg = svwhilelt_b32(d, state_width);
    const svfloat32_t kv_v = svld1_f32(pg, kv_row + d);
    svst1_f32(pg, state_row + d, kv_v);
  }

  float* score_state_row = state_row + state_width;
  d = 0;
  for (; d < state_width; d += static_cast<int64_t>(svcntw())) {
    const svbool_t pg = svwhilelt_b32(d, state_width);
    const svfloat32_t score_v = svld1_f32(pg, score_row + d);
    const svfloat32_t ape_v = svld1_f32(pg, ape_row + d);
    const svfloat32_t out_v = svadd_f32_x(pg, score_v, ape_v);
    svst1_f32(pg, score_state_row + d, out_v);
  }
}

bool TrySavePartialStatesSve(const at::Tensor& kv, const at::Tensor& score, const at::Tensor& ape,
                             const at::Tensor& positions, const at::Tensor& state_cache, const at::Tensor& slot_mapping,
                             int64_t compress_ratio) {
  const int64_t num_tokens = kv.size(0);
  const int64_t state_width = state_cache.size(-1) / 2;
  const bool supported = kv.scalar_type() == at::kFloat && score.scalar_type() == at::kFloat &&
                         ape.scalar_type() == at::kFloat && state_cache.scalar_type() == at::kFloat && kv.dim() == 2 &&
                         score.dim() == 2 && ape.dim() == 2 && state_cache.dim() == 3 && score.size(0) == num_tokens &&
                         kv.size(1) == state_width && score.size(1) == state_width && ape.size(1) == state_width &&
                         state_cache.size(-1) == 2 * state_width && kv.stride(1) == 1 && score.stride(1) == 1 &&
                         ape.stride(1) == 1 && state_cache.stride(2) == 1 && compress_ratio > 0;

#if FUSED_CPP_STRICT_MODE
  TORCH_CHECK(supported,
              "FUSED_CPP_STRICT_MODE SavePartialStates requires fp32 kv/score/ape/state_cache, "
              "2-D kv/score/ape, 3-D state_cache, contiguous last dims, and compress_ratio > 0");
#else
  if (!supported) {
    return false;
  }
#endif

  at::Tensor slots = ToLongCpu(slot_mapping).reshape({-1}).contiguous();
  at::Tensor pos_cpu = ToLongCpu(positions).reshape({-1}).contiguous();
#if FUSED_CPP_STRICT_MODE
  TORCH_CHECK(slots.numel() >= num_tokens,
              "FUSED_CPP_STRICT_MODE SavePartialStates slot_mapping must cover all tokens");
  TORCH_CHECK(pos_cpu.numel() >= num_tokens, "FUSED_CPP_STRICT_MODE SavePartialStates positions must cover all tokens");
  TORCH_CHECK(ape.size(0) >= compress_ratio,
              "FUSED_CPP_STRICT_MODE SavePartialStates ape rows must cover compress_ratio");
#else
  if (slots.numel() < num_tokens || pos_cpu.numel() < num_tokens || ape.size(0) < compress_ratio) {
    return false;
  }
#endif

  const int64_t block_size = state_cache.size(1);
  const int64_t* slot_data = slots.data_ptr<int64_t>();
  const int64_t* pos_data = pos_cpu.data_ptr<int64_t>();
  const float* kv_data = kv.data_ptr<float>();
  const float* score_data = score.data_ptr<float>();
  const float* ape_data = ape.data_ptr<float>();
  float* state_data = state_cache.data_ptr<float>();
  const int64_t kv_stride0 = kv.stride(0);
  const int64_t score_stride0 = score.stride(0);
  const int64_t ape_stride0 = ape.stride(0);
  const int64_t state_stride0 = state_cache.stride(0);
  const int64_t state_stride1 = state_cache.stride(1);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t i = 0; i < num_tokens; ++i) {
    const int64_t slot = slot_data[i];
    if (slot < 0) {
      continue;
    }
    const int64_t block = slot / block_size;
    const int64_t offset = slot % block_size;
    const int64_t ape_row = std::max<int64_t>(pos_data[i] % compress_ratio, 0);
    float* state_row = state_data + block * state_stride0 + offset * state_stride1;
    const float* kv_row = kv_data + i * kv_stride0;
    const float* score_row = score_data + i * score_stride0;
    const float* ape_row_ptr = ape_data + ape_row * ape_stride0;
    SavePartialStateRowSve(kv_row, score_row, ape_row_ptr, state_row, state_width);
  }
  return true;
}
#endif

inline void WriteSparseIndexerShortPathRowScalar(int32_t* dst, const int32_t* arange_src, int64_t valid_len,
                                                 int64_t row_width) {
  const int64_t take = std::max<int64_t>(0, valid_len);
  if (take > 0) {
    std::memcpy(dst, arange_src, static_cast<size_t>(take) * sizeof(int32_t));
  }
  std::fill(dst + take, dst + row_width, int32_t{-1});
}

bool TryWriteSparseIndexerShortPathRaw(const at::Tensor& topk_indices_buffer, const at::Tensor& ks_cpu,
                                       const at::Tensor& ke_cpu, const at::Tensor& arange_topk, int64_t num_tokens,
                                       int64_t topk_tokens) {
  if (!topk_indices_buffer.device().is_cpu() || topk_indices_buffer.scalar_type() != at::kInt ||
      topk_indices_buffer.dim() != 2 || topk_indices_buffer.size(0) < num_tokens ||
      topk_indices_buffer.size(1) < topk_tokens || topk_indices_buffer.stride(0) < topk_indices_buffer.size(1) ||
      topk_indices_buffer.stride(1) != 1 || !ks_cpu.device().is_cpu() || !ke_cpu.device().is_cpu() ||
      ks_cpu.scalar_type() != at::kLong || ke_cpu.scalar_type() != at::kLong || !ks_cpu.is_contiguous() ||
      !ke_cpu.is_contiguous() || ks_cpu.numel() < num_tokens || ke_cpu.numel() < num_tokens ||
      !arange_topk.device().is_cpu() || arange_topk.scalar_type() != at::kInt || !arange_topk.is_contiguous() ||
      arange_topk.numel() < topk_tokens) {
    return false;
  }

  int32_t* out = topk_indices_buffer.data_ptr<int32_t>();
  const int64_t out_stride0 = topk_indices_buffer.stride(0);
  const int64_t row_width = topk_indices_buffer.size(1);
  const int64_t* ks = ks_cpu.data_ptr<int64_t>();
  const int64_t* ke = ke_cpu.data_ptr<int64_t>();
  const int32_t* arange_src = arange_topk.data_ptr<int32_t>();

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (num_tokens > 1)
#endif
  for (int64_t i = 0; i < num_tokens; ++i) {
    int32_t* row = out + i * out_stride0;
    const int64_t valid_len = ke[i] - ks[i];
#if defined(__ARM_FEATURE_SVE)
    WriteSparseIndexerShortPathRowSve(row, arange_src, valid_len, row_width);
#else
    WriteSparseIndexerShortPathRowScalar(row, arange_src, valid_len, row_width);
#endif
  }
  return true;
}

const uint16_t* Bf16ConstData(const at::Tensor& tensor) {
  return reinterpret_cast<const uint16_t*>(tensor.data_ptr<at::BFloat16>());
}

uint16_t* Bf16Data(const at::Tensor& tensor) { return reinterpret_cast<uint16_t*>(tensor.data_ptr<at::BFloat16>()); }

#if defined(__aarch64__)
void DispatchPostGemmF32Neon(const uint16_t* A, const uint16_t* B_reo, float* C, uint16_t* A_reorder, int M, int K,
                             int N, int ldc) {
  gemm_params_t p;
  p.lda = K;
  p.ldb = K;
  p.ldc = ldc;

  int processed = 0;
  const int m_full = (M / 8) * 8;
  if (m_full > 0) {
    p.m = m_full;
    p.k = K;
    p.n = N;
    bf16gemm_k_ld(A, B_reo, C, A_reorder, &p);
    processed = m_full;
  }

  int m_rem = M - processed;
  if (m_rem == 0) {
    return;
  }

  const uint16_t* At = A + static_cast<int64_t>(processed) * K;
  float* Ct = C + static_cast<int64_t>(processed) * ldc;
  uint16_t* A_reo_t = A_reorder + static_cast<int64_t>(processed) * K;
  if (m_rem >= 4) {
    p.m = 4;
    p.k = K;
    p.n = N;
    bf16gemm_k_ld4(At, B_reo, Ct, A_reo_t, &p);
    processed += 4;
    m_rem -= 4;
    At = A + static_cast<int64_t>(processed) * K;
    Ct = C + static_cast<int64_t>(processed) * ldc;
    A_reo_t = A_reorder + static_cast<int64_t>(processed) * K;
  }
  if (m_rem >= 2) {
    p.m = 2;
    p.k = K;
    p.n = N;
    bf16gemm_k_ld2(At, B_reo, Ct, A_reo_t, &p);
    processed += 2;
    m_rem -= 2;
    At = A + static_cast<int64_t>(processed) * K;
    Ct = C + static_cast<int64_t>(processed) * ldc;
    A_reo_t = A_reorder + static_cast<int64_t>(processed) * K;
  }
  if (m_rem >= 1) {
    p.m = 1;
    p.k = K;
    p.n = N;
    bf16gemm_k_ld1(At, B_reo, Ct, A_reo_t, &p);
  }
}

void DispatchPostGemmBf16Neon(const uint16_t* A, const uint16_t* B_reo, uint16_t* C, uint16_t* A_reorder, int M, int K,
                              int N, int ldc) {
#if defined(__linux__)
  gemm_params_t p;
  p.lda = K;
  p.ldb = K;
  p.ldc = ldc;

  int processed = 0;
  const int m_full = (M / 8) * 8;
  if (m_full > 0) {
    p.m = m_full;
    p.k = K;
    p.n = N;
    bf16gemm_k_nld_b(A, B_reo, C, A_reorder, &p);
    processed = m_full;
  }

  int m_rem = M - processed;
  if (m_rem == 0) {
    return;
  }

  const uint16_t* At = A + static_cast<int64_t>(processed) * K;
  uint16_t* Ct = C + static_cast<int64_t>(processed) * ldc;
  uint16_t* A_reo_t = A_reorder + static_cast<int64_t>(processed) * K;
  if (m_rem >= 4) {
    p.m = 4;
    p.k = K;
    p.n = N;
    bf16gemm_k_nld4_b(At, B_reo, Ct, A_reo_t, &p);
    processed += 4;
    m_rem -= 4;
    At = A + static_cast<int64_t>(processed) * K;
    Ct = C + static_cast<int64_t>(processed) * ldc;
    A_reo_t = A_reorder + static_cast<int64_t>(processed) * K;
  }
  if (m_rem >= 2) {
    p.m = 2;
    p.k = K;
    p.n = N;
    bf16gemm_k_nld2_b(At, B_reo, Ct, A_reo_t, &p);
    processed += 2;
    m_rem -= 2;
    At = A + static_cast<int64_t>(processed) * K;
    Ct = C + static_cast<int64_t>(processed) * ldc;
    A_reo_t = A_reorder + static_cast<int64_t>(processed) * K;
  }
  if (m_rem >= 1) {
    p.m = 1;
    p.k = K;
    p.n = N;
    bf16gemm_k_nld1_b(At, B_reo, Ct, A_reo_t, &p);
  }
#else
  std::vector<float> tmp(static_cast<size_t>(M) * static_cast<size_t>(ldc));
  DispatchPostGemmF32Neon(A, B_reo, tmp.data(), A_reorder, M, K, N, ldc);
  for (int64_t row = 0; row < M; ++row) {
    for (int64_t col = 0; col < N; ++col) {
      const int64_t offset = row * ldc + col;
      uint32_t bits;
      std::memcpy(&bits, &tmp[static_cast<size_t>(offset)], sizeof(bits));
      const uint32_t lsb = (bits >> 16) & 1u;
      bits += 0x7fffu + lsb;
      C[offset] = static_cast<uint16_t>(bits >> 16);
    }
  }
#endif
}

void DispatchPostGemmRange(const uint16_t* a_ptr, const uint16_t* b_ptr, at::Tensor& output, uint16_t* thread_scratch,
                           PostGemmBackend backend, at::ScalarType dtype, int64_t row_start, int64_t row_count,
                           int64_t K, int64_t Np, int64_t n_begin, int64_t n_cols) {
  if (row_count <= 0 || n_cols <= 0) {
    return;
  }
  const uint16_t* a_row = a_ptr + row_start * K;
  const uint16_t* b_group = b_ptr + n_begin * K;
  if (dtype == at::kBFloat16) {
    uint16_t* c_row = Bf16Data(output) + row_start * Np + n_begin;
    if (backend == PostGemmBackend::kSve) {
      ::fused_cpp::deepseek_v4::attn_sve::dispatch_bf16(a_row, b_group, c_row, thread_scratch,
                                                        static_cast<int>(row_count), static_cast<int>(K),
                                                        static_cast<int>(n_cols), static_cast<int>(Np));
    } else {
      DispatchPostGemmBf16Neon(a_row, b_group, c_row, thread_scratch, static_cast<int>(row_count), static_cast<int>(K),
                               static_cast<int>(n_cols), static_cast<int>(Np));
    }
    return;
  }

  float* c_row = output.data_ptr<float>() + row_start * Np + n_begin;
  if (backend == PostGemmBackend::kSve) {
    ::fused_cpp::deepseek_v4::attn_sve::dispatch_f32(a_row, b_group, c_row, thread_scratch, static_cast<int>(row_count),
                                                     static_cast<int>(K), static_cast<int>(n_cols),
                                                     static_cast<int>(Np));
  } else {
    DispatchPostGemmF32Neon(a_row, b_group, c_row, thread_scratch, static_cast<int>(row_count), static_cast<int>(K),
                            static_cast<int>(n_cols), static_cast<int>(Np));
  }
}

void DispatchPostGemmPackedSveRange(const uint16_t* packed_a, const uint16_t* b_ptr, at::Tensor& output,
                                    at::ScalarType dtype, int64_t row_start, int64_t row_count, int64_t K, int64_t Np,
                                    int64_t n_begin, int64_t n_cols) {
  if (row_count <= 0 || n_cols <= 0) {
    return;
  }
  const uint16_t* packed_panel = packed_a + row_start * K;
  const uint16_t* b_group = b_ptr + n_begin * K;
  if (dtype == at::kBFloat16) {
    ::fused_cpp::deepseek_v4::attn_sve::dispatch_packed_bf16(
        packed_panel, b_group, Bf16Data(output) + row_start * Np + n_begin, static_cast<int>(row_count),
        static_cast<int>(K), static_cast<int>(n_cols), static_cast<int>(Np));
    return;
  }
  ::fused_cpp::deepseek_v4::attn_sve::dispatch_packed_f32(
      packed_panel, b_group, output.data_ptr<float>() + row_start * Np + n_begin, static_cast<int>(row_count),
      static_cast<int>(K), static_cast<int>(n_cols), static_cast<int>(Np));
}

void DispatchPostGemmPackedRange(const uint16_t* packed_a, const uint16_t* b_ptr, at::Tensor& output,
                                 PostGemmBackend backend, at::ScalarType dtype, int64_t row_start, int64_t row_count,
                                 int64_t K, int64_t Np, int64_t n_begin, int64_t n_cols) {
  if (backend == PostGemmBackend::kSve) {
    DispatchPostGemmPackedSveRange(packed_a, b_ptr, output, dtype, row_start, row_count, K, Np, n_begin, n_cols);
    return;
  }
  if (row_count <= 0 || n_cols <= 0) {
    return;
  }
  gemm_params_t params;
  params.m = static_cast<int>(row_count);
  params.k = static_cast<int>(K);
  params.n = static_cast<int>(n_cols);
  params.lda = static_cast<int>(K);
  params.ldb = static_cast<int>(K);
  params.ldc = static_cast<int>(Np);
  const uint16_t* panel_a = packed_a + row_start * K;
  const uint16_t* panel_b = b_ptr + n_begin * K;
  if (dtype == at::kBFloat16) {
    deepseek_v4_attn_gemm_packed_bf16(panel_a, panel_b, Bf16Data(output) + row_start * Np + n_begin, nullptr, &params);
    return;
  }
  deepseek_v4_attn_gemm_packed_f32(panel_a, panel_b, output.data_ptr<float>() + row_start * Np + n_begin, nullptr,
                                   &params);
}
#endif

void CheckPostLinearPrepackedArgs(const at::Tensor& input, const at::Tensor& packed_weight, int64_t K, int64_t N,
                                  int64_t Np, at::ScalarType dtype, const char* name) {
  TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kFloat, name, " only supports bf16/fp32 output, got ", dtype);
  CheckCpuTensor(input, "post GEMM input");
  CheckCpuTensor(packed_weight, name);
  CheckDim(input, "post GEMM input", 2);
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "post GEMM input must be torch.bfloat16, got ",
              input.scalar_type());
  TORCH_CHECK(packed_weight.scalar_type() == at::kBFloat16 && packed_weight.is_contiguous(), name,
              " must be contiguous torch.bfloat16");
  TORCH_CHECK(input.size(1) == K, name, " K mismatch: input K=", input.size(1), " packed K=", K);
  TORCH_CHECK(K > 0 && N > 0 && Np >= N, name, " invalid metadata K=", K, " N=", N, " Np=", Np);
  TORCH_CHECK(K % 8 == 0 && Np % 8 == 0, name, " requires K multiple of 8 and Np multiple of 8, got K=", K, " Np=", Np);
  TORCH_CHECK(packed_weight.numel() == K * Np, name, " numel mismatch: expected ", K * Np, ", got ",
              packed_weight.numel());
}

at::Tensor PostLinearPrepackedToDtypeWorkspace(const at::Tensor& input, const at::Tensor& packed_weight, int64_t K,
                                               int64_t N, int64_t Np, at::ScalarType dtype,
                                               ::fused_cpp::workspace::WorkspaceLease& workspace) {
  CheckPostLinearPrepackedArgs(input, packed_weight, K, N, Np, dtype,
                               "PostLinearPrepackedToDtypeWorkspace packed_weight");

  const int64_t M = input.size(0);
  if (M == 0) {
    at::Tensor output = at::empty({M, Np}, input.options().dtype(dtype));
    return N == Np ? output : output.narrow(1, 0, N).contiguous();
  }

#if !defined(__aarch64__)
  (void)workspace;
  return ::bf16_linear_prepacked_to_dtype(input, packed_weight, K, N, Np, dtype == at::kBFloat16, 0);
#else
  const PostGemmBackend backend = SelectedPostGemmBackend();
  if (backend == PostGemmBackend::kSve) {
    TORCH_CHECK(Np % ::fused_cpp::deepseek_v4::attn_sve::n_tile() == 0,
                "post GEMM SVE packed Np must be multiple of SVE n_tile=", ::fused_cpp::deepseek_v4::attn_sve::n_tile(),
                ", got ", Np);
  }

  int64_t num_threads = 1;
#ifdef _OPENMP
  num_threads = omp_get_max_threads();
  if (num_threads <= 0) {
    num_threads = 1;
  }
#endif
  const bool m8_aligned = PostGemmM8AlignedEnabled();
  const int64_t panel_rows = PostGemmMPanelRows(backend);
  const int64_t m_panels = (M + panel_rows - 1) / panel_rows;
  at::Tensor input_contig = input.contiguous();
  const uint16_t* a_ptr = Bf16ConstData(input_contig);
  const uint16_t* b_ptr = Bf16ConstData(packed_weight);

  if (!m8_aligned) {
    at::Tensor output = at::empty({M, Np}, input.options().dtype(dtype));
    const int64_t rows_per_thread = (M + num_threads - 1) / num_threads;
    const int64_t scratch_stride = backend == PostGemmBackend::kSve
                                       ? ::fused_cpp::deepseek_v4::attn_sve::a_scratch_elems(rows_per_thread, K)
                                       : std::max<int64_t>(1, rows_per_thread * K);
    at::Tensor scratch = workspace.empty({std::max<int64_t>(1, num_threads * scratch_stride)}, input.options());
    uint16_t* scratch_ptr = Bf16Data(scratch);

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
#ifdef _OPENMP
      const int64_t tid = omp_get_thread_num();
#else
      const int64_t tid = 0;
#endif
      const int64_t row_start = tid * rows_per_thread;
      const int64_t row_count = row_start >= M ? 0 : std::min<int64_t>(rows_per_thread, M - row_start);
      if (row_count > 0) {
        DispatchPostGemmRange(a_ptr, b_ptr, output, scratch_ptr + tid * scratch_stride, backend, dtype, row_start,
                              row_count, K, Np, 0, Np);
      }
    }
    return N == Np ? output : output.narrow(1, 0, N).contiguous();
  }

  const int64_t output_rows = backend == PostGemmBackend::kNeon ? m_panels * panel_rows : M;
  at::Tensor output = at::empty({output_rows, Np}, input.options().dtype(dtype));
  const int64_t packed_a_size =
      backend == PostGemmBackend::kSve ? ::fused_cpp::deepseek_v4::attn_sve::packed_a_elems(M, K) : output_rows * K;
  at::Tensor packed_a = workspace.empty({packed_a_size}, input.options());
  uint16_t* packed_a_ptr = Bf16Data(packed_a);
  const int64_t n_tile = backend == PostGemmBackend::kSve ? ::fused_cpp::deepseek_v4::attn_sve::n_tile() : 8;
  const int64_t n_tiles = Np / n_tile;
  const int64_t n_groups = ChoosePostGemmWindowGroups(K, Np, n_tile, num_threads);
  const int64_t m_splits = std::min<int64_t>(m_panels, std::max<int64_t>(1, (num_threads + n_groups - 1) / n_groups));
  const int64_t task_count = n_groups * m_splits;

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
  {
#ifdef _OPENMP
    const int64_t tid = omp_get_thread_num();
#else
    const int64_t tid = 0;
#endif
    const int64_t pack_begin = tid * m_panels / num_threads;
    const int64_t pack_end = (tid + 1) * m_panels / num_threads;
    if (backend == PostGemmBackend::kSve) {
      ::fused_cpp::deepseek_v4::attn_sve::pack_a_range(a_ptr, packed_a_ptr, static_cast<int>(M), static_cast<int>(K),
                                                       static_cast<int>(pack_begin), static_cast<int>(pack_end));
    } else {
      PackPostGemmANeonRange(a_ptr, packed_a_ptr, M, K, pack_begin, pack_end);
    }
#ifdef _OPENMP
#pragma omp barrier
#pragma omp for schedule(static, 1)
#endif
    for (int64_t task = 0; task < task_count; ++task) {
      const int64_t group = task / m_splits;
      const int64_t split = task % m_splits;
      const int64_t tile_begin = group * n_tiles / n_groups;
      const int64_t tile_end = (group + 1) * n_tiles / n_groups;
      const int64_t panel_begin = split * m_panels / m_splits;
      const int64_t panel_end = (split + 1) * m_panels / m_splits;
      const int64_t row_start = panel_begin * panel_rows;
      const int64_t physical_row_count = (panel_end - panel_begin) * panel_rows;
      if (backend == PostGemmBackend::kNeon) {
        DispatchPostGemmPackedRange(packed_a_ptr, b_ptr, output, backend, dtype, row_start, physical_row_count, K, Np,
                                    tile_begin * n_tile, (tile_end - tile_begin) * n_tile);
      } else {
        for (int64_t panel = panel_begin; panel < panel_end; ++panel) {
          const int64_t panel_row_start = panel * panel_rows;
          const int64_t panel_row_count = std::min<int64_t>(panel_rows, M - panel_row_start);
          DispatchPostGemmPackedRange(packed_a_ptr, b_ptr, output, backend, dtype, panel_row_start, panel_row_count, K,
                                      Np, tile_begin * n_tile, (tile_end - tile_begin) * n_tile);
        }
      }
    }
  }
  at::Tensor result = output_rows == M ? output : output.narrow(0, 0, M);
  return N == Np ? result : result.narrow(1, 0, N).contiguous();
#endif
}

std::pair<at::Tensor, at::Tensor> PostLinearPrepackedPairToDtypeWorkspace(
    const at::Tensor& input, const at::Tensor& first_weight, int64_t first_K, int64_t first_N, int64_t first_Np,
    const at::Tensor& second_weight, int64_t second_K, int64_t second_N, int64_t second_Np, at::ScalarType dtype,
    ::fused_cpp::workspace::WorkspaceLease& workspace) {
#if !defined(__aarch64__) || !defined(_OPENMP)
  at::Tensor first =
      PostLinearPrepackedToDtypeWorkspace(input, first_weight, first_K, first_N, first_Np, dtype, workspace);
  at::Tensor second =
      PostLinearPrepackedToDtypeWorkspace(input, second_weight, second_K, second_N, second_Np, dtype, workspace);
  return std::make_pair(first, second);
#else
  CheckPostLinearPrepackedArgs(input, first_weight, first_K, first_N, first_Np, dtype,
                               "shared main Q GEMM packed_weight");
  CheckPostLinearPrepackedArgs(input, second_weight, second_K, second_N, second_Np, dtype,
                               "shared indexer Q GEMM packed_weight");

  const int64_t M = input.size(0);
  at::Tensor first_output = at::empty({M, first_Np}, input.options().dtype(dtype));
  at::Tensor second_output = at::empty({M, second_Np}, input.options().dtype(dtype));
  if (M == 0) {
    if (first_N != first_Np) {
      first_output = first_output.narrow(1, 0, first_N).contiguous();
    }
    if (second_N != second_Np) {
      second_output = second_output.narrow(1, 0, second_N).contiguous();
    }
    return std::make_pair(first_output, second_output);
  }

  const PostGemmBackend backend = SelectedPostGemmBackend();
  TORCH_CHECK(first_K == second_K, "shared Main Q/Indexer Q GEMMs must have the same K, got ", first_K, " and ",
              second_K);
  if (backend == PostGemmBackend::kSve) {
    const int64_t n_tile = ::fused_cpp::deepseek_v4::attn_sve::n_tile();
    TORCH_CHECK(first_Np % n_tile == 0 && second_Np % n_tile == 0,
                "shared Q GEMM SVE packed Np values must be multiples of SVE n_tile=", n_tile, ", got ", first_Np,
                " and ", second_Np);
  }

  int64_t num_threads = omp_get_max_threads();
  if (num_threads <= 0) {
    num_threads = 1;
  }
  const int64_t panel_rows = PostGemmMPanelRows(backend);
  const int64_t m_panels = (M + panel_rows - 1) / panel_rows;
  const int64_t scratch_stride =
      backend == PostGemmBackend::kSve ? 1 : std::max<int64_t>(1, panel_rows * std::max(first_K, second_K));
  at::Tensor scratch = workspace.empty({num_threads * scratch_stride}, input.options());
  at::Tensor packed_a;
  if (backend == PostGemmBackend::kSve) {
    packed_a = workspace.empty({::fused_cpp::deepseek_v4::attn_sve::packed_a_elems(M, first_K)}, input.options());
  }
  at::Tensor input_contig = input.contiguous();

  struct PostGemmPairWork {
    const uint16_t* b_ptr;
    at::Tensor* output;
    int64_t K;
    int64_t Np;
  };
  struct PostGemmTaskGroup {
    int work_index;
    int64_t n_begin;
    int64_t n_cols;
  };
  struct alignas(64) PostGemmPanelCursor {
    std::atomic<int64_t> next_panel{0};
  };

  const uint16_t* a_ptr = Bf16ConstData(input_contig);
  uint16_t* scratch_ptr = Bf16Data(scratch);
  uint16_t* packed_a_ptr = packed_a.defined() ? Bf16Data(packed_a) : nullptr;
  std::array<PostGemmPairWork, 2> work = {
      PostGemmPairWork{Bf16ConstData(first_weight), &first_output, first_K, first_Np},
      PostGemmPairWork{Bf16ConstData(second_weight), &second_output, second_K, second_Np},
  };
  const int64_t n_tile = backend == PostGemmBackend::kSve ? ::fused_cpp::deepseek_v4::attn_sve::n_tile() : 8;
  std::array<int, 2> groups_per_work = {1, 1};
  int total_n_tiles = 0;
  for (const PostGemmPairWork& item : work) {
    total_n_tiles += static_cast<int>(item.Np / n_tile);
  }
  const int requested_groups = RequestedPostGemmNGroups(backend, static_cast<int>(num_threads));
  const int target_groups = std::min(total_n_tiles, requested_groups);
  for (int assigned = static_cast<int>(work.size()); assigned < target_groups; ++assigned) {
    int best = -1;
    for (int work_index = 0; work_index < static_cast<int>(work.size()); ++work_index) {
      const int tiles = static_cast<int>(work[static_cast<size_t>(work_index)].Np / n_tile);
      if (groups_per_work[static_cast<size_t>(work_index)] >= tiles) {
        continue;
      }
      if (best < 0) {
        best = work_index;
        continue;
      }
      const int best_tiles = static_cast<int>(work[static_cast<size_t>(best)].Np / n_tile);
      if (static_cast<int64_t>(tiles) * groups_per_work[static_cast<size_t>(best)] >
          static_cast<int64_t>(best_tiles) * groups_per_work[static_cast<size_t>(work_index)]) {
        best = work_index;
      }
    }
    if (best < 0) {
      break;
    }
    ++groups_per_work[static_cast<size_t>(best)];
  }

  std::vector<PostGemmTaskGroup> task_groups;
  task_groups.reserve(static_cast<size_t>(target_groups));
  for (int work_index = 0; work_index < static_cast<int>(work.size()); ++work_index) {
    const int64_t n_tiles = work[static_cast<size_t>(work_index)].Np / n_tile;
    const int splits = groups_per_work[static_cast<size_t>(work_index)];
    for (int split = 0; split < splits; ++split) {
      const int64_t tile_begin = static_cast<int64_t>(split) * n_tiles / splits;
      const int64_t tile_end = static_cast<int64_t>(split + 1) * n_tiles / splits;
      task_groups.push_back(PostGemmTaskGroup{work_index, tile_begin * n_tile, (tile_end - tile_begin) * n_tile});
    }
  }
  std::unique_ptr<PostGemmPanelCursor[]> cursors = std::make_unique<PostGemmPanelCursor[]>(task_groups.size());

#pragma omp parallel num_threads(num_threads)
  {
    const int64_t tid = omp_get_thread_num();
    uint16_t* thread_scratch = scratch_ptr + tid * scratch_stride;
    if (backend == PostGemmBackend::kSve) {
      const int panel_begin = static_cast<int>(tid * m_panels / num_threads);
      const int panel_end = static_cast<int>((tid + 1) * m_panels / num_threads);
      ::fused_cpp::deepseek_v4::attn_sve::pack_a_range(a_ptr, packed_a_ptr, static_cast<int>(M),
                                                       static_cast<int>(first_K), panel_begin, panel_end);
#pragma omp barrier
    }
    const int group_count = static_cast<int>(task_groups.size());
    int preferred_group = static_cast<int>(tid % group_count);
    while (true) {
      bool executed = false;
      for (int offset = 0; offset < group_count; ++offset) {
        const int group_index = (preferred_group + offset) % group_count;
        PostGemmPanelCursor& cursor = cursors[static_cast<size_t>(group_index)];
        if (cursor.next_panel.load(std::memory_order_relaxed) >= m_panels) {
          continue;
        }
        const int64_t panel = cursor.next_panel.fetch_add(1, std::memory_order_relaxed);
        if (panel >= m_panels) {
          continue;
        }
        const PostGemmTaskGroup& group = task_groups[static_cast<size_t>(group_index)];
        const int work_index = group.work_index;
        const PostGemmPairWork& item = work[static_cast<size_t>(work_index)];
        const int64_t row_start = panel * panel_rows;
        const int64_t row_count = std::min<int64_t>(panel_rows, M - row_start);
        if (backend == PostGemmBackend::kSve) {
          DispatchPostGemmPackedSveRange(packed_a_ptr, item.b_ptr, *item.output, dtype, row_start, row_count, item.K,
                                         item.Np, group.n_begin, group.n_cols);
        } else {
          DispatchPostGemmRange(a_ptr, item.b_ptr, *item.output, thread_scratch, backend, dtype, row_start, row_count,
                                item.K, item.Np, group.n_begin, group.n_cols);
        }
        preferred_group = group_index;
        executed = true;
        break;
      }
      if (!executed) {
        break;
      }
    }
  }

  if (first_N != first_Np) {
    first_output = first_output.narrow(1, 0, first_N).contiguous();
  }
  if (second_N != second_Np) {
    second_output = second_output.narrow(1, 0, second_N).contiguous();
  }
  return std::make_pair(first_output, second_output);
#endif
}

at::Tensor LinearToDtype(const at::Tensor& input, const at::Tensor& weight, at::ScalarType dtype) {
  TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kFloat,
              "LinearToDtype only supports bf16/fp32 output with bf16gemm, got ", dtype);
  return ::bf16_linear_to_dtype(input, weight, dtype == at::kBFloat16, 0);
}

at::Tensor LinearPrepackedToDtypeWorkspace(const at::Tensor& input, const at::Tensor& packed_weight, int64_t K,
                                           int64_t N, int64_t Np, at::ScalarType dtype,
                                           ::fused_cpp::workspace::WorkspaceLease& workspace) {
  TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kFloat,
              "LinearPrepackedToDtypeWorkspace only supports bf16/fp32 output "
              "with bf16gemm, got ",
              dtype);
  if (N != Np) {
    return PostLinearPrepackedToDtypeWorkspace(input, packed_weight, K, N, Np, dtype, workspace);
  }
  return PostLinearPrepackedToDtypeWorkspace(input, packed_weight, K, N, Np, dtype, workspace);
}

at::Tensor GptjRopeApply(const at::Tensor& x, const at::Tensor& cos_sin_cache, const at::Tensor& positions,
                         int64_t rope_head_dim) {
  TORCH_CHECK(rope_head_dim >= 0, "rope_head_dim must be non-negative");
  if (rope_head_dim == 0) {
    return x.to(at::kFloat);
  }
  TORCH_CHECK(rope_head_dim % 2 == 0, "rope_head_dim must be even, got ", rope_head_dim);
  CheckDim(cos_sin_cache, "cos_sin_cache", 2);
  TORCH_CHECK(cos_sin_cache.size(1) >= rope_head_dim, "cos_sin_cache last dim must cover rope_head_dim");

  const int64_t head_dim = x.size(-1);
  TORCH_CHECK(head_dim >= rope_head_dim, "x last dim must be >= rope_head_dim: ", head_dim, " vs ", rope_head_dim);
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

at::Tensor GptjRopeApplyScalar(const at::Tensor& x, const at::Tensor& cos_sin_cache, int64_t position,
                               int64_t rope_head_dim) {
  at::Tensor pos = at::full({}, position, at::TensorOptions().dtype(at::kLong).device(x.device()));
  return GptjRopeApply(x, cos_sin_cache, pos, rope_head_dim);
}

void NormRopeStoreCompressedAten(const at::Tensor& compressed, const at::Tensor& rms_weight, double rms_norm_eps,
                                 const at::Tensor& cos_sin_cache, int64_t compressed_pos, int64_t rope_head_dim,
                                 const at::Tensor& kv_cache, int64_t kv_slot, int64_t kv_cache_block_size) {
  at::Tensor var = compressed.pow(2).mean(-1, false);
  at::Tensor normed = compressed * at::rsqrt(var + rms_norm_eps) * rms_weight;
  at::Tensor rotated = GptjRopeApplyScalar(normed, cos_sin_cache, compressed_pos, rope_head_dim);

  const int64_t kv_block = kv_slot / kv_cache_block_size;
  const int64_t kv_offset = kv_slot % kv_cache_block_size;
  kv_cache.index({kv_block, kv_offset}).copy_(rotated.to(kv_cache.scalar_type()));
}

void CheckPositionsInRange(const at::Tensor& positions_long, int64_t num_tokens, int64_t max_position) {
  if (num_tokens == 0) {
    return;
  }
  if (positions_long.dim() == 0) {
    const int64_t pos = positions_long.item<int64_t>();
    TORCH_CHECK(pos >= 0 && pos < max_position, "position out of cos_sin_cache range: ", pos, " vs ", max_position);
    return;
  }
  const int64_t* pos_data = positions_long.data_ptr<int64_t>();
  for (int64_t i = 0; i < num_tokens; ++i) {
    const int64_t pos = pos_data[i];
    TORCH_CHECK(pos >= 0 && pos < max_position, "position out of cos_sin_cache range: ", pos, " vs ", max_position);
  }
}

void CheckMainQKvShape(const at::Tensor& q, const at::Tensor& kv) {
  TORCH_CHECK(kv.size(0) == q.size(0), "kv token count must match q");
  TORCH_CHECK(kv.size(1) == q.size(2), "kv last dim must match q head_dim");
}

template <typename scalar_t>
void QNormRopeFusedImpl(const at::Tensor& q, const at::Tensor& positions_long, const at::Tensor& cos_sin_f,
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
    const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(head_dim) + static_cast<float>(eps));

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
      const float even_norm =
          static_cast<float>(static_cast<scalar_t>(static_cast<float>(q_row[even_d * q_stride_d]) * inv_rms));
      const float odd_norm =
          static_cast<float>(static_cast<scalar_t>(static_cast<float>(q_row[odd_d * q_stride_d]) * inv_rms));
      const float c = cos_row[pair];
      const float s = sin_row[pair];
      q_row[even_d * q_stride_d] = static_cast<scalar_t>(even_norm * c - odd_norm * s);
      q_row[odd_d * q_stride_d] = static_cast<scalar_t>(odd_norm * c + even_norm * s);
    }
  }
}

void QNormRopeFused(const at::Tensor& q, const at::Tensor& positions, const at::Tensor& cos_sin_cache, double eps) {
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
  TORCH_CHECK(positions_long.dim() == 0 || positions_long.numel() == num_tokens,
              "positions must be scalar or have num_tokens elements");
  at::Tensor cos_sin_f = cos_sin_cache.to(at::kFloat).contiguous();
  if (rope_head_dim != 0) {
    CheckPositionsInRange(positions_long, num_tokens, cos_sin_f.size(0));
  }
#if FUSED_CPP_STRICT_MODE
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "FUSED_CPP_STRICT_MODE QNormRopeFused only supports bf16 q, got ",
              q.scalar_type());
#endif
  if (::fused_cpp::deepseek_v4::q_norm_rope_fused_sve(q, positions_long, cos_sin_f, eps)) {
    return;
  }

#if FUSED_CPP_STRICT_MODE
  TORCH_CHECK(false,
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
void KvRopeCacheInsertFusedImpl(const at::Tensor& kv, const at::Tensor& swa_kv_cache,
                                const at::Tensor& slot_mapping_long, const at::Tensor& positions_long,
                                const at::Tensor& cos_sin_f, bool do_rope) {
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
      cache_row =
          cache_data + (slot / cache_block_size) * cache_stride_block + (slot % cache_block_size) * cache_stride_offset;
    }
    for (int64_t d = 0; d < head_dim; ++d) {
      cache_row[d * cache_stride_d] = static_cast<cache_t>(static_cast<float>(kv_row[d * kv_stride_d]));
    }
  }
}

template <typename kv_t>
void DispatchKvRopeCacheInsertFusedCacheDtype(const at::Tensor& kv, const at::Tensor& swa_kv_cache,
                                              const at::Tensor& slot_mapping_long, const at::Tensor& positions_long,
                                              const at::Tensor& cos_sin_f, bool do_rope) {
  if (swa_kv_cache.numel() == 0 || swa_kv_cache.scalar_type() == at::kBFloat16) {
    KvRopeCacheInsertFusedImpl<kv_t, at::BFloat16>(kv, swa_kv_cache, slot_mapping_long, positions_long, cos_sin_f,
                                                   do_rope);
  } else if (swa_kv_cache.scalar_type() == at::kFloat) {
    KvRopeCacheInsertFusedImpl<kv_t, float>(kv, swa_kv_cache, slot_mapping_long, positions_long, cos_sin_f, do_rope);
  } else {
    TORCH_CHECK(false, "KvRopeCacheInsertFused only supports bf16/fp32 swa_kv_cache, got ", swa_kv_cache.scalar_type());
  }
}

void KvRopeCacheInsertFused(const at::Tensor& kv, const at::Tensor& swa_kv_cache, const at::Tensor& slot_mapping,
                            const at::Tensor& positions, const at::Tensor& cos_sin_cache) {
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
  TORCH_CHECK(positions_long.dim() == 0 || positions_long.numel() == num_tokens,
              "positions must be scalar or have num_tokens elements");
  at::Tensor slots_long;
  if (swa_kv_cache.numel() != 0) {
    TORCH_CHECK(swa_kv_cache.dim() == 2 || swa_kv_cache.dim() == 3, "swa_kv_cache must be 2-D or 3-D");
    TORCH_CHECK(swa_kv_cache.size(-1) == head_dim, "swa_kv_cache last dim must match kv head_dim");
    slots_long = ToLongCpu(slot_mapping).reshape({-1}).contiguous();
    TORCH_CHECK(slots_long.numel() >= num_tokens, "slot_mapping must cover all kv tokens");
    const int64_t slot_capacity =
        swa_kv_cache.dim() == 2 ? swa_kv_cache.size(0) : swa_kv_cache.size(0) * swa_kv_cache.size(1);
    const int64_t* slot_data = slots_long.data_ptr<int64_t>();
    for (int64_t i = 0; i < num_tokens; ++i) {
      const int64_t slot = slot_data[i];
      TORCH_CHECK(slot < slot_capacity, "slot_mapping slot exceeds swa_kv_cache capacity: ", slot, " vs ",
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
  TORCH_CHECK(kv.scalar_type() == at::kBFloat16,
              "FUSED_CPP_STRICT_MODE KvRopeCacheInsertFused only supports bf16 kv, got ", kv.scalar_type());
  if (swa_kv_cache.numel() != 0) {
    TORCH_CHECK(swa_kv_cache.scalar_type() == at::kBFloat16,
                "FUSED_CPP_STRICT_MODE KvRopeCacheInsertFused only supports bf16 swa_kv_cache, got ",
                swa_kv_cache.scalar_type());
  }
#endif

  if (!DeepSeekV4KvRopeWriteKvEnabled()) {
    const bool cache_insert_done = ::fused_cpp::deepseek_v4::kv_rope_cache_insert_fused_sve(
        kv, swa_kv_cache, slots_long, positions_long, cos_sin_f);
    if (cache_insert_done) {
      return;
    }
#if FUSED_CPP_STRICT_MODE
    TORCH_CHECK(swa_kv_cache.numel() == 0,
                "FUSED_CPP_STRICT_MODE KvRopeCacheInsertFused requires the SVE bf16 cache-insert fast path");
#endif
  }

  const bool rope_done = ::fused_cpp::deepseek_v4::kv_rope_fused_sve(kv, positions_long, cos_sin_f);
  if (rope_done && swa_kv_cache.numel() == 0) {
    return;
  }

#if FUSED_CPP_STRICT_MODE
  TORCH_CHECK(false,
              "FUSED_CPP_STRICT_MODE KvRopeCacheInsertFused requires the SVE bf16 fast path; "
              "fallback kv copy/rope path is disabled");
#else
  if (kv.scalar_type() == at::kBFloat16) {
    DispatchKvRopeCacheInsertFusedCacheDtype<at::BFloat16>(kv, swa_kv_cache, slots_long, positions_long, cos_sin_f,
                                                           !rope_done);
  } else if (kv.scalar_type() == at::kFloat) {
    DispatchKvRopeCacheInsertFusedCacheDtype<float>(kv, swa_kv_cache, slots_long, positions_long, cos_sin_f,
                                                    !rope_done);
  } else {
    TORCH_CHECK(false, "KvRopeCacheInsertFused only supports bf16/fp32 kv, got ", kv.scalar_type());
  }
#endif
}

void SavePartialStates(const at::Tensor& kv, const at::Tensor& score, const at::Tensor& ape,
                       const at::Tensor& positions, const at::Tensor& state_cache, const at::Tensor& slot_mapping,
                       int64_t compress_ratio) {
  if (state_cache.numel() == 0) {
    return;
  }
#if defined(__ARM_FEATURE_SVE)
  if (TrySavePartialStatesSve(kv, score, ape, positions, state_cache, slot_mapping, compress_ratio)) {
    return;
  }
#elif FUSED_CPP_STRICT_MODE
  TORCH_CHECK(false, "FUSED_CPP_STRICT_MODE SavePartialStates requires SVE");
#endif
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

#if defined(__ARM_FEATURE_SVE)
inline void CopyCompressorKvScoreRowSve4x(const float* kv_src, const float* score_src, float* kv_dst, float* score_dst,
                                          int64_t head_dim) {
  const int64_t vl = static_cast<int64_t>(svcntw());
  const int64_t step = 4 * vl;
  const svbool_t pg_all = svptrue_b32();
  int64_t d = 0;
  for (; d + step <= head_dim; d += step) {
    const svfloat32_t kv0 = svld1_f32(pg_all, kv_src + d);
    const svfloat32_t score0 = svld1_f32(pg_all, score_src + d);
    const svfloat32_t kv1 = svld1_f32(pg_all, kv_src + d + vl);
    const svfloat32_t score1 = svld1_f32(pg_all, score_src + d + vl);
    const svfloat32_t kv2 = svld1_f32(pg_all, kv_src + d + 2 * vl);
    const svfloat32_t score2 = svld1_f32(pg_all, score_src + d + 2 * vl);
    const svfloat32_t kv3 = svld1_f32(pg_all, kv_src + d + 3 * vl);
    const svfloat32_t score3 = svld1_f32(pg_all, score_src + d + 3 * vl);
    svst1_f32(pg_all, kv_dst + d, kv0);
    svst1_f32(pg_all, score_dst + d, score0);
    svst1_f32(pg_all, kv_dst + d + vl, kv1);
    svst1_f32(pg_all, score_dst + d + vl, score1);
    svst1_f32(pg_all, kv_dst + d + 2 * vl, kv2);
    svst1_f32(pg_all, score_dst + d + 2 * vl, score2);
    svst1_f32(pg_all, kv_dst + d + 3 * vl, kv3);
    svst1_f32(pg_all, score_dst + d + 3 * vl, score3);
  }
  for (; d < head_dim; d += vl) {
    const svbool_t pg = svwhilelt_b32(d, head_dim);
    const svfloat32_t kv = svld1_f32(pg, kv_src + d);
    const svfloat32_t score = svld1_f32(pg, score_src + d);
    svst1_f32(pg, kv_dst + d, kv);
    svst1_f32(pg, score_dst + d, score);
  }
}

inline void FillCompressorPaddingRowSve4x(float* kv_dst, float* score_dst, int64_t head_dim) {
  const int64_t vl = static_cast<int64_t>(svcntw());
  const int64_t step = 4 * vl;
  const svbool_t pg_all = svptrue_b32();
  const svfloat32_t zero = svdup_n_f32(0.0f);
  const svfloat32_t neg_inf = svdup_n_f32(-std::numeric_limits<float>::infinity());
  int64_t d = 0;
  for (; d + step <= head_dim; d += step) {
    svst1_f32(pg_all, kv_dst + d, zero);
    svst1_f32(pg_all, score_dst + d, neg_inf);
    svst1_f32(pg_all, kv_dst + d + vl, zero);
    svst1_f32(pg_all, score_dst + d + vl, neg_inf);
    svst1_f32(pg_all, kv_dst + d + 2 * vl, zero);
    svst1_f32(pg_all, score_dst + d + 2 * vl, neg_inf);
    svst1_f32(pg_all, kv_dst + d + 3 * vl, zero);
    svst1_f32(pg_all, score_dst + d + 3 * vl, neg_inf);
  }
  for (; d < head_dim; d += vl) {
    const svbool_t pg = svwhilelt_b32(d, head_dim);
    svst1_f32(pg, kv_dst + d, zero);
    svst1_f32(pg, score_dst + d, neg_inf);
  }
}

bool TryCopyCompressorWindowRowsSve(const at::Tensor& state_cache, const at::Tensor& block_table, int64_t req_idx,
                                    int64_t start, int64_t compress_ratio, int64_t coff, int64_t state_block_size,
                                    int64_t state_width, int64_t head_dim, at::Tensor& kv_stack,
                                    at::Tensor& score_stack) {
  const bool supported = state_cache.scalar_type() == at::kFloat && IsInt32Tensor(block_table) &&
                         kv_stack.scalar_type() == at::kFloat && score_stack.scalar_type() == at::kFloat &&
                         state_cache.device().is_cpu() && block_table.device().is_cpu() && state_cache.dim() == 3 &&
                         block_table.dim() == 2 && state_cache.stride(2) == 1 && kv_stack.stride(1) == 1 &&
                         score_stack.stride(1) == 1 && compress_ratio > 0 && (coff == 1 || coff == 2);

#if FUSED_CPP_STRICT_MODE
  TORCH_CHECK(supported,
              "FUSED_CPP_STRICT_MODE CompressRowsNormRopeInsert SVE copy requires fp32 CPU "
              "state_cache, int32 CPU 2-D block_table, contiguous last dims, and coff 1/2");
#else
  if (!supported) {
    return false;
  }
#endif

  const float* state_data = state_cache.data_ptr<float>();
  float* kv_data = kv_stack.data_ptr<float>();
  float* score_data = score_stack.data_ptr<float>();
  const int64_t state_stride_block = state_cache.stride(0);
  const int64_t state_stride_offset = state_cache.stride(1);
  const int64_t block_stride_req = block_table.stride(0);
  const int64_t block_stride_logical = block_table.stride(1);
  const int64_t kv_stride_row = kv_stack.stride(0);
  const int64_t score_stride_row = score_stack.stride(0);

  const auto copy_rows = [&](const auto* block_data) {
    const auto copy_window_row = [&](int64_t out_row, int64_t p, int64_t segment) {
      float* kv_dst = kv_data + out_row * kv_stride_row;
      float* score_dst = score_data + out_row * score_stride_row;
      if (p < 0) {
        FillCompressorPaddingRowSve4x(kv_dst, score_dst, head_dim);
        return;
      }
      const int64_t logical_block = p / state_block_size;
      const int64_t logical_offset = p % state_block_size;
      const int64_t block =
          static_cast<int64_t>(block_data[req_idx * block_stride_req + logical_block * block_stride_logical]);
      const float* row = state_data + block * state_stride_block + logical_offset * state_stride_offset;
      const int64_t kv_start = segment * head_dim;
      const int64_t score_start = state_width + segment * head_dim;
      CopyCompressorKvScoreRowSve4x(row + kv_start, row + score_start, kv_dst, score_dst, head_dim);
    };

    if (coff == 2) {
      if (compress_ratio != 4) {
#if FUSED_CPP_STRICT_MODE
        TORCH_CHECK(false, "FUSED_CPP_STRICT_MODE coff=2 compressor SVE copy requires compress_ratio=4, got ",
                    compress_ratio);
#else
        return false;
#endif
      }
      copy_window_row(0, start + 0, 0);
      copy_window_row(1, start + 1, 0);
      copy_window_row(2, start + 2, 0);
      copy_window_row(3, start + 3, 0);
      copy_window_row(4, start + 4, 1);
      copy_window_row(5, start + 5, 1);
      copy_window_row(6, start + 6, 1);
      copy_window_row(7, start + 7, 1);
    } else {
      for (int64_t t = 0; t < compress_ratio; ++t) {
        copy_window_row(t, start + t, 0);
      }
    }
    return true;
  };

  return copy_rows(block_table.data_ptr<int32_t>());
}

template <bool HasPadding>
void CompressCoff2WindowSveKernel(const float* const kv_src[8], const float* const score_src[8], int64_t head_dim,
                                  float* compressed_data) {
  const svfloat32_t zero = svdup_n_f32(0.0f);
  const svfloat32_t one = svdup_n_f32(1.0f);
  const svfloat32_t neg_inf = svdup_n_f32(-std::numeric_limits<float>::infinity());
  const svfloat32_t min_denom = svdup_n_f32(1.0e-20f);
  const svfloat32_t exp_hi = svdup_f32(87.0f);
  const svfloat32_t exp_lo = svdup_f32(-87.0f);
  const svfloat32_t inv_ln2 = svdup_f32(1.4426950408889634f);
  const svfloat32_t ln2 = svdup_f32(0.6931471805599453f);
  const svfloat32_t c0 = svdup_f32(1.0f);
  const svfloat32_t c1 = svdup_f32(1.0f);
  const svfloat32_t c2 = svdup_f32(0.5f);
  const svfloat32_t c3 = svdup_f32(1.0f / 6.0f);
  const svfloat32_t c4 = svdup_f32(1.0f / 24.0f);
  const svfloat32_t c5 = svdup_f32(1.0f / 120.0f);
  const svfloat32_t c6 = svdup_f32(1.0f / 720.0f);
  const int64_t vl = static_cast<int64_t>(svcntw());

  auto process_chunk = [&](int64_t d, svbool_t pg) __attribute__((always_inline)) {
    auto load_score = [&](int idx) -> svfloat32_t {
      if constexpr (HasPadding) {
        if (score_src[idx] == nullptr) {
          return neg_inf;
        }
      }
      return svld1_f32(pg, score_src[idx] + d);
    };

    const svfloat32_t s0 = load_score(0);
    const svfloat32_t s1 = load_score(1);
    const svfloat32_t s2 = load_score(2);
    const svfloat32_t s3 = load_score(3);
    const svfloat32_t s4 = load_score(4);
    const svfloat32_t s5 = load_score(5);
    const svfloat32_t s6 = load_score(6);
    const svfloat32_t s7 = load_score(7);

    const svfloat32_t m01 = svmax_f32_x(pg, s0, s1);
    const svfloat32_t m23 = svmax_f32_x(pg, s2, s3);
    const svfloat32_t m45 = svmax_f32_x(pg, s4, s5);
    const svfloat32_t m67 = svmax_f32_x(pg, s6, s7);
    const svfloat32_t m0123 = svmax_f32_x(pg, m01, m23);
    const svfloat32_t m4567 = svmax_f32_x(pg, m45, m67);
    const svfloat32_t max_score = svmax_f32_x(pg, m0123, m4567);

    svfloat32_t x0 = svsub_f32_x(pg, s0, max_score);
    svfloat32_t x1 = svsub_f32_x(pg, s1, max_score);
    svfloat32_t x2 = svsub_f32_x(pg, s2, max_score);
    svfloat32_t x3 = svsub_f32_x(pg, s3, max_score);
    x0 = svmin_f32_x(pg, x0, exp_hi);
    x1 = svmin_f32_x(pg, x1, exp_hi);
    x2 = svmin_f32_x(pg, x2, exp_hi);
    x3 = svmin_f32_x(pg, x3, exp_hi);
    x0 = svmax_f32_x(pg, x0, exp_lo);
    x1 = svmax_f32_x(pg, x1, exp_lo);
    x2 = svmax_f32_x(pg, x2, exp_lo);
    x3 = svmax_f32_x(pg, x3, exp_lo);

    const svfloat32_t y0 = svmul_f32_x(pg, x0, inv_ln2);
    const svfloat32_t y1 = svmul_f32_x(pg, x1, inv_ln2);
    const svfloat32_t y2 = svmul_f32_x(pg, x2, inv_ln2);
    const svfloat32_t y3 = svmul_f32_x(pg, x3, inv_ln2);
    const svfloat32_t fn0 = svrinta_f32_x(pg, y0);
    const svfloat32_t fn1 = svrinta_f32_x(pg, y1);
    const svfloat32_t fn2 = svrinta_f32_x(pg, y2);
    const svfloat32_t fn3 = svrinta_f32_x(pg, y3);
    const svfloat32_t r0 = svmls_f32_x(pg, x0, fn0, ln2);
    const svfloat32_t r1 = svmls_f32_x(pg, x1, fn1, ln2);
    const svfloat32_t r2 = svmls_f32_x(pg, x2, fn2, ln2);
    const svfloat32_t r3 = svmls_f32_x(pg, x3, fn3, ln2);
    const svint32_t n0 = svcvt_s32_f32_x(pg, fn0);
    const svint32_t n1 = svcvt_s32_f32_x(pg, fn1);
    const svint32_t n2 = svcvt_s32_f32_x(pg, fn2);
    const svint32_t n3 = svcvt_s32_f32_x(pg, fn3);

    svfloat32_t p0 = c6;
    svfloat32_t p1 = c6;
    svfloat32_t p2 = c6;
    svfloat32_t p3 = c6;
    p0 = svmla_f32_x(pg, c5, p0, r0);
    p1 = svmla_f32_x(pg, c5, p1, r1);
    p2 = svmla_f32_x(pg, c5, p2, r2);
    p3 = svmla_f32_x(pg, c5, p3, r3);
    p0 = svmla_f32_x(pg, c4, p0, r0);
    p1 = svmla_f32_x(pg, c4, p1, r1);
    p2 = svmla_f32_x(pg, c4, p2, r2);
    p3 = svmla_f32_x(pg, c4, p3, r3);
    p0 = svmla_f32_x(pg, c3, p0, r0);
    p1 = svmla_f32_x(pg, c3, p1, r1);
    p2 = svmla_f32_x(pg, c3, p2, r2);
    p3 = svmla_f32_x(pg, c3, p3, r3);
    p0 = svmla_f32_x(pg, c2, p0, r0);
    p1 = svmla_f32_x(pg, c2, p1, r1);
    p2 = svmla_f32_x(pg, c2, p2, r2);
    p3 = svmla_f32_x(pg, c2, p3, r3);
    p0 = svmla_f32_x(pg, c1, p0, r0);
    p1 = svmla_f32_x(pg, c1, p1, r1);
    p2 = svmla_f32_x(pg, c1, p2, r2);
    p3 = svmla_f32_x(pg, c1, p3, r3);
    p0 = svmla_f32_x(pg, c0, p0, r0);
    p1 = svmla_f32_x(pg, c0, p1, r1);
    p2 = svmla_f32_x(pg, c0, p2, r2);
    p3 = svmla_f32_x(pg, c0, p3, r3);

    auto scale_exp = [&](int idx, svfloat32_t p, svint32_t n) -> svfloat32_t {
      if constexpr (HasPadding) {
        if (score_src[idx] == nullptr) {
          return zero;
        }
      }
      return svscale_f32_x(pg, p, n);
    };

    const svfloat32_t e0 = scale_exp(0, p0, n0);
    const svfloat32_t e1 = scale_exp(1, p1, n1);
    const svfloat32_t e2 = scale_exp(2, p2, n2);
    const svfloat32_t e3 = scale_exp(3, p3, n3);

    svfloat32_t x4 = svsub_f32_x(pg, s4, max_score);
    svfloat32_t x5 = svsub_f32_x(pg, s5, max_score);
    svfloat32_t x6 = svsub_f32_x(pg, s6, max_score);
    svfloat32_t x7 = svsub_f32_x(pg, s7, max_score);
    x4 = svmin_f32_x(pg, x4, exp_hi);
    x5 = svmin_f32_x(pg, x5, exp_hi);
    x6 = svmin_f32_x(pg, x6, exp_hi);
    x7 = svmin_f32_x(pg, x7, exp_hi);
    x4 = svmax_f32_x(pg, x4, exp_lo);
    x5 = svmax_f32_x(pg, x5, exp_lo);
    x6 = svmax_f32_x(pg, x6, exp_lo);
    x7 = svmax_f32_x(pg, x7, exp_lo);

    const svfloat32_t y4 = svmul_f32_x(pg, x4, inv_ln2);
    const svfloat32_t y5 = svmul_f32_x(pg, x5, inv_ln2);
    const svfloat32_t y6 = svmul_f32_x(pg, x6, inv_ln2);
    const svfloat32_t y7 = svmul_f32_x(pg, x7, inv_ln2);
    const svfloat32_t fn4 = svrinta_f32_x(pg, y4);
    const svfloat32_t fn5 = svrinta_f32_x(pg, y5);
    const svfloat32_t fn6 = svrinta_f32_x(pg, y6);
    const svfloat32_t fn7 = svrinta_f32_x(pg, y7);
    const svfloat32_t r4 = svmls_f32_x(pg, x4, fn4, ln2);
    const svfloat32_t r5 = svmls_f32_x(pg, x5, fn5, ln2);
    const svfloat32_t r6 = svmls_f32_x(pg, x6, fn6, ln2);
    const svfloat32_t r7 = svmls_f32_x(pg, x7, fn7, ln2);
    const svint32_t n4 = svcvt_s32_f32_x(pg, fn4);
    const svint32_t n5 = svcvt_s32_f32_x(pg, fn5);
    const svint32_t n6 = svcvt_s32_f32_x(pg, fn6);
    const svint32_t n7 = svcvt_s32_f32_x(pg, fn7);

    svfloat32_t p4 = c6;
    svfloat32_t p5 = c6;
    svfloat32_t p6 = c6;
    svfloat32_t p7 = c6;
    p4 = svmla_f32_x(pg, c5, p4, r4);
    p5 = svmla_f32_x(pg, c5, p5, r5);
    p6 = svmla_f32_x(pg, c5, p6, r6);
    p7 = svmla_f32_x(pg, c5, p7, r7);
    p4 = svmla_f32_x(pg, c4, p4, r4);
    p5 = svmla_f32_x(pg, c4, p5, r5);
    p6 = svmla_f32_x(pg, c4, p6, r6);
    p7 = svmla_f32_x(pg, c4, p7, r7);
    p4 = svmla_f32_x(pg, c3, p4, r4);
    p5 = svmla_f32_x(pg, c3, p5, r5);
    p6 = svmla_f32_x(pg, c3, p6, r6);
    p7 = svmla_f32_x(pg, c3, p7, r7);
    p4 = svmla_f32_x(pg, c2, p4, r4);
    p5 = svmla_f32_x(pg, c2, p5, r5);
    p6 = svmla_f32_x(pg, c2, p6, r6);
    p7 = svmla_f32_x(pg, c2, p7, r7);
    p4 = svmla_f32_x(pg, c1, p4, r4);
    p5 = svmla_f32_x(pg, c1, p5, r5);
    p6 = svmla_f32_x(pg, c1, p6, r6);
    p7 = svmla_f32_x(pg, c1, p7, r7);
    p4 = svmla_f32_x(pg, c0, p4, r4);
    p5 = svmla_f32_x(pg, c0, p5, r5);
    p6 = svmla_f32_x(pg, c0, p6, r6);
    p7 = svmla_f32_x(pg, c0, p7, r7);

    const svfloat32_t e4 = scale_exp(4, p4, n4);
    const svfloat32_t e5 = scale_exp(5, p5, n5);
    const svfloat32_t e6 = scale_exp(6, p6, n6);
    const svfloat32_t e7 = scale_exp(7, p7, n7);

    const svfloat32_t d01 = svadd_f32_x(pg, e0, e1);
    const svfloat32_t d23 = svadd_f32_x(pg, e2, e3);
    const svfloat32_t d45 = svadd_f32_x(pg, e4, e5);
    const svfloat32_t d67 = svadd_f32_x(pg, e6, e7);
    const svfloat32_t d0123 = svadd_f32_x(pg, d01, d23);
    const svfloat32_t d4567 = svadd_f32_x(pg, d45, d67);
    const svfloat32_t denom = svadd_f32_x(pg, d0123, d4567);
    const svfloat32_t inv_denom = svdiv_f32_x(pg, one, svmax_f32_x(pg, denom, min_denom));

    auto accumulate = [&](svfloat32_t acc, int idx, svfloat32_t e) -> svfloat32_t {
      if constexpr (HasPadding) {
        if (kv_src[idx] == nullptr) {
          return acc;
        }
      }
      return svmla_f32_x(pg, acc, svld1_f32(pg, kv_src[idx] + d), svmul_f32_x(pg, e, inv_denom));
    };

    svfloat32_t acc = zero;
    acc = accumulate(acc, 0, e0);
    acc = accumulate(acc, 1, e1);
    acc = accumulate(acc, 2, e2);
    acc = accumulate(acc, 3, e3);
    acc = accumulate(acc, 4, e4);
    acc = accumulate(acc, 5, e5);
    acc = accumulate(acc, 6, e6);
    acc = accumulate(acc, 7, e7);
    svst1_f32(pg, compressed_data + d, acc);
  };

  const int64_t full_head_dim = (head_dim / vl) * vl;
  const svbool_t pg_all = svptrue_b32();
  int64_t d = 0;
  for (; d < full_head_dim; d += vl) {
    process_chunk(d, pg_all);
  }
  if (d < head_dim) {
    process_chunk(d, svwhilelt_b32(d, head_dim));
  }
}

void CompressCoff1WindowOnlineSveKernel(const float* const kv_src[128], const float* const score_src[128],
                                        int64_t head_dim, float* compressed_data) {
  const svfloat32_t zero = svdup_n_f32(0.0f);
  const svfloat32_t neg_inf = svdup_n_f32(-std::numeric_limits<float>::infinity());
  const svfloat32_t min_denom = svdup_n_f32(1.0e-20f);
  const svfloat32_t exp_hi = svdup_f32(87.0f);
  const svfloat32_t exp_lo = svdup_f32(-87.0f);
  const svfloat32_t inv_ln2 = svdup_f32(1.4426950408889634f);
  const svfloat32_t ln2 = svdup_f32(0.6931471805599453f);
  const svfloat32_t c0 = svdup_f32(1.0f);
  const svfloat32_t c1 = svdup_f32(1.0f);
  const svfloat32_t c2 = svdup_f32(0.5f);
  const svfloat32_t c3 = svdup_f32(1.0f / 6.0f);
  const svfloat32_t c4 = svdup_f32(1.0f / 24.0f);
  const svfloat32_t c5 = svdup_f32(1.0f / 120.0f);
  const svfloat32_t c6 = svdup_f32(1.0f / 720.0f);
  const int64_t vl = static_cast<int64_t>(svcntw());

  auto process_chunk = [&](int64_t d, svbool_t pg) __attribute__((always_inline)) {
    svfloat32_t global_max = neg_inf;
    svfloat32_t global_denom = zero;
    svfloat32_t global_acc = zero;

    for (int64_t row = 0; row < 128; row += 4) {
      const svfloat32_t s0 = svld1_f32(pg, score_src[row + 0] + d);
      const svfloat32_t s1 = svld1_f32(pg, score_src[row + 1] + d);
      const svfloat32_t s2 = svld1_f32(pg, score_src[row + 2] + d);
      const svfloat32_t s3 = svld1_f32(pg, score_src[row + 3] + d);

      const svfloat32_t m01 = svmax_f32_x(pg, s0, s1);
      const svfloat32_t m23 = svmax_f32_x(pg, s2, s3);
      const svfloat32_t block_max = svmax_f32_x(pg, m01, m23);

      svfloat32_t x0 = svsub_f32_x(pg, s0, block_max);
      svfloat32_t x1 = svsub_f32_x(pg, s1, block_max);
      svfloat32_t x2 = svsub_f32_x(pg, s2, block_max);
      svfloat32_t x3 = svsub_f32_x(pg, s3, block_max);
      x0 = svmin_f32_x(pg, x0, exp_hi);
      x1 = svmin_f32_x(pg, x1, exp_hi);
      x2 = svmin_f32_x(pg, x2, exp_hi);
      x3 = svmin_f32_x(pg, x3, exp_hi);
      x0 = svmax_f32_x(pg, x0, exp_lo);
      x1 = svmax_f32_x(pg, x1, exp_lo);
      x2 = svmax_f32_x(pg, x2, exp_lo);
      x3 = svmax_f32_x(pg, x3, exp_lo);

      const svfloat32_t y0 = svmul_f32_x(pg, x0, inv_ln2);
      const svfloat32_t y1 = svmul_f32_x(pg, x1, inv_ln2);
      const svfloat32_t y2 = svmul_f32_x(pg, x2, inv_ln2);
      const svfloat32_t y3 = svmul_f32_x(pg, x3, inv_ln2);
      const svfloat32_t fn0 = svrinta_f32_x(pg, y0);
      const svfloat32_t fn1 = svrinta_f32_x(pg, y1);
      const svfloat32_t fn2 = svrinta_f32_x(pg, y2);
      const svfloat32_t fn3 = svrinta_f32_x(pg, y3);
      const svfloat32_t r0 = svmls_f32_x(pg, x0, fn0, ln2);
      const svfloat32_t r1 = svmls_f32_x(pg, x1, fn1, ln2);
      const svfloat32_t r2 = svmls_f32_x(pg, x2, fn2, ln2);
      const svfloat32_t r3 = svmls_f32_x(pg, x3, fn3, ln2);
      const svint32_t n0 = svcvt_s32_f32_x(pg, fn0);
      const svint32_t n1 = svcvt_s32_f32_x(pg, fn1);
      const svint32_t n2 = svcvt_s32_f32_x(pg, fn2);
      const svint32_t n3 = svcvt_s32_f32_x(pg, fn3);

      svfloat32_t p0 = c6;
      svfloat32_t p1 = c6;
      svfloat32_t p2 = c6;
      svfloat32_t p3 = c6;
      p0 = svmla_f32_x(pg, c5, p0, r0);
      p1 = svmla_f32_x(pg, c5, p1, r1);
      p2 = svmla_f32_x(pg, c5, p2, r2);
      p3 = svmla_f32_x(pg, c5, p3, r3);
      p0 = svmla_f32_x(pg, c4, p0, r0);
      p1 = svmla_f32_x(pg, c4, p1, r1);
      p2 = svmla_f32_x(pg, c4, p2, r2);
      p3 = svmla_f32_x(pg, c4, p3, r3);
      p0 = svmla_f32_x(pg, c3, p0, r0);
      p1 = svmla_f32_x(pg, c3, p1, r1);
      p2 = svmla_f32_x(pg, c3, p2, r2);
      p3 = svmla_f32_x(pg, c3, p3, r3);
      p0 = svmla_f32_x(pg, c2, p0, r0);
      p1 = svmla_f32_x(pg, c2, p1, r1);
      p2 = svmla_f32_x(pg, c2, p2, r2);
      p3 = svmla_f32_x(pg, c2, p3, r3);
      p0 = svmla_f32_x(pg, c1, p0, r0);
      p1 = svmla_f32_x(pg, c1, p1, r1);
      p2 = svmla_f32_x(pg, c1, p2, r2);
      p3 = svmla_f32_x(pg, c1, p3, r3);
      p0 = svmla_f32_x(pg, c0, p0, r0);
      p1 = svmla_f32_x(pg, c0, p1, r1);
      p2 = svmla_f32_x(pg, c0, p2, r2);
      p3 = svmla_f32_x(pg, c0, p3, r3);

      const svfloat32_t e0 = svscale_f32_x(pg, p0, n0);
      const svfloat32_t e1 = svscale_f32_x(pg, p1, n1);
      const svfloat32_t e2 = svscale_f32_x(pg, p2, n2);
      const svfloat32_t e3 = svscale_f32_x(pg, p3, n3);

      const svfloat32_t d01 = svadd_f32_x(pg, e0, e1);
      const svfloat32_t d23 = svadd_f32_x(pg, e2, e3);
      const svfloat32_t block_denom = svadd_f32_x(pg, d01, d23);

      svfloat32_t block_acc = zero;
      block_acc = svmla_f32_x(pg, block_acc, svld1_f32(pg, kv_src[row + 0] + d), e0);
      block_acc = svmla_f32_x(pg, block_acc, svld1_f32(pg, kv_src[row + 1] + d), e1);
      block_acc = svmla_f32_x(pg, block_acc, svld1_f32(pg, kv_src[row + 2] + d), e2);
      block_acc = svmla_f32_x(pg, block_acc, svld1_f32(pg, kv_src[row + 3] + d), e3);

      const svfloat32_t new_max = svmax_f32_x(pg, global_max, block_max);
      const svfloat32_t old_scale = ExpApproxF32Sve(pg, svsub_f32_x(pg, global_max, new_max), exp_hi, exp_lo, inv_ln2,
                                                    ln2, c0, c1, c2, c3, c4, c5, c6);
      const svfloat32_t block_scale = ExpApproxF32Sve(pg, svsub_f32_x(pg, block_max, new_max), exp_hi, exp_lo, inv_ln2,
                                                      ln2, c0, c1, c2, c3, c4, c5, c6);

      global_denom = svmla_f32_x(pg, svmul_f32_x(pg, global_denom, old_scale), block_denom, block_scale);
      global_acc = svmla_f32_x(pg, svmul_f32_x(pg, global_acc, old_scale), block_acc, block_scale);
      global_max = new_max;
    }

    const svfloat32_t compressed = svdiv_f32_x(pg, global_acc, svmax_f32_x(pg, global_denom, min_denom));
    svst1_f32(pg, compressed_data + d, compressed);
  };

  const int64_t full_head_dim = (head_dim / vl) * vl;
  const svbool_t pg_all = svptrue_b32();
  int64_t d = 0;
  for (; d < full_head_dim; d += vl) {
    process_chunk(d, pg_all);
  }
  if (d < head_dim) {
    process_chunk(d, svwhilelt_b32(d, head_dim));
  }
}

bool TryCompressCoff1WindowOnlineSve(const at::Tensor& state_cache, const at::Tensor& block_table, int64_t req_idx,
                                     int64_t start, int64_t compress_ratio, int64_t state_block_size,
                                     int64_t state_width, int64_t head_dim, at::Tensor& compressed) {
  const bool supported =
      state_cache.scalar_type() == at::kFloat && IsInt32Tensor(block_table) && compressed.scalar_type() == at::kFloat &&
      state_cache.device().is_cpu() && block_table.device().is_cpu() && state_cache.dim() == 3 &&
      block_table.dim() == 2 && state_cache.stride(2) == 1 && compressed.dim() == 1 && compressed.size(0) == head_dim &&
      compressed.stride(0) == 1 && compress_ratio == 128 && state_width == head_dim && start >= 0 && head_dim > 0;

  if (!supported) {
    return false;
  }

  const float* state_data = state_cache.data_ptr<float>();
  float* compressed_data = compressed.data_ptr<float>();
  const int64_t state_stride_block = state_cache.stride(0);
  const int64_t state_stride_offset = state_cache.stride(1);
  const int64_t block_stride_req = block_table.stride(0);
  const int64_t block_stride_logical = block_table.stride(1);

  const auto run = [&](const auto* block_data) {
    std::array<const float*, 128> kv_src{};
    std::array<const float*, 128> score_src{};
    for (int64_t row_idx = 0; row_idx < 128; ++row_idx) {
      const int64_t p = start + row_idx;
      const int64_t logical_block = p / state_block_size;
      const int64_t logical_offset = p % state_block_size;
      const int64_t block =
          static_cast<int64_t>(block_data[req_idx * block_stride_req + logical_block * block_stride_logical]);
      const float* row = state_data + block * state_stride_block + logical_offset * state_stride_offset;
      kv_src[row_idx] = row;
      score_src[row_idx] = row + state_width;
    }

    CompressCoff1WindowOnlineSveKernel(kv_src.data(), score_src.data(), head_dim, compressed_data);
    return true;
  };
  return run(block_table.data_ptr<int32_t>());
}

bool TryCompressCoff2WindowSve(const at::Tensor& state_cache, const at::Tensor& block_table, int64_t req_idx,
                               int64_t start, int64_t compress_ratio, int64_t state_block_size, int64_t state_width,
                               int64_t head_dim, at::Tensor& compressed) {
  const bool supported =
      state_cache.scalar_type() == at::kFloat && IsInt32Tensor(block_table) && compressed.scalar_type() == at::kFloat &&
      state_cache.device().is_cpu() && block_table.device().is_cpu() && state_cache.dim() == 3 &&
      block_table.dim() == 2 && state_cache.stride(2) == 1 && compressed.dim() == 1 && compressed.size(0) == head_dim &&
      compressed.stride(0) == 1 && compress_ratio == 4 && state_width == 2 * head_dim && head_dim > 0;

#if FUSED_CPP_STRICT_MODE
  TORCH_CHECK(supported,
              "FUSED_CPP_STRICT_MODE coff=2 compressor fused SVE path requires fp32 CPU "
              "state_cache, int32 CPU 2-D block_table, contiguous compressed output, "
              "compress_ratio=4, and state_width=2*head_dim");
#else
  if (!supported) {
    return false;
  }
#endif

  const float* state_data = state_cache.data_ptr<float>();
  float* compressed_data = compressed.data_ptr<float>();
  const int64_t state_stride_block = state_cache.stride(0);
  const int64_t state_stride_offset = state_cache.stride(1);
  const int64_t block_stride_req = block_table.stride(0);
  const int64_t block_stride_logical = block_table.stride(1);

  const auto run = [&](const auto* block_data) {
    const float* kv_src[8] = {};
    const float* score_src[8] = {};
    for (int64_t row_idx = 0; row_idx < 8; ++row_idx) {
      const int64_t p = start + row_idx;
      if (p < 0) {
        continue;
      }
      const int64_t logical_block = p / state_block_size;
      const int64_t logical_offset = p % state_block_size;
      const int64_t block =
          static_cast<int64_t>(block_data[req_idx * block_stride_req + logical_block * block_stride_logical]);
      const float* row = state_data + block * state_stride_block + logical_offset * state_stride_offset;
      const int64_t segment = row_idx >= 4 ? 1 : 0;
      kv_src[row_idx] = row + segment * head_dim;
      score_src[row_idx] = row + state_width + segment * head_dim;
    }

    if (start < 0) {
      CompressCoff2WindowSveKernel<true>(kv_src, score_src, head_dim, compressed_data);
    } else {
      CompressCoff2WindowSveKernel<false>(kv_src, score_src, head_dim, compressed_data);
    }
    return true;
  };
  return run(block_table.data_ptr<int32_t>());
}

float SumSqCompressorF32Sve(const float* row, int64_t head_dim) {
  const int64_t vl = static_cast<int64_t>(svcntw());
  const int64_t step = 4 * vl;
  const svbool_t pg_all = svptrue_b32();
  svfloat32_t acc0 = svdup_f32(0.0f);
  svfloat32_t acc1 = svdup_f32(0.0f);
  svfloat32_t acc2 = svdup_f32(0.0f);
  svfloat32_t acc3 = svdup_f32(0.0f);

  int64_t d = 0;
  for (; d + step <= head_dim; d += step) {
    const svfloat32_t v0 = svld1_f32(pg_all, row + d);
    const svfloat32_t v1 = svld1_f32(pg_all, row + d + vl);
    const svfloat32_t v2 = svld1_f32(pg_all, row + d + 2 * vl);
    const svfloat32_t v3 = svld1_f32(pg_all, row + d + 3 * vl);
    acc0 = svmla_f32_x(pg_all, acc0, v0, v0);
    acc1 = svmla_f32_x(pg_all, acc1, v1, v1);
    acc2 = svmla_f32_x(pg_all, acc2, v2, v2);
    acc3 = svmla_f32_x(pg_all, acc3, v3, v3);
  }
  for (; d < head_dim; d += vl) {
    const svbool_t pg = svwhilelt_b32(d, head_dim);
    const svfloat32_t v = svld1_f32(pg, row + d);
    acc0 = svmla_f32_m(pg, acc0, v, v);
  }

  acc0 = svadd_f32_x(pg_all, acc0, acc1);
  acc2 = svadd_f32_x(pg_all, acc2, acc3);
  return svaddv_f32(pg_all, svadd_f32_x(pg_all, acc0, acc2));
}

template <typename cache_t>
void NormRopeStoreCompressedSveKernel(const float* compressed, const float* rms_weight, double rms_norm_eps,
                                      const float* cos_row, const float* sin_row, int64_t head_dim,
                                      int64_t rope_head_dim, cache_t* cache_row) {
  const int64_t nope_dim = head_dim - rope_head_dim;
  const int64_t rope_half = rope_head_dim / 2;
  const int64_t vl = static_cast<int64_t>(svcntw());
  const int64_t step = 4 * vl;
  const svbool_t pg_all = svptrue_b32();
  const float sum_sq = SumSqCompressorF32Sve(compressed, head_dim);
  const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(head_dim) + static_cast<float>(rms_norm_eps));
  const svfloat32_t inv = svdup_f32(inv_rms);

  int64_t d = 0;
  for (; d + step <= nope_dim; d += step) {
    const svfloat32_t v0 = svld1_f32(pg_all, compressed + d);
    const svfloat32_t v1 = svld1_f32(pg_all, compressed + d + vl);
    const svfloat32_t v2 = svld1_f32(pg_all, compressed + d + 2 * vl);
    const svfloat32_t v3 = svld1_f32(pg_all, compressed + d + 3 * vl);
    const svfloat32_t w0 = svld1_f32(pg_all, rms_weight + d);
    const svfloat32_t w1 = svld1_f32(pg_all, rms_weight + d + vl);
    const svfloat32_t w2 = svld1_f32(pg_all, rms_weight + d + 2 * vl);
    const svfloat32_t w3 = svld1_f32(pg_all, rms_weight + d + 3 * vl);
    const svfloat32_t o0 = svmul_f32_x(pg_all, svmul_f32_x(pg_all, v0, inv), w0);
    const svfloat32_t o1 = svmul_f32_x(pg_all, svmul_f32_x(pg_all, v1, inv), w1);
    const svfloat32_t o2 = svmul_f32_x(pg_all, svmul_f32_x(pg_all, v2, inv), w2);
    const svfloat32_t o3 = svmul_f32_x(pg_all, svmul_f32_x(pg_all, v3, inv), w3);
    StoreCompressorCacheF32Sve(pg_all, cache_row + d, o0);
    StoreCompressorCacheF32Sve(pg_all, cache_row + d + vl, o1);
    StoreCompressorCacheF32Sve(pg_all, cache_row + d + 2 * vl, o2);
    StoreCompressorCacheF32Sve(pg_all, cache_row + d + 3 * vl, o3);
  }
  for (; d < nope_dim; d += vl) {
    const svbool_t pg = svwhilelt_b32(d, nope_dim);
    const svfloat32_t v = svld1_f32(pg, compressed + d);
    const svfloat32_t w = svld1_f32(pg, rms_weight + d);
    StoreCompressorCacheF32Sve(pg, cache_row + d, svmul_f32_x(pg, svmul_f32_x(pg, v, inv), w));
  }

  int64_t pair = 0;
  for (; pair + vl <= rope_half; pair += vl) {
    const int64_t offset = nope_dim + 2 * pair;
    const svfloat32x2_t src = svld2_f32(pg_all, compressed + offset);
    const svfloat32x2_t weight = svld2_f32(pg_all, rms_weight + offset);
    const svfloat32_t c = svld1_f32(pg_all, cos_row + pair);
    const svfloat32_t s = svld1_f32(pg_all, sin_row + pair);
    const svfloat32_t even = svmul_f32_x(pg_all, svmul_f32_x(pg_all, svget2_f32(src, 0), inv), svget2_f32(weight, 0));
    const svfloat32_t odd = svmul_f32_x(pg_all, svmul_f32_x(pg_all, svget2_f32(src, 1), inv), svget2_f32(weight, 1));
    const svfloat32_t even_c = svmul_f32_x(pg_all, even, c);
    const svfloat32_t odd_c = svmul_f32_x(pg_all, odd, c);
    const svfloat32_t out_even = svmls_f32_x(pg_all, even_c, odd, s);
    const svfloat32_t out_odd = svmla_f32_x(pg_all, odd_c, even, s);
    StoreCompressorCacheEvenOddF32Sve(pg_all, cache_row + offset, out_even, out_odd);
  }
  if (pair < rope_half) {
    const svbool_t pg = svwhilelt_b32(pair, rope_half);
    const int64_t offset = nope_dim + 2 * pair;
    const svfloat32x2_t src = svld2_f32(pg, compressed + offset);
    const svfloat32x2_t weight = svld2_f32(pg, rms_weight + offset);
    const svfloat32_t c = svld1_f32(pg, cos_row + pair);
    const svfloat32_t s = svld1_f32(pg, sin_row + pair);
    const svfloat32_t even = svmul_f32_x(pg, svmul_f32_x(pg, svget2_f32(src, 0), inv), svget2_f32(weight, 0));
    const svfloat32_t odd = svmul_f32_x(pg, svmul_f32_x(pg, svget2_f32(src, 1), inv), svget2_f32(weight, 1));
    const svfloat32_t even_c = svmul_f32_x(pg, even, c);
    const svfloat32_t odd_c = svmul_f32_x(pg, odd, c);
    const svfloat32_t out_even = svmls_f32_x(pg, even_c, odd, s);
    const svfloat32_t out_odd = svmla_f32_x(pg, odd_c, even, s);
    StoreCompressorCacheEvenOddF32Sve(pg, cache_row + offset, out_even, out_odd);
  }
}

bool TryNormRopeStoreCompressedSve(const at::Tensor& compressed, const at::Tensor& rms_weight, double rms_norm_eps,
                                   const at::Tensor& cos_sin_cache, int64_t compressed_pos, int64_t rope_head_dim,
                                   const at::Tensor& kv_cache, int64_t kv_slot, int64_t kv_cache_block_size) {
  const int64_t head_dim = compressed.size(0);
  const bool supported =
      compressed.scalar_type() == at::kFloat && rms_weight.scalar_type() == at::kFloat &&
      cos_sin_cache.scalar_type() == at::kFloat &&
      (kv_cache.scalar_type() == at::kBFloat16 || kv_cache.scalar_type() == at::kFloat) &&
      compressed.device().is_cpu() && rms_weight.device().is_cpu() && cos_sin_cache.device().is_cpu() &&
      kv_cache.device().is_cpu() && compressed.dim() == 1 && rms_weight.dim() == 1 && cos_sin_cache.dim() == 2 &&
      kv_cache.dim() == 3 && rms_weight.size(0) == head_dim && head_dim > 0 && rope_head_dim >= 0 &&
      rope_head_dim <= head_dim && rope_head_dim % 2 == 0 && cos_sin_cache.size(1) >= rope_head_dim &&
      compressed_pos >= 0 && compressed_pos < cos_sin_cache.size(0) && kv_slot >= 0 && kv_cache_block_size > 0 &&
      compressed.stride(0) == 1 && rms_weight.stride(0) == 1 && cos_sin_cache.stride(1) == 1 &&
      kv_cache.stride(2) == 1 && kv_cache.size(2) >= head_dim;

#if FUSED_CPP_STRICT_MODE
  TORCH_CHECK(supported,
              "FUSED_CPP_STRICT_MODE compressor NormRopeStore SVE path requires fp32 contiguous "
              "compressed/rms/cos_sin, bf16/fp32 CPU kv_cache with contiguous last dim, "
              "valid compressed position, and even rope_head_dim");
#else
  if (!supported) {
    return false;
  }
#endif

  const int64_t kv_block = kv_slot / kv_cache_block_size;
  const int64_t kv_offset = kv_slot % kv_cache_block_size;
  TORCH_CHECK(kv_block >= 0 && kv_block < kv_cache.size(0) && kv_offset >= 0 && kv_offset < kv_cache.size(1),
              "compressor kv_slot out of kv_cache range: slot=", kv_slot, ", block=", kv_block, ", offset=", kv_offset);

  const float* compressed_data = compressed.data_ptr<float>();
  const float* rms_data = rms_weight.data_ptr<float>();
  const float* cs_row = cos_sin_cache.data_ptr<float>() + compressed_pos * cos_sin_cache.stride(0);
  const float* cos_row = cs_row;
  const float* sin_row = cs_row + rope_head_dim / 2;
  const int64_t cache_offset = kv_block * kv_cache.stride(0) + kv_offset * kv_cache.stride(1);

  if (kv_cache.scalar_type() == at::kFloat) {
    float* cache_row = kv_cache.data_ptr<float>() + cache_offset;
    NormRopeStoreCompressedSveKernel<float>(compressed_data, rms_data, rms_norm_eps, cos_row, sin_row, head_dim,
                                            rope_head_dim, cache_row);
  } else {
    uint16_t* cache_row = reinterpret_cast<uint16_t*>(kv_cache.data_ptr<at::BFloat16>()) + cache_offset;
    NormRopeStoreCompressedSveKernel<uint16_t>(compressed_data, rms_data, rms_norm_eps, cos_row, sin_row, head_dim,
                                               rope_head_dim, cache_row);
  }
  return true;
}

bool TryCompressRowsNormRopeInsertCoff2FusedSve(const at::Tensor& state_cache, const at::Tensor& block_table,
                                                int64_t req_idx, int64_t start, int64_t compress_ratio,
                                                int64_t state_block_size, int64_t state_width, int64_t head_dim,
                                                const at::Tensor& rms_weight, double rms_norm_eps,
                                                const at::Tensor& cos_sin_cache, int64_t compressed_pos,
                                                int64_t rope_head_dim, const at::Tensor& kv_cache, int64_t kv_slot,
                                                int64_t kv_cache_block_size) {
  at::Tensor compressed = at::empty({head_dim}, state_cache.options().dtype(at::kFloat));
  if (!TryCompressCoff2WindowSve(state_cache, block_table, req_idx, start, compress_ratio, state_block_size,
                                 state_width, head_dim, compressed)) {
    return false;
  }

  if (TryNormRopeStoreCompressedSve(compressed, rms_weight, rms_norm_eps, cos_sin_cache, compressed_pos, rope_head_dim,
                                    kv_cache, kv_slot, kv_cache_block_size)) {
    return true;
  }
  NormRopeStoreCompressedAten(compressed, rms_weight, rms_norm_eps, cos_sin_cache, compressed_pos, rope_head_dim,
                              kv_cache, kv_slot, kv_cache_block_size);
  return true;
}

bool TryCompressRowsNormRopeInsertCoff1OnlineSve(const at::Tensor& state_cache, const at::Tensor& block_table,
                                                 int64_t req_idx, int64_t start, int64_t compress_ratio,
                                                 int64_t state_block_size, int64_t state_width, int64_t head_dim,
                                                 const at::Tensor& rms_weight, double rms_norm_eps,
                                                 const at::Tensor& cos_sin_cache, int64_t compressed_pos,
                                                 int64_t rope_head_dim, const at::Tensor& kv_cache, int64_t kv_slot,
                                                 int64_t kv_cache_block_size) {
  at::Tensor compressed = at::empty({head_dim}, state_cache.options().dtype(at::kFloat));
  if (!TryCompressCoff1WindowOnlineSve(state_cache, block_table, req_idx, start, compress_ratio, state_block_size,
                                       state_width, head_dim, compressed)) {
    return false;
  }

  if (TryNormRopeStoreCompressedSve(compressed, rms_weight, rms_norm_eps, cos_sin_cache, compressed_pos, rope_head_dim,
                                    kv_cache, kv_slot, kv_cache_block_size)) {
    return true;
  }
  NormRopeStoreCompressedAten(compressed, rms_weight, rms_norm_eps, cos_sin_cache, compressed_pos, rope_head_dim,
                              kv_cache, kv_slot, kv_cache_block_size);
  return true;
}
#endif

void CompressRowsNormRopeInsertCoff1Compute(const at::Tensor& kv_stack, const at::Tensor& score_stack,
                                            const at::Tensor& rms_weight, double rms_norm_eps,
                                            const at::Tensor& cos_sin_cache, int64_t compressed_pos,
                                            int64_t rope_head_dim, const at::Tensor& kv_cache, int64_t kv_slot,
                                            int64_t kv_cache_block_size) {
  TORCH_CHECK(kv_stack.size(0) == score_stack.size(0), "coff=1 compressor kv/score window mismatch");
  at::Tensor score_for_softmax = score_stack;
#if !FUSED_CPP_STRICT_MODE
  at::Tensor all_neg_inf = score_for_softmax.eq(-std::numeric_limits<float>::infinity()).all(0, true);
  if (all_neg_inf.any().item<bool>()) {
    score_for_softmax =
        at::where(all_neg_inf.expand_as(score_for_softmax), at::zeros_like(score_for_softmax), score_for_softmax);
  }
#endif
  at::Tensor weights = at::softmax(score_for_softmax, 0);
  at::Tensor compressed = (kv_stack * weights).sum(0);
  NormRopeStoreCompressedAten(compressed, rms_weight, rms_norm_eps, cos_sin_cache, compressed_pos, rope_head_dim,
                              kv_cache, kv_slot, kv_cache_block_size);
}

void CompressRowsNormRopeInsertCoff2Compute(const at::Tensor& kv_stack, const at::Tensor& score_stack,
                                            const at::Tensor& rms_weight, double rms_norm_eps,
                                            const at::Tensor& cos_sin_cache, int64_t compressed_pos,
                                            int64_t rope_head_dim, const at::Tensor& kv_cache, int64_t kv_slot,
                                            int64_t kv_cache_block_size) {
  TORCH_CHECK(kv_stack.size(0) == 8 && score_stack.size(0) == 8,
              "coff=2 compressor compute expects window=8, got kv/score windows ", kv_stack.size(0), "/",
              score_stack.size(0));
  at::Tensor score_for_softmax = score_stack;
#if !FUSED_CPP_STRICT_MODE
  at::Tensor all_neg_inf = score_for_softmax.eq(-std::numeric_limits<float>::infinity()).all(0, true);
  if (all_neg_inf.any().item<bool>()) {
    score_for_softmax =
        at::where(all_neg_inf.expand_as(score_for_softmax), at::zeros_like(score_for_softmax), score_for_softmax);
  }
#endif
  at::Tensor weights = at::softmax(score_for_softmax, 0);
  at::Tensor compressed = (kv_stack * weights).sum(0);
  NormRopeStoreCompressedAten(compressed, rms_weight, rms_norm_eps, cos_sin_cache, compressed_pos, rope_head_dim,
                              kv_cache, kv_slot, kv_cache_block_size);
}

void CompressRowsNormRopeInsert(const at::Tensor& state_cache, const at::Tensor& block_table, int64_t req_idx,
                                int64_t start, int64_t compress_ratio, int64_t coff, int64_t state_block_size,
                                int64_t state_width, int64_t head_dim, const at::Tensor& rms_weight,
                                double rms_norm_eps, const at::Tensor& cos_sin_cache, int64_t compressed_pos,
                                int64_t rope_head_dim, const at::Tensor& kv_cache, int64_t kv_slot,
                                int64_t kv_cache_block_size) {
#if defined(__ARM_FEATURE_SVE)
  if (coff == 1 && TryCompressRowsNormRopeInsertCoff1OnlineSve(state_cache, block_table, req_idx, start, compress_ratio,
                                                               state_block_size, state_width, head_dim, rms_weight,
                                                               rms_norm_eps, cos_sin_cache, compressed_pos,
                                                               rope_head_dim, kv_cache, kv_slot, kv_cache_block_size)) {
    return;
  }
  if (coff == 2 && TryCompressRowsNormRopeInsertCoff2FusedSve(state_cache, block_table, req_idx, start, compress_ratio,
                                                              state_block_size, state_width, head_dim, rms_weight,
                                                              rms_norm_eps, cos_sin_cache, compressed_pos,
                                                              rope_head_dim, kv_cache, kv_slot, kv_cache_block_size)) {
    return;
  }
#endif

  const int64_t window = coff * compress_ratio;
  at::Tensor kv_stack = at::empty({window, head_dim}, state_cache.options().dtype(at::kFloat));
  at::Tensor score_stack = at::empty({window, head_dim}, state_cache.options().dtype(at::kFloat));

#if defined(__ARM_FEATURE_SVE)
  const bool copied_with_sve =
      TryCopyCompressorWindowRowsSve(state_cache, block_table, req_idx, start, compress_ratio, coff, state_block_size,
                                     state_width, head_dim, kv_stack, score_stack);
#elif FUSED_CPP_STRICT_MODE
  TORCH_CHECK(false, "FUSED_CPP_STRICT_MODE CompressRowsNormRopeInsert requires SVE copy path");
  const bool copied_with_sve = false;
#else
  const bool copied_with_sve = false;
#endif

  if (!copied_with_sve) {
    const auto copy_window_row = [&](int64_t out_row, int64_t p, int64_t segment) {
      at::Tensor kv_out = kv_stack.index({out_row});
      at::Tensor score_out = score_stack.index({out_row});
      if (p < 0) {
        kv_out.zero_();
        score_out.fill_(-std::numeric_limits<float>::infinity());
        return;
      }
      const int64_t logical_block = p / state_block_size;
      const int64_t logical_offset = p % state_block_size;
      const int64_t block = block_table.index({req_idx, logical_block}).item<int64_t>();
      at::Tensor row = state_cache.index({block, logical_offset}).to(at::kFloat);
      const int64_t kv_start = segment * head_dim;
      const int64_t score_start = state_width + segment * head_dim;
      kv_out.copy_(row.slice(0, kv_start, kv_start + head_dim));
      score_out.copy_(row.slice(0, score_start, score_start + head_dim));
    };

    if (coff == 2) {
      for (int64_t t = 0; t < compress_ratio; ++t) {
        copy_window_row(t, start + t, 0);
      }
      for (int64_t t = 0; t < compress_ratio; ++t) {
        copy_window_row(compress_ratio + t, start + compress_ratio + t, 1);
      }
    } else {
      for (int64_t t = 0; t < compress_ratio; ++t) {
        copy_window_row(t, start + t, 0);
      }
    }
  }

  if (coff == 2) {
    CompressRowsNormRopeInsertCoff2Compute(kv_stack, score_stack, rms_weight, rms_norm_eps, cos_sin_cache,
                                           compressed_pos, rope_head_dim, kv_cache, kv_slot, kv_cache_block_size);
  } else {
    CompressRowsNormRopeInsertCoff1Compute(kv_stack, score_stack, rms_weight, rms_norm_eps, cos_sin_cache,
                                           compressed_pos, rope_head_dim, kv_cache, kv_slot, kv_cache_block_size);
  }
}

std::vector<int64_t> BuildCompressorActiveIndices(const at::Tensor& pos_cpu, const at::Tensor& slot_cpu,
                                                  const at::Tensor& kv_slot_cpu, int64_t compress_ratio) {
  const int64_t num_tokens = pos_cpu.numel();
  std::vector<int64_t> active_indices;
  std::vector<int64_t> active_kv_slots;
  active_indices.reserve(static_cast<size_t>((num_tokens + compress_ratio - 1) / compress_ratio));
  active_kv_slots.reserve(active_indices.capacity());

  const int64_t* pos_data = pos_cpu.data_ptr<int64_t>();
  const int64_t* slot_data = slot_cpu.data_ptr<int64_t>();
  const int64_t* kv_slot_data = kv_slot_cpu.data_ptr<int64_t>();
  for (int64_t i = 0; i < num_tokens; ++i) {
    if (slot_data[i] < 0) {
      continue;
    }
    if ((pos_data[i] + 1) % compress_ratio != 0) {
      continue;
    }
    if (kv_slot_data[i] < 0) {
      continue;
    }
    active_indices.push_back(i);
    active_kv_slots.push_back(kv_slot_data[i]);
  }
  std::sort(active_kv_slots.begin(), active_kv_slots.end());
  for (size_t i = 1; i < active_kv_slots.size(); ++i) {
    TORCH_CHECK(active_kv_slots[i] != active_kv_slots[i - 1],
                "kv_slot_mapping[active] must not contain duplicate slots, got duplicate slot ", active_kv_slots[i]);
  }
  return active_indices;
}

void KvCompressNormRopeInsertCoff1(const at::Tensor& state_cache, const at::Tensor& block_table,
                                   const at::Tensor& rms_weight, double rms_norm_eps, const at::Tensor& cos_sin_cache,
                                   const at::Tensor& kv_cache, const at::Tensor& pos_cpu, const at::Tensor& kv_slot_cpu,
                                   const at::Tensor& req_cpu, int64_t compress_ratio, int64_t state_block_size,
                                   int64_t state_width, int64_t head_dim, int64_t rope_head_dim,
                                   int64_t kv_cache_block_size, const std::vector<int64_t>& active_indices) {
  const int64_t num_active = static_cast<int64_t>(active_indices.size());
  const int64_t* pos_data = pos_cpu.data_ptr<int64_t>();
  const int64_t* kv_slot_data = kv_slot_cpu.data_ptr<int64_t>();
  const int64_t* req_data = req_cpu.data_ptr<int64_t>();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (num_active > 1)
#endif
  for (int64_t active_idx = 0; active_idx < num_active; ++active_idx) {
    const int64_t i = active_indices[active_idx];
    const int64_t position = pos_data[i];
    const int64_t kv_slot = kv_slot_data[i];
    const int64_t req_idx = req_data[i];
    const int64_t start = position - compress_ratio + 1;

    const int64_t compressed_pos = (position / compress_ratio) * compress_ratio;
    CompressRowsNormRopeInsert(state_cache, block_table, req_idx, start, compress_ratio, 1, state_block_size,
                               state_width, head_dim, rms_weight, rms_norm_eps, cos_sin_cache, compressed_pos,
                               rope_head_dim, kv_cache, kv_slot, kv_cache_block_size);
  }
}

void KvCompressNormRopeInsertCoff2(const at::Tensor& state_cache, const at::Tensor& block_table,
                                   const at::Tensor& rms_weight, double rms_norm_eps, const at::Tensor& cos_sin_cache,
                                   const at::Tensor& kv_cache, const at::Tensor& pos_cpu, const at::Tensor& kv_slot_cpu,
                                   const at::Tensor& req_cpu, int64_t compress_ratio, int64_t state_block_size,
                                   int64_t state_width, int64_t head_dim, int64_t rope_head_dim,
                                   int64_t kv_cache_block_size, const std::vector<int64_t>& active_indices) {
  const int64_t window = 2 * compress_ratio;
  const int64_t num_active = static_cast<int64_t>(active_indices.size());
  const int64_t* pos_data = pos_cpu.data_ptr<int64_t>();
  const int64_t* kv_slot_data = kv_slot_cpu.data_ptr<int64_t>();
  const int64_t* req_data = req_cpu.data_ptr<int64_t>();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (num_active > 1)
#endif
  for (int64_t active_idx = 0; active_idx < num_active; ++active_idx) {
    const int64_t i = active_indices[active_idx];
    const int64_t position = pos_data[i];
    const int64_t kv_slot = kv_slot_data[i];
    const int64_t req_idx = req_data[i];
    const int64_t start = position - window + 1;

    const int64_t compressed_pos = (position / compress_ratio) * compress_ratio;
    CompressRowsNormRopeInsert(state_cache, block_table, req_idx, start, compress_ratio, 2, state_block_size,
                               state_width, head_dim, rms_weight, rms_norm_eps, cos_sin_cache, compressed_pos,
                               rope_head_dim, kv_cache, kv_slot, kv_cache_block_size);
  }
}

void KvCompressNormRopeInsert(const at::Tensor& state_cache, const at::Tensor& token_to_req_indices,
                              const at::Tensor& positions, const at::Tensor& slot_mapping,
                              const at::Tensor& block_table, const at::Tensor& rms_norm_weight, double rms_norm_eps,
                              const at::Tensor& cos_sin_cache, const at::Tensor& kv_cache,
                              const at::Tensor& kv_slot_mapping, int64_t compress_ratio) {
  if (kv_cache.numel() == 0 || state_cache.numel() == 0) {
    return;
  }
  const int64_t state_block_size = state_cache.size(1);
  const int64_t state_width = state_cache.size(-1) / 2;
  const int64_t head_dim = rms_norm_weight.size(0);
  TORCH_CHECK(compress_ratio > 0, "compress_ratio must be positive, got ", compress_ratio);
  TORCH_CHECK(block_table.scalar_type() == at::kInt, "compressor block_table must be int32, got ",
              block_table.scalar_type());
  TORCH_CHECK(state_width % head_dim == 0, "state width must be a multiple of head_dim");
  const int64_t coff = state_width / head_dim;
  TORCH_CHECK(coff == 1 || coff == 2, "compressor coff must be 1 or 2, got ", coff);
  TORCH_CHECK(coff != 2 || compress_ratio == 4, "coff=2 compressor requires compress_ratio=4, got ", compress_ratio);
  at::Tensor cos_sin_f = cos_sin_cache.to(at::kFloat).contiguous();
  const int64_t rope_head_dim = cos_sin_f.size(1);
  const int64_t kv_cache_block_size = kv_cache.size(1);

  at::Tensor pos_cpu = ToLongCpu(positions).reshape({-1}).contiguous();
  at::Tensor slot_cpu = ToLongCpu(slot_mapping).reshape({-1}).contiguous();
  at::Tensor kv_slot_cpu = ToLongCpu(kv_slot_mapping).reshape({-1}).contiguous();
  at::Tensor req_cpu = ToLongCpu(token_to_req_indices).reshape({-1}).contiguous();
  at::Tensor rms_weight = rms_norm_weight.to(at::kFloat).contiguous();
  const std::vector<int64_t> active_indices =
      BuildCompressorActiveIndices(pos_cpu, slot_cpu, kv_slot_cpu, compress_ratio);

  if (coff == 2) {
    KvCompressNormRopeInsertCoff2(state_cache, block_table, rms_weight, rms_norm_eps, cos_sin_f, kv_cache, pos_cpu,
                                  kv_slot_cpu, req_cpu, compress_ratio, state_block_size, state_width, head_dim,
                                  rope_head_dim, kv_cache_block_size, active_indices);
    return;
  }

  KvCompressNormRopeInsertCoff1(state_cache, block_table, rms_weight, rms_norm_eps, cos_sin_f, kv_cache, pos_cpu,
                                kv_slot_cpu, req_cpu, compress_ratio, state_block_size, state_width, head_dim,
                                rope_head_dim, kv_cache_block_size, active_indices);
}

std::tuple<at::Tensor, at::Tensor> IndexerQRopeQuant(const at::Tensor& positions, const at::Tensor& index_q,
                                                     const at::Tensor& cos_sin_cache, const at::Tensor& index_weights) {
  CheckDim(index_q, "indexer q", 3);
  CheckDim(cos_sin_cache, "indexer_cos_sin_cache", 2);
  const int64_t rope_dim = cos_sin_cache.size(1);
  TORCH_CHECK(rope_dim >= 0, "indexer rope_dim must be non-negative");
  TORCH_CHECK(rope_dim % 2 == 0, "indexer rope_dim must be even, got ", rope_dim);
  TORCH_CHECK(index_q.size(-1) >= rope_dim, "indexer q last dim must be >= rope_dim: ", index_q.size(-1), " vs ",
              rope_dim);
  at::Tensor positions_long = ToLongCpu(positions).contiguous();
  TORCH_CHECK(positions_long.dim() == 0 || positions_long.numel() == index_q.size(0),
              "positions must be scalar or have indexer q num_tokens elements");
  at::Tensor cos_sin_f = cos_sin_cache.to(at::kFloat).contiguous();
  if (rope_dim != 0) {
    CheckPositionsInRange(positions_long, index_q.size(0), cos_sin_f.size(0));
  }

  at::Tensor q_rot;
  if (index_q.scalar_type() == at::kBFloat16) {
    q_rot = index_q.contiguous();
    if (::fused_cpp::deepseek_v4::indexer_q_rope_fused_sve(q_rot, positions_long, cos_sin_f)) {
      const double softmax_scale = std::pow(static_cast<double>(index_q.size(-1)), -0.5);
      const double head_scale = std::pow(static_cast<double>(index_q.size(1)), -0.5);
      at::Tensor weights = index_weights.to(at::kFloat) * softmax_scale * head_scale;
      return std::make_tuple(q_rot, weights);
    }
  }
#if FUSED_CPP_STRICT_MODE
  TORCH_CHECK(false, "FUSED_CPP_STRICT_MODE IndexerQRopeQuant requires the SVE bf16 rows4 fast path; got q dtype ",
              index_q.scalar_type());
#else
  q_rot = GptjRopeApply(index_q, cos_sin_cache, positions, rope_dim).to(at::kBFloat16);
#endif
  const double softmax_scale = std::pow(static_cast<double>(index_q.size(-1)), -0.5);
  const double head_scale = std::pow(static_cast<double>(index_q.size(1)), -0.5);
  at::Tensor weights = index_weights.to(at::kFloat) * softmax_scale * head_scale;
  return std::make_tuple(q_rot, weights);
}

struct SparseIndexerPrefillPlan {
  at::Tensor ks_cpu;
  at::Tensor ke_cpu;
  bool select_all = false;
};

SparseIndexerPrefillPlan PrepareSparseIndexerPrefillPlan(int64_t num_tokens, int64_t topk_tokens,
                                                         const at::Tensor& cu_seqlen_ks, const at::Tensor& cu_seqlen_ke,
                                                         const at::Tensor& block_table) {
  TORCH_CHECK(block_table.scalar_type() == at::kInt, "prefill block_table must be int32, got ",
              block_table.scalar_type());

  at::Tensor ks_cpu = ToLongCpu(cu_seqlen_ks).reshape({-1}).contiguous();
  at::Tensor ke_cpu = ToLongCpu(cu_seqlen_ke).reshape({-1}).contiguous();
  TORCH_CHECK(ks_cpu.numel() >= num_tokens && ke_cpu.numel() >= num_tokens,
              "cu_seqlen_ks/cu_seqlen_ke must cover all sparse indexer tokens");
  int64_t max_valid_len = 0;
  const int64_t* ks_data = ks_cpu.data_ptr<int64_t>();
  const int64_t* ke_data = ke_cpu.data_ptr<int64_t>();
  for (int64_t i = 0; i < num_tokens; ++i) {
    max_valid_len = std::max(max_valid_len, ke_data[i] - ks_data[i]);
  }

  return SparseIndexerPrefillPlan{
      ks_cpu,
      ke_cpu,
      num_tokens > 0 && max_valid_len <= topk_tokens,
  };
}

void WriteSparseIndexerShortPath(const SparseIndexerPrefillPlan& plan, const at::Tensor& topk_indices_buffer,
                                 int64_t num_tokens, int64_t topk_tokens, PostGemmStageProfile* profile) {
  FUSED_CPP_PROFILE_START(phase_start);
  at::Tensor arange_topk = at::arange(topk_tokens, topk_indices_buffer.options().dtype(at::kInt));
  if (!TryWriteSparseIndexerShortPathRaw(topk_indices_buffer, plan.ks_cpu, plan.ke_cpu, arange_topk, num_tokens,
                                         topk_tokens)) {
    topk_indices_buffer.slice(0, 0, num_tokens).fill_(-1);
    const int64_t* ks_data = plan.ks_cpu.data_ptr<int64_t>();
    const int64_t* ke_data = plan.ke_cpu.data_ptr<int64_t>();
    for (int64_t i = 0; i < num_tokens; ++i) {
      const int64_t valid_len = ke_data[i] - ks_data[i];
      if (valid_len <= 0) {
        continue;
      }
      topk_indices_buffer.index({i, Slice(0, valid_len)}).copy_(arange_topk.slice(0, 0, valid_len));
    }
  }
  FUSED_CPP_PROFILE_ADD_IF_PTR(profile == nullptr ? nullptr : &profile->sparse_indexer_short_path_ms, phase_start);
}

void SparseAttnIndexerPrefillLong(const at::Tensor& q_quant, const at::Tensor& weights, const at::Tensor& kv_cache,
                                  const at::Tensor& topk_indices_buffer, int64_t topk_tokens,
                                  const at::Tensor& cu_seq_lens, const SparseIndexerPrefillPlan& plan,
                                  const at::Tensor& block_table, PostGemmStageProfile* profile) {
  const int64_t num_tokens = q_quant.size(0);
  const int64_t head_dim = q_quant.size(-1);
  const int64_t block_size = kv_cache.size(1);
  const int64_t* ks_data = plan.ks_cpu.data_ptr<int64_t>();
  const int64_t* ke_data = plan.ke_cpu.data_ptr<int64_t>();

  topk_indices_buffer.slice(0, 0, num_tokens).fill_(-1);

  CheckCpuTensor(q_quant, "indexer q");
  CheckCpuTensor(weights, "indexer weights");
  CheckCpuTensor(kv_cache, "indexer kv_cache");
  CheckCpuTensor(block_table, "prefill block_table");
  CheckDim(q_quant, "indexer q", 3);
  CheckDim(weights, "indexer weights", 2);
  CheckDim(kv_cache, "indexer kv_cache", 3);
  TORCH_CHECK(weights.size(0) == num_tokens && weights.size(1) == q_quant.size(1),
              "indexer weights must have shape [num_tokens, num_heads]");
  TORCH_CHECK(block_size > 0, "indexer kv_cache block size must be positive");
  TORCH_CHECK(kv_cache.size(2) == head_dim, "indexer kv_cache head_dim must match indexer q head_dim");

  at::Tensor cu_cpu = ToLongCpu(cu_seq_lens).reshape({-1}).contiguous();
  TORCH_CHECK(cu_cpu.numel() >= 1, "prefill cu_seq_lens must contain at least one element");
  const int64_t num_reqs = cu_cpu.numel() - 1;
  const int64_t* cu_data = cu_cpu.data_ptr<int64_t>();
  const int64_t total_seq_lens = cu_data[num_reqs];
  TORCH_CHECK(total_seq_lens >= 0, "prefill total sequence length must be non-negative");
  TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) >= num_reqs, "prefill block_table must cover all requests");

  const int64_t num_heads = q_quant.size(1);
  const bool use_sve_indexer =
      ::fused_cpp::deepseek_v4::indexer_sve::available() && q_quant.scalar_type() == at::kBFloat16 &&
      weights.scalar_type() == at::kFloat && kv_cache.scalar_type() == at::kBFloat16 && q_quant.is_contiguous() &&
      weights.is_contiguous() && kv_cache.is_contiguous() && num_heads > 0 && num_heads % 8 == 0 && head_dim > 0 &&
      head_dim % 4 == 0 && num_tokens <= std::numeric_limits<int>::max() &&
      num_heads <= std::numeric_limits<int>::max() && head_dim <= std::numeric_limits<int>::max() &&
      total_seq_lens <= std::numeric_limits<int>::max();
  if (profile != nullptr) {
    profile->sparse_indexer_sve_n_tile = use_sve_indexer ? ::fused_cpp::deepseek_v4::indexer_sve::n_tile() : 0;
  }

  at::Tensor packed_k;
  at::Tensor k_gathered;
  FUSED_CPP_PROFILE_START(phase_start);
  if (use_sve_indexer) {
    at::Tensor block_table_cpu = block_table.contiguous();
    const int32_t* block_ids = block_table_cpu.data_ptr<int32_t>();
    const int64_t block_table_stride = block_table_cpu.stride(0);
    std::vector<int64_t> key_row_offsets(static_cast<size_t>(total_seq_lens), -1);
    for (int64_t req = 0; req < num_reqs; ++req) {
      const int64_t seq_start = cu_data[req];
      const int64_t seq_end = cu_data[req + 1];
      TORCH_CHECK(seq_start >= 0 && seq_end >= seq_start && seq_end <= total_seq_lens,
                  "prefill cu_seq_lens must be non-negative and monotonic");
      const int64_t seq_len = seq_end - seq_start;
      const int64_t num_blocks = (seq_len + block_size - 1) / block_size;
      TORCH_CHECK(block_table_cpu.size(1) >= num_blocks, "prefill block_table row is too short for request ", req);
      for (int64_t local = 0; local < seq_len; ++local) {
        const int64_t logical_block = local / block_size;
        const int32_t block_id = block_ids[req * block_table_stride + logical_block];
        TORCH_CHECK(block_id >= 0 && block_id < kv_cache.size(0), "prefill block id out of range: ", block_id);
        key_row_offsets[seq_start + local] =
            (static_cast<int64_t>(block_id) * block_size + local % block_size) * head_dim;
      }
    }
    const int64_t packed_n = ::fused_cpp::deepseek_v4::indexer_sve::round_n(static_cast<int>(total_seq_lens));
    packed_k = at::empty({head_dim * packed_n}, q_quant.options().dtype(at::kBFloat16));
    ::fused_cpp::deepseek_v4::indexer_sve::pack_paged_k(Bf16ConstData(kv_cache), key_row_offsets.data(),
                                                        Bf16Data(packed_k), static_cast<int>(head_dim),
                                                        static_cast<int>(total_seq_lens), static_cast<int>(packed_n));
  } else {
    k_gathered = at::empty({total_seq_lens, head_dim}, q_quant.options().dtype(at::kFloat));
    for (int64_t req = 0; req < num_reqs; ++req) {
      const int64_t seq_start = cu_data[req];
      const int64_t seq_end = cu_data[req + 1];
      TORCH_CHECK(seq_start >= 0 && seq_end >= seq_start && seq_end <= total_seq_lens,
                  "prefill cu_seq_lens must be non-negative and monotonic");
      const int64_t seq_len = seq_end - seq_start;
      if (seq_len == 0) {
        continue;
      }
      const int64_t num_blocks = (seq_len + block_size - 1) / block_size;
      at::Tensor block_ids_tensor = block_table.index({req, Slice(0, num_blocks)}).to(at::kLong);
      at::Tensor gathered = kv_cache.index_select(0, block_ids_tensor).reshape({num_blocks * block_size, head_dim});
      k_gathered.index({Slice(seq_start, seq_end)}).copy_(gathered.slice(0, 0, seq_len).to(at::kFloat));
    }
  }
  FUSED_CPP_PROFILE_ADD_IF_PTR(profile == nullptr ? nullptr : &profile->sparse_indexer_gather_ms, phase_start);

  FUSED_CPP_PROFILE_RESTART(phase_start);
  at::Tensor logits;
  if (use_sve_indexer) {
    const int64_t packed_n = packed_k.numel() / head_dim;
    logits = at::empty({num_tokens, packed_n}, q_quant.options().dtype(at::kFloat));
    ::fused_cpp::deepseek_v4::indexer_sve::weighted_relu_scores(
        Bf16ConstData(q_quant), weights.data_ptr<float>(), Bf16ConstData(packed_k), logits.data_ptr<float>(),
        static_cast<int>(num_tokens), static_cast<int>(num_heads), static_cast<int>(head_dim),
        static_cast<int>(packed_n));
  } else {
    const at::Tensor q_f = q_quant.to(at::kFloat);
    const at::Tensor weights_f = weights.to(at::kFloat);
    logits = at::zeros({num_tokens, total_seq_lens}, q_quant.options().dtype(at::kFloat));
    const at::Tensor k_t = k_gathered.t();
    for (int64_t head = 0; head < num_heads; ++head) {
      at::Tensor head_scores = at::relu(at::matmul(q_f.select(1, head), k_t));
      head_scores.mul_(weights_f.select(1, head).unsqueeze(1));
      logits.add_(head_scores);
    }
  }
  FUSED_CPP_PROFILE_ADD_IF_PTR(profile == nullptr ? nullptr : &profile->sparse_indexer_score_ms, phase_start);

  FUSED_CPP_PROFILE_RESTART(phase_start);
  const bool use_native_topk =
      logits.device().is_cpu() && logits.scalar_type() == at::kFloat && logits.dim() == 2 && logits.stride(1) == 1 &&
      topk_indices_buffer.device().is_cpu() && topk_indices_buffer.scalar_type() == at::kInt &&
      topk_indices_buffer.dim() == 2 && topk_indices_buffer.size(0) >= num_tokens && topk_tokens >= 0 &&
      topk_indices_buffer.size(1) >= topk_tokens && topk_indices_buffer.stride(0) > 0 &&
      topk_indices_buffer.stride(1) > 0 &&
      (topk_tokens == 0 || (topk_indices_buffer.stride(1) <= std::numeric_limits<int64_t>::max() / topk_tokens &&
                            topk_indices_buffer.stride(0) >= topk_tokens * topk_indices_buffer.stride(1)));
  if (profile != nullptr) {
    profile->sparse_indexer_topk_native = use_native_topk ? 1 : 0;
  }
  if (use_native_topk) {
    ::fused_cpp::deepseek_v4::indexer_sve::batched_topk_indices(
        logits.data_ptr<float>(), logits.stride(0), logits.size(1), ks_data, ke_data,
        topk_indices_buffer.data_ptr<int32_t>(), topk_indices_buffer.stride(0), topk_indices_buffer.stride(1),
        num_tokens, topk_tokens);
  } else {
    for (int64_t i = 0; i < num_tokens; ++i) {
      const int64_t row_start = ks_data[i];
      const int64_t row_end = ke_data[i];
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
  }
  FUSED_CPP_PROFILE_ADD_IF_PTR(profile == nullptr ? nullptr : &profile->sparse_indexer_topk_ms, phase_start);
}

void RunCompressor(const at::Tensor& kv_score, const at::Tensor& positions, const at::Tensor& ape,
                   const at::Tensor& state_cache, const at::Tensor& state_slot_mapping,
                   const at::Tensor& token_to_req_indices, const at::Tensor& block_table, const at::Tensor& kv_cache,
                   const at::Tensor& kv_slot_mapping, const at::Tensor& norm_weight, const at::Tensor& cos_sin_cache,
                   int64_t compress_ratio, double rms_norm_eps, double* save_partial_states_ms,
                   double* compress_norm_rope_insert_ms) {
  const int64_t state_width = ape.size(1);
  TORCH_CHECK(kv_score.size(1) == 2 * state_width, "kv_score last dim must be 2 * ape width");
  at::Tensor kv = kv_score.slice(1, 0, state_width);
  at::Tensor score = kv_score.slice(1, state_width, 2 * state_width);
  FUSED_CPP_PROFILE_START(phase_start);
  SavePartialStates(kv, score, ape, positions, state_cache, state_slot_mapping, compress_ratio);
  FUSED_CPP_PROFILE_ADD_IF_PTR(save_partial_states_ms, phase_start);
  FUSED_CPP_PROFILE_RESTART(phase_start);
  KvCompressNormRopeInsert(state_cache, token_to_req_indices, positions, state_slot_mapping, block_table, norm_weight,
                           rms_norm_eps, cos_sin_cache, kv_cache, kv_slot_mapping, compress_ratio);
  FUSED_CPP_PROFILE_ADD_IF_PTR(compress_norm_rope_insert_ms, phase_start);
}

void RunMainQAndSwaPostprocess(const at::Tensor& q, const at::Tensor& kv, const at::Tensor& positions,
                               const at::Tensor& main_cos_sin_cache, const at::Tensor& swa_kv_cache,
                               const at::Tensor& swa_slot_mapping, double q_eps, PostGemmStageProfile* profile_ptr) {
  FUSED_CPP_PROFILE_START(phase_start);
  CheckMainQKvShape(q, kv);
  QNormRopeFused(q, positions, main_cos_sin_cache, q_eps);
  KvRopeCacheInsertFused(kv, swa_kv_cache, swa_slot_mapping, positions, main_cos_sin_cache);
  FUSED_CPP_PROFILE_ADD_IF_PTR(profile_ptr == nullptr ? nullptr : &profile_ptr->main_q_norm_rope_swa_insert_ms,
                               phase_start);
}

at::Tensor RunMainQAndSwaPrepacked(const at::Tensor& qr, const at::Tensor& kv, const at::Tensor& positions,
                                   const at::Tensor& main_wq_b_packed, int64_t main_wq_b_K, int64_t main_wq_b_N,
                                   int64_t main_wq_b_Np, const at::Tensor& main_cos_sin_cache,
                                   const at::Tensor& swa_kv_cache, const at::Tensor& swa_slot_mapping,
                                   int64_t main_head_dim, double q_eps, PostGemmStageProfile* profile_ptr) {
  TORCH_CHECK(main_head_dim > 0, "main_head_dim must be positive");
  TORCH_CHECK(main_wq_b_N % main_head_dim == 0, "main_wq_b_N must be divisible by main_head_dim");

  const int64_t main_num_heads = main_wq_b_N / main_head_dim;
  FUSED_CPP_PROFILE_START(phase_start);
  auto workspace_lease = ::fused_cpp::workspace::acquire();
  at::Tensor q = LinearPrepackedToDtypeWorkspace(qr, main_wq_b_packed, main_wq_b_K, main_wq_b_N, main_wq_b_Np,
                                                 qr.scalar_type(), workspace_lease)
                     .reshape({qr.size(0), main_num_heads, main_head_dim});
  FUSED_CPP_PROFILE_ADD_IF_PTR(profile_ptr == nullptr ? nullptr : &profile_ptr->main_q_gemm_ms, phase_start);

  RunMainQAndSwaPostprocess(q, kv, positions, main_cos_sin_cache, swa_kv_cache, swa_slot_mapping, q_eps, profile_ptr);
  return q;
}

}  // namespace

std::tuple<at::Tensor, int64_t, int64_t> deepseek_v4_post_gemm_prepare(at::Tensor weight) {
  return deepseek_v4_gemm_prepare_weight_for_backend(weight, SelectedPostGemmBackend() == PostGemmBackend::kSve);
}

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
    at::Tensor qr, at::Tensor kv, at::Tensor kv_score, at::Tensor indexer_kv_score, at::Tensor indexer_weights,
    at::Tensor positions, at::Tensor main_wq_b_weight, at::Tensor indexer_wq_b_weight, at::Tensor main_cos_sin_cache,
    at::Tensor indexer_cos_sin_cache, at::Tensor swa_kv_cache, at::Tensor swa_slot_mapping, at::Tensor mla_ape,
    at::Tensor mla_state_cache, at::Tensor mla_state_slot_mapping, at::Tensor mla_token_to_req_indices,
    at::Tensor mla_block_table, at::Tensor mla_kv_cache, at::Tensor mla_kv_slot_mapping, at::Tensor mla_norm_weight,
    at::Tensor indexer_ape, at::Tensor indexer_state_cache, at::Tensor indexer_state_slot_mapping,
    at::Tensor indexer_token_to_req_indices, at::Tensor indexer_block_table, at::Tensor indexer_kv_cache,
    at::Tensor indexer_kv_slot_mapping, at::Tensor indexer_norm_weight, at::Tensor topk_indices_buffer,
    at::Tensor prefill_cu_seq_lens, at::Tensor prefill_cu_seqlen_ks, at::Tensor prefill_cu_seqlen_ke,
    at::Tensor prefill_block_table, int64_t main_head_dim, double q_eps, int64_t mla_compress_ratio,
    double mla_rms_norm_eps, int64_t indexer_compress_ratio, double indexer_rms_norm_eps, int64_t topk_tokens) {
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
  const SparseIndexerPrefillPlan prefill_plan = PrepareSparseIndexerPrefillPlan(
      qr.size(0), topk_tokens, prefill_cu_seqlen_ks, prefill_cu_seqlen_ke, prefill_block_table);
  FUSED_CPP_PROFILE_ADD_IF_PTR(profile_ptr == nullptr ? nullptr : &profile_ptr->input_check_ms, phase_start);

  const int64_t main_num_heads = main_wq_b_weight.size(0) / main_head_dim;
  FUSED_CPP_PROFILE_RESTART(phase_start);
  at::Tensor q =
      LinearToDtype(qr, main_wq_b_weight, qr.scalar_type()).reshape({qr.size(0), main_num_heads, main_head_dim});
  FUSED_CPP_PROFILE_ADD_IF_PTR(profile_ptr == nullptr ? nullptr : &profile_ptr->main_q_gemm_ms, phase_start);

  FUSED_CPP_PROFILE_RESTART(phase_start);
  CheckMainQKvShape(q, kv);
  QNormRopeFused(q, positions, main_cos_sin_cache, q_eps);
  KvRopeCacheInsertFused(kv, swa_kv_cache, swa_slot_mapping, positions, main_cos_sin_cache);
  FUSED_CPP_PROFILE_ADD_IF_PTR(profile_ptr == nullptr ? nullptr : &profile_ptr->main_q_norm_rope_swa_insert_ms,
                               phase_start);

  const int64_t indexer_head_dim = indexer_norm_weight.size(0);
  at::Tensor q_quant;
  at::Tensor scaled_weights;
  if (!prefill_plan.select_all) {
    FUSED_CPP_PROFILE_RESTART(phase_start);
    at::Tensor indexer_q_linear = LinearToDtype(qr, indexer_wq_b_weight, qr.scalar_type());
    TORCH_CHECK(indexer_q_linear.size(1) % indexer_head_dim == 0,
                "indexer q linear out features must be divisible by indexer head_dim");
    FUSED_CPP_PROFILE_ADD_IF_PTR(profile_ptr == nullptr ? nullptr : &profile_ptr->indexer_q_gemm_ms, phase_start);

    const int64_t indexer_num_heads = indexer_q_linear.size(1) / indexer_head_dim;
    at::Tensor indexer_q = indexer_q_linear.reshape({indexer_q_linear.size(0), indexer_num_heads, indexer_head_dim});

    FUSED_CPP_PROFILE_RESTART(phase_start);
    auto indexer_q_and_weights = IndexerQRopeQuant(positions, indexer_q, indexer_cos_sin_cache, indexer_weights);
    q_quant = std::get<0>(indexer_q_and_weights);
    scaled_weights = std::get<1>(indexer_q_and_weights);
    FUSED_CPP_PROFILE_ADD_IF_PTR(profile_ptr == nullptr ? nullptr : &profile_ptr->indexer_q_rope_weights_ms,
                                 phase_start);
  }

  RunCompressor(kv_score, positions, mla_ape, mla_state_cache, mla_state_slot_mapping, mla_token_to_req_indices,
                mla_block_table, mla_kv_cache, mla_kv_slot_mapping, mla_norm_weight, main_cos_sin_cache,
                mla_compress_ratio, mla_rms_norm_eps,
                profile_ptr == nullptr ? nullptr : &profile_ptr->mla_save_partial_states_ms,
                profile_ptr == nullptr ? nullptr : &profile_ptr->mla_compress_norm_rope_insert_ms);
  RunCompressor(indexer_kv_score, positions, indexer_ape, indexer_state_cache, indexer_state_slot_mapping,
                indexer_token_to_req_indices, indexer_block_table, indexer_kv_cache, indexer_kv_slot_mapping,
                indexer_norm_weight, indexer_cos_sin_cache, indexer_compress_ratio, indexer_rms_norm_eps,
                profile_ptr == nullptr ? nullptr : &profile_ptr->indexer_save_partial_states_ms,
                profile_ptr == nullptr ? nullptr : &profile_ptr->indexer_compress_norm_rope_insert_ms);

  // Keep the migrated native post-GEMM path self-contained.  Do not dispatch
  // through the retired standalone sparse_attn_indexer_prefill_cpp_v0 symbol.
  if (prefill_plan.select_all) {
    WriteSparseIndexerShortPath(prefill_plan, topk_indices_buffer, qr.size(0), topk_tokens, profile_ptr);
  } else {
    SparseAttnIndexerPrefillLong(q_quant, scaled_weights, indexer_kv_cache, topk_indices_buffer, topk_tokens,
                                 prefill_cu_seq_lens, prefill_plan, prefill_block_table, profile_ptr);
  }
  FUSED_CPP_PROFILE_IF_ENABLED(profile_ptr != nullptr,
                               PrintPostGemmProfile(*profile_ptr, ::fused_cpp::profile::elapsed_ms(total_start)));
  return std::make_tuple(q, topk_indices_buffer);
}

at::Tensor deepseek_v4_post_gemm_dense_prepacked(at::Tensor qr, at::Tensor kv, at::Tensor positions,
                                                 at::Tensor main_wq_b_packed, int64_t main_wq_b_K, int64_t main_wq_b_N,
                                                 int64_t main_wq_b_Np, at::Tensor main_cos_sin_cache,
                                                 at::Tensor swa_kv_cache, at::Tensor swa_slot_mapping,
                                                 int64_t main_head_dim, double q_eps) {
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
  FUSED_CPP_PROFILE_ADD_IF_PTR(profile_ptr == nullptr ? nullptr : &profile_ptr->input_check_ms, phase_start);

  at::Tensor q =
      RunMainQAndSwaPrepacked(qr, kv, positions, main_wq_b_packed, main_wq_b_K, main_wq_b_N, main_wq_b_Np,
                              main_cos_sin_cache, swa_kv_cache, swa_slot_mapping, main_head_dim, q_eps, profile_ptr);
  FUSED_CPP_PROFILE_IF_ENABLED(profile_ptr != nullptr,
                               PrintPostGemmProfile(*profile_ptr, ::fused_cpp::profile::elapsed_ms(total_start)));
  return q;
}

at::Tensor deepseek_v4_post_gemm_c128a_prepacked(
    at::Tensor qr, at::Tensor kv, at::Tensor kv_score, at::Tensor positions, at::Tensor main_wq_b_packed,
    int64_t main_wq_b_K, int64_t main_wq_b_N, int64_t main_wq_b_Np, at::Tensor main_cos_sin_cache,
    at::Tensor swa_kv_cache, at::Tensor swa_slot_mapping, at::Tensor mla_ape, at::Tensor mla_state_cache,
    at::Tensor mla_state_slot_mapping, at::Tensor mla_token_to_req_indices, at::Tensor mla_block_table,
    at::Tensor mla_kv_cache, at::Tensor mla_kv_slot_mapping, at::Tensor mla_norm_weight, int64_t main_head_dim,
    double q_eps, int64_t mla_compress_ratio, double mla_rms_norm_eps) {
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
  CheckCpuTensor(kv_score, "kv_score");
  CheckDim(qr, "qr", 2);
  CheckDim(kv, "kv", 2);
  CheckDim(kv_score, "kv_score", 2);
  FUSED_CPP_PROFILE_ADD_IF_PTR(profile_ptr == nullptr ? nullptr : &profile_ptr->input_check_ms, phase_start);

  at::Tensor q =
      RunMainQAndSwaPrepacked(qr, kv, positions, main_wq_b_packed, main_wq_b_K, main_wq_b_N, main_wq_b_Np,
                              main_cos_sin_cache, swa_kv_cache, swa_slot_mapping, main_head_dim, q_eps, profile_ptr);
  RunCompressor(kv_score, positions, mla_ape, mla_state_cache, mla_state_slot_mapping, mla_token_to_req_indices,
                mla_block_table, mla_kv_cache, mla_kv_slot_mapping, mla_norm_weight, main_cos_sin_cache,
                mla_compress_ratio, mla_rms_norm_eps,
                profile_ptr == nullptr ? nullptr : &profile_ptr->mla_save_partial_states_ms,
                profile_ptr == nullptr ? nullptr : &profile_ptr->mla_compress_norm_rope_insert_ms);
  FUSED_CPP_PROFILE_IF_ENABLED(profile_ptr != nullptr,
                               PrintPostGemmProfile(*profile_ptr, ::fused_cpp::profile::elapsed_ms(total_start)));
  return q;
}

std::tuple<at::Tensor, at::Tensor> deepseek_v4_post_gemm_parallel_stage_prepacked(
    at::Tensor qr, at::Tensor kv, at::Tensor kv_score, at::Tensor indexer_kv_score, at::Tensor indexer_weights,
    at::Tensor positions, at::Tensor main_wq_b_packed, int64_t main_wq_b_K, int64_t main_wq_b_N, int64_t main_wq_b_Np,
    at::Tensor indexer_wq_b_packed, int64_t indexer_wq_b_K, int64_t indexer_wq_b_N, int64_t indexer_wq_b_Np,
    at::Tensor main_cos_sin_cache, at::Tensor indexer_cos_sin_cache, at::Tensor swa_kv_cache,
    at::Tensor swa_slot_mapping, at::Tensor mla_ape, at::Tensor mla_state_cache, at::Tensor mla_state_slot_mapping,
    at::Tensor mla_token_to_req_indices, at::Tensor mla_block_table, at::Tensor mla_kv_cache,
    at::Tensor mla_kv_slot_mapping, at::Tensor mla_norm_weight, at::Tensor indexer_ape, at::Tensor indexer_state_cache,
    at::Tensor indexer_state_slot_mapping, at::Tensor indexer_token_to_req_indices, at::Tensor indexer_block_table,
    at::Tensor indexer_kv_cache, at::Tensor indexer_kv_slot_mapping, at::Tensor indexer_norm_weight,
    at::Tensor topk_indices_buffer, at::Tensor prefill_cu_seq_lens, at::Tensor prefill_cu_seqlen_ks,
    at::Tensor prefill_cu_seqlen_ke, at::Tensor prefill_block_table, int64_t main_head_dim, double q_eps,
    int64_t mla_compress_ratio, double mla_rms_norm_eps, int64_t indexer_compress_ratio, double indexer_rms_norm_eps,
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
  TORCH_CHECK(main_wq_b_N % main_head_dim == 0, "main_wq_b_N must be divisible by main_head_dim");
  TORCH_CHECK(indexer_wq_b_N % indexer_norm_weight.size(0) == 0,
              "indexer_wq_b_N must be divisible by indexer head_dim");
  const SparseIndexerPrefillPlan prefill_plan = PrepareSparseIndexerPrefillPlan(
      qr.size(0), topk_tokens, prefill_cu_seqlen_ks, prefill_cu_seqlen_ke, prefill_block_table);
  FUSED_CPP_PROFILE_ADD_IF_PTR(profile_ptr == nullptr ? nullptr : &profile_ptr->input_check_ms, phase_start);

  const int64_t main_num_heads = main_wq_b_N / main_head_dim;
  const int64_t indexer_head_dim = indexer_norm_weight.size(0);
  at::Tensor q;
  at::Tensor indexer_q_linear;
  if (!prefill_plan.select_all && PostGemmSharedQPoolEnabled(qr.size(0))) {
    FUSED_CPP_PROFILE_RESTART(phase_start);
    auto workspace_lease = ::fused_cpp::workspace::acquire();
    auto q_pair = PostLinearPrepackedPairToDtypeWorkspace(qr, main_wq_b_packed, main_wq_b_K, main_wq_b_N, main_wq_b_Np,
                                                          indexer_wq_b_packed, indexer_wq_b_K, indexer_wq_b_N,
                                                          indexer_wq_b_Np, qr.scalar_type(), workspace_lease);
    q = std::get<0>(q_pair).reshape({qr.size(0), main_num_heads, main_head_dim});
    indexer_q_linear = std::get<1>(q_pair);
    FUSED_CPP_PROFILE_ADD_IF_PTR(profile_ptr == nullptr ? nullptr : &profile_ptr->shared_q_gemm_ms, phase_start);
    RunMainQAndSwaPostprocess(q, kv, positions, main_cos_sin_cache, swa_kv_cache, swa_slot_mapping, q_eps, profile_ptr);
  } else {
    q = RunMainQAndSwaPrepacked(qr, kv, positions, main_wq_b_packed, main_wq_b_K, main_wq_b_N, main_wq_b_Np,
                                main_cos_sin_cache, swa_kv_cache, swa_slot_mapping, main_head_dim, q_eps, profile_ptr);
    if (!prefill_plan.select_all) {
      auto workspace_lease = ::fused_cpp::workspace::acquire();
      FUSED_CPP_PROFILE_RESTART(phase_start);
      indexer_q_linear = LinearPrepackedToDtypeWorkspace(qr, indexer_wq_b_packed, indexer_wq_b_K, indexer_wq_b_N,
                                                         indexer_wq_b_Np, qr.scalar_type(), workspace_lease);
      FUSED_CPP_PROFILE_ADD_IF_PTR(profile_ptr == nullptr ? nullptr : &profile_ptr->indexer_q_gemm_ms, phase_start);
    }
  }
  at::Tensor q_quant;
  at::Tensor scaled_weights;
  if (!prefill_plan.select_all) {
    TORCH_CHECK(indexer_q_linear.size(1) % indexer_head_dim == 0,
                "indexer q linear out features must be divisible by indexer head_dim");

    const int64_t indexer_num_heads = indexer_q_linear.size(1) / indexer_head_dim;
    at::Tensor indexer_q = indexer_q_linear.reshape({indexer_q_linear.size(0), indexer_num_heads, indexer_head_dim});

    FUSED_CPP_PROFILE_RESTART(phase_start);
    auto indexer_q_and_weights = IndexerQRopeQuant(positions, indexer_q, indexer_cos_sin_cache, indexer_weights);
    q_quant = std::get<0>(indexer_q_and_weights);
    scaled_weights = std::get<1>(indexer_q_and_weights);
    FUSED_CPP_PROFILE_ADD_IF_PTR(profile_ptr == nullptr ? nullptr : &profile_ptr->indexer_q_rope_weights_ms,
                                 phase_start);
  }

  RunCompressor(kv_score, positions, mla_ape, mla_state_cache, mla_state_slot_mapping, mla_token_to_req_indices,
                mla_block_table, mla_kv_cache, mla_kv_slot_mapping, mla_norm_weight, main_cos_sin_cache,
                mla_compress_ratio, mla_rms_norm_eps,
                profile_ptr == nullptr ? nullptr : &profile_ptr->mla_save_partial_states_ms,
                profile_ptr == nullptr ? nullptr : &profile_ptr->mla_compress_norm_rope_insert_ms);
  RunCompressor(indexer_kv_score, positions, indexer_ape, indexer_state_cache, indexer_state_slot_mapping,
                indexer_token_to_req_indices, indexer_block_table, indexer_kv_cache, indexer_kv_slot_mapping,
                indexer_norm_weight, indexer_cos_sin_cache, indexer_compress_ratio, indexer_rms_norm_eps,
                profile_ptr == nullptr ? nullptr : &profile_ptr->indexer_save_partial_states_ms,
                profile_ptr == nullptr ? nullptr : &profile_ptr->indexer_compress_norm_rope_insert_ms);

  // Keep the migrated native post-GEMM path self-contained.  Do not dispatch
  // through the retired standalone sparse_attn_indexer_prefill_cpp_v0 symbol.
  if (prefill_plan.select_all) {
    WriteSparseIndexerShortPath(prefill_plan, topk_indices_buffer, qr.size(0), topk_tokens, profile_ptr);
  } else {
    SparseAttnIndexerPrefillLong(q_quant, scaled_weights, indexer_kv_cache, topk_indices_buffer, topk_tokens,
                                 prefill_cu_seq_lens, prefill_plan, prefill_block_table, profile_ptr);
  }
  FUSED_CPP_PROFILE_IF_ENABLED(profile_ptr != nullptr,
                               PrintPostGemmProfile(*profile_ptr, ::fused_cpp::profile::elapsed_ms(total_start)));
  return std::make_tuple(q, topk_indices_buffer);
}

at::Tensor deepseek_v4_post_gemm_dense_projected(at::Tensor main_q_linear, at::Tensor kv, at::Tensor positions,
                                                 at::Tensor main_cos_sin_cache, at::Tensor swa_kv_cache,
                                                 at::Tensor swa_slot_mapping, int64_t main_head_dim, double q_eps) {
  CheckCpuTensor(main_q_linear, "main_q_linear");
  CheckDim(main_q_linear, "main_q_linear", 2);
  TORCH_CHECK(main_q_linear.scalar_type() == at::kBFloat16, "main_q_linear must be torch.bfloat16");
  TORCH_CHECK(main_head_dim > 0 && main_q_linear.size(1) % main_head_dim == 0,
              "main_q_linear width must be divisible by main_head_dim");
  at::Tensor q = main_q_linear.reshape({main_q_linear.size(0), main_q_linear.size(1) / main_head_dim, main_head_dim});
  RunMainQAndSwaPostprocess(q, kv, positions, main_cos_sin_cache, swa_kv_cache, swa_slot_mapping, q_eps, nullptr);
  return q;
}

at::Tensor deepseek_v4_post_gemm_c128a_projected(
    at::Tensor main_q_linear, at::Tensor kv, at::Tensor kv_score, at::Tensor positions, at::Tensor main_cos_sin_cache,
    at::Tensor swa_kv_cache, at::Tensor swa_slot_mapping, at::Tensor mla_ape, at::Tensor mla_state_cache,
    at::Tensor mla_state_slot_mapping, at::Tensor mla_token_to_req_indices, at::Tensor mla_block_table,
    at::Tensor mla_kv_cache, at::Tensor mla_kv_slot_mapping, at::Tensor mla_norm_weight, int64_t main_head_dim,
    double q_eps, int64_t mla_compress_ratio, double mla_rms_norm_eps) {
  at::Tensor q = deepseek_v4_post_gemm_dense_projected(main_q_linear, kv, positions, main_cos_sin_cache, swa_kv_cache,
                                                       swa_slot_mapping, main_head_dim, q_eps);
  RunCompressor(kv_score, positions, mla_ape, mla_state_cache, mla_state_slot_mapping, mla_token_to_req_indices,
                mla_block_table, mla_kv_cache, mla_kv_slot_mapping, mla_norm_weight, main_cos_sin_cache,
                mla_compress_ratio, mla_rms_norm_eps, nullptr, nullptr);
  return q;
}

std::tuple<at::Tensor, at::Tensor> deepseek_v4_post_gemm_parallel_stage_projected(
    at::Tensor main_q_linear, at::Tensor indexer_q_linear, at::Tensor kv, at::Tensor kv_score,
    at::Tensor indexer_kv_score, at::Tensor indexer_weights, at::Tensor positions, at::Tensor main_cos_sin_cache,
    at::Tensor indexer_cos_sin_cache, at::Tensor swa_kv_cache, at::Tensor swa_slot_mapping, at::Tensor mla_ape,
    at::Tensor mla_state_cache, at::Tensor mla_state_slot_mapping, at::Tensor mla_token_to_req_indices,
    at::Tensor mla_block_table, at::Tensor mla_kv_cache, at::Tensor mla_kv_slot_mapping, at::Tensor mla_norm_weight,
    at::Tensor indexer_ape, at::Tensor indexer_state_cache, at::Tensor indexer_state_slot_mapping,
    at::Tensor indexer_token_to_req_indices, at::Tensor indexer_block_table, at::Tensor indexer_kv_cache,
    at::Tensor indexer_kv_slot_mapping, at::Tensor indexer_norm_weight, at::Tensor topk_indices_buffer,
    at::Tensor prefill_cu_seq_lens, at::Tensor prefill_cu_seqlen_ks, at::Tensor prefill_cu_seqlen_ke,
    at::Tensor prefill_block_table, int64_t main_head_dim, double q_eps, int64_t mla_compress_ratio,
    double mla_rms_norm_eps, int64_t indexer_compress_ratio, double indexer_rms_norm_eps, int64_t topk_tokens) {
  const int64_t num_tokens = main_q_linear.size(0);
  const SparseIndexerPrefillPlan prefill_plan = PrepareSparseIndexerPrefillPlan(
      num_tokens, topk_tokens, prefill_cu_seqlen_ks, prefill_cu_seqlen_ke, prefill_block_table);
  at::Tensor q = deepseek_v4_post_gemm_dense_projected(main_q_linear, kv, positions, main_cos_sin_cache, swa_kv_cache,
                                                       swa_slot_mapping, main_head_dim, q_eps);

  at::Tensor q_quant;
  at::Tensor scaled_weights;
  if (!prefill_plan.select_all) {
    CheckCpuTensor(indexer_q_linear, "indexer_q_linear");
    CheckDim(indexer_q_linear, "indexer_q_linear", 2);
    TORCH_CHECK(indexer_q_linear.scalar_type() == at::kBFloat16, "indexer_q_linear must be torch.bfloat16");
    const int64_t indexer_head_dim = indexer_norm_weight.size(0);
    TORCH_CHECK(indexer_head_dim > 0 && indexer_q_linear.size(0) == num_tokens &&
                    indexer_q_linear.size(1) % indexer_head_dim == 0,
                "indexer_q_linear shape is incompatible with indexer_norm_weight");
    at::Tensor indexer_q =
        indexer_q_linear.reshape({num_tokens, indexer_q_linear.size(1) / indexer_head_dim, indexer_head_dim});
    auto indexer_q_and_weights = IndexerQRopeQuant(positions, indexer_q, indexer_cos_sin_cache, indexer_weights);
    q_quant = std::get<0>(indexer_q_and_weights);
    scaled_weights = std::get<1>(indexer_q_and_weights);
  }

  RunCompressor(kv_score, positions, mla_ape, mla_state_cache, mla_state_slot_mapping, mla_token_to_req_indices,
                mla_block_table, mla_kv_cache, mla_kv_slot_mapping, mla_norm_weight, main_cos_sin_cache,
                mla_compress_ratio, mla_rms_norm_eps, nullptr, nullptr);
  RunCompressor(indexer_kv_score, positions, indexer_ape, indexer_state_cache, indexer_state_slot_mapping,
                indexer_token_to_req_indices, indexer_block_table, indexer_kv_cache, indexer_kv_slot_mapping,
                indexer_norm_weight, indexer_cos_sin_cache, indexer_compress_ratio, indexer_rms_norm_eps, nullptr,
                nullptr);

  if (prefill_plan.select_all) {
    WriteSparseIndexerShortPath(prefill_plan, topk_indices_buffer, num_tokens, topk_tokens, nullptr);
  } else {
    SparseAttnIndexerPrefillLong(q_quant, scaled_weights, indexer_kv_cache, topk_indices_buffer, topk_tokens,
                                 prefill_cu_seq_lens, prefill_plan, prefill_block_table, nullptr);
  }
  return std::make_tuple(q, topk_indices_buffer);
}
