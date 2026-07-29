#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include "deepseek_v4_attn_gemm_sve.h"
#include "workspace_pool.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

#ifdef __aarch64__
#include "gemm_params.h"

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

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

namespace {

constexpr int64_t kTile = 8;
constexpr int kAttnGemmPrepackMinNGroups = 24;

enum class AttnGemmBackend {
  kNeon,
  kSve,
};

enum class AttnGemmSchedule {
  kLegacy,
  kM8Aligned,
  kSharedPool,
  kMnPool,
};

int64_t ceil_div_int64(int64_t x, int64_t y) { return (x + y - 1) / y; }

int64_t ceil_to_multiple(int64_t x, int64_t multiple) { return ceil_div_int64(x, multiple) * multiple; }

bool env_false_local(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  return value[0] == '\0' || value[0] == '0' || std::strcmp(value, "false") == 0 || std::strcmp(value, "False") == 0 ||
         std::strcmp(value, "off") == 0 || std::strcmp(value, "OFF") == 0;
}

[[maybe_unused]] bool attn_gemm_prepack_a_enabled(AttnGemmSchedule schedule, int requested_n_groups) {
  const char* value = std::getenv("FUSED_CPP_ATTN_GEMM_PREPACK_A");
  if (value != nullptr) {
    return !env_false_local("FUSED_CPP_ATTN_GEMM_PREPACK_A");
  }
  return schedule == AttnGemmSchedule::kMnPool && requested_n_groups >= kAttnGemmPrepackMinNGroups;
}

AttnGemmSchedule selected_attn_gemm_schedule() {
  const char* value = std::getenv("FUSED_CPP_ATTN_GEMM_SCHEDULE");
  if (value == nullptr || value[0] == '\0') {
    return AttnGemmSchedule::kMnPool;
  }
  if (std::strcmp(value, "legacy") == 0) {
    return AttnGemmSchedule::kLegacy;
  }
  if (std::strcmp(value, "m8") == 0 || std::strcmp(value, "m8_aligned") == 0) {
    return AttnGemmSchedule::kM8Aligned;
  }
  if (std::strcmp(value, "pool") == 0 || std::strcmp(value, "shared_pool") == 0) {
    return AttnGemmSchedule::kSharedPool;
  }
  if (std::strcmp(value, "mn") == 0 || std::strcmp(value, "mn_pool") == 0) {
    return AttnGemmSchedule::kMnPool;
  }
  TORCH_CHECK(false, "FUSED_CPP_ATTN_GEMM_SCHEDULE must be one of legacy/m8/pool/mn, got ", value);
  return AttnGemmSchedule::kMnPool;
}

int requested_attn_gemm_n_groups(int num_threads) {
  const char* value = std::getenv("FUSED_CPP_ATTN_GEMM_N_GROUPS");
  if (value == nullptr || value[0] == '\0') {
    return num_threads <= 48 ? num_threads : std::max(1, num_threads / 2);
  }
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  TORCH_CHECK(end != value && *end == '\0' && parsed > 0 && parsed <= std::numeric_limits<int>::max(),
              "FUSED_CPP_ATTN_GEMM_N_GROUPS must be a positive integer, got ", value);
  return static_cast<int>(parsed);
}

AttnGemmBackend selected_attn_gemm_backend() {
  const char* backend = std::getenv("FUSED_CPP_POST_GEMM_BACKEND");
  if (backend == nullptr) {
    backend = std::getenv("FUSED_CPP_ATTN_GEMM_BACKEND");
  }
  if (backend != nullptr) {
    if (std::strcmp(backend, "sve") == 0 || std::strcmp(backend, "SVE") == 0) {
      TORCH_CHECK(::fused_cpp::deepseek_v4::attn_sve::available(),
                  "FUSED_CPP_ATTN_GEMM_BACKEND=sve requested but this "
                  "build/CPU does not support SVE BF16");
      return AttnGemmBackend::kSve;
    }
    if (std::strcmp(backend, "neon") == 0 || std::strcmp(backend, "NEON") == 0 ||
        std::strcmp(backend, "default") == 0) {
      return AttnGemmBackend::kNeon;
    }
    TORCH_CHECK(std::strcmp(backend, "auto") == 0 || std::strcmp(backend, "AUTO") == 0,
                "FUSED_CPP_ATTN_GEMM_BACKEND must be one of "
                "auto/neon/sve, got ",
                backend);
  }

  // Preserve the existing NEON path by default; SVE is opt-in while it is
  // being validated across the DeepSeek V4 operator shapes.
  if (::fused_cpp::deepseek_v4::attn_sve::available() && std::getenv("FUSED_CPP_ATTN_GEMM_SVE") != nullptr &&
      !env_false_local("FUSED_CPP_ATTN_GEMM_SVE")) {
    return AttnGemmBackend::kSve;
  }
  return AttnGemmBackend::kNeon;
}

int64_t attn_gemm_round_k(int64_t K, AttnGemmBackend backend) {
  if (backend == AttnGemmBackend::kSve) {
    return ::fused_cpp::deepseek_v4::attn_sve::round_k(static_cast<int>(K));
  }
  return ceil_to_multiple(K, kTile);
}

int64_t attn_gemm_round_n(int64_t N, AttnGemmBackend backend) {
  if (backend == AttnGemmBackend::kSve) {
    return ::fused_cpp::deepseek_v4::attn_sve::round_n(static_cast<int>(N));
  }
  return ceil_to_multiple(N, kTile);
}

int64_t attn_gemm_n_tile(AttnGemmBackend backend) {
  if (backend == AttnGemmBackend::kSve) {
    return ::fused_cpp::deepseek_v4::attn_sve::n_tile();
  }
  return kTile;
}

void check_bf16_cpu_2d(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must have dtype torch.bfloat16");
  TORCH_CHECK(tensor.dim() == 2, name, " must be 2-D, got ", tensor.dim(), "-D");
}

at::Tensor checked_norm_weight_f32(at::Tensor weight, int64_t dim, const char* name) {
  TORCH_CHECK(weight.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(weight.dim() == 1, name, " must be 1-D, got ", weight.dim(), "-D");
  TORCH_CHECK(weight.size(0) == dim, name, " size mismatch: expected ", dim, ", got ", weight.size(0));
  TORCH_CHECK(weight.scalar_type() == at::kFloat || weight.scalar_type() == at::kBFloat16, name,
              " must have dtype torch.float32 or torch.bfloat16");
  if (weight.scalar_type() == at::kFloat && weight.is_contiguous()) {
    return weight;
  }
  return weight.to(at::kFloat).contiguous();
}

void check_int_arg(int64_t value, const char* name) {
  TORCH_CHECK(value > 0, name, " must be positive, got ", value);
  TORCH_CHECK(value <= std::numeric_limits<int>::max(), name, " exceeds int32 kernel limit: ", value);
}

#ifdef __aarch64__

const uint16_t* bf16_data_const(const at::Tensor& tensor) {
  return reinterpret_cast<const uint16_t*>(tensor.data_ptr<at::BFloat16>());
}

bool bind_current_thread_to_cpu(int64_t cpu_id) {
#ifdef __linux__
  if (cpu_id < 0 || cpu_id >= CPU_SETSIZE) {
    return false;
  }
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(static_cast<int>(cpu_id), &mask);
  return pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) == 0;
#else
  (void)cpu_id;
  return true;
#endif
}

#ifdef __linux__
bool get_current_thread_affinity(cpu_set_t* mask) {
  return pthread_getaffinity_np(pthread_self(), sizeof(*mask), mask) == 0;
}

bool set_current_thread_affinity(const cpu_set_t* mask) {
  return pthread_setaffinity_np(pthread_self(), sizeof(*mask), mask) == 0;
}
#endif

uint16_t* bf16_data(at::Tensor& tensor) { return reinterpret_cast<uint16_t*>(tensor.data_ptr<at::BFloat16>()); }

void pack_a_reorder_m8_range(const uint16_t* A, uint16_t* packed, int rows, int K, int panel_begin, int panel_end) {
  const int k_blocks = K / 4;
  for (int panel = panel_begin; panel < panel_end; ++panel) {
    uint16_t* packed_panel = packed + static_cast<int64_t>(panel) * kTile * K;
    for (int k_block = 0; k_block < k_blocks; ++k_block) {
      uint16_t* dst = packed_panel + static_cast<int64_t>(k_block) * kTile * 4;
      for (int row = 0; row < kTile; ++row) {
        const int source_row = panel * kTile + row;
        uint16_t* dst_row = dst + row * 4;
        if (source_row < rows) {
          const uint16_t* src = A + static_cast<int64_t>(source_row) * K + k_block * 4;
          std::memcpy(dst_row, src, 4 * sizeof(uint16_t));
        } else {
          std::memset(dst_row, 0, 4 * sizeof(uint16_t));
        }
      }
    }
  }
}

void bf16_pack_b(const uint16_t* B, uint16_t* B_reo, int K, int N) {
  int64_t idx = 0;
  for (int cb = 0; cb < N / 8; ++cb) {
    for (int rb = 0; rb < K / 4; ++rb) {
      const int row_base = rb * 4;
      const int col_base = cb * 8;
      for (int cp = 0; cp < 4; ++cp) {
        const int c0 = col_base + cp * 2;
        const int c1 = c0 + 1;
        for (int i = 0; i < 4; ++i) {
          B_reo[idx++] = B[(row_base + i) * N + c0];
        }
        for (int i = 0; i < 4; ++i) {
          B_reo[idx++] = B[(row_base + i) * N + c1];
        }
      }
    }
  }
}

struct PackedWeight {
  at::Tensor tensor;
  int64_t K;
  int64_t N;
  int64_t K_pad;
  int64_t N_pad;
};

PackedWeight checked_packed_weight(at::Tensor packed, int64_t K, int64_t N, const char* name) {
  TORCH_CHECK(packed.device().is_cpu(), name, " packed weight must be CPU");
  TORCH_CHECK(packed.scalar_type() == at::kBFloat16, name, " packed weight must be torch.bfloat16");
  TORCH_CHECK(packed.is_contiguous(), name, " packed weight must be contiguous");
  check_int_arg(K, "K");
  check_int_arg(N, "N");
  const AttnGemmBackend backend = selected_attn_gemm_backend();
  const int64_t K_pad = attn_gemm_round_k(K, backend);
  const int64_t N_pad = attn_gemm_round_n(N, backend);
  TORCH_CHECK(K_pad <= std::numeric_limits<int>::max(), name, " padded K exceeds int32 kernel limit: ", K_pad);
  TORCH_CHECK(N_pad <= std::numeric_limits<int>::max(), name, " padded N exceeds int32 kernel limit: ", N_pad);
  TORCH_CHECK(packed.numel() == K_pad * N_pad, name, " packed weight numel mismatch: expected ", K_pad * N_pad,
              ", got ", packed.numel());
  return PackedWeight{packed, K, N, K_pad, N_pad};
}

at::Tensor narrow_output_if_needed(const at::Tensor& output, int64_t N, int64_t N_pad) {
  if (N == N_pad) {
    return output;
  }
  return output.narrow(1, 0, N).contiguous();
}

void dispatch_fp32_gemm(const uint16_t* A, const uint16_t* B_reo, float* C, uint16_t* A_reorder, int M, int K, int N,
                        int ldc) {
  if (selected_attn_gemm_backend() == AttnGemmBackend::kSve) {
    ::fused_cpp::deepseek_v4::attn_sve::dispatch_f32(A, B_reo, C, A_reorder, M, K, N, ldc);
    return;
  }
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

at::Tensor run_fp32_gemm(const at::Tensor& a_storage, const PackedWeight& weight, at::Tensor& scratch) {
  const int64_t M64 = a_storage.size(0);
  const auto options = at::TensorOptions().device(a_storage.device()).dtype(at::kFloat);
  at::Tensor output = at::zeros({M64, weight.N_pad}, options);
  if (M64 == 0) {
    return output.narrow(1, 0, weight.N).contiguous();
  }
  dispatch_fp32_gemm(bf16_data_const(a_storage), bf16_data_const(weight.tensor), output.data_ptr<float>(),
                     bf16_data(scratch), static_cast<int>(M64), static_cast<int>(weight.K_pad),
                     static_cast<int>(weight.N_pad), static_cast<int>(weight.N_pad));
  return narrow_output_if_needed(output, weight.N, weight.N_pad);
}

#ifdef __linux__
void dispatch_bf16_nld_gemm(const uint16_t* A, const uint16_t* B_reo, uint16_t* C, uint16_t* A_reorder, int M, int K,
                            int N, int ldc) {
  if (selected_attn_gemm_backend() == AttnGemmBackend::kSve) {
    ::fused_cpp::deepseek_v4::attn_sve::dispatch_bf16(A, B_reo, C, A_reorder, M, K, N, ldc);
    return;
  }
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
}
#endif

at::Tensor run_bf16_gemm(const at::Tensor& a_storage, const PackedWeight& weight, at::Tensor& scratch) {
#ifdef __APPLE__
  // macOS/Apple currently mis-handles bf16-output nld tail kernels for M<8.
  return run_fp32_gemm(a_storage, weight, scratch).to(at::kBFloat16);
#elif defined(__linux__)
  const int64_t M64 = a_storage.size(0);
  at::Tensor output = at::empty({M64, weight.N_pad}, a_storage.options());
  if (M64 == 0) {
    return output.narrow(1, 0, weight.N).contiguous();
  }
  dispatch_bf16_nld_gemm(bf16_data_const(a_storage), bf16_data_const(weight.tensor), bf16_data(output),
                         bf16_data(scratch), static_cast<int>(M64), static_cast<int>(weight.K_pad),
                         static_cast<int>(weight.N_pad), static_cast<int>(weight.N_pad));
  return narrow_output_if_needed(output, weight.N, weight.N_pad);
#else
  return run_fp32_gemm(a_storage, weight, scratch).to(at::kBFloat16);
#endif
}

[[maybe_unused]] void dispatch_fp32_gemm_to_output(const at::Tensor& a_storage, const PackedWeight& weight,
                                                   at::Tensor& output, at::Tensor& scratch, int64_t row_start,
                                                   int64_t row_count, int64_t scratch_offset) {
  if (row_count == 0) {
    return;
  }
  dispatch_fp32_gemm(bf16_data_const(a_storage) + row_start * weight.K_pad, bf16_data_const(weight.tensor),
                     output.data_ptr<float>() + row_start * weight.N_pad, bf16_data(scratch) + scratch_offset,
                     static_cast<int>(row_count), static_cast<int>(weight.K_pad), static_cast<int>(weight.N_pad),
                     static_cast<int>(weight.N_pad));
}

[[maybe_unused]] void dispatch_bf16_gemm_to_output(const at::Tensor& a_storage, const PackedWeight& weight,
                                                   at::Tensor& output, at::Tensor& scratch, int64_t row_start,
                                                   int64_t row_count, int64_t scratch_offset) {
  if (row_count == 0) {
    return;
  }
#if defined(__linux__)
  dispatch_bf16_nld_gemm(bf16_data_const(a_storage) + row_start * weight.K_pad, bf16_data_const(weight.tensor),
                         bf16_data(output) + row_start * weight.N_pad, bf16_data(scratch) + scratch_offset,
                         static_cast<int>(row_count), static_cast<int>(weight.K_pad), static_cast<int>(weight.N_pad),
                         static_cast<int>(weight.N_pad));
#else
  TORCH_CHECK(false, "direct bf16-output GEMM dispatch is only enabled on Linux");
#endif
}

void dispatch_fp32_gemm_range_to_output(const at::Tensor& a_storage, const PackedWeight& weight, at::Tensor& output,
                                        at::Tensor& scratch, int64_t row_start, int64_t row_count, int64_t n_begin,
                                        int64_t n_cols, int64_t scratch_offset) {
  if (row_count == 0 || n_cols == 0) {
    return;
  }
  dispatch_fp32_gemm(
      bf16_data_const(a_storage) + row_start * weight.K_pad,
      bf16_data_const(weight.tensor) + n_begin * weight.K_pad,
      output.data_ptr<float>() + row_start * weight.N_pad + n_begin, bf16_data(scratch) + scratch_offset,
      static_cast<int>(row_count), static_cast<int>(weight.K_pad), static_cast<int>(n_cols),
      static_cast<int>(weight.N_pad));
}

[[maybe_unused]] void dispatch_bf16_gemm_range_to_output(
    const at::Tensor& a_storage, const PackedWeight& weight, at::Tensor& output, at::Tensor& scratch, int64_t row_start,
    int64_t row_count, int64_t n_begin, int64_t n_cols, int64_t scratch_offset) {
  if (row_count == 0 || n_cols == 0) {
    return;
  }
#if defined(__linux__)
  dispatch_bf16_nld_gemm(
      bf16_data_const(a_storage) + row_start * weight.K_pad,
      bf16_data_const(weight.tensor) + n_begin * weight.K_pad,
      bf16_data(output) + row_start * weight.N_pad + n_begin, bf16_data(scratch) + scratch_offset,
      static_cast<int>(row_count), static_cast<int>(weight.K_pad), static_cast<int>(n_cols),
      static_cast<int>(weight.N_pad));
#else
  TORCH_CHECK(false, "direct bf16-output GEMM range dispatch is only enabled on Linux");
#endif
}

void dispatch_packed_gemm_range_to_output(const uint16_t* packed_a, const PackedWeight& weight, at::Tensor& output,
                                          bool bf16_output, int64_t row_start, int64_t n_begin, int64_t n_cols) {
  if (n_cols == 0) {
    return;
  }
  gemm_params_t params;
  params.m = kTile;
  params.k = static_cast<int>(weight.K_pad);
  params.n = static_cast<int>(n_cols);
  params.lda = static_cast<int>(weight.K_pad);
  params.ldb = static_cast<int>(weight.K_pad);
  params.ldc = static_cast<int>(weight.N_pad);
  const uint16_t* panel_a = packed_a + row_start * weight.K_pad;
  const uint16_t* panel_b = bf16_data_const(weight.tensor) + n_begin * weight.K_pad;
  if (bf16_output) {
    deepseek_v4_attn_gemm_packed_bf16(
        panel_a, panel_b, bf16_data(output) + row_start * weight.N_pad + n_begin, nullptr, &params);
  } else {
    deepseek_v4_attn_gemm_packed_f32(
        panel_a, panel_b, output.data_ptr<float>() + row_start * weight.N_pad + n_begin, nullptr, &params);
  }
}

struct AttnGemmSelectedOutputs {
  at::Tensor qr_kv;
  at::Tensor kv_score;
  at::Tensor indexer_kv_score;
  at::Tensor indexer_weights;
};

struct AttnGemmWork {
  const PackedWeight* weight;
  at::Tensor* output;
  bool bf16_output;
};

struct AttnGemmTaskGroup {
  int work_index;
  int64_t n_begin;
  int64_t n_cols;
};

struct alignas(64) AttnGemmTaskCursor {
  std::atomic<int64_t> next_panel{0};
};

std::vector<int> allocate_attn_gemm_n_groups(const std::vector<AttnGemmWork>& work, int requested_groups,
                                             int64_t n_tile) {
  std::vector<int> groups(work.size(), 1);
  int total_tiles = 0;
  for (const AttnGemmWork& item : work) {
    total_tiles += static_cast<int>(item.weight->N_pad / n_tile);
  }
  const int target = std::min(total_tiles, std::max(requested_groups, static_cast<int>(work.size())));
  for (int assigned = static_cast<int>(work.size()); assigned < target; ++assigned) {
    int best = -1;
    for (int i = 0; i < static_cast<int>(work.size()); ++i) {
      const int tiles = static_cast<int>(work[static_cast<size_t>(i)].weight->N_pad / n_tile);
      if (groups[static_cast<size_t>(i)] >= tiles) {
        continue;
      }
      if (best < 0) {
        best = i;
        continue;
      }
      const int best_tiles = static_cast<int>(work[static_cast<size_t>(best)].weight->N_pad / n_tile);
      if (static_cast<int64_t>(tiles) * groups[static_cast<size_t>(best)] >
          static_cast<int64_t>(best_tiles) * groups[static_cast<size_t>(i)]) {
        best = i;
      }
    }
    if (best < 0) {
      break;
    }
    ++groups[static_cast<size_t>(best)];
  }
  return groups;
}

std::vector<AttnGemmTaskGroup> make_attn_gemm_task_groups(const std::vector<AttnGemmWork>& work,
                                                         AttnGemmSchedule schedule, int requested_groups,
                                                         int64_t n_tile) {
  std::vector<int> groups_per_work(work.size(), 1);
  if (schedule == AttnGemmSchedule::kMnPool) {
    groups_per_work = allocate_attn_gemm_n_groups(work, requested_groups, n_tile);
  }

  std::vector<AttnGemmTaskGroup> groups;
  for (int work_index = 0; work_index < static_cast<int>(work.size()); ++work_index) {
    const int64_t n_tiles = work[static_cast<size_t>(work_index)].weight->N_pad / n_tile;
    const int splits = groups_per_work[static_cast<size_t>(work_index)];
    for (int split = 0; split < splits; ++split) {
      const int64_t tile_begin = static_cast<int64_t>(split) * n_tiles / splits;
      const int64_t tile_end = static_cast<int64_t>(split + 1) * n_tiles / splits;
      groups.push_back(
          AttnGemmTaskGroup{work_index, tile_begin * n_tile, (tile_end - tile_begin) * n_tile});
    }
  }
  return groups;
}

struct AttnGemmNormedOutputs {
  at::Tensor qr;
  at::Tensor kv;
  at::Tensor kv_score;
  at::Tensor indexer_kv_score;
  at::Tensor indexer_weights;
};

inline float bf16_u16_to_float(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline uint16_t float_to_bf16_u16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1U;
  const uint32_t rounding_bias = 0x7fffU + lsb;
  return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

#if defined(__ARM_FEATURE_SVE)
inline svfloat32_t bf16_load_f32_sve(svbool_t pg, const uint16_t* ptr) {
  const svuint32_t h = svld1uh_u32(pg, ptr);
  return svreinterpret_f32_u32(svlsl_n_u32_x(pg, h, 16));
}

inline svuint32_t f32_to_bf16_bits_sve(svbool_t pg, svfloat32_t value) {
  const svuint32_t bits = svreinterpret_u32_f32(value);
  const svuint32_t lsb = svand_n_u32_x(pg, svlsr_n_u32_x(pg, bits, 16), 1);
  const svuint32_t bias = svadd_n_u32_x(pg, lsb, 0x7fff);
  return svlsr_n_u32_x(pg, svadd_u32_x(pg, bits, bias), 16);
}

inline void bf16_store_f32_sve(svbool_t pg, uint16_t* ptr, svfloat32_t value) {
  svst1h_u32(pg, ptr, f32_to_bf16_bits_sve(pg, value));
}

float sum_sq_bf16_sve(const uint16_t* row, int64_t dim) {
  const int64_t vl = static_cast<int64_t>(svcntw());
  const int64_t step = 4 * vl;
  const svbool_t pg_all = svptrue_b32();
  svfloat32_t acc0 = svdup_f32(0.0f);
  svfloat32_t acc1 = svdup_f32(0.0f);
  svfloat32_t acc2 = svdup_f32(0.0f);
  svfloat32_t acc3 = svdup_f32(0.0f);
  int64_t d = 0;
  for (; d + step <= dim; d += step) {
    svfloat32_t v = bf16_load_f32_sve(pg_all, row + d);
    acc0 = svmla_f32_x(pg_all, acc0, v, v);
    v = bf16_load_f32_sve(pg_all, row + d + vl);
    acc1 = svmla_f32_x(pg_all, acc1, v, v);
    v = bf16_load_f32_sve(pg_all, row + d + 2 * vl);
    acc2 = svmla_f32_x(pg_all, acc2, v, v);
    v = bf16_load_f32_sve(pg_all, row + d + 3 * vl);
    acc3 = svmla_f32_x(pg_all, acc3, v, v);
  }
  for (; d < dim; d += vl) {
    const svbool_t pg = svwhilelt_b32(d, dim);
    const svfloat32_t v = bf16_load_f32_sve(pg, row + d);
    acc0 = svmla_f32_m(pg, acc0, v, v);
  }
  acc0 = svadd_f32_x(pg_all, acc0, acc1);
  acc2 = svadd_f32_x(pg_all, acc2, acc3);
  const svfloat32_t acc = svadd_f32_x(pg_all, acc0, acc2);
  return svaddv_f32(pg_all, acc);
}

void rmsnorm_bf16_row_to_bf16_sve(const uint16_t* src, uint16_t* dst, const float* weight, int64_t dim, double eps) {
  const float inv_rms = 1.0f / std::sqrt(sum_sq_bf16_sve(src, dim) / static_cast<float>(dim) + static_cast<float>(eps));
  const int64_t vl = static_cast<int64_t>(svcntw());
  int64_t d = 0;
  for (; d < dim; d += vl) {
    const svbool_t pg = svwhilelt_b32(d, dim);
    const svfloat32_t x = bf16_load_f32_sve(pg, src + d);
    const svfloat32_t w = svld1_f32(pg, weight + d);
    const svfloat32_t y = svmul_f32_x(pg, svmul_n_f32_x(pg, x, inv_rms), w);
    bf16_store_f32_sve(pg, dst + d, y);
  }
}
#endif

[[maybe_unused]] float sum_sq_bf16_scalar(const uint16_t* row, int64_t dim) {
  float sum = 0.0f;
  for (int64_t i = 0; i < dim; ++i) {
    const float value = bf16_u16_to_float(row[i]);
    sum += value * value;
  }
  return sum;
}

[[maybe_unused]] void rmsnorm_bf16_row_to_bf16_scalar(const uint16_t* src, uint16_t* dst, const float* weight,
                                                      int64_t dim, double eps) {
  const float inv_rms =
      1.0f / std::sqrt(sum_sq_bf16_scalar(src, dim) / static_cast<float>(dim) + static_cast<float>(eps));
  for (int64_t i = 0; i < dim; ++i) {
    const float value = bf16_u16_to_float(src[i]) * inv_rms * weight[i];
    dst[i] = float_to_bf16_u16(value);
  }
}

void rmsnorm_bf16_row_to_bf16(const uint16_t* src, uint16_t* dst, const float* weight, int64_t dim, double eps) {
#if defined(__ARM_FEATURE_SVE)
  rmsnorm_bf16_row_to_bf16_sve(src, dst, weight, dim, eps);
#else
  rmsnorm_bf16_row_to_bf16_scalar(src, dst, weight, dim, eps);
#endif
}

void check_qkv_rmsnorm_args(const PackedWeight& fused_wqa_wkv, int64_t q_lora_rank, int64_t kv_dim, double eps) {
  check_int_arg(q_lora_rank, "q_lora_rank");
  check_int_arg(kv_dim, "kv_dim");
  TORCH_CHECK(fused_wqa_wkv.N == q_lora_rank + kv_dim, "fused_wqa_wkv N mismatch for q/kv RMSNorm: expected ",
              q_lora_rank + kv_dim, ", got ", fused_wqa_wkv.N);
  TORCH_CHECK(eps > 0.0, "RMSNorm eps must be positive, got ", eps);
}

void rmsnorm_qkv_from_qr_kv_ptr(const uint16_t* src, int64_t qr_kv_stride, uint16_t* qr_dst, uint16_t* kv_dst,
                                const float* q_weight, const float* kv_weight, int64_t q_lora_rank, int64_t kv_dim,
                                double eps, int64_t row_start, int64_t row_count) {
  for (int64_t row = 0; row < row_count; ++row) {
    const int64_t global_row = row_start + row;
    const uint16_t* row_src = src + row * qr_kv_stride;
    rmsnorm_bf16_row_to_bf16(row_src, qr_dst + global_row * q_lora_rank, q_weight, q_lora_rank, eps);
    rmsnorm_bf16_row_to_bf16(row_src + q_lora_rank, kv_dst + global_row * kv_dim, kv_weight, kv_dim, eps);
  }
}

void rmsnorm_qkv_from_qr_kv(const at::Tensor& qr_kv, int64_t qr_kv_stride, at::Tensor& qr, at::Tensor& kv,
                            const float* q_weight, const float* kv_weight, int64_t q_lora_rank, int64_t kv_dim,
                            double eps, int64_t row_start, int64_t row_count) {
  rmsnorm_qkv_from_qr_kv_ptr(bf16_data_const(qr_kv), qr_kv_stride, bf16_data(qr), bf16_data(kv), q_weight, kv_weight,
                             q_lora_rank, kv_dim, eps, row_start, row_count);
}

void check_hidden_states_for_attn_gemm(const at::Tensor& hidden_states, int64_t* M, int64_t* K) {
  check_bf16_cpu_2d(hidden_states, "hidden_states");
  *M = hidden_states.size(0);
  *K = hidden_states.size(1);
  check_int_arg(*K, "hidden_states.size(1)");
  TORCH_CHECK(*M >= 0, "hidden_states.size(0) must be non-negative");
  TORCH_CHECK(*M <= std::numeric_limits<int>::max(), "hidden_states.size(0) exceeds int32 kernel limit: ", *M);
}

void check_weight_k(const PackedWeight& weight, int64_t K, int64_t K_pad, const char* name) {
  TORCH_CHECK(weight.K == K, name, " K mismatch: hidden K=", K, ", weight K=", weight.K);
  TORCH_CHECK(weight.K_pad == K_pad, name, " padded K mismatch: expected ", K_pad, ", got ", weight.K_pad);
}

template <bool kRunCompressor, bool kRunIndexer>
void check_selected_weights(const PackedWeight& fused_wqa_wkv, const PackedWeight* compressor_kv_score,
                            const PackedWeight* indexer_compressor_kv_score, const PackedWeight* indexer_weights_proj,
                            int64_t K) {
  static_assert(!kRunIndexer || kRunCompressor, "indexer GEMMs require the compressor GEMM");
  check_weight_k(fused_wqa_wkv, K, fused_wqa_wkv.K_pad, "fused_wqa_wkv");
  if constexpr (kRunCompressor) {
    TORCH_CHECK(compressor_kv_score != nullptr, "compressor_kv_score weight is required");
    check_weight_k(*compressor_kv_score, K, fused_wqa_wkv.K_pad, "compressor_kv_score");
  }
  if constexpr (kRunIndexer) {
    TORCH_CHECK(indexer_compressor_kv_score != nullptr, "indexer_compressor_kv_score weight is required");
    TORCH_CHECK(indexer_weights_proj != nullptr, "indexer_weights_proj weight is required");
    check_weight_k(*indexer_compressor_kv_score, K, fused_wqa_wkv.K_pad, "indexer_compressor_kv_score");
    check_weight_k(*indexer_weights_proj, K, fused_wqa_wkv.K_pad, "indexer_weights_proj");
  }
}

at::Tensor make_padded_hidden_states(const at::Tensor& hidden_states, int64_t M, int64_t K, int64_t K_pad) {
  if (K_pad == K && hidden_states.is_contiguous()) {
    return hidden_states;
  }
  at::Tensor a_storage = at::zeros({M, K_pad}, hidden_states.options());
  a_storage.narrow(1, 0, K).copy_(hidden_states);
  return a_storage;
}

template <bool kRunCompressor, bool kRunIndexer>
AttnGemmSelectedOutputs run_attn_gemm_selected_serial(const at::Tensor& hidden_states,
                                                      const PackedWeight& fused_wqa_wkv,
                                                      const PackedWeight* compressor_kv_score,
                                                      const PackedWeight* indexer_compressor_kv_score,
                                                      const PackedWeight* indexer_weights_proj) {
  int64_t M = 0;
  int64_t K = 0;
  check_hidden_states_for_attn_gemm(hidden_states, &M, &K);
  check_selected_weights<kRunCompressor, kRunIndexer>(fused_wqa_wkv, compressor_kv_score, indexer_compressor_kv_score,
                                                      indexer_weights_proj, K);

  const int64_t K_pad = fused_wqa_wkv.K_pad;
  at::Tensor a_storage = make_padded_hidden_states(hidden_states, M, K, K_pad);
  auto workspace_lease = ::fused_cpp::workspace::acquire();
  const AttnGemmBackend backend = selected_attn_gemm_backend();
  const int64_t scratch_elems = backend == AttnGemmBackend::kSve
                                    ? ::fused_cpp::deepseek_v4::attn_sve::a_scratch_elems(M, K_pad)
                                    : std::max<int64_t>(M, 1) * K_pad * 2;
  at::Tensor scratch = workspace_lease.empty({std::max<int64_t>(1, scratch_elems)}, hidden_states.options());

  AttnGemmSelectedOutputs outputs;
  outputs.qr_kv = run_bf16_gemm(a_storage, fused_wqa_wkv, scratch);
  if constexpr (kRunCompressor) {
    outputs.kv_score = run_fp32_gemm(a_storage, *compressor_kv_score, scratch);
  }
  if constexpr (kRunIndexer) {
    outputs.indexer_kv_score = run_fp32_gemm(a_storage, *indexer_compressor_kv_score, scratch);
    outputs.indexer_weights = run_bf16_gemm(a_storage, *indexer_weights_proj, scratch);
  }
  return outputs;
}

template <bool kRunCompressor, bool kRunIndexer>
AttnGemmSelectedOutputs run_attn_gemm_selected_mt(const at::Tensor& hidden_states, const PackedWeight& fused_wqa_wkv,
                                                  const PackedWeight* compressor_kv_score,
                                                  const PackedWeight* indexer_compressor_kv_score,
                                                  const PackedWeight* indexer_weights_proj,
                                                  const std::vector<int64_t>& core_ids) {
  if (core_ids.empty()) {
    return run_attn_gemm_selected_serial<kRunCompressor, kRunIndexer>(
        hidden_states, fused_wqa_wkv, compressor_kv_score, indexer_compressor_kv_score, indexer_weights_proj);
  }

#ifndef _OPENMP
  TORCH_CHECK(core_ids.size() <= 1,
              "deepseek_v4 attn gemm fused mt requires OpenMP when more "
              "than one core is requested");
  return run_attn_gemm_selected_serial<kRunCompressor, kRunIndexer>(hidden_states, fused_wqa_wkv, compressor_kv_score,
                                                                    indexer_compressor_kv_score, indexer_weights_proj);
#else
  int64_t M = 0;
  int64_t K = 0;
  check_hidden_states_for_attn_gemm(hidden_states, &M, &K);
  TORCH_CHECK(core_ids.size() <= static_cast<size_t>(std::numeric_limits<int>::max()),
              "core_ids size exceeds int32 limit");
  for (int64_t core_id : core_ids) {
    TORCH_CHECK(core_id >= 0, "core_ids must be non-negative, got ", core_id);
  }
  check_selected_weights<kRunCompressor, kRunIndexer>(fused_wqa_wkv, compressor_kv_score, indexer_compressor_kv_score,
                                                      indexer_weights_proj, K);

  const int64_t K_pad = fused_wqa_wkv.K_pad;
  at::Tensor a_storage = make_padded_hidden_states(hidden_states, M, K, K_pad);

  const int64_t num_threads = static_cast<int64_t>(core_ids.size());
  const AttnGemmBackend backend = selected_attn_gemm_backend();
  const AttnGemmSchedule schedule = selected_attn_gemm_schedule();
  const bool uses_task_pool =
      schedule == AttnGemmSchedule::kSharedPool || schedule == AttnGemmSchedule::kMnPool;
  const int requested_n_groups =
      uses_task_pool ? requested_attn_gemm_n_groups(static_cast<int>(num_threads)) : 1;
  const int64_t rows_per_thread = ceil_div_int64(std::max<int64_t>(M, 1), num_threads);
  const int64_t m_panels = ceil_div_int64(M, kTile);
#if defined(__linux__)
  const bool use_prepacked_a =
      backend == AttnGemmBackend::kNeon && uses_task_pool &&
      attn_gemm_prepack_a_enabled(schedule, requested_n_groups);
#else
  const bool use_prepacked_a = false;
#endif
  const int64_t output_rows = use_prepacked_a ? m_panels * kTile : M;
  int64_t scratch_rows = rows_per_thread;
  if (schedule == AttnGemmSchedule::kM8Aligned) {
    scratch_rows = std::max<int64_t>(1, ceil_div_int64(m_panels, num_threads) * kTile);
  } else if (schedule == AttnGemmSchedule::kSharedPool || schedule == AttnGemmSchedule::kMnPool) {
    scratch_rows = kTile;
  }
  const int64_t scratch_stride =
      use_prepacked_a
          ? 1
          : (backend == AttnGemmBackend::kSve
                 ? ::fused_cpp::deepseek_v4::attn_sve::a_scratch_elems(scratch_rows, K_pad)
                 : std::max<int64_t>(1, scratch_rows * K_pad * 2));
  auto workspace_lease = ::fused_cpp::workspace::acquire();
  at::Tensor scratch = workspace_lease.empty({num_threads * scratch_stride}, hidden_states.options());
  at::Tensor packed_a;
  if (use_prepacked_a) {
    packed_a = workspace_lease.empty({output_rows * K_pad}, hidden_states.options());
  }

  AttnGemmSelectedOutputs outputs;
#if defined(__APPLE__)
  at::Tensor qr_kv_acc =
      at::zeros({output_rows, fused_wqa_wkv.N_pad},
                at::TensorOptions().device(hidden_states.device()).dtype(at::kFloat));
#else
  at::Tensor qr_kv_acc = at::empty({output_rows, fused_wqa_wkv.N_pad}, hidden_states.options());
#endif
  at::Tensor kv_score_acc;
  at::Tensor indexer_kv_score_acc;
  at::Tensor indexer_weights_acc;
  if constexpr (kRunCompressor) {
    kv_score_acc = at::zeros({output_rows, compressor_kv_score->N_pad},
                             at::TensorOptions().device(hidden_states.device()).dtype(at::kFloat));
  }
  if constexpr (kRunIndexer) {
    indexer_kv_score_acc = at::zeros({output_rows, indexer_compressor_kv_score->N_pad},
                                     at::TensorOptions().device(hidden_states.device()).dtype(at::kFloat));
#if defined(__APPLE__)
    indexer_weights_acc = at::zeros({output_rows, indexer_weights_proj->N_pad},
                                    at::TensorOptions().device(hidden_states.device()).dtype(at::kFloat));
#else
    indexer_weights_acc = at::empty({output_rows, indexer_weights_proj->N_pad}, hidden_states.options());
#endif
  }

  std::vector<AttnGemmWork> work;
#if defined(__APPLE__)
  work.push_back(AttnGemmWork{&fused_wqa_wkv, &qr_kv_acc, false});
#else
  work.push_back(AttnGemmWork{&fused_wqa_wkv, &qr_kv_acc, true});
#endif
  if constexpr (kRunCompressor) {
    work.push_back(AttnGemmWork{compressor_kv_score, &kv_score_acc, false});
  }
  if constexpr (kRunIndexer) {
    work.push_back(AttnGemmWork{indexer_compressor_kv_score, &indexer_kv_score_acc, false});
#if defined(__APPLE__)
    work.push_back(AttnGemmWork{indexer_weights_proj, &indexer_weights_acc, false});
#else
    work.push_back(AttnGemmWork{indexer_weights_proj, &indexer_weights_acc, true});
#endif
  }

  const int64_t n_tile = attn_gemm_n_tile(backend);
  std::vector<AttnGemmTaskGroup> task_groups;
  std::unique_ptr<AttnGemmTaskCursor[]> task_cursors;
  if (uses_task_pool) {
    task_groups = make_attn_gemm_task_groups(work, schedule, requested_n_groups, n_tile);
    task_cursors = std::make_unique<AttnGemmTaskCursor[]>(task_groups.size());
  }

  auto dispatch_work_range = [&](const AttnGemmWork& item, int64_t row_start, int64_t row_count, int64_t n_begin,
                                 int64_t n_cols, int64_t scratch_offset) {
    if (use_prepacked_a) {
      dispatch_packed_gemm_range_to_output(bf16_data_const(packed_a), *item.weight, *item.output, item.bf16_output,
                                           row_start, n_begin, n_cols);
      return;
    }
    if (item.bf16_output) {
      dispatch_bf16_gemm_range_to_output(a_storage, *item.weight, *item.output, scratch, row_start, row_count, n_begin,
                                         n_cols, scratch_offset);
    } else {
      dispatch_fp32_gemm_range_to_output(a_storage, *item.weight, *item.output, scratch, row_start, row_count, n_begin,
                                         n_cols, scratch_offset);
    }
  };

  std::vector<int> bind_failed(static_cast<size_t>(num_threads), 0);
  const int old_dynamic = omp_get_dynamic();
#if defined(__linux__)
  cpu_set_t caller_affinity;
  const bool restore_caller_affinity = get_current_thread_affinity(&caller_affinity);
#endif
  omp_set_dynamic(0);

#pragma omp parallel num_threads(num_threads)
  {
    const int tid = omp_get_thread_num();
    if (tid < static_cast<int>(num_threads) && !bind_current_thread_to_cpu(core_ids[static_cast<size_t>(tid)])) {
      bind_failed[static_cast<size_t>(tid)] = 1;
    }

    const int64_t scratch_offset = static_cast<int64_t>(tid) * scratch_stride;

    if (use_prepacked_a) {
      const int panel_begin = static_cast<int>(static_cast<int64_t>(tid) * m_panels / num_threads);
      const int panel_end = static_cast<int>(static_cast<int64_t>(tid + 1) * m_panels / num_threads);
      pack_a_reorder_m8_range(bf16_data_const(a_storage), bf16_data(packed_a), static_cast<int>(M),
                              static_cast<int>(K_pad), panel_begin, panel_end);
#pragma omp barrier
    }

    if (schedule == AttnGemmSchedule::kLegacy || schedule == AttnGemmSchedule::kM8Aligned) {
      int64_t row_start = static_cast<int64_t>(tid) * rows_per_thread;
      int64_t row_count = row_start >= M ? 0 : std::min<int64_t>(rows_per_thread, M - row_start);
      if (schedule == AttnGemmSchedule::kM8Aligned) {
        const int64_t panel_begin = static_cast<int64_t>(tid) * m_panels / num_threads;
        const int64_t panel_end = static_cast<int64_t>(tid + 1) * m_panels / num_threads;
        row_start = panel_begin * kTile;
        row_count = row_start >= M ? 0 : std::min<int64_t>((panel_end - panel_begin) * kTile, M - row_start);
      }
      for (const AttnGemmWork& item : work) {
        dispatch_work_range(item, row_start, row_count, 0, item.weight->N_pad, scratch_offset);
      }
    } else {
      const int group_count = static_cast<int>(task_groups.size());
      int preferred_group = tid % group_count;
      while (true) {
        bool executed = false;
        for (int offset = 0; offset < group_count; ++offset) {
          const int group_index = (preferred_group + offset) % group_count;
          AttnGemmTaskCursor& cursor = task_cursors[static_cast<size_t>(group_index)];
          if (cursor.next_panel.load(std::memory_order_relaxed) >= m_panels) {
            continue;
          }
          const int64_t panel = cursor.next_panel.fetch_add(1, std::memory_order_relaxed);
          if (panel >= m_panels) {
            continue;
          }
          const AttnGemmTaskGroup& group = task_groups[static_cast<size_t>(group_index)];
          const AttnGemmWork& item = work[static_cast<size_t>(group.work_index)];
          const int64_t row_start = panel * kTile;
          const int64_t row_count = std::min<int64_t>(kTile, M - row_start);
          dispatch_work_range(item, row_start, row_count, group.n_begin, group.n_cols, scratch_offset);
          preferred_group = group_index;
          executed = true;
          break;
        }
        if (!executed) {
          break;
        }
      }
    }
  }

  omp_set_dynamic(old_dynamic);
#if defined(__linux__)
  if (restore_caller_affinity) {
    TORCH_CHECK(set_current_thread_affinity(&caller_affinity),
                "failed to restore caller CPU affinity after "
                "deepseek_v4 attn GEMM fused mt");
  }
#endif
  for (size_t i = 0; i < bind_failed.size(); ++i) {
    TORCH_CHECK(bind_failed[i] == 0, "failed to bind OpenMP thread ", i, " to CPU ", core_ids[i]);
  }

#if defined(__APPLE__)
  at::Tensor qr_kv_rows = output_rows == M ? qr_kv_acc : qr_kv_acc.narrow(0, 0, M);
  outputs.qr_kv = narrow_output_if_needed(qr_kv_rows, fused_wqa_wkv.N, fused_wqa_wkv.N_pad).to(at::kBFloat16);
#else
  at::Tensor qr_kv_rows = output_rows == M ? qr_kv_acc : qr_kv_acc.narrow(0, 0, M);
  outputs.qr_kv = narrow_output_if_needed(qr_kv_rows, fused_wqa_wkv.N, fused_wqa_wkv.N_pad);
#endif
  if constexpr (kRunCompressor) {
    at::Tensor kv_score_rows = output_rows == M ? kv_score_acc : kv_score_acc.narrow(0, 0, M);
    outputs.kv_score =
        narrow_output_if_needed(kv_score_rows, compressor_kv_score->N, compressor_kv_score->N_pad);
  }
  if constexpr (kRunIndexer) {
    at::Tensor indexer_kv_score_rows =
        output_rows == M ? indexer_kv_score_acc : indexer_kv_score_acc.narrow(0, 0, M);
    outputs.indexer_kv_score = narrow_output_if_needed(indexer_kv_score_rows, indexer_compressor_kv_score->N,
                                                       indexer_compressor_kv_score->N_pad);
#if defined(__APPLE__)
    at::Tensor indexer_weights_rows =
        output_rows == M ? indexer_weights_acc : indexer_weights_acc.narrow(0, 0, M);
    outputs.indexer_weights =
        narrow_output_if_needed(indexer_weights_rows, indexer_weights_proj->N, indexer_weights_proj->N_pad)
            .to(at::kBFloat16);
#else
    at::Tensor indexer_weights_rows =
        output_rows == M ? indexer_weights_acc : indexer_weights_acc.narrow(0, 0, M);
    outputs.indexer_weights = narrow_output_if_needed(indexer_weights_rows, indexer_weights_proj->N,
                                                      indexer_weights_proj->N_pad);
#endif
  }
  return outputs;
#endif
}

template <bool kRunCompressor, bool kRunIndexer>
AttnGemmNormedOutputs run_attn_gemm_normed_serial(const at::Tensor& hidden_states, const PackedWeight& fused_wqa_wkv,
                                                  const PackedWeight* compressor_kv_score,
                                                  const PackedWeight* indexer_compressor_kv_score,
                                                  const PackedWeight* indexer_weights_proj, at::Tensor q_norm_weight,
                                                  at::Tensor kv_norm_weight, int64_t q_lora_rank, int64_t kv_dim,
                                                  double eps) {
  int64_t M = 0;
  int64_t K = 0;
  check_hidden_states_for_attn_gemm(hidden_states, &M, &K);
  check_qkv_rmsnorm_args(fused_wqa_wkv, q_lora_rank, kv_dim, eps);
  at::Tensor q_weight_f32 = checked_norm_weight_f32(q_norm_weight, q_lora_rank, "q_norm_weight");
  at::Tensor kv_weight_f32 = checked_norm_weight_f32(kv_norm_weight, kv_dim, "kv_norm_weight");

  AttnGemmSelectedOutputs selected = run_attn_gemm_selected_serial<kRunCompressor, kRunIndexer>(
      hidden_states, fused_wqa_wkv, compressor_kv_score, indexer_compressor_kv_score, indexer_weights_proj);

  AttnGemmNormedOutputs outputs;
  outputs.qr = at::empty({M, q_lora_rank}, hidden_states.options());
  outputs.kv = at::empty({M, kv_dim}, hidden_states.options());
  rmsnorm_qkv_from_qr_kv(selected.qr_kv, selected.qr_kv.size(1), outputs.qr, outputs.kv, q_weight_f32.data_ptr<float>(),
                         kv_weight_f32.data_ptr<float>(), q_lora_rank, kv_dim, eps, 0, M);
  if constexpr (kRunCompressor) {
    outputs.kv_score = selected.kv_score;
  }
  if constexpr (kRunIndexer) {
    outputs.indexer_kv_score = selected.indexer_kv_score;
    outputs.indexer_weights = selected.indexer_weights;
  }
  return outputs;
}

template <bool kRunCompressor, bool kRunIndexer>
AttnGemmNormedOutputs run_attn_gemm_normed_mt(const at::Tensor& hidden_states, const PackedWeight& fused_wqa_wkv,
                                              const PackedWeight* compressor_kv_score,
                                              const PackedWeight* indexer_compressor_kv_score,
                                              const PackedWeight* indexer_weights_proj, at::Tensor q_norm_weight,
                                              at::Tensor kv_norm_weight, int64_t q_lora_rank, int64_t kv_dim,
                                              double eps, const std::vector<int64_t>& core_ids) {
  if (core_ids.empty()) {
    return run_attn_gemm_normed_serial<kRunCompressor, kRunIndexer>(
        hidden_states, fused_wqa_wkv, compressor_kv_score, indexer_compressor_kv_score, indexer_weights_proj,
        q_norm_weight, kv_norm_weight, q_lora_rank, kv_dim, eps);
  }

#if !defined(_OPENMP) || !defined(__linux__)
  return run_attn_gemm_normed_serial<kRunCompressor, kRunIndexer>(
      hidden_states, fused_wqa_wkv, compressor_kv_score, indexer_compressor_kv_score, indexer_weights_proj,
      q_norm_weight, kv_norm_weight, q_lora_rank, kv_dim, eps);
#else
  int64_t M = 0;
  int64_t K = 0;
  check_hidden_states_for_attn_gemm(hidden_states, &M, &K);
  TORCH_CHECK(core_ids.size() <= static_cast<size_t>(std::numeric_limits<int>::max()),
              "core_ids size exceeds int32 limit");
  for (int64_t core_id : core_ids) {
    TORCH_CHECK(core_id >= 0, "core_ids must be non-negative, got ", core_id);
  }
  check_selected_weights<kRunCompressor, kRunIndexer>(fused_wqa_wkv, compressor_kv_score, indexer_compressor_kv_score,
                                                      indexer_weights_proj, K);
  check_qkv_rmsnorm_args(fused_wqa_wkv, q_lora_rank, kv_dim, eps);
  at::Tensor q_weight_f32 = checked_norm_weight_f32(q_norm_weight, q_lora_rank, "q_norm_weight");
  at::Tensor kv_weight_f32 = checked_norm_weight_f32(kv_norm_weight, kv_dim, "kv_norm_weight");

  const int64_t K_pad = fused_wqa_wkv.K_pad;
  at::Tensor a_storage = make_padded_hidden_states(hidden_states, M, K, K_pad);

  const int64_t num_threads = static_cast<int64_t>(core_ids.size());
  const int64_t rows_per_thread = ceil_div_int64(std::max<int64_t>(M, 1), num_threads);
  const AttnGemmBackend backend = selected_attn_gemm_backend();
  const int64_t scratch_stride = backend == AttnGemmBackend::kSve
                                     ? ::fused_cpp::deepseek_v4::attn_sve::a_scratch_elems(rows_per_thread, K_pad)
                                     : std::max<int64_t>(1, rows_per_thread * K_pad * 2);
  auto workspace_lease = ::fused_cpp::workspace::acquire();
  at::Tensor scratch = workspace_lease.empty({num_threads * scratch_stride}, hidden_states.options());
  at::Tensor qr_kv_tmp =
      workspace_lease.empty({num_threads, rows_per_thread, fused_wqa_wkv.N_pad}, hidden_states.options());

  AttnGemmNormedOutputs outputs;
  outputs.qr = at::empty({M, q_lora_rank}, hidden_states.options());
  outputs.kv = at::empty({M, kv_dim}, hidden_states.options());
  at::Tensor kv_score_acc;
  at::Tensor indexer_kv_score_acc;
  at::Tensor indexer_weights_acc;
  if constexpr (kRunCompressor) {
    kv_score_acc = at::zeros({M, compressor_kv_score->N_pad},
                             at::TensorOptions().device(hidden_states.device()).dtype(at::kFloat));
  }
  if constexpr (kRunIndexer) {
    indexer_kv_score_acc = at::zeros({M, indexer_compressor_kv_score->N_pad},
                                     at::TensorOptions().device(hidden_states.device()).dtype(at::kFloat));
    indexer_weights_acc = at::empty({M, indexer_weights_proj->N_pad}, hidden_states.options());
  }

  const uint16_t* a_ptr = bf16_data_const(a_storage);
  const uint16_t* fused_weight_ptr = bf16_data_const(fused_wqa_wkv.tensor);
  uint16_t* scratch_ptr = bf16_data(scratch);
  uint16_t* tmp_ptr = bf16_data(qr_kv_tmp);
  uint16_t* qr_ptr = bf16_data(outputs.qr);
  uint16_t* kv_ptr = bf16_data(outputs.kv);
  const float* q_weight_ptr = q_weight_f32.data_ptr<float>();
  const float* kv_weight_ptr = kv_weight_f32.data_ptr<float>();

  std::vector<int> bind_failed(static_cast<size_t>(num_threads), 0);
  const int old_dynamic = omp_get_dynamic();
#if defined(__linux__)
  cpu_set_t caller_affinity;
  const bool restore_caller_affinity = get_current_thread_affinity(&caller_affinity);
#endif
  omp_set_dynamic(0);

#pragma omp parallel num_threads(num_threads)
  {
    const int tid = omp_get_thread_num();
    if (tid < static_cast<int>(num_threads) && !bind_current_thread_to_cpu(core_ids[static_cast<size_t>(tid)])) {
      bind_failed[static_cast<size_t>(tid)] = 1;
    }

    const int64_t row_start = static_cast<int64_t>(tid) * rows_per_thread;
    const int64_t row_count = row_start >= M ? 0 : std::min<int64_t>(rows_per_thread, M - row_start);
    const int64_t scratch_offset = static_cast<int64_t>(tid) * scratch_stride;
    uint16_t* thread_tmp = tmp_ptr + static_cast<int64_t>(tid) * rows_per_thread * fused_wqa_wkv.N_pad;

    if (row_count > 0) {
      dispatch_bf16_nld_gemm(a_ptr + row_start * K_pad, fused_weight_ptr, thread_tmp, scratch_ptr + scratch_offset,
                             static_cast<int>(row_count), static_cast<int>(K_pad),
                             static_cast<int>(fused_wqa_wkv.N_pad), static_cast<int>(fused_wqa_wkv.N_pad));
      rmsnorm_qkv_from_qr_kv_ptr(thread_tmp, fused_wqa_wkv.N_pad, qr_ptr, kv_ptr, q_weight_ptr, kv_weight_ptr,
                                 q_lora_rank, kv_dim, eps, row_start, row_count);
    }

    if constexpr (kRunCompressor) {
      dispatch_fp32_gemm_to_output(a_storage, *compressor_kv_score, kv_score_acc, scratch, row_start, row_count,
                                   scratch_offset);
    }
    if constexpr (kRunIndexer) {
      dispatch_fp32_gemm_to_output(a_storage, *indexer_compressor_kv_score, indexer_kv_score_acc, scratch, row_start,
                                   row_count, scratch_offset);
      dispatch_bf16_gemm_to_output(a_storage, *indexer_weights_proj, indexer_weights_acc, scratch, row_start, row_count,
                                   scratch_offset);
    }
  }

  omp_set_dynamic(old_dynamic);
#if defined(__linux__)
  if (restore_caller_affinity) {
    TORCH_CHECK(set_current_thread_affinity(&caller_affinity),
                "failed to restore caller CPU affinity after "
                "deepseek_v4 attn GEMM+RMSNorm fused mt");
  }
#endif
  for (size_t i = 0; i < bind_failed.size(); ++i) {
    TORCH_CHECK(bind_failed[i] == 0, "failed to bind OpenMP thread ", i, " to CPU ", core_ids[i]);
  }

  if constexpr (kRunCompressor) {
    outputs.kv_score = narrow_output_if_needed(kv_score_acc, compressor_kv_score->N, compressor_kv_score->N_pad);
  }
  if constexpr (kRunIndexer) {
    outputs.indexer_kv_score = narrow_output_if_needed(indexer_kv_score_acc, indexer_compressor_kv_score->N,
                                                       indexer_compressor_kv_score->N_pad);
    outputs.indexer_weights =
        narrow_output_if_needed(indexer_weights_acc, indexer_weights_proj->N, indexer_weights_proj->N_pad);
  }
  return outputs;
#endif
}

#endif

}  // namespace

std::tuple<at::Tensor, int64_t, int64_t>
fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepare(at::Tensor weight) {
#ifndef __aarch64__
  TORCH_CHECK(false, "deepseek_v4 attn gemm fused prepare requires AArch64");
#else
  check_bf16_cpu_2d(weight, "weight");
  const int64_t K = weight.size(0);
  const int64_t N = weight.size(1);
  check_int_arg(K, "K");
  check_int_arg(N, "N");
  const AttnGemmBackend backend = selected_attn_gemm_backend();
  const int64_t K_pad = attn_gemm_round_k(K, backend);
  const int64_t N_pad = attn_gemm_round_n(N, backend);

  at::Tensor weight_padded;
  if (K_pad == K && N_pad == N && weight.is_contiguous()) {
    weight_padded = weight;
  } else {
    weight_padded = at::zeros({K_pad, N_pad}, weight.options());
    weight_padded.narrow(0, 0, K).narrow(1, 0, N).copy_(weight);
  }

  at::Tensor packed = at::empty({K_pad * N_pad}, weight.options());
  if (backend == AttnGemmBackend::kSve) {
    ::fused_cpp::deepseek_v4::attn_sve::pack_b(bf16_data_const(weight_padded), bf16_data(packed),
                                               static_cast<int>(K_pad), static_cast<int>(N_pad));
  } else {
    bf16_pack_b(bf16_data_const(weight_padded), bf16_data(packed), static_cast<int>(K_pad), static_cast<int>(N_pad));
  }
  return std::make_tuple(packed, K, N);
#endif
}

at::Tensor fused_wqa_wkv_fused(at::Tensor hidden_states, at::Tensor fused_wqa_wkv_packed, int64_t fused_wqa_wkv_K,
                               int64_t fused_wqa_wkv_N) {
#ifndef __aarch64__
  TORCH_CHECK(false, "deepseek_v4 attn dense gemm fused requires AArch64");
#else
  PackedWeight fused_wqa_wkv =
      checked_packed_weight(fused_wqa_wkv_packed, fused_wqa_wkv_K, fused_wqa_wkv_N, "fused_wqa_wkv");
  AttnGemmSelectedOutputs outputs =
      run_attn_gemm_selected_serial<false, false>(hidden_states, fused_wqa_wkv, nullptr, nullptr, nullptr);
  return outputs.qr_kv;
#endif
}

at::Tensor fused_wqa_wkv_fused_mt(at::Tensor hidden_states, at::Tensor fused_wqa_wkv_packed, int64_t fused_wqa_wkv_K,
                                  int64_t fused_wqa_wkv_N, std::vector<int64_t> core_ids) {
#ifndef __aarch64__
  TORCH_CHECK(false, "deepseek_v4 attn dense gemm fused requires AArch64");
#else
  PackedWeight fused_wqa_wkv =
      checked_packed_weight(fused_wqa_wkv_packed, fused_wqa_wkv_K, fused_wqa_wkv_N, "fused_wqa_wkv");
  AttnGemmSelectedOutputs outputs =
      run_attn_gemm_selected_mt<false, false>(hidden_states, fused_wqa_wkv, nullptr, nullptr, nullptr, core_ids);
  return outputs.qr_kv;
#endif
}

std::tuple<at::Tensor, at::Tensor> fused_wqa_wkv_compressor_kv_score_fused(
    at::Tensor hidden_states, at::Tensor fused_wqa_wkv_packed, int64_t fused_wqa_wkv_K, int64_t fused_wqa_wkv_N,
    at::Tensor compressor_kv_score_packed, int64_t compressor_kv_score_K, int64_t compressor_kv_score_N) {
#ifndef __aarch64__
  TORCH_CHECK(false, "deepseek_v4 attn C128A gemm fused requires AArch64");
#else
  PackedWeight fused_wqa_wkv =
      checked_packed_weight(fused_wqa_wkv_packed, fused_wqa_wkv_K, fused_wqa_wkv_N, "fused_wqa_wkv");
  PackedWeight compressor_kv_score = checked_packed_weight(compressor_kv_score_packed, compressor_kv_score_K,
                                                           compressor_kv_score_N, "compressor_kv_score");
  AttnGemmSelectedOutputs outputs =
      run_attn_gemm_selected_serial<true, false>(hidden_states, fused_wqa_wkv, &compressor_kv_score, nullptr, nullptr);
  return std::make_tuple(outputs.qr_kv, outputs.kv_score);
#endif
}

std::tuple<at::Tensor, at::Tensor> fused_wqa_wkv_compressor_kv_score_fused_mt(
    at::Tensor hidden_states, at::Tensor fused_wqa_wkv_packed, int64_t fused_wqa_wkv_K, int64_t fused_wqa_wkv_N,
    at::Tensor compressor_kv_score_packed, int64_t compressor_kv_score_K, int64_t compressor_kv_score_N,
    std::vector<int64_t> core_ids) {
#ifndef __aarch64__
  TORCH_CHECK(false, "deepseek_v4 attn C128A gemm fused requires AArch64");
#else
  PackedWeight fused_wqa_wkv =
      checked_packed_weight(fused_wqa_wkv_packed, fused_wqa_wkv_K, fused_wqa_wkv_N, "fused_wqa_wkv");
  PackedWeight compressor_kv_score = checked_packed_weight(compressor_kv_score_packed, compressor_kv_score_K,
                                                           compressor_kv_score_N, "compressor_kv_score");
  AttnGemmSelectedOutputs outputs = run_attn_gemm_selected_mt<true, false>(
      hidden_states, fused_wqa_wkv, &compressor_kv_score, nullptr, nullptr, core_ids);
  return std::make_tuple(outputs.qr_kv, outputs.kv_score);
#endif
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused(
    at::Tensor hidden_states, at::Tensor fused_wqa_wkv_packed, int64_t fused_wqa_wkv_K, int64_t fused_wqa_wkv_N,
    at::Tensor compressor_kv_score_packed, int64_t compressor_kv_score_K, int64_t compressor_kv_score_N,
    at::Tensor indexer_compressor_kv_score_packed, int64_t indexer_compressor_kv_score_K,
    int64_t indexer_compressor_kv_score_N, at::Tensor indexer_weights_proj_packed, int64_t indexer_weights_proj_K,
    int64_t indexer_weights_proj_N) {
#ifndef __aarch64__
  TORCH_CHECK(false, "deepseek_v4 attn gemm fused requires AArch64");
#else
  PackedWeight fused_wqa_wkv =
      checked_packed_weight(fused_wqa_wkv_packed, fused_wqa_wkv_K, fused_wqa_wkv_N, "fused_wqa_wkv");
  PackedWeight compressor_kv_score = checked_packed_weight(compressor_kv_score_packed, compressor_kv_score_K,
                                                           compressor_kv_score_N, "compressor_kv_score");
  PackedWeight indexer_compressor_kv_score =
      checked_packed_weight(indexer_compressor_kv_score_packed, indexer_compressor_kv_score_K,
                            indexer_compressor_kv_score_N, "indexer_compressor_kv_score");
  PackedWeight indexer_weights_proj = checked_packed_weight(indexer_weights_proj_packed, indexer_weights_proj_K,
                                                            indexer_weights_proj_N, "indexer_weights_proj");
  AttnGemmSelectedOutputs outputs = run_attn_gemm_selected_serial<true, true>(
      hidden_states, fused_wqa_wkv, &compressor_kv_score, &indexer_compressor_kv_score, &indexer_weights_proj);
  return std::make_tuple(outputs.qr_kv, outputs.kv_score, outputs.indexer_kv_score, outputs.indexer_weights);
#endif
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_mt(
    at::Tensor hidden_states, at::Tensor fused_wqa_wkv_packed, int64_t fused_wqa_wkv_K, int64_t fused_wqa_wkv_N,
    at::Tensor compressor_kv_score_packed, int64_t compressor_kv_score_K, int64_t compressor_kv_score_N,
    at::Tensor indexer_compressor_kv_score_packed, int64_t indexer_compressor_kv_score_K,
    int64_t indexer_compressor_kv_score_N, at::Tensor indexer_weights_proj_packed, int64_t indexer_weights_proj_K,
    int64_t indexer_weights_proj_N, std::vector<int64_t> core_ids) {
#ifndef __aarch64__
  TORCH_CHECK(false, "deepseek_v4 attn gemm fused requires AArch64");
#else
  PackedWeight fused_wqa_wkv =
      checked_packed_weight(fused_wqa_wkv_packed, fused_wqa_wkv_K, fused_wqa_wkv_N, "fused_wqa_wkv");
  PackedWeight compressor_kv_score = checked_packed_weight(compressor_kv_score_packed, compressor_kv_score_K,
                                                           compressor_kv_score_N, "compressor_kv_score");
  PackedWeight indexer_compressor_kv_score =
      checked_packed_weight(indexer_compressor_kv_score_packed, indexer_compressor_kv_score_K,
                            indexer_compressor_kv_score_N, "indexer_compressor_kv_score");
  PackedWeight indexer_weights_proj = checked_packed_weight(indexer_weights_proj_packed, indexer_weights_proj_K,
                                                            indexer_weights_proj_N, "indexer_weights_proj");
  AttnGemmSelectedOutputs outputs =
      run_attn_gemm_selected_mt<true, true>(hidden_states, fused_wqa_wkv, &compressor_kv_score,
                                            &indexer_compressor_kv_score, &indexer_weights_proj, core_ids);
  return std::make_tuple(outputs.qr_kv, outputs.kv_score, outputs.indexer_kv_score, outputs.indexer_weights);
#endif
}

std::tuple<at::Tensor, at::Tensor> fused_wqa_wkv_qkv_rmsnorm_fused(at::Tensor hidden_states,
                                                                   at::Tensor fused_wqa_wkv_packed,
                                                                   int64_t fused_wqa_wkv_K, int64_t fused_wqa_wkv_N,
                                                                   at::Tensor q_norm_weight, at::Tensor kv_norm_weight,
                                                                   int64_t q_lora_rank, int64_t kv_dim, double eps) {
#ifndef __aarch64__
  TORCH_CHECK(false, "deepseek_v4 attn dense GEMM+RMSNorm fused requires AArch64");
#else
  PackedWeight fused_wqa_wkv =
      checked_packed_weight(fused_wqa_wkv_packed, fused_wqa_wkv_K, fused_wqa_wkv_N, "fused_wqa_wkv");
  AttnGemmNormedOutputs outputs = run_attn_gemm_normed_serial<false, false>(
      hidden_states, fused_wqa_wkv, nullptr, nullptr, nullptr, q_norm_weight, kv_norm_weight, q_lora_rank, kv_dim, eps);
  return std::make_tuple(outputs.qr, outputs.kv);
#endif
}

std::tuple<at::Tensor, at::Tensor> fused_wqa_wkv_qkv_rmsnorm_fused_mt(
    at::Tensor hidden_states, at::Tensor fused_wqa_wkv_packed, int64_t fused_wqa_wkv_K, int64_t fused_wqa_wkv_N,
    at::Tensor q_norm_weight, at::Tensor kv_norm_weight, int64_t q_lora_rank, int64_t kv_dim, double eps,
    std::vector<int64_t> core_ids) {
#ifndef __aarch64__
  TORCH_CHECK(false, "deepseek_v4 attn dense GEMM+RMSNorm fused requires AArch64");
#else
  PackedWeight fused_wqa_wkv =
      checked_packed_weight(fused_wqa_wkv_packed, fused_wqa_wkv_K, fused_wqa_wkv_N, "fused_wqa_wkv");
  AttnGemmNormedOutputs outputs =
      run_attn_gemm_normed_mt<false, false>(hidden_states, fused_wqa_wkv, nullptr, nullptr, nullptr, q_norm_weight,
                                            kv_norm_weight, q_lora_rank, kv_dim, eps, core_ids);
  return std::make_tuple(outputs.qr, outputs.kv);
#endif
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_wqa_wkv_compressor_kv_score_qkv_rmsnorm_fused(
    at::Tensor hidden_states, at::Tensor fused_wqa_wkv_packed, int64_t fused_wqa_wkv_K, int64_t fused_wqa_wkv_N,
    at::Tensor compressor_kv_score_packed, int64_t compressor_kv_score_K, int64_t compressor_kv_score_N,
    at::Tensor q_norm_weight, at::Tensor kv_norm_weight, int64_t q_lora_rank, int64_t kv_dim, double eps) {
#ifndef __aarch64__
  TORCH_CHECK(false, "deepseek_v4 attn C128A GEMM+RMSNorm fused requires AArch64");
#else
  PackedWeight fused_wqa_wkv =
      checked_packed_weight(fused_wqa_wkv_packed, fused_wqa_wkv_K, fused_wqa_wkv_N, "fused_wqa_wkv");
  PackedWeight compressor_kv_score = checked_packed_weight(compressor_kv_score_packed, compressor_kv_score_K,
                                                           compressor_kv_score_N, "compressor_kv_score");
  AttnGemmNormedOutputs outputs =
      run_attn_gemm_normed_serial<true, false>(hidden_states, fused_wqa_wkv, &compressor_kv_score, nullptr, nullptr,
                                               q_norm_weight, kv_norm_weight, q_lora_rank, kv_dim, eps);
  return std::make_tuple(outputs.qr, outputs.kv, outputs.kv_score);
#endif
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_wqa_wkv_compressor_kv_score_qkv_rmsnorm_fused_mt(
    at::Tensor hidden_states, at::Tensor fused_wqa_wkv_packed, int64_t fused_wqa_wkv_K, int64_t fused_wqa_wkv_N,
    at::Tensor compressor_kv_score_packed, int64_t compressor_kv_score_K, int64_t compressor_kv_score_N,
    at::Tensor q_norm_weight, at::Tensor kv_norm_weight, int64_t q_lora_rank, int64_t kv_dim, double eps,
    std::vector<int64_t> core_ids) {
#ifndef __aarch64__
  TORCH_CHECK(false, "deepseek_v4 attn C128A GEMM+RMSNorm fused requires AArch64");
#else
  PackedWeight fused_wqa_wkv =
      checked_packed_weight(fused_wqa_wkv_packed, fused_wqa_wkv_K, fused_wqa_wkv_N, "fused_wqa_wkv");
  PackedWeight compressor_kv_score = checked_packed_weight(compressor_kv_score_packed, compressor_kv_score_K,
                                                           compressor_kv_score_N, "compressor_kv_score");
  AttnGemmNormedOutputs outputs =
      run_attn_gemm_normed_mt<true, false>(hidden_states, fused_wqa_wkv, &compressor_kv_score, nullptr, nullptr,
                                           q_norm_weight, kv_norm_weight, q_lora_rank, kv_dim, eps, core_ids);
  return std::make_tuple(outputs.qr, outputs.kv, outputs.kv_score);
#endif
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_qkv_rmsnorm_fused(
    at::Tensor hidden_states, at::Tensor fused_wqa_wkv_packed, int64_t fused_wqa_wkv_K, int64_t fused_wqa_wkv_N,
    at::Tensor compressor_kv_score_packed, int64_t compressor_kv_score_K, int64_t compressor_kv_score_N,
    at::Tensor indexer_compressor_kv_score_packed, int64_t indexer_compressor_kv_score_K,
    int64_t indexer_compressor_kv_score_N, at::Tensor indexer_weights_proj_packed, int64_t indexer_weights_proj_K,
    int64_t indexer_weights_proj_N, at::Tensor q_norm_weight, at::Tensor kv_norm_weight, int64_t q_lora_rank,
    int64_t kv_dim, double eps) {
#ifndef __aarch64__
  TORCH_CHECK(false, "deepseek_v4 attn C4A GEMM+RMSNorm fused requires AArch64");
#else
  PackedWeight fused_wqa_wkv =
      checked_packed_weight(fused_wqa_wkv_packed, fused_wqa_wkv_K, fused_wqa_wkv_N, "fused_wqa_wkv");
  PackedWeight compressor_kv_score = checked_packed_weight(compressor_kv_score_packed, compressor_kv_score_K,
                                                           compressor_kv_score_N, "compressor_kv_score");
  PackedWeight indexer_compressor_kv_score =
      checked_packed_weight(indexer_compressor_kv_score_packed, indexer_compressor_kv_score_K,
                            indexer_compressor_kv_score_N, "indexer_compressor_kv_score");
  PackedWeight indexer_weights_proj = checked_packed_weight(indexer_weights_proj_packed, indexer_weights_proj_K,
                                                            indexer_weights_proj_N, "indexer_weights_proj");
  AttnGemmNormedOutputs outputs = run_attn_gemm_normed_serial<true, true>(
      hidden_states, fused_wqa_wkv, &compressor_kv_score, &indexer_compressor_kv_score, &indexer_weights_proj,
      q_norm_weight, kv_norm_weight, q_lora_rank, kv_dim, eps);
  return std::make_tuple(outputs.qr, outputs.kv, outputs.kv_score, outputs.indexer_kv_score, outputs.indexer_weights);
#endif
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_qkv_rmsnorm_fused_mt(
    at::Tensor hidden_states, at::Tensor fused_wqa_wkv_packed, int64_t fused_wqa_wkv_K, int64_t fused_wqa_wkv_N,
    at::Tensor compressor_kv_score_packed, int64_t compressor_kv_score_K, int64_t compressor_kv_score_N,
    at::Tensor indexer_compressor_kv_score_packed, int64_t indexer_compressor_kv_score_K,
    int64_t indexer_compressor_kv_score_N, at::Tensor indexer_weights_proj_packed, int64_t indexer_weights_proj_K,
    int64_t indexer_weights_proj_N, at::Tensor q_norm_weight, at::Tensor kv_norm_weight, int64_t q_lora_rank,
    int64_t kv_dim, double eps, std::vector<int64_t> core_ids) {
#ifndef __aarch64__
  TORCH_CHECK(false, "deepseek_v4 attn C4A GEMM+RMSNorm fused requires AArch64");
#else
  PackedWeight fused_wqa_wkv =
      checked_packed_weight(fused_wqa_wkv_packed, fused_wqa_wkv_K, fused_wqa_wkv_N, "fused_wqa_wkv");
  PackedWeight compressor_kv_score = checked_packed_weight(compressor_kv_score_packed, compressor_kv_score_K,
                                                           compressor_kv_score_N, "compressor_kv_score");
  PackedWeight indexer_compressor_kv_score =
      checked_packed_weight(indexer_compressor_kv_score_packed, indexer_compressor_kv_score_K,
                            indexer_compressor_kv_score_N, "indexer_compressor_kv_score");
  PackedWeight indexer_weights_proj = checked_packed_weight(indexer_weights_proj_packed, indexer_weights_proj_K,
                                                            indexer_weights_proj_N, "indexer_weights_proj");
  AttnGemmNormedOutputs outputs = run_attn_gemm_normed_mt<true, true>(
      hidden_states, fused_wqa_wkv, &compressor_kv_score, &indexer_compressor_kv_score, &indexer_weights_proj,
      q_norm_weight, kv_norm_weight, q_lora_rank, kv_dim, eps, core_ids);
  return std::make_tuple(outputs.qr, outputs.kv, outputs.kv_score, outputs.indexer_kv_score, outputs.indexer_weights);
#endif
}
