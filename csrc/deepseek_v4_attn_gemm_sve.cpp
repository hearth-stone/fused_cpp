#include "deepseek_v4_attn_gemm_sve.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "moe/arm/sve_bf16/jit_kernels.h"
#include "moe/arm/sve_bf16/packing.h"

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && \
    (defined(__ARM_FEATURE_BF16) || defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC))
#include <arm_sve.h>
#endif

namespace fused_cpp::deepseek_v4::attn_sve {
namespace {

bool env_true(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  return value[0] != '\0' && value[0] != '0' && std::strcmp(value, "false") != 0 && std::strcmp(value, "False") != 0 &&
         std::strcmp(value, "off") != 0 && std::strcmp(value, "OFF") != 0;
}

#if defined(FUSED_CPP_DEEPSEEK_V4_HAS_SVE_JIT_GEMM)
using KernelFn = ::fused_cpp::moe_sve::jit::KernelFn;

struct alignas(8) SveParams {
  gemm_params_t gemm{};
  int32_t kc = 0;
  int32_t packed_n = 0;
  int32_t n_begin = 0;
  int32_t mode = 0;
  float* partial_c = nullptr;
};

static_assert(sizeof(gemm_params_t) == 24, "unexpected GEMM params ABI");
static_assert(offsetof(SveParams, n_begin) == 32, "unexpected SVE N-begin params ABI");

void pack_a_panel(const uint16_t* input, uint16_t* packed, int rows, int physical_rows, int K) {
  for (int kb = 0; kb < K; kb += 4) {
    uint16_t* block = packed + static_cast<int64_t>(kb / 4) * physical_rows * 4;
    for (int row = 0; row < physical_rows; ++row) {
      uint16_t* destination = block + row * 4;
      if (row < rows) {
        const uint16_t* source = input + static_cast<int64_t>(row) * K + kb;
        destination[0] = source[0];
        destination[1] = source[1];
        destination[2] = source[2];
        destination[3] = source[3];
      } else {
        destination[0] = 0;
        destination[1] = 0;
        destination[2] = 0;
        destination[3] = 0;
      }
    }
  }
}

KernelFn get_jit_kernel(int rows, bool output_bf16) {
  std::string error;
  KernelFn kernel = output_bf16 ? ::fused_cpp::moe_sve::jit::get_gemm_bf16_kernel(rows, &error)
                                : ::fused_cpp::moe_sve::jit::get_gemm_f32_kernel(rows, &error);
  if (kernel == nullptr) {
    throw std::runtime_error("failed to generate DeepSeek V4 SVE M" + std::to_string(rows) + " GEMM: " + error);
  }
  return kernel;
}

template <typename Output>
void dispatch_packed_jit(const uint16_t* packed_A, const uint16_t* B_reo, Output* C, int M, int K, int N, int ldc,
                         bool output_bf16) {
  KernelFn kernel = get_jit_kernel(M, output_bf16);
  SveParams params;
  params.gemm.m = M;
  params.gemm.k = K;
  params.gemm.n = N;
  params.gemm.lda = K;
  params.gemm.ldb = K;
  params.gemm.ldc = ldc;
  params.kc = K;
  params.packed_n = N;
  kernel(packed_A, B_reo, C, nullptr, &params.gemm);
}

template <typename Output>
void dispatch_jit(const uint16_t* A, const uint16_t* B_reo, Output* C, uint16_t* A_reorder, int M, int K, int N,
                  int ldc, bool output_bf16) {
  KernelFn kernels[kMPanelRows + 1] = {};
  int processed = 0;
  while (processed < M) {
    const int rows = std::min(kMPanelRows, M - processed);
    const int physical_rows = rows <= 8 ? 8 : 12;
    pack_a_panel(A + static_cast<int64_t>(processed) * K, A_reorder, rows, physical_rows, K);

    if (kernels[rows] == nullptr) {
      kernels[rows] = get_jit_kernel(rows, output_bf16);
    }

    SveParams params;
    params.gemm.m = rows;
    params.gemm.k = K;
    params.gemm.n = N;
    params.gemm.lda = K;
    params.gemm.ldb = K;
    params.gemm.ldc = ldc;
    params.kc = K;
    params.packed_n = N;
    kernels[rows](A_reorder, B_reo, C + static_cast<int64_t>(processed) * ldc, nullptr, &params.gemm);
    processed += rows;
  }
}
#endif

[[maybe_unused]] int round_up(int value, int quantum) { return ((value + quantum - 1) / quantum) * quantum; }

}  // namespace

bool available() {
#if defined(FUSED_CPP_DEEPSEEK_V4_HAS_SVE_JIT_GEMM) && defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && \
    (defined(__ARM_FEATURE_BF16) || defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC))
  return ::fused_cpp::moe_sve::available() && ::fused_cpp::moe_sve::jit::built() &&
         ::fused_cpp::moe_sve::n_tile() == static_cast<int>(svcntb() / 2);
#else
  return false;
#endif
}

bool enabled_by_env() { return available() && env_true("FUSED_CPP_ATTN_GEMM_SVE"); }

int m_panel_rows() { return kMPanelRows; }

int n_tile() {
#if defined(FUSED_CPP_DEEPSEEK_V4_HAS_SVE_JIT_GEMM)
  return ::fused_cpp::moe_sve::n_tile();
#else
  return 8;
#endif
}

int round_k(int k) {
#if defined(FUSED_CPP_DEEPSEEK_V4_HAS_SVE_JIT_GEMM)
  return ::fused_cpp::moe_sve::round_k(k);
#else
  return round_up(std::max(k, 8), 8);
#endif
}

int round_n(int n) {
#if defined(FUSED_CPP_DEEPSEEK_V4_HAS_SVE_JIT_GEMM)
  return ::fused_cpp::moe_sve::round_n(n);
#else
  return round_up(std::max(n, 8), 8);
#endif
}

int64_t a_scratch_elems(int64_t m, int64_t k) {
  const int64_t physical_rows = m > 8 ? 12 : 8;
  return std::max<int64_t>(1, physical_rows * k);
}

int64_t packed_a_elems(int64_t m, int64_t k) {
  return std::max<int64_t>(1, ((m + kMPanelRows - 1) / kMPanelRows) * kMPanelRows * k);
}

void pack_b(const uint16_t* B, uint16_t* B_reo, int K, int N) {
#if defined(FUSED_CPP_DEEPSEEK_V4_HAS_SVE_JIT_GEMM)
  ::fused_cpp::moe_sve::pack_b(B, B_reo, K, N);
#else
  (void)B;
  (void)B_reo;
  (void)K;
  (void)N;
  throw std::runtime_error("DeepSeek V4 SVE BF16 JIT pack-B is unavailable");
#endif
}

void pack_a_range(const uint16_t* A, uint16_t* packed, int M, int K, int panel_begin, int panel_end) {
  if (!available()) {
    throw std::runtime_error("DeepSeek V4 SVE BF16 JIT pack-A is unavailable");
  }
#if defined(FUSED_CPP_DEEPSEEK_V4_HAS_SVE_JIT_GEMM)
  const int panel_count = (M + kMPanelRows - 1) / kMPanelRows;
  if (panel_begin < 0 || panel_end < panel_begin || panel_end > panel_count) {
    throw std::runtime_error("invalid DeepSeek V4 SVE packed-A panel range");
  }
  for (int panel = panel_begin; panel < panel_end; ++panel) {
    const int row_start = panel * kMPanelRows;
    const int rows = std::min(kMPanelRows, M - row_start);
    const int physical_rows = rows <= 8 ? 8 : 12;
    pack_a_panel(A + static_cast<int64_t>(row_start) * K, packed + static_cast<int64_t>(panel) * kMPanelRows * K, rows,
                 physical_rows, K);
  }
#else
  (void)A;
  (void)packed;
  (void)M;
  (void)K;
  (void)panel_begin;
  (void)panel_end;
#endif
}

void dispatch_packed_f32(const uint16_t* packed_A, const uint16_t* B_reo, float* C, int M, int K, int N, int ldc) {
  if (!available()) {
    throw std::runtime_error("DeepSeek V4 SVE BF16 JIT GEMM is unavailable");
  }
#if defined(FUSED_CPP_DEEPSEEK_V4_HAS_SVE_JIT_GEMM)
  dispatch_packed_jit(packed_A, B_reo, C, M, K, N, ldc, false);
#else
  (void)packed_A;
  (void)B_reo;
  (void)C;
  (void)M;
  (void)K;
  (void)N;
  (void)ldc;
#endif
}

void dispatch_packed_bf16(const uint16_t* packed_A, const uint16_t* B_reo, uint16_t* C, int M, int K, int N, int ldc) {
  if (!available()) {
    throw std::runtime_error("DeepSeek V4 SVE BF16 JIT GEMM is unavailable");
  }
#if defined(FUSED_CPP_DEEPSEEK_V4_HAS_SVE_JIT_GEMM)
  dispatch_packed_jit(packed_A, B_reo, C, M, K, N, ldc, true);
#else
  (void)packed_A;
  (void)B_reo;
  (void)C;
  (void)M;
  (void)K;
  (void)N;
  (void)ldc;
#endif
}

void dispatch_f32(const uint16_t* A, const uint16_t* B_reo, float* C, uint16_t* A_reorder, int M, int K, int N,
                  int ldc) {
  if (!available()) {
    throw std::runtime_error("DeepSeek V4 SVE BF16 JIT GEMM is unavailable");
  }
#if defined(FUSED_CPP_DEEPSEEK_V4_HAS_SVE_JIT_GEMM)
  dispatch_jit(A, B_reo, C, A_reorder, M, K, N, ldc, false);
#else
  (void)A;
  (void)B_reo;
  (void)C;
  (void)A_reorder;
  (void)M;
  (void)K;
  (void)N;
  (void)ldc;
#endif
}

void dispatch_bf16(const uint16_t* A, const uint16_t* B_reo, uint16_t* C, uint16_t* A_reorder, int M, int K, int N,
                   int ldc) {
  if (!available()) {
    throw std::runtime_error("DeepSeek V4 SVE BF16 JIT GEMM is unavailable");
  }
#if defined(FUSED_CPP_DEEPSEEK_V4_HAS_SVE_JIT_GEMM)
  dispatch_jit(A, B_reo, C, A_reorder, M, K, N, ldc, true);
#else
  (void)A;
  (void)B_reo;
  (void)C;
  (void)A_reorder;
  (void)M;
  (void)K;
  (void)N;
  (void)ldc;
#endif
}

}  // namespace fused_cpp::deepseek_v4::attn_sve
