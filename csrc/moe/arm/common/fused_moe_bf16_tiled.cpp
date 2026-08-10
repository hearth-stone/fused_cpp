#include <ATen/MemoryOverlap.h>
#include <torch/extension.h>
#include <torch/csrc/autograd/variable.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif
#ifdef __aarch64__
#include <arm_neon.h>
#endif
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "../../common/backend.h"
#include "../../common/route_merge.h"

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

#ifdef __aarch64__
#include "gemm_params.h"
#include "../sve_bf16/jit_kernels.h"
#include "../sve_bf16/packing.h"
#include "../../../page_policy.h"
#include "../../../page_tensor.h"
#include "../../../profile_utils.h"

extern "C" {
void bf16gemm_k_ld(const uint16_t* A, const uint16_t* B_reo, float* C, uint16_t* A_reorder,
                   const gemm_params_t* params);
// Packed-read plain fp32 GEMM (fused_cpp-owned asm): A already in reorder-m8
// layout; no in-kernel repack. Used by the fused-packa w2 stage. m8 only.
void bf16gemm_k_ldp(const uint16_t* A, const uint16_t* B_reo, float* C, uint16_t* A_reorder,
                    const gemm_params_t* params);
void bf16gemm_k_ld1(const uint16_t* A, const uint16_t* B_reo, float* C, uint16_t* A_reorder,
                    const gemm_params_t* params);
void bf16gemm_k_ld2(const uint16_t* A, const uint16_t* B_reo, float* C, uint16_t* A_reorder,
                    const gemm_params_t* params);
void bf16gemm_k_ld4(const uint16_t* A, const uint16_t* B_reo, float* C, uint16_t* A_reorder,
                    const gemm_params_t* params);
// Bias-fused variants: identical to the plain kernels but add a per-column
// fp32 bias during the (zero-init) store stage. bias points at N fp32 values.
void bf16gemm_k_ld_bias_f(const uint16_t* A, const uint16_t* B_reo, float* C, uint16_t* A_reorder,
                          const gemm_params_t* params, const float* bias);
void bf16gemm_k_ld1_bias_f(const uint16_t* A, const uint16_t* B_reo, float* C, uint16_t* A_reorder,
                           const gemm_params_t* params, const float* bias);
void bf16gemm_k_ld2_bias_f(const uint16_t* A, const uint16_t* B_reo, float* C, uint16_t* A_reorder,
                           const gemm_params_t* params, const float* bias);
void bf16gemm_k_ld4_bias_f(const uint16_t* A, const uint16_t* B_reo, float* C, uint16_t* A_reorder,
                           const gemm_params_t* params, const float* bias);
// Fused w13 + gate*up (Task 2: linear, no silu). Interleaved-packed w13,
// bf16 output C[M, F_pad] (ldc = F_pad), no C load. N spans 2*F_pad packed
// columns; each 8-col tile yields 4 output features.
void bf16gemm_k_ld_silu_linear(const uint16_t* A, const uint16_t* B_reo, uint16_t* C, uint16_t* A_reorder,
                               const gemm_params_t* params);
// Fused w13 + SiLU-and-mul (poly4/5/6 exp), interleaved-packed w13, bf16 out.
void bf16gemm_k_ld_silu_poly4(const uint16_t* A, const uint16_t* B_reo, uint16_t* C, uint16_t* A_reorder,
                              const gemm_params_t* params);
void bf16gemm_k_ld_silu_poly5(const uint16_t* A, const uint16_t* B_reo, uint16_t* C, uint16_t* A_reorder,
                              const gemm_params_t* params);
void bf16gemm_k_ld_silu_poly6(const uint16_t* A, const uint16_t* B_reo, uint16_t* C, uint16_t* A_reorder,
                              const gemm_params_t* params);
// Packed-C fused kernels (fused_cpp-owned asm): pre-packed A read + reorder-m8
// packed-C store, so intermediate is written directly in w2 pre-packed layout.
void bf16gemm_k_ldp_silu_poly4_packc(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_ldp_silu_poly5_packc(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_ldp_silu_poly6_packc(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
// Packed-read reorder-m8 M-tail fused-silu kernels (packc store, no repack):
// consume gather's pre-packed A directly, 64B/8-row-block stride reading only
// the first mr rows. Used by the per-expert tail dispatch (tail=rows%8).
void bf16gemm_k_ldp_silu_poly4_packc_m4(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_ldp_silu_poly5_packc_m4(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_ldp_silu_poly6_packc_m4(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_ldp_silu_poly4_packc_m2(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_ldp_silu_poly5_packc_m2(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_ldp_silu_poly6_packc_m2(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_ldp_silu_poly4_packc_m1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_ldp_silu_poly5_packc_m1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_ldp_silu_poly6_packc_m1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
// Packed-read reorder-m8 plain fp32 M-tail kernels (rowmajor store) for w2.
void bf16gemm_k_ldp_m4(const uint16_t*, const uint16_t*, float*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_ldp_m2(const uint16_t*, const uint16_t*, float*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_ldp_m1(const uint16_t*, const uint16_t*, float*, uint16_t*, const gemm_params_t*);
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
void moe_sve_w13_silu_poly4_packc_m12(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_silu_poly5_packc_m12(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_silu_poly6_packc_m12(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_silu_poly4_packc_m12_rows(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                           const gemm_params_t*);
void moe_sve_w13_silu_poly5_packc_m12_rows(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                           const gemm_params_t*);
void moe_sve_w13_silu_poly6_packc_m12_rows(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                           const gemm_params_t*);
void moe_sve_w13_silu_poly4_packc_m12_opt(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_silu_poly5_packc_m12_opt(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_silu_poly6_packc_m12_opt(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_silu_poly4_packc_m12_rows_opt(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                               const gemm_params_t*);
void moe_sve_w13_silu_poly5_packc_m12_rows_opt(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                               const gemm_params_t*);
void moe_sve_w13_silu_poly6_packc_m12_rows_opt(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                               const gemm_params_t*);
void moe_sve_w13_silu_poly4_packc_m12_recip1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                             const gemm_params_t*);
void moe_sve_w13_silu_poly5_packc_m12_recip1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                             const gemm_params_t*);
void moe_sve_w13_silu_poly6_packc_m12_recip1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                             const gemm_params_t*);
void moe_sve_w13_silu_poly4_packc_m12_rows_recip1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                                  const gemm_params_t*);
void moe_sve_w13_silu_poly5_packc_m12_rows_recip1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                                  const gemm_params_t*);
void moe_sve_w13_silu_poly6_packc_m12_rows_recip1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                                  const gemm_params_t*);
void moe_sve_w13_silu_poly4_packc_m12_recip2(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                             const gemm_params_t*);
void moe_sve_w13_silu_poly5_packc_m12_recip2(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                             const gemm_params_t*);
void moe_sve_w13_silu_poly6_packc_m12_recip2(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                             const gemm_params_t*);
void moe_sve_w13_silu_poly4_packc_m12_rows_recip2(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                                  const gemm_params_t*);
void moe_sve_w13_silu_poly5_packc_m12_rows_recip2(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                                  const gemm_params_t*);
void moe_sve_w13_silu_poly6_packc_m12_rows_recip2(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                                  const gemm_params_t*);
void moe_sve_w13_silu_minimax3_packc_m12(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_silu_minimax3_packc_m12_rows(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                              const gemm_params_t*);
void moe_sve_w13_silu_poly4_packc(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_silu_poly5_packc(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_silu_poly6_packc(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_silu_poly4_packc_m4(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_silu_poly5_packc_m4(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_silu_poly6_packc_m4(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_silu_poly4_packc_m2(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_silu_poly5_packc_m2(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_silu_poly6_packc_m2(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_silu_poly4_packc_m1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_silu_poly5_packc_m1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_silu_poly6_packc_m1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_identity_packc_m12(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_identity_packc_m12_rows(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_identity_packc(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_identity_packc_m4(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_identity_packc_m2(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w13_identity_packc_m1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w2_packed(const uint16_t*, const uint16_t*, float*, uint16_t*, const gemm_params_t*);
void moe_sve_w2_packed_m12(const uint16_t*, const uint16_t*, float*, uint16_t*, const gemm_params_t*);
void moe_sve_w2_packed_m4(const uint16_t*, const uint16_t*, float*, uint16_t*, const gemm_params_t*);
void moe_sve_w2_packed_m2(const uint16_t*, const uint16_t*, float*, uint16_t*, const gemm_params_t*);
void moe_sve_w2_packed_m1(const uint16_t*, const uint16_t*, float*, uint16_t*, const gemm_params_t*);
void moe_sve_w2_packed_direct(const uint16_t*, const uint16_t*, float*, const int64_t*, const gemm_params_t*);
void moe_sve_w2_packed_direct_m12(const uint16_t*, const uint16_t*, float*, const int64_t*, const gemm_params_t*);
void moe_sve_w2_packed_direct_m4(const uint16_t*, const uint16_t*, float*, const int64_t*, const gemm_params_t*);
void moe_sve_w2_packed_direct_m2(const uint16_t*, const uint16_t*, float*, const int64_t*, const gemm_params_t*);
void moe_sve_w2_packed_direct_m1(const uint16_t*, const uint16_t*, float*, const int64_t*, const gemm_params_t*);
void moe_sve_w2_packed_direct_bf16(const uint16_t*, const uint16_t*, uint16_t*, const int64_t*,
                                   const gemm_params_t*);
void moe_sve_w2_packed_direct_bf16_m12(const uint16_t*, const uint16_t*, uint16_t*, const int64_t*,
                                       const gemm_params_t*);
void moe_sve_w2_packed_direct_bf16_m4(const uint16_t*, const uint16_t*, uint16_t*, const int64_t*,
                                      const gemm_params_t*);
void moe_sve_w2_packed_direct_bf16_m2(const uint16_t*, const uint16_t*, uint16_t*, const int64_t*,
                                      const gemm_params_t*);
void moe_sve_w2_packed_direct_bf16_m1(const uint16_t*, const uint16_t*, uint16_t*, const int64_t*,
                                      const gemm_params_t*);
void moe_sve_w2_packed_bf16(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w2_packed_bf16_m12(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w2_packed_bf16_m4(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w2_packed_bf16_m2(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void moe_sve_w2_packed_bf16_m1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
// Production Kc kernels use an extended gemm_params_t payload and one
// canonical pointer ABI for every final epilogue.
void moe_sve_kc_kernel_m12(const uint16_t*, const uint16_t*, void*, const void*, const gemm_params_t*);
void moe_sve_kc_kernel_m8(const uint16_t*, const uint16_t*, void*, const void*, const gemm_params_t*);
void moe_sve_kc_kernel_m4(const uint16_t*, const uint16_t*, void*, const void*, const gemm_params_t*);
void moe_sve_kc_kernel_m2(const uint16_t*, const uint16_t*, void*, const void*, const gemm_params_t*);
#endif
// M-tail (m=1/2/4) fused silu kernels.
void bf16gemm_k_ld_silu_poly4_m1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_ld_silu_poly4_m2(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_ld_silu_poly4_m4(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_ld_silu_poly5_m1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_ld_silu_poly5_m2(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_ld_silu_poly5_m4(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_ld_silu_poly6_m1(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_ld_silu_poly6_m2(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
void bf16gemm_k_ld_silu_poly6_m4(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
}
#endif

namespace {

constexpr int64_t kKernelTile = 8;

int64_t ceil_div_int64(int64_t x, int64_t y) { return (x + y - 1) / y; }

int64_t ceil_to_multiple(int64_t x, int64_t multiple) { return ceil_div_int64(x, multiple) * multiple; }

int64_t sve_m12_main_rows(int64_t rows) {
  const int64_t full_rows = rows / 12 * 12;
  const int64_t tail_rows = rows - full_rows;
  // M8 plus M1/M2/M4 would stream B twice for a 9-11 row tail.
  return tail_rows >= 9 ? full_rows + 12 : full_rows;
}

int64_t sve_hybrid_packed_rows(int64_t rows) {
  const int64_t main = sve_m12_main_rows(rows);
  return main >= rows ? main : main + ceil_to_multiple(rows - main, int64_t{8});
}

struct SplitRange {
  int64_t begin = 0;
  int64_t size = 0;
};

SplitRange split_evenly(int64_t units, int64_t group_size, int64_t local_tid) {
  if (units <= 0 || group_size <= 0 || local_tid < 0 || local_tid >= group_size) {
    return SplitRange{};
  }
  const int64_t units_per_thread = units / group_size;
  const int64_t extra_units = units % group_size;
  if (local_tid < extra_units) {
    return SplitRange{local_tid * (units_per_thread + 1), units_per_thread + 1};
  }
  return SplitRange{extra_units * (units_per_thread + 1) + (local_tid - extra_units) * units_per_thread,
                    units_per_thread};
}

SplitRange n_split_range(int N, int64_t group_size, int64_t local_tid) {
  const SplitRange block_range = split_evenly(static_cast<int64_t>(N) / kKernelTile, group_size, local_tid);
  return SplitRange{block_range.begin * kKernelTile, block_range.size * kKernelTile};
}

SplitRange n_split_range_tile(int N, int64_t group_size, int64_t local_tid, int64_t tile) {
  const int64_t t = std::max<int64_t>(tile, kKernelTile);
  const SplitRange block_range = split_evenly(static_cast<int64_t>(N) / t, group_size, local_tid);
  return SplitRange{block_range.begin * t, block_range.size * t};
}

template <typename Fn>
void with_w2_scatter_owner(int N, int64_t group_size, int64_t local_tid, int64_t n_tile,
                           bool use_w2_n_owner, Fn&& fn) {
  fn(use_w2_n_owner ? n_split_range_tile(N, group_size, local_tid, n_tile)
                    : n_split_range(N, group_size, local_tid));
}

void check_positive_int(int64_t value, const char* name) {
  TORCH_CHECK(value > 0, name, " must be positive, got ", value);
  TORCH_CHECK(value <= std::numeric_limits<int>::max(), name, " exceeds int32 kernel limit: ", value);
}

void check_bf16_cpu(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must have dtype torch.bfloat16");
}

bool is_integer_dtype(at::ScalarType dtype) {
  return dtype == at::kByte || dtype == at::kChar || dtype == at::kShort || dtype == at::kInt || dtype == at::kLong;
}

bool is_floating_dtype(at::ScalarType dtype) {
  return dtype == at::kFloat || dtype == at::kDouble || dtype == at::kHalf || dtype == at::kBFloat16;
}

uint16_t* bf16_data(at::Tensor& tensor) { return reinterpret_cast<uint16_t*>(tensor.data_ptr<at::BFloat16>()); }

const uint16_t* bf16_data_const(const at::Tensor& tensor) {
  return reinterpret_cast<const uint16_t*>(tensor.data_ptr<at::BFloat16>());
}

inline uint16_t bf16_bits_from_float(float value) {
  c10::BFloat16 bf(value);
  uint16_t bits = 0;
  static_assert(sizeof(bits) == sizeof(bf));
  std::memcpy(&bits, &bf, sizeof(bits));
  return bits;
}

// Vectorized fp32 -> bf16 (round-to-nearest-even, matching c10::BFloat16) for
// the scatter stage. Uses ARM bfcvtn (8/iter) when BF16 intrinsics are
// available; scalar tail / fallback otherwise. dst is a uint16_t bf16 buffer.
inline void convert_f32_to_bf16(const float* src, uint16_t* dst, int64_t n) {
  int64_t i = 0;
#if defined(__aarch64__) && defined(__ARM_FEATURE_BF16)
  for (; i + 8 <= n; i += 8) {
    const bfloat16x8_t b = vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(vld1q_f32(src + i)), vld1q_f32(src + i + 4));
    vst1q_u16(dst + i, vreinterpretq_u16_bf16(b));
  }
#endif
  for (; i < n; ++i) dst[i] = bf16_bits_from_float(src[i]);
}

inline float bf16_bits_to_float(uint16_t bits) {
  uint32_t widened = static_cast<uint32_t>(bits) << 16;
  float value = 0.0f;
  std::memcpy(&value, &widened, sizeof(value));
  return value;
}

inline void accumulate_weighted_bf16(float* acc, const uint16_t* src, float weight, int64_t n) {
  int64_t i = 0;
#if defined(__aarch64__)
  for (; i + 8 <= n; i += 8) {
    const uint16x8_t packed = vld1q_u16(src + i);
    const uint32x4_t lo_bits = vshlq_n_u32(vmovl_u16(vget_low_u16(packed)), 16);
    const uint32x4_t hi_bits = vshlq_n_u32(vmovl_u16(vget_high_u16(packed)), 16);
    const float32x4_t lo = vreinterpretq_f32_u32(lo_bits);
    const float32x4_t hi = vreinterpretq_f32_u32(hi_bits);
    vst1q_f32(acc + i, vfmaq_n_f32(vld1q_f32(acc + i), lo, weight));
    vst1q_f32(acc + i + 4, vfmaq_n_f32(vld1q_f32(acc + i + 4), hi, weight));
  }
#endif
  for (; i < n; ++i) {
    acc[i] += bf16_bits_to_float(src[i]) * weight;
  }
}

inline void accumulate_weighted_f32(float* acc, const float* src, float weight, int64_t n) {
  int64_t i = 0;
#if defined(__aarch64__)
  for (; i + 8 <= n; i += 8) {
    vst1q_f32(acc + i, vfmaq_n_f32(vld1q_f32(acc + i), vld1q_f32(src + i), weight));
    vst1q_f32(acc + i + 4, vfmaq_n_f32(vld1q_f32(acc + i + 4), vld1q_f32(src + i + 4), weight));
  }
#endif
  for (; i < n; ++i) {
    acc[i] += src[i] * weight;
  }
}

int resolve_route_merge_unroll(bool use_sve_backend) {
  const char* variable = "FUSED_CPP_MOE_SVE_ROUTE_MERGE_UNROLL";
  const char* value = std::getenv(variable);
  if (value == nullptr || value[0] == '\0') {
    variable = "FUSED_CPP_MOE_SVE_ROUTE_MERGE_TREE_UNROLL";
    value = std::getenv(variable);
  }
  if (value == nullptr || value[0] == '\0') {
    return use_sve_backend && ::fused_cpp::moe_route_merge::sve_available() ? 1 : 0;
  }
  if (value[0] == '0' && value[1] == '\0') {
    return 0;
  }
  errno = 0;
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  TORCH_CHECK(errno == 0 && end != value && *end == '\0' && (parsed == 1 || parsed == 2 || parsed == 4), variable,
              " must be 0, 1, 2, or 4, got ", value);
  TORCH_CHECK(use_sve_backend, variable, " requires the SVE MoE backend");
  TORCH_CHECK(::fused_cpp::moe_route_merge::sve_available(), variable, " requires an SVE build/runtime");
  return static_cast<int>(parsed);
}

void merge_route_range(const float* route_out, const uint16_t* route_out_bf16, const float* weights, uint16_t* output,
                       int64_t token_begin, int64_t token_end, int64_t top_k, int64_t hidden_size, bool use_bf16_route,
                       int sve_unroll) {
  if (sve_unroll != 0) {
    if (use_bf16_route) {
      ::fused_cpp::moe_route_merge::merge_bf16_sve(route_out_bf16, weights, output, token_begin, token_end, top_k,
                                                   hidden_size, sve_unroll);
    } else {
      ::fused_cpp::moe_route_merge::merge_f32_sve(route_out, weights, output, token_begin, token_end, top_k,
                                                  hidden_size, sve_unroll);
    }
    return;
  }

  std::vector<float> acc(static_cast<size_t>(hidden_size));
  for (int64_t token = token_begin; token < token_end; ++token) {
    uint16_t* dst = output + token * hidden_size;
    std::fill(acc.begin(), acc.end(), 0.0f);
    for (int64_t slot = 0; slot < top_k; ++slot) {
      const int64_t flat = token * top_k + slot;
      const float weight = weights[flat];
      if (use_bf16_route) {
        accumulate_weighted_bf16(acc.data(), route_out_bf16 + flat * hidden_size, weight, hidden_size);
      } else {
        accumulate_weighted_f32(acc.data(), route_out + flat * hidden_size, weight, hidden_size);
      }
    }
    convert_f32_to_bf16(acc.data(), dst, hidden_size);
  }
}

enum class MoeGemmStage {
  kW13,
  kW2,
};

enum class MoeGemmSplit {
  kM,
  kN,
};

MoeGemmSplit choose_moe_gemm_split(MoeGemmStage stage, int M, int N, int64_t group_size) {
  if (group_size <= 1) {
    return MoeGemmSplit::kN;
  }

  if (stage == MoeGemmStage::kW13) {
    if (group_size == 8 && M >= 512) {
      return MoeGemmSplit::kM;
    }
    if (group_size == 4 && M >= 1024) {
      return MoeGemmSplit::kM;
    }
    if (group_size == 2 && M <= 32) {
      return MoeGemmSplit::kM;
    }
    return group_size > 8 && M > N ? MoeGemmSplit::kM : MoeGemmSplit::kN;
  }

  if (group_size == 8 && M >= 4096) {
    return MoeGemmSplit::kM;
  }
  if (group_size == 4 && M >= 8192) {
    return MoeGemmSplit::kM;
  }
  return group_size > 8 && M > N ? MoeGemmSplit::kM : MoeGemmSplit::kN;
}

// ── Middle-layer split planning (arch-independent, pure) ─────────────────
// M-split with 8-row alignment: distribute floor(M/kKernelTile) 8-row blocks
// evenly (each such thread runs the fast 8-row `bf16gemm_k_ld` kernel), and
// place the <kKernelTile remainder rows on a single thread (the owner of the
// last block, or thread 0 when there are no full blocks).
SplitRange m_split_range_aligned(int64_t M, int64_t group_size, int64_t local_tid) {
  if (M <= 0 || group_size <= 0 || local_tid < 0 || local_tid >= group_size) {
    return SplitRange{};
  }
  const int64_t blocks = M / kKernelTile;
  const int64_t rem = M % kKernelTile;
  if (blocks == 0) {
    return local_tid == 0 ? SplitRange{0, M} : SplitRange{};
  }
  const SplitRange block_range = split_evenly(blocks, group_size, local_tid);
  SplitRange range{block_range.begin * kKernelTile, block_range.size * kKernelTile};
  const int64_t last_block_owner = std::min(group_size, blocks) - 1;
  if (rem > 0 && local_tid == last_block_owner) {
    range.size += rem;
  }
  return range;
}

struct GemmSplitPlan {
  MoeGemmSplit split = MoeGemmSplit::kN;
};

struct Gemm2DSplitPlan {
  int64_t tm = 1;
  int64_t tn = 1;
  int64_t n_tile = kKernelTile;
};

struct Gemm2DThreadRange {
  int64_t m_block_begin = 0;
  int64_t m_blocks = 0;
  int64_t row_begin = 0;
  int64_t rows = 0;
  int64_t n_begin = 0;
  int64_t n_cols = 0;
};

// Per-thread work range for a chosen plan. M-split -> row range (8-row aligned
// save the single remainder thread); N-split -> column range (kKernelTile-block
// aligned). N is expected already padded to a multiple of kKernelTile.
SplitRange team_gemm_split_range(const GemmSplitPlan& plan, int64_t M, int64_t N, int64_t group_size,
                                 int64_t local_tid) {
  if (plan.split == MoeGemmSplit::kM) {
    return m_split_range_aligned(M, group_size, local_tid);
  }
  const SplitRange block_range = split_evenly(N / kKernelTile, group_size, local_tid);
  return SplitRange{block_range.begin * kKernelTile, block_range.size * kKernelTile};
}

// Threads that receive nonzero work under each candidate split.
int64_t m_split_active_threads(int64_t M, int64_t group_size) {
  if (M <= 0 || group_size <= 0) {
    return 0;
  }
  const int64_t blocks = M / kKernelTile;
  if (blocks == 0) {
    return 1;
  }
  return std::min(group_size, blocks);
}

int64_t n_split_active_threads(int64_t N, int64_t group_size) {
  if (N <= 0 || group_size <= 0) {
    return 0;
  }
  const int64_t blocks = N / kKernelTile;
  if (blocks == 0) {
    return 1;
  }
  return std::min(group_size, blocks);
}

Gemm2DSplitPlan plan_2d_gemm_split(int64_t M, int64_t K, int64_t N, int64_t group_size, int64_t n_tile) {
  (void)M;
  (void)K;
  (void)N;
  const int64_t threads = std::max<int64_t>(group_size, 1);
  const int64_t tile = std::max<int64_t>(n_tile, kKernelTile);
  return Gemm2DSplitPlan{1, threads, tile};
}

Gemm2DThreadRange gemm_2d_thread_range(const Gemm2DSplitPlan& plan, int64_t M, int64_t N, int64_t local_tid) {
  const int64_t team_threads = plan.tm * plan.tn;
  if (M <= 0 || N <= 0 || local_tid < 0 || local_tid >= team_threads) {
    return Gemm2DThreadRange{};
  }
  const int64_t m_id = local_tid / plan.tn;
  const int64_t n_id = local_tid % plan.tn;
  const int64_t m_blocks_total = ceil_div_int64(M, kKernelTile);
  const int64_t n_tiles_total = N / plan.n_tile;
  const SplitRange m_block_range = split_evenly(m_blocks_total, plan.tm, m_id);
  const SplitRange n_tile_range = split_evenly(n_tiles_total, plan.tn, n_id);
  Gemm2DThreadRange range;
  range.m_block_begin = m_block_range.begin;
  range.m_blocks = m_block_range.size;
  range.row_begin = m_block_range.begin * kKernelTile;
  if (range.row_begin < M) {
    range.rows = std::min<int64_t>(m_block_range.size * kKernelTile, M - range.row_begin);
  }
  range.n_begin = n_tile_range.begin * plan.n_tile;
  range.n_cols = n_tile_range.size * plan.n_tile;
  return range;
}

struct GemmSplitContext {
  MoeGemmStage stage = MoeGemmStage::kW13;
  int64_t M = 0;
  int64_t K = 0;
  int64_t N = 0;
  int64_t group_size = 1;
  int64_t m_active_threads = 1;
  int64_t n_active_threads = 1;
};

using SplitSelectorFn = MoeGemmSplit (*)(const GemmSplitContext&);

// Split selector — the single seam where the cooperative-team M-vs-N GEMM split
// is decided. Default: N-split.
//
// Why N-split is the default (empirical + structural):
//  * Measured on AWS Graviton (AmazonECS8Cores, 8-core) for the two MoE GEMM
//    shapes w13 (K=H=4096, N=2F=1024) and w2 (K=F=512, N=H=4096). Data in
//    cpu_moe_schedule_optimization/cost_model/profiles/split_mn_aws*.{csv,json}
//    and cost_model/SPLIT_MN_FINDINGS.md.
//  * N-split won at EVERY tested (M, group_size); M-split never won. There is NO
//    crossover even at M/N = 64 (M swept up to 65536):
//      - small M: N-split keeps the whole team busy, whereas an M-split only has
//        M/kKernelTile row-blocks and starves threads (N up to ~5x faster);
//      - large M: the GEMM is compute-bound and N-split's kernel is a stable
//        ~5-10% more efficient — it works on contiguous packed-B column slices,
//        while an M-split makes every thread traverse the FULL packed weight
//        matrix.
//  * So the decision is not a small-vs-large-M tradeoff; both regimes favor N.
//
// The ONLY case that prefers M-split is a shape too narrow in N to fill the team
// (N/kKernelTile < group_size) while M can — this does not occur for MoE
// w13/w2. TODO(cost-model): a measured team-GEMM cost table can refine this if
// future shapes/hardware differ; this selector is the one place to change.
MoeGemmSplit default_split_selector(const GemmSplitContext& ctx) {
  (void)ctx;
  return MoeGemmSplit::kN;  // default
}

GemmSplitPlan plan_team_gemm_split(MoeGemmStage stage, int64_t M, int64_t K, int64_t N, int64_t group_size,
                                   SplitSelectorFn selector = default_split_selector) {
  GemmSplitContext ctx;
  ctx.stage = stage;
  ctx.M = M;
  ctx.K = K;
  ctx.N = N;
  ctx.group_size = std::max<int64_t>(group_size, 1);
  ctx.m_active_threads = m_split_active_threads(M, ctx.group_size);
  ctx.n_active_threads = n_split_active_threads(N, ctx.group_size);
  GemmSplitPlan plan;
  plan.split = (selector != nullptr ? selector : default_split_selector)(ctx);
  return plan;
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

#ifdef __aarch64__

// Bottom layer: single-thread bf16 GEMM over the whole [M, N] slice.
// Dispatches the i8mm microkernels by M tile (8/4/2/1 rows), optional fused
// bias. B_reo is the pre-packed weight; A_reorder is per-call scratch. This is
// the sole compute primitive the middle-layer team_gemm builds on.
void single_thread_gemm(const uint16_t* A, const uint16_t* B_reo, float* C, uint16_t* A_reorder, int M, int K, int N,
                        int ldc, const float* bias = nullptr) {
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
    if (bias != nullptr) {
      bf16gemm_k_ld_bias_f(A, B_reo, C, A_reorder, &p, bias);
    } else {
      bf16gemm_k_ld(A, B_reo, C, A_reorder, &p);
    }
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
    if (bias != nullptr) {
      bf16gemm_k_ld4_bias_f(At, B_reo, Ct, A_reo_t, &p, bias);
    } else {
      bf16gemm_k_ld4(At, B_reo, Ct, A_reo_t, &p);
    }
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
    if (bias != nullptr) {
      bf16gemm_k_ld2_bias_f(At, B_reo, Ct, A_reo_t, &p, bias);
    } else {
      bf16gemm_k_ld2(At, B_reo, Ct, A_reo_t, &p);
    }
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
    if (bias != nullptr) {
      bf16gemm_k_ld1_bias_f(At, B_reo, Ct, A_reo_t, &p, bias);
    } else {
      bf16gemm_k_ld1(At, B_reo, Ct, A_reo_t, &p);
    }
  }
}

// Fused w13 + SiLU-and-mul kernel pointers for a given exp polynomial degree
// (4/5/6). Signature: (A, B_reo, C_bf16, A_reorder, params). Shared by the
// test binding and the MoE wiring.
using FusedSiluKernelFn = void (*)(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);

struct FusedSiluKernelSet {
  FusedSiluKernelFn m8 = nullptr;
  FusedSiluKernelFn m4 = nullptr;
  FusedSiluKernelFn m2 = nullptr;
  FusedSiluKernelFn m1 = nullptr;
  FusedSiluKernelFn m12 = nullptr;
  FusedSiluKernelFn m12_rows = nullptr;
};

FusedSiluKernelSet fused_silu_kernels_for_degree(int64_t degree) {
  switch (degree) {
    case 4:
      return {bf16gemm_k_ld_silu_poly4, bf16gemm_k_ld_silu_poly4_m4, bf16gemm_k_ld_silu_poly4_m2,
              bf16gemm_k_ld_silu_poly4_m1};
    case 5:
      return {bf16gemm_k_ld_silu_poly5, bf16gemm_k_ld_silu_poly5_m4, bf16gemm_k_ld_silu_poly5_m2,
              bf16gemm_k_ld_silu_poly5_m1};
    case 6:
      return {bf16gemm_k_ld_silu_poly6, bf16gemm_k_ld_silu_poly6_m4, bf16gemm_k_ld_silu_poly6_m2,
              bf16gemm_k_ld_silu_poly6_m1};
    default:
      return {};
  }
}

// Pre-packed-A m8 fused kernel for a given exp degree (4/5/6). Used by the
// N-split shared-A-pack path where rows are padded to a multiple of 8.
FusedSiluKernelFn fused_silu_packed_m8_for_degree(int64_t degree) {
  (void)degree;
  return nullptr;
}

// Packed-C m8 fused kernel: pre-packed A + reorder-m8 packed-C store. Writes
// intermediate directly in the layout w2 reads as pre-packed A (Part 2). M
// padded to a multiple of 8.
FusedSiluKernelFn fused_silu_packc_m8_for_degree(int64_t degree) {
  switch (degree) {
    case 4:
      return bf16gemm_k_ldp_silu_poly4_packc;
    case 5:
      return bf16gemm_k_ldp_silu_poly5_packc;
    case 6:
      return bf16gemm_k_ldp_silu_poly6_packc;
    default:
      return nullptr;
  }
}

// Packed-read reorder-m8 fused-silu kernel set {m8, m4, m2, m1} (packc store).
// m8 is the full-block kernel; m4/m2/m1 are the packed-read tail kernels.
FusedSiluKernelSet fused_silu_packc_set_for_degree(int64_t degree) {
  switch (degree) {
    case 4:
      return {bf16gemm_k_ldp_silu_poly4_packc, bf16gemm_k_ldp_silu_poly4_packc_m4, bf16gemm_k_ldp_silu_poly4_packc_m2,
              bf16gemm_k_ldp_silu_poly4_packc_m1};
    case 5:
      return {bf16gemm_k_ldp_silu_poly5_packc, bf16gemm_k_ldp_silu_poly5_packc_m4, bf16gemm_k_ldp_silu_poly5_packc_m2,
              bf16gemm_k_ldp_silu_poly5_packc_m1};
    case 6:
      return {bf16gemm_k_ldp_silu_poly6_packc, bf16gemm_k_ldp_silu_poly6_packc_m4, bf16gemm_k_ldp_silu_poly6_packc_m2,
              bf16gemm_k_ldp_silu_poly6_packc_m1};
    default:
      return {};
  }
}

#ifdef __aarch64__
// w13 fused-silu packc dispatch over `rows` (NOT padded to 8): m8 full blocks +
// per-tail (rows%8) strategy reading gather's pre-packed reorder-m8 A. Pointers
// packed_A / w13_packed / C are already sliced to this thread's N range.
//   tail 1->m1, 2->m2, 3->m4(pad4), 4->m4, 5/6/7->m8(pad8). (5,6 pad per
//   microbench: split tail re-reads B twice, slower than pad-8 in packed path.)
// reorder-m8: row r within a K-block is at uint16 offset r*4; block b starts at
// b*8*K (A) and b*8*ldc (packc C).
void packc_w13_tail_dispatch(const uint16_t* packed_A, const uint16_t* w13_packed, uint16_t* C, int rows, int K, int N,
                             int ldc, const FusedSiluKernelSet& ks) {
  gemm_params_t p;
  p.k = K;
  p.n = N;
  p.lda = K;
  p.ldb = K;
  p.ldc = ldc;
  const int nb_full = rows / 8;
  const int m_full = nb_full * 8;
  if (m_full > 0) {
    p.m = m_full;
    ks.m8(packed_A, w13_packed, C, nullptr, &p);
  }
  const int tail = rows - m_full;
  if (tail == 0) return;
  const uint16_t* At = packed_A + static_cast<int64_t>(nb_full) * 8 * K;
  uint16_t* Ct = C + static_cast<int64_t>(nb_full) * 8 * ldc;
  auto run = [&](FusedSiluKernelFn fn, int mr, int r0) {
    p.m = mr;
    fn(At + static_cast<int64_t>(r0) * 4, w13_packed, Ct + static_cast<int64_t>(r0) * 4, nullptr, &p);
  };
  switch (tail) {
    case 1:
      run(ks.m1, 1, 0);
      break;
    case 2:
      run(ks.m2, 2, 0);
      break;
    case 3:
      run(ks.m4, 4, 0);
      break;  // pad to 4 (row3 = gather-zero)
    case 4:
      run(ks.m4, 4, 0);
      break;
    // tail 5,6: microbench shows m4+m1/m4+m2 re-reads all of B twice
    // (508/510us) > pad-to-8 m8 (414us) in the packed world, so pad to 8.
    case 5:
      run(ks.m8, 8, 0);
      break;
    case 6:
      run(ks.m8, 8, 0);
      break;
    case 7:
      run(ks.m8, 8, 0);
      break;  // pad to 8 (rows>=tail = gather-zero)
  }
}

// w2 plain fp32 packed-read reorder-m8 tail dispatch (same tail table, rowmajor
// fp32 store). down / w2_packed already sliced to this thread's N range.
using PlainPackedKernelFn = void (*)(const uint16_t*, const uint16_t*, float*, uint16_t*, const gemm_params_t*);
void packed_w2_tail_dispatch(const uint16_t* packed_A, const uint16_t* w2_packed, float* down, int rows, int K, int N,
                             int ldc) {
  gemm_params_t p;
  p.k = K;
  p.n = N;
  p.lda = K;
  p.ldb = K;
  p.ldc = ldc;
  const int nb_full = rows / 8;
  const int m_full = nb_full * 8;
  if (m_full > 0) {
    p.m = m_full;
    bf16gemm_k_ldp(packed_A, w2_packed, down, nullptr, &p);
  }
  const int tail = rows - m_full;
  if (tail == 0) return;
  const uint16_t* At = packed_A + static_cast<int64_t>(nb_full) * 8 * K;
  float* Dt = down + static_cast<int64_t>(nb_full) * 8 * ldc;
  auto run = [&](PlainPackedKernelFn fn, int mr, int r0) {
    p.m = mr;
    fn(At + static_cast<int64_t>(r0) * 4, w2_packed, Dt + static_cast<int64_t>(r0) * ldc, nullptr, &p);
  };
  switch (tail) {
    case 1:
      run(bf16gemm_k_ldp_m1, 1, 0);
      break;
    case 2:
      run(bf16gemm_k_ldp_m2, 2, 0);
      break;
    case 3:
      run(bf16gemm_k_ldp_m4, 4, 0);
      break;
    case 4:
      run(bf16gemm_k_ldp_m4, 4, 0);
      break;
    // tail 5,6,7: pad to 8 (see packc_w13_tail_dispatch note); same strategy
    // as w13 so intermediate write/read row counts stay consistent.
    case 5:
    case 6:
    case 7:
      bf16gemm_k_ldp(At, w2_packed, Dt, nullptr, (p.m = 8, &p));
      break;
  }
}

bool sve_w13_skip_silu_enabled() {
  const char* value = std::getenv("FUSED_CPP_MOE_W13_SKIP_SILU");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool sve_w2_bf16_route_enabled() {
  const char* value = std::getenv("FUSED_CPP_MOE_W2_BF16_ROUTE");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool sve_w2_direct_route_enabled() {
  const char* value = std::getenv("FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE");
  if (value == nullptr || value[0] == '\0') {
    return true;
  }
  return value[0] != '0';
}

bool sve_w2_direct_route_offsets_fit(int64_t num_routes, int64_t route_stride, int64_t n_tile,
                                     int64_t element_bytes) {
  if (route_stride <= 0 || n_tile <= 0 || element_bytes <= 0 ||
      route_stride > std::numeric_limits<int32_t>::max() / element_bytes ||
      n_tile - 1 > std::numeric_limits<int32_t>::max() / element_bytes) {
    return false;
  }
  if (num_routes <= 1) {
    return true;
  }
  const int64_t row_bytes = route_stride * element_bytes;
  const int64_t max_tile_offset = (n_tile - 1) * element_bytes;
  const int64_t max_route_delta = std::numeric_limits<int32_t>::max() - max_tile_offset;
  return num_routes - 1 <= max_route_delta / row_bytes;
}

bool sve_w13_m12_epilogue_opt_enabled() {
  const char* value = std::getenv("FUSED_CPP_MOE_SILU_M12_OPT");
  if (value == nullptr || value[0] == '\0') {
    return true;
  }
  return value[0] != '0';
}

int sve_w13_silu_recip_nr_steps() {
  // Experimental M12 epilogue mode. M8/M4/M2/M1 tails retain exact FDIV.
  const char* value = std::getenv("FUSED_CPP_MOE_SILU_RECIP_NR");
  if (value == nullptr || value[0] == '\0') {
    return 0;
  }
  if (value[0] == '1' && value[1] == '\0') {
    return 1;
  }
  if (value[0] == '2' && value[1] == '\0') {
    return 2;
  }
  return 0;
}

bool sve_w13_silu_minimax3_enabled() {
  const char* value = std::getenv("FUSED_CPP_MOE_SILU_MINIMAX3");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
using SveKcKernelFn = void (*)(const uint16_t*, const uint16_t*, void*, const void*, const gemm_params_t*);

struct SveKcFusedSiluKernelSet {
  SveKcKernelFn m8 = nullptr;
  SveKcKernelFn m4 = nullptr;
  SveKcKernelFn m2 = nullptr;
  SveKcKernelFn m1 = nullptr;
  SveKcKernelFn m12 = nullptr;
  int mode = 0;
  int m12_mode = 0;
};

struct alignas(8) SveKBlockParams {
  gemm_params_t gemm;
  int32_t kc = 0;
  int32_t packed_n = 0;
  int32_t n_begin = 0;
  int32_t mode = 0;
  float* partial_c = nullptr;
};

static_assert(sizeof(gemm_params_t) == 24, "unexpected gemm_params_t ABI");
static_assert(offsetof(SveKBlockParams, kc) == 24, "unexpected SVE Kc params ABI");
static_assert(offsetof(SveKBlockParams, packed_n) == 28, "unexpected SVE packed-N params ABI");
static_assert(offsetof(SveKBlockParams, n_begin) == 32, "unexpected SVE N-begin params ABI");
static_assert(offsetof(SveKBlockParams, mode) == 36, "unexpected SVE mode params ABI");
static_assert(offsetof(SveKBlockParams, partial_c) == 40, "unexpected SVE partial-C params ABI");

thread_local std::vector<float> sve_kblock_partial_c;

SveKBlockParams make_sve_kblock_params(int m, int K, int N, int ldc, int packed_n, int n_begin, int mode) {
  TORCH_CHECK(m > 0 && K > 0 && N > 0 && ldc > 0, "invalid SVE Kc GEMM dimensions");
  TORCH_CHECK(packed_n >= N && n_begin >= 0 && n_begin + N <= packed_n,
              "invalid SVE Kc packed-N slice: packed_n=", packed_n, " begin=", n_begin, " size=", N);
  TORCH_CHECK(n_begin % ::fused_cpp::moe_sve::n_tile() == 0,
              "SVE Kc N slice must start on an N tile: begin=", n_begin);
  TORCH_CHECK(static_cast<uint64_t>(N) <= std::numeric_limits<size_t>::max() / 12,
              "SVE Kc partial-C size overflow");
  const size_t partial_elements = static_cast<size_t>(12) * static_cast<size_t>(N);
  if (sve_kblock_partial_c.size() < partial_elements) {
    sve_kblock_partial_c.resize(partial_elements);
  }
  SveKBlockParams params;
  params.gemm.m = m;
  params.gemm.k = K;
  params.gemm.n = N;
  params.gemm.lda = K;
  params.gemm.ldb = K;
  params.gemm.ldc = ldc;
  params.kc = ::fused_cpp::moe_sve::k_block(K);
  params.packed_n = packed_n;
  params.n_begin = n_begin;
  params.mode = mode;
  params.partial_c = sve_kblock_partial_c.data();
  return params;
}

using SveJitKernelFn = ::fused_cpp::moe_sve::jit::KernelFn;
using SveJitOperation = ::fused_cpp::moe_sve::jit::Operation;

const char* sve_jit_operation_name(SveJitOperation operation) {
  switch (operation) {
    case SveJitOperation::kW13:
      return "W13";
    case SveJitOperation::kW2:
      return "W2";
    case SveJitOperation::kW2Direct:
      return "W2 direct-route";
    case SveJitOperation::kGemmF32:
      return "plain GEMM FP32";
  }
  return "unknown";
}

bool sve_jit_bulk_m_enabled() {
  const char* value = std::getenv("FUSED_CPP_MOE_SVE_JIT_BULK_M");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool sve_jit_w13_first_panel_prefetch_enabled() {
  const char* value = std::getenv("FUSED_CPP_MOE_SVE_W13_FIRST_PANEL_PREFETCH");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool sve_jit_all_first_panel_prefetch_enabled() {
  const char* value = std::getenv("FUSED_CPP_MOE_SVE_FIRST_PANEL_PREFETCH");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool sve_jit_first_panel_prefetch_enabled(SveJitOperation operation) {
  return sve_jit_all_first_panel_prefetch_enabled() ||
         (operation == SveJitOperation::kW13 && sve_jit_w13_first_panel_prefetch_enabled());
}

bool sve_jit_any_first_panel_prefetch_enabled() {
  return sve_jit_all_first_panel_prefetch_enabled() || sve_jit_w13_first_panel_prefetch_enabled();
}

bool sve_jit_first_panel_prefetch_supported(SveJitOperation operation, int K) {
  const int64_t prefetch_distance_bytes = operation == SveJitOperation::kW13 ? 2048 : 1024;
  const int64_t b_tile_bytes = static_cast<int64_t>(K) * 2 * ::fused_cpp::moe_sve::n_tile();
  return b_tile_bytes > prefetch_distance_bytes;
}

bool sve_jit_configuration_supported(SveJitOperation operation, int K, int64_t degree, std::string* reason) {
  const auto mode = ::fused_cpp::moe_sve::jit::implementation_mode();
  if (mode == ::fused_cpp::moe_sve::jit::ImplementationMode::kAsm) {
    return false;
  }
  auto reject = [&](const std::string& message) {
    if (mode == ::fused_cpp::moe_sve::jit::ImplementationMode::kJit) {
      TORCH_CHECK(false, "FUSED_CPP_MOE_SVE_IMPL=jit cannot run ", sve_jit_operation_name(operation), ": ", message);
    }
    if (reason != nullptr) {
      *reason = message;
    }
    return false;
  };
  if (!::fused_cpp::moe_sve::jit::built()) {
    return reject("the extension was built without xbyak_aarch64");
  }
  if (::fused_cpp::moe_sve::k_block(K) != K) {
    return reject("split-K/Kc packing remains on the static asm fallback");
  }
  if (operation == SveJitOperation::kW13) {
    if (degree < 4 || degree > 6) {
      return reject("the exact-M JIT supports SiLU polynomial degrees 4, 5, and 6");
    }
    if (sve_w13_skip_silu_enabled()) {
      return reject("the identity epilogue remains on the static asm fallback");
    }
    if (sve_w13_silu_recip_nr_steps() != 0) {
      return reject("reciprocal-refinement SiLU remains on the static asm fallback");
    }
    if (degree == 5 && sve_w13_silu_minimax3_enabled()) {
      return reject("the minimax SiLU polynomial remains on the static asm fallback");
    }
  }
  return true;
}

void prewarm_sve_jit_exact_m_kernels(int w13_k, int w2_k) {
  if (!::fused_cpp::moe_sve::jit::requested_for_current_build()) {
    return;
  }
  if (::fused_cpp::moe_sve::k_block(w13_k) != w13_k || ::fused_cpp::moe_sve::k_block(w2_k) != w2_k) {
    return;
  }
  for (int degree = 4; degree <= 6; ++degree) {
    ::fused_cpp::moe_sve::jit::prewarm(SveJitOperation::kW13, degree);
  }
  ::fused_cpp::moe_sve::jit::prewarm(SveJitOperation::kW2, 0);
  ::fused_cpp::moe_sve::jit::prewarm(SveJitOperation::kW2Direct, 0);
  if (sve_jit_any_first_panel_prefetch_enabled()) {
    TORCH_CHECK(!sve_jit_bulk_m_enabled(), "SVE first-panel prefetch conflicts with FUSED_CPP_MOE_SVE_JIT_BULK_M");
  }
  if (sve_jit_first_panel_prefetch_enabled(SveJitOperation::kW13)) {
    for (int degree = 4; degree <= 6; ++degree) {
      ::fused_cpp::moe_sve::jit::prewarm_first_panel_prefetch(SveJitOperation::kW13, degree);
    }
  }
  if (sve_jit_first_panel_prefetch_enabled(SveJitOperation::kW2)) {
    ::fused_cpp::moe_sve::jit::prewarm_first_panel_prefetch(SveJitOperation::kW2, 0);
    ::fused_cpp::moe_sve::jit::prewarm_first_panel_prefetch(SveJitOperation::kW2Direct, 0);
  }
  if (sve_jit_bulk_m_enabled()) {
    for (int degree = 4; degree <= 6; ++degree) {
      ::fused_cpp::moe_sve::jit::prewarm_bulk_m12(SveJitOperation::kW13, degree);
    }
    ::fused_cpp::moe_sve::jit::prewarm_bulk_m12(SveJitOperation::kW2, 0);
    ::fused_cpp::moe_sve::jit::prewarm_bulk_m12(SveJitOperation::kW2Direct, 0);
  }
}

struct SveJitExactMKernelSet {
  SveJitKernelFn m12 = nullptr;
  SveJitKernelFn tail = nullptr;
  int main_rows = 0;
  int tail_rows = 0;
  bool bulk_m = false;
};

bool resolve_sve_jit_exact_m_kernels(SveJitOperation operation, int rows, int64_t degree, bool allow_bulk_m,
                                     SveJitExactMKernelSet* kernels) {
  TORCH_CHECK(rows > 0, "SVE JIT dispatch requires a positive row count");
  kernels->main_rows = rows / 12 * 12;
  kernels->tail_rows = rows - kernels->main_rows;
  kernels->bulk_m = allow_bulk_m && sve_jit_bulk_m_enabled() && kernels->main_rows >= 24;
  std::string error;
  if (kernels->main_rows > 0) {
    kernels->m12 = kernels->bulk_m
                       ? ::fused_cpp::moe_sve::jit::get_bulk_m12_kernel(operation, static_cast<int>(degree), &error)
                       : ::fused_cpp::moe_sve::jit::get_kernel(operation, 12, static_cast<int>(degree), &error);
  }
  if (kernels->tail_rows > 0 && kernels->m12 != nullptr) {
    kernels->tail =
        ::fused_cpp::moe_sve::jit::get_kernel(operation, kernels->tail_rows, static_cast<int>(degree), &error);
  } else if (kernels->tail_rows > 0 && kernels->main_rows == 0) {
    kernels->tail =
        ::fused_cpp::moe_sve::jit::get_kernel(operation, kernels->tail_rows, static_cast<int>(degree), &error);
  }
  const bool resolved =
      (kernels->main_rows == 0 || kernels->m12 != nullptr) && (kernels->tail_rows == 0 || kernels->tail != nullptr);
  if (!resolved &&
      ::fused_cpp::moe_sve::jit::implementation_mode() == ::fused_cpp::moe_sve::jit::ImplementationMode::kJit) {
    TORCH_CHECK(false, "failed to generate SVE Xbyak ", sve_jit_operation_name(operation), " kernel: ", error);
  }
  return resolved;
}

SveJitKernelFn resolve_sve_jit_first_panel_prefetch_kernel(SveJitOperation operation, int rows, int K, int64_t degree) {
  if (!sve_jit_first_panel_prefetch_enabled(operation)) {
    return nullptr;
  }
  TORCH_CHECK(!sve_jit_bulk_m_enabled(), "SVE first-panel prefetch conflicts with FUSED_CPP_MOE_SVE_JIT_BULK_M");
  if (!sve_jit_first_panel_prefetch_supported(operation, K)) {
    return nullptr;
  }
  std::string error;
  const SveJitKernelFn kernel = ::fused_cpp::moe_sve::jit::get_first_panel_prefetch_kernel(
      operation, std::min(rows, 12), static_cast<int>(degree), &error);
  TORCH_CHECK(kernel != nullptr, "failed to generate SVE Xbyak ", sve_jit_operation_name(operation),
              " first-panel prefetch kernel: ", error);
  return kernel;
}

bool sve_jit_packc_w13_exact_dispatch(const uint16_t* packed_A, const uint16_t* w13_packed, uint16_t* C, int rows,
                                      int K, int N, int ldc, int packed_N, int n_begin, int64_t degree) {
  if (!sve_jit_configuration_supported(SveJitOperation::kW13, K, degree, nullptr)) {
    return false;
  }
  const SveJitKernelFn first_panel_kernel =
      resolve_sve_jit_first_panel_prefetch_kernel(SveJitOperation::kW13, rows, K, degree);
  SveJitExactMKernelSet kernels;
  if (!resolve_sve_jit_exact_m_kernels(SveJitOperation::kW13, rows, degree, first_panel_kernel == nullptr, &kernels)) {
    return false;
  }
  SveKBlockParams p = make_sve_kblock_params(12, K, N, ldc, packed_N, n_begin, static_cast<int>(degree));
  const void* constants = ::fused_cpp::moe_sve::jit::silu_constants();
  if (kernels.bulk_m) {
    p.gemm.m = kernels.main_rows;
    kernels.m12(packed_A, w13_packed, C + static_cast<int64_t>(n_begin) * 6, constants, &p.gemm);
  } else {
    for (int mb = 0; mb < kernels.main_rows; mb += 12) {
      p.gemm.m = 12;
      const SveJitKernelFn kernel = mb == 0 && first_panel_kernel != nullptr ? first_panel_kernel : kernels.m12;
      kernel(packed_A + static_cast<int64_t>(mb) * K, w13_packed,
             C + static_cast<int64_t>(mb) * ldc + static_cast<int64_t>(n_begin) * 6, constants, &p.gemm);
    }
  }
  if (kernels.tail_rows > 0) {
    const int physical_pairs = kernels.tail_rows <= 8 ? 4 : 6;
    p.gemm.m = kernels.tail_rows;
    const SveJitKernelFn kernel =
        kernels.main_rows == 0 && first_panel_kernel != nullptr ? first_panel_kernel : kernels.tail;
    kernel(packed_A + static_cast<int64_t>(kernels.main_rows) * K, w13_packed,
           C + static_cast<int64_t>(kernels.main_rows) * ldc + static_cast<int64_t>(n_begin) * physical_pairs,
           constants, &p.gemm);
  }
  return true;
}

bool sve_jit_packed_w2_exact_dispatch(const uint16_t* packed_A, const uint16_t* w2_packed, float* down, int rows, int K,
                                      int N, int ldc, int packed_N, int n_begin) {
  if (!sve_jit_configuration_supported(SveJitOperation::kW2, K, 0, nullptr)) {
    return false;
  }
  const SveJitKernelFn first_panel_kernel =
      resolve_sve_jit_first_panel_prefetch_kernel(SveJitOperation::kW2, rows, K, 0);
  SveJitExactMKernelSet kernels;
  if (!resolve_sve_jit_exact_m_kernels(SveJitOperation::kW2, rows, 0, first_panel_kernel == nullptr, &kernels)) {
    return false;
  }
  SveKBlockParams p = make_sve_kblock_params(12, K, N, ldc, packed_N, n_begin, 0);
  if (kernels.bulk_m) {
    p.gemm.m = kernels.main_rows;
    kernels.m12(packed_A, w2_packed, down, nullptr, &p.gemm);
  } else {
    for (int mb = 0; mb < kernels.main_rows; mb += 12) {
      p.gemm.m = 12;
      const SveJitKernelFn kernel = mb == 0 && first_panel_kernel != nullptr ? first_panel_kernel : kernels.m12;
      kernel(packed_A + static_cast<int64_t>(mb) * K, w2_packed, down + static_cast<int64_t>(mb) * ldc, nullptr,
             &p.gemm);
    }
  }
  if (kernels.tail_rows > 0) {
    p.gemm.m = kernels.tail_rows;
    const SveJitKernelFn kernel =
        kernels.main_rows == 0 && first_panel_kernel != nullptr ? first_panel_kernel : kernels.tail;
    kernel(packed_A + static_cast<int64_t>(kernels.main_rows) * K, w2_packed,
           down + static_cast<int64_t>(kernels.main_rows) * ldc, nullptr, &p.gemm);
  }
  return true;
}

bool sve_jit_packed_gemm_f32_exact_dispatch(const uint16_t* packed_A, const uint16_t* packed_B, float* output,
                                            int rows, int K, int N, int ldc, int packed_N, int n_begin) {
  if (!sve_jit_configuration_supported(SveJitOperation::kGemmF32, K, 0, nullptr)) {
    return false;
  }
  SveJitExactMKernelSet kernels;
  if (!resolve_sve_jit_exact_m_kernels(SveJitOperation::kGemmF32, rows, 0, true, &kernels)) {
    return false;
  }
  SveKBlockParams p = make_sve_kblock_params(12, K, N, ldc, packed_N, n_begin, 0);
  if (kernels.bulk_m) {
    p.gemm.m = kernels.main_rows;
    kernels.m12(packed_A, packed_B, output, nullptr, &p.gemm);
  } else {
    for (int mb = 0; mb < kernels.main_rows; mb += 12) {
      p.gemm.m = 12;
      kernels.m12(packed_A + static_cast<int64_t>(mb) * K, packed_B,
                  output + static_cast<int64_t>(mb) * ldc, nullptr, &p.gemm);
    }
  }
  if (kernels.tail_rows > 0) {
    p.gemm.m = kernels.tail_rows;
    kernels.tail(packed_A + static_cast<int64_t>(kernels.main_rows) * K, packed_B,
                 output + static_cast<int64_t>(kernels.main_rows) * ldc, nullptr, &p.gemm);
  }
  return true;
}

bool sve_jit_packed_w2_direct_route_exact_dispatch(const uint16_t* packed_A, const uint16_t* w2_packed,
                                                   float* route_out, const int64_t* route_ids, int rows, int K, int N,
                                                   int route_stride, int packed_N, int n_begin) {
  if (!sve_jit_configuration_supported(SveJitOperation::kW2Direct, K, 0, nullptr)) {
    return false;
  }
  const SveJitKernelFn first_panel_kernel =
      resolve_sve_jit_first_panel_prefetch_kernel(SveJitOperation::kW2Direct, rows, K, 0);
  SveJitExactMKernelSet kernels;
  if (!resolve_sve_jit_exact_m_kernels(SveJitOperation::kW2Direct, rows, 0, first_panel_kernel == nullptr, &kernels)) {
    return false;
  }
  SveKBlockParams p = make_sve_kblock_params(12, K, N, route_stride, packed_N, n_begin, 3);
  if (kernels.bulk_m) {
    p.gemm.m = kernels.main_rows;
    kernels.m12(packed_A, w2_packed, route_out, route_ids, &p.gemm);
  } else {
    for (int mb = 0; mb < kernels.main_rows; mb += 12) {
      p.gemm.m = 12;
      const SveJitKernelFn kernel = mb == 0 && first_panel_kernel != nullptr ? first_panel_kernel : kernels.m12;
      kernel(packed_A + static_cast<int64_t>(mb) * K, w2_packed, route_out, route_ids + mb, &p.gemm);
    }
  }
  if (kernels.tail_rows > 0) {
    p.gemm.m = kernels.tail_rows;
    const SveJitKernelFn kernel =
        kernels.main_rows == 0 && first_panel_kernel != nullptr ? first_panel_kernel : kernels.tail;
    kernel(packed_A + static_cast<int64_t>(kernels.main_rows) * K, w2_packed, route_out, route_ids + kernels.main_rows,
           &p.gemm);
  }
  return true;
}

SveKcFusedSiluKernelSet sve_asm_fused_silu_packc_set_for_degree(int64_t degree) {
  const SveKcKernelFn m12 = moe_sve_kc_kernel_m12;
  const SveKcKernelFn m8 = moe_sve_kc_kernel_m8;
  const SveKcKernelFn m4 = moe_sve_kc_kernel_m4;
  const SveKcKernelFn m2 = moe_sve_kc_kernel_m2;
  if (sve_w13_skip_silu_enabled()) {
    return {m8, m4, m2, m2, m12, 1, 1};
  }
  const bool use_m12_opt = sve_w13_m12_epilogue_opt_enabled();
  const bool use_minimax3 = use_m12_opt && degree == 5 && sve_w13_silu_minimax3_enabled();
  const int recip_nr_steps = sve_w13_silu_recip_nr_steps();
  if (degree < 4 || degree > 6) {
    return {};
  }
  int m12_mode = static_cast<int>(degree);
  if (use_minimax3) {
    m12_mode = 43;
  } else if (use_m12_opt) {
    m12_mode += recip_nr_steps == 1 ? 20 : (recip_nr_steps == 2 ? 30 : 10);
  }
  return {m8, m4, m2, m2, m12, static_cast<int>(degree), m12_mode};
}

void sve_asm_packc_w13_tail_dispatch(const uint16_t* packed_A, const uint16_t* w13_packed, uint16_t* C, int rows, int K,
                                     int N, int ldc, int packed_N, int n_begin,
                                     const SveKcFusedSiluKernelSet& ks) {
  SveKBlockParams p = make_sve_kblock_params(8, K, N, ldc, packed_N, n_begin, ks.mode);
  const int nb_full = rows / 8;
  for (int mb = 0; mb < nb_full; ++mb) {
    p.gemm.m = 8;
    ks.m8(packed_A + static_cast<int64_t>(mb) * 8 * K, w13_packed,
          C + static_cast<int64_t>(mb) * 8 * ldc, nullptr, &p.gemm);
  }
  const int tail = rows - nb_full * 8;
  if (tail == 0) {
    return;
  }
  const uint16_t* At = packed_A + static_cast<int64_t>(nb_full) * 8 * K;
  uint16_t* Ct = C + static_cast<int64_t>(nb_full) * 8 * ldc;
  auto run = [&](SveKcKernelFn fn, int mr, int r0) {
    p.gemm.m = mr;
    fn(At + static_cast<int64_t>(r0) * 4, w13_packed, Ct + static_cast<int64_t>(r0) * 4, nullptr, &p.gemm);
  };
  switch (tail) {
    case 1:
      run(ks.m1, 1, 0);
      break;
    case 2:
      run(ks.m2, 2, 0);
      break;
    case 3:
      run(ks.m4, 4, 0);
      break;
    case 4:
      run(ks.m4, 4, 0);
      break;
    case 5:
    case 6:
    case 7:
      run(ks.m8, 8, 0);
      break;
  }
}

void sve_asm_packc_w13_hybrid_dispatch(const uint16_t* packed_A, const uint16_t* w13_packed, uint16_t* C, int rows,
                                       int K, int N, int ldc, int packed_N, int n_begin,
                                       const SveKcFusedSiluKernelSet& ks) {
  TORCH_CHECK(ks.m12 != nullptr, "SVE M12 fused silu kernel is unavailable");
  SveKBlockParams p = make_sve_kblock_params(12, K, N, ldc, packed_N, n_begin, ks.m12_mode);
  const int main_rows = static_cast<int>(sve_m12_main_rows(rows));
  if (main_rows > 0) {
    for (int mb = 0; mb < main_rows; mb += 12) {
      ks.m12(packed_A + static_cast<int64_t>(mb) * K, w13_packed,
             C + static_cast<int64_t>(mb) * ldc + static_cast<int64_t>(n_begin) * 6, nullptr, &p.gemm);
    }
  }
  const int tail = rows - main_rows;
  if (tail <= 0) {
    return;
  }
  sve_asm_packc_w13_tail_dispatch(packed_A + static_cast<int64_t>(main_rows) * K, w13_packed,
                                  C + static_cast<int64_t>(main_rows) * ldc + static_cast<int64_t>(n_begin) * 4, tail,
                                  K, N, ldc, packed_N, n_begin, ks);
}

void sve_asm_packed_w2_tail_dispatch(const uint16_t* packed_A, const uint16_t* w2_packed, float* down, int rows, int K,
                                     int N, int ldc, int packed_N, int n_begin) {
  SveKBlockParams p = make_sve_kblock_params(8, K, N, ldc, packed_N, n_begin, 0);
  const SveKcKernelFn k8 = moe_sve_kc_kernel_m8;
  const SveKcKernelFn k4 = moe_sve_kc_kernel_m4;
  const SveKcKernelFn k2 = moe_sve_kc_kernel_m2;
  const int nb_full = rows / 8;
  for (int mb = 0; mb < nb_full; ++mb) {
    p.gemm.m = 8;
    k8(packed_A + static_cast<int64_t>(mb) * 8 * K, w2_packed, down + static_cast<int64_t>(mb) * 8 * ldc,
       nullptr, &p.gemm);
  }
  const int tail = rows - nb_full * 8;
  if (tail == 0) {
    return;
  }
  const uint16_t* At = packed_A + static_cast<int64_t>(nb_full) * 8 * K;
  float* Dt = down + static_cast<int64_t>(nb_full) * 8 * ldc;
  auto run = [&](SveKcKernelFn fn, int mr, int r0) {
    p.gemm.m = mr;
    fn(At + static_cast<int64_t>(r0) * 4, w2_packed, Dt + static_cast<int64_t>(r0) * ldc, nullptr, &p.gemm);
  };
  switch (tail) {
    case 1:
      run(k2, 1, 0);
      break;
    case 2:
      run(k2, 2, 0);
      break;
    case 3:
      run(k4, 4, 0);
      break;
    case 4:
      run(k4, 4, 0);
      break;
    case 5:
    case 6:
    case 7:
      run(k8, 8, 0);
      break;
  }
}

void sve_asm_packed_w2_hybrid_dispatch(const uint16_t* packed_A, const uint16_t* w2_packed, float* down, int rows,
                                       int K, int N, int ldc, int packed_N, int n_begin) {
  SveKBlockParams p = make_sve_kblock_params(12, K, N, ldc, packed_N, n_begin, 0);
  const SveKcKernelFn k12 = moe_sve_kc_kernel_m12;
  const int main_rows = static_cast<int>(sve_m12_main_rows(rows));
  for (int mb = 0; mb < main_rows; mb += 12) {
    k12(packed_A + static_cast<int64_t>(mb) * K, w2_packed, down + static_cast<int64_t>(mb) * ldc, nullptr,
        &p.gemm);
  }
  const int tail = rows - main_rows;
  if (tail <= 0) {
    return;
  }
  sve_asm_packed_w2_tail_dispatch(packed_A + static_cast<int64_t>(main_rows) * K, w2_packed,
                                  down + static_cast<int64_t>(main_rows) * ldc, tail, K, N, ldc, packed_N, n_begin);
}

void sve_asm_packed_w2_direct_route_tail_dispatch(const uint16_t* packed_A, const uint16_t* w2_packed, float* route_out,
                                                  const int64_t* route_ids, int rows, int K, int N, int route_stride,
                                                  int packed_N, int n_begin) {
  SveKBlockParams p = make_sve_kblock_params(8, K, N, route_stride, packed_N, n_begin, 3);
  const SveKcKernelFn k8 = moe_sve_kc_kernel_m8;
  const SveKcKernelFn k4 = moe_sve_kc_kernel_m4;
  const SveKcKernelFn k2 = moe_sve_kc_kernel_m2;
  const int nb_full = rows / 8;
  for (int mb = 0; mb < nb_full; ++mb) {
    p.gemm.m = 8;
    k8(packed_A + static_cast<int64_t>(mb) * 8 * K, w2_packed, route_out,
       route_ids + static_cast<int64_t>(mb) * 8, &p.gemm);
  }
  const int tail = rows - nb_full * 8;
  if (tail == 0) {
    return;
  }
  const uint16_t* At = packed_A + static_cast<int64_t>(nb_full) * 8 * K;
  const int64_t* routes = route_ids + static_cast<int64_t>(nb_full) * 8;
  auto run = [&](SveKcKernelFn fn) {
    p.gemm.m = tail;
    fn(At, w2_packed, route_out, routes, &p.gemm);
  };
  switch (tail) {
    case 1:
      run(k2);
      break;
    case 2:
      run(k2);
      break;
    case 3:
    case 4:
      run(k4);
      break;
    case 5:
    case 6:
    case 7:
      run(k8);
      break;
  }
}

void sve_asm_packed_w2_direct_route_hybrid_dispatch(const uint16_t* packed_A, const uint16_t* w2_packed,
                                                    float* route_out, const int64_t* route_ids, int rows, int K, int N,
                                                    int route_stride, int packed_N, int n_begin) {
  SveKBlockParams p = make_sve_kblock_params(12, K, N, route_stride, packed_N, n_begin, 3);
  const SveKcKernelFn k12 = moe_sve_kc_kernel_m12;
  const int main_rows = static_cast<int>(sve_m12_main_rows(rows));
  for (int mb = 0; mb < main_rows; mb += 12) {
    p.gemm.m = std::min(12, rows - mb);
    k12(packed_A + static_cast<int64_t>(mb) * K, w2_packed, route_out, route_ids + mb, &p.gemm);
  }
  const int tail = rows - main_rows;
  if (tail <= 0) {
    return;
  }
  sve_asm_packed_w2_direct_route_tail_dispatch(packed_A + static_cast<int64_t>(main_rows) * K, w2_packed, route_out,
                                               route_ids + main_rows, tail, K, N, route_stride, packed_N, n_begin);
}

void sve_asm_packed_w2_direct_bf16_route_tail_dispatch(const uint16_t* packed_A, const uint16_t* w2_packed,
                                                       uint16_t* route_out, const int64_t* route_ids, int rows, int K,
                                                       int N, int route_stride, int packed_N, int n_begin) {
  SveKBlockParams p = make_sve_kblock_params(8, K, N, route_stride, packed_N, n_begin, 44);
  const SveKcKernelFn k8 = moe_sve_kc_kernel_m8;
  const SveKcKernelFn k4 = moe_sve_kc_kernel_m4;
  const SveKcKernelFn k2 = moe_sve_kc_kernel_m2;
  const int nb_full = rows / 8;
  for (int mb = 0; mb < nb_full; ++mb) {
    p.gemm.m = 8;
    k8(packed_A + static_cast<int64_t>(mb) * 8 * K, w2_packed, route_out,
       route_ids + static_cast<int64_t>(mb) * 8, &p.gemm);
  }
  const int tail = rows - nb_full * 8;
  if (tail == 0) {
    return;
  }
  const uint16_t* At = packed_A + static_cast<int64_t>(nb_full) * 8 * K;
  const int64_t* routes = route_ids + static_cast<int64_t>(nb_full) * 8;
  auto run = [&](SveKcKernelFn fn) {
    p.gemm.m = tail;
    fn(At, w2_packed, route_out, routes, &p.gemm);
  };
  switch (tail) {
    case 1:
      run(k2);
      break;
    case 2:
      run(k2);
      break;
    case 3:
    case 4:
      run(k4);
      break;
    case 5:
    case 6:
    case 7:
      run(k8);
      break;
  }
}

void sve_asm_packed_w2_direct_bf16_route_hybrid_dispatch(const uint16_t* packed_A, const uint16_t* w2_packed,
                                                         uint16_t* route_out, const int64_t* route_ids, int rows, int K,
                                                         int N, int route_stride, int packed_N, int n_begin) {
  SveKBlockParams p = make_sve_kblock_params(12, K, N, route_stride, packed_N, n_begin, 44);
  const SveKcKernelFn k12 = moe_sve_kc_kernel_m12;
  const int main_rows = static_cast<int>(sve_m12_main_rows(rows));
  for (int mb = 0; mb < main_rows; mb += 12) {
    p.gemm.m = std::min(12, rows - mb);
    k12(packed_A + static_cast<int64_t>(mb) * K, w2_packed, route_out, route_ids + mb, &p.gemm);
  }
  const int tail = rows - main_rows;
  if (tail <= 0) {
    return;
  }
  sve_asm_packed_w2_direct_bf16_route_tail_dispatch(
      packed_A + static_cast<int64_t>(main_rows) * K, w2_packed, route_out, route_ids + main_rows, tail, K, N,
      route_stride, packed_N, n_begin);
}

void sve_asm_packed_w2_bf16_tail_dispatch(const uint16_t* packed_A, const uint16_t* w2_packed, uint16_t* down, int rows,
                                          int K, int N, int ldc, int packed_N, int n_begin) {
  SveKBlockParams p = make_sve_kblock_params(8, K, N, ldc, packed_N, n_begin, 2);
  const SveKcKernelFn k8 = moe_sve_kc_kernel_m8;
  const SveKcKernelFn k4 = moe_sve_kc_kernel_m4;
  const SveKcKernelFn k2 = moe_sve_kc_kernel_m2;
  const int nb_full = rows / 8;
  for (int mb = 0; mb < nb_full; ++mb) {
    p.gemm.m = 8;
    k8(packed_A + static_cast<int64_t>(mb) * 8 * K, w2_packed, down + static_cast<int64_t>(mb) * 8 * ldc,
       nullptr, &p.gemm);
  }
  const int tail = rows - nb_full * 8;
  if (tail == 0) {
    return;
  }
  const uint16_t* At = packed_A + static_cast<int64_t>(nb_full) * 8 * K;
  uint16_t* Dt = down + static_cast<int64_t>(nb_full) * 8 * ldc;
  auto run = [&](SveKcKernelFn fn, int mr, int r0) {
    p.gemm.m = mr;
    fn(At + static_cast<int64_t>(r0) * 4, w2_packed, Dt + static_cast<int64_t>(r0) * ldc, nullptr, &p.gemm);
  };
  switch (tail) {
    case 1:
      run(k2, 1, 0);
      break;
    case 2:
      run(k2, 2, 0);
      break;
    case 3:
      run(k4, 4, 0);
      break;
    case 4:
      run(k4, 4, 0);
      break;
    case 5:
    case 6:
    case 7:
      run(k8, 8, 0);
      break;
  }
}

void sve_asm_packed_w2_bf16_hybrid_dispatch(const uint16_t* packed_A, const uint16_t* w2_packed, uint16_t* down,
                                            int rows, int K, int N, int ldc, int packed_N, int n_begin) {
  SveKBlockParams p = make_sve_kblock_params(12, K, N, ldc, packed_N, n_begin, 2);
  const SveKcKernelFn k12 = moe_sve_kc_kernel_m12;
  const int main_rows = static_cast<int>(sve_m12_main_rows(rows));
  for (int mb = 0; mb < main_rows; mb += 12) {
    k12(packed_A + static_cast<int64_t>(mb) * K, w2_packed, down + static_cast<int64_t>(mb) * ldc, nullptr,
        &p.gemm);
  }
  const int tail = rows - main_rows;
  if (tail <= 0) {
    return;
  }
  sve_asm_packed_w2_bf16_tail_dispatch(packed_A + static_cast<int64_t>(main_rows) * K, w2_packed,
                                       down + static_cast<int64_t>(main_rows) * ldc, tail, K, N, ldc, packed_N,
                                       n_begin);
}

void sve_packc_w13_hybrid_dispatch(const uint16_t* packed_A, const uint16_t* w13_packed, uint16_t* C, int rows,
                                   int K, int N, int ldc, int packed_N, int n_begin,
                                   const SveKcFusedSiluKernelSet& asm_kernels, int64_t degree) {
  if (sve_jit_packc_w13_exact_dispatch(packed_A, w13_packed, C, rows, K, N, ldc, packed_N, n_begin, degree)) {
    return;
  }
  sve_asm_packc_w13_hybrid_dispatch(packed_A, w13_packed, C, rows, K, N, ldc, packed_N, n_begin, asm_kernels);
}

void sve_packed_w2_hybrid_dispatch(const uint16_t* packed_A, const uint16_t* w2_packed, float* down, int rows, int K,
                                   int N, int ldc, int packed_N, int n_begin) {
  if (sve_jit_packed_w2_exact_dispatch(packed_A, w2_packed, down, rows, K, N, ldc, packed_N, n_begin)) {
    return;
  }
  sve_asm_packed_w2_hybrid_dispatch(packed_A, w2_packed, down, rows, K, N, ldc, packed_N, n_begin);
}

void sve_packed_w2_direct_route_hybrid_dispatch(const uint16_t* packed_A, const uint16_t* w2_packed,
                                                float* route_out, const int64_t* route_ids, int rows, int K, int N,
                                                int route_stride, int packed_N, int n_begin) {
  if (sve_jit_packed_w2_direct_route_exact_dispatch(packed_A, w2_packed, route_out, route_ids, rows, K, N,
                                                    route_stride, packed_N, n_begin)) {
    return;
  }
  sve_asm_packed_w2_direct_route_hybrid_dispatch(packed_A, w2_packed, route_out, route_ids, rows, K, N, route_stride,
                                                 packed_N, n_begin);
}
#endif

#endif  // __aarch64__

// Bottom-layer fused w13 + SiLU-and-mul over the whole [M] slice: interleaved
// w13 in B_reo, bf16 output C[M, ldc] (ldc = F_pad4), N = 2*F_pad4 packed
// columns. Dispatches m=8 blocks then 4/2/1-row tails, mirroring
// single_thread_gemm. `degree` picks the exp polynomial (4/5/6).
void single_thread_gemm_fused_silu(const uint16_t* A, const uint16_t* B_reo, uint16_t* C, uint16_t* A_reorder, int M,
                                   int K, int N, int ldc, int64_t degree) {
  const FusedSiluKernelSet ks = fused_silu_kernels_for_degree(degree);
  TORCH_CHECK(ks.m8 != nullptr, "unsupported fused silu exp degree ", degree, " (expected 4/5/6)");
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
    ks.m8(A, B_reo, C, A_reorder, &p);
    processed = m_full;
  }
  int m_rem = M - processed;
  auto advance = [&](int rows) {
    const uint16_t* At = A + static_cast<int64_t>(processed) * K;
    uint16_t* Ct = C + static_cast<int64_t>(processed) * ldc;
    uint16_t* Ar = A_reorder + static_cast<int64_t>(processed) * K;
    p.m = rows;
    p.k = K;
    p.n = N;
    FusedSiluKernelFn fn = (rows == 4) ? ks.m4 : (rows == 2) ? ks.m2 : ks.m1;
    fn(At, B_reo, Ct, Ar, &p);
    processed += rows;
  };
  if (m_rem >= 4) {
    advance(4);
    m_rem -= 4;
  }
  if (m_rem >= 2) {
    advance(2);
    m_rem -= 2;
  }
  if (m_rem >= 1) {
    advance(1);
  }
}

#endif

struct PackedExperts {
  at::Tensor tensor;
  int64_t E = 0;
  int64_t K = 0;
  int64_t N = 0;
  int64_t K_pad = 0;
  int64_t N_pad = 0;
  int64_t packed_stride = 0;
  int64_t n_tile = kKernelTile;
};

PackedExperts checked_packed_experts(const at::Tensor& packed, int64_t K, int64_t N, const char* name,
                                     int64_t n_tile = kKernelTile) {
  check_bf16_cpu(packed, name);
  TORCH_CHECK(packed.dim() == 2, name, " must be 2-D [experts, packed_numel]");
  TORCH_CHECK(packed.is_contiguous(), name, " must be contiguous");
  check_positive_int(K, "K");
  check_positive_int(N, "N");
  const int64_t K_pad = ceil_to_multiple(K, kKernelTile);
  const int64_t N_pad = ceil_to_multiple(N, std::max<int64_t>(n_tile, kKernelTile));
  const int64_t expected_stride = K_pad * N_pad;
  TORCH_CHECK(packed.size(1) == expected_stride, name, " packed stride mismatch: expected ", expected_stride, ", got ",
              packed.size(1));
  TORCH_CHECK(packed.size(0) > 0, name, " must contain at least one expert");
  return PackedExperts{
      packed, packed.size(0), K, N, K_pad, N_pad, expected_stride, std::max<int64_t>(n_tile, kKernelTile)};
}

void check_optional_bias(const c10::optional<at::Tensor>& bias, int64_t E, int64_t N, const char* name) {
  if (!bias.has_value() || !bias.value().defined()) {
    return;
  }
  const at::Tensor& b = bias.value();
  TORCH_CHECK(b.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(b.scalar_type() == at::kFloat || b.scalar_type() == at::kBFloat16, name,
              " must have dtype torch.float32 or torch.bfloat16");
  TORCH_CHECK(b.dim() == 2, name, " must be 2-D [experts, dim]");
  TORCH_CHECK(b.size(0) == E && b.size(1) == N, name, " shape mismatch: expected [", E, ", ", N, "], got [", b.size(0),
              ", ", b.size(1), "]");
  TORCH_CHECK(b.is_contiguous(), name, " must be contiguous");
}

void check_output_no_overlap(const at::Tensor& output, const at::Tensor& input, const char* name) {
  TORCH_CHECK(at::get_overlap_status(output, input) == at::MemOverlapStatus::No, "out must not overlap ", name);
}

at::Tensor prepare_moe_output(const at::Tensor& input, const at::Tensor& w13_packed, const at::Tensor& w2_packed,
                              const at::Tensor& topk_weights, const at::Tensor& topk_ids,
                              const c10::optional<at::Tensor>& out) {
  if (!out.has_value()) {
    return at::empty(input.sizes(), at::TensorOptions().device(input.device()).dtype(at::kBFloat16));
  }

  const at::Tensor& output = out.value();
  TORCH_CHECK(output.defined(), "out must be a defined tensor");
  check_bf16_cpu(output, "out");
  TORCH_CHECK(output.dim() == 2 && output.sizes() == input.sizes(), "out must have shape ", input.sizes(), ", got ",
              output.sizes());
  TORCH_CHECK(output.device() == input.device(), "out must be on the same device as input");
  TORCH_CHECK(output.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(!output.requires_grad(), "out with requires_grad=True is not supported");
  TORCH_CHECK(at::has_internal_overlap(output) == at::MemOverlap::No, "out must not have internal overlap");
  check_output_no_overlap(output, input, "input");
  check_output_no_overlap(output, w13_packed, "w13_packed");
  check_output_no_overlap(output, w2_packed, "w2_packed");
  check_output_no_overlap(output, topk_weights, "topk_weights");
  check_output_no_overlap(output, topk_ids, "topk_ids");
  return output;
}

at::Tensor finalize_moe_output(at::Tensor output, const c10::optional<at::Tensor>& out) {
  if (out.has_value()) {
    torch::autograd::impl::bump_version(output);
  }
  return output;
}

// Build a per-expert padded fp32 bias buffer [E, N_pad] from an optional
// [E, N] bias (fp32 or bf16). Padded columns [N, N_pad) are left zero so the
// fused-bias GEMM kernel contributes nothing there. Returns an undefined
// tensor when no bias is present, in which case callers fall back to the
// plain (non-bias) GEMM kernel.
at::Tensor build_padded_bias_f32(const c10::optional<at::Tensor>& bias, int64_t E, int64_t N, int64_t N_pad) {
  if (!bias.has_value() || !bias.value().defined()) {
    return at::Tensor();
  }
  at::Tensor out = at::zeros({E, N_pad}, at::TensorOptions().dtype(at::kFloat));
  float* dst = out.data_ptr<float>();
  const at::Tensor& b = bias.value();
  if (b.scalar_type() == at::kFloat) {
    const float* src = b.data_ptr<float>();
    for (int64_t e = 0; e < E; ++e) {
      std::copy(src + e * N, src + e * N + N, dst + e * N_pad);
    }
  } else {
    const uint16_t* src = bf16_data_const(b);
    for (int64_t e = 0; e < E; ++e) {
      for (int64_t n = 0; n < N; ++n) {
        dst[e * N_pad + n] = bf16_bits_to_float(src[e * N + n]);
      }
    }
  }
  return out;
}

void silu_and_mul_to_bf16(const float* gate_up, uint16_t* intermediate, int64_t rows, int64_t gate_up_stride,
                          int64_t intermediate_stride, int64_t F) {
  for (int64_t m = 0; m < rows; ++m) {
    const float* row = gate_up + m * gate_up_stride;
    uint16_t* out = intermediate + m * intermediate_stride;
    for (int64_t f = 0; f < F; ++f) {
      const float gate = row[f];
      const float up = row[F + f];
      const float silu = gate / (1.0f + std::exp(-gate));
      out[f] = bf16_bits_from_float(silu * up);
    }
  }
}

void gelu_and_mul_to_bf16(const float* gate_up, uint16_t* intermediate, int64_t rows, int64_t gate_up_stride,
                          int64_t intermediate_stride, int64_t F) {
  constexpr float kInvSqrt2 = 0.70710678118654752440f;
  for (int64_t m = 0; m < rows; ++m) {
    const float* row = gate_up + m * gate_up_stride;
    uint16_t* out = intermediate + m * intermediate_stride;
    for (int64_t f = 0; f < F; ++f) {
      const float gate = row[f];
      const float up = row[F + f];
      const float gelu = 0.5f * gate * (1.0f + std::erf(gate * kInvSqrt2));
      out[f] = bf16_bits_from_float(gelu * up);
    }
  }
}

void swigluoai_and_mul_to_bf16(const float* gate_up, uint16_t* intermediate, int64_t rows, int64_t gate_up_stride,
                               int64_t intermediate_stride, int64_t F) {
  constexpr float kAlpha = 1.702f;
  constexpr float kLimit = 7.0f;
  for (int64_t m = 0; m < rows; ++m) {
    const float* row = gate_up + m * gate_up_stride;
    uint16_t* out = intermediate + m * intermediate_stride;
    for (int64_t f = 0; f < F; ++f) {
      const float gate_raw = row[2 * f];
      const float up_raw = row[2 * f + 1];
      const float gate = std::min(gate_raw, kLimit);
      const float up = std::max(-kLimit, std::min(up_raw, kLimit));
      const float glu = gate / (1.0f + std::exp(-gate * kAlpha));
      out[f] = bf16_bits_from_float((up + 1.0f) * glu);
    }
  }
}

void activation_to_bf16(const std::string& activation, const float* gate_up, uint16_t* intermediate, int64_t rows,
                        int64_t gate_up_stride, int64_t intermediate_stride, int64_t F) {
  std::fill(intermediate, intermediate + rows * intermediate_stride, static_cast<uint16_t>(0));
  if (activation == "silu") {
    silu_and_mul_to_bf16(gate_up, intermediate, rows, gate_up_stride, intermediate_stride, F);
  } else if (activation == "gelu") {
    gelu_and_mul_to_bf16(gate_up, intermediate, rows, gate_up_stride, intermediate_stride, F);
  } else if (activation == "swigluoai") {
    swigluoai_and_mul_to_bf16(gate_up, intermediate, rows, gate_up_stride, intermediate_stride, F);
  } else {
    TORCH_CHECK(false, "unsupported MoE activation: ", activation);
  }
}

void activation_range_to_bf16(const std::string& activation, const float* gate_up, uint16_t* intermediate,
                              int64_t row_begin, int64_t rows, int64_t gate_up_stride, int64_t intermediate_stride,
                              int64_t F) {
  if (rows <= 0) {
    return;
  }
  const float* gate_up_slice = gate_up + row_begin * gate_up_stride;
  uint16_t* intermediate_slice = intermediate + row_begin * intermediate_stride;
  std::fill(intermediate_slice, intermediate_slice + rows * intermediate_stride, static_cast<uint16_t>(0));
  if (activation == "silu") {
    silu_and_mul_to_bf16(gate_up_slice, intermediate_slice, rows, gate_up_stride, intermediate_stride, F);
  } else if (activation == "gelu") {
    gelu_and_mul_to_bf16(gate_up_slice, intermediate_slice, rows, gate_up_stride, intermediate_stride, F);
  } else if (activation == "swigluoai") {
    swigluoai_and_mul_to_bf16(gate_up_slice, intermediate_slice, rows, gate_up_stride, intermediate_stride, F);
  } else {
    TORCH_CHECK(false, "unsupported MoE activation: ", activation);
  }
}

enum class MoeActivationKind {
  kSilu,
  kGelu,
  kSwigluOai,
};

MoeActivationKind moe_activation_kind(const std::string& activation) {
  if (activation == "silu") {
    return MoeActivationKind::kSilu;
  }
  if (activation == "gelu") {
    return MoeActivationKind::kGelu;
  }
  if (activation == "swigluoai") {
    return MoeActivationKind::kSwigluOai;
  }
  TORCH_CHECK(false, "unsupported MoE activation: ", activation);
}

struct ExpertTask {
  int64_t expert = 0;
  int64_t route_begin = 0;
  int64_t rows = 0;
};

struct TaskRange {
  size_t begin = 0;
  size_t end = 0;
};

struct ExpertTaskGroup {
  size_t begin = 0;
  size_t end = 0;
  int64_t rows = 0;
  int64_t threads = 1;
  int64_t extra_threads = 0;
  float last_marginal_gain = 0.0f;
};

float env_float_or_default_local(const char* name, float fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const float parsed = std::strtof(value, &end);
  if (end == value || parsed < 0.0f || !std::isfinite(parsed)) {
    return fallback;
  }
  return parsed;
}

int64_t env_int_or_default_local(const char* name, int64_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const long long parsed = std::strtoll(value, &end, 10);
  if (end == value) {
    return fallback;
  }
  return static_cast<int64_t>(parsed);
}

struct ExpertScheduleCostAlgorithmShape {
  float speedup_gain = 0.82f;
  float speedup_power = 0.72f;
  float alignment_penalty = 0.15f;
};

struct ExpertScheduleCostHardwareProfile {
  float fixed_overhead = 0.0f;
  float gemm_work_unit = 4194304.0f;
  float gemm_work_weight = 1.0f;
  float pack_work_unit = 65536.0f;
  float a_pack_weight = 0.10f;
  float weight_pack_weight = 0.0f;
  float activation_work_unit = 65536.0f;
  float activation_weight = 0.15f;
  float post_work_unit = 65536.0f;
  float post_weight = 0.08f;
  float thread_linear = 0.35f;
  float thread_quadratic = 0.02f;
  float m_split_sync = 0.05f;
  float n_split_sync = 0.08f;
  float m_split_cache = 0.02f;
  float n_split_cache = 0.08f;
  float n_split_repack_factor = 1.0f;
  float memory_token_weight = 0.0010f;
  float memory_thread_weight = 0.0500f;
  float memory_capacity = 8.0f;
  float memory_slowdown = 0.35f;
  float cache_token_weight = 0.0005f;
  float cache_thread_weight = 0.0300f;
  float cache_capacity = 8.0f;
  float cache_slowdown = 0.20f;
  float bandwidth_work_weight = 0.0010f;
  float bandwidth_thread_weight = 0.0500f;
  float bandwidth_capacity = 8.0f;
  float bandwidth_slowdown = 0.30f;
  float compute_slowdown = 0.25f;
  float active_gemm_slowdown = 0.02f;
};

struct ExpertScheduleCostRuntimeCalibration {
  std::atomic<float> scale{1.0f};
  std::atomic<float> bias{0.0f};
  std::atomic<float> hw_profile_drift{1.0f};
  std::atomic<uint64_t> observations{0};
  float update_alpha = 0.10f;
  int64_t update_interval = 1;

  ExpertScheduleCostRuntimeCalibration(float initial_scale, float initial_bias, float initial_hw_profile_drift,
                                       float alpha, int64_t interval)
      : scale(std::max(initial_scale, 1.0e-4f)),
        bias(initial_bias),
        hw_profile_drift(std::max(initial_hw_profile_drift, 1.0e-4f)),
        update_alpha(std::clamp(alpha, 0.0f, 1.0f)),
        update_interval(std::max<int64_t>(interval, 1)) {}
};

ExpertScheduleCostAlgorithmShape expert_schedule_algorithm_shape() {
  return ExpertScheduleCostAlgorithmShape{
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_SPEEDUP_GAIN", 0.82f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_SPEEDUP_POWER", 0.72f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_ALIGNMENT_PENALTY", 0.15f),
  };
}

ExpertScheduleCostHardwareProfile expert_schedule_hardware_profile() {
  return ExpertScheduleCostHardwareProfile{
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_FIXED_OVERHEAD", 0.0f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_GEMM_WORK_UNIT", 4194304.0f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_GEMM_WORK_WEIGHT", 1.0f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_PACK_WORK_UNIT", 65536.0f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_A_PACK_WEIGHT", 0.10f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_WEIGHT_PACK_WEIGHT", 0.0f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_ACTIVATION_WORK_UNIT", 65536.0f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_ACTIVATION_WEIGHT", 0.15f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_POST_WORK_UNIT", 65536.0f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_POST_WEIGHT", 0.08f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_THREAD_LINEAR", 0.35f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_THREAD_QUADRATIC", 0.02f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_M_SPLIT_SYNC", 0.05f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_N_SPLIT_SYNC", 0.08f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_M_SPLIT_CACHE", 0.02f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_N_SPLIT_CACHE", 0.08f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_N_SPLIT_REPACK_FACTOR", 1.0f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_MEMORY_TOKEN_WEIGHT", 0.0010f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_MEMORY_THREAD_WEIGHT", 0.0500f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_MEMORY_CAPACITY", 8.0f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_MEMORY_SLOWDOWN", 0.35f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_CACHE_TOKEN_WEIGHT", 0.0005f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_CACHE_THREAD_WEIGHT", 0.0300f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_CACHE_CAPACITY", 8.0f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_CACHE_SLOWDOWN", 0.20f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_BANDWIDTH_WORK_WEIGHT", 0.0010f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_BANDWIDTH_THREAD_WEIGHT", 0.0500f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_BANDWIDTH_CAPACITY", 8.0f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_BANDWIDTH_SLOWDOWN", 0.30f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_COMPUTE_SLOWDOWN", 0.25f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_ACTIVE_GEMM_SLOWDOWN", 0.02f),
  };
}

ExpertScheduleCostRuntimeCalibration& expert_schedule_runtime_calibration() {
  static ExpertScheduleCostRuntimeCalibration calibration(
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_INITIAL_SCALE", 1.0f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_INITIAL_BIAS", 0.0f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_INITIAL_HW_PROFILE_DRIFT", 1.0f),
      env_float_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_UPDATE_ALPHA", 0.10f),
      env_int_or_default_local("FUSED_CPP_MOE_SCHEDULE_COST_UPDATE_INTERVAL", 1));
  return calibration;
}

struct ExpertScheduleWorkloadConfig {
  int64_t w13_k = 0;
  int64_t w13_n = 0;
  int64_t w2_k = 0;
  int64_t w2_n = 0;
  MoeActivationKind activation = MoeActivationKind::kSilu;
  bool skip_weighted = false;
};

struct ExpertScheduleWorkloadShape {
  int64_t rows = 0;
  int64_t w13_k = 0;
  int64_t w13_n = 0;
  int64_t w2_k = 0;
  int64_t w2_n = 0;
  MoeActivationKind activation = MoeActivationKind::kSilu;
  bool skip_weighted = false;
};

struct ExpertScheduleGemmStageShape {
  MoeGemmStage stage = MoeGemmStage::kW13;
  int64_t M = 0;
  int64_t K = 0;
  int64_t N = 0;
};

struct ExpertScheduleStageCostEstimate {
  MoeGemmSplit split = MoeGemmSplit::kN;
  float compute = 0.0f;
  float pack = 0.0f;
  float split_overhead = 0.0f;
  float cache = 0.0f;
  float total = 0.0f;
};

struct ExpertScheduleStructuralCost {
  ExpertScheduleStageCostEstimate w13;
  ExpertScheduleStageCostEstimate w2;
  float activation = 0.0f;
  float post = 0.0f;
  float thread_penalty = 0.0f;
  float alignment = 0.0f;
  float total = 0.0f;
};

struct ExpertScheduleInterferenceVector {
  float compute = 0.0f;
  float memory = 0.0f;
  float bandwidth = 0.0f;
  float cache = 0.0f;
  float active_gemm = 0.0f;
};

struct ExpertScheduleCostEstimate {
  ExpertScheduleStructuralCost structural;
  float hardware_cost = 0.0f;
  ExpertScheduleInterferenceVector interference;
  float interference_score = 0.0f;
  float interference_cost = 0.0f;
  float total = 0.0f;
};

struct ExpertScheduleMarginalGain {
  float structural_gain = 0.0f;
  float interference_penalty = 0.0f;
  float total_gain = 0.0f;
};

ExpertScheduleWorkloadShape make_expert_schedule_workload(const ExpertScheduleWorkloadConfig& config, int64_t rows) {
  return ExpertScheduleWorkloadShape{
      std::max<int64_t>(rows, 0), config.w13_k,        config.w13_n, config.w2_k, config.w2_n,
      config.activation,          config.skip_weighted};
}

float normalize_schedule_work(double work, float unit) {
  const double safe_unit = static_cast<double>(std::max(unit, 1.0e-4f));
  const double normalized = std::max(0.0, work) / safe_unit;
  return static_cast<float>(std::min(normalized, 1.0e9));
}

float activation_cost_multiplier(MoeActivationKind activation) {
  switch (activation) {
    case MoeActivationKind::kSilu:
      return 1.0f;
    case MoeActivationKind::kGelu:
      return 1.35f;
    case MoeActivationKind::kSwigluOai:
      return 1.20f;
  }
  return 1.0f;
}

float estimate_parallel_speedup(float parallel_units, int64_t threads) {
  const ExpertScheduleCostAlgorithmShape shape = expert_schedule_algorithm_shape();
  const float thread_count = static_cast<float>(std::max<int64_t>(threads, 1));
  const float useful_threads = std::max(1.0f, std::min(std::max(parallel_units, 1.0f), thread_count));
  if (useful_threads <= 1.0f) {
    return 1.0f;
  }
  return 1.0f + shape.speedup_gain * std::pow(useful_threads - 1.0f, shape.speedup_power);
}

ExpertScheduleStageCostEstimate estimate_gemm_stage_split_cost(const ExpertScheduleGemmStageShape& stage,
                                                               int64_t threads, MoeGemmSplit split) {
  const ExpertScheduleCostHardwareProfile hw = expert_schedule_hardware_profile();
  const float thread_count = static_cast<float>(std::max<int64_t>(threads, 1));
  const float parallel_units = split == MoeGemmSplit::kM
                                   ? std::max(1.0f, static_cast<float>(stage.M) / static_cast<float>(kKernelTile))
                                   : std::max(1.0f, static_cast<float>(stage.N) / static_cast<float>(kKernelTile));
  const float speedup = estimate_parallel_speedup(parallel_units, threads);
  const float gemm_work = normalize_schedule_work(
      static_cast<double>(stage.M) * static_cast<double>(stage.K) * static_cast<double>(stage.N), hw.gemm_work_unit);
  const float compute = hw.gemm_work_weight * gemm_work / speedup;

  const float a_pack_repeat =
      split == MoeGemmSplit::kN ? 1.0f + (thread_count - 1.0f) * hw.n_split_repack_factor : 1.0f;
  const float a_pack =
      hw.a_pack_weight * normalize_schedule_work(static_cast<double>(stage.M) * static_cast<double>(stage.K) *
                                                     static_cast<double>(a_pack_repeat),
                                                 hw.pack_work_unit);
  const float b_pack =
      hw.weight_pack_weight *
      normalize_schedule_work(static_cast<double>(stage.K) * static_cast<double>(stage.N), hw.pack_work_unit);
  const float split_sync =
      thread_count <= 1.0f ? 0.0f
                           : (split == MoeGemmSplit::kM ? hw.m_split_sync : hw.n_split_sync) * (thread_count - 1.0f);
  const float split_cache_weight = split == MoeGemmSplit::kM ? hw.m_split_cache : hw.n_split_cache;
  const float split_cache =
      split_cache_weight * normalize_schedule_work(static_cast<double>(stage.M) * static_cast<double>(stage.N) *
                                                       static_cast<double>(thread_count),
                                                   hw.pack_work_unit);

  ExpertScheduleStageCostEstimate estimate;
  estimate.split = split;
  estimate.compute = compute;
  estimate.pack = a_pack + b_pack;
  estimate.split_overhead = split_sync;
  estimate.cache = split_cache;
  estimate.total = estimate.compute + estimate.pack + estimate.split_overhead + estimate.cache;
  return estimate;
}

ExpertScheduleStageCostEstimate estimate_gemm_stage_cost(const ExpertScheduleGemmStageShape& stage, int64_t threads) {
  const ExpertScheduleStageCostEstimate m_split = estimate_gemm_stage_split_cost(stage, threads, MoeGemmSplit::kM);
  const ExpertScheduleStageCostEstimate n_split = estimate_gemm_stage_split_cost(stage, threads, MoeGemmSplit::kN);
  const int safe_m =
      static_cast<int>(std::min<int64_t>(std::max<int64_t>(stage.M, 1), std::numeric_limits<int>::max()));
  const int safe_n =
      static_cast<int>(std::min<int64_t>(std::max<int64_t>(stage.N, 1), std::numeric_limits<int>::max()));
  const MoeGemmSplit preferred = choose_moe_gemm_split(stage.stage, safe_m, safe_n, threads);
  if (std::abs(m_split.total - n_split.total) <= 1.0e-4f) {
    return preferred == MoeGemmSplit::kM ? m_split : n_split;
  }
  return m_split.total < n_split.total ? m_split : n_split;
}

float estimate_activation_stage_cost(const ExpertScheduleWorkloadShape& workload) {
  const ExpertScheduleCostHardwareProfile hw = expert_schedule_hardware_profile();
  return hw.activation_weight * activation_cost_multiplier(workload.activation) *
         normalize_schedule_work(static_cast<double>(workload.rows) * static_cast<double>(workload.w2_k),
                                 hw.activation_work_unit);
}

float estimate_post_stage_cost(const ExpertScheduleWorkloadShape& workload) {
  const ExpertScheduleCostHardwareProfile hw = expert_schedule_hardware_profile();
  const float output_factor = workload.skip_weighted ? 1.15f : 0.90f;
  return hw.post_weight * output_factor *
         normalize_schedule_work(static_cast<double>(workload.rows) * static_cast<double>(workload.w2_n),
                                 hw.post_work_unit);
}

float estimate_shape_alignment_penalty(const ExpertScheduleWorkloadShape& workload) {
  const ExpertScheduleCostAlgorithmShape shape = expert_schedule_algorithm_shape();
  const int64_t row_tail = workload.rows % kKernelTile;
  const int64_t w13_k_tail = workload.w13_k % kKernelTile;
  const int64_t w13_n_tail = workload.w13_n % kKernelTile;
  const int64_t w2_k_tail = workload.w2_k % kKernelTile;
  const int64_t w2_n_tail = workload.w2_n % kKernelTile;
  const int64_t tail_count = (row_tail != 0 ? 1 : 0) + (w13_k_tail != 0 ? 1 : 0) + (w13_n_tail != 0 ? 1 : 0) +
                             (w2_k_tail != 0 ? 1 : 0) + (w2_n_tail != 0 ? 1 : 0);
  return shape.alignment_penalty * static_cast<float>(tail_count);
}

ExpertScheduleStructuralCost estimate_expert_schedule_structural_cost(const ExpertScheduleWorkloadShape& workload,
                                                                      int64_t threads) {
  const ExpertScheduleCostHardwareProfile hw = expert_schedule_hardware_profile();
  const float thread_count = static_cast<float>(std::max<int64_t>(threads, 1));
  const ExpertScheduleStageCostEstimate w13_cost = estimate_gemm_stage_cost(
      ExpertScheduleGemmStageShape{MoeGemmStage::kW13, workload.rows, workload.w13_k, workload.w13_n}, threads);
  const ExpertScheduleStageCostEstimate w2_cost = estimate_gemm_stage_cost(
      ExpertScheduleGemmStageShape{MoeGemmStage::kW2, workload.rows, workload.w2_k, workload.w2_n}, threads);
  const float activation_cost = estimate_activation_stage_cost(workload);
  const float post_cost = estimate_post_stage_cost(workload);
  const float thread_penalty =
      hw.thread_linear * (thread_count - 1.0f) + hw.thread_quadratic * thread_count * thread_count;
  const float alignment_cost = estimate_shape_alignment_penalty(workload);
  ExpertScheduleStructuralCost structural;
  structural.w13 = w13_cost;
  structural.w2 = w2_cost;
  structural.activation = activation_cost;
  structural.post = post_cost;
  structural.thread_penalty = thread_penalty;
  structural.alignment = alignment_cost;
  structural.total = hw.fixed_overhead + w13_cost.total + activation_cost + w2_cost.total + post_cost + thread_penalty +
                     alignment_cost;
  return structural;
}

float estimate_expert_schedule_hardware_cost(const ExpertScheduleStructuralCost& structural) {
  ExpertScheduleCostRuntimeCalibration& runtime = expert_schedule_runtime_calibration();
  const float scale = runtime.scale.load(std::memory_order_relaxed);
  const float drift = runtime.hw_profile_drift.load(std::memory_order_relaxed);
  const float bias = runtime.bias.load(std::memory_order_relaxed);
  return scale * drift * structural.total + bias;
}

struct ExpertScheduleResourceDemand {
  float compute_threads = 0.0f;
  float memory_pressure = 0.0f;
  float memory_bandwidth = 0.0f;
  float cache_pressure = 0.0f;
};

struct ExpertScheduleSystemState {
  float active_compute_threads = 0.0f;
  float active_memory_pressure = 0.0f;
  float active_memory_bandwidth = 0.0f;
  float active_cache_pressure = 0.0f;
  int64_t active_gemms = 0;
};

struct ExpertScheduleGemmInterval {
  float start = 0.0f;
  float end = 0.0f;
  ExpertScheduleResourceDemand demand;
};

ExpertScheduleResourceDemand estimate_expert_resource_demand(const ExpertScheduleWorkloadShape& workload,
                                                             int64_t threads) {
  const ExpertScheduleCostHardwareProfile hw = expert_schedule_hardware_profile();
  const float tokens = static_cast<float>(std::max<int64_t>(workload.rows, 0));
  const float thread_count = static_cast<float>(std::max<int64_t>(threads, 1));
  const float stream_work =
      normalize_schedule_work(static_cast<double>(workload.rows) *
                                  static_cast<double>(workload.w13_k + workload.w13_n + workload.w2_k + workload.w2_n),
                              hw.pack_work_unit);
  const float cache_work = normalize_schedule_work(
      static_cast<double>(workload.rows) * static_cast<double>(workload.w13_n + workload.w2_n), hw.pack_work_unit);
  return ExpertScheduleResourceDemand{
      thread_count,
      hw.memory_token_weight * tokens + hw.memory_thread_weight * thread_count,
      hw.bandwidth_work_weight * stream_work + hw.bandwidth_thread_weight * thread_count,
      hw.cache_token_weight * tokens + hw.cache_token_weight * cache_work + hw.cache_thread_weight * thread_count,
  };
}

ExpertScheduleInterferenceVector estimate_system_interference(const ExpertScheduleSystemState& state,
                                                              const ExpertScheduleResourceDemand& demand,
                                                              int64_t num_threads) {
  const ExpertScheduleCostHardwareProfile hw = expert_schedule_hardware_profile();
  const float compute_capacity = static_cast<float>(std::max<int64_t>(num_threads, 1));
  const float memory_capacity = std::max(hw.memory_capacity, 1.0e-4f);
  const float bandwidth_capacity = std::max(hw.bandwidth_capacity, 1.0e-4f);
  const float cache_capacity = std::max(hw.cache_capacity, 1.0e-4f);
  const float compute_over =
      std::max(0.0f, (state.active_compute_threads + demand.compute_threads) / compute_capacity - 1.0f);
  const float memory_over =
      std::max(0.0f, (state.active_memory_pressure + demand.memory_pressure) / memory_capacity - 1.0f);
  const float bandwidth_over =
      std::max(0.0f, (state.active_memory_bandwidth + demand.memory_bandwidth) / bandwidth_capacity - 1.0f);
  const float cache_over =
      std::max(0.0f, (state.active_cache_pressure + demand.cache_pressure) / cache_capacity - 1.0f);
  const float concurrent_gemms = static_cast<float>(std::max<int64_t>(state.active_gemms, 0));
  return ExpertScheduleInterferenceVector{hw.compute_slowdown * compute_over, hw.memory_slowdown * memory_over,
                                          hw.bandwidth_slowdown * bandwidth_over, hw.cache_slowdown * cache_over,
                                          hw.active_gemm_slowdown * concurrent_gemms};
}

float project_interference_vector(const ExpertScheduleInterferenceVector& interference) {
  const char* mode = std::getenv("FUSED_CPP_MOE_SCHEDULE_INTERFERENCE_PROJECTION");
  if (mode != nullptr && std::strcmp(mode, "max") == 0) {
    return std::max({interference.compute, interference.memory, interference.bandwidth, interference.cache,
                     interference.active_gemm});
  }
  return interference.compute + interference.memory + interference.bandwidth + interference.cache +
         interference.active_gemm;
}

ExpertScheduleCostEstimate estimate_expert_schedule_cost_estimate(const ExpertScheduleWorkloadShape& workload,
                                                                  int64_t threads,
                                                                  const ExpertScheduleSystemState& state,
                                                                  int64_t num_threads) {
  ExpertScheduleCostEstimate estimate;
  estimate.structural = estimate_expert_schedule_structural_cost(workload, threads);
  estimate.hardware_cost = estimate_expert_schedule_hardware_cost(estimate.structural);
  const ExpertScheduleResourceDemand demand = estimate_expert_resource_demand(workload, threads);
  estimate.interference = estimate_system_interference(state, demand, num_threads);
  estimate.interference_score = project_interference_vector(estimate.interference);
  estimate.interference_cost = std::max(0.0f, estimate.hardware_cost) * std::max(0.0f, estimate.interference_score);
  estimate.total = estimate.hardware_cost + estimate.interference_cost;
  return estimate;
}

float estimate_expert_schedule_cost(const ExpertScheduleWorkloadShape& workload, int64_t threads,
                                    const ExpertScheduleSystemState& state, int64_t num_threads) {
  return estimate_expert_schedule_cost_estimate(workload, threads, state, num_threads).total;
}

[[maybe_unused]] ExpertScheduleMarginalGain expert_schedule_marginal_gain_delta(
    const ExpertScheduleWorkloadShape& workload, int64_t threads, const ExpertScheduleSystemState& state,
    int64_t num_threads) {
  const ExpertScheduleCostEstimate current =
      estimate_expert_schedule_cost_estimate(workload, threads, state, num_threads);
  const ExpertScheduleCostEstimate next =
      estimate_expert_schedule_cost_estimate(workload, threads + 1, state, num_threads);
  float structural_gain = current.hardware_cost - next.hardware_cost;
  if (threads > 1) {
    const ExpertScheduleCostEstimate previous =
        estimate_expert_schedule_cost_estimate(workload, threads - 1, state, num_threads);
    const float previous_gain = std::max(0.0f, previous.hardware_cost - current.hardware_cost);
    structural_gain = std::min(std::max(0.0f, structural_gain), previous_gain);
  } else {
    structural_gain = std::max(0.0f, structural_gain);
  }
  const float interference_penalty = std::max(0.0f, next.interference_cost - current.interference_cost);
  return ExpertScheduleMarginalGain{structural_gain, interference_penalty,
                                    std::max(0.0f, structural_gain - interference_penalty)};
}

[[maybe_unused]] float expert_schedule_marginal_gain(const ExpertScheduleWorkloadShape& workload, int64_t threads,
                                                     const ExpertScheduleSystemState& state, int64_t num_threads) {
  return expert_schedule_marginal_gain_delta(workload, threads, state, num_threads).total_gain;
}

class ExpertScheduleSystemStateTracker {
 public:
  ExpertScheduleSystemState state_at(float time) const {
    ExpertScheduleSystemState state;
    for (const ExpertScheduleGemmInterval& interval : intervals_) {
      if (interval.start <= time && time < interval.end) {
        state.active_compute_threads += interval.demand.compute_threads;
        state.active_memory_pressure += interval.demand.memory_pressure;
        state.active_memory_bandwidth += interval.demand.memory_bandwidth;
        state.active_cache_pressure += interval.demand.cache_pressure;
        ++state.active_gemms;
      }
    }
    return state;
  }

  void add_interval(float start, float end, const ExpertScheduleResourceDemand& demand) {
    if (end <= start) {
      return;
    }
    intervals_.push_back(ExpertScheduleGemmInterval{start, end, demand});
  }

 private:
  std::vector<ExpertScheduleGemmInterval> intervals_;
};

enum class ExpertScheduleBeamPolicy {
  kLowLatency,
  kBalanced,
  kHighThroughput,
};

struct ExpertScheduleBeamState {
  ExpertScheduleBeamPolicy policy = ExpertScheduleBeamPolicy::kLowLatency;
  std::vector<std::vector<TaskRange>> ranges;
  std::vector<float> thread_load;
  ExpertScheduleSystemStateTracker system_state_tracker;
  float predicted_makespan = 0.0f;
  float predicted_core_time = 0.0f;
  float score = 0.0f;
};

int64_t expert_schedule_beam_width() {
  return std::clamp<int64_t>(env_int_or_default_local("FUSED_CPP_MOE_SCHEDULE_BEAM_WIDTH", 3), 1, 8);
}

float score_expert_schedule_beam(const ExpertScheduleBeamState& state) {
  if (state.thread_load.empty()) {
    return 0.0f;
  }
  const auto minmax = std::minmax_element(state.thread_load.begin(), state.thread_load.end());
  const float min_load = *minmax.first;
  const float max_load = *minmax.second;
  const float imbalance = max_load - min_load;
  const float mean_core_time =
      state.predicted_core_time / static_cast<float>(std::max<size_t>(state.thread_load.size(), 1));
  switch (state.policy) {
    case ExpertScheduleBeamPolicy::kLowLatency:
      return max_load;
    case ExpertScheduleBeamPolicy::kBalanced:
      return max_load + 0.10f * imbalance;
    case ExpertScheduleBeamPolicy::kHighThroughput:
      return max_load + 0.05f * mean_core_time;
  }
  return max_load;
}

bool better_expert_schedule_beam(const ExpertScheduleBeamState& lhs, const ExpertScheduleBeamState& rhs) {
  if (lhs.score != rhs.score) {
    return lhs.score < rhs.score;
  }
  if (lhs.predicted_makespan != rhs.predicted_makespan) {
    return lhs.predicted_makespan < rhs.predicted_makespan;
  }
  return lhs.predicted_core_time < rhs.predicted_core_time;
}

void update_expert_schedule_cost_feedback(float predicted_cost, double observed_latency_ms) {
  if (predicted_cost <= 0.0f || observed_latency_ms <= 0.0) {
    return;
  }
  ExpertScheduleCostRuntimeCalibration& runtime = expert_schedule_runtime_calibration();
  const uint64_t observation = runtime.observations.fetch_add(uint64_t{1}, std::memory_order_relaxed) + uint64_t{1};
  if ((observation % static_cast<uint64_t>(runtime.update_interval)) != uint64_t{0}) {
    return;
  }
  const float old_scale = runtime.scale.load(std::memory_order_relaxed);
  const float old_bias = runtime.bias.load(std::memory_order_relaxed);
  const float old_drift = runtime.hw_profile_drift.load(std::memory_order_relaxed);
  const float observed_ratio = static_cast<float>(observed_latency_ms) / predicted_cost;
  const float alpha = std::clamp(runtime.update_alpha, 0.0f, 1.0f);
  const float correction = (1.0f - alpha) + alpha * observed_ratio;
  const float new_scale = std::clamp(old_scale * correction, 1.0e-4f, 1.0e4f);
  const float drift_correction = (1.0f - 0.25f * alpha) + (0.25f * alpha) * observed_ratio;
  const float new_drift = std::clamp(old_drift * drift_correction, 1.0e-4f, 1.0e4f);
  const float residual = static_cast<float>(observed_latency_ms) - predicted_cost;
  const float new_bias = std::clamp((1.0f - alpha) * old_bias + alpha * residual, -1.0e4f, 1.0e4f);
  runtime.scale.store(new_scale, std::memory_order_relaxed);
  runtime.hw_profile_drift.store(new_drift, std::memory_order_relaxed);
  runtime.bias.store(new_bias, std::memory_order_relaxed);
}

struct ThreadScheduleDebug {
  int64_t rows = 0;
  int64_t tasks = 0;
  int64_t ranges = 0;
  double ms = 0.0;
  std::vector<int64_t> experts;
  std::vector<int64_t> expert_rows;
};

class ThreadBarrier {
 public:
  explicit ThreadBarrier(int64_t participants) : participants_(participants) {}

  // Sense/generation spin barrier. The MoE team stages are µs-scale, so a
  // mutex+condition_variable barrier (a futex + scheduler wakeup, ~µs, per
  // wait, x4-5 barriers per expert) dominated the "gap" overhead. All
  // participants always arrive, so a bounded spin with a yield fallback is
  // safe and much cheaper.
  void wait() {
    const int64_t gen = generation_.load(std::memory_order_acquire);
    if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == participants_) {
      // Last arrival: reset for the next round, then release the spinners.
      arrived_.store(0, std::memory_order_relaxed);
      generation_.store(gen + 1, std::memory_order_release);
      return;
    }
    int64_t spins = 0;
    while (generation_.load(std::memory_order_acquire) == gen) {
#if defined(__aarch64__)
      __asm__ __volatile__("yield" ::: "memory");
#endif
      if (++spins >= kSpinBudget) {
        std::this_thread::yield();
        spins = 0;
      }
    }
  }

 private:
  static constexpr int64_t kSpinBudget = 4096;
  const int64_t participants_;
  std::atomic<int64_t> arrived_{0};
  std::atomic<int64_t> generation_{0};
};

#ifdef __aarch64__
// ── Middle layer: cooperative team GEMM ──────────────────────────────────
// The upper layer owns the thread pool and hands each worker a TeamContext
// (its slot in the team + a shared barrier + per-thread A-reorder scratch).
// Every team member calls team_gemm with the same plan; each computes its
// slice via the bottom-layer single_thread_gemm, then waits on the barrier so
// the caller can safely read C once all members have finished this stage.
// group_size==1 (barrier==null) degenerates to a whole-slice single_thread_gemm.
struct TeamContext {
  int64_t group_size = 1;
  int64_t local_tid = 0;
  ThreadBarrier* barrier = nullptr;  // shared; may be null when group_size==1
  uint16_t* a_reorder = nullptr;     // per-thread scratch, >= slice_rows*K*2
};

void team_gemm(const TeamContext& team, const GemmSplitPlan& plan, const uint16_t* A, const uint16_t* B_packed,
               float* C, int64_t M, int64_t K, int64_t N, int64_t ldc, const float* bias) {
  const SplitRange range = team_gemm_split_range(plan, M, N, team.group_size, team.local_tid);
  if (range.size > 0) {
    if (plan.split == MoeGemmSplit::kM) {
      // Row split: each thread spans the full N; bias unchanged.
      single_thread_gemm(A + range.begin * K, B_packed, C + range.begin * ldc, team.a_reorder,
                         static_cast<int>(range.size), static_cast<int>(K), static_cast<int>(N), static_cast<int>(ldc),
                         bias);
    } else {
      // Column split: B sliced by kKernelTile blocks; bias offset by col.
      const int64_t start_block = range.begin / kKernelTile;
      const float* bias_slice = bias != nullptr ? bias + range.begin : nullptr;
      single_thread_gemm(A, B_packed + start_block * K * kKernelTile, C + range.begin, team.a_reorder,
                         static_cast<int>(M), static_cast<int>(K), static_cast<int>(range.size), static_cast<int>(ldc),
                         bias_slice);
    }
  }
  if (team.barrier != nullptr) {
    team.barrier->wait();
  }
}

// N-split cooperative fused w13 + SiLU-and-mul. Each team member computes its
// kN column slice of the interleaved 2F weight and writes its disjoint feature
// columns of `intermediate` (bf16, row stride = ldc). No internal barrier: the
// caller owns the inter-stage barrier (mirroring trace_dispatch_..._stage_split).
// `A` is the full [M,K] slice; `team.a_reorder` is this thread's private repack
// scratch (>= M*K*2), so the A-pack is duplicated across the team (accepted for
// the kN path; a shared pre-pack is a separate future optimization).
void team_fused_w13_silu(const TeamContext& team, const uint16_t* A, const uint16_t* w13_packed, uint16_t* intermediate,
                         int M, int K, int N13, int ldc, int64_t degree) {
  const GemmSplitPlan plan{MoeGemmSplit::kN};
  const SplitRange range = team_gemm_split_range(plan, M, N13, team.group_size, team.local_tid);
  if (range.size <= 0) {
    return;
  }
  const int64_t start_block = range.begin / kKernelTile;
  // 8 interleaved columns per block -> 4 output features; feature offset is
  // range.begin / 2, and this slice produces range.size / 2 features.
  single_thread_gemm_fused_silu(A, w13_packed + start_block * static_cast<int64_t>(K) * kKernelTile,
                                intermediate + range.begin / 2, team.a_reorder, M, K, static_cast<int>(range.size), ldc,
                                degree);
}

// Pack row-major A[total_rows, K] into the m8 reorder layout the bf16gemm
// cached path expects, for the 8-row blocks [block_begin, block_end). Rows
// beyond total_rows are zero-padded. `packed` holds ceil(total_rows/8)*8*K
// uint16. Cooperative: each team member packs its own block range once into a
// group-shared buffer, eliminating the per-member repack duplication.
void pack_a_reorder_m8(const uint16_t* A, uint16_t* packed, int total_rows, int K, int block_begin, int block_end) {
  const int kb_count = K / 4;
  for (int mb = block_begin; mb < block_end; ++mb) {
    uint16_t* blk = packed + static_cast<int64_t>(mb) * 8 * K;
    for (int kb = 0; kb < kb_count; ++kb) {
      uint16_t* base = blk + static_cast<int64_t>(kb) * 32;
      for (int r = 0; r < 8; ++r) {
        uint16_t* d = base + r * 4;
        const int gr = mb * 8 + r;
        if (gr < total_rows) {
          const uint16_t* src = A + static_cast<int64_t>(gr) * K + kb * 4;
          d[0] = src[0];
          d[1] = src[1];
          d[2] = src[2];
          d[3] = src[3];
        } else {
          d[0] = 0;
          d[1] = 0;
          d[2] = 0;
          d[3] = 0;
        }
      }
    }
  }
}

// Fused gather + m8 reorder pack: read token rows (token = expert_routes[gr] /
// top_k) straight from row-major input[H] into the m8 reorder layout the
// bf16gemm cached path expects, for 8-row blocks [block_begin, block_end).
// Eliminates the row-major scratch.input round-trip. Rows beyond total_rows and
// columns beyond H (K padding) are zero-filled. `packed` holds
// ceil(total_rows/8)*8*K_pad uint16. Bit-identical to gather-to-rowmajor(K_pad)
// followed by pack_a_reorder_m8.
void gather_pack_a_reorder_m8(const uint16_t* input, int64_t H, const int64_t* expert_routes, int64_t top_k,
                              uint16_t* packed, int total_rows, int K_pad, int block_begin, int block_end) {
  const int kb_count = K_pad / 4;
  const int h_full_kb = static_cast<int>(H) / 4;
  for (int mb = block_begin; mb < block_end; ++mb) {
    uint16_t* blk = packed + static_cast<int64_t>(mb) * 8 * K_pad;
    // Precompute the 8 row source pointers once (token = scattered).
    const uint16_t* srcs[8];
    for (int r = 0; r < 8; ++r) {
      const int gr = mb * 8 + r;
      srcs[r] = (gr < total_rows) ? input + (expert_routes[gr] / top_k) * H : nullptr;
    }
    // kb outer, r inner: the 8 rows x 4 cols of each K-block are written as
    // one contiguous 64-byte cache line before advancing. This avoids the
    // 8x write amplification of the row-outer order (which touched every
    // output line 8 times, 8 bytes each). Reads become 8 sequential streams.
    for (int kb = 0; kb < kb_count; ++kb) {
      uint16_t* d = blk + static_cast<int64_t>(kb) * 32;  // 64 bytes
      const int k0 = kb * 4;
      if (kb < h_full_kb) {
        for (int r = 0; r < 8; ++r) {
          const uint16_t* s = srcs[r];
          uint16_t* dr = d + r * 4;
          if (s != nullptr) {
            dr[0] = s[k0];
            dr[1] = s[k0 + 1];
            dr[2] = s[k0 + 2];
            dr[3] = s[k0 + 3];
          } else {
            dr[0] = 0;
            dr[1] = 0;
            dr[2] = 0;
            dr[3] = 0;
          }
        }
      } else {
        for (int r = 0; r < 8; ++r) {
          const uint16_t* s = srcs[r];
          uint16_t* dr = d + r * 4;
          for (int c = 0; c < 4; ++c) {
            const int kcol = k0 + c;
            dr[c] = (s != nullptr && kcol < H) ? s[kcol] : static_cast<uint16_t>(0);
          }
        }
      }
    }
  }
}

void gather_pack_a_reorder_m12(const uint16_t* input, int64_t H, const int64_t* expert_routes, int64_t top_k,
                               uint16_t* packed, int total_rows, int K_pad, int block_begin, int block_end) {
  const int kb_count = K_pad / 4;
  const int h_full_kb = static_cast<int>(H) / 4;
  for (int mb = block_begin; mb < block_end; ++mb) {
    uint16_t* blk = packed + static_cast<int64_t>(mb) * 12 * K_pad;
    const uint16_t* srcs[12];
    for (int r = 0; r < 12; ++r) {
      const int gr = mb * 12 + r;
      srcs[r] = (gr < total_rows) ? input + (expert_routes[gr] / top_k) * H : nullptr;
    }
    for (int kb = 0; kb < kb_count; ++kb) {
      uint16_t* d = blk + static_cast<int64_t>(kb) * 48;
      const int k0 = kb * 4;
      if (kb < h_full_kb) {
        for (int r = 0; r < 12; ++r) {
          const uint16_t* s = srcs[r];
          uint16_t* dr = d + r * 4;
          if (s != nullptr) {
            dr[0] = s[k0];
            dr[1] = s[k0 + 1];
            dr[2] = s[k0 + 2];
            dr[3] = s[k0 + 3];
          } else {
            dr[0] = 0;
            dr[1] = 0;
            dr[2] = 0;
            dr[3] = 0;
          }
        }
      } else {
        for (int r = 0; r < 12; ++r) {
          uint16_t* dr = d + r * 4;
          const uint16_t* s = srcs[r];
          for (int k = 0; k < 4; ++k) {
            const int kcol = k0 + k;
            dr[k] = (s != nullptr && kcol < H) ? s[kcol] : static_cast<uint16_t>(0);
          }
        }
      }
    }
  }
}

void gather_pack_a_reorder_sve_hybrid(const uint16_t* input, int64_t H, const int64_t* expert_routes, int64_t top_k,
                                      uint16_t* packed, int total_rows, int K_pad, int64_t group_size,
                                      int64_t local_tid) {
  const int64_t main_rows = sve_m12_main_rows(total_rows);
  const int64_t main_blocks = main_rows / 12;
  const SplitRange m12_range = split_evenly(main_blocks, group_size, local_tid);
  gather_pack_a_reorder_m12(input, H, expert_routes, top_k, packed, total_rows, K_pad,
                            static_cast<int>(m12_range.begin), static_cast<int>(m12_range.begin + m12_range.size));

  const int64_t tail_rows = total_rows - main_rows;
  if (tail_rows <= 0) {
    return;
  }
  const int64_t tail_blocks = ceil_div_int64(tail_rows, int64_t{8});
  const SplitRange tail_range = split_evenly(tail_blocks, group_size, local_tid);
  gather_pack_a_reorder_m8(input, H, expert_routes + main_rows, top_k, packed + main_rows * static_cast<int64_t>(K_pad),
                           static_cast<int>(tail_rows), K_pad, static_cast<int>(tail_range.begin),
                           static_cast<int>(tail_range.begin + tail_range.size));
}

// N-split fused w13 + SiLU-and-mul reading a group-shared pre-packed A (m8,
// M_padded a multiple of 8). Each member computes its kN feature-column slice
// and writes intermediate directly. No per-member A repack. Caller owns the
// inter-stage barrier.
void team_fused_w13_silu_packed(const TeamContext& team, const uint16_t* packed_A, const uint16_t* w13_packed,
                                uint16_t* intermediate, int M_padded, int K, int N13, int ldc, int64_t degree) {
  FusedSiluKernelFn k8 = fused_silu_packed_m8_for_degree(degree);
  TORCH_CHECK(k8 != nullptr,
              "packed-A fused w13 row-major store is unavailable; use "
              "FUSED_CPP_MOE_FUSED_PACKA=1 with packed w2/packC");
  const GemmSplitPlan plan{MoeGemmSplit::kN};
  const SplitRange range = team_gemm_split_range(plan, M_padded, N13, team.group_size, team.local_tid);
  if (range.size <= 0) {
    return;
  }
  const int64_t start_block = range.begin / kKernelTile;
  gemm_params_t p;
  p.m = M_padded;
  p.k = K;
  p.n = static_cast<int>(range.size);
  p.lda = K;
  p.ldb = K;
  p.ldc = ldc;
  k8(packed_A, w13_packed + start_block * static_cast<int64_t>(K) * kKernelTile, intermediate + range.begin / 2,
     nullptr, &p);
}

// N-split fused w13 + SiLU-and-mul with a PACKED-C store (Part 2): identical to
// team_fused_w13_silu_packed but the epilogue writes `intermediate` directly in
// the reorder-m8 layout w2 consumes as pre-packed A (no w2 repack). Each member
// owns a disjoint feature-column N-slice; the packed-C base for that slice is
// `intermediate + range.begin*4` uint16 (feature-block fb0 = range.begin/8, one
// K-block = 32 uint16). ldc = w2.K_pad. M_padded a multiple of 8.
void team_fused_w13_silu_packed_packc(const TeamContext& team, const uint16_t* packed_A, const uint16_t* w13_packed,
                                      uint16_t* intermediate, int rows, int K, int N13, int ldc, int64_t degree) {
  const FusedSiluKernelSet ks = fused_silu_packc_set_for_degree(degree);
  TORCH_CHECK(ks.m8 != nullptr, "unsupported fused silu exp degree ", degree);
  const GemmSplitPlan plan{MoeGemmSplit::kN};
  const int M_padded = (rows + 7) / 8 * 8;  // N-split range is over columns
  const SplitRange range = team_gemm_split_range(plan, M_padded, N13, team.group_size, team.local_tid);
  if (range.size <= 0) {
    return;
  }
  const int64_t start_block = range.begin / kKernelTile;
  // m8 over full 8-row blocks + per-tail (rows%8) packed dispatch on this
  // thread's N-slice. No padding-to-8 compute waste for small experts.
  packc_w13_tail_dispatch(packed_A, w13_packed + start_block * static_cast<int64_t>(K) * kKernelTile,
                          intermediate + range.begin * 4, rows, K, static_cast<int>(range.size), ldc, ks);
}

void team_fused_w13_silu_packed_packc_2d(const TeamContext& team, const Gemm2DSplitPlan& plan, const uint16_t* packed_A,
                                         const uint16_t* w13_packed, uint16_t* intermediate, int rows, int K, int N13,
                                         int ldc, int64_t degree) {
  const FusedSiluKernelSet ks = fused_silu_packc_set_for_degree(degree);
  TORCH_CHECK(ks.m8 != nullptr, "unsupported fused silu exp degree ", degree);
  const Gemm2DThreadRange range = gemm_2d_thread_range(plan, rows, N13, team.local_tid);
  if (range.rows <= 0 || range.n_cols <= 0) {
    return;
  }
  const int64_t start_block = range.n_begin / kKernelTile;
  packc_w13_tail_dispatch(
      packed_A + range.m_block_begin * kKernelTile * static_cast<int64_t>(K),
      w13_packed + start_block * static_cast<int64_t>(K) * kKernelTile,
      intermediate + range.m_block_begin * kKernelTile * static_cast<int64_t>(ldc) + range.n_begin * 4,
      static_cast<int>(range.rows), K, static_cast<int>(range.n_cols), ldc, ks);
}

void team_fused_w13_silu_packed_packc_sve(const TeamContext& team, const uint16_t* packed_A, const uint16_t* w13_packed,
                                          uint16_t* intermediate, int rows, int K, int N13, int ldc, int64_t degree,
                                          int64_t n_tile) {
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  const SveKcFusedSiluKernelSet ks = sve_asm_fused_silu_packc_set_for_degree(degree);
  TORCH_CHECK(ks.m8 != nullptr, "unsupported fused silu exp degree ", degree);
  const SplitRange range = n_split_range_tile(N13, team.group_size, team.local_tid, n_tile);
  if (range.size <= 0) {
    return;
  }
  sve_packc_w13_hybrid_dispatch(packed_A, w13_packed, intermediate, rows, K, static_cast<int>(range.size), ldc,
                                N13, static_cast<int>(range.begin), ks, degree);
#else
  (void)team;
  (void)packed_A;
  (void)w13_packed;
  (void)intermediate;
  (void)rows;
  (void)K;
  (void)N13;
  (void)ldc;
  (void)degree;
  (void)n_tile;
  TORCH_CHECK(false, "SVE MoE asm packC kernel is unavailable in this build");
#endif
}

// Explicit contiguous-N task used by the vLLM-style staged baseline. Unlike
// the cooperative team wrapper above, one caller owns the complete range and
// does not iterate the W13 ranges. This keeps the native GEMM body
// identical while making the task boundary match vLLM's (expert, N-range)
// work queue.
void vllm_staged_w13_range_sve(const uint16_t* packed_A, const uint16_t* w13_packed, uint16_t* intermediate,
                               int rows, int K, int ldc, int64_t degree, int64_t n_tile, int64_t n_begin,
                               int64_t n_cols) {
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  TORCH_CHECK(n_begin >= 0 && n_cols > 0 && n_begin % n_tile == 0 && n_cols % n_tile == 0,
              "vLLM-staged W13 range must be positive and N-tile aligned: begin=", n_begin, " size=", n_cols,
              " tile=", n_tile);
  const SveKcFusedSiluKernelSet kernels = sve_asm_fused_silu_packc_set_for_degree(degree);
  TORCH_CHECK(kernels.m8 != nullptr, "unsupported fused silu exp degree ", degree);
  sve_packc_w13_hybrid_dispatch(packed_A, w13_packed, intermediate, rows, K, static_cast<int>(n_cols), ldc, 2 * ldc,
                                static_cast<int>(n_begin), kernels, degree);
#else
  (void)packed_A;
  (void)w13_packed;
  (void)intermediate;
  (void)rows;
  (void)K;
  (void)ldc;
  (void)degree;
  (void)n_tile;
  (void)n_begin;
  (void)n_cols;
  TORCH_CHECK(false, "SVE MoE asm packC kernel is unavailable in this build");
#endif
}

void team_fused_w13_silu_packed_packc_sve_2d(const TeamContext& team, const Gemm2DSplitPlan& plan,
                                             const uint16_t* packed_A, const uint16_t* w13_packed,
                                             uint16_t* intermediate, int rows, int K, int N13, int ldc, int64_t degree,
                                             int64_t n_tile) {
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  const SveKcFusedSiluKernelSet ks = sve_asm_fused_silu_packc_set_for_degree(degree);
  TORCH_CHECK(ks.m8 != nullptr, "unsupported fused silu exp degree ", degree);
  Gemm2DSplitPlan n_plan = plan;
  n_plan.tn = plan.tm * plan.tn;
  n_plan.tm = 1;
  n_plan.n_tile = n_tile;
  const Gemm2DThreadRange range = gemm_2d_thread_range(n_plan, rows, N13, team.local_tid);
  if (range.rows <= 0 || range.n_cols <= 0) {
    return;
  }
  TORCH_CHECK(range.row_begin == 0, "SVE fused expert only supports N-split ranges");
  sve_packc_w13_hybrid_dispatch(packed_A, w13_packed, intermediate, static_cast<int>(range.rows), K,
                                static_cast<int>(range.n_cols), ldc, N13, static_cast<int>(range.n_begin), ks, degree);
#else
  (void)team;
  (void)plan;
  (void)packed_A;
  (void)w13_packed;
  (void)intermediate;
  (void)rows;
  (void)K;
  (void)N13;
  (void)ldc;
  (void)degree;
  (void)n_tile;
  TORCH_CHECK(false, "SVE MoE asm packC kernel is unavailable in this build");
#endif
}

void team_fused_w13_silu_sve(const TeamContext& team, const uint16_t* A, const uint16_t* w13_packed,
                             uint16_t* intermediate, int rows, int K, int N13, int ldc, uint16_t* a_reorder,
                             int64_t degree, int64_t n_tile) {
  const SplitRange range = n_split_range_tile(N13, team.group_size, team.local_tid, n_tile);
  if (range.size <= 0) {
    return;
  }
  const int64_t start_block = range.begin / n_tile;
  gemm_params_t p;
  p.m = rows;
  p.k = K;
  p.n = static_cast<int>(range.size);
  p.lda = K;
  p.ldb = K;
  p.ldc = ldc;
  ::fused_cpp::moe_sve::w13_silu_rowmajor(A, w13_packed + start_block * static_cast<int64_t>(K) * n_tile,
                                          intermediate + range.begin / 2, a_reorder, &p, degree);
}

// N-split plain fp32 w2 reading a PACKED intermediate (Part 2): each member
// computes its disjoint w2.N_pad output-column slice from the shared pre-packed
// A (the packed intermediate). No per-member repack, no a_reorder. down is
// row-major fp32 [M_padded, N], ldc = N. M_padded a multiple of 8.
void team_w2_packed(const TeamContext& team, const uint16_t* packed_A, const uint16_t* w2_packed, float* down, int rows,
                    int K, int N, int ldc) {
  const GemmSplitPlan plan{MoeGemmSplit::kN};
  const int M_padded = (rows + 7) / 8 * 8;
  const SplitRange range = team_gemm_split_range(plan, M_padded, N, team.group_size, team.local_tid);
  if (range.size <= 0) {
    return;
  }
  const int64_t start_block = range.begin / kKernelTile;
  // m8 full blocks + same per-tail dispatch as w13 (fp32 rowmajor store).
  packed_w2_tail_dispatch(packed_A, w2_packed + start_block * static_cast<int64_t>(K) * kKernelTile, down + range.begin,
                          rows, K, static_cast<int>(range.size), ldc);
}

void team_w2_packed_2d(const TeamContext& team, const Gemm2DSplitPlan& plan, const uint16_t* packed_A,
                       const uint16_t* w2_packed, float* down, int rows, int K, int N, int ldc) {
  const Gemm2DThreadRange range = gemm_2d_thread_range(plan, rows, N, team.local_tid);
  if (range.rows <= 0 || range.n_cols <= 0) {
    return;
  }
  const int64_t start_block = range.n_begin / kKernelTile;
  packed_w2_tail_dispatch(packed_A + range.m_block_begin * kKernelTile * static_cast<int64_t>(K),
                          w2_packed + start_block * static_cast<int64_t>(K) * kKernelTile,
                          down + range.row_begin * static_cast<int64_t>(ldc) + range.n_begin,
                          static_cast<int>(range.rows), K, static_cast<int>(range.n_cols), ldc);
}

void team_w2_packed_sve(const TeamContext& team, const uint16_t* packed_A, const uint16_t* w2_packed, float* down,
                        int rows, int K, int N, int ldc, int64_t n_tile) {
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  const SplitRange range = n_split_range_tile(N, team.group_size, team.local_tid, n_tile);
  if (range.size <= 0) {
    return;
  }
  sve_packed_w2_hybrid_dispatch(packed_A, w2_packed, down + range.begin, rows, K, static_cast<int>(range.size), ldc,
                                N, static_cast<int>(range.begin));
#else
  (void)team;
  (void)packed_A;
  (void)w2_packed;
  (void)down;
  (void)rows;
  (void)K;
  (void)N;
  (void)ldc;
  (void)n_tile;
  TORCH_CHECK(false, "SVE MoE asm w2 kernel is unavailable in this build");
#endif
}

void team_w2_packed_sve_2d(const TeamContext& team, const Gemm2DSplitPlan& plan, const uint16_t* packed_A,
                           const uint16_t* w2_packed, float* down, int rows, int K, int N, int ldc) {
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  const Gemm2DThreadRange range = gemm_2d_thread_range(plan, rows, N, team.local_tid);
  if (range.rows <= 0 || range.n_cols <= 0) {
    return;
  }
  TORCH_CHECK(range.row_begin == 0, "SVE fused expert only supports N-split ranges");
  sve_packed_w2_hybrid_dispatch(packed_A, w2_packed, down + range.n_begin, static_cast<int>(range.rows), K,
                                static_cast<int>(range.n_cols), ldc, N, static_cast<int>(range.n_begin));
#else
  (void)team;
  (void)plan;
  (void)packed_A;
  (void)w2_packed;
  (void)down;
  (void)rows;
  (void)K;
  (void)N;
  (void)ldc;
  TORCH_CHECK(false, "SVE MoE asm w2 kernel is unavailable in this build");
#endif
}

void team_w2_packed_sve_direct_route(const TeamContext& team, const uint16_t* packed_A, const uint16_t* w2_packed,
                                     float* route_out, const int64_t* route_ids, int rows, int K, int N,
                                     int route_stride, int64_t n_tile) {
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  const SplitRange range = n_split_range_tile(N, team.group_size, team.local_tid, n_tile);
  if (range.size <= 0) {
    return;
  }
  sve_packed_w2_direct_route_hybrid_dispatch(packed_A, w2_packed, route_out + range.begin, route_ids, rows, K,
                                             static_cast<int>(range.size), route_stride, N,
                                             static_cast<int>(range.begin));
#else
  (void)team;
  (void)packed_A;
  (void)w2_packed;
  (void)route_out;
  (void)route_ids;
  (void)rows;
  (void)K;
  (void)N;
  (void)route_stride;
  (void)n_tile;
  TORCH_CHECK(false, "SVE MoE asm direct-route w2 kernel is unavailable in this build");
#endif
}

void team_w2_packed_sve_direct_route_2d(const TeamContext& team, const Gemm2DSplitPlan& plan, const uint16_t* packed_A,
                                        const uint16_t* w2_packed, float* route_out, const int64_t* route_ids, int rows,
                                        int K, int N, int route_stride) {
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  const Gemm2DThreadRange range = gemm_2d_thread_range(plan, rows, N, team.local_tid);
  if (range.rows <= 0 || range.n_cols <= 0) {
    return;
  }
  TORCH_CHECK(range.row_begin == 0, "SVE fused expert only supports N-split ranges");
  sve_packed_w2_direct_route_hybrid_dispatch(packed_A, w2_packed, route_out + range.n_begin, route_ids,
                                             static_cast<int>(range.rows), K, static_cast<int>(range.n_cols),
                                             route_stride, N, static_cast<int>(range.n_begin));
#else
  (void)team;
  (void)plan;
  (void)packed_A;
  (void)w2_packed;
  (void)route_out;
  (void)route_ids;
  (void)rows;
  (void)K;
  (void)N;
  (void)route_stride;
  TORCH_CHECK(false, "SVE MoE asm direct-route w2 kernel is unavailable in this build");
#endif
}

void team_w2_packed_sve_direct_bf16_route(const TeamContext& team, const uint16_t* packed_A,
                                          const uint16_t* w2_packed, uint16_t* route_out, const int64_t* route_ids,
                                          int rows, int K, int N, int route_stride, int64_t n_tile) {
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  const SplitRange range = n_split_range_tile(N, team.group_size, team.local_tid, n_tile);
  if (range.size <= 0) {
    return;
  }
  sve_asm_packed_w2_direct_bf16_route_hybrid_dispatch(
      packed_A, w2_packed, route_out + range.begin, route_ids, rows, K, static_cast<int>(range.size), route_stride, N,
      static_cast<int>(range.begin));
#else
  (void)team;
  (void)packed_A;
  (void)w2_packed;
  (void)route_out;
  (void)route_ids;
  (void)rows;
  (void)K;
  (void)N;
  (void)route_stride;
  (void)n_tile;
  TORCH_CHECK(false, "SVE MoE asm direct-BF16-route w2 kernel is unavailable in this build");
#endif
}

void vllm_staged_w2_direct_route_range_sve(const uint16_t* packed_A, const uint16_t* w2_packed, float* route_out,
                                           const int64_t* route_ids, int rows, int K, int route_stride,
                                           int64_t n_tile, int64_t n_begin, int64_t n_cols) {
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  TORCH_CHECK(n_begin >= 0 && n_cols > 0 && n_begin % n_tile == 0 && n_cols % n_tile == 0,
              "vLLM-staged W2 range must be positive and N-tile aligned: begin=", n_begin, " size=", n_cols,
              " tile=", n_tile);
  sve_packed_w2_direct_route_hybrid_dispatch(packed_A, w2_packed, route_out + n_begin, route_ids, rows, K,
                                             static_cast<int>(n_cols), route_stride, route_stride,
                                             static_cast<int>(n_begin));
#else
  (void)packed_A;
  (void)w2_packed;
  (void)route_out;
  (void)route_ids;
  (void)rows;
  (void)K;
  (void)route_stride;
  (void)n_tile;
  (void)n_begin;
  (void)n_cols;
  TORCH_CHECK(false, "SVE MoE asm direct-route w2 kernel is unavailable in this build");
#endif
}

void vllm_staged_w2_direct_bf16_route_range_sve(const uint16_t* packed_A, const uint16_t* w2_packed,
                                                uint16_t* route_out, const int64_t* route_ids, int rows, int K,
                                                int route_stride, int64_t n_tile, int64_t n_begin, int64_t n_cols) {
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  TORCH_CHECK(n_begin >= 0 && n_cols > 0 && n_begin % n_tile == 0 && n_cols % n_tile == 0,
              "vLLM-staged W2 BF16 range must be positive and N-tile aligned: begin=", n_begin, " size=", n_cols,
              " tile=", n_tile);
  sve_asm_packed_w2_direct_bf16_route_hybrid_dispatch(
      packed_A, w2_packed, route_out + n_begin, route_ids, rows, K, static_cast<int>(n_cols), route_stride,
      route_stride, static_cast<int>(n_begin));
#else
  (void)packed_A;
  (void)w2_packed;
  (void)route_out;
  (void)route_ids;
  (void)rows;
  (void)K;
  (void)route_stride;
  (void)n_tile;
  (void)n_begin;
  (void)n_cols;
  TORCH_CHECK(false, "SVE MoE asm direct-BF16-route w2 kernel is unavailable in this build");
#endif
}

void team_w2_packed_sve_direct_bf16_route_2d(const TeamContext& team, const Gemm2DSplitPlan& plan,
                                             const uint16_t* packed_A, const uint16_t* w2_packed, uint16_t* route_out,
                                             const int64_t* route_ids, int rows, int K, int N, int route_stride) {
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  const Gemm2DThreadRange range = gemm_2d_thread_range(plan, rows, N, team.local_tid);
  if (range.rows <= 0 || range.n_cols <= 0) {
    return;
  }
  TORCH_CHECK(range.row_begin == 0, "SVE fused expert only supports N-split ranges");
  sve_asm_packed_w2_direct_bf16_route_hybrid_dispatch(
      packed_A, w2_packed, route_out + range.n_begin, route_ids, static_cast<int>(range.rows), K,
      static_cast<int>(range.n_cols), route_stride, N, static_cast<int>(range.n_begin));
#else
  (void)team;
  (void)plan;
  (void)packed_A;
  (void)w2_packed;
  (void)route_out;
  (void)route_ids;
  (void)rows;
  (void)K;
  (void)N;
  (void)route_stride;
  TORCH_CHECK(false, "SVE MoE asm direct-BF16-route w2 kernel is unavailable in this build");
#endif
}

void team_w2_packed_bf16_sve(const TeamContext& team, const uint16_t* packed_A, const uint16_t* w2_packed,
                             uint16_t* down, int rows, int K, int N, int ldc, int64_t n_tile) {
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  const SplitRange range = n_split_range_tile(N, team.group_size, team.local_tid, n_tile);
  if (range.size <= 0) {
    return;
  }
  sve_asm_packed_w2_bf16_hybrid_dispatch(packed_A, w2_packed, down + range.begin, rows, K,
                                         static_cast<int>(range.size), ldc, N, static_cast<int>(range.begin));
#else
  (void)team;
  (void)packed_A;
  (void)w2_packed;
  (void)down;
  (void)rows;
  (void)K;
  (void)N;
  (void)ldc;
  (void)n_tile;
  TORCH_CHECK(false, "SVE MoE asm bf16-output w2 kernel is unavailable in this build");
#endif
}

void team_w2_packed_bf16_sve_2d(const TeamContext& team, const Gemm2DSplitPlan& plan, const uint16_t* packed_A,
                                const uint16_t* w2_packed, uint16_t* down, int rows, int K, int N, int ldc) {
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  const Gemm2DThreadRange range = gemm_2d_thread_range(plan, rows, N, team.local_tid);
  if (range.rows <= 0 || range.n_cols <= 0) {
    return;
  }
  TORCH_CHECK(range.row_begin == 0, "SVE fused expert only supports N-split ranges");
  sve_asm_packed_w2_bf16_hybrid_dispatch(packed_A, w2_packed, down + range.n_begin,
                                         static_cast<int>(range.rows), K, static_cast<int>(range.n_cols), ldc, N,
                                         static_cast<int>(range.n_begin));
#else
  (void)team;
  (void)plan;
  (void)packed_A;
  (void)w2_packed;
  (void)down;
  (void)rows;
  (void)K;
  (void)N;
  (void)ldc;
  TORCH_CHECK(false, "SVE MoE asm bf16-output w2 kernel is unavailable in this build");
#endif
}

void team_w2_rowmajor_sve(const TeamContext& team, const uint16_t* A, const uint16_t* w2_packed, float* down, int rows,
                          int K, int N, int ldc, uint16_t* a_reorder, int64_t n_tile) {
  const SplitRange range = n_split_range_tile(N, team.group_size, team.local_tid, n_tile);
  if (range.size <= 0) {
    return;
  }
  const int64_t start_block = range.begin / n_tile;
  gemm_params_t p;
  p.m = rows;
  p.k = K;
  p.n = static_cast<int>(range.size);
  p.lda = K;
  p.ldb = K;
  p.ldc = ldc;
  ::fused_cpp::moe_sve::w2_rowmajor(A, w2_packed + start_block * static_cast<int64_t>(K) * n_tile, down + range.begin,
                                    a_reorder, &p);
}

void gather_pack_a_reorder_backend(bool use_sve_backend, const uint16_t* input, int64_t H, const int64_t* expert_routes,
                                   int64_t top_k, uint16_t* packed, int total_rows, int K_pad, int block_begin,
                                   int block_end) {
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  if (use_sve_backend) {
    ::fused_cpp::moe_sve::gather_pack_a(input, H, expert_routes, top_k, packed, total_rows, K_pad, block_begin,
                                        block_end);
    return;
  }
#else
  TORCH_CHECK(!use_sve_backend, "SVE MoE gather+packA is unavailable in this build");
#endif
  gather_pack_a_reorder_m8(input, H, expert_routes, top_k, packed, total_rows, K_pad, block_begin, block_end);
}

void team_fused_w13_silu_packed_packc_backend(bool use_sve_backend, bool use_2d_split, const TeamContext& team,
                                              const Gemm2DSplitPlan& plan, const uint16_t* packed_A,
                                              const uint16_t* w13_packed, uint16_t* intermediate, int rows, int K,
                                              int N13, int ldc, int64_t degree, int64_t n_tile) {
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  if (use_sve_backend) {
    if (use_2d_split) {
      team_fused_w13_silu_packed_packc_sve_2d(team, plan, packed_A, w13_packed, intermediate, rows, K, N13, ldc, degree,
                                              n_tile);
    } else {
      team_fused_w13_silu_packed_packc_sve(team, packed_A, w13_packed, intermediate, rows, K, N13, ldc, degree, n_tile);
    }
    return;
  }
#else
  TORCH_CHECK(!use_sve_backend, "SVE MoE asm packC kernel is unavailable in this build");
#endif
  if (use_2d_split) {
    team_fused_w13_silu_packed_packc_2d(team, plan, packed_A, w13_packed, intermediate, rows, K, N13, ldc, degree);
  } else {
    team_fused_w13_silu_packed_packc(team, packed_A, w13_packed, intermediate, rows, K, N13, ldc, degree);
  }
}

void team_w2_packed_backend(bool use_sve_backend, bool use_2d_split, const TeamContext& team,
                            const Gemm2DSplitPlan& plan, const uint16_t* packed_A, const uint16_t* w2_packed,
                            float* down, int rows, int K, int N, int ldc, int64_t n_tile) {
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  if (use_sve_backend) {
    if (use_2d_split) {
      team_w2_packed_sve_2d(team, plan, packed_A, w2_packed, down, rows, K, N, ldc);
    } else {
      team_w2_packed_sve(team, packed_A, w2_packed, down, rows, K, N, ldc, n_tile);
    }
    return;
  }
#else
  TORCH_CHECK(!use_sve_backend, "SVE MoE asm w2 kernel is unavailable in this build");
#endif
  if (use_2d_split) {
    team_w2_packed_2d(team, plan, packed_A, w2_packed, down, rows, K, N, ldc);
  } else {
    team_w2_packed(team, packed_A, w2_packed, down, rows, K, N, ldc);
  }
}

void team_w2_packed_sve_direct_route_backend(bool use_2d_split, const TeamContext& team, const Gemm2DSplitPlan& plan,
                                             const uint16_t* packed_A, const uint16_t* w2_packed, float* route_out,
                                             const int64_t* route_ids, int rows, int K, int N, int route_stride,
                                             int64_t n_tile) {
  if (use_2d_split) {
    team_w2_packed_sve_direct_route_2d(team, plan, packed_A, w2_packed, route_out, route_ids, rows, K, N,
                                       route_stride);
  } else {
    team_w2_packed_sve_direct_route(team, packed_A, w2_packed, route_out, route_ids, rows, K, N, route_stride, n_tile);
  }
}

void team_w2_packed_sve_direct_bf16_route_backend(
    bool use_2d_split, const TeamContext& team, const Gemm2DSplitPlan& plan, const uint16_t* packed_A,
    const uint16_t* w2_packed, uint16_t* route_out, const int64_t* route_ids, int rows, int K, int N, int route_stride,
    int64_t n_tile) {
  if (use_2d_split) {
    team_w2_packed_sve_direct_bf16_route_2d(team, plan, packed_A, w2_packed, route_out, route_ids, rows, K, N,
                                            route_stride);
  } else {
    team_w2_packed_sve_direct_bf16_route(team, packed_A, w2_packed, route_out, route_ids, rows, K, N, route_stride,
                                         n_tile);
  }
}

void team_w2_packed_bf16_sve_backend(bool use_2d_split, const TeamContext& team, const Gemm2DSplitPlan& plan,
                                     const uint16_t* packed_A, const uint16_t* w2_packed, uint16_t* down, int rows,
                                     int K, int N, int ldc, int64_t n_tile) {
  if (use_2d_split) {
    team_w2_packed_bf16_sve_2d(team, plan, packed_A, w2_packed, down, rows, K, N, ldc);
  } else {
    team_w2_packed_bf16_sve(team, packed_A, w2_packed, down, rows, K, N, ldc, n_tile);
  }
}
#endif  // __aarch64__

struct HierarchicalNSplitConfig {
  bool enabled = false;
  int64_t partitions = 2;
  int64_t groups_per_partition = 4;
  std::vector<int64_t> core_bases{0, 40};
};

struct MoeTraceConfig {
  bool enabled = false;
  std::string path = "/tmp/fused_cpp_moe_bf16_tiled_trace.log";
};

struct MoeGemmTraceRecord {
  uint64_t seq = 0;
  ::fused_cpp::profile::TimePoint begin_time;
  ::fused_cpp::profile::TimePoint end_time;
  int64_t tid = -1;
  int64_t cpu = -1;
  int64_t affinity_first_cpu = -1;
  int64_t affinity_cpu_count = 0;
  int64_t wave = -1;
  int64_t group = -1;
  int64_t local_tid = -1;
  int64_t expert = -1;
  int64_t route_begin = 0;
  int64_t rows = 0;
  const char* stage = "";
  int64_t M = 0;
  int64_t K = 0;
  int64_t N = 0;
  int64_t ldc = 0;
  int64_t n_begin = 0;
  int64_t n_cols = 0;
  double ms = 0.0;
};

struct MoePhaseTraceRecord {
  uint64_t seq = 0;
  ::fused_cpp::profile::TimePoint begin_time;
  ::fused_cpp::profile::TimePoint end_time;
  int64_t tid = -1;
  int64_t cpu = -1;
  int64_t affinity_first_cpu = -1;
  int64_t affinity_cpu_count = 0;
  int64_t wave = -1;
  int64_t group = -1;
  int64_t local_tid = -1;
  int64_t expert = -1;
  int64_t rows = 0;
  const char* stage = "";
  double start_ms = 0.0;
  double end_ms = 0.0;
  double ms = 0.0;
};

// Each logical worker has exactly one resident-thread writer. Keeping the
// vector control words on separate cache lines removes both locking and false
// sharing from the trace hot path; records are merged only after workers join.
struct alignas(128) MoeThreadTraceBuffer {
  std::vector<MoeGemmTraceRecord> gemm_records;
  std::vector<MoePhaseTraceRecord> phase_records;
  int64_t affinity_first_cpu = -1;
  int64_t affinity_cpu_count = 0;
  int64_t fixed_cpu = -1;
  bool metadata_initialized = false;
};

// Scratch-buffer page backing is decided by csrc/page_policy.h, which owns the
// single environment surface (FUSED_CPP_PAGES / FUSED_CPP_PAGE_SIZE_MB /
// FUSED_CPP_HUGETLBFS_PATH) and still honours the older FUSED_CPP_MOE_HUGETLB,
// FUSED_CPP_MOE_HUGETLB_MB and FUSED_CPP_MOE_THP names as deprecated aliases.
// The names below are kept so existing call sites read unchanged.
enum class ScratchBackend { kMalloc, kThp, kHugetlb };

inline ScratchBackend scratch_backend() {
  switch (fused_cpp::page_config().policy) {
    case fused_cpp::PagePolicy::kHugetlb:
      return ScratchBackend::kHugetlb;
    case fused_cpp::PagePolicy::kThp:
      return ScratchBackend::kThp;
    case fused_cpp::PagePolicy::kSmall:
      break;
  }
  return ScratchBackend::kMalloc;
}

inline size_t scratch_hugetlb_bytes() { return fused_cpp::page_config().hugetlb_bytes; }

inline size_t round_up_pow2(size_t x, size_t p) { return (x + p - 1) & ~(p - 1); }

template <typename T>
using backend_allocator = fused_cpp::PageAllocator<T>;

// Allocator that default-initializes (rather than value-initializes) elements,
// so vector::resize() on a trivial type allocates WITHOUT zeroing. Used for
// scratch buffers that are fully overwritten before being read (packed_a is
// fully written by the gather+pack; down is fully written by w2). intermediate
// keeps the default zeroing allocator because w2 reads its padding
// feature-blocks, which must stay zero.
template <typename T, typename Base = std::allocator<T>>
struct default_init_allocator : Base {
  using base_traits = std::allocator_traits<Base>;
  template <typename U>
  struct rebind {
    using other = default_init_allocator<U, typename base_traits::template rebind_alloc<U>>;
  };
  using Base::Base;
  default_init_allocator() noexcept = default;
  template <typename U>
  void construct(U* ptr) noexcept(std::is_nothrow_default_constructible<U>::value) {
    ::new (static_cast<void*>(ptr)) U;  // default-init: no zero for scalars
  }
  template <typename U, typename... Args>
  void construct(U* ptr, Args&&... args) {
    base_traits::construct(static_cast<Base&>(*this), ptr, std::forward<Args>(args)...);
  }
};

struct HierarchicalGroupScratch {
  explicit HierarchicalGroupScratch(int64_t group_size) : barrier(group_size) {}

  std::vector<uint16_t, backend_allocator<uint16_t>> input;
  std::vector<uint16_t, backend_allocator<uint16_t>> intermediate;
  std::vector<uint16_t, backend_allocator<uint16_t>> a_reorder;
  std::vector<uint16_t, default_init_allocator<uint16_t, backend_allocator<uint16_t>>> packed_a;
  std::vector<float, backend_allocator<float>> gate_up;
  std::vector<float, default_init_allocator<float, backend_allocator<float>>> down;
  std::vector<uint16_t, default_init_allocator<uint16_t, backend_allocator<uint16_t>>> down_bf16;
  std::atomic<int64_t> current_expert{-1};
  ThreadBarrier barrier;
};

// Persistent, per-calling-thread scratch pool. Buffers only grow and are reused
// across calls, so mmap + first-touch faults happen once, not every call.
// Correct because packed_a/down and every packed intermediate row consumed by
// W2 are fully overwritten before read. Keyed by group_size (the barrier is
// sized to it; a change rebuilds the pool). Memory is retained at the max size
// ever seen.
struct HierarchicalScratchPool {
  int64_t group_size = -1;
  std::vector<std::unique_ptr<HierarchicalGroupScratch>> groups;
  template <typename V>
  static void ensure(V& v, size_t need) {
    if (v.size() < need) v.resize(need);  // grow only; reuse warm otherwise
  }
};

bool env_flag_enabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool env_has_value(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0';
}

bool env_flag_enabled_by_default(const char* name) { return !env_has_value(name) || env_flag_enabled(name); }

int64_t env_int_or_default(const char* name, int64_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const long long parsed = std::strtoll(value, &end, 10);
  if (end == value) {
    return fallback;
  }
  return static_cast<int64_t>(parsed);
}

std::vector<int64_t> env_int_list_or_default(const char* name, const std::vector<int64_t>& fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }

  std::vector<int64_t> result;
  const char* cursor = value;
  while (*cursor != '\0') {
    while (*cursor == ',' || *cursor == ' ' || *cursor == '\t') {
      ++cursor;
    }
    if (*cursor == '\0') {
      break;
    }
    char* end = nullptr;
    const long long parsed = std::strtoll(cursor, &end, 10);
    if (end == cursor) {
      return fallback;
    }
    result.push_back(static_cast<int64_t>(parsed));
    cursor = end;
  }
  return result.empty() ? fallback : result;
}

int64_t current_affinity_first_cpu() {
#ifdef __linux__
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  if (pthread_getaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) == 0) {
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (CPU_ISSET(cpu, &cpuset)) {
        return static_cast<int64_t>(cpu);
      }
    }
  }
#endif
  return 0;
}

std::vector<int64_t> current_affinity_cpus(int64_t fallback_threads) {
  std::vector<int64_t> cpus;
#ifdef __linux__
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  if (pthread_getaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) == 0) {
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (CPU_ISSET(cpu, &cpuset)) {
        cpus.push_back(static_cast<int64_t>(cpu));
      }
    }
  }
#endif
  if (cpus.empty()) {
    const int64_t count = std::max<int64_t>(fallback_threads, 1);
    cpus.reserve(static_cast<size_t>(count));
    for (int64_t cpu = 0; cpu < count; ++cpu) {
      cpus.push_back(cpu);
    }
  }
  return cpus;
}

std::vector<int64_t> relative_core_bases_from_affinity(int64_t core_skip) {
  const int64_t first_cpu = current_affinity_first_cpu();
  return std::vector<int64_t>{first_cpu, first_cpu + core_skip};
}

HierarchicalNSplitConfig hierarchical_nsplit_config_from_env() {
  HierarchicalNSplitConfig config;
  config.enabled = env_flag_enabled("FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT");
  config.groups_per_partition = env_int_or_default("FUSED_CPP_MOE_N_SPLIT_GROUPS_PER_PARTITION", 4);
  if (env_has_value("FUSED_CPP_MOE_N_SPLIT_CORE_BASES")) {
    config.core_bases = env_int_list_or_default("FUSED_CPP_MOE_N_SPLIT_CORE_BASES", std::vector<int64_t>{0, 40});
  } else if (env_has_value("FUSED_CPP_MOE_N_SPLIT_CORE_SKIP")) {
    const int64_t core_skip = env_int_or_default("FUSED_CPP_MOE_N_SPLIT_CORE_SKIP", 40);
    TORCH_CHECK(core_skip > 0, "FUSED_CPP_MOE_N_SPLIT_CORE_SKIP must be positive, got ", core_skip);
    config.core_bases = relative_core_bases_from_affinity(core_skip);
  }
  config.partitions = static_cast<int64_t>(config.core_bases.size());
  return config;
}

MoeTraceConfig moe_trace_config_from_env() {
  MoeTraceConfig config;
  config.enabled = env_flag_enabled("FUSED_CPP_MOE_TRACE");
  const char* path = std::getenv("FUSED_CPP_MOE_TRACE_FILE");
  if (path != nullptr && path[0] != '\0') {
    config.path = path;
  }
  return config;
}

std::atomic<uint64_t> g_moe_trace_call_id{0};

class MoeTraceCollector {
 public:
  explicit MoeTraceCollector(MoeTraceConfig config)
      : config_(std::move(config)),
        call_id_(config_.enabled ? g_moe_trace_call_id.fetch_add(uint64_t{1}, std::memory_order_relaxed) : 0),
        origin_(::fused_cpp::profile::now()) {}

  bool enabled() const { return config_.enabled; }

  void prepare_thread_buffers(int64_t num_threads, size_t gemm_count_hint, size_t phase_count_hint) {
    if (!enabled() || num_threads <= 0) {
      return;
    }
    thread_buffers_.resize(static_cast<size_t>(num_threads));
    const size_t thread_count = static_cast<size_t>(num_threads);
    const size_t gemm_per_thread = (gemm_count_hint + thread_count - 1) / thread_count + 4;
    const size_t phase_per_thread = (phase_count_hint + thread_count - 1) / thread_count + 8;
    for (MoeThreadTraceBuffer& buffer : thread_buffers_) {
      buffer.gemm_records.reserve(gemm_per_thread);
      buffer.phase_records.reserve(phase_per_thread);
    }
    host_buffer_.gemm_records.reserve(16);
    host_buffer_.phase_records.reserve(16);
  }

  void record_gemm(int64_t tid, int64_t wave, int64_t group, int64_t local_tid, int64_t expert, int64_t route_begin,
                   int64_t rows, const char* stage, int64_t M, int64_t K, int64_t N, int64_t ldc, int64_t n_begin,
                   int64_t n_cols, ::fused_cpp::profile::TimePoint begin,
                   ::fused_cpp::profile::TimePoint end) {
    if (!enabled()) {
      return;
    }
    MoeThreadTraceBuffer& buffer = buffer_for_tid(tid);
    MoeGemmTraceRecord& record = buffer.gemm_records.emplace_back();
    record.begin_time = begin;
    record.end_time = end;
    record.tid = tid;
    record.cpu = record_cpu(buffer);
    record.affinity_first_cpu = buffer.affinity_first_cpu;
    record.affinity_cpu_count = buffer.affinity_cpu_count;
    record.wave = wave;
    record.group = group;
    record.local_tid = local_tid;
    record.expert = expert;
    record.route_begin = route_begin;
    record.rows = rows;
    record.stage = stage;
    record.M = M;
    record.K = K;
    record.N = N;
    record.ldc = ldc;
    record.n_begin = n_begin;
    record.n_cols = n_cols;
  }

  void record_phase(int64_t tid, int64_t wave, int64_t group, int64_t local_tid, int64_t expert, int64_t rows,
                    const char* stage, ::fused_cpp::profile::TimePoint begin) {
    if (!enabled()) {
      return;
    }
    const ::fused_cpp::profile::TimePoint end = ::fused_cpp::profile::now();
    MoeThreadTraceBuffer& buffer = buffer_for_tid(tid);
    MoePhaseTraceRecord& record = buffer.phase_records.emplace_back();
    record.begin_time = begin;
    record.end_time = end;
    record.tid = tid;
    record.cpu = record_cpu(buffer);
    record.affinity_first_cpu = buffer.affinity_first_cpu;
    record.affinity_cpu_count = buffer.affinity_cpu_count;
    record.wave = wave;
    record.group = group;
    record.local_tid = local_tid;
    record.expert = expert;
    record.rows = rows;
    record.stage = stage;
  }

  void write_report(const char* strategy, int64_t num_threads, int64_t num_tokens, int64_t top_k, int64_t num_experts,
                    int64_t num_routes, int64_t hidden_size, int64_t ffn_hidden_size, size_t schedule_units,
                    int64_t expert_tasks, int64_t nsplit_groups, int64_t nsplit_group_size, double e2e_ms) {
    if (!enabled()) {
      return;
    }

    std::vector<MoeGemmTraceRecord> records;
    std::vector<MoePhaseTraceRecord> phase_records;
    size_t gemm_count = host_buffer_.gemm_records.size();
    size_t phase_count = host_buffer_.phase_records.size();
    for (const MoeThreadTraceBuffer& buffer : thread_buffers_) {
      gemm_count += buffer.gemm_records.size();
      phase_count += buffer.phase_records.size();
    }
    records.reserve(gemm_count);
    phase_records.reserve(phase_count);
    append_buffer(host_buffer_, records, phase_records);
    for (const MoeThreadTraceBuffer& buffer : thread_buffers_) {
      append_buffer(buffer, records, phase_records);
    }
    for (MoeGemmTraceRecord& record : records) {
      record.ms = std::chrono::duration<double, std::milli>(record.end_time - record.begin_time).count();
    }
    for (MoePhaseTraceRecord& record : phase_records) {
      record.start_ms = std::chrono::duration<double, std::milli>(record.begin_time - origin_).count();
      record.end_ms = std::chrono::duration<double, std::milli>(record.end_time - origin_).count();
      record.ms = std::chrono::duration<double, std::milli>(record.end_time - record.begin_time).count();
    }
    std::sort(records.begin(), records.end(),
              [](const MoeGemmTraceRecord& lhs, const MoeGemmTraceRecord& rhs) {
                return std::tie(lhs.end_time, lhs.tid) < std::tie(rhs.end_time, rhs.tid);
              });
    std::sort(phase_records.begin(), phase_records.end(),
              [](const MoePhaseTraceRecord& lhs, const MoePhaseTraceRecord& rhs) {
                return std::tie(lhs.end_time, lhs.tid) < std::tie(rhs.end_time, rhs.tid);
              });
    size_t gemm_index = 0;
    size_t phase_index = 0;
    uint64_t seq = 0;
    while (gemm_index < records.size() || phase_index < phase_records.size()) {
      if (phase_index >= phase_records.size() ||
          (gemm_index < records.size() &&
           records[gemm_index].end_time <= phase_records[phase_index].end_time)) {
        records[gemm_index++].seq = seq++;
      } else {
        phase_records[phase_index++].seq = seq++;
      }
    }

    std::FILE* file = std::fopen(config_.path.c_str(), "a");
    if (file == nullptr) {
      return;
    }
    std::fprintf(file,
                 "MOE_CALL call_id=%llu strategy=%s e2e_ms=%.6f "
                 "threads=%lld tokens=%lld top_k=%lld experts=%lld routes=%lld "
                 "H=%lld F=%lld schedule_units=%zu expert_tasks=%lld "
                 "nsplit_groups=%lld nsplit_group_size=%lld gemm_count=%zu "
                 "phase_count=%zu\n",
                 static_cast<unsigned long long>(call_id_), strategy, e2e_ms, static_cast<long long>(num_threads),
                 static_cast<long long>(num_tokens), static_cast<long long>(top_k), static_cast<long long>(num_experts),
                 static_cast<long long>(num_routes), static_cast<long long>(hidden_size),
                 static_cast<long long>(ffn_hidden_size), schedule_units, static_cast<long long>(expert_tasks),
                 static_cast<long long>(nsplit_groups), static_cast<long long>(nsplit_group_size), records.size(),
                 phase_records.size());
    std::fprintf(file,
                 "GEMM_FIELDS call_id seq tid cpu affinity_first_cpu "
                 "affinity_cpu_count wave group local_tid expert route_begin "
                 "rows stage M K N ldc n_begin n_cols ms\n");
    for (const MoeGemmTraceRecord& record : records) {
      std::fprintf(file,
                   "GEMM call_id=%llu seq=%llu tid=%lld cpu=%lld "
                   "affinity_first_cpu=%lld affinity_cpu_count=%lld wave=%lld "
                   "group=%lld local_tid=%lld expert=%lld route_begin=%lld "
                   "rows=%lld stage=%s M=%lld K=%lld N=%lld ldc=%lld "
                   "n_begin=%lld n_cols=%lld ms=%.6f\n",
                   static_cast<unsigned long long>(call_id_), static_cast<unsigned long long>(record.seq),
                   static_cast<long long>(record.tid), static_cast<long long>(record.cpu),
                   static_cast<long long>(record.affinity_first_cpu), static_cast<long long>(record.affinity_cpu_count),
                   static_cast<long long>(record.wave), static_cast<long long>(record.group),
                   static_cast<long long>(record.local_tid), static_cast<long long>(record.expert),
                   static_cast<long long>(record.route_begin), static_cast<long long>(record.rows), record.stage,
                   static_cast<long long>(record.M), static_cast<long long>(record.K), static_cast<long long>(record.N),
                   static_cast<long long>(record.ldc), static_cast<long long>(record.n_begin),
                   static_cast<long long>(record.n_cols), record.ms);
    }
    std::fprintf(file,
                 "PHASE_FIELDS call_id seq tid cpu affinity_first_cpu "
                 "affinity_cpu_count wave group local_tid expert rows stage "
                 "start_ms end_ms ms\n");
    for (const MoePhaseTraceRecord& record : phase_records) {
      std::fprintf(file,
                   "PHASE call_id=%llu seq=%llu tid=%lld cpu=%lld "
                   "affinity_first_cpu=%lld affinity_cpu_count=%lld wave=%lld "
                   "group=%lld local_tid=%lld expert=%lld rows=%lld stage=%s "
                   "start_ms=%.6f end_ms=%.6f ms=%.6f\n",
                   static_cast<unsigned long long>(call_id_), static_cast<unsigned long long>(record.seq),
                   static_cast<long long>(record.tid), static_cast<long long>(record.cpu),
                   static_cast<long long>(record.affinity_first_cpu), static_cast<long long>(record.affinity_cpu_count),
                   static_cast<long long>(record.wave), static_cast<long long>(record.group),
                   static_cast<long long>(record.local_tid), static_cast<long long>(record.expert),
                   static_cast<long long>(record.rows), record.stage, record.start_ms, record.end_ms, record.ms);
    }
    std::fprintf(file, "MOE_CALL_END call_id=%llu\n", static_cast<unsigned long long>(call_id_));
    std::fclose(file);
  }

 private:
  static void append_buffer(const MoeThreadTraceBuffer& buffer, std::vector<MoeGemmTraceRecord>& records,
                            std::vector<MoePhaseTraceRecord>& phase_records) {
    records.insert(records.end(), buffer.gemm_records.begin(), buffer.gemm_records.end());
    phase_records.insert(phase_records.end(), buffer.phase_records.begin(), buffer.phase_records.end());
  }

  MoeThreadTraceBuffer& buffer_for_tid(int64_t tid) {
    if (tid < 0) {
      return host_buffer_;
    }
    TORCH_INTERNAL_ASSERT(static_cast<size_t>(tid) < thread_buffers_.size(), "trace tid is outside prepared buffers");
    return thread_buffers_[static_cast<size_t>(tid)];
  }

  static void initialize_thread_metadata(MoeThreadTraceBuffer& buffer) {
    if (buffer.metadata_initialized) {
      return;
    }
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (pthread_getaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) == 0) {
      for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &cpuset)) {
          continue;
        }
        if (buffer.affinity_first_cpu < 0) {
          buffer.affinity_first_cpu = static_cast<int64_t>(cpu);
        }
        ++buffer.affinity_cpu_count;
      }
      if (buffer.affinity_cpu_count == 1) {
        buffer.fixed_cpu = buffer.affinity_first_cpu;
      }
    }
    if (buffer.affinity_cpu_count == 0) {
      buffer.affinity_first_cpu = static_cast<int64_t>(sched_getcpu());
      buffer.affinity_cpu_count = 1;
    }
#else
    buffer.affinity_first_cpu = 0;
    buffer.affinity_cpu_count = 1;
    buffer.fixed_cpu = 0;
#endif
    buffer.metadata_initialized = true;
  }

  static int64_t record_cpu(MoeThreadTraceBuffer& buffer) {
    initialize_thread_metadata(buffer);
#ifdef __linux__
    return buffer.fixed_cpu >= 0 ? buffer.fixed_cpu : static_cast<int64_t>(sched_getcpu());
#else
    return buffer.fixed_cpu;
#endif
  }

  MoeTraceConfig config_;
  uint64_t call_id_ = 0;
  ::fused_cpp::profile::TimePoint origin_;
  MoeThreadTraceBuffer host_buffer_;
  std::vector<MoeThreadTraceBuffer> thread_buffers_;
};

#ifdef __aarch64__

void trace_dispatch_fp32_gemm(MoeTraceCollector& trace, const char* stage, int64_t tid, int64_t wave, int64_t group,
                              int64_t local_tid, int64_t expert, int64_t route_begin, int64_t rows, const uint16_t* A,
                              const uint16_t* B_reo, float* C, uint16_t* A_reorder, int M, int K, int N, int ldc,
                              const float* bias = nullptr) {
  // Default mode: one worker owns the whole expert GEMM. Route through the
  // middle layer as a degenerate group_size==1 team (whole slice, no barrier).
  TeamContext team;
  team.group_size = 1;
  team.local_tid = 0;
  team.barrier = nullptr;
  team.a_reorder = A_reorder;
  const GemmSplitPlan plan{MoeGemmSplit::kN};
  if (!trace.enabled()) {
    team_gemm(team, plan, A, B_reo, C, M, K, N, ldc, bias);
    return;
  }
  const auto begin = ::fused_cpp::profile::now();
  team_gemm(team, plan, A, B_reo, C, M, K, N, ldc, bias);
  const auto end = ::fused_cpp::profile::now();
  trace.record_gemm(tid, wave, group, local_tid, expert, route_begin, rows, stage, M, K, N, ldc, 0, N, begin, end);
}

void trace_dispatch_fp32_gemm_stage_split(MoeTraceCollector& trace, const char* stage_name, MoeGemmStage stage,
                                          int64_t tid, int64_t wave, int64_t group, int64_t local_tid, int64_t expert,
                                          int64_t route_begin, int64_t rows, const uint16_t* A, const uint16_t* B_reo,
                                          float* C, uint16_t* A_reorder, int M, int K, int N, int ldc,
                                          int64_t group_size, const float* bias = nullptr) {
  // Middle layer: the pluggable selector chooses M vs N; the caller owns the
  // inter-stage barriers, so team_gemm runs without its own (barrier=null).
  const GemmSplitPlan plan = plan_team_gemm_split(stage, M, K, N, group_size);
  const bool split_m = plan.split == MoeGemmSplit::kM;
  const SplitRange range = team_gemm_split_range(plan, M, N, group_size, local_tid);
  if (range.size <= 0) {
    return;
  }
  TeamContext team;
  team.group_size = group_size;
  team.local_tid = local_tid;
  team.barrier = nullptr;
  team.a_reorder = A_reorder;
  if (!trace.enabled()) {
    team_gemm(team, plan, A, B_reo, C, M, K, N, ldc, bias);
    return;
  }
  const auto begin = ::fused_cpp::profile::now();
  team_gemm(team, plan, A, B_reo, C, M, K, N, ldc, bias);
  const auto end = ::fused_cpp::profile::now();
  int64_t trace_route_begin = route_begin;
  int64_t trace_rows = rows;
  int64_t trace_n_begin = 0;
  int64_t trace_n_cols = N;
  if (split_m) {
    trace_route_begin += range.begin;
    trace_rows = range.size;
  } else {
    trace_n_begin = range.begin;
    trace_n_cols = range.size;
  }
  trace.record_gemm(tid, wave, group, local_tid, expert, trace_route_begin, trace_rows, stage_name, M, K, N, ldc,
                    trace_n_begin, trace_n_cols, begin, end);
}

#endif

int bind_current_thread_to_core(int64_t core) {
#ifdef __linux__
  if (core < 0 || core >= CPU_SETSIZE) {
    return EINVAL;
  }
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(static_cast<int>(core), &cpuset);
  return pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#else
  (void)core;
  return 0;
#endif
}

void report_affinity_bind_error(int64_t tid, int64_t core, int error) {
  if (error == 0) {
    return;
  }
  std::fprintf(stderr,
               "[fused_moe_bf16_tiled][affinity] failed to bind tid=%lld "
               "to core=%lld error=%d (%s)\n",
               static_cast<long long>(tid), static_cast<long long>(core), error, std::strerror(error));
}

class ThreadAffinityGuard {
 public:
  ThreadAffinityGuard() {
#ifdef __linux__
    valid_ = pthread_getaffinity_np(pthread_self(), sizeof(original_), &original_) == 0;
#endif
  }

  ~ThreadAffinityGuard() {
#ifdef __linux__
    if (valid_) {
      pthread_setaffinity_np(pthread_self(), sizeof(original_), &original_);
    }
#endif
  }

 private:
#ifdef __linux__
  cpu_set_t original_;
  bool valid_ = false;
#endif
};

int debug_schedule_level() {
  const char* value = std::getenv("FUSED_CPP_MOE_SCHEDULE_DEBUG");
  if (value == nullptr || value[0] == '\0' || value[0] == '0') {
    return 0;
  }
  const int parsed = std::atoi(value);
  return parsed <= 0 ? 1 : parsed;
}

ThreadScheduleDebug summarize_thread_schedule(const std::vector<ExpertTask>& tasks,
                                              const std::vector<TaskRange>& ranges) {
  ThreadScheduleDebug debug;
  debug.ranges = static_cast<int64_t>(ranges.size());
  int64_t last_expert = -1;
  for (const TaskRange& range : ranges) {
    for (size_t task_idx = range.begin; task_idx < range.end; ++task_idx) {
      const ExpertTask& task = tasks[task_idx];
      debug.rows += task.rows;
      ++debug.tasks;
      if (debug.experts.empty() || task.expert != last_expert) {
        debug.experts.push_back(task.expert);
        debug.expert_rows.push_back(0);
        last_expert = task.expert;
      }
      debug.expert_rows.back() += task.rows;
    }
  }
  return debug;
}

void print_schedule_debug_line(const char* label, int64_t tid, const ThreadScheduleDebug& debug) {
  std::fprintf(stderr,
               "[fused_moe_bf16_tiled][schedule] %s tid=%lld ms=%.3f rows=%lld "
               "tasks=%lld ranges=%lld experts=%zu experts=[",
               label, static_cast<long long>(tid), debug.ms, static_cast<long long>(debug.rows),
               static_cast<long long>(debug.tasks), static_cast<long long>(debug.ranges), debug.experts.size());
  for (size_t i = 0; i < debug.experts.size(); ++i) {
    if (i != 0) {
      std::fprintf(stderr, ",");
    }
    std::fprintf(stderr, "%lld:%lld", static_cast<long long>(debug.experts[i]),
                 static_cast<long long>(debug.expert_rows[i]));
  }
  std::fprintf(stderr, "]\n");
}

std::vector<std::vector<TaskRange>> split_tasks_by_expert_affinity(const std::vector<ExpertTask>& tasks,
                                                                   int64_t num_threads,
                                                                   const ExpertScheduleWorkloadConfig& workload_config,
                                                                   float* estimated_schedule_cost = nullptr) {
  std::vector<std::vector<TaskRange>> ranges(static_cast<size_t>(num_threads));
  if (estimated_schedule_cost != nullptr) {
    *estimated_schedule_cost = 0.0f;
  }
  if (tasks.empty()) {
    return ranges;
  }

  std::vector<ExpertTaskGroup> groups;
  groups.reserve(tasks.size());
  size_t task_cursor = 0;
  while (task_cursor < tasks.size()) {
    ExpertTaskGroup group;
    group.begin = task_cursor;
    const int64_t expert = tasks[task_cursor].expert;
    while (task_cursor < tasks.size() && tasks[task_cursor].expert == expert) {
      group.rows += tasks[task_cursor].rows;
      ++task_cursor;
    }
    group.end = task_cursor;
    groups.push_back(group);
  }

  std::vector<size_t> group_order(groups.size());
  std::iota(group_order.begin(), group_order.end(), size_t{0});
  const ExpertScheduleSystemState empty_state;
  std::sort(group_order.begin(), group_order.end(), [&](size_t lhs, size_t rhs) {
    const ExpertScheduleWorkloadShape lhs_workload = make_expert_schedule_workload(workload_config, groups[lhs].rows);
    const ExpertScheduleWorkloadShape rhs_workload = make_expert_schedule_workload(workload_config, groups[rhs].rows);
    const float lhs_cost = estimate_expert_schedule_cost(lhs_workload, 1, empty_state, num_threads);
    const float rhs_cost = estimate_expert_schedule_cost(rhs_workload, 1, empty_state, num_threads);
    if (lhs_cost != rhs_cost) {
      return lhs_cost > rhs_cost;
    }
    if (groups[lhs].rows != groups[rhs].rows) {
      return groups[lhs].rows > groups[rhs].rows;
    }
    return lhs < rhs;
  });

  const int64_t beam_width = expert_schedule_beam_width();
  const std::vector<ExpertScheduleBeamPolicy> policies{ExpertScheduleBeamPolicy::kLowLatency,
                                                       ExpertScheduleBeamPolicy::kBalanced,
                                                       ExpertScheduleBeamPolicy::kHighThroughput};
  std::vector<ExpertScheduleBeamState> beams;
  beams.reserve(policies.size());
  for (const ExpertScheduleBeamPolicy policy : policies) {
    ExpertScheduleBeamState state;
    state.policy = policy;
    state.ranges.resize(static_cast<size_t>(num_threads));
    state.thread_load.assign(static_cast<size_t>(num_threads), 0.0f);
    state.score = score_expert_schedule_beam(state);
    beams.push_back(std::move(state));
  }

  for (const size_t group_idx : group_order) {
    const ExpertTaskGroup& group = groups[group_idx];
    const ExpertScheduleWorkloadShape workload = make_expert_schedule_workload(workload_config, group.rows);
    std::vector<ExpertScheduleBeamState> expanded;
    expanded.reserve(beams.size() * static_cast<size_t>(num_threads));
    for (const ExpertScheduleBeamState& beam : beams) {
      for (int64_t tid_i = 0; tid_i < num_threads; ++tid_i) {
        const size_t tid = static_cast<size_t>(tid_i);
        ExpertScheduleBeamState candidate = beam;
        const float start = candidate.thread_load[tid];
        const ExpertScheduleSystemState state = candidate.system_state_tracker.state_at(start);
        const ExpertScheduleCostEstimate cost = estimate_expert_schedule_cost_estimate(workload, 1, state, num_threads);
        const float finish = start + cost.total;
        candidate.ranges[tid].push_back(TaskRange{group.begin, group.end});
        candidate.thread_load[tid] = finish;
        candidate.predicted_makespan = std::max(candidate.predicted_makespan, finish);
        candidate.predicted_core_time += cost.total;
        candidate.system_state_tracker.add_interval(start, finish, estimate_expert_resource_demand(workload, 1));
        candidate.score = score_expert_schedule_beam(candidate);
        expanded.push_back(std::move(candidate));
      }
    }

    std::vector<ExpertScheduleBeamState> next_beams;
    next_beams.reserve(static_cast<size_t>(beam_width) * policies.size());
    for (const ExpertScheduleBeamPolicy policy : policies) {
      std::vector<ExpertScheduleBeamState> policy_candidates;
      for (ExpertScheduleBeamState& candidate : expanded) {
        if (candidate.policy == policy) {
          policy_candidates.push_back(std::move(candidate));
        }
      }
      std::sort(policy_candidates.begin(), policy_candidates.end(), better_expert_schedule_beam);
      const size_t keep = std::min<size_t>(static_cast<size_t>(beam_width), policy_candidates.size());
      for (size_t idx = 0; idx < keep; ++idx) {
        next_beams.push_back(std::move(policy_candidates[idx]));
      }
    }
    beams = std::move(next_beams);
  }

  TORCH_CHECK(!beams.empty(), "unable to build beam MoE schedule");
  const auto best_it = std::min_element(beams.begin(), beams.end(),
                                        [](const ExpertScheduleBeamState& lhs, const ExpertScheduleBeamState& rhs) {
                                          if (lhs.predicted_makespan != rhs.predicted_makespan) {
                                            return lhs.predicted_makespan < rhs.predicted_makespan;
                                          }
                                          return lhs.score < rhs.score;
                                        });
  ranges = best_it->ranges;
  if (estimated_schedule_cost != nullptr) {
    *estimated_schedule_cost = best_it->predicted_makespan;
  }
  return ranges;
}

bool disable_resident_threads() { return env_flag_enabled("FUSED_CPP_MOE_DISABLE_RESIDENT_THREADS"); }

struct ThreadPinningConfig {
  bool enabled = false;
  std::vector<int64_t> cpus;

  int64_t core_for_tid(int64_t tid) const {
    if (!enabled || cpus.empty()) {
      return -1;
    }
    return cpus[static_cast<size_t>(tid) % cpus.size()];
  }
};

thread_local const ThreadPinningConfig* g_moe_thread_pinning_override = nullptr;

class ThreadPinningScope {
 public:
  explicit ThreadPinningScope(const ThreadPinningConfig* config) : previous_(g_moe_thread_pinning_override) {
    g_moe_thread_pinning_override = config;
  }

  ThreadPinningScope(const ThreadPinningScope&) = delete;
  ThreadPinningScope& operator=(const ThreadPinningScope&) = delete;

  ~ThreadPinningScope() { g_moe_thread_pinning_override = previous_; }

 private:
  const ThreadPinningConfig* previous_ = nullptr;
};

ThreadPinningConfig moe_thread_pinning_config(int64_t num_threads) {
  if (g_moe_thread_pinning_override != nullptr) {
    ThreadPinningConfig config = *g_moe_thread_pinning_override;
    TORCH_CHECK(!config.enabled || !config.cpus.empty(), "explicit MoE thread pinning requires at least one CPU id");
    return config;
  }

  ThreadPinningConfig config;
  config.enabled = env_flag_enabled("FUSED_CPP_MOE_PIN_THREADS");
  if (!config.enabled) {
    return config;
  }
  config.cpus = env_int_list_or_default("FUSED_CPP_MOE_PIN_THREAD_CPUS", current_affinity_cpus(num_threads));
  TORCH_CHECK(!config.cpus.empty(), "FUSED_CPP_MOE_PIN_THREAD_CPUS must not be empty");
  return config;
}

template <typename Fn>
void run_fixed_threads_spawn(int64_t num_threads, const Fn& fn) {
  if (num_threads <= 1) {
    fn(0);
    return;
  }
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(num_threads - 1));
  for (int64_t tid = 1; tid < num_threads; ++tid) {
    workers.emplace_back([&, tid]() { fn(tid); });
  }
  fn(0);
  for (auto& worker : workers) {
    worker.join();
  }
}

class ResidentThreadPool {
 public:
  ResidentThreadPool() = default;

  ResidentThreadPool(const ResidentThreadPool&) = delete;
  ResidentThreadPool& operator=(const ResidentThreadPool&) = delete;

  ~ResidentThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
      ++generation_;
    }
    start_cv_.notify_all();
    for (std::thread& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  void prepare(int64_t num_threads, const ThreadPinningConfig& pinning) {
    if (num_threads <= 1) {
      return;
    }
    {
      std::unique_lock<std::mutex> lock(mutex_);
      TORCH_CHECK(!job_active_,
                  "resident MoE thread pool does not support prepare "
                  "while a job is active");
      configure_worker_cores_locked(num_threads, pinning);
      ensure_workers_locked(num_threads - 1);
      ++generation_;
    }
    start_cv_.notify_all();
  }

  template <typename Fn>
  void run(int64_t num_threads, const ThreadPinningConfig& pinning, const Fn& fn) {
    if (num_threads <= 1) {
      ThreadAffinityGuard affinity_guard;
      if (pinning.enabled) {
        const int64_t core = pinning.core_for_tid(0);
        report_affinity_bind_error(0, core, bind_current_thread_to_core(core));
      }
      fn(0);
      return;
    }

    std::function<void(int64_t)> job = [&](int64_t tid) { fn(tid); };
    std::exception_ptr main_exception = nullptr;
    std::exception_ptr worker_exception = nullptr;

    {
      std::unique_lock<std::mutex> lock(mutex_);
      TORCH_CHECK(!job_active_, "resident MoE thread pool does not support nested jobs");
      configure_worker_cores_locked(num_threads, pinning);
      ensure_workers_locked(num_threads - 1);
      current_job_ = &job;
      requested_threads_ = num_threads;
      remaining_workers_ = num_threads - 1;
      worker_exception_ = nullptr;
      job_active_ = true;
      ++generation_;
    }
    start_cv_.notify_all();

    ThreadAffinityGuard affinity_guard;
    if (pinning.enabled) {
      const int64_t core = pinning.core_for_tid(0);
      report_affinity_bind_error(0, core, bind_current_thread_to_core(core));
    }

    try {
      fn(0);
    } catch (...) {
      main_exception = std::current_exception();
    }

    {
      std::unique_lock<std::mutex> lock(mutex_);
      done_cv_.wait(lock, [&]() { return remaining_workers_ == 0; });
      worker_exception = worker_exception_;
      current_job_ = nullptr;
      requested_threads_ = 0;
      job_active_ = false;
    }

    if (main_exception != nullptr) {
      std::rethrow_exception(main_exception);
    }
    if (worker_exception != nullptr) {
      std::rethrow_exception(worker_exception);
    }
  }

 private:
  void configure_worker_cores_locked(int64_t num_threads, const ThreadPinningConfig& pinning) {
    const int64_t worker_count = std::max<int64_t>(num_threads - 1, 0);
    worker_cores_.resize(static_cast<size_t>(worker_count), -1);
    for (int64_t tid = 1; tid < num_threads; ++tid) {
      worker_cores_[static_cast<size_t>(tid - 1)] = pinning.enabled ? pinning.core_for_tid(tid) : -1;
    }
  }

  void ensure_workers_locked(int64_t worker_count) {
    while (static_cast<int64_t>(workers_.size()) < worker_count) {
      const int64_t tid = static_cast<int64_t>(workers_.size()) + 1;
      workers_.emplace_back([this, tid]() { worker_loop(tid); });
    }
  }

  void worker_loop(int64_t tid) {
    int64_t seen_generation = 0;
    int64_t bound_core = std::numeric_limits<int64_t>::min();
    while (true) {
      std::function<void(int64_t)>* job = nullptr;
      int64_t desired_core = -1;
      bool should_run_job = false;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        start_cv_.wait(lock, [&]() { return stopping_ || generation_ != seen_generation; });
        if (stopping_) {
          return;
        }
        seen_generation = generation_;
        if (tid > 0 && static_cast<size_t>(tid - 1) < worker_cores_.size()) {
          desired_core = worker_cores_[static_cast<size_t>(tid - 1)];
        }
        should_run_job = current_job_ != nullptr && tid < requested_threads_;
        if (should_run_job) {
          job = current_job_;
        }
      }

      if (desired_core >= 0 && desired_core != bound_core) {
        const int error = bind_current_thread_to_core(desired_core);
        report_affinity_bind_error(tid, desired_core, error);
        if (error == 0) {
          bound_core = desired_core;
        }
      }
      if (!should_run_job) {
        continue;
      }

      try {
        (*job)(tid);
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (worker_exception_ == nullptr) {
          worker_exception_ = std::current_exception();
        }
      }

      {
        std::lock_guard<std::mutex> lock(mutex_);
        --remaining_workers_;
        if (remaining_workers_ == 0) {
          done_cv_.notify_one();
        }
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable start_cv_;
  std::condition_variable done_cv_;
  std::vector<std::thread> workers_;
  std::vector<int64_t> worker_cores_;
  std::function<void(int64_t)>* current_job_ = nullptr;
  std::exception_ptr worker_exception_ = nullptr;
  int64_t requested_threads_ = 0;
  int64_t remaining_workers_ = 0;
  int64_t generation_ = 0;
  bool job_active_ = false;
  bool stopping_ = false;
};

ResidentThreadPool& moe_resident_thread_pool() {
  static ResidentThreadPool pool;
  return pool;
}

template <typename Fn>
void run_fixed_threads(int64_t num_threads, const Fn& fn) {
  const ThreadPinningConfig pinning = moe_thread_pinning_config(num_threads);
  if (disable_resident_threads()) {
    run_fixed_threads_spawn(num_threads, fn);
    return;
  }
  moe_resident_thread_pool().prepare(num_threads, pinning);
  moe_resident_thread_pool().run(num_threads, pinning, fn);
}

void prepare_moe_threads_for_operator(int64_t num_threads) {
  if (disable_resident_threads()) {
    return;
  }
  const ThreadPinningConfig pinning = moe_thread_pinning_config(num_threads);
  moe_resident_thread_pool().prepare(num_threads, pinning);
}

struct VllmStagedNTask {
  int64_t expert = 0;
  int64_t n_begin = 0;
  int64_t n_cols = 0;
};

struct VllmStagedThreadScratch {
  std::vector<uint16_t, backend_allocator<uint16_t>> packed_a;
};

int64_t vllm_staged_available_l2_bytes() {
  static const int64_t available_bytes = []() {
    int64_t l2_bytes = 0;
#if defined(__linux__)
    const long detected = sysconf(_SC_LEVEL2_CACHE_SIZE);
    if (detected > 0) {
      l2_bytes = static_cast<int64_t>(detected);
    }
#endif
    // Match vLLM's get_available_l2_size policy: reserve half of private L2
    // for A, output, code, and unrelated live data. The fallback describes
    // the 2 MiB private-L2 target used by the SVE experiment machines.
    if (l2_bytes <= 0) {
      l2_bytes = 2LL * 1024 * 1024;
    }
    return std::max<int64_t>(l2_bytes / 2, 64LL * 1024);
  }();
  return available_bytes;
}

int64_t vllm_staged_task_n(MoeGemmStage stage, int64_t K, int64_t N, int64_t n_tile, int64_t num_threads,
                           int64_t top_k) {
  TORCH_CHECK(K > 0 && N > 0 && n_tile > 0 && N % n_tile == 0,
              "vLLM-staged task dimensions must be positive and N-tile aligned");
  const int64_t m_tile = 12;
  const int64_t available_l2 = vllm_staged_available_l2_bytes();
  const int64_t lanes_per_expert = std::max<int64_t>(1, num_threads / top_k);
  const int64_t thread_limit = std::max<int64_t>(n_tile, N / lanes_per_expert);

  int64_t minimum_task_n = n_tile;
  int64_t input_bytes = m_tile * K * static_cast<int64_t>(sizeof(uint16_t));
  int64_t bytes_per_column = K * static_cast<int64_t>(sizeof(uint16_t));
  if (stage == MoeGemmStage::kW13) {
    minimum_task_n = 2 * n_tile;
    bytes_per_column += m_tile * static_cast<int64_t>(sizeof(float));
  }
  input_bytes = ceil_to_multiple(input_bytes, int64_t{64});
  const int64_t cache_budget = std::max<int64_t>(0, available_l2 - input_bytes);
  const int64_t cache_limit = cache_budget / bytes_per_column;
  const int64_t candidate = std::min(cache_limit, thread_limit);
  const int64_t aligned = candidate / minimum_task_n * minimum_task_n;
  return std::min<int64_t>(N, std::max<int64_t>(minimum_task_n, aligned));
}

std::vector<VllmStagedNTask> build_vllm_staged_tasks(int64_t num_experts, int64_t N, int64_t task_n,
                                                     int64_t n_tile) {
  TORCH_CHECK(num_experts > 0 && N > 0 && task_n > 0 && n_tile > 0, "invalid vLLM-staged task geometry");
  TORCH_CHECK(N % n_tile == 0 && task_n % n_tile == 0,
              "vLLM-staged N and task width must be N-tile aligned: N=", N, " task_n=", task_n,
              " tile=", n_tile);
  const int64_t tasks_per_expert = ceil_div_int64(N, task_n);
  TORCH_CHECK(num_experts <= std::numeric_limits<int64_t>::max() / tasks_per_expert,
              "vLLM-staged task count exceeds int64");
  const int64_t total_tasks = num_experts * tasks_per_expert;
  std::vector<VllmStagedNTask> tasks;
  TORCH_CHECK(static_cast<uint64_t>(total_tasks) <= std::numeric_limits<size_t>::max(),
              "vLLM-staged task count exceeds size_t");
  tasks.reserve(static_cast<size_t>(total_tasks));
  for (int64_t expert = 0; expert < num_experts; ++expert) {
    for (int64_t n_begin = 0; n_begin < N; n_begin += task_n) {
      tasks.push_back(VllmStagedNTask{expert, n_begin, std::min<int64_t>(task_n, N - n_begin)});
    }
  }
  return tasks;
}

struct ThreadScratch {
  std::vector<uint16_t, backend_allocator<uint16_t>> input;
  std::vector<uint16_t, backend_allocator<uint16_t>> intermediate;
  std::vector<uint16_t, backend_allocator<uint16_t>> a_reorder;
  std::vector<uint16_t, backend_allocator<uint16_t>> packed_a;
  std::vector<float, backend_allocator<float>> gate_up;
  std::vector<float, backend_allocator<float>> down;
  std::vector<uint16_t, backend_allocator<uint16_t>> down_bf16;
};

struct ScheduledWaveRuntime {
  int64_t begin = 0;
  int64_t end = 0;
  int64_t total_threads = 0;
};

struct AsyncTaskRuntime {
  int64_t expert = 0;
  int64_t route_begin = 0;
  int64_t rows = 0;
  int64_t core_begin = 0;
  int64_t threads = 0;
  int64_t scratch_index = -1;
};

struct AsyncStrictTailStealTeam {
  int64_t core_begin = 0;
  int64_t threads = 0;
  int64_t scratch_index = -1;
  std::vector<int64_t> task_ids;
};

constexpr int64_t kAsyncPlanV2 = 2;
constexpr int64_t kAsyncExecutionStrict = 0;
constexpr int64_t kAsyncExecutionTailPool = 1;
constexpr int64_t kAsyncExecutionElastic = 2;
constexpr int64_t kAsyncPlacementFixed = 0;
constexpr int64_t kAsyncPlacementTailPool = 1;
constexpr int64_t kAsyncStageExpert = 0;
constexpr int64_t kAsyncResizeNone = 0;
constexpr int64_t kAsyncResizeBeforeW2 = 1;
constexpr int64_t kAsyncFullExpertRange = 0;
constexpr int64_t kAsyncEarlyMergeAuto = -1;
constexpr int64_t kAsyncEarlyMergeOff = 0;
constexpr int64_t kAsyncEarlyMergeOn = 1;
constexpr int64_t kAsyncElasticStatsCount = 10;
enum AsyncElasticStat : int64_t {
  kElasticEligibleTasks = 0,
  kElasticPreferredAssignments = 1,
  kElasticFallbackAssignments = 2,
  kElasticNaturalOpportunities = 3,
  kElasticWaitedPreferredAssignments = 4,
  kElasticTimeoutFallbacks = 5,
  kElasticCohortJobs = 6,
  kElasticBorrowedThreads = 7,
  kElasticTotalWaitNs = 8,
  kElasticMaxWaitNs = 9,
};

int64_t cpu_numa_node(int64_t cpu) {
#ifdef __linux__
  static std::mutex cache_mutex;
  static std::array<int64_t, CPU_SETSIZE> cache = [] {
    std::array<int64_t, CPU_SETSIZE> values{};
    values.fill(-2);
    return values;
  }();
  if (cpu >= 0 && cpu < CPU_SETSIZE) {
    const std::lock_guard<std::mutex> lock(cache_mutex);
    const int64_t cached = cache[static_cast<size_t>(cpu)];
    if (cached != -2) {
      return cached;
    }
  }
  char path[128];
  int64_t result = -1;
  for (int64_t node = 0; node < 1024; ++node) {
    std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%lld/node%lld",
                  static_cast<long long>(cpu), static_cast<long long>(node));
    if (::access(path, F_OK) == 0) {
      result = node;
      break;
    }
  }
  if (cpu >= 0 && cpu < CPU_SETSIZE) {
    const std::lock_guard<std::mutex> lock(cache_mutex);
    cache[static_cast<size_t>(cpu)] = result;
  }
  return result;
#else
  (void)cpu;
  return -1;
#endif
}

// Plan V2 metadata is call-owned and read only while the Python extension
// call is active. Elastic mode may expand or migrate a fixed team only at the
// W13-to-W2 boundary; strict and tail-pool retain fixed in-task widths.
struct AsyncPlanV2NativeArgs {
  int64_t plan_version = 0;
  int64_t execution_mode = kAsyncExecutionStrict;
  at::Tensor task_preferred_threads;
  at::Tensor task_min_threads;
  at::Tensor task_max_threads;
  at::Tensor task_allowed_thread_offsets;
  at::Tensor task_allowed_threads;
  at::Tensor task_placement_modes;
  at::Tensor task_numa_nodes;
  at::Tensor task_stage_ids;
  at::Tensor task_resize_points;
  at::Tensor task_range_granularities;
  c10::optional<at::Tensor> task_release_ns;
  c10::optional<at::Tensor> task_resize_timeout_ns;
  c10::optional<at::Tensor> task_preferred_core_begins;
  c10::optional<at::Tensor> elastic_stats_out;
  int64_t early_merge = kAsyncEarlyMergeAuto;
};

struct ScheduledScratchUnitConfig {
  int64_t thread_begin = 0;
  int64_t threads = 0;
  int64_t max_rows = 0;
  int64_t a_reorder_stride = 0;
  bool fused_packa = false;
  bool w2_bf16_route = false;
  bool w2_direct_route = false;
  bool external_intermediate = false;
  bool barrier_only = false;
};

struct ScheduledTeamScratch {
  explicit ScheduledTeamScratch(int64_t group_size) : threads(group_size), barrier(group_size) {}

  int64_t threads = 0;
  int64_t max_rows = 0;
  int64_t a_reorder_stride = 0;
  std::vector<uint16_t, backend_allocator<uint16_t>> input;
  std::vector<uint16_t, backend_allocator<uint16_t>> intermediate;
  std::vector<uint16_t, backend_allocator<uint16_t>> a_reorder;
  std::vector<uint16_t, backend_allocator<uint16_t>> packed_a;
  std::vector<float, backend_allocator<float>> gate_up;
  std::vector<float, backend_allocator<float>> down;
  std::vector<uint16_t, backend_allocator<uint16_t>> down_bf16;
  ThreadBarrier barrier;
};

void ensure_scheduled_scratch_capacity(ScheduledTeamScratch& scratch, const ScheduledScratchUnitConfig& config,
                                       const PackedExperts& w13, const PackedExperts& w2) {
  TORCH_CHECK(scratch.threads == config.threads, "scheduled scratch thread count mismatch: scratch=", scratch.threads,
              " config=", config.threads);
  scratch.max_rows = std::max(scratch.max_rows, config.max_rows);
  scratch.a_reorder_stride = std::max(scratch.a_reorder_stride, config.a_reorder_stride);
  if (config.barrier_only) {
    return;
  }
  const int64_t rows = scratch.max_rows;
  const int64_t rows_padded =
      config.fused_packa ? sve_hybrid_packed_rows(rows) : ceil_to_multiple(rows, int64_t{kKernelTile});
  scratch.input.resize(static_cast<size_t>(config.fused_packa ? 0 : rows * w13.K_pad));
  if (!config.external_intermediate) {
    scratch.intermediate.resize(static_cast<size_t>((config.fused_packa ? rows_padded : rows) * w2.K_pad));
  }
  scratch.a_reorder.resize(static_cast<size_t>(scratch.threads * scratch.a_reorder_stride));
  scratch.packed_a.resize(static_cast<size_t>(config.fused_packa ? rows_padded * w13.K_pad : 0));
  scratch.gate_up.resize(static_cast<size_t>(config.fused_packa ? 0 : rows * w13.N_pad));
  const size_t down_elements = static_cast<size_t>((config.fused_packa ? rows_padded : rows) * w2.N_pad);
  if (config.w2_direct_route) {
    return;
  }
  if (config.w2_bf16_route) {
    if (scratch.down_bf16.size() < down_elements) {
      scratch.down_bf16.resize(down_elements);
    }
  } else if (scratch.down.size() < down_elements) {
    scratch.down.resize(down_elements);
  }
}

class ScheduledScratchLease {
 public:
  ScheduledScratchLease(std::unique_lock<std::mutex> lock, std::vector<ScheduledTeamScratch*> scratches)
      : lock_(std::move(lock)), scratches_(std::move(scratches)) {}

  ScheduledScratchLease(const ScheduledScratchLease&) = delete;
  ScheduledScratchLease& operator=(const ScheduledScratchLease&) = delete;
  ScheduledScratchLease(ScheduledScratchLease&&) = default;
  ScheduledScratchLease& operator=(ScheduledScratchLease&&) = default;

  const std::vector<ScheduledTeamScratch*>& scratches() const { return scratches_; }

 private:
  std::unique_lock<std::mutex> lock_;
  std::vector<ScheduledTeamScratch*> scratches_;
};

class ResidentScheduledScratchPool {
 public:
  ResidentScheduledScratchPool() = default;

  ResidentScheduledScratchPool(const ResidentScheduledScratchPool&) = delete;
  ResidentScheduledScratchPool& operator=(const ResidentScheduledScratchPool&) = delete;

  ScheduledScratchLease lease(const std::vector<ScheduledScratchUnitConfig>& configs, const PackedExperts& w13,
                              const PackedExperts& w2) {
    std::unique_lock<std::mutex> lock(mutex_);
    std::vector<ScheduledTeamScratch*> scratches;
    scratches.reserve(configs.size());

    for (const ScheduledScratchUnitConfig& config : configs) {
      ScheduledTeamScratch* scratch = find_scratch_locked(config);
      if (scratch == nullptr) {
        units_.push_back(
            ScratchUnit{config.thread_begin, config.threads, std::make_unique<ScheduledTeamScratch>(config.threads)});
        scratch = units_.back().scratch.get();
      }
      ensure_scheduled_scratch_capacity(*scratch, config, w13, w2);
      scratches.push_back(scratch);
    }

    return ScheduledScratchLease(std::move(lock), std::move(scratches));
  }

 private:
  struct ScratchUnit {
    int64_t thread_begin = 0;
    int64_t threads = 0;
    std::unique_ptr<ScheduledTeamScratch> scratch;
  };

  ScheduledTeamScratch* find_scratch_locked(const ScheduledScratchUnitConfig& config) {
    for (ScratchUnit& unit : units_) {
      if (unit.thread_begin == config.thread_begin && unit.threads == config.threads) {
        return unit.scratch.get();
      }
    }
    return nullptr;
  }

  std::mutex mutex_;
  std::vector<ScratchUnit> units_;
};

ResidentScheduledScratchPool& resident_scheduled_scratch_pool() {
  static ResidentScheduledScratchPool pool;
  return pool;
}

int64_t stage_a_reorder_stride(MoeGemmStage stage, int64_t rows, int64_t K_pad, int64_t N_pad, int64_t group_size) {
  // Worst case: an N-split has every team thread re-reorder the full `rows`
  // (an M-split needs at most `rows` too). Size for that regardless of which
  // split the pluggable selector picks, so the per-thread scratch is always
  // large enough and stays correct if the selector changes.
  (void)stage;
  (void)N_pad;
  (void)group_size;
  return rows * K_pad * 2;
}

int64_t scheduled_a_reorder_stride(int64_t rows, int64_t group_size, const PackedExperts& w13,
                                   const PackedExperts& w2) {
  return std::max(stage_a_reorder_stride(MoeGemmStage::kW13, rows, w13.K_pad, w13.N_pad, group_size),
                  stage_a_reorder_stride(MoeGemmStage::kW2, rows, w2.K_pad, w2.N_pad, group_size));
}

std::vector<int64_t> tensor_to_i64_vector(at::Tensor tensor, const char* name) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(is_integer_dtype(tensor.scalar_type()), name, " must use an integer dtype");
  TORCH_CHECK(tensor.dim() == 1, name, " must be 1-D");
  tensor = tensor.to(at::kLong).contiguous();
  const int64_t* ptr = tensor.data_ptr<int64_t>();
  return std::vector<int64_t>(ptr, ptr + tensor.numel());
}

void pack_transposed_expert_weight(const uint16_t* weight, int64_t expert_offset, int64_t out_features,
                                   int64_t in_features, int64_t K_pad, int64_t N_pad, uint16_t* packed_dst,
                                   const ::fused_cpp::moe::MoeBackend* backend = nullptr) {
  std::vector<uint16_t> transposed(static_cast<size_t>(K_pad * N_pad), static_cast<uint16_t>(0));
  for (int64_t n = 0; n < out_features; ++n) {
    const uint16_t* src_row = weight + expert_offset + n * in_features;
    for (int64_t k = 0; k < in_features; ++k) {
      transposed[static_cast<size_t>(k * N_pad + n)] = src_row[k];
    }
  }
  if (backend == nullptr) {
    bf16_pack_b(transposed.data(), packed_dst, static_cast<int>(K_pad), static_cast<int>(N_pad));
  } else {
    backend->pack_b(transposed.data(), packed_dst, static_cast<int>(K_pad), static_cast<int>(N_pad));
  }
}

// Interleaved w13 pack for the fused SiLU epilogue. w13 weight is
// [2F, H] = [gate rows 0..F | up rows F..2F], in_features = H (= K13).
// The fused bf16 kernel produces one intermediate feature per (gate,up)
// column pair, and its 8x8 tile de-interleaves as cols[0:4]=gate, cols[4:8]=up
// (see STORE_C_ROWMAJOR). So we lay the packed columns out per 8-col N-block b
// as [g(4b) g(4b+1) g(4b+2) g(4b+3) | u(4b) u(4b+1) u(4b+2) u(4b+3)]:
//   gate feature f -> packed column 8*(f/4) + (f%4)
//   up   feature f -> packed column 8*(f/4) + 4 + (f%4)
// N_pad must be 2*ceil(F,4) (a multiple of 8). Padded gate/up features in
// [F, ceil(F,4)) stay zero, so they yield silu(0)*0 = 0 downstream.
void pack_transposed_expert_weight_interleaved(const uint16_t* weight, int64_t expert_offset, int64_t F, int64_t H,
                                               int64_t K_pad, int64_t N_pad, uint16_t* packed_dst,
                                               const ::fused_cpp::moe::MoeBackend* backend = nullptr) {
  std::vector<uint16_t> transposed(static_cast<size_t>(K_pad * N_pad), static_cast<uint16_t>(0));
  for (int64_t f = 0; f < F; ++f) {
    const int64_t blk = f / 4;
    const int64_t off = f % 4;
    const int64_t gate_col = blk * 8 + off;
    const int64_t up_col = blk * 8 + 4 + off;
    const uint16_t* gate_row = weight + expert_offset + f * H;
    const uint16_t* up_row = weight + expert_offset + (F + f) * H;
    for (int64_t k = 0; k < H; ++k) {
      transposed[static_cast<size_t>(k * N_pad + gate_col)] = gate_row[k];
      transposed[static_cast<size_t>(k * N_pad + up_col)] = up_row[k];
    }
  }
  if (backend == nullptr) {
    bf16_pack_b(transposed.data(), packed_dst, static_cast<int>(K_pad), static_cast<int>(N_pad));
  } else {
    backend->pack_b(transposed.data(), packed_dst, static_cast<int>(K_pad), static_cast<int>(N_pad));
  }
}

}  // namespace

// Test-only: expose the middle-layer split plan. Returns the selected split
// ("m"/"n") plus the per-thread (begin, size) ranges for BOTH the M and N
// candidates, so tests can pin coverage / disjointness / 8-alignment.
std::tuple<std::string, std::vector<std::pair<int64_t, int64_t>>, std::vector<std::pair<int64_t, int64_t>>>
fused_moe_test_split_plan(std::string stage, int64_t M, int64_t K, int64_t N, int64_t group_size) {
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  const MoeGemmStage s = (stage == "w2") ? MoeGemmStage::kW2 : MoeGemmStage::kW13;
  const GemmSplitPlan plan = plan_team_gemm_split(s, M, K, N, group_size);
  const std::string selected = (plan.split == MoeGemmSplit::kM) ? "m" : "n";
  const GemmSplitPlan m_plan{MoeGemmSplit::kM};
  const GemmSplitPlan n_plan{MoeGemmSplit::kN};
  std::vector<std::pair<int64_t, int64_t>> m_ranges;
  std::vector<std::pair<int64_t, int64_t>> n_ranges;
  m_ranges.reserve(static_cast<size_t>(group_size));
  n_ranges.reserve(static_cast<size_t>(group_size));
  for (int64_t t = 0; t < group_size; ++t) {
    const SplitRange mr = team_gemm_split_range(m_plan, M, N, group_size, t);
    const SplitRange nr = team_gemm_split_range(n_plan, M, N, group_size, t);
    m_ranges.emplace_back(mr.begin, mr.size);
    n_ranges.emplace_back(nr.begin, nr.size);
  }
  return std::make_tuple(selected, m_ranges, n_ranges);
}

// Test-only entry: run the bottom-layer single_thread_gemm for a dense
// A[M, K] x B[N, K]^T -> C[M, N] (fp32), packing B and padding A/bias exactly
// like the MoE path. Lets unit tests pin the bottom layer against a reference
// matmul without going through MoE routing.
at::Tensor fused_moe_test_single_thread_gemm(at::Tensor A, at::Tensor B, c10::optional<at::Tensor> bias) {
#ifndef __aarch64__
  TORCH_CHECK(false, "fused_moe_test_single_thread_gemm requires AArch64");
#else
  check_bf16_cpu(A, "A");
  check_bf16_cpu(B, "B");
  TORCH_CHECK(A.dim() == 2, "A must be 2-D [M, K]");
  TORCH_CHECK(B.dim() == 2, "B must be 2-D [N, K]");
  A = A.contiguous();
  B = B.contiguous();
  const int64_t M = A.size(0);
  const int64_t K = A.size(1);
  const int64_t N = B.size(0);
  TORCH_CHECK(B.size(1) == K, "B second dim must equal K=", K, ", got ", B.size(1));
  check_positive_int(M, "M");
  check_positive_int(K, "K");
  check_positive_int(N, "N");
  const int64_t K_pad = ceil_to_multiple(K, kKernelTile);
  const int64_t N_pad = ceil_to_multiple(N, kKernelTile);

  std::vector<uint16_t> packed(static_cast<size_t>(K_pad * N_pad), static_cast<uint16_t>(0));
  pack_transposed_expert_weight(bf16_data_const(B), 0, N, K, K_pad, N_pad, packed.data());

  std::vector<uint16_t> a_pad(static_cast<size_t>(M * K_pad), static_cast<uint16_t>(0));
  const uint16_t* a_src = bf16_data_const(A);
  for (int64_t m = 0; m < M; ++m) {
    std::copy(a_src + m * K, a_src + m * K + K, a_pad.data() + m * K_pad);
  }
  std::vector<uint16_t> a_reorder(static_cast<size_t>(M * K_pad * 2), static_cast<uint16_t>(0));

  const at::Tensor bias_f32 = build_padded_bias_f32(bias, 1, N, N_pad);
  const float* bias_base = bias_f32.defined() ? bias_f32.data_ptr<float>() : nullptr;

  at::Tensor C_pad = at::zeros({M, N_pad}, at::TensorOptions().dtype(at::kFloat));
  single_thread_gemm(a_pad.data(), packed.data(), C_pad.data_ptr<float>(), a_reorder.data(), static_cast<int>(M),
                     static_cast<int>(K_pad), static_cast<int>(N_pad), static_cast<int>(N_pad), bias_base);

  at::Tensor out = at::empty({M, N}, at::TensorOptions().dtype(at::kFloat));
  const float* c_ptr = C_pad.data_ptr<float>();
  float* o_ptr = out.data_ptr<float>();
  for (int64_t m = 0; m < M; ++m) {
    std::copy(c_ptr + m * N_pad, c_ptr + m * N_pad + N, o_ptr + m * N);
  }
  return out;
#endif
}

// Test-only: pack w13[2F, H] with the interleaved gate/up layout, then run the
// bottom-layer fp32 GEMM. Returns C[M, 2*F_pad] whose columns are in the
// INTERLEAVED order (8-col blocks of 4 gate + 4 up). Lets a test pin the
// prepack permutation: interleaved col 8*(f/4)+(f%4) must equal reference
// gate feature f, and col 8*(f/4)+4+(f%4) must equal reference up feature f.
at::Tensor fused_moe_test_pack_interleaved_gemm(at::Tensor A, at::Tensor w13) {
#ifndef __aarch64__
  TORCH_CHECK(false, "fused_moe_test_pack_interleaved_gemm requires AArch64");
#else
  check_bf16_cpu(A, "A");
  check_bf16_cpu(w13, "w13");
  TORCH_CHECK(A.dim() == 2, "A must be 2-D [M, H]");
  TORCH_CHECK(w13.dim() == 2, "w13 must be 2-D [2F, H]");
  A = A.contiguous();
  w13 = w13.contiguous();
  const int64_t M = A.size(0);
  const int64_t H = A.size(1);
  const int64_t N13 = w13.size(0);
  TORCH_CHECK(w13.size(1) == H, "w13 second dim must equal H=", H);
  TORCH_CHECK(N13 % 2 == 0, "w13 output dim must be even, got ", N13);
  const int64_t F = N13 / 2;
  check_positive_int(M, "M");
  check_positive_int(H, "H");

  const int64_t K_pad = ceil_to_multiple(H, kKernelTile);
  const int64_t F_pad4 = ceil_to_multiple(F, 4);
  const int64_t N_pad = 2 * F_pad4;  // multiple of 8

  std::vector<uint16_t> packed(static_cast<size_t>(K_pad * N_pad), static_cast<uint16_t>(0));
  pack_transposed_expert_weight_interleaved(bf16_data_const(w13), 0, F, H, K_pad, N_pad, packed.data());

  std::vector<uint16_t> a_pad(static_cast<size_t>(M * K_pad), static_cast<uint16_t>(0));
  const uint16_t* a_src = bf16_data_const(A);
  for (int64_t m = 0; m < M; ++m) {
    std::copy(a_src + m * H, a_src + m * H + H, a_pad.data() + m * K_pad);
  }
  std::vector<uint16_t> a_reorder(static_cast<size_t>(M * K_pad * 2), static_cast<uint16_t>(0));

  at::Tensor C_pad = at::zeros({M, N_pad}, at::TensorOptions().dtype(at::kFloat));
  single_thread_gemm(a_pad.data(), packed.data(), C_pad.data_ptr<float>(), a_reorder.data(), static_cast<int>(M),
                     static_cast<int>(K_pad), static_cast<int>(N_pad), static_cast<int>(N_pad), nullptr);
  return C_pad;
#endif
}

// Test-only (Task 2): fused w13 kernel with gate*up (NO silu), bf16 output.
// A[M,H], w13[2F,H]. M must be a multiple of 8 (m=8 kernel only until Task 5).
// Returns intermediate[M, F] bf16 = (A @ w13_gate^T) * (A @ w13_up^T).
at::Tensor fused_moe_test_fused_w13_linear(at::Tensor A, at::Tensor w13) {
#ifndef __aarch64__
  TORCH_CHECK(false, "fused_moe_test_fused_w13_linear requires AArch64");
#else
  check_bf16_cpu(A, "A");
  check_bf16_cpu(w13, "w13");
  TORCH_CHECK(A.dim() == 2, "A must be 2-D [M, H]");
  TORCH_CHECK(w13.dim() == 2, "w13 must be 2-D [2F, H]");
  A = A.contiguous();
  w13 = w13.contiguous();
  const int64_t M = A.size(0);
  const int64_t H = A.size(1);
  const int64_t N13 = w13.size(0);
  TORCH_CHECK(w13.size(1) == H, "w13 second dim must equal H=", H);
  TORCH_CHECK(N13 % 2 == 0, "w13 output dim must be even, got ", N13);
  TORCH_CHECK(M % 8 == 0, "M must be a multiple of 8 (m=8 kernel only)");
  const int64_t F = N13 / 2;
  check_positive_int(M, "M");
  check_positive_int(H, "H");

  const int64_t K_pad = ceil_to_multiple(H, kKernelTile);
  const int64_t F_pad4 = ceil_to_multiple(F, 4);
  const int64_t N_pad = 2 * F_pad4;  // multiple of 8

  std::vector<uint16_t> packed(static_cast<size_t>(K_pad * N_pad), static_cast<uint16_t>(0));
  pack_transposed_expert_weight_interleaved(bf16_data_const(w13), 0, F, H, K_pad, N_pad, packed.data());

  std::vector<uint16_t> a_pad(static_cast<size_t>(M * K_pad), static_cast<uint16_t>(0));
  const uint16_t* a_src = bf16_data_const(A);
  for (int64_t m = 0; m < M; ++m) {
    std::copy(a_src + m * H, a_src + m * H + H, a_pad.data() + m * K_pad);
  }
  std::vector<uint16_t> a_reorder(static_cast<size_t>(M * K_pad * 2), static_cast<uint16_t>(0));

  at::Tensor C = at::zeros({M, F_pad4}, at::TensorOptions().dtype(at::kBFloat16));
  gemm_params_t p;
  p.m = static_cast<int>(M);
  p.k = static_cast<int>(K_pad);
  p.n = static_cast<int>(N_pad);
  p.lda = static_cast<int>(K_pad);
  p.ldb = static_cast<int>(K_pad);
  p.ldc = static_cast<int>(F_pad4);
  bf16gemm_k_ld_silu_linear(a_pad.data(), packed.data(), bf16_data(C), a_reorder.data(), &p);
  return C.slice(1, 0, F).contiguous();
#endif
}

// Test-only (Task 3+): fused w13 SiLU-and-mul kernel, bf16 output [M,F].
// degree selects the exp polynomial (5 available in Task 3; 4/6 in Task 4).
// out[m,f] = silu(gate[m,f]) * up[m,f]. M must be a multiple of 8 until Task 5.
at::Tensor fused_moe_test_fused_w13_silu(at::Tensor A, at::Tensor w13, int64_t degree) {
#ifndef __aarch64__
  TORCH_CHECK(false, "fused_moe_test_fused_w13_silu requires AArch64");
#else
  check_bf16_cpu(A, "A");
  check_bf16_cpu(w13, "w13");
  TORCH_CHECK(A.dim() == 2, "A must be 2-D [M, H]");
  TORCH_CHECK(w13.dim() == 2, "w13 must be 2-D [2F, H]");
  A = A.contiguous();
  w13 = w13.contiguous();
  const int64_t M = A.size(0);
  const int64_t H = A.size(1);
  const int64_t N13 = w13.size(0);
  TORCH_CHECK(w13.size(1) == H, "w13 second dim must equal H=", H);
  TORCH_CHECK(N13 % 2 == 0, "w13 output dim must be even, got ", N13);
  const int64_t F = N13 / 2;
  check_positive_int(M, "M");
  check_positive_int(H, "H");

  const int64_t K_pad = ceil_to_multiple(H, kKernelTile);
  const int64_t F_pad4 = ceil_to_multiple(F, 4);
  const int64_t N_pad = 2 * F_pad4;

  std::vector<uint16_t> packed(static_cast<size_t>(K_pad * N_pad), static_cast<uint16_t>(0));
  pack_transposed_expert_weight_interleaved(bf16_data_const(w13), 0, F, H, K_pad, N_pad, packed.data());

  std::vector<uint16_t> a_pad(static_cast<size_t>(M * K_pad), static_cast<uint16_t>(0));
  const uint16_t* a_src = bf16_data_const(A);
  for (int64_t m = 0; m < M; ++m) {
    std::copy(a_src + m * H, a_src + m * H + H, a_pad.data() + m * K_pad);
  }
  std::vector<uint16_t> a_reorder(static_cast<size_t>(M * K_pad * 2), static_cast<uint16_t>(0));

  at::Tensor C = at::zeros({M, F_pad4}, at::TensorOptions().dtype(at::kBFloat16));
  single_thread_gemm_fused_silu(a_pad.data(), packed.data(), bf16_data(C), a_reorder.data(), static_cast<int>(M),
                                static_cast<int>(K_pad), static_cast<int>(N_pad), static_cast<int>(F_pad4), degree);
  return C.slice(1, 0, F).contiguous();
#endif
}

// Test-only (N-split): assemble intermediate[M,F] by running team_fused_w13_silu
// for every local_tid of a group of `group_size` into ONE shared buffer. The
// result must equal the whole-slice single_thread_gemm_fused_silu output.
at::Tensor fused_moe_test_team_fused_w13_silu(at::Tensor A, at::Tensor w13, int64_t group_size, int64_t degree) {
#ifndef __aarch64__
  TORCH_CHECK(false, "fused_moe_test_team_fused_w13_silu requires AArch64");
#else
  check_bf16_cpu(A, "A");
  check_bf16_cpu(w13, "w13");
  TORCH_CHECK(A.dim() == 2, "A must be 2-D [M, H]");
  TORCH_CHECK(w13.dim() == 2, "w13 must be 2-D [2F, H]");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  A = A.contiguous();
  w13 = w13.contiguous();
  const int64_t M = A.size(0);
  const int64_t H = A.size(1);
  const int64_t N13 = w13.size(0);
  TORCH_CHECK(w13.size(1) == H, "w13 second dim must equal H=", H);
  TORCH_CHECK(N13 % 2 == 0, "w13 output dim must be even, got ", N13);
  const int64_t F = N13 / 2;
  check_positive_int(M, "M");
  check_positive_int(H, "H");

  const int64_t K_pad = ceil_to_multiple(H, kKernelTile);
  const int64_t F_pad4 = ceil_to_multiple(F, 4);
  const int64_t N_pad = 2 * F_pad4;

  std::vector<uint16_t> packed(static_cast<size_t>(K_pad * N_pad), static_cast<uint16_t>(0));
  pack_transposed_expert_weight_interleaved(bf16_data_const(w13), 0, F, H, K_pad, N_pad, packed.data());

  std::vector<uint16_t> a_pad(static_cast<size_t>(M * K_pad), static_cast<uint16_t>(0));
  const uint16_t* a_src = bf16_data_const(A);
  for (int64_t m = 0; m < M; ++m) {
    std::copy(a_src + m * H, a_src + m * H + H, a_pad.data() + m * K_pad);
  }
  // Per-local_tid private A-reorder scratch (>= M*K_pad*2 each, matching the
  // single-thread margin that covers m1/m2 tail padding).
  const int64_t a_reorder_stride = M * K_pad * 2;
  std::vector<uint16_t> a_reorder(static_cast<size_t>(group_size * a_reorder_stride), static_cast<uint16_t>(0));

  at::Tensor C = at::zeros({M, F_pad4}, at::TensorOptions().dtype(at::kBFloat16));
  for (int64_t local_tid = 0; local_tid < group_size; ++local_tid) {
    TeamContext team;
    team.group_size = group_size;
    team.local_tid = local_tid;
    team.barrier = nullptr;
    team.a_reorder = a_reorder.data() + local_tid * a_reorder_stride;
    team_fused_w13_silu(team, a_pad.data(), packed.data(), bf16_data(C), static_cast<int>(M), static_cast<int>(K_pad),
                        static_cast<int>(N_pad), static_cast<int>(F_pad4), degree);
  }
  return C.slice(1, 0, F).contiguous();
#endif
}

// Test-only: pack row-major A[total_rows, K] into the m8 reorder layout via
// pack_a_reorder_m8 over all 8-row blocks. K must be a multiple of 4. Returns a
// flat bf16 buffer of length ceil(total_rows/8)*8*K.
at::Tensor fused_moe_test_pack_a_reorder_m8(at::Tensor A) {
#ifndef __aarch64__
  TORCH_CHECK(false, "fused_moe_test_pack_a_reorder_m8 requires AArch64");
#else
  check_bf16_cpu(A, "A");
  TORCH_CHECK(A.dim() == 2, "A must be 2-D [total_rows, K]");
  A = A.contiguous();
  const int64_t total_rows = A.size(0);
  const int64_t K = A.size(1);
  TORCH_CHECK(K % 4 == 0, "K must be a multiple of 4, got ", K);
  const int64_t nb = (total_rows + 7) / 8;
  at::Tensor out = at::zeros({nb * 8 * K}, at::TensorOptions().dtype(at::kBFloat16));
  pack_a_reorder_m8(bf16_data_const(A), bf16_data(out), static_cast<int>(total_rows), static_cast<int>(K), 0,
                    static_cast<int>(nb));
  return out;
#endif
}

// Test-only: fused gather + m8 reorder pack. input[num_tokens, H] bf16, routes
// [rows] int64 (flat indices; token = flat / top_k). Returns a flat bf16 buffer
// of length ceil(rows/8)*8*K_pad, bit-identical to gather-to-rowmajor(K_pad)
// followed by pack_a_reorder_m8.
at::Tensor fused_moe_test_gather_pack_a_reorder_m8(at::Tensor input, at::Tensor routes, int64_t top_k, int64_t K_pad) {
#ifndef __aarch64__
  TORCH_CHECK(false, "fused_moe_test_gather_pack_a_reorder_m8 requires AArch64");
#else
  check_bf16_cpu(input, "input");
  TORCH_CHECK(input.dim() == 2, "input must be 2-D [num_tokens, H]");
  TORCH_CHECK(routes.dim() == 1, "routes must be 1-D [rows]");
  TORCH_CHECK(routes.scalar_type() == at::kLong, "routes must be int64");
  TORCH_CHECK(top_k > 0, "top_k must be positive");
  TORCH_CHECK(K_pad % 4 == 0, "K_pad must be a multiple of 4, got ", K_pad);
  input = input.contiguous();
  routes = routes.contiguous();
  const int64_t H = input.size(1);
  const int64_t rows = routes.size(0);
  TORCH_CHECK(K_pad >= H, "K_pad must be >= H");
  const int64_t nb = (rows + 7) / 8;
  at::Tensor out = at::zeros({nb * 8 * K_pad}, at::TensorOptions().dtype(at::kBFloat16));
  gather_pack_a_reorder_m8(bf16_data_const(input), H, routes.data_ptr<int64_t>(), top_k, bf16_data(out),
                           static_cast<int>(rows), static_cast<int>(K_pad), 0, static_cast<int>(nb));
  return out;
#endif
}

// Test-only: fused w13 SiLU-and-mul with a packed-C (reorder-m8) store. Pads M
// to a multiple of 8, pre-packs A, runs the packc m8 kernel, and returns the
// packed intermediate as a flat bf16 buffer of length ceil(M/8)*8*F_pad4. Must
// equal fused_moe_test_fused_w13_silu (row-major) padded to [ceil8(M), F_pad4]
// then packed via pack_a_reorder_m8.
at::Tensor fused_moe_test_fused_w13_silu_packc_tail(at::Tensor A, at::Tensor w13, int64_t degree) {
#ifndef __aarch64__
  TORCH_CHECK(false, "fused_moe_test_fused_w13_silu_packc_tail requires AArch64");
#else
  check_bf16_cpu(A, "A");
  check_bf16_cpu(w13, "w13");
  TORCH_CHECK(A.dim() == 2, "A must be 2-D [M, H]");
  TORCH_CHECK(w13.dim() == 2, "w13 must be 2-D [2F, H]");
  A = A.contiguous();
  w13 = w13.contiguous();
  const int64_t M = A.size(0);
  const int64_t H = A.size(1);
  const int64_t N13 = w13.size(0);
  TORCH_CHECK(w13.size(1) == H, "w13 second dim must equal H=", H);
  TORCH_CHECK(N13 % 2 == 0, "w13 output dim must be even, got ", N13);
  const int64_t F = N13 / 2;
  check_positive_int(M, "M");
  check_positive_int(H, "H");
  const int64_t K_pad = ceil_to_multiple(H, kKernelTile);
  const int64_t F_pad4 = ceil_to_multiple(F, 4);
  const int64_t N_pad = 2 * F_pad4;
  const int64_t Mp = ceil_to_multiple(M, kKernelTile);

  std::vector<uint16_t> packed(static_cast<size_t>(K_pad * N_pad), static_cast<uint16_t>(0));
  pack_transposed_expert_weight_interleaved(bf16_data_const(w13), 0, F, H, K_pad, N_pad, packed.data());
  std::vector<uint16_t> a_pad(static_cast<size_t>(M * K_pad), static_cast<uint16_t>(0));
  const uint16_t* a_src = bf16_data_const(A);
  for (int64_t m = 0; m < M; ++m) {
    std::copy(a_src + m * H, a_src + m * H + H, a_pad.data() + m * K_pad);
  }
  std::vector<uint16_t> packed_a(static_cast<size_t>(Mp * K_pad), static_cast<uint16_t>(0));
  pack_a_reorder_m8(a_pad.data(), packed_a.data(), static_cast<int>(M), static_cast<int>(K_pad), 0,
                    static_cast<int>(Mp / 8));

  at::Tensor C = at::zeros({Mp * F_pad4}, at::TensorOptions().dtype(at::kBFloat16));
  const FusedSiluKernelSet ks = fused_silu_packc_set_for_degree(degree);
  TORCH_CHECK(ks.m8 != nullptr, "unsupported fused silu exp degree ", degree);
  packc_w13_tail_dispatch(packed_a.data(), packed.data(), bf16_data(C), static_cast<int>(M), static_cast<int>(K_pad),
                          static_cast<int>(N_pad), static_cast<int>(F_pad4), ks);
  return C;
#endif
}

at::Tensor fused_moe_test_fused_w13_silu_packc(at::Tensor A, at::Tensor w13, int64_t degree) {
#ifndef __aarch64__
  TORCH_CHECK(false, "fused_moe_test_fused_w13_silu_packc requires AArch64");
#else
  check_bf16_cpu(A, "A");
  check_bf16_cpu(w13, "w13");
  TORCH_CHECK(A.dim() == 2, "A must be 2-D [M, H]");
  TORCH_CHECK(w13.dim() == 2, "w13 must be 2-D [2F, H]");
  A = A.contiguous();
  w13 = w13.contiguous();
  const int64_t M = A.size(0);
  const int64_t H = A.size(1);
  const int64_t N13 = w13.size(0);
  TORCH_CHECK(w13.size(1) == H, "w13 second dim must equal H=", H);
  TORCH_CHECK(N13 % 2 == 0, "w13 output dim must be even, got ", N13);
  const int64_t F = N13 / 2;
  check_positive_int(M, "M");
  check_positive_int(H, "H");

  const int64_t K_pad = ceil_to_multiple(H, kKernelTile);
  const int64_t F_pad4 = ceil_to_multiple(F, 4);
  const int64_t N_pad = 2 * F_pad4;
  const int64_t Mp = ceil_to_multiple(M, kKernelTile);

  std::vector<uint16_t> packed(static_cast<size_t>(K_pad * N_pad), static_cast<uint16_t>(0));
  pack_transposed_expert_weight_interleaved(bf16_data_const(w13), 0, F, H, K_pad, N_pad, packed.data());

  std::vector<uint16_t> a_pad(static_cast<size_t>(M * K_pad), static_cast<uint16_t>(0));
  const uint16_t* a_src = bf16_data_const(A);
  for (int64_t m = 0; m < M; ++m) {
    std::copy(a_src + m * H, a_src + m * H + H, a_pad.data() + m * K_pad);
  }
  std::vector<uint16_t> packed_a(static_cast<size_t>(Mp * K_pad), static_cast<uint16_t>(0));
  pack_a_reorder_m8(a_pad.data(), packed_a.data(), static_cast<int>(M), static_cast<int>(K_pad), 0,
                    static_cast<int>(Mp / 8));

  at::Tensor C = at::zeros({Mp * F_pad4}, at::TensorOptions().dtype(at::kBFloat16));
  FusedSiluKernelFn k8 = fused_silu_packc_m8_for_degree(degree);
  TORCH_CHECK(k8 != nullptr, "unsupported fused silu exp degree ", degree);
  gemm_params_t p;
  p.m = static_cast<int>(Mp);
  p.k = static_cast<int>(K_pad);
  p.n = static_cast<int>(N_pad);
  p.lda = static_cast<int>(K_pad);
  p.ldb = static_cast<int>(K_pad);
  p.ldc = static_cast<int>(F_pad4);
  k8(packed_a.data(), packed.data(), bf16_data(C), nullptr, &p);
  return C;
#endif
}

// Test-only: run the middle-layer team_gemm on the resident thread pool for a
// dense A[M,K] x B[N,K]^T -> C[M,N] (fp32). split is "m", "n", or "auto".
at::Tensor fused_moe_test_team_gemm(at::Tensor A, at::Tensor B, int64_t group_size, std::string split,
                                    c10::optional<at::Tensor> bias) {
#ifndef __aarch64__
  TORCH_CHECK(false, "fused_moe_test_team_gemm requires AArch64");
#else
  check_bf16_cpu(A, "A");
  check_bf16_cpu(B, "B");
  TORCH_CHECK(A.dim() == 2, "A must be 2-D [M, K]");
  TORCH_CHECK(B.dim() == 2, "B must be 2-D [N, K]");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  A = A.contiguous();
  B = B.contiguous();
  const int64_t M = A.size(0);
  const int64_t K = A.size(1);
  const int64_t N = B.size(0);
  TORCH_CHECK(B.size(1) == K, "B second dim must equal K=", K);
  check_positive_int(M, "M");
  const int64_t K_pad = ceil_to_multiple(K, kKernelTile);
  const int64_t N_pad = ceil_to_multiple(N, kKernelTile);

  std::vector<uint16_t> packed(static_cast<size_t>(K_pad * N_pad), static_cast<uint16_t>(0));
  pack_transposed_expert_weight(bf16_data_const(B), 0, N, K, K_pad, N_pad, packed.data());
  std::vector<uint16_t> a_pad(static_cast<size_t>(M * K_pad), static_cast<uint16_t>(0));
  const uint16_t* a_src = bf16_data_const(A);
  for (int64_t m = 0; m < M; ++m) {
    std::copy(a_src + m * K, a_src + m * K + K, a_pad.data() + m * K_pad);
  }
  // Worst case (N-split): each thread re-reorders the full M rows.
  const int64_t reorder_stride = M * K_pad * 2;
  std::vector<uint16_t> a_reorder(static_cast<size_t>(group_size * reorder_stride), static_cast<uint16_t>(0));

  const at::Tensor bias_f32 = build_padded_bias_f32(bias, 1, N, N_pad);
  const float* bias_base = bias_f32.defined() ? bias_f32.data_ptr<float>() : nullptr;

  at::Tensor C_pad = at::zeros({M, N_pad}, at::TensorOptions().dtype(at::kFloat));
  float* C_ptr = C_pad.data_ptr<float>();

  GemmSplitPlan plan;
  if (split == "m") {
    plan.split = MoeGemmSplit::kM;
  } else if (split == "n") {
    plan.split = MoeGemmSplit::kN;
  } else {
    plan = plan_team_gemm_split(MoeGemmStage::kW13, M, K_pad, N_pad, group_size);
  }

  ThreadBarrier barrier(group_size);
  run_fixed_threads(group_size, [&](int64_t tid) {
    TeamContext team;
    team.group_size = group_size;
    team.local_tid = tid;
    team.barrier = group_size > 1 ? &barrier : nullptr;
    team.a_reorder = a_reorder.data() + tid * reorder_stride;
    team_gemm(team, plan, a_pad.data(), packed.data(), C_ptr, M, K_pad, N_pad, N_pad, bias_base);
  });

  at::Tensor out = at::empty({M, N}, at::TensorOptions().dtype(at::kFloat));
  const float* c_ptr = C_pad.data_ptr<float>();
  float* o_ptr = out.data_ptr<float>();
  for (int64_t m = 0; m < M; ++m) {
    std::copy(c_ptr + m * N_pad, c_ptr + m * N_pad + N, o_ptr + m * N);
  }
  return out;
#endif
}

// Test/bench-only: time ONLY the middle-layer team_gemm loop for a dense
// A[M,K] x B[N,K]^T -> C[M,N]. Weights are packed and A padded ONCE outside the
// timed region; the whole warmup+timed loop runs inside a single pooled job so
// per-iteration pool-dispatch overhead is excluded. Returns per-run
// milliseconds (length `runs`). split is "m", "n", or "auto".
std::vector<double> fused_moe_bench_team_gemm(at::Tensor A, at::Tensor B, int64_t group_size, std::string split,
                                              c10::optional<at::Tensor> bias, int64_t warmup, int64_t runs) {
#ifndef __aarch64__
  TORCH_CHECK(false, "fused_moe_bench_team_gemm requires AArch64");
#else
  check_bf16_cpu(A, "A");
  check_bf16_cpu(B, "B");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A[M,K], B[N,K] required");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(runs > 0, "runs must be positive");
  TORCH_CHECK(warmup >= 0, "warmup must be non-negative");
  A = A.contiguous();
  B = B.contiguous();
  const int64_t M = A.size(0);
  const int64_t K = A.size(1);
  const int64_t N = B.size(0);
  TORCH_CHECK(B.size(1) == K, "B second dim must equal K=", K);
  check_positive_int(M, "M");
  const int64_t K_pad = ceil_to_multiple(K, kKernelTile);
  const int64_t N_pad = ceil_to_multiple(N, kKernelTile);

  std::vector<uint16_t> packed(static_cast<size_t>(K_pad * N_pad), static_cast<uint16_t>(0));
  pack_transposed_expert_weight(bf16_data_const(B), 0, N, K, K_pad, N_pad, packed.data());
  std::vector<uint16_t> a_pad(static_cast<size_t>(M * K_pad), static_cast<uint16_t>(0));
  const uint16_t* a_src = bf16_data_const(A);
  for (int64_t m = 0; m < M; ++m) {
    std::copy(a_src + m * K, a_src + m * K + K, a_pad.data() + m * K_pad);
  }
  const int64_t reorder_stride = M * K_pad * 2;
  std::vector<uint16_t> a_reorder(static_cast<size_t>(group_size * reorder_stride), static_cast<uint16_t>(0));

  const at::Tensor bias_f32 = build_padded_bias_f32(bias, 1, N, N_pad);
  const float* bias_base = bias_f32.defined() ? bias_f32.data_ptr<float>() : nullptr;
  at::Tensor C_pad = at::zeros({M, N_pad}, at::TensorOptions().dtype(at::kFloat));
  float* C_ptr = C_pad.data_ptr<float>();

  GemmSplitPlan plan;
  if (split == "m") {
    plan.split = MoeGemmSplit::kM;
  } else if (split == "n") {
    plan.split = MoeGemmSplit::kN;
  } else {
    plan = plan_team_gemm_split(MoeGemmStage::kW13, M, K_pad, N_pad, group_size);
  }

  std::vector<double> times_ms(static_cast<size_t>(runs), 0.0);
  ThreadBarrier barrier(group_size);
  run_fixed_threads(group_size, [&](int64_t tid) {
    TeamContext team;
    team.group_size = group_size;
    team.local_tid = tid;
    team.barrier = group_size > 1 ? &barrier : nullptr;
    team.a_reorder = a_reorder.data() + tid * reorder_stride;
    for (int64_t r = 0; r < warmup; ++r) {
      team_gemm(team, plan, a_pad.data(), packed.data(), C_ptr, M, K_pad, N_pad, N_pad, bias_base);
    }
    for (int64_t r = 0; r < runs; ++r) {
      const auto t0 = ::fused_cpp::profile::now();
      team_gemm(team, plan, a_pad.data(), packed.data(), C_ptr, M, K_pad, N_pad, N_pad, bias_base);
      // team_gemm's end barrier means all threads have finished when tid 0
      // returns, so tid 0's elapsed captures the full team GEMM time.
      if (tid == 0) {
        times_ms[static_cast<size_t>(r)] = ::fused_cpp::profile::elapsed_ms(t0);
      }
    }
  });
  return times_ms;
#endif
}

at::Tensor fused_moe_test_sve_packed_gemm(at::Tensor A, at::Tensor packed_B, int64_t K, int64_t N,
                                          int64_t n_tile, bool use_jit) {
#if !defined(__aarch64__) || !defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  TORCH_CHECK(false, "fused_moe_test_sve_packed_gemm requires AArch64 SVE");
#else
  check_bf16_cpu(A, "A");
  TORCH_CHECK(A.dim() == 2, "A must be 2-D [M, K]");
  TORCH_CHECK(A.size(1) == K, "A second dimension must equal K=", K);
  check_positive_int(A.size(0), "M");
  check_positive_int(K, "K");
  check_positive_int(N, "N");
  TORCH_CHECK(K % 8 == 0, "K must be divisible by 8");
  TORCH_CHECK(n_tile == ::fused_cpp::moe_sve::n_tile(), "n_tile mismatch: requested ", n_tile, ", runtime ",
              ::fused_cpp::moe_sve::n_tile());
  TORCH_CHECK(N % n_tile == 0, "N must be divisible by n_tile");
  TORCH_CHECK(::fused_cpp::moe_sve::k_block(static_cast<int>(K)) == K,
              "pure SVE JIT GEMM test requires one-chunk K packing");
  A = A.contiguous();
  const PackedExperts weights = checked_packed_experts(packed_B, K, N, "packed_B", n_tile);

  const int rows = static_cast<int>(A.size(0));
  const int64_t packed_rows = sve_hybrid_packed_rows(rows);
  std::vector<uint16_t> packed_a(static_cast<size_t>(packed_rows * K), static_cast<uint16_t>(0));
  std::vector<int64_t> routes(static_cast<size_t>(rows));
  std::iota(routes.begin(), routes.end(), int64_t{0});
  gather_pack_a_reorder_sve_hybrid(bf16_data_const(A), K, routes.data(), 1, packed_a.data(), rows,
                                   static_cast<int>(K), int64_t{1}, int64_t{0});

  at::Tensor output = at::zeros({packed_rows, N}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  const uint16_t* packed_b = bf16_data_const(weights.tensor);
  if (use_jit) {
    const bool dispatched = sve_jit_packed_gemm_f32_exact_dispatch(
        packed_a.data(), packed_b, output.data_ptr<float>(), rows, static_cast<int>(K), static_cast<int>(N),
        static_cast<int>(N), static_cast<int>(N), 0);
    TORCH_CHECK(dispatched, "failed to dispatch standalone SVE JIT GEMM");
  } else {
    sve_asm_packed_w2_hybrid_dispatch(packed_a.data(), packed_b, output.data_ptr<float>(), rows,
                                      static_cast<int>(K), static_cast<int>(N), static_cast<int>(N),
                                      static_cast<int>(N), 0);
  }
  return output.narrow(0, 0, rows).clone();
#endif
}

// Benchmark only the standalone exact-M SVE JIT GEMM for a W13-shaped packed
// weight. Packing and allocation are outside the timed region. A may be either
// [M,K] or [copies,M,K]; timed iterations independently rotate packed A copies
// and packed-B experts so cache-state probes can stream either operand. Full
// M12-compatible probe modes may traverse any positive multiple of 12 rows.
std::vector<double> fused_moe_bench_sve_jit_w13_gemm(at::Tensor A, at::Tensor w13_packed, int64_t K, int64_t N,
                                                     int64_t n_tile, int64_t n_ranges, int64_t warmup, int64_t runs,
                                                     int64_t probe_mode) {
#if !defined(__aarch64__) || !defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  TORCH_CHECK(false, "fused_moe_bench_sve_jit_w13_gemm requires AArch64 SVE");
#else
  check_bf16_cpu(A, "A");
  TORCH_CHECK(A.dim() == 2 || A.dim() == 3, "A must be 2-D [M, K] or 3-D [copies, M, K]");
  TORCH_CHECK(A.size(-1) == K, "A last dimension must equal K=", K);
  const int64_t a_copies = A.dim() == 3 ? A.size(0) : 1;
  const int64_t rows64 = A.dim() == 3 ? A.size(1) : A.size(0);
  check_positive_int(a_copies, "A copies");
  check_positive_int(rows64, "M");
  check_positive_int(K, "K");
  check_positive_int(N, "N");
  TORCH_CHECK(n_tile == ::fused_cpp::moe_sve::n_tile(), "n_tile mismatch: requested ", n_tile, ", runtime ",
              ::fused_cpp::moe_sve::n_tile());
  TORCH_CHECK(N % n_tile == 0, "N must be divisible by n_tile");
  TORCH_CHECK(n_ranges > 0 && N % n_ranges == 0, "n_ranges must divide N");
  TORCH_CHECK((N / n_ranges) % n_tile == 0, "each N range must contain whole SVE N tiles");
  TORCH_CHECK(warmup >= 0, "warmup must be non-negative");
  TORCH_CHECK(runs > 0, "runs must be positive");
  TORCH_CHECK(probe_mode >= 0 && probe_mode <= 12, "probe_mode must be in [0, 12]");
  A = A.contiguous();
  const PackedExperts weights = checked_packed_experts(w13_packed, K, N, "w13_packed", n_tile);
  TORCH_CHECK(weights.E > 0, "w13_packed must contain at least one expert");
  TORCH_CHECK(sve_jit_configuration_supported(SveJitOperation::kGemmF32, static_cast<int>(K), 0, nullptr),
              "plain SVE JIT GEMM is unavailable for this configuration");

  const int rows = static_cast<int>(rows64);
  const int64_t packed_rows = sve_hybrid_packed_rows(rows);
  const size_t packed_a_stride = static_cast<size_t>(packed_rows * K);
  std::vector<uint16_t> packed_a(static_cast<size_t>(a_copies) * packed_a_stride, static_cast<uint16_t>(0));
  std::vector<int64_t> routes(static_cast<size_t>(rows));
  std::iota(routes.begin(), routes.end(), int64_t{0});
  const uint16_t* a_ptr = bf16_data_const(A);
  const size_t a_stride = static_cast<size_t>(rows64 * K);
  for (int64_t copy = 0; copy < a_copies; ++copy) {
    gather_pack_a_reorder_sve_hybrid(a_ptr + static_cast<size_t>(copy) * a_stride, K, routes.data(), 1,
                                     packed_a.data() + static_cast<size_t>(copy) * packed_a_stride, rows,
                                     static_cast<int>(K), int64_t{1}, int64_t{0});
  }
  at::Tensor output = at::empty({packed_rows, N}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  float* output_ptr = output.data_ptr<float>();
  const uint16_t* weights_ptr = bf16_data_const(weights.tensor);
  const int range_cols = static_cast<int>(N / n_ranges);
  ::fused_cpp::moe_sve::jit::KernelFn probe_kernel = nullptr;
  if (probe_mode != 0) {
    const int probe_rows = rows > 12 ? 12 : rows;
    TORCH_CHECK(rows <= 12 || rows % 12 == 0,
                "full-M SVE JIT probes require M to be at most 12 or a positive multiple of 12");
    std::string error;
    probe_kernel = ::fused_cpp::moe_sve::jit::get_probe_kernel(
        probe_rows, static_cast<::fused_cpp::moe_sve::jit::ProbeMode>(probe_mode), &error);
    TORCH_CHECK(probe_kernel != nullptr, "failed to generate SVE JIT probe kernel: ", error);
  }

  auto run_one = [&](int64_t iteration) {
    const int64_t a_copy = iteration % a_copies;
    const int64_t expert = iteration % weights.E;
    const uint16_t* packed_a_ptr = packed_a.data() + static_cast<size_t>(a_copy) * packed_a_stride;
    const uint16_t* packed_b = weights_ptr + expert * weights.packed_stride;
    for (int64_t range = 0; range < n_ranges; ++range) {
      const int n_begin = static_cast<int>(range * range_cols);
      if (probe_kernel != nullptr) {
        const int panel_rows = rows > 12 ? 12 : rows;
        for (int m_begin = 0; m_begin < rows; m_begin += panel_rows) {
          SveKBlockParams p = make_sve_kblock_params(panel_rows, static_cast<int>(K), range_cols, static_cast<int>(N),
                                                     static_cast<int>(N), n_begin, 0);
          probe_kernel(packed_a_ptr + static_cast<size_t>(m_begin) * K, packed_b,
                       output_ptr + static_cast<size_t>(m_begin) * N + n_begin, nullptr, &p.gemm);
        }
      } else {
        const bool dispatched = sve_jit_packed_gemm_f32_exact_dispatch(
            packed_a_ptr, packed_b, output_ptr + n_begin, rows, static_cast<int>(K), range_cols, static_cast<int>(N),
            static_cast<int>(N), n_begin);
        TORCH_CHECK(dispatched, "failed to dispatch plain SVE JIT W13 GEMM");
      }
    }
  };

#if defined(__linux__)
  const char* stop_after_setup = std::getenv("FUSED_CPP_MOE_BENCH_STOP_AFTER_SETUP");
  const char* profile_window = std::getenv("FUSED_CPP_MOE_BENCH_PROFILE_WINDOW");
  const bool stop_before_kernels =
      (stop_after_setup != nullptr && stop_after_setup[0] != '\0' && stop_after_setup[0] != '0') ||
      (profile_window != nullptr && profile_window[0] != '\0' && profile_window[0] != '0');
  if (stop_before_kernels) {
    std::raise(SIGSTOP);
  }
#endif
  for (int64_t iteration = 0; iteration < warmup; ++iteration) {
    run_one(iteration);
  }
  std::vector<double> times_ms(static_cast<size_t>(runs), 0.0);
  for (int64_t iteration = 0; iteration < runs; ++iteration) {
    const auto begin = ::fused_cpp::profile::now();
    run_one(iteration + warmup);
    times_ms[static_cast<size_t>(iteration)] = ::fused_cpp::profile::elapsed_ms(begin);
  }
#if defined(__linux__)
  if (profile_window != nullptr && profile_window[0] != '\0' && profile_window[0] != '0') {
    std::raise(SIGSTOP);
  }
#endif
  return times_ms;
#endif
}

// GEMM-only microbench for the w13 fused-silu packc path (single thread, weight
// packed once outside the loop). mode 0 = per-tail dispatch over `rows`; mode 1
// = always-pad-to-8 m8. Returns per-run milliseconds (length `runs`).
std::vector<double> fused_moe_bench_fused_w13_silu_packc_tail(at::Tensor A, at::Tensor w13, int64_t degree,
                                                              int64_t mode, int64_t warmup, int64_t runs) {
#ifndef __aarch64__
  TORCH_CHECK(false, "requires AArch64");
#else
  check_bf16_cpu(A, "A");
  check_bf16_cpu(w13, "w13");
  A = A.contiguous();
  w13 = w13.contiguous();
  const int64_t M = A.size(0);
  const int64_t H = A.size(1);
  const int64_t N13 = w13.size(0);
  const int64_t F = N13 / 2;
  const int64_t K_pad = ceil_to_multiple(H, kKernelTile);
  const int64_t F_pad4 = ceil_to_multiple(F, 4);
  const int64_t N_pad = 2 * F_pad4;
  const int64_t Mp = ceil_to_multiple(M, kKernelTile);
  std::vector<uint16_t> packed(static_cast<size_t>(K_pad * N_pad), static_cast<uint16_t>(0));
  pack_transposed_expert_weight_interleaved(bf16_data_const(w13), 0, F, H, K_pad, N_pad, packed.data());
  std::vector<uint16_t> a_pad(static_cast<size_t>(M * K_pad), static_cast<uint16_t>(0));
  const uint16_t* a_src = bf16_data_const(A);
  for (int64_t m = 0; m < M; ++m) {
    std::copy(a_src + m * H, a_src + m * H + H, a_pad.data() + m * K_pad);
  }
  std::vector<uint16_t> packed_a(static_cast<size_t>(Mp * K_pad), static_cast<uint16_t>(0));
  pack_a_reorder_m8(a_pad.data(), packed_a.data(), static_cast<int>(M), static_cast<int>(K_pad), 0,
                    static_cast<int>(Mp / 8));
  std::vector<uint16_t> C(static_cast<size_t>(Mp * F_pad4), 0);
  const FusedSiluKernelSet ks = fused_silu_packc_set_for_degree(degree);
  TORCH_CHECK(ks.m8 != nullptr, "bad degree");
  auto once = [&]() {
    if (mode == 0) {
      packc_w13_tail_dispatch(packed_a.data(), packed.data(), C.data(), static_cast<int>(M), static_cast<int>(K_pad),
                              static_cast<int>(N_pad), static_cast<int>(F_pad4), ks);
    } else {
      gemm_params_t p;
      p.m = static_cast<int>(Mp);
      p.k = static_cast<int>(K_pad);
      p.n = static_cast<int>(N_pad);
      p.lda = static_cast<int>(K_pad);
      p.ldb = static_cast<int>(K_pad);
      p.ldc = static_cast<int>(F_pad4);
      ks.m8(packed_a.data(), packed.data(), C.data(), nullptr, &p);
    }
  };
  for (int64_t r = 0; r < warmup; ++r) once();
  std::vector<double> times_ms(static_cast<size_t>(runs), 0.0);
  for (int64_t r = 0; r < runs; ++r) {
    const auto t0 = ::fused_cpp::profile::now();
    once();
    times_ms[static_cast<size_t>(r)] = ::fused_cpp::profile::elapsed_ms(t0);
  }
  return times_ms;
#endif
}

std::tuple<at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t, int64_t, int64_t>
fused_moe_bf16_tiled_prepare_weights(at::Tensor w13_weight, at::Tensor w2_weight, bool fuse_silu,
                                     std::string backend_name) {
  const ::fused_cpp::moe::MoeBackend& backend =
      ::fused_cpp::moe::resolve_backend(backend_name, fuse_silu);
#ifndef __aarch64__
  TORCH_CHECK(false, "fused_moe_bf16_tiled_prepare_weights has no implemented backend for this architecture");
#else
  check_bf16_cpu(w13_weight, "w13_weight");
  check_bf16_cpu(w2_weight, "w2_weight");
  TORCH_CHECK(w13_weight.dim() == 3, "w13_weight must be 3-D [experts, 2 * F, H]");
  TORCH_CHECK(w2_weight.dim() == 3, "w2_weight must be 3-D [experts, H, F]");
  TORCH_CHECK(w13_weight.size(0) == w2_weight.size(0), "w13_weight and w2_weight must have the same expert count");

  w13_weight = w13_weight.contiguous();
  w2_weight = w2_weight.contiguous();

  const int64_t E = w13_weight.size(0);
  const int64_t N13 = w13_weight.size(1);
  const int64_t H = w13_weight.size(2);
  TORCH_CHECK(N13 % 2 == 0, "w13 output dim must be even, got ", N13);
  const int64_t F = N13 / 2;
  TORCH_CHECK(w2_weight.size(1) == H && w2_weight.size(2) == F, "w2_weight shape mismatch: expected [", E, ", ", H,
              ", ", F, "], got [", w2_weight.size(0), ", ", w2_weight.size(1), ", ", w2_weight.size(2), "]");

  check_positive_int(H, "hidden size");
  check_positive_int(F, "ffn hidden size");
  check_positive_int(N13, "w13 output size");
  // Fused-SiLU layout requires F % 8 == 0 so the interleaved N13_pad (=2F)
  // stays 8-aligned and the fused intermediate stride equals w2's K_pad (=F).
  TORCH_CHECK(!fuse_silu || F % 8 == 0,
              "fuse_silu requires ffn hidden size F to be a multiple of 8, "
              "got F=",
              F);

  const int64_t K13 = H;
  const int64_t K2 = F;
  const int64_t N2 = H;
  const int64_t backend_id = static_cast<int64_t>(backend.id);
  const int64_t backend_n_tile = backend.n_tile();
  const int64_t K13_pad = backend.round_k(static_cast<int>(K13));
  const int64_t N13_pad = backend.round_n(static_cast<int>(N13));
  const int64_t K2_pad = backend.round_k(static_cast<int>(K2));
  const int64_t N2_pad = backend.round_n(static_cast<int>(N2));

  // Packed weights are the largest and hottest buffers in the whole path, so
  // they come from the shared page policy rather than the CPU caching allocator.
  // This also removes the second full-size copy the Python layer used to make in
  // order to relocate them onto a hugetlbfs file.
  at::Tensor w13_packed = fused_cpp::page_backed_empty({E, K13_pad * N13_pad}, w13_weight.options());
  at::Tensor w2_packed = fused_cpp::page_backed_empty({E, K2_pad * N2_pad}, w2_weight.options());

  const uint16_t* w13_ptr = bf16_data_const(w13_weight);
  const uint16_t* w2_ptr = bf16_data_const(w2_weight);
  uint16_t* w13_packed_ptr = bf16_data(w13_packed);
  uint16_t* w2_packed_ptr = bf16_data(w2_packed);

  int64_t prepack_threads = env_int_or_default("FUSED_CPP_MOE_PREPACK_THREADS", 1);
  TORCH_CHECK(prepack_threads > 0, "FUSED_CPP_MOE_PREPACK_THREADS must be positive, got ", prepack_threads);
  prepack_threads = std::min<int64_t>(prepack_threads, E);

  const int64_t w13_expert_stride = N13 * H;
  const int64_t w2_expert_stride = H * F;
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  if (backend.id == ::fused_cpp::moe::BackendId::kArmSveBf16 && fuse_silu) {
    prewarm_sve_jit_exact_m_kernels(static_cast<int>(K13_pad), static_cast<int>(K2_pad));
  }
#endif
  run_fixed_threads(prepack_threads, [&](int64_t tid) {
    for (int64_t e = tid; e < E; e += prepack_threads) {
      if (fuse_silu) {
        pack_transposed_expert_weight_interleaved(w13_ptr, e * w13_expert_stride, F, H, K13_pad, N13_pad,
                                                  w13_packed_ptr + e * K13_pad * N13_pad, &backend);
      } else {
        pack_transposed_expert_weight(w13_ptr, e * w13_expert_stride, N13, H, K13_pad, N13_pad,
                                      w13_packed_ptr + e * K13_pad * N13_pad, &backend);
      }
      pack_transposed_expert_weight(w2_ptr, e * w2_expert_stride, H, F, K2_pad, N2_pad,
                                    w2_packed_ptr + e * K2_pad * N2_pad, &backend);
    }
  });

  return std::make_tuple(w13_packed, K13, N13, w2_packed, K2, N2, backend_id, backend_n_tile);
#endif
}

at::Tensor fused_moe_bf16_tiled(at::Tensor input, at::Tensor w13_packed, int64_t w13_K, int64_t w13_N,
                                at::Tensor w2_packed, int64_t w2_K, int64_t w2_N, at::Tensor topk_weights,
                                at::Tensor topk_ids, c10::optional<at::Tensor> w13_bias,
                                c10::optional<at::Tensor> w2_bias, int64_t num_threads, std::string activation,
                                int64_t global_num_experts, bool skip_weighted, bool fuse_silu,
                                int64_t silu_poly_degree, int64_t gemm_backend, int64_t backend_n_tile,
                                c10::optional<at::Tensor> out) {
#ifndef __aarch64__
  TORCH_CHECK(false, "fused_moe_bf16_tiled requires AArch64");
#else
  const auto moe_trace_begin = ::fused_cpp::profile::now();
  MoeTraceCollector moe_trace(moe_trace_config_from_env());

  check_bf16_cpu(input, "input");
  TORCH_CHECK(input.dim() == 2, "input must be 2-D [tokens, hidden]");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(topk_ids.device().is_cpu(), "topk_ids must be CPU");
  TORCH_CHECK(topk_weights.device().is_cpu(), "topk_weights must be CPU");
  TORCH_CHECK(is_integer_dtype(topk_ids.scalar_type()), "topk_ids must use an integer dtype");
  TORCH_CHECK(is_floating_dtype(topk_weights.scalar_type()), "topk_weights must use a floating dtype");
  TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must be 2-D [tokens, top_k]");
  TORCH_CHECK(topk_weights.dim() == 2, "topk_weights must be 2-D [tokens, top_k]");
  TORCH_CHECK(topk_ids.sizes() == topk_weights.sizes(), "topk_ids and topk_weights shapes must match");
  TORCH_CHECK(topk_ids.size(0) == input.size(0), "topk first dimension must match input token count");
  TORCH_CHECK(topk_ids.size(1) > 0, "top_k must be non-zero");
  TORCH_CHECK(num_threads > 0, "num_threads must be positive, got ", num_threads);
  TORCH_CHECK(num_threads <= std::numeric_limits<int>::max(), "num_threads exceeds int32 limit: ", num_threads);

  const ::fused_cpp::moe::MoeBackend& backend = ::fused_cpp::moe::backend_from_id(gemm_backend);
  const bool use_sve_backend = backend.id == ::fused_cpp::moe::BackendId::kArmSveBf16;
  if (use_sve_backend) {
    TORCH_CHECK(fuse_silu, "SVE MoE backend currently requires fuse_silu=True");
  }
  TORCH_CHECK(backend_n_tile == backend.n_tile(), "MoE backend_n_tile mismatch for ", backend.name,
              ": weights use ", backend_n_tile, ", runtime uses ", backend.n_tile());

  PackedExperts w13 = checked_packed_experts(w13_packed, w13_K, w13_N, "w13_packed", backend_n_tile);
  PackedExperts w2 = checked_packed_experts(w2_packed, w2_K, w2_N, "w2_packed", backend_n_tile);
  TORCH_CHECK(w13.E == w2.E, "w13 and w2 expert count mismatch");
  TORCH_CHECK(w13.K == input.size(1), "input hidden size mismatch: input H=", input.size(1), ", w13 K=", w13.K);
  TORCH_CHECK(w13.N % 2 == 0, "w13 N must be even, got ", w13.N);
  const int64_t F = w13.N / 2;
  const int64_t H = input.size(1);
  TORCH_CHECK(w2.K == F && w2.N == H, "w2 shape mismatch: expected K=", F, " N=", H, ", got K=", w2.K, " N=", w2.N);
  check_optional_bias(w13_bias, w13.E, w13.N, "w13_bias");
  check_optional_bias(w2_bias, w2.E, w2.N, "w2_bias");
  // Decide once, up front, whether to use the fused-bias GEMM kernel:
  // a defined bias yields a padded [E, N_pad] fp32 buffer; otherwise the
  // base pointer stays null and the plain (non-bias) kernel is used.
  const at::Tensor w13_bias_f32 = build_padded_bias_f32(w13_bias, w13.E, w13.N, w13.N_pad);
  const at::Tensor w2_bias_f32 = build_padded_bias_f32(w2_bias, w2.E, w2.N, w2.N_pad);
  const float* w13_bias_base = w13_bias_f32.defined() ? w13_bias_f32.data_ptr<float>() : nullptr;
  const float* w2_bias_base = w2_bias_f32.defined() ? w2_bias_f32.data_ptr<float>() : nullptr;

  const int64_t num_tokens = input.size(0);
  const int64_t top_k = topk_ids.size(1);
  if (skip_weighted) {
    TORCH_CHECK(top_k == 1, "skip_weighted is only valid when top_k == 1");
  }
  if (num_tokens == 0) {
    return finalize_moe_output(prepare_moe_output(input, w13_packed, w2_packed, topk_weights, topk_ids, out), out);
  }

  const int64_t num_experts = global_num_experts < 0 ? w13.E : global_num_experts;
  TORCH_CHECK(num_experts > 0, "global_num_experts must be positive or -1, got ", global_num_experts);
  TORCH_CHECK(num_experts <= w13.E, "global_num_experts cannot exceed prepared expert weights: ", num_experts, " > ",
              w13.E);
  prepare_moe_threads_for_operator(num_threads);
  const int64_t actual_threads = num_threads;

  const bool stage_timing = env_flag_enabled("FUSED_CPP_MOE_STAGE_TIMING");
  const auto routing_cast_t0 = ::fused_cpp::profile::now();
  at::Tensor ids_i64 = topk_ids.to(at::kLong).contiguous();
  at::Tensor weights_f32 = topk_weights.to(at::kFloat).contiguous();
  const double routing_cast_ms = ::fused_cpp::profile::elapsed_ms(routing_cast_t0);
  const int64_t* ids = ids_i64.data_ptr<int64_t>();
  const int64_t num_routes = num_tokens * top_k;

  const auto route_build_t0 = ::fused_cpp::profile::now();
  std::vector<int64_t> route_counts(static_cast<size_t>(num_experts), 0);
  for (int64_t flat = 0; flat < num_routes; ++flat) {
    const int64_t expert = ids[flat];
    TORCH_CHECK(expert >= 0 && expert < num_experts, "topk_ids out of range: id=", expert, ", valid range [0, ",
                num_experts, ")");
    ++route_counts[static_cast<size_t>(expert)];
  }
  std::vector<std::vector<int64_t>> routes(static_cast<size_t>(num_experts));
  for (int64_t expert = 0; expert < num_experts; ++expert) {
    routes[static_cast<size_t>(expert)].reserve(static_cast<size_t>(route_counts[static_cast<size_t>(expert)]));
  }
  for (int64_t flat = 0; flat < num_routes; ++flat) {
    const int64_t expert = ids[flat];
    routes[static_cast<size_t>(expert)].push_back(flat);
  }
  const double route_build_ms = ::fused_cpp::profile::elapsed_ms(route_build_t0);

  std::vector<ExpertTask> tasks;
  tasks.reserve(static_cast<size_t>(num_experts));
  int64_t max_expert_rows = 0;
  for (int64_t e = 0; e < num_experts; ++e) {
    const auto& expert_routes = routes[static_cast<size_t>(e)];
    const int64_t rows = static_cast<int64_t>(expert_routes.size());
    if (rows > 0) {
      check_positive_int(rows, "expert task rows");
      tasks.push_back(ExpertTask{e, 0, rows});
      max_expert_rows = std::max(max_expert_rows, rows);
    }
  }

  at::Tensor output = prepare_moe_output(input, w13_packed, w2_packed, topk_weights, topk_ids, out);
  uint16_t* out_bf16_ptr = bf16_data(output);
  const uint16_t* input_ptr = bf16_data_const(input);
  const uint16_t* w13_ptr = bf16_data_const(w13.tensor);
  const uint16_t* w2_ptr = bf16_data_const(w2.tensor);

  const int schedule_debug_level = debug_schedule_level();
  const HierarchicalNSplitConfig nsplit_config = hierarchical_nsplit_config_from_env();
  // Fused SiLU-and-mul path (opt-in). Requires the interleaved w13 layout
  // from prepare_weights(fuse_silu=True) and activation == "silu". F % 8 == 0
  // makes the fused output stride (F) equal w2's K_pad, so the fused kernel
  // writes scratch.intermediate directly and the activation pass + gate_up
  // fp32 buffer are skipped. Only the default per-expert path is wired (Task
  // 6); hierarchical N-split is left on the legacy path for now.
  if (fuse_silu) {
    TORCH_CHECK(activation == "silu", "fuse_silu only supports activation='silu', got ", activation);
    TORCH_CHECK(F % 8 == 0, "fuse_silu requires F % 8 == 0, got F=", F);
    TORCH_CHECK(w13.N_pad == 2 * F, "fuse_silu expects interleaved w13 with N_pad=2F (=", 2 * F,
                "), got N_pad=", w13.N_pad);
    TORCH_CHECK(w13_bias_base == nullptr, "fuse_silu does not support w13_bias yet");
    TORCH_CHECK(!use_sve_backend || w2_bias_base == nullptr, "SVE fused MoE does not support w2_bias yet");
    TORCH_CHECK(silu_poly_degree == 4 || silu_poly_degree == 5 || silu_poly_degree == 6,
                "silu_poly_degree must be 4, 5, or 6, got ", silu_poly_degree);
  }
  const bool use_hierarchical_nsplit = nsplit_config.enabled;
  // Shared A pre-pack for the fused N-split path (opt-in, default OFF).
  // Benchmarks showed no benefit: the A-pack write is a one-time M*K store,
  // dwarfed by the repeated cached A-reads + compute, so de-duplicating it
  // changes nothing. Kept behind a flag for experimentation.
  bool fused_shared_apack = false;
  if (env_has_value("FUSED_CPP_MOE_FUSED_SHARED_APACK")) {
    fused_shared_apack = env_flag_enabled("FUSED_CPP_MOE_FUSED_SHARED_APACK");
  }
  // packA fusion: fold the A repack into the gather (w13, Part 1) and the
  // w13 epilogue store (w2, Part 2), so the GEMM kernels only compute and the
  // per-thread a_reorder scratch is no longer needed. Default ON; set 0 to
  // fall back to the row-major gather + in-kernel repack path.
  bool fused_packa = true;
  if (env_has_value("FUSED_CPP_MOE_FUSED_PACKA")) {
    fused_packa = env_flag_enabled("FUSED_CPP_MOE_FUSED_PACKA");
  }
  // The SVE backend is now the asm packed-A path. Keep the diagnostic
  // FUSED_CPP_MOE_FUSED_PACKA switch scoped to the NEON legacy backend so
  // gemm_backend=1 cannot fall back to the old SVE intrinsic GEMM path.
  if (use_sve_backend) {
    fused_packa = true;
  }
  // Part 2 (packed intermediate + packed-read w2) additionally requires w2 to
  // have no bias (the packed-read w2 kernel has no bias variant). When w2 has
  // bias, only Part 1 applies and w2 falls back to the row-major repack path.
  const bool fused_packa_w2 = fused_packa && (w2_bias_base == nullptr);
  bool use_w2_bf16_route = false;
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  use_w2_bf16_route = !skip_weighted && use_sve_backend && fuse_silu && fused_packa_w2 && sve_w2_bf16_route_enabled();
#endif
  bool use_w2_direct_route = false;
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  const int64_t route_element_bytes =
      use_w2_bf16_route ? static_cast<int64_t>(sizeof(uint16_t)) : static_cast<int64_t>(sizeof(float));
  use_w2_direct_route = !skip_weighted && use_sve_backend && fuse_silu && fused_packa_w2 && w2.N_pad == H &&
                        sve_w2_direct_route_offsets_fit(num_routes, H, w2.n_tile, route_element_bytes) &&
                        sve_w2_direct_route_enabled();
#endif
  at::Tensor route_out;
  float* route_out_ptr = nullptr;
  uint16_t* route_out_bf16_ptr = nullptr;
  if (!skip_weighted) {
    route_out =
        at::empty({num_routes, H},
                  at::TensorOptions().device(input.device()).dtype(use_w2_bf16_route ? at::kBFloat16 : at::kFloat));
    if (use_w2_bf16_route) {
      route_out_bf16_ptr = bf16_data(route_out);
    } else {
      route_out_ptr = route_out.data_ptr<float>();
    }
  }
  const bool fused_2d_split = env_flag_enabled("FUSED_CPP_MOE_FUSED_2D_SPLIT");
  const bool use_fused_2d_split = use_hierarchical_nsplit && fuse_silu && fused_packa_w2 && fused_2d_split;
  const bool use_w2_n_owner_scatter = use_sve_backend && fuse_silu && fused_packa_w2 &&
                                      env_flag_enabled_by_default("FUSED_CPP_MOE_SVE_W2_N_OWNER_SCATTER");
  const char* moe_trace_strategy =
      use_hierarchical_nsplit
          ? (use_fused_2d_split ? "hierarchical_fused_2d_split_dynamic_expert" : "hierarchical_mn_split_dynamic_expert")
          : "beam_calibrated_interference";
  int64_t moe_trace_expert_tasks = 0;
  for (const std::vector<int64_t>& expert_routes : routes) {
    if (!expert_routes.empty()) {
      ++moe_trace_expert_tasks;
    }
  }
  int64_t nsplit_total_groups = 0;
  int64_t nsplit_group_size = 0;
  std::vector<int64_t> nsplit_thread_cores;
  if (use_hierarchical_nsplit) {
    TORCH_CHECK(nsplit_config.partitions > 0, "FUSED_CPP_MOE_N_SPLIT_CORE_BASES must not be empty");
    TORCH_CHECK(nsplit_config.groups_per_partition > 0,
                "FUSED_CPP_MOE_N_SPLIT_GROUPS_PER_PARTITION must be "
                "positive, got ",
                nsplit_config.groups_per_partition);
    nsplit_total_groups = nsplit_config.partitions * nsplit_config.groups_per_partition;
    TORCH_CHECK(nsplit_total_groups > 0, "hierarchical N-split total group count must be positive");
    TORCH_CHECK(actual_threads >= nsplit_total_groups,
                "hierarchical N-split needs at least one thread per "
                "group: threads=",
                actual_threads, " groups=", nsplit_total_groups);
    TORCH_CHECK(actual_threads % nsplit_total_groups == 0,
                "hierarchical N-split requires equal-sized groups: "
                "threads=",
                actual_threads, " groups=", nsplit_total_groups);
    nsplit_group_size = actual_threads / nsplit_total_groups;
    nsplit_thread_cores.resize(static_cast<size_t>(actual_threads));
    for (int64_t tid = 0; tid < actual_threads; ++tid) {
      const int64_t group = tid / nsplit_group_size;
      const int64_t local_tid = tid % nsplit_group_size;
      const int64_t partition = group / nsplit_config.groups_per_partition;
      const int64_t group_in_partition = group % nsplit_config.groups_per_partition;
      nsplit_thread_cores[static_cast<size_t>(tid)] =
          nsplit_config.core_bases[static_cast<size_t>(partition)] + group_in_partition * nsplit_group_size + local_tid;
    }
  }

  if (!use_hierarchical_nsplit) {
    moe_trace.prepare_thread_buffers(actual_threads, tasks.size() * 2, 0);
    const ExpertScheduleWorkloadConfig workload_config{
        w13.K_pad, w13.N_pad, w2.K_pad, w2.N_pad, moe_activation_kind(activation), skip_weighted};
    float estimated_schedule_cost = 0.0f;
    std::vector<std::vector<TaskRange>> ranges =
        split_tasks_by_expert_affinity(tasks, actual_threads, workload_config, &estimated_schedule_cost);
    std::vector<ThreadScheduleDebug> schedule_debug;
    if (schedule_debug_level > 0) {
      schedule_debug.reserve(static_cast<size_t>(actual_threads));
      for (const std::vector<TaskRange>& thread_ranges : ranges) {
        schedule_debug.push_back(summarize_thread_schedule(tasks, thread_ranges));
      }
    }

    std::vector<ThreadScratch> scratches(static_cast<size_t>(actual_threads));
    for (ThreadScratch& scratch : scratches) {
      const int64_t max_k_pad = std::max(w13.K_pad, w2.K_pad);
      const int64_t max_fused_packa_rows = use_sve_backend ? sve_hybrid_packed_rows(max_expert_rows)
                                                           : ceil_to_multiple(max_expert_rows, int64_t{kKernelTile});
      const int64_t a_reorder_elems =
          use_sve_backend ? max_fused_packa_rows * max_k_pad : max_expert_rows * max_k_pad * 2;
      scratch.input.resize(static_cast<size_t>(max_expert_rows * w13.K_pad));
      scratch.intermediate.resize(
          static_cast<size_t>((use_sve_backend ? max_fused_packa_rows : max_expert_rows) * w2.K_pad));
      scratch.a_reorder.resize(static_cast<size_t>(a_reorder_elems));
      scratch.packed_a.resize(static_cast<size_t>(use_sve_backend ? max_fused_packa_rows * w13.K_pad : 0));
      scratch.gate_up.resize(static_cast<size_t>(max_expert_rows * w13.N_pad));
      const size_t down_elements =
          static_cast<size_t>((use_sve_backend ? max_fused_packa_rows : max_expert_rows) * w2.N_pad);
      scratch.down.resize(use_w2_bf16_route || use_w2_direct_route ? 0 : down_elements);
      scratch.down_bf16.resize(use_w2_bf16_route && !use_w2_direct_route ? down_elements : 0);
    }

    const auto schedule_compute_begin = ::fused_cpp::profile::now();
    run_fixed_threads(actual_threads, [&](int64_t tid) {
      const auto thread_begin = ::fused_cpp::profile::now();
      ThreadScratch& scratch = scratches[static_cast<size_t>(tid)];
      const std::vector<TaskRange>& thread_ranges = ranges[static_cast<size_t>(tid)];
      for (const TaskRange& range : thread_ranges) {
        for (size_t task_idx = range.begin; task_idx < range.end; ++task_idx) {
          const ExpertTask& task = tasks[task_idx];
          const auto& expert_routes = routes[static_cast<size_t>(task.expert)];
          const int64_t rows = task.rows;

          if (use_sve_backend && fuse_silu) {
            gather_pack_a_reorder_sve_hybrid(input_ptr, H, expert_routes.data() + task.route_begin, top_k,
                                             scratch.packed_a.data(), static_cast<int>(rows),
                                             static_cast<int>(w13.K_pad), int64_t{1}, int64_t{0});
          } else {
            std::fill(scratch.input.begin(), scratch.input.begin() + rows * w13.K_pad, static_cast<uint16_t>(0));
            for (int64_t m = 0; m < rows; ++m) {
              const int64_t flat = expert_routes[static_cast<size_t>(task.route_begin + m)];
              const int64_t token = flat / top_k;
              const uint16_t* src = input_ptr + token * H;
              uint16_t* dst = scratch.input.data() + m * w13.K_pad;
              std::copy(src, src + H, dst);
            }
          }

          if (fuse_silu) {
            // Fused w13 + SiLU-and-mul: interleaved w13 -> bf16
            // intermediate directly (skips gate_up + activation).
            // ldc = w2.K_pad (== F, since F % 8 == 0), so the
            // output stride matches w2's A stride exactly.
            if (use_sve_backend) {
              TeamContext team;
              team.group_size = 1;
              team.local_tid = 0;
              team.barrier = nullptr;
              team.a_reorder = nullptr;
              team_fused_w13_silu_packed_packc_sve(
                  team, scratch.packed_a.data(), w13_ptr + task.expert * w13.packed_stride, scratch.intermediate.data(),
                  static_cast<int>(rows), static_cast<int>(w13.K_pad), static_cast<int>(w13.N_pad),
                  static_cast<int>(w2.K_pad), silu_poly_degree, w13.n_tile);
            } else {
              single_thread_gemm_fused_silu(scratch.input.data(), w13_ptr + task.expert * w13.packed_stride,
                                            scratch.intermediate.data(), scratch.a_reorder.data(),
                                            static_cast<int>(rows), static_cast<int>(w13.K_pad),
                                            static_cast<int>(w13.N_pad), static_cast<int>(w2.K_pad), silu_poly_degree);
            }
          } else {
            trace_dispatch_fp32_gemm(moe_trace, "w13", tid, -1, -1, -1, task.expert, task.route_begin, rows,
                                     scratch.input.data(), w13_ptr + task.expert * w13.packed_stride,
                                     scratch.gate_up.data(), scratch.a_reorder.data(), static_cast<int>(rows),
                                     static_cast<int>(w13.K_pad), static_cast<int>(w13.N_pad),
                                     static_cast<int>(w13.N_pad),
                                     w13_bias_base != nullptr ? w13_bias_base + task.expert * w13.N_pad : nullptr);

            activation_to_bf16(activation, scratch.gate_up.data(), scratch.intermediate.data(), rows, w13.N_pad,
                               w2.K_pad, F);
          }

          if (use_sve_backend) {
            TORCH_CHECK(w2_bias_base == nullptr,
                        "SVE fused MoE w2 path does not support "
                        "w2_bias yet");
            TeamContext team;
            team.group_size = 1;
            team.local_tid = 0;
            team.barrier = nullptr;
            team.a_reorder = nullptr;
            if (use_w2_direct_route) {
              if (use_w2_bf16_route) {
                team_w2_packed_sve_direct_bf16_route(
                    team, scratch.intermediate.data(), w2_ptr + task.expert * w2.packed_stride, route_out_bf16_ptr,
                    expert_routes.data() + task.route_begin, static_cast<int>(rows), static_cast<int>(w2.K_pad),
                    static_cast<int>(w2.N_pad), static_cast<int>(H), w2.n_tile);
              } else {
                team_w2_packed_sve_direct_route(
                    team, scratch.intermediate.data(), w2_ptr + task.expert * w2.packed_stride, route_out_ptr,
                    expert_routes.data() + task.route_begin, static_cast<int>(rows), static_cast<int>(w2.K_pad),
                    static_cast<int>(w2.N_pad), static_cast<int>(H), w2.n_tile);
              }
            } else if (use_w2_bf16_route) {
              team_w2_packed_bf16_sve(team, scratch.intermediate.data(), w2_ptr + task.expert * w2.packed_stride,
                                      scratch.down_bf16.data(), static_cast<int>(rows), static_cast<int>(w2.K_pad),
                                      static_cast<int>(w2.N_pad), static_cast<int>(w2.N_pad), w2.n_tile);
            } else {
              team_w2_packed_sve(team, scratch.intermediate.data(), w2_ptr + task.expert * w2.packed_stride,
                                 scratch.down.data(), static_cast<int>(rows), static_cast<int>(w2.K_pad),
                                 static_cast<int>(w2.N_pad), static_cast<int>(w2.N_pad), w2.n_tile);
            }
          } else {
            trace_dispatch_fp32_gemm(moe_trace, "w2", tid, -1, -1, -1, task.expert, task.route_begin, rows,
                                     scratch.intermediate.data(), w2_ptr + task.expert * w2.packed_stride,
                                     scratch.down.data(), scratch.a_reorder.data(), static_cast<int>(rows),
                                     static_cast<int>(w2.K_pad), static_cast<int>(w2.N_pad), static_cast<int>(w2.N_pad),
                                     w2_bias_base != nullptr ? w2_bias_base + task.expert * w2.N_pad : nullptr);
          }

          if (!use_w2_direct_route) {
            for (int64_t m = 0; m < rows; ++m) {
              const int64_t flat = expert_routes[static_cast<size_t>(task.route_begin + m)];
              if (use_w2_bf16_route) {
                const uint16_t* src = scratch.down_bf16.data() + m * w2.N_pad;
                uint16_t* dst = route_out_bf16_ptr + flat * H;
                std::copy(src, src + H, dst);
              } else if (skip_weighted) {
                const float* src = scratch.down.data() + m * w2.N_pad;
                uint16_t* dst = out_bf16_ptr + flat * H;
                convert_f32_to_bf16(src, dst, H);
              } else {
                const float* src = scratch.down.data() + m * w2.N_pad;
                float* dst = route_out_ptr + flat * H;
                std::copy(src, src + H, dst);
              }
            }
          }
        }
      }
      if (schedule_debug_level > 0) {
        schedule_debug[static_cast<size_t>(tid)].ms = ::fused_cpp::profile::elapsed_ms(thread_begin);
      }
    });
    const double observed_schedule_ms = ::fused_cpp::profile::elapsed_ms(schedule_compute_begin);
    update_expert_schedule_cost_feedback(estimated_schedule_cost, observed_schedule_ms);

    if (stage_timing) {
      std::fprintf(stderr,
                   "[fused_moe_bf16_tiled][stage_timing] threads=%lld "
                   "routing_cast_ms=%.3f route_build_ms=%.3f "
                   "schedule_compute_ms=%.3f\n",
                   static_cast<long long>(actual_threads), routing_cast_ms, route_build_ms, observed_schedule_ms);
    }

    if (schedule_debug_level > 0) {
      int64_t longest_tid = -1;
      int64_t shortest_tid = -1;
      for (int64_t tid = 0; tid < actual_threads; ++tid) {
        const ThreadScheduleDebug& debug = schedule_debug[static_cast<size_t>(tid)];
        if (debug.rows == 0) {
          continue;
        }
        if (longest_tid < 0 || debug.ms > schedule_debug[static_cast<size_t>(longest_tid)].ms) {
          longest_tid = tid;
        }
        if (shortest_tid < 0 || debug.ms < schedule_debug[static_cast<size_t>(shortest_tid)].ms) {
          shortest_tid = tid;
        }
      }
      std::fprintf(stderr,
                   "[fused_moe_bf16_tiled][schedule] threads=%lld experts=%lld "
                   "routes=%lld expert_tasks=%zu "
                   "strategy=beam_calibrated_interference\n",
                   static_cast<long long>(actual_threads), static_cast<long long>(num_experts),
                   static_cast<long long>(num_routes), tasks.size());
      if (longest_tid >= 0) {
        print_schedule_debug_line("longest", longest_tid, schedule_debug[static_cast<size_t>(longest_tid)]);
      }
      if (shortest_tid >= 0) {
        print_schedule_debug_line("shortest", shortest_tid, schedule_debug[static_cast<size_t>(shortest_tid)]);
      }
      if (schedule_debug_level >= 2) {
        for (int64_t tid = 0; tid < actual_threads; ++tid) {
          print_schedule_debug_line("thread", tid, schedule_debug[static_cast<size_t>(tid)]);
        }
      }
    }
  } else {
    std::vector<int64_t> expert_order;
    expert_order.reserve(static_cast<size_t>(num_experts));
    for (int64_t expert = 0; expert < num_experts; ++expert) {
      if (!routes[static_cast<size_t>(expert)].empty()) {
        expert_order.push_back(expert);
      }
    }
    std::sort(expert_order.begin(), expert_order.end(), [&](int64_t lhs, int64_t rhs) {
      const size_t lhs_rows = routes[static_cast<size_t>(lhs)].size();
      const size_t rhs_rows = routes[static_cast<size_t>(rhs)].size();
      if (lhs_rows != rhs_rows) {
        return lhs_rows > rhs_rows;
      }
      return lhs < rhs;
    });
    moe_trace_expert_tasks = static_cast<int64_t>(expert_order.size());
    moe_trace.prepare_thread_buffers(
        actual_threads, expert_order.size() * static_cast<size_t>(nsplit_group_size) * 2, 0);
    std::atomic<size_t> next_expert_idx{0};
    std::vector<ThreadScheduleDebug> schedule_debug(static_cast<size_t>(nsplit_total_groups));

    int64_t max_expert_rows = 0;
    for (int64_t expert : expert_order) {
      max_expert_rows =
          std::max<int64_t>(max_expert_rows, static_cast<int64_t>(routes[static_cast<size_t>(expert)].size()));
    }
    const int64_t max_fused_packa_rows =
        use_sve_backend ? sve_hybrid_packed_rows(max_expert_rows) : ceil_to_multiple(max_expert_rows, int64_t{8});
    const int64_t max_k_pad = std::max(w13.K_pad, w2.K_pad);
    const int64_t a_reorder_stride =
        use_sve_backend ? max_fused_packa_rows * max_k_pad : max_expert_rows * max_k_pad * 2;
    double stage_alloc_ms = 0.0;
    // packA fusion frees the per-thread A-reorder scratch (the dominant
    // per-call alloc, ~268MB at G=8): w13 reads packed_a (Part 1) and, with
    // Part 2, w2 reads the packed intermediate — neither repacks. Also skip
    // the row-major input (gather writes packed_a directly) and gate_up
    // (only used by the non-fused activation path).
    const bool packa_skip_reorder = fuse_silu && fused_packa_w2;
    const bool packa_skip_input = fuse_silu && fused_packa;
    const bool packa_skip_gate_up = fuse_silu;
    const auto stage_alloc_t0 = ::fused_cpp::profile::now();
    // Per-calling-thread persistent pool: grow-only, reused across calls,
    // so mmap + first-touch faults happen once (see HierarchicalScratchPool).
    // Backend (malloc / THP / hugetlb) is chosen by backend_allocator.
    thread_local HierarchicalScratchPool scratch_pool;
    if (scratch_pool.group_size != nsplit_group_size) {
      scratch_pool.groups.clear();
      scratch_pool.group_size = nsplit_group_size;
    }
    while (static_cast<int64_t>(scratch_pool.groups.size()) < nsplit_total_groups) {
      scratch_pool.groups.push_back(std::make_unique<HierarchicalGroupScratch>(nsplit_group_size));
    }
    for (int64_t group = 0; group < nsplit_total_groups; ++group) {
      HierarchicalGroupScratch& sc = *scratch_pool.groups[group];
      HierarchicalScratchPool::ensure(sc.input,
                                      static_cast<size_t>(packa_skip_input ? 0 : max_expert_rows * w13.K_pad));
      const size_t interm_need = static_cast<size_t>(max_fused_packa_rows * w2.K_pad);
      HierarchicalScratchPool::ensure(sc.intermediate, interm_need);
      // w2 reads intermediate's padding feature-blocks, which must be
      // zero. With a pool the buffer may be reused across differently
      // shaped calls, so re-zero the used region each call (pages are
      // warm -> no faults; ~2MB). packed_a/down are fully overwritten and
      // need no zeroing.
      std::memset(sc.intermediate.data(), 0, interm_need * sizeof(uint16_t));
      HierarchicalScratchPool::ensure(sc.packed_a, static_cast<size_t>(max_fused_packa_rows * w13.K_pad));
      HierarchicalScratchPool::ensure(
          sc.a_reorder, static_cast<size_t>(packa_skip_reorder ? 0 : nsplit_group_size * a_reorder_stride));
      HierarchicalScratchPool::ensure(sc.gate_up,
                                      static_cast<size_t>(packa_skip_gate_up ? 0 : max_expert_rows * w13.N_pad));
      const size_t down_need = static_cast<size_t>(max_fused_packa_rows * w2.N_pad);
      if (use_w2_direct_route) {
        continue;
      }
      if (use_w2_bf16_route) {
        HierarchicalScratchPool::ensure(sc.down_bf16, down_need);
      } else {
        HierarchicalScratchPool::ensure(sc.down, down_need);
      }
    }
    std::vector<std::unique_ptr<HierarchicalGroupScratch>>& group_scratches = scratch_pool.groups;
    stage_alloc_ms = ::fused_cpp::profile::elapsed_ms(stage_alloc_t0);

    ThreadPinningConfig nsplit_pinning;
    nsplit_pinning.enabled = true;
    nsplit_pinning.cpus = nsplit_thread_cores;
    ThreadPinningScope nsplit_scope(&nsplit_pinning);
    double phase_gather_ms = 0, phase_w13_ms = 0, phase_w2_ms = 0,
           phase_scatter_ms = 0;  // group 0 / tid 0 only, under stage_timing
    run_fixed_threads(actual_threads, [&](int64_t tid) {
      const int64_t group = tid / nsplit_group_size;
      const int64_t local_tid = tid % nsplit_group_size;
      const auto thread_begin = ::fused_cpp::profile::now();
      const bool ph_on = stage_timing && group == 0 && local_tid == 0;
      auto time_phase = [&](double& acc, auto&& fn) {
        if (!ph_on) {
          fn();
          return;
        }
        const auto _t = ::fused_cpp::profile::now();
        fn();
        acc += ::fused_cpp::profile::elapsed_ms(_t);
      };
      HierarchicalGroupScratch& scratch = *group_scratches[static_cast<size_t>(group)];
      ThreadBarrier& barrier = scratch.barrier;
      uint16_t* a_reorder =
          scratch.a_reorder.empty() ? nullptr : scratch.a_reorder.data() + local_tid * a_reorder_stride;

      while (true) {
        if (local_tid == 0) {
          const size_t order_idx = next_expert_idx.fetch_add(size_t{1}, std::memory_order_relaxed);
          const int64_t expert = order_idx < expert_order.size() ? expert_order[order_idx] : -1;
          scratch.current_expert.store(expert, std::memory_order_release);
          if (schedule_debug_level > 0 && expert >= 0) {
            const int64_t expert_rows = static_cast<int64_t>(routes[static_cast<size_t>(expert)].size());
            ThreadScheduleDebug& debug = schedule_debug[static_cast<size_t>(group)];
            debug.rows += expert_rows;
            ++debug.tasks;
            ++debug.ranges;
            debug.experts.push_back(expert);
            debug.expert_rows.push_back(expert_rows);
          }
        }
        barrier.wait();

        const int64_t expert = scratch.current_expert.load(std::memory_order_acquire);
        if (expert < 0) {
          break;
        }

        const auto& expert_routes = routes[static_cast<size_t>(expert)];
        const int64_t rows = static_cast<int64_t>(expert_routes.size());

        if (fuse_silu && fused_packa) {
          // Fused gather + m8 reorder pack (Part 1): write the w13 A
          // directly into the packed layout, skipping the row-major
          // scratch.input round-trip and the in-kernel repack. Split
          // by 8-row blocks so each member packs whole blocks.
          const int64_t nb = (rows + 7) / 8;
          const SplitRange brange = split_evenly(nb, nsplit_group_size, local_tid);
          time_phase(phase_gather_ms, [&] {
            if (use_sve_backend) {
              gather_pack_a_reorder_sve_hybrid(input_ptr, H, expert_routes.data(), top_k, scratch.packed_a.data(),
                                               static_cast<int>(rows), static_cast<int>(w13.K_pad), nsplit_group_size,
                                               local_tid);
            } else {
              gather_pack_a_reorder_m8(input_ptr, H, expert_routes.data(), top_k, scratch.packed_a.data(),
                                       static_cast<int>(rows), static_cast<int>(w13.K_pad),
                                       static_cast<int>(brange.begin), static_cast<int>(brange.begin + brange.size));
            }
          });
        } else {
          // Parallel gather: each team thread copies its own row
          // slice (was serial on local_tid==0 with the rest idle).
          const SplitRange grange = split_evenly(rows, nsplit_group_size, local_tid);
          for (int64_t m = grange.begin; m < grange.begin + grange.size; ++m) {
            const int64_t flat = expert_routes[static_cast<size_t>(m)];
            const int64_t token = flat / top_k;
            uint16_t* dst = scratch.input.data() + m * w13.K_pad;
            std::fill(dst, dst + w13.K_pad, static_cast<uint16_t>(0));
            std::copy(input_ptr + token * H, input_ptr + token * H + H, dst);
          }
        }
        barrier.wait();

        if (fuse_silu) {
          // N-split fused w13 + SiLU-and-mul: each member writes its
          // disjoint feature-column slice of intermediate directly
          // (bf16, stride w2.K_pad). Collapses the w13 GEMM and the
          // activation pass into one stage, dropping a barrier and
          // the gate_up fp32 buffer.
          TeamContext team;
          team.group_size = nsplit_group_size;
          team.local_tid = local_tid;
          team.barrier = nsplit_group_size > 1 ? &barrier : nullptr;
          team.a_reorder = a_reorder;
          if (fused_packa) {
            // A already packed by the fused gather+pack stage; run
            // the packed-read fused kernel on the N-slice, no repack.
            // With Part 2 (fused_packa_w2) the epilogue writes the
            // intermediate directly in w2 pre-packed layout.
            const int64_t nb = (rows + 7) / 8;
            if (fused_packa_w2) {
              const Gemm2DSplitPlan w13_2d_plan =
                  plan_2d_gemm_split(rows, w13.K_pad, w13.N_pad, nsplit_group_size, w13.n_tile);
              time_phase(phase_w13_ms, [&] {
                if (use_sve_backend) {
                  if (use_fused_2d_split) {
                    team_fused_w13_silu_packed_packc_sve_2d(
                        team, w13_2d_plan, scratch.packed_a.data(), w13_ptr + expert * w13.packed_stride,
                        scratch.intermediate.data(), static_cast<int>(rows), static_cast<int>(w13.K_pad),
                        static_cast<int>(w13.N_pad), static_cast<int>(w2.K_pad), silu_poly_degree,
                        w13.n_tile);
                  } else {
                    team_fused_w13_silu_packed_packc_sve(
                        team, scratch.packed_a.data(), w13_ptr + expert * w13.packed_stride,
                        scratch.intermediate.data(), static_cast<int>(rows), static_cast<int>(w13.K_pad),
                        static_cast<int>(w13.N_pad), static_cast<int>(w2.K_pad), silu_poly_degree, w13.n_tile);
                  }
                } else {
                  if (use_fused_2d_split) {
                    team_fused_w13_silu_packed_packc_2d(
                        team, w13_2d_plan, scratch.packed_a.data(), w13_ptr + expert * w13.packed_stride,
                        scratch.intermediate.data(), static_cast<int>(rows), static_cast<int>(w13.K_pad),
                        static_cast<int>(w13.N_pad), static_cast<int>(w2.K_pad), silu_poly_degree);
                  } else {
                    team_fused_w13_silu_packed_packc(
                        team, scratch.packed_a.data(), w13_ptr + expert * w13.packed_stride,
                        scratch.intermediate.data(), static_cast<int>(rows), static_cast<int>(w13.K_pad),
                        static_cast<int>(w13.N_pad), static_cast<int>(w2.K_pad), silu_poly_degree);
                  }
                }
              });
            } else {
              team_fused_w13_silu_packed(team, scratch.packed_a.data(), w13_ptr + expert * w13.packed_stride,
                                         scratch.intermediate.data(), static_cast<int>(nb * 8),
                                         static_cast<int>(w13.K_pad), static_cast<int>(w13.N_pad),
                                         static_cast<int>(w2.K_pad), silu_poly_degree);
            }
            barrier.wait();
          } else if (fused_shared_apack) {
            // Pack the w13 A once into the group-shared buffer
            // (each member packs its 8-row block range), then all
            // members read it for their N-slice: no G-fold repack.
            const int64_t nb = (rows + 7) / 8;
            const SplitRange brange = split_evenly(nb, nsplit_group_size, local_tid);
            pack_a_reorder_m8(scratch.input.data(), scratch.packed_a.data(), static_cast<int>(rows),
                              static_cast<int>(w13.K_pad), static_cast<int>(brange.begin),
                              static_cast<int>(brange.begin + brange.size));
            barrier.wait();
            team_fused_w13_silu_packed(team, scratch.packed_a.data(), w13_ptr + expert * w13.packed_stride,
                                       scratch.intermediate.data(), static_cast<int>(nb * 8),
                                       static_cast<int>(w13.K_pad), static_cast<int>(w13.N_pad),
                                       static_cast<int>(w2.K_pad), silu_poly_degree);
            barrier.wait();
          } else {
            if (use_sve_backend) {
              team_fused_w13_silu_sve(team, scratch.input.data(), w13_ptr + expert * w13.packed_stride,
                                      scratch.intermediate.data(), static_cast<int>(rows), static_cast<int>(w13.K_pad),
                                      static_cast<int>(w13.N_pad), static_cast<int>(w2.K_pad), a_reorder,
                                      silu_poly_degree, w13.n_tile);
            } else {
              team_fused_w13_silu(team, scratch.input.data(), w13_ptr + expert * w13.packed_stride,
                                  scratch.intermediate.data(), static_cast<int>(rows), static_cast<int>(w13.K_pad),
                                  static_cast<int>(w13.N_pad), static_cast<int>(w2.K_pad), silu_poly_degree);
            }
            barrier.wait();
          }
        } else {
          trace_dispatch_fp32_gemm_stage_split(
              moe_trace, "w13", MoeGemmStage::kW13, tid, -1, group, local_tid, expert, 0, rows, scratch.input.data(),
              w13_ptr + expert * w13.packed_stride, scratch.gate_up.data(), a_reorder, static_cast<int>(rows),
              static_cast<int>(w13.K_pad), static_cast<int>(w13.N_pad), static_cast<int>(w13.N_pad), nsplit_group_size,
              w13_bias_base != nullptr ? w13_bias_base + expert * w13.N_pad : nullptr);
          barrier.wait();

          const SplitRange activation_range = split_evenly(rows, nsplit_group_size, local_tid);
          activation_range_to_bf16(activation, scratch.gate_up.data(), scratch.intermediate.data(),
                                   activation_range.begin, activation_range.size, w13.N_pad, w2.K_pad, F);
          barrier.wait();
        }

        TeamContext w2team;
        w2team.group_size = nsplit_group_size;
        w2team.local_tid = local_tid;
        w2team.barrier = nullptr;
        w2team.a_reorder = a_reorder;
        if (fused_packa_w2) {
          // w2 reads the PACKED intermediate directly (no repack);
          // N-split over w2.N_pad. M padded to a multiple of 8.
          const Gemm2DSplitPlan w2_2d_plan = plan_2d_gemm_split(rows, w2.K_pad, w2.N_pad, nsplit_group_size, w2.n_tile);
          time_phase(phase_w2_ms, [&] {
            if (use_w2_direct_route) {
              if (use_w2_bf16_route) {
                team_w2_packed_sve_direct_bf16_route_backend(
                    use_fused_2d_split, w2team, w2_2d_plan, scratch.intermediate.data(),
                    w2_ptr + expert * w2.packed_stride, route_out_bf16_ptr, expert_routes.data(),
                    static_cast<int>(rows), static_cast<int>(w2.K_pad), static_cast<int>(w2.N_pad),
                    static_cast<int>(H), w2.n_tile);
              } else {
                team_w2_packed_sve_direct_route_backend(
                    use_fused_2d_split, w2team, w2_2d_plan, scratch.intermediate.data(),
                    w2_ptr + expert * w2.packed_stride, route_out_ptr, expert_routes.data(), static_cast<int>(rows),
                    static_cast<int>(w2.K_pad), static_cast<int>(w2.N_pad), static_cast<int>(H), w2.n_tile);
              }
            } else if (use_w2_bf16_route) {
              team_w2_packed_bf16_sve_backend(use_fused_2d_split, w2team, w2_2d_plan, scratch.intermediate.data(),
                                              w2_ptr + expert * w2.packed_stride, scratch.down_bf16.data(),
                                              static_cast<int>(rows), static_cast<int>(w2.K_pad),
                                              static_cast<int>(w2.N_pad), static_cast<int>(w2.N_pad), w2.n_tile);
            } else if (use_sve_backend) {
              if (use_fused_2d_split) {
                team_w2_packed_sve_2d(w2team, w2_2d_plan, scratch.intermediate.data(),
                                      w2_ptr + expert * w2.packed_stride, scratch.down.data(), static_cast<int>(rows),
                                      static_cast<int>(w2.K_pad), static_cast<int>(w2.N_pad),
                                      static_cast<int>(w2.N_pad));
              } else {
                team_w2_packed_sve(w2team, scratch.intermediate.data(), w2_ptr + expert * w2.packed_stride,
                                   scratch.down.data(), static_cast<int>(rows), static_cast<int>(w2.K_pad),
                                   static_cast<int>(w2.N_pad), static_cast<int>(w2.N_pad), w2.n_tile);
              }
            } else {
              if (use_fused_2d_split) {
                team_w2_packed_2d(w2team, w2_2d_plan, scratch.intermediate.data(), w2_ptr + expert * w2.packed_stride,
                                  scratch.down.data(), static_cast<int>(rows), static_cast<int>(w2.K_pad),
                                  static_cast<int>(w2.N_pad), static_cast<int>(w2.N_pad));
              } else {
                team_w2_packed(w2team, scratch.intermediate.data(), w2_ptr + expert * w2.packed_stride,
                               scratch.down.data(), static_cast<int>(rows), static_cast<int>(w2.K_pad),
                               static_cast<int>(w2.N_pad), static_cast<int>(w2.N_pad));
              }
            }
          });
        } else {
          if (use_sve_backend) {
            team_w2_rowmajor_sve(w2team, scratch.intermediate.data(), w2_ptr + expert * w2.packed_stride,
                                 scratch.down.data(), static_cast<int>(rows), static_cast<int>(w2.K_pad),
                                 static_cast<int>(w2.N_pad), static_cast<int>(w2.N_pad), a_reorder, w2.n_tile);
          } else {
            trace_dispatch_fp32_gemm_stage_split(
                moe_trace, "w2", MoeGemmStage::kW2, tid, -1, group, local_tid, expert, 0, rows,
                scratch.intermediate.data(), w2_ptr + expert * w2.packed_stride, scratch.down.data(), a_reorder,
                static_cast<int>(rows), static_cast<int>(w2.K_pad), static_cast<int>(w2.N_pad),
                static_cast<int>(w2.N_pad), nsplit_group_size,
                w2_bias_base != nullptr ? w2_bias_base + expert * w2.N_pad : nullptr);
          }
        }
        if (!use_w2_direct_route && !use_w2_n_owner_scatter) {
          barrier.wait();
        }

        if (!use_w2_direct_route) {
          time_phase(phase_scatter_ms, [&] {
            with_w2_scatter_owner(static_cast<int>(w2.N_pad), nsplit_group_size, local_tid, w2.n_tile,
                                  use_w2_n_owner_scatter, [&](const SplitRange& h_range) {
                                        const int64_t h_begin = h_range.begin;
                                        const int64_t h_end = std::min<int64_t>(H, h_begin + h_range.size);
                                        if (h_begin >= h_end) {
                                          return;
                                        }
                                        for (int64_t m = 0; m < rows; ++m) {
                                          const int64_t flat = expert_routes[static_cast<size_t>(m)];
                                          if (use_w2_bf16_route) {
                                            const uint16_t* src = scratch.down_bf16.data() + m * w2.N_pad;
                                            uint16_t* dst = route_out_bf16_ptr + flat * H;
                                            std::copy(src + h_begin, src + h_end, dst + h_begin);
                                          } else if (skip_weighted) {
                                            const float* src = scratch.down.data() + m * w2.N_pad;
                                            uint16_t* dst = out_bf16_ptr + flat * H;
                                            convert_f32_to_bf16(src + h_begin, dst + h_begin, h_end - h_begin);
                                          } else {
                                            const float* src = scratch.down.data() + m * w2.N_pad;
                                            float* dst = route_out_ptr + flat * H;
                                            std::copy(src + h_begin, src + h_end, dst + h_begin);
                                          }
                                        }
                                      });
          });
        }
        barrier.wait();
      }

      if (schedule_debug_level > 0 && local_tid == 0) {
        schedule_debug[static_cast<size_t>(group)].ms = ::fused_cpp::profile::elapsed_ms(thread_begin);
      }
    });

    if (stage_timing) {
      std::fprintf(stderr,
                   "[fused_moe_bf16_tiled][stage_timing] threads=%lld "
                   "routing_cast_ms=%.3f route_build_ms=%.3f "
                   "scratch_alloc_ms=%.3f gather_ms=%.3f w13_ms=%.3f "
                   "w2_ms=%.3f scatter_ms=%.3f "
                   "(gather/w13/w2/scatter are group0/tid0; exclude barrier waits)\n",
                   static_cast<long long>(actual_threads), routing_cast_ms, route_build_ms, stage_alloc_ms,
                   phase_gather_ms, phase_w13_ms, phase_w2_ms, phase_scatter_ms);
    }
    if (schedule_debug_level > 0) {
      int64_t longest_group = -1;
      int64_t shortest_group = -1;
      for (int64_t group = 0; group < nsplit_total_groups; ++group) {
        const ThreadScheduleDebug& debug = schedule_debug[static_cast<size_t>(group)];
        if (debug.rows == 0) {
          continue;
        }
        if (longest_group < 0 || debug.ms > schedule_debug[static_cast<size_t>(longest_group)].ms) {
          longest_group = group;
        }
        if (shortest_group < 0 || debug.ms < schedule_debug[static_cast<size_t>(shortest_group)].ms) {
          shortest_group = group;
        }
      }
      std::fprintf(stderr,
                   "[fused_moe_bf16_tiled][schedule] threads=%lld experts=%lld "
                   "routes=%lld expert_tasks=%zu schedule_units=%zu "
                   "strategy=hierarchical_mn_split_dynamic_expert "
                   "partitions=%lld groups_per_partition=%lld groups=%lld "
                   "group_size=%lld core_bases=[",
                   static_cast<long long>(actual_threads), static_cast<long long>(num_experts),
                   static_cast<long long>(num_routes), expert_order.size(), tasks.size(),
                   static_cast<long long>(nsplit_config.partitions),
                   static_cast<long long>(nsplit_config.groups_per_partition),
                   static_cast<long long>(nsplit_total_groups), static_cast<long long>(nsplit_group_size));
      for (size_t i = 0; i < nsplit_config.core_bases.size(); ++i) {
        if (i != 0) {
          std::fprintf(stderr, ",");
        }
        std::fprintf(stderr, "%lld", static_cast<long long>(nsplit_config.core_bases[i]));
      }
      std::fprintf(stderr, "]\n");
      if (longest_group >= 0) {
        print_schedule_debug_line("longest_group", longest_group, schedule_debug[static_cast<size_t>(longest_group)]);
      }
      if (shortest_group >= 0) {
        print_schedule_debug_line("shortest_group", shortest_group,
                                  schedule_debug[static_cast<size_t>(shortest_group)]);
      }
      if (schedule_debug_level >= 2) {
        for (int64_t group = 0; group < nsplit_total_groups; ++group) {
          print_schedule_debug_line("group", group, schedule_debug[static_cast<size_t>(group)]);
        }
      }
    }
  }

  // 2a/2b: the weighted merge accumulates per token in a thread-local fp32
  // buffer and writes the bf16 result straight into `output`. When
  // skip_weighted is set the scatter already wrote `output` directly, so the
  // merge is skipped entirely.
  const float* topk_w = weights_f32.data_ptr<float>();
  const int route_merge_unroll = skip_weighted ? 0 : resolve_route_merge_unroll(use_sve_backend);

  auto merge_routes = [&](int64_t tid) {
    const int64_t rows_per_thread = ceil_div_int64(num_tokens, actual_threads);
    const int64_t token_begin = tid * rows_per_thread;
    const int64_t token_end = std::min<int64_t>(num_tokens, token_begin + rows_per_thread);
    merge_route_range(route_out_ptr, route_out_bf16_ptr, topk_w, out_bf16_ptr, token_begin, token_end, top_k, H,
                      use_w2_bf16_route, route_merge_unroll);
  };
  const auto merge_t0 = ::fused_cpp::profile::now();
  if (!skip_weighted) {
    if (use_hierarchical_nsplit) {
      ThreadPinningConfig nsplit_pinning;
      nsplit_pinning.enabled = true;
      nsplit_pinning.cpus = nsplit_thread_cores;
      ThreadPinningScope nsplit_scope(&nsplit_pinning);
      run_fixed_threads(actual_threads, merge_routes);
    } else {
      run_fixed_threads(actual_threads, merge_routes);
    }
  }
  if (stage_timing && !skip_weighted) {
    std::fprintf(stderr,
                 "[fused_moe_bf16_tiled][stage_timing] merge_routes_ms=%.3f "
                 "(top_k=%lld weighted reduce+bf16, all threads)\n",
                 ::fused_cpp::profile::elapsed_ms(merge_t0), static_cast<long long>(top_k));
  }

  if (moe_trace.enabled()) {
    const double e2e_ms = ::fused_cpp::profile::elapsed_ms(moe_trace_begin);
    moe_trace.write_report(moe_trace_strategy, actual_threads, num_tokens, top_k, num_experts, num_routes, H, F,
                           tasks.size(), moe_trace_expert_tasks, nsplit_total_groups, nsplit_group_size, e2e_ms);
  }
  return finalize_moe_output(output, out);
#endif
}

at::Tensor fused_moe_bf16_tiled_scheduled(at::Tensor input, at::Tensor w13_packed, int64_t w13_K, int64_t w13_N,
                                          at::Tensor w2_packed, int64_t w2_K, int64_t w2_N, at::Tensor topk_weights,
                                          at::Tensor topk_ids, at::Tensor wave_offsets, at::Tensor team_expert_ids,
                                          at::Tensor team_threads, c10::optional<at::Tensor> thread_cpu_ids,
                                          c10::optional<at::Tensor> w13_bias, c10::optional<at::Tensor> w2_bias,
                                          int64_t num_threads, std::string activation, int64_t global_num_experts,
                                          bool skip_weighted, bool fuse_silu, int64_t silu_poly_degree,
                                          int64_t gemm_backend, int64_t backend_n_tile,
                                          c10::optional<at::Tensor> out) {
#ifndef __aarch64__
  TORCH_CHECK(false, "fused_moe_bf16_tiled_scheduled requires AArch64");
#else
  const auto moe_trace_begin = ::fused_cpp::profile::now();
  MoeTraceCollector moe_trace(moe_trace_config_from_env());
  auto trace_phase_begin = [&]() -> ::fused_cpp::profile::TimePoint {
    if (!moe_trace.enabled()) {
      return {};
    }
    return ::fused_cpp::profile::now();
  };
  auto trace_phase_end = [&](int64_t tid, int64_t wave, int64_t group, int64_t local_tid, int64_t expert, int64_t rows,
                             const char* stage, ::fused_cpp::profile::TimePoint begin) {
    if (!moe_trace.enabled()) {
      return;
    }
    moe_trace.record_phase(tid, wave, group, local_tid, expert, rows, stage, begin);
  };

  check_bf16_cpu(input, "input");
  TORCH_CHECK(input.dim() == 2, "input must be 2-D [tokens, hidden]");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(topk_ids.device().is_cpu(), "topk_ids must be CPU");
  TORCH_CHECK(topk_weights.device().is_cpu(), "topk_weights must be CPU");
  TORCH_CHECK(is_integer_dtype(topk_ids.scalar_type()), "topk_ids must use an integer dtype");
  TORCH_CHECK(is_floating_dtype(topk_weights.scalar_type()), "topk_weights must use a floating dtype");
  TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must be 2-D [tokens, top_k]");
  TORCH_CHECK(topk_weights.dim() == 2, "topk_weights must be 2-D [tokens, top_k]");
  TORCH_CHECK(topk_ids.sizes() == topk_weights.sizes(), "topk_ids and topk_weights shapes must match");
  TORCH_CHECK(topk_ids.size(0) == input.size(0), "topk first dimension must match input token count");
  TORCH_CHECK(topk_ids.size(1) > 0, "top_k must be non-zero");
  TORCH_CHECK(num_threads > 0, "num_threads must be positive, got ", num_threads);
  TORCH_CHECK(num_threads <= std::numeric_limits<int>::max(), "num_threads exceeds int32 limit: ", num_threads);

  const ::fused_cpp::moe::MoeBackend& backend = ::fused_cpp::moe::backend_from_id(gemm_backend);
  const bool use_sve_backend = backend.id == ::fused_cpp::moe::BackendId::kArmSveBf16;
  if (use_sve_backend) {
    TORCH_CHECK(fuse_silu,
                "SVE scheduled MoE backend currently requires "
                "fuse_silu=True");
  }
  TORCH_CHECK(backend_n_tile == backend.n_tile(), "MoE backend_n_tile mismatch for ", backend.name,
              ": weights use ", backend_n_tile, ", runtime uses ", backend.n_tile());

  PackedExperts w13 = checked_packed_experts(w13_packed, w13_K, w13_N, "w13_packed", backend_n_tile);
  PackedExperts w2 = checked_packed_experts(w2_packed, w2_K, w2_N, "w2_packed", backend_n_tile);
  TORCH_CHECK(w13.E == w2.E, "w13 and w2 expert count mismatch");
  TORCH_CHECK(w13.K == input.size(1), "input hidden size mismatch: input H=", input.size(1), ", w13 K=", w13.K);
  TORCH_CHECK(w13.N % 2 == 0, "w13 N must be even, got ", w13.N);
  const int64_t F = w13.N / 2;
  const int64_t H = input.size(1);
  TORCH_CHECK(w2.K == F && w2.N == H, "w2 shape mismatch: expected K=", F, " N=", H, ", got K=", w2.K, " N=", w2.N);
  check_optional_bias(w13_bias, w13.E, w13.N, "w13_bias");
  check_optional_bias(w2_bias, w2.E, w2.N, "w2_bias");
  // Decide once, up front, whether to use the fused-bias GEMM kernel:
  // a defined bias yields a padded [E, N_pad] fp32 buffer; otherwise the
  // base pointer stays null and the plain (non-bias) kernel is used.
  const at::Tensor w13_bias_f32 = build_padded_bias_f32(w13_bias, w13.E, w13.N, w13.N_pad);
  const at::Tensor w2_bias_f32 = build_padded_bias_f32(w2_bias, w2.E, w2.N, w2.N_pad);
  const float* w13_bias_base = w13_bias_f32.defined() ? w13_bias_f32.data_ptr<float>() : nullptr;
  const float* w2_bias_base = w2_bias_f32.defined() ? w2_bias_f32.data_ptr<float>() : nullptr;

  if (fuse_silu) {
    TORCH_CHECK(activation == "silu", "fuse_silu only supports activation='silu', got ", activation);
    TORCH_CHECK(F % 8 == 0, "fuse_silu requires F % 8 == 0, got F=", F);
    TORCH_CHECK(w13.N_pad == 2 * F, "fuse_silu expects interleaved w13 with N_pad=2F (=", 2 * F,
                "), got N_pad=", w13.N_pad);
    TORCH_CHECK(w13_bias_base == nullptr, "scheduled fuse_silu does not support w13_bias yet");
    TORCH_CHECK(w2_bias_base == nullptr, "scheduled fuse_silu does not support w2_bias yet");
    TORCH_CHECK(silu_poly_degree == 4 || silu_poly_degree == 5 || silu_poly_degree == 6,
                "silu_poly_degree must be 4, 5, or 6, got ", silu_poly_degree);
  }
  const bool use_fused_2d_split = fuse_silu && env_flag_enabled("FUSED_CPP_MOE_FUSED_2D_SPLIT");
  // The SVE packC tails overwrite every row that the matching W2 tail reads.
  // Both switches default on; setting either environment flag to 0 restores
  // the corresponding legacy barrier for comparison or diagnosis.
  const bool elide_intermediate_zero =
      use_sve_backend && fuse_silu && env_flag_enabled_by_default("FUSED_CPP_MOE_SVE_ELIDE_INTERMEDIATE_ZERO");
  const bool use_w2_n_owner_scatter =
      use_sve_backend && fuse_silu && env_flag_enabled_by_default("FUSED_CPP_MOE_SVE_W2_N_OWNER_SCATTER");

  const int64_t num_tokens = input.size(0);
  const int64_t top_k = topk_ids.size(1);
  if (skip_weighted) {
    TORCH_CHECK(top_k == 1, "skip_weighted is only valid when top_k == 1");
  }
  bool use_w2_bf16_route = false;
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  use_w2_bf16_route =
      !skip_weighted && use_sve_backend && fuse_silu && w2_bias_base == nullptr && sve_w2_bf16_route_enabled();
#endif
  if (num_tokens == 0) {
    return finalize_moe_output(prepare_moe_output(input, w13_packed, w2_packed, topk_weights, topk_ids, out), out);
  }

  const int64_t num_experts = global_num_experts < 0 ? w13.E : global_num_experts;
  TORCH_CHECK(num_experts > 0, "global_num_experts must be positive or -1, got ", global_num_experts);
  TORCH_CHECK(num_experts <= w13.E, "global_num_experts cannot exceed prepared expert weights: ", num_experts, " > ",
              w13.E);

  ThreadPinningConfig scheduled_thread_pinning;
  bool has_scheduled_thread_pinning = false;
  if (thread_cpu_ids.has_value() && thread_cpu_ids->defined() && thread_cpu_ids->numel() > 0) {
    scheduled_thread_pinning.cpus = tensor_to_i64_vector(*thread_cpu_ids, "thread_cpu_ids");
    TORCH_CHECK(static_cast<int64_t>(scheduled_thread_pinning.cpus.size()) == num_threads,
                "thread_cpu_ids must have exactly num_threads entries: got ", scheduled_thread_pinning.cpus.size(),
                " vs ", num_threads);
    for (size_t idx = 0; idx < scheduled_thread_pinning.cpus.size(); ++idx) {
      TORCH_CHECK(scheduled_thread_pinning.cpus[idx] >= 0, "thread_cpu_ids[", idx, "] must be non-negative, got ",
                  scheduled_thread_pinning.cpus[idx]);
    }
    scheduled_thread_pinning.enabled = true;
    has_scheduled_thread_pinning = true;
  }
  ThreadPinningScope scheduled_thread_pinning_scope(has_scheduled_thread_pinning ? &scheduled_thread_pinning : nullptr);
  prepare_moe_threads_for_operator(num_threads);

  auto phase_begin = trace_phase_begin();
  const std::vector<int64_t> wave_offsets_v = tensor_to_i64_vector(wave_offsets, "wave_offsets");
  const std::vector<int64_t> team_expert_ids_v = tensor_to_i64_vector(team_expert_ids, "team_expert_ids");
  const std::vector<int64_t> team_threads_v = tensor_to_i64_vector(team_threads, "team_threads");
  trace_phase_end(-1, -1, -1, -1, -1, 0, "plan_materialize", phase_begin);
  TORCH_CHECK(wave_offsets_v.size() >= 2, "wave_offsets must contain at least [0, num_teams]");
  TORCH_CHECK(wave_offsets_v.front() == 0, "wave_offsets[0] must be 0, got ", wave_offsets_v.front());
  const int64_t num_teams = static_cast<int64_t>(team_expert_ids_v.size());
  TORCH_CHECK(static_cast<int64_t>(team_threads_v.size()) == num_teams,
              "team_threads must have the same length as team_expert_ids: ", team_threads_v.size(), " vs ", num_teams);
  TORCH_CHECK(wave_offsets_v.back() == num_teams, "last wave_offsets entry must equal num_teams=", num_teams, ", got ",
              wave_offsets_v.back());

  phase_begin = trace_phase_begin();
  at::Tensor ids_i64 = topk_ids.to(at::kLong).contiguous();
  at::Tensor weights_f32 = topk_weights.to(at::kFloat).contiguous();
  const int64_t* ids = ids_i64.data_ptr<int64_t>();
  const int64_t num_routes = num_tokens * top_k;
  bool use_w2_direct_route = false;
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  const int64_t route_element_bytes =
      use_w2_bf16_route ? static_cast<int64_t>(sizeof(uint16_t)) : static_cast<int64_t>(sizeof(float));
  use_w2_direct_route = !skip_weighted && use_sve_backend && fuse_silu && w2.N_pad == H &&
                        sve_w2_direct_route_offsets_fit(num_routes, H, w2.n_tile, route_element_bytes) &&
                        sve_w2_direct_route_enabled();
#endif

  std::vector<std::vector<int64_t>> routes(static_cast<size_t>(num_experts));
  for (int64_t flat = 0; flat < num_routes; ++flat) {
    const int64_t expert = ids[flat];
    TORCH_CHECK(expert >= 0 && expert < num_experts, "topk_ids out of range: id=", expert, ", valid range [0, ",
                num_experts, ")");
    routes[static_cast<size_t>(expert)].push_back(flat);
  }
  trace_phase_end(-1, -1, -1, -1, -1, num_routes, "route_build", phase_begin);

  phase_begin = trace_phase_begin();
  int64_t active_experts = 0;
  for (const std::vector<int64_t>& expert_routes : routes) {
    if (!expert_routes.empty()) {
      ++active_experts;
    }
  }
  TORCH_CHECK(num_teams == active_experts,
              "scheduled plan must contain exactly one team per active "
              "expert: teams=",
              num_teams, " active_experts=", active_experts);

  std::vector<int64_t> seen(static_cast<size_t>(num_experts), 0);
  std::vector<int64_t> team_rows(static_cast<size_t>(num_teams), 0);
  int64_t trace_gemm_hint = 0;
  for (int64_t team = 0; team < num_teams; ++team) {
    const int64_t expert = team_expert_ids_v[static_cast<size_t>(team)];
    const int64_t threads = team_threads_v[static_cast<size_t>(team)];
    TORCH_CHECK(expert >= 0 && expert < num_experts, "team_expert_ids[", team, "] out of range: ", expert,
                ", valid range [0, ", num_experts, ")");
    TORCH_CHECK(threads > 0, "team_threads[", team, "] must be positive, got ", threads);
    TORCH_CHECK(threads <= num_threads, "team_threads[", team, "]=", threads, " exceeds num_threads=", num_threads);
    TORCH_CHECK(seen[static_cast<size_t>(expert)] == 0, "scheduled plan contains duplicate expert ", expert);
    const int64_t rows = static_cast<int64_t>(routes[static_cast<size_t>(expert)].size());
    TORCH_CHECK(rows > 0, "scheduled plan contains inactive expert ", expert);
    check_positive_int(rows, "scheduled team rows");
    seen[static_cast<size_t>(expert)] = 1;
    team_rows[static_cast<size_t>(team)] = rows;
    trace_gemm_hint += threads * 2;
  }
  for (int64_t expert = 0; expert < num_experts; ++expert) {
    if (!routes[static_cast<size_t>(expert)].empty()) {
      TORCH_CHECK(seen[static_cast<size_t>(expert)] == 1, "scheduled plan is missing active expert ", expert);
    }
  }

  const int64_t num_waves = static_cast<int64_t>(wave_offsets_v.size()) - 1;
  std::vector<ScheduledWaveRuntime> waves;
  waves.reserve(static_cast<size_t>(num_waves));
  std::vector<int64_t> team_thread_starts(static_cast<size_t>(num_teams), 0);
  std::vector<int64_t> team_scratch_indices(static_cast<size_t>(num_teams), -1);
  std::vector<ScheduledScratchUnitConfig> scratch_unit_configs;
  for (int64_t wave = 0; wave < num_waves; ++wave) {
    const int64_t begin = wave_offsets_v[static_cast<size_t>(wave)];
    const int64_t end = wave_offsets_v[static_cast<size_t>(wave + 1)];
    TORCH_CHECK(begin <= end, "wave_offsets must be nondecreasing, got wave ", wave, " begin=", begin, " end=", end);
    TORCH_CHECK(begin >= 0 && end <= num_teams, "wave ", wave, " range [", begin, ", ", end,
                ") is outside num_teams=", num_teams);
    int64_t wave_threads = 0;
    for (int64_t team = begin; team < end; ++team) {
      const int64_t thread_begin = wave_threads;
      const int64_t team_threads = team_threads_v[static_cast<size_t>(team)];
      team_thread_starts[static_cast<size_t>(team)] = thread_begin;
      wave_threads += team_threads;
      TORCH_CHECK(wave_threads <= num_threads, "wave ", wave, " uses ", wave_threads,
                  " threads, exceeding num_threads=", num_threads);

      int64_t scratch_idx = -1;
      for (size_t idx = 0; idx < scratch_unit_configs.size(); ++idx) {
        const ScheduledScratchUnitConfig& config = scratch_unit_configs[idx];
        if (config.thread_begin == thread_begin && config.threads == team_threads) {
          scratch_idx = static_cast<int64_t>(idx);
          break;
        }
      }
      if (scratch_idx < 0) {
        scratch_idx = static_cast<int64_t>(scratch_unit_configs.size());
        scratch_unit_configs.push_back(ScheduledScratchUnitConfig{thread_begin, team_threads, 0, 0, fuse_silu,
                                                                  use_w2_bf16_route, use_w2_direct_route});
      }
      ScheduledScratchUnitConfig& scratch_config = scratch_unit_configs[static_cast<size_t>(scratch_idx)];
      const int64_t rows = team_rows[static_cast<size_t>(team)];
      scratch_config.max_rows = std::max(scratch_config.max_rows, rows);
      scratch_config.a_reorder_stride =
          std::max(scratch_config.a_reorder_stride,
                   fuse_silu ? int64_t{0} : scheduled_a_reorder_stride(rows, team_threads, w13, w2));
      scratch_config.fused_packa = scratch_config.fused_packa || fuse_silu;
      scratch_config.w2_bf16_route = scratch_config.w2_bf16_route || use_w2_bf16_route;
      scratch_config.w2_direct_route = scratch_config.w2_direct_route || use_w2_direct_route;
      team_scratch_indices[static_cast<size_t>(team)] = scratch_idx;
    }
    waves.push_back(ScheduledWaveRuntime{begin, end, wave_threads});
  }
  trace_phase_end(-1, -1, -1, -1, -1, num_teams, "plan_validate", phase_begin);

  at::Tensor output = prepare_moe_output(input, w13_packed, w2_packed, topk_weights, topk_ids, out);
  uint16_t* out_bf16_ptr = bf16_data(output);
  at::Tensor route_out;
  float* route_out_ptr = nullptr;
  uint16_t* route_out_bf16_ptr = nullptr;
  if (!skip_weighted) {
    route_out =
        at::empty({num_routes, H},
                  at::TensorOptions().device(input.device()).dtype(use_w2_bf16_route ? at::kBFloat16 : at::kFloat));
    if (use_w2_bf16_route) {
      route_out_bf16_ptr = bf16_data(route_out);
    } else {
      route_out_ptr = route_out.data_ptr<float>();
    }
  }
  const uint16_t* input_ptr = bf16_data_const(input);
  const uint16_t* w13_ptr = bf16_data_const(w13.tensor);
  const uint16_t* w2_ptr = bf16_data_const(w2.tensor);

  phase_begin = trace_phase_begin();
  ScheduledScratchLease scratch_lease = resident_scheduled_scratch_pool().lease(scratch_unit_configs, w13, w2);
  const std::vector<ScheduledTeamScratch*>& scratches = scratch_lease.scratches();
  trace_phase_end(-1, -1, -1, -1, -1, static_cast<int64_t>(scratch_unit_configs.size()), "scratch_alloc", phase_begin);

  const int schedule_debug_level = debug_schedule_level();
  if (schedule_debug_level > 0) {
    std::fprintf(stderr,
                 "[fused_moe_bf16_tiled][schedule] threads=%lld experts=%lld "
                 "routes=%lld waves=%lld teams=%lld strategy=external_plan\n",
                 static_cast<long long>(num_threads), static_cast<long long>(num_experts),
                 static_cast<long long>(num_routes), static_cast<long long>(num_waves),
                 static_cast<long long>(num_teams));
  }

  moe_trace.prepare_thread_buffers(
      num_threads, static_cast<size_t>(trace_gemm_hint),
      static_cast<size_t>(trace_gemm_hint) * 2 + static_cast<size_t>(num_threads) + 16);
  ThreadBarrier wave_barrier(num_threads);
  phase_begin = trace_phase_begin();
  run_fixed_threads(num_threads, [&](int64_t tid) {
    for (int64_t wave_idx = 0; wave_idx < num_waves; ++wave_idx) {
      const ScheduledWaveRuntime& wave = waves[static_cast<size_t>(wave_idx)];
      int64_t selected_team = -1;
      int64_t local_tid = -1;
      if (tid < wave.total_threads) {
        for (int64_t team = wave.begin; team < wave.end; ++team) {
          const int64_t thread_begin = team_thread_starts[static_cast<size_t>(team)];
          const int64_t thread_end = thread_begin + team_threads_v[static_cast<size_t>(team)];
          if (tid >= thread_begin && tid < thread_end) {
            selected_team = team;
            local_tid = tid - thread_begin;
            break;
          }
        }
      }

      if (selected_team >= 0) {
        const int64_t scratch_idx = team_scratch_indices[static_cast<size_t>(selected_team)];
        TORCH_CHECK(scratch_idx >= 0, "missing scratch unit for scheduled team ", selected_team);
        ScheduledTeamScratch& scratch = *scratches[static_cast<size_t>(scratch_idx)];
        ThreadBarrier& barrier = scratch.barrier;
        const int64_t expert = team_expert_ids_v[static_cast<size_t>(selected_team)];
        const int64_t rows = team_rows[static_cast<size_t>(selected_team)];
        const int64_t group_size = team_threads_v[static_cast<size_t>(selected_team)];
        TORCH_CHECK(scratch.threads == group_size, "scratch thread count mismatch for team ", selected_team);
        TORCH_CHECK(rows <= scratch.max_rows, "scratch row capacity mismatch for team ", selected_team);
        if (!fuse_silu) {
          TORCH_CHECK(scheduled_a_reorder_stride(rows, group_size, w13, w2) <= scratch.a_reorder_stride,
                      "scratch A reorder capacity mismatch for team ", selected_team);
        }
        const auto& expert_routes = routes[static_cast<size_t>(expert)];
        uint16_t* a_reorder =
            scratch.a_reorder.empty() ? nullptr : scratch.a_reorder.data() + local_tid * scratch.a_reorder_stride;

        {
          auto worker_phase_begin = trace_phase_begin();
          if (fuse_silu) {
            const int64_t nb = ceil_div_int64(rows, kKernelTile);
            const SplitRange brange = split_evenly(nb, group_size, local_tid);
            if (use_sve_backend) {
              gather_pack_a_reorder_sve_hybrid(input_ptr, H, expert_routes.data(), top_k, scratch.packed_a.data(),
                                               static_cast<int>(rows), static_cast<int>(w13.K_pad), group_size,
                                               local_tid);
            } else {
              gather_pack_a_reorder_backend(false, input_ptr, H, expert_routes.data(), top_k, scratch.packed_a.data(),
                                            static_cast<int>(rows), static_cast<int>(w13.K_pad),
                                            static_cast<int>(brange.begin),
                                            static_cast<int>(brange.begin + brange.size));
            }
            trace_phase_end(tid, wave_idx, selected_team, local_tid, expert, brange.size * kKernelTile, "gather_pack_a",
                            worker_phase_begin);
          } else {
            // Parallel gather: each team thread copies its own row
            // slice (was serial on local_tid==0 with the rest idle).
            const SplitRange grange = split_evenly(rows, group_size, local_tid);
            for (int64_t m = grange.begin; m < grange.begin + grange.size; ++m) {
              const int64_t flat = expert_routes[static_cast<size_t>(m)];
              const int64_t token = flat / top_k;
              uint16_t* dst = scratch.input.data() + m * w13.K_pad;
              std::fill(dst, dst + w13.K_pad, static_cast<uint16_t>(0));
              std::copy(input_ptr + token * H, input_ptr + token * H + H, dst);
            }
            trace_phase_end(tid, wave_idx, selected_team, local_tid, expert, grange.size, "gather_input",
                            worker_phase_begin);
          }
        }
        barrier.wait();

        auto worker_phase_begin = trace_phase_begin();
        if (fuse_silu) {
          if (!elide_intermediate_zero && local_tid == 0) {
            const int64_t rows_padded =
                use_sve_backend ? sve_hybrid_packed_rows(rows) : ceil_to_multiple(rows, int64_t{kKernelTile});
            std::fill(scratch.intermediate.begin(), scratch.intermediate.begin() + rows_padded * w2.K_pad,
                      static_cast<uint16_t>(0));
          }
          if (!elide_intermediate_zero) {
            barrier.wait();
          }
          TeamContext team;
          team.group_size = group_size;
          team.local_tid = local_tid;
          team.barrier = group_size > 1 ? &barrier : nullptr;
          team.a_reorder = nullptr;
          const Gemm2DSplitPlan w13_2d_plan = plan_2d_gemm_split(rows, w13.K_pad, w13.N_pad, group_size, w13.n_tile);
          team_fused_w13_silu_packed_packc_backend(
              use_sve_backend, use_fused_2d_split, team, w13_2d_plan, scratch.packed_a.data(),
              w13_ptr + expert * w13.packed_stride, scratch.intermediate.data(), static_cast<int>(rows),
              static_cast<int>(w13.K_pad), static_cast<int>(w13.N_pad), static_cast<int>(w2.K_pad), silu_poly_degree,
              w13.n_tile);
          trace_phase_end(tid, wave_idx, selected_team, local_tid, expert, rows, "w13_fused_silu_packc",
                          worker_phase_begin);
        } else {
          trace_dispatch_fp32_gemm_stage_split(moe_trace, "w13", MoeGemmStage::kW13, tid, wave_idx, selected_team,
                                               local_tid, expert, 0, rows, scratch.input.data(),
                                               w13_ptr + expert * w13.packed_stride, scratch.gate_up.data(), a_reorder,
                                               static_cast<int>(rows), static_cast<int>(w13.K_pad),
                                               static_cast<int>(w13.N_pad), static_cast<int>(w13.N_pad), group_size,
                                               w13_bias_base != nullptr ? w13_bias_base + expert * w13.N_pad : nullptr);
        }
        barrier.wait();

        if (!fuse_silu) {
          worker_phase_begin = trace_phase_begin();
          const SplitRange activation_range = split_evenly(rows, group_size, local_tid);
          activation_range_to_bf16(activation, scratch.gate_up.data(), scratch.intermediate.data(),
                                   activation_range.begin, activation_range.size, w13.N_pad, w2.K_pad, F);
          trace_phase_end(tid, wave_idx, selected_team, local_tid, expert, activation_range.size, "activation",
                          worker_phase_begin);
          barrier.wait();
        }

        worker_phase_begin = trace_phase_begin();
        if (fuse_silu) {
          TeamContext w2team;
          w2team.group_size = group_size;
          w2team.local_tid = local_tid;
          w2team.barrier = nullptr;
          w2team.a_reorder = nullptr;
          const Gemm2DSplitPlan w2_2d_plan = plan_2d_gemm_split(rows, w2.K_pad, w2.N_pad, group_size, w2.n_tile);
          if (use_w2_direct_route) {
            if (use_w2_bf16_route) {
              team_w2_packed_sve_direct_bf16_route_backend(
                  use_fused_2d_split, w2team, w2_2d_plan, scratch.intermediate.data(),
                  w2_ptr + expert * w2.packed_stride, route_out_bf16_ptr, expert_routes.data(), static_cast<int>(rows),
                  static_cast<int>(w2.K_pad), static_cast<int>(w2.N_pad), static_cast<int>(H), w2.n_tile);
            } else {
              team_w2_packed_sve_direct_route_backend(
                  use_fused_2d_split, w2team, w2_2d_plan, scratch.intermediate.data(),
                  w2_ptr + expert * w2.packed_stride, route_out_ptr, expert_routes.data(), static_cast<int>(rows),
                  static_cast<int>(w2.K_pad), static_cast<int>(w2.N_pad), static_cast<int>(H), w2.n_tile);
            }
          } else if (use_w2_bf16_route) {
            team_w2_packed_bf16_sve_backend(
                use_fused_2d_split, w2team, w2_2d_plan, scratch.intermediate.data(), w2_ptr + expert * w2.packed_stride,
                scratch.down_bf16.data(), static_cast<int>(rows), static_cast<int>(w2.K_pad),
                static_cast<int>(w2.N_pad), static_cast<int>(w2.N_pad), w2.n_tile);
          } else {
            team_w2_packed_backend(use_sve_backend, use_fused_2d_split, w2team, w2_2d_plan, scratch.intermediate.data(),
                                   w2_ptr + expert * w2.packed_stride, scratch.down.data(), static_cast<int>(rows),
                                   static_cast<int>(w2.K_pad), static_cast<int>(w2.N_pad), static_cast<int>(w2.N_pad),
                                   w2.n_tile);
          }
          trace_phase_end(tid, wave_idx, selected_team, local_tid, expert, rows,
                          use_w2_direct_route ? "w2_direct_route" : "w2_packed", worker_phase_begin);
        } else {
          trace_dispatch_fp32_gemm_stage_split(moe_trace, "w2", MoeGemmStage::kW2, tid, wave_idx, selected_team,
                                               local_tid, expert, 0, rows, scratch.intermediate.data(),
                                               w2_ptr + expert * w2.packed_stride, scratch.down.data(), a_reorder,
                                               static_cast<int>(rows), static_cast<int>(w2.K_pad),
                                               static_cast<int>(w2.N_pad), static_cast<int>(w2.N_pad), group_size,
                                               w2_bias_base != nullptr ? w2_bias_base + expert * w2.N_pad : nullptr);
        }
        if (!use_w2_direct_route && !use_w2_n_owner_scatter) {
          barrier.wait();
        }

        if (!use_w2_direct_route) {
          worker_phase_begin = trace_phase_begin();
          with_w2_scatter_owner(static_cast<int>(w2.N_pad), group_size, local_tid, w2.n_tile,
                                use_w2_n_owner_scatter, [&](const SplitRange& h_range) {
                                      const int64_t h_begin = h_range.begin;
                                      const int64_t h_end = std::min<int64_t>(H, h_begin + h_range.size);
                                      if (h_begin >= h_end) {
                                        return;
                                      }
                                      for (int64_t m = 0; m < rows; ++m) {
                                        const int64_t flat = expert_routes[static_cast<size_t>(m)];
                                        if (use_w2_bf16_route) {
                                          const uint16_t* src = scratch.down_bf16.data() + m * w2.N_pad;
                                          uint16_t* dst = route_out_bf16_ptr + flat * H;
                                          std::copy(src + h_begin, src + h_end, dst + h_begin);
                                        } else if (skip_weighted) {
                                          const float* src = scratch.down.data() + m * w2.N_pad;
                                          uint16_t* dst = out_bf16_ptr + flat * H;
                                          convert_f32_to_bf16(src + h_begin, dst + h_begin, h_end - h_begin);
                                        } else {
                                          const float* src = scratch.down.data() + m * w2.N_pad;
                                          float* dst = route_out_ptr + flat * H;
                                          std::copy(src + h_begin, src + h_end, dst + h_begin);
                                        }
                                      }
                                    });
          trace_phase_end(tid, wave_idx, selected_team, local_tid, expert, rows, "scatter_route_out",
                          worker_phase_begin);
        }
        barrier.wait();
      }
      wave_barrier.wait();
    }
  });
  trace_phase_end(-1, -1, -1, -1, -1, num_routes, "scheduled_compute", phase_begin);

  // 2a/2b: weighted merge writes bf16 straight into `output`; when
  // skip_weighted is set the scatter already filled `output`, so merge is
  // skipped.
  const float* topk_w = weights_f32.data_ptr<float>();
  const int route_merge_unroll = skip_weighted ? 0 : resolve_route_merge_unroll(use_sve_backend);

  auto merge_routes = [&](int64_t tid) {
    auto worker_phase_begin = trace_phase_begin();
    const int64_t rows_per_thread = ceil_div_int64(num_tokens, num_threads);
    const int64_t token_begin = tid * rows_per_thread;
    const int64_t token_end = std::min<int64_t>(num_tokens, token_begin + rows_per_thread);
    merge_route_range(route_out_ptr, route_out_bf16_ptr, topk_w, out_bf16_ptr, token_begin, token_end, top_k, H,
                      use_w2_bf16_route, route_merge_unroll);
    trace_phase_end(tid, -1, -1, -1, -1, std::max<int64_t>(0, token_end - token_begin), "merge_routes",
                    worker_phase_begin);
  };
  phase_begin = trace_phase_begin();
  if (!skip_weighted) {
    run_fixed_threads(num_threads, merge_routes);
  }
  trace_phase_end(-1, -1, -1, -1, -1, num_tokens, "merge_routes_total", phase_begin);

  if (moe_trace.enabled()) {
    const double e2e_ms = ::fused_cpp::profile::elapsed_ms(moe_trace_begin);
    moe_trace.write_report("external_plan_scheduled", num_threads, num_tokens, top_k, num_experts, num_routes, H, F,
                           static_cast<size_t>(active_experts), num_teams, num_teams, 0, e2e_ms);
  }
  return finalize_moe_output(output, out);
#endif
}

namespace {

at::Tensor run_fused_moe_bf16_tiled_async(
    at::Tensor input, at::Tensor w13_packed, int64_t w13_K, int64_t w13_N, at::Tensor w2_packed, int64_t w2_K,
    int64_t w2_N, at::Tensor topk_weights, at::Tensor topk_ids, at::Tensor task_expert_ids,
    at::Tensor task_core_begins, at::Tensor task_threads, at::Tensor task_dep_offsets, at::Tensor task_deps,
    c10::optional<at::Tensor> thread_cpu_ids, c10::optional<at::Tensor> w13_bias,
    c10::optional<at::Tensor> w2_bias, int64_t num_threads, std::string activation, int64_t global_num_experts,
    bool skip_weighted, bool fuse_silu, int64_t silu_poly_degree, int64_t gemm_backend, int64_t backend_n_tile,
    c10::optional<at::Tensor> out, const AsyncPlanV2NativeArgs* plan_v2) {
#ifndef __aarch64__
  TORCH_CHECK(false, "fused_moe_bf16_tiled_async requires AArch64");
#else
  const auto moe_trace_begin = ::fused_cpp::profile::now();
  MoeTraceCollector moe_trace(moe_trace_config_from_env());
  auto trace_phase_begin = [&]() -> ::fused_cpp::profile::TimePoint {
    if (!moe_trace.enabled()) {
      return {};
    }
    return ::fused_cpp::profile::now();
  };
  auto trace_phase_end = [&](int64_t tid, int64_t task, int64_t local_tid, int64_t expert, int64_t rows,
                             const char* stage, ::fused_cpp::profile::TimePoint begin) {
    if (!moe_trace.enabled()) {
      return;
    }
    moe_trace.record_phase(tid, -1, task, local_tid, expert, rows, stage, begin);
  };

  check_bf16_cpu(input, "input");
  TORCH_CHECK(input.dim() == 2, "input must be 2-D [tokens, hidden]");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(topk_ids.device().is_cpu(), "topk_ids must be CPU");
  TORCH_CHECK(topk_weights.device().is_cpu(), "topk_weights must be CPU");
  TORCH_CHECK(is_integer_dtype(topk_ids.scalar_type()), "topk_ids must use an integer dtype");
  TORCH_CHECK(is_floating_dtype(topk_weights.scalar_type()), "topk_weights must use a floating dtype");
  TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must be 2-D [tokens, top_k]");
  TORCH_CHECK(topk_weights.dim() == 2, "topk_weights must be 2-D [tokens, top_k]");
  TORCH_CHECK(topk_ids.sizes() == topk_weights.sizes(), "topk_ids and topk_weights shapes must match");
  TORCH_CHECK(topk_ids.size(0) == input.size(0), "topk first dimension must match input token count");
  TORCH_CHECK(topk_ids.size(1) > 0, "top_k must be non-zero");
  TORCH_CHECK(num_threads > 0, "num_threads must be positive, got ", num_threads);
  TORCH_CHECK(num_threads <= std::numeric_limits<int>::max(), "num_threads exceeds int32 limit: ", num_threads);

  const ::fused_cpp::moe::MoeBackend& backend = ::fused_cpp::moe::backend_from_id(gemm_backend);
  const bool use_sve_backend = backend.id == ::fused_cpp::moe::BackendId::kArmSveBf16;
  if (use_sve_backend) {
    TORCH_CHECK(fuse_silu, "SVE async MoE backend currently requires fuse_silu=True");
  }
  TORCH_CHECK(backend_n_tile == backend.n_tile(), "MoE backend_n_tile mismatch for ", backend.name,
              ": weights use ", backend_n_tile, ", runtime uses ", backend.n_tile());

  PackedExperts w13 = checked_packed_experts(w13_packed, w13_K, w13_N, "w13_packed", backend_n_tile);
  PackedExperts w2 = checked_packed_experts(w2_packed, w2_K, w2_N, "w2_packed", backend_n_tile);
  TORCH_CHECK(w13.E == w2.E, "w13 and w2 expert count mismatch");
  TORCH_CHECK(w13.K == input.size(1), "input hidden size mismatch: input H=", input.size(1), ", w13 K=", w13.K);
  TORCH_CHECK(w13.N % 2 == 0, "w13 N must be even, got ", w13.N);
  const int64_t F = w13.N / 2;
  const int64_t H = input.size(1);
  TORCH_CHECK(w2.K == F && w2.N == H, "w2 shape mismatch: expected K=", F, " N=", H, ", got K=", w2.K, " N=", w2.N);
  check_optional_bias(w13_bias, w13.E, w13.N, "w13_bias");
  check_optional_bias(w2_bias, w2.E, w2.N, "w2_bias");
  const at::Tensor w13_bias_f32 = build_padded_bias_f32(w13_bias, w13.E, w13.N, w13.N_pad);
  const at::Tensor w2_bias_f32 = build_padded_bias_f32(w2_bias, w2.E, w2.N, w2.N_pad);
  const float* w13_bias_base = w13_bias_f32.defined() ? w13_bias_f32.data_ptr<float>() : nullptr;
  const float* w2_bias_base = w2_bias_f32.defined() ? w2_bias_f32.data_ptr<float>() : nullptr;

  if (fuse_silu) {
    TORCH_CHECK(activation == "silu", "fuse_silu only supports activation='silu', got ", activation);
    TORCH_CHECK(F % 8 == 0, "fuse_silu requires F % 8 == 0, got F=", F);
    TORCH_CHECK(w13.N_pad == 2 * F, "fuse_silu expects interleaved w13 with N_pad=2F (=", 2 * F,
                "), got N_pad=", w13.N_pad);
    TORCH_CHECK(w13_bias_base == nullptr, "async fuse_silu does not support w13_bias yet");
    TORCH_CHECK(w2_bias_base == nullptr, "async fuse_silu does not support w2_bias yet");
    TORCH_CHECK(silu_poly_degree == 4 || silu_poly_degree == 5 || silu_poly_degree == 6,
                "silu_poly_degree must be 4, 5, or 6, got ", silu_poly_degree);
  }
  const bool use_fused_2d_split = fuse_silu && env_flag_enabled("FUSED_CPP_MOE_FUSED_2D_SPLIT");
  const bool elide_intermediate_zero =
      use_sve_backend && fuse_silu && env_flag_enabled_by_default("FUSED_CPP_MOE_SVE_ELIDE_INTERMEDIATE_ZERO");
  const bool use_w2_n_owner_scatter =
      use_sve_backend && fuse_silu && env_flag_enabled_by_default("FUSED_CPP_MOE_SVE_W2_N_OWNER_SCATTER");

  const int64_t num_tokens = input.size(0);
  const int64_t top_k = topk_ids.size(1);
  if (skip_weighted) {
    TORCH_CHECK(top_k == 1, "skip_weighted is only valid when top_k == 1");
  }
  bool use_w2_bf16_route = false;
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  use_w2_bf16_route =
      !skip_weighted && use_sve_backend && fuse_silu && w2_bias_base == nullptr && sve_w2_bf16_route_enabled();
#endif
  if (num_tokens == 0) {
    return finalize_moe_output(prepare_moe_output(input, w13_packed, w2_packed, topk_weights, topk_ids, out), out);
  }

  const int64_t num_experts = global_num_experts < 0 ? w13.E : global_num_experts;
  TORCH_CHECK(num_experts > 0, "global_num_experts must be positive or -1, got ", global_num_experts);
  TORCH_CHECK(num_experts <= w13.E, "global_num_experts cannot exceed prepared expert weights: ", num_experts, " > ",
              w13.E);

  ThreadPinningConfig async_thread_pinning;
  bool has_async_thread_pinning = false;
  if (thread_cpu_ids.has_value() && thread_cpu_ids->defined() && thread_cpu_ids->numel() > 0) {
    async_thread_pinning.cpus = tensor_to_i64_vector(*thread_cpu_ids, "thread_cpu_ids");
    TORCH_CHECK(static_cast<int64_t>(async_thread_pinning.cpus.size()) == num_threads,
                "thread_cpu_ids must have exactly num_threads entries: got ", async_thread_pinning.cpus.size(), " vs ",
                num_threads);
    for (size_t idx = 0; idx < async_thread_pinning.cpus.size(); ++idx) {
      TORCH_CHECK(async_thread_pinning.cpus[idx] >= 0, "thread_cpu_ids[", idx, "] must be non-negative, got ",
                  async_thread_pinning.cpus[idx]);
      if (plan_v2 != nullptr) {
        for (size_t prior = 0; prior < idx; ++prior) {
          TORCH_CHECK(async_thread_pinning.cpus[prior] != async_thread_pinning.cpus[idx],
                      "Plan V2 thread_cpu_ids must not contain duplicates: index=", idx,
                      " cpu=", async_thread_pinning.cpus[idx]);
        }
      }
    }
    async_thread_pinning.enabled = true;
    has_async_thread_pinning = true;
  }
  TORCH_CHECK(plan_v2 == nullptr || has_async_thread_pinning,
              "Plan V2 requires thread_cpu_ids with exactly num_threads entries");
  ThreadPinningScope async_thread_pinning_scope(has_async_thread_pinning ? &async_thread_pinning : nullptr);
  prepare_moe_threads_for_operator(num_threads);

  auto phase_begin = trace_phase_begin();
  const std::vector<int64_t> task_expert_ids_v = tensor_to_i64_vector(task_expert_ids, "task_expert_ids");
  const std::vector<int64_t> task_core_begins_v = tensor_to_i64_vector(task_core_begins, "task_core_begins");
  const std::vector<int64_t> task_threads_v = tensor_to_i64_vector(task_threads, "task_threads");
  const std::vector<int64_t> task_dep_offsets_v = tensor_to_i64_vector(task_dep_offsets, "task_dep_offsets");
  const std::vector<int64_t> task_deps_v = tensor_to_i64_vector(task_deps, "task_deps");
  const bool has_plan_v2 = plan_v2 != nullptr;
  std::vector<int64_t> task_preferred_threads_v;
  std::vector<int64_t> task_min_threads_v;
  std::vector<int64_t> task_max_threads_v;
  std::vector<int64_t> task_allowed_thread_offsets_v;
  std::vector<int64_t> task_allowed_threads_v;
  std::vector<int64_t> task_placement_modes_v;
  std::vector<int64_t> task_numa_nodes_v;
  std::vector<int64_t> task_stage_ids_v;
  std::vector<int64_t> task_resize_points_v;
  std::vector<int64_t> task_range_granularities_v;
  std::vector<int64_t> task_release_ns_v;
  std::vector<int64_t> task_resize_timeout_ns_v;
  std::vector<int64_t> task_preferred_core_begins_v;
  bool has_task_release_ns = false;
  bool has_task_resize_timeout_ns = false;
  bool has_task_preferred_core_begins = false;
  int64_t* elastic_stats_ptr = nullptr;
  if (has_plan_v2) {
    TORCH_CHECK(plan_v2->plan_version == kAsyncPlanV2, "plan_version must be ", kAsyncPlanV2, ", got ",
                plan_v2->plan_version);
    TORCH_CHECK(plan_v2->execution_mode == kAsyncExecutionStrict ||
                    plan_v2->execution_mode == kAsyncExecutionTailPool ||
                    plan_v2->execution_mode == kAsyncExecutionElastic,
                "Plan V2 execution_mode must be strict (", kAsyncExecutionStrict, "), tail_pool (",
                kAsyncExecutionTailPool, "), or elastic (", kAsyncExecutionElastic, "), got ",
                plan_v2->execution_mode);
    TORCH_CHECK(plan_v2->early_merge >= kAsyncEarlyMergeAuto &&
                    plan_v2->early_merge <= kAsyncEarlyMergeOn,
                "Plan V2 early_merge must be -1 (auto), 0 (off), or 1 (on), got ",
                plan_v2->early_merge);
    task_preferred_threads_v =
        tensor_to_i64_vector(plan_v2->task_preferred_threads, "task_preferred_threads");
    task_min_threads_v = tensor_to_i64_vector(plan_v2->task_min_threads, "task_min_threads");
    task_max_threads_v = tensor_to_i64_vector(plan_v2->task_max_threads, "task_max_threads");
    task_allowed_thread_offsets_v =
        tensor_to_i64_vector(plan_v2->task_allowed_thread_offsets, "task_allowed_thread_offsets");
    task_allowed_threads_v = tensor_to_i64_vector(plan_v2->task_allowed_threads, "task_allowed_threads");
    task_placement_modes_v = tensor_to_i64_vector(plan_v2->task_placement_modes, "task_placement_modes");
    task_numa_nodes_v = tensor_to_i64_vector(plan_v2->task_numa_nodes, "task_numa_nodes");
    task_stage_ids_v = tensor_to_i64_vector(plan_v2->task_stage_ids, "task_stage_ids");
    task_resize_points_v = tensor_to_i64_vector(plan_v2->task_resize_points, "task_resize_points");
    task_range_granularities_v =
        tensor_to_i64_vector(plan_v2->task_range_granularities, "task_range_granularities");
    has_task_release_ns =
        plan_v2->task_release_ns.has_value() && plan_v2->task_release_ns->defined();
    has_task_resize_timeout_ns =
        plan_v2->task_resize_timeout_ns.has_value() && plan_v2->task_resize_timeout_ns->defined();
    has_task_preferred_core_begins =
        plan_v2->task_preferred_core_begins.has_value() && plan_v2->task_preferred_core_begins->defined();
    if (has_task_release_ns) {
      task_release_ns_v = tensor_to_i64_vector(*plan_v2->task_release_ns, "task_release_ns");
    }
    if (has_task_resize_timeout_ns) {
      task_resize_timeout_ns_v =
          tensor_to_i64_vector(*plan_v2->task_resize_timeout_ns, "task_resize_timeout_ns");
    }
    if (has_task_preferred_core_begins) {
      task_preferred_core_begins_v =
          tensor_to_i64_vector(*plan_v2->task_preferred_core_begins, "task_preferred_core_begins");
    }
    if (plan_v2->elastic_stats_out.has_value() && plan_v2->elastic_stats_out->defined()) {
      at::Tensor stats = *plan_v2->elastic_stats_out;
      TORCH_CHECK(stats.device().is_cpu(), "elastic_stats_out must be a CPU tensor");
      TORCH_CHECK(stats.scalar_type() == at::kLong, "elastic_stats_out must use torch.int64");
      TORCH_CHECK(stats.is_contiguous(), "elastic_stats_out must be contiguous");
      TORCH_CHECK(stats.numel() >= kAsyncElasticStatsCount, "elastic_stats_out must contain at least ",
                  kAsyncElasticStatsCount, " elements, got ", stats.numel());
      stats.zero_();
      elastic_stats_ptr = stats.data_ptr<int64_t>();
    }
  }
  trace_phase_end(-1, -1, -1, -1, 0, "plan_materialize", phase_begin);

  const int64_t num_tasks = static_cast<int64_t>(task_expert_ids_v.size());
  TORCH_CHECK(num_tasks > 0, "async task plan must not be empty");
  TORCH_CHECK(static_cast<int64_t>(task_core_begins_v.size()) == num_tasks,
              "task_core_begins must match task_expert_ids length");
  TORCH_CHECK(static_cast<int64_t>(task_threads_v.size()) == num_tasks,
              "task_threads must match task_expert_ids length");
  TORCH_CHECK(static_cast<int64_t>(task_dep_offsets_v.size()) == num_tasks + 1,
              "task_dep_offsets must have num_tasks + 1 entries");
  TORCH_CHECK(task_dep_offsets_v.front() == 0, "task_dep_offsets[0] must be 0");
  TORCH_CHECK(task_dep_offsets_v.back() == static_cast<int64_t>(task_deps_v.size()),
              "last task_dep_offsets entry must equal task_deps length");
  if (has_plan_v2) {
    auto check_per_task_size = [&](const std::vector<int64_t>& values, const char* name) {
      TORCH_CHECK(static_cast<int64_t>(values.size()) == num_tasks, name, " must have one entry per task: got ",
                  values.size(), " vs ", num_tasks);
    };
    check_per_task_size(task_preferred_threads_v, "task_preferred_threads");
    check_per_task_size(task_min_threads_v, "task_min_threads");
    check_per_task_size(task_max_threads_v, "task_max_threads");
    check_per_task_size(task_placement_modes_v, "task_placement_modes");
    check_per_task_size(task_numa_nodes_v, "task_numa_nodes");
    check_per_task_size(task_stage_ids_v, "task_stage_ids");
    check_per_task_size(task_resize_points_v, "task_resize_points");
    check_per_task_size(task_range_granularities_v, "task_range_granularities");
    if (has_task_release_ns) {
      check_per_task_size(task_release_ns_v, "task_release_ns");
    } else {
      task_release_ns_v.assign(static_cast<size_t>(num_tasks), 0);
    }
    if (has_task_resize_timeout_ns) {
      check_per_task_size(task_resize_timeout_ns_v, "task_resize_timeout_ns");
    } else {
      task_resize_timeout_ns_v.assign(static_cast<size_t>(num_tasks), 0);
    }
    if (has_task_preferred_core_begins) {
      check_per_task_size(task_preferred_core_begins_v, "task_preferred_core_begins");
    } else {
      task_preferred_core_begins_v.assign(static_cast<size_t>(num_tasks), -1);
    }
    TORCH_CHECK(static_cast<int64_t>(task_allowed_thread_offsets_v.size()) == num_tasks + 1,
                "task_allowed_thread_offsets must have num_tasks + 1 entries");
    TORCH_CHECK(task_allowed_thread_offsets_v.front() == 0, "task_allowed_thread_offsets[0] must be 0");
    TORCH_CHECK(task_allowed_thread_offsets_v.back() == static_cast<int64_t>(task_allowed_threads_v.size()),
                "last task_allowed_thread_offsets entry must equal task_allowed_threads length");
    for (int64_t task = 0; task < num_tasks; ++task) {
      TORCH_CHECK(task_stage_ids_v[static_cast<size_t>(task)] == kAsyncStageExpert,
                  "Plan V2 currently only supports whole-expert tasks: task=", task);
      const int64_t route_granularity = task_range_granularities_v[static_cast<size_t>(task)];
      TORCH_CHECK(route_granularity >= kAsyncFullExpertRange,
                  "task_range_granularities must be non-negative: task=", task,
                  " granularity=", route_granularity);
      TORCH_CHECK(route_granularity == kAsyncFullExpertRange ||
                      plan_v2->execution_mode == kAsyncExecutionStrict,
                  "route-sliced tasks currently require strict execution: task=", task);
      TORCH_CHECK(task_release_ns_v[static_cast<size_t>(task)] >= 0,
                  "task_release_ns must be non-negative: task=", task);
      TORCH_CHECK(plan_v2->execution_mode == kAsyncExecutionStrict ||
                      task_release_ns_v[static_cast<size_t>(task)] == 0,
                  "nonzero task_release_ns currently requires strict execution: task=", task);
      TORCH_CHECK(task_resize_timeout_ns_v[static_cast<size_t>(task)] >= 0,
                  "task_resize_timeout_ns must be non-negative: task=", task);
      TORCH_CHECK(task_preferred_core_begins_v[static_cast<size_t>(task)] >= -1,
                  "task_preferred_core_begins must be -1 (derive) or non-negative: task=", task);
      const int64_t placement = task_placement_modes_v[static_cast<size_t>(task)];
      TORCH_CHECK(placement == kAsyncPlacementFixed || placement == kAsyncPlacementTailPool,
                  "task_placement_modes[", task, "] has unsupported value ", placement);
      TORCH_CHECK(plan_v2->execution_mode != kAsyncExecutionStrict || placement == kAsyncPlacementFixed,
                  "strict Plan V2 requires every task placement to be fixed: task=", task);
      TORCH_CHECK(plan_v2->execution_mode != kAsyncExecutionElastic || placement == kAsyncPlacementFixed,
                  "elastic Plan V2 requires every task placement to be fixed: task=", task);
      if (plan_v2->execution_mode == kAsyncExecutionElastic) {
        const int64_t resize_point = task_resize_points_v[static_cast<size_t>(task)];
        TORCH_CHECK(resize_point == kAsyncResizeNone || resize_point == kAsyncResizeBeforeW2,
                    "elastic Plan V2 only supports W13-to-W2 resizing: task=", task,
                    " resize_point=", resize_point);
      } else {
        TORCH_CHECK(task_numa_nodes_v[static_cast<size_t>(task)] == -1,
                    "strict and tail_pool Plan V2 require task_numa_nodes=-1: task=", task);
        TORCH_CHECK(task_resize_points_v[static_cast<size_t>(task)] == kAsyncResizeNone,
                    "strict and tail_pool Plan V2 do not support resize points: task=", task);
        TORCH_CHECK(task_resize_timeout_ns_v[static_cast<size_t>(task)] == 0,
                    "strict and tail_pool Plan V2 require zero resize timeout: task=", task);
        TORCH_CHECK(task_preferred_core_begins_v[static_cast<size_t>(task)] == -1,
                    "strict and tail_pool Plan V2 require task_preferred_core_begins=-1: task=", task);
      }

      const int64_t begin = task_allowed_thread_offsets_v[static_cast<size_t>(task)];
      const int64_t end = task_allowed_thread_offsets_v[static_cast<size_t>(task + 1)];
      TORCH_CHECK(begin >= 0 && begin < end && end <= static_cast<int64_t>(task_allowed_threads_v.size()),
                  "task ", task, " allowed-width range is invalid");
      int64_t previous_width = 0;
      bool selected_allowed = false;
      bool preferred_allowed = false;
      for (int64_t idx = begin; idx < end; ++idx) {
        const int64_t width = task_allowed_threads_v[static_cast<size_t>(idx)];
        TORCH_CHECK(width > previous_width && width <= num_threads, "task ", task,
                    " allowed widths must be strictly increasing and at most num_threads");
        previous_width = width;
        selected_allowed = selected_allowed || width == task_threads_v[static_cast<size_t>(task)];
        preferred_allowed = preferred_allowed || width == task_preferred_threads_v[static_cast<size_t>(task)];
      }
      TORCH_CHECK(selected_allowed, "task_threads[", task, "] is not present in its allowed widths");
      TORCH_CHECK(preferred_allowed, "task_preferred_threads[", task, "] is not present in its allowed widths");
      TORCH_CHECK(task_min_threads_v[static_cast<size_t>(task)] ==
                      task_allowed_threads_v[static_cast<size_t>(begin)],
                  "task_min_threads[", task, "] does not match its allowed widths");
      TORCH_CHECK(task_max_threads_v[static_cast<size_t>(task)] ==
                      task_allowed_threads_v[static_cast<size_t>(end - 1)],
                  "task_max_threads[", task, "] does not match its allowed widths");
      if (plan_v2->execution_mode == kAsyncExecutionElastic) {
        const int64_t selected = task_threads_v[static_cast<size_t>(task)];
        const int64_t preferred = task_preferred_threads_v[static_cast<size_t>(task)];
        const int64_t resize_point = task_resize_points_v[static_cast<size_t>(task)];
        if (resize_point == kAsyncResizeNone) {
          TORCH_CHECK(preferred == selected, "non-resizable elastic task must keep its selected width: task=", task);
          TORCH_CHECK(task_resize_timeout_ns_v[static_cast<size_t>(task)] == 0,
                      "non-resizable elastic task must use zero timeout: task=", task);
          TORCH_CHECK(task_preferred_core_begins_v[static_cast<size_t>(task)] == -1,
                      "non-resizable elastic task must not select a preferred core begin: task=", task);
          continue;
        }
        TORCH_CHECK(preferred > selected && preferred % selected == 0,
                    "elastic task preferred width must be a larger multiple of its selected width: task=", task,
                    " selected=", selected, " preferred=", preferred);
        const int64_t core_begin = task_core_begins_v[static_cast<size_t>(task)];
        const int64_t requested_core_begin =
            task_preferred_core_begins_v[static_cast<size_t>(task)];
        const int64_t preferred_core_begin =
            requested_core_begin >= 0 ? requested_core_begin : core_begin / preferred * preferred;
        TORCH_CHECK(preferred_core_begin >= 0 && preferred_core_begin + preferred <= num_threads,
                    "elastic task preferred cohort exceeds num_threads: task=", task,
                    " cohort=[", preferred_core_begin, ", ", preferred_core_begin + preferred,
                    ") num_threads=", num_threads);
        TORCH_CHECK(preferred_core_begin % preferred == 0,
                    "elastic task preferred cohort must align to its preferred width: task=", task,
                    " core_begin=", preferred_core_begin, " preferred=", preferred);
        const int64_t core_end = core_begin + selected;
        const int64_t preferred_core_end = preferred_core_begin + preferred;
        const bool source_is_contained =
            preferred_core_begin <= core_begin && core_end <= preferred_core_end;
        const bool source_is_disjoint =
            core_end <= preferred_core_begin || preferred_core_end <= core_begin;
        TORCH_CHECK(source_is_contained || source_is_disjoint,
                    "elastic task preferred cohort must contain or be disjoint from its selected team: task=", task,
                    " selected=[", core_begin, ", ", core_end, ") preferred=[", preferred_core_begin, ", ",
                    preferred_core_end, ")");
        const int64_t expected_node = task_numa_nodes_v[static_cast<size_t>(task)];
        TORCH_CHECK(expected_node >= 0, "elastic resize requires an explicit NUMA node: task=", task);
        for (int64_t logical_core = core_begin; logical_core < core_end; ++logical_core) {
          const int64_t cpu = async_thread_pinning.cpus[static_cast<size_t>(logical_core)];
          const int64_t actual_node = cpu_numa_node(cpu);
          TORCH_CHECK(actual_node >= 0, "cannot determine NUMA node for CPU ", cpu,
                      " while validating elastic task ", task);
          TORCH_CHECK(actual_node == expected_node, "elastic selected team crosses NUMA nodes: task=", task,
                      " logical_core=", logical_core, " cpu=", cpu, " expected_node=", expected_node,
                      " actual_node=", actual_node);
        }
        for (int64_t logical_core = preferred_core_begin; logical_core < preferred_core_begin + preferred;
             ++logical_core) {
          const int64_t cpu = async_thread_pinning.cpus[static_cast<size_t>(logical_core)];
          const int64_t actual_node = cpu_numa_node(cpu);
          TORCH_CHECK(actual_node >= 0, "cannot determine NUMA node for CPU ", cpu,
                      " while validating elastic task ", task);
          TORCH_CHECK(actual_node == expected_node, "elastic cohort crosses NUMA nodes: task=", task,
                      " logical_core=", logical_core, " cpu=", cpu, " expected_node=", expected_node,
                      " actual_node=", actual_node);
        }
      }
    }
  }

  phase_begin = trace_phase_begin();
  at::Tensor ids_i64 = topk_ids.to(at::kLong).contiguous();
  at::Tensor weights_f32 = topk_weights.to(at::kFloat).contiguous();
  const int64_t* ids = ids_i64.data_ptr<int64_t>();
  const int64_t num_routes = num_tokens * top_k;
  bool use_w2_direct_route = false;
#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
  const int64_t route_element_bytes =
      use_w2_bf16_route ? static_cast<int64_t>(sizeof(uint16_t)) : static_cast<int64_t>(sizeof(float));
  use_w2_direct_route = !skip_weighted && use_sve_backend && fuse_silu && w2.N_pad == H &&
                        sve_w2_direct_route_offsets_fit(num_routes, H, w2.n_tile, route_element_bytes) &&
                        sve_w2_direct_route_enabled();
#endif
  const int64_t early_merge_policy =
      has_plan_v2 ? plan_v2->early_merge : kAsyncEarlyMergeAuto;
  const bool ready_token_merge_enabled =
      env_flag_enabled_by_default("FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE");
  if (early_merge_policy == kAsyncEarlyMergeOn) {
    TORCH_CHECK(use_w2_direct_route,
                "early_merge=True requires the SVE direct-route W2 path");
  }
  bool use_async_ready_token_merge =
      use_w2_direct_route && ready_token_merge_enabled &&
      early_merge_policy != kAsyncEarlyMergeOff;
  const bool plan_v2_tail_pool = has_plan_v2 && plan_v2->execution_mode == kAsyncExecutionTailPool;
  const bool plan_v2_elastic = has_plan_v2 && plan_v2->execution_mode == kAsyncExecutionElastic;
  if (plan_v2_elastic) {
    TORCH_CHECK(early_merge_policy != kAsyncEarlyMergeOn,
                "elastic Plan V2 does not support early_merge=True");
    TORCH_CHECK(use_sve_backend && fuse_silu,
                "elastic Plan V2 currently requires the fused SVE backend");
    TORCH_CHECK(!plan_v2->elastic_stats_out.has_value() || elastic_stats_ptr != nullptr,
                "elastic_stats_out must be a defined int64 CPU tensor");
    // Ready-token merging consumes otherwise idle workers and would make the
    // first elastic experiment's availability measurement ambiguous.
    use_async_ready_token_merge = false;
  } else {
    TORCH_CHECK(elastic_stats_ptr == nullptr, "elastic_stats_out is only valid in elastic execution mode");
  }
  int64_t async_short_pool_threads = 0;
  int64_t async_short_pool_max_rows = 12;
  if (plan_v2_tail_pool) {
    for (int64_t task = 0; task < num_tasks; ++task) {
      if (task_placement_modes_v[static_cast<size_t>(task)] != kAsyncPlacementTailPool) {
        continue;
      }
      const int64_t width = task_threads_v[static_cast<size_t>(task)];
      TORCH_CHECK(async_short_pool_threads == 0 || async_short_pool_threads == width,
                  "all tail-pool tasks must use the same selected width: task=", task, " width=", width,
                  " expected=", async_short_pool_threads);
      async_short_pool_threads = width;
    }
    TORCH_CHECK(async_short_pool_threads > 0, "tail_pool execution requires at least one pooled task");
  } else if (!has_plan_v2) {
    async_short_pool_threads = env_int_or_default("FUSED_CPP_MOE_ASYNC_SHORT_POOL_THREADS", 0);
    async_short_pool_max_rows = env_int_or_default("FUSED_CPP_MOE_ASYNC_SHORT_POOL_MAX_ROWS", 12);
  }
  const bool use_async_short_pool = async_short_pool_threads > 0;
  if (use_async_short_pool) {
    TORCH_CHECK(use_sve_backend && fuse_silu,
                "async short-expert pool currently requires the fused SVE backend");
    if (!plan_v2_tail_pool) {
      TORCH_CHECK(async_short_pool_max_rows > 0,
                  "FUSED_CPP_MOE_ASYNC_SHORT_POOL_MAX_ROWS must be positive, got ", async_short_pool_max_rows);
    }
    TORCH_CHECK(async_short_pool_threads <= num_threads && num_threads % async_short_pool_threads == 0,
                "async tail-pool width must divide num_threads: pool_threads=", async_short_pool_threads,
                " num_threads=", num_threads);
  }

  std::vector<std::vector<int64_t>> routes(static_cast<size_t>(num_experts));
  for (int64_t flat = 0; flat < num_routes; ++flat) {
    const int64_t expert = ids[flat];
    TORCH_CHECK(expert >= 0 && expert < num_experts, "topk_ids out of range: id=", expert, ", valid range [0, ",
                num_experts, ")");
    routes[static_cast<size_t>(expert)].push_back(flat);
  }
  trace_phase_end(-1, -1, -1, -1, num_routes, "route_build", phase_begin);

  phase_begin = trace_phase_begin();
  int64_t active_experts = 0;
  for (const std::vector<int64_t>& expert_routes : routes) {
    if (!expert_routes.empty()) {
      ++active_experts;
    }
  }
  TORCH_CHECK(num_tasks >= active_experts,
              "async bridge requires at least one task per active expert: tasks=",
              num_tasks, " active_experts=", active_experts);

  std::vector<int64_t> seen(static_cast<size_t>(num_experts), 0);
  std::vector<int64_t> expert_covered_rows(static_cast<size_t>(num_experts), 0);
  std::vector<int64_t> expert_route_granularities(static_cast<size_t>(num_experts), -1);
  std::vector<AsyncTaskRuntime> tasks(static_cast<size_t>(num_tasks));
  std::vector<int8_t> is_short_pool_task(static_cast<size_t>(num_tasks), int8_t{0});
  std::vector<int64_t> short_pool_task_ids;
  std::vector<ScheduledScratchUnitConfig> scratch_unit_configs;
  std::vector<int64_t> short_pool_scratch_indices;
  std::vector<int64_t> elastic_preferred_core_begins(static_cast<size_t>(num_tasks), -1);
  std::vector<int64_t> elastic_preferred_threads(static_cast<size_t>(num_tasks), 0);
  std::vector<int64_t> elastic_w2_barrier_indices(static_cast<size_t>(num_tasks), -1);
  int64_t trace_gemm_hint = 0;

  auto ensure_scratch_config = [&](int64_t core_begin, int64_t threads, int64_t rows) {
    int64_t scratch_idx = -1;
    for (size_t idx = 0; idx < scratch_unit_configs.size(); ++idx) {
      const ScheduledScratchUnitConfig& config = scratch_unit_configs[idx];
      if (config.thread_begin == core_begin && config.threads == threads) {
        scratch_idx = static_cast<int64_t>(idx);
        break;
      }
    }
    if (scratch_idx < 0) {
      scratch_idx = static_cast<int64_t>(scratch_unit_configs.size());
      scratch_unit_configs.push_back(
          ScheduledScratchUnitConfig{core_begin, threads, 0, 0, fuse_silu, use_w2_bf16_route, use_w2_direct_route});
    }
    ScheduledScratchUnitConfig& scratch_config = scratch_unit_configs[static_cast<size_t>(scratch_idx)];
    scratch_config.max_rows = std::max(scratch_config.max_rows, rows);
    scratch_config.a_reorder_stride =
        std::max(scratch_config.a_reorder_stride,
                 fuse_silu ? int64_t{0} : scheduled_a_reorder_stride(rows, threads, w13, w2));
    scratch_config.fused_packa = scratch_config.fused_packa || fuse_silu;
    scratch_config.w2_bf16_route = scratch_config.w2_bf16_route || use_w2_bf16_route;
    scratch_config.w2_direct_route = scratch_config.w2_direct_route || use_w2_direct_route;
    return scratch_idx;
  };

  auto ensure_barrier_config = [&](int64_t core_begin, int64_t threads) {
    for (size_t idx = 0; idx < scratch_unit_configs.size(); ++idx) {
      const ScheduledScratchUnitConfig& config = scratch_unit_configs[idx];
      if (config.thread_begin == core_begin && config.threads == threads) {
        return static_cast<int64_t>(idx);
      }
    }
    const int64_t scratch_idx = static_cast<int64_t>(scratch_unit_configs.size());
    ScheduledScratchUnitConfig config;
    config.thread_begin = core_begin;
    config.threads = threads;
    config.barrier_only = true;
    scratch_unit_configs.push_back(config);
    return scratch_idx;
  };

  int64_t max_short_pool_rows = 0;
  for (int64_t task = 0; task < num_tasks; ++task) {
    const int64_t expert = task_expert_ids_v[static_cast<size_t>(task)];
    const int64_t core_begin = task_core_begins_v[static_cast<size_t>(task)];
    const int64_t threads = task_threads_v[static_cast<size_t>(task)];
    const int64_t route_granularity =
        has_plan_v2 ? task_range_granularities_v[static_cast<size_t>(task)] : kAsyncFullExpertRange;
    TORCH_CHECK(expert >= 0 && expert < num_experts, "task_expert_ids[", task, "] out of range: ", expert);
    TORCH_CHECK(threads > 0, "task_threads[", task, "] must be positive, got ", threads);
    const bool pooled_by_plan =
        plan_v2_tail_pool && task_placement_modes_v[static_cast<size_t>(task)] == kAsyncPlacementTailPool;
    if (pooled_by_plan) {
      TORCH_CHECK(core_begin == -1, "tail-pool task_core_begins[", task, "] must be -1, got ", core_begin);
      TORCH_CHECK(threads == async_short_pool_threads, "tail-pool task ", task, " width mismatch: got ", threads,
                  " expected ", async_short_pool_threads);
    } else {
      TORCH_CHECK(core_begin >= 0, "fixed task_core_begins[", task, "] must be non-negative, got ", core_begin);
      TORCH_CHECK(core_begin + threads <= num_threads, "task ", task, " interval [", core_begin, ", ",
                  core_begin + threads, ") exceeds num_threads=", num_threads);
      if (has_plan_v2 && !plan_v2_elastic) {
        TORCH_CHECK(core_begin + task_max_threads_v[static_cast<size_t>(task)] <= num_threads,
                    "task ", task, " allowed widths exceed its fixed logical-core placement");
      }
    }
    const int64_t expert_rows = static_cast<int64_t>(routes[static_cast<size_t>(expert)].size());
    int64_t route_begin = 0;
    int64_t rows = expert_rows;
    if (route_granularity == kAsyncFullExpertRange) {
      TORCH_CHECK(seen[static_cast<size_t>(expert)] == 0,
                  "duplicate expert tasks require a positive route granularity: expert=", expert);
      expert_covered_rows[static_cast<size_t>(expert)] = expert_rows;
    } else {
      TORCH_CHECK(has_plan_v2 && plan_v2->execution_mode == kAsyncExecutionStrict,
                  "route-sliced tasks require strict Plan V2 execution");
      TORCH_CHECK(!pooled_by_plan, "route-sliced tasks cannot use tail-pool placement");
      int64_t& expected_granularity = expert_route_granularities[static_cast<size_t>(expert)];
      TORCH_CHECK(expected_granularity < 0 || expected_granularity == route_granularity,
                  "all route slices for expert ", expert, " must use one granularity: got ",
                  route_granularity, " expected ", expected_granularity);
      expected_granularity = route_granularity;
      route_begin = expert_covered_rows[static_cast<size_t>(expert)];
      TORCH_CHECK(route_begin < expert_rows, "route slice starts beyond expert rows: expert=", expert,
                  " route_begin=", route_begin, " rows=", expert_rows);
      rows = std::min(route_granularity, expert_rows - route_begin);
      expert_covered_rows[static_cast<size_t>(expert)] += rows;
    }
    TORCH_CHECK(rows > 0, "async task contains inactive expert ", expert);
    check_positive_int(rows, "async task rows");
    ++seen[static_cast<size_t>(expert)];
    const bool pooled_by_legacy = !has_plan_v2 && use_async_short_pool && rows <= async_short_pool_max_rows;
    const bool pooled_task = pooled_by_plan || pooled_by_legacy;
    if (pooled_task) {
      is_short_pool_task[static_cast<size_t>(task)] = int8_t{1};
      short_pool_task_ids.push_back(task);
      max_short_pool_rows = std::max(max_short_pool_rows, rows);
      tasks[static_cast<size_t>(task)] =
          AsyncTaskRuntime{expert, route_begin, rows, -1, async_short_pool_threads, -1};
      trace_gemm_hint += async_short_pool_threads * 2;
    } else {
      const int64_t scratch_idx = ensure_scratch_config(core_begin, threads, rows);
      tasks[static_cast<size_t>(task)] = AsyncTaskRuntime{expert, route_begin, rows, core_begin, threads, scratch_idx};
      trace_gemm_hint += threads * 2;
    }
  }
  if (plan_v2_elastic) {
    for (int64_t task = 0; task < num_tasks; ++task) {
      const int64_t selected = task_threads_v[static_cast<size_t>(task)];
      const bool resizable =
          task_resize_points_v[static_cast<size_t>(task)] == kAsyncResizeBeforeW2;
      const int64_t preferred =
          resizable ? task_preferred_threads_v[static_cast<size_t>(task)] : selected;
      const int64_t requested_core_begin =
          resizable ? task_preferred_core_begins_v[static_cast<size_t>(task)] : -1;
      const int64_t preferred_core_begin =
          resizable ? (requested_core_begin >= 0
                           ? requested_core_begin
                           : task_core_begins_v[static_cast<size_t>(task)] / preferred * preferred)
                    : task_core_begins_v[static_cast<size_t>(task)];
      elastic_preferred_core_begins[static_cast<size_t>(task)] = preferred_core_begin;
      elastic_preferred_threads[static_cast<size_t>(task)] = preferred;
      elastic_w2_barrier_indices[static_cast<size_t>(task)] =
          preferred == selected
              ? tasks[static_cast<size_t>(task)].scratch_index
              : ensure_barrier_config(preferred_core_begin, preferred);
    }
  }
  if (use_async_short_pool) {
    if (plan_v2_tail_pool) {
      TORCH_CHECK(!short_pool_task_ids.empty(), "tail_pool execution requires at least one pooled task");
    } else {
      TORCH_CHECK(!short_pool_task_ids.empty(),
                  "async short-expert pool found no task with rows <= ", async_short_pool_max_rows);
    }
    std::sort(short_pool_task_ids.begin(), short_pool_task_ids.end(), [&](int64_t lhs, int64_t rhs) {
      const AsyncTaskRuntime& lhs_task = tasks[static_cast<size_t>(lhs)];
      const AsyncTaskRuntime& rhs_task = tasks[static_cast<size_t>(rhs)];
      if (lhs_task.rows != rhs_task.rows) {
        return lhs_task.rows > rhs_task.rows;
      }
      return lhs_task.expert < rhs_task.expert;
    });
    const int64_t short_pool_groups = num_threads / async_short_pool_threads;
    short_pool_scratch_indices.resize(static_cast<size_t>(short_pool_groups));
    for (int64_t group = 0; group < short_pool_groups; ++group) {
      short_pool_scratch_indices[static_cast<size_t>(group)] =
          ensure_scratch_config(group * async_short_pool_threads, async_short_pool_threads, max_short_pool_rows);
    }
  }
  for (int64_t expert = 0; expert < num_experts; ++expert) {
    if (!routes[static_cast<size_t>(expert)].empty()) {
      TORCH_CHECK(seen[static_cast<size_t>(expert)] > 0, "async task plan is missing active expert ", expert);
      TORCH_CHECK(
          (seen[static_cast<size_t>(expert)] == 1) ==
              (expert_route_granularities[static_cast<size_t>(expert)] < 0),
          "an expert must use either one full-range task or multiple positive-granularity slices: expert=", expert,
          " tasks=", seen[static_cast<size_t>(expert)], " granularity=",
          expert_route_granularities[static_cast<size_t>(expert)]);
      TORCH_CHECK(expert_covered_rows[static_cast<size_t>(expert)] ==
                      static_cast<int64_t>(routes[static_cast<size_t>(expert)].size()),
                  "route slices do not cover expert ", expert, ": covered=",
                  expert_covered_rows[static_cast<size_t>(expert)], " expected=",
                  routes[static_cast<size_t>(expert)].size());
    }
  }
  if (use_async_ready_token_merge && early_merge_policy != kAsyncEarlyMergeOn) {
    double min_team_load = std::numeric_limits<double>::infinity();
    double max_team_load = 0.0;
    for (const AsyncTaskRuntime& task : tasks) {
      const double team_load = static_cast<double>(ceil_div_int64(task.rows, kKernelTile)) / task.threads;
      min_team_load = std::min(min_team_load, team_load);
      max_team_load = std::max(max_team_load, team_load);
    }
    // Without a material team-load gap there is no useful interval in which
    // idle lanes can hide merge work, so retain the contiguous post barrier.
    use_async_ready_token_merge = max_team_load >= min_team_load * 1.25;
  }
  if (env_flag_enabled("FUSED_CPP_MOE_STAGE_TIMING")) {
    std::fprintf(stderr,
                 "[fused_moe_bf16_tiled_async][ready_token_policy] requested=%lld "
                 "environment=%d direct_route=%d effective=%d\n",
                 static_cast<long long>(early_merge_policy), ready_token_merge_enabled ? 1 : 0,
                 use_w2_direct_route ? 1 : 0, use_async_ready_token_merge ? 1 : 0);
  }
  const bool use_async_ready_token_drain =
      use_async_ready_token_merge && env_flag_enabled_by_default("FUSED_CPP_MOE_ASYNC_READY_TOKEN_DRAIN");
  const int64_t async_ready_token_batch =
      use_async_ready_token_merge
          ? std::clamp<int64_t>(env_int_or_default("FUSED_CPP_MOE_ASYNC_READY_TOKEN_BATCH", 2), 1, 64)
          : 1;
  const bool use_async_ready_token_prefetch = use_async_ready_token_merge && async_ready_token_batch > 1 &&
                                              env_flag_enabled_by_default("FUSED_CPP_MOE_ASYNC_READY_TOKEN_PREFETCH");

  std::vector<std::vector<int64_t>> successors(static_cast<size_t>(num_tasks));
  std::vector<std::atomic<int64_t>> deps_remaining(static_cast<size_t>(num_tasks));
  for (int64_t task = 0; task < num_tasks; ++task) {
    const int64_t begin = task_dep_offsets_v[static_cast<size_t>(task)];
    const int64_t end = task_dep_offsets_v[static_cast<size_t>(task + 1)];
    TORCH_CHECK(begin <= end, "task_dep_offsets must be nondecreasing at task ", task);
    TORCH_CHECK(begin >= 0 && end <= static_cast<int64_t>(task_deps_v.size()), "task ", task,
                " dependency range is out of bounds");
    const bool pooled_task = is_short_pool_task[static_cast<size_t>(task)] != 0;
    TORCH_CHECK(!has_plan_v2 || !pooled_task || begin == end, "tail-pool task must not have dependencies: task=", task);
    deps_remaining[static_cast<size_t>(task)].store(pooled_task ? 0 : end - begin, std::memory_order_relaxed);
    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t dep = task_deps_v[static_cast<size_t>(idx)];
      TORCH_CHECK(dep >= 0 && dep < num_tasks, "task dependency out of range: task=", task, " dep=", dep);
      TORCH_CHECK(dep != task, "async task cannot depend on itself: task=", task);
      TORCH_CHECK(dep < task,
                  "async task dependencies must refer to earlier tasks: "
                  "task=",
                  task, " dep=", dep);
      if (!pooled_task) {
        TORCH_CHECK(is_short_pool_task[static_cast<size_t>(dep)] == 0,
                    "a non-pooled async task cannot depend on a short-pool task: task=", task, " dep=", dep);
        successors[static_cast<size_t>(dep)].push_back(task);
      }
    }
  }

  const bool use_task_release_gate =
      has_plan_v2 && std::any_of(task_release_ns_v.begin(), task_release_ns_v.end(),
                                 [](int64_t release_ns) { return release_ns > 0; });

  const bool strict_tail_steal_requested =
      env_flag_enabled("FUSED_CPP_MOE_STRICT_TAIL_STEAL");
  bool use_strict_tail_steal =
      strict_tail_steal_requested && has_plan_v2 &&
      plan_v2->execution_mode == kAsyncExecutionStrict &&
      use_sve_backend && fuse_silu && !use_async_short_pool &&
      !use_task_release_gate;
  int64_t strict_tail_steal_width = 0;
  int64_t strict_tail_steal_numa_node = -1;
  int64_t strict_tail_steal_depth = 0;
  int64_t strict_tail_steal_min_donor_tasks = 0;
  int64_t strict_tail_steal_task_count = 0;
  std::vector<AsyncStrictTailStealTeam> strict_tail_steal_teams;
  std::vector<int64_t> strict_tail_head_task_counts;
  if (use_strict_tail_steal) {
    strict_tail_steal_width = tasks.front().threads;
    use_strict_tail_steal =
        strict_tail_steal_width > 0 &&
        num_threads % strict_tail_steal_width == 0 &&
        num_threads / strict_tail_steal_width >= 2;
  }
  if (use_strict_tail_steal) {
    const int64_t team_count = num_threads / strict_tail_steal_width;
    strict_tail_steal_teams.resize(static_cast<size_t>(team_count));
    for (int64_t team = 0; team < team_count; ++team) {
      AsyncStrictTailStealTeam& runtime_team =
          strict_tail_steal_teams[static_cast<size_t>(team)];
      runtime_team.core_begin = team * strict_tail_steal_width;
      runtime_team.threads = strict_tail_steal_width;
    }
    for (int64_t task_id = 0;
         task_id < num_tasks && use_strict_tail_steal; ++task_id) {
      const AsyncTaskRuntime& task = tasks[static_cast<size_t>(task_id)];
      const bool fixed_full_expert =
          task_range_granularities_v[static_cast<size_t>(task_id)] ==
              kAsyncFullExpertRange &&
          task_placement_modes_v[static_cast<size_t>(task_id)] ==
              kAsyncPlacementFixed &&
          seen[static_cast<size_t>(task.expert)] == 1;
      const bool aligned_team =
          task.threads == strict_tail_steal_width &&
          task.core_begin >= 0 &&
          task.core_begin % strict_tail_steal_width == 0 &&
          task.core_begin + strict_tail_steal_width <= num_threads;
      if (!fixed_full_expert || !aligned_team) {
        use_strict_tail_steal = false;
        break;
      }
      const int64_t team = task.core_begin / strict_tail_steal_width;
      AsyncStrictTailStealTeam& runtime_team =
          strict_tail_steal_teams[static_cast<size_t>(team)];
      if (runtime_team.scratch_index < 0) {
        runtime_team.scratch_index = task.scratch_index;
      } else if (runtime_team.scratch_index != task.scratch_index) {
        use_strict_tail_steal = false;
        break;
      }
      runtime_team.task_ids.push_back(task_id);
    }
  }
  if (use_strict_tail_steal) {
    // The accepted dependency graph is a set of per-team resource-order
    // chains. Experts do not have data dependencies, so an idle team may
    // execute the next unstarted node from another chain.
    for (const AsyncStrictTailStealTeam& runtime_team :
         strict_tail_steal_teams) {
      if (runtime_team.task_ids.empty() || runtime_team.scratch_index < 0) {
        use_strict_tail_steal = false;
        break;
      }
      int64_t prior_task = -1;
      for (const int64_t task_id : runtime_team.task_ids) {
        const int64_t begin =
            task_dep_offsets_v[static_cast<size_t>(task_id)];
        const int64_t end =
            task_dep_offsets_v[static_cast<size_t>(task_id + 1)];
        const bool dependency_is_resource_chain =
            begin == end ||
            (prior_task >= 0 && end == begin + 1 &&
             task_deps_v[static_cast<size_t>(begin)] == prior_task);
        if (!dependency_is_resource_chain) {
          use_strict_tail_steal = false;
          break;
        }
        prior_task = task_id;
      }
      if (!use_strict_tail_steal) {
        break;
      }
    }
  }
  if (use_strict_tail_steal) {
    for (const int64_t cpu : async_thread_pinning.cpus) {
      const int64_t node = cpu_numa_node(cpu);
      if (node < 0) {
        use_strict_tail_steal = false;
        break;
      }
      if (strict_tail_steal_numa_node < 0) {
        strict_tail_steal_numa_node = node;
      } else if (strict_tail_steal_numa_node != node) {
        use_strict_tail_steal = false;
        break;
      }
    }
  }
  if (use_strict_tail_steal) {
    strict_tail_steal_depth =
        env_int_or_default("FUSED_CPP_MOE_STRICT_TAIL_STEAL_DEPTH", 2);
    strict_tail_steal_min_donor_tasks = env_int_or_default(
        "FUSED_CPP_MOE_STRICT_TAIL_STEAL_MIN_DONOR_TASKS", 2);
    use_strict_tail_steal =
        strict_tail_steal_depth > 0 &&
        strict_tail_steal_min_donor_tasks > 0;
  }
  if (use_strict_tail_steal) {
    strict_tail_head_task_counts.resize(strict_tail_steal_teams.size());
    for (size_t team = 0; team < strict_tail_steal_teams.size(); ++team) {
      const std::vector<int64_t>& team_tasks =
          strict_tail_steal_teams[team].task_ids;
      const int64_t tail_count =
          std::min<int64_t>(strict_tail_steal_depth,
                            std::max<int64_t>(
                                0, static_cast<int64_t>(team_tasks.size()) - 1));
      const int64_t head_count =
          static_cast<int64_t>(team_tasks.size()) - tail_count;
      strict_tail_head_task_counts[team] = head_count;
      strict_tail_steal_task_count += tail_count;
    }
    use_strict_tail_steal = strict_tail_steal_task_count > 0;
  }
  if (!use_strict_tail_steal) {
    strict_tail_steal_teams.clear();
    strict_tail_head_task_counts.clear();
    strict_tail_steal_task_count = 0;
  }
  if (env_flag_enabled("FUSED_CPP_MOE_STAGE_TIMING")) {
    std::fprintf(
        stderr,
        "[fused_moe_bf16_tiled_async][strict_tail_steal_policy] "
        "requested=%d effective=%d teams=%zu width=%lld numa=%lld "
        "depth=%lld min_donor_tasks=%lld tail_tasks=%lld\n",
        strict_tail_steal_requested ? 1 : 0,
        use_strict_tail_steal ? 1 : 0, strict_tail_steal_teams.size(),
        static_cast<long long>(strict_tail_steal_width),
        static_cast<long long>(strict_tail_steal_numa_node),
        static_cast<long long>(strict_tail_steal_depth),
        static_cast<long long>(strict_tail_steal_min_donor_tasks),
        static_cast<long long>(strict_tail_steal_task_count));
  }

  std::vector<std::vector<int64_t>> short_pool_group_blockers;
  if (use_async_short_pool) {
    const int64_t short_pool_groups = num_threads / async_short_pool_threads;
    short_pool_group_blockers.resize(static_cast<size_t>(short_pool_groups));
    for (int64_t task_id = 0; task_id < num_tasks; ++task_id) {
      if (is_short_pool_task[static_cast<size_t>(task_id)] != 0) {
        continue;
      }
      const AsyncTaskRuntime& task = tasks[static_cast<size_t>(task_id)];
      TORCH_CHECK(task.core_begin % async_short_pool_threads == 0 && task.threads % async_short_pool_threads == 0,
                  "async short-expert pool requires long-task intervals aligned to the pool width: task=", task_id,
                  " core_begin=", task.core_begin, " threads=", task.threads,
                  " pool_threads=", async_short_pool_threads);
      const int64_t first_group = task.core_begin / async_short_pool_threads;
      const int64_t group_count = task.threads / async_short_pool_threads;
      for (int64_t group = first_group; group < first_group + group_count; ++group) {
        short_pool_group_blockers[static_cast<size_t>(group)].push_back(task_id);
      }
    }
  }
  trace_phase_end(-1, -1, -1, -1, num_tasks, "plan_validate", phase_begin);

  at::Tensor output = prepare_moe_output(input, w13_packed, w2_packed, topk_weights, topk_ids, out);
  uint16_t* out_bf16_ptr = bf16_data(output);
  at::Tensor route_out;
  float* route_out_ptr = nullptr;
  uint16_t* route_out_bf16_ptr = nullptr;
  if (!skip_weighted) {
    route_out =
        at::empty({num_routes, H},
                  at::TensorOptions().device(input.device()).dtype(use_w2_bf16_route ? at::kBFloat16 : at::kFloat));
    if (use_w2_bf16_route) {
      route_out_bf16_ptr = bf16_data(route_out);
    } else {
      route_out_ptr = route_out.data_ptr<float>();
    }
  }
  const uint16_t* input_ptr = bf16_data_const(input);
  const uint16_t* w13_ptr = bf16_data_const(w13.tensor);
  const uint16_t* w2_ptr = bf16_data_const(w2.tensor);

  phase_begin = trace_phase_begin();
  ScheduledScratchLease scratch_lease = resident_scheduled_scratch_pool().lease(scratch_unit_configs, w13, w2);
  const std::vector<ScheduledTeamScratch*>& scratches = scratch_lease.scratches();
  trace_phase_end(-1, -1, -1, -1, static_cast<int64_t>(scratch_unit_configs.size()), "scratch_alloc", phase_begin);

  const float* topk_w = weights_f32.data_ptr<float>();
  const int route_merge_unroll = skip_weighted ? 0 : resolve_route_merge_unroll(use_sve_backend);
  const size_t ready_token_count = static_cast<size_t>(use_async_ready_token_merge ? num_tokens : 0);
  std::vector<std::atomic<int64_t>> token_ready(ready_token_count);
  std::vector<std::atomic<int64_t>> token_merged(ready_token_count);
  for (size_t token = 0; token < ready_token_count; ++token) {
    token_ready[token].store(0, std::memory_order_relaxed);
    token_merged[token].store(0, std::memory_order_relaxed);
  }
  std::atomic<int64_t> published_ready_tokens{0};
  std::atomic<int64_t> completed_ready_token_merges{0};
  std::atomic<int64_t> expert_publication_epoch{0};

  moe_trace.prepare_thread_buffers(
      num_threads, static_cast<size_t>(trace_gemm_hint),
      static_cast<size_t>(trace_gemm_hint) * 2 + ready_token_count + static_cast<size_t>(num_threads) + 16);
  std::vector<std::atomic<int64_t>> task_states(static_cast<size_t>(num_tasks));
  for (int64_t task = 0; task < num_tasks; ++task) {
    task_states[static_cast<size_t>(task)].store(0, std::memory_order_relaxed);
  }
  std::vector<std::atomic<int64_t>> expert_slices_remaining(static_cast<size_t>(num_experts));
  std::vector<std::atomic<int64_t>> expert_completed(static_cast<size_t>(num_experts));
  for (int64_t expert = 0; expert < num_experts; ++expert) {
    expert_slices_remaining[static_cast<size_t>(expert)].store(seen[static_cast<size_t>(expert)],
                                                               std::memory_order_relaxed);
    expert_completed[static_cast<size_t>(expert)].store(0, std::memory_order_relaxed);
  }
  std::atomic<int64_t> completed_tasks{0};
  std::atomic<int64_t> next_short_pool_task{0};
  std::vector<std::atomic<int64_t>> short_pool_current_tasks(short_pool_scratch_indices.size());
  std::vector<int64_t> short_pool_tasks_by_group(short_pool_scratch_indices.size(), 0);
  for (std::atomic<int64_t>& current_task : short_pool_current_tasks) {
    current_task.store(-1, std::memory_order_relaxed);
  }
  std::vector<std::atomic<int64_t>> strict_tail_current_tasks(
      strict_tail_steal_teams.size());
  std::vector<std::atomic<int64_t>> strict_tail_assignment_epochs(
      strict_tail_steal_teams.size());
  std::vector<int64_t> strict_tail_tasks_by_team(
      strict_tail_steal_teams.size(), 0);
  std::vector<int64_t> strict_tail_stolen_by_team(
      strict_tail_steal_teams.size(), 0);
  std::vector<int64_t> strict_tail_stolen_rows_by_team(
      strict_tail_steal_teams.size(), 0);
  for (std::atomic<int64_t>& current_task : strict_tail_current_tasks) {
    current_task.store(-1, std::memory_order_relaxed);
  }
  for (std::atomic<int64_t>& epoch : strict_tail_assignment_epochs) {
    epoch.store(0, std::memory_order_relaxed);
  }

  const int64_t merge_tokens_per_owner = ceil_div_int64(num_tokens, num_threads);
  auto owned_token_range = [&](int64_t tid) -> SplitRange {
    const int64_t begin = std::min<int64_t>(num_tokens, tid * merge_tokens_per_owner);
    const int64_t end = std::min<int64_t>(num_tokens, begin + merge_tokens_per_owner);
    return SplitRange{begin, end - begin};
  };
  struct alignas(64) ReadyTokenOwnerState {
    int64_t scan_offset = 0;
  };
  std::vector<ReadyTokenOwnerState> ready_token_owner_states(
      static_cast<size_t>(use_async_ready_token_merge ? num_threads : 0));

  auto publish_ready_tokens = [&](const std::vector<int64_t>& expert_routes) {
    for (const int64_t flat : expert_routes) {
      const int64_t token = flat / top_k;
      bool ready = true;
      for (int64_t slot = 0; slot < top_k; ++slot) {
        const int64_t route_expert = ids[token * top_k + slot];
        TORCH_INTERNAL_ASSERT(seen[static_cast<size_t>(route_expert)] > 0, "ready token references inactive expert ",
                              route_expert);
        if (expert_completed[static_cast<size_t>(route_expert)].load(std::memory_order_acquire) == 0) {
          ready = false;
          break;
        }
      }
      if (!ready) {
        continue;
      }
      int64_t expected = 0;
      if (token_ready[static_cast<size_t>(token)].compare_exchange_strong(
              expected, 1, std::memory_order_release,
              std::memory_order_relaxed)) {
        published_ready_tokens.fetch_add(1, std::memory_order_release);
      }
    }
  };

  auto prefetch_ready_token = [&](int64_t token) {
    if (!use_async_ready_token_prefetch) {
      return;
    }
    const int64_t flat_begin = token * top_k;
    if (use_w2_bf16_route) {
      for (int64_t slot = 0; slot < top_k; ++slot) {
        __builtin_prefetch(route_out_bf16_ptr + (flat_begin + slot) * H, 0, 1);
      }
    } else {
      for (int64_t slot = 0; slot < top_k; ++slot) {
        __builtin_prefetch(route_out_ptr + (flat_begin + slot) * H, 0, 1);
      }
    }
    __builtin_prefetch(topk_w + flat_begin, 0, 1);
    __builtin_prefetch(out_bf16_ptr + token * H, 1, 1);
  };

  auto merge_ready_token = [&](int64_t tid, int64_t token) {
    auto worker_phase_begin = trace_phase_begin();
    merge_route_range(route_out_ptr, route_out_bf16_ptr, topk_w, out_bf16_ptr, token, token + 1, top_k, H,
                      use_w2_bf16_route, route_merge_unroll);
    token_merged[static_cast<size_t>(token)].store(1, std::memory_order_release);
    completed_ready_token_merges.fetch_add(1, std::memory_order_relaxed);
    // The trace task/group field carries the token id so owner assignment is
    // directly verifiable without adding hot-path metadata.
    trace_phase_end(tid, token, -1, -1, 1, "merge_ready_token", worker_phase_begin);
  };

  // Each logical worker owns the same contiguous token range used by final
  // merge. Only that worker reads token_merged for the range while compute is
  // active, so ready tokens retain locality without a contended global queue.
  auto try_merge_owned_ready_tokens = [&](int64_t tid, int64_t merge_limit) {
    const SplitRange owned = owned_token_range(tid);
    const int64_t batch_limit = std::min<int64_t>(async_ready_token_batch, merge_limit);
    if (owned.size <= 0 || batch_limit <= 0) {
      return false;
    }
    ReadyTokenOwnerState& owner_state = ready_token_owner_states[static_cast<size_t>(tid)];
    std::array<int64_t, 64> tokens{};
    int64_t token_count = 0;
    for (int64_t scanned = 0; scanned < owned.size && token_count < batch_limit; ++scanned) {
      const int64_t token = owned.begin + owner_state.scan_offset;
      owner_state.scan_offset = owner_state.scan_offset + 1 == owned.size ? 0 : owner_state.scan_offset + 1;
      if (token_ready[static_cast<size_t>(token)].load(std::memory_order_acquire) == 0 ||
          token_merged[static_cast<size_t>(token)].load(std::memory_order_acquire) != 0) {
        continue;
      }
      tokens[static_cast<size_t>(token_count++)] = token;
    }
    for (int64_t idx = 0; idx < token_count; ++idx) {
      if (idx + 1 < token_count) {
        prefetch_ready_token(tokens[static_cast<size_t>(idx + 1)]);
      }
      merge_ready_token(tid, tokens[static_cast<size_t>(idx)]);
    }
    return token_count > 0;
  };

  auto run_async_task = [&](int64_t tid, int64_t task_id, const AsyncTaskRuntime& task) {
    const int64_t local_tid = tid - task.core_begin;
    ScheduledTeamScratch& scratch = *scratches[static_cast<size_t>(task.scratch_index)];
    ThreadBarrier& barrier = scratch.barrier;
    const int64_t expert = task.expert;
    const int64_t rows = task.rows;
    const int64_t group_size = task.threads;
    const auto& expert_routes = routes[static_cast<size_t>(expert)];
    const int64_t* task_routes = expert_routes.data() + task.route_begin;
    uint16_t* a_reorder =
        scratch.a_reorder.empty() ? nullptr : scratch.a_reorder.data() + local_tid * scratch.a_reorder_stride;

    {
      auto worker_phase_begin = trace_phase_begin();
      if (fuse_silu) {
        const int64_t nb = ceil_div_int64(rows, kKernelTile);
        const SplitRange brange = split_evenly(nb, group_size, local_tid);
        if (use_sve_backend) {
          gather_pack_a_reorder_sve_hybrid(input_ptr, H, task_routes, top_k, scratch.packed_a.data(),
                                           static_cast<int>(rows), static_cast<int>(w13.K_pad), group_size, local_tid);
        } else {
          gather_pack_a_reorder_backend(false, input_ptr, H, task_routes, top_k, scratch.packed_a.data(),
                                        static_cast<int>(rows), static_cast<int>(w13.K_pad),
                                        static_cast<int>(brange.begin), static_cast<int>(brange.begin + brange.size));
        }
        trace_phase_end(tid, task_id, local_tid, expert, brange.size * kKernelTile, "gather_pack_a",
                        worker_phase_begin);
      } else {
        // Parallel gather: each team thread copies its own row slice
        // (was serial on local_tid==0 with the rest idle).
        const SplitRange grange = split_evenly(rows, group_size, local_tid);
        for (int64_t m = grange.begin; m < grange.begin + grange.size; ++m) {
          const int64_t flat = task_routes[m];
          const int64_t token = flat / top_k;
          uint16_t* dst = scratch.input.data() + m * w13.K_pad;
          std::fill(dst, dst + w13.K_pad, static_cast<uint16_t>(0));
          std::copy(input_ptr + token * H, input_ptr + token * H + H, dst);
        }
        trace_phase_end(tid, task_id, local_tid, expert, grange.size, "gather_input", worker_phase_begin);
      }
    }
    barrier.wait();

    auto worker_phase_begin = trace_phase_begin();
    if (fuse_silu) {
      if (!elide_intermediate_zero && local_tid == 0) {
        const int64_t rows_padded =
            use_sve_backend ? sve_hybrid_packed_rows(rows) : ceil_to_multiple(rows, int64_t{kKernelTile});
        std::fill(scratch.intermediate.begin(), scratch.intermediate.begin() + rows_padded * w2.K_pad,
                  static_cast<uint16_t>(0));
      }
      if (!elide_intermediate_zero) {
        barrier.wait();
      }
      TeamContext team;
      team.group_size = group_size;
      team.local_tid = local_tid;
      team.barrier = group_size > 1 ? &barrier : nullptr;
      team.a_reorder = nullptr;
      const Gemm2DSplitPlan w13_2d_plan = plan_2d_gemm_split(rows, w13.K_pad, w13.N_pad, group_size, w13.n_tile);
      team_fused_w13_silu_packed_packc_backend(
          use_sve_backend, use_fused_2d_split, team, w13_2d_plan, scratch.packed_a.data(),
          w13_ptr + expert * w13.packed_stride, scratch.intermediate.data(), static_cast<int>(rows),
          static_cast<int>(w13.K_pad), static_cast<int>(w13.N_pad), static_cast<int>(w2.K_pad), silu_poly_degree,
          w13.n_tile);
      trace_phase_end(tid, task_id, local_tid, expert, rows, "w13_fused_silu_packc", worker_phase_begin);
    } else {
      trace_dispatch_fp32_gemm_stage_split(
          moe_trace, "w13", MoeGemmStage::kW13, tid, -1, task_id, local_tid, expert, task.route_begin, rows,
          scratch.input.data(), w13_ptr + expert * w13.packed_stride, scratch.gate_up.data(), a_reorder,
          static_cast<int>(rows), static_cast<int>(w13.K_pad), static_cast<int>(w13.N_pad),
          static_cast<int>(w13.N_pad), group_size,
          w13_bias_base != nullptr ? w13_bias_base + expert * w13.N_pad : nullptr);
    }
    barrier.wait();

    if (!fuse_silu) {
      worker_phase_begin = trace_phase_begin();
      const SplitRange activation_range = split_evenly(rows, group_size, local_tid);
      activation_range_to_bf16(activation, scratch.gate_up.data(), scratch.intermediate.data(), activation_range.begin,
                               activation_range.size, w13.N_pad, w2.K_pad, F);
      trace_phase_end(tid, task_id, local_tid, expert, activation_range.size, "activation", worker_phase_begin);
      barrier.wait();
    }

    worker_phase_begin = trace_phase_begin();
    if (fuse_silu) {
      TeamContext w2team;
      w2team.group_size = group_size;
      w2team.local_tid = local_tid;
      w2team.barrier = nullptr;
      w2team.a_reorder = nullptr;
      const Gemm2DSplitPlan w2_2d_plan = plan_2d_gemm_split(rows, w2.K_pad, w2.N_pad, group_size, w2.n_tile);
      if (use_w2_direct_route) {
        if (use_w2_bf16_route) {
          team_w2_packed_sve_direct_bf16_route_backend(
              use_fused_2d_split, w2team, w2_2d_plan, scratch.intermediate.data(),
              w2_ptr + expert * w2.packed_stride, route_out_bf16_ptr, task_routes, static_cast<int>(rows),
              static_cast<int>(w2.K_pad), static_cast<int>(w2.N_pad), static_cast<int>(H), w2.n_tile);
        } else {
          team_w2_packed_sve_direct_route_backend(
              use_fused_2d_split, w2team, w2_2d_plan, scratch.intermediate.data(),
              w2_ptr + expert * w2.packed_stride, route_out_ptr, task_routes, static_cast<int>(rows),
              static_cast<int>(w2.K_pad), static_cast<int>(w2.N_pad), static_cast<int>(H), w2.n_tile);
        }
      } else if (use_w2_bf16_route) {
        team_w2_packed_bf16_sve_backend(use_fused_2d_split, w2team, w2_2d_plan, scratch.intermediate.data(),
                                        w2_ptr + expert * w2.packed_stride, scratch.down_bf16.data(),
                                        static_cast<int>(rows), static_cast<int>(w2.K_pad), static_cast<int>(w2.N_pad),
                                        static_cast<int>(w2.N_pad), w2.n_tile);
      } else {
        team_w2_packed_backend(use_sve_backend, use_fused_2d_split, w2team, w2_2d_plan, scratch.intermediate.data(),
                               w2_ptr + expert * w2.packed_stride, scratch.down.data(), static_cast<int>(rows),
                               static_cast<int>(w2.K_pad), static_cast<int>(w2.N_pad), static_cast<int>(w2.N_pad),
                               w2.n_tile);
      }
      trace_phase_end(tid, task_id, local_tid, expert, rows, use_w2_direct_route ? "w2_direct_route" : "w2_packed",
                      worker_phase_begin);
    } else {
      trace_dispatch_fp32_gemm_stage_split(
          moe_trace, "w2", MoeGemmStage::kW2, tid, -1, task_id, local_tid, expert, task.route_begin, rows,
          scratch.intermediate.data(), w2_ptr + expert * w2.packed_stride, scratch.down.data(), a_reorder,
          static_cast<int>(rows), static_cast<int>(w2.K_pad), static_cast<int>(w2.N_pad),
          static_cast<int>(w2.N_pad), group_size,
          w2_bias_base != nullptr ? w2_bias_base + expert * w2.N_pad : nullptr);
    }
    if (!use_w2_direct_route && !use_w2_n_owner_scatter) {
      barrier.wait();
    }

    if (!use_w2_direct_route) {
      worker_phase_begin = trace_phase_begin();
      with_w2_scatter_owner(static_cast<int>(w2.N_pad), group_size, local_tid, w2.n_tile, use_w2_n_owner_scatter,
                            [&](const SplitRange& h_range) {
                                  const int64_t h_begin = h_range.begin;
                                  const int64_t h_end = std::min<int64_t>(H, h_begin + h_range.size);
                                  if (h_begin >= h_end) {
                                    return;
                                  }
                                  for (int64_t m = 0; m < rows; ++m) {
                                    const int64_t flat = task_routes[m];
                                    if (use_w2_bf16_route) {
                                      const uint16_t* src = scratch.down_bf16.data() + m * w2.N_pad;
                                      uint16_t* dst = route_out_bf16_ptr + flat * H;
                                      std::copy(src + h_begin, src + h_end, dst + h_begin);
                                    } else if (skip_weighted) {
                                      const float* src = scratch.down.data() + m * w2.N_pad;
                                      uint16_t* dst = out_bf16_ptr + flat * H;
                                      convert_f32_to_bf16(src + h_begin, dst + h_begin, h_end - h_begin);
                                    } else {
                                      const float* src = scratch.down.data() + m * w2.N_pad;
                                      float* dst = route_out_ptr + flat * H;
                                      std::copy(src + h_begin, src + h_end, dst + h_begin);
                                    }
                                  }
                                });
      trace_phase_end(tid, task_id, local_tid, expert, rows, "scatter_route_out", worker_phase_begin);
    }
    barrier.wait();

    if (local_tid == 0) {
      task_states[static_cast<size_t>(task_id)].store(2);
      for (const int64_t child : successors[static_cast<size_t>(task_id)]) {
        deps_remaining[static_cast<size_t>(child)].fetch_sub(1);
      }
      const bool expert_finished =
          expert_slices_remaining[static_cast<size_t>(expert)].fetch_sub(1, std::memory_order_acq_rel) == 1;
      if (expert_finished) {
        expert_completed[static_cast<size_t>(expert)].store(1, std::memory_order_release);
      }
      if (use_async_ready_token_merge && expert_finished) {
        // The preceding acquire/release team barrier makes every N owner's W2
        // stores visible before the last route slice publishes this expert.
        // The slice counter and global RMW chain make prior slice/expert stores
        // visible before token readiness is checked.
        expert_publication_epoch.fetch_add(1, std::memory_order_acq_rel);
        publish_ready_tokens(expert_routes);
      }
      completed_tasks.fetch_add(1);
    }
    barrier.wait();
    if (use_async_ready_token_merge) {
      // Expert work remains the synchronization unit. At that safe boundary,
      // each worker advances one ready token from its own final-merge range
      // before it joins another expert task.
      try_merge_owned_ready_tokens(tid, 1);
    }
  };

  auto run_elastic_w13 = [&](int64_t tid, int64_t task_id, const AsyncTaskRuntime& task) {
    const int64_t local_tid = tid - task.core_begin;
    ScheduledTeamScratch& scratch = *scratches[static_cast<size_t>(task.scratch_index)];
    ThreadBarrier& barrier = scratch.barrier;
    const int64_t expert = task.expert;
    const int64_t rows = task.rows;
    const int64_t group_size = task.threads;
    const auto& expert_routes = routes[static_cast<size_t>(expert)];

    auto worker_phase_begin = trace_phase_begin();
    gather_pack_a_reorder_sve_hybrid(input_ptr, H, expert_routes.data(), top_k, scratch.packed_a.data(),
                                     static_cast<int>(rows), static_cast<int>(w13.K_pad), group_size, local_tid);
    trace_phase_end(tid, task_id, local_tid, expert, rows, "gather_pack_a", worker_phase_begin);
    barrier.wait();

    worker_phase_begin = trace_phase_begin();
    if (!elide_intermediate_zero && local_tid == 0) {
      const int64_t rows_padded = sve_hybrid_packed_rows(rows);
      std::fill(scratch.intermediate.begin(), scratch.intermediate.begin() + rows_padded * w2.K_pad,
                static_cast<uint16_t>(0));
    }
    if (!elide_intermediate_zero) {
      barrier.wait();
    }
    TeamContext team;
    team.group_size = group_size;
    team.local_tid = local_tid;
    team.barrier = group_size > 1 ? &barrier : nullptr;
    team.a_reorder = nullptr;
    const Gemm2DSplitPlan w13_2d_plan =
        plan_2d_gemm_split(rows, w13.K_pad, w13.N_pad, group_size, w13.n_tile);
    team_fused_w13_silu_packed_packc_backend(
        true, use_fused_2d_split, team, w13_2d_plan, scratch.packed_a.data(),
        w13_ptr + expert * w13.packed_stride, scratch.intermediate.data(), static_cast<int>(rows),
        static_cast<int>(w13.K_pad), static_cast<int>(w13.N_pad), static_cast<int>(w2.K_pad), silu_poly_degree,
        w13.n_tile);
    trace_phase_end(tid, task_id, local_tid, expert, rows, "w13_fused_silu_packc", worker_phase_begin);
    barrier.wait();
  };

  auto run_elastic_w2 = [&](int64_t tid, int64_t task_id, const AsyncTaskRuntime& task, int64_t core_begin,
                            int64_t group_size, ThreadBarrier& barrier) {
    const int64_t local_tid = tid - core_begin;
    ScheduledTeamScratch& scratch = *scratches[static_cast<size_t>(task.scratch_index)];
    const int64_t expert = task.expert;
    const int64_t rows = task.rows;
    const auto& expert_routes = routes[static_cast<size_t>(expert)];

    auto worker_phase_begin = trace_phase_begin();
    TeamContext w2team;
    w2team.group_size = group_size;
    w2team.local_tid = local_tid;
    w2team.barrier = nullptr;
    w2team.a_reorder = nullptr;
    const Gemm2DSplitPlan w2_2d_plan =
        plan_2d_gemm_split(rows, w2.K_pad, w2.N_pad, group_size, w2.n_tile);
    if (use_w2_direct_route) {
      if (use_w2_bf16_route) {
        team_w2_packed_sve_direct_bf16_route_backend(
            use_fused_2d_split, w2team, w2_2d_plan, scratch.intermediate.data(),
            w2_ptr + expert * w2.packed_stride, route_out_bf16_ptr, expert_routes.data(), static_cast<int>(rows),
            static_cast<int>(w2.K_pad), static_cast<int>(w2.N_pad), static_cast<int>(H), w2.n_tile);
      } else {
        team_w2_packed_sve_direct_route_backend(
            use_fused_2d_split, w2team, w2_2d_plan, scratch.intermediate.data(),
            w2_ptr + expert * w2.packed_stride, route_out_ptr, expert_routes.data(), static_cast<int>(rows),
            static_cast<int>(w2.K_pad), static_cast<int>(w2.N_pad), static_cast<int>(H), w2.n_tile);
      }
    } else if (use_w2_bf16_route) {
      team_w2_packed_bf16_sve_backend(
          use_fused_2d_split, w2team, w2_2d_plan, scratch.intermediate.data(),
          w2_ptr + expert * w2.packed_stride, scratch.down_bf16.data(), static_cast<int>(rows),
          static_cast<int>(w2.K_pad), static_cast<int>(w2.N_pad), static_cast<int>(w2.N_pad), w2.n_tile);
    } else {
      team_w2_packed_backend(
          true, use_fused_2d_split, w2team, w2_2d_plan, scratch.intermediate.data(),
          w2_ptr + expert * w2.packed_stride, scratch.down.data(), static_cast<int>(rows),
          static_cast<int>(w2.K_pad), static_cast<int>(w2.N_pad), static_cast<int>(w2.N_pad), w2.n_tile);
    }
    trace_phase_end(tid, task_id, local_tid, expert, rows, use_w2_direct_route ? "w2_direct_route" : "w2_packed",
                    worker_phase_begin);
    if (!use_w2_direct_route && !use_w2_n_owner_scatter) {
      barrier.wait();
    }

    if (!use_w2_direct_route) {
      worker_phase_begin = trace_phase_begin();
      with_w2_scatter_owner(static_cast<int>(w2.N_pad), group_size, local_tid, w2.n_tile, use_w2_n_owner_scatter,
                            [&](const SplitRange& h_range) {
                                  const int64_t h_begin = h_range.begin;
                                  const int64_t h_end = std::min<int64_t>(H, h_begin + h_range.size);
                                  if (h_begin >= h_end) {
                                    return;
                                  }
                                  for (int64_t m = 0; m < rows; ++m) {
                                    const int64_t flat = expert_routes[static_cast<size_t>(m)];
                                    if (use_w2_bf16_route) {
                                      const uint16_t* src = scratch.down_bf16.data() + m * w2.N_pad;
                                      uint16_t* dst = route_out_bf16_ptr + flat * H;
                                      std::copy(src + h_begin, src + h_end, dst + h_begin);
                                    } else if (skip_weighted) {
                                      const float* src = scratch.down.data() + m * w2.N_pad;
                                      uint16_t* dst = out_bf16_ptr + flat * H;
                                      convert_f32_to_bf16(src + h_begin, dst + h_begin, h_end - h_begin);
                                    } else {
                                      const float* src = scratch.down.data() + m * w2.N_pad;
                                      float* dst = route_out_ptr + flat * H;
                                      std::copy(src + h_begin, src + h_end, dst + h_begin);
                                    }
                                  }
                                });
      trace_phase_end(tid, task_id, local_tid, expert, rows, "scatter_route_out", worker_phase_begin);
    }
    barrier.wait();
  };

  const bool plan_v2_nonblocking_elastic =
      plan_v2_elastic &&
      std::all_of(task_resize_timeout_ns_v.begin(), task_resize_timeout_ns_v.end(),
                  [](int64_t timeout_ns) { return timeout_ns == 0; });

  auto steady_now_ns = []() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  };
  const int64_t task_release_epoch_ns = use_task_release_gate ? steady_now_ns() : 0;

  phase_begin = trace_phase_begin();
  if (plan_v2_nonblocking_elastic) {
    constexpr int64_t kElasticPending = 0;
    constexpr int64_t kElasticW13Running = 1;
    constexpr int64_t kElasticW2Ready = 2;
    constexpr int64_t kElasticW2Running = 3;
    constexpr int64_t kElasticComplete = 4;

    std::vector<std::atomic<int64_t>> core_owners(static_cast<size_t>(num_threads));
    std::vector<std::atomic<int64_t>> core_jobs(static_cast<size_t>(num_threads));
    std::vector<std::vector<int64_t>> w13_tasks_by_leader(static_cast<size_t>(num_threads));
    std::vector<int64_t> w2_core_begins(static_cast<size_t>(num_tasks), -1);
    std::vector<int64_t> w2_threads(static_cast<size_t>(num_tasks), 0);
    std::vector<int64_t> w2_barrier_indices(static_cast<size_t>(num_tasks), -1);
    std::vector<int64_t> w2_wait_ns(static_cast<size_t>(num_tasks), 0);
    std::vector<int8_t> w2_preferred(static_cast<size_t>(num_tasks), int8_t{0});
    for (int64_t core = 0; core < num_threads; ++core) {
      core_owners[static_cast<size_t>(core)].store(-1, std::memory_order_relaxed);
      core_jobs[static_cast<size_t>(core)].store(-1, std::memory_order_relaxed);
    }
    for (int64_t task_id = 0; task_id < num_tasks; ++task_id) {
      const AsyncTaskRuntime& task = tasks[static_cast<size_t>(task_id)];
      w13_tasks_by_leader[static_cast<size_t>(task.core_begin)].push_back(task_id);
    }

    auto release_claimed_cores = [&](int64_t task_id, int64_t core_begin, int64_t threads,
                                     const AsyncTaskRuntime& task) {
      for (int64_t core = core_begin; core < core_begin + threads; ++core) {
        if (core >= task.core_begin && core < task.core_begin + task.threads) {
          continue;
        }
        int64_t expected = task_id;
        const bool released = core_owners[static_cast<size_t>(core)].compare_exchange_strong(
            expected, -1, std::memory_order_acq_rel, std::memory_order_relaxed);
        TORCH_INTERNAL_ASSERT(released, "nonblocking elastic core ownership rollback failed");
      }
    };

    auto try_claim_team = [&](int64_t task_id, int64_t core_begin, int64_t threads,
                              const AsyncTaskRuntime& task) {
      for (int64_t core = task.core_begin; core < task.core_begin + task.threads; ++core) {
        if (core_owners[static_cast<size_t>(core)].load(std::memory_order_acquire) != task_id ||
            core_jobs[static_cast<size_t>(core)].load(std::memory_order_acquire) != task_id) {
          return false;
        }
      }
      int64_t claimed_until = core_begin;
      for (int64_t core = core_begin; core < core_begin + threads; ++core) {
        const bool selected_core = core >= task.core_begin && core < task.core_begin + task.threads;
        if (selected_core) {
          if (core_owners[static_cast<size_t>(core)].load(std::memory_order_acquire) != task_id ||
              core_jobs[static_cast<size_t>(core)].load(std::memory_order_acquire) != task_id) {
            release_claimed_cores(task_id, core_begin, claimed_until - core_begin, task);
            return false;
          }
        } else {
          if (core_jobs[static_cast<size_t>(core)].load(std::memory_order_acquire) >= 0) {
            release_claimed_cores(task_id, core_begin, claimed_until - core_begin, task);
            return false;
          }
          int64_t expected = -1;
          if (!core_owners[static_cast<size_t>(core)].compare_exchange_strong(
                  expected, task_id, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            release_claimed_cores(task_id, core_begin, claimed_until - core_begin, task);
            return false;
          }
        }
        claimed_until = core + 1;
      }
      return true;
    };

    auto assign_w2 = [&](int64_t task_id, int64_t core_begin, int64_t threads, int64_t barrier_index,
                         bool preferred, int64_t ready_ns) {
      const AsyncTaskRuntime& task = tasks[static_cast<size_t>(task_id)];
      w2_core_begins[static_cast<size_t>(task_id)] = core_begin;
      w2_threads[static_cast<size_t>(task_id)] = threads;
      w2_barrier_indices[static_cast<size_t>(task_id)] = barrier_index;
      w2_wait_ns[static_cast<size_t>(task_id)] = std::max<int64_t>(0, steady_now_ns() - ready_ns);
      w2_preferred[static_cast<size_t>(task_id)] = preferred ? int8_t{1} : int8_t{0};
      for (int64_t core = core_begin; core < core_begin + threads; ++core) {
        if (core_jobs[static_cast<size_t>(core)].load(std::memory_order_relaxed) != task_id) {
          core_jobs[static_cast<size_t>(core)].store(task_id, std::memory_order_release);
        }
      }
      const int64_t core_end = core_begin + threads;
      for (int64_t core = task.core_begin; core < task.core_begin + task.threads; ++core) {
        if (core >= core_begin && core < core_end) {
          continue;
        }
        int64_t expected_job = task_id;
        const bool job_released = core_jobs[static_cast<size_t>(core)].compare_exchange_strong(
            expected_job, -1, std::memory_order_acq_rel, std::memory_order_relaxed);
        TORCH_INTERNAL_ASSERT(job_released, "nonblocking elastic W2 source job ownership mismatch");
        int64_t expected_owner = task_id;
        const bool owner_released = core_owners[static_cast<size_t>(core)].compare_exchange_strong(
            expected_owner, -1, std::memory_order_acq_rel, std::memory_order_relaxed);
        TORCH_INTERNAL_ASSERT(owner_released, "nonblocking elastic W2 source ownership mismatch");
      }
      task_states[static_cast<size_t>(task_id)].store(kElasticW2Running, std::memory_order_release);
    };

    auto schedule_nonblocking_w2 = [&](int64_t task_id, int64_t ready_ns) {
      const AsyncTaskRuntime& task = tasks[static_cast<size_t>(task_id)];
      const bool resizable =
          task_resize_points_v[static_cast<size_t>(task_id)] == kAsyncResizeBeforeW2;
      if (resizable) {
        const int64_t preferred_core_begin =
            elastic_preferred_core_begins[static_cast<size_t>(task_id)];
        const int64_t preferred_threads = elastic_preferred_threads[static_cast<size_t>(task_id)];
        if (try_claim_team(task_id, preferred_core_begin, preferred_threads, task)) {
          assign_w2(task_id, preferred_core_begin, preferred_threads,
                    elastic_w2_barrier_indices[static_cast<size_t>(task_id)], true, ready_ns);
          return;
        }
      }
      assign_w2(task_id, task.core_begin, task.threads, task.scratch_index, false, ready_ns);
    };

    auto try_claim_w13 = [&](int64_t tid) {
      for (const int64_t task_id : w13_tasks_by_leader[static_cast<size_t>(tid)]) {
        const AsyncTaskRuntime& task = tasks[static_cast<size_t>(task_id)];
        if (task_states[static_cast<size_t>(task_id)].load(std::memory_order_acquire) != kElasticPending ||
            deps_remaining[static_cast<size_t>(task_id)].load(std::memory_order_acquire) != 0) {
          continue;
        }
        int64_t claimed = 0;
        for (; claimed < task.threads; ++claimed) {
          int64_t expected = -1;
          if (!core_owners[static_cast<size_t>(task.core_begin + claimed)].compare_exchange_strong(
                  expected, task_id, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            break;
          }
        }
        if (claimed != task.threads) {
          for (int64_t offset = 0; offset < claimed; ++offset) {
            int64_t expected = task_id;
            const bool released =
                core_owners[static_cast<size_t>(task.core_begin + offset)].compare_exchange_strong(
                    expected, -1, std::memory_order_acq_rel, std::memory_order_relaxed);
            TORCH_INTERNAL_ASSERT(released, "nonblocking elastic W13 ownership rollback failed");
          }
          continue;
        }
        task_states[static_cast<size_t>(task_id)].store(kElasticW13Running, std::memory_order_release);
        for (int64_t core = task.core_begin; core < task.core_begin + task.threads; ++core) {
          core_jobs[static_cast<size_t>(core)].store(task_id, std::memory_order_release);
        }
        return task_id;
      }
      return int64_t{-1};
    };

    run_fixed_threads(num_threads, [&](int64_t tid) {
      while (completed_tasks.load(std::memory_order_acquire) < num_tasks) {
        int64_t selected_task = core_jobs[static_cast<size_t>(tid)].load(std::memory_order_acquire);
        if (selected_task < 0 && !w13_tasks_by_leader[static_cast<size_t>(tid)].empty()) {
          selected_task = try_claim_w13(tid);
        }
        if (selected_task < 0) {
          std::this_thread::yield();
          continue;
        }

        const AsyncTaskRuntime& task = tasks[static_cast<size_t>(selected_task)];
        int64_t selected_state =
            task_states[static_cast<size_t>(selected_task)].load(std::memory_order_acquire);
        if (selected_state == kElasticW13Running) {
          run_elastic_w13(tid, selected_task, task);
          if (tid == task.core_begin) {
            const int64_t ready_ns = steady_now_ns();
            task_states[static_cast<size_t>(selected_task)].store(kElasticW2Ready, std::memory_order_release);
            schedule_nonblocking_w2(selected_task, ready_ns);
          } else {
            while (task_states[static_cast<size_t>(selected_task)].load(std::memory_order_acquire) ==
                   kElasticW13Running) {
#if defined(__aarch64__)
              __asm__ __volatile__("yield" ::: "memory");
#else
              std::this_thread::yield();
#endif
            }
          }
          selected_state =
              task_states[static_cast<size_t>(selected_task)].load(std::memory_order_acquire);
        }
        if (selected_state != kElasticW2Running) {
          std::this_thread::yield();
          continue;
        }

        const int64_t w2_core_begin = w2_core_begins[static_cast<size_t>(selected_task)];
        const int64_t w2_group_size = w2_threads[static_cast<size_t>(selected_task)];
        const int64_t barrier_index = w2_barrier_indices[static_cast<size_t>(selected_task)];
        if (tid < w2_core_begin || tid >= w2_core_begin + w2_group_size) {
          continue;
        }
        TORCH_INTERNAL_ASSERT(barrier_index >= 0, "nonblocking elastic W2 task has no barrier scratch");
        ThreadBarrier& barrier = scratches[static_cast<size_t>(barrier_index)]->barrier;
        run_elastic_w2(tid, selected_task, task, w2_core_begin, w2_group_size, barrier);
        if (tid == w2_core_begin) {
          for (int64_t core = w2_core_begin; core < w2_core_begin + w2_group_size; ++core) {
            TORCH_INTERNAL_ASSERT(
                core_jobs[static_cast<size_t>(core)].load(std::memory_order_relaxed) == selected_task,
                "nonblocking elastic W2 core job ownership mismatch");
            core_jobs[static_cast<size_t>(core)].store(-1, std::memory_order_release);
            int64_t expected = selected_task;
            const bool released = core_owners[static_cast<size_t>(core)].compare_exchange_strong(
                expected, -1, std::memory_order_acq_rel, std::memory_order_relaxed);
            TORCH_INTERNAL_ASSERT(released, "nonblocking elastic W2 core ownership mismatch");
          }
          for (const int64_t child : successors[static_cast<size_t>(selected_task)]) {
            deps_remaining[static_cast<size_t>(child)].fetch_sub(1, std::memory_order_acq_rel);
          }
          task_states[static_cast<size_t>(selected_task)].store(kElasticComplete, std::memory_order_release);
          completed_tasks.fetch_add(1, std::memory_order_acq_rel);
        }
        barrier.wait();
      }
    });

    std::vector<int64_t> elastic_stats(static_cast<size_t>(kAsyncElasticStatsCount), 0);
    for (int64_t task_id = 0; task_id < num_tasks; ++task_id) {
      if (task_resize_points_v[static_cast<size_t>(task_id)] != kAsyncResizeBeforeW2) {
        continue;
      }
      ++elastic_stats[static_cast<size_t>(kElasticEligibleTasks)];
      const int64_t wait_ns = w2_wait_ns[static_cast<size_t>(task_id)];
      elastic_stats[static_cast<size_t>(kElasticTotalWaitNs)] += wait_ns;
      elastic_stats[static_cast<size_t>(kElasticMaxWaitNs)] =
          std::max(elastic_stats[static_cast<size_t>(kElasticMaxWaitNs)], wait_ns);
      if (w2_preferred[static_cast<size_t>(task_id)] != 0) {
        ++elastic_stats[static_cast<size_t>(kElasticPreferredAssignments)];
        ++elastic_stats[static_cast<size_t>(kElasticNaturalOpportunities)];
        ++elastic_stats[static_cast<size_t>(kElasticCohortJobs)];
        elastic_stats[static_cast<size_t>(kElasticBorrowedThreads)] +=
            w2_threads[static_cast<size_t>(task_id)] - tasks[static_cast<size_t>(task_id)].threads;
      } else {
        ++elastic_stats[static_cast<size_t>(kElasticFallbackAssignments)];
      }
    }
    if (elastic_stats_ptr != nullptr) {
      std::copy(elastic_stats.begin(), elastic_stats.end(), elastic_stats_ptr);
    }
    if (env_flag_enabled("FUSED_CPP_MOE_STAGE_TIMING")) {
      std::fprintf(
          stderr,
          "[fused_moe_bf16_tiled_async][elastic] eligible=%lld preferred=%lld fallback=%lld natural=%lld "
          "waited=0 timeout=0 cohort_jobs=%lld borrowed_threads=%lld total_wait_us=%.3f max_wait_us=%.3f\n",
          static_cast<long long>(elastic_stats[static_cast<size_t>(kElasticEligibleTasks)]),
          static_cast<long long>(elastic_stats[static_cast<size_t>(kElasticPreferredAssignments)]),
          static_cast<long long>(elastic_stats[static_cast<size_t>(kElasticFallbackAssignments)]),
          static_cast<long long>(elastic_stats[static_cast<size_t>(kElasticNaturalOpportunities)]),
          static_cast<long long>(elastic_stats[static_cast<size_t>(kElasticCohortJobs)]),
          static_cast<long long>(elastic_stats[static_cast<size_t>(kElasticBorrowedThreads)]),
          static_cast<double>(elastic_stats[static_cast<size_t>(kElasticTotalWaitNs)]) / 1000.0,
          static_cast<double>(elastic_stats[static_cast<size_t>(kElasticMaxWaitNs)]) / 1000.0);
    }
  } else if (plan_v2_elastic) {
    constexpr int64_t kElasticPending = 0;
    constexpr int64_t kElasticW13Running = 1;
    constexpr int64_t kElasticW2Ready = 2;
    constexpr int64_t kElasticW2Running = 3;
    constexpr int64_t kElasticComplete = 4;

    std::mutex scheduler_mutex;
    std::vector<int64_t> core_owners(static_cast<size_t>(num_threads), -1);
    std::vector<int64_t> core_w2_jobs(static_cast<size_t>(num_threads), -1);
    std::vector<std::atomic<int64_t>> core_active_jobs(static_cast<size_t>(num_threads));
    std::vector<std::vector<int64_t>> w13_tasks_by_leader(static_cast<size_t>(num_threads));
    std::vector<int64_t> w2_core_begins(static_cast<size_t>(num_tasks), -1);
    std::vector<int64_t> w2_threads(static_cast<size_t>(num_tasks), 0);
    std::vector<int64_t> w2_barrier_indices(static_cast<size_t>(num_tasks), -1);
    std::vector<int64_t> w2_ready_since_ns(static_cast<size_t>(num_tasks), 0);
    std::vector<int8_t> resize_attempted(static_cast<size_t>(num_tasks), int8_t{0});
    std::vector<int64_t> elastic_stats(static_cast<size_t>(kAsyncElasticStatsCount), 0);
    for (std::atomic<int64_t>& active_job : core_active_jobs) {
      active_job.store(-1, std::memory_order_relaxed);
    }
    for (int64_t task = 0; task < num_tasks; ++task) {
      const AsyncTaskRuntime& runtime = tasks[static_cast<size_t>(task)];
      w13_tasks_by_leader[static_cast<size_t>(runtime.core_begin)].push_back(task);
      if (task_resize_points_v[static_cast<size_t>(task)] == kAsyncResizeBeforeW2) {
        ++elastic_stats[static_cast<size_t>(kElasticEligibleTasks)];
      }
    }

    auto steady_now_ns = []() {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(
                 std::chrono::steady_clock::now().time_since_epoch())
          .count();
    };

    auto assign_w2_locked = [&](int64_t task_id, int64_t core_begin, int64_t threads, int64_t barrier_index,
                                bool preferred, bool first_attempt, int64_t now_ns) {
      const AsyncTaskRuntime& task = tasks[static_cast<size_t>(task_id)];
      for (int64_t core = core_begin; core < core_begin + threads; ++core) {
        TORCH_INTERNAL_ASSERT(core_w2_jobs[static_cast<size_t>(core)] < 0,
                              "elastic W2 assignment overlaps an active W2 job");
        core_w2_jobs[static_cast<size_t>(core)] = task_id;
        if (core_owners[static_cast<size_t>(core)] < 0) {
          core_owners[static_cast<size_t>(core)] = task_id;
        }
      }
      const int64_t core_end = core_begin + threads;
      for (int64_t core = task.core_begin; core < task.core_begin + task.threads; ++core) {
        if (core >= core_begin && core < core_end) {
          continue;
        }
        TORCH_INTERNAL_ASSERT(core_w2_jobs[static_cast<size_t>(core)] < 0,
                              "elastic W2 migration releases a source core with an active W2 job");
        TORCH_INTERNAL_ASSERT(core_owners[static_cast<size_t>(core)] == task_id,
                              "elastic W2 migration source ownership mismatch");
        TORCH_INTERNAL_ASSERT(
            core_active_jobs[static_cast<size_t>(core)].load(std::memory_order_relaxed) < 0,
            "elastic W2 migration releases a source core with an active worker job");
        core_owners[static_cast<size_t>(core)] = -1;
      }
      w2_core_begins[static_cast<size_t>(task_id)] = core_begin;
      w2_threads[static_cast<size_t>(task_id)] = threads;
      w2_barrier_indices[static_cast<size_t>(task_id)] = barrier_index;
      if (task_resize_points_v[static_cast<size_t>(task_id)] == kAsyncResizeBeforeW2) {
        const int64_t wait_ns = std::max<int64_t>(0, now_ns - w2_ready_since_ns[static_cast<size_t>(task_id)]);
        elastic_stats[static_cast<size_t>(kElasticTotalWaitNs)] += wait_ns;
        elastic_stats[static_cast<size_t>(kElasticMaxWaitNs)] =
            std::max(elastic_stats[static_cast<size_t>(kElasticMaxWaitNs)], wait_ns);
      }
      if (preferred) {
        ++elastic_stats[static_cast<size_t>(kElasticPreferredAssignments)];
        ++elastic_stats[static_cast<size_t>(kElasticCohortJobs)];
        elastic_stats[static_cast<size_t>(kElasticBorrowedThreads)] += threads - task.threads;
        if (first_attempt) {
          ++elastic_stats[static_cast<size_t>(kElasticNaturalOpportunities)];
        } else {
          ++elastic_stats[static_cast<size_t>(kElasticWaitedPreferredAssignments)];
        }
      } else if (task_resize_points_v[static_cast<size_t>(task_id)] == kAsyncResizeBeforeW2) {
        ++elastic_stats[static_cast<size_t>(kElasticFallbackAssignments)];
        const int64_t timeout_ns = task_resize_timeout_ns_v[static_cast<size_t>(task_id)];
        if (timeout_ns > 0) {
          ++elastic_stats[static_cast<size_t>(kElasticTimeoutFallbacks)];
        }
      }
      task_states[static_cast<size_t>(task_id)].store(kElasticW2Running, std::memory_order_release);
      for (int64_t core = core_begin; core < core_begin + threads; ++core) {
        core_active_jobs[static_cast<size_t>(core)].store(task_id, std::memory_order_release);
      }
    };

    auto preferred_team_available_locked = [&](int64_t task_id, int64_t now_ns) {
      const AsyncTaskRuntime& task = tasks[static_cast<size_t>(task_id)];
      for (int64_t core = task.core_begin; core < task.core_begin + task.threads; ++core) {
        if (core_owners[static_cast<size_t>(core)] != task_id ||
            core_w2_jobs[static_cast<size_t>(core)] >= 0) {
          return false;
        }
      }
      const int64_t core_begin = elastic_preferred_core_begins[static_cast<size_t>(task_id)];
      const int64_t threads = elastic_preferred_threads[static_cast<size_t>(task_id)];
      const int64_t candidate_timeout_ns = task_resize_timeout_ns_v[static_cast<size_t>(task_id)];
      for (int64_t core = core_begin; core < core_begin + threads; ++core) {
        if (core_w2_jobs[static_cast<size_t>(core)] >= 0) {
          return false;
        }
        const int64_t owner = core_owners[static_cast<size_t>(core)];
        if (owner < 0) {
          continue;
        }
        if (owner != task_id && candidate_timeout_ns == 0) {
          return false;
        }
        if (task_states[static_cast<size_t>(owner)].load(std::memory_order_acquire) != kElasticW2Ready ||
            task_resize_points_v[static_cast<size_t>(owner)] != kAsyncResizeBeforeW2 ||
            elastic_preferred_core_begins[static_cast<size_t>(owner)] != core_begin ||
            elastic_preferred_threads[static_cast<size_t>(owner)] != threads) {
          return false;
        }
        if (owner != task_id) {
          const int64_t owner_timeout_ns = task_resize_timeout_ns_v[static_cast<size_t>(owner)];
          const int64_t owner_elapsed_ns =
              std::max<int64_t>(0, now_ns - w2_ready_since_ns[static_cast<size_t>(owner)]);
          if (owner_timeout_ns == 0 || owner_elapsed_ns >= owner_timeout_ns) {
            return false;
          }
        }
      }
      return true;
    };

    auto fallback_team_available_locked = [&](int64_t task_id) {
      const AsyncTaskRuntime& task = tasks[static_cast<size_t>(task_id)];
      for (int64_t core = task.core_begin; core < task.core_begin + task.threads; ++core) {
        if (core_w2_jobs[static_cast<size_t>(core)] >= 0 ||
            core_owners[static_cast<size_t>(core)] != task_id) {
          return false;
        }
      }
      return true;
    };

    auto schedule_ready_w2_locked = [&](int64_t now_ns) {
      bool made_progress = false;
      do {
        made_progress = false;
        for (int64_t task_id = 0; task_id < num_tasks; ++task_id) {
          if (task_states[static_cast<size_t>(task_id)].load(std::memory_order_acquire) != kElasticW2Ready) {
            continue;
          }
          const bool resizable =
              task_resize_points_v[static_cast<size_t>(task_id)] == kAsyncResizeBeforeW2;
          const bool first_attempt = resize_attempted[static_cast<size_t>(task_id)] == 0;
          if (resizable && first_attempt) {
            resize_attempted[static_cast<size_t>(task_id)] = int8_t{1};
          }
          if (resizable && preferred_team_available_locked(task_id, now_ns)) {
            assign_w2_locked(task_id, elastic_preferred_core_begins[static_cast<size_t>(task_id)],
                             elastic_preferred_threads[static_cast<size_t>(task_id)],
                             elastic_w2_barrier_indices[static_cast<size_t>(task_id)], true, first_attempt, now_ns);
            made_progress = true;
            continue;
          }
          const int64_t timeout_ns =
              resizable ? task_resize_timeout_ns_v[static_cast<size_t>(task_id)] : int64_t{0};
          const int64_t elapsed_ns =
              resizable ? std::max<int64_t>(0, now_ns - w2_ready_since_ns[static_cast<size_t>(task_id)])
                        : int64_t{0};
          if ((!resizable || timeout_ns == 0 || elapsed_ns >= timeout_ns) &&
              fallback_team_available_locked(task_id)) {
            const AsyncTaskRuntime& task = tasks[static_cast<size_t>(task_id)];
            assign_w2_locked(task_id, task.core_begin, task.threads, task.scratch_index, false, first_attempt,
                             now_ns);
            made_progress = true;
          }
        }
      } while (made_progress);
    };

    auto try_claim_w13_locked = [&](int64_t tid) {
      for (const int64_t task_id : w13_tasks_by_leader[static_cast<size_t>(tid)]) {
        const AsyncTaskRuntime& task = tasks[static_cast<size_t>(task_id)];
        if (task_states[static_cast<size_t>(task_id)].load(std::memory_order_acquire) != kElasticPending ||
            deps_remaining[static_cast<size_t>(task_id)].load(std::memory_order_acquire) != 0) {
          continue;
        }
        bool available = true;
        for (int64_t core = task.core_begin; core < task.core_begin + task.threads; ++core) {
          if (core_owners[static_cast<size_t>(core)] >= 0 || core_w2_jobs[static_cast<size_t>(core)] >= 0) {
            available = false;
            break;
          }
        }
        if (!available) {
          continue;
        }
        for (int64_t core = task.core_begin; core < task.core_begin + task.threads; ++core) {
          core_owners[static_cast<size_t>(core)] = task_id;
        }
        task_states[static_cast<size_t>(task_id)].store(kElasticW13Running, std::memory_order_release);
        for (int64_t core = task.core_begin; core < task.core_begin + task.threads; ++core) {
          core_active_jobs[static_cast<size_t>(core)].store(task_id, std::memory_order_release);
        }
        return task_id;
      }
      return int64_t{-1};
    };

    auto leader_needs_scheduler = [&](int64_t tid) {
      for (const int64_t task_id : w13_tasks_by_leader[static_cast<size_t>(tid)]) {
        const int64_t state = task_states[static_cast<size_t>(task_id)].load(std::memory_order_acquire);
        if (state == kElasticPending &&
            deps_remaining[static_cast<size_t>(task_id)].load(std::memory_order_acquire) == 0) {
          return true;
        }
        if (state == kElasticW2Ready &&
            task_resize_timeout_ns_v[static_cast<size_t>(task_id)] > 0) {
          return true;
        }
      }
      return false;
    };

    run_fixed_threads(num_threads, [&](int64_t tid) {
      while (completed_tasks.load(std::memory_order_acquire) < num_tasks) {
        int64_t selected_task = core_active_jobs[static_cast<size_t>(tid)].load(std::memory_order_acquire);
        if (selected_task < 0) {
          if (leader_needs_scheduler(tid)) {
            std::unique_lock<std::mutex> lock(scheduler_mutex, std::try_to_lock);
            if (lock.owns_lock()) {
              schedule_ready_w2_locked(steady_now_ns());
              selected_task = try_claim_w13_locked(tid);
            }
          }
          if (selected_task < 0) {
            std::this_thread::yield();
            continue;
          }
        }

        const AsyncTaskRuntime& task = tasks[static_cast<size_t>(selected_task)];
        const int64_t selected_state =
            task_states[static_cast<size_t>(selected_task)].load(std::memory_order_acquire);
        if (selected_state == kElasticW13Running) {
          run_elastic_w13(tid, selected_task, task);
          if (tid == task.core_begin) {
            std::lock_guard<std::mutex> lock(scheduler_mutex);
            for (int64_t core = task.core_begin; core < task.core_begin + task.threads; ++core) {
              TORCH_INTERNAL_ASSERT(
                  core_active_jobs[static_cast<size_t>(core)].load(std::memory_order_relaxed) == selected_task,
                  "elastic W13 core job ownership mismatch");
              core_active_jobs[static_cast<size_t>(core)].store(-1, std::memory_order_release);
            }
            w2_ready_since_ns[static_cast<size_t>(selected_task)] = steady_now_ns();
            task_states[static_cast<size_t>(selected_task)].store(kElasticW2Ready, std::memory_order_release);
            schedule_ready_w2_locked(steady_now_ns());
          } else {
            while (task_states[static_cast<size_t>(selected_task)].load(std::memory_order_acquire) ==
                   kElasticW13Running) {
#if defined(__aarch64__)
              __asm__ __volatile__("yield" ::: "memory");
#else
              std::this_thread::yield();
#endif
            }
          }
          continue;
        }
        if (selected_state != kElasticW2Running) {
          std::this_thread::yield();
          continue;
        }

        const int64_t w2_core_begin = w2_core_begins[static_cast<size_t>(selected_task)];
        const int64_t w2_group_size = w2_threads[static_cast<size_t>(selected_task)];
        const int64_t barrier_index = w2_barrier_indices[static_cast<size_t>(selected_task)];
        if (tid < w2_core_begin || tid >= w2_core_begin + w2_group_size) {
          continue;
        }
        TORCH_INTERNAL_ASSERT(barrier_index >= 0, "elastic W2 task has no barrier scratch");
        ThreadBarrier& barrier = scratches[static_cast<size_t>(barrier_index)]->barrier;
        run_elastic_w2(tid, selected_task, task, w2_core_begin, w2_group_size, barrier);
        if (tid == w2_core_begin) {
          std::lock_guard<std::mutex> lock(scheduler_mutex);
          for (int64_t core = w2_core_begin; core < w2_core_begin + w2_group_size; ++core) {
            TORCH_INTERNAL_ASSERT(core_w2_jobs[static_cast<size_t>(core)] == selected_task,
                                  "elastic W2 core job ownership mismatch");
            core_w2_jobs[static_cast<size_t>(core)] = -1;
            core_active_jobs[static_cast<size_t>(core)].store(-1, std::memory_order_release);
            if (core_owners[static_cast<size_t>(core)] == selected_task) {
              core_owners[static_cast<size_t>(core)] = -1;
            }
          }
          for (const int64_t child : successors[static_cast<size_t>(selected_task)]) {
            deps_remaining[static_cast<size_t>(child)].fetch_sub(1, std::memory_order_acq_rel);
          }
          task_states[static_cast<size_t>(selected_task)].store(kElasticComplete, std::memory_order_release);
          completed_tasks.fetch_add(1, std::memory_order_acq_rel);
          schedule_ready_w2_locked(steady_now_ns());
        }
        barrier.wait();
      }
    });

    if (elastic_stats_ptr != nullptr) {
      std::copy(elastic_stats.begin(), elastic_stats.end(), elastic_stats_ptr);
    }
    if (env_flag_enabled("FUSED_CPP_MOE_STAGE_TIMING")) {
      std::fprintf(
          stderr,
          "[fused_moe_bf16_tiled_async][elastic] eligible=%lld preferred=%lld fallback=%lld natural=%lld "
          "waited=%lld timeout=%lld cohort_jobs=%lld borrowed_threads=%lld total_wait_us=%.3f max_wait_us=%.3f\n",
          static_cast<long long>(elastic_stats[static_cast<size_t>(kElasticEligibleTasks)]),
          static_cast<long long>(elastic_stats[static_cast<size_t>(kElasticPreferredAssignments)]),
          static_cast<long long>(elastic_stats[static_cast<size_t>(kElasticFallbackAssignments)]),
          static_cast<long long>(elastic_stats[static_cast<size_t>(kElasticNaturalOpportunities)]),
          static_cast<long long>(elastic_stats[static_cast<size_t>(kElasticWaitedPreferredAssignments)]),
          static_cast<long long>(elastic_stats[static_cast<size_t>(kElasticTimeoutFallbacks)]),
          static_cast<long long>(elastic_stats[static_cast<size_t>(kElasticCohortJobs)]),
          static_cast<long long>(elastic_stats[static_cast<size_t>(kElasticBorrowedThreads)]),
          static_cast<double>(elastic_stats[static_cast<size_t>(kElasticTotalWaitNs)]) / 1000.0,
          static_cast<double>(elastic_stats[static_cast<size_t>(kElasticMaxWaitNs)]) / 1000.0);
    }
  } else if (use_strict_tail_steal) {
    run_fixed_threads(num_threads, [&](int64_t tid) {
      const int64_t team_index = tid / strict_tail_steal_width;
      const AsyncStrictTailStealTeam& runtime_team =
          strict_tail_steal_teams[static_cast<size_t>(team_index)];
      const int64_t local_tid = tid - runtime_team.core_begin;
      ScheduledTeamScratch& team_scratch =
          *scratches[static_cast<size_t>(runtime_team.scratch_index)];
      const int64_t head_task_count =
          strict_tail_head_task_counts[static_cast<size_t>(team_index)];

      // Execute the planner-owned prefix with the original fixed-team
      // protocol. Tail tasks are invisible here, so their later state-1
      // claim cannot cause the donor workers to join a migrated task.
      while (true) {
        int64_t selected_task = -1;
        for (int64_t index = 0; index < head_task_count; ++index) {
          const int64_t task_id =
              runtime_team.task_ids[static_cast<size_t>(index)];
          int64_t state =
              task_states[static_cast<size_t>(task_id)].load(
                  std::memory_order_acquire);
          if (state == 2) {
            continue;
          }
          if (state == 0) {
            if (deps_remaining[static_cast<size_t>(task_id)].load(
                    std::memory_order_acquire) != 0) {
              continue;
            }
            int64_t expected = 0;
            if (!task_states[static_cast<size_t>(task_id)]
                     .compare_exchange_strong(
                         expected, 1, std::memory_order_acq_rel,
                         std::memory_order_acquire)) {
              state = expected;
              if (state != 1) {
                continue;
              }
            }
          }
          selected_task = task_id;
          break;
        }
        if (selected_task >= 0) {
          run_async_task(tid, selected_task,
                         tasks[static_cast<size_t>(selected_task)]);
          if (local_tid == 0) {
            ++strict_tail_tasks_by_team[static_cast<size_t>(team_index)];
          }
          continue;
        }

        bool head_complete = true;
        for (int64_t index = 0; index < head_task_count; ++index) {
          const int64_t task_id =
              runtime_team.task_ids[static_cast<size_t>(index)];
          if (task_states[static_cast<size_t>(task_id)].load(
                  std::memory_order_acquire) != 2) {
            head_complete = false;
            break;
          }
        }
        if (head_complete) {
          break;
        }
        if (use_async_ready_token_merge) {
          try_merge_owned_ready_tokens(tid, 1);
        } else {
          std::this_thread::yield();
        }
      }

      const int64_t tail_running_state = 3 + team_index;

      // Keep the original decentralized join protocol for this team's own
      // suffix. Encoding the owner in the running state lets donor workers
      // distinguish a local claim from a migrated claim without a new
      // per-task handoff.
      while (true) {
        int64_t selected_task = -1;
        for (int64_t index = head_task_count;
             index < static_cast<int64_t>(runtime_team.task_ids.size());
             ++index) {
          const int64_t task_id =
              runtime_team.task_ids[static_cast<size_t>(index)];
          int64_t state =
              task_states[static_cast<size_t>(task_id)].load(
                  std::memory_order_acquire);
          if (state == 2) {
            continue;
          }
          if (state == 0) {
            int64_t expected = 0;
            if (task_states[static_cast<size_t>(task_id)]
                    .compare_exchange_strong(
                        expected, tail_running_state,
                        std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
              state = tail_running_state;
            } else {
              state = expected;
            }
          }
          if (state == tail_running_state) {
            selected_task = task_id;
            break;
          }
        }
        if (selected_task < 0) {
          break;
        }
        run_async_task(tid, selected_task,
                       tasks[static_cast<size_t>(selected_task)]);
        if (local_tid == 0) {
          ++strict_tail_tasks_by_team[static_cast<size_t>(team_index)];
        }
      }

      if (local_tid == 0) {
        while (true) {
          int64_t selected_task = -1;
          if (completed_tasks.load(std::memory_order_acquire) < num_tasks) {
            // The local suffix is complete or owned by another team. Claim
            // the first pending tail node from the heaviest peer suffix.
            for (size_t attempt = 0;
                 selected_task < 0 &&
                 attempt < strict_tail_steal_teams.size();
                 ++attempt) {
              int64_t best_task = -1;
              int64_t best_remaining_rows = -1;
              for (size_t donor = 0;
                   donor < strict_tail_steal_teams.size(); ++donor) {
                if (static_cast<int64_t>(donor) == team_index) {
                  continue;
                }
                int64_t donor_task = -1;
                int64_t remaining_tasks = 0;
                int64_t remaining_rows = 0;
                const AsyncStrictTailStealTeam& donor_team =
                    strict_tail_steal_teams[donor];
                const int64_t donor_head_count =
                    strict_tail_head_task_counts[donor];
                for (int64_t index = donor_head_count;
                     index <
                     static_cast<int64_t>(donor_team.task_ids.size());
                     ++index) {
                  const int64_t task_id =
                      donor_team.task_ids[static_cast<size_t>(index)];
                  if (task_states[static_cast<size_t>(task_id)].load(
                          std::memory_order_acquire) != 0) {
                    continue;
                  }
                  ++remaining_tasks;
                  remaining_rows +=
                      tasks[static_cast<size_t>(task_id)].rows;
                  if (donor_task < 0) {
                    donor_task = task_id;
                  }
                }
                if (remaining_tasks < strict_tail_steal_min_donor_tasks ||
                    donor_task < 0 ||
                    tasks[static_cast<size_t>(donor_task)].rows >
                        team_scratch.max_rows) {
                  continue;
                }
                if (remaining_rows > best_remaining_rows) {
                  best_task = donor_task;
                  best_remaining_rows = remaining_rows;
                }
              }
              if (best_task < 0) {
                break;
              }
              int64_t expected = 0;
              if (task_states[static_cast<size_t>(best_task)]
                      .compare_exchange_strong(
                          expected, tail_running_state,
                          std::memory_order_acq_rel,
                          std::memory_order_acquire)) {
                selected_task = best_task;
                ++strict_tail_stolen_by_team[static_cast<size_t>(team_index)];
                strict_tail_stolen_rows_by_team[static_cast<size_t>(team_index)] +=
                    tasks[static_cast<size_t>(best_task)].rows;
              }
            }
          }

          if (selected_task >= 0) {
            ++strict_tail_tasks_by_team[static_cast<size_t>(team_index)];
            strict_tail_current_tasks[static_cast<size_t>(team_index)].store(
                selected_task, std::memory_order_release);
            std::atomic<int64_t>& assignment_epoch =
                strict_tail_assignment_epochs[static_cast<size_t>(team_index)];
            assignment_epoch.fetch_add(1, std::memory_order_release);
            assignment_epoch.notify_all();
            AsyncTaskRuntime migrated_task =
                tasks[static_cast<size_t>(selected_task)];
            migrated_task.core_begin = runtime_team.core_begin;
            migrated_task.scratch_index = runtime_team.scratch_index;
            run_async_task(tid, selected_task, migrated_task);
            continue;
          }

          // No new pending task can appear after this point; task states only
          // move from pending to running to complete. Release the team now
          // instead of keeping idle workers alive until global completion.
          strict_tail_current_tasks[static_cast<size_t>(team_index)].store(
              -2, std::memory_order_release);
          std::atomic<int64_t>& assignment_epoch =
              strict_tail_assignment_epochs[static_cast<size_t>(team_index)];
          assignment_epoch.fetch_add(1, std::memory_order_release);
          assignment_epoch.notify_all();
          break;
        }
      } else {
        int64_t observed_epoch = 0;
        std::atomic<int64_t>& assignment_epoch =
            strict_tail_assignment_epochs[static_cast<size_t>(team_index)];
        while (true) {
          int64_t published_epoch =
              assignment_epoch.load(std::memory_order_acquire);
          while (published_epoch == observed_epoch) {
            assignment_epoch.wait(observed_epoch, std::memory_order_acquire);
            published_epoch =
                assignment_epoch.load(std::memory_order_acquire);
          }
          observed_epoch = published_epoch;
          const int64_t selected_task =
              strict_tail_current_tasks[static_cast<size_t>(team_index)].load(
                  std::memory_order_acquire);
          if (selected_task == -2) {
            break;
          }
          TORCH_INTERNAL_ASSERT(
              selected_task >= 0,
              "strict tail assignment published an invalid task: ",
              selected_task);
          AsyncTaskRuntime migrated_task =
              tasks[static_cast<size_t>(selected_task)];
          migrated_task.core_begin = runtime_team.core_begin;
          migrated_task.scratch_index = runtime_team.scratch_index;
          run_async_task(tid, selected_task, migrated_task);
        }
      }

      // Tail migration changes only compute placement. Preserve the existing
      // ready-token contract by letting released workers drain merge work
      // until every expert and token has been published.
      while (use_async_ready_token_drain) {
        const bool compute_complete =
            completed_tasks.load(std::memory_order_acquire) >= num_tasks;
        if (try_merge_owned_ready_tokens(
                tid, compute_complete ? async_ready_token_batch : 1)) {
          continue;
        }
        if (compute_complete) {
          TORCH_INTERNAL_ASSERT(
              published_ready_tokens.load(std::memory_order_acquire) == num_tokens,
              "strict tail stealing is missing published tokens: published=",
              published_ready_tokens.load(std::memory_order_relaxed),
              " expected=", num_tokens);
          if (completed_ready_token_merges.load(std::memory_order_acquire) >=
              num_tokens) {
            break;
          }
        }
        std::this_thread::yield();
      }
    });
  } else if (!use_async_short_pool) {
    run_fixed_threads(num_threads, [&](int64_t tid) {
      while (true) {
        const bool compute_complete = completed_tasks.load(std::memory_order_acquire) >= num_tasks;
        const int64_t elapsed_ns =
            use_task_release_gate ? steady_now_ns() - task_release_epoch_ns : 0;
        int64_t selected_task = -1;
        if (!compute_complete) {
          for (int64_t task_id = 0; task_id < num_tasks; ++task_id) {
            const AsyncTaskRuntime& task = tasks[static_cast<size_t>(task_id)];
            if (tid < task.core_begin || tid >= task.core_begin + task.threads) {
              continue;
            }
            int64_t state = task_states[static_cast<size_t>(task_id)].load();
            if (state == 2) {
              continue;
            }
            if (state == 0) {
              if (use_task_release_gate &&
                  task_release_ns_v[static_cast<size_t>(task_id)] > elapsed_ns) {
                continue;
              }
              if (deps_remaining[static_cast<size_t>(task_id)].load() != 0) {
                continue;
              }
              int64_t expected = 0;
              if (!task_states[static_cast<size_t>(task_id)].compare_exchange_strong(expected, 1)) {
                state = expected;
                if (state != 1) {
                  continue;
                }
              }
            }
            selected_task = task_id;
            break;
          }
        }
        if (selected_task >= 0) {
          run_async_task(tid, selected_task, tasks[static_cast<size_t>(selected_task)]);
          continue;
        }
        if (compute_complete && !use_async_ready_token_drain) {
          break;
        }
        if (use_async_ready_token_merge &&
            try_merge_owned_ready_tokens(tid, compute_complete ? async_ready_token_batch : 1)) {
          continue;
        }
        if (compute_complete && use_async_ready_token_drain) {
          TORCH_INTERNAL_ASSERT(published_ready_tokens.load(std::memory_order_acquire) == num_tokens,
                                "fixed-owner ready-token merge is missing published tokens: published=",
                                published_ready_tokens.load(std::memory_order_relaxed), " expected=", num_tokens);
        }
        if (compute_complete && (!use_async_ready_token_drain ||
                                 completed_ready_token_merges.load(std::memory_order_acquire) >= num_tokens)) {
          break;
        }
        std::this_thread::yield();
      }
    });
  } else {
    run_fixed_threads(num_threads, [&](int64_t tid) {
      const int64_t short_pool_group = tid / async_short_pool_threads;
      const int64_t short_pool_local_tid = tid % async_short_pool_threads;
      const int64_t short_pool_core_begin = short_pool_group * async_short_pool_threads;
      const int64_t short_pool_scratch_idx = short_pool_scratch_indices[static_cast<size_t>(short_pool_group)];
      ScheduledTeamScratch& short_pool_scratch = *scratches[static_cast<size_t>(short_pool_scratch_idx)];

      // A group joins the short queue only after every wider task covering
      // its cores is complete. Once joined, its workers remain in lockstep
      // and repeatedly claim one whole short expert.
      while (true) {
        int64_t selected_task = -1;
        for (int64_t task_id = 0; task_id < num_tasks; ++task_id) {
          if (is_short_pool_task[static_cast<size_t>(task_id)] != 0) {
            continue;
          }
          const AsyncTaskRuntime& task = tasks[static_cast<size_t>(task_id)];
          if (tid < task.core_begin || tid >= task.core_begin + task.threads) {
            continue;
          }
          int64_t state = task_states[static_cast<size_t>(task_id)].load(std::memory_order_acquire);
          if (state == 2) {
            continue;
          }
          if (state == 0) {
            if (deps_remaining[static_cast<size_t>(task_id)].load(std::memory_order_acquire) != 0) {
              continue;
            }
            int64_t expected = 0;
            if (!task_states[static_cast<size_t>(task_id)].compare_exchange_strong(
                    expected, 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
              state = expected;
              if (state != 1) {
                continue;
              }
            }
          }
          selected_task = task_id;
          break;
        }
        if (selected_task >= 0) {
          run_async_task(tid, selected_task, tasks[static_cast<size_t>(selected_task)]);
          continue;
        }

        bool group_released = true;
        for (const int64_t blocker : short_pool_group_blockers[static_cast<size_t>(short_pool_group)]) {
          if (task_states[static_cast<size_t>(blocker)].load(std::memory_order_acquire) != 2) {
            group_released = false;
            break;
          }
        }
        if (group_released) {
          break;
        }
        std::this_thread::yield();
      }

      while (true) {
        if (short_pool_local_tid == 0) {
          const int64_t queue_idx = next_short_pool_task.fetch_add(1, std::memory_order_relaxed);
          const int64_t task_id = queue_idx < static_cast<int64_t>(short_pool_task_ids.size())
                                      ? short_pool_task_ids[static_cast<size_t>(queue_idx)]
                                      : -1;
          if (task_id >= 0) {
            int64_t expected = 0;
            const bool claimed = task_states[static_cast<size_t>(task_id)].compare_exchange_strong(
                expected, 1, std::memory_order_acq_rel, std::memory_order_acquire);
            TORCH_INTERNAL_ASSERT(claimed, "short-pool task was already claimed: task=", task_id, " state=", expected);
            ++short_pool_tasks_by_group[static_cast<size_t>(short_pool_group)];
          }
          short_pool_current_tasks[static_cast<size_t>(short_pool_group)].store(task_id, std::memory_order_release);
        }
        short_pool_scratch.barrier.wait();

        const int64_t task_id =
            short_pool_current_tasks[static_cast<size_t>(short_pool_group)].load(std::memory_order_acquire);
        if (task_id < 0) {
          break;
        }
        const AsyncTaskRuntime& base_task = tasks[static_cast<size_t>(task_id)];
        const AsyncTaskRuntime pooled_task{base_task.expert,      base_task.route_begin, base_task.rows,
                                           short_pool_core_begin, async_short_pool_threads,
                                           short_pool_scratch_idx};
        run_async_task(tid, task_id, pooled_task);
      }

      while (true) {
        const bool compute_complete = completed_tasks.load(std::memory_order_acquire) >= num_tasks;
        if (compute_complete && !use_async_ready_token_drain) {
          break;
        }
        if (use_async_ready_token_merge &&
            try_merge_owned_ready_tokens(tid, compute_complete ? async_ready_token_batch : 1)) {
          continue;
        }
        if (compute_complete && use_async_ready_token_drain) {
          TORCH_INTERNAL_ASSERT(published_ready_tokens.load(std::memory_order_acquire) == num_tokens,
                                "fixed-owner ready-token merge is missing published tokens: published=",
                                published_ready_tokens.load(std::memory_order_relaxed), " expected=", num_tokens);
        }
        if (compute_complete && (!use_async_ready_token_drain ||
                                 completed_ready_token_merges.load(std::memory_order_acquire) >= num_tokens)) {
          break;
        }
        std::this_thread::yield();
      }
    });
  }
  TORCH_INTERNAL_ASSERT(
      !use_async_ready_token_drain || completed_ready_token_merges.load(std::memory_order_acquire) == num_tokens,
      "fixed-owner ready-token merge exited before draining every token");
  trace_phase_end(-1, -1, -1, -1, num_routes, "scheduled_compute", phase_begin);
  if (use_strict_tail_steal &&
      env_flag_enabled("FUSED_CPP_MOE_STAGE_TIMING")) {
    int64_t stolen_tasks = 0;
    int64_t stolen_rows = 0;
    std::fprintf(
        stderr,
        "[fused_moe_bf16_tiled_async][strict_tail_steal] "
        "teams=%zu width=%lld tasks_by_team=[",
        strict_tail_steal_teams.size(),
        static_cast<long long>(strict_tail_steal_width));
    for (size_t team = 0; team < strict_tail_tasks_by_team.size(); ++team) {
      std::fprintf(stderr, "%s%lld", team == 0 ? "" : ",",
                   static_cast<long long>(strict_tail_tasks_by_team[team]));
      stolen_tasks += strict_tail_stolen_by_team[team];
      stolen_rows += strict_tail_stolen_rows_by_team[team];
    }
    std::fprintf(stderr, "] stolen_by_team=[");
    for (size_t team = 0; team < strict_tail_stolen_by_team.size(); ++team) {
      std::fprintf(stderr, "%s%lld", team == 0 ? "" : ",",
                   static_cast<long long>(strict_tail_stolen_by_team[team]));
    }
    std::fprintf(stderr, "] stolen_tasks=%lld stolen_rows=%lld\n",
                 static_cast<long long>(stolen_tasks),
                 static_cast<long long>(stolen_rows));
  }
  if (use_async_ready_token_merge && env_flag_enabled("FUSED_CPP_MOE_STAGE_TIMING")) {
    const int64_t merged_in_worker_job = completed_ready_token_merges.load(std::memory_order_acquire);
    std::fprintf(stderr,
                 "[fused_moe_bf16_tiled_async][ready_token] assignment=fixed_owner same_job_drain=%d batch=%lld "
                 "prefetch=%d merged_in_worker_job=%lld remaining_for_final=%lld\n",
                 use_async_ready_token_drain ? 1 : 0, static_cast<long long>(async_ready_token_batch),
                 use_async_ready_token_prefetch ? 1 : 0, static_cast<long long>(merged_in_worker_job),
                 static_cast<long long>(num_tokens - merged_in_worker_job));
  }
  if (use_async_short_pool && env_flag_enabled("FUSED_CPP_MOE_STAGE_TIMING")) {
    std::fprintf(stderr,
                 "[fused_moe_bf16_tiled_async][short_pool] threads=%lld "
                 "pool_threads=%lld groups=%zu short_tasks=%zu tasks_by_group=[",
                 static_cast<long long>(num_threads), static_cast<long long>(async_short_pool_threads),
                 short_pool_tasks_by_group.size(), short_pool_task_ids.size());
    for (size_t group = 0; group < short_pool_tasks_by_group.size(); ++group) {
      std::fprintf(stderr, "%s%lld", group == 0 ? "" : ",",
                   static_cast<long long>(short_pool_tasks_by_group[group]));
    }
    std::fprintf(stderr, "]\n");
  }

  auto merge_routes = [&](int64_t tid) {
    auto worker_phase_begin = trace_phase_begin();
    const SplitRange owned = owned_token_range(tid);
    const int64_t token_begin = owned.begin;
    const int64_t token_end = owned.begin + owned.size;
    if (!use_async_ready_token_merge) {
      merge_route_range(route_out_ptr, route_out_bf16_ptr, topk_w, out_bf16_ptr, token_begin, token_end, top_k, H,
                        use_w2_bf16_route, route_merge_unroll);
    } else {
      int64_t token = token_begin;
      while (token < token_end) {
        while (token < token_end &&
               token_merged[static_cast<size_t>(token)].load(std::memory_order_acquire) != 0) {
          ++token;
        }
        const int64_t range_begin = token;
        while (token < token_end &&
               token_merged[static_cast<size_t>(token)].load(std::memory_order_acquire) == 0) {
          ++token;
        }
        if (range_begin < token) {
          merge_route_range(route_out_ptr, route_out_bf16_ptr, topk_w, out_bf16_ptr, range_begin, token, top_k, H,
                            use_w2_bf16_route, route_merge_unroll);
        }
      }
    }
    trace_phase_end(tid, -1, -1, -1, std::max<int64_t>(0, token_end - token_begin), "merge_routes", worker_phase_begin);
  };
  phase_begin = trace_phase_begin();
  if (!skip_weighted &&
      (!use_async_ready_token_merge || completed_ready_token_merges.load(std::memory_order_acquire) < num_tokens)) {
    run_fixed_threads(num_threads, merge_routes);
  }
  trace_phase_end(-1, -1, -1, -1, num_tokens, "merge_routes_total", phase_begin);

  if (moe_trace.enabled()) {
    const double e2e_ms = ::fused_cpp::profile::elapsed_ms(moe_trace_begin);
    moe_trace.write_report(use_async_short_pool ? "external_plan_async_short_pool" : "external_plan_async",
                           num_threads, num_tokens, top_k, num_experts, num_routes, H, F,
                           static_cast<size_t>(active_experts), num_tasks, num_tasks,
                           use_async_short_pool ? async_short_pool_threads : 0, e2e_ms);
  }
  return finalize_moe_output(output, out);
#endif
}

}  // namespace

at::Tensor fused_moe_bf16_tiled_async(at::Tensor input, at::Tensor w13_packed, int64_t w13_K, int64_t w13_N,
                                      at::Tensor w2_packed, int64_t w2_K, int64_t w2_N, at::Tensor topk_weights,
                                      at::Tensor topk_ids, at::Tensor task_expert_ids, at::Tensor task_core_begins,
                                      at::Tensor task_threads, at::Tensor task_dep_offsets, at::Tensor task_deps,
                                      c10::optional<at::Tensor> thread_cpu_ids, c10::optional<at::Tensor> w13_bias,
                                      c10::optional<at::Tensor> w2_bias, int64_t num_threads, std::string activation,
                                      int64_t global_num_experts, bool skip_weighted, bool fuse_silu,
                                      int64_t silu_poly_degree, int64_t gemm_backend, int64_t backend_n_tile,
                                      c10::optional<at::Tensor> out) {
  return run_fused_moe_bf16_tiled_async(
      std::move(input), std::move(w13_packed), w13_K, w13_N, std::move(w2_packed), w2_K, w2_N,
      std::move(topk_weights), std::move(topk_ids), std::move(task_expert_ids), std::move(task_core_begins),
      std::move(task_threads), std::move(task_dep_offsets), std::move(task_deps), std::move(thread_cpu_ids),
      std::move(w13_bias), std::move(w2_bias), num_threads, std::move(activation), global_num_experts, skip_weighted,
      fuse_silu, silu_poly_degree, gemm_backend, backend_n_tile, std::move(out), nullptr);
}

at::Tensor fused_moe_bf16_tiled_async_plan_v2(
    at::Tensor input, at::Tensor w13_packed, int64_t w13_K, int64_t w13_N, at::Tensor w2_packed, int64_t w2_K,
    int64_t w2_N, at::Tensor topk_weights, at::Tensor topk_ids, at::Tensor task_expert_ids,
    at::Tensor task_core_begins, at::Tensor task_threads, at::Tensor task_dep_offsets, at::Tensor task_deps,
    int64_t plan_version, int64_t execution_mode, at::Tensor task_preferred_threads, at::Tensor task_min_threads,
    at::Tensor task_max_threads, at::Tensor task_allowed_thread_offsets, at::Tensor task_allowed_threads,
    at::Tensor task_placement_modes, at::Tensor task_numa_nodes, at::Tensor task_stage_ids,
    at::Tensor task_resize_points, at::Tensor task_range_granularities,
    c10::optional<at::Tensor> thread_cpu_ids,
    c10::optional<at::Tensor> w13_bias, c10::optional<at::Tensor> w2_bias, int64_t num_threads,
    std::string activation, int64_t global_num_experts, bool skip_weighted, bool fuse_silu,
    int64_t silu_poly_degree, int64_t gemm_backend, int64_t backend_n_tile, c10::optional<at::Tensor> out,
    c10::optional<at::Tensor> task_release_ns,
    int64_t early_merge) {
  TORCH_CHECK(execution_mode != kAsyncExecutionElastic,
              "elastic Plan V2 requires fused_moe_bf16_tiled_async_plan_v2_elastic");
  const AsyncPlanV2NativeArgs plan_v2{
      plan_version,
      execution_mode,
      std::move(task_preferred_threads),
      std::move(task_min_threads),
      std::move(task_max_threads),
      std::move(task_allowed_thread_offsets),
      std::move(task_allowed_threads),
      std::move(task_placement_modes),
      std::move(task_numa_nodes),
      std::move(task_stage_ids),
      std::move(task_resize_points),
      std::move(task_range_granularities),
      std::move(task_release_ns),
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      early_merge,
  };
  return run_fused_moe_bf16_tiled_async(
      std::move(input), std::move(w13_packed), w13_K, w13_N, std::move(w2_packed), w2_K, w2_N,
      std::move(topk_weights), std::move(topk_ids), std::move(task_expert_ids), std::move(task_core_begins),
      std::move(task_threads), std::move(task_dep_offsets), std::move(task_deps), std::move(thread_cpu_ids),
      std::move(w13_bias), std::move(w2_bias), num_threads, std::move(activation), global_num_experts, skip_weighted,
      fuse_silu, silu_poly_degree, gemm_backend, backend_n_tile, std::move(out), &plan_v2);
}

at::Tensor fused_moe_bf16_tiled_async_plan_v2_elastic(
    at::Tensor input, at::Tensor w13_packed, int64_t w13_K, int64_t w13_N, at::Tensor w2_packed, int64_t w2_K,
    int64_t w2_N, at::Tensor topk_weights, at::Tensor topk_ids, at::Tensor task_expert_ids,
    at::Tensor task_core_begins, at::Tensor task_threads, at::Tensor task_dep_offsets, at::Tensor task_deps,
    int64_t plan_version, int64_t execution_mode, at::Tensor task_preferred_threads, at::Tensor task_min_threads,
    at::Tensor task_max_threads, at::Tensor task_allowed_thread_offsets, at::Tensor task_allowed_threads,
    at::Tensor task_placement_modes, at::Tensor task_numa_nodes, at::Tensor task_stage_ids,
    at::Tensor task_resize_points, at::Tensor task_range_granularities,
    c10::optional<at::Tensor> thread_cpu_ids,
    c10::optional<at::Tensor> w13_bias, c10::optional<at::Tensor> w2_bias, int64_t num_threads,
    std::string activation, int64_t global_num_experts, bool skip_weighted, bool fuse_silu,
    int64_t silu_poly_degree, int64_t gemm_backend, int64_t backend_n_tile, c10::optional<at::Tensor> out,
    c10::optional<at::Tensor> task_release_ns,
    c10::optional<at::Tensor> task_resize_timeout_ns,
    c10::optional<at::Tensor> elastic_stats_out,
    c10::optional<at::Tensor> task_preferred_core_begins,
    int64_t early_merge) {
  TORCH_CHECK(execution_mode == kAsyncExecutionElastic, "elastic Plan V2 entry point requires execution_mode=",
              kAsyncExecutionElastic, ", got ", execution_mode);
  const AsyncPlanV2NativeArgs plan_v2{
      plan_version,
      execution_mode,
      std::move(task_preferred_threads),
      std::move(task_min_threads),
      std::move(task_max_threads),
      std::move(task_allowed_thread_offsets),
      std::move(task_allowed_threads),
      std::move(task_placement_modes),
      std::move(task_numa_nodes),
      std::move(task_stage_ids),
      std::move(task_resize_points),
      std::move(task_range_granularities),
      std::move(task_release_ns),
      std::move(task_resize_timeout_ns),
      std::move(task_preferred_core_begins),
      std::move(elastic_stats_out),
      early_merge,
  };
  return run_fused_moe_bf16_tiled_async(
      std::move(input), std::move(w13_packed), w13_K, w13_N, std::move(w2_packed), w2_K, w2_N,
      std::move(topk_weights), std::move(topk_ids), std::move(task_expert_ids), std::move(task_core_begins),
      std::move(task_threads), std::move(task_dep_offsets), std::move(task_deps), std::move(thread_cpu_ids),
      std::move(w13_bias), std::move(w2_bias), num_threads, std::move(activation), global_num_experts, skip_weighted,
      fuse_silu, silu_poly_degree, gemm_backend, backend_n_tile, std::move(out), &plan_v2);
}

namespace {

struct PlannedStageRuntime {
  bool is_w13 = false;
  int64_t execution_mode = kAsyncExecutionStrict;
  int64_t pool_threads = 0;
  std::vector<AsyncTaskRuntime> tasks;
  std::vector<int8_t> is_pool_task;
  std::vector<std::vector<int64_t>> successors;
  std::vector<int64_t> initial_dependencies;
  std::vector<int64_t> pool_task_ids;
  std::vector<int64_t> pool_scratch_indices;
  std::vector<std::vector<int64_t>> pool_group_blockers;
};

template <typename EnsureScratch>
PlannedStageRuntime build_planned_stage_runtime(
    const char* stage_name, bool is_w13, int64_t execution_mode, const std::vector<int64_t>& task_expert_ids,
    const std::vector<int64_t>& task_core_begins, const std::vector<int64_t>& task_threads,
    const std::vector<int64_t>& task_dep_offsets, const std::vector<int64_t>& task_deps,
    const std::vector<int64_t>& task_placement_modes,
    const std::vector<std::vector<int64_t>>& routes, int64_t active_experts, int64_t num_threads,
    EnsureScratch&& ensure_scratch) {
  TORCH_CHECK(execution_mode == kAsyncExecutionStrict || execution_mode == kAsyncExecutionTailPool, stage_name,
              " execution_mode must be strict (", kAsyncExecutionStrict, ") or tail_pool (",
              kAsyncExecutionTailPool, "), got ", execution_mode);
  const int64_t num_tasks = static_cast<int64_t>(task_expert_ids.size());
  TORCH_CHECK(num_tasks > 0, stage_name, " plan must not be empty");
  TORCH_CHECK(num_tasks == active_experts, stage_name, " plan requires exactly one task per active expert: tasks=",
              num_tasks, " active_experts=", active_experts);
  auto check_per_task_size = [&](const std::vector<int64_t>& values, const char* name) {
    TORCH_CHECK(static_cast<int64_t>(values.size()) == num_tasks, stage_name, " ", name,
                " must have one entry per task: got ", values.size(), " vs ", num_tasks);
  };
  check_per_task_size(task_core_begins, "task_core_begins");
  check_per_task_size(task_threads, "task_threads");
  check_per_task_size(task_placement_modes, "task_placement_modes");
  TORCH_CHECK(static_cast<int64_t>(task_dep_offsets.size()) == num_tasks + 1, stage_name,
              " task_dep_offsets must have num_tasks + 1 entries");
  TORCH_CHECK(task_dep_offsets.front() == 0, stage_name, " task_dep_offsets[0] must be 0");
  TORCH_CHECK(task_dep_offsets.back() == static_cast<int64_t>(task_deps.size()), stage_name,
              " last task_dep_offsets entry must equal task_deps length");

  PlannedStageRuntime runtime;
  runtime.is_w13 = is_w13;
  runtime.execution_mode = execution_mode;
  runtime.tasks.resize(static_cast<size_t>(num_tasks));
  runtime.is_pool_task.assign(static_cast<size_t>(num_tasks), int8_t{0});
  runtime.successors.resize(static_cast<size_t>(num_tasks));
  runtime.initial_dependencies.resize(static_cast<size_t>(num_tasks), 0);

  if (execution_mode == kAsyncExecutionTailPool) {
    for (int64_t task = 0; task < num_tasks; ++task) {
      if (task_placement_modes[static_cast<size_t>(task)] != kAsyncPlacementTailPool) {
        continue;
      }
      const int64_t width = task_threads[static_cast<size_t>(task)];
      TORCH_CHECK(runtime.pool_threads == 0 || runtime.pool_threads == width, stage_name,
                  " all tail-pool tasks must use the same width: task=", task, " width=", width,
                  " expected=", runtime.pool_threads);
      runtime.pool_threads = width;
    }
    TORCH_CHECK(runtime.pool_threads > 0, stage_name, " tail_pool execution requires at least one pooled task");
    TORCH_CHECK(runtime.pool_threads <= num_threads && num_threads % runtime.pool_threads == 0, stage_name,
                " tail-pool width must divide num_threads: pool_threads=", runtime.pool_threads,
                " num_threads=", num_threads);
  }

  std::vector<int8_t> seen(routes.size(), int8_t{0});
  int64_t max_pool_rows = 0;
  for (int64_t task = 0; task < num_tasks; ++task) {
    const int64_t expert = task_expert_ids[static_cast<size_t>(task)];
    const int64_t core_begin = task_core_begins[static_cast<size_t>(task)];
    const int64_t threads = task_threads[static_cast<size_t>(task)];
    const int64_t placement = task_placement_modes[static_cast<size_t>(task)];
    TORCH_CHECK(expert >= 0 && expert < static_cast<int64_t>(routes.size()), stage_name,
                " task_expert_ids[", task, "] out of range: ", expert);
    TORCH_CHECK(seen[static_cast<size_t>(expert)] == 0, stage_name, " plan contains duplicate expert ", expert);
    TORCH_CHECK(!routes[static_cast<size_t>(expert)].empty(), stage_name, " task contains inactive expert ", expert);
    TORCH_CHECK(threads > 0, stage_name, " task_threads[", task, "] must be positive, got ", threads);
    TORCH_CHECK(placement == kAsyncPlacementFixed || placement == kAsyncPlacementTailPool, stage_name,
                " task_placement_modes[", task, "] has unsupported value ", placement);
    TORCH_CHECK(execution_mode != kAsyncExecutionStrict || placement == kAsyncPlacementFixed, stage_name,
                " strict execution requires every task placement to be fixed: task=", task);

    const int64_t rows = static_cast<int64_t>(routes[static_cast<size_t>(expert)].size());
    check_positive_int(rows, "planned-stage task rows");
    seen[static_cast<size_t>(expert)] = int8_t{1};
    if (placement == kAsyncPlacementTailPool) {
      TORCH_CHECK(execution_mode == kAsyncExecutionTailPool, stage_name,
                  " pooled placement requires tail_pool execution: task=", task);
      TORCH_CHECK(core_begin == -1, stage_name, " tail-pool task_core_begins[", task, "] must be -1, got ",
                  core_begin);
      TORCH_CHECK(threads == runtime.pool_threads, stage_name, " tail-pool task width mismatch: task=", task,
                  " width=", threads, " expected=", runtime.pool_threads);
      runtime.is_pool_task[static_cast<size_t>(task)] = int8_t{1};
      runtime.pool_task_ids.push_back(task);
      max_pool_rows = std::max(max_pool_rows, rows);
      runtime.tasks[static_cast<size_t>(task)] = AsyncTaskRuntime{expert, 0, rows, -1, runtime.pool_threads, -1};
      continue;
    }

    TORCH_CHECK(core_begin >= 0, stage_name, " fixed task_core_begins[", task,
                "] must be non-negative, got ", core_begin);
    TORCH_CHECK(core_begin < num_threads && threads <= num_threads - core_begin, stage_name, " task ", task,
                " interval exceeds num_threads: core_begin=", core_begin, " threads=", threads,
                " num_threads=", num_threads);
    const int64_t scratch_index = ensure_scratch(core_begin, threads, rows, is_w13);
    runtime.tasks[static_cast<size_t>(task)] =
        AsyncTaskRuntime{expert, 0, rows, core_begin, threads, scratch_index};
  }
  for (size_t expert = 0; expert < routes.size(); ++expert) {
    if (!routes[expert].empty()) {
      TORCH_CHECK(seen[expert] != 0, stage_name, " plan is missing active expert ", expert);
    }
  }

  std::vector<std::vector<int8_t>> ancestors(
      static_cast<size_t>(num_tasks), std::vector<int8_t>(static_cast<size_t>(num_tasks), int8_t{0}));
  for (int64_t task = 0; task < num_tasks; ++task) {
    const int64_t begin = task_dep_offsets[static_cast<size_t>(task)];
    const int64_t end = task_dep_offsets[static_cast<size_t>(task + 1)];
    TORCH_CHECK(begin >= 0 && begin <= end && end <= static_cast<int64_t>(task_deps.size()), stage_name,
                " task ", task, " dependency range is invalid");
    const bool pooled = runtime.is_pool_task[static_cast<size_t>(task)] != 0;
    TORCH_CHECK(!pooled || begin == end, stage_name, " tail-pool task must not have dependencies: task=", task);
    runtime.initial_dependencies[static_cast<size_t>(task)] = pooled ? 0 : end - begin;
    for (int64_t index = begin; index < end; ++index) {
      const int64_t dependency = task_deps[static_cast<size_t>(index)];
      TORCH_CHECK(dependency >= 0 && dependency < task, stage_name,
                  " task dependencies must refer to earlier task ids: task=", task, " dependency=", dependency);
      if (!pooled) {
        TORCH_CHECK(runtime.is_pool_task[static_cast<size_t>(dependency)] == 0, stage_name,
                    " fixed task must not depend on a tail-pool task: task=", task,
                    " dependency=", dependency);
        ancestors[static_cast<size_t>(task)][static_cast<size_t>(dependency)] = int8_t{1};
        for (int64_t ancestor = 0; ancestor < dependency; ++ancestor) {
          ancestors[static_cast<size_t>(task)][static_cast<size_t>(ancestor)] |=
              ancestors[static_cast<size_t>(dependency)][static_cast<size_t>(ancestor)];
        }
        runtime.successors[static_cast<size_t>(dependency)].push_back(task);
      }
    }
  }
  for (int64_t task = 0; task < num_tasks; ++task) {
    if (runtime.is_pool_task[static_cast<size_t>(task)] != 0) {
      continue;
    }
    const AsyncTaskRuntime& current = runtime.tasks[static_cast<size_t>(task)];
    for (int64_t prior = 0; prior < task; ++prior) {
      if (runtime.is_pool_task[static_cast<size_t>(prior)] != 0) {
        continue;
      }
      const AsyncTaskRuntime& previous = runtime.tasks[static_cast<size_t>(prior)];
      const bool overlaps =
          std::max(current.core_begin, previous.core_begin) <
          std::min(current.core_begin + current.threads, previous.core_begin + previous.threads);
      TORCH_CHECK(!overlaps || ancestors[static_cast<size_t>(task)][static_cast<size_t>(prior)] != 0,
                  stage_name, " overlapping fixed intervals require dependency ordering: prior=", prior,
                  " current=", task);
    }
  }

  if (execution_mode == kAsyncExecutionTailPool) {
    std::sort(runtime.pool_task_ids.begin(), runtime.pool_task_ids.end(), [&](int64_t lhs, int64_t rhs) {
      const AsyncTaskRuntime& lhs_task = runtime.tasks[static_cast<size_t>(lhs)];
      const AsyncTaskRuntime& rhs_task = runtime.tasks[static_cast<size_t>(rhs)];
      if (lhs_task.rows != rhs_task.rows) {
        return lhs_task.rows > rhs_task.rows;
      }
      return lhs_task.expert < rhs_task.expert;
    });
    const int64_t group_count = num_threads / runtime.pool_threads;
    runtime.pool_scratch_indices.resize(static_cast<size_t>(group_count));
    runtime.pool_group_blockers.resize(static_cast<size_t>(group_count));
    for (int64_t group = 0; group < group_count; ++group) {
      runtime.pool_scratch_indices[static_cast<size_t>(group)] =
          ensure_scratch(group * runtime.pool_threads, runtime.pool_threads, max_pool_rows, is_w13);
    }
    for (int64_t task = 0; task < num_tasks; ++task) {
      if (runtime.is_pool_task[static_cast<size_t>(task)] != 0) {
        continue;
      }
      const AsyncTaskRuntime& fixed = runtime.tasks[static_cast<size_t>(task)];
      TORCH_CHECK(fixed.core_begin % runtime.pool_threads == 0 && fixed.threads % runtime.pool_threads == 0,
                  stage_name, " fixed intervals must align to tail-pool width: task=", task,
                  " core_begin=", fixed.core_begin, " threads=", fixed.threads,
                  " pool_threads=", runtime.pool_threads);
      const int64_t first_group = fixed.core_begin / runtime.pool_threads;
      const int64_t group_count_for_task = fixed.threads / runtime.pool_threads;
      for (int64_t group = first_group; group < first_group + group_count_for_task; ++group) {
        runtime.pool_group_blockers[static_cast<size_t>(group)].push_back(task);
      }
    }
  }
  return runtime;
}

template <typename RunTask>
void execute_planned_stage(int64_t num_threads, const PlannedStageRuntime& runtime,
                           const std::vector<ScheduledTeamScratch*>& scratches, RunTask&& run_task) {
  const int64_t num_tasks = static_cast<int64_t>(runtime.tasks.size());
  std::vector<std::atomic<int64_t>> task_states(static_cast<size_t>(num_tasks));
  std::vector<std::atomic<int64_t>> dependencies(static_cast<size_t>(num_tasks));
  for (int64_t task = 0; task < num_tasks; ++task) {
    task_states[static_cast<size_t>(task)].store(0, std::memory_order_relaxed);
    dependencies[static_cast<size_t>(task)].store(
        runtime.initial_dependencies[static_cast<size_t>(task)], std::memory_order_relaxed);
  }
  std::atomic<int64_t> completed_tasks{0};

  auto run_and_complete = [&](int64_t tid, int64_t task_id, const AsyncTaskRuntime& task) {
    ScheduledTeamScratch& scratch = *scratches[static_cast<size_t>(task.scratch_index)];
    const int64_t local_tid = tid - task.core_begin;
    run_task(tid, task_id, task, scratch);
    scratch.barrier.wait();
    if (local_tid == 0) {
      task_states[static_cast<size_t>(task_id)].store(2, std::memory_order_release);
      for (const int64_t successor : runtime.successors[static_cast<size_t>(task_id)]) {
        dependencies[static_cast<size_t>(successor)].fetch_sub(1, std::memory_order_acq_rel);
      }
      completed_tasks.fetch_add(1, std::memory_order_release);
    }
    scratch.barrier.wait();
  };

  auto run_fixed_until_released = [&](int64_t tid, int64_t group) {
    while (true) {
      int64_t selected_task = -1;
      for (int64_t task_id = 0; task_id < num_tasks; ++task_id) {
        if (runtime.is_pool_task[static_cast<size_t>(task_id)] != 0) {
          continue;
        }
        const AsyncTaskRuntime& task = runtime.tasks[static_cast<size_t>(task_id)];
        if (tid < task.core_begin || tid >= task.core_begin + task.threads) {
          continue;
        }
        int64_t state = task_states[static_cast<size_t>(task_id)].load(std::memory_order_acquire);
        if (state == 2) {
          continue;
        }
        if (state == 0) {
          if (dependencies[static_cast<size_t>(task_id)].load(std::memory_order_acquire) != 0) {
            continue;
          }
          int64_t expected = 0;
          if (!task_states[static_cast<size_t>(task_id)].compare_exchange_strong(
                  expected, 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
            state = expected;
            if (state != 1) {
              continue;
            }
          }
        }
        selected_task = task_id;
        break;
      }
      if (selected_task >= 0) {
        run_and_complete(tid, selected_task, runtime.tasks[static_cast<size_t>(selected_task)]);
        continue;
      }
      if (group < 0) {
        if (completed_tasks.load(std::memory_order_acquire) == num_tasks) {
          return;
        }
      } else {
        bool released = true;
        for (const int64_t blocker : runtime.pool_group_blockers[static_cast<size_t>(group)]) {
          if (task_states[static_cast<size_t>(blocker)].load(std::memory_order_acquire) != 2) {
            released = false;
            break;
          }
        }
        if (released) {
          return;
        }
      }
      std::this_thread::yield();
    }
  };

  if (runtime.execution_mode == kAsyncExecutionStrict) {
    run_fixed_threads(num_threads, [&](int64_t tid) { run_fixed_until_released(tid, -1); });
    return;
  }

  std::atomic<int64_t> next_pool_task{0};
  std::vector<std::atomic<int64_t>> current_pool_tasks(runtime.pool_scratch_indices.size());
  for (std::atomic<int64_t>& task : current_pool_tasks) {
    task.store(-1, std::memory_order_relaxed);
  }
  run_fixed_threads(num_threads, [&](int64_t tid) {
    const int64_t group = tid / runtime.pool_threads;
    const int64_t local_tid = tid % runtime.pool_threads;
    const int64_t core_begin = group * runtime.pool_threads;
    const int64_t scratch_index = runtime.pool_scratch_indices[static_cast<size_t>(group)];
    ScheduledTeamScratch& pool_scratch = *scratches[static_cast<size_t>(scratch_index)];
    run_fixed_until_released(tid, group);

    while (true) {
      if (local_tid == 0) {
        const int64_t queue_index = next_pool_task.fetch_add(1, std::memory_order_relaxed);
        const int64_t task_id =
            queue_index < static_cast<int64_t>(runtime.pool_task_ids.size())
                ? runtime.pool_task_ids[static_cast<size_t>(queue_index)]
                : -1;
        if (task_id >= 0) {
          int64_t expected = 0;
          const bool claimed = task_states[static_cast<size_t>(task_id)].compare_exchange_strong(
              expected, 1, std::memory_order_acq_rel, std::memory_order_acquire);
          TORCH_INTERNAL_ASSERT(claimed, "planned-stage pool task was already claimed: task=", task_id,
                                " state=", expected);
        }
        current_pool_tasks[static_cast<size_t>(group)].store(task_id, std::memory_order_release);
      }
      pool_scratch.barrier.wait();
      const int64_t task_id = current_pool_tasks[static_cast<size_t>(group)].load(std::memory_order_acquire);
      if (task_id < 0) {
        break;
      }
      const AsyncTaskRuntime& base = runtime.tasks[static_cast<size_t>(task_id)];
      const AsyncTaskRuntime pooled{base.expert, base.route_begin, base.rows, core_begin,
                                    runtime.pool_threads, scratch_index};
      run_and_complete(tid, task_id, pooled);
    }
    while (completed_tasks.load(std::memory_order_acquire) < num_tasks) {
      std::this_thread::yield();
    }
  });
}

}  // namespace

at::Tensor fused_moe_bf16_tiled_planned_staged(
    at::Tensor input, at::Tensor w13_packed, int64_t w13_K, int64_t w13_N, at::Tensor w2_packed, int64_t w2_K,
    int64_t w2_N, at::Tensor topk_weights, at::Tensor topk_ids, at::Tensor w13_task_expert_ids,
    at::Tensor w13_task_core_begins, at::Tensor w13_task_threads, at::Tensor w13_task_dep_offsets,
    at::Tensor w13_task_deps, int64_t w13_execution_mode, at::Tensor w13_task_placement_modes,
    at::Tensor w2_task_expert_ids, at::Tensor w2_task_core_begins,
    at::Tensor w2_task_threads, at::Tensor w2_task_dep_offsets, at::Tensor w2_task_deps, int64_t w2_execution_mode,
    at::Tensor w2_task_placement_modes,
    c10::optional<at::Tensor> thread_cpu_ids, int64_t num_threads, int64_t global_num_experts, bool fuse_silu,
    int64_t silu_poly_degree, int64_t gemm_backend, int64_t backend_n_tile, c10::optional<at::Tensor> out) {
#ifndef __aarch64__
  TORCH_CHECK(false, "fused_moe_bf16_tiled_planned_staged requires AArch64");
#else
  const auto call_begin = ::fused_cpp::profile::now();
  check_bf16_cpu(input, "input");
  TORCH_CHECK(input.dim() == 2, "input must be 2-D [tokens, hidden]");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(topk_ids.device().is_cpu() && topk_weights.device().is_cpu(),
              "topk_ids and topk_weights must be CPU tensors");
  TORCH_CHECK(is_integer_dtype(topk_ids.scalar_type()), "topk_ids must use an integer dtype");
  TORCH_CHECK(is_floating_dtype(topk_weights.scalar_type()), "topk_weights must use a floating dtype");
  TORCH_CHECK(topk_ids.dim() == 2 && topk_weights.dim() == 2,
              "topk_ids and topk_weights must be 2-D [tokens, top_k]");
  TORCH_CHECK(topk_ids.sizes() == topk_weights.sizes(), "topk_ids and topk_weights shapes must match");
  TORCH_CHECK(topk_ids.size(0) == input.size(0), "topk first dimension must match input token count");
  TORCH_CHECK(topk_ids.size(1) > 0, "top_k must be non-zero");
  TORCH_CHECK(num_threads > 0 && num_threads <= std::numeric_limits<int>::max(),
              "num_threads must be in [1, INT_MAX], got ", num_threads);

  const ::fused_cpp::moe::MoeBackend& backend = ::fused_cpp::moe::backend_from_id(gemm_backend);
  TORCH_CHECK(backend.id == ::fused_cpp::moe::BackendId::kArmSveBf16,
              "planned staged MoE requires the SVE BF16 backend, got ", backend.name);
  TORCH_CHECK(backend_n_tile == backend.n_tile(), "MoE backend_n_tile mismatch for ", backend.name,
              ": weights use ", backend_n_tile, ", runtime uses ", backend.n_tile());
  TORCH_CHECK(fuse_silu, "planned staged MoE requires weights prepared with fuse_silu=True");
  TORCH_CHECK(silu_poly_degree == 4 || silu_poly_degree == 5 || silu_poly_degree == 6,
              "silu_poly_degree must be 4, 5, or 6, got ", silu_poly_degree);
  PackedExperts w13 = checked_packed_experts(w13_packed, w13_K, w13_N, "w13_packed", backend_n_tile);
  PackedExperts w2 = checked_packed_experts(w2_packed, w2_K, w2_N, "w2_packed", backend_n_tile);
  TORCH_CHECK(w13.E == w2.E, "w13 and w2 expert count mismatch");
  TORCH_CHECK(w13.K == input.size(1), "input hidden size mismatch: input H=", input.size(1), ", w13 K=", w13.K);
  TORCH_CHECK(w13.N % 2 == 0, "w13 N must be even, got ", w13.N);
  const int64_t F = w13.N / 2;
  const int64_t H = input.size(1);
  TORCH_CHECK(F % kKernelTile == 0, "planned staged fused SiLU requires F to be a multiple of ", kKernelTile,
              ", got ", F);
  TORCH_CHECK(w13.N_pad == 2 * F, "planned staged MoE requires interleaved W13 with N_pad=2F, got ",
              w13.N_pad, " vs ", 2 * F);
  TORCH_CHECK(w2.K == F && w2.N == H, "w2 shape mismatch: expected K=", F, " N=", H, ", got K=", w2.K,
              " N=", w2.N);
  TORCH_CHECK(w2.K_pad == F && w2.N_pad == H,
              "planned staged direct-route path requires unpadded W2 K/N: K_pad=", w2.K_pad,
              " N_pad=", w2.N_pad, " expected K=", F, " N=", H);

  const int64_t num_tokens = input.size(0);
  const int64_t top_k = topk_ids.size(1);
  if (num_tokens == 0) {
    return finalize_moe_output(prepare_moe_output(input, w13_packed, w2_packed, topk_weights, topk_ids, out), out);
  }
  const int64_t num_experts = global_num_experts < 0 ? w13.E : global_num_experts;
  TORCH_CHECK(num_experts > 0 && num_experts <= w13.E,
              "global_num_experts must be positive and no larger than packed experts: got ", num_experts,
              " packed=", w13.E);

  TORCH_CHECK(thread_cpu_ids.has_value() && thread_cpu_ids->defined(), "planned staged MoE requires thread_cpu_ids");
  ThreadPinningConfig pinning;
  pinning.cpus = tensor_to_i64_vector(*thread_cpu_ids, "thread_cpu_ids");
  TORCH_CHECK(static_cast<int64_t>(pinning.cpus.size()) == num_threads,
              "thread_cpu_ids must have exactly num_threads entries: got ", pinning.cpus.size(), " vs ",
              num_threads);
  for (size_t index = 0; index < pinning.cpus.size(); ++index) {
    TORCH_CHECK(pinning.cpus[index] >= 0, "thread_cpu_ids[", index, "] must be non-negative");
    for (size_t prior = 0; prior < index; ++prior) {
      TORCH_CHECK(pinning.cpus[prior] != pinning.cpus[index],
                  "thread_cpu_ids must not contain duplicates: index=", index, " cpu=", pinning.cpus[index]);
    }
  }
  pinning.enabled = true;
  ThreadPinningScope pinning_scope(&pinning);
  prepare_moe_threads_for_operator(num_threads);

  const auto route_build_begin = ::fused_cpp::profile::now();
  at::Tensor ids_i64 = topk_ids.to(at::kLong).contiguous();
  at::Tensor weights_f32 = topk_weights.to(at::kFloat).contiguous();
  const int64_t* ids = ids_i64.data_ptr<int64_t>();
  const int64_t num_routes = num_tokens * top_k;
  std::vector<std::vector<int64_t>> routes(static_cast<size_t>(num_experts));
  for (int64_t flat = 0; flat < num_routes; ++flat) {
    const int64_t expert = ids[flat];
    TORCH_CHECK(expert >= 0 && expert < num_experts, "topk_ids out of range: id=", expert,
                ", valid range [0, ", num_experts, ")");
    routes[static_cast<size_t>(expert)].push_back(flat);
  }
  int64_t active_experts = 0;
  std::vector<int64_t> intermediate_offsets(static_cast<size_t>(num_experts + 1), 0);
  for (int64_t expert = 0; expert < num_experts; ++expert) {
    const int64_t rows = static_cast<int64_t>(routes[static_cast<size_t>(expert)].size());
    active_experts += rows > 0 ? 1 : 0;
    const int64_t packed_rows = sve_hybrid_packed_rows(rows);
    TORCH_CHECK(packed_rows <= std::numeric_limits<int64_t>::max() / w2.K_pad,
                "planned staged intermediate size overflows int64");
    const int64_t elements = packed_rows * w2.K_pad;
    TORCH_CHECK(intermediate_offsets[static_cast<size_t>(expert)] <=
                    std::numeric_limits<int64_t>::max() - elements,
                "planned staged cumulative intermediate size overflows int64");
    intermediate_offsets[static_cast<size_t>(expert + 1)] =
        intermediate_offsets[static_cast<size_t>(expert)] + elements;
  }
  const double route_build_ms = ::fused_cpp::profile::elapsed_ms(route_build_begin);

  const auto plan_begin = ::fused_cpp::profile::now();
  std::vector<ScheduledScratchUnitConfig> scratch_configs;
  auto ensure_scratch = [&](int64_t core_begin, int64_t threads, int64_t rows, bool stage_is_w13) {
    int64_t index = -1;
    for (size_t candidate = 0; candidate < scratch_configs.size(); ++candidate) {
      if (scratch_configs[candidate].thread_begin == core_begin && scratch_configs[candidate].threads == threads) {
        index = static_cast<int64_t>(candidate);
        break;
      }
    }
    if (index < 0) {
      index = static_cast<int64_t>(scratch_configs.size());
      ScheduledScratchUnitConfig config;
      config.thread_begin = core_begin;
      config.threads = threads;
      config.external_intermediate = true;
      config.w2_direct_route = true;
      config.barrier_only = !stage_is_w13;
      scratch_configs.push_back(config);
    }
    ScheduledScratchUnitConfig& config = scratch_configs[static_cast<size_t>(index)];
    if (stage_is_w13) {
      config.max_rows = std::max(config.max_rows, rows);
      config.fused_packa = true;
      config.external_intermediate = true;
      config.w2_direct_route = true;
      config.barrier_only = false;
    }
    return index;
  };

  const PlannedStageRuntime w13_runtime = build_planned_stage_runtime(
      "W13", true, w13_execution_mode, tensor_to_i64_vector(w13_task_expert_ids, "w13_task_expert_ids"),
      tensor_to_i64_vector(w13_task_core_begins, "w13_task_core_begins"),
      tensor_to_i64_vector(w13_task_threads, "w13_task_threads"),
      tensor_to_i64_vector(w13_task_dep_offsets, "w13_task_dep_offsets"),
      tensor_to_i64_vector(w13_task_deps, "w13_task_deps"),
      tensor_to_i64_vector(w13_task_placement_modes, "w13_task_placement_modes"), routes, active_experts,
      num_threads, ensure_scratch);
  const PlannedStageRuntime w2_runtime = build_planned_stage_runtime(
      "W2", false, w2_execution_mode, tensor_to_i64_vector(w2_task_expert_ids, "w2_task_expert_ids"),
      tensor_to_i64_vector(w2_task_core_begins, "w2_task_core_begins"),
      tensor_to_i64_vector(w2_task_threads, "w2_task_threads"),
      tensor_to_i64_vector(w2_task_dep_offsets, "w2_task_dep_offsets"),
      tensor_to_i64_vector(w2_task_deps, "w2_task_deps"),
      tensor_to_i64_vector(w2_task_placement_modes, "w2_task_placement_modes"), routes, active_experts,
      num_threads, ensure_scratch);
  const double plan_validate_ms = ::fused_cpp::profile::elapsed_ms(plan_begin);

  at::Tensor output = prepare_moe_output(input, w13_packed, w2_packed, topk_weights, topk_ids, out);
  const bool use_bf16_route = sve_w2_bf16_route_enabled();
  const int64_t route_element_bytes =
      use_bf16_route ? static_cast<int64_t>(sizeof(uint16_t)) : static_cast<int64_t>(sizeof(float));
  TORCH_CHECK(sve_w2_direct_route_offsets_fit(num_routes, H, w2.n_tile, route_element_bytes),
              "planned staged direct-route offsets exceed the SVE kernel's int32 byte range");
  at::Tensor route_out =
      at::empty({num_routes, H}, input.options().dtype(use_bf16_route ? at::kBFloat16 : at::kFloat));
  at::Tensor intermediate = at::empty({intermediate_offsets.back()}, input.options());

  uint16_t* output_ptr = bf16_data(output);
  uint16_t* route_out_bf16_ptr = use_bf16_route ? bf16_data(route_out) : nullptr;
  float* route_out_f32_ptr = use_bf16_route ? nullptr : route_out.data_ptr<float>();
  uint16_t* intermediate_ptr = bf16_data(intermediate);
  const uint16_t* input_ptr = bf16_data_const(input);
  const uint16_t* w13_ptr = bf16_data_const(w13.tensor);
  const uint16_t* w2_ptr = bf16_data_const(w2.tensor);

  const auto scratch_begin = ::fused_cpp::profile::now();
  ScheduledScratchLease scratch_lease = resident_scheduled_scratch_pool().lease(scratch_configs, w13, w2);
  const std::vector<ScheduledTeamScratch*>& scratches = scratch_lease.scratches();
  const double scratch_ms = ::fused_cpp::profile::elapsed_ms(scratch_begin);

  const bool use_fused_2d_split = env_flag_enabled("FUSED_CPP_MOE_FUSED_2D_SPLIT");
  const bool elide_intermediate_zero =
      env_flag_enabled_by_default("FUSED_CPP_MOE_SVE_ELIDE_INTERMEDIATE_ZERO");
  const auto w13_begin = ::fused_cpp::profile::now();
  execute_planned_stage(num_threads, w13_runtime, scratches,
                        [&](int64_t tid, int64_t, const AsyncTaskRuntime& task,
                            ScheduledTeamScratch& scratch) {
                          const int64_t local_tid = tid - task.core_begin;
                          const int64_t expert = task.expert;
                          const int64_t rows = task.rows;
                          const auto& expert_routes = routes[static_cast<size_t>(expert)];
                          gather_pack_a_reorder_sve_hybrid(
                              input_ptr, H, expert_routes.data(), top_k, scratch.packed_a.data(),
                              static_cast<int>(rows), static_cast<int>(w13.K_pad), task.threads, local_tid);
                          scratch.barrier.wait();

                          uint16_t* expert_intermediate =
                              intermediate_ptr + intermediate_offsets[static_cast<size_t>(expert)];
                          if (!elide_intermediate_zero && local_tid == 0) {
                            std::fill(expert_intermediate,
                                      expert_intermediate +
                                          (intermediate_offsets[static_cast<size_t>(expert + 1)] -
                                           intermediate_offsets[static_cast<size_t>(expert)]),
                                      static_cast<uint16_t>(0));
                          }
                          if (!elide_intermediate_zero) {
                            scratch.barrier.wait();
                          }
                          TeamContext team;
                          team.group_size = task.threads;
                          team.local_tid = local_tid;
                          team.barrier = task.threads > 1 ? &scratch.barrier : nullptr;
                          const Gemm2DSplitPlan split =
                              plan_2d_gemm_split(rows, w13.K_pad, w13.N_pad, task.threads, w13.n_tile);
                          team_fused_w13_silu_packed_packc_backend(
                              true, use_fused_2d_split, team, split, scratch.packed_a.data(),
                              w13_ptr + expert * w13.packed_stride, expert_intermediate, static_cast<int>(rows),
                              static_cast<int>(w13.K_pad), static_cast<int>(w13.N_pad),
                              static_cast<int>(w2.K_pad), silu_poly_degree, w13.n_tile);
                        });
  const double w13_ms = ::fused_cpp::profile::elapsed_ms(w13_begin);

  const auto w2_begin = ::fused_cpp::profile::now();
  execute_planned_stage(num_threads, w2_runtime, scratches,
                        [&](int64_t tid, int64_t, const AsyncTaskRuntime& task,
                            ScheduledTeamScratch&) {
                          const int64_t local_tid = tid - task.core_begin;
                          const int64_t expert = task.expert;
                          const int64_t rows = task.rows;
                          TeamContext team;
                          team.group_size = task.threads;
                          team.local_tid = local_tid;
                          team.barrier = nullptr;
                          const Gemm2DSplitPlan split =
                              plan_2d_gemm_split(rows, w2.K_pad, w2.N_pad, task.threads, w2.n_tile);
                          const uint16_t* expert_intermediate =
                              intermediate_ptr + intermediate_offsets[static_cast<size_t>(expert)];
                          const auto& expert_routes = routes[static_cast<size_t>(expert)];
                          if (use_bf16_route) {
                            team_w2_packed_sve_direct_bf16_route_backend(
                                use_fused_2d_split, team, split, expert_intermediate,
                                w2_ptr + expert * w2.packed_stride, route_out_bf16_ptr, expert_routes.data(),
                                static_cast<int>(rows), static_cast<int>(w2.K_pad), static_cast<int>(w2.N_pad),
                                static_cast<int>(H), w2.n_tile);
                          } else {
                            team_w2_packed_sve_direct_route_backend(
                                use_fused_2d_split, team, split, expert_intermediate,
                                w2_ptr + expert * w2.packed_stride, route_out_f32_ptr, expert_routes.data(),
                                static_cast<int>(rows), static_cast<int>(w2.K_pad), static_cast<int>(w2.N_pad),
                                static_cast<int>(H), w2.n_tile);
                          }
                        });
  const double w2_ms = ::fused_cpp::profile::elapsed_ms(w2_begin);

  const float* topk_w = weights_f32.data_ptr<float>();
  const int route_merge_unroll = resolve_route_merge_unroll(true);
  const auto merge_begin = ::fused_cpp::profile::now();
  run_fixed_threads(num_threads, [&](int64_t tid) {
    const int64_t rows_per_thread = ceil_div_int64(num_tokens, num_threads);
    const int64_t token_begin = tid * rows_per_thread;
    const int64_t token_end = std::min<int64_t>(num_tokens, token_begin + rows_per_thread);
    merge_route_range(route_out_f32_ptr, route_out_bf16_ptr, topk_w, output_ptr, token_begin, token_end, top_k, H,
                      use_bf16_route, route_merge_unroll);
  });
  const double merge_ms = ::fused_cpp::profile::elapsed_ms(merge_begin);

  if (env_flag_enabled("FUSED_CPP_MOE_STAGE_TIMING")) {
    std::fprintf(
        stderr,
        "[fused_moe_bf16_tiled_planned_staged][stage_timing] threads=%lld experts=%lld active=%lld routes=%lld "
        "w13_mode=%lld w13_tasks=%zu w13_pool_threads=%lld w2_mode=%lld w2_tasks=%zu w2_pool_threads=%lld "
        "route_build_ms=%.3f plan_validate_ms=%.3f scratch_ms=%.3f w13_ms=%.3f w2_ms=%.3f merge_ms=%.3f "
        "e2e_ms=%.3f\n",
        static_cast<long long>(num_threads), static_cast<long long>(num_experts),
        static_cast<long long>(active_experts), static_cast<long long>(num_routes),
        static_cast<long long>(w13_runtime.execution_mode), w13_runtime.tasks.size(),
        static_cast<long long>(w13_runtime.pool_threads), static_cast<long long>(w2_runtime.execution_mode),
        w2_runtime.tasks.size(), static_cast<long long>(w2_runtime.pool_threads), route_build_ms, plan_validate_ms,
        scratch_ms, w13_ms, w2_ms, merge_ms, ::fused_cpp::profile::elapsed_ms(call_begin));
  }
  return finalize_moe_output(output, out);
#endif
}

at::Tensor fused_moe_bf16_tiled_vllm_staged(
    at::Tensor input, at::Tensor w13_packed, int64_t w13_K, int64_t w13_N, at::Tensor w2_packed, int64_t w2_K,
    int64_t w2_N, at::Tensor topk_weights, at::Tensor topk_ids, c10::optional<at::Tensor> thread_cpu_ids,
    int64_t num_threads, int64_t global_num_experts, bool fuse_silu, int64_t silu_poly_degree, int64_t gemm_backend,
    int64_t backend_n_tile, c10::optional<at::Tensor> out) {
#ifndef __aarch64__
  TORCH_CHECK(false, "fused_moe_bf16_tiled_vllm_staged requires AArch64");
#else
  const auto call_begin = ::fused_cpp::profile::now();
  check_bf16_cpu(input, "input");
  TORCH_CHECK(input.dim() == 2, "input must be 2-D [tokens, hidden]");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(topk_ids.device().is_cpu(), "topk_ids must be CPU");
  TORCH_CHECK(topk_weights.device().is_cpu(), "topk_weights must be CPU");
  TORCH_CHECK(is_integer_dtype(topk_ids.scalar_type()), "topk_ids must use an integer dtype");
  TORCH_CHECK(is_floating_dtype(topk_weights.scalar_type()), "topk_weights must use a floating dtype");
  TORCH_CHECK(topk_ids.dim() == 2 && topk_weights.dim() == 2,
              "topk_ids and topk_weights must be 2-D [tokens, top_k]");
  TORCH_CHECK(topk_ids.sizes() == topk_weights.sizes(), "topk_ids and topk_weights shapes must match");
  TORCH_CHECK(topk_ids.size(0) == input.size(0), "topk first dimension must match input token count");
  TORCH_CHECK(topk_ids.size(1) > 0, "top_k must be non-zero");
  TORCH_CHECK(num_threads > 0 && num_threads <= std::numeric_limits<int>::max(),
              "num_threads must be in [1, INT_MAX], got ", num_threads);

  const ::fused_cpp::moe::MoeBackend& backend = ::fused_cpp::moe::backend_from_id(gemm_backend);
  TORCH_CHECK(backend.id == ::fused_cpp::moe::BackendId::kArmSveBf16,
              "vLLM-staged baseline requires the SVE BF16 backend, got ", backend.name);
  TORCH_CHECK(backend_n_tile == backend.n_tile(), "MoE backend_n_tile mismatch for ", backend.name,
              ": weights use ", backend_n_tile, ", runtime uses ", backend.n_tile());
  TORCH_CHECK(fuse_silu, "vLLM-staged baseline requires weights prepared with fuse_silu=True");
  TORCH_CHECK(silu_poly_degree == 4 || silu_poly_degree == 5 || silu_poly_degree == 6,
              "silu_poly_degree must be 4, 5, or 6, got ", silu_poly_degree);

  PackedExperts w13 = checked_packed_experts(w13_packed, w13_K, w13_N, "w13_packed", backend_n_tile);
  PackedExperts w2 = checked_packed_experts(w2_packed, w2_K, w2_N, "w2_packed", backend_n_tile);
  TORCH_CHECK(w13.E == w2.E, "w13 and w2 expert count mismatch");
  TORCH_CHECK(w13.K == input.size(1), "input hidden size mismatch: input H=", input.size(1), ", w13 K=", w13.K);
  TORCH_CHECK(w13.N % 2 == 0, "w13 N must be even, got ", w13.N);
  const int64_t F = w13.N / 2;
  const int64_t H = input.size(1);
  TORCH_CHECK(F % kKernelTile == 0, "vLLM-staged fused SiLU requires F to be a multiple of ", kKernelTile,
              ", got ", F);
  TORCH_CHECK(w13.N_pad == 2 * F, "vLLM-staged baseline requires interleaved W13 with N_pad=2F, got ",
              w13.N_pad, " vs ", 2 * F);
  TORCH_CHECK(w2.K == F && w2.N == H, "w2 shape mismatch: expected K=", F, " N=", H, ", got K=", w2.K,
              " N=", w2.N);
  TORCH_CHECK(w2.K_pad == F, "vLLM-staged baseline requires unpadded W2 K, got K_pad=", w2.K_pad, " F=", F);
  TORCH_CHECK(w2.N_pad == H, "vLLM-staged direct-route store requires H to be N-tile aligned: H=", H,
              " N_pad=", w2.N_pad);

  const int64_t num_tokens = input.size(0);
  const int64_t top_k = topk_ids.size(1);
  if (num_tokens == 0) {
    return finalize_moe_output(prepare_moe_output(input, w13_packed, w2_packed, topk_weights, topk_ids, out), out);
  }
  const int64_t num_experts = global_num_experts < 0 ? w13.E : global_num_experts;
  TORCH_CHECK(num_experts > 0 && num_experts <= w13.E,
              "global_num_experts must be positive and no larger than packed experts: got ", num_experts,
              " packed=", w13.E);

  ThreadPinningConfig staged_thread_pinning;
  bool has_thread_pinning = false;
  if (thread_cpu_ids.has_value() && thread_cpu_ids->defined() && thread_cpu_ids->numel() > 0) {
    staged_thread_pinning.cpus = tensor_to_i64_vector(*thread_cpu_ids, "thread_cpu_ids");
    TORCH_CHECK(static_cast<int64_t>(staged_thread_pinning.cpus.size()) == num_threads,
                "thread_cpu_ids must have exactly num_threads entries: got ", staged_thread_pinning.cpus.size(),
                " vs ", num_threads);
    for (size_t idx = 0; idx < staged_thread_pinning.cpus.size(); ++idx) {
      TORCH_CHECK(staged_thread_pinning.cpus[idx] >= 0, "thread_cpu_ids[", idx, "] must be non-negative, got ",
                  staged_thread_pinning.cpus[idx]);
    }
    staged_thread_pinning.enabled = true;
    has_thread_pinning = true;
  }
  ThreadPinningScope staged_thread_pinning_scope(has_thread_pinning ? &staged_thread_pinning : nullptr);
  prepare_moe_threads_for_operator(num_threads);

  const auto route_build_begin = ::fused_cpp::profile::now();
  at::Tensor ids_i64 = topk_ids.to(at::kLong).contiguous();
  at::Tensor weights_f32 = topk_weights.to(at::kFloat).contiguous();
  const int64_t* ids = ids_i64.data_ptr<int64_t>();
  const int64_t num_routes = num_tokens * top_k;
  std::vector<int64_t> route_counts(static_cast<size_t>(num_experts), 0);
  for (int64_t flat = 0; flat < num_routes; ++flat) {
    const int64_t expert = ids[flat];
    TORCH_CHECK(expert >= 0 && expert < num_experts, "topk_ids out of range: id=", expert, ", valid range [0, ",
                num_experts, ")");
    ++route_counts[static_cast<size_t>(expert)];
  }
  std::vector<int64_t> route_offsets(static_cast<size_t>(num_experts + 1), 0);
  for (int64_t expert = 0; expert < num_experts; ++expert) {
    route_offsets[static_cast<size_t>(expert + 1)] =
        route_offsets[static_cast<size_t>(expert)] + route_counts[static_cast<size_t>(expert)];
  }
  std::vector<int64_t> route_cursors = route_offsets;
  route_cursors.pop_back();
  std::vector<int64_t> expert_routes(static_cast<size_t>(num_routes));
  for (int64_t flat = 0; flat < num_routes; ++flat) {
    const int64_t expert = ids[flat];
    const int64_t dst = route_cursors[static_cast<size_t>(expert)]++;
    expert_routes[static_cast<size_t>(dst)] = flat;
  }

  std::vector<int64_t> intermediate_offsets(static_cast<size_t>(num_experts + 1), 0);
  for (int64_t expert = 0; expert < num_experts; ++expert) {
    const int64_t packed_rows = sve_hybrid_packed_rows(route_counts[static_cast<size_t>(expert)]);
    TORCH_CHECK(packed_rows <= std::numeric_limits<int64_t>::max() / w2.K_pad,
                "vLLM-staged intermediate size overflows int64");
    const int64_t expert_elements = packed_rows * w2.K_pad;
    TORCH_CHECK(intermediate_offsets[static_cast<size_t>(expert)] <=
                    std::numeric_limits<int64_t>::max() - expert_elements,
                "vLLM-staged cumulative intermediate size overflows int64");
    intermediate_offsets[static_cast<size_t>(expert + 1)] =
        intermediate_offsets[static_cast<size_t>(expert)] + expert_elements;
  }
  const double route_build_ms = ::fused_cpp::profile::elapsed_ms(route_build_begin);

  at::Tensor output = prepare_moe_output(input, w13_packed, w2_packed, topk_weights, topk_ids, out);
  const bool use_bf16_route = sve_w2_bf16_route_enabled();
  const int64_t route_element_bytes =
      use_bf16_route ? static_cast<int64_t>(sizeof(uint16_t)) : static_cast<int64_t>(sizeof(float));
  TORCH_CHECK(sve_w2_direct_route_offsets_fit(num_routes, H, w2.n_tile, route_element_bytes),
              "vLLM-staged direct-route offsets exceed the SVE kernel's int32 byte range");
  at::Tensor route_out =
      at::empty({num_routes, H}, input.options().dtype(use_bf16_route ? at::kBFloat16 : at::kFloat));
  at::Tensor intermediate = at::empty({intermediate_offsets.back()}, input.options());

  uint16_t* output_ptr = bf16_data(output);
  uint16_t* intermediate_ptr = bf16_data(intermediate);
  uint16_t* route_out_bf16_ptr = use_bf16_route ? bf16_data(route_out) : nullptr;
  float* route_out_f32_ptr = use_bf16_route ? nullptr : route_out.data_ptr<float>();
  const uint16_t* input_ptr = bf16_data_const(input);
  const uint16_t* w13_ptr = bf16_data_const(w13.tensor);
  const uint16_t* w2_ptr = bf16_data_const(w2.tensor);

  const int64_t w13_task_n =
      vllm_staged_task_n(MoeGemmStage::kW13, w13.K_pad, w13.N_pad, w13.n_tile, num_threads, top_k);
  const int64_t w2_task_n =
      vllm_staged_task_n(MoeGemmStage::kW2, w2.K_pad, w2.N_pad, w2.n_tile, num_threads, top_k);
  const std::vector<VllmStagedNTask> w13_tasks =
      build_vllm_staged_tasks(num_experts, w13.N_pad, w13_task_n, w13.n_tile);
  const std::vector<VllmStagedNTask> w2_tasks =
      build_vllm_staged_tasks(num_experts, w2.N_pad, w2_task_n, w2.n_tile);
  std::vector<VllmStagedThreadScratch> scratches(static_cast<size_t>(num_threads));
  for (VllmStagedThreadScratch& scratch : scratches) {
    scratch.packed_a.resize(static_cast<size_t>(12 * w13.K_pad));
  }

  alignas(64) std::atomic<int64_t> next_w13_task{0};
  const auto w13_begin = ::fused_cpp::profile::now();
  run_fixed_threads(num_threads, [&](int64_t tid) {
    VllmStagedThreadScratch& scratch = scratches[static_cast<size_t>(tid)];
    while (true) {
      const int64_t task_id = next_w13_task.fetch_add(1, std::memory_order_relaxed);
      if (task_id >= static_cast<int64_t>(w13_tasks.size())) {
        break;
      }
      const VllmStagedNTask& task = w13_tasks[static_cast<size_t>(task_id)];
      const int64_t rows = route_counts[static_cast<size_t>(task.expert)];
      if (rows == 0) {
        continue;
      }
      const int64_t* routes = expert_routes.data() + route_offsets[static_cast<size_t>(task.expert)];
      uint16_t* expert_intermediate =
          intermediate_ptr + intermediate_offsets[static_cast<size_t>(task.expert)];
      const uint16_t* expert_w13 = w13_ptr + task.expert * w13.packed_stride;
      for (int64_t row_begin = 0; row_begin < rows; row_begin += 12) {
        const int64_t panel_rows = std::min<int64_t>(12, rows - row_begin);
        gather_pack_a_reorder_sve_hybrid(input_ptr, H, routes + row_begin, top_k, scratch.packed_a.data(),
                                         static_cast<int>(panel_rows), static_cast<int>(w13.K_pad), int64_t{1},
                                         int64_t{0});
        vllm_staged_w13_range_sve(
            scratch.packed_a.data(), expert_w13, expert_intermediate + row_begin * w2.K_pad,
            static_cast<int>(panel_rows), static_cast<int>(w13.K_pad), static_cast<int>(w2.K_pad),
            silu_poly_degree, w13.n_tile, task.n_begin, task.n_cols);
      }
    }
  });
  const double w13_ms = ::fused_cpp::profile::elapsed_ms(w13_begin);

  alignas(64) std::atomic<int64_t> next_w2_task{0};
  const auto w2_begin = ::fused_cpp::profile::now();
  run_fixed_threads(num_threads, [&](int64_t) {
    while (true) {
      const int64_t task_id = next_w2_task.fetch_add(1, std::memory_order_relaxed);
      if (task_id >= static_cast<int64_t>(w2_tasks.size())) {
        break;
      }
      const VllmStagedNTask& task = w2_tasks[static_cast<size_t>(task_id)];
      const int64_t rows = route_counts[static_cast<size_t>(task.expert)];
      if (rows == 0) {
        continue;
      }
      const int64_t* routes = expert_routes.data() + route_offsets[static_cast<size_t>(task.expert)];
      const uint16_t* expert_intermediate =
          intermediate_ptr + intermediate_offsets[static_cast<size_t>(task.expert)];
      const uint16_t* expert_w2 = w2_ptr + task.expert * w2.packed_stride;
      if (use_bf16_route) {
        vllm_staged_w2_direct_bf16_route_range_sve(
            expert_intermediate, expert_w2, route_out_bf16_ptr, routes, static_cast<int>(rows),
            static_cast<int>(w2.K_pad), static_cast<int>(H), w2.n_tile, task.n_begin, task.n_cols);
      } else {
        vllm_staged_w2_direct_route_range_sve(
            expert_intermediate, expert_w2, route_out_f32_ptr, routes, static_cast<int>(rows),
            static_cast<int>(w2.K_pad), static_cast<int>(H), w2.n_tile, task.n_begin, task.n_cols);
      }
    }
  });
  const double w2_ms = ::fused_cpp::profile::elapsed_ms(w2_begin);

  const float* topk_w = weights_f32.data_ptr<float>();
  const int route_merge_unroll = resolve_route_merge_unroll(true);
  const auto merge_begin = ::fused_cpp::profile::now();
  run_fixed_threads(num_threads, [&](int64_t tid) {
    const int64_t rows_per_thread = ceil_div_int64(num_tokens, num_threads);
    const int64_t token_begin = tid * rows_per_thread;
    const int64_t token_end = std::min<int64_t>(num_tokens, token_begin + rows_per_thread);
    merge_route_range(route_out_f32_ptr, route_out_bf16_ptr, topk_w, output_ptr, token_begin, token_end, top_k, H,
                      use_bf16_route, route_merge_unroll);
  });
  const double merge_ms = ::fused_cpp::profile::elapsed_ms(merge_begin);

  if (env_flag_enabled("FUSED_CPP_MOE_STAGE_TIMING")) {
    std::fprintf(stderr,
                 "[fused_moe_bf16_tiled_vllm_staged][stage_timing] threads=%lld experts=%lld routes=%lld "
                 "available_l2_bytes=%lld w13_task_n=%lld w13_tasks=%zu w2_task_n=%lld w2_tasks=%zu "
                 "route_build_ms=%.3f w13_ms=%.3f w2_ms=%.3f merge_ms=%.3f e2e_ms=%.3f\n",
                 static_cast<long long>(num_threads), static_cast<long long>(num_experts),
                 static_cast<long long>(num_routes), static_cast<long long>(vllm_staged_available_l2_bytes()),
                 static_cast<long long>(w13_task_n), w13_tasks.size(), static_cast<long long>(w2_task_n),
                 w2_tasks.size(), route_build_ms, w13_ms, w2_ms, merge_ms,
                 ::fused_cpp::profile::elapsed_ms(call_begin));
  }
  return finalize_moe_output(output, out);
#endif
}
