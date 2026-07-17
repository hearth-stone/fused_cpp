// SPDX-License-Identifier: Apache-2.0
#include "../sve_bf16/packing.h"

#include <stdexcept>

#if !defined(FUSED_CPP_MOE_HAS_ARM_SVE)
namespace fused_cpp::moe_sve {
namespace {

[[noreturn]] void unavailable() { throw std::runtime_error("SVE BF16 MoE backend is unavailable in this build"); }

}  // namespace

bool available() { return false; }
bool enabled_by_env() { return false; }
int n_tile() { unavailable(); }
int round_k(int) { unavailable(); }
int round_n(int) { unavailable(); }
void pack_b(const uint16_t*, uint16_t*, int, int) { unavailable(); }
void pack_a_block(const uint16_t*, uint16_t*, int, int) { unavailable(); }
void gather_pack_a(const uint16_t*, int64_t, const int64_t*, int64_t, uint16_t*, int, int, int, int) { unavailable(); }
void w13_silu_rowmajor(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*, int64_t) {
  unavailable();
}
void w13_silu_packed(const uint16_t*, const uint16_t*, uint16_t*, const gemm_params_t*, int64_t) { unavailable(); }
void w13_silu_packc(const uint16_t*, const uint16_t*, uint16_t*, const gemm_params_t*, int64_t, int64_t) {
  unavailable();
}
void w2_packed(const uint16_t*, const uint16_t*, float*, const gemm_params_t*, int64_t) { unavailable(); }
void w2_rowmajor(const uint16_t*, const uint16_t*, float*, uint16_t*, const gemm_params_t*) { unavailable(); }

}  // namespace fused_cpp::moe_sve
#endif
