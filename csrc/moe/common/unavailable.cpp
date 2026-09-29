// SPDX-License-Identifier: Apache-2.0
#include "api.h"

#include <stdexcept>

#if !defined(__aarch64__)
namespace {

[[noreturn]] void unavailable() {
  throw std::runtime_error("the requested fused MoE execution mode is not implemented for this architecture/backend");
}

}  // namespace

#if !defined(__x86_64__)
std::tuple<at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t, int64_t, int64_t>
fused_moe_bf16_tiled_prepare_weights(at::Tensor, at::Tensor, bool, std::string) {
  unavailable();
}

at::Tensor fused_moe_bf16_tiled(at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t, at::Tensor,
                                at::Tensor, c10::optional<at::Tensor>, c10::optional<at::Tensor>, int64_t, std::string,
                                int64_t, bool, bool, int64_t, int64_t, int64_t,
                                c10::optional<at::Tensor>) {
  unavailable();
}
#endif

#if !defined(FUSED_CPP_MOE_HAS_ARM_SVE)
bool deepseek_v4_inv_rope_woa_available() { return false; }

std::tuple<at::Tensor, int64_t, int64_t, int64_t> deepseek_v4_inv_rope_woa_prepare(
    at::Tensor, int64_t, int64_t, int64_t, int64_t, std::string) {
  unavailable();
}

at::Tensor deepseek_v4_inv_rope_grouped_woa(
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t,
    c10::optional<at::Tensor>, c10::optional<at::Tensor>) {
  unavailable();
}
#endif

std::tuple<at::Tensor, int64_t, int64_t, at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t>
fused_moe_w8a16_tiled_prepare_weights(at::Tensor, at::Tensor) {
  unavailable();
}

std::tuple<at::Tensor, int64_t, int64_t, at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t>
fused_moe_w8a16_tiled_prepare_quantized_weights(at::Tensor, at::Tensor, at::Tensor, at::Tensor) {
  unavailable();
}

std::tuple<at::Tensor, int64_t, int64_t, at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t>
fused_moe_w8a16_tiled_prepare_quantized_routed_shared_weights(at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                                              at::Tensor, at::Tensor, at::Tensor, at::Tensor) {
  unavailable();
}

std::tuple<at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t, int64_t, int64_t>
fused_moe_bf16_tiled_prepare_routed_shared_weights(at::Tensor, at::Tensor, at::Tensor, at::Tensor, std::string) {
  unavailable();
}

at::Tensor fused_moe_bf16_tiled_scheduled(at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t,
                                          at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                          c10::optional<at::Tensor>, c10::optional<at::Tensor>,
                                          c10::optional<at::Tensor>, int64_t, std::string, int64_t, bool, bool, int64_t,
                                          int64_t, int64_t, c10::optional<at::Tensor>) {
  unavailable();
}

at::Tensor fused_moe_bf16_tiled_async(at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t,
                                      at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                      at::Tensor, c10::optional<at::Tensor>, c10::optional<at::Tensor>,
                                      c10::optional<at::Tensor>, int64_t, std::string, int64_t, bool, bool, int64_t,
                                      int64_t, int64_t, c10::optional<at::Tensor>) {
  unavailable();
}

at::Tensor fused_moe_bf16_tiled_async_plan_v2(
    at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>, int64_t, std::string, int64_t,
    bool, bool, int64_t, int64_t, int64_t, c10::optional<at::Tensor>, int64_t) {
  unavailable();
}

at::Tensor fused_moe_w8a16_tiled_async_plan_v2(
    at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>, int64_t, std::string, int64_t,
    bool, int64_t, int64_t, c10::optional<at::Tensor>, int64_t, bool) {
  unavailable();
}

at::Tensor fused_moe_bf16_tiled_planned_staged(
    at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, int64_t, at::Tensor, c10::optional<at::Tensor>, int64_t, int64_t,
    bool, int64_t, int64_t, int64_t, c10::optional<at::Tensor>) {
  unavailable();
}

at::Tensor fused_moe_bf16_tiled_vllm_staged(at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t,
                                             at::Tensor, at::Tensor, c10::optional<at::Tensor>, int64_t, int64_t, bool,
                                             int64_t, int64_t, int64_t, c10::optional<at::Tensor>) {
  unavailable();
}
#endif
