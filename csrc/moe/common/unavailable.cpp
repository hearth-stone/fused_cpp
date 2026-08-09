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
                                int64_t, bool, bool, int64_t, int64_t, int64_t, int64_t, int64_t,
                                c10::optional<at::Tensor>) {
  unavailable();
}
#endif

at::Tensor fused_moe_bf16_tiled_scheduled(at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t,
                                          at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                          c10::optional<at::Tensor>, c10::optional<at::Tensor>,
                                          c10::optional<at::Tensor>, int64_t, std::string, int64_t, bool, bool, int64_t,
                                          int64_t, int64_t, int64_t, int64_t, c10::optional<at::Tensor>) {
  unavailable();
}

at::Tensor fused_moe_bf16_tiled_async(at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t,
                                      at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                      at::Tensor, c10::optional<at::Tensor>, c10::optional<at::Tensor>,
                                      c10::optional<at::Tensor>, int64_t, std::string, int64_t, bool, bool, int64_t,
                                      int64_t, int64_t, int64_t, int64_t, c10::optional<at::Tensor>) {
  unavailable();
}

at::Tensor fused_moe_bf16_tiled_async_plan_v2(
    at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>, int64_t, std::string, int64_t,
    bool, bool, int64_t, int64_t, int64_t, c10::optional<at::Tensor>, c10::optional<at::Tensor>, int64_t) {
  unavailable();
}

at::Tensor fused_moe_bf16_tiled_async_plan_v2_elastic(
    at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>, int64_t, std::string, int64_t,
    bool, bool, int64_t, int64_t, int64_t, c10::optional<at::Tensor>,
    c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>, int64_t) {
  unavailable();
}

at::Tensor fused_moe_bf16_tiled_planned_staged(
    at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, int64_t, at::Tensor, at::Tensor, c10::optional<at::Tensor>, int64_t, int64_t,
    bool, int64_t, int64_t, int64_t, c10::optional<at::Tensor>) {
  unavailable();
}

at::Tensor fused_moe_bf16_tiled_vllm_staged(at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t,
                                             at::Tensor, at::Tensor, c10::optional<at::Tensor>, int64_t, int64_t, bool,
                                             int64_t, int64_t, int64_t, c10::optional<at::Tensor>) {
  unavailable();
}
#endif
