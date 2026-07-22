#pragma once

#include <cstdint>
#include <string>

#include "gemm_params.h"

namespace fused_cpp::moe_sve::jit {

enum class ImplementationMode {
  kAuto,
  kJit,
  kAsm,
};

enum class Operation : uint8_t {
  kW13,
  kW2,
  kW2Direct,
};

using KernelFn = void (*)(const uint16_t*, const uint16_t*, void*, const void*, const gemm_params_t*);

bool built();
ImplementationMode implementation_mode();
bool requested_for_current_build();
const void* silu_constants();

// Returned code remains executable until process shutdown. Generation and cache
// lookup are thread-safe; `error` is populated when no JIT kernel is available.
KernelFn get_kernel(Operation operation, int rows, int degree, std::string* error);
// The bulk kernel consumes a positive multiple of 12 rows from params->m. It
// preserves the exact-M kernel ABI but advances packed A and output state inside
// generated code, amortizing the function prologue across all full M12 panels.
KernelFn get_bulk_m12_kernel(Operation operation, int degree, std::string* error);
void prewarm(Operation operation, int degree);
void prewarm_bulk_m12(Operation operation, int degree);

}  // namespace fused_cpp::moe_sve::jit
