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
  kW13Clamped,
  kW2,
  kW2Direct,
  kGemmF32,
  kW8W13,
  kW8W13Clamped,
  kW8W2Direct,
};

enum class ProbeMode : uint8_t {
  kNone = 0,
  kBOnly = 1,
  kFullNoStore = 4,
  kMatrixOnly = 10,
};

using KernelFn = void (*)(const uint16_t*, const uint16_t*, void*, const void*, const gemm_params_t*);
using W8KernelFn = void (*)(const uint16_t*, const int8_t*, const float*, void*, const void*, const gemm_params_t*);
using W8DequantFn = void (*)(const int8_t*, const float*, uint16_t*, int, int);

bool built();
ImplementationMode implementation_mode();
bool requested_for_current_build();
const void* silu_constants();

// Returned code remains executable until process shutdown. Generation and cache
// lookup are thread-safe; `error` is populated when no JIT kernel is available.
KernelFn get_kernel(Operation operation, int rows, int degree, std::string* error);
W8KernelFn get_w8_kernel(Operation operation, int rows, int degree, std::string* error);
W8DequantFn get_w8_dequant_kernel(std::string* error);
// Standalone packed-A/packed-B BF16 GEMM with row-major FP32 output.
KernelFn get_gemm_f32_kernel(int rows, std::string* error);
// Calibration-only variants of the pure GEMM kernel. Numeric values remain
// stable because they are recorded in machine-profile provenance.
KernelFn get_probe_kernel(int rows, ProbeMode mode, std::string* error);
void prewarm(Operation operation, int degree);

}  // namespace fused_cpp::moe_sve::jit
