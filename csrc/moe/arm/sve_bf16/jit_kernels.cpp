// SPDX-License-Identifier: Apache-2.0
#include "jit_kernels.h"

#include <array>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace fused_cpp::moe_sve::jit {
namespace {

struct alignas(64) SiluConstants {
  float one = 1.0f;
  float exp_c0 = 1.000003695487976f;
  float exp_c1 = 0.5000003576278687f;
  float inv_ln2 = 1.4426950408889634f;
  float ln2_hi = 0.693145751953125f;
  float ln2_lo = 1.428606765330187e-06f;
  float fexpa_shift = 196735.0f;
  float clamp_hi = 87.0f;
  float clamp_lo = -87.0f;
  float swiglu_limit = 10.0f;
};

const SiluConstants kSiluConstants;

// The assembly file for one fixed VL defines exactly one copy of each symbol.
// Declarations keep the existing kernel ABI used by the MoE and DSV4 callers.
#define DECLARE_NORMAL(op, row)                                                                            \
  extern "C" void fused_cpp_sve_##op##_m##row(                                                              \
      const uint16_t*, const uint16_t*, void*, const void*, const gemm_params_t*);
#define DECLARE_W8(op, row)                                                                                \
  extern "C" void fused_cpp_sve_##op##_m##row(                                                              \
      const uint16_t*, const int8_t*, const float*, void*, const void*, const gemm_params_t*);
#define FOR_EACH_ROW(macro, op) \
  macro(op, 1) macro(op, 2) macro(op, 3) macro(op, 4) macro(op, 5) macro(op, 6) \
      macro(op, 7) macro(op, 8) macro(op, 9) macro(op, 10) macro(op, 11) macro(op, 12)

FOR_EACH_ROW(DECLARE_NORMAL, w13)
FOR_EACH_ROW(DECLARE_NORMAL, w13_clamped)
FOR_EACH_ROW(DECLARE_NORMAL, w2)
FOR_EACH_ROW(DECLARE_NORMAL, w2_direct)
FOR_EACH_ROW(DECLARE_NORMAL, gemm_f32)
FOR_EACH_ROW(DECLARE_NORMAL, gemm_bf16)
FOR_EACH_ROW(DECLARE_W8, w8_w13)
FOR_EACH_ROW(DECLARE_W8, w8_w13_clamped)
FOR_EACH_ROW(DECLARE_W8, w8_w2_direct)

extern "C" void fused_cpp_sve_w8_dequant(const int8_t*, const float*, uint16_t*, int, int);
extern "C" void fused_cpp_sve_probe_b_m1(const uint16_t*, const uint16_t*, void*, const void*, const gemm_params_t*);
extern "C" void fused_cpp_sve_probe_b_m2(const uint16_t*, const uint16_t*, void*, const void*, const gemm_params_t*);
extern "C" void fused_cpp_sve_probe_full_nostore_m1(
    const uint16_t*, const uint16_t*, void*, const void*, const gemm_params_t*);
extern "C" void fused_cpp_sve_probe_full_nostore_m2(
    const uint16_t*, const uint16_t*, void*, const void*, const gemm_params_t*);
extern "C" void fused_cpp_sve_probe_full_nostore_m12(
    const uint16_t*, const uint16_t*, void*, const void*, const gemm_params_t*);
extern "C" void fused_cpp_sve_probe_matrix_m12(
    const uint16_t*, const uint16_t*, void*, const void*, const gemm_params_t*);

#define NORMAL_ENTRY(op, row) &fused_cpp_sve_##op##_m##row,
#define W8_ENTRY(op, row) &fused_cpp_sve_##op##_m##row,
#define KERNEL_TABLE(entry, op) {FOR_EACH_ROW(entry, op)}

const std::array<KernelFn, 12> kW13 = KERNEL_TABLE(NORMAL_ENTRY, w13);
const std::array<KernelFn, 12> kW13Clamped = KERNEL_TABLE(NORMAL_ENTRY, w13_clamped);
const std::array<KernelFn, 12> kW2 = KERNEL_TABLE(NORMAL_ENTRY, w2);
const std::array<KernelFn, 12> kW2Direct = KERNEL_TABLE(NORMAL_ENTRY, w2_direct);
const std::array<KernelFn, 12> kGemmF32 = KERNEL_TABLE(NORMAL_ENTRY, gemm_f32);
const std::array<KernelFn, 12> kGemmBf16 = KERNEL_TABLE(NORMAL_ENTRY, gemm_bf16);
const std::array<W8KernelFn, 12> kW8W13 = KERNEL_TABLE(W8_ENTRY, w8_w13);
const std::array<W8KernelFn, 12> kW8W13Clamped = KERNEL_TABLE(W8_ENTRY, w8_w13_clamped);
const std::array<W8KernelFn, 12> kW8W2Direct = KERNEL_TABLE(W8_ENTRY, w8_w2_direct);

#undef KERNEL_TABLE
#undef W8_ENTRY
#undef NORMAL_ENTRY
#undef FOR_EACH_ROW
#undef DECLARE_W8
#undef DECLARE_NORMAL

void SetError(std::string* error, const char* message) {
  if (error != nullptr) {
    *error = message;
  }
}

bool IsW13(Operation operation) {
  return operation == Operation::kW13 || operation == Operation::kW13Clamped ||
         operation == Operation::kW8W13 || operation == Operation::kW8W13Clamped;
}

bool ValidDegree(Operation operation, int degree) {
  return IsW13(operation) ? degree >= 4 && degree <= 6 : degree == 0;
}

}  // namespace

bool built() { return true; }

ImplementationMode implementation_mode() {
  const char* raw = std::getenv("FUSED_CPP_MOE_SVE_IMPL");
  const std::string value = raw == nullptr ? "auto" : raw;
  if (value.empty() || value == "auto") {
    return ImplementationMode::kAuto;
  }
  if (value == "jit") {
    // Historical selector: this now chooses the precompiled exact-M kernels.
    return ImplementationMode::kJit;
  }
  if (value == "asm") {
    return ImplementationMode::kAsm;
  }
  throw std::runtime_error("FUSED_CPP_MOE_SVE_IMPL must be auto, jit, or asm; got '" + value + "'");
}

bool requested_for_current_build() { return implementation_mode() != ImplementationMode::kAsm; }

const void* silu_constants() { return &kSiluConstants; }

KernelFn get_kernel(Operation operation, int rows, int degree, std::string* error) {
  if (rows < 1 || rows > 12 || !ValidDegree(operation, degree)) {
    SetError(error, "unsupported precompiled SVE kernel specialization");
    return nullptr;
  }
  const std::array<KernelFn, 12>* table = nullptr;
  switch (operation) {
    case Operation::kW13: table = &kW13; break;
    case Operation::kW13Clamped: table = &kW13Clamped; break;
    case Operation::kW2: table = &kW2; break;
    case Operation::kW2Direct: table = &kW2Direct; break;
    case Operation::kGemmF32: table = &kGemmF32; break;
    case Operation::kGemmBf16: table = &kGemmBf16; break;
    default: break;
  }
  if (table == nullptr) {
    SetError(error, "operation requires the W8 kernel signature");
    return nullptr;
  }
  SetError(error, "");
  return (*table)[static_cast<size_t>(rows - 1)];
}

W8KernelFn get_w8_kernel(Operation operation, int rows, int degree, std::string* error) {
  if (rows < 1 || rows > 12 || !ValidDegree(operation, degree)) {
    SetError(error, "unsupported precompiled SVE W8 kernel specialization");
    return nullptr;
  }
  const std::array<W8KernelFn, 12>* table = nullptr;
  switch (operation) {
    case Operation::kW8W13: table = &kW8W13; break;
    case Operation::kW8W13Clamped: table = &kW8W13Clamped; break;
    case Operation::kW8W2Direct: table = &kW8W2Direct; break;
    default: break;
  }
  if (table == nullptr) {
    SetError(error, "operation does not use the W8 kernel signature");
    return nullptr;
  }
  SetError(error, "");
  return (*table)[static_cast<size_t>(rows - 1)];
}

W8DequantFn get_w8_dequant_kernel(std::string* error) {
  SetError(error, "");
  return &fused_cpp_sve_w8_dequant;
}

KernelFn get_probe_kernel(int rows, ProbeMode mode, std::string* error) {
  KernelFn function = nullptr;
  if (mode == ProbeMode::kBOnly) {
    if (rows == 1) function = &fused_cpp_sve_probe_b_m1;
    if (rows == 2) function = &fused_cpp_sve_probe_b_m2;
  } else if (mode == ProbeMode::kFullNoStore) {
    if (rows == 1) function = &fused_cpp_sve_probe_full_nostore_m1;
    if (rows == 2) function = &fused_cpp_sve_probe_full_nostore_m2;
    if (rows == 12) function = &fused_cpp_sve_probe_full_nostore_m12;
  } else if (mode == ProbeMode::kMatrixOnly && rows == 12) {
    function = &fused_cpp_sve_probe_matrix_m12;
  }
  SetError(error, function == nullptr ? "unsupported precompiled SVE probe specialization" : "");
  return function;
}

KernelFn get_gemm_f32_kernel(int rows, std::string* error) {
  return get_kernel(Operation::kGemmF32, rows, 0, error);
}

KernelFn get_gemm_bf16_kernel(int rows, std::string* error) {
  return get_kernel(Operation::kGemmBf16, rows, 0, error);
}

void prewarm(Operation operation, int degree) {
  // Every specialization is linked into the extension; lookup is already warm.
  (void)operation;
  (void)degree;
}

}  // namespace fused_cpp::moe_sve::jit
