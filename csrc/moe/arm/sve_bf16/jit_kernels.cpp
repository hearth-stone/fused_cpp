#include "jit_kernels.h"
#include "vector_length.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#if defined(FUSED_CPP_MOE_HAS_XBYAK_AARCH64) && FUSED_CPP_MOE_HAS_XBYAK_AARCH64
#include <xbyak_aarch64/xbyak_aarch64.h>
#endif

namespace fused_cpp::moe_sve::jit {
namespace {

struct alignas(64) SiluConstants {
  float one = 1.0f;
  float half = 0.5f;
  float inv_ln2 = 1.4426950408889634f;
  float ln2 = 0.6931471805599453f;
  float c3 = 0.16666666f;
  float c4 = 0.04166666f;
  float c5 = 0.00833333f;
  float c6 = 0.0013888889f;
  float clamp_hi = 87.0f;
  float clamp_lo = -87.0f;
  float swiglu_limit = 10.0f;
};

const SiluConstants kSiluConstants;

}  // namespace

ImplementationMode implementation_mode() {
  const char* raw = std::getenv("FUSED_CPP_MOE_SVE_IMPL");
  const std::string value = raw == nullptr ? "auto" : raw;
  if (value.empty() || value == "auto") {
    return ImplementationMode::kAuto;
  }
  if (value == "jit") {
    return ImplementationMode::kJit;
  }
  if (value == "asm") {
    return ImplementationMode::kAsm;
  }
  throw std::runtime_error("FUSED_CPP_MOE_SVE_IMPL must be auto, jit, or asm; got '" + value + "'");
}

const void* silu_constants() { return &kSiluConstants; }

#if !defined(FUSED_CPP_MOE_HAS_XBYAK_AARCH64) || !FUSED_CPP_MOE_HAS_XBYAK_AARCH64

bool built() { return false; }

bool requested_for_current_build() {
  const ImplementationMode mode = implementation_mode();
  if (mode == ImplementationMode::kJit) {
    throw std::runtime_error(
        "FUSED_CPP_MOE_SVE_IMPL=jit requires a build with the xbyak_aarch64 submodule initialized");
  }
  return false;
}

KernelFn get_kernel(Operation, int, int, std::string* error) {
  if (error != nullptr) {
    *error = "xbyak_aarch64 is unavailable in this build";
  }
  return nullptr;
}

W8KernelFn get_w8_kernel(Operation, int, int, std::string* error) {
  if (error != nullptr) {
    *error = "xbyak_aarch64 is unavailable in this build";
  }
  return nullptr;
}

W8DequantFn get_w8_dequant_kernel(std::string* error) {
  if (error != nullptr) {
    *error = "xbyak_aarch64 is unavailable in this build";
  }
  return nullptr;
}

KernelFn get_probe_kernel(int, ProbeMode, std::string* error) {
  if (error != nullptr) {
    *error = "xbyak_aarch64 is unavailable in this build";
  }
  return nullptr;
}

void prewarm(Operation, int) {}

#else

using namespace Xbyak_aarch64;

bool probe_rows_supported(int rows, ProbeMode mode) {
  switch (mode) {
    case ProbeMode::kNone:
      return false;
    case ProbeMode::kBOnly:
      return rows >= 1 && rows <= 2;
    case ProbeMode::kFullNoStore:
      return (rows >= 1 && rows <= 2) || rows == 12;
    case ProbeMode::kMatrixOnly:
      return rows == 12;
  }
  return false;
}

struct KernelKey {
  Operation operation = Operation::kW13;
  uint8_t rows = 0;
  uint8_t degree = 0;
  ProbeMode probe_mode = ProbeMode::kNone;
};

class SveFusedGenerator final : public CodeGenerator {
 public:
  SveFusedGenerator(Operation operation, int rows, int degree, ProbeMode probe_mode)
      : CodeGenerator(64 * 1024, AutoGrow),
        operation_(operation),
        rows_(rows),
        degree_(degree),
        row_pairs_((rows + 1) / 2),
        accumulator_base_(rows <= 8 ? 16 : 8),
        physical_rows_(rows <= 8 ? 8 : 12),
        probe_mode_(probe_mode),
        clamp_swiglu_(operation == Operation::kW13Clamped || operation == Operation::kW8W13Clamped),
        w8_weights_(operation == Operation::kW8W13 || operation == Operation::kW8W13Clamped ||
                    operation == Operation::kW8W2Direct) {
    if (rows_ < 1 || rows_ > 12) {
      throw std::invalid_argument("SVE JIT rows must be in [1, 12]");
    }
    if (probe_mode_ != ProbeMode::kNone) {
      if (operation_ != Operation::kGemmF32 || !probe_rows_supported(rows_, probe_mode_)) {
        throw std::invalid_argument("SVE JIT probe mode is incompatible with the requested GEMM kernel");
      }
    }
    if ((operation_ == Operation::kW13 || operation_ == Operation::kW13Clamped ||
         operation_ == Operation::kW8W13 || operation_ == Operation::kW8W13Clamped) &&
        (degree_ < 4 || degree_ > 6)) {
      throw std::invalid_argument("SVE JIT W13 degree must be 4, 5, or 6");
    }
    if (operation_ != Operation::kW13 && operation_ != Operation::kW13Clamped &&
        operation_ != Operation::kW8W13 && operation_ != Operation::kW8W13Clamped && degree_ != 0) {
      throw std::invalid_argument("SVE JIT plain GEMM operations require degree 0");
    }
    generate();
    readyRE();
    dump_if_requested();
  }

  KernelFn function() const { return getCode<KernelFn>(); }

 private:
  static constexpr int kParamK = 4;
  static constexpr int kParamN = 8;
  static constexpr int kParamLdc = 20;
  static constexpr int kParamNBegin = 32;

  ZRegS accumulator(int pair, int column) const { return ZRegS(accumulator_base_ + pair * 4 + column); }
  void dump_if_requested() const {
    const char* directory = std::getenv("FUSED_CPP_MOE_SVE_JIT_DUMP_DIR");
    if (directory == nullptr || directory[0] == '\0') {
      return;
    }
    const char* operation = "gemm_f32";
    switch (operation_) {
      case Operation::kW13:
        operation = "w13";
        break;
      case Operation::kW13Clamped:
        operation = "w13_clamped";
        break;
      case Operation::kW2:
        operation = "w2";
        break;
      case Operation::kW2Direct:
        operation = "w2_direct";
        break;
      case Operation::kGemmF32:
        break;
      case Operation::kGemmBf16:
        operation = "gemm_bf16";
        break;
      case Operation::kW8W13:
        operation = "w8_w13";
        break;
      case Operation::kW8W13Clamped:
        operation = "w8_w13_clamped";
        break;
      case Operation::kW8W2Direct:
        operation = "w8_w2_direct";
        break;
    }
    const char* probe = "";
    switch (probe_mode_) {
      case ProbeMode::kNone:
        break;
      case ProbeMode::kBOnly:
        probe = "_probe_b";
        break;
      case ProbeMode::kFullNoStore:
        probe = "_probe_full_nostore";
        break;
      case ProbeMode::kMatrixOnly:
        probe = "_probe_matrix";
        break;
    }
    char path[512];
    const int written = std::snprintf(path, sizeof(path), "%s/moe_sve_%s_m%d_d%d%s.bin", directory, operation, rows_,
                                      degree_, probe);
    if (written <= 0 || static_cast<size_t>(written) >= sizeof(path)) {
      return;
    }
    if (std::FILE* file = std::fopen(path, "wb")) {
      std::fwrite(getCode(), 1, getSize(), file);
      std::fclose(file);
    }
  }

  void save_callee_simd() {
    stp(d8, d9, pre_ptr(sp, -16));
    stp(d10, d11, pre_ptr(sp, -16));
    stp(d12, d13, pre_ptr(sp, -16));
    stp(d14, d15, pre_ptr(sp, -16));
  }

  void restore_callee_simd() {
    ldp(d14, d15, post_ptr(sp, 16));
    ldp(d12, d13, post_ptr(sp, 16));
    ldp(d10, d11, post_ptr(sp, 16));
    ldp(d8, d9, post_ptr(sp, 16));
  }

  void load_b(int register_base) {
    ld1h(ZRegH(register_base), p0 / T_z, ptr(x14));
    ld1h(ZRegH(register_base + 1), p0 / T_z, ptr(x14, 1, MUL_VL));
    ld1h(ZRegH(register_base + 2), p0 / T_z, ptr(x14, 2, MUL_VL));
    ld1h(ZRegH(register_base + 3), p0 / T_z, ptr(x14, 3, MUL_VL));
    add(x14, x14, x9, LSL, 2);
  }

  void compute_pairs(int begin_pair, int end_pair, int a_register_base, int b_register_base) {
    if (probe_mode_ == ProbeMode::kBOnly) {
      return;
    }
    for (int column = 0; column < 4; ++column) {
      for (int pair = begin_pair; pair < end_pair; ++pair) {
        bfmmla(accumulator(pair, column), ZRegH(a_register_base + pair - begin_pair), ZRegH(b_register_base + column));
      }
    }
  }

  void load_ab_small(int a_register_base, int b_register_base) {
    load_b(b_register_base);
    if (probe_mode_ != ProbeMode::kBOnly) {
      for (int pair = 0; pair < row_pairs_; ++pair) {
        ld1rqh(ZRegH(a_register_base + pair), p0 / T_z, ptr(x13, pair * 16));
      }
    }
    add(x13, x13, physical_rows_ * 8);
  }

  // Match the static M2/M4/M8 state machine: compute the current K4 panel
  // while the alternate A/B register bank is already resident.
  void emit_small_double_buffered_k_loop() {
    constexpr int kCurrentABase = 0;
    constexpr int kCurrentBBase = 4;
    constexpr int kNextABase = 8;
    constexpr int kNextBBase = 12;
    Label k_loop;
    Label tail_current;
    Label tail_next;
    Label k_done;

    load_ab_small(kCurrentABase, kCurrentBBase);
    subs(w15, w15, 4);
    b(EQ, tail_current);

    L(k_loop);
    load_ab_small(kNextABase, kNextBBase);
    compute_pairs(0, row_pairs_, kCurrentABase, kCurrentBBase);
    subs(w15, w15, 4);
    b(EQ, tail_next);
    load_ab_small(kCurrentABase, kCurrentBBase);
    compute_pairs(0, row_pairs_, kNextABase, kNextBBase);
    subs(w15, w15, 4);
    b(GT, k_loop);

    L(tail_current);
    compute_pairs(0, row_pairs_, kCurrentABase, kCurrentBBase);
    b(k_done);

    L(tail_next);
    compute_pairs(0, row_pairs_, kNextABase, kNextBBase);
    L(k_done);
  }

  void emit_m12_k4() {
    if (probe_mode_ == ProbeMode::kMatrixOnly) {
      const int first_group = std::min(row_pairs_, 4);
      const int early_pairs = std::min(first_group, 2);
      compute_pairs(0, early_pairs, 0, 4);
      compute_pairs(early_pairs, first_group, early_pairs, 4);
      compute_pairs(4, row_pairs_, 0, 4);
      return;
    }
    load_b(4);
    const int first_group = std::min(row_pairs_, 4);
    for (int pair = 0; pair < first_group; ++pair) {
      ld1rqh(ZRegH(pair), p0 / T_z, ptr(x13, pair * 16));
    }
    if (physical_rows_ == 12) {
      const int early_pairs = std::min(first_group, 2);
      compute_pairs(0, early_pairs, 0, 4);
      for (int pair = 4; pair < row_pairs_; ++pair) {
        ld1rqh(ZRegH(pair - 4), p0 / T_z, ptr(x13, pair * 16));
      }
      compute_pairs(early_pairs, first_group, early_pairs, 4);
      compute_pairs(4, row_pairs_, 0, 4);
    } else {
      compute_pairs(0, first_group, 0, 4);
    }
    add(x13, x13, physical_rows_ * 8);
  }

  void emit_m12_k_loop() {
    Label k_loop;
    L(k_loop);
    emit_m12_k4();
    emit_m12_k4();
    subs(w15, w15, 8);
    b(GT, k_loop);
  }

  void load_w8_b() {
    ld1sb(z6.s, p1 / T_z, ptr(x14));
    add(x14, x14, x19);
    ld1sb(z7.s, p1 / T_z, ptr(x14));
    add(x14, x14, x19);
    scvtf(z6.s, p1 / T_m, z6.s);
    scvtf(z7.s, p1 / T_m, z7.s);
    bfcvt(z6.h, p1 / T_m, z6.s);
    bfcvtnt(z6.h, p1 / T_m, z7.s);
  }

  void emit_w8_k4() {
    for (int pair = 0; pair < row_pairs_; ++pair) {
      ld1rqh(ZRegH(pair), p0 / T_z, ptr(x13, pair * 16));
    }
    for (int column = 0; column < 4; ++column) {
      load_w8_b();
      for (int pair = 0; pair < row_pairs_; ++pair) {
        bfmmla(accumulator(pair, column), ZRegH(pair), z6.h);
      }
    }
    add(x13, x13, physical_rows_ * 8);
  }

  void emit_w8_k_loop() {
    Label k_loop;
    L(k_loop);
    emit_w8_k4();
    subs(w15, w15, 4);
    b(GT, k_loop);
  }

  void apply_w8_scales() {
    for (int column = 0; column < 4; ++column) {
      ld1w(z6.s, p1 / T_z, ptr(x20));
      add(x20, x20, x9);
      for (int pair = 0; pair < row_pairs_; ++pair) {
        fmul(accumulator(pair, column), p1 / T_m, z6.s);
      }
    }
  }

  void build_row_offsets(bool direct, bool output_bf16) {
    index(z0.s, 0, 1);
    mov(z1.d, z0.d);
    and_(z1.s, 3);
    mov(z2.d, z0.d);
    lsr(z2.s, z2.s, 2);
    lsl(z2.s, z2.s, output_bf16 ? 4 : 5);
    mov(z3.d, z1.d);
    and_(z3.s, 1);
    lsl(z3.s, z3.s, output_bf16 ? 1 : 2);
    add(z0.s, z2.s, z3.s);
    lsr(z1.s, z1.s, 1);
    and_(z1.s, 1);
    cmpeq(p4.s, p1 / T_z, z1.s, 0);
    if (!direct) {
      dup(z3.s, w16);
      mul(z1.s, p1 / T_m, z3.s);
      add(z0.s, p1 / T_m, z1.s);
    }
  }

  void store_gemm_bf16_vectors(const XReg& base, const ZRegS& offsets, const PReg& predicate, int pair) {
    for (int column = 0; column < 4; ++column) {
      const ZRegS value = accumulator(pair, column);
      bfcvt(ZRegH(value.getIdx()), p1 / T_m, value);
      if (column == 0) {
        st1h(value, predicate, ptr(base, offsets, SXTW));
      } else {
        add(x15, base, column * 4);
        st1h(value, predicate, ptr(x15, offsets, SXTW));
      }
    }
  }

  void store_w2_vectors(const XReg& base, const ZRegS& offsets, const PReg& predicate, int pair) {
    for (int column = 0; column < 4; ++column) {
      if (column == 0) {
        st1w(accumulator(pair, column), predicate, ptr(base, offsets, SXTW));
      } else {
        add(x15, base, column * 8);
        st1w(accumulator(pair, column), predicate, ptr(x15, offsets, SXTW));
      }
    }
  }

  void store_w2() {
    const bool direct = operation_ == Operation::kW2Direct || operation_ == Operation::kW8W2Direct;
    const bool output_bf16 = operation_ == Operation::kGemmBf16;
    build_row_offsets(direct, output_bf16);
    for (int pair = 0; pair < row_pairs_; ++pair) {
      const bool partial = (rows_ & 1) != 0 && pair == row_pairs_ - 1;
      const PReg predicate = partial ? p4 : p1;
      if (direct) {
        ldr(x13, ptr(x8, pair * 16));
        madd(x13, x13, x16, x7);
        if (partial) {
          store_w2_vectors(x13, z0.s, predicate, pair);
        } else {
          ldp(x14, x15, ptr(x8, pair * 16));
          sub(x15, x15, x14);
          mul(x15, x15, x16);
          dup(z2.s, w15);
          mul(z2.s, p1 / T_m, z1.s);
          add(z2.s, z0.s, z2.s);
          store_w2_vectors(x13, z2.s, predicate, pair);
        }
      } else {
        if (pair == 0) {
          mov(x13, x7);
        } else {
          mov(x14, pair * 2);
          madd(x13, x14, x16, x7);
        }
        if (output_bf16) {
          store_gemm_bf16_vectors(x13, z0.s, predicate, pair);
        } else {
          store_w2_vectors(x13, z0.s, predicate, pair);
        }
      }
    }
  }

  void build_packc_offsets() {
    index(z0.s, 0, 1);
    mov(z1.d, z0.d);
    lsr(z1.s, z1.s, 2);
    if (physical_rows_ == 12) {
      mov(z5.d, z1.d);
      lsl(z1.s, z1.s, 6);
      lsl(z5.s, z5.s, 5);
      add(z5.s, z5.s, z1.s);
    } else {
      lsl(z5.s, z1.s, 6);
    }
    mov(z2.d, z0.d);
    and_(z2.s, 3);
    mov(z3.d, z2.d);
    and_(z3.s, 1);
    lsl(z3.s, z3.s, 1);
    add(z5.s, z5.s, z3.s);
    lsr(z2.s, z2.s, 1);
    and_(z2.s, 1);
    cmpeq(p4.s, p1 / T_z, z2.s, 0);
    lsl(z2.s, z2.s, 3);
    add(z5.s, z5.s, z2.s);
  }

  void load_silu_constant(int offset) {
    if (offset == 0) {
      mov(x14, x8);
    } else {
      add(x14, x8, offset);
    }
    ld1rw(z4.s, p1 / T_z, ptr(x14));
  }

  void silu_to_bf16(const ZRegS& gate, const ZRegS& up) {
    if (clamp_swiglu_) {
      load_silu_constant(offsetof(SiluConstants, swiglu_limit));
      fmin(gate, p1 / T_m, z4.s);
      fmin(up, p1 / T_m, z4.s);
      fneg(z0.s, p1 / T_m, z4.s);
      fmax(up, p1 / T_m, z0.s);
    }
    fneg(z0.s, p1 / T_m, gate);
    load_silu_constant(offsetof(SiluConstants, clamp_hi));
    fmin(z0.s, p1 / T_m, z4.s);
    if (!clamp_swiglu_) {
      load_silu_constant(offsetof(SiluConstants, clamp_lo));
      fmax(z0.s, p1 / T_m, z4.s);
    }
    mov(z1.d, z0.d);
    load_silu_constant(offsetof(SiluConstants, inv_ln2));
    fmul(z1.s, p1 / T_m, z4.s);
    frintn(z1.s, p1 / T_m, z1.s);
    fcvtzs(z2.s, p1 / T_m, z1.s);
    load_silu_constant(offsetof(SiluConstants, ln2));
    fmls(z0.s, p1 / T_m, z1.s, z4.s);
    if (degree_ == 4) {
      load_silu_constant(offsetof(SiluConstants, c4));
      mov(z3.d, z4.d);
    } else if (degree_ == 5) {
      load_silu_constant(offsetof(SiluConstants, c5));
      mov(z3.d, z4.d);
      load_silu_constant(offsetof(SiluConstants, c4));
      fmla(z4.s, p1 / T_m, z3.s, z0.s);
      mov(z3.d, z4.d);
    } else {
      load_silu_constant(offsetof(SiluConstants, c6));
      mov(z3.d, z4.d);
      load_silu_constant(offsetof(SiluConstants, c5));
      fmla(z4.s, p1 / T_m, z3.s, z0.s);
      mov(z3.d, z4.d);
      load_silu_constant(offsetof(SiluConstants, c4));
      fmla(z4.s, p1 / T_m, z3.s, z0.s);
      mov(z3.d, z4.d);
    }
    load_silu_constant(offsetof(SiluConstants, c3));
    fmla(z4.s, p1 / T_m, z3.s, z0.s);
    mov(z3.d, z4.d);
    load_silu_constant(offsetof(SiluConstants, half));
    fmla(z4.s, p1 / T_m, z3.s, z0.s);
    mov(z3.d, z4.d);
    load_silu_constant(offsetof(SiluConstants, one));
    fmla(z4.s, p1 / T_m, z3.s, z0.s);
    mov(z3.d, z4.d);
    load_silu_constant(offsetof(SiluConstants, one));
    fmla(z4.s, p1 / T_m, z3.s, z0.s);
    mov(z3.d, z4.d);
    mov(z4.s, 127);
    add(z2.s, z2.s, z4.s);
    lsl(z2.s, z2.s, 23);
    fmul(z3.s, p1 / T_m, z2.s);
    load_silu_constant(offsetof(SiluConstants, one));
    fadd(z3.s, p1 / T_m, z4.s);
    mov(z0.d, ZRegD(gate.getIdx()));
    fmul(z0.s, p1 / T_m, up);
    fdiv(z0.s, p1 / T_m, z3.s);
    bfcvt(z0.h, p1 / T_m, z0.s);
  }

  void load_small_silu_constants() {
    mov(x14, x8);
    ld1rw(z6.s, p1 / T_z, ptr(x14));
    add(x14, x14, 4);
    ld1rw(z7.s, p1 / T_z, ptr(x14));
    add(x14, x14, 4);
    ld1rw(z8.s, p1 / T_z, ptr(x14));
    add(x14, x14, 4);
    ld1rw(z9.s, p1 / T_z, ptr(x14));
    add(x14, x14, 4);
    ld1rw(z10.s, p1 / T_z, ptr(x14));
    add(x14, x14, 4);
    ld1rw(z11.s, p1 / T_z, ptr(x14));
    add(x14, x14, 4);
    ld1rw(z12.s, p1 / T_z, ptr(x14));
    add(x14, x14, 4);
    ld1rw(z13.s, p1 / T_z, ptr(x14));
    add(x14, x14, 4);
    ld1rw(z14.s, p1 / T_z, ptr(x14));
    add(x14, x14, 4);
    if (clamp_swiglu_) {
      add(x14, x8, offsetof(SiluConstants, swiglu_limit));
    }
    ld1rw(z15.s, p1 / T_z, ptr(x14));
  }

  void silu_to_bf16_cached(const ZRegS& gate, const ZRegS& up) {
    if (clamp_swiglu_) {
      fmin(gate, p1 / T_m, z15.s);
      fmin(up, p1 / T_m, z15.s);
      fneg(z4.s, p1 / T_m, z15.s);
      fmax(up, p1 / T_m, z4.s);
    }
    fneg(z0.s, p1 / T_m, gate);
    fmin(z0.s, p1 / T_m, z14.s);
    if (!clamp_swiglu_) {
      fmax(z0.s, p1 / T_m, z15.s);
    }
    mov(z1.d, z0.d);
    fmul(z1.s, p1 / T_m, z8.s);
    frintn(z1.s, p1 / T_m, z1.s);
    fcvtzs(z2.s, p1 / T_m, z1.s);
    fmls(z0.s, p1 / T_m, z1.s, z9.s);
    if (degree_ == 4) {
      mov(z1.d, z11.d);
    } else if (degree_ == 5) {
      mov(z3.d, z12.d);
      mov(z1.d, z11.d);
      fmla(z1.s, p1 / T_m, z3.s, z0.s);
    } else {
      mov(z1.d, z13.d);
      mov(z3.d, z12.d);
      fmla(z3.s, p1 / T_m, z1.s, z0.s);
      mov(z1.d, z11.d);
      fmla(z1.s, p1 / T_m, z3.s, z0.s);
    }
    mov(z3.d, z10.d);
    fmla(z3.s, p1 / T_m, z1.s, z0.s);
    mov(z1.d, z7.d);
    fmla(z1.s, p1 / T_m, z3.s, z0.s);
    mov(z3.d, z6.d);
    fmla(z3.s, p1 / T_m, z1.s, z0.s);
    mov(z1.d, z6.d);
    fmla(z1.s, p1 / T_m, z3.s, z0.s);
    mov(z4.s, 127);
    add(z2.s, z2.s, z4.s);
    lsl(z2.s, z2.s, 23);
    fmul(z1.s, p1 / T_m, z2.s);
    fadd(z1.s, p1 / T_m, z6.s);
    mov(z0.d, ZRegD(gate.getIdx()));
    fmul(z0.s, p1 / T_m, up);
    fdiv(z0.s, p1 / T_m, z1.s);
    bfcvt(z0.h, p1 / T_m, z0.s);
  }

  void load_m12_silu_constants() {
    mov(x14, x8);
    ld1rw(z6.s, p1 / T_z, ptr(x14));
    add(x14, x14, 4);
    ld1rw(z7.s, p1 / T_z, ptr(x14));
    add(x14, x14, 4);
    ld1rw(z24.s, p1 / T_z, ptr(x14));
    add(x14, x14, 4);
    ld1rw(z25.s, p1 / T_z, ptr(x14));
    add(x14, x14, 4);
    ld1rw(z26.s, p1 / T_z, ptr(x14));
    add(x14, x14, 4);
    ld1rw(z27.s, p1 / T_z, ptr(x14));
    add(x14, x14, 4);
    ld1rw(z28.s, p1 / T_z, ptr(x14));
    add(x14, x14, 4);
    ld1rw(z29.s, p1 / T_z, ptr(x14));
    add(x14, x14, 4);
    ld1rw(z30.s, p1 / T_z, ptr(x14));
    add(x14, x14, 4);
    if (clamp_swiglu_) {
      add(x14, x8, offsetof(SiluConstants, swiglu_limit));
    }
    ld1rw(z31.s, p1 / T_z, ptr(x14));
  }

  void silu_pair_to_bf16_cached(int pair) {
    const ZRegS gate0 = accumulator(pair, 0);
    const ZRegS gate1 = accumulator(pair, 1);
    const ZRegS up0 = accumulator(pair, 2);
    const ZRegS up1 = accumulator(pair, 3);
    if (clamp_swiglu_) {
      fmin(gate0, p1 / T_m, z31.s);
      fmin(gate1, p1 / T_m, z31.s);
      fmin(up0, p1 / T_m, z31.s);
      fmin(up1, p1 / T_m, z31.s);
      fneg(z4.s, p1 / T_m, z31.s);
      fmax(up0, p1 / T_m, z4.s);
      fmax(up1, p1 / T_m, z4.s);
    }
    fneg(z0.s, p1 / T_m, gate0);
    fneg(z1.s, p1 / T_m, gate1);
    fmin(z0.s, p1 / T_m, z30.s);
    fmin(z1.s, p1 / T_m, z30.s);
    if (!clamp_swiglu_) {
      fmax(z0.s, p1 / T_m, z31.s);
      fmax(z1.s, p1 / T_m, z31.s);
    }
    mov(z2.d, z0.d);
    mov(z3.d, z1.d);
    fmul(z2.s, p1 / T_m, z24.s);
    fmul(z3.s, p1 / T_m, z24.s);
    frintn(z2.s, p1 / T_m, z2.s);
    frintn(z3.s, p1 / T_m, z3.s);
    fmls(z0.s, p1 / T_m, z2.s, z25.s);
    fmls(z1.s, p1 / T_m, z3.s, z25.s);
    fcvtzs(z2.s, p1 / T_m, z2.s);
    fcvtzs(z3.s, p1 / T_m, z3.s);
    fmul(gate0, p1 / T_m, up0);
    fmul(gate1, p1 / T_m, up1);
    if (degree_ == 4) {
      mov(ZRegD(up0.getIdx()), z27.d);
      mov(ZRegD(up1.getIdx()), z27.d);
    } else if (degree_ == 5) {
      mov(ZRegD(up0.getIdx()), z27.d);
      mov(ZRegD(up1.getIdx()), z27.d);
      fmla(up0, p1 / T_m, z28.s, z0.s);
      fmla(up1, p1 / T_m, z28.s, z1.s);
    } else {
      mov(ZRegD(up0.getIdx()), z28.d);
      mov(ZRegD(up1.getIdx()), z28.d);
      fmla(up0, p1 / T_m, z29.s, z0.s);
      fmla(up1, p1 / T_m, z29.s, z1.s);
      fmad(up0, p1 / T_m, z0.s, z27.s);
      fmad(up1, p1 / T_m, z1.s, z27.s);
    }
    fmad(up0, p1 / T_m, z0.s, z26.s);
    fmad(up1, p1 / T_m, z1.s, z26.s);
    fmad(up0, p1 / T_m, z0.s, z7.s);
    fmad(up1, p1 / T_m, z1.s, z7.s);
    fmad(up0, p1 / T_m, z0.s, z6.s);
    fmad(up1, p1 / T_m, z1.s, z6.s);
    fmad(up0, p1 / T_m, z0.s, z6.s);
    fmad(up1, p1 / T_m, z1.s, z6.s);
    mov(z4.s, 127);
    add(z2.s, z2.s, z4.s);
    add(z3.s, z3.s, z4.s);
    lsl(z2.s, z2.s, 23);
    lsl(z3.s, z3.s, 23);
    fmul(up0, p1 / T_m, z2.s);
    fmul(up1, p1 / T_m, z3.s);
    fadd(up0, p1 / T_m, z6.s);
    fadd(up1, p1 / T_m, z6.s);
    fdiv(gate0, p1 / T_m, up0);
    fdiv(gate1, p1 / T_m, up1);
    bfcvt(ZRegH(gate0.getIdx()), p1 / T_m, gate0);
    bfcvt(ZRegH(gate1.getIdx()), p1 / T_m, gate1);
  }

  void store_w13_pair_low_tmp(int pair) {
    const bool partial = (rows_ & 1) != 0 && pair == row_pairs_ - 1;
    const PReg predicate = partial ? p4 : p1;
    add(x13, x7, pair * 16);
    silu_to_bf16(accumulator(pair, 0), accumulator(pair, 2));
    st1h(z0.s, predicate, ptr(x13, z5.s, SXTW));
    add(x13, x13, 4);
    silu_to_bf16(accumulator(pair, 1), accumulator(pair, 3));
    st1h(z0.s, predicate, ptr(x13, z5.s, SXTW));
  }

  void store_w13_small_pair_cached(int pair) {
    const bool partial = (rows_ & 1) != 0 && pair == row_pairs_ - 1;
    const PReg predicate = partial ? p4 : p1;
    add(x13, x7, pair * 16);
    silu_to_bf16_cached(accumulator(pair, 0), accumulator(pair, 2));
    st1h(z0.s, predicate, ptr(x13, z5.s, SXTW));
    add(x13, x13, 4);
    silu_to_bf16_cached(accumulator(pair, 1), accumulator(pair, 3));
    st1h(z0.s, predicate, ptr(x13, z5.s, SXTW));
  }

  void store_w13() {
    build_packc_offsets();
    if (physical_rows_ == 12) {
      for (int pair = 4; pair < row_pairs_; ++pair) {
        store_w13_pair_low_tmp(pair);
      }
      load_m12_silu_constants();
      for (int pair = 0; pair < std::min(row_pairs_, 4); ++pair) {
        silu_pair_to_bf16_cached(pair);
        add(x13, x7, pair * 16);
        st1h(accumulator(pair, 0), p1, ptr(x13, z5.s, SXTW));
        add(x13, x13, 4);
        st1h(accumulator(pair, 1), p1, ptr(x13, z5.s, SXTW));
      }
      return;
    }
    load_small_silu_constants();
    for (int pair = 0; pair < row_pairs_; ++pair) {
      store_w13_small_pair_cached(pair);
    }
  }

  void generate() {
    if (w8_weights_) {
      stp(x19, x20, pre_ptr(sp, -16));
    }
    save_callee_simd();
    if (w8_weights_) {
      mov(x8, x4);
      mov(x4, x5);
    } else {
      mov(x8, x3);
    }
    ldr(w5, ptr(x4, kParamK));
    ldr(w6, ptr(x4, kParamNBegin));
    mov(w11, w5);
    mul(x6, x6, x11);
    if (!w8_weights_) {
      lsl(x6, x6, 1);
    }
    add(x6, x1, x6);
    ldr(w12, ptr(x4, kParamN));
    ldr(w7, ptr(x4, kParamLdc));
    mov(w16, w7);
    lsl(x16, x16, operation_ == Operation::kGemmBf16 ? 1 : 2);
    if (w8_weights_) {
      mov(x7, x3);
    } else {
      mov(x7, x2);
    }
    mov(x9, kVectorBytes);
    if (w8_weights_) {
      mov(x19, kVectorBytes / 4);
    }
    mov(x10, kNTile);
    mov(w11, w5);
    mul(x11, x11, x9);
    if (w8_weights_) {
      lsr(x11, x11, 1);
      ldr(w15, ptr(x4, kParamNBegin));
      lsl(x15, x15, 3);
      add(x2, x2, x15);
    }
    if (operation_ == Operation::kW13 || operation_ == Operation::kW13Clamped ||
        operation_ == Operation::kW8W13 || operation_ == Operation::kW8W13Clamped) {
      mov(x17, physical_rows_ / 2);
      mul(x17, x17, x9);
    } else {
      if (operation_ == Operation::kGemmBf16) {
        mov(x17, x9);
      } else {
        lsl(x17, x9, 1);
      }
    }
    ptrue(p0.b);
    ptrue(p1.s);

    Label n_loop;
    Label done;
    L(n_loop);
    cmp(x12, 0);
    b(LE, done);
    for (int pair = 0; pair < row_pairs_; ++pair) {
      for (int column = 0; column < 4; ++column) {
        mov(accumulator(pair, column), 0);
      }
    }
    mov(x13, x0);
    mov(x14, x6);
    mov(w15, w5);
    if (probe_mode_ == ProbeMode::kMatrixOnly) {
      for (int reg = 0; reg < 8; ++reg) {
        mov(ZRegD(reg), 0);
      }
    }
    if (w8_weights_) {
      emit_w8_k_loop();
      mov(x20, x2);
      apply_w8_scales();
    } else if (physical_rows_ == 8) {
      emit_small_double_buffered_k_loop();
    } else {
      emit_m12_k_loop();
    }

    if (probe_mode_ == ProbeMode::kNone) {
      if (operation_ == Operation::kW13 || operation_ == Operation::kW13Clamped ||
          operation_ == Operation::kW8W13 || operation_ == Operation::kW8W13Clamped) {
        store_w13();
      } else {
        store_w2();
      }
    }
    add(x6, x6, x11);
    if (w8_weights_) {
      add(x2, x2, x9, LSL, 2);
    }
    add(x7, x7, x17);
    sub(x12, x12, x10);
    b(n_loop);

    L(done);
    restore_callee_simd();
    if (w8_weights_) {
      ldp(x19, x20, post_ptr(sp, 16));
    }
    ret();
  }

  Operation operation_;
  int rows_;
  int degree_;
  int row_pairs_;
  int accumulator_base_;
  int physical_rows_;
  ProbeMode probe_mode_;
  bool clamp_swiglu_ = false;
  bool w8_weights_ = false;
};

class SveW8DequantGenerator final : public CodeGenerator {
 public:
  SveW8DequantGenerator() : CodeGenerator(16 * 1024, AutoGrow) {
    generate();
    readyRE();
  }

  W8DequantFn function() const { return getCode<W8DequantFn>(); }

 private:
  void generate() {
    ptrue(p0.s);
    ptrue(p1.h);
    mov(x5, x0);
    mov(x6, x2);
    mov(w8, w4);

    Label n_loop;
    Label k_loop;
    Label done;
    L(n_loop);
    cmp(w8, 0);
    b(LE, done);
    mov(w9, w3);
    L(k_loop);
    mov(x7, x1);
    for (int column = 0; column < 4; ++column) {
      ld1sb(z0.s, p0 / T_z, ptr(x5));
      add(x5, x5, kVectorBytes / 4);
      ld1sb(z1.s, p0 / T_z, ptr(x5));
      add(x5, x5, kVectorBytes / 4);
      scvtf(z0.s, p0 / T_m, z0.s);
      scvtf(z1.s, p0 / T_m, z1.s);
      ld1w(z2.s, p0 / T_z, ptr(x7));
      add(x7, x7, kVectorBytes);
      fmul(z0.s, p0 / T_m, z2.s);
      fmul(z1.s, p0 / T_m, z2.s);
      bfcvt(z0.h, p0 / T_m, z0.s);
      bfcvtnt(z0.h, p0 / T_m, z1.s);
      st1h(z0.h, p1, ptr(x6));
      add(x6, x6, kVectorBytes);
    }
    subs(w9, w9, 4);
    b(GT, k_loop);
    add(x1, x1, kVectorBytes * 4);
    sub(w8, w8, kNTile);
    b(n_loop);
    L(done);
    ret();
  }
};

struct KernelHandle {
  std::shared_ptr<SveFusedGenerator> owner;
  KernelFn function = nullptr;
  W8KernelFn w8_function = nullptr;
  std::string error;
};

KernelHandle create_kernel(const KernelKey& key) {
  KernelHandle handle;
  try {
    handle.owner = std::make_shared<SveFusedGenerator>(key.operation, key.rows, key.degree, key.probe_mode);
    if (key.operation == Operation::kW8W13 || key.operation == Operation::kW8W13Clamped ||
        key.operation == Operation::kW8W2Direct) {
      handle.w8_function = handle.owner->getCode<W8KernelFn>();
    } else {
      handle.function = handle.owner->function();
    }
  } catch (const std::exception& exception) {
    handle.error = exception.what();
  } catch (...) {
    handle.error = "unknown xbyak_aarch64 generation failure";
  }
  return handle;
}

struct KernelCacheSlot {
  std::once_flag once;
  KernelHandle handle;
};

constexpr size_t kOperationCount = 9;
constexpr size_t kRowCount = 12;
constexpr size_t kDegreeCount = 3;
constexpr size_t kProbeModeCount = 4;

size_t operation_index(Operation operation) { return static_cast<size_t>(operation); }

size_t degree_index(Operation operation, int degree) {
  return operation == Operation::kW13 || operation == Operation::kW13Clamped ||
                 operation == Operation::kW8W13 || operation == Operation::kW8W13Clamped
             ? static_cast<size_t>(degree - 4)
             : 0;
}

size_t probe_index(ProbeMode mode) {
  switch (mode) {
    case ProbeMode::kNone:
      return 0;
    case ProbeMode::kBOnly:
      return 1;
    case ProbeMode::kFullNoStore:
      return 2;
    case ProbeMode::kMatrixOnly:
      return 3;
  }
  throw std::invalid_argument("invalid SVE JIT calibration mode");
}

KernelHandle& cached_kernel(const KernelKey& key) {
  using ProbeCache = std::array<KernelCacheSlot, kProbeModeCount>;
  using DegreeCache = std::array<ProbeCache, kDegreeCount>;
  using RowCache = std::array<DegreeCache, kRowCount>;
  using OperationCache = std::array<RowCache, kOperationCount>;
  static OperationCache cache;
  KernelCacheSlot& slot =
      cache[operation_index(key.operation)][static_cast<size_t>(key.rows - 1)][degree_index(key.operation, key.degree)]
           [probe_index(key.probe_mode)];
  std::call_once(slot.once, [&slot, &key]() { slot.handle = create_kernel(key); });
  return slot.handle;
}

bool built() { return true; }

bool requested_for_current_build() { return implementation_mode() != ImplementationMode::kAsm; }

KernelFn get_kernel(Operation operation, int rows, int degree, std::string* error) {
  if (rows < 1 || rows > 12) {
    if (error != nullptr) {
      *error = "SVE Xbyak exact-M rows must be in [1, 12]";
    }
    return nullptr;
  }
  if ((operation == Operation::kW13 || operation == Operation::kW13Clamped) &&
      (degree < 4 || degree > 6)) {
    if (error != nullptr) {
      *error = "SVE Xbyak W13 degree must be 4, 5, or 6";
    }
    return nullptr;
  }
  const KernelKey key{operation, static_cast<uint8_t>(rows), static_cast<uint8_t>(degree), ProbeMode::kNone};
  KernelHandle& handle = cached_kernel(key);
  if (error != nullptr) {
    *error = handle.error;
  }
  return handle.function;
}

W8KernelFn get_w8_kernel(Operation operation, int rows, int degree, std::string* error) {
  const bool is_w8 = operation == Operation::kW8W13 || operation == Operation::kW8W13Clamped ||
                     operation == Operation::kW8W2Direct;
  if (!is_w8 || rows < 1 || rows > 12 ||
      ((operation == Operation::kW8W13 || operation == Operation::kW8W13Clamped) &&
       (degree < 4 || degree > 6))) {
    if (error != nullptr) {
      *error = "invalid SVE Xbyak W8A16 operation, rows, or degree";
    }
    return nullptr;
  }
  const KernelKey key{operation, static_cast<uint8_t>(rows), static_cast<uint8_t>(degree), ProbeMode::kNone};
  KernelHandle& handle = cached_kernel(key);
  if (error != nullptr) {
    *error = handle.error;
  }
  return handle.w8_function;
}

W8DequantFn get_w8_dequant_kernel(std::string* error) {
  struct DequantHandle {
    std::shared_ptr<SveW8DequantGenerator> owner;
    W8DequantFn function = nullptr;
    std::string error;
  };
  static std::once_flag once;
  static DequantHandle handle;
  std::call_once(once, [] {
    try {
      handle.owner = std::make_shared<SveW8DequantGenerator>();
      handle.function = handle.owner->function();
    } catch (const std::exception& exception) {
      handle.error = exception.what();
    } catch (...) {
      handle.error = "unknown xbyak_aarch64 W8 dequant generation failure";
    }
  });
  if (error != nullptr) {
    *error = handle.error;
  }
  return handle.function;
}

KernelFn get_probe_kernel(int rows, ProbeMode mode, std::string* error) {
  if (!probe_rows_supported(rows, mode) || mode == ProbeMode::kNone) {
    if (error != nullptr) {
      *error = "SVE JIT probe requires M1/M2, or an M12-compatible mode";
    }
    return nullptr;
  }
  const KernelKey key{Operation::kGemmF32, static_cast<uint8_t>(rows), 0, mode};
  KernelHandle& handle = cached_kernel(key);
  if (error != nullptr) {
    *error = handle.error;
  }
  return handle.function;
}

void prewarm(Operation operation, int degree) {
  const bool is_w8 = operation == Operation::kW8W13 || operation == Operation::kW8W13Clamped ||
                     operation == Operation::kW8W2Direct;
  for (int rows = 1; rows <= 12; ++rows) {
    std::string error;
    const bool resolved = is_w8 ? get_w8_kernel(operation, rows, degree, &error) != nullptr
                                : get_kernel(operation, rows, degree, &error) != nullptr;
    if (!resolved && implementation_mode() == ImplementationMode::kJit) {
      throw std::runtime_error("failed to generate SVE Xbyak kernel: " + error);
    }
  }
}

#endif

KernelFn get_gemm_f32_kernel(int rows, std::string* error) {
  return get_kernel(Operation::kGemmF32, rows, 0, error);
}

KernelFn get_gemm_bf16_kernel(int rows, std::string* error) {
  return get_kernel(Operation::kGemmBf16, rows, 0, error);
}

}  // namespace fused_cpp::moe_sve::jit
