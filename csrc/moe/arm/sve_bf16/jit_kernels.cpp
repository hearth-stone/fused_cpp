#include "jit_kernels.h"

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

void prewarm(Operation, int) {}

#else

using namespace Xbyak_aarch64;

struct KernelKey {
  Operation operation = Operation::kW13;
  uint8_t rows = 0;
  uint8_t degree = 0;
};

class SveFusedGenerator final : public CodeGenerator {
 public:
  SveFusedGenerator(Operation operation, int rows, int degree)
      : CodeGenerator(64 * 1024, AutoGrow),
        operation_(operation),
        rows_(rows),
        degree_(degree),
        row_pairs_((rows + 1) / 2),
        accumulator_base_(rows <= 8 ? 32 - row_pairs_ * 4 : 8),
        physical_rows_(rows <= 8 ? 8 : 12) {
    if (rows_ < 1 || rows_ > 12) {
      throw std::invalid_argument("SVE JIT rows must be in [1, 12]");
    }
    if (operation_ == Operation::kW13 && (degree_ < 4 || degree_ > 6)) {
      throw std::invalid_argument("SVE JIT W13 degree must be 4, 5, or 6");
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
    const char* operation = operation_ == Operation::kW13 ? "w13" : (operation_ == Operation::kW2 ? "w2" : "w2_direct");
    char path[512];
    const int written =
        std::snprintf(path, sizeof(path), "%s/moe_sve_%s_m%d_d%d.bin", directory, operation, rows_, degree_);
    if (written <= 0 || static_cast<size_t>(written) >= sizeof(path)) {
      return;
    }
    if (std::FILE* file = std::fopen(path, "wb")) {
      std::fwrite(getCode(), 1, getSize(), file);
      std::fclose(file);
    }
  }

  void save_callee_simd() {
    if (operation_ != Operation::kW13 && accumulator_base_ >= 16) {
      return;
    }
    stp(d8, d9, pre_ptr(sp, -16));
    stp(d10, d11, pre_ptr(sp, -16));
    stp(d12, d13, pre_ptr(sp, -16));
    stp(d14, d15, pre_ptr(sp, -16));
  }

  void restore_callee_simd() {
    if (operation_ != Operation::kW13 && accumulator_base_ >= 16) {
      return;
    }
    ldp(d14, d15, post_ptr(sp, 16));
    ldp(d12, d13, post_ptr(sp, 16));
    ldp(d10, d11, post_ptr(sp, 16));
    ldp(d8, d9, post_ptr(sp, 16));
  }

  void load_b() {
    ld1h(z4.h, p0 / T_z, ptr(x14));
    ld1h(z5.h, p0 / T_z, ptr(x14, 1, MUL_VL));
    ld1h(z6.h, p0 / T_z, ptr(x14, 2, MUL_VL));
    ld1h(z7.h, p0 / T_z, ptr(x14, 3, MUL_VL));
    add(x14, x14, x9, LSL, 2);
  }

  void compute_pairs(int begin_pair, int end_pair, int a_register_base) {
    for (int column = 0; column < 4; ++column) {
      for (int pair = begin_pair; pair < end_pair; ++pair) {
        bfmmla(accumulator(pair, column), ZRegH(a_register_base + pair - begin_pair), ZRegH(4 + column));
      }
    }
  }

  void emit_k4() {
    load_b();
    const int first_group = std::min(row_pairs_, 4);
    for (int pair = 0; pair < first_group; ++pair) {
      ld1rqh(ZRegH(pair), p0 / T_z, ptr(x13, pair * 16));
    }
    if (physical_rows_ == 12) {
      const int early_pairs = std::min(first_group, 2);
      compute_pairs(0, early_pairs, 0);
      for (int pair = 4; pair < row_pairs_; ++pair) {
        ld1rqh(ZRegH(pair - 4), p0 / T_z, ptr(x13, pair * 16));
      }
      compute_pairs(early_pairs, first_group, early_pairs);
      compute_pairs(4, row_pairs_, 0);
    } else {
      compute_pairs(0, first_group, 0);
    }
    add(x13, x13, physical_rows_ * 8);
  }

  void build_row_offsets(bool direct) {
    index(z0.s, 0, 1);
    mov(z1.d, z0.d);
    and_(z1.s, 3);
    mov(z2.d, z0.d);
    lsr(z2.s, z2.s, 2);
    lsl(z2.s, z2.s, 5);
    mov(z3.d, z1.d);
    and_(z3.s, 1);
    lsl(z3.s, z3.s, 2);
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
    const bool direct = operation_ == Operation::kW2Direct;
    build_row_offsets(direct);
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
        store_w2_vectors(x13, z0.s, predicate, pair);
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
    fneg(z0.s, p1 / T_m, gate);
    load_silu_constant(offsetof(SiluConstants, clamp_hi));
    fmin(z0.s, p1 / T_m, z4.s);
    load_silu_constant(offsetof(SiluConstants, clamp_lo));
    fmax(z0.s, p1 / T_m, z4.s);
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
    ld1rw(z15.s, p1 / T_z, ptr(x14));
  }

  void silu_to_bf16_cached(const ZRegS& gate, const ZRegS& up) {
    fneg(z0.s, p1 / T_m, gate);
    fmin(z0.s, p1 / T_m, z14.s);
    fmax(z0.s, p1 / T_m, z15.s);
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
    ld1rw(z31.s, p1 / T_z, ptr(x14));
  }

  void silu_pair_to_bf16_cached(int pair) {
    const ZRegS gate0 = accumulator(pair, 0);
    const ZRegS gate1 = accumulator(pair, 1);
    const ZRegS up0 = accumulator(pair, 2);
    const ZRegS up1 = accumulator(pair, 3);
    fneg(z0.s, p1 / T_m, gate0);
    fneg(z1.s, p1 / T_m, gate1);
    fmin(z0.s, p1 / T_m, z30.s);
    fmin(z1.s, p1 / T_m, z30.s);
    fmax(z0.s, p1 / T_m, z31.s);
    fmax(z1.s, p1 / T_m, z31.s);
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
      const bool partial = (rows_ & 1) != 0 && pair == row_pairs_ - 1;
      const PReg predicate = partial ? p4 : p1;
      add(x13, x7, pair * 16);
      silu_to_bf16_cached(accumulator(pair, 0), accumulator(pair, 2));
      st1h(z0.s, predicate, ptr(x13, z5.s, SXTW));
      add(x13, x13, 4);
      silu_to_bf16_cached(accumulator(pair, 1), accumulator(pair, 3));
      st1h(z0.s, predicate, ptr(x13, z5.s, SXTW));
    }
  }

  void generate() {
    save_callee_simd();
    mov(x8, x3);
    ldr(w5, ptr(x4, kParamK));
    ldr(w6, ptr(x4, kParamNBegin));
    mov(w11, w5);
    mul(x6, x6, x11);
    lsl(x6, x6, 1);
    add(x6, x1, x6);
    ldr(w12, ptr(x4, kParamN));
    ldr(w7, ptr(x4, kParamLdc));
    mov(w16, w7);
    lsl(x16, x16, 2);
    mov(x7, x2);
    cntb(x9);
    lsr(x10, x9, 1);
    mov(w11, w5);
    mul(x11, x11, x9);
    if (operation_ == Operation::kW13) {
      mov(x17, physical_rows_ / 2);
      mul(x17, x17, x9);
    } else {
      lsl(x17, x9, 1);
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
    Label k_loop;
    L(k_loop);
    emit_k4();
    emit_k4();
    subs(w15, w15, 8);
    b(GT, k_loop);

    if (operation_ == Operation::kW13) {
      store_w13();
    } else {
      store_w2();
    }
    add(x6, x6, x11);
    add(x7, x7, x17);
    sub(x12, x12, x10);
    b(n_loop);

    L(done);
    restore_callee_simd();
    ret();
  }

  Operation operation_;
  int rows_;
  int degree_;
  int row_pairs_;
  int accumulator_base_;
  int physical_rows_;
};

struct KernelHandle {
  std::shared_ptr<SveFusedGenerator> owner;
  KernelFn function = nullptr;
  std::string error;
};

KernelHandle create_kernel(const KernelKey& key) {
  KernelHandle handle;
  try {
    handle.owner = std::make_shared<SveFusedGenerator>(key.operation, key.rows, key.degree);
    handle.function = handle.owner->function();
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

constexpr size_t kOperationCount = 3;
constexpr size_t kRowCount = 12;
constexpr size_t kDegreeCount = 3;

size_t operation_index(Operation operation) { return static_cast<size_t>(operation); }

size_t degree_index(Operation operation, int degree) {
  return operation == Operation::kW13 ? static_cast<size_t>(degree - 4) : 0;
}

KernelHandle& cached_kernel(const KernelKey& key) {
  static std::array<std::array<std::array<KernelCacheSlot, kDegreeCount>, kRowCount>, kOperationCount> cache;
  KernelCacheSlot& slot =
      cache[operation_index(key.operation)][static_cast<size_t>(key.rows - 1)][degree_index(key.operation, key.degree)];
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
  if (operation == Operation::kW13 && (degree < 4 || degree > 6)) {
    if (error != nullptr) {
      *error = "SVE Xbyak W13 degree must be 4, 5, or 6";
    }
    return nullptr;
  }
  const KernelKey key{operation, static_cast<uint8_t>(rows), static_cast<uint8_t>(degree)};
  KernelHandle& handle = cached_kernel(key);
  if (error != nullptr) {
    *error = handle.error;
  }
  return handle.function;
}

void prewarm(Operation operation, int degree) {
  for (int rows = 1; rows <= 12; ++rows) {
    std::string error;
    const KernelFn function = get_kernel(operation, rows, degree, &error);
    if (function == nullptr && implementation_mode() == ImplementationMode::kJit) {
      throw std::runtime_error("failed to generate SVE Xbyak kernel: " + error);
    }
  }
}

#endif

}  // namespace fused_cpp::moe_sve::jit
