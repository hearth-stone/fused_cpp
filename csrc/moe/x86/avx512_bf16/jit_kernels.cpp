// SPDX-License-Identifier: Apache-2.0
#include "kernels.h"

#include "backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#if defined(FUSED_CPP_MOE_HAS_XBYAK) && FUSED_CPP_MOE_HAS_XBYAK
#include <xbyak/xbyak.h>
#endif

namespace fused_cpp::moe::x86::avx512_bf16 {
namespace {

enum class ImplementationMode { kAuto, kJit, kIntrinsic };

ImplementationMode GetImplementationMode() {
  const char* raw = std::getenv("FUSED_CPP_MOE_AVX512_IMPL");
  const std::string value = raw == nullptr ? "auto" : raw;
  if (value.empty() || value == "auto") {
    return ImplementationMode::kAuto;
  }
  if (value == "jit") {
    return ImplementationMode::kJit;
  }
  if (value == "intrinsic") {
    return ImplementationMode::kIntrinsic;
  }
  throw std::runtime_error("FUSED_CPP_MOE_AVX512_IMPL must be auto, jit, or intrinsic; got '" + value + "'");
}

#if defined(FUSED_CPP_MOE_HAS_XBYAK) && FUSED_CPP_MOE_HAS_XBYAK

// ISA is part of the shared cache/factory contract so AVX-512 and AMX exact-M
// kernels cannot alias even when every other specialization field matches.
enum class X86JitIsa : uint8_t { kAvx512Bf16, kAmxBf16 };
enum class JitOperation : uint8_t { kW13, kW2 };
enum class JitOutput : uint8_t { kPackedBf16, kRouteF32, kContiguousF32, kDirectBf16 };
enum class AmxJitPattern : uint8_t { kM1N2, kM2N2, kM1N4 };
enum class AmxSiluEpilogue : uint8_t { kBaseline, kResident, kPipelined, kRcp14 };
enum class AmxW2Epilogue : uint8_t { kBaseline, kCombined, kTileStore };
enum class AmxTileStateMode : uint8_t { kPerCall, kMacroM };

// Conservative H=4096/F=512 crossover on Amazon C8i; forced patterns remain
// available for machine-specific validation.
constexpr int kAmxM1N4MinRows = 76;
constexpr int64_t kAmxW13CacheTargetBytes = int64_t{1} << 20;
constexpr int64_t kAmxW2CacheTargetBytes = int64_t{512} << 10;

size_t AmxPatternIndex(AmxJitPattern pattern) { return static_cast<size_t>(pattern); }

size_t AmxTileStateModeIndex(AmxTileStateMode mode) { return static_cast<size_t>(mode); }

AmxJitPattern ResolveAmxJitPattern(int rows) {
  const char* raw = std::getenv("FUSED_CPP_MOE_AMX_PATTERN");
  const std::string value = raw == nullptr ? "auto" : raw;
  if (value.empty() || value == "auto") {
    return rows >= kAmxM1N4MinRows ? AmxJitPattern::kM1N4 : AmxJitPattern::kM2N2;
  }
  if (value == "m1n2") {
    return AmxJitPattern::kM1N2;
  }
  if (value == "m2n2") {
    return AmxJitPattern::kM2N2;
  }
  if (value == "m1n4") {
    return AmxJitPattern::kM1N4;
  }
  throw std::runtime_error("FUSED_CPP_MOE_AMX_PATTERN must be auto, m1n2, m2n2, or m1n4; got '" + value + "'");
}

AmxSiluEpilogue ResolveAmxSiluEpilogue() {
  const char* raw = std::getenv("FUSED_CPP_MOE_AMX_SILU_EPILOGUE");
  const std::string value = raw == nullptr ? "auto" : raw;
  if (value.empty() || value == "auto") {
    return AmxSiluEpilogue::kResident;
  }
  if (value == "baseline") {
    return AmxSiluEpilogue::kBaseline;
  }
  if (value == "resident") {
    return AmxSiluEpilogue::kResident;
  }
  if (value == "pipelined") {
    return AmxSiluEpilogue::kPipelined;
  }
  if (value == "rcp14") {
    return AmxSiluEpilogue::kRcp14;
  }
  throw std::runtime_error(
      "FUSED_CPP_MOE_AMX_SILU_EPILOGUE must be auto, baseline, resident, pipelined, or rcp14; got '" + value + "'");
}

AmxW2Epilogue ResolveAmxW2Epilogue() {
  const char* raw = std::getenv("FUSED_CPP_MOE_AMX_W2_EPILOGUE");
  const std::string value = raw == nullptr ? "auto" : raw;
  if (value.empty() || value == "auto") {
    return AmxW2Epilogue::kBaseline;
  }
  if (value == "baseline") {
    return AmxW2Epilogue::kBaseline;
  }
  if (value == "combined") {
    return AmxW2Epilogue::kCombined;
  }
  if (value == "tile_store") {
    return AmxW2Epilogue::kTileStore;
  }
  throw std::runtime_error("FUSED_CPP_MOE_AMX_W2_EPILOGUE must be auto, baseline, combined, or tile_store; got '" +
                           value + "'");
}

AmxW2Epilogue EffectiveAmxW2Epilogue(AmxW2Epilogue epilogue, bool direct_bf16) {
  // TMM accumulators contain FP32. Direct top-k=1 output still needs a ZMM
  // conversion to BF16, so it cannot use TILESTORED as its final store.
  return direct_bf16 && epilogue == AmxW2Epilogue::kTileStore ? AmxW2Epilogue::kCombined : epilogue;
}

AmxTileStateMode ResolveAmxTileStateMode() {
  const char* raw = std::getenv("FUSED_CPP_MOE_AMX_TILE_STATE");
  const std::string value = raw == nullptr ? "auto" : raw;
  if (value.empty() || value == "auto" || value == "per_call") {
    return AmxTileStateMode::kPerCall;
  }
  if (value == "macro_m") {
    return AmxTileStateMode::kMacroM;
  }
  throw std::runtime_error("FUSED_CPP_MOE_AMX_TILE_STATE must be auto, per_call, or macro_m; got '" + value + "'");
}

AmxTileStateMode EffectiveAmxTileStateMode(AmxTileStateMode requested, int panel_count) {
  return requested == AmxTileStateMode::kMacroM && panel_count > 1 ? AmxTileStateMode::kMacroM
                                                                   : AmxTileStateMode::kPerCall;
}

int GetCacheBlockWindow(const char* environment, int automatic_blocks = 0) {
  const char* raw = std::getenv(environment);
  if (raw == nullptr || raw[0] == '\0') {
    return automatic_blocks;
  }
  if (std::strcmp(raw, "auto") == 0) {
    return automatic_blocks;
  }
  char* end = nullptr;
  const long parsed = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string(environment) + " must be auto or a non-negative integer, got '" + raw + "'");
  }
  return static_cast<int>(parsed);
}

int AutomaticAmxCacheBlocks(int k_pad, int64_t target_bytes, AmxJitPattern pattern) {
  const int64_t bytes_per_block = static_cast<int64_t>(k_pad) * 64;
  int blocks = static_cast<int>(std::max<int64_t>(1, target_bytes / bytes_per_block));
  if (pattern == AmxJitPattern::kM1N4 && blocks > 1) {
    blocks -= blocks % 2;
  }
  return blocks;
}

template <typename Function>
void ForEachCacheBlockWindow(int block_begin, int block_end, int window_blocks, Function&& function) {
  if (window_blocks <= 0) {
    function(block_begin, block_end);
    return;
  }
  for (int begin = block_begin; begin < block_end;) {
    const int count = std::min(window_blocks, block_end - begin);
    function(begin, begin + count);
    begin += count;
  }
}

struct KernelKey {
  X86JitIsa isa = X86JitIsa::kAvx512Bf16;
  JitOperation operation = JitOperation::kW13;
  uint8_t rows = 0;
  uint8_t n_valid = 0;
  uint8_t silu_degree = 0;
  JitOutput output = JitOutput::kPackedBf16;
  AmxJitPattern amx_pattern = AmxJitPattern::kM1N2;
  AmxSiluEpilogue amx_silu_epilogue = AmxSiluEpilogue::kBaseline;
  AmxW2Epilogue amx_w2_epilogue = AmxW2Epilogue::kBaseline;
  AmxTileStateMode amx_tile_state_mode = AmxTileStateMode::kPerCall;

  bool operator<(const KernelKey& other) const {
    return std::tie(isa, operation, rows, n_valid, silu_degree, output, amx_pattern, amx_silu_epilogue, amx_w2_epilogue,
                    amx_tile_state_mode) <
           std::tie(other.isa, other.operation, other.rows, other.n_valid, other.silu_degree, other.output,
                    other.amx_pattern, other.amx_silu_epilogue, other.amx_w2_epilogue, other.amx_tile_state_mode);
  }
};

struct alignas(64) SiluConstants {
  float one = 1.0f;
  float negative_87 = -87.0f;
  float positive_87 = 87.0f;
  float log2e = 1.4426950408889634f;
  float ln2 = 0.6931471805599453f;
  float coefficient_6 = 0.0013888889f;
  float coefficient_5 = 0.00833333f;
  float coefficient_4 = 0.04166666f;
  float coefficient_3 = 0.16666666f;
  float coefficient_2 = 0.5f;
  int32_t exponent_bias = 127;
};

const SiluConstants kSiluConstants;

struct W13Call {
  const uint16_t* a;
  const uint16_t* b;
  uint16_t* c;
  int64_t k_pairs;
  const SiluConstants* constants;
};

struct W2Call {
  const uint16_t* a;
  const uint16_t* b;
  void* output;
  const int64_t* route_ids;
  int64_t k_pairs;
  int64_t route_stride_bytes;
};

struct AmxW13Call {
  const uint16_t* a;
  const uint16_t* b;
  uint16_t* c;
  int64_t k_blocks;
  int64_t a_stride_bytes;
  int64_t b_block_stride_bytes;
  int64_t c_stride_bytes;
  int64_t block_count;
  const SiluConstants* constants;
  int64_t m_panel_count;
};

struct AmxW2Call {
  const uint16_t* a;
  const uint16_t* b;
  void* output;
  const int64_t* route_ids;
  int64_t k_blocks;
  int64_t a_stride_bytes;
  int64_t b_block_stride_bytes;
  int64_t route_stride_bytes;
  int64_t block_count;
  int64_t m_panel_count;
};

using JitFunction = void (*)(const void*);

struct KernelHandle {
  std::shared_ptr<void> owner;
  JitFunction function = nullptr;
  size_t code_bytes = 0;
  std::string error;

  explicit operator bool() const { return function != nullptr; }
};

class W13Generator final : public Xbyak::CodeGenerator {
 public:
  W13Generator(int rows, int degree) : Xbyak::CodeGenerator(16 * 1024, Xbyak::AutoGrow), rows_(rows), degree_(degree) {
    Generate();
    readyRE();
  }

 private:
  static constexpr int kABytesPerPair = 64;
  static constexpr int kBBytesPerPair = 128;

  Xbyak::Zmm Accumulator(int row, int half) const { return Xbyak::Zmm(row * 2 + half); }

  void BroadcastFloat(const Xbyak::Zmm& destination, size_t offset) {
    vbroadcastss(destination, dword[rsi + static_cast<int>(offset)]);
  }

  void EmitSilu(int row) {
    const Xbyak::Zmm gate = Accumulator(row, 0);
    const Xbyak::Zmm up = Accumulator(row, 1);
    const Xbyak::Zmm zero(24);
    const Xbyak::Zmm x(25);
    const Xbyak::Zmm constant(26);
    const Xbyak::Zmm fn(27);
    const Xbyak::Zmm remainder(28);
    const Xbyak::Zmm polynomial(29);
    const Xbyak::Zmm exponent(30);

    vxorps(zero, zero, zero);
    vsubps(x, zero, gate);
    BroadcastFloat(constant, offsetof(SiluConstants, negative_87));
    vmaxps(x, x, constant);
    BroadcastFloat(constant, offsetof(SiluConstants, positive_87));
    vminps(x, x, constant);
    BroadcastFloat(constant, offsetof(SiluConstants, log2e));
    vmulps(fn, x, constant);
    vrndscaleps(fn, fn, 0x08);

    vmovaps(remainder, x);
    BroadcastFloat(constant, offsetof(SiluConstants, ln2));
    vfnmadd231ps(remainder, fn, constant);

    if (degree_ == 4) {
      BroadcastFloat(polynomial, offsetof(SiluConstants, coefficient_4));
    } else if (degree_ == 5) {
      BroadcastFloat(polynomial, offsetof(SiluConstants, coefficient_5));
      BroadcastFloat(constant, offsetof(SiluConstants, coefficient_4));
      vfmadd213ps(polynomial, remainder, constant);
    } else {
      BroadcastFloat(polynomial, offsetof(SiluConstants, coefficient_6));
      BroadcastFloat(constant, offsetof(SiluConstants, coefficient_5));
      vfmadd213ps(polynomial, remainder, constant);
      BroadcastFloat(constant, offsetof(SiluConstants, coefficient_4));
      vfmadd213ps(polynomial, remainder, constant);
    }
    BroadcastFloat(constant, offsetof(SiluConstants, coefficient_3));
    vfmadd213ps(polynomial, remainder, constant);
    BroadcastFloat(constant, offsetof(SiluConstants, coefficient_2));
    vfmadd213ps(polynomial, remainder, constant);
    BroadcastFloat(constant, offsetof(SiluConstants, one));
    vfmadd213ps(polynomial, remainder, constant);
    vfmadd213ps(polynomial, remainder, constant);

    vcvtps2dq(exponent, fn);
    vpbroadcastd(constant, dword[rsi + static_cast<int>(offsetof(SiluConstants, exponent_bias))]);
    vpaddd(exponent, exponent, constant);
    vpslld(exponent, exponent, 23);
    vmulps(polynomial, polynomial, exponent);
    BroadcastFloat(constant, offsetof(SiluConstants, one));
    vaddps(polynomial, polynomial, constant);
    vmulps(gate, gate, up);
    vdivps(gate, gate, polynomial);
  }

  void Generate() {
    if (rows_ < 1 || rows_ > 12 || (degree_ != 4 && degree_ != 5 && degree_ != 6)) {
      throw std::invalid_argument("invalid AVX-512 W13 JIT specialization");
    }

    const int stack_bytes = (rows_ * 32 + 63) & ~63;
    mov(r8, ptr[rdi + static_cast<int>(offsetof(W13Call, a))]);
    mov(r9, ptr[rdi + static_cast<int>(offsetof(W13Call, b))]);
    mov(r10, ptr[rdi + static_cast<int>(offsetof(W13Call, c))]);
    mov(rcx, ptr[rdi + static_cast<int>(offsetof(W13Call, k_pairs))]);
    mov(rsi, ptr[rdi + static_cast<int>(offsetof(W13Call, constants))]);
    sub(rsp, stack_bytes);

    for (int row = 0; row < rows_; ++row) {
      vxorps(Accumulator(row, 0), Accumulator(row, 0), Accumulator(row, 0));
      vxorps(Accumulator(row, 1), Accumulator(row, 1), Accumulator(row, 1));
    }

    Xbyak::Label k_loop;
    Xbyak::Label k_done;
    test(rcx, rcx);
    jz(k_done, T_NEAR);
    align(64);
    L(k_loop);
    vmovdqu16(Xbyak::Zmm(24), ptr[r9]);
    vmovdqu16(Xbyak::Zmm(25), ptr[r9 + 64]);
    prefetcht0(ptr[r9 + 8 * kBBytesPerPair]);
    for (int row = 0; row < rows_; ++row) {
      vpbroadcastd(Xbyak::Zmm(26), dword[r8 + row * 4]);
      vdpbf16ps(Accumulator(row, 0), Xbyak::Zmm(24), Xbyak::Zmm(26));
      vdpbf16ps(Accumulator(row, 1), Xbyak::Zmm(25), Xbyak::Zmm(26));
    }
    add(r8, kABytesPerPair);
    add(r9, kBBytesPerPair);
    dec(rcx);
    jnz(k_loop, T_NEAR);
    L(k_done);

    for (int row = 0; row < rows_; ++row) {
      EmitSilu(row);
      vcvtneps2bf16(Xbyak::Ymm(31), Accumulator(row, 0));
      vmovdqu16(ptr[rsp + row * 32], Xbyak::Ymm(31));
    }

    // Transpose Mx16 row-major BF16 values into the existing VNNI2 packed-A
    // layout consumed by W2. Exact-M kernels intentionally leave physical
    // padding rows untouched because W2 only broadcasts the logical rows.
    for (int feature = 0; feature < 16; ++feature) {
      for (int row = 0; row < rows_; ++row) {
        const int source_offset = row * 32 + feature * 2;
        const int destination_offset = (feature / 2) * 64 + row * 4 + (feature & 1) * 2;
        movzx(eax, word[rsp + source_offset]);
        mov(word[r10 + destination_offset], ax);
      }
    }

    add(rsp, stack_bytes);
    vzeroupper();
    ret();
  }

  int rows_;
  int degree_;
};

class W2Generator final : public Xbyak::CodeGenerator {
 public:
  W2Generator(int rows, int n_valid, bool direct_bf16)
      : Xbyak::CodeGenerator(16 * 1024, Xbyak::AutoGrow),
        rows_(rows),
        n_valid_(n_valid),
        halves_(n_valid > 16 ? 2 : 1),
        base_accumulators_(rows * halves_),
        split_accumulators_(base_accumulators_ <= 12),
        direct_bf16_(direct_bf16) {
    Generate();
    readyRE();
  }

 private:
  static constexpr int kABytesPerPair = 64;
  static constexpr int kBBytesPerPair = 128;

  Xbyak::Zmm Accumulator(int row, int half, int set = 0) const {
    return Xbyak::Zmm(set * base_accumulators_ + row * halves_ + half);
  }

  void EmitKPair(int accumulator_set) {
    vmovdqu16(Xbyak::Zmm(24), ptr[r9]);
    if (halves_ == 2) {
      vmovdqu16(Xbyak::Zmm(25), ptr[r9 + 64]);
    }
    for (int row = 0; row < rows_; ++row) {
      vpbroadcastd(Xbyak::Zmm(26), dword[r8 + row * 4]);
      vdpbf16ps(Accumulator(row, 0, accumulator_set), Xbyak::Zmm(24), Xbyak::Zmm(26));
      if (halves_ == 2) {
        vdpbf16ps(Accumulator(row, 1, accumulator_set), Xbyak::Zmm(25), Xbyak::Zmm(26));
      }
    }
    add(r8, kABytesPerPair);
    add(r9, kBBytesPerPair);
  }

  static uint16_t MaskFor(int valid) {
    if (valid >= 16) {
      return 0xffff;
    }
    return valid <= 0 ? 0 : static_cast<uint16_t>((uint32_t{1} << valid) - 1);
  }

  void EmitFloatStore(const Xbyak::Address& destination, const Xbyak::Zmm& value, int valid,
                      const Xbyak::Opmask& mask) {
    if (valid == 16) {
      vmovups(destination, value);
    } else {
      vmovups(destination | mask, value);
    }
  }

  void EmitBf16Store(const Xbyak::Address& destination, const Xbyak::Zmm& value, int valid, const Xbyak::Opmask& mask) {
    vcvtneps2bf16(Xbyak::Ymm(30), value);
    if (valid == 16) {
      vmovdqu16(destination, Xbyak::Ymm(30));
    } else {
      vmovdqu16(destination | mask, Xbyak::Ymm(30));
    }
  }

  void Generate() {
    if (rows_ < 1 || rows_ > 12 || n_valid_ < 1 || n_valid_ > 32) {
      throw std::invalid_argument("invalid AVX-512 W2 JIT specialization");
    }

    mov(r8, ptr[rdi + static_cast<int>(offsetof(W2Call, a))]);
    mov(r9, ptr[rdi + static_cast<int>(offsetof(W2Call, b))]);
    mov(r10, ptr[rdi + static_cast<int>(offsetof(W2Call, output))]);
    mov(r11, ptr[rdi + static_cast<int>(offsetof(W2Call, route_ids))]);
    mov(rcx, ptr[rdi + static_cast<int>(offsetof(W2Call, k_pairs))]);
    mov(rsi, ptr[rdi + static_cast<int>(offsetof(W2Call, route_stride_bytes))]);

    const int accumulator_sets = split_accumulators_ ? 2 : 1;
    for (int set = 0; set < accumulator_sets; ++set) {
      for (int row = 0; row < rows_; ++row) {
        for (int half = 0; half < halves_; ++half) {
          vxorps(Accumulator(row, half, set), Accumulator(row, half, set), Accumulator(row, half, set));
        }
      }
    }

    if (split_accumulators_) {
      Xbyak::Label k_loop;
      Xbyak::Label k_tail;
      Xbyak::Label k_reduce;
      cmp(rcx, 2);
      jb(k_tail, T_NEAR);
      align(64);
      L(k_loop);
      EmitKPair(0);
      EmitKPair(1);
      sub(rcx, 2);
      cmp(rcx, 2);
      jae(k_loop, T_NEAR);
      L(k_tail);
      test(rcx, rcx);
      jz(k_reduce, T_NEAR);
      EmitKPair(0);
      L(k_reduce);
      for (int row = 0; row < rows_; ++row) {
        for (int half = 0; half < halves_; ++half) {
          vaddps(Accumulator(row, half), Accumulator(row, half), Accumulator(row, half, 1));
        }
      }
    } else {
      Xbyak::Label k_loop;
      Xbyak::Label k_done;
      test(rcx, rcx);
      jz(k_done, T_NEAR);
      align(64);
      L(k_loop);
      EmitKPair(0);
      dec(rcx);
      jnz(k_loop, T_NEAR);
      L(k_done);
    }

    const int valid0 = std::min(n_valid_, 16);
    const int valid1 = std::max(n_valid_ - 16, 0);
    if (valid0 != 16) {
      mov(eax, MaskFor(valid0));
      kmovw(k1, eax);
    }
    if (valid1 > 0 && valid1 != 16) {
      mov(eax, MaskFor(valid1));
      kmovw(k2, eax);
    }

    for (int row = 0; row < rows_; ++row) {
      mov(rax, ptr[r11 + row * 8]);
      imul(rax, rsi);
      lea(rdx, ptr[r10 + rax]);
      if (direct_bf16_) {
        EmitBf16Store(ptr[rdx], Accumulator(row, 0), valid0, k1);
        if (valid1 > 0) {
          EmitBf16Store(ptr[rdx + 32], Accumulator(row, 1), valid1, k2);
        }
      } else {
        EmitFloatStore(ptr[rdx], Accumulator(row, 0), valid0, k1);
        if (valid1 > 0) {
          EmitFloatStore(ptr[rdx + 64], Accumulator(row, 1), valid1, k2);
        }
      }
    }

    vzeroupper();
    ret();
  }

  int rows_;
  int n_valid_;
  int halves_;
  int base_accumulators_;
  bool split_accumulators_;
  bool direct_bf16_;
};

#pragma pack(push, 1)
struct TileConfig {
  uint8_t palette_id = 1;
  uint8_t start_row = 0;
  uint8_t reserved[14] = {};
  uint16_t columns[16] = {};
  uint8_t rows[16] = {};
};
#pragma pack(pop)

static_assert(sizeof(TileConfig) == 64, "AMX tile configuration must occupy one cache line");

void SetTileShape(TileConfig* config, int tile, int rows, int columns) {
  config->rows[tile] = static_cast<uint8_t>(rows);
  config->columns[tile] = static_cast<uint16_t>(columns);
}

TileConfig MakeAmxW13Config(int rows, AmxJitPattern pattern) {
  TileConfig config;
  if (pattern == AmxJitPattern::kM2N2) {
    const int second_rows = rows - 16;
    SetTileShape(&config, 0, 16, 64);
    SetTileShape(&config, 1, 16, 64);
    SetTileShape(&config, 2, second_rows, 64);
    SetTileShape(&config, 3, second_rows, 64);
    SetTileShape(&config, 4, 16, 64);
    SetTileShape(&config, 5, second_rows, 64);
    SetTileShape(&config, 6, 16, 64);
    SetTileShape(&config, 7, 16, 64);
    return config;
  }
  if (pattern == AmxJitPattern::kM1N4) {
    for (int tile = 0; tile < 6; ++tile) {
      SetTileShape(&config, tile, rows, 64);
    }
    SetTileShape(&config, 6, 16, 64);
    SetTileShape(&config, 7, 16, 64);
    return config;
  }
  SetTileShape(&config, 0, rows, 64);
  SetTileShape(&config, 1, rows, 64);
  SetTileShape(&config, 2, rows, 64);
  SetTileShape(&config, 3, rows, 64);
  for (int tile = 4; tile < 8; ++tile) {
    SetTileShape(&config, tile, 16, 64);
  }
  return config;
}

TileConfig MakeAmxW2Config(int rows, int n_valid, AmxJitPattern pattern) {
  TileConfig config;
  if (pattern == AmxJitPattern::kM1N4) {
    for (int tile = 0; tile < 6; ++tile) {
      SetTileShape(&config, tile, rows, 64);
    }
    SetTileShape(&config, 6, 16, 64);
    SetTileShape(&config, 7, 16, 64);
    return config;
  }
  const int valid0 = std::min(n_valid, 16);
  const int valid1 = std::max(n_valid - 16, 0);
  if (pattern == AmxJitPattern::kM2N2) {
    const int second_rows = rows - 16;
    SetTileShape(&config, 0, 16, valid0 * 4);
    if (valid1 > 0) {
      SetTileShape(&config, 1, 16, valid1 * 4);
    }
    SetTileShape(&config, 2, second_rows, valid0 * 4);
    if (valid1 > 0) {
      SetTileShape(&config, 3, second_rows, valid1 * 4);
    }
    SetTileShape(&config, 4, 16, 64);
    SetTileShape(&config, 5, second_rows, 64);
    SetTileShape(&config, 6, 16, valid0 * 4);
    if (valid1 > 0) {
      SetTileShape(&config, 7, 16, valid1 * 4);
    }
    return config;
  }
  SetTileShape(&config, 0, rows, valid0 * 4);
  if (valid1 > 0) {
    SetTileShape(&config, 1, rows, valid1 * 4);
  }
  SetTileShape(&config, 2, rows, 64);
  SetTileShape(&config, 3, rows, 64);
  SetTileShape(&config, 4, 16, valid0 * 4);
  SetTileShape(&config, 6, 16, valid0 * 4);
  if (valid1 > 0) {
    SetTileShape(&config, 5, 16, valid1 * 4);
    SetTileShape(&config, 7, 16, valid1 * 4);
  }
  return config;
}

class AmxW13Generator final : public Xbyak::CodeGenerator {
 public:
  AmxW13Generator(int rows, int degree, AmxJitPattern pattern, AmxSiluEpilogue silu_epilogue,
                  AmxTileStateMode tile_state_mode)
      : Xbyak::CodeGenerator(16 * 1024, Xbyak::AutoGrow),
        rows_(rows),
        degree_(degree),
        pattern_(pattern),
        silu_epilogue_(silu_epilogue),
        tile_state_mode_(tile_state_mode),
        config_(MakeAmxW13Config(rows, pattern)) {
    Generate();
    readyRE();
  }

 private:
  static constexpr int kTileScratchBytes = 2048;
  static constexpr int kMacroMStateBytes = 16;
  static constexpr int kBBytesPerKBlock = 2048;

  bool UsesMacroM() const { return tile_state_mode_ == AmxTileStateMode::kMacroM; }

  int FrameBytes() const { return kTileScratchBytes + (UsesMacroM() ? kMacroMStateBytes : 0); }

  int CPanelStateOffset() const { return kTileScratchBytes; }

  int MPanelCountStateOffset() const { return kTileScratchBytes + 8; }

  void BroadcastFloat(const Xbyak::Zmm& destination, size_t offset) {
    vbroadcastss(destination, dword[rbx + static_cast<int>(offset)]);
  }

  struct SiluRowRegisters {
    Xbyak::Zmm gate;
    Xbyak::Zmm up;
    Xbyak::Zmm x;
    Xbyak::Zmm fn;
    Xbyak::Zmm remainder;
    Xbyak::Zmm polynomial;
    Xbyak::Zmm exponent;
  };

  void EmitBaselineSilu() {
    const Xbyak::Zmm gate(0);
    const Xbyak::Zmm up(1);
    const Xbyak::Zmm zero(24);
    const Xbyak::Zmm x(25);
    const Xbyak::Zmm constant(26);
    const Xbyak::Zmm fn(27);
    const Xbyak::Zmm remainder(28);
    const Xbyak::Zmm polynomial(29);
    const Xbyak::Zmm exponent(30);

    vxorps(zero, zero, zero);
    vsubps(x, zero, gate);
    BroadcastFloat(constant, offsetof(SiluConstants, negative_87));
    vmaxps(x, x, constant);
    BroadcastFloat(constant, offsetof(SiluConstants, positive_87));
    vminps(x, x, constant);
    BroadcastFloat(constant, offsetof(SiluConstants, log2e));
    vmulps(fn, x, constant);
    vrndscaleps(fn, fn, 0x08);

    vmovaps(remainder, x);
    BroadcastFloat(constant, offsetof(SiluConstants, ln2));
    vfnmadd231ps(remainder, fn, constant);

    if (degree_ == 4) {
      BroadcastFloat(polynomial, offsetof(SiluConstants, coefficient_4));
    } else if (degree_ == 5) {
      BroadcastFloat(polynomial, offsetof(SiluConstants, coefficient_5));
      BroadcastFloat(constant, offsetof(SiluConstants, coefficient_4));
      vfmadd213ps(polynomial, remainder, constant);
    } else {
      BroadcastFloat(polynomial, offsetof(SiluConstants, coefficient_6));
      BroadcastFloat(constant, offsetof(SiluConstants, coefficient_5));
      vfmadd213ps(polynomial, remainder, constant);
      BroadcastFloat(constant, offsetof(SiluConstants, coefficient_4));
      vfmadd213ps(polynomial, remainder, constant);
    }
    BroadcastFloat(constant, offsetof(SiluConstants, coefficient_3));
    vfmadd213ps(polynomial, remainder, constant);
    BroadcastFloat(constant, offsetof(SiluConstants, coefficient_2));
    vfmadd213ps(polynomial, remainder, constant);
    BroadcastFloat(constant, offsetof(SiluConstants, one));
    vfmadd213ps(polynomial, remainder, constant);
    vfmadd213ps(polynomial, remainder, constant);

    vcvtps2dq(exponent, fn);
    vpbroadcastd(constant, dword[rbx + static_cast<int>(offsetof(SiluConstants, exponent_bias))]);
    vpaddd(exponent, exponent, constant);
    vpslld(exponent, exponent, 23);
    vmulps(polynomial, polynomial, exponent);
    BroadcastFloat(constant, offsetof(SiluConstants, one));
    vaddps(polynomial, polynomial, constant);
    vmulps(gate, gate, up);
    vdivps(gate, gate, polynomial);
  }

  void EmitResidentSiluConstants() {
    BroadcastFloat(Xbyak::Zmm(2), offsetof(SiluConstants, one));
    BroadcastFloat(Xbyak::Zmm(3), offsetof(SiluConstants, negative_87));
    BroadcastFloat(Xbyak::Zmm(4), offsetof(SiluConstants, positive_87));
    BroadcastFloat(Xbyak::Zmm(5), offsetof(SiluConstants, log2e));
    BroadcastFloat(Xbyak::Zmm(6), offsetof(SiluConstants, ln2));
    BroadcastFloat(Xbyak::Zmm(7), offsetof(SiluConstants, coefficient_6));
    BroadcastFloat(Xbyak::Zmm(8), offsetof(SiluConstants, coefficient_5));
    BroadcastFloat(Xbyak::Zmm(9), offsetof(SiluConstants, coefficient_4));
    BroadcastFloat(Xbyak::Zmm(10), offsetof(SiluConstants, coefficient_3));
    BroadcastFloat(Xbyak::Zmm(11), offsetof(SiluConstants, coefficient_2));
    vpbroadcastd(Xbyak::Zmm(12), dword[rbx + static_cast<int>(offsetof(SiluConstants, exponent_bias))]);
    vxorps(Xbyak::Zmm(13), Xbyak::Zmm(13), Xbyak::Zmm(13));
  }

  void EmitResidentSilu(const SiluRowRegisters* rows, int row_count, bool use_rcp14) {
    const Xbyak::Zmm one(2);
    const Xbyak::Zmm negative_87(3);
    const Xbyak::Zmm positive_87(4);
    const Xbyak::Zmm log2e(5);
    const Xbyak::Zmm ln2(6);
    const Xbyak::Zmm coefficient_6(7);
    const Xbyak::Zmm coefficient_5(8);
    const Xbyak::Zmm coefficient_4(9);
    const Xbyak::Zmm coefficient_3(10);
    const Xbyak::Zmm coefficient_2(11);
    const Xbyak::Zmm exponent_bias(12);
    const Xbyak::Zmm zero(13);

    // Emit each arithmetic stage for every independent row before advancing
    // to the next stage. This preserves each row's operation order while
    // allowing the out-of-order core to overlap two dependency chains.
    for (int row = 0; row < row_count; ++row) {
      vsubps(rows[row].x, zero, rows[row].gate);
      vmaxps(rows[row].x, rows[row].x, negative_87);
      vminps(rows[row].x, rows[row].x, positive_87);
      vmulps(rows[row].fn, rows[row].x, log2e);
      vrndscaleps(rows[row].fn, rows[row].fn, 0x08);
      vmovaps(rows[row].remainder, rows[row].x);
      vfnmadd231ps(rows[row].remainder, rows[row].fn, ln2);
    }

    for (int row = 0; row < row_count; ++row) {
      if (degree_ == 4) {
        vmovaps(rows[row].polynomial, coefficient_4);
      } else if (degree_ == 5) {
        vmovaps(rows[row].polynomial, coefficient_5);
        vfmadd213ps(rows[row].polynomial, rows[row].remainder, coefficient_4);
      } else {
        vmovaps(rows[row].polynomial, coefficient_6);
        vfmadd213ps(rows[row].polynomial, rows[row].remainder, coefficient_5);
        vfmadd213ps(rows[row].polynomial, rows[row].remainder, coefficient_4);
      }
      vfmadd213ps(rows[row].polynomial, rows[row].remainder, coefficient_3);
      vfmadd213ps(rows[row].polynomial, rows[row].remainder, coefficient_2);
      vfmadd213ps(rows[row].polynomial, rows[row].remainder, one);
      vfmadd213ps(rows[row].polynomial, rows[row].remainder, one);
    }

    for (int row = 0; row < row_count; ++row) {
      vcvtps2dq(rows[row].exponent, rows[row].fn);
      vpaddd(rows[row].exponent, rows[row].exponent, exponent_bias);
      vpslld(rows[row].exponent, rows[row].exponent, 23);
      vmulps(rows[row].polynomial, rows[row].polynomial, rows[row].exponent);
      vaddps(rows[row].polynomial, rows[row].polynomial, one);
      vmulps(rows[row].gate, rows[row].gate, rows[row].up);
      if (use_rcp14) {
        vrcp14ps(rows[row].x, rows[row].polynomial);
        vmulps(rows[row].gate, rows[row].gate, rows[row].x);
      } else {
        vdivps(rows[row].gate, rows[row].gate, rows[row].polynomial);
      }
    }
  }

  void EmitKBlock(const Xbyak::Tmm& a_tile, const Xbyak::Tmm& gate_tile, const Xbyak::Tmm& up_tile, int byte_offset) {
    tileloadd(a_tile, ptr[rsi + r12 + byte_offset]);
    tileloadd(gate_tile, ptr[rdi + rcx + byte_offset * 32]);
    tileloadd(up_tile, ptr[rdi + rcx + byte_offset * 32 + 64]);
    tdpbf16ps(tmm0, a_tile, gate_tile);
    tdpbf16ps(tmm1, a_tile, up_tile);
  }

  void EmitM1N4KBlock(const Xbyak::Tmm& a_tile, int byte_offset) {
    tileloadd(a_tile, ptr[rsi + r12 + byte_offset]);
    tileloadd(tmm6, ptr[rdi + rcx + byte_offset * 32]);
    tileloadd(tmm7, ptr[rdi + rcx + byte_offset * 32 + 64]);
    tdpbf16ps(tmm0, a_tile, tmm6);
    tdpbf16ps(tmm1, a_tile, tmm7);
    tileloadd(tmm6, ptr[rdx + rcx + byte_offset * 32]);
    tileloadd(tmm7, ptr[rdx + rcx + byte_offset * 32 + 64]);
    tdpbf16ps(tmm2, a_tile, tmm6);
    tdpbf16ps(tmm3, a_tile, tmm7);
  }

  void EmitSiluStorePanel(const Xbyak::Tmm& gate_tile, const Xbyak::Tmm& up_tile, int rows) {
    mov(edx, 64);
    tilestored(ptr[rsp + rdx], gate_tile);
    tilestored(ptr[rsp + rdx + 1024], up_tile);
    if (silu_epilogue_ == AmxSiluEpilogue::kBaseline) {
      for (int row = 0; row < rows; ++row) {
        vmovups(Xbyak::Zmm(0), ptr[rsp + row * 64]);
        vmovups(Xbyak::Zmm(1), ptr[rsp + 1024 + row * 64]);
        EmitBaselineSilu();
        vcvtneps2bf16(Xbyak::Ymm(31), Xbyak::Zmm(0));
        vmovdqu16(ptr[rsi], Xbyak::Ymm(31));
        add(rsi, r14);
      }
      return;
    }

    const SiluRowRegisters silu_rows[2] = {
        {Xbyak::Zmm(0), Xbyak::Zmm(1), Xbyak::Zmm(14), Xbyak::Zmm(15), Xbyak::Zmm(16), Xbyak::Zmm(17), Xbyak::Zmm(18)},
        {Xbyak::Zmm(19), Xbyak::Zmm(20), Xbyak::Zmm(21), Xbyak::Zmm(22), Xbyak::Zmm(23), Xbyak::Zmm(24),
         Xbyak::Zmm(25)},
    };
    const bool pipelined = silu_epilogue_ != AmxSiluEpilogue::kResident;
    const bool use_rcp14 = silu_epilogue_ == AmxSiluEpilogue::kRcp14;
    int row = 0;
    if (pipelined) {
      for (; row + 1 < rows; row += 2) {
        vmovups(silu_rows[0].gate, ptr[rsp + row * 64]);
        vmovups(silu_rows[0].up, ptr[rsp + 1024 + row * 64]);
        vmovups(silu_rows[1].gate, ptr[rsp + (row + 1) * 64]);
        vmovups(silu_rows[1].up, ptr[rsp + 1024 + (row + 1) * 64]);
        EmitResidentSilu(silu_rows, 2, use_rcp14);
        vcvtneps2bf16(Xbyak::Ymm(26), silu_rows[0].gate);
        vcvtneps2bf16(Xbyak::Ymm(27), silu_rows[1].gate);
        vmovdqu16(ptr[rsi], Xbyak::Ymm(26));
        vmovdqu16(ptr[rsi + r14], Xbyak::Ymm(27));
        add(rsi, r14);
        add(rsi, r14);
      }
    }
    for (; row < rows; ++row) {
      vmovups(silu_rows[0].gate, ptr[rsp + row * 64]);
      vmovups(silu_rows[0].up, ptr[rsp + 1024 + row * 64]);
      EmitResidentSilu(silu_rows, 1, use_rcp14);
      vcvtneps2bf16(Xbyak::Ymm(26), silu_rows[0].gate);
      vmovdqu16(ptr[rsi], Xbyak::Ymm(26));
      add(rsi, r14);
    }
  }

  void EmitAmxPrologue() {
    // KernelHandle retains this generator, so the embedded absolute address
    // remains valid for the entire lifetime of the published JIT function.
    mov(rax, reinterpret_cast<uint64_t>(&config_));
    ldtilecfg(ptr[rax]);
    push(rbx);
    push(r12);
    push(r13);
    push(r14);
    push(r15);
    if (UsesMacroM()) {
      push(rbp);
      mov(rbp, rdi);
    }
    sub(rsp, FrameBytes());

    mov(r8, ptr[rdi + static_cast<int>(offsetof(AmxW13Call, a))]);
    mov(r9, ptr[rdi + static_cast<int>(offsetof(AmxW13Call, b))]);
    mov(r10, ptr[rdi + static_cast<int>(offsetof(AmxW13Call, c))]);
    mov(r11, ptr[rdi + static_cast<int>(offsetof(AmxW13Call, k_blocks))]);
    mov(r12, ptr[rdi + static_cast<int>(offsetof(AmxW13Call, a_stride_bytes))]);
    mov(r13, ptr[rdi + static_cast<int>(offsetof(AmxW13Call, b_block_stride_bytes))]);
    mov(r14, ptr[rdi + static_cast<int>(offsetof(AmxW13Call, c_stride_bytes))]);
    mov(r15, ptr[rdi + static_cast<int>(offsetof(AmxW13Call, block_count))]);
    mov(rbx, ptr[rdi + static_cast<int>(offsetof(AmxW13Call, constants))]);
    if (UsesMacroM()) {
      mov(ptr[rsp + CPanelStateOffset()], r10);
      mov(rax, ptr[rbp + static_cast<int>(offsetof(AmxW13Call, m_panel_count))]);
      mov(ptr[rsp + MPanelCountStateOffset()], rax);
    }
    if (silu_epilogue_ != AmxSiluEpilogue::kBaseline) {
      EmitResidentSiluConstants();
    }
  }

  void EmitMPanelBegin(Xbyak::Label& panel_loop, Xbyak::Label& all_done) {
    if (!UsesMacroM()) {
      test(r15, r15);
      jz(all_done, T_NEAR);
      return;
    }
    cmp(qword[rsp + MPanelCountStateOffset()], 0);
    jz(all_done, T_NEAR);
    L(panel_loop);
    mov(r9, ptr[rbp + static_cast<int>(offsetof(AmxW13Call, b))]);
    mov(r10, ptr[rsp + CPanelStateOffset()]);
    mov(r15, ptr[rbp + static_cast<int>(offsetof(AmxW13Call, block_count))]);
    test(r15, r15);
    jz(all_done, T_NEAR);
  }

  void EmitMPanelAdvance(Xbyak::Label& panel_loop) {
    if (!UsesMacroM()) {
      return;
    }
    imul(rax, r12, rows_);
    add(r8, rax);
    mov(r10, ptr[rsp + CPanelStateOffset()]);
    imul(rax, r14, rows_);
    add(r10, rax);
    mov(ptr[rsp + CPanelStateOffset()], r10);
    sub(qword[rsp + MPanelCountStateOffset()], 1);
    jnz(panel_loop, T_NEAR);
  }

  void EmitAmxReturn() {
    tilerelease();
    add(rsp, FrameBytes());
    if (UsesMacroM()) {
      pop(rbp);
    }
    pop(r15);
    pop(r14);
    pop(r13);
    pop(r12);
    pop(rbx);
    vzeroupper();
    ret();
  }

  void GenerateM2N2() {
    if (rows_ < 17 || rows_ > 32) {
      throw std::invalid_argument("invalid AMX W13 2M-by-2N specialization");
    }
    const int second_rows = rows_ - 16;
    EmitAmxPrologue();

    Xbyak::Label panel_loop;
    Xbyak::Label block_loop;
    Xbyak::Label k_loop;
    Xbyak::Label k_done;
    Xbyak::Label all_done;
    EmitMPanelBegin(panel_loop, all_done);
    L(block_loop);
    tilezero(tmm0);
    tilezero(tmm1);
    tilezero(tmm2);
    tilezero(tmm3);
    mov(rsi, r8);
    mov(rdx, r12);
    shl(rdx, 4);
    add(rdx, r8);
    mov(rdi, r9);
    mov(rax, r11);
    mov(ecx, 128);
    test(rax, rax);
    jz(k_done, T_NEAR);
    align(64);
    L(k_loop);
    tileloadd(tmm4, ptr[rsi + r12]);
    tileloadd(tmm5, ptr[rdx + r12]);
    tileloadd(tmm6, ptr[rdi + rcx]);
    tileloadd(tmm7, ptr[rdi + rcx + 64]);
    tdpbf16ps(tmm0, tmm4, tmm6);
    tdpbf16ps(tmm1, tmm4, tmm7);
    tdpbf16ps(tmm2, tmm5, tmm6);
    tdpbf16ps(tmm3, tmm5, tmm7);
    add(rsi, 64);
    add(rdx, 64);
    add(rdi, kBBytesPerKBlock);
    dec(rax);
    jnz(k_loop, T_NEAR);
    L(k_done);

    mov(rsi, r10);
    EmitSiluStorePanel(tmm0, tmm1, 16);
    EmitSiluStorePanel(tmm2, tmm3, second_rows);
    add(r9, r13);
    add(r10, 32);
    dec(r15);
    jnz(block_loop, T_NEAR);
    EmitMPanelAdvance(panel_loop);

    L(all_done);
    EmitAmxReturn();
  }

  void GenerateM1N4() {
    if (rows_ < 1 || rows_ > 16) {
      throw std::invalid_argument("invalid AMX W13 1M-by-4N specialization");
    }
    EmitAmxPrologue();

    Xbyak::Label panel_loop;
    Xbyak::Label block_loop;
    Xbyak::Label k_loop;
    Xbyak::Label k_tail;
    Xbyak::Label k_done;
    Xbyak::Label all_done;
    EmitMPanelBegin(panel_loop, all_done);
    L(block_loop);
    tilezero(tmm0);
    tilezero(tmm1);
    tilezero(tmm2);
    tilezero(tmm3);
    mov(rsi, r8);
    mov(rdi, r9);
    lea(rdx, ptr[r9 + r13]);
    mov(rax, r11);
    mov(ecx, 128);
    cmp(rax, 2);
    jb(k_tail, T_NEAR);
    align(64);
    L(k_loop);
    EmitM1N4KBlock(tmm4, 0);
    EmitM1N4KBlock(tmm5, 64);
    add(rsi, 128);
    add(rdi, 2 * kBBytesPerKBlock);
    add(rdx, 2 * kBBytesPerKBlock);
    sub(rax, 2);
    cmp(rax, 2);
    jae(k_loop, T_NEAR);
    L(k_tail);
    test(rax, rax);
    jz(k_done, T_NEAR);
    EmitM1N4KBlock(tmm4, 0);
    L(k_done);

    mov(rsi, r10);
    EmitSiluStorePanel(tmm0, tmm1, rows_);
    lea(rsi, ptr[r10 + 32]);
    EmitSiluStorePanel(tmm2, tmm3, rows_);
    lea(r9, ptr[r9 + r13 * 2]);
    add(r10, 64);
    dec(r15);
    jnz(block_loop, T_NEAR);
    EmitMPanelAdvance(panel_loop);

    L(all_done);
    EmitAmxReturn();
  }

  void Generate() {
    if (degree_ != 4 && degree_ != 5 && degree_ != 6) {
      throw std::invalid_argument("invalid AMX W13 SiLU polynomial degree");
    }
    if (pattern_ == AmxJitPattern::kM2N2) {
      GenerateM2N2();
      return;
    }
    if (pattern_ == AmxJitPattern::kM1N4) {
      GenerateM1N4();
      return;
    }
    GenerateM1N2();
  }

  void GenerateM1N2() {
    if (rows_ < 1 || rows_ > 16) {
      throw std::invalid_argument("invalid AMX W13 JIT specialization");
    }
    EmitAmxPrologue();

    Xbyak::Label panel_loop;
    Xbyak::Label block_loop;
    Xbyak::Label k_loop;
    Xbyak::Label k_tail;
    Xbyak::Label k_done;
    Xbyak::Label all_done;
    EmitMPanelBegin(panel_loop, all_done);
    L(block_loop);
    tilezero(tmm0);
    tilezero(tmm1);
    mov(rsi, r8);
    mov(rdi, r9);
    mov(rax, r11);
    mov(ecx, 128);
    cmp(rax, 2);
    jb(k_tail, T_NEAR);
    align(64);
    L(k_loop);
    EmitKBlock(tmm2, tmm4, tmm5, 0);
    EmitKBlock(tmm3, tmm6, tmm7, 64);
    add(rsi, 128);
    add(rdi, 2 * kBBytesPerKBlock);
    sub(rax, 2);
    cmp(rax, 2);
    jae(k_loop, T_NEAR);
    L(k_tail);
    test(rax, rax);
    jz(k_done, T_NEAR);
    EmitKBlock(tmm2, tmm4, tmm5, 0);
    L(k_done);

    mov(rsi, r10);
    EmitSiluStorePanel(tmm0, tmm1, rows_);
    add(r9, r13);
    add(r10, 32);
    dec(r15);
    jnz(block_loop, T_NEAR);
    EmitMPanelAdvance(panel_loop);

    L(all_done);
    EmitAmxReturn();
  }

  int rows_;
  int degree_;
  AmxJitPattern pattern_;
  AmxSiluEpilogue silu_epilogue_;
  AmxTileStateMode tile_state_mode_;
  alignas(64) TileConfig config_;
};

class AmxW2Generator final : public Xbyak::CodeGenerator {
 public:
  AmxW2Generator(int rows, int n_valid, bool direct_bf16, AmxJitPattern pattern, AmxW2Epilogue w2_epilogue,
                 AmxTileStateMode tile_state_mode)
      : Xbyak::CodeGenerator(16 * 1024, Xbyak::AutoGrow),
        rows_(rows),
        n_valid_(n_valid),
        halves_(n_valid > 16 ? 2 : 1),
        direct_bf16_(direct_bf16),
        pattern_(pattern),
        w2_epilogue_(w2_epilogue),
        tile_state_mode_(tile_state_mode),
        config_(MakeAmxW2Config(rows, n_valid, pattern)) {
    Generate();
    readyRE();
  }

 private:
  static constexpr int kOutputPanelScratchBytes = 2048;
  static constexpr int kM1N4ScratchBytes = 4096;
  static constexpr int kMacroMStateBytes = 24;
  static constexpr int kBBytesPerKBlock = 2048;

  bool UsesMacroM() const { return tile_state_mode_ == AmxTileStateMode::kMacroM; }

  int ScratchBytes() const {
    if (w2_epilogue_ == AmxW2Epilogue::kTileStore) {
      return 0;
    }
    return w2_epilogue_ == AmxW2Epilogue::kCombined && pattern_ == AmxJitPattern::kM1N4 ? kM1N4ScratchBytes
                                                                                        : kOutputPanelScratchBytes;
  }

  int FrameBytes() const { return ScratchBytes() + (UsesMacroM() ? kMacroMStateBytes : 0); }

  int OutputPanelStateOffset() const { return ScratchBytes(); }

  int RouteIdsStateOffset() const { return ScratchBytes() + 8; }

  int MPanelCountStateOffset() const { return ScratchBytes() + 16; }

  static uint16_t MaskFor(int valid) {
    if (valid >= 16) {
      return 0xffff;
    }
    return valid <= 0 ? 0 : static_cast<uint16_t>((uint32_t{1} << valid) - 1);
  }

  void EmitKBlock(const Xbyak::Tmm& a_tile, const Xbyak::Tmm& b0_tile, const Xbyak::Tmm& b1_tile, int byte_offset) {
    tileloadd(a_tile, ptr[rsi + r13 + byte_offset]);
    tileloadd(b0_tile, ptr[rdi + rcx + byte_offset * 32]);
    tdpbf16ps(tmm0, a_tile, b0_tile);
    if (halves_ == 2) {
      tileloadd(b1_tile, ptr[rdi + rcx + byte_offset * 32 + 64]);
      tdpbf16ps(tmm1, a_tile, b1_tile);
    }
  }

  void EmitM1N4KBlock(const Xbyak::Tmm& a_tile, int byte_offset) {
    tileloadd(a_tile, ptr[rsi + r13 + byte_offset]);
    tileloadd(tmm6, ptr[rdi + rcx + byte_offset * 32]);
    tileloadd(tmm7, ptr[rdi + rcx + byte_offset * 32 + 64]);
    tdpbf16ps(tmm0, a_tile, tmm6);
    tdpbf16ps(tmm1, a_tile, tmm7);
    tileloadd(tmm6, ptr[rdx + rcx + byte_offset * 32]);
    tileloadd(tmm7, ptr[rdx + rcx + byte_offset * 32 + 64]);
    tdpbf16ps(tmm2, a_tile, tmm6);
    tdpbf16ps(tmm3, a_tile, tmm7);
  }

  void EmitFloatStore(const Xbyak::Address& destination, const Xbyak::Zmm& value, int valid,
                      const Xbyak::Opmask& mask) {
    if (valid == 16) {
      vmovups(destination, value);
    } else {
      vmovups(destination | mask, value);
    }
  }

  void EmitBf16Store(const Xbyak::Address& destination, const Xbyak::Zmm& value, int valid, const Xbyak::Opmask& mask) {
    vcvtneps2bf16(Xbyak::Ymm(30), value);
    if (valid == 16) {
      vmovdqu16(destination, Xbyak::Ymm(30));
    } else {
      vmovdqu16(destination | mask, Xbyak::Ymm(30));
    }
  }

  void EmitOutputPanel(const Xbyak::Tmm& c0, const Xbyak::Tmm& c1, int rows, int route_row_offset, int column_offset,
                       int valid0, int valid1) {
    if (w2_epilogue_ == AmxW2Epilogue::kTileStore) {
      mov(rax, r10);
      if (route_row_offset != 0) {
        // The only current secondary panel begins 16 rows later.
        if (route_row_offset != 16) {
          throw std::invalid_argument("unsupported AMX W2 contiguous row offset");
        }
        mov(rdx, r15);
        shl(rdx, 4);
        add(rax, rdx);
      }
      mov(rdx, r15);
      tilestored(ptr[rax + rdx + column_offset * 4], c0);
      if (valid1 > 0) {
        tilestored(ptr[rax + rdx + (column_offset + 16) * 4], c1);
      }
      return;
    }

    mov(edx, 64);
    tilestored(ptr[rsp + rdx], c0);
    if (valid1 > 0) {
      tilestored(ptr[rsp + rdx + 1024], c1);
    }
    const int column_byte_offset = column_offset * (direct_bf16_ ? 2 : 4);
    for (int row = 0; row < rows; ++row) {
      mov(rax, ptr[r11 + (route_row_offset + row) * 8]);
      imul(rax, r15);
      lea(rdx, ptr[r10 + rax + column_byte_offset]);
      vmovups(Xbyak::Zmm(0), ptr[rsp + row * 64]);
      if (direct_bf16_) {
        EmitBf16Store(ptr[rdx], Xbyak::Zmm(0), valid0, k1);
      } else {
        EmitFloatStore(ptr[rdx], Xbyak::Zmm(0), valid0, k1);
      }
      if (valid1 > 0) {
        vmovups(Xbyak::Zmm(1), ptr[rsp + 1024 + row * 64]);
        if (direct_bf16_) {
          EmitBf16Store(ptr[rdx + 32], Xbyak::Zmm(1), valid1, k2);
        } else {
          EmitFloatStore(ptr[rdx + 64], Xbyak::Zmm(1), valid1, k2);
        }
      }
    }
  }

  void EmitM1N4Output() {
    if (w2_epilogue_ != AmxW2Epilogue::kCombined) {
      EmitOutputPanel(tmm0, tmm1, rows_, 0, 0, 16, 16);
      EmitOutputPanel(tmm2, tmm3, rows_, 0, 32, 16, 16);
      return;
    }

    mov(edx, 64);
    tilestored(ptr[rsp + rdx], tmm0);
    tilestored(ptr[rsp + rdx + 1024], tmm1);
    tilestored(ptr[rsp + rdx + 2048], tmm2);
    tilestored(ptr[rsp + rdx + 3072], tmm3);
    for (int row = 0; row < rows_; ++row) {
      mov(rax, ptr[r11 + row * 8]);
      imul(rax, r15);
      lea(rdx, ptr[r10 + rax]);
      vmovups(Xbyak::Zmm(0), ptr[rsp + row * 64]);
      vmovups(Xbyak::Zmm(1), ptr[rsp + 1024 + row * 64]);
      vmovups(Xbyak::Zmm(2), ptr[rsp + 2048 + row * 64]);
      vmovups(Xbyak::Zmm(3), ptr[rsp + 3072 + row * 64]);
      if (direct_bf16_) {
        EmitBf16Store(ptr[rdx], Xbyak::Zmm(0), 16, k1);
        EmitBf16Store(ptr[rdx + 32], Xbyak::Zmm(1), 16, k1);
        EmitBf16Store(ptr[rdx + 64], Xbyak::Zmm(2), 16, k1);
        EmitBf16Store(ptr[rdx + 96], Xbyak::Zmm(3), 16, k1);
      } else {
        EmitFloatStore(ptr[rdx], Xbyak::Zmm(0), 16, k1);
        EmitFloatStore(ptr[rdx + 64], Xbyak::Zmm(1), 16, k1);
        EmitFloatStore(ptr[rdx + 128], Xbyak::Zmm(2), 16, k1);
        EmitFloatStore(ptr[rdx + 192], Xbyak::Zmm(3), 16, k1);
      }
    }
  }

  void EmitAmxPrologue() {
    // KernelHandle retains this generator, so the embedded absolute address
    // remains valid for the entire lifetime of the published JIT function.
    mov(rax, reinterpret_cast<uint64_t>(&config_));
    ldtilecfg(ptr[rax]);
    push(rbx);
    push(r12);
    push(r13);
    push(r14);
    push(r15);
    if (UsesMacroM()) {
      push(rbp);
      mov(rbp, rdi);
    }
    if (FrameBytes() != 0) {
      sub(rsp, FrameBytes());
    }

    mov(r8, ptr[rdi + static_cast<int>(offsetof(AmxW2Call, a))]);
    mov(r9, ptr[rdi + static_cast<int>(offsetof(AmxW2Call, b))]);
    mov(r10, ptr[rdi + static_cast<int>(offsetof(AmxW2Call, output))]);
    mov(r11, ptr[rdi + static_cast<int>(offsetof(AmxW2Call, route_ids))]);
    mov(r12, ptr[rdi + static_cast<int>(offsetof(AmxW2Call, k_blocks))]);
    mov(r13, ptr[rdi + static_cast<int>(offsetof(AmxW2Call, a_stride_bytes))]);
    mov(r14, ptr[rdi + static_cast<int>(offsetof(AmxW2Call, b_block_stride_bytes))]);
    mov(r15, ptr[rdi + static_cast<int>(offsetof(AmxW2Call, route_stride_bytes))]);
    mov(rbx, ptr[rdi + static_cast<int>(offsetof(AmxW2Call, block_count))]);
    if (UsesMacroM()) {
      mov(ptr[rsp + OutputPanelStateOffset()], r10);
      mov(ptr[rsp + RouteIdsStateOffset()], r11);
      mov(rax, ptr[rbp + static_cast<int>(offsetof(AmxW2Call, m_panel_count))]);
      mov(ptr[rsp + MPanelCountStateOffset()], rax);
    }
  }

  void EmitMPanelBegin(Xbyak::Label& panel_loop, Xbyak::Label& all_done) {
    if (!UsesMacroM()) {
      test(rbx, rbx);
      jz(all_done, T_NEAR);
      return;
    }
    cmp(qword[rsp + MPanelCountStateOffset()], 0);
    jz(all_done, T_NEAR);
    L(panel_loop);
    mov(r9, ptr[rbp + static_cast<int>(offsetof(AmxW2Call, b))]);
    mov(r10, ptr[rsp + OutputPanelStateOffset()]);
    mov(r11, ptr[rsp + RouteIdsStateOffset()]);
    mov(rbx, ptr[rbp + static_cast<int>(offsetof(AmxW2Call, block_count))]);
    test(rbx, rbx);
    jz(all_done, T_NEAR);
  }

  void EmitMPanelAdvance(Xbyak::Label& panel_loop) {
    if (!UsesMacroM()) {
      return;
    }
    imul(rax, r13, rows_);
    add(r8, rax);
    if (w2_epilogue_ == AmxW2Epilogue::kTileStore) {
      mov(r10, ptr[rsp + OutputPanelStateOffset()]);
      imul(rax, r15, rows_);
      add(r10, rax);
      mov(ptr[rsp + OutputPanelStateOffset()], r10);
    } else {
      mov(r11, ptr[rsp + RouteIdsStateOffset()]);
      add(r11, rows_ * 8);
      mov(ptr[rsp + RouteIdsStateOffset()], r11);
    }
    sub(qword[rsp + MPanelCountStateOffset()], 1);
    jnz(panel_loop, T_NEAR);
  }

  void EmitAmxReturn() {
    tilerelease();
    if (FrameBytes() != 0) {
      add(rsp, FrameBytes());
    }
    if (UsesMacroM()) {
      pop(rbp);
    }
    pop(r15);
    pop(r14);
    pop(r13);
    pop(r12);
    pop(rbx);
    vzeroupper();
    ret();
  }

  void EmitMasks(int valid0, int valid1) {
    if (valid0 != 16) {
      mov(eax, MaskFor(valid0));
      kmovw(k1, eax);
    }
    if (valid1 > 0 && valid1 != 16) {
      mov(eax, MaskFor(valid1));
      kmovw(k2, eax);
    }
  }

  void GenerateM2N2() {
    if (rows_ < 17 || rows_ > 32 || n_valid_ < 1 || n_valid_ > 32) {
      throw std::invalid_argument("invalid AMX W2 2M-by-2N specialization");
    }
    const int second_rows = rows_ - 16;
    const int valid0 = std::min(n_valid_, 16);
    const int valid1 = std::max(n_valid_ - 16, 0);
    EmitAmxPrologue();
    EmitMasks(valid0, valid1);

    Xbyak::Label panel_loop;
    Xbyak::Label block_loop;
    Xbyak::Label k_loop;
    Xbyak::Label k_done;
    Xbyak::Label all_done;
    EmitMPanelBegin(panel_loop, all_done);
    L(block_loop);
    tilezero(tmm0);
    tilezero(tmm2);
    if (halves_ == 2) {
      tilezero(tmm1);
      tilezero(tmm3);
    }
    mov(rsi, r8);
    mov(rdx, r13);
    shl(rdx, 4);
    add(rdx, r8);
    mov(rdi, r9);
    mov(rax, r12);
    mov(ecx, 128);
    test(rax, rax);
    jz(k_done, T_NEAR);
    align(64);
    L(k_loop);
    tileloadd(tmm4, ptr[rsi + r13]);
    tileloadd(tmm5, ptr[rdx + r13]);
    tileloadd(tmm6, ptr[rdi + rcx]);
    tdpbf16ps(tmm0, tmm4, tmm6);
    tdpbf16ps(tmm2, tmm5, tmm6);
    if (halves_ == 2) {
      tileloadd(tmm7, ptr[rdi + rcx + 64]);
      tdpbf16ps(tmm1, tmm4, tmm7);
      tdpbf16ps(tmm3, tmm5, tmm7);
    }
    add(rsi, 64);
    add(rdx, 64);
    add(rdi, kBBytesPerKBlock);
    dec(rax);
    jnz(k_loop, T_NEAR);
    L(k_done);

    EmitOutputPanel(tmm0, tmm1, 16, 0, 0, valid0, valid1);
    EmitOutputPanel(tmm2, tmm3, second_rows, 16, 0, valid0, valid1);
    add(r9, r14);
    add(r10, direct_bf16_ ? 64 : 128);
    dec(rbx);
    jnz(block_loop, T_NEAR);
    EmitMPanelAdvance(panel_loop);

    L(all_done);
    EmitAmxReturn();
  }

  void GenerateM1N4() {
    if (rows_ < 1 || rows_ > 16 || n_valid_ != 64) {
      throw std::invalid_argument("invalid AMX W2 1M-by-4N specialization");
    }
    EmitAmxPrologue();

    Xbyak::Label panel_loop;
    Xbyak::Label block_loop;
    Xbyak::Label k_loop;
    Xbyak::Label k_tail;
    Xbyak::Label k_done;
    Xbyak::Label all_done;
    EmitMPanelBegin(panel_loop, all_done);
    L(block_loop);
    tilezero(tmm0);
    tilezero(tmm1);
    tilezero(tmm2);
    tilezero(tmm3);
    mov(rsi, r8);
    mov(rdi, r9);
    lea(rdx, ptr[r9 + r14]);
    mov(rax, r12);
    mov(ecx, 128);
    cmp(rax, 2);
    jb(k_tail, T_NEAR);
    align(64);
    L(k_loop);
    EmitM1N4KBlock(tmm4, 0);
    EmitM1N4KBlock(tmm5, 64);
    add(rsi, 128);
    add(rdi, 2 * kBBytesPerKBlock);
    add(rdx, 2 * kBBytesPerKBlock);
    sub(rax, 2);
    cmp(rax, 2);
    jae(k_loop, T_NEAR);
    L(k_tail);
    test(rax, rax);
    jz(k_done, T_NEAR);
    EmitM1N4KBlock(tmm4, 0);
    L(k_done);

    EmitM1N4Output();
    lea(r9, ptr[r9 + r14 * 2]);
    add(r10, direct_bf16_ ? 128 : 256);
    dec(rbx);
    jnz(block_loop, T_NEAR);
    EmitMPanelAdvance(panel_loop);

    L(all_done);
    EmitAmxReturn();
  }

  void Generate() {
    if (pattern_ == AmxJitPattern::kM2N2) {
      GenerateM2N2();
      return;
    }
    if (pattern_ == AmxJitPattern::kM1N4) {
      GenerateM1N4();
      return;
    }
    GenerateM1N2();
  }

  void GenerateM1N2() {
    if (rows_ < 1 || rows_ > 16 || n_valid_ < 1 || n_valid_ > 32) {
      throw std::invalid_argument("invalid AMX W2 JIT specialization");
    }

    const int valid0 = std::min(n_valid_, 16);
    const int valid1 = std::max(n_valid_ - 16, 0);
    EmitAmxPrologue();
    EmitMasks(valid0, valid1);

    Xbyak::Label panel_loop;
    Xbyak::Label block_loop;
    Xbyak::Label k_loop;
    Xbyak::Label k_tail;
    Xbyak::Label k_done;
    Xbyak::Label all_done;
    EmitMPanelBegin(panel_loop, all_done);
    L(block_loop);
    tilezero(tmm0);
    if (halves_ == 2) {
      tilezero(tmm1);
    }
    mov(rsi, r8);
    mov(rdi, r9);
    mov(rax, r12);
    mov(ecx, 128);
    cmp(rax, 2);
    jb(k_tail, T_NEAR);
    align(64);
    L(k_loop);
    EmitKBlock(tmm2, tmm4, tmm5, 0);
    EmitKBlock(tmm3, tmm6, tmm7, 64);
    add(rsi, 128);
    add(rdi, 2 * kBBytesPerKBlock);
    sub(rax, 2);
    cmp(rax, 2);
    jae(k_loop, T_NEAR);
    L(k_tail);
    test(rax, rax);
    jz(k_done, T_NEAR);
    EmitKBlock(tmm2, tmm4, tmm5, 0);
    L(k_done);

    EmitOutputPanel(tmm0, tmm1, rows_, 0, 0, valid0, valid1);
    add(r9, r14);
    add(r10, direct_bf16_ ? 64 : 128);
    dec(rbx);
    jnz(block_loop, T_NEAR);
    EmitMPanelAdvance(panel_loop);

    L(all_done);
    EmitAmxReturn();
  }

  int rows_;
  int n_valid_;
  int halves_;
  bool direct_bf16_;
  AmxJitPattern pattern_;
  AmxW2Epilogue w2_epilogue_;
  AmxTileStateMode tile_state_mode_;
  alignas(64) TileConfig config_;
};

struct KernelCache {
  std::mutex mutex;
  std::map<KernelKey, KernelHandle> kernels;
  JitStats stats;
};

KernelCache& GetKernelCache() {
  static KernelCache cache;
  return cache;
}

KernelHandle GenerateKernel(const KernelKey& key) {
  if (key.isa == X86JitIsa::kAmxBf16) {
    if (key.operation == JitOperation::kW13) {
      auto owner = std::make_shared<AmxW13Generator>(key.rows, key.silu_degree, key.amx_pattern, key.amx_silu_epilogue,
                                                     key.amx_tile_state_mode);
      return KernelHandle{owner, owner->getCode<JitFunction>(), owner->getSize(), {}};
    }
    const bool direct_bf16 = key.output == JitOutput::kDirectBf16;
    auto owner = std::make_shared<AmxW2Generator>(key.rows, key.n_valid, direct_bf16, key.amx_pattern,
                                                  key.amx_w2_epilogue, key.amx_tile_state_mode);
    return KernelHandle{owner, owner->getCode<JitFunction>(), owner->getSize(), {}};
  }
  if (key.operation == JitOperation::kW13) {
    auto owner = std::make_shared<W13Generator>(key.rows, key.silu_degree);
    return KernelHandle{owner, owner->getCode<JitFunction>(), owner->getSize(), {}};
  }
  const bool direct_bf16 = key.output == JitOutput::kDirectBf16;
  auto owner = std::make_shared<W2Generator>(key.rows, key.n_valid, direct_bf16);
  return KernelHandle{owner, owner->getCode<JitFunction>(), owner->getSize(), {}};
}

KernelHandle ResolveKernel(const KernelKey& key, ImplementationMode mode) {
  KernelCache& cache = GetKernelCache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  const auto found = cache.kernels.find(key);
  if (found != cache.kernels.end()) {
    if (!found->second && mode == ImplementationMode::kJit) {
      throw std::runtime_error(found->second.error);
    }
    return found->second;
  }

  const auto start = std::chrono::steady_clock::now();
  KernelHandle handle;
  try {
    handle = GenerateKernel(key);
  } catch (const std::exception& error) {
    const char* isa = key.isa == X86JitIsa::kAmxBf16 ? "AMX BF16" : "AVX-512 BF16";
    handle.error = std::string(isa) + " JIT generation failed: " + error.what();
  }
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start);
  cache.stats.generation_nanoseconds += static_cast<uint64_t>(elapsed.count());
  if (handle) {
    ++cache.stats.kernel_count;
    cache.stats.code_bytes += handle.code_bytes;
  }
  cache.kernels.emplace(key, handle);
  if (!handle && mode == ImplementationMode::kJit) {
    throw std::runtime_error(handle.error);
  }
  return handle;
}

KernelKey W13Key(int rows, int degree) {
  return KernelKey{X86JitIsa::kAvx512Bf16,       JitOperation::kW13,    static_cast<uint8_t>(rows), 16,
                   static_cast<uint8_t>(degree), JitOutput::kPackedBf16};
}

KernelKey W2Key(int rows, int n_valid, bool direct_bf16) {
  return KernelKey{X86JitIsa::kAvx512Bf16,
                   JitOperation::kW2,
                   static_cast<uint8_t>(rows),
                   static_cast<uint8_t>(n_valid),
                   0,
                   direct_bf16 ? JitOutput::kDirectBf16 : JitOutput::kRouteF32};
}

KernelKey AmxW13Key(int rows, int degree, AmxJitPattern pattern, AmxSiluEpilogue silu_epilogue,
                    AmxTileStateMode tile_state_mode) {
  return KernelKey{X86JitIsa::kAmxBf16,
                   JitOperation::kW13,
                   static_cast<uint8_t>(rows),
                   16,
                   static_cast<uint8_t>(degree),
                   JitOutput::kPackedBf16,
                   pattern,
                   silu_epilogue,
                   AmxW2Epilogue::kBaseline,
                   tile_state_mode};
}

KernelKey AmxW2Key(int rows, int n_valid, bool direct_bf16, AmxJitPattern pattern, AmxW2Epilogue w2_epilogue,
                   AmxTileStateMode tile_state_mode) {
  const JitOutput output =
      direct_bf16 ? JitOutput::kDirectBf16
                  : (w2_epilogue == AmxW2Epilogue::kTileStore ? JitOutput::kContiguousF32 : JitOutput::kRouteF32);
  return KernelKey{X86JitIsa::kAmxBf16,
                   JitOperation::kW2,
                   static_cast<uint8_t>(rows),
                   static_cast<uint8_t>(n_valid),
                   0,
                   output,
                   pattern,
                   AmxSiluEpilogue::kBaseline,
                   w2_epilogue,
                   tile_state_mode};
}

bool ResolveRowKernels(int rows, int degree, int hidden_size, bool direct_bf16, ImplementationMode mode) {
  if (!ResolveKernel(W13Key(rows, degree), mode)) {
    return false;
  }
  if (hidden_size >= 32 && !ResolveKernel(W2Key(rows, 32, direct_bf16), mode)) {
    return false;
  }
  const int tail = hidden_size % 32;
  if (tail != 0 && !ResolveKernel(W2Key(rows, tail, direct_bf16), mode)) {
    return false;
  }
  return true;
}

void ResolveAmxRowKernels(int rows, int degree, int hidden_size, bool direct_bf16, AmxJitPattern pattern,
                          AmxSiluEpilogue silu_epilogue, AmxW2Epilogue w2_epilogue, AmxTileStateMode tile_state_mode) {
  ResolveKernel(AmxW13Key(rows, degree, pattern, silu_epilogue, tile_state_mode), ImplementationMode::kJit);
  if (pattern == AmxJitPattern::kM1N4) {
    // W13 may have an odd F16 block and W2 may have an odd N32 block.
    ResolveKernel(AmxW13Key(rows, degree, AmxJitPattern::kM1N2, silu_epilogue, tile_state_mode),
                  ImplementationMode::kJit);
    if (hidden_size >= 64) {
      ResolveKernel(AmxW2Key(rows, 64, direct_bf16, pattern, w2_epilogue, tile_state_mode), ImplementationMode::kJit);
    }
    if ((hidden_size / 32) % 2 != 0) {
      ResolveKernel(AmxW2Key(rows, 32, direct_bf16, AmxJitPattern::kM1N2, w2_epilogue, tile_state_mode),
                    ImplementationMode::kJit);
    }
    const int tail = hidden_size % 32;
    if (tail != 0) {
      ResolveKernel(AmxW2Key(rows, tail, direct_bf16, AmxJitPattern::kM1N2, w2_epilogue, tile_state_mode),
                    ImplementationMode::kJit);
    }
    return;
  }
  if (hidden_size >= 32) {
    ResolveKernel(AmxW2Key(rows, 32, direct_bf16, pattern, w2_epilogue, tile_state_mode), ImplementationMode::kJit);
  }
  const int tail = hidden_size % 32;
  if (tail != 0) {
    ResolveKernel(AmxW2Key(rows, tail, direct_bf16, pattern, w2_epilogue, tile_state_mode), ImplementationMode::kJit);
  }
}

#endif  // FUSED_CPP_MOE_HAS_XBYAK

}  // namespace

void PrepareJitKernels(const std::vector<int>& row_counts, int silu_poly_degree, int hidden_size, bool direct_bf16) {
  const ImplementationMode mode = GetImplementationMode();
  if (mode == ImplementationMode::kIntrinsic) {
    return;
  }
#if defined(FUSED_CPP_MOE_HAS_XBYAK) && FUSED_CPP_MOE_HAS_XBYAK
  std::array<bool, 13> prepared{};
  for (int rows : row_counts) {
    if (rows <= 0) {
      continue;
    }
    if (rows >= 12 && !prepared[12]) {
      ResolveRowKernels(12, silu_poly_degree, hidden_size, direct_bf16, mode);
      prepared[12] = true;
    }
    const int tail = rows % 12;
    if (tail != 0 && !prepared[tail]) {
      ResolveRowKernels(tail, silu_poly_degree, hidden_size, direct_bf16, mode);
      prepared[tail] = true;
    }
  }
#else
  if (mode == ImplementationMode::kJit) {
    throw std::runtime_error("FUSED_CPP_MOE_AVX512_IMPL=jit requires a build with the Xbyak submodule available");
  }
  (void)row_counts;
  (void)silu_poly_degree;
  (void)hidden_size;
  (void)direct_bf16;
#endif
}

void PrepareAmxJitKernels(const std::vector<int>& row_counts, int silu_poly_degree, int hidden_size, bool direct_bf16) {
#if defined(FUSED_CPP_MOE_HAS_XBYAK) && FUSED_CPP_MOE_HAS_XBYAK
  const AmxSiluEpilogue silu_epilogue = ResolveAmxSiluEpilogue();
  const AmxW2Epilogue w2_epilogue = EffectiveAmxW2Epilogue(ResolveAmxW2Epilogue(), direct_bf16);
  const AmxTileStateMode requested_tile_state_mode = ResolveAmxTileStateMode();
  std::array<std::array<std::array<bool, 2>, 33>, 3> prepared{};
  auto prepare_rows = [&](int kernel_rows, AmxJitPattern kernel_pattern, AmxTileStateMode tile_state_mode) {
    bool& is_prepared = prepared[AmxPatternIndex(kernel_pattern)][kernel_rows][AmxTileStateModeIndex(tile_state_mode)];
    if (!is_prepared) {
      ResolveAmxRowKernels(kernel_rows, silu_poly_degree, hidden_size, direct_bf16, kernel_pattern, silu_epilogue,
                           w2_epilogue, tile_state_mode);
      is_prepared = true;
    }
  };
  for (int rows : row_counts) {
    if (rows <= 0) {
      continue;
    }
    const AmxJitPattern pattern = ResolveAmxJitPattern(rows);
    if (pattern == AmxJitPattern::kM2N2) {
      if (rows >= 32) {
        prepare_rows(32, pattern, EffectiveAmxTileStateMode(requested_tile_state_mode, rows / 32));
      }
      const int remainder = rows % 32;
      if (remainder > 16) {
        prepare_rows(remainder, pattern, AmxTileStateMode::kPerCall);
      } else if (remainder > 0) {
        prepare_rows(remainder, AmxJitPattern::kM1N2, AmxTileStateMode::kPerCall);
      }
      continue;
    }
    if (rows >= 16) {
      prepare_rows(16, pattern, EffectiveAmxTileStateMode(requested_tile_state_mode, rows / 16));
    }
    const int tail = rows % 16;
    if (tail != 0) {
      prepare_rows(tail, pattern, AmxTileStateMode::kPerCall);
    }
  }
#else
  (void)row_counts;
  (void)silu_poly_degree;
  (void)hidden_size;
  (void)direct_bf16;
  throw std::runtime_error("x86_amx_bf16 requires a build with the Xbyak submodule available");
#endif
}

bool AmxW2UsesContiguousRouteOutput() {
#if defined(FUSED_CPP_MOE_HAS_XBYAK) && FUSED_CPP_MOE_HAS_XBYAK
  return ResolveAmxW2Epilogue() == AmxW2Epilogue::kTileStore;
#else
  return false;
#endif
}

void ComputeW13(const uint16_t* a, int a_stride, const uint16_t* packed_b, uint16_t* c, int c_stride, int rows,
                int k_pad, int feature_block_begin, int feature_block_end, int silu_poly_degree) {
  const ImplementationMode mode = GetImplementationMode();
  if (mode == ImplementationMode::kIntrinsic) {
    ComputeW13Intrinsic(a, a_stride, packed_b, c, c_stride, rows, k_pad, feature_block_begin, feature_block_end,
                        silu_poly_degree);
    return;
  }
#if defined(FUSED_CPP_MOE_HAS_XBYAK) && FUSED_CPP_MOE_HAS_XBYAK
  const int full_panels = rows / 12;
  const int tail_rows = rows % 12;
  KernelHandle full_kernel;
  KernelHandle tail_kernel;
  if (full_panels > 0) {
    full_kernel = ResolveKernel(W13Key(12, silu_poly_degree), mode);
  }
  if (tail_rows > 0) {
    tail_kernel = ResolveKernel(W13Key(tail_rows, silu_poly_degree), mode);
  }
  if ((full_panels > 0 && !full_kernel) || (tail_rows > 0 && !tail_kernel)) {
    ComputeW13Intrinsic(a, a_stride, packed_b, c, c_stride, rows, k_pad, feature_block_begin, feature_block_end,
                        silu_poly_degree);
    return;
  }

  const int cache_blocks = GetCacheBlockWindow("FUSED_CPP_MOE_X86_W13_CACHE_BLOCKS");
  if (cache_blocks == 0) {
    for (int block = feature_block_begin; block < feature_block_end; ++block) {
      const uint16_t* b_block = packed_b + static_cast<int64_t>(block) * k_pad * 32;
      for (int panel = 0; panel < full_panels; ++panel) {
        W13Call call{a + static_cast<int64_t>(panel) * a_stride * 16, b_block,
                     c + static_cast<int64_t>(panel) * c_stride * 16 + static_cast<int64_t>(block) * 8 * 32, k_pad / 2,
                     &kSiluConstants};
        full_kernel.function(&call);
      }
      if (tail_rows > 0) {
        W13Call call{a + static_cast<int64_t>(full_panels) * a_stride * 16, b_block,
                     c + static_cast<int64_t>(full_panels) * c_stride * 16 + static_cast<int64_t>(block) * 8 * 32,
                     k_pad / 2, &kSiluConstants};
        tail_kernel.function(&call);
      }
    }
    return;
  }

  ForEachCacheBlockWindow(feature_block_begin, feature_block_end, cache_blocks, [&](int window_begin, int window_end) {
    for (int panel = 0; panel < full_panels; ++panel) {
      for (int block = window_begin; block < window_end; ++block) {
        const uint16_t* b_block = packed_b + static_cast<int64_t>(block) * k_pad * 32;
        W13Call call{a + static_cast<int64_t>(panel) * a_stride * 16, b_block,
                     c + static_cast<int64_t>(panel) * c_stride * 16 + static_cast<int64_t>(block) * 8 * 32, k_pad / 2,
                     &kSiluConstants};
        full_kernel.function(&call);
      }
    }
    if (tail_rows > 0) {
      for (int block = window_begin; block < window_end; ++block) {
        const uint16_t* b_block = packed_b + static_cast<int64_t>(block) * k_pad * 32;
        W13Call call{a + static_cast<int64_t>(full_panels) * a_stride * 16, b_block,
                     c + static_cast<int64_t>(full_panels) * c_stride * 16 + static_cast<int64_t>(block) * 8 * 32,
                     k_pad / 2, &kSiluConstants};
        tail_kernel.function(&call);
      }
    }
  });
#else
  if (mode == ImplementationMode::kJit) {
    throw std::runtime_error("FUSED_CPP_MOE_AVX512_IMPL=jit requires a build with the Xbyak submodule available");
  }
  ComputeW13Intrinsic(a, a_stride, packed_b, c, c_stride, rows, k_pad, feature_block_begin, feature_block_end,
                      silu_poly_degree);
#endif
}

void ComputeW2(const uint16_t* a, int a_stride, const uint16_t* packed_b, float* route_output, uint16_t* direct_output,
               const int64_t* route_ids, int route_stride, int rows, int k_pad, int hidden_size, int output_block_begin,
               int output_block_end, bool direct_bf16) {
  const ImplementationMode mode = GetImplementationMode();
  if (mode == ImplementationMode::kIntrinsic) {
    ComputeW2Intrinsic(a, a_stride, packed_b, route_output, direct_output, route_ids, route_stride, rows, k_pad,
                       hidden_size, output_block_begin, output_block_end, direct_bf16);
    return;
  }
#if defined(FUSED_CPP_MOE_HAS_XBYAK) && FUSED_CPP_MOE_HAS_XBYAK
  const int full_panels = rows / 12;
  const int tail_rows = rows % 12;
  KernelHandle full_main_kernel;
  KernelHandle full_tail_kernel;
  KernelHandle tail_main_kernel;
  KernelHandle tail_tail_kernel;
  bool needs_main_kernel = false;
  bool needs_tail_kernel = false;
  for (int block = output_block_begin; block < output_block_end; ++block) {
    const int n_valid = std::min(32, hidden_size - block * 32);
    if (n_valid <= 0) {
      continue;
    }
    if (n_valid == 32) {
      needs_main_kernel = true;
    } else {
      needs_tail_kernel = true;
    }
  }
  if (full_panels > 0 && needs_main_kernel) {
    full_main_kernel = ResolveKernel(W2Key(12, 32, direct_bf16), mode);
  }
  if (full_panels > 0 && needs_tail_kernel) {
    full_tail_kernel = ResolveKernel(W2Key(12, hidden_size % 32, direct_bf16), mode);
  }
  if (tail_rows > 0 && needs_main_kernel) {
    tail_main_kernel = ResolveKernel(W2Key(tail_rows, 32, direct_bf16), mode);
  }
  if (tail_rows > 0 && needs_tail_kernel) {
    tail_tail_kernel = ResolveKernel(W2Key(tail_rows, hidden_size % 32, direct_bf16), mode);
  }
  if ((full_panels > 0 && needs_main_kernel && !full_main_kernel) ||
      (full_panels > 0 && needs_tail_kernel && !full_tail_kernel) ||
      (tail_rows > 0 && needs_main_kernel && !tail_main_kernel) ||
      (tail_rows > 0 && needs_tail_kernel && !tail_tail_kernel)) {
    ComputeW2Intrinsic(a, a_stride, packed_b, route_output, direct_output, route_ids, route_stride, rows, k_pad,
                       hidden_size, output_block_begin, output_block_end, direct_bf16);
    return;
  }

  const int element_bytes = direct_bf16 ? 2 : 4;
  const int cache_blocks = GetCacheBlockWindow("FUSED_CPP_MOE_X86_W2_CACHE_BLOCKS");
  auto run_block = [&](int block, int panel, bool tail_panel) {
    const int column = block * 32;
    const int n_valid = std::min(32, hidden_size - column);
    if (n_valid <= 0) {
      return;
    }
    const uint16_t* b_block = packed_b + static_cast<int64_t>(block) * k_pad * 32;
    void* output_block =
        direct_bf16 ? static_cast<void*>(direct_output + column) : static_cast<void*>(route_output + column);
    KernelHandle& kernel = tail_panel ? (n_valid == 32 ? tail_main_kernel : tail_tail_kernel)
                                      : (n_valid == 32 ? full_main_kernel : full_tail_kernel);
    const int route_offset = tail_panel ? full_panels * 12 : panel * 12;
    const int64_t a_offset =
        tail_panel ? static_cast<int64_t>(full_panels) * a_stride * 16 : static_cast<int64_t>(panel) * a_stride * 16;
    W2Call call{a + a_offset, b_block,
                output_block, route_ids + route_offset,
                k_pad / 2,    static_cast<int64_t>(route_stride) * element_bytes};
    kernel.function(&call);
  };

  if (cache_blocks == 0) {
    for (int block = output_block_begin; block < output_block_end; ++block) {
      for (int panel = 0; panel < full_panels; ++panel) {
        run_block(block, panel, false);
      }
      if (tail_rows > 0) {
        run_block(block, full_panels, true);
      }
    }
    return;
  }

  ForEachCacheBlockWindow(output_block_begin, output_block_end, cache_blocks, [&](int window_begin, int window_end) {
    for (int panel = 0; panel < full_panels; ++panel) {
      for (int block = window_begin; block < window_end; ++block) {
        run_block(block, panel, false);
      }
    }
    if (tail_rows > 0) {
      for (int block = window_begin; block < window_end; ++block) {
        run_block(block, full_panels, true);
      }
    }
  });
#else
  if (mode == ImplementationMode::kJit) {
    throw std::runtime_error("FUSED_CPP_MOE_AVX512_IMPL=jit requires a build with the Xbyak submodule available");
  }
  ComputeW2Intrinsic(a, a_stride, packed_b, route_output, direct_output, route_ids, route_stride, rows, k_pad,
                     hidden_size, output_block_begin, output_block_end, direct_bf16);
#endif
}

void ComputeW13Amx(const uint16_t* a, int a_stride, const uint16_t* packed_b, uint16_t* c, int c_stride, int rows,
                   int k_pad, int feature_block_begin, int feature_block_end, int silu_poly_degree) {
#if defined(FUSED_CPP_MOE_HAS_XBYAK) && FUSED_CPP_MOE_HAS_XBYAK
  if (!EnsureAmxThreadPermission()) {
    throw std::runtime_error("x86_amx_bf16 could not enable XTILEDATA for the current worker thread");
  }
  const int block_count = feature_block_end - feature_block_begin;
  if (rows <= 0 || block_count <= 0) {
    return;
  }
  if (k_pad % 32 != 0) {
    throw std::invalid_argument("AMX W13 K padding must be a multiple of 32");
  }

  const AmxJitPattern pattern = ResolveAmxJitPattern(rows);
  const AmxSiluEpilogue silu_epilogue = ResolveAmxSiluEpilogue();
  const AmxTileStateMode requested_tile_state_mode = ResolveAmxTileStateMode();
  auto run_single_m = [&](AmxJitPattern kernel_pattern, int kernel_block_count, int block_begin) {
    if (kernel_block_count <= 0) {
      return;
    }
    const int full_panels = rows / 16;
    const int tail_rows = rows % 16;
    const AmxTileStateMode full_tile_state_mode = EffectiveAmxTileStateMode(requested_tile_state_mode, full_panels);
    KernelHandle full_kernel;
    KernelHandle tail_kernel;
    if (full_panels > 0) {
      full_kernel = ResolveKernel(AmxW13Key(16, silu_poly_degree, kernel_pattern, silu_epilogue, full_tile_state_mode),
                                  ImplementationMode::kJit);
    }
    if (tail_rows > 0) {
      tail_kernel = ResolveKernel(
          AmxW13Key(tail_rows, silu_poly_degree, kernel_pattern, silu_epilogue, AmxTileStateMode::kPerCall),
          ImplementationMode::kJit);
    }
    const uint16_t* b_begin = packed_b + static_cast<int64_t>(block_begin) * k_pad * 32;
    auto run_full_panels = [&](int panel, int panel_count) {
      AmxW13Call call{a + static_cast<int64_t>(panel) * 16 * a_stride,
                      b_begin,
                      c + static_cast<int64_t>(panel) * 16 * c_stride + block_begin * 16,
                      k_pad / 32,
                      static_cast<int64_t>(a_stride) * 2,
                      static_cast<int64_t>(k_pad) * 64,
                      static_cast<int64_t>(c_stride) * 2,
                      kernel_block_count,
                      &kSiluConstants,
                      panel_count};
      full_kernel.function(&call);
    };
    if (full_tile_state_mode == AmxTileStateMode::kMacroM) {
      run_full_panels(0, full_panels);
    } else {
      for (int panel = 0; panel < full_panels; ++panel) {
        run_full_panels(panel, 1);
      }
    }
    if (tail_rows > 0) {
      AmxW13Call call{a + static_cast<int64_t>(full_panels) * 16 * a_stride,
                      b_begin,
                      c + static_cast<int64_t>(full_panels) * 16 * c_stride + block_begin * 16,
                      k_pad / 32,
                      static_cast<int64_t>(a_stride) * 2,
                      static_cast<int64_t>(k_pad) * 64,
                      static_cast<int64_t>(c_stride) * 2,
                      kernel_block_count,
                      &kSiluConstants,
                      1};
      tail_kernel.function(&call);
    }
  };

  auto run_double_m = [&](int kernel_block_count, int block_begin) {
    const uint16_t* b_begin = packed_b + static_cast<int64_t>(block_begin) * k_pad * 32;
    const int full_pairs = rows / 32;
    const int remainder = rows % 32;
    const AmxTileStateMode full_tile_state_mode = EffectiveAmxTileStateMode(requested_tile_state_mode, full_pairs);
    KernelHandle full_kernel;
    if (full_pairs > 0) {
      full_kernel = ResolveKernel(AmxW13Key(32, silu_poly_degree, pattern, silu_epilogue, full_tile_state_mode),
                                  ImplementationMode::kJit);
    }
    auto run_full_pairs = [&](int pair, int pair_count) {
      AmxW13Call call{a + static_cast<int64_t>(pair) * 32 * a_stride,
                      b_begin,
                      c + static_cast<int64_t>(pair) * 32 * c_stride + block_begin * 16,
                      k_pad / 32,
                      static_cast<int64_t>(a_stride) * 2,
                      static_cast<int64_t>(k_pad) * 64,
                      static_cast<int64_t>(c_stride) * 2,
                      kernel_block_count,
                      &kSiluConstants,
                      pair_count};
      full_kernel.function(&call);
    };
    if (full_tile_state_mode == AmxTileStateMode::kMacroM) {
      run_full_pairs(0, full_pairs);
    } else {
      for (int pair = 0; pair < full_pairs; ++pair) {
        run_full_pairs(pair, 1);
      }
    }
    if (remainder > 0) {
      const AmxJitPattern remainder_pattern = remainder > 16 ? AmxJitPattern::kM2N2 : AmxJitPattern::kM1N2;
      KernelHandle remainder_kernel = ResolveKernel(
          AmxW13Key(remainder, silu_poly_degree, remainder_pattern, silu_epilogue, AmxTileStateMode::kPerCall),
          ImplementationMode::kJit);
      AmxW13Call call{a + static_cast<int64_t>(full_pairs) * 32 * a_stride,
                      b_begin,
                      c + static_cast<int64_t>(full_pairs) * 32 * c_stride + block_begin * 16,
                      k_pad / 32,
                      static_cast<int64_t>(a_stride) * 2,
                      static_cast<int64_t>(k_pad) * 64,
                      static_cast<int64_t>(c_stride) * 2,
                      kernel_block_count,
                      &kSiluConstants,
                      1};
      remainder_kernel.function(&call);
    }
  };

  auto run_range = [&](int block_begin, int range_block_count) {
    if (pattern == AmxJitPattern::kM1N4) {
      const int pair_count = range_block_count / 2;
      run_single_m(pattern, pair_count, block_begin);
      if (range_block_count % 2 != 0) {
        run_single_m(AmxJitPattern::kM1N2, 1, block_begin + pair_count * 2);
      }
      return;
    }
    if (pattern == AmxJitPattern::kM1N2) {
      run_single_m(pattern, range_block_count, block_begin);
      return;
    }
    run_double_m(range_block_count, block_begin);
  };

  const int cache_blocks = GetCacheBlockWindow("FUSED_CPP_MOE_X86_W13_CACHE_BLOCKS",
                                               AutomaticAmxCacheBlocks(k_pad, kAmxW13CacheTargetBytes, pattern));
  ForEachCacheBlockWindow(feature_block_begin, feature_block_end, cache_blocks, [&](int window_begin, int window_end) {
    run_range(window_begin, window_end - window_begin);
  });
#else
  (void)a;
  (void)a_stride;
  (void)packed_b;
  (void)c;
  (void)c_stride;
  (void)rows;
  (void)k_pad;
  (void)feature_block_begin;
  (void)feature_block_end;
  (void)silu_poly_degree;
  throw std::runtime_error("x86_amx_bf16 requires a build with the Xbyak submodule available");
#endif
}

void ComputeW2Amx(const uint16_t* a, int a_stride, const uint16_t* packed_b, float* route_output,
                  uint16_t* direct_output, const int64_t* route_ids, int route_stride, int rows, int k_pad,
                  int hidden_size, int output_block_begin, int output_block_end, bool direct_bf16) {
#if defined(FUSED_CPP_MOE_HAS_XBYAK) && FUSED_CPP_MOE_HAS_XBYAK
  if (!EnsureAmxThreadPermission()) {
    throw std::runtime_error("x86_amx_bf16 could not enable XTILEDATA for the current worker thread");
  }
  if (rows <= 0 || output_block_begin >= output_block_end) {
    return;
  }
  if (k_pad % 32 != 0) {
    throw std::invalid_argument("AMX W2 K padding must be a multiple of 32");
  }

  const AmxJitPattern pattern = ResolveAmxJitPattern(rows);
  const AmxW2Epilogue w2_epilogue = EffectiveAmxW2Epilogue(ResolveAmxW2Epilogue(), direct_bf16);
  const AmxTileStateMode requested_tile_state_mode = ResolveAmxTileStateMode();
  const bool contiguous_output = w2_epilogue == AmxW2Epilogue::kTileStore;
  if (!contiguous_output && route_ids == nullptr) {
    throw std::invalid_argument("AMX W2 route ids are required for a route-aware store epilogue");
  }
  const int full_blocks = hidden_size / 32;
  const int main_begin = std::min(output_block_begin, full_blocks);
  const int main_end = std::min(output_block_end, full_blocks);
  const int tail_columns = hidden_size % 32;
  const int element_bytes = direct_bf16 ? 2 : 4;
  auto output_for_row_offset = [&](void* output_begin, int row_offset) -> void* {
    if (!contiguous_output || row_offset == 0) {
      return output_begin;
    }
    return static_cast<void*>(static_cast<uint8_t*>(output_begin) +
                              static_cast<int64_t>(row_offset) * route_stride * element_bytes);
  };
  auto routes_for_row_offset = [&](int row_offset) -> const int64_t* {
    return contiguous_output ? nullptr : route_ids + row_offset;
  };

  auto run_single_m_range = [&](int block_begin, int kernel_block_count, int n_valid, AmxJitPattern kernel_pattern) {
    if (kernel_block_count <= 0) {
      return;
    }
    const int full_panels = rows / 16;
    const int tail_rows = rows % 16;
    const AmxTileStateMode full_tile_state_mode = EffectiveAmxTileStateMode(requested_tile_state_mode, full_panels);
    KernelHandle full_kernel;
    KernelHandle tail_kernel;
    if (full_panels > 0) {
      full_kernel = ResolveKernel(AmxW2Key(16, n_valid, direct_bf16, kernel_pattern, w2_epilogue, full_tile_state_mode),
                                  ImplementationMode::kJit);
    }
    if (tail_rows > 0) {
      tail_kernel = ResolveKernel(
          AmxW2Key(tail_rows, n_valid, direct_bf16, kernel_pattern, w2_epilogue, AmxTileStateMode::kPerCall),
          ImplementationMode::kJit);
    }
    const uint16_t* b_begin = packed_b + static_cast<int64_t>(block_begin) * k_pad * 32;
    void* output_begin = direct_bf16 ? static_cast<void*>(direct_output + block_begin * 32)
                                     : static_cast<void*>(route_output + block_begin * 32);
    auto run_full_panels = [&](int panel, int panel_count) {
      AmxW2Call call{a + static_cast<int64_t>(panel) * 16 * a_stride,
                     b_begin,
                     output_for_row_offset(output_begin, panel * 16),
                     routes_for_row_offset(panel * 16),
                     k_pad / 32,
                     static_cast<int64_t>(a_stride) * 2,
                     static_cast<int64_t>(k_pad) * 64,
                     static_cast<int64_t>(route_stride) * element_bytes,
                     kernel_block_count,
                     panel_count};
      full_kernel.function(&call);
    };
    if (full_tile_state_mode == AmxTileStateMode::kMacroM) {
      run_full_panels(0, full_panels);
    } else {
      for (int panel = 0; panel < full_panels; ++panel) {
        run_full_panels(panel, 1);
      }
    }
    if (tail_rows > 0) {
      AmxW2Call call{a + static_cast<int64_t>(full_panels) * 16 * a_stride,
                     b_begin,
                     output_for_row_offset(output_begin, full_panels * 16),
                     routes_for_row_offset(full_panels * 16),
                     k_pad / 32,
                     static_cast<int64_t>(a_stride) * 2,
                     static_cast<int64_t>(k_pad) * 64,
                     static_cast<int64_t>(route_stride) * element_bytes,
                     kernel_block_count,
                     1};
      tail_kernel.function(&call);
    }
  };

  auto run_double_m_range = [&](int block_begin, int kernel_block_count, int n_valid) {
    if (kernel_block_count <= 0) {
      return;
    }
    const int full_pairs = rows / 32;
    const int remainder = rows % 32;
    const AmxTileStateMode full_tile_state_mode = EffectiveAmxTileStateMode(requested_tile_state_mode, full_pairs);
    const uint16_t* b_begin = packed_b + static_cast<int64_t>(block_begin) * k_pad * 32;
    void* output_begin = direct_bf16 ? static_cast<void*>(direct_output + block_begin * 32)
                                     : static_cast<void*>(route_output + block_begin * 32);
    KernelHandle full_kernel;
    if (full_pairs > 0) {
      full_kernel =
          ResolveKernel(AmxW2Key(32, n_valid, direct_bf16, AmxJitPattern::kM2N2, w2_epilogue, full_tile_state_mode),
                        ImplementationMode::kJit);
    }
    auto run_full_pairs = [&](int pair, int pair_count) {
      AmxW2Call call{a + static_cast<int64_t>(pair) * 32 * a_stride,
                     b_begin,
                     output_for_row_offset(output_begin, pair * 32),
                     routes_for_row_offset(pair * 32),
                     k_pad / 32,
                     static_cast<int64_t>(a_stride) * 2,
                     static_cast<int64_t>(k_pad) * 64,
                     static_cast<int64_t>(route_stride) * element_bytes,
                     kernel_block_count,
                     pair_count};
      full_kernel.function(&call);
    };
    if (full_tile_state_mode == AmxTileStateMode::kMacroM) {
      run_full_pairs(0, full_pairs);
    } else {
      for (int pair = 0; pair < full_pairs; ++pair) {
        run_full_pairs(pair, 1);
      }
    }
    if (remainder > 0) {
      const AmxJitPattern remainder_pattern = remainder > 16 ? AmxJitPattern::kM2N2 : AmxJitPattern::kM1N2;
      KernelHandle remainder_kernel = ResolveKernel(
          AmxW2Key(remainder, n_valid, direct_bf16, remainder_pattern, w2_epilogue, AmxTileStateMode::kPerCall),
          ImplementationMode::kJit);
      AmxW2Call call{a + static_cast<int64_t>(full_pairs) * 32 * a_stride,
                     b_begin,
                     output_for_row_offset(output_begin, full_pairs * 32),
                     routes_for_row_offset(full_pairs * 32),
                     k_pad / 32,
                     static_cast<int64_t>(a_stride) * 2,
                     static_cast<int64_t>(k_pad) * 64,
                     static_cast<int64_t>(route_stride) * element_bytes,
                     kernel_block_count,
                     1};
      remainder_kernel.function(&call);
    }
  };

  auto run_range = [&](int block_begin, int range_block_count, int n_valid) {
    if (pattern == AmxJitPattern::kM2N2) {
      run_double_m_range(block_begin, range_block_count, n_valid);
      return;
    }
    if (pattern == AmxJitPattern::kM1N4 && n_valid == 32) {
      const int pair_count = range_block_count / 2;
      run_single_m_range(block_begin, pair_count, 64, pattern);
      if (range_block_count % 2 != 0) {
        run_single_m_range(block_begin + pair_count * 2, 1, 32, AmxJitPattern::kM1N2);
      }
      return;
    }
    run_single_m_range(block_begin, range_block_count, n_valid, AmxJitPattern::kM1N2);
  };

  const int cache_blocks = GetCacheBlockWindow("FUSED_CPP_MOE_X86_W2_CACHE_BLOCKS",
                                               AutomaticAmxCacheBlocks(k_pad, kAmxW2CacheTargetBytes, pattern));
  ForEachCacheBlockWindow(main_begin, main_end, cache_blocks, [&](int window_begin, int window_end) {
    run_range(window_begin, window_end - window_begin, 32);
  });
  const int tail_block = full_blocks;
  if (tail_columns > 0 && output_block_begin <= tail_block && tail_block < output_block_end) {
    run_range(tail_block, 1, tail_columns);
  }
#else
  (void)a;
  (void)a_stride;
  (void)packed_b;
  (void)route_output;
  (void)direct_output;
  (void)route_ids;
  (void)route_stride;
  (void)rows;
  (void)k_pad;
  (void)hidden_size;
  (void)output_block_begin;
  (void)output_block_end;
  (void)direct_bf16;
  throw std::runtime_error("x86_amx_bf16 requires a build with the Xbyak submodule available");
#endif
}

JitStats GetJitStats() {
#if defined(FUSED_CPP_MOE_HAS_XBYAK) && FUSED_CPP_MOE_HAS_XBYAK
  KernelCache& cache = GetKernelCache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  return cache.stats;
#else
  return {};
#endif
}

}  // namespace fused_cpp::moe::x86::avx512_bf16
