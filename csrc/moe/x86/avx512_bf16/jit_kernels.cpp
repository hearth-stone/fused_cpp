// SPDX-License-Identifier: Apache-2.0
#include "kernels.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

// The ISA is deliberately part of the cache/factory contract even though the
// first implementation only emits AVX-512 BF16. An AMX generator can be added
// without changing executor dispatch or the public packed-weight metadata.
enum class X86JitIsa : uint8_t { kAvx512Bf16, kAmxBf16 };
enum class JitOperation : uint8_t { kW13, kW2 };
enum class JitOutput : uint8_t { kPackedBf16, kRouteF32, kDirectBf16 };

struct KernelKey {
  X86JitIsa isa = X86JitIsa::kAvx512Bf16;
  JitOperation operation = JitOperation::kW13;
  uint8_t rows = 0;
  uint8_t n_valid = 0;
  uint8_t silu_degree = 0;
  JitOutput output = JitOutput::kPackedBf16;

  bool operator<(const KernelKey& other) const {
    return std::tie(isa, operation, rows, n_valid, silu_degree, output) <
           std::tie(other.isa, other.operation, other.rows, other.n_valid, other.silu_degree, other.output);
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
    throw std::runtime_error("AMX BF16 JIT is reserved but not implemented");
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
    handle.error = std::string("AVX-512 BF16 JIT generation failed: ") + error.what();
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
  for (int block = output_block_begin; block < output_block_end; ++block) {
    const int column = block * 32;
    const int n_valid = std::min(32, hidden_size - column);
    if (n_valid <= 0) {
      continue;
    }
    const uint16_t* b_block = packed_b + static_cast<int64_t>(block) * k_pad * 32;
    void* output_block =
        direct_bf16 ? static_cast<void*>(direct_output + column) : static_cast<void*>(route_output + column);
    KernelHandle& full_kernel = n_valid == 32 ? full_main_kernel : full_tail_kernel;
    KernelHandle& tail_kernel = n_valid == 32 ? tail_main_kernel : tail_tail_kernel;
    for (int panel = 0; panel < full_panels; ++panel) {
      W2Call call{
          a + static_cast<int64_t>(panel) * a_stride * 16,   b_block, output_block, route_ids + panel * 12, k_pad / 2,
          static_cast<int64_t>(route_stride) * element_bytes};
      full_kernel.function(&call);
    }
    if (tail_rows > 0) {
      W2Call call{a + static_cast<int64_t>(full_panels) * a_stride * 16,
                  b_block,
                  output_block,
                  route_ids + full_panels * 12,
                  k_pad / 2,
                  static_cast<int64_t>(route_stride) * element_bytes};
      tail_kernel.function(&call);
    }
  }
#else
  if (mode == ImplementationMode::kJit) {
    throw std::runtime_error("FUSED_CPP_MOE_AVX512_IMPL=jit requires a build with the Xbyak submodule available");
  }
  ComputeW2Intrinsic(a, a_stride, packed_b, route_output, direct_output, route_ids, route_stride, rows, k_pad,
                     hidden_size, output_block_begin, output_block_end, direct_bf16);
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
