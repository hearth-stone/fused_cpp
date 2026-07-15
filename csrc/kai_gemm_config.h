// csrc/kai_gemm_config.h
// SPDX-License-Identifier: Apache-2.0
//
// 基于 KleidiAI 微内核的多线程分块 GEMM —— 编译期配置
//
// 本头文件集中定义三级 Cache 大小、KleidiAI 微内核几何参数
// （mr/nr/kr），以及由这些量推导出的分块参数 Kc/Mc/Nc。
// 所有值均为 ``constexpr``，可在编译期展开。

#pragma once

#include <cstddef>
#include <cstdint>

namespace fused_cpp {
namespace kai_gemm {

// ---------------------------------------------------------------------------
// 1. 三级 Cache 大小（编译期常量，单位：字节）
// ---------------------------------------------------------------------------
//
// 默认按当前 ARM 服务器级别芯片的典型值配置：
//   - L1D per core : 64 KB
//   - L2  per core : 512 KB
//   - L3  shared   : 32 MB
// 用户可在构建时通过 -D 宏覆盖。
// ---------------------------------------------------------------------------

#ifndef KAI_GEMM_L1_CACHE_BYTES
#define KAI_GEMM_L1_CACHE_BYTES (64 * 1024)
#endif

#ifndef KAI_GEMM_L2_CACHE_BYTES
#define KAI_GEMM_L2_CACHE_BYTES (512 * 1024)
#endif

#ifndef KAI_GEMM_L3_CACHE_BYTES
#define KAI_GEMM_L3_CACHE_BYTES (32 * 1024 * 1024)
#endif

constexpr std::size_t kL1CacheBytes = static_cast<std::size_t>(KAI_GEMM_L1_CACHE_BYTES);
constexpr std::size_t kL2CacheBytes = static_cast<std::size_t>(KAI_GEMM_L2_CACHE_BYTES);
constexpr std::size_t kL3CacheBytes = static_cast<std::size_t>(KAI_GEMM_L3_CACHE_BYTES);

// ---------------------------------------------------------------------------
// 2. KleidiAI 微内核几何常量
//
//   使用的微内核族：matmul_clamp_{f32,f16}_bf16p8x4_bf16p12x4b_8x12_neon_mmla
//   - mr = 8     （LHS pack 行 tile）
//   - nr = 12    （RHS pack 列 tile）
//   - kr = 4     （K 方向 pack 粒度）
// ---------------------------------------------------------------------------

constexpr std::size_t kMr = 8;
constexpr std::size_t kNr = 12;
constexpr std::size_t kKr = 4;

// 每个 BF16 元素的字节数。
constexpr std::size_t kBf16Bytes = 2;
// 每个 FP32 bias 元素的字节数。
constexpr std::size_t kBiasBytes = 4;

// ---------------------------------------------------------------------------
// 3. 分块参数的编译期推导
//
// 分块策略：
//   - Kc 由 L1 Cache 约束：A 的一个 [mr, Kc] 微面板（packed BF16）应完全
//     驻留 L1，B 的数据流式通过 L1。这里只对 A 微面板施加约束，按经验
//     为 B 预留约 1/4 的 L1 空间。
//   - Mc 由 L2 Cache 约束：A 的一个 [Mc, Kc] 分块（packed BF16）应完全
//     驻留 L2。
//   - Nc 由 L3 Cache 约束：B 的一个 [Kc, Nc] 分块（packed BF16 + FP32 bias）
//     应完全驻留 L3。
// ---------------------------------------------------------------------------

// 按经验为 B 在 L1 中预留的比例（分子 / 分母）。
constexpr std::size_t kL1ReserveNum = 3;
constexpr std::size_t kL1ReserveDen = 4;

// 向下对齐到 ``align`` 的整数倍；至少保留 ``align`` 个元素。
constexpr std::size_t AlignDown(std::size_t v, std::size_t align) { return (v < align) ? align : (v / align) * align; }

// ---- Kc ------------------------------------------------------------------
// A 微面板大小 = mr * Kc * sizeof(bf16) <= L1 * kL1ReserveNum / kL1ReserveDen
// => Kc <= (L1 * num) / (den * mr * 2)
constexpr std::size_t ComputeKc() {
  const std::size_t raw = (kL1CacheBytes * kL1ReserveNum) / (kL1ReserveDen * kMr * kBf16Bytes);
  return AlignDown(raw, kKr);
}

constexpr std::size_t kKc = ComputeKc();

// ---- Mc ------------------------------------------------------------------
// A 分块大小 = Mc * Kc * sizeof(bf16) <= L2
// => Mc <= L2 / (Kc * 2)
constexpr std::size_t ComputeMc() {
  const std::size_t raw = kL2CacheBytes / (kKc * kBf16Bytes);
  return AlignDown(raw, kMr);
}

constexpr std::size_t kMc = ComputeMc();

// ---- Nc ------------------------------------------------------------------
// B 分块大小（含 bias）= Nc * (Kc * sizeof(bf16) + sizeof(float)) <= L3
// => Nc <= L3 / (Kc * 2 + 4)
constexpr std::size_t ComputeNc() {
  const std::size_t raw = kL3CacheBytes / (kKc * kBf16Bytes + kBiasBytes);
  return AlignDown(raw, kNr);
}

constexpr std::size_t kNc = ComputeNc();

// ---------------------------------------------------------------------------
// 4. 静态断言：确保分块参数的基本合法性
// ---------------------------------------------------------------------------

static_assert(kMr > 0 && kNr > 0 && kKr > 0, "microkernel tile must be positive");
static_assert(kKc >= kKr, "Kc must be >= kr");
static_assert(kMc >= kMr, "Mc must be >= mr");
static_assert(kNc >= kNr, "Nc must be >= nr");
static_assert(kKc % kKr == 0, "Kc must be multiple of kr");
static_assert(kMc % kMr == 0, "Mc must be multiple of mr");
static_assert(kNc % kNr == 0, "Nc must be multiple of nr");

}  // namespace kai_gemm
}  // namespace fused_cpp
