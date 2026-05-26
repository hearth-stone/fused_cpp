#pragma once
// ── 微内核管理框架：trait 接口 ─────────────────────────────────────────────
//
// 概览
// ──────
// 上层 SDPA 主循环（flash2_neon_cache 等）以前直接调用全局自由函数
// `gemm_qkt_8x8(...)` / `gemm_pv_8x8(...)` 等 inline 入口。这种调用方式
// 在「快速替换实现」时不灵活：要换一套实现就得改主路径里的 5 处函数名，
// 而且不便于按名字单独跑单测/基准。
//
// 本文件把这 5 个微内核形状抽象成一个 `MK` trait（静态成员函数集合）：
//
//   struct MK_<Name> {
//     static constexpr const char* kName = "<Name>";
//     static constexpr bool kEnabled = ...;       // 编译期开关
//
//     // QKᵀ 主体（Lq=8, Sk=8）
//     static void qkt_8x8(const at::BFloat16*, int64_t,
//                         const at::BFloat16*, int64_t,
//                         int64_t E, float scale, float* scores_buf);
//     static void qkt_8x8(const float*, int64_t,
//                         const float*, int64_t,
//                         int64_t E, float scale, float* scores_buf);
//
//     // QKᵀ 退化（Lq=8, Sk=4）
//     static void qkt_8x4(...);
//     // QKᵀ 任意尾部（Lq, Sk ∈ [1, 8]）
//     static void qkt_tail(...);
//     // P̂·V 主体（Lq=8, Ev=8）
//     static void pv_8x8(...);
//     // P̂·V 任意尾部
//     static void pv_tail(...);
//   };
//
// 上层 SDPA 改成 `template <class MK>` 接受 trait 类型，main loop 里写
// `MK::qkt_8x8(...)`；编译器对每个 trait 单独生成一份 fully-specialized
// SDPA，热路径里完全没有运行期间接跳转。
//
// 编译期选择：每个 impl 的头文件用 `FUSED_CPP_MK_ENABLE_<NAME>` 宏控制
// 是否定义 `kEnabled = 1`。setup.py 默认全部为 1；只想测一个时可以
// `FUSED_CPP_MK_DISABLE_*` 关掉其它的。
//
// 运行期选择：每个 enabled impl 都会注册一个 SDPA 版本名
// `flash2_neon_cache_<MK::kName>` 到 `REGISTER_SDPA_VERSION` 全局表，
// Python 侧通过 `sdpa_versioned(version="...")` 字符串选取。这一次
// 字符串查表在每次 SDPA 调用入口发生（每次调用 1 次），主循环内是 0
// 间接调用。
//
// 同名微内核也注册到本文件下方的 `MicrokernelRegistry`，提供：
//   * 列出所有 enabled impl 名字
//   * 给定 (impl, dtype, E, Sk) 对每个 op 跑一次正确性单测
//   * 给定 (impl, dtype, E, Sk, iters, warmup) 测每个 op 的 GFLOPS

#include <torch/extension.h>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "neon_cache_config.h"

namespace fused_cpp::sdpa_microkernels {

// ── 微内核形状常量（与 neon_cache_microkernels.h 中的 MICRO_* 一致） ──
constexpr int kMicroLq = 8;
constexpr int kMicroSk = 8;
constexpr int kMicroEv = 8;

// ── MK trait 概念 ──────────────────────────────────────────────────────
//
// 每个具体 impl 都是一个 `struct`（不是 `class`，所有成员都是 public 静态
// 函数）。下面的 helper 模板用 SFINAE 校验某个类型是否满足 trait（编译失败
// 时给出可读错误）。
//
// 我们不需要 C++20 concepts；C++17 够用。

namespace detail {

// 用 SFINAE 探测 `kEnabled`；若 `kEnabled=false` 则不应在 SDPA 中实例化。
template <class MK, class = void>
struct has_enabled : std::false_type {};

template <class MK>
struct has_enabled<MK, std::void_t<decltype(MK::kEnabled)>>
    : std::bool_constant<MK::kEnabled> {};

}  // namespace detail

template <class MK>
inline constexpr bool mk_is_enabled_v = detail::has_enabled<MK>::value;

// ── 单个微内核 op 的元信息（用于 registry / 单测 / 基准）─────────────
//
// dtype 字符串约定：
//   "fp32" / "float32" / "float"  → fp32
//   "bf16" / "bfloat16"            → bf16
// validate / benchmark 入口接收 dtype 字符串后选 op 实现。

// 单个 impl 的 dispatch 表（ABI：所有签名固定），由 mk_registry.cpp 统一
// 收集。这一张表只在 validate / benchmark Python API 走查表分发，热路径
// 上的 SDPA 主循环走模板，不会触碰本表。
struct MicrokernelEntry {
  const char* name;

  // —— 对 fp32 跑全部 op 的正确性 + GFLOPS（impl 自己实例化） ——
  std::map<std::string, double> (*validate_fp32)(int64_t E, int64_t Sk);
  std::map<std::string, double> (*validate_bf16)(int64_t E, int64_t Sk);

  std::map<std::string, double> (*benchmark_fp32)(
      int64_t E, int64_t Sk, int64_t iterations, int64_t warmup);
  std::map<std::string, double> (*benchmark_bf16)(
      int64_t E, int64_t Sk, int64_t iterations, int64_t warmup);
};

// 注册一个 impl。重名直接抛 std::runtime_error。
int mk_register_impl(const MicrokernelEntry& entry);

// 列出所有已注册 impl 名（按注册顺序）。
std::vector<std::string> mk_list_impls();

// 按名查 entry；找不到返回 nullptr。
const MicrokernelEntry* mk_find_impl(const std::string& name);

// validate / benchmark 顶层入口。dtype ∈ {"fp32", "bf16"}。
std::map<std::string, double> mk_validate_dispatch(
    const std::string& impl_name,
    const std::string& dtype,
    int64_t E,
    int64_t Sk);

std::map<std::string, double> mk_benchmark_dispatch(
    const std::string& impl_name,
    const std::string& dtype,
    int64_t E,
    int64_t Sk,
    int64_t iterations,
    int64_t warmup);

// ── 注册宏 ──────────────────────────────────────────────────────────────
//
// 使用：在某个 impl 的 .cpp（例如 mk_baseline.cpp）末尾写：
//   REGISTER_MICROKERNEL_IMPL(MK_Baseline);
// 该宏依赖 impl 类型已被显式实例化 validate/benchmark 模板（见
// mk_registry_helpers.h 中的 `MK_VALIDATE_TMPL` / `MK_BENCHMARK_TMPL`）。

#define MK_REG_CONCAT_INNER(a, b) a##b
#define MK_REG_CONCAT(a, b) MK_REG_CONCAT_INNER(a, b)

#define REGISTER_MICROKERNEL_IMPL(MK_TYPE)                                  \
  static int MK_REG_CONCAT(_mk_reg_, __COUNTER__) =                         \
      ::fused_cpp::sdpa_microkernels::mk_register_impl(                     \
          ::fused_cpp::sdpa_microkernels::MicrokernelEntry{                 \
              MK_TYPE::kName,                                               \
              &::fused_cpp::sdpa_microkernels::                             \
                  validate_microkernels_tmpl<MK_TYPE, float>,               \
              &::fused_cpp::sdpa_microkernels::                             \
                  validate_microkernels_tmpl<MK_TYPE, at::BFloat16>,        \
              &::fused_cpp::sdpa_microkernels::                             \
                  benchmark_microkernels_tmpl<MK_TYPE, float>,              \
              &::fused_cpp::sdpa_microkernels::                             \
                  benchmark_microkernels_tmpl<MK_TYPE, at::BFloat16>,       \
          })

}  // namespace fused_cpp::sdpa_microkernels
