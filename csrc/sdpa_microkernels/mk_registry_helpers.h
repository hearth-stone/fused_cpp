#pragma once
// ── 微内核管理框架：通用 validate / benchmark 模板 ───────────────────────
//
// 把 `csrc/sdpa_microkernels/neon_cache_benchmark.cpp` 中针对单一全局
// dispatcher 的 validate_microkernels_tmpl / benchmark_microkernels_tmpl
// 抽出来，参数化到 MK trait。每个被启用的 impl 都会显式实例化两个 dtype
// 各自的 validate + benchmark 模板（fp32 + bf16），由 REGISTER_MICROKERNEL_IMPL
// 宏注册到 MicrokernelRegistry。
//
// 设计要点：
//   * 复用 neon_cache_benchmark.cpp 中的 `reference_qkt_value` /
//     `reference_pv_value` / `update_max_abs` / `time_microkernel_loop`
//     等纯 helper，同时让 `MK::qkt_8x8 / qkt_8x4 / qkt_tail / pv_8x8 /
//     pv_tail` 替换原先的 `gemm_*` 全局自由函数。
//   * 模板需要在头文件中，因为 `REGISTER_MICROKERNEL_IMPL` 宏在每个 impl
//     的 .cpp 中显式实例化（一次性写完，setup.py 自动 glob 编译）。
//   * 我们不在这里测试 MK 自己的 enabled 标志：只要 impl 的 .cpp 里写了
//     `REGISTER_MICROKERNEL_IMPL(MK_X)`，就一律走通；编译期开关在 impl
//     头文件里直接控制是否定义 MK_X 类。

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include "neon_cache_config.h"
#include "neon_cache_microkernels.h"
#include "../profile_utils.h"

namespace fused_cpp::sdpa_microkernels {

// ── 计时与正确性的纯 helper（不依赖 MK trait） ───────────────────────────

inline void mk_bench_barrier(const void* p) {
#if defined(__GNUC__) || defined(__clang__)
  asm volatile("" : : "r"(p) : "memory");
#else
  (void)p;
#endif
}

template <typename Fn>
inline double time_microkernel_loop(Fn& fn, int64_t warmup, int64_t iterations) {
  for (int64_t i = 0; i < warmup; ++i) {
    fn();
  }
  mk_bench_barrier(nullptr);
  const auto t0 = ::fused_cpp::profile::now();
  for (int64_t i = 0; i < iterations; ++i) {
    fn();
    mk_bench_barrier(nullptr);
  }
  mk_bench_barrier(nullptr);
  return ::fused_cpp::profile::elapsed_ms(t0) / 1.0e3;
}

inline double mk_checksum_fp32_buffer(const float* data, int64_t len) {
  double sum = 0.0;
  for (int64_t i = 0; i < len; ++i) {
    sum += static_cast<double>(data[i]) *
           static_cast<double>((i % 17) + 1);
  }
  return sum;
}

template <typename scalar_t>
inline float mk_reference_qkt_value(
    const scalar_t* Q, int64_t q_row_stride,
    const scalar_t* K, int64_t k_row_stride,
    int64_t E, float scale, int i, int j) {
  float sum = 0.0f;
  for (int64_t e = 0; e < E; ++e) {
    sum += static_cast<float>(Q[i * q_row_stride + e]) *
           static_cast<float>(K[j * k_row_stride + e]);
  }
  return sum * scale;
}

template <typename scalar_t>
inline float mk_reference_pv_value(
    const float* P_hat, int64_t P_row_stride,
    const scalar_t* V, int64_t v_row_stride,
    int64_t Sk, int i, int j) {
  float sum = 0.0f;
  for (int64_t k = 0; k < Sk; ++k) {
    sum += P_hat[i * P_row_stride + k] *
           static_cast<float>(V[k * v_row_stride + j]);
  }
  return sum;
}

inline double mk_update_max_abs(double current, float actual, float expected) {
  return std::max(current, static_cast<double>(std::abs(actual - expected)));
}

template <class MK, class = void>
struct mk_has_pv_pbf16 : std::false_type {};

template <class MK>
struct mk_has_pv_pbf16<
    MK,
    std::void_t<decltype(MK::kHasPvPbf16), decltype(&MK::pv_8x8_pbf16)>>
    : std::bool_constant<MK::kHasPvPbf16> {};

template <class MK>
inline constexpr bool mk_has_pv_pbf16_v = mk_has_pv_pbf16<MK>::value;

template <class MK, class = void>
struct mk_has_qkt_kcol : std::false_type {};

template <class MK>
struct mk_has_qkt_kcol<
    MK,
    std::void_t<decltype(MK::kHasQktKcol), decltype(&MK::qkt_8x8_kcol)>>
    : std::bool_constant<MK::kHasQktKcol> {};

template <class MK>
inline constexpr bool mk_has_qkt_kcol_v = mk_has_qkt_kcol<MK>::value;

template <typename scalar_t>
inline float mk_reference_pv_pbf16_value(
    const at::BFloat16* P_bf16, int64_t P_row_stride,
    const scalar_t* V, int64_t v_row_stride,
    int64_t Sk, int i, int j) {
  float sum = 0.0f;
  for (int64_t k = 0; k < Sk; ++k) {
    sum += static_cast<float>(P_bf16[i * P_row_stride + k]) *
           static_cast<float>(V[k * v_row_stride + j]);
  }
  return sum;
}

// ── 通用 validate 模板：对一个具体 (MK, scalar_t) 跑全部 5 个 op ───────
//
// 返回的 map 包含：
//   qkt_8x8_max_abs / qkt_8x4_max_abs / qkt_tail_max_abs
//   pv_8x8_max_abs / pv_tail_max_abs
//   E, Sk, has_neon / has_bf16 / has_bfmmla / dtype_is_bf16
template <class MK, typename scalar_t>
std::map<std::string, double> validate_microkernels_tmpl(
    int64_t E, int64_t Sk) {
  constexpr float kScale = 0.75f;
  std::vector<scalar_t> q(8 * E);
  std::vector<scalar_t> k8(8 * E);
  std::vector<scalar_t> k4(4 * E);
  std::vector<scalar_t> ktail(3 * E);
  std::vector<scalar_t> v(8 * Sk);
  std::vector<float> p_hat(8 * Sk);
  std::vector<float> scores(8 * 8, 0.0f);
  std::vector<float> out(8 * 8, 0.0f);

  // 与 neon_cache_benchmark.cpp 中的种子相同，确保可复现。
  for (int64_t i = 0; i < static_cast<int64_t>(q.size()); ++i)
    q[i] = static_cast<scalar_t>(0.01f * static_cast<float>((i % 13) - 6));
  for (int64_t i = 0; i < static_cast<int64_t>(k8.size()); ++i)
    k8[i] = static_cast<scalar_t>(0.02f * static_cast<float>((i % 11) - 5));
  for (int64_t i = 0; i < static_cast<int64_t>(k4.size()); ++i)
    k4[i] = static_cast<scalar_t>(0.03f * static_cast<float>((i % 7) - 3));
  for (int64_t i = 0; i < static_cast<int64_t>(ktail.size()); ++i)
    ktail[i] = static_cast<scalar_t>(0.025f * static_cast<float>((i % 5) - 2));
  for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i)
    v[i] = static_cast<scalar_t>(0.015f * static_cast<float>((i % 9) - 4));
  for (int64_t i = 0; i < static_cast<int64_t>(p_hat.size()); ++i)
    p_hat[i] = 0.005f * static_cast<float>((i % 5) - 2);

  std::map<std::string, double> result;

  // qkt_8x8
  std::fill(scores.begin(), scores.end(), 0.0f);
  MK::qkt_8x8(q.data(), E, k8.data(), E, E, kScale, scores.data());
  double qkt_8x8_max_abs = 0.0;
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      qkt_8x8_max_abs = mk_update_max_abs(
          qkt_8x8_max_abs, scores[i * 8 + j],
          mk_reference_qkt_value(q.data(), E, k8.data(), E, E, kScale, i, j));
    }
  }
  result["qkt_8x8_max_abs"] = qkt_8x8_max_abs;

  if constexpr (std::is_same_v<scalar_t, at::BFloat16> &&
                mk_has_qkt_kcol_v<MK>) {
    std::vector<at::BFloat16> k_col(8 * E);
    pack_k_8rows_to_col_bf16(k8.data(), E, E, k_col.data());
    std::fill(scores.begin(), scores.end(), 0.0f);
    MK::qkt_8x8_kcol(q.data(), E, k_col.data(), E, kScale, scores.data());
    double qkt_8x8_kcol_max_abs = 0.0;
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        qkt_8x8_kcol_max_abs = mk_update_max_abs(
            qkt_8x8_kcol_max_abs, scores[i * 8 + j],
            mk_reference_qkt_value(q.data(), E, k8.data(), E, E, kScale, i, j));
      }
    }
    result["qkt_8x8_kcol_max_abs"] = qkt_8x8_kcol_max_abs;
  }

  // qkt_8x4 — scores 行步长用 8 与 baseline 兼容
  std::fill(scores.begin(), scores.end(), 0.0f);
  MK::qkt_8x4(q.data(), E, k4.data(), E, E, kScale, scores.data(), 8);
  double qkt_8x4_max_abs = 0.0;
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 4; ++j) {
      qkt_8x4_max_abs = mk_update_max_abs(
          qkt_8x4_max_abs, scores[i * 8 + j],
          mk_reference_qkt_value(q.data(), E, k4.data(), E, E, kScale, i, j));
    }
  }
  result["qkt_8x4_max_abs"] = qkt_8x4_max_abs;

  // qkt_tail (Lq=5, Sk=3)
  std::fill(scores.begin(), scores.end(), 0.0f);
  MK::qkt_tail(q.data(), E, ktail.data(), E, E, kScale, scores.data(), 8, 5, 3);
  double qkt_tail_max_abs = 0.0;
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 3; ++j) {
      qkt_tail_max_abs = mk_update_max_abs(
          qkt_tail_max_abs, scores[i * 8 + j],
          mk_reference_qkt_value(q.data(), E, ktail.data(), E, E, kScale, i, j));
    }
  }
  result["qkt_tail_max_abs"] = qkt_tail_max_abs;

  // pv_8x8
  std::fill(out.begin(), out.end(), 0.0f);
  MK::pv_8x8(p_hat.data(), Sk, v.data(), 8, Sk, out.data(), 8);
  double pv_8x8_max_abs = 0.0;
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      pv_8x8_max_abs = mk_update_max_abs(
          pv_8x8_max_abs, out[i * 8 + j],
          mk_reference_pv_value(p_hat.data(), Sk, v.data(), 8, Sk, i, j));
    }
  }
  result["pv_8x8_max_abs"] = pv_8x8_max_abs;

  if constexpr (std::is_same_v<scalar_t, at::BFloat16> &&
                mk_has_pv_pbf16_v<MK>) {
    std::vector<at::BFloat16> p_hat_bf16(8 * Sk);
    for (int64_t i = 0; i < static_cast<int64_t>(p_hat_bf16.size()); ++i) {
      p_hat_bf16[i] = static_cast<at::BFloat16>(p_hat[i]);
    }
    std::fill(out.begin(), out.end(), 0.0f);
    MK::pv_8x8_pbf16(p_hat_bf16.data(), Sk, v.data(), 8, Sk, out.data(), 8);
    double pv_8x8_pbf16_max_abs = 0.0;
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        pv_8x8_pbf16_max_abs = mk_update_max_abs(
            pv_8x8_pbf16_max_abs, out[i * 8 + j],
            mk_reference_pv_pbf16_value(
                p_hat_bf16.data(), Sk, v.data(), 8, Sk, i, j));
      }
    }
    result["pv_8x8_pbf16_max_abs"] = pv_8x8_pbf16_max_abs;
  }

  // pv_tail (Lq=5, Ev=3)
  std::fill(out.begin(), out.end(), 0.0f);
  MK::pv_tail(p_hat.data(), Sk, v.data(), 8, Sk, out.data(), 8, 5, 3);
  double pv_tail_max_abs = 0.0;
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 3; ++j) {
      pv_tail_max_abs = mk_update_max_abs(
          pv_tail_max_abs, out[i * 8 + j],
          mk_reference_pv_value(p_hat.data(), Sk, v.data(), 8, Sk, i, j));
    }
  }
  result["pv_tail_max_abs"] = pv_tail_max_abs;

  result["E"] = static_cast<double>(E);
  result["Sk"] = static_cast<double>(Sk);
  result["has_neon"] = static_cast<double>(FUSED_CPP_SDPA_CACHE_HAS_NEON);
  result["has_bf16"] = static_cast<double>(FUSED_CPP_SDPA_CACHE_HAS_BF16);
  result["has_bfmmla"] = static_cast<double>(FUSED_CPP_SDPA_CACHE_HAS_BFMMLA);
  result["dtype_is_bf16"] =
      static_cast<double>(std::is_same_v<scalar_t, at::BFloat16>);
  return result;
}

// ── 通用 benchmark 模板：对 (MK, scalar_t) 跑 qkt_8x8 / qkt_8x4 / pv_8x8 ──
//
// 返回的 map 包含 `<op>_seconds / <op>_us / <op>_gflops / <op>_checksum`，
// 以及 `direct_microkernel = 1`、`E / Sk / iterations / warmup`、
// `has_neon / has_bf16 / has_bfmmla / dtype_is_bf16`。
//
// 选择跑 8x8 / 8x4 / pv_8x8 三个：与原先 neon_cache_benchmark 对齐，
// 8x8 与 8x4 是 QKᵀ 的两条主路径；pv_8x8 是 P̂·V 的主路径；tail 性能
// 受限于 scalar 兜底，单独基准价值不大（仍可由 validate 验正确性）。
template <class MK, typename scalar_t>
std::map<std::string, double> benchmark_microkernels_tmpl(
    int64_t E, int64_t Sk, int64_t iterations, int64_t warmup) {
  std::vector<scalar_t> q(8 * E);
  std::vector<scalar_t> k8(8 * E);
  std::vector<scalar_t> k4(4 * E);
  std::vector<scalar_t> ktail(8 * E);  // qkt_tail 用，按 Lq=5/Sk=3 视图取
  std::vector<scalar_t> v(8 * Sk);
  std::vector<float> p_hat(8 * Sk);
  std::vector<float> scores(8 * 8, 0.0f);
  std::vector<float> out(8 * 8, 0.0f);

  for (int64_t i = 0; i < static_cast<int64_t>(q.size()); ++i)
    q[i] = static_cast<scalar_t>(0.01f * static_cast<float>((i % 13) + 1));
  for (int64_t i = 0; i < static_cast<int64_t>(k8.size()); ++i)
    k8[i] = static_cast<scalar_t>(0.02f * static_cast<float>((i % 11) + 1));
  for (int64_t i = 0; i < static_cast<int64_t>(k4.size()); ++i)
    k4[i] = static_cast<scalar_t>(0.03f * static_cast<float>((i % 7) + 1));
  for (int64_t i = 0; i < static_cast<int64_t>(ktail.size()); ++i)
    ktail[i] = static_cast<scalar_t>(0.025f * static_cast<float>((i % 5) + 1));
  for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i)
    v[i] = static_cast<scalar_t>(0.015f * static_cast<float>((i % 9) + 1));
  for (int64_t i = 0; i < static_cast<int64_t>(p_hat.size()); ++i)
    p_hat[i] = 0.005f * static_cast<float>((i % 5) + 1);

  auto qkt_8x8 = [&]() {
    MK::qkt_8x8(q.data(), E, k8.data(), E, E, 1.0f, scores.data());
    mk_bench_barrier(scores.data());
  };
  const double qkt_8x8_sec = time_microkernel_loop(qkt_8x8, warmup, iterations);
  const double qkt_8x8_checksum = mk_checksum_fp32_buffer(scores.data(), 8 * 8);

  auto qkt_8x4 = [&]() {
    MK::qkt_8x4(q.data(), E, k4.data(), E, E, 1.0f, scores.data(), 8);
    mk_bench_barrier(scores.data());
  };
  const double qkt_8x4_sec = time_microkernel_loop(qkt_8x4, warmup, iterations);
  const double qkt_8x4_checksum = mk_checksum_fp32_buffer(scores.data(), 8 * 8);

  std::fill(out.begin(), out.end(), 0.0f);
  auto pv_8x8 = [&]() {
    MK::pv_8x8(p_hat.data(), Sk, v.data(), 8, Sk, out.data(), 8);
    mk_bench_barrier(out.data());
  };
  const double pv_8x8_sec = time_microkernel_loop(pv_8x8, warmup, iterations);
  const double pv_8x8_checksum = mk_checksum_fp32_buffer(out.data(), 8 * 8);

  // ── qkt_tail (Lq=5, Sk=3) ──
  // 取 ktail 的前 3 行作为 K（行步长仍为 E），输出写到 scores（行步长 8）。
  // 与 5x3 的子矩阵语义一致；scratch 仍使用 8x8 buffer，超出 5x3 的位置
  // 由 micro-kernel 内部决定，被忽略不影响计时。
  auto qkt_tail = [&]() {
    MK::qkt_tail(q.data(), E, ktail.data(), E, E, 1.0f,
                 scores.data(), 8, 5, 3);
    mk_bench_barrier(scores.data());
  };
  const double qkt_tail_sec =
      time_microkernel_loop(qkt_tail, warmup, iterations);
  const double qkt_tail_checksum = mk_checksum_fp32_buffer(scores.data(), 8 * 8);

  // ── pv_tail (Lq=5, Ev=3) ──
  // V[Sk][8] 的前 3 列被读到，O 的写入限于 5x3 子块。
  std::fill(out.begin(), out.end(), 0.0f);
  auto pv_tail = [&]() {
    MK::pv_tail(p_hat.data(), Sk, v.data(), 8, Sk, out.data(), 8, 5, 3);
    mk_bench_barrier(out.data());
  };
  const double pv_tail_sec = time_microkernel_loop(pv_tail, warmup, iterations);
  const double pv_tail_checksum = mk_checksum_fp32_buffer(out.data(), 8 * 8);

  // GEMM FLOPs：2 * M * N * K
  const double qkt_8x8_flops = 2.0 * 8.0 * 8.0 *
                               static_cast<double>(E) *
                               static_cast<double>(iterations);
  const double qkt_8x4_flops = 2.0 * 8.0 * 4.0 *
                               static_cast<double>(E) *
                               static_cast<double>(iterations);
  const double pv_8x8_flops = 2.0 * 8.0 * 8.0 *
                              static_cast<double>(Sk) *
                              static_cast<double>(iterations);
  const double qkt_tail_flops = 2.0 * 5.0 * 3.0 *
                                static_cast<double>(E) *
                                static_cast<double>(iterations);
  const double pv_tail_flops = 2.0 * 5.0 * 3.0 *
                               static_cast<double>(Sk) *
                               static_cast<double>(iterations);

  std::map<std::string, double> result;
  result["E"] = static_cast<double>(E);
  result["Sk"] = static_cast<double>(Sk);
  result["iterations"] = static_cast<double>(iterations);
  result["warmup"] = static_cast<double>(warmup);
  result["direct_microkernel"] = 1.0;
  result["has_neon"] = static_cast<double>(FUSED_CPP_SDPA_CACHE_HAS_NEON);
  result["has_bf16"] = static_cast<double>(FUSED_CPP_SDPA_CACHE_HAS_BF16);
  result["has_bfmmla"] = static_cast<double>(FUSED_CPP_SDPA_CACHE_HAS_BFMMLA);
  result["dtype_is_bf16"] =
      static_cast<double>(std::is_same_v<scalar_t, at::BFloat16>);

  result["qkt_8x8_seconds"] = qkt_8x8_sec;
  result["qkt_8x8_us"] = qkt_8x8_sec * 1.0e6 /
                          static_cast<double>(iterations);
  result["qkt_8x8_gflops"] = qkt_8x8_flops / qkt_8x8_sec / 1.0e9;
  result["qkt_8x8_checksum"] = qkt_8x8_checksum;

  if constexpr (std::is_same_v<scalar_t, at::BFloat16> &&
                mk_has_qkt_kcol_v<MK>) {
    std::vector<at::BFloat16> k_col(8 * E);
    pack_k_8rows_to_col_bf16(k8.data(), E, E, k_col.data());
    auto qkt_8x8_kcol = [&]() {
      MK::qkt_8x8_kcol(q.data(), E, k_col.data(), E, 1.0f, scores.data());
      mk_bench_barrier(scores.data());
    };
    const double qkt_8x8_kcol_sec =
        time_microkernel_loop(qkt_8x8_kcol, warmup, iterations);
    const double qkt_8x8_kcol_checksum =
        mk_checksum_fp32_buffer(scores.data(), 8 * 8);
    result["qkt_8x8_kcol_seconds"] = qkt_8x8_kcol_sec;
    result["qkt_8x8_kcol_us"] =
        qkt_8x8_kcol_sec * 1.0e6 / static_cast<double>(iterations);
    result["qkt_8x8_kcol_gflops"] =
        qkt_8x8_flops / qkt_8x8_kcol_sec / 1.0e9;
    result["qkt_8x8_kcol_checksum"] = qkt_8x8_kcol_checksum;
  }

  result["qkt_8x4_seconds"] = qkt_8x4_sec;
  result["qkt_8x4_us"] = qkt_8x4_sec * 1.0e6 /
                          static_cast<double>(iterations);
  result["qkt_8x4_gflops"] = qkt_8x4_flops / qkt_8x4_sec / 1.0e9;
  result["qkt_8x4_checksum"] = qkt_8x4_checksum;

  result["pv_8x8_seconds"] = pv_8x8_sec;
  result["pv_8x8_us"] = pv_8x8_sec * 1.0e6 /
                         static_cast<double>(iterations);
  result["pv_8x8_gflops"] = pv_8x8_flops / pv_8x8_sec / 1.0e9;
  result["pv_8x8_checksum"] = pv_8x8_checksum;

  if constexpr (std::is_same_v<scalar_t, at::BFloat16> &&
                mk_has_pv_pbf16_v<MK>) {
    std::vector<at::BFloat16> p_hat_bf16(8 * Sk);
    for (int64_t i = 0; i < static_cast<int64_t>(p_hat_bf16.size()); ++i) {
      p_hat_bf16[i] = static_cast<at::BFloat16>(p_hat[i]);
    }
    std::fill(out.begin(), out.end(), 0.0f);
    auto pv_8x8_pbf16 = [&]() {
      MK::pv_8x8_pbf16(
          p_hat_bf16.data(), Sk, v.data(), 8, Sk, out.data(), 8);
      mk_bench_barrier(out.data());
    };
    const double pv_8x8_pbf16_sec =
        time_microkernel_loop(pv_8x8_pbf16, warmup, iterations);
    const double pv_8x8_pbf16_checksum =
        mk_checksum_fp32_buffer(out.data(), 8 * 8);
    result["pv_8x8_pbf16_seconds"] = pv_8x8_pbf16_sec;
    result["pv_8x8_pbf16_us"] =
        pv_8x8_pbf16_sec * 1.0e6 / static_cast<double>(iterations);
    result["pv_8x8_pbf16_gflops"] =
        pv_8x8_flops / pv_8x8_pbf16_sec / 1.0e9;
    result["pv_8x8_pbf16_checksum"] = pv_8x8_pbf16_checksum;
  }

  result["qkt_tail_seconds"] = qkt_tail_sec;
  result["qkt_tail_us"] = qkt_tail_sec * 1.0e6 /
                          static_cast<double>(iterations);
  result["qkt_tail_gflops"] = qkt_tail_flops / qkt_tail_sec / 1.0e9;
  result["qkt_tail_checksum"] = qkt_tail_checksum;

  result["pv_tail_seconds"] = pv_tail_sec;
  result["pv_tail_us"] = pv_tail_sec * 1.0e6 /
                         static_cast<double>(iterations);
  result["pv_tail_gflops"] = pv_tail_flops / pv_tail_sec / 1.0e9;
  result["pv_tail_checksum"] = pv_tail_checksum;

  return result;
}

}  // namespace fused_cpp::sdpa_microkernels
