// ── L3-Resident SDPA with Pre-Packed V ───────────────────────────────────
//
// 本文件实现 `flash2_neon_l3kv_packv` SDPA 变体（以及每个 enabled MK
// trait 对应的 `_baseline / _scalar / _pquad` 版本）。
//
// 与 `flash2_neon_l3kv` 的唯一差别：在 SDPA 入口处先把 V tensor 从
// 物理布局 `[B, N, S, Ev]` 重排成 `[B, N, Ev/8, S, 8]` 写入一份新
// buffer，再喂给 `process_q_tile_lc<..., kPackedV=true>`。
//
// 为什么要 pack？──────────────────────────────────────────────────────
//
// PV microkernel 的内层沿 Sk 维扫 V，每行只用 8 个元素。原始 layout 下
// 跨 Sk 行的 V 地址 stride = `Ev`（典型 128 fp32 = 512 字节 = 8 cache
// line）——HW stride prefetcher 能跨大 stride 跟踪是有条件的：多 PC
// stream 竞争 tracker 槽、置信度阈值随 stride 增大而提高、跨 4KiB page
// 强制 reset，cold-start 与小 Sc tile 下都可能掉链子。
//
// 把 V 重排成 `[ev_block, S, 8]` 后，同一个 8-col 输出块所有 Sk 行的 V
// 在内存里**完全连续**，行间跨度 = 32 字节（半 cache line），prefetch
// 由「下一行已经在同一/邻接 line 内」自然 cover，与 P̂ pack 一样把 LSU
// 端的访问模式变得规整。
//
// dtype 不变：bf16 → bf16、fp32 → fp32（pack 不 widen），减少访存数据量。
// pack 阶段多线程并行 over (b, n, ev_block)，与主计算阶段并列、不嵌套。
//
// 限制：`Ev % 8 == 0`（生产场景 head_dim_v ∈ {64, 128, 256} 全满足；
// 不满足时 TORCH_CHECK 报错，调用方应改走 `flash2_neon_l3kv`）。

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "sdpa_common.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sdpa_microkernels/neon_cache_config.h"
#include "sdpa_tile_sizes.h"
#include "sdpa_microkernels/mk_traits.h"
#include "sdpa_microkernels/all_impls.h"
#include "sdpa_flash2_neon_l3kv_impl.h"

namespace {

using ::fused_cpp::sdpa_tile_sizes::TileSizes;
using ::fused_cpp::sdpa_tile_sizes::effective_cache_bytes;
using ::fused_cpp::sdpa_tile_sizes::compute_tile_sizes_l3kv;
using ::fused_cpp::sdpa_flash2_neon_l3kv_impl::run_path_collapse3;
using ::fused_cpp::sdpa_flash2_neon_l3kv_impl::run_path_taskloop;

// ──────────────────────────────────────────────────────────────────────
// V pack：[B, N, S, Ev] → [B, N, Ev/8, S, 8]
//
// 工作划分：collapse(3) over (b, n, ev_block)，每个工作单元串行扫
// `S` 行做一个 `8 * sizeof(scalar_t)` 字节的 memcpy。
//
// * collapse(3) 不 collapse(4)：内层 `s` 循环是顺序写，留给单线程能让
//   写入流连续、store buffer 友好。
// * 每个 (b, n, ev_block) 块写出 `S * 8 * sizeof(scalar_t)` 字节
//   （bf16 S=2048 时 32 KiB，fp32 时 64 KiB），远 ≫ 64 字节 cache line，
//   不同线程绝不会写同一 line —— 无 false sharing。
// * memcpy size = 8 * sizeof(scalar_t) 是 16/32 字节常量，编译器内联
//   成单条 `ldr q + str q`（fp32）或 `ldr d + str d`（bf16）。
// ──────────────────────────────────────────────────────────────────────

template <typename scalar_t>
void pack_v_to_evblock8(
    const scalar_t* v_src,
    scalar_t* v_dst,
    int64_t B, int64_t N, int64_t S, int64_t Ev) {
  const int64_t Eb = Ev / 8;
  const int64_t bn_stride_src = N * S * Ev;
  const int64_t n_stride_src = S * Ev;
  const int64_t bn_stride_dst = N * Eb * S * 8;
  const int64_t n_stride_dst = Eb * S * 8;
  const int64_t evblock_stride_dst = S * 8;

#ifdef _OPENMP
  #pragma omp parallel for collapse(3) schedule(static)
#endif
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t eb = 0; eb < Eb; ++eb) {
        const scalar_t* src = v_src + b * bn_stride_src
                                    + n * n_stride_src
                                    + eb * 8;
        scalar_t* dst = v_dst + b * bn_stride_dst
                              + n * n_stride_dst
                              + eb * evblock_stride_dst;
        for (int64_t s = 0; s < S; ++s) {
          std::memcpy(dst + s * 8,
                      src + s * Ev,
                      8 * sizeof(scalar_t));
        }
      }
    }
  }
}

// ──────────────────────────────────────────────────────────────────────
// 顶层模板：pack V → 调 run_path_*<..., kPackedV=true>。
// ──────────────────────────────────────────────────────────────────────

template <class MK, typename scalar_t>
inline void sdpa_flash2_neon_l3kv_packv_with_mk_tmpl(
    const scalar_t* q_ptr,
    const scalar_t* k_ptr,
    const scalar_t* v_ptr,
    const SdpaParams& p) {
  TORCH_CHECK(p.Ev % 8 == 0,
              "flash2_neon_l3kv_packv requires Ev % 8 == 0, got Ev=", p.Ev);

  // ── 分配 packed V buffer：B*N*Ev/8*S*8 个元素（= B*N*S*Ev，与原 V 同尺寸）──
  const auto torch_dtype = (p.dtype == SdpaDtype::kBFloat16)
                               ? at::kBFloat16 : at::kFloat;
  at::Tensor v_packed = at::empty(
      {p.B, p.N, p.Ev / 8, p.S, 8},
      at::TensorOptions().dtype(torch_dtype));
  scalar_t* v_packed_ptr = static_cast<scalar_t*>(v_packed.data_ptr());

  // ── 多线程 pack（独立 omp parallel for；完成后再进主并行区）──
  pack_v_to_evblock8<scalar_t>(v_ptr, v_packed_ptr, p.B, p.N, p.S, p.Ev);

  // ── strides 计算 ──
  // 非 V 维度的 strides 与原始 l3kv 一致；V 走 packed layout：
  //   v_stride_b = N * (Ev/8) * S * 8     (= N*S*Ev)
  //   v_stride_n = (Ev/8) * S * 8         (= S*Ev)
  //   v_stride_s = 8                      ← 微内核行间步长 = 8
  //   v_evblock_stride = S * 8            ← packed 路径下跨 ev_block
  const int64_t Eb = p.Ev / 8;
  const int64_t v_evblock_stride = p.S * 8;
  const int64_t v_packed_stride_n = Eb * v_evblock_stride;
  const int64_t v_packed_stride_b = p.N * v_packed_stride_n;
  const int64_t v_packed_stride_s = 8;

  const int64_t q_stride_b = p.N * p.L * p.E;
  const int64_t q_stride_n = p.L * p.E;
  const int64_t q_stride_l = p.E;
  const int64_t k_stride_b = p.N * p.S * p.E;
  const int64_t k_stride_n = p.S * p.E;
  const int64_t k_stride_s = p.E;
  const int64_t m_stride_b = p.N * p.L * p.S;
  const int64_t m_stride_n = p.L * p.S;
  const int64_t m_stride_l = p.S;
  const int64_t o_stride_b = p.N * p.L * p.Ev;
  const int64_t o_stride_n = p.L * p.Ev;
  const int64_t o_stride_l = p.Ev;

  TileSizes ts = compute_tile_sizes_l3kv(p.B, p.N, p.S, p.L, p.E, p.Ev,
                                         sizeof(scalar_t));

  // path A/B 选择跟原 l3kv 完全一致：packed buffer 跟原 V 同大小，
  // 单 head 工作集 = S * (E + Ev) * sizeof，阈值不变。
  const int64_t kv_bytes_per_bn =
      p.S * (p.E + p.Ev) * static_cast<int64_t>(sizeof(scalar_t));
  const auto& cache_bytes = effective_cache_bytes();
  const int64_t l3_budget =
      static_cast<int64_t>(cache_bytes[2] * FUSED_CPP_SDPA_L3_RATIO);
  const bool kv_fits_l3 = kv_bytes_per_bn <= l3_budget;
  const int total_threads =
#ifdef _OPENMP
      omp_get_max_threads();
#else
      1;
#endif

  static bool s_debug_groups = []() {
    const char* dbg = std::getenv("FUSED_CPP_SDPA_L3KV_DEBUG_GROUPS");
    return dbg != nullptr && std::strcmp(dbg, "1") == 0;
  }();

  if (kv_fits_l3 || total_threads == 1) {
    if (s_debug_groups) {
      std::fprintf(stderr,
          "[flash2_neon_l3kv_packv] path=A B=%lld N=%lld L=%lld S=%lld "
          "E=%lld Ev=%lld kv_bytes=%lld l3_budget=%lld threads=%d "
          "Lc_l2=%lld Sc_l2=%lld Sc_l3=%lld\n",
          (long long)p.B, (long long)p.N, (long long)p.L, (long long)p.S,
          (long long)p.E, (long long)p.Ev,
          (long long)kv_bytes_per_bn, (long long)l3_budget, total_threads,
          (long long)ts.Lc_l2, (long long)ts.Sc_l2, (long long)ts.Sc_l3);
    }
    // ── 按 (kHasMask, kCausal) 编译期组合分发 packv 模板实例 ──
    if (p.mask_ptr != nullptr) {
      if (p.is_causal) {
        run_path_collapse3<MK, scalar_t, /*kPackedV=*/true,
                           /*kHasMask=*/true, /*kCausal=*/true>(
            q_ptr, k_ptr, v_packed_ptr, p, ts,
            q_stride_b, q_stride_n, q_stride_l,
            k_stride_b, k_stride_n, k_stride_s,
            v_packed_stride_b, v_packed_stride_n, v_packed_stride_s,
            v_evblock_stride,
            m_stride_b, m_stride_n, m_stride_l,
            o_stride_b, o_stride_n, o_stride_l);
      } else {
        run_path_collapse3<MK, scalar_t, /*kPackedV=*/true,
                           /*kHasMask=*/true, /*kCausal=*/false>(
            q_ptr, k_ptr, v_packed_ptr, p, ts,
            q_stride_b, q_stride_n, q_stride_l,
            k_stride_b, k_stride_n, k_stride_s,
            v_packed_stride_b, v_packed_stride_n, v_packed_stride_s,
            v_evblock_stride,
            m_stride_b, m_stride_n, m_stride_l,
            o_stride_b, o_stride_n, o_stride_l);
      }
    } else {
      if (p.is_causal) {
        run_path_collapse3<MK, scalar_t, /*kPackedV=*/true,
                           /*kHasMask=*/false, /*kCausal=*/true>(
            q_ptr, k_ptr, v_packed_ptr, p, ts,
            q_stride_b, q_stride_n, q_stride_l,
            k_stride_b, k_stride_n, k_stride_s,
            v_packed_stride_b, v_packed_stride_n, v_packed_stride_s,
            v_evblock_stride,
            m_stride_b, m_stride_n, m_stride_l,
            o_stride_b, o_stride_n, o_stride_l);
      } else {
        run_path_collapse3<MK, scalar_t, /*kPackedV=*/true,
                           /*kHasMask=*/false, /*kCausal=*/false>(
            q_ptr, k_ptr, v_packed_ptr, p, ts,
            q_stride_b, q_stride_n, q_stride_l,
            k_stride_b, k_stride_n, k_stride_s,
            v_packed_stride_b, v_packed_stride_n, v_packed_stride_s,
            v_evblock_stride,
            m_stride_b, m_stride_n, m_stride_l,
            o_stride_b, o_stride_n, o_stride_l);
      }
    }
  } else {
    const int max_concurrent_bn =
        std::max<int>(1, static_cast<int>(l3_budget / kv_bytes_per_bn));
    const int total_bn = static_cast<int>(p.B * p.N);
    const int num_groups = std::min<int>(
        total_bn, std::min<int>(total_threads, max_concurrent_bn));
    if (s_debug_groups) {
      std::fprintf(stderr,
          "[flash2_neon_l3kv_packv] path=B B=%lld N=%lld L=%lld S=%lld "
          "E=%lld Ev=%lld kv_bytes=%lld l3_budget=%lld threads=%d "
          "max_concurrent_bn=%d num_groups=%d Lc_l2=%lld Sc_l2=%lld\n",
          (long long)p.B, (long long)p.N, (long long)p.L, (long long)p.S,
          (long long)p.E, (long long)p.Ev,
          (long long)kv_bytes_per_bn, (long long)l3_budget, total_threads,
          max_concurrent_bn, num_groups,
          (long long)ts.Lc_l2, (long long)ts.Sc_l2);
    }
    if (p.mask_ptr != nullptr) {
      if (p.is_causal) {
        run_path_taskloop<MK, scalar_t, /*kPackedV=*/true,
                          /*kHasMask=*/true, /*kCausal=*/true>(
            q_ptr, k_ptr, v_packed_ptr, p, ts, num_groups,
            q_stride_b, q_stride_n, q_stride_l,
            k_stride_b, k_stride_n, k_stride_s,
            v_packed_stride_b, v_packed_stride_n, v_packed_stride_s,
            v_evblock_stride,
            m_stride_b, m_stride_n, m_stride_l,
            o_stride_b, o_stride_n, o_stride_l);
      } else {
        run_path_taskloop<MK, scalar_t, /*kPackedV=*/true,
                          /*kHasMask=*/true, /*kCausal=*/false>(
            q_ptr, k_ptr, v_packed_ptr, p, ts, num_groups,
            q_stride_b, q_stride_n, q_stride_l,
            k_stride_b, k_stride_n, k_stride_s,
            v_packed_stride_b, v_packed_stride_n, v_packed_stride_s,
            v_evblock_stride,
            m_stride_b, m_stride_n, m_stride_l,
            o_stride_b, o_stride_n, o_stride_l);
      }
    } else {
      if (p.is_causal) {
        run_path_taskloop<MK, scalar_t, /*kPackedV=*/true,
                          /*kHasMask=*/false, /*kCausal=*/true>(
            q_ptr, k_ptr, v_packed_ptr, p, ts, num_groups,
            q_stride_b, q_stride_n, q_stride_l,
            k_stride_b, k_stride_n, k_stride_s,
            v_packed_stride_b, v_packed_stride_n, v_packed_stride_s,
            v_evblock_stride,
            m_stride_b, m_stride_n, m_stride_l,
            o_stride_b, o_stride_n, o_stride_l);
      } else {
        run_path_taskloop<MK, scalar_t, /*kPackedV=*/true,
                          /*kHasMask=*/false, /*kCausal=*/false>(
            q_ptr, k_ptr, v_packed_ptr, p, ts, num_groups,
            q_stride_b, q_stride_n, q_stride_l,
            k_stride_b, k_stride_n, k_stride_s,
            v_packed_stride_b, v_packed_stride_n, v_packed_stride_s,
            v_evblock_stride,
            m_stride_b, m_stride_n, m_stride_l,
            o_stride_b, o_stride_n, o_stride_l);
      }
    }
  }
}

// dtype dispatch entry.
template <class MK>
inline void sdpa_flash2_neon_l3kv_packv_with_mk_impl(const SdpaParams& p) {
  if (p.dtype == SdpaDtype::kBFloat16) {
    sdpa_flash2_neon_l3kv_packv_with_mk_tmpl<MK, at::BFloat16>(
        static_cast<const at::BFloat16*>(p.q_ptr),
        static_cast<const at::BFloat16*>(p.k_ptr),
        static_cast<const at::BFloat16*>(p.v_ptr),
        p);
  } else {
    sdpa_flash2_neon_l3kv_packv_with_mk_tmpl<MK, float>(
        static_cast<const float*>(p.q_ptr),
        static_cast<const float*>(p.k_ptr),
        static_cast<const float*>(p.v_ptr),
        p);
  }
}

}  // anonymous namespace

// ── enabled MK impl 的 SDPA 入口 + 注册 ─────────────────────────────────
//
// 历史名 "flash2_neon_l3kv_packv" 绑定到 baseline。

#if FUSED_CPP_MK_ENABLE_BASELINE
namespace {
void sdpa_flash2_neon_l3kv_packv_baseline_entry(const SdpaParams& p) {
  sdpa_flash2_neon_l3kv_packv_with_mk_impl<
      ::fused_cpp::sdpa_microkernels::MK_Baseline>(p);
}
}  // anonymous namespace
REGISTER_SDPA_VERSION("flash2_neon_l3kv_packv",
                      sdpa_flash2_neon_l3kv_packv_baseline_entry);
REGISTER_SDPA_VERSION("flash2_neon_l3kv_packv_baseline",
                      sdpa_flash2_neon_l3kv_packv_baseline_entry);
#endif

#if FUSED_CPP_MK_ENABLE_SCALAR
namespace {
void sdpa_flash2_neon_l3kv_packv_scalar_entry(const SdpaParams& p) {
  sdpa_flash2_neon_l3kv_packv_with_mk_impl<
      ::fused_cpp::sdpa_microkernels::MK_Scalar>(p);
}
}  // anonymous namespace
REGISTER_SDPA_VERSION("flash2_neon_l3kv_packv_scalar",
                      sdpa_flash2_neon_l3kv_packv_scalar_entry);
#endif

#if FUSED_CPP_MK_ENABLE_PQUAD
namespace {
void sdpa_flash2_neon_l3kv_packv_pquad_entry(const SdpaParams& p) {
  sdpa_flash2_neon_l3kv_packv_with_mk_impl<
      ::fused_cpp::sdpa_microkernels::MK_PQuad>(p);
}
}  // anonymous namespace
REGISTER_SDPA_VERSION("flash2_neon_l3kv_packv_pquad",
                      sdpa_flash2_neon_l3kv_packv_pquad_entry);
#endif

#if FUSED_CPP_MK_ENABLE_QK_UBLOCK4
namespace {
void sdpa_flash2_neon_l3kv_packv_qk_ublock4_entry(const SdpaParams& p) {
  sdpa_flash2_neon_l3kv_packv_with_mk_impl<
      ::fused_cpp::sdpa_microkernels::MK_QkUblock4>(p);
}
}  // anonymous namespace
REGISTER_SDPA_VERSION("flash2_neon_l3kv_packv_qk_ublock4",
                      sdpa_flash2_neon_l3kv_packv_qk_ublock4_entry);
#endif
