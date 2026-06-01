// ── L3-Resident SDPA with Pre-Packed Q + K + V ──────────────────────────
//
// 本文件实现 `flash2_neon_l3kv_packqkv` SDPA 变体。与 `flash2_neon_l3kv_packv`
// 的区别：
//
//   * V 仍然在 SDPA 入口多线程 pre-pack 成 [B, N, Ev/8, S, 8]（与 packv 一致）；
//   * **K 在 SDPA 入口多线程 pre-pack** 成 [B, N, S/8, E_main/4, 32 u16]，
//     与 packqk_seq4 microkernel 的 K_seq 入参对齐；
//   * Q 在 process_q_tile_lc_packqkv 入口处一次性 pack 当前 q-tile 的所有
//     完整 8-row 子块到 thread-local q_seq_buf；q-tile 内所有 (s_l3, s_l2,
//     qi_inner, s_off) 复用 packed Q，**不重复 pack**。
//
// QKᵀ inner 走 gemm_qkt_microkernel_8x8_bf16_packqk_seq4_bmajor_inner：4 条
// 独立 vld1q_u16 + B-major BFMMLA（microkernel benchmark 上 +23% vs baseline）。
// PV 走 packed V 路径。
//
// 限制：
//   * **bf16-only**：fp32 没 BFMMLA 加速，pack K/Q 纯亏。fp32 输入直接 delegate
//     到 `flash2_neon_l3kv_packv`。
//   * **`S % 8 == 0` 且 `Ev % 8 == 0`**：partial S block 处理复杂，且生产场景
//     S 全是 2 的幂。不满足时建议改用 `flash2_neon_l3kv_packv`。
//   * `E % 4 != 0` 时 partial e_block 不 pack，inner 标量 tail 自动从原始 K
//     指针读 [E_main, E)；与 baseline bit-for-bit 等价。
//
// Path A（KV 装 L3）才启用 K pack；Path B（KV 不装 L3）会引入 3× DRAM 带宽
// （pack 读原 K + pack 写 K_packed + compute 读 K_packed），可能负收益——
// 退化为 packv 行为，跳过 K pack。

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
#include "sdpa_microkernels/neon_cache_microkernels.h"
#include "sdpa_flash2_neon_l3kv_impl.h"

namespace {

using ::fused_cpp::sdpa_tile_sizes::TileSizes;
using ::fused_cpp::sdpa_tile_sizes::effective_cache_bytes;
using ::fused_cpp::sdpa_tile_sizes::compute_tile_sizes_l3kv;
using ::fused_cpp::sdpa_flash2_neon_l3kv_impl::run_path_collapse3_packqkv;

// ──────────────────────────────────────────────────────────────────────
// V pack helper（与 flash2_neon_l3kv_packv.cpp 中实现一致；这里再放一份
// 以避免跨 .cpp 链接依赖。维护提醒：两份实现必须保持同步）。
// ──────────────────────────────────────────────────────────────────────

template <typename scalar_t>
void pack_v_to_evblock8_local(
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
// 顶层入口：packqkv (bf16 path)
// ──────────────────────────────────────────────────────────────────────

template <bool kPbf16PV>
void sdpa_flash2_neon_l3kv_packqkv_bf16_impl_tmpl(
    const SdpaParams& p,
    const char* version_name) {
  TORCH_CHECK(p.dtype == SdpaDtype::kBFloat16,
              "packqkv bf16 path called with non-bf16 dtype");
  TORCH_CHECK(p.S % 8 == 0,
              version_name, " requires S % 8 == 0, got S=", p.S,
              "; please use flash2_neon_l3kv_packv for non-aligned S.");
  TORCH_CHECK(p.Ev % 8 == 0,
              version_name, " requires Ev % 8 == 0, got Ev=", p.Ev);
  const bool profile_on = ::fused_cpp::sdpa_profile::enabled();
  if (profile_on) {
    ::fused_cpp::sdpa_profile::reset();
  }
  const uint64_t profile_total_t0 =
      profile_on ? ::fused_cpp::sdpa_profile::now_ns() : 0;

  const auto* q_ptr = static_cast<const at::BFloat16*>(p.q_ptr);
  const auto* k_ptr = static_cast<const at::BFloat16*>(p.k_ptr);
  const auto* v_ptr = static_cast<const at::BFloat16*>(p.v_ptr);

  // ── 分配 packed V buffer ──
  at::Tensor v_packed;
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kVAlloc);
    v_packed = at::empty(
        {p.B, p.N, p.Ev / 8, p.S, 8},
        at::TensorOptions().dtype(at::kBFloat16));
  }
  auto* v_packed_ptr = static_cast<at::BFloat16*>(v_packed.data_ptr());
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kVPack);
    pack_v_to_evblock8_local<at::BFloat16>(
        v_ptr, v_packed_ptr, p.B, p.N, p.S, p.Ev);
  }

  // ── strides ──
  const int64_t Eb = p.Ev / 8;
  const int64_t v_evblock_stride = p.S * 8;
  const int64_t v_packed_stride_n = Eb * v_evblock_stride;
  const int64_t v_packed_stride_b = p.N * v_packed_stride_n;
  const int64_t v_packed_stride_s = 8;

  const int64_t q_stride_b = p.N * p.L * p.E;
  const int64_t q_stride_n = p.L * p.E;
  const int64_t q_stride_l = p.E;
  const int64_t k_orig_stride_b = p.N * p.S * p.E;
  const int64_t k_orig_stride_n = p.S * p.E;
  const int64_t k_orig_stride_s = p.E;
  const int64_t m_stride_b = p.N * p.L * p.S;
  const int64_t m_stride_n = p.L * p.S;
  const int64_t m_stride_l = p.S;
  const int64_t o_stride_b = p.N * p.L * p.Ev;
  const int64_t o_stride_n = p.L * p.Ev;
  const int64_t o_stride_l = p.Ev;

  TileSizes ts = compute_tile_sizes_l3kv(p.B, p.N, p.S, p.L, p.E, p.Ev,
                                         sizeof(at::BFloat16));

  const int64_t kv_bytes_per_bn =
      p.S * (p.E + p.Ev) * static_cast<int64_t>(sizeof(at::BFloat16));
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

  // ── Path B：不 pack K，退化为 flash2_neon_l3kv_packv 行为 ──
  // 通过 sdpa_dispatch 路由，避免重复实现 path B 全套逻辑。注意：
  // V 已经 pack 过了，但 packv 入口会再 pack 一次（白做一份 V copy）。
  // 这是工程取舍——packv 入口的多线程 pack overhead 与一次 SDPA 计算
  // 相比是小项；为了避免在本文件再写一份完整的 path B taskloop 调用，
  // 接受这点冗余。如果实测显著影响 path B 性能，可以单独优化。
  if (!kv_fits_l3 && total_threads > 1) {
    sdpa_dispatch("flash2_neon_l3kv_packv_pquad", p);
    return;
  }

  // ── Path A：分配 packed K buffer 并 pack ──
  const int64_t E_main = p.E & ~int64_t{3};
  const int64_t e_blocks = E_main / 4;
  const int64_t kblock_u16 = e_blocks * 32;        // 单个 8-row 子块容量
  const int64_t S_blocks = p.S / 8;
  const int64_t k_packed_total_u16 = p.B * p.N * S_blocks * kblock_u16;

  // 用 BFloat16 dtype 分配（u16 兼容），元素数按 u16 算
  // 注意：at::empty 没有 kU16，用 BF16 alias，因为我们只关心字节数与对齐
  at::Tensor k_packed;
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kKAlloc);
    k_packed = at::empty(
        {k_packed_total_u16},
        at::TensorOptions().dtype(at::kBFloat16));
  }
  auto* k_packed_ptr = reinterpret_cast<uint16_t*>(k_packed.data_ptr());

  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kKPack);
    ::fused_cpp::sdpa_microkernels::pack_k_to_seq8<at::BFloat16>(
        k_ptr, k_packed_ptr, p.B, p.N, p.S, p.E);
  }

  const int64_t k_sblock_stride = kblock_u16;            // u16 单位
  const int64_t k_packed_stride_n = S_blocks * kblock_u16;
  const int64_t k_packed_stride_b = p.N * k_packed_stride_n;

  // ── 4 路 (kHasMask, kCausal) 分发 ──
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kMain);
    if (p.mask_ptr != nullptr) {
      if (p.is_causal) {
        run_path_collapse3_packqkv<true, true, kPbf16PV>(
            q_ptr, k_packed_ptr, k_ptr, v_packed_ptr, p, ts,
            q_stride_b, q_stride_n, q_stride_l,
            k_orig_stride_b, k_orig_stride_n, k_orig_stride_s,
            k_packed_stride_b, k_packed_stride_n, k_sblock_stride,
            v_packed_stride_b, v_packed_stride_n, v_packed_stride_s,
            v_evblock_stride,
            m_stride_b, m_stride_n, m_stride_l,
            o_stride_b, o_stride_n, o_stride_l);
      } else {
        run_path_collapse3_packqkv<true, false, kPbf16PV>(
            q_ptr, k_packed_ptr, k_ptr, v_packed_ptr, p, ts,
            q_stride_b, q_stride_n, q_stride_l,
            k_orig_stride_b, k_orig_stride_n, k_orig_stride_s,
            k_packed_stride_b, k_packed_stride_n, k_sblock_stride,
            v_packed_stride_b, v_packed_stride_n, v_packed_stride_s,
            v_evblock_stride,
            m_stride_b, m_stride_n, m_stride_l,
            o_stride_b, o_stride_n, o_stride_l);
      }
    } else {
      if (p.is_causal) {
        run_path_collapse3_packqkv<false, true, kPbf16PV>(
            q_ptr, k_packed_ptr, k_ptr, v_packed_ptr, p, ts,
            q_stride_b, q_stride_n, q_stride_l,
            k_orig_stride_b, k_orig_stride_n, k_orig_stride_s,
            k_packed_stride_b, k_packed_stride_n, k_sblock_stride,
            v_packed_stride_b, v_packed_stride_n, v_packed_stride_s,
            v_evblock_stride,
            m_stride_b, m_stride_n, m_stride_l,
            o_stride_b, o_stride_n, o_stride_l);
      } else {
        run_path_collapse3_packqkv<false, false, kPbf16PV>(
            q_ptr, k_packed_ptr, k_ptr, v_packed_ptr, p, ts,
            q_stride_b, q_stride_n, q_stride_l,
            k_orig_stride_b, k_orig_stride_n, k_orig_stride_s,
            k_packed_stride_b, k_packed_stride_n, k_sblock_stride,
            v_packed_stride_b, v_packed_stride_n, v_packed_stride_s,
            v_evblock_stride,
            m_stride_b, m_stride_n, m_stride_l,
            o_stride_b, o_stride_n, o_stride_l);
      }
    }
  }
  if (profile_on) {
    ::fused_cpp::sdpa_profile::add(
        ::fused_cpp::sdpa_profile::Slot::kTotal,
        ::fused_cpp::sdpa_profile::now_ns() - profile_total_t0);
    ::fused_cpp::sdpa_profile::print_summary(
        version_name,
        kPbf16PV ? "qk_packqk_seq4_bmajor_pv_pbf16_prepacked"
                 : "qk_packqk_seq4_bmajor_pv_pquad",
        p,
        "A");
  }
}

void sdpa_flash2_neon_l3kv_packqkv_bf16_impl(const SdpaParams& p) {
  sdpa_flash2_neon_l3kv_packqkv_bf16_impl_tmpl<false>(
      p, "flash2_neon_l3kv_packqkv");
}

void sdpa_flash2_neon_l3kv_packqkv_pbf16pv_bf16_impl(const SdpaParams& p) {
  sdpa_flash2_neon_l3kv_packqkv_bf16_impl_tmpl<true>(
      p, "flash2_neon_l3kv_packqkv_pbf16pv");
}

void sdpa_flash2_neon_l3kv_packqkv_entry(const SdpaParams& p) {
  // fp32 输入 delegate 到 flash2_neon_l3kv_packv_pquad（fp32 PV pquad，5/29
  // 方案 A 调度优化后远程 87% peak）。bf16 走本文件的 packqkv_bf16_impl，
  // PV 已切到 MK_QkPackqkSeq4BmajorPvPquad::pv_8x8（bf16 PV pquad）。
  if (p.dtype != SdpaDtype::kBFloat16) {
    sdpa_dispatch("flash2_neon_l3kv_packv_pquad", p);
    return;
  }
  sdpa_flash2_neon_l3kv_packqkv_bf16_impl(p);
}

void sdpa_flash2_neon_l3kv_packqkv_pbf16pv_entry(const SdpaParams& p) {
  if (p.dtype != SdpaDtype::kBFloat16) {
    sdpa_dispatch("flash2_neon_l3kv_packv_pquad", p);
    return;
  }
  sdpa_flash2_neon_l3kv_packqkv_pbf16pv_bf16_impl(p);
}

}  // anonymous namespace

REGISTER_SDPA_VERSION("flash2_neon_l3kv_packqkv",
                      sdpa_flash2_neon_l3kv_packqkv_entry);
REGISTER_SDPA_VERSION("flash2_neon_l3kv_packqkv_pbf16pv",
                      sdpa_flash2_neon_l3kv_packqkv_pbf16pv_entry);
