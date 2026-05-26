// ── L3-Resident K/V FlashAttention-2 SDPA Kernel ────────────────────────────
//
// 本文件实现一个新的 cache-aware SDPA 内核，注册名为 `flash2_neon_l3kv`
// （以及每个 enabled MK trait 对应的 `flash2_neon_l3kv_<MK::kName>`）。
// 与 `sdpa_flash2_neon_cache.cpp` 的差异（即此文件存在的全部理由）：
//
//   1. **L3 多线程共享 K/V**：当 `S * (E + Ev) * sizeof_elt > l3_budget`
//      时，把线程按 head 分组：组内的 worker 协同处理同一个 (b, n) 上不同
//      的 q_tile，从而让**同一份 K/V** 对组内所有 worker 都驻留在 L3。
//      通过 `omp parallel { omp single { ... omp taskloop ... } }` 实现，
//      不依赖嵌套 OMP 也不依赖 atomic 自旋。当 K/V 整体能放进 L3 时退化
//      为 `omp parallel for collapse(3)`，与现有 flash2_neon_cache 拓扑
//      一致。
//
//   2. **`Lc_l2` 嵌套**：把外层 q_tile 步长从 8 改成 `Lc_l2`（≤64，按 Ev
//      自适应收紧），内部以 8 行为单位串行处理 `Lc_l2/8` 个 inner 组。
//      关键点：每个 L2 KV tile 加载 **一次** 后，`Lc_l2/8` 个 inner 组
//      复用同一份 K/V，把 K/V 从 L3→L2 的搬运量降到约 1/(Lc_l2/8)。
//
//   3. **L1 软件预取（PLDL1KEEP）**：在外层 KV tile 边界仍然发
//      PLDL2KEEP（与 flash2_neon_cache 一致），但额外在 inner 组循环里
//      把当前 inner=k 处理时为 inner=k+2 的 V 行发 PLDL1KEEP。
//      Cortex-A/Neoverse 的 PLDL1KEEP 需要 ~80–200 cycle 的提前量，因此
//      预取目标必须超前两步。
//
// 数值等价性：本内核与 `flash2_neon_cache` / `flash2_neon` 在每行上做的
// online-softmax 完全一致（每行各自维护 `(running_max, running_sum,
// O_acc[i, :])` 三元组），仅是把多行的更新批量化到同一个 K/V tile 上，
// 因此 **逐行结果与现有内核在 fp32 下严格相等**（bf16 下在 vbfdotq /
// BFMMLA 求和顺序上有 ULP-级差异，与 flash2_neon_cache 同等）。
//
// 实现说明：`process_q_tile_lc` / `run_path_collapse3` / `run_path_taskloop`
// 三个模板已经被抽到 `sdpa_flash2_neon_l3kv_impl.h`，与 `_packv` 变体
// 共用。本文件只保留顶层 dtype dispatch + SDPA 名字注册。
//
// 依赖：仅 <arm_neon.h>（编译器自带）；构建走 setup.py 的
// glob("csrc/**/*.cpp")，不需修改 setup.py。

#include <torch/extension.h>
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
// 顶层模板：dtype-specialized 实现入口（kPackedV=false 走原始 V）。
// ──────────────────────────────────────────────────────────────────────

template <class MK, typename scalar_t>
inline void sdpa_flash2_neon_l3kv_with_mk_tmpl(
    const scalar_t* q_ptr,
    const scalar_t* k_ptr,
    const scalar_t* v_ptr,
    const SdpaParams& p) {
  const int64_t q_stride_b = p.N * p.L * p.E;
  const int64_t q_stride_n = p.L * p.E;
  const int64_t q_stride_l = p.E;
  const int64_t k_stride_b = p.N * p.S * p.E;
  const int64_t k_stride_n = p.S * p.E;
  const int64_t k_stride_s = p.E;
  const int64_t v_stride_b = p.N * p.S * p.Ev;
  const int64_t v_stride_n = p.S * p.Ev;
  const int64_t v_stride_s = p.Ev;
  const int64_t m_stride_b = p.N * p.L * p.S;
  const int64_t m_stride_n = p.L * p.S;
  const int64_t m_stride_l = p.S;
  const int64_t o_stride_b = p.N * p.L * p.Ev;
  const int64_t o_stride_n = p.L * p.Ev;
  const int64_t o_stride_l = p.Ev;

  TileSizes ts = compute_tile_sizes_l3kv(p.B, p.N, p.S, p.L, p.E, p.Ev,
                                         sizeof(scalar_t));

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

  // 调试钩子：FUSED_CPP_SDPA_L3KV_DEBUG_GROUPS=1 时打印决策信息。
  static bool s_debug_groups = []() {
    const char* dbg = std::getenv("FUSED_CPP_SDPA_L3KV_DEBUG_GROUPS");
    return dbg != nullptr && std::strcmp(dbg, "1") == 0;
  }();

  if (kv_fits_l3 || total_threads == 1) {
    if (s_debug_groups) {
      std::fprintf(stderr,
          "[flash2_neon_l3kv] path=A (collapse3) B=%lld N=%lld L=%lld "
          "S=%lld E=%lld Ev=%lld kv_bytes=%lld l3_budget=%lld threads=%d "
          "Lc_l2=%lld Sc_l2=%lld Sc_l3=%lld\n",
          (long long)p.B, (long long)p.N, (long long)p.L, (long long)p.S,
          (long long)p.E, (long long)p.Ev,
          (long long)kv_bytes_per_bn, (long long)l3_budget, total_threads,
          (long long)ts.Lc_l2, (long long)ts.Sc_l2, (long long)ts.Sc_l3);
    }
    // ── 按 (kHasMask, kCausal) 编译期组合分发模板实例 ──
    // 4 种组合各自展成一份 path-A 调用。模板参数透传到 process_q_tile_lc,
    // 由 if constexpr 把 mask add / causal mask / causal_lim 初始化整段
    // 从不需要的实例里彻底去掉。运行期没有 mask/causal 分支。
    if (p.mask_ptr != nullptr) {
      if (p.is_causal) {
        run_path_collapse3<MK, scalar_t, /*kPackedV=*/false,
                           /*kHasMask=*/true, /*kCausal=*/true>(
            q_ptr, k_ptr, v_ptr, p, ts,
            q_stride_b, q_stride_n, q_stride_l,
            k_stride_b, k_stride_n, k_stride_s,
            v_stride_b, v_stride_n, v_stride_s,
            /*v_evblock_stride=*/0,
            m_stride_b, m_stride_n, m_stride_l,
            o_stride_b, o_stride_n, o_stride_l);
      } else {
        run_path_collapse3<MK, scalar_t, /*kPackedV=*/false,
                           /*kHasMask=*/true, /*kCausal=*/false>(
            q_ptr, k_ptr, v_ptr, p, ts,
            q_stride_b, q_stride_n, q_stride_l,
            k_stride_b, k_stride_n, k_stride_s,
            v_stride_b, v_stride_n, v_stride_s,
            /*v_evblock_stride=*/0,
            m_stride_b, m_stride_n, m_stride_l,
            o_stride_b, o_stride_n, o_stride_l);
      }
    } else {
      if (p.is_causal) {
        run_path_collapse3<MK, scalar_t, /*kPackedV=*/false,
                           /*kHasMask=*/false, /*kCausal=*/true>(
            q_ptr, k_ptr, v_ptr, p, ts,
            q_stride_b, q_stride_n, q_stride_l,
            k_stride_b, k_stride_n, k_stride_s,
            v_stride_b, v_stride_n, v_stride_s,
            /*v_evblock_stride=*/0,
            m_stride_b, m_stride_n, m_stride_l,
            o_stride_b, o_stride_n, o_stride_l);
      } else {
        run_path_collapse3<MK, scalar_t, /*kPackedV=*/false,
                           /*kHasMask=*/false, /*kCausal=*/false>(
            q_ptr, k_ptr, v_ptr, p, ts,
            q_stride_b, q_stride_n, q_stride_l,
            k_stride_b, k_stride_n, k_stride_s,
            v_stride_b, v_stride_n, v_stride_s,
            /*v_evblock_stride=*/0,
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
          "[flash2_neon_l3kv] path=B (taskloop) B=%lld N=%lld L=%lld "
          "S=%lld E=%lld Ev=%lld kv_bytes=%lld l3_budget=%lld threads=%d "
          "max_concurrent_bn=%d num_groups=%d Lc_l2=%lld Sc_l2=%lld\n",
          (long long)p.B, (long long)p.N, (long long)p.L, (long long)p.S,
          (long long)p.E, (long long)p.Ev,
          (long long)kv_bytes_per_bn, (long long)l3_budget, total_threads,
          max_concurrent_bn, num_groups,
          (long long)ts.Lc_l2, (long long)ts.Sc_l2);
    }
    if (p.mask_ptr != nullptr) {
      if (p.is_causal) {
        run_path_taskloop<MK, scalar_t, /*kPackedV=*/false,
                          /*kHasMask=*/true, /*kCausal=*/true>(
            q_ptr, k_ptr, v_ptr, p, ts, num_groups,
            q_stride_b, q_stride_n, q_stride_l,
            k_stride_b, k_stride_n, k_stride_s,
            v_stride_b, v_stride_n, v_stride_s,
            /*v_evblock_stride=*/0,
            m_stride_b, m_stride_n, m_stride_l,
            o_stride_b, o_stride_n, o_stride_l);
      } else {
        run_path_taskloop<MK, scalar_t, /*kPackedV=*/false,
                          /*kHasMask=*/true, /*kCausal=*/false>(
            q_ptr, k_ptr, v_ptr, p, ts, num_groups,
            q_stride_b, q_stride_n, q_stride_l,
            k_stride_b, k_stride_n, k_stride_s,
            v_stride_b, v_stride_n, v_stride_s,
            /*v_evblock_stride=*/0,
            m_stride_b, m_stride_n, m_stride_l,
            o_stride_b, o_stride_n, o_stride_l);
      }
    } else {
      if (p.is_causal) {
        run_path_taskloop<MK, scalar_t, /*kPackedV=*/false,
                          /*kHasMask=*/false, /*kCausal=*/true>(
            q_ptr, k_ptr, v_ptr, p, ts, num_groups,
            q_stride_b, q_stride_n, q_stride_l,
            k_stride_b, k_stride_n, k_stride_s,
            v_stride_b, v_stride_n, v_stride_s,
            /*v_evblock_stride=*/0,
            m_stride_b, m_stride_n, m_stride_l,
            o_stride_b, o_stride_n, o_stride_l);
      } else {
        run_path_taskloop<MK, scalar_t, /*kPackedV=*/false,
                          /*kHasMask=*/false, /*kCausal=*/false>(
            q_ptr, k_ptr, v_ptr, p, ts, num_groups,
            q_stride_b, q_stride_n, q_stride_l,
            k_stride_b, k_stride_n, k_stride_s,
            v_stride_b, v_stride_n, v_stride_s,
            /*v_evblock_stride=*/0,
            m_stride_b, m_stride_n, m_stride_l,
            o_stride_b, o_stride_n, o_stride_l);
      }
    }
  }
}

// dtype dispatch entry.
template <class MK>
inline void sdpa_flash2_neon_l3kv_with_mk_impl(const SdpaParams& p) {
  if (p.dtype == SdpaDtype::kBFloat16) {
    sdpa_flash2_neon_l3kv_with_mk_tmpl<MK, at::BFloat16>(
        static_cast<const at::BFloat16*>(p.q_ptr),
        static_cast<const at::BFloat16*>(p.k_ptr),
        static_cast<const at::BFloat16*>(p.v_ptr),
        p);
  } else {
    sdpa_flash2_neon_l3kv_with_mk_tmpl<MK, float>(
        static_cast<const float*>(p.q_ptr),
        static_cast<const float*>(p.k_ptr),
        static_cast<const float*>(p.v_ptr),
        p);
  }
}

}  // anonymous namespace

// ── enabled MK impl 的 SDPA 入口 + 注册 ─────────────────────────────────
//
// 历史名 "flash2_neon_l3kv" 绑定到 baseline，等价于
// "flash2_neon_l3kv_baseline"。

#if FUSED_CPP_MK_ENABLE_BASELINE
namespace {
void sdpa_flash2_neon_l3kv_baseline_entry(const SdpaParams& p) {
  sdpa_flash2_neon_l3kv_with_mk_impl<
      ::fused_cpp::sdpa_microkernels::MK_Baseline>(p);
}
}  // anonymous namespace
REGISTER_SDPA_VERSION("flash2_neon_l3kv",
                      sdpa_flash2_neon_l3kv_baseline_entry);
REGISTER_SDPA_VERSION("flash2_neon_l3kv_baseline",
                      sdpa_flash2_neon_l3kv_baseline_entry);
#endif

#if FUSED_CPP_MK_ENABLE_SCALAR
namespace {
void sdpa_flash2_neon_l3kv_scalar_entry(const SdpaParams& p) {
  sdpa_flash2_neon_l3kv_with_mk_impl<
      ::fused_cpp::sdpa_microkernels::MK_Scalar>(p);
}
}  // anonymous namespace
REGISTER_SDPA_VERSION("flash2_neon_l3kv_scalar",
                      sdpa_flash2_neon_l3kv_scalar_entry);
#endif

#if FUSED_CPP_MK_ENABLE_PQUAD
namespace {
void sdpa_flash2_neon_l3kv_pquad_entry(const SdpaParams& p) {
  sdpa_flash2_neon_l3kv_with_mk_impl<
      ::fused_cpp::sdpa_microkernels::MK_PQuad>(p);
}
}  // anonymous namespace
REGISTER_SDPA_VERSION("flash2_neon_l3kv_pquad",
                      sdpa_flash2_neon_l3kv_pquad_entry);
#endif

#if FUSED_CPP_MK_ENABLE_QK_UBLOCK4
namespace {
void sdpa_flash2_neon_l3kv_qk_ublock4_entry(const SdpaParams& p) {
  sdpa_flash2_neon_l3kv_with_mk_impl<
      ::fused_cpp::sdpa_microkernels::MK_QkUblock4>(p);
}
}  // anonymous namespace
REGISTER_SDPA_VERSION("flash2_neon_l3kv_qk_ublock4",
                      sdpa_flash2_neon_l3kv_qk_ublock4_entry);
#endif
