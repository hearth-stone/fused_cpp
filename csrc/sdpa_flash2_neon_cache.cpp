// ── Cache-Aware Multi-Threaded NEON FlashAttention-2 SDPA Kernel ────────────
//
// 本文件实现一个 cache-aware 的 NEON FlashAttention-2 SDPA 内核，
// 注册名为 "flash2_neon_cache"。与 csrc/sdpa_flash2_neon.cpp 不同之处
// （按计划逐步落地）：
//
//   1. 显式三级 cache（L1 / L2 / L3）感知的 tile 推导：
//        L3 ←尽量驻留 K + V 全部矩阵（多核共享）
//        L2 ←Q tile 常驻 + 当前 K/V tile + 预取 K/V tile + Q-tile 级
//            running_max/running_sum
//        L1 ←micro-kernel 工作集 + L1 片级 local_max/local_sum
//      每级 cache 有独立的使用比例（L1=0.5 / L2=0.75 / L3=0.7）。
//
//   2. micro-kernel 形状：
//        默认 8×8（16 个 float32x4_t 累加器，剩余 16 寄存器作为
//          A/B 操作数 / softmax m/l 标量带 / lane 重排临时量）；
//        Sk 尾部退化到 8×4；Lq / Ev 尾部走标量兜底。
//
//   3. GEMM kernel 一律走 MLA 类指令：
//        - bf16 Q·K^T → fp32：默认 BFMMLA（vbfmmlaq_f32），
//          退化链为 BFMLALB/T → widen+FMLA；BFDOT 仅作可选后退库。
//        - fp32 Q·K^T 与 P̂·V → fp32：FMLA（vfmaq_f32）。
//        - bf16 P̂·V → fp32：V widen 到 fp32 后走 FMLA；严禁
//          BFMMLA / BFDOT / BFMLALB/T（一侧是 fp32）。
//
//   4. 双缓冲软件预取：进入每个 L2 KV tile 时对下一个 KV tile
//      发出 __builtin_prefetch（L2 hint）。
//
//   5. OpenMP 多核：(b, n, q_tile) 三维 collapse；K/V 在 L3 中
//      被多线程共享，禁止任何线程做 K/V 私有拷贝或 BLIS-style pack。
//
// 当前阶段（任务 1：骨架 + 自注册）：
//   入口先 fallback 到 flash2_neon 风格的朴素实现，仅用于保证
//   注册链路与等价性测试链路在编译期就能跑通。任务 2 起逐步替换
//   主循环为 cache-aware 实现。
//
// 依赖：仅 <arm_neon.h>（编译器自带）；构建走 setup.py 的
// glob("csrc/*.cpp")，不需修改 setup.py。

#include <torch/extension.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if defined(__linux__)
#include <unistd.h>
#endif

#include "sdpa_common.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sdpa_microkernels/neon_cache_config.h"
#include "sdpa_tile_sizes.h"

// ──────────────────────────────────────────────────────────────────────
// 三级 cache 常量、使用比例、运行时探测、TileSizes / compute_tile_sizes
// 已迁移至 csrc/sdpa_tile_sizes.h（与 sdpa_flash2_neon_l3kv.cpp 共用一份
// 实现）。本文件仅通过 `using` 在匿名命名空间中暴露需要的符号。
//
// 设计意图回顾（详见 sdpa_tile_sizes.h 顶部）：
//   * L3（默认 70 MB）容纳 K + V 全部矩阵，多核共享，避免 K/V 私有拷贝。
//   * L2（默认 1280 KB）容纳「Q tile + O tile（fp32 累加） +
//     当前 K/V tile + 双缓冲预取 K/V tile + Q-tile 级
//     running_max/running_sum」。
//   * L1（默认 64 KB）容纳 micro-kernel 工作集 + L1 片级
//     local_max/local_sum，比例固定 0.5 是为 HW prefetch 与
//     softmax m/l 标量带留余量。
// ──────────────────────────────────────────────────────────────────────

// ──────────────────────────────────────────────────────────────────────
// 软件预取开关（需求 7.4）
//
// 默认开启；编译期 `-DFUSED_CPP_SDPA_DISABLE_PREFETCH=1` 可关闭。
// 平台不支持 __builtin_prefetch 时静默退化。
// ──────────────────────────────────────────────────────────────────────

#ifndef FUSED_CPP_SDPA_DISABLE_PREFETCH
#define FUSED_CPP_SDPA_DISABLE_PREFETCH 0
#endif

#include "sdpa_microkernels/neon_cache_microkernels.h"
#include "sdpa_microkernels/mk_traits.h"
#include "sdpa_microkernels/all_impls.h"

namespace {

// 旧的 `using` 已经移除：本文件改成 `template <class MK>` 在主循环内
// 调用 `MK::qkt_8x8(...)` 等静态成员函数。具体 impl（MK_Baseline /
// MK_Scalar / 其他）由文件末尾的 `REGISTER_FLASH2_NEON_CACHE_MK(MK_X)`
// 显式实例化并注册到全局 SdpaRegistry，名字为 `flash2_neon_cache_<MK::kName>`。

// ──────────────────────────────────────────────────────────────────────
// 内部向量化 helper（与 sdpa_flash2_neon.cpp 中的 helper 等价；本文件
// 在 anonymous namespace 内独立持有副本，避免跨编译单元的私有依赖）。
// 任务 4-5 会在此基础上加入 8×8 / 8×4 micro-kernel + BFMMLA 主路径。
// ──────────────────────────────────────────────────────────────────────

#if FUSED_CPP_SDPA_CACHE_HAS_NEON

// vexpq_f32：基于 range reduction + 多项式估值的 NEON 向量化 expf 近似。
// 与 sdpa_flash2_neon.cpp 中的实现完全一致（绝对误差 ≤ 1e-4）。
static inline float32x4_t vexpq_f32(float32x4_t x) {
  const float32x4_t kLn2 = vdupq_n_f32(0.6931471805599453f);
  const float32x4_t kInvLn2 = vdupq_n_f32(1.4426950408889634f);
  const float32x4_t c1 = vdupq_n_f32(1.0f);
  const float32x4_t c2 = vdupq_n_f32(0.5f);
  const float32x4_t c3 = vdupq_n_f32(0.16666666f);
  const float32x4_t c4 = vdupq_n_f32(0.04166666f);
  const float32x4_t c5 = vdupq_n_f32(0.00833333f);

  const float32x4_t kHi = vdupq_n_f32(87.0f);
  const float32x4_t kLo = vdupq_n_f32(-87.0f);
  x = vminq_f32(x, kHi);
  x = vmaxq_f32(x, kLo);

  float32x4_t fn = vrndnq_f32(vmulq_f32(x, kInvLn2));
  int32x4_t n = vcvtq_s32_f32(fn);

  float32x4_t r = vfmsq_f32(x, fn, kLn2);

  float32x4_t r2 = vmulq_f32(r, r);
  (void)r2;
  float32x4_t poly = c5;
  poly = vfmaq_f32(c4, poly, r);
  poly = vfmaq_f32(c3, poly, r);
  poly = vfmaq_f32(c2, poly, r);
  poly = vfmaq_f32(c1, poly, r);
  poly = vfmaq_f32(c1, poly, r);

  int32x4_t exp_bits = vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(127)), 23);
  float32x4_t pow2n = vreinterpretq_f32_s32(exp_bits);

  return vmulq_f32(poly, pow2n);
}

// dot_fp32_neon：4 路展开的 fp32 GEMV 单行点积。
static inline float dot_fp32_neon(const float* a, const float* b, int64_t len) {
  float32x4_t acc = vdupq_n_f32(0.0f);
  int64_t i = 0;
  for (; i + 4 <= len; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    acc = vfmaq_f32(acc, va, vb);
  }
  float sum = vaddvq_f32(acc);
  for (; i < len; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

// widen_bf16x8_to_fp32x4_pair：把 8 个 bf16 元素 widen 为两组 fp32x4。
static inline void widen_bf16x8_to_fp32x4_pair(const uint16_t* src, float32x4_t* lo, float32x4_t* hi) {
  uint16x8_t bf = vld1q_u16(src);
  uint32x4_t u_lo = vmovl_u16(vget_low_u16(bf));
  uint32x4_t u_hi = vmovl_u16(vget_high_u16(bf));
  *lo = vreinterpretq_f32_u32(vshlq_n_u32(u_lo, 16));
  *hi = vreinterpretq_f32_u32(vshlq_n_u32(u_hi, 16));
}

// dot_bf16_neon：bf16 GEMV 单行点积。
//
// 当前骨架阶段沿用 flash2_neon 的策略：
//   * __ARM_FEATURE_BF16 启用时使用 vbfdotq_f32；
//   * 否则走 widen-to-fp32 + vfmaq_f32 路径。
// 任务 4 会以 BFMMLA 主路径 + BFMLALB/T → widen+FMLA 三级退化链替代。
static inline float dot_bf16_neon(const at::BFloat16* a, const at::BFloat16* b, int64_t len) {
  const uint16_t* ap = reinterpret_cast<const uint16_t*>(a);
  const uint16_t* bp = reinterpret_cast<const uint16_t*>(b);
  float32x4_t acc = vdupq_n_f32(0.0f);
  int64_t i = 0;

#if FUSED_CPP_SDPA_CACHE_HAS_BF16
  for (; i + 8 <= len; i += 8) {
    bfloat16x8_t va = vld1q_bf16(reinterpret_cast<const bfloat16_t*>(ap + i));
    bfloat16x8_t vb = vld1q_bf16(reinterpret_cast<const bfloat16_t*>(bp + i));
    acc = vbfdotq_f32(acc, va, vb);
  }
#else
  for (; i + 8 <= len; i += 8) {
    float32x4_t a_lo, a_hi, b_lo, b_hi;
    widen_bf16x8_to_fp32x4_pair(ap + i, &a_lo, &a_hi);
    widen_bf16x8_to_fp32x4_pair(bp + i, &b_lo, &b_hi);
    acc = vfmaq_f32(acc, a_lo, b_lo);
    acc = vfmaq_f32(acc, a_hi, b_hi);
  }
#endif

  float sum = vaddvq_f32(acc);
  for (; i + 4 <= len; i += 4) {
    uint16x4_t bf_a = vld1_u16(ap + i);
    uint16x4_t bf_b = vld1_u16(bp + i);
    float32x4_t fa = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(bf_a), 16));
    float32x4_t fb = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(bf_b), 16));
    float32x4_t prod = vmulq_f32(fa, fb);
    sum += vaddvq_f32(prod);
  }
  for (; i < len; ++i) {
    sum += static_cast<float>(a[i]) * static_cast<float>(b[i]);
  }
  return sum;
}

// fma_acc_fp32_neon：output_acc[ev] += scale * src[ev]，4 路展开。
static inline void fma_acc_fp32_neon(float* acc, const float* src, float scale, int64_t len) {
  const float32x4_t vs = vdupq_n_f32(scale);
  int64_t ev = 0;
  for (; ev + 4 <= len; ev += 4) {
    float32x4_t va = vld1q_f32(acc + ev);
    float32x4_t vv = vld1q_f32(src + ev);
    va = vfmaq_f32(va, vv, vs);
    vst1q_f32(acc + ev, va);
  }
  for (; ev < len; ++ev) {
    acc[ev] += scale * src[ev];
  }
}

// fma_acc_bf16_neon：output_acc[ev] += scale * widen(bf16 src[ev])，4 路展开。
static inline void fma_acc_bf16_neon(float* acc, const at::BFloat16* src, float scale, int64_t len) {
  const uint16_t* sp = reinterpret_cast<const uint16_t*>(src);
  const float32x4_t vs = vdupq_n_f32(scale);
  int64_t ev = 0;
  for (; ev + 4 <= len; ev += 4) {
    uint16x4_t bf = vld1_u16(sp + ev);
    float32x4_t vv = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(bf), 16));
    float32x4_t va = vld1q_f32(acc + ev);
    va = vfmaq_f32(va, vv, vs);
    vst1q_f32(acc + ev, va);
  }
  for (; ev < len; ++ev) {
    acc[ev] += scale * static_cast<float>(src[ev]);
  }
}

#endif  // FUSED_CPP_SDPA_CACHE_HAS_NEON

// ──────────────────────────────────────────────────────────────────────
// 标量 fallback helper：所有平台可用。
// ──────────────────────────────────────────────────────────────────────

template <typename scalar_t>
inline float dot_scalar(const scalar_t* a, const scalar_t* b, int64_t len) {
  float sum = 0.0f;
  for (int64_t i = 0; i < len; ++i) {
    sum += static_cast<float>(a[i]) * static_cast<float>(b[i]);
  }
  return sum;
}

template <typename scalar_t>
inline void fma_acc_scalar(float* acc, const scalar_t* src, float scale, int64_t len) {
  for (int64_t ev = 0; ev < len; ++ev) {
    acc[ev] += scale * static_cast<float>(src[ev]);
  }
}

inline float dispatch_dot_cache(const float* a, const float* b, int64_t len) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  return dot_fp32_neon(a, b, len);
#else
  return dot_scalar<float>(a, b, len);
#endif
}

inline float dispatch_dot_cache(const at::BFloat16* a, const at::BFloat16* b, int64_t len) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  return dot_bf16_neon(a, b, len);
#else
  return dot_scalar<at::BFloat16>(a, b, len);
#endif
}

inline void dispatch_fma_acc_cache(float* acc, const float* src, float scale, int64_t len) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  fma_acc_fp32_neon(acc, src, scale, len);
#else
  fma_acc_scalar<float>(acc, src, scale, len);
#endif
}

inline void dispatch_fma_acc_cache(float* acc, const at::BFloat16* src, float scale, int64_t len) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  fma_acc_bf16_neon(acc, src, scale, len);
#else
  fma_acc_scalar<at::BFloat16>(acc, src, scale, len);
#endif
}

// vectorized_exp_minus：批量 dst[j] = exp(src[j] - new_max)，返回块累加 sum。
inline float vectorized_exp_minus_cache(float* dst, const float* src, float new_max, int64_t len) {
  float block_sum = 0.0f;
  int64_t j = 0;
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  const float32x4_t vmax = vdupq_n_f32(new_max);
  float32x4_t vsum = vdupq_n_f32(0.0f);
  for (; j + 4 <= len; j += 4) {
    float32x4_t s = vld1q_f32(src + j);
    float32x4_t e = vexpq_f32(vsubq_f32(s, vmax));
    vst1q_f32(dst + j, e);
    vsum = vaddq_f32(vsum, e);
  }
  block_sum = vaddvq_f32(vsum);
#endif
  for (; j < len; ++j) {
    float e = std::exp(src[j] - new_max);
    dst[j] = e;
    block_sum += e;
  }
  return block_sum;
}

inline void scale_inplace_cache(float* buf, float scale, int64_t len) {
  int64_t i = 0;
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  const float32x4_t vs = vdupq_n_f32(scale);
  for (; i + 4 <= len; i += 4) {
    float32x4_t v = vld1q_f32(buf + i);
    v = vmulq_f32(v, vs);
    vst1q_f32(buf + i, v);
  }
#endif
  for (; i < len; ++i) {
    buf[i] *= scale;
  }
}

// ──────────────────────────────────────────────────────────────────────
// 三级 tile 尺寸推导（已迁至 csrc/sdpa_tile_sizes.h，本文件通过 using
// 引入到匿名命名空间）。
//
// 输出（TileSizes）含义：
//   * Sc_l3   ：L3 级 KV S 维切片长度（多核线程共享同一 L3 切片）
//   * Sc_l2   ：L2 级 KV S 维切片长度（一个线程在 L2 中常驻的 K/V tile）
//   * Lc_l2   ：L2 级 Q L 维 tile 长度（Q tile 常驻 L2）
//   * Sk_micro：micro-kernel 沿 KV S 维主体步长，固定 8（8×8 主体）
//   * Lq_micro：micro-kernel 沿 Q  L 维主体步长，固定 8
//   * Ev_micro：micro-kernel 沿 V  Ev 维主体步长，固定 8
//
// 不变式：
//   - Sc_l3 % Sc_l2 == 0
//   - Sc_l2 % Sk_micro == 0（Sk_micro=8）
//   - Lc_l2 % Lq_micro == 0（Lq_micro=8）
//   - Sc_l3, Sc_l2 ≤ S；Lc_l2 ≤ L；所有返回值 ≥ 1
// ──────────────────────────────────────────────────────────────────────

using ::fused_cpp::sdpa_tile_sizes::ceil_div_pos;
using ::fused_cpp::sdpa_tile_sizes::compute_tile_sizes;
using ::fused_cpp::sdpa_tile_sizes::effective_cache_bytes;
using ::fused_cpp::sdpa_tile_sizes::floor_to_mult_min;
using ::fused_cpp::sdpa_tile_sizes::TileSizes;

// ──────────────────────────────────────────────────────────────────────
// 8×8 micro-kernel：BFMMLA 主路径 + 三级退化链 + 两段 GEMM 精度分流
//
// ARM AArch64 PCS 寄存器分配（共 32 个 128-bit Vn 寄存器）：
//   v0..v15  ：临时 / A&B 操作数 / lane 重排 / range reduction（caller-saved）
//   v16..v31 ：accumulator（caller-saved，但 v8..v15 的 lo64 是 callee-saved
//              在 PCS 下；本 micro-kernel 标记 always_inline 后不会有
//              prologue 保存压力，由编译器自由分配）
//
// 8×8 累加器布局：
//   * BFMMLA 路径：C[8][8] 以 2×2 子块平铺，每个子块对应一个 float32x4_t；
//                 (8/2) × (8/2) = 16 个 float32x4_t，正好占用 v16..v31。
//                 子块 (i_blk, j_blk) ∈ [0,4)² 对应 C[2*i_blk:2*i_blk+2,
//                 2*j_blk:2*j_blk+2] 的「行优先 4 lane」打包：
//                 lane[0]=C[2i+0,2j+0], lane[1]=C[2i+0,2j+1],
//                 lane[2]=C[2i+1,2j+0], lane[3]=C[2i+1,2j+1]。
//   * 「行向 4 lane」路径（fp32 / FMLA / widen+FMLA / BFMLALB/T / BFDOT）：
//                 每行 8 列 → 2 个 float32x4_t，8 行 × 2 = 16 个寄存器。
//
// 两路布局之间的一次性切换通过 vzip/vuzp 完成（任务计划注释）。本实现
// 选择「BFMMLA 完成后立即在寄存器内重排到行向 4 lane 布局」，使后续
// scale + 写出 scores_buf 与 P̂·V 阶段共用同一种行向布局，避免在两段
// GEMM 之间反复切换。
// ──────────────────────────────────────────────────────────────────────

// ──────────────────────────────────────────────────────────────────────
// cache-aware 主循环（任务 6）：
//   外层并行单元：(b, n, q_tile)，q_tile 步长 Lc_l2（≥ Lq_micro=8）；
//   外层 KV 切片：[s_l3, s_l3+Sc_l3) → 在该切片内继续按 Sc_l2 切片
//                  → 内层调用 8×8 / 8×4 / 标量 micro-kernel；
//   每个 q_tile 维护 8 行级别的 running_max / running_sum / O_acc
//   常驻 L2，整轮 KV 扫描完成后做归一化写入 o_row。
//
// 任务 7 加入软件预取；任务 8 加入 OpenMP 多核。当前版本与 flash2_neon
// 在数值上严格等价（相同的 online softmax 算法 + 相同的 Q·K^T / P̂·V
// 分子计算）。
// ──────────────────────────────────────────────────────────────────────

constexpr int FLASH2_NEON_CACHE_BLOCK_S = 64;  // 历史保留（未启用）
constexpr int64_t FLASH2_NEON_CACHE_MAX_EV = 4096;

// process_q_tile_8rows：处理一个 q_tile 内连续 8 行（Q[q0:q0+8]）的
// 完整 attention（所有 KV）。固定 Lq=8。
//
// scores_l1   ：8 × Sc_l2 fp32 工作缓冲（L2 驻留，可被复用）
// O_acc       ：8 × Ev fp32 输出累加器（L2 驻留）
// running_max ：8 fp32（L2 驻留）
// running_sum ：8 fp32（L2 驻留）
template <class MK, typename scalar_t>
inline void process_q_tile_8rows_full(const scalar_t* q_ptr, const scalar_t* k_ptr, const scalar_t* v_ptr, int64_t b,
                                      int64_t n, int64_t q0, const SdpaParams& p, int64_t q_stride_b,
                                      int64_t q_stride_n, int64_t q_stride_l, int64_t k_stride_b, int64_t k_stride_n,
                                      int64_t k_stride_s, int64_t v_stride_b, int64_t v_stride_n, int64_t v_stride_s,
                                      int64_t m_stride_b, int64_t m_stride_n, int64_t m_stride_l, int64_t o_stride_b,
                                      int64_t o_stride_n, int64_t o_stride_l,
                                      int Lq_eff,  // 实际行数 ≤ 8（尾部 q_tile 用）
                                      const TileSizes& ts,
                                      float* scores_l1,    // 8 * Sc_l2
                                      float* O_acc,        // 8 * Ev
                                      float* running_max,  // 8
                                      float* running_sum,  // 8
                                      float* P_hat         // 8 * Sc_l2
) {
  const scalar_t* Qrow0 = q_ptr + b * q_stride_b + n * q_stride_n + q0 * q_stride_l;

  // 初始化 8 行 状态。
  for (int i = 0; i < 8; ++i) {
    running_max[i] = p.neg_inf;
    running_sum[i] = 0.0f;
  }
  for (int64_t k = 0; k < 8 * p.Ev; ++k) {
    O_acc[k] = 0.0f;
  }

  // 每行的 causal_limit。
  int64_t causal_lim[8];
  for (int i = 0; i < 8; ++i) {
    int64_t l_idx = q0 + i;
    causal_lim[i] = (i < Lq_eff) ? (l_idx + p.causal_offset) : -1;
  }

  // ── L3 → L2 → micro-kernel 嵌套 ────────────────────────────────────
  for (int64_t s_l3 = 0; s_l3 < p.S; s_l3 += ts.Sc_l3) {
    const int64_t s_l3_end = std::min(s_l3 + ts.Sc_l3, p.S);

    // 该 q_tile 在 L3 切片内的因果 early-exit（最大行的 limit）。
    int64_t max_causal = -1;
    for (int i = 0; i < Lq_eff; ++i) {
      if (causal_lim[i] > max_causal) max_causal = causal_lim[i];
    }
    if (p.is_causal && s_l3 > max_causal) break;

    for (int64_t s_l2 = s_l3; s_l2 < s_l3_end; s_l2 += ts.Sc_l2) {
      const int64_t s_l2_end = std::min(s_l2 + ts.Sc_l2, s_l3_end);
      const int64_t Sc_cur = s_l2_end - s_l2;

      if (p.is_causal && s_l2 > max_causal) break;

      // ── 双缓冲软件预取（需求 7.x）──
      // 进入当前 KV tile 处理之前，对**下一个** KV tile 发出
      // L2-level prefetch（K 与 V 各一次）；当前 tile 是该 q_tile 内
      // 最后一块时跳过预取。`__builtin_prefetch(addr, 0, 2)` 在 GCC /
      // Clang 上等价于 PRFM PLDL2KEEP（AArch64）。
#if !FUSED_CPP_SDPA_DISABLE_PREFETCH
      {
        const int64_t s_next = s_l2 + ts.Sc_l2;
        if (s_next < s_l3_end) {
          const scalar_t* K_next = k_ptr + b * k_stride_b + n * k_stride_n + s_next * k_stride_s;
          const scalar_t* V_next = v_ptr + b * v_stride_b + n * v_stride_n + s_next * v_stride_s;
          // 仅对 tile 起始的几个 cache line 发预取提示（HW prefetcher
          // 接管后续）。多发一些不会破坏正确性，只有性能影响。
          for (int line = 0; line < 4; ++line) {
            __builtin_prefetch(reinterpret_cast<const char*>(K_next) + line * 64, 0 /*read*/, 2 /*L2*/);
            __builtin_prefetch(reinterpret_cast<const char*>(V_next) + line * 64, 0, 2);
          }
        }
      }
#endif

      // ── 步骤 1: 计算 scores[8][Sc_cur] = scale * Q · K^T ──
      const scalar_t* Krow0 = k_ptr + b * k_stride_b + n * k_stride_n + s_l2 * k_stride_s;

      int64_t s_off = 0;
      // 主体：Sk_micro=8 步长。MK::qkt_8x8 写到固定 8x8 row-major
      // 临时 buffer，再 copy 到 scores_l1 的对应列段（scores_l1 行步长
      // 为 Sc_cur，与 micro-kernel 的内部 stride=8 不一致）。
      alignas(64) float tmp_qkt[8 * 8];
      for (; s_off + 8 <= Sc_cur; s_off += 8) {
        const scalar_t* K_tile = Krow0 + s_off * k_stride_s;
        if (Lq_eff == 8) {
          MK::qkt_8x8(Qrow0, q_stride_l, K_tile, k_stride_s, p.E, p.scale_f, tmp_qkt);
          for (int i = 0; i < 8; ++i) {
            ::fused_cpp::sdpa_pack_utils::copy_f32x8(tmp_qkt + i * 8, scores_l1 + i * Sc_cur + s_off);
          }
        } else {
          MK::qkt_tail(Qrow0, q_stride_l, K_tile, k_stride_s, p.E, p.scale_f, scores_l1 + s_off, Sc_cur, Lq_eff, 8);
        }
      }
      // 8×4 退化：Sk_remain ∈ [4, 7]。
      if (s_off + 4 <= Sc_cur) {
        const scalar_t* K_tile = Krow0 + s_off * k_stride_s;
        if (Lq_eff == 8) {
          MK::qkt_8x4(Qrow0, q_stride_l, K_tile, k_stride_s, p.E, p.scale_f, scores_l1 + s_off, Sc_cur);
        } else {
          MK::qkt_tail(Qrow0, q_stride_l, K_tile, k_stride_s, p.E, p.scale_f, scores_l1 + s_off, Sc_cur, Lq_eff, 4);
        }
        s_off += 4;
      }
      // 标量尾部：Sk_remain ∈ [1, 3]。
      if (s_off < Sc_cur) {
        const scalar_t* K_tile = Krow0 + s_off * k_stride_s;
        MK::qkt_tail(Qrow0, q_stride_l, K_tile, k_stride_s, p.E, p.scale_f, scores_l1 + s_off, Sc_cur, Lq_eff,
                     static_cast<int>(Sc_cur - s_off));
        s_off = Sc_cur;
      }

      // ── 步骤 2: additive attention mask ──
      if (p.mask_ptr) {
        for (int i = 0; i < Lq_eff; ++i) {
          const float* m_row = p.mask_ptr + b * m_stride_b + n * m_stride_n + (q0 + i) * m_stride_l + s_l2;
          float* sc_row = scores_l1 + i * Sc_cur;
          for (int64_t j = 0; j < Sc_cur; ++j) {
            sc_row[j] += m_row[j];
          }
        }
      }

      // ── 步骤 3: causal mask（按 row）──
      if (p.is_causal) {
        for (int i = 0; i < Lq_eff; ++i) {
          int64_t lim = causal_lim[i];
          float* sc_row = scores_l1 + i * Sc_cur;
          if (s_l2 + Sc_cur - 1 > lim) {
            for (int64_t j = 0; j < Sc_cur; ++j) {
              if (s_l2 + j > lim) sc_row[j] = p.neg_inf;
            }
          }
        }
      }

      // ── 步骤 4-8: 逐行 online softmax + P̂·V 累加 ──
      // 为了与 flash2_neon 数值等价，每行独立做 max / exp / correction，
      // 但 P̂·V 在 8 行级别批量化（micro-kernel 8x8 / 8x4）。
      //
      // 计算每行的 row_max[8]。
      float row_max[8];
      for (int i = 0; i < 8; ++i) row_max[i] = p.neg_inf;
      for (int i = 0; i < Lq_eff; ++i) {
        const float* sc_row = scores_l1 + i * Sc_cur;
        float m = p.neg_inf;
        for (int64_t j = 0; j < Sc_cur; ++j) {
          if (sc_row[j] > m) m = sc_row[j];
        }
        row_max[i] = m;
      }

      // new_max[i]、correction[i]、O_acc 校正、running_sum 校正。
      float new_max[8], correction[8];
      for (int i = 0; i < Lq_eff; ++i) {
        new_max[i] = std::max(running_max[i], row_max[i]);
        correction[i] = std::exp(running_max[i] - new_max[i]);
        running_sum[i] *= correction[i];
        // 校正 O_acc[i, :] *= correction[i]
        scale_inplace_cache(O_acc + i * p.Ev, correction[i], p.Ev);
      }

      // 计算 P_hat[i][j] = exp(scores[i][j] - new_max[i])，并累加 row_sum。
      for (int i = 0; i < Lq_eff; ++i) {
        const float* sc_row = scores_l1 + i * Sc_cur;
        float* p_row = P_hat + i * Sc_cur;
        float row_sum = vectorized_exp_minus_cache(p_row, sc_row, new_max[i], Sc_cur);
        running_sum[i] += row_sum;
      }
      // 把 P_hat 中超出 Lq_eff 的行清零（micro-kernel 会读到这些行）。
      for (int i = Lq_eff; i < 8; ++i) {
        std::memset(P_hat + i * Sc_cur, 0, sizeof(float) * Sc_cur);
      }

      // ── 步骤 9: O_acc[8][Ev] += P_hat[8][Sc_cur] · V[Sc_cur][Ev] ──
      // 沿 Sk 维以 Sk_micro=8 步长，沿 Ev 维以 Ev_micro=8 步长。
      const scalar_t* Vbase = v_ptr + b * v_stride_b + n * v_stride_n + s_l2 * v_stride_s;

      for (int64_t ev_off = 0; ev_off < p.Ev; ev_off += 8) {
        const int64_t Ev_cur = std::min<int64_t>(8, p.Ev - ev_off);

        int64_t k_off = 0;
        for (; k_off + 8 <= Sc_cur; k_off += 8) {
          const scalar_t* V_tile = Vbase + k_off * v_stride_s + ev_off;
          float* O_tile = O_acc + ev_off;
          if (Lq_eff == 8 && Ev_cur == 8) {
            MK::pv_8x8(P_hat + k_off, Sc_cur, V_tile, v_stride_s, 8, O_tile, p.Ev);
          } else {
            MK::pv_tail(P_hat + k_off, Sc_cur, V_tile, v_stride_s, 8, O_tile, p.Ev, Lq_eff, static_cast<int>(Ev_cur));
          }
        }
        // Sk 尾部 [k_off, Sc_cur)。
        if (k_off < Sc_cur) {
          const scalar_t* V_tile = Vbase + k_off * v_stride_s + ev_off;
          float* O_tile = O_acc + ev_off;
          MK::pv_tail(P_hat + k_off, Sc_cur, V_tile, v_stride_s, Sc_cur - k_off, O_tile, p.Ev, Lq_eff,
                      static_cast<int>(Ev_cur));
        }
      }

      // 更新 running_max。
      for (int i = 0; i < Lq_eff; ++i) {
        running_max[i] = new_max[i];
      }
    }  // s_l2
  }  // s_l3

  // ── 归一化并写入 o_row ──
  for (int i = 0; i < Lq_eff; ++i) {
    float* o_row = p.out_ptr + b * o_stride_b + n * o_stride_n + (q0 + i) * o_stride_l;
    if (running_sum[i] > 0.0f) {
      const float inv_sum = 1.0f / running_sum[i];
      int64_t ev = 0;
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
      const float32x4_t vinv = vdupq_n_f32(inv_sum);
      for (; ev + 4 <= p.Ev; ev += 4) {
        float32x4_t v = vld1q_f32(O_acc + i * p.Ev + ev);
        vst1q_f32(o_row + ev, vmulq_f32(v, vinv));
      }
#endif
      for (; ev < p.Ev; ++ev) {
        o_row[ev] = O_acc[i * p.Ev + ev] * inv_sum;
      }
    } else {
      for (int64_t ev = 0; ev < p.Ev; ++ev) {
        o_row[ev] = 0.0f;
      }
    }
  }
}

template <class MK, typename scalar_t>
inline void sdpa_flash2_neon_cache_with_mk_tmpl(const scalar_t* q_ptr, const scalar_t* k_ptr, const scalar_t* v_ptr,
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

  // 推导三级 tile 尺寸。
  TileSizes ts = compute_tile_sizes(p.B, p.N, p.S, p.L, p.E, p.Ev, sizeof(scalar_t));

  // q_tile 步长固定为 Lq_micro=8（process_q_tile_8rows_full 处理 8 行）。
  // Lc_l2 是更大的 q_tile 长度（用于 L2 占用估算），但实际计算单元仍是 8 行。
  // 后续优化可在 8 行 micro-kernel 之上嵌套 Lc_l2 / 8 个 micro-row 组以
  // 复用 K/V tile 的 L2 驻留，但此版本先保证正确性。
  constexpr int LQ_STEP = 8;
  const int64_t num_q_tiles = (p.L + LQ_STEP - 1) / LQ_STEP;

  // ── OpenMP 多核并行：(b, n, q_tile) 三维 collapse ─────────────────
  // 每个线程在 omp parallel 区域开头一次性分配私有 buffer，避免反复分配。
  // 多线程共享同一份驻留 L3 的 K/V，禁止 K/V 私有拷贝（设计约束 10）。
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    // 线程私有 buffer。Sc_l2 上限来自 compute_tile_sizes，O_acc 大小
    // 取决于 Ev。
    const int64_t sc_max = std::max<int64_t>(8, ts.Sc_l2);
    std::vector<float> scores_l1_vec(8 * sc_max);
    std::vector<float> P_hat_vec(8 * sc_max);
    std::vector<float> O_acc_vec(8 * p.Ev);
    float* scores_l1 = scores_l1_vec.data();
    float* P_hat = P_hat_vec.data();
    float* O_acc = O_acc_vec.data();
    alignas(64) float running_max[8];
    alignas(64) float running_sum[8];

#ifdef _OPENMP
#pragma omp for collapse(3) schedule(static)
#endif
    for (int64_t b = 0; b < p.B; ++b) {
      for (int64_t n = 0; n < p.N; ++n) {
        for (int64_t qi = 0; qi < num_q_tiles; ++qi) {
          const int64_t q0 = qi * LQ_STEP;
          int Lq_eff = static_cast<int>(std::min<int64_t>(LQ_STEP, p.L - q0));
          process_q_tile_8rows_full<MK, scalar_t>(
              q_ptr, k_ptr, v_ptr, b, n, q0, p, q_stride_b, q_stride_n, q_stride_l, k_stride_b, k_stride_n, k_stride_s,
              v_stride_b, v_stride_n, v_stride_s, m_stride_b, m_stride_n, m_stride_l, o_stride_b, o_stride_n,
              o_stride_l, Lq_eff, ts, scores_l1, O_acc, running_max, running_sum, P_hat);
        }
      }
    }
  }  // end omp parallel
}

}  // anonymous namespace

// ── dtype-erased 入口（注册到全局表）─────────────────────────────────
//
// 框架升级（微内核管理）：
//   原来这里只有一个 `sdpa_flash2_neon_cache_impl`，绑定到全局函数
//   `gemm_qkt_8x8 / gemm_pv_8x8 ...`。现在主循环模板化在 MK trait 上
//   （process_q_tile_8rows_full<MK, scalar_t> /
//    sdpa_flash2_neon_cache_with_mk_tmpl<MK, scalar_t>），每个 enabled
//   impl 都得到一份 fully-specialized SDPA，注册名为
//   `flash2_neon_cache_<MK::kName>`。运行期切换只需在 Python 侧把
//   version="flash2_neon_cache_<name>" 传给 sdpa_versioned。
//
//   原始 `flash2_neon_cache` 名字仍保留，绑定到 baseline impl，与历史
//   测试 / benchmark 文件对接。

namespace {

template <class MK>
inline void sdpa_flash2_neon_cache_with_mk_impl(const SdpaParams& p) {
  TORCH_CHECK(p.Ev <= FLASH2_NEON_CACHE_MAX_EV, "sdpa_flash2_neon_cache_impl<", MK::kName, ">: v_head_dim (", p.Ev,
              ") exceeds maximum supported value (", FLASH2_NEON_CACHE_MAX_EV, ")");

  if (p.dtype == SdpaDtype::kBFloat16) {
    sdpa_flash2_neon_cache_with_mk_tmpl<MK, at::BFloat16>(static_cast<const at::BFloat16*>(p.q_ptr),
                                                          static_cast<const at::BFloat16*>(p.k_ptr),
                                                          static_cast<const at::BFloat16*>(p.v_ptr), p);
  } else {
    sdpa_flash2_neon_cache_with_mk_tmpl<MK, float>(
        static_cast<const float*>(p.q_ptr), static_cast<const float*>(p.k_ptr), static_cast<const float*>(p.v_ptr), p);
  }
}

}  // anonymous namespace

// ── enabled MK impl 的 SDPA 入口 + 注册 ─────────────────────────────────
//
// 每个 impl 显式实例化一次 `sdpa_flash2_neon_cache_with_mk_impl<MK>`
// 并注册成 `flash2_neon_cache_<MK::kName>`。下面的 do_register 静态变量
// 在 .so 加载时执行，保证与原 `REGISTER_SDPA_VERSION` 的注册时机一致。
//
// 历史名 "flash2_neon_cache" 也保留：绑定到 baseline，等价于
// "flash2_neon_cache_baseline"。

#if FUSED_CPP_MK_ENABLE_BASELINE
namespace {
void sdpa_flash2_neon_cache_baseline_entry(const SdpaParams& p) {
  sdpa_flash2_neon_cache_with_mk_impl<::fused_cpp::sdpa_microkernels::MK_Baseline>(p);
}
}  // anonymous namespace
REGISTER_SDPA_VERSION("flash2_neon_cache", sdpa_flash2_neon_cache_baseline_entry);
REGISTER_SDPA_VERSION("flash2_neon_cache_baseline", sdpa_flash2_neon_cache_baseline_entry);
#endif

#if FUSED_CPP_MK_ENABLE_SCALAR
namespace {
void sdpa_flash2_neon_cache_scalar_entry(const SdpaParams& p) {
  sdpa_flash2_neon_cache_with_mk_impl<::fused_cpp::sdpa_microkernels::MK_Scalar>(p);
}
}  // anonymous namespace
REGISTER_SDPA_VERSION("flash2_neon_cache_scalar", sdpa_flash2_neon_cache_scalar_entry);
#endif

#if FUSED_CPP_MK_ENABLE_PQUAD
namespace {
void sdpa_flash2_neon_cache_pquad_entry(const SdpaParams& p) {
  sdpa_flash2_neon_cache_with_mk_impl<::fused_cpp::sdpa_microkernels::MK_PQuad>(p);
}
}  // anonymous namespace
REGISTER_SDPA_VERSION("flash2_neon_cache_pquad", sdpa_flash2_neon_cache_pquad_entry);
#endif

#if FUSED_CPP_MK_ENABLE_QK_UBLOCK4
namespace {
void sdpa_flash2_neon_cache_qk_ublock4_entry(const SdpaParams& p) {
  sdpa_flash2_neon_cache_with_mk_impl<::fused_cpp::sdpa_microkernels::MK_QkUblock4>(p);
}
}  // anonymous namespace
REGISTER_SDPA_VERSION("flash2_neon_cache_qk_ublock4", sdpa_flash2_neon_cache_qk_ublock4_entry);
#endif
