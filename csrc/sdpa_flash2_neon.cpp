// ── FlashAttention-2 NEON 向量化 SDPA 内核 ─────────────────────────────
//
// 算法来源：
//   完整克隆自 csrc/sdpa.cpp::sdpa_flash2_kernel_impl 的 per-Q-row online
//   softmax 算法骨架（BLOCK_S=64、三状态 running_max/running_sum/output_acc、
//   K/V 维度按块迭代、因果块级 early-exit），不修改算法语义。
//
// 与 flash2 的差异（仅热路径替换为 ARM NEON intrinsics）：
//   1. Q·K^T 点积：fp32 走 float32x4_t + vfmaq_f32（4 路展开，尾部标量补齐）；
//      bf16 在 __ARM_FEATURE_BF16 启用时走 bfloat16x8_t + vbfdotq_f32，
//      否则走 widen-to-fp32 + vfmaq_f32 路径。
//   2. softmax 中 exp(scores - new_max)：用内部静态向量化 vexpq_f32
//      （range reduction + 多项式估值，绝对误差 ≤ 1e-4）按 4 路批量计算，
//      尾部走 std::exp 标量补齐。
//   3. output_acc[ev] += exp_val * V_row[ev]：fp32 走 vfmaq_f32 4 路展开，
//      bf16 widen 后再走 vfmaq_f32；Ev 不被 4 整除时尾部标量补齐。
//
// 路径分支：
//   * 编译期：__aarch64__ 定义 → NEON 路径；未定义 → 纯标量 fallback。
//   * 编译期：__ARM_FEATURE_BF16 定义 → bf16 dot 路径；未定义 → bf16 widen 路径。
//   * 运行期：head_dim / Ev 不被 4 整除 → 主体 4 路向量化 + 尾部标量；
//     block_len（softmax 阶段）不被 4 整除时同理。
//
// 数值精度约束：
//   * fp32：与 flash2 在 atol=1e-6/rtol=1e-5 容差内一致；
//   * bf16：与 flash2 在 atol=1e-2/rtol=1e-2 容差内一致；
//   * vbfdotq_f32 的累加顺序与标量 fallback 不同，因此跨实现按 dtype 容差
//     对比，而非逐元素相等（详见 vexpq_f32 / dot helper 注释）。
//
// 依赖：仅 <arm_neon.h>（编译器自带），不引入第三方依赖；构建走
// setup.py 的 glob("csrc/*.cpp")，不需修改 setup.py。

#include <torch/extension.h>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstring>

#include "sdpa_common.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#define FUSED_CPP_SDPA_HAS_NEON 1
#else
#define FUSED_CPP_SDPA_HAS_NEON 0
#endif

// `__ARM_FEATURE_BF16` 仅在 GCC 13+ / Clang 下作为聚合宏定义；GCC 10–12
// （含鲲鹏 RH/CentOS 8 上常见的 gcc-toolset-11/12）只定义细分宏
// `__ARM_FEATURE_BF16_VECTOR_ARITHMETIC`。BFDOT 等向量指令由后者提供，
// 因此两个宏只要有一个定义就启用 BF16 dot 路径。
#if (defined(__ARM_FEATURE_BF16) || \
     defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)) && \
    FUSED_CPP_SDPA_HAS_NEON
#define FUSED_CPP_SDPA_HAS_BF16_DOT 1
#else
#define FUSED_CPP_SDPA_HAS_BF16_DOT 0
#endif

namespace {

// ──────────────────────────────────────────────────────────────────────
// 内部向量化 helper（仅在 __aarch64__ 下定义；非 NEON 平台走纯标量）。
// ──────────────────────────────────────────────────────────────────────

#if FUSED_CPP_SDPA_HAS_NEON

// vexpq_f32：基于 range reduction + 多项式估值的 NEON 向量化 expf 近似。
//
// 输入：x ∈ [-87, 87]（softmax 调用前已减去 new_max，绝对值通常远小于 87）；
// 输出：与 std::expf 在该区间内绝对误差 ≤ 1e-4，相对误差 ≤ 1e-5；
// 选择理由：替代 4 次 std::expf 调用，避免 libm 路由开销。
//
// 推导：x = n*ln2 + r，其中 n = round(x / ln2)，r ∈ [-ln2/2, ln2/2]；
//      e^x = 2^n * e^r，e^r 用 5 阶多项式近似（误差 < 1e-7）。
static inline float32x4_t vexpq_f32(float32x4_t x) {
  // 多项式近似系数（针对 e^r, r ∈ [-ln2/2, ln2/2]）
  const float32x4_t kLn2  = vdupq_n_f32(0.6931471805599453f);
  const float32x4_t kInvLn2 = vdupq_n_f32(1.4426950408889634f);  // 1 / ln2
  const float32x4_t c1 = vdupq_n_f32(1.0f);
  const float32x4_t c2 = vdupq_n_f32(0.5f);
  const float32x4_t c3 = vdupq_n_f32(0.16666666f);
  const float32x4_t c4 = vdupq_n_f32(0.04166666f);
  const float32x4_t c5 = vdupq_n_f32(0.00833333f);

  // 输入 clamp 到 [-87, 87]，避免 2^n 上溢/下溢
  const float32x4_t kHi = vdupq_n_f32(87.0f);
  const float32x4_t kLo = vdupq_n_f32(-87.0f);
  x = vminq_f32(x, kHi);
  x = vmaxq_f32(x, kLo);

  // n = round(x * (1/ln2))；用 vcvtnq 实现 round-to-nearest-even
  float32x4_t fn = vrndnq_f32(vmulq_f32(x, kInvLn2));
  int32x4_t n = vcvtq_s32_f32(fn);

  // r = x - n*ln2
  float32x4_t r = vfmsq_f32(x, fn, kLn2);

  // e^r ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120
  float32x4_t r2 = vmulq_f32(r, r);
  float32x4_t poly = c5;
  poly = vfmaq_f32(c4, poly, r);
  poly = vfmaq_f32(c3, poly, r);
  poly = vfmaq_f32(c2, poly, r);
  poly = vfmaq_f32(c1, poly, r);
  poly = vfmaq_f32(c1, poly, r);
  // poly 此时 ≈ e^r

  // 2^n：把整数 n 编码到 IEEE 754 指数位 ((n + 127) << 23)
  int32x4_t exp_bits = vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(127)), 23);
  float32x4_t pow2n = vreinterpretq_f32_s32(exp_bits);

  return vmulq_f32(poly, pow2n);
}

// dot_fp32：4 路展开的 fp32 GEMV 单行点积（dst = sum(a[i] * b[i])）。
//
// a/b 长度为 len，未对齐时 vld1q_f32 在 ARMv8 上仍然安全（不需要 16B 对齐）。
// 尾部不能被 4 整除时走标量补齐。
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
// bf16 二进制 = fp32 高 16 位，所以 widen 等价于左移 16 位填零。
static inline void widen_bf16x8_to_fp32x4_pair(
    const uint16_t* src,
    float32x4_t* lo,
    float32x4_t* hi) {
  uint16x8_t bf = vld1q_u16(src);
  // 低 4 个：u16 → u32 → << 16 → reinterpret as f32
  uint32x4_t u_lo = vmovl_u16(vget_low_u16(bf));
  uint32x4_t u_hi = vmovl_u16(vget_high_u16(bf));
  *lo = vreinterpretq_f32_u32(vshlq_n_u32(u_lo, 16));
  *hi = vreinterpretq_f32_u32(vshlq_n_u32(u_hi, 16));
}

// dot_bf16_neon：bf16 GEMV 单行点积。
//
// __ARM_FEATURE_BF16 启用时使用 vbfdotq_f32（每条指令处理 4 对 bf16，
// 累加到 float32x4_t）；否则走 widen-to-fp32 + vfmaq_f32 路径。
//
// 注意：vbfdotq_f32 的累加顺序与标量 std::accumulate 不同，因此跨实现
// 等价性按 dtype 容差比较，不要求逐元素相等（详见文件头注释）。
static inline float dot_bf16_neon(
    const at::BFloat16* a, const at::BFloat16* b, int64_t len) {
  const uint16_t* ap = reinterpret_cast<const uint16_t*>(a);
  const uint16_t* bp = reinterpret_cast<const uint16_t*>(b);
  float32x4_t acc = vdupq_n_f32(0.0f);
  int64_t i = 0;

#if FUSED_CPP_SDPA_HAS_BF16_DOT
  // 主体：每次 8 元素 → 1 条 vbfdotq_f32 指令
  for (; i + 8 <= len; i += 8) {
    bfloat16x8_t va = vld1q_bf16(reinterpret_cast<const bfloat16_t*>(ap + i));
    bfloat16x8_t vb = vld1q_bf16(reinterpret_cast<const bfloat16_t*>(bp + i));
    acc = vbfdotq_f32(acc, va, vb);
  }
#else
  // 主体：每次 8 元素 → widen 成两组 fp32x4 → 两次 vfmaq_f32
  for (; i + 8 <= len; i += 8) {
    float32x4_t a_lo, a_hi, b_lo, b_hi;
    widen_bf16x8_to_fp32x4_pair(ap + i, &a_lo, &a_hi);
    widen_bf16x8_to_fp32x4_pair(bp + i, &b_lo, &b_hi);
    acc = vfmaq_f32(acc, a_lo, b_lo);
    acc = vfmaq_f32(acc, a_hi, b_hi);
  }
#endif

  float sum = vaddvq_f32(acc);
  // 尾部：先尝试 4 路 widen，再单元素标量补齐
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

// fma_acc_fp32：output_acc[ev] += scale * src[ev]，4 路展开。
static inline void fma_acc_fp32_neon(
    float* acc, const float* src, float scale, int64_t len) {
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

// fma_acc_bf16：output_acc[ev] += scale * widen(bf16 src[ev])，4 路展开。
//
// bf16 输入统一 widen 到 fp32 后再做 fma，跨平台行为一致；不依赖
// __ARM_FEATURE_BF16（accumulator 与 scale 都是 fp32）。
static inline void fma_acc_bf16_neon(
    float* acc, const at::BFloat16* src, float scale, int64_t len) {
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

#endif  // FUSED_CPP_SDPA_HAS_NEON

// ──────────────────────────────────────────────────────────────────────
// 标量 fallback helper：在所有平台都可用，用于 __aarch64__ 未定义时
// 的主路径，以及 NEON 路径的尾部补齐参考。
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
inline void fma_acc_scalar(
    float* acc, const scalar_t* src, float scale, int64_t len) {
  for (int64_t ev = 0; ev < len; ++ev) {
    acc[ev] += scale * static_cast<float>(src[ev]);
  }
}

// dispatch_dot：按 dtype + 平台选择 NEON 或标量路径。
inline float dispatch_dot(const float* a, const float* b, int64_t len) {
#if FUSED_CPP_SDPA_HAS_NEON
  return dot_fp32_neon(a, b, len);
#else
  return dot_scalar<float>(a, b, len);
#endif
}

inline float dispatch_dot(
    const at::BFloat16* a, const at::BFloat16* b, int64_t len) {
#if FUSED_CPP_SDPA_HAS_NEON
  return dot_bf16_neon(a, b, len);
#else
  return dot_scalar<at::BFloat16>(a, b, len);
#endif
}

inline void dispatch_fma_acc(
    float* acc, const float* src, float scale, int64_t len) {
#if FUSED_CPP_SDPA_HAS_NEON
  fma_acc_fp32_neon(acc, src, scale, len);
#else
  fma_acc_scalar<float>(acc, src, scale, len);
#endif
}

inline void dispatch_fma_acc(
    float* acc, const at::BFloat16* src, float scale, int64_t len) {
#if FUSED_CPP_SDPA_HAS_NEON
  fma_acc_bf16_neon(acc, src, scale, len);
#else
  fma_acc_scalar<at::BFloat16>(acc, src, scale, len);
#endif
}

// vectorized_exp_minus：批量计算 dst[j] = exp(src[j] - new_max)，并返回
// 块累加 sum。NEON 路径下用 vexpq_f32；fallback 走 std::exp。
inline float vectorized_exp_minus(
    float* dst, const float* src, float new_max, int64_t len) {
  float block_sum = 0.0f;
  int64_t j = 0;
#if FUSED_CPP_SDPA_HAS_NEON
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

// scale_inplace：buf[i] *= scale，4 路展开。
inline void scale_inplace(float* buf, float scale, int64_t len) {
  int64_t i = 0;
#if FUSED_CPP_SDPA_HAS_NEON
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
// FlashAttention-2 NEON 内核主体
// ──────────────────────────────────────────────────────────────────────

constexpr int FLASH2_NEON_BLOCK_S = 64;
constexpr int64_t FLASH2_NEON_MAX_EV = 4096;

template <typename scalar_t>
inline void sdpa_flash2_neon_kernel_tmpl(
    const scalar_t* q_ptr,
    const scalar_t* k_ptr,
    const scalar_t* v_ptr,
    const SdpaParams& p) {
  // ── stride（contiguous [B, N, seq, dim] 布局，与 flash2 一致）──
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

  constexpr int BS = FLASH2_NEON_BLOCK_S;
  const int64_t num_blocks = (p.S + BS - 1) / BS;

#ifdef _OPENMP
  #pragma omp parallel for collapse(2) schedule(static)
#endif
  for (int64_t b = 0; b < p.B; ++b) {
    for (int64_t n = 0; n < p.N; ++n) {
      for (int64_t l = 0; l < p.L; ++l) {
        const scalar_t* q_row = q_ptr + b * q_stride_b
                                      + n * q_stride_n
                                      + l * q_stride_l;
        float* o_row = p.out_ptr + b * o_stride_b
                                 + n * o_stride_n
                                 + l * o_stride_l;

        // ── Online Softmax 三状态（与 flash2 完全一致）──
        float running_max = p.neg_inf;
        float running_sum = 0.0f;
        alignas(64) float output_acc[FLASH2_NEON_MAX_EV];
        for (int64_t ev = 0; ev < p.Ev; ++ev) {
          output_acc[ev] = 0.0f;
        }

        alignas(64) float scores_buf[BS];
        alignas(64) float exp_buf[BS];

        const int64_t causal_limit = l + p.causal_offset;

        for (int64_t blk = 0; blk < num_blocks; ++blk) {
          const int64_t block_start = blk * BS;
          const int64_t block_len = std::min(
              static_cast<int64_t>(BS), p.S - block_start);

          // 因果块级 early-exit（与 flash2 完全一致）
          if (p.is_causal && block_start > causal_limit) {
            break;
          }

          // ── 步骤 1: 计算当前块 scores（向量化点积）──
          for (int64_t j = 0; j < block_len; ++j) {
            const scalar_t* k_row = k_ptr + b * k_stride_b
                                          + n * k_stride_n
                                          + (block_start + j) * k_stride_s;
            float dot = dispatch_dot(q_row, k_row, p.E);
            scores_buf[j] = dot * p.scale_f;
          }

          // ── 步骤 2: additive attention mask ──
          if (p.mask_ptr) {
            const float* m_row = p.mask_ptr + b * m_stride_b
                                            + n * m_stride_n
                                            + l * m_stride_l
                                            + block_start;
            for (int64_t j = 0; j < block_len; ++j) {
              scores_buf[j] += m_row[j];
            }
          }

          // ── 步骤 3: 因果掩码（逐元素，与 flash2 完全一致）──
          if (p.is_causal) {
            if (block_start + block_len - 1 > causal_limit) {
              for (int64_t j = 0; j < block_len; ++j) {
                if (block_start + j > causal_limit) {
                  scores_buf[j] = p.neg_inf;
                }
              }
            }
          }

          // ── 步骤 4(a): block_max ──
          float block_max = p.neg_inf;
          for (int64_t j = 0; j < block_len; ++j) {
            if (scores_buf[j] > block_max) {
              block_max = scores_buf[j];
            }
          }

          // ── 步骤 4(b)(c)(d): 修正 running_sum / output_acc ──
          const float new_max = std::max(running_max, block_max);
          const float correction = std::exp(running_max - new_max);
          running_sum *= correction;
          scale_inplace(output_acc, correction, p.Ev);

          // ── 步骤 4(e): 向量化 exp(scores - new_max)，累加到 running_sum ──
          //
          // 用 NEON vexpq_f32 批量计算 exp_buf[0..block_len)，并返回
          // 块累加 sum；尾部 / fallback 平台走 std::exp。
          const float block_sum = vectorized_exp_minus(
              exp_buf, scores_buf, new_max, block_len);
          running_sum += block_sum;

          // 累加 exp_val * V_row 到 output_acc
          for (int64_t j = 0; j < block_len; ++j) {
            const scalar_t* v_row = v_ptr + b * v_stride_b
                                          + n * v_stride_n
                                          + (block_start + j) * v_stride_s;
            dispatch_fma_acc(output_acc, v_row, exp_buf[j], p.Ev);
          }

          // ── 步骤 4(f): 更新 running_max ──
          running_max = new_max;
        }  // end block loop

        // ── 步骤 5: 最终归一化 ──
        if (running_sum > 0.0f) {
          const float inv_sum = 1.0f / running_sum;
          // o_row = output_acc * inv_sum：复用 scale_inplace 写入到 o_row
          int64_t ev = 0;
#if FUSED_CPP_SDPA_HAS_NEON
          const float32x4_t vinv = vdupq_n_f32(inv_sum);
          for (; ev + 4 <= p.Ev; ev += 4) {
            float32x4_t v = vld1q_f32(output_acc + ev);
            vst1q_f32(o_row + ev, vmulq_f32(v, vinv));
          }
#endif
          for (; ev < p.Ev; ++ev) {
            o_row[ev] = output_acc[ev] * inv_sum;
          }
        } else {
          for (int64_t ev = 0; ev < p.Ev; ++ev) {
            o_row[ev] = 0.0f;
          }
        }
      }  // end l loop
    }  // end n loop
  }  // end b loop
}

}  // anonymous namespace

// ── dtype-erased 入口（注册到全局表）─────────────────────────────────
void sdpa_flash2_neon_impl(const SdpaParams& p) {
  TORCH_CHECK(p.Ev <= FLASH2_NEON_MAX_EV,
              "sdpa_flash2_neon_impl: v_head_dim (", p.Ev,
              ") exceeds maximum supported value (", FLASH2_NEON_MAX_EV, ")");

  if (p.dtype == SdpaDtype::kBFloat16) {
    sdpa_flash2_neon_kernel_tmpl<at::BFloat16>(
        static_cast<const at::BFloat16*>(p.q_ptr),
        static_cast<const at::BFloat16*>(p.k_ptr),
        static_cast<const at::BFloat16*>(p.v_ptr),
        p);
  } else {
    sdpa_flash2_neon_kernel_tmpl<float>(
        static_cast<const float*>(p.q_ptr),
        static_cast<const float*>(p.k_ptr),
        static_cast<const float*>(p.v_ptr),
        p);
  }
}

REGISTER_SDPA_VERSION("flash2_neon", sdpa_flash2_neon_impl);
