// microkernels.h
// 抽离自 fused_cpp/csrc/sdpa_microkernels/neon_cache_microkernels.h 的最小副本。
// 仅包含两个 microkernel 及它们的 pack helper，纯 C++（无 Torch）。
//
// 抽离的 kernel：
//   1. gemm_qkt_microkernel_8x8_bf16_packqk_seq4_bmajor_inner
//      （bf16 QKᵀ，BFMMLA + B-major 调度，对应远程实测 ~64.84 GFLOPS / 70% peak）
//   2. gemm_pv_microkernel_8x8_bf16_pquad
//      （bf16 PV，P 端 quad load + lane FMA + 跨迭代 V 预取「方案 A」）
//
// 平台：aarch64 + BFMMLA。Apple Silicon (M1+) ✓，Linux Neoverse / 鲲鹏 ✓。
//
// bf16 存储用 uint16_t（与 at::BFloat16 / __bf16 位等价）；widen 到 fp32
// 通过 (uint32(u16) << 16).reinterpret_as<float>() 完成。

#pragma once

#include <arm_neon.h>
#include <cstdint>
#include <cstring>

// ── BFMMLA 可用性检测 ─────────────────────────────────────────────────────
// Apple Silicon (M1+) 有 BFMMLA 但 Apple clang 不定义 __ARM_FEATURE_BF16；
// 用 __APPLE__ 兜底，与原 neon_cache_config.h:41 行为一致。
#ifndef HAS_BFMMLA
#if defined(__aarch64__) && \
    (defined(__ARM_FEATURE_BF16) || defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__APPLE__))
#define HAS_BFMMLA 1
#else
#define HAS_BFMMLA 0
#endif
#endif

#if !defined(__aarch64__)
#error "microkernels.h requires aarch64 (ARM 64-bit)"
#endif

// ── bf16 → fp32 widening helpers ─────────────────────────────────────────
// bf16 的二进制等价于 fp32 的高 16 位 → widen = (uint32(bf) << 16).as<f32>。

static inline float bf16_to_fp32_scalar(uint16_t bf) {
  uint32_t u = static_cast<uint32_t>(bf) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

// 把 4 个 bf16 widen 为 float32x4_t。
static inline float32x4_t widen_bf16x4_to_fp32(const uint16_t* src) {
  uint16x4_t bf = vld1_u16(src);
  return vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(bf), 16));
}

// ── pack helpers ──────────────────────────────────────────────────────────
// 把 8 行 × E 列的 bf16 矩阵 pre-pack 成 BFMMLA 友好的 [E_main/4, 32] u16 布局。
// 每个 e_block（4 列）打包 8 行 → 32 个 u16。E % 4 != 0 时 partial e_block
// 不 pack，由 inner kernel 的标量 tail 路径处理。
//
// Q 和 K 形状对称，但语义不同（Q 是 8 行 query、K 是 8 行 key）；
// 调用方各传自己的 row_stride 即可。

inline void pack_q_8rows_to_seq_bf16(const uint16_t* Q, int64_t q_row_stride, int64_t E, uint16_t* Q_seq) {
  const int64_t E_main = E & ~int64_t{3};
  for (int64_t e = 0; e < E_main; e += 4) {
    uint16_t* dst = Q_seq + (e / 4) * 32;
    for (int row = 0; row < 8; ++row) {
      std::memcpy(dst + row * 4, Q + row * q_row_stride + e, 8);
    }
  }
}

inline void pack_k_8rows_to_seq_bf16(const uint16_t* K, int64_t k_row_stride, int64_t E, uint16_t* K_seq) {
  const int64_t E_main = E & ~int64_t{3};
  for (int64_t e = 0; e < E_main; e += 4) {
    uint16_t* dst = K_seq + (e / 4) * 32;
    for (int row = 0; row < 8; ++row) {
      std::memcpy(dst + row * 4, K + row * k_row_stride + e, 8);
    }
  }
}

// ── Microkernel 1：gemm_qkt_microkernel_8x8_bf16_packqk_seq4_bmajor_inner ──
// 计算 scores[8][8] = scale * Q[8][E] · K[8][E]^T
// 输入：
//   Q_seq   pre-pack [E_main/4][32] u16（pack_q_8rows_to_seq_bf16 的输出）
//   Q_orig  原始 Q（仅用于 e ≥ E&~3 的标量 tail）
//   K_seq   同 Q_seq 但是对 K
//   K_orig  原始 K（标量 tail 用）
//   E       reduce 维
//   scale   缩放因子（softmax 前的 1/sqrt(d_k)）
// 输出：
//   scores_buf  [8][8] fp32，行向 4 lane 布局

static inline void gemm_qkt_microkernel_8x8_bf16_packqk_seq4_bmajor_inner(const uint16_t* Q_seq, const uint16_t* Q_orig,
                                                                          int64_t q_row_stride, const uint16_t* K_seq,
                                                                          const uint16_t* K_orig, int64_t k_row_stride,
                                                                          int64_t E, float scale, float* scores_buf) {
  float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
  float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
  float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
  float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
  float32x4_t c40 = vdupq_n_f32(0), c41 = vdupq_n_f32(0);
  float32x4_t c50 = vdupq_n_f32(0), c51 = vdupq_n_f32(0);
  float32x4_t c60 = vdupq_n_f32(0), c61 = vdupq_n_f32(0);
  float32x4_t c70 = vdupq_n_f32(0), c71 = vdupq_n_f32(0);

  int64_t e = 0;

#if HAS_BFMMLA
  float32x4_t bm00 = vdupq_n_f32(0), bm01 = vdupq_n_f32(0);
  float32x4_t bm02 = vdupq_n_f32(0), bm03 = vdupq_n_f32(0);
  float32x4_t bm10 = vdupq_n_f32(0), bm11 = vdupq_n_f32(0);
  float32x4_t bm12 = vdupq_n_f32(0), bm13 = vdupq_n_f32(0);
  float32x4_t bm20 = vdupq_n_f32(0), bm21 = vdupq_n_f32(0);
  float32x4_t bm22 = vdupq_n_f32(0), bm23 = vdupq_n_f32(0);
  float32x4_t bm30 = vdupq_n_f32(0), bm31 = vdupq_n_f32(0);
  float32x4_t bm32 = vdupq_n_f32(0), bm33 = vdupq_n_f32(0);

  for (; e + 4 <= E; e += 4) {
    const uint16_t* q_base = Q_seq + (e / 4) * 32;
    bfloat16x8_t a01 = vreinterpretq_bf16_u16(vld1q_u16(q_base + 0));
    bfloat16x8_t a23 = vreinterpretq_bf16_u16(vld1q_u16(q_base + 8));
    bfloat16x8_t a45 = vreinterpretq_bf16_u16(vld1q_u16(q_base + 16));
    bfloat16x8_t a67 = vreinterpretq_bf16_u16(vld1q_u16(q_base + 24));

    const uint16_t* k_base = K_seq + (e / 4) * 32;
    bfloat16x8_t b01 = vreinterpretq_bf16_u16(vld1q_u16(k_base + 0));
    bfloat16x8_t b23 = vreinterpretq_bf16_u16(vld1q_u16(k_base + 8));
    bfloat16x8_t b45 = vreinterpretq_bf16_u16(vld1q_u16(k_base + 16));
    bfloat16x8_t b67 = vreinterpretq_bf16_u16(vld1q_u16(k_base + 24));

    bm00 = vbfmmlaq_f32(bm00, a01, b01);
    bm10 = vbfmmlaq_f32(bm10, a23, b01);
    bm20 = vbfmmlaq_f32(bm20, a45, b01);
    bm30 = vbfmmlaq_f32(bm30, a67, b01);
    bm01 = vbfmmlaq_f32(bm01, a01, b23);
    bm11 = vbfmmlaq_f32(bm11, a23, b23);
    bm21 = vbfmmlaq_f32(bm21, a45, b23);
    bm31 = vbfmmlaq_f32(bm31, a67, b23);
    bm02 = vbfmmlaq_f32(bm02, a01, b45);
    bm12 = vbfmmlaq_f32(bm12, a23, b45);
    bm22 = vbfmmlaq_f32(bm22, a45, b45);
    bm32 = vbfmmlaq_f32(bm32, a67, b45);
    bm03 = vbfmmlaq_f32(bm03, a01, b67);
    bm13 = vbfmmlaq_f32(bm13, a23, b67);
    bm23 = vbfmmlaq_f32(bm23, a45, b67);
    bm33 = vbfmmlaq_f32(bm33, a67, b67);
  }

  // BFMMLA 输出是 2×2 子块布局，重排为「行向 4 lane」便于 store。
  auto unzip_pair = [](float32x4_t lo_block, float32x4_t hi_block, float32x4_t* row_lo, float32x4_t* row_hi) {
    *row_lo = vcombine_f32(vget_low_f32(lo_block), vget_low_f32(hi_block));
    *row_hi = vcombine_f32(vget_high_f32(lo_block), vget_high_f32(hi_block));
  };
  unzip_pair(bm00, bm01, &c00, &c10);
  unzip_pair(bm02, bm03, &c01, &c11);
  unzip_pair(bm10, bm11, &c20, &c30);
  unzip_pair(bm12, bm13, &c21, &c31);
  unzip_pair(bm20, bm21, &c40, &c50);
  unzip_pair(bm22, bm23, &c41, &c51);
  unzip_pair(bm30, bm31, &c60, &c70);
  unzip_pair(bm32, bm33, &c61, &c71);
#endif  // HAS_BFMMLA

  // 标量 tail（E % 4 != 0），从原始 Q/K 读。
  if (e < E) {
    float scalar_acc[8 * 8];
    auto store_cij = [&](int i_row, float* dst) {
      float32x4_t lo, hi;
      switch (i_row) {
        case 0:
          lo = c00;
          hi = c01;
          break;
        case 1:
          lo = c10;
          hi = c11;
          break;
        case 2:
          lo = c20;
          hi = c21;
          break;
        case 3:
          lo = c30;
          hi = c31;
          break;
        case 4:
          lo = c40;
          hi = c41;
          break;
        case 5:
          lo = c50;
          hi = c51;
          break;
        case 6:
          lo = c60;
          hi = c61;
          break;
        case 7:
          lo = c70;
          hi = c71;
          break;
        default:
          lo = vdupq_n_f32(0);
          hi = vdupq_n_f32(0);
      }
      vst1q_f32(dst + 0, lo);
      vst1q_f32(dst + 4, hi);
    };
    auto load_cij = [&](int i_row, const float* src) {
      float32x4_t lo = vld1q_f32(src + 0);
      float32x4_t hi = vld1q_f32(src + 4);
      switch (i_row) {
        case 0:
          c00 = lo;
          c01 = hi;
          break;
        case 1:
          c10 = lo;
          c11 = hi;
          break;
        case 2:
          c20 = lo;
          c21 = hi;
          break;
        case 3:
          c30 = lo;
          c31 = hi;
          break;
        case 4:
          c40 = lo;
          c41 = hi;
          break;
        case 5:
          c50 = lo;
          c51 = hi;
          break;
        case 6:
          c60 = lo;
          c61 = hi;
          break;
        case 7:
          c70 = lo;
          c71 = hi;
          break;
      }
    };
    for (int i_row = 0; i_row < 8; ++i_row) store_cij(i_row, scalar_acc + i_row * 8);
    for (int64_t e2 = e; e2 < E; ++e2) {
      for (int i_row = 0; i_row < 8; ++i_row) {
        float qv = bf16_to_fp32_scalar(Q_orig[i_row * q_row_stride + e2]);
        for (int j_col = 0; j_col < 8; ++j_col) {
          float kv = bf16_to_fp32_scalar(K_orig[j_col * k_row_stride + e2]);
          scalar_acc[i_row * 8 + j_col] += qv * kv;
        }
      }
    }
    for (int i_row = 0; i_row < 8; ++i_row) load_cij(i_row, scalar_acc + i_row * 8);
  }

  const float32x4_t vs = vdupq_n_f32(scale);
  vst1q_f32(scores_buf + 0 * 8 + 0, vmulq_f32(c00, vs));
  vst1q_f32(scores_buf + 0 * 8 + 4, vmulq_f32(c01, vs));
  vst1q_f32(scores_buf + 1 * 8 + 0, vmulq_f32(c10, vs));
  vst1q_f32(scores_buf + 1 * 8 + 4, vmulq_f32(c11, vs));
  vst1q_f32(scores_buf + 2 * 8 + 0, vmulq_f32(c20, vs));
  vst1q_f32(scores_buf + 2 * 8 + 4, vmulq_f32(c21, vs));
  vst1q_f32(scores_buf + 3 * 8 + 0, vmulq_f32(c30, vs));
  vst1q_f32(scores_buf + 3 * 8 + 4, vmulq_f32(c31, vs));
  vst1q_f32(scores_buf + 4 * 8 + 0, vmulq_f32(c40, vs));
  vst1q_f32(scores_buf + 4 * 8 + 4, vmulq_f32(c41, vs));
  vst1q_f32(scores_buf + 5 * 8 + 0, vmulq_f32(c50, vs));
  vst1q_f32(scores_buf + 5 * 8 + 4, vmulq_f32(c51, vs));
  vst1q_f32(scores_buf + 6 * 8 + 0, vmulq_f32(c60, vs));
  vst1q_f32(scores_buf + 6 * 8 + 4, vmulq_f32(c61, vs));
  vst1q_f32(scores_buf + 7 * 8 + 0, vmulq_f32(c70, vs));
  vst1q_f32(scores_buf + 7 * 8 + 4, vmulq_f32(c71, vs));
}

// ── Microkernel 2：gemm_pv_microkernel_8x8_bf16_pquad ─────────────────────
// 计算 O[8][8] += P_hat[8][Sk] · V[Sk][8]
// 输入：
//   P_hat   fp32 [8][Sk]，softmax 概率
//   V       bf16 [Sk][8]
//   Sk      reduce 维
// 输出：
//   O       fp32 [8][8]，accumulate（不覆盖写）
//
// 设计：P 端每 4-k 段一次性 vld1q_f32 拿 P 行 quad（替代 8 个标量 ldr）；
// V 端 1 个 vld1q_u16 一次拿 8 个 bf16 + 2 个 vshll widen；用
// vfmaq_laneq_f32 在 P quad 内做 lane 索引；段 2 中段提前发跨迭代 V[k+4]
// 预取（方案 A 调度优化）。

static inline void gemm_pv_microkernel_8x8_bf16_pquad(const float* P_hat, int64_t P_row_stride, const uint16_t* V,
                                                      int64_t v_row_stride, int64_t Sk, float* O,
                                                      int64_t o_row_stride) {
  float32x4_t o00 = vld1q_f32(O + 0 * o_row_stride + 0);
  float32x4_t o01 = vld1q_f32(O + 0 * o_row_stride + 4);
  float32x4_t o10 = vld1q_f32(O + 1 * o_row_stride + 0);
  float32x4_t o11 = vld1q_f32(O + 1 * o_row_stride + 4);
  float32x4_t o20 = vld1q_f32(O + 2 * o_row_stride + 0);
  float32x4_t o21 = vld1q_f32(O + 2 * o_row_stride + 4);
  float32x4_t o30 = vld1q_f32(O + 3 * o_row_stride + 0);
  float32x4_t o31 = vld1q_f32(O + 3 * o_row_stride + 4);
  float32x4_t o40 = vld1q_f32(O + 4 * o_row_stride + 0);
  float32x4_t o41 = vld1q_f32(O + 4 * o_row_stride + 4);
  float32x4_t o50 = vld1q_f32(O + 5 * o_row_stride + 0);
  float32x4_t o51 = vld1q_f32(O + 5 * o_row_stride + 4);
  float32x4_t o60 = vld1q_f32(O + 6 * o_row_stride + 0);
  float32x4_t o61 = vld1q_f32(O + 6 * o_row_stride + 4);
  float32x4_t o70 = vld1q_f32(O + 7 * o_row_stride + 0);
  float32x4_t o71 = vld1q_f32(O + 7 * o_row_stride + 4);

  // 局部 helper：从 8 bf16（一行 V 8 列）同时产生 lo/hi 两个 fp32 quad。
  auto widen_row = [](uint16x8_t bf16x8, float32x4_t& v_lo, float32x4_t& v_hi) {
    v_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf16x8), 16));
    v_hi = vreinterpretq_f32_u32(vshll_high_n_u16(bf16x8, 16));
  };

  const int64_t Sk4 = Sk & ~int64_t{3};
  int64_t k = 0;
  if (Sk4 > 0) {
    uint16x8_t vbf0 = vld1q_u16(V + 0 * v_row_stride);
    float32x4_t v_lo0, v_hi0;
    widen_row(vbf0, v_lo0, v_hi0);
    for (; k < Sk4; k += 4) {
      // 一次性 load 8 个 P 行 quad
      float32x4_t p0 = vld1q_f32(P_hat + 0 * P_row_stride + k);
      float32x4_t p1 = vld1q_f32(P_hat + 1 * P_row_stride + k);
      float32x4_t p2 = vld1q_f32(P_hat + 2 * P_row_stride + k);
      float32x4_t p3 = vld1q_f32(P_hat + 3 * P_row_stride + k);
      float32x4_t p4 = vld1q_f32(P_hat + 4 * P_row_stride + k);
      float32x4_t p5 = vld1q_f32(P_hat + 5 * P_row_stride + k);
      float32x4_t p6 = vld1q_f32(P_hat + 6 * P_row_stride + k);
      float32x4_t p7 = vld1q_f32(P_hat + 7 * P_row_stride + k);

      // 段 0 (lane 0)：用 v_lo0/hi0；段尾预取段 1 V
      o00 = vfmaq_laneq_f32(o00, v_lo0, p0, 0);
      o01 = vfmaq_laneq_f32(o01, v_hi0, p0, 0);
      o10 = vfmaq_laneq_f32(o10, v_lo0, p1, 0);
      o11 = vfmaq_laneq_f32(o11, v_hi0, p1, 0);
      o20 = vfmaq_laneq_f32(o20, v_lo0, p2, 0);
      o21 = vfmaq_laneq_f32(o21, v_hi0, p2, 0);
      o30 = vfmaq_laneq_f32(o30, v_lo0, p3, 0);
      o31 = vfmaq_laneq_f32(o31, v_hi0, p3, 0);
      o40 = vfmaq_laneq_f32(o40, v_lo0, p4, 0);
      o41 = vfmaq_laneq_f32(o41, v_hi0, p4, 0);
      o50 = vfmaq_laneq_f32(o50, v_lo0, p5, 0);
      o51 = vfmaq_laneq_f32(o51, v_hi0, p5, 0);
      uint16x8_t vbf1 = vld1q_u16(V + (k + 1) * v_row_stride);
      float32x4_t v_lo1, v_hi1;
      widen_row(vbf1, v_lo1, v_hi1);
      o60 = vfmaq_laneq_f32(o60, v_lo0, p6, 0);
      o61 = vfmaq_laneq_f32(o61, v_hi0, p6, 0);
      o70 = vfmaq_laneq_f32(o70, v_lo0, p7, 0);
      o71 = vfmaq_laneq_f32(o71, v_hi0, p7, 0);

      // 段 1 (lane 1)：用 v_lo1/hi1；段尾预取段 2 V
      o00 = vfmaq_laneq_f32(o00, v_lo1, p0, 1);
      o01 = vfmaq_laneq_f32(o01, v_hi1, p0, 1);
      o10 = vfmaq_laneq_f32(o10, v_lo1, p1, 1);
      o11 = vfmaq_laneq_f32(o11, v_hi1, p1, 1);
      o20 = vfmaq_laneq_f32(o20, v_lo1, p2, 1);
      o21 = vfmaq_laneq_f32(o21, v_hi1, p2, 1);
      o30 = vfmaq_laneq_f32(o30, v_lo1, p3, 1);
      o31 = vfmaq_laneq_f32(o31, v_hi1, p3, 1);
      o40 = vfmaq_laneq_f32(o40, v_lo1, p4, 1);
      o41 = vfmaq_laneq_f32(o41, v_hi1, p4, 1);
      o50 = vfmaq_laneq_f32(o50, v_lo1, p5, 1);
      o51 = vfmaq_laneq_f32(o51, v_hi1, p5, 1);
      uint16x8_t vbf2 = vld1q_u16(V + (k + 2) * v_row_stride);
      float32x4_t v_lo2, v_hi2;
      widen_row(vbf2, v_lo2, v_hi2);
      o60 = vfmaq_laneq_f32(o60, v_lo1, p6, 1);
      o61 = vfmaq_laneq_f32(o61, v_hi1, p6, 1);
      o70 = vfmaq_laneq_f32(o70, v_lo1, p7, 1);
      o71 = vfmaq_laneq_f32(o71, v_hi1, p7, 1);

      // 段 2 (lane 2)：同时预取段 3 V 和跨迭代段 0 V（方案 A）
      o00 = vfmaq_laneq_f32(o00, v_lo2, p0, 2);
      o01 = vfmaq_laneq_f32(o01, v_hi2, p0, 2);
      o10 = vfmaq_laneq_f32(o10, v_lo2, p1, 2);
      o11 = vfmaq_laneq_f32(o11, v_hi2, p1, 2);
      o20 = vfmaq_laneq_f32(o20, v_lo2, p2, 2);
      o21 = vfmaq_laneq_f32(o21, v_hi2, p2, 2);
      if (k + 4 < Sk4) {
        uint16x8_t vbf0_next = vld1q_u16(V + (k + 4) * v_row_stride);
        widen_row(vbf0_next, v_lo0, v_hi0);
      }
      o30 = vfmaq_laneq_f32(o30, v_lo2, p3, 2);
      o31 = vfmaq_laneq_f32(o31, v_hi2, p3, 2);
      o40 = vfmaq_laneq_f32(o40, v_lo2, p4, 2);
      o41 = vfmaq_laneq_f32(o41, v_hi2, p4, 2);
      o50 = vfmaq_laneq_f32(o50, v_lo2, p5, 2);
      o51 = vfmaq_laneq_f32(o51, v_hi2, p5, 2);
      uint16x8_t vbf3 = vld1q_u16(V + (k + 3) * v_row_stride);
      float32x4_t v_lo3, v_hi3;
      widen_row(vbf3, v_lo3, v_hi3);
      o60 = vfmaq_laneq_f32(o60, v_lo2, p6, 2);
      o61 = vfmaq_laneq_f32(o61, v_hi2, p6, 2);
      o70 = vfmaq_laneq_f32(o70, v_lo2, p7, 2);
      o71 = vfmaq_laneq_f32(o71, v_hi2, p7, 2);

      // 段 3 (lane 3)：纯 FMA，无 V load
      o00 = vfmaq_laneq_f32(o00, v_lo3, p0, 3);
      o01 = vfmaq_laneq_f32(o01, v_hi3, p0, 3);
      o10 = vfmaq_laneq_f32(o10, v_lo3, p1, 3);
      o11 = vfmaq_laneq_f32(o11, v_hi3, p1, 3);
      o20 = vfmaq_laneq_f32(o20, v_lo3, p2, 3);
      o21 = vfmaq_laneq_f32(o21, v_hi3, p2, 3);
      o30 = vfmaq_laneq_f32(o30, v_lo3, p3, 3);
      o31 = vfmaq_laneq_f32(o31, v_hi3, p3, 3);
      o40 = vfmaq_laneq_f32(o40, v_lo3, p4, 3);
      o41 = vfmaq_laneq_f32(o41, v_hi3, p4, 3);
      o50 = vfmaq_laneq_f32(o50, v_lo3, p5, 3);
      o51 = vfmaq_laneq_f32(o51, v_hi3, p5, 3);
      o60 = vfmaq_laneq_f32(o60, v_lo3, p6, 3);
      o61 = vfmaq_laneq_f32(o61, v_hi3, p6, 3);
      o70 = vfmaq_laneq_f32(o70, v_lo3, p7, 3);
      o71 = vfmaq_laneq_f32(o71, v_hi3, p7, 3);
    }
  }

  // 标量尾循环：处理 Sk % 4
  for (; k < Sk; ++k) {
    float32x4_t v_lo = widen_bf16x4_to_fp32(V + k * v_row_stride + 0);
    float32x4_t v_hi = widen_bf16x4_to_fp32(V + k * v_row_stride + 4);
    float p0 = P_hat[0 * P_row_stride + k];
    float p1 = P_hat[1 * P_row_stride + k];
    float p2 = P_hat[2 * P_row_stride + k];
    float p3 = P_hat[3 * P_row_stride + k];
    float p4 = P_hat[4 * P_row_stride + k];
    float p5 = P_hat[5 * P_row_stride + k];
    float p6 = P_hat[6 * P_row_stride + k];
    float p7 = P_hat[7 * P_row_stride + k];
    o00 = vfmaq_n_f32(o00, v_lo, p0);
    o01 = vfmaq_n_f32(o01, v_hi, p0);
    o10 = vfmaq_n_f32(o10, v_lo, p1);
    o11 = vfmaq_n_f32(o11, v_hi, p1);
    o20 = vfmaq_n_f32(o20, v_lo, p2);
    o21 = vfmaq_n_f32(o21, v_hi, p2);
    o30 = vfmaq_n_f32(o30, v_lo, p3);
    o31 = vfmaq_n_f32(o31, v_hi, p3);
    o40 = vfmaq_n_f32(o40, v_lo, p4);
    o41 = vfmaq_n_f32(o41, v_hi, p4);
    o50 = vfmaq_n_f32(o50, v_lo, p5);
    o51 = vfmaq_n_f32(o51, v_hi, p5);
    o60 = vfmaq_n_f32(o60, v_lo, p6);
    o61 = vfmaq_n_f32(o61, v_hi, p6);
    o70 = vfmaq_n_f32(o70, v_lo, p7);
    o71 = vfmaq_n_f32(o71, v_hi, p7);
  }

  vst1q_f32(O + 0 * o_row_stride + 0, o00);
  vst1q_f32(O + 0 * o_row_stride + 4, o01);
  vst1q_f32(O + 1 * o_row_stride + 0, o10);
  vst1q_f32(O + 1 * o_row_stride + 4, o11);
  vst1q_f32(O + 2 * o_row_stride + 0, o20);
  vst1q_f32(O + 2 * o_row_stride + 4, o21);
  vst1q_f32(O + 3 * o_row_stride + 0, o30);
  vst1q_f32(O + 3 * o_row_stride + 4, o31);
  vst1q_f32(O + 4 * o_row_stride + 0, o40);
  vst1q_f32(O + 4 * o_row_stride + 4, o41);
  vst1q_f32(O + 5 * o_row_stride + 0, o50);
  vst1q_f32(O + 5 * o_row_stride + 4, o51);
  vst1q_f32(O + 6 * o_row_stride + 0, o60);
  vst1q_f32(O + 6 * o_row_stride + 4, o61);
  vst1q_f32(O + 7 * o_row_stride + 0, o70);
  vst1q_f32(O + 7 * o_row_stride + 4, o71);
}
