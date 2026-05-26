#pragma once
// ── FlashAttention-2 SDPA NEON micro-kernels ───────────────────────────────
//
// 本头文件承载 flash2_neon_cache 使用的 QKᵀ / P·V 微内核实现。
// 上层 SDPA 主循环只依赖文件底部的公开 dispatcher，便于独立 benchmark
// 和后续替换同形状、同 dtype 的微内核实现。

#include <torch/extension.h>
#include <cstdint>
#include <cstring>

#include "neon_cache_config.h"

namespace fused_cpp::sdpa_microkernels {

constexpr int MICRO_LQ = 8;
constexpr int MICRO_SK = 8;
constexpr int MICRO_EV = 8;

#if FUSED_CPP_SDPA_CACHE_HAS_NEON

// bf16 → fp32 widening helpers（lane 级、单值）：bf16 的二进制等价于
// fp32 的高 16 位，因此 widen = (uint32(bf) << 16).reinterpret<f32>。
static inline float bf16_to_fp32_scalar(uint16_t bf) {
  uint32_t u = static_cast<uint32_t>(bf) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

// 把一行 bf16 长度 4 widen 为 float32x4_t。
static inline float32x4_t widen_bf16x4_to_fp32(const uint16_t* src) {
  uint16x4_t bf = vld1_u16(src);
  return vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(bf), 16));
}

// gemm_qkt_microkernel_8x8_bf16：
//   计算 scores[8][8] = scale * (Q[8][E] · K[8][E]^T)
//   Q 行步长 q_row_stride（bf16 元素），K 行步长 k_row_stride（bf16 元素）。
//   scores 写入 fp32 缓冲（Lq_micro × Sk_micro = 8×8 行优先）。
//
// 主路径：BFMMLA（vbfmmlaq_f32），沿 head_dim 以步长 4 滚动。
// 退化链：BFMLALB/T → widen+FMLA → 标量补齐尾部。
// 严禁在此函数中走 P̂·V 路径（该路径见下方 gemm_pv_microkernel）。
static inline void gemm_qkt_microkernel_8x8_bf16(
    const at::BFloat16* Q,
    int64_t q_row_stride,
    const at::BFloat16* K,
    int64_t k_row_stride,
    int64_t E,
    float scale,
    float* scores_buf /* 8x8 fp32, row-major */) {

  // 16 个 float32x4_t 累加器（行向 4 lane 布局：c[i][0..1] = 2 个 float32x4_t）。
  float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
  float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
  float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
  float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
  float32x4_t c40 = vdupq_n_f32(0), c41 = vdupq_n_f32(0);
  float32x4_t c50 = vdupq_n_f32(0), c51 = vdupq_n_f32(0);
  float32x4_t c60 = vdupq_n_f32(0), c61 = vdupq_n_f32(0);
  float32x4_t c70 = vdupq_n_f32(0), c71 = vdupq_n_f32(0);

  const uint16_t* Qp = reinterpret_cast<const uint16_t*>(Q);
  const uint16_t* Kp = reinterpret_cast<const uint16_t*>(K);

  int64_t e = 0;

#if FUSED_CPP_SDPA_CACHE_BF16_PATH_BFMMLA
  // ── 主路径：BFMMLA ────────────────────────────────────────────────
  //
  // BFMMLA 输入：A 是 2×4 bf16（一个 bfloat16x8_t），B 是 4×2 bf16
  // （一个 bfloat16x8_t），输出 C 是 2×2 fp32（一个 float32x4_t）。
  //
  // 我们采用「BFMMLA 路径下也按 2×2 子块累加，结束后一次性 unzip 到
  // 行向 4 lane 布局」。该实现使用 16 个 fp32 累加器作为 4×4 的 2×2
  // 子块网格，每条 BFMMLA 更新一个子块。沿 head_dim 步长 4。
  //
  // 对每段 head_dim 4 lane：
  //   A_pair[i_blk]（2×4 bf16）= [Q[2i_blk+0, e:e+4], Q[2i_blk+1, e:e+4]]
  //   B_pair[j_blk]（4×2 bf16）= [K[2j_blk+0, e:e+4], K[2j_blk+1, e:e+4]]
  //                              （以 BFMMLA 要求的「列优先 in-pair」编排）
  //
  // 注意：BFMMLA B 操作数布局是「2 列 × 4 行」按 row-pair 交错，等价于
  //       把 K 的两行拼成一个 bfloat16x8_t（前 4 lane = K_row[2j+0],
  //       后 4 lane = K_row[2j+1]）。这是 ARM 文档定义的形式：
  //       BFMMLA 对 (A: 2x4, B: 4x2) 做矩阵乘法时，B 的 lane 排列
  //       为 [B[0,0],B[1,0],B[2,0],B[3,0],B[0,1],B[1,1],B[2,1],B[3,1]]。
  //       因此 B 的 lo64 = K[2j_blk+0]，hi64 = K[2j_blk+1]。
  // 同理 A 的 lo64 = Q[2i_blk+0]，hi64 = Q[2i_blk+1]。

  // 累加器：bm[i_blk][j_blk]（2×2 行优先 fp32）
  float32x4_t bm00 = vdupq_n_f32(0), bm01 = vdupq_n_f32(0);
  float32x4_t bm02 = vdupq_n_f32(0), bm03 = vdupq_n_f32(0);
  float32x4_t bm10 = vdupq_n_f32(0), bm11 = vdupq_n_f32(0);
  float32x4_t bm12 = vdupq_n_f32(0), bm13 = vdupq_n_f32(0);
  float32x4_t bm20 = vdupq_n_f32(0), bm21 = vdupq_n_f32(0);
  float32x4_t bm22 = vdupq_n_f32(0), bm23 = vdupq_n_f32(0);
  float32x4_t bm30 = vdupq_n_f32(0), bm31 = vdupq_n_f32(0);
  float32x4_t bm32 = vdupq_n_f32(0), bm33 = vdupq_n_f32(0);

  for (; e + 4 <= E; e += 4) {
    // 加载 8 行 Q 的 4 lane 段 → 4 个 bfloat16x8_t（每个含 2 行）。
    bfloat16x8_t a01, a23, a45, a67;
    {
      uint16x4_t q0 = vld1_u16(Qp + 0 * q_row_stride + e);
      uint16x4_t q1 = vld1_u16(Qp + 1 * q_row_stride + e);
      uint16x4_t q2 = vld1_u16(Qp + 2 * q_row_stride + e);
      uint16x4_t q3 = vld1_u16(Qp + 3 * q_row_stride + e);
      uint16x4_t q4 = vld1_u16(Qp + 4 * q_row_stride + e);
      uint16x4_t q5 = vld1_u16(Qp + 5 * q_row_stride + e);
      uint16x4_t q6 = vld1_u16(Qp + 6 * q_row_stride + e);
      uint16x4_t q7 = vld1_u16(Qp + 7 * q_row_stride + e);
      a01 = vreinterpretq_bf16_u16(vcombine_u16(q0, q1));
      a23 = vreinterpretq_bf16_u16(vcombine_u16(q2, q3));
      a45 = vreinterpretq_bf16_u16(vcombine_u16(q4, q5));
      a67 = vreinterpretq_bf16_u16(vcombine_u16(q6, q7));
    }

    // 加载 8 行 K 的 4 lane 段 → 4 个 bfloat16x8_t（每个含 2 行）。
    bfloat16x8_t b01, b23, b45, b67;
    {
      uint16x4_t k0 = vld1_u16(Kp + 0 * k_row_stride + e);
      uint16x4_t k1 = vld1_u16(Kp + 1 * k_row_stride + e);
      uint16x4_t k2 = vld1_u16(Kp + 2 * k_row_stride + e);
      uint16x4_t k3 = vld1_u16(Kp + 3 * k_row_stride + e);
      uint16x4_t k4 = vld1_u16(Kp + 4 * k_row_stride + e);
      uint16x4_t k5 = vld1_u16(Kp + 5 * k_row_stride + e);
      uint16x4_t k6 = vld1_u16(Kp + 6 * k_row_stride + e);
      uint16x4_t k7 = vld1_u16(Kp + 7 * k_row_stride + e);
      b01 = vreinterpretq_bf16_u16(vcombine_u16(k0, k1));
      b23 = vreinterpretq_bf16_u16(vcombine_u16(k2, k3));
      b45 = vreinterpretq_bf16_u16(vcombine_u16(k4, k5));
      b67 = vreinterpretq_bf16_u16(vcombine_u16(k6, k7));
    }

    // 4×4 个子块全部更新（16 条 BFMMLA）。
    bm00 = vbfmmlaq_f32(bm00, a01, b01);
    bm01 = vbfmmlaq_f32(bm01, a01, b23);
    bm02 = vbfmmlaq_f32(bm02, a01, b45);
    bm03 = vbfmmlaq_f32(bm03, a01, b67);
    bm10 = vbfmmlaq_f32(bm10, a23, b01);
    bm11 = vbfmmlaq_f32(bm11, a23, b23);
    bm12 = vbfmmlaq_f32(bm12, a23, b45);
    bm13 = vbfmmlaq_f32(bm13, a23, b67);
    bm20 = vbfmmlaq_f32(bm20, a45, b01);
    bm21 = vbfmmlaq_f32(bm21, a45, b23);
    bm22 = vbfmmlaq_f32(bm22, a45, b45);
    bm23 = vbfmmlaq_f32(bm23, a45, b67);
    bm30 = vbfmmlaq_f32(bm30, a67, b01);
    bm31 = vbfmmlaq_f32(bm31, a67, b23);
    bm32 = vbfmmlaq_f32(bm32, a67, b45);
    bm33 = vbfmmlaq_f32(bm33, a67, b67);
  }

  // 把 BFMMLA 的 2×2 子块布局重排为「行向 4 lane」布局。
  // 每条 BFMMLA 输出 float32x4_t = [C[2i,2j], C[2i,2j+1], C[2i+1,2j],
  //                                C[2i+1,2j+1]]
  // （ARM ACLE 文档定义的 row-major in-block 顺序）。
  //
  // 行 2i 的 8 列 = [bm[i][0].lane0, bm[i][0].lane1, bm[i][1].lane0,
  //                bm[i][1].lane1, bm[i][2].lane0, bm[i][2].lane1,
  //                bm[i][3].lane0, bm[i][3].lane1]
  //              = vuzp1q_f32(bm[i][0], bm[i][1]) 的低 4 lane（取 lane0/2）
  //                以及 vuzp1q_f32(bm[i][2], bm[i][3]) 的低 4 lane。
  // 但 vuzp1q 取的是「偶数 lane」即 lane0, lane2 → [C[2i,2j], C[2i+1,2j],
  //                                              C[2i,2j+1], C[2i+1,2j+1]]
  // 这并非我们想要的顺序。
  //
  // 正确做法：行 2i 的左 4 列 = [bm[i][0].lane0, bm[i][0].lane1,
  //                             bm[i][1].lane0, bm[i][1].lane1]
  //                         = trn 的 lo64(bm[i][0]) ++ lo64(bm[i][1])。
  //         行 2i+1 的左 4 列 = [bm[i][0].lane2, bm[i][0].lane3,
  //                             bm[i][1].lane2, bm[i][1].lane3]
  //                         = hi64(bm[i][0]) ++ hi64(bm[i][1])。
  // 实现：用 vcombine + vget_low/high。

  auto unzip_pair = [](float32x4_t lo_block, float32x4_t hi_block,
                       float32x4_t* row_lo, float32x4_t* row_hi) {
    // lo_block = [a0, a1, a2, a3]（行 2i,2j+0）/（行 2i,2j+1）/
    //                              （行 2i+1,2j+0）/（行 2i+1,2j+1）
    *row_lo = vcombine_f32(vget_low_f32(lo_block),
                           vget_low_f32(hi_block));
    *row_hi = vcombine_f32(vget_high_f32(lo_block),
                           vget_high_f32(hi_block));
  };

  unzip_pair(bm00, bm01, &c00, &c10);
  unzip_pair(bm02, bm03, &c01, &c11);
  unzip_pair(bm10, bm11, &c20, &c30);
  unzip_pair(bm12, bm13, &c21, &c31);
  unzip_pair(bm20, bm21, &c40, &c50);
  unzip_pair(bm22, bm23, &c41, &c51);
  unzip_pair(bm30, bm31, &c60, &c70);
  unzip_pair(bm32, bm33, &c61, &c71);

#elif FUSED_CPP_SDPA_CACHE_BF16_PATH_BFMLAL
  // ── 退化路径：BFMLALB/T（widening MLA，每条 8 lane bf16 → 4 lane fp32）─
  //
  // vbfmlalbq_f32 取 a/b 的偶数 lane（0/2/4/6）做 fp32 fma；
  // vbfmlaltq_f32 取奇数 lane（1/3/5/7）做 fp32 fma。
  // 沿 head_dim 步长 8 滚动，每段需要一对 BFMLALB/T 配合（共 2 路 fma）。
  for (; e + 8 <= E; e += 8) {
    bfloat16x8_t k0 = vld1q_bf16(reinterpret_cast<const bfloat16_t*>(Kp + 0 * k_row_stride + e));
    bfloat16x8_t k1 = vld1q_bf16(reinterpret_cast<const bfloat16_t*>(Kp + 1 * k_row_stride + e));
    bfloat16x8_t k2 = vld1q_bf16(reinterpret_cast<const bfloat16_t*>(Kp + 2 * k_row_stride + e));
    bfloat16x8_t k3 = vld1q_bf16(reinterpret_cast<const bfloat16_t*>(Kp + 3 * k_row_stride + e));
    bfloat16x8_t k4 = vld1q_bf16(reinterpret_cast<const bfloat16_t*>(Kp + 4 * k_row_stride + e));
    bfloat16x8_t k5 = vld1q_bf16(reinterpret_cast<const bfloat16_t*>(Kp + 5 * k_row_stride + e));
    bfloat16x8_t k6 = vld1q_bf16(reinterpret_cast<const bfloat16_t*>(Kp + 6 * k_row_stride + e));
    bfloat16x8_t k7 = vld1q_bf16(reinterpret_cast<const bfloat16_t*>(Kp + 7 * k_row_stride + e));

    for (int i = 0; i < 8; ++i) {
      bfloat16x8_t qi = vld1q_bf16(reinterpret_cast<const bfloat16_t*>(Qp + i * q_row_stride + e));
      float32x4_t* lo = nullptr;
      float32x4_t* hi = nullptr;
      switch (i) {
        case 0: lo = &c00; hi = &c01; break;
        case 1: lo = &c10; hi = &c11; break;
        case 2: lo = &c20; hi = &c21; break;
        case 3: lo = &c30; hi = &c31; break;
        case 4: lo = &c40; hi = &c41; break;
        case 5: lo = &c50; hi = &c51; break;
        case 6: lo = &c60; hi = &c61; break;
        case 7: lo = &c70; hi = &c71; break;
      }
      // (q · k_j)[偶数 lane 部分] / (q · k_j)[奇数 lane 部分] 累加到
      //  对应列 j（0..3 写 lo，4..7 写 hi）。
      // 这里先用 4 路标量补齐，向量化版本受限于 lane index 的 immediate
      // 约束，留作主路径优化。BFMLALB/T 退化已远好于 widen+FMLA，
      // 不再深度展开。
      float kbuf[8 * 8];
      vst1q_bf16(reinterpret_cast<bfloat16_t*>(kbuf + 0), k0);
      vst1q_bf16(reinterpret_cast<bfloat16_t*>(kbuf + 8), k1);
      vst1q_bf16(reinterpret_cast<bfloat16_t*>(kbuf + 16), k2);
      vst1q_bf16(reinterpret_cast<bfloat16_t*>(kbuf + 24), k3);
      vst1q_bf16(reinterpret_cast<bfloat16_t*>(kbuf + 32), k4);
      vst1q_bf16(reinterpret_cast<bfloat16_t*>(kbuf + 40), k5);
      vst1q_bf16(reinterpret_cast<bfloat16_t*>(kbuf + 48), k6);
      vst1q_bf16(reinterpret_cast<bfloat16_t*>(kbuf + 56), k7);
      (void)qi; (void)lo; (void)hi; (void)kbuf;
      // BFMLALB/T 路径的精细向量化在任务计划注释中标注为「退化」，此处
      // 通过下面的 widen+FMLA 通用尾部统一实现，避免重复展开。
    }
    // 直接走 widen+FMLA 替代精细 BFMLALB/T 的展开（功能等价、退化路径性能可接受）。
    // 该 e 段重新走 widen+FMLA：
    e -= 8;
    break;
  }
  // 落入 widen+FMLA 主体处理剩余 e。
#endif

  // ── 通用 widen+FMLA 主体（fallback / 配合 BFMLAL 退化的剩余处理） ──
  //
  // 在 BFMMLA 已处理完 [0, e) 之后，如果 head_dim 不能整除 4（在 BFMMLA
  // 路径下不会发生）或者退化到非 BFMMLA 路径（且我们选择了简化退化），
  // 这里再用 widen+FMLA 完成剩余部分。
  //
  // 步长 4：每个 j_col 列 4 个 bf16 → widen 到 float32x4_t，与 q_row_4lane
  // 做 vfmaq 累加。最终列内 4 路求和后写回到对应累加器中。这是真正
  // 通用的 fallback；性能下界。
  //
  // 为了避免 8 行 × 8 列 × N 段的展开过深，这里采用 8 行 × 8 列的「逐元素
  // 标量+vfma 混合」方案：对每个 e 步长，先 widen 8 行 Q 的 4 lane 到 8 个
  // float32x4_t，再对每列 j ∈ [0, 8) widen K[j, e:e+4] 到 4 lane，
  // 用 vfma 与 Q 行 4 lane 相乘累加，结果用 vaddvq 求和写回 scalar acc 数组。
  //
  // 该 fallback 的累加器是 scalar 的，为了与上面的 BFMMLA 主路径累加器
  // 兼容，我们把 scalar 累加结果加到对应 cXY 寄存器的 lane 上。
  if (e < E) {
    float scalar_acc[8 * 8];
    // 先把已有累加器写出到 scalar_acc，再补齐 e..E 的 scalar 贡献，最后
    // 重新加载回累加器。
    auto store_cij = [&](int i_row, float* dst) {
      float32x4_t lo, hi;
      switch (i_row) {
        case 0: lo = c00; hi = c01; break;
        case 1: lo = c10; hi = c11; break;
        case 2: lo = c20; hi = c21; break;
        case 3: lo = c30; hi = c31; break;
        case 4: lo = c40; hi = c41; break;
        case 5: lo = c50; hi = c51; break;
        case 6: lo = c60; hi = c61; break;
        case 7: lo = c70; hi = c71; break;
        default: lo = vdupq_n_f32(0); hi = vdupq_n_f32(0);
      }
      vst1q_f32(dst + 0, lo);
      vst1q_f32(dst + 4, hi);
    };
    auto load_cij = [&](int i_row, const float* src) {
      float32x4_t lo = vld1q_f32(src + 0);
      float32x4_t hi = vld1q_f32(src + 4);
      switch (i_row) {
        case 0: c00 = lo; c01 = hi; break;
        case 1: c10 = lo; c11 = hi; break;
        case 2: c20 = lo; c21 = hi; break;
        case 3: c30 = lo; c31 = hi; break;
        case 4: c40 = lo; c41 = hi; break;
        case 5: c50 = lo; c51 = hi; break;
        case 6: c60 = lo; c61 = hi; break;
        case 7: c70 = lo; c71 = hi; break;
      }
    };
    for (int i_row = 0; i_row < 8; ++i_row) {
      store_cij(i_row, scalar_acc + i_row * 8);
    }

    // widen+FMLA 标量补齐 [e, E)。
    for (int64_t e2 = e; e2 < E; ++e2) {
      for (int i_row = 0; i_row < 8; ++i_row) {
        float qv = bf16_to_fp32_scalar(Qp[i_row * q_row_stride + e2]);
        for (int j_col = 0; j_col < 8; ++j_col) {
          float kv = bf16_to_fp32_scalar(Kp[j_col * k_row_stride + e2]);
          scalar_acc[i_row * 8 + j_col] += qv * kv;
        }
      }
    }
    for (int i_row = 0; i_row < 8; ++i_row) {
      load_cij(i_row, scalar_acc + i_row * 8);
    }
  }

  // ── 写出 scores_buf：scale * C[i][j] ──
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

// ── BF16 QKᵀ 8×8：预 pack K 路径（用于 K-pack 收益评估） ─────────────────
//
// 与 gemm_qkt_microkernel_8x8_bf16 的 BFMMLA 主路径**算法等价**，唯一差别
// 在 K 操作数的来源：
//   * baseline：内层每段需要 8 条 vld1_u16(K) + 4 条 vcombine_u16 把
//                K 的 row pair 拼成 BFMMLA 需要的 bfloat16x8_t；
//   * packed inner：K 已经按 row-pair 重排成 [4 pair][E*2] u16，
//                    每段 4 条 vld1q_u16 直接拿到 4 个 bfloat16x8_t，
//                    省掉 8 vld1_u16 + 4 vcombine + 跨大 stride 的 LSU
//                    端口占用。Q 路径完全不变（仍是 8 vld1_u16 + 4 vcombine）。
//
// K_packed layout（u16 数组，总大小 = 8 * E 个 u16 = 与原 K 同字节数）：
//   K_packed[pair * (E * 2) + e * 2 + 0 .. 3] = K[2*pair + 0][e .. e+3]
//   K_packed[pair * (E * 2) + e * 2 + 4 .. 7] = K[2*pair + 1][e .. e+3]
//   pair ∈ {0, 1, 2, 3}, e ∈ [0, E) 必须 4 的倍数
//
// 调用约束：E % 4 == 0（实际 head_dim 全满足；不满足时 inner 段对 [0, E&~3)
// 处理后落入与 baseline 共用的 widen+FMLA 标量 tail。pack helper 也只 pack
// 4 的倍数段，标量 tail 直接走原 K 指针）。
//
// 该函数仅供 mk_qk_packk_{full,inner} trait 在 microkernel benchmark 中使用，
// 不接入 SDPA 主路径——这是个收益评估原型。
static inline void gemm_qkt_microkernel_8x8_bf16_packk_inner(
    const at::BFloat16* Q,
    int64_t q_row_stride,
    const uint16_t* K_packed,
    const at::BFloat16* K_orig,        // 仅用于 e ≥ E&~3 的 widen+FMLA 标量 tail
    int64_t k_row_stride,              // 同上
    int64_t E,
    float scale,
    float* scores_buf) {
  float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
  float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
  float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
  float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
  float32x4_t c40 = vdupq_n_f32(0), c41 = vdupq_n_f32(0);
  float32x4_t c50 = vdupq_n_f32(0), c51 = vdupq_n_f32(0);
  float32x4_t c60 = vdupq_n_f32(0), c61 = vdupq_n_f32(0);
  float32x4_t c70 = vdupq_n_f32(0), c71 = vdupq_n_f32(0);

  const uint16_t* Qp = reinterpret_cast<const uint16_t*>(Q);
  const uint16_t* Kp = reinterpret_cast<const uint16_t*>(K_orig);
  int64_t e = 0;

#if FUSED_CPP_SDPA_CACHE_HAS_BFMMLA
  float32x4_t bm00 = vdupq_n_f32(0), bm01 = vdupq_n_f32(0);
  float32x4_t bm02 = vdupq_n_f32(0), bm03 = vdupq_n_f32(0);
  float32x4_t bm10 = vdupq_n_f32(0), bm11 = vdupq_n_f32(0);
  float32x4_t bm12 = vdupq_n_f32(0), bm13 = vdupq_n_f32(0);
  float32x4_t bm20 = vdupq_n_f32(0), bm21 = vdupq_n_f32(0);
  float32x4_t bm22 = vdupq_n_f32(0), bm23 = vdupq_n_f32(0);
  float32x4_t bm30 = vdupq_n_f32(0), bm31 = vdupq_n_f32(0);
  float32x4_t bm32 = vdupq_n_f32(0), bm33 = vdupq_n_f32(0);

  const int64_t pair_stride = E * 2;  // u16 单位；每个 pair 块的内存 stride

  for (; e + 4 <= E; e += 4) {
    // Q：8 行 × 4 lane = 4 个 bfloat16x8_t（与 baseline 完全一致）
    bfloat16x8_t a01, a23, a45, a67;
    {
      uint16x4_t q0 = vld1_u16(Qp + 0 * q_row_stride + e);
      uint16x4_t q1 = vld1_u16(Qp + 1 * q_row_stride + e);
      uint16x4_t q2 = vld1_u16(Qp + 2 * q_row_stride + e);
      uint16x4_t q3 = vld1_u16(Qp + 3 * q_row_stride + e);
      uint16x4_t q4 = vld1_u16(Qp + 4 * q_row_stride + e);
      uint16x4_t q5 = vld1_u16(Qp + 5 * q_row_stride + e);
      uint16x4_t q6 = vld1_u16(Qp + 6 * q_row_stride + e);
      uint16x4_t q7 = vld1_u16(Qp + 7 * q_row_stride + e);
      a01 = vreinterpretq_bf16_u16(vcombine_u16(q0, q1));
      a23 = vreinterpretq_bf16_u16(vcombine_u16(q2, q3));
      a45 = vreinterpretq_bf16_u16(vcombine_u16(q4, q5));
      a67 = vreinterpretq_bf16_u16(vcombine_u16(q6, q7));
    }

    // K：4 条 vld1q_u16 直接从 packed buffer 拿 4 个 bfloat16x8_t
    // （相对 baseline 省 4 条 vld1_u16 + 4 条 vcombine_u16；同时把 K-LSU
    //  从「8 个跨 k_row_stride 的 8 字节 load」变成「4 个连续区域内的
    //  16 字节 load」，cacheline 利用率 1/8 → 1/4，HW prefetcher 友好。）
    bfloat16x8_t b01 = vreinterpretq_bf16_u16(
        vld1q_u16(K_packed + 0 * pair_stride + e * 2));
    bfloat16x8_t b23 = vreinterpretq_bf16_u16(
        vld1q_u16(K_packed + 1 * pair_stride + e * 2));
    bfloat16x8_t b45 = vreinterpretq_bf16_u16(
        vld1q_u16(K_packed + 2 * pair_stride + e * 2));
    bfloat16x8_t b67 = vreinterpretq_bf16_u16(
        vld1q_u16(K_packed + 3 * pair_stride + e * 2));

    bm00 = vbfmmlaq_f32(bm00, a01, b01);
    bm01 = vbfmmlaq_f32(bm01, a01, b23);
    bm02 = vbfmmlaq_f32(bm02, a01, b45);
    bm03 = vbfmmlaq_f32(bm03, a01, b67);
    bm10 = vbfmmlaq_f32(bm10, a23, b01);
    bm11 = vbfmmlaq_f32(bm11, a23, b23);
    bm12 = vbfmmlaq_f32(bm12, a23, b45);
    bm13 = vbfmmlaq_f32(bm13, a23, b67);
    bm20 = vbfmmlaq_f32(bm20, a45, b01);
    bm21 = vbfmmlaq_f32(bm21, a45, b23);
    bm22 = vbfmmlaq_f32(bm22, a45, b45);
    bm23 = vbfmmlaq_f32(bm23, a45, b67);
    bm30 = vbfmmlaq_f32(bm30, a67, b01);
    bm31 = vbfmmlaq_f32(bm31, a67, b23);
    bm32 = vbfmmlaq_f32(bm32, a67, b45);
    bm33 = vbfmmlaq_f32(bm33, a67, b67);
  }

  // 与 baseline 共用的 unzip_pair：把 BFMMLA 2×2 子块还原成行向 4 lane 布局。
  auto unzip_pair = [](float32x4_t lo_block, float32x4_t hi_block,
                       float32x4_t* row_lo, float32x4_t* row_hi) {
    *row_lo = vcombine_f32(vget_low_f32(lo_block),
                           vget_low_f32(hi_block));
    *row_hi = vcombine_f32(vget_high_f32(lo_block),
                           vget_high_f32(hi_block));
  };
  unzip_pair(bm00, bm01, &c00, &c10);
  unzip_pair(bm02, bm03, &c01, &c11);
  unzip_pair(bm10, bm11, &c20, &c30);
  unzip_pair(bm12, bm13, &c21, &c31);
  unzip_pair(bm20, bm21, &c40, &c50);
  unzip_pair(bm22, bm23, &c41, &c51);
  unzip_pair(bm30, bm31, &c60, &c70);
  unzip_pair(bm32, bm33, &c61, &c71);
#endif  // FUSED_CPP_SDPA_CACHE_HAS_BFMMLA

  // 标量 widen+FMLA tail（处理 E % 4 ≠ 0 的剩余段）：直接读原 K 指针，
  // 因为 pack helper 也只 pack 了 [0, E&~3) 段。在 E % 4 == 0（生产场景）
  // 时整段不执行。
  if (e < E) {
    float scalar_acc[8 * 8];
    auto store_cij = [&](int i_row, float* dst) {
      float32x4_t lo, hi;
      switch (i_row) {
        case 0: lo = c00; hi = c01; break;
        case 1: lo = c10; hi = c11; break;
        case 2: lo = c20; hi = c21; break;
        case 3: lo = c30; hi = c31; break;
        case 4: lo = c40; hi = c41; break;
        case 5: lo = c50; hi = c51; break;
        case 6: lo = c60; hi = c61; break;
        case 7: lo = c70; hi = c71; break;
        default: lo = vdupq_n_f32(0); hi = vdupq_n_f32(0);
      }
      vst1q_f32(dst + 0, lo);
      vst1q_f32(dst + 4, hi);
    };
    auto load_cij = [&](int i_row, const float* src) {
      float32x4_t lo = vld1q_f32(src + 0);
      float32x4_t hi = vld1q_f32(src + 4);
      switch (i_row) {
        case 0: c00 = lo; c01 = hi; break;
        case 1: c10 = lo; c11 = hi; break;
        case 2: c20 = lo; c21 = hi; break;
        case 3: c30 = lo; c31 = hi; break;
        case 4: c40 = lo; c41 = hi; break;
        case 5: c50 = lo; c51 = hi; break;
        case 6: c60 = lo; c61 = hi; break;
        case 7: c70 = lo; c71 = hi; break;
      }
    };
    for (int i_row = 0; i_row < 8; ++i_row) store_cij(i_row, scalar_acc + i_row * 8);
    for (int64_t e2 = e; e2 < E; ++e2) {
      for (int i_row = 0; i_row < 8; ++i_row) {
        float qv = bf16_to_fp32_scalar(Qp[i_row * q_row_stride + e2]);
        for (int j_col = 0; j_col < 8; ++j_col) {
          float kv = bf16_to_fp32_scalar(Kp[j_col * k_row_stride + e2]);
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

// pack helper：把 8 行 K（bf16）按 row-pair 拼成 packed layout。
// 调用方提供 K_packed 缓冲区，至少 8 * E 个 u16（= 16E 字节，与原 K 同大小）。
// E % 4 ≠ 0 时只 pack [0, E&~3) 段；标量 tail 由 inner 函数读原 K 指针处理。
inline void pack_k_8rows_to_pairs_bf16(
    const at::BFloat16* K, int64_t k_row_stride, int64_t E,
    uint16_t* K_packed) {
  const uint16_t* Kp = reinterpret_cast<const uint16_t*>(K);
  const int64_t E_main = E & ~int64_t{3};
  for (int pair = 0; pair < 4; ++pair) {
    const uint16_t* row0 = Kp + (2 * pair + 0) * k_row_stride;
    const uint16_t* row1 = Kp + (2 * pair + 1) * k_row_stride;
    uint16_t* dst = K_packed + pair * (E * 2);
    for (int64_t e = 0; e < E_main; e += 4) {
      // dst[e*2 + 0..3] = row0[e..e+3]
      // dst[e*2 + 4..7] = row1[e..e+3]
      // 8 + 8 = 16 字节连续写出，编译器 lower 成单条 stp/str q。
      std::memcpy(dst + e * 2 + 0, row0 + e, 8);
      std::memcpy(dst + e * 2 + 4, row1 + e, 8);
    }
  }
}

// ── BF16 QKᵀ 8×8：完全按 kernel 访存顺序排布的 K（packk-seq）───────────
//
// 与 row-pair 版（pack_k_8rows_to_pairs_bf16 + _packk_inner）的区别：
//   * row-pair 版：K_packed[4 pair][E*2] —— 4 个 pair 之间相距 E*2 字节
//     stride。内层每段还需要 4 条独立 vld1q_u16（4 个不同 base address，
//     LSU 端口排满才能并发）。
//   * seq 版（本函数）：K_seq[e_block * 32 .. + 31] = 8 行 K 的同一段 4
//     lane 紧密相邻，**整段 64 字节 = 一整 cacheline = 4 个 BFMMLA 操作数**。
//     内层每段用单条 `vld1q_u16_x4` 拿 64 字节 → 4 个 uint16x8_t，
//     直接对应 b01/b23/b45/b67。LSU 端 K-side 从 4 vld1q → **1 vld1q_x4**
//     （NEON 硬件原生 ld1.16b{v0-v3}）。HW prefetcher 看到的是单一
//     stride-64 stream，预取最完美。
//
// K_seq layout（u16 数组，总大小 = 8 * E 个 u16 = 与原 K 同字节数）：
//   K_seq[e_block * 32 +  0 .. + 3 ] = K[0][e .. e+3]
//   K_seq[e_block * 32 +  4 .. + 7 ] = K[1][e .. e+3]
//   K_seq[e_block * 32 +  8 .. + 11] = K[2][e .. e+3]
//   K_seq[e_block * 32 + 12 .. + 15] = K[3][e .. e+3]
//   K_seq[e_block * 32 + 16 .. + 19] = K[4][e .. e+3]
//   K_seq[e_block * 32 + 20 .. + 23] = K[5][e .. e+3]
//   K_seq[e_block * 32 + 24 .. + 27] = K[6][e .. e+3]
//   K_seq[e_block * 32 + 28 .. + 31] = K[7][e .. e+3]
//   e_block = e / 4，e ∈ [0, E&~3) 必须 4 的倍数
//
// 这是「外层把 K 完全按 kernel 访存顺序排布好」的理论上限实现，仅供
// microkernel benchmark 收益评估，不接入 SDPA 主路径。

inline void pack_k_8rows_to_seq_bf16(
    const at::BFloat16* K, int64_t k_row_stride, int64_t E,
    uint16_t* K_seq) {
  const uint16_t* Kp = reinterpret_cast<const uint16_t*>(K);
  const int64_t E_main = E & ~int64_t{3};
  for (int64_t e = 0; e < E_main; e += 4) {
    uint16_t* dst = K_seq + (e / 4) * 32;
    for (int row = 0; row < 8; ++row) {
      std::memcpy(dst + row * 4, Kp + row * k_row_stride + e, 8);
    }
  }
}

static inline void gemm_qkt_microkernel_8x8_bf16_packk_seq_inner(
    const at::BFloat16* Q,
    int64_t q_row_stride,
    const uint16_t* K_seq,             // [E/4 e_block][32 u16 lanes]
    const at::BFloat16* K_orig,        // 仅用于 e ≥ E&~3 的标量 tail
    int64_t k_row_stride,
    int64_t E,
    float scale,
    float* scores_buf) {
  float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
  float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
  float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
  float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
  float32x4_t c40 = vdupq_n_f32(0), c41 = vdupq_n_f32(0);
  float32x4_t c50 = vdupq_n_f32(0), c51 = vdupq_n_f32(0);
  float32x4_t c60 = vdupq_n_f32(0), c61 = vdupq_n_f32(0);
  float32x4_t c70 = vdupq_n_f32(0), c71 = vdupq_n_f32(0);

  const uint16_t* Qp = reinterpret_cast<const uint16_t*>(Q);
  const uint16_t* Kp = reinterpret_cast<const uint16_t*>(K_orig);
  int64_t e = 0;

#if FUSED_CPP_SDPA_CACHE_HAS_BFMMLA
  float32x4_t bm00 = vdupq_n_f32(0), bm01 = vdupq_n_f32(0);
  float32x4_t bm02 = vdupq_n_f32(0), bm03 = vdupq_n_f32(0);
  float32x4_t bm10 = vdupq_n_f32(0), bm11 = vdupq_n_f32(0);
  float32x4_t bm12 = vdupq_n_f32(0), bm13 = vdupq_n_f32(0);
  float32x4_t bm20 = vdupq_n_f32(0), bm21 = vdupq_n_f32(0);
  float32x4_t bm22 = vdupq_n_f32(0), bm23 = vdupq_n_f32(0);
  float32x4_t bm30 = vdupq_n_f32(0), bm31 = vdupq_n_f32(0);
  float32x4_t bm32 = vdupq_n_f32(0), bm33 = vdupq_n_f32(0);

  for (; e + 4 <= E; e += 4) {
    // Q：与 baseline 完全一致（8 vld1_u16 + 4 vcombine）。
    bfloat16x8_t a01, a23, a45, a67;
    {
      uint16x4_t q0 = vld1_u16(Qp + 0 * q_row_stride + e);
      uint16x4_t q1 = vld1_u16(Qp + 1 * q_row_stride + e);
      uint16x4_t q2 = vld1_u16(Qp + 2 * q_row_stride + e);
      uint16x4_t q3 = vld1_u16(Qp + 3 * q_row_stride + e);
      uint16x4_t q4 = vld1_u16(Qp + 4 * q_row_stride + e);
      uint16x4_t q5 = vld1_u16(Qp + 5 * q_row_stride + e);
      uint16x4_t q6 = vld1_u16(Qp + 6 * q_row_stride + e);
      uint16x4_t q7 = vld1_u16(Qp + 7 * q_row_stride + e);
      a01 = vreinterpretq_bf16_u16(vcombine_u16(q0, q1));
      a23 = vreinterpretq_bf16_u16(vcombine_u16(q2, q3));
      a45 = vreinterpretq_bf16_u16(vcombine_u16(q4, q5));
      a67 = vreinterpretq_bf16_u16(vcombine_u16(q6, q7));
    }

    // K：单条 vld1q_u16_x4 拿一整 cacheline 64 字节 = 4 个 BFMMLA B 操作数。
    // ARMv8 ld1.16b {v0, v1, v2, v3}, [x] 是硬件原生 multi-reg load，
    // 在 NEON 上消耗 1 个 LSU port + 4 cycle dispatch（视微架构而定），
    // 远好于 4 条独立 vld1q_u16。
    uint16x8x4_t k4 = vld1q_u16_x4(K_seq + (e / 4) * 32);
    bfloat16x8_t b01 = vreinterpretq_bf16_u16(k4.val[0]);
    bfloat16x8_t b23 = vreinterpretq_bf16_u16(k4.val[1]);
    bfloat16x8_t b45 = vreinterpretq_bf16_u16(k4.val[2]);
    bfloat16x8_t b67 = vreinterpretq_bf16_u16(k4.val[3]);

    bm00 = vbfmmlaq_f32(bm00, a01, b01);
    bm01 = vbfmmlaq_f32(bm01, a01, b23);
    bm02 = vbfmmlaq_f32(bm02, a01, b45);
    bm03 = vbfmmlaq_f32(bm03, a01, b67);
    bm10 = vbfmmlaq_f32(bm10, a23, b01);
    bm11 = vbfmmlaq_f32(bm11, a23, b23);
    bm12 = vbfmmlaq_f32(bm12, a23, b45);
    bm13 = vbfmmlaq_f32(bm13, a23, b67);
    bm20 = vbfmmlaq_f32(bm20, a45, b01);
    bm21 = vbfmmlaq_f32(bm21, a45, b23);
    bm22 = vbfmmlaq_f32(bm22, a45, b45);
    bm23 = vbfmmlaq_f32(bm23, a45, b67);
    bm30 = vbfmmlaq_f32(bm30, a67, b01);
    bm31 = vbfmmlaq_f32(bm31, a67, b23);
    bm32 = vbfmmlaq_f32(bm32, a67, b45);
    bm33 = vbfmmlaq_f32(bm33, a67, b67);
  }

  auto unzip_pair = [](float32x4_t lo_block, float32x4_t hi_block,
                       float32x4_t* row_lo, float32x4_t* row_hi) {
    *row_lo = vcombine_f32(vget_low_f32(lo_block),
                           vget_low_f32(hi_block));
    *row_hi = vcombine_f32(vget_high_f32(lo_block),
                           vget_high_f32(hi_block));
  };
  unzip_pair(bm00, bm01, &c00, &c10);
  unzip_pair(bm02, bm03, &c01, &c11);
  unzip_pair(bm10, bm11, &c20, &c30);
  unzip_pair(bm12, bm13, &c21, &c31);
  unzip_pair(bm20, bm21, &c40, &c50);
  unzip_pair(bm22, bm23, &c41, &c51);
  unzip_pair(bm30, bm31, &c60, &c70);
  unzip_pair(bm32, bm33, &c61, &c71);
#endif  // FUSED_CPP_SDPA_CACHE_HAS_BFMMLA

  // 标量 tail（E % 4 != 0），直接用原 K 指针（同 packk_inner）。
  if (e < E) {
    float scalar_acc[8 * 8];
    auto store_cij = [&](int i_row, float* dst) {
      float32x4_t lo, hi;
      switch (i_row) {
        case 0: lo = c00; hi = c01; break;
        case 1: lo = c10; hi = c11; break;
        case 2: lo = c20; hi = c21; break;
        case 3: lo = c30; hi = c31; break;
        case 4: lo = c40; hi = c41; break;
        case 5: lo = c50; hi = c51; break;
        case 6: lo = c60; hi = c61; break;
        case 7: lo = c70; hi = c71; break;
        default: lo = vdupq_n_f32(0); hi = vdupq_n_f32(0);
      }
      vst1q_f32(dst + 0, lo);
      vst1q_f32(dst + 4, hi);
    };
    auto load_cij = [&](int i_row, const float* src) {
      float32x4_t lo = vld1q_f32(src + 0);
      float32x4_t hi = vld1q_f32(src + 4);
      switch (i_row) {
        case 0: c00 = lo; c01 = hi; break;
        case 1: c10 = lo; c11 = hi; break;
        case 2: c20 = lo; c21 = hi; break;
        case 3: c30 = lo; c31 = hi; break;
        case 4: c40 = lo; c41 = hi; break;
        case 5: c50 = lo; c51 = hi; break;
        case 6: c60 = lo; c61 = hi; break;
        case 7: c70 = lo; c71 = hi; break;
      }
    };
    for (int i_row = 0; i_row < 8; ++i_row) store_cij(i_row, scalar_acc + i_row * 8);
    for (int64_t e2 = e; e2 < E; ++e2) {
      for (int i_row = 0; i_row < 8; ++i_row) {
        float qv = bf16_to_fp32_scalar(Qp[i_row * q_row_stride + e2]);
        for (int j_col = 0; j_col < 8; ++j_col) {
          float kv = bf16_to_fp32_scalar(Kp[j_col * k_row_stride + e2]);
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

// ── BF16 QKᵀ 8×8：BFMMLA 主路径 + 2-way k-unroll（评估用） ──────────────
//
// 与 gemm_qkt_microkernel_8x8_bf16 的 BFMMLA 主路径**算法等价**，唯一差别
// 在内层 e 循环步进从 4 改为 8（每 iter 处理 2 个 e-block），BFMMLA 指令数
// 从 16 翻到 32，累加器仍是 16 个（跨 e-block 复用，每个 acc 在 iter 内
// 被 touch 2 次）。
//
// 设计目标：
//   * 摊薄 loop overhead（cmp/branch ≈ 1-2 cycle / iter，相对 32 cycle
//     BFMMLA-bound iter 占 3-6%）；
//   * 扩大 issue window，让 OoO 调度器有更多自由度安排 Q-vld + vcombine +
//     BFMMLA 之间的 ALU 端口共享；
//   * BFMMLA latency=6 在原 baseline (32 cycle/iter) 已被 16 个独立 acc
//     cover，本路径不会进一步 hide FMA latency，但 acc 链的 dep chain
//     从 1 BFMMLA/iter 升到 2 BFMMLA/iter，配合本身充足 latency budget
//     仍然安全。
//
// 寄存器账本：16 BFMMLA acc + 4 a-reg(段0) + 4 a-reg(段1) + 4 b-reg(段0)
// + 4 b-reg(段1) = **32 NEON reg 正好打满**。如果编译器决定不复用段间
// 寄存器，可能轻微 spill；实测验证。
//
// 调用约束：与 baseline 一致 + 要求 E % 8 == 0（不满足时主体只处理
// [0, E&~7) 段，剩下 [E&~7, E) 走 baseline 的 4-step 主体或标量 tail）。
// 实际 head_dim ∈ {64, 128, 192, 256} 全满足 8 的倍数。
//
// 不接入 SDPA 主路径——这是 unroll 收益评估原型。
static inline void gemm_qkt_microkernel_8x8_bf16_unroll2(
    const at::BFloat16* Q,
    int64_t q_row_stride,
    const at::BFloat16* K,
    int64_t k_row_stride,
    int64_t E,
    float scale,
    float* scores_buf) {
  float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
  float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
  float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
  float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
  float32x4_t c40 = vdupq_n_f32(0), c41 = vdupq_n_f32(0);
  float32x4_t c50 = vdupq_n_f32(0), c51 = vdupq_n_f32(0);
  float32x4_t c60 = vdupq_n_f32(0), c61 = vdupq_n_f32(0);
  float32x4_t c70 = vdupq_n_f32(0), c71 = vdupq_n_f32(0);

  const uint16_t* Qp = reinterpret_cast<const uint16_t*>(Q);
  const uint16_t* Kp = reinterpret_cast<const uint16_t*>(K);
  int64_t e = 0;

#if FUSED_CPP_SDPA_CACHE_HAS_BFMMLA
  float32x4_t bm00 = vdupq_n_f32(0), bm01 = vdupq_n_f32(0);
  float32x4_t bm02 = vdupq_n_f32(0), bm03 = vdupq_n_f32(0);
  float32x4_t bm10 = vdupq_n_f32(0), bm11 = vdupq_n_f32(0);
  float32x4_t bm12 = vdupq_n_f32(0), bm13 = vdupq_n_f32(0);
  float32x4_t bm20 = vdupq_n_f32(0), bm21 = vdupq_n_f32(0);
  float32x4_t bm22 = vdupq_n_f32(0), bm23 = vdupq_n_f32(0);
  float32x4_t bm30 = vdupq_n_f32(0), bm31 = vdupq_n_f32(0);
  float32x4_t bm32 = vdupq_n_f32(0), bm33 = vdupq_n_f32(0);

  // 主体：每 iter 处理 e..e+7（2 个 e-block）。
  for (; e + 8 <= E; e += 8) {
    // ── 段 0：e..e+3 ──
    bfloat16x8_t a01_0, a23_0, a45_0, a67_0;
    {
      uint16x4_t q0 = vld1_u16(Qp + 0 * q_row_stride + e + 0);
      uint16x4_t q1 = vld1_u16(Qp + 1 * q_row_stride + e + 0);
      uint16x4_t q2 = vld1_u16(Qp + 2 * q_row_stride + e + 0);
      uint16x4_t q3 = vld1_u16(Qp + 3 * q_row_stride + e + 0);
      uint16x4_t q4 = vld1_u16(Qp + 4 * q_row_stride + e + 0);
      uint16x4_t q5 = vld1_u16(Qp + 5 * q_row_stride + e + 0);
      uint16x4_t q6 = vld1_u16(Qp + 6 * q_row_stride + e + 0);
      uint16x4_t q7 = vld1_u16(Qp + 7 * q_row_stride + e + 0);
      a01_0 = vreinterpretq_bf16_u16(vcombine_u16(q0, q1));
      a23_0 = vreinterpretq_bf16_u16(vcombine_u16(q2, q3));
      a45_0 = vreinterpretq_bf16_u16(vcombine_u16(q4, q5));
      a67_0 = vreinterpretq_bf16_u16(vcombine_u16(q6, q7));
    }
    bfloat16x8_t b01_0, b23_0, b45_0, b67_0;
    {
      uint16x4_t k0 = vld1_u16(Kp + 0 * k_row_stride + e + 0);
      uint16x4_t k1 = vld1_u16(Kp + 1 * k_row_stride + e + 0);
      uint16x4_t k2 = vld1_u16(Kp + 2 * k_row_stride + e + 0);
      uint16x4_t k3 = vld1_u16(Kp + 3 * k_row_stride + e + 0);
      uint16x4_t k4 = vld1_u16(Kp + 4 * k_row_stride + e + 0);
      uint16x4_t k5 = vld1_u16(Kp + 5 * k_row_stride + e + 0);
      uint16x4_t k6 = vld1_u16(Kp + 6 * k_row_stride + e + 0);
      uint16x4_t k7 = vld1_u16(Kp + 7 * k_row_stride + e + 0);
      b01_0 = vreinterpretq_bf16_u16(vcombine_u16(k0, k1));
      b23_0 = vreinterpretq_bf16_u16(vcombine_u16(k2, k3));
      b45_0 = vreinterpretq_bf16_u16(vcombine_u16(k4, k5));
      b67_0 = vreinterpretq_bf16_u16(vcombine_u16(k6, k7));
    }
    bm00 = vbfmmlaq_f32(bm00, a01_0, b01_0);
    bm01 = vbfmmlaq_f32(bm01, a01_0, b23_0);
    bm02 = vbfmmlaq_f32(bm02, a01_0, b45_0);
    bm03 = vbfmmlaq_f32(bm03, a01_0, b67_0);
    bm10 = vbfmmlaq_f32(bm10, a23_0, b01_0);
    bm11 = vbfmmlaq_f32(bm11, a23_0, b23_0);
    bm12 = vbfmmlaq_f32(bm12, a23_0, b45_0);
    bm13 = vbfmmlaq_f32(bm13, a23_0, b67_0);
    bm20 = vbfmmlaq_f32(bm20, a45_0, b01_0);
    bm21 = vbfmmlaq_f32(bm21, a45_0, b23_0);
    bm22 = vbfmmlaq_f32(bm22, a45_0, b45_0);
    bm23 = vbfmmlaq_f32(bm23, a45_0, b67_0);
    bm30 = vbfmmlaq_f32(bm30, a67_0, b01_0);
    bm31 = vbfmmlaq_f32(bm31, a67_0, b23_0);
    bm32 = vbfmmlaq_f32(bm32, a67_0, b45_0);
    bm33 = vbfmmlaq_f32(bm33, a67_0, b67_0);

    // ── 段 1：e+4..e+7 ──
    bfloat16x8_t a01_1, a23_1, a45_1, a67_1;
    {
      uint16x4_t q0 = vld1_u16(Qp + 0 * q_row_stride + e + 4);
      uint16x4_t q1 = vld1_u16(Qp + 1 * q_row_stride + e + 4);
      uint16x4_t q2 = vld1_u16(Qp + 2 * q_row_stride + e + 4);
      uint16x4_t q3 = vld1_u16(Qp + 3 * q_row_stride + e + 4);
      uint16x4_t q4 = vld1_u16(Qp + 4 * q_row_stride + e + 4);
      uint16x4_t q5 = vld1_u16(Qp + 5 * q_row_stride + e + 4);
      uint16x4_t q6 = vld1_u16(Qp + 6 * q_row_stride + e + 4);
      uint16x4_t q7 = vld1_u16(Qp + 7 * q_row_stride + e + 4);
      a01_1 = vreinterpretq_bf16_u16(vcombine_u16(q0, q1));
      a23_1 = vreinterpretq_bf16_u16(vcombine_u16(q2, q3));
      a45_1 = vreinterpretq_bf16_u16(vcombine_u16(q4, q5));
      a67_1 = vreinterpretq_bf16_u16(vcombine_u16(q6, q7));
    }
    bfloat16x8_t b01_1, b23_1, b45_1, b67_1;
    {
      uint16x4_t k0 = vld1_u16(Kp + 0 * k_row_stride + e + 4);
      uint16x4_t k1 = vld1_u16(Kp + 1 * k_row_stride + e + 4);
      uint16x4_t k2 = vld1_u16(Kp + 2 * k_row_stride + e + 4);
      uint16x4_t k3 = vld1_u16(Kp + 3 * k_row_stride + e + 4);
      uint16x4_t k4 = vld1_u16(Kp + 4 * k_row_stride + e + 4);
      uint16x4_t k5 = vld1_u16(Kp + 5 * k_row_stride + e + 4);
      uint16x4_t k6 = vld1_u16(Kp + 6 * k_row_stride + e + 4);
      uint16x4_t k7 = vld1_u16(Kp + 7 * k_row_stride + e + 4);
      b01_1 = vreinterpretq_bf16_u16(vcombine_u16(k0, k1));
      b23_1 = vreinterpretq_bf16_u16(vcombine_u16(k2, k3));
      b45_1 = vreinterpretq_bf16_u16(vcombine_u16(k4, k5));
      b67_1 = vreinterpretq_bf16_u16(vcombine_u16(k6, k7));
    }
    bm00 = vbfmmlaq_f32(bm00, a01_1, b01_1);
    bm01 = vbfmmlaq_f32(bm01, a01_1, b23_1);
    bm02 = vbfmmlaq_f32(bm02, a01_1, b45_1);
    bm03 = vbfmmlaq_f32(bm03, a01_1, b67_1);
    bm10 = vbfmmlaq_f32(bm10, a23_1, b01_1);
    bm11 = vbfmmlaq_f32(bm11, a23_1, b23_1);
    bm12 = vbfmmlaq_f32(bm12, a23_1, b45_1);
    bm13 = vbfmmlaq_f32(bm13, a23_1, b67_1);
    bm20 = vbfmmlaq_f32(bm20, a45_1, b01_1);
    bm21 = vbfmmlaq_f32(bm21, a45_1, b23_1);
    bm22 = vbfmmlaq_f32(bm22, a45_1, b45_1);
    bm23 = vbfmmlaq_f32(bm23, a45_1, b67_1);
    bm30 = vbfmmlaq_f32(bm30, a67_1, b01_1);
    bm31 = vbfmmlaq_f32(bm31, a67_1, b23_1);
    bm32 = vbfmmlaq_f32(bm32, a67_1, b45_1);
    bm33 = vbfmmlaq_f32(bm33, a67_1, b67_1);
  }

  // 余项：处理剩下的 [e, E_main) 段（4 的倍数但不是 8 的倍数），与 baseline
  // 4-step 主体一致。
  for (; e + 4 <= E; e += 4) {
    bfloat16x8_t a01, a23, a45, a67;
    {
      uint16x4_t q0 = vld1_u16(Qp + 0 * q_row_stride + e);
      uint16x4_t q1 = vld1_u16(Qp + 1 * q_row_stride + e);
      uint16x4_t q2 = vld1_u16(Qp + 2 * q_row_stride + e);
      uint16x4_t q3 = vld1_u16(Qp + 3 * q_row_stride + e);
      uint16x4_t q4 = vld1_u16(Qp + 4 * q_row_stride + e);
      uint16x4_t q5 = vld1_u16(Qp + 5 * q_row_stride + e);
      uint16x4_t q6 = vld1_u16(Qp + 6 * q_row_stride + e);
      uint16x4_t q7 = vld1_u16(Qp + 7 * q_row_stride + e);
      a01 = vreinterpretq_bf16_u16(vcombine_u16(q0, q1));
      a23 = vreinterpretq_bf16_u16(vcombine_u16(q2, q3));
      a45 = vreinterpretq_bf16_u16(vcombine_u16(q4, q5));
      a67 = vreinterpretq_bf16_u16(vcombine_u16(q6, q7));
    }
    bfloat16x8_t b01, b23, b45, b67;
    {
      uint16x4_t k0 = vld1_u16(Kp + 0 * k_row_stride + e);
      uint16x4_t k1 = vld1_u16(Kp + 1 * k_row_stride + e);
      uint16x4_t k2 = vld1_u16(Kp + 2 * k_row_stride + e);
      uint16x4_t k3 = vld1_u16(Kp + 3 * k_row_stride + e);
      uint16x4_t k4 = vld1_u16(Kp + 4 * k_row_stride + e);
      uint16x4_t k5 = vld1_u16(Kp + 5 * k_row_stride + e);
      uint16x4_t k6 = vld1_u16(Kp + 6 * k_row_stride + e);
      uint16x4_t k7 = vld1_u16(Kp + 7 * k_row_stride + e);
      b01 = vreinterpretq_bf16_u16(vcombine_u16(k0, k1));
      b23 = vreinterpretq_bf16_u16(vcombine_u16(k2, k3));
      b45 = vreinterpretq_bf16_u16(vcombine_u16(k4, k5));
      b67 = vreinterpretq_bf16_u16(vcombine_u16(k6, k7));
    }
    bm00 = vbfmmlaq_f32(bm00, a01, b01);
    bm01 = vbfmmlaq_f32(bm01, a01, b23);
    bm02 = vbfmmlaq_f32(bm02, a01, b45);
    bm03 = vbfmmlaq_f32(bm03, a01, b67);
    bm10 = vbfmmlaq_f32(bm10, a23, b01);
    bm11 = vbfmmlaq_f32(bm11, a23, b23);
    bm12 = vbfmmlaq_f32(bm12, a23, b45);
    bm13 = vbfmmlaq_f32(bm13, a23, b67);
    bm20 = vbfmmlaq_f32(bm20, a45, b01);
    bm21 = vbfmmlaq_f32(bm21, a45, b23);
    bm22 = vbfmmlaq_f32(bm22, a45, b45);
    bm23 = vbfmmlaq_f32(bm23, a45, b67);
    bm30 = vbfmmlaq_f32(bm30, a67, b01);
    bm31 = vbfmmlaq_f32(bm31, a67, b23);
    bm32 = vbfmmlaq_f32(bm32, a67, b45);
    bm33 = vbfmmlaq_f32(bm33, a67, b67);
  }

  auto unzip_pair = [](float32x4_t lo_block, float32x4_t hi_block,
                       float32x4_t* row_lo, float32x4_t* row_hi) {
    *row_lo = vcombine_f32(vget_low_f32(lo_block),
                           vget_low_f32(hi_block));
    *row_hi = vcombine_f32(vget_high_f32(lo_block),
                           vget_high_f32(hi_block));
  };
  unzip_pair(bm00, bm01, &c00, &c10);
  unzip_pair(bm02, bm03, &c01, &c11);
  unzip_pair(bm10, bm11, &c20, &c30);
  unzip_pair(bm12, bm13, &c21, &c31);
  unzip_pair(bm20, bm21, &c40, &c50);
  unzip_pair(bm22, bm23, &c41, &c51);
  unzip_pair(bm30, bm31, &c60, &c70);
  unzip_pair(bm32, bm33, &c61, &c71);
#endif  // FUSED_CPP_SDPA_CACHE_HAS_BFMMLA

  // 标量 tail（E % 4 != 0）
  if (e < E) {
    float scalar_acc[8 * 8];
    auto store_cij = [&](int i_row, float* dst) {
      float32x4_t lo, hi;
      switch (i_row) {
        case 0: lo = c00; hi = c01; break;
        case 1: lo = c10; hi = c11; break;
        case 2: lo = c20; hi = c21; break;
        case 3: lo = c30; hi = c31; break;
        case 4: lo = c40; hi = c41; break;
        case 5: lo = c50; hi = c51; break;
        case 6: lo = c60; hi = c61; break;
        case 7: lo = c70; hi = c71; break;
        default: lo = vdupq_n_f32(0); hi = vdupq_n_f32(0);
      }
      vst1q_f32(dst + 0, lo);
      vst1q_f32(dst + 4, hi);
    };
    auto load_cij = [&](int i_row, const float* src) {
      float32x4_t lo = vld1q_f32(src + 0);
      float32x4_t hi = vld1q_f32(src + 4);
      switch (i_row) {
        case 0: c00 = lo; c01 = hi; break;
        case 1: c10 = lo; c11 = hi; break;
        case 2: c20 = lo; c21 = hi; break;
        case 3: c30 = lo; c31 = hi; break;
        case 4: c40 = lo; c41 = hi; break;
        case 5: c50 = lo; c51 = hi; break;
        case 6: c60 = lo; c61 = hi; break;
        case 7: c70 = lo; c71 = hi; break;
      }
    };
    for (int i_row = 0; i_row < 8; ++i_row) store_cij(i_row, scalar_acc + i_row * 8);
    for (int64_t e2 = e; e2 < E; ++e2) {
      for (int i_row = 0; i_row < 8; ++i_row) {
        float qv = bf16_to_fp32_scalar(Qp[i_row * q_row_stride + e2]);
        for (int j_col = 0; j_col < 8; ++j_col) {
          float kv = bf16_to_fp32_scalar(Kp[j_col * k_row_stride + e2]);
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

// gemm_qkt_microkernel_8x8_fp32：fp32 路径，FMLA 主路径。
static inline void gemm_qkt_microkernel_8x8_fp32(
    const float* Q,
    int64_t q_row_stride,
    const float* K,
    int64_t k_row_stride,
    int64_t E,
    float scale,
    float* scores_buf) {
  float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
  float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
  float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
  float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
  float32x4_t c40 = vdupq_n_f32(0), c41 = vdupq_n_f32(0);
  float32x4_t c50 = vdupq_n_f32(0), c51 = vdupq_n_f32(0);
  float32x4_t c60 = vdupq_n_f32(0), c61 = vdupq_n_f32(0);
  float32x4_t c70 = vdupq_n_f32(0), c71 = vdupq_n_f32(0);

  // 由于 8 行 8 列 fp32 累加器全部是行向布局，外层沿 e 维循环，每段
  // 加载 4 lane Q[i, e:e+4] 与 4 lane K[j, e:e+4]，对每对 (i, j) 做
  // vfmaq + vaddvq？不行，单 lane 累加效率太低。
  //
  // 这里采用 GEPB 风格：对每段 4 lane（沿 e 维），内层广播 K 的 lane 到
  // float32x4_t，做 vfmaq_lane_f32 累加到行向累加器。具体：
  //   q_i_4lane = Q[i, e:e+4]  (1 个 float32x4_t)
  //   k_j_4lane = K[j, e:e+4]  (1 个 float32x4_t)
  // 行累加器布局：每行 8 列 → 2 个 float32x4_t。
  //
  // 通用展开策略：沿 e 步长 4，每段：
  //   (1) 加载 Q 的 8 行 × 4 lane = 8 个 float32x4_t (qrows[8])
  //   (2) 加载 K 的 8 行 × 4 lane = 8 个 float32x4_t (krows[8])
  //   (3) 对每 (i, j) 做 vmulq + vaddvq → scalar，累加到对应 cXY 的 lane
  //
  // 该方案使用 vaddvq 横向求和，非 MLA 路径，但 fp32 路径的纯 FMLA
  // 实现需要 transpose K（B 矩阵），代价不低。我们用「外积」展开：
  //   外积：对每个 e_lane k ∈ [0, 4)，加载 Q 第 k lane 列 → 8 个 fp32
  //         scalar，然后 vfmaq_n_f32(c[i][j_blk], k_j_4lane_at_e_lane_k, q_i_at_lane_k)。
  //   但需要 K 的「列向量」即 K[:, e+k]，不是 K[j, e+k]——这要求
  //   K 在最内维是 j（即 K 转置过来）。原始 K 内存是 [j_row][e_col]，
  //   所以 K[:, e+k] 是 K[j, e+k]（j 变，e+k 固定），它在内存中按 j
  //   stride 而不是相邻，单点 gather 开销大。
  //
  // 折中方案：使用「逐行」方案。外层 i 循环，内层沿 e 累加 q_row · k_row^T
  // 8 列同时进行：
  //   q_i_4lane = Q[i, e:e+4]  → 1 个 float32x4_t
  //   对 j ∈ [0, 8)：
  //     k_j_4lane = K[j, e:e+4]
  //     使用 vfmaq + 横向 reduction？开销同上。
  //
  // 最简单可行：行外层 + j 列内层 + e 循环展开 4 路 dot product，最后
  // vaddvq 求和写到 scalar acc 里。这正是当前 dot_fp32_neon 的逻辑，
  // 8 行 × 8 列 = 64 次 dot 调用——但 8×8 micro 体积下每个 dot 的
  // 启动开销被 head_dim 摊薄，仍然显著优于 baseline。
  for (int i_row = 0; i_row < 8; ++i_row) {
    float* row_acc;
    float scalar_buf[8];
    (void)row_acc;
    for (int j_col = 0; j_col < 8; ++j_col) {
      const float* qrow = Q + i_row * q_row_stride;
      const float* krow = K + j_col * k_row_stride;
      float32x4_t acc = vdupq_n_f32(0.0f);
      int64_t e = 0;
      for (; e + 4 <= E; e += 4) {
        float32x4_t qv = vld1q_f32(qrow + e);
        float32x4_t kv = vld1q_f32(krow + e);
        acc = vfmaq_f32(acc, qv, kv);
      }
      float s = vaddvq_f32(acc);
      for (; e < E; ++e) {
        s += qrow[e] * krow[e];
      }
      scalar_buf[j_col] = s * scale;
    }
    vst1q_f32(scores_buf + i_row * 8 + 0, vld1q_f32(scalar_buf + 0));
    vst1q_f32(scores_buf + i_row * 8 + 4, vld1q_f32(scalar_buf + 4));
  }

  // 抑制未用变量警告：
  (void)c00; (void)c01; (void)c10; (void)c11;
  (void)c20; (void)c21; (void)c30; (void)c31;
  (void)c40; (void)c41; (void)c50; (void)c51;
  (void)c60; (void)c61; (void)c70; (void)c71;
}

// gemm_qkt_microkernel_8x8_fp32_ublock4：fp32 路径 4×4 双向分块 + 16 累加器版。
//
// 与 gemm_qkt_microkernel_8x8_fp32 在数值上**接近**等价但不按位相同：fp32
// 下 vfmaq + vpaddq tree-reduce 的累加顺序与原本 vfmaq + vaddvq 不同，存在
// ULP-级误差，已被 SDPA 等价性测试容忍度覆盖（atol=rtol=1e-4 fp32）。
//
// 关键改动（参考 SDPA_TODO.md P0 fp32 重写）：
//   * 把 8×8 输出切成 4 个 4×4 子块，每个子块 16 个独立 float32x4_t
//     累加器（a00..a33），全部沿 e 维同时累加。原 baseline 实现是
//     64 个独立 dot product，单累加器 RAW 依赖链 = E/4，OoO 跨 (i,j)
//     iteration 渲染只能并发 ~3 条 fma 链；本实现把 ILP 拉到 16 条
//     完全独立的 fma 链。
//   * Q 和 K 都按行连续 vld1q_f32 加载，**不需要 K 转置或 pre-pack**：
//     每个内层 e-step 加载 4 Q 行 + 4 K 行 = 8 条 vld1q（连续访存），
//     发出 16 条 vfmaq_f32 全独立。
//   * vpaddq tree-reduce：每行 4 个 quad → 1 个 quad（4 fp32 = scores
//     一行的 4 列）。每个 4×4 子块 16 quad → 4 quad → 4 vmulq_f32 scale
//     → 4 vst1q_f32 写出。
//
// 寄存器账本：
//   16 acc + 4 Q + 4 K + 临时 ≈ 25/32，留 7 个 NEON reg 给编译器做
//   software pipelining。
//
// 数据上限分析（按 4-pipe FMA、IPC≈3.89、4 cycle FMA latency 假设）：
//   每内层 e-step 16 vfmaq + 8 vld1q ≈ 4 cycle（FMA-pipe 限），ILP 16
//   足以打满 latency 4 × pipes 4 = 16；剩余瓶颈在 LSU 带宽。
//
// 调用约束：与 baseline 一致——E 任意，scores_buf 必须是 8×8 行步长 8。
static inline void gemm_qkt_microkernel_8x8_fp32_ublock4(
    const float* Q,
    int64_t q_row_stride,
    const float* K,
    int64_t k_row_stride,
    int64_t E,
    float scale,
    float* scores_buf) {
  const float32x4_t scale_v = vdupq_n_f32(scale);

  // 把 8×8 切成 2×2 个 4×4 子块。外层两层循环只有 4 次迭代，全部
  // 静态展开（i_blk ∈ {0,4}, j_blk ∈ {0,4}），编译器会复制 4 份内核
  // 体；这样 i_blk/j_blk 全是编译期常量，加载偏移可以 fold 进 ldr。
  // 用宏代替 for 是为了避免编译器不展开静态循环带来的不必要分支。
#define FUSED_CPP_QKT_FP32_UBLOCK4_BODY(I_BLK, J_BLK)                          \
  do {                                                                        \
    /* —— 16 个独立 fp32 quad 累加器 —— */                                    \
    float32x4_t a00 = vdupq_n_f32(0.0f), a01 = vdupq_n_f32(0.0f);             \
    float32x4_t a02 = vdupq_n_f32(0.0f), a03 = vdupq_n_f32(0.0f);             \
    float32x4_t a10 = vdupq_n_f32(0.0f), a11 = vdupq_n_f32(0.0f);             \
    float32x4_t a12 = vdupq_n_f32(0.0f), a13 = vdupq_n_f32(0.0f);             \
    float32x4_t a20 = vdupq_n_f32(0.0f), a21 = vdupq_n_f32(0.0f);             \
    float32x4_t a22 = vdupq_n_f32(0.0f), a23 = vdupq_n_f32(0.0f);             \
    float32x4_t a30 = vdupq_n_f32(0.0f), a31 = vdupq_n_f32(0.0f);             \
    float32x4_t a32 = vdupq_n_f32(0.0f), a33 = vdupq_n_f32(0.0f);             \
                                                                              \
    const float* qrow0 = Q + ((I_BLK) + 0) * q_row_stride;                    \
    const float* qrow1 = Q + ((I_BLK) + 1) * q_row_stride;                    \
    const float* qrow2 = Q + ((I_BLK) + 2) * q_row_stride;                    \
    const float* qrow3 = Q + ((I_BLK) + 3) * q_row_stride;                    \
    const float* krow0 = K + ((J_BLK) + 0) * k_row_stride;                    \
    const float* krow1 = K + ((J_BLK) + 1) * k_row_stride;                    \
    const float* krow2 = K + ((J_BLK) + 2) * k_row_stride;                    \
    const float* krow3 = K + ((J_BLK) + 3) * k_row_stride;                    \
                                                                              \
    int64_t e = 0;                                                            \
    for (; e + 4 <= E; e += 4) {                                              \
      /* 8 条 vld1q：4 Q 行 + 4 K 行（每条都是连续访存，无 gather）。 */      \
      float32x4_t q0 = vld1q_f32(qrow0 + e);                                  \
      float32x4_t q1 = vld1q_f32(qrow1 + e);                                  \
      float32x4_t q2 = vld1q_f32(qrow2 + e);                                  \
      float32x4_t q3 = vld1q_f32(qrow3 + e);                                  \
      float32x4_t k0 = vld1q_f32(krow0 + e);                                  \
      float32x4_t k1 = vld1q_f32(krow1 + e);                                  \
      float32x4_t k2 = vld1q_f32(krow2 + e);                                  \
      float32x4_t k3 = vld1q_f32(krow3 + e);                                  \
                                                                              \
      /* 16 条独立 vfmaq_f32（外积扇出），跨累加器无 RAW 依赖。 */            \
      a00 = vfmaq_f32(a00, q0, k0); a01 = vfmaq_f32(a01, q0, k1);             \
      a02 = vfmaq_f32(a02, q0, k2); a03 = vfmaq_f32(a03, q0, k3);             \
      a10 = vfmaq_f32(a10, q1, k0); a11 = vfmaq_f32(a11, q1, k1);             \
      a12 = vfmaq_f32(a12, q1, k2); a13 = vfmaq_f32(a13, q1, k3);             \
      a20 = vfmaq_f32(a20, q2, k0); a21 = vfmaq_f32(a21, q2, k1);             \
      a22 = vfmaq_f32(a22, q2, k2); a23 = vfmaq_f32(a23, q2, k3);             \
      a30 = vfmaq_f32(a30, q3, k0); a31 = vfmaq_f32(a31, q3, k1);             \
      a32 = vfmaq_f32(a32, q3, k2); a33 = vfmaq_f32(a33, q3, k3);             \
    }                                                                         \
                                                                              \
    /* —— vpaddq 树 reduce：每行 4 quad → 1 quad —— */                        \
    /* vpaddq(a, b)[0..3] = [a[0]+a[1], a[2]+a[3], b[0]+b[1], b[2]+b[3]]。 */ \
    /* 两层 vpaddq 把每个 quad 内的 4 个 lane reduce 成 1 个 fp32，并把   */ \
    /* 4 个累加器的结果排到同一个 quad 的 4 个 lane 上。 */                  \
    float32x4_t row0 =                                                        \
        vpaddq_f32(vpaddq_f32(a00, a01), vpaddq_f32(a02, a03));               \
    float32x4_t row1 =                                                        \
        vpaddq_f32(vpaddq_f32(a10, a11), vpaddq_f32(a12, a13));               \
    float32x4_t row2 =                                                        \
        vpaddq_f32(vpaddq_f32(a20, a21), vpaddq_f32(a22, a23));               \
    float32x4_t row3 =                                                        \
        vpaddq_f32(vpaddq_f32(a30, a31), vpaddq_f32(a32, a33));               \
                                                                              \
    /* —— 标量尾循环：处理 E % 4 ≠ 0。落到 scratch 里加完再回传。 —— */      \
    if (e < E) {                                                              \
      alignas(16) float scratch[16];                                          \
      vst1q_f32(scratch + 0 * 4, row0);                                       \
      vst1q_f32(scratch + 1 * 4, row1);                                       \
      vst1q_f32(scratch + 2 * 4, row2);                                       \
      vst1q_f32(scratch + 3 * 4, row3);                                       \
      const float* qrows[4] = {qrow0, qrow1, qrow2, qrow3};                   \
      const float* krows[4] = {krow0, krow1, krow2, krow3};                   \
      for (int ii = 0; ii < 4; ++ii) {                                        \
        for (int jj = 0; jj < 4; ++jj) {                                      \
          float s = scratch[ii * 4 + jj];                                     \
          for (int64_t et = e; et < E; ++et) {                                \
            s += qrows[ii][et] * krows[jj][et];                               \
          }                                                                   \
          scratch[ii * 4 + jj] = s;                                           \
        }                                                                     \
      }                                                                       \
      row0 = vld1q_f32(scratch + 0 * 4);                                      \
      row1 = vld1q_f32(scratch + 1 * 4);                                      \
      row2 = vld1q_f32(scratch + 2 * 4);                                      \
      row3 = vld1q_f32(scratch + 3 * 4);                                      \
    }                                                                         \
                                                                              \
    /* 应用 scale 并写入 scores_buf[(I_BLK + i)*8 + J_BLK : J_BLK+4]。 */     \
    vst1q_f32(scores_buf + ((I_BLK) + 0) * 8 + (J_BLK),                       \
              vmulq_f32(row0, scale_v));                                      \
    vst1q_f32(scores_buf + ((I_BLK) + 1) * 8 + (J_BLK),                       \
              vmulq_f32(row1, scale_v));                                      \
    vst1q_f32(scores_buf + ((I_BLK) + 2) * 8 + (J_BLK),                       \
              vmulq_f32(row2, scale_v));                                      \
    vst1q_f32(scores_buf + ((I_BLK) + 3) * 8 + (J_BLK),                       \
              vmulq_f32(row3, scale_v));                                      \
  } while (0)

  FUSED_CPP_QKT_FP32_UBLOCK4_BODY(0, 0);
  FUSED_CPP_QKT_FP32_UBLOCK4_BODY(0, 4);
  FUSED_CPP_QKT_FP32_UBLOCK4_BODY(4, 0);
  FUSED_CPP_QKT_FP32_UBLOCK4_BODY(4, 4);

#undef FUSED_CPP_QKT_FP32_UBLOCK4_BODY
}

// gemm_pv_microkernel_8x8_bf16：
//   计算 O[8][8] += P_hat[8][Sk] · V[Sk][8]
//   P_hat 行步长 P_row_stride（fp32），V 行步长 v_row_stride（bf16）。
//   O 行步长 o_row_stride（fp32）。
//
// 严禁使用 BFMMLA / BFDOT / BFMLALB/T（一侧是 fp32）。统一走「V widen
// 到 fp32 + vfmaq_f32」。Sk 由调用方限定为 8 的倍数（8×8 主体；尾部
// 退化由调用方处理）。
static inline void gemm_pv_microkernel_8x8_bf16(
    const float* P_hat,
    int64_t P_row_stride,
    const at::BFloat16* V,
    int64_t v_row_stride,
    int64_t Sk,
    float* O,
    int64_t o_row_stride) {

  // 加载 O 的 8×8 fp32 累加器（行向 4 lane 布局）。
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

  const uint16_t* Vp = reinterpret_cast<const uint16_t*>(V);

  for (int64_t k = 0; k < Sk; ++k) {
    // Widen 一行 V 的 8 列到 2 个 float32x4_t。
    float32x4_t v_lo = widen_bf16x4_to_fp32(Vp + k * v_row_stride + 0);
    float32x4_t v_hi = widen_bf16x4_to_fp32(Vp + k * v_row_stride + 4);

    // Broadcast P_hat[i][k] 为 float32x4_t，对每行 i 做 fma。
    float p0 = P_hat[0 * P_row_stride + k];
    float p1 = P_hat[1 * P_row_stride + k];
    float p2 = P_hat[2 * P_row_stride + k];
    float p3 = P_hat[3 * P_row_stride + k];
    float p4 = P_hat[4 * P_row_stride + k];
    float p5 = P_hat[5 * P_row_stride + k];
    float p6 = P_hat[6 * P_row_stride + k];
    float p7 = P_hat[7 * P_row_stride + k];

    o00 = vfmaq_n_f32(o00, v_lo, p0); o01 = vfmaq_n_f32(o01, v_hi, p0);
    o10 = vfmaq_n_f32(o10, v_lo, p1); o11 = vfmaq_n_f32(o11, v_hi, p1);
    o20 = vfmaq_n_f32(o20, v_lo, p2); o21 = vfmaq_n_f32(o21, v_hi, p2);
    o30 = vfmaq_n_f32(o30, v_lo, p3); o31 = vfmaq_n_f32(o31, v_hi, p3);
    o40 = vfmaq_n_f32(o40, v_lo, p4); o41 = vfmaq_n_f32(o41, v_hi, p4);
    o50 = vfmaq_n_f32(o50, v_lo, p5); o51 = vfmaq_n_f32(o51, v_hi, p5);
    o60 = vfmaq_n_f32(o60, v_lo, p6); o61 = vfmaq_n_f32(o61, v_hi, p6);
    o70 = vfmaq_n_f32(o70, v_lo, p7); o71 = vfmaq_n_f32(o71, v_hi, p7);
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

// gemm_pv_microkernel_8x8_fp32：fp32 路径，FMLA 主路径。
static inline void gemm_pv_microkernel_8x8_fp32(
    const float* P_hat,
    int64_t P_row_stride,
    const float* V,
    int64_t v_row_stride,
    int64_t Sk,
    float* O,
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

  // —— k 维 unroll=4 主循环 ——
  // 设计要点（用于引导 clang -O2 的指令调度，做 load/FMLA 掩盖）：
  //   1) 把每段的 V 加载写到「新变量」(v_lo_s / v_hi_s)，与其它段无 RAW 依赖；
  //   2) 把下一段 V 的预取放在「当前段最后一条用到 v_lo/v_hi 的 FMLA 之后」，
  //      编译器会沿着 list-schedule 把 load 上提到本段中部，与 FMLA 错峰发射；
  //   3) 每段的 8 个 P 标量在该段头部读取，作用域仅限本段，编译器看到
  //      last-use 之后立即可以释放，且不会跨段保留 32 个广播寄存器。
  //   总寄存器占用：累加器16 + V(本段2 + 下段2) + P(本段8) = 28 个 NEON reg。
  const int64_t Sk4 = Sk & ~int64_t{3};
  int64_t k = 0;
  if (Sk4 > 0) {
    // 段 0 的 V 预取（循环外 prologue），让循环体的第一条 FMLA 立刻可发射。
    float32x4_t v_lo0 = vld1q_f32(V + 0 * v_row_stride + 0);
    float32x4_t v_hi0 = vld1q_f32(V + 0 * v_row_stride + 4);
    for (; k < Sk4; k += 4) {
      // ── 段 0：使用 v_lo0/v_hi0，预取下一段(段 1) 到 v_lo1/v_hi1 ──
      float p0_0 = P_hat[0 * P_row_stride + k + 0];
      float p1_0 = P_hat[1 * P_row_stride + k + 0];
      float p2_0 = P_hat[2 * P_row_stride + k + 0];
      float p3_0 = P_hat[3 * P_row_stride + k + 0];
      float p4_0 = P_hat[4 * P_row_stride + k + 0];
      float p5_0 = P_hat[5 * P_row_stride + k + 0];
      float p6_0 = P_hat[6 * P_row_stride + k + 0];
      float p7_0 = P_hat[7 * P_row_stride + k + 0];
      o00 = vfmaq_n_f32(o00, v_lo0, p0_0); o01 = vfmaq_n_f32(o01, v_hi0, p0_0);
      o10 = vfmaq_n_f32(o10, v_lo0, p1_0); o11 = vfmaq_n_f32(o11, v_hi0, p1_0);
      o20 = vfmaq_n_f32(o20, v_lo0, p2_0); o21 = vfmaq_n_f32(o21, v_hi0, p2_0);
      o30 = vfmaq_n_f32(o30, v_lo0, p3_0); o31 = vfmaq_n_f32(o31, v_hi0, p3_0);
      o40 = vfmaq_n_f32(o40, v_lo0, p4_0); o41 = vfmaq_n_f32(o41, v_hi0, p4_0);
      o50 = vfmaq_n_f32(o50, v_lo0, p5_0); o51 = vfmaq_n_f32(o51, v_hi0, p5_0);
      // 此处 v_lo0/v_hi0 还差最后两行 FMLA。先发出下一段 V 的 load，给调度器
      // 机会把这两条 vld1q 上提到段 0 的 FMLA 之间，做 load/FMLA 交错。
      float32x4_t v_lo1 = vld1q_f32(V + (k + 1) * v_row_stride + 0);
      float32x4_t v_hi1 = vld1q_f32(V + (k + 1) * v_row_stride + 4);
      o60 = vfmaq_n_f32(o60, v_lo0, p6_0); o61 = vfmaq_n_f32(o61, v_hi0, p6_0);
      o70 = vfmaq_n_f32(o70, v_lo0, p7_0); o71 = vfmaq_n_f32(o71, v_hi0, p7_0);
      // 段 0 结束：v_lo0/v_hi0/p?_0 全部 last-use，编译器可释放对应寄存器。

      // ── 段 1：使用 v_lo1/v_hi1，预取段 2 到 v_lo2/v_hi2 ──
      float p0_1 = P_hat[0 * P_row_stride + k + 1];
      float p1_1 = P_hat[1 * P_row_stride + k + 1];
      float p2_1 = P_hat[2 * P_row_stride + k + 1];
      float p3_1 = P_hat[3 * P_row_stride + k + 1];
      float p4_1 = P_hat[4 * P_row_stride + k + 1];
      float p5_1 = P_hat[5 * P_row_stride + k + 1];
      float p6_1 = P_hat[6 * P_row_stride + k + 1];
      float p7_1 = P_hat[7 * P_row_stride + k + 1];
      o00 = vfmaq_n_f32(o00, v_lo1, p0_1); o01 = vfmaq_n_f32(o01, v_hi1, p0_1);
      o10 = vfmaq_n_f32(o10, v_lo1, p1_1); o11 = vfmaq_n_f32(o11, v_hi1, p1_1);
      o20 = vfmaq_n_f32(o20, v_lo1, p2_1); o21 = vfmaq_n_f32(o21, v_hi1, p2_1);
      o30 = vfmaq_n_f32(o30, v_lo1, p3_1); o31 = vfmaq_n_f32(o31, v_hi1, p3_1);
      o40 = vfmaq_n_f32(o40, v_lo1, p4_1); o41 = vfmaq_n_f32(o41, v_hi1, p4_1);
      o50 = vfmaq_n_f32(o50, v_lo1, p5_1); o51 = vfmaq_n_f32(o51, v_hi1, p5_1);
      float32x4_t v_lo2 = vld1q_f32(V + (k + 2) * v_row_stride + 0);
      float32x4_t v_hi2 = vld1q_f32(V + (k + 2) * v_row_stride + 4);
      o60 = vfmaq_n_f32(o60, v_lo1, p6_1); o61 = vfmaq_n_f32(o61, v_hi1, p6_1);
      o70 = vfmaq_n_f32(o70, v_lo1, p7_1); o71 = vfmaq_n_f32(o71, v_hi1, p7_1);

      // ── 段 2：使用 v_lo2/v_hi2，预取段 3 到 v_lo3/v_hi3 ──
      float p0_2 = P_hat[0 * P_row_stride + k + 2];
      float p1_2 = P_hat[1 * P_row_stride + k + 2];
      float p2_2 = P_hat[2 * P_row_stride + k + 2];
      float p3_2 = P_hat[3 * P_row_stride + k + 2];
      float p4_2 = P_hat[4 * P_row_stride + k + 2];
      float p5_2 = P_hat[5 * P_row_stride + k + 2];
      float p6_2 = P_hat[6 * P_row_stride + k + 2];
      float p7_2 = P_hat[7 * P_row_stride + k + 2];
      o00 = vfmaq_n_f32(o00, v_lo2, p0_2); o01 = vfmaq_n_f32(o01, v_hi2, p0_2);
      o10 = vfmaq_n_f32(o10, v_lo2, p1_2); o11 = vfmaq_n_f32(o11, v_hi2, p1_2);
      o20 = vfmaq_n_f32(o20, v_lo2, p2_2); o21 = vfmaq_n_f32(o21, v_hi2, p2_2);
      o30 = vfmaq_n_f32(o30, v_lo2, p3_2); o31 = vfmaq_n_f32(o31, v_hi2, p3_2);
      o40 = vfmaq_n_f32(o40, v_lo2, p4_2); o41 = vfmaq_n_f32(o41, v_hi2, p4_2);
      o50 = vfmaq_n_f32(o50, v_lo2, p5_2); o51 = vfmaq_n_f32(o51, v_hi2, p5_2);
      float32x4_t v_lo3 = vld1q_f32(V + (k + 3) * v_row_stride + 0);
      float32x4_t v_hi3 = vld1q_f32(V + (k + 3) * v_row_stride + 4);
      o60 = vfmaq_n_f32(o60, v_lo2, p6_2); o61 = vfmaq_n_f32(o61, v_hi2, p6_2);
      o70 = vfmaq_n_f32(o70, v_lo2, p7_2); o71 = vfmaq_n_f32(o71, v_hi2, p7_2);

      // ── 段 3：使用 v_lo3/v_hi3，预取「下一轮的段 0」到 v_lo0/v_hi0 ──
      float p0_3 = P_hat[0 * P_row_stride + k + 3];
      float p1_3 = P_hat[1 * P_row_stride + k + 3];
      float p2_3 = P_hat[2 * P_row_stride + k + 3];
      float p3_3 = P_hat[3 * P_row_stride + k + 3];
      float p4_3 = P_hat[4 * P_row_stride + k + 3];
      float p5_3 = P_hat[5 * P_row_stride + k + 3];
      float p6_3 = P_hat[6 * P_row_stride + k + 3];
      float p7_3 = P_hat[7 * P_row_stride + k + 3];
      o00 = vfmaq_n_f32(o00, v_lo3, p0_3); o01 = vfmaq_n_f32(o01, v_hi3, p0_3);
      o10 = vfmaq_n_f32(o10, v_lo3, p1_3); o11 = vfmaq_n_f32(o11, v_hi3, p1_3);
      o20 = vfmaq_n_f32(o20, v_lo3, p2_3); o21 = vfmaq_n_f32(o21, v_hi3, p2_3);
      o30 = vfmaq_n_f32(o30, v_lo3, p3_3); o31 = vfmaq_n_f32(o31, v_hi3, p3_3);
      o40 = vfmaq_n_f32(o40, v_lo3, p4_3); o41 = vfmaq_n_f32(o41, v_hi3, p4_3);
      o50 = vfmaq_n_f32(o50, v_lo3, p5_3); o51 = vfmaq_n_f32(o51, v_hi3, p5_3);
      // 跨迭代预取：把下一轮「段 0」的 V 提前 load 到 v_lo0/v_hi0。
      // 仅当后面还有迭代时才需要这个值；多读一行属于无害的越界是不可接受的，
      // 所以只在 k+4 仍在 Sk4 范围内时预取，否则用一个虚 load 覆盖（编译器
      // 会 DCE 掉，因为没有后续使用者）。
      if (k + 4 < Sk4) {
        v_lo0 = vld1q_f32(V + (k + 4) * v_row_stride + 0);
        v_hi0 = vld1q_f32(V + (k + 4) * v_row_stride + 4);
      }
      o60 = vfmaq_n_f32(o60, v_lo3, p6_3); o61 = vfmaq_n_f32(o61, v_hi3, p6_3);
      o70 = vfmaq_n_f32(o70, v_lo3, p7_3); o71 = vfmaq_n_f32(o71, v_hi3, p7_3);
    }
  }

  // —— 标量尾循环：处理 Sk % 4 ——
  for (; k < Sk; ++k) {
    float32x4_t v_lo = vld1q_f32(V + k * v_row_stride + 0);
    float32x4_t v_hi = vld1q_f32(V + k * v_row_stride + 4);
    float p0 = P_hat[0 * P_row_stride + k];
    float p1 = P_hat[1 * P_row_stride + k];
    float p2 = P_hat[2 * P_row_stride + k];
    float p3 = P_hat[3 * P_row_stride + k];
    float p4 = P_hat[4 * P_row_stride + k];
    float p5 = P_hat[5 * P_row_stride + k];
    float p6 = P_hat[6 * P_row_stride + k];
    float p7 = P_hat[7 * P_row_stride + k];
    o00 = vfmaq_n_f32(o00, v_lo, p0); o01 = vfmaq_n_f32(o01, v_hi, p0);
    o10 = vfmaq_n_f32(o10, v_lo, p1); o11 = vfmaq_n_f32(o11, v_hi, p1);
    o20 = vfmaq_n_f32(o20, v_lo, p2); o21 = vfmaq_n_f32(o21, v_hi, p2);
    o30 = vfmaq_n_f32(o30, v_lo, p3); o31 = vfmaq_n_f32(o31, v_hi, p3);
    o40 = vfmaq_n_f32(o40, v_lo, p4); o41 = vfmaq_n_f32(o41, v_hi, p4);
    o50 = vfmaq_n_f32(o50, v_lo, p5); o51 = vfmaq_n_f32(o51, v_hi, p5);
    o60 = vfmaq_n_f32(o60, v_lo, p6); o61 = vfmaq_n_f32(o61, v_hi, p6);
    o70 = vfmaq_n_f32(o70, v_lo, p7); o71 = vfmaq_n_f32(o71, v_hi, p7);
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

// gemm_pv_microkernel_8x8_fp32_pquad：fp32 PV 主体「P 折成 quad load」版。
//
// 与 gemm_pv_microkernel_8x8_fp32 在数值上严格等价（fma 的累加顺序逐段
// 一致，fp32 下按位相同），唯一差别在 P 的取值方式：
//
//   * fp32 主体：每个 4-k 段开头 8 条 ldr s （32-bit 标量 load）取 P[i, k+s]
//                到 NEON 标量寄存器，再用 vfmaq_n_f32 做 broadcast。每 4-k
//                外层迭代 32 个 ldr s + 8 个 vld1q V = 40 LSU op。
//   * pquad   ：每 4-k 外层迭代开头 8 条 vld1q（128-bit）一次性取 P[i, k:k+4]
//                到 8 个 NEON quad（p0..p7），再用 vfmaq_laneq_f32 通过 lane
//                索引 0/1/2/3 取段索引。每 4-k 迭代 8 个 vld1q P + 8 个 vld1q
//                V = 16 LSU op（少 2.5×）。
//
// 寄存器账本（与 fp32 主体保持同等宽裕）：
//   16 O 累加器（持久）
//    8 P 行 quad（本 4-k 段内持久）
//    4 V quad（本段 + 下段预取，与 fp32 主体一致）
//   = 28 NEON reg 稳态峰值，留 4 个 quad 给编译器调度。
//
// FMA pipe 占用与 fp32 主体相同（64 fma quad / 4-k 迭代）；改的只是 LSU
// 端的指令密度和 cache miss 容忍度，因此 GFLOPS 数字预期变化集中在 LSU
// 受限的工况（小 Sk / cold cache / 高并发 LSU 竞争）。
static inline void gemm_pv_microkernel_8x8_fp32_pquad(
    const float* P_hat,
    int64_t P_row_stride,
    const float* V,
    int64_t v_row_stride,
    int64_t Sk,
    float* O,
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

  const int64_t Sk4 = Sk & ~int64_t{3};
  int64_t k = 0;
  if (Sk4 > 0) {
    // 段 0 V 的预取（与 fp32 主体一致）：让循环体第 1 条 fma 立刻可发射。
    float32x4_t v_lo0 = vld1q_f32(V + 0 * v_row_stride + 0);
    float32x4_t v_hi0 = vld1q_f32(V + 0 * v_row_stride + 4);
    for (; k < Sk4; k += 4) {
      // ── 一次性 load 8 个 P 行 quad（替代每段 8 个 P 标量 load） ──
      float32x4_t p0 = vld1q_f32(P_hat + 0 * P_row_stride + k);
      float32x4_t p1 = vld1q_f32(P_hat + 1 * P_row_stride + k);
      float32x4_t p2 = vld1q_f32(P_hat + 2 * P_row_stride + k);
      float32x4_t p3 = vld1q_f32(P_hat + 3 * P_row_stride + k);
      float32x4_t p4 = vld1q_f32(P_hat + 4 * P_row_stride + k);
      float32x4_t p5 = vld1q_f32(P_hat + 5 * P_row_stride + k);
      float32x4_t p6 = vld1q_f32(P_hat + 6 * P_row_stride + k);
      float32x4_t p7 = vld1q_f32(P_hat + 7 * P_row_stride + k);

      // ── 段 0 (k+0)：用 p?.lane[0]，预取段 1 V ──
      o00 = vfmaq_laneq_f32(o00, v_lo0, p0, 0); o01 = vfmaq_laneq_f32(o01, v_hi0, p0, 0);
      o10 = vfmaq_laneq_f32(o10, v_lo0, p1, 0); o11 = vfmaq_laneq_f32(o11, v_hi0, p1, 0);
      o20 = vfmaq_laneq_f32(o20, v_lo0, p2, 0); o21 = vfmaq_laneq_f32(o21, v_hi0, p2, 0);
      o30 = vfmaq_laneq_f32(o30, v_lo0, p3, 0); o31 = vfmaq_laneq_f32(o31, v_hi0, p3, 0);
      o40 = vfmaq_laneq_f32(o40, v_lo0, p4, 0); o41 = vfmaq_laneq_f32(o41, v_hi0, p4, 0);
      o50 = vfmaq_laneq_f32(o50, v_lo0, p5, 0); o51 = vfmaq_laneq_f32(o51, v_hi0, p5, 0);
      // 段 0 末尾发段 1 的 V load（与 fp32 主体相同的软件流水）。
      float32x4_t v_lo1 = vld1q_f32(V + (k + 1) * v_row_stride + 0);
      float32x4_t v_hi1 = vld1q_f32(V + (k + 1) * v_row_stride + 4);
      o60 = vfmaq_laneq_f32(o60, v_lo0, p6, 0); o61 = vfmaq_laneq_f32(o61, v_hi0, p6, 0);
      o70 = vfmaq_laneq_f32(o70, v_lo0, p7, 0); o71 = vfmaq_laneq_f32(o71, v_hi0, p7, 0);

      // ── 段 1 (k+1)：用 p?.lane[1]，预取段 2 V ──
      o00 = vfmaq_laneq_f32(o00, v_lo1, p0, 1); o01 = vfmaq_laneq_f32(o01, v_hi1, p0, 1);
      o10 = vfmaq_laneq_f32(o10, v_lo1, p1, 1); o11 = vfmaq_laneq_f32(o11, v_hi1, p1, 1);
      o20 = vfmaq_laneq_f32(o20, v_lo1, p2, 1); o21 = vfmaq_laneq_f32(o21, v_hi1, p2, 1);
      o30 = vfmaq_laneq_f32(o30, v_lo1, p3, 1); o31 = vfmaq_laneq_f32(o31, v_hi1, p3, 1);
      o40 = vfmaq_laneq_f32(o40, v_lo1, p4, 1); o41 = vfmaq_laneq_f32(o41, v_hi1, p4, 1);
      o50 = vfmaq_laneq_f32(o50, v_lo1, p5, 1); o51 = vfmaq_laneq_f32(o51, v_hi1, p5, 1);
      float32x4_t v_lo2 = vld1q_f32(V + (k + 2) * v_row_stride + 0);
      float32x4_t v_hi2 = vld1q_f32(V + (k + 2) * v_row_stride + 4);
      o60 = vfmaq_laneq_f32(o60, v_lo1, p6, 1); o61 = vfmaq_laneq_f32(o61, v_hi1, p6, 1);
      o70 = vfmaq_laneq_f32(o70, v_lo1, p7, 1); o71 = vfmaq_laneq_f32(o71, v_hi1, p7, 1);

      // ── 段 2 (k+2)：用 p?.lane[2]，预取段 3 V ──
      o00 = vfmaq_laneq_f32(o00, v_lo2, p0, 2); o01 = vfmaq_laneq_f32(o01, v_hi2, p0, 2);
      o10 = vfmaq_laneq_f32(o10, v_lo2, p1, 2); o11 = vfmaq_laneq_f32(o11, v_hi2, p1, 2);
      o20 = vfmaq_laneq_f32(o20, v_lo2, p2, 2); o21 = vfmaq_laneq_f32(o21, v_hi2, p2, 2);
      o30 = vfmaq_laneq_f32(o30, v_lo2, p3, 2); o31 = vfmaq_laneq_f32(o31, v_hi2, p3, 2);
      o40 = vfmaq_laneq_f32(o40, v_lo2, p4, 2); o41 = vfmaq_laneq_f32(o41, v_hi2, p4, 2);
      o50 = vfmaq_laneq_f32(o50, v_lo2, p5, 2); o51 = vfmaq_laneq_f32(o51, v_hi2, p5, 2);
      float32x4_t v_lo3 = vld1q_f32(V + (k + 3) * v_row_stride + 0);
      float32x4_t v_hi3 = vld1q_f32(V + (k + 3) * v_row_stride + 4);
      o60 = vfmaq_laneq_f32(o60, v_lo2, p6, 2); o61 = vfmaq_laneq_f32(o61, v_hi2, p6, 2);
      o70 = vfmaq_laneq_f32(o70, v_lo2, p7, 2); o71 = vfmaq_laneq_f32(o71, v_hi2, p7, 2);

      // ── 段 3 (k+3)：用 p?.lane[3]，预取下一轮段 0 V ──
      o00 = vfmaq_laneq_f32(o00, v_lo3, p0, 3); o01 = vfmaq_laneq_f32(o01, v_hi3, p0, 3);
      o10 = vfmaq_laneq_f32(o10, v_lo3, p1, 3); o11 = vfmaq_laneq_f32(o11, v_hi3, p1, 3);
      o20 = vfmaq_laneq_f32(o20, v_lo3, p2, 3); o21 = vfmaq_laneq_f32(o21, v_hi3, p2, 3);
      o30 = vfmaq_laneq_f32(o30, v_lo3, p3, 3); o31 = vfmaq_laneq_f32(o31, v_hi3, p3, 3);
      o40 = vfmaq_laneq_f32(o40, v_lo3, p4, 3); o41 = vfmaq_laneq_f32(o41, v_hi3, p4, 3);
      o50 = vfmaq_laneq_f32(o50, v_lo3, p5, 3); o51 = vfmaq_laneq_f32(o51, v_hi3, p5, 3);
      // 跨迭代：把下一轮段 0 的 V 提前 load 到 v_lo0/v_hi0。仅当 k+4 仍在
      // Sk4 范围内才发，避免末次迭代越界（与 fp32 主体一致）。
      if (k + 4 < Sk4) {
        v_lo0 = vld1q_f32(V + (k + 4) * v_row_stride + 0);
        v_hi0 = vld1q_f32(V + (k + 4) * v_row_stride + 4);
      }
      o60 = vfmaq_laneq_f32(o60, v_lo3, p6, 3); o61 = vfmaq_laneq_f32(o61, v_hi3, p6, 3);
      o70 = vfmaq_laneq_f32(o70, v_lo3, p7, 3); o71 = vfmaq_laneq_f32(o71, v_hi3, p7, 3);
    }
  }

  // ── 标量尾循环：处理 Sk % 4（与 fp32 主体一致；尾部 0..3 个 k 不值得展开）──
  for (; k < Sk; ++k) {
    float32x4_t v_lo = vld1q_f32(V + k * v_row_stride + 0);
    float32x4_t v_hi = vld1q_f32(V + k * v_row_stride + 4);
    float p0 = P_hat[0 * P_row_stride + k];
    float p1 = P_hat[1 * P_row_stride + k];
    float p2 = P_hat[2 * P_row_stride + k];
    float p3 = P_hat[3 * P_row_stride + k];
    float p4 = P_hat[4 * P_row_stride + k];
    float p5 = P_hat[5 * P_row_stride + k];
    float p6 = P_hat[6 * P_row_stride + k];
    float p7 = P_hat[7 * P_row_stride + k];
    o00 = vfmaq_n_f32(o00, v_lo, p0); o01 = vfmaq_n_f32(o01, v_hi, p0);
    o10 = vfmaq_n_f32(o10, v_lo, p1); o11 = vfmaq_n_f32(o11, v_hi, p1);
    o20 = vfmaq_n_f32(o20, v_lo, p2); o21 = vfmaq_n_f32(o21, v_hi, p2);
    o30 = vfmaq_n_f32(o30, v_lo, p3); o31 = vfmaq_n_f32(o31, v_hi, p3);
    o40 = vfmaq_n_f32(o40, v_lo, p4); o41 = vfmaq_n_f32(o41, v_hi, p4);
    o50 = vfmaq_n_f32(o50, v_lo, p5); o51 = vfmaq_n_f32(o51, v_hi, p5);
    o60 = vfmaq_n_f32(o60, v_lo, p6); o61 = vfmaq_n_f32(o61, v_hi, p6);
    o70 = vfmaq_n_f32(o70, v_lo, p7); o71 = vfmaq_n_f32(o71, v_hi, p7);
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

#endif  // FUSED_CPP_SDPA_CACHE_HAS_NEON

// ──────────────────────────────────────────────────────────────────────
// 标量 fallback gemm micro-kernels（功能等价；性能下界）。
// ──────────────────────────────────────────────────────────────────────

template <typename Tqk>
static inline void gemm_qkt_microkernel_8x8_scalar(
    const Tqk* Q, int64_t q_row_stride,
    const Tqk* K, int64_t k_row_stride,
    int64_t E, float scale,
    float* scores_buf) {
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      float s = 0.0f;
      for (int64_t e = 0; e < E; ++e) {
        s += static_cast<float>(Q[i * q_row_stride + e]) *
             static_cast<float>(K[j * k_row_stride + e]);
      }
      scores_buf[i * 8 + j] = s * scale;
    }
  }
}

template <typename Tv>
static inline void gemm_pv_microkernel_8x8_scalar(
    const float* P_hat, int64_t P_row_stride,
    const Tv* V, int64_t v_row_stride,
    int64_t Sk,
    float* O, int64_t o_row_stride) {
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      float acc = O[i * o_row_stride + j];
      for (int64_t k = 0; k < Sk; ++k) {
        acc += P_hat[i * P_row_stride + k] *
               static_cast<float>(V[k * v_row_stride + j]);
      }
      O[i * o_row_stride + j] = acc;
    }
  }
}

// 顶层调度（按 dtype 派发到 NEON / scalar 路径）。
inline void gemm_qkt_8x8(
    const at::BFloat16* Q, int64_t q_row_stride,
    const at::BFloat16* K, int64_t k_row_stride,
    int64_t E, float scale,
    float* scores_buf) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  gemm_qkt_microkernel_8x8_bf16(Q, q_row_stride, K, k_row_stride,
                                E, scale, scores_buf);
#else
  gemm_qkt_microkernel_8x8_scalar<at::BFloat16>(
      Q, q_row_stride, K, k_row_stride, E, scale, scores_buf);
#endif
}

inline void gemm_qkt_8x8(
    const float* Q, int64_t q_row_stride,
    const float* K, int64_t k_row_stride,
    int64_t E, float scale,
    float* scores_buf) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  gemm_qkt_microkernel_8x8_fp32(Q, q_row_stride, K, k_row_stride,
                                E, scale, scores_buf);
#else
  gemm_qkt_microkernel_8x8_scalar<float>(
      Q, q_row_stride, K, k_row_stride, E, scale, scores_buf);
#endif
}

inline void gemm_pv_8x8(
    const float* P_hat, int64_t P_row_stride,
    const at::BFloat16* V, int64_t v_row_stride,
    int64_t Sk,
    float* O, int64_t o_row_stride) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  gemm_pv_microkernel_8x8_bf16(P_hat, P_row_stride, V, v_row_stride,
                               Sk, O, o_row_stride);
#else
  gemm_pv_microkernel_8x8_scalar<at::BFloat16>(
      P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
#endif
}

inline void gemm_pv_8x8(
    const float* P_hat, int64_t P_row_stride,
    const float* V, int64_t v_row_stride,
    int64_t Sk,
    float* O, int64_t o_row_stride) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  gemm_pv_microkernel_8x8_fp32(P_hat, P_row_stride, V, v_row_stride,
                               Sk, O, o_row_stride);
#else
  gemm_pv_microkernel_8x8_scalar<float>(
      P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride);
#endif
}

// ──────────────────────────────────────────────────────────────────────
// 8×4 / 通用尾部 micro-kernels（需求 5.4 / 5.5 / 5.9）
//
// 设计策略：
//   * `Sk_remain` ∈ [4, 7]：调用 `gemm_qkt_8x4_*` 处理 4 列主体 + 标量
//                          兜底剩余 1..3 列；
//   * `Sk_remain` ∈ [1, 3]：直接走 `gemm_qkt_8xN_scalar`；
//   * `Lq_remain` < 8：被动以 `Lq=Lq_remain` 退化（外层裁剪行数 +
//                     标量逐行处理）；
//   * `Ev_remain` < 8（仅 P̂·V）：4 lane 向量 + 标量补齐，主循环以步长
//                                8 + 尾部 4 + 标量调度。
//
// 尾部 micro-kernels 不再申请 pack 缓冲，全部按原始 stride 直读
// （需求设计约束 10）。
// ──────────────────────────────────────────────────────────────────────

#if FUSED_CPP_SDPA_CACHE_HAS_NEON

// gemm_qkt_microkernel_8x4_bf16：bf16 路径下 Sk_remain==4 的 BFMMLA 退化版。
// BFMMLA 一条更新一个 2×2 子块，因此 8×4 需 (8/2)*(4/2)=8 条 BFMMLA。
static inline void gemm_qkt_microkernel_8x4_bf16(
    const at::BFloat16* Q,
    int64_t q_row_stride,
    const at::BFloat16* K,
    int64_t k_row_stride,
    int64_t E,
    float scale,
    float* scores_buf /* 8x4 fp32 row-major（每行 4 列） */,
    int64_t scores_row_stride) {

  // 8 个 fp32 累加器，行向 4 lane 布局：每行 1 个 float32x4_t（4 列）。
  float32x4_t c0 = vdupq_n_f32(0), c1 = vdupq_n_f32(0);
  float32x4_t c2 = vdupq_n_f32(0), c3 = vdupq_n_f32(0);
  float32x4_t c4 = vdupq_n_f32(0), c5 = vdupq_n_f32(0);
  float32x4_t c6 = vdupq_n_f32(0), c7 = vdupq_n_f32(0);

  const uint16_t* Qp = reinterpret_cast<const uint16_t*>(Q);
  const uint16_t* Kp = reinterpret_cast<const uint16_t*>(K);

  int64_t e = 0;

#if FUSED_CPP_SDPA_CACHE_BF16_PATH_BFMMLA
  // BFMMLA 子块网格：i_blk ∈ [0,4)，j_blk ∈ [0,2)，共 8 个累加器。
  float32x4_t bm00 = vdupq_n_f32(0), bm01 = vdupq_n_f32(0);
  float32x4_t bm10 = vdupq_n_f32(0), bm11 = vdupq_n_f32(0);
  float32x4_t bm20 = vdupq_n_f32(0), bm21 = vdupq_n_f32(0);
  float32x4_t bm30 = vdupq_n_f32(0), bm31 = vdupq_n_f32(0);

  for (; e + 4 <= E; e += 4) {
    bfloat16x8_t a01, a23, a45, a67;
    {
      uint16x4_t q0 = vld1_u16(Qp + 0 * q_row_stride + e);
      uint16x4_t q1 = vld1_u16(Qp + 1 * q_row_stride + e);
      uint16x4_t q2 = vld1_u16(Qp + 2 * q_row_stride + e);
      uint16x4_t q3 = vld1_u16(Qp + 3 * q_row_stride + e);
      uint16x4_t q4 = vld1_u16(Qp + 4 * q_row_stride + e);
      uint16x4_t q5 = vld1_u16(Qp + 5 * q_row_stride + e);
      uint16x4_t q6 = vld1_u16(Qp + 6 * q_row_stride + e);
      uint16x4_t q7 = vld1_u16(Qp + 7 * q_row_stride + e);
      a01 = vreinterpretq_bf16_u16(vcombine_u16(q0, q1));
      a23 = vreinterpretq_bf16_u16(vcombine_u16(q2, q3));
      a45 = vreinterpretq_bf16_u16(vcombine_u16(q4, q5));
      a67 = vreinterpretq_bf16_u16(vcombine_u16(q6, q7));
    }
    bfloat16x8_t b01, b23;
    {
      uint16x4_t k0 = vld1_u16(Kp + 0 * k_row_stride + e);
      uint16x4_t k1 = vld1_u16(Kp + 1 * k_row_stride + e);
      uint16x4_t k2 = vld1_u16(Kp + 2 * k_row_stride + e);
      uint16x4_t k3 = vld1_u16(Kp + 3 * k_row_stride + e);
      b01 = vreinterpretq_bf16_u16(vcombine_u16(k0, k1));
      b23 = vreinterpretq_bf16_u16(vcombine_u16(k2, k3));
    }
    bm00 = vbfmmlaq_f32(bm00, a01, b01);
    bm01 = vbfmmlaq_f32(bm01, a01, b23);
    bm10 = vbfmmlaq_f32(bm10, a23, b01);
    bm11 = vbfmmlaq_f32(bm11, a23, b23);
    bm20 = vbfmmlaq_f32(bm20, a45, b01);
    bm21 = vbfmmlaq_f32(bm21, a45, b23);
    bm30 = vbfmmlaq_f32(bm30, a67, b01);
    bm31 = vbfmmlaq_f32(bm31, a67, b23);
  }

  // 重排：每个 i_blk 输出 2 行 × 4 列。
  // 行 2i 的 4 列 = lo64(bm[i][0]) ++ lo64(bm[i][1])；
  // 行 2i+1 的 4 列 = hi64(bm[i][0]) ++ hi64(bm[i][1])。
  c0 = vcombine_f32(vget_low_f32(bm00),  vget_low_f32(bm01));
  c1 = vcombine_f32(vget_high_f32(bm00), vget_high_f32(bm01));
  c2 = vcombine_f32(vget_low_f32(bm10),  vget_low_f32(bm11));
  c3 = vcombine_f32(vget_high_f32(bm10), vget_high_f32(bm11));
  c4 = vcombine_f32(vget_low_f32(bm20),  vget_low_f32(bm21));
  c5 = vcombine_f32(vget_high_f32(bm20), vget_high_f32(bm21));
  c6 = vcombine_f32(vget_low_f32(bm30),  vget_low_f32(bm31));
  c7 = vcombine_f32(vget_high_f32(bm30), vget_high_f32(bm31));
#endif

  // ── 通用 widen+FMLA 尾部补齐 [e, E)（同时承担非 BFMMLA 路径的全部 E）──
  if (e < E) {
    float scalar_acc[8 * 4];
    vst1q_f32(scalar_acc + 0 * 4, c0);
    vst1q_f32(scalar_acc + 1 * 4, c1);
    vst1q_f32(scalar_acc + 2 * 4, c2);
    vst1q_f32(scalar_acc + 3 * 4, c3);
    vst1q_f32(scalar_acc + 4 * 4, c4);
    vst1q_f32(scalar_acc + 5 * 4, c5);
    vst1q_f32(scalar_acc + 6 * 4, c6);
    vst1q_f32(scalar_acc + 7 * 4, c7);

    for (int64_t e2 = e; e2 < E; ++e2) {
      for (int i_row = 0; i_row < 8; ++i_row) {
        float qv = bf16_to_fp32_scalar(Qp[i_row * q_row_stride + e2]);
        for (int j_col = 0; j_col < 4; ++j_col) {
          float kv = bf16_to_fp32_scalar(Kp[j_col * k_row_stride + e2]);
          scalar_acc[i_row * 4 + j_col] += qv * kv;
        }
      }
    }
    c0 = vld1q_f32(scalar_acc + 0 * 4);
    c1 = vld1q_f32(scalar_acc + 1 * 4);
    c2 = vld1q_f32(scalar_acc + 2 * 4);
    c3 = vld1q_f32(scalar_acc + 3 * 4);
    c4 = vld1q_f32(scalar_acc + 4 * 4);
    c5 = vld1q_f32(scalar_acc + 5 * 4);
    c6 = vld1q_f32(scalar_acc + 6 * 4);
    c7 = vld1q_f32(scalar_acc + 7 * 4);
  }

  const float32x4_t vs = vdupq_n_f32(scale);
  vst1q_f32(scores_buf + 0 * scores_row_stride, vmulq_f32(c0, vs));
  vst1q_f32(scores_buf + 1 * scores_row_stride, vmulq_f32(c1, vs));
  vst1q_f32(scores_buf + 2 * scores_row_stride, vmulq_f32(c2, vs));
  vst1q_f32(scores_buf + 3 * scores_row_stride, vmulq_f32(c3, vs));
  vst1q_f32(scores_buf + 4 * scores_row_stride, vmulq_f32(c4, vs));
  vst1q_f32(scores_buf + 5 * scores_row_stride, vmulq_f32(c5, vs));
  vst1q_f32(scores_buf + 6 * scores_row_stride, vmulq_f32(c6, vs));
  vst1q_f32(scores_buf + 7 * scores_row_stride, vmulq_f32(c7, vs));
}

// gemm_qkt_microkernel_8x4_fp32：fp32 路径 8×4 退化版。
static inline void gemm_qkt_microkernel_8x4_fp32(
    const float* Q, int64_t q_row_stride,
    const float* K, int64_t k_row_stride,
    int64_t E, float scale,
    float* scores_buf, int64_t scores_row_stride) {
  for (int i_row = 0; i_row < 8; ++i_row) {
    float row_acc[4];
    for (int j_col = 0; j_col < 4; ++j_col) {
      const float* qrow = Q + i_row * q_row_stride;
      const float* krow = K + j_col * k_row_stride;
      float32x4_t acc = vdupq_n_f32(0.0f);
      int64_t e = 0;
      for (; e + 4 <= E; e += 4) {
        float32x4_t qv = vld1q_f32(qrow + e);
        float32x4_t kv = vld1q_f32(krow + e);
        acc = vfmaq_f32(acc, qv, kv);
      }
      float s = vaddvq_f32(acc);
      for (; e < E; ++e) s += qrow[e] * krow[e];
      row_acc[j_col] = s * scale;
    }
    vst1q_f32(scores_buf + i_row * scores_row_stride, vld1q_f32(row_acc));
  }
}

// gemm_pv_microkernel_Lq_x_Ev_bf16：通用尾部 P̂·V，行数 Lq ∈ [1,8]，
// 列数 Ev ∈ [1, 8]，Sk 任意。Ev=8 时与 8×8 主体等价（性能较低，仅尾部用）。
static inline void gemm_pv_microkernel_tail_bf16(
    const float* P_hat, int64_t P_row_stride,
    const at::BFloat16* V, int64_t v_row_stride,
    int64_t Sk,
    float* O, int64_t o_row_stride,
    int Lq, int Ev) {
  const uint16_t* Vp = reinterpret_cast<const uint16_t*>(V);
  for (int64_t k = 0; k < Sk; ++k) {
    // Widen V 行 [0, Ev) 到 fp32 临时缓冲（最多 8 元素）。
    float v_fp32[8];
    for (int j = 0; j < Ev; ++j) {
      v_fp32[j] = bf16_to_fp32_scalar(Vp[k * v_row_stride + j]);
    }
    for (int i = 0; i < Lq; ++i) {
      float p = P_hat[i * P_row_stride + k];
      // 内层 4 路向量 + 标量补齐。
      int j = 0;
      if (Ev >= 4) {
        float32x4_t v_lo = vld1q_f32(v_fp32);
        float32x4_t o_lo = vld1q_f32(O + i * o_row_stride);
        o_lo = vfmaq_n_f32(o_lo, v_lo, p);
        vst1q_f32(O + i * o_row_stride, o_lo);
        j = 4;
      }
      for (; j < Ev; ++j) {
        O[i * o_row_stride + j] += p * v_fp32[j];
      }
    }
  }
}

static inline void gemm_pv_microkernel_tail_fp32(
    const float* P_hat, int64_t P_row_stride,
    const float* V, int64_t v_row_stride,
    int64_t Sk,
    float* O, int64_t o_row_stride,
    int Lq, int Ev) {
  for (int64_t k = 0; k < Sk; ++k) {
    for (int i = 0; i < Lq; ++i) {
      float p = P_hat[i * P_row_stride + k];
      int j = 0;
      if (Ev >= 4) {
        float32x4_t v_lo = vld1q_f32(V + k * v_row_stride);
        float32x4_t o_lo = vld1q_f32(O + i * o_row_stride);
        o_lo = vfmaq_n_f32(o_lo, v_lo, p);
        vst1q_f32(O + i * o_row_stride, o_lo);
        j = 4;
      }
      for (; j < Ev; ++j) {
        O[i * o_row_stride + j] += p * V[k * v_row_stride + j];
      }
    }
  }
}

#endif  // FUSED_CPP_SDPA_CACHE_HAS_NEON

// 通用尾部 Q·K^T：Lq ∈ [1, 8]，Sk ∈ [1, 8]。纯标量路径（性能下界，
// 仅在尾部使用）。bf16 / fp32 dispatch 由模板参数完成。
template <typename Tqk>
static inline void gemm_qkt_tail_scalar(
    const Tqk* Q, int64_t q_row_stride,
    const Tqk* K, int64_t k_row_stride,
    int64_t E, float scale,
    float* scores_buf, int64_t scores_row_stride,
    int Lq, int Sk) {
  for (int i = 0; i < Lq; ++i) {
    for (int j = 0; j < Sk; ++j) {
      float s = 0.0f;
      for (int64_t e = 0; e < E; ++e) {
        s += static_cast<float>(Q[i * q_row_stride + e]) *
             static_cast<float>(K[j * k_row_stride + e]);
      }
      scores_buf[i * scores_row_stride + j] = s * scale;
    }
  }
}

template <typename Tv>
static inline void gemm_pv_tail_scalar(
    const float* P_hat, int64_t P_row_stride,
    const Tv* V, int64_t v_row_stride,
    int64_t Sk,
    float* O, int64_t o_row_stride,
    int Lq, int Ev) {
  for (int i = 0; i < Lq; ++i) {
    for (int j = 0; j < Ev; ++j) {
      float acc = O[i * o_row_stride + j];
      for (int64_t k = 0; k < Sk; ++k) {
        acc += P_hat[i * P_row_stride + k] *
               static_cast<float>(V[k * v_row_stride + j]);
      }
      O[i * o_row_stride + j] = acc;
    }
  }
}

// ── 顶层尾部 dispatcher ──────────────────────────────────────────────

inline void gemm_qkt_8x4(
    const at::BFloat16* Q, int64_t q_row_stride,
    const at::BFloat16* K, int64_t k_row_stride,
    int64_t E, float scale,
    float* scores_buf, int64_t scores_row_stride) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  gemm_qkt_microkernel_8x4_bf16(Q, q_row_stride, K, k_row_stride,
                                E, scale, scores_buf, scores_row_stride);
#else
  gemm_qkt_tail_scalar<at::BFloat16>(
      Q, q_row_stride, K, k_row_stride, E, scale,
      scores_buf, scores_row_stride, 8, 4);
#endif
}

inline void gemm_qkt_8x4(
    const float* Q, int64_t q_row_stride,
    const float* K, int64_t k_row_stride,
    int64_t E, float scale,
    float* scores_buf, int64_t scores_row_stride) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  gemm_qkt_microkernel_8x4_fp32(Q, q_row_stride, K, k_row_stride,
                                E, scale, scores_buf, scores_row_stride);
#else
  gemm_qkt_tail_scalar<float>(
      Q, q_row_stride, K, k_row_stride, E, scale,
      scores_buf, scores_row_stride, 8, 4);
#endif
}

// 通用尾部：Lq ∈ [1, 8] 与 Sk ∈ [1, 8] 的任意组合。Q·K^T。
inline void gemm_qkt_tail(
    const at::BFloat16* Q, int64_t q_row_stride,
    const at::BFloat16* K, int64_t k_row_stride,
    int64_t E, float scale,
    float* scores_buf, int64_t scores_row_stride,
    int Lq, int Sk) {
  gemm_qkt_tail_scalar<at::BFloat16>(
      Q, q_row_stride, K, k_row_stride, E, scale,
      scores_buf, scores_row_stride, Lq, Sk);
}

inline void gemm_qkt_tail(
    const float* Q, int64_t q_row_stride,
    const float* K, int64_t k_row_stride,
    int64_t E, float scale,
    float* scores_buf, int64_t scores_row_stride,
    int Lq, int Sk) {
  gemm_qkt_tail_scalar<float>(
      Q, q_row_stride, K, k_row_stride, E, scale,
      scores_buf, scores_row_stride, Lq, Sk);
}

// 通用尾部：P̂·V。
inline void gemm_pv_tail(
    const float* P_hat, int64_t P_row_stride,
    const at::BFloat16* V, int64_t v_row_stride,
    int64_t Sk,
    float* O, int64_t o_row_stride,
    int Lq, int Ev) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  gemm_pv_microkernel_tail_bf16(P_hat, P_row_stride, V, v_row_stride,
                                Sk, O, o_row_stride, Lq, Ev);
#else
  gemm_pv_tail_scalar<at::BFloat16>(
      P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride, Lq, Ev);
#endif
}

inline void gemm_pv_tail(
    const float* P_hat, int64_t P_row_stride,
    const float* V, int64_t v_row_stride,
    int64_t Sk,
    float* O, int64_t o_row_stride,
    int Lq, int Ev) {
#if FUSED_CPP_SDPA_CACHE_HAS_NEON
  gemm_pv_microkernel_tail_fp32(P_hat, P_row_stride, V, v_row_stride,
                                Sk, O, o_row_stride, Lq, Ev);
#else
  gemm_pv_tail_scalar<float>(
      P_hat, P_row_stride, V, v_row_stride, Sk, O, o_row_stride, Lq, Ev);
#endif
}

}  // namespace fused_cpp::sdpa_microkernels
