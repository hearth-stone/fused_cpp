// reference.h
// Naive C++ 参考实现，无 NEON 优化、无 widen 技巧；用于正确性校验。
//
// 关键约定（与 microkernel 严格对应）：
//   * bf16 用 uint16_t 存储；widen 通过 (uint32(bf) << 16).as<float>()
//   * QKᵀ：scores[i][j] = scale * sum_e(Q[i,e] * K[j,e])，scores 行向 row stride=8
//   * PV：O[i][ev] += sum_k(P_hat[i,k] * V[k,ev])，**累加**进 O（与 microkernel
//     的 vld1q_f32(O) → fma → vst1q_f32(O) 语义一致）

#pragma once

#include <cstdint>
#include <cstring>

static inline float bf16_to_fp32_ref(uint16_t bf) {
  uint32_t u = static_cast<uint32_t>(bf) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

// QKᵀ: scores_buf[i*8 + j] = scale * sum_e(Q[i*qs + e] * K[j*ks + e])
static inline void qkt_ref(
    const uint16_t* Q, int64_t q_row_stride,
    const uint16_t* K, int64_t k_row_stride,
    int64_t E, float scale,
    float* scores_buf) {
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      float s = 0.0f;
      for (int64_t e = 0; e < E; ++e) {
        float qv = bf16_to_fp32_ref(Q[i * q_row_stride + e]);
        float kv = bf16_to_fp32_ref(K[j * k_row_stride + e]);
        s += qv * kv;
      }
      scores_buf[i * 8 + j] = s * scale;
    }
  }
}

// PV: O[i*os + ev] += sum_k(P_hat[i*ps + k] * V[k*vs + ev])
// 与 microkernel 一样 accumulate 进 O。
static inline void pv_ref(
    const float* P_hat, int64_t P_row_stride,
    const uint16_t* V, int64_t v_row_stride,
    int64_t Sk,
    float* O, int64_t o_row_stride) {
  for (int i = 0; i < 8; ++i) {
    for (int ev = 0; ev < 8; ++ev) {
      float s = O[i * o_row_stride + ev];
      for (int64_t k = 0; k < Sk; ++k) {
        float p = P_hat[i * P_row_stride + k];
        float vv = bf16_to_fp32_ref(V[k * v_row_stride + ev]);
        s += p * vv;
      }
      O[i * o_row_stride + ev] = s;
    }
  }
}
