#include "deepseek_v4_attn_gemm_sve.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_BF16)
#include <arm_sve.h>
#endif

namespace fused_cpp::deepseek_v4::attn_sve {
namespace {

inline bool env_true(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  return value[0] != '\0' && value[0] != '0' && std::strcmp(value, "false") != 0 && std::strcmp(value, "False") != 0 &&
         std::strcmp(value, "off") != 0 && std::strcmp(value, "OFF") != 0;
}

inline int round_up(int value, int q) { return ((value + q - 1) / q) * q; }

inline uint16_t bf16_bits_from_float(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  const uint32_t rounding_bias = 0x7fffu + lsb;
  bits += rounding_bias;
  return static_cast<uint16_t>(bits >> 16);
}

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_BF16)

inline int lane_count() { return static_cast<int>(svcntw()); }

inline void pack_a_block_sve(const uint16_t* A, uint16_t* packed, int rows, int K, int lda) {
  for (int kb = 0; kb < K; kb += 4) {
    for (int rp = 0; rp < 4; ++rp) {
      const int r0 = rp * 2;
      const int r1 = r0 + 1;
      uint16_t* dst = packed + static_cast<int64_t>(kb / 4) * 32 + rp * 8;
      for (int k = 0; k < 4; ++k) {
        dst[k] = r0 < rows ? A[static_cast<int64_t>(r0) * lda + kb + k] : static_cast<uint16_t>(0);
        dst[4 + k] = r1 < rows ? A[static_cast<int64_t>(r1) * lda + kb + k] : static_cast<uint16_t>(0);
      }
    }
  }
}

inline svbfloat16_t load_bf16(const uint16_t* ptr) {
  return svld1_bf16(svptrue_b16(), reinterpret_cast<const __bf16*>(ptr));
}

inline void store_rowpair_f32(float* C, int ldc, int rows, int n_tile_base, int rp, svfloat32_t c0, svfloat32_t c1,
                              svfloat32_t c2, svfloat32_t c3) {
  constexpr int kMaxLanes = 64;
  alignas(64) float tmp[kMaxLanes];
  const int lanes = lane_count();
  const svbool_t pg = svptrue_b32();
  const int row0 = rp * 2;

  for (int cp = 0; cp < 4; ++cp) {
    if (cp == 0) {
      svst1_f32(pg, tmp, c0);
    } else if (cp == 1) {
      svst1_f32(pg, tmp, c1);
    } else if (cp == 2) {
      svst1_f32(pg, tmp, c2);
    } else {
      svst1_f32(pg, tmp, c3);
    }
    for (int lane = 0; lane < lanes; ++lane) {
      const int seg = lane / 4;
      const int pos = lane & 3;
      const int row_bit = (pos >> 1) & 1;
      const int col_bit = pos & 1;
      const int row = row0 + row_bit;
      if (row >= rows) {
        continue;
      }
      const int col = n_tile_base + seg * 8 + cp * 2 + col_bit;
      C[static_cast<int64_t>(row) * ldc + col] = tmp[lane];
    }
  }
}

inline void store_rowpair_bf16(uint16_t* C, int ldc, int rows, int n_tile_base, int rp, svfloat32_t c0, svfloat32_t c1,
                               svfloat32_t c2, svfloat32_t c3) {
  constexpr int kMaxLanes = 64;
  alignas(64) float tmp[kMaxLanes];
  const int lanes = lane_count();
  const svbool_t pg = svptrue_b32();
  const int row0 = rp * 2;

  for (int cp = 0; cp < 4; ++cp) {
    if (cp == 0) {
      svst1_f32(pg, tmp, c0);
    } else if (cp == 1) {
      svst1_f32(pg, tmp, c1);
    } else if (cp == 2) {
      svst1_f32(pg, tmp, c2);
    } else {
      svst1_f32(pg, tmp, c3);
    }
    for (int lane = 0; lane < lanes; ++lane) {
      const int seg = lane / 4;
      const int pos = lane & 3;
      const int row_bit = (pos >> 1) & 1;
      const int col_bit = pos & 1;
      const int row = row0 + row_bit;
      if (row >= rows) {
        continue;
      }
      const int col = n_tile_base + seg * 8 + cp * 2 + col_bit;
      C[static_cast<int64_t>(row) * ldc + col] = bf16_bits_from_float(tmp[lane]);
    }
  }
}

template <typename StoreFn, typename CPtr>
void gemm_packed_block(const uint16_t* packed_A, const uint16_t* B_reo, CPtr C, int rows, int K, int N, int ldc,
                       StoreFn store_fn) {
  const int nt = n_tile();
  const int lanes_h = static_cast<int>(svcnth());
  const svbool_t pg = svptrue_b16();

  for (int nb = 0; nb < N; nb += nt) {
    svfloat32_t c00 = svdup_f32(0.0f), c01 = svdup_f32(0.0f);
    svfloat32_t c02 = svdup_f32(0.0f), c03 = svdup_f32(0.0f);
    svfloat32_t c10 = svdup_f32(0.0f), c11 = svdup_f32(0.0f);
    svfloat32_t c12 = svdup_f32(0.0f), c13 = svdup_f32(0.0f);
    svfloat32_t c20 = svdup_f32(0.0f), c21 = svdup_f32(0.0f);
    svfloat32_t c22 = svdup_f32(0.0f), c23 = svdup_f32(0.0f);
    svfloat32_t c30 = svdup_f32(0.0f), c31 = svdup_f32(0.0f);
    svfloat32_t c32 = svdup_f32(0.0f), c33 = svdup_f32(0.0f);

    const uint16_t* Ap = packed_A;
    const uint16_t* Bp = B_reo + static_cast<int64_t>(nb / nt) * K * nt;
    for (int kb = 0; kb < K; kb += 4) {
      svbfloat16_t b0 = load_bf16(Bp + 0 * lanes_h);
      svbfloat16_t b1 = load_bf16(Bp + 1 * lanes_h);
      svbfloat16_t b2 = load_bf16(Bp + 2 * lanes_h);
      svbfloat16_t b3 = load_bf16(Bp + 3 * lanes_h);
      Bp += 4 * lanes_h;

      svbfloat16_t a0 = svld1rq_bf16(pg, reinterpret_cast<const __bf16*>(Ap + 0));
      svbfloat16_t a1 = svld1rq_bf16(pg, reinterpret_cast<const __bf16*>(Ap + 8));
      svbfloat16_t a2 = svld1rq_bf16(pg, reinterpret_cast<const __bf16*>(Ap + 16));
      svbfloat16_t a3 = svld1rq_bf16(pg, reinterpret_cast<const __bf16*>(Ap + 24));
      c00 = svbfmmla_f32(c00, a0, b0);
      c01 = svbfmmla_f32(c01, a0, b1);
      c02 = svbfmmla_f32(c02, a0, b2);
      c03 = svbfmmla_f32(c03, a0, b3);
      c10 = svbfmmla_f32(c10, a1, b0);
      c11 = svbfmmla_f32(c11, a1, b1);
      c12 = svbfmmla_f32(c12, a1, b2);
      c13 = svbfmmla_f32(c13, a1, b3);
      c20 = svbfmmla_f32(c20, a2, b0);
      c21 = svbfmmla_f32(c21, a2, b1);
      c22 = svbfmmla_f32(c22, a2, b2);
      c23 = svbfmmla_f32(c23, a2, b3);
      c30 = svbfmmla_f32(c30, a3, b0);
      c31 = svbfmmla_f32(c31, a3, b1);
      c32 = svbfmmla_f32(c32, a3, b2);
      c33 = svbfmmla_f32(c33, a3, b3);
      Ap += 32;
    }

    store_fn(C, ldc, rows, nb, 0, c00, c01, c02, c03);
    store_fn(C, ldc, rows, nb, 1, c10, c11, c12, c13);
    store_fn(C, ldc, rows, nb, 2, c20, c21, c22, c23);
    store_fn(C, ldc, rows, nb, 3, c30, c31, c32, c33);
  }
}

#endif

}  // namespace

bool available() {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_BF16)
  return true;
#else
  return false;
#endif
}

bool enabled_by_env() { return available() && env_true("FUSED_CPP_ATTN_GEMM_SVE"); }

int n_tile() {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_BF16)
  return static_cast<int>(svcntb() / 2);
#else
  return 8;
#endif
}

int round_k(int k) { return round_up(k < 8 ? 8 : k, 8); }

int round_n(int n) { return round_up(n < 8 ? 8 : n, n_tile()); }

int64_t a_scratch_elems(int64_t m, int64_t k) {
  const int64_t rows = ((std::max<int64_t>(m, 1) + 7) / 8) * 8;
  return rows * k;
}

void pack_b(const uint16_t* B, uint16_t* B_reo, int K, int N) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_BF16)
  const int segs = static_cast<int>(svcntb() / 16);
  const int nt = segs * 8;
  int64_t idx = 0;
  for (int nb = 0; nb < N; nb += nt) {
    for (int rb = 0; rb < K / 4; ++rb) {
      const int row_base = rb * 4;
      for (int cp = 0; cp < 4; ++cp) {
        for (int sg = 0; sg < segs; ++sg) {
          const int col_base = nb + sg * 8 + cp * 2;
          for (int i = 0; i < 4; ++i) {
            B_reo[idx++] = B[static_cast<int64_t>(row_base + i) * N + col_base];
          }
          for (int i = 0; i < 4; ++i) {
            B_reo[idx++] = B[static_cast<int64_t>(row_base + i) * N + col_base + 1];
          }
        }
      }
    }
  }
#else
  (void)B;
  (void)B_reo;
  (void)K;
  (void)N;
  throw std::runtime_error("DeepSeek V4 attn SVE BF16 pack_b is unavailable");
#endif
}

void dispatch_f32(const uint16_t* A, const uint16_t* B_reo, float* C, uint16_t* A_reorder, int M, int K, int N,
                  int ldc) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_BF16)
  for (int mb = 0; mb < M; mb += 8) {
    const int rows = std::min(8, M - mb);
    uint16_t* packed = A_reorder + static_cast<int64_t>(mb) * K;
    pack_a_block_sve(A + static_cast<int64_t>(mb) * K, packed, rows, K, K);
    gemm_packed_block(packed, B_reo, C + static_cast<int64_t>(mb) * ldc, rows, K, N, ldc, store_rowpair_f32);
  }
#else
  (void)A;
  (void)B_reo;
  (void)C;
  (void)A_reorder;
  (void)M;
  (void)K;
  (void)N;
  (void)ldc;
  throw std::runtime_error("DeepSeek V4 attn SVE BF16 dispatch_f32 is unavailable");
#endif
}

void dispatch_bf16(const uint16_t* A, const uint16_t* B_reo, uint16_t* C, uint16_t* A_reorder, int M, int K, int N,
                   int ldc) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_BF16)
  for (int mb = 0; mb < M; mb += 8) {
    const int rows = std::min(8, M - mb);
    uint16_t* packed = A_reorder + static_cast<int64_t>(mb) * K;
    pack_a_block_sve(A + static_cast<int64_t>(mb) * K, packed, rows, K, K);
    gemm_packed_block(packed, B_reo, C + static_cast<int64_t>(mb) * ldc, rows, K, N, ldc, store_rowpair_bf16);
  }
#else
  (void)A;
  (void)B_reo;
  (void)C;
  (void)A_reorder;
  (void)M;
  (void)K;
  (void)N;
  (void)ldc;
  throw std::runtime_error("DeepSeek V4 attn SVE BF16 dispatch_bf16 is unavailable");
#endif
}

}  // namespace fused_cpp::deepseek_v4::attn_sve
