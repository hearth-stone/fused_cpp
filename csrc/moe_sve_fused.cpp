#include "moe_sve_fused.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_BF16)
#include <arm_sve.h>
#endif

namespace fused_cpp::moe_sve {
namespace {

inline uint16_t bf16_bits_from_float(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1u;
    const uint32_t rounding_bias = 0x7fffu + lsb;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

inline bool env_false(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    return value[0] == '0' || value[0] == '\0' ||
           std::strcmp(value, "false") == 0 ||
           std::strcmp(value, "False") == 0 ||
           std::strcmp(value, "off") == 0 ||
           std::strcmp(value, "OFF") == 0;
}

inline int round_up(int x, int q) {
    return ((x + q - 1) / q) * q;
}

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_BF16)

inline svbfloat16_t load_bf16(const uint16_t* ptr) {
    return svld1_bf16(svptrue_b16(), reinterpret_cast<const __bf16*>(ptr));
}

inline svfloat32_t exp_poly_neg(svfloat32_t gate, int64_t degree) {
    const svbool_t pg = svptrue_b32();
    svfloat32_t x = svneg_f32_x(pg, gate);
    x = svmin_n_f32_x(pg, x, 87.0f);
    x = svmax_n_f32_x(pg, x, -87.0f);

    svfloat32_t fn = svmul_n_f32_x(pg, x, 1.4426950408889634f);
    fn = svrintn_f32_x(pg, fn);
    svint32_t ni = svcvt_s32_f32_x(pg, fn);
    svfloat32_t r = svmls_n_f32_x(pg, x, fn, 0.6931471805599453f);

    svfloat32_t poly;
    if (degree == 4) {
        poly = svdup_f32(0.04166666f);
    } else if (degree == 5) {
        poly = svmla_n_f32_x(pg, svdup_f32(0.04166666f), r, 0.00833333f);
    } else {
        svfloat32_t t = svmla_n_f32_x(
            pg, svdup_f32(0.00833333f), r, 0.0013888889f);
        poly = svmla_f32_x(pg, svdup_f32(0.04166666f), t, r);
    }
    svfloat32_t t3 = svmla_f32_x(pg, svdup_f32(0.16666666f), poly, r);
    svfloat32_t t2 = svmla_f32_x(pg, svdup_f32(0.5f), t3, r);
    svfloat32_t t1 = svmla_f32_x(pg, svdup_f32(1.0f), t2, r);
    poly = svmla_f32_x(pg, svdup_f32(1.0f), t1, r);

    svint32_t exp_bits =
        svlsl_n_s32_x(pg, svadd_n_s32_x(pg, ni, 127), 23);
    svfloat32_t pow2 = svreinterpret_f32_s32(exp_bits);
    return svmul_f32_x(pg, poly, pow2);
}

inline svfloat32_t silu_mul(svfloat32_t gate, svfloat32_t up, int64_t degree) {
    const svbool_t pg = svptrue_b32();
    svfloat32_t denom =
        svadd_n_f32_x(pg, exp_poly_neg(gate, degree), 1.0f);
    svfloat32_t num = svmul_f32_x(pg, gate, up);
    return svdiv_f32_x(pg, num, denom);
}

inline void pack_a_block_sve(const uint16_t* A, uint16_t* packed, int rows,
                             int K, int lda) {
    for (int kb = 0; kb < K; kb += 4) {
        for (int rp = 0; rp < 4; ++rp) {
            const int r0 = rp * 2;
            const int r1 = r0 + 1;
            uint16_t* dst = packed + static_cast<int64_t>(kb / 4) * 32 + rp * 8;
            for (int k = 0; k < 4; ++k) {
                dst[k] = r0 < rows ? A[static_cast<int64_t>(r0) * lda + kb + k]
                                   : static_cast<uint16_t>(0);
                dst[4 + k] =
                    r1 < rows ? A[static_cast<int64_t>(r1) * lda + kb + k]
                              : static_cast<uint16_t>(0);
            }
        }
    }
}

inline int lane_count() {
    return static_cast<int>(svcntw());
}

inline void store_w13_rowpair_rowmajor(uint16_t* C, int ldc, int rows,
                                       int n_tile_base, int rp,
                                       svfloat32_t g0, svfloat32_t g1,
                                       svfloat32_t u0, svfloat32_t u1,
                                       int64_t degree) {
    constexpr int kMaxLanes = 64;
    alignas(64) float out0[kMaxLanes];
    alignas(64) float out1[kMaxLanes];
    const int lanes = lane_count();
    const svbool_t pg = svptrue_b32();
    const int row0 = rp * 2;
    const int row1 = row0 + 1;
    svst1_f32(pg, out0, silu_mul(g0, u0, degree));
    svst1_f32(pg, out1, silu_mul(g1, u1, degree));
    for (int lane = 0; lane < lanes; ++lane) {
        const int seg = lane / 4;
        const int pos = lane & 3;
        const int row_bit = (pos >> 1) & 1;
        const int col_bit = pos & 1;
        const int row = (row_bit == 0) ? row0 : row1;
        if (row >= rows) {
            continue;
        }
        const int f0 = n_tile_base / 2 + seg * 4 + col_bit;
        const int f1 = n_tile_base / 2 + seg * 4 + 2 + col_bit;
        C[static_cast<int64_t>(row) * ldc + f0] =
            bf16_bits_from_float(out0[lane]);
        C[static_cast<int64_t>(row) * ldc + f1] =
            bf16_bits_from_float(out1[lane]);
    }
}

inline void store_w13_rowpair_packc(uint16_t* C, int rows, int n_tile_base,
                                    int rp, svfloat32_t g0, svfloat32_t g1,
                                    svfloat32_t u0, svfloat32_t u1,
                                    int64_t degree) {
    constexpr int kMaxLanes = 64;
    alignas(64) float out0[kMaxLanes];
    alignas(64) float out1[kMaxLanes];
    const int lanes = lane_count();
    const svbool_t pg = svptrue_b32();
    const int row0 = rp * 2;
    svst1_f32(pg, out0, silu_mul(g0, u0, degree));
    svst1_f32(pg, out1, silu_mul(g1, u1, degree));
    for (int lane = 0; lane < lanes; ++lane) {
        const int seg = lane / 4;
        const int pos = lane & 3;
        const int row_bit = (pos >> 1) & 1;
        const int col_bit = pos & 1;
        const int row = row0 + row_bit;
        if (row >= rows) {
            continue;
        }
        const int feature_block = n_tile_base / 2 / 4 + seg;
        const int base = feature_block * 32 + (row / 2) * 8 + row_bit * 4;
        C[base + col_bit] = bf16_bits_from_float(out0[lane]);
        C[base + 2 + col_bit] = bf16_bits_from_float(out1[lane]);
    }
}

inline void store_w2_rowpair(float* C, int ldc, int rows, int n_tile_base,
                             int rp, svfloat32_t c0, svfloat32_t c1,
                             svfloat32_t c2, svfloat32_t c3) {
    constexpr int kMaxLanes = 64;
    alignas(64) float tmp[kMaxLanes];
    const int lanes = lane_count();
    const svbool_t pg = svptrue_b32();
    const int row0 = rp * 2;
    const int row1 = row0 + 1;
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
            const int row = (row_bit == 0) ? row0 : row1;
            if (row >= rows) {
                continue;
            }
            const int col = n_tile_base + seg * 8 + cp * 2 + col_bit;
            C[static_cast<int64_t>(row) * ldc + col] = tmp[lane];
        }
    }
}

enum class StoreKind {
    kW13Rowmajor,
    kW13PackC,
    kW2Rowmajor,
};

void gemm_packed_block(const uint16_t* packed_A, const uint16_t* B_reo,
                       void* C, int rows, int K, int N, int ldc,
                       StoreKind kind, int64_t degree) {
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
            svbfloat16_t a0 =
                svld1rq_bf16(pg, reinterpret_cast<const __bf16*>(Ap + 0));
            svbfloat16_t a1 =
                svld1rq_bf16(pg, reinterpret_cast<const __bf16*>(Ap + 8));
            svbfloat16_t a2 =
                svld1rq_bf16(pg, reinterpret_cast<const __bf16*>(Ap + 16));
            svbfloat16_t a3 =
                svld1rq_bf16(pg, reinterpret_cast<const __bf16*>(Ap + 24));
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
        if (kind == StoreKind::kW13Rowmajor) {
            uint16_t* out = static_cast<uint16_t*>(C);
            store_w13_rowpair_rowmajor(out, ldc, rows, nb, 0, c00, c01, c02,
                                       c03, degree);
            store_w13_rowpair_rowmajor(out, ldc, rows, nb, 1, c10, c11, c12,
                                       c13, degree);
            store_w13_rowpair_rowmajor(out, ldc, rows, nb, 2, c20, c21, c22,
                                       c23, degree);
            store_w13_rowpair_rowmajor(out, ldc, rows, nb, 3, c30, c31, c32,
                                       c33, degree);
        } else if (kind == StoreKind::kW13PackC) {
            uint16_t* out = static_cast<uint16_t*>(C);
            store_w13_rowpair_packc(out, rows, nb, 0, c00, c01, c02, c03,
                                    degree);
            store_w13_rowpair_packc(out, rows, nb, 1, c10, c11, c12, c13,
                                    degree);
            store_w13_rowpair_packc(out, rows, nb, 2, c20, c21, c22, c23,
                                    degree);
            store_w13_rowpair_packc(out, rows, nb, 3, c30, c31, c32, c33,
                                    degree);
        } else {
            float* out = static_cast<float*>(C);
            store_w2_rowpair(out, ldc, rows, nb, 0, c00, c01, c02, c03);
            store_w2_rowpair(out, ldc, rows, nb, 1, c10, c11, c12, c13);
            store_w2_rowpair(out, ldc, rows, nb, 2, c20, c21, c22, c23);
            store_w2_rowpair(out, ldc, rows, nb, 3, c30, c31, c32, c33);
        }
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

bool enabled_by_env() {
    return available() && !env_false("FUSED_CPP_MOE_SVE");
}

int n_tile() {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_BF16)
    return static_cast<int>(svcntb() / 2);
#else
    return 8;
#endif
}

int round_k(int k) {
    return round_up(k < 8 ? 8 : k, 8);
}

int round_n(int n) {
    return round_up(n < 8 ? 8 : n, n_tile());
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
                        B_reo[idx++] =
                            B[static_cast<int64_t>(row_base + i) * N + col_base];
                    }
                    for (int i = 0; i < 4; ++i) {
                        B_reo[idx++] =
                            B[static_cast<int64_t>(row_base + i) * N +
                              col_base + 1];
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
    throw std::runtime_error("SVE BF16 MoE pack_b is unavailable");
#endif
}

void pack_a_block(const uint16_t* A, uint16_t* packed, int rows, int K) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_BF16)
    pack_a_block_sve(A, packed, rows, K, K);
#else
    (void)A;
    (void)packed;
    (void)rows;
    (void)K;
    throw std::runtime_error("SVE BF16 MoE pack_a_block is unavailable");
#endif
}

void gather_pack_a(const uint16_t* input, int64_t H,
                   const int64_t* expert_routes, int64_t top_k,
                   uint16_t* packed, int total_rows, int K_pad,
                   int block_begin, int block_end) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_BF16)
    for (int mb = block_begin; mb < block_end; ++mb) {
        uint16_t* block = packed + static_cast<int64_t>(mb) * 8 * K_pad;
        for (int kb = 0; kb < K_pad; kb += 4) {
            for (int rp = 0; rp < 4; ++rp) {
                for (int row_bit = 0; row_bit < 2; ++row_bit) {
                    const int r = rp * 2 + row_bit;
                    const int gr = mb * 8 + r;
                    uint16_t* dst =
                        block + static_cast<int64_t>(kb / 4) * 32 + rp * 8 +
                        row_bit * 4;
                    const uint16_t* src = nullptr;
                    if (gr < total_rows) {
                        const int64_t flat = expert_routes[gr];
                        src = input + (flat / top_k) * H;
                    }
                    for (int k = 0; k < 4; ++k) {
                        const int col = kb + k;
                        dst[k] = (src != nullptr && col < H)
                                     ? src[col]
                                     : static_cast<uint16_t>(0);
                    }
                }
            }
        }
    }
#else
    (void)input;
    (void)H;
    (void)expert_routes;
    (void)top_k;
    (void)packed;
    (void)total_rows;
    (void)K_pad;
    (void)block_begin;
    (void)block_end;
    throw std::runtime_error("SVE BF16 MoE gather_pack_a is unavailable");
#endif
}

void w13_silu_rowmajor(const uint16_t* A, const uint16_t* B_reo,
                       uint16_t* C, uint16_t* A_reorder,
                       const gemm_params_t* params, int64_t degree) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_BF16)
    const int M = params->m;
    const int K = params->k;
    const int N = params->n;
    const int ldc = params->ldc;
    for (int mb = 0; mb < M; mb += 8) {
        const int rows = std::min(8, M - mb);
        uint16_t* packed = A_reorder + static_cast<int64_t>(mb) * K;
        pack_a_block_sve(A + static_cast<int64_t>(mb) * params->lda,
                         packed, rows, K, params->lda);
        gemm_packed_block(packed, B_reo,
                          C + static_cast<int64_t>(mb) * ldc, rows, K, N,
                          ldc, StoreKind::kW13Rowmajor, degree);
    }
#else
    (void)A;
    (void)B_reo;
    (void)C;
    (void)A_reorder;
    (void)params;
    (void)degree;
    throw std::runtime_error("SVE BF16 MoE w13_silu_rowmajor is unavailable");
#endif
}

void w13_silu_packed(const uint16_t* packed_A, const uint16_t* B_reo,
                     uint16_t* C, const gemm_params_t* params,
                     int64_t degree) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_BF16)
    const int M = params->m;
    const int K = params->k;
    const int N = params->n;
    const int ldc = params->ldc;
    for (int mb = 0; mb < M; mb += 8) {
        const int rows = std::min(8, M - mb);
        const uint16_t* block = packed_A + static_cast<int64_t>(mb) * K;
        gemm_packed_block(block, B_reo,
                          C + static_cast<int64_t>(mb) * ldc, rows, K, N,
                          ldc, StoreKind::kW13Rowmajor, degree);
    }
#else
    (void)packed_A;
    (void)B_reo;
    (void)C;
    (void)params;
    (void)degree;
    throw std::runtime_error("SVE BF16 MoE w13_silu_packed is unavailable");
#endif
}

void w13_silu_packc(const uint16_t* packed_A, const uint16_t* B_reo,
                    uint16_t* C, const gemm_params_t* params,
                    int64_t rows_total, int64_t degree) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_BF16)
    const int K = params->k;
    const int N = params->n;
    const int ldc = params->ldc;
    for (int mb = 0; mb < rows_total; mb += 8) {
        const int rows = std::min<int64_t>(8, rows_total - mb);
        const uint16_t* block = packed_A + static_cast<int64_t>(mb) * K;
        uint16_t* Cb = C + static_cast<int64_t>(mb / 8) * 8 * ldc;
        gemm_packed_block(block, B_reo, Cb, rows, K, N, ldc,
                          StoreKind::kW13PackC, degree);
    }
#else
    (void)packed_A;
    (void)B_reo;
    (void)C;
    (void)params;
    (void)rows_total;
    (void)degree;
    throw std::runtime_error("SVE BF16 MoE w13_silu_packc is unavailable");
#endif
}

void w2_packed(const uint16_t* packed_A, const uint16_t* B_reo,
               float* C, const gemm_params_t* params, int64_t rows_total) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_BF16)
    const int K = params->k;
    const int N = params->n;
    const int ldc = params->ldc;
    for (int mb = 0; mb < rows_total; mb += 8) {
        const int rows = std::min<int64_t>(8, rows_total - mb);
        const uint16_t* block = packed_A + static_cast<int64_t>(mb) * K;
        float* Cb = C + static_cast<int64_t>(mb) * ldc;
        gemm_packed_block(block, B_reo, Cb, rows, K, N, ldc,
                          StoreKind::kW2Rowmajor, 0);
    }
#else
    (void)packed_A;
    (void)B_reo;
    (void)C;
    (void)params;
    (void)rows_total;
    throw std::runtime_error("SVE BF16 MoE w2_packed is unavailable");
#endif
}

void w2_rowmajor(const uint16_t* A, const uint16_t* B_reo, float* C,
                 uint16_t* A_reorder, const gemm_params_t* params) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_BF16)
    const int M = params->m;
    const int K = params->k;
    const int N = params->n;
    const int ldc = params->ldc;
    for (int mb = 0; mb < M; mb += 8) {
        const int rows = std::min(8, M - mb);
        uint16_t* packed = A_reorder + static_cast<int64_t>(mb) * K;
        pack_a_block_sve(A + static_cast<int64_t>(mb) * params->lda,
                         packed, rows, K, params->lda);
        gemm_packed_block(packed, B_reo,
                          C + static_cast<int64_t>(mb) * ldc, rows, K, N,
                          ldc, StoreKind::kW2Rowmajor, 0);
    }
#else
    (void)A;
    (void)B_reo;
    (void)C;
    (void)A_reorder;
    (void)params;
    throw std::runtime_error("SVE BF16 MoE w2_rowmajor is unavailable");
#endif
}

}  // namespace fused_cpp::moe_sve
