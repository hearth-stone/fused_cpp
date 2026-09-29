// i8gemm.h — Multi-threaded I8 GEMM API
//
//   C = A * B          int32  output
//   C = A * B + bias   fp32   output (bias in fp32)
//
// A: int8   M×K  row-major
// B: int8   K×N  row-major
// C: int32  M×N  row-major (zero-init before call)
//    or fp32 M×N  row-major for _f / _bias_f variants
//
// Build consumer:
//   cc -c consumer.c -march=armv8.6-a+i8mm -O2 -Wall -fopenmp
//   (link with i8gemm_k.S, i8gemm_k_bias.S, i8gemm_mt.c, and -fopenmp -lm)
//
// ═══════════════════════════════════════════════════════════════════════

#ifndef I8GEMM_H
#define I8GEMM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════
typedef int8_t  i8_t;
typedef int32_t i32_t;
typedef float   f32_t;

// ═══════════════════════════════════════════════════════════════════════
// B-packing (caller allocates B_reo: K*N bytes)
//
// Required pre-step before any _dispatch call.
// For repeated calls with the same B, pack once and reuse B_reo.
// B_reo layout (outer→inner):
//   for each N-block (8 cols):
//     for each K-block (8 rows):
//       for each column j in N-block:
//         for each row i in K-block:
//           B[Kblock*8 + i][Nblock*8 + j]
// ═══════════════════════════════════════════════════════════════════════
void i8_pack_B(const i8_t *B, i8_t *B_reo, int K, int N);

// ═══════════════════════════════════════════════════════════════════════
// Core dispatch (zero-allocation, put inside timed benchmark region)
//
// A:     M × K_r  row-major int8   (K_r multiple of 16, N_r multiple of 8)
// B_reo: pre-packed by i8_pack_B   (K_r × N_r int8)
// C:     M × N_r  row-major, zero-init before call
// bias:  length N_r fp32           (only for _bias_f variants)
// ═══════════════════════════════════════════════════════════════════════

// int32 output
void i8gemm_mt_dispatch(const i8_t *A, const i8_t *B_reo,
                         i32_t *C, int M, int K_r, int N_r,
                         int nthreads);

// Advisory: recommended thread count ("knee") for a shape on this machine.
// Returns the fewest threads expected to reach near-peak GOPS, bounded by the
// work-unit count and (for bandwidth-bound large-B shapes) a per-thread MAC
// floor (env I8_REC_MACS, default 512Ki; only applied when K_r*N_r exceeds
// I8_REC_CACHE, default 2Mi). ADVISORY ONLY: the floors are hardware-dependent;
// calibrate against a sweep. i8gemm_mt_dispatch uses it only when called with
// nthreads<=0 AND env I8_SVE_AUTO_THREADS=1; otherwise the explicit thread
// count is honoured.
int i8gemm_recommend_threads(int M, int K_r, int N_r, int max_threads);

// fp32 output (int32 accumulator → scvtf on store)
void i8gemm_mt_dispatch_f(const i8_t *A, const i8_t *B_reo,
                           f32_t *C, int M, int K_r, int N_r,
                           int nthreads);

// fp32 output + bias (accumulator zero-init, bias added in fp32 on store)
void i8gemm_mt_dispatch_bias_f(const i8_t *A, const i8_t *B_reo,
                                f32_t *C, int M, int K_r, int N_r,
                                int nthreads, const f32_t *bias);

// Direct scaled output. The microkernel converts int32 accumulators in
// registers, applies row_scale[m] * column_scale[n] and optional FP32 bias,
// then stores the requested final dtype without an MxN FP32 intermediate.
// N_r and the column-scale/bias arrays cover complete backend N tiles.
void i8gemm_mt_dispatch_scaled_f(const i8_t *A, const i8_t *B_reo,
                                  f32_t *C, int M, int K_r, int N_r,
                                  int nthreads, const f32_t *row_scale,
                                  const f32_t *column_scale,
                                  const f32_t *bias);
void i8gemm_mt_dispatch_scaled_b(const i8_t *A, const i8_t *B_reo,
                                  uint16_t *C, int M, int K_r, int N_r,
                                  int nthreads, const f32_t *row_scale,
                                  const f32_t *column_scale,
                                  const f32_t *bias);
void i8gemm_mt_dispatch_scaled_pair_b(
    const i8_t *A, const i8_t *first_B_reo, uint16_t *first_C,
    int first_N_r, const f32_t *first_column_scale, const f32_t *first_bias,
    const i8_t *second_B_reo, uint16_t *second_C, int second_N_r,
    const f32_t *second_column_scale, const f32_t *second_bias,
    int M, int K_r, int nthreads, const f32_t *row_scale);

// ═══════════════════════════════════════════════════════════════════════
// Convenience wrappers (handles padding, B-packing internally)
//
// A_orig: M×K row-major int8   (K, N need not be multiples of 8/16)
// B_orig: K×N row-major int8
// C:      M×N row-major, zero-init before call
// bias:   length N fp32
// ═══════════════════════════════════════════════════════════════════════

// int32 output
void i8gemm_mt(const i8_t *A_orig, const i8_t *B_orig,
                i32_t *C, int M, int K, int N,
                int nthreads);

// fp32 output
void i8gemm_mt_f(const i8_t *A_orig, const i8_t *B_orig,
                  f32_t *C, int M, int K, int N,
                  int nthreads);

// fp32 output + bias
void i8gemm_mt_bias_f(const i8_t *A_orig, const i8_t *B_orig,
                       f32_t *C, int M, int K, int N,
                       int nthreads, const f32_t *bias);

#ifdef __cplusplus
}
#endif

#endif // I8GEMM_H
