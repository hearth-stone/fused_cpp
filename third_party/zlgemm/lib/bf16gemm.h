// bf16gemm.h — Multi-threaded BF16 GEMM API
//
//   C = A * B          fp32  output
//   C = A * B + bias   fp32  output (bias in fp32)
//
// A: bf16  M×K  row-major (uint16_t)
// B: bf16  K×N  row-major (uint16_t)
// C: fp32  M×N  row-major (zero-init before call)
//
// Build consumer:
//   cc -c consumer.c -march=armv8.6-a+bf16 -O2 -Wall -fopenmp
//   (link with bf16gemm_k.S, bf16gemm_k_bias.S, bf16gemm_mt.c, and -fopenmp -lm)
//
// ═══════════════════════════════════════════════════════════════════════

#ifndef BF16GEMM_H
#define BF16GEMM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════
typedef uint16_t bf16_t;
typedef float    f32_t;

// ═══════════════════════════════════════════════════════════════════════
// B-packing (caller allocates B_reo: K*N*2 bytes)
//
// Required pre-step before any _dispatch call.
// For repeated calls with the same B, pack once and reuse B_reo.
// B_reo layout (outer→inner):
//   for each N-block (8 cols):
//     for each K-block (4 rows):
//       for each of 4 column-pairs: 4 K-rows of c0, then 4 K-rows of c1
// ═══════════════════════════════════════════════════════════════════════
void bf16_pack_B(const bf16_t *B, bf16_t *B_reo, int K, int N);

// Experimental reverse SVE-N12 B-packing.
//
// For bf16gemm_mt_dispatch_rev_n12 only. N must be padded to a multiple of 12.
// The kernel computes an M tile that scales with SVE VL (2VL bf16 rows) by
// loading A with ld1h and broadcasting B column-pairs with ld1rqh.
void bf16_pack_B_rev_n12(const bf16_t *B, bf16_t *B_reo, int K, int N);

// Experimental reverse SVE-N8 B-packing for bf16gemm_mt_dispatch_rev_n8_gather.
// N must be padded to a multiple of 8. A is not pre-packed by this path.
void bf16_pack_B_rev_n8(const bf16_t *B, bf16_t *B_reo, int K, int N);

// ═══════════════════════════════════════════════════════════════════════
// Core dispatch (zero-allocation, put inside timed benchmark region)
//
// A:     M × K_r  row-major bf16  (K_r multiple of 8, N_r multiple of 8)
// B_reo: pre-packed by bf16_pack_B (K_r × N_r bf16)
// C:     M × N_r  row-major fp32, zero-init before call
// bias:  length N_r fp32           (only for _bias_f variant)
// ═══════════════════════════════════════════════════════════════════════

// fp32 output
void bf16gemm_mt_dispatch(const bf16_t *A, const bf16_t *B_reo,
                           f32_t *C, int M, int K_r, int N_r,
                           int nthreads);

// Experimental reverse SVE-N12 dispatch. A is M×K_r row-major and is packed
// inside the dispatcher; B_reo must be packed by bf16_pack_B_rev_n12.
void bf16gemm_mt_dispatch_rev_n12(const bf16_t *A, const bf16_t *B_reo,
                                   f32_t *C, int M, int K_r, int N_r,
                                   int nthreads);

// Experimental reverse SVE-N8 dispatch. A is M×K_r row-major and is gathered
// directly inside the kernel; B_reo must be packed by bf16_pack_B_rev_n8.
void bf16gemm_mt_dispatch_rev_n8_gather(const bf16_t *A, const bf16_t *B_reo,
                                         f32_t *C, int M, int K_r, int N_r,
                                         int nthreads);

// fp32 output + bias (accumulator zero-init, bias added in fp32 on store)
void bf16gemm_mt_dispatch_bias_f(const bf16_t *A, const bf16_t *B_reo,
                                  f32_t *C, int M, int K_r, int N_r,
                                  int nthreads, const f32_t *bias);

// ═══════════════════════════════════════════════════════════════════════
// Convenience wrappers (handles padding, B-packing internally)
//
// A_orig: M×K row-major bf16  (K, N need not be multiples of 8)
// B_orig: K×N row-major bf16
// C:      M×N row-major fp32, zero-init before call
// bias:   length N fp32
// ═══════════════════════════════════════════════════════════════════════

// fp32 output
void bf16gemm_mt(const bf16_t *A_orig, const bf16_t *B_orig,
                  f32_t *C, int M, int K, int N,
                  int nthreads);

// fp32 output + bias
void bf16gemm_mt_bias_f(const bf16_t *A_orig, const bf16_t *B_orig,
                         f32_t *C, int M, int K, int N,
                         int nthreads, const f32_t *bias);

// ═══════════════════════════════════════════════════════════════════════
// bf16 output, no C load (accumulators zero-init, _nld_b suffix)
//
// C:     M × N_r  row-major bf16    (zero-init before call for dispatch)
// bias:  not supported
// ═══════════════════════════════════════════════════════════════════════

// Core dispatch: C is bf16* (not f32*). ldc_shift=1, c_adv=16.
void bf16gemm_mt_dispatch_nld_b(const bf16_t *A, const bf16_t *B_reo,
                                 bf16_t *C, int M, int K_r, int N_r,
                                 int nthreads);

// Convenience wrapper
void bf16gemm_mt_nld_b(const bf16_t *A_orig, const bf16_t *B_orig,
                        bf16_t *C, int M, int K, int N,
                        int nthreads);

#ifdef __cplusplus
}
#endif

#endif // BF16GEMM_H
