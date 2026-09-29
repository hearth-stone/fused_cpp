// gemm_params.h — Internal GEMM dimension struct (shared by bf16 and i8 kernels)
//
// Pack M/K/N/lda/ldb/ldc into one struct so the assembly kernel receives
// a single pointer instead of 10 scalar arguments (avoids stack spill).
// This is an internal detail — external APIs keep their normal signatures.

#ifndef GEMM_PARAMS_H
#define GEMM_PARAMS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int m;      // M dimension
    int k;      // K dimension
    int n;      // N dimension
    int lda;    // A leading dimension in elements (bf16: row stride in uint16_t; i8: row stride in bytes)
    int ldb;    // K_big of packed B — N-block stride in elements
    int ldc;    // C leading dimension in elements (or row stride in f32/i32 elements)
} gemm_params_t;

#ifdef __cplusplus
}
#endif

#endif // GEMM_PARAMS_H
