# Sparse Indexer SVE 8x2VL on Arm-codex-internal

Date: 2026-08-20

## Configuration

- Host: `Arm-codex-internal`
- Affinity: NUMA0 cores `0-(T-1)`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`
- Post-stage shape: `M=2048`, `context_start=2048`, compressed context 1024
- Score shape: 64 heads, head dimension 128, 1024 compressed keys
- Useful score work: `2 * 2048 * 64 * 128 * 1024 = 34.360 GFLOP`
- Candidate: SVE BF16 BFMMLA `8-head x 2VL-key`, fused per-head ReLU, weights and head reduction
- Baseline: 64 sequential FP32 ATen GEMMs with separate ReLU/weight/add
- Warmup/runs: 3/9; stage values are one additional warmed profile invocation

## Score Kernel

| Threads | FP32 fallback | SVE 8x2VL | Speedup | SVE TFLOP/s | SVE 1T linear efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 515.760 ms | 453.874 ms | 1.14x | 0.076 | 100.0% |
| 2 | 283.327 ms | 227.856 ms | 1.24x | 0.151 | 99.6% |
| 4 | 171.591 ms | 113.405 ms | 1.51x | 0.303 | 100.1% |
| 8 | 86.784 ms | 56.731 ms | 1.53x | 0.606 | 100.0% |
| 16 | 44.541 ms | 28.660 ms | 1.55x | 1.199 | 99.0% |
| 32 | 25.698 ms | 13.958 ms | 1.84x | 2.462 | 101.6% |
| 40 | 22.180 ms | 11.109 ms | 2.00x | 3.093 | 102.1% |
| 64 | 18.258 ms | 6.827 ms | 2.67x | 5.033 | 103.9% |
| 80 | 17.235 ms | 5.471 ms | 3.15x | 6.280 | 103.7% |

The greater-than-100% relative efficiencies at high width are measured scaling,
not a peak-efficiency claim; they include topology/cache effects relative to the
single-thread reference. At 80 threads the score kernel reaches 85.3% of the
7.36 TFLOP/s BF16 reference.

## End-to-End Effect

| Threads | Fallback post-stage | SVE-specialized post-stage | Throughput change |
| ---: | ---: | ---: | ---: |
| 1 | 1408.463 ms | 1358.202 ms | +3.70% |
| 2 | 734.644 ms | 679.522 ms | +8.11% |
| 4 | 400.158 ms | 341.704 ms | +17.11% |
| 8 | 201.661 ms | 171.424 ms | +17.64% |
| 16 | 102.584 ms | 85.787 ms | +19.58% |
| 32 | 55.285 ms | 43.652 ms | +26.65% |
| 40 | 46.461 ms | 35.516 ms | +30.81% |
| 64 | 33.887 ms | 22.220 ms | +52.51% |
| 80 | 30.034 ms | 18.184 ms | +65.17% |

At 80 threads, score+TopK falls from 18.019 to 6.262 ms and the profile reports
`sparse_indexer_backend=sve_8x2vl`, `sparse_indexer_n_tile=16`. The remaining
largest stage is dual-Q GEMM at 10.020 ms; sparse score is 5.471 ms and TopK is
0.790 ms.

## Build Root Cause

The source had been compiled as a generic object whose `available()` returned
false. Moving it to the existing fixed-VL SVE native-source channel was not
sufficient by itself because GCC 13 defines
`__ARM_FEATURE_BF16_VECTOR_ARITHMETIC` rather than `__ARM_FEATURE_BF16`.
Accepting either macro produces the intended SVE implementation while non-SVE
builds retain the stub and fallback.
