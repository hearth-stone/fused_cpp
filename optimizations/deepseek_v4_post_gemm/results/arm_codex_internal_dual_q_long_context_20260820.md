# Long-Context Dual-Q GEMM on Arm-codex-internal

Date: 2026-08-20

## Configuration

- Host: `Arm-codex-internal`
- Affinity: NUMA0 cores `0-(T-1)`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`
- Shape: Main Q and Indexer Q each `(2048,1024) x (1024,8192)`, BF16 output
- Context: `context_start=2048`, compressed valid length reaches 1024 and exceeds `topk_tokens=512`
- Useful dual-Q work: `4 * 2048 * 1024 * 8192 = 68.719 GFLOP`
- Backends: explicit NEON M8 and SVE exact-M/M12, aligned/window scheduling
- Warmup/runs/statistic: 3/9/median for the complete call
- GEMM-stage timing: one additional profiled invocation after warmup

## Dual-Q GEMM Stage

For NEON below 32 threads, the runtime executes Main Q and Indexer Q
sequentially, so the table sums `main_q_gemm_ms + indexer_q_gemm_ms`. At 32
threads and above, and for SVE at every width, it reports `shared_q_gemm_ms`.

| Threads | NEON time | NEON TFLOP/s | NEON linear efficiency | SVE time | SVE TFLOP/s | SVE linear efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 835.256 ms | 0.082 | 100.0% | 794.686 ms | 0.086 | 100.0% |
| 2 | 417.652 ms | 0.165 | 100.0% | 395.943 ms | 0.174 | 100.4% |
| 4 | 209.292 ms | 0.328 | 99.8% | 198.675 ms | 0.346 | 100.0% |
| 8 | 104.734 ms | 0.656 | 99.7% | 99.241 ms | 0.692 | 100.1% |
| 16 | 52.515 ms | 1.309 | 99.4% | 49.505 ms | 1.388 | 100.3% |
| 32 | 27.289 ms | 2.518 | 95.6% | 24.777 ms | 2.774 | 100.2% |
| 40 | 21.993 ms | 3.125 | 94.9% | 19.930 ms | 3.448 | 99.7% |
| 64 | 13.711 ms | 5.012 | 95.2% | 12.539 ms | 5.480 | 99.0% |
| 80 | 12.088 ms | 5.685 | 86.4% | 10.045 ms | 6.841 | 98.9% |

At 80 threads SVE reduces dual-Q GEMM latency by 20.3% relative to NEON. Its
6.841 TFLOP/s is 92.9% of the 7.36 TFLOP/s reference obtained from 92 GFLOP/s
per core, so the new default remains valid for the true long-context path.

## Complete Post Stage

| Threads | NEON median | SVE median | SVE change |
| ---: | ---: | ---: | ---: |
| 1 | 1448.002 ms | 1408.463 ms | +2.81% |
| 2 | 757.050 ms | 734.644 ms | +3.05% |
| 4 | 410.925 ms | 400.158 ms | +2.69% |
| 8 | 206.709 ms | 201.661 ms | +2.50% |
| 16 | 105.625 ms | 102.584 ms | +2.96% |
| 32 | 57.748 ms | 55.285 ms | +4.46% |
| 40 | 48.185 ms | 46.461 ms | +3.71% |
| 64 | 34.887 ms | 33.887 ms | +2.95% |
| 80 | 32.095 ms | 30.034 ms | +6.86% |

The complete-call gain is smaller because this initial run used the FP32 ATen
fallback for sparse-indexer scoring. A subsequent build fix enabled the existing
SVE `8x2VL` kernel and reduced the 80-thread score+TopK stage from 18.019 to
6.262 ms. See `arm_codex_internal_indexer_sve_8x2vl_20260820.md` for the
corrected current profile.
