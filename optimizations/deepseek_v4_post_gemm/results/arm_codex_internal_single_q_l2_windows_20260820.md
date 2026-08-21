# Single-Q L2 Windows on Arm-codex-internal

Date: 2026-08-20

## Configuration

- Host: `Arm-codex-internal`
- Affinity: NUMA0 cores `0-(T-1)`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`
- Shape: Main Q `(2048,1024) x (1024,8192)`, BF16 output
- Sparse-indexer state: `context_start=0`, so select-all skips Indexer Q
- FLOPs: `2 * 2048 * 1024 * 8192 = 34.360 GFLOP`
- Candidate: shared packed A, approximately 1 MiB packed-B windows, M split inside each window
- Baseline: `FUSED_CPP_POST_GEMM_M8_ALIGNED=0` historical pure M split
- Warmup/runs/statistic: 3/9/median

## Results

| Backend | Threads | Pure M split | L2 windows | Speedup | Candidate TFLOP/s | 1T linear efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NEON | 1 | 459.680 ms | 453.375 ms | +1.39% | 0.076 | 100.0% |
| NEON | 2 | 232.375 ms | 231.418 ms | +0.41% | 0.148 | 98.0% |
| NEON | 4 | 117.277 ms | 116.753 ms | +0.45% | 0.294 | 97.1% |
| NEON | 8 | 59.388 ms | 58.548 ms | +1.44% | 0.587 | 96.8% |
| NEON | 16 | 30.231 ms | 29.654 ms | +1.95% | 1.159 | 95.6% |
| NEON | 32 | 15.679 ms | 15.242 ms | +2.87% | 2.254 | 93.0% |
| NEON | 40 | 13.401 ms | 12.452 ms | +7.62% | 2.759 | 91.0% |
| NEON | 64 | 8.169 ms | 8.089 ms | +1.00% | 4.248 | 87.6% |
| NEON | 80 | 7.894 ms | 6.832 ms | +15.54% | 5.029 | 82.9% |
| SVE | 1 | 433.022 ms | 430.319 ms | +0.63% | 0.080 | 100.0% |
| SVE | 2 | 220.666 ms | 219.143 ms | +0.70% | 0.157 | 98.2% |
| SVE | 4 | 113.144 ms | 110.505 ms | +2.39% | 0.311 | 97.4% |
| SVE | 8 | 56.826 ms | 55.707 ms | +2.01% | 0.617 | 96.6% |
| SVE | 16 | 28.982 ms | 28.092 ms | +3.17% | 1.223 | 95.7% |
| SVE | 32 | 15.743 ms | 14.941 ms | +5.37% | 2.300 | 90.0% |
| SVE | 40 | 13.029 ms | 12.187 ms | +6.91% | 2.819 | 88.3% |
| SVE | 64 | 8.176 ms | 8.056 ms | +1.49% | 4.265 | 83.5% |
| SVE | 80 | 7.806 ms | 6.887 ms | +13.35% | 4.989 | 78.1% |

## Conclusion

The windowed schedule improves every measured point. Its effect is small while
the pure M split remains below the shared-cache service limit, then becomes
material at 40 and 80 threads. At 80 threads, the useful throughput is 5.029
TFLOP/s for NEON and 4.989 TFLOP/s for SVE; these values use the one GEMM that
actually executes, rather than the previous erroneous two-GEMM FLOP count.

NEON is 0.8% faster than SVE at 80 threads despite SVE leading through 64
threads. The remaining high-width difference is therefore still in backend
kernel scaling rather than the N/M task geometry, which is now shared.

An additional 80-thread profiled invocation isolates the GEMM stage:

| Backend | Schedule | Main Q GEMM | GEMM-only speedup |
| --- | --- | ---: | ---: |
| NEON | Pure M split | 6.766 ms | baseline |
| NEON | L2 windows | 5.369 ms | +26.02% |
| SVE | Pure M split | 6.734 ms | baseline |
| SVE | L2 windows | 5.424 ms | +24.15% |

The complete call improves less because compressor updates and Main-Q
postprocessing are unchanged. For this shape the automatic geometry uses 16 N
windows at 32, 64, and 80 threads (1 MiB B per window), with respectively 2,
4, and 5 M splits per window. At 40 threads it uses 20 N windows (approximately
0.8 MiB each) and 2 M splits so all workers receive one equal-sized task.

After adopting SVE as the automatic post-GEMM backend, an unset-backend
80-thread validation measured 6.645 ms median over 9 runs after 3 warmups. The
focused remote suite passed all 26 tests, including the assertion that automatic
weight preparation matches explicit SVE packing and remains distinct from
explicit NEON packing.
