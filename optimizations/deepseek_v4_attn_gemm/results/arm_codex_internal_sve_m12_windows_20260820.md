# SVE M12 Windows on Arm-codex-internal

Date: 2026-08-20

## Configuration

- Host: `Arm-codex-internal`
- Affinity: NUMA0 cores `0-7`
- Shape: `M=2048`, `K=4096`, output widths `1536,2048,256,64`
- Threads: 8
- Schedule: `mn`, 8 contiguous N groups
- Warmup/runs: 3/15
- Packed-weight preparation is excluded from latency.

## Result

| Backend | Median | Best | Median throughput |
| --- | ---: | ---: | ---: |
| NEON M8 | 101.415 ms | 101.174 ms | 645.841 GFLOP/s |
| SVE JIT M12/exact-M | 101.235 ms | 100.348 ms | 646.995 GFLOP/s |

SVE is 0.18% faster by median, which is within run-to-run noise. The first SVE
measurement was 103.510 ms because every M12 panel repeated the JIT registry
lookup. Caching each exact-M function pointer within one GEMM reduced the median
to 101.235 ms.

## Correctness

On the same host, all 39 tests in
`tests/test_deepseek_v4_attn_gemm_fused.py` passed. The focused SVE test covers
M values `1,8,11,12,13,23,24,25`, BF16 and FP32 stores, four GEMMs, and seven
contiguous N groups.

## Decision

The SVE path initially remained opt-in. After the shared packed-A result below,
it became the default on supported SVE BF16/Xbyak builds by explicit user
decision. NEON remains the automatic fallback and can be forced explicitly.

## M8-to-M12 Scaling

The historical `HEAD` implementation was exported into a separate build and
compared with the candidate on NUMA0 cores `0-79`. The historical source needed
one benchmark-only capability fix: accept
`__ARM_FEATURE_BF16_VECTOR_ARITHMETIC` as equivalent to `__ARM_FEATURE_BF16`.
The M8 kernel, packing, scheduler, and outputs were otherwise unchanged.

Both builds used `M=2048`, the four default V4 GEMM shapes, SVE, `mn`, automatic
thread-based N-group allocation, no shared prepacked A, three warmups, and 11
measured runs. The 32T and 80T endpoints were repeated with 5 warmups and 31
runs.

| Threads | M8 median | M8 GFLOP/s | M12 median | M12 GFLOP/s | M12 speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 862.786 ms | 75.9 | 777.986 ms | 84.2 | +10.90% |
| 2 | 417.296 ms | 157.0 | 391.384 ms | 167.3 | +6.62% |
| 4 | 207.555 ms | 315.6 | 197.518 ms | 331.6 | +5.08% |
| 8 | 105.079 ms | 623.3 | 99.824 ms | 656.1 | +5.26% |
| 16 | 54.316 ms | 1205.9 | 52.872 ms | 1238.8 | +2.73% |
| 32 | 29.086 ms | 2251.8 | 29.076 ms | 2252.7 | +0.03% |
| 40 | 24.086 ms | 2719.4 | 24.036 ms | 2725.0 | +0.21% |
| 64 | 15.099 ms | 4337.8 | 15.063 ms | 4348.2 | +0.24% |
| 80 | 12.473 ms | 5251.0 | 12.564 ms | 5213.0 | -0.72% |

M12 improves local kernel efficiency through 16 threads, but both versions
converge to about 5.2 TFLOP/s at 80 threads. At this stage the cause was not
identified. The shared packed-A experiment below supersedes the interpretation
that 5.2 TFLOP/s was a hard platform limit: removing repeated A packing raises
the same 80-thread workload to 6.76 TFLOP/s.

## Shared SVE Packed-A

The SVE MN path now cooperatively packs the complete A matrix once into fixed
M12-stride panels before publishing the task pool. The baseline below uses the
same M12 exact-M JIT but repacks A inside every `(GEMM, N-window, M-panel)`
task. Both variants use three warmups and 11 measured runs; the 32T-and-higher
candidate points were repeated with 5 warmups and 31 runs.

| Threads | Per-task pack median | Shared pack median | Shared throughput | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 783.655 ms | 748.044 ms | 87.6 GFLOP/s | +4.76% |
| 2 | 386.897 ms | 371.667 ms | 176.2 GFLOP/s | +4.10% |
| 4 | 193.832 ms | 186.467 ms | 351.3 GFLOP/s | +3.95% |
| 8 | 99.647 ms | 93.151 ms | 703.1 GFLOP/s | +6.97% |
| 16 | 53.367 ms | 46.991 ms | 1393.9 GFLOP/s | +13.57% |
| 32 | 29.068 ms | 23.934 ms | 2736.6 GFLOP/s | +21.45% |
| 40 | 24.202 ms | 19.344 ms | 3386.0 GFLOP/s | +25.11% |
| 64 | 14.999 ms | 12.442 ms | 5264.2 GFLOP/s | +20.55% |
| 80 | 12.529 ms | 9.684 ms | 6763.4 GFLOP/s | +29.38% |

At 80 threads, the shared-pack path reaches 91.2% of the 7.418 TFLOP/s
80-core SVE BFMMLA instruction reference. Boundary checks also improved:
`M=12,8T` by 4.3%, `M=96,8T` by 7.0%, and `M=384,32T` by 23.6%.
