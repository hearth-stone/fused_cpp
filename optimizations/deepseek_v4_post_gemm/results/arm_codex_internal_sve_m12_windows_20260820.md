# Post-GEMM SVE M12 Windows on Arm-codex-internal

Date: 2026-08-20

## Erratum

The default benchmark uses `context_start=0`, `M=2048`, compression ratio 4,
and `topk_tokens=512`. It therefore takes the sparse-indexer select-all short
path and skips Indexer Q. Sections below that describe two Q GEMMs or attribute
the default-shape result to shared-QR packing are incorrect. The timing
comparison remains a valid Main-Q NEON-versus-SVE comparison, but the reported
GEMM-equivalent throughput must be divided by two. Corrected single-Q scaling is
recorded in `arm_codex_internal_single_q_l2_windows_20260820.md`.

## Configuration

- Host: `Arm-codex-internal`
- Affinity: NUMA0 cores `0-7`
- Shape: benchmark default V4 post-GEMM shape, `M=2048`
- Threads: 8
- Schedule: aligned M panels, shared Q pool, 8 contiguous N groups
- Warmup/runs: 3/15

## Result

| Backend | Median | Minimum | GEMM-equivalent throughput |
| --- | ---: | ---: | ---: |
| NEON M8 | 59.264 ms | 59.152 ms | 0.580 TFLOP/s |
| SVE JIT M12/exact-M | 57.376 ms | 57.054 ms | 0.599 TFLOP/s |

The SVE M12 path reduces median latency by 3.19% on this configuration.

## Correctness

On the same host, all 25 tests in
`tests/test_deepseek_v4_post_gemm_stage.py` passed. The focused SVE test covers
M values `11,12,13,24,25` and validates dense Q output plus SWA cache writes
against the Torch baseline.

## Decision

This initial decision was superseded after both backends adopted the same
single-Q L2-window/M-split schedule. SVE is now the default when its BF16 JIT is
available; explicit `FUSED_CPP_POST_GEMM_BACKEND=neon` retains the fallback.

## Shared QR Pack And Default Decision

The SVE pair path now cooperatively packs QR once, then schedules Main Q and
Indexer Q as `(GEMM, contiguous N window, exact M12 panel)` tasks. A dedicated
post-GEMM prepare binding keeps explicit SVE/NEON packed weights consistent
without coupling the attention and post-GEMM backend environment variables.

The following Arm-codex-internal NUMA0 sweep compares the historical NEON
baseline (conservative shared-pool policy, two default groups) with the SVE M12
path using the requested thread-based N-group policy.
Shape is two `(2048,1024) x (1024,8192)` GEMMs; each point uses 3 warmups and
11 measured runs.

| Threads | NEON default | SVE M12/shared QR | Candidate speedup |
| ---: | ---: | ---: | ---: |
| 1 | 457.145 ms | 431.477 ms | +5.95% |
| 2 | 232.329 ms | 221.035 ms | +5.11% |
| 4 | 117.883 ms | 111.291 ms | +5.92% |
| 8 | 59.215 ms | 57.660 ms | +2.70% |
| 16 | 30.314 ms | 29.614 ms | +2.37% |
| 32 | 15.627 ms | 16.501 ms | -5.29% |
| 40 | 13.725 ms | 13.867 ms | -1.03% |
| 64 | 8.099 ms | 8.909 ms | -9.10% |
| 80 | 8.059 ms | 8.761 ms | -8.01% |

At 64T, changing the SVE candidate from 2 through 40 N groups leaves latency
in the narrow 8.80-8.96 ms range. At 80T, the best tested group count is 24 at
8.667 ms, still 7.5% slower than the NEON default. The high-thread regression
therefore comes from the K=1024 SVE M12 compute path rather than the N-window
count. This conclusion predates the shared single-Q L2-window scheduler and is
superseded for default dispatch by the corrected comparison linked above.

Holding SVE M12 and the N-group count fixed isolates shared QR packing:

| Threads/groups | Per-task QR pack | Shared QR pack | Change |
| --- | ---: | ---: | ---: |
| 8T / 8 groups | 57.427 ms | 57.541 ms | -0.20% |
| 80T / 24 groups | 8.837 ms | 8.725 ms | +1.28% |

Shared QR packing appeared performance-neutral in this run, but the erratum
above means this default shape did not execute the two-Q shared path. The
measurement cannot isolate shared QR packing and must not be used for that
claim.

## High-Thread Stall Attribution

`perf stat` on the same 80T/24-group workload, with 5 warmups and 31 measured
runs, rules out frequency and cache-miss pressure as the primary regression:

| Counter | NEON M8 | SVE M12 | SVE/NEON |
| --- | ---: | ---: | ---: |
| Frequency | 2.900 GHz | 2.900 GHz | 1.00x |
| Cycles | 122.25 B | 128.80 B | 1.05x |
| Instructions | 369.82 B | 387.04 B | 1.05x |
| Backend-stall cycles | 15.35 B | 35.22 B | 2.30x |
| L1D refills | 329.47 M | 196.00 M | 0.59x |
| L2D refills | 2.60 B | 1.79 B | 0.69x |
| LLC reads | 2.51 B | 1.70 B | 0.68x |
| LLC read misses | 14.69 M | 14.15 M | 0.96x |

SVE executes about 5% more instructions and spends more than twice as many
cycles stalled in the execution backend despite fewer cache refills and equal
frequency. The remaining gap is therefore in the K=1024 SVE JIT compute/store
pipeline (resource or dependency stalls and less fixed-cost amortization), not
in N-window sizing, QR packing, LLC/DRAM misses, or frequency throttling.
