# DeepSeek V4 attention GEMM scheduling on Amazon 192-core Arm

## Scope

This change affects only the non-normed
`deepseek_v4_attn_gemm_fused_prepacked` execution path used by
`attn_gemm_parallel_execute`. The packed-weight layout and Python API are
unchanged. The default schedule is `mn`; `legacy` remains available as an
explicit fallback.

The three cumulative variants are:

1. `m8`: split M only at M8 panel boundaries.
2. `pool`: put the M8 panels of all four GEMMs in one OpenMP worker pool.
3. `mn`: additionally split packed B into tile-aligned N groups.
4. `mn + prepack-A`: cooperatively reorder A once and let all MN groups and
   all four GEMMs consume the same packed-A buffer.

Selection can be overridden by `FUSED_CPP_ATTN_GEMM_SCHEDULE`. The optional
`FUSED_CPP_ATTN_GEMM_N_GROUPS` override sets the total N-group budget.
`FUSED_CPP_ATTN_GEMM_PREPACK_A=on|off` forces shared A packing on or off.

## Machine and method

- Machine: Amazon 192-core Arm, Neoverse V3
- CPU set: NUMA0 cores 0-95
- Memory binding: NUMA0
- Backend: NEON BF16 GEMM
- Shape: M=2048, K=4096
- Output N: 1536 + 2048 + 256 + 64 = 3904
- Work per call: 65.498 GFLOP
- Weight packing excluded; output allocation remains included
- Warmup: 3; measured runs: 15; table reports median

Command:

```bash
PYTHONPATH=src FUSED_CPP_ATTN_GEMM_BACKEND=neon \
numactl --physcpubind=0-95 --membind=0 \
.venv/bin/python tests/bench_deepseek_v4_attn_gemm_fused.py \
  --cores 0-95 --warmup 3 --runs 15 --schedule <variant>
```

## Cumulative result

| Schedule | Median ms | Aggregate TFLOP/s | Speedup vs previous | Speedup vs legacy |
|---|---:|---:|---:|---:|
| `legacy` | 12.361 | 5.299 | - | 1.00x |
| `m8` | 8.674 | 7.551 | 1.425x | 1.425x |
| `pool` | 7.914 | 8.276 | 1.096x | 1.562x |
| `mn` auto | 3.329 | 19.676 | 2.377x | 3.713x |

The M8 partition removes repeated tail-kernel work created by splitting 2048
rows into 96 arbitrary row ranges. The shared pool then removes imbalance
between the four unequal N dimensions. N grouping supplies the missing
intra-GEMM parallelism: M-only splitting cannot keep 96 workers busy after
the panel count becomes small.

## N-group policy

For this shape, a 96-thread sweep found 48 total N groups best:

| N groups | 24 | 32 | 40 | 48 | 56 | 64 | 96 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| TFLOP/s | 19.959 | 20.592 | 18.438 | 20.668 | 17.380 | 16.659 | 19.182 |

The non-monotonic result comes from tile divisibility and group-size
imbalance, so the implementation keeps an explicit override. The current
automatic policy uses one group per thread through 48 threads and half as
many groups above 48 threads. This policy is experimental and is not yet a
portable planner model.

## Auto-policy scaling

| Threads | Median ms | GFLOP/s | Speedup | Parallel efficiency |
|---:|---:|---:|---:|---:|
| 1 | 244.314 | 268.090 | 1.00x | 100.0% |
| 2 | 124.225 | 527.253 | 1.97x | 98.3% |
| 4 | 62.741 | 1043.941 | 3.89x | 97.4% |
| 8 | 34.901 | 1876.707 | 7.00x | 87.5% |
| 16 | 17.853 | 3668.711 | 13.68x | 85.5% |
| 24 | 11.465 | 5713.085 | 21.31x | 88.8% |
| 32 | 8.446 | 7754.869 | 28.93x | 90.4% |
| 48 | 5.710 | 11469.823 | 42.79x | 89.1% |
| 64 | 4.494 | 14574.107 | 54.36x | 84.9% |
| 80 | 3.715 | 17630.540 | 65.76x | 82.2% |
| 96 | 3.170 | 20660.274 | 77.07x | 80.3% |

## Paired legacy versus default MN

The following sweep alternates the two schedules at every thread count. It
uses the same process affinity, shape, warmup, and 15 measured runs as the
cumulative comparison.

| Threads | Legacy ms | Legacy TFLOP/s | MN auto ms | MN auto TFLOP/s | MN speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 245.735 | 0.267 | 244.852 | 0.268 | 1.004x |
| 2 | 125.213 | 0.523 | 124.148 | 0.528 | 1.009x |
| 4 | 63.597 | 1.030 | 63.429 | 1.033 | 1.003x |
| 8 | 39.180 | 1.672 | 35.135 | 1.864 | 1.115x |
| 16 | 23.154 | 2.829 | 18.139 | 3.611 | 1.276x |
| 24 | 21.345 | 3.069 | 11.686 | 5.605 | 1.827x |
| 32 | 20.845 | 3.142 | 8.670 | 7.555 | 2.404x |
| 48 | 22.017 | 2.975 | 5.970 | 10.972 | 3.688x |
| 64 | 13.140 | 4.985 | 4.628 | 14.152 | 2.839x |
| 80 | 12.562 | 5.214 | 3.875 | 16.905 | 3.242x |
| 96 | 12.440 | 5.265 | 3.331 | 19.663 | 3.735x |

MN has effectively no cost through four threads, begins to win at eight
threads, and becomes necessary once arbitrary legacy row slices create
inefficient M tails and cannot maintain cache-local packed-B windows across
the four unequal GEMMs.

## Pack A once

The MN row-major kernels previously repacked one M8 A panel for every
`(GEMM, N-group, M8-panel)` task. The new Linux NEON path:

1. partitions the input M8 panels across the OpenMP team;
2. writes one reorder-M8 packed-A buffer, zero-padding the final M tail;
3. executes one barrier to publish the complete buffer;
4. runs attention-prefixed packed-A M8 kernels for BF16 or FP32 row-major
   output; and
5. narrows padded output rows back to the logical M.

SVE and non-MN schedules retain their existing paths. The default policy
enables shared prepack only for MN with at least 24 requested N groups.
Explicit `on` remains available for experiments.

Forced `off` versus forced `on`, with 3 warmups and 11 measured runs:

| Threads | Repack A ms | Pack once ms | Pack-on speedup |
|---:|---:|---:|---:|
| 1 | 244.124 | 243.265 | 1.004x |
| 2 | 123.990 | 124.769 | 0.994x |
| 4 | 62.886 | 64.417 | 0.976x |
| 8 | 34.808 | 36.511 | 0.953x |
| 16 | 17.855 | 18.384 | 0.971x |
| 24 | 11.426 | 11.008 | 1.038x |
| 32 | 8.392 | 8.160 | 1.028x |
| 48 | 5.789 | 5.529 | 1.047x |
| 64 | 4.499 | 4.410 | 1.020x |
| 80 | 3.726 | 3.667 | 1.016x |
| 96 | 3.162 | 3.071 | 1.030x |

At low group counts the packed-A-only kernel plus publish barrier costs more
than the integrated row-major load/pack sequence. The crossover is 24 groups
on this machine. A 21-run 96-thread check measured `3.154 -> 3.059 ms`
(`20.764 -> 21.411 TFLOP/s`, 1.031x).

The automatic gate was checked at its boundary:

| Threads | Forced off ms | Auto ms | Forced on ms |
|---:|---:|---:|---:|
| 16 | 17.888 | 17.959 | 18.383 |
| 24 | 11.424 | 11.090 | 11.158 |
| 96 | 3.166 | 3.062 | 3.070 |

The small `off`/`auto` or `auto`/`on` differences are run-to-run noise; the
results confirm that auto selects repack below the threshold and shared
prepack at and above it. Shared prepack is not enabled by default for
`pool`: at 96 threads it changed `7.821 -> 8.084 ms` (0.967x).

## Correctness

- Local build: successful
- Local test suite: 31 passed
- Remote NEON: 31 passed
- Remote SVE: 31 passed

The tests cover `legacy`, `m8`, `pool`, and `mn`, all three exposed output
variants, M=17 global tails, a non-power-of-two seven-group override, and
packed-A M tails 1/3/5/7/8/13 against the original repack path.

## Decision and remaining work

Use `mn` with the automatic N-group and shared-prepack policies by default,
and preserve `legacy` plus `FUSED_CPP_ATTN_GEMM_PREPACK_A=off` as explicit
fallbacks. The current policies still need broader performance validation
across M, K, N, thread count, and both SVE and NEON. The normed execution
path continues to use the legacy row split and needs separate integration
and validation.
