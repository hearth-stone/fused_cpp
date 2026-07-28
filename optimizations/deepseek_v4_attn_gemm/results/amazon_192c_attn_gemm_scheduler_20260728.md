# DeepSeek V4 attention GEMM scheduling on Amazon 192-core Arm

## Scope

This change affects only the non-normed
`deepseek_v4_attn_gemm_fused_prepacked` execution path used by
`attn_gemm_parallel_execute`. The GEMM microkernels, packed-weight layout,
and Python API are unchanged. The default schedule is now `mn`; `legacy`
remains available as an explicit fallback.

The three cumulative variants are:

1. `m8`: split M only at M8 panel boundaries.
2. `pool`: put the M8 panels of all four GEMMs in one OpenMP worker pool.
3. `mn`: additionally split packed B into tile-aligned N groups.

Selection can be overridden by `FUSED_CPP_ATTN_GEMM_SCHEDULE`. The optional
`FUSED_CPP_ATTN_GEMM_N_GROUPS` override sets the total N-group budget.

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

## Correctness

- Local build: successful
- Local test suite: 25 passed
- Remote NEON: 25 passed
- Remote SVE: 25 passed

The tests cover `legacy`, `m8`, `pool`, and `mn`, all three exposed output
variants, M=17 global tails, and a non-power-of-two seven-group override.

## Decision and remaining work

Use `mn` with the automatic N-group policy by default and preserve `legacy`
as an explicit fallback. The current policy still needs broader performance
validation across M, K, N, thread count, and both SVE and NEON. The normed
execution path continues to use the legacy row split and needs separate
integration and validation.
