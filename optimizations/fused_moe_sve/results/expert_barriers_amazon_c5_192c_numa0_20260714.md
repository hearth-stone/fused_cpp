# SVE fused-expert barrier elision on Amazon 192C

## Environment

- Host: `AmazonC5192Cores`, 192 logical Neoverse-V3 cores
- Placement: NUMA node 0, CPUs `0-95`
- Shape: `H=4096`, `F=512`, split-W13 enabled
- Build: GCC 15.2, release, `-O2`, SVE + BF16 + i8mm
- Runtime: Python 3.12.13, PyTorch 2.13.0+cpu
- Timing: four variants interleaved in one process; 4 warmups, 17 samples;
  adaptive inner iterations target at least 10 ms per timed sample
- Gain: `baseline_median / variant_median - 1`

Variants:

- `baseline`: intermediate zero + barrier, then W2 + barrier + scatter
- `no_zero`: W13 fully overwrites the packed intermediate
- `owner_scatter`: each W2 worker scatters its own `n_tile`-aligned columns
- `combined`: both changes

## Unweighted single expert

Command:

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  env PYTHONPATH=src .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_expert_barriers.py \
  --threads 1,2,4,8,16,32,48,64,96 \
  --warmup 4 --runs 17 --sample-ms 10
```

Scheduled, route 12 scaling:

| Threads | Baseline ms | No-zero | Owner-scatter | Combined |
|---:|---:|---:|---:|---:|
| 1 | 0.5020 | 0.00% | -0.03% | -0.01% |
| 2 | 0.2763 | 0.15% | -0.37% | -0.05% |
| 4 | 0.1680 | 0.55% | 0.51% | 0.98% |
| 8 | 0.1163 | 0.57% | 0.69% | 1.15% |
| 16 | 0.1057 | 1.30% | 1.79% | 3.09% |
| 32 | 0.1333 | 2.35% | 1.99% | 3.89% |
| 48 | 0.1837 | 2.91% | 3.75% | 5.65% |
| 64 | 0.2270 | 2.55% | 2.39% | 6.65% |
| 96 | 0.3287 | 1.93% | 3.14% | 7.77% |

At 96 threads:

| Path | Routes | Baseline ms | No-zero | Owner-scatter | Combined |
|:---|---:|---:|---:|---:|---:|
| scheduled | 12 | 0.3287 | 1.93% | 3.14% | 7.77% |
| scheduled | 48 | 0.3826 | 2.59% | 2.86% | 6.43% |
| scheduled | 192 | 0.6315 | 2.04% | 2.97% | 1.94% |
| scheduled | 768 | 1.7487 | 2.89% | 1.40% | 3.21% |
| scheduled | 2040 | 4.2930 | 1.96% | -0.11% | 1.74% |
| async | 12 | 0.3243 | 2.36% | 5.03% | 9.04% |
| async | 48 | 0.3801 | 2.77% | 3.91% | 8.27% |
| async | 192 | 0.6217 | 0.87% | 1.24% | 3.85% |
| async | 768 | 1.7572 | 3.30% | 2.25% | 4.34% |
| async | 2040 | 4.2642 | 0.87% | 0.04% | 0.94% |

Combined-variant latency distribution at 96 threads:

| Path | Routes | Mean ms | Stddev ms | P50 ms | P90 ms | P99 ms |
|:---|---:|---:|---:|---:|---:|---:|
| scheduled | 12 | 0.3037 | 0.0052 | 0.3050 | 0.3079 | 0.3137 |
| scheduled | 48 | 0.3601 | 0.0090 | 0.3595 | 0.3661 | 0.3854 |
| scheduled | 192 | 0.6119 | 0.0183 | 0.6195 | 0.6275 | 0.6447 |
| scheduled | 768 | 1.7040 | 0.0371 | 1.6943 | 1.7306 | 1.8071 |
| scheduled | 2040 | 4.3449 | 0.2747 | 4.2197 | 4.8795 | 4.9496 |
| async | 12 | 0.2982 | 0.0060 | 0.2974 | 0.3052 | 0.3075 |
| async | 48 | 0.3505 | 0.0066 | 0.3511 | 0.3563 | 0.3601 |
| async | 192 | 0.5989 | 0.0117 | 0.5987 | 0.6087 | 0.6282 |
| async | 768 | 1.6856 | 0.0198 | 1.6842 | 1.7048 | 1.7277 |
| async | 2040 | 4.3355 | 0.2792 | 4.2244 | 4.8987 | 4.9286 |

Raw samples remain in the local benchmark workspace. Rerun the command with
`--output <path>` to regenerate the structured JSON.

## Weighted route path

This adds the BF16 route buffer and final weighted merge. It remains a
single-expert, `top_k=1` benchmark; it measures writeback dilution, not a full
multi-expert `top_k=6` schedule.

At 96 threads:

| Path | Routes | Baseline ms | Combined gain |
|:---|---:|---:|---:|
| scheduled | 12 | 0.4791 | 7.06% |
| scheduled | 48 | 0.5329 | 5.14% |
| scheduled | 192 | 0.8112 | 5.30% |
| scheduled | 768 | 1.9691 | 2.46% |
| scheduled | 2040 | 4.4818 | 2.45% |
| async | 12 | 0.4749 | 5.67% |
| async | 48 | 0.5346 | 4.39% |
| async | 192 | 0.8155 | 4.36% |
| async | 768 | 1.9656 | 2.29% |
| async | 2040 | 4.4694 | 1.31% |

Raw samples remain in the local benchmark workspace. Rerun the weighted command
with `--output <path>` to regenerate the structured JSON.

## Correctness

The dirty-scratch regression reuses one team scratch buffer across descending
route counts 24 through 1. It covers scheduled/async, FP32/BF16 W2 outputs, each
single feature, the combined variant, and the default-on configuration. Results:

- SVE default: `27 passed, 1 skipped`
- Forced NEON fallback (`FUSED_CPP_MOE_SVE=0`): `27 passed, 1 skipped`
- Hierarchical N-split, packA fusion, and team-GEMM suites: `894 passed`

The SVE default enables both features. Set either of these to `0` to restore its
legacy barrier independently:

- `FUSED_CPP_MOE_SVE_ELIDE_INTERMEDIATE_ZERO=0`
- `FUSED_CPP_MOE_SVE_W2_N_OWNER_SCATTER=0`
