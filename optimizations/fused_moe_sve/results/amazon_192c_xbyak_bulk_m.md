# Amazon C5 192-core Xbyak bulk-M result

Date: 2026-07-21

## Variant

`FUSED_CPP_MOE_SVE_JIT_BULK_M=1` makes one generated M12 kernel consume every
complete 12-row panel in `gemm_params_t::m`. The panel arithmetic and N traversal
are unchanged. Packed A advances by `12*K` BF16 values per panel. W13 advances
packed C by `12*ldc` BF16 values, regular W2 advances C by `12*ldc` FP32 values,
and direct-route W2 keeps the output base fixed while advancing 12 route IDs.
The existing exact-M kernel handles the final 1-11 rows.

The feature is explicit and remains off by default.

## Correctness

Host: `AmazonC5192Cores`, NUMA0 CPUs `0-7`. The test covered SiLU polynomial
degrees 4/5/6, normal/scheduled/async bridges, regular and direct-route FP32 W2,
M=1-13/23-25/35-37/48/192, and the static assembly fallback.

```bash
PYTHONPATH=src FUSED_CPP_MOE_SVE=1 OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
  numactl --cpunodebind=0 --membind=0 taskset -c 0-7 \
  .venv/bin/python -m pytest -q tests/test_fused_moe_bf16_tiled.py \
  -k 'sve_xbyak_exact_m_matches_static_asm or \
      sve_w2_direct_route_store_matches_scatter or \
      sve_xbyak_strict_mode_rejects_kc_fallback'
```

Result: `16 passed, 78 deselected`. Bulk-M and panel JIT outputs were bitwise
equal to static assembly for the tested FP32 route-output paths.

## Benchmark

- Host: `AmazonC5192Cores`, Neoverse V3, NUMA0 CPUs `0-95`.
- Shape: BF16 H=4096, F=512, 64 independent experts, 8 measured experts.
- Kernel policy: SVE JIT, split-W13 enabled, one K chunk, FP32 direct-route W2.
- Samples: 5 warmups, 31 timed calls, 5 calls per implementation block.
- Each implementation sees the same alternating pair of disjoint expert-weight
  windows in every block. Reversing implementation order between blocks avoids
  charging one implementation consistently for transition state.

```bash
PYTHONPATH=src OMP_NUM_THREADS=96 OMP_DYNAMIC=FALSE \
  numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python optimizations/fused_moe_sve/benchmarks/bench_xbyak_exact_m.py \
  --variants jit-panel,jit-bulk --routes 24,48,192,768,2040 \
  --threads 1,4,8,16,32,64,96 --warmup 5 --runs 31 \
  --switch-period 5 --output /tmp/xbyak_bulk_m_numa0_paired.json
```

Median end-to-end scheduled latency at the endpoints:

| M | 1T panel | 1T bulk | Gain | 96T panel | 96T bulk | Gain |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 8.704 ms | 8.702 ms | +0.03% | 1.012 ms | 1.024 ms | -1.15% |
| 48 | 16.678 ms | 16.498 ms | +1.09% | 1.288 ms | 1.294 ms | -0.46% |
| 192 | 64.042 ms | 64.031 ms | +0.02% | 3.088 ms | 3.086 ms | +0.07% |
| 768 | 255.440 ms | 255.468 ms | -0.01% | 11.847 ms | 11.829 ms | +0.15% |
| 2040 | 674.546 ms | 674.643 ms | -0.01% | 31.875 ms | 31.878 ms | -0.01% |

Full gain grid, where positive means bulk-M is faster:

| M | 1T | 4T | 8T | 16T | 32T | 64T | 96T |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | +0.03% | -0.21% | -0.23% | +0.14% | -0.06% | +0.07% | -1.15% |
| 48 | +1.09% | -0.56% | -0.51% | -0.64% | +0.01% | +0.39% | -0.46% |
| 192 | +0.02% | -0.57% | -0.06% | +0.13% | -0.05% | -0.28% | +0.07% |
| 768 | -0.01% | +0.01% | -0.01% | -0.02% | -0.10% | +0.04% | +0.15% |
| 2040 | -0.01% | -0.02% | -0.01% | +0.29% | -0.08% | +0.35% | -0.01% |

Across all 35 points, the median gain is `-0.011%` and the geometric-mean gain
is `-0.066%`. Fourteen points improve and 21 regress. This is performance-neutral
at the end-to-end expert level: panel-call prologue and invariant setup are too
small relative to the BFMMLA and epilogue work to justify changing the default.

## Measurement correction

The first run offset the two weight-window call indices by variant. Because the
two windows form a stable latency doublet, 31 samples assigned 16 fast windows to
one variant and 15 to the other. That produced a false `+4.54%` at M=48, 1T even
though the p10 and p90 values were nearly identical. The benchmark now feeds
every variant the same window sequence in each block. The corrected M=48, 1T
result is `+1.09%` in the full grid and `-0.16%` in a separate 31-sample rerun,
so neither is evidence of a stable speedup.
