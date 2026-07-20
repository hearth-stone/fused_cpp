# Xbyak exact-M SVE fused-expert validation

Date: 2026-07-20

## Scope

This experiment compares the default Xbyak-generated SVE fused-expert kernels
with the static assembly fallback. Both variants use identical SVE packed
weights, split-W13, M12 packed-A panels, polynomial-5 SiLU, FP32 W2 scratch
output followed by the top-k=1 BF16 scatter, and scheduled N-split execution.
The generated body matches the static
K loop's K8 granularity and accumulator mapping, but not its double-buffered
load/compute instruction schedule. It specializes every final logical M=1..12
panel.

The benchmark shape is H=4096, F=512, 64 allocated experts, and eight measured
experts per call. It rotates between two disjoint eight-expert weight windows.
Reported times are medians for the complete eight-expert call, not per-expert
latencies or isolated W13 times. Each implementation switch is followed by one
unmeasured call, then five steady timed calls. The full M=1..12 sweep uses five
initial warmups, 31 timed samples at 1T, and 21 timed samples at 2T/4T/8T.

## Correctness

On `AmazonC5192Cores`, CPUs 0-15:

```bash
PYTHONPATH=src FUSED_CPP_MOE_SVE=1 OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE \
  taskset -c 0-15 .venv/bin/python -m pytest -q \
  tests/test_fused_moe_bf16_tiled.py \
  -k 'sve_xbyak_exact_m_matches_static_asm or \
      sve_w2_direct_route_store_matches_scatter or \
      vllm_staged_matches_fused_sve_with_multiple_n_tasks'
```

Result: 19 passed. On `AmazonECS8Cores`, CPUs 0-7, the exact-M, direct-route,
static-reference, and strict-fallback subset passed 17 tests. For SiLU degrees
4/5/6, M=1..12, and normal, scheduled, and async bridges, JIT and
static-assembly outputs were bit exact. Strict JIT also rejects split-K packing
instead of silently using a static-assembly kernel; `auto` retains that
compatibility fallback.

## Amazon 192-core host, NUMA0

Command:

```bash
PYTHONPATH=src OMP_NUM_THREADS=96 OMP_DYNAMIC=FALSE \
  numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_xbyak_exact_m.py \
  --routes 1,2,3,4,5,6,7,8,9,10,11,12 --threads 1 \
  --warmup 5 --runs 31 --switch-period 5
```

| M | Static asm, 1T | Xbyak JIT, 1T | JIT gain |
| ---: | ---: | ---: | ---: |
| 1 | 2.620 ms | 2.631 ms | -0.43% |
| 2 | 2.604 ms | 2.607 ms | -0.12% |
| 3 | 2.704 ms | 2.732 ms | -1.02% |
| 4 | 2.699 ms | 2.757 ms | -2.09% |
| 5 | 3.480 ms | 3.081 ms | +12.93% |
| 6 | 3.483 ms | 3.098 ms | +12.43% |
| 7 | 3.486 ms | 3.585 ms | -2.75% |
| 8 | 3.529 ms | 3.597 ms | -1.91% |
| 9 | 4.296 ms | 3.893 ms | +10.35% |
| 10 | 4.303 ms | 3.899 ms | +10.34% |
| 11 | 4.314 ms | 4.332 ms | -0.42% |
| 12 | 4.324 ms | 4.347 ms | -0.51% |

The corresponding steady-state gains over the static assembly path are:

| M | 1T gain | 2T gain | 4T gain | 8T gain |
| ---: | ---: | ---: | ---: | ---: |
| 1 | -0.43% | -0.71% | -0.68% | +0.15% |
| 2 | -0.12% | -0.70% | +0.25% | -0.19% |
| 3 | -1.02% | -0.94% | -0.02% | +0.05% |
| 4 | -2.09% | -0.84% | -0.16% | -0.30% |
| 5 | +12.93% | +5.80% | +5.74% | -0.59% |
| 6 | +12.43% | +5.51% | +5.16% | -0.56% |
| 7 | -2.75% | -4.26% | -3.49% | -0.72% |
| 8 | -1.91% | -4.17% | -2.62% | -0.27% |
| 9 | +10.35% | +9.41% | +8.58% | +2.71% |
| 10 | +10.34% | +9.21% | +8.73% | +2.46% |
| 11 | -0.42% | -0.41% | -0.75% | -0.54% |
| 12 | -0.51% | -0.67% | -0.57% | -1.14% |

The V3 M=7/8 regressions at 2T and 4T are repeatable rather than a single
outlier: for example M=7/2T has non-overlapping asm and JIT p10-p90 intervals
of 1.898-1.956 ms and 1.955-2.017 ms. No cause is assigned from this full-call
measurement alone; it requires a JIT-aware isolated W13/W2 microbenchmark.

Steady long-route control:

| M | Threads | Static asm | Xbyak JIT | JIT gain |
| ---: | ---: | ---: | ---: | ---: |
| 192 | 1 | 63.855 ms | 64.229 ms | -0.58% |
| 192 | 4 | 17.332 ms | 17.317 ms | +0.09% |
| 192 | 8 | 8.751 ms | 8.791 ms | -0.45% |
| 192 | 16 | 4.682 ms | 4.722 ms | -0.86% |
| 192 | 32 | 3.381 ms | 3.393 ms | -0.35% |
| 192 | 64 | 3.216 ms | 3.191 ms | +0.79% |
| 192 | 96 | 3.211 ms | 3.198 ms | +0.39% |
| 2040 | 1 | 673.810 ms | 677.553 ms | -0.55% |
| 2040 | 4 | 175.478 ms | 176.168 ms | -0.39% |
| 2040 | 8 | 89.746 ms | 90.085 ms | -0.38% |
| 2040 | 16 | 48.545 ms | 48.557 ms | -0.03% |
| 2040 | 32 | 33.938 ms | 33.938 ms | +0.00% |
| 2040 | 64 | 32.153 ms | 32.233 ms | -0.25% |
| 2040 | 96 | 31.637 ms | 31.573 ms | +0.21% |

All M192/M2040 controls are within 0.86%. The earlier call-by-call A/B method
showed an isolated M192/8T regression that disappeared with steady blocks; it
was implementation-switch I-cache interference, not production steady-state
throughput.

## Amazon 8-core host

Command:

```bash
PYTHONPATH=src OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE taskset -c 0-7 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_xbyak_exact_m.py \
  --routes 1,2,3,4,5,6,7,8,9,10,11,12 --threads 1,2,4,8 \
  --warmup 5 --runs 21 --switch-period 5
```

| M | 1T gain | 2T gain | 4T gain | 8T gain |
| ---: | ---: | ---: | ---: | ---: |
| 1 | +0.22% | -0.37% | -0.82% | -0.79% |
| 2 | -0.41% | -1.05% | -0.40% | -1.36% |
| 3 | +0.14% | +0.22% | -0.08% | -0.21% |
| 4 | +0.21% | +0.75% | -0.63% | -0.07% |
| 5 | +6.17% | +11.86% | +11.97% | +6.89% |
| 6 | +5.62% | +11.62% | +12.38% | +4.92% |
| 7 | -1.75% | -1.48% | -0.79% | -0.44% |
| 8 | -1.19% | -0.97% | -0.27% | -0.22% |
| 9 | +10.93% | +10.70% | +8.17% | +7.01% |
| 10 | +12.18% | +10.55% | +8.48% | +7.25% |
| 11 | -0.16% | -0.43% | -1.03% | +0.12% |
| 12 | -0.26% | -0.42% | -1.13% | -0.36% |

At 1T, the raw 8-core-host medians are 3.428/3.420 ms for M=1,
4.225/3.980 ms for M=5, 4.245/4.019 ms for M=6, 5.271/4.752 ms
for M=9, 5.286/4.712 ms for M=10, and 5.317/5.331 ms for M=12
(asm/JIT respectively).

## Cost-model recalibration

Four schema-v2 profiles were regenerated with the JIT implementation pinned,
every route M=1..12 measured directly, and split/no-split kept as separate
calibration domains:

- 8-core standalone F512/E8: 80 isolated and 68 contention points per policy;
- 192-core dual-NUMA TP4 F512/E256: 180 isolated and 238 contention points per
  policy, with each sample reduced as the pairwise maximum of the two ranks.

The table reports the best measured full-call wall time and lane shape. The
8-core call contains eight experts; the dual-NUMA call contains 256 experts per
rank.

| Host | M | No-split best | Split best | Split throughput gain |
| --- | ---: | ---: | ---: | ---: |
| 8-core | 1 | 0.812 ms, 8x1T | 0.821 ms, 4x2T | -1.03% |
| 8-core | 12 | 1.133 ms, 8x1T | 1.204 ms, 4x2T | -5.93% |
| 8-core | 192 | 14.753 ms, 1x8T | 14.005 ms, 1x8T | +5.34% |
| 8-core | 2040 | 150.479 ms, 1x8T | 144.343 ms, 1x8T | +4.25% |
| 192-core | 1 | 8.660 ms, 24x4T | 8.689 ms, 24x4T | -0.33% |
| 192-core | 12 | 8.989 ms, 24x4T | 9.019 ms, 48x2T | -0.33% |
| 192-core | 192 | 34.661 ms, 6x16T | 35.744 ms, 6x16T | -3.03% |
| 192-core | 2040 | 323.427 ms, 12x8T | 332.454 ms, 24x4T | -2.72% |

These results do not establish one globally dominant W13 policy. The planner
therefore loads both profiles from the same implementation pair and chooses
between them for the requested route distribution.

## Conclusion

The exact-M implementation removes meaningful static tail overcompute for
M=5/6 and M=9/10 on both machines. These are the row counts where the JIT uses
one fewer physical BF16 row pair than the old M8/M12 bucket. M=1..4, M=7/8,
and M=11/12 execute the same number of BFMMLA row pairs as the old path, so
they are throughput controls rather than expected wins. Full M12 panels retain
static-assembly throughput within about 1%; static assembly remains the
compatibility and unsupported-epilogue fallback.
