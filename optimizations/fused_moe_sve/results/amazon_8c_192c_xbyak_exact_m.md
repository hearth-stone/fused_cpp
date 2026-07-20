# Xbyak exact-M SVE fused-expert validation

Date: 2026-07-20

## Scope

This experiment compares the default Xbyak-generated SVE fused-expert kernels
with the static assembly fallback. Both variants use identical SVE packed
weights, split-W13, M12 packed-A panels, polynomial-5 SiLU, FP32 W2 scratch
output followed by the top-k=1 BF16 scatter, and scheduled N-split execution.
For M<=8 the generated body matches the static M2/M4/M8 double-buffer K-loop
state machine and accumulator mapping. M=9..12 retains the static M12
register-reuse schedule. Every final logical M=1..12 panel remains specialized.

The benchmark shape is H=4096, F=512, 64 allocated experts, and eight measured
experts per call. It rotates between two disjoint eight-expert weight windows.
Reported times are medians for the complete eight-expert call, not per-expert
latencies or isolated W13 times. Each implementation switch is followed by one
unmeasured call, then four steady timed calls. The full M=1..12 sweep uses eight
initial warmups and 40 timed samples per point. The even switch period and
sample count give both implementations the same number of calls to each weight
window.

## Correctness

On `AmazonC5192Cores`, CPUs 0-15:

```bash
PYTHONPATH=src FUSED_CPP_MOE_SVE=1 OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE \
  taskset -c 0-15 .venv/bin/python -m pytest -q \
  tests/test_fused_moe_bf16_tiled.py \
  -k 'sve_xbyak_exact_m_matches_static_asm or \
      sve_w2_direct_route_store_matches_scatter or \
      sve_xbyak_strict_mode_rejects_kc_fallback'
```

Result: 16 passed. On `AmazonECS8Cores`, CPUs 0-7, the three exact-M degree
cases passed. For SiLU degrees 4/5/6, M=1..12, and normal, scheduled, and async
bridges, JIT and static-assembly outputs were bit exact. Strict JIT also rejects
split-K packing instead of silently using a static-assembly kernel; `auto`
retains that compatibility fallback.

## Generated-code check

`FUSED_CPP_MOE_SVE_JIT_DUMP_DIR=/tmp/moe_jit_double_buffer` dumped every
generated binary, which was decoded with GNU `objdump -D -b binary -m aarch64`.
For M2, M4, and M8, the K-loop instruction order matches the corresponding
static object body: preload `z0.../z4..7`, load the alternate
`z8.../z12..15` bank, compute the current bank, swap banks, and finish through
the same `tail_cur`/`tail_next` branches. M5/6 uses the same state machine with
three A row-pair registers. The generated W2 kernels now also preserve
`d8..d15`, as required after the alternate bank began using `z8..z15`.

## Amazon 192-core host, NUMA0

Command:

```bash
PYTHONPATH=src OMP_NUM_THREADS=96 OMP_DYNAMIC=FALSE \
  numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_xbyak_exact_m.py \
  --routes 1,2,3,4,5,6,7,8,9,10,11,12 --threads 1,2,4,8 \
  --warmup 8 --runs 40 --switch-period 4
```

| M | Static asm, 1T | Xbyak JIT, 1T | JIT gain |
| ---: | ---: | ---: | ---: |
| 1 | 2.650 ms | 2.648 ms | +0.05% |
| 2 | 2.599 ms | 2.597 ms | +0.08% |
| 3 | 2.714 ms | 2.721 ms | -0.25% |
| 4 | 2.735 ms | 2.734 ms | +0.03% |
| 5 | 3.482 ms | 3.065 ms | +13.61% |
| 6 | 3.494 ms | 3.073 ms | +13.69% |
| 7 | 3.498 ms | 3.499 ms | -0.02% |
| 8 | 3.512 ms | 3.503 ms | +0.26% |
| 9 | 4.303 ms | 3.904 ms | +10.22% |
| 10 | 4.308 ms | 3.911 ms | +10.13% |
| 11 | 4.317 ms | 4.335 ms | -0.43% |
| 12 | 4.331 ms | 4.348 ms | -0.38% |

The corresponding steady-state gains over the static assembly path are:

| M | 1T gain | 2T gain | 4T gain | 8T gain |
| ---: | ---: | ---: | ---: | ---: |
| 1 | +0.05% | -0.15% | -0.06% | -0.23% |
| 2 | +0.08% | +0.03% | -0.22% | +0.11% |
| 3 | -0.25% | -0.15% | -0.08% | -0.18% |
| 4 | +0.03% | -0.07% | -0.17% | -0.05% |
| 5 | +13.61% | +7.01% | +5.90% | +0.10% |
| 6 | +13.69% | +6.97% | +5.94% | -0.00% |
| 7 | -0.02% | -0.03% | +0.08% | -0.11% |
| 8 | +0.26% | +0.08% | +0.04% | -0.09% |
| 9 | +10.22% | +9.43% | +8.61% | +2.59% |
| 10 | +10.13% | +9.45% | +8.52% | +1.98% |
| 11 | -0.43% | -0.33% | -0.69% | -0.49% |
| 12 | -0.38% | -0.50% | -0.50% | -0.53% |

Before the double-buffer migration, M=7/8 regressed by 2.75%/1.91% at 1T,
4.26%/4.17% at 2T, and 3.49%/2.62% at 4T. Reproducing the static state machine
brings all six points to within 0.26% of static assembly. M=5/6 still executes
only three row pairs and retains its exact-M gain.

Prior steady long-route control (the M12 body is unchanged by this migration):

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
  --warmup 8 --runs 40 --switch-period 4
```

| M | 1T gain | 2T gain | 4T gain | 8T gain |
| ---: | ---: | ---: | ---: | ---: |
| 1 | -0.02% | -0.11% | -1.17% | -1.03% |
| 2 | +0.35% | +0.03% | -1.06% | -0.28% |
| 3 | +0.45% | +0.17% | -0.45% | -0.82% |
| 4 | -0.49% | -0.38% | -0.50% | -0.78% |
| 5 | +6.66% | +11.83% | +11.98% | +5.86% |
| 6 | +6.42% | +12.05% | +11.53% | +4.41% |
| 7 | +0.10% | +0.04% | -0.20% | -0.81% |
| 8 | +0.03% | -0.29% | -0.65% | +0.33% |
| 9 | +10.45% | +9.90% | +9.29% | +7.19% |
| 10 | +10.32% | +9.91% | +8.68% | +6.30% |
| 11 | -0.13% | -0.45% | -0.73% | -0.65% |
| 12 | -0.14% | -0.28% | -0.60% | -0.12% |

## Cost-model recalibration

The profiles in this section predate the double-buffer migration. Their schema
and scheduling domains remain valid, but M<=8 isolated/contention times should
be regenerated before using them for a new absolute-accuracy claim.

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
they are throughput controls rather than expected wins. The migrated
double-buffer state machine makes the M<=8 controls throughput-neutral on V3
and within 1.2% on V1. Full M12 panels retain static-assembly throughput within
about 1%; static assembly remains the compatibility and unsupported-epilogue
fallback.
