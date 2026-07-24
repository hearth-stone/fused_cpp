# Exact-M first-cold-panel software prefetch

## Scope

- Host: `AmazonC5192Cores`, NUMA node 0, CPUs 0-95.
- CPU topology relevant to this experiment: 64 KiB L1D and 2 MiB private L2
  per core, 96 MiB shared L3 per NUMA node.
- Shape: `H=4096`, `F=512`, split-W13 enabled. One expert owns an 8 MiB W13
  weight, processed as two serial 4 MiB N ranges.
- Build: native MoE sources compiled with `-O2`; 128-bit runtime SVE BF16.
- Workspace: parent `fe3eb3f` plus the uncommitted
  `load.w13_first_panel_prefetch` experiment.

The generated candidate preserves the M12 BFMMLA, SiLU, and packC sequences.
It inserts one `PRFM PLDL1STRM, [x14, #2048]` before each cache-line B load in
the first full M12 panel of every thread-owned N range. Later M panels use the
ordinary generated kernel. On the last N tile, the final 2 KiB uses ordinary
loads so the hint cannot cross the owned N range.

The candidate is selected only with:

```text
FUSED_CPP_MOE_SVE_IMPL=jit
FUSED_CPP_MOE_SVE_W13_FIRST_PANEL_PREFETCH=1
FUSED_CPP_MOE_SVE_JIT_BULK_M=0
```

It remains off by default.

## Correctness and generated code

The original M12-only test covered polynomial degrees 4, 5, and 6, one and
multiple M12 panels, and an exact tail. All outputs were bitwise equal. Binary
dump disassembly confirmed the 2 KiB `PLDL1STRM` before B `LD1H` and the
ordinary load tail on the final N tile. The current superset command and result
are recorded in the exact-M extension section below.

## Sequential-expert A/B

Each timed call ran eight experts in serial waves and alternated two disjoint
weight windows. Every point has five warmups and 31 timed samples. Positive
gain means the prefetch candidate is faster.

| Routes | 1T/expert gain | 4T/expert gain |
|---:|---:|---:|
| 12 | +1.87% | +2.39% |
| 24 | +1.03% | -1.54% |
| 48 | -0.20% | -4.94% |
| 96 | -0.38% | -3.13% |
| 192 | -0.49% | -0.23% |

Five independent 4T processes reproduced the boundary:

| Routes | Gain range | Median process gain |
|---:|---:|---:|
| 12 | +2.14% to +2.93% | **+2.19%** |
| 24 | -1.62% to -0.68% | **-1.57%** |
| 48 | -4.44% to -3.39% | **-4.35%** |
| 96 | -3.85% to -2.28% | **-2.62%** |

A thread-count sweep gave `M=12` gains of `+1.94%, +1.90%, +2.50%, +1.68%`
at 1T, 2T, 4T, and 8T. At `M=48` the corresponding changes were
`-0.35%, -1.70%, -5.30%, +0.24%`; the negative effect is not monotonic in
thread count, so thread count alone is not a valid dispatch rule.

Command:

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-7 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_xbyak_exact_m.py \
  --variants jit-panel,jit-prefetch --routes 12,24,48,96,192 \
  --threads 1,4 --experts 16 --measurement-experts 8 \
  --warmup 5 --runs 31 --switch-period 5
```

## Phase attribution

The scheduled-path trace used 15 distinct expert weights per variant. Values
below are medians; W13 and W2 are the maximum worker time in the single expert
team. The trace instrumentation changes absolute e2e latency, so only paired
values from the same row are compared.

| Routes | Threads | Stage | Panel JIT ms | Prefetch ms | Gain |
|---:|---:|---|---:|---:|---:|
| 12 | 1 | W13 | 0.3597 | 0.3379 | +6.43% |
| 12 | 1 | W2 | 0.1628 | 0.1627 | +0.09% |
| 12 | 1 | e2e | 0.5597 | 0.5414 | +3.39% |
| 12 | 4 | W13 | 0.0975 | 0.0926 | +5.35% |
| 12 | 4 | W2 | 0.0475 | 0.0479 | -0.94% |
| 12 | 4 | e2e | 0.1912 | 0.1853 | +3.21% |
| 48 | 1 | W13 | 1.3914 | 1.3983 | -0.49% |
| 48 | 1 | W2 | 0.6518 | 0.6505 | +0.20% |
| 48 | 1 | e2e | 2.1452 | 2.1529 | -0.36% |
| 48 | 4 | W13 | 0.3895 | 0.4420 | -11.88% |
| 48 | 4 | W2 | 0.1824 | 0.1805 | +1.04% |
| 48 | 4 | e2e | 0.6338 | 0.6830 | -7.20% |

The regression is inside W13 rather than a slower following W2. At 4T, each
thread's stripe in one 4 MiB W13 range is about 1 MiB, which fits the 2 MiB
private L2 and is reused by later M panels.

### Causal panel-sequence control

A follow-up standalone harness separated the `PRFM` data-side effect from the
larger generated function and the function-pointer transition. It compared:

- `normal`: the ordinary M12 JIT kernel for every panel;
- `prefetch`: the generated prefetch kernel for panel 1, then the ordinary
  kernel;
- `sham`: a byte-for-byte copy of the 2924-byte prefetch kernel with its four
  static `PLDL1STRM` instructions replaced in place by `NOP`, then the ordinary
  kernel.

Thus `prefetch` and `sham` have identical code size, branch layout, and
prefetch-to-normal function switch. Their only instruction difference is
`PLDL1STRM` versus `NOP`. All three outputs were bitwise equal. Tests used one
NUMA-local core and a 1 MiB B stripe (`K=4096,N=128`), matching one lane of a
4-thread split-W13 expert. Every timed sequence consumed a previously unused B
copy. Three independent processes produced:

| M12 panels | Prefetch latency change vs sham | Sham latency change vs normal |
|---:|---:|---:|
| 1 | -9.2% to -7.5% | -0.9% to -0.5% |
| 2 | +3.9% to +5.9% | -1.0% to +0.3% |
| 3 | +19.9% to +22.0% | -1.0% to +0.8% |
| 4 | +14.1% to +16.7% | +0.1% to +0.3% |

The sham result rules out JIT code size and the prefetch-to-normal function
switch as the material cause. The four-panel microkernel penalty also matches
the `M=48,4T` fused W13 phase regression above: `0.4420 / 0.3895 - 1 = 13.5%`.

A cumulative PMU run over 12,800 sequences per point gave the following
per-panel differences after subtracting adjacent 1/2/3/4-panel totals. The PMU
run rotates 128 B copies and therefore has a warmer cache history than the
one-use timing control; it is evidence for mechanism, not the production
latency estimate.

| Panel | Cycles, prefetch vs sham | Memory-stall cycles | `L2D_CACHE_LMISS_RD` |
|---:|---:|---:|---:|
| 1 | -4.6% | -47.8% | -5.8% |
| 2 | +16.4% | +67.0% | +5.8% |
| 3 | -0.8% | +18.5% | +2.9% |
| 4 | +0.7% | -19.5% | +56.6% |
| 1-4 total | +3.0% | +3.1% | +19.3% |

Retired instructions differed by less than 0.02%. The ordinary architectural
`L2D_CACHE_REFILL` event moved in the opposite direction from the
implementation-defined long-miss event, so it is not used here as a total
traffic estimate: software-prefetch and demand-refill attribution is not known
to be equivalent on this Neoverse-V3 PMU.

The supported conclusion is narrower than the original cache-residency
inference: `PLDL1STRM` removes stalls from the first cold panel but leaves a
cache/memory-system state that raises later long-miss and memory-stall cost by
more than it saved. These counters do not prove a specific per-line L2
allocation or replacement policy, and the exact panel receiving the delayed
cost changes with measurement and cache history.

### B-range boundary control

The generated M12 kernel has four static `PRFM` sites: two in the bounded loop
used by the final N tile, and two in the unbounded loop used only when another
N tile follows. Additional binary-patched controls retained only one pair. For
four M12 panels and the same 1 MiB B stripe, three independent processes gave:

| First-panel policy | Latency change vs all-NOP sham |
|---|---:|
| All N tiles prefetch | +15.7% to +16.5% |
| Non-final N tiles only | +13.3% to +15.3% |
| Final N tile only | -2.0% to +1.4% |

The final-tile loop also bounds the explicit addresses. At the 128-bit runtime
VL, one packed-B N tile is `4096 * 16 = 65,536` bytes and each two-body loop
advances 128 bytes. With a 2048-byte distance, its last two hints target
`B_end - 128` and `B_end - 64`; the remaining 2048 bytes use demand loads only.
An unbounded non-final tile can target up to 1984 bytes into the next contiguous
tile, but that path is selected only while such a tile exists.

Therefore an explicit hint beyond the caller-owned B range is not the source
of this regression. Nearly all of the penalty remains after every final-tile
hint is removed; it is caused by the bulk streaming-prefetch policy over the B
range. This experiment still does not identify the exact private-cache
replacement mechanism.

## Saturated 24x4T execution

Twenty-four experts ran concurrently on all 96 NUMA0 cores. Five independent
processes produced these e2e gain ranges:

| Routes | Gain range | Median process gain |
|---:|---:|---:|
| 12 | -1.58% to +2.23% | +0.12% |
| 48 | -2.46% to -0.14% | -1.82% |
| 192 | -2.05% to -1.36% | -1.64% |

The isolated M12 benefit is not stable once 24 cold experts compete for the
NUMA bandwidth/cache path. Longer routes remain a reproducible regression.

### Pure-M12 bandwidth crossover

A separate one-core-per-cold-B sweep isolates the best possible case for this
prefetch policy: one `M=12,K=4096,N=512` GEMM panel, a 4 MiB B stream, and no
later B reuse. `B GB/s` below is logical B bytes divided by synchronized wave
wall time, not an uncore DRAM counter. Values are medians of three independent
waves per point.

| Concurrent cores | Baseline B GB/s | Prefetch B GB/s | Prefetch change |
|---:|---:|---:|---:|
| 8 | 134.8 | 153.7 | +14.0% |
| 16 | 242.6 | 270.2 | +11.4% |
| 18 | 258.8 | 274.9 | +6.2% |
| 20 | 272.2 | 258.7 | -4.9% |
| 22 | 281.8 | 265.3 | -5.9% |
| 24 | 285.1 | 270.7 | -5.0% |
| 48 | 290.2 | 286.9 | -1.1% |
| 96 | 299.8 | 298.0 | -0.6% |

The ordinary kernel's measured plateau is about 300 effective B GB/s. The
prefetch crossover occurs between 18 and 20 concurrent streams, when baseline
traffic reaches roughly 86-91% of that kernel-specific plateau. Prefetch hides
exposed latency while service bandwidth and request capacity remain available;
near saturation it cannot reduce compulsory B bytes and instead competes with
demand loads for cache/memory request resources. Therefore prefetch selection
must include aggregate cold-B pressure, not just an `M <= 12` condition.

## Decision

Do not enable first-panel `PLDL1STRM` for general W13 dispatch. The production
wiring remains an explicit experimental flag and conflicts with bulk-M.

The useful domain is a single-panel `M=12` expert under low or moderate
contention: W13 improves by roughly 5-6% and the complete one-expert operation
by roughly 2-3%. Applying the same policy before reusable B panels is incorrect
for performance. A future candidate should either gate on `rows == 12` plus a
contention signal, or find a cache-retaining prefetch policy; it should not
prefetch the first panel of a long-route expert with `PLDL1STRM`.

## Exact-M and all-GEMM extension

The follow-up JIT generates both ordinary and prefetch kernels for every
logical M from 1 through 12. The first actual W13 panel uses
`PLDL1STRM #2048`; later panels use the ordinary function pointer. A separate
all-GEMM flag also selects `PLDL2STRM #1024` for the first FP32 W2 or
direct-route W2 panel. M1-M8 retain the original two-bank K-loop state machine.

Correctness covered M1-M12, M13, M24, and M25, polynomial degrees 4-6, and both
FP32 W2 output modes:

```bash
PYTHONPATH=src FUSED_CPP_MOE_SVE=1 OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
  taskset -c 0-7 .venv/bin/python -m pytest -q \
  tests/test_fused_moe_bf16_tiled.py \
  -k 'sve_first_panel_prefetch_matches_panel_jit or \
      sve_first_panel_prefetch_rejects_bulk_m or \
      sve_xbyak_exact_m_matches_static_asm'
```

Result: `10 passed, 91 deselected`; all comparisons were bitwise equal and the
bulk-M conflict was rejected as documented.
Generated M1 and M12 binaries contain the expected W13 `PLDL1STRM #2048` and
W2-direct `PLDL2STRM #1024` instructions.

Five independent processes compared eight serial experts per call. Values are
the median process gain relative to ordinary panel JIT:

| M | W13-only 1T | All GEMMs 1T | W13-only 4T | All GEMMs 4T |
|---:|---:|---:|---:|---:|
| 1 | +0.90% | +0.40% | +0.95% | +0.66% |
| 2 | +1.10% | +0.98% | +0.77% | +1.20% |
| 3 | +1.00% | -0.34% | +1.01% | +0.97% |
| 4 | +1.12% | -0.04% | +0.50% | +0.75% |
| 5 | +1.93% | +1.37% | +2.12% | +2.25% |
| 6 | +1.86% | +1.25% | +2.05% | +2.24% |
| 7 | +3.38% | +2.87% | +5.16% | +5.40% |
| 8 | +3.53% | +3.03% | +5.02% | +5.05% |
| 9 | +2.07% | +1.93% | +2.36% | +2.23% |
| 10 | +1.88% | +1.52% | +2.13% | +2.09% |
| 11 | +2.35% | +2.22% | +2.48% | +2.94% |
| 12 | +2.13% | +2.28% | +2.81% | +2.74% |

Across all M/process samples, the median W13-only gain was `+1.93%` at 1T and
`+2.10%` at 4T. Enabling W2 changed these to `+1.60%` and `+2.24%`; W2's
incremental medians were `-0.37%` and `+0.09%`, respectively.

The required contention control ran 24 experts concurrently with four threads
per expert on all 96 NUMA0 cores. Median process gains were:

| M | W13-only | All GEMMs | W2 incremental |
|---:|---:|---:|---:|
| 1 | +1.80% | -1.87% | -4.23% |
| 2 | +1.50% | -2.86% | -4.37% |
| 3 | +1.63% | -4.94% | -6.53% |
| 4 | -0.13% | -4.58% | -6.26% |
| 5 | +2.23% | -5.09% | -7.38% |
| 6 | +0.63% | -5.69% | -5.25% |
| 7 | +0.88% | -4.65% | -5.96% |
| 8 | +1.26% | -4.92% | -6.10% |
| 9 | +2.42% | -6.65% | -8.70% |
| 10 | +2.63% | -8.84% | -11.05% |
| 11 | +3.59% | -6.61% | -9.87% |
| 12 | +0.17% | -9.30% | -10.83% |

Across all M/process samples, W13-only was `+1.72%`, while all-GEMM prefetch
was `-5.99%`; the W2 incremental median was `-7.36%`. Therefore the exact-M
W13 specialization is a useful opt-in candidate, but W2 prefetch is rejected
for saturated execution and the all-GEMM flag must remain experimental and off
by default.

Base commands used `H=4096`, `F=512`, split-W13, five warmups, 31 samples, and
five independent processes:

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_xbyak_exact_m.py \
  --variants jit-panel,jit-prefetch,jit-prefetch-all \
  --routes 1,2,3,4,5,6,7,8,9,10,11,12 --threads 1,4 \
  --experts 64 --measurement-experts 8 --experts-per-wave 1 \
  --warmup 5 --runs 31 --switch-period 5

numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_xbyak_exact_m.py \
  --variants jit-panel,jit-prefetch,jit-prefetch-all \
  --routes 1,2,3,4,5,6,7,8,9,10,11,12 --threads 4 \
  --experts 48 --measurement-experts 24 --experts-per-wave 24 \
  --warmup 5 --runs 31 --switch-period 5
```
