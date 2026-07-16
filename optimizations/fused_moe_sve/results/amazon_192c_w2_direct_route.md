# Amazon C5 192-core SVE W2 direct route store

## Setup

- Host: `AmazonC5192Cores`
- CPU: 192-core Arm Neoverse-V3, SVE vector length 16 bytes
- Binding: NUMA0, CPUs `0-95`
- Runtime: Linux `7.0.0-1006-aws`, Python 3.12.13, PyTorch 2.13.0+cpu
- Build: GCC 15.2.0, AArch64 SVE/BF16 extension, final optimization flag
  `-O2`
- Workspace base: `8292661` plus the uncommitted direct-route feature
- Shape: `H=4096`, `F=512`, `E=8`, `top_k=6`, split-W13
- Schedule: eight concurrent expert teams, 12 threads per expert
- Statistic: long routes use 31 alternating A/B runs after five warmups;
  M=12/48/192 use 51 runs after ten warmups
- Baseline: FP32 W2 writes contiguous `down`, then N owners scatter to
  `route_out`
- Direct: the default W2 writes `route_out` directly and leaves the weighted
  FP32 merge unchanged; `FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE=0` selects baseline

## Results

| Routes/expert | Bridge | Baseline ms | Direct ms | E2E gain |
|---:|---|---:|---:|---:|
| 12 | async | 0.487 | 0.492 | -0.95% |
| 48 | async | 0.715 | 0.718 | -0.42% |
| 192 | async | 1.575 | 1.542 | 2.18% |
| 768 | async | 4.826 | 4.379 | 10.21% |
| 1536 | async | 9.410 | 8.125 | 15.82% |
| 1536 | scheduled | 9.442 | 8.265 | 14.24% |
| 1536 | normal | 261.006 | 214.114 | 21.90% |
| 2040 | async | 12.455 | 10.697 | 16.44% |

The normal bridge assigns these eight experts far less intra-expert parallelism
than the preplanned bridges and allocates per-call scratch, so its absolute time
is not a production schedule or kernel comparison. It is retained only to show
that the same epilogue is connected.

One instrumented async call gives the following critical-worker stage proxies.
Tracing perturbs E2E scheduling, so only the stage attribution is used here.

| Routes/expert | Variant | W2 ms | Scatter ms | W2 + scatter ms |
|---:|---|---:|---:|---:|
| 1536 | baseline | 1.995 | 1.329 | 3.324 |
| 1536 | direct | 2.009 | 0 | 2.009 |
| 2040 | baseline | 2.621 | 1.952 | 4.573 |
| 2040 | direct | 2.653 | 0 | 2.653 |

The irregular store increases W2 by only 0.7-1.2%. Removing scatter reduces
the combined W2-output stage by about 40-42%, which becomes a 14-16% operator
gain on the two long-route shapes.

## Why M <= 48 does not improve

The current N-owner path already has no W2-to-scatter team barrier. Both the
baseline and direct-route paths execute the same final team barrier, so the
candidate removes only the copy loop; it does not remove a synchronization
point.

Twenty-one repeated traced calls at M=48 give the following median
critical-worker times. The baseline combined value is computed as the maximum,
over workers, of that worker's W2 plus scatter time; it is not the sum of two
unrelated stage maxima.

| Variant | W2 us | Scatter us | Critical W2-output us |
|---|---:|---:|---:|
| baseline | 65.806 | 3.274 | 68.566 |
| direct route | 67.803 | 0 | 67.803 |

The saved critical-path work is therefore only 0.763 us, about 0.1% of the
roughly 0.7 ms operator. Traced E2E time is not compared because the baseline
records one additional phase per worker and trace locking perturbs the
schedule. In an untraced 1001-pair alternating repeat, after removing the one
sample with 96 minor faults, the baseline/direct medians were 0.77065/0.77535
ms. The paired median delta was +7.69 us, but its approximate nonparametric 95%
interval was [-3.06, +16.54] us and direct was faster in 47.5% of pairs.
Twenty-five-pair block medians ranged from -45 to +76 us. The roughly 0.8 us
stage signal is therefore below E2E scheduling noise; the robust conclusion is
no measurable short-route speedup, not a proven intrinsic regression.

For H=4096 and 12 threads per expert, one worker's logical scatter source plus
destination footprint is

```text
Wscatter/thread = 2 * M * H * sizeof(float) / T.
```

| Routes/expert | Down/expert | Scatter bytes/thread | Fraction of 2 MiB L2 |
|---:|---:|---:|---:|
| 12 | 192 KiB | 32 KiB | 1.6% |
| 48 | 768 KiB | 128 KiB | 6.2% |
| 192 | 3 MiB | 512 KiB | 25.0% |
| 768 | 12 MiB | 2 MiB | 100.0% |

At M=48 the just-produced `down` stripe is private-L2 hot. The 12 MiB logical
copy across all 96 workers completes in about 3.3 us, equivalent to distributed
cache bandwidth rather than NUMA DRAM bandwidth. A broader W2-stage working-set
estimate is

```text
Wthread ~= sizeof(B) / T + M * K * sizeof(bf16)
           + 2 * M * N * sizeof(float) / T.
```

For W2 K=512, N=4096, and T=12 this is about 0.51 MiB at M=48,
1.02 MiB at M=192, and 3.08 MiB at M=768. The latter exceeds the 2 MiB private
L2, matching the point where the direct path begins to remove material
L2/LLC/memory pressure.

`perf stat` was attached after warmup to all 96 existing worker TIDs. Counts
below are aggregate events per operator call; independent-process cycle counts
are too noisy for an A/B latency claim, but the cache-level transition is
large and consistent with the stage timings.

| M | Variant | L2 refills/call | LL reads/call | LL read misses/call |
|---:|---|---:|---:|---:|
| 12 | baseline | 286.7k | 25.4k | 349 |
| 12 | direct | 264.6k | 23.6k | 374 |
| 48 | baseline | 1.262M | 68.9k | 663 |
| 48 | direct | 1.098M | 52.3k | 565 |
| 768 | baseline | 29.019M | 1.744M | 1.255M |
| 768 | direct | 26.501M | 1.055M | 0.742M |

At M=48 the candidate removes L2 traffic, but both variants have only hundreds
of last-level read misses, so there is essentially no DRAM latency to save. At
M=768, direct route removes about 39% of LL reads and 41% of LL read misses;
the same run improves 4.843 ms to 4.378 ms. A separate M=48 TLB event run also
showed no increase in L1/L2 DTLB refills or page walks, ruling out translation
pressure as the cause of the short-route regression. Steady M=48 calls have
zero or one minor fault; one 1001-run direct sample had 96 faults, but it does
not affect the median, so first-touch is not the median-time cause either.

The remaining direct-store cost is visible in the assembly. For every M12
block, every SVE N tile, and every one of six row pairs, it reloads route IDs
and constructs scalar/vector byte offsets before the scatter stores. At M=48 a
12-thread owner processes 42-43 N tiles, so this is roughly
`4 * 43 * 6 = 1032` pair-address constructions per worker. The old scatter
computes one destination row address and then copies that worker's contiguous
H stripe. Hoisting direct-route row addresses out of the N loop is the relevant
kernel optimization if short-M support becomes important; otherwise dispatch
should retain the baseline while the per-thread W2/scatter footprint is safely
below private-L2 capacity.

## Allocation-tail control

At M=2040, separate baseline-only and direct-only processes both show periodic
large route-buffer first-touch events. Faulted baseline samples have roughly
9.2k-9.4k minor faults and the 31-run P90 is 27.732 ms; faulted direct samples
have roughly 9.6k-9.9k minor faults and P90 is 26.934 ms. Steady samples have
0-32 minor faults. Alternating A/B changes which variant receives these events,
so median and P10 are the appropriate steady-kernel comparison; neither variant
eliminates the operator's internal `route_out` allocation. A true native `out=`
buffer remains the separate fix for this tail source.

## Traffic

Let `R = routes * H * sizeof(float)` be one FP32 route tensor. The baseline
performs W2 write `down`, scatter read `down`, scatter write `route_out`, and
merge read `route_out`, or `4R` logical bytes. Direct route store performs W2
write `route_out` and merge read `route_out`, or `2R`.

| Routes/expert | R | Baseline | Direct | Saved |
|---:|---:|---:|---:|---:|
| 1536 | 192 MiB | 768 MiB | 384 MiB | 384 MiB |
| 2040 | 255 MiB | 1020 MiB | 510 MiB | 510 MiB |

The candidate also avoids the simultaneous contiguous `down` scratch: 24 MiB
per expert (192 MiB total) at M=1536 and 31.875 MiB per expert (255 MiB total)
at M=2040. These are logical/cache-level bytes, not a DRAM-bandwidth claim.

## Correctness

The focused test covers normal, scheduled, and async bridges; interleaved
TopK=6 route rows; N-split ownership; and expert route counts 1 through 23,
which exercise M12 plus every M8/M4/M2/M1 tail. Candidate and baseline BF16
operator outputs match bit for bit.

```bash
PYTHONPATH=src numactl --cpunodebind=0 --membind=0 taskset -c 0-4 \
  .venv/bin/python -m pytest -q tests/test_fused_moe_bf16_tiled.py \
  -k w2_direct_route_store_matches_scatter
```

Focused result: `6 passed, 58 deselected`. The complete file result is
`63 passed, 1 skipped`.

## Reproduction

```bash
PYTHONPATH=src numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_w2_direct_route.py \
  --path async --tokens 2048 --hidden 4096 --intermediate 512 \
  --experts 8 --top-k 6 --threads 96 --warmup 5 --runs 31

# 2720 * 6 / 8 = 2040 routes per expert.
PYTHONPATH=src numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_w2_direct_route.py \
  --path async --tokens 2720 --hidden 4096 --intermediate 512 \
  --experts 8 --top-k 6 --threads 96 --warmup 5 --runs 31
```

Direct route is default-on. M=12 and M=48 have no measurable gain, M=192 gains
only 2.18%, and long routes gain 14-16%; the default also removes the per-team
FP32 `down` allocation. `FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE=0` retains the
baseline for comparisons and machine-specific fallback experiments.
