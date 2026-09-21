# Synthetic M ladder and clustered/staggered scheduling

Status: completed. Staggered wins on three mixed grids, not on (1,8).
Lab-only, no production
default, calibration, cost formula, native build or packed-format changes.

## Results

Uniform-M reference measurements (all ten lanes have the same task M):

| M | Median time S1 / S2 ms | Read GB/s S1 / S2 | Occupancy/command S1 / S2 |
| ---: | ---: | ---: | ---: |
| 1 | 3.60303 / 3.59283 | 209.76 / 210.19 | 339.04 / 339.94 |
| 4 | 3.65147 / 3.64981 | 209.04 / 208.57 | 333.80 / 333.58 |
| 16 | 5.08699 / 5.11461 | 211.83 / 211.35 | 246.05 / 245.68 |
| 64 | 14.87418 / 14.81702 | 91.51 / 91.46 | 218.39 / 216.38 |
| 256 | 50.84256 / 50.86924 | 41.03 / 41.06 | 214.07 / 211.61 |

Small M sustains high aggregate read traffic under this concurrency, while
large-M average rates fall substantially. M1 is not uniquely highest by read
throughput (M16 is comparable); average rate alone is not a saturation proof.
Occupancy/command is a diagnostic ratio, not separately calibrated queue latency.

Mixed cases, positive gain means staggered is faster:

| Low/high M | S1 clustered / staggered ms | S2 clustered / staggered ms | Paired median gain S1 / S2 | P10 S1 / S2 | Wins S1 / S2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / 8 | 3.62062 / 3.58846 | 3.62504 / 3.56255 | +0.170% / +0.774% | −4.017% / −2.240% | 16/31 / 21/31 |
| 4 / 32 | 5.01538 / 4.82950 | 5.02327 / 4.85422 | **+3.514% / +3.369%** | +1.807% / +0.122% | 30/31 / 28/31 |
| 16 / 128 | 16.47851 / 13.15973 | 16.52491 / 13.21994 | **+25.384% / +25.060%** | +23.146% / +21.664% | 31/31 / 31/31 |
| 64 / 512 | 55.69962 / 47.43399 | 55.78260 / 47.46925 | **+17.265% / +17.576%** | +15.318% / +14.874% | 31/31 / 31/31 |

Three grids pass the predeclared both-session >2%-median/positive-P10 gate. The
(1,8) result is small and noisy and does not pass. Do not promote it based on
its favorable ratio of separate medians instead of the paired statistic.

For (16,128), staggered occupancy/command falls from about 208–209 to 152–153,
while read throughput increases from about 79 to 102–104 GB/s. For (64,512),
the ratio falls from about 208–211 to 165–166, with throughput rising from about
36.7 to 44.2–44.6 GB/s. These observations are consistent with reduced waiting
and better resource overlap, not a proof of a particular instantaneous pressure
curve. Total observed DRAM traffic is not assumed identical: cache/refill and
prefetch effects can change even when logical work and weights are unchanged.

## Interpretation and limits

This gives direct synthetic evidence that clustered versus staggered large/small
execution can matter greatly at fixed lane work/width. It does **not** support
"the smallest M or largest average bandwidth always yields the biggest gain".
A plausible interpretation is that (1,8) offers little resource complementarity,
whereas mixed compute/transfer behavior at larger M offers useful overlap; that
mechanism remains an inference, not an isolated causal measurement.

Large-M lower average DDR traffic does not establish all its stages are low
pressure. The (64,512) gain despite lower average rate is another reason not to
use average bandwidth alone as an acceptance/pruning metric. Only these fixed
templates, this kernel/shape, 8T width and early-merge-off configuration were
tested. No claim of globally optimal ordering or real-trace generalization.

All orders/cases passed bitwise BF16 equality on all four copies in both sessions.
All 48 event counters met >=99% running ratio. Both sessions have matching
complete configs (including routes and bridges), runner and extension identities.
The extension is `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`.
Scrub/copy rotation does not prove all-DRAM accesses, and actual allocation page
backing was not audited here. Case/plan order randomization and both-session
statistics retain noise rather than filtering it. No extra grid or model fit.

## Design

Fixed E60, H4096/F512, BF16 SVE N tile8, 10 lanes x8T, CPUs240–319/NUMA3.
Six distinct whole experts per lane, weights fixed across all cells and sessions.
Per expert W13=8 MiB, W2=4 MiB; owner footprints at 8T=1/0.5 MiB, windows0,
R13=R2=1. Each copy's expert-weight footprint is 720 MiB; four copies rotate.
No external background. Early merge is disabled in both plans so a different
ready-token merge schedule is not the intended intervention. End-to-end times
still include gather, final merge, dispatch and synchronization, not pure GEMM.

Uniform-M references: all 60 experts use M=1/4/16/64/256, five cells. These
measure the pressure ladder without pretending a reorder of identical M creates
a useful density contrast. Do not assume M1 has the highest measured pressure.

Mixed grids: (1,8), (4,32), (16,128), (64,512). Every lane owns three low-M
experts and three high-M experts. Clustered order is LLLHHH on every lane;
staggered order is LLLHHH on even lanes and HHHLLL on odd lanes. Core assignment,
per-lane total work and task sets are identical within each pair. No active delay,
task migration or width change. This is a simple staggered template, not an
optimized schedule and not proof that hardware density becomes uniform.

Route counts are realized exactly as TopK6 token rows with no duplicate expert
in a row. Total tokens = sum(counts)/6, so input/gather/output size and total FLOPs
intentionally grow across M grids. Compare **within-grid paired gains**, not
absolute times between different workloads. All data and routes are synthetic;
these are not captured real-trace inputs.

## Protocol and gates

Thirteen timed items per round: five uniform cases plus two orders for each of
four mixed cases. Randomize case order per round and order within mixed pairs.
Same packed copy per paired round; four-copy rotation and disjoint 216 MiB scrub
outside each timing/counter region. Five warmups, 31 effective rounds, two
independent process sessions (same tensor seeds, different order seeds).

Measure node3 SCCL25/27, all16 DDRC instances, three events each: read flux,
commands, occupancy. Require running/enabled >=99%, and bitwise output equality
for every order/case on all four copies before timing. Pilot uses one warmup and
one effective round, only to verify correctness and instrumentation.

Primary mixed gain is `100*(clustered/staggered-1)`. Report P10/median/P90 and
wins in each session. Positive median/P10 in both sessions supports a repeatable
local ordering benefit; >2% median in both is the prior actionability criterion.
Test whether gains are larger where **measured** pressure is higher; do not
rename M groups as pressure levels if counters contradict the expectation.
No post-hoc grid/budget expansion or model fit. Aggregate PMUs cannot prove
instantaneous uniformity or isolate individual experts' queue delay.

## Reproduction

Local ignored directory: `tmp/synthetic_m_density_20260907/`.
Remote: `Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/synthetic_m_density_20260907/`.
Runner: `optimizations/fused_moe_sve/benchmarks/bench_synthetic_m_density.py`.
No production sources are synced. Only runner and existing PMU helpers are copied.

```bash
env OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:src \
  timeout 300 numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/synthetic_m_density_20260907/bench_synthetic_m_density.py \
  --seed 20260907 --output tmp/synthetic_m_density_20260907/session1.json
```

Use `--pilot --output .../pilot.json` first; then two formal sessions with seeds
20260907/20260908, session1/session2 outputs. Existing outputs are rejected.
Records contain exact route histogram/hash, full PlanV2 bridges, event specs,
source/extension identity and every raw cell. No per-case native build.

Focused synthetic route/membership validation: 10 tests passed. Ruff passed.
With direct paired-analysis/completeness and canonical state tests, 17 tests pass.
Pilot verified all 13 cells, 48 DDRC events and all-copy bitwise correctness.

Analysis command:

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_synthetic_m_density.py \
  --sessions tmp/synthetic_m_density_20260907/session1.json tmp/synthetic_m_density_20260907/session2.json \
  --output tmp/synthetic_m_density_20260907/summary.json
```

The analyzer rejects incomplete 36x13 grids, inconsistent experiment identities,
nonfinite/invalid timings and pilot-only evidence. It retains all paired outcomes
and raw session hashes; no post-hoc favorable-case selection.
