# Per-width model error and which widths quick may use (C9g TP4, 2026-09-23)

Design frozen before measuring: `tmp/c9g_width_error_20260923/design.md`. C9g node 0 (CPUs
0-95, `--membind=0`), TP4 expert shape (H=4096, F=512, E=256, degree 4), jemalloc never-purge,
`wait_idle` before every measured process. Widths studied W = {1..16, 24, 32, 48, 96}.

Two calibrations:

- **R**: the guarded research calibration, as in production (widths 1/2/4/8/16/32/48/96,
  1T and 2T unreliable). A width outside its set is priced with the model's common
  fallback for every per-width term.
- **X**: one-click v2 with `supported_widths=W`. Its per-width overheads are fitted at every
  width (machine probe 15.6 s plus training 62.4 s; training fit MAPE 9.6%).

## Step 1: model error per width

**Isolated, held out** (`profile_contention_async.py`): routes 2/6/9/13/24/36/96/384/1024,
which are disjoint from X's training routes, at every width in W, two passes, 180 points.
Mean |error| (signed mean):

| width | 1 | 2 | 3 | 4 | 6 | 8 | 9-15 | 12 | 16 | 24 | 32 | 48 | 96 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R (fallback) | 3.5 (-0.4) | 2.8 (-0.4) | 8.1 (+6.8) | 4.3 (+1.9) | 11.5 (+7.6) | 8.4 (-4.2) | 19-21 (+18 to +21) | 19.9 (+19.8) | 9.4 (+1.0) | 23.4 (+23.4) | 12.0 (+0.7) | 13.8 (+5.1) | 6.1 (+4.5) |
| X | 4.0 (-1.0) | 3.9 (-1.9) | 5.5 (+1.3) | 5.2 (+2.7) | 10.2 (+3.2) | 9.3 (+2.8) | 10.7-11.5 (+2 to +4) | 11.5 (+3.3) | 9.0 (-1.1) | 11.4 (+0.1) | 12.6 (+0.2) | 12.8 (+3.9) | 3.5 (-0.6) |

- **Uncalibrated widths are overpriced by about 20% in isolation.** The fallback overheads do
  not fit them. Per-width training brings every width to the accuracy of the calibrated ones
  (signed error within +4%).
- **1T and 2T are accurate in isolation** (under 4% with either calibration). Their failure
  in whole plans therefore comes from concurrency and imbalance, not from the per-expert price.

**Whole plans** (M2, `bench.py`, protocol v2, bitwise gate, two sessions). Workloads: 18 real
TP4 layers (requests 008/016/022 x layers 2/9/15/25/36/40) and the 9 catalog presets. Each
has the homogeneous LPT plan of X for every divisor width in {1,2,3,4,6,8,12,16,24,32,48,96}.
Median signed error of X's quick score against the measured plan:

| width | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24 | 32 | 48 | 96 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| X quick score | -28% | -20% | -36% | -24% | -31% | -36% | -51% | -49% | -41% | -32% | -14% | -3% |
| fastest measured on | 0 | 1 | 0 | 5 | 0 | 15 | 4 | 2 | 0 | 0 | 0 | 0 |

The quick score is an isolated-LPT lower bound, so only its ranking matters. It is 13-15
points more optimistic at 12T and 16T than at 8T. That offset is why quick prefers 16T
where 8T is measured fastest on real layers (the base set's median regret against the
fastest width is 1.69%). It is also why 12T gets chosen where it loses.

## Step 2: offline search - which widths enter quick

Quick selects one homogeneous width per workload by its score. With every width's plan
measured, any width set's outcome is known offline: `planners/width_admission.py` evaluates
it. Base P = {4,8,16,32,48,96} (production's search). Each candidate is added alone and judged
by the frozen rule: excluded if any workload is more than 1% slower (mean of two sessions);
admitted if at least 1% faster on some workload or 0.3% in median; neutral otherwise.

| + width | workloads changed | best | worst | verdict | where |
| --- | --- | --- | --- | --- | --- |
| 1T | 0/27 | 0 | 0 | neutral | never selected |
| 2T | 4/27 | -3.0% | +22.2% | exclude | active-set-64 +21.6% / +22.9% per session |
| 3T | 1/27 | 0 | +29.6% | exclude | active-set-32 |
| 6T | 2/27 | -20.1% | 0 | **admit** | active-set-32 -18.7% / -21.4%, active-set-16 -4.4% / -7.4% |
| 12T | 14/27 | -23.9% | +5.8% | exclude | 12 real layers +1.8% to +5.9%, both sessions alike |
| 24T | 2/27 | -15.4% | 0 | **admit** | active-set-8 -11.8% / -18.6%; active-set-16 +1.2% / -4.7% (passes on the mean only) |

The joint set P + {6, 24} is nowhere slower than P. Its maximum regret against the fastest
measured width falls from 33.4% to 11.0%; the median stays 1.69% (the 8T/16T ranking above).

**Tier 2 (long/short partition, bimodal only; no real layer has a route gap of 8 or more).**
Measured `part_w` (5 long lanes of w cores, the remaining cores as 1T short lanes), in ms:
6T 18.13, 8T 13.41, 10T 11.75, 12T 10.47, 13T 10.37, **14T 10.22**, 15T 10.55, 16T 10.99.
The candidate selects 13T with X (regret 1.42%) and 12T with R's fallback (2.42%). Both fail
the frozen 1% rule, so the partition widths are **not admitted**, and the candidate stays
opt-in. The placed DAG underprices every partition by 11-31%, most at 11-13T.

## Step 3: graded release

- **Tier 1 released as a calibration step, not a code default.** The evidence is one machine
  and one shape, so no machine-independent width rule follows from it.
  `python cpu_moe_schedule_optimization/planners/width_admission.py --calibration X
  --planner-widths 4,6,8,16,24,32,48,96 --output ...` writes a calibration whose planner
  searches exactly those widths. Every other width stays runnable and is listed as
  unreliable. The tool refuses a width without its own fitted overheads, the failure found
  in step 1. For C9g TP4 this gives `tmp/c9g_width_error_20260923/cal_x_admitted.json`.
- **Check** (`verify_admitted.py`, on C9g with the native planner and locally with the Python
  one): quick with that calibration emits, on all 27 workloads, bridges identical to the
  measured fixed-width plans the offline search selected. Its outcome is therefore the measured
  one. Against production quick (R): median +0.00%, best -20.1% (active-set-32), worst +0.20%.
- **Tier 2** (partition widths) is not released. **1T/2T** are not released: 2T is excluded
  again and 1T is never selected. Fixing them needs a concurrency model change; recalibration
  does not help, since their isolated price is already accurate.

## Limits

One machine, one shape, 27 workloads. The small active-set presets are noisy (16T differed
by 6-14% between sessions), and both admitted widths draw their gains from them. The
tier-1 verdicts cover quick's homogeneous search only, not full search's mixed shapes.
