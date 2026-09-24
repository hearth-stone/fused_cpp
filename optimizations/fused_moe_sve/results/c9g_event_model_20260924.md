# Probe-calibrated event model (v10/v11) on C9g (2026-09-24)

User decision (2026-09-24): calibrate the event model that the fast (hot_wide) planner depends on
on C9g by the Arm-codex procedure, instead of running fast with Arm-codex terms. Design and
deviation: `tmp/c9g_event_model_20260924/design.md`. C9g node 0 (CPUs 0-95, `--membind=0`), TP4
expert shape, SVE-128 (n_tile 8), jemalloc never-purge. Phase skeleton: the guarded C9g research
calibration.

## Probes

The Arm-codex probes P1-P4 (four dilation curves), P5 (whole-call overhead) and P6 (v11 mid-M
terms) were ported with only the core grid changed. Background core counts are
{0, 8, 16, 32, 48, 64, 80} plus the largest 4T background next to the target (92 for 2T and 4T,
88 for 8T, 80 for 16T, 64 for 32T). There were two sessions, 246 P1-P4 cells and 187 P6 cells
each.

One deviation from the frozen design, before any build. Session 1 at the Arm-codex background
sizing (1.2x the modeled target) left 12 P1-P4 and 28 P6 cells uncovered: the target outlived
its background, because contention here is far stronger than the model the sizing uses. The
background was raised to 3.0x, session 1 was discarded, and both sessions were rerun. Only two
cells stay uncovered (dropped by rule).

Also changed on 2026-09-24 by user decision: C9g chains keep only a fixed cooldown after our own
previous process, with no load-average wait. The machine is exclusive.

## The calibration (`build_c9g.py`, `bench_assets/moe_paper/amazon_c9g_96c_tp4/probe_event_v11_c9g_20260924.json`)

| term | C9g | Arm-codex |
| --- | --- | --- |
| eps, O(t) (P2 n=0 single-expert cells, same grid fit) | 0.08, 1 us x t (fit MAE 1.2%) | 0.065, 10 + 2.5 us x t |
| t_over (P5, median of 16 plan-session values) | 0.45 ms | 0.78 ms |
| D_LL, 4T target: n = 32 / 48 / 64 / max | 2.56 / 3.76 / 4.72 / 5.86 (n = 92) | 1.12 / 1.32 / 1.64 / 1.90 (n = 76) |
| D_LS, max over widths | 1.48 (2T) - 2.25 (16T) | <= 1.35 |
| D_SL, 4T at max n | 1.28 | 1.13 |
| D_SS, 2T / 4T / 8T at max n | 1.92 / 1.51 / 1.16 | <= 1.03 |
| loading weight w(24 / 48 / 96 / 192 / 384) | 0.79 / 0.71 / 0.50 / 0.26 / 0.27 | 0.79 / 0.56 / 0.35 / 0.18 / 0.11 |
| isolated correction c(t, M 24-384) | 2T 1.03-1.10, 4T 1.00-1.08, 8T 0.99-1.11, 16T 0.92-0.95 | 2T 0.93-0.95, others ~1 |

Contention on C9g is far stronger than on Arm-codex, and it is not confined to weight loading.
Loading against loading reaches 5-6x at full occupancy, and even steady phases dilate by up to
1.9x on narrow lanes. 17 points differ by more than 3% between sessions (session mean used).
The mechanism test fails as on Arm-codex: core-time excess per expert does not collapse across
widths. The curves are a width-indexed service table.

Not re-measured: g0 = 0.02 (window isolated gain) is carried over from Arm-codex. The base
calibration prices only its calibrated widths (1/2/4/8/16/32/48/96). The probes cover 2-32T,
so 48T and 96T are extrapolated.

## Check on data the calibration never saw

Plans: the 27 workloads x homogeneous widths measured in `tmp/c9g_width_error_20260923` (18 real
TP4 layers plus 9 catalog presets, two sessions). Script: `check_width_table.py`.

Measured / event-predicted call time, median per width: 1T 1.28, 2T 1.11, 4T 1.10, 8T 1.05,
16T 1.06, 32T 0.83, 48T 0.66, 96T 0.61. The 4-16T level matches Arm-codex's 1.07-1.10. Wide
teams are overpriced beyond the probed range.

Regret of each scorer's width choice against the fastest measured width in the set:

| scorer | widths 4/8/16/32 (fast): median / mean / max / >1% | calibrated divisors 1-96: median / mean / max / >1% |
| --- | --- | --- |
| quick score | 0.34% / 2.01% / 24.2% / 12 of 27 | 1.31% / 2.88% / 23.6% / 14 |
| analytic DAG | 1.73% / 2.18% / 7.8% / 17 | 1.93% / 2.29% / 7.8% / 18 |
| **event v11 (C9g)** | **0.00% / 1.09% / 14.2% / 7** | **0.00% / 2.10% / 23.6% / 9** |

The C9g event model is the best width selector of the three. Its misses:

- Six workloads miss by 1.5-3.0%, mostly 16T preferred where 8T is faster.
- The long/short bimodal batch picks 8T where 16T is 14.2% faster: the long experts on 8T run
  slower than predicted.
- Over all widths, active-set-32 picks 2T at 23.6% worse: 2T is under-predicted, the same
  residual Arm-codex's v11 kept. The asset marks 1T/2T unreliable, and fast never uses them.

Next: derive the fast planner's templates and lane scales on C9g under this model, then validate
fast against production quick on fresh layers.

## Fast planner configuration on C9g (stage 3, model only)

`derive_fast.py`: on the 18 derivation layers (r008/r016/r022 x layers 2/9/15/25/36/40) the
model-objective LNS ran 60 s per layer (widths 4/8/16/32, one 96-core domain). It started from
quick's homogeneous plans and the default fast plan. The searched plans share one shape: 4T bulk
lanes plus at most five wide lanes, 8T (one to four) and 16T (zero or one), at most 48 wide cores,
never 32T. The three layer-2 workloads are all 4T. Under the model the searched plans are 15-20%
below the best homogeneous plan.

Lane scales. The Arm-codex rule, the median lane event / isolated load ratio of the searched
plans, gives 4T 2.56, 8T 1.32, 16T 1.28 here (Arm-codex: 1.119 / 1.065 / 1.045). Used as is, it
makes fast worse than with the Arm-codex constants (+25.2% against LNS in median, against
+17.8%). The ratio depends on what a lane holds (4T lanes carry the small, strongly
load-dilated experts), not on its width. `fit_fast_scale.py` therefore fits the scales on the
same 18 layers under the model instead (16T fixed at 1, since only ratios matter). The result is
a flat optimum at 4T 1.6 and 8T 1.15-1.6; 4T 1.60, 8T 1.45, 16T 1.00 are used.

Model-only result (`fast_check.py`, event call time, median over the 18 layers):

| plan | gap to LNS (median / max) |
| --- | --- |
| fast, C9g configuration | +10.3% / +15.2% |
| fast, Arm-codex defaults | +17.8% / +25.4% |
| production quick | +24.9% / +28.1% |

The model predicts fast 11.1% faster than production quick (range -13.1% to +3.7%). Python
planning takes 6.3 ms per layer. The configuration is written to the asset's `hot_wide` entry,
and `PlannedMoE(search_mode="hot_wide")` reads it (`_hot_wide_config`). Measured validation on
fresh layers follows.

## Measured validation on fresh layers (stage 4)

Frozen in `design.md` before plan generation. 18 real TP4 layers never used on C9g (r008/r016/r022
x layers 1/7/13/19/30/39). Four plans per layer, all with the registered window table:
production quick (`baseline`), `fast` (the C9g `hot_wide` entry), `fast_default` (Arm-codex
template and scales) and `lns` (60 s model LNS). Measured with the width-table bench (protocol v2,
bitwise gate) on node 0, two sessions, 120 s fixed cooldown; session spread median 0.21%, max
3.35%. Analysis: `analyze_e3.py`.

| comparison (measured, median over layers) | result | gate |
| --- | --- | --- |
| F1 `fast` vs production quick | +0.12% (range -3.31% to +2.66%, faster on 8 of 18) | FAIL (needs <= -5%) |
| F2 `fast` vs `lns` | +7.83% median, +12.47% max | FAIL (needs <= 2% / 5%) |
| `fast_default` vs production quick | +3.23% (+0.13% to +10.63%) | reported |
| `lns` vs production quick | **-7.63% (-10.56% to -0.06%), faster on 18 of 18** | reported |

Measured / event-predicted call time, median: production quick 1.070, fast 1.191,
fast_default 1.182, lns 1.223. The model underprices mixed plans (4T bulk with wide lanes) by
about 12% relative to homogeneous ones. It still ranks the searched plan first, but predicts its
gain at about 20% against 7.6% measured. Planning time (Python): fast 6.3 ms, production quick
12.3 ms.

Reading. The C9g event model is useful: plans searched under it are faster than production quick
on every fresh layer, 7.6% in median. The fast planner's rule (LPT on per-width scaled isolated
loads over fixed templates) does not capture that gain on C9g; it ties production quick. The
C9g configuration still beats the Arm-codex constants by about 3 points. Candidate next steps:
make the search usable on the request path (native event model plus a time-budgeted search, or
plans cached by route signature), and find the cause of the family-dependent underprediction.
