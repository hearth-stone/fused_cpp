# One-click calibration on C9g: from 21-31% to research accuracy in about a minute

## Status

`enable_moe_planner_quick` now produces a calibration as accurate as the research procedure on
Amazon C9g, in 49 s (TP4) and 70 s (TP2) per expert shape. Two changes did it: the quick
service probe runs three times and takes the per-point median, and a short shape-bound
isolated-expert measurement fits the operator overheads the machine probe cannot see.

## What the old one-click path got wrong

`calibrate_moe_planner_quick` unchanged, three idle runs on node 0 (CPUs 0-95), scored against
the research calibration (`cal_median`, three median service probes plus a training profile
and per-width overhead fit) on 434 held-out isolated points per shape - the dense sweep, the
pairs round and the triplet round, training routes excluded:

| | research | quick r1 | quick r2 | quick r3 |
| --- | --- | --- | --- | --- |
| TP4 mean absolute error | 7.66% | 28.73% | 20.69% | 31.43% |
| TP2 mean absolute error | 10.55% | 23.50% | 17.20% | 30.18% |

Two failures, both visible per width. With no operator overheads every width is
under-predicted and the widest the most (TP4 32/48/96T: -50%, -56%, -75%), because the fixed
cost around the GEMMs is a larger share of a wide expert. And one 5-second probe is a lottery:
the same idle machine gave a one-thread error of -16%, -17% and +51%.

## The change

- `calibrate_moe_planner_quick(..., service_repeats=3)` runs the probe three times (seeds offset
  by 1000) and combines them with `median_service_probe`, the function the research procedure
  already uses. Quick calibration version 2.
- `train_quick_operator_overheads(payload, hidden_size, intermediate_size, ...)` measures
  isolated experts of one shape the way the research training profile does (8 experts back to
  back on one team, per-expert time = call / 8, 3 warmup + 11 runs), at routes
  1/4/12/48/192/2040 on every supported width. It fits one common
  `(expert_fixed, route, stage_scale)` residual on 12/192/2040 with `fit_operator_residuals`,
  then one `(expert_fixed, route)` pair per width with the new `fit_width_overheads` - the lab
  `fit_by_width.py`, moved into `build_analytic_calibration.py` and checked to reproduce its
  entries exactly on both C9g shapes.
- `enable_moe_planner_quick(..., train_overheads=True)` runs the machine calibration, then the
  training, and writes and installs the trained file only after both succeed; it refuses an
  existing output before measuring anything. `train_overheads=False` keeps the old behaviour.

The training routes are a subset of the research training routes, so the held-out set is
untouched by either calibration.

## Result

Three independent end-to-end runs per shape, `wait_idle` before each, same scorer. The target,
frozen before the runs: every run within 2 points of the research calibration on both shapes,
and the three runs within 3 points of each other.

| | research | v2 r1 | v2 r2 | v2 r3 |
| --- | --- | --- | --- | --- |
| TP4 mean absolute error | 7.66% | 7.85% | 7.83% | 7.71% |
| TP2 mean absolute error | 10.55% | 10.55% | 8.86% | 8.72% |
| elapsed, TP4 (machine + training) | | 15.5 + 33.4 s | 15.5 + 33.4 s | 15.5 + 33.4 s |
| elapsed, TP2 | | 15.6 + 54.6 s | 15.6 + 54.7 s | 15.5 + 54.8 s |

Passed: the widest gap to research is +0.19 points, the widest run-to-run spread 1.83 points.
Per width the two calibrations now carry the same errors, including the multi-panel bias at wide
widths (`c9g_b_retention_20260923.md`), which is a model defect rather than a calibration one.

## What this does not show

- Isolated accuracy only. Plan quality from a v2 calibration against the research one - the
  planner's width and window choices on real layers - is not measured here.
- One machine. Arm-codex performance runs are parked; the change is not validated there.
- The trained file is bound to its expert shape: a new parallel strategy needs a new training
  run (about a minute), not a new machine probe - but `enable_moe_planner_quick` repeats both.
- Window tables stay per machine and are registered separately; an unregistered machine keeps
  the full stripe.

Lab: `tmp/c9g_quick_cal_20260923/` (old path), `tmp/c9g_quick_cal_v2_20260923/` (this change).

## Plan selection on real layers (added 2026-09-23)

The isolated result above left plan quality open. Measured: 18 real layers per shape (requests
008/016/022, layers 2/9/15/25/36/40, none used by an earlier C9g run), one quick planner
(`plan_quick`, widths 4/8/16/32, CPUs 0-95) with the shape's registered C9g window table, and
three calibrations as the only difference - `research`, `v2` (the first end-to-end one-click run,
chosen before any plan was generated) and `v1` (the old untrained single-probe one-click file,
a control). Every variant's Plan V2 differed from `research`'s: the widths agree on 34 of 36
layers (v1 picks 16T over 8T on two TP4 layers), but predicted times differ and so does the LPT
assignment of experts to lanes. The window validation's `bench.py`, unchanged, two sessions;
`research` repeats between sessions at a median 0.12% (TP4) and 0.06% (TP2).

| vs `research`, mean of two sessions | median | range | faster | > 2% slower |
| --- | --- | --- | --- | --- |
| TP4, `v2` | -0.04% | -0.25% to +0.24% | 10/18 | 0 |
| TP2, `v2` | -0.12% | -0.48% to +0.17% | 12/18 | 0 |
| TP4, `v1` (control) | +0.20% | -0.18% to +0.99% | 2/18 | 0 |
| TP2, `v1` (control) | -0.02% | -0.31% to +0.26% | 10/18 | 0 |

Frozen rule (median <= +0.5% and no layer > 2% slower): `v2` is **equivalent** to the research
calibration on both shapes.

So is the old one-click file, which is the more informative result. A calibration 20-31% off on
isolated experts picks plans within 0.2% of the research calibration's, because the quick
planner only chooses one homogeneous width among four and assigns experts by LPT: errors that
move every width the same way do not reorder them. Plan choice by this planner is robust to the
calibration; the v2 accuracy buys about 0.2% on TP4 here and nothing measurable on TP2. Where
absolute times drive decisions - full search, the fast planner, cross-rank or shared-expert
balance - the accuracy should matter more; that is not measured.

Lab: `tmp/c9g_quick_plans_20260923/` (design, gen, bench, score).
