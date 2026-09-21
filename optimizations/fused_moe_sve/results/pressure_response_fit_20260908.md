# Frozen-session memory-supply response validation

## Decision

**Keep the simple linear conditional response for all six M/width combinations.**
It improves both first-session leave-one-pressure-level-out (LOPO) error and
second-session frozen prediction against a constant isolated baseline. Do not
export a planner model: the supply input is independently measured B-only time,
not a plan-visible estimate. This is historical replay of two already-inspected
sessions, not a newly collected blind test or evidence of unseen-M generalization.

Second-session errors, six nonzero reader levels per width:

| M | Predictor | MAPE | Maximum absolute time error |
|---:|---|---:|---:|
| 1 | Constant session1 isolated | 7.516% | 25.526% |
| 1 | Frozen linear response | **0.896%** | **1.871%** |
| 12 | Constant session1 isolated | .538% | 2.084% |
| 12 | Frozen linear response | **.140%** | **.690%** |

Each M aggregate covers18 cell medians (three widths ×six levels), equally
weighted. Error denominator is measured kernel elapsed time, **not the small
incremental slowdown**. In particular, .140% total-time error does not mean
M12's approximately1% contention increment is predicted to .140% relative error.

## Model and frozen information boundary

For each M/width, let T0 and D0 be session1 zero-reader medians of real W13 and
independent B-only time. For any cell with measured supply D:

```text
p = max(0, D / D0 - 1)
linear:    T_hat = T0 * (1 + a * p)
threshold: T_hat = T0 * (1 + a * max(0, p - p0))
isolated:  T_hat = T0
```

Both T0 and D0 remain frozen in session2. No session2 real-kernel time or
zero-reader reanchoring enters the predictions. Session2 D is a conditional
measured input. Its zero-reader cell is reported separately, not used to
renormalize other cells. The pipeline serializes`frozen_fit.json` before loading
session2, then reloads the serialized model for evaluation.

Training uses ratios of cell medians, y=T/T0−1, rather than the prior report's
median of round-paired ratios. This makes absolute-time predictions coherent;
small numerical differences from the previous curve's reported slowdown are
expected. Raw31-round completeness, numerical-check scope, worker/counter
coverage and PMU running ratios are revalidated by the existing raw loader.

Fit a≥0 by equal-cell relative-time least squares:
sum[(a*max(0,p−p0)−y)/(1+y)]². No intercept or upper cap on a.
For the threshold candidate, scan1000 thresholds from0 to just below the
training pressure maximum. Positive p0 requires at least one positive-pressure
training point below it and two above it. Ties prefer the smaller threshold.
Coefficients are empirical response slopes, **not compute/memory percentages**.

## Selection policy, fixed before this fit

Zero readers defines the fixed calibration anchor; it is not an independently
held-out response point. LOPO withholds each of1/2/4/8/12/16 readers in turn,
including its victim and B-only point from parameter fitting. The held-out
B-only value is used only to evaluate its prediction. Each fold searches its
threshold solely on remaining training pressures; no pressure sorting by reader
count or enforced count monotonicity is assumed.

Select threshold only if all hold:

1. Positive full-fit threshold and positive thresholds in all six folds.
2. LOPO MAPE improves by at least10% relative and0.1 percentage point absolute.
3. LOPO maximum time error does not worsen.
4. Fold threshold range≤20% of the full training maximum pressure.
5. Fold slope coefficient of variation≤25%.

Otherwise retain linear. Session2 is not used for form selection, threshold
search, parameter tuning, or the stability gate. Its results for both candidates
are diagnostic comparisons, not an invitation to choose again.

## Per-width results

All entries below are **MAPE / maximum absolute percentage time error**.

| M | T | S1 LOPO isolated | S1 LOPO linear | S1 LOPO threshold | S2 isolated | S2 frozen linear |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 13.043 /25.879 | 1.030 /1.767 | 1.030 /1.767 | 12.868 /25.526 | .960 /1.420 |
| 1 | 2 | 4.544 /11.371 | 1.410 /2.712 | 1.410 /2.712 | 4.898 /11.818 | 1.038 /1.871 |
| 1 | 4 | 4.328 /8.927 | .654 /1.387 | .654 /1.387 | 4.783 /9.963 | .689 /1.602 |
| 12 | 1 | .334 /1.069 | .193 /.501 | .105 /.208 | .466 /1.109 | .148 /.234 |
| 12 | 2 | .372 /1.344 | .185 /.567 | .227 /.567 | .414 /1.339 | .058 /.122 |
| 12 | 4 | .586 /1.542 | .226 /.376 | .266 /.585 | .735 /2.084 | .213 /.690 |

Frozen linear slopes:

| M | 1T | 2T | 4T |
|---:|---:|---:|---:|
| 1 | 1.34248 | 1.04720 | .99022 |
| 12 | .03286 | .09112 | .13476 |

For example, a10% B-only time increase predicts approximately13.42% M1/1T
slowdown but only.329% M12/1T slowdown, **within this experimental regime**.
An M1 slope above1 is permitted because B-only is a proxy, not a fraction of
kernel work. It does not imply more than100% of time is memory cost.

Threshold details:

- M1/all widths: the full and fold solutions reduce to p0=0, i.e. linear.
- M12/1T: full p0≈8.09% supply degradation, slope.05963. LOPO improves
  .1929→.1047%, a real but small **.0882 percentage-point** gain. Fold
  parameters pass stability, but the gain misses the preset.1-point floor.
  Session2 MAPE is marginally better (.1361 vs.1485%), while maximum error is
  worse (.2861 vs.2340%). Do not claim this threshold is unfit or wholly unstable;
  it was rejected for insufficient added value under the declared rule.
- M12/2T: threshold LOPO worsens; slope CV≈30.1%, normalized threshold
  span38.1%, and some folds select zero. Reject as unstable/unhelpful.
- M12/4T: full fit is linear; some fold thresholds increase LOPO error.

## Scope, evidence and limitations

No new hardware calls. Inputs are the pressure-curve experiment's two sessions:
Arm-codex-internal NUMA3, H4096/F512, W13 K4096/N1024 BF16/SVE256/Ntile16,
M1/M12×1/2/4T,0/1/2/4/8/12/16 stream readers, persistent touched output,
256MiB scrub,4-copy rotation,5 warmups+31 recorded rounds. Full provenance and
PMU limitations remain in`memory_pressure_curve_20260908.md`.

Both sessions use binary SHA256
`aec4ec8050ecb32910ec92fc40b64ea049edba5e886891a13c3e39df45c7e8b6`.
Raw input hashes and the fitted coefficients are saved with generated outputs.
The six maximum-reader session2 points lie slightly above their session1
measured pressure ranges and are explicitly flagged in the evaluation artifact;
the reported errors include them. Do not interpret this as unrestricted
extrapolation validation. The highest-level LOPO fold likewise tests endpoint
extrapolation rather than interpolation.

Two sessions and six nonzero pressure levels are limited evidence. Calibration
anchors and measured supply have uncertainty; this fit treats medians as points
and reports no parameter confidence interval or formal population error bound.
The previously inspected data informed the model family; future independent
pressure-shape/footprint or mixed-kernel validation is needed for generalization.
DDRC queue pressure is not a fitted feature here. Good conditional prediction
does not solve the plan→supply-pressure mapping or validate a physical knee.

## Reproduction and implementation

Class M/E Lab candidate only; Production, v8, planner defaults, public schemas
and pruning unchanged. New helper:`fit_pressure_response.py`; reuses existing
raw loader without modifying the hardware runner. Tests check exact linear/
hinge recovery, withheld-point isolation, frozen victim/supply anchors, and
invalid input rejection. No additional dependency or production build changes.

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/fit_pressure_response.py \
  --sessions tmp/memory_pressure_curve_20260908/session1.jsonl tmp/memory_pressure_curve_20260908/session2.jsonl \
  --output-dir tmp/pressure_response_fit_20260908
.venv/bin/python -m pytest -q tests/test_moe_pressure_response.py tests/test_moe_pressure_curve.py
```

Outputs:`tmp/pressure_response_fit_20260908/frozen_fit.json` and`evaluation.json`.
These are ignored Lab artifacts, not production calibration profiles. Full
cell predictions, held-level fits and threshold-gate decisions are retained.
Rollback is confined to this helper, tests and documentation. No commit requested.

Final review:30 focused tests passed across pressure-response, pressure-curve,
phase-supply and kernel-response tests; Ruff, manifest parsing and diff whitespace
checks passed. Hardware/production integration tests were not rerun because this
change only fits and validates existing Lab records; it makes no new runtime claim.
