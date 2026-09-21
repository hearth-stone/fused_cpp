# Small-T isolated stage estimates — 2026-09-08

## Outcome

Completed an isolated-only1/2/4T error matrix and a stage-only estimator with
separate gather,W13,W2 costs. No concurrent-workload fit, no operator residual
and no total-time cancellation are used.

The existing independent affine GEMM candidate substantially improves M1 and
fit-shape errors, but does not improve every M: M7/W13 gets worse at1/2T.
Gather still fails its historical validation gate. **The full isolated model is
not fixed or adopted.** The stage estimator refuses `T_iso`/planner export.

This work reproduces, rather than rediscovers, the GEMM coefficients from the
[earlier floor-identification experiment](workspace_floor_identification_20260907.md).
New deliverables are the small-T-only matrix, executable stage estimates,
compute/supply/startup decomposition and explicit gather feature-collision bounds.
No fresh hardware evidence is claimed.

## Data and fit boundaries

- 29 shape/width cells,87 phase rows, two hardware session medians per row.
- 1T: M1/3/4/7/8/10/16/24/62/714/1341.
- 2T and4T: M1/3/4/7/8/16/24/62/1341.
- Fit23 shape/width cells using session1 only. M1 is a retrospective guard;
  M7 is historical validation. Neither enters either session's fit.
- Session2 is evaluated using session1 coefficients. A separate session2 refit
  and leave-one-M-out fits are only parameter-stability diagnostics.
- 17 source JSON hashes match the validated parent artifact, including compact
  isolated samples, sessions, frontiers, prior isolated summary and frozen v8.
  These inputs were previously checked for target-before-background execution
  and complete numerical correctness. Failed initial grid data are excluded.

The experiments used the fixed pretouched workspace,4-copy rotation and216 MiB
scrub protocol on Arm-codex NUMA3. They are isolated target intervals extracted
from executable real-route plans, not a new phase-only synthetic harness.
Different groups/sessions have different allocations; earlier points also have
placement differences. These limitations remain, especially for gather.

## Estimator: no operator residual

For each width and GEMM stage:

```text
prediction = fitted_startup + effective_stage_scale * frozen_physical_stage
```

Each physical phase is decomposed as:

```text
compute_bound
  + max(max(L2, LLC, DRAM endpoint bounds) - compute_bound, 0)
  + epilogue_bound
  + existing_stage_setup
```

The sums are checked against the exact frozen physical stage. The additional
fitted startup is reported separately, not folded into bandwidth demand.
The A/B payload,line,private-delivery/refill ledger is attached diagnostically;
actual A/B-address-specific refills remain unknown. Effective scale is not
presented as an independently identified compute rate or A/B bandwidth.

Importantly, the fit continues to use frozen v8 physical times. The newer endpoint
diagnostic differs at M16/4T W13 (0.422814 versus0.420740 ms); an identity check
caught this when trying to substitute the basis. It is now kept as a separate
diagnostic field, not silently used to refit old coefficients.

Gather is independently estimated as `fixed + row_service * ceil(M/T)`.
The old operator residual is neither a gather reference nor a fitting target.
No valid stage-specific v8 gather baseline is available in this dataset, so its
baseline-error field is null, not zero.

## GEMM result

MAPE below pools1/2/4T and both session evaluations, weighting each shape/stage
observation equally. Fit-shape session2 is repetition, not unseen-shape holdout.

| Stage and role | v8 MAPE | Stage candidate MAPE |
|---|---:|---:|
| W13 fit shapes, s1+s2 | 10.43% | 4.08% |
| W2 fit shapes, s1+s2 | 9.87% | 4.01% |
| W13 M1 guards | 19.26% | 3.28% |
| W2 M1 guards | 17.20% | 3.72% |
| W13 M7 validation | 5.76% | **10.13%** |
| W2 M7 validation | 7.47% | 7.47% |

M1 stage predictions and hardware medians, in ms:

| Width | W13 prediction | W13 hardware s1/s2 | W2 prediction | W2 hardware s1/s2 |
|---:|---:|---|---:|---|
| 1 | 0.29474 | 0.31250 /0.32024 | 0.15015 | 0.15102 /0.15313 |
| 2 | 0.25346 | 0.25050 /0.25381 | 0.12702 | 0.12216 /0.12207 |
| 4 | 0.21076 | 0.20534 /0.20652 | 0.10822 | 0.10205 /0.10239 |

All measured M1 GEMM guards improve over v8. All M7 GEMM errors remain below
the previously declared20% threshold, but that threshold must not hide regressions:
M7/W13 at2T changes from about1.1–1.3% error to16.5–16.7%, or63.5–64.3 us.
At1T it changes from below1% to9.5–11.5%, or72.3–85.8 us. The globally affine
correction improves startup-sensitive cells at the expense of some already-good
exact-M cells. Passing the historical threshold is not general model convergence.

All87 rows have signed error in both microseconds and percent. This matters:
large1T stages can have low relative error but hundreds of microseconds of
absolute error, while a gather error of only several microseconds can be a large
percentage. No full-plan MAPE or ranking result is computed here.

## Startup versus service is still not fully identifiable

Session1 GEMM coefficients:

| Width | W13 startup us / scale | W2 startup us / scale |
|---:|---|---|
| 1 | 37.98 /1.05459 | 25.03 /1.02776 |
| 2 | 57.23 /1.02946 | 30.74 /1.01029 |
| 4 | 7.47 /1.10120 | 8.18 /1.08384 |

At1T, session2 refits W13 startup to1.70 us and W2 startup to10.08 us, with
compensating scale changes. For example, measured M3/W13 shifts449.79→408.32 us
between sessions, whereas large-M times are much more stable in relative terms.
This prevents interpreting the1T intercept as a measured universal startup cost.
The2T coefficients are more repeat-stable, but still not separately measured
hardware service components. No additional A/B/compute rates were fitted from
the same total stage observation.

The remaining GEMM work should distinguish exact-M kernel family efficiency
from startup/state variation, rather than adding another global intercept or
silently retuning on M7. Additional exact-M evidence would be a separate experiment.

## Gather: a point model cannot explain all existing observations

M7 gather predictions, in microseconds:

| Width | Prediction | Hardware s1/s2 | Absolute error s1/s2 |
|---:|---:|---|---|
| 1 | 19.70 | 12.27 /10.71 | 7.43 /8.99 |
| 2 | 31.38 | 19.32 /22.31 | 12.06 /9.07 |
| 4 | 42.66 | 38.51 /46.27 | 4.15 /3.61 |

1T/2T fail the historical20% gate in both sessions. M1 gather also remains poor:
at1T the estimate is2.81 us versus7.71/7.60 us. No compensation is moved into
W13/W2 to make the sum look correct.

There are concrete identifiability limits:

1. Same M1/2T gather has medians12.97 and29.22 us. For any fixed prediction, the
   best possible maximum relative error against these two observed medians is
   **38.52%**, achieved at17.97 us. Merely changing slope/intercept cannot make
   both observations pass20%.
2. M7 and M8 at2T have the same `ceil(M/T)=4`, but gather observations range
   19.32–34.45 us. Any function of that feature alone has at least28.14% maximum
   error against these observations. It needs another discriminating input or
   an uncertainty/state treatment; a more flexible fit of the same feature is
   insufficient.

The bound is `(max-min)/(max+min)`, with minimax predictor `2*min*max/(min+max)`.
It concerns fitting these observed medians, not a statistical lower bound on
future hardware predictability. Measurement noise, worker arrival/synchronization,
input-route locality and cross-session allocation/state are possible contributors,
not proven causes. The current metric is a multi-worker gather stage envelope,
not pure row-copy service time.

## Artifacts, gates and next scope

`SmallTIsolatedEstimator.predict_stage()` provides stage predictions with explicit
components. `T_iso()` deliberately raises: the complete gather+GEMM candidate
is not eligible. The JSON stage artifact likewise forbids total/planner export
and concurrency use. Widths outside1/2/4T and routes outside1..1341 are rejected;
unmeasured M within that range are predictions, not validated claims.

The next useful separation is:

- GEMM: exact-M-specific residuals and startup repeatability, keeping M1 and
  M7 evaluations visible; no operator residual and no new contention term.
- Gather: useful work versus worker-arrival/envelope overhead and input state.
  Resolve the observational mismatch before claiming an accurate point formula.

No concurrent workload, wide/narrow refit, LNS/VND or hardware rerun was performed.
The original M7/M1 roles and gates are unchanged. This is a limited isolated-stage
result, not adoption of a new planner cost model.

## Reproduction and validation

Primary class M, Lab-only stage estimates and analysis. Changes are new
`small_t_isolated.py`, its focused tests, manifest/math notes and this report.
No production model/calibration/default/API/native changes. Rollback is limited
to these Lab files and documentation; existing user edits remain untouched.

```sh
PYTHONPATH=.:src .venv/bin/python optimizations/fused_moe_sve/benchmarks/small_t_isolated.py \
  --data tmp/workspace_floor_grid_20260907/corrected/analysis.json \
  --calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --output-dir tmp/small_t_isolated_20260908/validated
PYTHONPATH=.:src .venv/bin/python -m pytest -q \
  tests/test_moe_small_t_isolated.py tests/test_moe_workspace_phase_reaccount.py
```

Use a fresh output directory on replay. Retained local artifacts:
`tmp/small_t_isolated_20260908/validated/report.json` and `stage_candidate.json`.
The report contains the complete matrix, per-phase bounds/ledger, coefficients,
repeat/leave-one-M-out stability, gather lower bounds and source hashes.
The earlier top-level report is retained as the version before adding the
same-feature collision table; it has the same estimates.

Underlying hardware: Arm-codex-internal NUMA3 CPUs240–319, H4096/F512,
BF16/SVE256 N tile16, FP32 route output, fixed2048-token pretouched workspace,
4-copy rotation,216 MiB scrub,31 measured samples/session, full owner stripes.
Detailed task placement and group provenance are in the parent report. Calibration
SHA256 remains `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`.
Repository base remains `c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5` plus existing
dirty work and these Lab changes.

Validation:14 focused tests passed, including7 new fit-boundary/accounting tests;
17 JSON input hashes and all frozen GEMM baselines/previous affine coefficients
match. Stage component conservation passes. No total residual is used. No
production/native or hardware tests were rerun because no runtime behavior
changed. No commit made.
