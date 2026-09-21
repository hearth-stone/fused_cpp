# New Lab baseline without wide/narrow team residuals, 2026-09-10

## Change and active scope

The user requested removing the penalties/rewards from the new model and exposing the remaining errors. The current new Lab adapter is now `planner_no_team_residual.py` (`CorePressureModel`). It inherits the bounded M1 adapter and clears both `wide_team_pressure` and `narrow_team_contention_correction` in a new in-memory calibration object. Every width's isolated/cohort multiplier and narrow correction becomes1. All resource service/pressure, spill, explicit overhead, M1 response and other calibration fields remain unchanged. Original profile files and penalized implementations remain reference artifacts.

Class M, Lab-only behavioral change. No production/default dispatch, serialized schema or native kernel is changed. Existing archived search scripts retain their old adapter to preserve replay; use the new adapter explicitly for subsequent no-team Lab runs. Rollback is selecting the preserved penalized adapter. The present diagnostic holds both high-skew plans fixed and reuses their two existing trace sessions; it does not rerun search or measure a new speedup.

## Whole-plan and lane errors

| Fixed plan | Penalized prediction (ms) | No-team prediction (ms) | Actual S1 / S2 (ms) |
| --- | ---: | ---: | ---: |
| Baseline-selected |31.139|26.558|29.301 /29.347|
| M1-adapter-selected |31.285|26.284|28.654 /28.628|

The no-team model now ranks the faster M1-adapter-selected plan ahead by0.274ms, matching the measured direction (paired reduction0.642/0.711ms), but underestimates both totals. It is not yet an accurate magnitude or critical-path model: predicted last lane is68 for the baseline-selected plan and32 for the M1-adapter-selected plan; actual is69 and mainly77/78 respectively.

Mean predicted-minus-measured lane completion over both sessions:

| Plan | Width | Penalized (ms) | No-team (ms) |
| --- | ---: | ---: | ---: |
| anchor | 1 | -1.912 | -2.397 |
| anchor | 16 | +3.481 | -1.737 |
| repaired | 1 | -2.003 | -2.396 |
| repaired | 16 | +3.421 | -1.816 |

The current CorePressureModel had already overwritten the old concurrent narrow reward in GEMM events; removing its calibration explicitly is not responsible for the worsened1T endpoint bias. Removing the16T width multiplier makes simulated wide teams finish earlier, shortening simulated contention windows for1T. This is event-model feedback, not a measured hardware change.

## Stage accounting after removal

Values below are mean per-lane cumulative stage/envelope time across two sessions, in ms. These are diagnostic accounting sums, not additive physical causes. The actual gap residual is computed per call as lane completion minus the summed gather/W13/W2 envelopes, then summarized. It includes initial/inter-task/stage gaps and synchronization/scheduling effects; it is not a direct measurement of one named overhead. Independently summarized medians need not sum exactly to the median lane endpoint.

| Plan / width | Model W13 | Actual W13 | Model W2 | Actual W2 | Model operator residual | Actual gather | Actual unassigned gap residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anchor_1T | 15.336 | 18.517 | 7.545 | 9.040 | 2.983 | 0.529 | 0.177 |
| anchor_16T | 16.899 | 17.308 | 8.462 | 8.453 | 0.000 | 0.981 | 0.343 |
| repaired_1T | 15.285 | 18.432 | 7.520 | 9.012 | 2.954 | 0.528 | 0.182 |
| repaired_16T | 16.961 | 17.404 | 8.491 | 8.501 | 0.000 | 0.988 | 0.362 |

Three concrete accounting gaps are exposed:

1. **16T kernel time is much closer without the width factor.** W13 remains low by about0.41–0.44ms per lane; W2 is approximately matched. About0.98–0.99ms gather plus0.34–0.36ms unassigned gaps have no explicit corresponding cost in this wide-team configuration. The residual1.7–1.8ms low bias is not evidence for reinstating a25% GEMM penalty.
2. **1T kernel stages remain substantially low.** W13 is short by about3.15–3.18ms and W2 by1.49ms per lane. This is considerably larger than the final2.40ms lane error.
3. **The legacy1T operator residual compensates for some kernel underprediction.** Model operator cost is2.95–2.98ms per lane, while visible gather plus gap residual is about0.71ms. These are not identical semantic categories; the comparison exposes an opaque residual accounting issue, not a proof that the entire difference is removable overhead. It should be decomposed rather than blindly deleted or redistributed into a new team multiplier.

## Which1T tasks are most wrong?

For each M group, sum predicted placed stage times across its1T experts, and compare with the mean of the two sessions' median summed actual stage times. This measures relative error in cumulative service time, not full-plan wall time. Range below spans the two fixed plans:

| M group | W13 relative error | W2 relative error |
| --- | ---: | ---: |
| M1 | -56.0% to -54.6% | -34.6% to -32.4% |
| M2-4 | -49.0% to -48.7% | -52.2% to -52.0% |
| M5-8 | -15.6% to -15.6% | -18.6% to -18.6% |
| M9-12 | -7.9% to -7.8% | -10.8% to -10.8% |
| M13+ | -7.0% to -7.0% | -5.4% to -5.4% |

The largest relative gaps are M1–4. M1 was adapted, but its physical competition mapping remains an extrapolation; M2–4 still use the old model. Concurrent traces alone cannot uniquely divide these errors into isolated kernel-cost error and competition error. Do not infer that all missing time is memory contention.

## Validation and reproduction

Two focused tests pass: all supported widths' residual factors equal1; every other calibration field and original profile file is unchanged; isolated phase predictions are preserved; a placed wide-team DAG responds to multiplier removal. Ruff and diff checks pass. The fixed-plan old-model reconstruction matches previously retained clocks. No native numerical test, new full search or new hardware benchmark is claimed: runtime data and prior bitwise/trace validation are reused unchanged.

```bash
.venv/bin/pytest -q tests/test_moe_no_team_residual.py
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_no_team_residual.py   --source tmp/planner_repaired_pressure_20260910   --output-dir tmp/no_team_residual_20260910
.venv/bin/python tmp/no_team_residual_20260910/analyze_m_groups.py
```

Use a fresh output directory when rerunning the main analysis. For a future explicit planner run, the existing `bench_full_model_search.py --model new --adapter optimizations/fused_moe_sve/benchmarks/planner_no_team_residual.py` loads this version; supply the usual input/case/output and mean selector arguments.

Artifacts, source snapshots and before/after status are retained under `tmp/no_team_residual_20260910/`. Input traces remain in the parent experiment, on CPU240–319/NUMA3,4×16T+16×1T, H4096/F512 BF16 SVE256, N tile16, full stripes, early mergeoff and two31-call sessions. Full1T stage B is8MiB W13/4MiB W2; no allocation-policy change. The next diagnosis should prioritize real-plan M1–4 stage response and explicit overhead accounting, then the remaining16T gather/gap costs. No newly fitted compensating coefficient is introduced.
