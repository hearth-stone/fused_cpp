# Workspace phase reaccounting and placed-model ablation

## Decision

The Lab candidate now accounts for gather, W13 and W2 independently and has no
legacy operator residual. The fit improves measured training-shape phase error,
but **fails the withheld M1/1T regression guard**. It is rejected for adoption,
must not replace frozen v8, and is not usable for search or pruning.

The frozen-v8 ablation identifies `wide_team_pressure` as the dominant modeled
source of excessive large-M16T GEMM dilation in the selected counterexamples.
The narrow-team correction suppresses some real small-M1T penalties; removing
DRAM sharing makes those cases substantially worse. Do not globally remove
contention or deploy the best-looking single ablation.

## Change boundary and data

Class M/E, Lab only. No production/native/model formula or profile schema edits.
New script: `benchmarks/reaccount_workspace_phase_model.py`. Existing
`GatherPressureCalibration`, `StagePhaseCalibration`, and the floor/scale fitter
are reused. `MATHEMATICAL_MODEL.md` documents candidate semantics and unchanged
pruning. Frozen v8 input bytes are unchanged.

Input `tmp/workspace_isolated_width_20260907/summary.json` supplies only isolated
session1 phase observations to the fit. Session2 is a repeat at the same shapes,
not independent shape validation. All M1 rows are withheld from fitting, including
16T. Neither full-workload times nor whole-expert total times enter the fit.
The earlier locked route-sweep artifacts are not opened by this experiment.

Hardware evidence is the prior workspace audit: Arm-codex-internal, NUMA3,
CPU240–319, E256/H4096/F512,2048tokens,TopK6,BF16,actual Ntile16,merge on,
full owner stripes `(t,0,0,1,1)`; W13/W2 full weight bytes8MiB/4MiB per expert.
Fixed pre-touched workspace,4-copy rotation,216MiB scrub,5 warmups,31 paired
rounds,2 sessions. Trace-on diagnostics only; no new hardware run in this task.
Original source/build/profile identities and commands remain in
`workspace_isolated_width_20260907.md` and `workspace_phase_timeline_20260907.md`.

The fitted candidate only declares widths1/8/16. Width8 has one M714 observation,
so it cannot establish a floor or cross-M behavior. Candidate replay explicitly
skips median elite2ab because it contains2T/4T tasks; v8 ablations still replay it.
Full-plan replay at other unmeasured M values is exploratory, especially below
the minimum fitted M10. There is no claimed all-route calibration coverage.

## Independent phase accounting

Gather uses nonnegative weighted affine least squares:
`G(M,t) = fixed[t] + row[t] * ceil(M/t)`; weights use a20us normalization floor.
Each GEMM stage independently fits the existing
`max(stage_floor[t], stage_scale[t] * physical_stage_time)` by normalized absolute
error. Expert fixed, route and by-width operator overheads are zeroed before
candidate prediction, so no total residual gets redistributed back into GEMMs.
Architectural gather bytes use the existing default multiplier1; that traffic
assumption is not newly calibrated by phase durations.

| Stage | v8 error, fit/repeat | Candidate error, fit/repeat |
| --- | --- | --- |
| Gather | no explicit phase | 2.50% /5.78% |
| W13 | 10.10% /10.52% | 2.10% /2.61% |
| W2 | 6.61% /6.79% | 1.39% /1.59% |

These are unweighted MAPE across the nine fitted shape/width cells, excluding
M1. They are fitting/repetition diagnostics, not independent model accuracy.

### Withheld M1 prevents an unsafe promotion

| Width/stage | Hardware s1/s2 ms | Candidate ms | Candidate error s1/s2 |
| --- | --- | ---: | --- |
| 1T gather | 0.00771/0.00760 | 0.00297 | -61.5% /-60.9% |
| 1T W13 | 0.31250/0.32024 | 1.04047 | +233.0% /+224.9% |
| 1T W2 | 0.15102/0.15313 | 0.53070 | +251.4% /+246.6% |
| 16T W13 | 0.07288/0.07350 | 0.08204 | +12.6% /+11.6% |
| 16T W2 | 0.03296/0.03345 | 0.04084 | +23.9% /+22.1% |

The1T fit has no M1 training point and chooses a floor near its smallest training
case M10. Extrapolating that floor to M1 is invalid. The automatic guard compares
withheld GEMM absolute percentage errors to v8 separately by stage/session and
fails; no threshold or fitting exclusion was changed after observing failure.
This is a known-shape retrospective guard, not an untouched prospective holdout.
Candidate full-plan ablations are consequently diagnostic stress tests, not
usable repaired plan scores. Low fit MAPE does not override this rejection.

## One-at-a-time ablations

Twelve variants per model domain: baseline; no wide; no narrow; no both; wide
isolated floor removed (B=1,full S unchanged); wide peer increment removed
(S=B); no shared spill increment; and separate no GEMM/L2/LLC/DRAM/epilogue
sharing. All event timelines are recomputed. There are156 plan/variant results:
7 v8 plans and6 supported candidate plans,each×12 variants.

Resource-sharing interventions remove only the selected event resource scale,
not isolated service cost or demand. Spill intervention fixes each phase to its
own isolated spill fraction. They use scoped monkeypatches in the isolated,
single-threaded Lab process, restored even on exceptions. Never load this
intervention context into a concurrent production process. Existing disabled
domain-injection calibration is not refitted. One-at-a-time effects are
conditional and non-additive because active sets and bottlenecks change.

### Large16T targets: wide-team term dominates

Frozen-v8 predicted target stage duration in ms:

| Variant | Median elite M714 W13/W2 | High-skew anchor M1341 W13/W2 |
| --- | --- | --- |
| Baseline | 6.111 /2.822 | 11.746 /5.203 |
| No wide-team pressure | **4.362 /2.167** | **8.265 /4.079** |
| No narrow correction | 6.121 /2.829 | 11.820 /5.201 |
| No wide or narrow | 4.365 /2.167 | 8.270 /4.089 |
| No wide isolated floor | 5.844 /2.456 | 11.330 /4.367 |
| No wide peer increment | 5.420 /2.699 | 10.312 /5.050 |
| No shared spill increment | 5.948 /2.729 | 11.756 /5.564 |
| No GEMM sharing | 6.061 /2.814 | 11.654 /5.183 |
| No DRAM sharing | 5.915 /2.729 | 11.736 /5.700 |
| No LLC/L2/epilogue sharing (each separately) | 6.111 /2.822 | 11.746 /5.203 |
| Hardware full s1 | 4.563 /2.194 | 8.446 /4.097 |
| Hardware full s2 | 4.560 /2.195 | 8.439 /4.094 |

Wide16T calibration has isolated dilation1.25105 and full-cohort dilation1.47042.
It multiplies the phase resource result even when the long steady GEMM has little
measured full-vs-isolated increment. Removing it reduces target W13 by1.750ms
and3.481ms in these two cases. This is the principal modeled mechanism, not
evidence that actual hardware has no memory competition.

Whole-plan model time drops35.453→27.576ms for median elite and39.511→27.624ms
for high-skew anchor with no-wide. This is not a speedup measurement or a new
winner: other phase errors and wrong critical lanes remain. In particular,
median no-wide predicts expert41 terminal instead of hardware expert167.

### Small1T targets: guard against discarding real penalties

High-skew anchor M1 frozen predicted W13/W2:

- baseline0.5825/0.2877ms;
- no narrow correction0.7538/0.3544ms;
- no wide0.4614/0.2639ms;
- no DRAM sharing0.2626/0.1289ms;
- hardware0.8530/0.4380 and0.8365/0.4358ms.

Removing the narrow correction moves this case closer to hardware, while
removing DRAM sharing removes necessary delay. Removing wide also changes peer
timelines and makes this target prediction worse despite helping wide targets.
Therefore neither global contention removal nor selection by wide-only error
is justified.

For M62/1T in full_reference, baseline W13/W2 is5.8611/2.9305ms, no-narrow is
5.9310/2.9628ms, but hardware is6.9474/4.1312 and6.9430/4.1555ms. The two old
team corrections do not explain the remaining stage-specific context cost.
Keep this counterexample, rather than fitting the remaining total error away.

## Next bounded work

The first phase candidate is not promotable. Resolve small-M stage-floor
identifiability with independent small-M isolated evidence or a separately
justified stage form; preserve M1 as a withheld sentinel. Add2T/4T evidence
before rescoring the mixed-width median elite with a corrected calibration.
Only then consider replacing/removing wide and narrow corrections, with both
large16T and small1T phase-increment gates and independent plan-level validation.
No smoothing generator, search-space expansion or hard pruning change is needed.

## Reproduction and validation

Authoritative output: `tmp/workspace_phase_reaccount_20260907/validated/`:
`candidate_calibration.json` (rejected Lab candidate),`report.json` with all
phase rows,ablation targets,source hashes,skipped-width records and failed guard.
The first exploratory run in the parent directory is retained,not overwritten.

```bash
PYTHONPATH=.:src .venv/bin/python optimizations/fused_moe_sve/benchmarks/reaccount_workspace_phase_model.py \
  --calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --isolated-summary tmp/workspace_isolated_width_20260907/summary.json \
  --frontier-dir tmp/workspace_phase_timeline_20260907 \
  --output-dir tmp/workspace_phase_reaccount_20260907/reproduction
```

Focused tests:5 passed in `tests/test_moe_workspace_phase_reaccount.py`, covering
nonnegative gather fit,M1 exclusion,zero operator/unsupported widths,scoped
intervention restoration,isolated invariance and the withheld regression guard.

Broader command:
`PYTHONPATH=.:src .venv/bin/python -m pytest -q tests/test_moe_workspace_phase_reaccount.py tests/test_moe_analytic_model.py`
initially yielded107 passed/3 failed with the then4 new tests. All three failures
are existing runtime tests at `AnalyticPolicy.identity_key`, which references
missing `wide_team_pressure`. Running those original tests alone,without this
new module, reproduces3/3 failures. Production `analytic_model.py` was not edited
by this task. This is a separate runtime integration blocker; do not claim the
full suite passes or deploy this candidate. No unrelated user changes reverted.
Ruff/diff checks pass; no native test or new hardware run was needed for the
offline diagnostic. No commit,calibration deployment or production adoption.
