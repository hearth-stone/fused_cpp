# 16T overprediction: width-factor counterfactual, 2026-09-10

Read-only model diagnostic on the two existing high-skew selected plans. Reuse the same two hardware sessions; only newly constructed in-memory model instances are altered. No calibration file, planner baseline or production source is changed, and no new hardware run is performed.

## Finding

The retained16T isolated-width factor1.25105 is the largest tested contributor to the wide-lane overprediction. The configured full-cohort factor1.47042 is not the active extra penalty in the current CorePressureModel GEMM path: it is overwritten with a resource response multiplied only by isolated_scale(width). Setting the16T full-cohort scale down to its isolated scale therefore changes no prediction in either plan. The relevant code is `replay_core_pressure_moe.py:61–66`.

The active formula for a GEMM phase is `max(1, resource_duration/base_ns * isolated_width_scale)`, with the separate measured-core local response injected into LLC and DRAM resource scales. The name isolated_dilation must not be interpreted as proof of a universal physical16T cost. Its provenance is an empirical residual calibration, with original fit statistic `median(measured_median_ns / placement_event_predicted_ns)` on prior workload/model conditions.

## Fixed-plan counterfactuals

Signed error is predicted lane finish minus measured lane finish, averaged over the four16T lanes and two sessions. Each counterfactual reruns the complete phase event simulation, so effects include changed overlap and are not an additive causal partition.

| Intervention | Baseline-selected plan wide mean error (ms) | Repaired-selected plan wide mean error (ms) |
| --- | ---: | ---: |
| Current | +3.481 | +3.421 |
| Remove old16T cohort increment | +3.481 | +3.421 |
| Set16T isolated factor to1 | -1.737 | -1.816 |
| Remove16T local-core response only | +3.291 | +3.231 |
| Remove16T isolated factor and local-core response | -1.737 | -1.816 |

Removing the isolated factor shifts mean16T predicted completion down5.218ms in the baseline plan and5.237ms in the repaired plan, exceeding the observed3.4–3.5ms high bias. The prediction then underestimates by1.7–1.8ms. Thus the factor is a major source of overprediction, but zeroing it is not an established calibration fix. Removing only the16T local-core pressure contribution reduces mean high bias by about0.19ms in either plan, much less than the width-factor intervention.

The no-local intervention sets the16T local LLC scale to1 and drops its local DRAM max contribution, while retaining global resource pressure and spill. It is not equivalent to disabling all memory competition. Other widths and the M1 adapter remain unchanged. The width-only intervention changes only the16T isolated_dilation entry, leaving other widths and the full-cohort entry untouched; for singleton events its occupied-peer fraction is zero.

Lane32 illustrates the scale:

| Plan | Current prediction (ms) | No16T isolated factor (ms) | Actual S1 / S2 (ms) |
| --- | ---: | ---: | ---: |
| Baseline-selected |31.139|26.219|28.255 /27.774|
| Repaired-selected |31.285|26.284|28.476 /27.972|

Current overall makespans31.139/31.285ms become26.558/26.284ms with no16T isolated factor. These are model counterfactuals, not actual speedups; they do not repair the remaining1T underprediction and should not be judged solely by whether the global predicted winner changes.

## Historical consistency and limits

The [2026-09-07 phase reaccount ablation](workspace_phase_reaccount_ablation_20260907.md) already identified wide-team residual penalties as a principal overprediction mechanism in other traced cases. That earlier experiment disabled different combinations under an earlier model, so its numerical reductions are not substituted for this current ablation.

This confirms sensitivity to the retained width factor on these two plans; it does not prove that hardware has no wide-team overhead, that all16T workloads need the same lower factor, or that the factor is physically double-counting a particular resource. Recalibration would require a separate fit/holdout decision. The current calibration and planner remain unchanged.

## Reproduction

Run `.venv/bin/python tmp/planner_repaired_pressure_20260910/diagnose_wide_penalty.py` from the repository root. Inputs and machine/protocol are those of [the planner comparison](planner_repaired_pressure_20260910.md): high-skew route, CPU240–319/NUMA3,4×16T+16×1T, full stripes, BF16 SVE256, H4096/F512, early mergeoff and two31-sample traced sessions. Output `wide_penalty_diagnostic.json` contains lane endpoints, overall makespan and mean/absolute wide-lane errors for all five interventions. Current-model predictions match the prior reconstructed clocks; the full-cohort-only counterfactual matches them exactly. No new implementation tests or performance measurements are claimed for this diagnostic.
