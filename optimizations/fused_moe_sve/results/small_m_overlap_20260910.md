# Effective compute/memory overlap prototype, 2026-09-10

## Result

A callable Lab stage model now represents per-kernel effective core/memory service efficiencies and bounded overlap. On204 retrospective holdout condition medians it has2.035% MAPE and13.565us MAE, versus2.533%/17.843us for a recalibrated full-overlap comparator and3.786%/23.739us for an empirical serial-share comparator.

Mixed backgrounds remain a distinct limitation: all44 mixed-condition predictions are low, with5.319% MAPE and40.731us MAE. Serial-share is better on that slice at4.255% MAPE. The prototype is retained for subsequent experiments; it does not replace the user's active model/planner baseline and does not identify physical compute/memory time percentages.

## Form and constraints

The existing production analytical phase already uses `max(gemm_core, transfer)` plus its configured fixed/epilogue terms. This prototype extends the service components and overlap, rather than introducing overlap for the first time.

For a foreground stage and M, let `C_ref` be the frozen v8 core-service proxy and `D_ref` the maximum calibrated L2/LLC/DRAM endpoint-time proxy. Define:

```text
C = C_ref / eta_core(M, stage)
D0 = D_ref / eta_memory(M, stage)
D = D0 * g(stage, small_peers, large_peers)
T = C + D - rho(M, stage) * min(C, D)
```

Constraints are `0 < eta_core, eta_memory <= 1`, `0 <= rho <= 1`, and `T(g=1)=T0`, where T0 is the calibration cell's no-background median. Equivalently, search `C_ref <= C <= T0`, `D_ref <= D0 <= T0`, `C+D0 >= T0`; baseline equality fixes `rho=(C+D0-T0)/min(C,D0)`. No extra free startup constant is added. The selected1T calibration phases have zero explicit fixed/epilogue terms; the effective fitted services can still absorb unseparated stage costs.

`rho=1` is full overlap, `rho=0` is serial service. Resource occupancies `C/T` and `D/T` can both be nonzero at once; their sum need not be1. Subtracting the overlap contribution restores the total. These are model accounting quantities, not measured hardware counters.

Important physical limit: `C_ref` comes from the M12-derived hot core path, which already includes frontend and L1 delivery. `D_ref` is an endpoint-to-register service proxy. Neither is an independently isolated pure resource time for each M. Enforcing proxy bounds does not prove physical identifiability of eta or rho.

### Background composition

Use a shared effective small-background weight and stage-specific response curve:

```text
x = (large_peers + w_small * small_peers) / 38
g_stage(x) = 1 + a_stage*x + b_stage*x*x
```

Fit only session1 M1 foreground responses under four pure conditions:16/38 peers, each all-M2 or all-M120. All coefficients are nonnegative; `w_small` is searched on[1,3]. Fitted values:

| Parameter | Value |
|---|---:|
| Shared w_small |1.365|
| W13 a / b |0.740270 /0.584573|
| W2 a / b |0.666727 /0.567293|

M1/2 use the reference normalization `C=C_ref,D0=T0,rho=1`, so their response defines g. This is an assumption resolving the pressure/component scale ambiguity, not a measured proof that M1 has100% overlap. The fitted small weight is not an independently measured request-rate ratio.

### Per-M effective parameters

Odd M1/3/5/7/9/11 are calibration anchors. Even M2/4/6/8/10 share the previous odd anchor because their frozen core/endpoint proxy signatures are checked to be identical. Their measured values do not enter fitting or baseline anchoring. M11 is a single-member group; M12 is outside the API domain.

Illustrative W13 partial-overlap parameters; all are effective proxy-relative quantities:

| Foreground M | C, us | D0, us | rho | eta_core | eta_memory |
|---|---:|---:|---:|---:|---:|
|1–2|189.07|306.06|1.000, imposed normalization|1.000|0.796|
|3–4|378.13|305.83|0.854|1.000|0.796|
|5–6|611.97|319.44|0.934|0.927|0.762|
|7–8|756.26|339.32|0.942|1.000|0.718|
|9–10|1062.49|373.05|0.918|0.890|0.653|
|11|1208.39|419.19|0.940|0.939|0.581|

This creates a changing bottleneck: M1's memory service dominates its baseline, while M11 can hide much of its memory service until g grows enough. W2 has its own component and overlap table in `model.json`; it is not forced to reuse W13 coefficients.

## Fit and retrospective validation

Source: the complete [controlled-pressure experiment](small_m_controlled_pressure_20260910.md), using each standalone synthetic victim's own no-background timing. Both raw session hashes were verified. Do not substitute the real-expert isolated grid's absolute times.

Train on60 condition medians: session1, odd M, no background and the four pure background environments. Hold out all mixed environments, every even M and all session2 points. This is a retrospective split of an already inspected experiment, not a prospective blind holdout. No new hardware was collected.

The fit uses deterministic NumPy searches, without new dependencies. Background fitting scans2001 weights with nonnegative linear/quadratic least squares. Foreground full-overlap uses1001 memory-component points. Partial overlap searches401×401 component grids within the declared proxy/baseline bounds. Objective is squared relative time error on the four pure conditions. Parameters are written to `model.json` before evaluation results are computed. A second execution reproduces all parameters and metrics exactly.

Comparators share the same pressure mapping, odd-M baseline anchors and group assignment:

- Uniform pressure: multiply every M's T0 by the M1-normalized g; a sensitivity sanity check, not the current production model.
- Serial share: fit a single memory share with `C+D0=T0,rho=0`. This empirical comparator deliberately does not enforce the physical-proxy lower bounds.
- Full overlap: `C=T0,rho=1`, fit D0 within proxy bounds. **It is also recalibrated**; its error is not the current production/planner model's error.
- Partial overlap: fit bounded C/D0 and derive rho from the isolated baseline equality.

| Evaluation subset | N | Serial-share MAPE | Full-overlap MAPE | Partial-overlap MAPE | Partial MAE |
|---|---:|---:|---:|---:|---:|
| Training |60|3.323%|1.483%|0.887%|5.071us|
| All held out |204|3.786%|2.533%|2.035%|13.565us|
| Session2 |132|3.699%|2.296%|1.787%|11.655us|
| Even M, both sessions |120|3.898%|2.377%|1.843%|11.538us|
| Mixed, both sessions |44|4.255%|5.370%|5.319%|40.731us|

Subsets overlap and must not be summed. Partial-overlap maximum held-out absolute percentage error is7.409%; full-overlap10.370%; serial-share14.538%. Mixed partial-overlap bias is-40.731us because all44 errors are negative. Whole-stage time errors are reported, not full-forward/planner latency or isolated resource measurements.

## Mixed pressure and parameter-identification limits

The simple additive mixed demand underestimates the M1-derived service scale. At19 small plus19 large peers:

| Stage | Predicted g | Observed M1 time ratio S1 / S2 |
|---|---:|---:|
|W13|2.6928|2.8563 /2.8475|
|W2|2.5817|2.7705 /2.7638|

As a diagnostic only, substituting each held-out session's measured mixed-M1 scale while keeping all fitted components frozen reduces mixed MAPE from5.319% to1.791%, with5.709% maximum error. This uses a held-out measurement as an oracle and is **not a deployable prediction result**. It identifies pressure composition as a substantial limitation, while the remaining error shows that correcting that one scale does not close every foreground response.

The artifact also records near-optimal parameter ranges over grid solutions within `best_relative_MSE + 0.005^2`. These are sensitivity envelopes, not confidence intervals. Some fitted core efficiencies sit on the proxy bound. Even narrow envelopes under a fixed reference normalization do not identify true physical resource shares; the reference normalization and service proxies would require independent interventions to validate.

Supported scope: M1–11,1T foreground, W13/W2, H4096/F512, standalone SVE256/N16 harness, M2/M1201T W13 backgrounds totaling0–38 peers. Only0/16/38 pure levels and19+19 mixture have measured evidence. Other count combinations are interpolation queries, not validated claims. Multi-thread victims/backgrounds, real route-input history, wide-team transient timing, and full planner ranking remain outside scope.

## API, reproduction and checks

```python
from pathlib import Path
from optimizations.fused_moe_sve.benchmarks.small_m_overlap_model import SmallMOverlapModel

model = SmallMOverlapModel(Path("tmp/small_m_overlap_20260910/validated/model.json"))
prediction = model.predict(3, "w13", small_peers=19, large_peers=19)
# duration_us, compute_us, memory_us, overlap_us, effective efficiencies and occupancies
```

```bash
.venv/bin/pytest -q tests/test_moe_small_m_overlap.py
.venv/bin/python optimizations/fused_moe_sve/benchmarks/fit_small_m_overlap.py \
  --data tmp/small_m_pressure_20260910/report.json \
  --calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --output-dir tmp/small_m_overlap_20260910/validated
```

Use a fresh output directory for reruns. `validated/model.json` and `validated/report.json` are the final artifact/report; the earlier `fit/` output is preserved. `mixed_oracle_diagnostic.json` retains the explicitly non-predictive pressure check. Source raw measurements remain in `tmp/small_m_pressure_20260910/`.

Eight focused tests pass: serial/full/partial limits, isolated-baseline equality, invalid components/scales, API domain rejection and resource accounting.858 interpolation queries across all foreground M/stages and38-core compositions satisfy effective efficiency/occupancy bounds. Parameter/metric repeat checks, source hashes, Ruff and `git diff --check` pass. No production calibration/schema/native parity or new hardware test is claimed. Changes are confined to Lab model/fitter/tests and experiment records; the active new-model mean-best baseline remains unchanged.
