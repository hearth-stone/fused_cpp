# Single LLC-pressure proxy: retrospective M1 replay

## Working choice after the measured-rate comparison

On 2026-09-09 the user chose to continue with same-LLC background task count
as the local pressure proxy. Use the frozen candidate below as the working
baseline for subsequent Lab experiments: p=N_same/39, excluding the foreground,
with the existing global q term. Keep its original coefficients and anchor;
the HHA matched-data control fit is a separate comparison, not a replacement.

The [independent request-rate trial](measured_llc_pressure_20260909.md) did not
improve on count. Further decomposition of LLC mechanisms or replacement with
receive throughput is not a prerequisite for continuing this baseline.
Known high-pressure errors remain unresolved. This working choice does not
change the failed acceptance result, authorize production adoption, or establish
transfer to different background M/width/kernel types. No planner or calibration
change is made by this decision.

## Decision

A single local-count pressure proxy improves the fixed50 placement slice but
fails as a general replacement for the frozen measured cache-path feature.
Retain this bounded negative/control result; do not replace the original model.
This tests one concrete pressure definition, not the impossibility of aggregating
LLC mechanisms. No production model, planner, calibration, kernel or default changed.

Class E offline Lab diagnostic. New source/test/report plus one manifest entry
are the rollback boundary. No remote execution or new hardware measurements.

## Hypothesis and declaration before replay

Collapse local effects into p = same-LLC background count / 39, excluding the
foreground. All backgrounds are identical M12/1T streams; p is a concurrency
proxy, not measured LLC bandwidth, occupancy or utilization. It does not resolve
LLC arrays, miss admission, refills or return links individually.

Keep global q = max(0, Q/Q0 - 1), where Q is summed DDRC read-command occupancy
across both groups divided by summed read commands. Candidate inputs contain
one local scalar and one global scalar. Foreground miss ratio is no longer a
prediction term. Q is still measured during execution: the candidate is not a
fully plan-visible predictor, and does not establish causal attribution.

Compare T/T0 = 1+a*q+b*p and 1+a*q+b*p+c*q*p, using nonnegative coefficients,
equal-condition squared relative error and no free intercept. Interaction is
selected only with >=10% training count-CV MAPE improvement and nonworse maximum.
The same original training whitelist is used: dual session1 isolated and
same/other counts8/24/39. Each CV fold holds both placements of one count out.
No balanced, fixed50 or session2 data enters fitting or model selection.

Reference acceptance remains MAPE<=5% and maximum absolute error<=10% for EACH
validation slice. Prior results were already known: this is retrospective
replay, not a new blind/prospective validation. No feature or coefficient was
adjusted after the replay, and existing high-pressure holdouts were not fitted.

## Frozen candidate

```text
T0 = 302.74 us
Q0 = 23.363931351681227 (event-ratio units, not ns)
p = N_same / 39
q = max(0, Q/Q0 - 1)
T_hat = T0 * (1 + 0.10179803012319107*q
                + 0.4762778059621185*p
                + 0.18184212189635304*q*p)
```

Training-only CV MAPE/max: additive6.48/17.37%, interaction3.41/13.48%.
Selection favors interaction, but CV max itself exceeds10%; selection is not
an acceptance pass. The original interaction's training CV was1.57/5.30%.

## Data and baseline

Reused original native JSONL, checked by the existing complete-grid, numerical,
placement and PMU raw reader. Both datasets: Arm-codex-internal NUMA3,
foreground304, controller240, M1/1T real W13, K4096/N1024, BF16/SVE256,
N16 backend tile, 8MiB full owner B stripe, window_tiles0, R13=1; W2 unmeasured.
Backgrounds M12/1T with independent four-copy8MiB weights, no cross-panel reuse.
Same-process allocations, persistent workspace, two-domain256MiB scrub,
5ms background lead-in, five warmups and31 measured rounds per condition.
One observation is a condition median, not31 independent training conditions.

Native build provenance, ordinary aligned memory/THP caveats and noise protocol:
[dual LLC record](dual_llc_m12_20260908.md) and
[fixed-total record](fixed_total_placement_20260908.md). Each dataset's two
sessions have matching binaries and different seeds; fixed50 uses a different
Lab binary from dual LLC. It is scored as separate transfer evidence, never
pooled into training. Native sources used GCC13.2.0, C++17/O3/pthread,
armv8.2-a+bf16+sve, SVE256; source basis was c80c0c3 plus dirty Lab changes.
The current workspace also has pre-existing dirty/untracked work, preserved.

Baseline: unchanged frozen measured-feature model
`tmp/layered_supply_model_20260908/frozen_fit.json`, SHA256
`f3796a0f7e815dc5c5e297b3bafdb3bb64150358daa50150f91bec6897157d86`.
The replay verifies its bytes remain unchanged. It retains the local miss/refill
feature and queue interaction; no baseline refitting or session anchor reset.

## Results

Numbers are MAPE / maximum absolute percentage error across condition medians.

| Slice | Original S1 | Collapsed S1 | Original S2 | Collapsed S2 |
| --- | ---: | ---: | ---: | ---: |
| Same/other held16/32 | 1.71 / 3.75 | 3.29 / 9.37 | 1.54 / 4.18 | 3.58 / 9.75 |
| Balanced16/24/32 | 3.73 / 4.74 | 6.79 / 10.39 | 2.92 / 4.48 | 4.93 / 11.35 |
| Balanced48/64/78 | 11.06 / 15.47 | 11.18 / 17.54 | 11.70 / 15.57 | 12.57 / 18.67 |
| Fixed50 three placements | 7.84 / 14.39 | 4.42 / 9.13 | 9.04 / 13.90 | 5.85 / 9.26 |

Held same/other counts pass both sessions. Balanced lower/high pressure fail.
Fixed50 passes S1 but fails S2 MAPE; the improvement does not waive this gate.
All17 dual-session2 conditions: original3.22/15.57%, collapsed4.21/18.67%.
Session1 overall scores include training and are not holdout scores.

Selected absolute results, measured/predicted microseconds (signed error):

| Condition | Session1 | Session2 |
| --- | --- | --- |
| Dual balanced48 | 939.15 / 938.00 (-0.12%) | 947.83 / 932.94 (-1.57%) |
| Dual balanced64 | 1677.89 / 1411.58 (-15.87%) | 1689.71 / 1394.73 (-17.46%) |
| Dual balanced78 | 2029.84 / 1673.86 (-17.54%) | 2047.55 / 1665.28 (-18.67%) |
| Fixed25/25 | 1096.71 / 996.60 (-9.13%) | 1097.74 / 996.13 (-9.26%) |
| Fixed38/12 | 1270.31 / 1224.26 (-3.63%) | 1276.66 / 1219.54 (-4.47%) |
| Fixed12/38 | 791.06 / 795.07 (+0.51%) | 767.04 / 796.40 (+3.83%) |

The candidate captures the direction of fixed50 placement differences and repairs
the balanced48 overprediction, but underpredicts balanced64/78 more severely.
Consistent failures across both sessions make an overall acceptance inappropriate.
These are prediction errors, not execution speedups. No new bootstrap uncertainty
or parameter-identifiability claim is made; raw31-round inputs remain preserved.

## Reproduction and verification

From the local repository root:

```sh
.venv/bin/pytest -q tests/test_moe_collapsed_llc_pressure.py tests/test_moe_layered_supply_model.py
.venv/bin/ruff check optimizations/fused_moe_sve/benchmarks/collapsed_llc_pressure.py tests/test_moe_collapsed_llc_pressure.py
.venv/bin/python optimizations/fused_moe_sve/benchmarks/collapsed_llc_pressure.py \
  --output tmp/collapsed_llc_pressure_20260909/replay.json
```

Output is exclusive-create; use a new filename for another replay. Inputs:
`tmp/dual_llc_m12_20260908/session{1,2}.jsonl` and
`tmp/fixed_total_placement_20260908/session{1,2}.jsonl`.
The output preserves input hashes, model coefficients/CV and all46 condition
predictions with baseline comparisons. No native binaries or profiled timings
are used in fitting. Existing source/input dependencies remain in the workspace.

Direct tests:10 passed before replay; direct plus reused fitter tests:17 passed.
Tests check CPU-count geometry, synthetic coefficient recovery, independent
least-squares agreement, held-target isolation, and exclusion of target/miss
fraction from prediction terms. Targeted Ruff passed. A separate arithmetic
check recomputed all46 predictions from explicit condition counts and the
printed formula, agreeing within1e-9 us. No target build or production tests
were needed for this offline-only diagnostic.

## Next decision

A useful next question is whether one independently measured local request-rate
or service-pressure scalar transfers better than raw worker count. Keep the
single-local-scalar abstraction if useful; this result does not require naming
every internal LLC mechanism. It does require a better proxy or evidence for a
different response form before replacing the original model. Do not derive a
new knee from these scored high-pressure residuals or silently reuse them as
unseen validation after model redesign. Broader M/width/background-type and
prospective validation remain outside this completed first trial.
