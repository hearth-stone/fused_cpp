# Frozen measured-feature layered supply model — M1/1T W13

## Decision

The three-coefficient local-cache/DDR-queue interaction model substantially
improves retrospective prediction over total bandwidth alone. On independent
session2, held16/32 counts have MAPE1.54%/max4.18%; the entirely untrained
balanced placement at16/24/32 has MAPE2.92%/max4.48%. Both slices pass the
predeclared5%/10% reference targets.

High-pressure balanced48/64/78 extrapolation FAILS: MAPE11.70%, max15.57%.
Overall session2 MAPE3.22% must not hide this failure. Keep a bounded Lab
reference, not a production cost model, pruning rule or new physical constant.
No parameter or feature was changed after frozen replay.

## Data, split and leakage boundary

Use only the two `dual_llc_m12_20260908` raw sessions. Both use the same native
binary and revised one-victim-worker/two-domain-scrub protocol. Do not combine
the earlier16-worker/single-domain preparation data. Machine: Arm-codex-internal,
NUMA3, foreground CPU304, controller240, up to78 M12/1T backgrounds. Foreground
M1/1T real W13 only, K4096/N1024,8MiB B; no W2, other M/T, or plan-neighbor
claim. Each condition has31 measured rounds after5 warmups.

One regression observation is a condition's median, NOT31 independent training
conditions. Parameters use six nonzero conditions and one fixed isolated anchor.

| Partition | Conditions |
| --- | --- |
|Session1 training|isolated; same/other each8,24,39|
|Session1 held counts|same/other each16,32 (4 cells)|
|Session1 held placement, lower pressure|balanced16,24,32 (3 cells)|
|Session1 held placement, high pressure|balanced48,64,78 (3 cells)|
|Session2|all17 cells predicted without fitting or resetting anchors|

Before full fitting, leave-one-count-out on TRAINING counts8/24/39 removes
both same/other placements together. Compare additive and interaction models
using only these six internal-CV predictions. Select interaction only if CV
MAPE improves>=10% and maximum error does not worsen. Controls are isolated,
single bandwidth, single queue and single local feature models. All are fit
on exactly the same whitelist. No heldout-placement intercepts or count labels.

The historical aggregate results were already inspected in previous work;
this is a frozen retrospective replay, not a previously unseen prospective
experiment. Input PMU features come from the execution being estimated, so
this is a measured-feature conditional model, not a plan-visible predictor.
CPU cycles, backend stalls and target time are excluded from prediction inputs,
but observed queue/cache state is still endogenous to the workload. Good
prediction does not establish a causal decomposition.

## Features and frozen formula

Per round:

- `m = LL_CACHE_MISS_RD / L2D_CACHE_REFILL` on the foreground core.
- `Q = sum(DDRC read_cmd_occupancy) / sum(DDRC read_cmd)` across BOTH groups.
- `B = sum(32 * flux_rd / enabled_ns)` across both groups, in GB/s, for the
  bandwidth-only control. Do not divide counts by foreground time.

Take the31-round median of each ratio/rate and of the target kernel time.
Do not use the unvalidated L3C ref signal. Event ratios are not exact hit
probabilities or latency in ns. DDRC controller gates include foreground and
background and are longer than the foreground interval.

Freeze session1 isolated anchors:

```text
T0 = 302.74 us
m0 = 0.007668434373870074
Q0 = 23.363931351681227       (event-ratio units, not ns)
B0 = 15.1887824572449 GB/s    (both DDRC groups)

l = max(0, m - m0)
q = max(0, Q/Q0 - 1)

T_hat = T0 * (1 + 0.10179924958527288*q
                + 0.402750927692594*l
                + 0.20805818295290346*q*l)
```

Fit nonnegative coefficients by equal-cell squared relative-time error. No
free intercept, fitted knee, high-order term or operator residual. Exhaustive
active-subset least squares handles at most three columns using existing NumPy.
The model is anchored and monotone in its observed nonnegative inputs, but
the coefficients are effective responses, not independently identified physical
service costs or percentages of compute/memory time.

Training feature maxima: q3.8978, l0.97778, bandwidth pressure15.5687
(`max(0,B/B0-1)`). All48/64/78 balanced cells exceed the training queue and
bandwidth ranges; they remain scored and explicitly flagged, never clipped.
Second-session isolated features are also evaluated against the frozen first
anchor: prediction303.44us versus measured302.48us, not a forced exact match.

## Train-only model selection

| Model | Count-CV MAPE % | Count-CV max error % |
| --- | ---: | ---: |
|Isolated constant|26.61|60.29|
|Bandwidth only|15.39|41.16|
|Queue only|16.62|30.95|
|Local only|16.68|28.57|
|Queue + local|7.42|19.63|
|Queue + local + interaction|**1.57**|**5.30**|

Interaction is selected before session2 is loaded by replay. Across these
three training-count folds, queue coefficients range0.09946–0.10253, local
0.33801–0.41763 and interaction0.20262–0.26431. The small six-condition fit
does not establish universal parameter identifiability.

## Frozen holdout results

Selected model, MAPE / maximum absolute percentage error:

| Slice | Session1 % | Session2 % |5%/10% gate|
| --- | --- | --- | --- |
|Same/other held16/32|1.71 / 3.75|1.54 / 4.18|pass both|
|Balanced held16/24/32|3.73 / 4.74|2.92 / 4.48|pass both|
|Balanced high48/64/78|11.06 / 15.47|11.70 / 15.57|**fail both**|
|All10 held conditions|5.12 / 15.47|5.00 / 15.57|**fail both**|
|All17 conditions|3.14 / 15.47|3.22 / 15.57|**fail both**|

Session1 all17 includes training and is not a holdout score. Session2 repeats
the six training conditions at MAPE0.73%/max1.52%, independently of the
novel-count/placement slices above.

All17 session2 conditions, same training whitelist for every model:

| Model | MAPE % | Max error % |
| --- | ---: | ---: |
|Isolated constant|35.12|85.21|
|Bandwidth only|22.64|75.16|
|Queue only|18.82|48.11|
|Queue + local|9.74|50.10|
|Queue + local + interaction|**3.22**|**15.57**|

Examples from session2:

| Placement/count | Measured us | Frozen prediction us | Error % |
| --- | ---: | ---: | ---: |
|same16|424.41|420.94|-0.82|
|same32|649.50|622.33|-4.18|
|other16|329.94|330.85|+0.28|
|other32|377.45|374.17|-0.87|
|balanced16|382.23|366.57|-4.10|
|balanced24|416.62|417.36|+0.18|
|balanced32|543.55|567.90|+4.48|
|balanced48|947.83|1082.51|+14.21|
|balanced64|1689.71|1600.02|-5.31|
|balanced78|2047.55|1728.74|-15.57|

High-pressure residuals change sign; a uniform scale correction is not justified.
The condition-level direction diagnostic is136/136 in S1 and132/132 in S2
for median-time differences>=2%. It includes training-condition pairs, uses
observed PMU inputs and no paired-confidence filter: it is NOT planner-neighbor
direction validation, top-K recall or a false-pruning gate.

## Reproduction and frozen artifacts

No hardware run in this task. Reuse the preserved raw native artifacts.

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/layered_supply_model.py fit \
  --session1 tmp/dual_llc_m12_20260908/session1.jsonl \
  --output tmp/layered_supply_model_20260908/frozen_fit.json

.venv/bin/python optimizations/fused_moe_sve/benchmarks/layered_supply_model.py replay \
  --model tmp/layered_supply_model_20260908/frozen_fit.json \
  --sessions tmp/dual_llc_m12_20260908/session1.jsonl tmp/dual_llc_m12_20260908/session2.jsonl \
  --output tmp/layered_supply_model_20260908/replay.json
```

Outputs are exclusive-create; use a fresh output path for a new reproduction.
`replay_inputs.tar.gz` in the same ignored output directory preserves the
fitter, reader/runner sources, focused tests and frozen model. Raw native
inputs remain at the separately recorded dual-LLC artifact paths.
`fit` has no session2 argument. `replay` performs no fit, enforces raw-session/
binary identities and an independent second seed, and checks model bytes
unchanged. Artifacts are durable ignored files, not production profiles.

| Artifact | SHA256 |
| --- | --- |
|New frozen model|`f3796a0f7e815dc5c5e297b3bafdb3bb64150358daa50150f91bec6897157d86`|
|Replay|`9ce21a9768a28387259e71635c728c0adf6ef90fb6baecb208e22900d87ac44c`|
|Session1 raw|`143c726f4b0d079b985559b96026a037952595a1c0a19e21bef6b9286221ebd8`|
|Session2 raw|`81c68009b0c2a87921f57430f9ff0c1644f348811ac34a00e115e1b772b5303d`|
|Measured native binary|`740117153e96fb39cd715fa40f7d4b5767694659095699c4365a0b3f4ed3e070`|
|Old frozen response, unchanged|`b03819c0c5d04a5ce81424602851533e2dc2e5f8d39d68e104dfdb3bab0c4781`|

## Validation assessment: share with caveats

38 targeted tests pass (`test_moe_layered_supply_model`, `test_moe_dual_m12`,
`test_moe_m12_background`), including held-target perturbation isolation,
grouped CV, no target/cycle inputs, synthetic coefficient recovery, nonnegative
constraints and invalid input handling. Ruff and diff/manifest checks pass.
Existing raw reader revalidates complete grids, numerical flags, placement and
PMU running ratios. No production/kernel changes or tests are required for this
Lab-only formula; no planner parity/export claim is made.

Independent QA recomputed all selected-model predictions and both all-condition
MAPEs outside the predictor function, and recovered the same positive
coefficients with a direct weighted least-squares calculation. Frozen hashes
were rechecked after replay; both the new model andold response file are unchanged.

Keep the measured-feature model for independent confirmation of its supported
region. Do not tune on the failed high-pressure holdout, introduce a knee from
these three residuals, or feed PMU-conditioned predictions directly to planner
pruning. A future prospective session and a separately validated mapping from
plan geometry to the measured features remain necessary.
