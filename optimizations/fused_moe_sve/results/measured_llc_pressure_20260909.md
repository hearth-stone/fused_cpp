# Independently collected local request-rate proxy

## Decision

Replacing same-domain worker count with the empty-victim HHA request-rate proxy
fails the declared prediction gates. The measured rate loses sensitivity at
high load and is worse than the matched-data count and cache-path controls.
This rejects this observable/formula combination, not every aggregate LLC
pressure model. Retain as a bounded diagnostic. No production/planner model,
frozen calibration, kernel, default or remote state changed.

Class E offline Lab trial following [count proxy](collapsed_llc_pressure_20260909.md).
Rollback boundary: new analyzer, focused test, report and one manifest entry.
Existing unrelated dirty/untracked work preserved; no commit requested or made.

## Observable and independence

The source mapping controls associate CPU280–319 traffic predominantly with
HHA group27 rx_sccl. Use the SUM of count/enabled_ns for its four HHA devices,
in Gevents/s (events/ns), then median across31 empty-victim control rounds.
Backgrounds remain active in those control cells, but the real foreground
kernel does not run. Each device uses its own enabled window, never foreground
kernel_ns or controller gate_ns as the divisor. Subtract the training isolated
empty-control median and clamp the difference atzero.

This is an independently collected, source-correlated receive-throughput proxy,
not direct LLC load bandwidth, offered request demand, queue occupancy, service
latency, or a foreground-specific completion rate. Empty and real cells are
randomized separate runs with different counter windows, not simultaneous
measurements of identical instantaneous state. Controller activity and changing
background progress can affect the result. Calling it independent means
independent from target time arithmetic and target execution, not statistical
independence from all workload state.

The real-target global Q occupancy/read-command ratio remains unchanged to
isolate the local-feature substitution. Therefore the whole predictor remains
PMU-conditioned, not plan-visible or fully independently instrumented.
Unvalidated L3C ref and invariant-zero HHA retry are not fitted as features.

## Frozen design and data

Before fitting, choose only the above scalar. Compare additive q+p and
interaction q+p+q*p with nonnegative coefficients and equal-condition squared
relative error. Same train-only selection rule: interaction needs at least10%
CV MAPE improvement AND nonworse maximum error. Acceptance requires each slice
MAPE<=5%, max<=10%. No post-result feature/threshold/parameter adjustment.

Train only hha_source_mapping session1 isolated and same/other8/24/39. CV holds
both placements of one training count out. All balanced, held16/32, source
session2 and both fixed50 HHA sessions are excluded from fitting. Historical
results were already inspected; these are retrospective checks, not unseen
prospective data. Each condition is one median observation, not31 independent
training examples.

For a fair comparison, fit count and cache-path models on this SAME training
subset, with the same anchor, q, fitting and selection rules. These are NEW
matched-data control fits, not modifications of the earlier frozen models.
Do not compare their numbers to the prior non-HHA experiment as exact paired
improvements. Existing frozen JSON bytes remain unchanged and their hash is
recorded by the replay.

Inputs:
- `tmp/hha_source_mapping_20260908/session1.jsonl`, seed459808.
- `tmp/hha_source_mapping_20260908/session2.jsonl`, seed469808.
- `tmp/hha_path_sensitivity_20260908/session1.jsonl`, seed439808.
- `tmp/hha_path_sensitivity_20260908/session2.jsonl`, seed449808.

All four use native SHA256
`8a92c487b0e10b6270daaae9f3039c7d480f4b22e9fd3175dfb2191241bb24cd`.
Existing raw readers recheck complete grids, numerical/placement and PMU gates;
new extraction requires four mapped counters with running/enabled>=0.99 and
31 distinct empty-control rounds per condition. No new hardware collection.

Protocol: Arm-codex-internal NUMA3, CPU304 foreground/controller240, M1/1T W13,
K4096/N1024, BF16/SVE256, N16 tile, full8MiB B owner stripe, window_tiles0,
R13=1; W2 unmeasured. Independent M12/1T backgrounds rotate four8MiB B copies.
Persistent allocations/workspace, two-domain256MiB scrub, nominal5ms lead-in,
five warmups plus31 measured rounds. Ordinary aligned memory and existing THP
policy, no per-allocation HugeTLB claim. Native GCC13.2.0 C++17/O3/pthread,
armv8.2-a+bf16+sve, SVE256; source basis c80c0c3 plus dirty Lab changes.
Full original provenance and raw identities:
[HHA source](hha_source_mapping_20260908.md),
[HHA fixed50](hha_path_sensitivity_20260908.md).

## Fit and results

```text
T0 = 301.88 us
Q0 = 23.41629649571499 (event-ratio units)
r0 = 0.00048472570040233116 Gevents/s
q = max(0, Q/Q0 - 1)
p = max(0, empty_control_local_rate - r0)  # Gevents/s
T_hat = T0 * (1 + 0.10804791445818285*q + 0.46822868813003715*p)
```

Rate-model CV additive MAPE/max6.60/17.31%, interaction4.64/17.91%.
The interaction's larger maximum rejects its selection despite better average;
additive is frozen. Both CV maxima already exceed the10% reference target.
Count/cache controls select interaction. Full coefficients and CV are in JSON.

Second-session MAPE / max absolute percentage error, using matched-data fits:

| Slice | Empty-control request rate | Worker count | Cache-path ratio |
| --- | ---: | ---: | ---: |
| Held same/other16/32 | 4.91 / 11.50 | 3.31 / 8.71 | 1.84 / 4.21 |
| Balanced16/24/32 | 5.40 / 9.86 | 4.21 / 10.33 | 3.58 / 4.57 |
| Balanced48/64/78 | 41.37 / 54.87 | 12.16 / 19.38 | 10.14 / 15.24 |
| Fixed50 placements | 25.35 / 34.00 | 5.86 / 9.21 | 9.16 / 17.37 |

Rate-model S1 high-pressure MAPE/max41.81/55.09%; fixed50 S1 27.40/34.60%.
Failure repeats across sessions. These are prediction errors, not profiling
costs or measured execution speedups.

Selected S2 absolute values:

| Condition | Empty local Gevents/s | Real us | Predicted us | Error |
| --- | ---: | ---: | ---: | ---: |
| Source balanced48 | 1.07296 | 934.42 | 729.48 | -21.93% |
| Source balanced64 | 1.11216 | 1666.49 | 878.28 | -47.30% |
| Source balanced78 | 1.11734 | 2043.37 | 922.11 | -54.87% |
| Fixed25/25 | 1.07880 | 1095.64 | 751.23 | -31.43% |
| Fixed38/12 | 1.66725 | 1268.60 | 837.25 | -34.00% |
| Fixed12/38 | 0.52960 | 762.42 | 681.57 | -10.60% |

At balanced64/78 the source S2 empty-rate round CV is0.84/0.74%; rate changes
only about0.47% while target median time increases22.62%. S1 gives0.34% rate
change and22.38% time change. Rate CV here describes round variability, not a
confidence interval or hardware-bias guarantee. This plateau makes the measured
throughput proxy insensitive to increasing contention in this region. Global
q still varies, but the fitted combined response does not cover the growth.

## Diagnostic of the rejected interaction family

After observing failure, replay the already-declared interaction family to check
whether the selection rule alone explains it. Fit remains restricted to the
same original training whitelist; no holdout fit and no selection reversal.

Coefficients for q,p,q*p:0.10311301480924105,0.25343818033015375,
0.09412574474605967. S2 high-pressure MAPE/max22.99/37.44%, fixed50
8.13/13.60%; S1 high-pressure23.55/37.73%, fixed50 9.48/14.20%.
It is less poor than the selected additive response, but still fails and remains
worse than count in these slices. Thus the negative result is not solely the
additive selection. Retained in `interaction_diagnostic.json`; it must not be
reported as the preregistered selected result.

## Reproduction and verification

```sh
.venv/bin/pytest -q tests/test_moe_measured_llc_pressure.py tests/test_moe_collapsed_llc_pressure.py tests/test_moe_layered_supply_model.py
.venv/bin/ruff check optimizations/fused_moe_sve/benchmarks/measured_llc_pressure.py tests/test_moe_measured_llc_pressure.py
.venv/bin/python optimizations/fused_moe_sve/benchmarks/measured_llc_pressure.py \
  --output tmp/measured_llc_pressure_20260909/replay.json
```

Output uses exclusive-create; choose a new filename for a rerun. The replay
records four raw hashes, seeds, all three model fits and46 conditions per model.
The optional diagnostic is reproducible by `fit(training, rate_model['base'],
'rate', True)` and `score` in the same module, using only TRAIN_IDS to fit and
the unchanged replay rows for evaluation.

Seven direct tests passed before replay; combined24 tests passed. Tests cover
per-counter denominators, independence from target time, malformed/low-running
counters, synthetic parameter recovery, heldout isolation and exclusion of
count/cache from rate-model prediction. Independent arithmetic recomputed all46
selected rate predictions to within1e-9 us. Targeted Ruff and diff checks passed.
No target build, new bootstrap study or production integration was performed.

## Next boundary

Do not replace count with this receive-throughput scalar. A single scalar remains
an acceptable abstraction, but to explain additional slowdown past throughput
saturation it needs evidence sensitive to congestion, such as a separately
validated service probe, rather than another relabeling of accepted throughput.
The present experiment does not independently measure such service pressure.
Whether a service-sensitive proxy can transfer remains open; prior pointer-chain
and B/AB transfer caveats still apply. No new hardware experiment was started.
