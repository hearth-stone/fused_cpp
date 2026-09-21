# Joint-environment error diagnosis, 2026-09-09

## Technical summary

The largest M13/W13 miss coincides with an omitted workload class: early ready-token merge. The real p11 target overlaps substantially fewer GEMM workers than the model predicts, yet takes longer. Increasing the estimated GEMM count cannot explain that discrepancy. Merge is a concrete missing input and a leading causal hypothesis, not an identified allocation of the residual milliseconds.

M17/W2 is a different counterexample: actual and predicted GEMM counts agree, merge overlap is negligible, and the joint increment is still underestimated. The common model limitation is treating mixed worker activity through a response calibrated with homogeneous M12 competitors, with no explicit concurrent merge stream. A separate error in the simulated activity timeline also exists. No coefficient or production/model implementation changed; this is retrospective trace analysis only.

## Scope and definitions

Use the existing expert-layer-split experiment, sessions1/2, each31 measured rounds after5 warmups, paired copy/order protocol. Targets are real1T experts on CPU316 in p11; anchor M65 uses CPU308 and is not a matched placement comparison. Arm-codex-internal, NUMA3 CPU240–319, BF16/SVE256, H4096/F512, TopK6,2048 tokens. W13/W2 full weights8/4MiB, Ntile16; target owner windows equal the full stage, geometry `(1,0,0,1,1)`. Competitors include1T and16T teams; their per-worker full owner stripes differ from homogeneous1T synthetic backgrounds. Exact source/build/route identities and measurement environment are inherited from [expert_layer_split_20260909.md](expert_layer_split_20260909.md).

“Average active cores” is summed per-worker stage overlap in worker-ms divided by target stage duration, then the median across31 rounds. It is occupancy, not hardware request rate, bandwidth, cache misses or useful compute utilization. Real intervals use each worker's own start/end timestamps, not the slowest team-worker envelope. Same LLC is CPU280–319 for CPU316; CPU240–279 is the other domain. The target worker is excluded. Independently reported medians need not sum exactly.

Joint-increment error means `(predicted joint − predicted isolated) − (measured joint − measured isolated)`. It refers to the frozen core-pressure model, not a full-MoE integration of the new full/tail table. The latter has no validated real mixed-pressure mapping.

## M13: fewer GEMM peers but a new merge stream

Session2 M13/W13 actual anchor/p11 times are1.839/2.897ms; measured isolated is1.564ms. The p11 joint-increment error remains−0.974ms after isolating base error.

| M13/W13 context | Actual local GEMM cores | Model local GEMM cores | Actual local merge cores | Actual other-domain merge cores |
|---|---:|---:|---:|---:|
| anchor |38.1|37.8|approximately0|0|
| p11 |22.1|35.9|6.7|5.5|

p11 starts W13 at24.77ms after scheduled-compute origin; the model starts it at24.37ms. Anchor starts at14.20ms, model14.19ms. Both bridges set `early_merge=true`. Runtime workers execute `merge_ready_token` while outstanding experts continue. The current model's placed DAG has expert phases only; the core-pressure adapter counts `cold_b`/`steady_b` GEMM workers, and does not create concurrent ready-token merge events or their memory demand.

Merge reads the six FP32 route contributions and writes the BF16 output. For H4096/TopK6 this is nominal96KiB route-input plus8KiB output per token, excluding metadata, cache-line allocation and coherence. This is logical traffic, **not measured DDR traffic**. The runtime also updates token completion state. Prefetching/scanning outside the merge timer is not captured by merge occupancy.

Session1 repeats the pattern: p11 M13/W13 takes2.858ms with22.2 local GEMM cores,6.6 local merge cores and5.3 other-domain merge cores. Therefore the omission is not a single-round coincidence. However, M13/W2 also overlaps merge (local3.0, other6.7 cores in session2) and has a nearly correct modeled joint increment (+0.006ms signed error). Merge presence alone cannot specify slowdown; victim phase, full/tail composition, activity intensity and timing matter.

The observed target-duration denominator is endogenous: slowdown can extend the target into a later, quieter interval. The lower average GEMM occupancy is not proof that reduced concurrency caused slowdown. It does establish that simply replacing the model's count with this observed average would not supply a larger count-only penalty. This analysis does not perform an oracle resimulation of the nonlinear model.

## M17: count agreement does not validate the response

Session2 p11 M17/W2 takes1.178ms versus isolated0.888ms: measured increment0.290ms, modeled increment0.108ms, remaining error−0.182ms. Actual/model local GEMM occupancy is37.3/36.9 cores. Local merge occupancy is0.023 cores, other-domain0.009: the prominent M13 merge overlap is absent.

During this stage, local W13/W2 occupancies are8.9/28.3 cores, with23.3 cores belonging to16T teams. Anchor M17/W2 sees12.2/19.7 W13/W2 cores and lower total occupancy31.9, and takes1.018ms. Both total activity and stage composition differ; this comparison does not isolate a pure W2-background effect. Session1 p11 similarly has37.0 GEMM cores, negligible merge and1.221ms target time.

The homogeneous M12 core-pressure response has no distinction for competing W2 direct-route stores, team geometry or victim tail remainder. W2 output/coherence effects, pressure concentration within the target and history remain plausible mechanisms. The trace does not measure their request rates. This is evidence against explaining every residual with count-timeline correction or early merge alone.

## M35 and the contrasting low-residual cases

p11 M35/W13 starts at0.39ms, takes4.000ms, sees35.4 local GEMM cores against37.1 modeled, and has no merge overlap. Its joint-increment error is−0.328ms. Anchor has almost identical timing and composition. Thus a late execution prefix and early merge are not necessary for this residual; mixed-background response or victim full/tail behavior needs independent validation even near the beginning of the call.

M31/W13 and M65/W13 see actual/model occupancies38.6/37.3 and37.1/37.4, respectively; their remaining increment errors are only−0.039/−0.091ms. M35/W2 has38.9 actual versus38.0 modeled cores but near-zero increment error(+0.003ms). A global contention multiplier would therefore correct some stages while overcorrecting others. These are selected-case diagnostics, not population-wide attribution shares.

The synthetic full/tail table's p38 M13 estimate1.940ms is still0.957ms below real p11 W13. That synthetic sensitivity endpoint is not a bound for a background containing16T W13/W2, early merge and changing activity. Existing data cannot assign the discrepancy to either the full12-row block or the1-row tail.

## Timeline error is a separate problem

For p11 M13/W2, predicted start26.07ms versus measured27.66ms moves the predicted stage into a very different environment: model27.2 local GEMM cores versus actual2.8. This is downstream of W13 and preceding-chain duration errors. Nevertheless, its joint-increment prediction happens to be close, demonstrating cancellation and the limits of interpreting one accurate duration as correct pressure modeling.

There are two distinct mappings to validate: schedule-to-activity timeline, and activity-to-victim service cost. Trace occupancy diagnoses the first descriptively; M17 and M35 demonstrate that matching GEMM counts alone cannot validate the second. B history and full/tail response belong in the second mapping, but real per-panel timings are still missing.

## Reproduction, checks and limitations

Run `.venv/bin/python tmp/joint_environment_diagnosis_20260909/analyze.py` from the repository root with a fresh destination filename. It validates trace SHA against session metadata and uses the existing parser to validate frontier identity, output correctness, call count, round/copy order and worker/expert completeness. It then joins per-worker intervals to physical placement and replays the frozen model's event log. Outputs are `report_with_merge.json` (40 measured summaries,20 model summaries,1240 per-target samples) and the retained preliminary GEMM-only `report.json` in the same directory. No original artifacts were overwritten.

Additional review checked31 samples per cell, local GEMM+gather+merge occupancy at most39 cores for every target sample, and equality of stage medians with the prior isolation report. Both sessions support the principal patterns. Local current source establishes the missing phase representation; no native binary was rebuilt or newly measured. Existing trace-on/off endpoint sensitivity remains applicable; microsecond-scale discrepancies should not be treated as precise causal effects. Worker stage intervals may include stalls or descheduling, and do not identify cache levels, service rates, bandwidth, or B residency.

## Next discriminating experiments

1. Prioritize M13/W13: measure real full12-row and tail1-row times together with ready-token merge intervals. Add a matched `early_merge=false` control, acknowledging that it changes scheduling/history and total-forward time; compare target W13 and peer time courses, not only end-to-end time. A controlled synthetic merge-background replay can then isolate the service effect more tightly.
2. For M17/W2 and M35/W13, hold foreground, affinity and competitor count fixed while varying background W13 versus W2 and1T versus16T ownership. Preserve actual route-output stores. This separates activity type from count and foreground shape.
3. Only after these controls, choose whether task-count pressure needs separate weights for GEMM classes and merge. Compare an observed-timeline diagnostic against the autonomous simulated timeline to measure propagation error without fitting it away.

Open questions: which real panel absorbs the M13 loss, how much is attributable to merge under a matched intervention, and whether a compact weighted pressure adequately transfers across W13/W2 and team widths. No numeric allocation to these causal mechanisms is claimed yet.

Report delivery note: the optional portable HTML packager rejected the native chart because its source contract requires SQL query text. This analysis is sourced from native traces and Python, so no SQL provenance was invented. The complete Markdown report and JSON/Python companion remain the deliverables; no HTML rendering or browser QA is claimed. The attempted canonical artifact is retained for diagnosis. Static review and `git diff --check` passed; no code suite or hardware benchmark was rerun for this read-only diagnosis.
