# Equal-budget full/strict cost-model search comparison, 2026-09-09

## Result

Completed12 fresh-process cold searches and9 hardware sessions. Under the existing full/strict selection policy, the new core-pressure model produces a small high-skew improvement (about0.26% compute latency), an approximately2% median-workload improvement, and the identical uniformish plan. Cold search costs1.85–1.91 times as much as the old model for the same422 DAG evaluations.

The larger finding is a selection-policy failure: high-skew and median both contain a much faster reference plan in their evaluated candidate pools, but the existing uncertainty fallback replaces the16T-containing mean-best candidates with max-width8T plans. New selected plans remain about4.3–10.2ms slower than the measured references. Replacing only the cost model does not resolve that failure.

Class E/M Lab experiment. No cost-model fit, production/default change, kernel build or native model export. This evaluates the existing Python **full/strict** path, not additional VND/LNS expansion, dynamic tail pool, route slicing, or online quick planning.

## Baseline decision, 2026-09-09

Update2026-09-10: the user subsequently adopted new-model **mean-best selection** as the Lab baseline after the [fallback verification](fallback_compare_20260909.md). The earlier decision below records the configuration at the time of this experiment; it is superseded only in its selector policy. The original runner snapshot remains at `tmp/model_search_compare_20260909/bench_full_model_search.py`.

The user retained the new core-pressure model as the baseline for subsequent Lab experiments. Use `CorePressureModel` in `../benchmarks/replay_core_pressure_moe.py` with the existing v8 calibration and frozen response `f(x) = 1 + 0.28902964424300354*x + 0.4536242509371087*x*x`, where `x` is peer GEMM cores in the same LLC domain divided by32. Preserve the original model as a comparison control.

Keep early merge disabled and use scheduled-compute start to last W2 completion as the primary timing endpoint; report untraced full-forward latency separately. For follow-up full-search comparisons, retain the141-template/422-evaluation protocol unless the experiment explicitly changes the search budget. The existing uncertainty fallback remains part of this baseline; a fallback intervention must be reported as a separate policy change. The full/tail prototype and W2 panel observations are not incorporated into this frozen model.

This decision records the working experimental baseline, without changing production defaults, native exports, fitted coefficients or search behavior. Retain the measured1.85–1.91x search cost and the remaining joint-environment and selector errors when interpreting later results.

## Search population, budget and fairness

Use the three distinct route workloads underlying the previous archive: high-skew, median and uniformish. The prior four groups contain two search histories of the same high-skew route; they are not four distinct route inputs here. Read route counts from the original checked canonical state; hardware later reloads the SHA-checked original route file/layer. Hardware times are absent from the search objective and were not used to choose any result.

Each model independently runs `IntervalPlanner.plan(dynamic_tail_pool=False, bounded_tail_repartition=False)` from those counts. Both retain the original141 templates, calibrated width set1/2/4/8/16/32/40/80, isolated-LPT membership, LPT/reverse-odd/reverse-even temporal comparison, quick-winner baseline and `_select_analytic_full` uncertainty fallback. Every exported bridge explicitly sets early_merge=false. This is fresh full-search candidate generation and selection, not rescoring the historical46 entries.

All12 runs have the same141-template digest and exactly422 `_score` DAG calls each: the eligible temporal variants plus the quick-baseline event rescore. Counts include duplicate evaluations; they are not422 guaranteed unique schedules. For each workload, the ordered evaluated-task digest is also identical between models. That equality is expected because the new adapter preserves isolated T_iso, which feeds the template assignments, while changing placed joint scores. Scores differ and final choices differ in two workloads. This full mode is an enumerative bounded search; no claim of an adaptive VND/LNS trajectory change follows.

Two independent processes per workload/model verify identical selected bridges, scores, evaluation sequence and budget. Model order is old/new in repetition1 and new/old in repetition2. Each planning process is bound to CPU240/membind3, while its simulated placement remains CPU240–319. All loaded planner/cost-model source identities checked against the local source. Selections are frozen in `frozen.json` before any new hardware result exists.

The Lab subclass records score calls, forces early merge off and avoids constructing an unused native quick-planner object. The actual full search and selection methods remain unchanged; the Python quick-baseline helper is still executed. Initialization and search time are recorded separately. Imports/input reads/export/file writing are excluded; per-score ledger hashing is included. “Equal budget” here means equal DAG evaluations and templates, **not equal elapsed planning time**.

## Actual selected-plan compute time

Two31-round medians, ms. Compute completion is scheduled-compute start to the last W2 end, including gather/synchronization/scheduler gaps and excluding final merge.

| Workload | Old selected S1 / S2 | New selected S1 / S2 | Paired latency reduction S1 / S2 | Reference S1 / S2 |
|---|---:|---:|---:|---:|
| high-skew |38.120 /38.117|38.006 /38.023|0.265% /0.259%|28.340 /27.781|
| median |34.516 /34.482|33.824 /33.783|2.034% /1.963%|28.365 /29.468|
| uniformish |27.796 /27.662|27.796 /27.662|identical plan|27.796 /27.662|

High-skew paired new-minus-old differences are-0.101/-0.099ms, with95% paired-bootstrap intervals[-0.131,-0.087]/[-0.118,-0.061]ms. Median differences are-0.703/-0.676ms, intervals[-0.723,-0.655]/[-0.735,-0.656]ms. Percentages are medians of same-round `(old-new)/old`; they need not equal ratios of the marginal medians in the table. Bootstrap uses2000 IID paired-round resamples and does not eliminate temporal dependence.

Uniformish's old/new/reference bridges are byte-identical after canonical serialization. They are measured once per round under one role; zero difference is an identity result, not two independently observed samples with an inferred zero-width confidence interval. Deduplication leaves7 physical plans total:3 high-skew,3 median,1 uniformish.

High-skew's small benefit is below a2% practical-gain threshold. Median is near2%, with its second compute session just below2%; this is not a claim of passing the repository's two-session strict2% adoption gate. No hardware-selected replacement is substituted for either frozen planner output.

## No-trace full-forward control

One31-round paired no-trace session per workload, full forward including final merge:

| Workload | Old E2E median (ms) | New E2E median (ms) | Paired latency reduction |
|---|---:|---:|---:|
| high-skew |40.185|40.056|0.312%|
| median |36.551|35.906|1.762%|
| uniformish |30.015|30.015|identical plan|

Paired new-minus-old E2E differences are-0.125ms (95% interval[-0.157,-0.099]) and-0.644ms ([-0.697,-0.607]) for high-skew/median. These controls support the same direction of the small changes, but there is only one no-trace process per workload. They do not certify an uninstrumented internal compute endpoint.

The reference E2E medians are29.741ms for high-skew and30.325ms for median, also substantially faster than both selected outputs. Thus the selected/reference gap is not seen only in the traced compute endpoint.

## Search choices and the fallback failure

The selected layouts are:

- High-skew: old6×8T+16×2T, new7×8T+12×2T; both LPT.
- Median: old4×8T+24×2T with LPT, new10×8T with reverse-even ordering.
- Uniformish: both the same10×8T reference plan.

Search logs show:

| Workload / model | Minimum mean score (ms) | Final selected mean score (ms) | Minimum-score max width | Selected max width |
|---|---:|---:|---:|---:|
| high-skew /old |35.294|43.646|16|8|
| high-skew /new |31.174|40.413|16|8|
| median /old |34.116|38.826|16|8|
| median /new |29.908|35.731|16|8|
| uniformish /old |31.125|31.125|8|8|
| uniformish /new |28.426|28.426|8|8|

The reference is present in both models' actual evaluated DAG ledgers. It is the exact old-model mean minimum for high-skew and the exact old/new mean minimum for median, up to printed score rounding. New high-skew's reference score31.244ms is close to its minimum31.174ms but is not the identical mean-best plan; that distinct new mean-best plan was not hardware-measured here.

The existing `IntervalPlanner._select_analytic_full` first finds the lowest expected cost. If its maximum team width exceeds8, it seeks candidates with a narrower maximum width whose uncertainty intervals overlap, then chooses the lowest mean within that narrowed set. It has no explicit requirement that the replacement's mean be close to the original minimum. In these cases it moves from max-width16 to max-width8 even with a materially higher predicted mean.

That rule is preserved for both arms, not changed after seeing data. It explains mechanically why the mean-best candidates are not returned. In median, taking the existing mean minimum would return the already-measured reference; the current new selection is4.315–5.459ms slower. In high-skew, both final choices are about9.7–10.3ms slower than the reference. Do not assign an exact hardware benefit to disabling fallback for the distinct unmeasured new high-skew mean-best plan.

This also explains why earlier46-entry argmin rescoring cannot be substituted for a full-planner result: that analysis chose the minimum point prediction directly, while the actual full path applies this additional fallback. The experiment exposes both model effects and the policy that converts scores into a returned plan. The old uncertainty calibration was held fixed for fairness; its statistical validity for the new adapter was not established.

## Cold planning cost

Seconds, average of two independent fresh-process measurements; full per-run values are in JSON:

| Workload | Old cold search | New cold search | New / old |
|---|---:|---:|---:|
| high-skew |94.868|175.241|1.85×|
| median |136.134|259.839|1.91×|
| uniformish |144.050|272.100|1.89×|

Each run uses422 DAG evaluations, giving5,064 evaluations across12 formal searches. This is an offline Python full-search cost, not online quick-planner overhead. Two timings give a bounded repeat check rather than a broad throughput benchmark. No equal-wall-clock search comparison was performed. The new model's better point scoring therefore carries a substantial computation cost under this implementation.

## Runtime protocol, checks and limits

Hardware runs follow the previously validated workspace protocol on Arm-codex-internal, `/home/zhangxu/codex/fused_cpp`, NUMA3 CPU240–319/membind3. H4096/F512,E256,2048tokens,TopK6,BF16/SVE256,Ntile16; four weight copies,216MiB scrub,192MiB fixed pretouched FP32 route-output workspace. Full W13/W2 weights are8/4MiB per expert; original and selected plans use full owner stripes, `(t,0,0,1,1)`. No verified HugeTLB/residency claim. Native kernel/extension and workspace sources are unchanged.

Search finishes before hardware starts. Per workload: two trace processes, seeds596001/596002,596011/596012,596021/596022 respectively, each5 warmups and31 measured randomized/copy-paired rounds; one no-trace control using the first seed. Runs are serial. Summarization/compression only occur after each workload's measurements finish. Every run checks all four-copy outputs against its reference before timing.

All9 hardware runs pass output equality. All566 traced calls have zero early-merge events and final merge strictly after scheduled compute. Complete worker/expert intervals, frontier/extension/workspace identity and round/copy order pass the reused validators. The7 physical plans provide434 measured compute calls and217 no-trace calls. All12 search stderr,9 run stderr and3 summary stderr files are empty. All6 retrieved compressed traces match the original measurement SHA256 after streaming decompression.

Compute CV ranges0.125–1.176% in high-skew,0.189–1.065% in median and0.658–1.109% in uniformish. References shift across trace sessions (e.g. median28.365→29.468ms), so precise reference-gap values are session-specific. Trace-internal native E2E minus untraced Python E2E ranges[-0.200,+0.388]ms,[-0.293,+0.926]ms and[-0.590,-0.370]ms respectively. These separate endpoints/processes do not identify pure trace overhead. The much larger selected/reference gaps persist in the no-trace E2E controls.

Selected bridges round-trip through canonical executable state; all scores, full budgets and plans repeat exactly between fresh processes. The new model's GEMM evaluation counter proves its placed response was used. Three-template smoke validates9 equal evaluations and legal merge-off bridges; formal searches validate the full fallback path. Local paired-analysis checks verify sign convention and identical-plan handling. Ruff and `git diff --check` pass. No unrelated test suite, production rollout, native-model parity claim or Git commit was added.

## Artifacts and reproduction

Main Lab source: `optimizations/fused_moe_sve/benchmarks/bench_full_model_search.py`. Input, independent Lab adapter snapshot,12 search JSON/log pairs, `freeze_hardware.py`, `frozen.json`, `run_search.sh`, `run_hardware.sh`, parser snapshots, per-workload frontiers/raw traces/metadata/control/summaries, `analyze_pairs.py`, `report.json` and `raw_verified.json` are retained under local/remote `tmp/model_search_compare_20260909/` (analysis outputs primarily local). Original uncompressed traces remain remotely, compressed verified copies locally.

`input.json` contains counts and the pre-existing references, but no hardware timing target. `frozen.json` records roles, exact budget/sequence hashes, cold timings and reference membership in the evaluated pools before hardware. `report.json` preserves every selected-plan sample and paired contrast. Main runner and every loaded planner/cost-model source hash are in the search records; all match local sources. Native extension/workspace/trace identities are checked against the established early-merge-off protocol. No clean-commit claim is made for the shared dirty workspace.

Representative commands, using fresh output paths on reruns:

```sh
# On Arm:12 independent cold searches, CPU240, no hardware execution.
bash tmp/model_search_compare_20260909/run_search.sh
# Locally after retrieving all searches: validates budgets and freezes7 physical plans.
.venv/bin/python tmp/model_search_compare_20260909/freeze_hardware.py
# On Arm after transferring frozen data: two trace sessions and control per workload.
bash tmp/model_search_compare_20260909/run_hardware.sh
# Locally after retrieving verified summaries/metadata:
.venv/bin/python tmp/model_search_compare_20260909/analyze_pairs.py
```

The comparison is complete. Retain the modest new-versus-old gains, the roughly1.9× search cost, and the much larger fallback loss together. Reconsidering/calibrating the full selector should be a separate controlled change; do not silently remove it from these results or claim that replacing the cost model alone solves planner quality.
