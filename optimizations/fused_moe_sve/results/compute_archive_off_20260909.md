# Compute-only archive replay with early merge disabled, 2026-09-09

## Result

Completed all four original groups,46 plan entries, two trace sessions per group and one no-trace control per group. The current frozen core-pressure model has **MAE1.922ms, MAPE6.584%, signed bias+1.841ms and maximum absolute error5.131ms** across92 per-plan session medians at the uniform last-W2 compute endpoint. Of92 predictions,86 overestimate and6 underestimate. Absolute compute time therefore remains systematically high; p11's earlier close result is not representative of the full set.

The two complete sessions separately give MAE1.967/1.878ms and MAPE6.732/6.435%. Results contain measurement variation and are diagnostic, not a production calibration gate:9 of46 entries differ by more than0.5ms between session medians, and trace/native versus no-trace full-forward endpoints also differ materially for some entries. No observations were removed or parameters fitted.

Class E experiment. Every original bridge changes only `early_merge=true` to `false`; task order, dependencies, ownership and kernel configuration stay fixed. Both original/v8 and current core-pressure predictions are frozen before measurement and match their previously retained point predictions. Recent W13/W2 full/tail measurements are not integrated into these predictions. Production defaults and planner implementation are unchanged.

## Endpoint and evaluation population

Measured compute time is `max(expert W2 end) - scheduled_compute start`. It includes actual gather, synchronization and scheduler gaps before the last W2, and excludes final merge and outer preparation/logging. Scheduled-compute end, merge duration, trace-internal native E2E and untraced Python E2E are retained separately.

The model uses its frozen DAG completion estimate. Its call setup is zero and it has no explicit final-merge phase. Gather remains implicitly represented rather than an explicit modeled phase; the new measurements do not validate every absorbed overhead term. No measured merge duration is subtracted from a prediction or used to adjust parameters.

The population is exactly the prior13 VND high-skew,13 LNS median,11 LNS high-skew and9 LNS uniformish entries. Each of their two31-round compute medians receives equal weight. These are selected historical plans, not92 independent unseen workloads. Exact route/layer/off-bridge identities yield44 unique configurations. A secondary equal-configuration sensitivity, averaging repeated measured medians within each identity, gives MAE1.869ms/MAPE6.376%; the main conclusion is unchanged. Primary results retain the original92-observation weighting.

The earlier MAE1.143ms/MAPE3.652% used historical complete-forward labels with early merge on. It must not be presented as the directly comparable “before” value for this new computation-only endpoint.

## Error by original group

All errors in ms, except MAPE. Positive bias means overprediction.

| Group | Entries | Observations | Current MAE | Current MAPE | Current bias | Current max absolute error |
|---|---:|---:|---:|---:|---:|---:|
| VND high-skew |13|26|2.435|8.087%|+2.412|3.420|
| LNS median |13|26|1.667|5.652%|+1.667|5.131|
| LNS high-skew |11|22|2.737|9.723%|+2.737|3.373|
| LNS uniformish |9|18|0.555|1.922%|+0.171|1.578|
| All |46|92|1.922|6.584%|+1.841|5.131|

The high-skew groups have broad overprediction. Uniformish is closer overall but retains underpredicted cases, so a universal downward adjustment would not uniformly improve the set.

![Compute prediction and residuals](../../../tmp/compute_archive_off_20260909/compute_error.png)

The left panel compares each31-round measured median with its frozen prediction; the right panel shows signed error. Both sessions are shown without jitter, including overlapping repeated observations. Group colors/markers distinguish the populations. Most errors lie above zero, while uniformish p01 remains below zero. These points are medians, not uncertainty intervals.

For an apples-to-apples model comparison on **the same new compute observations**:

| Predictor | MAE (ms) | MAPE | Bias (ms) | Max absolute error (ms) |
|---|---:|---:|---:|---:|
| Original/v8 baseline |5.697|19.428%|+5.697|8.253|
| Current core-pressure adapter |1.922|6.584%|+1.841|5.131|

The current adapter remains substantially closer than the original baseline on this endpoint, but is not yet an accurate absolute computation-time model. This compares frozen predictors, not two runtime implementations or an end-to-end speedup.

## Plan selection is closer than absolute prediction

Choose the lowest frozen prediction within each original group and compare it with that session's measured best compute median. This is a finite-set point-median comparison; it is not a proof of a global optimum or a statistically certified winner.

| Group | Model choice | Measured best S1 / S2 | Choice loss S1 / S2 (ms) |
|---|---|---|---:|
| VND high-skew |p09|anchor /anchor|0.114 /0.050|
| LNS median |anchor|p08 /p08|0.444 /0.469|
| LNS high-skew |p05|p03 /p06|0.160 /0.064|
| LNS uniformish |p06|p06 /p06|0 /0|

The largest observed choice loss is0.469ms, much smaller than the largest absolute timing error. LNS median repeatedly misses p08. LNS high-skew's measured best changes across sessions, illustrating why very small ordering differences should not be overinterpreted. Pairwise point-order agreement is retained in JSON, with ties excluded, but is not treated as a noise-aware ranking guarantee.

## Representative residuals and retained failures

| Group / plan | Prediction (ms) | Compute S1 / S2 (ms) | Signed error S1 / S2 (ms) |
|---|---:|---:|---:|
| LNS median /p12 |34.388|29.257 /29.339|+5.131 /+5.049|
| VND high-skew /anchor |31.244|27.824 /27.838|+3.420 /+3.406|
| LNS median /p08 |31.224|27.932 /27.950|+3.292 /+3.274|
| LNS uniformish /p01 |29.217|30.795 /30.789|-1.578 /-1.572|
| LNS uniformish /p03 |28.452|28.656 /28.555|-0.204 /-0.103|

LNS median p12's compute CV is about0.79% in both sessions, much smaller than its approximately5ms systematic error. It is a concrete follow-up target. Uniformish p01/p03 remain counterexamples to a universal positive-bias explanation. No new per-stage causal attribution or parameter correction was performed in this whole-set replay.

## Repeatability and trace-control limitations

Across46 entries, the median absolute difference between the two measured compute medians is0.0765ms, the maximum2.0081ms, and9 entries exceed0.5ms. Five of92 cells have sample CV above5%. Rare long samples are retained: VND high-skew p12/session1 has a115.97ms maximum, while its P10/median/P90 are34.528/34.639/34.765ms. Its CV39.22% is therefore not a description of the central timing spread. The primary median statistic was declared before measurement; no trimming was introduced afterward.

Some session medians also shift: VND high-skew p12 is34.639/36.647ms, LNS median p07 is29.328/28.072ms. Those entries should not be interpreted at microsecond precision. The overall MAEs nevertheless remain close across sessions (1.967/1.878ms), and all samples remain in the primary score.

Trace-internal native E2E minus the same group's untraced Python E2E ranges-1.405 to+2.018ms across entries/sessions. These are different timer endpoints and separate processes; the difference includes process/execution-state variation and cannot be called a measured pure trace overhead. No untraced last-W2 endpoint was directly collected. Consequently, the primary numbers are accuracy against the declared **trace-observed compute endpoint**, not certified accuracy for an uninstrumented internal endpoint. Small selection losses are descriptive. No-trace complete-forward medians and per-plan trace/native/merge values remain in the JSON for audit.

A subsequent low-overhead compute-endpoint-only measurement would bound this uncertainty more directly if precise submillisecond prediction or deployment is requested. It is not silently substituted for this completed trace protocol.

## Measurement protocol and integrity

Arm-codex-internal, `/home/zhangxu/codex/fused_cpp`, NUMA3 CPU240–319/membind3. Same original route files, four-copy weights,216MiB scrub,192MiB output workspace,H4096/F512,2048tokens,TopK6,E256,BF16/SVE256,Ntile16. All plans retain full owner stripes `(t,w13_window_tiles,w2_window_tiles,R13,R2)=(t,0,0,1,1)`; full W13/W2 weights8/4MiB per expert, partitioned over the original1T–16T teams. Actual owner windows and physical placement remain in the frozen bridges.

Two trace sessions per group,5 warmups plus31 measured randomized/copy-paired rounds, followed by one no-trace control. Seeds by group are595001/595002,595011/595012,595021/595022,595031/595032; each control uses its group's first seed. Groups and runs execute serially. Parsing/compression occur only after a group's measurements finish, before the next group. Large raw-trace transfer occurs after all measurements finish.

All12 runs pass four-copy output equality before timing. All3688 complete traced calls, including correctness and warmup calls, have zero early-merge events and80 final-merge worker records; final merge always starts after scheduled compute, minimum margin0.08901ms. There are2852 measured trace calls forming92 medians and1426 measured no-trace calls forming46 E2E medians. Complete round/copy order, worker/expert identities, positive stage intervals, frontier/route/extension/workspace identity and model-point identity are validated. All12 run stderr and4 summary stderr files are empty.

The runtime extension SHA256 is `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`, matching the original archive. Workspace source SHA256 is `7e0124673b4742a4747f6b4b733cb364def2f455cc5919ef13379eca4cb098db`, also unchanged. Current trace-capable runner SHA256 is `ea81113c747c509b8556d5c9a698fd2bf85a8d01a4c8fc775ab9657f107219ba`, and explicit workspace profile SHA256 is `cdccff46365f217ac7984ea168da1d60731922ade9a9757d171bb95c3414ce41`, matching the preceding early-merge-off experiment. The older archive used runner `5c5b976dfb3ea2336254a939608e7f49bd71a48d89afc20c15671fe802adae35` without an explicit profile hash. That difference is recorded, not falsely asserted identical.

No native build, dependency installation, calibration update, production-default change or Git commit occurred. Existing unrelated dirty work is preserved. All46 off DAG/CPU-exclusion checks pass; both frozen predictor values remain identical to the previous replay. Local MAE/bias/regret arithmetic checks, source/artifact checks and `git diff --check` pass. Existing trace validators are reused; no unrelated test suite was rerun. The local companion scripts pass Ruff. The static Matplotlib plot was visually inspected; no fitted line or new dependency was added.

## Artifacts and reproduction

Local/remote experiment root: `tmp/compute_archive_off_20260909/`. `frozen.json` records both model predictions, plan identities, input/model hashes and seeds before data collection. Four subdirectories hold off frontiers, session/control JSON, logs, summaries and compressed raw traces. Original uncompressed traces also remain remotely. All eight retrieved compressed traces were decompressed in a stream and SHA-checked against measurement metadata; `raw_verified.json` preserves the checks.

Local `report.json` contains all92 observations, their31 compute samples, group/global errors, selection results, exact-configuration sensitivity and runtime identity. `compute_error.png`/`.svg` and `plot.py` provide the reviewed figure. Preparation/scoring artifacts and parser snapshots are retained. `summarize_group.py` exists as the executed snapshot remotely and a formatting-equivalent local companion; its measurement inputs and reported values are unchanged.

Commands, from the corresponding project root:

```sh
# Before acquisition, with fresh output directories:
.venv/bin/python tmp/compute_archive_off_20260909/prepare.py
# On Arm; uses the existing snapshot runner, no build:
bash tmp/compute_archive_off_20260909/run_remote.sh
# Local, after retrieving group JSON and summaries:
.venv/bin/python tmp/compute_archive_off_20260909/aggregate.py
MPLCONFIGDIR=/tmp/moe_compute_archive_mpl .venv/bin/python tmp/compute_archive_off_20260909/plot.py
```

The run script preserves exact commands, routes and environment. For new runs, choose fresh paths; do not overwrite this frozen data. To rerun group parsing locally, first decompress the relevant `.trace.gz` into a fresh location and preserve the expected directory structure; `summarize_group.py --folder <group> --frozen <frozen.json>` then validates and computes the endpoint.

The requested original-set replay is complete. Retain the systematic-bias and ranking results, plus noise/endpoint limitations, as the basis for the next diagnostic. Any model adjustment or wider adoption is separate work.
