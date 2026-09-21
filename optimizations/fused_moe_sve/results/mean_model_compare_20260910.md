# Old and new models under the same mean-best selection rule, 2026-09-10

## Result

The new model predicts absolute time more accurately, but does not select faster plans in this comparison. Under the same mean-best rule, median and uniformish select identical plans. High-skew selects a different temporal order that is consistently slower with the new model. The earlier improvement from removing fallback must not be attributed to the new cost model.

Common population: four unique plans, two session medians each, eight observations per model:

| Metric | Old model | New model |
|---|---:|---:|
| MAE |5.53128ms|1.73904ms|
| MAPE |19.4535%|6.1519%|
| Signed bias |+5.53128ms|+1.73904ms|
| Maximum absolute error |7.53280ms|3.48312ms|
| Equal-route-weight MAE |5.11753ms|1.43803ms|

All eight predictions from each model overestimate measured compute time. Per-route common-plan MAE improves from6.77254 to2.64207ms for high-skew,5.14627 to0.93761ms for median, and3.43378 to0.73443ms for uniformish. These are paired common-population error comparisons, not errors on different self-selected populations.

Selected-plan compute medians, ms:

| Route/session | Old-selected | New-selected | Paired new-minus-old, ms | Paired new-plan slowdown |
|---|---:|---:|---:|---:|
| high-skew S1 |27.76089|29.34854|+1.57838|5.6886%|
| high-skew S2 |27.78438|29.37486|+1.57571|5.6766%|
| median S1 / S2 |28.36002 /29.58015|same plan|0 by identity|unchanged|
| uniformish S1 / S2 |27.70370 /27.67897|same plan|0 by identity|unchanged|

High-skew paired delta95% intervals are[1.54677,1.68547] and[1.45908,1.68185]ms, both strictly positive. Compute CV ranges0.938–1.116%. The paired gain P10/P90 values are negative in both sessions: new-minus-old performance is worse across the reported central distribution, not just a marginal-median artifact. In the report JSON a negative `reduction_pct` denotes this slowdown.

Untraced high-skew E2E medians are29.90210ms old-selected and31.12023ms new-selected; paired median delta+1.26784ms,95% interval[1.21625,1.38241]ms, paired slowdown4.2584%. Identical-plan E2E controls are30.28569ms for median and29.75469ms for uniformish; these are reused identical-plan measurements, not independent model-arm samples.

The high-skew disagreement is temporal ordering within the same `4×16T+16×1T` shape. The new model predicts LPT ahead by0.06954ms, while hardware puts LPT behind by about1.58ms. The old model gets the direction right but predicts only0.09204ms separation; it also substantially underestimates the ordering effect. Better absolute calibration therefore does not establish accurate fine-grained ranking.

Decision: retain this negative selection result alongside the user's new-model experimental baseline. This experiment does not change that baseline or refit either model. It supports improved absolute prediction on the measured common set, but rejects an overall selected-runtime superiority claim on these three routes. Temporal-order differential prediction is a concrete remaining issue.

## Scope and method

Class E/M Lab comparison. Both models select the minimum `(makespan, pessimistic, working_set, resource_groups)` candidate, with early merge disabled. The experiment changes neither model coefficients nor the production planner. It separates prediction accuracy on a common plan population from actual runtime of each model's selected plan.

The original full-search input contains three distinct route workloads: high-skew, median and uniformish. Reconstruct both models'141-template/422-call search ledgers, checking every ordered DAG, complete ranking, repeated-run score identity, loaded model/planner source identities and calibration. Six reconstructed pools pass. Each selected bridge passes canonical-state roundtrip and full-stripe/merge-off checks. Cross-score both models on every plan in the deduplicated union of their selections before hardware measurement.

| Route | Old mean-best | New mean-best | Old / new prediction on their own choice, ms |
|---|---|---|---:|
| high-skew |4×16T +16×1T, reverse-odd|4×16T +16×1T, LPT|35.293690 /31.174466|
| median |1×16T +8×8T, reverse-odd|identical bridge|34.116357 /29.907690|
| uniformish |10×8T, reverse-even|identical bridge|31.125114 /28.425762|

There are four unique plans. Median and uniformish's old/new choices are byte-identical and match the plans just measured in [the fallback experiment](fallback_compare_20260909.md). Reuse those exact measurements, checking source-frontier route/layer/calibration/extension identities and the complete bridge. High-skew's choices differ and require a fresh same-process paired comparison; old and new timings from different historical sessions are not subtracted.

Both models' predictions on the common population:

| Route/plan | Old prediction, ms | New prediction, ms |
|---|---:|---:|
| high-skew, old-selected reverse-odd |35.293690|31.244006|
| high-skew, new-selected LPT |35.385728|31.174466|
| median, common selected plan |34.116357|29.907690|
| uniformish, common selected plan |31.125114|28.425762|

The old model prefers high-skew reverse-odd by0.092039ms; the new model prefers LPT by0.069540ms. These are model margins, not measured improvements. No fitted correction or hardware-selected replacement is introduced after freezing the selections.

Prediction metrics use the same four plans and two session medians each:8 observations for each model, with MAE/MAPE/signed bias/max absolute error. High-skew contributes two of the four plans; additionally report equal-route-weight MAE. Do not compare each model's error on only its own selected plans as a common-population metric. Selected-runtime metrics use paired31-round medians,2000 IID paired bootstrap samples with seed598999, and P10/P90 of the paired percentage reduction. A negative reduction means a new-model selection regression. Identical plans have no independent confidence interval.

## Execution and evidence

New measurements: high-skew two trace sessions and one untraced control, seeds598001/598002 and598001. Each has5 warmup and31 measured rounds, randomized old/new order, four rotating weight copies and216MiB scrub. Reused measurements: median and uniformish from `tmp/fallback_compare_20260909/`, selecting only their mean-best roles. They were collected earlier on2026-09-10 with the same protocol and binary. No unchanged-plan measurement is represented as a fresh run.

Host `Arm-codex-internal`, root `/home/zhangxu/codex/fused_cpp`, NUMA3 memory and CPUs240–319. H4096/F512/E256/T2048/top-k6, BF16 SVE256, N tile16,192MiB route-output workspace. Environment: `OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=false MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`; the runner sets the80-thread team after initial affinity validation. Same existing extension/workspace runner and page policy as the preceding experiment; no build, installation or production change.

Full stage packed-B bytes: W13=8MiB, W2=4MiB per expert. All selected plans use `(t,0,0,1,1)` runtime geometry. For widths1/8/16, full owner stripes have W13 footprints8/1/0.5MiB and W2 footprints4/0.5/0.25MiB per worker. Primary endpoint is scheduled-compute start to last W2 completion; untraced full-forward latency is separate.

The common-plan sample is deliberately small and selected by the models. It is not an independent route holdout or the complete candidate pool measured on hardware. Retain both reused session values and all new paired samples, including regressions. Existing cold full-search times remain the search-cost evidence; frozen-ledger replay does not measure cold-search performance.

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/prepare_mean_model_comparison.py \
  --archive tmp/model_search_compare_20260909 --output tmp/mean_model_compare_20260910
.venv/bin/pytest -q tests/test_moe_mean_model_comparison.py tests/test_moe_fallback_comparison.py
ssh Arm-codex-internal 'cd /home/zhangxu/codex/fused_cpp && bash tmp/mean_model_compare_20260910/run_hardware.sh'
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_mean_model_comparison.py \
  --root tmp/mean_model_compare_20260910 --reuse tmp/fallback_compare_20260909
```

Preparation and report creation require fresh output paths; preserve original artifacts when rerunning. Raw new artifacts, frozen common predictions and script snapshots are in `tmp/mean_model_compare_20260910/` locally and under the same relative remote path. Reused raw artifacts remain in their original fallback experiment directory. Runtime identities are verified against the reused measurement metadata; original loaded source and input identities are retained in `frozen.json`. Source baseline remains commit `c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5` plus the pre-existing dirty workspace recorded in `status_before.txt`; no commit was made.

Validation: seven focused tests pass, covering common-population error signs, paired regression direction, identical-plan intervals and frozen-ledger/selector behavior. Ruff and `git diff --check` pass. Three fresh high-skew hardware runs pass four-copy correctness; the reused source runs retain their checked correctness. All six source trace streams are hash-verified, covering406 calls with zero early merge and post-compute final merge. These include the unused fallback arm in the reused median source; the actual comparison population contains248 measured compute calls and124 measured untraced calls across four plans. All source run/summarizer stderr files are empty. No production/native-planner parity or new cold-search timing claim is made.
