# Frozen new-model fallback comparison

Experiment lineage: 2026-09-09; hardware execution: 2026-09-10. Class E with a bounded M selector counterfactual. Production planner, cost-model coefficients, native exports and kernels are unchanged.

## Measured selector effect

Returning the frozen mean-best candidate substantially improves both changed workloads. The existing fallback is a material source of selection loss on these routes.

Compute medians in milliseconds; percentages are medians of same-round percentage reductions, not ratios of marginal medians:

| Route/session | Fallback | Mean-best | Paired reduction | Paired delta95% interval, ms |
|---|---:|---:|---:|---:|
| high-skew S1 |38.01810|29.28043|22.928%|[-8.75158,-8.61942]|
| high-skew S2 |37.98759|29.29230|22.931%|[-8.73811,-8.68102]|
| median S1 |33.81652|28.36002|16.183%|[-5.50065,-5.43556]|
| median S2 |33.82536|29.58015|12.549%|[-4.31008,-4.15801]|
| uniformish S1 / S2 |27.70370 /27.67897|same plan|unchanged|identity, not independent samples|

Paired median deltas are-8.70508/-8.71757ms for high-skew and-5.47975/-4.25108ms for median. Gain P10 is21.52%/22.71% and15.80%/12.13%, respectively. Both changed workloads pass the predeclared two-session practical-gain gate. This supports a bounded Lab selector conclusion, not automatic production adoption.

Untraced full-forward controls agree with the primary endpoint:

| Route | Fallback / mean-best median, ms | Paired reduction | Paired delta95% interval, ms |
|---|---:|---:|---:|
| high-skew |40.08457 /31.23123|22.115%|[-8.89593,-8.84442]|
| median |35.85029 /30.28569|15.522%|[-5.57933,-5.53335]|
| uniformish |29.75469 /29.75469|unchanged plan|identity, not independent samples|

Median mean-best shifts by1.22013ms between sessions while its fallback is stable. Retain this variation: the measured improvement is12.55–16.18%, not one universally fixed value. Compute CVs are0.117–0.963% for high-skew and0.202–1.909% for median. No samples were removed, and all four paired intervals remain well below zero.

The result isolates the last selector step: model, score pool, search budget, routes, runtime and merge policy are fixed. It does not establish that the model's mean-best candidate is the hardware optimum over all candidates, nor that wide teams always win. The old cost model is not an arm of this experiment. In the reused machine-readable `report.json`, `old` denotes fallback and `new` denotes mean-best; both use the same frozen new model.

Verification decision: retain this as a bounded Lab reference supporting direct mean-best selection. The verification itself did not change fallback policy; the subsequent user adoption is recorded below. Search evaluation cost is unchanged by this comparison; the expensive cost-model simulation remains a separate issue.

## Adopted experimental baseline, 2026-09-10

The user adopted **the frozen new model plus direct minimum predicted compute time** for subsequent Lab experiments, retaining the inherited fallback as an explicit control. `bench_full_model_search.py` now defaults to `--model new --selector mean`. It retains the original secondary tie-breaks `(pessimistic, working_set, resource_groups)`, candidate generation, early-merge-off policy and full-stripe geometry. The v8 calibration and pressure coefficients remain frozen.

Use the same command with `--selector fallback` for the old selection rule under the new model. `--model old` remains available for cost-model comparisons; specify the same selector on both sides when isolating a model change. Production Python/native planner behavior is unchanged. The imported `AuditedFullPlanner` helper retains its fallback constructor default for archival consumers; the active CLI explicitly passes its mean default.

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/bench_full_model_search.py \
  --input tmp/model_search_compare_20260909/input.json --case high_skew \
  --adapter optimizations/fused_moe_sve/benchmarks/replay_core_pressure_moe.py \
  --output tmp/mean_baseline_20260910/high_skew_default.json
```

Use a fresh output filename for each run. Historical2026-09-09 full-search reproduction retains its original runner at `tmp/model_search_compare_20260909/bench_full_model_search.py`; its recorded runner hash does not refer to the subsequently updated CLI. Original result files and source snapshots are preserved.

Baseline implementation validation: four focused tests pass, including both selector behaviors and empty-pool rejection. Frozen-pool replay of both selectors on all three routes reproduces all six previously measured selections and complete rankings, each with422 ordered score calls. Artifacts are under `tmp/mean_baseline_20260910/`. The existing ARM paired measurements above provide performance evidence; no new hardware speedup is claimed for this wiring change.

The command above also completed a fresh local macOS high-skew full search with model/selector arguments omitted. Its output explicitly records `new/mean`; all422 DAGs and scores match the archived new-model search, and the selected bridge matches the hardware-measured mean-best plan. Local initialization plus search took74.751s; this is an entrypoint integration check on the local machine, not an ARM planning-speed comparison.

## Completed validation and limits

All9 hardware runs pass four-copy output correctness. All406 trace calls have zero early-merge events and all80 final-merge workers begin after scheduled compute completes; the smallest observed gap is0.09247ms. There are310 measured compute calls and155 untraced measured calls across five unique plans. All six locally retained compressed trace streams reproduce the metadata SHA256; all run and summarizer stderr files are empty. Runtime identities match the previous merge-off experiment.

Trace-internal native E2E minus separate untraced Python-call E2E ranges from-0.252 to-0.165ms for high-skew,-0.197 to+1.101ms for median and-0.269 to-0.236ms for uniformish. These different endpoints and processes are sensitivity evidence, not pure trace-overhead estimates. The independent untraced control supports the same gain direction; no untraced last-W2 timestamp is available.

The three routes are existing diagnostic workloads, not new holdouts. The measured conclusion is that this inherited fallback loses substantial time on the two changed routes; it does not validate removal on other shapes, machines, workloads, quick planning or the native planner.

## Question and frozen protocol

Does the existing full-selector uncertainty fallback improve execution over returning the new model's minimum predicted compute-time candidate? Reuse the three high-skew, median and uniformish route inputs and complete new-model score ledgers from [the full-search comparison](full_model_search_compare_20260909.md). No hardware measurements enter selection.

`prepare_fallback_comparison.py` regenerates all141 templates and422 ordered score calls per route. Each DAG must match the frozen task digest before its score is reused. The complete ranking and original fallback bridge must match the archived search. It exports both the original selection and the minimum `(makespan, pessimistic, working_set, resource_groups)` candidate from that same pool. All1266 ordered calls and all three original rankings/bridges match. Five distinct exported bridges pass canonical-state roundtrip validation, have early merge disabled and use full owner stripes. An independent live model rescore reproduces all five frozen predictions exactly.

This is a frozen-pool selector replay, not another cold full search. The prior two independent new-model cold searches per route remain the search-cost evidence; no claimed search speedup is inferred from cached-score reconstruction. Single local selector timings are diagnostic only, not ARM cold-planner latency.

| Route | Fallback shape/order | Mean-best shape/order | Fallback / mean prediction, ms |
|---|---|---|---:|
| high-skew |7×8T +12×2T, LPT|4×16T +16×1T, LPT|40.413229 /31.174466|
| median |10×8T, reverse-even|1×16T +8×8T, reverse-odd|35.731402 /29.907690|
| uniformish |10×8T, reverse-even|identical bridge|28.425762 /28.425762|

The high-skew mean-best bridge is `87a80ca93cc1a87d1fe987ba8965dbcd2efeda23356cfb0db5aa0889a1f8cb50`; it is measured directly here, not substituted with the previous reference. Median mean-best is `42ec2287201ba5aa3a656d1cf52294e4a830a67951d07bc92c7e220a3903a955`, the previously measured reference. Full identities are in `frozen.json`.

The inherited uncertainty is15% on these candidates. High-skew fallback and mean intervals are[34.351,46.475] and[26.498,35.851]ms; median intervals are[30.372,41.091] and[25.422,34.394]ms. They overlap despite fallback mean penalties of29.64% and19.47%. These are heuristic uncertainty bands, not newly calibrated probabilistic confidence intervals. The experiment does not refit them.

## Hardware methodology

- Host `Arm-codex-internal`, project `/home/zhangxu/codex/fused_cpp`; NUMA3 CPUs240–319 and memory node3, BF16 SVE256, backend N tile16.
- Shape H4096/F512/E256/T2048/top-k6, TP degree4 modeled with one concurrent rank. Same three route files/layers and SHA identities as the full-search input.
- Full packed-B W13=8MiB and W2=4MiB per expert. Widths1/2/8/16 have full-stripe per-worker footprints8/4/1/0.5MiB for W13 and4/2/0.5/0.25MiB for W2. Runtime geometry is `(t,0,0,1,1)` for `(t,w13_window_tiles,w2_window_tiles,R13,R2)`.
- Existing extension and runner, four rotating weight copies,216MiB scrub,192MiB route-output workspace; same page/workspace policy as the previous comparison. No build or dependency installation.
- `OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=false MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. The process initially retains all80 CPUs; the runner sets the80-thread team.
- Per route: two independent trace sessions,5 warmup and31 measured paired rounds, randomized plan order; one untraced control with the same counts. Seeds597001/2,597011/12,597021/22; controls use each first seed.
- Primary endpoint: scheduled-compute start to last W2 completion. Untraced full-forward time is reported separately. All output and trace merge-state gates must pass before interpreting performance.
- Compare paired `mean-best - fallback` latency and paired percentage reduction. Report2000 IID paired-round bootstrap95% intervals with seed597999, P10/P90 and per-plan variability. IID intervals do not eliminate temporal dependence. Keep every measured sample.
- Uniformish roles alias one identical bridge, measured once per round. It is an unchanged-plan control, not independent zero-variance evidence.
- Predeclared practical-gain gate: both sessions above2% paired median improvement and positive gain P10 for changed selections. Findings remain bounded to these existing routes; no universal or production-adoption claim.

## Reproduction and evidence

Run locally from the repository root with a fresh output directory:

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/prepare_fallback_comparison.py \
  --archive tmp/model_search_compare_20260909 --output tmp/fallback_compare_20260909
.venv/bin/pytest -q tests/test_moe_fallback_comparison.py
.venv/bin/pytest -q tests/test_moe_analytic_model.py \
  -k 'analytic_full_steps_down_one_width_inside_systematic_uncertainty or analytic_full_optimizes_expected_makespan_not_working_set'
```

The reconstruction command refuses an existing output directory. Preserve the original run and use another destination for a rerun. Hardware orchestration snapshots refer explicitly to the original experiment directory:

```bash
ssh Arm-codex-internal 'cd /home/zhangxu/codex/fused_cpp && bash tmp/fallback_compare_20260909/run_hardware.sh'
.venv/bin/python tmp/fallback_compare_20260909/analyze_pairs.py
```

Raw artifacts and scripts live under `tmp/fallback_compare_20260909/` locally and on the stated remote host/root. Existing source baseline is commit `c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5` plus the preserved dirty workspace; `status_before.txt` records that state. The original search records carry loaded-source identities, checked before pool replay. No commit was made.

Runtime identities retained from the previous comparison:

| Artifact | SHA256 |
|---|---|
| Runner |`ea81113c747c509b8556d5c9a698fd2bf85a8d01a4c8fc775ab9657f107219ba`|
| Workspace helper |`7e0124673b4742a4747f6b4b733cb364def2f455cc5919ef13379eca4cb098db`|
| Workspace profile |`cdccff46365f217ac7984ea168da1d60731922ade9a9757d171bb95c3414ce41`|
| Extension |`dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`|

Focused validation: frozen-ledger mismatch/budget regression test1 passed; existing full-selector tests2 passed; Ruff and `git diff --check` passed. Numerical execution and trace validation passed as recorded above. Native planner parity and production adoption are outside this selector-only Lab experiment. `report.json`, `raw_verified.json` and `independent_scores.json` retain the aggregate result, raw-stream checks and independently recomputed model scores.
