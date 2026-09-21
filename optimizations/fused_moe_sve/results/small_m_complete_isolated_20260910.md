# Complete M1–11 real-expert isolated1T grid, 2026-09-10

## Result

All eleven requested isolated1T points were measured in both sessions on CPU316. The current frozen model underestimates most GEMM stages, with substantial exact-M variation: M1/2 have the largest relative error, M7/8 W13 is near measured time, and M9/10 again have material underprediction. No parameters were fitted and the experimental baseline is unchanged.

Each measured column gives session1 / session2 medians. Signed error is `(prediction/measurement - 1) * 100`; negative means underprediction. All times below are microseconds.

### W13

| M | Frozen prediction | Measured S1 / S2 | Signed error S1 / S2 |
|---:|---:|---:|---:|
|1|243.478|323.450 /336.630|-24.72% /-27.67%|
|2|243.478|317.770 /328.570|-23.38% /-25.90%|
|3|378.132|415.890 /428.110|-9.08% /-11.67%|
|4|378.132|414.590 /423.150|-8.79% /-10.64%|
|5|567.199|583.550 /590.440|-2.80% /-3.94%|
|6|567.199|582.950 /589.230|-2.70% /-3.74%|
|7|756.265|752.870 /758.480|+0.45% /-0.29%|
|8|756.265|751.840 /759.480|+0.59% /-0.42%|
|9|945.331|1039.910 /1044.680|-9.09% /-9.51%|
|10|945.331|1040.520 /1040.510|-9.15% /-9.15%|
|11|1134.397|1206.110 /1208.310|-5.95% /-6.12%|

### W2

| M | Frozen prediction | Measured S1 / S2 | Signed error S1 / S2 |
|---:|---:|---:|---:|
|1|121.739|154.880 /165.560|-21.40% /-26.47%|
|2|121.739|153.770 /160.350|-20.83% /-24.08%|
|3|189.066|208.060 /211.300|-9.13% /-10.52%|
|4|189.066|207.280 /210.690|-8.79% /-10.26%|
|5|283.599|297.270 /303.320|-4.60% /-6.50%|
|6|283.599|296.880 /303.410|-4.47% /-6.53%|
|7|378.132|387.510 /390.950|-2.42% /-3.28%|
|8|378.132|385.900 /388.320|-2.01% /-2.62%|
|9|472.666|541.390 /534.770|-12.69% /-11.61%|
|10|472.666|532.440 /538.850|-11.23% /-12.28%|
|11|567.199|624.180 /625.980|-9.13% /-9.39%|

### Aggregate accuracy and repeatability

Equal weight over11 M values ×2 session medians per stage:

| Metric | W13 | W2 |
|---|---:|---:|
| MAE |51.012us|32.175us|
| MAPE |9.353%|10.466%|
| Signed bias |-50.301us|-32.175us|
| Maximum absolute error |99.349us|68.724us|

Within-session CV spans0.683–5.653% for W13 and0.530–6.957% for W2. M1/2 fluctuate most; for example M1 W13 shifts323.45→336.63us and W2 shifts154.88→165.56us between sessions. Keep both values. Their20–28% underprediction repeats despite that variability. M7/8 W13 is within0.6% in both sessions; W2 is2.0–3.3% low.

M5/6 real-expert W13 is only2.7–3.9% low and W2 is4.5–6.5% low. The prior synthetic M5 measurement had materially larger error under its own protocol; it is not substituted for these fresh real-expert points. The grid now includes the previously missing M2/6/9 and real-expert M5/11. Accuracy is not monotonic in M, so a single blanket uplift risks worsening M7/8.

Decision: retain a bounded complete-grid reference. Independent stage underprediction is confirmed and should be kept separate from the additional temporal-order/joint-environment error. No isolated correction, contention factor or production/default change is adopted by this measurement.

### Completed checks

All three hardware runs pass four-copy output correctness. Each trace session has481 valid calls, including440 isolated-target calls. Across both sessions, all880 isolated-target calls have no background task overlap and all targets run on physical CPU316. Minimum observed target-to-background gap is0.00056ms. All962 traced calls have zero early merge and80 final-merge workers after scheduled compute; minimum merge gap is0.09630ms. Both compressed trace streams match their recorded raw SHA256. There are682 measured isolated target calls, providing682 observations per target stage across the11 points.

The no-trace control has the same12 plans and36 rounds. Across isolated plans/sessions, trace-internal native E2E minus separate untraced Python-call E2E is between-0.229 and-0.110ms. This is a whole-call endpoint/process sensitivity check, not a stage instrumentation correction. All run and parser stderr files are empty.

Nine existing isolation/streaming tests and two new actual-CPU audit tests pass. Ruff and `git diff --check` pass. New code is Lab preparation/analysis only; production kernels, planner behavior, profile schema and model coefficients remain unchanged.

## Scope and frozen protocol

The user requested fresh measurements of every M below12. This Lab experiment measures one real expert for each M1–11, using one thread on physical CPU316. It evaluates frozen current-model W13 and W2 stage predictions; it does not fit parameters, adopt the previous affine/full-tail candidates, or change the active planner/model baseline.

Targets come from the same high-skew route/layer used in the order diagnosis. Counts and route inputs are unchanged:

| M | Expert |
|---:|---:|
|1|239|
|2|13|
|3|169|
|4|127|
|5|48|
|6|75|
|7|160|
|8|182|
|9|196|
|10|158|
|11|238|

Use the existing `isolate_bridge` dependency transformation: execute the target first on logical lane76/physical CPU316; every remaining task waits for target completion, and the original non-target lane dependencies remain. All expert work is retained. Each target is cold-first under the workload's standard copy-rotation/scrub protocol, rather than following the original plan's prefix. This is not a synthetic independent JIT harness. Moving some targets to the common CPU changes their placement relative to earlier joint traces; only targets already on CPU316 have exact CPU matching to those traces.

The frontier contains11 isolated plans plus the unchanged early-merge-off LPT anchor. Every isolated plan has a valid topological DAG and CPU exclusion. The existing isolation tests include the middle-of-lane splice case and rejection of background overlap. The trace parser verifies complete task-worker records and proves every non-target task starts after the isolated target finishes. A separate stream audit verifies actual target CPU316, zero early-merge events and all80 final-merge workers after scheduled compute.

Model: existing v8 calibration, SHA256 `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`, and frozen core-pressure coefficients. Isolated stage predictions come from `predict_expert(M,1)` before measurement; no competing GEMM is present during the target. The inherited `operator` residual is recorded separately and is not labelled as a gather prediction. Stage errors must not be silently substituted for full-expert `T_iso` error.

## Measurement method

- Host `Arm-codex-internal`, root `/home/zhangxu/codex/fused_cpp`; process affinity CPUs240–319, memory NUMA3, isolated target CPU316.
- H4096/F512/E256/T2048/top-k6, BF16 SVE256, backend N tile16, FP32 route output,192MiB pretouched route-output workspace, four rotating weight copies and216MiB scrub. Same existing binary/page policy as the prior experiments; no build or installation.
- Full packed-B W13=8MiB and W2=4MiB per expert. Target width1 has the same per-worker footprints; geometry `(t,w13_window_tiles,w2_window_tiles,R13,R2)=(1,0,0,1,1)`.
- `OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=false MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`; runner configures the80-worker pool after initial affinity validation.
- Two independently initialized sessions, each5 warmup and31 measured randomized, copy-matched rounds across all12 plans. Seeds599001/599002. A no-trace control uses599001 and the same warmup/sample counts.
- Four-copy output correctness precedes measurement. Trace order: anchor preflight, four correctness calls per plan, then36 rounds;481 calls per session. Target-stage timing excludes the remaining workload.
- Report both session medians, signed `(prediction/actual - 1)` error, absolute error, P10/P90 and CV. Equal-weight stage summary uses11 M values ×2 sessions. Retain all samples and distinguish session drift from repeatability within a session.
- Untraced whole-call controls are endpoint-sensitivity evidence only. They do not provide an untraced target-stage timestamp or a pure trace-overhead correction.

One expert per M does not measure input/expert variability at fixed M, and one CPU/machine does not establish cross-machine accuracy. All points are evaluation observations; no post-measurement training or correction is performed.

## Reproduction and artifacts

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/prepare_small_m_isolated.py \
  --input tmp/model_search_compare_20260909/input.json \
  --source tmp/mean_model_compare_20260910/high_skew/frontier.json \
  --output tmp/small_m_isolated_20260910
.venv/bin/pytest -q tests/test_moe_workspace_isolated_width.py tests/test_moe_small_m_isolated.py
ssh Arm-codex-internal 'cd /home/zhangxu/codex/fused_cpp && bash tmp/small_m_isolated_20260910/run_remote.sh'
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_small_m_isolated.py \
  --root tmp/small_m_isolated_20260910
```

Use a fresh destination when rerunning; preparation and final report refuse existing outputs. The retained shell script invokes the existing streaming isolation parser after each session and keeps compressed raw traces. Source snapshots, frozen predictions, frontier, full metadata, compact per-target samples and final report are retained under `tmp/small_m_isolated_20260910/` locally and on the stated remote host/root.

The source tree is commit `c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5` plus existing dirty work; `status_before.txt` records the pre-experiment status. The extension remains `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`, workspace profile `cdccff46365f217ac7984ea168da1d60731922ade9a9757d171bb95c3414ce41`, runner `ea81113c747c509b8556d5c9a698fd2bf85a8d01a4c8fc775ab9657f107219ba`. Native/runtime defaults are unchanged and no commit was made.
