# Early-merge-off joint-error experiment, 2026-09-09

## Result

Completed two31-round trace sessions and one31-round no-trace control on the original Arm machine. All seven experimental plans explicitly disable early merge. The p11 M13/W13 joint-increment underprediction falls from0.960/0.974ms to0.162/0.107ms in sessions1/2, while M17/W2 and M35/W13 retain substantial errors. This supports early merge and its associated execution timeline as a major contributor to the M13 discrepancy; it does not allocate a precise hardware service penalty to merge alone.

For the five selected p11 experts across W13/W2, mean absolute joint-increment error falls from0.183/0.178ms to0.097/0.092ms. These are10 selected stage observations per session, not full-forward error or a representative population score. No parameters were fitted.

Class E diagnostic. Production defaults, kernels, planner implementation and model coefficients are unchanged. The experiment uses explicit `early_merge=false` in new bridges; it does not globally change generated-plan defaults. The native explicit-off policy overrides the runner's merge environment variable.

## Protocol and comparison definition

Use the original expert-layer-split workload: anchor, p11 and cold-first isolated experts M13/17/31/35/65, with task order, dependencies, ownership and CPU placement unchanged apart from the already-existing isolation rewiring. Target p11/isolation CPU316; anchor M65 CPU308 remains an unmatched comparison. Primary attribution uses matched p11 rows.

Arm-codex-internal, `/home/zhangxu/codex/fused_cpp`, existing virtualenv, NUMA3/membind3, CPUs240–319. BF16/SVE256, Ntile16, H4096/F512,2048 tokens,TopK6,E256,4 weight copies,216MiB scrub, fixed192MiB route-output workspace. Full owner stripes;1T target geometry `(t,w13_window_tiles,w2_window_tiles,R13,R2)=(1,0,0,1,1)`, full W13/W2 owner bytes8/4MiB. Competitors retain the original1T/16T mixed teams.

Two trace seeds593001/593002, each5 warmups plus31 measured randomized, copy-matched rounds. No-trace control seed593001. All runs perform four-copy full-output equality checks before measurement. Stage point values are medians. Reacquire isolated I and joint J with merge off; never subtract an old merge-on isolation from a new off joint stage.

With unchanged model estimates Ihat/Jhat:

```text
base_error = Ihat - I
increment_error = (Jhat - Ihat) - (J - I)
total_error = Jhat - J = base_error + increment_error
```

Negative errors mean underprediction. Increment intervals use the existing2000-resample paired-round bootstrap. On/off measurements are historical separate-process comparisons, not randomized within-process on/off pairs. Disabling early merge changes both the workload overlapping a stage and the resulting time course. The five isolated costs, two independent sessions and low-merge-overlap stages serve as controls; they do not remove every process/time confound.

The model is the frozen core-pressure adapter. The separate full/tail table has not been integrated into this real mixed-environment prediction and is not silently substituted here.

## Joint-increment errors after removing early merge

Second-session signed errors, ms:

| M | W13 on | W13 off | W2 on | W2 off |
|---:|---:|---:|---:|---:|
|13|-0.974|-0.107|+0.006|+0.030|
|17|-0.111|-0.116|-0.182|-0.162|
|31|-0.039|-0.040|-0.021|-0.007|
|35|-0.328|-0.316|+0.003|+0.019|
|65|-0.091|-0.095|-0.029|-0.031|

The improvement is concentrated in M13/W13. Session1 off W13 errors for M13/17/31/35/65 are-0.162/-0.112/-0.035/-0.325/-0.070ms; M17/W2 remains-0.186ms. The qualitative result repeats.

Second-session p11 W13-only joint-increment MAE falls0.309→0.135ms; W2-only MAE is0.048→0.050ms. Thus “joint error improves” is supported for the selected combined set, but does not describe a uniform improvement across stages.

### M13 changes substantially, but is not solved

| Session | On isolated | On joint | Off isolated | Off joint | Off measured increment | Model increment |
|---:|---:|---:|---:|---:|---:|---:|
|1|1.539|2.858|1.576|2.097|0.521|0.359|
|2|1.564|2.897|1.588|2.054|0.467|0.359|

All times are ms. Joint time falls0.761/0.843ms; accounting for the newly measured isolation gives joint-error reductions0.798/0.867ms, approximately83%/89%. This is error reduction under the off protocol, not a causal percentage assigned exclusively to merge traffic.

Off paired median-increment95% intervals are[0.493,0.534]ms and[0.441,0.482]ms; the modeled increment0.359ms is still below both. In session2, total W13 underprediction0.356ms now splits into0.249ms base error and0.107ms increment error. Retaining that split prevents treating all remaining error as contention.

### M35/W13 and M17/W2 remain distinct targets

M35/W13 off session2 has measured increment0.356ms versus model0.040ms, interval[0.352,0.366]ms. Its remaining error0.316ms is the largest selected joint-increment miss. M17/W2 has measured increment0.270ms versus model0.108ms, interval[0.249,0.300]ms. Neither discrepancy disappears when early merge is removed.

M17/W13 also remains0.116ms low. Conversely, M35/W2 measured increment0.001ms has interval[-0.028,0.020]ms, consistent with little added cost. Raising a universal contention factor would target the wrong layer for that stage. Existing [joint-environment diagnosis](joint_environment_diagnosis_20260909.md) identifies mixed W13/W2 background and1T/16T ownership as discriminating controls; these mechanisms remain hypotheses pending controlled background experiments.

## Overall compute completion and end-to-end time

Compute completion is measured from scheduled-compute start to the last expert W2 end. It includes actual gather, team synchronization and scheduling gaps before that endpoint, and excludes final merge. The frozen model has zero call setup and no explicit gather phase, so agreement at this endpoint does not independently validate its omitted/absorbed costs.

| Plan | Model compute estimate | On compute S1 / S2 | Off compute S1 / S2 | Off signed error S1 / S2 |
|---|---:|---:|---:|---:|
|anchor|31.244|28.697 /28.590|27.850 /28.035|+3.394 /+3.209|
|p11|34.026|34.974 /34.894|33.903 /33.881|+0.123 /+0.145|

Values in ms. p11's computed completion becomes close, but anchor is still overpredicted by about11–12%. This is not evidence that the complete model is accurate. p11's final expert is242 in all62 off samples; anchor's final expert is230 in session1 and235 in session2, compared with mixed231/236/230 in the on sessions.

Final merge costs about1.06–1.07ms in the off traces. No-trace full-forward medians are:

| Plan | Historical on (ms) | New off (ms) | Change (ms) |
|---|---:|---:|---:|
|anchor|29.803|29.722|-0.081|
|p11|35.857|35.947|+0.091|

The whole-forward changes are small, below0.3%, and do not establish an end-to-end improvement. Earlier compute completion is offset by the final merge being performed afterward. Trace-internal native total minus no-trace Python elapsed ranges-0.348 to+0.254ms across all seven cases and both trace sessions. Different timer endpoints/processes prevent interpreting that range as pure instrumentation overhead. Outer trace-enabled Python time includes logging and is not used for stage labels.

## Integrity and reproducibility

All three runs report bitwise correctness for all four copies/seven plans, complete252 timing rows each, and empty stderr. Both traces contain281 complete calls (correctness/prefix plus warmup/measured calls),562 total. Across **every** call there are zero `merge_ready_token` events,80 final-merge worker records, and final merge starts after scheduled compute; minimum margins0.09032/0.09186ms.

All310 measured isolated targets finish before any background expert starts; minimum margin0.00067ms. Isolated stage CV ranges0.254–3.114%. Source, frontier, extension, workspace and trace hashes, round/copy ordering and unchanged model estimates were checked. All seven off bridges differ from their source only in the early-merge field and retain valid DAG/CPU exclusion.

Runner SHA256 `ea81113c747c509b8556d5c9a698fd2bf85a8d01a4c8fc775ab9657f107219ba`; workspace source `7e0124673b4742a4747f6b4b733cb364def2f455cc5919ef13379eca4cb098db`; profile `cdccff46365f217ac7984ea168da1d60731922ade9a9757d171bb95c3414ce41`; extension `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`. These match the original on sessions. The runner itself verifies extension and route identity before numerical checks and timing. No build or installation occurred.

Local/remote measurement artifacts: `tmp/early_merge_off_20260909/`; raw traces plus `.gz`, session JSON/logs and control JSON/logs retained. Local companion files: `prepare.py`, `preflight.json`, `compare.py`, `comparison.json`, `old_endpoints.py`, `on_endpoints.json`, analysis stdout. Historical latency and selection labels were removed from the new frontier; retained state hashes are explicitly source-geometry lookup identities, not the off bridge identity.

Exact remote command: `bash tmp/early_merge_off_20260909/run_remote.sh` from `/home/zhangxu/codex/fused_cpp`. The retained script invokes the original runner snapshot with the profile, route file, affinity and seeds above. Use fresh paths for a new run; do not overwrite completed logs or frozen artifacts.

Executed local analysis:

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_expert_layer_split.py \
  --directory tmp/early_merge_off_20260909 \
  --output tmp/early_merge_off_20260909/report.json
.venv/bin/python tmp/early_merge_off_20260909/compare.py
.venv/bin/python tmp/early_merge_off_20260909/old_endpoints.py
```

Preparation passed seven bridge/hash/DAG/CPU-exclusion and frozen-timeline checks. Comparison passed562-call merge-state/ordering checks, native identity/control checks and40 unchanged model-row comparisons. Static review and `git diff --check` passed. Existing source/tests were reused unchanged; no unrelated code suite was rerun. The initial SSH block was resolved before measurements, and the remote run exited successfully before trace compression and retrieval.

## Decision and remaining work

Retain early-merge-off as the explicit configuration for this computational-model experiment. The requested off joint-error measurement is complete. Do not describe the experiment as a production/default change or a full-forward speedup.

Next focus on M35/W13, M17/W2 and M17/W13 response under mixed GEMM backgrounds, while keeping isolated base errors separate. Real full/tail timing still matters for the remaining M13 response. Anchor's aggregate overprediction also needs separate accounting before generalizing p11's accurate completion time. No extra fitting or planner rollout was performed.
