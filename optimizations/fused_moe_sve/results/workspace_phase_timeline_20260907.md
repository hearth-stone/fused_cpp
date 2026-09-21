# Workspace phase timeline: two-session fixed-plan diagnosis

## Status

All four sessions completed, retrieved and validated. Connectivity interrupted
retrieval, not the remote benchmark loop. On reconnection all JSON/log files
were present and no benchmark process remained; no experiment was repeated.

Complete results are in `median_analysis.json` and `high_skew_analysis.json`
under the artifact directory below. All trace hashes, round/worker counts,
correctness flags, parent workspace identities and frozen placed predictions
pass validation. There are434 non-warmup plan calls (7 plans×31×2), plus
70 warmup calls and60 correctness/reference calls across four sessions.

Decision: the median inversion includes a reproducible false critical-lane
switch. High-skew shows opposing wide/narrow lane errors: overpredicted16T
GEMM chains and underpredicted1T chains. This identifies the next calibration
audit, not a uniquely established DDR mechanism or permission to fit a penalty.

## Fixed scope and methodology

No new candidates, physics terms, model fit or winner selection. Seven exact
existing bridges selected by `prepare_workspace_phase_timeline.py`:

- median: old anchor, elite2ab, existing explicit smooth;
- high-skew: old smooth anchor, full_reference, greedy, vnd_greedy.

Sources: `tmp/workspace_baseline_confirmation_20260907/smoothing.json` and
`high_skew.json`. The selector preserves source plan records verbatim and records
source hashes/key mappings. Historical controls are not new current anchors.

Baseline: `profiles/workspace_numa3_80c.json`; fixed2048-token pre-touched
workspace, four weight copies,216MiB scrub,5 warmups,31 paired rounds,2 seeds
20260911/20260912. Same-process weights/input/allocations within each session;
candidate order randomized each round. E256/H4096/F512,2048tokens,TopK6,BF16,
merge on; backend Ntile16, widths as in fixed bridges, W13/W2 full weight bytes
8MiB/4MiB per expert, full owner stripes `(t,0,0,1,1)`.
NUMA3/CPU240–319 on Arm-codex-internal, no kernel rebuild. Frozen extension SHA
dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f;
calibration SHA7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3.
HEAD c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5 with existing dirty workspace;
only Lab runner/analyzer/tests/manifest changed for this task.

The sole deliberate protocol deviation is `--phase-trace`. Sessions explicitly
record `diagnostic_phase_trace=true` and trace hash. Their baseline hash denotes
the parent workspace protocol, not trace-off equivalence. Ordinary winner
analysis rejects these sessions. Even explicit diagnostic validation produces
no accepted winner list. Trace-on timings must remain separate from trace-off
performance and partial-order calibration evidence.

The first unmeasured reference call allocates output normally, followed by all
plans×4 poisoned-workspace correctness calls, then36 measured/warmup rounds.
The parser excludes the correctness prefix and first5 rounds, validates every
task's worker ids/expert ids and three stage records. Timestamps are relative
to `scheduled_compute` start; stage envelopes are earliest worker start to latest
worker end. Quantiles of stages/starts/ends are computed separately and need not
sum exactly. Gather includes the traced stage's synchronization behavior; this
is not a pure memory-service counter.

## Median session1 decomposition

| Plan | Model final expert | Actual final expert in31/31 rounds | Model completion ms | Actual final W2 ms | Native e2e ms |
| --- | --- | --- | ---: | ---: | ---: |
| old anchor | 167, M1 | 167, M1 | 34.116 | 30.541 | 31.373 |
| elite2ab | 115, M714 | 167, M1 | 35.453 | 29.988 | 30.855 |
| smooth | 254, M23 | 254, M23 | 34.106 | 29.757 | 30.821 |

These are complete placed-event model predictions, not just isolated lane-load.
Native e2e includes work outside the expert compute interval; do not compare it
to model compute time as a stage-specific residual. No new speedup gate is
claimed from these trace-on numbers.

For elite2ab, M714 is on lane begin16/width16:

| Timestamp/duration | Placed model ms | Hardware session1 median ms |
| --- | ---: | ---: |
| Task start | 26.520 | 19.760 |
| Task end | 35.453 | 26.772 |
| W13 duration | 6.111 | 4.567 |
| W2 duration | 2.822 | 2.200 |
| Gather duration | no explicit phase | 0.222 |

Most of the8.68ms end-time discrepancy is already present at task start
(6.76ms), rather than entirely being M714's own runtime. The hardware final
W2 remains on lane0 at29.988ms, about3.2ms later than M714. The model incorrectly
switches the critical lane to lane16. Expert167 is the terminal task of a chain,
not evidence that this tiny expert by itself causes the whole delay.

Further stage accounting on elite lane16 (median of per-round sums):

- Model W13/W2 sums:23.846/11.607ms.
- Hardware W13/W2 envelope sums:17.150/8.308ms, gather sum1.081ms.
- Hardware lane finish:26.772ms.

Frozen v8 has `gather_pressure.enabled=false` for this calibration. This is a
phase-accounting gap, not proof that no gather cost was absorbed elsewhere.
Here the model nevertheless **overpredicts** GEMM stage totals substantially;
adding a positive gather penalty alone would not fix the observed discrepancy.
Whether the excess comes from isolated calibration or concurrency dilation
requires further separation; these full-workload traces do not measure isolated
hardware task service times.

## Two-session confirmation

### Median: false critical-lane switch, not a late M714 hardware tail

| Plan | Actual final W2 ms, s1/s2 | Native e2e ms, s1/s2 | Hardware final lane/expert |
| --- | --- | --- | --- |
| old anchor | 30.541 /30.762 | 31.373 /31.612 | lane0 /expert167,62/62 |
| elite2ab | 29.988 /30.031 | 30.855 /30.927 | lane0 /expert167,62/62 |
| smooth | 29.757 /29.979 | 30.821 /31.100 | lane0 /expert254,62/62 |

Elite M714 starts19.760/19.763ms and ends26.772/26.785ms, versus placed
prediction26.520→35.453ms. Its lane has median per-call slack3.188/3.124ms.
The model predicts that lane16 becomes critical; hardware retains lane0 in
both sessions. Lane0 has the same task ownership/order in old anchor and elite;
its observed end shifts earlier by roughly0.55/0.73ms (differences of medians).

For old anchor → smooth, the actual critical lane0 gather-envelope sums fall
1.574/1.623→0.923/0.951ms; W13 sums fall18.214/18.274→17.985/17.971ms;
W2 sums fall8.863/8.830→8.808/8.768ms. Gather's traced stage accounts for a
large observed change here, while v8 has no explicit gather phase. These are
stage-envelope differences, not additive causal estimates; synchronization and
inter-stage gaps remain outside this attribution. The point prediction changes
only34.116→34.106ms and misses most of the observed lane-time change.

Trace-on native paired gains versus old anchor are1.604%/2.148% for elite and
1.574%/1.297% for smooth. These differ from trace-off gains, so they do not
replace the prior winner evidence or justify a new smoothing adoption.

### High-skew: wide/narrow lane ordering is systematically wrong

| Plan | Model makespan ms | Actual final W2 ms, s1/s2 | Native e2e ms, s1/s2 |
| --- | ---: | --- | --- |
| old smooth anchor | 39.511 | 29.137 /28.406 | 30.190 /29.330 |
| full_reference | 35.294 | 29.099 /28.641 | 30.126 /29.604 |
| greedy | 39.649 | 33.243 /33.313 | 34.016 /34.082 |
| vnd_greedy | 38.328 | 32.108 /32.151 | 32.895 /32.912 |

Old smooth anchor:

- Model final: lane32/16T, expert135(M1341), end39.511ms.
- Hardware that lane ends27.048/26.892ms, with positive slack1.705/1.523ms.
- Hardware final: lane69/1T, expert63(M10),61/62 calls; lane68/1T,
  expert208(M20),1/62. Lane69 ends29.137/28.406ms; model end25.925ms.
- Lane32 model W13/W2 sums26.795/12.716ms; hardware17.190/8.359 and
  17.147/8.343ms. Lane69 model W13/W2 sums15.203/7.649ms;
  hardware19.235/9.002 and18.742/8.961ms.

Thus the model both overestimates the wide chain and underestimates the narrow
chain. The model lane69 finish also contains non-W13/W2 time; do not sum only
the two listed model GEMM values and treat that as its complete lane duration.

Full_reference:

- Model final: lane48/16T, expert171(M1239), end35.294ms.
- Hardware lane48 ends25.962/25.901ms, with slack3.182/2.728ms.
- Actual terminal lanes vary: s1 lane68(23),70(6),69(1),32(1);
  s2 lane0(16),68(9),70(6). The model-predicted lane48 is never terminal.
- Hardware full/smooth remains close; native paired full gains are
  +0.440%/-0.864%, not the predicted+11.950%. A modeled improvement on the
  wrong bottleneck creates an exaggerated plan-level gain.

Greedy → VND:

- All lanes are16T. Compute completion improves33.243→32.108ms in s1 and
  33.313→32.151ms in s2. Model also improves39.649→38.328ms.
- Actual terminal lane is mostly16 in s1,0 in s2 for both plans. Model predicts
  lane64 for greedy and32 for VND. Do not turn a single-session terminal lane
  into a fixed hardware label.
- Example lane16 W13/W2 sums fall19.718/9.933→19.045/9.598ms in s1 and
  19.760/9.946→19.044/9.603ms in s2. The improvement is distributed along
  the chain, not solely a change to its final M1 expert.
- VND remains slower than the strong old smooth anchor in both sessions
  (native paired gains-8.870%/-10.762%), despite the model predicting+3.088%.

### Interpretation and limits

The reliable finding is relative phase/lane timing bias and wrong bottleneck
selection, not a uniquely identified physical cause. Full-workload traces cannot
separate inaccurate isolated stage calibration from inaccurate concurrency
dilation. Gather stage envelopes also include whatever work/synchronization the
native trace boundary contains. Trace instrumentation changes small plan gaps;
the hardware ranking/adoption source remains the previous trace-off sessions.

The next useful audit is width-stratified isolated phase validation (especially
1T vs16T), followed by comparing measured concurrency increments for these same
tasks. Do not add a new positive gather or DDR penalty to an already overpredicted
wide GEMM chain, and do not replace critical-path reasoning with density alone.

## Artifacts and commands

Local/remote relative directory: `tmp/workspace_phase_timeline_20260907/`.
Remote root: `/home/zhangxu/codex/fused_cpp` on Arm-codex-internal.
Runner snapshot: `runner/bench_bounded_order_extension.py` and
`runner/fixed_route_workspace.py`; no production mirror source overwritten.

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/prepare_workspace_phase_timeline.py \
  --input-dir tmp/workspace_baseline_confirmation_20260907 \
  --output-dir tmp/workspace_phase_timeline_20260907
```

Already executed remotely for median/high_skew and session1/2 sequentially:

```bash
OMP_NUM_THREADS=80 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/workspace_phase_timeline_20260907/runner/bench_bounded_order_extension.py measure \
  --repo-root /home/zhangxu/codex/fused_cpp \
  --frontier tmp/workspace_phase_timeline_20260907/${trace}.json \
  --route-file "$route" \
  --experiment-baseline tmp/workspace_phase_timeline_20260907/workspace_numa3_80c.json \
  --phase-trace tmp/workspace_phase_timeline_20260907/${trace}_session${session}.log \
  --output tmp/workspace_phase_timeline_20260907/${trace}_session${session}.json \
  --seed $((20260910 + session))
```

Route median: `bench_assets/moe_paper/dsv4_routes_pt_20260830/measured_request016_case017_zh2048-018.pt`, layer4.
Route high_skew: `bench_assets/moe_paper/dsv4_routes_pt_20260830/measured_request008_case009_zh2048-010.pt`, layer38.

Verified local single-session parse:

```bash
PYTHONPATH=.:src .venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_transfer_task_trace.py \
  --frontier tmp/workspace_phase_timeline_20260907/median.json \
  --session tmp/workspace_phase_timeline_20260907/median_session1.json \
  --trace tmp/workspace_phase_timeline_20260907/median_session1.log \
  --output tmp/workspace_phase_timeline_20260907/median_s1_check.json
```

Completed for both traces using the new analyzer with `--frontier`,
`--sessions <session1.json> <session2.json>`, `--traces <session1.log> <session2.log>`,
`--route-frontier tmp/moe_partial_order_vnd_20260904/${trace}_template_lns_frontier.json`,
`--calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json`,
and a fresh `--output` path:
`PYTHONPATH=.:src .venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_workspace_phase_timeline.py ...`.

## Next decision

Collection and two-session analysis are complete. Retain the fixed-plan raw
traces and summaries for a width-stratified calibration audit; no additional
hardware work or model fitting was performed after reconnection. A subsequent
isolated probe must use the same workspace/output lifecycle and report gather,
W13 and W2 separately. Only after this can the remaining full-workload increment
be attributed to concurrency modeling rather than baseline stage error.

## Validation

29 tests passed:
`PYTHONPATH=.:src .venv/bin/python -m pytest -q tests/test_moe_transfer_task_trace.py tests/test_moe_bounded_order_extension.py tests/test_moe_fixed_route_workspace.py`.
Ruff passed on touched scripts/tests. New checks cover stage envelopes, placed
event interval accumulation and trace rejection by winner analysis.
Two-session target evidence is now complete. All four sessions match the parent
workspace profile and actual Ntile16; full trace hashes and parsing pass.
The29 unit tests/Ruff are the previously completed unchanged-code validation;
this continuation added artifact analysis and documentation only.
No commit or production change.
