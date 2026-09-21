# Full MoE replay with frozen core-pressure response

## Outcome

Completed offline full-plan replay on four retained workspace samples:46 plan
entries and eight hardware sessions,92 per-plan session medians. Baseline MAE
3.733ms/MAPE12.018%; Lab adapter MAE1.143ms/MAPE3.652%, max3.397ms.
Uniformish regresses; do not hide it behind the aggregate improvement. These are
selected historical samples with repeated plans across groups, not92 independent
holdouts, a new hardware run, or production adoption.

The user chose to continue the current approach and inspect overall error.
Current pressure version is the simpler core-count response from the team-demand
trial. This task adds an EXPLICIT full-plan memory-path extrapolation because
that local model did not previously define a complete MoE predictor. It is not
merely a rerun of an already integrated model.

Class M Lab-only model semantics: new standalone subclass/replay, focused tests,
manifest/report and a bounded mathematical-model note. Default planner, native
runtime, calibration and prior working baseline unchanged; no commit or remote
write. Existing unrelated dirty/untracked changes preserved.

## Adapter definition and limits

For each active GEMM phase, count active peer GEMM cores in each LLC domain,
excluding its own team. For spanning teams take the largest local peer count.
Use the frozen response f=1+0.28902964424300354*x+0.4536242509371087*x^2,
x=peer cores/32. No full-plan timing, target cycles, or measured phase duration
enters these features. The measured314.93us M1 anchor is NOT used as a universal
expert duration.

Start from the frozen v8 isolated phases and global resource calculation:
- preserve domain cache-spill fractions and global resource dilations;
- replace GEMM LLC scale with f;
- set GEMM DRAM scale to max(global DRAM dilation,f);
- retain max(compute,transfer) composition and epilogue/fixed costs;
- replace the old concurrent wide/narrow correction, keeping isolated width scale;
- preserve baseline behavior for a single active phase and non-GEMM phases;
- preserve DAG dependencies, phase progression and full-call overhead accounting.

This is an extrapolation of a total M1 response onto memory-path scales, not an
independently validated physical derivation. It simultaneously replaces older
local-response components, so improvement cannot be uniquely credited to the
core-count feature. The original prototype covered M1 W13 with M12 backgrounds,
widths1/2/4 in a steady-state loop. Evaluation uses routes M1–1945, widths1–16,
W13/W2, two LLCs and full-forward workspace execution. Maximum local peers are39
(versus38 directly measured); other sizes/types and cold/cache behavior remain
material transfer gaps. Mathematical definition recorded in
`cpu_moe_schedule_optimization/MATHEMATICAL_MODEL.md`.

## Data and validation procedure

Reuse `tmp/workspace_archive_replay_20260907/`:
`vnd_high_skew`, `lns_median`, `lns_high_skew`, `lns_uniformish` JSON frontiers
and their session1/session2 JSON files. Respect exact Plan V2 bridges and source
frontier route counts; no candidate regeneration, selection expansion or fit.

The existing reader validates full5-warmup/31-measured pairing, four-copy rotation,
frontier/extension/workspace identity, independent seeds and bitwise correctness.
Source frontier and calibration hashes are checked. Before scoring each candidate,
recompute baseline and require agreement with its preserved prediction within
0.001ns; all46 pass. Frozen team checkpoint bytes remain unchanged. Raw identities
and all individual predictions are retained in the new report JSON.

Original hardware protocol: Arm-codex-internal NUMA3, CPU240–319/membind3,
E256,H4096,F512,2048tokens,TopK6,BF16,SVE256,Ntile16, full stripes(t,0,0,1,1),
W13/W2 bytes8MiB/4MiB per expert. Fixed192MiB pretouched output workspace,
216MiB scrub,4-copy rotation, random plan order, five warmups and31 paired
rounds per session, phase trace OFF. Original native output correctness passed
for all retained plan/copy combinations. Source/build and methodology caveats:
[original archive audit](arm_codex_80c_workspace_model_search_audit_20260907.md).

Frozen calibration SHA256
`7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`;
measured extension SHA256
`dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`.
This task did not build or run that extension, change page policy or remeasure
its overhead. No claim that native team-demand loop and full-forward lifecycles
are equivalent. Original experimental samples were selected by historical roles,
including prior good/bad candidates; this is retrospective diagnostic coverage.

## Full-forward absolute error

Each group pools its two per-plan session medians with equal observation weight.
MAE and max are |prediction-measurement| in milliseconds. MAPE is per-observation
percentage error averaged, not MAE divided by pooled time.

| Sample | Plan entries | v8 MAE ms | Adapter MAE ms | Adapter max ms | v8 MAPE | Adapter MAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VND high-skew | 13 | 4.357 | 0.953 | 2.030 | 13.529% | 2.912% |
| LNS median | 13 | 3.465 | 1.055 | 3.117 | 11.021% | 3.288% |
| LNS high-skew | 11 | 5.412 | 1.003 | 1.355 | 18.074% | 3.358% |
| LNS uniformish | 9 | 1.166 | 1.718 | 3.397 | 3.871% | 5.606% |
| All sample entries | 46 | 3.733 | 1.143 | 3.397 | 12.018% | 3.652% |

Per-session adapter MAE S1/S2: VND high-skew0.905/1.000ms;
LNS median1.045/1.066ms; LNS high-skew0.997/1.010ms;
uniformish1.725/1.711ms. The uniformish regression repeats.

Anchor examples (ms):

| Group anchor | Measured S1 / S2 | Frozen v8 | Adapter |
| --- | ---: | ---: | ---: |
| VND high-skew | 29.771 / 29.796 | 35.294 | 31.244 |
| LNS median | 30.432 / 30.496 | 34.116 | 29.908 |
| LNS high-skew | 30.310 / 30.336 | 35.294 | 31.244 |
| LNS uniformish | 30.036 / 29.886 | 31.125 | 28.426 |

Uniformish is systematically underpredicted (mean bias-1.718ms). The first and
third anchor share the same prediction but were independently executed under
different sample sessions; this illustrates why pooled counts are not independent
plan coverage. Do not describe error reduction as execution speedup.

Phase trace was disabled in these timing sessions. Total error is measured;
individual gather/W13/W2/merge error cannot be uniquely decomposed from this
sum. The unchanged stages are not thereby proven accurate, and cancellations
between stage errors remain possible.

## Selection loss, separately from absolute accuracy

Loss is measured time of predicted-best minus fastest MEASURED sample plan,
using each session's medians. It is sample-relative, not regret against all
possible schedules. No tie/noise policy or pruning is introduced.

| Sample | Baseline loss S1 / S2 ms | Adapter loss S1 / S2 ms |
| --- | ---: | ---: |
| VND high-skew | 0.127 / 0.087 | 0.059 / 0.066 |
| LNS median | 1.251 / 1.217 | 0.780 / 0.567 |
| LNS high-skew | 0.015 / 0.180 | 0.099 / 0.226 |
| LNS uniformish | 0.000 / 0.166 | 0.109 / 0.000 |

Absolute accuracy improves strongly on LNS high-skew while its selected-plan
loss increases. Small losses are not claims of statistically significant winner
differences. LNS median still selects the anchor rather than the measured elite;
a1ms-average absolute model does not remove the need for hardware reranking or
uncertainty-aware ties.

## Reproduction, review and next boundary

```sh
.venv/bin/pytest -q tests/test_moe_core_pressure_replay.py
.venv/bin/python optimizations/fused_moe_sve/benchmarks/replay_core_pressure_moe.py \
  --output tmp/core_pressure_full_moe_20260909/report.json
```

Use a new output filename for replay; exclusive creation preserves the original.
Three focused tests pass for self/domain exclusion, spanning-team peer counts,
and separate absolute-error versus selection-loss accounting. All46 baseline
predictions reproduce with zero reported drift; source/session/bridge integrity
checks pass. Ruff/diff checks pass. No code or data fitting after result inspection.
Implementation uses a Lab subclass of the existing event simulator; it does not
register a new supported model, profile or planner mode.

The requested first overall replay is complete. Retain the roughly1ms MAE result
and uniformish failure as a working diagnostic, not proof that the overall model
is calibrated. A future independent full-forward dataset, especially uniformish,
and phase-specific evidence would distinguish transfer failure from inherited
isolated-stage error. No new measurements or further fitting were started here.


## Regression-case diagnosis (same frozen version, no refit)

Subsequent hardware stage/task tracing is recorded in
[per-case diagnosis](core_pressure_case_diagnosis_20260909.md). It supersedes
model-only hardware interpretations below: uniformish GEMM remains overpredicted
and unrepresented gather/gap/outer costs explain much of the exposed bias;
high-skew p11 instead shows a concrete1T M13 context-response failure.

Follow-up examines cases whose absolute error increased, with the user noting
that uniformish MAE1.166→1.718ms is a modest aggregate difference (~0.55ms,
roughly1.8% of a30ms forward), not a reason by itself to reject the current
working model. All cases stay in the official score.

Using the mean of two measured medians for readable signed-error examples:

| Group / case | Measured ms | Old prediction | Current prediction | Old error | Current error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniformish anchor | 29.961 | 31.125 | 28.426 | +1.164 | -1.535 |
| Uniformish p01 | 32.553 | 32.755 | 29.217 | +0.202 | -3.336 |
| Uniformish p03 | 30.544 | 31.182 | 28.452 | +0.638 | -2.092 |
| Uniformish p06 | 29.641 | 30.787 | 28.349 | +1.146 | -1.292 |
| VND high-skew p11 | 35.997 | 35.340 | 34.026 | -0.656 | -1.971 |

Seven of nine uniformish entries worsen, but most increases are small.
Retrospective concentration diagnostic ONLY: excluding p01 yields MAE1.286→1.516ms;
excluding p01+p03 yields1.379→1.433ms for the other seven. Thus these two cases
account for about0.510ms of the0.552ms net average-error increase. This does not
change sample eligibility, fit data, official metrics or acceptance gates.

Counterfactual replays restore one old model component at a time; they perform
no calibration and do not become new candidates. The model-level explanation
is the replacement of the old concurrent wide/narrow team response, combined
with new memory scales. Uniformish uses4/8T, with anchor/p01/p03/p06 all8T;
it is outside the original1/2/4T calibration domain. The frozen old8T isolated
scale is1.144 and full-cohort scale1.302563. The adapter keeps the former but
removes the concurrency interpolation, then applies core pressure only to the
memory bounds. Compute-dominated portions therefore do not receive that old
whole-phase concurrency factor.

Ordered diagnostic for p01: baseline32.755ms; removing only old concurrent team
correction gives28.783ms; the current memory-path replacement then gives29.217ms.
The same ordered path is31.125→27.357→28.426ms for anchor, and
31.182→27.404→28.452ms for p03. These are model intervention differences with
resimulated schedules, not physically measured stall components.

Keeping the current adapter elsewhere but restoring OLD W13 multipliers gives:
anchor30.288ms, p0131.529ms, p0330.257ms. Restoring only W2 gives29.258,
30.487,29.270ms respectively. W13 is the larger model-side sensitivity here;
these are not measurements of actual W13 error because phase trace is OFF.
Restoring the old team factor everywhere overshoots many uniformish cases
(anchor32.351ms), so blindly restoring it is not established as a general fix.

The mixed1T/16T VND p11 has a different interaction: restoring the old team
factor atop current memory response gives36.013ms, close to measured35.997ms.
Removing only the old team correction from baseline instead gives39.365ms,
showing competing narrow/wide and memory-response effects rather than one
universal missing constant. Do not tune to this one case.

Raw variability cannot explain away the largest misses: p01 P10–P90 spans
32.313–32.835ms inS1 and32.283–32.888ms inS2, while current predicts29.217ms.
p03 spans30.156–30.818 and30.079–30.796ms, versus prediction28.452ms.
Across these repeats round CV is about0.67–0.88%. Small aggregate regression
and a systematic local miss can both be true.

Decision relevance remains limited: uniformish predicted-best p06 is only
0.109ms slower than sample-best inS1 and is sample-best inS2. p01 is still
predicted slower than p06 (29.217 vs28.349ms); VND p11 remains slower than its
selected candidate. No observed false winner follows from these particular
large absolute misses. Preserve current version and treat them as bounded
absolute-bias diagnostics, not an automatic blocker or a reason to refit now.

Reproduce the component interventions with
`.venv/bin/python tmp/core_pressure_full_moe_20260909/diagnosis/diagnose.py`.
It reads preserved full-plan source/measurements and writes exclusive
`diagnosis/report.json`; use a fresh output path on rerun. All11 diagnostic
cases (nine uniformish plus VND p11/anchor) completed. Their baseline/current
predictions remain the previously verified ones. No hardware, production code,
calibration, current adapter or frozen model was changed in this diagnosis.
