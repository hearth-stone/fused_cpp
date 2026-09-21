# Cost-model and historical search audit with resident route workspace

Completed: reuse prior workspace order/transfer evidence and run8 new hardware
sessions for4 retrospective VND/LNS samples. No model fit, search regeneration,
new physical probe, production change or winner-file overwrite. Conclusions
change selectively: some local moves become useful, some old improvements
disappear, template-LNS retains useful candidates, and a known false-pruning
counterexample survives.

## Geometry correction

Frozen v8 contains `kernel.backend_n_tile=16`; a tiny packed-weight query against
the same frozen extension returns `arm_sve_bf16,16`. Earlier workspace reports
that say Ntile8 used the Python legacy default as prose, not the actual packed
object field. They now carry correction notices. Runtime packing was16 throughout;
this is not a new execution geometry or a reason to rewrite measured numbers.

## 1. Cost model: absolute and relative errors diverge

Reanalyze the existing eight order frontiers and two transfer frontiers without
rerunning them. Deduplicate full executable bridges within each trace; average
repeated per-session median times for each plan. Compare frozen predicted time
against those means. Relative-gain MAE similarly deduplicates anchor/candidate
bridge pairs and averages recorded session-median gains before taking errors.
These are selected-suite diagnostics, not unbiased holdout or refit results.

| Trace | Unique plans/pairs | Absolute MAPE old→workspace | Relative-gain MAE old→workspace |
| --- | ---: | ---: | ---: |
| median | 25/24 | 4.228→10.966% | 3.552→1.629 percentage points |
| high-skew | 27/26 | 17.684→22.753% | 5.643→4.875 percentage points |

Removing first-touch changes the target runtime, not the frozen prediction.
Less contaminated ranking labels can coexist with worse absolute-time error.
Do not declare v8 calibrated for the new lifecycle or re-fit evaluation traces.

The newly measured archive samples likewise have absolute MAPE changes:
VND high-skew5.600→13.528%, LNS median4.776→11.016%, LNS high-skew10.067→18.074%,
LNS uniformish3.524→3.871%. These samples are deliberately selected by historical
roles/results. In addition, their old protocol differed in scrub/copy/input seeds;
these are new-method comparisons, not isolated workspace A/B effect estimates.

## 2. VND: a previously weak greedy move now passes

High-skew candidate `5920f0531b01...`, alias p06, is the previously forced
same-lane insertion with predicted+3.448%, lower+1.679%, relation=incomparable.
Relative to the greedy parent p02:

- Old paired medians:+1.461/+0.081%; not actionable.
- Workspace:+3.350/+3.377%, P10=+2.732/+2.886%; both sessions pass.

It is not the global winner: candidate time33.009/33.003ms, while full anchor
is29.771/29.796ms. The old sampled absolute best `4115ab99...` becomes essentially
tied/slightly slower than full (paired-0.191/-0.058%). Thus the old conclusion
that no local move yields stable improvement is not portable to every starting
point under the new method; it does not imply VND beats the strongest baseline.
The full anchor wins the measured sample twice despite model rank5/13.

## 3. Template-LNS: useful winners survive, exact ranking changes

### Median

Historical elite `2ab43572d14e...` (p08) remains sample-best in both sessions:
29.652/29.929ms versus full30.432/30.496ms. Paired gains2.660/2.044%,
P10=2.270/1.296%. Its frozen model rank is8/13.

This is a surviving counterexample to pruning: the old partial-order label is
`candidate_worse`, predicted-3.770%, upper-2.001%. Applying that label as a hard
prune would discard the new measured winner. The replay itself did not prune it;
it was deliberately retained as historical measured evidence. This disproves
safety for that extended-domain use of the old comparator, not a claim about
its original narrower calibration population.

Other old positives disappear or weaken: p06 vs its parent changes7.940/6.969%
to3.282/1.240% with second P10 negative; p07 changes3.751/3.365% to0.403/1.183%.
The old positive sentinel p11 changes3.870/3.934% to-1.964/-2.271%. Do not erase
these reversals or pool the old and new lifecycle residuals.

### High-skew

`189d70b00890...` remains useful:2.539/3.034%, P10=0.028/1.037% versus its
historical parent `4115ab99...`. Against full, gains are2.352/3.129%.

The sample-best shifts to elite `61ea6f0d4095...` (p06):29.404/29.211ms versus
full30.310/30.336ms. Its gains versus the original parent are3.020/4.253%,
and versus full2.613/3.827% with P10=0.866/2.029%. Model rank is6/11.
This retains template-level search value without asserting which member wins
the unmeasured106-plan original frontier. No automatic elite promotion occurred.

### Uniformish

Three LNS candidates still improve a weaker parent: p03 gains6.468/7.109%,
p04 gains7.162/7.084%, p05 gains6.754/7.765%, all with positive P10. Old gains
were around11.7–14.9%, so magnitude decreases substantially.

They do not establish improvement over full. Sample-best differs by session:
p02=29.582ms in S1 and p06=29.591ms in S2, against full30.036/29.886ms.
Neither has both-session >2%/positive-P10 gains over full. The old sample-best
`50a7d4bee8aa...` becomes slower than full by1.456/1.346%. Model ranks of the
new per-session sample-best are1/9 and2/9. No consensus new global winner.

## 4. What remains valid

- Keep hardware evaluation and candidate retention; point-model top1 misses
  sample-best in VND high-skew and LNS median/high-skew.
- Do not use old `candidate_worse` labels as guaranteed safe pruning in the
  expanded LNS/workspace domain: median supplies a concrete counterexample.
- Local search can repair some starting points; this is different from beating
  the best existing start. LNS still beats full on selected median/high-skew cases.
- Refresh absolute-model validation under the workspace protocol, using separate
  calibration data if fitting is later requested. Do not add a new physical term
  merely to absorb the old output-page lifecycle residual.
- No full top-K recall/false-pruning certification is possible from these samples.
  Sampling-cost conclusions about old enumeration code were not changed/tested.

## Sampling, protocol and evidence

Four input frontiers under `tmp/moe_partial_order_vnd_20260904/`:
high_skew_hardware_frontier.json (76 original plans), median_template_lns_frontier.json
(145), high_skew_template_lns_frontier.json (106), uniformish_template_lns_frontier.json
(74). Freeze13/13/11/9 plans respectively,46 total. Preserve all unique original
anchors (4/4/3/2), one largest old predicted-gain candidate per anchor, three
additional historical measured-front plans, and two additional closest-boundary
old worse sentinels. Some original starts share anchors. Selection is retrospective
and deterministic, not a newly validated search selector or unbiased sample.
The previously unapproved452-plan pool experiment was not run.

`freeze_archive_workspace_replay.py` serializes exact original Plan V2 bridges,
records/hash provenance, old timings and source identities. Frozen-v8 point scores
are recomputed only for selected plans; all retained original parent-relative
predictions reproduce with zero drift. No formulas or calibration values changed.
`analyze_archive_workspace_replay.py` checks new sessions and reconstructs gains
against each retained historical parent, not just the sample's full anchor.

New runs: Arm-codex-internal, NUMA3 CPUs240–319/membind3, E256/H4096/F512,
2048tokens,TopK6,BF16,backend Ntile16, full stripes `(t,0,0,1,1)`, W13/W2
per-expert bytes8MiB/4MiB,early merge preserved. Fixed192MiB workspace initialized
and touched once; every selected plan x4 weight copies passes NaN-poisoned output
equality against allocation-mode reference before timing. Each session has5 warmups
and31 effective rounds,4-copy rotation,same copy per round,216MiB scrub,random
plan order,seeds20260907/08,phase trace OFF. All8 sessions complete and pass
identity/pairing/finite-timing checks. No new PMU/fault measurement.

The old archive runner used different synthetic input/weight seeds, copy selection
by round+position and no scrub in the inspected source. Therefore old→new changes
must be described as methodology-regime changes; the preceding output-pretouch
and workspace A/B reports isolate the output-allocation effect more directly.

Extension remains `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`;
calibration `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`.
Initial machine load0.04/0.46/2.90, no other benchmark found. Actual backing was
not audited; THP policy unchanged. Per-session workspace initialization is saved
separately from steady-state times. Existing records/calibration/winner files are
not overwritten.

Local ignored directory `tmp/workspace_archive_replay_20260907/` contains four
frozen samples, eight raw session JSONs and four summaries with all absolute
latencies, parent gains, P10/P90, model ranks and historical labels. Same remote
directory under `Arm-codex-internal:/home/zhangxu/codex/fused_cpp/` contains runner,
workspace helper, frozen samples and raw sessions. Other reused model evidence:
`tmp/workspace_order_replay_20260907/` and `tmp/workspace_transfer_replay_20260907/`.

Reproduce freezing, e.g. median:

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/freeze_archive_workspace_replay.py \
  --frontier tmp/moe_partial_order_vnd_20260904/median_template_lns_frontier.json \
  --sessions tmp/moe_partial_order_vnd_20260904/median_lns_session1.json tmp/moe_partial_order_vnd_20260904/median_lns_session2.json \
  --output <fresh-frontier.json>
```

Measure using the bounded runner `measure --frontier <sample> --route-file <source
route.file> --workspace-max-tokens 2048 --max-plans 13 --seed 20260907 --output
<fresh-session.json>` under the NUMA/OpenMP protocol above; repeat seed20260908.
Analyze with `PYTHONPATH=.:src .venv/bin/python
optimizations/fused_moe_sve/benchmarks/analyze_archive_workspace_replay.py
--frontier <sample> --sessions <session1> <session2> --output <fresh-summary>`.
Full raw comparisons are preserved; this report highlights conclusions, not a
replacement for the measured sample tables.

Validation:3 focused selector/gain tests pass, exact prediction-drift check=0 for
all4 samples, native output checks pass in all8 sessions. Ruff and static/diff
checks pass. No commit, production default, workspace growth or model fit.
