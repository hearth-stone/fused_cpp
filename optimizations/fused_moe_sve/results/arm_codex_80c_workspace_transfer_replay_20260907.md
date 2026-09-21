# Frozen transfer cases replayed with fixed route workspace

Geometry correction: Ntile8 below should read Ntile16. Frozen v8 and the actual
packed-weight object agree on16; this corrects prose, not executed geometry.

Completed: median and high-skew, each seven unchanged plans and two independent
sessions, all using fixed2048-token route storage. Phase tracing is OFF. All
plan/copy output comparisons pass after NaN poisoning. Neither trace has a
candidate passing >2% paired median gain and positive P10 in both sessions.
Retain both anchors; no calibration, candidate or production-default change.

Scope is the most recent same-width transfer suite, not every historical order,
LNS or synthetic-M experiment. Older ordering anchors themselves have not been
reselected under workspace reuse.

## What changed

Every gain below is relative to the corresponding in-session anchor. Old values
come from the original allocation-per-call untraced transfer sessions; new values
come from this untraced workspace replay. They are separate epochs, not a new
within-session allocation-versus-reuse A/B. The preceding fixed-workspace report
contains that direct A/B for three high-skew plans.

| Trace/candidate | Original gains % S1/S2 | Workspace gains % S1/S2 | Workspace latency ms S1/S2 |
| --- | ---: | ---: | ---: |
| median anchor | 0/0 | 0/0 | 31.326/30.317 |
| same_llc_swap_0 | +0.750/+0.576 | +0.010/-0.185 | 31.303/30.418 |
| same_llc_relocation_0 | -0.416/-0.500 | +0.014/+0.236 | 31.283/30.301 |
| cross_llc_swap_0 | +0.146/-0.221 | +0.721/-1.227 | 31.125/30.695 |
| cross_llc_relocation_0 | -0.245/-0.126 | +0.111/-0.008 | 31.270/30.325 |
| same_llc_swap_1 | -0.104/+0.053 | -0.023/-0.070 | 31.292/30.347 |
| same_llc_relocation_1 | -4.894/-4.471 | +1.109/+1.108 | 30.950/30.027 |
| high-skew anchor | 0/0 | 0/0 | 30.182/30.041 |
| same_llc_swap_0 | -7.121/-6.917 | -6.563/-6.642 | 32.218/32.083 |
| same_llc_relocation_0 | -13.123/-13.180 | -12.983/-13.668 | 34.466/34.387 |
| cross_llc_swap_0 | -6.238/-6.746 | -1.655/-1.861 | 30.644/30.418 |
| cross_llc_relocation_0 | -5.802/-6.105 | +0.254/-0.366 | 29.895/30.188 |
| same_llc_swap_1 | -0.585/-0.773 | -0.811/-1.507 | 30.032/30.193 |
| same_llc_relocation_1 | -0.015/+0.172 | +0.436/-0.309 | 29.842/29.789 |

Median's formerly positive swap is now approximately neutral. The formerly
bad relocation diversity representative now has a positive1.11% median in both
sessions, but P10 is+0.706/-0.748%; it is not a stable actionable winner.
The1.009ms median-anchor difference between sessions underscores why cross-epoch
absolute speedups and sub-percent sign changes need caution. No noise-driven
post-hoc margin adjustment or extra sampling was performed.

High-skew's cross-domain relocation penalty essentially disappears, consistent
with its output first-touch diagnosis. Cross-domain swap remains around1.7–1.9%
slower, substantially less severe than before. Meanwhile same-domain model-best
swap and relocation remain strongly negative. Their P10 gains are-8.728/-9.495%
and-15.266/-15.403%, respectively. Workspace does not explain away these cases.

Using the diagnostic definition of clearly bad (both medians below-2% and both
P90 gains negative), this sampled suite changes from5 clearly bad cases to2,
both high-skew same-domain moves. This is not a pruning validation or proof that
all remaining error is bandwidth-related. The frozen model still predicts small
positive gains for the two persistent regressions, so ranking is not solved.

## Protocol and validation

Reuse exact `tmp/same_width_transfer_20260907/{median,high_skew}_frozen.json`
bridges. No scoring/generation/refitting. Add explicit
`--workspace-max-tokens 2048` to the existing bounded runner; legacy default
remains allocation-per-call. Each workspace is initialized once, capacity192MiB,
then leased sequentially by every plan. Initialization costs: median8.072/8.216ms,
high-skew8.049/8.128ms, outside steady-state timing. Correctness uses a normal
allocation anchor reference, then NaN-poisons storage before every plan/copy
comparison. All7x4 checks pass per session. No poisoning/clearing during timing.

NUMA3 CPUs240–319/membind3, E256/H4096/F512,2048tokens,TopK6,BF16,Ntile8,
full stripes `(t,0,0,1,1)`, per-expert W13/W2 bytes8MiB/4MiB,early merge unchanged.
Median shape1x16T+8x8T; high-skew4x16T+16x1T. Same4-copy weights,216MiB scrub
before every call,5 warmups+31 effective rounds,randomized order,seeds20260907/08.
Initial load0.04/1.51/1.40, no other benchmark found. No new PMU/fault/phase
collection in this replay. Prior fault measurements must not be relabeled as new.

Frozen extension SHA256
`dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`;
v8 calibration `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`.
Workspace helper hash
`7e0124673b4742a4747f6b4b733cb364def2f455cc5919ef13379eca4cb098db`.
Analyzer validates matching workspace/runner/extension/frontier identities,
independent seeds, all252 unique cells/session and exact pairing/copy/warmup rules.
It rejects comparisons mixing workspace and allocation-mode sessions.

Local validation:27 focused workspace/generator/anchor tests passed; the updated
workspace-identity rejection regression then passed in the16-test bounded suite.
Ruff, manifest parsing and diff checks pass. No commit made.

## Reproduction and evidence

Local ignored directory `tmp/workspace_transfer_replay_20260907/`: four session
JSONs plus median_summary.json/high_skew_summary.json. Same remote directory under
`Arm-codex-internal:/home/zhangxu/codex/fused_cpp/` retains runner/helper and raw
session JSONs. Original allocation and traced workspace artifacts are unchanged.

From remote project root, median session1 (repeat with seed20260908/session2):

```bash
env FUSED_CPP_MOE_TRACE=0 OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:src \
  timeout 300 numactl --physcpubind=240-319 --membind=3 \
  .venv/bin/python tmp/workspace_transfer_replay_20260907/bench_bounded_order_extension.py measure \
  --frontier tmp/same_width_transfer_20260907/median_frozen.json \
  --route-file bench_assets/moe_paper/dsv4_routes_pt_20260830/measured_request016_case017_zh2048-018.pt \
  --workspace-max-tokens 2048 --seed 20260907 \
  --output tmp/workspace_transfer_replay_20260907/median_session1.json
```

High-skew uses high_skew_frozen.json and measured_request008_case009_zh2048-010.pt.
Analyze with existing `analyze_bounded_order_extension.py --frontier <frozen>
--sessions <session1> <session2> --output <fresh-summary>`.

## Decision

Retain anchors for this budget; preserve changed rankings and all regressions.
Do not train/prune against old allocation-lifecycle residuals as though they
represented the workspace regime. If continuing, first replay historical order
frontiers under the same workspace baseline, then revisit remaining high-skew
same-domain regressions. Neither step was run or implicitly authorized here.
