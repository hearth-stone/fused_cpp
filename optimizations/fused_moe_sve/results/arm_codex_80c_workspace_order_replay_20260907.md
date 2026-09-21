# Historical order frontiers on the fixed-workspace baseline

Geometry correction: references to Ntile8 below are a reporting error. Frozen
v8 and a direct packed-weight query against the same extension both report
backend_n_tile=16. Actual runtime packing was16 throughout; no timings changed.

Completed:8 unchanged frontiers,2 independent sessions each, no phase trace,
no search or calibration fit. Every plan/copy passes poisoned-workspace output
equality. Historical positive evidence changes substantially. Only the median
block-relocation near-elite retains both-session positive median and P10 among
the6 previously positive comparisons. Neither historical >2% winner retains its
original actionability gate. Do not overwrite historical winner artifacts or
claim the current anchor chain is still justified by those old gains.

## Historical gains revisited

Gain is `median_round(100*(anchor_ns/candidate_ns-1))`; positive means faster.
Each frontier keeps its original anchor. Old and new are separate epochs and
output-lifecycle regimes; use within-session paired gains, not cross-epoch raw
latency ratios as an A/B effect estimate.

| Historical positive | Old gain % S1/S2 | Workspace gain % S1/S2 | Workspace P10 % S1/S2 |
| --- | ---: | ---: | ---: |
| high-skew bounded rotation0 | 1.290/0.914 | -0.292/-0.155 | -3.003/-1.831 |
| high-skew bounded rotation1 | 3.887/2.243 | -1.734/-0.915 | -3.847/-2.521 |
| high-skew pressure bursty | 0.738/1.320 | -0.331/-0.739 | -1.993/-2.478 |
| high-skew pressure alternating | 1.037/1.960 | -0.584/-0.792 | -2.614/-3.015 |
| high-skew GEMM-density smooth | 2.355/3.464 | 1.218/2.151 | -1.340/-0.264 |
| median extended block p07 | 1.716/2.342 | 1.794/1.620 | 0.772/0.387 |

High-skew GEMM smoothing retains positive median direction but not positive P10;
do not describe that as no signal, nor as a stable benefit. Median block remains
a sub2% measured reference. Explicit interleave has no both-session actionable
candidate in either union. This does not reject every possible ordering or the
separate synthetic-M result, which was not replayed here.

## New positive signal and conflicting repeats

Median GEMM-density `smooth_global` passes in its3-plan frontier:
anchor30.564/30.573ms versus29.769/29.700ms; gains2.390/2.751%,
P10=0.823/1.146%. However, complete-bridge dedup shows this exact candidate and
anchor also occur in the5-plan pressure-balanced median frontier. There gains
are1.406/2.289%, P10=0.316/-0.305%, so those two sessions do not both pass.

All four median gain estimates are positive, but only2/4 individual sessions
meet both >2% and positive-P10 criteria. Treat this as a promising candidate
requiring focused confirmation, not as permission to cherry-pick the good
frontier and promote it. Keep other candidates and the original anchor unchanged.

## Remaining frozen-model error

There are39 candidate-anchor comparisons,38 unique complete-bridge pairs (the
median smooth duplicate above). A descriptive gain-error calculation averages
the session median gains per unique pair, including repeated-frontier evidence,
then averages `abs(frozen predicted gain - measured gain)` over38 pairs.
It changes from5.128 percentage points in the old regime to3.591 points under
workspace reuse. No model parameters changed. This is a selected-suite diagnostic,
not independent calibration validation or a global accuracy estimate.

Important workspace-regime counterexamples remain:

| Candidate | Frozen predicted gain | Measured gain S1/S2 |
| --- | ---: | ---: |
| high-skew union alternating p02 | +12.041% | -3.282/-2.430% |
| high-skew union alternating p04 | +11.260% | -3.432/-3.120% |
| high-skew GEMM-density smooth | -6.487% | +1.218/+2.151% |
| median GEMM-density smooth | +0.031% | +2.390/+2.751% in one frontier; see conflicting repeats |

Thus output first-touch explains part of the historical mismatch, not all of it.
Next model investigation should use these resident-workspace counterexamples and
compare predicted event/resource contributions, before adding physical terms.
Do not reuse old allocation-mode residual radii for automatic pruning/acceptance.
No new model formula, new probe, automatic anchor promotion or production change
was made in this replay.

## Complete workspace gains

All entries below give paired median gains % S1/S2. Exact absolute times,P10/P90,
win counts and validity evidence are in the saved per-frontier summaries.
Anchor latency medians are listed separately for every group.

| Group | Median anchor ms S1/S2 | High-skew anchor ms S1/S2 |
| --- | ---: | ---: |
| bounded order | 30.719/30.652 | 29.689/30.127 |
| pressure-balanced | 31.445/30.567 | 30.791/30.741 |
| GEMM-density | 30.564/30.573 | 30.567/30.482 |
| interleave/extended union | 30.625/30.572 | 29.965/30.079 |

| Group / candidate | Median gain S1/S2 | High-skew gain S1/S2 |
| --- | ---: | ---: |
| bounded independent_reverse_0 | +0.787/+0.420 | -1.691/-1.154 |
| bounded independent_reverse_1 | +0.500/+0.215 | -1.872/-2.643 |
| bounded cyclic_rotation_0 | -0.057/+0.115 | -0.292/-0.155 |
| bounded cyclic_rotation_1 | -1.751/-1.751 | -1.734/-0.915 |
| bounded joint_head_tail_0 | -0.556/-0.739 | -0.624/-0.203 |
| bounded joint_head_tail_1 | -0.433/-0.975 | -0.346/-0.592 |
| pressure smooth_global | +1.406/+2.289 | +1.682/+1.886 |
| pressure smooth_domain | +2.077/+0.397 | +1.069/+0.994 |
| pressure bursty_control | -5.065/-5.934 | -0.331/-0.739 |
| pressure alternating_routes | -3.755/-3.823 | -0.584/-0.792 |
| GEMM smooth_global | +2.390/+2.751 | +1.218/+2.151 |
| GEMM bursty_control | -5.586/-5.502 | -1.990/-1.780 |
| union p01 | +1.112/-1.770 | +0.446/-0.998 |
| union p02 | -3.748/-3.854 | -3.282/-2.430 |
| union p03 | -3.845/-3.919 | -3.852/-2.796 |
| union p04 | +0.022/-0.795 | -3.432/-3.120 |
| union p05 | -2.811/-2.784 | -1.207/+0.133 |
| union p06 | +1.050/-0.103 | +1.029/+1.132 |
| union p07 | +1.794/+1.620 | -0.279/-0.874 |
| union p08 | not present | -0.232/-0.202 |

The pXX mapping differs between the two unions; consult each frozen frontier's
family/strategy membership rather than treating pXX as the same operator across
traces. High-skew p06 has positive medians but P10=-1.336/-1.280%, not stable.

## Protocol and reproducibility

Eight original inputs, with local/remote SHA256 equality verified before running:

- `tmp/bounded_order_20260906/{median,high_skew}_frontier.json`:7 plans each.
- `tmp/pressure_balanced_orders_20260906/{median,high_skew}_frozen.json`:5 each.
- `tmp/gemm_density_20260907/{median,high_skew}_frozen.json`:3 each.
- `tmp/order_strategy_compare_20260907/{median,high_skew}_union.json`:8/9 plans.

Reuse unchanged bounded runner and FixedRouteWorkspace. `max_tokens=2048`,192MiB
FP32 route storage, initialized and touched once per process. Initialization
8.095–8.530ms is outside steady-state timing. Every plan/copy is checked against
an allocation-mode anchor reference after NaN poisoning; all checks pass. Timed
calls never poison or clear the workspace.16 processes, each5 warmups+31 effective
rounds,4-copy rotation,same copy within round,randomized plans,seeds20260907/08.
Actual largest frontier9 plans; explicit runner cap13 only enables existing unions.

Arm-codex-internal NUMA3 CPUs240–319/membind3,E256/H4096/F512,2048tokens,TopK6,
BF16,Ntile8,full stripes `(t,0,0,1,1)`,per-expert W13/W2 bytes8MiB/4MiB,
early merge unchanged. Median1x16T+8x8T; high-skew4x16T+16x1T.216MiB scrub before
each timed call. No phase trace, PMU or new fault measurement. Initial load
0.12/4.07/4.02, no competing benchmark found; session drift remains a caveat.
Same frozen extension `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`
and v8 calibration `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`.
No rebuild or production-source sync. Runner/helper hashes and initialization
time are in each session. Reuse prior focused tests for unchanged code; manifest
and report receive static/diff checks. All16 raw sessions pass pairing, identity,
copy/warmup, finite timing and correctness validation.

Remote code/raw sessions:
`Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/workspace_order_replay_20260907/`.
Local same-named directory contains all16 raw session JSONs and8 summaries.
Naming: `<family>_<trace>_session{1,2}.json`, `<family>_<trace>_summary.json`.
Family names are the four source directory names listed above. Original artifacts
remain unchanged. Union summaries retain strategy membership; freeze/search cost
inside them is historical metadata, not new generation work.

Reproduction uses the preceding workspace-transfer command, changing only the
frontier/output paths and adding `--max-plans 13`:

```bash
env FUSED_CPP_MOE_TRACE=0 OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:src \
  timeout 300 numactl --physcpubind=240-319 --membind=3 \
  .venv/bin/python tmp/workspace_order_replay_20260907/bench_bounded_order_extension.py measure \
  --frontier tmp/bounded_order_20260906/high_skew_frontier.json \
  --route-file bench_assets/moe_paper/dsv4_routes_pt_20260830/measured_request008_case009_zh2048-010.pt \
  --workspace-max-tokens 2048 --max-plans 13 --seed 20260907 \
  --output tmp/workspace_order_replay_20260907/bounded_order_20260906_high_skew_session1.json
```

Median route file is measured_request016_case017_zh2048-018.pt. Analyze the six
non-union frontiers with `analyze_bounded_order_extension.py`; analyze the two
unions with `compare_order_strategies.py analyze` (or the common analyzer's
`analyze(..., max_plans=13)`). All retain the original >2%/positive-P10 gate.
No commit made. Growth of workspace remains low priority and was not implemented.
