# Isolated versus joint real-expert stage validation

## Result

Completed two31-round trace sessions and one31-round no-trace control. All
seven full plans pass four-copy output equality; all310 measured isolated target
calls show every other expert starting strictly after target W2 completion
(minimum observed margin0.0006ms). No fit or current-model change.

Both error layers matter. M13/W13 and M17/W2 have prominent joint-increment
errors; M31/M65 W13 and M13/M35 W2 are dominated by isolated-base error; M35/W13
has both. Near-correct anchor M13 previously hid opposing base/increment errors.
Do not apply one universal small/medium-M scale or one global contention factor.

## Protocol declared before results

Class E Lab measurement, no fitting or production/calibration changes. Split
basic isolated stage error from joint/context increment error for real1T experts
56/M13,108/M17,29/M31,44/M35,227/M65 in the existing high-skew p11 plan.

Use the exact full-MoE hidden states, route slots, weights, stage sequence,
persistent workspace,80-worker pool and physical CPU316 for each isolated
expert. A new experimental Plan V2 dependency graph moves that expert first and
makes all remaining tasks depend on its completion. When removing it from its
original chain, successors inherit its former predecessors, preserving CPU
exclusion among remaining tasks. No expert computation is deleted or output
masked. All full outputs must remain bitwise equal to anchor for four copies.

Seven cases in one process: original anchor, original p11 and five target-first
isolations. Two independent trace sessions seeds591001/591002 and one trace-off
control seed591001; each5 warmups plus31 randomized, round/copy-matched samples.
Same cold216MiB scrub before every full call,4-copy rotation, fixed192MiB output
workspace,2048tokens,TopK6,E256,H4096,F512,BF16/SVE256,Ntile16,full stripes.
W13/W2 weight bytes8MiB/4MiB per expert. NUMA3 CPU240–319/membind3 on
Arm-codex-internal; helper/core identities match the prior case-trace experiment.
OMP_NUM_THREADS=1,OMP_DYNAMIC=FALSE,OMP_PROC_BIND=false; native CPU mapping remains
in the exact bridges. No kernel or extension build.

This isolates a cold-first expert WITHIN the full runtime, not a standalone
kernel with no worker pool. Joint-minus-isolated includes concurrency and prior
cache/execution history; this experiment does not identify those separately.
W2 follows the expert's real W13 as in joint execution. M65 in anchor is on
CPU308, so that one comparison is explicitly CPU-unmatched; p11 and all other
primary comparisons use the same CPU316.

Freeze current core-pressure model and its existing isolated phase estimates.
For each M,stage and context, with measured I,J and predictions Ihat,Jhat:

```text
base_error = Ihat - I
increment_error = (Jhat - Ihat) - (J - I)
total_error = Jhat - J = base_error + increment_error
```

Report signed components, not absolute shares that hide cancellation. Point
values use medians; a separate2000-resample IID paired-round bootstrap gives
95% intervals for the median measured increment. Two independent sessions check
repeatability; intervals do not establish absence of temporal dependence.
No parameters selected or fitted from these measurements.

## Implementation and validation

New Lab files `prepare_expert_layer_split.py` and `analyze_expert_layer_split.py`,
with `tests/test_moe_expert_layer_split.py`. No runtime source modification.
Three focused tests passed for dependency rewiring, root gating and signed
error decomposition. All seven generated bridges passed existing analytical
DAG/CPU-exclusion validation before remote execution. The existing remote runner
checks all four-copy full outputs BEFORE measurements.

Raw trace validation must additionally prove every other expert starts after
the target's W2 ends in each isolation call; metadata/trace hashes, task-worker
identities, complete rounds and frozen model identity are checked. Trace outer
Python elapsed is not stage timing and must not replace ordinary performance
labels. A same-protocol no-trace control is retained separately.

Artifacts local/remote: `tmp/expert_layer_split_20260909/`; exact generated
`frontier.json`, session1/session2 JSON and traces, control JSON, all stdout/stderr.
The standalone stage/error report is written locally after raw validation.

Runner reused remotely:
`tmp/workspace_phase_timeline_20260907/runner/bench_bounded_order_extension.py`.
Workspace profile:
`tmp/workspace_phase_timeline_20260907/workspace_numa3_80c.json`.
Source input frontier:
`tmp/workspace_archive_replay_20260907/vnd_high_skew.json`.

Local preparation and analysis commands:

```sh
.venv/bin/pytest -q tests/test_moe_expert_layer_split.py
.venv/bin/python optimizations/fused_moe_sve/benchmarks/prepare_expert_layer_split.py \
  --source tmp/workspace_archive_replay_20260907/vnd_high_skew.json \
  --output tmp/expert_layer_split_20260909/frontier.json
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_expert_layer_split.py \
  --directory tmp/expert_layer_split_20260909 \
  --output tmp/expert_layer_split_20260909/report.json
```

Use fresh paths on rerun (exclusive creation). Remote measurement uses the same
command as the case-trace experiment with this frontier, its preserved
`route_file`, max-plans7, workspace2048, seeds591001/591002; outputsession1/2 and
phase-tracesession1/2. Control omits `--phase-trace`. No remote source sync was
needed; only the generated experimental frontier was transferred.


## Independent base results

Values are isolated target stage medians, inms. Model is frozen and identical
across sessions. This is real cold-first expert execution, not synthetic inputs.

| M | W13 measured S1 / S2 | W13 model | W2 measured S1 / S2 | W2 model |
| --- | ---: | ---: | ---: | ---: |
| 13 | 1.539 / 1.564 | 1.339 | 0.778 / 0.793 | 0.670 |
| 17 | 1.801 / 1.807 | 1.702 | 0.892 / 0.888 | 0.851 |
| 31 | 3.189 / 3.192 | 3.025 | 1.554 / 1.553 | 1.513 |
| 35 | 3.628 / 3.632 | 3.403 | 1.842 / 1.851 | 1.702 |
| 65 | 6.628 / 6.625 | 6.239 | 3.195 / 3.192 | 3.120 |

All sampled isolated point estimates are low. W13 base errors are roughly5–14%
inS2, not proof of a universal coefficient for all M. Isolated round CV ranges
0.19–3.64% across measured stages/sessions. The observed basic errors therefore
cannot all be assigned to concurrent pressure.

## Joint error decomposition: p11, CPU-matched

Signed error=prediction minus measurement. Session2 values, ms; rounded columns
may differ in the last digit. Each exact row satisfies base+increment=total.

| M | W13 base error | W13 increment error | W13 total error | Interpretation |
| --- | ---: | ---: | ---: | --- |
| 13 | -0.224 | -0.974 | -1.199 | Joint/context response dominates |
| 17 | -0.105 | -0.111 | -0.216 | Both comparable |
| 31 | -0.167 | -0.039 | -0.206 | Mainly base |
| 35 | -0.228 | -0.328 | -0.557 | Both substantial |
| 65 | -0.385 | -0.091 | -0.476 | Mainly base |

Session1 base/increment/total W13 errors:
M13 -0.199/-0.960/-1.159ms;
M17 -0.099/-0.122/-0.221ms;
M31 -0.164/-0.053/-0.217ms;
M35 -0.225/-0.339/-0.564ms;
M65 -0.389/-0.069/-0.458ms. The attribution pattern repeats.

| M | W2 base error | W2 increment error | W2 total error | Interpretation |
| --- | ---: | ---: | ---: | --- |
| 13 | -0.124 | +0.006 | -0.117 | Mainly base; increment close |
| 17 | -0.037 | -0.182 | -0.219 | Mainly joint/context increment |
| 31 | -0.041 | -0.021 | -0.061 | Small errors, mainly base |
| 35 | -0.149 | +0.003 | -0.146 | Mainly base; increment close |
| 65 | -0.072 | -0.029 | -0.101 | Mainly base |

These five selected experts account for2.653ms of W13 pointwise underprediction
inS2:1.110ms base and1.543ms increment. The sum is a selected chain diagnostic,
not an independent full-forward error metric. W2 selected sum0.644ms splits
0.422ms base and0.222ms increment; other tasks, operator residual and gaps still
exist and can cancel these errors in the complete forward.

## M13: large joint miss and hidden cancellation in anchor

For W13 inS2:

```text
Measured isolated I = 1.564ms
Measured anchor J = 1.839ms
Measured p11 J = 2.897ms
Model isolated Ihat = 1.339ms
Model anchor Jhat = 1.895ms
Model p11 Jhat = 1.698ms
```

Anchor's base error-0.224ms is offset by increment error+0.281ms, leaving a
small+0.057ms joint total error. Thus prior good anchor total prediction did
not validate its isolated M13 base.

In p11, measured joint-minus-isolated increment is about1.333ms, model increment
only0.359ms. Its31 paired-round bootstrap median-increment95% interval is
[1.302,1.349]ms inS2, [1.285,1.348]ms inS1. Correcting the basic M13 time alone
would still leave roughly0.97ms of p11 W13 underprediction. The joint/context
problem remains independently visible after measuring the baseline.

This still does not distinguish concurrent service degradation from prior cache
history or tail-panel effects. The cold-first counterfactual removes both
concurrent expert work and the original execution prefix. Separate interventions
would be needed to identify those finer mechanisms.

M17/W2 similarly shows measured increment about0.290ms versus modeled0.108ms;
paired increment95% interval[0.266,0.301]ms inS2. Conversely M35/W2's measured
increment is small and interval[-0.015,0.034]ms includeszero; increasing its
contention penalty would target the wrong layer.

## Controls, integrity and limitations

Two trace sessions have full31-round pairing, exact worker/expert identities,
trace hashes and source/calibration/frontier identity checks. Current coefficient
file remains byte-identical. Three local focused tests pass; Ruff and diff checks
pass. No native code changed or rebuilt. Existing runner hash and profile match
[the per-case diagnosis](core_pressure_case_diagnosis_20260909.md).

No-trace full-forward anchor/p11 medians are29.803/35.857ms. Trace internal native
anchor is29.651/29.481ms and p1135.855/35.726ms. Across all seven cases and both
sessions, internal-native minus untraced Python elapsed ranges-0.470 to+0.082ms.
The outer trace-enabled elapsed includes logging (anchor~36.7–36.9ms,
p11~42.9–43.1ms), and is not used as a stage or calibration target. Separate
processes and different timer endpoints prevent claiming zero trace bias. The
large M13 incremental miss is much larger than observed endpoint differences;
small W2 discrepancies should not be overinterpreted as precise hardware causes.

The experimental iso plans execute the remaining full workload AFTER the target,
so their whole-forward times are not isolated-expert costs. Only target stage
envelopes are used. Output equality verifies full computation, while runtime
start/end checks establish absence of concurrent expert execution during the
isolated target. Idle worker-pool/runtime effects remain in this baseline.

Primary p11 comparisons are all CPU316-matched. Anchor M65 uses CPU308 and is
retained with cpu_matched=false; do not use it alone for causal attribution.
The other anchor comparisons retain CPU316. All decisions above use matched
p11 rows or explicitly matched M13 anchor rows.

## Exact remote invocation

From `/home/zhangxu/codex/fused_cpp`, with fresh output paths:

```sh
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=false MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/workspace_phase_timeline_20260907/runner/bench_bounded_order_extension.py measure \
  --frontier tmp/expert_layer_split_20260909/frontier.json \
  --route-file bench_assets/moe_paper/dsv4_routes_pt_20260830/measured_request008_case009_zh2048-010.pt \
  --experiment-baseline tmp/workspace_phase_timeline_20260907/workspace_numa3_80c.json \
  --workspace-max-tokens 2048 --max-plans 7 --seed 591001 \
  --phase-trace tmp/expert_layer_split_20260909/session1.trace \
  --output tmp/expert_layer_split_20260909/session1.json
```

Session2 uses591002. Control uses591001 and omits phase-trace. All three runs
complete with empty stderr; no active session remains. Local `report.json`
contains40 M/stage/context/session rows,10 isolation checks, source hashes and
paired intervals; `trace_sensitivity.json` preserves control comparisons.

The requested two-layer experiment is complete. No model correction, fitted
coefficient, new pruning rule or production adoption follows automatically.
