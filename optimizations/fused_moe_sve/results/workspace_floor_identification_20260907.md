# Small-M phase identification and missing2T/4T coverage

## Decision

The additional isolated data are complete. The old hard-floor form remains
unsafe for M1/1T; an affine GEMM-stage candidate passes all measured M1 GEMM
guards and all M7 GEMM checks. However, the shared gather model fails the
predeclared M7 gate, and some affine intercepts are not repeat-stable.
**Neither complete calibration is eligible; wide/narrow replacement was not
performed.** The gate was not relaxed after seeing the data.

This resolves the immediate modeling direction, not physical parameter
identifiability: do not interpret the minimum measured small-M stage time as a
universal floor. Retain affine GEMM as a Lab candidate, not a production model.

## Experiment and evidence

Three predeclared groups, each12 isolated cells plus the same full-plan anchor,
two independent sessions each:

- M3/4/8/16/24 at1/2/4/8/16T;
- M62/1341 at2/4T;
- M7 at all five widths, withheld from fitting;
- M1 at2/4T, reserved guards; earlier M1/1T and16T remain guards.

There are36 new isolated cells,6 complete sessions and3126 validated calls
including warmups/correctness. Each isolated cell has31 timed observations per
session. Every target finishes before any background task starts. All plans and
four weight copies pass bit-exact output checks with poisoned workspace.

Arm-codex-internal, `/home/zhangxu/codex/fused_cpp`, NUMA3 CPU240–319,
E256/H4096/F512,2048tokens,TopK6,BF16,actual Ntile16,merge on. Full owner
stripes `(t,0,0,1,1)`, W13/W2 full weight bytes8MiB/4MiB per expert.
Same parent workspace profile, fixed2048-token pre-touched output,4-copy
rotation,216MiB scrub,5 warmups,31 paired rounds,trace on; seeds20260915/16.
Real high-skew route layer38, synthetic weights/hidden inputs. Each group is
a separate process; allocations are shared within a session, not across groups.
New1T probes use logical begin68;2/4/8/16T use begin64.

Earlier non-M1 isolated points (M10/62/714/1341 at their measured widths) remain
scale anchors with their original provenance/placements, not relabelled new
measurements. Combined dataset:47 shape/width cells,141 phase rows;38 fit cells,
5 M7 validation cells,4 M1 guard cells. M1/8T is not measured. Cross-group and
earlier-point placement differences remain a limitation of width-only fitting.

Common-anchor native medians, s1/s2, are29.778/29.733ms (`small_1_16`),
29.192/29.485ms (`width_2_4`),29.270/30.205ms (`bridge_8_guard`). They are
diagnostic controls, not paired cross-group performance gains.

## Correctness issue found and fixed before accepting data

The first attempt stopped during correctness, before timed rounds, with NaNs.
A correctness-only diagnostic also failed. The failure was in the Lab isolation
generator's dependency rewrite, not a native change:

Original lane chain: expert193 →expert16(M3) →expert64. Moving expert16 to the
head without reconnecting expert64 to expert193 made the old successor ready
before the original prefix completed. Tasks could then misuse the same core
group. The corrected rule replaces dependencies on the removed target with its
old predecessors, while still requiring every background task to wait for the
new target head. A middle-of-chain regression test now covers this case.

Failed plans/logs remain in `tmp/workspace_floor_grid_20260907/`; none enter
analysis. Corrected plans and accepted data are in its `corrected/` subdirectory.
No tolerance was loosened, no failing run was retried until it happened to pass,
and no native kernel was modified. The rerun used a structural dependency fix
with the same grid,roles,seeds and gates. The diagnostic snapshot records
256 NaNs in output and route workspace on the reproduced failure.

Earlier isolated target phase data remain limited to their validated isolated
intervals: output correctness and target-before-background checks had passed.
Do not use their delayed-background order as evidence of an unchanged full DAG.
Replay historical evidence from its saved full bridges rather than regenerating
them with the corrected helper.

## Model comparison, fixed before measurement

Gather uses nonnegative affine row-count regression. GEMMs compare:

- floor: `max(floor, scale * physical_stage)`;
- affine: `fixed + scale * physical_stage`, both coefficients nonnegative.

Session1 alone fits parameters; session2 measures repeat error and separately
refits parameters for stability inspection. Neither M7 nor M1 enters either fit.
M7 is validation for these predeclared forms, not an untouched final test if it
is used for selection. Guards are retrospective known counterexamples.

Gate: every M7 stage must be within20% in both sessions, and every measured M1
GEMM stage must have no larger absolute percentage error than v8. Both forms
fail the full gate. Affine has no M1 GEMM guard regressions; floor fails the four
M1/1T stage/session checks. Both share the six gather validation failures below.

### GEMM prediction: affine resolves the severe M1 extrapolation failure

| M1 width/stage | Hardware s1/s2 ms | Floor prediction ms | Affine prediction ms |
| --- | --- | ---: | ---: |
| 1T W13 | 0.31250/0.32024 | 0.44420 | **0.29474** |
| 1T W2 | 0.15102/0.15313 | 0.21826 | **0.15015** |
| 2T W13 | 0.25050/0.25381 | 0.26737 | 0.25346 |
| 2T W2 | 0.12216/0.12207 | 0.13166 | 0.12702 |
| 4T W13 | 0.20534/0.20652 | 0.20654 | 0.21076 |
| 4T W2 | 0.10205/0.10239 | 0.10335 | 0.10823 |
| 16T W13 | 0.07288/0.07350 | 0.07416 | 0.07614 |
| 16T W2 | 0.03296/0.03345 | 0.03338 | 0.03503 |

Floor M1/1T errors remain+38.7–44.5%. Affine's maximum measured M1 GEMM
absolute error is7.96%. Its M7 W13 MAPE is6.89%/7.28%, maximum16.74%/16.52%;
M7 W2 MAPE4.92%/5.19%, maximum13.31%/13.79%. All these GEMM checks pass20%.
Floor actually has lower average M7 GEMM error, but fails the independent M1
guard; selection by average error alone would hide the problem.

### Parameters are not yet physical constants

For1T W13, the fitted floor is0.44420ms and leave-one-M-out fits keep it within
0.44420–0.44979ms. Despite that apparent stability, actual M1 is0.31250–0.32024ms.
Stable fitting at observed M does not establish a valid lower-M floor.

Affine1T W13 fits `fixed=37.98us,scale=1.0546` in session1, but
`fixed=1.70us,scale=1.0659` in session2. Session1 leave-one-M-out intercepts span
27.38–44.46us. The intercept/scale tradeoff is not sufficiently repeat-stable
to label the intercept a measured hardware startup cost. Affine improves
prediction without proving that physical interpretation.

The optional Lab adapter represents affine fixed time as a zero-resource
stage-setup phase and scales only the GEMM body. It does not feed fixed startup
back into bandwidth or wide-team dilation. Production schema and default
model are untouched. Because the complete gate failed, its full-plan
wide/narrow replay branch was not executed (`replay=[]`).

### Gather is the remaining gate failure

M7 gather:

| Width | Prediction ms | Hardware s1/s2 ms | Error s1/s2 |
| ---: | ---: | --- | --- |
| 1 | 0.01970 | 0.01227/0.01071 | +60.6%/+84.0% |
| 2 | 0.03138 | 0.01932/0.02231 | +62.4%/+40.7% |
| 4 | 0.04266 | 0.03851/0.04627 | +10.8%/-7.8% |
| 8 | 0.04748 | 0.05007/0.07310 | -5.2%/-35.0% |
| 16 | 0.06307 | 0.06244/0.05229 | +1.0%/+20.6% |

Both forms use the same gather model. Its M7 MAPE is27.99%/37.62%.
Failures are1T/2T both sessions and8T/16T session2. Absolute errors are roughly
7–26us at the failing points; these are not whole-plan percentage errors.
The metric is a multi-worker stage envelope, not pure copy service time.
Worker-arrival skew,synchronization and input locality are possible contributors,
not established causes. Do not silently drop gather from the predeclared gate.

## Added2T/4T evidence

Representative measured stage medians in ms, s1/s2:

| M | Width | Gather | W13 | W2 |
| ---: | ---: | --- | --- | --- |
| 3 | 2 | 0.02256/0.02129 | 0.26777/0.26870 | 0.13166/0.13119 |
| 24 | 2 | 0.05585/0.06700 | 1.21181/1.21229 | 0.62759/0.62177 |
| 62 | 2 | 0.11403/0.13203 | 3.25670/3.25898 | 1.54487/1.55233 |
| 1341 | 2 | 2.14542/2.14118 | 67.39095/67.45693 | 32.34608/32.35711 |
| 3 | 4 | 0.03569/0.05272 | 0.20505/0.20672 | 0.10335/0.10301 |
| 24 | 4 | 0.05227/0.06201 | 0.61406/0.61349 | 0.31641/0.31576 |
| 62 | 4 | 0.11028/0.10115 | 1.70064/1.70766 | 0.81902/0.81824 |
| 1341 | 4 | 1.25405/1.26422 | 33.67356/33.68442 | 16.20107/16.21777 |

These close the missing-width data gap, not an all-M accuracy guarantee.
No new smoothing generator,search change,calibration deployment or commit.

## Reproduction and retained artifacts

Local/remote relative root: `tmp/workspace_floor_grid_20260907/corrected/`.
All six session JSONs and compact phase samples are local. Full raw traces are
retained remotely as lossless `.log.gz`; session hashes refer to uncompressed
traces checked before compression. `analysis.json` contains both forms,all141
phase rows,coefficients,repeat fits,leave-one-M-out fits and gates.

Freeze:

```bash
PYTHONPATH=.:src .venv/bin/python optimizations/fused_moe_sve/benchmarks/prepare_workspace_floor_grid.py \
  --source tmp/workspace_phase_timeline_20260907/high_skew.json \
  --route-frontier tmp/moe_partial_order_vnd_20260904/high_skew_template_lns_frontier.json \
  --output-dir tmp/workspace_floor_grid_20260907/corrected
```

Each `group=small_1_16,width_2_4,bridge_8_guard`, `session=1,2` runs the saved
`corrected/runner/bench_bounded_order_extension.py measure` with `--max-plans13`,
the group frontier,NUMA3 workspace profile,explicit phase-trace and fresh output
paths,seed `20260914+session`. Full command shape and environment match
`workspace_isolated_width_20260907.md`; the route is
`bench_assets/moe_paper/dsv4_routes_pt_20260830/measured_request008_case009_zh2048-010.pt`.
The streaming analyzer validates each complete trace before `gzip -1` retention.

```bash
PYTHONPATH=.:src .venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_workspace_floor_grid.py \
  --input-dir tmp/workspace_floor_grid_20260907/corrected \
  --old-summary tmp/workspace_isolated_width_20260907/summary.json \
  --calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --frontier-dir tmp/workspace_phase_timeline_20260907 \
  --output tmp/workspace_floor_grid_20260907/corrected/reproduction.json
```

Focused validation:34 tests passed for isolation/bounded runner/workspace;
7 phase-reaccount tests passed,including affine phase adapter and floor-vs-affine
semantics. Ruff/diff checks pass. Earlier three production runtime tests with
missing `AnalyticPolicy.wide_team_pressure` remain an unrelated known issue;
production code was not changed or claimed validated in this task.

Next bounded analysis: use existing worker-level gather traces to distinguish
copy duration from arrival/synchronization envelope, then design an appropriate
independent gather validation. Do not change this experiment's gate retrospectively.
Only after an eligible phase baseline exists should corrected wide/narrow
replacement be assessed on both large16T and small1T counterexamples.
