# Frozen-v8 selection pair: hardware validation

## Offline workflow entrypoint

The opt-in Lab flow is now implemented:
`score-best + fallback -> complete-bridge dedup -> at most two measured plans -> winner artifact`.
Production `IntervalPlanner.plan()` and quick/default dispatch remain unchanged.

On the target host, from repository root, one command can freeze missing plans,
run two independent sessions for a two-plan shortlist, and persist the result:

```bash
env OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:src \
  numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_selection_pair.py tune \
  --route-file bench_assets/moe_paper/dsv4_routes_pt_20260830/measured_request008_case009_zh2048-010.pt \
  --route-layer 38 \
  --calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --plans tmp/new_selection_run/plans.json --output tmp/new_selection_run/winner.json
```

An existing `--plans` is reused after route/layer and bridge validation; otherwise
it is frozen first. New freeze artifacts also expose the deduplicated `shortlist`
mapping explicitly. Session files are derived from the winner filename and must
not already exist. Old median/high-skew runner snapshots stay unchanged.

To reuse already completed hardware evidence without measuring again:

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/offline_selection_winner.py \
  --plans tmp/selection_other_traces_20260906/high_skew_plans.json \
  --sessions tmp/selection_other_traces_20260906/high_skew_session1.json \
             tmp/selection_other_traces_20260906/high_skew_session2.json \
  --output tmp/selection_winners_20260906/high_skew.json
```

Winner JSON uses experimental schema `moe_offline_selection_winner_v1` and saves
the full `plan_v2_bridge`, bridge identity, route/layer/calibration, measured
extension, source-session hashes, session summaries and `actionable`. Consumers
must check `decision`; session disagreement intentionally has no winner bridge.

- `hardware_consensus_winner`: both sessions' median-time winners agree. A small
  gain may still have `actionable=false`; do not turn empirical rank into a
  confidence claim.
- `deduplicated_single_plan`: no hardware comparison, `hardware_measured=false`.
- `inconclusive_session_disagreement`: no winner, retain the shortlist/evidence
  for a follow-up rather than silently choosing from the model.

The generated local artifacts are `tmp/selection_winners_20260906/median.json`,
`high_skew.json`, and `uniformish.json`. The first two reuse the measured sessions
below and choose score-best with `actionable=true`; uniformish was exercised
through the actual `tune` CLI and saves the single deduplicated plan. Two-session
orchestration was unit-tested at the subprocess boundary; no new hardware rerun
was needed because the numerical/timed loop is unchanged. Final focused tests:
19 passed (`test_moe_selection_pair.py`, `test_moe_offline_selection_winner.py`).
Ruff, YAML parsing and diff checks pass. No commit or production adoption.

Status: median and high-skew both pass the two-session selection-counterexample
gate; uniformish candidates are identical after bridge deduplication.
Independent Lab diagnostic; no planner,
calibration, production dispatch or packed-format changes.

## Outcome

The selector excludes a genuinely faster candidate on this frozen-v8 median
trace. This is measured evidence, not the earlier 13.80% model-internal gap.

| Session | Selected median | Score-best median | Paired speedup median / P10 / P90 | Wins | Median-time reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| S1 | 37.89820 ms | 31.85368 ms | 19.129% / 18.108% / 19.709% | 31/31 | 15.949% |
| S2 | 37.72437 ms | 31.92408 ms | 18.268% / 17.400% / 19.016% | 31/31 | 15.375% |

P10/P90 use sorted paired samples at indices 3/27 for 31 rounds. Speedup and
time reduction have different denominators, as declared below. Both sessions
have matching frozen-plan, runner and extension identities. Each passed exact
BF16 output equality for both plans on all four packed copies before timing.

The model correctly orders these two plans, though it underestimates their
measured separation. Additional physical-model accuracy is not a prerequisite
to retaining this particular useful candidate. This does not establish that
the model generally ranks neighbors or other widths correctly.

Recommendation: next evaluate a bounded selection alternative that retains the
score winner alongside the historical fallback, on the other two trace classes
before changing defaults. Do not universally remove fallback based on one trace.
The two plans differ in width, assignment and order: this result identifies a
selection counterexample, not which of those mechanisms supplies the gain.

Limitations: this compares two plans, not the global optimum. Tensor values are
synthetic while routing is the captured trace. Cold weights use scrub plus copy
rotation; actual all-DRAM misses and per-allocation page backing were not measured
in this run. No HugeTLB pool or page policy was changed. Preflight showed no
active compute load; continuous background/PMU monitoring was not collected.
The large repeated separation supports the scoped comparison, not an attribution
to a specific DDR or LLC mechanism.

## Scope and historical rationale

### Extension to the other two trace classes (predeclared)

Completed high-skew result, using the unchanged protocol below:

| Session | Selected median | Score-best median | Paired speedup median / P10 / P90 | Wins | Median-time reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| S1 | 42.11705 ms | 32.18265 ms | 30.841% / 28.391% / 31.468% | 31/31 | 23.588% |
| S2 | 41.63758 ms | 32.61115 ms | 27.714% / 26.675% / 28.749% | 31/31 | 21.679% |

Both sessions passed exact output equality on both plans and all four weight
copies. Frozen-plan, runner and extension identities match across sessions;
the extension also matches the median experiment. The source hashes used to
freeze all three traces match. No model fitting or selection changes occurred.
Some absolute session drift remains (and the gain differs by about 3.1 percentage
points); every paired round nevertheless favors score-best by a large margin.

Across the known three-trace corpus, retaining the score winner alongside the
fallback offers a useful alternate on median and high-skew, while uniformish
deduplicates to the existing executable plan. This supports a bounded offline
two-plan shortlist followed by hardware selection. It does not establish a
universal measurement-free replacement selector, global optimality, or behavior
on other shapes/traces. The uniformish result is exact plan equivalence, **not**
a newly measured zero-regression result. The later opt-in Lab workflow is
documented above; no production shortlist integration has been implemented.

High-skew raw artifact SHA256 values:

- S1: `46b9f1ad57103a98167950ac283e72870ee0fbe0a8d86fdf2c6e346210cc90dc`
- S2: `ac0c9ee1bfe1b83339b317686e3e12bb14b16c846efe141bf817f6d2c1f342a7`

The user requested the same two-candidate retention check on high-skew
request008/layer38 and uniformish request022/layer20. Freeze both candidates
before measurement with the unchanged v8 and model/planner sources. The runner
now accepts an explicit route layer and verifies it against the frozen artifact;
the original median's numerical restoration check remains active for that case.
It also rejects extension identity changes relative to the median experiment.
If two bridges are identical, report a single unique candidate rather than
measuring a fictitious two-plan comparison.

The protocol and gates remain unchanged: two independent processes per trace,
five warmups, 31 paired rounds, fixed tensors, random order, same-copy pairing,
four-copy rotation, per-cell 216 MiB scrub. These are evaluations only, not fit
data. No production selector or calibration is modified. An oracle choosing the
faster of two measured plans is a hardware-assisted shortlist result, not proof
that a measurement-free selector can choose correctly. The three named traces
are a known evaluation corpus, not a newly unseen generalization set.

New artifacts are kept separately under `tmp/selection_other_traces_20260906/`
locally and in the same relative directory on Arm-codex-internal. The old median
runner snapshot and artifacts remain unchanged in their original directory.
Focused route/layer validation: 5 tests passed; Ruff passed before freezing.
The two existing analytic-full selector regression tests also passed (104
deselected). No uniformish hardware run was necessary because both bridges
are identical. No commit was created.

Restored candidates (both model/planner source hashes match the median freeze):

| Trace | Fallback selected | Score-best | Predicted selected / best |
| --- | --- | --- | ---: |
| high-skew, request008/layer38 | 6x8T + 16x2T, lpt | 4x16T + 16x1T, reverse_odd | 43.645737 / 35.293690 ms |
| uniformish, request022/layer20 | 10x8T, reverse_even | identical | 31.125114 / 31.125114 ms |

All are full-owner-stripe (both stage window tiles zero). For the new 1T tasks,
per-thread W13/W2 owner footprints are 8/4 MiB, R13=R2=1; other widths follow
the geometry table below. Uniformish selected and score-best bridge hashes both
equal `86b4b7a44eb4f415eecd057d5e6a82ce3487e17511692fb769972fa26d218f8a`.
It has one unique executable plan: adding score-best does not change execution
or require a second hardware slot. No new uniformish performance claim or
hardware rerun is made; equivalence here is established by the identical bridge.

High-skew bridge hashes:

- selected: `9d6d33d2779c2e9bff77e46393e8c6c1664344be60a7d5fbc026354810fdda99`
- score-best: `e0783526487e1192873c78a31e050b548bf5aa87d5af6a06f21a3ecfc1540bf1`

Freeze commands use the same runner/calibration as below, adding
`--route-layer 38` with `measured_request008_case009_zh2048-010.pt` and
`--route-layer 20` with `measured_request022_case023_zh2048-024.pt`, respectively.
Outputs are `high_skew_plans.json` and `uniformish_plans.json` in the new artifact
directory. Hardware commands use the copied runner from that new directory,
request008/layer38, `high_skew_plans.json`, seeds 20260906/20260907, and outputs
`high_skew_session1.json`/`high_skew_session2.json`. All other command arguments,
environment and NUMA placement are unchanged from the median commands below.

The specific example in `arm_codex_80c_order_width_selection_and_ddr_interleave_20260906.md`
is request016/layer4, conventionally the **median trace**, not high-skew. Preserve
this exact trace: E256, H4096/F512, 2048 tokens, TopK6, BF16, NUMA3 CPUs240–319.

Width fallback originated in `b2196270211b4049c2872350fce1b9188aa77de5`.
`arm_codex_80c_high_skew_planner_gate_20260901.md` records real gains of
5.69–17.95% from the narrower fallback under an older calibration, with prediction
gaps only 0.6–2.0%. It is a historical reliability guard, not an arbitrary cut.
That evidence does not establish its value under later frozen v8.

Freeze the current strict candidate set plus the analytic baseline, then save
complete PlanV2 bridges for the selector winner and minimum-score candidate.
Require the documented 28-lane/38.826 ms and 9-lane/34.116 ms predictions within
0.01 ms before hardware work. Record current dirty model/planner source hashes;
do not sync their changes to the remote production tree.

## Predeclared protocol

- Same process, input tensors, four packed-weight allocations for both plans;
  fixed tensor seed across sessions, independent execution-order seeds.
- Every warmup and measured cell uses disjoint 216 MiB scrub outside timing;
  same copy in each pair, four-copy rotation across rounds.
- Five warmup and 31 measured paired rounds per independent process session.
- Bitwise BF16 equality on both plans and every copy before timing.
- SVE BF16, direct W2 route output, FP32 route accumulation, ready-token merge
  enabled equally. No PMU overhead in this end-to-end comparison.
- Positive speedup is `100 * (selected_time / score_best_time - 1)`.
  Report time reduction separately; do not mix denominators.
- A counterexample to the fallback requires median speedup >2% and P10 >0
  in both sessions. No default behavior change is authorized by this result.

Next decision: if the score winner wins, investigate selection policy retention;
if it loses, retain a cross-width ordering counterexample. If unstable, do not
infer either superiority or a safe default change. Width and order are changed
together here; separating them requires a subsequent experiment.

## Frozen plans and identity

The 141 strict shapes plus analytic baseline produced 142 candidates. Both
documented plans reproduced without changing the selector or model:

| Plan | Widths | Order | Prediction |
| --- | --- | --- | ---: |
| selected | 4x8T + 24x2T | lpt | 38.8258167605 ms |
| score_best | 1x16T + 8x8T | reverse_odd | 34.1163570128 ms |

Both have W13/W2 window tiles 0 (full-owner-stripe). Full stage weights per
expert are W13 8 MiB and W2 4 MiB. Per-thread owner footprints are respectively
4/2 MiB at 2T, 1/0.5 MiB at 8T and 0.5/0.25 MiB at 16T; R13=R2=1.
The unchanged Arm SVE BF16 backend uses N tile 8.

Bridge hashes:

- selected: `c8f238324cde2b47d33f6cf665e669728a90299f808c17624ff3df147252a993`
- score_best: `af7741be6a7847e77d9e457969e38ec6fb0f294a9eba664b5d07ad4652cb7606`

Frozen v8 calibration SHA256:
`7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`.
Local planner/model sources include pre-existing user edits; their SHA256 values
are saved in `plans.json`. The relevant quick-width edit affects the additional
baseline candidate but does not prevent reproducing either reported plan.
The source tree was not reverted or synced remotely.

Measured extension SHA256 (same identity as the historical gate):
`dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`.
No native rebuild was performed. Local HEAD at experiment setup:
`c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5`.

## Reproduction

Local ignored artifact directory: `tmp/selection_pair_20260906/`.
Remote: `Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/selection_pair_20260906/`.
Both retain complete `plans.json` and raw session JSON, not just plan summaries.
Each session records runner/extension/plan identity and every warmup/timed round.
Raw SHA256 values:

- S1: `6b1d044d39eb96690d0ade6c3c432aa23d2224b324305038959f69c95f6bedd8`
- S2: `12ae65e72059f226d72fc846f2108e658d84cadcf20f05c9f6a61871f0eabf73`

From the local repository root:

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/bench_selection_pair.py freeze \
  --route-file bench_assets/moe_paper/dsv4_routes_pt_20260830/measured_request016_case017_zh2048-018.pt \
  --calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --plans tmp/selection_pair_20260906/plans.json
```

From the remote repository root, after copying only the new runner and plans:

```bash
env OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:src \
  timeout 300 numactl --physcpubind=240-319 --membind=3 \
  .venv/bin/python tmp/selection_pair_20260906/bench_selection_pair.py measure \
  --route-file bench_assets/moe_paper/dsv4_routes_pt_20260830/measured_request016_case017_zh2048-018.pt \
  --plans tmp/selection_pair_20260906/plans.json \
  --seed 20260906 --output tmp/selection_pair_20260906/session1.json
```

Repeat in a new process with `--seed 20260907` and `session2.json`. Existing
artifact paths are rejected; use new names for reruns. Only execution order
changes with session seed; weights and hidden/router tensors remain fixed.

Local selector regression command:

```bash
PYTHONPATH=.:src .venv/bin/python -m pytest -q tests/test_moe_analytic_model.py \
  -k 'analytic_full_optimizes_expected or analytic_full_steps_down'
```

Result: 2 passed, 104 deselected. Ruff and diff checks passed. This preserves
the current fallback semantics; it does not test an unimplemented replacement.
