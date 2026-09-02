# CPU MoE paper experiment runner

This directory provides a declarative SSH runner for the fused-expert paper
matrix. It orchestrates existing correctness tests and benchmarks; it does not
add a kernel variant or change production dispatch.

Each run is bound to one committed source revision. With `--sync` (the default),
the runner creates a clean `git archive`, verifies and includes the pinned
`xbyak_aarch64` submodule, and synchronizes that snapshot without copying local
uncommitted work. The repository's ignored `refs/i8gemm/lib` dependency is
copied separately and bound to a deterministic content hash in `summary.json`;
it must become a committed dependency before the final artifact freeze. Remote
virtual environments and build products are preserved.
The runner does not use `rsync --delete`, so a source deletion still requires the
explicit remote cleanup required by `docs/agent_remote_execution.md`.

Results are written under the ignored local directory
`tmp/moe_paper_runs/<run-id>/`. Every run contains machine and extension hashes,
commands, stdout/stderr, structured benchmark results, the source revision, and
a summary. A unique mirror remains under the machine's configured
`remote_results_root`; the runner does not remove it automatically.
Suite builds set `FUSED_CPP_BUILD_MOE_ONLY=1`, so a paper run rebuilds the
dedicated MoE extension without charging unrelated operators to setup time.

## Dry run

```bash
.venv/bin/python \
  optimizations/fused_moe_sve/paper_experiments/run_matrix.py \
  --machine optimizations/fused_moe_sve/paper_experiments/machines/arm_codex_internal.json \
  --suite optimizations/fused_moe_sve/paper_experiments/suites/fused_expert_smoke.json \
  --dry-run
```

## Execute

```bash
.venv/bin/python \
  optimizations/fused_moe_sve/paper_experiments/run_matrix.py \
  --machine optimizations/fused_moe_sve/paper_experiments/machines/amazon_ecs_8cores.json \
  --suite optimizations/fused_moe_sve/paper_experiments/suites/fused_expert_smoke.json
```

Use `--case ID` repeatedly to select a subset, `--no-build` only when the exact
snapshot has already been built, and `--no-sync` only when the remote source was
already populated from the same revision. The smoke suite validates the runner;
its 3--5 timing samples are not headline evidence. The pilot suite uses 15
samples and is still exploratory. Paper tables should use a separately declared
suite with at least 31 samples and the workload matrix required by
`docs/moe_paper_readiness.md`.

## External benchmark assets

Large route captures, calibration JSON, and raw benchmark dumps remain outside
Git. A machine may declare `snapshot_external_assets`, where each entry contains
a repository-relative ignored directory and its expected deterministic tree
SHA256. The runner verifies the digest before copying the directory into the
clean snapshot; a missing or modified asset fails closed.

The Arm-codex closure suite requires:

- `bench_assets/moe_paper/dsv4_routes_pt_20260830`, tree SHA256
  `cb263814f56665d6ac0e6af36572b0735d45794e99c95a288b7cae895923e840`;
- `bench_assets/moe_paper/arm_codex_numa3_80c`, tree SHA256
  `6b341ff7bc998f89c8ead2f784a78a6688abd4374779f71b93e7a94836967ebd`.

The route bundle contains three 43-layer, 2048-token TopK6 captures. The
calibration bundle contains the 80-core NUMA3 topology-v2 analytical machine
calibration. The current workspace copies came from the preserved
`tmp/moe_paper_archive/amazon_192c_20260831/` route artifacts and the recorded
Arm-codex calibration. Generated asset directories are ignored and must not be
staged.

Run the three-trace high-skew closure with:

```bash
.venv/bin/python \
  optimizations/fused_moe_sve/paper_experiments/run_matrix.py \
  --machine optimizations/fused_moe_sve/paper_experiments/machines/arm_codex_internal.json \
  --suite optimizations/fused_moe_sve/paper_experiments/suites/arm_high_skew_closure.json
```

The suite runs focused correctness followed by uniformish, median, and
high-skew 31-sample width/order matrices with four rotating packed-weight
copies. It reports both the legacy minimum-expected full winner and the current
one-step width uncertainty gate, so a regression remains visible rather than
being hidden by selection.

The first clean committed run completed successfully at revision `b219627`:
`20260901T080803Z-arm_codex_internal-arm_high_skew_closure-b2196270211b`.
Both external asset hashes matched, focused correctness reported 95 passed, and
all three structured result files were copied locally and retained remotely.
The result summary is
[`../results/arm_codex_80c_high_skew_planner_gate_20260901.md`](../results/arm_codex_80c_high_skew_planner_gate_20260901.md).

The domain-aware cold-phase CP-SAT follow-up is declared in
`suites/arm_cp_sat_shortlist.json`. It runs an independent 60-second proof
solve, a 32-plan repair pool, complete event-model reranking, and eight measured
plans per trace. Prefinal direct-sync results pass the 5% proof-gap and 5%
shortlist-regret gates, but median skew regresses 2.388% versus the one-step
control and fails the 2% no-regression gate. They are recorded in
[`../results/arm_codex_80c_cp_sat_shortlist_prefinal_20260901.md`](../results/arm_codex_80c_cp_sat_shortlist_prefinal_20260901.md).
Because those runs used an uncommitted tree, rerun this suite through
`run_matrix.py` after review and commit before treating it as a paper artifact.

The strict-only decomposition is declared in `suites/arm_strict_greedy_sat.json`.
It retains the original `plan_quick()` homogeneous-LPT plan as an exact fixed
branch of the CP-SAT union and compares one SAT fixed plan with one pure-greedy
fixed plan. Dynamic tail pool, repartition, stealing, and resize are disabled.
Prefinal direct-sync runs selected SAT on all three traces with 2.071--2.129%
union gap and improved paired median latency by 2.21%, 12.49%, and 7.27%; all
three won 31/31 pairs. See
[`../results/arm_codex_80c_strict_greedy_sat_prefinal_20260901.md`](../results/arm_codex_80c_strict_greedy_sat_prefinal_20260901.md).

The topology-preserving width-neighborhood audit is declared in
`suites/arm_width_neighborhood_audit.json`. It evaluates order-only,
domain-local width-only, and their candidate union from the same canonical
strict Plan V2 state, using equal per-operator budgets and one deduplicated
hardware shortlist. The first direct-sync development run is recorded in
[`../results/arm_codex_80c_width_neighborhood_audit_prefinal_20260903.md`](../results/arm_codex_80c_width_neighborhood_audit_prefinal_20260903.md);
rerun after commit before treating it as a paper artifact.

The independent `2x1T -> 1x2T` transition probe and frozen Arm calibration are
described in
[`../results/arm_codex_80c_narrow_lane_merge_calibration_20260903.md`](../results/arm_codex_80c_narrow_lane_merge_calibration_20260903.md).
The `arm_codex_internal_temporal_overhead` machine now points to that v8 asset;
therefore the next width-neighborhood suite run is the real-trace holdout and
must use the same committed model source.

The analytical heavy planner now has an additive placement-aware LLC event
path. Prefinal reruns correct greedy-versus-SAT ranking on high and median skew,
but uniformish regresses 1.89% and wins only 4/31 pairs; full-search time also
increases about 1.7--1.9x. The feature therefore remains experimental. See
[`../results/arm_codex_80c_placement_aware_cost_model_20260901.md`](../results/arm_codex_80c_placement_aware_cost_model_20260901.md).

The exact-M and direct-route cases use the paper's TP4-oriented
`H=4096,F=512` path. The explicit fused/unfused Lab comparator was repaired in
commit `266da2c`: both paths now use the production FEXPA-plus-quadratic SiLU
evaluator, while retaining the intentional rounding difference between two
independent W1/W3 GEMMs and one interleaved W13 GEMM. Production-size controls
pass the declared relative-L2 gate. The case is still absent from the active
suite because its current 31-sample result was collected outside this runner;
add it only together with structured parsing, the numerical gate, and a frozen
source/extension identity.

## Current closure status

The 2026-08-31 Amazon 192-core supplemental study is recorded in
[`../results/amazon_192c_paper_closure_20260831.md`](../results/amazon_192c_paper_closure_20260831.md).
It includes current exact-M, direct-route, tile-window, canonical fusion,
cost-model retention, planner-ranking, and one-kernel/two-stage results. Those
measurements are provisional because most use the pre-`266da2c` extension hash
and were orchestrated by temporary scripts rather than this declarative
runner.

Before artifact freeze, this directory still needs:

- an Amazon 192-core machine definition with NUMA-local placement;
- a >=31-sample closure suite covering the retained kernel, model, and planner
  cases;
- committed structured parsers/table builders for the supplemental outputs;
- a second-Arm-machine repeat from the same frozen commit.

## Adding a machine

Copy one JSON file under `machines/` and set:

- the SSH alias, project root, Python path, and remote result root;
- an explicit CPU/NUMA affinity prefix;
- SVE vector width and machine-sized thread variables;
- runtime environment controls;
- initialized submodules required by the committed build.
- any ignored external source directories, each recorded with a content hash.

Multi-core suites keep `OMP_NUM_THREADS=1` but set `OMP_PROC_BIND=FALSE`.
Otherwise libgomp may narrow the importing process to one CPU before the native
MoE worker pool reads its inherited affinity.

Do not hide machine-specific shape changes in shell scripts. Add a named machine
variable or a separate suite so the rendered command remains visible in each
run's `command.json`.

The first two-machine smoke, including negative findings and the exact raw run
ids, is recorded in
[`../results/arm_codex_aws8_fused_expert_smoke_20260827.md`](../results/arm_codex_aws8_fused_expert_smoke_20260827.md).
