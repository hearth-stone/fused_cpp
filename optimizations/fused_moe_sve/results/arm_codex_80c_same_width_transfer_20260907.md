# Bounded same-width task transfer

Status: completed after SSH recovered on 2026-09-07. Four independent process
sessions pass output equality and artifact/round validation. Neither trace has a
candidate passing the frozen both-session gate. Retain both anchors. No production
defaults, Plan V2, kernel, calibration formula or comparator radii changed.

## Outcome

Values below are session1/session2. Gains are medians of31 paired ratios
`100*(anchor_ns/candidate_ns-1)`, not ratios of latency medians.

| Trace / candidate | Median latency ms S1/S2 | Paired gain % S1/S2 | Gain P10 % S1/S2 |
| --- | ---: | ---: | ---: |
| median anchor | 31.8573 / 32.0395 | 0 / 0 | 0 / 0 |
| same_llc_swap_0 | 31.6368 / 31.7927 | +0.750 / +0.576 | +0.124 / +0.180 |
| same_llc_relocation_0 | 31.9903 / 32.1821 | -0.416 / -0.500 | -0.718 / -1.063 |
| cross_llc_swap_0 | 31.8156 / 32.0545 | +0.146 / -0.221 | -0.571 / -1.066 |
| cross_llc_relocation_0 | 31.9379 / 32.0371 | -0.245 / -0.126 | -0.571 / -0.966 |
| same_llc_swap_1 | 31.9067 / 32.0130 | -0.104 / +0.053 | -0.570 / -0.691 |
| same_llc_relocation_1 | 33.4837 / 33.5523 | -4.894 / -4.471 | -6.096 / -5.223 |
| high-skew anchor | 30.5414 / 30.3795 | 0 / 0 | 0 / 0 |
| same_llc_swap_0 | 32.9008 / 32.6461 | -7.121 / -6.917 | -7.566 / -7.311 |
| same_llc_relocation_0 | 35.1564 / 35.0043 | -13.123 / -13.180 | -13.454 / -13.452 |
| cross_llc_swap_0 | 32.5722 / 32.5814 | -6.238 / -6.746 | -6.511 / -7.057 |
| cross_llc_relocation_0 | 32.4332 / 32.3529 | -5.802 / -6.105 | -6.157 / -7.169 |
| same_llc_swap_1 | 30.7536 / 30.6091 | -0.585 / -0.773 | -1.058 / -1.303 |
| same_llc_relocation_1 | 30.5422 / 30.3437 | -0.015 / +0.172 | -0.412 / -0.225 |

Median has a repeatable small gain, not an actionable >2% improvement. Preserve
`same_llc_swap_0` (state `5d5bdf80a911788e57694eccdf1ab11e4aaf3d0520610b7761b8467f740fdc04`)
as a measured reference, not a promoted anchor. It exchanges expert61(M3) and
expert253(M106) between8T lanes on CPUs256–263 and272–279.

High-skew's model-best representatives predict small positive gains but regress
5.8–13.2%. In particular, `same_llc_relocation_0` predicts +0.504% yet measures
-13.123/-13.180%. It moves expert211(M43) from1T lane14 position8 to1T lane17
position1. Their isolated lane loads change from23.067/23.070ms to16.230/29.908ms.
This exposes a large ownership/load change hidden by a small model-score delta;
it does not establish bandwidth contention as the cause without new diagnostics.

Decision: stop this bounded transfer layer and retain the original anchors. Do
not infer that the entire same-width neighborhood is flat: only6/96 proposals
per trace were measured, with model-influenced selection. Next useful local work
is a replay-only audit of isolated critical-lane/load changes versus measured
regressions, before authorizing another shortlist or adding a model term. This
experiment does not validate automatic pruning or a new residual calibration.

## Frozen scope

Use retained median score-best and high-skew GEMM-density smooth_global anchors.
The previous median block near-elite remains historical reference only, without
an extra hardware slot. Each trace samples at most 24 unique candidates for each
of same-LLC swap, same-LLC relocation, cross-LLC swap, cross-LLC relocation.
Sampling uses deterministic random seed 20260906 plus family index, at most
1024 attempts per family. This is a bounded sample, not exhaustive enumeration.

Only equal-width, nonempty, fully single-domain lanes participate; relocation
does not empty the donor. Lanes spanning the domain boundary are unchanged.
Domain mapping is explicitly CPUs240–279=SCCL27 and280–319=SCCL25. Whole task
objects retain expert/routes/windows. Width, core ownership and merge stay fixed;
serial dependencies are rebuilt from the resulting sequences.

Each family reserves its model-best candidate. Remaining slots retain maximum
task-position distance representatives in family order (same-domain swap then
relocation), at most six candidates plus anchor. The allocation is intentionally
asymmetric, not an equal-budget same-versus-cross-LLC comparison. Frozen v8 scores
are proposals only; no old radius is used for pruning. Full executable canonical
payloads, complete bridges, anchor hash and before/after affected-lane sequences,
route counts and isolated loads are saved for every retained candidate.

Local artifacts: `tmp/same_width_transfer_20260907/{median,high_skew}_frozen.json`.
Do not overwrite these or refit after measurements. Adding domain metadata changes
canonical state identity but not the anchor's complete executable bridge.

## Reproduction commands

Freeze entrypoint is `bench_bounded_order_extension.py freeze --proposal-set transfer`.
Median uses `--source-plans tmp/selection_pair_20260906/plans.json` and
`--winner tmp/selection_winners_20260906/median.json`. High-skew uses
`--source-plans tmp/selection_other_traces_20260906/high_skew_plans.json`,
`--anchor-frontier tmp/gemm_density_20260907/high_skew_frozen.json`,
`--anchor-key smooth_global` and its two `high_skew_session{1,2}.json` files.
Both use frozen calibration
`bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json`.

The resumed run copied only this owned runner and frozen frontiers to
`Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/same_width_transfer_20260907/`.
No production changes were synced and no rebuild was performed.
Use the remote project virtualenv and existing extension hash from each frontier.
For each trace run two independent processes with seeds20260907/20260908:

```bash
env OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:src \
  timeout 300 numactl --physcpubind=240-319 --membind=3 \
  .venv/bin/python tmp/same_width_transfer_20260907/bench_bounded_order_extension.py measure \
  --frontier tmp/same_width_transfer_20260907/median_frozen.json \
  --route-file bench_assets/moe_paper/dsv4_routes_pt_20260830/measured_request016_case017_zh2048-018.pt \
  --seed 20260907 --output tmp/same_width_transfer_20260907/median_session1.json
```

High-skew route file is `measured_request008_case009_zh2048-010.pt`; layer is
embedded in its frontier. E256/H4096/F512,2048tokens,TopK6,BF16,full owner stripes,
early merge unchanged. Same process/allocations, four weight copies,216MiB scrub
before each call, randomized plan order, five warmups and31 effective paired
rounds. All plans/copies must pass zero-tolerance output comparison before timing.
Scrub is not proof of exclusively DRAM-sourced weights.

Analyze with `analyze_bounded_order_extension.py --help` for its existing CLI,
or `analyze(frontier_path, [session1_path, session2_path])`. Pass criterion remains
paired median speedup greater than2% and P10 greater than0 in both sessions.
Report absolute anchor/candidate latency and all regressions. No winner is promoted
without this evidence, and failure would apply only to this sampled shortlist.

## Validation

`PYTHONPATH=.:src .venv/bin/python -m pytest -q tests/test_moe_bounded_order_extension.py
tests/test_moe_order_strategy_compare.py tests/test_moe_executable_plan_state.py`:
25 passed. Ruff on the touched runner/tests, manifest YAML parsing and
`git diff --check` pass. Each trace scored96 candidates and retained6 plus anchor.
Diagnostic local freeze time: median4.214s, high-skew7.920s; scoring4.119/7.779s.
Best predicted gain is median0.383%, high-skew0.504%, not measured speedup.
All four sessions completed, each with7 plans x36 rounds (5 warmups,31 measured).
Each session verifies all7 plans x4 copies against the anchor with zero tolerance.
Analyzer verifies complete unique cells, pairing, independent seeds and matching
frontier/runner/extension identities; both `stable_candidates` lists are empty.
Local and remote raw files are `<trace>_session{1,2}.json`; local summaries are
`<trace>_summary.json` in the same ignored directory. No raw evidence was staged.

Hardware: HiSilicon320 cores, Linux5.10.0-247.0.0.146.oe2203sp4.aarch64, boot id
`8da74bfd-16d5-45a6-a6ad-1e0cd3af3b19`. Initial load0.04/0.01/0.00, no competing
benchmark found. THP policy `always`; actual allocation backing was not audited.
Reuse frozen extension SHA256
`dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`
and calibration SHA256
`7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`.
Backend Ntile8, per-expert W13/W2 bytes8MiB/4MiB; `(t,0,0,1,1)` full stripes.
Median shape1x16T+8x8T; high-skew4x16T+16x1T. No PMU collected, no causal
bandwidth claim, no cross-machine or full-model E2E claim. No commit made.
