# Isolated W2 panel experiment, 2026-09-09

## Result

Completed the31-cell numerical smoke and two independent5-warmup/31-measured-round sessions. All2263 native cells pass exact arithmetic, selected-route coverage and exhaustive untouched-output checks. W2 M17 is approximately0.96ms with a0.64ms full12-row block and0.32ms5-row tail. Later full blocks approach0.607ms and later5-row tails approach0.297ms.

Much of this history dependence is not specific to accessing the target B: eight disjoint-B W2 prepasses also reduce M17 time substantially. Target-B prepasses provide only another5–8us versus the matched disjoint-B control. The real expert108 route-address layout has a small, noisy effect in isolation, rather than a hundred-microsecond penalty. Neither observation explains the remaining real joint M17/W2 increment error by itself; no competition was present in this experiment.

Class E Lab measurement. Production JIT/kernel, planner, calibration and default dispatch are unchanged. No fitting, full-runtime numerical equivalence or whole-forward speedup is claimed.

## Scope and protocol

Single foreground CPU316 on Arm-codex-internal, NUMA3/membind3, BF16/SVE256. W2 K512/N4096, Ntile16, one full owner stripe; W2 geometry `(t,w2_window_tiles,R2)=(1,0,1)`,4MiB B per copy, four B copies. FP32 direct-route output uses a192MiB allocation, matching the capacity of the earlier workspace experiment. There are no competing workers, W13 invocation, worker pool or merge.

The probe uses unchanged production `kW2Direct` JIT kernels for exact rows1–12 and the same12-row traversal/packed-A addressing as the production dispatcher. Synthetic A varies by row as `(1 + row % 4)/64`, and B is constant1/64, so every output is exactly `(1 + row % 4)/8` in FP32. Each native invocation verifies all selected outputs and every untouched output element. Layout0 uses contiguous route IDs. Layout1 uses the exact17 flat route IDs of real expert108 extracted from the original checked route file/layer. The same output allocation is used in both layouts.

Before each timed target: pack A, poison the entire output, and scan256MiB scrub, all outside timing. Optional prepasses execute eight M12 W2 kernels with separate A/C, using either the selected target B or disjoint B with identical dimensions. Each prepass overwrites its separate output and does not pre-write the target's route output. Then measure the enclosing target and optionally each full/tail kernel call. Exhaustive output verification follows timing. No PMU, explicit cache flush or cache-residency proof is used. The large output-poison/scrub protocol is part of the experimental initial state, and differs from W2 immediately following real W13.

Grid frozen before formal data:19 normal shapes (full M12/24/36/48/96/192 and tails M1/5/11/13/17/23/25/29/35/53/65/101/185), six preconditioning cells for M5/17/185, two contiguous clock controls, and four real-route M17 cells. Total31 cells. Seeds594001/594002,5 warmups and31 measured rounds, randomized cell order and round-modulo4 copy pairing. Smoke seed594000 uses one round/no warmup.

## Full blocks have a repeatable position effect

Second-session medians, us:

| Shape / panel position | Full12-row panel time |
|---|---:|
| Standalone M12 |639.67|
| M192 panel1 |639.86|
| M192 panel2 |632.70|
| M192 panel4 |619.26|
| M192 panel6 |607.58|
| M192 panel16 |606.98|

Session1 similarly starts M192 at640.65us and ends at607.18us. The approximately33us difference is about5% per full block. Positions around the transition vary; do not declare an exact universal warmup block count. A single full-block constant loses this initial-state effect in this protocol.

These elapsed times do not prove that B residency alone causes the transition. In the matched preconditioning experiment below, even disjoint-B execution brings the first full block near608us. Frequency, instruction execution state, and other memory/runtime state were not separately measured, so no specific mechanism among them is identified.

## Five-row tails also depend on preceding execution

Second-session5-row tail medians, us; every row uses the same remainder kernel and no explicit prepass:

| Total M | Preceding full B scans | Tail time |
|---:|---:|---:|
|5|0|320.56|
|17|1|318.33|
|29|2|315.19|
|53|4|298.86|
|65|5|297.54|
|101|8|297.50|
|185|15|297.19|

Session1 M17/M29/M53/M65/M101/M185 tails are317.51/318.16/306.01/298.18/297.87/297.24us. The late plateau repeats, but the transition is not identical across sessions. History count is an experimental input, not a directly measured cache state.

M17 enclosing time is957.18/959.50us in sessions1/2. First-panel medians are635.25/637.61us and tails317.51/318.33us. A sum of separate panel medians need not equal the median enclosing time. Session2 M17 enclosing P10–P90 is940.01–985.95us; CV1.923%. The observed tail-history difference is roughly20us, not the approximately162us real joint-increment miss. Competition could change its magnitude, so this is not a bound on tail behavior under pressure.

The other remainders are retained without forcing monotonicity: standalone M1 is186.12us, the M13 one-row tail225.96us, and the M25 one-row tail146.32us in session2. This counterexample cautions against imposing one smooth cold-to-hot curve across remainder kernels.

## Matched controls separate target-B history from other pre-execution effects

Second-session M17 medians with contiguous route addresses, us:

| Pre-execution | Enclosing W2 | Full12-row block | Tail5-row block |
|---|---:|---:|---:|
| None |959.50|637.61|318.33|
| Eight disjoint-B M12 W2 calls |911.29|608.41|302.92|
| Eight target-B M12 W2 calls |905.16|607.27|297.18|

Both forms of pre-execution make the target faster. The extra benefit associated with target-B selection is much smaller than comparing target-prewarmed execution directly with the unconditioned target.

Paired-round median target-minus-disjoint total deltas are-7.72/-6.92us for contiguous layout, with95% bootstrap intervals[-9.19,-5.67]/[-11.21,-3.92]us. Corresponding tail deltas are-6.91/-5.95us. Real-route layout gives total deltas-5.96/-5.41us and tail deltas-5.16/-5.07us. The intervals use2000 IID paired-round bootstrap resamples and do not remove temporal dependence.

Paired deltas are not differences between marginal medians and need not equal the differences in the table. The matched control supports a small additional target-B-history effect in isolation, but does not prove a particular cache level or uniquely isolate every microarchitectural state.

## Real route addresses do not show a large isolated penalty

For unconditioned M17, real-layout medians are961.41/962.04us versus contiguous957.18/959.50us. Paired real-minus-contiguous effects are+5.92us (95% interval[1.73,11.11]) and+5.15us ([-11.37,13.67]). After either preconditioning, paired layout effects range-0.55 to+0.88us and all intervals includezero.

Thus this isolated test does not reproduce the approximately162us joint-increment miss as a route-address penalty. It does not exclude shared-output/coherence or store-service effects when other workers execute concurrently, because that treatment is absent here. Real addresses alone are not the complete real workload.

## Transfer to the real expert remains conditional

The earlier early-merge-off real M17/W2 stage is0.889ms isolated and1.159ms joint in session2. This probe's approximately0.96ms unconditioned isolated time is not an interchangeable baseline: A/B values differ, A is not produced by real W13, and the pre-target execution and memory initialization differ. Even the approximately0.91ms preconditioned probe should not be treated as matched to that real stage.

The result supports measuring both full-block and tail response with explicit pre-target conditions. It does not correct the real contention model, establish a universal W2 warmup curve, or resolve the M17/W2 joint discrepancy. A subsequent pressure experiment should preserve a defined pre-target state while varying background W13/W2 and team width; real W13→W2 per-panel measurements would address transfer directly.

## Controls, implementation and verification

Three panel-clock-off controls have paired enclosing-time changes between-1.061% and+0.278% across both sessions. The negative difference in one session demonstrates run variation; do not interpret submicrosecond clock overhead from these controls. No timing correction was applied.

New Lab files: `w2_panel_isolated_native.cpp`, `bench_w2_panel_isolated.py`, `tests/test_moe_w2_panel_isolated.py`. Rollback is limited to these files, the manifest entry and this report. Python checks reject wrong stage geometry, missing/inconsistent panel clocks, missing cells despite a completion marker, and missing untouched-output verification. All five focused tests pass; Ruff and `git diff --check` pass. Native31-cell smoke passed before formal sampling. The two formal sessions each have1116 cells, including961 measured cells, and identical native/route hashes. Every sample has valid CPU, timing and numerical checks; all native/build/runner stderr files are empty.

Local/remote root: `tmp/w2_panel_isolated_20260909/`. Raw session JSONL, smoke, stderr/stdout, exact route source and route IDs are retained. `report.json` contains62 cell summaries with medians, P10/P90, mean/CV and paired control effects. `diagnose.py`/`diagnostics.json` retain paired bootstrap contrasts and supplemental P99/standard deviations. No large raw artifacts are staged or committed.

Compiler GCC13.2.0, C++17/O3/pthread, armv8.2-a+bf16+sve/SVE256. Ordinary allocations under existing NUMA page policy; no verified HugeTLB claim. Production JIT SHA256 remains `1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629`. Native source SHA256 `533c1ea6201f1b78414622ff1b21e94be3ad8c8b18f8014d1d94cf7cc27c378a`; runner `9dbc4adda3316162204985dd64024f564c984e64b6564f54493420ed2e7db588`; binary `a393f37e882d38bb3a96049b4a6b755f5a4cf75fa0aab9d4e63d171dceb8ea6a`; routes `d675188ec1cafdb154414d20405607a75ed9f60678f3da4d238c12c2093ffdde`. Source basis is the shared dirty workspace; no clean-commit claim.

## Reproduction

From the remote project root, the retained `tmp/w2_panel_isolated_20260909/build_smoke.sh` exports expert108 routes from the SHA-checked original route file, builds only the standalone probe against unchanged production JIT/Xbyak sources, and runs the smoke. Use fresh output paths for reruns.

Formal invocation, session1 (session2 uses594002 and session2 output):

```sh
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/w2_panel_isolated_20260909/bench_w2_panel_isolated.py measure \
  --binary tmp/w2_panel_isolated_20260909/w2_panel_isolated_native \
  --routes tmp/w2_panel_isolated_20260909/routes.txt \
  --output tmp/w2_panel_isolated_20260909/session1.jsonl --seed 594001
```

Local commands executed:

```sh
.venv/bin/pytest -q tests/test_moe_w2_panel_isolated.py
.venv/bin/python optimizations/fused_moe_sve/benchmarks/bench_w2_panel_isolated.py analyze \
  --sessions tmp/w2_panel_isolated_20260909/session1.jsonl tmp/w2_panel_isolated_20260909/session2.jsonl \
  --output tmp/w2_panel_isolated_20260909/report.json
.venv/bin/python tmp/w2_panel_isolated_20260909/diagnose.py
.venv/bin/ruff check optimizations/fused_moe_sve/benchmarks/bench_w2_panel_isolated.py tests/test_moe_w2_panel_isolated.py
```

The requested standalone W2 collection is complete. Retain this as a bounded reference for initial-state and tail analysis; do not substitute its isolated times into the real joint model without matching the preceding W13/runtime state.
