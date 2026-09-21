# A-source intervention — 2026-09-07

## Decision

Repeating the first A panel largely removes the background-induced steady
LL read-miss increment for M1341/W13/16T. The effect independently repeats:
normal advancing A gives155–159 thousand extra events, while repeated A gives
only2.7–4.1 thousand, with intervals including zero. W2/16T also shows
suppression; W2/1T is inconclusive.

This supports **A advancement being a key contributor to the observed memory
effect**. It does not identify the address of each miss: repeating A reduces
its working set and can also change B residency, prefetch and cache conflicts.
Do not reinterpret the previous aggregate victim events as evidence that B
keeps returning to DRAM after the first panel.

No cost-model formula, calibration, planner, pruning or production code was
changed. These synthetic results do not authorize a physical parameter fit.

## Experiment

Extend the [first/steady panel probe](phase_panel_supply_20260907.md) with two
A-address policies, randomized in the same process on the same allocations:

- **Moving A:** each M12 call reads `A + row*K`, as before.
- **Repeated A:** every call reads the first A panel. The K loop, exact-M JIT,
  B address, N stripe, output address and route IDs are unchanged. The final M9
  call is preserved. All input values are1/64, so both policies have identical
  logical outputs and execute identical matrix operations.

M1341 ×1/16T ×W13/W2 ×background0/1 ×full/first/steady ×moving/repeated A:
48 real cells plus4 shared empty controls per round. Each real cell executes
and verifies the complete1341-row output, even when only one segment is counted.

The primary comparison is the difference in background increments:

```text
(moving_A_background - moving_A_isolated)
  - (repeated_A_background - repeated_A_isolated)
```

Pair all four measurements by round and weight-copy index. This separates
background sensitivity from a change in isolated baseline. Medians of paired
differences need not equal differences of separately computed medians.

## Steady LLC read-miss result

Counts sum victim workers; all entries are medians of31 paired background
increments. These are events, not byte counts or source-attributed accesses.

| Width/stage | Moving A, S1 / S2 | Repeated A, S1 / S2 | Interpretation |
|---|---:|---:|---|
| 16T W13 | +155,472 / +159,021 | +4,079 / +2,650 | Strong repeatable suppression |
| 16T W2 | +14,980 / +10,880 | +2,110 / −1,339 | Suppression repeats, smaller signal |
| 1T W13 | +60,144 / +49,251 | +19,959 / +22,106 | Reduced but residual/noise remains |
| 1T W2 | +10,599 / +270 | +1,829 / +9,858 | No stable policy effect |

For W13/16T, the moving-A intervals are150,796–166,098 and149,566–169,293;
the repeated-A intervals are−2,874–10,305 and−3,947–13,784.
Using the point estimates, the increment is about97–98% smaller, but do not
treat this percentage as a directly measured fraction of A-address misses.

The paired difference-in-differences is150,488 (95% interval137,093–162,042)
and148,473 (135,553–165,213). W2/16T gives11,531 (2,325–24,419) and16,119
(6,962–22,026). W2/1T intervals include zero in both sessions.

Important controls:

- The first panel accesses the same A region in both policies. Its miss
  difference-in-differences includes zero for every width/stage in both
  sessions, consistent with the intervention acting after the first panel.
- W13/16T full-stage controls also show suppression: difference-in-differences
  is151,704 and155,226, with positive95% intervals in both sessions. Thus the
  primary result is not only a segmented-counter artifact.
- Retired instruction counts agree between A policies and background states
  for each width/stage/segment. No computation was skipped.
- Suppressing the **background increment** does not eliminate baseline misses.
  For example, repeated-A W13/16T steady in session1 still has median114,390
  isolated and114,969 background LL read-miss events. It is not a proof that
  every steady read now hits private L2.

## Time and measurement quality

W13/16T steady absolute median times (ms):

| A policy | Session1 isolated → background | Session2 isolated → background |
|---|---:|---:|
| Moving | 8.37845 → 8.37224 | 8.38371 → 8.37432 |
| Repeated | 8.32394 → 8.33616 | 8.32665 → 8.33529 |

Both remain near-flat under background: moving-A paired changes are−0.105%
and−0.078%; repeated-A changes are+0.199% and+0.128%. The large event
suppression is **not** a comparable reduction in elapsed-time penalty.
This is an address-locality intervention, not a production optimization proposal.

The no-PMU timing control also has only small background penalties: W13/16T
moving A8.37173 →8.38879 ms (paired+0.176%), repeated A8.33142 →8.36172 ms
(+0.386%). W2/16T paired changes are+0.294% and+0.073%. Small signs/magnitudes
vary across sessions; event suppression should not be advertised as a stable
elapsed-time benefit.

Core events and panel boundaries are unchanged from the parent experiment.
No uncore events are collected: their gate is too wide for panel attribution.
Steady PMU enable gaps are retained in raw records, not subtracted as though
their cache effects were known. Full-stage controls have no inserted boundary.

First+steady worker-time recomposition deviations are at most0.24% in session1
and0.41% in session2; instruction deviations are below0.003%. Miss counts are
less additive: W13/16T deviations reach5.16%, and W2/16T8.20% in session1.
Do not use segment medians as an exact event ledger. The W13/16T full-stage
control and independent repetitions support the qualitative suppression result.

The raw/summary artifacts retain mean/std, P90/P99, paired confidence intervals,
per-worker counts and boundary gaps. Bootstrap intervals describe repeated
samples within a session, not uncertainty across machines or workloads.

## Scope and next interpretation

At16T, each thread owns512 KiB of W13 B or256 KiB of W2 B; CPU304 reports
1280 KiB private L2. A new M12 panel introduces96 KiB of W13 A or12 KiB of
W2 A per worker. Holding A fixed is consistent with reducing fresh-A supply
and its interference with cached data. At1T B is8/4 MiB, exceeding L2, so
identical source assumptions do not apply.

The experiment isolates **address progression**, not a physical memory tier.
Possible mechanisms include direct A fetches, changed B replacement, shared
cache reuse, and prefetch. It cannot distinguish those mechanisms, attribute
DDRC latency, or establish that all weights are cold DRAM reads. It also does
not reproduce the full real-trace workload's much larger small-M penalty.

For future model work, distinguish input-panel supply from repeated weight
supply and from exposed waiting. Do not fit a "B contention" correction using
these aggregate counters. No further physical probe is automatically authorized
by this report; address-attributed sampling would be a separate follow-up.

## Reproduction and retained evidence

Class E Lab diagnostic. Changes are limited to A-source selection in
`phase_supply_native.cpp`, `bench_phase_supply.py`, new `analyze_a_source.py`,
focused tests and manifest/report. Original whole-stage and panel modes retain
their default moving-A behavior. No production contract/default migration.

Host `Arm-codex-internal`, root `/home/zhangxu/codex/fused_cpp`; NUMA3 memory,
controller240, victim304 or304–319, background288–295. Same8×32 MiB background,
5 ms startup,256 MiB scrub,4 B copies and persistent touched output as parent.
W13 K4096,N1024 (8 MiB B), W2 K512,N4096 (4 MiB B); BF16, SVE256 N tile16,
full N owner stripes, R13=R2=1. Width1 owns64/256 tiles; width16 owns4/16 tiles.
No explicit HugeTLB or page-policy change. This is the synthetic standalone
JIT harness, not the production workspace object or a real-trace replay.

Two PMU sessions with seeds9307/19307 and a no-PMU timing control with seed29307;
each5 warmup+31 measured rounds. Raw session1/session2/timing_control and
correctness/pmu_smoke JSONL files reside under `tmp/phase_a_source_20260907/`
locally and remotely. Summary is `summary.json` in the local directory.
All three sessions completed5616 cells:5184 real full-output checks and432
empty controls. After excluding warmup,4464 real cells and372 controls remain.
All core PMU running ratios are1.0. Correctness and PMU smoke each additionally
verified48 real cells before the deciding sessions.

GCC13.2.0 C++17 `-O3`, same build command as the
[original report](phase_memory_supply_20260907.md), replacing its experiment
directory with `tmp/phase_a_source_20260907`. Build inputs archived remotely
as `build_sources.tar.gz` in that directory. Binary SHA256:
`a2e40e9154480dcb63b3a0089c8ba901e18ee2f028545e5d83104d2273d03eda`.
JIT source remains `1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629`;
production extension remains `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`.
Repository base `c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5` plus existing dirty
work and these Lab changes. Prior experimental snapshots remain intact.

```sh
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/phase_a_source_20260907/bench_phase_supply.py \
  --binary tmp/phase_a_source_20260907/phase_supply_native \
  --output tmp/phase_a_source_20260907/session1.jsonl \
  --a-source-contrast --core-only --seed 9307
```

Use seed19307 for session2; replace `--core-only` with `--no-pmu` and use29307
for timing control. Smoke adds `--warmup 0 --rounds 1`. Use fresh output paths;
exclusive creation intentionally prevents overwriting existing evidence.

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_a_source.py \
  tmp/phase_a_source_20260907/session1.jsonl \
  tmp/phase_a_source_20260907/session2.jsonl \
  tmp/phase_a_source_20260907/timing_control.jsonl \
  --output tmp/phase_a_source_20260907/summary.json
PYTHONPATH=.:src .venv/bin/python -m pytest -q tests/test_moe_phase_supply.py
```

Focused tests:12 passed, including grid pairing, incomplete-artifact rejection
and synthetic difference-in-differences arithmetic. The runner also checks
native-policy correspondence on every intervention cell. Native constant-value and poisoned-output
checks verify full outputs; this is not a new random-input production kernel
equivalence claim. Ruff, clang-format, YAML parse and `git diff --check` passed.
No production suite rerun or commit; production code did not change.
Original non-panel protocol passed its16-real-cell numerical smoke, retained
remotely as `legacy_smoke.jsonl`. Completed-artifact summary replay matches
the stored summary exactly.
