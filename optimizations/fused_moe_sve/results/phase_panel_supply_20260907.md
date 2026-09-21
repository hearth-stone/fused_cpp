# First-M12 versus steady-panel memory events — 2026-09-07

## Decision

The extra memory events are **not confined to the first M12 panel**. The
strongest evidence is M1341/W13/16T: both independent sessions show about
146–152 thousand additional victim LL read-miss events in rows12–1340, versus
only4.6–7.3 thousand in the first12 rows. Steady elapsed time remains near-flat;
the no-PMU timing control reproduces this behavior.

This rejects a first-panel-only explanation for the previously observed
large-M memory-event increase. It supports modeling **exposed** memory cost,
rather than charging every additional miss as serial waiting time. It does not
measure hidden service latency, prove compute overlap is the only mechanism,
or show that the increase is uniform throughout every subsequent panel.

No cost-model formula, calibration, planner, pruning rule or production kernel
was changed. Do not introduce a new fitted physical term from these data.

## Contrast and results

M1341, fixed1T or16T, W13 orW2, isolated versus the same8-reader background.
Each row below reports medians of31 within-round paired event differences
(background minus isolated); events are summed across victim workers.

| Width/stage | First-panel LL read-miss increment S1 / S2 | Steady LL read-miss increment S1 / S2 | Steady paired slowdown S1 / S2 |
|---|---:|---:|---:|
| 1T W13 | +17,225 / +18,925 | +74,140 / +66,904 | +0.187% / +0.203% |
| 1T W2 | +7,264 / +7,861 | +2,438 / +8,473 | +0.133% / +0.147% |
| 16T W13 | +4,626 / +7,289 | **+151,733 / +145,805** | −0.155% / −0.077% |
| 16T W2 | +448 / +1,385 | **+7,654 / +11,564** | +0.086% / −0.345% |

For W13/16T, approximately95–97% of the sum of the two segment event increments
lies after the first panel. This is an approximate attribution from independent
replays, not an exact partition of a single invocation. Steady LL read-miss
increments have bootstrap median95% intervals140,363–157,068 and
133,856–156,029. The increase is independently reproducible.

For W2/16T, steady event increments are also positive in both sessions
(intervals5,082–10,812 and8,891–17,697); first-panel intervals include zero.
For W2/1T, the steady event increment is uncertain in session1
(−5,088–8,445), whereas the first-panel increment is stable. Do not collapse
this into the stronger W13/16T conclusion.

Absolute median segment times, isolated → background, in ms:

| Width/stage | First, session1 | Steady, session1 | First, session2 | Steady, session2 |
|---|---:|---:|---:|---:|
| 1T W13 | 1.20508 → 1.21118 | 133.47272 → 133.71021 | 1.20507 → 1.21092 | 133.44657 → 133.71504 |
| 1T W2 | 0.62019 → 0.61120 | 63.83591 → 63.95501 | 0.61132 → 0.62136 | 63.87075 → 63.95125 |
| 16T W13 | 0.09503 → 0.09698 | 8.37944 → 8.36441 | 0.09178 → 0.09400 | 8.37964 → 8.36622 |
| 16T W2 | 0.04489 → 0.04479 | 4.05228 → 4.05609 | 0.04550 → 0.04530 | 4.04928 → 4.04012 |

Paired ratios need not equal ratios of independent time medians. Full-stage
controls retain the earlier shape dependence: W13/1T about+0.18–0.19%,
W13/16T about−0.27–0.28%; W2 is near-flat. Sub-percent negative differences
are not interpreted as an optimization gain.

W13/16T steady backend-stall paired increments are−533,226 and−109,357
cycles, despite positive miss increments. W13/1T steady stalls rise503,014 and
520,753 cycles, with approximately0.25–0.27 ms extra elapsed time over a
133 ms baseline. Thus "no exposed penalty" fits the16T case better than1T.
The hardware event is generic backend stall, not memory-specific stall.

## Instrumentation and controls

- Three replay variants: full rows[0,1341), first rows[0,12), steady
  rows[12,1341). The steady region contains110 full M12 panels plus one M9
  tail. **Every real replay executes all1341 rows and verifies the full output.**
- First-only counting disables/reads counters before executing the remainder.
  Steady-only counting executes the first panel, then enables counters and
  executes the remainder. These are separate randomized replays on the same
  allocations; no team barrier is inserted between panels.
- Core-only events: cycles, retired instructions, L2 refill, LL read-miss,
  backend stall. No uncore events are collected in this split experiment;
  their previously measured~1.1 ms gating overhead is unsuitable here.
- Every cell has independent reset/read; cumulative enabled/running clocks are
  differenced separately. All core running ratios are1.0.
- The steady boundary gap from first-panel completion to timed steady start
  is recorded. PMU-enabled gaps are on the order of14–19 us; the no-PMU
  control is about0.23–0.28 us. This is a real perturbation, not subtracted
  as though its cache/prefetch effects were known.
- First+steady worker-time sums are compared against the full control.
  Worker sums, not team envelopes, are additive: team envelopes can overlap
  across the unsynchronized boundary. All time/instruction recomposition
  checks pass the preregistered5% warning threshold; time discrepancies are
  below0.2% across the two PMU sessions, instruction discrepancies below0.003%.
- **Counter recomposition is less reliable for1T.** In session2 W13/1T with
  background, first+steady LL read-miss counts differ from full by+20.63%;
  W2/1T isolated differs by+10.27%. These exceed the timing discrepancies and
  prevent precise1T event attribution. All16T LL read-miss recomposition
  discrepancies are within3.2%, supporting the strongest16T conclusion.
- An independent no-PMU session reproduces near-flat16T steady timing:
  W13 8.37765 → 8.36650 ms (paired−0.159%); W2 4.04450 → 4.04130 ms
  (paired−0.014%). W13/1T steady remains slightly slower (+0.212%), as does
  W2/1T (+0.120%). It does not prove PMU leaves miss counts unperturbed.
- First-panel W2/1T timing varies across sessions and instrumentation modes;
  no stable first-panel speedup/slowdown claim is made.

## Interpretation boundary

1. The result establishes substantial **post-first-panel** event increments,
   particularly W13/16T, without a matching total-time penalty. It cannot say
   whether those increments are spread uniformly or cluster in a few later
   panels. No per-panel series was collected.
2. A counted miss is not a serial stall. Prefetch, overlap, reuse and concurrent
   requests can affect exposure; the present data do not separate them or
   determine per-request latency. Do not convert miss counts into "hidden ms".
3. This remains a synthetic phase harness, not the full real-trace workspace
   workload. The larger small-M full-workload penalty remains unexplained.
   First/steady here denotes row intervals, not proof that all first-panel
   reads originate in DRAM or all later reads hit a particular cache.
4. A cold/steady distinction alone is insufficient if the model then assumes
   zero steady memory effects or charges a uniform whole-stage multiplier.
   The next model audit should check how steady transfer demand and its exposed
   time are represented. This diagnosis does not authorize a new formula or fit.

## Reproducibility and validation

Class E, Lab-only; modified `phase_supply_native.cpp`, `bench_phase_supply.py`,
added `analyze_phase_panels.py`, and extended `tests/test_moe_phase_supply.py`.
The original non-panel protocol remains supported. Rollback is confined to
these Lab changes and the manifest/report; production contracts/defaults stay
unchanged. The previous experiment's source archive and data remain intact.

Environment and geometry are unchanged from
[the parent experiment](phase_memory_supply_20260907.md):
Arm-codex-internal, remote root `/home/zhangxu/codex/fused_cpp`, NUMA3;
controller240, victim304 or304–319, background288–295; SVE256 N tile16;
BF16 constant1/64; W13 K4096,N1024 (8 MiB B), W2 K512,N4096 (4 MiB B).
Full owner stripes, R13=R2=1; width1 owns64/256 tiles and width16 owns4/16
tiles for W13/W2. Persistent preallocated/touched outputs,4 B copies,
256 MiB scrub,8×32 MiB reader background with5 ms startup. Ordinary aligned
allocations, no explicit HugeTLB; no new page-policy setting.

Two independent PMU sessions, seeds9207/19207; independent no-PMU timing
control, seed29207. Each has5 warmup+31 recorded randomized rounds of28 cells
(24 real +4 empty). Three sessions complete3024 cells:2592 full-output checks
and432 empty controls. Excluding warmup,2232 real cells and372 controls remain.
Separate correctness and PMU smoke each verified24 real cells before the
formal sessions. Constant-input output checks reuse the parent's exact-value
and poisoned-coverage checks; no new random-input kernel-equivalence claim.

GCC13.2.0 C++17 `-O3`, same standalone build command as the parent report with
`tmp/phase_panel_supply_20260907` replacing `tmp/phase_memory_supply_20260907`.
No production extension rebuild. Binary SHA256:
`53d0633b841c543060c4f66b57b1e4030c65c05967a6ae6567840d62fbd29178`.
Production JIT source remains
`1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629`;
extension remains
`dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`.
Repository base remains `c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5` plus existing
dirty work and these Lab changes. Actual build inputs are archived remotely at
`tmp/phase_panel_supply_20260907/build_sources.tar.gz`.

Remote command (use a fresh output path on rerun):

```sh
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/phase_panel_supply_20260907/bench_phase_supply.py \
  --binary tmp/phase_panel_supply_20260907/phase_supply_native \
  --output tmp/phase_panel_supply_20260907/session1.jsonl \
  --panel-split --core-only --seed 9207
```

Session2 uses seed19207. Timing control replaces `--core-only` with `--no-pmu`
and uses seed29207. Smoke adds `--warmup 0 --rounds 1`.

Raw session1/session2/timing_control/correctness/pmu_smoke JSONL files are
retained both locally and remotely under `tmp/phase_panel_supply_20260907/`.
Local `summary.json` includes all segment medians, paired event/time deltas,
bootstrap intervals, mean/std, P90/P99 and recomposition checks.

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_phase_panels.py \
  tmp/phase_panel_supply_20260907/session1.jsonl \
  tmp/phase_panel_supply_20260907/session2.jsonl \
  tmp/phase_panel_supply_20260907/timing_control.jsonl \
  --output tmp/phase_panel_supply_20260907/summary.json
PYTHONPATH=.:src .venv/bin/python -m pytest -q tests/test_moe_phase_supply.py
```

Focused tests:9 passed. Ruff check/format, clang-format (Google,120 columns),
manifest YAML parse and `git diff --check` passed. No full production suite
rerun, since production code was not changed. No commit requested or made.
The original non-panel protocol also passed its16-real-cell numerical smoke;
`legacy_protocol_smoke.jsonl` is retained in the remote experiment directory.
Recomputing the summary from completed local artifacts gives an exact match.
