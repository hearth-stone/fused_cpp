# L1-hot load/matrix coexistence probe — 2026-09-08

## Decision

Verified L1-hot A/B loading removes the large-versus-small-background penalty
in the four-matrix-instruction probe: paired medians are +0.62/-0.53 us,
versus +33.82/+35.31 us in the retained streaming control. Both hot confidence
intervals include zero. This supports a lower-memory-service-sensitive
interaction, not an explanation based solely on a fixed instruction mix.
It does not identify a particular issue port, queue, or cache return path.
No cost-model parameters, production kernels, planner rules, or pruning change.

## Protocol and reproducibility

- Host: Arm-codex-internal, HiSilicon MIDR 0x480fd020; NUMA3 memory,
  allowed CPUs 240–319, controller 240, victim 304, background 288–303.
  CPU304 has no SMT sibling, 64 KiB L1D and private 1280 KiB L2.
- Victim: M1/1T W13 geometry, 65,536 K4 slots. Background: none,
  16 independent M1 W13 kernels, or 16 M120 W13 kernels.
- Hot variant wraps A addresses over 4 KiB and B over 8 KiB, preserving
  the five load instructions per K4. A selected hot call runs on the victim
  after background startup and before timed measurement. Streaming controls
  are not prewarmed. The hot ring is a diagnostic, not the real GEMM result.
- Two independent processes, seeds 129808/139808, 5 warmup + 31 measured rounds,
  randomized 39-cell grid per round. Same persistent output workspace,
  scrub and four-copy background rotation as the preceding experiment.
- Six core counters: cycles, retired instructions, L1 accesses/refills,
  L2 refills, backend stalls. All core/48 DDRC running ratios pass >=0.99.
  DDRC controller gate is not the victim's kernel duration.
- Ordinary aligned allocations under existing THP policy; no verified
  HugeTLB/page-residency claim. This is a dirty-tree direct-sync Lab experiment,
  not a clean-commit production benchmark.

Command (repeat with session2 and seed139808):

```sh
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/l1_matrix_mix_20260908/bench_phase_supply.py \
  --binary tmp/l1_matrix_mix_20260908/phase_supply_native \
  --output tmp/l1_matrix_mix_20260908/session1.jsonl --l1-mix --seed 129808
```

Analysis:

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_l1_matrix_mix.py \
  --sessions tmp/l1_matrix_mix_20260908/session1.jsonl tmp/l1_matrix_mix_20260908/session2.jsonl \
  --frozen-fit tmp/pressure_response_fit_20260908/frozen_fit.json \
  --output tmp/l1_matrix_mix_20260908/report.json
```

Raw, smoke records and exact build inputs are under local/remote
`tmp/l1_matrix_mix_20260908/` (ignored, not source-committed).

| Identity | SHA256 |
| --- | --- |
| Measured binary | `1b0d8b1ef0a5938ebfa213046d12119dcaebe2971bc6574c5bddd7867388cc08` |
| Frozen fit | `b03819c0c5d04a5ce81424602851533e2dc2e5f8d39d68e104dfdb3bab0c4781` |
| Session1 | `23326190cb7bf27ab03e9490ece0cc8e847b1a829df6f5d13471763b86c1030d` |
| Session2 | `773fc7e0ea0b97c4fd816afaeaae18944e129a70be16de331abdd2e1578642df` |

## L1-state gate

All 36 hot-load cell/session gates pass the predeclared median refill/access
ratio <=0.1% and P90 <=0.5%. Worst cell median is 0.01068%/0.01220%;
worst P90 is 0.01434%/0.01617%. These are PMU event ratios, not exact per-load
hit probabilities; the measurement wrapper also contributes events.

## Large background minus small background

Positive means the large-M background is slower. Values are medians of
round-paired differences, not differences of independent medians.

| Victim | Session1 us [95% bootstrap CI] | Session2 us [95% bootstrap CI] |
| --- | --- | --- |
| Real streaming W13 | +26.10 [21.44,30.90] | +31.04 [26.79,40.79] |
| Streaming AB loads only | -13.64 [-19.93,-10.71] | -9.53 [-11.01,-8.01] |
| Streaming loaded matrix4, no stores | +33.82 [21.83,39.63] | +35.31 [23.54,42.07] |
| Hot AB loads only | -0.17 [-0.39,0.20] | -0.05 [-0.33,0.16] |
| Hot loaded matrix2 | +0.07 [-0.22,0.87] | -0.08 [-1.28,1.19] |
| Hot loaded matrix4 | +0.62 [-2.13,7.06] | -0.53 [-1.96,1.03] |
| Hot loaded matrix8 | 0.00 [-0.17,0.18] | +0.13 [-0.03,0.17] |
| Hot register-resident matrix4 + loads | -0.28 [-4.74,0.71] | +0.01 [-1.89,3.30] |

The streaming counterexample repeats in the same sessions, so disappearance
of its large effect is not simply loss of the original experimental signal.
Small residual effects are not excluded: the first hot matrix4 interval
still permits +7.06 us. Do not claim exact zero or formal equivalence.

## Intrinsic mixed throughput

No-background medians, in us:

| Probe | Session1 | Session2 |
| --- | ---: | ---: |
| Hot loads only | 59.96 | 60.41 |
| Pure matrix2 | 92.14 | 92.65 |
| Hot loads + matrix2 | 134.98 | 136.26 |
| Pure matrix4 | 196.50 | 194.99 |
| Hot loads + matrix4 | 194.93 | 194.35 |
| Pure matrix8 | 389.03 | 387.78 |
| Hot loads + matrix8 | 362.56 | 362.71 |

Matrix2 mixing exceeds the paired max(single controls) by 44.55%/45.78%
(95% intervals 43.24–47.05%/42.47–47.06%). This is evidence against
unconditionally perfect overlap for this instruction schedule, even when hot.
Matrix4 is approximately at the pure-compute time; its paired excess intervals
include zero in both no-background sessions. Matrix8 mixing is faster than
the chosen pure control by approximately 6.5–6.7%.

The latter result is an important limitation, not a negative memory cost:
the pure control uses resident operands and a different instruction schedule.
All controls share loop overhead, so max/sum comparisons are diagnostics,
not independently measured additive resource costs. Hot pointer wrapping also
adds address/control instructions versus streaming. Raw hot-versus-streaming
speedup must not be attributed entirely to cache level. A fixed shared-port
coefficient is therefore not identifiable from this grid.

## Validation and next boundary

- Native build, 39-cell no-PMU and six-core/48-DDRC smoke passed.
- Production B-only and full-no-store4 generated code identity checks passed;
  victim/background numerics and no-store output poisoning passed.
- Two formal sessions pass grid, preparation, counter-running and instruction
  geometry checks; one-round old-protocol smoke passes with the same binary.
- Focused Python checks: 51 passed during implementation; final targeted
  phase-supply/pressure-curve/AB/L1 subset 33 passed. Ruff passes on changed
  Python sources/tests. No production end-to-end benchmark was run.

Keep this reference. If continuing localization, use cache-working-set levels
with the same address-generation template and verify L1/L2 refill changes,
retaining both loaded and register-resident matrix operands. Distinguish
latency/return-pattern exposure from intrinsic scheduling effects before
adding a model term. No threshold, slope or overlap coefficient was fitted here.
