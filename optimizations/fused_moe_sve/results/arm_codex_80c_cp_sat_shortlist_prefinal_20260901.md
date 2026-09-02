# Arm 80C LLC-domain CP-SAT shortlist prefinal

## Decision

The first domain-aware cold-phase shortlist closes solver, lowering, and
within-pool regret on all three captured traces, but it is **not accepted** for
43-layer or second-Arm expansion yet.  The median-skew trace regresses by
2.388% versus the current one-step gate, exceeding the declared 2% limit.

These are prefinal measurements from an uncommitted working tree synchronized
directly to `Arm-codex-internal`; they are not the commit-bound output of the
paper experiment runner.  The retained suite is
`paper_experiments/suites/arm_cp_sat_shortlist.json`.  A final artifact requires
a reviewed commit followed by a clean runner repeat.

## Method

- Machine: Arm-codex NUMA3, logical CPUs 240--319, `membind=3`, two 40-core LLC
  domains.
- Operator: BF16 SVE fused expert, H=4096, F=512, E=256, 2048-token TopK6
  captured routes.
- Calibration SHA256:
  `e0ec1cd4ede5dfbdb1ef1748807357292cbad2b1786431642a9934908e42884b`.
- Extension SHA256:
  `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`.
- Current one-step full plan projected to single-domain placement and used as
  the CP-SAT hint/incumbent.
- Proof solve: 60 s, 8 workers, requested relative gap 1%.
- Pool solve: independent 10 s root; top-16 route experts remain mutable;
  non-hot intervals are fixed; schedule-start no-good cuts generate 32 plans;
  pool objective slack is 50% and is not an optimality claim.
- All 32 plans use deterministic stage windows, contiguous-core lowering, and
  complete analytical event-model reranking; only the top 8 are measured.
- Measurement: 5 warmups, 31 randomized interleaved runs, four rotating packed
  weight copies.  Outputs are bit-exact across measured plans.
- Resource bound: calibrated-service, GEMM-only mode-relaxed LP with compulsory
  cold weights.  It is not a hardware-peak certificate.

## Results

| Trace | CP proof UB/LB (ms) | Gap | Plans lowered | Legacy full (ms) | One-step (ms) | Event-selected CP (ms) | Measured pool best (ms) | Pool regret | CP vs one-step | Resource LB (ms) | Execute/LB | T_plan (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| High skew, layer 38 | 23.710 / 23.154 | 2.345% | 32 / 32 | 38.216 | 34.891 | 31.269 | 31.166 | 0.330% | -10.380% | 21.230 | 1.473x | 168.235 |
| Median skew, layer 4 | 23.103 / 22.647 | 1.974% | 32 / 32 | 42.643 | 35.367 | 36.211 | 36.211 | 0.000% | **+2.388%** | 21.223 | 1.706x | 150.377 |
| Uniformish, layer 20 | 22.959 / 22.613 | 1.507% | 32 / 32 | 34.377 | 32.839 | 32.630 | 32.562 | 0.207% | -0.638% | 21.233 | 1.537x | 141.002 |

`T_plan + T_execute` is 168.266 s, 150.413 s, and 141.034 s respectively.
Planning time includes incumbent projection, the independent proof and pool
solves, 32 lowerings, event reranking, and surrounding Python orchestration; it
does not include the existing analytical full-control search reported
separately by the benchmark.

The best measured homogeneous fixed controls were 35.445 ms (`16T
reverse-even`) on high skew, 36.166 ms (`16T reverse-odd`) on median skew, and
32.690 ms (`8T reverse-odd`) on uniformish. Thus the selected CP plan beats the
best fixed control on high skew and uniformish, while the current one-step gate
remains the best measured controlled choice on median skew.

## Gate audit

| Gate | Result |
| --- | --- |
| CP-SAT gap <=1%, or accepted 5% | Pass at the 5% fallback: maximum 2.345% |
| 32 diverse schedules and executable lowering | Pass: 32/32 on every trace |
| Measured shortlist regret <=5% | Pass: maximum 0.330% |
| No trace more than 2% slower than one-step | **Fail: median +2.388%** |
| Report `T_plan + T_execute` | Pass |

## Interpretation and next decision

The CP surrogate can produce a tight near-optimal bound and a useful executable
pool.  The high-skew result also shows substantial scheduling headroom over the
current gate.  The remaining failure is cross-family selection: the complete
analytical event model ranks the current one-step plan at 23.564 ms and the
median CP plan at 23.960 ms, correctly preferring the incumbent there, but on
high skew it also prefers the 25.030 ms incumbent over CP plans predicted near
33.4 ms even though CP is 10.38% faster in hardware.  A blanket incumbent
fallback would remove the median regression but discard the high-skew gain.

Therefore the next experiment should target a holdout-safe selector between
the incumbent and CP family, or correct the event model's cross-family
placement/contention ranking.  Do not expand to all 43 layers or a second Arm
until that selector passes the same three-trace gate.  Production quick/full,
Plan V2, kernels, schemas, ABI, and default dispatch remain unchanged.
