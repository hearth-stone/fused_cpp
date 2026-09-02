# Arm 80C executable order-neighborhood audit

## Outcome

Step 1 rejects deterministic order-only VND/LNS with the current heavy event
model. The executable neighborhood is large and cheap enough to sample, but its
predicted local improvements do not survive hardware execution. Critical-event
selection also does not consistently enrich improving candidates relative to
the equal-budget random control.

This result does not reject the canonical executable plan state. It rejects the
next optimization step that would use the current event score to accept
order-only moves.

## Reproduction identity

- Commit: `5252d67480a7`
- Formal run:
  `20260902T100827Z-arm_codex_internal_wide_pressure-arm_executable_neighborhood_audit-5252d67480a7`
- Machine: `Arm-codex-internal`, NUMA node 3, physical CPUs `240-319`
- Model: BF16 TP4, `H=4096`, `F=512`, `E=256`, 2048 tokens, TopK6
- Calibration:
  `analytic_machine_numa3_80c_topology_wide_pressure_v3_20260901.json`
- Backend: Arm SVE BF16, 256-bit SVE, direct-route W2 enabled
- Measurement: 5 warmups, 31 randomized paired rounds, 4 rotating packed-weight
  copies, exact output equality against `full_selected`
- Search budget: 32 selected experts per arm, at most 64 event evaluations per
  operator, event top-4 per arm measured on hardware

Command:

```bash
.venv/bin/python optimizations/fused_moe_sve/paper_experiments/run_matrix.py \
  --machine optimizations/fused_moe_sve/paper_experiments/machines/arm_codex_internal_wide_pressure.json \
  --suite optimizations/fused_moe_sve/paper_experiments/suites/arm_executable_neighborhood_audit.json
```

The setup build passed. The focused target test set passed before measurement.

## Neighborhood and event evaluation

| Trace | Full shape | Full plan ms | Critical proposed / sampled | Random proposed / sampled | Event eval rate plans/s (C/R) |
| --- | --- | ---: | ---: | ---: | ---: |
| Uniformish | `10x8T` | 97,995 | 14,051 / 288 | 14,085 / 309 | 8.45 / 8.32 |
| Median | `3x16T + 32x1T` | 90,423 | 9,241 / 294 | 9,251 / 307 | 5.70 / 5.64 |
| High-skew | `4x16T + 16x1T` | 62,302 | 9,203 / 295 | 12,718 / 266 | 8.60 / 8.60 |

All generated proposals were legal executable states. Canonical hashing removed
41--58 duplicates per arm. The initial full planner still dominates planning
latency; sampled event evaluation itself took 31--54 seconds per arm.

Critical selection did not consistently improve event-space hit rate:

| Trace | Critical improving fraction | Random improving fraction | Best event gain C/R |
| --- | ---: | ---: | ---: |
| Uniformish | 22.6% | 24.6% | 0.622% / 0.666% |
| Median | 20.4% | 43.3% | 0.605% / 0.203% |
| High-skew | 48.8% | 47.4% | 0.115% / 0.092% |

## Hardware validation of event-top candidates

| Trace | Baseline median ms | Event/hardware Spearman | Stable improvements | Best paired median | Worst paired median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniformish | 32.628 | 0.143 | 0 / 8 | +0.256% | -0.157% |
| Median | 35.390 | -0.156 | 0 / 8 | +0.337% | -9.028% |
| High-skew | 34.653 | -0.690 | 0 / 8 | -0.050% | -14.497% |

“Stable” requires both paired median and paired P10 speedup to be positive.
Every one of the 24 candidates failed this criterion. The small positive
medians on uniformish and median remain inside paired noise because their P10
values are negative.

High-skew is a direct ranking counterexample. The event model predicted only
`0.068%--0.115%` gains for the eight selected moves, but all eight lost on
hardware. The critical top-4 lost `5.16%--11.78%`; the random top-4 lost
`0.05%--14.50%`. These are not marginal noise-only reversals.

## Decision and remaining work

Do not implement Step-2 VND against the current event evaluator. Repeated local
acceptance would optimize model artifacts and can move far away from the strong
`full_selected` incumbent, especially on high-skew temporal schedules.

The next valid choices are:

1. correct temporal-order/event ranking, using the measured counterexamples as
   holdouts, then rerun this exact audit; or
2. use template-level global search whose candidates are always measured or
   conservatively retained alongside immutable full/one-step/greedy incumbents.

This experiment covers one Arm machine and three captured layers. It does not
establish behavior across all 43 layers or another Arm system. Those expansions
remain inappropriate until the local ranking failure is resolved.
