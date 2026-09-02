# Arm-codex 80-core wide-team cost-model strict gate

Date: 2026-09-02.

## Scope

This experiment evaluates the placement-aware analytical event model after
separately calibrating single-team internal wide-gang dilation `B_t` and
full-cohort dilation `S_t`. It also evaluates an executable event union that
retains the current analytical-full, one-step, and exact-greedy incumbents
alongside the top CP-SAT shortlist. No tail pool, stealing, repartition, or
runtime resize is enabled.

The committed source revision is `6b6b4d10b9d1`. The formal run id is
`20260902T055308Z-arm_codex_internal_wide_pressure-arm_strict_greedy_sat-6b6b4d10b9d1`.
Raw command, stdout/stderr, result JSON, and run summary are under
`tmp/moe_paper_runs/<run-id>/` and are intentionally not source artifacts.

## Machine and method

- Host: `Arm-codex-internal`, NUMA3 CPUs 240--319, memory node 3, SVE256.
- Shape: BF16 TP4 proxy, H=4096, F=512, E=256, 2048 tokens, TopK6.
- Traces: request022/layer20 (uniformish), request016/layer4 (median), and
  request008/layer38 (high-skew). None was used to fit wide-team parameters.
- CP width domain: 1/2/4/8/16T; two 40-core LLC domains.
- Search: 60 s proof root, 60 s pool root, up to 32 diverse solutions, top 8
  after the complete event-model rerank.
- Measurement: 5 warmups, 31 randomized rounds, four rotating packed-weight
  copies; every candidate was bitwise identical to the reference.
- Calibration asset tree SHA256:
  `a06ed06e5f7b5099eedca50490e34df729ea02a648aa782a18725192340f0791`.

The fitted discrete values are:

| Width | Single-team B_t | Full-cohort S_t |
| ---: | ---: | ---: |
| 4 | 1.000 | 1.000 |
| 8 | 1.144 | 1.303 |
| 16 | 1.251 | 1.470 |
| 32 | 1.494 | 1.619 |
| 40 | 1.424 | 1.821 |
| 80 | 2.306 | 2.306 |

## Gate results

The final executable decision minimizes the complete event prediction over
`{full incumbent, one-step incumbent, exact greedy, measured CP shortlist}`.
Negative deltas are improvements.

| Trace | Selected | Predicted / measured | Proof gap | Regret vs measured union | vs one-step | vs greedy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| uniformish | full incumbent | 31.125 / 32.338 ms | 4.02% | 0.00% | -0.02% | -9.50% |
| median | full incumbent | 34.081 / 34.639 ms | 4.99% | 0.00% | -4.86% | -8.02% |
| high-skew | full incumbent | 34.483 / 34.484 ms | 4.81% | 4.03% | -15.75% | -5.66% |

All three pass the declared <=5% surrogate-gap, <=5% measured-regret, and
<=2% no-regression gates. The selected plan beat exact greedy in all 31 paired
rounds on every trace. CP planning plus selected execution was 259.621 s,
589.549 s, and 303.243 s for uniformish, median, and high-skew respectively;
this is an offline oracle, not a request-path planner.

## Interpretation

The wide-team model fixes the earlier escape to unmodeled 32/40/80T plans. In
particular, full-selected prediction error is -0.00%, -1.61%, and -3.75% on
high/median/uniformish. The final union is stable because
it never discards an existing executable incumbent after CP search.

The result does not show that CP-SAT itself dominates the incumbent. All three
event decisions retained `full_selected`. On high-skew, `cp_sat_06` measured
33.148 ms versus 34.484 ms for the selected incumbent, but the event model
predicted it at 34.693 ms and therefore left 4.03% regret. On median all 32 CP
lowerings required the delayed fallback and the best measured CP plan was
40.065 ms; uniformish CP plans lowered directly but the best measured one was
34.044 ms. Thus this experiment closes the declared approximate-plan gate for
the incumbent-plus-shortlist union, while temporal-order/lowering accuracy and
CP-only improvement remain open.

The evidence is limited to one Arm machine, three layers, BF16 TP4 proxy, and
ordinary page policy. It does not close 43-layer coverage, a second Arm host,
production quick-planner quality, or dynamic-tail recourse.
