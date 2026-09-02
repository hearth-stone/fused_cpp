# Arm-codex 80C narrow-lane merge calibration

## Result

The independent probe rejects the hypothesis that the high-skew lane-merge
miss is caused by a missing extra 1T slowdown. The dominant error is instead a
width-dependent overprediction of background contention for merged 2T lanes,
combined with an underpredicted isolated 2T per-expert overhead.

The calibrated analytical model adds:

- a discrete 2T expert overhead of `88,799.963 + 5,909.533 * routes` ns;
- a full-cohort narrow-team contention correction of `0.784898` for 1T and
  `0.532344` for 2T.

The correction may be below one because it corrects an existing analytical
overestimate; the final event dilation remains clamped to at least one. It is
applied only to explicitly calibrated widths and interpolates with measured
peer occupancy from identity at zero peers. Old calibrations and widths other
than 1T/2T are unchanged.

## Experimental design

- Machine: `Arm-codex-internal`, NUMA3 CPUs `240-319`, memory node 3.
- Operator: BF16 fused expert, `H=4096`, `F=512`, SVE exact-M/direct-route.
- Target cores: logical cores 64 and 65 within the 80-core rank.
- States: `2x1T` versus `1x2T`, both with background delayed and with the same
  concurrent `4x16T + 14x1T` background.
- Route cases: balanced, head-heavy, and tail-dense synthetic short-expert
  sequences; none is copied from the three planner holdout traces.
- Measurement: early merge off, five warmups, 31 randomized paired trace calls,
  four rotating packed-weight copies.
- Fit split: `merge_isolated` per-task spans fit the 2T fixed/per-route overhead;
  background target spans fit the two contention corrections through the
  complete placed event simulator. Isolated 1T absolute spans and all captured
  planner traces are excluded from the correction fit.
- Output correctness: all four plans in each case are bitwise identical before
  tracing.

The raw development result is retained outside Git at
`tmp/narrow_lane_merge_transition_v4.json`. The generated calibration is
`bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json`;
the containing external-asset tree SHA256 is
`23f5f7d222029181b1541d6dab5a01df23c2e340a2bcdb9f85371887fbbc68df`.

## Measurements and model error

Measured background/isolated target-span dilation is nearly invariant across
the topology change:

| Case | 2x1T measured dilation | 1x2T measured dilation |
| --- | ---: | ---: |
| balanced | 1.0852 | 1.0915 |
| head-heavy | 1.0726 | 1.0865 |
| tail-dense | 1.1389 | 1.1386 |

Before calibration, the model underpredicts isolated merged-2T spans by
`12.50%--20.86%` and overpredicts merged-2T background spans by
`17.55%--27.71%`. The calibrated result is:

| Case | 2x1T isolated error | 1x2T isolated error | 2x1T background error | 1x2T background error |
| --- | ---: | ---: | ---: | ---: |
| balanced | -1.74% | -1.08% | +0.12% | -1.89% |
| head-heavy | +2.37% | +1.68% | -0.63% | +1.01% |
| tail-dense | -2.85% | -2.95% | 0.00% | 0.00% |

The calibrated pair/merge background ratios are `1.0738`, `1.3621`, and
`0.9670`, versus measured `1.0522`, `1.3846`, and `0.9670`. This is sufficient
to advance to the frozen real-trace holdout, but it is not evidence that the
full planner ranking is closed.

After freezing the parameters, an event-only high-skew precheck increased the
best predicted lane-merge gain from about `0.020%` to `0.140%` and ranked 15 of
16 merge neighbors as positive. A split still ranked first at `0.278%`, and no
candidate crossed the 2% robust gate. No parameter was changed after this
precheck; real-trace hardware measurements remain the decisive holdout.

## Scope and next gate

This calibration is machine-, topology-, operator-, dtype-, and shape-specific.
It does not support interpolation to another Arm CPU, another rank size, W8A8,
W8A16, or a different H/F shape. It does not alter production quick planning or
Plan V2 execution semantics.

The decisive commit-bound `arm_width_neighborhood_audit` holdout subsequently
completed from revision `1fcbc7b`. It retained baseline on all three traces and
did not reproduce the prefinal merge gains; instead, it ranked a stable
high-skew `16T -> 8T+8T` split first within that run. A same-commit repeat kept
the split's median positive but not its P10, and no candidate was positive-P10
in both formal sessions. See
[`arm_codex_80c_width_neighborhood_narrow_calibrated_20260903.md`](arm_codex_80c_width_neighborhood_narrow_calibrated_20260903.md)
for the final decision.
