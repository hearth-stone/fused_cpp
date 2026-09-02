# Arm 80C placement-aware LLC event model

## Decision

The placement data path and per-domain LLC model are implemented and validated,
but the feature does **not** pass the three-trace ranking gate. It corrects the
SAT-versus-greedy direction on high and median skew; uniformish still regresses
1.89% and wins only 4 of 31 pairs. The remaining dominant error is wide-team
concurrent pressure/width scaling, not missing LLC placement.

These are prefinal direct-sync measurements from an uncommitted tree, not a
commit-bound paper-runner artifact.

## Implementation

- `IntervalPlanner._score()` maps each task's logical `(core_begin, threads)`
  through `cpu_ids` and calls the additive `dag_makespan_placed()` API.
- Each active phase partitions working set and LLC demand by owner-thread share
  across calibrated LLC domains.
- Each domain gets an independent capacity miss fraction, service curve,
  offered rate, utilization, and dilation.
- Multiple domains additionally share the calibrated rank-level LLC fabric
  ceiling.
- A cross-domain gang uses the maximum dilation of its active domains and the
  rank-level dilation.
- Task-specific spill affects only that task's spillable DRAM demand. DRAM
  contention remains NUMA-rank global.
- Placed tasks sharing a physical CPU must be ordered by a dependency path.
- The old unplaced API and no-topology fallback retain rank-global behavior.
- Empirical models, production quick planning, Plan V2, runtime, and kernels are
  unchanged.

## Method

- Arm-codex NUMA3, CPUs 240--319, `membind=3`, two 40-core LLC domains.
- BF16 SVE fused expert, H=4096, F=512, E=256, 2048-token TopK6 traces.
- One strict SAT proof incumbent versus the exact pure-greedy strict branch.
- No tail pool, repartition, stealing, resize, or modeled release gate.
- Five warmups, 31 randomized paired runs, four rotating packed-weight copies.
- Calibration SHA256:
  `e0ec1cd4ede5dfbdb1ef1748807357292cbad2b1786431642a9934908e42884b`.

## Results

| Trace | Greedy/SAT event prediction (ms) | Predicted SAT vs greedy | Greedy/SAT measured (ms) | Measured SAT vs greedy | Paired P10/P90 | SAT wins | Union gap | Full-plan time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| High skew, layer 38 | 26.974 / 26.306 | -2.47% | 37.766 / 35.942 | -4.83% | -5.59 / -4.06% | 31 / 31 | 2.275% | 52.687 |
| Median skew, layer 4 | 25.249 / 24.497 | -2.98% | 38.426 / 34.231 | -10.92% | -11.35 / -9.96% | 31 / 31 | 1.838% | 77.385 |
| Uniformish, layer 20 | 25.160 / 23.599 | -6.20% | 35.970 / 36.651 | **+1.89%** | -0.15 / +4.46% | **4 / 31** | 1.934% | 83.801 |

Placement-aware event prediction now gets the high/median ranking direction
right, although it still underestimates their measured SAT benefit. On
uniformish it confidently predicts the wrong direction. The SAT proof plan
uses `6x4T + 55x8T + 173x16T`, while the hardware controls favor homogeneous
8T temporal orders; this points to an underestimated penalty for concurrent
wide teams rather than an LLC-domain assignment error.

Placement-aware full-search time rises from the earlier roughly 29--51 seconds
to 53--84 seconds, about 1.7--1.9x. This overhead should be optimized only after
the ranking model passes.

## Validation

Synthetic tests cover:

- same-domain contention slower than symmetric split-domain placement;
- symmetric domain swaps producing equal scores;
- rejection of unordered overlapping CPU teams;
- physical CPU mapping through `IntervalPlanner`;
- exact rank-global fallback without topology.

Local placement/model suites pass 163 tests; Arm placement/model focused suites
pass 97 tests.

## Next step

Keep the placement-aware data path. Add a separately identifiable model term
for wide-team concurrent pressure, calibrated on fixed-placement width/cohort
controls. Rerun the same strict proof plans before enabling placement-aware
selection or moving to the dynamic-tail second stage.
