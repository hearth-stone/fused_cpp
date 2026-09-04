# Arm 80C stream-pressure count LOCO comparison

Date: 2026-09-04

## Technical summary

Neither DDRC queue latency nor victim LLC read-miss ratio is accurate enough as
a standalone quantitative slowdown model. Victim LLC miss ratio has the lower
leave-one-count-out (LOCO) mean absolute error, 0.0364 ms versus 0.0453 ms for
DDRC queue latency, but it is not a physically admissible monotone pressure
feature: its count-1 value is below isolated and its anchored prediction is
negative. DDRC queue latency keeps a nonnegative pressure direction and is much
better on two independent 4x1T layout checks, so it is the preferred follow-up
feature, not an accepted model.

No calibration was frozen, no existing holdout was read, and frozen v8 remains
unchanged.

> Superseded measurement note: the process-per-cell baseline limitation was
> removed later on 2026-09-04 with direct per-cell `perf_event_open` reads in one
> randomized long-lived process. That result strengthens DDRC queue latency as
> the preferred feature and is the current source of truth:
> `arm_codex_80c_stream_pressure_paired_pmu_20260904.md`.

## Count sweep

The main session fixes 16 peer threads and changes the number of distinct
transfer-bound packed-B blocks through 1x16T, 2x8T, 4x4T, 8x2T, and 16x1T.
Each cell has 5 warmups, 31 measured calls, four rotating measured copies, a
fifth disjoint 18-expert scrub copy, and simultaneous CPU304/L3C/DDRC counters.

| Distinct B count | Cell | Victim span ms | Delta from isolated ms | DDR queue latency cycles | Queue pressure cycles | Victim LLC miss ratio | LLC pressure |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | isolated | 0.5690 | 0.0000 | 36.46 | 0.00 | 0.3656 | 0.0000 |
| 1 | 1x16T | 0.5410 | -0.0280 | 45.36 | 8.89 | 0.2679 | -0.0977 |
| 2 | 2x8T | 0.5811 | +0.0122 | 49.96 | 13.50 | 0.3705 | +0.0049 |
| 4 | 4x4T | 0.6000 | +0.0310 | 57.38 | 20.92 | 0.3853 | +0.0197 |
| 8 | 8x2T | 0.6787 | +0.1097 | 65.02 | 28.56 | 0.4883 | +0.1227 |
| 16 | 16x1T | 0.6774 | +0.1085 | 57.81 | 21.35 | 0.4331 | +0.0675 |

All event running percentages are 100%. Native overlap medians are exactly
0/1/2/4/8/16. The count-1 negative median is not a stable speedup claim: its
unpaired P10--P90 interval, 0.521--0.564 ms, overlaps the isolated interval,
0.555--0.581 ms. It is retained in every primary LOCO fold rather than removed
after seeing the result.

Both PMU features rise through count 8 and fall at count 16 while the victim
span is flat at counts 8 and 16. That is qualitatively consistent with a
saturating pressure source, but it is not sufficient for quantitative accuracy.

## Anchored leave-one-count-out comparison

The primary model is predeclared as an isolated-relative, nonnegative,
through-origin single-feature fit:

```text
delta_target_ms = max(0, theta) * (feature_cell - feature_isolated)
```

Each positive count is held out once. The slope is fitted on the other four
positive counts; isolated remains the zero anchor.

| Held-out count | Measured delta ms | Queue prediction ms | Queue error ms | LLC prediction ms | LLC error ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -0.0280 | +0.0295 | +0.0574 | -0.1047 | -0.0767 |
| 2 | +0.0122 | +0.0441 | +0.0320 | +0.0040 | -0.0082 |
| 4 | +0.0310 | +0.0732 | +0.0422 | +0.0159 | -0.0151 |
| 8 | +0.1097 | +0.0712 | -0.0384 | +0.0906 | -0.0191 |
| 16 | +0.1085 | +0.0521 | -0.0564 | +0.0455 | -0.0630 |

| Metric | DDRC queue latency | Victim LLC miss ratio |
| --- | ---: | ---: |
| LOCO MAE | 0.0453 ms | 0.0364 ms |
| LOCO RMSE | 0.0464 ms | 0.0459 ms |
| Maximum absolute error | 0.0574 ms | 0.0767 ms |
| Nonnegative pressure at every count | yes | no |
| Nonpositive anchored predictions | 0 | 1 |

The LLC feature wins only the average-error column. It loses the maximum-error
and physical-direction checks. The queue feature underpredicts both count 8 and
count 16, so it also cannot be frozen.

## Free-intercept sensitivity does not rescue either feature

As a sensitivity check, each fold also fits `span_ms = intercept + slope *
feature` with a nonnegative slope. This removes dependence on an exact isolated
zero anchor but adds a fitted intercept.

| Metric | DDRC queue latency | Victim LLC miss ratio |
| --- | ---: | ---: |
| Affine LOCO MAE | 0.0379 ms | 0.0274 ms |
| Affine LOCO RMSE | 0.0405 ms | 0.0333 ms |
| Affine maximum absolute error | 0.0559 ms | 0.0530 ms |

The numerical errors improve slightly, but they remain 25--35% of the maximum
observed 0.1097 ms count delta. The affine result also does not repair the lack
of a structural predictor for LLC miss ratio. This sensitivity is diagnostic,
not an alternative calibration candidate.

## Independent 4x1T layout checks favor queue latency

The count model is fitted only on the fixed-16-thread ladder. A separate 4x1T
cell changes team layout while retaining four distinct packed-B blocks. The
second row comes from an independent process/session with its own isolated
baseline and seed.

| Layout session | Measured delta ms | Queue prediction ms | Queue error ms | LLC prediction ms | LLC error ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| main-session 4x1T | +0.0612 | +0.0338 | -0.0274 | +0.1170 | +0.0558 |
| independent 4x1T repeat | +0.0293 | +0.0284 | -0.0010 | +0.0542 | +0.0248 |

Queue latency generalizes substantially better across the layout change. The
4x1T measured delta itself moves by 0.0319 ms across sessions, confirming that
sub-0.03 ms point accuracy is not identifiable with the current process-level
baseline protocol.

## Decision

- Do not adopt either single-feature model.
- Retain DDRC queue latency as the preferred physical feature for the next
  measurement-design iteration because it stays nonnegative and transfers
  better to 4x1T.
- Retain victim LLC miss ratio as an explanatory/validation metric, not a
  standalone planner feature. The planner cannot currently predict this PMU
  ratio from an executable state, and the ratio is nonmonotone at count 1.
- Do not combine the two features yet. With five positive count points, a
  two-parameter fit would be weakly identified and would hide baseline drift.

## Remaining measurement issue

Each mode still runs in a separate initialized process because perf produces
one cumulative count set per process. The isolated absolute span changed from
0.477 ms in the previous joint session to 0.569 ms here, while count-16 changed
only from 0.704 to 0.677 ms. This makes isolated-relative deltas less stable
than the absolute loaded cells.

Before another formula attempt, run candidate and isolated cells from one
long-lived allocation/process, using per-cell counter reset/read or an attached
perf controller. That paired protocol, rather than another regression form, is
the next identifiability improvement.

## Artifacts and reproducibility

Main count session: `tmp/moe_stream_pressure_pmu_count_s1`, seed `20260918`.
Its `SHA256SUMS` file has SHA256
`2b707937680f69b06b3cc330b6ba620a4f1065d2b33689eba6efe32a379ef51d`.

Independent 4x1T repeat: `tmp/moe_stream_pressure_pmu_4x1_repeat`, seed
`20260919`. Its `SHA256SUMS` file has SHA256
`82e611c1ca7fd00432d612e42a7985d08237e07931dde46790791bc4bb5d18b7`.

LOCO output: `tmp/moe_stream_pressure_pmu_count_s1/loco.json`, SHA256
`9ba1834dcc47724fbed379016ed5b6a52aa79a7fde7997e839541d6dd8c6147b`.

Probe SHA256:
`a9fe07933aa162c51036d0c1c0cbc42cfe3079ee93e24ef58d8532555e823f7a`.
Analyzer SHA256:
`99813c8bfaac63ec43161d4c47e44b060eb02d9e6a545f97881167c67bbf906b`.

The machine, affinity, PMU event set, nested FIFO command pattern, frozen
calibration, extension SHA, shape, and cold-weight protocol are unchanged from
`arm_codex_80c_stream_pressure_pmu_20260904.md`. No existing route or real-trace
holdout was opened.
