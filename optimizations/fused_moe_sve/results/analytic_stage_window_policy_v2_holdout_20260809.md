# Analytical Stage-Window Policy V2 Holdout

## Scope

Policy v2 corrects the two stable policy-v1 errors without adding a route table
or a planner search variable:

- packed A is classified by a physical resident/streaming condition, with one
  M panel reserved for the kernel's in-flight load and prefetch state;
- the calibrated packed-B repeated-scan miss applies only until one effective
  L2 of distinct A panels has passed, after which B is physically resident or
  streaming;
- candidate objective differences below calibrated transfer uncertainty are
  treated as equivalent, then resolved toward the kernel-native two-tile/L1D
  owner window.

The empirical AmazonC5192Cores V4 production policy is unchanged. Policy v2 is
generated only by the analytical backend after the planner has selected the
existing task team width.

## Method

The workload, legal candidate set, coordinate-oracle construction, and
interleaved sampling protocol are unchanged from
`analytic_stage_window_policy_holdout_20260809.md`:

| Host | Grid | Full-call sampling |
| :--- | :--- | :--- |
| AmazonC5192Cores NUMA0 `0-95` | M=`28,72,120,216,320,768,2040`; T=`1,2,4,8`; 96 distinct experts | 2 warmups + 11 shuffled rounds; 32 MiB HugeTLB |
| AmazonECSV1 `0-7` | same M/T grid; 32 distinct experts | 2 warmups + 15 shuffled rounds; THP |

The coordinate oracle includes the analytical and inherited points, both full
one-dimensional axes, and a local 3x3 cross. It is not a full W13 x W2
Cartesian search, so regret remains a lower bound against all legal window
pairs. Primary statistics use candidate medians. The p10 analysis is a
one-sided noise sensitivity check, not a replacement performance estimator.

## Results

| Host / statistic | Median regret | P90 | Maximum | <=2% | <=5% | Median gain vs inherited | Median rank rho | Median p90/p10 spread |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 192C NUMA0, median | 1.50% | 2.98% | 3.38% | 17/28 | 28/28 | 56.13% | 0.903 | 1.46% |
| 192C NUMA0, p10 | 1.64% | 2.87% | 3.46% | 16/28 | 28/28 | 56.19% | 0.903 | - |
| 8C, raw median | 2.88% | 6.05% | 18.41% | 10/28 | 22/28 | 3.04% | 0.595 | 28.47% |
| 8C, p10 sensitivity | 2.01% | 2.61% | 4.21% | 14/28 | 28/28 | 4.58% | 0.677 | - |

On the stable 192-core host, policy v1's median/P90/maximum regret of
`1.63/6.57/11.32%` becomes `1.50/2.98/3.38%`. All 28 points now pass the 5%
gate. Median candidate rank correlation rises from 0.853 to 0.903, and 27/28
groups have positive correlation.

The largest remaining 192-core misses are:

| M | T | Analytical owner KiB W13/W2 | Coordinate oracle KiB W13/W2 | Regret |
| ---: | ---: | :--- | :--- | ---: |
| 216 | 2 | 128/64 | 64/128 | 3.38% |
| 2040 | 1 | 1024/256 | 1024/1024 | 3.27% |
| 2040 | 2 | 1024/256 | 1024/1024 | 3.10% |
| 72 | 2 | 128/64 | 64/128 | 2.98% |
| 768 | 8 | 512/64 | 512/256 | 2.92% |

These residuals are dominated by W2 or two-stage interaction. The former
stable failures are closed: M=216 no longer forces the large W13 window, and
M=768/2040 at narrow width now selects the 1 MiB W13 owner window.

## 8-Core Noise

The 8-core host remained subject to one-sided remote-agent preemption. Candidate
p90/p10 spread is `28.47/99.66/329.70%` at median/P90/maximum. For example,
M=2040/T=8 reports 681.64 ms for the analytical median and 575.65 ms for the
median-selected oracle, but the analytical point is only 1.11% above the p10
oracle. The raw 18.41% maximum is therefore not a valid model gate.

The p10 sensitivity maximum is 4.21%, but this host still uses a transferred
packed-B retention prior. A quiet rerun with a local multi-team retention probe
is required before treating it as an independent portability pass.

## Decision

The clean 192-core stage-window gate now passes. Keep policy v2 enabled for the
analytical backend and retain empirical V4 as the production default/fallback
until the remaining full-model contention, distributed lifetime, and 8-core
local-calibration gates pass. No planner candidate, pruning rule, or runtime ABI
changes in this correction.

## Artifacts

- `amazon_192c_numa0_analytic_stage_window_holdout_v2_interleaved_20260809.json`
- `amazon_ecs_v1_8c_analytic_stage_window_holdout_v2_interleaved_20260809.json`
- `../benchmarks/bench_analytic_stage_window_holdout.py`
