# Analytical Stage-Window Policy Holdout

> This records policy v1. The physical A-residency/B-turnover correction and
> repeated holdout are in `analytic_stage_window_policy_v2_holdout_20260809.md`.

## Scope

This validation asks whether the analytical SVE MoE model can generate W13 and
W2 stage windows directly, without a route-band timing table or a new planner
search variable. The planner first chooses the existing team shape. For each
task `(M, threads)`, `AnalyticStageWindowPolicy` then enumerates legal
tile-aligned windows and minimizes the model's stage-local incremental ECM
objective.

The empirical AmazonC5192Cores V4 policy remains unchanged. This experiment
only makes analytical models generate their own execution policy.

## Method

The candidate set keeps the inherited geometry and generated points satisfying:

- owner window at least one private L1D;
- enough N tiles to keep every naturally usable team thread active;
- a power-of-two owner tile count;
- at least two W13 ranges and one W2 range.

`M<=12` always inherits because one physical panel has no repeated packed-B
scan to protect. W13 and W2 are scored independently. The measured coordinate
oracle contains the analytical point, the inherited point, every W13-axis
point, every W2-axis point, and a local 3x3 cross around the prediction. It is
not a full Cartesian oracle, so reported regret is a lower bound on regret
against the complete legal space.

Every point uses the same packed weights, expert count, lane DAG, and output
path. Expert weights are independent, lane starts rotate across samples, and
all candidates for one `(M,T)` are shuffled and executed once per round before
the next round. This avoids the time-drift bias of measuring one candidate to
completion before starting the next.

| Host | Runtime geometry | Grid | Full-call sampling |
| :--- | :--- | :--- | :--- |
| AmazonC5192Cores NUMA0 `0-95` | 96 cores, L2=2 MiB/core, SVE N tile=8, 32 MiB HugeTLB | M=`28,72,120,216,320,768,2040`; T=`1,2,4,8`; 96 experts | 2 warmups + 11 shuffled rounds |
| AmazonECSV1 `0-7` | 8 cores, L2=1 MiB/core, SVE N tile=16, THP | same M/T grid; 32 experts | 2 warmups + 15 shuffled rounds |

The 8-core cache and service curves are machine-local. Its packed-B retention
fraction is explicitly marked as a transferred 192-core prior; this run tests
that prior's portability rather than claiming an independent local retention
calibration.

## Results

Primary statistics use per-candidate medians. `Gain` is relative to the
operator-wide inherited split-W13 geometry, not relative to the measured V4
production table.

| Host / statistic | Median regret | P90 regret | Max regret | <=2% | <=5% | Median gain | Median rank rho | Median p90/p10 spread |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 192C NUMA0, median | 1.63% | 6.57% | 11.32% | 14/28 | 21/28 | 53.83% | 0.853 | 1.44% |
| 8C, median | 1.39% | 3.51% | 15.22% | 18/28 | 27/28 | 5.59% | 0.424 | 19.56% |
| 8C, p10 sensitivity | 1.30% | 1.93% | 2.78% | 26/28 | 28/28 | 5.43% | 0.389 | - |

The 192-core host is sufficiently stable to interpret directly. Its worst
misses are:

| M | T | Analytical owner KiB W13/W2 | Coordinate-oracle KiB W13/W2 | Regret |
| ---: | ---: | :--- | :--- | ---: |
| 2040 | 1 | 512/256 | 1024/512 | 11.32% |
| 768 | 1 | 512/64 | 1024/64 | 10.28% |
| 216 | 1 | 512/64 | 128/64 | 7.55% |
| 72 | 8 | 256/128 | 64/128 | 6.57% |
| 216 | 2 | 512/128 | 64/128 | 6.33% |

These differences remain when the 192-core data is recomputed from p10: the
median/P90/max regret is `0.87%/6.48%/11.22%`. They are model errors rather than
sampling noise. The analytical objective captures the broad ordering in 26/28
groups, but misses the M=216 transition and undersizes the W13 owner window for
long narrow-team cases.

The 8-core host has heavy one-sided OS contention from concurrent remote-agent
processes. Candidate p90/p10 spread has a 19.56% median, 101.50% P90, and
665.35% maximum. The raw 15.22% maximum comes from M=2040/T=4: the analytical
point has a 681.50 ms median but a 578.28 ms p10, while the median-selected
oracle has a 591.46 ms median and 576.39 ms p10. The low-tail result therefore
supports the policy's direction, but the raw medians cannot establish a strict
5% production gate on this host.

Against the existing measured AmazonC5192Cores V4 policy, 27 of 28 points were
present in the coordinate set. Analytical minus V4 has a 0.00% median: 9 points
are faster by more than 0.1%, 12 are within 0.1%, and 6 are slower; the range is
`-57.40%` to `+5.67%`. The large improvements are long 1T/2T cases outside the
V4 calibrated bands. This shows that the formula is a useful portable fallback,
but it has not displaced the exact measured policy on its home machine.

## M12 Control

An independent `M=12,T=4` control used 21 shuffled rounds on each host. The
inherited owner windows are 1024/1024 KiB. On 192C, shrinking to the minimum
legal W13/W2 windows changes 3.2409 ms to 3.3742/3.2870 ms. On 8C, it changes
4.4063 ms to 4.5329/4.5638 ms. Intermediate W2 points are non-monotone, so these
data do not justify fitting a global per-range latency. Keeping M<=12 on the
inherited geometry is correct.

## Decision

The analytical backend now generates and binds this policy automatically. The
planner's shape and kernel-variant spaces are unchanged, and empirical models
continue to use the exact-profile V4 policy or their previous fallback.

Do not promote the analytical policy to replace the empirical production
default yet. The clean 192-core holdout fails the existing 5% maximum-regret
gate. The next model work is to represent the M=216 transition, long-route
range/A-scan balance, and residual W13/W2 interaction without adding a route
latency table. The 8-core host additionally needs a machine-local multi-team
packed-B retention probe and a quieter strict rerun.

## Artifacts

- `amazon_192c_numa0_analytic_stage_window_holdout_interleaved_20260809.json`
- `amazon_ecs_v1_8c_analytic_stage_window_holdout_interleaved_20260809.json`
- `amazon_192c_numa0_stage_window_range_calibration_m12_interleaved_20260809.json`
- `amazon_ecs_v1_8c_stage_window_range_calibration_m12_interleaved_20260809.json`
- `../benchmarks/bench_analytic_stage_window_holdout.py`
