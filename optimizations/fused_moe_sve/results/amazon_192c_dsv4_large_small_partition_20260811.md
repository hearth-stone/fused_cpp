# DSV4 large/small route core partition

Date: 2026-08-11

## Scope

This is a benchmark-only scheduling experiment. It does not change the native
kernel, Plan V2 ABI, stage-window policy, merge path, or production planner.

The candidate separates active experts by one route threshold and runs both
classes from the beginning of the call on disjoint NUMA-local core regions:

- large class: multi-thread teams for compute and packed-B reuse;
- small class: many narrow teams for independent cold-B streams;
- independent isolated-cost LPT lane assignment inside each region;
- no cross-region stealing, active delay, route slicing, or stage barrier.

The benchmark CLI is:

```text
--large-small-partition M:LCORES:LT:ST
```

where routes greater than `M` use `LCORES` cores and `LT` threads per expert;
the remaining cores run routes at or below `M` using `ST` threads per expert.

## System

- Host: `AmazonC5192Cores`
- CPU set: NUMA0, logical CPUs `0-95`
- Page policy: 32 MiB HugeTLB packed weights
- Shape: TP4, `H=4096`, `F=512`, `E=256`
- Workload: `dsv4-real-2048-seq70`, 2048 tokens, TopK=6
- Active experts/routes: 223 / 12288
- Route output: BF16
- Runtime extension SHA256:
  `55aeb0d7760f7511faf0a49dff47a46034ca95e4964489f26edfa5abd6c2ef00`
- Profile extension SHA256:
  `903fa2e01941f5cfd2907ab270ef5edad4886c0a509c3a857516ab291e0bfc9f`

The profile binary identity is stale. All performance claims below are direct
same-process operator comparisons on the current extension. The stale profile
is used only for LPT ordering and is not accepted as absolute-time evidence.

At threshold 48:

| class | experts | routes | useful FLOP |
| --- | ---: | ---: | ---: |
| `M > 48` | 28 | 8875 | 111.673 TFLOP |
| `1 <= M <= 48` | 195 | 3413 | 42.945 TFLOP |

Every comparator produced bit-identical output to production strict
(`atol=0`, `rtol=0`). Local and remote workload tests both passed 8/8.

## Core split screen

This screen fixed threshold 48, large width 8T, and small width 4T. Results are
medians of 9 interleaved rounds after two warmups.

| large/small cores | time | gain vs strict |
| --- | ---: | ---: |
| 32 / 64 | 15.565 ms | -14.44% |
| 40 / 56 | 13.199 ms | +0.90% |
| 48 / 48 | 11.765 ms | +13.19% |
| 56 / 40 | 11.885 ms | +12.06% |
| 64 / 32 | 12.403 ms | +7.37% |
| 72 / 24 | 13.752 ms | -3.14% |
| 80 / 16 | 17.787 ms | -25.11% |

The useful region is narrow. Below 48 large cores, the large class is the
tail; above roughly 56 large cores, the small class becomes the tail.

## Width and threshold screen

With a 48/48 core split:

| configuration | time | gain vs strict |
| --- | ---: | ---: |
| threshold 48, large 8T, small 1T | 11.546 ms | +15.34% |
| threshold 48, large 8T, small 2T | 11.634 ms | +14.47% |
| threshold 48, large 8T, small 4T | 11.765 ms | +13.19% |
| threshold 48, large 8T, small 8T | 11.978 ms | +11.19% |
| threshold 12, large 8T, small 4T | 15.989 ms | -16.71% |
| threshold 28, large 8T, small 4T | 11.717 ms | +13.66% |
| threshold 96, large 8T, small 4T | 11.740 ms | +13.44% |
| threshold 192, large 8T, small 4T | 11.762 ms | +13.23% |

Thresholds 28 through 192 are close for this histogram. Threshold 12 fails
because 118 experts with routes 13 through 28 move into only six large lanes.

A 51-round interleaved comparison resolved the large-team choice:

| variant | median | P10 | P90 | aggregate |
| --- | ---: | ---: | ---: | ---: |
| production strict `12x8T` | 13.367 ms | 13.328 | 13.499 | 11.567 TFLOP/s |
| 48C `6x8T` + 48C `48x1T` | 11.625 ms | 11.528 | 11.778 | 13.301 TFLOP/s |
| 48C `3x16T` + 48C `48x1T` | 11.650 ms | 11.598 | 11.767 | 13.272 TFLOP/s |

The 8T and 16T large groups differ by 0.21%, below the approximately 1% noise
floor. The representative candidate remains 8T because it preserves twice as
many large lanes. An independent 31-round run measured 13.345 versus 11.551 ms,
or +15.53% throughput.

## Actual timeline

The traced candidate measured 11.593 ms untraced and 11.515 ms traced. Expert
compute began at approximately 0.269 ms.

| region | actual compute end | region span | core-time |
| --- | ---: | ---: | ---: |
| large cores 0-47 | 11.087 ms | 10.817 ms | 481.79 core-ms |
| small cores 48-95 | 11.301 ms | 11.032 ms | 462.76 core-ms |

The class completion skew is 0.214 ms. Compared with strict, traced stage
core-time changed as follows:

| stage | strict | large/small | delta |
| --- | ---: | ---: | ---: |
| gather/pack A | 18.85 | 18.95 | +0.5% |
| fused W13 | 744.95 | 630.55 | -15.4% |
| direct-route W2 | 361.04 | 295.06 | -18.3% |

The gain is therefore in expert GEMM service, not gather removal or a shorter
merge. Internal idle fell from 73.56 to 22.27 core-ms. Total idle did not fall,
so idle elimination alone cannot explain the wall-time gain.

## PMU evidence

Process-level `perf stat` used a final 300-call tight loop of only the selected
variant. Startup and correctness calls remain in the count, but repeated expert
traffic dominates the measurement. Candidate deltas relative to strict were:

| event | total delta |
| --- | ---: |
| cycles | -13.92% |
| instructions | -2.32% |
| L2D cache refill | -6.61% |
| LL cache read | +6.64% |
| memory-stall cycles | -29.07% |

In a second counter group, `mem_access/cycle`, `L2D refill/cycle`, and
`bus_access/cycle` increased by 19.17%, 9.05%, and 1.72%. The candidate issues
and services lower-cache work more densely while requiring fewer total refill
transactions and substantially fewer memory-stall cycles. This supports
resource complementarity, not a claim that the useful algorithmic DRAM bytes
increased.

## Bounded M<=12 streaming lanes

The original two-region trace schedules the `M<=12` experts late inside the
small region because each small lane preserves descending isolated-cost LPT
order. In this workload the 75 short experts comprise 71 `M=1`, three `M=6`,
and one `M=10` expert. Their first W13 starts at 5.177 ms, their median W13
start is 7.847 ms, and peak short-expert concurrency is 26.

A three-region benchmark-only extension tested whether a bounded number of
short streams should instead remain active from call start:

```text
--large-medium-stream-partition LM:SM:LCORES:LT:SCORES
```

For `48:12:48:8:q`, cores 0--47 retain the six 8T large lanes, the next
`48-q` cores run `13<=M<=48` experts on 1T lanes, and the final `q` cores run
only `M<=12` experts on persistent 1T lanes. All three fixed regions are
runnable at call start. Kernel, windows, merge policy, weights, and aggregate
shape remain identical to the two-region comparator.

The isolated short service sum is 27.504 core-ms, which suggests only three
lanes for an 11 ms target if cross-class dilation is ignored. Direct 31-round
screening rejected that estimate:

| short stream lanes | medium lanes | median |
| ---: | ---: | ---: |
| 4 | 44 | 14.295 ms |
| 6 | 42 | 13.001 ms |
| 8 | 40 | 12.158 ms |
| 12 | 36 | 11.971 ms |
| 16 | 32 | 11.851 ms |
| original shared 48-lane region | n/a | 11.878 ms |

A 51-round refinement measured the original two-region plan at 11.595 ms and
the 12/13/14/15/16-stream plans at 11.891/11.900/11.850/11.881/11.789 ms.
The best fixed stream plan was therefore still 1.67% slower. Wider 20/24-lane
plans measured 12.676/12.335 ms in a separate 51-round comparison.

The 16-stream trace confirms that the intended temporal shaping occurred:
short GEMM starts moved from 5.177 to 0.277 ms and peak concurrency fell from
26 to exactly 16. It did not improve short-expert service. Short W13+W2
core-time changed from 133.55 to 136.49 core-ms, while medium W13 increased
from 212.06 to 223.92 core-ms (+5.6%). The short, large, and medium regions
finished at 9.535, 11.104, and 11.588 ms, respectively, leaving the static
short region idle for roughly 2 ms.

This rejects a static three-region production policy. Limiting cold streams
must not be scored from compulsory weight bytes or isolated service alone: in
this complete allocation, advancing short work did not reduce its core-time
and coincided with slower medium W13. A future experiment may allow a bounded
short cohort to release its cores non-blockingly to pending same-width medium
work, but it must retain the original two-region plan as the baseline and pass
the 2% gate.

A dependency-only control kept the exact 16-stream expert membership, cores,
widths, and windows, but made each short root depend on every medium-lane
terminal task. Medium then ran without short streams, while large overlap
remained unchanged. Relative to eager 16-stream execution, medium W13/W2
core-time fell from
223.92/109.01 to 148.88/80.23 core-ms (-33.5%/-26.4%); large W13 fell from
322.08 to 299.24 core-ms (-7.1%). Short W13+W2 fell from 136.49 to 57.30
core-ms (-58.0%) because it ran mostly after medium completion. Medium ended
at 8.237 ms instead of 11.588 ms. The delayed plan still took 12.340 ms versus
11.784 ms eager because removing overlap serialized the tail. This control
proves that task service time depends strongly on the active mix, but it does
not isolate a root cause: delaying short work also changes active-core count,
wave structure, and the serialized tail.

A stricter paired-order control then fixed all of those first-order quantities.
It used the same 48 `M=28` experts, 48 `M=1` experts, 48 physical cores, 1T
width, weights, and per-core task count. The grouped arms placed `M->S` or
`S->M` on every lane; the crossed arms placed opposite orders on the two core
halves or on alternating cores. A 41-round no-background run on cores 48--95
measured 7.1902/7.1724/7.2368/7.2291 ms, respectively.

An initial control put long experts into the same DAG and compared full-DAG
wall time. That metric was discarded because a long-expert critical path can
hide the foreground difference. The corrected control ran the same DSV4 long
chains continuously in a separate process on a disjoint core prefix and timed
only the paired M/S foreground on cores 48--95:

| unmeasured long-background cores | all M->S | all S->M | crossed halves | crossed adjacent cores |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 7.1902 ms | 7.1724 ms | 7.2368 ms | 7.2291 ms |
| 16 | 8.6269 ms | 8.7477 ms | 8.7028 ms | 8.6770 ms |
| 32 | 9.8333 ms | 9.8894 ms | 9.8020 ms | 9.7385 ms |
| 48 | 10.6404 ms | 10.6613 ms | 10.6034 ms | 10.6871 ms |

The background is real interference: grouped foreground latency rises by
20.0%, 36.8%, and 48.0%. Uniform crossing does not mitigate it monotonically.
At 16 cores it is slower, at 48 cores neutral, and only the 32-core point shows
a weak roughly 1% improvement. Two extra 32-core paired runs gave adjacent-core
median speedups of 0.31% and 1.39%, but both P10--P90 intervals crossed zero by
roughly 6 percentage points because foreground calls sample different phases
of the continuous background.

Therefore `M+M -> S+S` versus `M+S -> S+M` is not a measured source of the
three-region regression. Hardware contention is part of the schedule-dependent
cost, not a cause separable from scheduling. The only supported conclusions
are that service rates are state-dependent and that the complete fixed
three-region allocation is slower. Uniform M/S mixing is not justified as a
planner constraint; at most it is a tie-break after core allocation and
critical-path balance. The residual must be decomposed with core allocation,
release, and tail structure held independently fixed.

**Recorded decision:** do not require temporal uniformity between medium
(`13<=M<=48`) and short (`M<=12`) experts. Planner ordering follows predicted
critical-path and lane-load balance. Uniform mixing is allowed only as a
zero-cost deterministic tie-break and must not reserve a static core region or
override a faster grouped order.

## Model boundary

The current isolated LPT estimate is not a valid scorer for this candidate:

| quantity | predicted | actual |
| --- | ---: | ---: |
| large region | 8.812 ms | 10.817 ms |
| small region | 3.760 ms | 11.032 ms |
| whole event simulation | 14.932 ms | 11.593 ms |

The small region has 2.93x isolated dilation, while the current global event
simulation overpredicts final wall time by 28.81%. Production integration must
model phase-local compute, private-L2 refill, LLC, and DRAM offered demand for
both classes. A pairwise fitted slowdown table is not required, but isolated
lane sums are insufficient.

## Decision

Keep `--large-small-partition` and `--large-medium-stream-partition` as
experimental Plan V2 comparators. The static bounded-stream variant is a
negative result and must not enter production search. Do not add the original
two-region candidate to production search or the plan cache yet. Adoption
requires:

1. cross-class absolute-time error at or below 3%;
2. selected-plan measured regret at or below 2%;
3. no held-out workload regression above 2%;
4. validation on additional DSV4 routing seeds, catalog mixed distributions,
   and a second ARM machine.
