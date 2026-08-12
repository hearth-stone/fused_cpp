# Amazon 192C Static Tail Repartition

Date: 2026-07-30

## Scope

- Host: Amazon 192-core ARM, NUMA0 CPUs `0-95`
- Kernel: fused SVE, split-W13, exact-M JIT, direct BF16 route store
- Shape: TP4-style `H=4096`, `F=512`, `E=256`
- Workload: `moe256-active-set-8`, 2048 tokens, TopK=6, eight experts
  with 1536 routes each
- Baseline plan: six fixed 16-thread lanes, with two experts in the second
  wave
- Experiment: keep the first wave at `6x16T`, then repartition its two
  remaining experts to `2x24T`, `2x32T`, or `2x48T`

The runtime extension hash differs from the older calibration profile. All
comparisons below use measured E2E time; the profile prediction is not used
for the new 24T and 48T widths.

## Placement

The first-wave tasks occupy:

```text
expert 0: cores  0-15
expert 1: cores 16-31
expert 2: cores 32-47
expert 3: cores 48-63
expert 4: cores 64-79
expert 5: cores 80-95
```

The tail cohorts are aligned to the two 48-core halves:

| Tail width | Expert 6 | Expert 7 | Dependencies |
| ---: | --- | --- | --- |
| 24T | cores `0-23` | cores `48-71` | `{0,1}` and `{3,4}` |
| 32T | cores `0-31` | cores `48-79` | `{0,1}` and `{3,4}` |
| 48T | cores `0-47` | cores `48-95` | `{0,1,2}` and `{3,4,5}` |

An initial 32T placement on cores `0-31` and `32-63` measured 10.36 ms. Moving
the second cohort to `48-79` reduced it to 10.00-10.15 ms, so core placement
must be held constant when comparing widths.

## E2E Results

Each pooled row combines two independent 51-run measurements. The second run
used reverse variant order (`48T,32T,24T`) to check ordering bias.

| Plan | Samples | Median | P10 | P90 | Aggregate | Speedup vs strict | Latency reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Strict `6x16T` | 102 | 11.143 ms | - | - | 13.88 TFLOP/s | baseline | baseline |
| Tail `2x24T` | 102 | **9.809 ms** | 9.682 ms | 9.978 ms | **15.76 TFLOP/s** | **+13.60%** | **11.98%** |
| Tail `2x32T` | 102 | 10.057 ms | 9.739 ms | 10.633 ms | 15.37 TFLOP/s | +10.78% | 9.73% |
| Tail `2x48T` | 102 | 9.902 ms | 9.810 ms | 10.083 ms | 15.62 TFLOP/s | +12.54% | 11.14% |

Individual 51-run medians:

| Run | Strict | 24T | 32T aligned | 48T |
| --- | ---: | ---: | ---: | ---: |
| Forward | 11.154 ms | 9.778 ms | 10.151 ms | 9.896 ms |
| Reverse | 11.141 ms | 9.841 ms | 10.004 ms | 9.909 ms |

24T is consistently the best measured width, although it is only about 0.9%
faster than 48T. 32T has the widest run-to-run and P10/P90 spread.

## Native Trace

The table uses the full per-team envelope from the first stage start to the
last worker end. Values show the two tail experts separately.

| Plan | Tail start | Gather | W13 | W2 | Tail finish |
| --- | --- | --- | --- | --- | --- |
| Strict 16T | 5.825 / 6.051 | 0.122 / 0.121 | 3.320 / 3.384 | 1.370 / 1.351 | 10.686 / 10.929 |
| 24T | 5.917 / 6.047 | 0.115 / 0.118 | **2.358 / 2.400** | 0.946 / 0.930 | 9.348 / 9.581 |
| 32T aligned | 5.923 / 5.939 | 0.098 / 0.085 | 2.764 / 2.332 | 0.724 / 0.739 | 9.574 / 9.147 |
| 48T | 5.981 / 6.010 | 0.070 / 0.081 | 2.865 / 2.850 | **0.532 / 0.539** | 9.512 / 9.505 |

Increasing the cohort width continues to improve W2. W13 does not improve
past 24T in the traced E2E run, so 48T's faster W2 only nearly recovers its
W13 loss. The result argues for stage-aware width selection rather than
blindly assigning every idle core to each tail expert.

## Decision

Do not implement an unrestricted dynamic thread pool yet.

The experiment proves that one task-boundary repartition has substantial
headroom: the best static plan improves throughput by 13.6%. It does not show
that arbitrary runtime work stealing is required. A planner-directed,
single-repartition tail policy can capture this case without queueing,
preemption, cross-NUMA acquisition, or a continuously changing team.

The next implementation should therefore be a bounded moldable tail:

1. Keep the planner's fixed teams for the main wave.
2. At the terminal task boundary, allow only planner-declared cohorts.
3. Calibrate isolated 24T and 48T expert costs and score stage-aware candidates
   from `{16,24,32,48}`.
4. Acquire only an immediately available, NUMA-local cohort; otherwise run on
   the original team without waiting.
5. Re-evaluate a general dynamic pool only if irregular multi-wave workloads
   leave material regret after this policy.

## Cost-model calibration

A fresh 101-run measurement with the current runtime extension
`54a8824324af7e2c017ce4c6d00feade3bc4854a63f2e0db34413dd765ff5b00`
confirmed the aligned ordering:

| Plan | Median | P10 | P90 |
| --- | ---: | ---: | ---: |
| Strict `6x16T` | 11.137 ms | 11.011 ms | 11.469 ms |
| Tail `2x24T` | **9.797 ms** | 9.684 ms | 9.910 ms |
| Tail `2x32T` | 9.997 ms | 9.769 ms | 10.667 ms |
| Tail `2x48T` | 9.866 ms | 9.769 ms | 9.994 ms |

The ordinary contention profiler did not reproduce this ranking: at
`M=1536`, isolated 24/32/48T measured 3.424/3.261/3.746 ms and ordinary
two-lane contention measured 4.508/3.918/3.784 ms. Those shapes use contiguous
logical lanes, whereas the bounded tail uses starts 0 and 48. A width-only
shape signature therefore cannot identify the tail placement.

The production profile now carries an exact-layout bounded-tail anchor for
this route, root shape, placement, and extension. It does not alter `T_iso` and
does not interpolate to other routes. Python and native planners use the
anchor only on an exact signature match and otherwise retain the stage-aware
DAG model.

The calibrated production path was then rerun for 51 iterations. It selected
24T and measured 9.842 ms median (9.719--9.992 ms P10--P90), versus 11.103 ms
for strict, a 12.81% gain. The 9.797 ms anchor differs from that independent
production median by 0.045 ms, or 0.46%. An unseen `M=1548` route does not
match the anchor and continues through the simulator.

## Tail M split

The remaining idle cores can be used without widening either whole expert.
Each `M=1536` terminal expert is split into two contiguous `M=768` route
slices. The resulting four fixed 24T tasks use:

```text
expert 6, routes    0-767: cores  0-23
expert 6, routes  768-1535: cores 24-47
expert 7, routes    0-767: cores 48-71
expert 7, routes  768-1535: cores 72-95
```

Each slice independently performs gather/pack, W13, W2, and direct route
store. The route rows are disjoint, so no intermediate or output write is
shared. The runtime publishes an expert as complete only after both slices
finish; ready-token merge therefore retains the original TopK dependency.
This is a strict preplanned DAG, not migration or runtime work stealing.

A 51-run explicit experiment gave:

| Plan | Median | P10 | P90 | Aggregate |
| --- | ---: | ---: | ---: | ---: |
| Strict `2x16T` whole expert | 11.122 ms | 10.948 ms | 11.382 ms | 13.902 TFLOP/s |
| `2x24T` whole expert | 9.804 ms | - | - | 15.77 TFLOP/s |
| Grouped `4x24T`, `M=768` | **8.710 ms** | 8.647 ms | 8.830 ms | **17.751 TFLOP/s** |
| Interleaved `4x24T`, `M=768` | 8.755 ms | - | - | 17.66 TFLOP/s |

Grouped M split improves throughput by 12.56% over the whole-expert 24T tail
and by 27.68% over strict. A bitwise equality check ran before timing.
Grouped order is 0.51% faster than interleaving the two experts, so the cost
profile binds both route-slice count and physical task order.

The production planner exposes this only through an exact-layout anchor:
`root=6x16T`, `route=1536`, `tail_width=24T`, `route_slices=2`, grouped
placement, and the matching kernel/stage-window identity. It does not infer an
M-split candidate for unmeasured routes or machines.

After integration, an independent 51-run production benchmark selected
`tail_repartition_width=24` and `tail_repartition_route_slices=2`:

| Production plan | Median | P10 | P90 | Aggregate | Gain vs strict |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict | 11.180 ms | 11.044 ms | 11.401 ms | 13.830 TFLOP/s | baseline |
| Auto M split | **8.753 ms** | 8.684 ms | 8.919 ms | **17.665 TFLOP/s** | **+27.73%** |
| Explicit grouped M split | 8.746 ms | 8.669 ms | 8.898 ms | 17.678 TFLOP/s | +27.83% |

Auto and the explicit grouped plan differ by only 0.08%. The 8.710 ms anchor
differs from the production-auto median by 0.49%.

## Reproduction

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_vllm_staged_schedule.py \
  --preset moe256-active-set-8 \
  --production-profile <amazon-192c-tp4-profile.json> \
  --static-tail-widths 24 --static-tail-m-split \
  --route-dtype bf16 --warmup 7 --runs 51
```

Raw records:

- `static_tail_repartition_active8_51runs_timing.json`
- `static_tail_repartition_active8_32aligned_51runs.json`
- `static_tail_repartition_active8_reverse_51runs.json`
- `tmp/moe_timeline/moe256-active-set-8/active8_static_tail_{24,32aligned,48}_timeline.json`
- `tmp/moe_timeline/moe256-active-set-8/active8_static_tail_{24,32aligned,48}_timeline.trace`
- `static_tail_calibration_active8_101runs_20260730.json`
- `active8_calibrated_auto_51runs_20260730.json`
- `active8_tail_msplit_51runs_20260730.json`
- `active8_tail_msplit_production_51runs_20260730.json`
