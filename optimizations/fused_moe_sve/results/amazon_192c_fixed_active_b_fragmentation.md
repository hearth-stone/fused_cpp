# Fixed-active-B route fragmentation

Date: 2026-07-15

## Scope

This experiment ran on NUMA node 0, CPUs 0-95, of `AmazonC5192Cores`. It used
the production SVE M12 fused W13 and BF16 W2 assembly kernels, with split-W13
fixed at two ranges. The benchmark binary was built by GCC 15.2.0 with
`-O2 -march=armv8.6-a+sve+bf16+i8mm`; its SHA-256 was:

```text
71e21c2d7713a4aedfaa9990c57664e2ec35669942cbd1b86ba2ba5678af2895
```

The controlled baseline has 24 teams, four threads per team, and one H=4096,
F=512, M=2040 expert per team. Replacing one baseline expert creates Q distinct
experts of `M=2040/Q`. The dynamic scheduler starts long routes first; whenever
a four-thread team finishes, it claims the next expert from a shared queue.
This keeps 24 expert slots occupied until fewer than 24 tasks remain. The
following quantities remain fixed:

```text
workers                 = 24 * 4 = 96
total routes            = 24 * 2040 = 48960
total GEMM FLOPs         = 616.059 GFLOP
maximum active experts  = 24
active split-W13/W2 B   = 24 * 4 MiB = 96 MiB
```

Each logical expert has independent packed W13 and W2 weights totaling 12 MiB.
Increasing Q therefore increases total unique weights consumed per invocation,
but not the instantaneous active-B capacity. Each point used two warmups and
seven timed invocations, with nine complete data copies so no invocation reused
the same weight addresses.

The E2E timing covers task dequeue, gather-pack, fused W13/SiLU/multiply/packC,
W2 BF16 GEMM, and task-local team barriers. It excludes the production planner,
scatter, and route-weight accumulation. A fixed-slot mode was also measured as
a no-rebalancing control.

Correctness used a small two-team case and compared the parallel intermediate
and output buffers bit-for-bit with the same tasks executed one at a time by a
single N lane:

```text
intermediate_mismatches=0/3072
output_mismatches=0/3072
```

Command:

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/run_fragmented_route_pipeline.py \
  --output /tmp/split_w13_fixed_active_b_fragmentation_dynamic_20260715.json \
  --teams 24 --base-routes 2040 --hidden 4096 --intermediate 512 \
  --threads-per-team 4 --schedule dynamic --split-factors 1,2,5,10 \
  --replaced-teams 6,12,24 --warmup 2 --runs 7
```

## Results

`Replaced` is the number of the 24 teams whose one long expert was replaced.
`Tasks` counts all distinct experts consumed by one invocation. Active B is
96 MiB in every row.

| Q | Replaced | Short M | Tasks | Unique W13+W2 | Wall | TFLOP/s | vs. baseline |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 2040 | 24 | 288 MiB | 26.580 ms | 23.178 | baseline |
| 2 | 6 | 1020 | 30 | 360 MiB | 26.223 ms | 23.493 | +1.36% |
| 2 | 12 | 1020 | 36 | 432 MiB | 26.443 ms | 23.297 | +0.51% |
| 2 | 24 | 1020 | 48 | 576 MiB | 27.383 ms | 22.498 | -2.93% |
| 5 | 6 | 408 | 48 | 576 MiB | 26.164 ms | 23.546 | +1.59% |
| 5 | 12 | 408 | 72 | 864 MiB | 27.490 ms | 22.410 | -3.31% |
| 5 | 24 | 408 | 120 | 1440 MiB | 29.516 ms | 20.872 | -9.95% |
| 10 | 6 | 204 | 78 | 936 MiB | 27.443 ms | 22.449 | -3.15% |
| 10 | 12 | 204 | 132 | 1584 MiB | 28.502 ms | 21.615 | -6.74% |
| 10 | 24 | 204 | 240 | 2880 MiB | 35.246 ms | 17.479 | -24.59% |

The hypothesis holds well for partial replacement. Replacing 25% of baseline
experts keeps throughput within 3.2% for all tested short routes, including
M=204. Replacing 50% remains within 3.4% for M=1020 and M=408, while M=204 loses
6.7%. Replacing every expert exposes a monotonic fragmentation cost: 2.9%,
10.0%, and 24.6% for Q=2, 5, and 10 respectively.

## Stage attribution

For homogeneous endpoints, maximum team stage sums are directly additive and
show where the full-replacement loss occurs:

| Configuration | Pack | Fused W13 | W2 | E2E |
|---:|---:|---:|---:|---:|
| Q=1 baseline | 2.274 ms | 16.470 ms | 7.937 ms | 26.580 ms |
| Q=2, all replaced | 2.016 ms | 17.205 ms | 8.244 ms | 27.383 ms |
| Q=5, all replaced | 1.442 ms | 18.865 ms | 9.309 ms | 29.516 ms |
| Q=10, all replaced | 1.317 ms | 22.850 ms | 11.585 ms | 35.246 ms |

From baseline to Q=10, W13 adds 6.380 ms and W2 adds 3.648 ms, while pack becomes
0.957 ms faster. The regression is therefore inside GEMM stages rather than
route packing. For partially replaced cases, per-stage maxima can come from
different teams and must not be summed as an E2E decomposition.

Dynamic scheduling mainly helps the imbalanced extreme. At Q=10 with 12
baseline experts replaced, it improves throughput from 19.956 TFLOP/s in
fixed-slot mode to 21.615 TFLOP/s, an 8.3% gain. Most other points differ by
roughly 0-2%, showing that task rebalancing removes tail imbalance but not the
underlying packed-weight turnover cost.

## Interpretation

Active B describes capacity pressure but not weight turnover. The M12 rows
kernel repeatedly consumes each thread's B stripe for every M12 A panel. A
baseline M=2040 expert has 170 panels over which to reuse its packed B. Splitting
it by Q reduces reuse to `170/Q` panels and introduces Q distinct weights and Q
sets of kernel calls/barriers. Consequently:

```text
active B capacity       = active teams * 4 MiB                 (fixed)
unique B per invocation = logical expert tasks * 12 MiB        (grows)
B reuse panels/expert   = M / 12                                (shrinks)
kernel/barrier count    = logical expert tasks                 (grows)
```

Total packed-A elements and GEMM FLOPs remain fixed. The measured boundary means
active B is a useful first-order predictor when short experts are a minority or
M remains at least roughly 400, but it is not a sufficient scalar cost model.
A scheduling model also needs packed-weight turnover, route/M-panel count,
expert-call count, and queue occupancy near the tail. The strongest defensible
claim is:

> At fixed 96 MiB active B, replacing up to 25% of M=2040 experts with as many
> as ten M=204 experts changes fused expert-pipeline throughput by at most 3.2%.

The broader claim that expert count never matters at fixed active B is disproved
by the all-Q=10 point, which loses 24.6% despite identical total FLOPs and active
B capacity.

## Worker-PMU validation

On 2026-07-16, the homogeneous Q=1/2/5/10 endpoints were rerun with perf
attached only to the 96 pinned worker TIDs. The benchmark stopped after all
allocation, weight initialization, and worker-pool creation. Perf was attached
while the process was stopped, so initialization and the main thread were
excluded. Each point used two warmups, seven timed invocations, nine distinct
complete data copies, and three independent process repeats. All counters were
scheduled for 100% of the measurement window. The profiling binary SHA-256 was:

```text
a83570c8f785885223ac3b6efe28d1234a1d7425e4ca256b031fcb6698293047
```

```bash
.venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/profile_fragmented_route_pipeline.py \
  --output /tmp/fixed_active_b_fragmentation_perf_20260716.json \
  --teams 24 --base-routes 2040 --hidden 4096 --intermediate 512 \
  --threads-per-team 4 --schedule dynamic --split-factors 1,2,5,10 \
  --cpu-start 0 --numa-node 0 --warmup 2 --runs 7 --repeats 3
```

Counts below are medians across process repeats and are normalized per complete
invocation, including the same proportion of unique-weight warmups and timed
runs. `L2 refill GiB-eq` is `l2d_cache_refill * 64`; it is a cache-line
equivalent, not a memory-controller byte counter.

| Q | M | Unique B | Wall | TFLOP/s | Instructions | Cycles | L2 refill GiB-eq | IPC | W13 | W2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2040 | 288 MiB | 26.635 ms | 23.130 | 30.515 B | 8.274 B | 4.618 | 3.688 | 16.485 ms | 7.919 ms |
| 2 | 1020 | 576 MiB | 27.374 ms | 22.505 | 30.515 B | 8.438 B | 5.153 | 3.617 | 17.204 ms | 8.191 ms |
| 5 | 408 | 1440 MiB | 29.306 ms | 21.022 | 30.516 B | 8.764 B | 6.789 | 3.482 | 18.801 ms | 9.207 ms |
| 10 | 204 | 2880 MiB | 35.506 ms | 17.351 | 30.517 B | 10.573 B | 10.141 | 2.886 | 23.132 ms | 11.651 ms |

The instruction count changes by only 0.006% from Q=1 to Q=10, while worker
cycles increase 27.8%, L2 refills increase 119.6%, and IPC falls 21.7%. Pack
time falls from 2.271 ms to 1.339 ms while both GEMM stages grow, so neither
extra arithmetic nor pack-A explains the regression.

The host exposes no memory-controller or mesh uncore PMU. A second two-repeat
run therefore used the architected `bus_access` and `mem_access` events as a
cross-check, without interpreting either as exact DRAM bytes:

| M | L2 refill GiB-eq | Bus-access GiB-eq | Mem-access GiB-eq | Backend-memory stall cycles |
|---:|---:|---:|---:|---:|
| 2040 | 4.594 | 14.948 | 499.287 | 0.376 B |
| 204 | 10.143 | 33.986 | 499.317 | 0.524 B |

The number of memory-access instructions is unchanged to 0.006%, but bus
accesses rise 127.4%, L2 refills rise 120.8%, and backend-memory stall cycles
rise 39.1%. Together with the fixed FLOP and instruction counts, this directly
supports the narrower mechanism: short, distinct experts lose packed-B cache
reuse and generate more lower-cache traffic inside W13/W2. The PMU does not
prove an exact DRAM-byte fraction, so the result should be described as an
LLC/memory-to-private-cache traffic bottleneck rather than a measured DRAM-only
bottleneck.
