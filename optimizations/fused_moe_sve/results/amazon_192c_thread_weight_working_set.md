# Split-W13 thread/weight working sets

Date: 2026-07-15

## Scope

The experiment ran on NUMA node 0, CPUs 0-95, of `AmazonC5192Cores`. The host
has 96 MiB shared L3 per NUMA node and 2 MiB private L2 per core. The binary was
built with GCC 15.2.0 and `-O2 -march=armv8.6-a+sve+bf16+i8mm`. Its SHA-256 was:

```text
41686b32fe78c2dffd77ddfa24b3e8497b1bb58a2df613b5df9afe9e40ca4fd7
```

The direct fused pipeline calls the production M12 SVE assembly kernels. W13
always uses two N ranges (`w13_ranges=2`), followed by the production BF16 W2
kernel. It retains synchronization between pack-A, W13, and W2, but excludes
the async planner, scatter, and route-weight accumulation.

Each point used two warmups and seven timed runs. Nine packed-weight copies made
every warmup and timed invocation use a distinct weight address. The original
thread-mapping grid used routes 192 and 2040. The fixed-work expert-team control
used 2304 total routes split evenly among its active experts. Hidden size was
4096, and all route counts were M12 aligned.

Command:

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/run_thread_weight_working_set.py \
  --output /tmp/split_w13_thread_working_set_20260715.json \
  --threads 1,2,4,8,16,32,64,96 --routes 192,2040 \
  --experiments nsplit,expert-fixed,expert-total \
  --nsplit-stage-mib 4,16,64 --expert-stage-mib 0.5,2 \
  --total-stage-mib 32,64,96 --warmup 2 --runs 7
```

## Working-set definitions

For hidden size H and intermediate size F, split-W13 makes one W13 range the
same size as W2:

```text
W_stage = W13_chunk = W2 = 2 * H * F bytes
W_full_packed_per_expert = W13 + W2 = 3 * W_stage
```

For one expert split over T N lanes:

```text
aggregate active B stage = W_stage
largest per-thread B stripe ~= W_stage / T, rounded to N tiles
```

For T concurrent one-thread experts:

```text
aggregate active B stage = T * W_stage_per_expert
per-thread B stage = W_stage_per_expert
```

These are capacity quantities. Input traffic is separate. Ignoring cache hits,
one expert has `A13 = 2*M*H` input bytes and `A2 = 2*M*F` intermediate bytes.
N-split scans the same A from T lanes; one-expert-per-thread scans T distinct A
matrices. In the fixed-total experiment, `T*F` and B bytes remain approximately
constant, but W13 input bytes grow with T.

## One expert, N-split

The aggregate B stage remains constant as threads increase. The table shows
route 2040; `Stripe` is the largest B stripe assigned to one thread.

| B stage | T | Stripe | Wall | Aggregate TFLOP/s | Linear efficiency |
|---:|---:|---:|---:|---:|---:|
| 4 MiB | 1 | 4 MiB | 82.549 ms | 0.311 | 100.0% |
| 4 MiB | 8 | 0.50 MiB | 10.808 ms | 2.375 | 95.5% |
| 4 MiB | 16 | 0.25 MiB | 5.742 ms | 4.470 | 89.9% |
| 4 MiB | 32 | 0.125 MiB | 4.055 ms | 6.330 | 63.6% |
| 4 MiB | 64 | 0.0625 MiB | 3.880 ms | 6.615 | 33.2% |
| 16 MiB | 1 | 16 MiB | 316.717 ms | 0.324 | 100.0% |
| 16 MiB | 8 | 2 MiB | 44.332 ms | 2.316 | 89.3% |
| 16 MiB | 16 | 1 MiB | 20.725 ms | 4.954 | 95.5% |
| 16 MiB | 32 | 0.50 MiB | 10.647 ms | 9.643 | 93.0% |
| 16 MiB | 64 | 0.25 MiB | 6.591 ms | 15.577 | 75.1% |
| 16 MiB | 96 | 0.1875 MiB | 5.978 ms | 17.176 | 55.2% |
| 64 MiB | 1 | 64 MiB | 1285.794 ms | 0.319 | 100.0% |
| 64 MiB | 8 | 8 MiB | 175.930 ms | 2.334 | 91.4% |
| 64 MiB | 16 | 4 MiB | 98.532 ms | 4.168 | 81.6% |
| 64 MiB | 32 | 2 MiB | 54.101 ms | 7.591 | 74.3% |
| 64 MiB | 64 | 1 MiB | 23.202 ms | 17.702 | 86.6% |
| 64 MiB | 96 | 0.75 MiB | 17.418 ms | 23.580 | 76.9% |

The total B capacity does not grow with T. Increasing T distributes B over
private L2 caches, and crossing from a roughly 2 MiB stripe to 1 MiB produces a
large gain for the 16 and 64 MiB stages. Small 4 MiB experts instead run out of
N tiles: 32 to 64 threads changes the stripe from 128 to 64 KiB but improves
throughput by only 4.5%.

## One expert per thread

With fixed per-expert weight size, total active B grows linearly. `Slowdown` is
concurrent wall time divided by the corresponding one-expert wall time.

| Route | B/expert | T | Total B | Wall | Slowdown | Aggregate TFLOP/s |
|---:|---:|---:|---:|---:|---:|---:|
| 2040 | 0.5 MiB | 1 | 0.5 MiB | 14.318 ms | 1.000x | 0.224 |
| 2040 | 0.5 MiB | 8 | 4 MiB | 15.731 ms | 1.099x | 1.632 |
| 2040 | 0.5 MiB | 16 | 8 MiB | 16.779 ms | 1.172x | 3.060 |
| 2040 | 0.5 MiB | 32 | 16 MiB | 19.122 ms | 1.336x | 5.370 |
| 2040 | 0.5 MiB | 64 | 32 MiB | 23.953 ms | 1.673x | 8.573 |
| 2040 | 0.5 MiB | 96 | 48 MiB | 30.781 ms | 2.150x | 10.007 |
| 2040 | 2 MiB | 1 | 2 MiB | 45.810 ms | 1.000x | 0.280 |
| 2040 | 2 MiB | 8 | 16 MiB | 47.061 ms | 1.027x | 2.182 |
| 2040 | 2 MiB | 16 | 32 MiB | 48.709 ms | 1.063x | 4.216 |
| 2040 | 2 MiB | 32 | 64 MiB | 55.816 ms | 1.218x | 7.358 |
| 2040 | 2 MiB | 64 | 128 MiB | 71.427 ms | 1.559x | 11.500 |
| 2040 | 2 MiB | 96 | 192 MiB | 86.834 ms | 1.896x | 14.189 |

For 2 MiB experts, per-expert wall remains within 6.3% through 16 experts and a
32 MiB aggregate stage. It is 21.8% slower at 64 MiB and 55.9% slower at
128 MiB. This is consistent with increasing shared-cache and bandwidth pressure.

The 0.5 MiB experts slow down earlier despite a smaller B stage. At route 2040,
each expert has 15.94 MiB of W13 input but relatively little GEMM work. Distinct
input streams, pack-A, synchronization, and lower arithmetic intensity dominate
before B reaches the LLC capacity boundary. B bytes alone are therefore not a
sufficient contention predictor for small F.

## Same aggregate B, different mapping

The most direct control fixes aggregate B near 64 MiB and approximately fixes
total GEMM FLOPs. One side N-splits a single F=8192 expert; the other runs T
one-thread experts while reducing F approximately as `8192/T`.

| T | N-split B/thread | N-split TFLOP/s | Expert B/thread | Expert TFLOP/s | N-split / experts |
|---:|---:|---:|---:|---:|---:|
| 1 | 64 MiB | 0.319 | 64 MiB | 0.319 | 1.00x |
| 2 | 32 MiB | 0.635 | 32 MiB | 0.644 | 0.99x |
| 4 | 16 MiB | 1.275 | 16 MiB | 1.287 | 0.99x |
| 8 | 8 MiB | 2.334 | 8 MiB | 2.336 | 1.00x |
| 16 | 4 MiB | 4.168 | 4 MiB | 4.128 | 1.01x |
| 32 | 2 MiB | 7.591 | 2 MiB | 7.308 | 1.04x |
| 64 | 1 MiB | 17.702 | 1 MiB | 12.125 | 1.46x |
| 96 | 0.75 MiB | 23.580 | 0.6875 MiB | 12.953 | 1.82x |

The mappings are nearly identical through 32 threads. At 64 threads, both put
about 1 MiB of B on each core, but N-split reuses one packed input while the
expert mapping streams 64 distinct inputs and pays 64 expert fixed costs. The
gap therefore cannot be represented by aggregate B bytes alone.

For a fixed 32 MiB aggregate B stage, route-2040 expert throughput peaks at
8.72 TFLOP/s with 64 threads and falls to 6.41 TFLOP/s at 96 threads. The
96-thread point has only five W13 N tiles per expert. This independently shows
that per-task N-tile count and fixed stage cost must accompany working-set bytes.

## Fixed 96-thread multi-thread expert teams

The earlier comparison used the two extremes `1 x T` and `T x 1`. A second
experiment fixed the total worker count at 96 and compared genuine multi-thread
expert teams:

```text
1x96, 2x48, 3x32, 4x24, 6x16, 8x12, 12x8
```

Command:

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/run_thread_weight_working_set.py \
  --output /tmp/split_w13_expert_team_20260715.json \
  --experiments team-fixed-route,team-fixed-work \
  --team-experts 1,2,3,4,6,8,12 --team-total-threads 96 \
  --team-route 2040 --team-total-routes 2304 \
  --team-stage-mib 16 --hidden 4096 --warmup 2 --runs 7
```

All experts used H=4096, F=2048, and a 16 MiB split-W13 stage. The first control
kept each expert at route 2040. Total work therefore grows with expert count;
only aggregate throughput, not wall time, is comparable.

| Experts x threads | Active B stage | B stripe/thread | Wall | Aggregate TFLOP/s | vs. 1x96 |
|---:|---:|---:|---:|---:|---:|
| 1x96 | 16 MiB | 0.1875 MiB | 5.970 ms | 17.199 | 1.000x |
| 2x48 | 32 MiB | 0.375 MiB | 9.008 ms | 22.797 | 1.325x |
| 3x32 | 48 MiB | 0.500 MiB | 12.153 ms | 25.346 | 1.474x |
| 4x24 | 64 MiB | 0.6875 MiB | 16.348 ms | 25.123 | 1.461x |
| 6x16 | 96 MiB | 1.000 MiB | 24.258 ms | 25.396 | 1.477x |
| 8x12 | 128 MiB | 1.375 MiB | 33.000 ms | 24.892 | 1.447x |
| 12x8 | 192 MiB | 2.000 MiB | 73.184 ms | 16.836 | 0.979x |

Three teams are enough to raise steady aggregate throughput by 47.4%. Throughput
then remains near 25 TFLOP/s through eight teams, so additional teams do not
increase machine throughput. At 12 teams, the 2 MiB stripe consumes the entire
private L2 capacity and the aggregate active stage reaches 192 MiB; throughput
falls below the three-to-eight-team plateau by about one third.

The second control fixed total routes at 2304. Route count per expert is
`2304/E`, so total GEMM work (115.96 GFLOP), physical input elements, and fused
intermediate elements are invariant. Every route count remains M12 aligned.

| Experts x threads | Route/expert | Active B stage | Wall | Aggregate TFLOP/s | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1x96 | 2304 | 16 MiB | 6.674 ms | 17.375 | 1.000x |
| 2x48 | 1152 | 32 MiB | 5.195 ms | 22.321 | 1.285x |
| 3x32 | 768 | 48 MiB | 4.827 ms | 24.025 | 1.383x |
| 4x24 | 576 | 64 MiB | 5.058 ms | 22.925 | 1.319x |
| 6x16 | 384 | 96 MiB | 5.558 ms | 20.863 | 1.201x |
| 8x12 | 288 | 128 MiB | 5.935 ms | 19.538 | 1.125x |
| 12x8 | 192 | 192 MiB | 11.273 ms | 10.287 | 0.592x |

`3x32` is the best fixed-work shape: wall time is 27.7% lower, equivalent to
38.3% higher throughput, than `1x96`. Its p99 was 4.841 ms versus a 4.827 ms
median, so the result is well above the observed run-to-run noise.

### Source of the gain

The rows kernel gives each N-split lane a B stripe and makes that lane traverse
the expert's complete packed A. Under the project's cache accounting, repeated
A reads by later N tiles on the same core can hit private cache, but each lane
must establish its own A copy. For fixed total routes:

```text
A13 physical = 2304 * 4096 * 2 = 18 MiB
A2  physical = 2304 * 2048 * 2 =  9 MiB
modeled private-cache A demand = (18 + 9) MiB * threads_per_expert
```

| Shape | Modeled A demand | Unique packed B/full pipeline | Active B stage | B stripe/thread |
|---:|---:|---:|---:|---:|
| 1x96 | 2592 MiB | 48 MiB | 16 MiB | 0.1875 MiB |
| 3x32 | 864 MiB | 144 MiB | 48 MiB | 0.5000 MiB |
| 8x12 | 324 MiB | 384 MiB | 128 MiB | 1.3750 MiB |
| 12x8 | 216 MiB | 576 MiB | 192 MiB | 2.0000 MiB |

These A-demand values are a cache-level model, not literal measured DRAM bytes.
They expose the tradeoff: more expert teams reduce replicated A demand as
`1/E`, but increase unique B bytes and aggregate B capacity as `E`.

Stage timing attributes the useful part of this tradeoff primarily to W13:

| Stage | 1x96 | 3x32 | Time reduction | Share of wall-time saving |
|---:|---:|---:|---:|---:|
| Gather/pack A | 0.063 ms | 0.061 ms | 3.2% | 0.1% |
| Fused W13 | 4.839 ms | 3.225 ms | 33.4% | 87.4% |
| W2 | 1.758 ms | 1.555 ms | 11.5% | 11.0% |

At `3x32`, each W13 lane gets exactly 8 N tiles and each W2 lane gets exactly
16, eliminating the 96-lane long-lane imbalance. The corresponding lane
utilization rises from 88.9% to 100%; this can explain at most a 1.125x stage
speedup, less than the measured 1.500x W13 speedup. Its 0.5 MiB B stripe remains
comfortably inside private L2, while modeled replicated A demand falls by
66.7%, accounting for the remaining cache/granularity benefit. Smaller team
barriers are secondary. Further splitting shortens each route and grows B
pressure; at `12x8`, the 2 MiB stripe crosses the separately measured roughly
1.5 MiB robust private-L2 window. The route-2040 control also collapses at
`12x8`, showing that the final drop is not explained only by short routes.

## 8 MiB W13 plus 4 MiB W2 experts

The smaller TP4-like shape uses H=4096 and F=512. Full W13 is 8 MiB, W2 is
4 MiB, and each of the two split-W13 ranges is 4 MiB. One W13 range contains
only 64 N tiles, so a single expert cannot use more than 64 N-split lanes.

The direct one-versus-many comparison therefore fixed 64 total workers and used
`1x64, 2x32, 4x16, 8x8, 16x4`. With 2304 total routes:

| Experts x threads | Route/expert | Active B stage | B stripe/thread | Wall | Aggregate TFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 1x64 | 2304 | 4 MiB | 0.0625 MiB | 4.455 ms | 6.508 |
| 2x32 | 1152 | 8 MiB | 0.1250 MiB | 2.823 ms | 10.271 |
| 4x16 | 576 | 16 MiB | 0.2500 MiB | 1.997 ms | 14.518 |
| 8x8 | 288 | 32 MiB | 0.5000 MiB | **1.927 ms** | **15.044** |
| 16x4 | 144 | 64 MiB | 1.0000 MiB | 2.490 ms | 11.645 |

`8x8` is 2.312x faster than `1x64`, reducing wall time by 56.7%. All these
shapes divide both the 64 W13 tiles and 512 W2 tiles exactly, so long-lane
imbalance does not explain the gain. Physical packed-A sizes are 18 MiB for W13
and 2.25 MiB for W2. The modeled private-cache A demand falls from 1296 MiB at
`1x64` to 162 MiB at `8x8`, while each B stripe grows from 64 KiB to 512 KiB and
remains inside private L2.

The stage change is almost entirely W13: `1x64 -> 8x8` changes fused W13 from
3.804 to 1.246 ms, while W2 changes from 0.591 to 0.618 ms and gather/pack from
0.063 to 0.070 ms. The combined W13+W2 effective throughput at `8x8` is
15.55 TFLOP/s.

For completeness, a second grid used all 96 NUMA-local workers. A one-expert
baseline is impossible for this shape, so this grid only locates the best
multi-expert team shape under the larger thread budget:

| Experts x threads | Route/expert | Active B stage | B stripe/thread | Wall | Aggregate TFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 2x48 | 1152 | 8 MiB | 0.1250 MiB | 2.545 ms | 11.391 |
| 3x32 | 768 | 12 MiB | 0.1250 MiB | 1.982 ms | 14.625 |
| 4x24 | 576 | 16 MiB | 0.1875 MiB | 1.824 ms | 15.891 |
| 6x16 | 384 | 24 MiB | 0.2500 MiB | 1.554 ms | 18.660 |
| 8x12 | 288 | 32 MiB | 0.3750 MiB | **1.548 ms** | **18.725** |
| 12x8 | 192 | 48 MiB | 0.5000 MiB | 1.599 ms | 18.132 |
| 16x6 | 144 | 64 MiB | 0.6875 MiB | 1.826 ms | 15.873 |
| 24x4 | 96 | 96 MiB | 1.0000 MiB | 2.192 ms | 13.227 |

`6x16` and `8x12` differ by only 0.35% and form the practical optimum. Relative
to the same eight experts at `8x8`, `8x12` uses 50% more workers and reduces
wall time by 19.7%. Its W13+W2 effective throughput is 19.33 TFLOP/s. The
route-2040 control reaches a 23.1 TFLOP/s plateau at `16x6` to `24x4`, showing
that the fixed-work decline beyond eight teams is mainly short-route/fixed-cost
pressure rather than a B-residency collapse; all tested stripes remain at or
below 1 MiB.

When every expert instead keeps route 2040, extending the fixed-route grid
beyond 24 concurrent experts locates the actual saturation boundary:

| Experts x threads | Active B stage | B stripe/thread | Wall | Aggregate TFLOP/s |
|---:|---:|---:|---:|---:|
| 16x6 | 64 MiB | 0.6875 MiB | 17.790 ms | 23.086 |
| 24x4 | 96 MiB | 1.0000 MiB | 26.648 ms | **23.118** |
| 32x3 | 128 MiB | 1.3750 MiB | 36.298 ms | 22.630 |
| 48x2 | 192 MiB | 2.0000 MiB | 75.093 ms | 16.408 |
| 96x1 | 384 MiB | 4.0000 MiB | 290.800 ms | 8.474 |

The practical throughput plateau is therefore 16-24 concurrent experts with
4-6 threads per expert. At 32 experts, throughput is already 2.1% below the
plateau. At 48 experts, the 2 MiB stripe crosses the robust private-L2 window
and throughput falls by 29.0%; 96 one-thread experts are 63.3% below the
plateau. At 32 or more ready experts, balanced waves of 16-24 experts become
preferable. For example, 48 experts take about 53.30 ms as two `24x4` waves,
versus 75.09 ms as one `48x2` wave; 96 experts take about 106.59 ms as four
`24x4` waves, versus 290.80 ms as one `96x1` wave. Counts 25-31 were not
measured directly and should remain planner candidates rather than being
forcibly split.

Commands:

```bash
# Comparable single-expert baseline, 64 total workers
numactl --cpunodebind=0 --membind=0 taskset -c 0-63 \
  .venv/bin/python optimizations/fused_moe_sve/benchmarks/run_thread_weight_working_set.py \
  --output /tmp/split_w13_expert_team_8m4m_20260715.json \
  --experiments team-fixed-route,team-fixed-work \
  --team-experts 1,2,4,8,16 --team-total-threads 64 \
  --team-route 2040 --team-total-routes 2304 --team-stage-mib 4 \
  --hidden 4096 --warmup 2 --runs 7

# Full NUMA-node budget, 96 total workers
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python optimizations/fused_moe_sve/benchmarks/run_thread_weight_working_set.py \
  --output /tmp/split_w13_expert_team_8m4m_96t_20260715.json \
  --experiments team-fixed-route,team-fixed-work \
  --team-experts 2,3,4,6,8,12,16,24 --team-total-threads 96 \
  --team-route 2040 --team-total-routes 2304 --team-stage-mib 4 \
  --hidden 4096 --warmup 2 --runs 7

# Extend the route-2040 saturation boundary
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python optimizations/fused_moe_sve/benchmarks/run_thread_weight_working_set.py \
  --output /tmp/split_w13_expert_team_8m4m_long_route_tail_20260715.json \
  --experiments team-fixed-route --team-experts 24,32,48,96 \
  --team-total-threads 96 --team-route 2040 --team-stage-mib 4 \
  --hidden 4096 --warmup 2 --runs 7
```

## Conclusions

1. N-splitting one expert does not multiply its aggregate B capacity by T. It
   divides that B across private caches, while A scan traffic and synchronization
   grow with the participating lane count.
2. Running one expert per thread multiplies aggregate B capacity by T when F is
   fixed. Reducing F as `1/T` can hold B capacity constant, but it does not hold
   W13 input traffic, task granularity, or fixed overhead constant.
3. A useful model needs at least aggregate B stage, maximum per-thread B stripe,
   W13 input bytes, route count, and N tiles per task. A scalar aggregate-B
   working set is insufficient across these two mappings.
4. On this host, a per-thread B stripe around 1 MiB is a useful split point for
   exploiting private L2, while aggregate stages beyond roughly 32-64 MiB begin
   to expose shared-cache/bandwidth derate for the tested one-thread experts.
5. With 96 total workers and 16 MiB split-W13 experts, `3x32` is the best tested
   fixed-work team shape. The planner must balance replicated A demand, B stripe
   residency, aggregate B capacity, M12 lane balance, and route granularity;
   neither maximum threads per expert nor maximum concurrent experts is optimal.
6. Reducing full expert weights to 8 MiB W13 plus 4 MiB W2 moves the optimum to
   substantially more teams: `8x8` for a comparable 64-worker grid, and
   `6x16`/`8x12` for the full 96-worker grid. Team size must therefore depend on
   N-tile count and weight shape rather than only total available cores.
7. For 8 MiB plus 4 MiB experts whose routes are all 2040, cap each 96-worker
   wave near 16-24 experts and use 4-6 threads per expert. At 32 experts and
   above, larger waves lose enough B residency that balanced multiple waves are
   preferable; counts 25-31 still require direct validation.
