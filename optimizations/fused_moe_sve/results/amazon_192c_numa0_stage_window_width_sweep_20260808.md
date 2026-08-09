# Amazon 192-core NUMA0 stage-window team-width sweep

## Question

After landing the route-dependent W13/W2 packed-B windows, does a homogeneous
full-NUMA MoE wave still prefer at most eight threads per expert, and how does
that differ from the width that minimizes one expert's latency?

## Configuration

- Host: AmazonC5192Cores, NUMA0 CPUs `0-95`, memory bound to node 0.
- Shape: TP4 `H=4096`, `F=512`, 256 local/global experts, BF16 SiLU.
- Kernel: SVE JIT exact-M, split-W13 with two fallback ranges.
- Task windows: `amazon_c5_192c_tp4_f512_v4`; unsupported widths inherit the
  operator-wide split-W13 geometry.
- Pages: explicit 32 MiB HugeTLB through `/dev/hugepages-32M`.
- Inputs: consecutive distinct expert weights; isolated calls traverse 8
  experts, homogeneous calls traverse 192 experts.
- Samples: 2 warmups and 7 measured calls per point, randomized width order at
  each M.
- M grid: exact `1-12`, all production window boundaries and interior control
  points through 576, then `768,1024,1536,2040`.
- Isolated widths: every integer `1-32T`.
- Homogeneous widths: every equal-width shape that fills 96 cores exactly,
  `1,2,3,4,6,8,12,16,24,32T`.

The stale `profile_pure_gemm_probe.sh` process tree that had held a stopped
process on CPU 48 was terminated before the run. A three-second `mpstat` check
then reported all CPUs `0-95` 100% idle, and both NUMA nodes had all 160 of their
32 MiB HugeTLB pages free.

Exact command:

```bash
PYTHONPATH=src PYTHONUNBUFFERED=1 \
FUSED_CPP_PAGES=hugetlb FUSED_CPP_PAGE_SIZE_MB=32 \
FUSED_CPP_HUGETLBFS_PATH=/dev/hugepages-32M \
FUSED_CPP_MOE_HUGETLBFS_PATH=/dev/hugepages-32M \
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
numactl --cpunodebind=0 --membind=0 \
.venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_stage_window_team_widths.py \
  --output optimizations/fused_moe_sve/results/amazon_192c_numa0_stage_window_width_sweep_20260808.json \
  --cpu-ids 0-95 --warmup 2 --runs 7 --store-samples
```

## Data quality

The result contains 50 M values, 1,600 isolated points and 500 homogeneous
points, for 2,100 points total. Every point has all seven requested samples.

| Mode | median `p90/p10 - 1` | P90 | maximum |
| --- | ---: | ---: | ---: |
| Isolated | 0.87% | 6.05% | 16.04% |
| Homogeneous | 0.79% | 1.96% | 6.94% |

The conclusions below use homogeneous throughput gaps of at least 4.19%; exact
ties among nearby 1T/2T or 4T/8T points should still be treated as transition
regions when their margin is below about 2%.

## Main result

At every one of the 50 sampled M values, the best homogeneous full-NUMA shape
uses at most 8 threads per expert. The best width of 12T or wider loses between
4.19% and 26.14% aggregate throughput to the best width in 1-8T:

- median loss over all M: 10.97%;
- median at `M<=12`: 5.80%;
- median at sampled `24<=M<=215`: 17.40%;
- median at sampled `M>=216`: 10.37%;
- worst point: `M=72`, where the best wide shape loses 26.14%.

This is different from isolated latency. Every isolated point prefers more than
8 threads: `16-24T` at the very shortest routes, then usually `32T` from M=36
onward. At `M>=48`, 32T reduces isolated latency by roughly 13-20x versus 1T,
but using 32T for every expert reduces the number of concurrent experts enough
to lose aggregate throughput.

Representative points:

| M | isolated best T | isolated speedup vs 1T | full-NUMA best T | aggregate TFLOP/s | best `>=12T` loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 17 | 4.55x | 2 | 0.378 | 5.77% |
| 12 | 24 | 6.73x | 2 | 4.456 | 9.73% |
| 13 | 22 | 8.17x | 2 | 4.335 | 9.69% |
| 28 | 22 | 10.57x | 4 | 8.439 | 12.28% |
| 48 | 32 | 13.24x | 4 | 12.689 | 15.14% |
| 72 | 32 | 15.42x | 4 | 17.760 | 26.14% |
| 96 | 32 | 16.57x | 4 | 20.007 | 24.57% |
| 120 | 32 | 17.17x | 4 | 20.588 | 21.90% |
| 144 | 32 | 17.91x | 4 | 21.250 | 20.62% |
| 192 | 32 | 19.18x | 4 | 20.930 | 12.15% |
| 216 | 32 | 18.79x | 8 | 20.534 | 10.97% |
| 256 | 32 | 18.66x | 8 | 20.916 | 12.92% |
| 320 | 32 | 18.20x | 8 | 21.300 | 10.69% |
| 512 | 32 | 18.92x | 8 | 21.445 | 6.99% |
| 768 | 32 | 19.14x | 8 | 21.379 | 5.34% |
| 1024 | 32 | 19.27x | 8 | 21.263 | 5.33% |
| 1536 | 32 | 19.61x | 4 | 21.374 | 6.39% |
| 2040 | 32 | 19.99x | 4 | 21.423 | 6.59% |

## Width transitions

For the sampled M grid, the full-NUMA winner is:

- `M=1`: 2T;
- `M=2,3`: 1T;
- `M=4`: 2T;
- sampled `M=5-10`: 1T;
- sampled `M=11-13`: 2T;
- sampled `M=24-132`: 4T;
- `M=143`: 8T;
- sampled `M=144-215`: 4T;
- sampled `M=216-1024`: 8T;
- `M=1536,2040`: 4T.

These are measured grid results, not continuous route boundaries. Several
transitions are close: 1T/2T below M=13, 4T/8T near M=132-144 and M=216, and
4T/6T/8T from M=1024 onward.

The isolated winner is at the upper end of the search space for most routes, so
this scan does not claim that 32T is the global isolated optimum. It only proves
that limiting an isolated expert to 8T leaves substantial latency reduction on
the table.

## Interpretation

The current policy explains the main transitions:

1. At `M<=13`, one or two threads per expert preserve the maximum number of
   simultaneous cold-weight streams. Extra intra-expert parallelism lowers one
   expert's latency but lowers full-NUMA throughput.
2. In the short and medium bands, 4T balances packed-B retention, expert-level
   concurrency and team overhead. Unsupported 3T/6T/12T widths inherit the
   larger legacy window, so their measured result is the current production
   behavior, not an independently optimized-window upper bound.
3. At M=216 the policy increases W13 from 128 to 512 KiB/thread. A 4T task then
   uses four W13 ranges, while 8T uses two; W2 changes from eight ranges to four.
   Once the W13 A scan approaches private-L2 capacity, reducing those rescans is
   worth using the wider team, and 8T wins through M=1024.
4. Above the calibrated band, both widths inherit the legacy two-range W13 and
   one-range W2 geometry. At M=1536/2040, 4T/6T/8T aggregate performance is
   nearly flat; 4T wins by only 0.4-1.7%, so the return to 4T is a broad compute
   plateau rather than a sharp threshold.

## Scheduling consequence

For this exact TP4/96-core profile, the main-wave search may treat 1-8T as the
high-throughput domain when there are enough ready experts. Widths above 8T
must remain available for low-active-set and terminal-tail scheduling because
they substantially reduce isolated latency. This homogeneous sweep alone does
not justify pruning wide teams from mixed distributions or tail repartition.

Raw samples and all geometry fields are in
`amazon_192c_numa0_stage_window_width_sweep_20260808.json`.
