# Amazon C5 192-core SVE route merge

## Setup

- Host: `AmazonC5192Cores`
- CPU/memory binding: NUMA0, CPUs `0-95`
- Input: `tokens=2048`, `top_k=6`, `H=4096`
- Fused-MoE shape: `E=8`, `F=512`
- E2E schedule: eight concurrent experts, 12 threads per expert, async bridge
- W13 policy: split-W13 enabled
- Statistic: median of 31 timed runs after five warmups

The tree kernel computes `(r0*w0 + r1*w1)`, `(r2*w2 + r3*w3)`, and
`(r4*w4 + r5*w5)`, then reduces the three pair sums and stores BF16. Its
reduction order intentionally differs from the sequential-FMA baseline.

## Isolated merge

The reported bandwidth is logical route payload divided by wall time. It is
not a DRAM-bandwidth claim because the benchmark repeatedly consumes the same
post-scatter buffer and the aggregate private/shared caches can serve part of
it.

| Route dtype | Variant | Median ms | Payload GB/s | Speedup |
|---|---:|---:|---:|---:|
| FP32 | sequential | 0.216130 | 1009.36 | 1.000x |
| FP32 | tree U1 | 0.192161 | 1135.26 | 1.125x |
| FP32 | tree U2 | **0.188875** | **1155.01** | **1.144x** |
| FP32 | tree U4 | 0.195887 | 1113.67 | 1.103x |
| BF16 | sequential | 0.145035 | 810.08 | 1.000x |
| BF16 | tree U1 | 0.133747 | 878.45 | 1.084x |
| BF16 | tree U2 | 0.124275 | 945.40 | 1.167x |
| BF16 | tree U4 | **0.122908** | **955.92** | **1.180x** |

U2 is the best isolated FP32-route kernel. U4 is best for BF16 route, where
four independent widening/load chains hide more latency. Disassembly of U4's
full-vector loop shows no SVE stack spills.

## Full fused MoE

### FP32 route buffer (current default)

| Variant | Median ms | E2E gain | Max abs vs sequential |
|---|---:|---:|---:|
| sequential | 9.312 | 0.00% | 0 |
| tree U1 | **9.276** | **0.38%** | 0 |
| tree U2 | 9.289 | 0.25% | 0 |
| tree U4 | 9.304 | 0.08% | 0 |

### BF16 route buffer

| Variant | Median ms | E2E gain | Max abs vs sequential |
|---|---:|---:|---:|
| sequential | 8.242 | 0.00% | 0 |
| tree U1 | **8.185** | **0.69%** | 0 |
| tree U2 | 8.189 | 0.65% | 0 |
| tree U4 | 8.192 | 0.61% | 0 |

The isolated kernel improves materially, but the absolute saving is only
about 0.02-0.03 ms at this 96-thread point. W13/W2 dominate the 8-9 ms E2E
operator, so the retained E2E gain is below 1%. U1 has the best E2E median;
the larger isolated unroll does not matter once GEMM and dispatch costs are
included.

Running the normal entrypoint measured about 256 ms because it recomputed the
planner decision on every invocation. Its internal expert-compute stage was
about 67 ms and hot merge was 0.58-1.16 ms. That entrypoint is therefore not a
valid microkernel comparison; the preplanned async bridge is used above.

## Correctness and decision

- The standalone check bit-matched U1/U2/U4 against an explicit scalar
  6-to-3-to-2-to-1 reference for FP32 and BF16 route sources, including a
  non-vector-aligned hidden tail.
- Sync, scheduled, and async fused-MoE tests passed for both route dtypes and
  all three unrolls.
- The long-input E2E sample happened to bit-match the sequential BF16 output,
  but this is not guaranteed for arbitrary data because the FP32 reduction
  order changed.

The initial measurement decision kept the feature experimental and default-off:
the measured sub-1% E2E gain alone did not justify changing numerical ordering.
The later default-adoption decision below also considers removal of the
thread-local FP32 allocation and accumulator traffic.

## Generalized TopK templates and fallback

The follow-up implementation specializes `top_k=2/4/6/8` with compile-time
adjacent binary trees and sends other positive values to an ordered runtime
slot loop. Both policies have U1/U2/U4 hidden-axis variants. The existing
top-k=1 direct-scatter path is unchanged. Disassembly confirms that fixed
TopK bodies contain no slot-loop branch, dynamic bodies contain one, and the
top-k=8 U4 body has no SVE register spill. The FP32 and BF16 implementations
together add about 18 KiB of object text.

The isolated table below uses `tokens=2048`, `H=4096`, 96 threads on NUMA0,
five warmups, and the median of 31 runs. Speedup is against the existing
sequential FP32-accumulator implementation. Odd top-k values exercise the
dynamic fallback.

| TopK | Policy | FP32 best | FP32 speedup | BF16 best | BF16 speedup |
|---:|---|---:|---:|---:|---:|
| 2 | fixed tree | U4, 0.090320 ms | 1.222x | U4, 0.094276 ms | 1.129x |
| 3 | dynamic | U4, 0.096490 ms | 1.119x | U4, 0.098341 ms | 1.115x |
| 4 | fixed tree | U4, 0.099653 ms | 1.277x | U4, 0.106782 ms | 1.147x |
| 5 | dynamic | U4, 0.112992 ms | 1.224x | U4, 0.107453 ms | 1.225x |
| 6 | fixed tree | U1, 0.190731 ms | 1.090x | U4, 0.122466 ms | 1.200x |
| 7 | dynamic | U4, 0.345139 ms | 0.895x | U4, 0.124691 ms | 1.219x |
| 8 | fixed tree | U4, 0.447480 ms | 0.792x | U4, 0.136636 ms | 1.243x |

```bash
for top_k in 2 3 4 5 6 7 8; do
  numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
    optimizations/fused_moe_sve/benchmarks/bench_route_merge_tree \
    --tokens 2048 --top-k "$top_k" --hidden 4096 --threads 96 \
    --source both --warmup 5 --runs 31 --check
done
```

FP32 route stops scaling at top-k 7-8 even without register spills. Those
shapes interleave seven or eight 16 KiB FP32 route rows per token instead of
streaming one row at a time as the baseline does; cache-stream and prefetch
behavior is the leading explanation, but no PMU attribution was collected in
this run. BF16 halves each stream and remains faster for every tested top-k.

The preplanned async E2E table uses `E=8`, `F=512`, split-W13, 12 threads per
expert, three warmups, and the median of 11 runs. It reports the best unroll for
each route dtype.

| TopK | FP32 baseline | FP32 best gain | BF16 baseline | BF16 best gain |
|---:|---:|---:|---:|---:|
| 2 | 3.270 ms | U2, 1.04% | 3.009 ms | U2, 0.88% |
| 4 | 6.262 ms | U2, 0.82% | 5.670 ms | U4, 0.37% |
| 5 | 7.774 ms | U1, 0.45% | 6.966 ms | U4, 0.68% |
| 6 | 9.314 ms | U1, 0.22% | 8.299 ms | U4, 0.67% |
| 8 | 12.491 ms | U4, 0.30% | 10.909 ms | U4, 0.59% |

Each row was collected with the following command, changing `--top-k` to the
row value and adding `--bf16-route` for the BF16 column:

```bash
PYTHONPATH=src numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_route_merge_e2e.py \
  --path async --tokens 2048 --hidden 4096 --intermediate 512 \
  --experts 8 --top-k 6 --threads 96 --warmup 3 --runs 11
```

All sampled E2E outputs matched the sequential BF16 outputs exactly. Fixed
trees can still differ for other data because their FP32 association changes.
U2 and U4 therefore remain explicit experimental variants; U1 was subsequently
adopted as the SVE default under the precision contract described below.

## TopK=6 thread sweep

The standalone runner uses a persistent pinned thread pool. A worker owns a
contiguous token range and reduces every TopK route across the full hidden
dimension for those tokens; a token row is never split across workers. Timed
runs include the pool's start and completion barriers, but not thread creation
or destruction. With 2048 tokens, 96 workers receive at most 22 token rows
each.

The table reports the fastest SVE unroll at each thread count. Speedup compares
that variant with the sequential-accumulator kernel at the same thread count.

| Threads | FP32 baseline | FP32 best | FP32 speedup | BF16 baseline | BF16 best | BF16 speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.392 ms | U4, 5.699 ms | 1.122x | 5.719 ms | U4, 4.821 ms | 1.186x |
| 2 | 3.548 ms | U4, 3.206 ms | 1.107x | 2.876 ms | U4, 2.450 ms | 1.174x |
| 4 | 1.869 ms | U4, 1.655 ms | 1.129x | 1.437 ms | U4, 1.262 ms | 1.139x |
| 8 | 1.205 ms | U2, 1.117 ms | 1.079x | 0.725 ms | U4, 0.636 ms | 1.140x |
| 16 | 0.747 ms | U1, 0.724 ms | 1.032x | 0.376 ms | U2, 0.379 ms | 0.993x |
| 32 | 0.565 ms | U2, 0.637 ms | 0.887x | 0.268 ms | U2, 0.221 ms | 1.213x |
| 48 | 0.408 ms | U4, 0.488 ms | 0.836x | 0.172 ms | U4, 0.135 ms | 1.279x |
| 64 | 0.332 ms | U4, 0.425 ms | 0.781x | 0.150 ms | U4, 0.121 ms | 1.236x |
| 96 | 0.227 ms | U2, 0.190 ms | 1.195x | 0.147 ms | U4, 0.125 ms | 1.181x |

These `copies=1` results measure repeated consumption of the same buffer. The
optimized FP32 kernel scales from 5.699 ms at one thread to 0.190 ms at 96
threads, or 30.0x. BF16 reaches its minimum of 0.121 ms at 64 threads (39.8x
versus one thread) and does not improve at 96 threads. These are useful
hot-buffer numbers, but they do not represent the compulsory traffic of one
merge invocation.

To isolate that cache transition, a second sweep fixed FP32 route at 64 workers
and varied only the token count. The route working set below excludes output
stores and is `ceil(tokens / 64) * 6 * 4096 * 4` bytes per worker.

| Tokens | Rows/worker | Route MiB/worker | Best tree speedup |
|---:|---:|---:|---:|
| 512 | 8 | 0.75 | 1.217x |
| 768 | 12 | 1.13 | 1.163x |
| 1024 | 16 | 1.50 | 1.248x |
| 1280 | 20 | 1.88 | 1.264x |
| 1536 | 24 | 2.25 | 1.088x |
| 2048 | 32 | 3.00 | 0.788x |
| 2560 | 40 | 3.75 | 0.836x |
| 3072 | 48 | 4.50 | 0.843x |
| 4096 | 64 | 6.00 | 0.784x |

The reversal near the machine's 2 MiB private L2 demonstrates a benchmark
cache-residency artifact: stable worker pinning and token ranges let successive
timed iterations reuse the same route lines. It is not an intrinsic working-set
requirement of route merge. Within one invocation each route element is read
once and each output element is written once.

## Single-pass streaming merge

The benchmark's `--copies 8` mode rotates both source and destination through
eight physical buffers. For TopK=6 the ring is about 1.63 GiB for FP32 route or
0.88 GiB for BF16 route, so a source is not reused until substantially more
than the aggregate cache capacity has been traversed. This models compulsory
single-pass traffic rather than repeated hot-buffer access.

| Threads | FP32 baseline | FP32 best | FP32 speedup | BF16 baseline | BF16 best | BF16 speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.596 ms | U4, 6.244 ms | 1.056x | 5.940 ms | U4, 4.920 ms | 1.207x |
| 16 | 0.718 ms | U2, 0.746 ms | 0.962x | 0.451 ms | U4, 0.424 ms | 1.064x |
| 32 | 0.733 ms | U4, 0.841 ms | 0.871x | 0.367 ms | U4, 0.431 ms | 0.852x |
| 64 | 0.657 ms | U4, 0.738 ms | 0.891x | 0.344 ms | U4, 0.392 ms | 0.877x |
| 96 | 0.612 ms | U2, 0.667 ms | 0.917x | 0.347 ms | U2, 0.354 ms | 0.981x |

The old and new kernels have the same compulsory route reads and output
writes. The old FP32 accumulator is only 16 KiB and its extra traffic remains
in L1. At enough concurrency, external traffic dominates and removing that L1
traffic no longer helps. The baseline also streams one route row at a time,
whereas the tree interleaves six route streams per worker and computes six
multiplies plus five adds instead of six FMAs. Under cold high-thread traffic,
the baseline therefore reaches the memory hierarchy more efficiently. Cache
still matters for producer-consumer residency and refill machinery, but there
is no capacity reuse intrinsic to a single route-merge call.

## PMU attribution of the streaming reversal

The cold FP32 reversal was profiled on 2026-07-21 with Linux
`7.0.0-1006-aws`, GCC 15.2.0, perf 7.0.12, and
`kernel.perf_event_paranoid=1`. The benchmark used `tokens=2048`, `top_k=6`,
`H=4096`, `copies=8`, 16 warmups, 128 timed invocations, and three fresh
processes per point. Each process was bound to NUMA0 CPUs starting at CPU 0.

The benchmark's `--stop-before-run` option stops after allocating and
initializing all buffers and creating the pinned worker pool. Perf was then
attached only to the worker TIDs before the process was resumed. The PMU
counts therefore include warmups, measured calls, and worker barriers, but not
allocation, input generation, thread creation, or the main thread.

FP32 route input is exactly 192 MiB per invocation. Including BF16 output and
weights, the logical payload is 208.05 MiB. The table uses V3 implementation
events `L1D_CACHE_MISS` (`0x8144`), `L2D_CACHE_HWPRF` (`0x8155`),
`STALL_BACKEND_MEMBOUND` (`0x8164`), and `STALL_BACKEND_L1D` (`0x8165`).
Event semantics come from the
[Arm Neoverse V3 Core Telemetry Specification](https://documentation-service.arm.com/static/66f71ac61669c0388dca6d9b).
Stall categories can overlap and must not be added together.

| T | Variant | Median ms | Demand-L1 miss rate | L2 HW prefetches/call | L2 refill GiB/call | L2 refill GB/s | Mem-resource stall | Pending-L1 stall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 14 | sequential | 0.756 | 3.03% | 4.680M | 0.18819 | 267.2 | 58.9% | 49.5% |
| 14 | tree U1 | **0.738** | 8.73% | 3.559M | 0.18803 | **273.6** | 61.6% | 56.0% |
| 16 | sequential | **0.716** | 3.59% | 4.687M | 0.18818 | **282.2** | 61.8% | 46.0% |
| 16 | tree U1 | 0.753 | 8.66% | 3.463M | 0.18807 | 268.2 | 65.8% | 56.3% |
| 20 | sequential | **0.697** | 5.04% | 4.671M | 0.18824 | **289.9** | 69.2% | 38.2% |
| 20 | tree U1 | 0.839 | 8.47% | 3.457M | 0.18820 | 240.9 | 74.8% | 63.0% |

Both kernels refill essentially the same 0.1882 GiB per call, close to the
192 MiB compulsory route input. The reversal is therefore not caused by byte
amplification or a larger LLC working set. The fixed tree issues TopK source
streams at each hidden-axis vector, while the sequential kernel completes one
contiguous route row before starting the next and keeps its 16 KiB accumulator
in L1. The sequential pattern generates about 32-35% more L2 hardware-prefetch
accesses and materially fewer demand L1 misses.

`L2D_CACHE_REFILL` names the destination of the fill: the private L2 of the
requesting core. It does not imply that private L2 arrays contend with each
other. Both an L3 hit returned to L2 and an L3 miss serviced from memory end in
an L2 refill, so the shared bottleneck can be in the L3, interconnect, memory
controllers, or their request/response queues. At 20 workers `BUS_ACCESS_RD`
was 7.08M beats/call for sequential and 7.16M for U1, again showing nearly
equal downstream traffic; the observed delivery rate was 10.1 versus
8.5 Gbeat/s.

This host has a 96 MiB L3 shared by NUMA0 CPUs 0-95, while one FP32 route source
is 192 MiB and the benchmark rotates eight copies. The workload therefore
cannot remain L3-resident. The host exposes no uncore/CMN L3 PMU, its
`L3D_CACHE_REFILL` core event reports zero, and the observed `LL_CACHE_*` counts
cover too little traffic to serve as a total-LLC byte counter here. The current
data can attribute the reversal to the shared downstream service path, but
cannot separate L3-bank/interconnect contention from DRAM-controller pressure.

At 14 workers, U1 still wins because it retires 71.36M instructions per call
versus 78.99M for the sequential FP32 path and avoids accumulator traffic. At
16 workers the memory penalty overtakes that saving. At 20 workers U1 has 27%
more memory-resource stall cycles and 94% more pending-L1 stall cycles than the
sequential path; TLB and store stalls are below 0.01% of cycles. Frontend stall
is also negligible. Conversely, `STALL_BACKEND_BUSY` is higher for the
sequential kernel (33.1% versus 9.2% at 20 workers), showing that the baseline
retains more execution-side pressure while U1 has shifted to the memory side.

A fixed-route-byte control kept FP32 route input at 192 MiB and 20 workers,
then changed TopK while setting `tokens * top_k = 12288`. U1's demand-L1 miss
rate rose from 1.96% at TopK=2 to 5.85%, 8.46%, and 12.57% at TopK=4/6/8.
The sequential path stayed between 4.29% and 5.41%. Output traffic changes with
the token count, so this is supporting rather than single-variable evidence,
but the monotonic U1 miss trend directly tracks the number of interleaved
source streams.

Halving source width moves the transition later. In an unprofiled BF16 sweep,
U1 was 3.8% faster at 20 workers (`0.3849` versus `0.3995` ms) but 13.9% slower
in latency at 24 workers (`0.4270` versus `0.3748` ms). At 24 workers PMU still
reported nearly equal refill volume (0.0942 versus 0.0948 GiB/call), while U1
had a 3.20% demand-miss rate versus 0.65% and delivered 237.0 versus
278.7 GB/s of L2 refills. This shift from the FP32 14-16 worker crossover to
the BF16 20-24 worker crossover confirms that shared refill pressure is part
of the mechanism.

The PMU can localize the limit to demand-load tracking/refill resources between
L1 and the shared downstream hierarchy, amplified when shared-service latency
rises. It cannot identify one physical queue (for example, a load queue, miss
status entry, or prefetch stream tracker) from these aggregate events alone.
The defensible planner signal is therefore concurrent cold stream count plus
measured refill pressure, not logical bytes or STREAM bandwidth alone.

## Non-temporal output store (rejected)

An experimental output path packed the low BF16 halfwords with `UZP1` and used
SVE `STNT1H` instead of the cached narrowing `ST1H`. SVE has no non-temporal
narrowing store from `z.s`, so the pack is required. Correctness passed for
TopK 1-9, FP32/BF16 route inputs, and a non-vector-aligned hidden tail.

At 96 threads, the isolated results were:

| Input state | Route dtype | Best cached | Best non-temporal | NT change |
|---|---|---:|---:|---:|
| hot (`copies=1`) | FP32 | 0.1865 ms | 0.1897 ms | -1.7% |
| hot (`copies=1`) | BF16 | 0.1216 ms | 0.1282 ms | -5.2% |
| streaming (`copies=8`) | FP32 | 0.6675 ms | 0.6711 ms | -0.5% |
| streaming (`copies=8`) | BF16 | 0.3604 ms | 0.3565 ms | +1.1% |

A same-process, order-rotated async E2E comparison used 31 samples:

| Route dtype | Best cached | Best non-temporal | NT change |
|---|---:|---:|---:|
| FP32 | 9.303 ms | 9.332 ms | -0.3% |
| BF16 | 8.231 ms | 8.247 ms | -0.2% |

Only 16 MiB of final output is stored versus 112 MiB (BF16 route) or 208 MiB
(FP32 route) of logical merge payload. Avoiding some write allocation cannot
offset packing overhead consistently, and the output is consumed immediately
by the next model stage. The production option was therefore removed rather
than retaining a default-off duplicate kernel set.

## Default adoption

On 2026-07-16, U1 became the default weighted-route merge for the SVE backend.
This removes the per-worker `H * sizeof(float)` allocation, its zero-fill, and
the repeated FP32 accumulator loads/stores. The existing E2E measurements show
U1 at 9.276 ms versus 9.312 ms for FP32 route output and 8.185 ms versus
8.242 ms for BF16 route output on the documented TopK=6 shape. The performance
gain is small, but avoiding a hot-path allocation is the primary adoption
reason.

The fixed TopK 2/4/6/8 templates use adjacent tree association, so arbitrary
inputs may differ from the old sequential-FMA path within the existing BF16
output tolerance. Other TopK values retain ordered runtime accumulation.
Setting `FUSED_CPP_MOE_SVE_ROUTE_MERGE_UNROLL=0` restores the old accumulator
path; non-SVE backends continue to select it automatically. Cached narrowing
`ST1H` remains the output store.
