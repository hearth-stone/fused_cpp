# Four-core M12 refill contention

Date: 2026-07-19

## Question

Four one-thread M12 expert workers become substantially slower when they run
beside 23 long-route, four-thread expert teams. This experiment identifies the
measured cause of that slowdown and separates it from frequency, instruction
count, generic DRAM saturation, and warmup effects.

The result is expressed in terms of PMU-visible behavior. The host exposes no
uncore PMU, so this report does not assign the final bottleneck to an
unobservable LLC bank, coherent-mesh link, home node, or memory-controller
queue.

## Environment

- Host: `AmazonC5192Cores`, Arm Neoverse-V3, 192 cores.
- NUMA nodes: CPUs `0-95` and `96-191`.
- Cache topology: private 2 MiB L2 per core; one 96 MiB L3 shared by each
  96-core NUMA node; 64-byte cache lines.
- Kernel: Linux `7.0.0-1006-aws`.
- Compiler: GCC `15.2.0`, `-O2 -march=armv9-a+sve+bf16 -pthread`.
- Perf: `7.0.6`; ordinary-user PMU collection with
  `kernel.perf_event_paranoid=1`.
- Local source base: `d1f5ba6cee02696cd35673cb06268206a2812aa8`, with
  the current uncommitted production one-K-chunk workspace changes.
- Dynamic mixed-workload binary SHA-256:
  `1ddf4940551cec9be58e44eab878e18e05ba923c57c96d81dd654e19908b4287`.
- Synthetic resource-saturator binary SHA-256:
  `711e6ee3c1d0ff69ade1013edb82b87733a58e4f4b79e94f5e3e726488383536`.

Every fused expert used the production SVE M12 assembly bodies, split W13 into
two N ranges, and used one full K chunk. The shape was `H=4096, F=512`:

```text
W13 range 0: 4 MiB
W13 range 1: 4 MiB
W2:           4 MiB
packed weights per expert: 12 MiB
```

## Workloads and PMU scope

The short workload was a barrier-free dynamic queue of 512 distinct `M=12`
experts on four pinned workers. It rotated over 6 GiB of distinct packed
weights per invocation, so repeatedly timing one cached expert was not part of
the measurement.

The production long workload was:

```text
23 experts x 4 threads, M=2040, CPUs 0-91
4 consecutive rounds with distinct weight copies
```

The harness initialized and performed two warmups, printed a profiler-ready
marker, and stopped itself with `SIGSTOP`. Perf was then attached only to the
four short-worker TIDs. Initialization, long-worker counters, controller
threads, and stopped idle pools therefore did not enter the short-worker PMU
counts. The primary comparison used 30 measured invocations.

Counter groups were collected in separate otherwise-identical processes:

```text
task-clock,cycles,instructions,stall_backend_mem
l2d_cache_refill,ll_cache_rd,ll_cache_miss_rd,mem_access
```

Effective L2-refill delivery bandwidth is reported as:

```text
64 * l2d_cache_refill / short wall time
```

This is delivered cache-line bandwidth into the four private L2 caches, not a
DRAM-controller bandwidth counter.

## Primary result

| Short-worker metric, 30 invocations | Isolated | Local 23x4 long workload | Delta |
|---|---:|---:|---:|
| Short wall-time sum | 2087.987 ms | 2947.975 ms | +41.19% |
| Average cycle rate | 3.299575 GHz | 3.299639 GHz | +0.002% |
| Instructions | 114.9319 B | 115.0087 B | +0.067% |
| `stall_backend_mem` | 305.336 M | 924.172 M | +202.67% |
| L2 refills | 3.086084 B | 3.086220 B | +0.004% |
| `mem_access` | 31.5604 B | 31.5792 B | +0.060% |
| Effective L2-refill bandwidth | 94.53 GB/s | 67.10 GB/s | -29.01% |

The short workers execute the same instructions and complete the same number
of L2 refills at the same cycle rate. The time ratio is:

```text
98.1170 / 69.6476 = 1.4088
94.528  / 67.103  = 1.4087
```

The inverse refill-bandwidth ratio accounts for the measured wall-time ratio.
The directly observed failure mode is therefore slower service of a fixed
refill volume, not extra short-worker work or core-frequency loss.

## Resource-type controls

An independent 92-thread saturator occupied either CPUs `0-91` or `96-187`.
The register-only mode repeatedly issued independent SVE BFMMLA instructions.
The private-stream mode read one distinct 64 MiB buffer per worker. The
L2-resident mode repeatedly read one distinct 1 MiB buffer per worker, larger
than the 64 KiB L1 and smaller than the 2 MiB private L2. A four-worker-shared
mode let each group of four workers scan the same buffer.

| Concurrent control | Placement | Short time | Slowdown |
|---|---|---:|---:|
| None | NUMA0 | 69.61 ms | reference |
| Register-only BFMMLA, 92 cores | NUMA0 | 70.38 ms | +1.11% |
| Register-only BFMMLA, 92 cores | NUMA1 | 70.19 ms | +0.83% |
| Private 64 MiB stream, 92 cores | NUMA0 | 228.29 ms | +227.96% |
| Private 64 MiB stream, 92 cores | NUMA1 | 69.80 ms | +0.28% |
| Four-core-shared stream, 92 cores | NUMA1 | 69.85 ms | +0.35% |
| Private 1 MiB L2-resident stream, 92 cores | NUMA0 | 81.43 ms | +16.98% |
| Private 1 MiB L2-resident stream, 92 cores | NUMA1 | 81.21 ms | +16.66% |

The 1 MiB controls issued 7182 and 7197 GiB/s of logical load bytes from NUMA0
and NUMA1 respectively. That is a loop-byte rate, not a claim about physical
uncore bandwidth. Under the remote 1 MiB control, short-worker instructions
changed by only +0.037%, average cycle rate by -0.001%, and L2-refill count by
-0.015%, while `stall_backend_mem` increased by 39.38%.

The production `23x4` long workload independently measured the following over
ten four-round invocations:

| Production long metric | Count | Rate |
|---|---:|---:|
| `l1d_cache` | 372.922 B | 352.01 B events/s |
| `mem_access` | 321.531 B | 303.50 B events/s |
| L1D refill | 4.036 B | 243.82 GB/s |
| L2D refill | 3.036 B | 183.41 GB/s |

Thus high private-cache data activity exists in the real long kernel, and the
L2-resident synthetic control is sufficient to reproduce its cross-NUMA
effect. Generic remote DRAM streaming and register-only compute are not.

## Cross-NUMA production dose control

Two independent benchmark processes placed the short weights on NUMA0 and the
long weights on NUMA1. `numastat -p` measured 10387.46 MiB of the short process
on node 0 and 10384.70 MiB of the long process on node 1. Leaving the warmed
long process stopped gave 69.60 ms, ruling out the second process and warmup
state as causes.

The production long work and its total `mem_access` count remained
approximately fixed while its team width changed. Higher team width completed
that work faster and raised the concurrent event rate:

| Remote long shape | Long `mem_access` rate | Short time | Short slowdown |
|---|---:|---:|---:|
| 23 experts x 1T | 72.14 B events/s | 69.90 ms | +0.42% |
| 23 experts x 2T | 145.84 B events/s | 75.03 ms | +7.80% |
| 23 experts x 4T | 303.25 B events/s | 79.34 ms | +13.99% |

This dose response connects the cross-NUMA residual to the production
kernel's concurrent data-access rate without attributing it to an unmeasured
physical power or interconnect mechanism.

## Topology-local control

The next control fixed the long work at 23 one-thread experts and moved only
the 23-core window. With the short workers on CPUs `92-95`:

| Long cores | Short time |
|---|---:|
| none | 69.62 ms |
| 0-22 | 77.86 ms |
| 23-45 | 77.78 ms |
| 46-68 | 89.11 ms |
| 69-91 | 98.93 ms |

Mirroring the short workers to CPUs `0-3` reversed the direction:

| Long placement relative to short CPUs 0-3 | Short time | Slowdown | Refill bandwidth | `stall_backend_mem` delta |
|---|---:|---:|---:|---:|
| None | 69.57 ms | reference | 94.67 GB/s | reference |
| Far, CPUs 69-91 | 78.42 ms | +12.73% | 83.70 GB/s | +60.14% |
| Near, CPUs 4-26 | 107.23 ms | +54.15% | 61.18 GB/s | +169.30% |

Near and far runs changed short-worker instruction count by at most 0.042%,
average cycle rate by at most 0.002%, and refill count by at most 0.089%.
The position result is therefore a topology-local refill-service effect, not a
different short-worker code path.

Linux exposes all CPUs `0-95` as one 96 MiB L3 domain and exposes no smaller
cache cluster. The experiment proves locality within that domain but cannot
name its internal hardware boundary.

## Decomposition and conclusion

For the original CPUs `92-95` placement:

```text
isolated short                         69.600 ms
remote-NUMA production 23x4 long      79.341 ms  (+9.742 ms)
local-NUMA production 23x4 long       98.266 ms  (+28.666 ms total)
local-placement increment                        18.925 ms
```

The local-placement increment is 66.02% of the original added latency. The
measurements support two components:

1. A chip-wide effect associated with high concurrent private-cache data
   activity. It is reproduced by a 1 MiB-per-core L2-resident load from either
   NUMA node and follows the real long kernel's `mem_access` event rate.
2. An additional topology-local reduction in the service rate of private-L2
   refills from the hierarchy below L2. It follows core placement in both
   directions and accounts for most of the original added latency.

This is bandwidth contention in the operational sense that the four short
cores receive less effective L2-refill delivery bandwidth. It is not equivalent
to the NUMA DRAM bandwidth reaching its STREAM ceiling.

## A-traffic boundary

The experiment does not produce an A-refill percentage. Changing the long team
from 1T to 2T to 4T also changes each thread's B stripe from 4 MiB to 2 MiB to
1 MiB. The first two sizes do not have the same private-L2 reuse behavior as
the 1 MiB stripe. The measured long L2-refill rates were approximately
327/326/183 GB/s for 1T/2T/4T, while interference increased with proximity and
concurrent event rate rather than aggregate refill bandwidth. Assigning that
difference to A alone would therefore be unsupported.

## Scheduling consequence

For this host, a contention model based only on aggregate packed-B bytes or
NUMA DRAM bandwidth is incomplete. It needs at least two calibrated pressures:

```text
chip-wide concurrent cache-data activity
topology-local L2-refill service pressure
```

M12 experts have no reusable-B capacity demand, but they still consume refill
service bandwidth. Conversely, a long-route expert whose B stripe is resident
in private L2 can still interfere through its high cache-data event rate.

## Reproduction

The temporary harness sources and consolidated machine-readable result remain
under `tmp/stagger_experiment/` in the measurement workspace. Representative
commands on the remote host were:

```bash
c++ -std=c++20 -O2 -march=armv9-a+sve+bf16 -pthread \
  -I/home/ubuntu/zhangxu/fused_cpp/refs/i8gemm/lib \
  /tmp/fused_cpp_mixed_23x4/unfused_pipeline.cpp \
  /tmp/fused_cpp_mixed_23x4/bench_multi_round_dynamic_short.cpp \
  /home/ubuntu/zhangxu/fused_cpp/csrc/moe/arm/sve_bf16/kernels.S \
  -o /tmp/fused_cpp_mixed_23x4/bench_multi_round_dynamic_short

bash /tmp/fused_cpp_mixed_23x4/profile_dynamic_short.sh short core
bash /tmp/fused_cpp_mixed_23x4/profile_dynamic_short.sh mixed cache
bash /tmp/fused_cpp_mixed_23x4/profile_cross_numa_short.sh core 30 active
bash /tmp/fused_cpp_mixed_23x4/profile_short_with_saturator.sh l2_remote 20
```

The consolidated local result is
`tmp/stagger_experiment/results_four_short_core_memory_interference_amazon_192c_20260719.json`.
