# Independent W13/W2 packed-B windows on AmazonC5192Cores

Date: 2026-07-26

## Question

Test whether a packed-B budget of 1 MiB per active W13 worker and 0.5 MiB
per active W2 worker, observed on another machine, transfers to the 192-core
Neoverse-V3 host.

The existing public API has one global `weight_window_bytes` value. This
experiment adds async-only environment overrides:

```text
FUSED_CPP_MOE_EXPERIMENT_W13_WEIGHT_WINDOW_BYTES
FUSED_CPP_MOE_EXPERIMENT_W2_WEIGHT_WINDOW_BYTES
```

They fall back to the public global value when unset. The benchmark accepts
`--window-pairs-mib W13:W2,...`. The production default and planner candidate
space are unchanged.

## Setup

- Host: `AmazonC5192Cores`, Neoverse-V3, 192 cores, two 96-core NUMA nodes.
- Primary affinity: NUMA0 CPUs `0-95`, local memory.
- Cross-check affinity: NUMA1 CPUs `96-191`, local memory.
- Shape: TP4 BF16 SiLU expert, `H=4096`, `F=512`.
- Schedule: 24 experts x 4 threads unless stated otherwise.
- Weights: distinct packed W13/W2 tensors for every active expert.
- Samples: 5-10 warmups and 20-50 measured calls.
- Every window pair passed the benchmark's bit-exact BF16 preflight.

For a 4-thread team, a `4:2` target means approximately 1 MiB of W13 and
0.5 MiB of W2 packed B per active worker. The baseline `4:4` target gives
approximately 1 MiB per worker for both stages.

Example:

```bash
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE FUSED_CPP_MOE_PREPACK_THREADS=96 \
PYTHONPATH=src numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_weight_windows.py \
  --experts 24 --threads-per-expert 4 --routes 2040 \
  --window-pairs-mib 4:4,4:2,4:1,2:4,2:2 \
  --warmup 5 --runs 20
```

## Transfer test

The delta is candidate `4:2` time relative to baseline `4:4`; negative is
faster.

| Routes | Baseline 4:4 ms | Candidate 4:2 ms | Time delta |
| ---: | ---: | ---: | ---: |
| 12 | 0.587 | 0.589 | +0.3% |
| 192 | 4.219 | 3.966 | -6.0% |
| 384 | 6.923 | 6.774 | -2.2% |
| 768 | 12.432 | 12.459 | +0.2% |
| 1536 | 23.579 | 24.066 | +2.1% |
| 2040 | 30.824 | 31.632 | +2.6% |

The `M=2040` result reproduced after reversing variant order:
30.843/31.645 ms. NUMA1 reproduced it at 30.879/31.631 ms, or a 2.4%
candidate regression.

The same long-route conclusion holds at another team width:

| Schedule | Equal per-worker target | Half-size W2 target | Time delta |
| --- | ---: | ---: | ---: |
| 24 x 4T, `4:4` vs `4:2` | 30.824 ms | 31.632 ms | +2.6% |
| 48 x 2T, `2:2` vs `2:1` | 62.423 ms | 64.229 ms | +2.9% |

The 96 x 1T point retained the same ordering but remained subject to the
known 96T host instability, so it is not used for the decision.

## Independent sweeps

At `M=384`, holding W2 at 0.5 MiB identifies a 2 MiB W13 range:

| W13:W2 MiB | NUMA0 ms | NUMA1 ms |
| ---: | ---: | ---: |
| 1:0.5 | 6.744 | 6.726 |
| 2:0.5 | **6.299** | **6.276** |
| 4:0.5 | 6.485 | 6.426 |

Holding W13 at 2 MiB prefers a much smaller W2 range:

| W13:W2 MiB | NUMA0 ms | NUMA1 ms |
| ---: | ---: | ---: |
| 2:0.5 | **6.299** | **6.276** |
| 2:1 | 6.313 | 6.304 |
| 2:2 | 6.445 | 6.414 |
| 2:4 | 6.578 | 6.526 |

Thus the `M=384`, 4T optimum in this grid is approximately 0.5 MiB of W13
and 0.125-0.25 MiB of W2 per worker. This does establish a stage difference,
but it is not the 1/0.5 MiB per-worker point transferred from the other host.

At `M=192`, the best tested region moves down again: `0.5:0.5` takes
3.106-3.127 ms, while `1:0.5` takes 3.302-3.305 ms and `4:4` takes
4.163-4.219 ms. At `M>=768`, the advantage disappears; by `M=1536-2040`,
shrinking either stage is a repeatable regression.

## PMU check at M=384

Two 300-call process-level `perf stat` runs compared `4:4` with the measured
`2:0.5` pair. Initialization was identical and retired instructions differed
by only 0.06%.

| Counter | 4:4 MiB | 2:0.5 MiB | Change |
| --- | ---: | ---: | ---: |
| Median time | 6.801 ms | 6.288 ms | -7.5% |
| Cycles | 650.59 B | 597.55 B | -8.2% |
| Instructions | 1,934.77 B | 1,933.58 B | -0.06% |
| L2D refills | 8.761 B | 10.577 B | +20.7% |
| LL-cache reads | 613.56 M | 604.10 M | -1.5% |
| LL-cache read misses | 499.66 M | 465.39 M | -6.9% |
| Memory-stall cycles | 65.85 B | 39.54 B | -39.9% |

The improvement is therefore not fewer load instructions or fewer L2
refills. The smaller active range causes more L2 refill activity but fewer
last-level misses and much less high-latency memory stall. This is consistent
with shortening B reuse distance and keeping a larger fraction of refills
served by the shared cache instead of lower memory. The host's
`l3d_cache_lmiss_rd` event remained zero and is not used.

## Conclusion

W13 and W2 can have different optimal windows, but neither stage has a
route-independent per-worker constant on this host:

- Long routes (`M=1536-2040`) prefer about 1 MiB per worker for both stages.
- Medium routes (`M=192-384`) prefer smaller windows, and `M=384` clearly
  prefers a smaller W2 window than W13.
- One physical M12 panel is insensitive at the resolution of this test.

The portable decision variable is therefore
`(routes, team_width, w13_window_bytes, w2_window_bytes)`, not one global
window or a fixed W13:W2 ratio. Independent stage windows should remain an
experimental dimension until their complete isolated/contention profile grid
is available; this result does not justify changing the production default.
