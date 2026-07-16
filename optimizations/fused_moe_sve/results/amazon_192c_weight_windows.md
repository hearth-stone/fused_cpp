# Configurable packed-B weight windows on AmazonC5192Cores

Date: 2026-07-16

## Setup

- Host: `AmazonC5192Cores`, Neoverse-V3, 192 cores, 2 NUMA nodes.
- Affinity: NUMA0, cores `0-95`, local memory.
- Build: AArch64 SVE BF16 extension, project default `-O2`.
- Shape: TP4 expert, `H=4096`, `F=512`, `M=2040` per expert.
- Weights: distinct packed W13/W2 tensors for every simultaneously active expert.
- Operator: production async fused expert, split-W13 enabled, W2 owner-scatter enabled.
- Samples: 2 warmups and 7 measured calls per size; the size order was reversed
  in a second 5-sample run to check order sensitivity.

The primary command was:

```bash
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE FUSED_CPP_MOE_PREPACK_THREADS=96 \
PYTHONPATH=src numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_weight_windows.py \
  --experts E --threads-per-expert T --window-mib 0,4,2,1 \
  --routes 2040 --hidden 4096 --intermediate 512 --warmup 2 --runs 7
```

`0 MiB` is the legacy policy: two 4 MiB W13 ranges and one 4 MiB W2 range.
An explicit 4 MiB target produces the same ranges.

## Results

Median end-to-end operator time and aggregate useful GEMM throughput:

| Experts x threads | Window | W13/W2 ranges | Time (ms) | TFLOP/s |
| --- | ---: | ---: | ---: | ---: |
| 24 x 4 | legacy | 2 / 1 | 30.555 | 20.162 |
| 24 x 4 | 4 MiB | 2 / 1 | 30.554 | 20.163 |
| 24 x 4 | 2 MiB | 4 / 2 | 32.138 | 19.169 |
| 24 x 4 | 1 MiB | 8 / 4 | 36.728 | 16.774 |
| 48 x 2 | legacy | 2 / 1 | 85.958 | 14.334 |
| 48 x 2 | 4 MiB | 2 / 1 | 85.904 | 14.343 |
| 48 x 2 | 2 MiB | 4 / 2 | 61.717 | 19.964 |
| 48 x 2 | 1 MiB | 8 / 4 | 65.794 | 18.727 |
| 96 x 1 | legacy | 2 / 1 | 311.614 | 7.908 |
| 96 x 1 | 4 MiB | 2 / 1 | 310.605 | 7.934 |
| 96 x 1 | 2 MiB | 4 / 2 | 179.358 | 13.739 |
| 96 x 1 | 1 MiB | 8 / 4 | 133.123 | 18.511 |

The reverse-order run reproduced the selected medians within normal noise:

- `48 x 2`: 1/2/4/legacy MiB = 66.665/61.609/84.980/86.239 ms.
- `96 x 1`: 1/2/4/legacy MiB = 133.456/179.468/312.739/312.270 ms.

All window sizes produced bit-exact BF16 output in the benchmark's preflight
comparison. The focused normal/scheduled/async API tests also passed 8/8.

## Interpretation

The best measured window follows the team width in this experiment:

| Threads per expert | Best expert window | Window per active thread |
| ---: | ---: | ---: |
| 4 | 4 MiB | 1 MiB |
| 2 | 2 MiB | 1 MiB |
| 1 | 1 MiB | 1 MiB |

This is consistent with retaining about 1 MiB of reusable packed B per worker
inside the 2 MiB private L2 while leaving capacity for packed A and other live
state. It also keeps the nominal aggregate active B window near 96 MiB, so this
experiment does not independently identify private-L2 and aggregate-cache
effects.

The target is not monotonically beneficial. At `24 x 4`, reducing the window
below 4 MiB adds A rescans and range-dispatch overhead without relieving the
limiting cache pressure. At `48 x 2`, 2 MiB improves wall time by 28.2% (1.39x
speedup) over legacy; at `96 x 1`, 1 MiB improves wall time by 57.3% (2.34x).
Therefore the byte window should remain an explicit kernel/schedule dimension
until isolated and contention profiles include it; a global 1 MiB or 2 MiB
default would regress valid schedules.
