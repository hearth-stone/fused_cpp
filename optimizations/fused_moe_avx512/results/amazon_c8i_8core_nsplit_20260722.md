# Amazon C8i 8-core cooperative N-split results (2026-07-22)

## Outcome

The synchronous x86 executor now supports 1--256 requested workers and uses
the same disjoint-N ownership principle as the SVE executor. Balanced calls
with enough active experts retain the atomic expert queue. Underfilled calls
form route-weighted expert teams; strongly skewed calls use sorted waves so the
hot expert receives a wider team before the cold tail runs.

On the measured AMX H4096/F512/M2048 cases:

- one hot expert scales from 26.446 ms at 1T to 5.755 ms at 8T (4.60x);
- `[1536,256,256]` routes scale from 39.225 ms to 11.893 ms (3.30x);
- the 8T fused path is 3.41x and 1.99x faster than the corresponding staged
  PyTorch/oneDNN baseline.

AVX-512 also scales from 262.063 ms to 39.461 ms (6.64x) for the one-hot
M2048 case, although that kernel remains slower than oneDNN in absolute time.

## Machine and software

- host alias: `AmazonC8i8Cores`
- CPU: Intel Xeon 6975P-C under KVM
- topology: 1 socket, 8 physical cores, 1 thread/core, CPUs `0-7`
- cache reported by `lscpu`: 48 KiB L1d/core, 2 MiB L2/core, 480 MiB shared L3
- ISA: AVX-512 BF16, AMX-TILE, AMX-BF16
- PyTorch: 2.8.0+cpu
- PyTorch oneDNN: v3.7.1
- build: GCC 13.3, C++17, OpenMP 4.5
- workspace: `/home/ubuntu/zhangxu/fused_cpp`

The unrelated main `_C` extension is not buildable on this x86 checkout due
to ARM-only SDPA sources. Measurements use the independently built
`fused_cpp._moe_C`; the benchmark's `packed.backend_name` confirms
`x86_amx_bf16` or `x86_avx512_bf16`. The package-level warning about a PyTorch
fallback refers to the unavailable main extension, not the MoE kernel.

Build command:

```bash
FUSED_CPP_BUILD_MOE_ONLY=1 MAX_JOBS=8 \
  .venv/bin/python setup.py build_ext --inplace
```

## Correctness

```bash
OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
OMP_PLACES=cores MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
FUSED_CPP_MOE_W13_SPLIT_N=1 \
taskset -c 0-7 env PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_moe_avx512_bf16.py
```

Result: `163 passed, 1 warning in 0.61s`. The warning is the environment's
missing optional NumPy package.

Coverage added for this policy includes:

- AVX-512 and AMX `m1n2`, `m2n2`, and `m1n4`;
- 1/2/4/8 workers, non-multiple H/F/K/N tails, and cache-block windows;
- one active expert, fewer active experts than workers, eight balanced active
  experts, and a strongly skewed wave with eight active experts;
- exact BF16 equality between 1T and 8T, plus tolerance against the naive
  reference;
- barrier cancellation when a generated stage reports an invalid runtime
  cache policy, preventing peers from deadlocking.

## Method

All timings pin the process to physical CPUs `0-7` and set:

```text
OMP_NUM_THREADS=8
OMP_DYNAMIC=FALSE
OMP_PROC_BIND=close
OMP_PLACES=cores
OMP_WAIT_POLICY=PASSIVE
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
FUSED_CPP_MOE_W13_SPLIT_N=1
```

The benchmark uses BF16, H=4096, F=512, top-k=1, AMX auto pattern/cache
policy, 8 warmups, 31 custom samples, and 11 oneDNN samples. Weight prepack and
JIT warmup are excluded. `fused_moe_naive` is the staged PyTorch/oneDNN
baseline and is assigned the same requested thread count as the custom path.
Reported FLOP/s uses `routes * 6 * H * F`.

Representative command:

```bash
OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
OMP_PLACES=cores OMP_WAIT_POLICY=PASSIVE MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 FUSED_CPP_MOE_W13_SPLIT_N=1 \
taskset -c 0-7 env PYTHONPATH=src .venv/bin/python \
  tests/bench_moe_avx512_bf16.py \
  --backend x86_amx_bf16 --tokens 2048 --hidden 4096 \
  --intermediate 512 --experts 3 --top-k 1 --routing skewed \
  --threads 8 --warmup 8 --runs 31 --baseline-runs 11
```

## AMX results

### One hot expert, M=2048

| workers | fused median ms | fused best ms | fused median GFLOP/s | oneDNN median ms | fused speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 26.445894 | 25.905806 | 974.4 | 82.227594 | 3.109x |
| 2 | 14.208511 | 13.973877 | 1813.7 | 49.511616 | 3.485x |
| 4 | 7.917323 | 7.868901 | 3254.9 | 27.420140 | 3.463x |
| 8 | 5.755011 | 4.872927 | 4477.8 | 19.611457 | 3.408x |

The final 8T median is 4.60x faster than 1T. The gap from ideal 8x scaling is
consistent with parallel gather, two phase barriers, merge, memory/cache
traffic, OpenMP entry, and cloud-frequency variation.

### Strongly skewed routes, `[1536,256,256]`

| workers | fused median ms | fused best ms | fused median GFLOP/s | oneDNN median ms | fused speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 39.224863 | 38.581726 | 657.0 | 89.864066 | 2.291x |
| 2 | 23.310553 | 22.950778 | 1105.5 | 57.718277 | 2.476x |
| 4 | 14.586680 | 13.887443 | 1766.7 | 35.358730 | 2.424x |
| 8 | 11.893386 | 11.082108 | 2166.7 | 23.639623 | 1.988x |

The first simple prototype ran all active experts concurrently and gave the
hot expert too few workers; its 8T median was about 21.54 ms. Sorted waves plus
right-sized reusable scratch reduce the final median to 11.89 ms. This was a
cross-run prototype comparison, not a rotated same-process A/B, so it supports
the dispatch choice rather than a precise percentage claim.

### Small hot and balanced controls

A separate small-M sweep used 12 warmups and 51 samples with the oneDNN
baseline disabled. Even one routed row still contains enough H4096/F512 N work
to benefit from the team; no small-M width cap was needed on this machine.

| hot M | 1T median ms | 8T median ms | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.461119 | 0.061564 | 7.49x |
| 4 | 0.476457 | 0.172913 | 2.76x |
| 16 | 0.518603 | 0.105990 | 4.89x |
| 48 | 0.805754 | 0.218121 | 3.69x |
| 64 | 1.024328 | 0.256546 | 3.99x |

The following controls return to the primary 8-warmup/31-sample method and
include the oneDNN baseline:

| shape/routing | workers | fused median ms | fused median GFLOP/s | oneDNN median ms | fused speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| M256, E1 hot | 1 | 3.573765 | 901.4 | 8.503480 | 2.379x |
| M256, E1 hot | 8 | 0.855311 | 3766.1 | 2.425455 | 2.836x |
| M2048, E2 `[1024,1024]` | 8 | 9.182864 | 2806.3 | 21.747238 | 2.368x |
| M2048, E8 `[256,...,256]` | 8 | 8.708844 | 2959.0 | 23.881680 | 2.742x |

E2 is underfilled and therefore uses two cooperative teams. E8 balanced has
one expert per worker and retains the atomic queue. A prototype that forced
every call through sorted waves regressed balanced controls, so the final
policy gates waves on both `largest >= 64` and `largest >= 2 * second`.

## AVX-512 control

One hot expert, H4096/F512/M2048:

| workers | fused median ms | fused best ms | fused median GFLOP/s | oneDNN median ms | fused speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 262.063017 | 259.619072 | 98.3 | 73.597691 | 0.281x |
| 8 | 39.461304 | 38.531158 | 653.0 | 15.846656 | 0.402x |

This is a useful separation of concerns: cooperative N splitting scales the
existing AVX-512 kernel by 6.64x, but does not repair its lower single-core
microkernel throughput relative to oneDNN.

## Dispatch and portability limits

- `num_threads` is a requested width, not automatic CPU discovery. Pin and
  request no more workers than available physical cores.
- 64 rows/thread and the 2x skew threshold are deterministic C8i-calibrated
  heuristics, not ISA constants.
- Every wave enters a parallel region; a persistent x86 worker pool remains a
  possible follow-up for very short experts or many waves.
- The benchmark did not lock CPU frequency. Medians are primary; unusually
  fast best samples, especially the one-hot 8T case, should not be treated as
  sustained throughput.
- Different H/F, top-k, expert count, sockets, or cache topology require new
  held-out routing/thread measurements before changing the automatic policy.
