# Amazon C8i8 AMX N32 B-side load-hint experiment

## Decision

Use `TILELOADDT1` automatically for the retained N32 packed-B layout when one
expert has at least 128 routed rows. Keep `TILELOADD` below that threshold.
Do not enable the tested software-prefetch schedules.

The threshold is per expert, not the global token count. On the measured
H=4096/F=512 fused expert, `TILELOADDT1` was neutral or slower below M=128 and
improved median latency by 5.3%-9.7% from M=128 through M=2048 across
1/2/4/8 threads. Counters at M=512 show that it reduces L1 replacement,
pending-miss cycles, and top-down memory-bound slots without adding
instructions.

The earlier N64/K32 layout was already rejected and was intentionally excluded
from this experiment. Explicit non-default load hints reject N64 packed
weights, while N64's unchanged `auto` path remains `TILELOADD` only for
reproducibility of that retired layout experiment.

## Variants

- `tileloadd`: the previous baseline. A and B tiles use `TILELOADD`.
- `tileloaddt1`: A remains `TILELOADD`; only packed-B loads use
  `TILELOADDT1`.
- `prefetch_t0`: B still uses `TILELOADD`. While computing one N32 panel, the
  JIT prefetches every cache line needed by the next N32 panel to L1.
- `prefetch_t1`: the same next-panel schedule with L2-oriented
  `PREFETCHT1`.

Software prefetch is emitted only when another block exists in the current
cache window. M1N4 prefetches the next pair of N32 panels; M1N2 and M2N2
prefetch one panel. The load policy is part of the JIT cache key for both W13
and W2, so all modes can alternate safely in one process.

`FUSED_CPP_MOE_AMX_B_LOAD_HINT` accepts `auto`, `tileloadd`,
`tileloaddt1`, `prefetch_t0`, or `prefetch_t1`. The last four values are
validation overrides. `auto` implements the M=128 crossover.

## Method

- Workspace: base commit `050fcd5` plus the uncommitted N64 experiment and
  this load-hint change
- Machine: Amazon C8i, Intel Xeon 6975P-C, 8 physical cores/8 hardware threads
- Runtime: Linux 7.0.0-1006-aws, GCC 15.2.0, Python 3.12.13,
  PyTorch 2.13.0+cpu
- Build: `-DNDEBUG`; the compile command contains toolchain `-O3` followed by
  extension `-O2`; OpenMP, Xbyak, AVX-512 BF16, and AMX BF16 enabled
- Shape: one expert, top-k=1, H=4096, F=512, BF16, hot routing
- Route counts: M=32, 64, 128, 512, and 2048
- Threads/affinity: 1/2/4/8 threads pinned to cores `0`, `0-1`, `0-3`, and
  `0-7`
- Other automatic policy: automatic AMX pattern, resident SiLU, baseline W2
  store, per-call tile state, automatic W13/W2 cache windows
- Timing: variants rotate order in one process after correctness and JIT
  warm-up; five warm-ups and 31 measured runs per variant; weight prepack is
  outside timing
- Statistics: the benchmark emits median, p90, p99, best, mean, standard
  deviation, and median/best GFLOP/s

Build and rotated timing command:

```bash
FUSED_CPP_BUILD_MOE_ONLY=1 .venv/bin/python setup.py build_ext --inplace

taskset -c 0-7 env PYTHONPATH=src \
  .venv/bin/python benchmarks/bench_amx_bf16_patterns.py \
  --tokens 512 --hidden 4096 --intermediate 512 \
  --experts 1 --top-k 1 --routing hot --threads 8 \
  --patterns auto \
  --b-load-hints tileloadd,tileloaddt1,prefetch_t0,prefetch_t1 \
  --warmup 5 --runs 31
```

Positive speedup means `TILELOADDT1` is faster than `TILELOADD`.

## TILELOADDT1 crossover

| threads | M | TILELOADD ms | TILELOADDT1 ms | speedup |
|---:|---:|---:|---:|---:|
| 1 | 32 | 0.56768 | 0.58815 | -3.48% |
| 1 | 64 | 0.93653 | 0.90361 | +3.64% |
| 1 | 128 | 1.50891 | 1.43337 | +5.27% |
| 1 | 512 | 4.96879 | 4.62464 | +7.44% |
| 1 | 2048 | 18.71202 | 17.09676 | +9.45% |
| 2 | 32 | 0.31395 | 0.31460 | -0.21% |
| 2 | 64 | 0.48647 | 0.48561 | +0.18% |
| 2 | 128 | 0.82010 | 0.76972 | +6.54% |
| 2 | 512 | 2.50716 | 2.31488 | +8.31% |
| 2 | 2048 | 9.63328 | 8.84734 | +8.88% |
| 4 | 32 | 0.17206 | 0.17329 | -0.71% |
| 4 | 64 | 0.25717 | 0.26186 | -1.79% |
| 4 | 128 | 0.43978 | 0.41391 | +6.25% |
| 4 | 512 | 1.34553 | 1.26270 | +6.56% |
| 4 | 2048 | 5.07081 | 4.62366 | +9.67% |
| 8 | 32 | 0.08133 | 0.08117 | +0.19% |
| 8 | 64 | 0.13048 | 0.13181 | -1.01% |
| 8 | 128 | 0.24391 | 0.22994 | +6.07% |
| 8 | 512 | 0.86168 | 0.78792 | +9.36% |
| 8 | 2048 | 2.93618 | 2.74764 | +6.86% |

M=64 changes sign across thread counts and therefore remains on the
`TILELOADD` side of the automatic threshold. M=128 is the first tested route
count with a repeatable gain at every thread count.

An additional three-way dispatch check alternated `auto`, `tileloadd`, and
`tileloaddt1` in one process. At M=64, `auto` tracked `tileloadd`; at M=128 and
M=512 it tracked `tileloaddt1`, with zero output mismatches.

## Software-prefetch result

The table is a fresh M=512, 31-run rotation after the final build.

| threads | mode | median ms | p90 ms | best ms | speedup vs TILELOADD |
|---:|:---|---:|---:|---:|---:|
| 1 | `tileloadd` | 4.92709 | 4.95212 | 4.90951 | baseline |
| 1 | `tileloaddt1` | 4.52572 | 4.54056 | 4.50431 | +8.87% |
| 1 | `prefetch_t0` | 6.38944 | 6.40411 | 6.36615 | -22.88% |
| 1 | `prefetch_t1` | 6.13541 | 6.14944 | 6.12670 | -19.69% |
| 8 | `tileloadd` | 0.74799 | 0.88116 | 0.69474 | baseline |
| 8 | `tileloaddt1` | 0.69311 | 0.84316 | 0.65744 | +7.92% |
| 8 | `prefetch_t0` | 0.95564 | 1.02558 | 0.84936 | -21.73% |
| 8 | `prefetch_t1` | 0.87716 | 0.98894 | 0.84039 | -14.73% |

The packed B stream is regular enough for the hardware prefetcher. Covering
all next-panel cache lines adds too many front-end instructions and does not
reduce the observed pending-miss cost.

## Counter validation

The host has `kernel.perf_event_paranoid=4`, so counters were collected with
non-interactive `sudo perf stat`. Each mode used M=512, one pinned core, 10
warm-ups, 201 measured calls, five complete process repeats, and this
non-multiplexed event set:

```bash
sudo -n perf stat -x, -r 5 \
  -e cycles,instructions,l1d_pend_miss.pending_cycles,l1d.replacement,topdown.memory_bound_slots \
  taskset -c 0 env PYTHONPATH=src \
  .venv/bin/python benchmarks/bench_amx_bf16_patterns.py \
  --tokens 512 --threads 1 --patterns auto \
  --b-load-hints tileloaddt1 --warmup 10 --runs 201
```

Counts include identical process setup, reference construction, and warm-up
work. The long timed loop dominates, and the same methodology is used for all
variants.

| counter | TILELOADD | TILELOADDT1 | T1 change | PREFETCHT1 | prefetch change |
|:---|---:|---:|---:|---:|---:|
| cycles | 7,497,590,132 | 7,177,597,501 | -4.27% | 8,469,167,989 | +12.96% |
| instructions | 6,394,743,059 | 6,394,347,495 | -0.01% | 7,272,563,833 | +13.73% |
| L1D pending-miss cycles | 4,445,494,159 | 4,209,294,165 | -5.31% | 4,959,184,256 | +11.55% |
| L1D replacements | 1,802,361,925 | 358,244,013 | -80.12% | 1,806,510,418 | +0.23% |
| top-down memory-bound slots | 22,954,923,680 | 21,149,922,794 | -7.86% | 26,368,310,848 | +14.87% |

`TILELOADDT1` behaves as intended: the B stream no longer displaces nearly as
much near-cache state, and both pending-miss cycles and memory-bound slots
fall. The software-prefetch path instead adds 13.7% instructions and raises
the load-stall indicators.

## Correctness

The focused tests alternate all four hint modes after a baseline JIT entry for
`m1n2`, `m2n2`, and `m1n4`. H=129/F=65 exercises K/N tails and enough N32
blocks to execute the next-panel prefetch path. Every variant is BF16
bit-exact with `TILELOADD` and passes the PyTorch expert tolerance. Invalid
values fail explicitly.

The complete x86 backend-dispatch and AVX-512/AMX suite passed after the final
automatic-policy build:

```text
215 passed, 4 skipped
```

## Portability caveat

M=128 is calibrated for the Xeon 6975P-C and H=4096/F=512. Correctness does not
depend on that threshold, and explicit overrides preserve reproducibility, but
another CPU or materially different H/F can move the crossover. A future
dimension-aware policy should treat this threshold as a machine-tuned input,
not an ISA-wide constant.
