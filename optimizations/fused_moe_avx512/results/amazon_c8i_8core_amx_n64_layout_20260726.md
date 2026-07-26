# Amazon C8i8 AMX N32 versus N64/K32 packed-B layout

## Decision

Do not switch the automatic AMX backend from N32 to N64. Keep N32 as the
production packed-weight layout and retain N64 only as an explicit experiment.

The N64 layout did what it was designed to do: `m1n4` can load the left and
right N32 tiles from one contiguous 4 KiB K32 chunk. That reduced neither the
one-thread end-to-end latency beyond run noise nor the cooperative N-split
latency. The automatic pattern was between -0.3% and +0.9% at one thread, but
N64 was 3.4%-9.3% slower at 8 threads for M64-2048.

## Layouts under test

- `x86_amx_bf16` (backend ID 102): the established N32-major layout. A complete
  K stream for one logical N32 block is contiguous.
- `x86_amx_bf16_n64` (backend ID 103): an N64 superblock composed of adjacent
  logical N32 blocks. Every K32 chunk is
  `[left K32xN32][right K32xN32]`, or 4 KiB total.

Both layouts use the same AMX JIT patterns, epilogues, cache-window policy,
route handling, and K32 padding. The N64 layout has a separate backend ID and
JIT cache key so packed objects can safely alternate in one process.

## Method

- Workspace: base commit `050fcd5` plus the uncommitted N64 experiment described
  here
- Machine: Amazon C8i, Intel Xeon 6975P-C, 8 physical cores/8 hardware threads
- Runtime: Linux 7.0.0-1006-aws, GCC 15.2.0, Python 3.12.13,
  PyTorch 2.13.0+cpu
- Build: `-DNDEBUG`; the compile command contains the toolchain's `-O3` and a
  later extension `-O2` (effective optimization level `-O2`); OpenMP, Xbyak,
  AVX-512 BF16, and AMX BF16 enabled
- Shape: one expert, top-k=1, H=4096, F=512, BF16
- Route counts: M=16, 32, 64, 128, 512, 2048 at one thread; M=64, 128, 512,
  2048 at eight threads
- Patterns: forced `m1n2`, `m2n2`, and `m1n4`
- Epilogue/policy controls: resident W13 SiLU, baseline W2 store,
  per-call tile state, automatic cache windows
- Measurement: both packed objects prepared before timing; candidates rotate
  order in one process after JIT warm-up. CPU calls are synchronous, so no
  additional synchronization is required.
- Statistics: one thread used 5 warm-ups and 21 measured runs; eight threads
  used 7 warm-ups and 31 measured runs. The benchmark emits median, p90, p99,
  best, mean, standard deviation, and median GFLOP/s.

Build:

```bash
FUSED_CPP_BUILD_MOE_ONLY=1 .venv/bin/python setup.py build_ext --inplace
```

One thread:

```bash
taskset -c 0 env PYTHONPATH=src \
  .venv/bin/python benchmarks/bench_amx_bf16_layouts.py \
  --routes 16,32,64,128,512,2048 \
  --patterns m1n2,m2n2,m1n4 \
  --threads 1 --warmup 5 --runs 21
```

Eight threads:

```bash
taskset -c 0-7 env PYTHONPATH=src \
  .venv/bin/python benchmarks/bench_amx_bf16_layouts.py \
  --routes 64,128,512,2048 \
  --patterns m1n2,m2n2,m1n4 \
  --threads 8 --warmup 7 --runs 31
```

Positive delta means N64 is faster. The tables show median latency.

## One-thread results

| M | pattern | N32 ms | N64 ms | N64 delta |
|---:|:---|---:|---:|---:|
| 16 | m1n2 | 0.493 | 0.494 | -0.2% |
| 16 | m2n2 | 0.493 | 0.507 | -2.8% |
| 16 | m1n4 | 0.477 | 0.489 | -2.3% |
| 32 | m1n2 | 0.693 | 0.701 | -1.1% |
| 32 | m2n2 | 0.569 | 0.569 | +0.0% |
| 32 | m1n4 | 0.668 | 0.667 | +0.1% |
| 64 | m1n2 | 1.006 | 1.009 | -0.3% |
| 64 | m2n2 | 0.877 | 0.870 | +0.9% |
| 64 | m1n4 | 0.948 | 0.954 | -0.6% |
| 128 | m1n2 | 1.707 | 1.746 | -2.2% |
| 128 | m2n2 | 1.654 | 1.657 | -0.1% |
| 128 | m1n4 | 1.558 | 1.562 | -0.2% |
| 512 | m1n2 | 5.443 | 5.596 | -2.7% |
| 512 | m2n2 | 5.024 | 5.075 | -1.0% |
| 512 | m1n4 | 4.938 | 4.946 | -0.1% |
| 2048 | m1n2 | 20.455 | 21.038 | -2.8% |
| 2048 | m2n2 | 18.713 | 18.916 | -1.1% |
| 2048 | m1n4 | 18.431 | 18.368 | +0.3% |

N64 prepack was also slightly slower in this run: 9.29 ms versus 8.99 ms.
Packing is one-time work and was excluded from inference latency.

## Eight-thread results

| M | pattern | N32 ms | N64 ms | N64 delta |
|---:|:---|---:|---:|---:|
| 64 | m1n2 | 0.181 | 0.190 | -4.6% |
| 64 | m2n2 | 0.168 | 0.174 | -3.4% |
| 64 | m1n4 | 0.182 | 0.183 | -0.4% |
| 128 | m1n2 | 0.286 | 0.302 | -5.4% |
| 128 | m2n2 | 0.294 | 0.302 | -2.6% |
| 128 | m1n4 | 0.276 | 0.289 | -4.5% |
| 512 | m1n2 | 0.838 | 0.916 | -8.5% |
| 512 | m2n2 | 0.851 | 0.872 | -2.4% |
| 512 | m1n4 | 0.793 | 0.874 | -9.3% |
| 2048 | m1n2 | 2.953 | 3.300 | -10.5% |
| 2048 | m2n2 | 2.828 | 2.912 | -2.9% |
| 2048 | m1n4 | 2.790 | 2.986 | -6.6% |

## Why N64 loses

`m1n4` is the only pattern that naturally consumes both N32 halves for every
K32 step. Its one-thread result shows that making the aggregate 4 KiB access
contiguous is not a material bottleneck at this shape.

`m1n2` and `m2n2` consume one N32 half at a time. Under N64 they either skip
the adjacent 2 KiB half or move from the right half to the next superblock,
creating a 4 KiB K-step stream plus parity handling instead of one contiguous
full-K N32 stream.

For the measured H4096/F512 shape, the eight-way N ranges are pair-aligned, so
the multi-thread regression is not explained by odd-range peeling. One
plausible mechanism is that every individual generated `TILELOADD` site moves
4 KiB between K32 chunks under N64 instead of 2 KiB under N32. Although the
four `m1n4` B tiles collectively cover each 4 KiB chunk, the hardware observes
separate instruction streams at page-sized strides; eight workers amplify any
prefetch or translation weakness. This needs counter validation and is not
claimed as proven causality.

Other dimensions can additionally split on an odd logical N32 boundary. Those
workers peel one `m1n2` block before entering `m1n4`, so they cannot fully use
the paired layout. The measured result already rejects N64 without relying on
that extra disadvantage.

## Correctness

The targeted test alternates N32 and N64 packed objects in one process for
`m1n2`, `m2n2`, and `m1n4`, M17/M33/M77, one/four threads, and H67/F35 tails:

```text
6 passed, 202 deselected
```

The complete x86 backend-dispatch and AVX-512/AMX regression files passed
after the final rebuild:

```text
211 passed, 4 skipped
```

The same six N64 cases also passed with `macro_m`, `tile_store`, and forced
weighted-direct overrides enabled together.

All measured H4096/F512 cases had maximum absolute error
`1.1920928955078125e-07` versus the PyTorch expert reference and zero maximum
absolute difference between N32 and N64 outputs.
