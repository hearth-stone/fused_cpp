# Amazon C8i 8-core AMX macro-M tile-state validation

Date: 2026-07-26

## Outcome

The explicit `macro_m` tile-state lifetime is correct, including exact-M
tails, but it does not improve the tested end-to-end fused expert. Keep
`auto` on `per_call` and retain `macro_m` as an experimental validation
override.

At M=2048, `macro_m` regressed median latency by 4.8% at one thread and 1.5%
at two threads. Four and eight threads were effectively tied: the first sweep
measured +0.5% and -0.3% throughput respectively, while the reversed-order
eight-thread repeat changed the sign to +0.1%.

## Workspace and machine

- Local workspace: branch `claude-wip`, commit `36b585c`, with the active
  uncommitted x86 MoE optimization work synchronized to the remote machine.
- Machine: `AmazonC8i8Cores`, Intel Xeon 6975P-C, 8 physical cores, one
  thread per core.
- ISA: `amx_bf16`, `amx_tile`, `amx_int8`, and `avx512_bf16`.
- Cache: 48 KiB L1d and 2 MiB L2 per core; 480 MiB shared L3 as reported by
  `lscpu`.
- OS: Linux `7.0.0-1006-aws`, x86-64.
- Compiler: GCC 15.2.0, release build with `-O3` and OpenMP.
- Runtime: Python 3.12.13, PyTorch 2.13.0+cpu.
- Extension:
  `/home/ubuntu/zhangxu/fused_cpp/src/fused_cpp/_moe_C.cpython-312-x86_64-linux-gnu.so`.
  The package also printed its unrelated general `_C` fallback warning, but
  the separate MoE extension imported successfully and reported
  `x86_amx_bf16`.

Build command:

```bash
FUSED_CPP_BUILD_MOE_ONLY=1 CMAKE_BUILD_PARALLEL_LEVEL=8 \
  .venv/bin/python setup.py build_ext --inplace
```

## Correctness

The validation covers `m1n2`, `m2n2`, and `m1n4`, full macro-M units followed
by non-tile-aligned M tails, and 1/2/4/8 worker threads. Separate cases cover
the direct-BF16 W2 output and expert-contiguous `tile_store` output.

```bash
OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
OMP_PLACES=cores OMP_WAIT_POLICY=PASSIVE \
FUSED_CPP_MOE_W13_SPLIT_N=1 taskset -c 0-7 \
env PYTHONPATH=src .venv/bin/python -m pytest -q \
tests/test_moe_avx512_bf16.py -k macro_m_tile_state
```

Result: `14 passed, 172 deselected`. Every `macro_m` result was BF16 bit-exact
with `per_call`; the largest absolute difference from the staged PyTorch
reference in the performance shape was `1.341104507446289e-07`.

The complete x86 MoE file then passed `186` tests. Running it together with
`tests/test_moe_backend_dispatch.py` passed `189` tests and skipped the four
backend cases that are unsupported on this host.

## Benchmark method

- Shape: E=1, top-k=1, hot routing, H=4096, F=512, M=2048, BF16.
- Automatic AMX pattern: `m1n4`.
- W13 N split: enabled with `FUSED_CPP_MOE_W13_SPLIT_N=1`.
- SiLU and W2 epilogues: automatic defaults.
- Inputs and packed weights: identical for both candidates in each process.
- Warm-up: 8 calls per candidate.
- Measurement: 101 samples per candidate, alternating the candidate order
  every iteration.
- Affinity: T threads bound to cores `0..T-1`; OpenMP dynamic teams disabled,
  close/core binding, passive wait.
- Throughput formula: `M * 6 * H * F / latency`.
- Baseline: `per_call`; variant: `macro_m`.

Command template:

```bash
OMP_NUM_THREADS=<T> OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
OMP_PLACES=cores OMP_WAIT_POLICY=PASSIVE \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
FUSED_CPP_MOE_W13_SPLIT_N=1 taskset -c <0..T-1> \
env PYTHONPATH=src .venv/bin/python \
benchmarks/bench_amx_bf16_patterns.py \
  --patterns auto --silu-epilogues auto --w2-epilogues auto \
  --tile-states per_call,macro_m \
  --tokens 2048 --hidden 4096 --intermediate 512 \
  --experts 1 --top-k 1 --routing hot --threads <T> \
  --warmup 8 --runs 101
```

## Primary rotated sweep

| Threads | `per_call` median | `macro_m` median | `per_call` GFLOP/s | `macro_m` GFLOP/s | `per_call / macro_m` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 28.530 ms | 29.891 ms | 903.2 | 862.1 | 0.954x |
| 2 | 17.945 ms | 18.212 ms | 1436.0 | 1415.0 | 0.985x |
| 4 | 10.836 ms | 10.787 ms | 2378.1 | 2388.9 | 1.005x |
| 8 | 9.416 ms | 9.445 ms | 2736.8 | 2728.3 | 0.997x |

Tail latency and dispersion:

| Threads | Candidate | p90 | p99 | Standard deviation |
| ---: | --- | ---: | ---: | ---: |
| 1 | `per_call` | 28.812 ms | 28.928 ms | 0.141 ms |
| 1 | `macro_m` | 30.210 ms | 30.549 ms | 0.189 ms |
| 2 | `per_call` | 18.178 ms | 18.404 ms | 0.153 ms |
| 2 | `macro_m` | 18.417 ms | 18.593 ms | 0.139 ms |
| 4 | `per_call` | 11.042 ms | 11.275 ms | 0.165 ms |
| 4 | `macro_m` | 10.988 ms | 11.246 ms | 0.157 ms |
| 8 | `per_call` | 9.589 ms | 9.692 ms | 0.161 ms |
| 8 | `macro_m` | 9.582 ms | 9.653 ms | 0.161 ms |

## Reversed-order repeat

The `--tile-states` list was reversed to `macro_m,per_call` while preserving
the per-iteration rotation.

| Threads | `per_call` median | `macro_m` median | `per_call / macro_m` |
| ---: | ---: | ---: | ---: |
| 1 | 28.557 ms | 29.587 ms | 0.965x |
| 8 | 9.443 ms | 9.432 ms | 1.001x |

The one-thread regression reproduced. The eight-thread result remained within
0.4% and changed sign, so it is noise-level rather than evidence for a default
switch.

## Decision

- Correctness validation is complete.
- `macro_m` remains a separately keyed, explicit experimental variant.
- `auto` remains `per_call`; no default dispatch behavior changes.
- The tested hypothesis is not supported end to end. The saved tile-state and
  call-frame work is too small to produce a stable gain at this shape, and the
  one-thread path has a repeatable regression. Determining the exact cause
  would require instruction/cycle counter work and is separate from this
  validation.
