# Amazon C8i 2-core AMX BF16 fused-expert results

Date: 2026-07-19. Host alias: `AmazonC8i2Cores`. The host exposed two physical
Intel Xeon 6975P-C cores, 48 KiB L1D and 2 MiB L2 per core, and a shared
240 MiB L3 slice. Linux was 7.0.0-1006-aws, GCC was 15.2.0, PyTorch was
2.8.0+cpu, and the standalone oneDNN control used version 3.14. Processes were
pinned to core 0 or cores 0,1. Weight preparation was outside timed regions.
The workspace was based on commit `c9bfb94` plus the uncommitted AMX changes
described here.

The extension build compiled ordinary MoE translation units with `-O2`; the
preserved AVX-512 intrinsic translation unit used `-O3`, C++17, AVX-512 F/BW/VL
and BF16, plus FMA. Xbyak 7.37 supplied header-only runtime code generation.
The standalone controls used the same ISA flags plus `-DNDEBUG`,
`-DFUSED_CPP_MOE_HAS_XBYAK=1`, and
`-DFUSED_CPP_MOE_HAS_X86_AVX512_BF16=1`; the W2 control linked the prebuilt
oneDNN shared library. No `-mamx-*` compiler flag is needed because Xbyak emits
the tile instructions directly.

## Implementation and correctness

The explicit `x86_amx_bf16` backend uses ID 102; `auto` continues to select
backend 101 `x86_avx512_bf16`. AMX reuses the K-pair/N32 packed-weight layout,
pads K to 32, and uses row-major M16 input and intermediate panels. Xbyak emits
exact-M1..16 W13 and W2 kernels. Each worker obtains Linux XTILEDATA permission
before executing `LDTILECFG`, `TILELOADD`, `TDPBF16PS`, `TILESTORED`, and
`TILERELEASE`.

```bash
MAX_JOBS=2 FUSED_CPP_BUILD_MOE_ONLY=1 \
  .venv/bin/python setup.py build_ext --inplace
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_moe_avx512_bf16.py
```

After adding the tile-pattern coverage, the x86 file passed all 72 cases. The
combined dispatch/x86 run passed 75 with four architecture skips; earlier
forced-AVX-JIT and intrinsic gates also passed before the AMX-only additions.
AMX coverage includes exact M=1 through 17, M16+tail composition, M through
2048 for all three schedules, H/F/N tails, polynomial degrees 4/5/6, one and
two threads, weighted routing/merge, direct BF16 output, output buffers,
explicit metadata, default preservation, and the AMX kill switch. Seven
selected AMX one/two-thread cases also passed with `OMP_THREAD_LIMIT=1`,
exercising the standard-thread fallback. Compiling the backend and JIT
translation units with the Xbyak macro absent passed a separate syntax-only
build.

Representative full-operator maximum absolute error against PyTorch was at
most `1.19e-7`; the standalone W2 maximum absolute error against oneDNN was
`1.40e-9` and was zero for every AMX-selected oneDNN shape. A whole-file
`OMP_THREAD_LIMIT=1` stress run remains unsuitable as an all-backend gate on
this host: two legacy AVX M48 cases fail inside the staged PyTorch/oneDNN
reference, including when AVX is forced to its preserved intrinsic path. The
same cases pass without that process-wide limit, and the targeted AMX fallback
cases pass with it.

## Standalone W2

The command used 10 warmups and 51 samples on core 0. oneDNN selected
`brg_matmul:avx10_1_512_amx` except at M1, where it selected AVX-512 BF16.

```bash
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE taskset -c 0 \
  benchmarks/bench_amx_bf16_gemm M 512 4096 10 51
```

| M | Custom AMX | oneDNN | Custom/oneDNN | JIT code / generation |
|---:|---:|---:|---:|---:|
| 1 | 30.12 GFLOP/s | 30.01 GFLOP/s | 1.00x | 364 B / 0.029 ms |
| 4 | 118.95 GFLOP/s | 99.67 GFLOP/s | 1.19x | 487 B / 0.031 ms |
| 8 | 232.93 GFLOP/s | 193.57 GFLOP/s | 1.20x | 651 B / 0.031 ms |
| 12 | 312.03 GFLOP/s | 278.25 GFLOP/s | 1.12x | 815 B / 0.034 ms |
| 16 | 408.27 GFLOP/s | 346.78 GFLOP/s | 1.18x | 979 B / 0.035 ms |
| 17 | 232.60 GFLOP/s | 368.65 GFLOP/s | 0.63x | 1343 B / 0.045 ms |
| 48 | 390.53 GFLOP/s | 336.00 GFLOP/s | 1.16x | 979 B / 0.034 ms |

Packing row-major A in the timed region changed M16 from 408.27 to
406.78 GFLOP/s. M17 is the known discontinuity: the first implementation runs
an efficient M16 panel and then a separate M1 panel, loading the full B range
twice. A future two-M-tile kernel can share each B tile across M16 plus tail.
The final-build M16 rerun measured 408.10 GFLOP/s versus oneDNN's 346.86
GFLOP/s, with zero maximum absolute difference.

## Fused W13

This benchmark includes gate and up GEMMs, both tile stores, degree-5
SiLU-times-up, BF16 conversion, and row-major intermediate output.

```bash
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE taskset -c 0 \
  benchmarks/bench_amx_bf16_w13 M 4096 512 10 51 5
FUSED_CPP_MOE_AVX512_IMPL=jit OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE \
  taskset -c 0 benchmarks/bench_avx512_bf16_w13 M 4096 512 10 51 5
```

| M | AMX JIT | AVX-512 JIT | AMX/AVX-512 |
|---:|---:|---:|---:|
| 12 | 328.67 GFLOP/s | 116.79 GFLOP/s | 2.81x |
| 16 | 401.16 GFLOP/s | 116.40 GFLOP/s | 3.45x |
| 48 | 393.43 GFLOP/s | 117.56 GFLOP/s | 3.35x |

AMX code generation took 0.07-0.09 ms and produced 3.0-4.0 KiB per exact-M
W13 specialization. The final-build M16 rerun reached 402.70 GFLOP/s.

## Full fused expert

The benchmark counts W1, W3, and W2 GEMM FLOPs as `routes * 6 * H * F` and
uses 5 warmups plus 31 custom samples; the staged PyTorch/oneDNN baseline uses
2 warmups plus 7 samples. oneDNN keeps its default AMX selection.
`OMP_WAIT_POLICY=PASSIVE` was used for the two-core runs. AMX and AVX-512 were
run as separate processes with otherwise identical arguments:

```bash
OMP_NUM_THREADS=THREADS OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE \
  taskset -c CORES env PYTHONPATH=src .venv/bin/python \
  tests/bench_moe_avx512_bf16.py --backend BACKEND \
  --tokens TOKENS --hidden 4096 --intermediate 512 \
  --experts EXPERTS --top-k TOP_K --routing ROUTING --threads THREADS \
  --warmup 5 --runs 31 --baseline-runs 7
```

`BACKEND` was `x86_amx_bf16` or `x86_avx512_bf16`; balanced rows used
`TOKENS=16`, `EXPERTS=8`, `TOP_K=6`, and hot rows used the displayed M with
`EXPERTS=1`, `TOP_K=1`. `CORES` was `0` for one thread and `0,1` for two.

| Shape | Threads | AMX fused | AVX-512 fused | AMX/AVX-512 | AMX/staged oneDNN |
|---|---:|---:|---:|---:|---:|
| hot M12, H4096, F512 | 1 | 0.5112 ms, 295.40 GFLOP/s | 1.4044 ms, 107.52 GFLOP/s | 2.75x | 2.14x |
| hot M16, H4096, F512 | 1 | 0.5200 ms, 387.14 GFLOP/s | 1.8712 ms, 107.59 GFLOP/s | 3.60x | 2.56x |
| hot M48, H4096, F512 | 1 | 1.5327 ms, 394.06 GFLOP/s | 5.6110 ms, 107.64 GFLOP/s | 3.66x | 1.52x |
| balanced 96 routes, E8, top-k6 | 1 | 5.7112 ms, 211.51 GFLOP/s | 15.0481 ms, 80.27 GFLOP/s | 2.64x | 2.10x |
| balanced 96 routes, E8, top-k6 | 2 | 2.2516 ms, 536.50 GFLOP/s | 7.4864 ms, 161.35 GFLOP/s | 3.32x | 2.80x |
| hot M48, N-split | 2 | 0.8239 ms, 733.08 GFLOP/s | 3.0324 ms, 199.17 GFLOP/s | 3.68x | 2.00x |

The final balanced two-core AMX best sample reached 548.51 GFLOP/s; hot M48
reached 772.67 GFLOP/s in the shape sweep. The default backend did not change,
so these gains require
explicit `backend="x86_amx_bf16"` preparation.

The later `2M x 2N` and `1M x 4N` schedule comparison, including M=2048 and
balanced/skewed/hot route distributions, is recorded in
[`amazon_c8i_2core_amx_patterns_20260719.md`](amazon_c8i_2core_amx_patterns_20260719.md).
