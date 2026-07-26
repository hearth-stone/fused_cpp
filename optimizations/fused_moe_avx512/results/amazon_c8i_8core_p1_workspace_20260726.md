# Amazon C8i 8-core x86 MoE P1 workspace results

Date: 2026-07-26

## Scope and conclusion

This report closes the two remaining P1 workspace items:

1. weighted top-k=1 W2 output now applies the route scale in ZMM and writes
   BF16 caller output directly, eliminating the FP32 route workspace and merge;
2. gathered input now has a concurrency-safe, grow-only BF16 scratch pool,
   complementing the existing direct-input bypass and persistent intermediate.

Both changes are exact against their prior x86 paths. Weighted-direct is
automatically enabled only when it removes at least 256 KiB. Persistent input
is automatically enabled only for AMX when aggregate gathered-input scratch is
at least 256 KiB. Both retain `1` and `0` environment overrides for rotated A/B
testing.

For H=4096/F=512/M=2048, weighted-direct AMX speedups were 1.60x/1.86x/2.19x/
3.46x at 1/2/4/8 threads for one hot expert. Balanced E8 reached 1.60x/2.16x
at 1/8 threads. Persistent gathered input reached 1.05x/1.10x/1.13x/1.27x at
1/2/4/8 threads. The AVX-512 input-pool control was nearly neutral, so AVX-512
keeps transient input in automatic mode.

## Machine and methodology

- Host alias: `AmazonC8i8Cores`
- CPU: Intel Xeon 6975P-C, 8 vCPUs, one hardware thread per core
- Cache reported by `lscpu`: 48 KiB L1d and 2 MiB L2 per core, 480 MiB shared L3
- ISA: AVX-512 BF16, AMX-TILE, and AMX-BF16
- Affinity: `taskset -c 0-(threads-1)`
- OpenMP: `OMP_DYNAMIC=FALSE`, `OMP_PROC_BIND=close`,
  `OMP_PLACES=cores`, `OMP_WAIT_POLICY=PASSIVE`
- Shapes use BF16 input/weights with standard deviation 0.01 and a fixed
  20260726 seed. Weighted tests use non-unit sigmoid route weights.
- Packing, input generation, JIT warm-up, and output allocation are excluded.
  Packed weights and caller output are reused.
- Each pair runs in one process, alternates A/B order, warms each path 10 times,
  and records 31 samples per path.
- CPU frequency and thermal state were not locked. The tables therefore retain
  median, best, and p90 instead of relying on one timing.

The MoE-only extension was built with:

```bash
FUSED_CPP_BUILD_MOE_ONLY=1 MAX_JOBS=8 \
  .venv/bin/python setup.py build_ext --inplace
```

The package prints a warning that its unrelated monolithic `_C` extension is
unavailable on this x86 build. The benchmarked packed object reports
`x86_amx_bf16` or `x86_avx512_bf16`, confirming that the separate `_moe_C`
extension executes these measurements.

## Correctness

The final weighted-direct targeted run passed:

```text
12 passed, 190 deselected
```

Coverage includes AVX-512 intrinsic and JIT paths, AMX `m1n2`, `m2n2`, and
`m1n4`, one and eight threads, M/N/K tails, forced `tile_store` fallback,
top-k>1 fallback, and invalid environment values. Direct and workspace output
are BF16 bit-exact. The benchmark's maximum absolute difference from the
PyTorch reference was at most 1.1920929e-7.

Persistent-input tests cover AVX-512, AMX K32-tail clearing across pooled-buffer
reuse, one and eight threads, transient/persistent bit equality, and invalid
environment values. Before the final test-only `tile_store` addition, the full
x86 file passed 202 tests; x86 plus backend dispatch passed 205 with 4 skipped.

## Weighted top-1 direct BF16 output

Times are milliseconds. `workspace` is the prior FP32 route-buffer plus merge
path; `direct` scales the W2 row in ZMM and stores BF16 caller output. Each
M=2048 case removes 32 MiB of route workspace.

| backend / routing | M | T | workspace median / best / p90 | direct median / best / p90 | speedup |
|---|---:|---:|---:|---:|---:|
| AMX E1 hot | 1 | 1 | 0.460 / 0.440 / 0.470 | 0.453 / 0.438 / 0.475 | 1.016x |
| AMX E1 hot | 16 | 1 | 0.504 / 0.497 / 0.517 | 0.500 / 0.481 / 0.515 | 1.009x |
| AMX E1 hot | 2048 | 1 | 29.435 / 29.120 / 29.611 | 18.363 / 18.310 / 18.447 | 1.603x |
| AMX E1 hot | 2048 | 2 | 18.204 / 17.525 / 18.611 | 9.809 / 9.780 / 9.845 | 1.856x |
| AMX E1 hot | 2048 | 4 | 11.101 / 10.808 / 11.469 | 5.074 / 4.994 / 5.236 | 2.188x |
| AMX E1 hot | 2048 | 8 | 9.563 / 8.980 / 9.789 | 2.762 / 2.692 / 2.890 | 3.462x |
| AMX E8 balanced | 2048 | 1 | 37.570 / 36.708 / 38.080 | 23.430 / 23.211 / 23.558 | 1.603x |
| AMX E8 balanced | 2048 | 8 | 7.318 / 6.674 / 7.491 | 3.385 / 3.260 / 3.438 | 2.162x |
| AVX-512 E1 hot | 2048 | 1 | 246.677 / 243.908 / 249.768 | 234.757 / 233.981 / 237.152 | 1.051x |
| AVX-512 E1 hot | 2048 | 8 | 37.129 / 36.756 / 37.463 | 30.902 / 30.825 / 30.970 | 1.202x |

M=1 removes only 16 KiB and is effectively noise-level. M=16 removes exactly
256 KiB and was slightly positive in this repeat. The 256 KiB automatic
threshold keeps smaller calls on the established path while capturing the
large-M bandwidth and allocation savings. `skip_weighted=True` remains the
lower-overhead exact-one route; top-k>1 still needs reduction and therefore
retains the FP32 workspace.

Representative command:

```bash
OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
OMP_PLACES=cores OMP_WAIT_POLICY=PASSIVE \
PYTHONPATH=src taskset -c 0-7 \
  .venv/bin/python benchmarks/bench_x86_bf16_weighted_top1.py \
  --backend x86_amx_bf16 --tokens 2048 --hidden 4096 \
  --intermediate 512 --experts 1 --routing hot --threads 8 \
  --warmup 10 --runs 31
```

## Persistent gathered-input scratch

The benchmark uses balanced E2 routes so the aligned single-active-expert
input bypass cannot remove the gather. Persistent intermediate is forced on
for both variants to isolate input-scratch lifecycle.

| backend | M | T | transient median / best / p90 | persistent median / best / p90 | speedup |
|---|---:|---:|---:|---:|---:|
| AMX | 64 | 1 | 1.158 / 1.140 / 1.171 | 1.151 / 1.139 / 1.173 | 1.006x |
| AMX | 512 | 8 | 1.152 / 1.123 / 1.307 | 0.900 / 0.865 / 0.982 | 1.280x |
| AMX | 2048 | 1 | 21.126 / 21.087 / 21.162 | 20.141 / 20.100 / 20.180 | 1.049x |
| AMX | 2048 | 2 | 11.014 / 10.949 / 11.136 | 10.000 / 9.948 / 10.110 | 1.101x |
| AMX | 2048 | 4 | 6.345 / 6.239 / 6.398 | 5.626 / 5.510 / 5.678 | 1.128x |
| AMX | 2048 | 8 | 4.022 / 3.926 / 4.337 | 3.157 / 3.041 / 3.277 | 1.274x |
| AVX-512 | 2048 | 1 | 228.330 / 228.227 / 228.579 | 227.991 / 227.888 / 228.224 | 1.001x |
| AVX-512 | 2048 | 8 | 30.018 / 29.936 / 30.041 | 28.945 / 28.821 / 28.996 | 1.037x |

The M64/1T AMX threshold case is neutral, while larger or more concurrent AMX
calls benefit. AVX-512 has much longer compute phases, making allocation
removal a small fraction of latency, so its automatic mode remains transient.
The explicit override is retained for future CPU-specific policy calibration.

Representative command:

```bash
OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
OMP_PLACES=cores OMP_WAIT_POLICY=PASSIVE \
PYTHONPATH=src taskset -c 0-7 \
  .venv/bin/python benchmarks/bench_x86_bf16_input_scratch.py \
  --backend x86_amx_bf16 --tokens 2048 --hidden 4096 \
  --intermediate 512 --experts 2 --threads 8 \
  --warmup 10 --runs 31
```

## Final policy

- Weighted top-k=1 direct output: automatic when the eliminated FP32 route
  workspace is at least 256 KiB; force with
  `FUSED_CPP_MOE_X86_WEIGHTED_TOP1_DIRECT=1/0`.
- Gathered-input pool: automatic for AMX when aggregate scratch is at least
  256 KiB and the direct-input bypass is unavailable; force with
  `FUSED_CPP_MOE_X86_PERSISTENT_INPUT=1/0`.
- Intermediate pool: unchanged, automatic for AMX at 256 KiB.
- AMX per-kernel stack scratch remains pattern-specific: zero bytes when no
  vector epilogue scratch is needed, 2 KiB for two TMM results, and 4 KiB for
  four-result `m1n4` combined output.
- Top-k>1 and small weighted top-k=1 calls retain the prior workspace/merge
  baseline for correctness and low fixed overhead.
