# Xbyak Pure GEMM vs i8mm BF16 GEMM

## Environment

- Date: 2026-07-25
- Workspace base: `58ad53e` plus the uncommitted `compute.xbyak_pure_gemm` change
- Host: Amazon C5, Neoverse-V3, 192 CPUs, two NUMA nodes
- Measurement CPU: `48` on NUMA0
- Kernel: Linux `7.0.0-1006-aws`
- Compiler: GCC `15.2.0`
- Build: `-O2 -march=armv8.6-a+sve+bf16+i8mm -msve-vector-bits=128`
- Inputs: packed BF16 A/B, row-major FP32 output, one thread
- Timing: same process and buffers, alternating JIT/i8mm order, 32 warmups
  and 201 measured samples; tables report median kernel time
- Reference: the matching M1/M2/M4/M8/M12 entry in
  `refs/i8gemm/lib/bf16gemm_sve.S`

## Warm W13 Shape

Command:

```bash
taskset -c 48 optimizations/fused_moe_sve/benchmarks/run_jit_vs_i8mm_pure_gemm.sh \
  --rows 1,2,4,8,12 --k 4096 --n 1024 --experts 1 --warmup 32 --runs 201
```

| M | JIT us | i8mm us | JIT time delta | JIT GFLOP/s | i8mm GFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 180.981 | 180.944 | +0.020% | 46.351 | 46.360 |
| 2 | 180.626 | 180.083 | +0.302% | 92.884 | 93.164 |
| 4 | 191.705 | 191.812 | -0.056% | 175.032 | 174.934 |
| 8 | 246.888 | 246.506 | +0.155% | 271.819 | 272.240 |
| 12 | 291.288 | 291.444 | -0.054% | 345.580 | 345.395 |

## Rotating Cold W13 Weights

The same shape used 64 separate packed weights, a 512 MiB aggregate B working
set, with 64 warmups and 201 measured samples.

| M | JIT us | i8mm us | JIT time delta | JIT GFLOP/s | i8mm GFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 232.547 | 233.633 | -0.465% | 36.073 | 35.905 |
| 2 | 230.185 | 233.660 | -1.487% | 72.886 | 71.802 |
| 4 | 240.304 | 238.596 | +0.716% | 139.633 | 140.633 |
| 8 | 289.311 | 289.918 | -0.209% | 231.961 | 231.475 |
| 12 | 354.134 | 355.358 | -0.344% | 284.252 | 283.273 |

## Warm W2 Shape

Command:

```bash
taskset -c 48 optimizations/fused_moe_sve/benchmarks/run_jit_vs_i8mm_pure_gemm.sh \
  --rows 1,2,4,8,12 --k 512 --n 4096 --experts 1 --warmup 32 --runs 201
```

| M | JIT us | i8mm us | JIT time delta | JIT GFLOP/s | i8mm GFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 75.667 | 76.618 | -1.241% | 55.431 | 54.743 |
| 2 | 75.773 | 76.891 | -1.454% | 110.707 | 109.097 |
| 4 | 86.189 | 86.471 | -0.326% | 194.656 | 194.021 |
| 8 | 119.644 | 120.477 | -0.691% | 280.452 | 278.513 |
| 12 | 156.084 | 156.186 | -0.065% | 322.465 | 322.255 |

## Result

All compared outputs were bitwise equal. Warm W13 time is within 0.31% of
i8mm, rotating-cold W13 time is within 1.49%, and warm W2 is 0.07-1.45%
faster. The standalone JIT therefore matches the corresponding i8mm BF16 GEMM
instruction schedules and measured performance for the tested canonical M
shapes.
