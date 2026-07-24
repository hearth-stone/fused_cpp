# Single-core exact-M compute and packed-B efficiency

Date: 2026-07-23

## Configuration

- Host: `AmazonC5192Cores`, Neoverse-V3, NUMA0 CPU 48.
- Kernel: production SVE JIT exact-M, split-W13, one expert and one thread.
- Shape: H=4096, F=512, W13=8 MiB and W2=4 MiB.
- Weight policy: 64 distinct experts rotate between samples. The 768 MiB
  packed-weight population exceeds the NUMA-local 96 MiB LLC.
- Timing: W13 fused-SiLU/packC plus W2 packed GEMM stages only. Gather, scatter,
  scheduling, allocation, and trace-file writes are outside the reported
  stage interval.
- Samples: 192 cold-weight calls per M and two independent full sweeps.
- Pages: ordinary 4 KiB pages. The host THP policy is `madvise`, and the
  standalone read ceiling explicitly uses `MADV_NOHUGEPAGE`.

The single-core BFMMLA reference peak is 403.8 GFLOP/s. For this shape, useful
expert work and packed-B traffic are

```text
useful_flops = 6 * M * H * F
packed_B_bytes = 6 * H * F = 12 MiB
compute_rows = 2 * ceil(M / 2)
```

The table uses three distinct efficiencies:

```text
lane_efficiency = M / compute_rows
useful_compute_efficiency = useful_GFLOP/s / 403.8
physical_issue_efficiency = padded_physical_GFLOP/s / 403.8
memory_efficiency = effective_packed_B_GB/s / 40.04
```

`memory_efficiency` is an effective CPU packed-B service ratio, not a PMU
memory-controller byte ratio.

## Single-core SVE read ceiling

The standalone reader executes only eight-way-unrolled SVE `LD1H`: no BFMMLA,
output store, reduction, or software prefetch. It measures both a continuous
768 MiB traversal and the production-like case that rotates 64 distinct
12 MiB chunks in a coprime order.

```bash
make -C optimizations/fused_moe_sve/benchmarks bench_single_core_sve_read
numactl --cpunodebind=0 --membind=0 taskset -c 48 \
  optimizations/fused_moe_sve/benchmarks/bench_single_core_sve_read
```

| Run | Continuous 768 MiB | Rotating 64x12 MiB |
| ---: | ---: | ---: |
| 1 | 41.250 GB/s | 40.051 GB/s |
| 2 | 41.223 GB/s | 40.035 GB/s |
| 3, Makefile target | 41.228 GB/s | 40.006 GB/s |
| Median of run medians | 41.228 GB/s | 40.035 GB/s |

The calibrated production-like single-core packed-B read ceiling is therefore
40.04 GB/s. The M2 fused kernel's 36.49 GB/s is 91.1% of this ceiling, not the
ceiling itself.

## Exact-M sweep

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  numactl --cpunodebind=0 --membind=0 taskset -c 48 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_single_expert_m1_12_efficiency.py
```

The table reports the final fixed-ceiling sweep; the two preceding independent
sweeps differed by less than 1% at every point.

| M | GEMM ms | Lane efficiency | Useful GFLOP/s | Useful compute efficiency | Physical issue efficiency | Packed-B GB/s | Memory efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3472 | 50.00% | 36.24 | 8.97% | 17.95% | 36.239 | 90.51% |
| 2 | 0.3447 | 100.00% | 73.00 | 18.08% | 18.08% | 36.500 | 91.16% |
| 3 | 0.3585 | 75.00% | 105.31 | 26.08% | 34.77% | 35.102 | 87.67% |
| 4 | 0.3580 | 100.00% | 140.58 | 34.81% | 34.81% | 35.144 | 87.77% |
| 5 | 0.3861 | 83.33% | 162.93 | 40.35% | 48.42% | 32.586 | 81.38% |
| 6 | 0.3855 | 100.00% | 195.86 | 48.50% | 48.50% | 32.643 | 81.53% |
| 7 | 0.4336 | 87.50% | 203.12 | 50.30% | 57.49% | 29.018 | 72.47% |
| 8 | 0.4336 | 100.00% | 232.16 | 57.49% | 57.49% | 29.020 | 72.48% |
| 9 | 0.4723 | 90.00% | 239.79 | 59.38% | 65.98% | 26.643 | 66.54% |
| 10 | 0.4723 | 100.00% | 266.42 | 65.98% | 65.98% | 26.642 | 66.54% |
| 11 | 0.5245 | 91.67% | 263.88 | 65.35% | 71.29% | 23.989 | 59.91% |
| 12 | 0.5245 | 100.00% | 287.90 | 71.30% | 71.30% | 23.992 | 59.92% |

Adjacent odd/even M values have nearly identical stage time and physical issue
efficiency because they execute the same number of BFMMLA row pairs. The odd
point loses only the padded second row's useful work. As M increases, the fixed
12 MiB weight scan is amortized across more rows: useful compute efficiency
rises, while effective packed-B GB/s falls because BFMMLA and the fused
epilogue occupy more of the same stage interval.

This calibration does not change the production planner or its contention
tables. It supplies a single-core packed-B ceiling for future analytic
compute/memory decomposition.
