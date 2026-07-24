# Amazon 192C Pure W13 GEMM, M1/M2

## Scope

This benchmark isolates the production exact-M SVE JIT GEMM on the W13 shape:

- `M=1/2`, `K=4096`, `N=1024`
- one NUMA0 core (`CPU48`)
- 64 rotating packed W13 experts (8 MiB per expert)
- packed A and all output allocation are outside the timed region
- `Operation::kW2` supplies the plain FP32 GEMM store
- no SiLU, gate/up multiply, BF16 conversion, pack-C, W2, routing, or scatter
- one-chunk K path (`FUSED_CPP_MOE_KC=0`)

The use of `Operation::kW2` changes only the epilogue. Its packed-A/B loads,
double-buffered K-loop, and BFMMLA instructions are the same as the W13 JIT
kernel.

## Results

Two independent process runs are shown because single-core frequency state
produces about 1% run-to-run variation.

| N ranges | M | Median range, ms | Physical GFLOP/s range | Weight GB/s range | Read-ceiling efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.2341-0.2365 | 70.93-71.66 | 35.46-35.83 | 88.6-89.5% |
| 1 | 2 | 0.2310-0.2342 | 71.63-72.62 | 35.81-36.31 | 89.4-90.7% |
| 2 | 1 | 0.2299-0.2325 | 72.17-72.96 | 36.08-36.48 | 90.1-91.1% |
| 2 | 2 | 0.2292-0.2321 | 72.28-73.21 | 36.14-36.61 | 90.3-91.4% |

The calibrated single-core rotating packed-B ceiling is 40.04 GB/s. The
production two-range policy therefore leaves roughly 9% in the GEMM itself.
Removing the fused epilogue does not raise M2 above the complete fused-stage
measurement of 36.50 GB/s, so SiLU and pack-C do not explain that gap.

M1 and M2 have nearly identical physical throughput because both execute the
same two-row kernel. M1 retains only one useful row and therefore has 50%
useful-lane efficiency even though its physical GEMM and weight-read rates are
close to M2.

## Reproduction

```bash
OMP_NUM_THREADS=1 \
FUSED_CPP_MOE_SVE_IMPL=jit \
FUSED_CPP_MOE_KC=0 \
numactl --cpunodebind=0 --membind=0 \
taskset -c 48 \
.venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_pure_w13_gemm_m1_m2.py
```
