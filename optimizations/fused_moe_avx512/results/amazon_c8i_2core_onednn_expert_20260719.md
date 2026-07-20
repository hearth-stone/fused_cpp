# Amazon C8i 2-core oneDNN BF16 expert controls

Date: 2026-07-19. Host alias: `AmazonC8i2Cores`. oneDNN version: 3.14.0,
OpenMP runtime. The machine and custom AMX implementation are described in
[`amazon_c8i_2core_amx_20260719.md`](amazon_c8i_2core_amx_20260719.md).

## Implementations

`benchmarks/bench_onednn_bf16_expert.cpp` implements the dense single-expert
formula

```text
up = X @ Wup
gate = X @ Wgate
intermediate = SiLU(gate) * up
output = intermediate @ Wdown
```

entirely with public oneDNN primitives:

- `basic`: up matmul, gate matmul, BF16 swish, BF16 binary multiply, and down
  matmul: five timed primitives;
- `postop_fused`: up matmul, gate matmul with
  `eltwise_swish + binary_mul(up)` post-ops, and down matmul: three timed
  primitives.

All matmuls request any-format weights. The program reorders Wgate, Wup, and
Wdown into each primitive's preferred blocked layout before timing. Both
variants count `6*M*H*F` FLOPs and write BF16 output. A verbose run reported
`attr-post-ops:eltwise_swish:1+binary_mul` on the fused gate matmul, confirming
that no standalone activation or multiply primitive executes in that variant.

## Build and method

```bash
g++ -std=c++17 -O3 -DNDEBUG -Wall -Wextra \
  -I/home/ubuntu/zhangxu/onednn-install/include \
  benchmarks/bench_onednn_bf16_expert.cpp \
  -L/home/ubuntu/zhangxu/onednn-install/lib \
  -Wl,-rpath,/home/ubuntu/zhangxu/onednn-install/lib \
  -ldnnl -fopenmp -o benchmarks/bench_onednn_bf16_expert

OMP_NUM_THREADS=THREADS OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
  OMP_WAIT_POLICY=PASSIVE taskset -c CORES \
  benchmarks/bench_onednn_bf16_expert M 4096 512 5 21
```

Single-thread runs used CPU 0; two-thread runs used CPUs 0,1. Basic and fused
are warmed five times and alternate first position across 21 samples in one
process. Primitive creation, JIT setup, and weight reorders are excluded from
the timed region. `ONEDNN_MAX_CPU_ISA` was unset, so every matmul selected
`brg_matmul:avx10_1_512_amx`. The packed weights total 12 MiB for each variant.

## Correctness

The non-aligned M17/H131/F83 case computes an independent scalar FP32 expert:

| Comparison | Maximum absolute error | RMS error | Cosine similarity |
|---|---:|---:|---:|
| basic vs scalar | 1.325e-9 | 2.304e-10 | 0.999992965 |
| post-op fused vs scalar | 8.609e-10 | 1.741e-10 | 0.999995990 |
| post-op fused vs basic | 9.313e-10 | 2.332e-10 | 0.999992783 |

At H4096/F512 and M=16 through 2048, fused versus basic had maximum absolute
error `1.1921e-7` and cosine similarity at least `0.9999911`. The difference is
expected: basic rounds the gate and standalone activation to BF16, while the
post-op form applies swish before the gate matmul's BF16 destination rounding.

An AVX-512-only M64 smoke test selected `brg_matmul:avx512_core_bf16` for all
three matmuls and produced the same error bound.

## AMX performance

| M | Threads | basic | post-op fused | Fused/basic |
|---:|---:|---:|---:|---:|
| 16 | 1 | 0.5141 ms / 391.64 GFLOP/s | 0.5102 ms / 394.58 GFLOP/s | 1.007x |
| 16 | 2 | 0.2981 ms / 675.47 GFLOP/s | 0.2840 ms / 708.85 GFLOP/s | 1.049x |
| 64 | 1 | 1.3630 ms / 590.85 GFLOP/s | 1.4057 ms / 572.90 GFLOP/s | 0.970x |
| 64 | 2 | 0.8229 ms / 978.58 GFLOP/s | 0.7354 ms / 1095.13 GFLOP/s | 1.119x |
| 256 | 1 | 5.0040 ms / 643.73 GFLOP/s | 5.0377 ms / 639.42 GFLOP/s | 0.993x |
| 256 | 2 | 4.2588 ms / 756.37 GFLOP/s | 4.3375 ms / 742.65 GFLOP/s | 0.982x |
| 2048 | 1 | 35.8775 ms / 718.27 GFLOP/s | 35.3379 ms / 729.24 GFLOP/s | 1.015x |
| 2048 | 2 | 17.9498 ms / 1435.66 GFLOP/s | 17.6377 ms / 1461.07 GFLOP/s | 1.018x |

Reducing primitive count does not generally improve this shape. The separate
BF16 swish/multiply process only M*F elements and are small beside three
matmuls; adding two post-ops can slightly reduce the gate matmul's efficiency.
The M64 two-thread case is the exception, where eliminating two launches and
two intermediate passes improved latency by 11.9%. Other changes ranged from
-3.0% to +4.9%, with large M improving about 1.5%-1.8%.

With `ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16`, the M64 one-thread basic and fused
medians were 7.1820 and 7.1752 ms, a neutral 1.001x ratio.

## Relation to the custom fused expert

The oneDNN executable is a dense compute control: it does not include routing,
input gather, route placement, or weighted merge. The custom measurements do
include those operations and use polynomial SiLU, so the following separately
run figures are directional rather than an exact operator-boundary comparison.

At H4096/F512 with W13/W2 cache blocks 4/16, the custom AMX JIT was faster than
the oneDNN post-op control at M64 and M256. At M2048, oneDNN was faster: 35.34
vs 40.85 ms on one core and 17.64 vs 25.61 ms on two cores. This reinforces
the need for a large-M dispatch or improved custom cache/thread scheduling;
fusion alone does not overcome oneDNN's mature large-GEMM blocking.

## Conclusion

The public oneDNN fused form is valid and removes two primitives, but it is not
a universal optimization. Keep `basic` as the primary all-oneDNN baseline and
retain `postop_fused` as a measured experimental control. The result does not
justify changing any production dispatch.
