# Amazon M5 192-Core Shared MLP, 2026-08-16

## Configuration

- Source: local working tree based on `b1b79b7`; shared-MLP changes were uncommitted during measurement.
- Machine: `AmazonM5192Cores`, 192 Neoverse V3 cores, SVE 128-bit.
- Placement: CPUs `0-191`, memory interleaved across NUMA nodes 0 and 1.
- Pages: ordinary pages; no HugeTLB environment was configured.
- Shape: `M=2048`, `H=7168`, `F=2048`; BF16 input, weights, intermediate, and output.
- Backend: `arm_sve_bf16`, `n_tile=8`, 192 native worker threads.
- Warmup/samples: 3 warmups per implementation, 11 paired samples with alternating order.
- Baseline: `fused_moe_bf16_tiled`, E=1, top-k=1, unit route weight, `skip_weighted=True`.
- Candidate: `shared_mlp_bf16_tiled` using the same packed E=1 weights.

Command:

```bash
PYTHONPATH=src FUSED_CPP_MOE_SVE=1 OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
numactl --physcpubind=0-191 --interleave=0,1 \
/mnt/develop/vllm/bin/python \
optimizations/fused_moe_sve/benchmarks/bench_shared_mlp.py \
--rows 2048 --hidden 7168 --intermediate 2048 --threads 192 \
--warmup 3 --runs 11
```

## Correctness

The benchmark checks bitwise equality against the E=1 baseline before timing.
The focused target suite also passed all seven selected shared-MLP and existing
vLLM-staged regression cases.

## Result

| Implementation | Median (ms) | Relative |
| --- | ---: | ---: |
| General E=1 fused MoE | 1309.744 | 1.00x |
| Standalone shared MLP | 24.230 | 54.06x |

The general baseline includes its normal routing and scheduling machinery; the
ratio is therefore an entrypoint-level result, not a claim that the GEMM kernel
itself became 54x faster.

With `FUSED_CPP_MOE_STAGE_TIMING=1`, the standalone operator selected 256 W13
tasks (`task_n=16`) and 896 W2 tasks (`task_n=8`). A representative warmed call
reported pack 0.598 ms, W13 12.426 ms, W2 10.136 ms, and E2E 23.356 ms. Both
stage queues contain more tasks than the 192 workers; M splitting remains the
second dimension when an N-window count alone is insufficient.
