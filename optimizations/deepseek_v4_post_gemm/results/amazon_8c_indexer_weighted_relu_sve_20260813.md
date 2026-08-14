# Amazon 8C DeepSeek V4 Indexer Weighted-ReLU SVE 8x2VL

## Outcome

The scalable-SVE candidate is numerically correct on the requested Amazon
8-core machine and its generated 8x2VL hot kernel has no SVE register spills.
For the production score shape `M=2048, H=64, D=128, N=1536`, the isolated
kernel reaches 1.406 TFLOP/s on 8 cores. In the full post-GEMM stage, score
generation takes 36.154 ms while the unchanged row-wise TopK takes 75.155 ms;
TopK is now the larger long-path cost.

This is not reported as a speedup over the retired weighted-Q fold: that path
computed a different, incorrect formula. A comparable previous correct native
baseline does not exist.

## Machine and build

- Host: `AmazonECS8Cores`
- CPU: 8 physical Neoverse-V1 cores, CPUs `0-7`, one NUMA node
- Cache: 64 KiB L1d/core, 1 MiB L2/core, 32 MiB shared L3
- Runtime SVE vector length: 32 bytes / 256 bits
- Indexer tile: `8 heads x 16 keys` (`n_tile=16`)
- Main extension flags: `-O2 -fopenmp -march=armv8.6-a+sve+bf16+i8mm`
- Affinity: `taskset -c 0-7`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`,
  `OMP_DYNAMIC=FALSE`
- The unrelated MoE extension was rebuilt with
  `FUSED_CPP_SVE_VECTOR_BITS=256` so the package could be imported at SVL256.

## Correctness and assembly

The standalone score checker compared FP32 scalar reference scores with the
SVE kernel for all 45 combinations of:

- `H={8,16,64}`
- `D={4,16,128}`
- `N={1,7,16,19,33}`, including partial 2VL tiles

Result: maximum absolute error `9.53674e-7`, maximum relative error
`2.6939e-6`. Padded packed-B columns produced zero scores.

Integration tests:

```text
25 passed, 5 skipped in 0.54s
```

The five skips are the intentionally retired standalone `cpp_v0` tests, not
failures of the post-GEMM SVE path.

`objdump` of `weighted_relu_tile` (`0x1c0-0x4d0` in the object) shows 16
BFMMLA instructions in the K-loop body, register ReLU/weight FMA, register
TRN/UZP head reduction, and final scatter stores. There are no `z` loads or
stores to the stack; only the normal ABI save/restore of scalar `d8/d9` occurs.

Commands:

```bash
PYTHONPATH=src OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
OMP_PROC_BIND=close OMP_PLACES=cores taskset -c 0-7 \
.venv/bin/python -m pytest -q \
tests/test_deepseek_v4_post_gemm_stage.py tests/test_sparse_attn_indexer.py
```

The standalone checker/benchmark is reproducible with:

```bash
g++ -std=c++20 -O2 -fopenmp -march=armv8.6-a+sve+bf16+i8mm \
  -Icsrc benchmarks/bench_deepseek_v4_indexer_sve.cpp \
  csrc/deepseek_v4_indexer_sve.cpp \
  -o /tmp/bench_deepseek_v4_indexer_sve

OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores \
  taskset -c 0 /tmp/bench_deepseek_v4_indexer_sve
OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores \
  taskset -c 0-7 /tmp/bench_deepseek_v4_indexer_sve
```

## Isolated score performance

Shape: BF16 Q/K, FP32 weights, `M=2048, H=64, D=128, N=1536`; 51.540 GFLOP
from BF16 dot products. Each row packs Q once; packed K is reused. Three
warmups and 15 measured runs were used.

| Threads / affinity | Median | Min | Max | Throughput |
|---|---:|---:|---:|---:|
| 1T / core 0 | 202.285 ms | 202.195 ms | 202.430 ms | 254.787 GFLOP/s |
| 8T / cores 0-7 | 36.653 ms | 36.576 ms | 36.701 ms | 1.406 TFLOP/s |

The 1T-to-8T speedup is 5.52x, or 69.0% parallel efficiency.

## Full post-GEMM long path

Configuration: `M=2048`, `context_start=4096`, compression ratio 4,
compressed `N=1536`, `topk=512`, Indexer `H=64, D=128`; 8 threads on cores
`0-7`, three warmups and 15 measured runs.

```bash
PYTHONPATH=src OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
OMP_PROC_BIND=close OMP_PLACES=cores taskset -c 0-7 \
.venv/bin/python tests/bench_deepseek_v4_post_gemm_stage.py \
  --m 2048 --context-start 4096 --threads 8 \
  --warmup 3 --runs 15 --profile
```

- Full-stage median: `186.251 ms`
- Range: `184.244-186.937 ms`
- Profiled run total: `186.860 ms`
- Direct paged-K packB: `0.044 ms`
- Weighted-ReLU score kernel including per-query Q pack: `36.154 ms`
- Row-wise TopK: `75.155 ms`
- Combined score + TopK: `111.310 ms`
- Runtime dispatch: `sve_8x2vl`, `n_tile=16`

## Remaining validation and next decision

Only the user-requested 8-core SVL256 machine has been tested in this round.
Single- and two-request long-path TopK integration both pass. The candidate
remains pending for the 192-core/SVL128 machine; the separate row-wise TopK
optimization is also not part of this change. The scalable layout is designed
for any legal SVE vector length, but that is not a substitute for the pending
SVL128 run.
