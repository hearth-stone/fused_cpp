# DeepSeek V4 Post-GEMM Shared Q Worker Pool

## Configuration

- Date: 2026-07-29
- Machine: `AmazonC5192Cores`, 192-core AArch64 host
- NUMA binding: node 0, CPUs `0-95`
- Shape: C4A TP4, Main Q and Indexer Q GEMMs both use
  `(M, 1024) x (1024, 8192)`
- Dtype: BF16 input, packed BF16 weights and BF16 outputs
- Baseline: M8-aligned sequential Main Q then Indexer Q GEMMs
- Candidate: one OpenMP region with two M8-panel cursors
- Build: effective `-O2`, OpenMP enabled

Each worker starts from a preferred GEMM and repeatedly claims one M8 panel.
Once that GEMM is exhausted, it steals panels from the other cursor. The
outputs remain separate and all Norm, RoPE, cache, compressor and Top-K stages
run in their original order after both GEMMs complete.

## Correctness

```text
NEON/default: 13 passed in 0.37s
SVE:          13 passed in 0.34s
```

The focused regression compares the shared and sequential paths with `M=25`
and seven threads. It covers a partial final panel and checks Q, Top-K, SWA
cache and both compressor state caches.

## Headline Results

Five warmups and 31 measured calls were used for the headline medians.

| Backend | Threads | Pool | Median (ms) | GEMM-equivalent TFLOP/s |
|---|---:|---|---:|---:|
| NEON | 96 | sequential | 10.309 | 6.666 |
| NEON | 96 | shared | 9.415 | 7.299 |
| SVE | 96 | sequential | 10.287 | 6.680 |
| SVE | 96 | shared | 10.225 | 6.721 |

The shared pool reduces NEON stage latency by 8.7% and raises equivalent
throughput by 9.5%. SVE improves by 0.6%, which is effectively neutral but
does not require a separate runtime path.

The NEON profile attributes the improvement to the combined GEMM region:

| Metric | Sequential (ms) | Shared (ms) |
|---|---:|---:|
| Main Q GEMM | 4.592 | included in shared |
| Indexer Q GEMM | 4.559 | included in shared |
| Combined Q GEMM wall time | 9.151 | 8.400 |
| Full post-GEMM stage | 10.187 | 9.383 |

## Forced Thread Sweep

The table forces the shared implementation at every thread count. Each point
uses a fresh process, three warmups and nine measured calls. Positive delta
means the shared pool is slower.

| Threads | Sequential (ms) | Shared (ms) | Latency delta |
|---:|---:|---:|---:|
| 1 | 272.337 | 272.572 | +0.1% |
| 2 | 150.087 | 138.568 | -7.7% |
| 4 | 72.706 | 72.983 | +0.4% |
| 8 | 46.572 | 47.272 | +1.5% |
| 16 | 28.033 | 29.044 | +3.6% |
| 24 | 22.215 | 22.397 | +0.8% |
| 32 | 24.274 | 23.942 | -1.4% |
| 40 | 21.521 | 21.329 | -0.9% |
| 48 | 19.680 | 19.093 | -3.0% |
| 56 | 16.700 | 16.035 | -4.0% |
| 64 | 15.019 | 15.061 | +0.3% |
| 72 | 13.121 | 12.720 | -3.1% |
| 80 | 11.986 | 10.999 | -8.2% |
| 88 | 10.866 | 10.248 | -5.7% |
| 96 | 10.419 | 9.799 | -5.9% |

Low-thread execution is not consistently improved because each GEMM receives
roughly half of the active workers. The automatic policy therefore requires at
least 32 threads.

## Route-Length Sweep And Auto Policy

The forced 96-thread sweep exposed a second boundary:

| M | M8 panels per GEMM | Sequential (ms) | Shared (ms) | Latency delta |
|---:|---:|---:|---:|---:|
| 12 | 2 | 1.229 | 0.819 | -33.4% |
| 48 | 6 | 1.243 | 1.185 | -4.6% |
| 192 | 24 | 1.411 | 1.622 | +14.9% |
| 768 | 96 | 3.652 | 3.629 | -0.6% |
| 2048 | 256 | 10.164 | 9.717 | -4.4% |

At `M=192`, the sequential path activates 24 full-N weight streams at a time,
while the shared path activates 48. The extra concurrency reduces GEMM
efficiency. Once each GEMM has at least one M8 panel per thread, both paths
already use all cores and sharing no longer increases the active stream count.

The default policy is therefore deliberately conservative:

```text
threads >= 32 and ceil(M / 8) >= threads
```

`FUSED_CPP_POST_GEMM_SHARED_Q_POOL=0` disables the pool. Any non-false value
forces it for experiments, including small M cases. Disabling M8 alignment
also disables the shared pool.

## Reproduction

```bash
PYTHONPATH=src FUSED_CPP_POST_GEMM_BACKEND=neon \
OMP_NUM_THREADS=96 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores \
numactl --physcpubind=0-95 --membind=0 \
.venv/bin/python tests/bench_deepseek_v4_post_gemm_stage.py \
  --m 2048 --threads 96 --warmup 5 --runs 31 \
  --schedule m8 --q-pool shared --profile
```

Use `--q-pool legacy` for the sequential reference and `--q-pool auto` for the
default runtime selection.
