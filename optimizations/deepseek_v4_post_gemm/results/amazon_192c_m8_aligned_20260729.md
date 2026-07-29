# DeepSeek V4 Post-GEMM M8-Aligned Scheduling

## Configuration

- Date: 2026-07-29
- Machine: `AmazonC5192Cores`, 192-core AArch64 host
- NUMA binding: node 0, CPUs `0-95`
- Operator: DeepSeek V4 C4A post-GEMM stage
- Shape: `M=2048`; Main Q and Indexer Q GEMMs are both
  `(2048, 1024) x (1024, 8192)`
- Dtype: BF16 GEMM input, packed BF16 weights and BF16 Q outputs
- Build: effective `-O2`, OpenMP enabled
- Timing: three warmups; headline results use 15 measured calls and report the
  median

The legacy schedule assigns `ceil(M / threads)` rows to each thread. The new
schedule partitions `ceil(M / 8)` panels, so every non-final thread range is a
multiple of the M8 kernel height. Set `FUSED_CPP_POST_GEMM_M8_ALIGNED=0` to
restore the legacy schedule.

## Correctness

```text
NEON/default: 12 passed in 0.26s
SVE:          12 passed in 0.26s
```

The focused regression uses `M=25`, seven threads, and a partial final panel.
It compares Q, Top-K, SWA cache and both compressor state caches against the
legacy row split.

## Headline Results

| Backend | Threads | Schedule | Median (ms) | GEMM-equivalent TFLOP/s |
|---|---:|---|---:|---:|
| NEON | 96 | legacy | 13.942 | 4.929 |
| NEON | 96 | M8-aligned | 10.187 | 6.746 |
| NEON | 88 | legacy | 11.071 | 6.207 |
| NEON | 88 | M8-aligned | 10.770 | 6.381 |
| SVE | 96 | legacy | 11.082 | 6.201 |
| SVE | 96 | M8-aligned | 10.318 | 6.660 |

At 96 threads, M8 alignment reduces NEON latency by 26.9% and increases the
GEMM-equivalent throughput by 36.9%. The new 96-thread result is 5.4% faster
than the new 88-thread result, removing the previous non-monotonic optimum.
SVE latency improves by 6.9%.

The NEON 96-thread profile attributes the improvement to both GEMMs:

| Stage | Legacy (ms) | M8-aligned (ms) |
|---|---:|---:|
| Main Q GEMM | 6.507 | 4.447 |
| Indexer Q GEMM | 6.447 | 4.619 |
| Profile total | 14.115 | 10.045 |

## NEON Thread Curve

Each point below uses a fresh process, three warmups and nine measured calls.
Negative delta means the M8 schedule is faster.

| Threads | Legacy (ms) | M8-aligned (ms) | Latency delta |
|---:|---:|---:|---:|
| 1 | 272.376 | 272.450 | +0.0% |
| 2 | 146.206 | 148.311 | +1.4% |
| 4 | 72.984 | 74.074 | +1.5% |
| 8 | 46.566 | 46.917 | +0.8% |
| 16 | 28.234 | 28.265 | +0.1% |
| 24 | 24.756 | 22.245 | -10.1% |
| 32 | 24.510 | 24.318 | -0.8% |
| 40 | 23.012 | 21.622 | -6.0% |
| 48 | 25.080 | 19.847 | -20.9% |
| 56 | 21.389 | 17.034 | -20.4% |
| 64 | 15.377 | 15.156 | -1.4% |
| 72 | 17.652 | 12.900 | -26.9% |
| 80 | 14.008 | 12.232 | -12.7% |
| 88 | 11.168 | 10.910 | -2.3% |
| 96 | 13.926 | 10.389 | -25.4% |

For 1, 2, 4, 8, 16, 32 and 64 threads, the legacy row count is already an M8
multiple, so both schedules produce the same row ranges; the observed
differences of at most 1.5% are run-to-run noise. The material gains occur when
the legacy split produces M4/M2 tails.

## Reproduction

```bash
PYTHONPATH=src FUSED_CPP_ATTN_GEMM_BACKEND=neon \
OMP_NUM_THREADS=96 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores \
numactl --physcpubind=0-95 --membind=0 \
.venv/bin/python tests/bench_deepseek_v4_post_gemm_stage.py \
  --m 2048 --threads 96 --warmup 3 --runs 15 --schedule m8 --profile
```

Run the same command with `--schedule legacy` for the reference.

## Conclusion

M8-aligned static partitioning is enabled by default. It preserves the legacy
API and numerical behavior, has a runtime fallback, removes the 96-thread
regression, and does not show a material regression when the old partition was
already M8-aligned. The next scheduling step is a shared MN task pool; this
feature remains its alignment prerequisite.
