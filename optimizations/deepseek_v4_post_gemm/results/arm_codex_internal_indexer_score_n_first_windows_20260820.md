# Indexer score N-first window scheduling

## Configuration

- Machine: `Arm-codex-internal`, NUMA-local CPUs `0-79`
- Backend: SVE `8x2VL` score kernel, `VL=256`
- Shape: `M=2048`, `H=64`, `K=128`, compressed `N=8704`
- Useful score work: `2*M*H*K*N = 292.058 GFLOP`
- Candidate: packed-K N windows no larger than 1 MiB, then M split
- Baseline: the same SVE kernel with M-only scheduling
- Timing: median of 9 runs for the full sweep; the 40/64/80-thread candidate
  was repeated three times with 5 warmups and 15 timed runs

## Score kernel result

| Threads | M-only ms | N-first ms | Speedup | N-first TFLOP/s |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 480.644 | 477.758 | 1.006x | 0.611 |
| 16 | 239.458 | 239.546 | 1.000x | 1.219 |
| 32 | 119.465 | 119.187 | 1.002x | 2.451 |
| 40 | 96.216 | 96.050 | 1.002x | 3.041 |
| 64 | 124.717 | 60.197 | 2.072x | 4.852 |
| 80 | 94.578 | 47.963 | 1.972x | 6.089 |

The packed-K matrix is about 2.125 MiB in this case. The old row-only split
becomes non-monotonic above 40 threads. N-first ownership removes that collapse
without changing the microkernel, packed format, or numerical path. When
packed-K is below 1 MiB, the geometry selects one N window and retains the
historical M split.

The 64-thread full-stage samples remain noisy (occasional approximately 2x
outliers), so the table reports the stable profile score component rather than
using one full-stage sample as the kernel result.

## Validation

```text
PYTHONPATH=src:. OMP_NUM_THREADS=80 taskset -c 0-79 \
  .venv/bin/python -m pytest -q \
  tests/test_deepseek_v4_post_gemm_stage.py \
  tests/test_deepseek_v4_inv_rope_woa.py

46 passed, 1 skipped
```
