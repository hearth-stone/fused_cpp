# Amazon ECS 8C SVE fused inverse-RoPE grouped WO_A

## Configuration

- Host: `AmazonECS8Cores`, AArch64, SVE vector length 256 bits.
- CPU affinity: cores `0-7`; scaling runs use prefixes of that set.
- Build: project default C++17 `-O2`, OpenMP enabled, xbyak_aarch64 enabled.
- Shape: DeepSeek V4 Flash TP2, `G=4`, `P=8`, `DH=512`, `RG=64`,
  `DG=4096`, `R=1024`, BF16 input/weight/output and FP32 accumulation.
- Baseline: materialized Torch inverse RoPE followed by
  `einsum("tgd,grd->tgr")`.
- Candidate: inverse RoPE during shared packed-A construction followed by SVE
  JIT `(group, M-panel, N-tile)` GEMM tasks.
- Weight prepare is excluded. Both paths reuse explicit output buffers.
- Warmup: 3. Samples: 15. Statistic: median and candidate P90 wall time.
- `native_tflops` counts only `2*T*G*R*DG` WO_A GEMM operations.

## Correctness

```text
PYTHONPATH=src OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close
OMP_PLACES=cores taskset -c 0-7 .venv/bin/python -m pytest -q
tests/test_deepseek_v4_inv_rope_woa.py

20 passed, 1 skipped
```

The native-tail sweep covers `T=1,7,8,9,11,12,13,24,25`. Across the
performance shapes, maximum absolute error versus the Torch reference is no
larger than `0.0020` BF16 output units.

## Eight-thread result

| T | Torch median ms | Native median ms | Speedup | Native P90 ms | Native TFLOP/s | Max abs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.292 | 0.293 | 4.411x | 0.305 | 0.115 | 0.0005 |
| 12 | 3.561 | 0.556 | 6.402x | 0.696 | 0.724 | 0.0010 |
| 64 | 8.715 | 2.328 | 3.743x | 2.605 | 0.922 | 0.0020 |
| 256 | 27.737 | 8.502 | 3.262x | 9.830 | 1.010 | 0.0020 |
| 2048 | 201.327 | 62.955 | 3.198x | 63.173 | 1.092 | 0.0020 |

## Thread scaling

| T | Threads | Torch median ms | Native median ms | Speedup | Native TFLOP/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 1 | 16.635 | 2.296 | 7.245x | 0.175 |
| 12 | 2 | 9.084 | 1.364 | 6.660x | 0.295 |
| 12 | 4 | 5.642 | 0.784 | 7.196x | 0.514 |
| 12 | 8 | 3.561 | 0.556 | 6.402x | 0.724 |
| 256 | 1 | 138.532 | 46.731 | 2.964x | 0.184 |
| 256 | 2 | 82.218 | 27.837 | 2.954x | 0.309 |
| 256 | 4 | 46.699 | 15.219 | 3.068x | 0.564 |
| 256 | 8 | 27.737 | 8.502 | 3.262x | 1.010 |
| 2048 | 1 | 1071.007 | 342.803 | 3.124x | 0.200 |
| 2048 | 2 | 642.276 | 210.434 | 3.052x | 0.327 |
| 2048 | 4 | 360.654 | 116.505 | 3.096x | 0.590 |
| 2048 | 8 | 201.327 | 62.955 | 3.198x | 1.092 |

For `T=2048`, native 1T-to-8T scaling is `5.445x`, or 68.1% parallel
efficiency. The result validates the fused implementation against the Torch
baseline, but it does not yet isolate pack-A/RoPE and GEMM stage times or
compare against an optimized two-stage native baseline.

## N-range correction and fixed-cost decomposition

The first implementation created one JIT call per SVE N tile. It was corrected
to split N only when `(group, M panel)` tasks cannot occupy the requested
threads; every task now lets one JIT invocation traverse a contiguous
multi-tile N range. Correctness remains `20 passed, 1 skipped`. At `T=2048,
R=1024,8T`, 31-sample median/P90 changed from `62.955/63.173 ms` to
`61.276/61.619 ms`, a 2.74% median improvement.

To separate fixed pack/RoPE work from GEMM, a 1T sweep held `T=2048`, `G=4`,
and `DG=4096` constant while changing output rank:

| R | Native median ms |
| ---: | ---: |
| 64 | 112.534 |
| 512 | 219.206 |
| 1024 | 340.909 |

The `R=512 -> 1024` increment contains 34.360 GFLOP and takes 121.703 ms,
which gives 282.3 GFLOP/s for the incremental GEMM work. Linear extrapolation
leaves about 97.5 ms of fixed packed-A allocation/construction, inverse-RoPE,
and phase overhead. For `R=1024`, the decomposition is therefore approximately
243.4 ms GEMM plus 97.5 ms fixed work. The reported end-to-end 201.6 GFLOP/s
does not mean the GEMM body regressed to that rate; the GEMM slope remains in
the previously observed 270--300 GFLOP/s range.

## Vectorized NoPE pack and reusable scratch

The pack path was then changed without altering the packed format or GEMM:

- each group's contiguous `P*DH` source is mapped once per physical row;
- NoPE K4 units use two 64-bit loads and one 128-bit NEON store for two packed
  rows;
- only K4 units inside the final `RG` dimensions execute FP32 inverse-RoPE;
- a caller-thread-local grow-only packed-A buffer replaces per-call `at::empty`.

Correctness remains `20 passed, 1 skipped`. The 1T output-rank sweep became:

| R | Before median ms | After median ms |
| ---: | ---: | ---: |
| 64 | 112.534 | 31.577 |
| 512 | 219.206 | 137.010 |
| 1024 | 340.909 | 259.564 |

The `R=512 -> 1024` slope is 280.4 GFLOP/s, while the fixed intercept falls
from about 97.5 ms to about 14.5 ms, an approximately 85% reduction. At the
production `R=1024` point, complete 1T equivalent throughput rises from 201.6
to 264.7 GFLOP/s.

The final 8T sweep, using five warmups and 31 measured calls, is:

| T | Torch median ms | Native median ms | Speedup | Native P90 ms | Native TFLOP/s | Max abs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.232 | 0.274 | 4.490x | 0.286 | 0.122 | 0.0005 |
| 12 | 3.513 | 0.396 | 8.873x | 0.406 | 1.017 | 0.0010 |
| 64 | 8.264 | 1.892 | 4.367x | 1.911 | 1.135 | 0.0020 |
| 256 | 25.871 | 6.671 | 3.878x | 6.698 | 1.288 | 0.0020 |
| 2048 | 195.420 | 51.276 | 3.811x | 51.388 | 1.340 | 0.0020 |

Relative to the initial single-N-tile task implementation, the final
`T=2048,8T` latency improves from 62.955 to 51.276 ms, or 18.55%.
