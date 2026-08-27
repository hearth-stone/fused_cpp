# SVE fixed-K4 plus rank-one mHC post

## Decision

Keep `epilogue.sve_post_k4_rank1` in the explicit SVE candidate. It meets the
primary-machine correctness and performance gates. Public dispatch remains on
the Torch baseline pending a second-machine holdout of the complete candidate.

## Kernel

For fixed `C=4`, the kernel computes:

```text
output[t,j,h] = sum_i comb_mix[t,i,j] * residual[t,i,h]
                + post_mix[t,j,0] * layer_output[t,h]
```

The H-U2 loop holds eight FP32 output vectors. Each chain starts with `FMUL`
from residual stream zero, consumes residual streams one through three and the
rank-one layer injection with four `FMLA` instructions, then rounds to BF16 and
stores with `ST1H`. T is the only OpenMP split.

GNU `objdump` of the SVE256 build confirmed the intended sequence; for example,
the chain beginning at address `0xfa71c` is one `FMUL` followed by four `FMLA`
instructions. There is no accumulator-zeroing instruction.

## Method

- Machine: `Arm-codex-internal`, NUMA3.
- CPUs: `240` for 1T, `240-247` for 8T, `240-319` for 80T.
- Build: GCC 13, C++17, `-O2`, SVE256.
- Shape: `T=2048`, `C=4`, `H=4096`; BF16 residual/layer/output and FP32 controls.
- Placement: `numactl --physcpubind=<cpus> --membind=3`.
- Runtime: Torch bundled libgomp preloaded, `OMP_DYNAMIC=FALSE`,
  `OMP_PROC_BIND=close`, `OMP_PLACES=cores`.
- Statistic: five warmups, 21 alternating AB/BA samples, median reported.

Benchmark command:

```bash
PYTHONPATH=src:. LD_PRELOAD=<torch-bundled-libgomp> \
OMP_NUM_THREADS=<threads> OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
OMP_PLACES=cores numactl --physcpubind=<cpus> --membind=3 \
.venv/bin/python optimizations/deepseek_v4_mhc/benchmarks/\
bench_sve_control_postprocess.py --tokens 2048 --hidden-size 4096 \
--threads <threads> --warmup 5 --runs 21
```

## Isolated post

| Threads | Torch, ms | SVE, ms | Speedup | Latency reduction |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 54.179 | 14.113 | 3.84x | 74.0% |
| 8 | 22.700 | 3.299 | 6.88x | 85.5% |
| 80 | 12.890 | 1.503 | 8.58x | 88.3% |

Maximum BF16 absolute error was `0.015625`.

## Complete post-pre

This comparison includes post, the next N24 projection and control processing,
pre residual application, the explicit BF16 boundaries, and RMSNorm.

| Threads | Torch baseline, ms | SVE candidate, ms | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 172.904 | 58.349 | 2.96x |
| 8 | 57.131 | 10.008 | 5.71x |
| 80 | 30.603 | 2.345 | 13.05x |

Post residual relative L2 error was `9.89e-6`; downstream normed-input maximum
absolute error was `0.03125` and relative L2 error was `3.75e-5`.

## Correctness

SVE256 and process-forced SVE128 each passed all 36 tests:

```bash
PYTHONPATH=src:. LD_PRELOAD=<torch-bundled-libgomp> \
OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores \
numactl --physcpubind=240-247 --membind=3 \
.venv/bin/python -m pytest -q tests/test_deepseek_v4_mhc.py
```

The forced-SVE128 run set `PR_SVE_SET_VL` to 16 bytes before importing the
extension. The remote build was restored to SVE256 after validation.

## H-unroll experiment

An H-U4 main loop was compared with the adopted U2 loop using the same shape,
placement, warmup count and 21-sample median method. U2 and U4 were separate
builds; they were not interleaved within one process.

| Threads | U2, ms | U4, ms | U4 latency change |
| ---: | ---: | ---: | ---: |
| 1 | 14.113 | 16.806 | +19.1% |
| 8 | 3.299 | 3.303 | +0.1% |
| 16 | 1.224 | 2.136 | +74.5% |
| 32 | 1.755 | 0.740 | -57.8% |
| 64 | 0.481 | 0.759 | +57.9% |
| 80 | 1.503 | 1.387 | -7.7% |

The crossover is strongly non-monotonic: U4 improves the sampled 32T and 80T
points but materially regresses 1T, 16T and 64T. A thread-count threshold would
therefore encode unstable machine-specific behavior rather than a robust
kernel property. The U4 source was removed and U2 remains the candidate path.

An additional U1-only build disabled the U2 loop and sent the full H range
through the single-vector tail loop. U2 was faster at every sampled point:

| Threads | U1, ms | U2, ms | U2 latency reduction | U2 speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 23.349 | 14.113 | 39.6% | 1.65x |
| 8 | 4.143 | 3.299 | 20.4% | 1.26x |
| 16 | 2.581 | 1.224 | 52.6% | 2.11x |
| 32 | 1.867 | 1.755 | 6.0% | 1.06x |
| 64 | 0.557 | 0.481 | 13.6% | 1.16x |
| 80 | 1.566 | 1.503 | 4.0% | 1.04x |

This supports the microarchitectural argument: U1 exposes four independent
output chains, while U2 exposes eight, enough to cover a four-cycle FMLA latency
at two FMLA instructions per cycle without the register pressure of U4. The U1
experiment source was removed after measurement.

## Pre pipeline experiment

A first-stage pipeline kept projection and Sinkhorn unchanged, but executed
each token's control row and pre-apply/RMSNorm in one OpenMP region. It removed
the FP32 `pre_mix[T,4]` tensor and one parallel-region launch. Existing staged
native entrypoints and the pipelined candidate were measured in the same
process with alternating AB/BA order.

| Threads | Staged, ms | Pipelined, ms | Pipeline latency change |
| ---: | ---: | ---: | ---: |
| 1 | 41.410 | 41.253 | -0.4% |
| 8 | 6.521 | 6.586 | +1.0% |
| 16 | 3.488 | 3.519 | +0.9% |
| 32 | 1.954 | 2.006 | +2.7% |
| 64 | 1.125 | 1.241 | +10.4% |
| 80 | 0.805 | 0.819 | +1.8% |

All three outputs were bit-identical. The eliminated tensor is only 32 KiB at
this shape, while binding the light control work to the much heavier H scan
removes the independent scheduling of the two stages. The experiment therefore
failed the non-regression gate; its implementation was removed and the staged
native path restored.
