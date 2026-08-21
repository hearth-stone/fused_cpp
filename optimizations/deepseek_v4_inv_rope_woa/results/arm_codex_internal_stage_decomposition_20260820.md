# WOA pack/RoPE and prepacked-GEMM decomposition

## Question

Does the 80-thread inverse-RoPE WOA operator fail to reach the historical
four-GEMM 91.2% peak efficiency because of its GEMM body, or because its
reported FLOP/s includes pack-A and inverse RoPE while counting only GEMM
operations?

## Configuration

- Host: `Arm-codex-internal`, CPUs `0-79`
- Shape: `M=2048`, `G=4`, `K=4096`, `N=1024`, BF16 output
- Useful work: `68.719 GFLOP`
- Reference peak: `7.418 TFLOP/s`
- Production schedule: eight 1 MiB N windows/group, ten M splits/window,
  320 static tasks
- Statistic: five warmups, 31 measured runs, median and P90

The Lab-only prepacked probe uses the production M12/exact-tail JIT kernels,
packed-A and packed-B layouts, output row stride, and
`(group, N-window, M-split)` geometry. Its timed region excludes input packing,
inverse RoPE, allocation, and weight preparation.

## Primary result

| Measurement | Median | P90 | Throughput | Peak efficiency |
| --- | ---: | ---: | ---: | ---: |
| Complete WOA | 11.173 ms | 11.477 ms | 6.151 TFLOP/s | 82.92% |
| Prepacked pure GEMM | 10.622 ms | 10.637 ms | 6.470 TFLOP/s | 87.22% |
| Difference | 0.551 ms | - | - | - |

Pack-A, inverse RoPE, and the pack-to-GEMM phase transition therefore account
for approximately `0.551 / 11.173 = 4.93%` of complete WOA latency. The GEMM
body accounts for the remaining 95.1% and reaches 87.2% of the instruction
reference.

## Output-rank sweep

| N | Complete WOA ms | Prepacked GEMM ms | Non-GEMM difference ms |
| ---: | ---: | ---: | ---: |
| 64 | 1.530 | 0.919 | 0.611 |
| 128 | 2.425 | 1.821 | 0.604 |
| 256 | 3.711 | 3.159 | 0.552 |
| 384 | 5.425 | 4.844 | 0.581 |
| 512 | 6.250 | 5.479 | 0.771 |
| 768 | 9.778 | 9.273 | 0.505 |
| 1024 | 11.187 | 10.625 | 0.562 |

The median difference across ranks is 0.581 ms. The nonlinearity in total time
comes from discrete N-window and M-split geometries, so a single linear
intercept is less accurate than the paired same-N subtraction.

## Ruled-out explanations

### Number of independent group A matrices

Holding total `G*N=4096`, B/C bytes, FLOPs, N owners, and task count constant
gave 6.46-6.49 TFLOP/s for `G=1,2,4,8`. Four independent packed-A matrices do
not explain the pure-GEMM gap.

### M task granularity and dynamic scheduling

| Tasks/thread target | Tasks | Static TFLOP/s |
| ---: | ---: | ---: |
| 1 | 80 | 6.204 |
| 2 | 160 | 6.393 |
| 4 | 320 | **6.466** |
| 8 | 640 | 6.348 |
| 16 | 1280 | 5.656 |
| 32 | 2560 | 4.771 |
| 64 | 5120 | 4.771 |

At 2/4/8 tasks per thread, OpenMP dynamic scheduling measured
6.409/6.377/6.253 TFLOP/s, respectively. The production four-task static
geometry is the best tested point; adopting the four-GEMM atomic panel cursor
would not close the gap.

### BF16 store epilogue

With every other input fixed, BF16 and FP32 stores measured 6.470 and 6.476
TFLOP/s. BF16 conversion/store is not the cause.

### N-window width

The production 1 MiB window (`N=128`) measured 6.458 TFLOP/s in the width
sweep. Smaller 0.625-0.875 MiB windows measured only 5.74-5.97 TFLOP/s, and
irregular 7/9-window geometries were worse. The current 1 MiB window is the
best tested choice.

## Same-build four-GEMM comparison

The historical four-GEMM result was 9.684 ms and 6.763 TFLOP/s, or 91.2% of
the same peak. One immediate rerun measured an unstable 15.124 ms and was
rejected as contaminated. After verifying that CPUs 0-79 were idle, three
independent `5 warmup + 31 run` repetitions measured:

| Repetition | Median ms | Best ms | Median TFLOP/s | Peak efficiency |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 9.906 | 9.870 | 6.612 | 89.13% |
| 2 | 9.932 | 9.893 | 6.594 | 88.90% |
| 3 | 9.866 | 9.816 | 6.639 | 89.50% |

The median of repetition medians is 9.906 ms and 6.612 TFLOP/s. The historical
91.2% point remains 2.3 percentage points higher, but the clean current result
is close and stable rather than the earlier contaminated 58.4% sample.

Against this same-build reference, WOA prepacked GEMM is 2.2% slower
(`6.470` versus `6.612 TFLOP/s`). The larger end-to-end efficiency difference
comes from adding the measured 0.551 ms pack/RoPE/phase cost.

## Conclusion

The current WOA result is explained as:

- 87.2% peak efficiency in the prepacked full GEMM, including normal A/B load,
  loop/control, and C-store costs relative to an instruction reference;
- a further 4.9% complete-operator latency from pack-A, inverse RoPE, and phase
  transition;
- 82.9% effective end-to-end GEMM throughput.

No tested WOA scheduling, output-store, group-layout, or N-window variant
closes the pure-GEMM result above 87.2%. The clean current four-GEMM reference
is 89.1%, leaving a 2.2% pure-GEMM difference; the exact historical 91.2% point
was not reproduced.
