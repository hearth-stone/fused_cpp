# Inverse-RoPE WOA N-first window scheduling

## Configuration

- Machine: `Arm-codex-internal`, NUMA-local CPUs `0-79`
- Shape: tokens `M=2048`, groups `4`, grouped K `4096`, output N `1024`
- Packed-B: 8 MiB per group; candidate uses eight 1 MiB N windows/group
- Useful GEMM work: `68.719 GFLOP`
- Candidate: N-window ownership followed by equal M12-panel tasks, targeting
  approximately four tasks per worker
- Baseline: the same packed-A and SVE JIT kernels with M-panel-first scheduling
- Timing: 3 warmups and 15 timed runs; median reported

## Result

| Threads | M-first ms | N-first ms | Speedup | N-first TFLOP/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 803.759 | 799.007 | 1.006x | 0.086 |
| 8 | 100.945 | 101.067 | 0.999x | 0.680 |
| 16 | 51.233 | 50.961 | 1.005x | 1.348 |
| 32 | 26.643 | 26.481 | 1.006x | 2.595 |
| 40 | 21.791 | 21.760 | 1.001x | 3.158 |
| 64 | 13.846 | 13.987 | 0.990x | 4.913 |
| 80 | 11.437 | 11.194 | 1.022x | 6.139 |

The dedicated schedule is performance-neutral over most of the sweep and
improves the 80-thread point by 2.17%; the worst measured regression is 1.01%
at 64 threads. A first implementation that reduced the window to 0.8 MiB only
to divide 40/80 threads regressed by 4-6%. Keeping 1 MiB N ownership and
overdecomposing equal M work removed that regression.

## Validation

The combined post-GEMM and WOA suite passed on the target SVE machine:
`46 passed, 1 skipped`.
