# NEON M12xN4 final HC head

## Decision

Keep the implementation as an explicit candidate. It passes the primary-host
correctness and performance gates. Public dispatch remains on the Torch
baseline pending a second-machine holdout.

## Implementation

The candidate performs:

```text
native SVE post
→ BF16 final_residual [T,4,H]
→ residual sqrsum
→ NEON FP32 M12xN4 head projection
→ SVE head sigmoid
→ four-stream BF16 reduction
→ final RMSNorm
```

The FP32 head weight is `[4H,4]`. The M12 kernel loads one 128-bit N4 weight
per K and reuses it across twelve row accumulators. A partial final block is
zero-padded to M12. This kernel is independent of SVE VL; SVE is retained for
post, sigmoid, BF16 conversion and final reduction/RMSNorm.

GNU `objdump` showed twelve live accumulators (`v8`, `v21`, `v31` through
`v22`) receiving indexed FMLA instructions, with no Q-register stack spill.

## Method

- Machine: `Arm-codex-internal`, NUMA3.
- Shape: `T=2048`, `C=4`, `H=4096`, head GEMM `M=2048,K=16384,N=4`.
- Types: BF16 residual/layer/output, FP32 controls and head weight.
- CPUs: `240`, `240-247`, `240-255`, `240-271`, `240-303`, `240-319`.
- Build: GCC 13, C++17, `-O2`, SVE256; NEON M12xN4 head kernel.
- Runtime: Torch bundled libgomp preloaded, static binding to NUMA3.
- Statistic: five warmups, 21 samples, median reported.

## Performance

| Threads | Torch, ms | Native, ms | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 152.839 | 55.076 | 2.78x |
| 8 | 48.922 | 8.404 | 5.82x |
| 16 | 33.435 | 4.465 | 7.49x |
| 32 | 27.779 | 2.844 | 9.77x |
| 64 | 24.994 | 2.242 | 11.15x |
| 80 | 25.226 | 1.917 | 13.16x |

## Accuracy

| Output | Maximum absolute error | Relative L2 |
| --- | ---: | ---: |
| `final_residual` BF16 | 0.015625 | 9.89e-6 |
| `hidden_states` BF16 | 0.03125 | 4.10e-5 |

SVE256 and process-forced SVE128 each passed all 39 tests in
`tests/test_deepseek_v4_mhc.py`. The remote build was restored to SVE256 after
validation.
