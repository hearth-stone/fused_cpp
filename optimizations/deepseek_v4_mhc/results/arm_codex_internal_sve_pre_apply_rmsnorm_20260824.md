# SVE mHC pre-apply and RMSNorm

## Decision

Keep the native pre consumer in the explicit SVE candidate. It preserves the
requested FP32 `pre_mix` memory boundary, passes both supported SVE vector
lengths, and removes the dominant Torch residual-reduction/RMSNorm cost. Public
dispatch remains unchanged pending a native post kernel and second-host check.

## Kernel

Input and output contracts are:

```text
residual    [T,4,H] BF16
pre_mix    [T,4]   FP32
norm_weight[H]     BF16
output      [T,H]  BF16
```

Threads split only T. Each token executes two H-vectorized passes:

1. Load four BF16 residual vectors, widen to FP32, apply four scalar pre
   coefficients with FMA, round to BF16, and store into the output buffer. The
   rounded value is widened again before accumulating its square, preserving
   the required pre-weighted-input BF16 boundary.
2. Compute the scalar RMS factor, immediately reload the token-local BF16 raw
   input and BF16 norm weight, and overwrite the same output buffer with BF16
   RMSNorm output.

At H=4096 the raw token row is 8 KiB, so the second pass can reuse private
cache. No `[T,4,H]` FP32 product or reduction tensor is materialized.

The high-level SVE pre candidate remains one Python/native call. Internally it
runs projection, writes FP32 pre/post/comb controls, then invokes this separate
consumer kernel. Post and combination controls remain live across the Attention
or FFN sublayer.

## Correctness

On `Arm-codex-internal`, NUMA3 CPUs `240-247`, with the Torch bundled `libgomp`
preloaded, both SVE256 and forced SVE128 passed:

```text
31 passed
```

The direct pre-consumer test covers T=`0/1/9/17` and H=`65/128`; the full suite
also covers projection K splits, Sinkhorn tails and strict post-to-pre BF16
boundaries. At T=2048/C=4/H=4096:

| Metric | Value |
| --- | ---: |
| normed-input maximum absolute error | `0.03125` |
| normed-input relative L2, pre | `3.25e-5` |
| normed-input relative L2, post-pre | `3.36e-5` |
| NaN / Inf | `0 / 0` |

## Performance

Configuration matches the control experiment: SVE256, NUMA3-local placement,
BF16 residual, T=2048/C=4/H=4096, five warmups, 21 AB/BA-alternating samples,
and median wall time. Torch's bundled `libgomp` is preloaded to avoid the host's
dual-runtime cpuset issue.

Isolated pre-apply plus RMSNorm:

| Threads | Torch, ms | SVE, ms | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 43.069 | 9.442 | 4.56x |
| 8 | 13.611 | 1.438 | 9.46x |
| 80 | 8.522 | 0.726 | 11.74x |

Full SVE projection plus postprocess, using the same projection on both sides:

| Threads | Torch postprocess, ms | SVE postprocess, ms | Latency reduction | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 82.934 | 42.630 | 48.60% | 1.945x |
| 8 | 23.804 | 7.368 | 69.05% | 3.231x |
| 80 | 12.791 | 1.588 | 87.59% | 8.057x |

The existing full-stage benchmark at 80 threads measured:

| Stage | Torch, ms | SVE candidate, ms | Speedup |
| --- | ---: | ---: | ---: |
| pre | 18.882 | 0.943 | 20.03x |
| post-pre | 29.464 | 12.719 | 2.32x |

`post-pre` is now dominated by the still-Torch fixed-K4 residual mixing and
layer-output injection. That is the next independent fusion boundary.
