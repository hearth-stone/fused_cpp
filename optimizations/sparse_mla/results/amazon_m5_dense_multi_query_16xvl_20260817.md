# Amazon M5 Sparse MLA dense multi-query 16xVL experiment

Date: 2026-08-17

## Decision

Reject the dense-only two-token kernel. It halves packed-B vector loads by
changing each half tile from one token's `8x2VL` to two tokens' `16xVL`, but a
clean same-mode comparison regressed 8192 shared-dense by 2.75% relative to the
direct packed-P checkpoint. Preserve the implementation in Git, then restore
the step-2 dense path.

## Change

Two adjacent dense query tokens retain independent Q packs, score buffers,
online max/sum/output, packed probabilities, output, and statistics. New
`qkt_16xvl_bf16` and `pv_16xvl_bf16` kernels use the same 16 SVE FP32
accumulators as `8x2VL`: eight 2-row A panels from two tokens multiply two
packed-B column-pair vectors. Two half-tile calls cover the existing four-pair
2VL K/V layout in original column order, so no K/V repack is added.

Only fully shared dense dispatch uses token pairs. An odd final token falls
back to `8x2VL`; shared-prefix, indexed sparse, non-SVE, and public APIs are
unchanged.

## Correctness

- M5/SVL128 direct native-versus-naive checks passed for even dense tokens,
  odd dense token tails, return statistics, shared-prefix, and later sparse.
- Amazon ECS 8C/SVL256, four threads, full `tests/test_sparse_mla.py`:
  `26 passed`.
- Timed checksums matched step 2 and the original baseline.

## Benchmark method

M5 NUMA1 cores 96--191, native SVL128, 96 threads, BF16
`h_q=32,d_qk=192,d_v=128`, seed 20260817, ten warmups and 21 samples. Three
orders were step4/step2/baseline, baseline/step2/step4, and
step2/step4/baseline. Shapes and commands match the preceding checkpoints.

## Results

| Case | Step 4 medians | Step 2 medians | Original baseline | Conclusion |
|---|---|---|---|---|
| 2048 shared-prefix | 11.395 / 11.419 / 11.396 ms | 11.401 / 11.391 / 11.384 ms | 11.654 / 11.649 / 11.612 ms | unchanged, about +0.04% vs step 2 |
| 8192 shared dense | 9.071 / 10.488 / 9.056 ms | 10.296 / 9.854 / 8.814 ms | 9.211 / 9.213 / 9.219 ms | clean low-mode session: +2.75% vs step 2, -1.77% vs baseline |
| Later low-overlap sparse | 2.488 / 2.488 / 2.488 ms | 2.487 / 2.483 / 2.492 ms | 2.610 / 2.605 / 2.605 ms | unchanged, about +0.04% vs step 2 |

The 8192 machine state remained multimodal. Session 3 is the clean same-mode
comparison: step 2 and step 4 were stable at 8.814 and 9.056 ms respectively.

## Interpretation

Packed-B loads were not expensive enough to justify splitting each 2VL tile
into two function/epilogue phases and doubling per-task token state. The result
also agrees with the earlier token-panel experiment: adjacent-token K/V reuse
alone is not the current bottleneck. A useful larger-M kernel would need more
physical accumulators or a different end-to-end QK/softmax/PV organization,
not only a fixed 16-accumulator aspect-ratio exchange.
