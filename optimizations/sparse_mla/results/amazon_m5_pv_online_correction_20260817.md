# Amazon M5 Sparse MLA PV online-output correction fusion

Date: 2026-08-17

## Decision

Reject the candidate. It removes the standalone output-rescale pass, but adds a
row scale multiply to every PV output tile. Relative to direct packed-P, 2048
shared-prefix was neutral/slightly slower and later sparse regressed about 0.7%.
Only 8192 showed a small same-mode gain. Preserve this checkpoint in Git, then
restore the step-2 implementation before the multi-query experiment.

## Change

The original online update scales every FP32 output row by
`exp(old_max-new_max)` before PV. The candidate keeps each row correction in an
eight-float array and changes the SVE PV store epilogue from
`old_output + pv_tile` to `old_output * correction + pv_tile`. Scalar max/sum
updates and their order are unchanged. The `8x2VL` BFMMLA body is unchanged.

## Correctness

- M5/SVL128 direct native-versus-naive dense/shared-prefix/later-sparse output
  and statistics checks passed.
- Amazon ECS 8C/SVL256, four threads, `tests/test_sparse_mla.py`: `26 passed`.
- Timed checksums matched step 2 and the original baseline.

## Benchmark method

Same M5 NUMA1 cores 96--191, native SVL128, 96 threads, BF16 shapes, seed,
ten warmups, and 21-sample medians as the first two checkpoints. Three session
orders were step3/step2/baseline, baseline/step2/step3, and
step2/step3/baseline.

## Results

| Case | Step 3 medians | Step 2 medians | Original baseline | Step 3 vs step 2 |
|---|---|---|---|---:|
| 2048 shared-prefix | 11.463 / 11.402 / 11.411 ms | 11.399 / 11.375 / 11.400 ms | 11.629 / 11.642 / 11.647 ms | +0.11% |
| 8192 shared dense | 8.899 / 10.251 / 8.789 ms | mixed 8.952 / 10.436 / 8.815 ms | 10.333 / 9.222 / 9.209 ms | same-mode about -0.3% to -1.8% |
| Later low-overlap sparse | 2.500 / 2.500 / 2.496 ms | 2.483 / 2.487 / 2.481 ms | 2.607 / 2.609 / 2.607 ms | +0.68% |

The percentages for 2048 and later sparse compare the median of session
medians. The 8192 host continued to change performance mode within and between
runs, so only approximate same-mode differences are stated.

## Interpretation

The removed scale pass is a contiguous NEON multiply over each 128-element
output row. Folding it into PV avoids that traffic but repeats scale setup and
multiply work across every `2VL` output tile and lengthens the already busy PV
store epilogue. Short indexed chunks are especially sensitive to that fixed
epilogue cost. The next experiment should keep step 2's separate vector scale
and seek reuse across multiple query tokens instead.
