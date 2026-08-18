# Amazon M5 Sparse MLA direct packed-P generation

Date: 2026-08-17

## Status

Experimental checkpoint layered on score-copy/max fusion. The implementation
passes SVL128 and SVL256 correctness and adds a repeatable 1--2% marginal gain
on the stable cases. Keep it for the next online-state epilogue fusion step;
the final combined sequence still needs a stable 8192 adoption result.

## Change

The SVE head-major path previously wrote softmax probabilities as row-major
BF16 `[8,key_count]`, then reread and copied them into the BFMMLA A-panel layout
`[K/4][8 rows][4 BF16]`. The new exp helper writes each four-probability group
directly to its final packed-P location and explicitly zero-pads the final K=4
block. The row-major BF16 P allocation is empty on SVE and the separate
`pack_p_8rows_bf16` pass is gone.

The exp polynomial, FP32 sum accumulation order, BF16 conversion, online chunk
order, `8x2VL` PV kernel, non-SVE fallback, and public API are unchanged.

## Correctness

- M5/SVL128 cores 96--99: direct native-versus-naive output/statistics checks
  passed for shared dense, ragged shared-prefix, and later sparse.
- Amazon ECS 8C/SVL256 cores 0--3: `tests/test_sparse_mla.py` reported
  `26 passed` with four threads.
- Timed checksums matched step 1 and the original baseline.

## Benchmark method

M5 NUMA1 cores 96--191, native SVL128, 96 threads, close/core binding, BF16
`h_q=32,d_qk=192,d_v=128`, seed 20260817. Each run used ten warmups and 21
timed samples. Three sessions used orders step2/step1/baseline,
baseline/step1/step2, and step1/step2/baseline. Commands and shapes match
`amazon_m5_score_copy_max_20260817.md`.

## Results

| Case | Step 2 session medians | Step 1 session medians | Original baseline | Step 2 vs step 1 | Step 2 vs baseline |
|---|---|---|---|---:|---:|
| 2048 shared-prefix | 11.379 / 11.384 / 11.391 ms | 11.503 / 11.480 / 11.514 ms | 11.611 / 11.656 / 11.625 ms | -1.03% | -2.07% |
| 8192 shared dense | 8.753 / 10.445 / 8.765 ms | 8.936 / 8.967 / 8.937 ms | 9.196 / 9.227 / 9.253 ms | low-mode -1.9% | low-mode about -5.0% |
| Later low-overlap sparse | 2.495 / 2.479 / 2.485 ms | 2.539 / 2.539 / 2.534 ms | 2.608 / 2.605 / 2.609 ms | -2.13% | -4.72% |

The reported 2048 and later-sparse percentages compare the median of the three
session medians. For 8192, two step-2 sessions were a stable 8.75--8.77 ms and
one entered the known 10.45 ms high mode; only low-mode comparisons are shown,
and no unconditional 8192 claim is made.

## Interpretation and next decision

Eliminating the row-major P write/read and repack has a larger effect on short
indexed chunks, where pack overhead is a greater fraction of QK/PV work. The
2048 cumulative improvement is still about 2.1%, so the next step will fuse the
online `(m,l,O)` correction and PV update boundary. The combined variant will
continue to be compared with both step 2 and the original baseline.
