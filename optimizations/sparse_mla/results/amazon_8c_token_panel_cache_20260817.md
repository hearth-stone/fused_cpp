# Amazon 8C Sparse MLA token-panel cache scheduling

Date: 2026-08-17

## Decision

Retire the prototype. Grouping adjacent tokens so A/C state stays in private L2
and shared packed-K/V B chunks are visited outside the token loop produced a
small, repeatable gain at panel 4, but missed the predeclared 3% gate. Panel 8
regressed at eight threads. No production source or runtime selector remains.

## Hypothesis and variants

The production shared-prefix path owns one token per OpenMP task. For each
token it traverses the complete compressed and causal/SWA packed-K/V ranges.
The prototype retained token panels as the primary parallel dimension but
changed the local loop order to:

```text
token panel -> segment -> packed B chunk -> token -> QK/softmax/PV
```

Packed Q, score/P scratch, and online `(m,l,O)` state were allocated for the
whole panel. Each token saw the same segment and B-chunk order as production.
Panels 4 and 8 were compared. A separate diagnostic forced the existing
8-query token-major fallback; it is not an isolated scheduler comparison
because that fallback also uses the older fixed-width compute path.

## Method

- Source baseline: commit `4e1df1e`; prototype was never committed.
- Host: `AmazonECS8Cores`, native SVL256, cores 0 or 0--7.
- Shape: BF16 `q[2048,32,192]`, `kv[2560,1,192]`, `d_v=128`, compressed
  capacity 512, causal/SWA window 2048, ratio 4, `context_start=0`.
- Work: 2,621,952 valid query-key pairs; identical seed 20260817.
- Timing: five warmups, 21 samples, median wall time; `OMP_DYNAMIC=FALSE`,
  close binding, dependent libraries limited to one thread.
- Correctness: panel 4 passed the two shared-prefix output/statistics tests;
  timed checksums were identical to baseline at each thread count.

Command template:

```bash
OMP_NUM_THREADS=<1|8> OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
taskset -c <0|0-7> .venv/bin/python tests/bench_sparse_mla_scalable.py \
  --pattern sparse --s-q 2048 --h-q 32 --d-qk 192 --d-v 128 \
  --compressed-capacity 512 --window-size 2048 --compress-ratio 4 \
  --context-start 0 --threads <1|8> --warmup 5 --iters 21 \
  --seed 20260817
```

## Results

| Variant/order | 1T median | Change | 8T median | Change |
|---|---:|---:|---:|---:|
| Baseline, first order | 454.085 ms | reference | 76.714 ms | reference |
| Panel 4, first order | 447.179 ms | -1.52% | 75.983 ms | -0.95% |
| Panel 4, reverse order | 447.129 ms | -1.48% | 76.165 ms | -0.88% |
| Baseline, reverse order | 453.853 ms | reference | 76.838 ms | reference |
| Panel 8 | 447.572 ms | -1.43% | 77.526 ms | +1.06% |
| Existing token-major fallback | 827.890 ms | +82.32% | 137.295 ms | +78.97% |

Panel 4 checksum was `-46800.308594` at one thread and `-46800.312500` at
eight threads, identical to production. Panel 8 was also identical. The
token-major fallback differs by one BF16 reduction-order checksum step while
passing the shared-prefix reference tolerances.

## Interpretation

Moving the B-chunk loop outside four tokens confirms a small cache-reuse
benefit. It does not enlarge the 8-row BFMMLA microkernel: the same number of B
operands is still loaded by the same 8-head groups, only from a warmer cache.
At panel 8, the larger Q, output, score, probability, and packed-P live set
starts to offset that benefit and slightly hurts eight-thread execution.

The old token-major fallback is not a viable replacement because it also gives
up the current scalable head-major QK/PV path. A true larger-M kernel would need
more than the current 16 scalable FP32 accumulators; doubling M would consume
all 32 SVE registers before A/B temporaries, forcing spills or accumulator
round-trips.

The requested M5 follow-up is recorded in
`amazon_m5_token_panel_cache_20260817.md`. A full `{4,8,16}` token-panel by
`{64,128,256}` B-block sweep on cores 96--191 also missed the adoption gate;
PMU data showed reduced L2 refill traffic for the 2048 case without a material
wall-time improvement.
