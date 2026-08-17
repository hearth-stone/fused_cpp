# Amazon M5 Sparse MLA score-copy/max fusion

Date: 2026-08-17

## Status

Experimental checkpoint for the ordered QK/softmax/PV fusion sequence. Keep it
on `exp/sdpa/sparse-attn` for the next combination step; do not treat it as an
independently adopted production optimization. It gives a small repeatable
gain for 2048 shared-prefix and later sparse, but misses the standalone 3% gate
and the 8192 result is obscured by the host's two performance modes.

## Change

`run_heads_qkpv_chunk_bf16` previously copied every `8x2VL` QK tile from
`qkt_tile` to the chunk score scratch and later scanned the complete score row
again only to compute its maximum. The candidate performs the copy and NEON
maximum reduction in one pass while the tile is hot, retains one scalar maximum
per head, and feeds that value into the unchanged online-softmax epilogue.

The score buffer, exp implementation, BF16 probability buffer, P pack, PV
kernel, chunk order, exact-statistics path, and public API are unchanged.

## Correctness

- M5, native SVL128, cores 96--99, four threads: direct native-versus-naive
  output/statistics checks passed for shared dense, ragged shared-prefix, and
  later indexed sparse.
- Amazon ECS 8C, native SVL256, cores 0--3, four threads:
  `OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores taskset -c 0-3
  .venv/bin/python -m pytest -q tests/test_sparse_mla.py` produced `26 passed`.
- All timed checksums matched the baseline.

## Benchmark method

- Host: `AmazonM5192Cores`, Neoverse V3, native SVL128.
- Placement: NUMA1 cores 96--191, 96 OpenMP threads, close/core binding,
  `numactl --physcpubind=96-191 --membind=1`.
- Data: BF16, `h_q=32`, `d_qk=192`, `d_v=128`, seed 20260817.
- Each run: ten warmups, 21 timed samples, median wall time.
- Three paired sessions used candidate/baseline, baseline/candidate, then
  candidate/baseline order. Each extension was a separate build swapped into
  the same Python environment.

Command template:

```bash
OMP_NUM_THREADS=96 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores \
numactl --physcpubind=96-191 --membind=1 \
  .venv/bin/python tests/bench_sparse_mla_scalable.py \
  --h-q 32 --d-qk 192 --d-v 128 --threads 96 --warmup 10 --iters 21 \
  --seed 20260817 <case arguments>
```

Cases were the same as the cache-scheduling experiment: 2048 shared-prefix
(compressed 512 + causal/SWA 2048, 2,621,952 valid pairs), 8192 shared dense
(top-k 640), and context-start 10000 later sparse (compressed 512 + window 128,
1,310,720 valid pairs).

## Results

| Case | Candidate session medians | Baseline session medians | Stable conclusion |
|---|---|---|---|
| 2048 shared-prefix | 11.493 / 11.484 / 11.475 ms | 11.626 / 11.738 / 11.613 ms | median-of-sessions 11.484 vs 11.626 ms, -1.22% |
| 8192 shared dense | 10.386 / 8.944 / 10.584 ms | 9.205 / 10.342 / 9.190 ms | inconclusive two-mode host state |
| Later low-overlap sparse | 2.537 / 2.534 / 2.541 ms | 2.605 / 2.606 / 2.607 ms | median-of-sessions 2.537 vs 2.606 ms, -2.65% |

The 8192 samples switch between an approximately 8.9--9.2 ms mode and a
10.3--10.6 ms mode. The mode follows neither binary nor execution order, so no
8192 speedup or regression is claimed from this run.

## Interpretation and next decision

The result confirms that the max-only score scan is measurable but not the
dominant cost. The later sparse case benefits more because its smaller chunks
make fixed score/softmax passes a larger fraction of work. The next ordered
step will eliminate the row-major BF16 probability buffer and separate P-pack
pass by writing exp results directly into the `8x2VL` PV A layout. The combined
variant will be judged again against the original baseline; this checkpoint is
not independently promoted.
