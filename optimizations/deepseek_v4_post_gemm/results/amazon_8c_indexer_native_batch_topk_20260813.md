# Amazon 8C DeepSeek V4 Indexer Native Batched Exact TopK

## Outcome

Replacing 2,048 independent ATen row-wise TopK calls with one native OpenMP
batch reduced the TopK profile from 75.155 ms to 9.637 ms (7.799x) on the
requested Amazon 8-core machine. The full post-GEMM median improved from
186.251 ms to 120.758 ms (1.542x). Weighted-ReLU score time was unchanged at
about 36.16 ms.

The candidate preserves exact largest-K selection and score-descending output.
It remains a candidate because this round intentionally stopped after the
8-core SVL256 validation; 192-core/SVL128 remains untested.

## Change and dispatch

- One native call handles all rows with OpenMP static row scheduling.
- Each worker allocates one reusable `(float score, int32 index)` candidate
  buffer sized to the largest valid row.
- Each row uses `std::nth_element` and sorts only the selected K entries.
- Only int32 indices are written; no TopK values tensor, dtype conversion, or
  per-row tensor copy is created.
- NaN ranks above finite values and equal values use the lower local index.
- CPU contiguous-row FP32 scores and non-overlapping strided int32 outputs use
  the native path. Other tensor contracts retain the ATen row-wise fallback.
- Score generation and the full logits tensor are unchanged in this feature.

## Machine and build

- Host: `AmazonECS8Cores`
- CPU: eight physical Neoverse-V1 cores, CPUs `0-7`, one NUMA node
- Runtime SVE vector length: 256 bits; score tile `8 heads x 16 keys`
- Build: `-O2 -fopenmp -march=armv8.6-a+sve+bf16+i8mm`
- Affinity: `taskset -c 0-7`, `OMP_NUM_THREADS=8`, `OMP_DYNAMIC=FALSE`,
  `OMP_PROC_BIND=close`, `OMP_PLACES=cores`
- Source state: local working tree with the weighted-ReLU 8x2VL candidate and
  this uncommitted native TopK feature

Build command:

```bash
FUSED_CPP_SVE_VECTOR_BITS=256 MAX_JOBS=8 \
  .venv/bin/python setup.py build_ext --inplace
```

## Correctness

The standalone checker covers 20 batched TopK cases across empty rows,
`K={0,1,4,20}`, tied scores, NaN, partial row ranges, and stride-2 output. All
passed. Its existing 45 weighted-ReLU score shapes also remained unchanged:

```text
batched_topk_validation=passed cases=20
validation_cases=45 max_abs=9.53674e-07 max_rel=2.6939e-06
```

The integration command was:

```bash
PYTHONPATH=src OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
OMP_PROC_BIND=close OMP_PLACES=cores taskset -c 0-7 \
.venv/bin/python -m pytest -q \
tests/test_deepseek_v4_post_gemm_stage.py \
tests/test_sparse_attn_indexer.py
```

Result: `25 passed, 5 skipped in 0.56s`. The skips are intentionally retired
standalone `cpp_v0` cases. The long-path integration covers two requests and a
stride-2 TopK output view.

## Performance

Shape and method: BF16 Q/K, FP32 weights, `M=2048`, `H=64`, `D=128`, compressed
`N=1536`, `context_start=4096`, `topk=512`, 8 threads on cores `0-7`, three
warmups and 15 measured runs. Baseline and candidate use the same score kernel,
input generator, affinity, build flags, and measurement command:

```bash
PYTHONPATH=src OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
OMP_PROC_BIND=close OMP_PLACES=cores taskset -c 0-7 \
.venv/bin/python tests/bench_deepseek_v4_post_gemm_stage.py \
  --m 2048 --context-start 4096 --threads 8 \
  --warmup 3 --runs 15 --profile
```

| Metric | ATen row-wise baseline | Native batch | Change |
|---|---:|---:|---:|
| Full-stage median | 186.251 ms | 120.758 ms | -35.16%, 1.542x |
| Full-stage range | 184.244-186.937 ms | 119.443-121.979 ms | — |
| Profiled score | 36.154 ms | 36.158 ms | +0.01% |
| Profiled TopK | 75.155 ms | 9.637 ms | -87.18%, 7.799x |
| Profiled score + TopK | 111.310 ms | 45.795 ms | -58.86%, 2.431x |

A separate identical candidate run produced a 120.940 ms median, a
119.003-124.821 ms range, and 9.656 ms profiled TopK. The two candidate medians
differ by 0.15%.

## Decision and remaining work

The feature clears the predeclared local gate: exact selection passes, score
regression is below 2%, and full-stage latency improves by more than 5%. It is
retained as a combined candidate with the weighted-ReLU SVE score kernel.

No score/TopK fusion or invalid-key score pruning is included. Validation on
the 192-core/SVL128 machine remains required before adoption.
