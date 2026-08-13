# Sparse MLA Head-major 8x8 Experiments (2026-08-13)

## Scope

This experiment changes the QK 8x8 M dimension from eight query tokens of one
head to eight query heads of one token.  The production entrypoint and default
`indexed_4x4_2d` dispatch are unchanged; the candidates are reachable only via
the internal variant binding.

The first checkpoint covers the all-dense, shared-contiguous-index case through
`heads_dense_8x8`.  A later checkpoint in this result file will cover mixed
sparse/dense indices through `heads_sparse_8x8`.

Primary change class: O (production optimization candidate).  Public API, ABI,
packed formats, numerical defaults, and default dispatch are unchanged.  The
rollback boundary is the internal variant, its tests, and benchmark selector.

## Method

- Source checkpoint before this experiment: `097d547`.
- Build: release extension, GCC 13.2, `-O2 -march=armv8.6-a+sve+bf16+i8mm`.
- Both target processes reported a 256-bit runtime SVE vector length through
  `prctl(PR_SVE_GET_VL)`; builds used `FUSED_CPP_SVE_VECTOR_BITS=256`.
- Shape: BF16 `q[2048,32,192]`, `kv[640,1,192]`, `d_v=128`; every query uses
  the same contiguous indices `[0,640)`.
- Timed region includes index validation, K/V packing, Q packing, online
  softmax, PV, and output conversion.
- `3` warmups and `11` interleaved rotating samples; median reported.
- Threading libraries were constrained to the requested OpenMP thread count.
- `indexed_4x4` is the current token-major dense fast path for this workload;
  `heads_dense_8x8` is the head-major candidate.

Representative command (replace host, affinity, and thread count as listed in
the table):

```bash
env PYTHONPATH=src OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE \
  OMP_PROC_BIND=close OMP_PLACES=cores MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 taskset -c 0 \
  .venv/bin/python tests/bench_sparse_mla_tail_variants.py \
  --workload dense-shared --s-q 2048 --dense-kv 640 --h-q 32 \
  --d-qk 192 --d-v 128 --threads 1 --warmup 3 --iters 11 \
  --variants indexed_4x4 heads_dense_8x8
```

## Dense Results

| Machine / affinity | Threads | token-major | head-major | Relative result |
|---|---:|---:|---:|---:|
| AmazonECS8Cores / CPU 0 | 1 | 335.131 ms | 332.821 ms | head-major 1.0069x, -0.69% latency |
| AmazonECS8Cores / CPUs 0-7 | 8 | 54.648 ms | 54.658 ms | head-major 0.9998x, +0.02% latency |
| Arm-codex-internal / CPU 0 | 1 | 421.224 ms | 423.739 ms | head-major 0.9941x, +0.60% latency |
| Arm-codex-internal / CPUs 0-79 (NUMA 0) | 80 | 7.325 ms | 5.936 ms | head-major 1.2340x, -18.96% latency |

All output comparisons were bitwise equal in the benchmark (`max_abs=0`).
Source FLOPs are 26,843,545,600, giving 63.73 versus 63.35 GFLOP/s at one
Arm-codex core and 3.665 versus 4.522 TFLOP/s at 80 cores.

The one-core result shows that changing which rows occupy M does not make the
8x8 arithmetic itself faster.  The Arm-codex 80-core gain instead comes from
scheduling granularity: token-major has 256 eight-token OpenMP tasks, whereas
head-major has 2,048 one-token tasks.  Static assignment over 80 workers gives
the former an unavoidable 3-versus-4 heavy-task imbalance, while the latter is
25-versus-26.  The 8-core machine divides both 256 and 2,048 evenly, so that
load-balance advantage disappears and the variants tie.

## Correctness

- AmazonECS8Cores, CPUs 0-3, 256-bit SVE build:
  `PYTHONPATH=src OMP_NUM_THREADS=4 ... pytest -q tests/test_sparse_mla.py`
  -> `37 passed`.
- Arm-codex-internal, CPUs 0-3, 256-bit SVE build: same focused suite ->
  `37 passed`.
- The targeted tests cover output, optional max/LSE statistics, a non-zero
  contiguous run start, and exact fallback for non-contiguous indices.

The local macOS extension was not rebuilt because this checkout lacks
`refs/i8gemm/lib/bf16gemm_mt.c`; Python syntax validation was still run locally.

## Decision

Keep `heads_dense_8x8` experimental.  It is valuable as a high-core-count
scheduling candidate, but it does not establish a single-thread compute win or
an 8-core win.  Do not change the public default.  Next, evaluate whether one
token's sparse indices can be packed once and reused across eight heads in the
same online-softmax state.
