# Sparse MLA Head-major 8x8 Experiments (2026-08-13)

> Adoption update (2026-08-14): the user approved `heads_sparse_8x8` as the
> production BF16 default. The named comparison selector was removed; the
> combined head-major mechanism now lives behind `flash_mla_sparse_fwd`.
> Guarded 2D indexed scheduling remains the short-chunk fallback, and the
> original measurements and checkpoint decisions below are retained verbatim
> as historical evidence.

## Scope

This experiment changes the QK 8x8 M dimension from eight query tokens of one
head to eight query heads of one token.  The production entrypoint and default
`indexed_4x4_2d` dispatch are unchanged; the candidates are reachable only via
the internal variant binding.

The first checkpoint covered the all-dense, shared-contiguous-index case through
`heads_dense_8x8`.  The second checkpoint adds `heads_sparse_8x8`: it packs one
token's selected K/V rows once, reuses them across every eight-head group, and
keeps one online-softmax state per head.  The combined candidate first attempts
the dense-head path, so globally shared contiguous indices retain one
call-level K/V pack instead of paying per-token sparse packing.

Primary change class: O (production optimization candidate).  Public API, ABI,
packed formats, numerical defaults, and default dispatch are unchanged.  The
rollback boundary is the internal variant, its tests, and benchmark selector.

## Method

- Source checkpoint before this experiment: `097d547`.
- The final sparse and combined tables used a release extension built with
  `FUSED_CPP_RELEASE=1`, `FUSED_CPP_ENABLE_PROFILING=0`, and
  `-O2 -march=armv8.6-a+sve+bf16+i8mm`.  Amazon used GCC 12.4; Arm-codex used
  GCC 13.2.  The earlier dense-only checkpoint table is retained as the first
  experimental checkpoint; it predates the explicit build-macro audit.  Its
  audited no-profiler repetition appears in the combined table below.
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
an 8-core win.  Do not change the public default.  The next checkpoint below
evaluates whether one token's sparse indices can be packed once and reused
across eight heads in the same online-softmax state.

## Sparse Head-major Checkpoint

### Implementation

`heads_sparse_8x8` accepts BF16 MQA shapes with `Hq % 8 == 0`, `Dqk % 4 == 0`,
and `Dv % 8 == 0`; unsupported shapes retain the indexed implementation.  It
uses one OpenMP task per token.  Within a task it:

1. packs each eight-head Q group once;
2. filters negative padding while preserving positive-index order and
   duplicates;
3. gathers and packs each selected K/V tile once, shared by every head group;
4. computes packed BFMMLA QK and BF16-probability PV in L2-sized chunks;
5. merges the chunks with the existing online-softmax max/sum/output state;
6. applies optional attention sinks and writes output only after normalization.

The internal combined variant tries `heads_dense_8x8` first.  Consequently,
the dense-shared case uses one call-level K/V pack and the sparse case uses
per-token gather/pack.  The public `flash_mla_sparse_fwd` entrypoint and its
default guarded `indexed_4x4_2d` dispatch remain unchanged.

### Sparse Method

- Workload: `dsv4-sparse`, BF16 `q[2048,32,192]`, `d_v=128`, top-k capacity
  640 (`512` compressed + `128` recent-window slots), `context_start=0`,
  `kv[2560,1,192]`.
- There are 777,792 valid KV pairs per head and 15,929,180,160 source FLOPs.
- Timed region includes input index conversion/validation, packing, QK,
  online softmax, PV, output conversion, and OpenMP scheduling.
- Each comparison used identical tensors (`seed=20260812`), `3` warmups and
  `11` interleaved rotating samples; the median is reported.
- `indexed_4x4_2d` is the public default candidate.  At `s_q=2048`, its KV
  split guard is inactive and it is statistically equivalent to the
  `indexed_4x4` reference.  Relative results below use `indexed_4x4` so raw
  values can be compared with earlier records.
- Amazon processes were bound to CPU 0 or CPUs 0-7.  Arm-codex processes were
  bound to CPU 0 or CPUs 0-79; no cross-NUMA memory binding was requested.

Representative command:

```bash
env PYTHONPATH=src OMP_NUM_THREADS=<threads> FUSED_CPP_NUM_THREADS=<threads> \
  OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  taskset -c <cores> .venv/bin/python \
  tests/bench_sparse_mla_tail_variants.py --workload dsv4-sparse \
  --s-q 2048 --context-start 0 --threads <threads> --warmup 3 --iters 11 \
  --variants indexed_4x4 indexed_4x4_2d heads_sparse_8x8
```

### Sparse Results

| Machine / affinity | Threads | indexed 4x4 | default 4x4 2D | combined head-major | Relative to indexed |
|---|---:|---:|---:|---:|---:|
| AmazonECS8Cores / CPU 0 | 1 | 360.442 ms | 360.281 ms | 251.737 ms | 1.4318x, -30.16% latency |
| AmazonECS8Cores / CPUs 0-7 | 8 | 55.664 ms | 55.325 ms | 38.682 ms | 1.4390x, -30.51% latency |
| Arm-codex-internal / CPU 0 | 1 | 384.633 ms | 385.627 ms | 327.617 ms | 1.1740x, -14.82% latency |
| Arm-codex-internal / CPUs 0-79 | 80 | 10.262 ms | 10.291 ms | 4.704 ms | 2.1815x, -54.16% latency |

Range across the 11 timed samples (minimum--maximum), shown to expose run
dispersion rather than only the medians:

| Machine / threads | indexed 4x4 | default 4x4 2D | combined head-major |
|---|---:|---:|---:|
| AmazonECS8Cores / 1T | 359.844--361.966 ms | 359.813--361.438 ms | 251.105--252.896 ms |
| AmazonECS8Cores / 8T | 55.441--55.752 ms | 55.177--55.700 ms | 38.515--38.725 ms |
| Arm-codex-internal / 1T | 383.326--387.477 ms | 379.914--388.774 ms | 324.328--328.161 ms |
| Arm-codex-internal / 80T | 10.206--10.717 ms | 10.221--10.532 ms | 4.694--5.083 ms |

Effective throughput for indexed versus head-major was 44.19 versus
63.28 GFLOP/s on one Amazon core, 286.17 versus 411.80 GFLOP/s on eight Amazon
cores, 41.41 versus 48.62 GFLOP/s on one Arm-codex core, and 1.552 versus
3.386 TFLOP/s on 80 Arm-codex cores.  The candidate's maximum absolute output
difference from indexed was 0.015625; it passed the suite's BF16
`atol=1e-2, rtol=1e-2` criterion.  Max-logit and LSE checks use
`atol=rtol=1e-5`.

The 80-thread speedup is larger than the one-thread kernel speedup because it
also changes scheduling granularity from 256 eight-token plans to 2,048
independent token tasks.  The per-token work grows with position in this
context-zero workload, so dynamic token scheduling avoids the coarse-plan load
imbalance.  On Amazon, the 8-thread ratio closely tracks the one-thread ratio,
which is consistent with both task counts providing enough parallel work.

### Dense + Sparse Combined Results

The same combined variant was compared with both the current token-major dense
path and the dense-only head-major variant using shared `[0,640)` indices:

| Machine / affinity | Threads | token-major | dense-head | combined head-major | Combined vs token-major |
|---|---:|---:|---:|---:|---:|
| AmazonECS8Cores / CPU 0 | 1 | 334.937 ms | 333.123 ms | 333.057 ms | 1.0056x, -0.56% latency |
| AmazonECS8Cores / CPUs 0-7 | 8 | 54.571 ms | 54.588 ms | 54.605 ms | 0.9994x, +0.06% latency |
| Arm-codex-internal / CPU 0 | 1 | 426.001 ms | 417.453 ms | 417.426 ms | 1.0205x, -2.01% latency |
| Arm-codex-internal / CPUs 0-79 | 80 | 7.318 ms | 6.022 ms | 6.014 ms | 1.2168x, -17.82% latency |

All dense outputs were bitwise equal (`max_abs=0`).  The combined and
dense-only medians are within 0.14%, demonstrating that the dense-first
dispatch preserves call-level K/V reuse.

| Machine / threads | token-major range | dense-head range | combined range |
|---|---:|---:|---:|
| AmazonECS8Cores / 1T | 334.711--335.863 ms | 332.796--334.024 ms | 332.636--333.644 ms |
| AmazonECS8Cores / 8T | 54.466--54.767 ms | 54.509--54.939 ms | 54.500--54.788 ms |
| Arm-codex-internal / 1T | 425.903--427.141 ms | 417.393--418.688 ms | 417.359--418.518 ms |
| Arm-codex-internal / 80T | 7.312--7.332 ms | 6.014--6.034 ms | 6.010--6.025 ms |

### Correctness and Diagnostics

- AmazonECS8Cores, CPUs 0-3: `43 passed` in
  `tests/test_sparse_mla.py`.
- Arm-codex-internal, CPUs 0-3: `43 passed` in the same suite.
- Both machines used runtime and build-time SVL256.  The named targets did not
  provide a runtime SVL128 configuration in this session.
- Coverage includes ragged and all-negative rows, duplicate/non-contiguous
  indices, positive/out-of-range validation, attention sinks, output-only and
  max/LSE modes, dense+sparse state sharing, dense-first dispatch, generic
  fallback shapes, and a 4,097-key row spanning multiple L2 chunks.
- Local macOS native build remained unavailable because this checkout lacks
  `refs/i8gemm/lib/bf16gemm_mt.c`; Python syntax checks passed locally.
- A one-call profiler run on Amazon attributed the head-major sparse candidate
  approximately as PV 37.9%, QK 26.4%, K/V gather+pack 16.4%, softmax 14.4%,
  and Q pack 1.5%.  These diagnostic percentages include profiler overhead and
  are not performance-table values.

### Decision

Keep `heads_sparse_8x8` experimental and leave the public default unchanged.
This checkpoint demonstrates material gains on the two requested machines and
retains the dense-head benefit, but the governance adoption gate calls for
three independent sessions and a representative sequence/head/thread sweep.
The rollback boundary is the internal variant enum/dispatch, executor, tests,
benchmark selector, manifest entry, and this result section.
