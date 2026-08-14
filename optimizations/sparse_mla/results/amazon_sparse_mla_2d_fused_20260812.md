# Sparse MLA fused epilogue and 2D KV scheduling

## Decision

Enable guarded 2D KV splitting in the public BF16 `flash_mla_sparse_fwd`
path. Keep the fused 8x8 online-softmax epilogue and its FMLA, BFMLAL, and
BFMMLA PV backends as experimental comparison paths.

The adopted scheduler is useful for short query chunks against a long KV
history: on the 192-core host using cores 96--191, a 64-token chunk ending at
context 4096 improves from 13.600 to 2.558 ms (5.317x). It automatically does
no split when query-block parallelism is already sufficient: full 4096 and
8192 prefills differ from the 1D reference by -0.22% and +0.19%, respectively.

There is no universal winner among the fused PV backends. For the isolated
causal-tail workload, SVL256 favors fused FMLA, while SVL128 favors the existing
materialized pruned kernel. Fused BFMMLA is valid on both tested SVLs but is not
the fastest on either, so it should not replace FMLA or the materialized path.

## Implementation summary

### Fused 8x8 epilogue

The fused candidates preserve the existing online-softmax algorithm. They keep
one pruned 8x8 QK tile in SVE registers, update the running `(max, sum)` state,
round the local probabilities to BF16, and accumulate PV without writing the
intermediate score or `p_hat` tiles to memory. The cross-tile FP32 `O` state is
still loaded and stored by each tile; only the final normalized BF16 output is
written once after all sparse-plan components have been processed.

Three PV implementations share the same fused QK and softmax code:

- `masked_dense_8x8_fused_fmla`: widen BF16 V to FP32 and use SVE FMLA;
- `masked_dense_8x8_fused_bfmlal`: use BF16 lane widening multiply-add;
- `masked_dense_8x8_fused_bfmmla`: pack P as 2x4 and V as 4x2 matrices for
  BFMMLA, then deinterleave the 2x2 result.

The BFMMLA layout is explicitly supported for SVL128 and SVL256. Other SVE
vector lengths reject the fused helper and use the materialized fallback. This
is deliberate: only the two target layouts have been built and validated.

### 2D KV split

The scheduler first constructs the normal eight-query `BlockPlan`. If the
number of plans is no more than half the requested thread count, it greedily
adds KV shards to the heaviest plans until the worker pool is filled or each
plan reaches eight shards. Dense segments are split on eight-key boundaries;
masked and indexed tiles remain indivisible.

Each shard produces an unnormalized local online-softmax state:

```text
m_s = max score in shard s
l_s = sum(exp(score - m_s))
O_s = sum(exp(score - m_s) * V)
```

The merge is associative:

```text
m = max(m_a, m_b)
l = l_a * exp(m_a - m) + l_b * exp(m_b - m)
O = O_a * exp(m_a - m) + O_b * exp(m_b - m)
```

`attn_sink` is applied once after all shards are merged. Real-key `max_logits`
and `lse` are merged separately so sink mass does not change their semantics.
When no plan is split, partial buffers and the merge OpenMP region are skipped.

The planner also recognizes the common intersection of sliding recent-window
runs as a dense segment. This avoids representing the 128-key trapezoid as a
large set of indexed 4x4 tiles; its asymmetric fringes remain indexed or masked.

## Workload model

`tests/bench_sparse_mla_tail_variants.py --workload dsv4-sparse` models two
disjoint sources:

- a compress-ratio-4 cache with at most 512 sorted, non-contiguous selected
  entries (the 4a path is not another triangle);
- a 128-token recent window that grows initially and then slides at constant
  width (the full-sequence view is trapezoidal).

The compressed and recent caches occupy disjoint index ranges. The benchmark
uses BF16 `h_q=32`, `d_qk=192`, `d_v=128`, fixed seed 20260812, preallocated
outputs, 5 warmups, 21 measured calls, rotating interleaved variant order, and
reports median wall time. The timed region includes input-index conversion,
planning, packing, partial-state allocation, computation, and merge.

The 192-core host had 20 GiB of 32 MiB HugeTLB pages configured on each NUMA
node. These tensors were ordinary PyTorch allocations and were not explicitly
HugeTLB-backed, so the measurements must not be interpreted as HugeTLB-backed
operator results.

All FLOP/s values are source-work accounting:

```text
2 * valid_pairs * h_q * (d_qk + d_v)
```

They are not hardware-peak utilization because pruning, planning, softmax, and
packing work are not represented by that formula.

## 192-core host results

Placement for every result in this section:

```bash
numactl --physcpubind=96-191 --membind=1 env \
  PYTHONPATH=src OMP_NUM_THREADS=96 OMP_DYNAMIC=FALSE \
  OMP_PROC_BIND=close OMP_PLACES=cores <command>
```

The extension was built from HEAD `5bca97a` plus the uncommitted changes in
this report using:

```bash
FUSED_CPP_RELEASE=1 MAX_JOBS=1 FUSED_CPP_SVE_VECTOR_BITS=128 \
  .venv/bin/python setup.py build_ext --inplace
```

### Short chunks ending at context 4096

All rows use top-k 640 (`512 + 128`) and the same context endpoint. The output
maximum absolute difference between 2D and 1D is 0.00195312.

| Query chunk | Context start | Valid pairs/head | 1D ms | 2D ms | Speedup | 2D source GFLOP/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 4032 | 40,960 | 13.600 | 2.558 | 5.317x | 327.95 |
| 128 | 3968 | 81,920 | 13.876 | 3.584 | 3.872x | 468.05 |
| 256 | 3840 | 163,840 | 14.657 | 6.439 | 2.276x | 521.10 |

Command template:

```bash
.venv/bin/python tests/bench_sparse_mla_tail_variants.py \
  --workload dsv4-sparse --s-q <64|128|256> \
  --context-start <4032|3968|3840> --h-q 32 --d-qk 192 --d-v 128 \
  --compressed-capacity 512 --window-size 128 --compress-ratio 4 \
  --threads 96 --warmup 5 --iters 21 \
  --variants indexed_4x4 indexed_4x4_2d
```

### Full prefill lengths

At 2048, 4096, and 8192 tokens, query-block count exceeds half of 96 workers;
the guarded 2D variant therefore runs the original 1D schedule. Outputs are
bitwise equal and latency differences are within 0.22%.

| `s_q` | KV rows | Valid pairs/head | 1D ms | Guarded 2D ms | Latency change | 2D source GFLOP/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 2,560 | 777,792 | 4.818 | 4.824 | +0.12% | 3,302.24 |
| 4096 | 5,120 | 2,088,512 | 51.269 | 51.156 | -0.22% | 836.13 |
| 8192 | 10,240 | 4,709,952 | 138.608 | 138.866 | +0.19% | 694.62 |

The 2048 distributions contained a temporary 4.8/7.1 ms frequency or system
state shift in both variants; their medians still matched. The stable 4096 and
8192 distributions support the no-regression conclusion more strongly.

## 8-core host results

The 8-core extension was built with `FUSED_CPP_SVE_VECTOR_BITS=256`; benchmarks
used `taskset -c 0-7`, eight OpenMP threads, close binding, and the same BF16
shape, seed, warmup, and sample count.

| Workload | 1D ms | Guarded 2D ms | Change | Max abs |
| --- | ---: | ---: | ---: | ---: |
| 64 tokens ending at 4096 | 26.278 | 26.298 | +0.08% | 0 |
| Full 2048 prefill | 57.020 | 56.837 | -0.32% | 0 |

For 64 tokens there are eight query blocks, which is more than half of the
eight-worker pool; the threshold correctly avoids an unnecessary split.

## FMLA, BFMLAL, and BFMMLA PV comparison

This comparison isolates 1,024 independent causal 8x8 tails. It uses one
thread, BF16 `q[8192,32,192]`, `kv[8192,1,192]`, `d_v=128`, 5 warmups, and 21
rotating interleaved samples. All candidates have max absolute BF16 difference
0.015625 versus indexed 4x4.

| Host / SVL | Materialized pruned | Fused FMLA | Fused BFMLAL | Fused BFMMLA | Best |
| --- | ---: | ---: | ---: | ---: | --- |
| 192-core, CPU 96, SVL128 | 27.361 ms | 29.532 ms | 28.267 ms | 46.403 ms | materialized |
| 8-core, CPU 0, SVL256 | 55.238 ms | 45.385 ms | 49.876 ms | 49.505 ms | fused FMLA |

On SVL256, eliminating score and `p_hat` scratch traffic is worthwhile: fused
FMLA is 17.84% faster than materialized pruned. On SVL128, the same fused FMLA
is 7.93% slower, and even BFMLAL is 3.31% slower. BFMMLA needs probability
replication, a second V layout, 2x2 output interleave/deinterleave, and works on
only four K values per matrix instruction; those fixed costs dominate the
nominal instruction reduction. It is 69.60% slower than materialized on
SVL128 and 10.38% faster on SVL256, but still 9.08% slower than fused FMLA on
that machine.

Command template:

```bash
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
.venv/bin/python tests/bench_sparse_mla_tail_variants.py \
  --workload tail-causal --tail-blocks 1024 --h-q 32 --d-qk 192 \
  --d-v 128 --threads 1 --warmup 5 --iters 21 \
  --variants indexed_4x4 masked_dense_8x8_pruned \
  masked_dense_8x8_fused_fmla masked_dense_8x8_fused_bfmlal \
  masked_dense_8x8_fused_bfmmla
```

## Correctness and build validation

Both target builds completed successfully after the final formatting and SVL
guard changes. The focused test includes all recognized masked-tail patterns,
generic and boundary fallbacks, sliding-window intersection, output-only fused
epilogues, and a real 2D split/merge with `return_stats=True` and `attn_sink`.

```text
AmazonC5192Cores, CPUs 96-99, NUMA1, SVL128: 34 passed in 0.35s
AmazonECS8Cores, CPUs 0-3, SVL256:          34 passed in 0.65s
```

The 8-core run emitted one environment warning because NumPy is not installed;
the sparse MLA tests passed and do not depend on NumPy.

Static checks:

```text
python3 -m py_compile tests/test_sparse_mla.py \
  tests/bench_sparse_mla_tail_variants.py
git diff --check
git-clang-format --force --style Google HEAD -- \
  csrc/sparse_mla.cpp csrc/sparse_mla_tail_microkernels.cpp \
  csrc/sparse_mla_tail_microkernels.h
```

## Residual risks and next optimization directions

1. **Move planning out of the hot call.** The index pattern is metadata. Cache
   `BlockPlan`, shard assignment, and packed layout across repeated forwards or
   expose an opaque prepared plan; measure lifecycle and invalidation first.
2. **Calibrate the split policy.** The current half-pool threshold and eight-
   shard cap are robust on the measured shapes, but production chunk-size and
   KV-length distributions could support a `(query_blocks, KV_work, threads)`
   cost model.
3. **Parallelize or fuse the merge only when it matters.** For very small query
   chunks there are few merge-plan tasks. A plan/head merge grid or final-shard
   owner merge may help, but must be compared against a second OpenMP-region
   cost and partial-buffer traffic.
4. **Reduce duplicate packing among KV shards.** Dense K/V shards currently
   pack independently. A prepared, read-only packed KV cache could remove this
   cost and make 2D splitting useful at larger query-block counts.
5. **Do not prioritize BFMMLA PV next.** Its matrix shape and packing overhead
   are a mismatch for K=8. If fused epilogues are adopted later, use an SVL-
   aware choice between materialized and FMLA; add `return_stats` support before
   making any fused variant production-facing.
6. **Validate with vLLM metadata and end-to-end TTFT.** This report uses a
   structurally faithful synthetic DSV4 source. The next gate is real indexer
   output, real prefill chunk distributions, and TTFT rather than another
   isolated kernel-only shape sweep.
