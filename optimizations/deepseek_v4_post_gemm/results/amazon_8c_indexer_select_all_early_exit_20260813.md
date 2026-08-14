# Amazon 8C DeepSeek V4 Indexer Select-All Early Exit

## Outcome

The existing sparse-indexer select-all condition is now evaluated before the
Main/Indexer Q scheduling decision. When every row has
`valid_len <= topk_tokens`, the stage runs Main Q and both compressors but skips
the Indexer Q GEMM and Indexer Q RoPE/weight scaling. The TopK buffer is still
written after both compressors.

For `M=2048`, `context_start=0`, and `topk=512` on the requested Amazon 8-core
machine, the full-stage median fell from 74.309 ms to 40.176 ms: a 45.93%
latency reduction and 1.850x speedup. The profiled 34.854 ms Indexer Q GEMM plus
RoPE/weight work was eliminated.

## Semantics and dispatch

- The early plan converts `cu_seqlen_ks` and `cu_seqlen_ke` to contiguous CPU
  int64 tensors once and computes the maximum valid row length.
- A positive-token call selects the early path only when every row satisfies
  `cu_seqlen_ke[i] - cu_seqlen_ks[i] <= topk_tokens`.
- Main Q GEMM, Main Q norm/RoPE/SWA insertion, and both MLA and Indexer
  compressor updates are unchanged.
- Raw-weight and prepacked entrypoints use the same plan. The prepacked path
  bypasses the shared Main/Indexer Q pool and invokes the Main-Q-only path.
- The final select-all writer and its fallback are unchanged in meaning and
  remain after compressor execution. Calls outside the condition retain the
  existing score and exact-TopK path.
- Public signatures, packed weights, output layout, dtype contracts, and
  numerical behavior are unchanged.

## Machine and build

- Host: `AmazonECS8Cores`
- CPU placement: cores `0-7`
- Runtime SVE vector length: 256 bits
- Build: `-O2 -fopenmp -march=armv8.6-a+sve+bf16+i8mm`
- Affinity: `taskset -c 0-7`, `OMP_NUM_THREADS=8`, `OMP_DYNAMIC=FALSE`,
  `OMP_PROC_BIND=close`, `OMP_PLACES=cores`
- Source baseline: commit `d998350` plus the pre-existing unrelated working-tree
  changes; candidate adds only the early-dispatch implementation, tests, and
  optimization records described here.
- Page policy: no HugeTLB-specific environment variable was set for either
  side of the comparison; baseline and candidate used the same allocator and
  page policy.

Build command:

```bash
FUSED_CPP_SVE_VECTOR_BITS=256 MAX_JOBS=8 \
  .venv/bin/python setup.py build_ext --inplace
```

## Correctness

The two focused tests force the select-all condition through the raw-weight
fallback and through the prepacked entrypoint with the shared-Q pool forced on.
They compare Q, TopK, SWA cache, MLA compressor state/cache, and Indexer
compressor state/cache against the independent Torch baseline. Profile output
must report zero for shared-Q GEMM, Indexer Q GEMM, and Indexer Q RoPE/weights.

```bash
PYTHONPATH=src OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
OMP_PROC_BIND=close OMP_PLACES=cores taskset -c 0-7 \
.venv/bin/python -m pytest -q \
tests/test_deepseek_v4_post_gemm_stage.py -k select_all_short_path
```

Result: `2 passed, 18 deselected in 0.48s`.

The complete post-GEMM file also exercises the non-select-all score/TopK path,
multi-request paged packing, strided TopK output, M8 scheduling, shared Q pool,
and MN-group validation:

```text
20 passed in 0.54s
```

Both test commands emitted only the known warning that NumPy is not installed
in the remote virtual environment.

## Performance

Shape and method: BF16 post-GEMM C4A stage, `M=2048`, `context_start=0`,
`topk=512`, NEON Q-GEMM backend, eight threads, three warmups, and 15 measured
runs. The benchmark reports the median and min/max wall-clock samples. Baseline
and candidate used the same input generator, build flags, CPU placement,
environment, and command:

```bash
PYTHONPATH=src FUSED_CPP_POST_GEMM_BACKEND=neon \
OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores \
taskset -c 0-7 .venv/bin/python \
tests/bench_deepseek_v4_post_gemm_stage.py \
  --m 2048 --context-start 0 --threads 8 --warmup 3 --runs 15 \
  --schedule m8 --q-pool auto --profile
```

| Metric | Before early exit | Early exit | Change |
|---|---:|---:|---:|
| Full-stage median | 74.309 ms | 40.176 ms | -45.93%, 1.850x |
| Full-stage range | 73.611-75.125 ms | 39.112-40.345 ms | disjoint |
| Profiled total | 74.135 ms | 39.074 ms | -47.29%, 1.897x |
| Main Q GEMM | 33.429 ms | 32.989 ms | -1.32% |
| Indexer Q GEMM | 34.063 ms | 0.000 ms | eliminated |
| Indexer Q RoPE/weights | 0.791 ms | 0.000 ms | eliminated |
| Indexer compressor norm/RoPE/insert | 4.291 ms | 4.561 ms | +6.29% |
| Sparse select-all writer | 0.038 ms | 0.038 ms | unchanged |

The eliminated 34.854 ms represented 47.01% of the baseline profiled total.
The wall-clock improvement agrees with that removed work. This benchmark does
not claim a long-path speedup; long-path behavior is protected by the complete
correctness file.

## Decision and remaining validation

The feature is enabled in Production because it hoists an already-established
select-all decision and writer, adds no ISA-specific computation, preserves
cache side effects, and has direct raw/prepacked plus long-path coverage. The
rollback boundary is the prefill plan, the two entrypoint branches, and the two
new focused tests in `deepseek_v4_post_gemm_stage.cpp` and its test file.

Validation was also attempted on `AmazonC5192Cores`, but the host did not return
even a read-only `stat` over SSH and the check was stopped rather than silently
switching machines. Therefore no 192-core/SVL128 result is claimed in this
record; rerunning the same complete test file on NUMA1 cores `96-191` remains an
operational validation gap.
