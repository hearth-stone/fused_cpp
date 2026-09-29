# Agent Benchmark Hygiene

Read this document before collecting, comparing, or reporting performance.

## Required Reporting

Record enough information to reproduce the result:

- repository commit and relevant uncommitted changes;
- machine, CPU affinity, NUMA and memory placement;
- compiler/build mode and relevant environment variables;
- operator, shape, dtype, layout, implementation, and backend;
- warmup count, sample count, synchronization, and reported statistic;
- baseline and candidate absolute values plus relative change;
- failed shapes, regressions, skipped checks, and measurement noise.

Do not mix debug and release builds, different input data, different page
policies, or different synchronization methods in one comparison.

## Fused MoE Geometry

Production fused MoE benchmark and analysis work must report:

- team width and backend N tile;
- full W13 and W2 stage bytes;
- per-thread owner window for each stage;
- `(t, w13_window_tiles, w2_window_tiles, R13, R2)`.

A stage is determined by `(threads, window_tiles)`, where `window_tiles` counts
whole packed-B N tiles per thread. A team consumes
`threads * window_tiles` tiles per window and covers the stage in:

```text
ceil(total_tiles / (threads * window_tiles))
```

windows; the last may be short. `window_tiles = 0` selects the full owner
stripe and the single-window `full_n_team_stripes` geometry. Use that name only
when both stages use full stripes.

Report runtime windows as tile counts, not bytes. Byte budgets may be converted
only by `FullStageGeometry.window_tiles_from_bytes`; byte values cannot identify
a unique execution geometry. Retired byte-window and split-W13 controls must not
be reintroduced. External N-range loops must be labelled experimental and must
not emit production calibration profiles.

## Single-Thread Benchmarks

Bind each process to one dedicated core and constrain dependent libraries:

```bash
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
taskset -c <core> <command>
```

On `Arm-codex-internal` / `Arm-codex`, independent benchmark cores are `0`,
`80`, `160`, and `240`. Use one process per core and report the selected core.

On `AmazonECS8Cores`, use one process per core from `0` through `7`. Example:

```bash
ssh AmazonECS8Cores 'cd /home/ubuntu/zhangxu/fused_cpp && OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 taskset -c 0 .venv/bin/python tests/bench_microkernel_qkt.py'
```

## Sanity Checks

Before comparing GFLOP/s with peak, read
`docs/agent_performance_references.md`. If measured throughput exceeds the
relevant peak, first verify thread count, OpenMP/runtime behavior, timing scope,
and the FLOP formula.

Performance conclusions require measured data. A successful build, profiler
estimate, analytical prediction, or one favorable sample is not a speedup.
