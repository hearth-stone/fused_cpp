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

## Machine Sharing And Other NUMA Nodes

`Arm-codex` has four 80-core NUMA nodes (node0 `0-79`, node1 `80-159`, node2
`160-239`, node3 `240-319`) over two sockets, so nodes 2 and 3 share a socket.
Measurements run on node3 with `--physcpubind=240-319 --membind=3`; other work
often runs at the same time. What that costs was measured on 2026-09-20 with the
unchanged plan benchmark on node3 (22 points, two sessions per condition,
`tmp/numa_interference_20260920/decision.md`), background placed only on the
other nodes:

| background on other nodes | node3 plan time |
| --- | --- |
| BF16 matmul, one node at 3.9 TFLOP/s | +0.01% (node2), +0.24% (node1) |
| memory streaming, one node at 266 GB/s | +0.83% (node2), +1.84% (node1) |
| memory streaming, three nodes | +13.1% |

Rules that follow:

- Compute-bound work - builds, tests, planner searches, analysis - may run on
  nodes 0-2 during a measurement on node3.
- Memory-streaming work must not overlap a measurement, wherever it is placed:
  large copies, dataset generation, archive extraction, another MoE benchmark.
  Sequence it before or after, or accept and report the contamination.
- The 1-minute load-average idle guard stays the cheap pre-session check, but it
  means "no foreign work of unknown kind", not "other nodes are harmless".
- A measurement whose cores are shared with a foreign job is not recoverable by
  any of this. The E6 r022 sessions measured 79-100 ms against 27-28 ms (+190%),
  far beyond anything foreign nodes cause, and were discarded.

Node-3 core frequency stays at its 2900 MHz maximum under the heaviest of these
backgrounds, so the effect is memory-path contention beyond the node, not power
or frequency; the reason node1 (other socket) costs more than node2 (same
socket) is not identified.

## Sanity Checks

Before comparing GFLOP/s with peak, read
`docs/agent_performance_references.md`. If measured throughput exceeds the
relevant peak, first verify thread count, OpenMP/runtime behavior, timing scope,
and the FLOP formula.

Performance conclusions require measured data. A successful build, profiler
estimate, analytical prediction, or one favorable sample is not a speedup.
