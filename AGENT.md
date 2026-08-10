# Agent Guide

Keep this file small: it is loaded by default. Large reference data and detailed
optimization governance live in on-demand docs listed below.

## Remote Execution

Remote development and benchmark runs primarily use the `Arm-codex` SSH alias.
When the user says `aws机器`, use the `AmazonECS8Cores` SSH alias.

If a requested remote instance cannot be reached, stop the task immediately
and report the connection failure. Do not switch to another machine, continue
with local implementation or analysis, or perform unrelated fallback work.

### Arm-codex

- Host alias: `Arm-codex`
- User: `zhangxu`
- Remote project root: `/home/zhangxu/codex/fused_cpp`
- Local project root: this repository root
- Python executable: `/home/zhangxu/codex/fused_cpp/.venv/bin/python`
- Package manager: `uv`

Sync local files before remote tests:

```bash
bash rsync.sh
```

Example remote Python check:

```bash
ssh Arm-codex 'cd /home/zhangxu/codex/fused_cpp && .venv/bin/python -c "from fused_cpp import _C; print(_C.has_openmp())"'
```

Avoid relying on remote system `python`; it may not match the virtualenv ABI used
to build `fused_cpp._C`. Install missing benchmark or analysis packages with
project-local `uv`, for example:

```bash
ssh Arm-codex 'cd /home/zhangxu/codex/fused_cpp && uv pip install <package>'
```

### AWS Machine

This is the target machine whenever the user says `aws机器`.

- Host alias: `AmazonECS8Cores`
- Remote project root: `/home/ubuntu/zhangxu/fused_cpp`
- Python activate script: `/home/ubuntu/zhangxu/fused_cpp/.venv/bin/activate`
- Python executable: `/home/ubuntu/zhangxu/fused_cpp/.venv/bin/python`
- Benchmark cores: bind benchmark processes to cores `0` through `7`

Sync local files to the AWS machine with:

```bash
bash rsync_aws.sh
```

Example:

```bash
ssh AmazonECS8Cores 'cd /home/ubuntu/zhangxu/fused_cpp && . .venv/bin/activate && python -c "import sys; print(sys.executable)"'
```

### Amazon C8i 2 Cores

- Host alias: `AmazonC8i2Cores`
- Remote work root: `/home/ubuntu/zhangxu`

### Amazon C5 64 Cores

- Host alias: `AmazonC564Cores`
- Remote work root: `/home/ubuntu/zhangxu`
- Benchmark cores: bind benchmark processes to cores `0` through `31`

### Amazon C5 192 Cores

- Host alias: `AmazonC5192Cores`
- Remote work root: `/data/zhangxu`
- Remote project root: `/data/zhangxu/fused_cpp`
- Python executable: `/data/zhangxu/fused_cpp/.venv/bin/python`
- The work root moved off `/home/ubuntu/zhangxu` on 2026-08-10 because the root
  filesystem is only 8.7 GiB and filled up. `/data` is a separate 49 GiB NVMe
  volume. Do not recreate anything under `/home/ubuntu`; that path no longer
  exists.
- The remote tree is an rsync mirror, not a git checkout. A leftover empty `.git`
  directory makes `git -C` commands fail rather than report a clean tree, so
  never infer remote state from git. `rsync -a` also does not delete, so a file
  removed locally lingers remotely; delete stale files explicitly after a
  refactor, and remember `rsync -a` preserves mtime, which makes a rebuild
  silently skip a reverted source unless it is `touch`ed first.
- CPU topology: 192 CPUs across two NUMA nodes, `0-95` and `96-191`
- Explicit HugeTLB benchmark pool: 320 x 32 MiB pages (10 GiB total),
  distributed as 160 pages per NUMA node and mounted at
  `/dev/hugepages-32M`.
- Fused MoE benchmarks on this host default to
  `FUSED_CPP_MOE_HUGETLBFS_PATH=/dev/hugepages-32M`; report explicit 32 MiB
  HugeTLB backing with the result. Unset the variable only for a named 4 KiB
  baseline comparison.
- Keep benchmark processes NUMA-local unless the test explicitly covers both
  nodes.

## Benchmark Hygiene

Production fused MoE benchmark and analysis work must report the stage geometry,
the team width, the backend N tile, the full W13/W2 stage bytes, and the per-thread
owner window for each stage.

A stage's computation pattern is determined by the pair `(threads, window_tiles)`,
where `window_tiles` counts whole packed-B N tiles per thread. The team consumes
`threads * window_tiles` tiles per window and covers the stage in
`ceil(total_tiles / (threads * window_tiles))` windows, the last of which may be
short. `window_tiles = 0` selects the full stripe, which is the single-window
`full_n_team_stripes` geometry and the default; report that name only when the
window is the full stripe for both stages.

Report windows as tile counts, not bytes. The runtime ABI carries tile counts and
`FullStageGeometry.window_tiles_from_bytes` is the only place a byte budget may be
converted, so a byte-denominated window in a report cannot be mapped back to a
unique pattern. Quote `(t, w13_window_tiles, w2_window_tiles, R13, R2)`.

The retired controls are the **byte-window** and **split-W13** environment and plan
fields, which were ambiguous: several byte budgets mapped to one pattern, and a
split and a range could express the same geometry. Those must not come back. The
tile-counted per-thread window above is not one of them; it is the parameter the
planner selects and the runtime executes.

Benchmarks that retain their own N-range loop outside this mechanism must label it
experimental and must not emit production calibration profiles.

For single-thread microbenchmarks, bind each process to one dedicated core with
`taskset` and pin Python/native libraries to one thread:

```bash
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
taskset -c 80 <command>
```

On `Arm-codex`, independent benchmark cores are `0`, `80`, `160`, and `240`.
Use one benchmark process per core, record the core id, and do not put two
single-thread benchmarks on the same core.

On `AmazonECS8Cores`, use one process per core from `0` to `7`, for example:

```bash
ssh AmazonECS8Cores 'cd /home/ubuntu/zhangxu/fused_cpp && OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 taskset -c 0 .venv/bin/python tests/bench_microkernel_qkt.py'
```

When comparing measured GFLOP/s with peak, read
`docs/agent_performance_references.md` and first check thread count, OpenMP
runtime, or benchmark FLOP formulas if a result exceeds the relevant peak.

## Optimization Log

Record every optimization attempt in `csrc/SDPA_VERSIONS.md`, even when the
result does not improve performance. Include date, kernel/version, hypothesis,
exact benchmark command, core id, before/after numbers, and conclusion.

## On-Demand References

Read these only when the task needs them:

- `docs/agent_performance_references.md`: Arm-codex and AWS peak FLOPs, cache,
  frequency, and instruction reference tables for benchmark sanity checks.
- `docs/agent_optimization_governance.md`: repository organization rules for
  optimization work, including legacy boundaries, manifests, feature/variant
  naming, dispatch/default behavior, validation, benchmarks, and reporting.

For operator/kernel optimization tasks, read
`docs/agent_optimization_governance.md` before editing. For ordinary bug fixes or
small code changes, do not load the long on-demand docs unless the work touches
performance references, optimization manifests, feature/variant organization, or
default kernel dispatch.

## Model Configs

Model-related configuration files live under
`/Users/zhangxu/Codes/vllm-aarch64/models`.

## Always-On Repo Rules

- Before code changes, check `git status --short` or equivalent and preserve
  user changes.
- Keep legacy API, ABI, build entrypoints, and default dispatch stable unless the
  user explicitly asks for a change.
- Correctness comes before performance; never claim unrun tests passed.
- Do not commit, push, rebase, or force-push unless the user asks.
