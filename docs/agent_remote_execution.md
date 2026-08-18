# Agent Remote Execution

Read this document before remote builds, tests, or benchmarks. The local Git
checkout is the source of truth unless a machine entry explicitly says
otherwise.

## General Rules

- Remote development and benchmark runs primarily use the
  `Arm-codex-internal` SSH alias. `Arm-codex` is the alternate alias for the
  same machine. When the user says `aws机器` without naming another host, use
  `AmazonECS8Cores`.
- If a requested remote instance cannot be reached, stop the task immediately
  and report the connection failure. Do not silently switch machines or treat a
  local run as equivalent.
- Use the remote project's virtualenv Python. Do not rely on system `python`.
- Sync local files before remote validation. Preserve unrelated remote data and
  remove stale mirrored source explicitly when a local refactor deletes files.

## Arm-codex-internal / Arm-codex

- Preferred host alias: `Arm-codex-internal`
- Alternate host alias: `Arm-codex`
- User: `zhangxu`
- Remote project root: `/home/zhangxu/code`
- Local project root: repository root
- Python: `/home/zhangxu/code/.venv/bin/python`
- Package manager: `uv`

The aliases name the same machine. Prefer `Arm-codex-internal` on the internal
network; use `Arm-codex` when that is the configured reachable alias. Commands
below use the preferred alias.

The fused_cpp project root moved from `/home/zhangxu/codex/fused_cpp` to
`/home/zhangxu/code` on 2026-08-12. Do not recreate or use the old project
path. Populate the new root with `rsync.sh` before its first build, then create
or migrate the project virtual environment there as needed.

Sync with:

```bash
bash rsync.sh
```

Example check:

```bash
ssh Arm-codex-internal 'cd /home/zhangxu/code && .venv/bin/python -c "from fused_cpp import _C; print(_C.has_openmp())"'
```

Install missing benchmark dependencies into the project environment:

```bash
ssh Arm-codex-internal 'cd /home/zhangxu/code && uv pip install <package>'
```

## Amazon ECS 8 Cores

This is the default host for an unqualified `aws机器` request.

- Host alias: `AmazonECS8Cores`
- Remote project root: `/home/ubuntu/zhangxu/fused_cpp`
- Activate script: `/home/ubuntu/zhangxu/fused_cpp/.venv/bin/activate`
- Python: `/home/ubuntu/zhangxu/fused_cpp/.venv/bin/python`
- Benchmark cores: `0-7`

Sync with:

```bash
bash rsync_aws.sh
```

Example check:

```bash
ssh AmazonECS8Cores 'cd /home/ubuntu/zhangxu/fused_cpp && . .venv/bin/activate && python -c "import sys; print(sys.executable)"'
```

## Amazon C8i 2 Cores

- Host alias: `AmazonC8i2Cores`
- Remote work root: `/home/ubuntu/zhangxu`

## Amazon C5 64 Cores

- Host alias: `AmazonC564Cores`
- Remote work root: `/home/ubuntu/zhangxu`
- Benchmark cores: `0-31`

## Amazon C5 192 Cores

- Host alias: `AmazonC5192Cores`
- Remote work root: `/data/zhangxu`
- Remote project root: `/data/zhangxu/fused_cpp`
- Python: `/data/zhangxu/fused_cpp/.venv/bin/python`
- CPU topology: NUMA0=`0-95`, NUMA1=`96-191`
- Required work cores: `96-191` (NUMA1); smaller CPU sets must be subsets of
  this range
- Required memory node: NUMA1 (`1`)
- HugeTLB mount: `/dev/hugepages-32M`
- HugeTLB pool: 1,280 x 32 MiB pages (40 GiB total), 640 pages (20 GiB) per
  NUMA node
- HugeTLB persistence: runtime-only; the current boot configuration does not
  restore this pool after a reboot

The project moved from `/home/ubuntu/zhangxu` to `/data/zhangxu` on
2026-08-10. Do not recreate the old path; the root filesystem is too small.

The remote tree is an rsync mirror, not a Git checkout:

- Never infer remote source state from `git`.
- `rsync -a` does not delete files, so remove stale mirrored source after local
  deletions.
- `rsync -a` preserves mtimes; force the affected object to rebuild when a
  reverted source would otherwise appear unchanged.

Fused MoE benchmarks default to:

```bash
FUSED_CPP_MOE_HUGETLBFS_PATH=/dev/hugepages-32M
```

Bind routine builds, tests, benchmarks, and profilers to NUMA1:

```bash
numactl --physcpubind=96-191 --membind=1 <command>
```

Report the 32 MiB HugeTLB backing with results. Unset it only for a named 4 KiB
baseline. Keep CPU and memory placement NUMA-local. Do not use NUMA0 CPUs
`0-95` for new work unless the user explicitly requests a cross-NUMA or NUMA0
experiment.

## AmazonM5192Cores

- Host alias: `AmazonM5192Cores`
- Remote work root: `/data`
- Remote project-root rule: `/data/<local-project-directory-name>`
- Main worktree remote project root: `/data/fused_cpp`
- Sparse-attention worktree remote project root:
  `/data/fused_cpp-sparse-attn`
- Architecture: AArch64, Neoverse V3
- Online CPUs: `0-191`
- CPU topology: NUMA0=`0-95`, NUMA1=`96-191`

The remote project directory must retain the current local project directory
name. Run synchronization from the local project root, derive the destination
name from that root's basename, and preserve source-relative paths below the
remote project root. Do not copy project files directly into `/data` and
do not reuse another worktree's generic directory name. For this repository,
use `/data/fused_cpp/` from the main worktree and
`/data/fused_cpp-sparse-attn/` from the sparse-attention worktree.
