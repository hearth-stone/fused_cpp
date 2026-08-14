# Agent Remote Execution

Read this document before remote builds, tests, or benchmarks. The local Git
checkout is the source of truth unless a machine entry explicitly says
otherwise.

## General Rules

- Remote development and benchmark runs primarily use the Arm Codex machine.
  Use `Arm-codex-internal` on the internal network and `Arm-codex` through the
  external route. When the user says `aws机器` without naming another host,
  use `AmazonECS8Cores`.
- If a requested remote instance cannot be reached, stop the task immediately
  and report the connection failure. Do not silently switch machines or treat a
  local run as equivalent.
- Use the remote project's virtualenv Python. Do not rely on system `python`.
- Sync local files before remote validation. Preserve unrelated remote data and
  remove stale mirrored source explicitly when a local refactor deletes files.

## Arm Codex

- Host aliases: `Arm-codex-internal` and `Arm-codex`
- Preferred/default alias: `Arm-codex-internal`
- User: `zhangxu`
- Remote project/work directory: `/home/zhangxu/code`
- Local project root: repository root
- Python: `/home/zhangxu/code/.venv/bin/python`
- Package manager: `uv`

Both aliases identify the same machine and use the same remote paths. The
internal alias is the repository default because it is reachable from the
current development network. The external alias must be configured in the
caller's SSH config before use.

Sync with:

```bash
bash rsync.sh
```

Use the external alias explicitly when needed:

```bash
ARM_CODEX_HOST=Arm-codex bash rsync.sh
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
- HugeTLB mount: `/dev/hugepages-32M`
- HugeTLB pool: 320 x 32 MiB pages, 160 pages per NUMA node

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

Report the 32 MiB HugeTLB backing with results. Unset it only for a named 4 KiB
baseline. Keep benchmarks NUMA-local unless the experiment explicitly covers
both nodes.
