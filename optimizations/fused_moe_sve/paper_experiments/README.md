# CPU MoE paper experiment runner

This directory provides a declarative SSH runner for the fused-expert paper
matrix. It orchestrates existing correctness tests and benchmarks; it does not
add a kernel variant or change production dispatch.

Each run is bound to one committed source revision. With `--sync` (the default),
the runner creates a clean `git archive`, verifies and includes the pinned
`xbyak_aarch64` submodule, and synchronizes that snapshot without copying local
uncommitted work. Remote virtual environments and build products are preserved.
The runner does not use `rsync --delete`, so a source deletion still requires the
explicit remote cleanup required by `docs/agent_remote_execution.md`.

Results are written under the ignored local directory
`tmp/moe_paper_runs/<run-id>/`. Every run contains machine and extension hashes,
commands, stdout/stderr, structured benchmark results, the source revision, and
a summary. A unique mirror remains under the machine's configured
`remote_results_root`; the runner does not remove it automatically.

## Dry run

```bash
.venv/bin/python \
  optimizations/fused_moe_sve/paper_experiments/run_matrix.py \
  --machine optimizations/fused_moe_sve/paper_experiments/machines/arm_codex_internal.json \
  --suite optimizations/fused_moe_sve/paper_experiments/suites/fused_expert_smoke.json \
  --dry-run
```

## Execute

```bash
.venv/bin/python \
  optimizations/fused_moe_sve/paper_experiments/run_matrix.py \
  --machine optimizations/fused_moe_sve/paper_experiments/machines/amazon_ecs_8cores.json \
  --suite optimizations/fused_moe_sve/paper_experiments/suites/fused_expert_smoke.json
```

Use `--case ID` repeatedly to select a subset, `--no-build` only when the exact
snapshot has already been built, and `--no-sync` only when the remote source was
already populated from the same revision. The smoke suite validates the runner;
its 3--5 timing samples are not headline evidence. The pilot suite uses 15
samples and is still exploratory. Paper tables should use a separately declared
suite with at least 31 samples and the workload matrix required by
`docs/moe_paper_readiness.md`.

## Adding a machine

Copy one JSON file under `machines/` and set:

- the SSH alias, project root, Python path, and remote result root;
- an explicit CPU/NUMA affinity prefix;
- SVE vector width and machine-sized thread variables;
- runtime environment controls;
- initialized submodules required by the committed build.

Do not hide machine-specific shape changes in shell scripts. Add a named machine
variable or a separate suite so the rendered command remains visible in each
run's `command.json`.
