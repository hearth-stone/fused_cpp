# CPU MoE paper experiment runner

This directory provides a declarative SSH runner for the fused-expert paper
matrix. It orchestrates existing correctness tests and benchmarks; it does not
add a kernel variant or change production dispatch.

Each run is bound to one committed source revision. With `--sync` (the default),
the runner creates a clean `git archive`, verifies and includes the pinned
`xbyak_aarch64` submodule, and synchronizes that snapshot without copying local
uncommitted work. The repository's ignored `refs/i8gemm/lib` dependency is
copied separately and bound to a deterministic content hash in `summary.json`;
it must become a committed dependency before the final artifact freeze. Remote
virtual environments and build products are preserved.
The runner does not use `rsync --delete`, so a source deletion still requires the
explicit remote cleanup required by `docs/agent_remote_execution.md`.

Results are written under the ignored local directory
`tmp/moe_paper_runs/<run-id>/`. Every run contains machine and extension hashes,
commands, stdout/stderr, structured benchmark results, the source revision, and
a summary. A unique mirror remains under the machine's configured
`remote_results_root`; the runner does not remove it automatically.
Suite builds set `FUSED_CPP_BUILD_MOE_ONLY=1`, so a paper run rebuilds the
dedicated MoE extension without charging unrelated operators to setup time.

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

The explicit fused/unfused reference stays in its validated EP2
`H=4096,F=2048` numerical domain. The exact-M and direct-route cases use the
paper's TP4-oriented `H=4096,F=512` path. These are mechanism experiments, not
one same-shape cumulative ablation.

## Adding a machine

Copy one JSON file under `machines/` and set:

- the SSH alias, project root, Python path, and remote result root;
- an explicit CPU/NUMA affinity prefix;
- SVE vector width and machine-sized thread variables;
- runtime environment controls;
- initialized submodules required by the committed build.
- any ignored external source directories, each recorded with a content hash.

Multi-core suites keep `OMP_NUM_THREADS=1` but set `OMP_PROC_BIND=FALSE`.
Otherwise libgomp may narrow the importing process to one CPU before the native
MoE worker pool reads its inherited affinity.

Do not hide machine-specific shape changes in shell scripts. Add a named machine
variable or a separate suite so the rendered command remains visible in each
run's `command.json`.
