# Agent Guide

This repository is an independent C++/Python extension workspace. Keep this
always-loaded file limited to stable repository-wide rules. Load the linked
topic documents only when their scope applies.

## Scope And Communication

- Treat the nearest `.git` directory as the repository boundary.
- Use Chinese by default when discussing work with the user.
- Use English for upstream-facing documentation, code comments, and commit
  messages, or when the surrounding file is already English.
- State assumptions, risks, validation gaps, and performance methodology
  explicitly.
- Do not claim correctness or a performance improvement without naming the
  command, dataset or shape, configuration, baseline, and measured result.

## Navigation And Required Workflows

- Use CodeGraph first for structural questions in this indexed repository:
  definitions, callers, callees, impact analysis, and control or data flow.
- Use `rg` for literal text, comments, log messages, and configuration keys, or
  after CodeGraph identifies the relevant files.
- Prefer existing local patterns and helper APIs before adding abstractions.

The parent workspace skills are mandatory workflow gates when their trigger
applies:

| Skill | Required when | File |
| --- | --- | --- |
| `impact-analysis` | Before code or behavior-affecting configuration changes. | `../skills/impact-analysis/SKILL.md` |
| `test-selector` | When selecting validation for a change. | `../skills/test-selector/SKILL.md` |
| `safe-refactor` | Before refactors, moves, renames, or boundary changes. | `../skills/safe-refactor/SKILL.md` |
| `code-review-gate` | Before finalizing edits, committing, or preparing a PR. | `../skills/code-review-gate/SKILL.md` |
| `api-db-change` | Before API, schema, migration, auth, or persistence changes. | `../skills/api-db-change/SKILL.md` |

If a required tool is unavailable, report that and use the fallback documented
by the relevant skill.

## Repository Safety

- Check `git status --short` before changes and preserve unrelated user work.
- Never revert, delete, overwrite, or reformat unrelated changes.
- Keep edits scoped to the requested operator, module, and behavioral surface.
- Keep generated builds, benchmark dumps, model files, caches, traces, and
  temporary outputs out of source commits unless explicitly requested.
- Read `docs/public_contracts.md` before changing exports, signatures, native C
  ABI, packed objects, plan schemas, backend ids, numerical behavior, or
  supported configuration.
- Register every repository-owned environment variable read by Production or
  the build in `docs/production_environment.yaml` before adding its parser.
- Preserve public APIs, ABI, packed formats, plan schemas, backend ids, build
  entrypoints, numerical behavior, and default dispatch unless the user
  explicitly requests a migration or behavior change.
- Do not introduce a production dependency or change the global build for one
  experiment without explicit approval.

## Coding And Validation

Classify nontrivial work with `docs/change_policy.md` before editing. State the
primary change class, contracts/defaults affected, required validation level,
and rollback boundary. When work spans classes, apply the strictest relevant
gate.

Read the matching parent rule before editing:

| Area | Rule file |
| --- | --- |
| C, C ABI, low-level kernels | `../rules/c.md` |
| C++, headers, assembly wrappers | `../rules/cpp.md` |
| Python source | `../rules/python.md` |
| Python tests and benchmarks | `../rules/python.md`, `../rules/python-test.md` |
| GitHub, commits, PRs, review | `../rules/github.md` |

- Run the smallest relevant correctness test before broader tests or
  benchmarks.
- For kernel and numerical changes, run focused correctness tests before making
  performance claims.
- Never claim an unrun test passed. Name skipped validation and residual risk.
- Ask before long builds, long benchmarks, model downloads, or remote GitHub
  mutations unless the user already requested that operation.

## Operator Gates

- Before SDPA or microkernel optimization work, read the local TODO/version
  documents and keep implementation, tests, benchmarks, and conclusions in
  sync. Record every optimization attempt in `csrc/SDPA_VERSIONS.md`.
- Before changing the CPU MoE planner decision space, cost model, resources, or
  pruning rules, read `cpu_moe_schedule_optimization/MATHEMATICAL_MODEL.md` and
  update formulas, pruning, validation notes, and changelog in the same change.
- Before operator or kernel optimization work, read
  `docs/agent_optimization_governance.md`.

## Remote And Performance Work

- Read `docs/agent_remote_execution.md` before using a remote machine. It is the
  source of truth for aliases, paths, topology, sync behavior, and HugeTLB setup.
- Read `docs/agent_benchmark_hygiene.md` before collecting or comparing
  performance data.
- Read `docs/agent_performance_references.md` before comparing measured results
  with hardware peaks.
- Model-related configuration files live under
  `/Users/zhangxu/Codes/vllm-aarch64/models`.

## Git

- Do not commit, push, rebase, force-push, or mutate remote repository state
  unless the user asks.
- When commits are requested, keep each commit focused and review the exact
  staged diff before committing.
- Follow `../rules/github.md` for commit message and review requirements.

## Guiding Principle

Preserve contracts and experimental knowledge, not experimental source by
default. Keep production, active experiments, and retired work physically
separate; correctness comes before performance, and measured data comes before
conclusions.
