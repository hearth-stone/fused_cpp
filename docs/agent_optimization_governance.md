# Agent Optimization Governance

Read this file before operator/kernel optimization work, optimization manifest
changes, feature/variant organization, or default dispatch changes. Ordinary bug
fixes do not need to load this file unless they touch those areas.

These rules keep optimization work traceable without forcing a repository-wide
rewrite. Follow, in order: the user's current instruction, any nearer
`AGENTS.override.md`/`AGENTS.md`, the repository `AGENTS.md`, this file, then
existing repository conventions.

Core intent:

- Preserve existing implementations as compatibility baselines.
- Apply the rules to active optimization work and newly introduced code.
- Make features and variants identifiable, testable, reproducible, and comparable.
- Treat MUST/MUST NOT as required/prohibited; SHOULD as expected unless there is
  a clear engineering reason.

## Legacy Boundary

`ADOPTION_BASELINE` is the parent commit of the first commit adding this
governance. Tracked files at that point are legacy by default. If this has not
been committed yet, already tracked workspace files are legacy.

Active code includes new files, user-requested optimization work, relevant
uncommitted optimization changes, and new or updated manifests, tests, benchmarks,
dispatch logic, or adapters.

Before work, check `git status --short` or equivalent. Preserve user changes,
treat existing implementations as behavior/correctness baselines, and edit legacy
code only when integration genuinely requires it. Keep legacy API, ABI, default
dispatch, build entrypoints, and behavior stable unless the user explicitly asks.

Do not move, rename, reorder, format, refactor, copy, delete, or overwrite legacy
code just to satisfy this guide. Do not replace a default implementation without
explicit instruction. If legacy code changes, report which files changed, why,
how compatibility was preserved, and which tests verified it.

## Optimization Model

- `baseline`: existing reference or current default implementation. It is frozen
  by default, may be called through adapters, and must not be copied as a new
  version or replaced just because one benchmark variant is faster.
- `part`: an independently analyzable operator stage such as `load`, `layout`,
  `compute`, `reduce`, `store`, `epilogue`, or `dispatch`. Prefer local names.
- `feature`: one primary optimization idea, named `<part>.<mechanism>`, with a
  unique id, part, falsifiable hypothesis, implementation location, dependencies,
  conflicts, status, correctness method, and benchmark method.
- `variant`: an executable combination of explicit features. Its feature set
  must be listed in the manifest. Combined variants must declare dependencies and
  conflicts, have their own correctness tests, and be compared against baseline
  plus main single-feature variants. Git branches, hidden macros, undocumented
  compiler flags, or branch state are not variant registries.

## File Placement

Prefer existing operator, test, and benchmark directories. If no suitable
structure exists, use:

```text
optimizations/<operator>/
  manifest.yaml
  features/
  variants/
  tests/
  benchmarks/
```

Do not move legacy implementations just to create this layout. Connect new code
through adapters, wrappers, registries, or minimal build integration. Use existing
test/benchmark frameworks. Small local duplication in new optimization code is
acceptable when it avoids risky legacy changes. Copying a whole legacy operator
or large legacy file requires explicit user approval.

## Manifest

Each adopted operator has one primary manifest, defaulting to
`optimizations/<operator>/manifest.yaml`. Equivalent JSON, TOML, or code registry
is acceptable if the semantics match.

Required semantics:

- `baseline` exists and points to the existing reference entrypoint.
- Each feature uses `<part>.<mechanism>` and records status, implementation,
  hypothesis, `depends_on`, `conflicts_with`, correctness command, and benchmark
  command.
- Each variant records id, feature list, entrypoint, and status.
- Feature order follows operator pipeline order, then lexicographic order within
  a part.
- Manifest additions, removals, and renames match real buildable entrypoints.
- Experimental work is not presented as production.

Use semantic names, not `v1`, `v2`, `new`, `new2`, `latest`, `fast`, or `final`.
Recommended statuses: `reference`, `experimental`, `candidate`, `enabled`,
`retired`. New variants default to `experimental`.

## Feature and Variant Work

For a single-feature optimization: identify the part, create a unique feature id,
write a falsifiable hypothesis, keep unrelated dimensions unchanged, add or reuse
correctness tests and benchmarks, register the variant, and report results
relative to baseline.

For a combined optimization: list every feature, state application order, declare
dependencies/conflicts, ensure the combination is not implicit behavior, test it
separately, compare baseline plus main single-feature variants, and report whether
effects stack, cancel, or regress. Do not assume individually good features work
well together.

Prefer clear parameterization, templates, compile-time configuration, strategy
objects, or registries when they reduce real duplication. Keep shared abstractions
inside new optimization code. Do not create untraceable macro matrices, many
nested boolean switches without registered variants, speculative frameworks, or
mass legacy refactors.

## Dispatch and Defaults

Adding a variant is not enabling it. Preserve original default dispatch.
Experimental variants should be reachable only through explicit tests,
benchmarks, build options, or experimental entrypoints. Dispatch rules must record
shape, dtype, layout, hardware, and other preconditions, safely fall back to the
baseline, and keep fallback paths testable.

Do not switch defaults from one local benchmark, remove baseline fallback, expand
supported domains without correctness evidence, or silently change precision,
tolerances, overflow behavior, or determinism.

## Correctness

Correctness comes before performance. New variants must compare against baseline
or an authoritative reference on the same inputs, cover typical/boundary/non-tile
aligned inputs, cover declared dtype/layout/shape support, validate invalid or
unsupported inputs, use existing tolerance standards, and justify any tolerance
change. Async device tests and benchmarks must synchronize before reading results
or ending timing.

If tests cannot run, state which tests were skipped, why, what alternative
validation was done, and what risk remains. Never claim unrun tests passed.

## Benchmarks

Performance conclusions must be reproducible. Record workspace state, hardware,
runtime/driver/compiler versions, build type/flags, operator, variant, shape,
dtype, layout, warmups, measurement count, synchronization method, latency stats
(at least median or p50 and sample count), throughput/bandwidth where applicable,
baseline result, absolute values, and relative change.

Use the same inputs, build mode, and measurement method for baseline and variants.
Single-feature variants compare to baseline; combined variants compare to baseline
and main single-feature variants. Do not use debug-build results as release
evidence, report profiler estimates or compilation success as speedup, hide
regressions, invent pass thresholds, or promote variants when performance cannot
be reproduced. Keep large profiler outputs, binary traces, and temporary artifacts
out of commits unless the repo already says otherwise.

## Git

Git records implementation history; the manifest records version facts. Use
short-lived branches such as `opt/<operator>/<feature>` when useful, keep commits
focused, and separate single-feature work from unrelated optimizations. Do not use
long-lived `v1/v2/v3` branches, create permanent branches for parameter
combinations, modify unrelated history, or commit/push/rebase/force-push unless
the user asks.

## AI Workflow

For optimization tasks:

1. Read the repository `AGENTS.md`, this file, and nearer instruction files.
2. Check `git status --short`.
3. Locate baseline, call chain, build entrypoint, tests, and benchmarks.
4. Identify operator, part, feature, variant, legacy files, and active files.
5. Before editing, note: target operator/parts, feature ids, variant ids, legacy
   edits, correctness command, benchmark command, expected mechanism, and risks.
6. Implement a single feature first, register it, test it, and benchmark it.
7. Add combined variants only when needed; register, test, and benchmark them
   separately.
8. Preserve default behavior and avoid unrelated legacy edits.

Before finishing, inspect `git diff --stat` and `git diff`; ensure no accidental
format churn, manifest entries match real entrypoints, dependencies/conflicts and
compiler options are declared, baseline remains buildable/callable/fallback-ready,
and reported tests/benchmarks actually ran.

Final reports for optimization work must include changed files, features,
variants, legacy edits and why, correctness commands/results, benchmark
commands/environment/results, baseline/single-feature/combined comparisons,
incomplete validation, remaining risk, and whether default behavior changed.

## Prohibited Without Current User Request

- Repository-wide migrations or legacy operator rewrites.
- Deleting legacy implementations or switching default kernels.
- Refactoring all optimization code at once.
- Generating every theoretical combination.
- Creating many permanent branches.
- Copying complete legacy implementations into version files.
- Modifying the global build system for a single experiment.
- Introducing new production dependencies.
- Lowering test standards or relaxing numeric tolerances.
- Replacing measured benchmarks with theoretical gains.
- Hiding negative optimizations, failing shapes, or unsupported conditions.

Guiding principle: keep legacy stable; add optimizations incrementally; features
express single ideas; variants express explicit combinations; correctness comes
before performance; data comes before conclusions; experiments do not change
defaults without explicit approval.
