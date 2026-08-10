# Agent Optimization Governance

Read this file before operator/kernel optimization work, optimization manifest
changes, feature/variant organization, or default dispatch changes. Ordinary bug
fixes do not need to load this file unless they touch those areas.

These rules keep optimization work traceable without forcing a repository-wide
rewrite. Follow, in order: the user's current instruction, any nearer
`AGENTS.override.md`/`AGENTS.md`, the repository `AGENTS.md`, this file, then
existing repository conventions.

Core intent:

- Preserve public contracts and experimental knowledge, not every historical
  implementation.
- Apply the rules to active optimization work and newly introduced code.
- Make features and variants identifiable, testable, reproducible, and comparable.
- Keep production, active experiments, and retired work physically separate.
- Treat MUST/MUST NOT as required/prohibited; SHOULD as expected unless there is
  a clear engineering reason.

## Legacy Boundary

`ADOPTION_BASELINE` is the parent commit of the first commit adding this
governance. Tracked files at that point are legacy by default. If this has not
been committed yet, already tracked workspace files are legacy.

Active code includes new files, user-requested optimization work, relevant
uncommitted optimization changes, and new or updated manifests, tests, benchmarks,
dispatch logic, or adapters.

Before work, check `git status --short` or equivalent. Preserve user changes and
treat established public behavior as the compatibility baseline. Keep public API,
ABI, packed formats, backend ids, plan schemas, default dispatch, build
entrypoints, and numerical behavior stable unless the user explicitly asks.

Legacy source is not automatically a permanent compatibility surface. A legacy
implementation may be removed when it is internal, superseded, or experimentally
rejected, provided its externally observable contract remains covered and the
manifest retains the decision, evidence, and Git location. If legacy code changes,
report which files changed, why, how compatibility was preserved, and which tests
verified it.

## Compatibility Classes

- `public`: Python/C++ API, ABI, packed data format, backend id, serialized plan
  schema, documented numerical behavior, and supported build entrypoint. Changes
  require an explicit migration or user request.
- `internal-stable`: production kernel ABI, dispatcher contract, and required ISA
  fallback. Changes require focused integration tests but do not promise source
  compatibility.
- `experimental`: opt-in environment variables, probe enums, benchmark bindings,
  diagnostic kernels, and candidate-only entrypoints. These carry no compatibility
  promise and must not force production code to preserve rejected designs.

Every non-public control should be identifiable as `internal-stable` or
`experimental`. An undocumented default-off switch is experimental, not an
accidental public API.

## Optimization Model

- `baseline`: the authoritative behavior or correctness reference for one public
  contract. Keep at most one practical baseline per contract and ISA. It may be
  an active implementation, a small reference, an upstream implementation, or a
  Git-located retired implementation; baseline status does not imply permanent
  inclusion in the default build.
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

Optimization source has three physical lifecycles:

- `Production`: `csrc/`, `src/`, and default-built support code. Only enabled
  features, required fallbacks, and internal-stable interfaces belong here.
- `Lab`: `optimizations/<operator>/features`, experiment-only tests, benchmarks,
  and optional build targets. Experimental, diagnostic, and benchmark-reference
  code belongs here and must not enter the default extension.
- `Archive`: manifest tombstones, concise result documents, and Git commits or
  annotated tags. Retired source must not remain in active build inputs merely to
  make an old experiment easy to rerun.

Use the existing operator, test, and benchmark directories. If no suitable Lab
structure exists, use:

```text
optimizations/<operator>/
  manifest.yaml
  features/
  variants/
  tests/
  benchmarks/
```

Connect Lab code through a narrow internal-stable kernel ABI, adapters, or an
explicit experiment build target. Production code must not include Lab headers or
branch on Lab-only variants. Small local duplication in Lab code is acceptable
when it prevents an experiment from expanding the production compatibility
matrix. Copying a whole legacy operator or large legacy file requires explicit
user approval.

## Lifecycle And Retention

Features move through `experimental -> candidate -> enabled` or
`experimental/candidate -> retired`.

- `experimental`: Lab only, opt-in, no compatibility promise.
- `candidate`: still opt-in; must have an owner or decision date, full
  correctness coverage, and explicit adoption criteria.
- `enabled`: production-supported and eligible for default dispatch.
- `retired`: no active source or default-build entrypoint. Keep only the result,
  retirement reason, last implementation commit/tag, and reproduction command.
- `reference`, `diagnostic`, and `benchmark_reference` describe roles, not a
  reason to compile source into production. Diagnostic and benchmark-reference
  implementations belong in Lab.

Once evidence rejects an experiment or shows it performance-neutral without a
separate maintenance benefit, retire it. Do not retain it behind a default-off
flag. An inconclusive experiment may remain in Lab only while a concrete next
decision exists. Superseded internal implementations should be deleted after the
replacement covers their contract; retain a thin adapter only for a public API.
Existing manifests and source predating this lifecycle are a migration backlog;
clean them incrementally, but do not add or expand a nonconforming path.

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
- Candidate entries record an adoption gate and decision date or next decision.
- Retired entries record `implementation: Removed from the active tree`, a result
  or retirement reason, and a `history` commit/tag when one is known. Their
  variants must not name a buildable production entrypoint.

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
Experimental variants should be reachable only through explicit Lab tests,
benchmarks, optional build targets, or experimental entrypoints. They must not add
branches, cache-key dimensions, environment parsing, or pybind symbols to the
default production extension. Dispatch rules for enabled features must record
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

For rejected work developed outside the main branch, preserve the final experiment
with an annotated `archive/<operator>/<feature>` tag or another durable commit
reference before deleting the branch. Do not copy retired source into an archive
directory; Git is the source archive.

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
8. Promote accepted code into Production or retire rejected code from the active
   tree. Preserve default behavior and avoid unrelated legacy edits.

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
- Switching default kernels without the requested evidence and validation.
- Refactoring all optimization code at once.
- Generating every theoretical combination.
- Creating many permanent branches.
- Copying complete legacy implementations into version files.
- Modifying the global build system for a single experiment.
- Introducing new production dependencies.
- Lowering test standards or relaxing numeric tolerances.
- Replacing measured benchmarks with theoretical gains.
- Hiding negative optimizations, failing shapes, or unsupported conditions.

Guiding principle: preserve contracts and experimental knowledge, not experimental
source by default. Add optimizations incrementally; features express single ideas;
variants express explicit combinations; correctness comes before performance;
data comes before conclusions; experiments do not change defaults without
explicit approval.
