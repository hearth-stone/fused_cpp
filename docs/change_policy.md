# Change Classification And Validation Policy

Classify work before editing so its compatibility, evidence, and rollback
requirements are explicit. The classification is a planning tool, not a commit
message type and not a reason to run every test for every change.

Use one primary class. If work crosses classes, apply all relevant requirements
and the highest validation level. Split unrelated classes into separate commits
where practical.

## Change Declaration

For nontrivial work, state this compact declaration before editing:

```text
Change class:
Scope:
Public/internal contracts affected:
Default behavior affected:
Expected mechanism or invariant:
Minimum validation level:
Target machine/ISA, if any:
Rollback boundary:
```

Documentation-only typo fixes may use an abbreviated declaration. The
declaration belongs in the work log or agent update; it does not need to become
a permanent repository file.

## Change Classes

### D: Documentation Or Repository Rules

Use for prose, navigation, governance, comments, and metadata that do not alter
runtime, build, schema, or generated behavior.

Required:

- verify referenced paths, commands, names, and values against the repository;
- check formatting and exact diff scope;
- state explicitly when no runtime validation is needed.

Minimum: L0.

### B: Bug Fix

Use when current behavior violates an existing contract or documented intent.

Required:

- reproduce the failure or identify a deterministic failing invariant;
- add or identify a regression test that fails before and passes after;
- preserve unrelated behavior and supported fallbacks;
- name the root cause, not only the observed symptom.

Minimum: L1. Use L2 when the bug is architecture-, process-, ABI-, build-, or
concurrency-specific.

### R: Behavior-Preserving Refactor

Use for moves, renames, decomposition, deduplication, interface reshaping, and
internal rewrites intended not to change observable behavior.

Required:

- run `impact-analysis` and `safe-refactor`;
- list preserved invariants and affected internal callers;
- separate mechanical movement from behavior changes;
- keep each stage independently revertible;
- compare representative outputs before and after.

Minimum: L1. Use L2 for native boundaries, ISA code, threading, packing, or
cross-extension changes.

### C: Public Contract Or Migration

Use for changes to a surface listed in `docs/public_contracts.md`, including
exports, signatures, C ABI, backend ids, plan versions, packed object semantics,
or documented numerical behavior.

Required:

- explicit current user approval for the contract change;
- old/new contract and caller impact;
- additive compatibility, deprecation, adapter, or versioned migration plan;
- positive and negative compatibility tests;
- updated public contract and detailed API/schema documentation;
- separate migration commit from unrelated optimization.

Minimum: L2. Add L3 if the migration also changes performance defaults or model
decisions.

### O: Production Optimization

Use for a candidate intended to enter Production or change default dispatch.

Required:

- read `docs/agent_optimization_governance.md`;
- identify baseline, one primary feature, variant, preconditions, and fallback;
- pass correctness before measuring performance;
- define adoption and rejection criteria before the deciding benchmark;
- compare the same build, inputs, placement, and measurement method;
- record absolute results, noise, regressions, and unsupported shapes;
- adopt into Production or retire the implementation after the decision.

Minimum: L2 plus L3. Kernel-local development may start at L1 but cannot be
adopted or described as an improvement without L2/L3.

### E: Lab Experiment Or Diagnostic

Use for a falsifiable optimization experiment, probe, comparator, oracle, or
diagnostic that is not production-supported.

Required:

- place source under `optimizations/` or another explicit optional target;
- register hypothesis, implementation, correctness, benchmark, owner/decision,
  dependencies, conflicts, and status in the operator manifest;
- avoid production flags, pybind branches, cache dimensions, and build inputs;
- after the decision, adopt or retire; do not leave an indefinite default-off
  implementation.

Minimum: L1 for implementation correctness and L3 for a performance conclusion.
L2 is required only when the experiment claims target-runtime integration.

### M: Planner, Cost Model, Or Calibration Semantics

Use for candidate space, pruning, objective, resource equations, contention,
calibration interpretation, profile schema, or plan lowering.

Required:

- update `cpu_moe_schedule_optimization/MATHEMATICAL_MODEL.md` in the same
  change;
- separate analytical structure from calibrated parameters;
- state assumptions, supported domain, identifiability, and uncertainty;
- test limits/invariants and Python/native parity where applicable;
- report holdout error and regret against the declared oracle or measured set;
- measure cold-planning overhead when search changes.

Minimum: L1 plus model holdout at L3. Use L2 when serialized plans or native
runtime execution change.

### I: Build, Dependency, Or Development Infrastructure

Use for setup/build logic, compiler flags, dependencies, CI, code generation,
sync scripts, or machine tooling.

Required:

- state affected platforms and artifact/ABI implications;
- avoid adding production dependencies for one experiment;
- verify clean and incremental workflows where relevant;
- preserve an actionable failure message for unsupported environments.

Minimum: L1 for tooling logic and L2 on each affected build platform.

## Validation Levels

Levels are cumulative in intent: L3 performance evidence never replaces L1
correctness, and an L2 target build never replaces schema/unit validation.

### L0: Static And Scope Validation

- `git status --short` before and after;
- `git diff --check`, `git diff --stat`, and targeted diff review;
- formatter/linter/type/schema checks required by touched files;
- referenced paths, manifest entrypoints, generated inputs, and docs verified;
- no unrelated changes or generated artifacts staged.

### L1: Focused Correctness

- smallest direct unit/regression test for changed behavior;
- negative and boundary cases when validation or shape domains change;
- deterministic repeated invocation for stateful paths;
- no benchmark claim yet.

### L2: Target Integration

- build and run on the affected architecture/ISA/runtime;
- exercise public bridge, internal backend boundary, fallback, and concurrency
  paths touched by the change;
- verify clean/incremental build or process boundary when relevant;
- name any supported target that was not validated.

### L3: Performance Or Model Evidence

- reproducible benchmark or holdout command and environment;
- absolute baseline/candidate values, statistic, sample count, and relative
  difference;
- noise/repeatability and material regressions;
- cross-shape and cross-machine coverage required by the claim;
- adoption/rejection decision or model gate result.

## Escalation Rules

Escalate the validation level when any of these apply:

- public contract, serialized schema, packed layout, or backend identity;
- assembly/JIT, numerical precision, atomics, barriers, queues, or thread pool;
- default dispatch, compiler flags, page policy, or hardware feature detection;
- planner search space, pruning, profile compatibility, or model objective;
- behavior differs by ISA, NUMA, page size, process lifetime, or concurrency.

The user may explicitly accept a validation gap. Report the exact skipped gate
and residual risk; never silently downgrade it.

## Completion Record

The final report for a nontrivial change states:

- class and affected contract/default;
- implementation and rollback boundary;
- commands run and results by L0-L3;
- performance/model decision when applicable;
- skipped validation and remaining risk;
- commit(s), when the user requested commits.
