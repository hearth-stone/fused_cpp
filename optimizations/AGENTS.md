# Optimization Lab Guide

Everything under `optimizations/` is Lab or Archive material unless an operator
manifest explicitly points to an adopted Production implementation elsewhere.
These rules supplement `docs/agent_optimization_governance.md`.

## Isolation

- Lab source must not enter the default extension build and Production source
  must not include Lab headers.
- Connect experiments to Production only through narrow internal-stable kernel
  services, adapters, or explicit optional build targets.
- Small local duplication is acceptable when it prevents an experiment from
  expanding the Production compatibility matrix.
- Do not copy a complete production operator into Lab without explicit user
  approval.

## Manifest And Lifecycle

- Every experiment needs one feature id, a falsifiable hypothesis, dependencies,
  conflicts, correctness command, benchmark command, status, and implementation
  location in the operator manifest.
- Variants list their complete feature set and buildable entrypoint. Do not use
  branches, macros, or undocumented flags as the variant registry.
- Active experiment source must have a concrete next decision. Rejected,
  superseded, or neutral source is removed; retain the conclusion, measurements,
  reproduction command, and final Git commit/tag.
- A result document is evidence, not an alternative source archive. Do not copy
  retired implementations into `results/`.

## Artifacts And Evidence

- Keep concise Markdown results and small production-consumed calibration files
  in Git. Keep raw perf output, traces, binaries, generated assembly dumps,
  caches, and large tables outside Git.
- Benchmark baseline and candidate in the same process/build/data regime where
  practical. Report noise and all material regressions.
- An experiment does not change Production defaults until correctness,
  cross-shape performance, adoption criteria, and fallback behavior are recorded.
