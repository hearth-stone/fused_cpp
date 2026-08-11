# SDPA Microkernel Guide

These rules apply under `csrc/sdpa_microkernels/` in addition to the repository
guide.

## Required Context

- Read `csrc/SDPA_TODO.md`, `csrc/SDPA_VERSIONS.md`, and
  `csrc/SDPA_STANDALONE.md` before changing microkernels, layouts, registration,
  or standalone benchmarks.
- Record every optimization attempt in `csrc/SDPA_VERSIONS.md`, including
  rejected and neutral results.

## Implementation Boundaries

- Keep the scalar or established baseline callable as the correctness reference.
- Register implementation names centrally through `mk_registry.cpp`; names are
  stable benchmark identifiers and must not be silently reused for new code.
- Keep layout, packing, compute, reduction, and store assumptions explicit in
  traits or entry metadata. Validate them before dispatch.
- Do not make an experimental implementation the default from a single shape or
  one machine result.
- Avoid growing one header with unrelated variants. Put a new coherent
  implementation in `impls/` and share only abstractions that remove real
  duplication.

## Correctness And Performance

- Validate supported dtype, E/Sk tails, masking, row strides, scratch layout,
  and invalid input behavior against the authoritative reference.
- Use `tests/test_microkernel_framework.py` and the relevant SDPA equivalence or
  cache-microkernel tests before benchmarks.
- Performance reports must name implementation, dtype, E, Sk, core, warmups,
  samples, build flags, baseline, and absolute/relative results.
- Keep correctness tests free of timing thresholds. Put performance comparisons
  in benchmark tools and result documentation.
