# MoE Native Code Guide

These rules apply under `csrc/moe/` in addition to the repository guide.

## Architecture Boundaries

- Keep architecture-neutral API, backend selection, and unavailable stubs under
  `common/`.
- Keep Arm NEON and SVE implementation details under `arm/neon_bf16/` and
  `arm/sve_bf16/`; shared Arm execution code belongs under `arm/common/`.
- Keep x86 implementation details under the matching `x86/<isa>/` directory.
- Architecture-specific source must not be included by another architecture.
  Connect implementations through the existing backend and kernel interfaces.
- Preserve a tested fallback for every supported backend. An optimization may
  narrow its own domain but must not silently narrow the public operator domain.

## Contracts

- Use `docs/public_contracts.md` as the authoritative compatibility boundary.
- Treat `common/api.h`, module bindings, backend ids, packed-weight layouts,
  Plan V2 inputs, output dtype/layout, and numerical behavior as compatibility
  surfaces.
- Treat production kernel symbols and dispatcher signatures as internal-stable:
  they may change only with all callers, tests, and Lab adapters updated in the
  same change.
- Runtime SVE vector length must match the build-time contract. Do not introduce
  a second layout or dispatch interpretation behind an undocumented flag.
- Keep required ISA fallbacks explicit; do not route an unsupported ISA into a
  stronger instruction path.

## Optimization Work

- Read `csrc/moe/README.md`, `docs/agent_optimization_governance.md`, and the
  matching `optimizations/fused_moe_<isa>/manifest.yaml` before kernel or
  executor optimization.
- Production source may expose narrow internal-stable services to Lab code, but
  it must not include Lab headers or branch on Lab-only variants.
- Accepted optimizations must update implementation, correctness tests,
  benchmark evidence, manifest status, and operator documentation together.
- Rejected or neutral implementations belong in Git history plus a manifest
  tombstone, not behind a default-off production branch.

## Validation

- Select focused tests from `tests/test_moe_backend_dispatch.py`,
  `tests/test_fused_moe_bf16_tiled.py`, and the ISA-specific MoE tests.
- Packing, tail, direct-store, merge, or precision changes must cover boundary M,
  nontrivial TopK, output dtype, fallback behavior, and unsupported inputs.
- Concurrency changes must cover single-thread, multi-thread, task completion,
  barrier/queue termination, and repeated invocation.
- Build and run native correctness tests on the target ISA before reporting
  performance. A local non-target build is not sufficient evidence.
