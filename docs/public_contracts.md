# Public Contracts

This document defines which repository surfaces require compatibility. It is a
boundary document, not a complete API reference: individual signatures and
shape rules remain documented beside their implementation.

The purpose is to prevent two opposite mistakes:

1. silently breaking a real caller-facing contract during optimization; and
2. preserving every internal symbol, benchmark hook, or rejected experiment as
   if it were a public API.

## Compatibility Classes

### Public

A public surface is intended for callers outside its implementation module.
Changing or removing it requires an explicit user request, an updated contract,
a migration path or version transition, and compatibility-focused tests.

### Internal-Stable

An internal-stable surface connects production modules, ISA backends, or a
documented integration. It may change atomically with all in-repository callers,
tests, and adapters, but it must not break the public surface or required
fallbacks.

### Experimental

An experimental surface exists for Lab work, diagnostics, profiling, or a
candidate not yet adopted. It carries no compatibility promise. It must not
force Production to retain a rejected implementation.

If a surface is not listed here, explicitly exported, or documented as a
supported integration, do not assume it is public. Established external use
still requires impact analysis before removal.

## Python Package

The public Python symbol set is defined by the explicit `__all__` declarations
in:

- `src/fused_cpp/__init__.py`;
- each exported subpackage, including `src/fused_cpp/moe/__init__.py`;
- public operator modules such as `sdpa.py`, `bf16_linear.py`, and `i8gemm.py`.

For these symbols, the following are public unless documented otherwise:

- import path and exported name;
- callable parameter names, positional/keyword behavior, and defaults;
- accepted tensor rank, dtype, device, layout, and shape domain;
- return type, output shape/dtype/device, and documented `out=` behavior;
- dataclass field names and meanings;
- documented errors for unsupported inputs or unavailable backends.

Capability names beginning with `_HAS_` or `_supports_` are import-compatible
when deliberately listed in `__all__`; their boolean value is machine- and
build-dependent.

Implementation modules `fused_cpp._C` and `fused_cpp._moe_C` are not new public
API surfaces. Existing `_C.fused_moe_*` forwarding is a legacy compatibility
shim: do not expand it, and require an explicit migration before removing it.
Test and benchmark bindings exposed only through a native extension are
experimental unless separately documented here.

## Native C ABI

The following native integration surfaces are public:

- declarations and ABI values in `csrc/sdpa_c_api.h`;
- llama.cpp-facing exports documented in
  `docs/llama_cpp_sdpa_integration.md`.

Their exported names, calling convention, enum/numeric values, parameter
layout, ownership rules, error reporting contract, and symbol visibility must
remain compatible unless explicitly versioned.

Other `extern "C"` symbols used by bundled assembly, JIT code, or internal
translation units are internal-stable kernel ABI, not automatically public C
API. Change them only with every caller, static layout assertion, target build,
and Lab adapter updated together.

## BF16 Tiled MoE

### Prepared Weights

`PreparedBF16TiledFusedMoEWeights` is public with these fields:

- `w13: tuple[Tensor, int, int]`;
- `w2: tuple[Tensor, int, int]`;
- `fused_silu: bool`;
- `gemm_backend: int`;
- `backend_n_tile: int`;
- `backend_name: str`.

The object returned by `prepare_fused_moe_bf16_tiled_weights` must remain
accepted by the matching execution entrypoints in the same compatible build.
The packed tensors are opaque implementation data: cross-version persistence,
manual construction, and deserialization into a different ISA/VL are not public
contracts unless a separate serialized format is introduced.

Changing the in-memory packed layout is therefore an internal-stable migration:
update packers, all consumers, backend metadata, correctness tests, and Lab
adapters atomically. Do not reinterpret an existing prepared object silently.

### Backend Identity

These backend ids and canonical names are stable:

| Id | Name | Status |
| ---: | --- | --- |
| `0` | `arm_neon_bf16` | implemented when build/runtime supports it |
| `1` | `arm_sve_bf16` | implemented when build/runtime supports it |
| `100` | `x86_avx2` | reserved, currently not implemented |
| `101` | `x86_avx512_bf16` | implemented when build/runtime supports it |
| `102` | `x86_amx_bf16` | implemented when build/runtime supports it |

Aliases accepted by backend selection may be extended, but an existing id or
canonical name must not be reused for a different layout or ISA.

`auto` selection may choose a stronger supported backend, but it must preserve
operator semantics and reject unsupported combinations rather than executing an
incompatible kernel. Runtime SVE VL must match the build-time packed/JIT
contract; import-time mismatch remains an error, not a silent fallback.

### Execution And Numerical Behavior

The exported normal, scheduled, async, Plan V2, planned-staged, and vLLM-staged
functions retain their documented Python signatures and validation behavior.
For supported inputs they return a contiguous tensor matching the input token
and hidden dimensions and documented dtype/device. A valid `out=` buffer is
written in place and returned.

TopK routing semantics, route weights, activation selection, expert-id bounds,
empty input behavior, and `skip_weighted` restrictions are public. Internal
GEMM tiling, thread ownership, cache windows, barriers, intermediate dtype, and
merge implementation are not public unless they change documented numerical
behavior.

Bitwise equality between different backends is not promised unless a test or
operator document explicitly requires it. Tolerance changes need a numerical
justification and an explicit contract update; they must not be relaxed only to
adopt a faster candidate.

## Async MoE Plan V2

`ASYNC_MOE_PLAN_VERSION == 2` and `AsyncMoEPlanV2.from_dict()` define a public,
versioned plan schema. The authoritative field and validation description is
`cpu_moe_schedule_optimization/planners/plan_schema.md`.

The contract includes:

- required version, execution mode, thread CPU ids, task/dependency arrays,
  placement fields, stage ids, resize/range fields, and their integer semantics;
- optional per-task `task_w13_window_tiles` and `task_w2_window_tiles`, where
  zero means the full owner stripe;
- the `early_merge` tri-state;
- deterministic rejection of missing, stale, ambiguous, or invalid payloads;
- `upgrade_legacy_async_plan` as the documented compatibility bridge.

Adding an optional field requires a default that preserves old behavior.
Removing or reinterpreting a field requires a new schema version or an explicit
migration. Runtime-only environment variables must not reinterpret the same
serialized plan differently.

## SDPA And Registered Implementations

The public SDPA dispatcher functions and `VersionInfo` exported through
`fused_cpp` retain their signatures and documented semantics. Registered
version names are stable identifiers while documented as available; adding a
version does not make it the default.

Microkernel registry names used only by standalone validation and benchmarks are
experimental. The scalar/reference implementation and supported default
dispatch remain correctness/fallback contracts even when optimized versions are
added.

## Sparse MLA

The exported `flash_mla_sparse_fwd`, `flash_mla_sparse_fwd_naive`, and
`sparse_mla` Python names retain their documented signatures, tensor validation,
output and optional statistics semantics, attention-sink behavior, and `out=`
contract.

For supported BF16 MQA shapes, the public optimized entrypoint uses head-major
8x8 QK/PV once query parallelism is sufficient. Short query chunks retain
guarded 2D KV partitioning; FP32 and unsupported BF16 shapes retain the indexed
fallback. Task layout, packing, split count, partial buffers, and merge order
are internal. Every dispatch must preserve the documented numerical tolerances,
return-statistics definitions, attention-sink semantics, and safe fallback for
unsupported or non-canonical sparse patterns.

## Configuration And Defaults

Default dispatch, backend fallback order, numerical mode, precision, page policy,
and thread behavior are observable contracts when documented for users. A new
optimization must not change them merely because its source is compiled.

Environment variables are not public Python API. A variable explicitly
documented as a supported operational control is internal-stable configuration;
diagnostic and experimental variables carry no compatibility promise. Every
repository-owned variable read by Production or the build must be classified in
`docs/production_environment.yaml`. Its `supported`, `diagnostic`, or
`temporary` class controls compatibility and retirement; merely appearing in
the registry does not make it public API. An undocumented default-off switch is
experimental and must not be retained as accidental compatibility.

Exact latency, throughput, internal file layout, private class names, trace text,
benchmark CLI details, generated symbol addresses, and cache/JIT implementation
are not public contracts.

## Contract Change Procedure

Before changing a public surface:

1. identify external and in-repository callers with CodeGraph plus literal
   search;
2. state old and new contracts and why compatibility cannot be preserved;
3. choose an additive change, deprecation, compatibility adapter, or explicit
   version migration;
4. update this file and the detailed API/schema documentation;
5. add tests for both the new contract and the intended legacy behavior or
   rejection;
6. validate every supported architecture or name the unvalidated risk;
7. keep the contract migration separate from unrelated optimization work.

Internal refactors do not require deprecation, but they must preserve all public
contracts above and keep required fallbacks buildable and tested.
