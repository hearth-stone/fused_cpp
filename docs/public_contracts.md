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

`PreparedW8A8TiledFusedMoEWeights` is an additive public implementation-selecting
weight type with the same planner-facing shape metadata and these packed fields:

- `w13: tuple[Tensor, int, int, Tensor]`, containing packed INT8 weights,
  logical K/N, and FP32 per-output-channel scales;
- `w2` with the same structure;
- `fused_silu=True`, `gemm_backend=1`, the runtime SVE N tile, and canonical
  name `arm_sve_w8a8_i8mm`.

It is constructed by `prepare_fused_moe_w8a8_tiled_weights` from BF16 weights
or by `prepare_fused_moe_w8a8_tiled_quantized_weights` from checkpoint INT8
weights and FP32 channel scales. It is selected explicitly by passing this
type to `fused_moe_tiled`; BF16 remains the default implementation.

The initial W8A8 execution contract is Linux AArch64 SVE+i8mm, BF16 input and
output, per-row symmetric dynamic A8 quantization, per-output-channel W8,
`activation="silu"`, `swiglu_limit=10.0`, FP32 weighted TopK merge, and a
strict homogeneous fixed-team Plan V2. H and F must be multiples of 16.
Biases, `skip_weighted`, expert-parallel id remapping, shared experts, dynamic
resize/tail-pool plans, and early merge are rejected rather than silently
lowered. There is no unplanned W8A8 fallback; callers must install a compatible
planner or call `fused_moe_w8a8_tiled_async_plan` with an owned plan.

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

The normal, scheduled, async, Plan V2, combined routed+shared, and standalone
shared APIs accept the additive keyword `swiglu_limit`. The only positive value
currently supported is `10.0`, with `activation="silu"`, fused-SiLU packed
weights, and backend `arm_sve_bf16`; it selects the DeepSeek-V4 SVE JIT
clamped-SwiGLU operation. The experimental planned-staged and vLLM-staged
comparators do not expose this mode.

`MoePlannerRuntime` is an additive, process-local scheduling API for the SVE
BF16 fused-SiLU path. Calibration is explicit through
`calibrate_moe_planner_quick`; importing the package and the first operator call
must never start calibration. `set_default_moe_planner_runtime(runtime)` makes
the normal `fused_moe_bf16_tiled` entrypoint use a compatible Plan V2 and
returns the previously installed runtime. Passing `None` restores the existing
native dispatcher. Calls outside the runtime's calibrated CPU, expert-shape,
backend, activation, or thread domain retain the existing dispatcher. The
registry and planner are thread-safe; replacing a runtime does not cancel an
invocation which already obtained the previous object. Runtime route plans are
not cached because routing histograms are request-specific. The initial runtime
uses a bounded homogeneous-team strict-plan search; mixed-width and dynamic-tail
search remain offline behavior and are not part of this initial public runtime
contract.

`MoePlannerRuntime.initialize_planner(max_routes)` explicitly precomputes every
`T_iso(M,T)` scalar for `1 <= M <= max_routes` and every calibrated thread
width. This synchronous deployment step is initialization, not runtime
planning, and may persist the scalar table through the configured cost cache.
Normal dispatch does not implicitly run dense initialization.

`FUSED_CPP_MOE_PLANNER_FIXED_THREADS=8` forces routed-expert production planning
to use homogeneous 8-thread teams with the same descending-route LPT assignment.
Unset or `0` retains calibrated C++ quick width search. Unsupported values fail
explicitly; the control does not alter full offline or synthetic shared-expert
planning.

`MoePlannerRuntime` uses a versioned analytical-cost disk cache by default at
`~/.fused_cpp/cache/moe_costs`; callers may pass `cost_cache_dir=` to relocate
it or `None` to disable it. A cache hit requires exact formula-source/model/calibration,
shape, topology, backend, and supported-width identity. The cache contains only
`T_iso(M,T)` scalars, never packed weights, routing tensors, or Plan V2 payloads.
Missing, stale, malformed, or unwritable cache files must fall back to the same
analytical computation without changing planner results or operator behavior.

`enable_moe_planner_quick(...)` is the deployment convenience API. It runs the
same explicit synchronous quick calibration, constructs a shape-bound runtime,
and installs it only after both steps succeed. After it returns, compatible
calls to the normal fused-MoE entrypoint require no additional planner API.
Deployments requiring disjoint initialization/runtime accounting call
`runtime.initialize_planner(max_routes)` before serving.

Bitwise equality between different backends is not promised unless a test or
operator document explicitly requires it. Tolerance changes need a numerical
justification and an explicit contract update; they must not be relaxed only to
adopt a faster candidate.

### Shared MLP

`prepare_shared_mlp_bf16_tiled_weights` and `shared_mlp_bf16_tiled` are the
standalone one-expert SVE BF16 MLP API. Dense weights use `[2 * F, H]` W13 and
`[H, F]` W2 layouts; the prepared object reuses the opaque E=1 fused-MoE packed
layout with fused SiLU enabled. Execution accepts contiguous CPU BF16 `[M, H]`
input and returns contiguous CPU BF16 `[M, H]` output. A valid `out=` buffer is
written in place and returned. Biases and non-SVE backends are not supported by
this initial version.

N-window selection, M12 row partitioning, dynamic task order, cache sizing,
scratch layout, and thread ownership are internal scheduling details. The
operator is thread-compatible, not thread-safe: callers must serialize
invocations because they use the process-wide resident MoE worker pool.
Prepared weights are read-only and reusable across serialized calls.

`prepare_routed_shared_moe_bf16_tiled_weights` and
`fused_moe_bf16_tiled_with_shared` are the additive combined scheduling API.
The initial contract accepts one shared expert whose `[2 * F, H]` / `[H, F]`
weights match the routed experts' rank-local `H` and `F`. Packing writes routed
experts followed by the shared expert into one opaque SVE allocation without a
full routed-weight copy. Execution appends one unit-weight shared route per
token, scales routed route weights by `routed_scaling_factor`, and performs one
FP32 route merge followed by one BF16 output store.

The combined API supports fused SiLU, standalone and TP only. Passing
`swiglu_limit=10.0` selects DeepSeek-V4 clamped SwiGLU: gate is capped at 10,
up is clamped to `[-10, 10]`, then the existing `silu(gate) * up` epilogue is
evaluated. This mode requires the SVE BF16 JIT; static asm, NEON, x86, biases,
EP, multiple shared experts, and a distinct shared intermediate size are not
supported. `None` and `0.0` retain the existing unclamped behavior bitwise;
other positive limits are rejected. `MoePlannerRuntime(shared_experts=1)` may
select a bounded mixed-width strict plan for this API. Existing runtimes default
to `shared_experts=0`; routed-only API behavior and Plan V2 schema remain
unchanged.

## Async MoE Plan V2

### Experimental W8A16 Plan V2

`prepare_fused_moe_w8a16_tiled_weights` prepares symmetric signed INT8
weights with one FP32 scale per output channel. The opaque prepared object is
accepted by `fused_moe_w8a16_tiled_async_plan`, by the planner-owning
`fused_moe_w8a16_tiled` entrypoint, and by the type-dispatched
`fused_moe_tiled` facade. `fused_moe_w8a16_tiled` requires an explicitly
installed compatible `MoePlannerRuntime`; it does not construct an uncalibrated
fallback plan. `fused_moe_tiled` selects BF16 or W8A16 solely from the prepared
weight type, so existing BF16 calls and defaults are unchanged. By default, the
SVE JIT loads packed INT8 B in the GEMM
K-loop, converts it to BF16 registers immediately before BFMMLA, and applies
packed per-output-channel FP32 scales once after accumulation.
`cache_dequant=True` is an explicit experimental alternative: for tasks with
more than 12 routes, each Plan V2 B window is converted once into thread-local
BF16 scratch and consumed by the existing BF16 JIT kernels. Tasks with at most
12 routes retain register dequantization because they have no cross-panel B
reuse. A zero window still means the full owner stripe; callers evaluating
cache dequantization must provide machine-calibrated nonzero windows when a
full stripe does not fit the intended cache level. Packed formats, expanded
scale vectors, and cache scratch are internal.

`prepare_fused_moe_w8a16_tiled_quantized_weights` accepts checkpoint-native
signed INT8 `[E, 2 * F, H]` / `[E, H, F]` weights and one FP32 scale per
output channel. It preserves the supplied quantized values and scales while
producing the same opaque runtime layout; it never dequantizes or requantizes
the source. `prepare_routed_shared_moe_w8a16_tiled_quantized_weights` appends
one same-shape shared expert without materializing a full `E + 1` source copy.
The returned `PreparedW8A16TiledRoutedSharedMoEWeights` is accepted by
`fused_moe_w8a16_tiled_with_shared`, which requires a shared-aware compatible
`MoePlannerRuntime`, appends one unit-weight shared route per token, and has no
unplanned fallback. Existing BF16 and BF16-source W8A16 entrypoints remain
compatible.

This API is experimental. It supports no bias or non-SVE fallback, requires
the direct-route W2 path, and currently uses FP32 route storage. Quantization
changes numerical behavior relative to BF16 and callers must validate
model-level quality. The planner schedules an explicitly supplied W8A16
prepared object but never converts or selects BF16 weights as W8A16.

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

## DeepSeek V4 Inverse RoPE And Grouped WO_A

`PreparedDeepseekV4InvRopeWoa`, `prepare_deepseek_v4_inv_rope_woa`,
`deepseek_v4_inv_rope_grouped_woa`, and
`deepseek_v4_inv_rope_grouped_woa_torch_reference` are public Python symbols.
The prepare function accepts a contiguous CPU BF16 weight with shape
`[G * R, P * DH]` and records the explicit `G`, `P`, `DH`, and even `RG`
geometry. The execution function accepts:

- `o`: contiguous CPU BF16 `[T, G * P, DH]`;
- `positions`: contiguous CPU int64 `[T]`;
- `cos_sin_cache`: contiguous CPU FP32 `[max_position, RG]`, with cosine in
  the first `RG / 2` columns and sine in the remaining columns;
- optional non-aliasing contiguous CPU BF16 `out`: `[T, G, R]`.

The last `RG` elements of every head use inverse GPT-J RoPE pair semantics
`(even * cos + odd * sin, odd * cos - even * sin)`; the first `DH - RG`
elements are unchanged. Heads are then flattened within each group and
multiplied by that group's WO_A weight with FP32 accumulation and BF16 output.
`T == 0` returns `[0, G, R]` without reading `positions`. Input tensors and the
prepared weight are read-only, and `out` must not alias them.

`backend="auto"` selects a native SVE BF16 implementation only when exported
by the extension and otherwise selects the materialized Torch reference.
`backend="torch"` is the stable correctness fallback. Native packing layout,
thread assignment, and fusion strategy are internal implementation details.

## DeepSeek V4 Multi-Head Hyper-Connections

`PreparedDeepSeekV4MHCWeight` and `prepare_mhc_weight` define the reusable
FP32 projection-weight contract for DeepSeek V4 mHC. Preparation accepts a
contiguous CPU FP32 `[N,C*H]` tensor, owns a copy, infers `C/H`, and requires
`kind="pre"` with `N=C*C+2*C` or `kind="head"` with `N=C`. The only supported
baseline backend is the explicit value `"fp32"`; it never narrows the weight.

`mhc_pre_rmsnorm`, `mhc_post_pre_rmsnorm`, and
`mhc_post_hc_head_rmsnorm` currently dispatch to their corresponding
`*_torch_baseline` functions. Inputs are contiguous CPU tensors: residual and
layer/norm tensors use BF16, while mixing coefficients, scales, bases, and mHC
projection weights use FP32. `num_threads` is validated for API compatibility;
the Torch baseline follows the process-wide Torch intra-op thread setting.

The numerical contract uses FP32 projection, accumulation, sigmoid, softmax,
Sinkhorn, residual mixing, and weighted sums. It explicitly rounds post output
to BF16 before the next pre, rounds pre weighted input to BF16 before RMSNorm,
and rounds HC-head weighted output to BF16 before final RMSNorm. Outputs are
contiguous and have these forms:

- pre: FP32 `[T,C,1]`, FP32 `[T,C,C]`, BF16 `[T,H]`;
- post-pre: BF16 `[T,C,H]` followed by the three pre outputs;
- post-head: BF16 `[T,H]` and BF16 `[T,C,H]` final residual.

All three interfaces accept `T=0` and do not mutate inputs. The current Torch
baseline supports any positive geometry represented by a valid prepared
weight; a future optimized backend may explicitly restrict its supported
geometry without changing baseline semantics.

The explicit `mhc_pre_rmsnorm_sve_candidate` and
`mhc_post_pre_rmsnorm_sve_candidate` entrypoints are experimental and are not
re-exported from the package root. They require SVE128 or SVE256 and fixed
projection width `N=24`; their packed-B layout, approximately 1 MiB K-window,
and M-first/K-fallback thread decomposition are not public contracts. Their
control postprocess is also internal: pre/post sigmoid uses FP32 FEXPA plus a
degree-2 residual polynomial, while the FP32 `4x4` Sinkhorn kernel gathers one
matrix position across the T dimension and retains each vector batch in SVE
registers for all normalization iterations. The candidate materializes FP32
`pre_mix` before a separate H-vectorized SVE consumer performs four-stream
residual reduction, the required BF16 round trip, and in-place two-pass
RMSNorm; `post_mix` and `comb_mix` remain materialized across the sublayer.
The candidate post consumer specializes the fixed `C=4` geometry: it initializes
each FP32 output accumulator from the first residual stream with multiply,
fused-multiply-adds the remaining three residual streams and rank-one layer
injection, then performs the required BF16 rounding and store. This native ABI
is internal and requires contiguous `post_mix` shape `[T,4,1]`.
The explicit `mhc_post_hc_head_rmsnorm_sve_candidate` is also internal and
requires fixed `C=4`. It reuses the native SVE post boundary, computes the
FP32 `[T,4H] x [4H,4]` head projection with a VL-independent NEON M12xN4
kernel, applies FP32 head sigmoid controls, performs the four-stream BF16
reduction boundary, and runs final RMSNorm. The public final-head entrypoint
continues to dispatch to the Torch baseline.

## DeepSeek V4 Attention W8A8 Projections

`PreparedDeepSeekV4W8A8LinearWeight`,
`prepare_deepseek_v4_w8a8_linear_weight`, and
`prepare_deepseek_v4_w8a8_linear_quantized_weight` provide an explicit W8A8
selection for the TP-local `attn.wq_b`, `attn.indexer.wq_b`, and `attn.wo_b`
linear projections. The BF16 prepare function quantizes `[N, K]` weights once;
the quantized prepare function accepts checkpoint INT8 `[N, K]` weights and
contiguous FP32 per-output-channel scales. Packed tensors are opaque and are
valid only for their compatible build and ISA.

Execution accepts contiguous CPU BF16 `[M, K]` input, applies symmetric
per-row dynamic A8 quantization, accumulates the INT8 GEMM into INT32, converts
and applies row/output-channel scales in the SVE microkernel registers, and
stores contiguous CPU BF16 `[M, N]`. The generic linear API may instead request
direct FP32 store. Logical-N tails use final-dtype padded scratch and crop; no
SVE W8A8 path materializes a full FP32 accumulator.
`deepseek_v4_w8a8_linear_pair` shares one activation quantization between the
Main-Q and Indexer-Q projections. On SVE it also shares one packed-A buffer and
one M12/M8 worker pool; approximately 1 MiB packed-B N windows from both
projections participate in the same task graph. Dynamic activation
quantization, logical-N copies, and fallback row work use the extension's
OpenMP runtime rather than the Torch intra-op pool. `deepseek_v4_wo_b_w8a8`
computes only the TP-local output projection; the caller remains responsible for the existing TP
collective and any following operation.

`PreparedDeepSeekV4PostGemmW8A8Weights` selects W8A8 Main-Q and optional
Indexer-Q execution in `post_gemm_parallel_stage_cpp_prepacked`. Dense, C128A,
and C4A postprocessing semantics remain unchanged. The C4A select-all path does
not compute the unused Indexer-Q projection.

This implementation is supported only by compatible Linux AArch64 SVE+i8mm
builds. Selection is explicit through a prepared W8A8 type; BF16 packing and
dispatch remain the default. The supported numerical contract is the result of
the documented per-row/per-output-channel quantization, not equivalence to the
BF16 projection. The native projected postprocessing entrypoints and the
generic paired i8gemm binding are internal-stable implementation details.

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

For supported SVE BF16 MQA shapes, the public optimized entrypoint uses
head-major scalable 8x2VL QK/PV kernels for both the gathered sparse fast path
and the fully shared contiguous-dense fast path once query parallelism is
sufficient. The dense path packs its common K/V interval directly into the
scalable layout once per call. Non-SVE builds retain fixed 8x8 head-major
kernels. Short query chunks retain guarded 2D KV partitioning; FP32 and
unsupported BF16 shapes retain the indexed fallback. Task layout, packing,
split count, partial buffers, and merge order are internal. Every dispatch must
preserve the documented numerical tolerances, return-statistics definitions,
attention-sink semantics, and safe fallback for unsupported or non-canonical
sparse patterns.

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
