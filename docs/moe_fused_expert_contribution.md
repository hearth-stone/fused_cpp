# Moldable Fused-Expert Execution: Implementation and Evidence Dossier

Status date: 2026-08-27.

This document is the paper-drafting source for the fused-expert part of the CPU
MoE project. It records what was implemented, why each mechanism exists, the
current numerical and execution contracts, reusable evidence, negative results,
and open gates. It is deliberately more precise than a paper section: a later AI
or human author should compress it into a narrative without broadening its
claims.

This is not the source of truth for planner or cost-model semantics. Use
[`../cpu_moe_schedule_optimization/MATHEMATICAL_MODEL.md`](../cpu_moe_schedule_optimization/MATHEMATICAL_MODEL.md)
for those components and
[`moe_paper_readiness.md`](moe_paper_readiness.md) for the paper-wide claim and
evaluation gates.

## Technical summary

The implemented contribution is a **moldable fused-expert execution substrate
for many-core Arm CPUs**, not one monolithic kernel and not a claim that the
entire MoE layer is fused. It transforms each routed expert into a
non-preemptive task whose team width and per-stage packed-weight window are
explicit scheduling controls. Within a task, it removes or compresses the main
materialization boundaries:

1. routed BF16 token rows are gathered directly into the packed-A layout;
2. W13 computes gate/up, applies SwiGLU, and produces W2-ready BF16 packed-C;
3. W2 consumes packed-C and writes its FP32 result directly to token-major
   route rows;
4. a deterministic FP32 weighted reduction produces one BF16 token output.

The primary paper path uses BF16 inputs and weights, FP32 GEMM accumulation, a
BF16 packed SwiGLU intermediate, FP32 route storage and accumulation, and one
final BF16 store. The current Arm SVE implementation additionally contains
exact-M JIT kernels, adaptive gather work sharing, stage tile windows, ready-token
merge overlap, and explicit W8A16/W8A8 extensions. BF16 is the production and
paper-reference path; W8A16 is experimental; W8A8 is an enabled explicit
narrow-domain path but is not an automatic default.

The implementation is substantially complete. The remaining fused-expert work
is evidence closure: a frozen-build cumulative ablation, a current upstream Arm
baseline, a repaired same-activation explicit pipeline, full captured routing,
multi-layer integration, second-machine repeats for all headline mechanisms,
and model-level quality for quantized paths.

## Paper-facing claim and novelty boundary

The recommended claim is:

> We design a moldable fused-expert substrate for BF16-capable many-core Arm
> CPUs. It shares routed packed-A construction across W13 owners, specializes
> exact expert row counts, produces a W2-ready packed activation in the W13
> epilogue, stores W2 results directly into route order, and exposes team width
> and packed-weight tile windows through a planner-executable task ABI.

The paper should not reduce this contribution to "fusing SiLU with GEMM."
Existing CPU MoE implementations already fuse common activation boundaries.
The specific combination implemented here is:

- exact-M Arm SVE execution for routed small-M GEMMs;
- one shared packed-A object per expert stage rather than one gather per N
  owner;
- fused W13, SwiGLU, BF16 conversion, and W2 packed-C production;
- W2 direct token-major route output;
- deterministic FP32 TopK reduction;
- task-local team width and W13/W2 owner-window controls;
- an executable Plan V2 boundary connecting these controls to the runtime
  planner.

The contribution should also not be described as whole-layer fusion. Routing,
route-metadata construction, W13, W2, and final merge remain distinct logical
stages, and W13 and W2 are separate GEMM kernels. The contribution is a fused
**expert dataflow and execution substrate**.

## Scope and notation

The initial paper scope is:

- Linux AArch64;
- Arm SVE with BF16/SVEBF16 support;
- standalone or tensor-parallel rank-local expert execution;
- fixed packed expert weights;
- non-preemptive expert tasks;
- prefill-oriented routing, with decode/small-M included as a boundary case;
- one NUMA-local CPU set per rank;
- FP32 route storage for headline numerical results.

Let:

- `T` be the number of input tokens;
- `K` be router TopK;
- `E` be the number of locally packed experts;
- `H` be hidden size;
- `F` be the rank-local expert intermediate size;
- `M_e` be the routed row count of expert `e`;
- `v` be the backend BF16 N tile;
- `p_{t,k}` be the router weight for token `t`, slot `k`.

The routed expert computation is:

```text
[gate, up] = X_e @ W13_e^T
z          = SiLU(gate) * up
r_e        = z @ W2_e^T
y_t        = sum_k p[t,k] * r[t,k]
```

The DeepSeek-V4 extension clamps gate/up before the SwiGLU operation when
`swiglu_limit=10.0` is explicitly selected.

## Primary numerical contract

The default path has the following boundaries:

```text
BF16 token input
  -> BF16 W13 BFMMLA with FP32 accumulators
  -> FP32 SwiGLU epilogue
  -> BF16 W2-ready packed-C
  -> BF16 W2 BFMMLA with FP32 accumulators
  -> FP32 route_out
  -> FP32 weighted TopK reduction
  -> BF16 token output
```

| Value | Shape | Default dtype | Notes |
| --- | --- | --- | --- |
| Input | `[T,H]` | BF16 | Contiguous CPU tensor |
| W13 | `[E,2F,H]` before packing | BF16 | Gate/up layout |
| W2 | `[E,H,F]` before packing | BF16 | Down projection |
| Router weights | `[T,K]` | Any floating input, converted to FP32 | CPU |
| Router ids | `[T,K]` | Integer | Local packed-expert indices |
| Packed SwiGLU intermediate | expert-local | BF16 | Direct W2 input |
| Route output | `[T*K,H]` | FP32 | Paper default |
| Final output | `[T,H]` | BF16 | Optional caller-owned `out=` |

An experimental BF16 route buffer halves post-W2 logical bytes while keeping
the final reduction in FP32. It is not a primary numerical path because it adds
a rounding boundary and lacks model-level quality validation.

For `TopK=1`, `skip_weighted=true` writes W2 output directly to the final BF16
tensor and skips route merge. For larger TopK, W2 produces one route row and
merge performs FP32 weighting and accumulation.

The SVE fixed-TopK merge templates for `K=2/4/6/8` use deterministic adjacent
tree association. For example, TopK=6 is:

```text
((slot0 + slot1) + (slot2 + slot3)) + (slot4 + slot5)
```

Each leaf is first multiplied by its router weight. This preserves deterministic
slot membership but is not bitwise equivalent to a sequential FP32 FMA loop.
Other TopK values use the runtime-loop reduction. A paper must not claim that
all fixed-TopK paths preserve sequential-FMA association.

## Implemented expert pipeline

### Backend-specific reusable weight preparation

Weights are packed once outside the request path into an opaque backend-specific
object. The object records the backend id, N tile, padded dimensions, packed
strides, and whether the W13 layout is fused-SiLU compatible.

Implemented BF16 backends include:

- `arm_neon_bf16`, N tile 8, requiring Arm BF16/BFMMLA;
- `arm_sve_bf16`, N tile derived from the build-time SVE vector length;
- `x86_avx512_bf16`, N tile 32;
- `x86_amx_bf16`, N tile 32.

The paper scope is Arm SVE. NEON and x86 demonstrate internal backend separation
and provide compatibility paths; they are not current cross-ISA performance
claims. SVE packed weights are vector-length-specific, and import rejects a
runtime SVE VL that differs from the build.

The SVE JIT kernels are prewarmed during weight preparation, outside forward
timing. This removes code generation from the measured request path but means
paper reproducibility must separately report preparation and cold-start cost.

### Shared adaptive gather-pack-A

The original expert path first materialized routed input rows and then packed
them for GEMM. The fused gather consumes token-major input plus route metadata
and writes directly into the M12/M8 packed-A layout.

The packed-A object is shared by all W13 N owners. Gather work is not replicated
per owner. When complete physical M panels are fewer than team threads, the
implementation flattens `(M panel, K stripe)` into a cooperative work domain:

- K stripes begin on K8 boundaries;
- every stripe contains at least 32 BF16 elements;
- M12/M8 stripe boundaries advance output by 192/128 bytes;
- workers therefore do not write the same cache line;
- adjacent stripes assigned to one worker are coalesced;
- the pre-W13 barrier remains because all N owners consume the complete packed
  A.

This is an input-packing work-sharing optimization, not an M-split GEMM. W13
still uses N ownership.

### Exact-M SVE JIT execution

Static kernels historically mapped small routed experts to M1/M2/M4/M8/M12
families and sometimes executed padded row pairs. The JIT generates the existing
packed-A BFMMLA body for every logical `M=1..12`:

- M1-M8 retain the physical M8 panel;
- M9-M12 retain the physical M12 panel;
- only `2*ceil(M/2)` logical rows execute;
- W13 and W2 share the production packed-A ABI;
- full M12 is a neutral control rather than an expected speedup.

The M1-M8 generated K loop uses the production two-bank load/compute state
machine. M9-M12 use the M12 register-reuse schedule because their accumulator
set cannot coexist with two complete A/B banks. Static assembly remains the
fallback for unsupported JIT surfaces and explicitly static-only epilogues.

### Fused W13, SwiGLU, and packed-C production

The W13 kernel computes gate and up projections together. Its epilogue:

1. evaluates the activation in FP32;
2. multiplies gate and up;
3. converts once to BF16;
4. writes the W2 packed-A layout directly.

This removes separate FP32 gate/up tensors, a materialized SiLU tensor, a
materialized product tensor, and a second pack-A pass before W2.

The SVE compatibility selectors `silu_poly_degree=4/5/6` now share one
FEXPA+degree-2 residual evaluator. The current evaluator improved isolated SiLU
latency and accuracy but was approximately neutral for full standard-SiLU W13;
its stronger complete-expert result is on the clamp-10 path. Historical
poly5-based explicit-fusion comparisons therefore cannot be directly rerun
against the current production fused path without updating their activation
contract.

Supported activation surfaces are broader at the generic API than in the paper
path. Materialized BF16 execution supports SiLU, GELU, and `swigluoai`; the
optimized Arm SVE paper backend requires fused SiLU, with optional DeepSeek-V4
clamp-10 through the SVE JIT.

### W2 consumes packed-C and writes route order directly

W2 consumes the BF16 packed-C produced by W13 without another layout conversion.
Its N owners hold disjoint H-column ranges. The default SVE direct-route
epilogue uses each expert's route-row table and writes those disjoint columns
directly into token-major FP32 `route_out`.

The replaced path was:

```text
W2 -> per-team contiguous down -> owner scatter -> route_out -> merge
```

The current default is:

```text
W2 -> route_out -> merge
```

If `R = routes * H * sizeof(float)`, the old logical post-W2 payload was
approximately `4R`; direct route reduces it to `2R`. The fallback remains
available when the direct-store preconditions fail, including unsupported bias
or offset cases.

### Deterministic FP32 weighted merge

The SVE merge keeps an H-vector accumulator in registers, multiplies each route
row by its FP32 router weight, reduces TopK, converts once, and stores BF16
output. This removed the former per-worker FP32 row allocation and repeated
accumulator traffic.

The asynchronous executor can additionally merge a token after every expert in
its TopK set completes. The accepted readiness design uses one completion RMW
per expert, one CAS per ready token, and batched publication. Expert compute
remains preferred over merge work. This mechanism is enabled on its supported
path primarily to hide expert-tail time; measured E2E gains so far are below
0.4% on the tested balanced and two-group distributions.

### Moldable teams and per-stage tile windows

Each expert task is non-preemptive and covers complete W13 and W2 stages. The
runtime exposes two scheduling controls:

- `threads`: task team width;
- `window_tiles`: the number of packed-B N tiles processed per worker in one
  serial owner window.

A team window covers:

```text
threads * window_tiles
```

N tiles, and a stage uses:

```text
ceil(total_tiles / (threads * window_tiles))
```

windows. `window_tiles=0` selects each worker's full owner stripe. Windows
reorder complete-N traversal; they do not partition a task into independently
schedulable weight ranges.

This control exists because team width and private-L2 owner-stripe size are
coupled. A wider team reduces each worker's stripe but increases coordination
and concurrent resource demand. A smaller window can improve packed-B cache
behavior but repeats stage traversal and synchronization. The cost model and
planner reason about these explicit choices.

### Plan V2 execution boundary

The kernel substrate is connected to the planner through Plan V2 rather than a
paper-only schedule description. A task record carries:

- expert id and route range;
- core interval and team width;
- dependencies and placement metadata;
- preferred/minimum/maximum width metadata;
- W13 and W2 tile windows;
- optional bounded route slicing;
- strict or tail-pool execution mode.

Current production tasks do not resize while running. "Moldable" means the
planner selects a legal shape before task execution. The native executor supports
strict whole-expert tasks, whole-expert tail-pool placement, and bounded
route-sliced strict tasks.

### Routed and shared expert composition

One same-shape shared expert can be represented as an all-token synthetic expert
inside the same Plan V2 execution. The quick planner searches a bounded
`shared_width + routed_width` family, and the shared lane may execute routed
work after shared completion. Routed scaling, the shared contribution, and
final accumulation remain FP32 before one BF16 store.

This path is implemented but is not part of the initial fused-kernel headline
unless it is validated with the real model rather than only synthetic shared
workloads.

## Mechanism inventory

| Mechanism | Problem addressed | Current state | Strongest reusable evidence | Remaining gate |
| --- | --- | --- | --- | --- |
| Shared fused gather-pack | Extract+pack traffic and idle gather workers | Production | 2.3-12.6% complete-call gain for sampled underfilled M1-37 on the 8-core host | Second-machine E2E repeat |
| Exact-M JIT | M8/M12 tail overcompute | Production | Gains concentrated at M5/6 and M9/10 on two Arm machines; M12 neutral | Frozen-build full matrix |
| Double-buffered small-M K loop | Generated-kernel load latency | Production | Restored M<=8 controls to neutral/near-neutral while retaining exact-M wins | No new gate beyond frozen repeat |
| W13+SwiGLU+packed-C | Intermediate tensors and second pack-A | Production | Historical same-GEMM explicit fusion reduced latency 3.29-5.63% as the working set grew, with lower LLC misses/stalls | Rebuild explicit baseline with current FEXPA activation |
| FEXPA+poly2 SiLU | Activation latency and approximation error | Production SVE | Isolated SiLU improved 18-34%; full standard-SiLU W13 was neutral, clamp path up to 1.53% complete-expert gain | Model-level quality for approximate activation |
| W2 FP32 direct route | Per-team down buffer and scatter | Production default | M192 +2.18%; M1536 +15.82%; M2040 +16.44%; short routes near noise | Captured distributions on frozen build |
| BF16 direct route | Halve route traffic again | Experimental | M1536 +4.68% versus FP32 direct; short M mostly neutral | Model-level quality |
| Register-resident SVE merge | Worker FP32 accumulator allocation/traffic | Production | Small E2E gain, but removes hot-path allocation | Fixed-TopK numerical disclosure and broader shapes |
| Ready-token merge | Hide merge behind expert tails | Production on supported async path | Less than 0.4% change on existing balanced/skew tests; one trace merged 605/2048 tokens early | Captured multi-wave heavy tails |
| Per-stage tile windows | Owner-stripe cache pressure | Production policy on declared domains | Recorded planner A/B: +12.4% uniform, +10.6% captured DSV4 | Frozen-binary rerun and wider holdout |
| Plan V2 task ABI | Make kernel controls executable | Production | Strict, tail-pool, and bounded route-sliced execution tested | Paper pseudocode and current end-to-end matrix |

## Quantized execution extensions

The low-precision extensions are evidence that the dataflow can survive weight
compression and INT8 compute. They are not currently co-equal paper paths.

### W8A16: compressed weights with BF16 compute

W8A16 uses BF16 activation inputs and per-output-channel INT8 weights with FP32
scales. It accepts either BF16 source weights for offline quantization or
checkpoint-native INT8+scale tensors.

Two execution modes exist:

- register dequantization converts W8 fragments inside each GEMM K loop;
- cache dequantization converts a selected packed-B window once into
  thread-local BF16 scratch and reuses the BF16 JIT kernel across M panels.

The latter makes tile-window selection part of the precision-specific execution
policy. Register dequantization can win for cold/short experts but can regress
when long experts reuse B and repeatedly pay conversion. The complete W13,
SwiGLU, W2 direct-route, FP32 merge, BF16-output path exists, including a
routed+shared wrapper.

Current status: experimental. Missing pieces are model-level quality, a cost
model that accounts for dequantization and BF16 scratch, automatic selection of
BF16 versus register/cache W8A16, an intrinsic unplanned fallback, and broader
cross-machine validation.

Recorded evidence includes:

- V4-Pro TP4 balanced: 101.540 to 51.555 ms (1.970x), with 50.21% packed bytes;
- V4-Flash TP2 balanced: 56.565 to 34.804 ms (1.625x);
- captured long-route register dequant: 29.135 to 36.013 ms, a regression;
- captured cache dequant at a searched window: 27.378 to 20.565 ms (1.331x);
- Arm-codex TP4 follow-up: 36.042 to 34.403 ms (1.0476x), below the 5% adoption gate.

### W8A8: dynamic activation and weight INT8 compute

W8A8 retains a BF16 public input but dynamically quantizes every route row to
signed INT8, applies per-output-channel W8 scales, runs SMMLA with INT32
accumulators, dequantizes the W13 result, computes clamp-10 SwiGLU, rounds the
intermediate to BF16, quantizes it again, runs W2, and writes FP32 route output.
The existing FP32 weighted merge and BF16 final store are reused.

Its explicitly supported domain is narrow:

- Arm SVE+i8mm;
- H and F multiples of 16;
- strict homogeneous fixed-team Plan V2;
- local expert ids and weighted TopK;
- `activation=silu` with `swiglu_limit=10.0`;
- no bias, EP remapping, general shared expert, dynamic tail, early merge, or
  unplanned fallback.

Within that domain the end-to-end kernel pipeline and explicit public entrypoint
are implemented. It is not selected automatically. Arm-codex captured TP4
measured BF16 at 36.29-36.44 ms and packed-M12 W8A8 at 12.108-12.300 ms, but
model-level quality and held-out shapes remain open adoption gates.

### Precision status summary

| Path | Kernel pipeline | System integration | Model quality | Paper role |
| --- | --- | --- | --- | --- |
| BF16 | Complete | Broad production integration | Primary reference | Main path |
| W8A16 | Complete vertical path | Partial; Plan V2 and policy gaps | Open | P0 quantized extension |
| W8A8 | Complete in a narrow domain | Explicit strict-plan path only | Open | Secondary/appendix extension |

## Evidence that can be reused

### Exact-M and short-expert behavior

The exact-M implementation has evidence on two Arm machines. Gains occur where
the JIT executes fewer BF16 row pairs than the static bucket, especially M5/6
and M9/10. M12 is the same work and should be treated as a neutral control.

The 2026-08-27 cross-machine framework smoke reproduced this direction, but it
used only three samples. On Arm-codex, M5/6 1T gains were approximately 28-29%
and M9/10 gains were approximately 13-14% across tested widths. On the AWS
8-core host, target gains were mostly 6-14%. The AWS M12/8T control was unstable
and is not admissible evidence.

Primary source:
[`../optimizations/fused_moe_sve/results/amazon_8c_192c_xbyak_exact_m.md`](../optimizations/fused_moe_sve/results/amazon_8c_192c_xbyak_exact_m.md).
Framework smoke:
[`../optimizations/fused_moe_sve/results/arm_codex_aws8_fused_expert_smoke_20260827.md`](../optimizations/fused_moe_sve/results/arm_codex_aws8_fused_expert_smoke_20260827.md).

### Fusion mechanism evidence

The historical same-GEMM explicit pipeline measured 3.29-5.63% lower latency
for the fused path as routed working sets increased. PMU runs also reported
lower LLC read misses and backend memory-stall cycles. The experiment isolates
the dataflow mechanism and is not an upstream-system comparison.

It is historical only. A current rerun is blocked because its explicit path
evaluates poly5 while current fused W13 evaluates FEXPA+poly2. F512 and F2048
controls both failed the comparator's predeclared relative-L2 limit. No new
timing should be reported until both variants have identical activation and
rounding semantics.

Source:
[`../optimizations/fused_moe_sve/results/amazon_192c_unfused_pipeline.md`](../optimizations/fused_moe_sve/results/amazon_192c_unfused_pipeline.md).

### Direct-route evidence

The FP32 direct-route path is bitwise identical to FP32 scatter in focused
tests. Its benefit grows with route count:

| Routes per expert | Recorded E2E effect |
| ---: | ---: |
| 12 | -0.95%, within the short-route noise region |
| 48 | no measurable gain in the repeated control |
| 192 | +2.18% |
| 1536 | +15.82% |
| 2040 | +16.44% |

The correct paper interpretation is not that direct route always speeds up an
expert. It removes one materialization/scatter boundary and strongly benefits
large route tensors; short calls are dominated by fixed kernel and scheduling
costs.

Source:
[`../optimizations/fused_moe_sve/results/amazon_192c_w2_direct_route.md`](../optimizations/fused_moe_sve/results/amazon_192c_w2_direct_route.md).

### Merge evidence

The retained SVE merge implementation produced only a sub-1% E2E change in the
recorded TopK=6 full operator, but it removes a hot-path per-worker FP32
allocation. BF16-route isolated merge gains are larger because source traffic
is halved, but model quality is still open.

Source:
[`../optimizations/fused_moe_sve/results/amazon_192c_route_merge_tree.md`](../optimizations/fused_moe_sve/results/amazon_192c_route_merge_tree.md).

### Tile-window evidence

The recorded stage-window policy A/B improved uniform and captured DSV4 calls
by approximately 12.4% and 10.6%. This is evidence that stage traversal and
owner-stripe cache fit are useful controls. It must be rerun with the frozen
paper binary and matching current profile before becoming a headline result.

Source:
[`../optimizations/fused_moe_sve/results/amazon_192c_stage_window_tiles_20260810.md`](../optimizations/fused_moe_sve/results/amazon_192c_stage_window_tiles_20260810.md).

## Negative results and intentionally bounded fusion

Negative results are part of the design record and should inform, not clutter,
the paper narrative.

### Full W13-to-W2 kernel fusion is deprioritized

Avoiding the BF16 `[M,F]` packed intermediate would require either recomputing
W13 for different W2 H tiles or retaining/spilling much larger FP32 partial W2
outputs. No loop ordering and traffic proof currently justifies this change.

### Full W2-to-merge fusion is deprioritized

An expert-major W2 schedule completes one token's TopK routes in different
teams and at different times. Preserving deterministic FP32 reduction without
`route_out` would require contended atomics, a rendezvous buffer, or a
token-major schedule that sacrifices expert-weight locality. Reconsider only
if a real multi-layer workload shows merge plus route traffic remains at least
10-15% of local MoE latency after overlap.

Multiplying router weights in the W2 epilogue is only a low-priority bounded
experiment. It leaves route write/read traffic unchanged and changes rounding
from merge FMA to pre-rounded multiplication plus addition.

### K-blocking and deeper microkernel changes were not universally beneficial

The active SVE BF16 path returned to the one-chunk non-interleaved K loop after
K-block and ILV variants failed to show a consistent full-operator advantage.
The current direction is to preserve the closed compute baseline and improve
system evidence rather than accumulate default-off kernel branches.

### Ready-token merge is a latency-hiding mechanism, not a proven headline gain

The first per-route atomic readiness design regressed by 23.35%. The accepted
expert-completion design removed that contention, but existing E2E changes are
below 0.4%. It belongs in the design as a bounded overlap mechanism and in an
ablation only after captured heavy-tail validation.

## Implementation completeness

### Complete or substantially complete

- BF16 input/weight/output contract;
- backend-specific reusable packing;
- Arm SVE/NEON runtime dispatch and fallback;
- SVE exact-M JIT and static fallback;
- adaptive gather-pack;
- fused W13+SwiGLU+packed-C;
- W2 FP32 direct route plus scatter fallback;
- deterministic SVE weighted merge;
- normal, scheduled, async, and Plan V2 execution;
- strict, tail-pool, and bounded route-sliced task lowering;
- caller-owned final output;
- one synthetic shared-expert composition;
- explicit W8A16 and W8A8 vertical pipelines;
- focused correctness and mechanism benchmarks.

### Paper-critical implementation or artifact gaps

- repair the current same-activation explicit-unfused control;
- pin or commit the ignored external `refs/i8gemm/lib` source rather than only
  recording its content hash;
- freeze one source/extension/profile identity;
- refresh the empirical profiles after the final default kernel/dataflow;
- collect complete multi-layer TopK traces;
- run a real multi-layer vLLM integration;
- measure TP/EP communication before distributed claims;
- close W8A16/W8A8 model quality;
- make precision-specific cost and planner decisions explicit before claiming
  automatic low-precision selection.

## Claims supported now

The current implementation and evidence support these bounded statements:

- the Arm SVE executor implements a fused, moldable expert substrate with
  explicit task width and per-stage tile-window controls;
- exact-M generation removes selected small-M tail overcompute without changing
  the packed-A ABI;
- shared adaptive gather-pack activates otherwise idle team workers on
  underfilled expert calls;
- fused W13 produces a W2-ready BF16 packed intermediate without materializing
  the explicit activation pipeline;
- FP32 W2 direct route removes the per-team down buffer and scatter traffic,
  with the strongest measured benefit on long routes;
- Plan V2 makes width/window/task dependencies executable in the native
  runtime;
- the same dataflow has explicit W8A16 and W8A8 extensions, subject to their
  declared domains and open quality gates.

## Claims not supported yet

A draft must not claim:

- state-of-the-art or best CPU MoE performance;
- a current general speedup over upstream vLLM, oneDNN, or another production
  baseline;
- one monolithic kernel fuses the whole MoE layer;
- current explicit-fusion speedup is 3.29-5.63% without labeling it historical;
- all fixed-TopK reductions preserve sequential FP32 FMA association;
- BF16 route storage is quality-neutral at model level;
- W8A16 or W8A8 preserves model quality;
- W8A16/W8A8 is automatically selected by the production planner;
- x86, NEON, or cross-ISA performance portability from Arm evidence;
- ready-token merge materially improves E2E latency on general workloads;
- full-layer, full-model, serving, TP, or EP speedup from operator-only timing.

## Recommended paper-section structure

The fused-expert section can be drafted in this order.

### 1. Motivation: routed experts are moldable and memory-sensitive

Explain that per-expert M varies after routing, while H/F and packed weights are
fixed. A fixed team width creates underfilled small-M calls, tail overcompute,
and non-monotonic cache behavior. Materialized activation and route boundaries
become expensive as routed working sets grow.

### 2. Design overview: one expert task, two GEMM stages, explicit controls

Introduce an expert as a non-preemptive W13->W2 chain with one preselected team
width and separate W13/W2 tile windows. Show the dataflow from token-major input
to packed-A, fused W13 packed-C, direct-route W2, and final FP32 merge.

### 3. Exact-M and cooperative input construction

Describe the M8/M12 physical layouts, logical exact-M execution, JIT prewarming,
and adaptive `(M panel,K stripe)` gather work sharing. Emphasize shared packed-A
ownership across W13 N owners.

### 4. Fused W13-to-W2 activation boundary

Describe FP32 accumulation, activation, one BF16 rounding point, and direct
packed-C production. Avoid presenting generic SwiGLU fusion as the novelty;
focus on producing the exact W2 input layout and avoiding materialized stages.

### 5. Direct route output and deterministic reduction

Explain why N owners can directly write disjoint H columns and quantify the
logical traffic change from `4R` to `2R`. Describe the fixed-TopK tree and
runtime fallback accurately.

### 6. Planner-executable moldability

Define team width and window tiles, explain their cache/parallelism trade-off,
and show how Plan V2 carries the decision to the executor. This subsection is
the bridge to the cost-model and planner contributions rather than a second
planner description.

### 7. Quantized extensions and limitations

Use BF16 as the numerical reference, W8A16 as the primary low-precision
extension, and W8A8 as a narrow INT8-compute extension. State that quality and
automatic precision selection remain open.

## Instructions for an AI drafting the first paper version

### Terminology to use

- "moldable fused-expert execution";
- "expert-major execution";
- "shared packed-A ownership";
- "exact-M SVE JIT";
- "W2-ready BF16 packed-C";
- "FP32 direct-route output";
- "deterministic weighted TopK reduction";
- "team width" and "per-stage tile window";
- "planner-executable Plan V2 task ABI".

### Terminology to avoid

- "fully fused MoE layer";
- "single fused MoE kernel";
- "ordered reduction" when it implies sequential-FMA association;
- "precision agnostic" before W8 cost/planner integration is complete;
- "production W8A8 default";
- "model-quality preserving" for any unvalidated quantized path;
- "general CPU portability" from Arm-only measurements.

### Facts that must remain paired with caveats

- Pair the 3.29-5.63% fusion result with "historical same-GEMM control; current
  FEXPA comparator must be rebuilt."
- Pair direct-route 15-16% gains with "long routes" and the neutral short-route
  controls.
- Pair adaptive gather gains with "underfilled 8T points on one performance
  machine."
- Pair tile-window 10-12% gains with "recorded planner A/B requiring frozen
  rerun."
- Pair W8A16 speedups with the captured register-dequant regression and open
  model-quality gate.
- Pair W8A8 near-3x captured speed with its strict clamp-10 domain and open
  model-quality gate.
- Pair ready-token overlap with its sub-0.4% measured E2E change.

### Preferred contribution paragraph template

A first draft may adapt the following structure but should not copy it as an
unsupported abstract claim:

> We implement each routed expert as a moldable non-preemptive task whose team
> width and packed-weight traversal are selected before execution. The task
> gathers routed BF16 rows directly into a shared packed-A layout, executes
> exact-M SVE W13 kernels whose epilogue produces W2-ready packed SwiGLU output,
> and lets disjoint W2 N owners store FP32 columns directly into token-major
> route rows. A deterministic FP32 reduction completes TopK routing. These
> controls are encoded in an executable Plan V2 interface, enabling the cost
> model and planner to reason about parallelism, cache residency, and memory
> contention without changing expert arithmetic or router semantics.

### Suggested prompt for generating a first draft

```text
Read docs/moe_fused_expert_contribution.md and docs/moe_paper_readiness.md as
the factual contract. Draft the Moldable Fused-Expert Execution section of a
systems paper for a technical audience. Explain the problem, dataflow,
exact-M/adaptive gather mechanisms, fused W13 packed-C boundary, W2 direct-route
store, deterministic merge, and planner-executable width/window controls.

Use only claims marked as currently supported. Preserve every caveat attached
to a quantitative result. Treat historical, experimental, and production
evidence as distinct. Do not invent measurements, citations, model-quality
results, upstream comparisons, or whole-model speedups. Mark missing evidence
as TODO rather than filling it by inference. Describe BF16 as the primary
numerical reference, W8A16 as the main quantized extension, and W8A8 as a
narrow-domain secondary extension. Do not call the implementation a single
fully fused MoE-layer kernel.

Return: (1) a section outline, (2) a first prose draft, (3) proposed figure and
table captions, and (4) a list of claims that still need experiments.
```

## Source map

Implementation:

- native Arm executor:
  [`../csrc/moe/arm/common/fused_moe_bf16_tiled.cpp`](../csrc/moe/arm/common/fused_moe_bf16_tiled.cpp);
- SVE JIT:
  [`../csrc/moe/arm/sve_bf16/jit_kernels.cpp`](../csrc/moe/arm/sve_bf16/jit_kernels.cpp);
- static SVE kernels:
  [`../csrc/moe/arm/sve_bf16/kernels.S`](../csrc/moe/arm/sve_bf16/kernels.S);
- SVE route merge:
  [`../csrc/moe/arm/sve_bf16/route_merge.cpp`](../csrc/moe/arm/sve_bf16/route_merge.cpp);
- W8A8 kernels:
  [`../csrc/moe/arm/i8mm_w8a8/kernels.cpp`](../csrc/moe/arm/i8mm_w8a8/kernels.cpp);
- Python public wrapper:
  [`../src/fused_cpp/moe/bf16_tiled.py`](../src/fused_cpp/moe/bf16_tiled.py);
- backend registry:
  [`../csrc/moe/common/backend.cpp`](../csrc/moe/common/backend.cpp).

Contracts and lifecycle:

- integration and tensor contract:
  [`vllm_bf16_tiled_moe_integration.md`](vllm_bf16_tiled_moe_integration.md);
- backend layout:
  [`../csrc/moe/README.md`](../csrc/moe/README.md);
- optimization manifest:
  [`../optimizations/fused_moe_sve/manifest.yaml`](../optimizations/fused_moe_sve/manifest.yaml);
- implementation checklist:
  [`../cpu_moe_schedule_optimization/TODO.md`](../cpu_moe_schedule_optimization/TODO.md);
- paper-wide evidence map:
  [`moe_paper_readiness.md`](moe_paper_readiness.md).

Primary result reports:

- explicit fusion:
  [`../optimizations/fused_moe_sve/results/amazon_192c_unfused_pipeline.md`](../optimizations/fused_moe_sve/results/amazon_192c_unfused_pipeline.md);
- exact-M:
  [`../optimizations/fused_moe_sve/results/amazon_8c_192c_xbyak_exact_m.md`](../optimizations/fused_moe_sve/results/amazon_8c_192c_xbyak_exact_m.md);
- adaptive gather-pack:
  [`../optimizations/fused_moe_sve/results/amazon_8c_adaptive_mk_gather_pack_20260812.md`](../optimizations/fused_moe_sve/results/amazon_8c_adaptive_mk_gather_pack_20260812.md);
- W2 direct route:
  [`../optimizations/fused_moe_sve/results/amazon_192c_w2_direct_route.md`](../optimizations/fused_moe_sve/results/amazon_192c_w2_direct_route.md);
- route merge:
  [`../optimizations/fused_moe_sve/results/amazon_192c_route_merge_tree.md`](../optimizations/fused_moe_sve/results/amazon_192c_route_merge_tree.md);
- ready-token merge:
  [`../optimizations/fused_moe_sve/results/amazon_192c_async_ready_token_merge.md`](../optimizations/fused_moe_sve/results/amazon_192c_async_ready_token_merge.md);
- tile windows:
  [`../optimizations/fused_moe_sve/results/amazon_192c_stage_window_tiles_20260810.md`](../optimizations/fused_moe_sve/results/amazon_192c_stage_window_tiles_20260810.md);
- W8A16:
  [`../optimizations/fused_moe_sve/results/amazon_m5_96c_w8a16_plan_v2_20260817.md`](../optimizations/fused_moe_sve/results/amazon_m5_96c_w8a16_plan_v2_20260817.md);
- W8A8:
  [`../optimizations/fused_moe_sve/results/arm_codex_80c_w8a8_i8mm_20260821.md`](../optimizations/fused_moe_sve/results/arm_codex_80c_w8a8_i8mm_20260821.md);
- two-machine framework smoke:
  [`../optimizations/fused_moe_sve/results/arm_codex_aws8_fused_expert_smoke_20260827.md`](../optimizations/fused_moe_sve/results/arm_codex_aws8_fused_expert_smoke_20260827.md).
