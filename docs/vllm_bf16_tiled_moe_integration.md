# vLLM CPU BF16 MoE Integration Guide

This document describes the supported Python integration for the `fused_cpp`
CPU BF16 fused MoE operator. The recommended production path is the ARM SVE
fused-SiLU backend with an explicitly calibrated Plan V2 runtime.

The integration has three lifecycle phases:

1. calibrate the machine or load an existing calibration profile;
2. pack each MoE layer's weights once;
3. call `fused_moe_tiled()` with BF16 or explicitly prepared W8A16 weights.

Calibration never runs at package import time or on the first MoE request.

## Supported Scope

The public APIs used by an integration are exported from both `fused_cpp.moe`
and `fused_cpp`:

```python
from fused_cpp.moe import (
    MoePlannerRuntime,
    available_fused_moe_bf16_tiled_backends,
    enable_moe_planner_quick,
    fused_moe_bf16_tiled,
    fused_moe_bf16_tiled_with_shared,
    fused_moe_tiled,
    get_default_moe_planner_runtime,
    prepare_fused_moe_bf16_tiled_weights,
    prepare_fused_moe_w8a16_tiled_weights,
    prepare_routed_shared_moe_bf16_tiled_weights,
    set_default_moe_planner_runtime,
)
```

The calibrated production runtime currently supports:

- Linux AArch64 with the `arm_sve_bf16` backend;
- BF16 weights packed with `fuse_silu=True`;
- `activation="silu"`;
- standalone and tensor-parallel execution;
- one process-local runtime bound to a fixed CPU set and expert shape.

Expert-parallel execution is not yet owned by `MoePlannerRuntime`. Calls that
do not satisfy the runtime domain use the existing native dispatcher instead.
The normal operator remains usable on other advertised backends without the
calibrated runtime.

Per-channel W8A16 is an optional experimental ARM SVE implementation. It is
selected only by passing `PreparedW8A16TiledFusedMoEWeights` to
`fused_moe_tiled()`. It requires a compatible installed planner, supports no
W13/W2 bias, and has no native unplanned fallback. BF16 remains the default and
supported fallback for integrations that do not explicitly prepare W8A16.

Check backend availability before creating the optimized layer:

```python
backends = available_fused_moe_bf16_tiled_backends()
if "arm_sve_bf16" not in backends:
    # Keep the framework's existing CPU MoE implementation.
    ...
```

## Tensor Contract

Let `T` be the token count, `K` the router TopK, `E` the number of locally
packed experts, `H` the hidden size, and `F` the rank-local intermediate size.

| Value | Shape | Dtype | Device | Notes |
| --- | --- | --- | --- | --- |
| `input` | `[T, H]` | `torch.bfloat16` | CPU | Hidden states |
| `w13_weight` | `[E, 2 * F, H]` | `torch.bfloat16` | CPU | vLLM gate/up layout |
| `w2_weight` | `[E, H, F]` | `torch.bfloat16` | CPU | vLLM down layout |
| `topk_weights` | `[T, K]` | floating | CPU | Router weights |
| `topk_ids` | `[T, K]` | integer | CPU | Indices into the packed local expert array |
| result | `[T, H]` | `torch.bfloat16` | CPU | Weighted routed output |

Optional biases use `[E, 2 * F]` for W13 and `[E, H]` for W2. A supplied
`out=` tensor must be contiguous CPU BF16 storage with the same shape as
`input`, must not require gradients, and must not alias any input or packed
weight tensor.

`skip_weighted=True` is valid only for `K == 1`. The default weighted path
supports larger TopK values and performs the route accumulation in FP32 after
loading the BF16 route results.

## Build And Startup Check

Initialize the SVE JIT dependency and build the extension:

```bash
git submodule update --init --recursive 3rdparty/xbyak_aarch64
python setup.py build_ext --inplace
```

At service startup, fail over to the framework implementation if the required
backend is unavailable. Do not attempt to recover an SVE vector-length mismatch
inside the request path: generated kernels and packed weights are tied to the
build/runtime SVE vector-length contract.

## Quick Calibration

Run quick calibration once for each machine type, CPU placement, and planner
width domain. It is synchronous and should run during deployment setup or
service initialization, before accepting requests.

```python
from fused_cpp.moe import enable_moe_planner_quick

runtime = enable_moe_planner_quick(
    cpu_ids=tuple(range(96)),
    output="/var/cache/fused_cpp/moe-numa0.json",
    hidden_size=4096,
    intermediate_size=512,
    global_experts=256,
    local_experts=256,
    mode="tp",
    degree=4,
    concurrent_ranks=1,
    report=print,
)
```

This call measures the machine, writes the calibration profile, creates a
shape-bound `MoePlannerRuntime`, and installs it as the process default only
after calibration and runtime construction both succeed.

Important calibration rules:

- `cpu_ids` is an ordered list and becomes part of the profile contract.
- The inference call must use exactly `len(cpu_ids)` threads.
- Reserve the same CPUs for the rank at runtime; Plan V2 places workers on
  those CPU IDs.
- `output` is not overwritten unless `overwrite=True` is explicitly passed.
- `supported_widths` may restrict planner-legal team widths; omit it to use
  the topology-derived quick defaults.
- `degree` is the TP degree for `mode="tp"`; it is not the worker count.
- `concurrent_ranks` describes ranks concurrently sharing the calibrated
  machine resources.

Do not run quick calibration independently in several concurrent ranks on the
same CPU or memory domain. Calibrate each placement without competing service
load, then persist and reuse the result.

## Loading A Saved Calibration

Normal process restarts should load the saved profile rather than recalibrate:

```python
from fused_cpp.moe import MoePlannerRuntime, set_default_moe_planner_runtime

runtime = MoePlannerRuntime(
    "/var/cache/fused_cpp/moe-numa0.json",
    hidden_size=4096,
    intermediate_size=512,
    global_experts=256,
    local_experts=256,
    mode="tp",
    degree=4,
    concurrent_ranks=1,
    cpu_ids=tuple(range(96)),
    cost_cache_dir="/var/cache/fused_cpp/moe-costs",
)
previous_runtime = set_default_moe_planner_runtime(runtime)
```

The registry is process-global and thread-safe. Save `previous_runtime` when a
test or temporary component needs to restore the prior process state:

```python
try:
    set_default_moe_planner_runtime(runtime)
    ...
finally:
    set_default_moe_planner_runtime(previous_runtime)
```

Passing `None` disables calibrated planning and restores the established native
dispatcher:

```python
set_default_moe_planner_runtime(None)
```

The runtime separately caches analytical `T_iso(M,T)` values on disk. Its
default directory is `~/.fused_cpp/cache/moe_costs`; `cost_cache_dir=None`
disables disk caching. Exact machine calibration, model dimensions, TP mode,
backend geometry, analytical model version, and supported widths are part of
the cache identity. Matching values are loaded at runtime construction;
missing points are computed normally and atomically written back. Corrupt,
stale, or unwritable files are ignored rather than failing inference. Final
plans are not persisted because they still depend on the current routing input.

## Weight Preparation

Pack weights once after model weights are loaded, not in `forward()`:

```python
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights

packed_weights = prepare_fused_moe_bf16_tiled_weights(
    w13_weight.contiguous(),
    w2_weight.contiguous(),
    fuse_silu=True,
    backend="arm_sve_bf16",
)
```

The returned `PreparedBF16TiledFusedMoEWeights` is reusable across requests.
It records the selected backend, packed dimensions, backend N tile, and whether
the fused SiLU layout was used.

To opt one layer into experimental per-channel W8A16, prepare the same BF16
source weights with the W8A16 packer instead:

```python
from fused_cpp.moe import prepare_fused_moe_w8a16_tiled_weights

packed_weights = prepare_fused_moe_w8a16_tiled_weights(
    w13_weight.contiguous(),
    w2_weight.contiguous(),
)
```

This is the implementation selection point. The planner does not quantize or
replace BF16 weights at request time.

Packing considerations:

- `FUSED_CPP_MOE_PREPACK_THREADS` parallelizes packing across experts.
- Packed weights are ISA- and SVE-vector-length-specific; repack on a machine
  with a different contract.
- Expert IDs in `topk_ids` must index this packed local expert array.
- In TP, each rank normally packs all experts with rank-local `F`.
- In EP, keep using the framework/native fallback until the calibrated runtime
  explicitly supports `local_experts != global_experts`.

Page placement is controlled by the repository-wide page policy documented in
`docs/production_environment.yaml`. Configure it before packing because page
placement applies to the reusable packed tensors.

## Forward Integration

Once a compatible runtime is installed, the type-dispatched operator is the
only request-path API needed:

```python
out = fused_moe_tiled(
    hidden_states,
    packed_weights,
    topk_weights,
    topk_ids,
    num_threads=96,
    activation="silu",
    global_num_experts=-1,
    skip_weighted=False,
    out=output_buffer,
)
```

For W8A16, `cache_dequant=False` selects register dequantization. Set
`cache_dequant=True` only when the installed Plan V2 policy supplies suitable
stage windows; zero still means a full owner stripe and is not automatically
an L2-sized window. W8A16 rejects bias arguments instead of silently ignoring
them.

The wrapper counts routes, obtains or reuses a cached plan, and lowers a
compatible call through `fused_moe_bf16_tiled_async_plan()`. Native execution
runs outside the planner cache lock. Callers should not invoke the Plan V2 API
directly unless they own plan construction and validation.

`global_num_experts=-1` is the recommended value when `topk_ids` already use
the packed local expert index space. For the current TP runtime, an explicit
value must equal the runtime's `global_experts`.

A minimal vLLM-style layer adapter is:

```python
class CpuSVEFusedMoE:
    def __init__(self, w13_weight, w2_weight, *, num_threads):
        self.num_threads = num_threads
        self.packed_weights = prepare_fused_moe_bf16_tiled_weights(
            w13_weight,
            w2_weight,
            fuse_silu=True,
            backend="arm_sve_bf16",
        )

    def forward(self, hidden_states, topk_weights, topk_ids, *, out=None):
        return fused_moe_bf16_tiled(
            hidden_states,
            self.packed_weights,
            topk_weights,
            topk_ids,
            num_threads=self.num_threads,
            activation="silu",
            out=out,
        )
```

The framework must retain its existing fallback for unavailable backends,
unsupported devices or dtypes, non-SiLU activations, EP layouts, and extension
load failures.

## Combined Routed And Shared Expert

For standalone or TP models with one shared expert and the same rank-local
`F` as each routed expert, pack both arrays without concatenating the full raw
weights:

```python
packed_with_shared = prepare_routed_shared_moe_bf16_tiled_weights(
    routed_w13,
    routed_w2,
    shared_w13,
    shared_w2,
)
```

Install the runtime with the routed expert count and `shared_experts=1`:

```python
runtime = MoePlannerRuntime(
    calibration_path,
    hidden_size=H,
    intermediate_size=F_per_rank,
    global_experts=routed_experts,
    local_experts=routed_experts,
    mode="tp",
    degree=tp_degree,
    cpu_ids=rank_cpu_ids,
    shared_experts=1,
)
set_default_moe_planner_runtime(runtime)
```

The request path then uses one combined call:

```python
output = fused_moe_bf16_tiled_with_shared(
    hidden_states,
    packed_with_shared,
    topk_weights,
    topk_ids,
    num_threads=len(rank_cpu_ids),
    routed_scaling_factor=2.5,
    swiglu_limit=10.0,
)
```

Internally, the shared expert is one all-token synthetic expert. The quick
planner searches a bounded `shared_width + routed_width` family and allows the
shared lane to process routed experts after the shared task completes. Route
weights and the shared contribution are accumulated in FP32 before the single
BF16 output store. DeepSeek-V4 clamped SwiGLU is enabled only by explicitly
passing `swiglu_limit=10.0`; it requires the SVE JIT path. EP, biases, multiple
shared experts, mismatched shared `F`, static asm, and other clamp limits remain
unsupported by this initial combined API.

## Runtime Ownership And Fallback

The runtime owns a call only when all of the following are true:

| Condition | Required value |
| --- | --- |
| Planner mode | `standalone` or `tp` |
| Expert ownership | `local_experts == global_experts` |
| Threads | exactly the calibrated core count |
| Activation | `silu` |
| Packed layout | `fuse_silu=True` |
| Backend | `arm_sve_bf16` |
| Backend N tile | equal to the calibrated policy |
| Shape | runtime `H`, `F`, and `E` match packed weights |
| Global expert count | omitted with `-1`, or equal to runtime value |

If any condition fails, `fused_moe_bf16_tiled()` does not raise a
planner-compatibility error. It bypasses Plan V2 and calls the existing native
dispatcher. This preserves the pre-runtime behavior, but it also means an
integration must inspect diagnostics when it needs to prove that calibrated
scheduling was used.

## Diagnostics

After a compatible invocation, inspect the latest planning decision:

```python
runtime = get_default_moe_planner_runtime()
if runtime is not None:
    diagnostics = runtime.last_plan
    print(diagnostics)
```

`last_plan` is a snapshot intended for diagnostics and benchmarking. It
contains the latest planner decision and cache information; callers should not
treat its internal keys as a serialized public plan schema.

The `cost_disk_cache` diagnostic reports `status`, `path`, `loaded_entries`,
`total_entries`, and any non-fatal `error`. Expected states are `miss`, `hit`,
`stored`, `disabled`, and `error`.

To distinguish Plan V2 from fallback in an integration test, use a known
compatible call and assert that `last_plan` was updated. Also exercise one
deliberately incompatible call and compare it with the established native
dispatcher.

## Recommended Initialization Order

Use this order in each CPU rank:

1. Set rank affinity and page policy.
2. Import `fused_cpp` and verify `arm_sve_bf16` availability.
3. Load a saved calibration into `MoePlannerRuntime`, or explicitly run quick
   calibration during machine provisioning.
4. Install the runtime with `set_default_moe_planner_runtime()`.
5. Load model weights and prepack each MoE layer with `fuse_silu=True`.
6. Warm up representative route shapes outside the measured/request path so
   the planner cache contains common distributions.
7. Start serving and call only `fused_moe_bf16_tiled()` from layer forwards.

The calibration is machine/placement-specific, while packed weights are
model-layer/ISA-specific. Their lifecycles should therefore remain separate.

## Validation

Run focused API and dispatch tests after integration changes:

```bash
python -m pytest -q \
  tests/test_moe_planner_runtime.py \
  tests/test_moe_backend_dispatch.py
```

Run numerical coverage on the target ARM machine:

```bash
python -m pytest -q tests/test_fused_moe_bf16_tiled.py -m 'not slow'
```

For a deployment smoke test, verify all of the following:

- the SVE backend is advertised;
- quick calibration or saved-profile loading succeeds on the intended CPU IDs;
- packed weights report `backend_name == "arm_sve_bf16"` and
  `fused_silu is True`;
- a compatible call updates `runtime.last_plan`;
- planned output matches the framework reference within the established BF16
  tolerance;
- a deliberately incompatible activation or thread count follows the fallback
  path without changing numerical semantics.

Do not report a performance improvement without recording the machine, NUMA
placement, CPU IDs, route distribution, `T/K/E/H/F`, TP degree, thread count,
page policy, warmup/runs, statistic, baseline, and absolute measured time.

## Source Map

- Python operator and weight packing: `src/fused_cpp/moe/bf16_tiled.py`
- Calibrated process runtime: `src/fused_cpp/moe/planner_runtime.py`
- Public Plan V2 types: `src/fused_cpp/moe/plan.py`
- Quick calibration: `cpu_moe_schedule_optimization/cost_model/quick_calibration.py`
- Analytical model: `cpu_moe_schedule_optimization/cost_model/analytic_model.py`
- Production planner: `cpu_moe_schedule_optimization/planners/planned_moe.py`
- Plan schema: `cpu_moe_schedule_optimization/planners/plan_schema.md`
- Public contracts: `docs/public_contracts.md`
- Production environment variables: `docs/production_environment.yaml`
