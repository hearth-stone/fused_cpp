# vLLM BF16 Tiled MoE Integration Guide

This document describes how to integrate the `fused_cpp` BF16 tiled fused MoE
CPU kernel into a vLLM-style MoE path.

The intended use case is an AArch64 CPU backend with BF16 matrix instructions.
Weights are packed once when the vLLM MoE layer is initialized, then reused by
decode or prefill forward calls.

## Kernel Scope

The public Python API is in `src/fused_cpp/moe/bf16_tiled.py`:

```python
from fused_cpp.moe import (
    _HAS_BF16_TILED_FUSED_MOE,
    fused_moe_bf16_tiled,
    prepare_fused_moe_bf16_tiled_weights,
)
```

Supported tensors:

| Tensor | Shape | Dtype | Device | Notes |
| --- | --- | --- | --- | --- |
| `hidden_states` | `[num_tokens, hidden_size]` | `torch.bfloat16` | CPU | Contiguous input activations |
| `w13_weight` | `[num_experts, 2 * ffn_hidden_size, hidden_size]` | `torch.bfloat16` | CPU | Gate/up projection, vLLM layout |
| `w2_weight` | `[num_experts, hidden_size, ffn_hidden_size]` | `torch.bfloat16` | CPU | Down projection, vLLM layout |
| `topk_weights` | `[num_tokens, top_k]` | floating | CPU | Routing weights |
| `topk_ids` | `[num_tokens, top_k]` | integer | CPU | Local expert ids |
| `w13_bias` | `[num_experts, 2 * ffn_hidden_size]` | `float32` or `bfloat16` | CPU | Optional |
| `w2_bias` | `[num_experts, hidden_size]` | `float32` or `bfloat16` | CPU | Optional |

The output is `[num_tokens, hidden_size]` in `torch.bfloat16`.

Supported activations:

```text
silu
gelu
swigluoai
```

`skip_weighted=True` is only valid when `top_k == 1`.

## Build Requirements

The extension is built by `setup.py`. On AArch64, it uses `refs/i8gemm/lib` for
the low-level BF16 GEMM kernel and packing headers.

Build:

```bash
python setup.py build_ext --inplace
```

Basic validation:

```bash
python -m pytest -q tests/test_fused_moe_bf16_tiled.py -m 'not slow'
```

The backend is available only when:

```python
_HAS_BF16_TILED_FUSED_MOE is True
```

vLLM should keep a fallback path for non-AArch64 hosts, missing extension builds,
unsupported dtype/device combinations, and unsupported activations.

## Weight Preparation

Pack weights once during MoE layer initialization or after weights are loaded.
Do not pack inside every forward call.

```python
packed = prepare_fused_moe_bf16_tiled_weights(
    w13_weight.contiguous(),
    w2_weight.contiguous(),
)
```

`packed` is a `PreparedBF16TiledFusedMoEWeights` object containing packed
`w13` and `w2` tensors plus their original `K` and `N` dimensions.

Large expert sets can pack weights in parallel:

```bash
FUSED_CPP_MOE_PREPACK_THREADS=16
```

Packing is split by expert, so each thread writes disjoint slices of the packed
`w13` and `w2` tensors. The default is `1` to keep initialization behavior
predictable unless explicitly enabled.

Recommended vLLM layer state:

```python
class CpuBF16TiledMoEState:
    def __init__(self, w13_weight, w2_weight):
        self.packed_weights = prepare_fused_moe_bf16_tiled_weights(
            w13_weight,
            w2_weight,
        )
```

Important:

- Expert ids in `topk_ids` must match the packed weight expert index.
- Pass `global_num_experts=-1` for normal local packed weights.
- Do not pass the model-wide expert count if it is larger than the packed local
  expert count. The C++ path checks that the requested expert count does not
  exceed prepared weights.

## Forward Call

Minimal call:

```python
out = fused_moe_bf16_tiled(
    hidden_states,
    packed_weights,
    topk_weights,
    topk_ids,
    num_threads=num_threads,
    activation=activation,
)
```

With optional bias and output buffer:

```python
out = fused_moe_bf16_tiled(
    hidden_states,
    packed_weights,
    topk_weights,
    topk_ids,
    w13_bias=w13_bias,
    w2_bias=w2_bias,
    num_threads=num_threads,
    activation="silu",
    global_num_experts=-1,
    skip_weighted=False,
    out=out_buffer,
)
```

`out_buffer` must be a contiguous CPU BF16 tensor with the same shape as
`hidden_states` and `requires_grad=False`. The native normal, scheduled, and
async entrypoints write the final BF16 result directly into this storage,
increment its version counter, and the Python wrapper returns the same tensor
object; there is no temporary final-output tensor or trailing `copy_`. The
buffer must not overlap the input, packed weights, `topk_weights`, or `topk_ids`.
Internal per-route storage used for weighted TopK merge remains operator-owned.

Recommended vLLM adapter shape:

```python
def cpu_bf16_tiled_moe_forward(
    hidden_states,
    packed_weights,
    topk_weights,
    topk_ids,
    *,
    activation,
    num_threads,
    w13_bias=None,
    w2_bias=None,
):
    if not _HAS_BF16_TILED_FUSED_MOE:
        return fallback_moe(...)
    if hidden_states.dtype != torch.bfloat16 or hidden_states.device.type != "cpu":
        return fallback_moe(...)

    return fused_moe_bf16_tiled(
        hidden_states.contiguous(),
        packed_weights,
        topk_weights.contiguous(),
        topk_ids.contiguous(),
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        num_threads=num_threads,
        activation=activation,
    )
```

## Threading Modes

### Default Mode

By default, `num_threads` creates that many `std::thread` workers.

Scheduling:

```text
expert-affinity greedy assignment
one worker processes its assigned expert/tile ranges
```

This mode is simple and requires no environment variables.

### Hierarchical Dynamic Expert N-Split Mode

Enable with:

```bash
FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT=1
```

This mode is intended for high core counts and large BF16 MoE shapes.

Scheduling:

```text
1. Count routed rows per expert.
2. Sort experts by routed rows descending.
3. Split total threads into fixed groups.
4. Each group dynamically takes the next expert from a shared queue.
5. Threads inside the group compute that expert together by splitting GEMM N.
6. Inside one expert, rows are still processed in 16-row tiles for the
   low-level GEMM microkernel.
```

Default group topology:

```text
partitions = len(core_bases), default 2
groups_per_partition = FUSED_CPP_MOE_N_SPLIT_GROUPS_PER_PARTITION, default 4
groups = partitions * groups_per_partition
group_size = num_threads / groups
```

Default core bases use absolute CPU IDs:

```bash
FUSED_CPP_MOE_N_SPLIT_CORE_BASES=0,40
```

For multi-rank launches where each rank already has a different CPU affinity,
you can generate core bases relative to the first CPU in the current affinity:

```bash
FUSED_CPP_MOE_N_SPLIT_CORE_SKIP=40
```

With `num_threads=64`, `groups_per_partition=4`, and affinity starting at CPU
80, the generated core bases are `80,120`, so the worker cores are `80-111`
and `120-151`. If `FUSED_CPP_MOE_N_SPLIT_CORE_BASES` is set, it takes
precedence and is interpreted as absolute CPU IDs.

Examples:

```text
num_threads=64 -> groups=8, group_size=8
cores: 0-31 and 40-71

num_threads=80 -> groups=8, group_size=10
cores: 0-39 and 40-79
```

The thread count must be divisible by:

```text
len(core_bases) * groups_per_partition
```

Recommended launch on the Arm test machine:

```bash
taskset -c 0-79 env \
  OMP_PROC_BIND=FALSE \
  FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT=1 \
  FUSED_CPP_MOE_N_SPLIT_CORE_SKIP=40 \
  python tests/bench_fused_moe_bf16_tiled.py \
    --experts 256 --threads 80
```

Notes:

- The kernel uses `std::thread`, not OpenMP, for MoE worker threads.
- `OMP_PROC_BIND=FALSE` is recommended when running in an environment where
  other OpenMP users may be present.
- The hierarchical mode uses `pthread_setaffinity_np` on Linux and restores the
  caller thread's original affinity when the call returns.
- On non-Linux platforms, core binding is ignored but the scheduling mode still
  functions.

## Runtime Configuration

| Env var | Default | Meaning |
| --- | --- | --- |
| `FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT` | `0` | Enable dynamic expert group mode |
| `FUSED_CPP_MOE_N_SPLIT_CORE_BASES` | `0,40` | Partition core base list |
| `FUSED_CPP_MOE_N_SPLIT_CORE_SKIP` | unset | Generate two partition core bases from current affinity first CPU and this skip; ignored when `FUSED_CPP_MOE_N_SPLIT_CORE_BASES` is set |
| `FUSED_CPP_MOE_N_SPLIT_GROUPS_PER_PARTITION` | `4` | Number of groups in each partition |
| `FUSED_CPP_MOE_SCHEDULE_DEBUG` | `0` | `1` prints summary, `2` prints all workers/groups |
| `FUSED_CPP_MOE_PREPACK_THREADS` | `1` | Number of threads used when packing expert weights |

Debug example:

```bash
taskset -c 0-79 env \
  FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT=1 \
  FUSED_CPP_MOE_N_SPLIT_CORE_SKIP=40 \
  FUSED_CPP_MOE_SCHEDULE_DEBUG=2 \
  python tests/bench_fused_moe_bf16_tiled.py \
    --experts 256 --threads 80 --warmup 0 --runs 1
```

Expected debug header:

```text
strategy=hierarchical_n_split_dynamic_expert
groups=8
group_size=10
core_bases=[0,40]
```

Per-group debug lines include:

```text
rows=<routed rows processed by this group>
tiles=<16-row micro tiles processed by this group>
experts=[expert_id:rows,...]
```

## vLLM Integration Points

Recommended integration flow:

1. Add a CPU BF16 tiled MoE implementation option in the vLLM MoE dispatch layer.
2. During layer weight loading, identify BF16 CPU expert weights in vLLM layout:
   `w13=[E, 2F, H]`, `w2=[E, H, F]`.
3. Call `prepare_fused_moe_bf16_tiled_weights` once and store the packed object
   on the layer or implementation object.
4. In forward, after routing has produced `topk_weights` and `topk_ids`, call
   `fused_moe_bf16_tiled`.
5. Choose `num_threads` from vLLM CPU worker configuration, a model config knob,
   or an environment override.
6. Keep a fallback path when the backend is unavailable or inputs do not satisfy
   the shape/dtype/device constraints.

Pseudo-code:

```python
class VllmCpuBF16TiledMoE:
    def __init__(self, w13_weight, w2_weight, activation, num_threads):
        self.activation = activation
        self.num_threads = num_threads
        self.packed_weights = prepare_fused_moe_bf16_tiled_weights(
            w13_weight,
            w2_weight,
        )

    def forward(self, hidden_states, topk_weights, topk_ids):
        return fused_moe_bf16_tiled(
            hidden_states,
            self.packed_weights,
            topk_weights,
            topk_ids,
            num_threads=self.num_threads,
            activation=self.activation,
        )
```

## Validation Checklist

Correctness:

```bash
python -m pytest -q tests/test_fused_moe_bf16_tiled.py -m 'not slow'
```

Default benchmark:

```bash
python tests/bench_fused_moe_bf16_tiled.py \
  --experts 256 --threads 64 --warmup 2 --runs 5
```

Hierarchical benchmark:

```bash
taskset -c 0-31,40-71 env \
  OMP_PROC_BIND=FALSE \
  FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT=1 \
  FUSED_CPP_MOE_N_SPLIT_CORE_SKIP=40 \
  python tests/bench_fused_moe_bf16_tiled.py \
    --experts 256 --threads 64 --warmup 2 --runs 5
```

Metrics to track:

- `median_ms`
- `median_gflops`
- effective efficiency:

```text
efficiency = median_gflops / (num_threads * single_core_peak_gflops)
```

On the Arm test machine used during development, `single_core_peak_gflops` was
measured as approximately `92.704 GFLOP/s/core`.

The benchmark defaults model the DeepSeek-V4-Flash TP=4 per-rank MoE shape:

```text
tokens=2048
experts=256
top_k=6
H=4096
F_per_rank=2048 / 4 = 512
w13=[256, 1024, 4096]
w2=[256, 4096, 512]
```

Older development reference results for shape
`tokens=2048, experts=256, top_k=8, H=7168, F=512`:

| Mode | Threads | Median ms | Median GFLOP/s | Efficiency |
| --- | ---: | ---: | ---: | ---: |
| Default expert/tile workers | 64 | 199.065 | 1812.356 | 30.5% |
| Hierarchical dynamic expert N-split | 64 | 138.526 | 2604.398 | 43.9% |
| Hierarchical dynamic expert N-split | 80 | 126.069 | 2861.737 | 38.6% |

Use these numbers as regression references, not as fixed guarantees. They are
host, affinity, and routing-distribution dependent.

## Common Failure Modes

- Repacking weights every forward call. Packing can take seconds for large
  expert counts and must be outside the timed path.
- Passing global model expert ids when only local expert weights are packed.
- Running with PyTorch intra-op threads greater than one and oversubscribing the
  CPU. Prefer `torch.set_num_threads(1)` around this benchmark path.
- Setting `OMP_PROC_BIND=close` in a process that also uses `std::thread`; this
  can accidentally pin worker threads poorly in some environments.
- Enabling hierarchical mode with a `num_threads` value that is not divisible by
  `len(core_bases) * groups_per_partition`.
- Using absolute `FUSED_CPP_MOE_N_SPLIT_CORE_BASES` in multi-process runs
  without adjusting it for each rank. Prefer `FUSED_CPP_MOE_N_SPLIT_CORE_SKIP`
  when every rank already has a distinct CPU affinity.
