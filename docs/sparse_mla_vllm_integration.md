# Sparse MLA vLLM Integration Guide

本文档说明如何把 `fused_cpp` 的 CPU sparse MLA forward 接入 vLLM
DeepSeek V4 CPU prefill attention 路径。

当前目标不是替换整个 DeepSeek V4 attention backend，而是先替换
`cpu_sparse_attn_prefill` 里的 pure torch sparse attention fallback。
decode、indexer、KV cache 写入和 output projection 仍沿用 vLLM 现有 CPU
路径。

## Kernel Scope

公开 Python API:

```python
from fused_cpp.sparse_mla import flash_mla_sparse_fwd
```

签名:

```python
flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int | None = None,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

Supported tensors:

| Tensor | Shape | Dtype | Device | Notes |
| --- | --- | --- | --- | --- |
| `q` | `[s_q, h_q, d_qk]` | `bf16` or `fp32` | CPU | Contiguous is preferred. |
| `kv` | `[s_kv, 1, d_qk]` | same as `q` | CPU | MQA only. `h_kv` must be `1`. |
| `indices` | `[s_q, 1, topk]` | integer | CPU | Negative values are padding. Positive out-of-range values raise. |
| `attn_sink` | `[h_q]` or larger | `fp32` preferred | CPU | Optional. Adds sink denominator mass with zero value. |
| `out` | `[s_q, h_q, d_v]` | same as `q` | CPU | Optional output buffer. Last dimension must be contiguous. |

Return behavior:

- `return_stats=False`, default: returns `out` only.
- `return_stats=True`: returns `(out, max_logits, lse)`.
- DeepSeek V4 CPU prefill should keep `return_stats=False`; current model path
  does not consume `max_logits` or `lse`.

Important semantic details:

- `d_v` defaults to `kv.size(-1)`. For vLLM integration, pass
  `d_v=output.shape[-1]` explicitly so the kernel writes the exact width
  expected by the surrounding DeepSeek V4 CPU path.
- `topk_length` is accepted for API compatibility but the current C++ path
  follows the vLLM CPU fallback behavior and relies on negative padding in
  `indices`. Do not leave garbage positive values beyond `topk_length`.
- The output dtype matches `q.dtype`. The bf16 path accumulates in fp32 and
  stores bf16.
- The kernel is forward-only. There is no backward support.

## Current Execution Model

The C++ implementation lives in:

```text
csrc/sparse_mla.cpp
src/fused_cpp/sparse_mla.py
```

Scheduling:

```text
1. Supported BF16 shapes use one token per task once the number of 8-token
   query blocks exceeds half of the requested workers.
2. Each task treats eight query heads as M. SVE uses scalable 8x2VL BFMMLA
   QK/PV; the non-SVE implementation retains its fixed-width path.
3. On SVE, a fully shared contiguous run, or up to two fixed-start growing
   prefixes, packs K/V once per call. Query-dependent indices use per-token
   fused K/V gather-pack and reuse that data across all query-head groups.
4. Short BF16 chunks build one plan per 8-query block. Heavy plans can split
   into at most eight disjoint KV shards when query blocks leave at least half
   of the worker pool idle.
5. Each shard computes a local online-softmax state `(m, l, O)`. The states are
   merged associatively and `attn_sink` is applied exactly once after the merge.
6. Unsupported BF16 shapes and fp32 retain the indexed query-block fallback;
   fp32 does not use the 2D KV split.
```

The public `flash_mla_sparse_fwd` entrypoint selects the head-major path for
long BF16 queries and retains the guarded 2D scheduler for short BF16 chunks.
These schedules do not alter the Python signature or accepted layouts. Because
KV shards change the online-softmax reduction order, 2D BF16 results are
tolerance-equivalent rather than bitwise-identical to the 1D path.

Dense paths:

- If all queries share one full contiguous run, the head-major fast path packs
  K/V once per call.
- On SVE, up to two fixed-start growing prefixes, such as compressed and
  causal/SWA regions before the window starts sliding, also use call-level
  shared packing.
- Inside the short-query planner, fully shared dense segments within an
  8-query block use packed dense segment code.

MQA cache policy:

- `kv.shape[1] == 1` is required.
- KV is shared by all heads and query blocks.
- The OpenMP worker cap does not use the old MHA formula
  `L3_budget / raw_kv_bytes`, because raw KV is not duplicated per worker in
  MQA. If a future cap is needed, it should be based on per-worker packed
  scratch, not on shared KV bytes.

Threading:

```bash
OMP_NUM_THREADS=64
OMP_DYNAMIC=FALSE
GOMP_CPU_AFFINITY=80-143
taskset -c 80-143 ...
```

Use `fused_cpp._C.get_omp_runtime_info()` to verify that the extension sees the
expected `max_threads` and `num_procs`.

## vLLM Integration Point

For the current DeepSeek V4 CPU branch, the main entry is:

```text
vllm/models/deepseek_v4/cpu.py
```

The relevant functions/classes are:

```text
DeepseekV4CPUSparseMLAImpl.forward_mqa
DeepseekV4CPUSparseMLAImpl._forward_prefill
cpu_sparse_attn_prefill
```

Recommended first integration:

1. Add an optional import near the top of `cpu.py`.
2. Add a guarded fast path inside `cpu_sparse_attn_prefill`.
3. Keep the existing `_cpu_sparse_attention` fallback unchanged.
4. Enable the C++ path behind a feature flag first.
5. Compare output against the torch fallback on the same prefill chunks.

Suggested optional import:

```python
try:
    from fused_cpp.sparse_mla import flash_mla_sparse_fwd as _cpu_flash_mla_sparse_fwd

    _HAS_FUSED_CPP_SPARSE_MLA = True
except (ImportError, AttributeError):
    _cpu_flash_mla_sparse_fwd = None
    _HAS_FUSED_CPP_SPARSE_MLA = False
```

Suggested feature flag:

```python
import os

_USE_FUSED_CPP_SPARSE_MLA = (
    os.environ.get("VLLM_CPU_SPARSE_MLA_FUSED_CPP", "0") == "1"
)
```

Suggested fast path inside `cpu_sparse_attn_prefill`:

```python
def cpu_sparse_attn_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    scale: float,
    head_dim: int,
    attn_sink: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    if (
        _USE_FUSED_CPP_SPARSE_MLA
        and _HAS_FUSED_CPP_SPARSE_MLA
        and q.device.type == "cpu"
        and kv.device.type == "cpu"
        and indices.device.type == "cpu"
        and q.dtype in (torch.bfloat16, torch.float32)
        and q.dtype == kv.dtype
        and kv.ndim == 3
        and kv.shape[1] == 1
        and output.shape[:2] == q.shape[:2]
        and output.shape[-1] <= kv.shape[-1]
    ):
        _cpu_flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            scale,
            d_v=output.shape[-1],
            attn_sink=attn_sink,
            topk_length=topk_length,
            out=output,
            return_stats=False,
        )
        return

    out_chunk = _cpu_sparse_attention(
        q=q,
        kv=kv,
        indices=indices,
        scale=scale,
        attn_sink=attn_sink,
    )
    output.copy_(out_chunk.to(output.dtype))
```

Notes:

- Do not swallow all C++ exceptions in the hot path once the gate is enabled by
  default. During bring-up, it is acceptable to catch once, log a warning, and
  fall back to torch.
- The feature flag should default to off until correctness and performance are
  verified on representative DeepSeek V4 prefill workloads.
- If vLLM later changes the CPU output width from `q.shape[-1]` to
  `v_head_dim`, this integration should keep passing `d_v=output.shape[-1]`.

## DeepSeek V4 Source Shape

For a single prefill chunk, vLLM builds a flattened MQA KV workspace:

```python
kv_view = kv.view(-1, 1, q.shape[-1])
indices_chunk = combined_indices.unsqueeze(1)
```

The combined source is:

```text
[compressed sparse source] + [SWA/dense source]
```

For a V4-like input length 2048 with compress ratio 4:

```text
compressed source length = 512
SWA/dense source length  = 2048
total KV rows            = 2560
```

The benchmark source-pair count used in this repo is:

```python
dense_pairs = 2048 * 2049 // 2
compressed_pairs = sum((i + 1) // 4 for i in range(2048))
total_pairs_per_head = dense_pairs + compressed_pairs
```

This source-pair definition is for FLOP accounting only. The actual kernel
uses `indices` and only treats negative values as padding.

## Build Requirements

Build `fused_cpp`:

```bash
python setup.py build_ext --inplace
```

On restricted local sandboxes, ccache may try to write outside the workspace.
Use:

```bash
CCACHE_DISABLE=1 python setup.py build_ext --inplace
```

Basic validation:

```bash
python -m pytest tests/test_sparse_mla.py -q
```

Remote Arm-codex-internal/Arm-codex flow used during development:

```bash
bash rsync.sh
ssh Arm-codex-internal 'cd /home/zhangxu/code && touch csrc/sparse_mla.cpp && .venv/bin/python setup.py build_ext --inplace'
```

## Correctness Checklist

Run these before enabling the vLLM fast path:

1. `tests/test_sparse_mla.py` in `fused_cpp`.
2. vLLM CPU prefill comparison with `VLLM_CPU_SPARSE_MLA_FUSED_CPP=0` and `1`.
3. Cases with `attn_sink is None` and `attn_sink is not None`.
4. Dense shared indices: every query row attends the same contiguous run.
5. Hybrid V4-like indices: compressed lower triangle plus SWA lower triangle.
6. Indexed-only random or non-contiguous indices.
7. Tail query blocks where `s_q` is not a multiple of 8.
8. Positive out-of-range indices should fail rather than silently mask.
9. Negative padding should produce the same result as the torch fallback.

Suggested numerical tolerance for bf16:

```python
torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
```

## Performance Checklist

Use fixed affinity when comparing versions:

```bash
taskset -c 80-143 env \
  OMP_NUM_THREADS=64 \
  OMP_DYNAMIC=FALSE \
  GOMP_CPU_AFFINITY=80-143 \
  python bench.py
```

Report:

```text
median_ms
min_ms
max_ms
source-pair GFLOP/s
per-core GFLOP/s
efficiency vs Arm-codex-internal/Arm-codex single-core peak
```

For Arm-codex-internal/Arm-codex, use the peak table referenced by `AGENTS.md`.
The fp32 FMLA and bf16 MMLA single-core references are both around 92.7 GFLOP/s
on that machine.

Important current performance caveats:

- Supported BF16 shapes use one token per task and treat eight query heads as
  the QK/PV M dimension once eight-query block count exceeds half of the
  requested workers. Selected MQA K/V rows are packed once per token and reused
  by all head groups.
- The 2D split is retained for shorter prefill chunks with long KV histories.
  It is disabled when query-block count is greater than half of the requested
  worker count, because duplicate packing and partial-state reduction then cost
  more than the extra concurrency saves.
- A split plan currently permits at most eight KV shards. This is sufficient to
  fill 96 workers from 12 heavy query blocks, but is a policy constant rather
  than a calibrated machine model.
- Split shards still process all query heads. Head-group splitting was not
  adopted because it would duplicate or require sharing packed segment state.
- Planning, packing, partial-state allocation, and merge are inside the timed
  forward call. Production chunk-size distributions should be measured before
  changing the underfill threshold or shard cap.

## Rollout Plan

Recommended rollout order:

1. Land the optional import and feature flag in vLLM with default off.
2. Add a unit test that compares `cpu_sparse_attn_prefill` with and without the
   fused path on synthetic dense/shared, indexed, and hybrid indices.
3. Add one DeepSeek V4 prefill integration test using real metadata construction
   if available.
4. Enable on Arm CPU CI or benchmark hosts only.
5. After correctness is stable, flip the default on for AArch64 CPU bf16
   DeepSeek V4 prefill.

Keep fallback conditions explicit:

```text
missing fused_cpp extension
non-CPU tensors
unsupported dtype
q/kv dtype mismatch
h_kv != 1
output width > kv width
```

## Known Limitations

- Forward only; no backward support.
- CPU only.
- MQA only: `kv.shape[1] == 1`.
- No FP8 KV cache input. vLLM CPU must gather/dequantize or otherwise provide
  bf16/fp32 `kv` before calling this kernel.
- `topk_length` is not used to mask positive values. Invalid slots must be
  negative in `indices`.
- Decode is not covered by this kernel.
- BF16 requires `Hq % 8 == 0`, `Dqk % 4 == 0`, and `Dv % 8 == 0` for the
  head-major path. Other shapes use the indexed fallback.
- BF16 short-chunk parallelism uses the guarded 2D KV split described above;
  fp32 remains limited to 8-query task granularity and the per-task all-heads
  loop.

## Files To Touch In vLLM

Expected minimal patch:

```text
vllm/models/deepseek_v4/cpu.py
```

Optional follow-up files:

```text
tests/models/deepseek_v4/...
vllm/envs.py
docs/source/...
```

The first patch should not change the metadata builder, indexer, cache layout,
or decode path. It should only replace the final prefill attention compute when
all shape/dtype gates pass.
