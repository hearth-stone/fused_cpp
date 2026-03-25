# fused-mla-cpp

CPU Fused Multi-head Latent Attention — a standalone, pip-installable pure
PyTorch implementation of the fused MLA forward pass (projection, RoPE,
KV-cache write, attention, output projection).

## Installation

```bash
pip install -e fused_mla_cpp/
```

## Quick Start

```python
from fused_mla_cpp import CPUFusedMLAImpl

impl = CPUFusedMLAImpl(
    num_heads=16,
    head_size=128,
    scale=0.1,
    num_kv_heads=1,
    kv_cache_dtype="auto",
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    qk_head_dim=192,
    v_head_dim=128,
    kv_b_proj=kv_b_proj_layer,
)
```

## Wrapper Interface Contract

`CPUFusedMLAImpl.forward_fused` accepts a `wrapper` object that must conform
to the protocol described below. The wrapper is duck-typed — any object
exposing the required attributes and sub-modules will work.

### Integer Attributes

The wrapper must expose the following integer attributes:

| Attribute | Description |
| --- | --- |
| `q_lora_rank` | Rank of the low-rank Q projection (`int` or `None`) |
| `kv_lora_rank` | Rank of the compressed KV representation |
| `qk_rope_head_dim` | Dimension of the RoPE portion of Q/K |
| `qk_nope_head_dim` | Dimension of the non-RoPE portion of Q/K |
| `qk_head_dim` | Total Q/K head dimension (`qk_nope_head_dim + qk_rope_head_dim`) |
| `num_heads` | Number of attention heads |
| `v_head_dim` | Dimension of each value head |

### Sub-Modules (when `q_lora_rank is not None`)

When the model uses low-rank Q projection, the wrapper must provide:

| Sub-Module | Required Attributes | Description |
| --- | --- | --- |
| `fused_qkv_a_proj` | `.weight` | Fused Q/KV down-projection layer |
| `q_a_layernorm` | `.weight`, `.variance_epsilon` | RMSNorm applied to the Q compressed representation |
| `q_b_proj` | `.weight` | Q up-projection layer |

### Sub-Modules (when `q_lora_rank is None`)

When the model does not use low-rank Q projection, the wrapper must provide:

| Sub-Module | Required Attributes | Description |
| --- | --- | --- |
| `q_proj` | `.weight` | Direct Q projection layer |
| `kv_a_proj_with_mqa` | `.weight` | KV down-projection with MQA |

### Always-Required Sub-Modules

Regardless of the `q_lora_rank` path, the wrapper must always provide:

| Sub-Module | Required Attributes | Description |
| --- | --- | --- |
| `kv_a_layernorm` | `.weight`, `.variance_epsilon` | RMSNorm applied to the compressed KV representation |
| `rotary_emb` | `.cos_sin_cache`, optional `.is_neox_style` | Rotary position embedding; `cos_sin_cache` is a `[max_positions, rope_dim]` tensor. `is_neox_style` defaults to `True` if absent. |
| `o_proj` | `.weight` | Output projection layer |

### `attn_metadata` Interface

The `attn_metadata` object passed to `forward_fused` must expose:

| Attribute | Type | Description |
| --- | --- | --- |
| `slot_mapping` | `Tensor` | Slot indices for KV cache writes |
| `num_decode_tokens` | `int` | Number of decode tokens in the batch |
| `num_decodes` | `int` | Number of decode sequences |
| `num_prefills` | `int` | Number of prefill sequences |
| `prefill` | object | Sub-object with `query_start_loc`, `max_query_len`, `block_table`, and optional `chunked_context` |
| `decode` | object | Sub-object with `block_table`, `seq_lens` |

## Debug Environment Variables

| Variable | Effect |
| --- | --- |
| `FUSED_MLA_USE_ORIG_RMSNORM=1` | Delegate RMSNorm to the wrapper's layernorm instead of the internal implementation |
| `FUSED_MLA_USE_ORIG_ROPE=1` | Delegate RoPE to the wrapper's `rotary_emb` instead of the internal implementation |

## License

See the project root for license details.
