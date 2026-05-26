# CPUFusedMLAImpl API 参数说明

## 1. `__init__` 构造函数参数

### 1.1 基础注意力参数

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `num_heads` | `int` | 注意力头数。当 TP（张量并行）启用时，这里是**本 rank 拥有的本地头数**（即 `num_heads // tp_size`），由 vllm 的 `ColumnParallelLinear` 在权重加载时自动分片。 |
| `head_size` | `int` | 每个注意力头的维度大小，通常等于 `qk_head_dim`。用于与 vllm 注意力后端接口保持一致。 |
| `scale` | `float` | 注意力分数的缩放因子，通常为 `1 / sqrt(qk_head_dim)`。在计算 `attn_scores = Q @ K^T * scale` 时使用。 |
| `num_kv_heads` | `int` | KV 头数。MLA 架构中通常为 1（Multi-Query Attention），因为 KV 使用低秩压缩表示，所有注意力头共享同一组压缩 KV。 |
| `kv_cache_dtype` | `str` | KV cache 的数据类型字符串，如 `"auto"`、`"fp16"`、`"bf16"` 等。`"auto"` 表示与模型激活值使用相同的 dtype。 |

### 1.2 MLA 专用参数

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `q_lora_rank` | `int \| None` | Q 投影的低秩维度。**DeepSeek V3/R1** 使用低秩 Q 投影时不为 `None`（如 1536）；**DeepSeek V2** 不使用低秩 Q 投影时为 `None`。此参数决定了 `forward_fused` 中 Q 投影的计算路径。 |
| `kv_lora_rank` | `int` | KV 压缩表示的低秩维度（如 512）。`kv_c`（压缩后的 KV）的最后一维大小即为此值。KV cache 中每个 token 存储的向量维度为 `kv_lora_rank + qk_rope_head_dim`。 |
| `qk_nope_head_dim` | `int` | Q/K 中**不参与 RoPE** 的维度（如 128）。这部分通过 absorption trick 与 `W_UK` 矩阵相乘来避免显式展开 K。 |
| `qk_rope_head_dim` | `int` | Q/K 中**参与 RoPE** 的维度（如 64）。这部分需要单独提取并应用旋转位置编码。 |
| `qk_head_dim` | `int` | Q/K 的完整头维度，等于 `qk_nope_head_dim + qk_rope_head_dim`（如 192）。 |
| `v_head_dim` | `int` | Value 头的维度（如 128）。最终注意力输出的每个头维度为此值。 |
| `kv_b_proj` | `torch.nn.Module` | KV 上投影层（`kv_b_proj`），将压缩的 `kv_c`（维度 `kv_lora_rank`）投影回完整的 K_nope 和 V。权重 shape 为 `[num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]`。构造时会提取其 `weight` 和 `bias`，并在 `process_weights_after_loading` 中用于构建吸收矩阵 `W_UK_T` 和 `W_UV`。 |

### 1.3 张量并行（TP）参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `tp_size` | `int` | `1` | 张量并行的总 rank 数。`tp_size=1` 表示单卡推理，无额外开销。`tp_size > 1` 时，`o_proj` 输出后会执行 all-reduce 操作。 |
| `tp_rank` | `int` | `0` | 当前 rank 的编号（0-indexed）。用于 C++ 扩展的 `tp_all_reduce` 调用。 |
| `reduce_fn` | `Optional[Callable[[torch.Tensor], torch.Tensor]]` | `None` | Python 级别的 reduce 回调函数，**仅用于单元测试和开发**（如 mock reduce）。生产环境中使用 C++ 扩展的 `_C.tp_all_reduce` 或 `torch.distributed.all_reduce`。 |

### 1.4 其他

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `**kwargs` | `Any` | 额外的关键字参数，被静默忽略。用于兼容 vllm 注意力后端接口传入的其他参数。 |

---

## 2. `forward_fused` 前向计算参数

```python
def forward_fused(
    self,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    wrapper: Any,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
) -> torch.Tensor:
```

### 2.1 参数详解

| 参数 | 类型 | Shape | 说明 |
| --- | --- | --- | --- |
| `hidden_states` | `torch.Tensor` | `[num_tokens, hidden_size]` | 输入隐藏状态。`num_tokens` 是当前 batch 中所有 token 的总数（decode + prefill）。`hidden_size` 是模型的隐藏层维度。 |
| `positions` | `torch.Tensor` | `[num_tokens]` | 每个 token 的位置索引（`int64`），用于 RoPE 旋转位置编码。从 `cos_sin_cache` 中按位置索引取出对应的 cos/sin 值。 |
| `wrapper` | `Any`（实际为 `MultiHeadLatentAttentionWrapper`） | — | MLA 层的包装对象，提供所有子模块和配置属性（duck-typed 协议）。详见下方 §2.2。 |
| `kv_cache` | `torch.Tensor` | `[num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]` | 分页 KV cache。每个 slot 存储一个 token 的压缩 KV 表示（`kv_c_normed` 拼接 `k_pe`）。`num_blocks` 是总块数，`block_size` 是每块的 token 数。 |
| `attn_metadata` | `Any` | — | 注意力元数据对象，包含 batch 的调度信息。详见下方 §2.3。 |

**返回值**: `torch.Tensor`，shape = `[num_tokens, hidden_size]`，经过完整 MLA 计算（Q/KV 投影 → RoPE → KV cache 写入 → 注意力 → o_proj → 可选 TP all-reduce）后的输出。

### 2.2 `wrapper` 协议

`wrapper` 是一个 duck-typed 对象（通常是 `MultiHeadLatentAttentionWrapper` 实例），必须暴露以下属性：

#### 整型属性

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `q_lora_rank` | `int \| None` | Q 低秩维度，决定 Q 投影路径 |
| `kv_lora_rank` | `int` | KV 压缩维度 |
| `qk_rope_head_dim` | `int` | RoPE 维度 |
| `qk_nope_head_dim` | `int` | 非 RoPE 维度 |
| `qk_head_dim` | `int` | 完整 Q/K 头维度 |
| `num_heads` | `int` | 注意力头数 |
| `v_head_dim` | `int` | Value 头维度 |

#### 子模块（`q_lora_rank is not None` 时，即 DeepSeek V3/R1 路径）

| 子模块 | 类型 | 说明 |
| --- | --- | --- |
| `fused_qkv_a_proj` | `nn.Module`（含 `.weight`） | 融合的 Q/KV 下投影层，输出维度 = `q_lora_rank + kv_lora_rank + qk_rope_head_dim` |
| `q_a_layernorm` | `nn.Module`（含 `.weight`, `.variance_epsilon`） | Q 压缩表示的 RMSNorm |
| `q_b_proj` | `nn.Module`（含 `.weight`） | Q 上投影层，输出维度 = `num_heads * qk_head_dim` |

#### 子模块（`q_lora_rank is None` 时，即 DeepSeek V2 路径）

| 子模块 | 类型 | 说明 |
| --- | --- | --- |
| `q_proj` | `nn.Module`（含 `.weight`） | 直接 Q 投影层 |
| `kv_a_proj_with_mqa` | `nn.Module`（含 `.weight`） | KV 下投影层（含 MQA），输出维度 = `kv_lora_rank + qk_rope_head_dim` |

#### 始终必需的子模块

| 子模块 | 类型 | 说明 |
| --- | --- | --- |
| `kv_a_layernorm` | `nn.Module`（含 `.weight`, `.variance_epsilon`） | KV 压缩表示的 RMSNorm |
| `rotary_emb` | `nn.Module`（含 `.cos_sin_cache`，可选 `.is_neox_style`） | 旋转位置编码模块。`cos_sin_cache` shape = `[max_positions, qk_rope_head_dim * 2]` |
| `o_proj` | `nn.Module`（含 `.weight`） | 输出投影层，将 `[T, num_heads * v_head_dim]` 映射回 `[T, hidden_size]` |

### 2.3 `attn_metadata` 协议

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `slot_mapping` | `torch.Tensor` `[num_tokens]` | 每个 token 在 KV cache 中的 slot 索引。负值表示跳过写入。 |
| `num_decode_tokens` | `int` | batch 中 decode token 的数量。decode token 排列在 batch 的前部。 |
| `num_decodes` | `int` | decode 序列的数量。`> 0` 时触发 decode 注意力路径。 |
| `num_prefills` | `int` | prefill 序列的数量。`> 0` 时触发 prefill 注意力路径。 |
| `prefill` | `object` | Prefill 元数据子对象，包含以下属性： |
| ↳ `.query_start_loc` | `torch.Tensor` `[num_prefills + 1]` | 每个 prefill 序列的起始位置（cumulative sum） |
| ↳ `.max_query_len` | `int` | 最长 prefill 序列的长度 |
| ↳ `.block_table` | `torch.Tensor` `[num_prefills, max_blocks]` | 每个 prefill 序列的块表 |
| ↳ `.chunked_context` | `object \| None` | 分块上下文信息（chunked prefill 场景），含 `.cu_seq_lens` 属性。为 `None` 时表示无历史上下文。 |
| `decode` | `object` | Decode 元数据子对象，包含以下属性： |
| ↳ `.block_table` | `torch.Tensor` `[batch_size, max_blocks]` | 每个 decode 序列的块表 |
| ↳ `.seq_lens` | `torch.Tensor` `[batch_size]` | 每个 decode 序列的当前长度 |

---

## 3. `process_weights_after_loading` 参数

```python
def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
```

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `act_dtype` | `torch.dtype` | 模型激活值的数据类型（如 `torch.bfloat16`、`torch.float16`）。用于将 `kv_b_proj` 权重转换为目标 dtype，并构建吸收矩阵 `W_UK_T`（shape `[H, d_nope, R_kv]`）和 `W_UV`（shape `[H, R_kv, d_v]`）。 |

---

## 4. 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `FUSED_MLA_USE_ORIG_RMSNORM` | `"0"` | 设为 `"1"` 时强制使用 PyTorch 原生 RMSNorm 实现（调试用） |
| `FUSED_MLA_USE_ORIG_ROPE` | `"0"` | 设为 `"1"` 时强制使用 PyTorch 原生 RoPE 实现（调试用） |
| `FUSED_MLA_NUM_THREADS` | `"0"` | 并行线程数。`0` 或未设置时使用 `torch.get_num_threads()` 的值 |
