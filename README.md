# fused-cpp

CPU Fused MLA (Multi-head Latent Attention) and MoE (Mixture of Experts) —
a standalone, pip-installable PyTorch extension package.

- **MLA**: fused forward pass (projection, RoPE, KV-cache write, attention, output projection)
- **MoE**: full-token expert-parallel MoE forward pass

## Installation

```bash
# Runtime only, from this directory.
uv pip install -e .

# Tests and benchmark helpers.
uv pip install -e ".[dev]"
```

Equivalent `pip` commands work as well:

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"
```

`uv` is preferred on Linux / Windows because the project maps `torch` to
the PyTorch CPU wheel index via `tool.uv.sources`.

`vLLM` is intentionally not a default dependency. vLLM-specific integration
tests are skipped unless vLLM is importable in the active environment.

## Code Style

Install the pinned formatting tools without installing the runtime dependencies:

```bash
python -m pip install -r requirements-style.txt
```

Check Python lint/formatting and C/C++ formatting with the same entrypoint used
by CI:

```bash
bash scripts/check_code_style.sh
```

Apply all safe Ruff fixes, Ruff formatting, and clang-format formatting:

```bash
bash scripts/format_code.sh
```

Python uses Ruff with a 120-column limit. C/C++ uses clang-format 20.1.5 with
Google style and a 120-column limit. The frozen `refs/` directory and assembly
sources are intentionally excluded from automatic formatting.

## Quick Start

```python
from fused_cpp.mla import CPUFusedMLAImpl

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

## SDPA 多版本基准对比

`fused_cpp` 提供了一个 **多版本 SDPA 计算与基准比较框架**，便于在同一进程
中横向对比不同 SDPA 计算范式的延迟与算力利用率。

### 内置版本

| 名称 | 来源 | 算法 |
| --- | --- | --- |
| `naive`        | C++（参考实现）   | 全量物化 `attn_scores[L, S]` + 标准两遍 softmax |
| `flash1`       | C++              | FlashAttention-1 风格：K/V 外层 + Q 中层分块 + 两遍 Online-Softmax |
| `flash2`       | C++（默认内核）  | FlashAttention-2 风格：Q 行外层 + K/V 分块内层 + 单遍 Online-Softmax |
| `naive_torch`  | Python fallback  | 纯 PyTorch 朴素实现，与 `naive` 同 `causal_offset = S - L` 语义 |
| `pytorch_sdpa` | Python baseline  | 直接调用 `torch.nn.functional.scaled_dot_product_attention` |

> **注意**：本框架的 causal 语义是 `causal_offset = S - L` (lower-right)；
> PyTorch 内置 SDPA 在 `L != S` 时使用 upper-left 语义。等价性测试会
> 自动跳过 `pytorch_sdpa` 在 `L != S && is_causal=True` 子集下的互比。

### Python API

```python
from fused_cpp import sdpa_versioned, available_sdpa_versions

# 列出所有已注册版本
for vi in available_sdpa_versions():
    print(vi.name, vi.source, vi.tags, vi.description)

# 按版本调用
out = sdpa_versioned(q, k, v, version="flash1", is_causal=True)
```

### 注册一个新版本（≤ 30 行 Python）

```python
import torch
from fused_cpp import register_sdpa_version

@register_sdpa_version(
    "my_variant",
    source="python",
    supports_dtypes=(torch.float32, torch.bfloat16),
    supports_causal=True,
    tags=("experimental",),
    description="An experimental SDPA variant.",
)
def my_variant(query, key, value, *, attn_mask, is_causal, scale):
    # 你的实现 ...
    return out
```

注册后 **无需修改任何测试或 benchmark 文件**；新版本会被自动纳入：
* `pytest -m equiv` 等价性矩阵；
* `pytest -m bench` 基准矩阵；
* GFLOP/s 报告。

新增 C++ 内核时请参考模板：[csrc/sdpa_versions/_template.cpp](csrc/sdpa_versions/_template.cpp)
（≤ 5 步清单）。

### 运行等价性矩阵

```bash
# 默认快速矩阵（小 shape，5 个版本，~100 个用例，<1 秒）
pytest -m equiv tests/test_sdpa_versions_equiv.py

# 仅跑特定版本
pytest -m equiv tests/test_sdpa_versions_equiv.py --sdpa-versions=naive,flash2

# 仅跑特定 tag 的版本
pytest -m equiv tests/test_sdpa_versions_equiv.py --sdpa-tags=flash

# 启用大 shape 慢用例
pytest -m "equiv and slow" tests/test_sdpa_versions_equiv.py
```

### 运行基准（含 GFLOP/s 报告）

```bash
# 默认所有 5 个版本 × 3 shape × 2 dtype × 2 causal = 60 用例
pytest -m bench -s tests/bench_sdpa_versions.py

# 固定 8 线程，强制亲和（不绑核，由 OS + libomp 决定）
OMP_NUM_THREADS=8 OMP_PROC_BIND=close \
  pytest -m bench -s tests/bench_sdpa_versions.py

# 自动 sweep：1/2/4/8/16 线程，输出 scaling 表 + efficiency
pytest -m bench -s tests/bench_sdpa_versions.py \
    --sdpa-thread-sweep=1,2,4,8,16

# 自定义输出目录
pytest -m bench -s tests/bench_sdpa_versions.py \
    --sdpa-bench-output-dir=/tmp/sdpa_bench
```

输出文件落盘到 `<repo>/bench/sdpa/`：
* `sdpa_versions_<timestamp>.{csv,json}`：单次基准全部记录；
* `sdpa_versions_<timestamp>_sweep.{csv,json}`：thread sweep 全部记录；
* `sdpa_versions_<timestamp>_scaling.{csv,json}`：每版本扩展效率
  `efficiency(N) = gflops(N) / (N * gflops(1))`。

### 注意事项

1. **测量纪律**：每条用例执行 ≥ 5 次 warmup + ≥ 20 轮稳态测量，输出
   mean / stddev / median / P90 / P99 / GFLOP/s。
2. **FLOPs 公式**：dense 公式 `2*B*N*L*S*(E+Ev) + 5*B*N*L*S`；causal 模式
   按 `S_eff(l) = clip(l + (S - L) + 1, 0, S)` 折算。`attn_mask` **不**影响
   FLOPs 计算（避免 mask 稀疏度差异污染对比）。
3. **本框架不实现 CPU 核心绑定**：绑核行为完全由 `OMP_PROC_BIND` /
   `OMP_PLACES` 与 OS 调度器决定。
4. **macOS 友好**：构建系统会优先复用 PyTorch wheel 自带的
   ``libomp.dylib``，因此 Apple Silicon 上 ``_C.has_openmp()`` 默认返回
   ``True``，多线程基准与 thread sweep 可正常运行。如需关闭可设置环境
   变量 ``FUSED_CPP_DISABLE_OMP=1`` 重新构建。

## Debug Environment Variables

| Variable | Effect |
| --- | --- |
| `FUSED_MLA_USE_ORIG_RMSNORM=1` | Delegate RMSNorm to the wrapper's layernorm instead of the internal implementation |
| `FUSED_MLA_USE_ORIG_ROPE=1` | Delegate RoPE to the wrapper's `rotary_emb` instead of the internal implementation |

## Memory Page Policy

Every large buffer — MoE scratch, the attention workspace pool, and the packed
MoE weights — is backed through one policy, so the page size is chosen in a
single place rather than per allocation site.

| Variable | Effect |
| --- | --- |
| `FUSED_CPP_PAGES=small\|thp\|hugetlb` | Page backing. `thp` (default) uses anonymous `mmap` plus `madvise(MADV_HUGEPAGE)`; `hugetlb` uses `MAP_HUGETLB` and falls back to `thp` if the pool is exhausted; `small` uses plain aligned allocation. |
| `FUSED_CPP_PAGE_SIZE_MB=<int>` | Huge page size for `hugetlb`, default `32`. Must be a power of two the kernel supports, otherwise the default is used. |
| `FUSED_CPP_PAGE_MIN_KB=<int>` | Smallest request that may consume a whole huge page; smaller ones use `thp`. Defaults to one full page, which keeps waste per mapping under 2x. Without it many small scratch buffers each round up to a whole page. |
| `FUSED_CPP_HUGETLBFS_PATH=<mount>` | Implies `hugetlb` and probes the page size from the mount. |

`fused_cpp._moe_C.page_policy_info()` reports what was actually resolved plus
live, peak and fallback counters. The policy latches on the first allocation,
because freeing recomputes the mapping length from it, so change it through the
environment before the first forward pass rather than mid-run.

`FUSED_CPP_MOE_HUGETLB`, `FUSED_CPP_MOE_HUGETLB_MB`, `FUSED_CPP_MOE_THP` and
`FUSED_CPP_MOE_HUGETLBFS_PATH` remain as deprecated aliases and are consulted
only when none of the variables above are set. Packed weights are now allocated
on the requested pages directly; `FUSED_CPP_MOE_FORCE_HUGETLBFS_COPY=1` restores
the older behaviour of relocating them after packing, which costs a second
full-size copy and twice the huge-page budget.

## Build-Time Environment Variables

| Variable | Effect |
| --- | --- |
| `FUSED_CPP_DISABLE_OMP=1` | Build the C++ extension **without** OpenMP. `_C.has_openmp()` will return `False` and all `#pragma omp` directives become no-ops. |
| `LIBOMP_ROOT=<path>` | (macOS only) Use a custom libomp installation. The directory must contain `include/omp.h` and `lib/libomp.{dylib,a}`. By default the build prefers PyTorch's bundled `libomp.dylib`, then falls back to Homebrew (`/opt/homebrew/opt/libomp` or `/usr/local/opt/libomp`). |
| `FUSED_CPP_TARGET_CPU=<cpu/arch>` | (aarch64 only) Override the default `-march`/`-mcpu`. Strings starting with `apple-`, `cortex-`, or `neoverse-` are passed as `-mcpu=`; anything else as `-march=`. |
| `FUSED_CPP_ENABLE_KLEIDIAI=1` | Enable the optional KleidiAI BFMMLA backend. |

On **Linux** the build adds `-fopenmp` to both compile and link stages
(GCC / system Clang). On **macOS** it uses Apple Clang's
`-Xpreprocessor -fopenmp` and links against PyTorch's bundled libomp via
the absolute install-name `/opt/llvm-openmp/lib/libomp.dylib`, which keeps
the extension and PyTorch sharing a single OpenMP runtime and avoids
`OMP: Error #15` at import time.

## License

See the project root for license details.
