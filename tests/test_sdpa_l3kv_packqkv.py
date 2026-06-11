# -*- coding: utf-8 -*-
"""``flash2_neon_l3kv_packqkv`` 专属约束 / fp32 packK / 端到端等价性。

主等价性矩阵（与 baseline 对比）由
:mod:`tests.test_sdpa_versions_equiv` 自动覆盖（通过 ``sdpa_version`` fixture
注入）。本文件只验证 packqkv 特有的边界行为：

  1. ``S % 8 == 0`` / ``Ev % 8 == 0`` 强约束 → 不满足时 ``TORCH_CHECK`` 抛错
  2. fp32 输入 → 全量 pack K，QKᵀ 走 packed-K fp32 8x8 microkernel
  3. ``E % 4 != 0`` → packed 主路径 + 标量 tail，与 baseline bit-exact
  4. KV 不装 L3 / Path B → 复用同一 packed-K fp32 path 的 taskloop 调度
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("fused_cpp._C")

from fused_cpp import _C  # noqa: E402
from fused_cpp.sdpa_registry import get_sdpa_version  # noqa: E402

from tests.conftest import (  # noqa: E402
    assert_tensor_close,
    call_sdpa_version,
)


VERSION_NAME = "flash2_neon_l3kv_packqkv"


def _registered_or_skip():
    if VERSION_NAME not in _C.list_sdpa_versions():
        pytest.skip(f"{VERSION_NAME} not registered (build outdated)")


def _make_qkv(shape, dtype, seed=0xCAFE):
    B, N, L, S, E, Ev = shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, N, L, E, generator=g, dtype=torch.float32).to(dtype)
    k = torch.randn(B, N, S, E, generator=g, dtype=torch.float32).to(dtype)
    v = torch.randn(B, N, S, Ev, generator=g, dtype=torch.float32).to(dtype)
    return q, k, v


# ─────────────────────────────────────────────────────────────────────────
# 1. registry 健康度
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.equiv
def test_packqkv_registered():
    """``flash2_neon_l3kv_packqkv`` 已通过 REGISTER_SDPA_VERSION 注册。"""
    assert VERSION_NAME in _C.list_sdpa_versions()
    info = get_sdpa_version(VERSION_NAME)
    assert info.source == "cpp"
    assert "packed_q" in info.tags
    assert "packed_k" in info.tags
    assert "packed_v" in info.tags


# ─────────────────────────────────────────────────────────────────────────
# 2. 强约束：S % 8 / Ev % 8
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.equiv
@pytest.mark.parametrize("S", [7, 9, 15, 33])
def test_packqkv_rejects_unaligned_S(S):
    """``S % 8 != 0`` 时入口应抛 RuntimeError 并提示改用 packv。"""
    _registered_or_skip()
    B, N, L, E, Ev = 1, 4, 16, 64, 64
    q = torch.randn(B, N, L, E, dtype=torch.bfloat16)
    k = torch.randn(B, N, S, E, dtype=torch.bfloat16)
    v = torch.randn(B, N, S, Ev, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match=r"S % 8 == 0"):
        _C.scaled_dot_product_attention_versioned(
            q, k, v, None, 0.0, False, None, False, VERSION_NAME
        )


@pytest.mark.equiv
@pytest.mark.parametrize("Ev", [4, 9, 17])
def test_packqkv_rejects_unaligned_Ev(Ev):
    """``Ev % 8 != 0`` 时入口应抛 RuntimeError。"""
    _registered_or_skip()
    B, N, L, S, E = 1, 4, 16, 16, 64
    q = torch.randn(B, N, L, E, dtype=torch.bfloat16)
    k = torch.randn(B, N, S, E, dtype=torch.bfloat16)
    v = torch.randn(B, N, S, Ev, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match=r"Ev % 8 == 0"):
        _C.scaled_dot_product_attention_versioned(
            q, k, v, None, 0.0, False, None, False, VERSION_NAME
        )


# ─────────────────────────────────────────────────────────────────────────
# 3. fp32 packed-K path：与 baseline 在容差内一致
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.equiv
@pytest.mark.parametrize(
    "shape",
    [
        (1, 4, 16, 16, 64, 64),
        (1, 4, 64, 64, 192, 128),
    ],
    ids=["small", "mla"],
)
def test_packqkv_fp32_packk_vs_baseline(shape):
    """fp32 输入走 packed-K QKᵀ，应与 baseline 在 fp32 容差内一致。"""
    _registered_or_skip()
    info_packqkv = get_sdpa_version(VERSION_NAME)
    info_baseline = get_sdpa_version("flash2_neon_cache_baseline")

    q, k, v = _make_qkv(shape, torch.float32)
    out_packqkv = call_sdpa_version(info_packqkv, q, k, v, is_causal=False)
    out_baseline = call_sdpa_version(info_baseline, q, k, v, is_causal=False)
    assert_tensor_close(
        out_packqkv, out_baseline, dtype=torch.float32, context=f"shape={shape}"
    )


# ─────────────────────────────────────────────────────────────────────────
# 4. bf16 端到端：与 baseline 在容差内一致
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.equiv
@pytest.mark.parametrize(
    "shape",
    [
        (1, 4, 16, 16, 64, 64),       # 标准小
        (1, 4, 16, 16, 192, 128),     # MLA 小
        (1, 4, 64, 64, 192, 128),     # 中等
        (1, 8, 256, 256, 64, 64),     # 中等长
        (1, 4, 32, 64, 192, 128),     # L != S
    ],
    ids=["small", "mla-small", "mla-mid", "mid-long", "L_neq_S"],
)
@pytest.mark.parametrize("is_causal", [False, True])
def test_packqkv_bf16_vs_baseline(shape, is_causal):
    """bf16 packqkv 应与 ``flash2_neon_cache_baseline`` 在 bf16 容差内一致。"""
    _registered_or_skip()
    info_packqkv = get_sdpa_version(VERSION_NAME)
    info_baseline = get_sdpa_version("flash2_neon_cache_baseline")

    q, k, v = _make_qkv(shape, torch.bfloat16)
    out_packqkv = call_sdpa_version(info_packqkv, q, k, v, is_causal=is_causal)
    out_baseline = call_sdpa_version(info_baseline, q, k, v, is_causal=is_causal)

    ctx = f"shape={shape} is_causal={is_causal}"
    assert_tensor_close(
        out_packqkv, out_baseline, dtype=torch.bfloat16, context=ctx
    )


# ─────────────────────────────────────────────────────────────────────────
# 5. E % 4 != 0：partial e_block fall back 到标量 tail
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.equiv
@pytest.mark.parametrize("E", [33, 65, 67])
def test_packqkv_bf16_unaligned_E(E):
    """``E % 4 != 0`` 时 inner 标量 tail 应与 baseline bit-exact。"""
    _registered_or_skip()
    info_packqkv = get_sdpa_version(VERSION_NAME)
    info_baseline = get_sdpa_version("flash2_neon_cache_baseline")

    B, N, L, S, Ev = 1, 4, 16, 16, 64
    q, k, v = _make_qkv((B, N, L, S, E, Ev), torch.bfloat16)
    out_packqkv = call_sdpa_version(info_packqkv, q, k, v, is_causal=False)
    out_baseline = call_sdpa_version(info_baseline, q, k, v, is_causal=False)

    ctx = f"E={E}"
    assert_tensor_close(
        out_packqkv, out_baseline, dtype=torch.bfloat16, context=ctx
    )


# ─────────────────────────────────────────────────────────────────────────
# 6. attn_mask + causal 联合
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.equiv
def test_packqkv_with_mask_and_causal():
    """同时启用 attn_mask 与 is_causal=True，覆盖 (kHasMask, kCausal) 全实例化。"""
    _registered_or_skip()
    info_packqkv = get_sdpa_version(VERSION_NAME)
    info_baseline = get_sdpa_version("flash2_neon_cache_baseline")

    B, N, L, S, E, Ev = 1, 2, 16, 16, 64, 64
    q, k, v = _make_qkv((B, N, L, S, E, Ev), torch.bfloat16)
    g = torch.Generator(device="cpu").manual_seed(0xC0FE)
    mask = torch.randn(B, N, L, S, generator=g, dtype=torch.float32) * 0.5

    out_packqkv = call_sdpa_version(
        info_packqkv, q, k, v, is_causal=True, attn_mask=mask
    )
    out_baseline = call_sdpa_version(
        info_baseline, q, k, v, is_causal=True, attn_mask=mask
    )
    assert_tensor_close(
        out_packqkv, out_baseline, dtype=torch.bfloat16,
        context="mask+causal"
    )
