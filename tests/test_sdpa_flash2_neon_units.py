# -*- coding: utf-8 -*-
"""``flash2_neon`` 内核的 NEON 专项细粒度单元测试。

本文件聚焦 NEON 向量化中两个最容易出错的子模块：

1. **向量化 ``vexpq_f32`` 近似**：通过 fp32 + 长 ``S`` 序列下
   ``flash2_neon`` 与 ``naive_torch`` 的余弦相似度断言间接验证；
2. **``Q*K^T`` GEMV 内核（含尾部标量补齐）**：用 ``head_dim`` 不能被
   4 整除的形状（如 ``E=192, Ev=128``）覆盖尾部回退分支。

依赖：

* ``fused_cpp._C`` 必须可用（用 :func:`pytest.importorskip` 守卫）；
* 平台不要求是 AArch64：``__aarch64__`` 未定义时本测试验证的是 fallback
  标量路径，**仍应通过等价性断言**（与 NEON 路径在容差内一致）。
"""
from __future__ import annotations

import pytest

pytest.importorskip("fused_cpp._C")

import torch  # noqa: E402  在 importorskip 之后导入

from fused_cpp.sdpa_registry import get_sdpa_version  # noqa: E402

from tests.conftest import (  # noqa: E402
    SDPA_TOLERANCE,
    assert_tensor_close,
    call_sdpa_version,
)


# ── 工具 ─────────────────────────────────────────────────────────────


def _make_qkv(shape, dtype, seed: int = 0xF2_0E0_0):
    """与多版本等价性测试一致的固定 seed 构造。"""
    b, n, l, s, e, ev = shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(b, n, l, e, generator=g, dtype=torch.float32).to(dtype)
    k = torch.randn(b, n, s, e, generator=g, dtype=torch.float32).to(dtype)
    v = torch.randn(b, n, s, ev, generator=g, dtype=torch.float32).to(dtype)
    return q, k, v


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.detach().float().flatten().unsqueeze(0)
    b32 = b.detach().float().flatten().unsqueeze(0)
    return float(torch.nn.functional.cosine_similarity(a32, b32).item())


# ── 1. flash2_neon vs flash2 max_abs 报告（固定 seed）────────────────


@pytest.mark.equiv
def test_flash2_neon_vs_flash2_max_abs_report():
    """报告并断言 fp32 / bf16 下 ``flash2_neon`` 与 ``flash2`` 的逐元素 max_abs。

    断言 ``max_abs <= 2 * atol``（取 :data:`SDPA_TOLERANCE`），避免把
    NEON 累加顺序差异当成 bug。同时 ``print`` 出指标，方便排障。
    """
    neon = get_sdpa_version("flash2_neon")
    flash2 = get_sdpa_version("flash2")
    shape = (1, 4, 32, 64, 64, 64)

    for dtype in (torch.float32, torch.bfloat16):
        q, k, v = _make_qkv(shape, dtype)
        out_neon = call_sdpa_version(neon, q, k, v, is_causal=False)
        out_flash2 = call_sdpa_version(flash2, q, k, v, is_causal=False)

        err = _max_abs(out_neon, out_flash2)
        tol = SDPA_TOLERANCE.get(dtype, SDPA_TOLERANCE[torch.float32])
        bound = 2.0 * tol.atol
        assert err <= bound, (
            f"flash2_neon vs flash2 max_abs={err:.3e} exceeds {bound:.3e} "
            f"(dtype={dtype}, shape={shape})"
        )


# ── 2. fp32 下 flash2_neon vs naive_torch 余弦相似度 ≥ 0.99999 ─────


@pytest.mark.equiv
def test_flash2_neon_fp32_cosine_vs_naive_torch():
    """fp32 + 长 ``S`` 下，``flash2_neon`` 与 ``naive_torch`` 余弦相似度 >= 0.99999。

    长 ``S`` 让块级 online softmax 经历多次修正与多次 ``vexpq_f32``
    调用，是检测向量化 exp 近似累计误差的关键路径。
    """
    neon = get_sdpa_version("flash2_neon")
    ref = get_sdpa_version("naive_torch")
    # S=160 不被 BLOCK_S(64) 整除，强制 (160 // 64) + 1 块 + 尾部
    shape = (1, 2, 32, 160, 64, 64)

    q, k, v = _make_qkv(shape, torch.float32)
    out_neon = call_sdpa_version(neon, q, k, v, is_causal=False)
    out_ref = call_sdpa_version(ref, q, k, v, is_causal=False)

    cos = _cosine(out_neon, out_ref)
    assert cos >= 0.99999, (
        f"flash2_neon fp32 cosine vs naive_torch={cos:.6f} < 0.99999 "
        f"(shape={shape})"
    )


# ── 3. head_dim 不被 4 整除：覆盖尾部标量回退分支 ────────────────


@pytest.mark.equiv
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16],
                         ids=["fp32", "bf16"])
def test_flash2_neon_head_dim_tail_remainder(dtype):
    """``head_dim`` 不被 4 整除（``E=11, Ev=7``）时尾部标量回退分支正确。

    使用很小的素数 ``head_dim`` 显式逼迫 NEON 主体退化（4 路展开后
    ``E=11`` 剩 3 个、``Ev=7`` 剩 3 个走标量补齐），与 ``naive_torch``
    在标准 dtype 容差内一致。
    """
    neon = get_sdpa_version("flash2_neon")
    ref = get_sdpa_version("naive_torch")
    shape = (1, 2, 8, 16, 11, 7)

    q, k, v = _make_qkv(shape, dtype)
    out_neon = call_sdpa_version(neon, q, k, v, is_causal=False)
    out_ref = call_sdpa_version(ref, q, k, v, is_causal=False)

    assert_tensor_close(
        out_neon, out_ref,
        dtype=dtype,
        context=f"flash2_neon tail-remainder shape={shape} dtype={dtype}",
    )


# ── 4. MLA 形状 + bf16 widen / dot 路径覆盖 ───────────────────────


@pytest.mark.equiv
def test_flash2_neon_bf16_mla_shape():
    """``E=192, Ev=128`` MLA 形状下 ``flash2_neon`` bf16 路径与 ``flash2`` 一致。

    专门覆盖 bf16 ``Q*K^T`` 的 ``vbfdotq_f32``（启用时）/ widen 路径，
    以及 ``output_acc`` 的 widen + ``vfmaq_f32`` 累加。
    """
    neon = get_sdpa_version("flash2_neon")
    flash2 = get_sdpa_version("flash2")
    shape = (1, 4, 16, 64, 192, 128)

    q, k, v = _make_qkv(shape, torch.bfloat16)
    out_neon = call_sdpa_version(neon, q, k, v, is_causal=False)
    out_flash2 = call_sdpa_version(flash2, q, k, v, is_causal=False)

    err = _max_abs(out_neon, out_flash2)
    tol = SDPA_TOLERANCE[torch.bfloat16]
    bound = 2.0 * tol.atol
    assert err <= bound, (
        f"flash2_neon bf16 MLA shape vs flash2 max_abs={err:.3e} > "
        f"{bound:.3e} (shape={shape})"
    )
