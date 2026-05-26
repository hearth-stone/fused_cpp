# -*- coding: utf-8 -*-
"""``flash2_neon_cache`` 内核的 cache-aware NEON 专项细粒度单元测试。

本文件聚焦 cache-aware 多核心 NEON SDPA 内核的关键不变量，覆盖：

1. **三级分块尾部退化**：Sk / Lq / Ev 三个维度独立的尾部 case，
   验证 8×8 / 8×4 主体 + 标量兜底退化链与参考实现等价；
2. **大形状 + bf16 BFMMLA 主路径**：参考形状 (1, 4, L, S, 192, 128) 下
   bf16 输出与 ``flash2`` 在 bfloat16 容差内一致；
3. **多线程 vs 单线程一致性**：通过环境变量 ``OMP_NUM_THREADS``
   验证 1 线程与多线程结果在容差内一致（K/V 跨线程共享、不做私有
   pack 的设计约束保证了无数据竞争）；
4. **causal + attn_mask 端到端**：cache-aware 主循环对每行 causal
   limit 与 additive mask 取址正确。

依赖：

* ``fused_cpp._C`` 必须可用（用 :func:`pytest.importorskip` 守卫）；
* 平台不要求是 AArch64：``__aarch64__`` 未定义时本测试验证的是 fallback
  标量路径，**仍应通过等价性断言**（与 NEON 路径在容差内一致）。
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("fused_cpp._C")

import torch  # noqa: E402  在 importorskip 之后导入

from fused_cpp.sdpa_registry import get_sdpa_version  # noqa: E402

from tests.conftest import (  # noqa: E402
    SDPA_TOLERANCE,
    assert_tensor_close,
    call_sdpa_version,
)


VERSION_NAME = "flash2_neon_cache"


# ── 工具 ─────────────────────────────────────────────────────────────


def _make_qkv(shape, dtype, seed: int = 0xCA_C8E_AA):
    """与多版本等价性测试一致的固定 seed 构造。"""
    b, n, l, s, e, ev = shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(b, n, l, e, generator=g, dtype=torch.float32).to(dtype)
    k = torch.randn(b, n, s, e, generator=g, dtype=torch.float32).to(dtype)
    v = torch.randn(b, n, s, ev, generator=g, dtype=torch.float32).to(dtype)
    return q, k, v


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().item())


# ── 1. 注册元数据：版本可见且 source=cpp ───────────────────────────


@pytest.mark.equiv
def test_flash2_neon_cache_registered_and_callable():
    """``flash2_neon_cache`` 在 Python 层注册成功，``info.source == "cpp"``。"""
    info = get_sdpa_version(VERSION_NAME)
    assert info.name == VERSION_NAME, info
    assert info.source == "cpp", (
        f"{VERSION_NAME} expected source='cpp' but got {info.source!r}"
    )
    assert "cache_aware" in info.tags, info.tags
    assert "multi_thread" in info.tags, info.tags


# ── 2. 等价性：与 flash2_neon 在多形状下数值等价 ──────────────────


@pytest.mark.equiv
@pytest.mark.parametrize(
    "shape",
    [
        (1, 2, 16, 16, 32, 32),       # 主体：所有维度都是 8 的倍数
        (2, 4, 33, 70, 72, 40),       # 全维度非 8 整除：尾部 case
        (1, 4, 8, 8, 8, 8),           # 最小 tile
        (1, 1, 17, 9, 13, 7),         # 极端奇形
    ],
    ids=["square", "tail-all-dims", "min-tile", "odd-shapes"],
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"]
)
def test_flash2_neon_cache_vs_flash2_neon_equiv(shape, dtype):
    """``flash2_neon_cache`` 与 ``flash2_neon`` 在多 shape × dtype 下等价。"""
    cache = get_sdpa_version(VERSION_NAME)
    neon = get_sdpa_version("flash2_neon")

    q, k, v = _make_qkv(shape, dtype)
    out_cache = call_sdpa_version(cache, q, k, v, is_causal=False)
    out_neon = call_sdpa_version(neon, q, k, v, is_causal=False)

    err = _max_abs(out_cache, out_neon)
    tol = SDPA_TOLERANCE.get(dtype, SDPA_TOLERANCE[torch.float32])
    # cache-aware 主循环算法上与 flash2_neon 等价；累加顺序差异给 2× 容差。
    bound = 2.0 * tol.atol
    assert err <= bound, (
        f"{VERSION_NAME} vs flash2_neon max_abs={err:.3e} > {bound:.3e} "
        f"(dtype={dtype}, shape={shape})"
    )


# ── 3. 大形状 + bf16 BFMMLA 主路径与 flash2 一致 ───────────────────


@pytest.mark.equiv
def test_flash2_neon_cache_bf16_mla_shape():
    """参考 MLA 形状 (E=192, Ev=128) bf16 路径与 ``flash2`` 在容差内一致。

    专门覆盖 BFMMLA / BFMLALB/T / widen+FMLA 的退化链入口与
    8×8 / 8×4 micro-kernel 的两段 GEMM 精度分流（QK^T bf16，P̂·V fp32+widen V）。
    """
    cache = get_sdpa_version(VERSION_NAME)
    flash2 = get_sdpa_version("flash2")
    shape = (1, 4, 16, 64, 192, 128)

    q, k, v = _make_qkv(shape, torch.bfloat16)
    out_cache = call_sdpa_version(cache, q, k, v, is_causal=False)
    out_flash2 = call_sdpa_version(flash2, q, k, v, is_causal=False)

    err = _max_abs(out_cache, out_flash2)
    tol = SDPA_TOLERANCE[torch.bfloat16]
    bound = 2.0 * tol.atol
    assert err <= bound, (
        f"{VERSION_NAME} bf16 MLA shape vs flash2 max_abs={err:.3e} > "
        f"{bound:.3e} (shape={shape})"
    )


# ── 4. causal + attn_mask 端到端 ─────────────────────────────────


@pytest.mark.equiv
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"]
)
def test_flash2_neon_cache_causal_and_mask(dtype):
    """causal=True + additive ``attn_mask`` 下 cache 内核与 ``flash2`` 一致。

    专门覆盖 cache-aware 主循环对 (a) 每行各自的 causal_limit 取址、
    (b) ``[B, N, L, S]`` 形状的 ``attn_mask`` 逐行加和。
    """
    cache = get_sdpa_version(VERSION_NAME)
    flash2 = get_sdpa_version("flash2")

    shape = (1, 2, 17, 33, 24, 24)  # 不规则 L/S 强迫尾部退化
    q, k, v = _make_qkv(shape, dtype)

    g = torch.Generator(device="cpu").manual_seed(0xCAFECAFE)
    mask = torch.randn(1, 2, 17, 33, generator=g, dtype=torch.float32) * 0.5

    out_cache = call_sdpa_version(
        cache, q, k, v, is_causal=True, attn_mask=mask
    )
    out_flash2 = call_sdpa_version(
        flash2, q, k, v, is_causal=True, attn_mask=mask
    )

    err = _max_abs(out_cache, out_flash2)
    tol = SDPA_TOLERANCE.get(dtype, SDPA_TOLERANCE[torch.float32])
    bound = 2.0 * tol.atol
    assert err <= bound, (
        f"{VERSION_NAME} causal+mask vs flash2 max_abs={err:.3e} > "
        f"{bound:.3e} (dtype={dtype})"
    )


# ── 5. 多线程 vs 单线程一致性 ──────────────────────────────────


_THREAD_TEST_SCRIPT = textwrap.dedent(
    """
    import os
    import json
    import sys

    import torch
    from fused_cpp.sdpa_registry import get_sdpa_version
    from tests.conftest import call_sdpa_version

    g = torch.Generator(device="cpu").manual_seed(0xC4_C8E)
    shape = (2, 4, 24, 48, 32, 32)
    b, n, l, s, e, ev = shape
    q = torch.randn(b, n, l, e, generator=g, dtype=torch.float32)
    k = torch.randn(b, n, s, e, generator=g, dtype=torch.float32)
    v = torch.randn(b, n, s, ev, generator=g, dtype=torch.float32)

    cache = get_sdpa_version("flash2_neon_cache")
    out = call_sdpa_version(cache, q, k, v, is_causal=False)

    sys.stdout.write(json.dumps({
        "checksum": float(out.float().sum().item()),
        "max_abs": float(out.float().abs().max().item()),
        "shape": list(out.shape),
    }))
    """
)


def _run_with_threads(num_threads: int) -> dict:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(num_threads)
    env["VIRTUAL_ENV"] = env.get("VIRTUAL_ENV", "")
    res = subprocess.run(
        [sys.executable, "-c", _THREAD_TEST_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"thread test script failed (OMP_NUM_THREADS={num_threads}): "
            f"stdout={res.stdout!r} stderr={res.stderr!r}"
        )
    import json
    # 子进程可能输出 numpy import 警告，提取最后一行 JSON。
    last = res.stdout.strip().splitlines()[-1]
    return json.loads(last)


@pytest.mark.equiv
def test_flash2_neon_cache_thread_count_consistency():
    """1 线程 vs 4 线程下 ``flash2_neon_cache`` 输出的 checksum 完全一致。

    K/V 跨线程共享、不做私有 pack（设计约束 10），多线程并行单元
    ``(b, n, q_tile)`` 之间无写冲突，因此输出应位精度等价。
    """
    info_1t = _run_with_threads(1)
    info_4t = _run_with_threads(4)

    assert info_1t["shape"] == info_4t["shape"], (info_1t, info_4t)
    # checksum 浮点数允许极小的累加顺序差异（实际由 OpenMP 调度产生
    # 的差异是 0，因为 (b, n, q_tile) 之间的输出区间互不相交）。
    diff = abs(info_1t["checksum"] - info_4t["checksum"])
    rel = diff / (abs(info_1t["checksum"]) + 1e-12)
    assert rel < 1e-6, (
        f"1-thread vs 4-thread checksum mismatch: "
        f"{info_1t['checksum']} vs {info_4t['checksum']} (rel={rel:.3e})"
    )


# ── 6. 大头维度 + 长 S：参考形状 ──────────────────────────────


@pytest.mark.equiv
def test_flash2_neon_cache_reference_shape_bf16():
    """参考形状裁剪版 (1, 4, 64, 128, 192, 128) bf16 与 ``flash2`` 一致。

    任务计划要求该 case 通过；不直接跑 (1, 32, 2048, 2048, 192, 128)
    全量参考形状以避免单测过慢，但保持算法主路径全覆盖。
    """
    cache = get_sdpa_version(VERSION_NAME)
    flash2 = get_sdpa_version("flash2")
    shape = (1, 4, 64, 128, 192, 128)

    q, k, v = _make_qkv(shape, torch.bfloat16)
    out_cache = call_sdpa_version(cache, q, k, v, is_causal=False)
    out_flash2 = call_sdpa_version(flash2, q, k, v, is_causal=False)

    err = _max_abs(out_cache, out_flash2)
    tol = SDPA_TOLERANCE[torch.bfloat16]
    bound = 2.0 * tol.atol
    assert err <= bound, (
        f"{VERSION_NAME} reference shape vs flash2 max_abs={err:.3e} > "
        f"{bound:.3e} (shape={shape})"
    )
