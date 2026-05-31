# -*- coding: utf-8 -*-
"""微内核管理框架的端到端测试。

测试覆盖：
  1. ``_C.list_microkernel_impls()`` 至少包含 ``baseline`` 和 ``scalar``。
  2. ``_C.validate_microkernel(impl, dtype, E, Sk)`` 对每个 enabled impl 的
     5 个 op（qkt_8x8 / qkt_8x4 / qkt_tail / pv_8x8 / pv_tail），在 fp32 与
     bf16 两种 dtype 下都返回 max-abs < 容差。
  3. ``_C.benchmark_microkernel(impl, dtype, E, Sk, iters, warmup)`` 返回
     合法 GFLOPS 数值（>0、有限）。
  4. SDPA 主路径：``flash2_neon_cache_<impl>`` 名字注册到 SdpaRegistry，
     不同 impl 之间数值在 dtype 容差内一致。``flash2_neon_cache``（无
     后缀）应等价于 ``flash2_neon_cache_baseline``（按位 0 差异）。

前 3 项是直接对微内核 op 的单测；第 4 项串到完整 SDPA 主循环，确认
模板化重构（``template <class MK>``）没破坏热路径语义。
"""
from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("fused_cpp._C")

from fused_cpp import _C  # noqa: E402

VALIDATE_OPS = (
    "qkt_8x8_max_abs",
    "qkt_8x4_max_abs",
    "qkt_tail_max_abs",
    "pv_8x8_max_abs",
    "pv_tail_max_abs",
)

BENCHMARK_OPS = ("qkt_8x8", "qkt_8x4", "pv_8x8")

# fp32 容差宽于 bf16 的 atol，因为 NEON 路径的 FMA 顺序与标量参考不同
# （vbfdotq_f32 / vbfmmlaq_f32 等会改变累加顺序）。
TOL_PER_DTYPE = {"fp32": 5e-3, "bf16": 1e-1}


def _impls() -> list[str]:
    return list(_C.list_microkernel_impls())


@pytest.fixture(scope="module")
def impls() -> list[str]:
    names = _impls()
    assert "baseline" in names, names
    assert "scalar" in names, names
    return names


# ─────────────────────────────────────────────────────────────────────────
# 1. registry 列表
# ─────────────────────────────────────────────────────────────────────────


def test_microkernel_registry_has_known_impls(impls):
    assert {"baseline", "scalar"}.issubset(set(impls))


# ─────────────────────────────────────────────────────────────────────────
# 2. validate：每个 impl × dtype 组合的 5 个 op 全部 max_abs < tol
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("impl", _impls())
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
@pytest.mark.parametrize(
    "shape",
    [(17, 19), (32, 32), (64, 8)],
    ids=["E17_Sk19", "E32_Sk32", "E64_Sk8"],
)
def test_microkernel_validate(impl, dtype, shape):
    E, Sk = shape
    result = _C.validate_microkernel(impl=impl, dtype=dtype, E=E, Sk=Sk)
    tol = TOL_PER_DTYPE[dtype]
    for op in VALIDATE_OPS:
        v = float(result[op])
        assert math.isfinite(v), (impl, dtype, op, v, result)
        assert v <= tol, (
            f"impl={impl} dtype={dtype} {op}={v:.3e} > tol={tol:.3e} "
            f"E={E} Sk={Sk} result={result}"
        )


# ─────────────────────────────────────────────────────────────────────────
# 3. benchmark：返回 GFLOPS / us / checksum 三元组合法
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("impl", _impls())
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_microkernel_benchmark_smoke(impl, dtype):
    result = _C.benchmark_microkernel(
        impl=impl, dtype=dtype, E=64, Sk=64, iterations=200, warmup=10
    )
    assert result["direct_microkernel"] == 1.0
    for op in BENCHMARK_OPS:
        sec = float(result[f"{op}_seconds"])
        us = float(result[f"{op}_us"])
        gflops = float(result[f"{op}_gflops"])
        chk = float(result[f"{op}_checksum"])
        assert math.isfinite(sec) and sec > 0.0, (impl, dtype, op, result)
        assert math.isfinite(us) and us > 0.0, (impl, dtype, op, result)
        assert math.isfinite(gflops) and gflops > 0.0, (impl, dtype, op, result)
        assert math.isfinite(chk), (impl, dtype, op, result)


def test_microkernel_unknown_impl_raises():
    with pytest.raises(RuntimeError, match="not registered"):
        _C.validate_microkernel(impl="__nonexistent__", dtype="fp32", E=8, Sk=8)


def test_microkernel_unknown_dtype_raises():
    with pytest.raises(RuntimeError, match="dtype must be"):
        _C.validate_microkernel(impl="baseline", dtype="int8", E=8, Sk=8)


# ─────────────────────────────────────────────────────────────────────────
# 4. SDPA 主路径：flash2_neon_cache_<impl> 注册成 SDPA 版本，dtype 容差内一致
# ─────────────────────────────────────────────────────────────────────────


def _registered_sdpa_versions() -> list[str]:
    return list(_C.list_sdpa_versions())


def test_sdpa_versions_have_per_impl_entries():
    """每个 enabled MK impl 都注册 flash2_neon_cache_<name> SDPA 版本。"""
    versions = _registered_sdpa_versions()
    for impl in _impls():
        name = f"flash2_neon_cache_{impl}"
        assert name in versions, (name, versions)
    # 历史名仍然在。
    assert "flash2_neon_cache" in versions, versions


def _run_sdpa(version: str, q, k, v):
    return _C.scaled_dot_product_attention_versioned(
        q, k, v, None, 0.0, False, None, False, version
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sdpa_flash2_neon_cache_alias_exact(dtype):
    """``flash2_neon_cache`` 应与 ``flash2_neon_cache_baseline`` 按位等价。"""
    torch.manual_seed(0)
    B, N, L, S, E, Ev = 1, 4, 16, 16, 64, 64
    q = torch.randn(B, N, L, E, dtype=dtype)
    k = torch.randn(B, N, S, E, dtype=dtype)
    v = torch.randn(B, N, S, Ev, dtype=dtype)
    a = _run_sdpa("flash2_neon_cache", q, k, v)
    b = _run_sdpa("flash2_neon_cache_baseline", q, k, v)
    assert torch.equal(a, b), (a - b).abs().max().item()


@pytest.mark.parametrize(
    "impl",
    [
        "qk_packqk_seq",
        "qk_packqk_seq4",
        "qk_packqk_seq4_ptr",
        "qk_packqk_seq4_bmajor",
        "qk_packqk_seq4_pipe_a",
        "qk_packqk_seq4_pipe_b",
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "head_dim",
    [32, 64, 128, 33, 67],
    ids=["E32", "E64", "E128", "E33-tail", "E67-tail"],
)
def test_microkernel_packqk_seq_vs_baseline_bit_exact(impl, dtype, head_dim):
    """Q/K 双 pack 微内核应与 ``baseline`` 在 max-abs 上按位一致。

    (与 framework 的 validate_microkernel 比较——两者都用同一个
    scalar reference，所以 baseline 与 packqk 系列的 max_abs 应相同。
    head_dim 33/67 专门测 E % 4 != 0 的 tail 路径。)

    重点：这些 packqk variants 仅 bf16 路径有自己的实现（fp32 fall through
    到 baseline），所以 fp32 case 期望 max_abs **完全相同**。
    """
    if impl not in _impls():
        pytest.skip(f"{impl} impl not built")
    dtype_str = "fp32" if dtype == torch.float32 else "bf16"
    Sk = head_dim  # validate_microkernel 用 (E, Sk) 两个维度
    base = _C.validate_microkernel(
        impl="baseline", dtype=dtype_str, E=head_dim, Sk=Sk
    )
    pq = _C.validate_microkernel(
        impl=impl, dtype=dtype_str, E=head_dim, Sk=Sk
    )
    # 两者都对同一个 scalar reference 算 max_abs；packqk 系列与 baseline
    # 的 BFMMLA 累加顺序按位一致（仅 Q/K 来源不同），因此 qkt_8x8 应**完全相同**。
    base_qkt = float(base["qkt_8x8_max_abs"])
    pq_qkt = float(pq["qkt_8x8_max_abs"])
    assert pq_qkt == base_qkt, (
        f"{impl} qkt_8x8 max_abs={pq_qkt:.3e} != baseline {base_qkt:.3e} "
        f"(dtype={dtype_str}, E={head_dim})"
    )
    # fp32 fall through，5 个 op 全部应与 baseline 完全相同。
    if dtype_str == "fp32":
        for op in (
            "qkt_8x8_max_abs",
            "qkt_8x4_max_abs",
            "qkt_tail_max_abs",
            "pv_8x8_max_abs",
            "pv_tail_max_abs",
        ):
            assert float(pq[op]) == float(base[op]), (
                f"{impl} {op}={pq[op]} != baseline {base[op]} (fp32 fall through)"
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sdpa_flash2_neon_cache_impls_equiv(dtype):
    """所有 ``flash2_neon_cache_<impl>`` 之间在 dtype 容差内一致。"""
    torch.manual_seed(0)
    B, N, L, S, E, Ev = 1, 4, 16, 16, 64, 64
    q = torch.randn(B, N, L, E, dtype=dtype)
    k = torch.randn(B, N, S, E, dtype=dtype)
    v = torch.randn(B, N, S, Ev, dtype=dtype)
    ref = _run_sdpa("flash2_neon_cache_baseline", q, k, v)
    atol = 1e-4 if dtype == torch.float32 else 5e-2
    rtol = 1e-4 if dtype == torch.float32 else 5e-2
    for impl in _impls():
        if impl == "baseline":
            continue
        got = _run_sdpa(f"flash2_neon_cache_{impl}", q, k, v)
        assert got.shape == ref.shape and got.dtype == ref.dtype
        torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)
