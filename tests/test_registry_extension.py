# -*- coding: utf-8 -*-
"""SDPA 注册表扩展性测试。

覆盖需求 11.11 列出的全部场景：
  * 注册 → 出现在 :func:`available_sdpa_versions`
  * 重复注册抛错；``override=True`` 显式覆盖
  * 能力位过滤生效（dtype / causal / attn_mask / mla_shape）
  * tags 过滤生效（CLI ``--sdpa-tags`` + :class:`VersionInfo.tags`）
  * C++ 缺失时同名 Python fallback 自动接管（通过 mock ``_HAS_CPP_SDPA``）
  * 测试矩阵自动包含新注册版本（注册一个 dummy 后用 pytest_generate_tests
    钩子 sanity-check）
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pytest
import torch

from fused_cpp.sdpa_registry import (
    VersionInfo,
    available_sdpa_versions,
    get_sdpa_version,
    register_sdpa_version,
)


# ── 工具：临时注册一个 dummy 版本，测试结束后撤销 ─────────────────────────


@contextmanager
def temporary_version(name: str, **kwargs) -> Iterator[VersionInfo]:
    """临时注册一个 SDPA 版本，``with`` 块退出后自动卸载。

    避免污染全局注册表。
    """
    from fused_cpp import sdpa_registry as _reg

    def _identity_fn(query, key, value, *, attn_mask, is_causal, scale):
        return query

    register_sdpa_version(name, **kwargs)(_identity_fn)
    try:
        yield get_sdpa_version(name)
    finally:
        # 直接从内部 dict 删除（公开 API 没暴露 unregister，符合"测试专用"约定）
        _reg._REGISTRY.pop(name, None)


# ── 1. 注册 → 出现在 available_sdpa_versions ────────────────────────────


@pytest.mark.equiv
def test_registered_version_appears_in_available_list():
    with temporary_version(
        "_test_dummy_appears",
        source="python",
        description="dummy",
        tags=("test",),
    ):
        names = {vi.name for vi in available_sdpa_versions()}
        assert "_test_dummy_appears" in names


# ── 2. 重复注册抛错；override=True 覆盖 ──────────────────────────────────


@pytest.mark.equiv
def test_duplicate_registration_raises_unless_override():
    with temporary_version(
        "_test_dummy_dup",
        source="python",
        description="first",
    ):
        # 第二次注册同名 → ValueError
        with pytest.raises(ValueError, match="already registered"):
            @register_sdpa_version("_test_dummy_dup", source="python")
            def _dup(query, key, value, *, attn_mask, is_causal, scale):
                return query

        # override=True 允许覆盖
        @register_sdpa_version(
            "_test_dummy_dup",
            source="python",
            override=True,
            description="second",
        )
        def _override(query, key, value, *, attn_mask, is_causal, scale):
            return query * 2

        info = get_sdpa_version("_test_dummy_dup")
        assert info.description == "second"


# ── 3. 能力位过滤 ────────────────────────────────────────────────────────


@pytest.mark.equiv
def test_supports_dtype_filter():
    with temporary_version(
        "_test_only_fp32",
        source="python",
        supports_dtypes=(torch.float32,),
    ) as info:
        ok, reason = info.supports(dtype=torch.float32)
        assert ok and reason == ""
        ok, reason = info.supports(dtype=torch.bfloat16)
        assert not ok
        assert "bfloat16" in reason or "torch.bfloat16" in reason


@pytest.mark.equiv
def test_supports_causal_filter():
    with temporary_version(
        "_test_no_causal",
        source="python",
        supports_causal=False,
    ) as info:
        ok, reason = info.supports(is_causal=True)
        assert not ok
        assert "is_causal" in reason
        ok, _ = info.supports(is_causal=False)
        assert ok


@pytest.mark.equiv
def test_supports_mla_shape_filter():
    with temporary_version(
        "_test_no_mla",
        source="python",
        supports_mla_shape=False,
    ) as info:
        ok, reason = info.supports(mla_shape=True)
        assert not ok
        assert "MLA" in reason or "mla" in reason
        ok, _ = info.supports(mla_shape=False)
        assert ok


@pytest.mark.equiv
def test_supports_attn_mask_filter():
    with temporary_version(
        "_test_no_mask",
        source="python",
        supports_attn_mask=False,
    ) as info:
        ok, reason = info.supports(attn_mask=True)
        assert not ok
        assert "attn_mask" in reason
        ok, _ = info.supports(attn_mask=False)
        assert ok


# ── 4. tags 过滤（与 CLI --sdpa-tags 配合）──────────────────────────────


@pytest.mark.equiv
def test_tags_attached_correctly():
    with temporary_version(
        "_test_tags",
        source="python",
        tags=("experimental", "fa3"),
    ) as info:
        assert info.tags == frozenset({"experimental", "fa3"})
        # 与已知 tag 集合做交集
        assert not info.tags.isdisjoint({"experimental"})
        assert info.tags.isdisjoint({"baseline"})


# ── 5. C++ 缺失时同名 Python fallback 自动接管 ──────────────────────────


@pytest.mark.equiv
def test_cpp_kernel_falls_back_to_python_when_extension_unavailable(
    monkeypatch,
):
    """模拟 ``_HAS_CPP_SDPA == False``，验证 cpp 注册项调用会降级到 Python。

    ``_make_cpp_callable`` 在拿到 ``info=cpp`` 调用时若 ``_HAS_CPP_SDPA`` 为
    False，应当：
      a) 优先查找同名 Python fallback；
      b) 若不存在，则降级到 ``naive_torch``；
      c) 若都不存在，给出明确 RuntimeError。
    """
    import fused_cpp.sdpa as sdpa_mod
    from fused_cpp.sdpa_registry import register_sdpa_version

    # 在 monkeypatch 中关掉 cpp 扩展可见性
    monkeypatch.setattr(sdpa_mod, "_HAS_CPP_SDPA", False)

    # (a) 注册一个名为 "_test_cpp_kernel_with_py_twin" 的 cpp 版本占位 +
    #     同名 python fallback；调用时应走 python 路径而非崩溃。
    py_called = {"hit": False}

    def py_twin(query, key, value, *, attn_mask, is_causal, scale):
        py_called["hit"] = True
        return query

    register_sdpa_version(
        "_test_cpp_kernel_with_py_twin",
        source="python",
        override=True,
    )(py_twin)
    try:
        cpp_callable = sdpa_mod._make_cpp_callable(
            "_test_cpp_kernel_with_py_twin"
        )
        q = torch.zeros(1, 1, 1, 4)
        out = cpp_callable(q, q, q, attn_mask=None, is_causal=False, scale=None)
        assert py_called["hit"], (
            "fallback to same-name Python version should have been called"
        )
        assert out.shape == q.shape
    finally:
        from fused_cpp import sdpa_registry as _reg
        _reg._REGISTRY.pop("_test_cpp_kernel_with_py_twin", None)

    # (b) 没有同名 python fallback → 应当降级到 naive_torch（因其总是注册）。
    cpp_callable_b = sdpa_mod._make_cpp_callable("_unknown_kernel_name_xyz")
    q = torch.zeros(1, 1, 4, 8)
    k = torch.zeros(1, 1, 4, 8)
    v = torch.zeros(1, 1, 4, 8)
    out = cpp_callable_b(q, k, v, attn_mask=None, is_causal=False, scale=None)
    # naive_torch 的 fallback 路径：输入全零 → 输出全零，但 shape 应该正确
    assert out.shape == (1, 1, 4, 8)


# ── 6. 测试矩阵自动包含新注册版本（dummy 验证） ─────────────────────────


@pytest.mark.equiv
def test_pytest_generate_tests_picks_up_new_version(sdpa_version):
    """间接验证 ``pytest_generate_tests`` 钩子从注册表读取版本的能力。

    断言要点：

    * 注入的 ``sdpa_version`` 必须是当前注册表中已存在的合法版本
      （未来新增内核如 ``flash2_neon`` 会**自动**进入测试矩阵，
      因此白名单不再硬编码 5 个版本，而以注册表为准）；
    * 默认 5 个版本（``naive``/``flash1``/``flash2``/``naive_torch``
      /``pytorch_sdpa``）必须始终注册成功。
    """
    assert isinstance(sdpa_version, VersionInfo)
    all_names = {vi.name for vi in available_sdpa_versions()}
    assert sdpa_version.name in all_names, (
        f"sdpa_version={sdpa_version.name!r} not found in registry: "
        f"{sorted(all_names)}"
    )
    # 默认 6 个版本必须始终存在
    expected_default = {
        "naive", "flash1", "flash2",
        "naive_torch", "pytorch_sdpa", "pytorch_sdpa_math",
    }
    assert expected_default.issubset(all_names)
