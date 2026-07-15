#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vLLM ``AWQLinearMethod`` + ``fused_cpp.w4a8_linear`` 集成分派等价性测试。

覆盖三件事：

1. 环境变量守卫：``VLLM_CPU_AWQ_USE_FUSED_CPP`` + CPU 平台 → 分派到 fused_cpp；
   其他组合（非 CPU 平台 / 环境变量关闭 / fused_cpp 导入失败）→ 回落到原 AWQ 路径。
2. scales 既可为 fp16 也可为 bf16（对齐 ``--dtype=bfloat16`` 运行时契约）。
3. 激活 bf16 输入经 ``AWQLinearMethod._apply_fused_cpp`` 的结果与完整反量化参考
   实现（AWQ 解包 + fp32 matmul → bf16）在容差内等价。

规范对齐：本测试仅依赖 vLLM 算子层（``AWQLinearMethod`` / ``AWQConfig``），不引入
``LLMEngine`` / ``ModelRunner`` 等高层 API，遵循测试规则 §11.4 vLLM 专项。
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import fused_cpp
from fused_cpp.w4a8_linear import unpack_awq_qweight, unpack_awq_qzeros

# 被测模块（vLLM 侧的 AWQ 分派点）。
# 若 vLLM 不可导入（例如未 install 或仅装 fused_cpp），则本文件全量 skip，
# 因为集成测试的前提就是 vLLM 运行时可用。
vllm_awq = pytest.importorskip(
    "vllm.model_executor.layers.quantization.awq",
    reason="需要在 vLLM 可导入的环境下运行集成测试",
)
vllm_envs = importlib.import_module("vllm.envs")
AWQConfig = vllm_awq.AWQConfig
AWQLinearMethod = vllm_awq.AWQLinearMethod


# ── 容差 ──
# 与 tests/test_w4a8_linear.py::_TOL_W4A8 保持一致：per-token int8 激活量化噪声
# 导致 max_abs 常落在 2e-2 ~ 5e-2，余弦相似度阈值仍保持严格。
_TOL_W4A8 = dict(rtol=5e-2, atol=5e-2, cos_sim_threshold=0.999)

# AWQ interleave 顺序：第 i 个 nibble 对应 N 维第 AWQ_ORDER[i] 列
_AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


# ── 辅助函数 ──


def _pack_awq_along_n(unpacked: torch.Tensor) -> torch.Tensor:
    """沿 N 维按 AWQ interleave 顺序将 int4 (0..15) pack 为 int32。"""
    assert unpacked.shape[-1] % 8 == 0
    lead = unpacked.shape[:-1]
    n = unpacked.shape[-1]
    reshaped = unpacked.reshape(*lead, n // 8, 8).to(torch.int32) & 0xF
    order_idx = torch.tensor(_AWQ_ORDER, dtype=torch.long)
    picked = reshaped.index_select(-1, order_idx)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32).view(
        *([1] * len(lead)),
        1,
        8,
    )
    return (picked << shifts).sum(dim=-1).to(torch.int32)


def _build_awq_weights(
    k: int,
    n: int,
    group_size: int,
    scales_dtype: torch.dtype,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造随机 (qweight, qzeros, scales)。scales_dtype 支持 fp16 / bf16。"""
    gen = torch.Generator().manual_seed(seed)
    w_int4 = torch.randint(0, 16, (k, n), generator=gen, dtype=torch.int32)
    groups = k // group_size
    z_int4 = torch.randint(0, 16, (groups, n), generator=gen, dtype=torch.int32)
    scales_fp32 = torch.rand((groups, n), generator=gen, dtype=torch.float32) * 0.02 + 0.001
    qweight = _pack_awq_along_n(w_int4)
    qzeros = _pack_awq_along_n(z_int4)
    return qweight, qzeros, scales_fp32.to(scales_dtype)


def _reference_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """完整反量化 + fp32 matmul → bf16 的参考实现。"""
    groups = qzeros.shape[0]
    k = qweight.shape[0]
    group_size = k // groups

    w_q = unpack_awq_qweight(qweight).to(torch.float32)
    w_z = unpack_awq_qzeros(qzeros).to(torch.float32)
    w_z_expanded = w_z.repeat_interleave(group_size, dim=0)
    s_expanded = scales.to(torch.float32).repeat_interleave(group_size, dim=0)
    w_fp = (w_q - w_z_expanded) * s_expanded

    out = torch.matmul(x.to(torch.float32), w_fp)
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(torch.bfloat16)


def _assert_tensor_close(
    actual: torch.Tensor,
    ref: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    cos_sim_threshold: float,
) -> None:
    """统一等价性断言：误差 + 余弦相似度 + NaN/Inf 鲁棒性。"""
    assert actual.shape == ref.shape, f"shape mismatch: {actual.shape} vs {ref.shape}"
    assert actual.dtype == ref.dtype, f"dtype mismatch: {actual.dtype} vs {ref.dtype}"
    for name, t in (("actual", actual), ("ref", ref)):
        n_nan = torch.isnan(t).sum().item()
        n_inf = torch.isinf(t).sum().item()
        assert n_nan == 0, f"{name} has {n_nan} NaN"
        assert n_inf == 0, f"{name} has {n_inf} Inf"
    diff = (actual.float() - ref.float()).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (ref.float().abs() + 1e-12)).max().item()
    cos = torch.nn.functional.cosine_similarity(
        actual.flatten().float().unsqueeze(0),
        ref.flatten().float().unsqueeze(0),
    ).item()
    msg = f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} cos={cos:.6f}"
    assert max_abs <= atol or max_rel <= rtol, f"tolerance failed: {msg}"
    assert cos >= cos_sim_threshold, f"cosine similarity failed: {msg}"


def _make_fake_layer(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
) -> Any:
    """构造最小 `layer` 占位对象，只暴露 ``AWQLinearMethod`` 需要的三个属性。"""
    return SimpleNamespace(qweight=qweight, qzeros=qzeros, scales=scales)


def _make_awq_method() -> AWQLinearMethod:
    """构造一个最小 AWQLinearMethod 实例（group_size / zero_point 均采用常见默认）。"""
    cfg = AWQConfig(weight_bits=4, group_size=32, zero_point=True)
    return AWQLinearMethod(cfg)


# ── 1. 环境变量 + 平台守卫 ──


class TestFusedCppDispatchGuard:
    """``_should_use_fused_cpp_awq`` 的开关与平台守卫。"""

    def test_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """默认 ``VLLM_CPU_AWQ_USE_FUSED_CPP`` 未开启 → 不走 fused_cpp。"""
        monkeypatch.setattr(vllm_envs, "VLLM_CPU_AWQ_USE_FUSED_CPP", False)
        assert vllm_awq._should_use_fused_cpp_awq() is False

    def test_enabled_on_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CPU 平台 + 环境变量开 + fused_cpp 可导入 → 走 fused_cpp。"""
        monkeypatch.setattr(vllm_envs, "VLLM_CPU_AWQ_USE_FUSED_CPP", True)
        monkeypatch.setattr(
            vllm_awq.current_platform,
            "is_cpu",
            lambda: True,
        )
        # 保证 fused_cpp 缓存未被前序失败污染
        monkeypatch.setattr(vllm_awq, "_FUSED_CPP_LOAD_FAILED", False)
        assert vllm_awq._should_use_fused_cpp_awq() is True

    def test_disabled_on_non_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """非 CPU 平台即使开启环境变量也不分派到 fused_cpp。"""
        monkeypatch.setattr(vllm_envs, "VLLM_CPU_AWQ_USE_FUSED_CPP", True)
        monkeypatch.setattr(
            vllm_awq.current_platform,
            "is_cpu",
            lambda: False,
        )
        assert vllm_awq._should_use_fused_cpp_awq() is False

    def test_fallback_on_import_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """fused_cpp 不可导入时应静默回落（返回 False）。"""
        monkeypatch.setattr(vllm_envs, "VLLM_CPU_AWQ_USE_FUSED_CPP", True)
        monkeypatch.setattr(
            vllm_awq.current_platform,
            "is_cpu",
            lambda: True,
        )
        monkeypatch.setattr(vllm_awq, "_FUSED_CPP_W4A8_LINEAR", None)
        monkeypatch.setattr(vllm_awq, "_FUSED_CPP_LOAD_FAILED", True)
        assert vllm_awq._should_use_fused_cpp_awq() is False


# ── 2. AWQLinearMethod._apply_fused_cpp 等价性 ──


class TestAWQLinearMethodFusedCpp:
    """分派进入 ``_apply_fused_cpp`` 后的数值等价性。"""

    @pytest.mark.parametrize(
        "scales_dtype",
        [torch.float16, torch.bfloat16],
        ids=["fp16_scales", "bf16_scales"],
    )
    @pytest.mark.parametrize(
        "shape",
        [(4, 128, 64), (16, 256, 128)],
        ids=["mid", "large"],
    )
    def test_apply_fused_cpp_matches_reference(
        self,
        scales_dtype: torch.dtype,
        shape: tuple[int, int, int],
    ) -> None:
        """``_apply_fused_cpp`` 的输出与完整反量化参考在容差内等价。"""
        # Arrange
        m, k, n = shape
        group_size = 32
        torch.manual_seed(0)
        qweight, qzeros, scales = _build_awq_weights(
            k,
            n,
            group_size,
            scales_dtype,
            seed=0,
        )
        x = torch.randn(m, k, dtype=torch.bfloat16)
        layer = _make_fake_layer(qweight, qzeros, scales)
        method = _make_awq_method()

        # Act
        out = method._apply_fused_cpp(layer, x, bias=None)
        ref = _reference_linear(x, qweight, qzeros, scales, bias=None)

        # Assert
        _assert_tensor_close(out, ref, **_TOL_W4A8)

    def test_apply_fused_cpp_with_bias(self) -> None:
        """带 bias 分派同样应与参考实现一致。"""
        # Arrange
        m, k, n, group_size = 8, 128, 64, 32
        torch.manual_seed(1)
        qweight, qzeros, scales = _build_awq_weights(
            k,
            n,
            group_size,
            torch.bfloat16,
            seed=1,
        )
        x = torch.randn(m, k, dtype=torch.bfloat16)
        bias = torch.randn(n, dtype=torch.bfloat16)
        layer = _make_fake_layer(qweight, qzeros, scales)
        method = _make_awq_method()

        # Act
        out = method._apply_fused_cpp(layer, x, bias=bias)
        ref = _reference_linear(x, qweight, qzeros, scales, bias=bias)

        # Assert
        _assert_tensor_close(out, ref, **_TOL_W4A8)

    def test_apply_fused_cpp_preserves_leading_dims(self) -> None:
        """3D 输入 (batch × seq × hidden) 应保留前导维度。"""
        # Arrange
        b, s, k, n, group_size = 2, 16, 128, 64, 32
        torch.manual_seed(2)
        qweight, qzeros, scales = _build_awq_weights(
            k,
            n,
            group_size,
            torch.bfloat16,
            seed=2,
        )
        x = torch.randn(b, s, k, dtype=torch.bfloat16)
        layer = _make_fake_layer(qweight, qzeros, scales)
        method = _make_awq_method()

        # Act
        out = method._apply_fused_cpp(layer, x, bias=None)
        ref = _reference_linear(x, qweight, qzeros, scales, bias=None)

        # Assert
        assert out.shape == (b, s, n)
        _assert_tensor_close(out, ref, **_TOL_W4A8)


# ── 3. apply() 的分派路径 ──


class TestAWQLinearMethodApplyDispatch:
    """验证 ``apply()`` 在守卫开启/关闭下分别走 fused_cpp / 原 AWQ 路径。"""

    def test_apply_dispatches_to_fused_cpp_when_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """守卫 True 时 ``apply()`` 走 ``_apply_fused_cpp``。"""
        # 强制守卫返回 True，避免依赖真实 current_platform
        monkeypatch.setattr(
            vllm_awq,
            "_should_use_fused_cpp_awq",
            lambda: True,
        )

        # Arrange
        m, k, n, group_size = 4, 128, 64, 32
        torch.manual_seed(3)
        qweight, qzeros, scales = _build_awq_weights(
            k,
            n,
            group_size,
            torch.bfloat16,
            seed=3,
        )
        x = torch.randn(m, k, dtype=torch.bfloat16)
        layer = _make_fake_layer(qweight, qzeros, scales)
        method = _make_awq_method()

        sentinel = torch.zeros(m, n, dtype=torch.bfloat16)
        called = {"fused": False}

        def _fake_apply_fused_cpp(_self: Any, _layer: Any, _x: Any, _bias: Any) -> torch.Tensor:
            called["fused"] = True
            return sentinel

        monkeypatch.setattr(
            AWQLinearMethod,
            "_apply_fused_cpp",
            _fake_apply_fused_cpp,
        )

        # Act
        out = method.apply(layer, x, bias=None)

        # Assert
        assert called["fused"] is True
        assert out is sentinel

    def test_apply_skips_fused_cpp_when_guard_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """守卫 False 时 ``apply()`` 不会调用 ``_apply_fused_cpp``。"""
        monkeypatch.setattr(
            vllm_awq,
            "_should_use_fused_cpp_awq",
            lambda: False,
        )

        # Arrange
        m, k, n, group_size = 4, 128, 64, 32
        torch.manual_seed(4)
        qweight, qzeros, scales = _build_awq_weights(
            k,
            n,
            group_size,
            torch.float16,
            seed=4,
        )
        x = torch.randn(m, k, dtype=torch.float16)
        layer = _make_fake_layer(qweight, qzeros, scales)
        method = _make_awq_method()

        called = {"fused": False}

        def _fake_apply_fused_cpp(*_args: Any, **_kwargs: Any) -> torch.Tensor:
            called["fused"] = True
            raise AssertionError("不应被调用")

        monkeypatch.setattr(
            AWQLinearMethod,
            "_apply_fused_cpp",
            _fake_apply_fused_cpp,
        )

        # Act / Assert（只验证分派未触发，不验证原 CUDA AWQ 数值结果——该路径需 GPU）
        try:
            method.apply(layer, x, bias=None)
        except Exception:
            # 原路径依赖 CUDA kernels；在 CPU-only 环境下 raise 属于预期。
            # 关键仅在于 `_apply_fused_cpp` 未被触发。
            pass
        assert called["fused"] is False


# ── 4. 冒烟测试：fused_cpp.w4a8_linear 自身可调用 ──


def test_fused_cpp_w4a8_linear_smoke() -> None:
    """冒烟测试：``fused_cpp.w4a8_linear`` 在 bf16 scales 下能直接跑通。"""
    k, n, group_size = 64, 32, 32
    qweight, qzeros, scales = _build_awq_weights(
        k,
        n,
        group_size,
        torch.bfloat16,
        seed=5,
    )
    x = torch.randn(2, k, dtype=torch.bfloat16)
    out = fused_cpp.w4a8_linear(x, qweight, qzeros, scales, bias=None)
    assert out.shape == (2, n)
    assert out.dtype == torch.bfloat16
