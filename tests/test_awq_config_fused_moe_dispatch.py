#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``AWQConfig.get_quant_method`` 在 CPU / 非 CPU 平台对 FusedMoE 的分派回归测试。

背景：方案 B 的整体纪律是"ARM CPU 上 AWQ 的 Linear 与 MoE 都必须走
``fused_cpp`` 路径"，而**非** vLLM 原生的 marlin / MoeWNA16 路径。因此：

* CPU 平台 + fused_cpp 可用 + ``VLLM_CPU_AWQ_USE_FUSED_CPP=1`` →
  必须返回 :class:`CPUAWQFusedMoEMethod`。
* CPU 平台 + 环境变量关闭 →
  保持 vLLM 原有分支（让上层感知"CPU AWQ 没有可用 fused moe 路径"，避免
  静默走到 Marlin 路径在 CPU 上崩溃）。
* 非 CPU 平台 → 保持 marlin-first 原有策略不变。

本测试通过 ``monkeypatch`` 打桩 ``current_platform.is_cpu`` 及环境变量，
不启动真正的 vLLM engine。
"""

from __future__ import annotations

import pytest

pytest.importorskip("vllm", reason="本测试依赖 vLLM 源码树；请在 vLLM 仓库内运行")


@pytest.fixture
def awq_config():
    """最小可用 :class:`AWQConfig`：4bit, g128, zero_point=True。"""
    from vllm.model_executor.layers.quantization.awq import AWQConfig

    return AWQConfig(
        weight_bits=4,
        group_size=128,
        zero_point=True,
        modules_to_not_convert=None,
    )


@pytest.fixture
def fake_fused_moe_layer():
    """伪造一个 ``FusedMoE`` 实例用于 ``isinstance`` 判断。

    真实 ``FusedMoE`` 构造依赖大量 vLLM 上下文，这里用动态子类 + ``__new__``
    规避 ``__init__``，保留 ``isinstance(obj, FusedMoE) is True`` 的契约，
    并塞入 ``moe_config`` 哨兵供下游 method 的 ``__init__`` 使用。
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    FakeCls = type("FakeFusedMoE", (FusedMoE,), {})
    obj = FakeCls.__new__(FakeCls)
    object.__setattr__(obj, "moe_config", object())
    return obj


class TestAwqConfigGetQuantMethodFusedMoe:
    """CPU / 非 CPU × 环境变量开关下的 FusedMoE 分派策略。"""

    def test_cpu_with_env_on_routes_to_cpu_awq_fused_moe(
        self,
        awq_config,
        fake_fused_moe_layer,
        monkeypatch,
    ) -> None:
        """CPU 平台 + VLLM_CPU_AWQ_USE_FUSED_CPP=1 时应返回 ``CPUAWQFusedMoEMethod``。"""
        # Arrange
        import vllm.envs as envs
        import vllm.model_executor.layers.quantization.awq as awq_mod
        from vllm.model_executor.layers.quantization.awq import (
            CPUAWQFusedMoEMethod,
        )

        monkeypatch.setattr(
            awq_mod.current_platform,
            "is_cpu",
            lambda: True,
        )
        monkeypatch.setattr(envs, "VLLM_CPU_AWQ_USE_FUSED_CPP", True, raising=False)

        # Act
        method = awq_config.get_quant_method(
            fake_fused_moe_layer,
            prefix="model.layers.0.mlp.experts",
        )

        # Assert
        assert isinstance(method, CPUAWQFusedMoEMethod), (
            f"CPU+env 分支应返回 CPUAWQFusedMoEMethod，实际 {type(method).__name__}"
        )

    def test_cpu_with_env_off_does_not_route_to_cpu_awq_fused_moe(
        self,
        awq_config,
        fake_fused_moe_layer,
        monkeypatch,
    ) -> None:
        """CPU 平台但环境变量未开启时**不应**返回 ``CPUAWQFusedMoEMethod``。

        关闭开关时应按 vLLM 原生策略处理（marlin / MoeWNA16）。本用例只校验
        "没有误走到 CPUAWQFusedMoEMethod"，不强约束具体落到哪个 GPU 方法
        （那不是本 PR 的行为）。
        """
        # Arrange
        import vllm.envs as envs
        import vllm.model_executor.layers.quantization.awq as awq_mod
        from vllm.model_executor.layers.quantization.awq import (
            CPUAWQFusedMoEMethod,
        )

        monkeypatch.setattr(
            awq_mod.current_platform,
            "is_cpu",
            lambda: True,
        )
        monkeypatch.setattr(envs, "VLLM_CPU_AWQ_USE_FUSED_CPP", False, raising=False)

        # Act：此路径内部可能尝试构造 marlin workspace 导致失败，我们只关心
        # 是否落在 CPUAWQFusedMoEMethod；异常时也视为"未走到新路径"，合法。
        try:
            method = awq_config.get_quant_method(
                fake_fused_moe_layer,
                prefix="model.layers.0.mlp.experts",
            )
        except Exception:
            method = None

        # Assert
        assert not isinstance(method, CPUAWQFusedMoEMethod), (
            "关闭 VLLM_CPU_AWQ_USE_FUSED_CPP 时不应落到 CPUAWQFusedMoEMethod"
        )

    def test_non_cpu_platform_preserves_marlin_path(
        self,
        awq_config,
        fake_fused_moe_layer,
        monkeypatch,
    ) -> None:
        """非 CPU 分支必须保持 marlin-first 策略，不走 fused_cpp。"""
        # Arrange
        import vllm.model_executor.layers.quantization.awq as awq_mod

        monkeypatch.setattr(
            awq_mod.current_platform,
            "is_cpu",
            lambda: False,
        )

        called: dict[str, bool] = {"marlin": False}

        from vllm.model_executor.layers.quantization import awq_marlin as marlin_mod

        class _FakeAWQMarlinConfig:
            @classmethod
            def from_config(cls, cfg):  # noqa: D401 - stub
                called["marlin"] = True
                return cls()

            def get_quant_method(self, layer, prefix):  # noqa: D401 - stub
                return "marlin-method-sentinel"

        monkeypatch.setattr(
            marlin_mod,
            "AWQMarlinConfig",
            _FakeAWQMarlinConfig,
        )
        from vllm.model_executor.layers.quantization.utils import (
            marlin_utils,
        )

        monkeypatch.setattr(
            marlin_utils,
            "check_moe_marlin_supports_layer",
            lambda layer, group_size: True,
        )

        # Act
        result = awq_config.get_quant_method(
            fake_fused_moe_layer,
            prefix="model.layers.0.mlp.experts",
        )

        # Assert
        assert called["marlin"], "非 CPU 分支必须触达 AWQMarlinConfig"
        assert result == "marlin-method-sentinel", f"应返回 AWQMarlinConfig 产生的 method，实际 {result!r}"
