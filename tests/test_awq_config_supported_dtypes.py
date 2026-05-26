#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``AWQConfig.get_supported_act_dtypes`` 的平台分支回归测试。

背景：R1-AWQ 在 AArch64 CPU + ``--dtype=bfloat16`` 下启动时，vLLM 会调用
:meth:`vllm.model_executor.layers.quantization.awq.AWQConfig.get_supported_act_dtypes`
校验激活 dtype。原始实现只返回 ``[torch.half]``，会直接抛
``ValueError: torch.bfloat16 is not supported for quantization method awq``。

我们的修复：在 CPU 平台放行 ``torch.bfloat16``（因为 CPU 后端的 AWQ 通路走的是
``fused_cpp.w4a8_linear`` / ``CPUAWQFusedMoEMethod``，原生支持 bf16）；GPU 路径
保持不变，仅返回 ``[torch.half]``。

本测试用 ``monkeypatch`` 打桩 ``current_platform.is_cpu()`` 的返回值，覆盖
CPU 分支与非 CPU 分支；不依赖任何模型权重、不触发 engine 初始化。
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("vllm", reason="本测试依赖 vLLM 源码树；请在 vLLM 仓库内运行")


@pytest.fixture
def awq_config():
    """构造最小可用的 :class:`AWQConfig`（4bit, g128, 非 zero_point 简化）。"""
    from vllm.model_executor.layers.quantization.awq import AWQConfig

    return AWQConfig(
        weight_bits=4,
        group_size=128,
        zero_point=True,
        modules_to_not_convert=None,
    )


class TestAwqConfigSupportedActDtypes:
    """覆盖 CPU / 非 CPU 平台下的 dtype 放行策略。"""

    def test_cpu_platform_allows_bfloat16(self, awq_config, monkeypatch) -> None:
        """CPU 平台必须同时允许 fp16 与 bf16。"""
        # Arrange：打桩 current_platform.is_cpu() → True
        import vllm.model_executor.layers.quantization.awq as awq_mod

        monkeypatch.setattr(
            awq_mod.current_platform, "is_cpu", lambda: True,
        )

        # Act
        dtypes = awq_config.get_supported_act_dtypes()

        # Assert
        assert torch.half in dtypes, f"fp16 应始终被支持，实际 {dtypes}"
        assert torch.bfloat16 in dtypes, (
            f"CPU 平台必须支持 bf16 以匹配 R1-AWQ + --dtype=bfloat16，实际 {dtypes}"
        )

    def test_non_cpu_platform_only_fp16(self, awq_config, monkeypatch) -> None:
        """GPU / 其他平台必须保持原有 fp16-only 契约。"""
        # Arrange
        import vllm.model_executor.layers.quantization.awq as awq_mod

        monkeypatch.setattr(
            awq_mod.current_platform, "is_cpu", lambda: False,
        )

        # Act
        dtypes = awq_config.get_supported_act_dtypes()

        # Assert
        assert dtypes == [torch.half], (
            f"非 CPU 平台应仅返回 [torch.half]，实际 {dtypes}"
        )
