# -*- coding: utf-8 -*-
"""dispatch_cpu_unquantized_gemm ACL 路径集成测试。

通过 mock 模拟不同的平台环境，验证 ACL 后端在 dispatch 中的接入逻辑：
- ACL 可用时优先使用 ACL 后端
- remove_weight=True 时权重被替换为空张量
- ACL 不可用时正确回退到后续后端
"""

import importlib.util
import platform
from unittest import mock

import pytest
import torch

_is_aarch64 = platform.machine() in ("aarch64", "arm64")

try:
    import fused_cpp

    _acl_available = fused_cpp._supports_acl
except ImportError:
    _acl_available = False

_vllm_available = importlib.util.find_spec("vllm") is not None

pytestmark = pytest.mark.skipif(
    not (_is_aarch64 and _acl_available and _vllm_available),
    reason="ACL dispatch 集成测试仅在 AArch64、ACL 可用且 vLLM 可导入时运行",
)


def _make_linear_layer(
    K: int,
    N: int,
    dtype: torch.dtype = torch.float32,
    with_bias: bool = False,
) -> torch.nn.Module:
    """创建一个模拟的 Linear 层，带有 weight 和可选 bias 属性。"""
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.randn(N, K, dtype=dtype), requires_grad=False)
    if with_bias:
        layer.bias = torch.nn.Parameter(torch.randn(N, dtype=dtype), requires_grad=False)
    else:
        layer.bias = None
    return layer


class TestACLDispatchIntegration:
    """测试 ACL 后端在 dispatch_cpu_unquantized_gemm 中的集成。"""

    def test_acl_backend_selected(self) -> None:
        """ACL 可用时，dispatch 应选择 ACL 后端。"""
        from vllm.model_executor.layers.utils import (
            dispatch_cpu_unquantized_gemm,
        )

        layer = _make_linear_layer(128, 256)

        # 禁用 SGL kernel 路径，确保走到 ACL 分支
        with mock.patch("vllm.envs.VLLM_CPU_SGL_KERNEL", False):
            dispatch_cpu_unquantized_gemm(layer, remove_weight=False)

        # 验证 cpu_linear 已被设置
        assert hasattr(layer, "cpu_linear")
        assert callable(layer.cpu_linear)

        # 验证 cpu_linear 能正确执行
        x = torch.randn(4, 128, dtype=torch.float32)
        result = layer.cpu_linear(x, layer.weight, layer.bias)
        assert result.shape == (4, 256)

        # 验证结果与参考实现的一致性
        expected = torch.nn.functional.linear(x, layer.weight, layer.bias)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_acl_remove_weight(self) -> None:
        """remove_weight=True 时，权重应被替换为空张量。"""
        from vllm.model_executor.layers.utils import (
            dispatch_cpu_unquantized_gemm,
        )

        layer = _make_linear_layer(128, 256)
        original_weight = layer.weight.clone()

        with mock.patch("vllm.envs.VLLM_CPU_SGL_KERNEL", False):
            dispatch_cpu_unquantized_gemm(layer, remove_weight=True)

        # 权重应被替换为空张量
        assert layer.weight.numel() == 0

        # cpu_linear 仍应能正常执行（使用预打包的权重）
        x = torch.randn(4, 128, dtype=torch.float32)
        result = layer.cpu_linear(x, layer.weight, None)
        assert result.shape == (4, 256)

        # 验证结果与参考实现的一致性（使用原始权重）
        expected = torch.nn.functional.linear(x, original_weight, None)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_acl_with_bias(self) -> None:
        """带偏置的 Linear 层应正确处理。"""
        from vllm.model_executor.layers.utils import (
            dispatch_cpu_unquantized_gemm,
        )

        layer = _make_linear_layer(128, 256, with_bias=True)

        with mock.patch("vllm.envs.VLLM_CPU_SGL_KERNEL", False):
            dispatch_cpu_unquantized_gemm(layer, remove_weight=False)

        x = torch.randn(4, 128, dtype=torch.float32)
        result = layer.cpu_linear(x, layer.weight, layer.bias)

        expected = torch.nn.functional.linear(x, layer.weight, layer.bias)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_fallback_when_acl_unavailable(self) -> None:
        """ACL 不可用时应回退到后续后端（oneDNN 或 torch.linear）。"""
        from vllm.model_executor.layers.utils import (
            dispatch_cpu_unquantized_gemm,
        )

        layer = _make_linear_layer(128, 256)

        # mock fused_cpp._supports_acl 为 False，模拟 ACL 不可用
        with (
            mock.patch("vllm.envs.VLLM_CPU_SGL_KERNEL", False),
            mock.patch.dict(
                "sys.modules",
                {
                    "fused_cpp": mock.MagicMock(
                        _supports_acl=False,
                    )
                },
            ),
        ):
            dispatch_cpu_unquantized_gemm(layer, remove_weight=False)

        # cpu_linear 应被设置（回退到 oneDNN 或 torch.linear）
        assert hasattr(layer, "cpu_linear")
        assert callable(layer.cpu_linear)

        # 验证仍能正常执行
        x = torch.randn(4, 128, dtype=torch.float32)
        result = layer.cpu_linear(x, layer.weight, layer.bias)
        assert result.shape == (4, 256)

    def test_meta_weight_skipped(self) -> None:
        """meta 权重应直接跳过，使用 torch.nn.functional.linear。"""
        from vllm.model_executor.layers.utils import (
            dispatch_cpu_unquantized_gemm,
        )

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.empty(256, 128, device="meta"), requires_grad=False)
        layer.bias = None

        dispatch_cpu_unquantized_gemm(layer, remove_weight=False)

        assert hasattr(layer, "cpu_linear")
        # meta 权重时 cpu_linear 应为 torch.nn.functional.linear
        assert layer.cpu_linear is torch.nn.functional.linear
