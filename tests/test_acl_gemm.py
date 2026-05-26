# -*- coding: utf-8 -*-
"""ACL GEMM 后端单元测试。

测试 handler 创建/销毁、GEMM 计算正确性、多种输入形状、
bias 有/无、非连续输入、不支持的 dtype/维度等场景。
"""
import platform

import pytest
import torch

# 在非 AArch64 平台或 ACL 不可用时跳过全部测试
_is_aarch64 = platform.machine() in ("aarch64", "arm64")

try:
    import fused_cpp
    _acl_available = fused_cpp._supports_acl
except ImportError:
    _acl_available = False

pytestmark = pytest.mark.skipif(
    not (_is_aarch64 and _acl_available),
    reason="ACL GEMM 仅在 AArch64 平台且 ACL 可用时测试",
)


# ── 辅助函数 ──

def _reference_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """使用 torch.nn.functional.linear 作为参考实现。"""
    return torch.nn.functional.linear(x, weight, bias)


# ── Handler 创建与销毁 ──

class TestACLGEMMHandlerLifecycle:
    """测试 handler 的创建和销毁。"""

    @pytest.mark.parametrize("fast_math", [False, True], ids=["precise", "fast"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_create_and_release(self, dtype: torch.dtype,
                                fast_math: bool) -> None:
        """handler 创建后应持有有效指针，销毁后资源应被释放。"""
        K, N = 128, 256
        weight = torch.randn(K, N, dtype=dtype)
        handler = fused_cpp.create_acl_gemm(weight, fast_math=fast_math)

        assert isinstance(handler, fused_cpp.ACLGEMMHandler)
        assert handler.k == K
        assert handler.n == N
        assert handler._handler_ptr != 0
        assert handler.fast_math == fast_math

        # 显式删除，不应抛出异常
        del handler

    def test_create_invalid_dim(self) -> None:
        """3D 权重应抛出异常。"""
        weight = torch.randn(2, 3, 4, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="2D"):
            fused_cpp.create_acl_gemm(weight)

    def test_create_invalid_dtype(self) -> None:
        """不支持的 dtype（如 float16）应抛出异常。"""
        weight = torch.randn(128, 256).to(torch.float16)
        with pytest.raises(RuntimeError):
            fused_cpp.create_acl_gemm(weight)


# ── GEMM 计算正确性 ──

class TestACLGEMMCorrectness:
    """测试 GEMM 计算结果与 torch.nn.functional.linear 的一致性。"""

    @pytest.mark.parametrize("fast_math", [False, True], ids=["precise", "fast"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize(
        "M,K,N",
        [
            (1, 128, 256),
            (4, 128, 256),
            (32, 512, 1024),
            (1, 768, 5120),
        ],
    )
    def test_gemm_2d_no_bias(
        self, M: int, K: int, N: int, dtype: torch.dtype,
        fast_math: bool,
    ) -> None:
        """2D 输入、无偏置的 GEMM 正确性。"""
        weight = torch.randn(K, N, dtype=dtype)
        x = torch.randn(M, K, dtype=dtype)

        handler = fused_cpp.create_acl_gemm(weight, fast_math=fast_math)
        result = fused_cpp.acl_gemm(handler, x, None)

        # 参考实现：F.linear(x, weight.T) = x @ weight
        # 注意 ACL 计算的是 x @ W^T，而 weight 形状为 [K, N]
        # 所以 F.linear 需要 weight 转置为 [N, K]
        expected = _reference_linear(x, weight.t())

        assert result.shape == expected.shape
        # fast_math 下 FP32 使用 BF16 中间精度，容差需放宽
        if fast_math:
            atol = 1e-1 if dtype == torch.bfloat16 else 5e-2
            rtol = 1e-1 if dtype == torch.bfloat16 else 5e-2
        else:
            atol = 1e-2 if dtype == torch.bfloat16 else 1e-4
            rtol = 1e-2 if dtype == torch.bfloat16 else 1e-4
        torch.testing.assert_close(result, expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("fast_math", [False, True], ids=["precise", "fast"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_gemm_2d_with_bias(self, dtype: torch.dtype,
                                fast_math: bool) -> None:
        """2D 输入、有偏置的 GEMM 正确性。"""
        M, K, N = 8, 128, 256
        weight = torch.randn(K, N, dtype=dtype)
        x = torch.randn(M, K, dtype=dtype)
        bias = torch.randn(N, dtype=dtype)

        handler = fused_cpp.create_acl_gemm(weight, fast_math=fast_math)
        result = fused_cpp.acl_gemm(handler, x, bias)

        expected = _reference_linear(x, weight.t(), bias)

        if fast_math:
            atol = 1e-1 if dtype == torch.bfloat16 else 5e-2
            rtol = 1e-1 if dtype == torch.bfloat16 else 5e-2
        else:
            atol = 1e-2 if dtype == torch.bfloat16 else 1e-4
            rtol = 1e-2 if dtype == torch.bfloat16 else 1e-4
        torch.testing.assert_close(result, expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("fast_math", [False, True], ids=["precise", "fast"])
    def test_gemm_3d_input(self, fast_math: bool) -> None:
        """3D 输入 [B, M, K] 应正确 reshape 并返回 [B, M, N]。"""
        B, M, K, N = 2, 4, 128, 256
        weight = torch.randn(K, N, dtype=torch.float32)
        x = torch.randn(B, M, K, dtype=torch.float32)

        handler = fused_cpp.create_acl_gemm(weight, fast_math=fast_math)
        result = fused_cpp.acl_gemm(handler, x, None)

        expected = _reference_linear(x, weight.t())

        assert result.shape == (B, M, N)
        atol = 5e-2 if fast_math else 1e-4
        rtol = 5e-2 if fast_math else 1e-4
        torch.testing.assert_close(result, expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("fast_math", [False, True], ids=["precise", "fast"])
    def test_gemm_m_zero(self, fast_math: bool) -> None:
        """M=0 边界情况应返回空张量。"""
        K, N = 128, 256
        weight = torch.randn(K, N, dtype=torch.float32)
        x = torch.randn(0, K, dtype=torch.float32)

        handler = fused_cpp.create_acl_gemm(weight, fast_math=fast_math)
        result = fused_cpp.acl_gemm(handler, x, None)

        assert result.shape == (0, N)
        assert result.numel() == 0

    @pytest.mark.parametrize("fast_math", [False, True], ids=["precise", "fast"])
    def test_gemm_non_contiguous_input(self, fast_math: bool) -> None:
        """非连续输入张量应被自动转为连续后正确计算。"""
        M, K, N = 8, 128, 256
        weight = torch.randn(K, N, dtype=torch.float32)
        # 创建非连续张量：通过转置再 slice
        x_full = torch.randn(K, M, dtype=torch.float32).t()
        assert not x_full.is_contiguous()

        handler = fused_cpp.create_acl_gemm(weight, fast_math=fast_math)
        result = fused_cpp.acl_gemm(handler, x_full, None)

        expected = _reference_linear(x_full.contiguous(), weight.t())
        atol = 5e-2 if fast_math else 1e-4
        rtol = 5e-2 if fast_math else 1e-4
        torch.testing.assert_close(result, expected, atol=atol, rtol=rtol)
