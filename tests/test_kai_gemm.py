# -*- coding: utf-8 -*-
"""KleidiAI GEMM 后端单元测试。

覆盖：
    - ``kai_gemm_prepare`` 的形状/dtype 校验与 bias 路径。
    - ``KAIGEMMHandler`` 生命周期与不同 ``max_threads`` 组合。
    - GEMM 计算在多组 ``(M, K, N)`` 下的正确性（FP32/BF16 输出）。
    - 多线程与单线程结果一致性。
    - 边界情况：M=0、非 contiguous、非对齐尺寸、3D 输入。

非 AArch64 或 KleidiAI 后端不可用时，整个模块被 ``pytestmark`` 跳过。
"""
import platform

import pytest
import torch

_is_aarch64 = platform.machine() in ("aarch64", "arm64")

try:
    import fused_cpp
    _kai_available = getattr(fused_cpp, "_supports_kai", False)
except ImportError:
    _kai_available = False

pytestmark = pytest.mark.skipif(
    not (_is_aarch64 and _kai_available),
    reason="KleidiAI GEMM 仅在 AArch64 平台且 KleidiAI 后端可用时测试",
)


# ── 辅助函数 ──

def _reference_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """参考实现：``out = x @ weight + bias``，以 FP32 累加后转到 ``out_dtype``。

    ``weight`` 形状为 ``[K, N]``，与 KAI GEMM 约定一致。为了模拟 BF16 中间精度，
    会先把输入转换到 BF16，再在 BF16 下完成矩阵乘法后升回 FP32，再应用 bias。
    """
    # 模拟 BF16 量化：KAI 内部将 LHS/RHS 都量化为 BF16 后用 BFMMLA 计算。
    x_bf16 = x.to(torch.bfloat16).to(torch.float32)
    w_bf16 = weight.to(torch.bfloat16).to(torch.float32)
    out = x_bf16 @ w_bf16
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(out_dtype)


def _build_handler(
    k: int,
    n: int,
    bias: torch.Tensor = None,
    max_threads: int = 1,
    weight: torch.Tensor = None,
) -> "fused_cpp.KAIGEMMHandler":
    """快捷构造 KAIGEMMHandler，并保留 weight 引用供对比。"""
    if weight is None:
        weight = torch.randn(k, n, dtype=torch.float32)
    packed, pk, pn = fused_cpp.kai_gemm_prepare(weight, bias)
    handler = fused_cpp.create_kai_gemm(packed, pk, pn, max_threads=max_threads)
    return handler, weight


# ── Prepare 阶段 ──

class TestKAIPrepare:
    """``kai_gemm_prepare`` 的输入校验与返回值。"""

    def test_prepare_fp32_no_bias(self) -> None:
        k, n = 128, 256
        weight = torch.randn(k, n, dtype=torch.float32)
        packed, pk, pn = fused_cpp.kai_gemm_prepare(weight)

        assert packed.dtype == torch.uint8
        assert packed.dim() == 1
        assert packed.numel() > 0
        assert pk == k
        assert pn == n

    def test_prepare_fp32_with_bias(self) -> None:
        k, n = 128, 256
        weight = torch.randn(k, n, dtype=torch.float32)
        bias = torch.randn(n, dtype=torch.float32)
        packed, pk, pn = fused_cpp.kai_gemm_prepare(weight, bias)

        assert packed.dtype == torch.uint8
        assert pk == k
        assert pn == n

    def test_prepare_bf16_weight(self) -> None:
        k, n = 128, 256
        weight = torch.randn(k, n, dtype=torch.bfloat16)
        packed, pk, pn = fused_cpp.kai_gemm_prepare(weight)

        assert packed.dtype == torch.uint8
        assert pk == k
        assert pn == n

    def test_prepare_invalid_dim(self) -> None:
        weight = torch.randn(2, 3, 4, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="2D"):
            fused_cpp.kai_gemm_prepare(weight)

    def test_prepare_invalid_dtype(self) -> None:
        weight = torch.randn(128, 256, dtype=torch.float16)
        with pytest.raises(RuntimeError):
            fused_cpp.kai_gemm_prepare(weight)

    def test_prepare_non_contiguous(self) -> None:
        weight = torch.randn(256, 128, dtype=torch.float32).t()
        assert not weight.is_contiguous()
        with pytest.raises(RuntimeError, match="contiguous"):
            fused_cpp.kai_gemm_prepare(weight)

    def test_prepare_bias_wrong_len(self) -> None:
        weight = torch.randn(128, 256, dtype=torch.float32)
        bias = torch.randn(255, dtype=torch.float32)
        with pytest.raises(RuntimeError):
            fused_cpp.kai_gemm_prepare(weight, bias)

    def test_prepare_bias_wrong_dtype(self) -> None:
        weight = torch.randn(128, 256, dtype=torch.float32)
        bias = torch.randn(256, dtype=torch.bfloat16)
        with pytest.raises(RuntimeError):
            fused_cpp.kai_gemm_prepare(weight, bias)


# ── Handler 生命周期 ──

class TestKAIHandlerLifecycle:
    """handler 的创建/销毁，以及 max_threads 参数。"""

    @pytest.mark.parametrize("max_threads", [1, 2, 4])
    def test_create_and_release(self, max_threads: int) -> None:
        k, n = 128, 256
        weight = torch.randn(k, n, dtype=torch.float32)
        packed, pk, pn = fused_cpp.kai_gemm_prepare(weight)

        handler = fused_cpp.create_kai_gemm(packed, pk, pn, max_threads=max_threads)
        assert isinstance(handler, fused_cpp.KAIGEMMHandler)
        assert handler.k == k
        assert handler.n == n
        assert handler._handler_ptr != 0
        del handler  # 不应抛出异常

    def test_invalid_max_threads(self) -> None:
        k, n = 128, 256
        weight = torch.randn(k, n, dtype=torch.float32)
        packed, pk, pn = fused_cpp.kai_gemm_prepare(weight)
        with pytest.raises(RuntimeError):
            fused_cpp.create_kai_gemm(packed, pk, pn, max_threads=0)


# ── GEMM 正确性 ──

_SHAPES = [
    (1, 128, 256),
    (8, 128, 256),
    (32, 512, 1024),
    (17, 256, 384),   # 非 mr/nr 整数倍
    (64, 768, 512),
]


class TestKAIGEMMCorrectness:
    """GEMM 计算正确性。"""

    @pytest.mark.parametrize("M,K,N", _SHAPES)
    def test_gemm_fp32_no_bias(self, M: int, K: int, N: int) -> None:
        handler, weight = _build_handler(K, N)
        x = torch.randn(M, K, dtype=torch.float32)

        result = fused_cpp.kai_gemm(handler, x, output_dtype=torch.float32)
        expected = _reference_matmul(x, weight, out_dtype=torch.float32)

        assert result.shape == (M, N)
        assert result.dtype == torch.float32
        torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize("M,K,N", _SHAPES)
    def test_gemm_fp32_with_bias(self, M: int, K: int, N: int) -> None:
        bias = torch.randn(N, dtype=torch.float32)
        handler, weight = _build_handler(K, N, bias=bias)
        x = torch.randn(M, K, dtype=torch.float32)

        result = fused_cpp.kai_gemm(handler, x, output_dtype=torch.float32)
        expected = _reference_matmul(x, weight, bias=bias, out_dtype=torch.float32)

        torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize("M,K,N", _SHAPES)
    def test_gemm_bf16_output(self, M: int, K: int, N: int) -> None:
        handler, weight = _build_handler(K, N)
        x = torch.randn(M, K, dtype=torch.float32)

        result = fused_cpp.kai_gemm(handler, x, output_dtype=torch.bfloat16)
        expected = _reference_matmul(x, weight, out_dtype=torch.bfloat16)

        assert result.dtype == torch.bfloat16
        torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)

    def test_gemm_3d_input(self) -> None:
        B, M, K, N = 2, 4, 128, 256
        handler, weight = _build_handler(K, N)
        x = torch.randn(B, M, K, dtype=torch.float32)

        result = fused_cpp.kai_gemm(handler, x)
        expected = _reference_matmul(
            x.reshape(-1, K), weight, out_dtype=torch.float32,
        ).reshape(B, M, N)

        assert result.shape == (B, M, N)
        torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)

    def test_gemm_zero_matrix(self) -> None:
        M, K, N = 8, 128, 256
        weight = torch.zeros(K, N, dtype=torch.float32)
        handler, _ = _build_handler(K, N, weight=weight)
        x = torch.randn(M, K, dtype=torch.float32)

        result = fused_cpp.kai_gemm(handler, x, output_dtype=torch.float32)
        assert torch.all(result == 0)


# ── 边界情况 ──

class TestKAIEdgeCases:
    """边界输入与异常参数。"""

    def test_m_zero_returns_empty(self) -> None:
        K, N = 128, 256
        handler, _ = _build_handler(K, N)
        x = torch.empty(0, K, dtype=torch.float32)

        result = fused_cpp.kai_gemm(handler, x)
        assert result.shape == (0, N)
        assert result.numel() == 0

    def test_non_contiguous_input_auto_fixed(self) -> None:
        M, K, N = 8, 128, 256
        handler, weight = _build_handler(K, N)
        x = torch.randn(K, M, dtype=torch.float32).t()
        assert not x.is_contiguous()

        result = fused_cpp.kai_gemm(handler, x, output_dtype=torch.float32)
        expected = _reference_matmul(
            x.contiguous(), weight, out_dtype=torch.float32,
        )
        torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)

    def test_wrong_input_dim(self) -> None:
        K, N = 128, 256
        handler, _ = _build_handler(K, N)
        x = torch.randn(4, K - 1, dtype=torch.float32)
        with pytest.raises(RuntimeError):
            fused_cpp.kai_gemm(handler, x)

    def test_wrong_input_dtype(self) -> None:
        K, N = 128, 256
        handler, _ = _build_handler(K, N)
        x = torch.randn(4, K, dtype=torch.bfloat16)
        with pytest.raises(RuntimeError):
            fused_cpp.kai_gemm(handler, x)


# ── 多线程一致性 ──

class TestKAIMultiThread:
    """多线程与单线程结果一致性。"""

    @pytest.mark.parametrize("max_threads", [2, 4])
    @pytest.mark.parametrize(
        "M,K,N",
        [
            (8, 128, 256),
            (64, 512, 1024),
            (1024, 256, 512),  # 保证有足够多的 Mc 块让多个线程同时工作
        ],
    )
    def test_multi_thread_matches_single(
        self, M: int, K: int, N: int, max_threads: int,
    ) -> None:
        torch.manual_seed(0)
        weight = torch.randn(K, N, dtype=torch.float32)
        x = torch.randn(M, K, dtype=torch.float32)

        # 单线程参考（pool=None）
        packed, pk, pn = fused_cpp.kai_gemm_prepare(weight)
        handler = fused_cpp.create_kai_gemm(packed, pk, pn)
        out_st = fused_cpp.kai_gemm(handler, x, output_dtype=torch.float32)

        # 多线程：显式创建 KAIThreadPool，max_threads 即 cpu_ids 长度
        # （不指定具体 CPU ID，与单线程相同的计算语义）
        with fused_cpp.KAIThreadPool(list(range(max_threads))) as pool:
            out_mt = fused_cpp.kai_gemm(
                handler, x, output_dtype=torch.float32, pool=pool,
            )

        torch.testing.assert_close(out_mt, out_st, atol=0, rtol=0)

    def test_m_less_than_mc_uses_single_thread_path(self) -> None:
        """M 很小（只有一个 Mc 块）时多线程路径自动退化为单线程。"""
        K, N = 128, 256
        weight = torch.randn(K, N, dtype=torch.float32)
        x = torch.randn(4, K, dtype=torch.float32)

        packed, pk, pn = fused_cpp.kai_gemm_prepare(weight)
        handler = fused_cpp.create_kai_gemm(packed, pk, pn)

        out_st = fused_cpp.kai_gemm(handler, x, output_dtype=torch.float32)
        with fused_cpp.KAIThreadPool(list(range(8))) as pool:
            out_mt = fused_cpp.kai_gemm(
                handler, x, output_dtype=torch.float32, pool=pool,
            )
        torch.testing.assert_close(out_mt, out_st, atol=0, rtol=0)
