# -*- coding: utf-8 -*-
"""W4A8 Linear 算子等价性测试（AWQ 权重布局）。

参考实现：将 AWQ 打包的 ``qweight`` / ``qzeros`` / ``scales`` 解包并反量化为全
精度权重，再以 ``torch.matmul`` 计算。本测试验证 per-token int8 GEMM 路径在容
差内逼近该参考实现，并检查 NaN / Inf 鲁棒性与余弦相似度。

AWQ 布局权威参考：
- https://github.com/casper-hansen/AutoAWQ/blob/main/awq/utils/packing_utils.py
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/awq.py
"""

from __future__ import annotations

import pytest
import torch

import fused_cpp
from fused_cpp.w4a8_linear import (
    AWQ_ORDER,
    unpack_awq_qweight,
    unpack_awq_qzeros,
)


# ── 容差表 ──
# w4a8_linear 在参考 bf16 反量化路径上额外引入了「per-token int8 激活量化」噪声，
# 其理论最大绝对误差量级约为 ``|w| * max|x| / 127 * sqrt(K)``；对 K=128/256、bf16
# 参考输出量级 ~ O(1) 的矩阵而言，max_abs 常落在 2e-2 ~ 5e-2。因此相对 python.md
# 第 8.2 节 bf16 基础容差（1e-2 / 1e-2）放宽一档，余弦相似度阈值保持严格。
_TOL_W4A8 = dict(rtol=5e-2, atol=5e-2, cos_sim_threshold=0.999)


# ── 辅助函数 ──


def _assert_tensor_close(
    actual: torch.Tensor,
    ref: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    cos_sim_threshold: float,
) -> None:
    """统一等价性断言：误差 + 余弦相似度 + NaN / Inf 鲁棒性。"""
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


def _pack_awq_along_n(unpacked: torch.Tensor) -> torch.Tensor:
    """把 ``[..., N] int32`` (值域 0..15) 按 AWQ interleave 顺序沿 N 方向 pack 成 int32。

    输入最后一维为 N（要求 N % 8 == 0），输出形状为 ``[..., N // 8]``。
    pack 规则：第 i 个 nibble（bits ``[4 * i, 4 * i + 4)``）来源于 N 维第
    ``AWQ_ORDER[i]`` 列。
    """
    assert unpacked.shape[-1] % 8 == 0
    lead = unpacked.shape[:-1]
    n = unpacked.shape[-1]
    reshaped = unpacked.reshape(*lead, n // 8, 8).to(torch.int32) & 0xF

    # 依 AWQ_ORDER 从 N 方向取 8 个值，分别占据 nibble 0..7
    order_idx = torch.tensor(AWQ_ORDER, dtype=torch.long)
    picked = reshaped.index_select(-1, order_idx)  # [..., N//8, 8]

    shifts = torch.arange(0, 32, 4, dtype=torch.int32).view(
        *([1] * len(lead)),
        1,
        8,
    )
    return (picked << shifts).sum(dim=-1).to(torch.int32)


def _build_random_awq_weights(
    k: int,
    n: int,
    group_size: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造随机的 (qweight, qzeros, scales, w_fp_ref) 四元组（AWQ 布局）。

    其中 ``w_fp_ref`` 是对应的 float32 反量化参考权重，shape ``[K, N]``。
    """
    gen = torch.Generator().manual_seed(seed)

    # 原始 4-bit 权重 / 零点，值域 0..15
    w_int4 = torch.randint(0, 16, (k, n), generator=gen, dtype=torch.int32)
    groups = k // group_size
    z_int4 = torch.randint(0, 16, (groups, n), generator=gen, dtype=torch.int32)
    scales = (torch.rand((groups, n), generator=gen, dtype=torch.float32) * 0.02 + 0.001).to(torch.float16)

    qweight = _pack_awq_along_n(w_int4)  # [K, N//8] int32
    qzeros = _pack_awq_along_n(z_int4)  # [G, N//8] int32

    # 参考反量化权重：w_fp[k, n] = (w_int4[k, n] - z_int4[k//g, n]) * scales[k//g, n]
    z_expanded = z_int4.repeat_interleave(group_size, dim=0)  # [K, N]
    s_expanded = scales.to(torch.float32).repeat_interleave(group_size, dim=0)
    w_fp = (w_int4.to(torch.float32) - z_expanded.to(torch.float32)) * s_expanded

    return qweight, qzeros, scales, w_fp


def _reference_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """参考实现：完整反量化 + fp32 matmul + 截断到 bf16。"""
    groups, _ = qzeros.shape
    k = qweight.shape[0]
    group_size = k // groups

    w_q = unpack_awq_qweight(qweight).to(torch.float32)  # [K, N]
    w_z = unpack_awq_qzeros(qzeros).to(torch.float32)  # [G, N]
    w_z_expanded = w_z.repeat_interleave(group_size, dim=0)  # [K, N]
    s_expanded = scales.to(torch.float32).repeat_interleave(group_size, dim=0)
    w_fp = (w_q - w_z_expanded) * s_expanded  # [K, N]

    out = torch.matmul(x.to(torch.float32), w_fp)
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(torch.bfloat16)


# ── 解包单元测试 ──


class TestUnpack:
    """qweight / qzeros 解包工具函数自洽性。"""

    def test_unpack_qweight_roundtrip(self) -> None:
        """AWQ pack → unpack 应还原原始 4-bit 值。"""
        torch.manual_seed(0)
        k, n = 32, 16
        w_int4 = torch.randint(0, 16, (k, n), dtype=torch.int32)
        qweight = _pack_awq_along_n(w_int4)
        assert qweight.shape == (k, n // 8)
        restored = unpack_awq_qweight(qweight).to(torch.int32)
        assert torch.equal(restored, w_int4)

    def test_unpack_qzeros_roundtrip(self) -> None:
        """AWQ qzeros 沿 N interleaved pack 的往返。"""
        torch.manual_seed(0)
        groups, n = 4, 32
        z_int4 = torch.randint(0, 16, (groups, n), dtype=torch.int32)
        qzeros = _pack_awq_along_n(z_int4)
        assert qzeros.shape == (groups, n // 8)
        restored = unpack_awq_qzeros(qzeros).to(torch.int32)
        assert torch.equal(restored, z_int4)

    def test_unpack_respects_awq_order(self) -> None:
        """直接构造一个已知 bit-pattern，验证 interleave 顺序为 AWQ_ORDER。"""
        # 构造单个 int32：nibble_i = i （0..7）
        # 则 packed = 0x76543210
        packed = torch.tensor([[0x76543210]], dtype=torch.int32)
        # 解包后沿 N 的第 AWQ_ORDER[i] 列应为 i
        # 即 unpacked[0, AWQ_ORDER[i]] == i  →  unpacked[0, :] 为
        # [0, 4, 1, 5, 2, 6, 3, 7]（因为 AWQ_ORDER = 0,2,4,6,1,3,5,7）
        unpacked = unpack_awq_qweight(packed).to(torch.int32)
        expected = torch.tensor(
            [[0, 4, 1, 5, 2, 6, 3, 7]],
            dtype=torch.int32,
        )
        assert torch.equal(unpacked, expected)


# ── W4A8 linear 正确性 ──


class TestW4A8LinearCorrectness:
    """w4a8_linear 与完整反量化参考实现的等价性。"""

    @pytest.mark.parametrize(
        "shape",
        [(1, 64, 32), (4, 128, 64), (16, 256, 128)],
        ids=["tiny", "mid", "large"],
    )
    @pytest.mark.parametrize("group_size", [32, 64], ids=["g32", "g64"])
    def test_matches_reference(
        self,
        shape: tuple[int, int, int],
        group_size: int,
    ) -> None:
        """多种 shape × group_size 下与参考实现一致。"""
        # Arrange
        m, k, n = shape
        if k % group_size != 0:
            pytest.skip(f"K={k} 不能被 group_size={group_size} 整除")

        torch.manual_seed(0)
        qweight, qzeros, scales, _ = _build_random_awq_weights(
            k,
            n,
            group_size,
            seed=0,
        )
        x = torch.randn(m, k, dtype=torch.bfloat16)

        # Act
        out = fused_cpp.w4a8_linear(x, qweight, qzeros, scales, bias=None)
        ref = _reference_linear(x, qweight, qzeros, scales, bias=None)

        # Assert
        _assert_tensor_close(out, ref, **_TOL_W4A8)

    def test_with_bias(self) -> None:
        """带 bias 的计算应与参考实现一致。"""
        # Arrange
        m, k, n, group_size = 8, 128, 64, 32
        torch.manual_seed(1)
        qweight, qzeros, scales, _ = _build_random_awq_weights(
            k,
            n,
            group_size,
            seed=1,
        )
        x = torch.randn(m, k, dtype=torch.bfloat16)
        bias = torch.randn(n, dtype=torch.bfloat16)

        # Act
        out = fused_cpp.w4a8_linear(x, qweight, qzeros, scales, bias=bias)
        ref = _reference_linear(x, qweight, qzeros, scales, bias=bias)

        # Assert
        _assert_tensor_close(out, ref, **_TOL_W4A8)

    def test_3d_input(self) -> None:
        """3D 输入（batch × seq × hidden）应保留前导维度。"""
        # Arrange
        b, s, k, n, group_size = 2, 16, 128, 64, 32
        torch.manual_seed(2)
        qweight, qzeros, scales, _ = _build_random_awq_weights(
            k,
            n,
            group_size,
            seed=2,
        )
        x = torch.randn(b, s, k, dtype=torch.bfloat16)

        # Act
        out = fused_cpp.w4a8_linear(x, qweight, qzeros, scales, bias=None)
        ref = _reference_linear(x, qweight, qzeros, scales, bias=None)

        # Assert
        assert out.shape == (b, s, n)
        _assert_tensor_close(out, ref, **_TOL_W4A8)

    def test_empty_input(self) -> None:
        """M=0 空输入应返回空张量，而非报错。"""
        k, n, group_size = 64, 32, 32
        qweight, qzeros, scales, _ = _build_random_awq_weights(
            k,
            n,
            group_size,
            seed=3,
        )
        x = torch.empty(0, k, dtype=torch.bfloat16)
        out = fused_cpp.w4a8_linear(x, qweight, qzeros, scales, bias=None)
        assert out.shape == (0, n)
        assert out.dtype == torch.bfloat16


# ── 输入校验 ──


class TestW4A8LinearValidation:
    """非法输入应抛出清晰的 RuntimeError。"""

    def _dummy_weights(
        self,
        k: int = 32,
        n: int = 16,
        group_size: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _build_random_awq_weights(k, n, group_size, seed=0)[:3]

    def test_wrong_x_dtype(self) -> None:
        qweight, qzeros, scales = self._dummy_weights()
        x = torch.randn(2, 32, dtype=torch.float16)
        with pytest.raises(RuntimeError, match="bfloat16"):
            fused_cpp.w4a8_linear(x, qweight, qzeros, scales)

    def test_wrong_qweight_dtype(self) -> None:
        qweight, qzeros, scales = self._dummy_weights()
        x = torch.randn(2, 32, dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="qweight"):
            fused_cpp.w4a8_linear(x, qweight.to(torch.int64), qzeros, scales)

    def test_wrong_k(self) -> None:
        qweight, qzeros, scales = self._dummy_weights()
        x = torch.randn(2, 64, dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="K"):
            fused_cpp.w4a8_linear(x, qweight, qzeros, scales)
