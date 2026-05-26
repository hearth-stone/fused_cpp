#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AWQ MoE expert FFN 参考实现的等价性测试（Step 1 / 方案 B）。

测试目标（与 `fused_cpp/src/fused_cpp/moe/awq_moe.py` 对齐）：

1. :func:`fused_cpp.awq_moe_expert_ffn_reference` 本身的正确性
   —— 通过"单 expert / top_k=1 / 仅一个 token"退化到
   :func:`fused_cpp.w4a8_linear` 的 dequant-only 等价路径来交叉验证。
2. :func:`fused_cpp.awq_moe_expert_ffn_w4a8` 与 reference 在容差内一致（R1 级
   `num_experts=16 / top_k=8 / H=7168 / F=2048 / g=64` 的真实 shape 子采样）。
3. 路由"恒等情形"（每 token 只选 1 个 expert，topk_weights=1）下，MoE 的输出
   应退化为对应 expert 的稠密 FFN 结果。
4. 数值鲁棒性：输出不含 NaN / Inf。

约束（对齐项目测试规则 §11）：

- 参考实现独立于被测实现；本模块的"参考"即 :func:`awq_moe_expert_ffn_reference`
  （AWQ 解包到 bf16 + 朴素 matmul），与被测 :func:`awq_moe_expert_ffn_w4a8`
  **不共享**任何数值通路（前者不走 w4a8_linear，后者逐 expert 调用之）。
- 所有张量 seed 固定，`assert_tensor_close` 同时报告 max_abs / max_rel / cos。
- 不引入 vLLM 高层 API；仅依赖 PyTorch 与 fused_cpp 自身。
"""
from __future__ import annotations

import pytest
import torch

from fused_cpp.moe.awq_moe import (
    AWQExpertWeights,
    awq_moe_expert_ffn_reference,
    awq_moe_expert_ffn_w4a8,
    dequant_awq_to_bf16,
)

# ── 容差 ──
# `awq_moe_expert_ffn_w4a8` 引入 per-token int8 激活量化（两级：gate/up 与 down），
# 相对 reference 的误差主要来自此。与 tests/test_awq_fused_cpp_integration.py 中
# `_TOL_W4A8` 一致，余弦相似度保持严格。
_TOL_W4A8_MOE = dict(rtol=5e-2, atol=5e-2, cos_sim_threshold=0.999)

# Dequant 等价性（reference 与其自身的退化对比）严格到 bf16 native 容差。
_TOL_BF16_STRICT = dict(rtol=1e-2, atol=1e-2, cos_sim_threshold=0.9995)


# AWQ pack / unpack 工具（与 tests/test_awq_fused_cpp_integration.py 同风格）

_AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def _pack_awq_along_n(unpacked: torch.Tensor) -> torch.Tensor:
    """沿 N 维按 AWQ interleave 顺序将 int4 (0..15) pack 为 int32。"""
    assert unpacked.shape[-1] % 8 == 0
    lead = unpacked.shape[:-1]
    n = unpacked.shape[-1]
    reshaped = unpacked.reshape(*lead, n // 8, 8).to(torch.int32) & 0xF
    order_idx = torch.tensor(_AWQ_ORDER, dtype=torch.long)
    picked = reshaped.index_select(-1, order_idx)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32).view(
        *([1] * len(lead)), 1, 8,
    )
    return (picked << shifts).sum(dim=-1).to(torch.int32)


def _make_awq_weight(
    k: int,
    n: int,
    group_size: int,
    scales_dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造一组随机 (qweight, qzeros, scales)，shape 与 AWQ 约定一致。"""
    gen = torch.Generator().manual_seed(seed)
    w_int4 = torch.randint(0, 16, (k, n), generator=gen, dtype=torch.int32)
    groups = k // group_size
    z_int4 = torch.randint(0, 16, (groups, n), generator=gen, dtype=torch.int32)
    # 让 scales 幅值与 R1 接近（~1e-3），避免过大幅度触发激活 outlier
    scales_fp32 = (
        torch.rand((groups, n), generator=gen, dtype=torch.float32) * 5e-3 + 1e-3
    )
    qweight = _pack_awq_along_n(w_int4)
    qzeros = _pack_awq_along_n(z_int4)
    return qweight, qzeros, scales_fp32.to(scales_dtype)


def _make_expert(
    hidden_size: int,
    ffn_hidden: int,
    group_size: int,
    scales_dtype: torch.dtype,
    seed: int,
) -> AWQExpertWeights:
    """构造单个 AWQ expert（gate / up / down 三个 Linear）。"""
    gate_qw, gate_qz, gate_s = _make_awq_weight(
        hidden_size, ffn_hidden, group_size, scales_dtype, seed,
    )
    up_qw, up_qz, up_s = _make_awq_weight(
        hidden_size, ffn_hidden, group_size, scales_dtype, seed + 1,
    )
    down_qw, down_qz, down_s = _make_awq_weight(
        ffn_hidden, hidden_size, group_size, scales_dtype, seed + 2,
    )
    return AWQExpertWeights(
        gate_qweight=gate_qw,
        gate_qzeros=gate_qz,
        gate_scales=gate_s,
        up_qweight=up_qw,
        up_qzeros=up_qz,
        up_scales=up_s,
        down_qweight=down_qw,
        down_qzeros=down_qz,
        down_scales=down_s,
    )


def _random_topk(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成随机但合法的 (topk_ids, topk_weights)。

    每行 ``top_k`` 个 **互不相同**的 expert id，权重做 softmax 归一化以贴近
    R1 实际运行时的数值范围（top_k=8, renormalize=True）。
    """
    gen = torch.Generator().manual_seed(seed)
    topk_ids = torch.stack(
        [torch.randperm(num_experts, generator=gen)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int64)
    raw = torch.randn((num_tokens, top_k), generator=gen, dtype=torch.float32)
    topk_weights = torch.softmax(raw, dim=-1)
    return topk_ids, topk_weights


def _assert_tensor_close(
    actual: torch.Tensor,
    ref: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    cos_sim_threshold: float,
) -> None:
    """统一等价性断言：误差 + 余弦相似度 + NaN/Inf 鲁棒性。"""
    assert actual.shape == ref.shape, (
        f"shape mismatch: {actual.shape} vs {ref.shape}"
    )
    assert actual.dtype == ref.dtype, (
        f"dtype mismatch: {actual.dtype} vs {ref.dtype}"
    )
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


# ── 1. dequant_awq_to_bf16 的自洽性 ──

class TestDequantAwqToBf16:
    """``dequant_awq_to_bf16`` 与 :func:`fused_cpp.w4a8_linear` 的数值一致性。"""

    @pytest.mark.parametrize(
        "scales_dtype",
        [torch.float16, torch.bfloat16],
        ids=["fp16_scales", "bf16_scales"],
    )
    def test_dequant_matches_w4a8_linear_on_identity_activation(
        self,
        scales_dtype: torch.dtype,
    ) -> None:
        """反量化后的权重作 bf16 matmul，与 w4a8_linear 在小幅激活下一致。

        注：本测试的目的是佐证 dequant 的**权重语义**与 `w4a8_linear` 一致；
        受 per-token 激活量化影响，容差设为 _TOL_W4A8_MOE。
        """
        # Arrange
        from fused_cpp import w4a8_linear  # 局部 import 避免模块级依赖

        k, n, group_size = 128, 64, 32
        qweight, qzeros, scales = _make_awq_weight(
            k, n, group_size, scales_dtype, seed=0,
        )
        torch.manual_seed(0)
        # 用小幅激活减少 int8 激活量化截断噪声
        x = torch.randn(4, k, dtype=torch.bfloat16) * 0.1

        # Act
        w_bf16 = dequant_awq_to_bf16(qweight, qzeros, scales)
        ref = torch.matmul(x.float(), w_bf16.float()).to(torch.bfloat16)
        out = w4a8_linear(x, qweight, qzeros, scales, bias=None)

        # Assert
        _assert_tensor_close(out, ref, **_TOL_W4A8_MOE)


# ── 2. MoE reference vs w4a8 的端到端等价性 ──

class TestAwqMoeEquivalence:
    """``awq_moe_expert_ffn_w4a8`` 与 reference 在容差内一致。"""

    @pytest.mark.parametrize(
        "scales_dtype",
        [torch.float16, torch.bfloat16],
        ids=["fp16_scales", "bf16_scales"],
    )
    @pytest.mark.parametrize(
        "scenario",
        [
            # (num_tokens, num_experts, top_k, hidden, ffn, group_size)
            (4, 4, 2, 128, 64, 32),         # 最小矩阵
            (8, 8, 4, 256, 128, 32),        # 中等
            (16, 16, 8, 1024, 256, 64),     # 接近 R1 拓扑（num_experts 子采样）
        ],
        ids=["tiny", "mid", "r1-ish"],
    )
    def test_w4a8_matches_reference(
        self,
        scales_dtype: torch.dtype,
        scenario: tuple[int, int, int, int, int, int],
    ) -> None:
        """W4A8 MoE 与 reference MoE 的数值误差在 W4A8 噪声容差内。"""
        # Arrange
        num_tokens, num_experts, top_k, h, f_dim, g = scenario
        torch.manual_seed(0)
        experts = [
            _make_expert(h, f_dim, g, scales_dtype, seed=e * 7)
            for e in range(num_experts)
        ]
        hidden = torch.randn(num_tokens, h, dtype=torch.bfloat16) * 0.1
        topk_ids, topk_w = _random_topk(
            num_tokens, num_experts, top_k, seed=123,
        )

        # Act
        ref = awq_moe_expert_ffn_reference(hidden, topk_ids, topk_w, experts)
        out = awq_moe_expert_ffn_w4a8(hidden, topk_ids, topk_w, experts)

        # Assert
        _assert_tensor_close(out, ref, **_TOL_W4A8_MOE)


# ── 3. 路由退化情形：top_k=1 + weight=1 → 对应 expert 的稠密 FFN ──

class TestRoutingDegenerate:
    """路由退化为恒等时，MoE 输出应精确等于对应 expert 的朴素 FFN 输出。"""

    def test_reference_equals_dense_expert_when_top1_weight1(self) -> None:
        """每 token 只选 1 个 expert，且权重为 1.0 → reference == 朴素 FFN。"""
        # Arrange
        import torch.nn.functional as F

        h, f_dim, g = 128, 64, 32
        num_experts = 3
        num_tokens = 5
        torch.manual_seed(0)
        experts = [
            _make_expert(h, f_dim, g, torch.bfloat16, seed=e * 11)
            for e in range(num_experts)
        ]

        # 每 token 选 expert = token_idx % num_experts
        topk_ids = torch.arange(num_tokens, dtype=torch.int64).remainder(
            num_experts,
        ).unsqueeze(-1)                                             # [T, 1]
        topk_w = torch.ones((num_tokens, 1), dtype=torch.float32)
        hidden = torch.randn(num_tokens, h, dtype=torch.bfloat16) * 0.1

        # Act
        moe_out = awq_moe_expert_ffn_reference(hidden, topk_ids, topk_w, experts)

        # 参考：每 token 直接走对应 expert 的朴素 bf16 FFN
        expected = torch.zeros_like(hidden, dtype=torch.float32)
        for t in range(num_tokens):
            e = int(topk_ids[t, 0].item())
            w = experts[e]
            gate_w = dequant_awq_to_bf16(w.gate_qweight, w.gate_qzeros, w.gate_scales)
            up_w = dequant_awq_to_bf16(w.up_qweight, w.up_qzeros, w.up_scales)
            down_w = dequant_awq_to_bf16(w.down_qweight, w.down_qzeros, w.down_scales)
            x_t = hidden[t:t + 1].float()
            gate_out = x_t @ gate_w.float()
            up_out = x_t @ up_w.float()
            inter = F.silu(gate_out) * up_out
            expected[t:t + 1] = inter @ down_w.float()
        ref = expected.to(torch.bfloat16)

        # Assert
        _assert_tensor_close(moe_out, ref, **_TOL_BF16_STRICT)


# ── 4. R1 真实 shape 冒烟 ──

def test_r1_real_shape_smoke() -> None:
    """R1 真实 shape 子采样（8 个 expert）下，w4a8 与 reference 在容差内一致。

    - hidden_size    = 7168   （R1 真实值）
    - ffn_hidden     = 2048   （R1 moe_intermediate_size）
    - group_size     = 64     （R1 awq group_size）
    - num_experts    = 8      （真实 R1 是 256，此处子采样以控制测试时长）
    - top_k          = 4      （真实 R1 是 8；此处按比例缩放）
    - num_tokens     = 2      （decode 级 batch）
    """
    # Arrange
    h, f_dim, g = 7168, 2048, 64
    num_experts, top_k, num_tokens = 8, 4, 2
    torch.manual_seed(0)
    experts = [
        _make_expert(h, f_dim, g, torch.float16, seed=e * 13)
        for e in range(num_experts)
    ]
    hidden = torch.randn(num_tokens, h, dtype=torch.bfloat16) * 0.05
    topk_ids, topk_w = _random_topk(num_tokens, num_experts, top_k, seed=7)

    # Act
    ref = awq_moe_expert_ffn_reference(hidden, topk_ids, topk_w, experts)
    out = awq_moe_expert_ffn_w4a8(hidden, topk_ids, topk_w, experts)

    # Assert
    assert out.shape == (num_tokens, h)
    assert out.dtype == torch.bfloat16
    _assert_tensor_close(out, ref, **_TOL_W4A8_MOE)
