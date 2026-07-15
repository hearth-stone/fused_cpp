#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``AWQFusedMoEImpl`` 的等价性与契约测试（Step 2 / 方案 B）。

测试目标：

1. **等价性**（核心）：``AWQFusedMoEImpl.forward`` 与
   :func:`fused_cpp.awq_moe_expert_ffn_reference` 在容差内一致。
   - 参考实现：dequant→bf16 + 朴素 matmul（走 :func:`dequant_awq_to_bf16`）
   - 被测实现：``AWQFusedMoEImpl.forward`` → :func:`awq_moe_expert_ffn_w4a8`
     → :func:`fused_cpp.w4a8_linear`
   二者不共享任何 FFN 数值通路，满足项目测试规则 §11.1。

2. **堆叠权重契约**：vLLM ``FusedMoEMethodBase.create_weights`` 总是以
   ``[num_experts, ...]`` 堆叠注册权重。本测试校验 ``AWQFusedMoEImpl`` 接受
   堆叠张量后切出的 per-expert 视图与"手工构造 list[AWQExpertWeights]"走
   reference 路径的结果数值一致。

3. **形状 / dtype 校验**：契约违反应抛 ``RuntimeError`` / ``NotImplementedError``
   / ``ValueError``，而不是静默产出错误数值。

4. **路由退化**：top_k=1 + 权重 1.0 时，MoE 输出等于对应 expert 的稠密 FFN。

5. **R1 冒烟**：H=7168 / F=2048 / g=64 的真实 shape 子采样下能跑通且与
   reference 在容差内一致。

约束（对齐项目测试规则 §5 / §11）：

- 所有张量 seed 固定。
- 浮点断言使用 :func:`_assert_tensor_close`，同时报告 max_abs / max_rel / cos。
- 不引入 vLLM 高层 API；仅依赖 PyTorch + fused_cpp。
"""

from __future__ import annotations

import pytest
import torch

from fused_cpp import AWQFusedMoEImpl
from fused_cpp.moe.awq_moe import (
    AWQExpertWeights,
    awq_moe_expert_ffn_reference,
    dequant_awq_to_bf16,
)

# W4A8 激活量化引入的容差（与 test_awq_moe_vs_reference.py 一致）
_TOL_W4A8_MOE = dict(rtol=5e-2, atol=5e-2, cos_sim_threshold=0.999)
_TOL_BF16_STRICT = dict(rtol=1e-2, atol=1e-2, cos_sim_threshold=0.9995)


# ── 共享工具（与 test_awq_moe_vs_reference.py 完全一致，避免跨文件 import） ──

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
        *([1] * len(lead)),
        1,
        8,
    )
    return (picked << shifts).sum(dim=-1).to(torch.int32)


def _make_awq_weight(
    k: int,
    n: int,
    group_size: int,
    scales_dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造一组 AWQ (qweight, qzeros, scales)。"""
    gen = torch.Generator().manual_seed(seed)
    w_int4 = torch.randint(0, 16, (k, n), generator=gen, dtype=torch.int32)
    groups = k // group_size
    z_int4 = torch.randint(0, 16, (groups, n), generator=gen, dtype=torch.int32)
    scales_fp32 = torch.rand((groups, n), generator=gen, dtype=torch.float32) * 5e-3 + 1e-3
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
        hidden_size,
        ffn_hidden,
        group_size,
        scales_dtype,
        seed,
    )
    up_qw, up_qz, up_s = _make_awq_weight(
        hidden_size,
        ffn_hidden,
        group_size,
        scales_dtype,
        seed + 1,
    )
    down_qw, down_qz, down_s = _make_awq_weight(
        ffn_hidden,
        hidden_size,
        group_size,
        scales_dtype,
        seed + 2,
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


def _stack_experts(
    experts: list[AWQExpertWeights],
) -> dict[str, torch.Tensor]:
    """把 list[AWQExpertWeights] 堆叠为 ``AWQFusedMoEImpl`` 需要的 9 张量。"""
    return {
        "gate_qweight": torch.stack([e.gate_qweight for e in experts], dim=0),
        "gate_qzeros": torch.stack([e.gate_qzeros for e in experts], dim=0),
        "gate_scales": torch.stack([e.gate_scales for e in experts], dim=0),
        "up_qweight": torch.stack([e.up_qweight for e in experts], dim=0),
        "up_qzeros": torch.stack([e.up_qzeros for e in experts], dim=0),
        "up_scales": torch.stack([e.up_scales for e in experts], dim=0),
        "down_qweight": torch.stack([e.down_qweight for e in experts], dim=0),
        "down_qzeros": torch.stack([e.down_qzeros for e in experts], dim=0),
        "down_scales": torch.stack([e.down_scales for e in experts], dim=0),
    }


def _random_topk(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """随机但合法的 (topk_ids, topk_weights)；softmax 归一权重。"""
    gen = torch.Generator().manual_seed(seed)
    topk_ids = torch.stack([torch.randperm(num_experts, generator=gen)[:top_k] for _ in range(num_tokens)]).to(
        torch.int64
    )
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


# ── 1. AWQFusedMoEImpl vs reference 的核心等价性 ───────────────────────────


class TestAWQFusedMoEImplEquivalence:
    """``AWQFusedMoEImpl.forward`` 与 dequant 参考路径在容差内一致。"""

    @pytest.mark.parametrize(
        "scales_dtype",
        [torch.float16, torch.bfloat16],
        ids=["fp16_scales", "bf16_scales"],
    )
    @pytest.mark.parametrize(
        "scenario",
        [
            # (num_tokens, num_experts, top_k, hidden, ffn, group_size)
            (4, 4, 2, 128, 64, 32),
            (8, 8, 4, 256, 128, 32),
            (16, 16, 8, 1024, 256, 64),
        ],
        ids=["tiny", "mid", "r1-ish"],
    )
    def test_forward_matches_reference(
        self,
        scales_dtype: torch.dtype,
        scenario: tuple[int, int, int, int, int, int],
    ) -> None:
        """AWQFusedMoEImpl 封装不破坏 w4a8_linear 的数值正确性。"""
        # Arrange
        num_tokens, num_experts, top_k, h, f_dim, g = scenario
        torch.manual_seed(0)
        experts = [_make_expert(h, f_dim, g, scales_dtype, seed=e * 7) for e in range(num_experts)]
        stacked = _stack_experts(experts)
        hidden = torch.randn(num_tokens, h, dtype=torch.bfloat16) * 0.1
        topk_ids, topk_w = _random_topk(
            num_tokens,
            num_experts,
            top_k,
            seed=123,
        )

        impl = AWQFusedMoEImpl(
            num_experts=num_experts,
            hidden_size=h,
            ffn_hidden_size=f_dim,
            **stacked,
        )

        # Act
        ref = awq_moe_expert_ffn_reference(hidden, topk_ids, topk_w, experts)
        out = impl.forward(hidden, topk_w, topk_ids)

        # Assert
        _assert_tensor_close(out, ref, **_TOL_W4A8_MOE)


# ── 2. 堆叠张量 vs list 的数值一致性 ───────────────────────────────────────


class TestStackedWeightsSemantics:
    """堆叠张量在 __init__ 中被切成 per-expert view 后，与手工 list 一致。"""

    def test_stacked_view_matches_manual_list_routing(self) -> None:
        """``AWQFusedMoEImpl`` 内部自切 experts 与外部手工 list 对 reference
        等价路径的结果数值一致。"""
        # Arrange
        h, f_dim, g = 128, 64, 32
        num_experts, top_k, num_tokens = 6, 3, 5
        torch.manual_seed(0)
        experts = [_make_expert(h, f_dim, g, torch.float16, seed=e * 17) for e in range(num_experts)]
        stacked = _stack_experts(experts)
        hidden = torch.randn(num_tokens, h, dtype=torch.bfloat16) * 0.1
        topk_ids, topk_w = _random_topk(
            num_tokens,
            num_experts,
            top_k,
            seed=321,
        )

        impl = AWQFusedMoEImpl(
            num_experts=num_experts,
            hidden_size=h,
            ffn_hidden_size=f_dim,
            **stacked,
        )

        # Act：两条独立路径
        #   path_impl   = Impl(stacked)          → w4a8_linear
        #   path_ref    = reference(list)        → dequant + matmul
        ref = awq_moe_expert_ffn_reference(hidden, topk_ids, topk_w, experts)
        out = impl.forward(hidden, topk_w, topk_ids)

        # Assert
        _assert_tensor_close(out, ref, **_TOL_W4A8_MOE)

    def test_stacked_slice_is_zero_copy(self) -> None:
        """Impl 切出的 per-expert view 共享底层存储（避免重复内存）。"""
        # Arrange
        h, f_dim, g = 128, 64, 32
        num_experts = 3
        experts = [_make_expert(h, f_dim, g, torch.float16, seed=e) for e in range(num_experts)]
        stacked = _stack_experts(experts)

        # Act
        impl = AWQFusedMoEImpl(
            num_experts=num_experts,
            hidden_size=h,
            ffn_hidden_size=f_dim,
            **stacked,
        )

        # Assert：per-expert view 与堆叠张量共享 data_ptr（torch 基础切片零拷贝）
        for e in range(num_experts):
            assert impl._experts[e].gate_qweight.data_ptr() == (stacked["gate_qweight"][e].data_ptr()), (
                f"expert {e} gate_qweight 视图未共享存储"
            )
            assert impl._experts[e].down_scales.data_ptr() == (stacked["down_scales"][e].data_ptr()), (
                f"expert {e} down_scales 视图未共享存储"
            )


# ── 3. 契约 / 形状 / dtype 校验 ────────────────────────────────────────────


class TestContractValidation:
    """不合法的输入应显式报错，而非静默产出错误数值。"""

    @pytest.fixture
    def base_stacked(self) -> dict[str, torch.Tensor]:
        experts = [_make_expert(64, 32, 16, torch.float16, seed=e) for e in range(2)]
        return _stack_experts(experts)

    def test_ep_size_gt_one_raises_not_implemented(
        self,
        base_stacked: dict[str, torch.Tensor],
    ) -> None:
        with pytest.raises(NotImplementedError, match="ep_size=1"):
            AWQFusedMoEImpl(
                num_experts=2,
                hidden_size=64,
                ffn_hidden_size=32,
                ep_size=2,
                **base_stacked,
            )

    def test_ep_rank_out_of_range_raises(
        self,
        base_stacked: dict[str, torch.Tensor],
    ) -> None:
        # ep_size=1 下 ep_rank 必须为 0，否则应报 ValueError；
        # 若实现先触发 ep_size!=1 校验，则本用例仅验证 ep_rank 的校验存在。
        with pytest.raises((ValueError, NotImplementedError)):
            AWQFusedMoEImpl(
                num_experts=2,
                hidden_size=64,
                ffn_hidden_size=32,
                ep_size=1,
                ep_rank=3,
                **base_stacked,
            )

    def test_wrong_gate_qweight_shape_raises(
        self,
        base_stacked: dict[str, torch.Tensor],
    ) -> None:
        broken = dict(base_stacked)
        # 把 gate_qweight 的 N 维弄错
        broken["gate_qweight"] = torch.zeros(
            (2, 64, 5),
            dtype=torch.int32,
        )
        with pytest.raises(RuntimeError, match="gate_qweight"):
            AWQFusedMoEImpl(
                num_experts=2,
                hidden_size=64,
                ffn_hidden_size=32,
                **broken,
            )

    def test_wrong_scales_dtype_raises(
        self,
        base_stacked: dict[str, torch.Tensor],
    ) -> None:
        broken = dict(base_stacked)
        broken["gate_scales"] = broken["gate_scales"].to(torch.float32)
        with pytest.raises(RuntimeError, match="scales"):
            AWQFusedMoEImpl(
                num_experts=2,
                hidden_size=64,
                ffn_hidden_size=32,
                **broken,
            )

    def test_inconsistent_group_size_raises(
        self,
        base_stacked: dict[str, torch.Tensor],
    ) -> None:
        """gate / up / down 的 group_size 不一致应报错。"""
        # 重新构造一个 group_size 不一致的组合
        experts = [_make_expert(64, 32, 16, torch.float16, seed=0) for _ in range(2)]
        stacked = _stack_experts(experts)
        # 替换 down 的 group_size=8 的 experts（down 的 K=32，groups=32/8=4）
        bad_down = [_make_awq_weight(32, 64, 8, torch.float16, seed=i) for i in range(2)]
        stacked["down_qweight"] = torch.stack([x[0] for x in bad_down], dim=0)
        stacked["down_qzeros"] = torch.stack([x[1] for x in bad_down], dim=0)
        stacked["down_scales"] = torch.stack([x[2] for x in bad_down], dim=0)

        with pytest.raises(RuntimeError, match="group_size"):
            AWQFusedMoEImpl(
                num_experts=2,
                hidden_size=64,
                ffn_hidden_size=32,
                **stacked,
            )

    def test_hidden_states_wrong_dtype_raises(
        self,
        base_stacked: dict[str, torch.Tensor],
    ) -> None:
        impl = AWQFusedMoEImpl(
            num_experts=2,
            hidden_size=64,
            ffn_hidden_size=32,
            **base_stacked,
        )
        x_fp16 = torch.zeros((3, 64), dtype=torch.float16)
        topk_ids = torch.zeros((3, 1), dtype=torch.int64)
        topk_w = torch.ones((3, 1), dtype=torch.float32)
        with pytest.raises(RuntimeError, match="bfloat16"):
            impl.forward(x_fp16, topk_w, topk_ids)

    def test_hidden_states_wrong_last_dim_raises(
        self,
        base_stacked: dict[str, torch.Tensor],
    ) -> None:
        impl = AWQFusedMoEImpl(
            num_experts=2,
            hidden_size=64,
            ffn_hidden_size=32,
            **base_stacked,
        )
        x = torch.zeros((3, 32), dtype=torch.bfloat16)
        topk_ids = torch.zeros((3, 1), dtype=torch.int64)
        topk_w = torch.ones((3, 1), dtype=torch.float32)
        with pytest.raises(RuntimeError, match="H="):
            impl.forward(x, topk_w, topk_ids)

    def test_topk_ids_out_of_range_raises(
        self,
        base_stacked: dict[str, torch.Tensor],
    ) -> None:
        impl = AWQFusedMoEImpl(
            num_experts=2,
            hidden_size=64,
            ffn_hidden_size=32,
            **base_stacked,
        )
        x = torch.zeros((3, 64), dtype=torch.bfloat16)
        bad_ids = torch.tensor([[0], [1], [2]], dtype=torch.int64)  # 2 越界
        topk_w = torch.ones((3, 1), dtype=torch.float32)
        with pytest.raises(RuntimeError, match="越界"):
            impl.forward(x, topk_w, bad_ids)

    def test_topk_ids_shape_mismatch_raises(
        self,
        base_stacked: dict[str, torch.Tensor],
    ) -> None:
        impl = AWQFusedMoEImpl(
            num_experts=2,
            hidden_size=64,
            ffn_hidden_size=32,
            **base_stacked,
        )
        x = torch.zeros((3, 64), dtype=torch.bfloat16)
        ids = torch.zeros((3, 1), dtype=torch.int64)
        w = torch.ones((3, 2), dtype=torch.float32)  # shape 不一致
        with pytest.raises(RuntimeError, match="形状不一致"):
            impl.forward(x, w, ids)


# ── 4. 路由退化：top_k=1 + weight=1 → 对应 expert 稠密 FFN ─────────────────


class TestRoutingDegenerate:
    """top_k=1 且权重 1.0 时，MoE 输出等价于 per-token 选中的 expert 朴素 FFN。"""

    def test_top1_weight1_matches_dense_expert(self) -> None:
        # Arrange
        import torch.nn.functional as F

        h, f_dim, g = 128, 64, 32
        num_experts = 3
        num_tokens = 5
        torch.manual_seed(0)
        experts = [_make_expert(h, f_dim, g, torch.bfloat16, seed=e * 11) for e in range(num_experts)]
        stacked = _stack_experts(experts)

        topk_ids = (
            torch.arange(num_tokens, dtype=torch.int64)
            .remainder(
                num_experts,
            )
            .unsqueeze(-1)
        )
        topk_w = torch.ones((num_tokens, 1), dtype=torch.float32)
        hidden = torch.randn(num_tokens, h, dtype=torch.bfloat16) * 0.1

        impl = AWQFusedMoEImpl(
            num_experts=num_experts,
            hidden_size=h,
            ffn_hidden_size=f_dim,
            **stacked,
        )

        # Act：Impl 路径（w4a8）与 "每 token 对应 expert 的朴素 bf16 FFN" 对比
        out = impl.forward(hidden, topk_w, topk_ids)

        expected = torch.zeros_like(hidden, dtype=torch.float32)
        for t in range(num_tokens):
            e = int(topk_ids[t, 0].item())
            w = experts[e]
            gate_w = dequant_awq_to_bf16(w.gate_qweight, w.gate_qzeros, w.gate_scales)
            up_w = dequant_awq_to_bf16(w.up_qweight, w.up_qzeros, w.up_scales)
            down_w = dequant_awq_to_bf16(w.down_qweight, w.down_qzeros, w.down_scales)
            x_t = hidden[t : t + 1].float()
            gate_out = x_t @ gate_w.float()
            up_out = x_t @ up_w.float()
            inter = F.silu(gate_out) * up_out
            expected[t : t + 1] = inter @ down_w.float()
        ref = expected.to(torch.bfloat16)

        # Assert
        _assert_tensor_close(out, ref, **_TOL_W4A8_MOE)


# ── 5. R1 真实 shape 冒烟 ──────────────────────────────────────────────────


def test_awq_fused_moe_impl_r1_real_shape_smoke() -> None:
    """R1 真实 shape 子采样：H=7168, F=2048, g=64, num_experts=8, top_k=4。

    - 目的：确认在 DeepSeek-R1-AWQ 的 expert FFN 维度下 Impl 能跑通且与
      reference 在 W4A8 容差内一致。
    - num_experts 从 256 子采样到 8，保持测试运行时间可控。
    """
    # Arrange
    h, f_dim, g = 7168, 2048, 64
    num_experts, top_k, num_tokens = 8, 4, 2
    torch.manual_seed(0)
    experts = [_make_expert(h, f_dim, g, torch.float16, seed=e * 13) for e in range(num_experts)]
    stacked = _stack_experts(experts)
    hidden = torch.randn(num_tokens, h, dtype=torch.bfloat16) * 0.05
    topk_ids, topk_w = _random_topk(num_tokens, num_experts, top_k, seed=7)

    impl = AWQFusedMoEImpl(
        num_experts=num_experts,
        hidden_size=h,
        ffn_hidden_size=f_dim,
        **stacked,
    )

    # Act
    ref = awq_moe_expert_ffn_reference(hidden, topk_ids, topk_w, experts)
    out = impl.forward(hidden, topk_w, topk_ids)

    # Assert
    assert out.shape == (num_tokens, h)
    assert out.dtype == torch.bfloat16
    _assert_tensor_close(out, ref, **_TOL_W4A8_MOE)
