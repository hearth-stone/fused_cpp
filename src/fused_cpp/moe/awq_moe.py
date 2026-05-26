#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AWQ 权重的 MoE expert FFN 参考实现（纯 PyTorch + fused_cpp.w4a8_linear）。

本模块**只关心 expert FFN 本身的数值语义**，不涉及 router / topk / grouped_topk
等路由逻辑。路由由调用方（通常是 vLLM 的 ``FusedMoE.select_experts``）完成。

两种实现，共享完全一致的输入输出契约，便于数值等价性对比：

1. :func:`awq_moe_expert_ffn_reference`
   **金标准**路径：把每个 expert 的 ``(qweight, qzeros, scales)`` 按 AWQ
   interleave 约定解包、反量化为 bf16 权重，再走朴素 ``matmul → SiLU*up → matmul``。
   唯一引入的数值损失是 AWQ 4bit 权重本身的量化损失；**不**包含 per-token 激活
   量化。作为 Step 1 与后续实现对比的基线。

2. :func:`awq_moe_expert_ffn_w4a8`
   逐 expert 调用 :func:`fused_cpp.w4a8_linear`。每个 token→expert 的投影路径上
   会额外引入 per-token int8 激活量化损失。作为 Step 3 的生产路径。

DeepSeek-V2 / V3 / R1 风格的 expert FFN 结构（每 expert 3 个 Linear）::

    gate_out = x @ gate_proj.T                 # [T, F]
    up_out   = x @ up_proj.T                   # [T, F]
    hidden   = SiLU(gate_out) * up_out         # [T, F]
    out      = hidden @ down_proj.T            # [T, H]

AWQ 权重张量维度约定（本模块与 HF safetensors 原始存储对齐，参见 R1 checkpoint
key ``model.layers.L.mlp.experts.E.{gate_proj,up_proj,down_proj}.{qweight,qzeros,scales}``）::

    gate_proj / up_proj : (H → F)
        qweight : [H,          F // 8]    int32
        qzeros  : [H // g,     F // 8]    int32
        scales  : [H // g,     F   ]      fp16 / bf16

    down_proj : (F → H)
        qweight : [F,          H // 8]    int32
        qzeros  : [F // g,     H // 8]    int32
        scales  : [F // g,     H   ]      fp16 / bf16

其中 ``H`` 为 hidden_size（例如 R1 为 7168），``F`` 为 moe_intermediate_size
（例如 R1 为 2048），``g`` 为 AWQ group_size（例如 R1 为 64）。

所有路由相关参数（``top_k`` / ``renormalize`` / ``grouped_topk`` / ``sigmoid
scoring`` / ``e_score_correction_bias``）**不**在本模块出现；Step 2 接入 vLLM
时直接复用 vLLM 已有的 ``FusedMoE.select_experts``。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from fused_cpp.w4a8_linear import (
    unpack_awq_qweight,
    unpack_awq_qzeros,
    w4a8_linear,
)

__all__ = [
    "AWQExpertWeights",
    "dequant_awq_to_bf16",
    "awq_moe_expert_ffn_reference",
    "awq_moe_expert_ffn_w4a8",
]


@dataclass
class AWQExpertWeights:
    """单个 expert 的三组 AWQ 量化权重。

    所有张量的 N 维必须是 8 的倍数；group_size 由 ``K // qzeros.shape[0]`` 隐式给出。

    :param gate_qweight: ``[H, F // 8]`` int32。
    :param gate_qzeros:  ``[H // g, F // 8]`` int32。
    :param gate_scales:  ``[H // g, F]`` fp16 或 bf16。
    :param up_qweight:   ``[H, F // 8]`` int32。
    :param up_qzeros:    ``[H // g, F // 8]`` int32。
    :param up_scales:    ``[H // g, F]`` fp16 或 bf16。
    :param down_qweight: ``[F, H // 8]`` int32。
    :param down_qzeros:  ``[F // g, H // 8]`` int32。
    :param down_scales:  ``[F // g, H]`` fp16 或 bf16。
    """

    gate_qweight: torch.Tensor
    gate_qzeros: torch.Tensor
    gate_scales: torch.Tensor
    up_qweight: torch.Tensor
    up_qzeros: torch.Tensor
    up_scales: torch.Tensor
    down_qweight: torch.Tensor
    down_qzeros: torch.Tensor
    down_scales: torch.Tensor


def dequant_awq_to_bf16(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """将 AWQ 量化权重反量化为 bf16 的稠密权重矩阵。

    语义（与 :mod:`fused_cpp.w4a8_linear` 完全一致）::

        w_fp[k, j] = (w_q[k, j] - w_z[k // g, j]) * scales[k // g, j]

    :returns: ``[K, N]`` bf16。
    """
    if qweight.dim() != 2 or qzeros.dim() != 2 or scales.dim() != 2:
        raise RuntimeError("qweight / qzeros / scales 必须均为 2D")

    k, n_over_8 = qweight.shape
    n = n_over_8 * 8
    groups = qzeros.shape[0]
    if k % groups != 0:
        raise RuntimeError(f"K={k} 不能被 groups={groups} 整除")
    group_size = k // groups

    w_q = unpack_awq_qweight(qweight.contiguous()).to(torch.float32)
    w_z = unpack_awq_qzeros(qzeros.contiguous()).to(torch.float32)
    w_z_full = w_z.repeat_interleave(group_size, dim=0)
    s_full = scales.to(torch.float32).repeat_interleave(group_size, dim=0)

    w_fp = (w_q - w_z_full) * s_full                                # [K, N] fp32
    if w_fp.shape != (k, n):
        raise RuntimeError(
            f"反量化结果形状异常：{tuple(w_fp.shape)} 期望 {(k, n)}"
        )
    return w_fp.to(torch.bfloat16)


def _expert_ffn_bf16(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
) -> torch.Tensor:
    """DeepSeek-V2 风格 expert FFN 的朴素 bf16 实现：``SiLU(xW_g) * xW_u @ W_d``。

    :param x:      ``[T, H]`` bf16。
    :param gate_w: ``[H, F]`` bf16。
    :param up_w:   ``[H, F]`` bf16。
    :param down_w: ``[F, H]`` bf16。
    :returns:      ``[T, H]`` bf16。
    """
    # fp32 累加保证 SiLU 与乘法前的数值精度
    gate_out = torch.matmul(x.to(torch.float32), gate_w.to(torch.float32))
    up_out = torch.matmul(x.to(torch.float32), up_w.to(torch.float32))
    intermediate = F.silu(gate_out) * up_out                        # [T, F]
    out = torch.matmul(intermediate, down_w.to(torch.float32))
    return out.to(torch.bfloat16)


def _dispatch_tokens_to_experts(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    expert_fn,
) -> torch.Tensor:
    """通用的 "按 expert 分桶→执行→按权重加权累加" 调度骨架。

    ``expert_fn(e, x_sub) -> y_sub`` 负责为 expert ``e`` 计算 FFN 输出；本函数
    把 R1 级 top_k=8 的路由打平，逐 expert 聚合对应 token、执行 FFN、按
    ``topk_weights`` 加权写回输出，不做任何数值运算（只做 index 与 scatter）。

    :param hidden_states: ``[T, H]`` bf16。
    :param topk_ids:      ``[T, K]`` int32/int64，每行选中的 expert id。
    :param topk_weights:  ``[T, K]`` fp32/bf16，与 topk_ids 对齐的权重。
    :param num_experts:   全局 expert 数（用于防御越界）。
    :param expert_fn:     ``(int, [T_e, H] bf16) -> [T_e, H] bf16``。
    :returns:             ``[T, H]`` bf16。
    """
    t, h = hidden_states.shape
    topk = topk_ids.shape[1]
    if topk_ids.shape != (t, topk):
        raise RuntimeError(
            f"topk_ids 形状应为 ({t}, {topk})，当前 {tuple(topk_ids.shape)}"
        )
    if topk_weights.shape != (t, topk):
        raise RuntimeError(
            f"topk_weights 形状应为 ({t}, {topk})，当前 {tuple(topk_weights.shape)}"
        )

    # 用 fp32 做累加，避免 top_k=8 重复加法的 bf16 精度坍塌
    output = torch.zeros((t, h), dtype=torch.float32, device=hidden_states.device)
    weights_fp32 = topk_weights.to(torch.float32)

    # 每个 expert 挑出所有 (token_idx, slot_idx) 对，拼成一次 FFN
    for e in range(num_experts):
        match = (topk_ids == e).nonzero(as_tuple=False)             # [n_e, 2]
        if match.numel() == 0:
            continue
        tok_idx = match[:, 0]
        slot_idx = match[:, 1]

        x_sub = hidden_states.index_select(0, tok_idx)              # [n_e, H]
        y_sub = expert_fn(e, x_sub)                                 # [n_e, H]
        if y_sub.shape != x_sub.shape:
            raise RuntimeError(
                f"expert {e} 输出形状 {tuple(y_sub.shape)} 与输入"
                f"{tuple(x_sub.shape)} 不一致"
            )
        w = weights_fp32[tok_idx, slot_idx].unsqueeze(-1)           # [n_e, 1]
        # 用 index_add_ 以避免重复 token 被覆盖（scatter_add 语义）
        output.index_add_(0, tok_idx, y_sub.to(torch.float32) * w)

    return output.to(torch.bfloat16)


def awq_moe_expert_ffn_reference(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    experts: list[AWQExpertWeights],
) -> torch.Tensor:
    """AWQ MoE expert FFN 的**金标准**：每 expert 反量化到 bf16 后走朴素 FFN。

    唯一的数值损失是 AWQ 4bit 权重量化；不含激活量化。作为等价性测试基线。

    :param hidden_states: ``[T, H]`` bf16。
    :param topk_ids:      ``[T, K]`` 整数，每行选中的 expert id。
    :param topk_weights:  ``[T, K]`` 浮点权重（已在外部应用 renormalize /
                          routed_scaling_factor / sigmoid 等路由语义）。
    :param experts:       长度为 ``num_experts`` 的 AWQ 权重列表。
    :returns:             ``[T, H]`` bf16。
    """
    if hidden_states.dtype != torch.bfloat16:
        raise RuntimeError(
            f"hidden_states 必须为 bfloat16，当前 {hidden_states.dtype}"
        )

    # 惰性 dequant：遇到用不上的 expert 不做反量化
    dequant_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _get_bf16(e: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = dequant_cache.get(e)
        if cached is not None:
            return cached
        w = experts[e]
        gate_w = dequant_awq_to_bf16(w.gate_qweight, w.gate_qzeros, w.gate_scales)
        up_w = dequant_awq_to_bf16(w.up_qweight, w.up_qzeros, w.up_scales)
        down_w = dequant_awq_to_bf16(w.down_qweight, w.down_qzeros, w.down_scales)
        dequant_cache[e] = (gate_w, up_w, down_w)
        return dequant_cache[e]

    def _fn(e: int, x_sub: torch.Tensor) -> torch.Tensor:
        gate_w, up_w, down_w = _get_bf16(e)
        return _expert_ffn_bf16(x_sub, gate_w, up_w, down_w)

    return _dispatch_tokens_to_experts(
        hidden_states, topk_ids, topk_weights, len(experts), _fn,
    )


def awq_moe_expert_ffn_w4a8(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    experts: list[AWQExpertWeights],
) -> torch.Tensor:
    """AWQ MoE expert FFN 的 W4A8 实现：逐 expert 调用 ``fused_cpp.w4a8_linear``。

    与 :func:`awq_moe_expert_ffn_reference` 相比，额外引入 per-token int8 激活
    量化损失。**调用契约与 reference 完全一致**，便于直接对齐。

    :returns: ``[T, H]`` bf16。
    """
    if hidden_states.dtype != torch.bfloat16:
        raise RuntimeError(
            f"hidden_states 必须为 bfloat16，当前 {hidden_states.dtype}"
        )

    def _fn(e: int, x_sub: torch.Tensor) -> torch.Tensor:
        w = experts[e]
        # gate / up：H → F
        gate_out = w4a8_linear(x_sub, w.gate_qweight, w.gate_qzeros, w.gate_scales)
        up_out = w4a8_linear(x_sub, w.up_qweight, w.up_qzeros, w.up_scales)
        intermediate = F.silu(gate_out.to(torch.float32)) * up_out.to(torch.float32)
        # down：F → H
        return w4a8_linear(
            intermediate.to(torch.bfloat16),
            w.down_qweight,
            w.down_qzeros,
            w.down_scales,
        )

    return _dispatch_tokens_to_experts(
        hidden_states, topk_ids, topk_weights, len(experts), _fn,
    )
