#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AWQFusedMoEImpl: AWQ 量化的 Fused MoE Expert FFN 封装。

本类专注于 **topk 选择完成之后** 的 expert FFN 计算，**不涉及任何路由逻辑**：

* router 的 softmax / sigmoid / grouped top-k / ``e_score_correction_bias``
  一律由调用方（通常是 vLLM 的 :class:`SharedFusedMoE.forward`）完成。
* shared expert 也**不**在本类处理；shared expert 的 ``gate_proj`` /
  ``up_proj`` / ``down_proj`` 作为独立 AWQ Linear，由 vLLM 的
  ``AWQConfig.get_quant_method(LinearBase)`` 分派到已有的
  ``CPUAWQLinearMethod``（即 :func:`fused_cpp.w4a8_linear`）路径。
* EP all-reduce 由调用方完成；本类在 ``ep_size > 1`` 时暂不支持，便于最小化
  第一版实现的风险面。

与 :class:`fused_cpp.moe.impl.FusedMoEImpl`（bf16 版）形成对照：骨架完全一致，
只是"每 expert 的 3 个 matmul"被换成"每 expert 的 3 次
:func:`fused_cpp.w4a8_linear`"。per-expert 实现复用
:func:`fused_cpp.moe.awq_moe.awq_moe_expert_ffn_w4a8`，确保与已通过数值验证的
单 expert 实现共享同一条数值通路。

典型调用形态（对应 vLLM ``FusedMoEMethodBase.apply`` 的签名）::

    impl = AWQFusedMoEImpl(
        num_experts=256, hidden_size=7168, ffn_hidden_size=2048,
        gate_qweight=w1_qweight,  # [E, H, F//8] int32
        gate_qzeros=w1_qzeros,    # [E, H//g, F//8] int32
        gate_scales=w1_scales,    # [E, H//g, F] fp16/bf16
        up_qweight=w3_qweight, up_qzeros=w3_qzeros, up_scales=w3_scales,
        down_qweight=w2_qweight,  # [E, F, H//8] int32
        down_qzeros=w2_qzeros,    # [E, F//g, H//8] int32
        down_scales=w2_scales,    # [E, F//g, H] fp16/bf16
    )
    out = impl.forward(hidden_states, topk_weights, topk_ids)
"""

from __future__ import annotations

import torch

from fused_cpp.moe.awq_moe import (
    AWQExpertWeights,
    awq_moe_expert_ffn_w4a8,
)

__all__ = ["AWQFusedMoEImpl"]


class AWQFusedMoEImpl:
    """AWQ Fused MoE expert FFN（topk 之后），per-expert 调 ``w4a8_linear``。

    形状约定（与 AutoAWQ / R1-AWQ checkpoint 完全一致）::

        H = hidden_size, F = ffn_hidden_size, E = num_experts
        group_size 由 scales 的 leading 维推断：g = H // gate_scales.shape[1]
                                              = F // down_scales.shape[1]

        gate_qweight: [E, H,      F // 8]   int32
        gate_qzeros : [E, H // g, F // 8]   int32
        gate_scales : [E, H // g, F      ]  fp16 / bf16
        up_qweight  : [E, H,      F // 8]   int32
        up_qzeros   : [E, H // g, F // 8]   int32
        up_scales   : [E, H // g, F      ]  fp16 / bf16
        down_qweight: [E, F,      H // 8]   int32
        down_qzeros : [E, F // g, H // 8]   int32
        down_scales : [E, F // g, H      ]  fp16 / bf16

    :param num_experts:     全局 expert 数 ``E``（当前 ``ep_size=1``，即本 rank 拥有全部 expert）。
    :param hidden_size:     ``H``。
    :param ffn_hidden_size: ``F``（单 expert 的中间维，**不**是 ``2 * F``）。
    :param gate_qweight:    AWQ gate 投影量化权重。
    :param gate_qzeros:     AWQ gate 投影零点。
    :param gate_scales:     AWQ gate 投影 scales。
    :param up_qweight:      AWQ up 投影量化权重。
    :param up_qzeros:       AWQ up 投影零点。
    :param up_scales:       AWQ up 投影 scales。
    :param down_qweight:    AWQ down 投影量化权重。
    :param down_qzeros:     AWQ down 投影零点。
    :param down_scales:     AWQ down 投影 scales。
    :param ep_size:         EP 并行度；**当前仅支持 1**，>1 将抛 ``NotImplementedError``。
    :param ep_rank:         EP rank，仅在 ``ep_size>1`` 时使用。
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        ffn_hidden_size: int,
        gate_qweight: torch.Tensor,
        gate_qzeros: torch.Tensor,
        gate_scales: torch.Tensor,
        up_qweight: torch.Tensor,
        up_qzeros: torch.Tensor,
        up_scales: torch.Tensor,
        down_qweight: torch.Tensor,
        down_qzeros: torch.Tensor,
        down_scales: torch.Tensor,
        *,
        ep_size: int = 1,
        ep_rank: int = 0,
    ) -> None:
        if ep_size != 1:
            raise NotImplementedError(
                f"AWQFusedMoEImpl 当前仅支持 ep_size=1，收到 ep_size={ep_size}。"
                "EP 路径需在 vLLM 侧完成分片与 all-reduce，该能力预计在后续步骤补齐。"
            )
        if not (0 <= ep_rank < ep_size):
            raise ValueError(f"ep_rank={ep_rank} 越界（ep_size={ep_size}）")

        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.ep_size = ep_size
        self.ep_rank = ep_rank

        self._validate_stacked_weights(
            gate_qweight,
            gate_qzeros,
            gate_scales,
            up_qweight,
            up_qzeros,
            up_scales,
            down_qweight,
            down_qzeros,
            down_scales,
        )

        # 保留堆叠张量引用供诊断 / 后续 EP 使用
        self._gate_qweight = gate_qweight
        self._gate_qzeros = gate_qzeros
        self._gate_scales = gate_scales
        self._up_qweight = up_qweight
        self._up_qzeros = up_qzeros
        self._up_scales = up_scales
        self._down_qweight = down_qweight
        self._down_qzeros = down_qzeros
        self._down_scales = down_scales

        # 展开为 per-expert AWQExpertWeights list：仅构建对每个 expert 的 view，
        # 不复制底层存储（torch 的基础切片是 zero-copy）。
        self._experts: list[AWQExpertWeights] = [
            AWQExpertWeights(
                gate_qweight=gate_qweight[e],
                gate_qzeros=gate_qzeros[e],
                gate_scales=gate_scales[e],
                up_qweight=up_qweight[e],
                up_qzeros=up_qzeros[e],
                up_scales=up_scales[e],
                down_qweight=down_qweight[e],
                down_qzeros=down_qzeros[e],
                down_scales=down_scales[e],
            )
            for e in range(num_experts)
        ]

    # ── 校验 ────────────────────────────────────────────────────────────────

    def _validate_stacked_weights(
        self,
        gate_qweight: torch.Tensor,
        gate_qzeros: torch.Tensor,
        gate_scales: torch.Tensor,
        up_qweight: torch.Tensor,
        up_qzeros: torch.Tensor,
        up_scales: torch.Tensor,
        down_qweight: torch.Tensor,
        down_qzeros: torch.Tensor,
        down_scales: torch.Tensor,
    ) -> None:
        """严格校验 9 个堆叠张量的 shape / dtype 是否符合 AWQ 契约。"""
        e = self.num_experts
        h = self.hidden_size
        f = self.ffn_hidden_size

        if f % 8 != 0 or h % 8 != 0:
            raise RuntimeError(f"hidden_size={h}、ffn_hidden_size={f} 必须均为 8 的倍数")

        # gate / up：H → F
        for name, qw, qz, sc in (
            ("gate", gate_qweight, gate_qzeros, gate_scales),
            ("up", up_qweight, up_qzeros, up_scales),
        ):
            self._check_one_proj(name, qw, qz, sc, e, h, f)

        # down：F → H
        self._check_one_proj("down", down_qweight, down_qzeros, down_scales, e, f, h)

        # gate / up / down 的 group_size 必须一致
        g_gate = h // gate_scales.shape[1]
        g_up = h // up_scales.shape[1]
        g_down = f // down_scales.shape[1]
        if not (g_gate == g_up == g_down):
            raise RuntimeError(
                f"gate / up / down 的 group_size 必须一致，当前：gate={g_gate}, up={g_up}, down={g_down}"
            )
        self.group_size = g_gate

    @staticmethod
    def _check_one_proj(
        name: str,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        e: int,
        k: int,
        n: int,
    ) -> None:
        """校验某一个投影（gate/up/down）的 3 个张量形状与 dtype。"""
        if qweight.dtype != torch.int32:
            raise RuntimeError(f"{name}_qweight 必须为 int32，当前 {qweight.dtype}")
        if qzeros.dtype != torch.int32:
            raise RuntimeError(f"{name}_qzeros 必须为 int32，当前 {qzeros.dtype}")
        if scales.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(f"{name}_scales 必须为 fp16 或 bf16，当前 {scales.dtype}")

        if qweight.shape != (e, k, n // 8):
            raise RuntimeError(f"{name}_qweight 形状应为 ({e}, {k}, {n // 8})，当前 {tuple(qweight.shape)}")

        g = qzeros.shape[1]
        if qzeros.shape != (e, g, n // 8):
            raise RuntimeError(f"{name}_qzeros 形状应为 ({e}, G, {n // 8})，当前 {tuple(qzeros.shape)}")
        if scales.shape != (e, g, n):
            raise RuntimeError(f"{name}_scales 形状应为 ({e}, G={g}, {n})，当前 {tuple(scales.shape)}")
        if k % g != 0:
            raise RuntimeError(f"{name}: K={k} 不能被 groups={g} 整除")

    # ── 入口 ────────────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """topk 选择完成后的 AWQ MoE expert FFN。

        :param hidden_states: ``[T, H]`` bfloat16。
        :param topk_weights:  ``[T, K]`` 浮点权重（renormalize/scaling 等由调用方完成）。
        :param topk_ids:      ``[T, K]`` 整数 expert id。
        :returns:             ``[T, H]`` bfloat16。
        """
        if hidden_states.dim() != 2:
            raise RuntimeError(f"hidden_states 必须为 2D [T, H]，当前 {hidden_states.dim()}D")
        if hidden_states.shape[1] != self.hidden_size:
            raise RuntimeError(f"hidden_states 最后一维应为 H={self.hidden_size}，当前 {hidden_states.shape[1]}")
        if hidden_states.dtype != torch.bfloat16:
            raise RuntimeError(f"hidden_states 必须为 bfloat16，当前 {hidden_states.dtype}")
        if topk_ids.shape != topk_weights.shape:
            raise RuntimeError(
                f"topk_ids 与 topk_weights 形状不一致：{tuple(topk_ids.shape)} vs {tuple(topk_weights.shape)}"
            )
        if topk_ids.shape[0] != hidden_states.shape[0]:
            raise RuntimeError(f"topk_ids 第 0 维应等于 token 数 {hidden_states.shape[0]}，当前 {topk_ids.shape[0]}")

        # 越界 id 防御：vLLM 在 EP/expert_map 路径下允许出现 -1 代表 "不在本 rank"，
        # 当前 ep_size=1 不应出现，严格检查以便早报错。
        if topk_ids.numel() > 0:
            id_min = int(topk_ids.min().item())
            id_max = int(topk_ids.max().item())
            if id_min < 0 or id_max >= self.num_experts:
                raise RuntimeError(f"topk_ids 越界：min={id_min}, max={id_max}，合法范围 [0, {self.num_experts})")

        return awq_moe_expert_ffn_w4a8(
            hidden_states,
            topk_ids,
            topk_weights,
            self._experts,
        )
