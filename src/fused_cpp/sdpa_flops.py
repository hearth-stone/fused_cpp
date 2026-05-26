# -*- coding: utf-8 -*-
"""SDPA 理论 FLOPs 计算工具。

公式（与 :doc:`requirements.md` 第 6 节一致）：

  Dense 形式（不考虑 causal mask）：

  * ``qk_flops      = 2 * B * N * L * S * E``
  * ``softmax_flops = 5 * B * N * L * S``    （max/sub/exp/sum/div 共 5 ops）
  * ``av_flops      = 2 * B * N * L * S * Ev``
  * ``total_flops   = qk_flops + softmax_flops + av_flops``

  Causal-adjusted 形式（``is_causal=True``）：

  * ``S_eff(l) = clip(l + (S - L) + 1, 0, S)``
  * ``Σ_l S_eff(l)`` 即下三角的有效 token 对数；
  * ``qk_flops_causal      = 2 * B * N * E  * Σ_l S_eff(l)``
  * ``av_flops_causal      = 2 * B * N * Ev * Σ_l S_eff(l)``
  * ``softmax_flops_causal = 5 * B * N      * Σ_l S_eff(l)``

约定：``attn_mask`` 不参与 FLOPs 折扣（避免 mask 稀疏度差异污染版本对比）。
"""
from __future__ import annotations

from typing import Dict, Optional

__all__ = [
    "compute_sdpa_flops",
    "compute_causal_effective_pairs",
    "gflops",
]


def compute_causal_effective_pairs(L: int, S: int) -> int:
    """计算 ``Σ_l min(l + (S - L) + 1, S)`` 在 ``l ∈ [0, L)`` 上的累加。

    ``S_eff(l)`` 表示在「lower-right causal」语义下 query 行 ``l`` 可见的
    key 数（与 C++ 内核中 ``causal_offset = S - L`` 对齐）。

    实现采用闭式公式以避免 ``L`` 巨大时的循环开销：

        S_eff(l) = clip(l + offset + 1, 0, S)，offset = S - L

    上述函数在 ``l`` 上分三段：
      a) ``l <  -offset`` → 0
      b) ``l <= S - 1 - offset`` → ``l + offset + 1``
      c) ``l >  S - 1 - offset`` → ``S``

    :param L: query 序列长度
    :param S: key/value 序列长度
    :return: ``Σ_l S_eff(l)``，整数。
    """
    if L <= 0 or S <= 0:
        return 0
    offset = S - L

    # 段 a: l < -offset  → S_eff = 0
    a_end = max(0, -offset)            # 不含此值
    # 段 c: l > S - 1 - offset → S_eff = S
    c_start = (S - 1 - offset) + 1     # 含此值
    if c_start > L:
        c_start = L
    if c_start < 0:
        c_start = 0
    # 段 b: a_end <= l < c_start → S_eff = l + offset + 1
    b_start = a_end
    b_end = c_start

    total = 0
    # 段 b 的求和：Σ_{l=b_start}^{b_end-1} (l + offset + 1)
    if b_end > b_start:
        n = b_end - b_start
        first = b_start + offset + 1
        last = b_end - 1 + offset + 1
        total += (first + last) * n // 2
    # 段 c 贡献：(L - c_start) * S
    if L > c_start:
        total += (L - c_start) * S

    return total


def compute_sdpa_flops(
    B: int,
    N: int,
    L: int,
    S: int,
    E: int,
    Ev: int,
    *,
    is_causal: bool = False,
) -> Dict[str, int]:
    """计算 SDPA 的理论 FLOPs。

    :return: 字典，至少包含：

      * ``qk_flops``、``softmax_flops``、``av_flops``、``total_flops``：
        dense 公式结果；
      * ``causal_pairs``：``Σ_l S_eff(l)``，``is_causal=False`` 时等于 ``L*S``；
      * ``qk_flops_causal``、``softmax_flops_causal``、``av_flops_causal``、
        ``total_flops_causal``：causal-adjusted 公式结果。

    所有字段均为 Python ``int``（避免 numpy/torch 类型）。
    """
    # ── Dense 公式 ──
    pairs_dense = L * S
    qk_flops = 2 * B * N * pairs_dense * E
    softmax_flops = 5 * B * N * pairs_dense
    av_flops = 2 * B * N * pairs_dense * Ev
    total_flops = qk_flops + softmax_flops + av_flops

    # ── Causal-adjusted 公式 ──
    if is_causal:
        pairs_causal = compute_causal_effective_pairs(L, S)
    else:
        pairs_causal = pairs_dense

    qk_flops_causal = 2 * B * N * pairs_causal * E
    softmax_flops_causal = 5 * B * N * pairs_causal
    av_flops_causal = 2 * B * N * pairs_causal * Ev
    total_flops_causal = qk_flops_causal + softmax_flops_causal + av_flops_causal

    return {
        "B": int(B),
        "N": int(N),
        "L": int(L),
        "S": int(S),
        "E": int(E),
        "Ev": int(Ev),
        "is_causal": bool(is_causal),
        "qk_flops": int(qk_flops),
        "softmax_flops": int(softmax_flops),
        "av_flops": int(av_flops),
        "total_flops": int(total_flops),
        "causal_pairs": int(pairs_causal),
        "qk_flops_causal": int(qk_flops_causal),
        "softmax_flops_causal": int(softmax_flops_causal),
        "av_flops_causal": int(av_flops_causal),
        "total_flops_causal": int(total_flops_causal),
    }


def gflops(flops: int, seconds: float) -> float:
    """便捷工具：将 FLOPs / 秒数 转成 GFLOP/s。

    秒数 ``<= 0`` 时返回 0.0，避免除零。
    """
    if seconds <= 0:
        return 0.0
    return float(flops) / float(seconds) / 1e9
