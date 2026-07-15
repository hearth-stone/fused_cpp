#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模拟真实 MoE prefill 场景下的 per-expert token 分布。

支持三种路由模式：

1. ``uniform``：每个 token 独立均匀采样 top-k 个 expert（近似 aux-loss 达到
   完美平衡的理想情况），作为"最乐观基线"。
2. ``dirichlet``：从 Dirichlet 分布采样 expert 路由概率，再按概率多项式
   抽样，可通过 ``alpha`` 控制偏斜程度（alpha 越小越偏斜）。
3. ``grouped``：DeepSeek-V3 风格的 group-limited routing：先在组粒度上
   top-m，再在选中的组内部采 top-k，组间会出现明显分化。

用法::

    python tests/simulate_moe_routing.py \\
        --num-tokens 2048 --num-experts 256 --top-k 8 \\
        --mode dirichlet --alpha 0.5 \\
        --shards 1 2 4 5 8 10 20 40 \\
        --seed 0
"""

import argparse
import math
import random
import sys
from dataclasses import dataclass
from typing import List, Tuple


# ── 分布生成器 ──


def _routing_uniform(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    rng: random.Random,
) -> List[int]:
    """每个 token 独立等概率选 top-k 个不同 expert，返回 per-expert 计数。"""
    counts = [0] * num_experts
    all_ids = list(range(num_experts))
    for _ in range(num_tokens):
        picks = rng.sample(all_ids, top_k)
        for e in picks:
            counts[e] += 1
    return counts


def _sample_dirichlet(
    alpha: float,
    num_experts: int,
    rng: random.Random,
) -> List[float]:
    """对称 Dirichlet(alpha) 采样，返回长度 num_experts 的概率向量。

    通过 Gamma(alpha, 1) 采样后归一化实现，避免依赖 numpy。
    """

    # Marsaglia-Tsang 方法对 alpha >= 1 稳定；alpha < 1 时用 Johnk 方法。
    def _gamma(shape: float) -> float:
        if shape >= 1.0:
            d = shape - 1.0 / 3.0
            c = 1.0 / math.sqrt(9.0 * d)
            while True:
                x = rng.gauss(0.0, 1.0)
                v = (1.0 + c * x) ** 3
                if v <= 0.0:
                    continue
                u = rng.random()
                if u < 1.0 - 0.0331 * (x**4):
                    return d * v
                if math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
                    return d * v
        # alpha < 1：用 Gamma(shape + 1) * U^(1/shape)
        g = _gamma(shape + 1.0)
        u = rng.random()
        return g * (u ** (1.0 / shape))

    samples = [_gamma(alpha) for _ in range(num_experts)]
    total = sum(samples)
    if total <= 0.0:
        return [1.0 / num_experts] * num_experts
    return [s / total for s in samples]


def _weighted_top_k(
    weights: List[float],
    top_k: int,
    rng: random.Random,
) -> List[int]:
    """按 ``weights`` 不放回采样 top_k 个 id，类似 softmax 路由 + top-k。

    使用 Gumbel top-k trick：score_i = log(w_i) + Gumbel(0,1)，取最大的 k 个。
    """
    scored: List[Tuple[float, int]] = []
    for i, w in enumerate(weights):
        if w <= 0.0:
            score = -float("inf")
        else:
            u = rng.random()
            # 避免 log(0)
            u = max(u, 1e-20)
            gumbel = -math.log(-math.log(u))
            score = math.log(w) + gumbel
        scored.append((score, i))
    scored.sort(reverse=True)
    return [idx for _, idx in scored[:top_k]]


def _routing_dirichlet(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    alpha: float,
    rng: random.Random,
) -> List[int]:
    """Dirichlet(alpha) 产生全局 expert 权重，逐 token Gumbel top-k 采样。"""
    weights = _sample_dirichlet(alpha, num_experts, rng)
    counts = [0] * num_experts
    for _ in range(num_tokens):
        picks = _weighted_top_k(weights, top_k, rng)
        for e in picks:
            counts[e] += 1
    return counts


def _routing_grouped(
    num_tokens: int,
    num_experts: int,
    num_groups: int,
    top_m_groups: int,
    top_k: int,
    alpha: float,
    rng: random.Random,
) -> List[int]:
    """DeepSeek-V3 风格：先组级 top-m，再组内 top-k。

    流程：
        1. 从 Dirichlet(alpha) 采样全局 expert 权重 ``w``（每次运行固定）。
        2. 每组的分数 = 该组内 top-2 experts 的权重之和（DeepSeek-V3 做法）。
        3. 每 token 按组分数 Gumbel 采 top_m_groups 个组。
        4. 在选中的组内合并权重再 Gumbel top-k。
    """
    if num_experts % num_groups != 0:
        raise ValueError(
            f"num_experts={num_experts} 无法被 num_groups={num_groups} 整除",
        )
    per_group = num_experts // num_groups
    weights = _sample_dirichlet(alpha, num_experts, rng)

    # 预计算每组的 top-2 权重和（作为组分数）。
    group_scores: List[float] = []
    for g in range(num_groups):
        seg = weights[g * per_group : (g + 1) * per_group]
        top2 = sorted(seg, reverse=True)[:2]
        group_scores.append(sum(top2))

    counts = [0] * num_experts
    for _ in range(num_tokens):
        chosen_groups = _weighted_top_k(group_scores, top_m_groups, rng)
        # 拼接选中组的 expert 权重
        cand_ids: List[int] = []
        cand_w: List[float] = []
        for g in chosen_groups:
            for j in range(per_group):
                cand_ids.append(g * per_group + j)
                cand_w.append(weights[g * per_group + j])
        local = _weighted_top_k(cand_w, top_k, rng)
        for li in local:
            counts[cand_ids[li]] += 1
    return counts


# ── 统计与报告 ──


@dataclass
class DistStats:
    """分布统计摘要。"""

    total_tokens: int
    num_experts: int
    active_experts: int
    m_max: int
    m_min: int
    m_mean: float
    m_median: float
    m_p90: int
    m_p99: int
    cv: float  # 变异系数 std/mean
    lbc: float  # load balance coefficient = max / mean


def _percentile(sorted_vals: List[int], p: float) -> int:
    if not sorted_vals:
        return 0
    idx = max(0, min(len(sorted_vals) - 1, int(math.ceil(p * len(sorted_vals)) - 1)))
    return sorted_vals[idx]


def _stats(counts: List[int]) -> DistStats:
    total = sum(counts)
    n = len(counts)
    active = sum(1 for c in counts if c > 0)
    mean = total / n if n > 0 else 0.0
    var = sum((c - mean) ** 2 for c in counts) / n if n > 0 else 0.0
    std = math.sqrt(var)
    sc = sorted(counts)
    return DistStats(
        total_tokens=total,
        num_experts=n,
        active_experts=active,
        m_max=max(counts) if counts else 0,
        m_min=min(counts) if counts else 0,
        m_mean=mean,
        m_median=sc[n // 2] if n > 0 else 0.0,
        m_p90=_percentile(sc, 0.90),
        m_p99=_percentile(sc, 0.99),
        cv=std / mean if mean > 0 else 0.0,
        lbc=(max(counts) / mean) if mean > 0 else 0.0,
    )


def _print_stats(stats: DistStats, label: str) -> None:
    print(f"\n[{label}]")
    print(f"  total_tokens     : {stats.total_tokens}")
    print(f"  num_experts      : {stats.num_experts}")
    print(f"  active_experts   : {stats.active_experts} ({100.0 * stats.active_experts / stats.num_experts:.1f}%)")
    print(f"  M_max / M_mean   : {stats.m_max} / {stats.m_mean:.2f} (LBC={stats.lbc:.2f}x)")
    print(f"  M_min            : {stats.m_min}")
    print(f"  M_median         : {stats.m_median:.1f}")
    print(f"  M_p90 / M_p99    : {stats.m_p90} / {stats.m_p99}")
    print(f"  CV (std/mean)    : {stats.cv:.3f}")


def _print_histogram(counts: List[int], num_bins: int = 10) -> None:
    if not counts:
        return
    hi = max(counts)
    if hi == 0:
        print("  所有 expert M=0，跳过直方图")
        return
    bin_width = max(1, (hi + num_bins) // num_bins)
    bins = [0] * num_bins
    for c in counts:
        b = min(num_bins - 1, c // bin_width)
        bins[b] += 1
    max_bin = max(bins) if bins else 1
    print("  直方图（per-expert M 分布）：")
    for i, cnt in enumerate(bins):
        lo = i * bin_width
        hi_b = (i + 1) * bin_width - 1
        bar_len = int(40 * cnt / max_bin) if max_bin > 0 else 0
        bar = "█" * bar_len
        print(f"    M ∈ [{lo:>4}, {hi_b:>4}]: {cnt:>4}  {bar}")


# ── Shard 均衡性模拟 ──


def _lpt_partition(
    counts: List[int],
    num_shards: int,
) -> Tuple[List[List[int]], List[int]]:
    """LPT 贪心：按 M 降序依次塞进当前累计最小的 shard。

    只返回每个 shard 内的 expert id 列表与累计 M。
    """
    indexed = sorted(enumerate(counts), key=lambda ic: -ic[1])
    shards_ids: List[List[int]] = [[] for _ in range(num_shards)]
    shard_load = [0] * num_shards
    for eid, m in indexed:
        if m == 0:
            continue  # 零负载 expert 不分配
        s = min(range(num_shards), key=lambda i: shard_load[i])
        shards_ids[s].append(eid)
        shard_load[s] += m
    return shards_ids, shard_load


def _roundrobin_partition(
    counts: List[int],
    num_shards: int,
) -> Tuple[List[List[int]], List[int]]:
    """Round-robin：按 expert id 顺序轮流分配，模拟原 bench 的静态切分。"""
    shards_ids: List[List[int]] = [[] for _ in range(num_shards)]
    shard_load = [0] * num_shards
    for eid, m in enumerate(counts):
        if m == 0:
            continue
        s = eid % num_shards
        shards_ids[s].append(eid)
        shard_load[s] += m
    return shards_ids, shard_load


def _report_partition(
    label: str,
    shard_load: List[int],
    total: int,
) -> None:
    max_load = max(shard_load) if shard_load else 0
    min_load = min(shard_load) if shard_load else 0
    mean_load = total / len(shard_load) if shard_load else 0.0
    # 关键：wall time 受 max_load 决定，speedup 上限 = total / max_load
    upper_speedup = total / max_load if max_load > 0 else float("inf")
    print(
        f"  {label:<14} max={max_load:>5} min={min_load:>5} "
        f"mean={mean_load:>7.1f}  "
        f"imbalance(max/mean)={max_load / mean_load:.2f}x  "
        f"speedup_upper={upper_speedup:.2f}x",
    )


def _shard_analysis(counts: List[int], shard_list: List[int]) -> None:
    total = sum(counts)
    print("\n[Shard 均衡性分析]")
    print(f"  total_M={total}   理论 speedup 上限 = total / max_shard_load")
    print(f"  {'shards':<14}{'':<20}{'':<20}{'':<20}")
    for s in shard_list:
        print(f"  -- shards={s} --")
        _, rr_load = _roundrobin_partition(counts, s)
        _, lpt_load = _lpt_partition(counts, s)
        _report_partition("round-robin", rr_load, total)
        _report_partition("LPT", lpt_load, total)


# ── 主入口 ──


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tokens", type=int, default=2048, help="输入 token 数（prefill 长度），默认 2048")
    parser.add_argument("--num-experts", type=int, default=256, help="总 expert 数，默认 256（DeepSeek-V3 风格）")
    parser.add_argument("--top-k", type=int, default=8, help="每 token 激活的 expert 数，默认 8")
    parser.add_argument(
        "--mode",
        default="grouped",
        choices=["uniform", "dirichlet", "grouped", "all"],
        help="路由模式，默认 grouped（最贴近真实）；all 表示三种都跑",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Dirichlet 浓度参数；alpha 越小越偏斜；默认 0.5（中度偏斜）"
    )
    parser.add_argument("--num-groups", type=int, default=8, help="grouped 模式的组数，默认 8")
    parser.add_argument("--top-m-groups", type=int, default=4, help="grouped 模式每 token 选的组数，默认 4")
    parser.add_argument(
        "--shards",
        nargs="+",
        type=int,
        default=[1, 2, 4, 5, 8, 10, 20, 40],
        help="用于 shard 均衡性分析的 shard 数列表",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--show-histogram", action="store_true", help="打印 M 分布直方图")
    parser.add_argument("--show-topk-experts", type=int, default=10, help="打印 top-N 热门 expert 的 M 值；0 关闭")
    return parser


def _run_one_mode(mode: str, args: argparse.Namespace) -> List[int]:
    rng = random.Random(args.seed)
    if mode == "uniform":
        return _routing_uniform(
            args.num_tokens,
            args.num_experts,
            args.top_k,
            rng,
        )
    if mode == "dirichlet":
        return _routing_dirichlet(
            args.num_tokens,
            args.num_experts,
            args.top_k,
            args.alpha,
            rng,
        )
    if mode == "grouped":
        return _routing_grouped(
            args.num_tokens,
            args.num_experts,
            args.num_groups,
            args.top_m_groups,
            args.top_k,
            args.alpha,
            rng,
        )
    raise ValueError(f"unknown mode: {mode}")


def _run_and_report(mode: str, args: argparse.Namespace) -> List[int]:
    print("\n" + "=" * 80)
    print(
        f"路由模式: {mode}   (tokens={args.num_tokens}, "
        f"experts={args.num_experts}, top_k={args.top_k}"
        + (f", groups={args.num_groups}, top_m={args.top_m_groups}" if mode == "grouped" else "")
        + (f", alpha={args.alpha}" if mode != "uniform" else "")
        + ")"
    )
    print("=" * 80)

    counts = _run_one_mode(mode, args)
    stats = _stats(counts)
    _print_stats(stats, f"分布统计 ({mode})")

    if args.show_topk_experts > 0:
        top_experts = sorted(
            enumerate(counts),
            key=lambda ic: -ic[1],
        )[: args.show_topk_experts]
        print(f"  top-{args.show_topk_experts} 热门 expert (id, M):")
        for eid, m in top_experts:
            print(f"    expert {eid:>4}: M={m}")

    if args.show_histogram:
        _print_histogram(counts)

    _shard_analysis(counts, args.shards)
    return counts


def main() -> int:
    args = build_argparser().parse_args()
    modes = ["uniform", "dirichlet", "grouped"] if args.mode == "all" else [args.mode]
    for m in modes:
        _run_and_report(m, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
