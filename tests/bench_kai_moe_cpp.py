#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MoE-like KleidiAI GEMM 压测脚本（显式线程池 + Python 调度）。

新架构：
    - 每个 group 一个独立的 :class:`KAIThreadPool`，绑到该 group 的 CPU 段；
    - 每个 GEMM handler 是纯数据容器，可被不同 pool 复用；
    - ``group_splits > 1`` 时在 Python 侧起 ``group_splits`` 个线程，每线程
      独占一个 pool，串行派发其名下的 GEMM 任务；
    - 调度 + 计时在 Python 侧完成，C++ 侧只负责单次 GEMM 的多线程计算。

用法示例::

    # 旧行为：M 在 [m_min, m_max] 均匀采样
    taskset -c 0-79 \\
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python tests/bench_kai_moe_cpp.py \\
        --routing-mode legacy \\
        --num-groups 160 --m-min 50 --m-max 150 \\
        --group-splits 1 2 4 5 8 10 20 40 \\
        --warmup 1 --repeat 3 \\
        --output-csv bench_moe_cpp.csv

    # 新行为：模拟 DeepSeek-V3 风格的 grouped top-k 路由
    taskset -c 0-79 \\
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python tests/bench_kai_moe_cpp.py \\
        --routing-mode grouped \\
        --num-tokens 2048 --num-experts 256 --top-k 8 \\
        --num-route-groups 8 --top-m-groups 4 --alpha 0.5 \\
        --assignment lpt \\
        --group-splits 1 2 4 5 8 10 20 40 \\
        --warmup 1 --repeat 3
"""

import argparse
import csv
import math
import os
import platform
import random
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

# 复用同目录下的路由分布模拟器。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulate_moe_routing as smr  # noqa: E402  pylint: disable=wrong-import-position


# ── 数据结构 ──


@dataclass
class GEMMSpec:
    """单次 GEMM 规格。"""

    group_id: int
    stage: int
    m: int
    k: int
    n: int

    def flops(self) -> float:
        return 2.0 * self.m * self.k * self.n


@dataclass
class StrategyResult:
    """一次策略运行的聚合结果。"""

    strategy: str
    total_cpus: int
    group_splits: int
    cpus_per_group: int
    num_gemms: int
    total_flops: float
    wall_time_ms: float
    gflops: float
    assignment: str = "roundrobin"
    speedup_upper: float = 0.0


# ── 任务生成 ──


def _build_specs(
    num_groups: int,
    m_min: int,
    m_max: int,
    k1: int,
    n1: int,
    k2: int,
    n2: int,
    seed: int,
) -> List[GEMMSpec]:
    """生成 ``num_groups`` 组，每组 2 个 GEMM 的 spec 列表（均匀 M 分布）。"""
    rng = random.Random(seed)
    specs: List[GEMMSpec] = []
    for g in range(num_groups):
        m = rng.randint(m_min, m_max)
        specs.append(GEMMSpec(group_id=g, stage=0, m=m, k=k1, n=n1))
        specs.append(GEMMSpec(group_id=g, stage=1, m=m, k=k2, n=n2))
    return specs


def _build_specs_from_routing(
    counts: List[int],
    k1: int,
    n1: int,
    k2: int,
    n2: int,
    keep_zero: bool = False,
) -> List[GEMMSpec]:
    """根据 per-expert token 计数生成 GEMM spec 列表。

    每个 M > 0 的 expert 产生 2 个 GEMM（stage0/stage1），M=0 的 expert
    默认跳过（``keep_zero=False``）以贴近真实 MoE 执行行为。
    """
    specs: List[GEMMSpec] = []
    for eid, m in enumerate(counts):
        if m <= 0 and not keep_zero:
            continue
        m_val = int(m) if m > 0 else 1  # keep_zero 时最小用 1，避免零尺寸张量
        specs.append(GEMMSpec(group_id=eid, stage=0, m=m_val, k=k1, n=n1))
        specs.append(GEMMSpec(group_id=eid, stage=1, m=m_val, k=k2, n=n2))
    return specs


def _build_inputs_and_outputs(
    specs: List[GEMMSpec],
    out_dtype: torch.dtype,
    seed: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """按 spec 预分配 FP32 输入张量与输出张量。"""
    inputs: List[torch.Tensor] = []
    outputs: List[torch.Tensor] = []
    for s in specs:
        gen = torch.Generator().manual_seed(
            seed ^ (s.group_id * 1000003 + s.stage * 19 + s.m),
        )
        inputs.append(
            torch.randn(
                s.m,
                s.k,
                dtype=torch.float32,
                generator=gen,
            )
        )
        outputs.append(torch.empty(s.m, s.n, dtype=out_dtype))
    return inputs, outputs


# ── 权重与 handler 管理 ──


class HandlerRegistry:
    """按 ``(K, N)`` 缓存 packed 权重和 handler。

    新架构下 handler 不再绑定 CPU 或线程数，因此对于同一 (K, N) 可以
    **完全共享** 一份 packed_weight 和一个 handler，避免重复分配。
    handler 在运行时通过 ``kai_gemm(..., pool=...)`` 显式接收 pool，
    不同的 pool 可以复用同一个 handler。
    """

    def __init__(self, fused_cpp_mod, seed: int) -> None:
        self._mod = fused_cpp_mod
        self._seed = seed
        self._packed_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self._handler_cache: Dict[Tuple[int, int], int] = {}
        # 保活：避免 packed_weight 被 GC 掉导致 handler 悬空。
        self._alive_packed: List[torch.Tensor] = []

    def _get_packed(self, k: int, n: int) -> torch.Tensor:
        key = (k, n)
        packed = self._packed_cache.get(key)
        if packed is not None:
            return packed
        gen = torch.Generator().manual_seed(self._seed ^ (k * 131071 + n))
        weight = torch.randn(k, n, dtype=torch.float32, generator=gen)
        packed = self._mod.kai_gemm_prepare(weight)
        self._packed_cache[key] = packed
        self._alive_packed.append(packed)
        return packed

    def build_handler(self, k: int, n: int) -> int:
        """获取或创建 (K, N) 对应的 handler，返回其 int64 指针。"""
        key = (k, n)
        existing = self._handler_cache.get(key)
        if existing is not None:
            return existing
        packed = self._get_packed(k, n)
        handler = self._mod.create_kai_gemm_handler(
            packed,
            int(k),
            int(n),
        )
        self._handler_cache[key] = handler
        return handler

    def release_all(self) -> None:
        for ptr in self._handler_cache.values():
            try:
                self._mod.release_kai_gemm_handler(ptr)
            except Exception:  # pylint: disable=broad-except
                pass
        self._handler_cache.clear()
        self._packed_cache.clear()
        self._alive_packed.clear()


# ── 任务分配 ──


def _assign_roundrobin(
    specs: List[GEMMSpec],
    group_splits: int,
) -> List[List[int]]:
    """按 spec 索引轮询分配（原始行为）。"""
    assignments: List[List[int]] = [[] for _ in range(group_splits)]
    for i in range(len(specs)):
        assignments[i % group_splits].append(i)
    return assignments


def _assign_lpt(
    specs: List[GEMMSpec],
    group_splits: int,
) -> List[List[int]]:
    """LPT 贪心分配：按单次 GEMM FLOPs 降序塞进当前最小负载的 shard。"""
    indexed = sorted(
        enumerate(specs),
        key=lambda ispec: -ispec[1].flops(),
    )
    assignments: List[List[int]] = [[] for _ in range(group_splits)]
    loads: List[float] = [0.0] * group_splits
    for idx, s in indexed:
        tgt = min(range(group_splits), key=lambda i: loads[i])
        assignments[tgt].append(idx)
        loads[tgt] += s.flops()
    return assignments


def _speedup_upper_bound(
    assignments: List[List[int]],
    specs: List[GEMMSpec],
) -> float:
    """根据分配结果计算理论 speedup 上限 = total_flops / max_shard_flops。"""
    shard_flops = [sum(specs[i].flops() for i in group) for group in assignments]
    total = sum(shard_flops)
    mx = max(shard_flops) if shard_flops else 0.0
    return (total / mx) if mx > 0 else 0.0


# ── 策略压测 ──


def _split_cpus(cpus: List[int], num_groups: int) -> List[List[int]]:
    """将 ``cpus`` 顺序切分为 ``num_groups`` 段。"""
    if len(cpus) % num_groups != 0:
        raise ValueError(
            f"可用 CPU 数 {len(cpus)} 无法被 group_splits={num_groups} 整除",
        )
    per = len(cpus) // num_groups
    return [cpus[g * per : (g + 1) * per] for g in range(num_groups)]


def _run_one_gemm(
    fused_cpp_mod,
    handler_ptr: int,
    pool_handle: int,
    x: torch.Tensor,
    y: torch.Tensor,
) -> None:
    """在给定 pool 上执行一次 GEMM，结果写回 ``y``。

    直接调用底层 ``fused_cpp._C.kai_gemm``，避免 Python wrapper 重复分配
    output 张量。
    """
    fused_cpp_mod.kai_gemm(y, x, handler_ptr, pool_handle)


def _bench_one(
    strategy: str,
    group_splits: int,
    cpus: List[int],
    specs: List[GEMMSpec],
    inputs: List[torch.Tensor],
    outputs: List[torch.Tensor],
    registry: HandlerRegistry,
    fused_cpp_mod,
    warmup: int,
    repeat: int,
    assignment: str = "roundrobin",
) -> StrategyResult:
    """执行一次策略压测，返回结果。

    策略：
        - 将 ``specs`` 按 ``assignment`` 切分给 ``group_splits`` 个 group；
        - 每个 group 独占一个 KAIThreadPool，对应一段固定 CPU；
        - ``group_splits > 1`` 时每 group 启一个 Python 线程串行派发任务。
    """
    total_flops = sum(s.flops() for s in specs)
    num_gemms = len(specs)
    cpus_per_group = len(cpus) // group_splits

    group_cpu_lists = _split_cpus(cpus, group_splits)

    # 创建 group_splits 个 pool。每个 pool 的 cpu_ids[0] 即该 group 的
    # 调度线程绑核位置，cpu_ids[1..] 分给 worker。
    pools: List[int] = []
    for cpu_list in group_cpu_lists:
        pools.append(
            fused_cpp_mod.create_kai_thread_pool(
                [int(c) for c in cpu_list],
            )
        )

    # 为每个 spec 创建（按 (K,N) 去重后的）handler 指针。
    handler_ptrs: List[int] = [registry.build_handler(s.k, s.n) for s in specs]

    # 将 specs 的索引按策略切分给 group_splits 个 group。
    if assignment == "lpt":
        assignments = _assign_lpt(specs, group_splits)
    else:
        assignments = _assign_roundrobin(specs, group_splits)

    upper = _speedup_upper_bound(assignments, specs)

    def run_once() -> None:
        if group_splits == 1:
            pool_handle = pools[0]
            for i in range(num_gemms):
                _run_one_gemm(
                    fused_cpp_mod,
                    handler_ptrs[i],
                    pool_handle,
                    inputs[i],
                    outputs[i],
                )
            return

        # 多 group：每个 group 一个 Python 线程，独占一个 pool 串行派发。
        threads: List[threading.Thread] = []
        exc_box: List[BaseException] = []
        exc_lock = threading.Lock()

        def worker(gidx: int) -> None:
            pool_handle = pools[gidx]
            try:
                for idx in assignments[gidx]:
                    _run_one_gemm(
                        fused_cpp_mod,
                        handler_ptrs[idx],
                        pool_handle,
                        inputs[idx],
                        outputs[idx],
                    )
            except BaseException as exc:  # pylint: disable=broad-except
                with exc_lock:
                    exc_box.append(exc)

        for g in range(group_splits):
            t = threading.Thread(target=worker, args=(g,), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        if exc_box:
            raise exc_box[0]

    try:
        for _ in range(warmup):
            run_once()

        best_wall_ms = math.inf
        for _ in range(repeat):
            t0 = time.perf_counter()
            run_once()
            t1 = time.perf_counter()
            ms = (t1 - t0) * 1000.0
            if ms < best_wall_ms:
                best_wall_ms = ms
    finally:
        for p in pools:
            try:
                fused_cpp_mod.destroy_kai_thread_pool(p)
            except Exception:  # pylint: disable=broad-except
                pass

    gflops = total_flops / (best_wall_ms / 1e3) / 1e9 if best_wall_ms > 0 else 0.0
    return StrategyResult(
        strategy=strategy,
        total_cpus=len(cpus),
        group_splits=group_splits,
        cpus_per_group=cpus_per_group,
        num_gemms=num_gemms,
        total_flops=total_flops,
        wall_time_ms=best_wall_ms,
        gflops=gflops,
        assignment=assignment,
        speedup_upper=upper,
    )


# ── 结果输出 ──


def _write_csv(results: List[StrategyResult], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "strategy",
                "assignment",
                "total_cpus",
                "group_splits",
                "cpus_per_group",
                "num_gemms",
                "total_flops",
                "wall_time_ms",
                "gflops",
                "speedup_vs_seq",
                "speedup_upper",
            ]
        )
        seq = next(
            (r for r in results if r.strategy == "sequential"),
            None,
        )
        base = seq.wall_time_ms if seq else None
        for r in results:
            speedup = base / r.wall_time_ms if base and r.wall_time_ms > 0 else ""
            writer.writerow(
                [
                    r.strategy,
                    r.assignment,
                    r.total_cpus,
                    r.group_splits,
                    r.cpus_per_group,
                    r.num_gemms,
                    f"{r.total_flops:.3e}",
                    f"{r.wall_time_ms:.3f}",
                    f"{r.gflops:.2f}",
                    f"{speedup:.3f}" if speedup != "" else "",
                    f"{r.speedup_upper:.3f}",
                ]
            )


def _print_summary(results: List[StrategyResult]) -> None:
    print()
    print("=" * 112)
    print(
        f"{'strategy':<12}{'assign':<12}{'cpus':>6}{'x':>4}{'c/g':>5}"
        f"{'wall_ms':>12}{'GFLOP/s':>12}{'speedup':>10}{'upper':>10}"
        f"{'eff%':>8}",
    )
    print("=" * 112)
    seq = next((r for r in results if r.strategy == "sequential"), None)
    base = seq.wall_time_ms if seq else None
    for r in results:
        speedup_val = base / r.wall_time_ms if base and r.wall_time_ms > 0 else 0.0
        speedup_str = f"{speedup_val:.2f}x" if speedup_val > 0 else "-"
        upper_str = f"{r.speedup_upper:.2f}x" if r.speedup_upper > 0 else "-"
        eff = 100.0 * speedup_val / r.speedup_upper if r.speedup_upper > 0 and speedup_val > 0 else 0.0
        eff_str = f"{eff:.1f}%" if eff > 0 else "-"
        print(
            f"{r.strategy:<12}{r.assignment:<12}{r.total_cpus:>6}"
            f"{r.group_splits:>4}{r.cpus_per_group:>5}"
            f"{r.wall_time_ms:>12.2f}{r.gflops:>12.2f}"
            f"{speedup_str:>10}{upper_str:>10}{eff_str:>8}",
        )
    print("=" * 112)


# ── 主入口 ──


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # —— 路由 / 任务分布 ——
    parser.add_argument(
        "--routing-mode",
        default="legacy",
        choices=["legacy", "uniform", "dirichlet", "grouped"],
        help="GEMM 分布生成模式：legacy 使用 --m-min/--m-max"
        "均匀采样（默认，保持旧行为）；其它三种调用"
        "simulate_moe_routing 生成真实路由分布",
    )
    parser.add_argument("--num-tokens", type=int, default=2048, help="prefill 总 token 数，仅用于非 legacy 模式")
    parser.add_argument("--num-experts", type=int, default=256, help="expert 总数，仅用于非 legacy 模式")
    parser.add_argument("--top-k", type=int, default=8, help="每 token 激活的 expert 数，仅用于非 legacy 模式")
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Dirichlet 浓度，alpha 越小越偏斜，仅用于dirichlet / grouped 模式"
    )
    parser.add_argument("--num-route-groups", type=int, default=8, help="grouped 模式的路由组数，默认 8")
    parser.add_argument("--top-m-groups", type=int, default=4, help="grouped 模式每 token 选的组数，默认 4")
    parser.add_argument(
        "--keep-zero-experts", action="store_true", help="保留 M=0 的 expert（退化为 M=1 的 GEMM），默认跳过"
    )
    # —— legacy 模式参数 ——
    parser.add_argument("--num-groups", type=int, default=160, help="[legacy] 组数（每组 2 个 GEMM），默认 160")
    parser.add_argument("--m-min", type=int, default=50, help="[legacy] M 随机范围下界，默认 50")
    parser.add_argument("--m-max", type=int, default=150, help="[legacy] M 随机范围上界，默认 150")
    # —— GEMM 形状 ——
    parser.add_argument("--k1", type=int, default=768, help="GEMM1 的 K，默认 768")
    parser.add_argument("--n1", type=int, default=5120, help="GEMM1 的 N，默认 5120")
    parser.add_argument("--k2", type=int, default=5120, help="GEMM2 的 K，默认 5120")
    parser.add_argument("--n2", type=int, default=384, help="GEMM2 的 N，默认 384")
    # —— 调度 / 资源 ——
    parser.add_argument(
        "--cpus",
        default="",
        help='手动指定可用 CPU 列表（形如 "0-79" 或 "0,1,2,...,79"）；未指定则读取 os.sched_getaffinity(0)',
    )
    parser.add_argument(
        "--group-splits",
        nargs="+",
        type=int,
        default=[1, 2, 4, 5, 8, 10, 20],
        help="grouped 策略的分组数列表，必须整除 len(cpus)；1 表示 sequential",
    )
    parser.add_argument(
        "--assignment",
        nargs="+",
        default=["roundrobin"],
        choices=["roundrobin", "lpt"],
        help="任务分配策略列表，可同时跑多种对比；默认 roundrobin",
    )
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"], help="输出 dtype，默认 bf16")
    parser.add_argument("--warmup", type=int, default=1, help="warmup 轮数，默认 1")
    parser.add_argument("--repeat", type=int, default=3, help="正式测量轮数，取最短 wall time，默认 3")
    parser.add_argument("--seed", type=int, default=0, help="随机种子，默认 0")
    parser.add_argument("--output-csv", default="bench_kai_moe_cpp.csv", help="CSV 输出路径")
    parser.add_argument("--show-histogram", action="store_true", help="打印 M 分布直方图（仅非 legacy 模式）")
    return parser


def _parse_cpu_list(s: str) -> Optional[List[int]]:
    """解析 "0-79" 或 "0,1,2,...,79" 或两者混合的字符串。"""
    if not s:
        return None
    cpus: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_str, hi_str = part.split("-", 1)
            lo, hi = int(lo_str), int(hi_str)
            cpus.extend(range(lo, hi + 1))
        else:
            cpus.append(int(part))
    return cpus


def main() -> int:
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(1)

    try:
        import fused_cpp  # type: ignore[import-not-found]
    except ImportError as exc:
        print(f"[ERROR] 无法导入 fused_cpp：{exc}", file=sys.stderr)
        return 1

    kai_ok = getattr(fused_cpp, "_supports_kai", False)
    is_aarch64 = platform.machine() in ("aarch64", "arm64")
    if not (kai_ok and is_aarch64):
        print(
            "[ERROR] KleidiAI 后端不可用（需要 AArch64 且已启用 KleidiAI 构建）",
            file=sys.stderr,
        )
        return 2

    mod = fused_cpp._C  # 直接取低层模块
    for name in ("create_kai_thread_pool", "destroy_kai_thread_pool", "create_kai_gemm_handler", "kai_gemm"):
        if not hasattr(mod, name):
            print(
                f"[ERROR] 当前已编译的 fused_cpp 不包含 {name}，请重新编译 C++ 扩展",
                file=sys.stderr,
            )
            return 3

    # 解析可用 CPU。
    cpus = _parse_cpu_list(args.cpus)
    if cpus is None:
        if hasattr(os, "sched_getaffinity"):
            cpus = sorted(os.sched_getaffinity(0))
        else:
            cpus = list(range(os.cpu_count() or 1))
    if not cpus:
        print("[ERROR] 可用 CPU 列表为空", file=sys.stderr)
        return 4

    for x in args.group_splits:
        if x < 1 or len(cpus) % x != 0:
            print(
                f"[ERROR] group_splits 中的 {x} 无法整除 len(cpus)={len(cpus)}",
                file=sys.stderr,
            )
            return 5

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16}
    out_dtype = dtype_map[args.dtype]

    # —— 根据 routing-mode 生成 GEMM specs ——
    if args.routing_mode == "legacy":
        specs = _build_specs(
            args.num_groups,
            args.m_min,
            args.m_max,
            args.k1,
            args.n1,
            args.k2,
            args.n2,
            args.seed,
        )
        print(
            f"[INFO] 分布: legacy uniform(M ∈ [{args.m_min}, {args.m_max}]), num_groups={args.num_groups}",
        )
    else:
        route_rng = random.Random(args.seed)
        if args.routing_mode == "uniform":
            counts = smr._routing_uniform(  # pylint: disable=protected-access
                args.num_tokens,
                args.num_experts,
                args.top_k,
                route_rng,
            )
        elif args.routing_mode == "dirichlet":
            counts = smr._routing_dirichlet(  # pylint: disable=protected-access
                args.num_tokens,
                args.num_experts,
                args.top_k,
                args.alpha,
                route_rng,
            )
        else:  # grouped
            counts = smr._routing_grouped(  # pylint: disable=protected-access
                args.num_tokens,
                args.num_experts,
                args.num_route_groups,
                args.top_m_groups,
                args.top_k,
                args.alpha,
                route_rng,
            )
        stats = smr._stats(counts)  # pylint: disable=protected-access
        smr._print_stats(  # pylint: disable=protected-access
            stats,
            f"路由分布 ({args.routing_mode})",
        )
        if args.show_histogram:
            smr._print_histogram(counts)  # pylint: disable=protected-access
        specs = _build_specs_from_routing(
            counts,
            args.k1,
            args.n1,
            args.k2,
            args.n2,
            keep_zero=args.keep_zero_experts,
        )
        print(
            f"[INFO] 分布: routing={args.routing_mode}, tokens="
            f"{args.num_tokens}, experts={args.num_experts}, "
            f"top_k={args.top_k}, 生成 {len(specs)} 个 GEMM "
            f"(active={stats.active_experts}/{args.num_experts})",
        )

    inputs, outputs = _build_inputs_and_outputs(specs, out_dtype, args.seed)

    env_info = {
        "pid": os.getpid(),
        "cpu_count": os.cpu_count(),
        "affinity_size": len(cpus),
        "affinity_preview": cpus[:4] + (["..."] if len(cpus) > 8 else []) + cpus[-4:],
        "machine": platform.machine(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "<unset>"),
    }
    print(f"[INFO] env: {env_info}")
    print(
        f"[INFO] 共 {len(specs)} 个 GEMM，total_cpus={len(cpus)}，dtype={args.dtype}",
    )

    results: List[StrategyResult] = []

    registry = HandlerRegistry(mod, args.seed)
    try:
        for x in args.group_splits:
            strategy = "sequential" if x == 1 else "grouped"
            # sequential 只跑一次（不同 assignment 结果相同）。
            assign_list = ["roundrobin"] if x == 1 else args.assignment
            for assign in assign_list:
                print(
                    f"[INFO] 运行 {strategy}(x={x}, assign={assign}) ({len(cpus) // x} CPUs/group) ...",
                )
                r = _bench_one(
                    strategy=strategy,
                    group_splits=x,
                    cpus=cpus,
                    specs=specs,
                    inputs=inputs,
                    outputs=outputs,
                    registry=registry,
                    fused_cpp_mod=mod,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    assignment=assign,
                )
                print(
                    f"       wall={r.wall_time_ms:.2f} ms, "
                    f"{r.gflops:.2f} GFLOP/s, "
                    f"speedup_upper={r.speedup_upper:.2f}x",
                )
                results.append(r)
    finally:
        registry.release_all()

    _write_csv(results, args.output_csv)
    _print_summary(results)
    print(f"\n[OK] CSV  -> {args.output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
