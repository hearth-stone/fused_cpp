#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACL GEMM configure 开销基准测试。

测量以下场景的耗时：
1. 同一 M 值反复调用（命中 cached_M，无 reconfigure）
2. 两个 M 值交替调用（每次都触发 reconfigure）
3. 连续不同 M 值调用（模拟 MoE 场景，每次都 reconfigure）
4. 纯 create handler 的开销（含 configure + prepare）

通过对比场景 1 和场景 2 的差值，可以精确得到单次 reconfigure 的开销。
"""

import argparse
import csv
import platform
import sys
import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import fused_cpp


def check_platform() -> None:
    """检查平台是否支持 ACL GEMM。"""
    if platform.machine() not in ("aarch64", "arm64"):
        print("错误: 此测试仅在 AArch64 平台上运行")
        sys.exit(1)

    try:
        import fused_cpp

        if not fused_cpp._supports_acl:
            print("错误: ACL 后端不可用")
            sys.exit(1)
    except ImportError:
        print("错误: fused_cpp 模块未安装")
        sys.exit(1)


def bench_same_m(
    handler: "fused_cpp.ACLGEMMHandler",
    M: int,
    K: int,
    dtype: torch.dtype,
    num_iters: int,
) -> float:
    """场景 1: 同一 M 值反复调用（无 reconfigure）。

    :returns: 每次调用的平均耗时（微秒）。
    """
    import fused_cpp

    x = torch.randn(M, K, dtype=dtype)

    # 预热：先调用一次，确保 cached_M 已设置
    fused_cpp.acl_gemm(handler, x, None)

    start = time.perf_counter()
    for __ in range(num_iters):
        fused_cpp.acl_gemm(handler, x, None)
    elapsed = time.perf_counter() - start

    return (elapsed / num_iters) * 1e6  # 转为微秒


def bench_alternating_m(
    handler: "fused_cpp.ACLGEMMHandler",
    M1: int,
    M2: int,
    K: int,
    dtype: torch.dtype,
    num_iters: int,
) -> float:
    """场景 2: 两个 M 值交替调用（每次都触发 reconfigure）。

    :returns: 每次调用的平均耗时（微秒）。
    """
    import fused_cpp

    x1 = torch.randn(M1, K, dtype=dtype)
    x2 = torch.randn(M2, K, dtype=dtype)

    # 预热
    fused_cpp.acl_gemm(handler, x1, None)
    fused_cpp.acl_gemm(handler, x2, None)

    start = time.perf_counter()
    for i in range(num_iters):
        if i % 2 == 0:
            fused_cpp.acl_gemm(handler, x1, None)
        else:
            fused_cpp.acl_gemm(handler, x2, None)
    elapsed = time.perf_counter() - start

    return (elapsed / num_iters) * 1e6


def bench_sequential_different_m(
    handler: "fused_cpp.ACLGEMMHandler",
    m_values: list[int],
    K: int,
    dtype: torch.dtype,
    num_rounds: int,
) -> float:
    """场景 3: 连续不同 M 值调用（模拟 MoE 场景）。

    每轮遍历所有 m_values，统计每次调用的平均耗时。

    :returns: 每次调用的平均耗时（微秒）。
    """
    import fused_cpp

    # 预先创建所有输入张量
    inputs = [torch.randn(m, K, dtype=dtype) for m in m_values]

    # 预热一轮
    for x in inputs:
        fused_cpp.acl_gemm(handler, x, None)

    total_calls = 0
    start = time.perf_counter()
    for __ in range(num_rounds):
        for x in inputs:
            fused_cpp.acl_gemm(handler, x, None)
            total_calls += 1
    elapsed = time.perf_counter() - start

    return (elapsed / total_calls) * 1e6


def bench_create_handler(
    K: int,
    N: int,
    dtype: torch.dtype,
    fast_math: bool,
    num_iters: int,
) -> float:
    """场景 4: 测量 create_acl_gemm_handler 的开销（含 configure + prepare）。

    :returns: 每次创建的平均耗时（微秒）。
    """
    import fused_cpp

    weight = torch.randn(K, N, dtype=dtype)

    # 预热
    h = fused_cpp.create_acl_gemm(weight, fast_math=fast_math)
    del h

    start = time.perf_counter()
    handlers = []
    for __ in range(num_iters):
        h = fused_cpp.create_acl_gemm(weight, fast_math=fast_math)
        handlers.append(h)
    elapsed = time.perf_counter() - start

    # 清理
    for h in handlers:
        del h

    return (elapsed / num_iters) * 1e6


def run_benchmark(args: argparse.Namespace) -> list[dict]:
    """运行所有基准测试场景。"""
    import fused_cpp

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    results = []

    print("=" * 80)
    print("ACL GEMM Configure 开销基准测试")
    print(f"  K={args.K}, N={args.N}, dtype={args.dtype}, fast_math={args.fast_math}")
    print(f"  迭代次数: {args.num_iters}")
    print("=" * 80)

    # ── 场景 4: create handler 开销 ──
    print("\n[场景 4] create_acl_gemm_handler 开销 (configure + prepare)...")
    create_us = bench_create_handler(
        args.K,
        args.N,
        dtype,
        args.fast_math,
        num_iters=min(args.num_iters, 50),
    )
    print(f"  create handler: {create_us:.1f} μs/次")
    results.append(
        {
            "场景": "create_handler (configure+prepare)",
            "M": "-",
            "平均耗时(μs)": f"{create_us:.1f}",
            "reconfigure开销(μs)": "-",
        }
    )

    # 创建一个共享的 handler 用于后续测试
    weight = torch.randn(args.K, args.N, dtype=dtype)
    handler = fused_cpp.create_acl_gemm(weight, fast_math=args.fast_math)

    # ── 场景 1: 同一 M 值 ──
    print("\n[场景 1] 同一 M 值反复调用（无 reconfigure）...")
    same_m_results = {}
    for M in args.m_values:
        avg_us = bench_same_m(handler, M, args.K, dtype, args.num_iters)
        same_m_results[M] = avg_us
        print(f"  M={M:>4d}: {avg_us:.1f} μs/次")
        results.append(
            {
                "场景": "same_M (cached)",
                "M": str(M),
                "平均耗时(μs)": f"{avg_us:.1f}",
                "reconfigure开销(μs)": "0 (cached)",
            }
        )

    # ── 场景 2: 两个 M 值交替 ──
    print("\n[场景 2] 两个 M 值交替调用（每次触发 reconfigure）...")
    m_pairs = []
    for i in range(len(args.m_values)):
        for j in range(i + 1, len(args.m_values)):
            m_pairs.append((args.m_values[i], args.m_values[j]))

    for M1, M2 in m_pairs:
        avg_us = bench_alternating_m(
            handler,
            M1,
            M2,
            args.K,
            dtype,
            args.num_iters,
        )
        # 估算 reconfigure 开销 = 交替耗时 - 两个 M 的平均 cached 耗时
        cached_avg = (same_m_results[M1] + same_m_results[M2]) / 2
        reconf_overhead = avg_us - cached_avg
        print(f"  M={M1}<->{M2}: {avg_us:.1f} μs/次, reconfigure 开销 ≈ {reconf_overhead:.1f} μs")
        results.append(
            {
                "场景": "alternating_M",
                "M": f"{M1}<->{M2}",
                "平均耗时(μs)": f"{avg_us:.1f}",
                "reconfigure开销(μs)": f"{reconf_overhead:.1f}",
            }
        )

    # ── 场景 3: 连续不同 M（模拟 MoE） ──
    print("\n[场景 3] 连续不同 M 值（模拟 MoE 场景）...")

    # 3a: 少量 M 值（8 个专家）
    moe_m_values_small = [3, 17, 28, 41, 12, 35, 8, 22]
    avg_us_small = bench_sequential_different_m(
        handler,
        moe_m_values_small,
        args.K,
        dtype,
        num_rounds=max(args.num_iters // len(moe_m_values_small), 10),
    )
    # 对比基准：取 M=16 的 cached 耗时（如果有的话），否则取最接近的
    baseline_m = min(same_m_results.keys(), key=lambda m: abs(m - 16))
    baseline_us = same_m_results[baseline_m]
    overhead_small = avg_us_small - baseline_us
    print(
        f"  8 个不同 M {moe_m_values_small}: {avg_us_small:.1f} μs/次, "
        f"vs cached M={baseline_m} ({baseline_us:.1f} μs), "
        f"额外开销 ≈ {overhead_small:.1f} μs"
    )
    results.append(
        {
            "场景": "MoE_8experts",
            "M": str(moe_m_values_small),
            "平均耗时(μs)": f"{avg_us_small:.1f}",
            "reconfigure开销(μs)": f"{overhead_small:.1f}",
        }
    )

    # 3b: 更多 M 值（模拟更大 batch 下的 MoE）
    moe_m_values_large = list(range(1, 65))  # M=1~64，64 个不同值
    avg_us_large = bench_sequential_different_m(
        handler,
        moe_m_values_large,
        args.K,
        dtype,
        num_rounds=max(args.num_iters // len(moe_m_values_large), 5),
    )
    baseline_m32 = min(same_m_results.keys(), key=lambda m: abs(m - 32))
    baseline_us32 = same_m_results[baseline_m32]
    overhead_large = avg_us_large - baseline_us32
    print(
        f"  64 个不同 M [1..64]: {avg_us_large:.1f} μs/次, "
        f"vs cached M={baseline_m32} ({baseline_us32:.1f} μs), "
        f"额外开销 ≈ {overhead_large:.1f} μs"
    )
    results.append(
        {
            "场景": "MoE_64_different_M",
            "M": "[1..64]",
            "平均耗时(μs)": f"{avg_us_large:.1f}",
            "reconfigure开销(μs)": f"{overhead_large:.1f}",
        }
    )

    # ── 汇总 ──
    print("\n" + "=" * 80)
    print("汇总: reconfigure 开销估算")
    print("=" * 80)
    print(f"{'场景':<35s} {'M':<15s} {'平均耗时(μs)':<15s} {'reconfigure(μs)':<15s}")
    print("-" * 80)
    for r in results:
        print(f"{r['场景']:<35s} {r['M']:<15s} {r['平均耗时(μs)']:<15s} {r['reconfigure开销(μs)']:<15s}")

    del handler
    return results


def save_results(results: list[dict], output_path: str) -> None:
    """将结果保存到 CSV 文件。"""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n结果已保存到: {output_path}")


def main() -> None:
    """主入口。"""
    parser = argparse.ArgumentParser(
        description="ACL GEMM configure 开销基准测试",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=5120,
        help="输入维度 K（默认 4096）",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=768,
        help="输出维度 N（默认 14336，对应 LLaMA MoE 的 intermediate_size）",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "bf16"],
        help="数据类型（默认 bf16）",
    )
    parser.add_argument(
        "--fast-math",
        action="store_true",
        default=False,
        help="启用 fast_math 低精度加速路径",
    )
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=[1, 4, 16, 32, 64, 128],
        help="场景 1/2 中测试的 M 值列表（默认 1 4 16 32 64 128）",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=200,
        help="每个场景的迭代次数（默认 200）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="结果输出 CSV 文件路径（可选）",
    )
    args = parser.parse_args()

    check_platform()
    results = run_benchmark(args)

    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
