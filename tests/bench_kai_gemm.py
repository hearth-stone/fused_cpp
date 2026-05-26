#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KleidiAI GEMM FLOP/s 性能基准测试。

用法示例::

    python tests/bench_kai_gemm.py \\
        --shapes 1x4096x4096 32x4096x11008 128x4096x4096 \\
        --max-threads 1 2 4 \\
        --warmup 10 --repeat 100 \\
        --output bench_kai_gemm_results.csv \\
        --dtypes fp32 bf16

支持的 backend：
    - ``kai``  : 本项目的 KleidiAI GEMM（不同 max_threads 会多次测试）
    - ``torch``: ``torch.mm`` 作为基线（默认启用，可用 ``--no-compare-torch`` 关闭）
    - ``acl`` : ``fused_cpp.acl_gemm``（仅在 ACL 可用时启用）

输出：
    - CSV 文件：``backend,M,K,N,max_threads,output_dtype,avg_ms,min_ms,max_ms,gflops``
    - 终端 Summary：以 shape 为行、backend 为列的 GFLOP/s + 相对 torch 的加速比
"""
import argparse
import csv
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch


@dataclass
class BenchResult:
    """单次测试结果。"""

    backend: str
    m: int
    k: int
    n: int
    max_threads: int
    output_dtype: str
    avg_ms: float
    min_ms: float
    max_ms: float
    gflops: float
    status: str = "ok"  # ok / fail
    notes: str = ""


# ── 通用工具 ──

def _parse_shape(token: str) -> Tuple[int, int, int]:
    """解析 ``MxKxN`` 形如字符串为三元组。"""
    parts = token.lower().split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"非法的 shape: {token!r}，应为 'MxKxN' 形式"
        )
    try:
        m, k, n = (int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"非法的 shape: {token!r}，所有维度必须为整数"
        ) from exc
    if m < 0 or k <= 0 or n <= 0:
        raise argparse.ArgumentTypeError(
            f"非法的 shape: {token!r}，K、N 必须为正且 M 非负"
        )
    return m, k, n


def _parse_dtype(name: str) -> torch.dtype:
    table = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if name.lower() not in table:
        raise argparse.ArgumentTypeError(f"不支持的 dtype: {name!r}")
    return table[name.lower()]


def _dtype_name(dtype: torch.dtype) -> str:
    return {torch.float32: "fp32", torch.bfloat16: "bf16"}.get(dtype, str(dtype))


def _time_loop(
    fn: Callable[[], torch.Tensor],
    warmup: int,
    repeat: int,
) -> Tuple[float, float, float]:
    """对 ``fn()`` 执行 warmup + repeat 次计时；返回 (avg_ms, min_ms, max_ms)。"""
    for _ in range(warmup):
        fn()

    samples: List[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1e3)

    return (sum(samples) / len(samples), min(samples), max(samples))


def _compute_gflops(m: int, k: int, n: int, avg_ms: float) -> float:
    """FLOPS = 2 * M * N * K；转换为 GFLOP/s。"""
    if avg_ms <= 0:
        return 0.0
    flops = 2.0 * m * n * k
    return flops / (avg_ms / 1e3) / 1e9


def _relative_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    """最大相对误差（以 ref 的 max abs 作为归一化基）。"""
    diff = (out.to(torch.float32) - ref.to(torch.float32)).abs()
    denom = ref.to(torch.float32).abs().max().clamp_min(1e-6)
    return (diff.max() / denom).item()


# ── 参考实现 ──

def _torch_reference(x: torch.Tensor, weight: torch.Tensor,
                     out_dtype: torch.dtype) -> torch.Tensor:
    """使用 torch.mm 在 BF16 中间精度下计算参考结果。

    为了与 KAI GEMM 的语义一致（BF16 输入、BF16 累加指令），
    这里把 x/weight 先量化为 BF16 再提升为 FP32 做 MM，最后转到目标 dtype。
    """
    x_bf16 = x.to(torch.bfloat16).to(torch.float32)
    w_bf16 = weight.to(torch.bfloat16).to(torch.float32)
    return (x_bf16 @ w_bf16).to(out_dtype)


# ── 各 backend 的基准函数 ──

def _bench_kai(
    m: int, k: int, n: int,
    x: torch.Tensor, weight: torch.Tensor,
    out_dtype: torch.dtype,
    max_threads: int,
    warmup: int, repeat: int,
    fused_cpp_mod,
) -> BenchResult:
    dtype_name = _dtype_name(out_dtype)
    packed, pk, pn = fused_cpp_mod.kai_gemm_prepare(weight)
    handler = fused_cpp_mod.create_kai_gemm(packed, pk, pn)

    # 当 max_threads > 1 时创建独立线程池，cpu_ids 简单用 [0, 1, ..] 占位
    # （不绑具体核；KAIThreadPool 的 cpu_ids 仅用于声明总并发度）。
    import fused_cpp as _fcpp  # 延迟导入避免循环
    pool = None
    if max_threads > 1:
        pool = _fcpp.KAIThreadPool(list(range(max_threads)))

    try:
        # 正确性检查
        out = fused_cpp_mod.kai_gemm(
            handler, x, output_dtype=out_dtype, pool=pool,
        )
        ref = _torch_reference(x, weight, out_dtype)
        rel_err = _relative_error(out, ref)
        if rel_err > 0.02:
            return BenchResult(
                backend="kai", m=m, k=k, n=n, max_threads=max_threads,
                output_dtype=dtype_name,
                avg_ms=0.0, min_ms=0.0, max_ms=0.0, gflops=0.0,
                status="fail",
                notes=f"rel_err={rel_err:.4f} > 0.02",
            )

        avg, mn, mx = _time_loop(
            lambda: fused_cpp_mod.kai_gemm(
                handler, x, output_dtype=out_dtype, pool=pool,
            ),
            warmup, repeat,
        )
    finally:
        if pool is not None:
            pool.close()

    return BenchResult(
        backend="kai", m=m, k=k, n=n, max_threads=max_threads,
        output_dtype=dtype_name,
        avg_ms=avg, min_ms=mn, max_ms=mx,
        gflops=_compute_gflops(m, k, n, avg),
    )


def _bench_torch(
    m: int, k: int, n: int,
    x: torch.Tensor, weight: torch.Tensor,
    out_dtype: torch.dtype,
    warmup: int, repeat: int,
) -> BenchResult:
    dtype_name = _dtype_name(out_dtype)

    # 为了对 KAI 公平，torch 这里也使用 BF16 路径。
    x_bf16 = x.to(torch.bfloat16)
    w_bf16 = weight.to(torch.bfloat16)

    def _run() -> torch.Tensor:
        return (x_bf16 @ w_bf16).to(out_dtype)

    _run()  # 触发 lazy init
    avg, mn, mx = _time_loop(_run, warmup, repeat)
    return BenchResult(
        backend="torch", m=m, k=k, n=n, max_threads=0,
        output_dtype=dtype_name,
        avg_ms=avg, min_ms=mn, max_ms=mx,
        gflops=_compute_gflops(m, k, n, avg),
    )


def _bench_acl(
    m: int, k: int, n: int,
    x: torch.Tensor, weight: torch.Tensor,
    out_dtype: torch.dtype,
    warmup: int, repeat: int,
    fused_cpp_mod,
) -> Optional[BenchResult]:
    dtype_name = _dtype_name(out_dtype)

    # ACL 的 GemmMatrixMultiplyKernel 仅支持 FP32 输入/输出；
    # "BF16 加速" 在 ACL 中通过 fast_math=True 在 FP32 张量上启用（权重会被
    # 内部转换为 BF16 做中间累加）。因此对于 bf16 输出路径，我们仍然传 FP32
    # 张量给 ACL，仅通过 fast_math 区分 FP32 / BF16 两种加速模式。
    if out_dtype == torch.float32:
        fast_math = False
    elif out_dtype == torch.bfloat16:
        fast_math = True
    else:
        return None

    weight_acl = weight.to(torch.float32)
    x_acl = x.to(torch.float32)
    handler = fused_cpp_mod.create_acl_gemm(weight_acl, fast_math=fast_math)

    def _run() -> torch.Tensor:
        return fused_cpp_mod.acl_gemm(handler, x_acl, None)

    _run()
    avg, mn, mx = _time_loop(_run, warmup, repeat)
    return BenchResult(
        backend="acl", m=m, k=k, n=n, max_threads=0,
        output_dtype=dtype_name,
        avg_ms=avg, min_ms=mn, max_ms=mx,
        gflops=_compute_gflops(m, k, n, avg),
    )


# ── Summary 打印 ──

def _print_summary(results: List[BenchResult]) -> None:
    # 以 (M, K, N, dtype) 为行，(backend, max_threads) 为列。
    grouped: Dict[Tuple[int, int, int, str], Dict[str, BenchResult]] = {}
    for r in results:
        key = (r.m, r.k, r.n, r.output_dtype)
        col = f"{r.backend}/t={r.max_threads}" if r.backend == "kai" else r.backend
        grouped.setdefault(key, {})[col] = r

    if not grouped:
        return

    # 收集所有列名并排序，保证 torch 列在前方便对齐。
    col_order: List[str] = []
    for row in grouped.values():
        for col in row.keys():
            if col not in col_order:
                col_order.append(col)

    def _col_rank(col: str) -> Tuple[int, str]:
        order = {"torch": 0, "acl": 1}
        if col in order:
            return (order[col], col)
        return (2, col)

    col_order.sort(key=_col_rank)

    header = ["shape (MxKxN)", "dtype"] + col_order + [f"{c}/speedup" for c in col_order if c != "torch"]
    print()
    print("=" * 120)
    print("Summary: GFLOP/s (higher is better)")
    print("=" * 120)
    print("  ".join(f"{h:>18}" for h in header))

    for (m, k, n, dtype), row in sorted(grouped.items()):
        shape_str = f"{m}x{k}x{n}"
        base = row.get("torch")
        base_gflops = base.gflops if (base and base.status == "ok") else 0.0

        vals: List[str] = [shape_str, dtype]
        for col in col_order:
            r = row.get(col)
            if r is None:
                vals.append("-")
            elif r.status != "ok":
                vals.append("FAIL")
            else:
                vals.append(f"{r.gflops:.1f}")

        for col in col_order:
            if col == "torch":
                continue
            r = row.get(col)
            if r is None or r.status != "ok" or base_gflops <= 0:
                vals.append("-")
            else:
                vals.append(f"{r.gflops / base_gflops:.2f}x")
        print("  ".join(f"{v:>18}" for v in vals))
    print("=" * 120)


# ── CSV 写入 ──

def _write_csv(results: List[BenchResult], path: str) -> None:
    fields = ["backend", "M", "K", "N", "max_threads", "output_dtype",
              "avg_ms", "min_ms", "max_ms", "gflops", "status", "notes"]
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(fields)
        for r in results:
            writer.writerow([
                r.backend, r.m, r.k, r.n, r.max_threads, r.output_dtype,
                f"{r.avg_ms:.6f}", f"{r.min_ms:.6f}", f"{r.max_ms:.6f}",
                f"{r.gflops:.3f}", r.status, r.notes,
            ])


# ── 主入口 ──

def build_argparser() -> argparse.ArgumentParser:
    default_shapes = ["1x4096x4096", "32x4096x11008", "128x4096x4096"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes", nargs="+", default=default_shapes,
        help=f"测试 (M, K, N) 组合，默认 {default_shapes}",
    )
    parser.add_argument(
        "--max-threads", nargs="+", type=int, default=[1, 2, 4],
        help="KAI GEMM 最大线程数列表，默认 [1, 2, 4]",
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="预热迭代次数，默认 10",
    )
    parser.add_argument(
        "--repeat", type=int, default=100,
        help="正式测量迭代次数，默认 100",
    )
    parser.add_argument(
        "--output", default="bench_kai_gemm_results.csv",
        help="CSV 输出文件路径",
    )
    parser.add_argument(
        "--dtypes", nargs="+", default=["fp32", "bf16"],
        help="输出 dtype 列表，可选 fp32/bf16",
    )
    parser.add_argument(
        "--compare-torch", dest="compare_torch",
        action="store_true", default=True,
        help="是否对比 torch.mm（默认启用）",
    )
    parser.add_argument(
        "--no-compare-torch", dest="compare_torch",
        action="store_false",
        help="关闭 torch.mm 对比",
    )
    parser.add_argument(
        "--compare-acl", dest="compare_acl",
        action="store_true", default=True,
        help="是否对比 ACL GEMM（默认启用，不可用时自动跳过）",
    )
    parser.add_argument(
        "--no-compare-acl", dest="compare_acl",
        action="store_false",
        help="关闭 ACL 对比",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="torch 随机种子",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()

    torch.manual_seed(args.seed)
    shapes = [_parse_shape(s) for s in args.shapes]
    dtypes = [_parse_dtype(d) for d in args.dtypes]

    # 按需导入 fused_cpp；不可用时尽早报错。
    try:
        import fused_cpp  # type: ignore[import-not-found]
    except ImportError as exc:
        print(f"[ERROR] 无法导入 fused_cpp：{exc}", file=sys.stderr)
        return 1

    kai_ok = getattr(fused_cpp, "_supports_kai", False)
    acl_ok = getattr(fused_cpp, "_supports_acl", False)
    is_aarch64 = platform.machine() in ("aarch64", "arm64")

    if not (kai_ok and is_aarch64):
        print(
            "[ERROR] KleidiAI 后端不可用（需要 AArch64 且已启用 KleidiAI 构建）",
            file=sys.stderr,
        )
        return 2

    results: List[BenchResult] = []
    for m, k, n in shapes:
        for out_dtype in dtypes:
            x = torch.randn(m, k, dtype=torch.float32)
            weight = torch.randn(k, n, dtype=torch.float32)

            if args.compare_torch:
                results.append(_bench_torch(
                    m, k, n, x, weight, out_dtype,
                    args.warmup, args.repeat,
                ))

            if args.compare_acl and acl_ok:
                acl_r = _bench_acl(
                    m, k, n, x, weight, out_dtype,
                    args.warmup, args.repeat, fused_cpp,
                )
                if acl_r is not None:
                    results.append(acl_r)

            for max_threads in args.max_threads:
                results.append(_bench_kai(
                    m, k, n, x, weight, out_dtype, max_threads,
                    args.warmup, args.repeat, fused_cpp,
                ))

    _write_csv(results, args.output)
    _print_summary(results)
    print(f"\nCSV written to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
