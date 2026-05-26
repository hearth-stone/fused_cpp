#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SDPA QKᵀ microkernel benchmark（跨机器评估用，独立 CLI 脚本）。

用途：
    在另一台机器上对比 QKᵀ 8×8 microkernel 的多个实现：
      - baseline（fp32 = 单累加器 dot product；bf16 = BFMMLA 主路径）
      - qk_ublock4（fp32 4×4 双向分块 + 16 累加器外积扇出）
      - qk_packk_inner（bf16，K 按 row-pair 拼成 [4 pair][E*2]，amortized）
      - qk_packk_seq  （bf16，K 完全按 kernel 访存顺序排成 [E/4][32]，
                        amortized；用 vld1q_u16_x4）
      - qk_packk_full（bf16，每次都 pack——pessimistic 下界）

调用模型：thread_local cache 的 `qk_packk_inner` / `qk_packk_seq` 在 microkernel
benchmark 框架的 inner-loop 里只 pack 一次（K 指针不变），后续 N-1 次跳过 pack。
这是 SDPA 真实场景中 K 在外层 pack 一次给 L/8 个 i-tile 共用的 amortize 上界。

前提：fused_cpp C++ 扩展已就地编译（``cd fused_cpp && python setup.py
build_ext --inplace``）。脚本只 import ``fused_cpp._C``，不需要 torch SDPA。

用法（最小）::

    python tests/bench_microkernel_qkt.py

带峰值参考 + 自定义 shape::

    python tests/bench_microkernel_qkt.py \\
        --peak-fp32 109 --peak-bf16-half 55 --peak-bf16-full 109 \\
        --shapes 192,128 512,512 1024,680 2048,2048 \\
        --iters 4000 --warmup 200 --runs 4

输出：correctness max_abs（vs scalar reference）+ GFLOPS 表 + 相对 baseline 的
speedup + 占指令峰值的百分比（如果传了 --peak-*）。退出码 0 = 全 pass，
非 0 = 至少一个 max_abs 超过容忍度。
"""
from __future__ import annotations

import argparse
import math
import sys
from typing import List, Optional, Tuple

# ── 实现 → 测试 dtype 的预定义矩阵 ───────────────────────────────────────
#
# 每个 impl 对应它**目标改动的 dtype**：
#   * baseline / scalar          —— 两 dtype 都跑（参考线）
#   * qk_ublock4                 —— fp32（fp32 路径重写，bf16 fall through）
#   * qk_packk_full / inner / seq —— bf16（bf16 路径替代 K layout，fp32 fall through）

IMPL_DTYPE_MATRIX: List[Tuple[str, Tuple[str, ...]]] = [
    ("baseline", ("fp32", "bf16")),
    ("qk_ublock4", ("fp32",)),
    ("qk_packk_full", ("bf16",)),
    ("qk_packk_inner", ("bf16",)),
    ("qk_packk_seq", ("bf16",)),
    ("qk_unroll2", ("bf16",)),
]

# correctness 容忍度：bf16 的 BFMMLA 与 scalar 参考累加序不一致，预期 < 0.1；
# fp32 的 fma 顺序差异预期 < 1e-4。
TOL_PER_DTYPE = {"fp32": 5e-3, "bf16": 1e-1}


def parse_shapes(spec: List[str]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for s in spec:
        try:
            e_str, sk_str = s.split(",")
            out.append((int(e_str), int(sk_str)))
        except Exception as exc:
            raise SystemExit(f"--shapes 格式错误（应为 'E,Sk'）: {s!r} ({exc})")
    return out


def load_C():
    try:
        from fused_cpp import _C  # noqa: WPS433
    except ImportError as exc:
        sys.stderr.write(
            "ERROR: 无法 import fused_cpp._C。在 fused_cpp/ 目录下先跑：\n"
            "    python setup.py build_ext --inplace\n"
            f"原始错误：{exc}\n"
        )
        sys.exit(2)
    return _C


def best_of(C, impl: str, dtype: str, E: int, Sk: int,
            iters: int, warmup: int, runs: int) -> float:
    """跑 ``runs`` 次 benchmark，返回 qkt_8x8 的最大 GFLOPS。"""
    best = 0.0
    for _ in range(runs):
        r = C.benchmark_microkernel(
            impl=impl, dtype=dtype, E=E, Sk=Sk,
            iterations=iters, warmup=warmup,
        )
        v = r.get("qkt_8x8_gflops")
        if v is None or not math.isfinite(v):
            continue
        best = max(best, v)
    return best


def main() -> int:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        "--shapes", nargs="+", default=["192,128", "512,512", "1024,680", "2048,2048"],
        help="(E,Sk) 列表，逗号分隔。默认覆盖 short / R1-like / long-ctx / extra-long。",
    )
    p.add_argument("--iters", type=int, default=4000,
                   help="每次 benchmark 的内层 iteration 数（默认 4000）")
    p.add_argument("--warmup", type=int, default=200,
                   help="warmup iteration 数（默认 200）")
    p.add_argument("--runs", type=int, default=4,
                   help="best-of-N 次数（默认 4）")
    p.add_argument("--peak-fp32", type=float, default=None,
                   help="本机 fp32 fmla 单核峰值 GFLOPS（用于 %% peak 列）")
    p.add_argument(
        "--peak-bf16-half", type=float, default=None,
        help="本机 BFMMLA half-rate 单核峰值 GFLOPS（baseline / packk_* 的参考）",
    )
    p.add_argument(
        "--peak-bf16-full", type=float, default=None,
        help="本机 BFMLALB/T full-rate 单核峰值 GFLOPS（信息性参考；当前 microkernel 不走这条）",
    )
    p.add_argument(
        "--skip-correctness", action="store_true",
        help="跳过正确性检查，只跑 GFLOPS（在已知 impl 正确的机器上加速 sweep）",
    )
    p.add_argument(
        "--correctness-shapes", nargs="+",
        default=["32,32", "192,128", "512,512", "1024,680"],
        help="正确性测试用的 (E,Sk) 列表",
    )
    args = p.parse_args()

    C = load_C()
    avail = list(C.list_microkernel_impls())
    print(f"available microkernel impls: {avail}\n")

    # 过滤出当前 .so 里实际存在的 impl。
    matrix = [
        (impl, dtypes)
        for impl, dtypes in IMPL_DTYPE_MATRIX
        if impl in avail
    ]
    print("test matrix:")
    for impl, dtypes in matrix:
        print(f"  {impl:18s} dtypes={list(dtypes)}")
    print()

    shapes = parse_shapes(args.shapes)
    correctness_shapes = parse_shapes(args.correctness_shapes)

    # ── 1. 正确性检查 ─────────────────────────────────────────────────
    n_fail = 0
    if not args.skip_correctness:
        print("=" * 78)
        print("CORRECTNESS (qkt_8x8_max_abs vs scalar reference, lower = better)")
        print("=" * 78)
        for E, Sk in correctness_shapes:
            print(f"  E={E:4d} Sk={Sk:4d}")
            for impl, dtypes in matrix:
                for dtype in dtypes:
                    r = C.validate_microkernel(impl=impl, dtype=dtype, E=E, Sk=Sk)
                    val = r["qkt_8x8_max_abs"]
                    tol = TOL_PER_DTYPE[dtype]
                    ok = val < tol
                    flag = "OK" if ok else f"FAIL (tol={tol})"
                    print(f"    {impl:18s} {dtype:5s} max_abs={val:.3e}  [{flag}]")
                    if not ok:
                        n_fail += 1
        print()

    if n_fail > 0:
        print(f"WARNING: {n_fail} correctness check(s) failed; bench data below "
              "may not be meaningful.\n")

    # ── 2. GFLOPS benchmark ────────────────────────────────────────────
    print("=" * 78)
    print(f"BENCHMARK (best of {args.runs} runs, iters={args.iters}, warmup={args.warmup})")
    print("=" * 78)

    # 按 dtype 分组打印（baseline 同时出现在 fp32 和 bf16）。
    for dtype in ("fp32", "bf16"):
        impls_for_dtype = [impl for impl, ds in matrix if dtype in ds]
        if not impls_for_dtype:
            continue
        peak = (
            args.peak_fp32 if dtype == "fp32"
            else args.peak_bf16_half
        )
        peak_label = (
            "fmla peak" if dtype == "fp32"
            else "BFMMLA half-rate peak"
        )
        print(f"\n--- dtype={dtype}"
              + (f"   (peak ref: {peak} GFLOPS = {peak_label})" if peak else "")
              + " ---")

        # 表头
        col_impls = "  ".join(f"{im:>16s}" for im in impls_for_dtype)
        col_speedups = "  ".join(f"{im+'_x':>16s}" for im in impls_for_dtype)
        col_pcts = (
            "  ".join(f"{im+'_%peak':>16s}" for im in impls_for_dtype)
            if peak else ""
        )
        print(f"  {'E':>5s} {'Sk':>5s} | {col_impls} | {col_speedups}"
              + (f" | {col_pcts}" if peak else ""))
        print("  " + "-" * (12 + len(col_impls) + 3 + len(col_speedups)
                            + (3 + len(col_pcts) if peak else 0)))

        for E, Sk in shapes:
            # 全部 impl 跑一遍，取每个的最大 GFLOPS
            gf = {}
            for impl in impls_for_dtype:
                gf[impl] = best_of(C, impl, dtype, E, Sk,
                                   args.iters, args.warmup, args.runs)
            base = gf.get("baseline", 0.0) or 1e-9

            row = f"  {E:5d} {Sk:5d} |"
            for impl in impls_for_dtype:
                row += f"  {gf[impl]:14.2f}  "
            row += " |"
            for impl in impls_for_dtype:
                row += f"  {(gf[impl] / base):14.2f}x "
            if peak:
                row += " |"
                for impl in impls_for_dtype:
                    row += f"  {(gf[impl] / peak * 100):14.1f}% "
            print(row)

    print()
    print("=" * 78)
    print("INTERPRETATION GUIDE")
    print("=" * 78)
    print("""\
fp32 路径
  baseline      = 64 个独立 dot product，单累加器 RAW dep chain → ~10 GFLOPS
  qk_ublock4    = 4×4 块 × 16 独立累加器外积扇出 + vpaddq 树 reduce → 接近 fmla peak

bf16 路径（BFMMLA 主路径下）
  baseline      = 8 vld1_u16(K) + 4 vcombine + 16 BFMMLA。86%~ BFMMLA half-peak
  qk_packk_full = 每次调用都 pack K（pessimistic 下界，实测总是负收益）
  qk_packk_inner= K 按 row-pair 拼 [4 pair][E*2]，thread_local 跳过重复 pack
  qk_packk_seq  = K 完全按 kernel 访存顺序 [E/4][32]，单条 vld1q_u16_x4 拿
                   一整 cacheline 64 字节。同样 amortized。

Apple Silicon 实测：packk_inner 与 packk_seq 几乎打平（~50 GFLOPS / 90% half-peak），
说明 BFMMLA 路径下的 K-LSU 已不是瓶颈，剩 ~8% 缺口在 Q 路径 + BFMMLA pipeline。
换机器后如果 packk_seq 显著超过 packk_inner，说明该机器 LSU 端口数 / multi-reg
load 实现跟 Apple Silicon 不同——这个差异本身就是有用信号。
""")

    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
