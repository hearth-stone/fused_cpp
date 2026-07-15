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
      - qk_packqk_seq （bf16，Q 与 K 都按 kernel 访存顺序 pre-pack，
                        amortized；A/B 双侧都用 vld1q_u16_x4。仅 inner-loop
                        cache 命中场景；不能外推到端到端 SDPA。）
      - qk_packqk_seq4（bf16，同样 Q/K 双侧 seq pre-pack，但 A/B 双侧都用
                        4 条独立 vld1q_u16，验证 x4 multi-reg load 是否被串行化。）
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
from typing import List, Tuple

# ── 实现 → 测试 dtype × 是否接入 SDPA 主路径 ────────────────────────────
#
# 每个 impl 对应它**目标改动的 dtype**：
#   * baseline / scalar          —— 两 dtype 都跑（参考线）
#   * qk_ublock4                 —— fp32（fp32 路径重写，bf16 fall through）
#   * qk_packk_full / inner / seq —— bf16（bf16 路径替代 K layout，fp32 fall through）
#   * qk_packqk_seq              —— bf16（Q/K 双侧 pre-pack + vld1q_u16_x4）
#   * qk_packqk_seq4             —— bf16（Q/K 双侧 pre-pack + 4 条独立 vld1q_u16）
#   * qk_packqk_seq4_*           —— bf16（seq4 的指针递增 / B-major / pipe 调度变体）
#
# 第三个字段 `wired_to_sdpa`：该 microkernel 是否在 sdpa_flash2_neon_cache.cpp
# 中通过 REGISTER_SDPA_VERSION 注册成 `flash2_neon_cache_<name>` 入口，可被
# Python 侧 sdpa_versioned 直接调用。**评估专用 trait（每次需要外层手工 pack
# 才能用）一律 False**——它们的 GFLOPS 数据仅反映 microkernel 内核本身的
# 上限，不反映 SDPA 端到端收益。
#
# 与 csrc/sdpa_flash2_neon_cache.cpp 末尾的 REGISTER_SDPA_VERSION 列表保持一致。

# (impl_name, dtypes, wired_to_sdpa)
IMPL_DTYPE_MATRIX: List[Tuple[str, Tuple[str, ...], bool]] = [
    ("baseline", ("fp32", "bf16"), True),
    ("qk_ublock4", ("fp32",), True),
    ("qk_packk_full", ("bf16",), False),
    ("qk_packk_inner", ("bf16",), False),
    ("qk_packk_seq", ("bf16",), False),
    ("qk_packqk_seq", ("bf16",), False),
    ("qk_packqk_seq4", ("bf16",), False),
    ("qk_packqk_seq4_ptr", ("bf16",), False),
    ("qk_packqk_seq4_bmajor", ("bf16",), False),
    ("qk_packqk_seq4_pipe_a", ("bf16",), False),
    ("qk_packqk_seq4_pipe_b", ("bf16",), False),
    ("qk_unroll2", ("bf16",), False),
]

# 在表格 / 说明中给 eval-only impl 加上 ` *` 后缀，便于一眼看出。
EVAL_ONLY_SUFFIX = " *"


def _label(impl: str, wired: bool) -> str:
    """显示用：未接入 SDPA 的 microkernel 名后缀加 ` *`。"""
    return impl if wired else impl + EVAL_ONLY_SUFFIX


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


def best_of(C, impl: str, dtype: str, E: int, Sk: int, iters: int, warmup: int, runs: int) -> float:
    """跑 ``runs`` 次 benchmark，返回 qkt_8x8 的最大 GFLOPS。"""
    best = 0.0
    for _ in range(runs):
        r = C.benchmark_microkernel(
            impl=impl,
            dtype=dtype,
            E=E,
            Sk=Sk,
            iterations=iters,
            warmup=warmup,
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
        "--shapes",
        nargs="+",
        default=["192,128", "512,512", "1024,680", "2048,2048"],
        help="(E,Sk) 列表，逗号分隔。默认覆盖 short / R1-like / long-ctx / extra-long。",
    )
    p.add_argument("--iters", type=int, default=4000, help="每次 benchmark 的内层 iteration 数（默认 4000）")
    p.add_argument("--warmup", type=int, default=200, help="warmup iteration 数（默认 200）")
    p.add_argument("--runs", type=int, default=4, help="best-of-N 次数（默认 4）")
    p.add_argument("--peak-fp32", type=float, default=None, help="本机 fp32 fmla 单核峰值 GFLOPS（用于 %% peak 列）")
    p.add_argument(
        "--peak-bf16-half",
        type=float,
        default=None,
        help="本机 BFMMLA half-rate 单核峰值 GFLOPS（baseline / packk_* 的参考）",
    )
    p.add_argument(
        "--peak-bf16-full",
        type=float,
        default=None,
        help="本机 BFMLALB/T full-rate 单核峰值 GFLOPS（信息性参考；当前 microkernel 不走这条）",
    )
    p.add_argument(
        "--skip-correctness",
        action="store_true",
        help="跳过正确性检查，只跑 GFLOPS（在已知 impl 正确的机器上加速 sweep）",
    )
    p.add_argument(
        "--correctness-shapes",
        nargs="+",
        default=["32,32", "192,128", "512,512", "1024,680"],
        help="正确性测试用的 (E,Sk) 列表",
    )
    args = p.parse_args()

    C = load_C()
    avail = list(C.list_microkernel_impls())
    sdpa_versions = set(C.list_sdpa_versions())
    print(f"available microkernel impls: {avail}\n")

    # 过滤出当前 .so 里实际存在的 impl，并交叉验证 wired_to_sdpa 字段
    # 与 sdpa_versions 一致（防止 IMPL_DTYPE_MATRIX 与 .cpp 注册漂移）。
    matrix: List[Tuple[str, Tuple[str, ...], bool]] = []
    for impl, dtypes, wired in IMPL_DTYPE_MATRIX:
        if impl not in avail:
            continue
        actual_wired = f"flash2_neon_cache_{impl}" in sdpa_versions
        if actual_wired != wired:
            print(
                f"WARNING: IMPL_DTYPE_MATRIX 中 {impl!r} wired_to_sdpa={wired} "
                f"但实际 sdpa_versions={'YES' if actual_wired else 'NO'}; "
                f"以实际为准。请同步 IMPL_DTYPE_MATRIX 与 sdpa_flash2_neon_cache.cpp。"
            )
        matrix.append((impl, dtypes, actual_wired))

    eval_only_impls = [impl for impl, _, w in matrix if not w]
    print("test matrix (* = evaluation-only, NOT wired into SDPA dispatcher):")
    for impl, dtypes, wired in matrix:
        flag = "" if wired else "  [eval-only, requires external pack to use]"
        print(f"  {_label(impl, wired):20s} dtypes={list(dtypes)}{flag}")
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
            for impl, dtypes, wired in matrix:
                for dtype in dtypes:
                    r = C.validate_microkernel(impl=impl, dtype=dtype, E=E, Sk=Sk)
                    val = r["qkt_8x8_max_abs"]
                    tol = TOL_PER_DTYPE[dtype]
                    ok = val < tol
                    flag = "OK" if ok else f"FAIL (tol={tol})"
                    print(f"    {_label(impl, wired):20s} {dtype:5s} max_abs={val:.3e}  [{flag}]")
                    if not ok:
                        n_fail += 1
        print()

    if n_fail > 0:
        print(f"WARNING: {n_fail} correctness check(s) failed; bench data below may not be meaningful.\n")

    # ── 2. GFLOPS benchmark ────────────────────────────────────────────
    print("=" * 78)
    print(f"BENCHMARK (best of {args.runs} runs, iters={args.iters}, warmup={args.warmup})")
    print("=" * 78)
    if eval_only_impls:
        print(
            f"NOTE: 标 ` *` 的 impl 是评估专用 microkernel，**未接入 SDPA 主路径**\n"
            f"      ({', '.join(eval_only_impls)})。\n"
            f"      它们的 GFLOPS 反映 microkernel 内核本身上限（外层假设已 pre-pack），\n"
            f"      **不能直接外推为 SDPA 端到端收益**——真实 SDPA 主循环里 Q/K\n"
            f"      在每个 i-tile 都会变化、每次都需付出 pack overhead。"
        )

    # 按 dtype 分组打印（baseline 同时出现在 fp32 和 bf16）。
    for dtype in ("fp32", "bf16"):
        impls_for_dtype = [(impl, wired) for impl, ds, wired in matrix if dtype in ds]
        if not impls_for_dtype:
            continue
        peak = args.peak_fp32 if dtype == "fp32" else args.peak_bf16_half
        peak_label = "fmla peak" if dtype == "fp32" else "BFMMLA half-rate peak"
        print(f"\n--- dtype={dtype}" + (f"   (peak ref: {peak} GFLOPS = {peak_label})" if peak else "") + " ---")

        # 表头：未接入 SDPA 的 impl 名加 ` *` 后缀
        labels = [_label(im, w) for im, w in impls_for_dtype]
        col_impls = "  ".join(f"{lbl:>17s}" for lbl in labels)
        col_speedups = "  ".join(f"{lbl + '_x':>17s}" for lbl in labels)
        col_pcts = "  ".join(f"{lbl + '_%peak':>17s}" for lbl in labels) if peak else ""
        print(f"  {'E':>5s} {'Sk':>5s} | {col_impls} | {col_speedups}" + (f" | {col_pcts}" if peak else ""))
        print("  " + "-" * (12 + len(col_impls) + 3 + len(col_speedups) + (3 + len(col_pcts) if peak else 0)))

        for E, Sk in shapes:
            # 全部 impl 跑一遍，取每个的最大 GFLOPS
            gf = {}
            for impl, _ in impls_for_dtype:
                gf[impl] = best_of(C, impl, dtype, E, Sk, args.iters, args.warmup, args.runs)
            base = gf.get("baseline", 0.0) or 1e-9

            row = f"  {E:5d} {Sk:5d} |"
            for impl, _ in impls_for_dtype:
                row += f"  {gf[impl]:15.2f}  "
            row += " |"
            for impl, _ in impls_for_dtype:
                row += f"  {(gf[impl] / base):15.2f}x "
            if peak:
                row += " |"
                for impl, _ in impls_for_dtype:
                    row += f"  {(gf[impl] / peak * 100):15.1f}% "
            print(row)

    print()
    print("=" * 78)
    print("INTERPRETATION GUIDE")
    print("=" * 78)
    print("""\
图例：impl 名后的 ` *` 表示该 microkernel **未通过 REGISTER_SDPA_VERSION 接入
SDPA 主路径**——仅供 microkernel benchmark 评估。要让 SDPA 真正用上它，
需要在主循环外层显式 pack，并加 SDPA 版本绑定。

fp32 路径
  baseline      = 64 个独立 dot product，单累加器 RAW dep chain → ~10 GFLOPS
  qk_ublock4    = 4×4 块 × 16 独立累加器外积扇出 + vpaddq 树 reduce → 接近 fmla peak

bf16 路径（BFMMLA 主路径下）
  baseline       = 8 vld1_u16(K) + 4 vcombine + 16 BFMMLA。86%~ BFMMLA half-peak
  qk_packk_full *= 每次调用都 pack K（pessimistic 下界，实测总是负收益）
  qk_packk_inner*= K 按 row-pair 拼 [4 pair][E*2]，thread_local 跳过重复 pack
  qk_packk_seq  *= K 完全按 kernel 访存顺序 [E/4][32]，单条 vld1q_u16_x4 拿
                    一整 cacheline 64 字节。同样 amortized。
  qk_packqk_seq *= Q + K 都按 kernel 访存顺序 pre-pack，A/B 双侧都用 vld1q_u16_x4。
                    Q-LSU 上限测试：检查 baseline 的 8 vld1_u16 + 4 vcombine 是否
                    仍是瓶颈。⚠️  Q 在真实 SDPA 主路径不会预 pack（每 i-tile 滑窗即
                    miss），此值不可外推为端到端收益。
  qk_packqk_seq4*= 同样 Q + K 都按 seq 布局 pre-pack，但 A/B 双侧都改为
                    4 条独立 vld1q_u16，而不是 vld1q_u16_x4。用于验证目标核
                    对 multi-reg load 是否串行化；若 seq4 > seq，说明独立 load
                    更适合该机器。
  qk_packqk_seq4_ptr*    = seq4 + 显式 q_ptr/k_ptr 每轮 +=32，去掉 (e/4)*32 地址表达式。
  qk_packqk_seq4_bmajor* = seq4 + 16 条 BFMMLA 按 B 操作数复用顺序发射。
  qk_packqk_seq4_pipe_a* = seq4 + K 全 load，Q 分批 load 后立即计算对应 A 行组。
  qk_packqk_seq4_pipe_b* = seq4 + Q 全 load，K 分批 load 后立即计算对应 B 列组。
  qk_unroll2    *= BFMMLA 主路径 + 2-way k-unroll，摊薄 loop overhead。

Apple Silicon 实测：packk_inner 与 packk_seq 几乎打平（~50 GFLOPS / 90% half-peak），
说明 BFMMLA 路径下的 K-LSU 已不是瓶颈，剩 ~8% 缺口在 Q 路径 + BFMMLA pipeline。
换机器后如果 packk_seq 显著超过 packk_inner，说明该机器 LSU 端口数 / multi-reg
load 实现跟 Apple Silicon 不同——这个差异本身就是有用信号。
""")

    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
