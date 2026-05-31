#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单核 microkernel 对比：QKᵀ 与 PV 在 bf16 / fp32 下的算力。

用途：诊断 SDPA 端到端单核 wall-clock 输给 PyTorch 的根因——具体是
QKᵀ 还是 PV 哪一步在 microkernel 层就慢。

跑法：
    OMP_NUM_THREADS=1 python tests/bench_pv_vs_qkt.py
    OMP_NUM_THREADS=1 python tests/bench_pv_vs_qkt.py --E 192 --Sk 128 --iters 30000

依赖：仅 fused_cpp._C；不依赖 pytest。
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, Set, Tuple

# ── 每个 trait 真正重写的 (op, dtype) 集合 ────────────────────────────────
#
# 该表与 csrc/sdpa_microkernels/impls/mk_*.h 中各 trait 的实际实现保持一致：
# 没有列出的 (op, dtype) 表示该 trait 在该 cell 直接 fall through 到 baseline
# 的 gemm_qkt_* / gemm_pv_*。bench 中这些 cell 应显示为 “—”，避免被误读成
# 「该 trait 自己的实现」。
#
# 新增 trait 时，请同步更新这里。op 名称：
#   "qkt_8x8"  — QKᵀ 主体 8×8
#   "qkt_8x4"  — QKᵀ 退化 8×4
#   "pv_8x8"   — PV 主体 8×8
TRAIT_OVERRIDES: Dict[str, Set[Tuple[str, str]]] = {
    "baseline": {
        ("qkt_8x8", "bf16"), ("qkt_8x8", "fp32"),
        ("qkt_8x4", "bf16"), ("qkt_8x4", "fp32"),
        ("pv_8x8",  "bf16"), ("pv_8x8",  "fp32"),
    },
    "scalar": {
        ("qkt_8x8", "bf16"), ("qkt_8x8", "fp32"),
        ("qkt_8x4", "bf16"), ("qkt_8x4", "fp32"),
        ("pv_8x8",  "bf16"), ("pv_8x8",  "fp32"),
    },
    "pquad": {
        ("pv_8x8", "fp32"),
        ("pv_8x8", "bf16"),
    },
    "qk_ublock4": {
        ("qkt_8x8", "fp32"), ("qkt_8x4", "fp32"),
    },
    "qk_packk_full":   {("qkt_8x8", "bf16")},
    "qk_packk_inner":  {("qkt_8x8", "bf16")},
    "qk_packk_seq":    {("qkt_8x8", "bf16")},
    "qk_unroll2":      {("qkt_8x8", "bf16")},
    "qk_packqk_seq":         {("qkt_8x8", "bf16")},
    "qk_packqk_seq4":        {("qkt_8x8", "bf16")},
    "qk_packqk_seq4_ptr":    {("qkt_8x8", "bf16")},
    "qk_packqk_seq4_bmajor": {("qkt_8x8", "bf16")},
    "qk_packqk_seq4_pipe_a": {("qkt_8x8", "bf16")},
    "qk_packqk_seq4_pipe_b": {("qkt_8x8", "bf16")},
    # 组合 trait：bf16 qkt 走 packqk_seq4_bmajor，bf16/fp32 pv 都走 pquad。
    "qk_packqk_seq4_bmajor_pv_pquad": {
        ("qkt_8x8", "bf16"),
        ("pv_8x8",  "fp32"),
        ("pv_8x8",  "bf16"),
    },
}


def _owns(impl: str, op: str, dtype: str) -> bool:
    """impl 是否在 (op, dtype) cell 重写了实现（而非 fall through 到 baseline）。

    未在 TRAIT_OVERRIDES 注册的 impl 一律视为「全部重写」，避免新加的 trait
    被默认隐藏（仍可通过更新表来显式标注 fall through cell）。"""
    if impl not in TRAIT_OVERRIDES:
        return True
    return (op, dtype) in TRAIT_OVERRIDES[impl]


def _fmt_gflops(value: float, owned: bool) -> str:
    return f"{value:>10.2f}" if owned else f"{'—':>10}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--E", type=int, default=192, help="head_dim (QKᵀ reduction)")
    p.add_argument("--Sk", type=int, default=128,
                   help="kv length within micro-tile (PV reduction)")
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument(
        "--impls",
        default="baseline,pquad,qk_packqk_seq4_bmajor,qk_packqk_seq4_bmajor_pv_pquad",
        help="comma-separated impl names",
    )
    p.add_argument(
        "--show-fallthrough",
        action="store_true",
        help="显示 fall-through 到 baseline 的 cell（默认显示为 —）",
    )
    args = p.parse_args()

    try:
        from fused_cpp import _C
    except ImportError as e:
        sys.stderr.write(f"failed to import fused_cpp._C: {e}\n")
        return 2

    avail = set(_C.list_microkernel_impls())
    impls = [x for x in args.impls.split(",") if x in avail]
    missing = [x for x in args.impls.split(",") if x not in avail]
    if missing:
        sys.stderr.write(f"WARNING: not registered, skipped: {missing}\n")

    print(f"\nE={args.E}  Sk={args.Sk}  iters={args.iters}  warmup={args.warmup}")
    print(f"  qkt_8x8 reduction = E   = {args.E}")
    print(f"  pv_8x8  reduction = Sk  = {args.Sk}")
    print(f"  fall-through 单元格 = '—'（即该 trait 在此 cell 直接调 baseline）")
    print("=" * 78)
    print(f"{'impl':<32} {'dtype':<6} "
          f"{'qkt_8x8':>10} {'pv_8x8':>10} {'qkt_8x4':>10}    PV/QKT")
    print(f"{'':<32} {'':<6} "
          f"{'GFLOPS':>10} {'GFLOPS':>10} {'GFLOPS':>10}")
    print("-" * 78)

    for impl in impls:
        for dtype in ("fp32", "bf16"):
            r = _C.benchmark_microkernel(
                impl=impl, dtype=dtype,
                E=args.E, Sk=args.Sk,
                iterations=args.iters, warmup=args.warmup,
            )
            qkt8_v = r["qkt_8x8_gflops"]
            qkt4_v = r["qkt_8x4_gflops"]
            pv_v   = r["pv_8x8_gflops"]

            qkt8_owned = args.show_fallthrough or _owns(impl, "qkt_8x8", dtype)
            qkt4_owned = args.show_fallthrough or _owns(impl, "qkt_8x4", dtype)
            pv_owned   = args.show_fallthrough or _owns(impl, "pv_8x8",  dtype)

            # PV/QKT ratio：两端任一 fall through 时无意义，显示为 —
            if qkt8_owned and pv_owned and qkt8_v > 0:
                ratio_s = f"{pv_v / qkt8_v:>5.2f}x"
            else:
                ratio_s = f"{'—':>6}"

            print(f"{impl:<32} {dtype:<6} "
                  f"{_fmt_gflops(qkt8_v, qkt8_owned)} "
                  f"{_fmt_gflops(pv_v, pv_owned)} "
                  f"{_fmt_gflops(qkt4_v, qkt4_owned)}   {ratio_s}")
    print("=" * 78)

    print("""
解读：
  * QKᵀ bf16 (BFMMLA 路径)：理论 peak 约 fp32 FMA 的 2×（half-rate BFMMLA）
    或 4×（full-rate BFMMLA）。Apple Silicon / Neoverse 多在 half-rate。
  * PV bf16 (widen + fp32 FMA 路径)：理论 peak 等于 fp32 FMA peak。
    所以 PV/QKT ratio 反映了 fp32-FMA 路径 vs BFMMLA 路径的相对速度。
    如果 ratio < 0.7，说明 PV 比 QKᵀ 慢明显，端到端 SDPA wall-clock
    被 PV step 拖住。
  * fp32 行作为对照（QKᵀ 和 PV 都走 fp32 FMA，应该 ratio ≈ 1）。
  * pquad trait 是 fp32 PV 优化版（quad load + lane FMA），bf16 PV
    没用 pquad；理想下一步是把 pquad 思路移植到 bf16 PV。
  * qk_packqk_seq4_bmajor_pv_pquad 是组合 trait：bf16 QKᵀ 走 packqk_seq4_bmajor，
    fp32 PV 走 pquad；其它 cell 仍 fall through 到 baseline（显示为 —）。
  * 加 --show-fallthrough 强制显示 fall-through cell 的实测数字，可与对应
    baseline 行做交叉验证（数值应非常接近）。
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
