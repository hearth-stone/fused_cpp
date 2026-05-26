#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""微内核 GFLOPS 基准 CLI。

用法：
    python bench/sdpa/bench_microkernels.py
    python bench/sdpa/bench_microkernels.py --impls baseline --dtypes fp32
    python bench/sdpa/bench_microkernels.py --E 64 --Sk 64 --iters 50000 --json out.json

背后调用 ``_C.benchmark_microkernel(impl, dtype, E, Sk, iters, warmup)``，
该 C++ 入口对每个 impl 的 5 个 op 中的 `qkt_8x8 / qkt_8x4 / pv_8x8` 做
warmup → steady-state 计时 → GFLOPS = 2*M*N*K*iters / seconds，并附带
checksum 防止编译器优化掉调用。

不依赖 pytest；直接 ``python`` 跑即可。
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

try:
    from fused_cpp import _C  # type: ignore
except Exception as e:  # pragma: no cover
    sys.stderr.write(f"failed to import fused_cpp._C: {e}\n")
    sys.exit(2)


def _format_row(impl: str, dtype: str, op: str, r: Dict[str, float]) -> str:
    return (
        f"{impl:<12} {dtype:<5} {op:<8} "
        f"us={r[op + '_us']:>8.3f}  "
        f"GFLOPS={r[op + '_gflops']:>7.2f}  "
        f"sec={r[op + '_seconds']:>7.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Microkernel GFLOPS benchmark (qkt_8x8 / qkt_8x4 / pv_8x8)."
    )
    parser.add_argument(
        "--impls",
        default=None,
        help="comma-separated impl names (default: all registered).",
    )
    parser.add_argument(
        "--dtypes",
        default="fp32,bf16",
        help="comma-separated dtypes (default: fp32,bf16).",
    )
    parser.add_argument("--E", type=int, default=128, help="qk_head_dim.")
    parser.add_argument("--Sk", type=int, default=128, help="kv_seq within micro-tile.")
    parser.add_argument("--iters", type=int, default=20000, help="bench iterations.")
    parser.add_argument("--warmup", type=int, default=200, help="warmup iterations.")
    parser.add_argument(
        "--json",
        default=None,
        help="optional: write all results to this JSON file.",
    )
    args = parser.parse_args()

    available = list(_C.list_microkernel_impls())
    impls: List[str]
    if args.impls is None:
        impls = available
    else:
        impls = [s.strip() for s in args.impls.split(",") if s.strip()]
        unknown = sorted(set(impls) - set(available))
        if unknown:
            sys.stderr.write(
                f"unknown impl names: {unknown}; available: {available}\n"
            )
            return 2

    dtypes = [s.strip() for s in args.dtypes.split(",") if s.strip()]
    ops = ("qkt_8x8", "qkt_8x4", "pv_8x8")

    print(f"# E={args.E} Sk={args.Sk} iterations={args.iters} warmup={args.warmup}")
    print(f"# impls={impls} dtypes={dtypes}")
    print()

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for impl in impls:
        all_results.setdefault(impl, {})
        for dtype in dtypes:
            r = _C.benchmark_microkernel(
                impl=impl,
                dtype=dtype,
                E=args.E,
                Sk=args.Sk,
                iterations=args.iters,
                warmup=args.warmup,
            )
            all_results[impl][dtype] = dict(r)
            for op in ops:
                print(_format_row(impl, dtype, op, r))
            print()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"# wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
