#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L1-resident SDPA microkernel throughput benchmark.

This script drives the registered ``benchmark_microkernel`` C++ entrypoint
(`fused_cpp._C.benchmark_microkernel`) under a "single working set fits in
~half of L1 data cache" model, so the second iteration onward is fully L1
hit. It is intended as the fast / day-to-day complement to the standalone
binary ``bench_microkernel_l1`` (same numbers, but built without Python
overhead, suitable for ``perf record`` / ``objdump``).

工作集模型（单次调用，K = E for QKᵀ, K = Sk for P̂·V）:
  * qkt_8x8 / qkt_8x4 / qkt_tail :  Q[8 × E] + K[8 × E]  ≈ 16·E·sizeof(elt)
  * pv_8x8  / pv_tail            :  P̂[8 × Sk · 4] + V[Sk × 8 · sizeof(elt)]
                                    ≈ Sk · (32 + 8·sizeof(elt))

L1 探测：macOS 走 ``sysctl -n hw.l1dcachesize``；Linux 读
``/sys/devices/system/cpu/cpu0/cache/index0/size``；fallback 64 KB
（Neoverse N1/V1、Apple M1 efficient core 都在 64–128 KB 之间）。

Usage:
  python benchmarks/bench_microkernel_l1.py
  python benchmarks/bench_microkernel_l1.py --impl baseline --target-frac 0.5 \
      --iters 200000 --warmup 5000
  python benchmarks/bench_microkernel_l1.py --E 1024 --Sk 1024
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
from typing import List, Tuple

from fused_cpp import _C


def detect_l1d_bytes() -> int:
    """Return per-core L1d size in bytes; 64 KB fallback when unknown."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.l1dcachesize"],
                stderr=subprocess.DEVNULL,
            )
            value = int(out.strip())
            if value > 0:
                return value
        except (subprocess.CalledProcessError, ValueError, OSError):
            pass

    sys_path = "/sys/devices/system/cpu/cpu0/cache/index0/size"
    if os.path.exists(sys_path):
        try:
            text = open(sys_path).read().strip()
            multiplier = 1
            if text.endswith(("K", "k")):
                multiplier = 1024
                text = text[:-1]
            elif text.endswith(("M", "m")):
                multiplier = 1024 * 1024
                text = text[:-1]
            return int(text) * multiplier
        except (ValueError, OSError):
            pass

    return 64 * 1024


def derive_E(l1_bytes: int, sizeof_elt: int, target_frac: float) -> int:
    """E so that 16 · E · sizeof(elt) ≈ target_frac · L1, aligned to 4."""
    e_max = int(target_frac * l1_bytes / 16 / sizeof_elt)
    return max(8, (e_max // 4) * 4)


def derive_Sk(l1_bytes: int, sizeof_elt: int, target_frac: float) -> int:
    """Sk so that pv_8x8 working set ≈ target_frac · L1, aligned to 8."""
    # P̂ : 8 · Sk · 4 fp32   |   V : Sk · 8 · sizeof(elt)
    per_sk = 32 + 8 * sizeof_elt
    sk_max = int(target_frac * l1_bytes / per_sk)
    return max(8, (sk_max // 8) * 8)


def _format_bytes(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MiB"
    if n >= 1024:
        return f"{n / 1024:.2f} KiB"
    return f"{n} B"


def _microkernel_flops(m: int, n: int, k: int) -> int:
    """Return GEMM FLOPs using the standard 2*M*N*K convention."""
    return 2 * m * n * k


def _microkernel_bytes(op: str, m: int, n: int, k: int, sizeof_elt: int) -> int:
    """Return the theoretical bytes touched by one microkernel call.

    Conventions:
      * ``qkt_*`` computes ``scores = Q @ K^T`` and overwrites the fp32 output,
        so bytes = ``Q + K + scores_store``.
      * ``pv_*`` computes ``O += P_hat @ V`` and therefore performs a
        read-modify-write on the fp32 output,
        so bytes = ``P_hat + V + O_load + O_store``.

    This is intentionally different from the L1 working-set estimate printed
    above, which answers cache residency rather than arithmetic intensity.
    """
    if op.startswith("qkt"):
        q_bytes = m * k * sizeof_elt
        k_bytes = n * k * sizeof_elt
        scores_store_bytes = m * n * 4
        return q_bytes + k_bytes + scores_store_bytes

    if op.startswith("pv"):
        p_hat_bytes = m * k * 4
        v_bytes = k * n * sizeof_elt
        out_load_bytes = m * n * 4
        out_store_bytes = m * n * 4
        return p_hat_bytes + v_bytes + out_load_bytes + out_store_bytes

    raise ValueError(f"unknown microkernel op: {op}")


def _arithmetic_intensity(flops: int, bytes_per_call: int) -> float:
    """Return arithmetic intensity in FLOPs / byte."""
    if bytes_per_call <= 0:
        return 0.0
    return flops / bytes_per_call


def run_one_dtype(
    impl: str,
    dtype: str,
    sizeof_elt: int,
    l1_bytes: int,
    target_frac: float,
    iters: int,
    warmup: int,
    E_override: int,
    Sk_override: int,
) -> None:
    E = E_override if E_override > 0 else derive_E(l1_bytes, sizeof_elt, target_frac)
    Sk = Sk_override if Sk_override > 0 else derive_Sk(
        l1_bytes, sizeof_elt, target_frac
    )

    ws_qkt = 16 * E * sizeof_elt + 256
    ws_pv = (32 + 8 * sizeof_elt) * Sk + 256
    ratio_qkt = ws_qkt / l1_bytes
    ratio_pv = ws_pv / l1_bytes

    print(
        f"=== dtype={dtype}  impl={impl}  E={E}  Sk={Sk}\n"
        f"    qkt_* working set = {_format_bytes(ws_qkt)} "
        f"({ratio_qkt * 100:.1f}% of L1d)\n"
        f"    pv_*  working set = {_format_bytes(ws_pv)} "
        f"({ratio_pv * 100:.1f}% of L1d)"
    )

    r = _C.benchmark_microkernel(impl, dtype, E, Sk, iters, warmup)

    rows: List[Tuple[str, int, int, int, float, float]] = [
        ("qkt_8x8",  8, 8, E,  r["qkt_8x8_us"],  r["qkt_8x8_gflops"]),
        ("qkt_8x4",  8, 4, E,  r["qkt_8x4_us"],  r["qkt_8x4_gflops"]),
        ("qkt_tail", 5, 3, E,  r["qkt_tail_us"], r["qkt_tail_gflops"]),
        ("pv_8x8",   8, 8, Sk, r["pv_8x8_us"],   r["pv_8x8_gflops"]),
        ("pv_tail",  5, 3, Sk, r["pv_tail_us"],  r["pv_tail_gflops"]),
    ]
    print(
        f"  {'op':<10}{'M':>4}{'N':>4}{'K':>8}"
        f"{'FLOPs/op':>14}{'bytes/op':>14}{'AI(F/B)':>12}"
        f"{'us/iter':>14}{'GFLOPS':>12}"
    )
    for op, M, N, K, us, gflops in rows:
        flops = _microkernel_flops(M, N, K)
        bytes_per_call = _microkernel_bytes(op, M, N, K, sizeof_elt)
        ai = _arithmetic_intensity(flops, bytes_per_call)
        print(
            f"  {op:<10}{M:>4}{N:>4}{K:>8}"
            f"{flops:>14}{bytes_per_call:>14}{ai:>12.4f}"
            f"{us:>14.4f}{gflops:>12.2f}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "L1-resident microkernel throughput benchmark for "
            "fused_cpp._C.benchmark_microkernel"
        )
    )
    parser.add_argument(
        "--impl", default="baseline",
        help="microkernel impl name (default: baseline). "
        "Use _C.list_microkernel_impls() to list all enabled impls.",
    )
    parser.add_argument(
        "--dtypes", nargs="+", default=["bf16", "fp32"],
        choices=["bf16", "bfloat16", "fp32", "float32", "float"],
        help="dtypes to sweep (default: bf16 fp32)",
    )
    parser.add_argument(
        "--target-frac", type=float, default=0.5,
        help="working set / L1d ratio used to derive E and Sk (default 0.5)",
    )
    parser.add_argument(
        "--iters", type=int, default=100_000,
        help="benchmark iterations per op (default 100k)",
    )
    parser.add_argument(
        "--warmup", type=int, default=2_000,
        help="warmup iterations per op (default 2k)",
    )
    parser.add_argument(
        "--E", type=int, default=0,
        help="override head_dim E (0 = auto from L1)",
    )
    parser.add_argument(
        "--Sk", type=int, default=0,
        help="override Sk for pv_* (0 = auto from L1)",
    )
    parser.add_argument(
        "--l1-bytes", type=int, default=0,
        help="override detected L1d size (0 = auto)",
    )
    args = parser.parse_args()

    l1 = args.l1_bytes if args.l1_bytes > 0 else detect_l1d_bytes()
    impls = _C.list_microkernel_impls()
    print(
        f"detected L1d = {l1} bytes ({_format_bytes(l1)})\n"
        f"target_frac = {args.target_frac}\n"
        f"iters = {args.iters}, warmup = {args.warmup}\n"
        f"available impls = {impls}\n"
    )

    if args.impl not in impls:
        raise SystemExit(
            f"impl '{args.impl}' not registered; available: {impls}"
        )

    for dt in args.dtypes:
        sizeof_elt = 2 if dt in ("bf16", "bfloat16") else 4
        run_one_dtype(
            args.impl,
            dt,
            sizeof_elt,
            l1,
            args.target_frac,
            args.iters,
            args.warmup,
            args.E,
            args.Sk,
        )


if __name__ == "__main__":
    main()
