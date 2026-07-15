#!/usr/bin/env python3
"""Sweep M-split vs N-split for the two MoE GEMM shapes across thread counts.

Times ONLY the middle-layer ``team_gemm`` (weights packed / A padded once, the
warmup+timed loop runs inside a single pooled job) via
``fused_cpp._C.fused_moe_bench_team_gemm``, so packing/scatter/routing do not
pollute the split comparison.

The two MoE GEMM shapes (per expert), for hidden size H and per-rank FFN F:
    w13:  A[M, H] x W13[2F, H]^T -> C[M, 2F]      => K = H,  N = 2F
    w2 :  A[M, F] x W2[H, F]^T   -> C[M, H]       => K = F,  N = H

On the AWS machine, pin the whole process, e.g.:
    OMP_NUM_THREADS=1 taskset -c 0-7 python bench_split_mn.py --threads 1,2,4,8

Output: a per-shape winner/ratio table, a crossover map, and CSV/JSON dumps.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fused_cpp.moe import _HAS_BF16_TILED_FUSED_MOE  # noqa: E402


def parse_int_list(text: str) -> List[int]:
    vals = [int(x) for x in text.split(",") if x.strip()]
    if not vals or any(v <= 0 for v in vals):
        raise ValueError(f"invalid positive int list: {text!r}")
    return vals


def bf16_normal(shape, gen, std):
    t = torch.empty(shape, dtype=torch.bfloat16)
    return t.normal_(mean=0.0, std=std, generator=gen)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hidden-size", type=int, default=4096, help="H")
    p.add_argument("--ffn-hidden-size", type=int, default=512, help="F per rank")
    p.add_argument(
        "--m-values",
        default="1,2,4,8,16,32,48,64,128,256,512,1024,2048",
        help="Routed rows per expert (M) to sweep.",
    )
    p.add_argument(
        "--threads",
        default="1,2,4,8",
        help="Team sizes (group_size = threads cooperating on one expert GEMM).",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--std", type=float, default=0.02)
    p.add_argument("--with-bias", action="store_true")
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-csv", type=Path, default=None)
    return p.parse_args()


def shapes_for(H: int, F: int) -> Dict[str, Dict[str, int]]:
    return {
        "w13": {"K": H, "N": 2 * F},
        "w2": {"K": F, "N": H},
    }


def main() -> int:
    args = parse_args()
    if not _HAS_BF16_TILED_FUSED_MOE:
        raise RuntimeError("BF16 tiled fused MoE backend unavailable")
    from fused_cpp import _C

    m_values = parse_int_list(args.m_values)
    thread_list = parse_int_list(args.threads)
    gen = torch.Generator().manual_seed(args.seed)
    torch.set_num_threads(1)

    shapes = shapes_for(args.hidden_size, args.ffn_hidden_size)
    rows: List[Dict[str, object]] = []

    print(
        f"machine={platform.machine()} H={args.hidden_size} "
        f"F={args.ffn_hidden_size} warmup={args.warmup} runs={args.runs}"
    )
    for stage, dims in shapes.items():
        K, N = dims["K"], dims["N"]
        weight = bf16_normal((N, K), gen, args.std)  # [N, K]
        bias = torch.randn(N, dtype=torch.float32, generator=gen) * 0.1 if args.with_bias else None
        print(f"\n=== stage={stage}  K={K}  N={N}  (N_blocks={N // 8}) ===")
        print("M      T     m_ms      n_ms      winner  n/m     m_gflops  n_gflops")
        for M in m_values:
            A = bf16_normal((M, K), gen, args.std)
            for T in thread_list:
                res: Dict[str, float] = {}
                for split in ("m", "n"):
                    times = _C.fused_moe_bench_team_gemm(A, weight, T, split, bias, args.warmup, args.runs)
                    res[split] = float(statistics.median(times))
                flops = 2.0 * M * K * N
                m_ms, n_ms = res["m"], res["n"]
                m_g = flops / (m_ms * 1e6) if m_ms > 0 else 0.0
                n_g = flops / (n_ms * 1e6) if n_ms > 0 else 0.0
                winner = "m" if m_ms < n_ms else "n"
                n_over_m = n_ms / m_ms if m_ms > 0 else float("nan")
                rows.append(
                    {
                        "stage": stage,
                        "K": K,
                        "N": N,
                        "M": M,
                        "threads": T,
                        "m_ms": m_ms,
                        "n_ms": n_ms,
                        "winner": winner,
                        "n_over_m": n_over_m,
                        "m_gflops": m_g,
                        "n_gflops": n_g,
                    }
                )
                mark = "" if T > 1 else "  (T=1: split irrelevant)"
                print(
                    f"{M:<6} {T:<5} {m_ms:9.4f} {n_ms:9.4f} {winner:<7} {n_over_m:6.3f}  {m_g:8.1f}  {n_g:8.1f}{mark}"
                )

    print("\n=== winner map (rows=M, cols=T>1); '.' = tie (<3%) ===")
    for stage in shapes:
        srows = [r for r in rows if r["stage"] == stage]
        ts = sorted({int(r["threads"]) for r in srows if int(r["threads"]) > 1})
        ms = sorted({int(r["M"]) for r in srows})
        print(f"\nstage={stage}  N={shapes[stage]['N']}")
        print("M\\T   " + "  ".join(f"{t:>4}" for t in ts))
        for M in ms:
            cells = []
            for t in ts:
                r = next(x for x in srows if int(x["M"]) == M and int(x["threads"]) == t)
                ratio = float(r["n_over_m"])
                if 0.97 <= ratio <= 1.03:
                    cells.append("   .")
                else:
                    cells.append(f"{r['winner']:>4}")
            print(f"{M:<5} " + "  ".join(cells))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "machine": platform.machine(),
                    "hidden_size": args.hidden_size,
                    "ffn_hidden_size": args.ffn_hidden_size,
                    "warmup": args.warmup,
                    "runs": args.runs,
                    "rows": rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote_json={args.output_json}")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "stage",
                    "K",
                    "N",
                    "M",
                    "threads",
                    "m_ms",
                    "n_ms",
                    "winner",
                    "n_over_m",
                    "m_gflops",
                    "n_gflops",
                ],
            )
            w.writeheader()
            w.writerows(rows)
        print(f"wrote_csv={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
