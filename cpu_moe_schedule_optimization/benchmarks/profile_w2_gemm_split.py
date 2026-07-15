#!/usr/bin/env python3
"""Benchmark isolated MoE BF16 GEMMs with forced M/N split modes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from fused_cpp import bf16_linear  # noqa: E402


def parse_int_list(text: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"invalid positive integer list: {text!r}")
    return values


def parse_split_list(text: str) -> List[str]:
    values = [item.strip().lower() for item in text.split(",") if item.strip()]
    allowed = {"m", "n", "auto"}
    if not values or any(value not in allowed for value in values):
        raise ValueError(f"invalid split list {text!r}; allowed: m,n,auto")
    return values


def bf16_normal(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    std: float,
) -> torch.Tensor:
    tensor = torch.empty(shape, dtype=torch.bfloat16)
    return tensor.normal_(mean=0.0, std=std, generator=generator)


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("empty measurement list")
    return float(statistics.median(values))


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("empty measurement list")
    return float(statistics.fmean(values))


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        raise ValueError("empty measurement list")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return float(ordered[idx])


def measure_call(run, *, runs: int) -> List[float]:
    times_ms: List[float] = []
    for _ in range(runs):
        begin = time.perf_counter_ns()
        out = run()
        _ = float(out.flatten()[0])
        times_ms.append((time.perf_counter_ns() - begin) / 1e6)
    return times_ms


def set_split_env(split: str) -> None:
    if split == "auto":
        os.environ.pop("BF16_NEON_SPLIT", None)
    else:
        os.environ["BF16_NEON_SPLIT"] = split


def validate_output(
    x: torch.Tensor,
    weight: torch.Tensor,
    packed: bf16_linear.PreparedBF16LinearWeight,
    *,
    split: str,
    threads: int,
) -> None:
    set_split_env(split)
    actual = bf16_linear.linear(x, packed, out_dtype=torch.float32, nthreads=threads)
    expected = x.float() @ weight.float().T
    torch.testing.assert_close(actual, expected, atol=8e-2, rtol=8e-2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure an isolated MoE GEMM with "
            "BF16_NEON_SPLIT forced to m/n/auto. This uses the refs/i8gemm "
            "bf16_linear path, so it isolates the GEMM kernel from MoE "
            "routing, activation, scatter, and scheduler overhead."
        )
    )
    parser.add_argument(
        "--stage",
        choices=["w13", "w2"],
        default="w2",
        help=("MoE GEMM shape to measure. w13 is A[M,H] x W13[H,2F]; w2 is A[M,F] x W2[F,H]."),
    )
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--ffn-hidden-size", type=int, default=512)
    parser.add_argument(
        "--m-values",
        "--routes",
        dest="m_values",
        default="16,32,64,128,256,512,1024,1536,2048,3072,4096,6144,8192",
        help="Comma-separated M values, i.e. routed rows for one expert.",
    )
    parser.add_argument("--threads", default="1,2,4,8")
    parser.add_argument(
        "--splits",
        default="m,n",
        help="Comma-separated split modes to test: m,n,auto.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--std", type=float, default=0.01)
    parser.add_argument(
        "--keep-thread-clamp",
        action="store_true",
        help=(
            "Keep refs/i8gemm's BF16_NEON_CLAMP_THREADS behavior. By default "
            "the script disables it so requested thread counts are tested."
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Check each measured output against torch fp32 matmul.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not bf16_linear._supports_bf16_linear:
        raise RuntimeError("bf16_linear backend is unavailable")
    if args.hidden_size <= 0 or args.ffn_hidden_size <= 0:
        raise ValueError("hidden and FFN sizes must be positive")
    if args.hidden_size % 8 != 0 or args.ffn_hidden_size % 8 != 0:
        raise ValueError("hidden and FFN sizes must be multiples of 8")
    if args.warmup < 0 or args.runs <= 0:
        raise ValueError("warmup must be non-negative and runs positive")

    m_values = parse_int_list(args.m_values)
    threads_list = parse_int_list(args.threads)
    splits = parse_split_list(args.splits)

    torch.set_num_threads(1)
    if not args.keep_thread_clamp:
        os.environ["BF16_NEON_CLAMP_THREADS"] = "0"

    generator = torch.Generator().manual_seed(args.seed)
    if args.stage == "w13":
        k_dim = args.hidden_size
        n_dim = 2 * args.ffn_hidden_size
        shape_text = f"w13: MxH x Hx2F, H={args.hidden_size} F={args.ffn_hidden_size}"
    else:
        k_dim = args.ffn_hidden_size
        n_dim = args.hidden_size
        shape_text = f"w2: MxF x FxH, H={args.hidden_size} F={args.ffn_hidden_size}"

    weight = bf16_normal((n_dim, k_dim), generator=generator, std=args.std)
    packed = bf16_linear.prepare(weight)

    print(f"shape {shape_text}; output=float32")
    print("M      threads split median_ms mean_ms min_ms p90_ms gflops speedup_vs_t1_same_split")

    rows: List[Dict[str, object]] = []
    baseline_by_split_m: Dict[tuple[str, int], float] = {}
    for m_value in m_values:
        x = bf16_normal(
            (m_value, k_dim),
            generator=generator,
            std=args.std,
        )
        for threads in threads_list:
            for split in splits:
                if args.validate:
                    validate_output(x, weight, packed, split=split, threads=threads)

                set_split_env(split)

                def run() -> torch.Tensor:
                    return bf16_linear.linear(x, packed, out_dtype=torch.float32, nthreads=threads)

                for _ in range(args.warmup):
                    out = run()
                    _ = float(out.flatten()[0])

                times_ms = measure_call(run, runs=args.runs)
                median_ms = median(times_ms)
                key = (split, m_value)
                if threads == 1:
                    baseline_by_split_m[key] = median_ms
                baseline_ms = baseline_by_split_m.get(key, median_ms)
                flops = 2.0 * m_value * k_dim * n_dim
                gflops = flops / (median_ms * 1.0e6) if median_ms > 0.0 else 0.0
                row = {
                    "M": m_value,
                    "threads": threads,
                    "split": split,
                    "median_ms": median_ms,
                    "mean_ms": mean(times_ms),
                    "min_ms": min(times_ms),
                    "p90_ms": percentile(times_ms, 0.90),
                    "gflops": gflops,
                    "speedup_vs_t1_same_split": (baseline_ms / median_ms if median_ms > 0.0 else 0.0),
                    "times_ms": times_ms,
                }
                rows.append(row)
                print(
                    f"{m_value:<6} {threads:<7} {split:<5} "
                    f"{median_ms:9.3f} {row['mean_ms']:7.3f} "
                    f"{row['min_ms']:6.3f} {row['p90_ms']:6.3f} "
                    f"{gflops:7.1f} {row['speedup_vs_t1_same_split']:24.3f}",
                    flush=True,
                )

    by_case: Dict[tuple[int, int], List[Dict[str, object]]] = {}
    for row in rows:
        by_case.setdefault((int(row["M"]), int(row["threads"])), []).append(row)
    comparisons: List[Dict[str, object]] = []
    for (m_value, threads), case_rows in sorted(by_case.items()):
        best = min(case_rows, key=lambda row: float(row["median_ms"]))
        split_ms = {str(row["split"]): float(row["median_ms"]) for row in case_rows}
        m_ms = split_ms.get("m")
        n_ms = split_ms.get("n")
        comparisons.append(
            {
                "M": m_value,
                "threads": threads,
                "best_split": best["split"],
                "best_median_ms": best["median_ms"],
                "m_median_ms": m_ms,
                "n_median_ms": n_ms,
                "n_over_m": (n_ms / m_ms if m_ms is not None and n_ms is not None and m_ms > 0.0 else None),
                "m_over_n": (m_ms / n_ms if m_ms is not None and n_ms is not None and n_ms > 0.0 else None),
            }
        )

    print("\ncomparison M/N:")
    print("M      threads best m_ms n_ms n_over_m")
    for item in comparisons:
        if item["m_median_ms"] is None or item["n_median_ms"] is None:
            continue
        print(
            f"{item['M']:<6} {item['threads']:<7} {item['best_split']:<4} "
            f"{item['m_median_ms']:7.3f} {item['n_median_ms']:7.3f} "
            f"{item['n_over_m']:8.3f}",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "shape": {
            "stage": args.stage,
            "hidden_size": args.hidden_size,
            "ffn_hidden_size": args.ffn_hidden_size,
            "K": k_dim,
            "N": n_dim,
            "output_dtype": "float32",
            "gemm": args.stage,
        },
        "warmup": args.warmup,
        "runs": args.runs,
        "splits": splits,
        "thread_clamp": "kept" if args.keep_thread_clamp else "disabled",
        "rows": rows,
        "comparisons": comparisons,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote_json={args.output_json}")
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "M",
                "threads",
                "split",
                "median_ms",
                "mean_ms",
                "min_ms",
                "p90_ms",
                "gflops",
                "speedup_vs_t1_same_split",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row[key] for key in fieldnames})
        print(f"wrote_csv={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
