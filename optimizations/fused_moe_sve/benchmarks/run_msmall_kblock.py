#!/usr/bin/env python3
"""Build and run isolated M8/M4/M2/M1 assembly K-block experiments."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_BINARY = BENCHMARK_DIR / "bench_msmall_kblock"
HEIGHTS = ("8", "4", "2", "1")
VARIANTS = ("baseline", "kblock")


def csv_values(value: str, allowed: tuple[str, ...], name: str) -> list[str]:
    values = value.split(",")
    if not values or any(item not in allowed for item in values):
        raise argparse.ArgumentTypeError(
            f"{name} must be a comma-separated subset of {','.join(allowed)}"
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} contains duplicates")
    return values


def k_blocks(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("K blocks must be integers") from error
    if not values or any(item <= 0 or item % 8 != 0 for item in values):
        raise argparse.ArgumentTypeError("K blocks must be positive multiples of 8")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("K blocks contain duplicates")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure each small-M baseline or K-block variant in a separate "
            "process with a distinct cold packed-B matrix per invocation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument(
        "--m",
        type=lambda value: csv_values(value, HEIGHTS, "M"),
        default=list(HEIGHTS),
    )
    parser.add_argument(
        "--variants",
        type=lambda value: csv_values(value, VARIANTS, "variants"),
        default=list(VARIANTS),
    )
    parser.add_argument("--k-blocks", type=k_blocks, default=[1024])
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=51)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--cpu", type=int, default=48)
    parser.add_argument("--numa-node", type=int, default=0)
    parser.add_argument("--cold-tail-mib", type=int, default=192)
    parser.add_argument("--weight-color", type=int, choices=range(4))
    parser.add_argument("--unique-scratch", action="store_true")
    parser.add_argument("--prewarm-a", action="store_true")
    parser.add_argument("--output", type=Path, help="also save benchmark stdout")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(
        args.jobs,
        args.k,
        args.n,
        args.runs,
        args.repeat,
    ) <= 0:
        raise ValueError("jobs/K/N/runs/repeat must be positive")
    if min(args.warmup, args.cpu, args.numa_node, args.cold_tail_mib) < 0:
        raise ValueError("warmup/CPU/NUMA node/cold-tail size must be non-negative")

    if not args.no_build:
        subprocess.run(
            [
                "make",
                "-C",
                str(BENCHMARK_DIR),
                f"-j{args.jobs}",
                "bench_msmall_kblock",
            ],
            check=True,
        )
    if not args.binary.is_file():
        raise FileNotFoundError(f"benchmark binary does not exist: {args.binary}")

    outputs: list[str] = []
    repeated_results: dict[tuple[str, str], list[dict[str, float | int | str]]] = {}
    for repeat in range(args.repeat):
        for height in args.m:
            for variant in args.variants:
                variant_k_blocks = [args.k_blocks[0]] if variant == "baseline" else args.k_blocks
                for k_block in variant_k_blocks:
                    command = [
                        "numactl",
                        f"--cpunodebind={args.numa_node}",
                        f"--membind={args.numa_node}",
                        "taskset",
                        "-c",
                        str(args.cpu),
                        str(args.binary),
                        "--m",
                        height,
                        "--variant",
                        variant,
                        "--k",
                        str(args.k),
                        "--n",
                        str(args.n),
                        "--k-block",
                        str(k_block),
                        "--warmup",
                        str(args.warmup),
                        "--runs",
                        str(args.runs),
                        "--cpu",
                        str(args.cpu),
                        "--cold-tail-mib",
                        str(args.cold_tail_mib),
                    ]
                    if args.weight_color is not None:
                        command.extend(("--weight-color", str(args.weight_color)))
                    if args.unique_scratch:
                        command.append("--unique-scratch")
                    if args.prewarm_a:
                        command.append("--prewarm-a")
                    completed = subprocess.run(
                        command,
                        check=True,
                        text=True,
                        stdout=subprocess.PIPE,
                    )
                    label = f"repeat={repeat + 1}/{args.repeat} M={height} variant={variant}"
                    if variant == "kblock":
                        label += f" Kc={k_block}"
                    print(label)
                    print(completed.stdout, end="")
                    outputs.append(label + "\n" + completed.stdout)
                    for line in completed.stdout.splitlines():
                        if not line.startswith("RESULT_JSON "):
                            continue
                        result = json.loads(line.removeprefix("RESULT_JSON "))
                        key = (str(result["height"]), str(result["variant"]))
                        repeated_results.setdefault(key, []).append(result)

    baseline_medians: dict[str, float] = {}
    for (height, variant), results in repeated_results.items():
        if variant == "baseline_ld1h":
            baseline_medians[height] = statistics.median(
                float(result["median_ms"]) for result in results
            )

    summary_lines = [
        "repeat_summary height variant median_ms_range median_of_medians_ms "
        "gflops_median gain_vs_baseline_pct"
    ]
    for (height, variant), results in repeated_results.items():
        medians = [float(result["median_ms"]) for result in results]
        throughputs = [float(result["gflops"]) for result in results]
        median_ms = statistics.median(medians)
        baseline_ms = baseline_medians.get(height)
        gain = 0.0 if baseline_ms is None else (baseline_ms / median_ms - 1.0) * 100.0
        summary_lines.append(
            f"REPEAT_SUMMARY height={height} variant={variant} "
            f"median_ms_range={min(medians):.4f}-{max(medians):.4f} "
            f"median_of_medians_ms={median_ms:.4f} "
            f"gflops_median={statistics.median(throughputs):.4f} "
            f"gain_vs_baseline_pct={gain:.4f}"
        )
    summary = "\n".join(summary_lines) + "\n"
    print(summary, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(outputs) + summary, encoding="utf-8")
        print(f"wrote_output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
