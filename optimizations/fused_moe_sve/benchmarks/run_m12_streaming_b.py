#!/usr/bin/env python3
"""Build and run the standalone M12 load-policy and K-block experiments."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_BINARY = BENCHMARK_DIR / "bench_m12_streaming_b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the production M12 LD1H body with streaming-prefetch, "
            "LDNT1H, and assembly K-block variants using one distinct cold "
            "packed-B matrix per call."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument("--shape", choices=("all", "w13", "w2"), default="all")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--m", type=int, default=12)
    parser.add_argument("--k", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--cpu", type=int, default=48)
    parser.add_argument("--numa-node", type=int, default=0)
    parser.add_argument("--cold-tail-mib", type=int, default=192)
    parser.add_argument("--weight-color", type=int, choices=range(4))
    parser.add_argument("--unique-scratch", action="store_true")
    parser.add_argument("--prewarm-a", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, help="also save benchmark stdout")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.k is None) != (args.n is None):
        raise ValueError("--k and --n must be supplied together")
    if args.jobs <= 0 or args.runs <= 0 or args.repeat <= 0 or args.warmup < 0:
        raise ValueError("jobs/runs/repeat must be positive and warmup non-negative")
    if args.m <= 0 or args.m % 12 != 0:
        raise ValueError("M must be a positive multiple of 12")
    if min(args.cpu, args.numa_node, args.cold_tail_mib) < 0:
        raise ValueError("CPU, NUMA node, and cold-tail size must be non-negative")

    if not args.no_build:
        subprocess.run(
            ["make", "-C", str(BENCHMARK_DIR), f"-j{args.jobs}", "bench_m12_streaming_b"],
            check=True,
        )
    if not args.binary.is_file():
        raise FileNotFoundError(f"benchmark binary does not exist: {args.binary}")

    command = [
        "numactl",
        f"--cpunodebind={args.numa_node}",
        f"--membind={args.numa_node}",
        "taskset",
        "-c",
        str(args.cpu),
        str(args.binary),
        "--shape",
        args.shape,
        "--variants",
        args.variants,
        "--m",
        str(args.m),
        "--warmup",
        str(args.warmup),
        "--runs",
        str(args.runs),
        "--cpu",
        str(args.cpu),
        "--cold-tail-mib",
        str(args.cold_tail_mib),
    ]
    if args.k is not None:
        command.extend(("--k", str(args.k), "--n", str(args.n)))
    if args.weight_color is not None:
        command.extend(("--weight-color", str(args.weight_color)))
    if args.unique_scratch:
        command.append("--unique-scratch")
    if args.prewarm_a:
        command.append("--prewarm-a")
    if args.check_only:
        command.append("--check-only")

    outputs: list[str] = []
    repeated_results: dict[tuple[str, str], list[dict[str, float | str]]] = {}
    for repeat in range(args.repeat):
        completed = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE)
        outputs.append(completed.stdout)
        if args.repeat > 1:
            print(f"repeat={repeat + 1}/{args.repeat}")
        print(completed.stdout, end="")
        for line in completed.stdout.splitlines():
            if not line.startswith("RESULT_JSON "):
                continue
            result = json.loads(line.removeprefix("RESULT_JSON "))
            key = (str(result["shape"]), str(result["variant"]))
            repeated_results.setdefault(key, []).append(result)

    summary = ""
    if args.repeat > 1:
        summary_lines = [
            "repeat_summary shape variant median_ms_range median_of_medians_ms "
            "paired_gain_range_pct median_paired_gain_pct"
        ]
        for (shape, variant), results in repeated_results.items():
            medians = [float(result["median_ms"]) for result in results]
            paired_gains = [float(result["paired_gain_pct"]) for result in results]
            summary_lines.append(
                f"REPEAT_SUMMARY shape={shape} variant={variant} "
                f"median_ms_range={min(medians):.4f}-{max(medians):.4f} "
                f"median_of_medians_ms={statistics.median(medians):.4f} "
                f"paired_gain_range_pct={min(paired_gains):.4f}-{max(paired_gains):.4f} "
                f"median_paired_gain_pct={statistics.median(paired_gains):.4f}"
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
