#!/usr/bin/env python3
"""Sweep route fragmentation while total FLOPs and active range-local B stay fixed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


MIB = 1 << 20


@dataclass(frozen=True)
class Point:
    split_factor: int
    replaced_teams: int
    fragment_routes: int
    task_count: int
    total_routes: int
    active_stage_bytes: int
    unique_weight_bytes: int


def parse_positive_ints(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"expected positive integers, got {text!r}")
    return values


def parse_nonnegative_ints(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError(f"expected non-negative integers, got {text!r}")
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_points(
    *,
    teams: int,
    base_routes: int,
    hidden: int,
    intermediate: int,
    split_factors: list[int],
    replaced_teams: list[int],
) -> list[Point]:
    if teams <= 0 or base_routes <= 0 or hidden <= 0 or intermediate <= 0:
        raise ValueError("teams and dimensions must be positive")
    if any(replaced < 0 or replaced > teams for replaced in replaced_teams):
        raise ValueError("replaced team count is outside [0, teams]")

    active_stage_bytes = teams * 2 * hidden * intermediate
    total_routes = teams * base_routes
    points: list[Point] = []
    for split_factor in split_factors:
        if base_routes % split_factor != 0 or (base_routes // split_factor) % 12 != 0:
            raise ValueError(f"split_factor={split_factor} does not produce an M12-aligned route")
        replacements = [0] if split_factor == 1 else replaced_teams
        for replaced in replacements:
            task_count = teams + replaced * (split_factor - 1)
            points.append(
                Point(
                    split_factor=split_factor,
                    replaced_teams=replaced,
                    fragment_routes=base_routes // split_factor,
                    task_count=task_count,
                    total_routes=total_routes,
                    active_stage_bytes=active_stage_bytes,
                    unique_weight_bytes=task_count * 6 * hidden * intermediate,
                )
            )
    return points


def benchmark_command(
    binary: Path,
    point: Point,
    *,
    teams: int,
    base_routes: int,
    hidden: int,
    intermediate: int,
    threads_per_team: int,
    schedule: str,
    copies: int,
    cpu_start: int,
    warmup: int,
    runs: int,
    stop_before_run: bool = False,
) -> list[str]:
    command = [
        str(binary),
        "--teams",
        str(teams),
        "--base-routes",
        str(base_routes),
        "--replaced-teams",
        str(point.replaced_teams),
        "--split-factor",
        str(point.split_factor),
        "--hidden",
        str(hidden),
        "--intermediate",
        str(intermediate),
        "--threads-per-team",
        str(threads_per_team),
        "--schedule",
        schedule,
        "--w13-ranges",
        "2",
        "--copies",
        str(copies),
        "--cpu-start",
        str(cpu_start),
        "--warmup",
        str(warmup),
        "--iters",
        str(runs),
        "--skip-check",
    ]
    if stop_before_run:
        command.append("--stop-before-run")
    return command


def run_command(command: list[str]) -> str:
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"benchmark failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}")
    return completed.stdout


def parse_output(output: str) -> tuple[dict[str, float | int], dict[str, float], dict[str, float]]:
    config_match = re.search(
        r"\bactive_stage_mib=([0-9.]+)\s+unique_weight_mib=([0-9.]+)\s+allocated_gib=([0-9.]+)",
        output,
    )
    if config_match is None:
        raise RuntimeError(f"benchmark config data missing: {output[-1000:]}")
    stages = {
        name: float(value)
        for name, value in re.findall(r"stage name=(\S+) median_max_team_sum_ms=([0-9.]+)", output)
    }
    result: dict[str, float | int] | None = None
    for line in output.splitlines():
        if line.startswith("RESULT_JSON "):
            result = json.loads(line.removeprefix("RESULT_JSON "))
    if result is None or len(stages) != 3:
        raise RuntimeError(f"benchmark result data missing: {output[-1000:]}")
    config = {
        "active_stage_mib": float(config_match.group(1)),
        "unique_weight_mib": float(config_match.group(2)),
        "allocated_gib": float(config_match.group(3)),
    }
    return result, stages, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(__file__).with_name("bench_fragmented_route_pipeline"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teams", type=int, default=24)
    parser.add_argument("--base-routes", type=int, default=2040)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--threads-per-team", type=int, default=4)
    parser.add_argument("--schedule", choices=("dynamic", "slot"), default="dynamic")
    parser.add_argument("--split-factors", type=parse_positive_ints, default=parse_positive_ints("1,2,5,10"))
    parser.add_argument("--replaced-teams", type=parse_nonnegative_ints, default=parse_nonnegative_ints("6,12,24"))
    parser.add_argument("--cpu-start", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.binary.is_file():
        raise FileNotFoundError(f"benchmark binary does not exist: {args.binary}")
    if args.threads_per_team <= 0 or args.cpu_start < 0 or args.warmup < 0 or args.runs <= 0:
        raise ValueError("thread/runs must be positive and cpu-start/warmup non-negative")
    points = build_points(
        teams=args.teams,
        base_routes=args.base_routes,
        hidden=args.hidden,
        intermediate=args.intermediate,
        split_factors=args.split_factors,
        replaced_teams=args.replaced_teams,
    )
    copies = args.warmup + args.runs
    rows: list[dict[str, object]] = []
    for index, point in enumerate(points, start=1):
        command = benchmark_command(
            args.binary,
            point,
            teams=args.teams,
            base_routes=args.base_routes,
            hidden=args.hidden,
            intermediate=args.intermediate,
            threads_per_team=args.threads_per_team,
            schedule=args.schedule,
            copies=copies,
            cpu_start=args.cpu_start,
            warmup=args.warmup,
            runs=args.runs,
        )
        print(
            f"[{index:02d}/{len(points):02d}] q={point.split_factor:<2} replaced={point.replaced_teams:<2} "
            f"fragment_m={point.fragment_routes:<4} tasks={point.task_count:<3} "
            f"active={point.active_stage_bytes / MIB:6.1f} MiB unique_B={point.unique_weight_bytes / MIB:7.1f} MiB",
            flush=True,
        )
        if args.dry_run:
            result: dict[str, float | int] = {}
            stages: dict[str, float] = {}
            config: dict[str, float] = {}
        else:
            output = run_command(command)
            result, stages, config = parse_output(output)
            print(
                f"         wall={float(result['median_ms']):8.3f} ms "
                f"throughput={float(result['tflops']):7.3f} TFLOP/s",
                flush=True,
            )
        rows.append(
            {
                **asdict(point),
                "replacement_fraction": point.replaced_teams / args.teams,
                "median_ms": result.get("median_ms"),
                "p99_ms": result.get("p99_ms"),
                "aggregate_tflops": result.get("tflops"),
                "stage_median_max_team_sum_ms": stages,
                "allocated_gib": config.get("allocated_gib"),
                "command": command,
            }
        )

    if not args.dry_run:
        baseline = next(row for row in rows if row["split_factor"] == 1 and row["replaced_teams"] == 0)
        baseline_ms = float(baseline["median_ms"])
        baseline_tflops = float(baseline["aggregate_tflops"])
        for row in rows:
            row["wall_vs_baseline"] = float(row["median_ms"]) / baseline_ms
            row["tflops_vs_baseline"] = float(row["aggregate_tflops"]) / baseline_tflops

    payload = {
        "schema_version": 1,
        "kind": "stage_range_fixed_active_b_route_fragmentation",
        "target": {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "os": platform.platform(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        },
        "kernel": {
            "entrypoint": "production fused SVE M12 W13 and W2 kernels",
            "w13_ranges": 2,
            "parallel_axis": "N",
            "binary": str(args.binary.resolve()),
            "binary_sha256": sha256_file(args.binary),
        },
        "measurement": {
            "teams": args.teams,
            "base_routes_per_team": args.base_routes,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "threads_per_team": args.threads_per_team,
            "schedule": args.schedule,
            "split_factors": args.split_factors,
            "replaced_teams": args.replaced_teams,
            "warmup": args.warmup,
            "runs": args.runs,
            "copies": copies,
            "weight_reuse": "one distinct task-weight copy per warmup and timed invocation",
            "invariants": [
                "teams * threads_per_team workers",
                "teams * base_routes total routes",
                "6 * total_routes * hidden * intermediate FLOPs",
                "teams * 2 * hidden * intermediate active W13/W2 range bytes",
            ],
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
