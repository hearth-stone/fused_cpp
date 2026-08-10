#!/usr/bin/env python3
"""Measure full-stage packed-B owner stripes under several thread mappings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


MIB = 1 << 20
DEFAULT_THREADS = "1,2,4,8,16,32,64,96"
DEFAULT_ROUTES = "192,2040"
DEFAULT_EXPERIMENTS = "nsplit,expert-fixed,expert-total"
DEFAULT_TEAM_EXPERTS = "1,2,3,4,6,8,12"


@dataclass(frozen=True)
class Point:
    experiment: str
    routes: int
    threads: int
    experts: int
    threads_per_expert: int
    intermediate: int
    requested_stage_mib: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_positive_ints(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"expected positive integers, got {text!r}")
    return values


def parse_positive_floats(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError(f"expected positive values, got {text!r}")
    return values


def parse_experiments(text: str) -> list[str]:
    allowed = {
        "nsplit",
        "expert-fixed",
        "expert-total",
        "team-fixed-route",
        "team-fixed-work",
    }
    values = [item.strip() for item in text.split(",") if item.strip()]
    unknown = sorted(set(values) - allowed)
    if not values or unknown:
        raise argparse.ArgumentTypeError(f"invalid experiments: {', '.join(unknown) if unknown else text!r}")
    return list(dict.fromkeys(values))


def stage_bytes(hidden: int, intermediate: int) -> int:
    """Return the larger full packed-B stage (W13) for one expert."""
    return 4 * hidden * intermediate


def intermediate_for_exact_stage(hidden: int, requested_mib: float, n_tile: int) -> int:
    requested_bytes = round(requested_mib * MIB)
    denominator = 4 * hidden
    if requested_bytes % denominator != 0:
        raise ValueError(f"{requested_mib} MiB is not exact for hidden={hidden}")
    intermediate = requested_bytes // denominator
    if intermediate < n_tile or intermediate % n_tile != 0:
        raise ValueError(f"intermediate={intermediate} must be a positive multiple of n_tile={n_tile}")
    return intermediate


def intermediate_for_total_stage(hidden: int, total_mib: float, threads: int, n_tile: int) -> int:
    bytes_per_tile_per_expert = 4 * hidden * n_tile
    ideal_tiles = total_mib * MIB / (threads * bytes_per_tile_per_expert)
    return max(1, math.floor(ideal_tiles + 0.5)) * n_tile


def max_nsplit_thread_stage_bytes(hidden: int, intermediate: int, threads: int, n_tile: int) -> int:
    w13_tiles = 2 * intermediate // n_tile
    w2_tiles = hidden // n_tile
    max_w13_tiles = (w13_tiles + threads - 1) // threads
    max_w2_tiles = (w2_tiles + threads - 1) // threads
    w13_bytes = max_w13_tiles * n_tile * hidden * 2
    w2_bytes = max_w2_tiles * n_tile * intermediate * 2
    return max(w13_bytes, w2_bytes)


def build_points(
    *,
    experiments: list[str],
    routes: list[int],
    threads: list[int],
    hidden: int,
    n_tile: int,
    nsplit_stage_mib: list[float],
    expert_stage_mib: list[float],
    total_stage_mib: list[float],
    max_total_stage_mib: float,
) -> list[Point]:
    points: list[Point] = []
    for route_count in routes:
        if "nsplit" in experiments:
            for requested_mib in nsplit_stage_mib:
                intermediate = intermediate_for_exact_stage(hidden, requested_mib, n_tile)
                max_threads = min(hidden // n_tile, 2 * intermediate // n_tile)
                for thread_count in threads:
                    if thread_count <= max_threads:
                        points.append(
                            Point("nsplit", route_count, thread_count, 1, thread_count, intermediate, requested_mib)
                        )
        if "expert-fixed" in experiments:
            for requested_mib in expert_stage_mib:
                intermediate = intermediate_for_exact_stage(hidden, requested_mib, n_tile)
                for thread_count in threads:
                    if thread_count * requested_mib <= max_total_stage_mib + 1.0e-9:
                        points.append(
                            Point(
                                "expert-fixed",
                                route_count,
                                thread_count,
                                thread_count,
                                1,
                                intermediate,
                                requested_mib,
                            )
                        )
        if "expert-total" in experiments:
            for requested_mib in total_stage_mib:
                for thread_count in threads:
                    intermediate = intermediate_for_total_stage(hidden, requested_mib, thread_count, n_tile)
                    points.append(
                        Point(
                            "expert-total",
                            route_count,
                            thread_count,
                            thread_count,
                            1,
                            intermediate,
                            requested_mib,
                        )
                    )
    return points


def build_team_points(
    *,
    experiments: list[str],
    expert_counts: list[int],
    total_threads: int,
    fixed_route: int,
    total_routes: int,
    hidden: int,
    n_tile: int,
    stage_mib: float,
) -> list[Point]:
    team_experiments = [
        experiment
        for experiment in experiments
        if experiment in {"team-fixed-route", "team-fixed-work"}
    ]
    if not team_experiments:
        return []

    intermediate = intermediate_for_exact_stage(hidden, stage_mib, n_tile)
    max_threads_per_expert = min(hidden // n_tile, intermediate // n_tile)
    points: list[Point] = []
    for experiment in team_experiments:
        for experts in expert_counts:
            if total_threads % experts != 0:
                raise ValueError(f"total_threads={total_threads} is not divisible by experts={experts}")
            threads_per_expert = total_threads // experts
            if threads_per_expert > max_threads_per_expert:
                raise ValueError(
                    f"threads_per_expert={threads_per_expert} exceeds the available N tiles "
                    f"({max_threads_per_expert}) for experts={experts}"
                )
            if experiment == "team-fixed-route":
                routes = fixed_route
            else:
                if total_routes % experts != 0:
                    raise ValueError(f"total_routes={total_routes} is not divisible by experts={experts}")
                routes = total_routes // experts
            if routes % 12 != 0:
                raise ValueError(f"routes={routes} must be a multiple of the SVE M12 row tile")
            points.append(
                Point(
                    experiment,
                    routes,
                    total_threads,
                    experts,
                    threads_per_expert,
                    intermediate,
                    stage_mib,
                )
            )
    return points


def parse_config(output: str) -> tuple[int, float]:
    match = re.search(r"\bn_tile=(\d+).*\ballocated_gib=([0-9.]+)", output)
    if match is None:
        raise RuntimeError(f"benchmark config line missing from output: {output[-1000:]}")
    return int(match.group(1)), float(match.group(2))


def parse_benchmark_output(output: str) -> tuple[dict[str, float | str], dict[str, float], float]:
    _, allocated_gib = parse_config(output)
    stages: dict[str, float] = {}
    for name, value in re.findall(
        r"stage variant=production_fused name=(\S+) median_max_team_ms=([0-9.]+)",
        output,
    ):
        stages[name] = float(value)
    result: dict[str, float | str] | None = None
    for line in output.splitlines():
        if line.startswith("RESULT_JSON "):
            candidate = json.loads(line.removeprefix("RESULT_JSON "))
            if candidate.get("variant") == "production_fused":
                result = candidate
    if result is None or not stages:
        raise RuntimeError(f"production fused result missing from output: {output[-1000:]}")
    return result, stages, allocated_gib


def benchmark_command(
    binary: Path,
    point: Point,
    *,
    hidden: int,
    copies: int,
    cpu_start: int,
    warmup: int,
    runs: int,
) -> list[str]:
    return [
        str(binary),
        "--experts",
        str(point.experts),
        "--m",
        str(point.routes),
        "--h",
        str(hidden),
        "--f",
        str(point.intermediate),
        "--threads-per-expert",
        str(point.threads_per_expert),
        "--w13-ranges",
        "1",
        "--copies",
        str(copies),
        "--cpu-start",
        str(cpu_start),
        "--warmup",
        str(warmup),
        "--iters",
        str(runs),
        "--variant",
        "fused",
        "--skip-check",
    ]


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


def probe_n_tile(binary: Path, cpu_start: int) -> int:
    point = Point("probe", 12, 1, 1, 1, 64, 0.5)
    output = run_command(
        benchmark_command(
            binary,
            point,
            hidden=64,
            copies=1,
            cpu_start=cpu_start,
            warmup=0,
            runs=1,
        )
    )
    n_tile, _ = parse_config(output)
    return n_tile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(__file__).with_name("bench_unfused_pipeline"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiments", type=parse_experiments, default=parse_experiments(DEFAULT_EXPERIMENTS))
    parser.add_argument("--threads", type=parse_positive_ints, default=parse_positive_ints(DEFAULT_THREADS))
    parser.add_argument("--routes", type=parse_positive_ints, default=parse_positive_ints(DEFAULT_ROUTES))
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--n-tile", type=int, default=0, help="0 probes the production binary")
    parser.add_argument("--nsplit-stage-mib", type=parse_positive_floats, default=parse_positive_floats("4,16,64"))
    parser.add_argument("--expert-stage-mib", type=parse_positive_floats, default=parse_positive_floats("0.5,2"))
    parser.add_argument("--total-stage-mib", type=parse_positive_floats, default=parse_positive_floats("32,64,96"))
    parser.add_argument("--max-total-stage-mib", type=float, default=192.0)
    parser.add_argument("--team-experts", type=parse_positive_ints, default=parse_positive_ints(DEFAULT_TEAM_EXPERTS))
    parser.add_argument("--team-total-threads", type=int, default=96)
    parser.add_argument("--team-route", type=int, default=2040)
    parser.add_argument("--team-total-routes", type=int, default=2304)
    parser.add_argument("--team-stage-mib", type=float, default=16.0)
    parser.add_argument("--cpu-start", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.binary.is_file():
        raise FileNotFoundError(f"benchmark binary does not exist: {args.binary}")
    if args.hidden <= 0 or args.cpu_start < 0 or args.warmup < 0 or args.runs <= 0:
        raise ValueError("hidden/runs must be positive and cpu-start/warmup non-negative")
    if args.max_total_stage_mib <= 0.0:
        raise ValueError("max-total-stage-mib must be positive")
    if (
        args.team_total_threads <= 0
        or args.team_route <= 0
        or args.team_total_routes <= 0
        or args.team_stage_mib <= 0.0
    ):
        raise ValueError("team thread, route, and stage parameters must be positive")
    n_tile = args.n_tile or probe_n_tile(args.binary, args.cpu_start)
    if n_tile <= 0 or args.hidden % n_tile != 0:
        raise ValueError(f"hidden={args.hidden} must be divisible by n_tile={n_tile}")
    points = build_points(
        experiments=args.experiments,
        routes=args.routes,
        threads=args.threads,
        hidden=args.hidden,
        n_tile=n_tile,
        nsplit_stage_mib=args.nsplit_stage_mib,
        expert_stage_mib=args.expert_stage_mib,
        total_stage_mib=args.total_stage_mib,
        max_total_stage_mib=args.max_total_stage_mib,
    )
    points.extend(
        build_team_points(
            experiments=args.experiments,
            expert_counts=args.team_experts,
            total_threads=args.team_total_threads,
            fixed_route=args.team_route,
            total_routes=args.team_total_routes,
            hidden=args.hidden,
            n_tile=n_tile,
            stage_mib=args.team_stage_mib,
        )
    )
    copies = args.warmup + args.runs
    rows: list[dict[str, object]] = []
    for index, point in enumerate(points, start=1):
        per_expert_bytes = stage_bytes(args.hidden, point.intermediate)
        total_bytes = point.experts * per_expert_bytes
        max_thread_bytes = (
            max_nsplit_thread_stage_bytes(
                args.hidden,
                point.intermediate,
                point.threads_per_expert,
                n_tile,
            )
            if point.experiment in {"nsplit", "team-fixed-route", "team-fixed-work"}
            else per_expert_bytes
        )
        total_routes = point.experts * point.routes
        total_flops = 6 * total_routes * args.hidden * point.intermediate
        command = benchmark_command(
            args.binary,
            point,
            hidden=args.hidden,
            copies=copies,
            cpu_start=args.cpu_start,
            warmup=args.warmup,
            runs=args.runs,
        )
        print(
            f"[{index:03d}/{len(points):03d}] {point.experiment:<12} routes={point.routes:<4} "
            f"shape={point.experts}x{point.threads_per_expert:<3} T={point.threads:<3} "
            f"F={point.intermediate:<5} per={per_expert_bytes / MIB:6.2f} MiB "
            f"total={total_bytes / MIB:7.2f} MiB",
            flush=True,
        )
        if args.dry_run:
            result: dict[str, float | str] = {}
            stages: dict[str, float] = {}
            allocated_gib = 0.0
        else:
            output = run_command(command)
            result, stages, allocated_gib = parse_benchmark_output(output)
            print(
                f"             wall={float(result['median_ms']):9.3f} ms  "
                f"throughput={float(result['tflops']):7.3f} TFLOP/s",
                flush=True,
            )
        rows.append(
            {
                **asdict(point),
                "n_tile": n_tile,
                "stage_bytes_per_expert": per_expert_bytes,
                "active_stage_bytes": total_bytes,
                "max_stage_bytes_per_thread": max_thread_bytes,
                "full_packed_weight_bytes_per_expert": 3 * per_expert_bytes,
                "w13_input_bytes_per_expert": 2 * point.routes * args.hidden,
                "w2_input_bytes_per_expert": 2 * point.routes * point.intermediate,
                "total_routes": total_routes,
                "total_flops": total_flops,
                "median_ms": result.get("median_ms"),
                "p99_ms": result.get("p99_ms"),
                "aggregate_tflops": result.get("tflops"),
                "stage_median_max_team_ms": stages,
                "allocated_gib": allocated_gib,
                "command": command,
            }
        )

    payload = {
        "schema_version": 1,
        "kind": "full_stage_thread_weight_working_set",
        "target": {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "os": platform.platform(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        },
        "kernel": {
            "entrypoint": "production_fused direct SVE M12 kernels",
            "stage_geometry": "full_n_team_stripes",
            "parallel_axis": "N",
            "n_tile": n_tile,
            "binary": str(args.binary.resolve()),
            "binary_sha256": sha256_file(args.binary),
        },
        "measurement": {
            "hidden": args.hidden,
            "routes": args.routes,
            "threads": args.threads,
            "experiments": args.experiments,
            "team_experts": args.team_experts,
            "team_total_threads": args.team_total_threads,
            "team_route": args.team_route,
            "team_total_routes": args.team_total_routes,
            "team_stage_mib": args.team_stage_mib,
            "warmup": args.warmup,
            "runs": args.runs,
            "copies": copies,
            "weight_reuse": "one distinct packed-weight copy per warmup and timed invocation",
            "working_set_definition": "active experts * max(full W13 bytes, full W2 bytes)",
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
