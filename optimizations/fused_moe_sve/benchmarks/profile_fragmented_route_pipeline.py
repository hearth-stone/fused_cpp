#!/usr/bin/env python3
"""Profile fixed-active-B route fragmentation on pinned worker threads."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import signal
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from run_fragmented_route_pipeline import (
    Point,
    benchmark_command,
    build_points,
    parse_output,
    parse_positive_ints,
    sha256_file,
)


DEFAULT_EVENTS = (
    "cycles",
    "instructions",
    "l2d_cache_refill",
    "ll_cache_rd",
    "ll_cache_miss_rd",
    "stall_backend_mem",
)
REQUIRED_EVENTS = (
    "cycles",
    "instructions",
    "l2d_cache_refill",
    "stall_backend_mem",
)
LINE_BYTES = 64


def parse_event_list(text: str) -> tuple[str, ...]:
    events = tuple(item.strip() for item in text.split(",") if item.strip())
    if not events or len(events) != len(set(events)):
        raise argparse.ArgumentTypeError("perf events must be a non-empty list without duplicates")
    return events


def optional_gib_per_invocation(per_invocation: dict[str, float], event: str) -> float | None:
    count = per_invocation.get(event)
    return None if count is None else count * LINE_BYTES / 2**30


def median_optional(rows: list[dict[str, object]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.median(values) if values else None


def format_optional(value: object, width: int, precision: int = 3) -> str:
    if value is None:
        return f"{'-':>{width}}"
    return f"{float(value):{width}.{precision}f}"


def parse_perf_csv(text: str, expected_events: tuple[str, ...]) -> tuple[dict[str, int], dict[str, float]]:
    counters: dict[str, int] = {}
    running_percent: dict[str, float] = {}
    for fields in csv.reader(text.splitlines()):
        if len(fields) < 3 or not fields[0] or fields[0].startswith("#"):
            continue
        raw_count = fields[0].replace(" ", "")
        event = fields[2].removesuffix(":u")
        if raw_count.startswith("<"):
            raise RuntimeError(f"perf did not count {event}: {raw_count}")
        try:
            count = round(float(raw_count))
        except ValueError:
            continue
        counters[event] = counters.get(event, 0) + count
        if len(fields) >= 5 and fields[4]:
            try:
                percent = float(fields[4].rstrip("%"))
            except ValueError:
                pass
            else:
                running_percent[event] = min(percent, running_percent.get(event, 100.0))

    missing = set(expected_events) - counters.keys()
    if missing:
        raise RuntimeError(f"missing perf counters: {', '.join(sorted(missing))}; output={text!r}")
    return counters, running_percent


def wait_until_stopped(process: subprocess.Popen[str]) -> None:
    _, status = os.waitpid(process.pid, os.WUNTRACED)
    if not os.WIFSTOPPED(status):
        raise RuntimeError(f"benchmark exited before profiler attach: status={status}")


def worker_tids(pid: int, expected_workers: int) -> list[int]:
    task_dir = Path(f"/proc/{pid}/task")
    tids = sorted(int(entry.name) for entry in task_dir.iterdir() if entry.name.isdigit() and int(entry.name) != pid)
    if len(tids) != expected_workers:
        raise RuntimeError(f"expected {expected_workers} worker TIDs for pid={pid}, found {len(tids)}")
    return tids


def profiled_command(
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
    numa_node: int,
) -> list[str]:
    workers = teams * threads_per_team
    benchmark = benchmark_command(
        binary,
        point,
        teams=teams,
        base_routes=base_routes,
        hidden=hidden,
        intermediate=intermediate,
        threads_per_team=threads_per_team,
        schedule=schedule,
        copies=copies,
        cpu_start=cpu_start,
        warmup=warmup,
        runs=runs,
        stop_before_run=True,
    )
    return [
        "numactl",
        f"--cpunodebind={numa_node}",
        f"--membind={numa_node}",
        "taskset",
        "-c",
        f"{cpu_start}-{cpu_start + workers - 1}",
        *benchmark,
    ]


def run_point(
    args: argparse.Namespace,
    point: Point,
    repeat: int,
) -> dict[str, object]:
    invocations = args.warmup + args.runs
    command = profiled_command(
        args.binary,
        point,
        teams=args.teams,
        base_routes=args.base_routes,
        hidden=args.hidden,
        intermediate=args.intermediate,
        threads_per_team=args.threads_per_team,
        schedule=args.schedule,
        copies=invocations,
        cpu_start=args.cpu_start,
        warmup=args.warmup,
        runs=args.runs,
        numa_node=args.numa_node,
    )
    environment = os.environ.copy()
    environment.update(
        {
            "LC_ALL": "C",
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
    )
    perf: subprocess.Popen[str] | None = None
    try:
        wait_until_stopped(process)
        tids = worker_tids(process.pid, args.teams * args.threads_per_team)
        perf_command = [
            "perf",
            "stat",
            "-x,",
            "--no-big-num",
            "-e",
            ",".join(f"{event}:u" for event in args.events),
            "-t",
            ",".join(str(tid) for tid in tids),
        ]
        perf = subprocess.Popen(
            perf_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        time.sleep(args.attach_delay_ms / 1000.0)
        if perf.poll() is not None:
            perf_output, _ = perf.communicate()
            raise subprocess.CalledProcessError(perf.returncode, perf_command, output=perf_output)
        os.kill(process.pid, signal.SIGCONT)
        benchmark_output, _ = process.communicate()
        perf_output, _ = perf.communicate(timeout=30)
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, output=benchmark_output)
        if perf.returncode != 0:
            raise subprocess.CalledProcessError(perf.returncode, perf_command, output=perf_output)
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        if perf is not None and perf.poll() is None:
            perf.kill()
            perf.wait()
        raise

    result, stages, config = parse_output(benchmark_output)
    counters, running_percent = parse_perf_csv(perf_output, args.events)
    cycles = counters["cycles"]
    instructions = counters["instructions"]
    per_invocation = {event: count / invocations for event, count in counters.items()}
    return {
        **asdict(point),
        "repeat": repeat,
        "median_ms": result["median_ms"],
        "p99_ms": result["p99_ms"],
        "aggregate_tflops": result["tflops"],
        "stage_median_max_team_sum_ms": stages,
        "allocated_gib": config["allocated_gib"],
        "perf_invocations": invocations,
        "perf": counters,
        "perf_running_percent": running_percent,
        "perf_per_invocation": per_invocation,
        "ipc": instructions / max(1, cycles),
        "memory_stall_fraction": counters["stall_backend_mem"] / max(1, cycles),
        "l2_refill_gib_per_invocation": optional_gib_per_invocation(per_invocation, "l2d_cache_refill"),
        "ll_read_gib_per_invocation": optional_gib_per_invocation(per_invocation, "ll_cache_rd"),
        "ll_miss_gib_per_invocation": optional_gib_per_invocation(per_invocation, "ll_cache_miss_rd"),
        "bus_access_gib_per_invocation": optional_gib_per_invocation(per_invocation, "bus_access"),
        "mem_access_gib_per_invocation": optional_gib_per_invocation(per_invocation, "mem_access"),
        "command": command,
    }


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for split_factor in sorted({int(row["split_factor"]) for row in rows}):
        group = [row for row in rows if row["split_factor"] == split_factor]
        first = group[0]
        summaries.append(
            {
                "split_factor": split_factor,
                "fragment_routes": first["fragment_routes"],
                "task_count": first["task_count"],
                "active_stage_bytes": first["active_stage_bytes"],
                "unique_weight_bytes": first["unique_weight_bytes"],
                "samples": len(group),
                "median_ms": statistics.median(float(row["median_ms"]) for row in group),
                "aggregate_tflops": statistics.median(float(row["aggregate_tflops"]) for row in group),
                "ipc": statistics.median(float(row["ipc"]) for row in group),
                "memory_stall_fraction": statistics.median(float(row["memory_stall_fraction"]) for row in group),
                "l2_refill_gib_per_invocation": median_optional(group, "l2_refill_gib_per_invocation"),
                "ll_read_gib_per_invocation": median_optional(group, "ll_read_gib_per_invocation"),
                "ll_miss_gib_per_invocation": median_optional(group, "ll_miss_gib_per_invocation"),
                "bus_access_gib_per_invocation": median_optional(group, "bus_access_gib_per_invocation"),
                "mem_access_gib_per_invocation": median_optional(group, "mem_access_gib_per_invocation"),
            }
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    parser.add_argument("--cpu-start", type=int, default=0)
    parser.add_argument("--numa-node", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--attach-delay-ms", type=int, default=100)
    parser.add_argument("--events", type=parse_event_list, default=DEFAULT_EVENTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.binary.is_file():
        raise FileNotFoundError(f"benchmark binary does not exist: {args.binary}")
    if min(args.teams, args.base_routes, args.hidden, args.intermediate, args.threads_per_team) <= 0:
        raise ValueError("teams and dimensions must be positive")
    if min(args.runs, args.repeats, args.attach_delay_ms) <= 0 or args.warmup < 0:
        raise ValueError("runs/repeats/attach delay must be positive and warmup non-negative")
    if args.cpu_start < 0 or args.numa_node < 0:
        raise ValueError("CPU start and NUMA node must be non-negative")
    missing_required_events = set(REQUIRED_EVENTS) - set(args.events)
    if missing_required_events:
        raise ValueError(f"missing required perf events: {', '.join(sorted(missing_required_events))}")

    points = build_points(
        teams=args.teams,
        base_routes=args.base_routes,
        hidden=args.hidden,
        intermediate=args.intermediate,
        split_factors=args.split_factors,
        replaced_teams=[args.teams],
    )
    rows: list[dict[str, object]] = []
    print("Q  M     tasks unique_B  wall_ms TFLOP/s L2refill_GiB bus_GiB LLread_GiB LLmiss_GiB stall% IPC")
    for repeat in range(args.repeats):
        for point in points:
            row = run_point(args, point, repeat)
            rows.append(row)
            print(
                f"{point.split_factor:<2} {point.fragment_routes:<5} {point.task_count:<5} "
                f"{point.unique_weight_bytes / 2**20:7.0f} "
                f"{float(row['median_ms']):8.3f} {float(row['aggregate_tflops']):7.3f} "
                f"{format_optional(row['l2_refill_gib_per_invocation'], 12)} "
                f"{format_optional(row['bus_access_gib_per_invocation'], 7)} "
                f"{format_optional(row['ll_read_gib_per_invocation'], 10)} "
                f"{format_optional(row['ll_miss_gib_per_invocation'], 10)} "
                f"{float(row['memory_stall_fraction']) * 100:6.2f} {float(row['ipc']):5.3f}",
                flush=True,
            )

    payload = {
        "schema_version": 1,
        "kind": "fixed_active_b_route_fragmentation_perf",
        "target": {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "os": platform.platform(),
        },
        "kernel": {
            "entrypoint": "production fused SVE M12 W13 and W2 kernels",
            "w13_ranges": 2,
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
            "warmup": args.warmup,
            "runs": args.runs,
            "repeats": args.repeats,
            "events": list(args.events),
            "perf_scope": "all pre-created benchmark worker TIDs; initialization and main thread excluded",
            "weight_reuse": "one distinct task-weight copy per warmup and timed invocation",
        },
        "rows": rows,
        "summary": summarize_rows(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
