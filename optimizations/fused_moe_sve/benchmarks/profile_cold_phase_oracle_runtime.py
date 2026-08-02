#!/usr/bin/env python3
"""Attach perf to warmed native workers for one cold-phase runtime variant."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import time
from pathlib import Path


DEFAULT_EVENTS = (
    "cycles",
    "instructions",
    "l2d_cache_refill",
    "ll_cache_rd",
    "ll_cache_miss_rd",
    "stall_backend_mem",
)
SOFTWARE_EVENTS = {"context-switches", "cpu-migrations", "task-clock"}


def _parse_events(value: str) -> tuple[str, ...]:
    events = tuple(item.strip() for item in value.split(",") if item.strip())
    if not events or len(events) != len(set(events)):
        raise argparse.ArgumentTypeError("events must be a non-empty list without duplicates")
    return events


def _wait_until_stopped(process: subprocess.Popen[str]) -> None:
    _, status = os.waitpid(process.pid, os.WUNTRACED)
    if not os.WIFSTOPPED(status):
        raise RuntimeError(f"benchmark exited before perf attach: status={status}")


def _allowed_cpu(tid: int, pid: int) -> int | None:
    status = Path(f"/proc/{pid}/task/{tid}/status").read_text(encoding="utf-8")
    for line in status.splitlines():
        if not line.startswith("Cpus_allowed_list:"):
            continue
        value = line.split(":", maxsplit=1)[1].strip()
        return int(value) if value.isdigit() else None
    return None


def _worker_tids(pid: int, expected_workers: int) -> tuple[list[int], list[int | None]]:
    tids = sorted(
        int(entry.name)
        for entry in Path(f"/proc/{pid}/task").iterdir()
        if entry.name.isdigit()
    )
    workers = [pid, *(tid for tid in tids if tid != pid and _allowed_cpu(tid, pid) is not None)]
    if len(workers) != expected_workers:
        affinities = {tid: _allowed_cpu(tid, pid) for tid in tids}
        raise RuntimeError(
            f"expected {expected_workers} single-CPU worker TIDs, found {len(workers)}; "
            f"all child affinities={affinities}"
        )
    return workers, [_allowed_cpu(tid, pid) for tid in workers]


def _parse_perf_csv(text: str, expected_events: tuple[str, ...]) -> tuple[dict[str, int], dict[str, float]]:
    counters: dict[str, int] = {}
    running_percent: dict[str, float] = {}
    for fields in csv.reader(text.splitlines()):
        if len(fields) < 3 or not fields[0] or fields[0].startswith("#"):
            continue
        event = fields[2].removesuffix(":u")
        raw_count = fields[0].replace(" ", "")
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
        raise RuntimeError(f"missing perf events {sorted(missing)}; output={text!r}")
    return counters, running_percent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-report", type=Path, required=True)
    parser.add_argument("--placement-input", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--collapse-internal-waits", action="store_true")
    parser.add_argument("--cpu-ids", default="0-95")
    parser.add_argument("--numa-node", type=int, default=0)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--attach-delay-ms", type=int, default=100)
    parser.add_argument("--events", type=_parse_events, default=DEFAULT_EVENTS)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if min(args.workers, args.runs, args.attach_delay_ms) <= 0 or args.warmup < 0:
        raise ValueError("workers, runs, and attach delay must be positive; warmup must be non-negative")
    benchmark = Path(__file__).with_name("bench_cold_phase_oracle_runtime.py")
    benchmark_output = args.output.with_suffix(".benchmark.json")
    command = [
        "numactl",
        f"--cpunodebind={args.numa_node}",
        f"--membind={args.numa_node}",
        "taskset",
        "-c",
        args.cpu_ids,
        "env",
        "PYTHONPATH=src",
        ".venv/bin/python",
        str(benchmark),
        "--oracle-report",
        str(args.oracle_report),
        "--placement-input",
        str(args.placement_input),
        "--release-scales",
        "0",
        "--warmup",
        str(args.warmup),
        "--runs",
        str(args.runs),
        "--measure-variant",
        args.variant,
        "--stop-before-measure",
        "--output",
        str(benchmark_output),
    ]
    if args.collapse_internal_waits:
        command.append("--collapse-internal-waits")

    environment = os.environ.copy()
    environment.update(
        {
            "LC_ALL": "C",
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
        _wait_until_stopped(process)
        tids, cpus = _worker_tids(process.pid, args.workers)
        perf_command = [
            "perf",
            "stat",
            "-x,",
            "--no-big-num",
            "-e",
            ",".join(event if event in SOFTWARE_EVENTS else f"{event}:u" for event in args.events),
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
        stdout, _ = process.communicate()
        perf_output, _ = perf.communicate(timeout=30)
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, output=stdout)
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

    counters, running_percent = _parse_perf_csv(perf_output, args.events)
    benchmark_result = json.loads(benchmark_output.read_text(encoding="utf-8"))
    record = benchmark_result["records"][0]
    per_call = {event: count / args.runs for event, count in counters.items()}
    result = {
        "variant": args.variant,
        "oracle_report": str(args.oracle_report),
        "placement_input": str(args.placement_input),
        "collapse_internal_waits": bool(args.collapse_internal_waits),
        "workers": args.workers,
        "worker_cpus": cpus,
        "warmup": args.warmup,
        "runs": args.runs,
        "median_ms": record["median_ms"],
        "aggregate_tflops": record["aggregate_tflops"],
        "events": list(args.events),
        "perf": counters,
        "perf_per_call": per_call,
        "perf_running_percent": running_percent,
        "ipc": counters["instructions"] / max(counters["cycles"], 1),
        "memory_stall_fraction": counters["stall_backend_mem"] / max(counters["cycles"], 1),
        "command": command,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.variant}: {record['median_ms']:.3f} ms, "
        f"IPC={result['ipc']:.3f}, memory_stall={100.0 * result['memory_stall_fraction']:.2f}%, "
        f"L2_refill/call={per_call['l2d_cache_refill']:.0f}, "
        f"LL_read/call={per_call['ll_cache_rd']:.0f}, "
        f"LL_miss/call={per_call['ll_cache_miss_rd']:.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
