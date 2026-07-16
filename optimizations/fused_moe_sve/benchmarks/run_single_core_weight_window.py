#!/usr/bin/env python3
"""Sweep one-core SVE GEMM weight sizes with unique cold weights per call."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import time
from pathlib import Path


DEFAULT_N_VALUES = "64,128,192,256,384,512,1024,2048,4096,6144,8192,10240,12288,14336,16384,20480,24576"
DEFAULT_EVENTS = (
    "cycles",
    "instructions",
    "l2d_cache",
    "l2d_cache_refill",
    "ll_cache_rd",
    "ll_cache_miss_rd",
    "stall_backend_mem",
)


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"invalid positive integer list: {text!r}")
    return values


def parse_event_list(text: str) -> tuple[str, ...]:
    events = tuple(item.strip() for item in text.split(",") if item.strip())
    if len(events) != len(set(events)):
        raise argparse.ArgumentTypeError("perf event list contains duplicates")
    return events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the cache-resident weight window of the production M12 SVE BF16 GEMM. "
            "Every warmup and timed call uses a distinct packed-B allocation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(__file__).with_name("bench_single_core_weight_window"),
    )
    parser.add_argument("--n-values", default=DEFAULT_N_VALUES)
    parser.add_argument("--m", type=int, default=120)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--numa-node", type=int, default=0)
    parser.add_argument("--cold-tail-mib", type=int, default=192)
    parser.add_argument("--attach-delay-ms", type=int, default=100)
    parser.add_argument(
        "--extra-events",
        type=parse_event_list,
        default=(),
        help="comma-separated perf events appended to the default event set",
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def parse_perf_csv(text: str, expected_events: tuple[str, ...]) -> dict[str, int]:
    counters: dict[str, int] = {}
    for fields in csv.reader(text.splitlines()):
        if len(fields) < 3 or not fields[0] or fields[0].startswith("#"):
            continue
        raw_count = fields[0].replace(",", "")
        event = fields[2].removesuffix(":u")
        if raw_count.startswith("<"):
            raise RuntimeError(f"perf did not count {event}: {raw_count}")
        try:
            counters[event] = int(raw_count)
        except ValueError:
            continue
    missing = set(expected_events) - counters.keys()
    if missing:
        raise RuntimeError(f"missing perf counters: {', '.join(sorted(missing))}; output={text!r}")
    return counters


def parse_result(lines: list[str]) -> dict[str, int | float]:
    for line in lines:
        if line.startswith("RESULT_JSON "):
            return json.loads(line.removeprefix("RESULT_JSON "))
    raise RuntimeError(f"benchmark did not emit RESULT_JSON: {lines!r}")


def wait_until_stopped(process: subprocess.Popen[str]) -> None:
    _, status = os.waitpid(process.pid, os.WUNTRACED)
    if not os.WIFSTOPPED(status):
        raise RuntimeError(f"benchmark exited before profiler attach: status={status}")


def run_point(args: argparse.Namespace, n_value: int) -> dict[str, object]:
    events = DEFAULT_EVENTS + args.extra_events
    command = [
        "numactl",
        f"--cpunodebind={args.numa_node}",
        f"--membind={args.numa_node}",
        "taskset",
        "-c",
        str(args.cpu),
        str(args.binary),
        "--m",
        str(args.m),
        "--k",
        str(args.k),
        "--n",
        str(n_value),
        "--warmup",
        str(args.warmup),
        "--runs",
        str(args.runs),
        "--cpu",
        str(args.cpu),
        "--cold-tail-mib",
        str(args.cold_tail_mib),
        "--stop-before-run",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_until_stopped(process)
        perf_command = [
            "perf",
            "stat",
            "-x,",
            "-e",
            ",".join(f"{event}:u" for event in events),
            "-p",
            str(process.pid),
        ]
        perf = subprocess.Popen(
            perf_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            time.sleep(args.attach_delay_ms / 1000.0)
            os.kill(process.pid, signal.SIGCONT)
            benchmark_output, _ = process.communicate()
            perf_output, _ = perf.communicate(timeout=30)
        except BaseException:
            perf.kill()
            perf.wait()
            raise
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, output=benchmark_output)
        if perf.returncode != 0:
            raise subprocess.CalledProcessError(perf.returncode, perf_command, output=perf_output)
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        raise

    result = parse_result(benchmark_output.splitlines())
    counters = parse_perf_csv(perf_output, events)
    invocations = args.warmup + args.runs
    weight_lines = int(result["weight_bytes"]) / 64.0
    result.update(
        {
            "perf": counters,
            "perf_invocations": invocations,
            "l2_refills_per_weight_line": counters["l2d_cache_refill"] / invocations / weight_lines,
            "ll_reads_per_weight_line": counters["ll_cache_rd"] / invocations / weight_lines,
            "ll_misses_per_weight_line": counters["ll_cache_miss_rd"] / invocations / weight_lines,
            "ll_read_miss_ratio": counters["ll_cache_miss_rd"] / max(1, counters["ll_cache_rd"]),
            "backend_mem_stall_fraction": counters["stall_backend_mem"] / max(1, counters["cycles"]),
        }
    )
    return result


def main() -> int:
    args = parse_args()
    duplicate_events = set(DEFAULT_EVENTS).intersection(args.extra_events)
    if duplicate_events:
        raise ValueError(f"extra perf events duplicate defaults: {', '.join(sorted(duplicate_events))}")
    if not args.binary.is_file():
        raise FileNotFoundError(f"benchmark binary does not exist: {args.binary}")
    if args.m <= 0 or args.m % 12 != 0 or args.k <= 0 or args.k % 8 != 0:
        raise ValueError("M must be a positive multiple of 12 and K a positive multiple of 8")
    if min(args.runs, args.cpu + 1, args.numa_node + 1, args.cold_tail_mib + 1, args.attach_delay_ms) <= 0:
        raise ValueError("runs/attach delay must be positive and CPU/NUMA/cold tail non-negative")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")

    rows: list[dict[str, object]] = []
    print("N      weight  median_ms GFLOP/s L2fill/line LLrd/line LLmiss/line LLmiss% memstall%")
    for n_value in parse_int_list(args.n_values):
        row = run_point(args, n_value)
        rows.append(row)
        print(
            f"{n_value:<6} {int(row['weight_bytes']) / 2**20:6.1f} "
            f"{float(row['median_ms']):9.3f} {float(row['gflops']):7.1f} "
            f"{float(row['l2_refills_per_weight_line']):11.3f} "
            f"{float(row['ll_reads_per_weight_line']):9.4f} "
            f"{float(row['ll_misses_per_weight_line']):11.4f} "
            f"{float(row['ll_read_miss_ratio']) * 100:7.2f} "
            f"{float(row['backend_mem_stall_fraction']) * 100:9.3f}",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "kind": "single_core_unique_weight_window",
        "config": {
            "m": args.m,
            "k": args.k,
            "warmup": args.warmup,
            "runs": args.runs,
            "cpu": args.cpu,
            "numa_node": args.numa_node,
            "cold_tail_mib": args.cold_tail_mib,
            "events": list(DEFAULT_EVENTS + args.extra_events),
            "weight_selection": "one_distinct_cold_packed_b_per_invocation",
        },
        "rows": rows,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote_json={args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
