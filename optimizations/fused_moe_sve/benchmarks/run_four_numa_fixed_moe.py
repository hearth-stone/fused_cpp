#!/usr/bin/env python3
"""Run four synchronized fixed-plan MoE ranks before and after a NUMA swap."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


REPO_ROOT = Path(__file__).resolve().parents[3]
CAPTURE = REPO_ROOT / "optimizations/fused_moe_sve/benchmarks/capture_schedule_timeline.py"


def percentile(samples: list[float], fraction: float) -> float:
    values = sorted(samples)
    return values[min(len(values) - 1, int((len(values) - 1) * fraction))]


def summarize(samples: list[float]) -> dict[str, float]:
    mean = statistics.mean(samples)
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": percentile(samples, 0.10),
        "p90_ms": percentile(samples, 0.90),
        "mean_ms": mean,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "cv": statistics.pstdev(samples) / mean,
    }


def run_mapping(
    name: str,
    mapping: tuple[int, int, int, int],
    output_root: Path,
    warmup: int,
    runs: int,
) -> dict:
    phase_dir = output_root / name
    phase_dir.mkdir(parents=True, exist_ok=True)
    start_file = phase_dir / "start"
    start_file.unlink(missing_ok=True)
    processes: list[tuple[int, int, subprocess.Popen, object]] = []
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "src:.",
            "OMP_NUM_THREADS": "80",
            "OMP_DYNAMIC": "FALSE",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    for rank, node in enumerate(mapping):
        ready_file = phase_dir / f"rank{rank}.ready"
        output_file = phase_dir / f"rank{rank}.json"
        trace_file = phase_dir / f"rank{rank}.trace"
        log_file = phase_dir / f"rank{rank}.log"
        for path in (ready_file, output_file, trace_file, log_file):
            path.unlink(missing_ok=True)
        cpu_begin = node * 80
        cpu_end = cpu_begin + 79
        command = [
            "numactl",
            f"--physcpubind={cpu_begin}-{cpu_end}",
            f"--membind={node}",
            str(REPO_ROOT / ".venv/bin/python"),
            str(CAPTURE),
            "--preset",
            "dsv4-real-2048-seq70",
            "--threads",
            "80",
            "--large-small-partition",
            "48:56:8:1",
            "--route-dtype",
            "bf16",
            "--warmup",
            str(warmup),
            "--runs",
            str(runs),
            "--ready-file",
            str(ready_file),
            "--start-file",
            str(start_file),
            "--trace-file",
            str(trace_file),
            "--output",
            str(output_file),
        ]
        log = log_file.open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append((rank, node, process, log))

    deadline = time.monotonic() + 900.0
    ready_files = [phase_dir / f"rank{rank}.ready" for rank in range(4)]
    while not all(path.is_file() for path in ready_files):
        failed = [(rank, process.returncode) for rank, _, process, _ in processes if process.poll() is not None]
        if failed:
            raise RuntimeError(f"rank failed before synchronized start: {failed}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for ranks in {name}")
        time.sleep(0.05)

    begin = time.perf_counter()
    start_file.write_text("start\n", encoding="utf-8")
    for rank, _, process, log in processes:
        returncode = process.wait()
        log.close()
        if returncode != 0:
            raise RuntimeError(f"rank {rank} failed with status {returncode}")
    elapsed_ms = (time.perf_counter() - begin) * 1e3

    rank_results = []
    for rank, node, _, _ in processes:
        payload = json.loads((phase_dir / f"rank{rank}.json").read_text(encoding="utf-8"))
        samples = [float(value) for value in payload["actual"]["untraced_samples_ms"]]
        rank_results.append({"rank": rank, "numa": node, **summarize(samples)})
    return {
        "name": name,
        "rank_to_numa": list(mapping),
        "concurrent_elapsed_ms": elapsed_ms,
        "concurrent_ms_per_iteration": elapsed_ms / runs,
        "ranks": rank_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        run_mapping("identity", (0, 1, 2, 3), args.output_dir, args.warmup, args.runs),
        run_mapping("swap_rank0_rank2", (2, 1, 0, 3), args.output_dir, args.warmup, args.runs),
    ]
    summary = {
        "machine": "Arm-codex-internal",
        "shape": "DSV4 TP4 H4096 F512 E256 T2048 TopK6",
        "fixed_plan": "routes>48: 56 cores / 8T; routes<=48: 24 cores / 1T",
        "warmup": args.warmup,
        "runs": args.runs,
        "results": results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
