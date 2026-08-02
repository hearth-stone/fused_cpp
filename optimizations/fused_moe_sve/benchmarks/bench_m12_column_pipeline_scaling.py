#!/usr/bin/env python3
"""Compare baseline and column-pipelined M12 L1-hot scaling."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpu_moe_schedule_optimization/cost_model"))

from analytic_probe_geometry import m12_gemm_geometry, read_cache_info  # noqa: E402
from fused_cpp import _moe_C  # noqa: E402
from profile_analytic_services import (  # noqa: E402
    M12_ROWS,
    PROBE_FULL_NO_STORE,
    PROBE_FULL_NO_STORE_COLUMN_PIPELINE,
    bind_current_thread,
    packed_weights,
    parse_cpu_ids,
    parse_int_list,
)


VARIANTS = {
    "baseline": PROBE_FULL_NO_STORE,
    "column_pipeline": PROBE_FULL_NO_STORE_COLUMN_PIPELINE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=parse_int_list, default=parse_int_list("1,48,64,80,96"))
    parser.add_argument("--cpu-ids", type=parse_cpu_ids, default=parse_cpu_ids("0-95"))
    parser.add_argument("--batch-runs", type=int, default=262_144)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_runs <= 0 or args.repeats <= 0:
        parser.error("batch-runs and repeats must be positive")
    if max(args.widths) > len(args.cpu_ids):
        parser.error("cpu-ids must cover the largest width")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    cache_info = read_cache_info(args.cpu_ids[0])
    geometry = m12_gemm_geometry(cache_info["l1d_bytes_per_core"], 16, cache_fraction=0.625)
    packed = packed_weights(experts=1, k=geometry.k, n=geometry.n, seed=args.seed)
    packed_b = packed.w13[0][:1]
    n_tile = int(packed.backend_n_tile)
    a = torch.zeros((M12_ROWS, geometry.k), dtype=torch.bfloat16)
    flops_per_kernel = 2 * M12_ROWS * geometry.k * geometry.n

    samples: dict[str, dict[int, list[float]]] = {
        variant: {width: [] for width in args.widths} for variant in VARIANTS
    }
    cases = [(variant, width) for width in args.widths for variant in VARIANTS]
    for repeat in range(args.repeats):
        random.Random(args.seed + repeat).shuffle(cases)
        for variant, width in cases:
            probe_mode = VARIANTS[variant]
            ready = threading.Barrier(width)

            def worker(worker_id: int) -> float:
                bind_current_thread(args.cpu_ids[worker_id])
                _moe_C.fused_moe_bench_sve_jit_w13_gemm(
                    a, packed_b, geometry.k, geometry.n, n_tile, 1, 256, 1, probe_mode
                )
                ready.wait()
                begin = time.perf_counter_ns()
                _moe_C.fused_moe_bench_sve_jit_w13_gemm(
                    a,
                    packed_b,
                    geometry.k,
                    geometry.n,
                    n_tile,
                    1,
                    args.batch_runs,
                    1,
                    probe_mode,
                )
                return (time.perf_counter_ns() - begin) * 1e-9

            with ThreadPoolExecutor(max_workers=width) as executor:
                worker_seconds = list(executor.map(worker, range(width)))
            elapsed = max(worker_seconds)
            total_flops = width * (args.batch_runs + 1) * flops_per_kernel
            rate = total_flops / elapsed
            samples[variant][width].append(rate)
            print(
                f"repeat={repeat} variant={variant:<15} width={width:>2} "
                f"rate={rate / 1e12:>9.4f} TFLOP/s "
                f"worker_s[min/max]={min(worker_seconds):.6f}/{elapsed:.6f}",
                flush=True,
            )

    rows = []
    baseline_rates: dict[int, float] = {}
    single_core_rates = {
        variant: statistics.median(by_width[1]) for variant, by_width in samples.items()
    }
    for variant, by_width in samples.items():
        for width in sorted(by_width):
            rate = statistics.median(by_width[width])
            if variant == "baseline":
                baseline_rates[width] = rate
            rows.append(
                {
                    "variant": variant,
                    "threads": width,
                    "median_tflops": rate / 1e12,
                    "min_tflops": min(by_width[width]) / 1e12,
                    "max_tflops": max(by_width[width]) / 1e12,
                    "linear_efficiency": rate / (width * single_core_rates[variant]),
                    "samples_tflops": [sample / 1e12 for sample in by_width[width]],
                }
            )

    for row in rows:
        baseline = baseline_rates[row["threads"]]
        row["speedup_over_baseline"] = row["median_tflops"] * 1e12 / baseline - 1.0

    print("\nvariant\tthreads\tmedian_TFLOP/s\tlinear_efficiency\tvs_baseline")
    for row in rows:
        print(
            f"{row['variant']}\t{row['threads']}\t{row['median_tflops']:.6f}\t"
            f"{100.0 * row['linear_efficiency']:.3f}%\t"
            f"{100.0 * row['speedup_over_baseline']:+.3f}%"
        )

    payload = {
        "host": os.uname().nodename,
        "cpu_ids": args.cpu_ids,
        "widths": args.widths,
        "batch_runs": args.batch_runs,
        "repeats": args.repeats,
        "geometry": {
            "m": M12_ROWS,
            "k": geometry.k,
            "n": geometry.n,
            "working_set_bytes": geometry.working_set_bytes,
            "l1d_bytes_per_core": cache_info["l1d_bytes_per_core"],
        },
        "hugetlbfs_path": os.environ.get("FUSED_CPP_MOE_HUGETLBFS_PATH", ""),
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
