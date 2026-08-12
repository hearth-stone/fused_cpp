#!/usr/bin/env python3
"""Profile the generated GEMM's stage-window service independently of MoE E2E.

Every CPU executes one W13 owner stripe. Total M/K/N work per CPU stays fixed;
only the number of native N ranges changes. Packed B rotates through disjoint
weights whose aggregate reuse distance exceeds LLC, while the native benchmark
stops each process after packing/allocation so the parent can release the whole
CPU set together. This isolates the range/cache tradeoff without fitting routed
expert latencies.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpu_moe_schedule_optimization" / "cost_model"))

from profile_analytic_services import packed_weights, parse_cpu_ids, parse_int_list  # noqa: E402
from profile_gemm_memory_services import (  # noqa: E402
    MemoryState,
    PROBE_FULL_NO_STORE,
    concurrent_probe,
)
from profile_analytic_services import PROBE_FUSED_W13  # noqa: E402


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-ids", type=parse_cpu_ids, default=parse_cpu_ids("0-95"))
    parser.add_argument("--routes", type=parse_int_list, default=parse_int_list("72,216,384"))
    parser.add_argument("--team-widths", type=parse_int_list, default=parse_int_list("4,16"))
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--total-n", type=int, default=1024)
    parser.add_argument("--n-tile", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--operation",
        choices=("pure-gemm", "fused-w13"),
        default="pure-gemm",
        help="kernel body measured inside each N range",
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if min(args.k, args.total_n, args.n_tile, args.runs, args.repeats) <= 0 or args.warmup < 0:
        raise ValueError("dimensions/runs/repeats must be positive and warmup non-negative")
    if args.k % 8 or args.total_n % args.n_tile:
        raise ValueError("K must be K8 aligned and total N must be N-tile aligned")
    if any(routes > 12 and routes % 12 for routes in args.routes):
        raise ValueError("full-no-store stage-window probes require M<=12 or a multiple of 12")
    if any(args.total_n % width for width in args.team_widths):
        raise ValueError("each team width must divide total N")

    torch.set_num_threads(1)
    os.environ["FUSED_CPP_MOE_SVE_IMPL"] = "jit"
    cpu_count = len(args.cpu_ids)
    stream_state = MemoryState("a_hot_b_stream", stream_a=False, stream_b=True)
    entries: list[dict] = []
    point_order: list[dict] = []
    randomizer = random.Random(args.seed)

    for routes in args.routes:
        for team_width in args.team_widths:
            a = torch.zeros((cpu_count // team_width, routes, args.k), dtype=torch.bfloat16)
            owner_n = args.total_n // team_width
            if owner_n % args.n_tile:
                raise ValueError("owner stripe must contain whole N tiles")
            owner_tiles = owner_n // args.n_tile
            range_counts = []
            ranges = 1
            while ranges <= owner_tiles:
                if owner_tiles % ranges == 0:
                    range_counts.append(ranges)
                ranges *= 2

            copies = args.warmup + args.runs
            packed = packed_weights(
                experts=cpu_count * copies,
                k=args.k,
                n=owner_n,
                seed=args.seed + routes * 131 + team_width,
            )
            runtime_tile = int(packed.backend_n_tile)
            if runtime_tile != args.n_tile:
                raise RuntimeError(f"runtime N tile {runtime_tile} != requested {args.n_tile}")

            async_samples_by_ranges = {value: [] for value in range_counts}
            synchronized_samples_by_ranges = {value: [] for value in range_counts}
            repeat_orders = []
            for _ in range(args.repeats):
                order = list(range_counts)
                randomizer.shuffle(order)
                repeat_orders.append(order)
                for n_ranges in order:
                    worker_samples = concurrent_probe(
                        a=a,
                        packed_b=packed.w13[0],
                        state=stream_state,
                        width=cpu_count,
                        copies_per_worker=copies,
                        cpu_ids=args.cpu_ids,
                        k=args.k,
                        n=owner_n,
                        n_tile=runtime_tile,
                        warmup=args.warmup,
                        runs=args.runs,
                        probe_mode=(
                            PROBE_FUSED_W13
                            if args.operation == "fused-w13"
                            else PROBE_FULL_NO_STORE
                        ),
                        n_ranges=n_ranges,
                        a_group_size=team_width,
                    )
                    async_per_scan_ms = max(sum(worker) for worker in worker_samples) / args.runs
                    team_makespans = []
                    for team_begin in range(0, cpu_count, team_width):
                        team_samples = worker_samples[team_begin : team_begin + team_width]
                        team_makespans.append(
                            sum(
                                max(worker[run] for worker in team_samples)
                                for run in range(args.runs)
                            )
                        )
                    synchronized_per_scan_ms = max(team_makespans) / args.runs
                    async_samples_by_ranges[n_ranges].append(async_per_scan_ms)
                    synchronized_samples_by_ranges[n_ranges].append(synchronized_per_scan_ms)

            physical_flops = 2 * routes * args.k * owner_n * cpu_count
            for n_ranges in range_counts:
                samples = synchronized_samples_by_ranges[n_ranges]
                async_samples = async_samples_by_ranges[n_ranges]
                median_ms = statistics.median(samples)
                entry = {
                    "routes": routes,
                    "team_width": team_width,
                    "active_cpus": cpu_count,
                    "owner_n": owner_n,
                    "owner_tiles": owner_tiles,
                    "window_tiles": owner_tiles // n_ranges,
                    "n_ranges": n_ranges,
                    "median_ms": median_ms,
                    "p10_ms": percentile(samples, 0.10),
                    "p90_ms": percentile(samples, 0.90),
                    "async_worker_critical_median_ms": statistics.median(async_samples),
                    "synchronization_tax": (
                        median_ms / statistics.median(async_samples) - 1.0
                    ),
                    "aggregate_tflops": physical_flops / median_ms / 1e9,
                    "samples_ms": samples,
                    "async_worker_critical_samples_ms": async_samples,
                }
                entries.append(entry)
                print(
                    f"M={routes:<4} T={team_width:<2} ownerN={owner_n:<3} "
                    f"R={n_ranges:<3} w={entry['window_tiles']:<3} "
                    f"time={median_ms:>8.4f} ms rate={entry['aggregate_tflops']:>7.3f} TF/s",
                    flush=True,
                )
            point_order.append(
                {
                    "routes": routes,
                    "team_width": team_width,
                    "repeat_orders": repeat_orders,
                }
            )
            del packed
            gc.collect()

    payload = {
        "schema_version": 1,
        "kind": "sve_stage_window_service_probe",
        "config": {
            "cpu_ids": args.cpu_ids,
            "routes": args.routes,
            "team_widths": args.team_widths,
            "k": args.k,
            "total_n": args.total_n,
            "n_tile": args.n_tile,
            "warmup": args.warmup,
            "runs": args.runs,
            "repeats": args.repeats,
            "sampling": "shuffled_ranges_synchronized_process_wave",
            "operation": args.operation,
            "b_state": "disjoint_streaming_weights",
            "a_state": "hot_within_team_disjoint_between_teams",
        },
        "entries": entries,
        "point_order": point_order,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
