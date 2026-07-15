#!/usr/bin/env python3
"""Search the concurrent expert weight working set supported by a machine.

The timed calls rotate through distinct expert weights.  The number of active
lanes controls the instantaneous packed-weight working set, while the total
number of expert tasks remains fixed so throughput is comparable across points.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
from pathlib import Path

import torch

from profile_contention_async import (
    SyncClient,
    bf16,
    clamp_measurement_experts,
    detect_llc_bytes,
    kernel_metadata,
    make_async_run,
    measure,
    parse_cpu_ids,
    parse_int_list,
    summarize_times,
)
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights


DEFAULT_ROUTES = "12,48,192,768,2040"


def available_cpu_ids() -> list[int]:
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        return sorted(get_affinity(0))
    return list(range(os.cpu_count() or 1))


def balanced_thread_shape(total_cores: int, active_experts: int) -> list[int]:
    """Split all cores as evenly as possible across active expert lanes."""
    if total_cores <= 0 or not 0 < active_experts <= total_cores:
        raise ValueError("active experts must be in [1, total_cores]")
    quotient, remainder = divmod(total_cores, active_experts)
    return [quotient + 1] * remainder + [quotient] * (active_experts - remainder)


def default_active_experts(total_cores: int, num_experts: int) -> list[int]:
    """Use an exhaustive threshold search at normal per-rank expert counts."""
    upper = min(total_cores, num_experts)
    if upper <= 64:
        return list(range(1, upper + 1))
    values = set(range(1, 33))
    values.update(range(36, min(64, upper) + 1, 4))
    values.update(range(72, upper + 1, 8))
    values.add(upper)
    return sorted(value for value in values if value <= upper)


def parse_active_experts(
    text: str, total_cores: int, num_experts: int
) -> list[int]:
    if text.strip().lower() == "auto":
        return default_active_experts(total_cores, num_experts)
    values: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first, last = (int(value) for value in item.split("-", 1))
            if first <= 0 or last < first:
                raise ValueError(f"invalid active-expert range: {item!r}")
            values.extend(range(first, last + 1))
        else:
            values.append(int(item))
    values = sorted(set(values))
    upper = min(total_cores, num_experts)
    if not values or values[0] <= 0 or values[-1] > upper:
        raise ValueError(f"active experts must be in [1, {upper}]")
    return values


def assign_by_thread_capacity(
    measurement_experts: int, shape: list[int]
) -> list[list[int]]:
    """Balance equal-route tasks without requiring a pre-calibrated cost table."""
    lanes: list[list[int]] = [[] for _ in shape]
    for expert in range(measurement_experts):
        lane = min(
            range(len(shape)),
            key=lambda index: (
                (len(lanes[index]) + 1) / shape[index],
                len(lanes[index]),
                index,
            ),
        )
        lanes[lane].append(expert)
    return lanes


def expert_flops(routes: int, hidden_size: int, intermediate_size: int) -> int:
    return 6 * routes * hidden_size * intermediate_size


def summarize_search(
    entries: list[dict], throughput_tolerance: float
) -> list[dict]:
    summaries: list[dict] = []
    keys = sorted({(entry["allocation"], entry["routes"]) for entry in entries})
    for allocation, routes in keys:
        rows = [
            entry
            for entry in entries
            if entry["allocation"] == allocation and entry["routes"] == routes
        ]
        best = max(rows, key=lambda entry: entry["aggregate_tflops"])
        cutoff = best["aggregate_tflops"] * (1.0 - throughput_tolerance)
        near_peak = [entry for entry in rows if entry["aggregate_tflops"] >= cutoff]
        minimum = min(near_peak, key=lambda entry: entry["working_set_bytes"])
        maximum = max(near_peak, key=lambda entry: entry["working_set_bytes"])
        summaries.append(
            {
                "allocation": allocation,
                "routes": routes,
                "throughput_tolerance": throughput_tolerance,
                "best_active_experts": best["active_experts"],
                "best_working_set_bytes": best["working_set_bytes"],
                "best_aggregate_tflops": best["aggregate_tflops"],
                "near_peak_min_active_experts": minimum["active_experts"],
                "near_peak_min_working_set_bytes": minimum["working_set_bytes"],
                "near_peak_max_active_experts": maximum["active_experts"],
                "near_peak_max_working_set_bytes": maximum["working_set_bytes"],
            }
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search concurrent packed expert weight working-set limits."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--ffn-hidden-size", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--global-experts", type=int, default=64)
    parser.add_argument("--parallel-mode", choices=("standalone", "tp", "ep"), default="ep")
    parser.add_argument("--parallel-degree", type=int, default=2)
    parser.add_argument("--route-buckets", default=DEFAULT_ROUTES)
    parser.add_argument(
        "--allocation",
        choices=("one-thread", "uniform-cores", "both"),
        default="both",
        help="one thread per expert, all cores split across experts, or both",
    )
    parser.add_argument(
        "--active-experts",
        default="auto",
        help="auto, a comma list, or ranges such as 1-8,12,16,24,32",
    )
    parser.add_argument(
        "--measurement-experts",
        type=int,
        default=0,
        help="distinct expert tasks per timed call; 0 uses all local experts",
    )
    parser.add_argument("--cpu-ids", default=None, help="physical CPUs, e.g. 0-95")
    parser.add_argument("--numa-node", type=int, default=-1)
    parser.add_argument("--llc-bytes", type=int, default=None)
    parser.add_argument("--w13-split", type=int, choices=(0, 1), default=1)
    parser.add_argument("--w13-split-chunks", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--std", type=float, default=0.01)
    parser.add_argument("--throughput-tolerance", type=float, default=0.10)
    parser.add_argument("--store-samples", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cpu_ids = (
        parse_cpu_ids(args.cpu_ids)
        if args.cpu_ids is not None
        else available_cpu_ids()
    )
    total_cores = len(cpu_ids)
    active_values = parse_active_experts(
        args.active_experts, total_cores, args.num_experts
    )
    routes_values = parse_int_list(args.route_buckets)
    measurement_experts = clamp_measurement_experts(
        args.num_experts, args.measurement_experts
    )
    if active_values[-1] > measurement_experts:
        raise ValueError(
            "--measurement-experts must be at least the largest active-expert point"
        )
    if args.global_experts < args.num_experts:
        raise ValueError("--global-experts cannot be smaller than --num-experts")
    if not 0.0 <= args.throughput_tolerance < 1.0:
        raise ValueError("--throughput-tolerance must be in [0, 1)")
    if args.w13_split and args.w13_split_chunks != 2:
        raise ValueError("the current split-W13 kernel has exactly two chunks")

    allocations = (
        ["one-thread", "uniform-cores"]
        if args.allocation == "both"
        else [args.allocation]
    )
    torch.set_num_threads(1)
    os.environ["FUSED_CPP_MOE_W13_SPLIT_N"] = "1" if args.w13_split else "0"
    generator = torch.Generator().manual_seed(args.seed)
    w13 = bf16(
        (args.num_experts, 2 * args.ffn_hidden_size, args.hidden_size),
        generator,
        args.std,
    )
    w2 = bf16(
        (args.num_experts, args.hidden_size, args.ffn_hidden_size),
        generator,
        args.std,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
    del w13, w2
    w13_bytes = packed.w13[0].numel() * packed.w13[0].element_size()
    w2_bytes = packed.w2[0].numel() * packed.w2[0].element_size()
    split_chunks = args.w13_split_chunks if args.w13_split else 1
    w13_stage_bytes = w13_bytes // args.num_experts // split_chunks
    w2_stage_bytes = w2_bytes // args.num_experts
    stage_bytes = max(w13_stage_bytes, w2_stage_bytes)
    llc_bytes = args.llc_bytes or detect_llc_bytes(cpu_ids[0])
    sync_client = SyncClient(0, 0)

    entries: list[dict] = []
    for allocation in allocations:
        for active_experts in active_values:
            shape = (
                [1] * active_experts
                if allocation == "one-thread"
                else balanced_thread_shape(total_cores, active_experts)
            )
            lane_experts = assign_by_thread_capacity(measurement_experts, shape)
            for routes in routes_values:
                run, groups, tasks, lane_counts = make_async_run(
                    packed=packed,
                    hidden_size=args.hidden_size,
                    routes=routes,
                    shape=shape,
                    measurement_experts=measurement_experts,
                    num_profile_experts=args.num_experts,
                    cpu_ids=cpu_ids,
                    w13_split=bool(args.w13_split),
                    generator=generator,
                    std=args.std,
                    lane_experts=lane_experts,
                )
                samples = measure(
                    run, warmup=args.warmup, runs=args.runs, sync_client=sync_client
                )
                timing = summarize_times(samples)
                median_ns = timing["median_ns"]
                tflops = (
                    tasks
                    * expert_flops(routes, args.hidden_size, args.ffn_hidden_size)
                    / median_ns
                    / 1e3
                )
                entry = {
                    "allocation": allocation,
                    "active_experts": active_experts,
                    "shape": shape,
                    "total_threads": sum(shape),
                    "routes": routes,
                    "working_set_bytes": active_experts * stage_bytes,
                    "working_set_mib": active_experts * stage_bytes / 2**20,
                    "working_set_to_llc": (
                        active_experts * stage_bytes / llc_bytes
                        if llc_bytes
                        else None
                    ),
                    "full_call_median_ns": median_ns,
                    "full_call_p10_ns": timing["p10_ns"],
                    "full_call_p90_ns": timing["p90_ns"],
                    "measurement_experts": tasks,
                    "measurement_groups": groups,
                    "lane_task_counts": lane_counts,
                    "aggregate_tflops": tflops,
                }
                if args.store_samples:
                    entry["full_call_samples_ns"] = samples
                entries.append(entry)
                print(
                    f"{allocation:<13} routes={routes:<4} active={active_experts:<3} "
                    f"threads={sum(shape):<3} ws={entry['working_set_mib']:7.1f} MiB "
                    f"wall={median_ns / 1e6:9.3f} ms  {tflops:7.3f} TFLOP/s"
                )

    sync_client.close()
    summaries = summarize_search(entries, args.throughput_tolerance)
    print("\nNear-peak working-set ranges:")
    for row in summaries:
        print(
            f"{row['allocation']:<13} routes={row['routes']:<4} "
            f"best={row['best_working_set_bytes'] / 2**20:7.1f} MiB, "
            f"within {args.throughput_tolerance:.0%}: "
            f"{row['near_peak_min_working_set_bytes'] / 2**20:.1f}-"
            f"{row['near_peak_max_working_set_bytes'] / 2**20:.1f} MiB"
        )

    payload = {
        "schema_version": 1,
        "kind": "expert_working_set_search",
        "target": {
            "machine": platform.machine(),
            "cpu": platform.processor() or platform.machine(),
            "host_logical_cores": os.cpu_count(),
            "cpu_ids": cpu_ids,
            "numa_node": args.numa_node,
            "llc_bytes": llc_bytes,
            "os": platform.platform(),
        },
        "kernel": {
            "name": "fused_moe_bf16_tiled_async",
            "backend": "sve" if int(packed.gemm_backend) == 1 else "neon",
            "gemm_backend": int(packed.gemm_backend),
            "backend_n_tile": int(packed.backend_n_tile),
            "parallel_axis": "N",
            "w13_split": bool(args.w13_split),
            "w13_split_chunks": split_chunks,
            **kernel_metadata(),
        },
        "parallelism": {
            "mode": args.parallel_mode,
            "degree": args.parallel_degree,
            "global_experts": args.global_experts,
            "local_experts": args.num_experts,
        },
        "expert_shape": {
            "dtype": "bf16",
            "hidden_size": args.hidden_size,
            "intermediate_size": args.ffn_hidden_size,
            "activation": "silu",
            "measurement_experts": measurement_experts,
            "expert_selection": "consecutive_window",
            "weight_reuse": "streaming_distinct_experts",
        },
        "working_set": {
            "w13_packed_bytes_per_expert": w13_bytes // args.num_experts,
            "w13_stage_bytes_per_expert": w13_stage_bytes,
            "w2_packed_bytes_per_expert": w2_bytes // args.num_experts,
            "max_weight_stage_bytes_per_expert": stage_bytes,
            "definition": "active_experts * max(w13_chunk_bytes, w2_bytes)",
        },
        "measurement": {
            "allocations": allocations,
            "active_experts": active_values,
            "routes": routes_values,
            "runs": args.runs,
            "warmup": args.warmup,
            "throughput_tolerance": args.throughput_tolerance,
            "pinning": "interval (thread_cpu_ids, disjoint per task)",
            "comparison": "fixed distinct expert tasks per timed call",
        },
        "entries": entries,
        "summary": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
