#!/usr/bin/env python3
"""Measure whether compute-bound and memory-bound experts overlap for free.

The probe builds one heterogeneous workload -- a few long-route experts plus
many M12 experts -- and executes it several ways on the same physical cores:

``big_only``
    Only the long-route experts, on a contiguous core prefix.  One team per
    expert.

``small_only``
    Only the M12 experts, one thread per lane, on a contiguous core suffix.
    Sweeping the lane count gives the packed-B bandwidth saturation curve.

``segregated``
    All cores finish every long-route expert, then all cores drain the M12
    experts.  This is the no-co-scheduling baseline.

``mixed``
    Long-route experts occupy a core prefix while the M12 experts stream
    concurrently on the remaining cores.  Comparing this against
    ``max(big_only, small_only)`` measures how much of the co-scheduling
    overlap is actually free, and comparing it against ``segregated`` measures
    the schedule-level win.

Every timed task uses a distinct expert, so packed weights are streamed rather
than re-read from cache.  The core placement of ``big_only`` / ``small_only``
matches the corresponding ``mixed`` point exactly, so the three wall times are
directly comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import torch

from profile_contention_async import (
    SyncClient,
    bf16,
    detect_llc_bytes,
    kernel_metadata,
    measure,
    parse_cpu_ids,
    parse_int_list,
    summarize_times,
)

from fused_cpp.moe import (  # noqa: E402
    ASYNC_MOE_EXECUTION_STRICT,
    ASYNC_MOE_FULL_EXPERT_RANGE,
    ASYNC_MOE_PLACEMENT_FIXED,
    ASYNC_MOE_PLAN_VERSION,
    ASYNC_MOE_RESIZE_NONE,
    ASYNC_MOE_STAGE_EXPERT,
    AsyncMoEPlanV2,
    fused_moe_bf16_tiled_async,
    fused_moe_bf16_tiled_async_plan,
    prepare_fused_moe_bf16_tiled_weights,
)


DEFAULT_BIG_CORE_SPLITS = "24,36,48,60,72"
DEFAULT_SMALL_LANE_SWEEP = "12,24,36,48,60,72,96"


class Lane:
    """One contiguous logical-thread interval and its serial task chain."""

    def __init__(self, core_begin: int, threads: int, experts: list[int], first_deps: list[int] | None = None):
        self.core_begin = int(core_begin)
        self.threads = int(threads)
        self.experts = list(experts)
        self.first_deps = list(first_deps or ())


def expert_flops(routes: int, hidden_size: int, intermediate_size: int) -> int:
    return 6 * routes * hidden_size * intermediate_size


def chain_lengths(total: int, lanes: int) -> list[int]:
    """Spread ``total`` tasks over ``lanes`` chains as evenly as possible."""
    if lanes <= 0:
        raise ValueError("lanes must be positive")
    quotient, remainder = divmod(total, lanes)
    return [quotient + 1] * remainder + [quotient] * (lanes - remainder)


def build_run(
    *,
    packed,
    hidden_size: int,
    lanes: list[Lane],
    expert_routes: dict[int, int],
    cpu_ids: list[int],
    physical_cores: list[int],
    num_profile_experts: int,
    w13_split: bool,
    generator: torch.Generator,
    std: float,
    weight_window_bytes: int = 0,
    stage_window_bytes: tuple[int, int] | None = None,
):
    """Lower ``lanes`` to the async task DAG and return a callable plus metadata.

    ``weight_window_bytes`` is the operator-wide packed-B budget, which the v1
    async op applies to both stages. ``stage_window_bytes`` instead sets the W13
    and W2 budgets independently, which requires the Plan V2 path because
    per-task stage windows only exist there.
    """
    used = [expert for lane in lanes for expert in lane.experts]
    if len(set(used)) != len(used):
        raise ValueError("each timed task must use a distinct expert")
    missing = [expert for expert in used if expert not in expert_routes]
    if missing:
        raise ValueError(f"experts without a route count: {missing[:4]}")

    # The native bridge requires one task per routed expert, so only the experts
    # scheduled in this run receive rows.
    active_routes = {expert: expert_routes[expert] for expert in used}
    total_tokens = sum(active_routes.values())
    x = bf16((total_tokens, hidden_size), generator, std)
    topk_ids = torch.empty((total_tokens, 1), dtype=torch.int32)
    cursor = 0
    for expert in sorted(active_routes):
        routes = active_routes[expert]
        topk_ids[cursor : cursor + routes, 0] = expert
        cursor += routes
    topk_weights = torch.ones((total_tokens, 1), dtype=torch.float32)
    output = torch.empty_like(x)

    task_experts: list[int] = []
    task_cores: list[int] = []
    task_threads: list[int] = []
    dep_offsets = [0]
    deps: list[int] = []
    for lane in lanes:
        previous: int | None = None
        for expert in lane.experts:
            task_id = len(task_experts)
            task_experts.append(expert)
            task_cores.append(lane.core_begin)
            task_threads.append(lane.threads)
            if previous is None:
                for dependency in lane.first_deps:
                    if dependency >= task_id:
                        raise ValueError("dependencies must reference earlier tasks")
                    deps.append(dependency)
            else:
                deps.append(previous)
            dep_offsets.append(len(deps))
            previous = task_id

    num_threads = len(physical_cores)
    for lane in lanes:
        if lane.core_begin < 0 or lane.core_begin + lane.threads > num_threads:
            raise ValueError(
                f"lane [{lane.core_begin}, {lane.core_begin + lane.threads}) exceeds {num_threads} threads"
            )
    thread_cpu_ids = torch.tensor([cpu_ids[core] for core in physical_cores], dtype=torch.int32)

    tensors = {
        "task_expert_ids": torch.tensor(task_experts, dtype=torch.int32),
        "task_core_begins": torch.tensor(task_cores, dtype=torch.int32),
        "task_threads": torch.tensor(task_threads, dtype=torch.int32),
        "task_dep_offsets": torch.tensor(dep_offsets, dtype=torch.int32),
        "task_deps": torch.tensor(deps, dtype=torch.int32),
    }

    def run() -> torch.Tensor:
        return fused_moe_bf16_tiled_async(
            x,
            packed,
            topk_weights,
            topk_ids,
            tensors["task_expert_ids"],
            tensors["task_core_begins"],
            tensors["task_threads"],
            tensors["task_dep_offsets"],
            tensors["task_deps"],
            thread_cpu_ids=thread_cpu_ids,
            num_threads=num_threads,
            activation="silu",
            global_num_experts=num_profile_experts,
            skip_weighted=True,
            w13_split=w13_split,
            weight_window_bytes=weight_window_bytes,
            out=output,
        )

    if stage_window_bytes is not None:
        num_tasks = len(task_experts)
        plan = AsyncMoEPlanV2.from_dict(
            {
                "plan_version": ASYNC_MOE_PLAN_VERSION,
                "execution_mode": ASYNC_MOE_EXECUTION_STRICT,
                "num_threads": num_threads,
                "thread_cpu_ids": [int(value) for value in thread_cpu_ids.tolist()],
                "task_expert_ids": task_experts,
                "task_core_begins": task_cores,
                "task_threads": task_threads,
                "task_dep_offsets": dep_offsets,
                "task_deps": deps,
                "task_preferred_threads": list(task_threads),
                "task_min_threads": list(task_threads),
                "task_max_threads": list(task_threads),
                "task_allowed_thread_offsets": list(range(num_tasks + 1)),
                "task_allowed_threads": list(task_threads),
                "task_placement_modes": [ASYNC_MOE_PLACEMENT_FIXED] * num_tasks,
                "task_numa_nodes": [-1] * num_tasks,
                "task_stage_ids": [ASYNC_MOE_STAGE_EXPERT] * num_tasks,
                "task_resize_points": [ASYNC_MOE_RESIZE_NONE] * num_tasks,
                "task_range_granularities": [ASYNC_MOE_FULL_EXPERT_RANGE] * num_tasks,
                "task_w13_window_bytes": [stage_window_bytes[0]] * num_tasks,
                "task_w2_window_bytes": [stage_window_bytes[1]] * num_tasks,
            }
        )

        def run() -> torch.Tensor:  # noqa: F811
            return fused_moe_bf16_tiled_async_plan(
                x,
                packed,
                topk_weights,
                topk_ids,
                plan,
                activation="silu",
                global_num_experts=num_profile_experts,
                skip_weighted=True,
                w13_split=w13_split,
                weight_window_bytes=weight_window_bytes,
                out=output,
            )

    routes_by_task = [expert_routes[expert] for expert in task_experts]
    return run, {
        "tasks": len(task_experts),
        "threads": num_threads,
        "physical_cores": [cpu_ids[core] for core in physical_cores],
        "task_routes": routes_by_task,
        "total_routes": sum(routes_by_task),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe heterogeneous expert co-scheduling headroom.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--ffn-hidden-size", type=int, default=512)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--big-experts", type=int, default=4)
    parser.add_argument("--big-routes", type=int, default=2040)
    parser.add_argument("--small-experts", type=int, default=176)
    parser.add_argument("--small-routes", type=int, default=12)
    parser.add_argument(
        "--small-threads",
        type=int,
        default=1,
        help="threads per memory-bound lane; >1 shrinks the per-thread packed-B stage window",
    )
    parser.add_argument(
        "--big-core-splits",
        default=DEFAULT_BIG_CORE_SPLITS,
        help="cores handed to the long-route experts in mixed mode",
    )
    parser.add_argument(
        "--small-lane-sweep",
        default=DEFAULT_SMALL_LANE_SWEEP,
        help="one-thread lane counts for the small-only bandwidth curve",
    )
    parser.add_argument(
        "--mixed-small-lanes",
        default=None,
        help=(
            "optional lane counts to use in mixed mode instead of all remaining cores; "
            "the small workload is unchanged, so this isolates concurrency from total bytes"
        ),
    )
    parser.add_argument("--cpu-ids", default=None, help="physical CPUs, e.g. 0-95")
    parser.add_argument("--llc-bytes", type=int, default=None)
    parser.add_argument("--w13-split", type=int, choices=(0, 1), default=1)
    parser.add_argument("--w13-split-chunks", type=int, default=2)
    parser.add_argument(
        "--small-window-sweep",
        default=None,
        help=(
            "packed-B window targets in MiB for the memory-bound class (0 = legacy split-W13). "
            "When given, only the small-only matrix is measured."
        ),
    )
    parser.add_argument(
        "--small-w2-window-sweep",
        default=None,
        help=(
            "optional W2 window targets in MiB, crossed with --small-window-sweep (which then means W13). "
            "Requires the Plan V2 path because per-task stage windows only exist there. "
            "0 is not allowed here: legacy geometry cannot be requested per stage."
        ),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--std", type=float, default=0.01)
    parser.add_argument("--store-samples", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cpu_ids = parse_cpu_ids(args.cpu_ids) if args.cpu_ids is not None else sorted(os.sched_getaffinity(0))
    total_cores = len(cpu_ids)
    splits = parse_int_list(args.big_core_splits)
    lane_sweep = parse_int_list(args.small_lane_sweep)
    mixed_lanes = parse_int_list(args.mixed_small_lanes) if args.mixed_small_lanes else None
    if args.big_experts <= 0 or args.small_experts <= 0:
        raise ValueError("--big-experts and --small-experts must be positive")
    if args.big_experts + args.small_experts > args.num_experts:
        raise ValueError("--num-experts must cover every distinct timed expert")
    if any(not args.big_experts <= split < total_cores for split in splits):
        raise ValueError(f"--big-core-splits must be in [{args.big_experts}, {total_cores})")
    small_threads = int(args.small_threads)
    if small_threads <= 0 or total_cores % small_threads:
        raise ValueError(f"--small-threads must be a positive divisor of {total_cores}")
    window_sweep = None
    if args.small_window_sweep:
        window_sweep = [float(item) for item in args.small_window_sweep.split(",") if item.strip()]
        if not window_sweep or any(value < 0 for value in window_sweep):
            raise ValueError("--small-window-sweep must contain non-negative MiB values")
    w2_window_sweep = None
    if args.small_w2_window_sweep:
        if window_sweep is None:
            raise ValueError("--small-w2-window-sweep requires --small-window-sweep for the W13 axis")
        w2_window_sweep = [float(item) for item in args.small_w2_window_sweep.split(",") if item.strip()]
        if not w2_window_sweep or any(value <= 0 for value in w2_window_sweep):
            raise ValueError("--small-w2-window-sweep must contain positive MiB values")
        if any(value <= 0 for value in window_sweep):
            raise ValueError("the W13 axis must be positive when W2 is swept separately")
    if any(lanes * small_threads > total_cores for lanes in lane_sweep):
        raise ValueError(f"--small-lane-sweep values must not exceed {total_cores // small_threads}")
    if args.w13_split and args.w13_split_chunks != 2:
        raise ValueError("the current split-W13 kernel has exactly two chunks")

    big_experts = list(range(args.big_experts))
    small_experts = list(range(args.big_experts, args.big_experts + args.small_experts))
    expert_routes = {expert: args.big_routes for expert in big_experts}
    expert_routes.update({expert: args.small_routes for expert in small_experts})

    torch.set_num_threads(1)
    os.environ["FUSED_CPP_MOE_W13_SPLIT_N"] = "1" if args.w13_split else "0"
    generator = torch.Generator().manual_seed(args.seed)
    w13 = bf16((args.num_experts, 2 * args.ffn_hidden_size, args.hidden_size), generator, args.std)
    w2 = bf16((args.num_experts, args.hidden_size, args.ffn_hidden_size), generator, args.std)
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
    del w13, w2

    w13_bytes = packed.w13[0].numel() * packed.w13[0].element_size() // args.num_experts
    w2_bytes = packed.w2[0].numel() * packed.w2[0].element_size() // args.num_experts
    expert_weight_bytes = w13_bytes + w2_bytes
    split_chunks = args.w13_split_chunks if args.w13_split else 1
    stage_bytes = max(w13_bytes // split_chunks, w2_bytes)
    llc_bytes = args.llc_bytes or detect_llc_bytes(cpu_ids[0])
    sync_client = SyncClient(0, 0)

    common = {
        "packed": packed,
        "hidden_size": args.hidden_size,
        "expert_routes": expert_routes,
        "cpu_ids": cpu_ids,
        "num_profile_experts": args.num_experts,
        "w13_split": bool(args.w13_split),
        "generator": generator,
        "std": args.std,
    }

    def write_payload(entries: list[dict]) -> None:
        payload = {
            "schema_version": 1,
            "kind": "heterogeneous_expert_overlap",
            "target": {
                "machine": platform.machine(),
                "host_logical_cores": os.cpu_count(),
                "cpu_ids": cpu_ids,
                "llc_bytes": llc_bytes,
                "os": platform.platform(),
            },
            "kernel": {
                "name": "fused_moe_bf16_tiled_async",
                "backend": "sve" if int(packed.gemm_backend) == 1 else "neon",
                "gemm_backend": int(packed.gemm_backend),
                "backend_n_tile": int(packed.backend_n_tile),
                "w13_split": bool(args.w13_split),
                "w13_split_chunks": split_chunks,
                "hugetlbfs_path": os.environ.get("FUSED_CPP_MOE_HUGETLBFS_PATH", ""),
                **kernel_metadata(),
            },
            "expert_shape": {
                "dtype": "bf16",
                "hidden_size": args.hidden_size,
                "intermediate_size": args.ffn_hidden_size,
                "activation": "silu",
                "local_experts": args.num_experts,
                "weight_reuse": "streaming_distinct_experts",
                "w13_packed_bytes_per_expert": w13_bytes,
                "w2_packed_bytes_per_expert": w2_bytes,
                "weight_bytes_per_expert": expert_weight_bytes,
                "max_weight_stage_bytes_per_expert": stage_bytes,
            },
            "workload": {
                "big_experts": args.big_experts,
                "big_routes": args.big_routes,
                "small_experts": args.small_experts,
                "small_routes": args.small_routes,
                "small_threads": small_threads,
                "total_routes": args.big_experts * args.big_routes + args.small_experts * args.small_routes,
            },
            "measurement": {
                "big_core_splits": splits,
                "small_lane_sweep": lane_sweep,
                "mixed_small_lanes": mixed_lanes,
                "small_window_sweep_mib": window_sweep,
                "small_w2_window_sweep_mib": w2_window_sweep,
                "warmup": args.warmup,
                "runs": args.runs,
                "timer": "perf_counter_ns",
                "statistic": "median",
            },
            "entries": entries,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")

    def timed(
        label: str,
        lanes: list[Lane],
        physical_cores: list[int],
        extra: dict,
        weight_window_bytes: int = 0,
        stage_window_bytes: tuple[int, int] | None = None,
    ) -> dict:
        run, meta = build_run(
            lanes=lanes,
            physical_cores=physical_cores,
            weight_window_bytes=weight_window_bytes,
            stage_window_bytes=stage_window_bytes,
            **common,
        )
        samples = measure(run, warmup=args.warmup, runs=args.runs, sync_client=sync_client)
        timing = summarize_times(samples)
        median_ns = timing["median_ns"]
        flops = sum(expert_flops(routes, args.hidden_size, args.ffn_hidden_size) for routes in meta["task_routes"])
        entry = {
            "mode": label,
            **extra,
            **{key: value for key, value in meta.items() if key != "task_routes"},
            "weight_window_bytes": weight_window_bytes,
            "stage_window_bytes": list(stage_window_bytes) if stage_window_bytes else None,
            "median_ns": median_ns,
            "p10_ns": timing["p10_ns"],
            "p90_ns": timing["p90_ns"],
            "min_ns": timing["min_ns"],
            "mean_ns": timing["mean_ns"],
            "tflops": flops / median_ns / 1e3,
            "weight_bytes": meta["tasks"] * expert_weight_bytes,
            "weight_gbps": meta["tasks"] * expert_weight_bytes / median_ns,
        }
        if args.store_samples:
            entry["samples_ns"] = samples
        print(
            f"{label:<11} {extra.get('label', ''):<22} threads={meta['threads']:<3} "
            f"tasks={meta['tasks']:<4} wall={median_ns / 1e6:8.3f} ms "
            f"{entry['tflops']:7.3f} TFLOP/s {entry['weight_gbps']:7.1f} GB/s"
        )
        return entry

    def small_lanes(
        lane_count: int,
        experts: list[int],
        core_offset: int,
        first_deps=None,
        threads: int = 1,
    ) -> list[Lane]:
        counts = chain_lengths(len(experts), lane_count)
        lanes: list[Lane] = []
        cursor = 0
        for index, count in enumerate(counts):
            core = core_offset + index * threads
            deps = first_deps(core) if callable(first_deps) else None
            lanes.append(Lane(core, threads, experts[cursor : cursor + count], deps))
            cursor += count
        return lanes

    entries: list[dict] = []
    small_only: dict[int, dict] = {}

    def measure_small_only(
        lane_count: int,
        window_bytes: int = 0,
        w2_window_bytes: int | None = None,
    ) -> dict:
        """Small experts alone on the same core suffix that mixed mode uses."""
        key = (lane_count, window_bytes, w2_window_bytes)
        if key not in small_only:
            cores_used = lane_count * small_threads
            lanes = small_lanes(lane_count, small_experts, 0, threads=small_threads)
            cores = list(range(total_cores - cores_used, total_cores))
            per_thread = window_bytes / small_threads / 2**20 if window_bytes else None
            if w2_window_bytes is None:
                window_label = "legacy" if window_bytes == 0 else f"{window_bytes / 2**20:g}MiB"
                w2_per_thread = per_thread
            else:
                window_label = f"{window_bytes / 2**20:g}/{w2_window_bytes / 2**20:g}MiB"
                w2_per_thread = w2_window_bytes / small_threads / 2**20
            entry = timed(
                "small_only",
                lanes,
                cores,
                {
                    "label": f"{lane_count}x{small_threads}T w={window_label}",
                    "small_lanes": lane_count,
                    "small_threads": small_threads,
                    "big_cores": 0,
                    "window_per_thread_mib": per_thread,
                    "w2_window_per_thread_mib": w2_per_thread,
                },
                weight_window_bytes=0 if w2_window_bytes is not None else window_bytes,
                stage_window_bytes=None if w2_window_bytes is None else (window_bytes, w2_window_bytes),
            )
            small_only[key] = entry
            entries.append(entry)
        return small_only[key]

    print("--- small experts alone (packed-B bandwidth curve) ---")
    if w2_window_sweep is not None:
        assert window_sweep is not None
        for window_mib in window_sweep:
            for w2_mib in w2_window_sweep:
                for lane_count in lane_sweep:
                    measure_small_only(lane_count, int(window_mib * 2**20), int(w2_mib * 2**20))
        print("\n--- summary: useful packed-B bandwidth vs per-thread (W13, W2) window ---")
        best: dict[int, dict] = {}
        for entry in entries:
            lanes_used = entry["small_lanes"]
            if entry["weight_gbps"] > best.get(lanes_used, {"weight_gbps": 0.0})["weight_gbps"]:
                best[lanes_used] = entry
            print(
                f"{entry['label']:<30} "
                f"w13={entry['window_per_thread_mib']:>8.4g} w2={entry['w2_window_per_thread_mib']:>8.4g} MiB/thread  "
                f"wall={entry['median_ns'] / 1e6:8.3f} ms  "
                f"{entry['weight_gbps']:7.1f} GB/s  {entry['tflops']:7.3f} TFLOP/s"
            )
        print("\n--- best per-thread pair at each lane count ---")
        for lanes_used in sorted(best):
            entry = best[lanes_used]
            print(
                f"{lanes_used:>3}x{small_threads}T  "
                f"w13={entry['window_per_thread_mib']:g} w2={entry['w2_window_per_thread_mib']:g} MiB/thread  "
                f"{entry['weight_gbps']:7.1f} GB/s"
            )
        write_payload(entries)
        sync_client.close()
        return

    if window_sweep is not None:
        for window_mib in window_sweep:
            for lane_count in lane_sweep:
                measure_small_only(lane_count, int(window_mib * 2**20))
        print("\n--- summary: useful packed-B bandwidth vs per-thread window ---")
        for entry in entries:
            per_thread = entry["window_per_thread_mib"]
            print(
                f"{entry['label']:<26} per-thread={'legacy(4)' if per_thread is None else f'{per_thread:g}':>10} MiB  "
                f"wall={entry['median_ns'] / 1e6:8.3f} ms  "
                f"{entry['weight_gbps']:7.1f} GB/s  {entry['tflops']:7.3f} TFLOP/s"
            )
        write_payload(entries)
        sync_client.close()
        return 0
    for lane_count in lane_sweep:
        measure_small_only(lane_count)

    def big_lanes(cores: int) -> tuple[list[Lane], str]:
        """One team per long-route expert, splitting ``cores`` as evenly as possible."""
        widths = chain_lengths(cores, args.big_experts)
        lanes: list[Lane] = []
        core = 0
        for width, expert in zip(widths, big_experts, strict=True):
            lanes.append(Lane(core, width, [expert]))
            core += width
        unique = sorted(set(widths), reverse=True)
        label = "+".join(f"{widths.count(width)}x{width}T" for width in unique)
        return lanes, label

    print("--- big experts alone ---")
    big_only: dict[int, dict] = {}
    for split in splits:
        lanes, label = big_lanes(split)
        entry = timed(
            "big_only",
            lanes,
            list(range(split)),
            {"label": label, "big_cores": split, "small_lanes": 0},
        )
        big_only[split] = entry
        entries.append(entry)

    print("--- segregated: all cores on big experts, then all cores on small experts ---")
    wave1, wave1_label = big_lanes(total_cores)
    # A true global barrier: every small lane waits for every long-route expert,
    # so the two classes never run concurrently.
    barrier = list(range(len(wave1)))
    segregated_lanes = wave1 + small_lanes(
        total_cores // small_threads,
        small_experts,
        0,
        first_deps=lambda core: barrier,
        threads=small_threads,
    )
    segregated = timed(
        "segregated",
        segregated_lanes,
        list(range(total_cores)),
        {
            "label": f"{wave1_label} then {total_cores // small_threads}x{small_threads}T",
            "big_cores": total_cores,
            "small_lanes": total_cores // small_threads,
            "small_threads": small_threads,
        },
    )
    entries.append(segregated)

    print("--- mixed: big experts and small experts concurrent ---")
    for split in splits:
        candidates = [(total_cores - split) // small_threads] if mixed_lanes is None else mixed_lanes
        for lane_count in candidates:
            cores_used = lane_count * small_threads
            if lane_count <= 0 or split + cores_used > total_cores:
                continue
            offset = total_cores - cores_used
            lanes, label = big_lanes(split)
            lanes += small_lanes(lane_count, small_experts, offset, threads=small_threads)
            reference = measure_small_only(lane_count)
            entry = timed(
                "mixed",
                lanes,
                list(range(total_cores)),
                {
                    "label": f"{label} + {lane_count}x{small_threads}T",
                    "big_cores": split,
                    "small_lanes": lane_count,
                    "small_threads": small_threads,
                    "idle_cores": total_cores - split - cores_used,
                },
            )
            ideal_ns = max(big_only[split]["median_ns"], reference["median_ns"])
            serial_ns = big_only[split]["median_ns"] + reference["median_ns"]
            entry["ideal_overlap_ns"] = ideal_ns
            entry["serial_ns"] = serial_ns
            entry["overlap_efficiency"] = ideal_ns / entry["median_ns"]
            entry["realized_overlap_fraction"] = (
                (serial_ns - entry["median_ns"]) / (serial_ns - ideal_ns) if serial_ns > ideal_ns else None
            )
            entry["speedup_vs_segregated"] = segregated["median_ns"] / entry["median_ns"]
            entries.append(entry)

    print("\n--- summary ---")
    print(f"segregated baseline: {segregated['median_ns'] / 1e6:.3f} ms")
    for entry in entries:
        if entry["mode"] != "mixed":
            continue
        efficiency = entry["overlap_efficiency"]
        realized = entry["realized_overlap_fraction"]
        print(
            f"mixed {entry['label']:<22} wall={entry['median_ns'] / 1e6:8.3f} ms  "
            f"ideal={entry['ideal_overlap_ns'] / 1e6:8.3f} ms  "
            f"overlap_eff={efficiency:6.3f}  "
            f"realized={'n/a' if realized is None else f'{realized:6.3f}'}  "
            f"vs_segregated={entry['speedup_vs_segregated']:6.3f}x"
        )

    sync_client.close()
    write_payload(entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
