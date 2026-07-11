#!/usr/bin/env python3
"""Profile async MoE isolated cost and homogeneous contention derates.

Measurements use consecutive windows of distinct experts. This avoids the old
single-expert hot-cache profile and better matches MoE forwards where each
expert weight block is streamed when that expert is visited.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fused_cpp.moe import (  # noqa: E402
    fused_moe_bf16_tiled_async,
    prepare_fused_moe_bf16_tiled_weights,
)


DEFAULT_ISOLATED_ROUTES = "1,2,4,8,12,24,48,96,192,384,768,1536,2040"
DEFAULT_CONTENTION_ROUTES = "12,48,192,768,2040"
DEFAULT_THREADS = "1,2,4,8,16,32"
DEFAULT_NUM_PROFILE_EXPERTS = 64
DEFAULT_MEASUREMENT_EXPERTS = 0
DEFAULT_ISOLATED_MEASUREMENT_EXPERTS = 8
DEFAULT_SHAPES = (
    "32;"
    "16x2;"
    "16,8,8;"
    "16,8,4,4;"
    "16,4,4,4,4;"
    "8x4;"
    "8,8,8,4,4;"
    "8,8,4,4,4,4;"
    "8,4,4,4,4,4,4;"
    "4x8;"
    "2x16;"
    "1x32"
)


def parse_int_list(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values or any(v <= 0 for v in values):
        raise ValueError(f"invalid positive integer list: {text!r}")
    return values


def parse_cpu_ids(text: str) -> list[int]:
    values: list[int] = []
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            first_text, last_text = part.split("-", 1)
            first, last = int(first_text), int(last_text)
            if first < 0 or last < first:
                raise ValueError(f"invalid CPU range: {part!r}")
            values.extend(range(first, last + 1))
        else:
            value = int(part)
            if value < 0:
                raise ValueError(f"invalid CPU id: {value}")
            values.append(value)
    if not values or len(values) != len(set(values)):
        raise ValueError(f"CPU ids must be non-empty and unique: {text!r}")
    return values


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kernel_metadata() -> dict:
    source_paths = (
        ROOT / "csrc" / "fused_moe_bf16_tiled.cpp",
        ROOT / "csrc" / "moe_sve_fused_asm.S",
        ROOT / "src" / "fused_cpp" / "moe" / "bf16_tiled.py",
    )
    source_digest = hashlib.sha256()
    for path in source_paths:
        source_digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        source_digest.update(path.read_bytes())

    def git(args: list[str]) -> str | None:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    git_commit = git(["rev-parse", "HEAD"])
    git_status = git(["status", "--short"])
    extension = sys.modules.get("fused_cpp._C")
    extension_path = Path(extension.__file__) if extension is not None else None
    return {
        "git_available": git_commit is not None,
        "git_commit": git_commit,
        "git_worktree_dirty": None if git_status is None else bool(git_status),
        "source_sha256": source_digest.hexdigest(),
        "extension_path": str(extension_path) if extension_path else None,
        "extension_sha256": sha256_file(extension_path) if extension_path else None,
    }


def detect_llc_bytes(cpu_id: int) -> int | None:
    cache_root = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/cache")
    for index in sorted(cache_root.glob("index*")):
        try:
            if (index / "level").read_text().strip() != "3":
                continue
            size = (index / "size").read_text().strip().upper()
        except OSError:
            continue
        scales = {"K": 1024, "M": 1024**2, "G": 1024**3}
        suffix = size[-1]
        if suffix in scales:
            return int(size[:-1]) * scales[suffix]
        return int(size)
    return None


class SyncClient:
    def __init__(self, port: int, rank: int):
        self._socket: socket.socket | None = None
        if port > 0:
            sock = socket.create_connection(("127.0.0.1", port), timeout=600.0)
            sock.sendall(bytes([rank]))
            self._socket = sock

    def wait(self) -> None:
        if self._socket is None:
            return
        self._socket.sendall(b"R")
        if self._socket.recv(1) != b"G":
            raise RuntimeError("invalid profile synchronization response")

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None


def parse_shapes(text: str) -> list[list[int]]:
    shapes: list[list[int]] = []
    for raw_shape in text.split(";"):
        raw_shape = raw_shape.strip()
        if not raw_shape:
            continue
        shape: list[int] = []
        for raw_item in raw_shape.split(","):
            item = raw_item.strip()
            if not item:
                continue
            if "x" in item:
                left, right = item.split("x", 1)
                value = int(left)
                count = int(right)
                shape.extend([value] * count)
            else:
                shape.append(int(item))
        if not shape:
            raise ValueError(f"contention shape must not be empty: {raw_shape!r}")
        if any(v <= 0 for v in shape):
            raise ValueError(f"invalid contention shape: {raw_shape!r}")
        shapes.append(shape)
    if not shapes:
        raise ValueError("at least one contention shape is required")
    return shapes


def percentile_ns(values: list[int], p: float) -> int:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    w = rank - lo
    return int(round(ordered[lo] * (1.0 - w) + ordered[hi] * w))


def summarize_times(values: list[int]) -> dict:
    return {
        "median_ns": int(statistics.median(values)),
        "p10_ns": percentile_ns(values, 0.10),
        "p90_ns": percentile_ns(values, 0.90),
        "min_ns": min(values),
        "mean_ns": int(statistics.mean(values)),
        "num_iters": len(values),
    }


def clamp_measurement_experts(num_experts: int, requested: int) -> int:
    if num_experts < 2:
        raise ValueError(
            "--num-experts must be at least 2; single-expert hot-cache "
            "profiling is intentionally unsupported"
        )
    if requested == 0:
        return num_experts
    if requested < 0:
        raise ValueError("--measurement-experts must be non-negative")
    return min(num_experts, requested)


def bf16(shape: tuple[int, ...], generator: torch.Generator, std: float) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.bfloat16).normal_(0.0, std, generator=generator)


def measure(
    run,
    *,
    warmup: int,
    runs: int,
    sync_client: SyncClient,
) -> list[int]:
    for _ in range(warmup):
        sync_client.wait()
        out = run()
        _ = float(out.flatten()[0])
    gc.collect()

    times: list[int] = []
    for _ in range(runs):
        sync_client.wait()
        t0 = time.perf_counter_ns()
        out = run()
        _ = float(out.flatten()[0])
        times.append(time.perf_counter_ns() - t0)
    return times


def make_async_run(
    *,
    packed,
    hidden_size: int,
    routes: int,
    shape: list[int],
    measurement_experts: int,
    num_profile_experts: int,
    cpu_ids: list[int],
    w13_split: bool | None,
    generator: torch.Generator,
    std: float,
    lane_experts: list[list[int]] | None = None,
):
    num_lanes = len(shape)
    num_tasks = measurement_experts
    if lane_experts is None:
        lane_experts = [[] for _ in shape]
        for expert in range(num_tasks):
            lane_experts[expert % num_lanes].append(expert)
    flattened = [expert for lane in lane_experts for expert in lane]
    if len(lane_experts) != num_lanes or sorted(flattened) != list(range(num_tasks)):
        raise ValueError("lane_experts must assign each measured expert exactly once")
    num_groups = max(len(lane) for lane in lane_experts)
    total_tokens = num_tasks * routes
    x = bf16((total_tokens, hidden_size), generator, std)
    topk_ids = torch.empty((total_tokens, 1), dtype=torch.int32)
    for expert_id in range(num_tasks):
        begin = expert_id * routes
        topk_ids[begin : begin + routes, 0] = expert_id
    topk_weights = torch.ones((total_tokens, 1), dtype=torch.float32)

    begins: list[int] = []
    core = 0
    for threads in shape:
        begins.append(core)
        core += threads

    task_expert_values: list[int] = []
    task_core_values: list[int] = []
    task_thread_values: list[int] = []
    dep_offsets = [0]
    deps: list[int] = []
    for lane, experts in enumerate(lane_experts):
        previous: int | None = None
        for expert in experts:
            task_id = len(task_expert_values)
            task_expert_values.append(expert)
            task_core_values.append(begins[lane])
            task_thread_values.append(shape[lane])
            if previous is not None:
                deps.append(previous)
            dep_offsets.append(len(deps))
            previous = task_id
    task_expert_ids = torch.tensor(task_expert_values, dtype=torch.int32)
    task_core_begins = torch.tensor(task_core_values, dtype=torch.int32)
    task_threads = torch.tensor(task_thread_values, dtype=torch.int32)
    task_dep_offsets = torch.tensor(dep_offsets, dtype=torch.int32)
    task_deps = torch.tensor(deps, dtype=torch.int32)
    if core > len(cpu_ids):
        raise ValueError(
            f"shape {shape} needs {core} CPUs, but only {len(cpu_ids)} were provided"
        )
    thread_cpu_ids = torch.tensor(cpu_ids[:core], dtype=torch.int32)

    def run() -> torch.Tensor:
        return fused_moe_bf16_tiled_async(
            x,
            packed,
            topk_weights,
            topk_ids,
            task_expert_ids,
            task_core_begins,
            task_threads,
            task_dep_offsets,
            task_deps,
            thread_cpu_ids=thread_cpu_ids,
            num_threads=core,
            activation="silu",
            global_num_experts=num_profile_experts,
            skip_weighted=True,
            w13_split=w13_split,
        )

    return run, num_groups, num_tasks, [len(lane) for lane in lane_experts]


def assign_uniform_experts(
    measurement_experts: int,
    shape: list[int],
    routes: int,
    iso_lookup: dict[tuple[int, int], int],
) -> list[list[int]]:
    """Use the same earliest-finish LPT assignment as IntervalPlanner."""
    loads = [0] * len(shape)
    lanes: list[list[int]] = [[] for _ in shape]
    for expert in range(measurement_experts):
        lane = min(
            range(len(shape)),
            key=lambda index: loads[index] + iso_lookup[(routes, shape[index])],
        )
        lanes[lane].append(expert)
        loads[lane] += iso_lookup[(routes, shape[lane])]
    return lanes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--ffn-hidden-size", type=int, default=512)
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_PROFILE_EXPERTS)
    parser.add_argument(
        "--measurement-experts",
        type=int,
        default=DEFAULT_MEASUREMENT_EXPERTS,
        help=(
            "Number of consecutive distinct experts represented in each timed "
            "call; 0 (default) uses every local expert. The final shape group "
            "is left partial rather than repeating weights."
        ),
    )
    parser.add_argument(
        "--isolated-measurement-experts",
        type=int,
        default=DEFAULT_ISOLATED_MEASUREMENT_EXPERTS,
        help=(
            "Distinct experts used to establish streaming isolated cost; "
            "defaults to 8 to exceed LLC without making slow 1T points scale "
            "with the full local expert count."
        ),
    )
    parser.add_argument(
        "--route-buckets",
        default=None,
        help=(
            "Legacy alias: when set, uses the same route buckets for isolated "
            "and contention measurements."
        ),
    )
    parser.add_argument("--isolated-route-buckets", default=DEFAULT_ISOLATED_ROUTES)
    parser.add_argument("--contention-route-buckets", default=DEFAULT_CONTENTION_ROUTES)
    parser.add_argument("--thread-buckets", default=DEFAULT_THREADS)
    parser.add_argument("--contention-shapes", default=DEFAULT_SHAPES)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--std", type=float, default=0.01)
    parser.add_argument(
        "--cpu-ids",
        default=None,
        help="Physical CPU ids/ranges used by logical workers, e.g. 0-31.",
    )
    parser.add_argument("--numa-node", type=int, default=-1)
    parser.add_argument("--rank-id", type=int, default=0)
    parser.add_argument("--concurrent-ranks", type=int, default=1)
    parser.add_argument(
        "--parallel-mode",
        choices=("standalone", "tp", "ep"),
        default="standalone",
    )
    parser.add_argument("--parallel-degree", type=int, default=1)
    parser.add_argument("--global-experts", type=int, default=None)
    parser.add_argument("--w13-split", type=int, choices=(0, 1), default=0)
    parser.add_argument("--w13-split-chunks", type=int, default=2)
    parser.add_argument("--llc-bytes", type=int, default=None)
    parser.add_argument(
        "--sync-port",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--store-samples", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.route_buckets:
        isolated_routes = parse_int_list(args.route_buckets)
        contention_routes = list(isolated_routes)
    else:
        isolated_routes = parse_int_list(args.isolated_route_buckets)
        contention_routes = parse_int_list(args.contention_route_buckets)
    thread_buckets = parse_int_list(args.thread_buckets)
    shapes = parse_shapes(args.contention_shapes)
    max_shape_cores = max(sum(shape) for shape in shapes)
    max_threads = max(thread_buckets)
    required_cpus = max(max_shape_cores, max_threads)
    cpu_ids = (
        parse_cpu_ids(args.cpu_ids)
        if args.cpu_ids is not None
        else list(range(required_cpus))
    )
    if len(cpu_ids) < required_cpus:
        raise ValueError(
            f"profile requires {required_cpus} CPUs, got {len(cpu_ids)}: {cpu_ids}"
        )
    if args.concurrent_ranks <= 0:
        raise ValueError("--concurrent-ranks must be positive")
    if not 0 <= args.rank_id < args.concurrent_ranks:
        raise ValueError("--rank-id must be in [0, concurrent-ranks)")
    if args.parallel_degree <= 0:
        raise ValueError("--parallel-degree must be positive")
    if args.w13_split and args.w13_split_chunks <= 1:
        raise ValueError("split W13 requires at least two chunks")
    global_experts = args.global_experts or args.num_experts
    if global_experts < args.num_experts:
        raise ValueError("--global-experts cannot be smaller than --num-experts")
    measurement_experts = clamp_measurement_experts(
        args.num_experts, args.measurement_experts
    )
    isolated_measurement_experts = clamp_measurement_experts(
        args.num_experts, args.isolated_measurement_experts
    )
    missing_contention_routes = sorted(set(contention_routes) - set(isolated_routes))
    if missing_contention_routes:
        raise ValueError(
            "contention routes must also be present in isolated routes: "
            f"{missing_contention_routes}"
        )

    shape_threads = sorted({t for shape in shapes for t in shape})
    missing = [t for t in shape_threads if t not in thread_buckets]
    if missing:
        raise ValueError(f"contention shapes use thread counts missing from table: {missing}")

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
    sync_client = SyncClient(args.sync_port, args.rank_id)

    isolated: list[dict] = []
    iso_lookup: dict[tuple[int, int], int] = {}
    for routes in isolated_routes:
        for threads in thread_buckets:
            run, num_groups, num_tasks, lane_task_counts = make_async_run(
                packed=packed,
                hidden_size=args.hidden_size,
                routes=routes,
                shape=[threads],
                measurement_experts=isolated_measurement_experts,
                num_profile_experts=args.num_experts,
                cpu_ids=cpu_ids,
                w13_split=bool(args.w13_split),
                generator=generator,
                std=args.std,
            )
            full_times = measure(
                run,
                warmup=args.warmup,
                runs=args.runs,
                sync_client=sync_client,
            )
            times = [max(1, int(round(value / num_groups))) for value in full_times]
            summary = summarize_times(times)
            entry = {"routes": routes, "threads": threads, **summary}
            entry["measurement_experts"] = isolated_measurement_experts
            entry["measurement_tasks"] = num_tasks
            entry["measurement_complete_groups"] = num_tasks
            entry["measurement_partial_lanes"] = 0
            entry["lane_task_counts"] = lane_task_counts
            entry["full_call_median_ns"] = int(statistics.median(full_times))
            if args.store_samples:
                entry["samples_ns"] = times
                entry["full_call_samples_ns"] = full_times
            isolated.append(entry)
            iso_lookup[(routes, threads)] = int(summary["median_ns"])
            print(
                f"isolated routes={routes:<5} threads={threads:<3} "
                f"per_expert_median={summary['median_ns'] / 1e6:8.3f} ms "
                f"full_median={entry['full_call_median_ns'] / 1e6:8.3f} ms"
            )

    entries: list[dict] = []
    for shape in shapes:
        for routes in contention_routes:
            lane_experts = assign_uniform_experts(
                measurement_experts, shape, routes, iso_lookup
            )
            run, num_groups, num_tasks, lane_task_counts = make_async_run(
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
            full_times = measure(
                run,
                warmup=args.warmup,
                runs=args.runs,
                sync_client=sync_client,
            )
            group_times = [
                max(1, int(round(value / num_groups))) for value in full_times
            ]
            summary = summarize_times(group_times)
            iso_max = max(iso_lookup[(routes, threads)] for threads in shape)
            iso_baseline = max(
                count * iso_lookup[(routes, threads)]
                for count, threads in zip(lane_task_counts, shape)
            )
            full_call_median = int(statistics.median(full_times))
            derate = float(full_call_median) / float(iso_baseline)
            entry = {
                "shape": shape,
                "distinct_experts": len(shape),
                "routes": routes,
                "makespan_ns": int(summary["median_ns"]),
                "full_call_median_ns": full_call_median,
                "measurement_experts": measurement_experts,
                "measurement_groups": num_groups,
                "measurement_tasks": num_tasks,
                "measurement_complete_groups": num_tasks // len(shape),
                "measurement_partial_lanes": num_tasks % len(shape),
                "lane_task_counts": lane_task_counts,
                "iso_max_ns": int(iso_max),
                "iso_baseline_makespan_ns": int(iso_baseline),
                "derate": derate,
                "p10_ns": int(summary["p10_ns"]),
                "p90_ns": int(summary["p90_ns"]),
                "num_iters": int(summary["num_iters"]),
            }
            if args.store_samples:
                entry["samples_ns"] = group_times
                entry["full_call_samples_ns"] = full_times
            entries.append(entry)
            print(
                f"contention shape={shape} routes={routes:<5} "
                f"per_group_median={summary['median_ns'] / 1e6:8.3f} ms "
                f"full_median={entry['full_call_median_ns'] / 1e6:8.3f} ms "
                f"derate={derate:6.3f}"
            )

    sync_client.close()
    w13_packed_bytes = packed.w13[0].numel() * packed.w13[0].element_size()
    w2_packed_bytes = packed.w2[0].numel() * packed.w2[0].element_size()
    split_chunks = args.w13_split_chunks if args.w13_split else 1
    llc_bytes = args.llc_bytes or detect_llc_bytes(cpu_ids[0])
    payload = {
        "schema_version": 2,
        "kind": "contention_derate",
        "target": {
            "machine": platform.machine(),
            "cpu": platform.processor() or platform.machine(),
            "host_logical_cores": os.cpu_count(),
            "cores_per_rank": len(cpu_ids),
            "cpu_ids": cpu_ids,
            "numa_node": args.numa_node,
            "llc_bytes_per_rank": llc_bytes,
            "rank_id": args.rank_id,
            "concurrent_ranks": args.concurrent_ranks,
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
            "global_experts": global_experts,
            "local_experts": args.num_experts,
        },
        "expert_shape": {
            "dtype": "bf16",
            "hidden_size": args.hidden_size,
            "intermediate_size": args.ffn_hidden_size,
            "kernel": "fused_moe_bf16_tiled_async",
            "activation": "silu",
            "num_experts": args.num_experts,
            "global_experts": global_experts,
            "measurement_experts": measurement_experts,
            "isolated_measurement_experts": isolated_measurement_experts,
            "expert_selection": "consecutive_window",
            "weight_reuse": "streaming_distinct_experts",
            "top_k": 1,
            "skip_weighted": True,
        },
        "working_set": {
            "w13_dense_bytes_per_expert": (
                2 * args.ffn_hidden_size * args.hidden_size * 2
            ),
            "w2_dense_bytes_per_expert": (
                args.hidden_size * args.ffn_hidden_size * 2
            ),
            "w13_packed_bytes_per_expert": w13_packed_bytes // args.num_experts,
            "w2_packed_bytes_per_expert": w2_packed_bytes // args.num_experts,
            "w13_chunk_bytes_per_expert": (
                w13_packed_bytes // args.num_experts // split_chunks
            ),
            "max_weight_stage_bytes_per_expert": max(
                w13_packed_bytes // args.num_experts // split_chunks,
                w2_packed_bytes // args.num_experts,
            ),
        },
        "measurement": {
            "path": "fused_moe_bf16_tiled_async",
            "pinning": "interval (thread_cpu_ids, disjoint per task)",
            "rank_synchronization": "socket_barrier" if args.sync_port else "none",
            "omp_proc_bind": os.environ.get("OMP_PROC_BIND", ""),
            "runs": args.runs,
            "warmup": args.warmup,
            "assignment": "earliest_finish_lpt_using_streaming_T_iso",
            "derate": "full_call_median / LPT_isolated_baseline_makespan",
        },
        "routes": contention_routes,
        "isolated_routes": isolated_routes,
        "contention_routes": contention_routes,
        "thread_buckets": thread_buckets,
        "contention_shapes": shapes,
        "isolated": isolated,
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
