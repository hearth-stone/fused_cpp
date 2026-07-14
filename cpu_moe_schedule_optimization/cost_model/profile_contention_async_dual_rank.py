#!/usr/bin/env python3
"""Profile two NUMA-local MoE ranks concurrently and merge their wall costs."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from iso_formula import fit_from_measurements
except ImportError:  # pragma: no cover - package-style import
    from .iso_formula import fit_from_measurements  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("profile_contention_async.py")
DEFAULT_ISOLATED_ROUTES = "1,2,4,8,12,24,48,96,192,384,768,1536,2040"
DEFAULT_CONTENTION_ROUTES = "1,2,4,8,12,24,48,192,768,2040"
DEFAULT_THREADS = "1,2,4,8,16,32"
DEFAULT_SHAPES = (
    "32;16x2;16,8,8;16,8,4,4;16,4,4,4,4;8x4;"
    "8,8,8,4,4;8,8,4,4,4,4;8,4,4,4,4,4,4;4x8;2x16;1x32"
)


def split_nonempty(text: str, separator: str = ",") -> list[str]:
    return [value.strip() for value in text.split(separator) if value.strip()]


def percentile_ns(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return int(round(ordered[lo] * (1.0 - weight) + ordered[hi] * weight))


def summarize(values: list[int]) -> dict[str, int]:
    return {
        "median_ns": int(statistics.median(values)),
        "p10_ns": percentile_ns(values, 0.10),
        "p90_ns": percentile_ns(values, 0.90),
        "min_ns": min(values),
        "mean_ns": int(statistics.mean(values)),
        "num_iters": len(values),
    }


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    while size:
        chunk = sock.recv(size)
        if not chunk:
            raise RuntimeError("rank profiler closed its synchronization socket")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def paired_samples(first: dict, second: dict, key: str) -> list[int]:
    left = [int(value) for value in first.pop(key)]
    right = [int(value) for value in second.pop(key)]
    if len(left) != len(right):
        raise ValueError(f"rank sample count mismatch for {key}")
    return [max(a, b) for a, b in zip(left, right)]


def merge_isolated(first: dict, second: dict) -> dict:
    if (first["routes"], first["threads"]) != (
        second["routes"],
        second["threads"],
    ):
        raise ValueError("isolated profile keys differ between ranks")
    rank_medians = [int(first["median_ns"]), int(second["median_ns"])]
    samples = paired_samples(first, second, "samples_ns")
    full_samples = paired_samples(first, second, "full_call_samples_ns")
    result = copy.deepcopy(first)
    result.update(summarize(samples))
    result["full_call_median_ns"] = int(statistics.median(full_samples))
    result["full_call_p10_ns"] = percentile_ns(full_samples, 0.10)
    result["full_call_p90_ns"] = percentile_ns(full_samples, 0.90)
    result["rank_median_ns"] = rank_medians
    result.pop("samples_ns", None)
    result.pop("full_call_samples_ns", None)
    return result


def merge_contention(first: dict, second: dict) -> dict:
    if (first["shape"], first["routes"]) != (
        second["shape"],
        second["routes"],
    ):
        raise ValueError("contention profile keys differ between ranks")
    rank_medians = [int(first["makespan_ns"]), int(second["makespan_ns"])]
    first.pop("samples_ns")
    second.pop("samples_ns")
    full_samples = paired_samples(first, second, "full_call_samples_ns")
    result = copy.deepcopy(first)
    result["rank_lane_task_counts"] = [
        first.get("lane_task_counts"),
        second.get("lane_task_counts"),
    ]
    result["full_call_median_ns"] = int(statistics.median(full_samples))
    result["full_call_p10_ns"] = percentile_ns(full_samples, 0.10)
    result["full_call_p90_ns"] = percentile_ns(full_samples, 0.90)
    result["rank_median_ns"] = rank_medians
    result.pop("full_call_samples_ns", None)
    result["_full_call_samples_ns"] = full_samples
    return result


def validate_profile(profile: dict) -> None:
    if profile.get("schema_version") != 2:
        raise ValueError("merged profile must use schema version 2")
    target = profile["target"]
    rank_count = int(target["concurrent_ranks"])
    if rank_count != len(target["cpu_ids_by_rank"]):
        raise ValueError("concurrent rank count does not match CPU sets")
    if rank_count != len(target["numa_nodes"]):
        raise ValueError("concurrent rank count does not match NUMA nodes")
    if any(
        len(cpu_ids) != int(target["cores_per_rank"])
        for cpu_ids in target["cpu_ids_by_rank"]
    ):
        raise ValueError("rank CPU set does not match cores_per_rank")
    kernel = profile["kernel"]
    if not kernel.get("source_sha256") or not kernel.get("extension_sha256"):
        raise ValueError("schema-v2 profiles require source and extension hashes")

    iso_lookup: dict[tuple[int, int], int] = {}
    for entry in profile["isolated"]:
        if not entry["p10_ns"] <= entry["median_ns"] <= entry["p90_ns"]:
            raise ValueError(f"invalid isolated percentiles: {entry}")
        iso_lookup[(int(entry["routes"]), int(entry["threads"]))] = int(
            entry["median_ns"]
        )
    for entry in profile["entries"]:
        if not entry["p10_ns"] <= entry["makespan_ns"] <= entry["p90_ns"]:
            raise ValueError(f"invalid contention percentiles: {entry}")
        route_count = int(entry["routes"])
        iso_max = max(
            iso_lookup[(route_count, int(threads))]
            for threads in entry["shape"]
        )
        if int(entry["iso_max_ns"]) != iso_max:
            raise ValueError(f"invalid iso_max_ns: {entry}")
        iso_baseline = max(
            int(count) * iso_lookup[(route_count, int(threads))]
            for count, threads in zip(
                entry["lane_task_counts"], entry["shape"]
            )
        )
        if int(entry["iso_baseline_makespan_ns"]) != iso_baseline:
            raise ValueError(f"invalid iso_baseline_makespan_ns: {entry}")
        expected_derate = float(entry["full_call_median_ns"]) / float(iso_baseline)
        if not math.isclose(float(entry["derate"]), expected_derate, rel_tol=1e-12):
            raise ValueError(f"invalid contention derate: {entry}")


def merge_profiles(paths: list[Path], output: Path) -> dict:
    profiles = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(profiles) != 2 or any(p.get("schema_version") != 2 for p in profiles):
        raise ValueError("dual-rank merge requires two schema-v2 profiles")
    first, second = profiles
    for key in (
        "kernel",
        "parallelism",
        "expert_shape",
        "working_set",
        "isolated_routes",
        "contention_routes",
        "thread_buckets",
        "contention_shapes",
    ):
        if first[key] != second[key]:
            raise ValueError(f"rank profiles differ in {key}")

    isolated = [
        merge_isolated(copy.deepcopy(a), copy.deepcopy(b))
        for a, b in zip(first["isolated"], second["isolated"])
    ]
    iso_lookup = {
        (int(entry["routes"]), int(entry["threads"])): int(entry["median_ns"])
        for entry in isolated
    }
    entries = [
        merge_contention(copy.deepcopy(a), copy.deepcopy(b))
        for a, b in zip(first["entries"], second["entries"])
    ]

    def merged_lane_counts(entry: dict) -> list[int]:
        shape = [int(value) for value in entry["shape"]]
        routes = int(entry["routes"])
        loads = [0] * len(shape)
        counts = [0] * len(shape)
        for _ in range(int(entry["measurement_tasks"])):
            lane = min(
                range(len(shape)),
                key=lambda index: loads[index]
                + iso_lookup[(routes, shape[index])],
            )
            counts[lane] += 1
            loads[lane] += iso_lookup[(routes, shape[lane])]
        return counts

    for entry in entries:
        route_count = int(entry["routes"])
        iso_max = max(
            iso_lookup[(route_count, int(threads))]
            for threads in entry["shape"]
        )
        entry["lane_task_counts"] = merged_lane_counts(entry)
        entry["measurement_groups"] = max(entry["lane_task_counts"])
        full_samples = entry.pop("_full_call_samples_ns")
        group_samples = [
            max(1, int(round(value / entry["measurement_groups"])))
            for value in full_samples
        ]
        group_summary = summarize(group_samples)
        entry["makespan_ns"] = group_summary["median_ns"]
        entry["p10_ns"] = group_summary["p10_ns"]
        entry["p90_ns"] = group_summary["p90_ns"]
        entry["min_ns"] = group_summary["min_ns"]
        entry["mean_ns"] = group_summary["mean_ns"]
        entry["num_iters"] = group_summary["num_iters"]
        iso_baseline = max(
            int(count) * iso_lookup[(route_count, int(threads))]
            for count, threads in zip(
                entry["lane_task_counts"], entry["shape"]
            )
        )
        entry["iso_max_ns"] = iso_max
        entry["iso_baseline_makespan_ns"] = iso_baseline
        entry["derate"] = float(entry["full_call_median_ns"]) / float(iso_baseline)

    merged = copy.deepcopy(first)
    rank_targets = [copy.deepcopy(profile["target"]) for profile in profiles]
    merged["target"] = {
        "machine": first["target"]["machine"],
        "cpu": first["target"]["cpu"],
        "host_logical_cores": first["target"]["host_logical_cores"],
        "aggregate_profiled_cores": sum(
            int(target["cores_per_rank"]) for target in rank_targets
        ),
        "cores_per_rank": int(first["target"]["cores_per_rank"]),
        "concurrent_ranks": 2,
        "cpu_ids_by_rank": [target["cpu_ids"] for target in rank_targets],
        "numa_nodes": [target["numa_node"] for target in rank_targets],
        "llc_bytes_by_rank": [
            target["llc_bytes_per_rank"] for target in rank_targets
        ],
        "os": first["target"]["os"],
    }
    merged["measurement"]["rank_synchronization"] = "socket_barrier_per_call"
    merged["measurement"]["rank_aggregation"] = "median_of_pairwise_max"
    merged["measurement"]["profile_scope"] = "concurrent_rank_pair"
    merged["measurement"]["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    merged["isolated"] = isolated
    merged["iso_formula"] = fit_from_measurements(
        (entry["routes"], entry["threads"], entry["median_ns"])
        for entry in isolated
    ).to_dict()
    merged["entries"] = entries
    validate_profile(merged)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallel-mode", choices=("tp", "ep"), required=True)
    parser.add_argument("--parallel-degree", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--ffn-hidden-size", type=int, required=True)
    parser.add_argument("--global-experts", type=int, required=True)
    parser.add_argument("--local-experts", type=int, required=True)
    parser.add_argument(
        "--measurement-experts",
        type=int,
        default=0,
        help="Consecutive local experts per call; 0 profiles all local experts.",
    )
    parser.add_argument("--isolated-measurement-experts", type=int, default=8)
    parser.add_argument("--w13-split", type=int, choices=(0, 1), required=True)
    parser.add_argument("--w13-split-chunks", type=int, default=2)
    parser.add_argument("--cpu-groups", default="0-31;32-63")
    parser.add_argument("--numa-nodes", default="0,1")
    parser.add_argument("--llc-bytes", type=int, default=None)
    parser.add_argument("--isolated-route-buckets", default=DEFAULT_ISOLATED_ROUTES)
    parser.add_argument("--contention-route-buckets", default=DEFAULT_CONTENTION_ROUTES)
    parser.add_argument("--thread-buckets", default=DEFAULT_THREADS)
    parser.add_argument("--contention-shapes", default=DEFAULT_SHAPES)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prepack-threads", type=int, default=32)
    parser.add_argument("--keep-rank-profiles", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cpu_groups = split_nonempty(args.cpu_groups, ";")
    numa_nodes = [int(value) for value in split_nonempty(args.numa_nodes)]
    if len(cpu_groups) != 2 or len(numa_nodes) != 2:
        raise ValueError("dual-rank profiling requires two CPU groups and NUMA nodes")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(2)
    server.settimeout(1.0)
    port = int(server.getsockname()[1])

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "FUSED_CPP_MOE_SVE": "1",
            "FUSED_CPP_MOE_PREPACK_THREADS": str(args.prepack_threads),
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONUNBUFFERED": "1",
        }
    )

    with tempfile.TemporaryDirectory(prefix="moe-profile-v2-") as tmp_dir:
        rank_paths = [Path(tmp_dir) / f"rank{rank}.json" for rank in range(2)]
        processes: list[subprocess.Popen] = []
        connections: dict[int, socket.socket] = {}
        try:
            for rank in range(2):
                command = [
                    "numactl",
                    f"--cpunodebind={numa_nodes[rank]}",
                    f"--membind={numa_nodes[rank]}",
                    sys.executable,
                    str(WORKER),
                    "--output",
                    str(rank_paths[rank]),
                    "--hidden-size",
                    str(args.hidden_size),
                    "--ffn-hidden-size",
                    str(args.ffn_hidden_size),
                    "--num-experts",
                    str(args.local_experts),
                    "--global-experts",
                    str(args.global_experts),
                    "--measurement-experts",
                    str(args.measurement_experts),
                    "--isolated-measurement-experts",
                    str(args.isolated_measurement_experts),
                    "--parallel-mode",
                    args.parallel_mode,
                    "--parallel-degree",
                    str(args.parallel_degree),
                    "--w13-split",
                    str(args.w13_split),
                    "--w13-split-chunks",
                    str(args.w13_split_chunks),
                    "--cpu-ids",
                    cpu_groups[rank],
                    "--numa-node",
                    str(numa_nodes[rank]),
                    "--rank-id",
                    str(rank),
                    "--concurrent-ranks",
                    "2",
                    "--isolated-route-buckets",
                    args.isolated_route_buckets,
                    "--contention-route-buckets",
                    args.contention_route_buckets,
                    "--thread-buckets",
                    args.thread_buckets,
                    "--contention-shapes",
                    args.contention_shapes,
                    "--warmup",
                    str(args.warmup),
                    "--runs",
                    str(args.runs),
                    "--seed",
                    str(args.seed + rank),
                    "--sync-port",
                    str(port),
                    "--store-samples",
                ]
                if args.llc_bytes is not None:
                    command.extend(("--llc-bytes", str(args.llc_bytes)))
                processes.append(
                    subprocess.Popen(command, cwd=ROOT, env=env)
                )

            accept_deadline = time.monotonic() + 600.0
            while len(connections) != 2:
                try:
                    connection, _address = server.accept()
                except socket.timeout:
                    failed = [
                        (rank, process.returncode)
                        for rank, process in enumerate(processes)
                        if process.poll() is not None
                    ]
                    if failed:
                        raise RuntimeError(
                            f"rank profiler exited before synchronization: {failed}"
                        )
                    if time.monotonic() >= accept_deadline:
                        raise TimeoutError("rank profilers did not connect in 600 seconds")
                    continue
                connection.settimeout(600.0)
                rank = recv_exact(connection, 1)[0]
                if rank in connections:
                    raise RuntimeError(f"duplicate rank connection: {rank}")
                connections[rank] = connection

            isolated_points = len(split_nonempty(args.isolated_route_buckets)) * len(
                split_nonempty(args.thread_buckets)
            )
            contention_points = len(split_nonempty(args.contention_shapes, ";")) * len(
                split_nonempty(args.contention_route_buckets)
            )
            barriers = (isolated_points + contention_points) * (
                args.warmup + args.runs
            )
            for _ in range(barriers):
                for rank in range(2):
                    if recv_exact(connections[rank], 1) != b"R":
                        raise RuntimeError("invalid rank synchronization request")
                for rank in range(2):
                    connections[rank].sendall(b"G")

            for process in processes:
                if process.wait(timeout=600.0) != 0:
                    raise RuntimeError(f"rank profiler failed with {process.returncode}")

            if args.keep_rank_profiles:
                for rank, path in enumerate(rank_paths):
                    destination = args.output.with_name(
                        f"{args.output.stem}.rank{rank}{args.output.suffix}"
                    )
                    shutil.copy2(path, destination)
            merged = merge_profiles(rank_paths, args.output)
            print(
                f"wrote {args.output} "
                f"({len(merged['isolated'])} isolated, {len(merged['entries'])} contention)"
            )
        finally:
            server.close()
            for connection in connections.values():
                connection.close()
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
