#!/usr/bin/env python3
"""Validate policy-aware plans against exhaustive measured TP2/EP2 choices.

The parent launches one NUMA-local worker per rank and synchronizes every timed
operator call.  Each worker uses its rank-local histogram and may therefore
choose a different plan.  Reported wall samples are pairwise rank maxima.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COST_MODEL_DIR = ROOT / "cpu_moe_schedule_optimization" / "cost_model"
PLANNER_DIR = ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path[:0] = [str(ROOT / "src"), str(COST_MODEL_DIR), str(PLANNER_DIR)]


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    while size:
        chunk = sock.recv(size)
        if not chunk:
            raise RuntimeError("validation synchronization socket closed")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def sync(sock: socket.socket) -> None:
    sock.sendall(b"R")
    if recv_exact(sock, 1) != b"G":
        raise RuntimeError("invalid synchronization response")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def uniform_histogram(total_routes: int, experts: int) -> list[int]:
    base, remainder = divmod(total_routes, experts)
    return [base + (index < remainder) for index in range(experts)]


def hotspot_histogram(total_routes: int, experts: int) -> list[int]:
    if experts != 64 or total_routes != 12_288:
        raise ValueError("built-in hotspot currently requires 12288 routes/64 experts")
    # All counts stay M12-aligned: four hot, twelve warm, and 48 cold experts.
    return [768] * 4 + [384] * 12 + [96] * 48


def load_histograms(path: Path | None, total_routes: int, experts: int) -> dict:
    histograms = {
        "uniform": uniform_histogram(total_routes, experts),
        "hotspot": hotspot_histogram(total_routes, experts),
    }
    if path is None:
        return histograms
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "histograms" in payload:
        histograms = {}
        for name, raw_values in payload["histograms"].items():
            if not isinstance(raw_values, list) or len(raw_values) != experts:
                raise ValueError(f"routing histogram {name!r} must contain {experts} counts")
            values = [int(value) for value in raw_values]
            if any(value < 0 for value in values):
                raise ValueError(f"routing histogram {name!r} contains negatives")
            histograms[str(name)] = values
        if not histograms:
            raise ValueError("histograms mapping must not be empty")
        return histograms
    values = payload.get("histogram", payload) if isinstance(payload, dict) else payload
    if isinstance(payload, dict) and "top_experts" in payload:
        from workload_catalog import load_routing_workload

        captured = load_routing_workload(path)
        values = [0] * experts
        for expert, routes in enumerate(captured.histogram):
            values[expert % experts] += routes
    if not isinstance(values, list) or len(values) != experts:
        raise ValueError(f"routing histogram must contain {experts} counts")
    trace = [int(value) for value in values]
    if any(value < 0 for value in trace):
        raise ValueError("routing counts must be non-negative")
    histograms["trace"] = trace
    return histograms


def select_models(profile_dir: Path, mode: str):
    from phase_model import ContentionCostModel
    from profile_catalog import ProfileCatalog, ProfileQuery

    ffn = 1024 if mode == "tp" else 2048
    local_experts = 64 if mode == "tp" else 32
    catalog = ProfileCatalog.from_directory(profile_dir, "*_v2_r1_20260713.json")
    query = ProfileQuery(
        mode=mode,
        degree=2,
        hidden_size=4096,
        intermediate_size=ffn,
        global_experts=64,
        local_experts=local_experts,
        backend="sve",
        backend_n_tile=8,
        sve_implementation="asm",
        m_tail_policy="static_bucketed",
        activation="silu",
        dtype="bf16",
        measurement_experts=local_experts,
        cores_per_rank=32,
        concurrent_ranks=2,
    )
    no_split, split = catalog.stage_range_pair(query)
    return (
        ContentionCostModel(no_split.path),
        ContentionCostModel(split.path),
    )


def parse_shape(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    shape = tuple(int(item) for item in value.split(",") if item.strip())
    if not shape or any(width <= 0 for width in shape):
        raise ValueError(f"invalid forced shape: {value!r}")
    return shape


def build_call(
    packed,
    hidden,
    ids,
    weights,
    bridge,
    split_w13: bool,
    skip_weighted: bool,
):
    from fused_cpp.moe import AsyncMoEPlanV2, fused_moe_bf16_tiled_async_plan

    plan = AsyncMoEPlanV2.from_dict(bridge)

    def run():
        return fused_moe_bf16_tiled_async_plan(
            hidden,
            packed,
            weights,
            ids,
            plan,
            activation="silu",
            skip_weighted=skip_weighted,
            w13_split=split_w13,
        )

    return run


def local_histogram(global_histogram: list[int], mode: str, rank: int) -> list[int]:
    if mode == "tp":
        return list(global_histogram)
    local = len(global_histogram) // 2
    return list(global_histogram[rank * local : (rank + 1) * local])


def worker(args: argparse.Namespace) -> int:
    import torch
    from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights
    from interval_planner import IntervalPlanner, PolicyAwarePlanner

    torch.set_num_threads(1)
    models = select_models(args.profile_dir, args.mode)
    assert models[0].policy is not None
    cpu_ids = models[0].policy.cpu_ids_by_rank[args.rank]
    planners = tuple(IntervalPlanner(model, 32, cpu_ids=cpu_ids) for model in models)
    joint = PolicyAwarePlanner(models, 32, cpu_ids=cpu_ids)
    ffn = 1024 if args.mode == "tp" else 2048
    local_experts = 64 if args.mode == "tp" else 32

    generator = torch.Generator().manual_seed(args.seed + args.rank)
    w13 = torch.empty((local_experts, 2 * ffn, 4096), dtype=torch.bfloat16).normal_(0.0, 0.01, generator=generator)
    w2 = torch.empty((local_experts, 4096, ffn), dtype=torch.bfloat16).normal_(0.0, 0.01, generator=generator)
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
    del w13, w2

    global_histograms = load_histograms(args.routes_json, 12_288, 64)
    sock = socket.create_connection(("127.0.0.1", args.port), timeout=1800.0)
    sock.sendall(bytes([args.rank]))
    result: dict[str, object] = {
        "mode": args.mode,
        "rank": args.rank,
        "cases": {},
    }
    for case_name, global_counts in global_histograms.items():
        counts = local_histogram(global_counts, args.mode, args.rank)
        experts = [(expert, routes) for expert, routes in enumerate(counts) if routes]
        forced_shape = parse_shape(args.force_shape)
        if forced_shape is None:
            selected = joint.plan(experts)
        else:
            forced_split = bool(args.force_split)
            forced_planner = planners[int(forced_split)]
            predicted, tasks = forced_planner.score_shape(experts, forced_shape)
            selected = {
                "w13_split": forced_split,
                "shape": forced_shape,
                "makespan_ns": predicted,
                "bridge": forced_planner.to_async_bridge(tasks),
            }
        candidates: list[dict] = [
            {
                "key": "planner",
                "split": bool(selected["w13_split"]),
                "shape": tuple(selected["shape"]),
                "predicted_ns": float(selected["makespan_ns"]),
                "bridge": selected["bridge"],
            }
        ]
        if args.noise_only:
            candidates.append({**candidates[0], "key": "planner_clone"})
        else:
            for planner in planners:
                split = bool(planner.model.policy.w13_split)
                for shape in planner.shapes:
                    predicted, tasks = planner.score_shape(experts, shape)
                    candidates.append(
                        {
                            "key": f"{'split' if split else 'nosplit'}:{','.join(map(str, shape))}",
                            "split": split,
                            "shape": tuple(shape),
                            "predicted_ns": float(predicted),
                            "bridge": planner.to_async_bridge(tasks),
                        }
                    )

        total_routes = sum(counts)
        if total_routes % args.top_k:
            raise ValueError(f"local routes {total_routes} are not divisible by top_k={args.top_k}")
        tokens = total_routes // args.top_k
        hidden = torch.empty((tokens, 4096), dtype=torch.bfloat16).normal_(0.0, 0.01, generator=generator)
        ids = torch.cat(
            [torch.full((routes,), expert, dtype=torch.int32) for expert, routes in enumerate(counts) if routes]
        ).reshape(tokens, args.top_k)
        weights = torch.full((tokens, args.top_k), 1.0 / args.top_k, dtype=torch.float32)
        calls = {
            candidate["key"]: build_call(
                packed,
                hidden,
                ids,
                weights,
                candidate["bridge"],
                candidate["split"],
                args.top_k == 1,
            )
            for candidate in candidates
        }

        for _ in range(args.warmup):
            for candidate in candidates:
                sync(sock)
                output = calls[candidate["key"]]()
                _ = int(output.view(torch.int16)[0, 0])

        samples = {candidate["key"]: [] for candidate in candidates}
        for iteration in range(args.runs):
            shift = iteration % len(candidates)
            order = candidates[shift:] + candidates[:shift]
            if (iteration // len(candidates)) % 2:
                order.reverse()
            for candidate in order:
                sync(sock)
                begin = time.perf_counter_ns()
                output = calls[candidate["key"]]()
                _ = int(output.view(torch.int16)[0, 0])
                samples[candidate["key"]].append(time.perf_counter_ns() - begin)

        if args.stage_trace_file is not None:
            os.environ["FUSED_CPP_MOE_TRACE"] = "1"
            os.environ["FUSED_CPP_MOE_TRACE_FILE"] = str(args.stage_trace_file)
            try:
                sync(sock)
                output = calls["planner"]()
                _ = int(output.view(torch.int16)[0, 0])
            finally:
                os.environ.pop("FUSED_CPP_MOE_TRACE", None)
                os.environ.pop("FUSED_CPP_MOE_TRACE_FILE", None)

        result["cases"][case_name] = {
            "histogram": counts,
            "candidates": {
                candidate["key"]: {
                    "split": candidate["split"],
                    "shape": candidate["shape"],
                    "predicted_ns": candidate["predicted_ns"],
                    "samples_ns": samples[candidate["key"]],
                }
                for candidate in candidates
            },
        }
    sock.close()
    args.worker_output.write_text(json.dumps(result), encoding="utf-8")
    return 0


def run_pair(args: argparse.Namespace, mode: str) -> dict:
    models = select_models(args.profile_dir, mode)
    assert models[0].policy is not None
    numa_nodes = models[0].policy.numa_nodes
    if len(numa_nodes) != 2:
        raise ValueError("dual-rank validation requires two profiled NUMA nodes")
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
            "FUSED_CPP_MOE_PREPACK_THREADS": "32",
            "PYTHONPATH": str(ROOT / "src"),
        }
    )
    if args.stage_trace_dir is not None:
        args.stage_trace_dir.mkdir(parents=True, exist_ok=True)
        for rank in range(2):
            trace_file = args.stage_trace_dir / f"{mode}_rank{rank}.log"
            trace_file.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(prefix="moe-policy-validation-") as tmp:
        outputs = [Path(tmp) / f"rank{rank}.json" for rank in range(2)]
        processes = []
        connections: dict[int, socket.socket] = {}
        try:
            for rank in range(2):
                command = [
                    "numactl",
                    f"--cpunodebind={numa_nodes[rank]}",
                    f"--membind={numa_nodes[rank]}",
                    sys.executable,
                    __file__,
                    "--worker",
                    "--mode",
                    mode,
                    "--rank",
                    str(rank),
                    "--port",
                    str(port),
                    "--worker-output",
                    str(outputs[rank]),
                    "--profile-dir",
                    str(args.profile_dir),
                    "--warmup",
                    str(args.warmup),
                    "--runs",
                    str(args.runs),
                    "--seed",
                    str(args.seed),
                    "--top-k",
                    str(args.top_k),
                ]
                if args.noise_only:
                    command.append("--noise-only")
                if args.force_shape is not None:
                    command.extend(("--force-shape", args.force_shape))
                    command.extend(("--force-split", str(args.force_split)))
                if args.stage_trace_dir is not None:
                    trace_file = args.stage_trace_dir / f"{mode}_rank{rank}.log"
                    command.extend(("--stage-trace-file", str(trace_file)))
                if args.routes_json is not None:
                    command.extend(("--routes-json", str(args.routes_json)))
                processes.append(
                    subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )
            while len(connections) < 2:
                for rank, process in enumerate(processes):
                    if process.poll() is not None and rank not in connections:
                        stdout, stderr = process.communicate()
                        raise RuntimeError(f"{mode} rank{rank} failed before connecting:\n{stdout}\n{stderr}")
                try:
                    connection, _ = server.accept()
                except TimeoutError:
                    continue
                connection.settimeout(1800.0)
                rank = recv_exact(connection, 1)[0]
                connections[rank] = connection

            case_count = len(load_histograms(args.routes_json, 12_288, 64))
            shape_count = len(models[0].supported_shapes)
            candidate_count = 2 if args.noise_only else 1 + 2 * shape_count
            barriers = case_count * candidate_count * (args.warmup + args.runs)
            if args.stage_trace_dir is not None:
                barriers += case_count
            for _ in range(barriers):
                for rank in range(2):
                    if recv_exact(connections[rank], 1) != b"R":
                        raise RuntimeError("invalid worker barrier request")
                for rank in range(2):
                    connections[rank].sendall(b"G")

            worker_results = []
            for rank, process in enumerate(processes):
                stdout, stderr = process.communicate(timeout=1800.0)
                if process.returncode:
                    raise RuntimeError(f"{mode} rank{rank} failed:\n{stdout}\n{stderr}")
                worker_results.append(json.loads(outputs[rank].read_text()))
        finally:
            server.close()
            for connection in connections.values():
                connection.close()
            for process in processes:
                if process.poll() is None:
                    process.terminate()

    merged = {"mode": mode, "cases": {}}
    for case_name in worker_results[0]["cases"]:
        rank_cases = [result["cases"][case_name] for result in worker_results]
        rows = []
        for key in rank_cases[0]["candidates"]:
            entries = [case["candidates"][key] for case in rank_cases]
            paired = [max(left, right) for left, right in zip(entries[0]["samples_ns"], entries[1]["samples_ns"])]
            row = {
                "key": key,
                "rank_plans": [{"split": entry["split"], "shape": entry["shape"]} for entry in entries],
                "predicted_ns": max(entry["predicted_ns"] for entry in entries),
                "median_ns": statistics.median(paired),
                "p10_ns": percentile(paired, 0.10),
                "p90_ns": percentile(paired, 0.90),
                "rank_median_ns": [statistics.median(entry["samples_ns"]) for entry in entries],
                "rank_p10_ns": [percentile(entry["samples_ns"], 0.10) for entry in entries],
                "rank_p90_ns": [percentile(entry["samples_ns"], 0.90) for entry in entries],
            }
            if args.noise_only:
                row["samples_ns"] = paired
            rows.append(row)
        fixed = [row for row in rows if row["key"] != "planner"]
        best = min(fixed, key=lambda row: row["median_ns"])
        selected = next(row for row in rows if row["key"] == "planner")
        raw_regret = selected["median_ns"] / best["median_ns"] - 1.0
        selected["raw_regret"] = raw_regret
        selected["regret"] = max(raw_regret, 0.0)
        selected["prediction_error"] = selected["predicted_ns"] / selected["median_ns"] - 1.0
        merged["cases"][case_name] = {
            "rank_histograms": [case["histogram"] for case in rank_cases],
            "best_fixed": best,
            "selected": selected,
            "candidates": sorted(rows, key=lambda row: row["median_ns"]),
        }
    return merged


def print_summary(result: dict) -> None:
    print(result["mode"].upper())
    print(
        f"{'case':<10} {'selected rank plans':<48} {'pred ms':>9} "
        f"{'actual ms':>10} {'best ms':>9} {'regret':>8} {'error':>8}"
    )
    for name, case in result["cases"].items():
        selected = case["selected"]
        best = case["best_fixed"]
        plans = "/".join(f"{'S' if plan['split'] else 'N'}:{tuple(plan['shape'])}" for plan in selected["rank_plans"])
        print(
            f"{name:<10} {plans:<48} {selected['predicted_ns'] / 1e6:9.3f} "
            f"{selected['median_ns'] / 1e6:10.3f} {best['median_ns'] / 1e6:9.3f} "
            f"{selected['regret'] * 100:7.2f}% "
            f"{selected['prediction_error'] * 100:+7.2f}%"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=("tp", "ep", "both"), default="both")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=COST_MODEL_DIR / "profiles",
    )
    parser.add_argument("--routes-json", type=Path)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--force-shape")
    parser.add_argument("--force-split", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--noise-only",
        action="store_true",
        help="measure the selected schedule and an identical clone only",
    )
    parser.add_argument("--stage-trace-dir", type=Path)
    parser.add_argument("--stage-trace-file", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        if None in (args.rank, args.port, args.worker_output):
            raise ValueError("worker mode requires rank, port, and worker-output")
        if args.mode == "both":
            raise ValueError("worker mode requires a concrete parallel mode")
        return worker(args)

    modes = ("tp", "ep") if args.mode == "both" else (args.mode,)
    result = {
        "schema_version": 1,
        "warmup": args.warmup,
        "runs": args.runs,
        "top_k": args.top_k,
        "results": [],
    }
    for mode in modes:
        mode_result = run_pair(args, mode)
        result["results"].append(mode_result)
        print_summary(mode_result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
