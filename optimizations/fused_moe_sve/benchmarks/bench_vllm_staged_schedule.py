#!/usr/bin/env python3
"""Compare production, team-transition, and vLLM-style MoE schedules."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib
import json
import math
import os
import random
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
COST_MODEL_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"
PLANNER_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path[:0] = [str(REPO_ROOT / "src"), str(COST_MODEL_DIR), str(PLANNER_DIR)]

from fused_cpp.moe import (  # noqa: E402
    fused_moe_bf16_tiled_async,
    fused_moe_bf16_tiled_vllm_staged,
    prepare_fused_moe_bf16_tiled_weights,
)
from workload_catalog import synthetic_offline_workloads  # noqa: E402


PAPER_WORKLOADS = synthetic_offline_workloads()
DYNAMIC_POOL_VARIANT = "dynamic_16t_to_4t_pool"
STATIC_SPLIT_VARIANT = "static_16t_to_4x4t"
VLLM_VARIANT = "vllm_staged"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--team-threads", type=int, default=0)
    routing = parser.add_mutually_exclusive_group()
    routing.add_argument("--distribution", choices=("hot-topk", "round-robin"))
    routing.add_argument("--preset", choices=sorted(PAPER_WORKLOADS))
    parser.add_argument(
        "--production-profile",
        type=Path,
        help="schema-v2 contention profile; enables the production PlannedMoE schedule",
    )
    parser.add_argument(
        "--production-ready-token-merge",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable production ready-token merge (default: enabled with --production-profile)",
    )
    parser.add_argument(
        "--static-16-to-4",
        action="store_true",
        help="benchmark a static DAG that releases each 16T long team as four 4T short-team lanes",
    )
    parser.add_argument(
        "--dynamic-short-pool",
        action="store_true",
        help="send each released 4T group to one global short-expert task pool",
    )
    parser.add_argument(
        "--static-long-route-threshold",
        type=int,
        default=12,
        help="experts above this route count use the 16T head of the static split DAG",
    )
    parser.add_argument("--route-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--stage-timing", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def materialize_topk_ids(
    histogram: list[int] | tuple[int, ...],
    *,
    tokens: int,
    top_k: int,
    seed: int,
) -> torch.Tensor:
    """Build exact per-expert degrees with no duplicate expert inside a token."""
    counts = [int(value) for value in histogram]
    if tokens <= 0 or top_k <= 0:
        raise ValueError("tokens and top-k must be positive")
    if any(value < 0 or value > tokens for value in counts):
        raise ValueError("each route count must be in [0, tokens]")
    if sum(counts) != tokens * top_k:
        raise ValueError("route counts must sum to tokens * top-k")

    remaining = [(-count, expert) for expert, count in enumerate(counts) if count]
    heapq.heapify(remaining)
    rows: list[list[int]] = []
    for _ in range(tokens):
        if len(remaining) < top_k:
            raise ValueError("route histogram cannot form distinct TopK rows")
        selected = [heapq.heappop(remaining) for _ in range(top_k)]
        rows.append([expert for _, expert in selected])
        for negative_count, expert in selected:
            if negative_count < -1:
                heapq.heappush(remaining, (negative_count + 1, expert))
    if remaining:
        raise RuntimeError("route materialization left unassigned expert degrees")

    generator = random.Random(seed)
    generator.shuffle(rows)
    for row in rows:
        generator.shuffle(row)
    return torch.tensor(rows, dtype=torch.int32)


def make_topk_ids(args: argparse.Namespace) -> torch.Tensor:
    if args.preset is not None:
        workload = PAPER_WORKLOADS[args.preset]
        expected = (workload.tokens, workload.top_k, workload.num_experts)
        actual = (args.tokens, args.top_k, args.experts)
        if actual != expected:
            raise ValueError(f"preset {args.preset!r} requires tokens/top-k/experts={expected}, got {actual}")
        return materialize_topk_ids(
            workload.histogram,
            tokens=args.tokens,
            top_k=args.top_k,
            seed=args.seed,
        )
    distribution = args.distribution or "hot-topk"
    if distribution == "hot-topk":
        return torch.arange(args.top_k, dtype=torch.int32).repeat(args.tokens, 1)
    return torch.tensor(
        [[(token * args.top_k + slot) % args.experts for slot in range(args.top_k)] for token in range(args.tokens)],
        dtype=torch.int32,
    )


def make_fixed_team_schedule(
    active_experts: torch.Tensor,
    *,
    threads: int,
    requested_team_threads: int,
) -> tuple[torch.Tensor, ...]:
    active_count = int(active_experts.numel())
    if active_count == 0:
        raise ValueError("at least one expert must be active")
    team_threads = requested_team_threads or max(1, threads // min(active_count, threads))
    if team_threads <= 0 or team_threads > threads:
        raise ValueError(f"team-threads must be in [1, {threads}], got {team_threads}")
    slots = threads // team_threads
    if slots <= 0:
        raise ValueError("team-threads leaves no runnable team slot")

    core_begins: list[int] = []
    team_widths: list[int] = []
    dep_offsets = [0]
    deps: list[int] = []
    previous_by_slot = [-1] * slots
    for task_id in range(active_count):
        slot = task_id % slots
        core_begins.append(slot * team_threads)
        team_widths.append(team_threads)
        previous = previous_by_slot[slot]
        if previous >= 0:
            deps.append(previous)
        dep_offsets.append(len(deps))
        previous_by_slot[slot] = task_id

    return (
        active_experts.to(torch.int32),
        torch.tensor(core_begins, dtype=torch.int32),
        torch.tensor(team_widths, dtype=torch.int32),
        torch.tensor(dep_offsets, dtype=torch.int32),
        torch.tensor(deps, dtype=torch.int32),
    )


def make_static_16t_to_4x4t_schedule(
    route_counts: torch.Tensor,
    *,
    threads: int,
    long_route_threshold: int,
    iso_time_ns: Callable[[int, int], float],
) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
    """Build a fixed DAG whose 16T long heads release four 4T short lanes."""
    long_team_threads = 16
    short_team_threads = 4
    if route_counts.dim() != 1:
        raise ValueError(f"route_counts must be one-dimensional, got shape={tuple(route_counts.shape)}")
    if threads <= 0 or threads % long_team_threads != 0:
        raise ValueError(f"threads must be a positive multiple of {long_team_threads}, got {threads}")
    if long_route_threshold < 0:
        raise ValueError(f"static long-route threshold must be non-negative, got {long_route_threshold}")

    jobs = [(expert, int(routes)) for expert, routes in enumerate(route_counts.tolist()) if routes > 0]
    long_jobs = [(expert, routes) for expert, routes in jobs if routes > long_route_threshold]
    short_jobs = [(expert, routes) for expert, routes in jobs if routes <= long_route_threshold]
    superteam_count = threads // long_team_threads
    if not long_jobs or not short_jobs:
        raise ValueError(
            "static 16T -> 4x4T schedule requires both long and short active experts: "
            f"long={len(long_jobs)} short={len(short_jobs)} threshold={long_route_threshold}"
        )
    if len(long_jobs) > superteam_count:
        raise ValueError(
            "static 16T -> 4x4T schedule supports at most one long head per 16T superteam: "
            f"long={len(long_jobs)} superteams={superteam_count}"
        )

    iso_costs: dict[tuple[int, int], float] = {}

    def isolated_cost(routes: int, team_threads: int) -> float:
        key = (routes, team_threads)
        cost = iso_costs.get(key)
        if cost is None:
            cost = float(iso_time_ns(routes, team_threads))
            if not math.isfinite(cost) or cost <= 0.0:
                raise ValueError(
                    f"isolated time must be finite and positive for routes={routes}, threads={team_threads}, got {cost}"
                )
            iso_costs[key] = cost
        return cost

    long_jobs.sort(key=lambda job: (-isolated_cost(job[1], long_team_threads), -job[1], job[0]))
    short_jobs.sort(key=lambda job: (-isolated_cost(job[1], short_team_threads), -job[1], job[0]))

    short_lanes_per_superteam = long_team_threads // short_team_threads
    short_lane_count = threads // short_team_threads
    parent_by_superteam: list[tuple[int, int] | None] = [None] * superteam_count
    lane_load_ns = [0.0] * short_lane_count
    lane_release_ns = [0.0] * short_lane_count
    lane_jobs: list[list[int]] = [[] for _ in range(short_lane_count)]
    for superteam, job in enumerate(long_jobs):
        parent_by_superteam[superteam] = job
        release_ns = isolated_cost(job[1], long_team_threads)
        lane_begin = superteam * short_lanes_per_superteam
        for lane in range(lane_begin, lane_begin + short_lanes_per_superteam):
            lane_load_ns[lane] = release_ns
            lane_release_ns[lane] = release_ns

    for expert, routes in short_jobs:
        lane = min(range(short_lane_count), key=lambda candidate: (lane_load_ns[candidate], candidate))
        lane_jobs[lane].append(expert)
        lane_load_ns[lane] += isolated_cost(routes, short_team_threads)

    task_experts: list[int] = []
    task_core_begins: list[int] = []
    task_threads: list[int] = []
    task_dep_offsets = [0]
    task_deps: list[int] = []

    def append_task(expert: int, core_begin: int, team_threads: int, dependency: int | None) -> int:
        task_id = len(task_experts)
        task_experts.append(expert)
        task_core_begins.append(core_begin)
        task_threads.append(team_threads)
        if dependency is not None:
            task_deps.append(dependency)
        task_dep_offsets.append(len(task_deps))
        return task_id

    parent_task_by_superteam: list[int | None] = [None] * superteam_count
    for superteam, parent in enumerate(parent_by_superteam):
        if parent is None:
            continue
        parent_task_by_superteam[superteam] = append_task(
            parent[0],
            superteam * long_team_threads,
            long_team_threads,
            None,
        )

    for lane, experts in enumerate(lane_jobs):
        superteam = lane // short_lanes_per_superteam
        previous = parent_task_by_superteam[superteam]
        for expert in experts:
            previous = append_task(
                expert,
                lane * short_team_threads,
                short_team_threads,
                previous,
            )

    schedule = (
        torch.tensor(task_experts, dtype=torch.int32),
        torch.tensor(task_core_begins, dtype=torch.int32),
        torch.tensor(task_threads, dtype=torch.int32),
        torch.tensor(task_dep_offsets, dtype=torch.int32),
        torch.tensor(task_deps, dtype=torch.int32),
    )
    metadata = {
        "variant": STATIC_SPLIT_VARIANT,
        "long_route_threshold": long_route_threshold,
        "long_team_threads": long_team_threads,
        "short_team_threads": short_team_threads,
        "long_tasks": len(long_jobs),
        "short_tasks": len(short_jobs),
        "long_experts": [expert for expert, _ in long_jobs],
        "short_lane_task_counts": [len(experts) for experts in lane_jobs],
        "short_lane_release_ms": [value / 1.0e6 for value in lane_release_ns],
        "short_lane_finish_ms": [value / 1.0e6 for value in lane_load_ns],
        "predicted_isolated_makespan_ms": max(lane_load_ns) / 1.0e6,
    }
    return schedule, metadata


def make_production_schedule(
    route_counts: torch.Tensor,
    *,
    profile: Path,
    threads: int,
    cpu_ids: list[int],
    hidden: int,
    intermediate: int,
    experts: int,
) -> tuple[tuple[torch.Tensor, ...], dict[str, object], Callable[[int, int], float]]:
    from phase_model import ContentionCostModel
    from planned_moe import PlannedMoE

    model = ContentionCostModel(profile)
    policy = model.policy
    if policy is None:
        raise ValueError(f"production profile must use schema v2: {profile}")
    expected = (hidden, intermediate, experts, threads, True)
    actual = (
        policy.hidden_size,
        policy.intermediate_size,
        policy.local_experts,
        policy.cores_per_rank,
        policy.w13_split,
    )
    if actual != expected:
        raise ValueError(
            "production profile hidden/intermediate/local-experts/cores/split mismatch: "
            f"expected {expected}, got {actual}"
        )

    native_module = importlib.import_module("fused_cpp._moe_C")
    runtime_extension_path = Path(native_module.__file__)
    runtime_extension_sha256 = hashlib.sha256(runtime_extension_path.read_bytes()).hexdigest()
    counts = [(expert, int(routes)) for expert, routes in enumerate(route_counts.tolist()) if routes > 0]
    planner = PlannedMoE(model, threads, cpu_ids=cpu_ids)
    begin = time.perf_counter_ns()
    planner.plan_spec_for(counts)
    cold_plan_ns = time.perf_counter_ns() - begin
    begin = time.perf_counter_ns()
    spec = planner.plan_spec_for(counts)
    warm_plan_ns = time.perf_counter_ns() - begin
    bridge = spec["bridge"]
    schedule = (
        torch.tensor(bridge["task_expert_ids"], dtype=torch.int32),
        torch.tensor(bridge["task_core_begins"], dtype=torch.int32),
        torch.tensor(bridge["task_threads"], dtype=torch.int32),
        torch.tensor(bridge["task_dep_offsets"], dtype=torch.int32),
        torch.tensor(bridge["task_deps"], dtype=torch.int32),
    )
    return schedule, {
        "profile": str(profile),
        "profile_extension_sha256": policy.extension_sha256,
        "runtime_extension_sha256": runtime_extension_sha256,
        "extension_hash_match": runtime_extension_sha256 == policy.extension_sha256,
        "shape": list(spec["shape"]),
        "w13_split": bool(spec["w13_split"]),
        "cold_plan_ms": cold_plan_ns / 1.0e6,
        "warm_plan_ms": warm_plan_ns / 1.0e6,
    }, model.T_iso


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    positive = (
        args.tokens,
        args.hidden,
        args.intermediate,
        args.experts,
        args.top_k,
        args.threads,
        args.runs,
    )
    if min(positive) <= 0 or args.warmup < 0:
        raise ValueError("shapes, threads, and runs must be positive; warmup must be non-negative")
    if args.top_k > args.experts:
        raise ValueError("top-k cannot exceed experts")
    if args.hidden % 8 != 0 or args.intermediate % 8 != 0:
        raise ValueError("hidden and intermediate must be multiples of 8")
    if args.production_profile is not None and not args.production_profile.is_file():
        raise ValueError(f"production profile does not exist: {args.production_profile}")
    if args.static_16_to_4 and args.production_profile is None:
        raise ValueError("--static-16-to-4 requires --production-profile for isolated-time lane assignment")
    if args.dynamic_short_pool and args.production_profile is None:
        raise ValueError("--dynamic-short-pool requires --production-profile")
    if args.dynamic_short_pool and (args.static_long_route_threshold <= 0 or args.threads % 4 != 0):
        raise ValueError("--dynamic-short-pool requires a positive route threshold and threads divisible by 4")

    affinity = sorted(os.sched_getaffinity(0))
    if args.threads > len(affinity):
        raise ValueError(f"threads={args.threads} exceeds affinity size {len(affinity)}")
    cpu_ids = affinity[: args.threads]
    thread_cpu_ids = torch.tensor(cpu_ids, dtype=torch.int32)
    torch.set_num_threads(1)

    topk_ids = make_topk_ids(args)
    route_counts = torch.bincount(topk_ids.flatten().to(torch.int64), minlength=args.experts)
    active_experts = torch.nonzero(route_counts, as_tuple=False).flatten()
    planner_metadata: dict[str, object] | None = None
    iso_time_ns: Callable[[int, int], float] | None = None
    if args.production_profile is None:
        baseline_variant = "fixed_team_async"
        schedule = make_fixed_team_schedule(
            active_experts,
            threads=args.threads,
            requested_team_threads=args.team_threads,
        )
        team_threads = int(schedule[2][0])
        w13_split = True
    else:
        baseline_variant = "production_async"
        schedule, planner_metadata, iso_time_ns = make_production_schedule(
            route_counts,
            profile=args.production_profile,
            threads=args.threads,
            cpu_ids=cpu_ids,
            hidden=args.hidden,
            intermediate=args.intermediate,
            experts=args.experts,
        )
        team_threads = None
        w13_split = bool(planner_metadata["w13_split"])
    static_schedule: tuple[torch.Tensor, ...] | None = None
    static_metadata: dict[str, object] | None = None
    variant_names = [baseline_variant]
    if args.static_16_to_4:
        assert iso_time_ns is not None
        static_schedule, static_metadata = make_static_16t_to_4x4t_schedule(
            route_counts,
            threads=args.threads,
            long_route_threshold=args.static_long_route_threshold,
            iso_time_ns=iso_time_ns,
        )
        variant_names.append(STATIC_SPLIT_VARIANT)
    if args.dynamic_short_pool:
        variant_names.append(DYNAMIC_POOL_VARIANT)
    variant_names.append(VLLM_VARIANT)
    variants = tuple(variant_names)
    ready_token_merge = args.production_ready_token_merge
    if ready_token_merge is None:
        ready_token_merge = args.production_profile is not None

    generator = torch.Generator().manual_seed(args.seed)
    hidden = torch.empty((args.tokens, args.hidden), dtype=torch.bfloat16)
    hidden.normal_(mean=0.0, std=0.01, generator=generator)
    w13 = torch.empty((args.experts, 2 * args.intermediate, args.hidden), dtype=torch.bfloat16)
    w13.normal_(mean=0.0, std=0.01, generator=generator)
    w2 = torch.empty((args.experts, args.hidden, args.intermediate), dtype=torch.bfloat16)
    w2.normal_(mean=0.0, std=0.01, generator=generator)
    topk_weights = torch.softmax(torch.randn((args.tokens, args.top_k), generator=generator), dim=-1)

    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_W13_SPLIT_N"] = "1"
    os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE"] = "1" if ready_token_merge else "0"
    os.environ["FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "1" if args.route_dtype == "bf16" else "0"
    os.environ["FUSED_CPP_MOE_SVE_ROUTE_MERGE_UNROLL"] = "1"
    os.environ["FUSED_CPP_MOE_ASYNC_SHORT_POOL_THREADS"] = "0"
    os.environ["FUSED_CPP_MOE_ASYNC_SHORT_POOL_MAX_ROWS"] = str(args.static_long_route_threshold)
    os.environ["FUSED_CPP_MOE_STAGE_TIMING"] = "0"
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="arm_sve_bf16")
    if packed.gemm_backend != 1:
        raise RuntimeError("benchmark requires the SVE BF16 fused MoE backend")
    del w13, w2

    outputs = {name: torch.empty_like(hidden) for name in variants}

    def run(name: str) -> torch.Tensor:
        os.environ["FUSED_CPP_MOE_ASYNC_SHORT_POOL_THREADS"] = "4" if name == DYNAMIC_POOL_VARIANT else "0"
        if name != VLLM_VARIANT:
            selected_schedule = schedule
            if name == STATIC_SPLIT_VARIANT:
                assert static_schedule is not None
                selected_schedule = static_schedule
            return fused_moe_bf16_tiled_async(
                hidden,
                packed,
                topk_weights,
                topk_ids,
                *selected_schedule,
                thread_cpu_ids=thread_cpu_ids,
                num_threads=args.threads,
                global_num_experts=args.experts,
                w13_split=w13_split,
                out=outputs[name],
            )
        return fused_moe_bf16_tiled_vllm_staged(
            hidden,
            packed,
            topk_weights,
            topk_ids,
            thread_cpu_ids=thread_cpu_ids,
            num_threads=args.threads,
            global_num_experts=args.experts,
            out=outputs[name],
        )

    reference = run(baseline_variant).clone()
    for name in variants:
        if name == baseline_variant:
            continue
        candidate = run(name).clone()
        torch.testing.assert_close(candidate.float(), reference.float(), atol=0, rtol=0)

    warmup_order = random.Random(args.seed ^ 0xA5A5)
    for _ in range(args.warmup):
        order = list(variants)
        warmup_order.shuffle(order)
        for name in order:
            run(name)

    samples = {name: [] for name in variants}
    timed_order = random.Random(args.seed ^ 0x5A5A)
    sink = 0
    for _ in range(args.runs):
        order = list(variants)
        timed_order.shuffle(order)
        for name in order:
            begin = time.perf_counter_ns()
            result = run(name)
            samples[name].append((time.perf_counter_ns() - begin) / 1.0e6)
            sink ^= int(result.view(torch.int16)[0, 0])

    if args.stage_timing:
        os.environ["FUSED_CPP_MOE_STAGE_TIMING"] = "1"
        for name in variants:
            run(name)
        os.environ["FUSED_CPP_MOE_STAGE_TIMING"] = "0"

    total_flops = 6 * args.tokens * args.top_k * args.hidden * args.intermediate
    baseline_ms = statistics.median(samples[baseline_variant])
    records: list[dict[str, object]] = []
    for name in variants:
        median_ms = statistics.median(samples[name])
        records.append(
            {
                "variant": name,
                "median_ms": median_ms,
                "p10_ms": percentile(samples[name], 0.10),
                "p90_ms": percentile(samples[name], 0.90),
                "aggregate_tflops": total_flops / median_ms / 1.0e9,
                "gain_pct": 100.0 * (baseline_ms / median_ms - 1.0),
                "samples_ms": samples[name],
            }
        )

    result = {
        "shape": {
            "tokens": args.tokens,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "experts": args.experts,
            "top_k": args.top_k,
            "threads": args.threads,
            "distribution": args.preset or args.distribution or "hot-topk",
            "active_experts": int(active_experts.numel()),
            "team_threads": team_threads,
            "route_dtype": args.route_dtype,
            "route_counts": route_counts.tolist(),
        },
        "method": {
            "baseline_variant": baseline_variant,
            "w13_split": w13_split,
            "async_ready_token_merge": ready_token_merge,
            "direct_route_store": True,
            "route_merge_unroll": 1,
            "planner": planner_metadata,
            "static_16_to_4": static_metadata,
            "dynamic_short_pool": {
                "pool_threads": 4,
                "max_rows": args.static_long_route_threshold,
            }
            if args.dynamic_short_pool
            else None,
            "warmup": args.warmup,
            "runs": args.runs,
        },
        "records": records,
        "sink": sink,
    }

    if planner_metadata is not None:
        if not planner_metadata["extension_hash_match"]:
            print(
                "WARNING: runtime extension hash differs from the calibration profile: "
                f"{planner_metadata['runtime_extension_sha256']} != "
                f"{planner_metadata['profile_extension_sha256']}"
            )
        print(
            f"production shape={tuple(planner_metadata['shape'])} "
            f"plan_ms(cold/warm)={planner_metadata['cold_plan_ms']:.3f}/{planner_metadata['warm_plan_ms']:.3f}"
        )
    print("variant                  median_ms   TFLOP/s   gain_pct     p10_ms     p90_ms")
    for record in records:
        print(
            f"{record['variant']:<24} {record['median_ms']:>9.3f} "
            f"{record['aggregate_tflops']:>9.3f} {record['gain_pct']:>10.2f} "
            f"{record['p10_ms']:>10.3f} {record['p90_ms']:>10.3f}"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
