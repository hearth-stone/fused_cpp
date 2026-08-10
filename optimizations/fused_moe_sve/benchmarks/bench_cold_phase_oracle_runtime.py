#!/usr/bin/env python3
"""Execute a cold-phase CP-SAT incumbent through strict Plan V2."""

from __future__ import annotations

import argparse
import heapq
import json
import os
import random
import signal
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
COST_MODEL_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"
PLANNER_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path[:0] = [str(REPO_ROOT / "src"), str(COST_MODEL_DIR), str(PLANNER_DIR)]

from cold_phase_cp_sat_oracle import (  # noqa: E402
    ColdPhaseAssignment,
    ColdPhaseRuntimePlacement,
    ColdPhaseRuntimeTask,
    PhaseAssignment,
    cold_phase_runtime_bridge,
    materialize_cold_phase_runtime_placement,
)
from fused_cpp.moe import (  # noqa: E402
    AsyncMoEPlanV2,
    fused_moe_bf16_tiled_async_plan,
    prepare_fused_moe_bf16_tiled_weights,
)
from interval_planner import IntervalPlanner  # noqa: E402
from phase_model import ContentionCostModel  # noqa: E402


def _parse_cpu_ids(value: str) -> tuple[int, ...]:
    cpus: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            raw_begin, raw_end = part.split("-", 1)
            begin, end = int(raw_begin), int(raw_end)
            if end < begin:
                raise argparse.ArgumentTypeError(f"invalid CPU range {part!r}")
            cpus.extend(range(begin, end + 1))
        else:
            cpus.append(int(part))
    if not cpus or min(cpus) < 0 or len(set(cpus)) != len(cpus):
        raise argparse.ArgumentTypeError("CPU ids must be unique non-negative integers")
    return tuple(cpus)


def _parse_scales(value: str) -> tuple[float, ...]:
    scales = tuple(float(item) for item in value.split(",") if item.strip())
    if not scales or any(scale < 0.0 for scale in scales):
        raise argparse.ArgumentTypeError("release scales must be non-negative")
    return scales


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item.strip())
    if len(values) != len(set(values)) or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be unique positive integers")
    return values


def _materialize_topk_ids(
    histogram: list[int],
    *,
    tokens: int,
    top_k: int,
    seed: int,
) -> torch.Tensor:
    if len(histogram) == 0 or any(count < 0 or count > tokens for count in histogram):
        raise ValueError("route histogram entries must be within [0, tokens]")
    if sum(histogram) != tokens * top_k:
        raise ValueError("route histogram must sum to tokens * top_k")
    remaining = [(-count, expert) for expert, count in enumerate(histogram) if count]
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


def _assignment_from_dict(
    row: dict[str, object],
    *,
    collapse_internal_waits: bool,
) -> ColdPhaseAssignment:
    phases = tuple(PhaseAssignment(**phase) for phase in row.get("phases", []))
    start_ns = int(row["start_ns"])
    service_ns = int(row["service_ns"])
    wait_ns = 0 if collapse_internal_waits else int(row["wait_ns"])
    end_ns = start_ns + service_ns if collapse_internal_waits else int(row["end_ns"])
    return ColdPhaseAssignment(
        expert_id=int(row["expert_id"]),
        routes=int(row["routes"]),
        threads=int(row["threads"]),
        start_ns=start_ns,
        service_ns=service_ns,
        wait_ns=wait_ns,
        end_ns=end_ns,
        phases=phases,
    )


def _placement_from_dict(payload: dict[str, object]) -> ColdPhaseRuntimePlacement:
    tasks = tuple(ColdPhaseRuntimeTask(**task) for task in payload["tasks"])
    return ColdPhaseRuntimePlacement(
        status=str(payload["status"]),
        wall_time_s=float(payload["wall_time_s"]),
        num_cores=int(payload["num_cores"]),
        tasks=tasks,
    )


def _add_small_task_slot_dependencies(
    placement: ColdPhaseRuntimePlacement,
    *,
    slots: int,
    max_routes: int,
) -> ColdPhaseRuntimePlacement:
    if slots <= 0 or max_routes <= 0:
        raise ValueError("small-task slots and route limit must be positive")
    tasks = list(placement.tasks)
    available = [(0, slot, -1) for slot in range(slots)]
    heapq.heapify(available)
    for task_id, task in enumerate(tasks):
        if task.threads != 1 or task.routes > max_routes:
            continue
        slot_ready_ns, slot, predecessor = heapq.heappop(available)
        dependencies = set(task.dependencies)
        if predecessor >= 0:
            dependencies.add(predecessor)
        tasks[task_id] = replace(task, dependencies=tuple(sorted(dependencies)))
        service_ns = task.modeled_end_ns - task.release_ns
        finish_ns = max(slot_ready_ns, task.release_ns) + service_ns
        heapq.heappush(available, (finish_ns, slot, task_id))
    return replace(placement, tasks=tuple(tasks))


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]


def _resolve_profile(report: dict[str, object], override: Path | None) -> Path:
    profile = override or Path(str(report["profile"]))
    if not profile.is_absolute():
        profile = REPO_ROOT / profile
    if not profile.exists():
        raise FileNotFoundError(profile)
    return profile


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-report", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--cpu-ids", type=_parse_cpu_ids, default=_parse_cpu_ids("0-95"))
    parser.add_argument("--release-scales", type=_parse_scales, default=_parse_scales("0,1"))
    parser.add_argument(
        "--small-task-slots",
        type=_parse_positive_ints,
        default=(),
        help="diagnostic slot caps for 1T tasks; comma-separated",
    )
    parser.add_argument("--small-task-slot-max-routes", type=int, default=32)
    parser.add_argument("--placement-input", type=Path)
    parser.add_argument("--placement-output", type=Path)
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument(
        "--collapse-internal-waits",
        action="store_true",
        help="diagnostic: retain task starts but replace internal phase waits with contiguous service",
    )
    parser.add_argument("--warmup", type=int, default=7)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument(
        "--measure-variant",
        action="append",
        default=[],
        help="restrict timed calls to this variant; repeatable",
    )
    parser.add_argument(
        "--stop-before-measure",
        action="store_true",
        help="send SIGSTOP after warmup so an external profiler can attach",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--trace-variant",
        action="append",
        default=[],
        help="variant to execute once with native tracing after timing; repeatable",
    )
    parser.add_argument("--trace-dir", type=Path, default=Path("/tmp/cold_phase_runtime_traces"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.warmup < 0 or args.runs <= 0:
        raise ValueError("warmup must be non-negative and runs must be positive")
    report = json.loads(args.oracle_report.read_text(encoding="utf-8"))
    num_cores = int(report["num_cores"])
    if len(args.cpu_ids) != num_cores:
        raise ValueError(f"expected {num_cores} CPU ids, got {len(args.cpu_ids)}")
    routes = [int(value) for value in report["routes"]]
    metadata = report.get("workload_metadata") or {}
    tokens = int(metadata.get("tokens", sum(routes)))
    top_k = int(metadata.get("top_k", 1))
    num_experts = int(metadata.get("num_experts", len(routes)))
    if len(routes) != num_experts:
        raise ValueError(f"report contains {len(routes)} route counts for E={num_experts}")

    profile = _resolve_profile(report, args.profile)
    model = ContentionCostModel(profile)
    policy = model.policy
    if policy is None:
        raise ValueError("runtime benchmark requires a schema-v2 profile")
    if policy.cores_per_rank != num_cores or policy.local_experts != num_experts:
        raise ValueError("oracle report and profile core/expert geometry differ")
    planner = IntervalPlanner(
        model,
        num_cores,
        cpu_ids=args.cpu_ids,
        native_cold_planner=False,
    )
    experts = [(expert, count) for expert, count in enumerate(routes) if count > 0]

    baseline_shape = tuple(int(width) for width in report["current_strict_plan"]["baseline_shape"])
    if sum(baseline_shape) != num_cores:
        raise ValueError(f"invalid baseline shape {baseline_shape}")
    _, baseline_tasks = planner.score_shape(experts, baseline_shape)
    baseline_bridge = planner.to_async_bridge(baseline_tasks)
    baseline_bridge["early_merge"] = False
    plans: dict[str, AsyncMoEPlanV2] = {
        "fixed_12x8t": AsyncMoEPlanV2.from_dict(baseline_bridge),
    }

    if args.placement_input is not None:
        placement = _placement_from_dict(json.loads(args.placement_input.read_text(encoding="utf-8")))
    else:
        assignments = tuple(
            _assignment_from_dict(
                row,
                collapse_internal_waits=args.collapse_internal_waits,
            )
            for row in report["mixed_oracle"]["assignments"]
        )
        placement = materialize_cold_phase_runtime_placement(
            assignments,
            num_cores=num_cores,
            max_time_s=30.0,
            workers=min(8, num_cores),
            random_seed=args.seed,
        )
    if not placement.tasks:
        raise RuntimeError(f"could not materialize mixed oracle placement: {placement.status}")
    if args.placement_output is not None:
        args.placement_output.parent.mkdir(parents=True, exist_ok=True)
        args.placement_output.write_text(json.dumps(placement.to_dict(), indent=2) + "\n", encoding="utf-8")
    if args.materialize_only:
        if args.placement_output is None:
            raise ValueError("--materialize-only requires --placement-output")
        print(
            f"placement status={placement.status} tasks={len(placement.tasks)} "
            f"solve_s={placement.wall_time_s:.3f}"
        )
        return 0

    for scale in args.release_scales:
        bridge = cold_phase_runtime_bridge(placement, planner, timed=scale != 0.0, early_merge=False)
        bridge["task_release_ns"] = [
            int(round(task.release_ns * scale)) for task in placement.tasks
        ]
        name = "mixed_eager" if scale == 0.0 else f"mixed_release_{scale:g}x"
        plans[name] = AsyncMoEPlanV2.from_dict(bridge)
    for slots in args.small_task_slots:
        slotted = _add_small_task_slot_dependencies(
            placement,
            slots=slots,
            max_routes=args.small_task_slot_max_routes,
        )
        bridge = cold_phase_runtime_bridge(slotted, planner, timed=False, early_merge=False)
        plans[f"mixed_small_slots_{slots}"] = AsyncMoEPlanV2.from_dict(bridge)

    topk_ids = _materialize_topk_ids(routes, tokens=tokens, top_k=top_k, seed=args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    hidden = torch.empty((tokens, policy.hidden_size), dtype=torch.bfloat16)
    hidden.normal_(mean=0.0, std=0.01, generator=generator)
    w13 = torch.empty(
        (num_experts, 2 * policy.intermediate_size, policy.hidden_size),
        dtype=torch.bfloat16,
    )
    w13.normal_(mean=0.0, std=0.01, generator=generator)
    w2 = torch.empty(
        (num_experts, policy.hidden_size, policy.intermediate_size),
        dtype=torch.bfloat16,
    )
    w2.normal_(mean=0.0, std=0.01, generator=generator)
    topk_weights = torch.softmax(torch.randn((tokens, top_k), generator=generator), dim=-1)

    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE"] = "0"
    os.environ["FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "1"
    os.environ["FUSED_CPP_MOE_SVE_ROUTE_MERGE_UNROLL"] = "1"
    os.environ["FUSED_CPP_MOE_ASYNC_SHORT_POOL_THREADS"] = "0"
    os.environ["FUSED_CPP_MOE_STRICT_TAIL_STEAL"] = "0"
    os.environ["FUSED_CPP_MOE_STAGE_TIMING"] = "0"
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="arm_sve_bf16")
    del w13, w2

    outputs = {name: torch.empty_like(hidden) for name in plans}

    def run(name: str) -> torch.Tensor:
        return fused_moe_bf16_tiled_async_plan(
            hidden,
            packed,
            topk_weights,
            topk_ids,
            plans[name],
            global_num_experts=num_experts,
            out=outputs[name],
        )

    reference = run("fixed_12x8t").clone()
    for name in plans:
        candidate = run(name).clone()
        torch.testing.assert_close(candidate.float(), reference.float(), atol=0, rtol=0)

    order_rng = random.Random(args.seed ^ 0xC01D)
    all_names = list(plans)
    for _ in range(args.warmup):
        order_rng.shuffle(all_names)
        for name in all_names:
            run(name)
    measure_names = list(args.measure_variant) or list(plans)
    if len(set(measure_names)) != len(measure_names):
        raise ValueError("measure variants must not contain duplicates")
    unknown_measure_names = sorted(set(measure_names) - plans.keys())
    if unknown_measure_names:
        raise ValueError(f"unknown measure variants {unknown_measure_names}; choices={list(plans)}")
    if args.stop_before_measure:
        os.kill(os.getpid(), signal.SIGSTOP)
    samples = {name: [] for name in measure_names}
    sink = 0
    for _ in range(args.runs):
        order_rng.shuffle(measure_names)
        for name in measure_names:
            begin = time.perf_counter_ns()
            result = run(name)
            samples[name].append((time.perf_counter_ns() - begin) / 1.0e6)
            sink ^= int(result.view(torch.int16)[0, 0])

    trace_files: dict[str, str] = {}
    if args.trace_variant:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        for name in args.trace_variant:
            if name not in plans:
                raise ValueError(f"unknown trace variant {name!r}; choices={list(plans)}")
            trace_file = args.trace_dir / f"{name}.trace"
            trace_file.unlink(missing_ok=True)
            os.environ["FUSED_CPP_MOE_TRACE_FILE"] = str(trace_file)
            os.environ["FUSED_CPP_MOE_TRACE"] = "1"
            run(name)
            os.environ["FUSED_CPP_MOE_TRACE"] = "0"
            trace_files[name] = str(trace_file)

    total_flops = 6 * tokens * top_k * policy.hidden_size * policy.intermediate_size
    gain_reference = "fixed_12x8t" if "fixed_12x8t" in samples else measure_names[0]
    baseline_ms = statistics.median(samples[gain_reference])
    records = []
    for name in measure_names:
        median_ms = statistics.median(samples[name])
        records.append(
            {
                "variant": name,
                "median_ms": median_ms,
                "p10_ms": _percentile(samples[name], 0.10),
                "p90_ms": _percentile(samples[name], 0.90),
                "aggregate_tflops": total_flops / median_ms / 1.0e9,
                "gain_pct": 100.0 * (baseline_ms / median_ms - 1.0),
                "samples_ms": samples[name],
            }
        )
    result = {
        "workload": report["workload"],
        "profile": str(profile),
        "shape": {
            "tokens": tokens,
            "top_k": top_k,
            "experts": num_experts,
            "active_experts": len(experts),
            "hidden": policy.hidden_size,
            "intermediate": policy.intermediate_size,
            "cores": num_cores,
        },
        "oracle": {
            "fixed_objective_ns": report["fixed_baseline_oracle"]["objective_ns"],
            "mixed_objective_ns": report["mixed_oracle"]["objective_ns"],
            "comparison": report["comparison"],
            "placement_status": placement.status,
            "placement_wall_time_s": placement.wall_time_s,
            "mixed_width_histogram": report["mixed_oracle"]["mode_histogram"],
        },
        "release_scales": list(args.release_scales),
        "gain_reference": gain_reference,
        "trace_files": trace_files,
        "records": records,
        "sink": sink,
    }
    print("variant                 median_ms   TFLOP/s   gain_pct     p10_ms     p90_ms")
    for record in records:
        print(
            f"{record['variant']:<23} {record['median_ms']:>9.3f} "
            f"{record['aggregate_tflops']:>9.3f} {record['gain_pct']:>10.2f} "
            f"{record['p10_ms']:>10.3f} {record['p90_ms']:>10.3f}"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
