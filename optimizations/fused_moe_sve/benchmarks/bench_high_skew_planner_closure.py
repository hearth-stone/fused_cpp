#!/usr/bin/env python3
"""Measure analytical planner ranking on one complete high-skew MoE trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
COST_MODEL_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"
PLANNER_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path[:0] = [str(REPO_ROOT / "src"), str(COST_MODEL_DIR), str(PLANNER_DIR)]

from analytic_model import AnalyticMoeCostModel  # noqa: E402
from cold_phase_cp_sat_oracle import ColdPhaseJob, build_cold_phase_jobs  # noqa: E402
from cold_phase_domain_oracle import (  # noqa: E402
    LlcDomain,
    compare_strict_greedy_union,
    materialize_domain_candidate,
    solve_fixed_strict_greedy_cp_sat,
    solve_cold_phase_domain_shortlist,
)
from fused_cpp import _moe_C  # noqa: E402
from fused_cpp.moe import (  # noqa: E402
    AsyncMoEPlanV2,
    fused_moe_bf16_tiled_async_plan,
    prepare_fused_moe_bf16_tiled_weights,
)
from interval_planner import IntervalPlanner  # noqa: E402
from planned_moe import PlannedMoE  # noqa: E402
from resource_lower_bound import build_lower_bound_certificate  # noqa: E402
from sve_fused_expert_lower_bound import (  # noqa: E402
    ServiceUpperBound,
    SveFusedExpertHardwareEnvelope,
    build_sve_fused_expert_lower_bound_problem,
)


ORDERS = ("lpt", "reverse_odd", "reverse_even")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-file", type=Path, required=True)
    parser.add_argument("--route-layer", type=int, required=True)
    parser.add_argument("--analytic-calibration", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--threads", type=int, default=80)
    parser.add_argument("--widths", default="4,8,16")
    parser.add_argument("--orders", default=",".join(ORDERS))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--weight-copies", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--cp-sat-shortlist", type=int, default=0)
    parser.add_argument("--cp-sat-measure", type=int, default=8)
    parser.add_argument("--cp-sat-root-time", type=float, default=60.0)
    parser.add_argument("--cp-sat-pool-root-time", type=float, default=10.0)
    parser.add_argument("--cp-sat-next-time", type=float, default=5.0)
    parser.add_argument("--cp-sat-gap", type=float, default=0.01)
    parser.add_argument("--cp-sat-slack", type=float, default=0.10)
    parser.add_argument("--cp-sat-workers", type=int, default=8)
    parser.add_argument("--cp-sat-repair-experts", type=int, default=16)
    parser.add_argument(
        "--cp-sat-incumbent",
        choices=("one_step", "greedy"),
        default="one_step",
    )
    parser.add_argument(
        "--cp-sat-domain-hint",
        choices=("one_step", "greedy"),
        default="one_step",
    )
    parser.add_argument("--cold-panel-rows", type=int, default=12)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_ints(value: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(int(item) for item in value.split(",") if item))
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated integer list") from error
    if not values or min(values) <= 0:
        raise ValueError(f"{name} must contain positive integers")
    return values


def _parse_orders(value: str) -> tuple[str, ...]:
    orders = tuple(dict.fromkeys(item for item in value.split(",") if item))
    unknown = set(orders) - set(ORDERS)
    if not orders or unknown:
        raise ValueError(f"orders must be selected from {ORDERS}, got {sorted(unknown)}")
    return orders


def load_route_layer(path: Path, layer: int, experts: int) -> tuple[torch.Tensor, dict[str, object]]:
    if not path.is_file():
        raise ValueError(f"route file does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"expert_ids", "layer_ids"}:
        raise ValueError("route file must contain exactly expert_ids and layer_ids tensors")
    expert_ids = payload["expert_ids"]
    layer_ids = payload["layer_ids"]
    if not isinstance(expert_ids, torch.Tensor) or expert_ids.dim() != 3:
        raise ValueError("expert_ids must be a [tokens,layers,top_k] tensor")
    if not isinstance(layer_ids, torch.Tensor) or layer_ids.dim() != 1:
        raise ValueError("layer_ids must be a one-dimensional tensor")
    if expert_ids.shape[1] != layer_ids.numel():
        raise ValueError("expert_ids layer dimension must match layer_ids")
    if not 0 <= layer < expert_ids.shape[1]:
        raise ValueError(f"route layer must be in [0,{expert_ids.shape[1] - 1}], got {layer}")
    selected = expert_ids[:, layer, :].to(torch.int32).contiguous()
    if selected.numel() == 0 or int(selected.min()) < 0 or int(selected.max()) >= experts:
        raise ValueError(f"route ids must be in [0,{experts - 1}]")
    duplicates = sum(len(set(row)) != len(row) for row in selected.tolist())
    if duplicates:
        raise ValueError(f"route trace contains {duplicates} token rows with duplicate experts")
    counts = torch.bincount(selected.flatten().to(torch.int64), minlength=experts)
    top8 = torch.topk(counts, min(8, experts)).values.sum()
    return selected, {
        "file": str(path),
        "sha256": _sha256(path),
        "layer_index": layer,
        "layer_id": int(layer_ids[layer]),
        "tokens": int(selected.shape[0]),
        "top_k": int(selected.shape[1]),
        "active_experts": int((counts > 0).sum()),
        "max_routes": int(counts.max()),
        "top8_share": float(top8 / counts.sum()),
        "duplicate_token_rows": duplicates,
    }


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def _stats(samples: list[float]) -> dict[str, object]:
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": _percentile(samples, 0.10),
        "p90_ms": _percentile(samples, 0.90),
        "samples_ms": samples,
    }


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    begin = 0
    while begin < len(order):
        end = begin + 1
        while end < len(order) and values[order[end]] == values[order[begin]]:
            end += 1
        rank = (begin + end - 1) / 2.0
        for index in order[begin:end]:
            ranks[index] = rank
        begin = end
    return ranks


def _spearman(first: list[float], second: list[float]) -> float:
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("Spearman correlation requires equal lists with at least two values")
    left = _ranks(first)
    right = _ranks(second)
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_norm = sum((x - left_mean) ** 2 for x in left) ** 0.5
    right_norm = sum((y - right_mean) ** 2 for y in right) ** 0.5
    return numerator / (left_norm * right_norm)


def _plan_summary(bridge: dict[str, object]) -> dict[str, object]:
    task_threads = [int(value) for value in bridge["task_threads"]]
    return {
        "execution_mode": bridge["execution_mode"],
        "tasks": len(task_threads),
        "width_histogram": dict(sorted(Counter(task_threads).items())),
        "early_merge": bridge.get("early_merge"),
        "w13_window_histogram": dict(
            sorted(Counter(int(value) for value in bridge["task_w13_window_tiles"]).items())
        ),
        "w2_window_histogram": dict(
            sorted(Counter(int(value) for value in bridge["task_w2_window_tiles"]).items())
        ),
    }


def _select_expected(candidates: list[dict[str, object]]) -> dict[str, object]:
    if not candidates:
        raise ValueError("expected-makespan selection requires at least one candidate")
    return min(
        candidates,
        key=lambda candidate: (
            float(candidate["makespan_ns"]),
            float(candidate["pessimistic_ns"]),
            int(candidate["active_working_set_bytes"]),
            int(candidate["resource_groups"]),
        ),
    )


def _strict_candidate(
    planned: PlannedMoE,
    counts: list[tuple[int, int]],
    topk_ids: torch.Tensor,
    *,
    width: int,
    order: str,
) -> tuple[dict[str, object], float]:
    if planned.num_cores % width:
        raise ValueError(f"width={width} must divide threads={planned.num_cores}")
    interval = planned.interval_planners[0]
    shape = (width,) * (planned.num_cores // width)
    lanes = interval._lanes(shape)
    lpt = interval._assign_lpt(counts, lanes)
    assignment = interval._assignment_for_order(lpt, order)
    tasks = interval._build_tasks(counts, lanes, assignment)
    prediction = float(interval._score(tasks))
    return interval.to_async_bridge(tasks, topk_ids=topk_ids), prediction


def _prepare_weights(args: argparse.Namespace) -> list[object]:
    generator = torch.Generator().manual_seed(args.seed)
    w13 = torch.empty((args.experts, 2 * args.intermediate, args.hidden), dtype=torch.bfloat16)
    w13.normal_(mean=0.0, std=0.01, generator=generator)
    w2 = torch.empty((args.experts, args.hidden, args.intermediate), dtype=torch.bfloat16)
    w2.normal_(mean=0.0, std=0.01, generator=generator)
    packed = [
        prepare_fused_moe_bf16_tiled_weights(
            w13,
            w2,
            fuse_silu=True,
            backend="arm_sve_bf16",
        )
        for _ in range(args.weight_copies)
    ]
    if any(item.gemm_backend != 1 for item in packed):
        raise RuntimeError("benchmark requires the Arm SVE BF16 backend")
    return packed


def _full_width_jobs(
    jobs: tuple[ColdPhaseJob, ...],
    tasks: list[tuple[int, int, int, int, list[int]]],
) -> tuple[ColdPhaseJob, ...]:
    width_by_expert = {int(expert): int(width) for expert, _, _, width, _ in tasks}
    if len(width_by_expert) != len(tasks):
        raise ValueError("CP-SAT incumbent projection requires one task per active expert")
    projected = []
    for job in jobs:
        width = width_by_expert[job.expert_id]
        modes = tuple(mode for mode in job.modes if mode.threads == width)
        if len(modes) != 1:
            raise ValueError(f"full-plan width={width} is unavailable for expert_id={job.expert_id}")
        projected.append(ColdPhaseJob(job.expert_id, job.routes, modes))
    return tuple(projected)


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    positive = (
        args.hidden,
        args.intermediate,
        args.experts,
        args.threads,
        args.runs,
        args.weight_copies,
    )
    if min(positive) <= 0 or args.warmup < 0:
        raise ValueError("dimensions, threads, runs, and weight copies must be positive")
    widths = _parse_ints(args.widths, name="widths")
    orders = _parse_orders(args.orders)
    if any(args.threads % width for width in widths):
        raise ValueError(f"every width must divide threads={args.threads}, got {widths}")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < args.threads:
        raise ValueError(f"threads={args.threads} exceeds affinity size {len(affinity)}")
    cpu_ids = affinity[: args.threads]
    torch.set_num_threads(1)

    topk_ids, route_metadata = load_route_layer(args.route_file, args.route_layer, args.experts)
    tokens, top_k = topk_ids.shape
    route_counts = torch.bincount(topk_ids.flatten().to(torch.int64), minlength=args.experts)
    counts = [(expert, int(routes)) for expert, routes in enumerate(route_counts.tolist()) if routes]
    model = AnalyticMoeCostModel(
        args.analytic_calibration,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        global_experts=args.experts,
        local_experts=args.experts,
        mode="tp",
        degree=4,
        concurrent_ranks=1,
        down_output_element_bytes=4,
    )
    if model.calibration.cores_per_rank != args.threads:
        raise ValueError(
            f"calibration cores_per_rank={model.calibration.cores_per_rank} does not match threads={args.threads}"
        )
    planned = PlannedMoE(
        model,
        num_cores=args.threads,
        cpu_ids=cpu_ids,
        search_mode="full",
        cache_plans=False,
    )
    interval = planned.interval_planners[0]

    calibration = model.calibration
    frontend = calibration.frontend_instructions
    resource_envelope = SveFusedExpertHardwareEnvelope(
        num_cores=args.threads,
        bfmmla=ServiceUpperBound(
            "bfmmla_flops",
            calibration.matrix_flops.saturated_rate,
            calibration.matrix_flops.single_thread_rate,
        ),
        key_instructions=ServiceUpperBound(
            "key_instructions",
            frontend.saturated_rate if frontend is not None else 1.0e30,
            frontend.single_thread_rate if frontend is not None else 1.0e30,
        ),
        l1_load_bytes=ServiceUpperBound(
            "l1_load_bytes",
            calibration.l1_bytes.saturated_rate,
            calibration.l1_bytes.single_thread_rate,
        ),
        epilogue_elements=ServiceUpperBound(
            "epilogue_elements",
            (
                calibration.epilogue_elements.saturated_rate
                if calibration.epilogue_elements is not None
                else 1.0e30
            ),
            (
                calibration.epilogue_elements.single_thread_rate
                if calibration.epilogue_elements is not None
                else 1.0e30
            ),
        ),
        dram_bytes=ServiceUpperBound("dram_bytes", calibration.dram_bytes.saturated_rate),
    )
    resource_problem = build_sve_fused_expert_lower_bound_problem(
        route_counts.tolist(),
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        widths=tuple(width for width in model.supported_widths if width <= args.threads),
        n_tile=calibration.backend_n_tile,
        envelope=resource_envelope,
        include_weight_dram=True,
    )
    resource_certificate = build_lower_bound_certificate(resource_problem)

    plan_begin = time.perf_counter_ns()
    strict_candidates = [interval._candidate(counts, shape) for shape in interval.shapes]
    analytic_baseline = interval._analytic_full_baseline(counts)
    if analytic_baseline is not None:
        strict_candidates.append(analytic_baseline)
    full_selected = _select_expected(strict_candidates)
    full_result = interval._finalize_plan(
        full_selected,
        strict_candidates,
        topk_ids=topk_ids,
        planner_backend="python_analytic_full",
        planner_workers=1,
        strict_candidates=len(strict_candidates),
        dynamic_candidates=0,
        tail_repartition_candidates=0,
    )
    full_plan_ms = (time.perf_counter_ns() - plan_begin) / 1.0e6
    conservative_selected = IntervalPlanner._select_analytic_full(strict_candidates)
    conservative_result = interval._finalize_plan(
        conservative_selected,
        strict_candidates,
        topk_ids=topk_ids,
        planner_backend="python_analytic_full_conservative_width",
        planner_workers=1,
        strict_candidates=len(strict_candidates),
        dynamic_candidates=0,
        tail_repartition_candidates=0,
    )
    bridges: dict[str, dict[str, object]] = {
        "full_selected": full_result["bridge"],
        "conservative_selected": conservative_result["bridge"],
    }
    predicted_ms = {
        "full_selected": float(full_result["makespan_ns"]) / 1.0e6,
        "conservative_selected": float(conservative_result["makespan_ns"]) / 1.0e6,
    }
    candidate_metadata: dict[str, dict[str, object]] = {
        "full_selected": {
            "shape": list(full_result["shape"]),
            "assignment_order": full_result["assignment_order"],
            "predicted_ms": predicted_ms["full_selected"],
            "plan": _plan_summary(full_result["bridge"]),
        },
        "conservative_selected": {
            "shape": list(conservative_result["shape"]),
            "assignment_order": conservative_result["assignment_order"],
            "predicted_ms": predicted_ms["conservative_selected"],
            "plan": _plan_summary(conservative_result["bridge"]),
        },
    }
    manual_begin = time.perf_counter_ns()
    for width in widths:
        for order in orders:
            name = f"w{width}_{order}"
            bridge, prediction = _strict_candidate(
                planned,
                counts,
                topk_ids,
                width=width,
                order=order,
            )
            bridges[name] = bridge
            predicted_ms[name] = prediction / 1.0e6
            candidate_metadata[name] = {
                "shape": [width] * (args.threads // width),
                "assignment_order": order,
                "predicted_ms": predicted_ms[name],
                "plan": _plan_summary(bridge),
            }
    manual_plan_ms = (time.perf_counter_ns() - manual_begin) / 1.0e6
    cp_sat_report = None
    cp_sat_plan_ms = 0.0
    if args.cp_sat_shortlist:
        if args.cp_sat_measure <= 0 or args.cp_sat_measure > args.cp_sat_shortlist:
            raise ValueError("cp-sat-measure must be in [1, cp-sat-shortlist]")
        if not model.calibration.llc_domains:
            raise ValueError("CP-SAT domain search requires LLC topology in the analytical calibration")
        cp_begin = time.perf_counter_ns()
        greedy_begin = time.perf_counter_ns()
        greedy_result = interval.plan_quick(counts, topk_ids=topk_ids)
        greedy_plan_ms = (time.perf_counter_ns() - greedy_begin) / 1.0e6
        if (
            greedy_result["execution_mode"] != "strict"
            or greedy_result["tail_pool_threads"] is not None
            or greedy_result["tail_pool_tasks"]
            or greedy_result["tail_repartition_tasks"]
        ):
            raise RuntimeError("pure greedy control must be a fixed strict plan without a tail pool")
        greedy_event_ns = float(interval._score(greedy_result["tasks"]))
        bridges["greedy_strict"] = greedy_result["bridge"]
        predicted_ms["greedy_strict"] = greedy_event_ns / 1.0e6
        candidate_metadata["greedy_strict"] = {
            "shape": list(greedy_result["shape"]),
            "assignment_order": greedy_result["assignment_order"],
            "quick_predicted_ms": float(greedy_result["makespan_ns"]) / 1.0e6,
            "event_model_ms": greedy_event_ns / 1.0e6,
            "planner_ms": greedy_plan_ms,
            "plan": _plan_summary(greedy_result["bridge"]),
        }
        domain_cursor = 0
        domains = []
        for calibrated_domain in model.calibration.llc_domains:
            core_count = len(calibrated_domain.cpu_ids)
            domains.append(LlcDomain(calibrated_domain.domain_id, domain_cursor, core_count))
            domain_cursor += core_count
        dram_bandwidth_gbps = model.calibration.dram_bytes.saturated_rate / 1.0e9
        requested_incumbent_tasks = (
            greedy_result["tasks"]
            if args.cp_sat_domain_hint == "greedy"
            else conservative_selected["tasks"]
        )
        full_widths = {int(task[3]) for task in requested_incumbent_tasks}
        greedy_widths = {int(task[3]) for task in greedy_result["tasks"]}
        largest_domain = max(domain.core_count for domain in domains)
        projection_tasks = requested_incumbent_tasks
        incumbent_source = f"{args.cp_sat_domain_hint}_strict_plan"
        if any(int(task[3]) > largest_domain for task in projection_tasks):
            projection_tasks = conservative_selected["tasks"]
            incumbent_source = f"{args.cp_sat_domain_hint}_exact_union_with_one_step_domain_hint"
        cp_sat_widths = tuple(
            sorted(
                set(widths) | full_widths | greedy_widths
            )
        )
        jobs = build_cold_phase_jobs(
            counts,
            cp_sat_widths,
            interval.model,
            num_cores=args.threads,
            cold_panel_rows=args.cold_panel_rows,
            phase_granularity="expert",
        )
        incumbent_projection = solve_cold_phase_domain_shortlist(
            _full_width_jobs(jobs, projection_tasks),
            num_cores=args.threads,
            domains=domains,
            dram_bandwidth_gbps=dram_bandwidth_gbps,
            solution_limit=1,
            max_time_s=args.cp_sat_root_time,
            subsequent_time_s=args.cp_sat_next_time,
            workers=args.cp_sat_workers,
            relative_gap_limit=args.cp_sat_gap,
        )
        if not incumbent_projection.candidates:
            raise RuntimeError(
                f"failed to project the current full plan into LLC domains: {incumbent_projection.status}"
            )
        proof = solve_cold_phase_domain_shortlist(
            jobs,
            num_cores=args.threads,
            domains=domains,
            dram_bandwidth_gbps=dram_bandwidth_gbps,
            initial_assignments=incumbent_projection.candidates[0].assignments,
            solution_limit=1,
            max_signature_changes=args.cp_sat_repair_experts,
            shortlist_objective_slack=args.cp_sat_slack,
            max_time_s=args.cp_sat_root_time,
            subsequent_time_s=args.cp_sat_next_time,
            workers=args.cp_sat_workers,
            relative_gap_limit=args.cp_sat_gap,
        )
        greedy_oracle = None
        strict_union = None
        if args.cp_sat_incumbent == "greedy":
            greedy_oracle = solve_fixed_strict_greedy_cp_sat(
                jobs,
                greedy_result["tasks"],
                num_cores=args.threads,
                dram_bandwidth_gbps=dram_bandwidth_gbps,
                max_time_s=args.cp_sat_root_time,
                workers=args.cp_sat_workers,
                relative_gap_limit=args.cp_sat_gap,
            )
            strict_union = compare_strict_greedy_union(greedy_oracle, proof)
        if args.cp_sat_shortlist == 1 and args.cp_sat_measure == 1:
            shortlist = proof
        else:
            shortlist = solve_cold_phase_domain_shortlist(
                jobs,
                num_cores=args.threads,
                domains=domains,
                dram_bandwidth_gbps=dram_bandwidth_gbps,
                initial_assignments=incumbent_projection.candidates[0].assignments,
                solution_limit=args.cp_sat_shortlist,
                max_signature_changes=args.cp_sat_repair_experts,
                shortlist_objective_slack=args.cp_sat_slack,
                max_time_s=args.cp_sat_pool_root_time,
                subsequent_time_s=args.cp_sat_next_time,
                workers=args.cp_sat_workers,
                relative_gap_limit=args.cp_sat_gap,
            )
        reranked = []
        lowering_ms = 0.0
        for candidate_index, candidate in enumerate(shortlist.candidates):
            lower_begin = time.perf_counter_ns()
            placement = materialize_domain_candidate(
                candidate,
                domains=domains,
                max_time_s=args.cp_sat_next_time,
                workers=args.cp_sat_workers,
                random_seed=args.seed + candidate_index,
            )
            lowering_ms += (time.perf_counter_ns() - lower_begin) / 1.0e6
            if not placement.tasks:
                continue
            tasks = [task.planner_tuple() for task in placement.tasks]
            event_makespan_ns = float(interval._score(tasks))
            reranked.append((event_makespan_ns, candidate_index, candidate, placement, tasks))
        reranked.sort(key=lambda row: (row[0], row[2].objective_ns, row[1]))
        if len(reranked) < args.cp_sat_measure:
            raise RuntimeError(
                f"only {len(reranked)} of {len(shortlist.candidates)} CP-SAT candidates lowered; "
                f"need {args.cp_sat_measure}"
            )
        for rank, (event_ns, candidate_index, candidate, placement, tasks) in enumerate(
            reranked[: args.cp_sat_measure]
        ):
            name = f"cp_sat_{rank:02d}"
            bridge = interval.to_async_bridge(tasks, topk_ids=topk_ids)
            bridges[name] = bridge
            predicted_ms[name] = event_ns / 1.0e6
            candidate_metadata[name] = {
                "shortlist_index": candidate_index,
                "surrogate_ms": candidate.objective_ns / 1.0e6,
                "event_model_ms": event_ns / 1.0e6,
                "surrogate_mode_histogram": candidate.mode_histogram(),
                "surrogate_domain_histogram": candidate.domain_histogram(),
                "lowering_status": placement.status,
                "lowering_ms": placement.wall_time_s * 1.0e3,
                "plan": _plan_summary(bridge),
            }
        cp_sat_plan_ms = (time.perf_counter_ns() - cp_begin) / 1.0e6
        cp_sat_report = {
            "incumbent_source": incumbent_source,
            "greedy_union_enabled": args.cp_sat_incumbent == "greedy",
            "searched_widths": list(cp_sat_widths),
            "repair_experts": args.cp_sat_repair_experts,
            "pure_greedy": {
                "planner_ms": greedy_plan_ms,
                "plan": _plan_summary(greedy_result["bridge"]),
                "event_model_ms": greedy_event_ns / 1.0e6,
                "oracle": (
                    greedy_oracle.to_dict(include_phases=False)
                    if greedy_oracle is not None
                    else None
                ),
            },
            "incumbent_projection": incumbent_projection.to_dict(include_phases=False),
            "proof": proof.to_dict(include_phases=False),
            "strict_union": strict_union.to_dict() if strict_union is not None else None,
            "shortlist": shortlist.to_dict(include_phases=False),
            "lowered_candidates": len(reranked),
            "measured_candidates": args.cp_sat_measure,
            "lowering_wall_ms": lowering_ms,
            "reranked": [
                {
                    "shortlist_index": candidate_index,
                    "surrogate_ms": candidate.objective_ns / 1.0e6,
                    "event_model_ms": event_ns / 1.0e6,
                    "lowering_status": placement.status,
                }
                for event_ns, candidate_index, candidate, placement, _ in reranked
            ],
        }
    plans = {name: AsyncMoEPlanV2.from_dict(bridge) for name, bridge in bridges.items()}

    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "0"
    os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE"] = "1"
    generator = torch.Generator().manual_seed(args.seed ^ 0x1234)
    hidden = torch.empty((tokens, args.hidden), dtype=torch.bfloat16)
    hidden.normal_(mean=0.0, std=0.01, generator=generator)
    topk_weights = torch.softmax(torch.randn((tokens, top_k), generator=generator), dim=-1)
    packed_copies = _prepare_weights(args)
    outputs = {name: torch.empty_like(hidden) for name in plans}

    def run(name: str, copy_index: int) -> torch.Tensor:
        return fused_moe_bf16_tiled_async_plan(
            hidden,
            packed_copies[copy_index],
            topk_weights,
            topk_ids,
            plans[name],
            global_num_experts=args.experts,
            out=outputs[name],
        )

    reference = run("full_selected", 0).clone()
    for name in plans:
        candidate = run(name, 0).clone()
        torch.testing.assert_close(candidate.float(), reference.float(), atol=0, rtol=0)

    names = list(plans)
    warmup_order = random.Random(args.seed ^ 0xA5A5)
    for round_index in range(args.warmup):
        warmup_order.shuffle(names)
        for position, name in enumerate(names):
            run(name, (round_index + position) % len(packed_copies))

    samples = {name: [] for name in plans}
    timed_order = random.Random(args.seed ^ 0x5A5A)
    sink = 0
    for round_index in range(args.runs):
        names = list(plans)
        timed_order.shuffle(names)
        for position, name in enumerate(names):
            begin = time.perf_counter_ns()
            output = run(name, (round_index + position) % len(packed_copies))
            samples[name].append((time.perf_counter_ns() - begin) / 1.0e6)
            sink ^= int(output.view(torch.int16)[0, 0])

    stats = {name: _stats(values) for name, values in samples.items()}
    measured_best = min(stats, key=lambda name: float(stats[name]["median_ms"]))
    manual_names = [name for name in plans if name != "full_selected"]
    rank_spearman = _spearman(
        [predicted_ms[name] for name in manual_names],
        [float(stats[name]["median_ms"]) for name in manual_names],
    )
    full_samples = samples["full_selected"]
    best_samples = samples[measured_best]
    paired_speedup = [100.0 * (full / best - 1.0) for full, best in zip(full_samples, best_samples)]
    cp_sat_measured = None
    cp_names = sorted(name for name in plans if name.startswith("cp_sat_"))
    if cp_names:
        cp_measured_best = min(cp_names, key=lambda name: float(stats[name]["median_ms"]))
        cp_selected = cp_names[0]
        cp_best_ms = float(stats[cp_measured_best]["median_ms"])
        cp_selected_ms = float(stats[cp_selected]["median_ms"])
        one_step_ms = float(stats["conservative_selected"]["median_ms"])
        greedy_ms = float(stats["greedy_strict"]["median_ms"])
        lower_bound_ms = resource_certificate.lower_bound_s * 1.0e3
        paired_vs_greedy = [
            100.0 * (cp / greedy - 1.0)
            for cp, greedy in zip(samples[cp_selected], samples["greedy_strict"], strict=True)
        ]
        cp_sat_measured = {
            "event_selected": cp_selected,
            "measured_best": cp_measured_best,
            "shortlist_regret_pct": 100.0 * (cp_selected_ms / cp_best_ms - 1.0),
            "vs_one_step_gate_pct": 100.0 * (cp_selected_ms / one_step_ms - 1.0),
            "vs_pure_greedy_pct": 100.0 * (cp_selected_ms / greedy_ms - 1.0),
            "paired_vs_pure_greedy_pct": {
                "median": statistics.median(paired_vs_greedy),
                "p10": _percentile(paired_vs_greedy, 0.10),
                "p90": _percentile(paired_vs_greedy, 0.90),
                "wins": sum(value < 0.0 for value in paired_vs_greedy),
                "runs": len(paired_vs_greedy),
            },
            "resource_lower_bound_ms": lower_bound_ms,
            "measured_to_resource_lb_ratio": cp_selected_ms / lower_bound_ms,
            "t_plan_plus_t_execute_ms": cp_sat_plan_ms + cp_selected_ms,
        }

    result = {
        "kind": "high_skew_planner_closure",
        "machine_affinity": cpu_ids,
        "route": route_metadata,
        "shape": {
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "experts": args.experts,
            "tokens": tokens,
            "top_k": top_k,
            "threads": args.threads,
        },
        "method": {
            "warmup": args.warmup,
            "runs": args.runs,
            "weight_copies": args.weight_copies,
            "widths": list(widths),
            "orders": list(orders),
            "route_dtype": "fp32",
            "full_dynamic_tail_pool": False,
            "full_bounded_tail_repartition": False,
        },
        "identity": {
            "calibration": str(args.analytic_calibration),
            "calibration_sha256": _sha256(args.analytic_calibration),
            "extension": str(_moe_C.__file__),
            "extension_sha256": _sha256(Path(_moe_C.__file__)),
        },
        "planning": {
            "full_cold_ms": full_plan_ms,
            "manual_candidates_ms": manual_plan_ms,
            "cp_sat_total_ms": cp_sat_plan_ms,
            "strict_candidates": len(strict_candidates),
            "dynamic_candidates": full_result["dynamic_candidates"],
            "conservative_rule": (
                "when the winner uses more than 8 threads, step its maximum team width down "
                "by one calibrated level inside the uncertainty overlap, then minimize expected makespan"
            ),
        },
        "cp_sat": cp_sat_report,
        "cp_sat_measured": cp_sat_measured,
        "resource_lower_bound": {
            "scope": "calibrated-service GEMM-only relaxation with compulsory cold weights",
            "is_hardware_peak_certificate": False,
            "certificate": resource_certificate.to_dict(),
        },
        "full_ranking_top20": full_result["ranking"][:20],
        "candidates": candidate_metadata,
        "stats": stats,
        "measured_best": measured_best,
        "manual_model_rank_spearman": rank_spearman,
        "measured_best_vs_full_selected_speedup_pct": {
            "median": statistics.median(paired_speedup),
            "p10": _percentile(paired_speedup, 0.10),
            "p90": _percentile(paired_speedup, 0.90),
        },
        "sink": sink,
    }
    print(
        "variant                predicted_ms  median_ms   p10_ms   p90_ms"
    )
    for name in sorted(plans, key=lambda item: float(stats[item]["median_ms"])):
        item = stats[name]
        print(
            f"{name:<22} {predicted_ms[name]:>12.3f} "
            f"{float(item['median_ms']):>10.3f} {float(item['p10_ms']):>8.3f} "
            f"{float(item['p90_ms']):>8.3f}"
        )
    print(
        f"measured_best={measured_best} full_selected={full_result['shape']}/"
        f"{full_result['assignment_order']} rank_spearman={rank_spearman:.3f}"
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
