#!/usr/bin/env python3
"""Audit local executable MoE neighborhoods against event scores and hardware."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import Counter
from math import ceil
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
COST_MODEL_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"
PLANNER_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path[:0] = [
    str(REPO_ROOT),
    str(REPO_ROOT / "src"),
    str(COST_MODEL_DIR),
    str(PLANNER_DIR),
]

from analytic_model import AnalyticMoeCostModel  # noqa: E402
from executable_plan_neighborhood import (  # noqa: E402
    ORDER_ONLY_OPERATORS,
    WIDTH_ONLY_OPERATORS,
    critical_expert_scores,
    is_resolvable_improvement,
    placed_tasks,
    sample_combined_neighborhood,
    sample_order_only_neighborhood,
    sample_width_only_neighborhood,
    score_executable_plan,
)
from executable_plan_state import ExecutablePlanState  # noqa: E402
from fused_cpp import _moe_C  # noqa: E402
from fused_cpp.moe import (  # noqa: E402
    AsyncMoEPlanV2,
    fused_moe_bf16_tiled_async_plan,
)
from planned_moe import PlannedMoE  # noqa: E402

from optimizations.fused_moe_sve.benchmarks.bench_high_skew_planner_closure import (  # noqa: E402
    _prepare_weights,
    _select_expected,
    _sha256,
    _spearman,
    _stats,
    load_route_layer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-file", type=Path, required=True)
    parser.add_argument("--route-layer", type=int, required=True)
    parser.add_argument("--analytic-calibration", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--threads", type=int, default=80)
    parser.add_argument("--critical-experts", type=int, default=32)
    parser.add_argument("--neighbors-per-operator", type=int, default=64)
    parser.add_argument(
        "--neighborhood-mode",
        choices=("order", "width", "combined"),
        default="order",
    )
    parser.add_argument("--event-top-per-strategy", type=int, default=4)
    parser.add_argument("--minimum-actionable-gain-pct", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--weight-copies", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--measure-hardware", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _full_strict_result(
    planned: PlannedMoE,
    counts: list[tuple[int, int]],
    topk_ids: torch.Tensor,
) -> tuple[dict[str, object], float]:
    interval = planned.interval_planners[0]
    begin = time.perf_counter_ns()
    candidates = [interval._candidate(counts, shape) for shape in interval.shapes]
    analytic_baseline = interval._analytic_full_baseline(counts)
    if analytic_baseline is not None:
        candidates.append(analytic_baseline)
    selected = _select_expected(candidates)
    result = interval._finalize_plan(
        selected,
        candidates,
        topk_ids=topk_ids,
        planner_backend="python_analytic_full",
        planner_workers=1,
        strict_candidates=len(candidates),
        dynamic_candidates=0,
        tail_repartition_candidates=0,
    )
    return result, (time.perf_counter_ns() - begin) / 1.0e6


def _window_geometry(
    state: ExecutablePlanState,
    *,
    hidden: int,
    intermediate: int,
    backend_n_tile: int,
) -> list[dict[str, int]]:
    w13_tiles = 2 * intermediate // backend_n_tile
    w2_tiles = hidden // backend_n_tile
    counts = Counter(
        (lane.threads, task.w13_window_tiles, task.w2_window_tiles)
        for lane in state.lanes
        for task in lane.tasks
    )
    return [
        {
            "tasks": task_count,
            "threads": threads,
            "w13_window_tiles": w13_window,
            "w2_window_tiles": w2_window,
            "w13_ranges": 1 if w13_window == 0 else ceil(w13_tiles / (threads * w13_window)),
            "w2_ranges": 1 if w2_window == 0 else ceil(w2_tiles / (threads * w2_window)),
        }
        for (threads, w13_window, w2_window), task_count in sorted(counts.items())
    ]


def _score_neighborhood(
    model: AnalyticMoeCostModel,
    state: ExecutablePlanState,
    *,
    interval,
    neighborhood_mode: str,
    expert_ids: list[int],
    per_operator: int,
    seed: int,
) -> tuple[dict[str, object], dict[str, ExecutablePlanState]]:
    policy = interval._stage_window_policy()

    def window_selector(routes: int, threads: int) -> tuple[int, int]:
        return (0, 0) if policy is None else policy.select(routes, threads)

    sample_kwargs = {
        "expert_filter": expert_ids,
        "per_operator": per_operator,
        "seed": seed,
    }
    if neighborhood_mode == "order":
        sampled = sample_order_only_neighborhood(state, **sample_kwargs)
        operators = ORDER_ONLY_OPERATORS
    else:
        width_kwargs = {
            "allowed_widths": interval.widths,
            "isolated_cost": model.T_iso,
            "window_selector": window_selector,
            **sample_kwargs,
        }
        if neighborhood_mode == "width":
            sampled = sample_width_only_neighborhood(state, **width_kwargs)
            operators = WIDTH_ONLY_OPERATORS
        else:
            sampled = sample_combined_neighborhood(state, **width_kwargs)
            operators = (*ORDER_ONLY_OPERATORS, *WIDTH_ONLY_OPERATORS)
    baseline_score = score_executable_plan(model, state)
    scored = []
    states = {}
    begin = time.perf_counter_ns()
    for neighbor in sampled.neighbors:
        state_hash = neighbor.state.canonical_hash()
        score = score_executable_plan(model, neighbor.state)
        states[state_hash] = neighbor.state
        scored.append(
            {
                "state_hash": state_hash,
                "operator": neighbor.operator,
                "moved_experts": list(neighbor.moved_experts),
                "event_ns": score.event_ns,
                "lane_guard_ns": score.lane_guard_ns,
                "robust_ns": score.robust_ns,
                "event_gain_pct": 100.0 * (baseline_score.event_ns / score.event_ns - 1.0),
                "robust_gain_pct": 100.0 * (baseline_score.robust_ns / score.robust_ns - 1.0),
            }
        )
    score_wall_s = (time.perf_counter_ns() - begin) / 1.0e9
    scored.sort(key=lambda item: (float(item["event_ns"]), str(item["state_hash"])))
    operator_summary = {}
    for operator in operators:
        rows = [item for item in scored if item["operator"] == operator]
        operator_summary[operator] = {
            "proposed": sampled.proposed_by_operator[operator],
            "valid": sampled.proposed_by_operator[operator],
            "invalid": 0,
            "duplicates": sampled.duplicate_by_operator[operator],
            "unique": sampled.unique_by_operator[operator],
            "sampled": sampled.sampled_by_operator[operator],
            "event_improving": sum(float(item["event_gain_pct"]) > 0.0 for item in rows),
            "best_event_gain_pct": max(
                (float(item["event_gain_pct"]) for item in rows),
                default=None,
            ),
        }
    report = {
        "selected_expert_ids": expert_ids,
        "proposed": sampled.proposed,
        "valid": sampled.proposed,
        "invalid": 0,
        "duplicates": sampled.duplicates,
        "unique": sampled.unique,
        "sampled": len(scored),
        "evaluation_wall_s": score_wall_s,
        "evaluations_per_second": len(scored) / score_wall_s if score_wall_s else None,
        "event_improving": sum(float(item["event_gain_pct"]) > 0.0 for item in scored),
        "event_improving_fraction": (
            sum(float(item["event_gain_pct"]) > 0.0 for item in scored) / len(scored) if scored else 0.0
        ),
        "best_event_gain_pct": max(
            (float(item["event_gain_pct"]) for item in scored),
            default=None,
        ),
        "best_robust_gain_pct": max(
            (float(item["robust_gain_pct"]) for item in scored),
            default=None,
        ),
        "operators": operator_summary,
        "scored": scored,
    }
    ablation_operators = {
        "order_only": frozenset(ORDER_ONLY_OPERATORS),
        "width_only": frozenset(WIDTH_ONLY_OPERATORS),
    }
    if neighborhood_mode == "combined":
        ablation_operators["combined"] = frozenset(operators)
    report["ablations"] = {
        name: {
            "sampled": len(rows := [item for item in scored if item["operator"] in selected_operators]),
            "event_improving": sum(float(item["event_gain_pct"]) > 0.0 for item in rows),
            "best_event_gain_pct": max((float(item["event_gain_pct"]) for item in rows), default=None),
            "best_robust_gain_pct": max((float(item["robust_gain_pct"]) for item in rows), default=None),
            "ordered_state_hashes": [str(item["state_hash"]) for item in rows],
        }
        for name, selected_operators in ablation_operators.items()
        if any(item["operator"] in selected_operators for item in scored)
    }
    return report, states


def _hardware_measurement(
    args: argparse.Namespace,
    topk_ids: torch.Tensor,
    states: dict[str, ExecutablePlanState],
    selected_hashes: list[str],
) -> dict[str, object]:
    if args.runs <= 0 or args.weight_copies <= 0 or args.warmup < 0:
        raise ValueError("hardware warmup/runs/weight-copies must be non-negative/positive")
    tokens, top_k = topk_ids.shape
    plans = {"baseline": AsyncMoEPlanV2.from_dict(states["baseline"].to_bridge())}
    for index, state_hash in enumerate(selected_hashes):
        plans[f"neighbor_{index:02d}"] = AsyncMoEPlanV2.from_dict(states[state_hash].to_bridge())

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

    reference = run("baseline", 0).clone()
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
    baseline_samples = samples["baseline"]
    candidates = []
    for index, state_hash in enumerate(selected_hashes):
        name = f"neighbor_{index:02d}"
        paired = [
            100.0 * (baseline / candidate - 1.0)
            for baseline, candidate in zip(baseline_samples, samples[name], strict=True)
        ]
        candidates.append(
            {
                "name": name,
                "state_hash": state_hash,
                "stats": stats[name],
                "paired_speedup_pct": {
                    "median": statistics.median(paired),
                    "p10": sorted(paired)[round((len(paired) - 1) * 0.10)],
                    "p90": sorted(paired)[round((len(paired) - 1) * 0.90)],
                    "wins": sum(value > 0.0 for value in paired),
                    "runs": len(paired),
                },
                "stable_improvement": (
                    statistics.median(paired) > 0.0 and sorted(paired)[round((len(paired) - 1) * 0.10)] > 0.0
                ),
            }
        )
    return {
        "baseline": stats["baseline"],
        "candidates": candidates,
        "sink": sink,
    }


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if (
        min(
            args.hidden,
            args.intermediate,
            args.experts,
            args.threads,
            args.critical_experts,
            args.neighbors_per_operator,
            args.event_top_per_strategy,
        )
        <= 0
    ):
        raise ValueError("dimensions and audit budgets must be positive")
    if not 0.0 <= args.minimum_actionable_gain_pct < 100.0:
        raise ValueError("minimum-actionable-gain-pct must be in [0, 100)")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < args.threads:
        raise ValueError(f"threads={args.threads} exceeds affinity size {len(affinity)}")
    cpu_ids = affinity[: args.threads]
    torch.set_num_threads(1)

    topk_ids, route_metadata = load_route_layer(args.route_file, args.route_layer, args.experts)
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
        raise ValueError("calibration cores_per_rank must match --threads")
    planned = PlannedMoE(
        model,
        num_cores=args.threads,
        cpu_ids=cpu_ids,
        search_mode="full",
        cache_plans=False,
    )
    interval = planned.interval_planners[0]
    full_result, full_plan_ms = _full_strict_result(planned, counts, topk_ids)
    llc_domains = tuple((domain.domain_id, domain.cpu_ids) for domain in model.calibration.llc_domains)
    baseline = ExecutablePlanState.from_planner_result(
        full_result,
        llc_domains=llc_domains,
    )
    backend_n_tile = int(model.policy.backend_n_tile)
    explanation = model.explain_dag_placed(placed_tasks(baseline))
    scores = critical_expert_scores(baseline, explanation)
    budget = min(args.critical_experts, len(scores))
    critical_ids = sorted(scores, key=lambda expert: (-scores[expert], expert))[:budget]
    random_ids = sorted(random.Random(args.seed).sample(list(scores), budget))

    reports = {}
    all_states = {"baseline": baseline}
    for strategy, expert_ids, seed_delta in (
        ("critical", critical_ids, 0xC17),
        ("random", random_ids, 0xA11),
    ):
        report, states = _score_neighborhood(
            model,
            baseline,
            interval=interval,
            neighborhood_mode=args.neighborhood_mode,
            expert_ids=expert_ids,
            per_operator=args.neighbors_per_operator,
            seed=args.seed ^ seed_delta,
        )
        reports[strategy] = report
        all_states.update(states)

    baseline_score = score_executable_plan(model, baseline)
    decision_rows = [(strategy, row) for strategy, report in reports.items() for row in report["scored"]]
    if not decision_rows:
        raise RuntimeError(f"no legal {args.neighborhood_mode} neighbors were generated")
    best_strategy, best_row = min(
        decision_rows,
        key=lambda item: (
            float(item[1]["robust_ns"]),
            str(item[1]["state_hash"]),
        ),
    )
    best_state_hash = str(best_row["state_hash"])
    best_score = score_executable_plan(model, all_states[best_state_hash])
    accepted = is_resolvable_improvement(
        baseline_score,
        best_score,
        minimum_gain_fraction=args.minimum_actionable_gain_pct / 100.0,
    )
    decision = {
        "minimum_actionable_gain_pct": args.minimum_actionable_gain_pct,
        "selected": best_state_hash if accepted else "baseline",
        "candidate_strategy": best_strategy,
        "candidate_state_hash": best_state_hash,
        "candidate_event_gain_pct": float(best_row["event_gain_pct"]),
        "candidate_robust_gain_pct": float(best_row["robust_gain_pct"]),
        "resolvable": accepted,
    }

    ablation_decisions = {}
    ablation_names = sorted(
        {name for report in reports.values() for name in report["ablations"]}
    )
    for ablation_name in ablation_names:
        allowed_hashes = {
            state_hash
            for report in reports.values()
            for state_hash in report["ablations"].get(ablation_name, {}).get("ordered_state_hashes", [])
        }
        ablation_rows = [
            (strategy, row)
            for strategy, report in reports.items()
            for row in report["scored"]
            if row["state_hash"] in allowed_hashes
        ]
        if not ablation_rows:
            continue
        ablation_strategy, ablation_row = min(
            ablation_rows,
            key=lambda item: (float(item[1]["robust_ns"]), str(item[1]["state_hash"])),
        )
        ablation_hash = str(ablation_row["state_hash"])
        ablation_score = score_executable_plan(model, all_states[ablation_hash])
        ablation_accepted = is_resolvable_improvement(
            baseline_score,
            ablation_score,
            minimum_gain_fraction=args.minimum_actionable_gain_pct / 100.0,
        )
        ablation_decisions[ablation_name] = {
            "selected": ablation_hash if ablation_accepted else "baseline",
            "candidate_strategy": ablation_strategy,
            "candidate_state_hash": ablation_hash,
            "candidate_event_gain_pct": float(ablation_row["event_gain_pct"]),
            "candidate_robust_gain_pct": float(ablation_row["robust_gain_pct"]),
            "resolvable": ablation_accepted,
        }

    selected_hashes = []
    selected_sources: dict[str, list[str]] = {}
    for strategy, report in reports.items():
        for ablation_name, ablation in report["ablations"].items():
            for state_hash in ablation["ordered_state_hashes"][: args.event_top_per_strategy]:
                selected_sources.setdefault(state_hash, []).append(f"{strategy}:{ablation_name}")
                if state_hash not in selected_hashes:
                    selected_hashes.append(state_hash)
    if accepted and best_state_hash not in selected_hashes:
        selected_hashes.append(best_state_hash)
    for ablation_name, ablation_decision in ablation_decisions.items():
        candidate_hash = str(ablation_decision["candidate_state_hash"])
        selected_sources.setdefault(candidate_hash, []).append(f"decision:{ablation_name}")
        if candidate_hash not in selected_hashes:
            selected_hashes.append(candidate_hash)

    hardware = None
    if args.measure_hardware:
        hardware = _hardware_measurement(args, topk_ids, all_states, selected_hashes)
        hardware_by_hash = {str(item["state_hash"]): item for item in hardware["candidates"]}
        scored_by_hash = {
            str(row["state_hash"]): row
            for report in reports.values()
            for row in report["scored"]
        }
        for state_hash, item in hardware_by_hash.items():
            scored_row = scored_by_hash[state_hash]
            item["operator"] = scored_row["operator"]
            item["moved_experts"] = scored_row["moved_experts"]
            item["event_gain_pct"] = scored_row["event_gain_pct"]
            item["robust_gain_pct"] = scored_row["robust_gain_pct"]
            item["sources"] = selected_sources[state_hash]
            item["shape"] = list(all_states[state_hash].shape)
            item["window_geometry"] = _window_geometry(
                all_states[state_hash],
                hidden=args.hidden,
                intermediate=args.intermediate,
                backend_n_tile=backend_n_tile,
            )
        event_values = []
        measured_values = []
        for state_hash in selected_hashes:
            event_ns = model.dag_makespan_placed(placed_tasks(all_states[state_hash]))
            event_values.append(event_ns)
            measured_values.append(float(hardware_by_hash[state_hash]["stats"]["median_ms"]))
        hardware["event_hardware_spearman"] = (
            _spearman(event_values, measured_values) if len(event_values) >= 2 else None
        )
        hardware["stable_improving_candidates"] = sum(
            bool(item["stable_improvement"]) for item in hardware["candidates"]
        )
        baseline_median_ms = float(hardware["baseline"]["median_ms"])
        selected_median_ms = (
            float(hardware_by_hash[best_state_hash]["stats"]["median_ms"]) if accepted else baseline_median_ms
        )
        shortlist_best_ms = min(
            baseline_median_ms,
            *(float(item["stats"]["median_ms"]) for item in hardware["candidates"]),
        )
        decision["measured_selected_ms"] = selected_median_ms
        decision["measured_shortlist_best_ms"] = shortlist_best_ms
        decision["measured_shortlist_regret_pct"] = 100.0 * (selected_median_ms / shortlist_best_ms - 1.0)
        for ablation_name, ablation_decision in ablation_decisions.items():
            measured_hashes = {
                state_hash
                for report in reports.values()
                for state_hash in report["ablations"].get(ablation_name, {}).get("ordered_state_hashes", [])
                if state_hash in hardware_by_hash
            }
            candidate_hash = str(ablation_decision["candidate_state_hash"])
            selected_ablation_ms = (
                float(hardware_by_hash[candidate_hash]["stats"]["median_ms"])
                if ablation_decision["resolvable"]
                else baseline_median_ms
            )
            ablation_best_ms = min(
                baseline_median_ms,
                *(float(hardware_by_hash[state_hash]["stats"]["median_ms"]) for state_hash in measured_hashes),
            )
            ablation_decision["measured_selected_ms"] = selected_ablation_ms
            ablation_decision["measured_shortlist_best_ms"] = ablation_best_ms
            ablation_decision["measured_shortlist_regret_pct"] = 100.0 * (
                selected_ablation_ms / ablation_best_ms - 1.0
            )

    result = {
        "kind": "executable_neighborhood_audit",
        "route": route_metadata,
        "shape": {
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "experts": args.experts,
            "tokens": int(topk_ids.shape[0]),
            "top_k": int(topk_ids.shape[1]),
            "threads": args.threads,
            "backend_n_tile": backend_n_tile,
            "w13_stage_bytes": 4 * args.hidden * args.intermediate,
            "w2_stage_bytes": 2 * args.hidden * args.intermediate,
        },
        "method": {
            "search_state": "strict_fixed_whole_expert_v1",
            "neighborhood_mode": args.neighborhood_mode,
            "width_policy": "domain_local_split_merge_adjacent_migration_with_deterministic_windows",
            "criticality": "tail_weighted_event_duration_times_task_dilation",
            "critical_experts": budget,
            "neighbors_per_operator": args.neighbors_per_operator,
            "event_top_per_strategy": args.event_top_per_strategy,
            "minimum_actionable_gain_pct": args.minimum_actionable_gain_pct,
            "hardware_warmup": args.warmup if args.measure_hardware else 0,
            "hardware_runs": args.runs if args.measure_hardware else 0,
            "weight_copies": args.weight_copies if args.measure_hardware else 0,
        },
        "identity": {
            "calibration": str(args.analytic_calibration),
            "calibration_sha256": _sha256(args.analytic_calibration),
            "extension": str(_moe_C.__file__),
            "extension_sha256": _sha256(Path(_moe_C.__file__)),
        },
        "baseline": {
            "state_hash": baseline.canonical_hash(),
            "shape": list(baseline.shape),
            "event_ms": float(explanation["makespan_ns"]) / 1.0e6,
            "planning_ms": full_plan_ms,
            "window_geometry": _window_geometry(
                baseline,
                hidden=args.hidden,
                intermediate=args.intermediate,
                backend_n_tile=backend_n_tile,
            ),
        },
        "critical_scores": [
            {"expert_id": expert, "score": scores[expert]}
            for expert in sorted(scores, key=lambda expert: (-scores[expert], expert))
        ],
        "strategies": reports,
        "uncertainty_aware_decision": decision,
        "ablation_decisions": ablation_decisions,
        "hardware_shortlist": {
            "state_hashes": selected_hashes,
            "sources": selected_sources,
            "measurement": hardware,
        },
    }
    for strategy in reports.values():
        strategy["scored"] = strategy["scored"][: max(args.event_top_per_strategy, 16)]

    print(
        f"baseline event={result['baseline']['event_ms']:.3f} ms "
        f"shape={result['baseline']['shape']} planning={full_plan_ms:.1f} ms"
    )
    for strategy, report in reports.items():
        print(
            f"{strategy:<8} proposed={report['proposed']} unique={report['unique']} "
            f"sampled={report['sampled']} improving={report['event_improving']} "
            f"best_gain={report['best_event_gain_pct']:.3f}% "
            f"rate={report['evaluations_per_second']:.1f}/s"
        )
    if hardware is not None:
        print(
            f"hardware candidates={len(hardware['candidates'])} "
            f"stable_improving={hardware['stable_improving_candidates']} "
            f"spearman={hardware['event_hardware_spearman']}"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
