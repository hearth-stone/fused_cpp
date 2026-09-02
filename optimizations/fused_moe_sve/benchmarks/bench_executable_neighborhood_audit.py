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
    critical_expert_scores,
    placed_tasks,
    sample_order_only_neighborhood,
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
    parser.add_argument("--event-top-per-strategy", type=int, default=4)
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


def _score_neighborhood(
    model: AnalyticMoeCostModel,
    state: ExecutablePlanState,
    *,
    expert_ids: list[int],
    per_operator: int,
    seed: int,
) -> tuple[dict[str, object], dict[str, ExecutablePlanState]]:
    sampled = sample_order_only_neighborhood(
        state,
        expert_filter=expert_ids,
        per_operator=per_operator,
        seed=seed,
    )
    baseline_ns = model.dag_makespan_placed(placed_tasks(state))
    scored = []
    states = {}
    begin = time.perf_counter_ns()
    for neighbor in sampled.neighbors:
        state_hash = neighbor.state.canonical_hash()
        score_ns = model.dag_makespan_placed(placed_tasks(neighbor.state))
        states[state_hash] = neighbor.state
        scored.append(
            {
                "state_hash": state_hash,
                "operator": neighbor.operator,
                "moved_experts": list(neighbor.moved_experts),
                "event_ns": score_ns,
                "event_gain_pct": 100.0 * (baseline_ns / score_ns - 1.0),
            }
        )
    score_wall_s = (time.perf_counter_ns() - begin) / 1.0e9
    scored.sort(key=lambda item: (float(item["event_ns"]), str(item["state_hash"])))
    operator_summary = {}
    for operator in ORDER_ONLY_OPERATORS:
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
        "operators": operator_summary,
        "scored": scored,
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
    full_result, full_plan_ms = _full_strict_result(planned, counts, topk_ids)
    llc_domains = tuple((domain.domain_id, domain.cpu_ids) for domain in model.calibration.llc_domains)
    baseline = ExecutablePlanState.from_planner_result(
        full_result,
        llc_domains=llc_domains,
    )
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
            expert_ids=expert_ids,
            per_operator=args.neighbors_per_operator,
            seed=args.seed ^ seed_delta,
        )
        reports[strategy] = report
        all_states.update(states)

    selected_hashes = []
    selected_sources: dict[str, list[str]] = {}
    for strategy, report in reports.items():
        for row in report["scored"][: args.event_top_per_strategy]:
            state_hash = str(row["state_hash"])
            selected_sources.setdefault(state_hash, []).append(strategy)
            if state_hash not in selected_hashes:
                selected_hashes.append(state_hash)

    hardware = None
    if args.measure_hardware:
        hardware = _hardware_measurement(args, topk_ids, all_states, selected_hashes)
        hardware_by_hash = {str(item["state_hash"]): item for item in hardware["candidates"]}
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
        },
        "method": {
            "search_state": "strict_fixed_whole_expert_v1",
            "criticality": "tail_weighted_event_duration_times_task_dilation",
            "critical_experts": budget,
            "neighbors_per_operator": args.neighbors_per_operator,
            "event_top_per_strategy": args.event_top_per_strategy,
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
        },
        "critical_scores": [
            {"expert_id": expert, "score": scores[expert]}
            for expert in sorted(scores, key=lambda expert: (-scores[expert], expert))
        ],
        "strategies": reports,
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
