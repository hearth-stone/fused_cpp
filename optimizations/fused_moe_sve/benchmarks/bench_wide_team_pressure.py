#!/usr/bin/env python3
"""Measure residual concurrent-team-width pressure on one strict MoE layer."""

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
sys.path[:0] = [str(REPO_ROOT / "src"), str(COST_MODEL_DIR), str(PLANNER_DIR)]

from analytic_model import AnalyticMoeCostModel  # noqa: E402
from bench_high_skew_planner_closure import (  # noqa: E402
    ORDERS,
    _parse_ints,
    _parse_orders,
    _plan_summary,
    _prepare_weights,
    _sha256,
    _stats,
    _strict_candidate,
    load_route_layer,
)
from fused_cpp import _moe_C  # noqa: E402
from fused_cpp.moe import AsyncMoEPlanV2, fused_moe_bf16_tiled_async_plan  # noqa: E402
from planned_moe import PlannedMoE  # noqa: E402


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
    parser.add_argument("--reference-width", type=int, default=8)
    parser.add_argument("--early-merge", choices=("off", "on"), default="off")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=9)
    parser.add_argument("--weight-copies", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    widths = _parse_ints(args.widths, name="widths")
    orders = _parse_orders(args.orders)
    if args.reference_width not in widths:
        raise ValueError("reference-width must be present in widths")
    if min(args.hidden, args.intermediate, args.experts, args.threads, args.runs, args.weight_copies) <= 0:
        raise ValueError("dimensions, threads, runs, and weight copies must be positive")
    if args.warmup < 0 or any(args.threads % width for width in widths):
        raise ValueError("warmup must be non-negative and every width must divide threads")
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
    planned = PlannedMoE(model, num_cores=args.threads, cpu_ids=cpu_ids, search_mode="full", cache_plans=False)

    bridges = {}
    predicted_ms = {}
    metadata = {}
    plan_begin = time.perf_counter_ns()
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
            bridge["early_merge"] = args.early_merge == "on"
            bridges[name] = bridge
            predicted_ms[name] = prediction / 1.0e6
            metadata[name] = {
                "width": width,
                "order": order,
                "predicted_ms": prediction / 1.0e6,
                "plan": _plan_summary(bridge),
            }
    planning_ms = (time.perf_counter_ns() - plan_begin) / 1.0e6
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

    reference_name = f"w{args.reference_width}_{orders[0]}"
    reference = run(reference_name, 0).clone()
    for name in plans:
        torch.testing.assert_close(run(name, 0).float(), reference.float(), atol=0, rtol=0)

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
    width_residuals = {}
    for width in widths:
        ratios = [
            float(stats[f"w{width}_{order}"]["median_ms"])
            / predicted_ms[f"w{width}_{order}"]
            for order in orders
        ]
        width_residuals[width] = statistics.median(ratios)
    reference_residual = width_residuals[args.reference_width]
    relative_scales = {
        width: residual / reference_residual for width, residual in width_residuals.items()
    }
    result = {
        "kind": "wide_team_pressure_control",
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
            "reference_width": args.reference_width,
            "dynamic_tail_pool": False,
            "early_merge": args.early_merge == "on",
        },
        "identity": {
            "calibration": str(args.analytic_calibration),
            "calibration_sha256": _sha256(args.analytic_calibration),
            "extension": str(_moe_C.__file__),
            "extension_sha256": _sha256(Path(_moe_C.__file__)),
        },
        "planning_ms": planning_ms,
        "candidates": metadata,
        "stats": stats,
        "width_residuals": {str(width): value for width, value in width_residuals.items()},
        "relative_width_scales": {str(width): value for width, value in relative_scales.items()},
        "sink": sink,
    }
    for width in widths:
        print(
            f"width={width:>2} residual={width_residuals[width]:.4f} "
            f"relative_to_{args.reference_width}t={relative_scales[width]:.4f}"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
