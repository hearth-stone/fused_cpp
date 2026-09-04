#!/usr/bin/env python3
"""Collect an isolated gather/W13/W2 fit corpus without whole-expert residual fitting."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
COST_MODEL_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src"), str(COST_MODEL_DIR)]

from analytic_model import AnalyticMoeCostModel  # noqa: E402
from fused_cpp import _moe_C  # noqa: E402
from fused_cpp.moe import (  # noqa: E402
    AsyncMoEPlanV2,
    fused_moe_bf16_tiled_async_plan,
    prepare_fused_moe_bf16_tiled_weights,
)
from optimizations.fused_moe_sve.benchmarks.bench_small_expert_context import (  # noqa: E402
    _sha256,
    _stats,
)


DEFAULT_ROUTES = (3, 4, 7, 8, 10, 16, 24, 48, 68, 600, 1800)
DEFAULT_WIDTHS = (1, 2, 4, 8, 16, 32, 40, 80)
FORBIDDEN_HOLDOUT_ROUTES = frozenset({1, 2, 5, 6, 12})
EXPERT_COUNT = 17
SCRUB_ROUTES = 68


def _positive_int_list(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytic-calibration", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--threads", type=int, default=80)
    parser.add_argument("--routes", type=_positive_int_list, default=DEFAULT_ROUTES)
    parser.add_argument("--widths", type=_positive_int_list, default=DEFAULT_WIDTHS)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--weight-copies", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--trace-dir", type=Path, default=Path("/tmp/moe_isolated_phase_reaccount"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _bridge(
    model: AnalyticMoeCostModel,
    *,
    thread_cpu_ids: tuple[int, ...],
    tasks: tuple[tuple[int, int, int, int], ...],
) -> dict[str, object]:
    policy = model.shadow_stage_window_policy()
    windows = [policy.select(routes, width) for _, routes, _, width in tasks]
    widths = [width for _, _, _, width in tasks]
    return {
        "plan_version": 2,
        "execution_mode": "strict",
        "num_threads": len(thread_cpu_ids),
        "thread_cpu_ids": list(thread_cpu_ids),
        "task_expert_ids": [expert for expert, _, _, _ in tasks],
        "task_core_begins": [core_begin for _, _, core_begin, _ in tasks],
        "task_threads": widths,
        "task_dep_offsets": [0] * (len(tasks) + 1),
        "task_deps": [],
        "task_preferred_threads": widths,
        "task_min_threads": widths,
        "task_max_threads": widths,
        "task_allowed_thread_offsets": list(range(len(tasks) + 1)),
        "task_allowed_threads": widths,
        "task_placement_modes": [0] * len(tasks),
        "task_numa_nodes": [-1] * len(tasks),
        "task_stage_ids": [0] * len(tasks),
        "task_resize_points": [0] * len(tasks),
        "task_range_granularities": [0] * len(tasks),
        "task_w13_window_tiles": [window[0] for window in windows],
        "task_w2_window_tiles": [window[1] for window in windows],
        "early_merge": False,
    }


def _measured_bridge(
    model: AnalyticMoeCostModel,
    *,
    thread_cpu_ids: tuple[int, ...],
    routes: int,
    width: int,
) -> dict[str, object]:
    if width <= 0 or width > len(thread_cpu_ids):
        raise ValueError("isolated fit width must be within the available rank")
    core_begin = 80 - width if width >= 32 else 64
    return _bridge(
        model,
        thread_cpu_ids=thread_cpu_ids,
        tasks=((0, routes, core_begin, width),),
    )


def _scrub_bridge(
    model: AnalyticMoeCostModel,
    *,
    thread_cpu_ids: tuple[int, ...],
) -> dict[str, object]:
    return _bridge(
        model,
        thread_cpu_ids=thread_cpu_ids,
        tasks=tuple((expert, SCRUB_ROUTES, expert, 1) for expert in range(EXPERT_COUNT)),
    )


def _validate_fit_domain(routes: tuple[int, ...], widths: tuple[int, ...], threads: int) -> None:
    overlap = sorted(set(routes) & FORBIDDEN_HOLDOUT_ROUTES)
    if overlap:
        raise ValueError(f"fit routes overlap locked holdout routes: {overlap}")
    if widths[-1] > threads:
        raise ValueError("fit widths must be no larger than available threads")


def _parse_stage_envelopes(path: Path) -> dict[str, list[float]]:
    calls: list[dict[str, float]] = []
    stage_bounds: dict[str, tuple[float, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if not words:
            continue
        fields = dict(word.split("=", 1) for word in words[1:] if "=" in word)
        if words[0] == "MOE_CALL":
            stage_bounds = {}
        elif words[0] == "PHASE" and int(fields.get("expert", -1)) == 0:
            stage = fields["stage"]
            begin = float(fields["start_ms"])
            end = float(fields["end_ms"])
            old_begin, old_end = stage_bounds.get(stage, (begin, end))
            stage_bounds[stage] = (min(old_begin, begin), max(old_end, end))
        elif words[0] == "MOE_CALL_END":
            if not stage_bounds:
                raise RuntimeError(f"trace call in {path} has no target expert phases")
            call_begin = min(begin for begin, _ in stage_bounds.values())
            call_end = max(end for _, end in stage_bounds.values())
            calls.append(
                {
                    "span": call_end - call_begin,
                    **{
                        stage: end - begin
                        for stage, (begin, end) in stage_bounds.items()
                    },
                }
            )
    if not calls:
        raise RuntimeError(f"no complete target calls in {path}")
    keys = sorted({key for call in calls for key in call})
    return {key: [call.get(key, 0.0) for call in calls] for key in keys}


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if args.threads != 80:
        raise ValueError("the Arm isolated phase fit requires exactly 80 threads")
    if min(args.hidden, args.intermediate, args.runs, args.weight_copies) <= 0 or args.warmup < 0:
        raise ValueError("dimensions/runs/copies must be positive and warmup non-negative")
    _validate_fit_domain(args.routes, args.widths, args.threads)
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < args.threads:
        raise ValueError("process affinity does not expose 80 CPUs")
    cpu_ids = tuple(affinity[: args.threads])
    torch.set_num_threads(1)
    model = AnalyticMoeCostModel(
        args.analytic_calibration,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        global_experts=EXPERT_COUNT,
        local_experts=EXPERT_COUNT,
        mode="tp",
        degree=4,
        concurrent_ranks=1,
        down_output_element_bytes=4,
    )
    points = tuple((routes, width) for width in args.widths for routes in args.routes)
    bridges = {
        point: _measured_bridge(
            model,
            thread_cpu_ids=cpu_ids,
            routes=point[0],
            width=point[1],
        )
        for point in points
    }
    plans = {point: AsyncMoEPlanV2.from_dict(bridge) for point, bridge in bridges.items()}
    scrub_plan = AsyncMoEPlanV2.from_dict(_scrub_bridge(model, thread_cpu_ids=cpu_ids))
    generator = torch.Generator().manual_seed(args.seed)
    w13 = torch.empty((EXPERT_COUNT, 2 * args.intermediate, args.hidden), dtype=torch.bfloat16)
    w13.normal_(mean=0.0, std=0.01, generator=generator)
    w2 = torch.empty((EXPERT_COUNT, args.hidden, args.intermediate), dtype=torch.bfloat16)
    w2.normal_(mean=0.0, std=0.01, generator=generator)
    packed = [
        prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="arm_sve_bf16")
        for _ in range(args.weight_copies + 1)
    ]
    scrub_copy = args.weight_copies
    point_inputs = {}
    for routes, width in points:
        del width
        if routes in point_inputs:
            continue
        hidden = torch.empty((routes, args.hidden), dtype=torch.bfloat16)
        hidden.normal_(mean=0.0, std=0.01, generator=generator)
        point_inputs[routes] = (
            hidden,
            torch.ones((routes, 1), dtype=torch.float32),
            torch.zeros((routes, 1), dtype=torch.int32),
        )
    scrub_rows = EXPERT_COUNT * SCRUB_ROUTES
    scrub_hidden = torch.empty((scrub_rows, args.hidden), dtype=torch.bfloat16)
    scrub_hidden.normal_(mean=0.0, std=0.01, generator=generator)
    scrub_weights = torch.ones((scrub_rows, 1), dtype=torch.float32)
    scrub_ids = torch.repeat_interleave(
        torch.arange(EXPERT_COUNT, dtype=torch.int32),
        SCRUB_ROUTES,
    ).reshape(-1, 1)
    point_outputs = {
        point: torch.empty_like(point_inputs[point[0]][0])
        for point in points
    }
    scrub_output = torch.empty_like(scrub_hidden)
    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "0"
    os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE"] = "0"

    def run_point(point: tuple[int, int], copy_index: int) -> torch.Tensor:
        hidden, topk_weights, topk_ids = point_inputs[point[0]]
        return fused_moe_bf16_tiled_async_plan(
            hidden,
            packed[copy_index],
            topk_weights,
            topk_ids,
            plans[point],
            global_num_experts=EXPERT_COUNT,
            out=point_outputs[point],
        )

    def scrub() -> None:
        fused_moe_bf16_tiled_async_plan(
            scrub_hidden,
            packed[scrub_copy],
            scrub_weights,
            scrub_ids,
            scrub_plan,
            global_num_experts=EXPERT_COUNT,
            out=scrub_output,
        )

    for point in points:
        reference = run_point(point, 0).clone()
        torch.testing.assert_close(run_point(point, 1).float(), reference.float(), atol=0, rtol=0)
    warmup_order = random.Random(args.seed ^ 0xA5A5)
    for round_index in range(args.warmup):
        shuffled = list(points)
        warmup_order.shuffle(shuffled)
        for position, point in enumerate(shuffled):
            scrub()
            run_point(point, (round_index + position) % args.weight_copies)

    trace_paths = {
        point: args.trace_dir / f"m{point[0]}_t{point[1]}.log"
        for point in points
    }
    for trace_path in trace_paths.values():
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.unlink(missing_ok=True)
    trace_order = random.Random(args.seed ^ 0x5A5A)
    for run_index in range(args.runs):
        shuffled = list(points)
        trace_order.shuffle(shuffled)
        for position, point in enumerate(shuffled):
            os.environ["FUSED_CPP_MOE_TRACE"] = "0"
            scrub()
            os.environ["FUSED_CPP_MOE_TRACE_FILE"] = str(trace_paths[point])
            os.environ["FUSED_CPP_MOE_TRACE"] = "1"
            run_point(point, (run_index + position) % args.weight_copies)
            os.environ["FUSED_CPP_MOE_TRACE"] = "0"

    rows = []
    for routes, width in points:
        trace = _parse_stage_envelopes(trace_paths[(routes, width)])
        bridge = bridges[(routes, width)]
        rows.append(
            {
                "routes": routes,
                "threads": width,
                "core_begin": bridge["task_core_begins"][0],
                "w13_window_tiles": bridge["task_w13_window_tiles"][0],
                "w2_window_tiles": bridge["task_w2_window_tiles"][0],
                "target_span": _stats(trace["span"]),
                "phases": {
                    stage: _stats(values)
                    for stage, values in trace.items()
                    if stage != "span"
                },
            }
        )
    result = {
        "kind": "moe_isolated_phase_reaccount_fit",
        "artifact_role": "fit_only",
        "identity": {
            "calibration": str(args.analytic_calibration),
            "calibration_sha256": _sha256(args.analytic_calibration),
            "extension": str(_moe_C.__file__),
            "extension_sha256": _sha256(Path(_moe_C.__file__)),
        },
        "shape": {
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "experts": EXPERT_COUNT,
            "threads": args.threads,
            "routes": list(args.routes),
            "widths": list(args.widths),
        },
        "method": {
            "warmup": args.warmup,
            "trace_runs": args.runs,
            "trace_order": "randomized_rounds_across_all_points",
            "weight_copies": args.weight_copies,
            "scrub_policy": "dedicated_disjoint_17_expert_68_route_copy_before_every_sample",
            "scrub_weight_working_set_bytes": EXPERT_COUNT * 12 * 1024 * 1024,
            "forbidden_holdout_routes": sorted(FORBIDDEN_HOLDOUT_ROUTES),
            "whole_expert_total_used_for_fit": False,
        },
        "points": rows,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
