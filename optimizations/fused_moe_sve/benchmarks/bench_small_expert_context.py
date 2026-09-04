#!/usr/bin/env python3
"""Measure one fixed 1T expert across route and background contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
COST_MODEL_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"
sys.path[:0] = [str(REPO_ROOT / "src"), str(COST_MODEL_DIR)]

from analytic_model import AnalyticMoeCostModel  # noqa: E402
from fused_cpp import _moe_C  # noqa: E402
from fused_cpp.moe import (  # noqa: E402
    AsyncMoEPlanV2,
    fused_moe_bf16_tiled_async_plan,
    prepare_fused_moe_bf16_tiled_weights,
)


MODES = (
    "full_head",
    "full_after_68",
    "wide_only_head",
    "narrow_only_head",
    "isolated_head",
)
TARGET_EXPERT = 0
PREDECESSOR_EXPERT = 1
DEFAULT_TARGET_ROUTES = 1
PREDECESSOR_ROUTES = 68
TARGET_CORE_BEGIN = 64
WIDE_BACKGROUND_ROUTES = ((1800, 600),) * 4
NARROW_BACKGROUND_ROUTES = ((68, 35, 13),) * 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytic-calibration", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--threads", type=int, default=80)
    parser.add_argument("--target-routes", type=int, default=DEFAULT_TARGET_ROUTES)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--weight-copies", type=int, default=4)
    parser.add_argument("--scrub-passes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--trace-dir", type=Path, default=Path("/tmp/moe_small_expert_context"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: list[float]) -> dict[str, object]:
    if not values:
        raise ValueError("statistics require at least one sample")
    ordered = sorted(float(value) for value in values)
    return {
        "median_ms": statistics.median(ordered),
        "p10_ms": ordered[round((len(ordered) - 1) * 0.10)],
        "p90_ms": ordered[round((len(ordered) - 1) * 0.90)],
        "samples_ms": list(values),
    }


def _paired_delta_stats(candidate: list[float], anchor: list[float]) -> dict[str, object]:
    if not candidate or len(candidate) != len(anchor):
        raise ValueError("paired samples must be non-empty and equal length")
    delta_ms = [value - base for value, base in zip(candidate, anchor, strict=True)]
    slowdown_pct = [100.0 * (value / base - 1.0) for value, base in zip(candidate, anchor, strict=True)]
    return {
        "delta": _stats(delta_ms),
        "slowdown_pct": _stats(slowdown_pct),
    }


def _task_specs(target_routes: int = DEFAULT_TARGET_ROUTES) -> list[tuple[int, int, int, int, str]]:
    if target_routes <= 0:
        raise ValueError("target_routes must be positive")
    tasks = [
        (TARGET_EXPERT, target_routes, TARGET_CORE_BEGIN, 1, "target"),
        (PREDECESSOR_EXPERT, PREDECESSOR_ROUTES, TARGET_CORE_BEGIN, 1, "target"),
    ]
    expert = 2
    for lane, routes_for_lane in enumerate(WIDE_BACKGROUND_ROUTES):
        for routes in routes_for_lane:
            tasks.append((expert, routes, 16 * lane, 16, "wide"))
            expert += 1
    for lane, routes_for_lane in enumerate(NARROW_BACKGROUND_ROUTES):
        for routes in routes_for_lane:
            tasks.append((expert, routes, TARGET_CORE_BEGIN + 1 + lane, 1, "narrow"))
            expert += 1
    return tasks


def _build_bridge(
    model: AnalyticMoeCostModel,
    *,
    thread_cpu_ids: tuple[int, ...],
    mode: str,
    target_routes: int = DEFAULT_TARGET_ROUTES,
) -> tuple[dict[str, object], dict[int, int]]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    tasks = _task_specs(target_routes)
    if mode == "full_after_68":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    delay_wide = mode in {"narrow_only_head", "isolated_head"}
    delay_narrow = mode in {"wide_only_head", "isolated_head"}
    target_tail = 1
    dependencies: list[list[int]] = []
    lane_prior: dict[tuple[int, int], int] = {}
    for task_id, (_, _, core_begin, width, family) in enumerate(tasks):
        lane = (core_begin, width)
        if lane in lane_prior:
            deps = [lane_prior[lane]]
        elif (family == "wide" and delay_wide) or (family == "narrow" and delay_narrow):
            deps = [target_tail]
        else:
            deps = []
        dependencies.append(deps)
        lane_prior[lane] = task_id

    flat_dependencies = [dependency for values in dependencies for dependency in values]
    dependency_offsets = [0]
    for values in dependencies:
        dependency_offsets.append(dependency_offsets[-1] + len(values))
    policy = model.shadow_stage_window_policy()
    windows = [policy.select(routes, width) for _, routes, _, width, _ in tasks]
    widths = [width for _, _, _, width, _ in tasks]
    num_tasks = len(tasks)
    bridge = {
        "plan_version": 2,
        "execution_mode": "strict",
        "num_threads": len(thread_cpu_ids),
        "thread_cpu_ids": list(thread_cpu_ids),
        "task_expert_ids": [expert for expert, _, _, _, _ in tasks],
        "task_core_begins": [core_begin for _, _, core_begin, _, _ in tasks],
        "task_threads": widths,
        "task_dep_offsets": dependency_offsets,
        "task_deps": flat_dependencies,
        "task_preferred_threads": widths,
        "task_min_threads": widths,
        "task_max_threads": widths,
        "task_allowed_thread_offsets": list(range(num_tasks + 1)),
        "task_allowed_threads": widths,
        "task_placement_modes": [0] * num_tasks,
        "task_numa_nodes": [-1] * num_tasks,
        "task_stage_ids": [0] * num_tasks,
        "task_resize_points": [0] * num_tasks,
        "task_range_granularities": [0] * num_tasks,
        "task_w13_window_tiles": [window[0] for window in windows],
        "task_w2_window_tiles": [window[1] for window in windows],
        "early_merge": False,
    }
    return bridge, {expert: routes for expert, routes, _, _, _ in tasks}


def _placed_tasks(
    bridge: dict[str, object],
    routes_by_expert: dict[int, int],
) -> list[tuple[int, int, tuple[int, ...], list[int]]]:
    cpu_ids = tuple(int(value) for value in bridge["thread_cpu_ids"])
    experts = [int(value) for value in bridge["task_expert_ids"]]
    begins = [int(value) for value in bridge["task_core_begins"]]
    widths = [int(value) for value in bridge["task_threads"]]
    offsets = [int(value) for value in bridge["task_dep_offsets"]]
    dependencies = [int(value) for value in bridge["task_deps"]]
    return [
        (
            routes_by_expert[expert],
            width,
            cpu_ids[core_begin : core_begin + width],
            dependencies[offsets[index] : offsets[index + 1]],
        )
        for index, (expert, core_begin, width) in enumerate(zip(experts, begins, widths, strict=True))
    ]


def _model_target_span_ms(
    model: AnalyticMoeCostModel,
    bridge: dict[str, object],
    routes_by_expert: dict[int, int],
) -> float:
    explanation = model.explain_dag_placed(_placed_tasks(bridge, routes_by_expert))
    experts = [int(value) for value in bridge["task_expert_ids"]]
    target_index = experts.index(TARGET_EXPERT)
    finish = float(explanation["task_finish_ns"][target_index])
    if target_index == 0:
        start = 0.0
    else:
        start = float(explanation["task_finish_ns"][target_index - 1])
    return (finish - start) / 1.0e6


def _parse_target_calls(path: Path) -> dict[str, list[float]]:
    calls: list[dict[str, float]] = []
    current: dict[str, float] = {}
    start_ms: float | None = None
    end_ms: float | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if not words:
            continue
        fields = dict(word.split("=", 1) for word in words[1:] if "=" in word)
        if words[0] == "MOE_CALL":
            current = {}
            start_ms = None
            end_ms = None
        elif words[0] == "PHASE" and int(fields.get("expert", -1)) == TARGET_EXPERT:
            stage = fields["stage"]
            begin = float(fields["start_ms"])
            end = float(fields["end_ms"])
            current[stage] = current.get(stage, 0.0) + float(fields["ms"])
            start_ms = begin if start_ms is None else min(start_ms, begin)
            end_ms = end if end_ms is None else max(end_ms, end)
        elif words[0] == "MOE_CALL_END":
            if start_ms is None or end_ms is None:
                raise RuntimeError(f"trace call in {path} has no target expert phases")
            calls.append({"span": end_ms - start_ms, **current})
    if not calls:
        raise RuntimeError(f"no complete target calls in {path}")
    keys = sorted({key for call in calls for key in call})
    return {key: [call.get(key, 0.0) for call in calls] for key in keys}


def _decomposition(spans: dict[str, list[float]]) -> dict[str, object]:
    isolated = spans["isolated_head"]
    comparisons = {
        "after_68_vs_full_head": _paired_delta_stats(spans["full_after_68"], spans["full_head"]),
        "wide_only_vs_isolated": _paired_delta_stats(spans["wide_only_head"], isolated),
        "narrow_only_vs_isolated": _paired_delta_stats(spans["narrow_only_head"], isolated),
        "full_vs_isolated": _paired_delta_stats(spans["full_head"], isolated),
    }
    interaction = [
        full - wide - narrow + base
        for full, wide, narrow, base in zip(
            spans["full_head"],
            spans["wide_only_head"],
            spans["narrow_only_head"],
            isolated,
            strict=True,
        )
    ]
    comparisons["wide_narrow_interaction_ms"] = _stats(interaction)
    return comparisons


def _run_scrubbed(
    run,
    *,
    mode: str,
    measured_copy: int,
    scrub_copy: int,
    scrub_passes: int,
):
    if scrub_passes <= 0:
        raise ValueError("scrub_passes must be positive")
    for _ in range(scrub_passes):
        run("full_head", scrub_copy)
    return run(mode, measured_copy)


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if args.threads != 80:
        raise ValueError("the initial context decomposition requires exactly 80 threads")
    if min(
        args.hidden,
        args.intermediate,
        args.target_routes,
        args.runs,
        args.weight_copies,
        args.scrub_passes,
    ) <= 0 or args.warmup < 0:
        raise ValueError("dimensions/runs/copies must be positive and warmup non-negative")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < args.threads:
        raise ValueError("process affinity does not expose 80 CPUs")
    cpu_ids = tuple(affinity[: args.threads])
    torch.set_num_threads(1)
    expert_count = len(_task_specs(args.target_routes))
    model = AnalyticMoeCostModel(
        args.analytic_calibration,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        global_experts=expert_count,
        local_experts=expert_count,
        mode="tp",
        degree=4,
        concurrent_ranks=1,
        down_output_element_bytes=4,
    )
    built = {
        mode: _build_bridge(
            model,
            thread_cpu_ids=cpu_ids,
            mode=mode,
            target_routes=args.target_routes,
        )
        for mode in MODES
    }
    bridges = {mode: value[0] for mode, value in built.items()}
    routes_by_expert = {mode: value[1] for mode, value in built.items()}
    plans = {mode: AsyncMoEPlanV2.from_dict(bridge) for mode, bridge in bridges.items()}
    route_map = routes_by_expert[MODES[0]]
    topk_ids = torch.repeat_interleave(
        torch.arange(expert_count, dtype=torch.int32),
        torch.tensor([route_map[expert] for expert in range(expert_count)], dtype=torch.int64),
    ).reshape(-1, 1)
    generator = torch.Generator().manual_seed(args.seed)
    hidden = torch.empty((topk_ids.shape[0], args.hidden), dtype=torch.bfloat16)
    hidden.normal_(mean=0.0, std=0.01, generator=generator)
    topk_weights = torch.ones((topk_ids.shape[0], 1), dtype=torch.float32)
    w13 = torch.empty((expert_count, 2 * args.intermediate, args.hidden), dtype=torch.bfloat16)
    w13.normal_(mean=0.0, std=0.01, generator=generator)
    w2 = torch.empty((expert_count, args.hidden, args.intermediate), dtype=torch.bfloat16)
    w2.normal_(mean=0.0, std=0.01, generator=generator)
    packed = [
        prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="arm_sve_bf16")
        for _ in range(args.weight_copies + 1)
    ]
    measured_packed = packed[: args.weight_copies]
    scrub_copy = args.weight_copies
    outputs = {mode: torch.empty_like(hidden) for mode in MODES}
    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "0"
    os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE"] = "0"

    def run(mode: str, copy_index: int) -> torch.Tensor:
        return fused_moe_bf16_tiled_async_plan(
            hidden,
            packed[copy_index],
            topk_weights,
            topk_ids,
            plans[mode],
            global_num_experts=expert_count,
            out=outputs[mode],
        )

    reference = run(MODES[0], 0).clone()
    for mode in MODES:
        torch.testing.assert_close(run(mode, 0).float(), reference.float(), atol=0, rtol=0)
    order = random.Random(args.seed ^ 0xA5A5)
    for round_index in range(args.warmup):
        names = list(MODES)
        order.shuffle(names)
        for position, mode in enumerate(names):
            _run_scrubbed(
                run,
                mode=mode,
                measured_copy=(round_index + position) % len(measured_packed),
                scrub_copy=scrub_copy,
                scrub_passes=args.scrub_passes,
            )

    trace_paths = {mode: args.trace_dir / f"{mode}.log" for mode in MODES}
    for trace_path in trace_paths.values():
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.unlink(missing_ok=True)
    trace_order = random.Random(args.seed ^ 0x5A5A)
    for run_index in range(args.runs):
        names = list(MODES)
        trace_order.shuffle(names)
        for position, mode in enumerate(names):
            os.environ["FUSED_CPP_MOE_TRACE"] = "0"
            for _ in range(args.scrub_passes):
                run("full_head", scrub_copy)
            os.environ["FUSED_CPP_MOE_TRACE_FILE"] = str(trace_paths[mode])
            os.environ["FUSED_CPP_MOE_TRACE"] = "1"
            run(mode, (run_index + position) % len(measured_packed))
            os.environ["FUSED_CPP_MOE_TRACE"] = "0"

    traces = {mode: _parse_target_calls(path) for mode, path in trace_paths.items()}
    spans = {mode: trace["span"] for mode, trace in traces.items()}
    model_spans = {
        mode: _model_target_span_ms(model, bridges[mode], routes_by_expert[mode])
        for mode in MODES
    }
    result = {
        "kind": "small_expert_context_decomposition",
        "identity": {
            "calibration": str(args.analytic_calibration),
            "calibration_sha256": _sha256(args.analytic_calibration),
            "extension": str(_moe_C.__file__),
            "extension_sha256": _sha256(Path(_moe_C.__file__)),
        },
        "shape": {
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "experts": expert_count,
            "threads": args.threads,
            "target_core_begin": TARGET_CORE_BEGIN,
            "target_expert": TARGET_EXPERT,
            "target_routes": args.target_routes,
            "predecessor_expert": PREDECESSOR_EXPERT,
            "predecessor_routes": PREDECESSOR_ROUTES,
            "wide_background": [list(values) for values in WIDE_BACKGROUND_ROUTES],
            "narrow_background": [list(values) for values in NARROW_BACKGROUND_ROUTES],
        },
        "method": {
            "warmup": args.warmup,
            "trace_runs": args.runs,
            "trace_order": "randomized_paired_rounds",
            "weight_copies": args.weight_copies,
            "packed_weight_copies": len(packed),
            "scrub_policy": "dedicated_disjoint_full_workload_packed_copy_before_every_sample",
            "scrub_plan": "full_head",
            "scrub_passes_per_sample": args.scrub_passes,
            "scrub_trace_enabled": False,
            "scrub_in_target_span": False,
            "scrub_barrier": "synchronous_plan_return",
            "same_tasks_routes_weights_across_modes": True,
            "disabled_background_policy": "first task depends on target-lane tail",
            "metric": "target expert first-compute-phase start to last-compute-phase end",
            "fit_parameters": False,
        },
        "modes": {
            mode: {
                "target_span": _stats(spans[mode]),
                "stages": {stage: _stats(values) for stage, values in traces[mode].items() if stage != "span"},
                "model_target_span_ms": model_spans[mode],
                "model_error_pct": 100.0 * (model_spans[mode] / statistics.median(spans[mode]) - 1.0),
            }
            for mode in MODES
        },
        "decomposition": _decomposition(spans),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
