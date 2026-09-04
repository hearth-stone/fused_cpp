#!/usr/bin/env python3
"""Compare one 16T stream against sixteen 1T streams on the same LLC cores.

The leftover same-LLC tax tracks concurrent transfer-bound fills. This probe
asks whether a 16-thread team is one stream or sixteen fill ports. Victim stays
M=1 1T. All aggressors are M=1 so every peer is transfer-bound. Do not add a
structure from this probe.
"""

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
from optimizations.fused_moe_sve.benchmarks.bench_gather_injection_overlap import (  # noqa: E402
    TARGET_EXPERT,
    TARGET_ROUTES,
    _cpu_mapping,
    _parse_calls,
)
from optimizations.fused_moe_sve.benchmarks.bench_small_expert_context import (  # noqa: E402
    _paired_delta_stats,
    _sha256,
    _stats,
)


OUTPUT_KIND = "moe_fill_port_vs_stream_probe"
SCHEMA_VERSION = 1
DELAY_EXPERT = 1
DELAY_ROUTES = 1
BACKGROUND_EXPERTS = 16
BACKGROUND_ROUTES = 1
TARGET_CORE_BEGIN = 64
TEAM_WIDTH = 16
SAME_LLC_TEAM = tuple(range(48, 64))
CROSS_LLC_TEAM = tuple(range(0, 16))
MODES = (
    "isolated_head",
    "one_1t_same_head",
    "wide16_same_head",
    "many16_1t_same_head",
    "wide16_cross_head",
    "many16_1t_cross_head",
)
FILL_PORT_CLOSE_MS = 0.10
STREAM_CLOSE_MS = 0.10
MANY_MINUS_WIDE_MS = 0.20
REMOTE_NEAR_ZERO_MS = 0.08
SATURATED_STREAM_MS = 0.40
OVERLAP_EXPERT_FRACTION = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytic-calibration", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--threads", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--weight-copies", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--trace-dir", type=Path, default=Path("/tmp/moe_fill_port_vs_stream"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def parse_mode(mode: str) -> tuple[str, str, str]:
    known = {
        "isolated_head": ("isolated", "1t", "same_llc"),
        "one_1t_same_head": ("one", "1t", "same_llc"),
        "wide16_same_head": ("wide", "16t", "same_llc"),
        "many16_1t_same_head": ("many", "1t", "same_llc"),
        "wide16_cross_head": ("wide", "16t", "cross_llc"),
        "many16_1t_cross_head": ("many", "1t", "cross_llc"),
    }
    if mode not in known:
        raise ValueError(f"unsupported fill-port mode {mode!r}")
    return known[mode]


def _team_cores(placement: str) -> tuple[int, ...]:
    if placement == "same_llc":
        return SAME_LLC_TEAM
    if placement == "cross_llc":
        return CROSS_LLC_TEAM
    raise ValueError(f"unsupported placement {placement!r}")


def _park_cores(placement: str) -> tuple[int, ...]:
    return CROSS_LLC_TEAM if placement == "same_llc" else SAME_LLC_TEAM


def _task_specs(mode: str) -> list[tuple[int, int, int, int, bool]]:
    kind, _width, placement = parse_mode(mode)
    target = [
        (TARGET_EXPERT, TARGET_ROUTES, TARGET_CORE_BEGIN, 1, False),
        (DELAY_EXPERT, DELAY_ROUTES, TARGET_CORE_BEGIN, 1, True),
    ]
    team = _team_cores(placement)
    park = _park_cores(placement)
    background: list[tuple[int, int, int, int, bool]] = []
    if kind == "isolated":
        for index in range(BACKGROUND_EXPERTS):
            background.append((2 + index, BACKGROUND_ROUTES, SAME_LLC_TEAM[index], 1, True))
    elif kind == "one":
        background.append((2, BACKGROUND_ROUTES, team[0], 1, False))
        for index in range(1, BACKGROUND_EXPERTS):
            background.append((2 + index, BACKGROUND_ROUTES, team[index], 1, True))
    elif kind == "many":
        for index in range(BACKGROUND_EXPERTS):
            background.append((2 + index, BACKGROUND_ROUTES, team[index], 1, False))
    else:
        background.append((2, BACKGROUND_ROUTES, team[0], TEAM_WIDTH, False))
        for index in range(1, BACKGROUND_EXPERTS):
            background.append((2 + index, BACKGROUND_ROUTES, park[index - 1], 1, True))
    return target + background


def _build_bridge(
    model: AnalyticMoeCostModel,
    *,
    thread_cpu_ids: tuple[int, ...],
    mode: str,
) -> tuple[dict[str, object], dict[int, int]]:
    tasks = _task_specs(mode)
    dependencies: list[list[int]] = []
    for index, (_expert, _routes, _core, _width, delayed) in enumerate(tasks):
        if index == 0:
            dependencies.append([])
        elif index == 1:
            dependencies.append([0])
        elif delayed:
            dependencies.append([1])
        else:
            dependencies.append([])
    flat_dependencies = [dependency for values in dependencies for dependency in values]
    dependency_offsets = [0]
    for values in dependencies:
        dependency_offsets.append(dependency_offsets[-1] + len(values))
    policy = model.shadow_stage_window_policy()
    windows = [policy.select(routes, width) for _, routes, _, width, _ in tasks]
    widths = [width for _, _, _, width, _ in tasks]
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
    return bridge, {expert: routes for expert, routes, _, _, _ in tasks}


def _overlap_ok(mode_calls: dict[str, list[float]], expected: float) -> bool:
    experts = mode_calls.get("peer_overlap_experts")
    if not experts:
        return expected <= 0.0
    return float(_stats(experts)["median_ms"]) >= OVERLAP_EXPERT_FRACTION * expected


def decide_fill_ports(comparisons: dict[str, dict[str, object]], calls: dict[str, dict[str, list[float]]]) -> dict[str, object]:
    def delta(mode: str) -> float:
        return float(comparisons[f"{mode}_vs_isolated"]["delta"]["median_ms"])

    n1 = delta("one_1t_same_head")
    wide = delta("wide16_same_head")
    many = delta("many16_1t_same_head")
    wide_cross = delta("wide16_cross_head")
    many_cross = delta("many16_1t_cross_head")
    fill_ports = abs(wide - many) <= FILL_PORT_CLOSE_MS and many >= SATURATED_STREAM_MS
    one_stream = abs(wide - n1) <= STREAM_CLOSE_MS and many - wide >= MANY_MINUS_WIDE_MS
    remote_near_zero = abs(wide_cross) <= REMOTE_NEAR_ZERO_MS and abs(many_cross) <= REMOTE_NEAR_ZERO_MS
    overlap_ok = (
        _overlap_ok(calls["one_1t_same_head"], 1.0)
        and _overlap_ok(calls["wide16_same_head"], 1.0)
        and _overlap_ok(calls["many16_1t_same_head"], float(BACKGROUND_EXPERTS))
    )
    if not overlap_ok:
        signature = "invalid_overlap"
    elif fill_ports:
        signature = "fill_ports"
    elif one_stream:
        signature = "one_stream"
    else:
        signature = "inconclusive"
    return {
        "add_default_off_structure": False,
        "signature": signature,
        "overlap_ok": overlap_ok,
        "fill_ports": fill_ports,
        "one_stream": one_stream,
        "remote_near_zero": remote_near_zero,
        "one_1t_same_ms": n1,
        "wide16_same_ms": wide,
        "many16_1t_same_ms": many,
        "wide16_cross_ms": wide_cross,
        "many16_1t_cross_ms": many_cross,
        "gates": {
            "fill_port_close_ms": FILL_PORT_CLOSE_MS,
            "stream_close_ms": STREAM_CLOSE_MS,
            "many_minus_wide_ms": MANY_MINUS_WIDE_MS,
            "remote_near_zero_ms": REMOTE_NEAR_ZERO_MS,
            "saturated_stream_ms": SATURATED_STREAM_MS,
        },
        "reason": (
            "Do not add a structure. fill_ports means 1x16T matches 16x1T. "
            "one_stream means 1x16T matches a single 1T neighbor."
        ),
    }


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if args.threads != 80:
        raise ValueError("the Arm fill-port probe requires exactly 80 threads")
    if min(args.hidden, args.intermediate, args.runs, args.weight_copies) <= 0 or args.warmup < 0:
        raise ValueError("dimensions/runs/copies must be positive and warmup non-negative")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < args.threads:
        raise ValueError("process affinity does not expose 80 CPUs")
    cpu_ids = tuple(affinity[: args.threads])
    torch.set_num_threads(1)
    expert_count = 2 + BACKGROUND_EXPERTS
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
    built = {mode: _build_bridge(model, thread_cpu_ids=cpu_ids, mode=mode) for mode in MODES}
    route_map = built[MODES[0]][1]
    if any(built[mode][1] != route_map for mode in MODES):
        raise RuntimeError("fill-port modes must keep the same expert route histogram")
    plans = {mode: AsyncMoEPlanV2.from_dict(value[0]) for mode, value in built.items()}
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
    scrub_copy = args.weight_copies
    outputs = {mode: torch.empty_like(hidden) for mode in MODES}
    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "0"
    os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE"] = "0"
    scrub = "isolated_head"

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
    warmup_order = random.Random(args.seed ^ 0xA5A5)
    for round_index in range(args.warmup):
        names = list(MODES)
        warmup_order.shuffle(names)
        for position, mode in enumerate(names):
            run(scrub, scrub_copy)
            run(mode, (round_index + position) % args.weight_copies)

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
            run(scrub, scrub_copy)
            os.environ["FUSED_CPP_MOE_TRACE_FILE"] = str(trace_paths[mode])
            os.environ["FUSED_CPP_MOE_TRACE"] = "1"
            run(mode, (run_index + position) % args.weight_copies)
            os.environ["FUSED_CPP_MOE_TRACE"] = "0"

    calls = {mode: _parse_calls(path) for mode, path in trace_paths.items()}
    spans = {mode: values["target_span"] for mode, values in calls.items()}
    isolated = spans["isolated_head"]
    comparisons = {
        f"{mode}_vs_isolated": _paired_delta_stats(spans[mode], isolated)
        for mode in MODES
        if mode != "isolated_head"
    }
    decision = decide_fill_ports(comparisons, calls)
    result = {
        "kind": OUTPUT_KIND,
        "schema_version": SCHEMA_VERSION,
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
            "target_routes": TARGET_ROUTES,
            "background_experts": BACKGROUND_EXPERTS,
            "background_routes": BACKGROUND_ROUTES,
            "team_width": TEAM_WIDTH,
            "same_llc_team_cores": list(SAME_LLC_TEAM),
            "cross_llc_team_cores": list(CROSS_LLC_TEAM),
            "target_core_begin": TARGET_CORE_BEGIN,
        },
        "cpu_mapping": _cpu_mapping(cpu_ids),
        "method": {
            "warmup": args.warmup,
            "trace_runs": args.runs,
            "trace_order": "randomized_paired_rounds",
            "weight_copies": args.weight_copies,
            "scrub_policy": "dedicated_disjoint_isolated_packed_copy_before_every_sample",
            "scrub_mode": scrub,
            "same_tasks_routes_weights_across_modes": True,
            "unused_aggressors": "dependency_delayed_after_target_lane_tail",
            "metric": "target first-phase start to final-phase end",
        },
        "modes": {
            mode: {
                "kind": parse_mode(mode)[0],
                "width": parse_mode(mode)[1],
                "placement": parse_mode(mode)[2],
                **{key: _stats(values) for key, values in mode_calls.items()},
            }
            for mode, mode_calls in calls.items()
        },
        "comparisons": comparisons,
        "decision": decision,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
