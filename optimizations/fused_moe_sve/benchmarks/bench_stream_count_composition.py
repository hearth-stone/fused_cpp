#!/usr/bin/env python3
"""Test whether leftover tax tracks stream count, not thread count.

A 16-thread team on one expert is one packed-B stream. This probe asks whether
8+8+4+1 therefore taxes like four 1T streams, not like 21 independent fills.
Equal-thread ladder on cores 48-63 holds thread count at 16 while stream count
goes 1/2/4/16. The mix uses cores 43-63. Do not add a structure.
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


OUTPUT_KIND = "moe_stream_count_composition_probe"
SCHEMA_VERSION = 1
DELAY_EXPERT = 1
DELAY_ROUTES = 1
BACKGROUND_EXPERTS = 21
BACKGROUND_ROUTES = 1
TARGET_CORE_BEGIN = 64
SHEET16 = tuple(range(48, 64))
MIX21 = tuple(range(43, 64))
CROSS21 = tuple(range(0, 21))
MIX_WIDTHS = (8, 8, 4, 1)
MODES = (
    "isolated_head",
    "wide16_same_head",
    "two_8t_same_head",
    "four_4t_same_head",
    "four_1t_same_head",
    "many16_1t_same_head",
    "mix_8841_same_head",
    "four_1t_mix_same_head",
    "many21_1t_same_head",
    "mix_8841_cross_head",
    "many21_1t_cross_head",
)
STREAM_MATCH_MARGIN_MS = 0.02
REMOTE_NEAR_ZERO_MS = 0.08
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
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--trace-dir", type=Path, default=Path("/tmp/moe_stream_count_composition"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def parse_mode(mode: str) -> tuple[str, str]:
    known = {
        "isolated_head": ("isolated", "same_llc"),
        "wide16_same_head": ("wide16", "same_llc"),
        "two_8t_same_head": ("two_8t", "same_llc"),
        "four_4t_same_head": ("four_4t", "same_llc"),
        "four_1t_same_head": ("four_1t", "same_llc"),
        "many16_1t_same_head": ("many16", "same_llc"),
        "mix_8841_same_head": ("mix_8841", "same_llc"),
        "four_1t_mix_same_head": ("four_1t_mix", "same_llc"),
        "many21_1t_same_head": ("many21", "same_llc"),
        "mix_8841_cross_head": ("mix_8841", "cross_llc"),
        "many21_1t_cross_head": ("many21", "cross_llc"),
    }
    if mode not in known:
        raise ValueError(f"unsupported stream-count mode {mode!r}")
    return known[mode]


def place_widths(cores: tuple[int, ...], widths: tuple[int, ...]) -> list[tuple[int, int]]:
    offset = 0
    placed: list[tuple[int, int]] = []
    for width in widths:
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if offset + width > len(cores):
            raise ValueError(f"widths {widths} overflow {len(cores)} cores")
        placed.append((cores[offset], width))
        offset += width
    if offset != len(cores):
        raise ValueError(f"widths {widths} cover {offset} cores, not {len(cores)}")
    return placed


def start_cores(placed: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(core, 1) for core, _width in placed]


def occupied_cores(placed: list[tuple[int, int]]) -> list[int]:
    cores: list[int] = []
    for begin, width in placed:
        cores.extend(range(begin, begin + width))
    return cores


def _sheet_cores(placement: str, count: int) -> tuple[int, ...]:
    if placement == "same_llc":
        return MIX21 if count == 21 else SHEET16
    if count == 21:
        return CROSS21
    return CROSS21[:16]


def live_teams(mode: str) -> list[tuple[int, int]]:
    kind, placement = parse_mode(mode)
    sheet16 = _sheet_cores(placement, 16)
    mix21 = _sheet_cores(placement, 21)
    if kind == "isolated":
        return []
    if kind == "wide16":
        return place_widths(sheet16, (16,))
    if kind == "two_8t":
        return place_widths(sheet16, (8, 8))
    if kind == "four_4t":
        return place_widths(sheet16, (4, 4, 4, 4))
    if kind == "four_1t":
        return start_cores(place_widths(sheet16, (4, 4, 4, 4)))
    if kind == "many16":
        return place_widths(sheet16, (1,) * 16)
    if kind == "mix_8841":
        return place_widths(mix21, MIX_WIDTHS)
    if kind == "four_1t_mix":
        return start_cores(place_widths(mix21, MIX_WIDTHS))
    if kind == "many21":
        return place_widths(mix21, (1,) * 21)
    raise ValueError(f"unsupported stream-count kind {kind!r}")


def _task_specs(mode: str) -> list[tuple[int, int, int, int, bool]]:
    kind, placement = parse_mode(mode)
    target = [
        (TARGET_EXPERT, TARGET_ROUTES, TARGET_CORE_BEGIN, 1, False),
        (DELAY_EXPERT, DELAY_ROUTES, TARGET_CORE_BEGIN, 1, True),
    ]
    live = live_teams(mode)
    park = MIX21 if kind == "isolated" else (CROSS21 if placement == "same_llc" else MIX21)
    background: list[tuple[int, int, int, int, bool]] = []
    for index in range(BACKGROUND_EXPERTS):
        if index < len(live):
            core_begin, width = live[index]
            background.append((2 + index, BACKGROUND_ROUTES, core_begin, width, False))
            continue
        background.append(
            (
                2 + index,
                BACKGROUND_ROUTES,
                park[(index - len(live)) % len(park)],
                1,
                True,
            )
        )
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


def _closer(value: float, matched: float, other: float, margin: float = STREAM_MATCH_MARGIN_MS) -> bool:
    return abs(value - matched) + margin < abs(value - other)


def decide_stream_count(
    comparisons: dict[str, dict[str, object]],
    calls: dict[str, dict[str, list[float]]],
) -> dict[str, object]:
    def delta(mode: str) -> float:
        return float(comparisons[f"{mode}_vs_isolated"]["delta"]["median_ms"])

    wide16 = delta("wide16_same_head")
    two_8t = delta("two_8t_same_head")
    four_4t = delta("four_4t_same_head")
    four_1t = delta("four_1t_same_head")
    many16 = delta("many16_1t_same_head")
    mix = delta("mix_8841_same_head")
    four_1t_mix = delta("four_1t_mix_same_head")
    many21 = delta("many21_1t_same_head")
    mix_cross = delta("mix_8841_cross_head")
    many21_cross = delta("many21_1t_cross_head")
    overlap_ok = (
        _overlap_ok(calls["wide16_same_head"], 1.0)
        and _overlap_ok(calls["two_8t_same_head"], 2.0)
        and _overlap_ok(calls["four_4t_same_head"], 4.0)
        and _overlap_ok(calls["four_1t_same_head"], 4.0)
        and _overlap_ok(calls["many16_1t_same_head"], 16.0)
        and _overlap_ok(calls["mix_8841_same_head"], 4.0)
        and _overlap_ok(calls["four_1t_mix_same_head"], 4.0)
        and _overlap_ok(calls["many21_1t_same_head"], 21.0)
    )
    four_matches_stream = _closer(four_4t, four_1t, many16)
    mix_matches_stream = _closer(mix, four_1t_mix, many21)
    four_matches_threads = _closer(four_4t, many16, four_1t)
    mix_matches_threads = _closer(mix, many21, four_1t_mix)
    stream_count = four_matches_stream and mix_matches_stream
    thread_count = four_matches_threads and mix_matches_threads
    remote_near_zero = abs(mix_cross) <= REMOTE_NEAR_ZERO_MS and abs(many21_cross) <= REMOTE_NEAR_ZERO_MS
    if not overlap_ok:
        signature = "invalid_overlap"
    elif stream_count:
        signature = "stream_count"
    elif thread_count:
        signature = "thread_count"
    else:
        signature = "inconclusive"
    return {
        "add_default_off_structure": False,
        "signature": signature,
        "overlap_ok": overlap_ok,
        "stream_count": stream_count,
        "thread_count": thread_count,
        "four_matches_stream": four_matches_stream,
        "mix_matches_stream": mix_matches_stream,
        "four_matches_threads": four_matches_threads,
        "mix_matches_threads": mix_matches_threads,
        "remote_near_zero": remote_near_zero,
        "wide16_same_ms": wide16,
        "two_8t_same_ms": two_8t,
        "four_4t_same_ms": four_4t,
        "four_1t_same_ms": four_1t,
        "many16_1t_same_ms": many16,
        "mix_8841_same_ms": mix,
        "four_1t_mix_same_ms": four_1t_mix,
        "many21_1t_same_ms": many21,
        "mix_8841_cross_ms": mix_cross,
        "many21_1t_cross_ms": many21_cross,
        "gates": {
            "stream_match_margin_ms": STREAM_MATCH_MARGIN_MS,
            "remote_near_zero_ms": REMOTE_NEAR_ZERO_MS,
        },
        "reason": (
            "Do not add a structure. stream_count means 4x4T matches 4x1T and "
            "8+8+4+1 matches four 1T starts, not the equal-thread many-1T fills."
        ),
    }


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if args.threads != 80:
        raise ValueError("the Arm stream-count probe requires exactly 80 threads")
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
        raise RuntimeError("stream-count modes must keep the same expert route histogram")
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
    decision = decide_stream_count(comparisons, calls)
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
            "mix_widths": list(MIX_WIDTHS),
            "sheet16_cores": list(SHEET16),
            "mix21_cores": list(MIX21),
            "cross21_cores": list(CROSS21),
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
                "placement": parse_mode(mode)[1],
                "live_teams": live_teams(mode),
                "live_streams": len(live_teams(mode)),
                "live_threads": sum(width for _core, width in live_teams(mode)),
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
