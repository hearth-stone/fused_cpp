#!/usr/bin/env python3
"""Sweep aggressor route count M at fixed peer occupancy.

The leftover same-LLC tax is either occupancy (people on LLC7) or utilization
(peer bytes). This probe keeps victim M=1 and aggressor count in {0,1,4}, and
varies aggressor routes through {1,4,16,68}. Packed-B is almost independent of
M; A/C streaming and GEMM duration grow with M. Do not fit T_sat on the old
count curve. Do not add a structure from this probe alone.
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
    BACKGROUND_EXPERTS,
    BACKGROUND_ROUTES,
    TARGET_ROUTES,
    _build_bridge,
    _comparisons,
    _cpu_mapping,
    _parse_calls,
    _scrub_mode,
    enumerate_probe_modes,
    mode_name,
    parse_aggressor_counts,
    parse_mode,
)
from optimizations.fused_moe_sve.benchmarks.bench_small_expert_context import (  # noqa: E402
    _model_target_span_ms,
    _sha256,
    _stats,
)


OUTPUT_KIND = "moe_aggressor_m_occupancy_probe"
SCHEMA_VERSION = 1
RANK_CPU_IDS = tuple(range(240, 320))
DEFAULT_AGGRESSOR_COUNTS = (0, 1, 4)
DEFAULT_AGGRESSOR_ROUTES = (1, 4, 16, 68)
ANCHOR_ROUTES = (1, 68)
REQUIRED_COUNTS = (0, 1, 4)
REMOTE_NEAR_ZERO_MS = 0.08
CONTRAST_PRESENT_MS = 0.10
OCCUPANCY_STABLE_MS = 0.05
UTILIZATION_GROWTH_MS = 0.15
OVERLAP_EXPERT_FRACTION = 0.8
HEAD_N4 = ("same_llc", "head", 4)
HEAD_N1 = ("same_llc", "head", 1)
CROSS_HEAD_N4 = ("cross_llc", "head", 4)


def parse_occupancy_counts(raw: str) -> tuple[int, ...]:
    counts = parse_aggressor_counts(raw)
    missing = [value for value in REQUIRED_COUNTS if value not in counts]
    if missing:
        raise ValueError(f"--aggressor-counts must include 0,1,4; missing {missing}")
    return counts


def parse_occupancy_routes(raw: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("--aggressor-routes must list at least one integer")
    routes: list[int] = []
    for part in parts:
        try:
            value = int(part, 10)
        except ValueError as exc:
            raise ValueError(f"aggressor routes {part!r} is not an integer") from exc
        if value <= 0:
            raise ValueError(f"aggressor routes {value} must be positive")
        routes.append(value)
    unique = tuple(sorted(set(routes)))
    if len(unique) != len(routes):
        raise ValueError("--aggressor-routes must not contain duplicates")
    if unique != tuple(routes):
        raise ValueError("--aggressor-routes must be strictly increasing")
    missing = [value for value in ANCHOR_ROUTES if value not in unique]
    if missing:
        raise ValueError(f"--aggressor-routes must include anchors 1 and 68; missing {missing}")
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytic-calibration", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--threads", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--weight-copies", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument(
        "--aggressor-counts",
        default=",".join(str(count) for count in DEFAULT_AGGRESSOR_COUNTS),
        help="strictly increasing counts including 0,1,4",
    )
    parser.add_argument(
        "--aggressor-routes",
        default=",".join(str(routes) for routes in DEFAULT_AGGRESSOR_ROUTES),
        help="strictly increasing aggressor M including anchors 1 and 68",
    )
    parser.add_argument("--trace-dir", type=Path, default=Path("/tmp/moe_aggressor_m_occupancy"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="score analytic spans on the occupancy DAGs without running the kernel",
    )
    return parser.parse_args()


def occupancy_modes(counts: tuple[int, ...]) -> tuple[str, ...]:
    return enumerate_probe_modes(counts)


def cell_name(routes: int, mode: str) -> str:
    return f"m{routes}/{mode}"


def parse_cell_name(name: str) -> tuple[int, str]:
    prefix, mode = name.split("/", 1)
    if not prefix.startswith("m"):
        raise ValueError(f"occupancy cell {name!r} must start with m<routes>/")
    try:
        routes = int(prefix[1:], 10)
    except ValueError as exc:
        raise ValueError(f"occupancy cell {name!r} has a non-integer route count") from exc
    return routes, mode


def _delta_ms(comparisons: dict[str, dict[str, object]], key: str) -> float:
    return float(comparisons[key]["delta"]["median_ms"])


def _mode_delta(
    comparisons_by_routes: dict[int, dict[str, dict[str, object]]],
    routes: int,
    placement: str,
    phase: str,
    count: int,
) -> float:
    key = f"{mode_name(placement, phase, count)}_vs_isolated"
    return _delta_ms(comparisons_by_routes[routes], key)


def _overlap_ok(mode_calls: dict[str, list[float]], count: int) -> bool:
    if count <= 0:
        return True
    experts = mode_calls.get("peer_overlap_experts")
    if not experts:
        return False
    return float(_stats(experts)["median_ms"]) >= OVERLAP_EXPERT_FRACTION * count


def decide_occupancy(
    *,
    comparisons_by_routes: dict[int, dict[str, dict[str, object]]],
    calls_by_cell: dict[str, dict[str, list[float]]],
    routes: tuple[int, ...],
) -> dict[str, object]:
    same_n1 = {m: _mode_delta(comparisons_by_routes, m, *HEAD_N1) for m in routes}
    same_n4 = {m: _mode_delta(comparisons_by_routes, m, *HEAD_N4) for m in routes}
    cross_n4 = {m: _mode_delta(comparisons_by_routes, m, *CROSS_HEAD_N4) for m in routes}
    contrast_n4 = {m: same_n4[m] - cross_n4[m] for m in routes}
    low, high = ANCHOR_ROUTES
    n4_growth = same_n4[high] - same_n4[low]
    n1_growth = same_n1[high] - same_n1[low]
    mid = 16 if 16 in routes else None
    late_growth = same_n4[high] - same_n4[mid] if mid is not None else n4_growth
    occupancy_stable = abs(n4_growth) <= OCCUPANCY_STABLE_MS and abs(n1_growth) <= OCCUPANCY_STABLE_MS
    duration_occupancy = (
        mid is not None
        and same_n4[low] <= REMOTE_NEAR_ZERO_MS
        and same_n4[high] >= CONTRAST_PRESENT_MS
        and abs(late_growth) <= OCCUPANCY_STABLE_MS
    )
    utilization_growth = n4_growth >= UTILIZATION_GROWTH_MS and abs(late_growth) >= UTILIZATION_GROWTH_MS
    remote_near_zero = all(abs(cross_n4[m]) <= REMOTE_NEAR_ZERO_MS for m in routes)
    contrast_present = all(contrast_n4[m] >= CONTRAST_PRESENT_MS for m in routes)
    overlap_ok = all(
        _overlap_ok(calls_by_cell[cell_name(m, mode_name("same_llc", "head", count))], count)
        for m in routes
        for count in (1, 4)
    )
    if not overlap_ok:
        signature = "invalid_overlap"
    elif occupancy_stable:
        signature = "occupancy"
    elif duration_occupancy:
        signature = "duration_occupancy"
    elif utilization_growth:
        signature = "utilization"
    else:
        signature = "inconclusive"
    return {
        "add_default_off_structure": False,
        "signature": signature,
        "overlap_ok": overlap_ok,
        "occupancy_stable": occupancy_stable,
        "utilization_growth": utilization_growth,
        "duration_occupancy": duration_occupancy,
        "remote_near_zero": remote_near_zero,
        "contrast_present": contrast_present,
        "same_llc_head_n1_ms": same_n1,
        "same_llc_head_n4_ms": same_n4,
        "cross_llc_head_n4_ms": cross_n4,
        "n4_growth_m68_minus_m1_ms": n4_growth,
        "n1_growth_m68_minus_m1_ms": n1_growth,
        "n4_growth_m68_minus_m16_ms": late_growth,
        "gates": {
            "occupancy_stable_ms": OCCUPANCY_STABLE_MS,
            "utilization_growth_ms": UTILIZATION_GROWTH_MS,
            "remote_near_zero_ms": REMOTE_NEAR_ZERO_MS,
            "contrast_present_ms": CONTRAST_PRESENT_MS,
        },
        "reason": (
            "Do not add a structure from this M sweep. Occupancy means the leftover "
            "same-LLC tax is stable from M=1 to M=68. Utilization means it grows with "
            "aggressor routes. Duration occupancy means the tax appears only after the "
            "peer is long enough to cover victim W13."
        ),
    }


def _cost_model(calibration: Path, hidden: int, intermediate: int) -> AnalyticMoeCostModel:
    expert_count = 2 + BACKGROUND_EXPERTS
    return AnalyticMoeCostModel(
        calibration,
        hidden_size=hidden,
        intermediate_size=intermediate,
        global_experts=expert_count,
        local_experts=expert_count,
        mode="tp",
        degree=4,
        concurrent_ranks=1,
        down_output_element_bytes=4,
    )


def predict_occupancy_spans(
    model: AnalyticMoeCostModel,
    *,
    thread_cpu_ids: tuple[int, ...],
    counts: tuple[int, ...],
    route_values: tuple[int, ...],
) -> dict[str, float]:
    spans: dict[str, float] = {}
    for routes in route_values:
        for mode in occupancy_modes(counts):
            bridge, route_map = _build_bridge(
                model,
                thread_cpu_ids=thread_cpu_ids,
                mode=mode,
                background_routes=routes,
            )
            spans[cell_name(routes, mode)] = _model_target_span_ms(model, bridge, route_map)
    return spans


def _comparisons_from_predicted(spans: dict[str, float], route_values: tuple[int, ...]) -> dict[int, dict[str, dict[str, object]]]:
    report: dict[int, dict[str, dict[str, object]]] = {}
    for routes in route_values:
        prefix = f"m{routes}/"
        group = {
            mode[len(prefix) :]: [spans[mode]]
            for mode in spans
            if mode.startswith(prefix)
        }
        report[routes] = _comparisons(group)
    return report


def _empty_overlap_calls(counts: tuple[int, ...], route_values: tuple[int, ...]) -> dict[str, dict[str, list[float]]]:
    calls: dict[str, dict[str, list[float]]] = {}
    for routes in route_values:
        for mode in occupancy_modes(counts):
            _placement, _phase, count = parse_mode(mode)
            calls[cell_name(routes, mode)] = {
                "peer_overlap_experts": [float(count)],
            }
    return calls


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    counts = parse_occupancy_counts(args.aggressor_counts)
    route_values = parse_occupancy_routes(args.aggressor_routes)
    modes = occupancy_modes(counts)
    if min(args.hidden, args.intermediate, args.runs, args.weight_copies) <= 0 or args.warmup < 0:
        raise ValueError("dimensions/runs/copies must be positive and warmup non-negative")
    model = _cost_model(args.analytic_calibration, args.hidden, args.intermediate)
    if args.predict_only:
        predicted = predict_occupancy_spans(
            model,
            thread_cpu_ids=RANK_CPU_IDS,
            counts=counts,
            route_values=route_values,
        )
        comparisons_by_routes = _comparisons_from_predicted(predicted, route_values)
        decision = decide_occupancy(
            comparisons_by_routes=comparisons_by_routes,
            calls_by_cell=_empty_overlap_calls(counts, route_values),
            routes=route_values,
        )
        result = {
            "kind": OUTPUT_KIND,
            "schema_version": SCHEMA_VERSION,
            "predict_only": True,
            "identity": {
                "calibration": str(args.analytic_calibration),
                "calibration_sha256": _sha256(args.analytic_calibration),
            },
            "shape": {
                "target_routes": TARGET_ROUTES,
                "background_experts": BACKGROUND_EXPERTS,
                "aggressor_counts": list(counts),
                "aggressor_routes": list(route_values),
            },
            "predicted_target_span_ms": predicted,
            "comparisons_by_aggressor_routes": {
                str(routes): payload for routes, payload in comparisons_by_routes.items()
            },
            "decision": decision,
        }
        rendered = json.dumps(result, indent=2, sort_keys=True)
        print(rendered)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return 0

    if args.threads != 80:
        raise ValueError("the Arm occupancy probe requires exactly 80 threads")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < args.threads:
        raise ValueError("process affinity does not expose 80 CPUs")
    cpu_ids = tuple(affinity[: args.threads])
    torch.set_num_threads(1)
    expert_count = 2 + BACKGROUND_EXPERTS
    groups: dict[int, dict[str, object]] = {}
    for routes in route_values:
        built = {
            mode: _build_bridge(
                model,
                thread_cpu_ids=cpu_ids,
                mode=mode,
                background_routes=routes,
            )
            for mode in modes
        }
        route_map = built[modes[0]][1]
        if any(built[mode][1] != route_map for mode in modes):
            raise RuntimeError("modes at one aggressor M must keep the same expert route histogram")
        if any(route_map[expert] != routes for expert in range(2, expert_count)):
            raise RuntimeError("background experts must use the requested aggressor M")
        topk_ids = torch.repeat_interleave(
            torch.arange(expert_count, dtype=torch.int32),
            torch.tensor([route_map[expert] for expert in range(expert_count)], dtype=torch.int64),
        ).reshape(-1, 1)
        generator = torch.Generator().manual_seed(args.seed ^ routes)
        hidden = torch.empty((topk_ids.shape[0], args.hidden), dtype=torch.bfloat16)
        hidden.normal_(mean=0.0, std=0.01, generator=generator)
        topk_weights = torch.ones((topk_ids.shape[0], 1), dtype=torch.float32)
        groups[routes] = {
            "plans": {mode: AsyncMoEPlanV2.from_dict(value[0]) for mode, value in built.items()},
            "route_map": route_map,
            "hidden": hidden,
            "topk_ids": topk_ids,
            "topk_weights": topk_weights,
            "outputs": {mode: torch.empty_like(hidden) for mode in modes},
        }
    weight_generator = torch.Generator().manual_seed(args.seed)
    w13 = torch.empty((expert_count, 2 * args.intermediate, args.hidden), dtype=torch.bfloat16)
    w13.normal_(mean=0.0, std=0.01, generator=weight_generator)
    w2 = torch.empty((expert_count, args.hidden, args.intermediate), dtype=torch.bfloat16)
    w2.normal_(mean=0.0, std=0.01, generator=weight_generator)
    packed = [
        prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="arm_sve_bf16")
        for _ in range(args.weight_copies + 1)
    ]
    scrub_copy = args.weight_copies
    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "0"
    os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE"] = "0"
    scrub_routes = max(route_values)
    scrub_mode = _scrub_mode(modes)
    jobs = tuple((routes, mode) for routes in route_values for mode in modes)

    def run(routes: int, mode: str, copy_index: int) -> torch.Tensor:
        group = groups[routes]
        return fused_moe_bf16_tiled_async_plan(
            group["hidden"],
            packed[copy_index],
            group["topk_weights"],
            group["topk_ids"],
            group["plans"][mode],
            global_num_experts=expert_count,
            out=group["outputs"][mode],
        )

    for routes in route_values:
        reference = run(routes, modes[0], 0).clone()
        for mode in modes:
            torch.testing.assert_close(run(routes, mode, 0).float(), reference.float(), atol=0, rtol=0)

    warmup_order = random.Random(args.seed ^ 0xA5A5)
    for round_index in range(args.warmup):
        names = list(jobs)
        warmup_order.shuffle(names)
        for position, (routes, mode) in enumerate(names):
            run(scrub_routes, scrub_mode, scrub_copy)
            run(routes, mode, (round_index + position) % args.weight_copies)

    trace_paths = {
        cell_name(routes, mode): args.trace_dir / f"m{routes}_{mode}.log" for routes, mode in jobs
    }
    for trace_path in trace_paths.values():
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.unlink(missing_ok=True)
    trace_order = random.Random(args.seed ^ 0x5A5A)
    for run_index in range(args.runs):
        names = list(jobs)
        trace_order.shuffle(names)
        for position, (routes, mode) in enumerate(names):
            os.environ["FUSED_CPP_MOE_TRACE"] = "0"
            run(scrub_routes, scrub_mode, scrub_copy)
            os.environ["FUSED_CPP_MOE_TRACE_FILE"] = str(trace_paths[cell_name(routes, mode)])
            os.environ["FUSED_CPP_MOE_TRACE"] = "1"
            run(routes, mode, (run_index + position) % args.weight_copies)
            os.environ["FUSED_CPP_MOE_TRACE"] = "0"

    calls = {name: _parse_calls(path) for name, path in trace_paths.items()}
    comparisons_by_routes = {}
    for routes in route_values:
        prefix = f"m{routes}/"
        spans = {
            name[len(prefix) :]: values["target_span"]
            for name, values in calls.items()
            if name.startswith(prefix)
        }
        comparisons_by_routes[routes] = _comparisons(spans)
    decision = decide_occupancy(
        comparisons_by_routes=comparisons_by_routes,
        calls_by_cell=calls,
        routes=route_values,
    )
    result = {
        "kind": OUTPUT_KIND,
        "schema_version": SCHEMA_VERSION,
        "predict_only": False,
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
            "locked_count_sweep_background_routes": BACKGROUND_ROUTES,
            "aggressor_counts": list(counts),
            "aggressor_routes": list(route_values),
        },
        "cpu_mapping": _cpu_mapping(cpu_ids),
        "method": {
            "warmup": args.warmup,
            "trace_runs": args.runs,
            "trace_order": "randomized_paired_rounds_across_m_and_placement",
            "weight_copies": args.weight_copies,
            "scrub_policy": "dedicated_disjoint_max_m_isolated_or_first_mode_packed_copy_before_every_sample",
            "scrub_routes": scrub_routes,
            "scrub_mode": scrub_mode,
            "same_tasks_routes_weights_across_modes_at_fixed_m": True,
            "unused_aggressors": "dependency_delayed_after_target_lane_tail",
            "metric": "target first-phase start to final-phase end",
        },
        "modes": {
            name: {
                "aggressor_routes": parse_cell_name(name)[0],
                "placement": parse_mode(parse_cell_name(name)[1])[0],
                "target_phase": parse_mode(parse_cell_name(name)[1])[1],
                "aggressor_count": parse_mode(parse_cell_name(name)[1])[2],
                **{key: _stats(values) for key, values in mode_calls.items()},
            }
            for name, mode_calls in calls.items()
        },
        "comparisons_by_aggressor_routes": {
            str(routes): payload for routes, payload in comparisons_by_routes.items()
        },
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
