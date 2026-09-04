#!/usr/bin/env python3
"""Sweep local/remote/split gather aggressor count for absolute memory pressure."""

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
    _paired_delta_stats,
    _sha256,
    _stats,
)


OUTPUT_KIND = "moe_gather_absolute_pressure_probe"
SCHEMA_VERSION = 1
DEFAULT_AGGRESSOR_COUNTS = (0, 1, 2, 4, 8, 15)
SPLIT_AGGRESSOR_COUNTS = frozenset({8, 15})
PLACEMENTS = ("same_llc", "cross_llc", "split", "isolated")
PHASES = ("head", "after_1")
TARGET_EXPERT = 0
DELAY_EXPERT = 1
TARGET_ROUTES = 1
DELAY_ROUTES = 1
BACKGROUND_ROUTES = 68
BACKGROUND_EXPERTS = 15
TARGET_CORE_BEGIN = 64
SAME_LLC_CORES = tuple(range(65, 80))
CROSS_LLC_CORES = tuple(range(0, 15))
LOCAL_BACKGROUND_BEGIN = SAME_LLC_CORES[0]
REMOTE_BACKGROUND_BEGIN = CROSS_LLC_CORES[0]
LEGACY_MODES = (
    "local_head",
    "remote_head",
    "local_after_1",
    "remote_after_1",
    "isolated_head",
    "isolated_after_1",
)
LEGACY_MODE_SPECS = {
    "local_head": ("same_llc", "head", 15),
    "remote_head": ("cross_llc", "head", 15),
    "local_after_1": ("same_llc", "after_1", 15),
    "remote_after_1": ("cross_llc", "after_1", 15),
    "isolated_head": ("isolated", "head", 0),
    "isolated_after_1": ("isolated", "after_1", 0),
}


def parse_aggressor_counts(raw: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("--aggressor-counts must list at least one integer")
    counts: list[int] = []
    for part in parts:
        try:
            value = int(part, 10)
        except ValueError as exc:
            raise ValueError(f"aggressor count {part!r} is not an integer") from exc
        if value < 0 or value > BACKGROUND_EXPERTS:
            raise ValueError(f"aggressor count {value} is outside 0..{BACKGROUND_EXPERTS}")
        counts.append(value)
    unique = tuple(sorted(set(counts)))
    if len(unique) != len(counts):
        raise ValueError("--aggressor-counts must not contain duplicates")
    if unique != tuple(counts):
        raise ValueError("--aggressor-counts must be strictly increasing")
    if unique[0] != 0:
        raise ValueError("--aggressor-counts must include 0 as the matched isolated control")
    return unique


def mode_name(placement: str, phase: str, count: int) -> str:
    if placement not in PLACEMENTS or phase not in PHASES:
        raise ValueError(f"unsupported placement/phase {placement!r}/{phase!r}")
    if placement == "isolated":
        if count != 0:
            raise ValueError("isolated modes are only defined for aggressor_count=0")
        return f"isolated_{phase}"
    if count <= 0:
        raise ValueError(f"{placement} modes require a positive aggressor count")
    if placement == "split" and count not in SPLIT_AGGRESSOR_COUNTS:
        raise ValueError(f"split placement is only defined for {sorted(SPLIT_AGGRESSOR_COUNTS)}")
    return f"{placement}_{phase}_n{count}"


def enumerate_probe_modes(counts: tuple[int, ...] = DEFAULT_AGGRESSOR_COUNTS) -> tuple[str, ...]:
    modes: list[str] = []
    for count in counts:
        for phase in PHASES:
            if count == 0:
                modes.append(mode_name("isolated", phase, 0))
                continue
            modes.append(mode_name("same_llc", phase, count))
            modes.append(mode_name("cross_llc", phase, count))
            if count in SPLIT_AGGRESSOR_COUNTS:
                modes.append(mode_name("split", phase, count))
    return tuple(modes)


MODES = enumerate_probe_modes(DEFAULT_AGGRESSOR_COUNTS)


def parse_mode(mode: str) -> tuple[str, str, int]:
    if mode in LEGACY_MODE_SPECS:
        return LEGACY_MODE_SPECS[mode]
    for placement in ("same_llc", "cross_llc", "split"):
        for phase in ("after_1", "head"):
            prefix = f"{placement}_{phase}_n"
            if mode.startswith(prefix):
                try:
                    count = int(mode[len(prefix) :], 10)
                except ValueError as exc:
                    raise ValueError(f"unsupported probe mode {mode!r}") from exc
                expected = mode_name(placement, phase, count)
                if mode != expected:
                    raise ValueError(f"mode {mode!r} is not canonical; expected {expected!r}")
                return (placement, phase, count)
    raise ValueError(f"unsupported probe mode {mode!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytic-calibration", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--threads", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--weight-copies", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--aggressor-counts",
        default=",".join(str(count) for count in DEFAULT_AGGRESSOR_COUNTS),
        help="strictly increasing counts including 0, for example 0,1,2,4,8,15",
    )
    parser.add_argument("--trace-dir", type=Path, default=Path("/tmp/moe_gather_absolute_pressure"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _background_cores(placement: str, count: int) -> tuple[int, ...]:
    if placement in ("same_llc", "isolated"):
        return SAME_LLC_CORES
    if placement == "cross_llc":
        return CROSS_LLC_CORES
    if placement != "split":
        raise ValueError(f"unsupported background placement {placement!r}")
    if count not in SPLIT_AGGRESSOR_COUNTS:
        raise ValueError(f"split placement is only defined for {sorted(SPLIT_AGGRESSOR_COUNTS)}")
    remote_n = count // 2
    local_n = count - remote_n
    assigned = list(CROSS_LLC_CORES[:remote_n]) + list(SAME_LLC_CORES[:local_n])
    used = set(assigned)
    leftovers = [cpu for cpu in (*CROSS_LLC_CORES, *SAME_LLC_CORES) if cpu not in used]
    assigned.extend(leftovers[: BACKGROUND_EXPERTS - count])
    if len(assigned) != BACKGROUND_EXPERTS:
        raise ValueError("split placement must assign every background expert a core")
    return tuple(assigned)


def _task_specs(
    mode: str,
    *,
    background_routes: int = BACKGROUND_ROUTES,
) -> list[tuple[int, int, int, int, str]]:
    if background_routes <= 0:
        raise ValueError("background_routes must be positive")
    placement, phase, _count = parse_mode(mode)
    target_lane = [
        (TARGET_EXPERT, TARGET_ROUTES, TARGET_CORE_BEGIN, 1, "target"),
        (DELAY_EXPERT, DELAY_ROUTES, TARGET_CORE_BEGIN, 1, "delay"),
    ]
    if phase == "after_1":
        target_lane.reverse()
    cores = _background_cores(placement, _count)
    return target_lane + [
        (2 + index, background_routes, cores[index], 1, "background")
        for index in range(BACKGROUND_EXPERTS)
    ]


def _build_bridge(
    model: AnalyticMoeCostModel,
    *,
    thread_cpu_ids: tuple[int, ...],
    mode: str,
    background_routes: int = BACKGROUND_ROUTES,
) -> tuple[dict[str, object], dict[int, int]]:
    _placement, _phase, count = parse_mode(mode)
    tasks = _task_specs(mode, background_routes=background_routes)
    dependencies: list[list[int]] = [[], [0]]
    dependencies.extend([] if index < count else [1] for index in range(BACKGROUND_EXPERTS))
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


def _parse_calls(path: Path) -> dict[str, list[float]]:
    calls: list[dict[str, float]] = []
    phases: list[dict[str, float | int | str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if not words:
            continue
        fields = dict(word.split("=", 1) for word in words[1:] if "=" in word)
        if words[0] == "MOE_CALL":
            phases = []
        elif words[0] == "PHASE":
            phases.append(
                {
                    "expert": int(fields.get("expert", -1)),
                    "stage": fields["stage"],
                    "start_ms": float(fields["start_ms"]),
                    "end_ms": float(fields["end_ms"]),
                    "ms": float(fields["ms"]),
                }
            )
        elif words[0] == "MOE_CALL_END":
            target = [phase for phase in phases if phase["expert"] == TARGET_EXPERT]
            if not target:
                raise RuntimeError(f"trace call in {path} has no target expert phases")
            target_start = min(float(phase["start_ms"]) for phase in target)
            target_end = max(float(phase["end_ms"]) for phase in target)
            row: dict[str, float] = {"target_span": target_end - target_start}
            stage_starts: dict[str, float] = {}
            stage_ends: dict[str, float] = {}
            for phase in target:
                stage = str(phase["stage"])
                start = float(phase["start_ms"])
                end = float(phase["end_ms"])
                stage_starts[stage] = start if stage not in stage_starts else min(stage_starts[stage], start)
                stage_ends[stage] = end if stage not in stage_ends else max(stage_ends[stage], end)
            for stage, start in stage_starts.items():
                row[f"target.{stage}"] = stage_ends[stage] - start
            overlapping_experts: set[int] = set()
            for phase in phases:
                expert = int(phase["expert"])
                if expert < 2:
                    continue
                begin = max(target_start, float(phase["start_ms"]))
                end = min(target_end, float(phase["end_ms"]))
                if end <= begin:
                    continue
                stage = str(phase["stage"])
                row[f"peer_overlap_core_ms.{stage}"] = (
                    row.get(f"peer_overlap_core_ms.{stage}", 0.0) + end - begin
                )
                overlapping_experts.add(expert)
                if float(phase["start_ms"]) <= target_start < float(phase["end_ms"]):
                    row[f"peer_active_at_target_start.{stage}"] = (
                        row.get(f"peer_active_at_target_start.{stage}", 0.0) + 1.0
                    )
            row["peer_overlap_experts"] = float(len(overlapping_experts))
            calls.append(row)
    if not calls:
        raise RuntimeError(f"no complete calls in {path}")
    keys = sorted({key for call in calls for key in call})
    return {key: [call.get(key, 0.0) for call in calls] for key in keys}


def _legacy_span_aliases(spans: dict[str, list[float]]) -> dict[str, list[float]]:
    unified = dict(spans)
    aliases = {
        "same_llc_head_n15": "local_head",
        "cross_llc_head_n15": "remote_head",
        "same_llc_after_1_n15": "local_after_1",
        "cross_llc_after_1_n15": "remote_after_1",
    }
    for canonical, legacy in aliases.items():
        if canonical in unified and legacy not in unified:
            unified[legacy] = unified[canonical]
        if legacy in unified and canonical not in unified:
            unified[canonical] = unified[legacy]
    return unified


def _comparisons(spans: dict[str, list[float]]) -> dict[str, object]:
    unified = _legacy_span_aliases(spans)
    report: dict[str, object] = {}
    for phase in PHASES:
        isolated = f"isolated_{phase}"
        if isolated not in unified:
            continue
        counts = sorted(
            {
                parse_mode(mode)[2]
                for mode in unified
                if parse_mode(mode)[1] == phase and parse_mode(mode)[2] > 0
            }
        )
        for count in counts:
            same = mode_name("same_llc", phase, count)
            cross = mode_name("cross_llc", phase, count)
            split = mode_name("split", phase, count) if count in SPLIT_AGGRESSOR_COUNTS else None
            if same in unified:
                report[f"{same}_vs_isolated"] = _paired_delta_stats(unified[same], unified[isolated])
            if cross in unified:
                report[f"{cross}_vs_isolated"] = _paired_delta_stats(unified[cross], unified[isolated])
            if split is not None and split in unified:
                report[f"{split}_vs_isolated"] = _paired_delta_stats(unified[split], unified[isolated])
            if same in unified and cross in unified:
                report[f"same_minus_cross_{phase}_n{count}"] = _paired_delta_stats(
                    unified[same], unified[cross]
                )
    if "local_head" in unified and "isolated_head" in unified:
        report["local_head_vs_isolated"] = _paired_delta_stats(
            unified["local_head"], unified["isolated_head"]
        )
        report["remote_head_vs_isolated"] = _paired_delta_stats(
            unified["remote_head"], unified["isolated_head"]
        )
        report["local_vs_remote_head"] = _paired_delta_stats(
            unified["local_head"], unified["remote_head"]
        )
    if "local_after_1" in unified and "isolated_after_1" in unified:
        report["local_after_1_vs_isolated"] = _paired_delta_stats(
            unified["local_after_1"], unified["isolated_after_1"]
        )
        report["remote_after_1_vs_isolated"] = _paired_delta_stats(
            unified["remote_after_1"], unified["isolated_after_1"]
        )
        report["local_vs_remote_after_1"] = _paired_delta_stats(
            unified["local_after_1"], unified["remote_after_1"]
        )
    return report


def _read_sysfs(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _parse_cpu_list(spec: str) -> tuple[int, ...]:
    cpus: list[int] = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            begin, end = token.split("-", 1)
            cpus.extend(range(int(begin, 10), int(end, 10) + 1))
        else:
            cpus.append(int(token, 10))
    return tuple(cpus)


def _llc_sysfs(cpu_dir: Path) -> tuple[int | None, str | None]:
    cache_root = cpu_dir / "cache"
    if not cache_root.is_dir():
        return None, None
    for index_dir in sorted(cache_root.glob("index*")):
        if _read_sysfs(index_dir / "level") != "3" or _read_sysfs(index_dir / "type") != "Unified":
            continue
        cache_id = _read_sysfs(index_dir / "id")
        shared = _read_sysfs(index_dir / "shared_cpu_list")
        domain = int(cache_id, 10) if cache_id not in (None, "") else None
        return domain, shared
    return None, None


def _llc_domain_index(sysfs_root: Path) -> dict[str, int]:
    shared_lists: set[str] = set()
    for cpu_dir in sysfs_root.glob("cpu[0-9]*"):
        if not cpu_dir.name[3:].isdigit():
            continue
        _domain, shared = _llc_sysfs(cpu_dir)
        if shared:
            shared_lists.add(shared)
    return {
        shared: index
        for index, shared in enumerate(sorted(shared_lists, key=lambda spec: _parse_cpu_list(spec)[0]))
    }


def _cpu_mapping(
    cpu_ids: tuple[int, ...],
    *,
    sysfs_root: Path = Path("/sys/devices/system/cpu"),
) -> list[dict[str, object]]:
    domain_index = _llc_domain_index(sysfs_root)
    rows: list[dict[str, object]] = []
    for logical_index, os_cpu in enumerate(cpu_ids):
        cpu_dir = sysfs_root / f"cpu{os_cpu}"
        numa_node: int | None = None
        for child in sorted(cpu_dir.glob("node[0-9]*")):
            numa_node = int(child.name[4:])
            break
        llc_id, llc_shared = _llc_sysfs(cpu_dir)
        if llc_id is None and llc_shared in domain_index:
            llc_id = domain_index[llc_shared]
        rows.append(
            {
                "logical_index": logical_index,
                "os_cpu": os_cpu,
                "physical_cpu": os_cpu,
                "numa_node": numa_node,
                "llc_domain": llc_id,
                "llc_shared_cpus": llc_shared,
            }
        )
    return rows


def _scrub_mode(modes: tuple[str, ...]) -> str:
    if "isolated_head" in modes:
        return "isolated_head"
    return modes[0]


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    counts = parse_aggressor_counts(args.aggressor_counts)
    modes = enumerate_probe_modes(counts)
    if args.threads != 80:
        raise ValueError("the Arm injection probe requires exactly 80 threads")
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
    built = {
        mode: _build_bridge(model, thread_cpu_ids=cpu_ids, mode=mode)
        for mode in modes
    }
    plans = {mode: AsyncMoEPlanV2.from_dict(value[0]) for mode, value in built.items()}
    route_map = built[modes[0]][1]
    if any(built[mode][1] != route_map for mode in modes):
        raise RuntimeError("probe modes must keep the same expert route histogram")
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
    outputs = {mode: torch.empty_like(hidden) for mode in modes}
    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "0"
    os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE"] = "0"
    scrub = _scrub_mode(modes)

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

    reference = run(modes[0], 0).clone()
    for mode in modes:
        torch.testing.assert_close(run(mode, 0).float(), reference.float(), atol=0, rtol=0)
    warmup_order = random.Random(args.seed ^ 0xA5A5)
    for round_index in range(args.warmup):
        names = list(modes)
        warmup_order.shuffle(names)
        for position, mode in enumerate(names):
            run(scrub, scrub_copy)
            run(mode, (round_index + position) % args.weight_copies)

    trace_paths = {mode: args.trace_dir / f"{mode}.log" for mode in modes}
    for trace_path in trace_paths.values():
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.unlink(missing_ok=True)
    trace_order = random.Random(args.seed ^ 0x5A5A)
    for run_index in range(args.runs):
        names = list(modes)
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
            "delay_routes": DELAY_ROUTES,
            "background_experts": BACKGROUND_EXPERTS,
            "background_routes": BACKGROUND_ROUTES,
            "target_core_begin": TARGET_CORE_BEGIN,
            "same_llc_background_cores": list(SAME_LLC_CORES),
            "cross_llc_background_cores": list(CROSS_LLC_CORES),
            "split_rule": "first_half_cross_llc6_then_remaining_same_llc7",
            "aggressor_counts": list(counts),
            "local_background_cores": [SAME_LLC_CORES[0], SAME_LLC_CORES[-1]],
            "remote_background_cores": [CROSS_LLC_CORES[0], CROSS_LLC_CORES[-1]],
        },
        "cpu_mapping": _cpu_mapping(cpu_ids),
        "method": {
            "warmup": args.warmup,
            "trace_runs": args.runs,
            "trace_order": "randomized_paired_rounds",
            "weight_copies": args.weight_copies,
            "scrub_policy": "dedicated_disjoint_isolated_or_first_mode_packed_copy_before_every_sample",
            "scrub_mode": scrub,
            "scrub_working_set_bytes": expert_count * 12 * 1024 * 1024,
            "same_tasks_routes_weights_across_modes": True,
            "unused_aggressors": "dependency_delayed_after_target_lane_tail",
            "metric": "target first-phase start to final-phase end",
            "target_stage_metric": "stage_envelope",
        },
        "modes": {
            mode: {
                "placement": parse_mode(mode)[0],
                "target_phase": parse_mode(mode)[1],
                "aggressor_count": parse_mode(mode)[2],
                **{key: _stats(values) for key, values in mode_calls.items()},
            }
            for mode, mode_calls in calls.items()
        },
        "comparisons": _comparisons(spans),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
