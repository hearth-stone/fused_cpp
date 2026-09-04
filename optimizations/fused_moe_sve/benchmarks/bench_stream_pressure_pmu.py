#!/usr/bin/env python3
"""Measure one stream-pressure cell inside a perf-controlled counter window.

This Lab probe intentionally omits dependency-delayed background tasks. Every
task in the measured plan overlaps the one-route victim at lane head, so L3C
and DDRC counts are not diluted by work that starts after the victim finishes.
Run one mode per process and use perf's FIFO control to exclude initialization,
warmup, and the disjoint packed-weight scrub from the counter window.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

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
    _sha256,
    _stats,
)
from optimizations.fused_moe_sve.benchmarks.bench_stream_count_composition import (  # noqa: E402
    SHEET16,
    TARGET_CORE_BEGIN,
    live_teams as _composition_live_teams,
    parse_mode,
    place_widths,
)


OUTPUT_KIND = "moe_stream_pressure_pmu_cell"
SCHEMA_VERSION = 1
PMU_MODES = (
    "isolated_head",
    "wide16_same_head",
    "two_8t_same_head",
    "four_4t_same_head",
    "four_2t_same_head",
    "eight_2t_same_head",
    "eight_1t_same_head",
    "grid4_1t_same_head",
    "grid4_2t_same_head",
    "grid6_1t_same_head",
    "grid6_2t_same_head",
    "grid8_1t_same_head",
    "grid8_2t_same_head",
    "four_1t_same_head",
    "many16_1t_same_head",
)
EXPERT_COUNT = 18
SCRUB_CORES = tuple(range(48, 66))


def pmu_live_teams(mode: str) -> list[tuple[int, int]]:
    grid = {
        "grid4_1t_same_head": (4, 1),
        "grid4_2t_same_head": (4, 2),
        "grid6_1t_same_head": (6, 1),
        "grid6_2t_same_head": (6, 2),
        "grid8_1t_same_head": (8, 1),
        "grid8_2t_same_head": (8, 2),
    }
    if mode in grid:
        count, width = grid[mode]
        return [(48 + 2 * index, width) for index in range(count)]
    if mode == "four_2t_same_head":
        return [(core, 2) for core in (48, 52, 56, 60)]
    if mode == "eight_2t_same_head":
        return place_widths(SHEET16, (2,) * 8)
    if mode == "eight_1t_same_head":
        return [(core, 1) for core in range(48, 64, 2)]
    return _composition_live_teams(mode)


def pmu_kind_name(mode: str) -> str:
    added = {
        "four_2t_same_head": "four_2t",
        "eight_2t_same_head": "eight_2t",
        "eight_1t_same_head": "eight_1t",
        "grid4_1t_same_head": "grid4_1t",
        "grid4_2t_same_head": "grid4_2t",
        "grid6_1t_same_head": "grid6_1t",
        "grid6_2t_same_head": "grid6_2t",
        "grid8_1t_same_head": "grid8_1t",
        "grid8_2t_same_head": "grid8_2t",
    }
    if mode in added:
        return added[mode]
    return parse_mode(mode)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytic-calibration", type=Path, required=True)
    parser.add_argument("--mode", choices=PMU_MODES, required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--threads", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--weight-copies", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--trace", type=Path, default=Path("/tmp/moe_stream_pressure_pmu.log"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--perf-control", type=Path, action="append", default=[])
    parser.add_argument("--perf-ack", type=Path, action="append", default=[])
    return parser.parse_args()


def _bridge(
    model: AnalyticMoeCostModel,
    *,
    thread_cpu_ids: tuple[int, ...],
    mode: str,
) -> tuple[dict[str, object], dict[int, int]]:
    teams = pmu_live_teams(mode)
    tasks = [(TARGET_EXPERT, TARGET_ROUTES, TARGET_CORE_BEGIN, 1)]
    tasks.extend((2 + index, 1, core, width) for index, (core, width) in enumerate(teams))
    policy = model.shadow_stage_window_policy()
    widths = [width for _expert, _routes, _core, width in tasks]
    windows = [policy.select(routes, width) for _expert, routes, _core, width in tasks]
    bridge = {
        "plan_version": 2,
        "execution_mode": "strict",
        "num_threads": len(thread_cpu_ids),
        "thread_cpu_ids": list(thread_cpu_ids),
        "task_expert_ids": [expert for expert, _routes, _core, _width in tasks],
        "task_core_begins": [core for _expert, _routes, core, _width in tasks],
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
    return bridge, {expert: routes for expert, routes, _core, _width in tasks}


def _scrub_bridge(
    model: AnalyticMoeCostModel,
    *,
    thread_cpu_ids: tuple[int, ...],
) -> tuple[dict[str, object], dict[int, int]]:
    policy = model.shadow_stage_window_policy()
    widths = [1] * EXPERT_COUNT
    windows = [policy.select(1, 1) for _ in range(EXPERT_COUNT)]
    bridge = {
        "plan_version": 2,
        "execution_mode": "strict",
        "num_threads": len(thread_cpu_ids),
        "thread_cpu_ids": list(thread_cpu_ids),
        "task_expert_ids": list(range(EXPERT_COUNT)),
        "task_core_begins": list(SCRUB_CORES),
        "task_threads": widths,
        "task_dep_offsets": [0] * (EXPERT_COUNT + 1),
        "task_deps": [],
        "task_preferred_threads": widths,
        "task_min_threads": widths,
        "task_max_threads": widths,
        "task_allowed_thread_offsets": list(range(EXPERT_COUNT + 1)),
        "task_allowed_threads": widths,
        "task_placement_modes": [0] * EXPERT_COUNT,
        "task_numa_nodes": [-1] * EXPERT_COUNT,
        "task_stage_ids": [0] * EXPERT_COUNT,
        "task_resize_points": [0] * EXPERT_COUNT,
        "task_range_granularities": [0] * EXPERT_COUNT,
        "task_w13_window_tiles": [window[0] for window in windows],
        "task_w2_window_tiles": [window[1] for window in windows],
        "early_merge": False,
    }
    return bridge, {expert: 1 for expert in range(EXPERT_COUNT)}


class PerfControl:
    """Send enable/disable commands to a perf stat FIFO control channel."""

    def __init__(self, control_paths: list[Path], ack_paths: list[Path]) -> None:
        if len(control_paths) != len(ack_paths):
            raise ValueError("--perf-control and --perf-ack counts must match")
        self._control_paths = control_paths
        self._ack_paths = ack_paths
        self._controls: list[BinaryIO] = []
        self._acks: list[BinaryIO] = []

    def __enter__(self) -> PerfControl:
        for control_path, ack_path in zip(self._control_paths, self._ack_paths, strict=True):
            self._controls.append(control_path.open("wb", buffering=0))
            self._acks.append(ack_path.open("rb", buffering=0))
        return self

    def command(self, value: str) -> None:
        for control in self._controls:
            control.write(f"{value}\n".encode())
        for ack in self._acks:
            acknowledgement = ack.readline().decode().strip("\x00\r\n ")
            if acknowledgement != "ack":
                raise RuntimeError(f"perf did not acknowledge {value!r}: {acknowledgement!r}")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        for ack in self._acks:
            ack.close()
        for control in self._controls:
            control.close()


def _ids(route_map: dict[int, int]) -> torch.Tensor:
    values = [expert for expert, routes in sorted(route_map.items()) for _ in range(routes)]
    return torch.tensor(values, dtype=torch.int32).reshape(-1, 1)


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if args.threads != 80:
        raise ValueError("the Arm PMU probe requires exactly 80 threads")
    if min(args.hidden, args.intermediate, args.runs, args.weight_copies) <= 0 or args.warmup < 0:
        raise ValueError("dimensions/runs/copies must be positive and warmup non-negative")
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
    bridge, route_map = _bridge(model, thread_cpu_ids=cpu_ids, mode=args.mode)
    scrub_bridge, scrub_routes = _scrub_bridge(model, thread_cpu_ids=cpu_ids)
    plan = AsyncMoEPlanV2.from_dict(bridge)
    scrub_plan = AsyncMoEPlanV2.from_dict(scrub_bridge)
    generator = torch.Generator().manual_seed(args.seed)
    hidden = torch.empty((_ids(route_map).shape[0], args.hidden), dtype=torch.bfloat16)
    hidden.normal_(mean=0.0, std=0.01, generator=generator)
    topk_ids = _ids(route_map)
    topk_weights = torch.ones((topk_ids.shape[0], 1), dtype=torch.float32)
    scrub_hidden = torch.empty((EXPERT_COUNT, args.hidden), dtype=torch.bfloat16)
    scrub_hidden.normal_(mean=0.0, std=0.01, generator=generator)
    scrub_ids = _ids(scrub_routes)
    scrub_weights = torch.ones((EXPERT_COUNT, 1), dtype=torch.float32)
    w13 = torch.empty((EXPERT_COUNT, 2 * args.intermediate, args.hidden), dtype=torch.bfloat16)
    w13.normal_(mean=0.0, std=0.01, generator=generator)
    w2 = torch.empty((EXPERT_COUNT, args.hidden, args.intermediate), dtype=torch.bfloat16)
    w2.normal_(mean=0.0, std=0.01, generator=generator)
    packed = [
        prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="arm_sve_bf16")
        for _ in range(args.weight_copies + 1)
    ]
    output = torch.empty_like(hidden)
    scrub_output = torch.empty_like(scrub_hidden)
    scrub_copy = args.weight_copies
    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "0"
    os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE"] = "0"

    def run(copy_index: int) -> torch.Tensor:
        return fused_moe_bf16_tiled_async_plan(
            hidden,
            packed[copy_index],
            topk_weights,
            topk_ids,
            plan,
            global_num_experts=EXPERT_COUNT,
            out=output,
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

    reference = run(0).clone()
    torch.testing.assert_close(run(0).float(), reference.float(), atol=0, rtol=0)
    for warmup_index in range(args.warmup):
        scrub()
        run(warmup_index % args.weight_copies)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.trace.unlink(missing_ok=True)
    os.environ["FUSED_CPP_MOE_TRACE_FILE"] = str(args.trace)
    order = list(range(args.runs))
    random.Random(args.seed ^ 0x5A5A).shuffle(order)
    wall_ns: list[float] = []
    with PerfControl(args.perf_control, args.perf_ack) as perf:
        for run_index in order:
            scrub()
            os.environ["FUSED_CPP_MOE_TRACE"] = "1"
            perf.command("enable")
            start_ns = time.perf_counter_ns()
            run(run_index % args.weight_copies)
            wall_ns.append(float(time.perf_counter_ns() - start_ns))
            perf.command("disable")
            os.environ["FUSED_CPP_MOE_TRACE"] = "0"
    calls = _parse_calls(args.trace)
    result = {
        "kind": OUTPUT_KIND,
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "calibration": str(args.analytic_calibration),
            "calibration_sha256": _sha256(args.analytic_calibration),
            "extension": str(_moe_C.__file__),
            "extension_sha256": _sha256(Path(_moe_C.__file__)),
        },
        "mode": args.mode,
        "kind_name": pmu_kind_name(args.mode),
        "live_teams": pmu_live_teams(args.mode),
        "live_streams": len(pmu_live_teams(args.mode)),
        "live_threads": sum(width for _core, width in pmu_live_teams(args.mode)),
        "cpu_mapping": _cpu_mapping(cpu_ids),
        "method": {
            "warmup": args.warmup,
            "runs": args.runs,
            "weight_copies": args.weight_copies,
            "counter_window": "measured overlapping cell only",
            "perf_control_channels": len(args.perf_control),
            "scrub_policy": "disjoint packed copy touching all 18 experts before every sample",
            "delayed_background_tasks": False,
        },
        "wall_ms": _stats([value / 1e6 for value in wall_ns]),
        "trace": {key: _stats(values) for key, values in calls.items()},
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
