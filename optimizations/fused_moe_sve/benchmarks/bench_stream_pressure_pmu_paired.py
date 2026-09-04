#!/usr/bin/env python3
"""Measure randomized stream-pressure cells with per-cell perf counter reads."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
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
    _cpu_mapping,
    _parse_calls,
)
from optimizations.fused_moe_sve.benchmarks.bench_small_expert_context import (  # noqa: E402
    _sha256,
    _stats,
)
from optimizations.fused_moe_sve.benchmarks.bench_stream_pressure_pmu import (  # noqa: E402
    EXPERT_COUNT,
    PMU_MODES,
    _bridge,
    _ids,
    _scrub_bridge,
    pmu_kind_name,
    pmu_live_teams,
)
from optimizations.fused_moe_sve.benchmarks.linux_perf_event import (  # noqa: E402
    PerfCounterSet,
    stream_pressure_specs,
)


OUTPUT_KIND = "moe_stream_pressure_pmu_paired"
SCHEMA_VERSION = 1
DEFAULT_MODES = (
    "isolated_head",
    "wide16_same_head",
    "two_8t_same_head",
    "four_4t_same_head",
    "eight_2t_same_head",
    "many16_1t_same_head",
    "four_1t_same_head",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytic-calibration", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=PMU_MODES, default=list(DEFAULT_MODES))
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--threads", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--weight-copies", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--trace-dir", type=Path, default=Path("/tmp/moe_stream_pressure_pmu_paired"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def validate_modes(modes: list[str]) -> tuple[str, ...]:
    values = tuple(modes)
    if not values:
        raise ValueError("at least one mode is required")
    if len(set(values)) != len(values):
        raise ValueError("paired PMU modes must be unique")
    if "isolated_head" not in values:
        raise ValueError("paired PMU modes must include isolated_head")
    return values


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    modes = validate_modes(args.modes)
    if args.threads != 80:
        raise ValueError("the Arm paired PMU probe requires exactly 80 threads")
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
    built = {mode: _bridge(model, thread_cpu_ids=cpu_ids, mode=mode) for mode in modes}
    plans = {mode: AsyncMoEPlanV2.from_dict(bridge) for mode, (bridge, _routes) in built.items()}
    scrub_bridge, scrub_routes = _scrub_bridge(model, thread_cpu_ids=cpu_ids)
    scrub_plan = AsyncMoEPlanV2.from_dict(scrub_bridge)
    generator = torch.Generator().manual_seed(args.seed)
    inputs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for mode, (_bridge_value, route_map) in built.items():
        topk_ids = _ids(route_map)
        hidden = torch.empty((topk_ids.shape[0], args.hidden), dtype=torch.bfloat16)
        hidden.normal_(mean=0.0, std=0.01, generator=generator)
        inputs[mode] = (
            hidden,
            topk_ids,
            torch.ones((topk_ids.shape[0], 1), dtype=torch.float32),
            torch.empty_like(hidden),
        )
    scrub_hidden = torch.empty((EXPERT_COUNT, args.hidden), dtype=torch.bfloat16)
    scrub_hidden.normal_(mean=0.0, std=0.01, generator=generator)
    scrub_ids = _ids(scrub_routes)
    scrub_weights = torch.ones((EXPERT_COUNT, 1), dtype=torch.float32)
    scrub_output = torch.empty_like(scrub_hidden)
    w13 = torch.empty((EXPERT_COUNT, 2 * args.intermediate, args.hidden), dtype=torch.bfloat16)
    w13.normal_(mean=0.0, std=0.01, generator=generator)
    w2 = torch.empty((EXPERT_COUNT, args.hidden, args.intermediate), dtype=torch.bfloat16)
    w2.normal_(mean=0.0, std=0.01, generator=generator)
    packed = [
        prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="arm_sve_bf16")
        for _ in range(args.weight_copies + 1)
    ]
    scrub_copy = args.weight_copies
    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "0"
    os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE"] = "0"

    def run(mode: str, copy_index: int) -> torch.Tensor:
        hidden, topk_ids, topk_weights, output = inputs[mode]
        return fused_moe_bf16_tiled_async_plan(
            hidden,
            packed[copy_index],
            topk_weights,
            topk_ids,
            plans[mode],
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

    for mode in modes:
        reference = run(mode, 0).clone()
        torch.testing.assert_close(run(mode, 0).float(), reference.float(), atol=0, rtol=0)
    warmup_order = random.Random(args.seed ^ 0xA5A5)
    for round_index in range(args.warmup):
        names = list(modes)
        warmup_order.shuffle(names)
        for position, mode in enumerate(names):
            scrub()
            run(mode, (round_index + position) % args.weight_copies)

    trace_paths = {mode: args.trace_dir / f"{mode}.log" for mode in modes}
    for path in trace_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
    specs = stream_pressure_specs()
    pmu_samples: dict[str, dict[str, list[dict[str, int | float]]]] = {
        mode: {spec.name: [] for spec in specs} for mode in modes
    }
    wall_ms: dict[str, list[float]] = {mode: [] for mode in modes}
    measured_order: list[list[str]] = []
    order_rng = random.Random(args.seed ^ 0x5A5A)
    with PerfCounterSet(specs) as counters:
        os.sched_setaffinity(0, {cpu_ids[0]})
        for run_index in range(args.runs):
            names = list(modes)
            order_rng.shuffle(names)
            measured_order.append(names)
            for position, mode in enumerate(names):
                scrub()
                os.environ["FUSED_CPP_MOE_TRACE_FILE"] = str(trace_paths[mode])
                os.environ["FUSED_CPP_MOE_TRACE"] = "1"
                counters.reset_enable()
                start_ns = time.perf_counter_ns()
                try:
                    run(mode, (run_index + position) % args.weight_copies)
                finally:
                    values = counters.disable_read()
                wall_ms[mode].append((time.perf_counter_ns() - start_ns) / 1e6)
                os.environ["FUSED_CPP_MOE_TRACE"] = "0"
                for name, value in values.items():
                    pmu_samples[mode][name].append(
                        {
                            "count": value.count,
                            "time_enabled_ns": value.time_enabled_ns,
                            "time_running_ns": value.time_running_ns,
                            "running_ratio": value.running_ratio,
                        }
                    )
    calls = {mode: _parse_calls(path) for mode, path in trace_paths.items()}
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
            "experts": EXPERT_COUNT,
            "threads": args.threads,
            "control_cpu": cpu_ids[0],
            "victim_cpu": 304,
        },
        "cpu_mapping": _cpu_mapping(cpu_ids),
        "method": {
            "warmup": args.warmup,
            "runs": args.runs,
            "weight_copies": args.weight_copies,
            "same_process": True,
            "same_packed_allocations": True,
            "randomized_paired_rounds": True,
            "per_cell_counter_reset_read": True,
            "scrub_policy": "disjoint packed copy touching all 18 experts before every sample",
            "control_thread_pinned_off_victim": True,
        },
        "event_specs": [
            {
                "name": spec.name,
                "pmu_type": spec.pmu_type,
                "config": spec.config,
                "cpu": spec.cpu,
                "exclude_guest": spec.exclude_guest,
            }
            for spec in specs
        ],
        "measured_order": measured_order,
        "modes": {
            mode: {
                "kind_name": pmu_kind_name(mode),
                "live_teams": pmu_live_teams(mode),
                "live_streams": len(pmu_live_teams(mode)),
                "live_threads": sum(width for _core, width in pmu_live_teams(mode)),
                "wall_ms": _stats(wall_ms[mode]),
                "trace": {key: _stats(values) for key, values in calls[mode].items()},
                "pmu": pmu_samples[mode],
            }
            for mode in modes
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
