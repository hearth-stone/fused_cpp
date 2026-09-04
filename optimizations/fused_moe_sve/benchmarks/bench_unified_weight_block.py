#!/usr/bin/env python3
"""Compare split W13/W2 tensors against one contiguous weight block.

Packed experts are already [E, packed_numel] per matrix. This probe asks
whether putting W13 and W2 for the whole layer into one DRAM allocation
reduces the 16x1T leftover tax. It does not add a structure.
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
    PreparedBF16TiledFusedMoEWeights,
    fused_moe_bf16_tiled_async_plan,
    prepare_fused_moe_bf16_tiled_weights,
)
from optimizations.fused_moe_sve.benchmarks.bench_fill_port_vs_stream import (  # noqa: E402
    BACKGROUND_EXPERTS,
    _build_bridge,
)
from optimizations.fused_moe_sve.benchmarks.bench_gather_injection_overlap import (  # noqa: E402
    _cpu_mapping,
    _parse_calls,
)
from optimizations.fused_moe_sve.benchmarks.bench_small_expert_context import (  # noqa: E402
    _paired_delta_stats,
    _sha256,
    _stats,
)


OUTPUT_KIND = "moe_unified_weight_block_probe"
SCHEMA_VERSION = 1
ALIGN_ELEMS = 128
LAYOUTS = ("split", "unified")
MODES = (
    "isolated_head",
    "wide16_same_head",
    "many16_1t_same_head",
    "many16_1t_cross_head",
)
LAYOUT_NEUTRAL_MS = 0.04
UNIFIED_HELPS_MS = 0.08
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
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--trace-dir", type=Path, default=Path("/tmp/moe_unified_weight_block"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def cell_name(layout: str, mode: str) -> str:
    return f"{layout}:{mode}"


def parse_cell(name: str) -> tuple[str, str]:
    if ":" not in name:
        raise ValueError(f"unsupported unified-block cell {name!r}")
    layout, mode = name.split(":", 1)
    if layout not in LAYOUTS or mode not in MODES:
        raise ValueError(f"unsupported unified-block cell {name!r}")
    return layout, mode


def unify_prepared(
    prepared: PreparedBF16TiledFusedMoEWeights,
) -> tuple[PreparedBF16TiledFusedMoEWeights, torch.Tensor]:
    w13 = prepared.w13[0].contiguous()
    w2 = prepared.w2[0].contiguous()
    w13_n = int(w13.numel())
    pad = (ALIGN_ELEMS - (w13_n % ALIGN_ELEMS)) % ALIGN_ELEMS
    owner = torch.empty(w13_n + pad + int(w2.numel()), dtype=w13.dtype)
    owner[:w13_n].copy_(w13.reshape(-1))
    if pad:
        owner[w13_n : w13_n + pad].zero_()
    owner[w13_n + pad :].copy_(w2.reshape(-1))
    w13_u = owner[:w13_n].view(w13.shape)
    w2_u = owner[w13_n + pad :].view(w2.shape)
    if not w13_u.is_contiguous() or not w2_u.is_contiguous():
        raise RuntimeError("unified packed views must stay contiguous")
    unified = PreparedBF16TiledFusedMoEWeights(
        w13=(w13_u, prepared.w13[1], prepared.w13[2]),
        w2=(w2_u, prepared.w2[1], prepared.w2[2]),
        fused_silu=prepared.fused_silu,
        gemm_backend=prepared.gemm_backend,
        backend_n_tile=prepared.backend_n_tile,
        backend_name=prepared.backend_name,
    )
    return unified, owner


def storage_ptr(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().data_ptr())


def layout_report(prepared: PreparedBF16TiledFusedMoEWeights) -> dict[str, object]:
    w13 = prepared.w13[0]
    w2 = prepared.w2[0]
    return {
        "w13_ptr": int(w13.data_ptr()),
        "w2_ptr": int(w2.data_ptr()),
        "w13_storage_ptr": storage_ptr(w13),
        "w2_storage_ptr": storage_ptr(w2),
        "same_storage": storage_ptr(w13) == storage_ptr(w2),
        "w13_bytes": int(w13.nbytes),
        "w2_bytes": int(w2.nbytes),
        "w13_shape": list(w13.shape),
        "w2_shape": list(w2.shape),
        "span_bytes": abs(int(w2.data_ptr()) - int(w13.data_ptr())) + int(w2.nbytes),
    }


def _overlap_ok(mode_calls: dict[str, list[float]], expected: float) -> bool:
    experts = mode_calls.get("peer_overlap_experts")
    if not experts:
        return expected <= 0.0
    return float(_stats(experts)["median_ms"]) >= OVERLAP_EXPERT_FRACTION * expected


def decide_unified_block(
    comparisons: dict[str, dict[str, object]],
    calls: dict[str, dict[str, list[float]]],
) -> dict[str, object]:
    def delta(layout: str, mode: str) -> float:
        return float(comparisons[f"{cell_name(layout, mode)}_vs_isolated"]["delta"]["median_ms"])

    many_split = delta("split", "many16_1t_same_head")
    many_unified = delta("unified", "many16_1t_same_head")
    wide_split = delta("split", "wide16_same_head")
    wide_unified = delta("unified", "wide16_same_head")
    many_cross_split = delta("split", "many16_1t_cross_head")
    many_cross_unified = delta("unified", "many16_1t_cross_head")
    overlap_ok = (
        _overlap_ok(calls[cell_name("split", "wide16_same_head")], 1.0)
        and _overlap_ok(calls[cell_name("unified", "wide16_same_head")], 1.0)
        and _overlap_ok(calls[cell_name("split", "many16_1t_same_head")], 16.0)
        and _overlap_ok(calls[cell_name("unified", "many16_1t_same_head")], 16.0)
    )
    layout_neutral = abs(many_split - many_unified) <= LAYOUT_NEUTRAL_MS
    unified_helps = many_split - many_unified >= UNIFIED_HELPS_MS
    remote_near_zero = (
        abs(many_cross_split) <= REMOTE_NEAR_ZERO_MS and abs(many_cross_unified) <= REMOTE_NEAR_ZERO_MS
    )
    if not overlap_ok:
        signature = "invalid_overlap"
    elif unified_helps:
        signature = "unified_helps"
    elif layout_neutral:
        signature = "layout_neutral"
    else:
        signature = "inconclusive"
    return {
        "add_default_off_structure": False,
        "signature": signature,
        "overlap_ok": overlap_ok,
        "layout_neutral": layout_neutral,
        "unified_helps": unified_helps,
        "remote_near_zero": remote_near_zero,
        "many16_split_ms": many_split,
        "many16_unified_ms": many_unified,
        "wide16_split_ms": wide_split,
        "wide16_unified_ms": wide_unified,
        "many16_cross_split_ms": many_cross_split,
        "many16_cross_unified_ms": many_cross_unified,
        "gates": {
            "layout_neutral_ms": LAYOUT_NEUTRAL_MS,
            "unified_helps_ms": UNIFIED_HELPS_MS,
            "remote_near_zero_ms": REMOTE_NEAR_ZERO_MS,
        },
        "reason": (
            "Do not add a structure. unified_helps means one DRAM block lowers "
            "the 16x1T leftover. layout_neutral means split W13/W2 already "
            "behaves like one block."
        ),
    }


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if args.threads != 80:
        raise ValueError("the Arm unified-block probe requires exactly 80 threads")
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
        raise RuntimeError("unified-block modes must keep the same expert route histogram")
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
    split_copies = [
        prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="arm_sve_bf16")
        for _ in range(args.weight_copies + 1)
    ]
    unified_copies = []
    owners: list[torch.Tensor] = []
    for prepared in split_copies:
        unified, owner = unify_prepared(prepared)
        unified_copies.append(unified)
        owners.append(owner)
    packed = {"split": split_copies, "unified": unified_copies}
    scrub_copy = args.weight_copies
    outputs = {cell_name(layout, mode): torch.empty_like(hidden) for layout in LAYOUTS for mode in MODES}
    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "0"
    os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE"] = "0"
    cells = [cell_name(layout, mode) for layout in LAYOUTS for mode in MODES]

    def run(layout: str, mode: str, copy_index: int) -> torch.Tensor:
        name = cell_name(layout, mode)
        return fused_moe_bf16_tiled_async_plan(
            hidden,
            packed[layout][copy_index],
            topk_weights,
            topk_ids,
            plans[mode],
            global_num_experts=expert_count,
            out=outputs[name],
        )

    reference = run("split", MODES[0], 0).clone()
    for layout in LAYOUTS:
        for mode in MODES:
            torch.testing.assert_close(run(layout, mode, 0).float(), reference.float(), atol=0, rtol=0)
    warmup_order = random.Random(args.seed ^ 0xA5A5)
    for round_index in range(args.warmup):
        names = list(cells)
        warmup_order.shuffle(names)
        for position, name in enumerate(names):
            layout, mode = parse_cell(name)
            run(layout, "isolated_head", scrub_copy)
            run(layout, mode, (round_index + position) % args.weight_copies)

    trace_paths = {name: args.trace_dir / f"{name.replace(':', '_')}.log" for name in cells}
    for trace_path in trace_paths.values():
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.unlink(missing_ok=True)
    trace_order = random.Random(args.seed ^ 0x5A5A)
    for run_index in range(args.runs):
        names = list(cells)
        trace_order.shuffle(names)
        for position, name in enumerate(names):
            layout, mode = parse_cell(name)
            os.environ["FUSED_CPP_MOE_TRACE"] = "0"
            run(layout, "isolated_head", scrub_copy)
            os.environ["FUSED_CPP_MOE_TRACE_FILE"] = str(trace_paths[name])
            os.environ["FUSED_CPP_MOE_TRACE"] = "1"
            run(layout, mode, (run_index + position) % args.weight_copies)
            os.environ["FUSED_CPP_MOE_TRACE"] = "0"

    calls = {name: _parse_calls(path) for name, path in trace_paths.items()}
    comparisons = {}
    for layout in LAYOUTS:
        isolated = calls[cell_name(layout, "isolated_head")]["target_span"]
        for mode in MODES:
            if mode == "isolated_head":
                continue
            name = cell_name(layout, mode)
            comparisons[f"{name}_vs_isolated"] = _paired_delta_stats(calls[name]["target_span"], isolated)
    decision = decide_unified_block(comparisons, calls)
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
            "background_experts": BACKGROUND_EXPERTS,
        },
        "cpu_mapping": _cpu_mapping(cpu_ids),
        "layouts": {
            "split": layout_report(split_copies[0]),
            "unified": layout_report(unified_copies[0]),
        },
        "method": {
            "warmup": args.warmup,
            "trace_runs": args.runs,
            "trace_order": "randomized_paired_rounds",
            "weight_copies": args.weight_copies,
            "scrub_policy": "dedicated_disjoint_isolated_packed_copy_before_every_sample",
            "same_tasks_routes_weights_across_layouts": True,
            "metric": "target first-phase start to final-phase end",
            "hugetlb": "not reserved on this host; unified is one anonymous allocation",
        },
        "modes": {
            name: {
                "layout": parse_cell(name)[0],
                "mode": parse_cell(name)[1],
                **{key: _stats(values) for key, values in mode_calls.items()},
            }
            for name, mode_calls in calls.items()
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
