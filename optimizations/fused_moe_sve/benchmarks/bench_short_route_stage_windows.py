#!/usr/bin/env python3
"""E2E A/B for short-route packed-B stage windows on the production planner.

The calibrated default policy ``amazon_c5_192c_tp4_f512_v1`` has no route band
below 49 routes, so every expert with ``M < 49`` inherits the operator-wide
legacy split-W13 window (two 4 MiB W13 ranges and one 4 MiB W2 range). Isolated
probes on this host show that short-route experts lose 16%-114% of their useful
packed-B bandwidth in that regime, because each additional ``M12`` panel
re-reads the whole window.

This benchmark measures whether closing that gap survives end to end:

``legacy``
    Production auto plan with the stage-window policy disabled. This is the
    historical comparator.

``policy``
    Production auto plan with the calibrated default policy. Since V2 this
    includes the ``13 <= M <= 48`` band, so the cost model re-scores every
    candidate with the short-route windows lowered.

``manual``
    The ``legacy`` plan with the short-route band's windows applied post hoc. The
    task graph, widths and placement are byte-identical to ``legacy``, so this
    isolates the window effect from any planner re-search. Widening a band's
    coverage disables the cost model's full-workload anchor for the widths it
    adds, which can move the chosen shape, so keeping the two apart stays useful
    whenever the table changes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
COST_MODEL_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"
PLANNER_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "planners"
BENCH_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO_ROOT / "src"), str(COST_MODEL_DIR), str(PLANNER_DIR), str(BENCH_DIR)]

from bench_vllm_staged_schedule import materialize_topk_ids  # noqa: E402
from fused_cpp.moe import (  # noqa: E402
    AsyncMoEPlanV2,
    fused_moe_bf16_tiled_async_plan,
    prepare_fused_moe_bf16_tiled_weights,
)
from phase_model import ContentionCostModel  # noqa: E402
from planned_moe import PlannedMoE  # noqa: E402
from stage_window_policy import (  # noqa: E402
    AMAZON_C5_192C_TP4_F512_SHORT_ROUTE_BAND,
    AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V2,
)
from workload_catalog import default_offline_workloads  # noqa: E402


DEFAULT_PROFILE = (
    REPO_ROOT
    / "cpu_moe_schedule_optimization"
    / "cost_model"
    / "profiles"
    / "contention_async_amazon_c5_192c_dual_numa_tp4_sve_F512_E256_splitw13_schema_v2_xbyak_exactm_20260727.json"
)

STAGE_GEOMETRY = {"hidden_size": 4096, "intermediate_size": 512, "backend_n_tile": 8}
SHORT_ROUTE_BAND = AMAZON_C5_192C_TP4_F512_SHORT_ROUTE_BAND
SHORT_ROUTE_MIN = SHORT_ROUTE_BAND.min_routes
SHORT_ROUTE_MAX = SHORT_ROUTE_BAND.max_routes

# Lowered from the production band so ``manual`` cannot drift from ``policy``.
SHORT_ROUTE_WINDOWS: dict[int, tuple[int, int]] = {
    threads: AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V2.select(SHORT_ROUTE_MIN, threads)
    for threads in SHORT_ROUTE_BAND.widths
}

LEGACY = "legacy"
POLICY = "policy"
MANUAL = "manual"
VARIANTS = (LEGACY, POLICY, MANUAL)


def short_route_window(routes: int, threads: int) -> tuple[int, int] | None:
    if not SHORT_ROUTE_MIN <= routes <= SHORT_ROUTE_MAX:
        return None
    return SHORT_ROUTE_WINDOWS.get(threads)


def apply_short_route_windows(bridge: dict, routes_by_expert: dict[int, int]) -> tuple[dict, int]:
    """Overwrite per-task windows without touching the task graph."""
    updated = dict(bridge)
    w13 = list(bridge["task_w13_window_bytes"])
    w2 = list(bridge["task_w2_window_bytes"])
    overridden = 0
    for index, (expert, threads) in enumerate(zip(bridge["task_expert_ids"], bridge["task_threads"])):
        selected = short_route_window(routes_by_expert[int(expert)], int(threads))
        if selected is None:
            continue
        w13[index], w2[index] = selected
        overridden += 1
    updated["task_w13_window_bytes"] = w13
    updated["task_w2_window_bytes"] = w2
    return updated, overridden


def parse_cpu_ids(text: str) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first, last = (int(value) for value in item.split("-", 1))
            values.extend(range(first, last + 1))
        else:
            values.append(int(item))
    if len(values) != len(set(values)):
        raise ValueError("duplicate CPU id")
    return values


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/B short-route stage windows end to end.")
    parser.add_argument("--preset", default="dsv4-real-2048-seq70")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--cpu-ids", default="0-95")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--std", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--reverse-order",
        action="store_true",
        help="reverse the interleaved variant order to check for position bias",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cpu_ids = parse_cpu_ids(args.cpu_ids)
    if len(cpu_ids) != args.threads:
        raise ValueError(f"--cpu-ids must contain exactly {args.threads} CPUs")
    if (args.hidden, args.intermediate) != (STAGE_GEOMETRY["hidden_size"], STAGE_GEOMETRY["intermediate_size"]):
        raise ValueError(
            f"the calibrated stage-window policies are bound to H={STAGE_GEOMETRY['hidden_size']} "
            f"F={STAGE_GEOMETRY['intermediate_size']}, got H={args.hidden} F={args.intermediate}"
        )

    workloads = default_offline_workloads()
    if args.preset not in workloads:
        raise ValueError(f"unknown preset {args.preset!r}; available: {sorted(workloads)}")
    workload = workloads[args.preset]
    histogram = list(workload.histogram)
    counts = [(expert, routes) for expert, routes in enumerate(histogram) if routes > 0]
    routes_by_expert = dict(counts)
    experts = workload.num_experts
    short_route_experts = sum(SHORT_ROUTE_MIN <= routes <= SHORT_ROUTE_MAX for _, routes in counts)
    short_route_routes = sum(routes for _, routes in counts if SHORT_ROUTE_MIN <= routes <= SHORT_ROUTE_MAX)

    model = ContentionCostModel(args.profile)
    if model.policy is None:
        raise ValueError(f"production profile must be schema v2: {args.profile}")

    planners = {
        LEGACY: PlannedMoE(model, args.threads, cpu_ids=cpu_ids, use_default_stage_window_policy=False),
        POLICY: PlannedMoE(model, args.threads, cpu_ids=cpu_ids),
    }
    specs = {name: planner.plan_spec_for(counts) for name, planner in planners.items()}
    metadata = {name: dict(planner.last) for name, planner in planners.items()}

    bridges = {name: spec["bridge"] for name, spec in specs.items()}
    bridges[MANUAL], manual_overridden = apply_short_route_windows(bridges[LEGACY], routes_by_expert)
    specs[MANUAL] = dict(specs[LEGACY])
    metadata[MANUAL] = dict(metadata[LEGACY])
    metadata[MANUAL]["overridden_tasks"] = manual_overridden

    for name in (POLICY,):
        pairs = list(zip(bridges[name]["task_w13_window_bytes"], bridges[name]["task_w2_window_bytes"]))
        metadata[name]["overridden_tasks"] = sum(w13 >= 0 or w2 >= 0 for w13, w2 in pairs)
        metadata[name]["window_pairs"] = sorted({f"{w13}:{w2}" for w13, w2 in pairs if w13 >= 0 or w2 >= 0})

    plans = {name: AsyncMoEPlanV2.from_dict(bridge) for name, bridge in bridges.items()}

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(args.seed)
    topk_ids = materialize_topk_ids(histogram, tokens=workload.tokens, top_k=workload.top_k, seed=args.seed)
    topk_weights = torch.empty((workload.tokens, workload.top_k), dtype=torch.float32).uniform_(
        0.5, 1.5, generator=generator
    )
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    hidden = torch.empty((workload.tokens, args.hidden), dtype=torch.bfloat16).normal_(
        0.0, args.std, generator=generator
    )
    w13 = torch.empty((experts, 2 * args.intermediate, args.hidden), dtype=torch.bfloat16).normal_(
        0.0, args.std, generator=generator
    )
    w2 = torch.empty((experts, args.hidden, args.intermediate), dtype=torch.bfloat16).normal_(
        0.0, args.std, generator=generator
    )
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
    del w13, w2
    outputs = {name: torch.empty_like(hidden) for name in VARIANTS}

    def run(name: str) -> torch.Tensor:
        spec = specs[name]
        return fused_moe_bf16_tiled_async_plan(
            hidden,
            packed,
            topk_weights,
            topk_ids,
            plans[name],
            activation="silu",
            global_num_experts=experts,
            w13_split=spec["w13_split"],
            weight_window_bytes=spec["weight_window_bytes"],
            out=outputs[name],
        )

    for name in VARIANTS:
        run(name)
    reference = outputs[LEGACY].clone()
    mismatched = [name for name in VARIANTS if not torch.equal(outputs[name], reference)]
    if mismatched:
        raise RuntimeError(f"variants are not bit-exact against {LEGACY}: {mismatched}")

    samples: dict[str, list[float]] = {name: [] for name in VARIANTS}
    order = tuple(reversed(VARIANTS)) if args.reverse_order else VARIANTS
    for _ in range(args.warmup):
        for name in order:
            run(name)
    for _ in range(args.runs):
        for name in order:
            begin = time.perf_counter_ns()
            run(name)
            samples[name].append((time.perf_counter_ns() - begin) / 1e6)

    flops = 6 * workload.routes * args.hidden * args.intermediate
    results = {}
    for name in VARIANTS:
        median = statistics.median(samples[name])
        results[name] = {
            "median_ms": median,
            "p10_ms": percentile(samples[name], 0.10),
            "p90_ms": percentile(samples[name], 0.90),
            "min_ms": min(samples[name]),
            "tflops": flops / median / 1e9,
            "samples_ms": samples[name],
            "plan": {
                "shape": list(specs[name]["shape"]),
                "execution_mode": specs[name]["execution_mode"],
                "tail_pool_threads": specs[name]["tail_pool_threads"],
                "tail_repartition_width": specs[name]["tail_repartition_width"],
                "task_stage_window_policy": specs[name].get("task_stage_window_policy"),
                "tasks": len(bridges[name]["task_expert_ids"]),
                "overridden_tasks": metadata[name].get("overridden_tasks", 0),
                "window_pairs": metadata[name].get("window_pairs", []),
            },
        }

    baseline = results[LEGACY]["median_ms"]
    print(f"preset={args.preset} active_experts={len(counts)} routes={workload.routes}")
    print(
        f"short-route band [{SHORT_ROUTE_MIN},{SHORT_ROUTE_MAX}]: {short_route_experts} experts, {short_route_routes} routes"
    )
    print()
    header = f"{'variant':<12}{'median':>10}{'p10':>10}{'p90':>10}{'TFLOP/s':>10}{'ovr':>6}  {'shape / mode':<28}{'vs legacy':>11}"
    print(header)
    print("-" * len(header))
    for name in VARIANTS:
        entry = results[name]
        plan = entry["plan"]
        shape = "x".join(str(width) for width in plan["shape"][:4]) + ("..." if len(plan["shape"]) > 4 else "")
        print(
            f"{name:<12}{entry['median_ms']:>9.3f}ms{entry['p10_ms']:>9.3f}ms{entry['p90_ms']:>9.3f}ms"
            f"{entry['tflops']:>10.3f}{plan['overridden_tasks']:>6}  "
            f"{shape + ' / ' + str(plan['execution_mode']):<28}"
            f"{(baseline / entry['median_ms'] - 1) * 100:>10.2f}%"
        )

    if args.output is not None:
        native_module = importlib.import_module("fused_cpp._moe_C")
        payload = {
            "schema_version": 1,
            "kind": "short_route_stage_window_ab",
            "target": {
                "machine": platform.machine(),
                "cpu_ids": cpu_ids,
                "threads": args.threads,
                "os": platform.platform(),
            },
            "kernel": {
                "extension": str(native_module.__file__),
                "extension_sha256": hashlib.sha256(Path(native_module.__file__).read_bytes()).hexdigest(),
                "hugetlbfs_path": os.environ.get("FUSED_CPP_MOE_HUGETLBFS_PATH", ""),
                "backend_n_tile": int(packed.backend_n_tile),
                "gemm_backend": int(packed.gemm_backend),
            },
            "workload": {
                "preset": args.preset,
                "tokens": workload.tokens,
                "top_k": workload.top_k,
                "num_experts": experts,
                "active_experts": len(counts),
                "routes": workload.routes,
                "short_route_band": [SHORT_ROUTE_MIN, SHORT_ROUTE_MAX],
                "short_route_experts": short_route_experts,
                "short_route_routes": short_route_routes,
            },
            "policy": {
                "name": AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V2.name,
                "short_route_windows": {str(k): list(v) for k, v in SHORT_ROUTE_WINDOWS.items()},
            },
            "measurement": {
                "profile": str(args.profile),
                "warmup": args.warmup,
                "runs": args.runs,
                "order": "interleaved_reversed" if args.reverse_order else "interleaved",
                "statistic": "median",
            },
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
