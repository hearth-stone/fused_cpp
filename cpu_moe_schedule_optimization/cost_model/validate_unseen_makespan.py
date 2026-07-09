#!/usr/bin/env python3
"""Validate contention model on off-grid routes and unseen async DAGs."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpu_moe_schedule_optimization" / "cost_model"))

from phase_model import ContentionCostModel  # noqa: E402
from fused_cpp.moe import (  # noqa: E402
    fused_moe_bf16_tiled_async,
    prepare_fused_moe_bf16_tiled_weights,
)


PHASE_CONFIGS = {
    "phase_3way_offgrid": [(1536, 16), (384, 8), (160, 8)],
    "phase_4way_mixed": [(1280, 8), (960, 8), (448, 8), (224, 8)],
    "phase_8way_balanced_offgrid": [(768, 4)] * 8,
    "phase_tail_heavy": [(640, 16), (320, 8), (160, 4), (160, 4)],
}

DAG_PLANS = {
    "dag_two_waves": [
        (768, 8, 0, []),
        (768, 8, 8, []),
        (320, 8, 16, []),
        (320, 8, 24, []),
        (160, 8, 16, [2]),
        (160, 8, 24, [3]),
    ],
    "dag_staggered": [
        (1536, 16, 0, []),
        (448, 8, 16, []),
        (224, 4, 24, []),
        (224, 4, 28, []),
        (640, 8, 16, [1]),
        (160, 4, 24, [2]),
        (160, 4, 28, [3]),
    ],
    "dag_chain_plus_side": [
        (960, 16, 0, []),
        (960, 16, 0, [0]),
        (384, 8, 16, []),
        (384, 8, 24, []),
        (192, 8, 16, [2]),
        (192, 8, 24, [3]),
    ],
}


def bf16(shape: tuple[int, ...], generator: torch.Generator, std: float) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.bfloat16).normal_(0.0, std, generator=generator)


def measure(run, *, warmup: int, runs: int) -> list[int]:
    for _ in range(warmup):
        out = run()
        _ = float(out.flatten()[0])
    gc.collect()
    times: list[int] = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        out = run()
        _ = float(out.flatten()[0])
        times.append(time.perf_counter_ns() - t0)
    return times


def build_run(
    *,
    packed,
    hidden_size: int,
    plan: list[tuple[int, int, int, list[int]]],
    generator: torch.Generator,
    std: float,
):
    total_tokens = sum(routes for routes, _, _, _ in plan)
    x = bf16((total_tokens, hidden_size), generator, std)
    topk_ids = torch.empty((total_tokens, 1), dtype=torch.int32)
    cursor = 0
    for expert_id, (routes, _, _, _) in enumerate(plan):
        topk_ids[cursor : cursor + routes, 0] = expert_id
        cursor += routes
    topk_weights = torch.ones((total_tokens, 1), dtype=torch.float32)

    dep_offsets = [0]
    dep_flat: list[int] = []
    for _, _, _, deps in plan:
        dep_flat.extend(deps)
        dep_offsets.append(len(dep_flat))

    num_threads = max(core_begin + threads for _, threads, core_begin, _ in plan)
    task_expert_ids = torch.arange(len(plan), dtype=torch.int32)
    task_core_begins = torch.tensor([core_begin for _, _, core_begin, _ in plan], dtype=torch.int32)
    task_threads = torch.tensor([threads for _, threads, _, _ in plan], dtype=torch.int32)
    task_dep_offsets = torch.tensor(dep_offsets, dtype=torch.int32)
    task_deps = torch.tensor(dep_flat, dtype=torch.int32)
    thread_cpu_ids = torch.arange(num_threads, dtype=torch.int32)

    def run() -> torch.Tensor:
        return fused_moe_bf16_tiled_async(
            x,
            packed,
            topk_weights,
            topk_ids,
            task_expert_ids,
            task_core_begins,
            task_threads,
            task_dep_offsets,
            task_deps,
            thread_cpu_ids=thread_cpu_ids,
            num_threads=num_threads,
            activation="silu",
            global_num_experts=len(plan),
            skip_weighted=True,
        )

    return run


def phase_to_plan(config: list[tuple[int, int]]) -> list[tuple[int, int, int, list[int]]]:
    plan: list[tuple[int, int, int, list[int]]] = []
    core = 0
    for routes, threads in config:
        plan.append((routes, threads, core, []))
        core += threads
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp" / "moe_unseen_validate.json")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--ffn-hidden-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--std", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = ContentionCostModel(str(args.profile))
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(args.seed)

    max_tasks = max(
        max(len(v) for v in PHASE_CONFIGS.values()),
        max(len(v) for v in DAG_PLANS.values()),
    )
    w13 = bf16((max_tasks, 2 * args.ffn_hidden_size, args.hidden_size), generator, args.std)
    w2 = bf16((max_tasks, args.hidden_size, args.ffn_hidden_size), generator, args.std)
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)

    rows: list[dict] = []
    print("%-28s %8s %8s %8s" % ("case", "meas_ms", "pred_ms", "err%"))

    for name, config in PHASE_CONFIGS.items():
        plan = phase_to_plan(config)
        run = build_run(
            packed=packed,
            hidden_size=args.hidden_size,
            plan=plan,
            generator=generator,
            std=args.std,
        )
        median_ns = int(statistics.median(measure(run, warmup=args.warmup, runs=args.runs)))
        pred_ns = float(model.phase_makespan(config))
        err_pct = (pred_ns - median_ns) / median_ns * 100.0
        rows.append(
            {
                "kind": "phase",
                "name": name,
                "config": config,
                "measured_ns": median_ns,
                "predicted_ns": int(pred_ns),
                "err_pct": err_pct,
            }
        )
        print("%-28s %8.3f %8.3f %+8.1f" % (name, median_ns / 1e6, pred_ns / 1e6, err_pct))

    for name, plan in DAG_PLANS.items():
        run = build_run(
            packed=packed,
            hidden_size=args.hidden_size,
            plan=plan,
            generator=generator,
            std=args.std,
        )
        median_ns = int(statistics.median(measure(run, warmup=args.warmup, runs=args.runs)))
        tasks = [(routes, threads, deps) for routes, threads, _, deps in plan]
        pred_ns = float(model.dag_makespan(tasks))
        err_pct = (pred_ns - median_ns) / median_ns * 100.0
        rows.append(
            {
                "kind": "dag",
                "name": name,
                "plan": plan,
                "measured_ns": median_ns,
                "predicted_ns": int(pred_ns),
                "err_pct": err_pct,
            }
        )
        print("%-28s %8.3f %8.3f %+8.1f" % (name, median_ns / 1e6, pred_ns / 1e6, err_pct))

    abs_errs = [abs(r["err_pct"]) for r in rows]
    summary = {
        "abs_err_median_pct": statistics.median(abs_errs),
        "abs_err_max_pct": max(abs_errs),
        "num_cases": len(rows),
    }
    payload = {"profile": str(args.profile), "rows": rows, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("summary", json.dumps(summary, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
