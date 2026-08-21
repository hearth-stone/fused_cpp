#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare the Lab W8A8 i8mm pipeline with production BF16 on one Plan V2."""

from __future__ import annotations

import argparse
import gc
import statistics
import struct
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import torch

from cpu_moe_schedule_optimization.planners.workload_catalog import load_routing_workload
from fused_cpp.moe.bf16_tiled import (
    fused_moe_bf16_tiled_async_plan,
    prepare_fused_moe_bf16_tiled_weights,
)
from fused_cpp.moe.planner_runtime import MoePlannerRuntime
from optimizations.fused_moe_sve.benchmarks.bench_vllm_staged_schedule import materialize_topk_ids


def _median_ms(samples: list[float]) -> float:
    return statistics.median(samples) * 1.0e3


def _write_routes(path: Path, topk_ids: torch.Tensor) -> None:
    values = [int(value) for value in topk_ids.reshape(-1).tolist()]
    path.write_bytes(struct.pack(f"<{len(values)}i", *values))


def _write_schedule(path: Path, plan, team_width: int) -> None:
    lanes = [[] for _ in range(plan.num_threads // team_width)]
    experts = [int(value) for value in plan.task_expert_ids.tolist()]
    core_begins = [int(value) for value in plan.task_core_begins.tolist()]
    widths = [int(value) for value in plan.task_threads.tolist()]
    for expert, core_begin, width in zip(experts, core_begins, widths, strict=True):
        if width != team_width or core_begin % team_width:
            raise ValueError("W8A8 Lab benchmark requires one fixed team width")
        lanes[core_begin // team_width].append(expert)
    path.write_text("".join(",".join(str(expert) for expert in lane) + "\n" for lane in lanes), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--threads", type=int, default=80)
    parser.add_argument("--team-width", type=int, default=8)
    parser.add_argument("--cpu-start", type=int, default=240)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--w13-window", type=int, default=4)
    parser.add_argument("--w2-window", type=int, default=32)
    parser.add_argument("--w8a8-gemm-kernel", choices=("hybrid", "packed_m12"), default="packed_m12")
    parser.add_argument("--w8a8-w13-window", type=int, default=1)
    parser.add_argument("--w8a8-w2-window", type=int, default=4)
    parser.add_argument("--w8a8-window-pairs")
    parser.add_argument("--warmup", type=int, default=7)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    args = parser.parse_args()

    workload = load_routing_workload(Path(args.workload))
    expected = (workload.tokens, workload.top_k, workload.num_experts)
    if (args.tokens, args.top_k, args.experts) != expected:
        raise ValueError(f"workload requires tokens/top-k/experts={expected}")
    topk_ids = materialize_topk_ids(workload.histogram, tokens=args.tokens, top_k=args.top_k, seed=args.seed)
    topk_weights = torch.full((args.tokens, args.top_k), 1.0 / args.top_k, dtype=torch.float32)

    generator = torch.Generator().manual_seed(args.seed)
    hidden_states = (torch.randn((args.tokens, args.hidden), generator=generator) * 0.2).to(torch.bfloat16)
    w13 = (torch.randn((args.experts, 2 * args.intermediate, args.hidden), generator=generator) * 0.02).to(
        torch.bfloat16
    )
    w2 = (torch.randn((args.experts, args.hidden, args.intermediate), generator=generator) * 0.02).to(torch.bfloat16)
    bf16_weights = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="sve")
    runtime = MoePlannerRuntime(
        args.profile,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        global_experts=args.experts,
        local_experts=args.experts,
        mode="tp",
        degree=args.tp_degree,
        cpu_ids=tuple(range(args.cpu_start, args.cpu_start + args.threads)),
    )
    plan = runtime.plan_for_dispatch(
        bf16_weights,
        topk_ids,
        num_threads=args.threads,
        activation="silu",
        global_num_experts=-1,
    )
    if plan is None:
        raise RuntimeError("planner rejected W8A8 benchmark shape")
    if set(int(value) for value in plan.task_threads.tolist()) != {args.team_width}:
        raise RuntimeError(f"planner selected incompatible widths: {sorted(set(plan.task_threads.tolist()))}")
    plan = replace(
        plan,
        task_w13_window_tiles=torch.full_like(plan.task_w13_window_tiles, args.w13_window),
        task_w2_window_tiles=torch.full_like(plan.task_w2_window_tiles, args.w2_window),
        early_merge=False,
    )
    print(f"planner={runtime.last_plan}")

    def run_bf16(weights=bf16_weights) -> torch.Tensor:
        return fused_moe_bf16_tiled_async_plan(
            hidden_states,
            weights,
            topk_weights,
            topk_ids,
            plan,
            swiglu_limit=args.swiglu_limit,
        )

    for _ in range(args.warmup):
        run_bf16()
    bf16_samples = []
    bf16_output = run_bf16()
    for _ in range(args.runs):
        begin = time.perf_counter()
        bf16_output = run_bf16()
        bf16_samples.append(time.perf_counter() - begin)
    bf16_ms = _median_ms(bf16_samples)
    print(f"bf16_median_ms={bf16_ms:.6f} checksum={float(bf16_output.float().sum()):.6f}")

    del bf16_output, run_bf16, bf16_weights, w13, w2
    gc.collect()
    with tempfile.TemporaryDirectory(prefix="fused_cpp_w8a8_") as directory:
        directory_path = Path(directory)
        routes_path = directory_path / "routes.bin"
        schedule_path = directory_path / "schedule.csv"
        _write_routes(routes_path, topk_ids)
        _write_schedule(schedule_path, plan, args.team_width)
        runner = Path(__file__).with_name("run_w8a8_i8mm.sh")
        command = [
            str(runner),
            "--routes",
            str(routes_path),
            "--schedule",
            str(schedule_path),
            "--tokens",
            str(args.tokens),
            "--top-k",
            str(args.top_k),
            "--experts",
            str(args.experts),
            "--hidden",
            str(args.hidden),
            "--intermediate",
            str(args.intermediate),
            "--threads",
            str(args.threads),
            "--team-width",
            str(args.team_width),
            "--gemm-kernel",
            args.w8a8_gemm_kernel,
            "--w13-window",
            str(args.w8a8_w13_window),
            "--w2-window",
            str(args.w8a8_w2_window),
            "--cpu-start",
            str(args.cpu_start),
            "--warmup",
            str(args.warmup),
            "--runs",
            str(args.runs),
            "--swiglu-limit",
            str(args.swiglu_limit),
            "--seed",
            str(args.seed),
        ]
        if args.w8a8_window_pairs:
            command.extend(("--window-pairs", args.w8a8_window_pairs))
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        print(completed.stdout.strip())


if __name__ == "__main__":
    main()
