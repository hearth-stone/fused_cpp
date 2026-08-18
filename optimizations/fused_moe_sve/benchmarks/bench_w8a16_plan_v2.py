from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

import torch

from fused_cpp.moe.bf16_tiled import (
    fused_moe_bf16_tiled_async_plan,
    fused_moe_w8a16_tiled_async_plan,
    prepare_fused_moe_bf16_tiled_weights,
    prepare_fused_moe_w8a16_tiled_weights,
)
from fused_cpp.moe.planner_runtime import MoePlannerRuntime
from fused_cpp.moe.plan import AsyncMoEPlanV2, upgrade_legacy_async_plan
from cpu_moe_schedule_optimization.planners.workload_catalog import load_routing_workload
from optimizations.fused_moe_sve.benchmarks.bench_vllm_staged_schedule import materialize_topk_ids


def _make_plan(experts: int, threads: int, team_width: int, cpu_start: int) -> AsyncMoEPlanV2:
    if threads % team_width != 0:
        raise ValueError("threads must be divisible by team_width")
    teams = threads // team_width
    core_begins: list[int] = []
    dependencies: list[int] = []
    dependency_offsets = [0]
    last_task = [-1] * teams
    for expert in range(experts):
        team = expert % teams
        core_begins.append(team * team_width)
        if last_task[team] >= 0:
            dependencies.append(last_task[team])
        dependency_offsets.append(len(dependencies))
        last_task[team] = expert
    bridge = upgrade_legacy_async_plan(
        {
            "num_threads": threads,
            "thread_cpu_ids": list(range(cpu_start, cpu_start + threads)),
            "task_expert_ids": list(range(experts)),
            "task_core_begins": core_begins,
            "task_threads": [team_width] * experts,
            "task_dep_offsets": dependency_offsets,
            "task_deps": dependencies,
        }
    )
    bridge["early_merge"] = False
    return AsyncMoEPlanV2.from_dict(bridge)


def _median_ms(samples: list[float]) -> float:
    return statistics.median(samples) * 1.0e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=36)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--intermediate", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--team-width", type=int, default=2)
    parser.add_argument("--cpu-start", type=int, default=0)
    parser.add_argument("--profile")
    parser.add_argument("--tp-degree", type=int, default=1)
    parser.add_argument("--workload")
    parser.add_argument("--w8-mode", choices=("register", "cache"), default="register")
    parser.add_argument("--w13-window-sweep")
    parser.add_argument("--w2-window-sweep")
    parser.add_argument("--sweep-warmup", type=int, default=2)
    parser.add_argument("--sweep-runs", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--swiglu-limit", type=float, default=0.0)
    args = parser.parse_args()
    if args.tokens * args.top_k % args.experts != 0:
        raise ValueError("tokens * top_k must be divisible by experts")
    generator = torch.Generator().manual_seed(args.seed)
    hidden_states = (torch.randn((args.tokens, args.hidden), generator=generator) * 0.1).to(torch.bfloat16)
    w13 = (torch.randn((args.experts, 2 * args.intermediate, args.hidden), generator=generator) * 0.02).to(
        torch.bfloat16
    )
    w2 = (torch.randn((args.experts, args.hidden, args.intermediate), generator=generator) * 0.02).to(torch.bfloat16)
    workload = None
    if args.workload:
        workload = load_routing_workload(Path(args.workload))
        actual = (args.tokens, args.top_k, args.experts)
        expected = (workload.tokens, workload.top_k, workload.num_experts)
        if actual != expected:
            raise ValueError(f"workload requires tokens/top-k/experts={expected}, got {actual}")
        topk_ids = materialize_topk_ids(
            workload.histogram,
            tokens=args.tokens,
            top_k=args.top_k,
            seed=args.seed,
        )
    else:
        flat_ids = torch.arange(args.tokens * args.top_k, dtype=torch.int64) % args.experts
        topk_ids = flat_ids.reshape(args.tokens, args.top_k).to(torch.int32)
    topk_weights = torch.full((args.tokens, args.top_k), 1.0 / args.top_k, dtype=torch.float32)
    bf16_weights = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="sve")
    w8_weights = prepare_fused_moe_w8a16_tiled_weights(w13, w2)
    if args.profile:
        runtime = MoePlannerRuntime(
            args.profile,
            hidden_size=args.hidden,
            intermediate_size=args.intermediate,
            global_experts=args.experts,
            local_experts=args.experts,
            mode="tp" if args.tp_degree > 1 else "standalone",
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
            raise RuntimeError("the configured planner runtime rejected the benchmark shape")
        print(f"planner={runtime.last_plan}")
    else:
        plan = _make_plan(args.experts, args.threads, args.team_width, args.cpu_start)
    print(
        f"plan_windows w13={sorted(set(plan.task_w13_window_tiles.tolist()))} "
        f"w2={sorted(set(plan.task_w2_window_tiles.tolist()))}"
    )
    call_kwargs = {"swiglu_limit": args.swiglu_limit or None}

    def run_bf16() -> torch.Tensor:
        return fused_moe_bf16_tiled_async_plan(
            hidden_states, bf16_weights, topk_weights, topk_ids, plan, **call_kwargs
        )

    def run_w8() -> torch.Tensor:
        return fused_moe_w8a16_tiled_async_plan(
            hidden_states,
            w8_weights,
            topk_weights,
            topk_ids,
            plan,
            cache_dequant=args.w8_mode == "cache",
            **call_kwargs,
        )

    if bool(args.w13_window_sweep) != bool(args.w2_window_sweep):
        raise ValueError("both window sweep arguments must be provided together")
    if args.w13_window_sweep:
        w13_candidates = [int(value) for value in args.w13_window_sweep.split(",")]
        w2_candidates = [int(value) for value in args.w2_window_sweep.split(",")]
        for w13_window, w2_window in product(w13_candidates, w2_candidates):
            candidate_plan = replace(
                plan,
                task_w13_window_tiles=torch.full_like(plan.task_w13_window_tiles, w13_window),
                task_w2_window_tiles=torch.full_like(plan.task_w2_window_tiles, w2_window),
            )

            def run_candidate_bf16() -> torch.Tensor:
                return fused_moe_bf16_tiled_async_plan(
                    hidden_states, bf16_weights, topk_weights, topk_ids, candidate_plan, **call_kwargs
                )

            def run_candidate_w8() -> torch.Tensor:
                return fused_moe_w8a16_tiled_async_plan(
                    hidden_states,
                    w8_weights,
                    topk_weights,
                    topk_ids,
                    candidate_plan,
                    cache_dequant=True,
                    **call_kwargs,
                )

            for _ in range(args.sweep_warmup):
                run_candidate_bf16()
                run_candidate_w8()
            candidate_bf16 = run_candidate_bf16()
            candidate_w8 = run_candidate_w8()
            candidate_difference = (candidate_w8.float() - candidate_bf16.float()).abs()
            candidate_relative_l2 = float(
                torch.linalg.vector_norm(candidate_difference)
                / torch.linalg.vector_norm(candidate_bf16.float())
            )
            sweep_samples = {"bf16": [], "w8": []}
            for sample in range(args.sweep_runs):
                order = ("bf16", "w8") if sample % 2 == 0 else ("w8", "bf16")
                for name in order:
                    begin = time.perf_counter()
                    (run_candidate_bf16 if name == "bf16" else run_candidate_w8)()
                    sweep_samples[name].append(time.perf_counter() - begin)
            bf16_sweep_ms = _median_ms(sweep_samples["bf16"])
            w8_sweep_ms = _median_ms(sweep_samples["w8"])
            print(
                f"WINDOW_SWEEP w13_tiles={w13_window} w2_tiles={w2_window} "
                f"bf16_ms={bf16_sweep_ms:.6f} w8_cache_ms={w8_sweep_ms:.6f} "
                f"speedup={bf16_sweep_ms / w8_sweep_ms:.6f} "
                f"max_abs={float(candidate_difference.max()):.8f} "
                f"relative_l2={candidate_relative_l2:.8f}"
            )

    bf16_output = run_bf16()
    w8_output = run_w8()
    difference = (w8_output.float() - bf16_output.float()).abs()
    relative_l2 = float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(bf16_output.float()))
    print(
        f"shape=E{args.experts}_T{args.tokens}_K{args.top_k}_H{args.hidden}_F{args.intermediate} "
        f"threads={args.threads} team_width={args.team_width} "
        f"w8_mode={args.w8_mode} "
        f"max_abs={float(difference.max()):.8f} relative_l2={relative_l2:.8f}"
    )
    if workload is not None:
        active = [routes for routes in workload.histogram if routes > 0]
        print(
            f"workload={workload.name} active_experts={len(active)} "
            f"routes_min={min(active)} routes_max={max(active)} routes_std={workload.observed_routes_std:.6f}"
        )
    bf16_bytes = bf16_weights.w13[0].nbytes + bf16_weights.w2[0].nbytes
    w8_bytes = (
        w8_weights.w13[0].nbytes
        + w8_weights.w13[3].nbytes
        + w8_weights.w2[0].nbytes
        + w8_weights.w2[3].nbytes
    )
    print(f"bf16_weight_bytes={bf16_bytes} w8_weight_bytes={w8_bytes} ratio={w8_bytes / bf16_bytes:.6f}")

    for iteration in range(args.warmup):
        (run_bf16 if iteration % 2 == 0 else run_w8)()
    samples = {"bf16": [], "w8a16": []}
    functions = {"bf16": run_bf16, "w8a16": run_w8}
    for sample in range(args.runs):
        order = ("bf16", "w8a16") if sample % 2 == 0 else ("w8a16", "bf16")
        for name in order:
            begin = time.perf_counter()
            functions[name]()
            samples[name].append(time.perf_counter() - begin)
    bf16_ms = _median_ms(samples["bf16"])
    w8_ms = _median_ms(samples["w8a16"])
    print(
        f"bf16_median_ms={bf16_ms:.6f} w8a16_median_ms={w8_ms:.6f} "
        f"speedup={bf16_ms / w8_ms:.6f} gain_pct={(bf16_ms / w8_ms - 1.0) * 100.0:.3f}"
    )


if __name__ == "__main__":
    main()
