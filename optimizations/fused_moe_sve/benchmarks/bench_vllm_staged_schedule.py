#!/usr/bin/env python3
"""Compare fixed expert teams with a vLLM-style global staged task pool."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fused_cpp.moe import (  # noqa: E402
    fused_moe_bf16_tiled_async,
    fused_moe_bf16_tiled_vllm_staged,
    prepare_fused_moe_bf16_tiled_weights,
)


VARIANTS = ("fixed_team_async", "vllm_staged")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--team-threads", type=int, default=0)
    parser.add_argument("--distribution", choices=("hot-topk", "round-robin"), default="hot-topk")
    parser.add_argument("--route-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--stage-timing", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def make_topk_ids(args: argparse.Namespace) -> torch.Tensor:
    if args.distribution == "hot-topk":
        return torch.arange(args.top_k, dtype=torch.int32).repeat(args.tokens, 1)
    return torch.tensor(
        [
            [(token * args.top_k + slot) % args.experts for slot in range(args.top_k)]
            for token in range(args.tokens)
        ],
        dtype=torch.int32,
    )


def make_fixed_team_schedule(
    active_experts: torch.Tensor,
    *,
    threads: int,
    requested_team_threads: int,
) -> tuple[torch.Tensor, ...]:
    active_count = int(active_experts.numel())
    if active_count == 0:
        raise ValueError("at least one expert must be active")
    team_threads = requested_team_threads or max(1, threads // min(active_count, threads))
    if team_threads <= 0 or team_threads > threads:
        raise ValueError(f"team-threads must be in [1, {threads}], got {team_threads}")
    slots = threads // team_threads
    if slots <= 0:
        raise ValueError("team-threads leaves no runnable team slot")

    core_begins: list[int] = []
    team_widths: list[int] = []
    dep_offsets = [0]
    deps: list[int] = []
    previous_by_slot = [-1] * slots
    for task_id in range(active_count):
        slot = task_id % slots
        core_begins.append(slot * team_threads)
        team_widths.append(team_threads)
        previous = previous_by_slot[slot]
        if previous >= 0:
            deps.append(previous)
        dep_offsets.append(len(deps))
        previous_by_slot[slot] = task_id

    return (
        active_experts.to(torch.int32),
        torch.tensor(core_begins, dtype=torch.int32),
        torch.tensor(team_widths, dtype=torch.int32),
        torch.tensor(dep_offsets, dtype=torch.int32),
        torch.tensor(deps, dtype=torch.int32),
    )


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    positive = (
        args.tokens,
        args.hidden,
        args.intermediate,
        args.experts,
        args.top_k,
        args.threads,
        args.runs,
    )
    if min(positive) <= 0 or args.warmup < 0:
        raise ValueError("shapes, threads, and runs must be positive; warmup must be non-negative")
    if args.top_k > args.experts:
        raise ValueError("top-k cannot exceed experts")
    if args.hidden % 8 != 0 or args.intermediate % 8 != 0:
        raise ValueError("hidden and intermediate must be multiples of 8")

    affinity = sorted(os.sched_getaffinity(0))
    if args.threads > len(affinity):
        raise ValueError(f"threads={args.threads} exceeds affinity size {len(affinity)}")
    thread_cpu_ids = torch.tensor(affinity[: args.threads], dtype=torch.int32)
    torch.set_num_threads(1)

    topk_ids = make_topk_ids(args)
    route_counts = torch.bincount(topk_ids.flatten().to(torch.int64), minlength=args.experts)
    active_experts = torch.nonzero(route_counts, as_tuple=False).flatten()
    schedule = make_fixed_team_schedule(
        active_experts,
        threads=args.threads,
        requested_team_threads=args.team_threads,
    )
    team_threads = int(schedule[2][0])

    generator = torch.Generator().manual_seed(args.seed)
    hidden = torch.empty((args.tokens, args.hidden), dtype=torch.bfloat16)
    hidden.normal_(mean=0.0, std=0.01, generator=generator)
    w13 = torch.empty((args.experts, 2 * args.intermediate, args.hidden), dtype=torch.bfloat16)
    w13.normal_(mean=0.0, std=0.01, generator=generator)
    w2 = torch.empty((args.experts, args.hidden, args.intermediate), dtype=torch.bfloat16)
    w2.normal_(mean=0.0, std=0.01, generator=generator)
    topk_weights = torch.softmax(torch.randn((args.tokens, args.top_k), generator=generator), dim=-1)

    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_W13_SPLIT_N"] = "1"
    os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE"] = "0"
    os.environ["FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "1" if args.route_dtype == "bf16" else "0"
    os.environ["FUSED_CPP_MOE_SVE_ROUTE_MERGE_UNROLL"] = "1"
    os.environ["FUSED_CPP_MOE_STAGE_TIMING"] = "0"
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="arm_sve_bf16")
    if packed.gemm_backend != 1:
        raise RuntimeError("benchmark requires the SVE BF16 fused MoE backend")
    del w13, w2

    outputs = {name: torch.empty_like(hidden) for name in VARIANTS}

    def run(name: str) -> torch.Tensor:
        if name == "fixed_team_async":
            return fused_moe_bf16_tiled_async(
                hidden,
                packed,
                topk_weights,
                topk_ids,
                *schedule,
                thread_cpu_ids=thread_cpu_ids,
                num_threads=args.threads,
                global_num_experts=args.experts,
                w13_split=True,
                out=outputs[name],
            )
        return fused_moe_bf16_tiled_vllm_staged(
            hidden,
            packed,
            topk_weights,
            topk_ids,
            thread_cpu_ids=thread_cpu_ids,
            num_threads=args.threads,
            global_num_experts=args.experts,
            out=outputs[name],
        )

    reference = run("fixed_team_async").clone()
    candidate = run("vllm_staged").clone()
    torch.testing.assert_close(candidate.float(), reference.float(), atol=0, rtol=0)

    warmup_order = random.Random(args.seed ^ 0xA5A5)
    for _ in range(args.warmup):
        order = list(VARIANTS)
        warmup_order.shuffle(order)
        for name in order:
            run(name)

    samples = {name: [] for name in VARIANTS}
    timed_order = random.Random(args.seed ^ 0x5A5A)
    sink = 0
    for _ in range(args.runs):
        order = list(VARIANTS)
        timed_order.shuffle(order)
        for name in order:
            begin = time.perf_counter_ns()
            result = run(name)
            samples[name].append((time.perf_counter_ns() - begin) / 1.0e6)
            sink ^= int(result.view(torch.int16)[0, 0])

    if args.stage_timing:
        os.environ["FUSED_CPP_MOE_STAGE_TIMING"] = "1"
        for name in VARIANTS:
            run(name)
        os.environ["FUSED_CPP_MOE_STAGE_TIMING"] = "0"

    total_flops = 6 * args.tokens * args.top_k * args.hidden * args.intermediate
    baseline_ms = statistics.median(samples["fixed_team_async"])
    records: list[dict[str, object]] = []
    for name in VARIANTS:
        median_ms = statistics.median(samples[name])
        records.append(
            {
                "variant": name,
                "median_ms": median_ms,
                "p10_ms": percentile(samples[name], 0.10),
                "p90_ms": percentile(samples[name], 0.90),
                "aggregate_tflops": total_flops / median_ms / 1.0e9,
                "gain_pct": 100.0 * (baseline_ms / median_ms - 1.0),
                "samples_ms": samples[name],
            }
        )

    result = {
        "shape": {
            "tokens": args.tokens,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "experts": args.experts,
            "top_k": args.top_k,
            "threads": args.threads,
            "distribution": args.distribution,
            "active_experts": int(active_experts.numel()),
            "team_threads": team_threads,
            "route_dtype": args.route_dtype,
            "route_counts": route_counts.tolist(),
        },
        "method": {
            "fixed_team_w13_split": True,
            "async_ready_token_merge": False,
            "direct_route_store": True,
            "route_merge_unroll": 1,
            "warmup": args.warmup,
            "runs": args.runs,
        },
        "records": records,
        "sink": sink,
    }

    print("variant             median_ms   TFLOP/s   gain_pct     p10_ms     p90_ms")
    for record in records:
        print(
            f"{record['variant']:<19} {record['median_ms']:>9.3f} "
            f"{record['aggregate_tflops']:>9.3f} {record['gain_pct']:>10.2f} "
            f"{record['p10_ms']:>10.3f} {record['p90_ms']:>10.3f}"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
