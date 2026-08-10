#!/usr/bin/env python3
"""Compare post-expert and fixed-owner early-merge drain policies."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fused_cpp.moe import (  # noqa: E402
    fused_moe_bf16_tiled_async,
    prepare_fused_moe_bf16_tiled_weights,
)


READY_FLAG = "FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE"
DRAIN_FLAG = "FUSED_CPP_MOE_ASYNC_READY_TOKEN_DRAIN"
BATCH_FLAG = "FUSED_CPP_MOE_ASYNC_READY_TOKEN_BATCH"
PREFETCH_FLAG = "FUSED_CPP_MOE_ASYNC_READY_TOKEN_PREFETCH"


@dataclass(frozen=True)
class Variant:
    name: str
    ready: bool
    drain: bool
    batch: int
    prefetch: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--distribution", choices=("balanced", "two-group"), default="balanced")
    parser.add_argument("--short-fraction", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def select_variant(variant: Variant) -> None:
    os.environ[READY_FLAG] = "1" if variant.ready else "0"
    os.environ[DRAIN_FLAG] = "1" if variant.drain else "0"
    os.environ[BATCH_FLAG] = str(variant.batch)
    os.environ[PREFETCH_FLAG] = "1" if variant.prefetch else "0"


def make_topk_ids(args: argparse.Namespace) -> torch.Tensor:
    if args.distribution == "balanced":
        return torch.tensor(
            [
                [(token + slot) % args.experts for slot in range(args.top_k)]
                for token in range(args.tokens)
            ],
            dtype=torch.int32,
        )

    if args.experts != 2 * args.top_k:
        raise ValueError("two-group distribution requires experts == 2 * top-k")
    if not 0.0 < args.short_fraction < 1.0:
        raise ValueError("short-fraction must be in (0, 1)")
    short_tokens = max(1, min(args.tokens - 1, round(args.tokens * args.short_fraction)))
    topk_ids = torch.empty((args.tokens, args.top_k), dtype=torch.int32)
    topk_ids[:short_tokens] = torch.arange(args.top_k, dtype=torch.int32)
    topk_ids[short_tokens:] = torch.arange(args.top_k, 2 * args.top_k, dtype=torch.int32)
    return topk_ids


def parse_trace(path: Path) -> dict[str, float | int]:
    ready_records = 0
    ready_worker_ms = 0.0
    final_merge_records = 0
    final_merge_worker_ms = 0.0
    final_merge_total_ms = 0.0
    scheduled_compute_ms = 0.0
    e2e_ms = 0.0
    for line in path.read_text().splitlines():
        fields = {part.split("=", 1)[0]: part.split("=", 1)[1] for part in line.split() if "=" in part}
        if line.startswith("MOE_CALL "):
            e2e_ms = float(fields["e2e_ms"])
        elif line.startswith("PHASE "):
            stage = fields.get("stage")
            if stage == "merge_ready_token":
                ready_records += 1
                ready_worker_ms += float(fields["ms"])
            elif stage == "merge_routes":
                final_merge_records += 1
                final_merge_worker_ms += float(fields["ms"])
            elif stage == "merge_routes_total":
                final_merge_total_ms = float(fields["ms"])
            elif stage == "scheduled_compute":
                scheduled_compute_ms = float(fields["ms"])
    return {
        "e2e_ms": e2e_ms,
        "ready_tokens": ready_records,
        "ready_worker_ms_sum": ready_worker_ms,
        "final_merge_records": final_merge_records,
        "final_merge_worker_ms_sum": final_merge_worker_ms,
        "final_merge_total_ms": final_merge_total_ms,
        "scheduled_compute_ms": scheduled_compute_ms,
    }


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    positive = (args.tokens, args.hidden, args.intermediate, args.experts, args.top_k, args.threads, args.runs)
    if min(positive) <= 0 or args.warmup < 0:
        raise ValueError("shapes, threads, and runs must be positive; warmup must be non-negative")
    if args.experts < args.top_k:
        raise ValueError("experts must be at least top-k")
    if not 1 <= args.batch <= 64:
        raise ValueError("batch must be in [1, 64]")

    variants = (
        Variant("post_barrier", ready=False, drain=False, batch=1, prefetch=False),
        Variant("fixed_owner_early_final", ready=True, drain=False, batch=1, prefetch=False),
        Variant("fixed_owner_drain", ready=True, drain=True, batch=args.batch, prefetch=True),
    )

    affinity = sorted(os.sched_getaffinity(0))
    if args.threads > len(affinity):
        raise ValueError(f"threads={args.threads} exceeds affinity size {len(affinity)}")
    topk_ids = make_topk_ids(args)
    route_counts = torch.bincount(topk_ids.flatten().to(torch.int64), minlength=args.experts)
    active_experts = torch.nonzero(route_counts, as_tuple=False).flatten().to(torch.int32)
    if args.threads % active_experts.numel() != 0:
        raise ValueError("threads must be divisible by active experts")
    threads_per_expert = args.threads // active_experts.numel()

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(args.seed)
    hidden_states = torch.empty((args.tokens, args.hidden), dtype=torch.bfloat16)
    hidden_states.normal_(mean=0.0, std=0.01, generator=generator)
    w13 = torch.empty((args.experts, 2 * args.intermediate, args.hidden), dtype=torch.bfloat16)
    w13.normal_(mean=0.0, std=0.01, generator=generator)
    w2 = torch.empty((args.experts, args.hidden, args.intermediate), dtype=torch.bfloat16)
    w2.normal_(mean=0.0, std=0.01, generator=generator)
    topk_weights = torch.softmax(torch.randn((args.tokens, args.top_k), generator=generator), dim=-1)

    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "0"
    os.environ["FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE"] = "1"
    os.environ["FUSED_CPP_MOE_SVE_ROUTE_MERGE_UNROLL"] = "1"
    os.environ["FUSED_CPP_MOE_TRACE"] = "0"
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
    if packed.gemm_backend != 1:
        raise RuntimeError("benchmark requires the SVE BF16 fused MoE backend")
    del w13, w2

    num_active = active_experts.numel()
    core_begins = torch.arange(num_active, dtype=torch.int32) * threads_per_expert
    team_widths = torch.full((num_active,), threads_per_expert, dtype=torch.int32)
    dep_offsets = torch.zeros(num_active + 1, dtype=torch.int32)
    deps = torch.empty(0, dtype=torch.int32)
    thread_cpu_ids = torch.tensor(affinity[: args.threads], dtype=torch.int32)

    def run() -> torch.Tensor:
        return fused_moe_bf16_tiled_async(
            hidden_states,
            packed,
            topk_weights,
            topk_ids,
            active_experts,
            core_begins,
            team_widths,
            dep_offsets,
            deps,
            thread_cpu_ids=thread_cpu_ids,
            num_threads=args.threads,
            global_num_experts=args.experts,
        )

    outputs: dict[str, torch.Tensor] = {}
    for variant in variants:
        select_variant(variant)
        outputs[variant.name] = run().clone()
    for variant in variants[1:]:
        torch.testing.assert_close(
            outputs[variant.name].float(),
            outputs["post_barrier"].float(),
            atol=0,
            rtol=0,
        )

    warmup_order = random.Random(args.seed ^ 0xA5A5)
    for _ in range(args.warmup):
        ordered = list(variants)
        warmup_order.shuffle(ordered)
        for variant in ordered:
            select_variant(variant)
            run()

    samples = {variant.name: [] for variant in variants}
    sink = 0
    timed_order = random.Random(args.seed ^ 0x5A5A)
    for _ in range(args.runs):
        ordered = list(variants)
        timed_order.shuffle(ordered)
        for variant in ordered:
            select_variant(variant)
            begin = time.perf_counter_ns()
            output = run()
            samples[variant.name].append((time.perf_counter_ns() - begin) / 1.0e6)
            sink ^= int(output.view(torch.int16)[0, 0])

    traces: dict[str, object] = {}
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        for variant in variants:
            trace_path = args.trace_dir / f"{args.distribution}_{variant.name}.log"
            trace_path.unlink(missing_ok=True)
            select_variant(variant)
            os.environ["FUSED_CPP_MOE_TRACE"] = "1"
            os.environ["FUSED_CPP_MOE_TRACE_FILE"] = str(trace_path)
            run()
            os.environ["FUSED_CPP_MOE_TRACE"] = "0"
            traces[variant.name] = parse_trace(trace_path)

    baseline_ms = statistics.median(samples["post_barrier"])
    early_final_ms = statistics.median(samples["fixed_owner_early_final"])
    records: list[dict[str, object]] = []
    for variant in variants:
        median_ms = statistics.median(samples[variant.name])
        records.append(
            {
                "variant": variant.name,
                "ready_token_merge": variant.ready,
                "same_job_drain": variant.drain,
                "batch": variant.batch,
                "prefetch": variant.prefetch,
                "median_ms": median_ms,
                "p10_ms": percentile(samples[variant.name], 0.10),
                "p90_ms": percentile(samples[variant.name], 0.90),
                "gain_vs_post_pct": 100.0 * (baseline_ms / median_ms - 1.0),
                "gain_vs_early_final_pct": 100.0 * (early_final_ms / median_ms - 1.0),
                "samples": samples[variant.name],
            }
        )

    result = {
        "shape": {
            "tokens": args.tokens,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "experts": args.experts,
            "active_experts": num_active,
            "top_k": args.top_k,
            "threads": args.threads,
            "threads_per_expert": threads_per_expert,
            "distribution": args.distribution,
            "short_fraction": args.short_fraction if args.distribution == "two-group" else None,
            "route_counts": route_counts.tolist(),
        },
        "records": records,
        "trace": traces,
        "sink": sink,
    }
    print("variant                    median_ms   vs_post%  vs_early%     p10_ms     p90_ms")
    for record in records:
        print(
            f"{record['variant']:<26} {record['median_ms']:>9.3f} "
            f"{record['gain_vs_post_pct']:>10.2f} {record['gain_vs_early_final_pct']:>10.2f} "
            f"{record['p10_ms']:>10.3f} {record['p90_ms']:>10.3f}"
        )
    if traces:
        print(json.dumps(traces, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
