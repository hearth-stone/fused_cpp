#!/usr/bin/env python3
"""Measure a frozen partial-order frontier in one randomized process session."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path

import torch

from fused_cpp import _moe_C
from fused_cpp.moe import AsyncMoEPlanV2, fused_moe_bf16_tiled_async_plan

from optimizations.fused_moe_sve.benchmarks.bench_high_skew_planner_closure import (
    _prepare_weights,
    _stats,
    load_route_layer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--route-file", type=Path, required=True)
    parser.add_argument("--route-layer", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--threads", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--weight-copies", type=int, default=4)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if min(args.hidden, args.intermediate, args.experts, args.threads, args.runs, args.weight_copies) <= 0:
        raise ValueError("dimensions, runs, threads, and weight copies must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < args.threads:
        raise ValueError(f"threads={args.threads} exceeds affinity size {len(affinity)}")
    torch.set_num_threads(1)

    frontier = json.loads(args.frontier.read_text(encoding="utf-8"))
    if frontier.get("kind") != "partial_order_hardware_frontier":
        raise ValueError("frontier has an unexpected kind")
    expected_extension = str(frontier["identity"]["extension_sha256"])
    actual_extension = _sha256(Path(_moe_C.__file__))
    if actual_extension != expected_extension:
        raise ValueError("frontier extension identity does not match the loaded extension")
    if (
        int(frontier["shape"]["hidden"]) != args.hidden
        or int(frontier["shape"]["intermediate"]) != args.intermediate
        or int(frontier["shape"]["experts"]) != args.experts
        or int(frontier["shape"]["threads"]) != args.threads
    ):
        raise ValueError("frontier shape does not match benchmark arguments")

    topk_ids, route_metadata = load_route_layer(args.route_file, args.route_layer, args.experts)
    if route_metadata["sha256"] != frontier["route"]["sha256"]:
        raise ValueError("frontier route identity does not match --route-file")
    plans = {
        state_hash: AsyncMoEPlanV2.from_dict(plan["plan_v2_bridge"])
        for state_hash, plan in frontier["plans"].items()
    }
    if len(plans) != int(frontier["unique_plans"]):
        raise ValueError("frontier unique plan count is inconsistent")

    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "0"
    os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE"] = "1"
    generator = torch.Generator().manual_seed(args.seed ^ 0x1234)
    tokens, top_k = topk_ids.shape
    hidden = torch.empty((tokens, args.hidden), dtype=torch.bfloat16)
    hidden.normal_(mean=0.0, std=0.01, generator=generator)
    topk_weights = torch.softmax(torch.randn((tokens, top_k), generator=generator), dim=-1)
    packed_copies = _prepare_weights(args)
    output = torch.empty_like(hidden)

    def run(state_hash: str, copy_index: int) -> torch.Tensor:
        return fused_moe_bf16_tiled_async_plan(
            hidden,
            packed_copies[copy_index],
            topk_weights,
            topk_ids,
            plans[state_hash],
            global_num_experts=args.experts,
            out=output,
        )

    anchor_hashes = sorted(
        state_hash
        for state_hash, record in frontier["records"].items()
        if "anchor" in record["roles"]
    )
    reference = run(anchor_hashes[0], 0).clone()
    for state_hash in plans:
        candidate = run(state_hash, 0).clone()
        torch.testing.assert_close(candidate.float(), reference.float(), atol=0, rtol=0)

    state_hashes = list(plans)
    warmup_order = random.Random(args.seed ^ 0xA5A5)
    for round_index in range(args.warmup):
        warmup_order.shuffle(state_hashes)
        for position, state_hash in enumerate(state_hashes):
            run(state_hash, (round_index + position) % len(packed_copies))

    samples = {state_hash: [] for state_hash in plans}
    timed_order = random.Random(args.seed ^ 0x5A5A)
    sink = 0
    begin = time.perf_counter_ns()
    for round_index in range(args.runs):
        state_hashes = list(plans)
        timed_order.shuffle(state_hashes)
        for position, state_hash in enumerate(state_hashes):
            call_begin = time.perf_counter_ns()
            result = run(state_hash, (round_index + position) % len(packed_copies))
            samples[state_hash].append((time.perf_counter_ns() - call_begin) / 1.0e6)
            sink ^= int(result.view(torch.int16)[0, 0])
    measurement_wall_s = (time.perf_counter_ns() - begin) / 1.0e9

    comparisons = []
    for candidate_hash, record in frontier["records"].items():
        for source in record["comparisons"]:
            if source["role"] in {"anchor", "carried_anchor"}:
                continue
            anchor_hash = str(source["anchor_state_hash"])
            paired = [
                100.0 * (anchor / candidate - 1.0)
                for anchor, candidate in zip(
                    samples[anchor_hash],
                    samples[candidate_hash],
                    strict=True,
                )
            ]
            median = statistics.median(paired)
            p10 = _percentile(paired, 0.10)
            comparisons.append(
                {
                    **source,
                    "candidate_state_hash": candidate_hash,
                    "paired_gain_pct": {
                        "median": median,
                        "p10": p10,
                        "p90": _percentile(paired, 0.90),
                        "wins": sum(value > 0.0 for value in paired),
                        "runs": len(paired),
                        "samples": paired,
                    },
                    "session_stable_over_2pct": median > 2.0 and p10 > 0.0,
                }
            )

    result = {
        "kind": "partial_order_hardware_frontier_session",
        "identity": {
            "frontier": str(args.frontier),
            "frontier_sha256": _sha256(args.frontier),
            "extension": str(_moe_C.__file__),
            "extension_sha256": actual_extension,
        },
        "route": route_metadata,
        "method": {
            "processes": 1,
            "randomized_interleaving": True,
            "paired_by_round": True,
            "warmup": args.warmup,
            "runs": args.runs,
            "weight_copies": args.weight_copies,
            "seed": args.seed,
            "cpu_affinity": affinity[: args.threads],
        },
        "plan_stats": {state_hash: _stats(values) for state_hash, values in samples.items()},
        "comparisons": comparisons,
        "summary": {
            "plans": len(plans),
            "comparisons": len(comparisons),
            "stable_over_2pct": sum(item["session_stable_over_2pct"] for item in comparisons),
            "measurement_wall_s": measurement_wall_s,
            "sink": sink,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
