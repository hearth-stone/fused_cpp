#!/usr/bin/env python3
"""Benchmark sequential versus synthetic-expert routed+shared MoE execution."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from fused_cpp.moe import (
    MoePlannerRuntime,
    PreparedBF16TiledFusedMoEWeights,
    fused_moe_bf16_tiled,
    fused_moe_bf16_tiled_with_shared,
    prepare_routed_shared_moe_bf16_tiled_weights,
    set_default_moe_planner_runtime,
    shared_mlp_bf16_tiled,
)


def _views(weights):
    packed = weights.packed
    routed = PreparedBF16TiledFusedMoEWeights(
        w13=(packed.w13[0][:-1], packed.w13[1], packed.w13[2]),
        w2=(packed.w2[0][:-1], packed.w2[1], packed.w2[2]),
        fused_silu=True,
        gemm_backend=packed.gemm_backend,
        backend_n_tile=packed.backend_n_tile,
        backend_name=packed.backend_name,
    )
    shared = PreparedBF16TiledFusedMoEWeights(
        w13=(packed.w13[0][-1:], packed.w13[1], packed.w13[2]),
        w2=(packed.w2[0][-1:], packed.w2[1], packed.w2[2]),
        fused_silu=True,
        gemm_backend=packed.gemm_backend,
        backend_n_tile=packed.backend_n_tile,
        backend_name=packed.backend_name,
    )
    return routed, shared


def _routes(name: str, tokens: int, experts: int, top_k: int, generator: torch.Generator):
    if name == "balanced":
        ids = torch.arange(tokens * top_k, dtype=torch.int64).reshape(tokens, top_k).remainder(experts)
        weights = torch.full((tokens, top_k), 1.0 / top_k)
        return weights, ids.to(torch.int32)
    if name == "uniform":
        scores = torch.rand((tokens, experts), generator=generator)
        weights, ids = torch.topk(scores, top_k, dim=1)
        return torch.softmax(weights, dim=1), ids.to(torch.int32)
    if name == "hotspot":
        probabilities = torch.ones(experts)
        probabilities[: max(1, experts // 8)] = 8.0
        ids = torch.multinomial(probabilities.expand(tokens, -1), top_k, replacement=False, generator=generator)
        weights = torch.rand((tokens, top_k), generator=generator)
        return torch.softmax(weights, dim=1), ids.to(torch.int32)
    raise ValueError(name)


def _measure_pair(baseline, candidate, warmup: int, runs: int):
    for _ in range(warmup):
        baseline()
        candidate()
    samples = {"baseline": [], "candidate": []}
    for index in range(runs):
        order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        for name in order:
            begin = time.perf_counter_ns()
            (baseline if name == "baseline" else candidate)()
            samples[name].append((time.perf_counter_ns() - begin) / 1e6)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=768)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--routed-scaling-factor", type=float, default=2.5)
    parser.add_argument("--swiglu-limit", type=float, default=None)
    parser.add_argument("--cases", default="balanced,uniform,hotspot")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=21)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(args.seed)
    cpu_ids = tuple(range(args.threads))
    baseline_runtime = MoePlannerRuntime(
        args.profile,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        global_experts=args.experts,
        local_experts=args.experts,
        mode="tp",
        degree=args.tp_degree,
        cpu_ids=cpu_ids,
    )
    candidate_runtime = MoePlannerRuntime(
        args.profile,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        global_experts=args.experts,
        local_experts=args.experts,
        mode="tp",
        degree=args.tp_degree,
        cpu_ids=cpu_ids,
        shared_experts=1,
    )

    hidden = torch.empty((args.tokens, args.hidden), dtype=torch.bfloat16).normal_(
        mean=0.0,
        std=0.01,
        generator=generator,
    )
    routed_w13 = torch.empty(
        (args.experts, 2 * args.intermediate, args.hidden),
        dtype=torch.bfloat16,
    ).fill_(0.001)
    routed_w2 = torch.empty(
        (args.experts, args.hidden, args.intermediate),
        dtype=torch.bfloat16,
    ).fill_(0.001)
    shared_w13 = torch.empty((2 * args.intermediate, args.hidden), dtype=torch.bfloat16).fill_(0.001)
    shared_w2 = torch.empty((args.hidden, args.intermediate), dtype=torch.bfloat16).fill_(0.001)
    pack_begin = time.perf_counter_ns()
    weights = prepare_routed_shared_moe_bf16_tiled_weights(
        routed_w13,
        routed_w2,
        shared_w13,
        shared_w2,
    )
    pack_ms = (time.perf_counter_ns() - pack_begin) / 1e6
    del routed_w13, routed_w2, shared_w13, shared_w2
    routed_weights, shared_weights = _views(weights)
    thread_cpu_ids = torch.tensor(cpu_ids, dtype=torch.int32)
    routed_out = torch.empty_like(hidden)
    shared_out = torch.empty_like(hidden)
    baseline_out = torch.empty_like(hidden)
    candidate_out = torch.empty_like(hidden)
    total_flops = 6.0 * args.tokens * (args.top_k + 1) * args.hidden * args.intermediate

    results = []
    for case in [item.strip() for item in args.cases.split(",") if item.strip()]:
        topk_weights, topk_ids = _routes(case, args.tokens, args.experts, args.top_k, generator)
        scaled_topk_weights = topk_weights * args.routed_scaling_factor

        def baseline():
            set_default_moe_planner_runtime(baseline_runtime)
            fused_moe_bf16_tiled(
                hidden,
                routed_weights,
                scaled_topk_weights,
                topk_ids,
                num_threads=args.threads,
                activation="silu",
                swiglu_limit=args.swiglu_limit,
                out=routed_out,
            )
            shared_mlp_bf16_tiled(
                hidden,
                shared_weights,
                thread_cpu_ids=thread_cpu_ids,
                num_threads=args.threads,
                swiglu_limit=args.swiglu_limit,
                out=shared_out,
            )
            torch.add(routed_out, shared_out, out=baseline_out)
            return baseline_out

        def candidate():
            set_default_moe_planner_runtime(candidate_runtime)
            return fused_moe_bf16_tiled_with_shared(
                hidden,
                weights,
                topk_weights,
                topk_ids,
                num_threads=args.threads,
                routed_scaling_factor=args.routed_scaling_factor,
                swiglu_limit=args.swiglu_limit,
                out=candidate_out,
            )

        reference = baseline()
        actual = candidate()
        max_abs = float((actual.float() - reference.float()).abs().max())
        cosine = float(
            torch.nn.functional.cosine_similarity(
                actual.float().flatten().unsqueeze(0),
                reference.float().flatten().unsqueeze(0),
            )
        )
        cold_plan_ms = candidate_runtime.last_plan["planner_overhead_ns"] / 1e6
        samples = _measure_pair(baseline, candidate, args.warmup, args.runs)
        baseline_median = statistics.median(samples["baseline"])
        candidate_median = statistics.median(samples["candidate"])
        counts = torch.bincount(topk_ids.flatten().to(torch.int64), minlength=args.experts)
        result = {
            "case": case,
            "routes_min_median_max": [int(counts.min()), float(counts.float().median()), int(counts.max())],
            "baseline_median_ms": baseline_median,
            "candidate_median_ms": candidate_median,
            "speedup": baseline_median / candidate_median,
            "candidate_tflops": total_flops / (candidate_median / 1e3) / 1e12,
            "baseline_samples_ms": samples["baseline"],
            "candidate_samples_ms": samples["candidate"],
            "max_abs_vs_sequential": max_abs,
            "cosine_vs_sequential": cosine,
            "cold_plan_ms": cold_plan_ms,
            "hit_plan_ms": candidate_runtime.last_plan["planner_overhead_ns"] / 1e6,
            "plan": candidate_runtime.last_plan,
        }
        results.append(result)
        print("CASE_RESULT " + json.dumps(result, default=str, sort_keys=True), flush=True)

    set_default_moe_planner_runtime(None)
    print(
        "SUMMARY "
        + json.dumps(
            {
                "shape": {
                    "tokens": args.tokens,
                    "hidden": args.hidden,
                    "intermediate": args.intermediate,
                    "routed_experts": args.experts,
                    "top_k": args.top_k,
                    "threads": args.threads,
                    "tp_degree": args.tp_degree,
                    "swiglu_limit": args.swiglu_limit,
                },
                "pack_ms": pack_ms,
                "warmup": args.warmup,
                "runs": args.runs,
                "results": results,
            },
            default=str,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
