#!/usr/bin/env python3
"""Validate and benchmark the DeepSeek-V4 limit-10 SVE JIT epilogue."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

from fused_cpp import _moe_C
from fused_cpp.moe import fused_moe_bf16_tiled
from fused_cpp.moe import fused_moe_bf16_tiled_async
from fused_cpp.moe import fused_moe_bf16_tiled_scheduled
from fused_cpp.moe import fused_moe_bf16_tiled_with_shared
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights
from fused_cpp.moe import prepare_routed_shared_moe_bf16_tiled_weights
from fused_cpp.moe import prepare_shared_mlp_bf16_tiled_weights
from fused_cpp.moe import shared_mlp_bf16_tiled


def _validate_exact_m_tails() -> dict[str, float]:
    generator = torch.Generator().manual_seed(20260818)
    route_counts = list(range(1, 13))
    experts = len(route_counts)
    tokens = sum(route_counts)
    hidden_size = 64
    intermediate_size = 32
    hidden = torch.randn((tokens, hidden_size), dtype=torch.bfloat16, generator=generator)
    w13 = torch.randn(
        (experts, 2 * intermediate_size, hidden_size),
        dtype=torch.bfloat16,
        generator=generator,
    )
    w2 = torch.empty(
        (experts, hidden_size, intermediate_size),
        dtype=torch.bfloat16,
    ).normal_(mean=0.0, std=0.02, generator=generator)
    topk_ids = torch.cat(
        [torch.full((routes,), expert, dtype=torch.int32) for expert, routes in enumerate(route_counts)]
    ).reshape(-1, 1)
    topk_weights = torch.ones((tokens, 1), dtype=torch.float32)
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="arm_sve_bf16",
    )

    unclamped = fused_moe_bf16_tiled(hidden, packed, topk_weights, topk_ids, num_threads=1)
    zero_limit = fused_moe_bf16_tiled(
        hidden,
        packed,
        topk_weights,
        topk_ids,
        num_threads=1,
        swiglu_limit=0.0,
    )
    clamped_by_degree = {
        degree: fused_moe_bf16_tiled(
            hidden,
            packed,
            topk_weights,
            topk_ids,
            num_threads=1,
            silu_poly_degree=degree,
            swiglu_limit=10.0,
        )
        for degree in (4, 5, 6)
    }
    clamped = clamped_by_degree[5]
    if not torch.equal(unclamped, zero_limit):
        raise AssertionError("swiglu_limit=0 changed the standard fused-SiLU result")

    expert_ids = torch.arange(experts, dtype=torch.int32)
    one_thread = torch.ones(experts, dtype=torch.int32)
    scheduled = fused_moe_bf16_tiled_scheduled(
        hidden,
        packed,
        topk_weights,
        topk_ids,
        torch.arange(experts + 1, dtype=torch.int32),
        expert_ids,
        one_thread,
        num_threads=1,
        swiglu_limit=10.0,
    )
    dep_offsets = torch.arange(experts + 1, dtype=torch.int32)
    dep_offsets[1:] -= 1
    asynchronous = fused_moe_bf16_tiled_async(
        hidden,
        packed,
        topk_weights,
        topk_ids,
        expert_ids,
        torch.zeros(experts, dtype=torch.int32),
        one_thread,
        dep_offsets,
        torch.arange(experts - 1, dtype=torch.int32),
        num_threads=1,
        swiglu_limit=10.0,
    )
    if not torch.equal(scheduled, clamped) or not torch.equal(asynchronous, clamped):
        raise AssertionError("scheduled/async bridges did not preserve clamped SwiGLU")

    reference = torch.empty_like(hidden)
    row_begin = 0
    for expert, rows in enumerate(route_counts):
        x = hidden[row_begin : row_begin + rows].float()
        gate, up = (x @ w13[expert].float().T).chunk(2, dim=-1)
        intermediate = (
            torch.nn.functional.silu(gate.clamp(max=10.0))
            * up.clamp(min=-10.0, max=10.0)
        ).to(torch.bfloat16)
        reference[row_begin : row_begin + rows] = (
            intermediate.float() @ w2[expert].float().T
        ).to(torch.bfloat16)
        row_begin += rows

    max_abs_by_degree = {
        degree: float((output.float() - reference.float()).abs().max())
        for degree, output in clamped_by_degree.items()
    }
    max_abs = max(max_abs_by_degree.values())
    max_rel = float(
        ((clamped.float() - reference.float()).abs() / reference.float().abs().clamp_min(1e-4)).max()
    )
    clamp_delta = float((clamped.float() - unclamped.float()).abs().max())
    if max_abs > 0.25 or clamp_delta <= 0.5:
        raise AssertionError(
            f"clamped SwiGLU validation failed: max_abs={max_abs}, clamp_delta={clamp_delta}"
        )

    shared_packed = prepare_shared_mlp_bf16_tiled_weights(w13[0], w2[0])
    shared_hidden = hidden[:12]
    shared = shared_mlp_bf16_tiled(
        shared_hidden,
        shared_packed,
        num_threads=1,
        swiglu_limit=10.0,
    )
    routed = fused_moe_bf16_tiled(
        shared_hidden,
        shared_packed,
        torch.ones((12, 1), dtype=torch.float32),
        torch.zeros((12, 1), dtype=torch.int32),
        num_threads=1,
        swiglu_limit=10.0,
    )
    if not torch.equal(shared, routed):
        raise AssertionError("standalone shared MLP did not propagate swiglu_limit")

    combined_packed = prepare_routed_shared_moe_bf16_tiled_weights(
        w13[:1],
        w2[:1],
        w13[1],
        w2[1],
    )
    combined_ids = torch.tensor([[0, 1]] * 12, dtype=torch.int32)
    combined_weights = torch.ones((12, 2), dtype=torch.float32)
    combined = fused_moe_bf16_tiled_with_shared(
        shared_hidden,
        combined_packed,
        combined_weights[:, :1],
        combined_ids[:, :1],
        num_threads=1,
        swiglu_limit=10.0,
    )
    explicit = fused_moe_bf16_tiled(
        shared_hidden,
        combined_packed.packed,
        combined_weights,
        combined_ids,
        num_threads=1,
        global_num_experts=2,
        swiglu_limit=10.0,
    )
    if not torch.equal(combined, explicit):
        raise AssertionError("combined routed+shared API did not propagate swiglu_limit")

    previous = os.environ.get("FUSED_CPP_MOE_SVE_IMPL")
    os.environ["FUSED_CPP_MOE_SVE_IMPL"] = "asm"
    try:
        try:
            fused_moe_bf16_tiled(
                hidden,
                packed,
                topk_weights,
                topk_ids,
                num_threads=1,
                swiglu_limit=10.0,
            )
        except RuntimeError as error:
            if "requires the SVE JIT" not in str(error):
                raise
        else:
            raise AssertionError("static asm unexpectedly accepted clamped SwiGLU")
    finally:
        if previous is None:
            os.environ.pop("FUSED_CPP_MOE_SVE_IMPL", None)
        else:
            os.environ["FUSED_CPP_MOE_SVE_IMPL"] = previous
    return {
        "max_abs": max_abs,
        "max_abs_by_degree": max_abs_by_degree,
        "max_rel": max_rel,
        "clamp_delta": clamp_delta,
        "shared_exact": True,
        "combined_exact": True,
        "scheduled_async_exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2040)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=768)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=31)
    args = parser.parse_args()

    os.environ["FUSED_CPP_MOE_SVE_IMPL"] = "jit"
    validation = _validate_exact_m_tails()
    generator = torch.Generator().manual_seed(20260819)
    hidden = torch.empty((args.rows, args.hidden), dtype=torch.bfloat16).normal_(
        mean=0.0, std=0.01, generator=generator
    )
    w13 = torch.empty(
        (1, 2 * args.intermediate, args.hidden), dtype=torch.bfloat16
    ).normal_(mean=0.0, std=0.01, generator=generator)
    w2 = torch.empty(
        (1, args.hidden, args.intermediate), dtype=torch.bfloat16
    ).normal_(mean=0.0, std=0.01, generator=generator)
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="arm_sve_bf16",
    )
    topk_ids = torch.zeros((args.rows, 1), dtype=torch.int32)
    topk_weights = torch.ones((args.rows, 1), dtype=torch.float32)

    def standard():
        return fused_moe_bf16_tiled(
            hidden,
            packed,
            topk_weights,
            topk_ids,
            num_threads=args.threads,
        )

    def clamped():
        return fused_moe_bf16_tiled(
            hidden,
            packed,
            topk_weights,
            topk_ids,
            num_threads=args.threads,
            swiglu_limit=10.0,
        )

    for _ in range(args.warmup):
        standard()
        clamped()
    standard_samples = []
    clamped_samples = []
    for iteration in range(args.runs):
        first, second = (standard, clamped) if iteration % 2 == 0 else (clamped, standard)
        for name, function in (("first", first), ("second", second)):
            begin = time.perf_counter_ns()
            function()
            elapsed = (time.perf_counter_ns() - begin) / 1e6
            is_standard = (iteration % 2 == 0 and name == "first") or (
                iteration % 2 == 1 and name == "second"
            )
            (standard_samples if is_standard else clamped_samples).append(elapsed)

    standard_median = statistics.median(standard_samples)
    clamped_median = statistics.median(clamped_samples)
    pure_standard_samples = _moe_C.fused_moe_bench_sve_jit_w13_gemm(
        hidden,
        packed.w13[0],
        packed.w13[1],
        packed.w13[2],
        packed.backend_n_tile,
        2,
        args.warmup,
        args.runs,
        11,
        False,
    )
    pure_clamped_samples = _moe_C.fused_moe_bench_sve_jit_w13_gemm(
        hidden,
        packed.w13[0],
        packed.w13[1],
        packed.w13[2],
        packed.backend_n_tile,
        2,
        args.warmup,
        args.runs,
        11,
        True,
    )
    pure_standard_median = statistics.median(pure_standard_samples)
    pure_clamped_median = statistics.median(pure_clamped_samples)
    print(
        json.dumps(
            {
                "shape": vars(args),
                "validation": validation,
                "standard_median_ms": standard_median,
                "clamped_median_ms": clamped_median,
                "clamped_overhead": clamped_median / standard_median - 1.0,
                "pure_w13_standard_median_ms": pure_standard_median,
                "pure_w13_clamped_median_ms": pure_clamped_median,
                "pure_w13_clamped_overhead": pure_clamped_median / pure_standard_median - 1.0,
                "standard_samples_ms": standard_samples,
                "clamped_samples_ms": clamped_samples,
                "pure_w13_standard_samples_ms": pure_standard_samples,
                "pure_w13_clamped_samples_ms": pure_clamped_samples,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
