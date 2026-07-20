# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

from fused_cpp.moe import fused_moe_bf16_tiled
from fused_cpp.moe import fused_moe_naive
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights


_PATTERNS = ("auto", "m1n2", "m2n2", "m1n4")
_SILU_EPILOGUES = ("auto", "baseline", "resident", "pipelined", "rcp14")


def _parse_patterns(raw: str) -> tuple[str, ...]:
    patterns: list[str] = []
    for pattern in raw.split(","):
        if pattern not in _PATTERNS:
            raise argparse.ArgumentTypeError(f"unknown AMX pattern {pattern!r}; expected one of {_PATTERNS}")
        if pattern not in patterns:
            patterns.append(pattern)
    if not patterns:
        raise argparse.ArgumentTypeError("at least one AMX pattern is required")
    return tuple(patterns)


def _parse_silu_epilogues(raw: str) -> tuple[str, ...]:
    epilogues: list[str] = []
    for epilogue in raw.split(","):
        if epilogue not in _SILU_EPILOGUES:
            raise argparse.ArgumentTypeError(
                f"unknown AMX SiLU epilogue {epilogue!r}; expected one of {_SILU_EPILOGUES}",
            )
        if epilogue not in epilogues:
            epilogues.append(epilogue)
    if not epilogues:
        raise argparse.ArgumentTypeError("at least one AMX SiLU epilogue is required")
    return tuple(epilogues)


def _select_variant(pattern: str, silu_epilogue: str) -> None:
    if pattern == "auto":
        os.environ.pop("FUSED_CPP_MOE_AMX_PATTERN", None)
    else:
        os.environ["FUSED_CPP_MOE_AMX_PATTERN"] = pattern
    if silu_epilogue == "auto":
        os.environ.pop("FUSED_CPP_MOE_AMX_SILU_EPILOGUE", None)
    else:
        os.environ["FUSED_CPP_MOE_AMX_SILU_EPILOGUE"] = silu_epilogue


def _routing(tokens: int, experts: int, top_k: int, mode: str) -> torch.Tensor:
    routes = tokens * top_k
    if mode == "hot":
        flat = [0] * routes
    elif mode == "skewed":
        hot_routes = (routes * 3) // 4
        flat = [0] * hot_routes
        if experts == 1:
            flat.extend([0] * (routes - hot_routes))
        else:
            flat.extend(1 + index % (experts - 1) for index in range(routes - hot_routes))
    else:
        flat = [index % experts for index in range(routes)]
    return torch.tensor(flat, dtype=torch.int32).view(tokens, top_k)


def _median_and_best(samples: list[float]) -> tuple[float, float]:
    return statistics.median(samples), min(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare AMX fused-expert JIT tile patterns and SiLU epilogues.")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--routing", choices=("balanced", "hot", "skewed"), default="hot")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--patterns", type=_parse_patterns, default=_PATTERNS)
    parser.add_argument("--silu-epilogues", type=_parse_silu_epilogues, default=("auto",))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=21)
    parser.add_argument("--baseline-runs", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(20260719)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    inputs = torch.empty((args.tokens, args.hidden), dtype=torch.bfloat16).normal_(std=0.01)
    w13 = torch.empty(
        (args.experts, 2 * args.intermediate, args.hidden),
        dtype=torch.bfloat16,
    ).normal_(std=0.01)
    w2 = torch.empty(
        (args.experts, args.hidden, args.intermediate),
        dtype=torch.bfloat16,
    ).normal_(std=0.01)
    topk_ids = _routing(args.tokens, args.experts, args.top_k, args.routing)
    topk_weights = torch.softmax(torch.randn((args.tokens, args.top_k)), dim=-1)
    route_histogram = torch.bincount(topk_ids.flatten().to(torch.int64), minlength=args.experts).tolist()

    pack_start = time.perf_counter_ns()
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_amx_bf16",
    )
    if packed.backend_name != "x86_amx_bf16":
        raise RuntimeError(f"requested x86_amx_bf16, got {packed.backend_name}")
    pack_ms = (time.perf_counter_ns() - pack_start) / 1e6

    def custom() -> torch.Tensor:
        return fused_moe_bf16_tiled(
            inputs,
            packed,
            topk_weights,
            topk_ids,
            num_threads=args.threads,
        )

    torch.set_num_threads(args.threads)
    reference = fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids)
    torch.set_num_threads(1)
    variants = tuple((pattern, epilogue) for pattern in args.patterns for epilogue in args.silu_epilogues)
    labels = {
        variant: variant[0] if args.silu_epilogues == ("auto",) else f"{variant[0]}:{variant[1]}"
        for variant in variants
    }
    max_abs: dict[str, float] = {}
    outputs: dict[str, torch.Tensor] = {}
    for pattern, epilogue in variants:
        _select_variant(pattern, epilogue)
        output = custom()
        label = labels[(pattern, epilogue)]
        outputs[label] = output
        max_abs[label] = float((output.float() - reference.float()).abs().max())
        for _ in range(args.warmup):
            custom()

    samples = {label: [] for label in labels.values()}
    for iteration in range(args.runs):
        offset = iteration % len(variants)
        order = variants[offset:] + variants[:offset]
        for pattern, epilogue in order:
            _select_variant(pattern, epilogue)
            label = labels[(pattern, epilogue)]
            start = time.perf_counter_ns()
            custom()
            samples[label].append((time.perf_counter_ns() - start) / 1e6)

    routes = args.tokens * args.top_k
    flops = float(routes * 6 * args.hidden * args.intermediate)
    reference_label = next(
        (labels[variant] for variant in variants if variant[1] == "baseline"),
        labels[variants[0]],
    )
    pattern_results = {}
    for label in labels.values():
        median_ms, best_ms = _median_and_best(samples[label])
        difference = (outputs[label].float() - outputs[reference_label].float()).abs()
        pattern_results[label] = {
            "median_ms": median_ms,
            "best_ms": best_ms,
            "median_gflops": flops / median_ms / 1e6,
            "best_gflops": flops / best_ms / 1e6,
            "max_abs_vs_torch": max_abs[label],
            "mismatches_vs_reference": int((outputs[label] != outputs[reference_label]).sum()),
            "max_abs_vs_reference": float(difference.max()),
        }

    baseline = None
    if args.baseline_runs > 0:
        torch.set_num_threads(args.threads)
        for _ in range(2):
            fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids)
        baseline_samples = []
        for _ in range(args.baseline_runs):
            start = time.perf_counter_ns()
            fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids)
            baseline_samples.append((time.perf_counter_ns() - start) / 1e6)
        median_ms, best_ms = _median_and_best(baseline_samples)
        baseline = {
            "median_ms": median_ms,
            "best_ms": best_ms,
            "median_gflops": flops / median_ms / 1e6,
            "best_gflops": flops / best_ms / 1e6,
        }

    print(
        json.dumps(
            {
                "shape": {
                    "tokens": args.tokens,
                    "hidden": args.hidden,
                    "intermediate": args.intermediate,
                    "experts": args.experts,
                    "top_k": args.top_k,
                    "routes": routes,
                    "routes_per_expert": route_histogram,
                    "routing": args.routing,
                },
                "threads": args.threads,
                "backend": packed.backend_name,
                "warmup_per_pattern": args.warmup,
                "runs_per_pattern": args.runs,
                "prepack_ms": pack_ms,
                "silu_epilogues": args.silu_epilogues,
                "variant_reference": reference_label,
                "patterns": pattern_results,
                "torch_onednn_staged": baseline,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
