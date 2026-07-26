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
_W2_EPILOGUES = ("auto", "baseline", "combined", "tile_store")
_TILE_STATES = ("auto", "per_call", "macro_m")


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


def _parse_w2_epilogues(raw: str) -> tuple[str, ...]:
    epilogues: list[str] = []
    for epilogue in raw.split(","):
        if epilogue not in _W2_EPILOGUES:
            raise argparse.ArgumentTypeError(
                f"unknown AMX W2 epilogue {epilogue!r}; expected one of {_W2_EPILOGUES}",
            )
        if epilogue not in epilogues:
            epilogues.append(epilogue)
    if not epilogues:
        raise argparse.ArgumentTypeError("at least one AMX W2 epilogue is required")
    return tuple(epilogues)


def _parse_tile_states(raw: str) -> tuple[str, ...]:
    tile_states: list[str] = []
    for tile_state in raw.split(","):
        if tile_state not in _TILE_STATES:
            raise argparse.ArgumentTypeError(
                f"unknown AMX tile state {tile_state!r}; expected one of {_TILE_STATES}",
            )
        if tile_state not in tile_states:
            tile_states.append(tile_state)
    if not tile_states:
        raise argparse.ArgumentTypeError("at least one AMX tile state is required")
    return tuple(tile_states)


def _select_variant(pattern: str, silu_epilogue: str, w2_epilogue: str, tile_state: str) -> None:
    if pattern == "auto":
        os.environ.pop("FUSED_CPP_MOE_AMX_PATTERN", None)
    else:
        os.environ["FUSED_CPP_MOE_AMX_PATTERN"] = pattern
    if silu_epilogue == "auto":
        os.environ.pop("FUSED_CPP_MOE_AMX_SILU_EPILOGUE", None)
    else:
        os.environ["FUSED_CPP_MOE_AMX_SILU_EPILOGUE"] = silu_epilogue
    if w2_epilogue == "auto":
        os.environ.pop("FUSED_CPP_MOE_AMX_W2_EPILOGUE", None)
    else:
        os.environ["FUSED_CPP_MOE_AMX_W2_EPILOGUE"] = w2_epilogue
    if tile_state == "auto":
        os.environ.pop("FUSED_CPP_MOE_AMX_TILE_STATE", None)
    else:
        os.environ["FUSED_CPP_MOE_AMX_TILE_STATE"] = tile_state


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


def _percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _latency_summary(samples: list[float]) -> dict[str, float]:
    median_ms, best_ms = _median_and_best(samples)
    return {
        "median_ms": median_ms,
        "best_ms": best_ms,
        "p90_ms": _percentile(samples, 0.90),
        "p99_ms": _percentile(samples, 0.99),
        "mean_ms": statistics.mean(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare AMX fused-expert JIT tile patterns, epilogues, and tile-state lifetimes.",
    )
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--routing", choices=("balanced", "hot", "skewed"), default="hot")
    parser.add_argument("--threads", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--patterns", type=_parse_patterns, default=_PATTERNS)
    parser.add_argument("--silu-epilogues", type=_parse_silu_epilogues, default=("auto",))
    parser.add_argument("--w2-epilogues", type=_parse_w2_epilogues, default=("auto",))
    parser.add_argument("--tile-states", type=_parse_tile_states, default=("auto",))
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
    variants = tuple(
        (pattern, silu_epilogue, w2_epilogue, tile_state)
        for pattern in args.patterns
        for silu_epilogue in args.silu_epilogues
        for w2_epilogue in args.w2_epilogues
        for tile_state in args.tile_states
    )
    labels = {
        variant: (
            variant[0]
            if args.silu_epilogues == ("auto",)
            and args.w2_epilogues == ("auto",)
            and args.tile_states == ("auto",)
            else f"{variant[0]}:silu={variant[1]}:w2={variant[2]}:tile={variant[3]}"
        )
        for variant in variants
    }
    max_abs: dict[str, float] = {}
    outputs: dict[str, torch.Tensor] = {}
    for pattern, silu_epilogue, w2_epilogue, tile_state in variants:
        _select_variant(pattern, silu_epilogue, w2_epilogue, tile_state)
        output = custom()
        label = labels[(pattern, silu_epilogue, w2_epilogue, tile_state)]
        outputs[label] = output
        max_abs[label] = float((output.float() - reference.float()).abs().max())
        for _ in range(args.warmup):
            custom()

    samples = {label: [] for label in labels.values()}
    for iteration in range(args.runs):
        offset = iteration % len(variants)
        order = variants[offset:] + variants[:offset]
        for pattern, silu_epilogue, w2_epilogue, tile_state in order:
            _select_variant(pattern, silu_epilogue, w2_epilogue, tile_state)
            label = labels[(pattern, silu_epilogue, w2_epilogue, tile_state)]
            start = time.perf_counter_ns()
            custom()
            samples[label].append((time.perf_counter_ns() - start) / 1e6)

    routes = args.tokens * args.top_k
    flops = float(routes * 6 * args.hidden * args.intermediate)
    reference_silu = "baseline" if "baseline" in args.silu_epilogues else args.silu_epilogues[0]
    reference_w2 = "baseline" if "baseline" in args.w2_epilogues else args.w2_epilogues[0]
    reference_tile_state = "per_call" if "per_call" in args.tile_states else args.tile_states[0]
    reference_label = next(
        labels[variant]
        for variant in variants
        if variant[1] == reference_silu and variant[2] == reference_w2 and variant[3] == reference_tile_state
    )
    pattern_results = {}
    for label in labels.values():
        latency = _latency_summary(samples[label])
        median_ms = latency["median_ms"]
        best_ms = latency["best_ms"]
        difference = (outputs[label].float() - outputs[reference_label].float()).abs()
        pattern_results[label] = {
            **latency,
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
        latency = _latency_summary(baseline_samples)
        median_ms = latency["median_ms"]
        best_ms = latency["best_ms"]
        baseline = {
            **latency,
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
                "w2_epilogues": args.w2_epilogues,
                "tile_states": args.tile_states,
                "variant_reference": reference_label,
                "patterns": pattern_results,
                "torch_onednn_staged": baseline,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
