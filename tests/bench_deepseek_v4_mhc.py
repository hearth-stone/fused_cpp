#!/usr/bin/env python3
"""Benchmark and compare DeepSeek V4 mHC implementations against Torch."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections.abc import Callable, Sequence

import torch

from fused_cpp.deepseek_v4_mhc import (
    _HAS_DEEPSEEK_V4_MHC_SVE_PROJECTION,
    mhc_post_hc_head_rmsnorm,
    mhc_post_hc_head_rmsnorm_sve_candidate,
    mhc_post_hc_head_rmsnorm_torch_baseline,
    mhc_post_pre_rmsnorm,
    mhc_post_pre_rmsnorm_sve_candidate,
    mhc_post_pre_rmsnorm_torch_baseline,
    mhc_pre_rmsnorm,
    mhc_pre_rmsnorm_sve_candidate,
    mhc_pre_rmsnorm_torch_baseline,
    prepare_mhc_weight,
)


TensorOutputs = torch.Tensor | tuple[torch.Tensor, ...]


def _measure(function: Callable[[], TensorOutputs], warmup: int, runs: int) -> dict[str, float]:
    for _ in range(warmup):
        function()
    gc.collect()
    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "min_ms": samples[0],
        "max_ms": samples[-1],
    }


def _tensor_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float | int]:
    actual_fp32 = actual.float().reshape(-1)
    reference_fp32 = reference.float().reshape(-1)
    if actual_fp32.numel() == 0:
        return {
            "max_abs": 0.0,
            "relative_l2": 0.0,
            "cosine_similarity": 1.0,
            "actual_nan": 0,
            "actual_inf": 0,
            "reference_nan": 0,
            "reference_inf": 0,
        }
    delta = actual_fp32 - reference_fp32
    denominator = max(float(torch.linalg.vector_norm(reference_fp32)), 1e-30)
    cosine = torch.nn.functional.cosine_similarity(actual_fp32, reference_fp32, dim=0, eps=1e-30).clamp(-1.0, 1.0)
    return {
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(delta)) / denominator,
        "cosine_similarity": float(cosine),
        "actual_nan": int(torch.isnan(actual_fp32).sum()),
        "actual_inf": int(torch.isinf(actual_fp32).sum()),
        "reference_nan": int(torch.isnan(reference_fp32).sum()),
        "reference_inf": int(torch.isinf(reference_fp32).sum()),
    }


def _output_metrics(
    actual: TensorOutputs,
    reference: TensorOutputs,
    names: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    actual_values: Sequence[torch.Tensor] = actual if isinstance(actual, tuple) else (actual,)
    reference_values: Sequence[torch.Tensor] = reference if isinstance(reference, tuple) else (reference,)
    if len(actual_values) != len(reference_values) or len(actual_values) != len(names):
        raise RuntimeError("candidate, reference, and output-name counts differ")
    return {
        name: _tensor_metrics(actual_value, reference_value)
        for name, actual_value, reference_value in zip(names, actual_values, reference_values, strict=True)
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hc-mult", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--candidate", choices=("auto", "torch", "sve"), default="auto")
    parser.add_argument("--b-window-bytes", type=int, default=1 << 20)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    if min(args.tokens, args.threads, args.warmup) < 0 or min(args.hc_mult, args.hidden_size, args.runs) <= 0:
        raise ValueError("tokens/threads/warmup must be non-negative and hc-mult/hidden-size/runs positive")
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    generator = torch.Generator().manual_seed(args.seed)
    t, c, h = args.tokens, args.hc_mult, args.hidden_size
    p = 2 * c + c * c
    residual = torch.randn((t, c, h), generator=generator).bfloat16().contiguous()
    layer_output = torch.randn((t, h), generator=generator).bfloat16().contiguous()
    pre_fn = (torch.randn((p, c * h), generator=generator) * 1e-4).contiguous()
    head_fn = (torch.randn((c, c * h), generator=generator) * 1e-4).contiguous()
    hc_scale = (torch.randn((3,), generator=generator) * 0.1).contiguous()
    hc_base = (torch.randn((p,), generator=generator) * 0.1).contiguous()
    head_scale = (torch.randn((1,), generator=generator) * 0.1).contiguous()
    head_base = (torch.randn((c,), generator=generator) * 0.1).contiguous()
    norm_weight = torch.randn((h,), generator=generator).bfloat16().contiguous()
    prepared_pre = prepare_mhc_weight(pre_fn, kind="pre")
    prepared_head = prepare_mhc_weight(head_fn, kind="head")
    selected_candidate = args.candidate
    if selected_candidate == "auto":
        selected_candidate = "sve" if _HAS_DEEPSEEK_V4_MHC_SVE_PROJECTION else "torch"
    if selected_candidate == "sve" and not _HAS_DEEPSEEK_V4_MHC_SVE_PROJECTION:
        raise RuntimeError("--candidate sve requires an SVE VL128/VL256 build")

    def baseline_pre() -> TensorOutputs:
        return mhc_pre_rmsnorm_torch_baseline(
            residual, prepared_pre, hc_scale, hc_base, norm_weight, num_threads=args.threads
        )

    def candidate_pre() -> TensorOutputs:
        if selected_candidate == "sve":
            return mhc_pre_rmsnorm_sve_candidate(
                residual,
                prepared_pre,
                hc_scale,
                hc_base,
                norm_weight,
                num_threads=args.threads,
                b_window_bytes=args.b_window_bytes,
            )
        return mhc_pre_rmsnorm(residual, prepared_pre, hc_scale, hc_base, norm_weight, num_threads=args.threads)

    previous_post, previous_comb, _ = baseline_pre()

    def baseline_post_pre() -> TensorOutputs:
        return mhc_post_pre_rmsnorm_torch_baseline(
            layer_output,
            residual,
            previous_post,
            previous_comb,
            prepared_pre,
            hc_scale,
            hc_base,
            norm_weight,
            num_threads=args.threads,
        )

    def candidate_post_pre() -> TensorOutputs:
        if selected_candidate == "sve":
            return mhc_post_pre_rmsnorm_sve_candidate(
                layer_output,
                residual,
                previous_post,
                previous_comb,
                prepared_pre,
                hc_scale,
                hc_base,
                norm_weight,
                num_threads=args.threads,
                b_window_bytes=args.b_window_bytes,
            )
        return mhc_post_pre_rmsnorm(
            layer_output,
            residual,
            previous_post,
            previous_comb,
            prepared_pre,
            hc_scale,
            hc_base,
            norm_weight,
            num_threads=args.threads,
        )

    def baseline_tail() -> TensorOutputs:
        return mhc_post_hc_head_rmsnorm_torch_baseline(
            layer_output,
            residual,
            previous_post,
            previous_comb,
            prepared_head,
            head_scale,
            head_base,
            norm_weight,
            num_threads=args.threads,
        )

    def candidate_tail() -> TensorOutputs:
        if selected_candidate == "sve":
            return mhc_post_hc_head_rmsnorm_sve_candidate(
                layer_output,
                residual,
                previous_post,
                previous_comb,
                prepared_head,
                head_scale,
                head_base,
                norm_weight,
                num_threads=args.threads,
            )
        return mhc_post_hc_head_rmsnorm(
            layer_output,
            residual,
            previous_post,
            previous_comb,
            prepared_head,
            head_scale,
            head_base,
            norm_weight,
            num_threads=args.threads,
        )

    stages = {
        "pre_rmsnorm": (
            baseline_pre,
            candidate_pre,
            ("post_mix", "comb_mix", "normed_input"),
        ),
        "post_pre_rmsnorm": (
            baseline_post_pre,
            candidate_post_pre,
            ("residual", "post_mix", "comb_mix", "normed_input"),
        ),
        "post_hc_head_rmsnorm": (
            baseline_tail,
            candidate_tail,
            ("hidden_states", "final_residual"),
        ),
    }
    result: dict[str, object] = {
        "shape": {"tokens": t, "hc_mult": c, "hidden_size": h},
        "configuration": {
            "requested_threads": args.threads,
            "torch_threads": torch.get_num_threads(),
            "warmup": args.warmup,
            "runs": args.runs,
            "seed": args.seed,
            "candidate": selected_candidate,
            "b_window_bytes": args.b_window_bytes,
        },
        "stages": {},
    }
    stage_results: dict[str, object] = {}
    for name, (baseline, candidate, output_names) in stages.items():
        reference_output = baseline()
        candidate_output = candidate()
        baseline_timing = _measure(baseline, args.warmup, args.runs)
        candidate_timing = _measure(candidate, args.warmup, args.runs)
        stage_results[name] = {
            "accuracy": _output_metrics(candidate_output, reference_output, output_names),
            "torch_baseline": baseline_timing,
            "candidate": candidate_timing,
            "speedup": baseline_timing["median_ms"] / candidate_timing["median_ms"],
        }
    result["stages"] = stage_results
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
