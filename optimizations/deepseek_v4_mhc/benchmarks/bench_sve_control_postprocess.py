#!/usr/bin/env python3
"""Compare Torch and native-SVE mHC control postprocessing."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections.abc import Callable

import torch

from fused_cpp.deepseek_v4_mhc import (
    _mhc_pre_postprocess,
    _native_sve_control_postprocess,
    _native_sve_post,
    _native_sve_pre_apply_rmsnorm,
    _native_sve_projection,
    _rmsnorm_from_bf16,
    mhc_pre_rmsnorm_sve_candidate,
    prepare_mhc_weight,
)


def _summary(samples: list[float]) -> dict[str, float]:
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "min_ms": samples[0],
        "max_ms": samples[-1],
    }


def _measure_pair(
    first: Callable[[], object], second: Callable[[], object], warmup: int, runs: int
) -> tuple[dict[str, float], dict[str, float]]:
    for _ in range(warmup):
        first()
        second()
    gc.collect()
    first_samples = []
    second_samples = []
    for iteration in range(runs):
        functions = ((first, first_samples), (second, second_samples))
        if iteration % 2 != 0:
            functions = tuple(reversed(functions))
        for function, samples in functions:
            start = time.perf_counter_ns()
            function()
            samples.append((time.perf_counter_ns() - start) / 1e6)
    return _summary(first_samples), _summary(second_samples)


def _max_abs(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]) -> list[float]:
    return [float((left.float() - right.float()).abs().max()) for left, right in zip(actual, expected, strict=True)]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    if min(args.tokens, args.hidden_size, args.threads, args.runs) <= 0 or args.warmup < 0:
        raise ValueError("tokens/hidden-size/threads/runs must be positive and warmup non-negative")
    if _native_sve_projection is None or _native_sve_control_postprocess is None or _native_sve_post is None:
        raise RuntimeError("the SVE mHC projection/control/post candidate is unavailable")
    torch.set_num_threads(args.threads)

    generator = torch.Generator().manual_seed(args.seed)
    t, c, h = args.tokens, 4, args.hidden_size
    residual = torch.randn((t, c, h), generator=generator).bfloat16().contiguous()
    layer_output = torch.randn((t, h), generator=generator).bfloat16().contiguous()
    fn = (torch.randn((24, c * h), generator=generator) * 1e-4).contiguous()
    hc_scale = (torch.randn((3,), generator=generator) * 0.1).contiguous()
    hc_base = (torch.randn((24,), generator=generator) * 0.1).contiguous()
    norm_weight = torch.randn((h,), generator=generator).bfloat16().contiguous()
    prepared = prepare_mhc_weight(fn, kind="pre")
    mixes, sqrsum, _, _ = _native_sve_projection(residual, prepared.packed, args.threads, 1 << 20)
    normalized_mixes = mixes * torch.rsqrt(sqrsum.reshape(t, 1) / (c * h) + 1e-6)

    def torch_control() -> tuple[torch.Tensor, ...]:
        pre = torch.sigmoid(normalized_mixes[:, :4] * hc_scale[0] + hc_base[:4]) + 1e-6
        post = (torch.sigmoid(normalized_mixes[:, 4:8] * hc_scale[1] + hc_base[4:8]) * 2.0).reshape(t, 4, 1)
        comb = normalized_mixes[:, 8:].reshape(t, 4, 4) * hc_scale[2] + hc_base[8:].reshape(1, 4, 4)
        comb = torch.softmax(comb, dim=-1) + 1e-6
        comb = comb / (comb.sum(dim=-2, keepdim=True) + 1e-6)
        for _ in range(19):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + 1e-6)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + 1e-6)
        return pre, post, comb

    def sve_control() -> tuple[torch.Tensor, ...]:
        return _native_sve_control_postprocess(
            mixes, sqrsum, hc_scale, hc_base, c * h, 1e-6, 1e-6, 2.0, 1e-6, 20, args.threads
        )

    pre_for_apply, post_for_post, comb_for_post = torch_control()

    def torch_post() -> torch.Tensor:
        mixed = torch.bmm(comb_for_post.transpose(1, 2), residual.float())
        return (mixed + post_for_post * layer_output.float().unsqueeze(1)).bfloat16()

    def sve_post() -> torch.Tensor:
        return _native_sve_post(layer_output, residual, post_for_post, comb_for_post, args.threads)

    def torch_pre_apply() -> torch.Tensor:
        raw = (pre_for_apply.unsqueeze(-1) * residual.float()).sum(dim=1).to(torch.bfloat16)
        return _rmsnorm_from_bf16(raw, norm_weight, 1e-6)

    def sve_pre_apply() -> torch.Tensor:
        assert _native_sve_pre_apply_rmsnorm is not None
        return _native_sve_pre_apply_rmsnorm(residual, pre_for_apply, norm_weight, 1e-6, args.threads)

    def old_full_pre() -> tuple[torch.Tensor, ...]:
        old_mixes, old_sqrsum, _, _ = _native_sve_projection(residual, prepared.packed, args.threads, 1 << 20)
        old_normalized = old_mixes * torch.rsqrt(old_sqrsum.reshape(t, 1) / (c * h) + 1e-6)
        return _mhc_pre_postprocess(
            residual,
            old_normalized,
            hc_scale,
            hc_base,
            norm_weight,
            hc_pre_eps=1e-6,
            hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=20,
            norm_eps=1e-6,
        )

    def new_full_pre() -> tuple[torch.Tensor, ...]:
        return mhc_pre_rmsnorm_sve_candidate(
            residual, prepared, hc_scale, hc_base, norm_weight, num_threads=args.threads
        )

    torch_control_output = torch_control()
    sve_control_output = sve_control()
    torch_pre_apply_output = torch_pre_apply()
    sve_pre_apply_output = sve_pre_apply()
    torch_post_output = torch_post()
    sve_post_output = sve_post()
    old_full_output = old_full_pre()
    new_full_output = new_full_pre()
    torch_control_timing, sve_control_timing = _measure_pair(torch_control, sve_control, args.warmup, args.runs)
    torch_pre_apply_timing, sve_pre_apply_timing = _measure_pair(torch_pre_apply, sve_pre_apply, args.warmup, args.runs)
    torch_post_timing, sve_post_timing = _measure_pair(torch_post, sve_post, args.warmup, args.runs)
    old_full_timing, new_full_timing = _measure_pair(old_full_pre, new_full_pre, args.warmup, args.runs)
    result = {
        "shape": {"tokens": t, "hc_mult": c, "hidden_size": h},
        "threads": args.threads,
        "warmup": args.warmup,
        "runs": args.runs,
        "control": {
            "torch": torch_control_timing,
            "sve": sve_control_timing,
            "max_abs_pre_post_comb": _max_abs(sve_control_output, torch_control_output),
        },
        "pre_apply_rmsnorm": {
            "torch": torch_pre_apply_timing,
            "sve": sve_pre_apply_timing,
            "max_abs_normed_input": float((sve_pre_apply_output.float() - torch_pre_apply_output.float()).abs().max()),
        },
        "post_k4_rank1": {
            "torch": torch_post_timing,
            "sve": sve_post_timing,
            "max_abs_bf16": float((sve_post_output.float() - torch_post_output.float()).abs().max()),
        },
        "full_pre": {
            "projection_plus_torch_postprocess": old_full_timing,
            "projection_plus_sve_postprocess": new_full_timing,
            "max_abs_post_comb_normed": _max_abs(new_full_output, old_full_output),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
