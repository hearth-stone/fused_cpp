# -*- coding: utf-8 -*-
"""Benchmark the BF16 tiled fused MoE kernel."""

from __future__ import annotations

import argparse
import gc
import statistics
import time

import torch

from fused_cpp.moe import _HAS_BF16_TILED_FUSED_MOE
from fused_cpp.moe import fused_moe_bf16_tiled
from fused_cpp.moe import fused_moe_naive
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights


DEEPSEEK_V4_FLASH_HIDDEN_SIZE = 4096
DEEPSEEK_V4_FLASH_MOE_INTERMEDIATE_SIZE = 2048
DEEPSEEK_V4_FLASH_TP_SIZE = 4
DEEPSEEK_V4_FLASH_FFN_PER_RANK = (
    DEEPSEEK_V4_FLASH_MOE_INTERMEDIATE_SIZE // DEEPSEEK_V4_FLASH_TP_SIZE
)
DEEPSEEK_V4_FLASH_ROUTED_EXPERTS = 256
DEEPSEEK_V4_FLASH_TOP_K = 6


def _bf16_normal(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    std: float,
) -> torch.Tensor:
    tensor = torch.empty(shape, dtype=torch.bfloat16)
    return tensor.normal_(mean=0.0, std=std, generator=generator)


def _make_topk(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.rand(num_tokens, num_experts, generator=generator)
    topk_weights, topk_ids = torch.topk(scores, k=top_k, dim=-1)
    return torch.softmax(topk_weights, dim=-1), topk_ids.to(torch.int32)


def _moe_flops(
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    ffn_hidden_size: int,
) -> float:
    num_routes = num_tokens * top_k
    w13 = 2.0 * num_routes * hidden_size * (2 * ffn_hidden_size)
    w2 = 2.0 * num_routes * ffn_hidden_size * hidden_size
    return w13 + w2


def _bench_call(
    name: str,
    run,
    *,
    warmup: int,
    runs: int,
    total_flops: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    out = None
    for _ in range(warmup):
        out = run()
        _ = float(out.flatten()[0])
    gc.collect()

    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = run()
        _ = float(out.flatten()[0])
        t1 = time.perf_counter()
        times.append(t1 - t0)

    if out is None:
        out = run()
    median_s = statistics.median(times)
    mean_s = statistics.mean(times)
    best_s = min(times)
    result = {
        "name": name,
        "median_s": median_s,
        "best_s": best_s,
        "mean_s": mean_s,
        "median_gflops": total_flops / median_s / 1e9,
        "best_gflops": total_flops / best_s / 1e9,
        "times": times,
    }
    return out, result


def _print_result(result: dict[str, object]) -> None:
    name = str(result["name"])
    median_s = float(result["median_s"])
    best_s = float(result["best_s"])
    mean_s = float(result["mean_s"])
    median_gflops = float(result["median_gflops"])
    best_gflops = float(result["best_gflops"])
    times = result["times"]
    assert isinstance(times, list)
    print(
        f"{name}: "
        f"median_ms={median_s * 1e3:.3f} "
        f"best_ms={best_s * 1e3:.3f} "
        f"mean_ms={mean_s * 1e3:.3f} "
        f"median_gflops={median_gflops:.3f} "
        f"best_gflops={best_gflops:.3f} "
        f"times_ms={[round(float(t) * 1e3, 3) for t in times]}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int,
                        default=DEEPSEEK_V4_FLASH_HIDDEN_SIZE)
    parser.add_argument("--ffn-hidden-size", type=int,
                        default=DEEPSEEK_V4_FLASH_FFN_PER_RANK)
    parser.add_argument("--experts", type=int,
                        default=DEEPSEEK_V4_FLASH_ROUTED_EXPERTS)
    parser.add_argument("--top-k", type=int,
                        default=DEEPSEEK_V4_FLASH_TOP_K)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260218)
    parser.add_argument("--activation", choices=("silu", "swigluoai"), default="silu")
    parser.add_argument("--std", type=float, default=0.01)
    parser.add_argument(
        "--baseline",
        choices=("none", "torch"),
        default="none",
        help="Optional baseline to run on the same inputs.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="PyTorch intra-op threads for --baseline torch; defaults to --threads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not _HAS_BF16_TILED_FUSED_MOE:
        raise RuntimeError(
            "BF16 tiled fused MoE backend is unavailable; "
            "build the C++ extension first."
        )
    if args.top_k > args.experts:
        raise ValueError(f"top_k={args.top_k} cannot exceed experts={args.experts}")

    torch.set_num_threads(1)
    torch_threads = args.threads if args.torch_threads is None else args.torch_threads
    generator = torch.Generator().manual_seed(args.seed)

    hidden_states = _bf16_normal(
        (args.tokens, args.hidden_size),
        generator=generator,
        std=args.std,
    )
    w13_weight = _bf16_normal(
        (args.experts, 2 * args.ffn_hidden_size, args.hidden_size),
        generator=generator,
        std=args.std,
    )
    w2_weight = _bf16_normal(
        (args.experts, args.hidden_size, args.ffn_hidden_size),
        generator=generator,
        std=args.std,
    )
    topk_weights, topk_ids = _make_topk(
        args.tokens,
        args.experts,
        args.top_k,
        generator=generator,
    )

    pack_t0 = time.perf_counter()
    packed = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight)
    pack_t1 = time.perf_counter()

    def run_fused() -> torch.Tensor:
        torch.set_num_threads(1)
        return fused_moe_bf16_tiled(
            hidden_states,
            packed,
            topk_weights,
            topk_ids,
            num_threads=args.threads,
            activation=args.activation,
        )

    def run_torch_baseline() -> torch.Tensor:
        torch.set_num_threads(torch_threads)
        return fused_moe_naive(
            hidden_states,
            w13_weight,
            w2_weight,
            topk_weights,
            topk_ids,
            activation=args.activation,
        )

    counts = torch.bincount(topk_ids.flatten().to(torch.int64), minlength=args.experts)
    total_flops = _moe_flops(
        args.tokens,
        args.top_k,
        args.hidden_size,
        args.ffn_hidden_size,
    )
    fused_out, fused_result = _bench_call(
        "fused_bf16_tiled_moe",
        run_fused,
        warmup=args.warmup,
        runs=args.runs,
        total_flops=total_flops,
    )

    print("bf16_tiled_fused_moe benchmark")
    print(
        f"tokens={args.tokens} experts={args.experts} top_k={args.top_k} "
        f"threads={args.threads} activation={args.activation}"
    )
    print(
        f"H={args.hidden_size} F_per_rank={args.ffn_hidden_size} "
        f"w13=[{args.experts},{2 * args.ffn_hidden_size},{args.hidden_size}] "
        f"w2=[{args.experts},{args.hidden_size},{args.ffn_hidden_size}]"
    )
    print(
        f"routes={args.tokens * args.top_k} "
        f"routes_per_expert_min={int(counts.min().item())} "
        f"max={int(counts.max().item())} "
        f"mean={float(counts.float().mean().item()):.2f}"
    )
    print(f"pack_ms={(pack_t1 - pack_t0) * 1e3:.3f}  # not included below")
    print(f"total_flops={total_flops / 1e9:.3f} GF")
    _print_result(fused_result)

    if args.baseline == "torch":
        baseline_out, baseline_result = _bench_call(
            f"torch_fused_moe_naive_baseline(torch_threads={torch_threads})",
            run_torch_baseline,
            warmup=args.warmup,
            runs=args.runs,
            total_flops=total_flops,
        )
        max_diff = float(
            (fused_out.float() - baseline_out.float()).abs().max().item())
        _print_result(baseline_result)
        speedup = (
            float(baseline_result["median_s"]) /
            float(fused_result["median_s"])
        )
        print(f"max_abs_diff_vs_torch_baseline={max_diff:.6f}")
        print(f"speedup_fused_vs_torch_baseline_median={speedup:.3f}x")


if __name__ == "__main__":
    main()
