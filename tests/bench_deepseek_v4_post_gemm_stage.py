#!/usr/bin/env python3
"""Benchmark the DeepSeek V4 C4A post-GEMM stage on CPU."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

from fused_cpp.deepseek_v4_post_gemm_stage import (
    CompressorState,
    PostGemmStageInputs,
    PreparedDeepSeekV4PostGemmWeights,
    SWACacheState,
    SparseIndexerPrefillMetadata,
    post_gemm_parallel_stage_cpp_prepacked,
    prepare_deepseek_v4_post_gemm_weights,
)


def _cos_sin_cache(max_pos: int, rope_dim: int) -> torch.Tensor:
    half = rope_dim // 2
    positions = torch.arange(max_pos, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.arange(half, dtype=torch.float32).unsqueeze(0)
    angles = positions * 0.0001 + frequencies * 0.031
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=1).contiguous()


def _bf16_random(*shape: int) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.bfloat16).uniform_(-0.1, 0.1)


def _make_compressor(
    *,
    num_tokens: int,
    head_dim: int,
    state_width: int,
    compress_ratio: int,
    block_size: int,
) -> CompressorState:
    state_blocks = (num_tokens + block_size - 1) // block_size
    compressed_tokens = (num_tokens + compress_ratio - 1) // compress_ratio
    kv_blocks = (compressed_tokens + block_size - 1) // block_size
    positions = torch.arange(num_tokens, dtype=torch.int64)
    active = (positions + 1).remainder(compress_ratio).eq(0)
    kv_slots = torch.where(
        active,
        positions.div(compress_ratio, rounding_mode="floor"),
        torch.full_like(positions, -1),
    )
    return CompressorState(
        ape=torch.empty(compress_ratio, state_width, dtype=torch.float32).uniform_(-0.01, 0.01),
        state_cache=torch.zeros(
            state_blocks,
            block_size,
            2 * state_width,
            dtype=torch.float32,
        ),
        state_slot_mapping=positions.clone(),
        token_to_req_indices=torch.zeros(num_tokens, dtype=torch.int64),
        block_table=torch.arange(state_blocks, dtype=torch.int32).view(1, state_blocks),
        kv_cache=torch.zeros(kv_blocks, block_size, head_dim, dtype=torch.bfloat16),
        kv_slot_mapping=kv_slots,
        norm_weight=torch.ones(head_dim, dtype=torch.float32),
        compress_ratio=compress_ratio,
        rms_norm_eps=1e-6,
    )


def _make_inputs(num_tokens: int) -> tuple[PostGemmStageInputs, PreparedDeepSeekV4PostGemmWeights]:
    torch.manual_seed(20260729)
    q_lora_rank = 1024
    main_num_heads = 16
    main_head_dim = 512
    indexer_num_heads = 64
    indexer_head_dim = 128
    rope_dim = 64
    compress_ratio = 4
    block_size = 64
    topk_tokens = 512
    main_state_width = 2 * main_head_dim
    indexer_state_width = indexer_head_dim

    qr = _bf16_random(num_tokens, q_lora_rank)
    main_weight = _bf16_random(main_num_heads * main_head_dim, q_lora_rank)
    indexer_weight = _bf16_random(indexer_num_heads * indexer_head_dim, q_lora_rank)
    weights = prepare_deepseek_v4_post_gemm_weights(main_weight, indexer_weight)

    positions = torch.arange(num_tokens, dtype=torch.int64)
    state_blocks = (num_tokens + block_size - 1) // block_size
    compressed_tokens = (num_tokens + compress_ratio - 1) // compress_ratio
    compressed_blocks = (compressed_tokens + block_size - 1) // block_size
    valid_compressed = (positions + 1).div(compress_ratio, rounding_mode="floor")

    inputs = PostGemmStageInputs(
        qr=qr,
        kv=_bf16_random(num_tokens, main_head_dim),
        kv_score=torch.empty(num_tokens, 2 * main_state_width, dtype=torch.float32).uniform_(-0.1, 0.1),
        indexer_kv_score=torch.empty(
            num_tokens,
            2 * indexer_state_width,
            dtype=torch.float32,
        ).uniform_(-0.1, 0.1),
        indexer_weights=_bf16_random(num_tokens, indexer_num_heads),
        positions=positions,
        main_wq_b_weight=main_weight,
        indexer_wq_b_weight=indexer_weight,
        main_cos_sin_cache=_cos_sin_cache(num_tokens + 1, rope_dim),
        indexer_cos_sin_cache=_cos_sin_cache(num_tokens + 1, rope_dim),
        swa=SWACacheState(
            kv_cache=torch.zeros(
                state_blocks,
                block_size,
                main_head_dim,
                dtype=torch.bfloat16,
            ),
            slot_mapping=positions.clone(),
        ),
        mla_compressor=_make_compressor(
            num_tokens=num_tokens,
            head_dim=main_head_dim,
            state_width=main_state_width,
            compress_ratio=compress_ratio,
            block_size=block_size,
        ),
        indexer_compressor=_make_compressor(
            num_tokens=num_tokens,
            head_dim=indexer_head_dim,
            state_width=indexer_state_width,
            compress_ratio=compress_ratio,
            block_size=block_size,
        ),
        topk_indices_buffer=torch.empty(num_tokens, topk_tokens, dtype=torch.int32),
        prefill=SparseIndexerPrefillMetadata(
            cu_seq_lens=torch.tensor([0, compressed_tokens], dtype=torch.int64),
            cu_seqlen_ks=torch.zeros(num_tokens, dtype=torch.int64),
            cu_seqlen_ke=valid_compressed,
            block_table=torch.arange(compressed_blocks, dtype=torch.int32).view(1, compressed_blocks),
            topk_tokens=topk_tokens,
        ),
        main_head_dim=main_head_dim,
        q_eps=1e-6,
    )
    return inputs, weights


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--schedule", choices=("legacy", "m8"), default="m8")
    parser.add_argument("--q-pool", choices=("auto", "legacy", "shared"), default="auto")
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ["FUSED_CPP_POST_GEMM_M8_ALIGNED"] = "1" if args.schedule == "m8" else "0"
    if args.q_pool == "auto":
        os.environ.pop("FUSED_CPP_POST_GEMM_SHARED_Q_POOL", None)
    else:
        os.environ["FUSED_CPP_POST_GEMM_SHARED_Q_POOL"] = "1" if args.q_pool == "shared" else "0"
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    inputs, weights = _make_inputs(args.m)

    with torch.inference_mode():
        for _ in range(args.warmup):
            post_gemm_parallel_stage_cpp_prepacked(inputs, weights)

        samples_ms = []
        for _ in range(args.runs):
            start_ns = time.perf_counter_ns()
            post_gemm_parallel_stage_cpp_prepacked(inputs, weights)
            samples_ms.append((time.perf_counter_ns() - start_ns) / 1e6)

        if args.profile:
            os.environ["FUSED_CPP_DEEPSEEK_V4_POST_GEMM_PROFILE"] = "1"
            post_gemm_parallel_stage_cpp_prepacked(inputs, weights)

    median_ms = statistics.median(samples_ms)
    gemm_flops = 4 * args.m * 1024 * 8192
    result = {
        "m": args.m,
        "threads": args.threads,
        "schedule": args.schedule,
        "q_pool": args.q_pool,
        "warmup": args.warmup,
        "runs": args.runs,
        "median_ms": median_ms,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "gemm_equiv_tflops": gemm_flops / (median_ms * 1e9),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
