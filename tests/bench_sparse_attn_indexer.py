# -*- coding: utf-8 -*-
"""Compare DeepSeek V4 sparse attention indexer versions."""

from __future__ import annotations

import argparse
import statistics
import time
from types import SimpleNamespace

import torch

from fused_cpp import (
    available_sparse_attn_indexer_versions,
    cpu_sparse_attn_indexer_op,
)

DEEPSEEK_V4_FLASH_INPUT_TOKENS = 2048
DEEPSEEK_V4_FLASH_COMPRESS_RATIO = 4
DEEPSEEK_V4_FLASH_INDEX_TOPK = 512
DEEPSEEK_V4_FLASH_INDEX_HEADS = 64
DEEPSEEK_V4_FLASH_INDEX_HEAD_DIM = 128
DEEPSEEK_V4_FLASH_MLA_BLOCK_SIZE = 256


def _make_prefill_metadata(
    *,
    num_tokens: int,
    num_reqs: int,
    model_seq_len: int,
    indexer_seq_len: int,
    compress_ratio: int,
    block_size: int,
) -> SimpleNamespace:
    if num_tokens % num_reqs != 0:
        raise ValueError("num_tokens must be divisible by num_reqs for this benchmark")
    query_len = num_tokens // num_reqs
    if query_len > model_seq_len:
        raise ValueError("per-request query length must not exceed model_seq_len")

    cu_seq_lens = torch.arange(
        0, (num_reqs + 1) * indexer_seq_len, indexer_seq_len, dtype=torch.int32
    )
    starts = []
    ends = []
    for req_idx in range(num_reqs):
        req_base = req_idx * indexer_seq_len
        start_pos = model_seq_len - query_len
        for token_idx in range(query_len):
            pos = start_pos + token_idx
            end = min((pos + 1) // compress_ratio, indexer_seq_len)
            starts.append(req_base)
            ends.append(req_base + end)
    max_blocks = (indexer_seq_len + block_size - 1) // block_size
    block_table = torch.arange(num_reqs * max_blocks, dtype=torch.int32).reshape(num_reqs, max_blocks)
    chunk = SimpleNamespace(
        block_table=block_table,
        cu_seqlen_ks=torch.tensor(starts, dtype=torch.int32),
        cu_seqlen_ke=torch.tensor(ends, dtype=torch.int32),
        cu_seq_lens=cu_seq_lens,
        total_seq_lens=num_reqs * indexer_seq_len,
        token_start=0,
        token_end=num_tokens,
        num_reqs=num_reqs,
    )
    return SimpleNamespace(
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=num_reqs,
        prefill=SimpleNamespace(chunks=[chunk]),
    )


def _time_call(
    *,
    q_quant: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_tokens: int,
    metadata: SimpleNamespace,
    version: str,
    runs: int,
    warmup: int,
) -> tuple[torch.Tensor, float]:
    out = torch.empty((q_quant.shape[0], topk_tokens), dtype=torch.int32)
    for _ in range(warmup):
        cpu_sparse_attn_indexer_op(
            q_quant,
            weights,
            kv_cache,
            out,
            topk_tokens,
            metadata,
            version=version,
        )
    timings = []
    for _ in range(runs):
        out.fill_(12345)
        start = time.perf_counter()
        cpu_sparse_attn_indexer_op(
            q_quant,
            weights,
            kv_cache,
            out,
            topk_tokens,
            metadata,
            version=version,
        )
        timings.append((time.perf_counter() - start) * 1000.0)
    return out.clone(), statistics.median(timings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions", default="torch,auto", help="Comma-separated versions to compare")
    parser.add_argument(
        "--tokens",
        type=int,
        default=DEEPSEEK_V4_FLASH_INPUT_TOKENS,
        help="Total query tokens across requests.",
    )
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument(
        "--model-seq-len",
        type=int,
        default=DEEPSEEK_V4_FLASH_INPUT_TOKENS,
        help="Uncompressed per-request sequence length.",
    )
    parser.add_argument(
        "--compress-ratio",
        type=int,
        default=DEEPSEEK_V4_FLASH_COMPRESS_RATIO,
        help="DeepSeek V4 indexer compression ratio.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Compressed/indexer sequence length override. Defaults to model_seq_len // compress_ratio.",
    )
    parser.add_argument("--heads", type=int, default=DEEPSEEK_V4_FLASH_INDEX_HEADS)
    parser.add_argument("--head-dim", type=int, default=DEEPSEEK_V4_FLASH_INDEX_HEAD_DIM)
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEEPSEEK_V4_FLASH_MLA_BLOCK_SIZE // DEEPSEEK_V4_FLASH_COMPRESS_RATIO,
        help="Compressed KV cache block size seen by the C++ indexer.",
    )
    parser.add_argument("--topk", type=int, default=DEEPSEEK_V4_FLASH_INDEX_TOPK)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    indexer_seq_len = (
        args.seq_len if args.seq_len is not None else args.model_seq_len // args.compress_ratio
    )
    torch.manual_seed(args.seed)
    metadata = _make_prefill_metadata(
        num_tokens=args.tokens,
        num_reqs=args.requests,
        model_seq_len=args.model_seq_len,
        indexer_seq_len=indexer_seq_len,
        compress_ratio=args.compress_ratio,
        block_size=args.block_size,
    )
    num_blocks = args.requests * ((indexer_seq_len + args.block_size - 1) // args.block_size)
    q_quant = torch.randn(args.tokens, args.heads, args.head_dim, dtype=torch.bfloat16)
    weights = torch.randn(args.tokens, args.heads, dtype=torch.float32)
    kv_cache = torch.randn(num_blocks, args.block_size, args.head_dim, dtype=torch.bfloat16)

    requested_versions = tuple(v.strip() for v in args.versions.split(",") if v.strip())
    available = set(available_sparse_attn_indexer_versions())
    print(
        "sparse_attn_indexer benchmark "
        f"tokens={args.tokens} requests={args.requests} "
        f"model_seq_len={args.model_seq_len} compress_ratio={args.compress_ratio} "
        f"indexer_seq_len={indexer_seq_len} block_size={args.block_size} "
        f"heads={args.heads} head_dim={args.head_dim} topk={args.topk}"
    )
    print(f"available_versions={sorted(available)}")

    ref, ref_ms = _time_call(
        q_quant=q_quant,
        weights=weights,
        kv_cache=kv_cache,
        topk_tokens=args.topk,
        metadata=metadata,
        version="torch",
        runs=args.runs,
        warmup=args.warmup,
    )
    print(f"version=torch median_ms={ref_ms:.3f} mismatches=0")

    for version in requested_versions:
        if version == "torch":
            continue
        normalized = "cpp_v0" if version in {"cpp", "native"} else version
        if normalized not in available:
            print(f"version={version} skipped unavailable")
            continue
        out, median_ms = _time_call(
            q_quant=q_quant,
            weights=weights,
            kv_cache=kv_cache,
            topk_tokens=args.topk,
            metadata=metadata,
            version=version,
            runs=args.runs,
            warmup=args.warmup,
        )
        mismatches = int((out != ref).sum().item())
        print(f"version={version} median_ms={median_ms:.3f} mismatches={mismatches}")


if __name__ == "__main__":
    main()
