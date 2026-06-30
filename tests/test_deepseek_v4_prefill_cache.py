# -*- coding: utf-8 -*-
from __future__ import annotations

import pytest
import torch

from fused_cpp.deepseek_v4_prefill_cache import (
    _HAS_DEEPSEEK_V4_PREFILL_CACHE_OPS,
    combine_topk_swa_indices_cpp,
    combine_topk_swa_indices_torch_baseline,
    dequantize_and_gather_dual_k_cache_cpp,
    dequantize_and_gather_dual_k_cache_torch_baseline,
    dequantize_and_gather_k_cache_cpp,
    dequantize_and_gather_k_cache_torch_baseline,
)


def _make_k_cache() -> torch.Tensor:
    values = torch.arange(8 * 4 * 6, dtype=torch.float32).reshape(8, 4, 6)
    return (values * 0.125 - 7.0).to(torch.bfloat16)


def _make_compressed_k_cache() -> torch.Tensor:
    values = torch.arange(6 * 2 * 6, dtype=torch.float32).reshape(6, 2, 6)
    return (values * 0.25 + 3.0).to(torch.bfloat16)


@pytest.mark.skipif(
    not _HAS_DEEPSEEK_V4_PREFILL_CACHE_OPS,
    reason="DeepSeek V4 prefill cache C++ ops are unavailable",
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_dequantize_and_gather_k_cache_cpp_matches_torch_baseline(
    index_dtype: torch.dtype,
) -> None:
    k_cache = _make_k_cache()
    seq_lens = torch.tensor([7, 3, 0], dtype=index_dtype)
    block_table = torch.tensor(
        [
            [2, 5, 0],
            [6, 1, 0],
            [3, 4, 0],
        ],
        dtype=index_dtype,
    )
    ref_out = torch.full((3, 12, 6), -99.0, dtype=torch.bfloat16)
    cpp_out = ref_out.clone()

    dequantize_and_gather_k_cache_torch_baseline(
        ref_out,
        k_cache,
        seq_lens,
        None,
        block_table,
        block_size=4,
        offset=2,
    )
    dequantize_and_gather_k_cache_cpp(
        cpp_out,
        k_cache,
        seq_lens,
        None,
        block_table,
        block_size=4,
        offset=2,
    )

    torch.testing.assert_close(cpp_out, ref_out, rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    not _HAS_DEEPSEEK_V4_PREFILL_CACHE_OPS,
    reason="DeepSeek V4 prefill cache C++ ops are unavailable",
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_dequantize_and_gather_dual_k_cache_cpp_matches_two_baselines(
    index_dtype: torch.dtype,
) -> None:
    compressed_k_cache = _make_compressed_k_cache()
    swa_k_cache = _make_k_cache()
    compressed_seq_lens = torch.tensor([3, 2], dtype=index_dtype)
    swa_seq_lens = torch.tensor([7, 5], dtype=index_dtype)
    swa_gather_lens = torch.tensor([5, 2], dtype=index_dtype)
    compressed_block_table = torch.tensor(
        [
            [1, 2],
            [3, 0],
        ],
        dtype=index_dtype,
    )
    swa_block_table = torch.tensor(
        [
            [2, 5, 0],
            [6, 1, 0],
        ],
        dtype=index_dtype,
    )
    ref_out = torch.full((2, 14, 6), -13.0, dtype=torch.bfloat16)
    cpp_out = ref_out.clone()

    dequantize_and_gather_dual_k_cache_torch_baseline(
        ref_out,
        compressed_k_cache,
        compressed_seq_lens,
        compressed_block_table,
        compressed_block_size=2,
        compressed_offset=1,
        has_compressed=True,
        swa_k_cache=swa_k_cache,
        swa_seq_lens=swa_seq_lens,
        swa_gather_lens=swa_gather_lens,
        swa_block_table=swa_block_table,
        swa_block_size=4,
        swa_offset=8,
    )
    dequantize_and_gather_dual_k_cache_cpp(
        cpp_out,
        compressed_k_cache,
        compressed_seq_lens,
        compressed_block_table,
        compressed_block_size=2,
        compressed_offset=1,
        has_compressed=True,
        swa_k_cache=swa_k_cache,
        swa_seq_lens=swa_seq_lens,
        swa_gather_lens=swa_gather_lens,
        swa_block_table=swa_block_table,
        swa_block_size=4,
        swa_offset=8,
    )

    torch.testing.assert_close(cpp_out, ref_out, rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    not _HAS_DEEPSEEK_V4_PREFILL_CACHE_OPS,
    reason="DeepSeek V4 prefill cache C++ ops are unavailable",
)
def test_dequantize_and_gather_dual_k_cache_cpp_matches_swa_only_baseline() -> None:
    compressed_k_cache = _make_compressed_k_cache()
    swa_k_cache = _make_k_cache()
    compressed_seq_lens = torch.tensor([3, 2], dtype=torch.int64)
    swa_seq_lens = torch.tensor([7, 5], dtype=torch.int64)
    swa_gather_lens = torch.tensor([5, 2], dtype=torch.int64)
    compressed_block_table = torch.tensor([[1, 2], [3, 0]], dtype=torch.int64)
    swa_block_table = torch.tensor([[2, 5, 0], [6, 1, 0]], dtype=torch.int64)
    ref_out = torch.full((2, 12, 6), -21.0, dtype=torch.bfloat16)
    cpp_out = ref_out.clone()

    dequantize_and_gather_dual_k_cache_torch_baseline(
        ref_out,
        compressed_k_cache,
        compressed_seq_lens,
        compressed_block_table,
        compressed_block_size=2,
        compressed_offset=1,
        has_compressed=False,
        swa_k_cache=swa_k_cache,
        swa_seq_lens=swa_seq_lens,
        swa_gather_lens=swa_gather_lens,
        swa_block_table=swa_block_table,
        swa_block_size=4,
        swa_offset=4,
    )
    dequantize_and_gather_dual_k_cache_cpp(
        cpp_out,
        compressed_k_cache,
        compressed_seq_lens,
        compressed_block_table,
        compressed_block_size=2,
        compressed_offset=1,
        has_compressed=False,
        swa_k_cache=swa_k_cache,
        swa_seq_lens=swa_seq_lens,
        swa_gather_lens=swa_gather_lens,
        swa_block_table=swa_block_table,
        swa_block_size=4,
        swa_offset=4,
    )

    torch.testing.assert_close(cpp_out, ref_out, rtol=0.0, atol=0.0)
    torch.testing.assert_close(cpp_out[:, :4], torch.full_like(cpp_out[:, :4], -21.0))


@pytest.mark.skipif(
    not _HAS_DEEPSEEK_V4_PREFILL_CACHE_OPS,
    reason="DeepSeek V4 prefill cache C++ ops are unavailable",
)
def test_dequantize_and_gather_k_cache_cpp_matches_recent_window_baseline() -> None:
    k_cache = _make_k_cache()
    seq_lens = torch.tensor([7, 5], dtype=torch.int64)
    gather_lens = torch.tensor([5, 2], dtype=torch.int32)
    block_table = torch.tensor(
        [
            [2, 5, 0],
            [6, 1, 0],
        ],
        dtype=torch.int32,
    )
    ref_out = torch.full((2, 10, 6), -3.0, dtype=torch.bfloat16)
    cpp_out = ref_out.clone()

    dequantize_and_gather_k_cache_torch_baseline(
        ref_out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size=4,
        offset=3,
    )
    dequantize_and_gather_k_cache_cpp(
        cpp_out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size=4,
        offset=3,
    )

    torch.testing.assert_close(cpp_out, ref_out, rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    not _HAS_DEEPSEEK_V4_PREFILL_CACHE_OPS,
    reason="DeepSeek V4 prefill cache C++ ops are unavailable",
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_combine_topk_swa_indices_cpp_matches_torch_baseline(
    index_dtype: torch.dtype,
) -> None:
    topk_indices = torch.tensor(
        [
            [10, 11, 12, -1, -1],
            [20, 21, 22, -1, -1],
            [30, 31, 32, -1, -1],
            [40, 41, 42, -1, -1],
            [50, 51, 52, -1, -1],
        ],
        dtype=torch.int32,
    )
    query_start_loc = torch.tensor([10, 13, 15], dtype=index_dtype)
    seq_lens = torch.tensor([8, 5], dtype=index_dtype)
    gather_lens = torch.tensor([6, 4], dtype=index_dtype)

    ref_indices, ref_lens = combine_topk_swa_indices_torch_baseline(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size=4,
        compress_ratio=4,
        topk=3,
        M=32,
        N=9,
    )
    cpp_indices, cpp_lens = combine_topk_swa_indices_cpp(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size=4,
        compress_ratio=4,
        topk=3,
        M=32,
        N=9,
    )

    assert torch.equal(cpp_lens, ref_lens)
    assert torch.equal(cpp_indices, ref_indices)


@pytest.mark.skipif(
    not _HAS_DEEPSEEK_V4_PREFILL_CACHE_OPS,
    reason="DeepSeek V4 prefill cache C++ ops are unavailable",
)
def test_combine_topk_swa_indices_cpp_matches_swa_only_baseline() -> None:
    topk_indices = torch.full((4, 5), -1, dtype=torch.int32)
    query_start_loc = torch.tensor([3, 5, 7], dtype=torch.int64)
    seq_lens = torch.tensor([4, 9], dtype=torch.int64)
    gather_lens = torch.tensor([4, 3], dtype=torch.int64)

    ref_indices, ref_lens = combine_topk_swa_indices_torch_baseline(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size=3,
        compress_ratio=4,
        topk=0,
        M=20,
        N=0,
    )
    cpp_indices, cpp_lens = combine_topk_swa_indices_cpp(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size=3,
        compress_ratio=4,
        topk=0,
        M=20,
        N=0,
    )

    assert torch.equal(cpp_lens, ref_lens)
    assert torch.equal(cpp_indices, ref_indices)


@pytest.mark.skipif(
    not _HAS_DEEPSEEK_V4_PREFILL_CACHE_OPS,
    reason="DeepSeek V4 prefill cache C++ ops are unavailable",
)
def test_combine_topk_swa_indices_cpp_matches_empty_baseline() -> None:
    topk_indices = torch.empty((0, 2), dtype=torch.int32)
    query_start_loc = torch.tensor([7], dtype=torch.int64)
    seq_lens = torch.empty((0,), dtype=torch.int64)
    gather_lens = torch.empty((0,), dtype=torch.int64)

    ref_indices, ref_lens = combine_topk_swa_indices_torch_baseline(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size=3,
        compress_ratio=4,
        topk=0,
        M=20,
        N=0,
    )
    cpp_indices, cpp_lens = combine_topk_swa_indices_cpp(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size=3,
        compress_ratio=4,
        topk=0,
        M=20,
        N=0,
    )

    assert torch.equal(cpp_lens, ref_lens)
    assert torch.equal(cpp_indices, ref_indices)


@pytest.mark.skipif(
    not _HAS_DEEPSEEK_V4_PREFILL_CACHE_OPS,
    reason="DeepSeek V4 prefill cache C++ ops are unavailable",
)
def test_combine_topk_swa_indices_cpp_threaded_matches_torch_baseline() -> None:
    query_lens = [96, 71, 128, 37]
    query_start_loc = torch.tensor(
        [13, 13 + 96, 13 + 96 + 71, 13 + 96 + 71 + 128, 13 + sum(query_lens)],
        dtype=torch.int64,
    )
    num_tokens = sum(query_lens)
    topk_width = 16
    topk_indices = torch.arange(
        num_tokens * topk_width, dtype=torch.int32
    ).reshape(num_tokens, topk_width)
    seq_lens = torch.tensor([256, 159, 1024, 49], dtype=torch.int32)
    gather_lens = torch.tensor([180, 120, 384, 20], dtype=torch.int32)

    ref_indices, ref_lens = combine_topk_swa_indices_torch_baseline(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size=64,
        compress_ratio=4,
        topk=12,
        M=2048,
        N=333,
    )

    old_threads = torch.get_num_threads()
    try:
        for threads in (1, 2, 4, 8):
            torch.set_num_threads(threads)
            cpp_indices, cpp_lens = combine_topk_swa_indices_cpp(
                topk_indices,
                query_start_loc,
                seq_lens,
                gather_lens,
                window_size=64,
                compress_ratio=4,
                topk=12,
                M=2048,
                N=333,
            )
            assert torch.equal(cpp_lens, ref_lens)
            assert torch.equal(cpp_indices, ref_indices)
    finally:
        torch.set_num_threads(old_threads)
