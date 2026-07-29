# -*- coding: utf-8 -*-
from __future__ import annotations

import platform

import pytest
import torch

from fused_cpp.deepseek_v4_post_gemm_stage import (
    CompressorState,
    PostGemmStageInputs,
    SWACacheState,
    SparseIndexerPrefillMetadata,
    _HAS_DEEPSEEK_V4_POST_GEMM_C128A_PREPACKED,
    _HAS_DEEPSEEK_V4_POST_GEMM_DENSE_PREPACKED,
    _HAS_DEEPSEEK_V4_POST_GEMM_STAGE,
    _HAS_DEEPSEEK_V4_POST_GEMM_STAGE_PREPACKED,
    post_gemm_parallel_stage_cpp,
    post_gemm_parallel_stage_cpp_prepacked,
    post_gemm_parallel_stage_torch_baseline,
    prepare_deepseek_v4_post_gemm_weights,
)


def _cos_sin_cache(max_pos: int, rope_dim: int) -> torch.Tensor:
    half = rope_dim // 2
    positions = torch.arange(max_pos, dtype=torch.float32).unsqueeze(1)
    freqs = torch.arange(half, dtype=torch.float32).unsqueeze(0)
    angles = positions * 0.17 + freqs * 0.031
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)


def _bf16_randn(*shape: int) -> torch.Tensor:
    return (torch.randn(*shape) * 0.2).to(torch.bfloat16)


def _make_compressor_state(
    *,
    num_tokens: int,
    head_dim: int,
    compress_ratio: int,
    block_size: int,
    num_blocks: int,
) -> CompressorState:
    state_width = head_dim
    return CompressorState(
        ape=torch.randn(compress_ratio, state_width, dtype=torch.float32) * 0.01,
        state_cache=torch.zeros(
            num_blocks,
            block_size,
            2 * state_width,
            dtype=torch.float32,
        ),
        state_slot_mapping=torch.arange(num_tokens, dtype=torch.int64),
        token_to_req_indices=torch.zeros(num_tokens, dtype=torch.int64),
        block_table=torch.arange(num_blocks, dtype=torch.int32).view(1, num_blocks),
        kv_cache=torch.zeros(num_blocks, block_size, head_dim, dtype=torch.bfloat16),
        kv_slot_mapping=torch.arange(num_tokens, dtype=torch.int64),
        norm_weight=(1.0 + torch.randn(head_dim, dtype=torch.float32) * 0.01),
        compress_ratio=compress_ratio,
        rms_norm_eps=1e-6,
    )


def _make_inputs(seed: int = 0, num_tokens: int = 6) -> PostGemmStageInputs:
    torch.manual_seed(seed)
    q_lora_rank = 8
    main_num_heads = 2
    main_head_dim = 6
    indexer_num_heads = 4
    indexer_head_dim = 6
    rope_dim = 2
    block_size = 4
    num_blocks = (num_tokens + block_size - 1) // block_size
    topk_tokens = 2

    return PostGemmStageInputs(
        qr=_bf16_randn(num_tokens, q_lora_rank),
        kv=_bf16_randn(num_tokens, main_head_dim),
        kv_score=torch.randn(num_tokens, 2 * main_head_dim, dtype=torch.float32) * 0.1,
        indexer_kv_score=torch.randn(
            num_tokens,
            2 * indexer_head_dim,
            dtype=torch.float32,
        )
        * 0.1,
        indexer_weights=_bf16_randn(num_tokens, indexer_num_heads),
        positions=torch.arange(num_tokens, dtype=torch.int64),
        main_wq_b_weight=_bf16_randn(main_num_heads * main_head_dim, q_lora_rank),
        indexer_wq_b_weight=_bf16_randn(
            indexer_num_heads * indexer_head_dim,
            q_lora_rank,
        ),
        main_cos_sin_cache=_cos_sin_cache(max(16, num_tokens + 1), rope_dim),
        indexer_cos_sin_cache=_cos_sin_cache(max(16, num_tokens + 1), rope_dim),
        swa=SWACacheState(
            kv_cache=torch.zeros(
                num_blocks,
                block_size,
                main_head_dim,
                dtype=torch.bfloat16,
            ),
            slot_mapping=torch.arange(num_tokens, dtype=torch.int64),
        ),
        mla_compressor=_make_compressor_state(
            num_tokens=num_tokens,
            head_dim=main_head_dim,
            compress_ratio=1,
            block_size=block_size,
            num_blocks=num_blocks,
        ),
        indexer_compressor=_make_compressor_state(
            num_tokens=num_tokens,
            head_dim=indexer_head_dim,
            compress_ratio=1,
            block_size=block_size,
            num_blocks=num_blocks,
        ),
        topk_indices_buffer=torch.empty(num_tokens, topk_tokens, dtype=torch.int32),
        prefill=SparseIndexerPrefillMetadata(
            cu_seq_lens=torch.tensor([0, num_tokens], dtype=torch.int64),
            cu_seqlen_ks=torch.zeros(num_tokens, dtype=torch.int64),
            cu_seqlen_ke=torch.arange(1, num_tokens + 1, dtype=torch.int64),
            block_table=torch.arange(num_blocks, dtype=torch.int32).view(1, num_blocks),
            topk_tokens=topk_tokens,
        ),
        main_head_dim=main_head_dim,
        q_eps=1e-6,
    )


def _make_dense_inputs(seed: int = 0) -> PostGemmStageInputs:
    inputs = _make_inputs(seed)
    inputs.kv_score = None
    inputs.indexer_kv_score = None
    inputs.indexer_weights = None
    inputs.indexer_wq_b_weight = None
    inputs.indexer_cos_sin_cache = None
    inputs.mla_compressor = None
    inputs.indexer_compressor = None
    inputs.topk_indices_buffer = None
    inputs.prefill = None
    return inputs


def _make_c128a_inputs(seed: int = 0) -> PostGemmStageInputs:
    inputs = _make_inputs(seed)
    inputs.indexer_kv_score = None
    inputs.indexer_weights = None
    inputs.indexer_wq_b_weight = None
    inputs.indexer_cos_sin_cache = None
    inputs.indexer_compressor = None
    inputs.topk_indices_buffer = None
    inputs.prefill = None
    return inputs


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=5e-2,
        rtol=5e-2,
        msg=lambda msg: f"{name}: {msg}",
    )


def test_post_gemm_torch_baseline_smoke() -> None:
    inputs = _make_inputs()

    q, topk = post_gemm_parallel_stage_torch_baseline(inputs)

    assert q.shape == (6, 2, 6)
    assert q.dtype == torch.bfloat16
    assert topk.shape == (6, 2)
    assert topk.dtype == torch.int32
    assert bool((topk >= 0).any())
    assert bool((inputs.swa.kv_cache != 0).any())
    assert bool((inputs.mla_compressor.kv_cache != 0).any())
    assert bool((inputs.indexer_compressor.kv_cache != 0).any())


def test_post_gemm_dense_torch_baseline_smoke() -> None:
    inputs = _make_dense_inputs()

    q, topk = post_gemm_parallel_stage_torch_baseline(inputs)

    assert q.shape == (6, 2, 6)
    assert q.dtype == torch.bfloat16
    assert topk is None
    assert bool((inputs.swa.kv_cache != 0).any())


def test_post_gemm_c128a_torch_baseline_smoke() -> None:
    inputs = _make_c128a_inputs()

    q, topk = post_gemm_parallel_stage_torch_baseline(inputs)

    assert q.shape == (6, 2, 6)
    assert q.dtype == torch.bfloat16
    assert topk is None
    assert bool((inputs.swa.kv_cache != 0).any())
    assert inputs.mla_compressor is not None
    assert bool((inputs.mla_compressor.kv_cache != 0).any())


@pytest.mark.skipif(
    not _HAS_DEEPSEEK_V4_POST_GEMM_DENSE_PREPACKED,
    reason="DeepSeek V4 dense post-GEMM prepacked C++ stage is unavailable",
)
def test_post_gemm_dense_cpp_prepacked_matches_torch_baseline() -> None:
    ref_inputs = _make_dense_inputs(seed=11)
    cpp_inputs = _make_dense_inputs(seed=11)
    weights = prepare_deepseek_v4_post_gemm_weights(cpp_inputs.main_wq_b_weight)

    ref_q, ref_topk = post_gemm_parallel_stage_torch_baseline(ref_inputs)
    cpp_q, cpp_topk = post_gemm_parallel_stage_cpp_prepacked(cpp_inputs, weights)

    _assert_close(cpp_q, ref_q, "dense q")
    assert ref_topk is None
    assert cpp_topk is None
    _assert_close(cpp_inputs.swa.kv_cache, ref_inputs.swa.kv_cache, "dense swa kv_cache")


@pytest.mark.skipif(
    not _HAS_DEEPSEEK_V4_POST_GEMM_C128A_PREPACKED,
    reason="DeepSeek V4 C128A post-GEMM prepacked C++ stage is unavailable",
)
def test_post_gemm_c128a_cpp_prepacked_matches_torch_baseline() -> None:
    ref_inputs = _make_c128a_inputs(seed=12)
    cpp_inputs = _make_c128a_inputs(seed=12)
    weights = prepare_deepseek_v4_post_gemm_weights(cpp_inputs.main_wq_b_weight)

    ref_q, ref_topk = post_gemm_parallel_stage_torch_baseline(ref_inputs)
    cpp_q, cpp_topk = post_gemm_parallel_stage_cpp_prepacked(cpp_inputs, weights)

    _assert_close(cpp_q, ref_q, "c128a q")
    assert ref_topk is None
    assert cpp_topk is None
    _assert_close(cpp_inputs.swa.kv_cache, ref_inputs.swa.kv_cache, "c128a swa kv_cache")
    assert cpp_inputs.mla_compressor is not None
    assert ref_inputs.mla_compressor is not None
    _assert_close(
        cpp_inputs.mla_compressor.state_cache,
        ref_inputs.mla_compressor.state_cache,
        "c128a mla state_cache",
    )
    _assert_close(
        cpp_inputs.mla_compressor.kv_cache,
        ref_inputs.mla_compressor.kv_cache,
        "c128a mla kv_cache",
    )


@pytest.mark.skipif(
    not _HAS_DEEPSEEK_V4_POST_GEMM_STAGE,
    reason="DeepSeek V4 post-GEMM C++ stage is unavailable",
)
def test_post_gemm_cpp_matches_torch_baseline() -> None:
    ref_inputs = _make_inputs(seed=1)
    cpp_inputs = _make_inputs(seed=1)

    ref_q, ref_topk = post_gemm_parallel_stage_torch_baseline(ref_inputs)
    cpp_q, cpp_topk = post_gemm_parallel_stage_cpp(cpp_inputs)

    _assert_close(cpp_q, ref_q, "q")
    assert torch.equal(cpp_topk, ref_topk)
    _assert_close(cpp_inputs.swa.kv_cache, ref_inputs.swa.kv_cache, "swa kv_cache")
    _assert_close(
        cpp_inputs.mla_compressor.state_cache,
        ref_inputs.mla_compressor.state_cache,
        "mla state_cache",
    )
    _assert_close(
        cpp_inputs.mla_compressor.kv_cache,
        ref_inputs.mla_compressor.kv_cache,
        "mla kv_cache",
    )
    _assert_close(
        cpp_inputs.indexer_compressor.state_cache,
        ref_inputs.indexer_compressor.state_cache,
        "indexer state_cache",
    )
    _assert_close(
        cpp_inputs.indexer_compressor.kv_cache,
        ref_inputs.indexer_compressor.kv_cache,
        "indexer kv_cache",
    )


@pytest.mark.skipif(
    not _HAS_DEEPSEEK_V4_POST_GEMM_STAGE,
    reason="DeepSeek V4 post-GEMM C++ stage is unavailable",
)
def test_post_gemm_cpp_rejects_int64_block_tables() -> None:
    inputs = _make_inputs(seed=7)
    inputs.mla_compressor.block_table = inputs.mla_compressor.block_table.to(torch.int64)

    with pytest.raises(RuntimeError, match="block_table must be int32"):
        post_gemm_parallel_stage_cpp(inputs)


@pytest.mark.skipif(
    not _HAS_DEEPSEEK_V4_POST_GEMM_STAGE,
    reason="DeepSeek V4 post-GEMM C++ stage is unavailable",
)
def test_post_gemm_cpp_write_kv_env_matches_torch_baseline(monkeypatch) -> None:
    monkeypatch.setenv("FUSED_CPP_DEEPSEEK_V4_KV_ROPE_WRITE_KV", "1")
    ref_inputs = _make_inputs(seed=5)
    cpp_inputs = _make_inputs(seed=5)

    ref_q, ref_topk = post_gemm_parallel_stage_torch_baseline(ref_inputs)
    cpp_q, cpp_topk = post_gemm_parallel_stage_cpp(cpp_inputs)

    _assert_close(cpp_q, ref_q, "q")
    assert torch.equal(cpp_topk, ref_topk)
    _assert_close(cpp_inputs.kv, ref_inputs.kv, "mutated kv")
    _assert_close(cpp_inputs.swa.kv_cache, ref_inputs.swa.kv_cache, "swa kv_cache")


@pytest.mark.skipif(
    not _HAS_DEEPSEEK_V4_POST_GEMM_STAGE_PREPACKED,
    reason="DeepSeek V4 post-GEMM prepacked C++ stage is unavailable",
)
def test_post_gemm_cpp_prepacked_matches_raw_cpp() -> None:
    raw_inputs = _make_inputs(seed=3)
    prepacked_inputs = _make_inputs(seed=3)
    weights = prepare_deepseek_v4_post_gemm_weights(
        prepacked_inputs.main_wq_b_weight,
        prepacked_inputs.indexer_wq_b_weight,
    )

    raw_q, raw_topk = post_gemm_parallel_stage_cpp(raw_inputs)
    prepacked_q, prepacked_topk = post_gemm_parallel_stage_cpp_prepacked(
        prepacked_inputs,
        weights,
    )

    _assert_close(prepacked_q, raw_q, "q")
    assert torch.equal(prepacked_topk, raw_topk)
    _assert_close(prepacked_inputs.kv, raw_inputs.kv, "mutated kv")
    _assert_close(
        prepacked_inputs.swa.kv_cache,
        raw_inputs.swa.kv_cache,
        "swa kv_cache",
    )
    _assert_close(
        prepacked_inputs.mla_compressor.state_cache,
        raw_inputs.mla_compressor.state_cache,
        "mla state_cache",
    )
    _assert_close(
        prepacked_inputs.mla_compressor.kv_cache,
        raw_inputs.mla_compressor.kv_cache,
        "mla kv_cache",
    )
    _assert_close(
        prepacked_inputs.indexer_compressor.state_cache,
        raw_inputs.indexer_compressor.state_cache,
        "indexer state_cache",
    )
    _assert_close(
        prepacked_inputs.indexer_compressor.kv_cache,
        raw_inputs.indexer_compressor.kv_cache,
        "indexer kv_cache",
    )


@pytest.mark.skipif(
    platform.machine() not in ("aarch64", "arm64") or not _HAS_DEEPSEEK_V4_POST_GEMM_STAGE_PREPACKED,
    reason="M8-aligned post-GEMM scheduling requires the AArch64 prepacked C++ stage",
)
def test_post_gemm_m8_aligned_matches_legacy_row_split(monkeypatch) -> None:
    """M8-panel scheduling must preserve C4A outputs for a partial final panel."""
    legacy_inputs = _make_inputs(seed=17, num_tokens=25)
    aligned_inputs = _make_inputs(seed=17, num_tokens=25)
    weights = prepare_deepseek_v4_post_gemm_weights(
        aligned_inputs.main_wq_b_weight,
        aligned_inputs.indexer_wq_b_weight,
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(7)
    try:
        monkeypatch.setenv("FUSED_CPP_POST_GEMM_SHARED_Q_POOL", "0")
        monkeypatch.setenv("FUSED_CPP_POST_GEMM_M8_ALIGNED", "0")
        legacy_q, legacy_topk = post_gemm_parallel_stage_cpp_prepacked(legacy_inputs, weights)
        monkeypatch.setenv("FUSED_CPP_POST_GEMM_M8_ALIGNED", "1")
        aligned_q, aligned_topk = post_gemm_parallel_stage_cpp_prepacked(aligned_inputs, weights)
    finally:
        torch.set_num_threads(previous_threads)

    _assert_close(aligned_q, legacy_q, "M8-aligned q")
    assert torch.equal(aligned_topk, legacy_topk)
    _assert_close(aligned_inputs.swa.kv_cache, legacy_inputs.swa.kv_cache, "M8-aligned swa kv_cache")
    _assert_close(
        aligned_inputs.mla_compressor.state_cache,
        legacy_inputs.mla_compressor.state_cache,
        "M8-aligned mla state_cache",
    )
    _assert_close(
        aligned_inputs.indexer_compressor.state_cache,
        legacy_inputs.indexer_compressor.state_cache,
        "M8-aligned indexer state_cache",
    )


@pytest.mark.skipif(
    platform.machine() not in ("aarch64", "arm64") or not _HAS_DEEPSEEK_V4_POST_GEMM_STAGE_PREPACKED,
    reason="shared post-GEMM Q worker pool requires the AArch64 prepacked C++ stage",
)
def test_post_gemm_shared_q_pool_matches_sequential(monkeypatch) -> None:
    """The shared Main/Indexer Q pool must preserve all C4A outputs."""
    sequential_inputs = _make_inputs(seed=23, num_tokens=25)
    shared_inputs = _make_inputs(seed=23, num_tokens=25)
    weights = prepare_deepseek_v4_post_gemm_weights(
        shared_inputs.main_wq_b_weight,
        shared_inputs.indexer_wq_b_weight,
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(7)
    try:
        monkeypatch.setenv("FUSED_CPP_POST_GEMM_M8_ALIGNED", "1")
        monkeypatch.setenv("FUSED_CPP_POST_GEMM_SHARED_Q_POOL", "0")
        sequential_q, sequential_topk = post_gemm_parallel_stage_cpp_prepacked(sequential_inputs, weights)
        monkeypatch.setenv("FUSED_CPP_POST_GEMM_SHARED_Q_POOL", "1")
        shared_q, shared_topk = post_gemm_parallel_stage_cpp_prepacked(shared_inputs, weights)
    finally:
        torch.set_num_threads(previous_threads)

    _assert_close(shared_q, sequential_q, "shared Q pool q")
    assert torch.equal(shared_topk, sequential_topk)
    _assert_close(shared_inputs.swa.kv_cache, sequential_inputs.swa.kv_cache, "shared Q pool swa kv_cache")
    _assert_close(
        shared_inputs.mla_compressor.state_cache,
        sequential_inputs.mla_compressor.state_cache,
        "shared Q pool mla state_cache",
    )
    _assert_close(
        shared_inputs.indexer_compressor.state_cache,
        sequential_inputs.indexer_compressor.state_cache,
        "shared Q pool indexer state_cache",
    )


@pytest.mark.skipif(
    not _HAS_DEEPSEEK_V4_POST_GEMM_STAGE_PREPACKED,
    reason="DeepSeek V4 post-GEMM prepacked C++ stage is unavailable",
)
def test_post_gemm_cpp_uses_prepared_weights_from_inputs() -> None:
    raw_inputs = _make_inputs(seed=4)
    prepared_inputs = _make_inputs(seed=4)
    prepared_inputs.prepared_weights = prepare_deepseek_v4_post_gemm_weights(
        prepared_inputs.main_wq_b_weight,
        prepared_inputs.indexer_wq_b_weight,
    )

    raw_q, raw_topk = post_gemm_parallel_stage_cpp(raw_inputs)
    prepared_q, prepared_topk = post_gemm_parallel_stage_cpp(prepared_inputs)

    _assert_close(prepared_q, raw_q, "q")
    assert torch.equal(prepared_topk, raw_topk)


@pytest.mark.skipif(
    not _HAS_DEEPSEEK_V4_POST_GEMM_STAGE,
    reason="DeepSeek V4 post-GEMM C++ stage is unavailable",
)
def test_post_gemm_cpp_profile_summary(monkeypatch, capfd) -> None:
    monkeypatch.setenv("FUSED_CPP_DEEPSEEK_V4_POST_GEMM_PROFILE", "1")

    post_gemm_parallel_stage_cpp(_make_inputs(seed=2))

    stderr = capfd.readouterr().err
    assert "deepseek_v4_post_gemm_stage_profile" in stderr
    for field in (
        "total_ms=",
        "input_check_ms=",
        "main_q_gemm_ms=",
        "main_q_norm_rope_swa_insert_ms=",
        "indexer_q_gemm_ms=",
        "indexer_q_rope_weights_ms=",
        "mla_save_partial_states_ms=",
        "mla_compress_norm_rope_insert_ms=",
        "indexer_save_partial_states_ms=",
        "indexer_compress_norm_rope_insert_ms=",
        "sparse_indexer_short_path_ms=",
        "sparse_indexer_gather_ms=",
        "sparse_indexer_fold_q_ms=",
        "sparse_indexer_score_topk_ms=",
        "other_ms=",
    ):
        assert field in stderr
