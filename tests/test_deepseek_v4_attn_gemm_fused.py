# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import platform

import pytest
import torch

from fused_cpp import _C
from fused_cpp.deepseek_v4_attn_gemm_fused import (
    _HAS_DEEPSEEK_V4_ATTN_GEMM_FUSED,
    deepseek_v4_attn_gemm_fused_prepacked,
    deepseek_v4_attn_gemm_fused_prepacked_normed,
    prepare_deepseek_v4_attn_gemm_weights,
)

pytestmark = pytest.mark.skipif(
    platform.machine() not in ("aarch64", "arm64") or not _HAS_DEEPSEEK_V4_ATTN_GEMM_FUSED,
    reason="DeepSeek V4 fused GEMM kernel is only available on AArch64",
)

_HAS_OPENMP = bool(getattr(_C, "has_openmp", lambda: False)())


def _bf16_randn(*shape: int) -> torch.Tensor:
    return (torch.randn(*shape) * 0.25).to(torch.bfloat16)


def _test_core_ids(count: int) -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        return allowed[: min(count, len(allowed))]
    return list(range(count))


def _rmsnorm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    out = x_f * torch.rsqrt(var + eps) * weight.float()
    return out.to(x.dtype)


@pytest.mark.parametrize(
    ("M", "K", "Ns", "variant"),
    [
        (7, 16, (24, 16, 24, 8), "dense"),
        (7, 16, (24, 16, 24, 8), "c128a"),
        (7, 16, (24, 16, 24, 8), "c4a"),
        (5, 13, (11, 17, 9, 15), "c4a"),
    ],
)
def test_deepseek_v4_attn_gemm_fused_matches_torch(
    M: int,
    K: int,
    Ns: tuple[int, int, int, int],
    variant: str,
) -> None:
    torch.manual_seed(0)
    hidden_states = _bf16_randn(M, K)
    weights = tuple(_bf16_randn(K, N) for N in Ns)
    if variant == "dense":
        packed = prepare_deepseek_v4_attn_gemm_weights(weights[0])
    elif variant == "c128a":
        packed = prepare_deepseek_v4_attn_gemm_weights(weights[0], weights[1])
    else:
        packed = prepare_deepseek_v4_attn_gemm_weights(
            weights[0],
            weights[1],
            weights[2],
            weights[3],
        )

    qr_kv, kv_score, indexer_kv_score, indexer_weights = deepseek_v4_attn_gemm_fused_prepacked(
        hidden_states,
        packed,
    )

    ref_qr_kv = (hidden_states.float() @ weights[0].float()).to(torch.bfloat16)

    assert qr_kv.dtype == torch.bfloat16
    assert qr_kv.shape == ref_qr_kv.shape
    torch.testing.assert_close(qr_kv.float(), ref_qr_kv.float(), atol=5e-2, rtol=5e-2)

    if variant == "dense":
        assert kv_score is None
        assert indexer_kv_score is None
        assert indexer_weights is None
        return

    ref_kv_score = hidden_states.float() @ weights[1].float()
    assert kv_score is not None
    assert kv_score.dtype == torch.float32
    assert kv_score.shape == ref_kv_score.shape
    torch.testing.assert_close(kv_score, ref_kv_score, atol=5e-2, rtol=5e-2)

    if variant == "c128a":
        assert indexer_kv_score is None
        assert indexer_weights is None
        return

    ref_indexer_kv_score = hidden_states.float() @ weights[2].float()
    ref_indexer_weights = (hidden_states.float() @ weights[3].float()).to(torch.bfloat16)
    assert indexer_kv_score is not None
    assert indexer_weights is not None
    assert indexer_kv_score.dtype == torch.float32
    assert indexer_weights.dtype == torch.bfloat16
    assert indexer_kv_score.shape == ref_indexer_kv_score.shape
    assert indexer_weights.shape == ref_indexer_weights.shape
    torch.testing.assert_close(indexer_kv_score, ref_indexer_kv_score, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(
        indexer_weights.float(),
        ref_indexer_weights.float(),
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.skipif(not _HAS_OPENMP, reason="OpenMP is unavailable")
@pytest.mark.parametrize("variant", ["dense", "c128a", "c4a"])
@pytest.mark.parametrize("schedule", [None, "legacy", "m8", "pool", "mn"])
def test_deepseek_v4_attn_gemm_fused_mt_matches_serial(
    variant: str,
    schedule: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1)
    M = 17
    K = 13
    Ns = (19, 23, 11, 7)
    hidden_states = _bf16_randn(M, K)
    weights = tuple(_bf16_randn(K, N) for N in Ns)
    if variant == "dense":
        packed = prepare_deepseek_v4_attn_gemm_weights(weights[0])
    elif variant == "c128a":
        packed = prepare_deepseek_v4_attn_gemm_weights(weights[0], weights[1])
    else:
        packed = prepare_deepseek_v4_attn_gemm_weights(
            weights[0],
            weights[1],
            weights[2],
            weights[3],
        )
    core_ids = _test_core_ids(3)
    if len(core_ids) < 2:
        pytest.skip("need at least two available CPU cores")

    if schedule is None:
        monkeypatch.delenv("FUSED_CPP_ATTN_GEMM_SCHEDULE", raising=False)
    else:
        monkeypatch.setenv("FUSED_CPP_ATTN_GEMM_SCHEDULE", schedule)
    monkeypatch.setenv("FUSED_CPP_ATTN_GEMM_N_GROUPS", "7")
    serial_outputs = deepseek_v4_attn_gemm_fused_prepacked(
        hidden_states,
        packed,
    )
    mt_outputs = deepseek_v4_attn_gemm_fused_prepacked(
        hidden_states,
        packed,
        cores=core_ids,
    )

    for serial, mt in zip(serial_outputs, mt_outputs):
        if serial is None:
            assert mt is None
            continue
        assert mt is not None
        torch.testing.assert_close(
            mt.float(),
            serial.float(),
            atol=5e-2,
            rtol=5e-2,
        )


@pytest.mark.skipif(not _HAS_OPENMP, reason="OpenMP is unavailable")
@pytest.mark.parametrize("M", [1, 8, 11, 12, 13, 23, 24, 25])
def test_deepseek_v4_attn_gemm_sve_m12_windows_match_torch(
    M: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SVE M12 panels and exact-M tails must preserve all MN-window outputs."""
    torch.manual_seed(20260820 + M)
    K = 16
    Ns = (37, 29, 41, 23)
    hidden_states = _bf16_randn(M, K)
    weights = tuple(_bf16_randn(K, N) for N in Ns)
    monkeypatch.setenv("FUSED_CPP_ATTN_GEMM_BACKEND", "sve")
    monkeypatch.setenv("FUSED_CPP_ATTN_GEMM_SCHEDULE", "mn")
    monkeypatch.setenv("FUSED_CPP_ATTN_GEMM_N_GROUPS", "7")
    try:
        packed = prepare_deepseek_v4_attn_gemm_weights(*weights)
    except RuntimeError as error:
        if "does not support SVE BF16" in str(error):
            pytest.skip("SVE BF16 JIT GEMM is unavailable")
        raise

    core_ids = _test_core_ids(4)
    if len(core_ids) < 2:
        pytest.skip("need at least two available CPU cores")
    actual = deepseek_v4_attn_gemm_fused_prepacked(hidden_states, packed, cores=core_ids)
    expected = (
        (hidden_states.float() @ weights[0].float()).to(torch.bfloat16),
        hidden_states.float() @ weights[1].float(),
        hidden_states.float() @ weights[2].float(),
        (hidden_states.float() @ weights[3].float()).to(torch.bfloat16),
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert actual_tensor is not None
        torch.testing.assert_close(actual_tensor.float(), expected_tensor.float(), atol=5e-2, rtol=5e-2)


def test_deepseek_v4_attn_gemm_defaults_to_sve_with_neon_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default packing must select supported SVE while retaining the explicit NEON fallback."""
    weight = _bf16_randn(16, 17)
    monkeypatch.delenv("FUSED_CPP_POST_GEMM_BACKEND", raising=False)
    monkeypatch.delenv("FUSED_CPP_ATTN_GEMM_SVE", raising=False)
    monkeypatch.setenv("FUSED_CPP_ATTN_GEMM_BACKEND", "sve")
    try:
        sve = prepare_deepseek_v4_attn_gemm_weights(weight)
    except RuntimeError as error:
        if "does not support SVE BF16" in str(error):
            pytest.skip("SVE BF16 JIT GEMM is unavailable")
        raise

    monkeypatch.delenv("FUSED_CPP_ATTN_GEMM_BACKEND")
    default = prepare_deepseek_v4_attn_gemm_weights(weight)
    assert default.fused_wqa_wkv[0].numel() == sve.fused_wqa_wkv[0].numel()
    torch.testing.assert_close(default.fused_wqa_wkv[0], sve.fused_wqa_wkv[0], rtol=0.0, atol=0.0)

    monkeypatch.setenv("FUSED_CPP_POST_GEMM_BACKEND", "neon")
    default_with_post_override = prepare_deepseek_v4_attn_gemm_weights(weight)
    assert default_with_post_override.fused_wqa_wkv[0].numel() == sve.fused_wqa_wkv[0].numel()
    monkeypatch.delenv("FUSED_CPP_POST_GEMM_BACKEND")

    monkeypatch.setenv("FUSED_CPP_ATTN_GEMM_SVE", "0")
    fallback = prepare_deepseek_v4_attn_gemm_weights(weight)
    monkeypatch.setenv("FUSED_CPP_ATTN_GEMM_BACKEND", "neon")
    neon = prepare_deepseek_v4_attn_gemm_weights(weight)
    assert fallback.fused_wqa_wkv[0].numel() == neon.fused_wqa_wkv[0].numel()
    torch.testing.assert_close(fallback.fused_wqa_wkv[0], neon.fused_wqa_wkv[0], rtol=0.0, atol=0.0)
    assert sve.fused_wqa_wkv[0].numel() != neon.fused_wqa_wkv[0].numel()


@pytest.mark.skipif(not _HAS_OPENMP, reason="OpenMP is unavailable")
@pytest.mark.parametrize("M", [1, 3, 5, 7, 8, 13])
def test_deepseek_v4_attn_gemm_prepacked_a_matches_repack(
    M: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(11 + M)
    K = 13
    Ns = (19, 23, 11, 7)
    hidden_states = _bf16_randn(M, K)
    weights = tuple(_bf16_randn(K, N) for N in Ns)
    packed = prepare_deepseek_v4_attn_gemm_weights(*weights)
    core_ids = _test_core_ids(3)
    if len(core_ids) < 2:
        pytest.skip("need at least two available CPU cores")

    monkeypatch.setenv("FUSED_CPP_ATTN_GEMM_SCHEDULE", "mn")
    monkeypatch.setenv("FUSED_CPP_ATTN_GEMM_N_GROUPS", "7")
    monkeypatch.setenv("FUSED_CPP_ATTN_GEMM_PREPACK_A", "0")
    repacked_outputs = deepseek_v4_attn_gemm_fused_prepacked(
        hidden_states,
        packed,
        cores=core_ids,
    )
    monkeypatch.setenv("FUSED_CPP_ATTN_GEMM_PREPACK_A", "1")
    prepacked_outputs = deepseek_v4_attn_gemm_fused_prepacked(
        hidden_states,
        packed,
        cores=core_ids,
    )

    for repacked, prepacked in zip(repacked_outputs, prepacked_outputs):
        assert repacked is not None
        assert prepacked is not None
        assert prepacked.shape == repacked.shape
        assert prepacked.dtype == repacked.dtype
        torch.testing.assert_close(
            prepacked.float(),
            repacked.float(),
            atol=5e-2,
            rtol=5e-2,
        )


@pytest.mark.parametrize("variant", ["dense", "c128a", "c4a"])
def test_deepseek_v4_attn_gemm_fused_normed_matches_torch(
    variant: str,
) -> None:
    torch.manual_seed(2)
    M = 9
    K = 13
    q_lora_rank = 11
    kv_dim = 7
    Ns = (q_lora_rank + kv_dim, 17, 9, 15)
    eps = 1e-6
    hidden_states = _bf16_randn(M, K)
    weights = tuple(_bf16_randn(K, N) for N in Ns)
    q_norm_weight = (1.0 + torch.randn(q_lora_rank) * 0.1).to(torch.bfloat16)
    kv_norm_weight = (1.0 + torch.randn(kv_dim) * 0.1).to(torch.bfloat16)
    if variant == "dense":
        packed = prepare_deepseek_v4_attn_gemm_weights(weights[0])
    elif variant == "c128a":
        packed = prepare_deepseek_v4_attn_gemm_weights(weights[0], weights[1])
    else:
        packed = prepare_deepseek_v4_attn_gemm_weights(
            weights[0],
            weights[1],
            weights[2],
            weights[3],
        )

    qr, kv, kv_score, indexer_kv_score, indexer_weights = deepseek_v4_attn_gemm_fused_prepacked_normed(
        hidden_states,
        packed,
        q_norm_weight,
        kv_norm_weight,
        q_lora_rank,
        kv_dim,
        eps,
    )

    ref_qr_kv = (hidden_states.float() @ weights[0].float()).to(torch.bfloat16)
    ref_qr = _rmsnorm_ref(ref_qr_kv[:, :q_lora_rank], q_norm_weight, eps)
    ref_kv = _rmsnorm_ref(ref_qr_kv[:, q_lora_rank:], kv_norm_weight, eps)
    assert qr.dtype == torch.bfloat16
    assert kv.dtype == torch.bfloat16
    assert qr.shape == ref_qr.shape
    assert kv.shape == ref_kv.shape
    torch.testing.assert_close(qr.float(), ref_qr.float(), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(kv.float(), ref_kv.float(), atol=5e-2, rtol=5e-2)

    if variant == "dense":
        assert kv_score is None
        assert indexer_kv_score is None
        assert indexer_weights is None
        return

    ref_kv_score = hidden_states.float() @ weights[1].float()
    assert kv_score is not None
    torch.testing.assert_close(kv_score, ref_kv_score, atol=5e-2, rtol=5e-2)

    if variant == "c128a":
        assert indexer_kv_score is None
        assert indexer_weights is None
        return

    ref_indexer_kv_score = hidden_states.float() @ weights[2].float()
    ref_indexer_weights = (hidden_states.float() @ weights[3].float()).to(torch.bfloat16)
    assert indexer_kv_score is not None
    assert indexer_weights is not None
    torch.testing.assert_close(indexer_kv_score, ref_indexer_kv_score, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(
        indexer_weights.float(),
        ref_indexer_weights.float(),
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.skipif(not _HAS_OPENMP, reason="OpenMP is unavailable")
@pytest.mark.parametrize("variant", ["dense", "c128a", "c4a"])
def test_deepseek_v4_attn_gemm_fused_normed_mt_matches_serial(
    variant: str,
) -> None:
    torch.manual_seed(3)
    M = 19
    K = 13
    q_lora_rank = 11
    kv_dim = 7
    Ns = (q_lora_rank + kv_dim, 17, 9, 15)
    eps = 1e-6
    hidden_states = _bf16_randn(M, K)
    weights = tuple(_bf16_randn(K, N) for N in Ns)
    q_norm_weight = (1.0 + torch.randn(q_lora_rank) * 0.1).to(torch.bfloat16)
    kv_norm_weight = (1.0 + torch.randn(kv_dim) * 0.1).to(torch.bfloat16)
    if variant == "dense":
        packed = prepare_deepseek_v4_attn_gemm_weights(weights[0])
    elif variant == "c128a":
        packed = prepare_deepseek_v4_attn_gemm_weights(weights[0], weights[1])
    else:
        packed = prepare_deepseek_v4_attn_gemm_weights(
            weights[0],
            weights[1],
            weights[2],
            weights[3],
        )
    core_ids = _test_core_ids(3)
    if len(core_ids) < 2:
        pytest.skip("need at least two available CPU cores")

    serial_outputs = deepseek_v4_attn_gemm_fused_prepacked_normed(
        hidden_states,
        packed,
        q_norm_weight,
        kv_norm_weight,
        q_lora_rank,
        kv_dim,
        eps,
    )
    mt_outputs = deepseek_v4_attn_gemm_fused_prepacked_normed(
        hidden_states,
        packed,
        q_norm_weight,
        kv_norm_weight,
        q_lora_rank,
        kv_dim,
        eps,
        cores=core_ids,
    )

    for serial, mt in zip(serial_outputs, mt_outputs):
        if serial is None:
            assert mt is None
            continue
        assert mt is not None
        torch.testing.assert_close(
            mt.float(),
            serial.float(),
            atol=5e-2,
            rtol=5e-2,
        )
