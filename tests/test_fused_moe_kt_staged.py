"""The KTransformers-dataflow comparator: KTransformers' passes, this repository's GEMM."""

from __future__ import annotations

import os
import platform

import pytest
import torch

from fused_cpp.moe import _HAS_BF16_TILED_FUSED_MOE
from fused_cpp.moe import fused_moe_bf16_tiled
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights
from fused_cpp.moe.bf16_tiled import fused_moe_bf16_tiled_kt_staged
from fused_cpp.moe.bf16_tiled import prepare_fused_moe_bf16_tiled_kt_weights

pytestmark = pytest.mark.skipif(
    platform.machine() not in ("aarch64", "arm64") or not _HAS_BF16_TILED_FUSED_MOE,
    reason="BF16 tiled fused MoE kernel is only available on AArch64",
)


def _case(num_tokens: int = 29, hidden: int = 128, intermediate: int = 64, experts: int = 8, top_k: int = 6):
    generator = torch.Generator().manual_seed(20260923)
    x = torch.empty((num_tokens, hidden), dtype=torch.bfloat16).normal_(0.0, 1.0, generator=generator)
    w13 = torch.empty((experts, 2 * intermediate, hidden), dtype=torch.bfloat16)
    w13.normal_(0.0, hidden**-0.5, generator=generator)
    w2 = torch.empty((experts, hidden, intermediate), dtype=torch.bfloat16)
    w2.normal_(0.0, intermediate**-0.5, generator=generator)
    # Uneven expert loads, including one-row and non-multiple-of-12 experts.
    ids = torch.tensor(
        [[(token * 3 + slot * (1 + token % 2)) % experts for slot in range(top_k)] for token in range(num_tokens)],
        dtype=torch.int32,
    )
    for row in ids:
        seen = set()
        for slot in range(top_k):
            while int(row[slot]) in seen:
                row[slot] = (int(row[slot]) + 1) % experts
            seen.add(int(row[slot]))
    weights = torch.softmax(torch.randn((num_tokens, top_k), generator=generator), dim=-1)
    return x, w13, w2, weights, ids


def _ktransformers_reference(x, w13, w2, weights, ids) -> torch.Tensor:
    """KTransformers' rounding points: BF16 gate, up, activation, down; FP32 sum."""
    intermediate = w13.shape[1] // 2
    out = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.float32)
    for token in range(x.shape[0]):
        row = x[token].float()
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            gate = (w13[expert, :intermediate].float() @ row).to(torch.bfloat16).float()
            up = (w13[expert, intermediate:].float() @ row).to(torch.bfloat16).float()
            act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).float()
            down = (w2[expert].float() @ act).to(torch.bfloat16).float()
            out[token] += float(weights[token, slot]) * down
    return out.to(torch.bfloat16)


def _threads(count: int) -> tuple[int, torch.Tensor]:
    cpus = sorted(os.sched_getaffinity(0))[:count] if hasattr(os, "sched_getaffinity") else list(range(count))
    return len(cpus), torch.tensor(cpus, dtype=torch.int32)


def _plain(w13, w2):
    packed = prepare_fused_moe_bf16_tiled_kt_weights(w13, w2)
    if packed.gemm_backend != 1:
        pytest.skip("requires an SVE BF16 build/runtime")
    return packed


def test_kt_staged_follows_ktransformers_rounding() -> None:
    x, w13, w2, weights, ids = _case()
    threads, cpus = _threads(8)
    actual = fused_moe_bf16_tiled_kt_staged(x, _plain(w13, w2), weights, ids, num_threads=threads,
                                            thread_cpu_ids=cpus, n_block=16)
    expected = _ktransformers_reference(x, w13, w2, weights, ids)
    relative = float((actual.float() - expected.float()).norm() / expected.float().norm())
    assert relative <= 5e-3
    torch.testing.assert_close(actual.float(), expected.float(), atol=7e-2, rtol=7e-2)


def test_kt_staged_agrees_with_the_fused_path_within_the_bf16_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FUSED_CPP_MOE_SVE", "1")
    x, w13, w2, weights, ids = _case()
    threads, cpus = _threads(8)
    fused = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
    if fused.gemm_backend != 1:
        pytest.skip("requires an SVE BF16 build/runtime")
    reference = fused_moe_bf16_tiled(x, fused, weights, ids, num_threads=threads)
    actual = fused_moe_bf16_tiled_kt_staged(x, _plain(w13, w2), weights, ids, num_threads=threads,
                                            thread_cpu_ids=cpus)
    torch.testing.assert_close(actual.float(), reference.float(), atol=7e-2, rtol=7e-2)


@pytest.mark.parametrize("num_tokens", [1, 5, 13, 29, 64])
def test_kt_staged_is_independent_of_block_and_thread_count(num_tokens: int) -> None:
    x, w13, w2, weights, ids = _case(num_tokens=num_tokens)
    packed = _plain(w13, w2)
    one, one_cpu = _threads(1)
    many, many_cpus = _threads(7)
    base = fused_moe_bf16_tiled_kt_staged(x, packed, weights, ids, num_threads=one, thread_cpu_ids=one_cpu,
                                          n_block=64)
    for n_block in (8, 16, 256):
        other = fused_moe_bf16_tiled_kt_staged(x, packed, weights, ids, num_threads=many,
                                               thread_cpu_ids=many_cpus, n_block=n_block)
        torch.testing.assert_close(other, base, atol=0, rtol=0)
    out = torch.empty_like(x)
    assert fused_moe_bf16_tiled_kt_staged(x, packed, weights, ids, num_threads=many, thread_cpu_ids=many_cpus,
                                          out=out) is out
    torch.testing.assert_close(out, base, atol=0, rtol=0)


def test_kt_staged_rejects_fused_silu_weights() -> None:
    x, w13, w2, weights, ids = _case()
    fused = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
    if fused.gemm_backend != 1:
        pytest.skip("requires an SVE BF16 build/runtime")
    with pytest.raises(ValueError, match="fuse_silu=False"):
        fused_moe_bf16_tiled_kt_staged(x, fused, weights, ids)
