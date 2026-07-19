from __future__ import annotations

import pytest
import torch

from fused_cpp.moe import bf16_tiled
from fused_cpp.moe import fused_moe_naive


def test_available_backends_match_support_flag() -> None:
    backends = bf16_tiled.available_fused_moe_bf16_tiled_backends()
    assert isinstance(backends, tuple)
    assert set(backends) <= {"arm_neon_bf16", "arm_sve_bf16"}
    assert bf16_tiled._HAS_BF16_TILED_FUSED_MOE is bool(backends)


def test_prepare_forwards_backend_and_records_native_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[bool, str]] = []

    def fake_prepare(w13: torch.Tensor, w2: torch.Tensor, fuse_silu: bool, backend: str):
        calls.append((fuse_silu, backend))
        return w13, 16, 32, w2, 16, 16, 1, 32, "arm_sve_bf16"

    monkeypatch.setattr(bf16_tiled, "_HAS_BF16_TILED_FUSED_MOE", True)
    monkeypatch.setattr(bf16_tiled, "_prepare_bf16_tiled_impl", fake_prepare)
    w13 = torch.empty((2, 32, 16), dtype=torch.bfloat16)
    w2 = torch.empty((2, 16, 16), dtype=torch.bfloat16)

    packed = bf16_tiled.prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="arm_sve_bf16",
    )

    assert calls == [(True, "arm_sve_bf16")]
    assert packed.gemm_backend == 1
    assert packed.backend_n_tile == 32
    assert packed.backend_name == "arm_sve_bf16"


def test_prepare_rejects_non_string_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bf16_tiled, "_HAS_BF16_TILED_FUSED_MOE", True)
    w13 = torch.empty((1, 16, 8), dtype=torch.bfloat16)
    w2 = torch.empty((1, 8, 8), dtype=torch.bfloat16)

    with pytest.raises(TypeError, match="backend must be a string"):
        bf16_tiled.prepare_fused_moe_bf16_tiled_weights(w13, w2, backend=1)  # type: ignore[arg-type]


def test_moe_symbols_are_aliased_on_legacy_extension() -> None:
    if bf16_tiled._moe_native is None:
        pytest.skip("native MoE extension is not built")

    try:
        from fused_cpp import _C, _moe_C
    except ImportError:
        pytest.skip("native extensions are not built")

    names = (
        "fused_moe_bf16_tiled_prepare_weights",
        "fused_moe_bf16_tiled",
        "fused_moe_bf16_tiled_scheduled",
        "fused_moe_bf16_tiled_async",
        "fused_moe_test_team_gemm",
    )
    for name in names:
        assert getattr(_C, name) is getattr(_moe_C, name)


def test_explicit_neon_prepare_reports_stable_backend_metadata() -> None:
    if "arm_neon_bf16" not in bf16_tiled.available_fused_moe_bf16_tiled_backends():
        pytest.skip("ARM NEON BF16 backend is unavailable")

    w13 = torch.randn((1, 16, 8), dtype=torch.bfloat16)
    w2 = torch.randn((1, 8, 8), dtype=torch.bfloat16)
    packed = bf16_tiled.prepare_fused_moe_bf16_tiled_weights(w13, w2, backend="arm_neon_bf16")

    assert packed.gemm_backend == 0
    assert packed.backend_n_tile == 8
    assert packed.backend_name == "arm_neon_bf16"


def test_auto_fused_silu_prefers_sve_when_available() -> None:
    if "arm_sve_bf16" not in bf16_tiled.available_fused_moe_bf16_tiled_backends():
        pytest.skip("ARM SVE BF16 backend is unavailable")

    w13 = torch.randn((1, 32, 16), dtype=torch.bfloat16)
    w2 = torch.randn((1, 16, 16), dtype=torch.bfloat16)
    packed = bf16_tiled.prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)

    assert packed.gemm_backend == 1
    assert packed.backend_name == "arm_sve_bf16"


def test_sve_environment_override_forces_neon(monkeypatch: pytest.MonkeyPatch) -> None:
    if "arm_neon_bf16" not in bf16_tiled.available_fused_moe_bf16_tiled_backends():
        pytest.skip("ARM NEON BF16 backend is unavailable")

    monkeypatch.setenv("FUSED_CPP_MOE_SVE", "0")
    assert bf16_tiled.available_fused_moe_bf16_tiled_backends() == ("arm_neon_bf16",)

    generator = torch.Generator().manual_seed(20260718)
    hidden = torch.empty((3, 16), dtype=torch.bfloat16).normal_(0.0, 0.01, generator=generator)
    w13 = torch.empty((1, 32, 16), dtype=torch.bfloat16).normal_(0.0, 0.01, generator=generator)
    w2 = torch.empty((1, 16, 16), dtype=torch.bfloat16).normal_(0.0, 0.01, generator=generator)
    packed = bf16_tiled.prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)

    assert packed.gemm_backend == 0
    assert packed.backend_name == "arm_neon_bf16"

    topk_ids = torch.zeros((3, 1), dtype=torch.int32)
    topk_weights = torch.ones((3, 1), dtype=torch.float32)
    candidate = bf16_tiled.fused_moe_bf16_tiled(
        hidden,
        packed,
        topk_weights,
        topk_ids,
        num_threads=1,
    )
    reference = fused_moe_naive(hidden.float(), w13.float(), w2.float(), topk_weights, topk_ids).to(torch.bfloat16)
    torch.testing.assert_close(candidate.float(), reference.float(), atol=2.0e-3, rtol=2.0e-2)
