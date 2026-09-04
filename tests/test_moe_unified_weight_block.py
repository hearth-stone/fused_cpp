from __future__ import annotations

import torch

from fused_cpp.moe import PreparedBF16TiledFusedMoEWeights
from optimizations.fused_moe_sve.benchmarks.bench_unified_weight_block import (
    LAYOUTS,
    MODES,
    cell_name,
    decide_unified_block,
    parse_cell,
    storage_ptr,
    unify_prepared,
)


def _prepared(w13: torch.Tensor, w2: torch.Tensor) -> PreparedBF16TiledFusedMoEWeights:
    return PreparedBF16TiledFusedMoEWeights(
        w13=(w13, 8, 16),
        w2=(w2, 16, 8),
        fused_silu=True,
        gemm_backend=1,
        backend_n_tile=8,
        backend_name="arm_sve_bf16",
    )


def _delta(median_ms: float) -> dict[str, dict[str, float]]:
    return {"delta": {"median_ms": median_ms}}


def _calls(overlap: float) -> dict[str, list[float]]:
    return {"peer_overlap_experts": [overlap]}


def test_unify_puts_w13_and_w2_on_one_storage() -> None:
    w13 = torch.arange(24, dtype=torch.bfloat16).reshape(2, 12)
    w2 = torch.arange(24, 36, dtype=torch.bfloat16).reshape(2, 6)
    unified, owner = unify_prepared(_prepared(w13, w2))

    assert storage_ptr(unified.w13[0]) == storage_ptr(owner)
    assert storage_ptr(unified.w2[0]) == storage_ptr(owner)
    assert unified.w13[0].is_contiguous()
    assert unified.w2[0].is_contiguous()
    torch.testing.assert_close(unified.w13[0].float(), w13.float(), atol=0, rtol=0)
    torch.testing.assert_close(unified.w2[0].float(), w2.float(), atol=0, rtol=0)
    assert int(unified.w2[0].data_ptr()) > int(unified.w13[0].data_ptr())


def test_cell_names_cover_split_and_unified() -> None:
    assert cell_name("unified", "many16_1t_same_head") == "unified:many16_1t_same_head"
    assert parse_cell("split:wide16_same_head") == ("split", "wide16_same_head")
    assert "isolated_head" in MODES
    assert LAYOUTS == ("split", "unified")


def test_decide_unified_helps_when_one_block_drops_many16() -> None:
    comparisons = {
        "split:wide16_same_head_vs_isolated": _delta(0.01),
        "unified:wide16_same_head_vs_isolated": _delta(0.01),
        "split:many16_1t_same_head_vs_isolated": _delta(0.16),
        "unified:many16_1t_same_head_vs_isolated": _delta(0.04),
        "split:many16_1t_cross_head_vs_isolated": _delta(0.03),
        "unified:many16_1t_cross_head_vs_isolated": _delta(0.02),
    }
    calls = {
        "split:wide16_same_head": _calls(1.0),
        "unified:wide16_same_head": _calls(1.0),
        "split:many16_1t_same_head": _calls(16.0),
        "unified:many16_1t_same_head": _calls(16.0),
    }

    decision = decide_unified_block(comparisons, calls)

    assert decision["signature"] == "unified_helps"
    assert decision["add_default_off_structure"] is False


def test_decide_layout_neutral_when_block_matches_split() -> None:
    comparisons = {
        "split:wide16_same_head_vs_isolated": _delta(0.01),
        "unified:wide16_same_head_vs_isolated": _delta(0.02),
        "split:many16_1t_same_head_vs_isolated": _delta(0.16),
        "unified:many16_1t_same_head_vs_isolated": _delta(0.15),
        "split:many16_1t_cross_head_vs_isolated": _delta(0.03),
        "unified:many16_1t_cross_head_vs_isolated": _delta(0.04),
    }
    calls = {
        "split:wide16_same_head": _calls(1.0),
        "unified:wide16_same_head": _calls(1.0),
        "split:many16_1t_same_head": _calls(16.0),
        "unified:many16_1t_same_head": _calls(16.0),
    }

    decision = decide_unified_block(comparisons, calls)

    assert decision["signature"] == "layout_neutral"
    assert decision["unified_helps"] is False
    assert decision["add_default_off_structure"] is False
