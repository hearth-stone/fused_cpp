from __future__ import annotations

from pathlib import Path

import pytest
import torch

from optimizations.fused_moe_sve.benchmarks.bench_high_skew_planner_closure import (
    _parse_ints,
    _parse_orders,
    _spearman,
    load_route_layer,
)


def test_route_layer_loader_selects_layer_and_rejects_duplicate_experts(tmp_path: Path) -> None:
    route_path = tmp_path / "routes.pt"
    expert_ids = torch.tensor(
        [
            [[0, 1], [1, 2]],
            [[2, 3], [3, 0]],
        ],
        dtype=torch.int16,
    )
    torch.save(
        {"expert_ids": expert_ids, "layer_ids": torch.tensor([10, 20], dtype=torch.int16)},
        route_path,
    )

    selected, metadata = load_route_layer(route_path, 1, 4)

    torch.testing.assert_close(selected, torch.tensor([[1, 2], [3, 0]], dtype=torch.int32))
    assert metadata["layer_id"] == 20
    assert metadata["active_experts"] == 4
    assert metadata["duplicate_token_rows"] == 0

    expert_ids[0, 1] = torch.tensor([2, 2], dtype=torch.int16)
    torch.save(
        {"expert_ids": expert_ids, "layer_ids": torch.tensor([10, 20], dtype=torch.int16)},
        route_path,
    )
    with pytest.raises(ValueError, match="duplicate experts"):
        load_route_layer(route_path, 1, 4)


def test_candidate_argument_parsers_and_spearman() -> None:
    assert _parse_ints("4,8,8,16", name="widths") == (4, 8, 16)
    assert _parse_orders("lpt,reverse_even") == ("lpt", "reverse_even")
    assert _spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)
    assert _spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="selected from"):
        _parse_orders("lpt,unknown")
