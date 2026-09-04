from __future__ import annotations

import pytest

from optimizations.fused_moe_sve.benchmarks.analyze_stream_pressure_pmu_paired import _paired


def test_paired_subtracts_same_round_baseline() -> None:
    assert _paired([2.0, 5.0, 7.0], [1.0, 3.0, 4.0]) == [1.0, 2.0, 3.0]


def test_paired_rejects_different_lengths() -> None:
    with pytest.raises(ValueError, match="lengths differ"):
        _paired([1.0], [1.0, 2.0])
