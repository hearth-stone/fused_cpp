from __future__ import annotations

import pytest

from optimizations.fused_moe_sve.benchmarks.bench_stream_pressure_pmu_paired import (
    DEFAULT_MODES,
    validate_modes,
)


def test_default_modes_cover_count_sweep_and_layout_control() -> None:
    assert validate_modes(list(DEFAULT_MODES)) == DEFAULT_MODES
    assert DEFAULT_MODES[0] == "isolated_head"
    assert "eight_2t_same_head" in DEFAULT_MODES
    assert "four_1t_same_head" in DEFAULT_MODES


def test_validate_modes_requires_unique_isolated_control() -> None:
    with pytest.raises(ValueError, match="include isolated"):
        validate_modes(["wide16_same_head"])
    with pytest.raises(ValueError, match="must be unique"):
        validate_modes(["isolated_head", "isolated_head"])
