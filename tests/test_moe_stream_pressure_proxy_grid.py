from __future__ import annotations

import pytest

from optimizations.fused_moe_sve.benchmarks.analyze_stream_pressure_proxy_grid import (
    _relative_drift,
    _same_nonzero_direction,
    _slope,
    mode_name,
)


def test_grid_mode_names_keep_count_and_width_explicit() -> None:
    assert mode_name(6, 1) == "grid6_1t_same_head"
    assert mode_name(8, 2) == "grid8_2t_same_head"


def test_proxy_slope_is_anchored_and_nonnegative() -> None:
    points = [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 4.0}]
    assert _slope(points, "x", "y") == pytest.approx(2.0)
    assert _slope([{"x": 1.0, "y": -1.0}], "x", "y") == 0.0


def test_relative_drift_uses_larger_parameter_as_scale() -> None:
    assert _relative_drift(1.0, 0.8) == pytest.approx(0.2)
    assert _relative_drift(0.0, 0.0) == 0.0


def test_direction_requires_nonzero_prediction() -> None:
    assert _same_nonzero_direction(1.0, 2.0)
    assert _same_nonzero_direction(-1.0, -2.0)
    assert not _same_nonzero_direction(0.0, 2.0)
    assert not _same_nonzero_direction(-1.0, 2.0)
