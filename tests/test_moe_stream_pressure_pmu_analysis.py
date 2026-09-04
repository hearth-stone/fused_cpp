from __future__ import annotations

import pytest

from optimizations.fused_moe_sve.benchmarks.analyze_stream_pressure_pmu import fit_loco, fit_loco_affine


def _points(feature: str, values: list[float]) -> list[dict[str, float]]:
    return [
        {"count": float(index), "response_ms": 2.0 * value, feature: value}
        for index, value in enumerate(values)
    ]


def test_fit_loco_recovers_anchored_linear_feature() -> None:
    result = fit_loco(_points("pressure", [0.0, 1.0, 2.0, 3.0]), "pressure")
    assert result["full_slope_ms_per_unit"] == pytest.approx(2.0)
    assert result["loco_mae_ms"] == pytest.approx(0.0)
    assert result["loco_false_nonpositive"] == 0
    assert result["eligible_for_monotone_pressure"] is True


def test_fit_loco_rejects_zero_energy_feature() -> None:
    with pytest.raises(ValueError, match="no positive fit energy"):
        fit_loco(_points("pressure", [0.0, 0.0, 0.0]), "pressure")


def test_fit_loco_marks_negative_pressure_ineligible() -> None:
    result = fit_loco(_points("pressure", [0.0, -1.0, 2.0, 3.0]), "pressure")
    assert result["feature_nonnegative"] is False
    assert result["eligible_for_monotone_pressure"] is False


def test_fit_loco_affine_recovers_intercept_and_slope() -> None:
    points = [
        {"count": float(index), "span_ms": 0.5 + 2.0 * value, "pressure": value}
        for index, value in enumerate([0.0, 1.0, 2.0, 3.0])
    ]
    result = fit_loco_affine(points, "pressure")
    assert result["full_intercept_ms"] == pytest.approx(0.5)
    assert result["full_slope_ms_per_unit"] == pytest.approx(2.0)
    assert result["loco_mae_ms"] == pytest.approx(0.0)
