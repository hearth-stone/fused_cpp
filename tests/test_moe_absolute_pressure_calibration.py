from __future__ import annotations

import json
from pathlib import Path

import pytest

from optimizations.fused_moe_sve.benchmarks.fit_absolute_pressure_calibration import (
    COARSE_CAPACITY,
    COARSE_MULTIPLIER,
    PRESSURE_KIND,
    SCHEMA_VERSION,
    _assert_exclusive_inputs,
    _validate_pressure_artifact,
    absolute_rows,
    contrast_rows,
    identifiability_report,
    joint_loss,
    signed_bias,
)
from optimizations.fused_moe_sve.benchmarks.fit_phase_reaccount_calibration import (
    LOCKED_HOLDOUT_SHA256,
)


def _payload() -> dict:
    modes = {
        "isolated_head": {"target_span": {"median_ms": 1.0, "samples_ms": [1.0] * 31}},
        "isolated_after_1": {"target_span": {"median_ms": 1.1, "samples_ms": [1.1] * 31}},
        "same_llc_head_n1": {"target_span": {"median_ms": 1.4, "samples_ms": [1.4] * 31}},
        "cross_llc_head_n1": {"target_span": {"median_ms": 1.05, "samples_ms": [1.05] * 31}},
        "same_llc_after_1_n1": {"target_span": {"median_ms": 1.3, "samples_ms": [1.3] * 31}},
        "cross_llc_after_1_n1": {"target_span": {"median_ms": 1.12, "samples_ms": [1.12] * 31}},
        "split_head_n8": {"target_span": {"median_ms": 1.35, "samples_ms": [1.35] * 31}},
    }
    return {
        "kind": PRESSURE_KIND,
        "schema_version": SCHEMA_VERSION,
        "modes": modes,
    }


def test_pressure_artifact_rejects_wrong_kind() -> None:
    with pytest.raises(ValueError, match="wrong kind"):
        _validate_pressure_artifact({"kind": "moe_gather_injection_overlap_probe"})


def test_exclusive_inputs_reject_holdout_and_duplicates() -> None:
    holdout = next(iter(LOCKED_HOLDOUT_SHA256))
    with pytest.raises(ValueError, match="locked holdout"):
        _assert_exclusive_inputs("aaa", holdout)
    with pytest.raises(ValueError, match="distinct SHA256"):
        _assert_exclusive_inputs("abc", "abc")


def test_absolute_and_contrast_rows_use_isolated_relative_values() -> None:
    payload = _payload()

    absolute = absolute_rows(payload)
    contrast = contrast_rows(payload)

    assert absolute["same_llc_head_n1"] == pytest.approx(0.4)
    assert absolute["cross_llc_head_n1"] == pytest.approx(0.05)
    assert absolute["split_head_n8"] == pytest.approx(0.35)
    assert "isolated_head" not in absolute
    assert contrast["same_minus_cross_head_n1"] == pytest.approx(0.35)
    assert contrast["same_minus_cross_after_1_n1"] == pytest.approx(0.18)


def test_joint_loss_does_not_let_absolute_rows_dominate() -> None:
    measured_abs = {f"row{index}": 1.0 for index in range(10)}
    predicted_abs = {name: 2.0 for name in measured_abs}
    measured_contrast = {"c1": 0.5, "c2": 0.5}
    predicted_contrast = {"c1": 0.5, "c2": 0.5}

    loss = joint_loss(measured_abs, predicted_abs, measured_contrast, predicted_contrast)

    assert loss["absolute_mae_ms"] == pytest.approx(1.0)
    assert loss["contrast_mae_ms"] == pytest.approx(0.0)
    assert loss["joint_mae_ms"] == pytest.approx(1.0)


def test_identifiability_rejects_boundary_and_flat_valley() -> None:
    bounds_c = (COARSE_CAPACITY[0], COARSE_CAPACITY[-1])
    bounds_m = (COARSE_MULTIPLIER[0], COARSE_MULTIPLIER[-1])
    boundary = identifiability_report(
        (0.1, COARSE_CAPACITY[0], 1.0),
        [(0.1, COARSE_CAPACITY[0], 1.0), (0.11, 0.4, 1.0)],
        capacity_bounds=bounds_c,
        multiplier_bounds=bounds_m,
    )
    flat = identifiability_report(
        (1.0, 0.8, 1.0),
        [(1.0, 0.8, 1.0), (1.04, 0.4, 1.0), (1.04, 1.6, 2.0)],
        capacity_bounds=bounds_c,
        multiplier_bounds=bounds_m,
    )

    assert boundary["on_search_boundary"] is True
    assert boundary["identifiable"] is False
    assert flat["degenerate_flat_valley"] is True
    assert flat["identifiable"] is False


def test_signed_bias_flags_systematic_same_sign_error() -> None:
    ok = signed_bias([0.01, -0.02, 0.005])
    bad = signed_bias([0.04, 0.05, 0.06])

    assert ok["systematic"] is False
    assert bad["systematic"] is True
    assert bad["all_same_sign"] is True


def test_training_reader_rejects_locked_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from optimizations.fused_moe_sve.benchmarks.fit_absolute_pressure_calibration import (
        _read_pressure_artifact,
    )

    path = tmp_path / "input.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    digest = next(iter(LOCKED_HOLDOUT_SHA256))
    monkeypatch.setattr(
        "optimizations.fused_moe_sve.benchmarks.fit_phase_reaccount_calibration._sha256",
        lambda unused: digest,
    )

    with pytest.raises(ValueError, match="locked holdout artifact"):
        _read_pressure_artifact(path)
