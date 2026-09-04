from __future__ import annotations

import json
from pathlib import Path

import pytest

from optimizations.fused_moe_sve.benchmarks.fit_phase_reaccount_calibration import (
    LOCKED_HOLDOUT_SHA256,
    _best_floor_scale,
    _read_training_artifact,
    _validate_phase_artifact,
)


def test_floor_scale_fit_recovers_plateau_and_linear_region() -> None:
    floor, scale, loss = _best_floor_scale(
        [1.0, 2.0, 4.0, 8.0],
        [3.0, 3.0, 4.0, 8.0],
        denominator=lambda value: value,
    )

    assert floor == pytest.approx(3.0)
    assert scale == pytest.approx(1.0)
    assert loss == pytest.approx(0.0)


def test_phase_artifact_rejects_holdout_routes() -> None:
    with pytest.raises(ValueError, match="overlaps locked holdout routes"):
        _validate_phase_artifact(
            {
                "kind": "moe_isolated_phase_reaccount_fit",
                "artifact_role": "fit_only",
                "points": [{"routes": 1}],
            }
        )


def test_training_reader_rejects_locked_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"kind": "anything"}), encoding="utf-8")
    digest = next(iter(LOCKED_HOLDOUT_SHA256))
    monkeypatch.setattr(
        "optimizations.fused_moe_sve.benchmarks.fit_phase_reaccount_calibration._sha256",
        lambda unused: digest,
    )

    with pytest.raises(ValueError, match="locked holdout artifact"):
        _read_training_artifact(path)
