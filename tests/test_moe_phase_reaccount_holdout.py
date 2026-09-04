from __future__ import annotations

import json
from pathlib import Path

import pytest

from optimizations.fused_moe_sve.benchmarks.evaluate_phase_reaccount_holdout import (
    _error_metrics,
    _read_locked,
    _spearman,
)
from optimizations.fused_moe_sve.benchmarks.fit_phase_reaccount_calibration import (
    LOCKED_HOLDOUT_SHA256,
)


def test_error_metrics_reports_distribution() -> None:
    metrics = _error_metrics([0.1, -0.2, 0.3])

    assert metrics["mape"] == pytest.approx(0.2)
    assert metrics["median_absolute_relative_error"] == pytest.approx(0.2)
    assert metrics["max_absolute_relative_error"] == pytest.approx(0.3)


def test_spearman_detects_same_and_reversed_order() -> None:
    assert _spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)
    assert _spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == pytest.approx(-1.0)


def test_holdout_reader_rejects_unregistered_artifact(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps({"kind": "anything"}), encoding="utf-8")

    with pytest.raises(ValueError, match="not in the locked holdout set"):
        _read_locked(path)


def test_holdout_registry_contains_route_and_real_trace_hashes() -> None:
    assert "61c0e929ad7a575831f972bdbb2c690df243e028062cdbb10044e66d15867e34" in LOCKED_HOLDOUT_SHA256
    assert "b23ece71ffecbbbc839747ac4be679b34c02c7998a3737764aeee63964ddb67f" in LOCKED_HOLDOUT_SHA256
