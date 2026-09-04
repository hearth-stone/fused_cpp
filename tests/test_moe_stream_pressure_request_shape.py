from __future__ import annotations

from optimizations.fused_moe_sve.benchmarks.analyze_stream_pressure_request_shape import _summary


def test_summary_preserves_predeclared_sign_gate() -> None:
    result = _summary([0.3, 0.1, 0.2])
    assert result == {"median": 0.2, "p10": 0.1, "p90": 0.3}
