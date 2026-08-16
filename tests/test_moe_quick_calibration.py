from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
COST_MODEL = ROOT / "cpu_moe_schedule_optimization" / "cost_model"
sys.path.insert(0, str(COST_MODEL))

import quick_calibration  # noqa: E402


def _service_rows(widths: tuple[int, ...], base: float) -> list[dict]:
    return [
        {"threads": width, "aggregate_rate": base * width}
        for width in widths
    ]


def _quick_probe() -> dict:
    widths = (1, 2, 4)
    services = {
        name: {"rows": _service_rows(widths, base)}
        for name, base in {
            "matrix_flops": 100.0,
            "gemm_core_flops": 80.0,
            "l1_bytes": 60.0,
            "l2_bytes": 40.0,
            "llc_bytes": 20.0,
            "dram_bytes": 10.0,
        }.items()
    }
    services["panel_range_restart"] = {"panel_range_restart_ns": 7.0}
    return {
        "schema_version": 2,
        "kind": "moe_analytic_service_probe",
        "machine": {
            "id": "test-host",
            "architecture": "aarch64",
            "logical_cpus": 4,
            "cpu_ids": [0, 1, 2, 3],
            "cores_per_rank": 4,
        },
        "topology": {
            "rank_cpu_ids": [0, 1, 2, 3],
            "llc_domains": [
                {"id": "0", "cpu_ids": [0, 1, 2, 3], "capacity_bytes": 8 * 1024 * 1024}
            ],
            "dram_scope": "numa_rank",
        },
        "kernel": {
            "packed_panel_columns": 16,
            "bfmmla_flops_per_instruction": 32,
            "bfmmla_instructions_per_cycle": 4,
            "frontend_instructions_per_cycle": 5,
        },
        "caches": {
            "l1d_bytes_per_core": 64 * 1024,
            "l2_bytes_per_core": 1024 * 1024,
            "llc_bytes_per_rank": 8 * 1024 * 1024,
        },
        "services": services,
    }


def test_quick_widths_include_topology_knees_and_separate_planner_domain() -> None:
    assert quick_calibration.quick_service_widths(96, (96,)) == (1, 2, 4, 8, 16, 48, 96)
    assert quick_calibration.quick_service_widths(80, (40, 40)) == (1, 2, 4, 8, 16, 20, 40, 80)
    assert quick_calibration.quick_supported_widths(96, (96,)) == (1, 2, 4, 8, 16, 32, 48, 64, 96)


def test_quick_calibration_returns_valid_model_and_writes_atomically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    probe = _quick_probe()
    monkeypatch.setattr(quick_calibration, "_validate_host", lambda: None)
    monkeypatch.setattr(
        quick_calibration,
        "_collect_quick_service_probe",
        lambda cpu_ids, seed, report: (probe, {}, (1, 2, 4)),
    )
    output = tmp_path / "machine.json"

    result = quick_calibration.calibrate_moe_planner_quick(
        (0, 1, 2, 3),
        output=output,
        machine_id="quick-test",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.calibration.machine_id == "quick-test"
    assert result.cpu_ids == (0, 1, 2, 3)
    assert result.service_widths == (1, 2, 4)
    assert result.supported_widths == (1, 2, 4)
    assert payload["provenance"]["quick_calibration"] == {
        "version": 1,
        "service_widths": [1, 2, 4],
        "supported_widths": [1, 2, 4],
        "seed": 20260814,
        "operator_residual_training": False,
    }


def test_quick_calibration_refuses_to_replace_profile_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(quick_calibration, "_validate_host", lambda: None)
    output = tmp_path / "machine.json"
    output.write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        quick_calibration.calibrate_moe_planner_quick((0,), output=output)
