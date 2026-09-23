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
    seeds = []
    monkeypatch.setattr(quick_calibration, "_validate_host", lambda: None)

    def collect(cpu_ids, seed, report):
        seeds.append(seed)
        return probe, {}, (1, 2, 4)

    monkeypatch.setattr(quick_calibration, "_collect_quick_service_probe", collect)
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
    assert seeds == [20260814, 20261814, 20262814]
    assert payload["provenance"]["quick_calibration"] == {
        "version": 2,
        "service_widths": [1, 2, 4],
        "supported_widths": [1, 2, 4],
        "seed": 20260814,
        "service_repeats": 3,
        "operator_residual_training": False,
    }
    assert result.payload == payload


def test_quick_calibration_takes_the_median_of_repeated_service_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scales = iter((1.0, 3.0, 1.2))
    monkeypatch.setattr(quick_calibration, "_validate_host", lambda: None)

    def collect(cpu_ids, seed, report):
        probe = _quick_probe()
        scale = next(scales)
        for row in probe["services"]["gemm_core_flops"]["rows"]:
            row["aggregate_rate"] *= scale
        return probe, {}, (1, 2, 4)

    monkeypatch.setattr(quick_calibration, "_collect_quick_service_probe", collect)
    result = quick_calibration.calibrate_moe_planner_quick((0, 1, 2, 3), machine_id="median-test")

    assert result.calibration.service_rate("gemm_core_flops", 1) == pytest.approx(80.0 * 1.2)


def test_quick_calibration_rejects_non_positive_service_repeats(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quick_calibration, "_validate_host", lambda: None)
    with pytest.raises(ValueError, match="service_repeats"):
        quick_calibration.calibrate_moe_planner_quick((0,), service_repeats=0)


def _machine_payload(monkeypatch: pytest.MonkeyPatch) -> dict:
    probe = _quick_probe()
    monkeypatch.setattr(quick_calibration, "_validate_host", lambda: None)
    monkeypatch.setattr(
        quick_calibration,
        "_collect_quick_service_probe",
        lambda cpu_ids, seed, report: (probe, {}, (1, 2, 4)),
    )
    return dict(quick_calibration.calibrate_moe_planner_quick((0, 1, 2, 3), machine_id="train-test").payload)


def _shape_profile(rows: list[dict]) -> dict:
    return {
        "isolated": rows,
        "expert_shape": {"hidden_size": 64, "intermediate_size": 32, "activation": "silu", "dtype": "bf16"},
        "kernel": {"m_tail_policy": "xbyak_exact_m", "backend_n_tile": 8},
        "parallelism": {"mode": "tp", "degree": 2, "global_experts": 8, "local_experts": 8},
        "target": {"concurrent_ranks": 1},
    }


def _physics(payload: dict):
    from analytic_model import AnalyticMachineCalibration
    from validate_analytic_model import _model_from_profile

    return _model_from_profile(
        AnalyticMachineCalibration.from_dict(payload), _shape_profile([]), down_output_element_bytes=2
    )


def test_operator_overhead_training_fits_the_shape_and_leaves_the_machine_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from analytic_model import AnalyticMachineCalibration

    machine = _machine_payload(monkeypatch)
    physics = _physics(machine)
    captured = {}

    def measure(**kwargs):
        captured.update(kwargs)
        rows = [
            {"routes": routes, "threads": width, "median_ns": physics.T_iso(routes, width) + 25_000.0 + 40.0 * routes}
            for routes in kwargs["routes"]
            for width in kwargs["widths"]
        ]
        return rows, 8

    monkeypatch.setattr(quick_calibration, "_measure_isolated_training_rows", measure)
    payload, report = quick_calibration.train_quick_operator_overheads(
        machine,
        hidden_size=64,
        intermediate_size=32,
        global_experts=8,
        local_experts=8,
        cpu_ids=(0, 1, 2, 3),
        mode="tp",
        degree=2,
    )

    assert captured["widths"] == (1, 2, 4)
    assert captured["routes"] == quick_calibration.QUICK_TRAIN_ROUTES
    assert payload["stage_scales"]["w13"] == pytest.approx(1.0)
    assert payload["overheads"]["expert_fixed_ns"] == pytest.approx(25_000.0)
    assert payload["overheads"]["route_ns"] == pytest.approx(40.0)
    assert [entry["threads"] for entry in payload["overheads"]["by_width"]] == [1, 2, 4]
    assert report["by_width"]["mape"] == pytest.approx(0.0, abs=1e-9)
    training = payload["provenance"]["quick_calibration"]["operator_residual_training"]
    assert (training["hidden_size"], training["intermediate_size"], training["degree"]) == (64, 32, 2)
    assert machine["overheads"]["expert_fixed_ns"] == 0.0
    AnalyticMachineCalibration.from_dict(payload)


def test_operator_overhead_training_rejects_widths_beyond_the_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    machine = _machine_payload(monkeypatch)
    with pytest.raises(ValueError, match="exceed"):
        quick_calibration.train_quick_operator_overheads(
            machine, hidden_size=64, intermediate_size=32, global_experts=8, local_experts=8, cpu_ids=(0, 1)
        )


def test_width_overheads_recover_a_distinct_pair_per_width(monkeypatch: pytest.MonkeyPatch) -> None:
    from build_analytic_calibration import fit_width_overheads

    machine = _machine_payload(monkeypatch)
    machine["stage_scales"] = {"w13": 1.0, "w2": 1.0}
    physics = _physics(machine)
    truth = {1: (30_000.0, 50.0), 2: (20_000.0, 40.0), 4: (10_000.0, 0.0)}
    rows = [
        {"routes": routes, "threads": width, "median_ns": physics.T_iso(routes, width) + fixed + route * routes}
        for width, (fixed, route) in truth.items()
        for routes in (1, 12, 192)
    ]

    fit = fit_width_overheads(machine, _shape_profile(rows))

    for entry in fit["by_width"]:
        fixed, route = truth[entry["threads"]]
        assert entry["expert_fixed_ns"] == pytest.approx(fixed)
        assert entry["route_ns"] == pytest.approx(route, abs=1e-6)


def test_quick_calibration_refuses_to_replace_profile_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(quick_calibration, "_validate_host", lambda: None)
    output = tmp_path / "machine.json"
    output.write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        quick_calibration.calibrate_moe_planner_quick((0,), output=output)
