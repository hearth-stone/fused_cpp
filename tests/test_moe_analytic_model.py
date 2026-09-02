from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
COST_MODEL = ROOT / "cpu_moe_schedule_optimization" / "cost_model"
PLANNERS = ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path[:0] = [str(COST_MODEL), str(PLANNERS)]

from analytic_model import (  # noqa: E402
    _SHARED_RESOURCES,
    AnalyticMachineCalibration,
    AnalyticMoeCostModel,
    CacheCalibration,
    LlcDomainCalibration,
    RuntimeOverheads,
    SaturatingServiceCurve,
    WideTeamPressureCalibration,
    analytic_candidate_shapes,
)
from analytic_probe_geometry import (  # noqa: E402
    b_only_geometry,
    m12_gemm_geometry,
    read_cache_info,
    read_llc_domains,
)
from build_analytic_calibration import (  # noqa: E402
    build_calibration,
    fit_nonnegative_residuals,
    select_curve,
)
from interval_planner import IntervalPlanner  # noqa: E402
from planned_moe import PlannedMoE  # noqa: E402
from full_stage_geometry import full_stage_geometry  # noqa: E402
from validate_analytic_model import build_validation_report  # noqa: E402
from fused_cpp.moe import MoePlannerRuntime, PreparedBF16TiledFusedMoEWeights  # noqa: E402


def _curve(
    single: float,
    saturated: float,
    threads: int,
    *,
    curve: str = "power",
) -> SaturatingServiceCurve:
    return SaturatingServiceCurve(
        single_thread_rate=single,
        saturated_rate=saturated,
        saturation_threads=threads,
        curve=curve,
    )


def _calibration(
    *,
    cores: int = 8,
    supported_widths: tuple[int, ...] = (1, 2, 4, 8),
    dram_saturated_rate: float = 80e9,
) -> AnalyticMachineCalibration:
    return AnalyticMachineCalibration(
        machine_id=f"synthetic-{cores}c",
        cores_per_rank=cores,
        caches=CacheCalibration(
            l1d_bytes_per_core=64 * 1024,
            l2_bytes_per_core=256 * 1024,
            llc_bytes_per_rank=2 * 1024 * 1024,
        ),
        matrix_flops=_curve(100e9, 100e9 * cores, cores),
        gemm_core_flops=_curve(60e9, 60e9 * cores, cores),
        frontend_instructions=_curve(20e9, 20e9 * cores, cores),
        l1_bytes=_curve(100e9, 100e9 * cores, cores, curve="shared_bottleneck"),
        l2_bytes=_curve(50e9, 50e9 * cores, cores, curve="shared_bottleneck"),
        llc_bytes=_curve(25e9, 25e9 * cores, cores, curve="shared_bottleneck"),
        dram_bytes=_curve(10e9, dram_saturated_rate, cores, curve="shared_bottleneck"),
        epilogue_elements=_curve(1e9, 1e9 * cores, cores),
        overheads=RuntimeOverheads(
            call_setup_ns=500.0,
            expert_fixed_ns=200.0,
            route_ns=2.0,
            stage_fixed_ns=100.0,
            panel_range_restart_ns=3.0,
        ),
        supported_widths=supported_widths,
        relative_uncertainty=0.04,
    )


def _model(
    calibration: AnalyticMachineCalibration | None = None,
) -> AnalyticMoeCostModel:
    return AnalyticMoeCostModel(
        calibration or _calibration(),
        hidden_size=64,
        intermediate_size=32,
        global_experts=8,
        local_experts=8,
    )


def test_task_stage_phases_exposes_isolated_stage_time_and_packed_b_bytes() -> None:
    model = _model()
    prediction = model.predict_expert(12, 2)

    assert model.task_stage_phases("w13", 12, 2) == (
        (prediction.w13_ns, prediction.w13_demand.stage_bytes),
    )
    assert model.task_stage_phases("w2", 12, 2) == (
        (prediction.w2_ns, prediction.w2_demand.stage_bytes),
    )
    with pytest.raises(ValueError, match="unsupported stage"):
        model.task_stage_phases("merge", 12, 2)


def _runtime_weights(experts: int = 8) -> PreparedBF16TiledFusedMoEWeights:
    import torch

    packed = torch.empty((experts, 1), dtype=torch.bfloat16)
    return PreparedBF16TiledFusedMoEWeights(
        w13=(packed, 64, 64),
        w2=(packed, 32, 64),
        fused_silu=True,
        gemm_backend=1,
        backend_n_tile=8,
        backend_name="arm_sve_bf16",
    )


def test_runtime_reuses_persisted_t_iso_without_recomputing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    calibration = _calibration()
    topk_ids = torch.tensor([[0], [0], [1], [1], [1], [2], [3], [3]], dtype=torch.int32)
    kwargs = {
        "hidden_size": 64,
        "intermediate_size": 32,
        "global_experts": 8,
        "local_experts": 8,
        "cpu_ids": tuple(range(8)),
        "cost_cache_dir": tmp_path,
    }
    first = MoePlannerRuntime(calibration, **kwargs)
    first_plan = first.plan_for_dispatch(
        _runtime_weights(),
        topk_ids,
        num_threads=8,
        activation="silu",
        global_num_experts=-1,
    )
    assert first_plan is not None
    assert first.last_plan["cost_disk_cache"]["status"] == "stored"
    assert first.last_plan["cost_disk_cache"]["total_entries"] > 0

    second = MoePlannerRuntime(calibration, **kwargs)

    def fail_predict(*args, **kwargs):
        raise AssertionError("persisted T_iso should bypass predict_expert")

    monkeypatch.setattr(second.model, "predict_expert", fail_predict)
    second_plan = second.plan_for_dispatch(
        _runtime_weights(),
        topk_ids,
        num_threads=8,
        activation="silu",
        global_num_experts=-1,
    )
    assert second_plan is not None
    assert second.last_plan["cost_disk_cache"]["status"] == "hit"
    assert second.last_plan["cost_disk_cache"]["loaded_entries"] > 0
    assert second_plan.task_threads.tolist() == first_plan.task_threads.tolist()
    assert second_plan.task_core_begins.tolist() == first_plan.task_core_begins.tolist()

    disabled = MoePlannerRuntime(calibration, **{**kwargs, "cost_cache_dir": None})
    disabled_plan = disabled.plan_for_dispatch(
        _runtime_weights(),
        topk_ids,
        num_threads=8,
        activation="silu",
        global_num_experts=-1,
    )
    assert disabled_plan is not None
    assert disabled.last_plan["cost_disk_cache"]["status"] == "disabled"


def test_service_curve_uses_two_hardware_anchors() -> None:
    curve = _curve(100.0, 400.0, 4)

    assert curve.rate(1) == pytest.approx(100.0)
    assert curve.rate(2) == pytest.approx(200.0)
    assert curve.rate(4) == pytest.approx(400.0)
    assert curve.rate(16) == pytest.approx(400.0)


def test_packed_b_retention_knee_is_distinct_from_general_l2_capacity() -> None:
    calibration = replace(
        _calibration(),
        caches=replace(
            _calibration().caches,
            l2_effective_fraction=0.75,
            l2_b_reuse_effective_fraction=0.125,
            l2_b_reuse_miss_floor=0.1,
            l2_b_reuse_miss_at_capacity=0.8,
            l2_b_reuse_miss_ceiling=0.95,
        ),
    )
    model = _model(calibration)
    working_set = 0.5 * calibration.caches.l2_bytes_per_core

    assert model._l2_miss_fraction(working_set) == 0.0
    assert model._l2_b_reuse_miss_fraction(working_set) > calibration.caches.l2_b_reuse_miss_floor


def test_shared_bottleneck_curve_preserves_linear_low_thread_scaling() -> None:
    curve = _curve(40.0, 360.0, 86, curve="shared_bottleneck")

    assert curve.rate(1) == pytest.approx(40.0)
    assert curve.rate(24) > 280.0
    assert curve.rate(86) == pytest.approx(360.0)
    assert curve.rate(128) == pytest.approx(360.0)


def test_piecewise_service_curve_interpolates_measured_widths_and_saturates() -> None:
    curve = SaturatingServiceCurve(
        single_thread_rate=40.0,
        saturated_rate=180.0,
        saturation_threads=8,
        curve="piecewise_linear",
        points=((1, 40.0), (2, 70.0), (4, 120.0), (8, 180.0)),
    )

    assert curve.rate(1) == pytest.approx(40.0)
    assert curve.rate(3) == pytest.approx(95.0)
    assert curve.rate(6) == pytest.approx(150.0)
    assert curve.rate(16) == pytest.approx(180.0)
    assert SaturatingServiceCurve.from_dict(curve.to_dict()) == curve


def test_cache_probe_geometry_uses_detected_hardware_capacity() -> None:
    l1_gemm = m12_gemm_geometry(64 * 1024, 16, cache_fraction=0.625)
    l2_gemm = m12_gemm_geometry(2 * 1024 * 1024, 16, cache_fraction=0.5)
    l1_load = b_only_geometry(64 * 1024, 16, cache_fraction=0.5)
    l2_load = b_only_geometry(2 * 1024 * 1024, 16, cache_fraction=0.5)

    assert (l1_gemm.k, l1_gemm.n, l1_gemm.working_set_bytes) == (728, 16, 40_768)
    assert l2_gemm.k == 18_720
    assert l2_gemm.working_set_bytes <= 1024 * 1024
    assert (l1_load.k, l1_load.n, l1_load.working_set_bytes) == (1024, 16, 32 * 1024)
    assert (l2_load.k, l2_load.n, l2_load.working_set_bytes) == (4096, 128, 1024 * 1024)


def test_cache_info_reads_linux_sysfs_hierarchy(tmp_path: Path) -> None:
    cache_root = tmp_path / "cpu3" / "cache"
    entries = (
        ("index0", "1", "Data", "64K", "64"),
        ("index1", "1", "Instruction", "64K", "64"),
        ("index2", "2", "Unified", "2M", "64"),
        ("index3", "3", "Unified", "96M", "64"),
    )
    for name, level, cache_type, size, line_size in entries:
        index = cache_root / name
        index.mkdir(parents=True)
        (index / "level").write_text(level, encoding="utf-8")
        (index / "type").write_text(cache_type, encoding="utf-8")
        (index / "size").write_text(size, encoding="utf-8")
        (index / "coherency_line_size").write_text(line_size, encoding="utf-8")

    assert read_cache_info(3, sysfs_cpu_root=tmp_path) == {
        "l1d_bytes_per_core": 64 * 1024,
        "l2_bytes_per_core": 2 * 1024 * 1024,
        "llc_bytes_per_rank": 96 * 1024 * 1024,
        "cache_line_bytes": 64,
    }


def test_llc_domains_follow_sysfs_shared_cpu_lists(tmp_path: Path) -> None:
    for cpu in range(4):
        index = tmp_path / f"cpu{cpu}" / "cache" / "index3"
        index.mkdir(parents=True)
        domain = "0-1" if cpu < 2 else "2-3"
        (index / "level").write_text("3", encoding="utf-8")
        (index / "type").write_text("Unified", encoding="utf-8")
        (index / "size").write_text("8M", encoding="utf-8")
        (index / "shared_cpu_list").write_text(domain, encoding="utf-8")
        (index / "id").write_text("0" if cpu < 2 else "1", encoding="utf-8")

    assert read_llc_domains((0, 1, 2, 3), sysfs_cpu_root=tmp_path) == (
        {"id": "0", "cpu_ids": [0, 1], "capacity_bytes": 8 * 1024 * 1024},
        {"id": "1", "cpu_ids": [2, 3], "capacity_bytes": 8 * 1024 * 1024},
    )


def _service_probe() -> dict:
    widths = (1, 2, 4, 8)

    def service(rates: tuple[float, ...]) -> dict:
        return {"rows": [{"threads": threads, "aggregate_rate": rate} for threads, rate in zip(widths, rates)]}

    return {
        "kind": "moe_analytic_service_probe",
        "machine": {"id": "probe-host", "cores_per_rank": 8},
        "kernel": {
            "bfmmla_flops_per_instruction": 32,
            "bfmmla_instructions_per_cycle": 4,
            "frontend_instructions_per_cycle": 5,
        },
        "caches": {
            "l1d_bytes_per_core": 64 * 1024,
            "l2_bytes_per_core": 2 * 1024 * 1024,
            "llc_bytes_per_rank": 8 * 1024 * 1024,
        },
        "services": {
            "panel_range_restart": {"panel_range_restart_ns": 7.5},
            "w13_fused_panel_range_restart": {"panel_range_restart_ns": 41.0},
            "gemm_core_flops": service((60.0, 115.0, 210.0, 360.0)),
            "gemm_l2_flops": service((45.0, 85.0, 150.0, 250.0)),
            "matrix_flops": service((100.0, 190.0, 350.0, 600.0)),
            "l1_bytes": service((80.0, 155.0, 290.0, 500.0)),
            "l2_bytes": service((50.0, 98.0, 185.0, 320.0)),
            "llc_bytes": service((20.0, 38.0, 68.0, 70.0)),
            "dram_bytes": service((10.0, 19.0, 35.0, 34.0)),
        },
    }


def test_thin_calibration_extracts_private_and_shared_service_curves() -> None:
    probe = _service_probe()

    private, private_fit = select_curve(probe, "matrix_flops")
    shared, shared_fit = select_curve(probe, "dram_bytes")
    calibration, report = build_calibration(
        probe,
        machine_id="test-thin",
        l2_effective_fraction=0.75,
        llc_effective_fraction=0.625,
        l2_b_reuse_miss_floor=0.18,
        l2_b_reuse_miss_at_capacity=0.62,
        l2_b_reuse_miss_ceiling=0.87,
        relative_uncertainty=0.15,
    )

    assert private == {
        "single_thread_rate": 100.0,
        "saturated_rate": 600.0,
        "saturation_threads": 8,
        "curve": "power",
    }
    assert shared["curve"] == "piecewise_linear"
    assert shared["saturation_threads"] == 8
    assert [point["rate"] for point in shared["points"]] == pytest.approx((10.0, 19.0, 34.5, 34.5))
    assert private_fit["rows"][-1]["relative_error"] == pytest.approx(0.0)
    assert shared_fit["rows"][-1]["predicted_rate"] == pytest.approx(shared["saturated_rate"])
    assert calibration["services"]["dram_bytes"] == shared
    assert calibration["schema_version"] == 2
    assert calibration["kernel"]["backend_n_tile"] == 8
    assert calibration["services"]["gemm_core_flops"]["single_thread_rate"] == pytest.approx(60.0)
    assert calibration["services"]["frontend_instructions"]["single_thread_rate"] == pytest.approx(3.90625)
    assert calibration["caches"]["l2_b_reuse_effective_fraction"] == pytest.approx(0.75)
    assert calibration["caches"]["l2_b_reuse_miss_floor"] == pytest.approx(0.18)
    assert calibration["caches"]["l2_b_reuse_miss_at_capacity"] == pytest.approx(0.62)
    assert calibration["caches"]["l2_b_reuse_miss_ceiling"] == pytest.approx(0.87)
    assert calibration["planner"]["supported_widths"] == [1, 2, 4, 8]
    assert calibration["overheads"]["panel_range_restart_ns"] == pytest.approx(7.5)
    assert calibration["overheads"]["w13_panel_range_restart_ns"] == pytest.approx(41.0)
    assert calibration["overheads"]["w2_panel_range_restart_ns"] == pytest.approx(7.5)
    assert calibration["provenance"]["contention_measurements_used"] is False
    assert calibration["provenance"]["gemm_core_service"] == "m12_l1_hot_full_no_store"
    assert calibration["provenance"]["panel_range_restart_service"] == {
        "w13": "m12_l1_hot_fused_w13_extra_n_range",
        "w2": "m12_l1_hot_full_no_store_extra_n_range",
    }
    assert calibration["provenance"]["l2_b_retention_calibration"]["kind"].startswith("independent_")
    assert report["dram_bytes"] == shared_fit
    assert report["dram_bytes"]["leave_one_sampled_width_out"]["points"] == 2
    assert report["gemm_l2_flops"]["curve"]["single_thread_rate"] == pytest.approx(45.0)


def test_topology_aware_llc_service_sums_domains_then_applies_rank_cap() -> None:
    domain_curve = SaturatingServiceCurve(
        single_thread_rate=40.0,
        saturated_rate=300.0,
        saturation_threads=4,
        curve="piecewise_linear",
        points=((1, 40.0), (2, 200.0), (4, 300.0)),
    )
    rank_curve = SaturatingServiceCurve(
        single_thread_rate=40.0,
        saturated_rate=550.0,
        saturation_threads=8,
        curve="piecewise_linear",
        points=((1, 40.0), (2, 90.0), (4, 300.0), (8, 550.0)),
    )
    base = _calibration(cores=8)
    calibration = replace(
        base,
        caches=replace(base.caches, llc_bytes_per_rank=16 * 1024 * 1024),
        llc_bytes=rank_curve,
        rank_cpu_ids=tuple(range(8)),
        llc_domains=(
            LlcDomainCalibration("0", (0, 1, 2, 3), 8 * 1024 * 1024, domain_curve),
            LlcDomainCalibration("1", (4, 5, 6, 7), 8 * 1024 * 1024, domain_curve),
        ),
    )

    assert calibration.service_rate("llc_bytes", 4, active_cpu_ids=(0, 1, 2, 3)) == pytest.approx(300.0)
    assert calibration.service_rate("llc_bytes", 4, active_cpu_ids=(0, 1, 4, 5)) == pytest.approx(400.0)
    assert calibration.service_rate("llc_bytes", 8, active_cpu_ids=range(8)) == pytest.approx(550.0)
    assert calibration.llc_capacity_bytes(active_cpu_ids=(0, 1)) == 8 * 1024 * 1024
    assert calibration.llc_capacity_bytes(active_cpu_ids=(0, 4)) == 16 * 1024 * 1024
    assert calibration.service_rate("dram_bytes", 4, active_cpu_ids=(0, 1, 4, 5)) == pytest.approx(
        calibration.service_rate("dram_bytes", 4)
    )
    model = _model(calibration)
    assert model._llc_miss_fraction(10 * 1024 * 1024, active_cpu_ids=(0, 1)) > 0.0
    assert model._llc_miss_fraction(10 * 1024 * 1024, active_cpu_ids=(0, 4)) == 0.0


def _placement_sensitive_model() -> AnalyticMoeCostModel:
    fast = _curve(1e15, 8e15, 8)
    base = replace(
        _calibration(),
        caches=replace(
            _calibration().caches,
            llc_bytes_per_rank=64 * 1024,
            llc_effective_fraction=0.5,
        ),
        matrix_flops=fast,
        gemm_core_flops=fast,
        frontend_instructions=fast,
        l1_bytes=fast,
        l2_bytes=fast,
        llc_bytes=_curve(1e8, 4e8, 8),
        dram_bytes=fast,
        epilogue_elements=fast,
        rank_cpu_ids=tuple(range(8)),
    )
    domain_curve = _curve(1e8, 2e8, 2)
    calibration = replace(
        base,
        llc_domains=(
            LlcDomainCalibration("0", (0, 1, 2, 3), 32 * 1024, domain_curve),
            LlcDomainCalibration("1", (4, 5, 6, 7), 32 * 1024, domain_curve),
        ),
    )
    return _model(calibration)


def test_placed_dag_models_llc_domains_and_symmetric_swap() -> None:
    model = _placement_sensitive_model()
    same_domain = model.dag_makespan_placed(
        [(48, 2, (0, 1), ()), (48, 2, (2, 3), ())]
    )
    split_domains = model.dag_makespan_placed(
        [(48, 2, (0, 1), ()), (48, 2, (4, 5), ())]
    )
    swapped_domains = model.dag_makespan_placed(
        [(48, 2, (4, 5), ()), (48, 2, (0, 1), ())]
    )

    assert split_domains < same_domain
    assert swapped_domains == pytest.approx(split_domains)


def test_placed_dag_rejects_unordered_overlapping_cpu_teams() -> None:
    model = _placement_sensitive_model()

    with pytest.raises(ValueError, match="overlapping placed tasks"):
        model.dag_makespan_placed(
            [(48, 2, (0, 1), ()), (48, 2, (1, 2), ())]
        )


def test_interval_planner_scores_analytic_tasks_with_physical_placement() -> None:
    model = _placement_sensitive_model()
    planner = IntervalPlanner(
        model,
        num_cores=8,
        cpu_ids=tuple(range(8)),
        shapes=((4, 4),),
        native_cold_planner=False,
    )
    tasks = [
        (0, 48, 0, 2, []),
        (1, 48, 4, 2, []),
    ]

    expected = model.dag_makespan_placed(
        [(48, 2, (0, 1), ()), (48, 2, (4, 5), ())]
    )
    assert planner._score(tasks) == pytest.approx(expected)


def test_placed_dag_without_topology_preserves_rank_aggregate_fallback() -> None:
    model = _model()
    tasks = [(48, 2, (0, 1), ()), (48, 2, (2, 3), ())]

    assert model.dag_makespan_placed(tasks) == pytest.approx(
        model.dag_makespan([(48, 2, ()), (48, 2, ())])
    )


def test_calibration_builder_attaches_explicit_and_symmetric_llc_domains() -> None:
    probe = _service_probe()
    probe["machine"]["cpu_ids"] = list(range(8))
    probe["topology"] = {
        "rank_cpu_ids": list(range(8)),
        "dram_scope": "numa_rank",
        "llc_domains": [
            {"id": "0", "cpu_ids": [0, 1, 2, 3], "capacity_bytes": 4 * 1024 * 1024},
            {"id": "1", "cpu_ids": [4, 5, 6, 7], "capacity_bytes": 4 * 1024 * 1024},
        ],
    }
    domain_probe = _service_probe()
    domain_probe["machine"].update({"cpu_ids": [0, 1, 2, 3], "cores_per_rank": 4})
    for service in domain_probe["services"].values():
        if "rows" in service:
            service["rows"] = service["rows"][:3]

    payload, report = build_calibration(
        probe,
        machine_id="topology-v2",
        l2_effective_fraction=0.75,
        llc_effective_fraction=0.625,
        l2_b_reuse_miss_floor=0.0,
        l2_b_reuse_miss_at_capacity=1.0,
        l2_b_reuse_miss_ceiling=1.0,
        relative_uncertainty=0.15,
        llc_domain_probes={"0": domain_probe},
        supported_widths=(1, 2, 4),
    )
    calibration = AnalyticMachineCalibration.from_dict(payload)

    assert payload["planner"]["supported_widths"] == [1, 2, 4]
    assert [domain["service"] for domain in payload["topology"]["llc_domains"]][0] == (
        payload["topology"]["llc_domains"][1]["service"]
    )
    assert report["llc_domains"]["0"]["source"] == "explicit_domain_probe:0"
    assert report["llc_domains"]["1"]["source"] == "symmetric_clone:0"
    assert report["llc_capacity"]["corrected_from_legacy_single_domain_value"] is False
    assert report["llc_topology_composition"]["rank_to_summed_domain_ratio"] == pytest.approx(70.0 / 136.0)
    assert calibration.service_rate("llc_bytes", 8, active_cpu_ids=range(8)) == pytest.approx(70.0)


def test_schema_v2_probe_requires_planner_widths_separate_from_service_points() -> None:
    probe = _service_probe()
    probe["schema_version"] = 2

    with pytest.raises(ValueError, match="explicit supported_widths"):
        build_calibration(
            probe,
            machine_id="missing-width-domain",
            l2_effective_fraction=0.75,
            llc_effective_fraction=0.625,
            l2_b_reuse_miss_floor=0.0,
            l2_b_reuse_miss_at_capacity=1.0,
            l2_b_reuse_miss_ceiling=1.0,
            relative_uncertainty=0.15,
        )


def test_schema_v1_machine_calibration_remains_readable() -> None:
    payload = _calibration().to_dict()
    payload["schema_version"] = 1
    payload.pop("topology", None)

    restored = AnalyticMachineCalibration.from_dict(payload)

    assert restored.machine_id == _calibration().machine_id
    assert restored.llc_domains == ()
    with pytest.raises(ValueError, match="calibrated LLC domains"):
        restored.service_rate("dram_bytes", 2, active_cpu_ids=(0, 1))


def test_thin_calibration_marks_legacy_w13_restart_fallback() -> None:
    probe = _service_probe()
    del probe["services"]["w13_fused_panel_range_restart"]

    calibration, _ = build_calibration(
        probe,
        machine_id="test-legacy-restart",
        l2_effective_fraction=0.75,
        llc_effective_fraction=0.625,
        l2_b_reuse_miss_floor=0.18,
        l2_b_reuse_miss_at_capacity=0.62,
        l2_b_reuse_miss_ceiling=0.87,
        relative_uncertainty=0.15,
    )

    assert calibration["overheads"]["w13_panel_range_restart_ns"] == pytest.approx(7.5)
    assert calibration["provenance"]["panel_range_restart_service"]["w13"] == (
        "fallback_to_m12_l1_hot_full_no_store_extra_n_range"
    )


def test_thin_residual_fit_recovers_nonnegative_operator_terms() -> None:
    expected = [250.0, 3.0, 1.2]
    features = [
        [1.0, 12.0, 1_000.0],
        [1.0, 192.0, 5_000.0],
        [1.0, 2040.0, 40_000.0],
        [1.0, 768.0, 16_000.0],
    ]
    targets = [sum(coefficient * value for coefficient, value in zip(expected, row)) for row in features]

    actual, sse = fit_nonnegative_residuals(features, targets)

    assert actual == pytest.approx(expected)
    assert sse == pytest.approx(0.0, abs=1e-12)


def test_machine_calibration_json_round_trip() -> None:
    calibration = replace(
        _calibration(),
        wide_team_pressure=WideTeamPressureCalibration(
            isolated_dilation=((4, 1.05), (8, 1.2)),
            full_cohort_dilation=((4, 1.1), (8, 1.4)),
        ),
    )

    restored = AnalyticMachineCalibration.from_dict(calibration.to_dict())

    assert restored == calibration


def test_wide_team_pressure_calibration_is_discrete_and_rejects_speedup() -> None:
    pressure = WideTeamPressureCalibration(
        isolated_dilation=((4, 1.0), (8, 1.2)),
        full_cohort_dilation=((4, 1.1), (8, 1.4)),
    )

    assert pressure.isolated_scale(8) == pytest.approx(1.2)
    assert pressure.full_cohort_scale(4) == pytest.approx(1.1)
    assert pressure.full_cohort_scale(6) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="dilation must be at least one"):
        WideTeamPressureCalibration(isolated_dilation=((4, 0.9),))
    with pytest.raises(ValueError, match="cannot be below isolated"):
        WideTeamPressureCalibration(
            isolated_dilation=((4, 1.2),),
            full_cohort_dilation=((4, 1.1),),
        )


def test_placed_phase_interpolates_internal_and_concurrent_wide_team_pressure() -> None:
    base = _placement_sensitive_model()
    calibration = replace(
        base.calibration,
        wide_team_pressure=WideTeamPressureCalibration(
            isolated_dilation=((4, 1.25),),
            full_cohort_dilation=((4, 2.0),),
        ),
    )
    model = _model(calibration)
    phase = next(item for item in model.predict_expert(48, 4).phases if item.kind == "cold_b")

    single = model._active_phase_state_placed({0: phase}, ((0, 1, 2, 3),))
    full_cohort = model._active_phase_state_placed(
        {0: phase, 1: phase},
        ((0, 1, 2, 3), (4, 5, 6, 7)),
    )

    assert single[-1] == {0: pytest.approx(1.25)}
    assert full_cohort[-1] == {0: pytest.approx(2.0), 1: pytest.approx(2.0)}
    assert full_cohort[3][0] > single[3][0]


def test_model_defaults_to_calibrated_runtime_n_tile() -> None:
    calibration = replace(_calibration(), backend_n_tile=16)
    model = AnalyticMoeCostModel(
        calibration,
        hidden_size=64,
        intermediate_size=32,
        global_experts=8,
        local_experts=8,
    )

    assert model.policy.backend_n_tile == 16
    assert model.w13_tile_bytes == 64 * 16 * 2


def test_machine_calibration_rejects_non_sve_n_tile() -> None:
    with pytest.raises(ValueError, match="multiple of eight"):
        replace(_calibration(), backend_n_tile=12)


def test_hot_gemm_core_service_subsumes_frontend_and_l1_resources() -> None:
    model = _model()
    phase = next(item for item in model.predict_expert(12, 1).phases if item.kind == "cold_b")

    assert phase.gemm_core_ns > 0.0
    assert phase.resource_demand("gemm_core_flops", phase.isolated_spill_fraction) == phase.matrix_flops
    assert phase.resource_demand("matrix_flops", phase.isolated_spill_fraction) == 0.0
    assert phase.resource_demand("frontend_instructions", phase.isolated_spill_fraction) == 0.0
    assert phase.resource_demand("l1_bytes", phase.isolated_spill_fraction) == 0.0
    pressures = model.active_resource_pressure((phase,))
    assert pressures["gemm_core_flops"].offered_rate > 0.0
    assert pressures["matrix_flops"].offered_rate == 0.0
    assert pressures["frontend_instructions"].offered_rate == 0.0
    assert pressures["l1_bytes"].offered_rate == 0.0


def test_phase_resource_vectors_match_scalar_resource_accessors() -> None:
    model = _model()
    phase = next(item for item in model.predict_expert(12, 1).phases if item.kind == "cold_b")
    spill_fraction = 0.375

    demands, times = phase.resource_vectors(spill_fraction)

    assert demands == tuple(
        phase.resource_demand(resource, spill_fraction)
        for resource in _SHARED_RESOURCES
    )
    assert times == tuple(
        phase.resource_times_ns(spill_fraction)[resource]
        for resource in _SHARED_RESOURCES
    )
    assert phase._duration_from_resource_times(times) == phase.duration_ns(
        spill_fraction=spill_fraction
    )


def test_machine_without_l1_hot_gemm_peak_is_rejected() -> None:
    payload = _calibration().to_dict()
    del payload["services"]["gemm_core_flops"]

    with pytest.raises(ValueError, match="L1-hot gemm_core_flops"):
        AnalyticMachineCalibration.from_dict(payload)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("activation", "gelu"),
        ("dtype", "fp16"),
    ],
)
def test_model_rejects_unmapped_kernel_semantics(option: str, value: str) -> None:
    arguments = {
        "calibration": _calibration(),
        "hidden_size": 64,
        "intermediate_size": 32,
        "global_experts": 8,
        "local_experts": 8,
        option: value,
    }

    with pytest.raises(ValueError, match="supports only"):
        AnalyticMoeCostModel(**arguments)


def test_model_rejects_dimensions_not_representable_by_sve_mapper() -> None:
    with pytest.raises(ValueError, match="packed K"):
        AnalyticMoeCostModel(
            _calibration(),
            hidden_size=64,
            intermediate_size=30,
            global_experts=8,
            local_experts=8,
        )


def test_full_stage_geometry_uses_team_width_as_the_only_window_control() -> None:
    geometry = full_stage_geometry(k=64, n=80, n_tile=8)

    assert geometry.total_tiles == 10
    assert geometry.stage_bytes == 64 * 80 * 2
    assert geometry.tiles_per_worker(1) == 10
    assert geometry.tiles_per_worker(2) == 5
    assert geometry.tiles_per_worker(3) == 4
    assert geometry.bytes_per_worker(3) == 4 * 64 * 8 * 2
    assert geometry.active_threads(16) == 10


def test_exact_m1_charges_physical_m2_compute() -> None:
    prediction = _model().predict_expert(routes=1, threads=1)
    mapping = prediction.w13_demand.mapping

    assert mapping.compute_rows == 2
    assert mapping.store_rows == 1
    assert mapping.executed_flops == 2 * mapping.useful_flops


def test_cold_weight_is_compulsory_dram_but_packed_a_is_not() -> None:
    prediction = _model().predict_expert(routes=12, threads=2)
    w13 = prediction.w13_demand
    w2 = prediction.w2_demand

    assert w13.compulsory_dram_bytes == 2 * 64 * 32 * 2
    assert w2.compulsory_dram_bytes == 64 * 32 * 2
    assert w13.a_l2_refill_bytes > 0
    assert w13.spillable_dram_bytes >= w13.a_l2_refill_bytes


def test_single_panel_weight_does_not_consume_reusable_llc_budget() -> None:
    model = _model()
    single_panel = model.predict_expert(routes=12, threads=1).w13_demand
    two_panels = model.predict_expert(routes=24, threads=1).w13_demand

    assert single_panel.reusable_b_bytes == 0
    assert two_panels.reusable_b_bytes == two_panels.stage_bytes
    assert two_panels.llc_working_set_bytes > single_panel.llc_working_set_bytes


def test_packed_b_l2_retention_uses_calibrated_miss_anchors() -> None:
    calibration = _calibration()
    cache = replace(
        calibration.caches,
        l2_b_reuse_miss_floor=0.18,
        l2_b_reuse_miss_at_capacity=0.62,
        l2_b_reuse_miss_ceiling=0.87,
    )
    model = _model(replace(calibration, caches=cache))

    assert model._l2_b_reuse_miss_fraction(cache.effective_l2_bytes_per_core) == pytest.approx(0.18)
    assert model._l2_b_reuse_miss_fraction(cache.l2_bytes_per_core) == pytest.approx(0.62)
    assert model._l2_b_reuse_miss_fraction(2 * cache.l2_bytes_per_core) == pytest.approx(0.87)


def test_team_width_naturally_controls_full_stage_worker_window() -> None:
    calibration = _calibration()
    cache = replace(
        calibration.caches,
        l2_bytes_per_core=2 * 1024 * 1024,
        llc_bytes_per_rank=96 * 1024 * 1024,
        l2_effective_fraction=0.75,
        l2_b_reuse_effective_fraction=0.125,
        l2_b_reuse_miss_floor=0.18,
        l2_b_reuse_miss_at_capacity=0.623,
        l2_b_reuse_miss_ceiling=0.869,
    )
    model = AnalyticMoeCostModel(
        replace(calibration, caches=cache),
        hidden_size=4096,
        intermediate_size=512,
        global_experts=8,
        local_experts=8,
    )

    assert model.w13_stage_bytes == 8 * 1024 * 1024
    assert model.w2_stage_bytes == 4 * 1024 * 1024
    assert [model.stage_bytes_per_worker("w13", threads) for threads in (1, 2, 4, 8)] == [
        8 * 1024 * 1024,
        4 * 1024 * 1024,
        2 * 1024 * 1024,
        1 * 1024 * 1024,
    ]
    assert [model.stage_bytes_per_worker("w2", threads) for threads in (1, 2, 4, 8)] == [
        4 * 1024 * 1024,
        2 * 1024 * 1024,
        1 * 1024 * 1024,
        512 * 1024,
    ]
    assert model.predict_expert(routes=216, threads=4).w13_demand.owner_window_bytes == 2 * 1024 * 1024
    assert model.predict_expert(routes=2040, threads=4).w13_demand.owner_window_bytes == 2 * 1024 * 1024


def _tp4_analytic_model(*, supported_widths: tuple[int, ...] = (1, 2, 4, 8)) -> AnalyticMoeCostModel:
    profile = (
        COST_MODEL
        / "profiles"
        / "analytic_machine_amazon_c5_192c_numa0_sve_jit_hot_gemm_20260802.json"
    )
    return AnalyticMoeCostModel(
        profile,
        hidden_size=4096,
        intermediate_size=512,
        global_experts=256,
        local_experts=256,
        supported_widths=supported_widths,
    )


def test_stage_window_score_uses_exact_runtime_tile_geometry() -> None:
    model = _tp4_analytic_model(supported_widths=(4,))
    score = model.score_stage_window("w13", routes=120, threads=4, window_tiles=2)

    assert score.window_tiles == 2
    assert score.range_tiles == 8
    assert score.windows == 16
    assert score.full_stripe_window_tiles == 32
    assert score.owner_window_bytes == 128 * 1024
    assert sum(window.n_tiles for window in score.window_demands) == 128
    assert all(window.owner_tiles == 2 for window in score.window_demands)
    assert score.compulsory_dram_bytes == model.w13_stage_bytes
    assert score.starves_any_thread is False


def test_stage_window_score_exposes_b_retention_vs_a_rescan_tradeoff() -> None:
    model = _tp4_analytic_model(supported_widths=(4,))
    narrow = model.score_stage_window("w13", routes=2040, threads=4, window_tiles=1)
    full = model.score_stage_window("w13", routes=2040, threads=4, window_tiles=0)

    assert narrow.b_l2_refill_bytes < full.b_l2_refill_bytes
    assert narrow.a_l2_refill_bytes > full.a_l2_refill_bytes
    assert narrow.windows == 32
    assert full.windows == 1
    assert narrow.objective_ns > full.objective_ns


def test_stage_window_a_residency_uses_effective_private_l2_capacity() -> None:
    model = _tp4_analytic_model(supported_widths=(16,))
    resident = model.score_stage_window("w13", routes=120, threads=16, window_tiles=1)
    streaming = model.score_stage_window("w13", routes=216, threads=16, window_tiles=1)
    cache = model.calibration.caches

    expected_capacity = cache.effective_l2_bytes_per_core - 12 * 4096 * 2
    assert resident.window_demands[0].a_residency_capacity_bytes == expected_capacity
    assert resident.l2_miss_fraction_a == 0.0
    assert 0.0 < streaming.l2_miss_fraction_a < 1.0
    assert streaming.a_l2_refill_bytes > resident.a_l2_refill_bytes


def test_stage_window_uses_stage_specific_range_restart_services() -> None:
    model = _tp4_analytic_model(supported_widths=(4,))
    w13 = model.score_stage_window("w13", routes=216, threads=4, window_tiles=2)
    w2 = model.score_stage_window("w2", routes=216, threads=4, window_tiles=8)

    assert w13.panel_range_restart_ns == pytest.approx(45.2695652173913)
    assert w2.panel_range_restart_ns == pytest.approx(6.695652173913044)
    assert w13.range_restart_ns == pytest.approx(18 * 15 * w13.panel_range_restart_ns)
    assert w2.range_restart_ns == pytest.approx(18 * 15 * w2.panel_range_restart_ns)


def test_single_panel_analytical_window_policy_selects_full_stripe() -> None:
    model = _tp4_analytic_model()
    policy = model.shadow_stage_window_policy()

    assert policy.select(routes=12, threads=4) == (0, 0)
    assert policy.explain(routes=12, threads=4)["reason"] == "single_panel_has_no_packed_b_reuse"


def test_analytical_window_policy_uses_formula_candidates_and_explains_deltas() -> None:
    model = _tp4_analytic_model(supported_widths=(4, 16))
    policy = model.shadow_stage_window_policy()

    assert policy.candidate_window_tiles("w13", routes=216, threads=4) == (1, 2, 4, 8, 16, 32)
    assert policy.select(routes=216, threads=4) == (8, 16)
    assert policy.select(routes=216, threads=16) == (0, 16)
    assert policy.select(routes=320, threads=4) == (8, 16)

    explanation = policy.explain(routes=216, threads=4)
    w13 = explanation["candidates"]["w13"]
    w2 = explanation["candidates"]["w2"]
    assert w13["selected"]["window_tiles"] == 8
    assert w2["selected"]["owner_window_bytes"] == 128 * 1024
    assert w13["selected"]["shared_b_llc_spill_ns"] < w13["full_stripe"]["shared_b_llc_spill_ns"]
    assert w13["selected_vs_full"]["b_l2_refill_bytes"] < 0
    assert "window_demands" not in w13["selected"]


def test_analytical_window_policy_adds_only_cohort_induced_reusable_b_spill() -> None:
    model = _tp4_analytic_model(supported_widths=(4,))
    policy = model.shadow_stage_window_policy()
    evaluations = policy.stage_evaluations("w13", routes=216, threads=4)
    by_tiles = {item.score.window_tiles: item for item in evaluations}

    assert by_tiles[2].shared_b_llc_spill_ns == 0.0
    assert by_tiles[32].shared_b_llc_spill_ns > 0.0
    assert by_tiles[32].policy_objective_ns == pytest.approx(
        by_tiles[32].score.objective_ns + by_tiles[32].shared_b_llc_spill_ns
    )


def test_shadow_window_scoring_does_not_change_default_expert_prediction() -> None:
    model = _tp4_analytic_model(supported_widths=(4,))
    before = model.predict_expert(routes=120, threads=4)
    _ = model.shadow_stage_window_policy().explain(routes=120, threads=4)
    after = model.predict_expert(routes=120, threads=4)

    assert after is before
    assert after.w13_demand.owner_window_bytes == 2 * 1024 * 1024
    assert after.w13_demand.stripe_demand is not None


def test_single_panel_stage_is_entirely_cold_b() -> None:
    prediction = _model().predict_expert(routes=12, threads=2)

    for stage, demand in (("w13", prediction.w13_demand), ("w2", prediction.w2_demand)):
        phases = [phase for phase in prediction.phases if phase.name.startswith(stage)]
        gemm_phases = [phase for phase in phases if phase.kind in {"cold_b", "steady_b"}]

        assert gemm_phases
        assert all(phase.kind == "cold_b" for phase in gemm_phases)
        assert sum(phase.compulsory_dram_bytes for phase in gemm_phases) == demand.compulsory_dram_bytes
        assert sum(phase.panel_count for phase in gemm_phases) == len(demand.mapping.panels)


def test_long_stage_splits_cold_and_steady_demand_without_changing_work() -> None:
    prediction = _model().predict_expert(routes=24, threads=2)

    for stage, demand in (("w13", prediction.w13_demand), ("w2", prediction.w2_demand)):
        phases = [
            phase
            for phase in prediction.phases
            if phase.name.startswith(stage) and phase.kind in {"cold_b", "steady_b"}
        ]
        cold = [phase for phase in phases if phase.kind == "cold_b"]
        steady = [phase for phase in phases if phase.kind == "steady_b"]

        assert len(cold) == 1
        assert len(steady) == 1
        assert all(phase.compulsory_dram_bytes > 0 for phase in cold)
        assert all(phase.compulsory_dram_bytes == 0 for phase in steady)
        assert sum(phase.matrix_flops for phase in phases) == demand.mapping.demand.balanced_executed_flops
        assert sum(phase.l2_bytes for phase in phases) == pytest.approx(demand.l2_bytes)
        assert sum(phase.llc_bytes for phase in phases) == pytest.approx(demand.llc_bytes)
        assert sum(phase.compulsory_dram_bytes for phase in phases) == demand.compulsory_dram_bytes
        assert sum(phase.spillable_dram_bytes for phase in phases) == pytest.approx(demand.spillable_dram_bytes)


@pytest.mark.parametrize("routes", [1, 2, 8, 12, 13, 25, 192, 2040])
@pytest.mark.parametrize("threads", [1, 2, 4, 8])
def test_phase_lowering_conserves_kernel_demand_across_routes_and_widths(routes: int, threads: int) -> None:
    prediction = _model().predict_expert(routes=routes, threads=threads)

    for stage, demand in (("w13", prediction.w13_demand), ("w2", prediction.w2_demand)):
        phases = [
            phase
            for phase in prediction.phases
            if phase.name.startswith(stage) and phase.kind in {"cold_b", "steady_b"}
        ]

        assert sum(phase.panel_count for phase in phases) == len(demand.mapping.panels)
        assert sum(phase.matrix_flops for phase in phases) == demand.mapping.demand.balanced_executed_flops
        assert (
            sum(phase.frontend_instructions for phase in phases) == demand.mapping.demand.balanced_key_body_instructions
        )
        assert sum(phase.l1_bytes for phase in phases) == demand.mapping.demand.balanced_l1_load_bytes
        assert sum(phase.epilogue_elements for phase in phases) == demand.mapping.demand.balanced_epilogue_elements
        assert sum(phase.l2_bytes for phase in phases) == pytest.approx(demand.l2_bytes)
        assert sum(phase.llc_bytes for phase in phases) == pytest.approx(demand.llc_bytes)
        assert sum(phase.compulsory_dram_bytes for phase in phases) == pytest.approx(demand.compulsory_dram_bytes)
        assert sum(phase.spillable_dram_bytes for phase in phases) == pytest.approx(demand.spillable_dram_bytes)


def test_shared_resource_capacity_derates_parallel_experts() -> None:
    calibration = replace(
        _calibration(dram_saturated_rate=10e9),
        matrix_flops=_curve(1e9, 1e9, 8),
        gemm_core_flops=_curve(1e9, 1e9, 8),
    )
    model = _model(calibration)
    isolated = model.T_iso(12, 1)

    parallel = model.phase_makespan([(12, 1), (12, 1)])
    sequential = model.dag_makespan([(12, 1, []), (12, 1, [0])])
    finish_times = model.dag_task_finish_times([(12, 1, []), (12, 1, [0])])

    assert parallel > isolated + calibration.overheads.call_setup_ns
    assert parallel < sequential
    assert sequential == pytest.approx(2 * isolated + calibration.overheads.call_setup_ns)
    assert finish_times[0] < finish_times[1]
    assert max(finish_times) == pytest.approx(sequential)


def test_resource_pressure_uses_requesting_threads_and_named_capacity() -> None:
    model = _model()
    prediction = model.predict_expert(routes=24, threads=2)
    setup = next(phase for phase in prediction.phases if phase.kind == "stage_setup")
    cold = next(phase for phase in prediction.phases if phase.kind == "cold_b")

    pressures = model.active_resource_pressure((setup, cold))
    dram = pressures["dram_bytes"]

    assert dram.active_threads == cold.active_threads
    assert dram.capacity == model.calibration.dram_bytes.rate(cold.active_threads)
    assert dram.offered_rate > 0.0
    assert dram.utilization == pytest.approx(dram.offered_rate / dram.capacity)
    assert dram.dilation == pytest.approx(max(1.0, dram.utilization))
    assert dram.allocated_rate <= dram.capacity * (1.0 + 1e-12)
    assert dram.allocated_utilization <= 1.0 + 1e-12

    explained_cold = next(phase for phase in model.explain(routes=24, threads=2)["phases"] if phase["kind"] == "cold_b")
    assert explained_cold["resource_pressure"]["llc_bytes"]["path"] == "shared_llc_to_private_l2_refill"
    assert explained_cold["resource_pressure"]["dram_bytes"]["offered_rate"] > 0.0


def test_dag_explanation_exposes_phase_local_resource_contention() -> None:
    fast = _curve(1e15, 8e15, 8)
    calibration = replace(
        _calibration(),
        gemm_core_flops=fast,
        matrix_flops=fast,
        frontend_instructions=fast,
        l1_bytes=fast,
        l2_bytes=fast,
        llc_bytes=fast,
        dram_bytes=_curve(1e9, 1e9, 8),
        epilogue_elements=fast,
    )
    model = _model(calibration)
    tasks = [(12, 1, []), (12, 1, [])]

    explanation = model.explain_dag(tasks)
    contended = [
        event for event in explanation["events"] if event["resources"].get("dram_bytes", {}).get("dilation", 1.0) > 1.0
    ]

    assert explanation["makespan_ns"] == pytest.approx(model.dag_makespan(tasks))
    assert max(explanation["task_finish_ns"]) == pytest.approx(explanation["makespan_ns"])
    assert contended
    assert any("cold_b" in event["phase_kinds"].values() for event in contended)
    assert all(event["resources"]["dram_bytes"]["path"] == "dram_to_llc_compulsory_and_spill" for event in contended)
    assert all(
        pressure["allocated_rate"] <= pressure["capacity"] * (1.0 + 1e-12)
        for event in explanation["events"]
        for pressure in event["resources"].values()
        if pressure["capacity"] is not None
    )


def test_isolated_llc_spill_matches_single_task_dag() -> None:
    calibration = replace(
        _calibration(),
        gemm_core_flops=_curve(1e15, 8e15, 8),
    )
    tiny_llc = replace(
        calibration,
        caches=replace(
            calibration.caches,
            llc_bytes_per_rank=4 * 1024,
            llc_effective_fraction=0.5,
        ),
    )
    model = _model(tiny_llc)
    no_spill_model = _model(calibration)

    isolated = model.T_iso(24, 1)
    single_task = model.dag_makespan([(24, 1, [])])

    assert isolated > no_spill_model.T_iso(24, 1)
    assert single_task == pytest.approx(isolated + calibration.overheads.call_setup_ns)


def test_analytic_shape_space_uses_at_most_two_widths() -> None:
    shapes = analytic_candidate_shapes(12, (1, 2, 3, 4, 6, 12))

    assert (3, 3, 3, 3) in shapes
    assert (6, 3, 3) in shapes
    assert all(len(set(shape)) <= 2 for shape in shapes)


def test_planner_and_runtime_accept_analytic_model_without_profile_shapes() -> None:
    calibration = _calibration(
        cores=6,
        supported_widths=(1, 2, 3, 6),
        dram_saturated_rate=40e9,
    )
    model = _model(calibration)
    experts = [(expert, routes) for expert, routes in enumerate((48, 24, 12, 6))]

    planner = IntervalPlanner(model, num_cores=6)
    result = planner.plan(experts)
    runtime = PlannedMoE(model, num_cores=6)
    spec = runtime.plan_spec_for(experts)
    cached = runtime.plan_spec_for(experts)

    assert (3, 3) in planner.shapes
    assert sum(result["shape"]) == 6
    assert spec["plan_version"] == 2
    assert spec["bridge"]["num_threads"] == 6
    assert cached["bridge"] == spec["bridge"]
    assert runtime.last["cache_hit"] is True


def test_analytic_full_searches_all_shapes_with_a_quick_baseline() -> None:
    calibration = _calibration(
        cores=6,
        supported_widths=(1, 2, 3, 6),
        dram_saturated_rate=40e9,
    )
    model = _model(calibration)
    experts = [(expert, routes) for expert, routes in enumerate((48, 24, 12, 6))]
    planner = IntervalPlanner(model, num_cores=6, native_cold_planner=False)

    quick = planner.plan_quick(experts)
    full = planner.plan(experts, dynamic_tail_pool=False)

    assert full["planner_backend"] == "python_analytic_full"
    assert full["strict_candidates"] == len(planner.shapes) + 1
    assert full["dynamic_candidates"] == 0
    assert len(full["ranking"]) == len(planner.shapes) + 1
    assert any(len(set(candidate["shape"])) > 1 for candidate in full["ranking"])
    quick_ranking = [candidate for candidate in full["ranking"] if candidate["shape"] == quick["shape"]]
    assert quick_ranking
    rounded_quick_ns = min(candidate["makespan_ms"] * 1e6 for candidate in quick_ranking)
    assert full["makespan_ns"] <= rounded_quick_ns + 500.0

    explicit = IntervalPlanner(
        model,
        num_cores=6,
        shapes=((6,), (3, 3)),
        native_cold_planner=False,
    ).plan(experts, dynamic_tail_pool=False)

    assert explicit["planner_backend"] == "python"
    assert explicit["strict_candidates"] == 2


def test_analytic_full_optimizes_expected_makespan_not_working_set() -> None:
    fastest = {
        "makespan_ns": 80.0,
        "uncertainty_ns": 1.0,
        "pessimistic_ns": 100.0,
        "active_working_set_bytes": 2,
        "resource_groups": 2,
        "shape": (2, 2),
    }
    smaller_working_set = {
        "makespan_ns": 81.0,
        "uncertainty_ns": 1.0,
        "pessimistic_ns": 90.0,
        "active_working_set_bytes": 1,
        "resource_groups": 1,
        "shape": (4,),
    }

    assert IntervalPlanner._select_analytic_full([smaller_working_set, fastest]) is fastest


def test_analytic_full_steps_down_one_width_inside_systematic_uncertainty() -> None:
    fastest = {
        "makespan_ns": 100.0,
        "uncertainty_ns": 15.0,
        "pessimistic_ns": 115.0,
        "active_working_set_bytes": 10,
        "resource_groups": 4,
        "shape": (32, 32, 8, 8),
    }
    one_step_narrower = {
        "makespan_ns": 104.0,
        "uncertainty_ns": 15.6,
        "pessimistic_ns": 119.6,
        "active_working_set_bytes": 20,
        "resource_groups": 6,
        "shape": (16, 16, 16, 16, 8, 8),
    }
    two_steps_narrower = {
        "makespan_ns": 107.0,
        "uncertainty_ns": 16.05,
        "pessimistic_ns": 123.05,
        "active_working_set_bytes": 30,
        "resource_groups": 10,
        "shape": (8,) * 10,
    }

    selected = IntervalPlanner._select_analytic_full(
        [two_steps_narrower, one_step_narrower, fastest]
    )

    assert selected is one_step_narrower


def test_analytic_uncertainty_is_systematic_across_waves() -> None:
    calibration = _calibration(
        cores=4,
        supported_widths=(1, 2, 4),
        dram_saturated_rate=40e9,
    )
    model = _model(calibration)
    planner = IntervalPlanner(model, num_cores=4, native_cold_planner=False)
    experts = [(expert, 12) for expert in range(16)]
    makespan = 1_000.0

    uncertainty = planner._uncertainty(
        experts,
        (1, 1, 1, 1),
        makespan,
        use_full_workload_anchor=False,
    )

    assert uncertainty == pytest.approx(makespan * model.relative_error)


def test_holdout_validator_reports_absolute_error_and_shape_regret() -> None:
    calibration = _calibration()
    model = _model(calibration)
    isolated = [
        {
            "routes": routes,
            "threads": threads,
            "median_ns": model.T_iso(routes, threads),
        }
        for routes, threads in ((12, 1), (24, 2))
    ]
    tasks = [(12, 2, []) for _ in range(4)]
    profile = {
        "schema_version": 2,
        "measurement": {"profile_id": "synthetic-holdout"},
        "target": {"cores_per_rank": 8, "concurrent_ranks": 1},
        "expert_shape": {
            "dtype": "bf16",
            "hidden_size": 64,
            "intermediate_size": 32,
            "activation": "silu",
        },
        "kernel": {
            "backend_n_tile": 8,
            "m_tail_policy": "xbyak_exact_m",
            "stage_geometry": "full_n_team_stripes",
        },
        "parallelism": {
            "mode": "standalone",
            "degree": 1,
            "global_experts": 8,
            "local_experts": 8,
        },
        "isolated": isolated,
        "entries": [
            {
                "routes": 12,
                "shape": [2, 2, 2, 2],
                "lane_task_counts": [1, 1, 1, 1],
                "full_call_median_ns": model.dag_makespan(tasks),
            }
        ],
    }

    report = build_validation_report(calibration, profile, isolated_training_points={(12, 1)})

    assert report["analytic_model_schema_version"] == 8
    assert report["analytic_model"] == "phase_ecm_llc_domain_team_pressure_v5"
    assert report["isolated"]["coverage"] == {
        "profile_points": 2,
        "evaluated_points": 2,
        "skipped_points": 0,
    }
    assert report["isolated"]["summary"]["mape"] == pytest.approx(0.0)
    assert report["isolated"]["holdout_summary"]["points"] == 1
    assert report["isolated"]["training_points"] == [{"routes": 12, "threads": 1}]
    assert report["contention"]["summary"]["mape"] == pytest.approx(0.0)
    assert report["contention"]["ranking"]["max_regret"] == pytest.approx(0.0)


def test_holdout_validator_rejects_wrong_core_topology() -> None:
    profile = {
        "schema_version": 2,
        "target": {"cores_per_rank": 16},
    }

    with pytest.raises(ValueError, match="cores_per_rank mismatch"):
        build_validation_report(_calibration(), profile)


def test_runtime_uses_one_calibration_and_emits_no_stage_split_controls() -> None:
    calibration = _calibration()
    model = _model(calibration)
    with pytest.raises(ValueError, match="one calibration model"):
        PlannedMoE((model, _model(calibration)), num_cores=8)

    runtime = PlannedMoE(model, num_cores=8)
    experts = [(expert, 24) for expert in range(8)]

    first = runtime.plan_spec_for(experts)
    cached = runtime.plan_spec_for(experts)

    assert "operator_options" not in first
    assert "w13_window_ranges" not in first["policy"]
    assert "w2_window_ranges" not in first["policy"]
    assert "task_w13_ranges" not in first["bridge"]
    assert "task_w2_ranges" not in first["bridge"]
    assert first["bridge"] == cached["bridge"]
