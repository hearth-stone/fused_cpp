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
    AnalyticMachineCalibration,
    AnalyticMoeCostModel,
    CacheCalibration,
    RuntimeOverheads,
    SaturatingServiceCurve,
    analytic_candidate_shapes,
)
from interval_planner import IntervalPlanner  # noqa: E402
from planned_moe import PlannedMoE  # noqa: E402
from sve_bf16_kernel_model import allocate_n_tiles  # noqa: E402
from validate_analytic_model import build_validation_report  # noqa: E402


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
            range_fixed_ns=50.0,
        ),
        supported_widths=supported_widths,
        relative_uncertainty=0.04,
    )


def _model(
    calibration: AnalyticMachineCalibration | None = None,
    *,
    w13_split: bool = True,
    w13_split_chunks: int = 2,
) -> AnalyticMoeCostModel:
    return AnalyticMoeCostModel(
        calibration or _calibration(),
        hidden_size=64,
        intermediate_size=32,
        global_experts=8,
        local_experts=8,
        w13_split=w13_split,
        w13_split_chunks=w13_split_chunks,
    )


def test_service_curve_uses_two_hardware_anchors() -> None:
    curve = _curve(100.0, 400.0, 4)

    assert curve.rate(1) == pytest.approx(100.0)
    assert curve.rate(2) == pytest.approx(200.0)
    assert curve.rate(4) == pytest.approx(400.0)
    assert curve.rate(16) == pytest.approx(400.0)


def test_shared_bottleneck_curve_preserves_linear_low_thread_scaling() -> None:
    curve = _curve(40.0, 360.0, 86, curve="shared_bottleneck")

    assert curve.rate(1) == pytest.approx(40.0)
    assert curve.rate(24) > 280.0
    assert curve.rate(86) == pytest.approx(360.0)
    assert curve.rate(128) == pytest.approx(360.0)


def test_machine_calibration_json_round_trip() -> None:
    calibration = _calibration()

    restored = AnalyticMachineCalibration.from_dict(calibration.to_dict())

    assert restored == calibration


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


def test_uneven_weight_ranges_preserve_all_n_tiles() -> None:
    allocation = allocate_n_tiles(
        n_columns=80,
        n_tile=8,
        threads=2,
        n_ranges=3,
    )

    assert allocation.range_tiles == (4, 3, 3)
    assert sum(allocation.range_tiles) == allocation.total_tiles
    assert allocation.per_thread_tiles == (6, 4)


def test_uneven_weight_ranges_use_per_range_traffic() -> None:
    calibration = replace(_calibration(), supported_widths=(1, 2, 3, 4, 8))
    model = AnalyticMoeCostModel(
        calibration,
        hidden_size=64,
        intermediate_size=32,
        global_experts=8,
        local_experts=8,
        w13_split=False,
        w13_split_chunks=1,
        weight_window_bytes=3500,
    )

    prediction = model.predict_expert(routes=24, threads=3)
    demand = prediction.w13_demand
    phases = [phase for phase in prediction.phases if phase.name.startswith("w13")]

    assert [item.n_tiles for item in demand.range_demands] == [3, 3, 2]
    assert sum(item.balanced_work_fraction for item in demand.range_demands) == pytest.approx(8 / 9)
    assert sum(item.compulsory_dram_bytes for item in demand.range_demands) == 2 * 64 * 32 * 2
    assert len(phases) == 3
    assert [phase.active_threads for phase in phases] == [3, 3, 2]
    assert phases[-1].working_set_bytes < phases[0].working_set_bytes
    explained = model.explain(routes=24, threads=3)["w13"]
    assert explained["executed_flops"] == sum(phase.matrix_flops for phase in phases)
    assert explained["executed_flops"] < explained["mapping_balanced_executed_flops_upper_bound"]


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
    assert two_panels.reusable_b_bytes == two_panels.window_bytes
    assert two_panels.llc_working_set_bytes > single_panel.llc_working_set_bytes


def test_split_w13_keeps_gemm_work_but_adds_range_overhead() -> None:
    split = _model()
    unsplit = _model(w13_split=False, w13_split_chunks=1)

    split_prediction = split.predict_expert(routes=24, threads=2)
    unsplit_prediction = unsplit.predict_expert(routes=24, threads=2)

    assert split_prediction.w13_demand.ranges == 2
    assert unsplit_prediction.w13_demand.ranges == 1
    assert split_prediction.w13_demand.mapping.executed_flops == unsplit_prediction.w13_demand.mapping.executed_flops
    assert split_prediction.w13_ns > unsplit_prediction.w13_ns


def test_shared_resource_capacity_derates_parallel_experts() -> None:
    calibration = replace(
        _calibration(dram_saturated_rate=10e9),
        matrix_flops=_curve(1e9, 1e9, 8),
    )
    model = _model(calibration)
    isolated = model.T_iso(12, 1)

    parallel = model.phase_makespan([(12, 1), (12, 1)])
    sequential = model.dag_makespan([(12, 1, []), (12, 1, [0])])

    assert parallel > isolated + calibration.overheads.call_setup_ns
    assert parallel < sequential
    assert sequential == pytest.approx(2 * isolated + calibration.overheads.call_setup_ns)


def test_isolated_llc_spill_matches_single_task_dag() -> None:
    calibration = _calibration()
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

    assert (3, 3) in planner.shapes
    assert sum(result["shape"]) == 6
    assert spec["plan_version"] == 2
    assert spec["bridge"]["num_threads"] == 6


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
            "w13_split": True,
            "w13_split_chunks": 2,
            "weight_window_bytes": 0,
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

    report = build_validation_report(calibration, profile)

    assert report["isolated"]["coverage"] == {
        "profile_points": 2,
        "evaluated_points": 2,
        "skipped_points": 0,
    }
    assert report["isolated"]["summary"]["mape"] == pytest.approx(0.0)
    assert report["contention"]["summary"]["mape"] == pytest.approx(0.0)
    assert report["contention"]["ranking"]["max_regret"] == pytest.approx(0.0)


def test_holdout_validator_rejects_wrong_core_topology() -> None:
    profile = {
        "schema_version": 2,
        "target": {"cores_per_rank": 16},
    }

    with pytest.raises(ValueError, match="cores_per_rank mismatch"):
        build_validation_report(_calibration(), profile)


def test_policy_runtime_distinguishes_variants_sharing_one_machine_file() -> None:
    calibration = _calibration()
    split = _model(calibration)
    unsplit = _model(calibration, w13_split=False, w13_split_chunks=1)
    runtime = PlannedMoE((split, unsplit), num_cores=8)
    experts = [(expert, 24) for expert in range(8)]

    first = runtime.plan_spec_for(experts)
    cached = runtime.plan_spec_for(experts)

    assert first["operator_options"] == cached["operator_options"]
    assert first["operator_options"]["w13_split"] == first["policy"]["w13_split"]
