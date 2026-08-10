from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
COST_MODEL = ROOT / "cpu_moe_schedule_optimization" / "cost_model"
PLANNERS = ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path[:0] = [str(COST_MODEL), str(PLANNERS)]

from interval_planner import IntervalPlanner  # noqa: E402
from phase_model import ContentionCostModel  # noqa: E402


PROFILE = (
    COST_MODEL
    / "profiles"
    / "contention_async_amazon_ecs_8c_standalone_sve_F512_E8_fulln_xbyak_exactm_v2_20260727.json"
)
TP4_96C_PROFILE = (
    COST_MODEL
    / "profiles"
    / "contention_async_amazon_c5_192c_numa0_tp4_sve_F512_E256_fulln_schema_v2_xbyak_exactm_20260727.json"
)


def _native_extension():
    try:
        extension = importlib.import_module("fused_cpp._C")
    except ImportError:
        pytest.skip("fused_cpp._C is not built")
    if not hasattr(extension, "NativeIntervalPlanner"):
        pytest.skip("fused_cpp._C was built without NativeIntervalPlanner")
    return extension


def _assert_plan_equivalent(reference: dict, actual: dict) -> None:
    exact_fields = (
        "shape",
        "execution_mode",
        "tail_pool_threads",
        "tail_pool_max_routes",
        "tail_pool_tasks",
        "tail_repartition_width",
        "tail_repartition_tasks",
        "tail_repartition_route_slices",
        "active_working_set_bytes",
        "resource_groups",
        "window_bytes_per_worker",
        "tasks",
        "bridge",
        "strict_candidates",
        "dynamic_candidates",
        "tail_repartition_candidates",
        "ranking",
    )
    for field in exact_fields:
        assert actual[field] == reference[field], field
    assert actual["makespan_ns"] == pytest.approx(reference["makespan_ns"], rel=1e-13)
    assert actual["uncertainty_ns"] == pytest.approx(
        reference["uncertainty_ns"],
        rel=1e-13,
    )


@pytest.mark.parametrize("iso_mode", ["table", "formula"])
def test_native_cost_model_matches_python(iso_mode: str) -> None:
    extension = _native_extension()
    model = ContentionCostModel(PROFILE, iso_mode=iso_mode)
    planner = IntervalPlanner(model, num_cores=8, native_cold_planner=False)
    native = extension.NativeIntervalPlanner(
        8,
        list(planner.widths),
        [list(shape) for shape in planner.shapes],
        model.native_interval_planner_payload(),
        2,
    )

    for routes in (1, 2, 4, 8, 12, 13, 24, 48, 192, 768, 2040, 4080):
        for threads in planner.widths:
            assert native._estimate_isolated(routes, threads) == pytest.approx(
                model.T_iso(routes, threads),
                rel=1e-13,
            )

    dag = [
        (2040, 4, []),
        (12, 1, []),
        (768, 2, [0]),
        (4, 1, [1]),
    ]
    assert native._score_dag(
        [routes for routes, _, _ in dag],
        [threads for _, threads, _ in dag],
        [dependencies for _, _, dependencies in dag],
    ) == pytest.approx(model.dag_makespan(dag), rel=1e-13)


@pytest.mark.parametrize("iso_mode", ["table", "formula"])
def test_native_parallel_cold_search_matches_python(iso_mode: str) -> None:
    _native_extension()
    model = ContentionCostModel(PROFILE, iso_mode=iso_mode)
    reference = IntervalPlanner(model, num_cores=8, native_cold_planner=False)
    native_1t = IntervalPlanner(
        model,
        num_cores=8,
        native_cold_planner=True,
        planner_threads=1,
    )
    native_4t = IntervalPlanner(
        model,
        num_cores=8,
        native_cold_planner=True,
        planner_threads=4,
    )
    cases = (
        (
            [(expert, 2040) for expert in range(8)],
            {"dynamic_tail_pool": False},
        ),
        (
            [
                (0, 2040),
                (1, 768),
                (2, 192),
                (3, 48),
                (4, 12),
                (5, 8),
                (6, 4),
                (7, 1),
            ],
            {},
        ),
        (
            [(0, 2040), (1, 768), (2, 12), (3, 8), (4, 4), (5, 2), (6, 1)],
            {"forced_tail_pool_threads": 2},
        ),
    )

    for experts, options in cases:
        expected = reference.plan(experts, **options)
        for planner, workers in ((native_1t, 1), (native_4t, 4)):
            actual = planner.plan(experts, **options)
            _assert_plan_equivalent(expected, actual)
            assert actual["planner_backend"] == "cpp"
            assert actual["planner_workers"] == workers


def test_native_bounded_tail_repartition_matches_python() -> None:
    extension = _native_extension()
    model = ContentionCostModel(TP4_96C_PROFILE)
    reference = IntervalPlanner(
        model,
        num_cores=96,
        native_cold_planner=False,
    )
    native = IntervalPlanner(
        model,
        num_cores=96,
        native_cold_planner=True,
        planner_threads=4,
    )
    experts = [(expert, 1536) for expert in range(8)]

    assert model.T_iso(1536, 24) > 0
    with pytest.raises(KeyError, match="threads=24"):
        model.T_iso(24, 24)

    expected = reference.plan(
        experts,
        dynamic_tail_pool=False,
        bounded_tail_repartition=True,
    )
    actual = native.plan(
        experts,
        dynamic_tail_pool=False,
        bounded_tail_repartition=True,
    )

    _assert_plan_equivalent(expected, actual)
    assert actual["shape"] == (32, 32, 32)
    assert actual["tail_repartition_width"] is None
    assert actual["tail_repartition_tasks"] == 0
    assert actual["tail_repartition_route_slices"] == 1
    assert actual["tail_repartition_candidates"] == 0

    direct = extension.NativeIntervalPlanner(
        96,
        list(reference.widths),
        [list(shape) for shape in reference.shapes],
        model.native_interval_planner_payload(),
        4,
        list(reference.tail_repartition_widths),
    )
    expert_ids = [expert_id for expert_id, _ in experts]
    routes = [route_count for _, route_count in experts]
    auto = direct.plan(expert_ids, routes)
    strict = direct.plan(expert_ids, routes, dynamic_tail_pool=False)
    assert auto["tail_repartition_candidates"] == 0
    assert auto["selected"]["tail_repartition_width"] is None
    assert auto["selected"]["tail_repartition_route_slices"] == 1
    assert strict["tail_repartition_candidates"] == 0
    assert strict["selected"]["tail_repartition_width"] is None


def test_explicit_native_request_rejects_unsupported_model() -> None:
    class UnsupportedModel:
        schema_version = 1
        supported_widths = (1,)
        supported_shapes = ()

    with pytest.raises(RuntimeError, match="does not support native calibration export"):
        IntervalPlanner(
            UnsupportedModel(),
            num_cores=1,
            widths=(1,),
            shapes=((1,),),
            native_cold_planner=True,
        )


def test_native_parallel_search_propagates_worker_errors() -> None:
    extension = _native_extension()
    model = ContentionCostModel(PROFILE)
    planner = IntervalPlanner(model, num_cores=8, native_cold_planner=False)
    payload = model.native_interval_planner_payload()
    payload["p10_curves"] = []
    native = extension.NativeIntervalPlanner(
        8,
        list(planner.widths),
        [list(shape) for shape in planner.shapes],
        payload,
        4,
    )

    with pytest.raises(IndexError, match="shape was not measured"):
        native.plan(
            list(range(7)),
            [2040, 768, 12, 8, 4, 2, 1],
        )
