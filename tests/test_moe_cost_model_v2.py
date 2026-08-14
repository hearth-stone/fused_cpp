from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
COST_MODEL = ROOT / "cpu_moe_schedule_optimization" / "cost_model"
PLANNERS = ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path[:0] = [str(COST_MODEL), str(PLANNERS)]

from interval_planner import IntervalPlanner, PlannedTwoStagePlanner  # noqa: E402
from iso_formula import IsoFormula, fit_from_measurements  # noqa: E402
from phase_model import ContentionCostModel  # noqa: E402
from planned_moe import PlannedMoE, signature  # noqa: E402
from profile_catalog import (  # noqa: E402
    ProfileCatalog,
    ProfileCompatibilityError,
    ProfileQuery,
)
from simulate_schedules import PRESETS  # noqa: E402
from full_stage_geometry import full_stage_geometry  # noqa: E402
from tp_vs_ep_model import HierarchicalTopology, ParallelLayerEvaluator  # noqa: E402
from workload_catalog import load_routing_workload  # noqa: E402


PROFILE_DIR = COST_MODEL / "profiles"
XBYAK_AARCH64_COMMIT = "3f8c682b9c6ff562dc008c7d1b8307a683f05d20"


def profile_paths() -> list[Path]:
    return sorted(PROFILE_DIR.glob("*fulln*_v2_r1_20260713.json"))


def xbyak_profile_paths() -> list[Path]:
    return sorted(PROFILE_DIR.glob("*fulln*xbyak_exactm*.json"))


@pytest.fixture(scope="module")
def catalog() -> ProfileCatalog:
    return ProfileCatalog.from_paths(profile_paths())


def model_for(catalog: ProfileCatalog, mode: str, ffn: int, local_experts: int):
    query = ProfileQuery(
        mode=mode,
        degree=2,
        hidden_size=4096,
        intermediate_size=ffn,
        global_experts=64,
        local_experts=local_experts,
        backend="sve",
        backend_n_tile=8,
        activation="silu",
        dtype="bf16",
        measurement_experts=local_experts,
        cores_per_rank=32,
        concurrent_ranks=2,
    )
    record = catalog.select(query)
    return ContentionCostModel(record.path, expected_policy=query)


class _DeterministicTailPoolModel:
    schema_version = 1
    supported_shapes: tuple[tuple[int, ...], ...] = ()
    supported_widths = (1, 2, 4)
    profile_path = Path("deterministic-tail-pool.json")
    policy = None
    max_stage_bytes = 1
    has_full_workload_anchors = False
    local_experts = 0
    profile_runs = 1

    def T_iso(self, routes: int, threads: int) -> float:
        if routes >= 100:
            return {1: 400.0, 2: 220.0, 4: 100.0}[threads]
        return {1: 10.0, 2: 8.0, 4: 7.0}[threads]

    def dag_makespan(self, tasks) -> float:
        finish: list[float] = []
        for routes, threads, dependencies in tasks:
            start = max((finish[dependency] for dependency in dependencies), default=0.0)
            finish.append(start + self.T_iso(routes, threads))
        return max(finish, default=0.0)

    def supports_shape(self, shape) -> bool:
        return tuple(shape) == (4,)

    def relative_uncertainty(self, routes: int, shape) -> float:
        del routes, shape
        return 0.0

    def relative_full_call_uncertainty(self, routes: int, shape) -> float:
        del routes, shape
        return 0.0

    def profiled_full_call_time(self, routes: int, shape) -> float:
        del routes, shape
        raise AssertionError("schema-v1 test model has no full-call anchors")

    def window_bytes_per_worker(self, threads: int, routes: int | None = None) -> int:
        del routes
        return threads


class _QuickPlannerModel(_DeterministicTailPoolModel):
    def dag_makespan(self, tasks) -> float:
        del tasks
        raise AssertionError("quick planner must not run the event-time simulator")


class _DeterministicStageModel(_DeterministicTailPoolModel):
    call_setup_ns = 3.0

    def stage_T_iso(self, stage: str, routes: int, threads: int) -> float:
        del routes
        if stage == "w13":
            return {1: 200.0, 2: 55.0, 4: 30.0}[threads]
        if stage == "w2":
            return {1: 20.0, 2: 30.0, 4: 50.0}[threads]
        raise ValueError(stage)

    def stage_dag_makespan(self, stage: str, tasks) -> float:
        finish: list[float] = []
        for routes, threads, dependencies in tasks:
            start = max((finish[dependency] for dependency in dependencies), default=0.0)
            finish.append(start + self.stage_T_iso(stage, routes, threads))
        return max(finish, default=0.0)

    def task_stage_bytes(self, stage: str, routes: int, threads: int) -> int:
        del routes
        return threads * (2 if stage == "w13" else 1)

    def stage_bytes_per_worker(
        self,
        stage: str,
        threads: int,
        routes: int | None = None,
    ) -> int:
        del routes
        return threads * (2 if stage == "w13" else 1)


class _FinishAwareModel(_DeterministicTailPoolModel):
    def dag_task_finish_times(self, tasks) -> tuple[float, ...]:
        finish: list[float] = []
        for routes, threads, dependencies in tasks:
            start = max((finish[dependency] for dependency in dependencies), default=0.0)
            finish.append(start + self.T_iso(routes, threads))
        return tuple(finish)


class _RoutingFinishAwareModel(_FinishAwareModel):
    supported_shapes = ((1, 1),)
    supported_widths = (1,)

    def supports_shape(self, shape) -> bool:
        return tuple(shape) == self.supported_shapes[0]


class _DeterministicTailRepartitionModel(_DeterministicTailPoolModel):
    schema_version = 2
    supported_shapes = ((2, 2, 2, 2, 2, 2),)
    supported_widths = (2, 3, 4, 6)
    profile_runs = 1

    def T_iso(self, routes: int, threads: int) -> float:
        if threads == 3 and routes % 100:
            raise KeyError("3T is only calibrated for the synthetic 100-route point")
        return {2: 100.0, 3: 65.0, 4: 72.0, 6: 85.0}[threads]

    def supports_shape(self, shape) -> bool:
        return tuple(shape) == self.supported_shapes[0]


class _TemporalOrderingModel(_DeterministicTailPoolModel):
    """Synthetic event model where like resource phases contend."""

    supported_widths = (1,)

    def __init__(self) -> None:
        self.dag_calls = 0

    def T_iso(self, routes: int, threads: int) -> float:
        assert threads == 1
        return 10.0 if routes >= 100 else 2.0

    def dag_makespan(self, tasks) -> float:
        self.dag_calls += 1
        task_list = list(tasks)
        remaining = [self.T_iso(routes, threads) for routes, threads, _ in task_list]
        completed = [False] * len(task_list)
        active = {
            task_id
            for task_id, (_, _, dependencies) in enumerate(task_list)
            if not dependencies
        }
        elapsed = 0.0
        while active:
            resource_classes = {task_list[task_id][0] >= 100 for task_id in active}
            slowdown = 2.0 if len(active) > 1 and len(resource_classes) == 1 else 1.0
            step = min(remaining[task_id] * slowdown for task_id in active)
            elapsed += step
            for task_id in active:
                remaining[task_id] -= step / slowdown
            finished = [task_id for task_id in active if remaining[task_id] <= 1e-12]
            for task_id in finished:
                active.remove(task_id)
                completed[task_id] = True
            for task_id, (_, _, dependencies) in enumerate(task_list):
                if not completed[task_id] and task_id not in active and all(
                    completed[dependency] for dependency in dependencies
                ):
                    active.add(task_id)
        assert all(completed)
        return elapsed


def test_quick_planner_uses_bounded_homogeneous_lpt_search() -> None:
    planner = IntervalPlanner(
        _QuickPlannerModel(),
        num_cores=4,
        native_cold_planner=False,
    )

    large = planner.plan_quick([(0, 100), (1, 100)])
    short = planner.plan_quick([(expert, 1) for expert in range(8)])

    assert large["shape"] == (4,)
    assert large["makespan_ns"] == 200.0
    assert short["shape"] == (1, 1, 1, 1)
    assert short["makespan_ns"] == 20.0
    assert large["planner_backend"] == "python_quick"
    assert large["dynamic_candidates"] == 0


def test_planned_moe_quick_search_caches_selected_shape() -> None:
    planner = PlannedMoE(_QuickPlannerModel(), num_cores=4, search_mode="quick")

    first = planner.plan_spec_for([(0, 100), (1, 100)])
    assert first["shape"] == (4,)
    assert planner.last["planner_backend"] == "python_quick"
    assert not planner.last["cache_hit"]

    second = planner.plan_spec_for([(0, 100), (1, 100)])
    assert second["shape"] == (4,)
    assert planner.last["cache_hit"]


def test_planned_moe_rejects_unknown_search_mode() -> None:
    with pytest.raises(ValueError, match="search_mode"):
        PlannedMoE(_QuickPlannerModel(), num_cores=4, search_mode="unknown")


def _expert_ids_by_core(tasks) -> dict[int, tuple[int, ...]]:
    result: dict[int, list[int]] = {}
    for expert_id, _, core_begin, _, _ in tasks:
        result.setdefault(core_begin, []).append(expert_id)
    return {core_begin: tuple(expert_ids) for core_begin, expert_ids in result.items()}


def test_temporal_assignment_interleaves_resource_classes_without_moving_lanes() -> None:
    planner = IntervalPlanner(
        _TemporalOrderingModel(),
        num_cores=2,
        widths=(1,),
        shapes=((1, 1),),
        native_cold_planner=False,
        tail_repartition_widths=(),
    )
    experts = [(0, 100), (1, 100), (2, 1), (3, 1)]
    lanes = planner._lanes((1, 1))
    lpt_tasks = planner._build_tasks(experts, lanes, planner._assign_lpt(experts, lanes))

    makespan, tasks = planner.score_shape(experts, (1, 1))

    assert planner._score(lpt_tasks) == pytest.approx(24.0)
    assert makespan == pytest.approx(20.0)
    assert sorted(routes for _, routes, _, _, dependencies in tasks if not dependencies) == [1, 100]
    assert {
        core_begin: frozenset(expert_ids)
        for core_begin, expert_ids in _expert_ids_by_core(tasks).items()
    } == {
        core_begin: frozenset(expert_ids)
        for core_begin, expert_ids in _expert_ids_by_core(lpt_tasks).items()
    }


def test_temporal_assignment_retains_lpt_on_model_tie() -> None:
    planner = IntervalPlanner(
        _DeterministicTailPoolModel(),
        num_cores=2,
        widths=(1,),
        shapes=((1, 1),),
        native_cold_planner=False,
        tail_repartition_widths=(),
    )
    experts = [(0, 100), (1, 100), (2, 1), (3, 1)]
    lanes = planner._lanes((1, 1))
    lpt_tasks = planner._build_tasks(experts, lanes, planner._assign_lpt(experts, lanes))

    _, tasks = planner.score_shape(experts, (1, 1))

    assert tasks == lpt_tasks


def test_cached_plan_reuses_temporal_order_without_rescoring() -> None:
    model = _TemporalOrderingModel()
    runtime = PlannedMoE(model, num_cores=2)
    experts = [(0, 100), (1, 100), (2, 1), (3, 1)]

    cold = runtime.plan_spec_for(experts, dynamic_tail_pool=False)
    assert model.dag_calls > 0
    model.dag_calls = 0

    cached = runtime.plan_spec_for(experts, dynamic_tail_pool=False)

    assert runtime.last["cache_hit"] is True
    assert cached["assignment_order"] == cold["assignment_order"] == "reverse_odd"
    assert cached["bridge"] == cold["bridge"]
    assert model.dag_calls == 0


def test_planned_two_stage_planner_can_choose_different_stage_shapes() -> None:
    planner = PlannedTwoStagePlanner(
        _DeterministicStageModel(),
        num_cores=4,
        widths=(1, 2, 4),
        shapes=((4,), (2, 2), (1, 1, 1, 1)),
    )

    selected = planner.plan([(expert, 100) for expert in range(4)], dynamic_tail_pool=False)

    assert selected["w13"]["shape"] == (2, 2)
    assert selected["w2"]["shape"] == (1, 1, 1, 1)
    assert selected["w13"]["bridge"]["num_threads"] == 4
    assert selected["w2"]["bridge"]["num_threads"] == 4
    assert selected["makespan_ns"] == pytest.approx(
        selected["call_setup_ns"] + selected["w13"]["makespan_ns"] + selected["w2"]["makespan_ns"]
    )


def test_strict_bridge_disables_early_merge_for_equal_predicted_finishes() -> None:
    planner = IntervalPlanner(
        _FinishAwareModel(),
        num_cores=4,
        widths=(2,),
        shapes=((2, 2),),
        native_cold_planner=False,
        tail_repartition_widths=(),
    )
    balanced = [
        (0, 100, 0, 2, []),
        (1, 100, 2, 2, []),
    ]
    staggered = [
        (0, 100, 0, 2, []),
        (1, 100, 2, 2, [0]),
    ]

    assert planner.to_async_bridge(balanced)["early_merge"] is False
    assert planner.to_async_bridge(staggered)["early_merge"] is None


def test_routing_tail_gate_is_recomputed_for_same_histogram_cache_hit() -> None:
    runtime = PlannedMoE(_RoutingFinishAwareModel(), num_cores=2)
    counts = [(expert, 16) for expert in range(4)]
    overlapping_tail = torch.tensor(
        [[2, 3]] * 16 + [[0, 1]] * 16,
        dtype=torch.int32,
    )
    concentrated_tail = torch.tensor(
        [[0, 1, 2, 3]] * 16,
        dtype=torch.int32,
    )

    overlap = runtime.plan_spec_for(
        counts,
        topk_ids=overlapping_tail,
        dynamic_tail_pool=False,
    )
    assert overlap["bridge"]["early_merge"] is None
    assert runtime.last["cache_hit"] is False

    concentrated = runtime.plan_spec_for(
        counts,
        topk_ids=concentrated_tail,
        dynamic_tail_pool=False,
    )
    assert concentrated["bridge"]["early_merge"] is False
    assert runtime.last["cache_hit"] is True
    assert runtime.last["routing_aware_early_merge"] is True

    without_routing = runtime.plan_spec_for(counts, dynamic_tail_pool=False)
    assert without_routing["bridge"]["early_merge"] is None
    assert runtime.last["cache_hit"] is True
    assert runtime.last["routing_aware_early_merge"] is False


def test_routing_tail_gate_uses_one_owner_batch_as_conservative_cutoff() -> None:
    planner = IntervalPlanner(
        _RoutingFinishAwareModel(),
        num_cores=2,
        widths=(1,),
        shapes=((1, 1),),
        native_cold_planner=False,
        tail_repartition_widths=(),
    )
    tasks = [
        (0, 4, 0, 1, []),
        (2, 28, 0, 1, [0]),
        (1, 4, 1, 1, []),
        (3, 28, 1, 1, [2]),
    ]
    topk_ids = torch.tensor(
        [[2, 3]] * 28 + [[0, 1]] * 4,
        dtype=torch.int64,
    )

    policy, diagnostics = planner._early_merge_decision(tasks, topk_ids)

    assert policy is False
    assert diagnostics == {
        "reason": "routing_tail_bound",
        "dominant_expert": 2,
        "dominant_wave_experts": (2, 3),
        "dominant_wave_finish_ns": 20.0,
        "burst_tokens_lower_bound": 28,
        "outside_burst_tokens_upper_bound": 4,
        "later_route_occurrences": 0,
        "drain_capacity": 4,
    }


def test_planner_selects_tail_pool_width_and_can_disable_dynamic() -> None:
    planner = IntervalPlanner(
        _DeterministicTailPoolModel(),
        num_cores=4,
        widths=(1, 2, 4),
        shapes=((4,),),
    )
    experts = [(0, 100), (1, 100), *((expert, 1) for expert in range(2, 10))]

    selected = planner.plan(experts)
    strict = planner.plan(experts, dynamic_tail_pool=False, tail_pool_max_routes=0)
    forced = planner.plan(experts, forced_tail_pool_threads=2)

    assert selected["execution_mode"] == "tail_pool"
    assert selected["tail_pool_threads"] == 1
    assert selected["tail_pool_max_routes"] == 1
    assert selected["tail_pool_tasks"] == 8
    assert selected["makespan_ns"] < strict["makespan_ns"]
    assert strict["execution_mode"] == "strict"
    assert forced["execution_mode"] == "tail_pool"
    assert forced["tail_pool_threads"] == 2


def test_planner_keeps_strict_when_tail_pool_does_not_improve_makespan() -> None:
    planner = IntervalPlanner(
        _DeterministicTailPoolModel(),
        num_cores=4,
        widths=(1, 2, 4),
        shapes=((4,),),
    )

    selected = planner.plan([(0, 100), (1, 100), (2, 1)])

    assert selected["execution_mode"] == "strict"
    assert selected["tail_pool_threads"] is None


def test_planned_moe_defaults_to_auto_and_isolates_strict_cache_entries() -> None:
    runtime = PlannedMoE(_DeterministicTailPoolModel(), num_cores=4)
    experts = [(0, 100), (1, 100), *((expert, 1) for expert in range(2, 10))]

    automatic = runtime.plan_spec_for(experts)
    cached = runtime.plan_spec_for(experts)
    assert runtime.last["cache_hit"] is True
    strict = runtime.plan_spec_for(experts, dynamic_tail_pool=False)

    assert automatic["execution_mode"] == "tail_pool"
    assert automatic["tail_pool_threads"] == 1
    assert cached["bridge"] == automatic["bridge"]
    assert strict["execution_mode"] == "strict"
    assert strict["bridge"] != automatic["bridge"]
    assert runtime.last["cache_hit"] is False


def test_planned_moe_tail_pool_cache_tracks_threshold_eligibility() -> None:
    runtime = PlannedMoE(_DeterministicTailPoolModel(), num_cores=4)
    first = [(0, 100), (1, 100), (2, 11), (3, 11), *((expert, 13) for expert in range(4, 10))]
    second = [(0, 100), (1, 100), (2, 11), *((expert, 13) for expert in range(3, 10))]
    assert signature(first) == signature(second)

    first_plan = runtime.plan_spec_for(first)
    second_plan = runtime.plan_spec_for(second)

    assert first_plan["execution_mode"] == "tail_pool"
    assert second_plan["execution_mode"] == "strict"
    assert runtime.last["cache_hit"] is False


def test_planner_selects_one_bounded_terminal_repartition() -> None:
    planner = IntervalPlanner(
        _DeterministicTailRepartitionModel(),
        num_cores=12,
        widths=(2, 3, 4, 6),
        shapes=((2, 2, 2, 2, 2, 2),),
        tail_repartition_widths=(3, 4, 6),
    )
    experts = [(expert, 100) for expert in range(8)]

    selected = planner.plan(
        experts,
        dynamic_tail_pool=False,
        bounded_tail_repartition=True,
    )
    strict = planner.plan(
        experts,
        dynamic_tail_pool=False,
        bounded_tail_repartition=False,
    )

    assert selected["execution_mode"] == "strict"
    assert selected["tail_repartition_width"] == 3
    assert selected["tail_repartition_tasks"] == 2
    assert selected["tail_repartition_route_slices"] == 1
    assert selected["tail_repartition_candidates"] == 3
    assert selected["makespan_ns"] == pytest.approx(165.0)
    assert selected["makespan_ns"] < strict["makespan_ns"]
    assert selected["tasks"] == [
        (0, 100, 0, 2, []),
        (1, 100, 2, 2, []),
        (2, 100, 4, 2, []),
        (3, 100, 6, 2, []),
        (4, 100, 8, 2, []),
        (5, 100, 10, 2, []),
        (6, 100, 0, 3, [0, 1]),
        (7, 100, 6, 3, [3, 4]),
    ]
    assert selected["bridge"]["task_threads"][-2:] == [3, 3]
    assert strict["tail_repartition_width"] is None
    assert strict["tail_repartition_candidates"] == 0


def test_bounded_tail_repartition_requires_exactly_one_terminal_wave() -> None:
    planner = IntervalPlanner(
        _DeterministicTailRepartitionModel(),
        num_cores=12,
        widths=(2, 3, 4, 6),
        shapes=((2, 2, 2, 2, 2, 2),),
        tail_repartition_widths=(3, 4, 6),
    )

    selected = planner.plan(
        [(expert, 100) for expert in range(9)],
        dynamic_tail_pool=False,
        bounded_tail_repartition=True,
    )

    assert selected["tail_repartition_width"] is None
    assert selected["tail_repartition_candidates"] == 0


def test_fulln_profile_does_not_reuse_legacy_tail_repartition_anchor() -> None:
    profile = (
        PROFILE_DIR
        / "contention_async_amazon_c5_192c_numa0_tp4_sve_F512_E256_fulln_schema_v2_xbyak_exactm_20260727.json"
    )
    model = ContentionCostModel(profile)
    root_shape = (16, 16, 16, 16, 16, 16)

    assert not model.can_use_bounded_tail_repartition_anchor(1536, root_shape, 24)
    assert not model.can_use_bounded_tail_repartition_anchor(1536, root_shape, 24, 2)
    assert not model.can_use_bounded_tail_repartition_anchor(1548, root_shape, 24)
    assert not model.can_use_bounded_tail_repartition_anchor(1536, root_shape, 16)

    planner = IntervalPlanner(model, num_cores=96, native_cold_planner=False)
    selected = planner.plan(
        [(expert, 1536) for expert in range(8)],
        dynamic_tail_pool=False,
        bounded_tail_repartition=True,
    )
    assert selected["shape"] == (32, 32, 32)
    assert selected["tail_repartition_width"] is None
    assert selected["tail_repartition_candidates"] == 0


def test_planned_moe_cache_rebuilds_bounded_tail_tasks() -> None:
    runtime = PlannedMoE(
        _DeterministicTailRepartitionModel(),
        num_cores=12,
        tail_repartition_widths=(3, 4, 6),
    )
    experts = [(expert, 100) for expert in range(8)]

    first = runtime.plan_spec_for(
        experts,
        dynamic_tail_pool=False,
        bounded_tail_repartition=True,
    )
    second = runtime.plan_spec_for(
        experts,
        dynamic_tail_pool=False,
        bounded_tail_repartition=True,
    )

    assert runtime.last["cache_hit"] is True
    assert first["tail_repartition_width"] == 3
    assert first["tail_repartition_route_slices"] == 1
    assert second["tail_repartition_width"] == 3
    assert second["bridge"] == first["bridge"]

    changed = runtime.plan_spec_for(
        [(expert, 101) for expert in range(8)],
        dynamic_tail_pool=False,
        bounded_tail_repartition=True,
    )
    assert runtime.last["cache_hit"] is False
    assert changed["tail_repartition_width"] == 4


def test_iso_formula_recovers_separable_measurements() -> None:
    alpha, beta = 0.03, 0.001
    thread_points = [
        (
            team,
            (1.0 + alpha * (team - 1) + beta * team * (team - 1)) / team,
        )
        for team in (1, 2, 4, 8)
    ]
    truth = IsoFormula(
        100.0,
        300.0,
        alpha,
        beta,
        [(route, 10.0 * route) for route in (1, 2, 4, 8, 12, 48, 256, 512)],
        thread_points,
    )
    points = [
        (route, team, truth.T_iso(route, team)) for route in (1, 2, 4, 8, 12, 48, 256, 512) for team in (1, 2, 4, 8)
    ]
    fitted = fit_from_measurements(points, phi_route_min=256)

    assert fitted.o0 == pytest.approx(truth.o0)
    assert fitted.o1 == pytest.approx(truth.o1)
    assert fitted.alpha == pytest.approx(truth.alpha)
    assert fitted.beta == pytest.approx(truth.beta)
    assert fitted.T_iso(96, 4) == pytest.approx(truth.T_iso(96, 4))
    assert IsoFormula.from_dict(fitted.to_dict()).T_iso(96, 4) == pytest.approx(truth.T_iso(96, 4))
    with pytest.raises(ValueError, match="outside calibrated domain"):
        fitted.T_iso(96, 16)


def test_schema_v2_uses_formula_with_table_fallback(catalog: ProfileCatalog) -> None:
    record = catalog.select(
        ProfileQuery(
            mode="tp",
            degree=2,
            hidden_size=4096,
            intermediate_size=1024,
            local_experts=64,
        )
    )
    default = ContentionCostModel(record.path)
    formula = ContentionCostModel(record.path, iso_mode="formula")
    table = ContentionCostModel(record.path, iso_mode="table")

    assert default.iso_mode == "table"
    assert formula.iso_mode == "formula"
    assert formula.iso_formula is not None
    assert table.iso_mode == "table"
    assert table.iso_formula is None
    assert formula.T_iso(24, 8) == table.T_iso(24, 8)
    assert formula.T_iso(192, 8) != pytest.approx(table.T_iso(192, 8))


def test_catalog_requires_exact_calibration_domain(catalog: ProfileCatalog) -> None:
    query = ProfileQuery(
        mode="tp",
        degree=2,
        hidden_size=4096,
        intermediate_size=1024,
        local_experts=64,
    )
    record = catalog.select(query)
    assert record.policy.llc_bytes_per_rank == 48 * 1024 * 1024
    assert record.payload["kernel"]["stage_geometry"] == "full_n_team_stripes"

    with pytest.raises(ProfileCompatibilityError, match="no exact profile"):
        catalog.select(
            ProfileQuery(
                mode="tp",
                degree=4,
                hidden_size=4096,
                intermediate_size=512,
            )
        )


def test_full_stage_geometry_matches_native_tile_partition() -> None:
    w13 = full_stage_geometry(k=4096, n=1024, n_tile=8)
    w2 = full_stage_geometry(k=512, n=4096, n_tile=8)

    assert w13.stage_bytes == 8 * 1024 * 1024
    assert w2.stage_bytes == 4 * 1024 * 1024
    assert w13.bytes_per_worker(1) == 8 * 1024 * 1024
    assert w13.bytes_per_worker(8) == 1024 * 1024
    assert w13.bytes_per_worker(32) == 256 * 1024
    assert w2.bytes_per_worker(32) == 128 * 1024


@pytest.mark.parametrize("threads", [1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
def test_full_stripe_plan_is_the_single_window_endpoint(threads: int) -> None:
    """`window_tiles = tiles_per_worker(t)` must reproduce full_n_team_stripes."""
    for k, n in ((4096, 1024), (512, 4096)):
        geometry = full_stage_geometry(k=k, n=n, n_tile=8)
        plan = geometry.full_stripe_plan(threads)

        assert plan.windows == 1
        assert plan.bytes_per_worker == geometry.bytes_per_worker(threads)

        covered = [plan.thread_range(0, tid) for tid in range(threads)]
        assert sum(item.tiles for item in covered) == geometry.total_tiles
        assert sum(1 for item in covered if not item.is_empty) == geometry.active_threads(threads)
        # Ranges tile the stage in order with no gap and no overlap.
        cursor = 0
        for item in covered:
            assert item.begin_tile == cursor
            cursor += item.tiles


@pytest.mark.parametrize("threads", [1, 2, 4, 8, 16, 32])
def test_power_of_two_widths_never_starve_a_worker(threads: int) -> None:
    """When `threads` divides `total_tiles` the short tail stays a multiple of it."""
    for k, n in ((4096, 1024), (512, 4096)):
        geometry = full_stage_geometry(k=k, n=n, n_tile=8)
        assert geometry.total_tiles % threads == 0
        for window_tiles in range(1, geometry.tiles_per_worker(threads) + 1):
            plan = geometry.window_plan(threads, window_tiles)
            assert not plan.starves_any_thread(), f"t={threads} omega={window_tiles}"


def test_short_tail_window_can_starve_a_non_power_of_two_width() -> None:
    """t=6, omega=7 leaves a 2-tile tail for 6 workers, only in the last window."""
    geometry = full_stage_geometry(k=4096, n=1024, n_tile=8)
    plan = geometry.window_plan(threads=6, window_tiles=7)

    assert plan.range_tiles == 42
    assert plan.windows == 4
    assert plan.window_tile_span(3) == (126, 2)

    for index in range(plan.windows - 1):
        assert plan.idle_threads(index) == 0
    assert plan.idle_threads(3) == 4
    assert plan.starves_any_thread()


def test_windows_tile_the_stage_exactly() -> None:
    geometry = full_stage_geometry(k=4096, n=1024, n_tile=8)
    for threads in (1, 2, 3, 4, 6, 8):
        for window_tiles in (1, 2, 3, 5, 7, 16, geometry.tiles_per_worker(threads)):
            plan = geometry.window_plan(threads, window_tiles)
            seen: list[tuple[int, int]] = []
            for index in range(plan.windows):
                for tid in range(threads):
                    item = plan.thread_range(index, tid)
                    if not item.is_empty:
                        seen.append((item.begin_tile, item.tiles))
            seen.sort()
            cursor = 0
            for begin, tiles in seen:
                assert begin == cursor, f"t={threads} omega={window_tiles}"
                cursor += tiles
            assert cursor == geometry.total_tiles, f"t={threads} omega={window_tiles}"


def test_window_tiles_from_bytes_is_the_only_byte_conversion() -> None:
    w13 = full_stage_geometry(k=4096, n=1024, n_tile=8)
    assert w13.bytes_per_tile == 64 * 1024

    assert w13.window_tiles_from_bytes(64 * 1024) == 1
    assert w13.window_tiles_from_bytes(128 * 1024) == 2
    # Floors to whole tiles, so a range of byte budgets shares one pattern.
    assert w13.window_tiles_from_bytes(191 * 1024) == 2
    # Clamps: a tile cannot be subdivided, and the window cannot exceed the stage.
    assert w13.window_tiles_from_bytes(1) == 1
    assert w13.window_tiles_from_bytes(64 * 1024 * 1024) == w13.total_tiles

    with pytest.raises(ValueError, match="must be positive"):
        w13.window_tiles_from_bytes(0)


def test_window_plan_rejects_illegal_windows() -> None:
    geometry = full_stage_geometry(k=4096, n=1024, n_tile=8)
    with pytest.raises(ValueError, match="window tiles must be positive"):
        geometry.window_plan(threads=4, window_tiles=0)
    with pytest.raises(ValueError, match="exceeds stage tiles"):
        geometry.window_plan(threads=4, window_tiles=geometry.total_tiles + 1)
    with pytest.raises(ValueError, match="threads must be positive"):
        geometry.window_plan(threads=0, window_tiles=1)


def test_profile_rejects_legacy_split_stage_calibration(
    catalog: ProfileCatalog,
    tmp_path: Path,
) -> None:
    record = catalog.select(
        ProfileQuery(
            mode="tp",
            degree=2,
            hidden_size=4096,
            intermediate_size=1024,
            local_experts=64,
        )
    )
    payload = json.loads(record.path.read_text(encoding="utf-8"))
    payload["kernel"]["w13_window_ranges"] = 2
    payload["kernel"]["w2_window_ranges"] = 1
    path = tmp_path / "legacy_split_stage.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProfileCompatibilityError, match="split-stage calibration"):
        ProfileCatalog.from_paths([path])


def test_profile_full_n_geometry_is_not_a_query_dimension(catalog: ProfileCatalog) -> None:
    record = catalog.select(
        ProfileQuery(
            mode="tp",
            degree=2,
            hidden_size=4096,
            intermediate_size=1024,
            local_experts=64,
        )
    )

    assert record.payload["kernel"]["stage_geometry"] == "full_n_team_stripes"
    assert not hasattr(record.policy, "w13_window_ranges")
    assert not hasattr(record.policy, "w2_window_ranges")
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        ProfileQuery(w13_window_ranges=1)


def test_catalog_rejects_duplicate_full_n_domain(
    catalog: ProfileCatalog,
    tmp_path: Path,
) -> None:
    record = catalog.select(
        ProfileQuery(
            mode="tp",
            degree=2,
            hidden_size=4096,
            intermediate_size=1024,
            local_experts=64,
        )
    )
    duplicate = tmp_path / "duplicate_full_n.json"
    duplicate.write_text(record.path.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ProfileCompatibilityError, match="duplicate calibration domain"):
        ProfileCatalog.from_paths([record.path, duplicate])


def test_planned_moe_rejects_multiple_calibration_models(catalog: ProfileCatalog) -> None:
    model = model_for(catalog, "tp", 1024, 64)
    with pytest.raises(ValueError, match="one calibration model"):
        PlannedMoE((model, model), 32)


def test_tail_pool_bridge_relinks_fixed_lane_dependencies(
    catalog: ProfileCatalog,
) -> None:
    model = model_for(catalog, "tp", 1024, 64)
    runtime = PlannedMoE(model, 32)
    planner = runtime.interval_planners[0]
    tasks = [
        (0, 2040, 0, 4, []),
        (1, 12, 0, 4, [0]),
        (2, 8, 0, 4, [1]),
        (3, 2040, 0, 4, [2]),
    ]

    bridge = planner.to_tail_pool_bridge(
        tasks,
        pool_threads=2,
        max_pooled_routes=12,
    )

    assert bridge["execution_mode"] == "tail_pool"
    assert bridge["task_core_begins"] == [0, -1, -1, 0]
    assert bridge["task_threads"] == [4, 2, 2, 4]
    assert bridge["task_placement_modes"] == [0, 1, 1, 0]
    assert bridge["task_dep_offsets"] == [0, 0, 0, 0, 1]
    assert bridge["task_deps"] == [0]

    spec = runtime.plan_spec_for(
        [(0, 2040), (1, 12), (2, 8), (3, 2040)],
        tail_pool_threads=1,
        tail_pool_max_routes=12,
    )
    assert spec["execution_mode"] == "tail_pool"
    assert spec["bridge"]["task_placement_modes"].count(1) == 2
    assert runtime.last["execution_mode"] == "tail_pool"


def test_m12_tail_composition(catalog: ProfileCatalog) -> None:
    model = model_for(catalog, "tp", 1024, 64)
    assert model.m12_effective_rows(3) == 4
    assert model.m12_effective_rows(7) == 8
    assert model.m12_effective_rows(11) == 12
    assert model.m12_effective_rows(23) == 24
    assert model.T_iso(3, 4) == model.T_iso(4, 4)
    assert model.T_iso(9, 4) == model.T_iso(12, 4)

    overhead = model._O[4]
    expected = overhead + (model.T_iso(12, 4) - overhead) + (model.T_iso(1, 4) - overhead)
    assert model.T_iso(13, 4) == pytest.approx(expected)


def test_exact_m_profile_does_not_round_tail_routes(
    catalog: ProfileCatalog,
    tmp_path: Path,
) -> None:
    record = catalog.select(
        ProfileQuery(
            mode="tp",
            degree=2,
            hidden_size=4096,
            intermediate_size=1024,
            local_experts=64,
        )
    )
    payload = json.loads(record.path.read_text(encoding="utf-8"))
    payload["kernel"]["sve_implementation"] = "jit"
    payload["kernel"]["m_tail_policy"] = "xbyak_exact_m"
    path = tmp_path / "exact_m_profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    mixed_catalog = ProfileCatalog.from_paths([record.path, path])
    with pytest.raises(ProfileCompatibilityError, match="ambiguous"):
        mixed_catalog.select(
            ProfileQuery(
                mode="tp",
                degree=2,
                hidden_size=4096,
                intermediate_size=1024,
                local_experts=64,
            )
        )
    selected = mixed_catalog.select(
        ProfileQuery(
            mode="tp",
            degree=2,
            hidden_size=4096,
            intermediate_size=1024,
            local_experts=64,
            sve_implementation="jit",
            m_tail_policy="xbyak_exact_m",
        )
    )
    assert selected.path == path

    model = ContentionCostModel(path, iso_mode="table")
    assert model.m12_effective_rows(3) == 3
    assert model.m12_effective_rows(7) == 7
    assert model.m12_effective_rows(11) == 11
    assert model.m12_effective_rows(23) == 23
    assert model.T_iso(3, 4) != model.T_iso(4, 4)


def test_checked_in_xbyak_profiles_cover_exact_m_endpoints() -> None:
    paths = xbyak_profile_paths()
    assert len(paths) >= 4
    exact_catalog = ProfileCatalog.from_paths(paths)

    identities: set[tuple[object, ...]] = set()
    for record in exact_catalog.records:
        payload = record.payload
        policy = record.policy
        assert policy.sve_implementation == "jit"
        assert policy.m_tail_policy == "xbyak_exact_m"
        assert payload["kernel"]["stage_geometry"] == "full_n_team_stripes"
        assert "w13_window_ranges" not in payload["kernel"]
        assert "w2_window_ranges" not in payload["kernel"]
        assert payload["kernel"]["xbyak_aarch64_commit"] == XBYAK_AARCH64_COMMIT
        assert set(range(1, 13)).issubset(map(int, payload["isolated_routes"]))
        assert set(range(1, 13)).issubset(map(int, payload["contention_routes"]))
        assert payload["kernel"]["source_sha256"]
        assert payload["kernel"]["extension_sha256"]
        assert policy.identity_key() not in identities
        identities.add(policy.identity_key())

        model = ContentionCostModel(
            record.path,
            expected_policy=ProfileQuery(
                sve_implementation="jit",
                m_tail_policy="xbyak_exact_m",
            ),
        )
        assert [model.m12_effective_rows(routes) for routes in range(1, 13)] == list(range(1, 13))

    assert len(identities) == len(exact_catalog.records)


def test_parallel_evaluator_prefers_jit_calibration_when_available(
    catalog: ProfileCatalog,
    tmp_path: Path,
) -> None:
    query = ProfileQuery(
        mode="tp",
        degree=2,
        hidden_size=4096,
        intermediate_size=1024,
        global_experts=64,
        local_experts=64,
        backend="sve",
        backend_n_tile=8,
        activation="silu",
        dtype="bf16",
        measurement_experts=64,
        cores_per_rank=32,
        concurrent_ranks=2,
    )
    asm_record = catalog.select(query)
    payload = json.loads(asm_record.path.read_text(encoding="utf-8"))
    payload["kernel"]["sve_implementation"] = "jit"
    payload["kernel"]["m_tail_policy"] = "xbyak_exact_m"
    jit_path = tmp_path / "jit.json"
    jit_path.write_text(json.dumps(payload), encoding="utf-8")

    topology = HierarchicalTopology(2, 1, 60e9, 20e9, 1e-6)

    asm_only = ParallelLayerEvaluator(
        ProfileCatalog.from_paths([asm_record.path]),
        topology,
        hidden_size=4096,
        full_intermediate_size=2048,
        global_experts=64,
        cores_per_rank=32,
    )._model("tp", 1024, 64)
    assert asm_only.policy.sve_implementation == "asm"

    selected = ParallelLayerEvaluator(
        ProfileCatalog.from_paths([asm_record.path, jit_path]),
        topology,
        hidden_size=4096,
        full_intermediate_size=2048,
        global_experts=64,
        cores_per_rank=32,
    )._model("tp", 1024, 64)
    assert selected.policy.sve_implementation == "jit"


def test_exact_shape_and_stage_working_sets(catalog: ProfileCatalog) -> None:
    model = model_for(catalog, "ep", 2048, 32)
    assert model.supports_shape((32,))
    assert not model.supports_shape((24, 8))
    with pytest.raises(ProfileCompatibilityError, match="was not measured"):
        model.profiled_group_time(192, (24, 8))

    calibration_worksets = [workset for _, workset in model._task_phases(192, 16)]
    assert [value for value in calibration_worksets if value] == [
        32 * 1024 * 1024,
        16 * 1024 * 1024,
    ]
    tasks = [(192, 16, []), (192, 16, [])]
    finish_times = model.dag_task_finish_times(tasks)
    assert finish_times[0] == pytest.approx(finish_times[1])
    assert max(finish_times) == pytest.approx(model.dag_makespan(tasks))


def test_joint_planner_and_physical_cpu_mapping(catalog: ProfileCatalog) -> None:
    tp_model = model_for(catalog, "tp", 1024, 64)
    assert tp_model.has_full_workload_anchors
    tp_planner = IntervalPlanner(tp_model, 32, cpu_ids=range(32, 64))
    tp = tp_planner.plan([(expert, 192) for expert in range(64)])
    assert tp["shape"] == (16, 16)
    assert tp["active_working_set_bytes"] == 32 * 1024 * 1024
    assert tp["bridge"]["thread_cpu_ids"] == list(range(32, 64))
    assert "w13_window_ranges" not in tp
    assert "w2_window_ranges" not in tp

    anchored_ms, _ = tp_planner.score_shape([(expert, 192) for expert in range(64)], (8, 8, 8, 8))
    assert anchored_ms == tp_model.profiled_full_call_time(192, (8, 8, 8, 8))

    ep = IntervalPlanner(model_for(catalog, "ep", 2048, 32), 32).plan(
        [(expert, 192) for expert in range(32)]
    )
    assert ep["shape"] == (32,)


def test_domain_bound_cache_and_richer_signature(catalog: ProfileCatalog) -> None:
    first = [(0, 10), (1, 6), (2, 2), (3, 2)]
    second = [(0, 10), (1, 4), (2, 4), (3, 2)]
    assert signature(first) != signature(second)

    planner = PlannedMoE(model_for(catalog, "tp", 1024, 64), 32)
    counts = [(expert, 192) for expert in range(64)]
    spec = planner.plan_spec_for(counts)
    assert "operator_options" not in spec
    assert "w13_window_ranges" not in spec
    assert "w2_window_ranges" not in spec
    assert "task_w13_ranges" not in spec["bridge"]
    assert "task_w2_ranges" not in spec["bridge"]
    assert planner.last["cache_hit"] is False
    cached = planner.plan_spec_for(counts)
    assert cached["shape"] == spec["shape"]
    assert planner.last["cache_hit"] is True


def test_real_routing_summary_offline_plan_and_cost_model(
    catalog: ProfileCatalog,
) -> None:
    workload = load_routing_workload()
    active = [routes for routes in workload.histogram if routes > 0]
    assert workload.name == "dsv4-real-2048-seq70"
    assert workload.tail_reconstructed
    assert workload.routes == workload.tokens * workload.top_k == 12_288
    assert len(workload.histogram) == workload.num_experts == 256
    assert len(active) == workload.observed_active_experts == 223
    assert (min(active), max(active)) == (1, 918)
    mean = sum(active) / len(active)
    reconstructed_std = math.sqrt(sum((routes - mean) ** 2 for routes in active) / len(active))
    assert reconstructed_std == pytest.approx(
        workload.observed_routes_std,
        abs=1e-4,
    )
    assert workload.histogram[71] == 918
    assert workload.histogram[45] == 674
    assert PRESETS[workload.name] == workload.experts

    planner = IntervalPlanner(model_for(catalog, "tp", 1024, 64), 32)
    result = planner.plan(workload.experts)
    strict_result = planner.plan(workload.experts, dynamic_tail_pool=False)
    assert math.isfinite(result["makespan_ns"])
    assert result["makespan_ns"] > 0
    assert result["makespan_ns"] <= strict_result["makespan_ns"]
    assert len(result["tasks"]) == workload.observed_active_experts
    assert sum(task[1] for task in result["tasks"]) == workload.routes
    assert len(result["bridge"]["task_expert_ids"]) == len(result["tasks"])
    assert "policy_ranking" not in result

    rescored_ns, rescored_tasks = planner.score_shape(
        workload.experts,
        strict_result["shape"],
    )
    assert rescored_ns == pytest.approx(strict_result["makespan_ns"])
    assert rescored_tasks == strict_result["tasks"]


def test_tp_ep_evaluator_and_generic_p2_collectives(
    catalog: ProfileCatalog,
) -> None:
    topology = HierarchicalTopology(
        ranks=2,
        ranks_per_group=1,
        intra_bytes_per_second=60e9,
        inter_bytes_per_second=20e9,
        latency_seconds=1e-6,
    )
    message = 2048 * 4096 * 2
    assert topology.allreduce_ms(message) == pytest.approx((message / 20e9 + 2e-6) * 1e3)
    outgoing = 1024 * 6 * 4096 * 2
    assert topology.alltoall_ms(outgoing) == pytest.approx(2 * (outgoing / 2 / 20e9 + 1e-6) * 1e3)

    evaluator = ParallelLayerEvaluator(
        catalog,
        topology,
        hidden_size=4096,
        full_intermediate_size=2048,
        global_experts=64,
        cores_per_rank=32,
    )
    tp = evaluator.evaluate_tp(2048, 6)
    ep = evaluator.evaluate_ep(2048, 6)
    assert tp.compute_ms == max(rank.predicted_ms for rank in tp.rank_compute)
    assert ep.compute_ms == max(rank.predicted_ms for rank in ep.rank_compute)
    assert tp.rank_compute[0].shape == (16, 16)
    assert ep.rank_compute[0].shape == (32,)
    assert tp.communication_ms < ep.communication_ms
    assert math.isclose(tp.total_ms, tp.compute_ms + tp.communication_ms)

    hotspot = [768] * 4 + [384] * 12 + [96] * 48
    ep_hotspot = evaluator.evaluate_ep(2048, 6, global_histogram=hotspot)
    assert [rank.routes for rank in ep_hotspot.rank_compute] == [9216, 3072]
    assert ep_hotspot.compute_ms == max(rank.predicted_ms for rank in ep_hotspot.rank_compute)
    assert ep_hotspot.rank_compute[0].predicted_ms != pytest.approx(ep_hotspot.rank_compute[1].predicted_ms)

    p4 = HierarchicalTopology(
        ranks=4,
        ranks_per_group=2,
        intra_bytes_per_second=60e9,
        inter_bytes_per_second=20e9,
        latency_seconds=1e-6,
    )
    assert p4.allreduce_ms(message) == pytest.approx((message / 60e9 + message / 20e9 + 4e-6) * 1e3)
    assert p4.alltoall_ms(outgoing) == pytest.approx(2 * (outgoing / 20e9 + 3e-6) * 1e3)


def test_ep_rank_lifetime_switches_to_single_rank_profile(
    catalog: ProfileCatalog,
    tmp_path: Path,
) -> None:
    query = ProfileQuery(
        mode="ep",
        degree=2,
        hidden_size=4096,
        intermediate_size=2048,
        global_experts=64,
        local_experts=32,
        backend="sve",
        backend_n_tile=8,
        activation="silu",
        dtype="bf16",
        measurement_experts=32,
        cores_per_rank=32,
        concurrent_ranks=2,
    )
    dual_record = catalog.select(query)
    paths = [dual_record.path]
    payload = json.loads(dual_record.path.read_text(encoding="utf-8"))
    target = payload["target"]
    target["concurrent_ranks"] = 1
    target["cpu_ids_by_rank"] = [target["cpu_ids_by_rank"][0]]
    target["numa_nodes"] = [target["numa_nodes"][0]]
    target["llc_bytes_by_rank"] = [target["llc_bytes_by_rank"][0]]
    for section in ("isolated", "entries"):
        for entry in payload[section]:
            for key, value in list(entry.items()):
                if key.endswith("_ns") and isinstance(value, (int, float)):
                    entry[key] = value * 0.5
    single_path = tmp_path / f"single_{dual_record.path.name}"
    single_path.write_text(json.dumps(payload), encoding="utf-8")
    paths.append(single_path)

    topology = HierarchicalTopology(2, 1, 60e9, 20e9, 1e-6)
    conservative = ParallelLayerEvaluator(
        catalog,
        topology,
        hidden_size=4096,
        full_intermediate_size=2048,
        global_experts=64,
        cores_per_rank=32,
    )
    switched = ParallelLayerEvaluator(
        ProfileCatalog.from_paths(paths),
        topology,
        hidden_size=4096,
        full_intermediate_size=2048,
        global_experts=64,
        cores_per_rank=32,
    )
    hotspot = [768] * 4 + [384] * 12 + [96] * 48
    conservative_result = conservative.evaluate_ep(2048, 6, global_histogram=hotspot)
    switched_result = switched.evaluate_ep(2048, 6, global_histogram=hotspot)

    assert switched_result.compute_ms < conservative_result.compute_ms
    assert switched_result.compute_ms > min(rank.predicted_ms for rank in switched_result.rank_compute)
    assert switched_result.compute_ms == max(rank.predicted_ms for rank in switched_result.rank_compute)
