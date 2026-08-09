from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
COST_MODEL = ROOT / "cpu_moe_schedule_optimization" / "cost_model"
PLANNERS = ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path[:0] = [str(COST_MODEL), str(PLANNERS)]

from interval_planner import IntervalPlanner, PlannedTwoStagePlanner, PolicyAwarePlanner  # noqa: E402
from iso_formula import IsoFormula, fit_from_measurements  # noqa: E402
from phase_model import ContentionCostModel  # noqa: E402
import planned_moe as planned_moe_module  # noqa: E402
from planned_moe import PlannedMoE, signature  # noqa: E402
from profile_catalog import (  # noqa: E402
    ProfileCatalog,
    ProfileCompatibilityError,
    ProfileQuery,
)
from simulate_schedules import PRESETS  # noqa: E402
from stage_window_policy import (  # noqa: E402
    AMAZON_C5_192C_TP4_F512_SHORT_ROUTE_BAND,
    AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1,
    AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V2,
    AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V3,
    AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V4,
    INHERIT_STAGE_WINDOW,
    StageWindowBand,
    StaticStageWindowPolicy,
    default_task_stage_window_policy,
)
from tp_vs_ep_model import HierarchicalTopology, ParallelLayerEvaluator  # noqa: E402
from weight_window import (  # noqa: E402
    achievable_worker_windows,
    fused_moe_weight_windows,
    range_bytes_for_worker_window,
    stage_weight_window_geometry,
)
from workload_catalog import load_routing_workload  # noqa: E402


PROFILE_DIR = COST_MODEL / "profiles"
XBYAK_AARCH64_COMMIT = "3f8c682b9c6ff562dc008c7d1b8307a683f05d20"


def profile_paths() -> list[Path]:
    return sorted(PROFILE_DIR.glob("*_v2_r1_20260713.json"))


def xbyak_profile_paths() -> list[Path]:
    return sorted(PROFILE_DIR.glob("*xbyak_exactm*.json"))


@pytest.fixture(scope="module")
def catalog() -> ProfileCatalog:
    return ProfileCatalog.from_paths(profile_paths())


def models(catalog: ProfileCatalog, mode: str, ffn: int, local_experts: int):
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
    no_split, split = catalog.stage_range_pair(query)
    return ContentionCostModel(no_split.path), ContentionCostModel(split.path)


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

    def window_bytes_per_worker(self, threads: int) -> int:
        return threads

    def task_stage_ranges(self, routes: int, threads: int) -> tuple[int, int]:
        del routes, threads
        return 2, 1


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

    def stage_window_bytes_per_worker(
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


def test_bounded_tail_repartition_uses_exact_layout_anchor() -> None:
    profile = (
        PROFILE_DIR
        / "contention_async_amazon_c5_192c_numa0_tp4_sve_F512_E256_splitw13_schema_v2_xbyak_exactm_20260727.json"
    )
    model = ContentionCostModel(profile)
    root_shape = (16, 16, 16, 16, 16, 16)

    assert model.can_use_bounded_tail_repartition_anchor(1536, root_shape, 24)
    assert model.can_use_bounded_tail_repartition_anchor(1536, root_shape, 24, 2)
    assert not model.can_use_bounded_tail_repartition_anchor(1548, root_shape, 24)
    assert not model.can_use_bounded_tail_repartition_anchor(1536, root_shape, 16)
    unsliced_median, unsliced_uncertainty = model.profiled_bounded_tail_repartition(
        1536,
        root_shape,
        24,
    )
    sliced_median, sliced_uncertainty = model.profiled_bounded_tail_repartition(
        1536,
        root_shape,
        24,
        2,
    )
    assert unsliced_median == pytest.approx(9_796_654.0)
    assert sliced_median == pytest.approx(8_710_369.0)
    assert min(unsliced_uncertainty, sliced_uncertainty) > 0.0

    planner = IntervalPlanner(model, num_cores=96, native_cold_planner=False)
    selected = planner.plan(
        [(expert, 1536) for expert in range(8)],
        dynamic_tail_pool=False,
        bounded_tail_repartition=True,
    )
    assert selected["tail_repartition_width"] == 24
    assert selected["tail_repartition_route_slices"] == 2
    assert selected["tail_repartition_candidates"] == 4
    assert selected["makespan_ns"] == pytest.approx(sliced_median)
    assert selected["tasks"][-4:] == [
        (6, 768, 0, 24, [0, 1]),
        (6, 768, 24, 24, [1, 2]),
        (7, 768, 48, 24, [3, 4]),
        (7, 768, 72, 24, [4, 5]),
    ]
    assert selected["bridge"]["task_range_granularities"][-4:] == [768, 768, 768, 768]


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
            w13_split=True,
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


def test_catalog_requires_exact_policy(catalog: ProfileCatalog) -> None:
    query = ProfileQuery(
        mode="tp",
        degree=2,
        hidden_size=4096,
        intermediate_size=1024,
        local_experts=64,
        w13_split=True,
    )
    record = catalog.select(query)
    assert record.policy.w13_split
    assert record.policy.llc_bytes_per_rank == 48 * 1024 * 1024
    assert record.policy.weight_window_bytes == 0
    assert record.policy.w13_window_ranges == record.policy.w13_split_chunks
    assert record.policy.w2_window_ranges == 1

    with pytest.raises(ProfileCompatibilityError, match="no exact profile"):
        catalog.select(
            ProfileQuery(
                mode="tp",
                degree=4,
                hidden_size=4096,
                intermediate_size=512,
            )
        )


def test_weight_window_geometry_matches_native_tile_partition() -> None:
    w13, w2 = fused_moe_weight_windows(
        hidden_size=4096,
        intermediate_size=512,
        n_tile=8,
        target_bytes=1024 * 1024,
        w13_fallback_ranges=1,
    )

    assert (w13.ranges, w2.ranges) == (8, 4)
    assert (w13.max_range_bytes, w2.max_range_bytes) == (1024 * 1024, 1024 * 1024)
    assert w13.bytes_per_worker(1) == 1024 * 1024
    assert w13.bytes_per_worker(32) == 64 * 1024
    assert w13.active_threads(32) == 16


MIB = 1024 * 1024

# TP4 H=4096 F=512 stage geometry: W13 packs [2F, H], W2 packs [H, F].
W13_STAGE = {"k": 4096, "n": 1024, "n_tile": 8}
W2_STAGE = {"k": 512, "n": 4096, "n_tile": 8}
STAGE_WIDTHS = (1, 2, 4, 8, 16, 32)

# The calibrated ``amazon_c5_192c_tp4_f512_v1`` table exactly as it was measured,
# keyed by ``(min_routes, max_routes, threads)`` in per-range bytes. Expressing
# the policy in per-thread windows must reproduce these values byte for byte.
STAGE_WINDOW_V1_LEGACY_BYTES: dict[tuple[int, int, int], tuple[int, int]] = {
    (49, 95, 8): (1 * MIB, 1 * MIB),
    (96, 143, 1): (MIB // 8, MIB // 8),
    (96, 143, 2): (MIB // 8, MIB // 4),
    (96, 143, 4): (MIB // 4, MIB // 2),
    (96, 143, 8): (1 * MIB, MIB // 2),
    (144, 287, 1): (MIB // 8, MIB // 8),
    (144, 287, 2): (MIB // 4, MIB // 4),
    (144, 287, 4): (MIB // 2, MIB // 2),
    (144, 287, 8): (1 * MIB, MIB // 2),
    (288, 575, 1): (1 * MIB, MIB // 2),
    (288, 575, 2): (1 * MIB, MIB // 4),
    (288, 575, 4): (2 * MIB, MIB // 2),
    (288, 575, 8): (4 * MIB, 1 * MIB),
}


@pytest.mark.parametrize("stage", [W13_STAGE, W2_STAGE], ids=["w13", "w2"])
@pytest.mark.parametrize("threads", STAGE_WIDTHS)
def test_achievable_worker_windows_round_trip(stage: dict, threads: int) -> None:
    windows = achievable_worker_windows(threads=threads, **stage)

    for worker_bytes, (ranges, range_bytes) in windows.items():
        assert range_bytes_for_worker_window(threads=threads, target_worker_bytes=worker_bytes, **stage) == range_bytes
        geometry = stage_weight_window_geometry(target_bytes=range_bytes, **stage)
        assert geometry.ranges == ranges
        assert geometry.max_range_bytes == range_bytes
        assert geometry.bytes_per_worker(threads) == worker_bytes


@pytest.mark.parametrize(
    ("stage", "bytes_per_tile", "total_bytes"),
    [(W13_STAGE, 64 * 1024, 8 * MIB), (W2_STAGE, 8 * 1024, 4 * MIB)],
    ids=["w13", "w2"],
)
@pytest.mark.parametrize("threads", STAGE_WIDTHS)
def test_achievable_worker_windows_are_tile_quantized(
    stage: dict,
    bytes_per_tile: int,
    total_bytes: int,
    threads: int,
) -> None:
    windows = achievable_worker_windows(threads=threads, **stage)

    assert all(worker_bytes % bytes_per_tile == 0 for worker_bytes in windows)
    assert min(windows) == bytes_per_tile
    assert max(windows) == total_bytes // threads


@pytest.mark.parametrize("stage", [W13_STAGE, W2_STAGE], ids=["w13", "w2"])
def test_range_bytes_for_worker_window_is_monotone(stage: dict) -> None:
    previous_ranges = 0
    for target in (4 * MIB, 2 * MIB, 1 * MIB, MIB // 2, MIB // 4, MIB // 8, MIB // 16):
        range_bytes = range_bytes_for_worker_window(threads=4, target_worker_bytes=target, **stage)
        ranges = stage_weight_window_geometry(target_bytes=range_bytes, **stage).ranges
        assert ranges >= previous_ranges
        previous_ranges = ranges


def test_range_bytes_for_worker_window_rejects_unreachable_target() -> None:
    with pytest.raises(ValueError, match="smallest achievable per-worker window is 65536"):
        range_bytes_for_worker_window(threads=1, target_worker_bytes=32 * 1024, **W13_STAGE)
    with pytest.raises(ValueError, match="target_worker_bytes must be positive"):
        range_bytes_for_worker_window(threads=1, target_worker_bytes=0, **W13_STAGE)


def test_calibrated_v1_windows_are_all_reachable() -> None:
    """Every measured V1 range budget is an achievable per-thread window."""
    for (_, _, threads), (w13_bytes, w2_bytes) in STAGE_WINDOW_V1_LEGACY_BYTES.items():
        for stage, range_bytes in ((W13_STAGE, w13_bytes), (W2_STAGE, w2_bytes)):
            reachable = {value for _, value in achievable_worker_windows(threads=threads, **stage).values()}
            assert range_bytes in reachable, f"threads={threads} range_bytes={range_bytes} stage={stage}"


# The V1 table re-expressed as band-level per-thread windows plus the deviating
# cells. Task 3 lands this shape in production; the test below proves the two
# forms are interchangeable before that happens.
STAGE_WINDOW_V1_PER_THREAD = (
    (49, 95, (8,), (MIB // 8, MIB // 8), ()),
    (
        96,
        143,
        (1, 2, 4, 8),
        (MIB // 8, MIB // 8),
        ((2, MIB // 16, MIB // 8), (4, MIB // 16, MIB // 8), (8, MIB // 8, MIB // 16)),
    ),
    (144, 287, (1, 2, 4, 8), (MIB // 8, MIB // 8), ((8, MIB // 8, MIB // 16),)),
    (288, 575, (1, 2, 4, 8), (MIB // 2, MIB // 8), ((1, 1 * MIB, MIB // 2),)),
)
STAGE_WINDOW_GEOMETRY = {"hidden_size": 4096, "intermediate_size": 512, "backend_n_tile": 8}


def _per_thread_policy() -> StaticStageWindowPolicy:
    return StaticStageWindowPolicy(
        name="dual_form_per_thread",
        bands=tuple(
            StageWindowBand(
                min_routes=min_routes,
                max_routes=max_routes,
                widths=widths,
                w13_bytes_per_thread=window[0],
                w2_bytes_per_thread=window[1],
                thread_overrides=overrides,
            )
            for min_routes, max_routes, widths, window, overrides in STAGE_WINDOW_V1_PER_THREAD
        ),
        **STAGE_WINDOW_GEOMETRY,
    )


def _literal_policy() -> StaticStageWindowPolicy:
    by_band: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for (min_routes, max_routes, threads), windows in STAGE_WINDOW_V1_LEGACY_BYTES.items():
        by_band.setdefault((min_routes, max_routes), []).append((threads, *windows))
    return StaticStageWindowPolicy(
        name="dual_form_literal",
        bands=tuple(
            StageWindowBand.from_thread_windows(min_routes, max_routes, sorted(rows))
            for (min_routes, max_routes), rows in sorted(by_band.items())
        ),
    )


def test_stage_window_band_forms_agree() -> None:
    """A per-thread band and its lowered per-range twin select identically."""
    per_thread = _per_thread_policy()
    literal = _literal_policy()

    for routes in (12, 48, 49, 95, 96, 143, 144, 287, 288, 575, 576, 2040):
        for threads in STAGE_WIDTHS:
            assert per_thread.select(routes, threads) == literal.select(routes, threads), (routes, threads)


def test_stage_window_band_forms_agree_on_cost_model_entries() -> None:
    def bands(policy: StaticStageWindowPolicy) -> set[tuple[int, ...]]:
        return {
            (entry.min_routes, entry.max_routes, entry.threads, entry.w13_window_bytes, entry.w2_window_bytes)
            for entry in policy.cost_model_entries()
        }

    assert bands(_per_thread_policy()) == bands(_literal_policy())


def test_stage_window_policy_reports_achieved_worker_windows() -> None:
    policy = _per_thread_policy()

    assert policy.worker_windows(200, 4) == (MIB // 8, MIB // 8)
    assert policy.worker_windows(400, 8) == (MIB // 2, MIB // 8)
    assert policy.worker_windows(12, 4) == (INHERIT_STAGE_WINDOW, INHERIT_STAGE_WINDOW)


def test_stage_window_v1_omega_form_matches_legacy_bytes() -> None:
    """The production per-thread table lowers to the measured per-range bytes."""
    policy = AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1

    for (min_routes, max_routes, threads), expected in STAGE_WINDOW_V1_LEGACY_BYTES.items():
        for routes in (min_routes, (min_routes + max_routes) // 2, max_routes):
            assert policy.select(routes, threads) == expected, (routes, threads)

    entries = {
        (entry.min_routes, entry.max_routes, entry.threads): (entry.w13_window_bytes, entry.w2_window_bytes)
        for entry in policy.cost_model_entries()
    }
    assert entries == STAGE_WINDOW_V1_LEGACY_BYTES


def test_stage_window_v1_coverage_set_unchanged() -> None:
    """Widening coverage disables the full-workload anchor, so freeze the holes."""
    policy = AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1
    covered = {(band[0], band[1], threads) for band in STAGE_WINDOW_V1_PER_THREAD for threads in band[2]}

    for routes in (12, 48, 49, 95, 96, 143, 144, 287, 288, 575, 576, 2040):
        for threads in STAGE_WIDTHS:
            in_band = any(
                min_routes <= routes <= max_routes and (min_routes, max_routes, threads) in covered
                for min_routes, max_routes, _, _, _ in STAGE_WINDOW_V1_PER_THREAD
            )
            inherited = policy.select(routes, threads) == (INHERIT_STAGE_WINDOW, INHERIT_STAGE_WINDOW)
            assert inherited is not in_band, (routes, threads)


def test_stage_window_v2_adds_short_route_band_without_touching_v1() -> None:
    """V2 is V1 plus one band, so nothing at 49 routes or above may move."""
    v1 = AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1
    v2 = AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V2

    assert v2.bands[0] is AMAZON_C5_192C_TP4_F512_SHORT_ROUTE_BAND
    assert v2.bands[1:] == v1.bands

    for routes in (12, 49, 95, 96, 143, 144, 287, 288, 575, 576, 2040):
        for threads in STAGE_WIDTHS:
            assert v2.select(routes, threads) == v1.select(routes, threads), (routes, threads)

    # The band lowers to the per-range budgets the isolated sweep measured.
    expected = {1: (MIB // 4, MIB // 4), 2: (MIB // 2, MIB // 2), 4: (1 * MIB, 1 * MIB), 8: (1 * MIB, 1 * MIB)}
    for threads, windows in expected.items():
        assert v2.select(13, threads) == windows
        assert v2.select(48, threads) == windows
    for threads in (16, 32):
        assert v2.select(24, threads) == (INHERIT_STAGE_WINDOW, INHERIT_STAGE_WINDOW)


def test_stage_window_v3_fills_the_mid_route_band_without_moving_v2_cells() -> None:
    """V3 only adds 49-95 at 1, 2 and 4 threads; every other cell is frozen."""
    v1 = AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1
    v2 = AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V2
    v3 = AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V3

    # The 8-thread cell V1 calibrated is reproduced byte for byte.
    assert v3.select(72, 8) == v1.select(72, 8) == (1 * MIB, 1 * MIB)

    # The three new cells are the isolated optima measured at M=72.
    assert v3.worker_windows(72, 1) == (MIB // 8, MIB // 8)
    assert v3.worker_windows(72, 2) == (MIB // 8, MIB // 8)
    assert v3.worker_windows(72, 4) == (MIB // 16, MIB // 16)
    for threads in (1, 2, 4):
        assert v1.select(72, threads) == (INHERIT_STAGE_WINDOW, INHERIT_STAGE_WINDOW)

    # Widths above 8 stay inherited: the legacy 4 MiB range divided by a wide team
    # already lands near the optimum, so there is much less to gain there.
    for threads in (16, 32):
        assert v3.select(72, threads) == (INHERIT_STAGE_WINDOW, INHERIT_STAGE_WINDOW)

    # Nothing outside the 49-95 band moves.
    for routes in (13, 28, 48, 96, 120, 143, 144, 287, 288, 575, 576, 2040):
        for threads in STAGE_WIDTHS:
            assert v3.select(routes, threads) == v2.select(routes, threads), (routes, threads)


def test_stage_window_v4_splits_the_mid_route_band_at_the_shared_a_threshold() -> None:
    """V4 only moves the 144-287 band's upper part to the larger window."""
    v1 = AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1
    v3 = AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V3
    v4 = AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V4

    assert [(band.min_routes, band.max_routes) for band in v4.bands] == [
        (13, 48),
        (49, 95),
        (96, 143),
        (144, 215),
        (216, 287),
        (288, 575),
    ]

    # Below the seam nothing moves, and 144-215 still carries V1's own values.
    for routes in (13, 28, 48, 49, 72, 95, 96, 120, 143, 144, 180, 215):
        for threads in STAGE_WIDTHS:
            assert v4.select(routes, threads) == v3.select(routes, threads), (routes, threads)
    for routes in (144, 180, 215):
        for threads in STAGE_WIDTHS:
            assert v4.select(routes, threads) == v1.select(routes, threads), (routes, threads)

    # The per-thread shared-A scan is 2*M*H, so it crosses the 2 MiB private L2 at
    # M=256. The measured optimum steps to 0.5 MiB per thread and is
    # width-independent, which is why the new band carries no thread overrides.
    for routes in (216, 240, 256, 287):
        for threads in (1, 2, 4, 8):
            assert v4.worker_windows(routes, threads) == (MIB // 2, MIB // 8), (routes, threads)
        for threads in (16, 32):
            assert v4.select(routes, threads) == (INHERIT_STAGE_WINDOW, INHERIT_STAGE_WINDOW)

    # Above the seam the 288-575 band is untouched, including its 1T override.
    for routes in (288, 400, 575, 576):
        for threads in STAGE_WIDTHS:
            assert v4.select(routes, threads) == v3.select(routes, threads), (routes, threads)
    assert v4.worker_windows(320, 1) == (1 * MIB, MIB // 2)


def test_stage_window_band_rejects_invalid_shapes() -> None:
    valid = {"min_routes": 10, "max_routes": 20, "widths": (1, 2)}

    with pytest.raises(ValueError, match="either positive per-thread windows or literal"):
        StageWindowBand(**valid)
    with pytest.raises(ValueError, match="either positive per-thread windows or literal"):
        StageWindowBand(
            **valid,
            w13_bytes_per_thread=MIB,
            w2_bytes_per_thread=MIB,
            literal_windows=((1, MIB, MIB), (2, MIB, MIB)),
        )
    with pytest.raises(ValueError, match="both the W13 and the W2 window"):
        StageWindowBand(**valid, w13_bytes_per_thread=MIB)
    with pytest.raises(ValueError, match="must target covered widths"):
        StageWindowBand(
            **valid,
            w13_bytes_per_thread=MIB,
            w2_bytes_per_thread=MIB,
            thread_overrides=((4, MIB, MIB),),
        )
    with pytest.raises(ValueError, match="do not support thread overrides"):
        StageWindowBand(
            **valid,
            literal_windows=((1, MIB, MIB), (2, MIB, MIB)),
            thread_overrides=((1, MIB, MIB),),
        )
    with pytest.raises(ValueError, match="cover exactly the band's widths"):
        StageWindowBand(**valid, literal_windows=((1, MIB, MIB),))
    with pytest.raises(ValueError, match="unique within a route band"):
        StageWindowBand(min_routes=10, max_routes=20, widths=(1, 1), literal_windows=((1, MIB, MIB),))
    with pytest.raises(ValueError, match="at least one thread width"):
        StageWindowBand(min_routes=10, max_routes=20, widths=(), literal_windows=())


def test_stage_window_policy_rejects_unlowerable_bands() -> None:
    band = StageWindowBand(
        min_routes=10,
        max_routes=20,
        widths=(1,),
        w13_bytes_per_thread=MIB,
        w2_bytes_per_thread=MIB,
    )

    with pytest.raises(ValueError, match="require hidden_size, intermediate_size and backend_n_tile"):
        StaticStageWindowPolicy(name="missing_geometry", bands=(band,))

    with pytest.raises(ValueError, match="cannot lower w13 for routes 10-20 at 1 threads"):
        StaticStageWindowPolicy(
            name="below_floor",
            bands=(
                StageWindowBand(
                    min_routes=10,
                    max_routes=20,
                    widths=(1,),
                    w13_bytes_per_thread=32 * 1024,
                    w2_bytes_per_thread=32 * 1024,
                ),
            ),
            **STAGE_WINDOW_GEOMETRY,
        )

    with pytest.raises(ValueError, match="route bands must not overlap"):
        StaticStageWindowPolicy(
            name="overlapping",
            bands=(
                StageWindowBand.from_thread_windows(10, 20, ((1, MIB, MIB),)),
                StageWindowBand.from_thread_windows(20, 30, ((1, MIB, MIB),)),
            ),
        )


def test_positive_window_profile_rejects_legacy_split_flag(
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
            w13_split=True,
        )
    )
    payload = json.loads(record.path.read_text(encoding="utf-8"))
    payload["kernel"]["weight_window_bytes"] = 1024 * 1024
    path = tmp_path / "invalid_window_split.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProfileCompatibilityError, match="canonical w13_split=false"):
        ProfileCatalog.from_paths([path])


def test_profile_kernel_policy_identity_is_canonical_stage_ranges(catalog: ProfileCatalog) -> None:
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

    one_range, two_range = catalog.stage_range_pair(query)

    assert one_range.policy.w13_split is False
    assert two_range.policy.w13_split is True
    assert one_range.policy.kernel_policy_key() == ("stage_ranges", 1, 1)
    assert two_range.policy.kernel_policy_key() == ("stage_ranges", 2, 1)


def test_policy_variants_reject_duplicate_stage_range_identity(
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
    one_range, _ = catalog.stage_range_pair(query)
    payload = json.loads(one_range.path.read_text(encoding="utf-8"))
    payload["kernel"].update(
        {
            "weight_window_bytes": 1 << 40,
            "w13_window_ranges": 1,
            "w2_window_ranges": 1,
        }
    )
    duplicate = tmp_path / "duplicate_r1_r1.json"
    duplicate.write_text(json.dumps(payload), encoding="utf-8")
    duplicate_catalog = ProfileCatalog.from_paths([one_range.path, duplicate])

    with pytest.raises(ProfileCompatibilityError, match="duplicate kernel policy profile"):
        duplicate_catalog.policy_variants(query)


def test_window_policy_and_thread_shape_are_selected_jointly(
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
    no_split, split = catalog.stage_range_pair(query)
    window_paths: list[Path] = []
    preferred = {
        1024 * 1024: (tuple([1] * 32), 2_000_000),
        2 * 1024 * 1024: (tuple([2] * 16), 1_000_000),
    }
    for window_bytes, (preferred_shape, preferred_ns) in preferred.items():
        payload = json.loads(no_split.path.read_text(encoding="utf-8"))
        w13, w2 = fused_moe_weight_windows(
            hidden_size=4096,
            intermediate_size=1024,
            n_tile=8,
            target_bytes=window_bytes,
            w13_fallback_ranges=1,
        )
        payload["kernel"].update(
            {
                "w13_split": False,
                "w13_split_chunks": 1,
                "weight_window_bytes": window_bytes,
                "w13_window_ranges": w13.ranges,
                "w2_window_ranges": w2.ranges,
            }
        )
        payload["working_set"].update(
            {
                "weight_window_target_bytes": window_bytes,
                "w13_chunk_bytes_per_expert": w13.max_range_bytes,
                "w2_chunk_bytes_per_expert": w2.max_range_bytes,
                "w13_window_bytes_per_expert": w13.max_range_bytes,
                "w2_window_bytes_per_expert": w2.max_range_bytes,
                "max_weight_stage_bytes_per_expert": max(w13.max_range_bytes, w2.max_range_bytes),
            }
        )
        for entry in payload["entries"]:
            value = preferred_ns if tuple(entry["shape"]) == preferred_shape else 1_000_000_000
            entry["makespan_ns"] = value
            entry["p10_ns"] = value
            entry["p90_ns"] = value
            entry["full_call_median_ns"] = value
            entry["full_call_p10_ns"] = value
            entry["full_call_p90_ns"] = value
        path = tmp_path / f"window_{window_bytes}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        window_paths.append(path)

    policy_catalog = ProfileCatalog.from_paths([no_split.path, split.path, *window_paths])
    records = policy_catalog.policy_variants(query)
    policy_models = [ContentionCostModel(record.path) for record in records]
    result = PolicyAwarePlanner(policy_models, 32).plan([(expert, 192) for expert in range(64)])

    assert len(records) == 4
    assert result["weight_window_bytes"] == 2 * 1024 * 1024
    assert result["shape"] == tuple([2] * 16)
    assert result["active_working_set_bytes"] == 32 * 1024 * 1024
    assert result["window_bytes_per_worker"] == tuple([1024 * 1024] * 16)
    selected_model = next(model for model in policy_models if model.weight_window_bytes == 2 * 1024 * 1024)
    selected_worksets = [workset for _, workset in selected_model._task_phases(192, 2) if workset]
    assert selected_worksets == [2 * 1024 * 1024] * 12

    runtime = PlannedMoE(policy_models, 32)
    spec = runtime.plan_spec_for([(expert, 192) for expert in range(64)])
    assert spec["plan_version"] == 2
    assert spec["operator_options"] == {
        "w13_split": False,
        "weight_window_bytes": 2 * 1024 * 1024,
    }
    bridge = spec["bridge"]
    assert bridge["plan_version"] == 2
    assert bridge["execution_mode"] == "strict"
    assert bridge["task_preferred_threads"] == bridge["task_threads"]
    assert bridge["task_min_threads"] == bridge["task_threads"]
    assert bridge["task_max_threads"] == bridge["task_threads"]
    assert bridge["task_allowed_thread_offsets"] == list(range(len(bridge["task_threads"]) + 1))
    assert bridge["task_allowed_threads"] == bridge["task_threads"]
    assert bridge["task_placement_modes"] == [0] * len(bridge["task_threads"])
    assert bridge["task_stage_ids"] == [0] * len(bridge["task_threads"])
    assert bridge["task_resize_points"] == [0] * len(bridge["task_threads"])
    assert bridge["task_range_granularities"] == [0] * len(bridge["task_threads"])
    assert bridge["task_w13_window_bytes"] == [-1] * len(bridge["task_threads"])
    assert bridge["task_w2_window_bytes"] == [-1] * len(bridge["task_threads"])
    assert bridge["task_w13_ranges"] == [8] * len(bridge["task_threads"])
    assert bridge["task_w2_ranges"] == [4] * len(bridge["task_threads"])
    cached_spec = runtime.plan_spec_for([(expert, 192) for expert in range(64)])
    assert cached_spec["operator_options"] == spec["operator_options"]
    assert cached_spec["bridge"] == bridge
    assert runtime.last["cache_hit"] is True
    evaluator = ParallelLayerEvaluator(
        policy_catalog,
        HierarchicalTopology(2, 1, 60e9, 20e9, 1e-6),
        hidden_size=4096,
        full_intermediate_size=2048,
        global_experts=64,
        cores_per_rank=32,
    )
    tp = evaluator.evaluate_tp(2048, 6)
    assert {rank.weight_window_bytes for rank in tp.rank_compute} == {2 * 1024 * 1024}


def test_tail_pool_bridge_relinks_fixed_lane_dependencies(
    catalog: ProfileCatalog,
) -> None:
    _, model = models(catalog, "tp", 1024, 64)
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


def test_static_stage_window_policy_is_lowered_per_task(
    catalog: ProfileCatalog,
) -> None:
    _, model = models(catalog, "tp", 1024, 64)
    baseline = IntervalPlanner(model, 32, native_cold_planner=False)
    planner = IntervalPlanner(
        model,
        32,
        native_cold_planner=False,
        task_stage_window_policy=AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1,
    )
    tasks = [
        (0, 48, 0, 8, []),
        (1, 96, 8, 8, []),
        (2, 192, 16, 8, []),
        (3, 384, 24, 8, []),
    ]

    bridge = planner.to_async_bridge(tasks)

    assert bridge["task_w13_ranges"] == [2, 16, 16, 4]
    assert bridge["task_w2_ranges"] == [1, 16, 16, 8]
    assert bridge["task_w13_window_bytes"] == [-1] * 4
    assert bridge["task_w2_window_bytes"] == [-1] * 4
    assert bridge["task_range_granularities"] == [0, 0, 0, 0]

    sliced_bridge = planner.to_async_bridge(
        [
            (0, 96, 0, 8, []),
            (0, 96, 8, 8, []),
            (1, 192, 16, 8, []),
        ]
    )
    assert sliced_bridge["task_range_granularities"] == [96, 96, 0]

    pooled_bridge = planner.to_tail_pool_bridge(
        tasks,
        pool_threads=2,
        max_pooled_routes=96,
    )
    assert pooled_bridge["task_threads"] == [2, 2, 8, 8]
    assert pooled_bridge["task_w13_ranges"] == [2, 128, 16, 4]
    assert pooled_bridge["task_w2_ranges"] == [1, 32, 16, 8]
    assert planner.shapes == baseline.shapes
    assert planner.model is not model
    assert planner.model.T_iso(192, 8) == model.T_iso(192, 8)
    assert planner.model._task_stage_geometry(192, 8) == (
        16,
        1024 * 1024,
        16,
        512 * 1024,
    )
    concurrent = [(192, 8, []), (192, 8, [])]
    assert planner.model.dag_makespan(concurrent) < model.dag_makespan(concurrent)
    assert planner.model.can_use_full_workload_anchor(48, (16, 16))
    assert not planner.model.can_use_full_workload_anchor(96, (8,) * 4)
    assert len(planner.model.native_interval_planner_payload()["task_stage_windows"]) == 13


def test_elastic_w2_bridge_lowers_local_cohorts_and_stage_widths(
    catalog: ProfileCatalog,
) -> None:
    _, model = models(catalog, "tp", 1024, 64)
    planner = IntervalPlanner(
        model,
        32,
        native_cold_planner=False,
        task_stage_window_policy=AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1,
    )
    tasks_8t = [
        (0, 192, 0, 8, []),
        (1, 192, 8, 8, []),
        (2, 192, 16, 8, []),
        (3, 192, 24, 8, []),
    ]

    bridge_8t = planner.to_elastic_w2_bridge(
        tasks_8t,
        width_transitions={8: 16},
        numa_node=0,
        resize_timeout_ns=5000,
    )

    assert bridge_8t["execution_mode"] == "elastic"
    assert bridge_8t["task_threads"] == [8, 8, 8, 8]
    assert bridge_8t["task_preferred_threads"] == [16, 16, 16, 16]
    assert bridge_8t["task_allowed_thread_offsets"] == [0, 2, 4, 6, 8]
    assert bridge_8t["task_allowed_threads"] == [8, 16] * 4
    assert bridge_8t["task_resize_points"] == [1] * 4
    assert bridge_8t["task_resize_timeout_ns"] == [5000] * 4
    assert bridge_8t["task_numa_nodes"] == [0] * 4
    assert bridge_8t["task_preferred_core_begins"] == [0, 0, 16, 16]
    assert bridge_8t["task_w13_ranges"] == [16] * 4
    assert bridge_8t["task_w2_ranges"] == [1] * 4

    migrated_bridge = planner.to_elastic_w2_bridge(
        tasks_8t,
        width_transitions={8: 16},
        numa_node=0,
        resize_timeout_ns=100_000,
        resizable_task_ids=[0, 1],
        task_preferred_core_begins={0: 0, 1: 16},
    )
    assert migrated_bridge["task_preferred_threads"] == [16, 16, 8, 8]
    assert migrated_bridge["task_resize_points"] == [1, 1, 0, 0]
    assert migrated_bridge["task_preferred_core_begins"] == [0, 16, -1, -1]
    assert migrated_bridge["task_resize_timeout_ns"] == [100_000, 100_000, 0, 0]

    tasks_2t = [(expert, 48, expert * 2, 2, []) for expert in range(4)]
    bridge_2t = planner.to_elastic_w2_bridge(
        tasks_2t,
        width_transitions={2: 8},
        numa_node=0,
    )
    assert bridge_2t["task_preferred_threads"] == [8] * 4
    assert bridge_2t["task_resize_points"] == [1] * 4
    assert bridge_2t["task_resize_timeout_ns"] == [0] * 4


def test_default_stage_window_policy_requires_exact_profile(
    catalog: ProfileCatalog,
) -> None:
    _, model = models(catalog, "tp", 1024, 64)
    assert model.policy is not None
    cpu_ids_by_rank = (tuple(range(96)), tuple(range(96, 192)))
    matching = replace(
        model.policy,
        mode="tp",
        degree=4,
        hidden_size=4096,
        intermediate_size=512,
        global_experts=256,
        local_experts=256,
        backend="sve",
        backend_n_tile=8,
        sve_implementation="jit",
        m_tail_policy="xbyak_exact_m",
        activation="silu",
        dtype="bf16",
        w13_split=True,
        w13_split_chunks=2,
        weight_window_bytes=0,
        w13_window_ranges=2,
        w2_window_ranges=1,
        measurement_experts=256,
        cores_per_rank=96,
        concurrent_ranks=2,
        llc_bytes_per_rank=96 * 1024 * 1024,
        numa_nodes=(0, 1),
        cpu_ids_by_rank=cpu_ids_by_rank,
    )

    assert (
        default_task_stage_window_policy(
            matching,
            num_cores=96,
            cpu_ids=cpu_ids_by_rank[0],
        )
        is AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V4
    )
    assert (
        default_task_stage_window_policy(
            matching,
            num_cores=96,
            cpu_ids=cpu_ids_by_rank[1],
        )
        is AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V4
    )
    assert (
        default_task_stage_window_policy(
            replace(matching, w13_split=False),
            num_cores=96,
            cpu_ids=cpu_ids_by_rank[0],
        )
        is None
    )
    assert (
        default_task_stage_window_policy(
            matching,
            num_cores=96,
            cpu_ids=tuple(range(1, 97)),
        )
        is None
    )


def test_planned_moe_applies_default_stage_windows_per_profile(
    monkeypatch: pytest.MonkeyPatch,
    catalog: ProfileCatalog,
) -> None:
    no_split, split = models(catalog, "tp", 1024, 64)

    def select_default(profile, *, num_cores, cpu_ids):
        del num_cores, cpu_ids
        if profile is not None and profile.w13_split:
            return AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1
        return None

    monkeypatch.setattr(planned_moe_module, "default_task_stage_window_policy", select_default)
    runtime = PlannedMoE((no_split, split), 32, cpu_ids=tuple(range(32)))

    assert runtime.task_stage_window_policies == (
        None,
        AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1,
    )
    tasks = [(0, 96, 0, 8, [])]
    assert runtime.interval_planners[0].to_async_bridge(tasks)["task_w13_ranges"] == [1]
    assert runtime.interval_planners[1].to_async_bridge(tasks)["task_w13_ranges"] == [16]

    disabled = PlannedMoE(
        (no_split, split),
        32,
        cpu_ids=tuple(range(32)),
        use_default_stage_window_policy=False,
    )
    assert disabled.task_stage_window_policies == (None, None)


def test_m12_tail_composition(catalog: ProfileCatalog) -> None:
    _, model = models(catalog, "tp", 1024, 64)
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
            w13_split=True,
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
                w13_split=True,
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
            w13_split=True,
        )
    )
    assert selected.path == path

    model = ContentionCostModel(path, iso_mode="table")
    assert model.m12_effective_rows(3) == 3
    assert model.m12_effective_rows(7) == 7
    assert model.m12_effective_rows(11) == 11
    assert model.m12_effective_rows(23) == 23
    assert model.T_iso(3, 4) != model.T_iso(4, 4)


def test_checked_in_xbyak_profiles_cover_exact_m_range_endpoints() -> None:
    paths = xbyak_profile_paths()
    assert len(paths) >= 4
    exact_catalog = ProfileCatalog.from_paths(paths)

    pairs: dict[tuple[object, ...], dict[tuple[int, int], Path]] = {}
    for record in exact_catalog.records:
        payload = record.payload
        policy = record.policy
        assert policy.sve_implementation == "jit"
        assert policy.m_tail_policy == "xbyak_exact_m"
        assert payload["kernel"]["xbyak_aarch64_commit"] == XBYAK_AARCH64_COMMIT
        assert set(range(1, 13)).issubset(map(int, payload["isolated_routes"]))
        assert set(range(1, 13)).issubset(map(int, payload["contention_routes"]))
        assert payload["kernel"]["source_sha256"]
        assert payload["kernel"]["extension_sha256"]
        if policy.weight_window_bytes == 0:
            range_variants = pairs.setdefault(policy.key_without_kernel_policy(), {})
            range_key = (policy.w13_window_ranges, policy.w2_window_ranges)
            assert range_key not in range_variants
            range_variants[range_key] = record.path

        model = ContentionCostModel(
            record.path,
            expected_policy=ProfileQuery(
                sve_implementation="jit",
                m_tail_policy="xbyak_exact_m",
                w13_window_ranges=policy.w13_window_ranges,
                w2_window_ranges=policy.w2_window_ranges,
            ),
        )
        assert [model.m12_effective_rows(routes) for routes in range(1, 13)] == list(range(1, 13))

    assert len(pairs) >= 2
    assert all(set(range_variants) == {(1, 1), (2, 1)} for range_variants in pairs.values())
    for range_variants in pairs.values():
        pair_catalog = ProfileCatalog.from_paths(range_variants.values())
        pair_catalog.stage_range_pair(
            ProfileQuery(
                sve_implementation="jit",
                m_tail_policy="xbyak_exact_m",
            )
        )


def test_parallel_evaluator_auto_accepts_available_range_variants(
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
    asm_no_split, asm_split = catalog.stage_range_pair(query)

    jit_paths: list[Path] = []
    for record in (asm_no_split, asm_split):
        payload = json.loads(record.path.read_text(encoding="utf-8"))
        payload["kernel"]["sve_implementation"] = "jit"
        payload["kernel"]["m_tail_policy"] = "xbyak_exact_m"
        path = tmp_path / f"jit_{int(record.policy.w13_split)}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        jit_paths.append(path)

    topology = HierarchicalTopology(2, 1, 60e9, 20e9, 1e-6)

    incomplete = ProfileCatalog.from_paths([asm_no_split.path, asm_split.path, jit_paths[1]])
    partial = ParallelLayerEvaluator(
        incomplete,
        topology,
        hidden_size=4096,
        full_intermediate_size=2048,
        global_experts=64,
        cores_per_rank=32,
    )._models("tp", 1024, 64)
    assert {model.policy.sve_implementation for model in partial} == {"jit"}
    assert [model.policy.kernel_policy_key() for model in partial] == [("stage_ranges", 2, 1)]

    complete = ProfileCatalog.from_paths([asm_no_split.path, asm_split.path, *jit_paths])
    selected = ParallelLayerEvaluator(
        complete,
        topology,
        hidden_size=4096,
        full_intermediate_size=2048,
        global_experts=64,
        cores_per_rank=32,
    )._models("tp", 1024, 64)
    assert {model.policy.sve_implementation for model in selected} == {"jit"}
    assert {model.policy.kernel_policy_key() for model in selected} == {
        ("stage_ranges", 1, 1),
        ("stage_ranges", 2, 1),
    }


def test_exact_shape_and_stage_working_sets(catalog: ProfileCatalog) -> None:
    no_split, split = models(catalog, "ep", 2048, 32)
    assert split.supports_shape((32,))
    assert not split.supports_shape((24, 8))
    with pytest.raises(ProfileCompatibilityError, match="was not measured"):
        split.profiled_group_time(192, (24, 8))

    split_worksets = [workset for _, workset in split._task_phases(192, 16)]
    no_split_worksets = [workset for _, workset in no_split._task_phases(192, 16)]
    assert [value for value in split_worksets if value] == [
        16 * 1024 * 1024,
        16 * 1024 * 1024,
        16 * 1024 * 1024,
    ]
    assert [value for value in no_split_worksets if value] == [
        32 * 1024 * 1024,
        16 * 1024 * 1024,
    ]
    tasks = [(192, 16, []), (192, 16, [])]
    assert no_split.dag_makespan(tasks) != pytest.approx(no_split.flat_dag_makespan(tasks))
    finish_times = no_split.dag_task_finish_times(tasks)
    assert finish_times[0] == pytest.approx(finish_times[1])
    assert max(finish_times) == pytest.approx(no_split.dag_makespan(tasks))


def test_joint_planner_and_physical_cpu_mapping(catalog: ProfileCatalog) -> None:
    tp_models = models(catalog, "tp", 1024, 64)
    assert all(model.has_full_workload_anchors for model in tp_models)
    tp = PolicyAwarePlanner(tp_models, 32, cpu_ids=range(32, 64)).plan([(expert, 192) for expert in range(64)])
    assert tp["w13_split"] is True
    assert tp["shape"] == (8, 8, 8, 8)
    assert tp["active_working_set_bytes"] == 32 * 1024 * 1024
    assert tp["bridge"]["thread_cpu_ids"] == list(range(32, 64))

    split_model = tp_models[1]
    split_planner = PolicyAwarePlanner([split_model], 32, cpu_ids=range(32, 64)).planners[0]
    anchored_ms, _ = split_planner.score_shape([(expert, 192) for expert in range(64)], (8, 8, 8, 8))
    assert anchored_ms == split_model.profiled_full_call_time(192, (8, 8, 8, 8))

    ep_models = models(catalog, "ep", 2048, 32)
    ep = PolicyAwarePlanner(ep_models, 32).plan([(expert, 192) for expert in range(32)])
    assert ep["w13_split"] is True
    assert ep["shape"] == (32,)


def test_policy_aware_cache_and_richer_signature(catalog: ProfileCatalog) -> None:
    first = [(0, 10), (1, 6), (2, 2), (3, 2)]
    second = [(0, 10), (1, 4), (2, 4), (3, 2)]
    assert signature(first) != signature(second)

    planner = PlannedMoE(models(catalog, "tp", 1024, 64), 32)
    counts = [(expert, 192) for expert in range(64)]
    spec = planner.plan_spec_for(counts)
    assert spec["operator_options"] == {
        "w13_split": True,
        "weight_window_bytes": 0,
    }
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

    policy_planner = PolicyAwarePlanner(
        models(catalog, "tp", 1024, 64),
        32,
    )
    result = policy_planner.plan(workload.experts)
    strict_result = policy_planner.plan(workload.experts, dynamic_tail_pool=False)
    assert math.isfinite(result["makespan_ns"])
    assert result["makespan_ns"] > 0
    assert result["makespan_ns"] <= strict_result["makespan_ns"]
    assert len(result["tasks"]) == workload.observed_active_experts
    assert sum(task[1] for task in result["tasks"]) == workload.routes
    assert len(result["bridge"]["task_expert_ids"]) == len(result["tasks"])
    assert len(result["policy_ranking"]) == 2

    selected = next(
        planner
        for planner in policy_planner.planners
        if planner.model.policy is not None and planner.model.policy.w13_split == strict_result["w13_split"]
    )
    rescored_ns, rescored_tasks = selected.score_shape(
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
    assert tp.rank_compute[0].shape == (8, 8, 8, 8)
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
    dual_records = catalog.stage_range_pair(query)
    paths = [record.path for record in dual_records]
    for record in dual_records:
        payload = json.loads(record.path.read_text(encoding="utf-8"))
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
        single_path = tmp_path / f"single_{record.path.name}"
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
