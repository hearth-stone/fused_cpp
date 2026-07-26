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

from interval_planner import IntervalPlanner, PolicyAwarePlanner  # noqa: E402
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
    AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1,
    default_task_stage_window_policy,
)
from tp_vs_ep_model import HierarchicalTopology, ParallelLayerEvaluator  # noqa: E402
from weight_window import fused_moe_weight_windows  # noqa: E402
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
    no_split, split = catalog.split_pair(query)
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
    no_split, split = catalog.split_pair(query)
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

    assert bridge["task_w13_window_bytes"] == [-1, 1024 * 1024, 1024 * 1024, 4 * 1024 * 1024]
    assert bridge["task_w2_window_bytes"] == [-1, 512 * 1024, 512 * 1024, 1024 * 1024]

    pooled_bridge = planner.to_tail_pool_bridge(
        tasks,
        pool_threads=2,
        max_pooled_routes=96,
    )
    assert pooled_bridge["task_threads"] == [2, 2, 8, 8]
    assert pooled_bridge["task_w13_window_bytes"] == [-1, 128 * 1024, 1024 * 1024, 4 * 1024 * 1024]
    assert pooled_bridge["task_w2_window_bytes"] == [-1, 256 * 1024, 512 * 1024, 1024 * 1024]
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
        is AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1
    )
    assert (
        default_task_stage_window_policy(
            matching,
            num_cores=96,
            cpu_ids=cpu_ids_by_rank[1],
        )
        is AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1
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
    assert runtime.interval_planners[0].to_async_bridge(tasks)["task_w13_window_bytes"] == [-1]
    assert runtime.interval_planners[1].to_async_bridge(tasks)["task_w13_window_bytes"] == [1024 * 1024]

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


def test_checked_in_xbyak_profiles_are_complete_exact_m_pairs() -> None:
    paths = xbyak_profile_paths()
    assert len(paths) >= 4
    exact_catalog = ProfileCatalog.from_paths(paths)

    pairs: dict[tuple[object, ...], dict[bool, Path]] = {}
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
            split_variants = pairs.setdefault(policy.key_without_kernel_policy(), {})
            assert policy.w13_split not in split_variants
            split_variants[policy.w13_split] = record.path

        model = ContentionCostModel(
            record.path,
            expected_policy=ProfileQuery(
                sve_implementation="jit",
                m_tail_policy="xbyak_exact_m",
                w13_split=policy.w13_split,
            ),
        )
        assert [model.m12_effective_rows(routes) for routes in range(1, 13)] == list(range(1, 13))

    assert len(pairs) >= 2
    assert all(set(split_variants) == {False, True} for split_variants in pairs.values())
    for split_variants in pairs.values():
        pair_catalog = ProfileCatalog.from_paths(split_variants.values())
        pair_catalog.split_pair(
            ProfileQuery(
                sve_implementation="jit",
                m_tail_policy="xbyak_exact_m",
            )
        )


def test_parallel_evaluator_auto_requires_a_complete_variant_pair(
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
    asm_no_split, asm_split = catalog.split_pair(query)

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
    fallback = ParallelLayerEvaluator(
        incomplete,
        topology,
        hidden_size=4096,
        full_intermediate_size=2048,
        global_experts=64,
        cores_per_rank=32,
    )._models("tp", 1024, 64)
    assert {model.policy.sve_implementation for model in fallback} == {"asm"}

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
