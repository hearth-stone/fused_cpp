from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
COST_MODEL = ROOT / "cpu_moe_schedule_optimization" / "cost_model"
PLANNERS = ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path[:0] = [str(COST_MODEL), str(PLANNERS)]

from interval_planner import PolicyAwarePlanner  # noqa: E402
from iso_formula import IsoFormula, fit_from_measurements  # noqa: E402
from phase_model import ContentionCostModel  # noqa: E402
from planned_moe import PlannedMoE, signature  # noqa: E402
from profile_catalog import (  # noqa: E402
    ProfileCatalog,
    ProfileCompatibilityError,
    ProfileQuery,
)
from simulate_schedules import PRESETS  # noqa: E402
from tp_vs_ep_model import HierarchicalTopology, ParallelLayerEvaluator  # noqa: E402
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

    with pytest.raises(ProfileCompatibilityError, match="no exact profile"):
        catalog.select(
            ProfileQuery(
                mode="tp",
                degree=4,
                hidden_size=4096,
                intermediate_size=512,
            )
        )


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
        split_variants = pairs.setdefault(policy.key_without_split(), {})
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
    assert spec["operator_options"] == {"w13_split": True}
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
    assert math.isfinite(result["makespan_ns"])
    assert result["makespan_ns"] > 0
    assert len(result["tasks"]) == workload.observed_active_experts
    assert sum(task[1] for task in result["tasks"]) == workload.routes
    assert len(result["bridge"]["task_expert_ids"]) == len(result["tasks"])
    assert len(result["policy_ranking"]) == 2

    selected = next(
        planner
        for planner in policy_planner.planners
        if planner.model.policy is not None and planner.model.policy.w13_split == result["w13_split"]
    )
    rescored_ns, rescored_tasks = selected.score_shape(
        workload.experts,
        result["shape"],
    )
    assert rescored_ns == pytest.approx(result["makespan_ns"])
    assert rescored_tasks == result["tasks"]


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
