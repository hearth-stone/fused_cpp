from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "cpu_moe_schedule_optimization" / "cost_model"),
    str(ROOT / "cpu_moe_schedule_optimization" / "planners"),
]

from interval_planner import IntervalPlanner  # noqa: E402
from model_lns import Lane, ModelLnsSearch, lanes_from_planner_tasks, pack_lanes, placed_from_lanes  # noqa: E402
from probe_event_model import ProbeEventModel  # noqa: E402
from stage_window_policy import ARM_CODEX_NUMA3_80C_TP4_F512_N16_V3 as TABLE  # noqa: E402

CALIBRATION = ROOT / "bench_assets/moe_paper/arm_codex_numa3_80c_jemalloc/probe_event_v10_20260919.json"
V11 = ROOT / "bench_assets/moe_paper/arm_codex_numa3_80c_jemalloc/probe_event_v11_20260920.json"
CPUS = tuple(range(240, 320))
ANALYTIC = dict(hidden_size=4096, intermediate_size=512, global_experts=256, local_experts=256, mode="tp",
                degree=4, concurrent_ranks=1, down_output_element_bytes=4)


@pytest.fixture(scope="module")
def model() -> ProbeEventModel:
    return ProbeEventModel.from_calibration(CALIBRATION, **ANALYTIC)


@pytest.fixture(scope="module")
def windowed() -> ProbeEventModel:
    return ProbeEventModel.from_calibration(CALIBRATION, window_policy=TABLE, **ANALYTIC)


def _reference_plan():
    return ([(384, 4, CPUS[0:4], ()), (96, 4, CPUS[0:4], (0,)), (1024, 8, CPUS[4:12], ()),
             (24, 2, CPUS[12:14], ()), (48, 2, CPUS[12:14], (3,))]
            + [(384, 4, CPUS[16 + 4 * k : 20 + 4 * k], ()) for k in range(14)])


def test_matches_frozen_lab_v10_values(model, windowed) -> None:
    # Values of the frozen Lab implementation (tmp/v10_model_20260919/v10.py) on the same plan.
    assert model.dag_makespan_placed(_reference_plan()) == pytest.approx(20144693.91842572, rel=1e-12)
    assert windowed.dag_makespan_placed(_reference_plan()) == pytest.approx(20128665.200195182, rel=1e-12)
    assert model.T_iso(384, 4) == pytest.approx(14631500.259939512, rel=1e-12)


def test_single_task_runs_at_isolated_time(model) -> None:
    for routes, threads in ((12, 2), (384, 4), (2048, 32)):
        placed = [(routes, threads, CPUS[:threads], ())]
        assert model.dag_makespan_placed(placed) == pytest.approx(model.T_iso(routes, threads), rel=1e-12)
        assert model.call_time_placed(placed) == pytest.approx(model.T_iso(routes, threads) + model.t_over_ns)


def test_serial_lane_is_sum_and_concurrency_only_dilates(model) -> None:
    chain = [(384, 4, CPUS[0:4], ()), (96, 4, CPUS[0:4], (0,))]
    assert model.dag_makespan_placed(chain) == pytest.approx(model.T_iso(384, 4) + model.T_iso(96, 4), rel=1e-12)
    crowd = [(48, 4, CPUS[4 * k : 4 * k + 4], ()) for k in range(20)]
    assert model.dag_makespan_placed(crowd) > 1.2 * model.T_iso(48, 4)


def test_isolated_window_gain_is_g0(model, windowed) -> None:
    routes, threads = 256, 4
    assert any(TABLE.select(routes, threads))
    phases = sum(tau for tau, _ in model.phase_table(routes, threads))
    placed = [(routes, threads, CPUS[:threads], ())]
    expected = model.overhead_ns(threads) + (1.0 - model.g0) * phases
    assert windowed.dag_makespan_placed(placed) == pytest.approx(expected, rel=1e-12)


def test_planner_facing_delegation(model) -> None:
    assert model.native_interval_planner_payload is None
    assert model.native_quick_planner_payload is None
    assert model.quick_homogeneous_scale(8, 80, 80) == 1.0
    assert (8,) * 10 in model.candidate_shapes(80)
    explained = model.explain_dag_placed(_reference_plan())
    assert explained["makespan_ns"] == pytest.approx(model.dag_makespan_placed(_reference_plan()))
    assert all(event["duration_ns"] >= 0.0 for event in explained["events"])


def test_quick_plan_uses_probe_model(windowed) -> None:
    planner = IntervalPlanner(windowed, 80, cpu_ids=CPUS, native_cold_planner=False, stage_window_policy=TABLE)
    experts = [(e, r) for e, r in enumerate([900, 400, 300] + [60] * 60 + [12] * 40)]
    plan = planner.plan_quick(experts)
    assert plan["planner_backend"] == "python_quick"
    assert sorted(expert for expert, *_ in plan["tasks"]) == [e for e, _ in experts]


def test_pack_lanes_respects_llc_domains() -> None:
    assert pack_lanes([Lane(32, ()), Lane(32, ()), Lane(16, ())], (40, 40)) is None
    begins = pack_lanes([Lane(32, ()), Lane(8, ()), Lane(16, ()), Lane(16, ()), Lane(8, ())], (40, 40))
    assert begins is not None
    spans = sorted((b, b + w) for b, w in zip(begins, (32, 8, 16, 16, 8)))
    assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(spans, spans[1:]))
    assert all(start // 40 == (end - 1) // 40 for start, end in spans)


def test_model_lns_never_returns_worse_than_start(model) -> None:
    experts = [(e, r) for e, r in enumerate([700, 300, 200, 90, 80, 60, 40, 30, 20, 12, 12, 6])]
    cpus = CPUS[:16]
    search = ModelLnsSearch(model, cpus, widths=(2, 4, 8), domain_cores=(16,), batch=12, patience=3, seed=1)
    start = search.lpt(experts, (4, 4, 4, 4))
    result = search.search([start], time_limit_s=2.0, max_evaluations=300)
    assert result.makespan_ns <= result.start_makespan_ns
    assert sorted(e for lane in result.lanes for e in lane.experts) == sorted(experts)
    assert sum(lane.width for lane in result.lanes) <= 16
    begins = pack_lanes(result.lanes, (16,))
    assert model.dag_makespan_placed(placed_from_lanes(result.lanes, begins, cpus)) == pytest.approx(result.makespan_ns)


def test_lanes_from_planner_tasks_round_trip() -> None:
    tasks = [(3, 90, 0, 4, []), (5, 20, 0, 4, [0]), (7, 400, 4, 8, [])]
    lanes = lanes_from_planner_tasks(tasks)
    assert lanes == (Lane(4, ((3, 90), (5, 20))), Lane(8, ((7, 400),)))


def test_hot_wide_planner_template_and_order(windowed) -> None:
    from hot_wide_planner import HotWidePlanner

    planner = HotWidePlanner(windowed, window_policy=TABLE)
    assert all(sum(shape) <= 80 for shape in planner.shapes)
    experts = [(e, r) for e, r in enumerate([1900, 1500, 900, 700] + [120] * 20 + [40] * 60 + [8] * 100)]
    plan = planner.plan(experts)
    assert sorted(e for lane in plan.lanes for e in lane.experts) == sorted(experts)
    assert max(lane.width for lane in plan.lanes) >= 8  # the hot experts get a wide lane
    for lane in plan.lanes:
        routes = [r for _, r in lane.experts]
        assert routes[0] == max(routes) and routes[1:] == sorted(routes[1:])
    tasks = plan.tasks()
    spans = sorted({(core, threads) for _, _, core, threads, _ in tasks})
    assert all(start // 40 == (start + width - 1) // 40 for start, width in spans)
    assert windowed.dag_makespan_placed([(r, t, CPUS[c:c + t], d) for _, r, c, t, d in tasks]) > 0.0


def test_window_table_resolves_only_on_its_machine() -> None:
    from stage_window_policy import ARM_CODEX_NUMA3_80C_TP4_F512_N16_V3, default_stage_window_policy

    shape = dict(hidden_size=4096, intermediate_size=512, backend_n_tile=16)
    assert default_stage_window_policy(**shape) is None
    assert default_stage_window_policy(**shape, machine_id="some_other_machine") is None
    resolved = default_stage_window_policy(**shape, machine_id=ARM_CODEX_NUMA3_80C_TP4_F512_N16_V3.machine_ids[0])
    assert resolved is ARM_CODEX_NUMA3_80C_TP4_F512_N16_V3


def test_planned_moe_hot_wide_mode_and_cache(windowed) -> None:
    from planned_moe import PlannedMoE

    experts = [(e, r) for e, r in enumerate([1800, 1200, 700] + [90] * 40 + [30] * 80 + [5] * 60)]
    planner = PlannedMoE(windowed, num_cores=80, cpu_ids=CPUS, search_mode="hot_wide", cache_plans=True)
    first = planner.plan_spec_for(experts)
    second = planner.plan_spec_for(experts)  # served from the shape cache, rebuilt from the template
    assert first["bridge"]["task_expert_ids"] == second["bridge"]["task_expert_ids"]
    assert first["bridge"]["task_core_begins"] == second["bridge"]["task_core_begins"]
    assert first["bridge"]["task_threads"] == second["bridge"]["task_threads"]
    assert sorted(first["bridge"]["task_expert_ids"]) == [e for e, _ in experts]
    assert max(first["bridge"]["task_threads"]) >= 8 and min(first["bridge"]["task_threads"]) == 4
    assert any(first["bridge"]["task_w13_window_tiles"]), "the machine's window table must be applied"


def test_probe_model_t_iso_cache_round_trip(model) -> None:
    value = model.T_iso(384, 4)
    exported = model.export_t_iso_cache()
    assert exported[(384, 4)] == value
    fresh = ProbeEventModel.from_calibration(CALIBRATION, **ANALYTIC)
    assert fresh.import_t_iso_cache({(384, 4): value, (1, 3): 5.0, (7, 4): -1.0}) == 1
    assert fresh.T_iso(384, 4) == value
    assert fresh.t_iso_cache_identity()["model"] == model.t_iso_cache_identity()["model"]


def test_route_slicing_move_produces_equal_ranges(model) -> None:
    """A sliced expert must lower to equal fixed ranges, which is what the runtime accepts."""
    experts = [(0, 2048), (1, 600), (2, 400), (3, 200), (4, 120), (5, 64), (6, 32), (7, 17)]
    cpus = CPUS[:32]
    search = ModelLnsSearch(model, cpus, widths=(4, 8, 16), domain_cores=(32,), batch=24, seed=3,
                            allow_route_slicing=True)
    start = search.lpt(experts, (16, 8, 4, 4))
    seen = []
    for _ in range(40):
        for kind, plan in search.neighbors(start, []):
            if kind == "slice":
                seen.append(plan)
        if seen:
            break
    assert seen, "the slice move never fired on a plan with a dominant expert"
    plan = seen[0]
    counts: dict[int, list[int]] = {}
    for lane in plan:
        for expert, routes in lane.experts:
            counts.setdefault(expert, []).append(routes)
    sliced = {expert: routes for expert, routes in counts.items() if len(routes) > 1}
    assert sliced, "a slice move must leave one expert on two lanes"
    for expert, routes in sliced.items():
        assert len(set(routes)) == 1, "the runtime requires equal route ranges per slice"
        assert sum(routes) == dict(experts)[expert]
    begins = pack_lanes(plan, (32,))
    assert begins is not None
    planner = IntervalPlanner(model, 32, cpu_ids=cpus, native_cold_planner=False, stage_window_policy=None)
    from model_lns import planner_tasks

    bridge = planner.to_async_bridge(planner_tasks(plan, begins))
    for expert, routes in sliced.items():
        granularities = [g for e, g in zip(bridge["task_expert_ids"], bridge["task_range_granularities"]) if e == expert]
        assert granularities and len(set(granularities)) == 1 and granularities[0] == routes[0]


def test_planner_keeps_pool_widths_inside_the_trusted_set() -> None:
    """One-thread pools were chosen because the curves clamp; they ran 3.2-4.3x their estimate."""
    windowed = ProbeEventModel.from_calibration(V11, window_policy=TABLE, **ANALYTIC)
    assert windowed.calibrated_widths == (2, 4, 8, 16, 32)
    assert windowed.reliable_widths == (4, 8, 16, 32)  # two-thread lanes are refuted by whole plans
    assert not windowed.supports_width(1) and not windowed.supports_width(2)
    assert windowed.supports_width(8)
    experts = [(e, r) for e, r in enumerate([1600, 900, 500] + [80] * 30 + [40] * 40 + [6] * 60)]
    planner = IntervalPlanner(windowed, 80, cpu_ids=CPUS, native_cold_planner=False, stage_window_policy=TABLE)
    strict = sorted((planner._candidate(experts, shape) for shape in planner.shapes),
                    key=lambda candidate: candidate["makespan_ns"])[:2]
    pooled = planner._tail_pool_candidates(experts, planner._tail_pool_head_candidates(strict),
                                           max_pooled_routes=12, forced_pool_threads=None)
    assert pooled, "the tail-pool family must still be reachable on trusted widths"
    assert {candidate["tail_pool_threads"] for candidate in pooled} <= set(windowed.reliable_widths)
    forced = planner._tail_pool_candidates(experts, strict[:1], max_pooled_routes=12, forced_pool_threads=1)
    assert forced, "an explicit request for an untrusted width is still honoured"


def test_model_lns_defaults_to_the_trusted_widths() -> None:
    model = ProbeEventModel.from_calibration(V11, **ANALYTIC)
    assert model.reliable_widths == (4, 8, 16, 32)
    assert ModelLnsSearch(model, CPUS).widths == model.reliable_widths
    assert ModelLnsSearch(model, CPUS, widths=(2, 4)).widths == (2, 4)  # an explicit set still wins
