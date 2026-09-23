from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
COST_MODEL = ROOT / "cpu_moe_schedule_optimization" / "cost_model"
PLANNERS = ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(COST_MODEL), str(PLANNERS)]

from analytic_model import AnalyticMoeCostModel  # noqa: E402
from interval_planner import IntervalPlanner  # noqa: E402
from planned_moe import PlannedMoE  # noqa: E402

PROFILE = COST_MODEL / "profiles" / "analytic_machine_amazon_c5_192c_numa0_sve_jit_hot_gemm_20260802.json"
CORES = 96
# Five compute-bound long experts and many bandwidth-bound short ones.
BIMODAL = [(expert, 2040) for expert in range(5)] + [(expert, 12) for expert in range(5, 179)]


def _model(partition_widths=()) -> AnalyticMoeCostModel:
    return AnalyticMoeCostModel(
        PROFILE,
        hidden_size=4096,
        intermediate_size=512,
        global_experts=256,
        local_experts=256,
        mode="tp",
        degree=4,
        partition_widths=partition_widths,
    )


def test_partition_widths_are_priced_but_never_searched() -> None:
    base = _model()
    model = _model(partition_widths=(12, 14))

    assert base.partition_widths == ()
    assert base.priced_widths == base.supported_widths
    assert model.supported_widths == base.supported_widths
    assert model.planner_widths == base.planner_widths
    assert model.candidate_shapes(CORES) == base.candidate_shapes(CORES)
    assert model.priced_widths == tuple(sorted(set(base.supported_widths) | {12, 14}))
    assert model.T_iso(2040, 14) > 0.0
    with pytest.raises(KeyError, match="unsupported analytical thread width 14"):
        base.T_iso(2040, 14)
    with pytest.raises(ValueError, match="partition_widths"):
        _model(partition_widths=(0,))


def test_partition_tasks_give_each_long_expert_a_lane_and_the_rest_to_1t_lanes() -> None:
    planner = IntervalPlanner(_model(partition_widths=(14,)), CORES, cpu_ids=range(CORES))
    shape = (14,) * 5 + (1,) * 26
    tasks = planner.long_short_partition_tasks(BIMODAL, shape)

    assert sorted(expert for expert, *_ in tasks) == [expert for expert, _ in BIMODAL]
    longs = [task for task in tasks if task[3] == 14]
    assert [(expert, core, deps) for expert, _, core, _, deps in longs] == [
        (expert, 14 * expert, []) for expert in range(5)
    ]
    shorts = [task for task in tasks if task[3] == 1]
    assert {core for _, _, core, _, _ in shorts} == set(range(70, CORES))
    for index, (_, _, core, _, deps) in enumerate(tasks):
        # Each task waits only for the previous task on its own lane.
        assert all(tasks[dep][2] == core and dep < index for dep in deps)
    with pytest.raises(ValueError, match="not a long/short partition"):
        planner.long_short_partition_tasks(BIMODAL, (16,) * 6)


def test_partition_candidate_requires_a_route_gap_and_partition_widths() -> None:
    uniform = [(expert, 48) for expert in range(64)]
    skewed = [(expert, 64 - expert) for expert in range(64)]

    assert IntervalPlanner(_model(), CORES, cpu_ids=range(CORES)).long_short_partition_candidate(BIMODAL) is None
    planner = IntervalPlanner(_model(partition_widths=(12, 14)), CORES, cpu_ids=range(CORES))
    assert planner.long_short_partition_candidate(uniform) is None
    assert planner.long_short_partition_candidate(skewed) is None
    candidate = planner.long_short_partition_candidate(BIMODAL)
    assert candidate is not None
    assert candidate["shape"][0] in (12, 14) and candidate["shape"][:5] == (candidate["shape"][0],) * 5
    assert candidate["makespan_ns"] == planner._score(candidate["tasks"])


def test_quick_selects_a_partition_only_when_it_scores_faster_and_caches_it() -> None:
    model = _model(partition_widths=range(2, 17))
    planner = PlannedMoE(model, num_cores=CORES, search_mode="quick", long_short_partition=True)
    spec = planner.plan_spec_for(BIMODAL)
    reference = PlannedMoE(model, num_cores=CORES, search_mode="quick").plan_spec_for(BIMODAL)
    interval = planner.interval_planners[0]
    partition = interval.long_short_partition_candidate(BIMODAL)
    homogeneous_score = interval._score(interval.plan_quick(BIMODAL)["tasks"])

    if partition["makespan_ns"] < homogeneous_score:
        assert spec["assignment_order"] == "long_short_partition"
        assert tuple(spec["shape"]) == partition["shape"]
    else:
        assert spec["bridge"]["task_core_begins"] == reference["bridge"]["task_core_begins"]
    cached = planner.plan_spec_for(BIMODAL)
    assert planner.last["cache_hit"]
    assert cached["bridge"]["task_expert_ids"] == spec["bridge"]["task_expert_ids"]
    assert cached["bridge"]["task_core_begins"] == spec["bridge"]["task_core_begins"]
    assert cached["bridge"]["task_threads"] == spec["bridge"]["task_threads"]


def test_partition_flag_is_quick_only_and_off_by_default() -> None:
    model = _model(partition_widths=(12,))
    with pytest.raises(ValueError, match="long_short_partition"):
        PlannedMoE(model, num_cores=CORES, search_mode="full", long_short_partition=True)
    with pytest.raises(ValueError, match="long_short_partition"):
        PlannedMoE(model, num_cores=CORES, search_mode="quick", fixed_threads=16, long_short_partition=True)
    default = PlannedMoE(model, num_cores=CORES, search_mode="quick").plan_spec_for(BIMODAL)
    assert default["assignment_order"] != "long_short_partition"
