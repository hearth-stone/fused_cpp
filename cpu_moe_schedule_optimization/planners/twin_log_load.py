#!/usr/bin/env python3
"""Twin implementation of the LOG_LOAD planner for cross-language validation."""

import math
from typing import Dict, Sequence

import cpu_moe_schedule_optimization.planners.offline_simulator as sim


def plan_log_load(
    active: Sequence[sim.ExpertWork],
    num_cores: int,
    cost_model: sim.ExpertCostModel,
    planner_cost_model: sim.PlannerCostModel,
) -> sim.Plan:
    """LOG_LOAD planner: thread allocation proportional to log(routes).

    Algorithm:
      C = num_cores
      weight[a] = 1.0 + log(routes[a])  (natural log)
      sumw = sum weight[a] over active experts
      threads[a] = clamp(floor(C * weight[a] / sumw + 0.5), 1, C)
      teams sorted by (-time_ns, -threads, expert_id)
      FFD packing
    """
    if not active:
        raise ValueError("active expert list must not be empty")

    # First pass: compute sum of weights
    sumw = sum(1.0 + math.log(work.routes) for work in active)

    # Build teams with thread allocation proportional to log(routes)
    teams = []
    for work in active:
        weight = 1.0 + math.log(work.routes)
        threads = max(1, min(num_cores, int(math.floor(num_cores * weight / sumw + 0.5))))
        teams.append(sim.make_team(work, threads, cost_model))

    # Sort by (-time_ns, -threads, expert_id)
    teams.sort(key=lambda t: (-t.estimated_time_ns, -t.threads, t.expert_id))

    # FFD packing
    waves = sim.first_fit_decreasing(teams, num_cores)

    return sim.Plan(
        kind=sim.PlanKind.GREEDY_MARGINAL_GAIN,
        num_cores=num_cores,
        active_experts=list(active),
        waves=waves,
        measured_plan_cost_ns=0,
        estimated_plan_cost_ns=0,
        exactness_scope="heuristic",
        metadata={},
    )
