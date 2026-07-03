#!/usr/bin/env python3
# ⚠ DEPRECATED (wave, 后续不考虑) — see cpu_moe_schedule_optimization/DEPRECATED_WAVE.md
"""Twin implementation of the SQRT_LOAD planner.

This file mirrors csrc/moe_planner/plan_sqrt_load.cpp and is used for
cross-language equivalence testing. It is intentionally minimal and does not
handle edge cases beyond what the C++ implementation handles.
"""

import math
from typing import Dict, List, Sequence

import offline_simulator as sim


def plan_sqrt_load(
    active: Sequence[sim.ExpertWork],
    num_cores: int,
    cost_model: sim.ExpertCostModel,
    planner_cost_model: sim.PlannerCostModel,
) -> sim.Plan:
    """Allocate threads proportionally to sqrt(routes), then FFD pack.

    Algorithm:
        weight[a] = sqrt(routes[a])
        sumw = sum(weight[a]) over active experts in order
        threads[a] = clamp(floor(C * weight[a] / sumw + 0.5), 1, C)
        Build teams, sort by (-time_ns, -threads, expert_id), FFD pack.

    Args:
        active: Active experts sorted by (-routes, expert_id).
        num_cores: Core budget for each wave.
        cost_model: Expert cost model T_expert(routes, threads).
        planner_cost_model: Planner overhead cost model (unused for heuristic).

    Returns:
        A Plan with GREEDY_MARGINAL_GAIN kind (matches C++ emit_ffd_waves output).
    """
    if not active:
        return sim.Plan(
            kind=sim.PlanKind.GREEDY_MARGINAL_GAIN,
            num_cores=num_cores,
            active_experts=[],
            waves=[],
            measured_plan_cost_ns=0,
            estimated_plan_cost_ns=0,
            exactness_scope="heuristic",
            metadata={},
        )

    C = num_cores

    # First pass: compute sum of sqrt(routes) in active order.
    sumw = sum(math.sqrt(work.routes) for work in active)

    # Second pass: compute thread allocation.
    teams: List[sim.Team] = []
    for work in active:
        weight = math.sqrt(work.routes)
        threads = int(math.floor(C * weight / sumw + 0.5))
        threads = max(1, min(C, threads))
        teams.append(sim.make_team(work, threads, cost_model))

    # Sort by (-time_ns, -threads, expert_id).
    teams.sort(key=lambda t: (-t.estimated_time_ns, -t.threads, t.expert_id))

    # FFD pack into waves.
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
