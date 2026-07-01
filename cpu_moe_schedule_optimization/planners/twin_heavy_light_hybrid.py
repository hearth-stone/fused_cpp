#!/usr/bin/env python3
"""Twin implementation of the HEAVY_LIGHT_HYBRID planner.

This file mirrors csrc/moe_planner/plan_heavy_light_hybrid.cpp and is used for
cross-language equivalence testing. It is intentionally minimal and does not
handle edge cases beyond what the C++ implementation handles.
"""

import math
from typing import Sequence

import offline_simulator as sim


def plan_heavy_light_hybrid(
    active: Sequence[sim.ExpertWork],
    num_cores: int,
    cost_model: sim.ExpertCostModel,
    planner_cost_model: sim.PlannerCostModel,
) -> sim.Plan:
    """Allocate threads based on heavy/light classification, then FFD pack.

    Algorithm:
        C = num_cores; A = len(active); total = sum(routes) (integer).
        mean = total / A (floating-point division).
        An expert is HEAVY if routes[a] >= mean, else LIGHT.
        sum_heavy = sum of routes[a] over heavy experts (integer).
        For heavy experts:
            threads[a] = clamp(floor(C * routes[a] / sum_heavy + 0.5), 1, C)
        For light experts:
            threads[a] = 1
        (If sum_heavy == 0 — only possible if A == 0 — return empty plan.)
        Build teams = (expert_id, threads, cost(a,threads)) for ALL experts.
        Sort by (-time_ns, -threads, expert_id).
        FFD pack.

    Args:
        active: Active experts sorted by (-routes, expert_id).
        num_cores: Core budget for each wave.
        cost_model: Expert cost model T_expert(routes, threads).
        planner_cost_model: Planner overhead cost model (unused for heuristic).

    Returns:
        A Plan with GREEDY_MARGINAL_GAIN kind (matches C++ emit_ffd_waves output).
    """
    # Handle empty active case.
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
    A = len(active)

    # Compute total routes and mean.
    total = sum(w.routes for w in active)
    mean = total / A  # Floating-point division.

    # Compute sum_heavy over heavy experts (routes >= mean).
    sum_heavy = sum(w.routes for w in active if w.routes >= mean)

    # Allocate threads.
    teams = []
    if sum_heavy == 0:
        # All experts are light (fallback: 1 thread each).
        for work in active:
            teams.append(sim.make_team(work, 1, cost_model))
    else:
        for work in active:
            if work.routes >= mean:
                # Heavy expert: proportional allocation.
                th = int(math.floor(C * work.routes / sum_heavy + 0.5))
                th = max(1, min(C, th))
            else:
                # Light expert: 1 thread.
                th = 1
            teams.append(sim.make_team(work, th, cost_model))

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
