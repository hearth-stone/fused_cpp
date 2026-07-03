# ⚠ DEPRECATED (wave, 后续不考虑) — see cpu_moe_schedule_optimization/DEPRECATED_WAVE.md
"""Twin LOAD_PROPORTIONAL planner for cross-language equivalence testing."""

import math
import offline_simulator as sim


def plan_load_proportional(
    active, num_cores, cost_model, planner_cost_model
):
    """Allocate threads proportional to route counts and pack with FFD.

    Algorithm:
      total = sum of routes over active experts.
      For each active expert a:
        threads = clamp(round(num_cores * routes[a] / total), 1, num_cores)
      Build teams, sort by (-time, -threads, expert_id), pack with FFD.

    Args:
        active: Sequence of ExpertWork objects.
        num_cores: Number of CPU cores (bin capacity).
        cost_model: ExpertCostModel for T(routes, threads).
        planner_cost_model: PlannerCostModel (unused, kept for API parity).

    Returns:
        sim.Plan with waves and estimated execution cost.
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

    total_routes = sum(work.routes for work in active)
    teams = []

    for work in active:
        # Round-half-up via floor(x + 0.5), matching C++ implementation.
        frac = num_cores * work.routes / total_routes
        threads = int(math.floor(frac + 0.5))
        threads = max(1, min(num_cores, threads))
        team = sim.make_team(work, threads, cost_model)
        teams.append(team)

    # Sort by (-time, -threads, expert_id): largest time first.
    teams.sort(key=lambda t: (-t.estimated_time_ns, -t.threads, t.expert_id))

    # Pack with first-fit-decreasing.
    waves = sim.first_fit_decreasing(teams, num_cores)

    # Compute estimated execution cost (sum of per-wave max times).
    execute_ns = sum(wave.estimated_wave_time_ns for wave in waves)

    # Note: kind is a placeholder; only bridge arrays + execute cost are compared.
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
