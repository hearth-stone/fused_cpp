// LOAD_PROPORTIONAL planner.
//
// Allocate threads proportional to each active expert's route count,
// then pack with first-fit-decreasing. Matches plan_load_proportional().
//
// Algorithm:
//   C = num_cores; total = sum of routes over active experts (integer sum).
//   For each active expert a:
//     threads = clamp(floor(C * routes[a] / total + 0.5), 1, C)
//   Build teams = (expert_id, threads, cost(a,threads)).
//   Sort teams by key (-time_ns, -threads, expert_id).
//   Pack with FFD via emit_ffd_waves.

#include <algorithm>
#include <cmath>
#include <vector>

#include "planner_dispatch.h"

namespace moe_planner {

PlanResult plan_load_proportional(const Workload& w) {
    const int A = w.num_active;
    const int C = w.num_cores;

    PlanResult r;
    r.kind = PlanKind::LOAD_PROPORTIONAL;
    r.num_cores = C;
    if (A == 0) {
        r.wave_offsets.push_back(0);
        fill_active(r, w);
        return r;
    }

    // Compute total routes over active experts.
    int64_t total_routes = 0;
    for (int a = 0; a < A; ++a) {
        total_routes += w.routes[a];
    }

    // Allocate threads proportionally: clamp(round(C * routes[a] / total), 1, C)
    // Use double division and floor(x + 0.5) for round-half-up.
    std::vector<int32_t> tc(A);
    for (int a = 0; a < A; ++a) {
        double frac = static_cast<double>(C) * w.routes[a] / total_routes;
        int threads = static_cast<int>(std::floor(frac + 0.5));
        threads = std::max(1, std::min(C, threads));
        tc[a] = threads;
    }

    // Build teams with precomputed costs.
    std::vector<TeamTmp> teams(A);
    for (int a = 0; a < A; ++a) {
        teams[a] = {w.expert_ids[a], tc[a], w.cost(a, tc[a])};
    }

    // Sort by (-time, -threads, expert_id): largest time first.
    std::sort(teams.begin(), teams.end(),
              [](const TeamTmp& x, const TeamTmp& y) {
                  if (x.time_ns != y.time_ns) return x.time_ns > y.time_ns;
                  if (x.threads != y.threads) return x.threads > y.threads;
                  return x.expert_id < y.expert_id;
              });

    r.estimated_execute_cost_ns = emit_ffd_waves(teams, C, r);
    fill_active(r, w);
    return r;
}

}  // namespace moe_planner
