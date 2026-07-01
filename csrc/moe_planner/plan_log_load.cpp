// LOG_LOAD planner.
//
// Allocates threads proportionally to log(routes) weight, then packs the
// resulting teams with first-fit-decreasing. Matches plan_log_load().
//
// Algorithm:
//   C = num_cores
//   weight[a] = 1.0 + log(routes[a])  (natural log)
//   sumw = sum weight[a] over active experts
//   threads[a] = clamp(floor(C * weight[a] / sumw + 0.5), 1, C)
//   teams sorted by (-time_ns, -threads, expert_id)
//   FFD packing

#include <algorithm>
#include <cmath>
#include <vector>

#include "planner_dispatch.h"

namespace moe_planner {

PlanResult plan_log_load(const Workload& w) {
    const int A = w.num_active;
    const int C = w.num_cores;

    PlanResult r;
    r.kind = PlanKind::LOG_LOAD;
    r.num_cores = C;
    fill_active(r, w);

    if (A == 0) {
        r.wave_offsets.push_back(0);
        r.estimated_execute_cost_ns = 0;
        return r;
    }

    // First pass: compute sum of weights
    double sumw = 0.0;
    for (int a = 0; a < A; ++a) {
        const double weight = 1.0 + std::log(static_cast<double>(w.routes[a]));
        sumw += weight;
    }

    // Second pass: compute thread counts
    std::vector<TeamTmp> teams(A);
    for (int a = 0; a < A; ++a) {
        const double weight = 1.0 + std::log(static_cast<double>(w.routes[a]));
        int threads = static_cast<int>(std::floor(C * weight / sumw + 0.5));
        threads = std::max(1, std::min(C, threads));
        teams[a] = {w.expert_ids[a], threads, w.cost(a, threads)};
    }

    // Sort by (-time_ns, -threads, expert_id): largest time first
    std::sort(teams.begin(), teams.end(),
              [](const TeamTmp& x, const TeamTmp& y) {
                  if (x.time_ns != y.time_ns) return x.time_ns > y.time_ns;
                  if (x.threads != y.threads) return x.threads > y.threads;
                  return x.expert_id < y.expert_id;
              });

    r.estimated_execute_cost_ns = emit_ffd_waves(teams, C, r);
    return r;
}

}  // namespace moe_planner
