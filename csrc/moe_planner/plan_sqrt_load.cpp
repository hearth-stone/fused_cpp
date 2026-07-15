// SQRT_LOAD planner.
//
// Allocates threads to experts proportionally to sqrt(routes), then packs the
// resulting teams with first-fit-decreasing. Matches plan_sqrt_load() in the
// Python twin.
//
// Algorithm:
//   weight[a] = sqrt(routes[a])
//   sumw = sum(weight[a]) over active experts a=0..A-1
//   threads[a] = clamp(floor(C * weight[a] / sumw + 0.5), 1, C)
//   Build teams (expert_id, threads, cost(a, threads))
//   Sort by (-time_ns, -threads, expert_id)
//   FFD pack into waves
//   Return sum of per-wave max times

#include <algorithm>
#include <cmath>
#include <vector>

#include "planner_dispatch.h"

namespace moe_planner {

PlanResult plan_sqrt_load(const Workload& w) {
  const int A = w.num_active;
  const int C = w.num_cores;

  PlanResult r;
  r.kind = PlanKind::SQRT_LOAD;
  r.num_cores = C;

  if (A == 0) {
    r.wave_offsets.push_back(0);
    fill_active(r, w);
    return r;
  }

  // First pass: compute sum of sqrt(routes) over active experts in index order.
  double sumw = 0.0;
  for (int a = 0; a < A; ++a) {
    sumw += std::sqrt(static_cast<double>(w.routes[a]));
  }

  // Second pass: compute thread allocation for each expert.
  std::vector<TeamTmp> teams(A);
  for (int a = 0; a < A; ++a) {
    double weight = std::sqrt(static_cast<double>(w.routes[a]));
    int threads = static_cast<int>(std::floor(C * weight / sumw + 0.5));
    threads = std::max(1, std::min(C, threads));
    teams[a] = {w.expert_ids[a], threads, w.cost(a, threads)};
  }

  // Sort by (-time_ns, -threads, expert_id): largest time first.
  std::sort(teams.begin(), teams.end(), [](const TeamTmp& x, const TeamTmp& y) {
    if (x.time_ns != y.time_ns) return x.time_ns > y.time_ns;
    if (x.threads != y.threads) return x.threads > y.threads;
    return x.expert_id < y.expert_id;
  });

  r.estimated_execute_cost_ns = emit_ffd_waves(teams, C, r);
  fill_active(r, w);
  return r;
}

}  // namespace moe_planner
