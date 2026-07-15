// FIXED_GLOBAL_THREADS planner.
//
// One thread per active expert, packed sequentially into waves. Baseline that
// ignores load imbalance. Matches plan_fixed_global_threads() in the Python
// reference.
#include "planner_dispatch.h"

namespace moe_planner {

PlanResult plan_fixed_global_threads(const Workload& w) {
  PlanResult r;
  r.kind = PlanKind::FIXED_GLOBAL_THREADS;
  r.num_cores = w.num_cores;

  std::vector<TeamTmp> teams;
  teams.reserve(w.num_active);
  for (int a = 0; a < w.num_active; ++a) {
    teams.push_back({w.expert_ids[a], 1, w.cost(a, 1)});
  }
  r.estimated_execute_cost_ns = emit_sequential_waves(teams, w.num_cores, r);
  fill_active(r, w);
  return r;
}

}  // namespace moe_planner
