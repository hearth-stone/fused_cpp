// SORTED_TOKEN_BALANCED_1T planner (LPT, strong baseline).
//
// Sort active experts by routed-token count (already done in the workload) and
// greedily assign each one-thread expert to the lightest logical core queue by
// routed-token load. Queues are then emitted as depth-aligned waves.
// Matches plan_sorted_token_balanced_1t() in the Python reference.
#include <vector>

#include "planner_dispatch.h"

namespace moe_planner {

PlanResult plan_sorted_token_balanced_1t(const Workload& w) {
  PlanResult r;
  r.kind = PlanKind::SORTED_TOKEN_BALANCED_1T;
  r.num_cores = w.num_cores;

  const int C = w.num_cores;
  std::vector<std::vector<TeamTmp>> queues(C);
  std::vector<int64_t> route_loads(C, 0);

  // Active experts are pre-sorted by (-routes, expert_id). Assign each to the
  // queue with the smallest (load, index).
  for (int a = 0; a < w.num_active; ++a) {
    int core = 0;
    int64_t best_load = route_loads[0];
    for (int q = 1; q < C; ++q) {
      if (route_loads[q] < best_load) {  // strict: ties keep lower index
        best_load = route_loads[q];
        core = q;
      }
    }
    queues[core].push_back({w.expert_ids[a], 1, w.cost(a, 1)});
    route_loads[core] += w.routes[a];
  }

  // Emit waves by depth: wave d gathers queue[q][d] for q = 0..C-1.
  int max_depth = 0;
  for (const auto& q : queues) max_depth = std::max(max_depth, static_cast<int>(q.size()));

  r.wave_offsets.push_back(0);
  int64_t execute = 0;
  for (int d = 0; d < max_depth; ++d) {
    int64_t wave_max = 0;
    bool any = false;
    for (int q = 0; q < C; ++q) {
      if (d < static_cast<int>(queues[q].size())) {
        const TeamTmp& t = queues[q][d];
        r.team_expert_ids.push_back(t.expert_id);
        r.team_threads.push_back(t.threads);
        if (t.time_ns > wave_max) wave_max = t.time_ns;
        any = true;
      }
    }
    if (any) {
      r.wave_offsets.push_back(static_cast<int32_t>(r.team_expert_ids.size()));
      execute += wave_max;
    }
  }
  r.estimated_execute_cost_ns = execute;
  fill_active(r, w);
  return r;
}

}  // namespace moe_planner
