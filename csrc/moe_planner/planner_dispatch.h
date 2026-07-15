// Public planner API: one entry point per PlanKind, plus a kind dispatcher.
//
// Each planner takes a prepared Workload (active experts sorted + dense cost
// table) and returns a PlanResult. Keeping the heavy prepare_workload() step
// out of the planners lets a caller score several planners on one workload.
#pragma once

#include "planner_common.h"
#include "planner_types.h"

namespace moe_planner {

// Each planner lives in its own translation unit.
PlanResult plan_fixed_global_threads(const Workload& w);
PlanResult plan_sorted_token_balanced_1t(const Workload& w);
PlanResult plan_uniform_waves(const Workload& w);
PlanResult plan_greedy_marginal_gain(const Workload& w, int extra_wave_window = 8);
PlanResult plan_enumerate_core_groups(const Workload& w);

// Extended planner family.
PlanResult plan_load_proportional(const Workload& w);
PlanResult plan_sqrt_load(const Workload& w);
PlanResult plan_log_load(const Workload& w);
PlanResult plan_heavy_light_hybrid(const Workload& w);
PlanResult plan_karmarkar_karp(const Workload& w);

PlanResult run_planner(PlanKind kind, const Workload& w);

// Convenience: prepare the workload from a routing histogram and run one
// planner end to end.
PlanResult plan_from_routes(PlanKind kind, const int32_t* routes_hist, int num_experts, int num_cores,
                            const CostModel& cm);

// Copy the workload's active-expert view into the result.
inline void fill_active(PlanResult& r, const Workload& w) {
  r.active_expert_ids = w.expert_ids;
  r.active_routes = w.routes;
}

}  // namespace moe_planner
