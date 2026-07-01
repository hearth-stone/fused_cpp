// Small-case EXACT optimum for the wave-barrier execution objective.
//
// Objective (identical to what every planner approximates):
//   assign threads_e >= 1 to each active expert and partition the experts into
//   waves with sum(threads) <= C per wave; wave time = max_e cost(e, threads_e);
//   total = sum of wave times. Minimize total.
//
// This is moldable-task makespan-style scheduling (NP-hard in general), solved
// exactly here by subset DP for small active-expert counts. It provides the
// TRUE optimum that planner regret is measured against (DESIGN.md Phase 3 /
// research questions Q3-Q5). Torch-free.
#pragma once

#include <cstdint>

#include "planner_common.h"

namespace moe_planner {

struct ExactResult {
    bool feasible;       // false if num_active > max_active (DP too large)
    int num_active;
    int64_t execute_ns;  // optimal sum-of-wave-max; -1 when infeasible
};

// Exact optimum via subset DP. Refuses (feasible=false) when num_active exceeds
// max_active, since the DP is O(3^A). Default cap keeps it well under a second.
ExactResult exact_optimum(const Workload& w, int max_active = 16);

}  // namespace moe_planner
