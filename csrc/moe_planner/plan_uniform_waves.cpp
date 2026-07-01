// UNIFORM_WAVES planner.
//
// Enumerate one uniform threads-per-expert value in [1, num_cores], pack
// sequentially into waves, and keep the value with the lowest estimated
// execution cost (ties keep the smallest thread count). Exact within this
// enumerated space. Matches plan_uniform_waves() in the Python reference.
#include <limits>
#include <vector>

#include "planner_dispatch.h"

namespace moe_planner {

PlanResult plan_uniform_waves(const Workload& w) {
    const int C = w.num_cores;
    const int A = w.num_active;

    PlanResult best;
    best.kind = PlanKind::UNIFORM_WAVES;
    best.num_cores = C;
    int64_t best_exec = std::numeric_limits<int64_t>::max();

    std::vector<TeamTmp> teams(A);
    for (int tpe = 1; tpe <= C; ++tpe) {
        for (int a = 0; a < A; ++a) {
            teams[a] = {w.expert_ids[a], tpe, w.cost(a, tpe)};
        }
        PlanResult tmp;
        tmp.kind = PlanKind::UNIFORM_WAVES;
        tmp.num_cores = C;
        const int64_t exec = emit_sequential_waves(teams, C, tmp);
        if (exec < best_exec) {  // strict: smallest tpe wins on tie
            best_exec = exec;
            tmp.estimated_execute_cost_ns = exec;
            tmp.selected_threads_per_expert = tpe;
            best = std::move(tmp);
        }
    }

    fill_active(best, w);
    return best;
}

}  // namespace moe_planner
