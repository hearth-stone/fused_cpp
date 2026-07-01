#include "planner_dispatch.h"

#include <stdexcept>

namespace moe_planner {

PlanResult run_planner(PlanKind kind, const Workload& w) {
    switch (kind) {
        case PlanKind::FIXED_GLOBAL_THREADS:
            return plan_fixed_global_threads(w);
        case PlanKind::SORTED_TOKEN_BALANCED_1T:
            return plan_sorted_token_balanced_1t(w);
        case PlanKind::UNIFORM_WAVES:
            return plan_uniform_waves(w);
        case PlanKind::GREEDY_MARGINAL_GAIN:
            return plan_greedy_marginal_gain(w);
        case PlanKind::ENUMERATE_CORE_GROUPS:
            return plan_enumerate_core_groups(w);
        case PlanKind::LOAD_PROPORTIONAL:
            return plan_load_proportional(w);
        case PlanKind::SQRT_LOAD:
            return plan_sqrt_load(w);
        case PlanKind::LOG_LOAD:
            return plan_log_load(w);
        case PlanKind::HEAVY_LIGHT_HYBRID:
            return plan_heavy_light_hybrid(w);
        case PlanKind::KARMARKAR_KARP:
            return plan_karmarkar_karp(w);
    }
    throw std::invalid_argument("unknown PlanKind");
}

PlanResult plan_from_routes(PlanKind kind, const int32_t* routes_hist,
                            int num_experts, int num_cores, const CostModel& cm) {
    Workload w = prepare_workload(routes_hist, num_experts, num_cores, cm);
    return run_planner(kind, w);
}

}  // namespace moe_planner
