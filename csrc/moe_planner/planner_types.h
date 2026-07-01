// CPU MoE schedule planners — torch-free core types.
//
// This header is intentionally free of any libtorch / pybind11 dependency so
// the planner core can be compiled both into the `fused_cpp._C` extension and
// into a tiny standalone benchmark harness (see standalone/moe_planner/).
//
// Semantics mirror cpu_moe_schedule_optimization/planners/offline_simulator.py
// exactly, so the C++ planners can be equivalence-tested against the Python
// reference simulator.
#pragma once

#include <cstdint>
#include <vector>

namespace moe_planner {

enum class PlanKind : int {
    FIXED_GLOBAL_THREADS = 0,
    SORTED_TOKEN_BALANCED_1T = 1,
    UNIFORM_WAVES = 2,
    GREEDY_MARGINAL_GAIN = 3,
    ENUMERATE_CORE_GROUPS = 4,
    // Extended planner family (each implemented in its own translation unit).
    LOAD_PROPORTIONAL = 5,
    SQRT_LOAD = 6,
    LOG_LOAD = 7,
    HEAVY_LIGHT_HYBRID = 8,
    KARMARKAR_KARP = 9,
};

inline const char* plan_kind_name(PlanKind kind) {
    switch (kind) {
        case PlanKind::FIXED_GLOBAL_THREADS:     return "FIXED_GLOBAL_THREADS";
        case PlanKind::SORTED_TOKEN_BALANCED_1T: return "SORTED_TOKEN_BALANCED_1T";
        case PlanKind::UNIFORM_WAVES:            return "UNIFORM_WAVES";
        case PlanKind::GREEDY_MARGINAL_GAIN:     return "GREEDY_MARGINAL_GAIN";
        case PlanKind::ENUMERATE_CORE_GROUPS:    return "ENUMERATE_CORE_GROUPS";
        case PlanKind::LOAD_PROPORTIONAL:        return "LOAD_PROPORTIONAL";
        case PlanKind::SQRT_LOAD:                return "SQRT_LOAD";
        case PlanKind::LOG_LOAD:                 return "LOG_LOAD";
        case PlanKind::HEAVY_LIGHT_HYBRID:       return "HEAVY_LIGHT_HYBRID";
        case PlanKind::KARMARKAR_KARP:           return "KARMARKAR_KARP";
    }
    return "UNKNOWN";
}

// Compact scheduled-bridge plan, matching Plan.to_scheduled_bridge() in the
// Python simulator. Teams in wave i occupy
// [wave_offsets[i], wave_offsets[i + 1]).
struct PlanResult {
    PlanKind kind = PlanKind::FIXED_GLOBAL_THREADS;
    int num_cores = 0;

    // Active experts, sorted by the Python key (-routes, expert_id):
    // descending routes, ascending expert_id on ties. Parallel arrays.
    std::vector<int32_t> active_expert_ids;
    std::vector<int32_t> active_routes;

    // Scheduled bridge tensors (int32-compatible).
    std::vector<int32_t> wave_offsets;     // length num_waves + 1
    std::vector<int32_t> team_expert_ids;  // length num_teams
    std::vector<int32_t> team_threads;     // length num_teams

    // sum over waves of max team time in that wave (barrier-after-wave model).
    int64_t estimated_execute_cost_ns = 0;

    // Optional diagnostics (planner-specific; -1 when unused).
    int selected_threads_per_expert = -1;            // UNIFORM_WAVES
    int selected_wave_budget = -1;                   // GREEDY_MARGINAL_GAIN
    std::vector<int32_t> selected_core_group_shape;  // ENUMERATE_CORE_GROUPS

    int num_waves() const {
        return wave_offsets.empty() ? 0
                                    : static_cast<int>(wave_offsets.size()) - 1;
    }
    int num_teams() const { return static_cast<int>(team_expert_ids.size()); }
};

}  // namespace moe_planner
