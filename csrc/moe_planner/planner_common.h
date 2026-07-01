// Shared infrastructure for the MoE planners.
//
//   * Workload      : active experts (sorted) + a dense precomputed cost table
//                     cost_rows[a * C + (t-1)] = T_expert(routes[a], t).
//                     This is the single biggest lever for planner latency:
//                     planners then index a contiguous int64 array instead of
//                     re-evaluating the cost model (pow / map lookups).
//   * wave builders : sequential packing and first-fit-decreasing, matching
//                     the Python reference exactly.
//   * shape enum    : integer partitions of num_cores, matching
//                     enumerate_core_group_shapes() including the 512 cap.
#pragma once

#include <cstdint>
#include <vector>

#include "cost_model.h"
#include "planner_types.h"

namespace moe_planner {

constexpr int kMaxCoreGroupShapes = 512;

struct Workload {
    int num_cores = 0;
    int num_active = 0;
    std::vector<int32_t> expert_ids;   // sorted by (-routes, expert_id)
    std::vector<int32_t> routes;       // parallel to expert_ids
    std::vector<int64_t> cost_rows;    // num_active * num_cores

    // T_expert(routes[a], threads), threads in [1, num_cores].
    inline int64_t cost(int a, int threads) const {
        return cost_rows[static_cast<size_t>(a) * num_cores + (threads - 1)];
    }
};

// Build the active-expert view and precompute the dense cost table.
// `routes_hist` has length num_experts. Experts with routes <= 0 are dropped.
Workload prepare_workload(const int32_t* routes_hist, int num_experts,
                          int num_cores, const CostModel& cm);

// One expert execution bound to a thread count, with its precomputed time.
struct TeamTmp {
    int32_t expert_id;
    int32_t threads;
    int64_t time_ns;
};

// Sequential packing into waves with thread budget `num_cores`, teams kept in
// the given order (matches build_waves_from_ordered_teams). Fills r.wave_offsets
// / r.team_expert_ids / r.team_threads and returns sum of per-wave max times.
int64_t emit_sequential_waves(const std::vector<TeamTmp>& teams, int num_cores,
                              PlanResult& r);

// First-fit-decreasing bin packing (bin capacity = num_cores), teams placed in
// the given order (matches first_fit_decreasing). Fills the same fields and
// returns sum of per-wave (per-bin) max times.
int64_t emit_ffd_waves(const std::vector<TeamTmp>& teams, int num_cores,
                       PlanResult& r);

// Integer partitions of num_cores, matching enumerate_core_group_shapes():
// sorted by (num_parts asc, parts desc-lex); capped at max_shapes with the
// [num_cores] full-core shape and every uniform divisor grouping preserved.
std::vector<std::vector<int32_t>> enumerate_core_group_shapes(
    int num_cores, int max_shapes = kMaxCoreGroupShapes);

}  // namespace moe_planner
