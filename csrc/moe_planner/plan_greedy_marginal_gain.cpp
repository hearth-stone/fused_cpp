// GREEDY_MARGINAL_GAIN planner.
//
// For each candidate wave budget, hand out thread slots one at a time to the
// expert with the largest predicted execution-time reduction, then pack the
// resulting teams with first-fit-decreasing. Keep the budget with the lowest
// estimated execution cost. Matches plan_greedy_marginal_gain().
//
// Performance:
//   * The marginal-gain inner loop maintains a contiguous per-expert gain[]
//     array and uses a NEON/SVE argmax (planner_simd.h) each step.
//   * The independent wave budgets are evaluated in parallel.
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "planner_dispatch.h"
#include "planner_simd.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace moe_planner {

namespace {

constexpr int64_t kNegInf = std::numeric_limits<int64_t>::min();

// Greedy thread allocation for one wave budget. Writes per-expert thread counts
// into `tc` (size num_active). Mirrors allocate_threads_by_marginal_gain().
void allocate_by_marginal_gain(const Workload& w, int wave_budget,
                               std::vector<int32_t>& tc) {
    const int A = w.num_active;
    const int C = w.num_cores;
    tc.assign(A, 1);

    const long total_slots = static_cast<long>(wave_budget) * C;
    long remaining = total_slots - A;
    if (remaining <= 0 || C <= 1) return;

    // gain[a] = T(routes,tc) - T(routes,tc+1) at the expert's current tc, or
    // -inf when the expert is already at the core budget.
    std::vector<int64_t> gain(A);
    for (int a = 0; a < A; ++a) {
        gain[a] = w.cost(a, 1) - w.cost(a, 2);  // tc starts at 1, C >= 2 here
    }

    while (remaining > 0) {
        int64_t best = 0;
        const int idx = simd::argmax_i64(gain.data(), A, &best);
        if (idx < 0 || best <= 0) break;  // no strictly-positive gain left

        const int nt = ++tc[idx];
        if (nt >= C) {
            gain[idx] = kNegInf;
        } else {
            gain[idx] = w.cost(idx, nt) - w.cost(idx, nt + 1);
        }
        --remaining;
    }
}

// Build FFD-packed teams from a thread allocation and return execution cost.
int64_t build_and_pack(const Workload& w, const std::vector<int32_t>& tc,
                       PlanResult& out) {
    const int A = w.num_active;
    std::vector<TeamTmp> teams(A);
    for (int a = 0; a < A; ++a) {
        teams[a] = {w.expert_ids[a], tc[a], w.cost(a, tc[a])};
    }
    // Sort by (-time, -threads, expert_id): largest time first.
    std::sort(teams.begin(), teams.end(),
              [](const TeamTmp& x, const TeamTmp& y) {
                  if (x.time_ns != y.time_ns) return x.time_ns > y.time_ns;
                  if (x.threads != y.threads) return x.threads > y.threads;
                  return x.expert_id < y.expert_id;
              });
    return emit_ffd_waves(teams, w.num_cores, out);
}

}  // namespace

PlanResult plan_greedy_marginal_gain(const Workload& w, int extra_wave_window) {
    const int A = w.num_active;
    const int C = w.num_cores;

    PlanResult r;
    r.kind = PlanKind::GREEDY_MARGINAL_GAIN;
    r.num_cores = C;
    if (A == 0) {
        r.wave_offsets.push_back(0);
        fill_active(r, w);
        return r;
    }

    const int min_waves = (A + C - 1) / C;  // ceil(A / C)
    const int max_waves = std::min(A, min_waves + extra_wave_window);
    const int num_budgets = max_waves - min_waves + 1;

    // Evaluate each budget independently (parallel): allocation + FFD cost.
    std::vector<std::vector<int32_t>> tc_all(num_budgets);
    std::vector<int64_t> exec(num_budgets);
    const bool parallel = (static_cast<long>(num_budgets) * A * C) >= 8192;
    (void)parallel;
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) if (parallel)
#endif
    for (int b = 0; b < num_budgets; ++b) {
        const int wave_budget = min_waves + b;
        allocate_by_marginal_gain(w, wave_budget, tc_all[b]);
        PlanResult scratch;
        exec[b] = build_and_pack(w, tc_all[b], scratch);
    }

    // argmin with ties broken by the smallest wave budget.
    int best_b = 0;
    int64_t best_exec = std::numeric_limits<int64_t>::max();
    for (int b = 0; b < num_budgets; ++b) {
        if (exec[b] < best_exec) {
            best_exec = exec[b];
            best_b = b;
        }
    }

    r.estimated_execute_cost_ns = build_and_pack(w, tc_all[best_b], r);
    r.selected_wave_budget = min_waves + best_b;
    fill_active(r, w);
    return r;
}

}  // namespace moe_planner
