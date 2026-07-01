// ENUMERATE_CORE_GROUPS planner.
//
// Enumerate integer core-group shapes (partitions of num_cores), assign the
// largest experts to the largest slots, and pick the shape with the lowest
// estimated execution cost. Shape evaluation is independent per shape, so it
// is parallelized across shapes (the one planner where threading reliably pays
// off — up to 512 shapes). Matches plan_enumerate_core_groups().
#include <limits>
#include <vector>

#include "planner_dispatch.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace moe_planner {

namespace {

// Estimated execution cost of assigning sorted experts to one shape's slots:
// experts fill descending slots, len(slots) experts per wave, wave time = max
// team time. Matches build_waves_for_core_group_shape().
int64_t shape_execute_cost(const Workload& w, const std::vector<int32_t>& shape) {
    const int A = w.num_active;
    const int k = static_cast<int>(shape.size());
    int64_t execute = 0;
    for (int ws = 0; ws < A; ws += k) {
        int64_t wave_max = 0;
        const int end = std::min(ws + k, A);
        for (int i = ws; i < end; ++i) {
            const int64_t t = w.cost(i, shape[i - ws]);
            if (t > wave_max) wave_max = t;
        }
        execute += wave_max;
    }
    return execute;
}

}  // namespace

PlanResult plan_enumerate_core_groups(const Workload& w) {
    const int C = w.num_cores;
    const std::vector<std::vector<int32_t>> shapes =
        enumerate_core_group_shapes(C);
    const int S = static_cast<int>(shapes.size());

    std::vector<int64_t> exec(S);
    // Per-shape work is light (mostly L1/L2 cost-row reads); only thread when the
    // shape count is large enough to amortize fork/join (e.g. C >= 32).
    const bool parallel = (static_cast<long>(S) * w.num_active) >= 100000;
    (void)parallel;
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 8) if (parallel)
#endif
    for (int s = 0; s < S; ++s) {
        exec[s] = shape_execute_cost(w, shapes[s]);
    }

    // argmin with ties broken by the lowest (earliest enumerated) shape.
    int best_s = 0;
    int64_t best_exec = std::numeric_limits<int64_t>::max();
    for (int s = 0; s < S; ++s) {
        if (exec[s] < best_exec) {
            best_exec = exec[s];
            best_s = s;
        }
    }

    // Rebuild the winning plan's wave/team arrays.
    PlanResult r;
    r.kind = PlanKind::ENUMERATE_CORE_GROUPS;
    r.num_cores = C;
    const std::vector<int32_t>& shape = shapes[best_s];
    const int k = static_cast<int>(shape.size());
    const int A = w.num_active;
    r.wave_offsets.push_back(0);
    for (int ws = 0; ws < A; ws += k) {
        const int end = std::min(ws + k, A);
        for (int i = ws; i < end; ++i) {
            r.team_expert_ids.push_back(w.expert_ids[i]);
            r.team_threads.push_back(shape[i - ws]);
        }
        r.wave_offsets.push_back(static_cast<int32_t>(r.team_expert_ids.size()));
    }
    r.estimated_execute_cost_ns = best_exec;
    r.selected_core_group_shape = shape;
    fill_active(r, w);
    return r;
}

}  // namespace moe_planner
