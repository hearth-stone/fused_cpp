// HEAVY_LIGHT_HYBRID planner.
//
// Classify experts as HEAVY (routes >= mean) or LIGHT (routes < mean).
// Heavy experts get threads proportional to their route share;
// light experts get 1 thread each. Then pack with FFD.
//
// Algorithm:
//   C = num_cores; A = num_active; total = sum(routes) (integer).
//   mean = total / A (floating-point division).
//   An expert is HEAVY if routes[a] >= mean, else LIGHT.
//   sum_heavy = sum of routes[a] over heavy experts (integer).
//   For heavy experts:
//     threads[a] = clamp(floor(C * routes[a] / sum_heavy + 0.5), 1, C)
//   For light experts:
//     threads[a] = 1
//   (If sum_heavy == 0 — only possible if A == 0 — return empty plan.)
//   Build teams = (expert_id, threads, cost(a,threads)) for ALL experts.
//   Sort by key (-time_ns, -threads, expert_id).
//   Pack with FFD via emit_ffd_waves.

#include <algorithm>
#include <cmath>
#include <vector>

#include "planner_dispatch.h"

namespace moe_planner {

PlanResult plan_heavy_light_hybrid(const Workload& w) {
    const int A = w.num_active;
    const int C = w.num_cores;

    PlanResult r;
    r.kind = PlanKind::HEAVY_LIGHT_HYBRID;
    r.num_cores = C;

    // Handle empty active case.
    if (A == 0) {
        r.wave_offsets.push_back(0);
        fill_active(r, w);
        return r;
    }

    // Compute total routes and mean.
    int64_t total_routes = 0;
    for (int a = 0; a < A; ++a) {
        total_routes += w.routes[a];
    }
    const double mean = static_cast<double>(total_routes) / A;

    // Compute sum_heavy over heavy experts (routes[a] >= mean).
    int64_t sum_heavy = 0;
    for (int a = 0; a < A; ++a) {
        if (w.routes[a] >= mean) {
            sum_heavy += w.routes[a];
        }
    }

    // Allocate threads.
    std::vector<int32_t> tc(A);
    if (sum_heavy == 0) {
        // All experts are light (only possible if all routes == 0,
        // but routes > 0 for active experts by definition).
        // Fallback: 1 thread each.
        for (int a = 0; a < A; ++a) {
            tc[a] = 1;
        }
    } else {
        for (int a = 0; a < A; ++a) {
            if (w.routes[a] >= mean) {
                // Heavy expert: proportional allocation.
                double frac = static_cast<double>(C) * w.routes[a] / sum_heavy;
                int threads = static_cast<int>(std::floor(frac + 0.5));
                threads = std::max(1, std::min(C, threads));
                tc[a] = threads;
            } else {
                // Light expert: 1 thread.
                tc[a] = 1;
            }
        }
    }

    // Build teams with precomputed costs.
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

    r.estimated_execute_cost_ns = emit_ffd_waves(teams, C, r);
    fill_active(r, w);
    return r;
}

}  // namespace moe_planner
