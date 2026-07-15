#include "planner_common.h"

#include <algorithm>
#include <functional>
#include <set>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace moe_planner {

Workload prepare_workload(const int32_t* routes_hist, int num_experts, int num_cores, const CostModel& cm) {
  Workload w;
  w.num_cores = num_cores;
  w.expert_ids.reserve(num_experts);
  w.routes.reserve(num_experts);

  // Active experts, then sort by the Python key (-routes, expert_id).
  struct AW {
    int32_t id;
    int32_t routes;
  };
  std::vector<AW> active;
  active.reserve(num_experts);
  for (int e = 0; e < num_experts; ++e) {
    if (routes_hist[e] > 0) active.push_back({e, routes_hist[e]});
  }
  std::sort(active.begin(), active.end(), [](const AW& a, const AW& b) {
    if (a.routes != b.routes) return a.routes > b.routes;  // -routes asc
    return a.id < b.id;                                    // expert_id asc
  });

  w.num_active = static_cast<int>(active.size());
  for (const auto& a : active) {
    w.expert_ids.push_back(a.id);
    w.routes.push_back(a.routes);
  }

  // Dense cost table cost_rows[a * C + (t-1)] = T_expert(routes[a], t).
  const int A = w.num_active;
  const int C = num_cores;
  w.cost_rows.resize(static_cast<size_t>(A) * C);

  // Only worth threading when the table is large; otherwise fork/join
  // overhead dwarfs the work (see the latency analysis).
  // The precomputed (pow-free) fill is ~6us even at A=256,C=64, far below the
  // OpenMP fork/join overhead, so it stays serial at realistic sizes.
  const bool parallel = (static_cast<long>(A) * C) >= 262144;
  (void)parallel;

  if (cm.kind() == CostModel::Kind::Synthetic) {
    // Fast path: hoist pow() and the per-thread overhead out of the
    // A x C inner loop. useful = min(t, routes) in [1, C], so the speedup
    // factor depends only on (useful - 1) in [0, C-1]; precompute both
    // tables once (C pow() calls instead of A*C). Bit-identical to
    // synthetic_ns(); the harness invariant check (which recomputes via
    // the pow path) verifies this.
    std::vector<double> speedup(C);   // index d = useful - 1
    std::vector<double> overhead(C);  // index t - 1
    for (int t = 1; t <= C; ++t) {
      speedup[t - 1] = CostModel::synth_speedup(t);  // useful = t
      overhead[t - 1] = CostModel::synth_overhead(t);
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (parallel)
#endif
    for (int a = 0; a < A; ++a) {
      const int routes = w.routes[a];
      const double serial = CostModel::synth_serial(routes);
      int64_t* row = &w.cost_rows[static_cast<size_t>(a) * C];
      // useful = min(t, routes): for t <= routes, useful = t; beyond
      // that useful saturates at routes -> speedup[routes - 1].
      const int sat = std::min(C, routes);
      for (int t = 1; t <= sat; ++t) {
        row[t - 1] = static_cast<int64_t>(serial / speedup[t - 1] + overhead[t - 1]);
      }
      for (int t = sat + 1; t <= C; ++t) {
        row[t - 1] = static_cast<int64_t>(serial / speedup[routes - 1] + overhead[t - 1]);
      }
    }
    return w;
  }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (parallel)
#endif
  for (int a = 0; a < A; ++a) {
    const int routes = w.routes[a];
    int64_t* row = &w.cost_rows[static_cast<size_t>(a) * C];
    for (int t = 1; t <= C; ++t) {
      row[t - 1] = cm.estimate(routes, t);
    }
  }
  return w;
}

int64_t emit_sequential_waves(const std::vector<TeamTmp>& teams, int num_cores, PlanResult& r) {
  r.wave_offsets.clear();
  r.team_expert_ids.clear();
  r.team_threads.clear();
  r.wave_offsets.push_back(0);

  int used = 0;
  int64_t wave_max = 0;
  int64_t execute = 0;
  int wave_start = 0;  // team index where the current wave began

  for (const auto& t : teams) {
    const bool cur_nonempty = static_cast<int>(r.team_expert_ids.size()) > wave_start;
    if (cur_nonempty && used + t.threads > num_cores) {
      r.wave_offsets.push_back(static_cast<int32_t>(r.team_expert_ids.size()));
      execute += wave_max;
      wave_start = static_cast<int>(r.team_expert_ids.size());
      used = 0;
      wave_max = 0;
    }
    r.team_expert_ids.push_back(t.expert_id);
    r.team_threads.push_back(t.threads);
    used += t.threads;
    if (t.time_ns > wave_max) wave_max = t.time_ns;
  }
  if (static_cast<int>(r.team_expert_ids.size()) > wave_start) {
    r.wave_offsets.push_back(static_cast<int32_t>(r.team_expert_ids.size()));
    execute += wave_max;
  }
  return execute;
}

int64_t emit_ffd_waves(const std::vector<TeamTmp>& teams, int num_cores, PlanResult& r) {
  struct Bin {
    int used;
    int64_t maxt;
    std::vector<TeamTmp> teams;
  };
  std::vector<Bin> bins;

  for (const auto& t : teams) {
    bool placed = false;
    for (auto& b : bins) {
      if (b.used + t.threads <= num_cores) {
        b.teams.push_back(t);
        b.used += t.threads;
        if (t.time_ns > b.maxt) b.maxt = t.time_ns;
        placed = true;
        break;
      }
    }
    if (!placed) bins.push_back({t.threads, t.time_ns, {t}});
  }

  r.wave_offsets.clear();
  r.team_expert_ids.clear();
  r.team_threads.clear();
  r.wave_offsets.push_back(0);
  int64_t execute = 0;
  for (const auto& b : bins) {
    for (const auto& t : b.teams) {
      r.team_expert_ids.push_back(t.expert_id);
      r.team_threads.push_back(t.threads);
    }
    r.wave_offsets.push_back(static_cast<int32_t>(r.team_expert_ids.size()));
    execute += b.maxt;
  }
  return execute;
}

namespace {

// Generate partitions of `rem` into exactly `k` non-increasing parts, each
// <= max_part, in descending-lexicographic order (largest first part first).
// This is exactly the order Python's sort key (len, tuple(-item)) produces
// within a fixed part count. `emit` returns false to stop generation early.
bool gen_k_parts(int rem, int k, int max_part, std::vector<int32_t>& cur,
                 const std::function<bool(const std::vector<int32_t>&)>& emit) {
  if (k == 0) {
    if (rem == 0) return emit(cur);
    return true;
  }
  int high = std::min(max_part, rem - (k - 1));
  int low = (rem + k - 1) / k;  // ceil(rem / k)
  for (int v = high; v >= low; --v) {
    cur.push_back(v);
    bool keep_going = gen_k_parts(rem - v, k - 1, v, cur, emit);
    cur.pop_back();
    if (!keep_going) return false;
  }
  return true;
}

}  // namespace

std::vector<std::vector<int32_t>> enumerate_core_group_shapes(int num_cores, int max_shapes) {
  std::vector<std::vector<int32_t>> shapes;

  // Pass 1: emit in final sorted order (parts-count asc, desc-lex), stopping
  // as soon as we exceed max_shapes so we never materialize p(num_cores)
  // partitions for large core counts.
  bool overflow = false;
  std::vector<int32_t> cur;
  auto collect = [&](const std::vector<int32_t>& s) -> bool {
    shapes.push_back(s);
    if (static_cast<int>(shapes.size()) > max_shapes) {
      overflow = true;
      return false;
    }
    return true;
  };
  for (int k = 1; k <= num_cores && !overflow; ++k) {
    cur.clear();
    if (!gen_k_parts(num_cores, k, num_cores, cur, collect)) break;
  }

  if (!overflow) return shapes;  // full set fits; pure sorted order.

  // Pass 2 (cap triggered): preserve [num_cores] and every uniform divisor
  // grouping, then fill from sorted order. Matches the Python keep logic.
  std::vector<std::vector<int32_t>> keep;
  std::set<std::vector<int32_t>> seen;
  auto add = [&](std::vector<int32_t> s) {
    if (seen.insert(s).second) keep.push_back(std::move(s));
  };
  add({static_cast<int32_t>(num_cores)});
  for (int gs = num_cores; gs >= 1; --gs) {
    if (num_cores % gs == 0) {
      add(std::vector<int32_t>(num_cores / gs, static_cast<int32_t>(gs)));
    }
  }
  auto fill = [&](const std::vector<int32_t>& s) -> bool {
    add(s);
    return static_cast<int>(keep.size()) < max_shapes;
  };
  for (int k = 1; k <= num_cores; ++k) {
    cur.clear();
    if (!gen_k_parts(num_cores, k, num_cores, cur, fill)) break;
  }
  return keep;
}

}  // namespace moe_planner
