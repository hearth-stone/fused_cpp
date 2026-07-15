#include "exact_solver.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace moe_planner {

namespace {

constexpr int64_t kInf = std::numeric_limits<int64_t>::max() / 4;

// Exact min-max thread allocation for one wave (the set of experts in `mask`):
// minimize max_e cost(e, t_e) subject to t_e in [1,C], sum t_e <= C.
//
// The optimal max M* is the smallest achievable cost threshold for which the
// minimum-thread allocation (each expert uses the fewest threads to reach <= M)
// still fits the core budget. We enumerate candidate thresholds = the distinct
// cost(e,t) values and take the smallest feasible one. Exact even when cost is
// non-monotone in threads (the per-thread overhead term can make it U-shaped).
int64_t wave_cost(uint32_t mask, const Workload& w) {
  const int C = w.num_cores;
  int k = 0;
  int experts[32];
  for (int a = 0; a < w.num_active; ++a)
    if (mask & (1u << a)) experts[k++] = a;
  if (k > C) return kInf;  // cannot give every expert >= 1 thread

  // Candidate thresholds.
  std::vector<int64_t> cands;
  cands.reserve(static_cast<size_t>(k) * C);
  for (int i = 0; i < k; ++i)
    for (int t = 1; t <= C; ++t) cands.push_back(w.cost(experts[i], t));
  std::sort(cands.begin(), cands.end());
  cands.erase(std::unique(cands.begin(), cands.end()), cands.end());

  for (int64_t M : cands) {
    long total = 0;
    bool ok = true;
    for (int i = 0; i < k && ok; ++i) {
      // fewest threads for this expert to reach cost <= M.
      int best_t = -1;
      for (int t = 1; t <= C; ++t) {
        if (w.cost(experts[i], t) <= M) {
          best_t = t;
          break;
        }
      }
      if (best_t < 0)
        ok = false;
      else
        total += best_t;
    }
    if (ok && total <= C) return M;
  }
  return kInf;
}

}  // namespace

ExactResult exact_optimum(const Workload& w, int max_active) {
  const int A = w.num_active;
  if (A == 0) return {true, 0, 0};
  if (A > max_active || A > 30) return {false, A, -1};

  const uint32_t full = (A == 32) ? 0xFFFFFFFFu : ((1u << A) - 1u);
  const size_t n = static_cast<size_t>(full) + 1;

  // Precompute wave cost for every subset (kInf if it cannot form one wave).
  std::vector<int64_t> wc(n);
  wc[0] = kInf;
  for (uint32_t m = 1; m < n; ++m) wc[m] = wave_cost(m, w);

  // f[S] = min total over partitions of S into waves.
  std::vector<int64_t> f(n, kInf);
  f[0] = 0;
  for (uint32_t S = 1; S < n; ++S) {
    int64_t best = kInf;
    // Enumerate non-empty sub-masks T of S as the next wave.
    for (uint32_t T = S; T; T = (T - 1) & S) {
      const int64_t wcT = wc[T];
      if (wcT >= kInf) continue;
      const int64_t rest = f[S ^ T];
      if (rest >= kInf) continue;
      const int64_t cand = wcT + rest;
      if (cand < best) best = cand;
    }
    f[S] = best;
  }

  return {true, A, f[full]};
}

}  // namespace moe_planner
