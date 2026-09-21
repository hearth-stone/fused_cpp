// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "interval_planner.h"

namespace moe_planner {

// Native port of cpu_moe_schedule_optimization/planners/hot_wide_planner.py: the templates a
// near-optimal plan takes (a few wide lanes for the hottest experts, 4T lanes for the rest),
// heterogeneous LPT on per-width costs, and the lane order the searched plans show.
//
// The costs come from the Python model as immutable doubles - one row per width, one entry per
// expert, before the per-width lane scale - so this class owns no Python state. It keeps the
// template that won the previous call, which only reorders the search, so a stale value can
// cost time but never changes the result.
class NativeHotWidePlanner {
 public:
  NativeHotWidePlanner(int num_cores, std::vector<int> domain_cores, int bulk_width,
                       std::vector<int> wide_widths, int max_wide_lanes, int max_wide_cores,
                       std::vector<std::pair<int, double>> lane_scale);

  struct Plan {
    std::vector<IntervalTask> tasks;
    std::vector<int> shape;
    double score_ns = 0.0;
    int templates = 0;
  };

  // ``costs_by_width[i][j]``: cost of expert ``j`` (in the caller's order) on width
  // ``cost_widths[i]``, before the lane scale.
  Plan PlanExperts(const std::vector<int>& expert_ids, const std::vector<int>& routes,
                   const std::vector<int>& cost_widths,
                   const std::vector<std::vector<double>>& costs_by_width) const;

  const std::vector<std::vector<int>>& shapes() const { return shapes_; }

 private:
  int num_cores_ = 0;
  std::vector<int> domain_cores_;
  int bulk_width_ = 4;
  std::vector<std::pair<int, double>> lane_scale_;
  std::vector<std::vector<int>> shapes_;
  mutable int last_shape_ = -1;
};

}  // namespace moe_planner
