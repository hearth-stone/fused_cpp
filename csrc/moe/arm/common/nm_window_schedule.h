// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstdint>

namespace fused_cpp::nm_window {

inline constexpr int64_t kDefaultTargetBytes = 1 << 20;

struct Geometry {
  int64_t n_windows = 1;
  int64_t m_splits = 1;

  int64_t task_count(int64_t outer_groups = 1) const { return outer_groups * n_windows * m_splits; }
};

// Choose N ownership before M parallelism. Each outer group has an independent
// packed-B matrix. M overdecomposition is optional for panelized callers that
// need several equal-sized tasks per worker to absorb panel remainders.
inline Geometry Choose(int64_t n_tiles, int64_t bytes_per_tile, int64_t m_units, int64_t threads,
                       int64_t outer_groups = 1, int64_t target_bytes = kDefaultTargetBytes,
                       int64_t tasks_per_thread = 1) {
  Geometry geometry;
  if (n_tiles <= 0 || bytes_per_tile <= 0 || m_units <= 0 || threads <= 0 || outer_groups <= 0 || target_bytes <= 0 ||
      tasks_per_thread <= 0) {
    return geometry;
  }

  const int64_t target_tasks = threads * tasks_per_thread;
  const int64_t min_windows =
      std::min<int64_t>(n_tiles, std::max<int64_t>(1, (n_tiles * bytes_per_tile + target_bytes - 1) / target_bytes));
  geometry.n_windows = min_windows;
  if (tasks_per_thread == 1 && target_tasks >= outer_groups * min_windows) {
    for (int64_t windows = min_windows; windows <= n_tiles; ++windows) {
      const int64_t owners = outer_groups * windows;
      const int64_t max_window_tiles = (n_tiles + windows - 1) / windows;
      if (owners > target_tasks) {
        break;
      }
      if (target_tasks % owners == 0 && max_window_tiles * bytes_per_tile <= target_bytes) {
        geometry.n_windows = windows;
        break;
      }
    }
  }

  const int64_t owners = outer_groups * geometry.n_windows;
  geometry.m_splits = std::min<int64_t>(m_units, std::max<int64_t>(1, (target_tasks + owners - 1) / owners));
  return geometry;
}

}  // namespace fused_cpp::nm_window
