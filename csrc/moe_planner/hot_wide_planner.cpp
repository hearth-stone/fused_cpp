// SPDX-License-Identifier: Apache-2.0
#include "hot_wide_planner.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>

namespace moe_planner {
namespace {

// First-fit decreasing by width into the domain with the most free cores, as pack_lanes does.
bool PackLanes(const std::vector<int>& widths, const std::vector<int>& domain_cores, std::vector<int>* begins) {
  std::vector<int> free = domain_cores;
  std::vector<int> cursor(domain_cores.size(), 0);
  int running = 0;
  for (size_t domain = 0; domain < domain_cores.size(); ++domain) {
    cursor[domain] = running;
    running += domain_cores[domain];
  }
  std::vector<int> order(widths.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&widths](int a, int b) {
    return widths[a] != widths[b] ? widths[a] > widths[b] : a < b;
  });
  begins->assign(widths.size(), 0);
  for (int index : order) {
    size_t domain = 0;
    for (size_t candidate = 1; candidate < free.size(); ++candidate) {
      if (free[candidate] > free[domain]) {
        domain = candidate;
      }
    }
    if (free[domain] < widths[index]) {
      return false;
    }
    (*begins)[index] = cursor[domain];
    cursor[domain] += widths[index];
    free[domain] -= widths[index];
  }
  return true;
}

struct LaneHeapEntry {
  double load;
  int lane;
  bool operator>(const LaneHeapEntry& other) const {
    return load != other.load ? load > other.load : lane > other.lane;
  }
};

}  // namespace

NativeHotWidePlanner::NativeHotWidePlanner(int num_cores, std::vector<int> domain_cores, int bulk_width,
                                           std::vector<int> wide_widths, int max_wide_lanes, int max_wide_cores,
                                           std::vector<std::pair<int, double>> lane_scale)
    : num_cores_(num_cores),
      domain_cores_(std::move(domain_cores)),
      bulk_width_(bulk_width),
      lane_scale_(std::move(lane_scale)) {
  int total = 0;
  for (int cores : domain_cores_) {
    total += cores;
  }
  if (total != num_cores_) {
    throw std::invalid_argument("domain cores must cover num_cores");
  }
  if (bulk_width_ <= 0) {
    throw std::invalid_argument("bulk width must be positive");
  }
  std::sort(wide_widths.begin(), wide_widths.end(), std::greater<int>());
  // Every multiset of at most max_wide_lanes wide lanes, the rest of the machine in bulk lanes.
  std::vector<std::vector<int>> combos{{}};
  for (int count = 1; count <= max_wide_lanes; ++count) {
    std::vector<std::vector<int>> next;
    for (const std::vector<int>& combo : combos) {
      if (static_cast<int>(combo.size()) != count - 1) {
        continue;
      }
      for (int width : wide_widths) {
        if (!combo.empty() && width > combo.back()) {
          continue;  // combinations with replacement, non-increasing
        }
        std::vector<int> extended = combo;
        extended.push_back(width);
        next.push_back(std::move(extended));
      }
    }
    combos.insert(combos.end(), next.begin(), next.end());
  }
  for (const std::vector<int>& combo : combos) {
    int wide = 0;
    for (int width : combo) {
      wide += width;
    }
    if (wide > max_wide_cores) {
      continue;
    }
    std::vector<int> shape = combo;
    for (int lane = 0; lane < (num_cores_ - wide) / bulk_width_; ++lane) {
      shape.push_back(bulk_width_);
    }
    std::vector<int> begins;
    if (!PackLanes(shape, domain_cores_, &begins)) {
      continue;
    }
    if (std::find(shapes_.begin(), shapes_.end(), shape) == shapes_.end()) {
      shapes_.push_back(std::move(shape));
    }
  }
  if (shapes_.empty()) {
    throw std::invalid_argument("no template packs into the LLC domains");
  }
}

NativeHotWidePlanner::Plan NativeHotWidePlanner::PlanExperts(
    const std::vector<int>& expert_ids, const std::vector<int>& routes, const std::vector<int>& cost_widths,
    const std::vector<std::vector<double>>& costs_by_width) const {
  if (expert_ids.size() != routes.size()) {
    throw std::invalid_argument("expert ids and routes must have the same length");
  }
  if (cost_widths.size() != costs_by_width.size()) {
    throw std::invalid_argument("cost widths and cost rows must have the same length");
  }
  // Largest routes first, ties by expert id, as the Python planner sorts.
  std::vector<int> order;
  order.reserve(expert_ids.size());
  for (size_t index = 0; index < expert_ids.size(); ++index) {
    if (routes[index] > 0) {
      order.push_back(static_cast<int>(index));
    }
  }
  if (order.empty()) {
    throw std::invalid_argument("at least one active expert is required");
  }
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return routes[a] != routes[b] ? routes[a] > routes[b] : expert_ids[a] < expert_ids[b];
  });

  auto scale_of = [this](int width) {
    for (const std::pair<int, double>& entry : lane_scale_) {
      if (entry.first == width) {
        return entry.second;
      }
    }
    return 1.0;
  };
  auto row_of = [&](int width) {
    for (size_t index = 0; index < cost_widths.size(); ++index) {
      if (cost_widths[index] == width) {
        return static_cast<int>(index);
      }
    }
    throw std::invalid_argument("missing cost row for a template width");
  };

  std::vector<int> best_members;      // lane index per expert, in `order` sequence
  std::vector<int> best_shape;
  double best_score = std::numeric_limits<double>::infinity();
  std::vector<int> members(order.size());

  std::vector<int> sequence(shapes_.size());
  std::iota(sequence.begin(), sequence.end(), 0);
  if (last_shape_ >= 0 && last_shape_ < static_cast<int>(shapes_.size())) {
    // The template that won the previous call first: it usually survives and then prunes the rest.
    sequence.erase(sequence.begin() + last_shape_);
    sequence.insert(sequence.begin(), last_shape_);
  }

  for (int shape_index : sequence) {
    const std::vector<int>& shape = shapes_[shape_index];
    std::vector<std::priority_queue<LaneHeapEntry, std::vector<LaneHeapEntry>, std::greater<LaneHeapEntry>>> heaps;
    std::vector<int> heap_width;
    std::vector<double> scaled;
    std::vector<int> rows;
    for (size_t lane = 0; lane < shape.size(); ++lane) {
      int width = shape[lane];
      size_t slot = 0;
      while (slot < heap_width.size() && heap_width[slot] != width) {
        ++slot;
      }
      if (slot == heap_width.size()) {
        heap_width.push_back(width);
        heaps.emplace_back();
        rows.push_back(row_of(width));
        scaled.push_back(scale_of(width));
      }
      heaps[slot].push(LaneHeapEntry{0.0, static_cast<int>(lane)});
    }
    double worst = 0.0;
    bool aborted = false;
    for (size_t position = 0; position < order.size(); ++position) {
      int expert = order[position];
      double best_load = std::numeric_limits<double>::infinity();
      int best_lane = -1;
      size_t best_slot = 0;
      for (size_t slot = 0; slot < heaps.size(); ++slot) {
        const LaneHeapEntry& top = heaps[slot].top();
        double load = top.load + costs_by_width[rows[slot]][expert] * scaled[slot];
        if (load < best_load || (load == best_load && top.lane < best_lane)) {
          best_load = load;
          best_lane = top.lane;
          best_slot = slot;
        }
      }
      heaps[best_slot].pop();
      heaps[best_slot].push(LaneHeapEntry{best_load, best_lane});
      members[position] = best_lane;
      worst = std::max(worst, best_load);
      if (best_load >= best_score) {  // LPT loads only grow: this template cannot win
        aborted = true;
        break;
      }
    }
    if (!aborted && worst < best_score) {
      best_score = worst;
      best_members = members;
      best_shape = shape;
      last_shape_ = shape_index;
    }
  }
  if (best_shape.empty()) {
    throw std::runtime_error("no template produced an assignment");
  }

  // Lane contents in the searched plans' order: the hottest expert first, the rest ascending.
  std::vector<std::vector<int>> lanes(best_shape.size());
  for (size_t position = 0; position < order.size(); ++position) {
    lanes[best_members[position]].push_back(order[position]);
  }
  std::vector<int> used_widths;
  std::vector<std::vector<int>> used_lanes;
  for (size_t lane = 0; lane < lanes.size(); ++lane) {
    if (lanes[lane].empty()) {
      continue;
    }
    std::vector<int> entries = lanes[lane];
    if (entries.size() > 1) {
      std::reverse(entries.begin() + 1, entries.end());
    }
    used_lanes.push_back(std::move(entries));
    used_widths.push_back(best_shape[lane]);
  }
  std::vector<int> begins;
  if (!PackLanes(used_widths, domain_cores_, &begins)) {
    throw std::runtime_error("a subset of a packable template failed to pack");
  }

  Plan plan;
  plan.shape = best_shape;
  plan.score_ns = best_score;
  plan.templates = static_cast<int>(shapes_.size());
  for (size_t lane = 0; lane < used_lanes.size(); ++lane) {
    int previous = -1;
    for (int index : used_lanes[lane]) {
      IntervalTask task;
      task.expert_id = expert_ids[index];
      task.routes = routes[index];
      task.core_begin = begins[lane];
      task.threads = used_widths[lane];
      if (previous >= 0) {
        task.dependencies.push_back(previous);
      }
      plan.tasks.push_back(std::move(task));
      previous = static_cast<int>(plan.tasks.size()) - 1;
    }
  }
  return plan;
}

}  // namespace moe_planner
