// SPDX-License-Identifier: Apache-2.0
#include "probe_event_sim.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace moe_planner {
namespace {

constexpr double kDoneFraction = 1e-12;

// ProbeEventModel reads a curve at int(round(cores * resolution)), clamped below at zero;
// Python's round() rounds halves to even, as nearbyint does in the default rounding mode.
double Lookup(const std::vector<double>& table, double cores, int resolution) {
  const double scaled = std::nearbyint(cores * resolution);
  const size_t index = scaled <= 0.0 ? 0 : std::min(static_cast<size_t>(scaled), table.size() - 1);
  return table[index];
}

}  // namespace

NativeProbeEventSim::NativeProbeEventSim(int resolution, double g0) : resolution_(resolution), g0_(g0) {
  if (resolution_ <= 0 || !(g0_ >= 0.0 && g0_ < 1.0)) {
    throw std::invalid_argument("probe event simulator needs a positive resolution and 0 <= g0 < 1");
  }
}

int NativeProbeEventSim::add_table(std::vector<double> values) {
  if (values.empty()) {
    throw std::invalid_argument("probe curve tables must not be empty");
  }
  tables_.push_back(std::move(values));
  return static_cast<int>(tables_.size() - 1);
}

int NativeProbeEventSim::add_profile(ProbeEventProfile profile) {
  if (profile.phase_ns.size() != profile.phase_load.size() || profile.width <= 0) {
    throw std::invalid_argument("probe event profile needs one load flag per phase and a positive width");
  }
  for (int table : {profile.ll_table, profile.ls_table, profile.sl_table, profile.ss_table}) {
    if (table < 0 || table >= static_cast<int>(tables_.size())) {
      throw std::invalid_argument("probe event profile names an unknown curve table");
    }
  }
  profiles_.push_back(std::move(profile));
  return static_cast<int>(profiles_.size() - 1);
}

ProbeEventResult NativeProbeEventSim::simulate(const std::vector<int>& profile_ids,
                                               const std::vector<std::vector<int>>& dependencies) const {
  const size_t count = profile_ids.size();
  if (dependencies.size() != count) {
    throw std::invalid_argument("one dependency list per task is required");
  }
  std::vector<const ProbeEventProfile*> task(count);
  std::vector<int> pending(count);
  std::vector<std::vector<int>> successors(count);
  for (size_t index = 0; index < count; ++index) {
    const int id = profile_ids[index];
    if (id < 0 || id >= static_cast<int>(profiles_.size())) {
      throw std::invalid_argument("unknown probe event profile " + std::to_string(id));
    }
    task[index] = &profiles_[static_cast<size_t>(id)];
    pending[index] = static_cast<int>(dependencies[index].size());
    for (int dependency : dependencies[index]) {
      if (dependency < 0 || dependency >= static_cast<int>(index)) {
        throw std::invalid_argument("placed task dependencies must refer to earlier tasks");
      }
      successors[static_cast<size_t>(dependency)].push_back(static_cast<int>(index));
    }
  }

  std::vector<int> phase(count, 0);
  std::vector<double> remaining(count, 1.0);
  std::vector<int> active;
  active.reserve(count);
  double load_cores = 0.0;
  double steady_cores = 0.0;

  const auto share = [&](size_t index) -> std::pair<double, double> {
    const ProbeEventProfile& profile = *task[index];
    const int p = phase[index];
    if (p < 0 || p >= static_cast<int>(profile.phase_ns.size())) {
      return {0.0, 0.0};
    }
    if (profile.phase_load[static_cast<size_t>(p)]) {
      return {static_cast<double>(profile.width), 0.0};
    }
    return {profile.steady_load_cores, profile.steady_steady_cores};
  };
  const auto enter = [&](size_t index) {
    const ProbeEventProfile& profile = *task[index];
    while (phase[index] < static_cast<int>(profile.phase_ns.size()) &&
           profile.phase_ns[static_cast<size_t>(phase[index])] <= 0.0) {
      ++phase[index];
    }
    const auto [load, steady] = share(index);
    load_cores += load;
    steady_cores += steady;
  };
  const auto leave = [&](size_t index) {
    const auto [load, steady] = share(index);
    load_cores -= load;
    steady_cores -= steady;
  };
  std::vector<int> started;
  const auto start = [&](size_t index) {
    remaining[index] = 1.0;
    started.push_back(static_cast<int>(index));
    if (task[index]->overhead_ns > 0.0) {
      phase[index] = -1;
      return;
    }
    phase[index] = 0;
    enter(index);
  };

  for (size_t index = 0; index < count; ++index) {
    if (pending[index] == 0) {
      start(index);
    }
  }
  active.swap(started);
  started.clear();

  ProbeEventResult result;
  result.finish_ns.assign(count, 0.0);
  std::vector<double> duration(count, 0.0);
  std::vector<int> done;
  double now = 0.0;
  while (!active.empty()) {
    double step = std::numeric_limits<double>::infinity();
    for (int id : active) {
      const size_t index = static_cast<size_t>(id);
      const ProbeEventProfile& profile = *task[index];
      const int p = phase[index];
      if (p < 0) {
        duration[index] = profile.overhead_ns;
      } else {
        const double tau = profile.phase_ns[static_cast<size_t>(p)];
        double loading_excess = 0.0;
        double steady_excess = 0.0;
        if (profile.phase_load[static_cast<size_t>(p)]) {
          const double others = load_cores - static_cast<double>(profile.width);
          loading_excess = Lookup(tables_[static_cast<size_t>(profile.ll_table)], others, resolution_) - 1.0;
          steady_excess = Lookup(tables_[static_cast<size_t>(profile.ls_table)], steady_cores, resolution_) - 1.0;
        } else {
          const double others = load_cores - profile.steady_load_cores;
          loading_excess = Lookup(tables_[static_cast<size_t>(profile.sl_table)], others, resolution_) - 1.0;
          steady_excess = Lookup(tables_[static_cast<size_t>(profile.ss_table)],
                                 steady_cores - profile.steady_steady_cores, resolution_) - 1.0;
        }
        const double dilation = 1.0 + loading_excess + steady_excess;
        double value = tau * dilation;
        if (profile.windowed) {
          if (profile.e_cal < 0.01 || dilation - 1.0 >= profile.e_cal) {
            value *= profile.r_cal;
          } else {
            const double level = (dilation - 1.0) / profile.e_cal;
            value = tau * ((1.0 - g0_) + level * (profile.r_cal * (1.0 + profile.e_cal) - (1.0 - g0_)));
          }
        }
        duration[index] = value;
      }
      step = std::min(step, remaining[index] * duration[index]);
    }
    now += step;
    done.clear();
    for (int id : active) {
      const size_t index = static_cast<size_t>(id);
      remaining[index] -= duration[index] > 0.0 ? step / duration[index] : 1.0;
      if (remaining[index] <= kDoneFraction) {
        done.push_back(id);
      }
    }
    for (int id : done) {
      const size_t index = static_cast<size_t>(id);
      leave(index);
      ++phase[index];
      remaining[index] = 1.0;
      enter(index);
      if (phase[index] >= static_cast<int>(task[index]->phase_ns.size())) {
        active.erase(std::find(active.begin(), active.end(), id));
        result.finish_ns[index] = now;
        for (int successor : successors[index]) {
          if (--pending[static_cast<size_t>(successor)] == 0) {
            start(static_cast<size_t>(successor));
          }
        }
      }
    }
    for (int id : started) {
      active.insert(std::lower_bound(active.begin(), active.end(), id), id);
    }
    started.clear();
  }
  for (int value : pending) {
    if (value != 0) {
      throw std::invalid_argument("placed task dependencies contain a cycle");
    }
  }
  result.makespan_ns = now;
  return result;
}

std::vector<double> NativeProbeEventSim::makespans(const std::vector<std::vector<int>>& profiles,
                                                   const std::vector<std::vector<std::vector<int>>>& dependencies,
                                                   int workers) const {
  if (profiles.size() != dependencies.size()) {
    throw std::invalid_argument("one dependency set per plan is required");
  }
  const size_t count = profiles.size();
  std::vector<double> result(count, 0.0);
  std::vector<std::exception_ptr> errors(count);
  const int active_workers = std::max(1, std::min<int>(workers, static_cast<int>(count)));
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(active_workers) if (active_workers > 1)
#else
  (void)active_workers;
#endif
  for (ptrdiff_t index = 0; index < static_cast<ptrdiff_t>(count); ++index) {
    try {
      result[static_cast<size_t>(index)] =
          simulate(profiles[static_cast<size_t>(index)], dependencies[static_cast<size_t>(index)]).makespan_ns;
    } catch (...) {
      errors[static_cast<size_t>(index)] = std::current_exception();
    }
  }
  for (const std::exception_ptr& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
  return result;
}

}  // namespace moe_planner
