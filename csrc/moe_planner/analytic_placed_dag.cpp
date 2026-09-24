// SPDX-License-Identifier: Apache-2.0
#include "analytic_placed_dag.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kCompletionToleranceNs = 1e-6;
constexpr double kMinimumServiceSeconds = 1e-30;

double SmoothCapacityMiss(double working_set, double effective_capacity, double physical_capacity) {
  if (working_set <= effective_capacity) {
    return 0.0;
  }
  if (working_set >= physical_capacity || physical_capacity <= effective_capacity) {
    return 1.0;
  }
  const double x = (working_set - effective_capacity) / (physical_capacity - effective_capacity);
  return x * x * (3.0 - 2.0 * x);
}

double Dilation(double offered_rate, double capacity) {
  const double utilization = std::isfinite(capacity) ? offered_rate / capacity : 0.0;
  return std::max(1.0, utilization);
}

// Offered request rate of one phase on one resource: its demand over its own
// service time on that resource (AnalyticMoeCostModel._active_phase_state_placed).
double OfferedRate(double demand, double residual_scale, double time_ns) {
  return demand / std::max(residual_scale * time_ns * 1e-9, kMinimumServiceSeconds);
}

template <typename Function>
void ParallelFor(size_t count, int workers, Function&& function) {
  if (count == 0) {
    return;
  }
  const int active_workers = std::max(1, std::min<int>(workers, static_cast<int>(count)));
  std::vector<std::exception_ptr> errors(count);
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(active_workers) if (active_workers > 1)
#else
  (void)active_workers;
#endif
  for (ptrdiff_t index = 0; index < static_cast<ptrdiff_t>(count); ++index) {
    try {
      function(static_cast<size_t>(index));
    } catch (...) {
      errors[static_cast<size_t>(index)] = std::current_exception();
    }
  }
  for (const std::exception_ptr& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
}

// _normalize_placed_tasks: unique CPUs per thread, dependencies on earlier tasks,
// and tasks sharing a CPU ordered by the dependency closure.
std::vector<std::vector<int>> NormalizedDependencies(const AnalyticDagTasks& tasks) {
  const size_t count = tasks.routes.size();
  if (tasks.threads.size() != count || tasks.cpu_ids.size() != count || tasks.dependencies.size() != count) {
    throw std::invalid_argument("placed DAG task arrays must have equal lengths");
  }
  int max_cpu = 0;
  for (const std::vector<int>& cpus : tasks.cpu_ids) {
    for (int cpu : cpus) {
      if (cpu < 0) {
        throw std::invalid_argument("placed DAG CPU ids must be non-negative");
      }
      max_cpu = std::max(max_cpu, cpu);
    }
  }
  const size_t cpu_words = static_cast<size_t>(max_cpu) / 64 + 1;
  const size_t task_words = count / 64 + 1;
  std::vector<uint64_t> cpu_masks(count * cpu_words, 0);
  std::vector<uint64_t> ancestors(count * task_words, 0);
  std::vector<std::vector<int>> normalized(count);
  for (size_t task = 0; task < count; ++task) {
    const int threads = tasks.threads[task];
    const std::vector<int>& cpus = tasks.cpu_ids[task];
    uint64_t* mask = &cpu_masks[task * cpu_words];
    bool unique = threads > 0 && static_cast<int>(cpus.size()) == threads;
    for (int cpu : cpus) {
      const uint64_t bit = uint64_t{1} << (cpu % 64);
      unique = unique && (mask[cpu / 64] & bit) == 0;
      mask[cpu / 64] |= bit;
    }
    if (!unique) {
      throw std::invalid_argument("task " + std::to_string(task) + " placement must contain one unique CPU per thread");
    }
    std::vector<int> dependencies = tasks.dependencies[task];
    std::sort(dependencies.begin(), dependencies.end());
    dependencies.erase(std::unique(dependencies.begin(), dependencies.end()), dependencies.end());
    if (!dependencies.empty() && (dependencies.front() < 0 || dependencies.back() >= static_cast<int>(task))) {
      throw std::invalid_argument("placed task dependencies must refer to earlier tasks");
    }
    uint64_t* reachable = &ancestors[task * task_words];
    for (int dependency : dependencies) {
      reachable[dependency / 64] |= uint64_t{1} << (dependency % 64);
      const uint64_t* inherited = &ancestors[static_cast<size_t>(dependency) * task_words];
      for (size_t word = 0; word < task_words; ++word) {
        reachable[word] |= inherited[word];
      }
    }
    for (size_t earlier = 0; earlier < task; ++earlier) {
      const uint64_t* earlier_mask = &cpu_masks[earlier * cpu_words];
      bool overlap = false;
      for (size_t word = 0; word < cpu_words && !overlap; ++word) {
        overlap = (mask[word] & earlier_mask[word]) != 0;
      }
      if (overlap && (reachable[earlier / 64] & (uint64_t{1} << (earlier % 64))) == 0) {
        throw std::invalid_argument("overlapping placed tasks must be ordered by dependencies: earlier_task=" +
                                    std::to_string(earlier) + ", task=" + std::to_string(task));
      }
    }
    normalized[task] = std::move(dependencies);
  }
  return normalized;
}

}  // namespace

NativeAnalyticPlacedDag::NativeAnalyticPlacedDag(AnalyticDagMachine machine) : machine_(std::move(machine)) {
  const size_t cores = static_cast<size_t>(machine_.cores_per_rank);
  const size_t domains = machine_.domain_sizes.size();
  if (machine_.cores_per_rank <= 0) {
    throw std::invalid_argument("analytic DAG requires positive cores_per_rank");
  }
  if (!(machine_.rank_llc_capacity_bytes > 0.0)) {
    throw std::invalid_argument("analytic DAG requires a positive rank LLC capacity");
  }
  for (const std::vector<double>* table :
       {&machine_.gemm_rate, &machine_.l2_rate, &machine_.llc_rate, &machine_.dram_rate, &machine_.epilogue_rate,
        &machine_.wide_isolated_scale, &machine_.wide_full_cohort_scale, &machine_.narrow_full_cohort_correction}) {
    if (table->size() != cores + 1) {
      throw std::invalid_argument("per-thread tables must have cores_per_rank + 1 entries");
    }
  }
  if (machine_.domain_capacity_bytes.size() != domains || machine_.domain_llc_rate.size() != domains) {
    throw std::invalid_argument("LLC domain tables must match the domain count");
  }
  for (size_t domain = 0; domain < domains; ++domain) {
    if (machine_.domain_llc_rate[domain].size() != static_cast<size_t>(machine_.domain_sizes[domain]) + 1) {
      throw std::invalid_argument("each LLC domain rate table must have domain size + 1 entries");
    }
  }
  for (int domain : machine_.cpu_domain) {
    if (domain < -1 || domain >= static_cast<int>(domains)) {
      throw std::invalid_argument("cpu_domain entries must name a domain or be -1");
    }
  }
}

void NativeAnalyticPlacedDag::register_phases(int64_t routes, int threads, std::vector<AnalyticDagPhase> phases) {
  if (phases.empty()) {
    throw std::invalid_argument("analytical DAG tasks must have positive route counts");
  }
  for (AnalyticDagPhase& phase : phases) {
    if (!(phase.base_ns > 0.0) || !(phase.dram_rate > 0.0)) {
      throw std::invalid_argument("analytic phases need a positive base time and DRAM rate");
    }
    phase.gemm_offered = OfferedRate(phase.gemm_demand, phase.residual_scale, phase.gemm_ns);
    phase.l2_offered = OfferedRate(phase.l2_demand, phase.residual_scale, phase.l2_ns);
    phase.epilogue_offered = OfferedRate(phase.epilogue_demand, phase.residual_scale, phase.epilogue_ns);
    phase.llc_service_seconds = std::max(phase.residual_scale * phase.llc_ns * 1e-9, kMinimumServiceSeconds);
  }
  phases_[{routes, threads}] = std::move(phases);
}

bool NativeAnalyticPlacedDag::has_phases(int64_t routes, int threads) const {
  return phases_.count({routes, threads}) != 0;
}

double NativeAnalyticPlacedDag::makespan(const AnalyticDagTasks& tasks) const {
  const size_t count = tasks.routes.size();
  if (count == 0) {
    return 0.0;
  }
  const size_t domains = machine_.domain_sizes.size();
  if (domains == 0) {
    throw std::invalid_argument("placement-aware phase state requires calibrated LLC domains");
  }
  const std::vector<std::vector<int>> dependencies = NormalizedDependencies(tasks);
  const int cores = machine_.cores_per_rank;

  // Per task: its phases and, per phase, how many of the phase's active CPUs
  // (the placement's prefix) fall in each LLC domain.
  std::vector<const std::vector<AnalyticDagPhase>*> task_phases(count);
  std::vector<size_t> phase_offset(count + 1, 0);
  for (size_t task = 0; task < count; ++task) {
    const auto found = phases_.find({tasks.routes[task], tasks.threads[task]});
    if (found == phases_.end()) {
      throw std::out_of_range("phases are not registered for routes=" + std::to_string(tasks.routes[task]) +
                              " threads=" + std::to_string(tasks.threads[task]));
    }
    task_phases[task] = &found->second;
    phase_offset[task + 1] = phase_offset[task] + found->second.size();
  }
  std::vector<int> domain_counts(phase_offset[count] * domains, 0);
  for (size_t task = 0; task < count; ++task) {
    const std::vector<AnalyticDagPhase>& phases = *task_phases[task];
    for (size_t phase = 0; phase < phases.size(); ++phase) {
      const int active_threads = phases[phase].active_threads;
      if (active_threads > static_cast<int>(tasks.cpu_ids[task].size())) {
        throw std::invalid_argument("task " + std::to_string(task) + " placement is narrower than its active phase");
      }
      int* counts = &domain_counts[(phase_offset[task] + phase) * domains];
      for (int thread = 0; thread < active_threads; ++thread) {
        const int cpu = tasks.cpu_ids[task][static_cast<size_t>(thread)];
        const int domain =
            cpu < static_cast<int>(machine_.cpu_domain.size()) ? machine_.cpu_domain[static_cast<size_t>(cpu)] : -1;
        if (domain < 0) {
          throw std::invalid_argument("task " + std::to_string(task) + " uses CPU " + std::to_string(cpu) +
                                      " outside the calibrated rank");
        }
        ++counts[domain];
      }
    }
  }
  const size_t max_events = phase_offset[count] + count + 2;

  std::vector<int> waiting(count);
  std::vector<std::vector<int>> successors(count);
  for (size_t task = 0; task < count; ++task) {
    waiting[task] = static_cast<int>(dependencies[task].size());
    for (int dependency : dependencies[task]) {
      successors[static_cast<size_t>(dependency)].push_back(static_cast<int>(task));
    }
  }
  std::vector<size_t> phase_index(count, 0);
  std::vector<double> remaining(count);
  // Active tasks in ascending id: every shared-resource sum below runs in this order,
  // the order in which the Python reference iterates its active phases.
  std::vector<int> active;
  active.reserve(count);
  for (size_t task = 0; task < count; ++task) {
    remaining[task] = (*task_phases[task])[0].base_ns;
    if (waiting[task] == 0) {
      active.push_back(static_cast<int>(task));
    }
  }

  // Per-slot state, rebuilt for each event (structure of arrays).
  std::vector<const AnalyticDagPhase*> phase_of(count);
  std::vector<const int*> counts_of(count);
  std::vector<int> width_of(count);
  std::vector<double> dram_bytes(count), dram_ns(count), multipliers(count);
  std::vector<int> domain_threads(domains), requesting(domains), dram_threads(domains);
  std::vector<double> domain_working_set(domains), domain_spill(domains), llc_offered(domains), llc_dilation(domains),
      dram_offered(domains), dram_dilation(domains);
  std::vector<int> started_now;

  double wall_ns = machine_.call_setup_ns;
  size_t finished_count = 0;
  size_t guard = 0;
  while (finished_count < count) {
    if (++guard > 2 * max_events) {
      throw std::runtime_error("analytical stage simulator did not converge");
    }
    const size_t slots = active.size();
    if (slots == 0) {
      throw std::invalid_argument("DAG deadlock (cycle or unreachable task)");
    }

    // LLC working set and threads by domain.
    std::fill(domain_threads.begin(), domain_threads.end(), 0);
    std::fill(domain_working_set.begin(), domain_working_set.end(), 0.0);
    int occupied = 0;
    for (size_t slot = 0; slot < slots; ++slot) {
      const size_t task = static_cast<size_t>(active[slot]);
      const size_t global_phase = phase_offset[task] + phase_index[task];
      const AnalyticDagPhase* phase = &(*task_phases[task])[phase_index[task]];
      const int* counts = &domain_counts[global_phase * domains];
      phase_of[slot] = phase;
      counts_of[slot] = counts;
      width_of[slot] = static_cast<int>(tasks.cpu_ids[task].size());
      occupied += width_of[slot];
      for (size_t domain = 0; domain < domains; ++domain) {
        if (counts[domain] > 0) {
          const double share = static_cast<double>(counts[domain]) / phase->active_threads;
          domain_threads[domain] += counts[domain];
          domain_working_set[domain] += phase->working_set_bytes * share;
        }
      }
    }
    occupied = std::min(occupied, cores);
    for (size_t domain = 0; domain < domains; ++domain) {
      const double physical = machine_.domain_capacity_bytes[domain];
      domain_spill[domain] =
          domain_threads[domain] > 0
              ? SmoothCapacityMiss(domain_working_set[domain], physical * machine_.llc_effective_fraction, physical)
              : 0.0;
    }

    // Spill-dependent DRAM traffic and the threads requesting each rank resource.
    int gemm_threads = 0, l2_threads = 0, dram_rank_threads = 0, epilogue_threads = 0;
    for (size_t slot = 0; slot < slots; ++slot) {
      const AnalyticDagPhase& phase = *phase_of[slot];
      const int* counts = counts_of[slot];
      double spill = 0.0;
      for (size_t domain = 0; domain < domains; ++domain) {
        if (counts[domain] > 0) {
          spill += static_cast<double>(counts[domain]) / phase.active_threads * domain_spill[domain];
        }
      }
      dram_bytes[slot] = phase.compulsory_dram_bytes + spill * phase.spillable_dram_bytes;
      dram_ns[slot] = dram_bytes[slot] / phase.dram_rate * 1e9;
      gemm_threads += phase.gemm_demand > 0.0 ? phase.active_threads : 0;
      l2_threads += phase.l2_demand > 0.0 ? phase.active_threads : 0;
      dram_rank_threads += dram_bytes[slot] > 0.0 ? phase.active_threads : 0;
      epilogue_threads += phase.epilogue_demand > 0.0 ? phase.active_threads : 0;
    }
    const auto capacity_of = [&](const std::vector<double>& rates, int threads) {
      threads = std::min(threads, cores);
      return threads > 0 ? rates[static_cast<size_t>(threads)] : kInfinity;
    };
    const double gemm_capacity = capacity_of(machine_.gemm_rate, gemm_threads);
    const double l2_capacity = capacity_of(machine_.l2_rate, l2_threads);
    const double dram_capacity = capacity_of(machine_.dram_rate, dram_rank_threads);
    const double epilogue_capacity = capacity_of(machine_.epilogue_rate, epilogue_threads);

    // Offered rates: one sequential sum per resource, plus LLC and DRAM injection by domain.
    double gemm_offered = 0.0, l2_offered = 0.0, dram_rank_offered = 0.0, epilogue_offered = 0.0;
    std::fill(requesting.begin(), requesting.end(), 0);
    std::fill(llc_offered.begin(), llc_offered.end(), 0.0);
    std::fill(dram_threads.begin(), dram_threads.end(), 0);
    std::fill(dram_offered.begin(), dram_offered.end(), 0.0);
    for (size_t slot = 0; slot < slots; ++slot) {
      const AnalyticDagPhase& phase = *phase_of[slot];
      const double residual = phase.residual_scale;
      if (phase.gemm_demand > 0.0) {
        gemm_offered += phase.gemm_offered;
      }
      if (phase.l2_demand > 0.0) {
        l2_offered += phase.l2_offered;
      }
      if (dram_bytes[slot] > 0.0) {
        dram_rank_offered += OfferedRate(dram_bytes[slot], residual, dram_ns[slot]);
      }
      if (phase.epilogue_demand > 0.0) {
        epilogue_offered += phase.epilogue_offered;
      }
      const int* counts = counts_of[slot];
      for (size_t domain = 0; domain < domains; ++domain) {
        const int in_domain = counts[domain];
        if (in_domain <= 0) {
          continue;
        }
        const double share = static_cast<double>(in_domain) / phase.active_threads;
        if (phase.llc_demand > 0.0) {
          llc_offered[domain] += phase.llc_demand * share / phase.llc_service_seconds;
          requesting[domain] += in_domain;
        }
        if (machine_.dram_injection && dram_bytes[slot] > 0.0) {
          dram_offered[domain] +=
              dram_bytes[slot] * share / std::max(residual * dram_ns[slot] * 1e-9, kMinimumServiceSeconds);
          dram_threads[domain] += in_domain;
        }
      }
    }
    // A resource with no requesting thread has infinite capacity, so its sum is unused.
    const double gemm_dilation = Dilation(gemm_offered, gemm_capacity);
    const double l2_dilation = Dilation(l2_offered, l2_capacity);
    const double dram_rank_dilation = Dilation(dram_rank_offered, dram_capacity);
    const double epilogue_dilation = Dilation(epilogue_offered, epilogue_capacity);

    double total_llc_offered = 0.0;
    int total_requesting = 0, requesting_domains = 0;
    double requesting_rate = 0.0;
    for (size_t domain = 0; domain < domains; ++domain) {
      const int threads = requesting[domain];
      const double capacity =
          threads > 0 ? machine_.domain_llc_rate[domain][static_cast<size_t>(threads)] : kInfinity;
      llc_dilation[domain] = Dilation(llc_offered[domain], capacity);
      total_llc_offered += llc_offered[domain];
      if (threads > 0) {
        total_requesting += threads;
        ++requesting_domains;
        requesting_rate += capacity;
      }
      if (machine_.dram_injection) {
        double injection_capacity = kInfinity;
        if (dram_threads[domain] > 0) {
          const double active_capacity =
              machine_.dram_rate[static_cast<size_t>(std::min(dram_threads[domain], cores))];
          const double equal_share = machine_.dram_saturated_rate / static_cast<double>(domains);
          injection_capacity = std::min(active_capacity, machine_.dram_injection_capacity_scale * equal_share);
        }
        dram_dilation[domain] = Dilation(dram_offered[domain], injection_capacity);
      }
    }
    const double rank_llc_capacity =
        total_requesting > 0
            ? (requesting_domains == 1 ? requesting_rate : std::min(requesting_rate, machine_.llc_saturated_rate))
            : kInfinity;
    const double rank_llc_dilation = Dilation(total_llc_offered, rank_llc_capacity);

    // ECM composition, then the team-width pressure terms.
    double elapsed = kInfinity;
    for (size_t slot = 0; slot < slots; ++slot) {
      const AnalyticDagPhase& phase = *phase_of[slot];
      const int* counts = counts_of[slot];
      double llc_scale = 1.0;
      double dram_scale = dram_bytes[slot] > 0.0 ? dram_rank_dilation : 1.0;
      if (phase.llc_demand > 0.0 || (machine_.dram_injection && dram_bytes[slot] > 0.0)) {
        double llc_local = 0.0, dram_local = 0.0;
        bool any = false;
        for (size_t domain = 0; domain < domains; ++domain) {
          if (counts[domain] > 0) {
            llc_local = any ? std::max(llc_local, llc_dilation[domain]) : llc_dilation[domain];
            dram_local = any ? std::max(dram_local, dram_dilation[domain]) : dram_dilation[domain];
            any = true;
          }
        }
        if (!any) {
          llc_local = dram_local = 1.0;
        }
        if (phase.llc_demand > 0.0) {
          llc_scale = std::max(rank_llc_dilation, llc_local);
        }
        if (machine_.dram_injection && dram_bytes[slot] > 0.0) {
          dram_scale = std::max(dram_scale, dram_local);
        }
      }
      const double gemm_ns = phase.gemm_ns * (phase.gemm_demand > 0.0 ? gemm_dilation : 1.0);
      const double transfer_ns = std::max({phase.l2_ns * (phase.l2_demand > 0.0 ? l2_dilation : 1.0),
                                           phase.llc_ns * llc_scale, dram_ns[slot] * dram_scale});
      const double epilogue_ns = phase.epilogue_ns * (phase.epilogue_demand > 0.0 ? epilogue_dilation : 1.0);
      const double duration = phase.residual_scale * (phase.fixed_ns + std::max(gemm_ns, transfer_ns) + epilogue_ns);
      double multiplier = duration / phase.base_ns;
      const int width = width_of[slot];
      const int available = cores - width;
      const int peers = std::max(occupied - width, 0);
      const double peer_fraction =
          available > 0 ? std::min(static_cast<double>(peers) / static_cast<double>(available), 1.0) : 0.0;
      double dilation = 1.0;
      double correction = 1.0;
      if (phase.gemm) {
        const double isolated = machine_.wide_isolated_scale[static_cast<size_t>(width)];
        const double full = machine_.wide_full_cohort_scale[static_cast<size_t>(width)];
        dilation = isolated + (full - isolated) * peer_fraction;
        correction = 1.0 + (machine_.narrow_full_cohort_correction[static_cast<size_t>(width)] - 1.0) * peer_fraction;
      }
      multiplier = std::max(1.0, multiplier * dilation * correction);
      multipliers[slot] = multiplier;
      elapsed = std::min(elapsed, remaining[static_cast<size_t>(active[slot])] * multiplier);
    }

    wall_ns += elapsed;
    started_now.clear();
    size_t kept = 0;
    for (size_t slot = 0; slot < slots; ++slot) {
      const size_t task = static_cast<size_t>(active[slot]);
      remaining[task] -= elapsed / multipliers[slot];
      bool done = false;
      if (remaining[task] <= kCompletionToleranceNs) {
        ++phase_index[task];
        if (phase_index[task] < task_phases[task]->size()) {
          remaining[task] = (*task_phases[task])[phase_index[task]].base_ns;
        } else {
          done = true;
          ++finished_count;
          for (int successor : successors[task]) {
            if (--waiting[static_cast<size_t>(successor)] == 0) {
              started_now.push_back(successor);
            }
          }
        }
      }
      if (!done) {
        active[kept++] = active[slot];
      }
    }
    active.resize(kept);
    for (int task : started_now) {
      active.insert(std::lower_bound(active.begin(), active.end(), task), task);
    }
  }
  return wall_ns;
}

double NativeAnalyticPlacedDag::makespan_aggregate(const AnalyticDagTasks& tasks,
                                                    std::vector<double>* finish_times) const {
  const size_t count = tasks.routes.size();
  if (tasks.threads.size() != count || tasks.dependencies.size() != count) {
    throw std::invalid_argument("DAG task arrays must have equal lengths");
  }
  if (finish_times != nullptr) {
    finish_times->assign(count, 0.0);
  }
  if (count == 0) {
    return 0.0;
  }
  const int cores = machine_.cores_per_rank;
  std::vector<const std::vector<AnalyticDagPhase>*> task_phases(count);
  size_t max_events = count + 2;
  std::vector<int> waiting(count);
  std::vector<std::vector<int>> successors(count);
  for (size_t task = 0; task < count; ++task) {
    const auto found = phases_.find({tasks.routes[task], tasks.threads[task]});
    if (found == phases_.end()) {
      throw std::out_of_range("phases are not registered for routes=" + std::to_string(tasks.routes[task]) +
                              " threads=" + std::to_string(tasks.threads[task]));
    }
    task_phases[task] = &found->second;
    max_events += found->second.size();
    // _dag_state keeps duplicate dependencies: each one is counted and released.
    waiting[task] = static_cast<int>(tasks.dependencies[task].size());
    for (int dependency : tasks.dependencies[task]) {
      if (dependency < 0 || dependency >= static_cast<int>(task)) {
        throw std::invalid_argument("task dependencies must refer to earlier tasks: task=" + std::to_string(task) +
                                    ", dependency=" + std::to_string(dependency));
      }
      successors[static_cast<size_t>(dependency)].push_back(static_cast<int>(task));
    }
  }
  std::vector<size_t> phase_index(count, 0);
  std::vector<double> remaining(count);
  std::vector<int> active;
  active.reserve(count);
  for (size_t task = 0; task < count; ++task) {
    remaining[task] = (*task_phases[task])[0].base_ns;
    if (waiting[task] == 0) {
      active.push_back(static_cast<int>(task));
    }
  }
  std::vector<const AnalyticDagPhase*> phase_of(count);
  std::vector<double> dram_bytes(count), dram_ns(count), multipliers(count);
  std::vector<int> started_now;
  const double physical_llc = machine_.rank_llc_capacity_bytes;
  const double effective_llc = physical_llc * machine_.llc_effective_fraction;

  double wall_ns = machine_.call_setup_ns;
  size_t finished_count = 0;
  size_t guard = 0;
  while (finished_count < count) {
    if (++guard > 2 * max_events) {
      throw std::runtime_error("analytical stage simulator did not converge");
    }
    const size_t slots = active.size();
    if (slots == 0) {
      throw std::invalid_argument("DAG deadlock (cycle or unreachable task)");
    }
    double working_set = 0.0;
    for (size_t slot = 0; slot < slots; ++slot) {
      const size_t task = static_cast<size_t>(active[slot]);
      phase_of[slot] = &(*task_phases[task])[phase_index[task]];
      working_set += phase_of[slot]->working_set_bytes;
    }
    const double spill = SmoothCapacityMiss(working_set, effective_llc, physical_llc);

    int gemm_threads = 0, l2_threads = 0, llc_threads = 0, dram_threads = 0, epilogue_threads = 0;
    for (size_t slot = 0; slot < slots; ++slot) {
      const AnalyticDagPhase& phase = *phase_of[slot];
      dram_bytes[slot] = phase.compulsory_dram_bytes + spill * phase.spillable_dram_bytes;
      dram_ns[slot] = dram_bytes[slot] / phase.dram_rate * 1e9;
      gemm_threads += phase.gemm_demand > 0.0 ? phase.active_threads : 0;
      l2_threads += phase.l2_demand > 0.0 ? phase.active_threads : 0;
      llc_threads += phase.llc_demand > 0.0 ? phase.active_threads : 0;
      dram_threads += dram_bytes[slot] > 0.0 ? phase.active_threads : 0;
      epilogue_threads += phase.epilogue_demand > 0.0 ? phase.active_threads : 0;
    }
    const auto capacity_of = [&](const std::vector<double>& rates, int threads) {
      threads = std::min(threads, cores);
      return threads > 0 ? rates[static_cast<size_t>(threads)] : kInfinity;
    };
    double gemm_offered = 0.0, l2_offered = 0.0, llc_offered = 0.0, dram_offered = 0.0, epilogue_offered = 0.0;
    for (size_t slot = 0; slot < slots; ++slot) {
      const AnalyticDagPhase& phase = *phase_of[slot];
      if (phase.gemm_demand > 0.0) {
        gemm_offered += phase.gemm_offered;
      }
      if (phase.l2_demand > 0.0) {
        l2_offered += phase.l2_offered;
      }
      if (phase.llc_demand > 0.0) {
        llc_offered += phase.llc_demand / phase.llc_service_seconds;
      }
      if (dram_bytes[slot] > 0.0) {
        dram_offered += OfferedRate(dram_bytes[slot], phase.residual_scale, dram_ns[slot]);
      }
      if (phase.epilogue_demand > 0.0) {
        epilogue_offered += phase.epilogue_offered;
      }
    }
    // One dilation per resource, applied to every active phase (the rank-aggregate model).
    const double gemm_dilation = Dilation(gemm_offered, capacity_of(machine_.gemm_rate, gemm_threads));
    const double l2_dilation = Dilation(l2_offered, capacity_of(machine_.l2_rate, l2_threads));
    const double llc_dilation = Dilation(llc_offered, capacity_of(machine_.llc_rate, llc_threads));
    const double dram_dilation = Dilation(dram_offered, capacity_of(machine_.dram_rate, dram_threads));
    const double epilogue_dilation = Dilation(epilogue_offered, capacity_of(machine_.epilogue_rate, epilogue_threads));

    double elapsed = kInfinity;
    for (size_t slot = 0; slot < slots; ++slot) {
      const AnalyticDagPhase& phase = *phase_of[slot];
      const double transfer_ns =
          std::max({phase.l2_ns * l2_dilation, phase.llc_ns * llc_dilation, dram_ns[slot] * dram_dilation});
      const double body_ns = std::max(phase.gemm_ns * gemm_dilation, transfer_ns);
      const double duration = phase.residual_scale * (phase.fixed_ns + body_ns + phase.epilogue_ns * epilogue_dilation);
      multipliers[slot] = duration / phase.base_ns;
      elapsed = std::min(elapsed, remaining[static_cast<size_t>(active[slot])] * multipliers[slot]);
    }

    wall_ns += elapsed;
    started_now.clear();
    size_t kept = 0;
    for (size_t slot = 0; slot < slots; ++slot) {
      const size_t task = static_cast<size_t>(active[slot]);
      remaining[task] -= elapsed / multipliers[slot];
      bool done = false;
      if (remaining[task] <= kCompletionToleranceNs) {
        ++phase_index[task];
        if (phase_index[task] < task_phases[task]->size()) {
          remaining[task] = (*task_phases[task])[phase_index[task]].base_ns;
        } else {
          done = true;
          ++finished_count;
          if (finish_times != nullptr) {
            (*finish_times)[task] = wall_ns;
          }
          for (int successor : successors[task]) {
            if (--waiting[static_cast<size_t>(successor)] == 0) {
              started_now.push_back(successor);
            }
          }
        }
      }
      if (!done) {
        active[kept++] = active[slot];
      }
    }
    active.resize(kept);
    for (int task : started_now) {
      active.insert(std::lower_bound(active.begin(), active.end(), task), task);
    }
  }
  return wall_ns;
}

std::vector<double> NativeAnalyticPlacedDag::makespans(const std::vector<AnalyticDagTasks>& batch, int workers) const {
  std::vector<double> result(batch.size(), 0.0);
  ParallelFor(batch.size(), workers, [&](size_t index) { result[index] = makespan(batch[index]); });
  return result;
}

}  // namespace moe_planner
