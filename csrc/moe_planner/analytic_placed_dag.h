// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace moe_planner {

// One analytic phase of an expert task, as AnalyticPhase carries it. Only the
// fields the placed contention simulator reads are kept; the Python model
// computes them (and base_ns) so both paths start from identical values.
struct AnalyticDagPhase {
  bool gemm = false;  // kind in {"cold_b", "steady_b"}
  int active_threads = 0;
  double fixed_ns = 0.0;
  double residual_scale = 1.0;
  double base_ns = 0.0;
  double working_set_bytes = 0.0;
  double gemm_demand = 0.0;  // matrix_flops, charged to gemm_core_flops
  double l2_demand = 0.0;
  double llc_demand = 0.0;
  double epilogue_demand = 0.0;
  double compulsory_dram_bytes = 0.0;
  double spillable_dram_bytes = 0.0;
  double dram_rate = 1.0;
  double gemm_ns = 0.0;
  double l2_ns = 0.0;
  double llc_ns = 0.0;
  double epilogue_ns = 0.0;
  // Derived at registration; each equals the expression the simulator would
  // otherwise evaluate on every event, so results are unchanged.
  double gemm_offered = 0.0;
  double l2_offered = 0.0;
  double epilogue_offered = 0.0;
  double llc_service_seconds = 0.0;
};

// Machine terms of AnalyticMachineCalibration the simulator needs, tabulated by
// integer thread count so no curve is evaluated inside the event loop.
struct AnalyticDagMachine {
  int cores_per_rank = 0;
  double call_setup_ns = 0.0;
  std::vector<int> cpu_domain;  // indexed by physical CPU id, -1 outside the rank
  std::vector<int> domain_sizes;
  std::vector<double> domain_capacity_bytes;
  double llc_effective_fraction = 1.0;
  // Rank service rate by active threads 0..cores_per_rank (entry 0 unused);
  // +inf when the calibration has no curve for the resource.
  std::vector<double> gemm_rate;
  std::vector<double> l2_rate;
  std::vector<double> dram_rate;
  std::vector<double> epilogue_rate;
  std::vector<std::vector<double>> domain_llc_rate;  // [domain][threads 0..size]
  std::vector<double> llc_rate;                      // rank curve, used by the aggregate simulator
  double llc_saturated_rate = 0.0;
  double rank_llc_capacity_bytes = 0.0;
  bool dram_injection = false;
  double dram_injection_capacity_scale = 1.0;
  double dram_saturated_rate = 0.0;
  // Rank DRAM capacity when two or more concurrent phases demand DRAM, by active threads
  // 0..cores_per_rank; empty disables the term (a lone phase always keeps dram_rate).
  std::vector<double> dram_multi_stream_rate;
  // Per team width (index = width, 0..cores_per_rank).
  std::vector<double> wide_isolated_scale;
  std::vector<double> wide_full_cohort_scale;
  std::vector<double> narrow_full_cohort_correction;
};

struct AnalyticDagTasks {
  std::vector<int64_t> routes;
  std::vector<int> threads;
  std::vector<std::vector<int>> cpu_ids;
  std::vector<std::vector<int>> dependencies;
};

// C++ port of AnalyticMoeCostModel's analytic event simulator (_dag_result):
// the placement-aware form (task_cpu_ids given; _active_phase_state_placed and
// _normalize_placed_tasks) and the rank-aggregate form (_active_phase_state).
class NativeAnalyticPlacedDag {
 public:
  explicit NativeAnalyticPlacedDag(AnalyticDagMachine machine);

  void register_phases(int64_t routes, int threads, std::vector<AnalyticDagPhase> phases);
  bool has_phases(int64_t routes, int threads) const;
  // Placement-aware; requires calibrated LLC domains.
  double makespan(const AnalyticDagTasks& tasks) const;
  // Scores independent DAGs concurrently; each simulation itself is sequential.
  std::vector<double> makespans(const std::vector<AnalyticDagTasks>& batch, int workers) const;
  // Rank-aggregate; ignores cpu_ids. Fills per-task finish times when asked.
  double makespan_aggregate(const AnalyticDagTasks& tasks, std::vector<double>* finish_times = nullptr) const;

 private:
  // Rank DRAM capacity for `streams` concurrent DRAM-demanding phases (multi-stream term).
  double ContendedDramCapacity(double capacity, int threads, int streams) const {
    if (machine_.dram_multi_stream_rate.empty() || streams < 2 || threads <= 0) {
      return capacity;
    }
    const int capped = threads < machine_.cores_per_rank ? threads : machine_.cores_per_rank;
    const double multi = machine_.dram_multi_stream_rate[static_cast<size_t>(capped)];
    return multi < capacity ? multi : capacity;
  }

  AnalyticDagMachine machine_;
  std::map<std::pair<int64_t, int>, std::vector<AnalyticDagPhase>> phases_;
};

}  // namespace moe_planner
