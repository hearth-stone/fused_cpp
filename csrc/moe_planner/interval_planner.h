#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace moe_planner {

struct IntervalIsoEntry {
  int routes = 0;
  int threads = 0;
  double nanoseconds = 0.0;
};

struct IntervalDerateEntry {
  int experts = 0;
  int routes = 0;
  int max_threads = 0;
  double value = 1.0;
};

struct IntervalShapeCurveEntry {
  std::vector<int> shape;
  int routes = 0;
  double value = 0.0;
};

struct IntervalIsoFormulaConfig {
  bool enabled = false;
  double o0 = 0.0;
  double o1 = 0.0;
  double alpha = 0.0;
  double beta = 0.0;
  std::vector<std::pair<int, double>> route_work;
  std::vector<std::pair<int, double>> measured_phi;
};

struct IntervalTailRepartitionEntry {
  std::vector<int> root_shape;
  int tail_width = 0;
  int route_slices = 1;
  int routes = 0;
  double median_ns = 0.0;
  double p10_ns = 0.0;
  double p90_ns = 0.0;
  int num_iters = 1;
};

struct IntervalCostModelConfig {
  int schema_version = 0;
  bool exact_m = false;
  bool use_formula_iso = false;
  bool use_max_team_derate = false;
  bool use_shape_derate = false;
  bool use_stage_model = false;
  bool has_full_workload_anchors = false;
  int local_experts = 0;
  int profile_runs = 1;
  int measurement_experts = 0;
  int64_t w13_stage_bytes = 0;
  int64_t w2_stage_bytes = 0;
  int64_t max_stage_bytes = 0;
  int64_t w13_tile_bytes = 0;
  int64_t w2_tile_bytes = 0;
  double call_setup_ns = 0.0;
  std::vector<IntervalIsoEntry> isolated;
  std::vector<std::pair<int, double>> overheads;
  IntervalIsoFormulaConfig iso_formula;
  std::vector<IntervalDerateEntry> derate_2d;
  std::vector<IntervalDerateEntry> derate_3d;
  std::vector<IntervalShapeCurveEntry> shape_derate;
  std::vector<IntervalShapeCurveEntry> group_curves;
  std::vector<IntervalShapeCurveEntry> full_call_curves;
  std::vector<IntervalShapeCurveEntry> p10_curves;
  std::vector<IntervalShapeCurveEntry> p90_curves;
  std::vector<IntervalShapeCurveEntry> full_call_p10_curves;
  std::vector<IntervalShapeCurveEntry> full_call_p90_curves;
  std::vector<IntervalTailRepartitionEntry> tail_repartition_anchors;
};

struct IntervalTask {
  int expert_id = 0;
  int routes = 0;
  int core_begin = 0;
  int threads = 0;
  std::vector<int> dependencies;
};

enum class IntervalExecutionMode {
  kStrict,
  kTailPool,
};

enum class IntervalAssignmentOrder {
  kLpt,
  kReverseOdd,
  kReverseEven,
};

struct IntervalCandidate {
  std::vector<int> shape;
  IntervalAssignmentOrder assignment_order = IntervalAssignmentOrder::kLpt;
  IntervalExecutionMode execution_mode = IntervalExecutionMode::kStrict;
  std::optional<int> tail_pool_threads;
  std::optional<int> tail_pool_max_routes;
  int tail_pool_tasks = 0;
  std::optional<int> tail_repartition_width;
  int tail_repartition_tasks = 0;
  int tail_repartition_route_slices = 1;
  double makespan_ns = 0.0;
  double uncertainty_ns = 0.0;
  double pessimistic_ns = 0.0;
  std::vector<IntervalTask> tasks;
  int64_t active_working_set_bytes = 0;
  std::vector<int64_t> window_bytes_per_worker;
  int resource_groups = 0;
};

struct IntervalPlanResult {
  IntervalCandidate selected;
  std::vector<IntervalCandidate> candidates;
  int configured_workers = 1;
  int strict_candidates = 0;
  int dynamic_candidates = 0;
  int tail_repartition_candidates = 0;
};

// Native implementation of the Python IntervalPlanner cold search. The object
// owns immutable calibration data and is safe to call repeatedly. Candidate
// evaluation is parallelized; each candidate's event simulation remains local
// to one worker.
class NativeIntervalPlanner {
 public:
  struct Impl;

  NativeIntervalPlanner(int num_cores, std::vector<int> widths, std::vector<std::vector<int>> shapes,
                        IntervalCostModelConfig model, int planner_threads,
                        std::vector<int> tail_repartition_widths = {});

  NativeIntervalPlanner(const NativeIntervalPlanner&) = delete;
  NativeIntervalPlanner& operator=(const NativeIntervalPlanner&) = delete;

  IntervalPlanResult Plan(const std::vector<int>& expert_ids, const std::vector<int>& routes, bool dynamic_tail_pool,
                          int tail_pool_max_routes, std::optional<int> forced_tail_pool_threads,
                          bool bounded_tail_repartition) const;

  double EstimateIsolated(int routes, int threads) const;
  double ScoreDag(const std::vector<int>& routes, const std::vector<int>& threads,
                  const std::vector<std::vector<int>>& dependencies) const;

  int configured_workers() const { return configured_workers_; }

 private:
  int num_cores_ = 0;
  std::vector<int> widths_;
  std::vector<int> tail_repartition_widths_;
  std::vector<std::vector<int>> shapes_;
  int configured_workers_ = 1;
  std::unique_ptr<Impl> impl_;

 public:
  ~NativeIntervalPlanner();
};

}  // namespace moe_planner
