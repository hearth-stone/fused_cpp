// pybind11 binding for the MoE schedule planners.
//
// Exposes a single entry point `moe_schedule_plan` returning the scheduled
// bridge arrays plus diagnostics, so it can be (a) fed straight into
// fused_moe_bf16_tiled_scheduled and (b) equivalence-tested against
// cpu_moe_schedule_optimization/planners/offline_simulator.py.
//
// This is the only planner translation unit that depends on libtorch /
// pybind11; the planner core stays torch-free.
#include <torch/extension.h>

#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "exact_solver.h"
#include "interval_planner.h"
#include "planner_dispatch.h"

namespace py = pybind11;

namespace {

using namespace moe_planner;

PlanKind parse_kind(const std::string& s) {
  // Canonical names and the common aliases used by the Python CLI.
  if (s == "FIXED_GLOBAL_THREADS" || s == "fixed" || s == "fixed_global_threads") return PlanKind::FIXED_GLOBAL_THREADS;
  if (s == "SORTED_TOKEN_BALANCED_1T" || s == "balanced" || s == "lpt" || s == "sorted_token_balanced_1t" ||
      s == "token_balanced")
    return PlanKind::SORTED_TOKEN_BALANCED_1T;
  if (s == "UNIFORM_WAVES" || s == "uniform" || s == "uniform_waves") return PlanKind::UNIFORM_WAVES;
  if (s == "GREEDY_MARGINAL_GAIN" || s == "greedy" || s == "greedy_marginal_gain")
    return PlanKind::GREEDY_MARGINAL_GAIN;
  if (s == "ENUMERATE_CORE_GROUPS" || s == "groups" || s == "enumerate" || s == "core_groups" ||
      s == "enumerate_core_groups")
    return PlanKind::ENUMERATE_CORE_GROUPS;
  if (s == "LOAD_PROPORTIONAL" || s == "proportional" || s == "load_proportional") return PlanKind::LOAD_PROPORTIONAL;
  if (s == "SQRT_LOAD" || s == "sqrt" || s == "sqrt_load") return PlanKind::SQRT_LOAD;
  if (s == "LOG_LOAD" || s == "log" || s == "log_load") return PlanKind::LOG_LOAD;
  if (s == "HEAVY_LIGHT_HYBRID" || s == "heavy_light" || s == "hybrid" || s == "heavy_light_hybrid")
    return PlanKind::HEAVY_LIGHT_HYBRID;
  if (s == "KARMARKAR_KARP" || s == "kk" || s == "karmarkar_karp") return PlanKind::KARMARKAR_KARP;
  throw std::invalid_argument("unknown planner kind: " + s);
}

CostModel build_cost_model(const c10::optional<std::vector<int64_t>>& route_buckets,
                           const c10::optional<std::vector<int64_t>>& thread_buckets,
                           const c10::optional<std::vector<int64_t>>& table_values) {
  if (!route_buckets.has_value()) return CostModel::synthetic();
  std::vector<int> rb(route_buckets->begin(), route_buckets->end());
  std::vector<int> tb(thread_buckets->begin(), thread_buckets->end());
  std::vector<int64_t> vals(table_values->begin(), table_values->end());
  if (vals.size() != rb.size() * tb.size())
    throw std::invalid_argument("table_values size must equal len(route_buckets) * len(thread_buckets)");
  return CostModel::from_table(std::move(rb), std::move(tb), std::move(vals));
}

int64_t elapsed_ns(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}

int64_t median_ns(std::vector<int64_t> values) {
  if (values.empty()) return 0;
  const size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  int64_t hi = values[mid];
  if ((values.size() % 2) != 0) return hi;
  std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
  return (hi + values[mid - 1]) / 2;
}

int64_t percentile_ns(std::vector<int64_t> values, double percentile) {
  if (values.empty()) return 0;
  if (percentile <= 0.0) return *std::min_element(values.begin(), values.end());
  if (percentile >= 1.0) return *std::max_element(values.begin(), values.end());
  const size_t idx = static_cast<size_t>(
      std::min<double>(values.size() - 1, std::max<double>(0.0, percentile * (values.size() - 1) + 0.5)));
  std::nth_element(values.begin(), values.begin() + idx, values.end());
  return values[idx];
}

template <typename T>
T required_value(const py::dict& payload, const char* key) {
  if (!payload.contains(key)) {
    throw std::invalid_argument(std::string("native planner payload is missing ") + key);
  }
  return py::cast<T>(payload[key]);
}

std::vector<std::pair<int, double>> parse_pairs(const py::handle& values) {
  std::vector<std::pair<int, double>> result;
  for (const py::handle row_handle : py::reinterpret_borrow<py::iterable>(values)) {
    const py::sequence row = py::reinterpret_borrow<py::sequence>(row_handle);
    if (py::len(row) != 2) {
      throw std::invalid_argument("native planner pair rows must have two fields");
    }
    result.emplace_back(py::cast<int>(row[0]), py::cast<double>(row[1]));
  }
  return result;
}

std::vector<IntervalIsoEntry> parse_iso_entries(const py::handle& values) {
  std::vector<IntervalIsoEntry> result;
  for (const py::handle row_handle : py::reinterpret_borrow<py::iterable>(values)) {
    const py::sequence row = py::reinterpret_borrow<py::sequence>(row_handle);
    if (py::len(row) != 3) {
      throw std::invalid_argument("native planner isolated rows must have three fields");
    }
    result.push_back({
        py::cast<int>(row[0]),
        py::cast<int>(row[1]),
        py::cast<double>(row[2]),
    });
  }
  return result;
}

std::vector<IntervalDerateEntry> parse_derate_entries(const py::handle& values, bool has_team) {
  std::vector<IntervalDerateEntry> result;
  for (const py::handle row_handle : py::reinterpret_borrow<py::iterable>(values)) {
    const py::sequence row = py::reinterpret_borrow<py::sequence>(row_handle);
    const size_t expected = has_team ? 4 : 3;
    if (py::len(row) != expected) {
      throw std::invalid_argument("native planner derate row has an invalid field count");
    }
    result.push_back({
        py::cast<int>(row[0]),
        py::cast<int>(row[1]),
        has_team ? py::cast<int>(row[2]) : 0,
        py::cast<double>(row[expected - 1]),
    });
  }
  return result;
}

std::vector<IntervalShapeCurveEntry> parse_shape_curve_entries(const py::handle& values) {
  std::vector<IntervalShapeCurveEntry> result;
  for (const py::handle row_handle : py::reinterpret_borrow<py::iterable>(values)) {
    const py::sequence row = py::reinterpret_borrow<py::sequence>(row_handle);
    if (py::len(row) != 3) {
      throw std::invalid_argument("native planner shape-curve rows must have three fields");
    }
    result.push_back({
        py::cast<std::vector<int>>(row[0]),
        py::cast<int>(row[1]),
        py::cast<double>(row[2]),
    });
  }
  return result;
}

std::vector<IntervalTailRepartitionEntry> parse_tail_repartition_entries(const py::handle& values) {
  std::vector<IntervalTailRepartitionEntry> result;
  for (const py::handle row_handle : py::reinterpret_borrow<py::iterable>(values)) {
    const py::sequence row = py::reinterpret_borrow<py::sequence>(row_handle);
    if (py::len(row) != 8) {
      throw std::invalid_argument("native planner bounded-tail anchor rows must have eight fields");
    }
    result.push_back({
        py::cast<std::vector<int>>(row[0]),
        py::cast<int>(row[1]),
        py::cast<int>(row[2]),
        py::cast<int>(row[3]),
        py::cast<double>(row[4]),
        py::cast<double>(row[5]),
        py::cast<double>(row[6]),
        py::cast<int>(row[7]),
    });
  }
  return result;
}

IntervalIsoFormulaConfig parse_iso_formula(const py::handle& value) {
  IntervalIsoFormulaConfig result;
  if (value.is_none()) {
    return result;
  }
  const py::dict payload = py::reinterpret_borrow<py::dict>(value);
  result.enabled = true;
  result.o0 = required_value<double>(payload, "o0");
  result.o1 = required_value<double>(payload, "o1");
  result.alpha = required_value<double>(payload, "alpha");
  result.beta = required_value<double>(payload, "beta");
  result.route_work = parse_pairs(payload["c_pts"]);
  result.measured_phi = parse_pairs(payload["phi_pts"]);
  return result;
}

IntervalCostModelConfig parse_interval_cost_model(const py::dict& payload) {
  IntervalCostModelConfig result;
  result.schema_version = required_value<int>(payload, "schema_version");
  result.exact_m = required_value<bool>(payload, "exact_m");
  result.use_formula_iso = required_value<bool>(payload, "use_formula_iso");
  result.use_max_team_derate = required_value<bool>(payload, "use_max_team_derate");
  result.use_shape_derate = required_value<bool>(payload, "use_shape_derate");
  result.use_stage_model = required_value<bool>(payload, "use_stage_model");
  result.has_full_workload_anchors = required_value<bool>(payload, "has_full_workload_anchors");
  result.local_experts = required_value<int>(payload, "local_experts");
  result.profile_runs = required_value<int>(payload, "profile_runs");
  result.measurement_experts = required_value<int>(payload, "measurement_experts");
  result.w13_stage_bytes = required_value<int64_t>(payload, "w13_stage_bytes");
  result.w2_stage_bytes = required_value<int64_t>(payload, "w2_stage_bytes");
  result.max_stage_bytes = required_value<int64_t>(payload, "max_stage_bytes");
  result.w13_tile_bytes = required_value<int64_t>(payload, "w13_tile_bytes");
  result.w2_tile_bytes = required_value<int64_t>(payload, "w2_tile_bytes");
  result.call_setup_ns = required_value<double>(payload, "call_setup_ns");
  result.isolated = parse_iso_entries(payload["isolated"]);
  result.overheads = parse_pairs(payload["overheads"]);
  result.iso_formula = parse_iso_formula(payload["iso_formula"]);
  result.derate_2d = parse_derate_entries(payload["derate_2d"], false);
  result.derate_3d = parse_derate_entries(payload["derate_3d"], true);
  result.shape_derate = parse_shape_curve_entries(payload["shape_derate"]);
  result.group_curves = parse_shape_curve_entries(payload["group_curves"]);
  result.full_call_curves = parse_shape_curve_entries(payload["full_call_curves"]);
  result.p10_curves = parse_shape_curve_entries(payload["p10_curves"]);
  result.p90_curves = parse_shape_curve_entries(payload["p90_curves"]);
  result.full_call_p10_curves = parse_shape_curve_entries(payload["full_call_p10_curves"]);
  result.full_call_p90_curves = parse_shape_curve_entries(payload["full_call_p90_curves"]);
  if (payload.contains("tail_repartition_anchors")) {
    result.tail_repartition_anchors = parse_tail_repartition_entries(payload["tail_repartition_anchors"]);
  }
  return result;
}

const char* interval_execution_mode_name(IntervalExecutionMode mode) {
  return mode == IntervalExecutionMode::kStrict ? "strict" : "tail_pool";
}

const char* interval_assignment_order_name(IntervalAssignmentOrder order) {
  switch (order) {
    case IntervalAssignmentOrder::kLpt:
      return "lpt";
    case IntervalAssignmentOrder::kReverseOdd:
      return "reverse_odd";
    case IntervalAssignmentOrder::kReverseEven:
      return "reverse_even";
  }
  throw std::invalid_argument("unknown interval assignment order");
}

py::list interval_tasks_to_python(const std::vector<IntervalTask>& tasks) {
  py::list result;
  for (const IntervalTask& task : tasks) {
    result.append(py::make_tuple(task.expert_id, task.routes, task.core_begin, task.threads, task.dependencies));
  }
  return result;
}

py::dict interval_candidate_to_python(const IntervalCandidate& candidate) {
  py::dict result;
  result["shape"] = candidate.shape;
  result["assignment_order"] = interval_assignment_order_name(candidate.assignment_order);
  result["execution_mode"] = interval_execution_mode_name(candidate.execution_mode);
  result["tail_pool_threads"] =
      candidate.tail_pool_threads.has_value() ? py::cast(*candidate.tail_pool_threads) : py::none();
  result["tail_pool_max_routes"] =
      candidate.tail_pool_max_routes.has_value() ? py::cast(*candidate.tail_pool_max_routes) : py::none();
  result["tail_pool_tasks"] = candidate.tail_pool_tasks;
  result["tail_repartition_width"] =
      candidate.tail_repartition_width.has_value() ? py::cast(*candidate.tail_repartition_width) : py::none();
  result["tail_repartition_tasks"] = candidate.tail_repartition_tasks;
  result["tail_repartition_route_slices"] = candidate.tail_repartition_route_slices;
  result["makespan_ns"] = candidate.makespan_ns;
  result["uncertainty_ns"] = candidate.uncertainty_ns;
  result["pessimistic_ns"] = candidate.pessimistic_ns;
  result["tasks"] = interval_tasks_to_python(candidate.tasks);
  result["active_working_set_bytes"] = candidate.active_working_set_bytes;
  result["window_bytes_per_worker"] = candidate.window_bytes_per_worker;
  result["resource_groups"] = candidate.resource_groups;
  return result;
}

py::dict native_interval_plan(const NativeIntervalPlanner& planner, const std::vector<int>& expert_ids,
                              const std::vector<int>& routes, bool dynamic_tail_pool, int tail_pool_max_routes,
                              std::optional<int> forced_tail_pool_threads,
                              std::optional<bool> bounded_tail_repartition) {
  const bool enable_bounded_tail_repartition =
      bounded_tail_repartition.value_or(dynamic_tail_pool && !forced_tail_pool_threads.has_value());
  IntervalPlanResult native_result;
  {
    py::gil_scoped_release release;
    native_result = planner.Plan(expert_ids, routes, dynamic_tail_pool, tail_pool_max_routes, forced_tail_pool_threads,
                                 enable_bounded_tail_repartition);
  }
  py::dict result;
  result["selected"] = interval_candidate_to_python(native_result.selected);
  py::list candidates;
  for (const IntervalCandidate& candidate : native_result.candidates) {
    candidates.append(interval_candidate_to_python(candidate));
  }
  result["candidates"] = std::move(candidates);
  result["configured_workers"] = native_result.configured_workers;
  result["strict_candidates"] = native_result.strict_candidates;
  result["dynamic_candidates"] = native_result.dynamic_candidates;
  result["tail_repartition_candidates"] = native_result.tail_repartition_candidates;
  return result;
}

// routes_hist: 1-D histogram (length num_experts). Returns a dict mirroring
// Plan.to_scheduled_bridge() plus active-expert view and cost estimate.
py::dict moe_schedule_plan(std::vector<int64_t> routes_hist, int64_t num_cores, const std::string& kind,
                           c10::optional<std::vector<int64_t>> route_buckets,
                           c10::optional<std::vector<int64_t>> thread_buckets,
                           c10::optional<std::vector<int64_t>> table_values) {
  if (num_cores <= 0) throw std::invalid_argument("num_cores must be positive");

  std::vector<int32_t> routes(routes_hist.size());
  for (size_t i = 0; i < routes_hist.size(); ++i) routes[i] = static_cast<int32_t>(routes_hist[i]);

  const CostModel cm = build_cost_model(route_buckets, thread_buckets, table_values);

  py::gil_scoped_release release;  // pure C++ work; let other threads run
  PlanResult r = plan_from_routes(parse_kind(kind), routes.data(), static_cast<int>(routes.size()),
                                  static_cast<int>(num_cores), cm);
  py::gil_scoped_acquire acquire;

  // thread_cpu_ids defaults to [0, num_cores) like Plan.to_scheduled_bridge.
  std::vector<int32_t> thread_cpu_ids(r.num_cores);
  for (int i = 0; i < r.num_cores; ++i) thread_cpu_ids[i] = i;

  py::dict d;
  d["kind"] = plan_kind_name(r.kind);
  d["num_cores"] = r.num_cores;
  d["num_threads"] = r.num_cores;
  d["active_expert_ids"] = r.active_expert_ids;
  d["active_routes"] = r.active_routes;
  d["thread_cpu_ids"] = thread_cpu_ids;
  d["wave_offsets"] = r.wave_offsets;
  d["team_expert_ids"] = r.team_expert_ids;
  d["team_threads"] = r.team_threads;
  d["estimated_execute_cost_ns"] = r.estimated_execute_cost_ns;
  d["num_waves"] = r.num_waves();
  d["selected_threads_per_expert"] = r.selected_threads_per_expert;
  d["selected_wave_budget"] = r.selected_wave_budget;
  d["selected_core_group_shape"] = r.selected_core_group_shape;
  return d;
}

py::dict moe_schedule_plan_timed(std::vector<int64_t> routes_hist, int64_t num_cores, const std::string& kind,
                                 int64_t warmup, int64_t iters, c10::optional<std::vector<int64_t>> route_buckets,
                                 c10::optional<std::vector<int64_t>> thread_buckets,
                                 c10::optional<std::vector<int64_t>> table_values) {
  if (num_cores <= 0) throw std::invalid_argument("num_cores must be positive");
  if (warmup < 0) throw std::invalid_argument("warmup must be non-negative");
  if (iters <= 0) throw std::invalid_argument("iters must be positive");

  std::vector<int32_t> routes(routes_hist.size());
  for (size_t i = 0; i < routes_hist.size(); ++i) routes[i] = static_cast<int32_t>(routes_hist[i]);

  const PlanKind parsed_kind = parse_kind(kind);
  const CostModel cm = build_cost_model(route_buckets, thread_buckets, table_values);

  std::vector<int64_t> prepare_samples;
  std::vector<int64_t> plan_samples;
  prepare_samples.reserve(static_cast<size_t>(iters));
  plan_samples.reserve(static_cast<size_t>(iters));

  PlanResult last_result;
  int last_active = 0;
  {
    py::gil_scoped_release release;
    for (int64_t i = 0; i < warmup + iters; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      Workload w = prepare_workload(routes.data(), static_cast<int>(routes.size()), static_cast<int>(num_cores), cm);
      auto t1 = std::chrono::steady_clock::now();
      PlanResult r = run_planner(parsed_kind, w);
      auto t2 = std::chrono::steady_clock::now();

      if (i >= warmup) {
        prepare_samples.push_back(elapsed_ns(t0, t1));
        plan_samples.push_back(elapsed_ns(t1, t2));
      }
      last_result = std::move(r);
      last_active = w.num_active;
    }
  }

  std::vector<int64_t> total_samples;
  total_samples.reserve(prepare_samples.size());
  int64_t prepare_sum = 0;
  int64_t plan_sum = 0;
  for (size_t i = 0; i < prepare_samples.size(); ++i) {
    prepare_sum += prepare_samples[i];
    plan_sum += plan_samples[i];
    total_samples.push_back(prepare_samples[i] + plan_samples[i]);
  }

  py::dict d;
  d["kind"] = plan_kind_name(last_result.kind);
  d["num_cores"] = last_result.num_cores;
  d["num_active"] = last_active;
  d["warmup"] = warmup;
  d["iters"] = iters;
  d["prepare_median_ns"] = median_ns(prepare_samples);
  d["prepare_mean_ns"] = prepare_sum / static_cast<int64_t>(prepare_samples.size());
  d["prepare_p90_ns"] = percentile_ns(prepare_samples, 0.90);
  d["prepare_p99_ns"] = percentile_ns(prepare_samples, 0.99);
  d["plan_median_ns"] = median_ns(plan_samples);
  d["plan_mean_ns"] = plan_sum / static_cast<int64_t>(plan_samples.size());
  d["plan_p90_ns"] = percentile_ns(plan_samples, 0.90);
  d["plan_p99_ns"] = percentile_ns(plan_samples, 0.99);
  d["total_native_median_ns"] = median_ns(total_samples);
  d["total_native_mean_ns"] = (prepare_sum + plan_sum) / static_cast<int64_t>(total_samples.size());
  d["total_native_p90_ns"] = percentile_ns(total_samples, 0.90);
  d["total_native_p99_ns"] = percentile_ns(total_samples, 0.99);
  d["estimated_execute_cost_ns"] = last_result.estimated_execute_cost_ns;
  d["num_waves"] = last_result.num_waves();
  d["num_teams"] = last_result.num_teams();
  d["selected_threads_per_expert"] = last_result.selected_threads_per_expert;
  d["selected_wave_budget"] = last_result.selected_wave_budget;
  d["selected_core_group_shape"] = last_result.selected_core_group_shape;
  return d;
}

}  // namespace

// Exact small-case optimum (subset DP) of the wave-barrier objective.
static py::dict moe_exact_optimum(std::vector<int64_t> routes_hist, int64_t num_cores, int64_t max_active,
                                  c10::optional<std::vector<int64_t>> route_buckets,
                                  c10::optional<std::vector<int64_t>> thread_buckets,
                                  c10::optional<std::vector<int64_t>> table_values) {
  if (num_cores <= 0) throw std::invalid_argument("num_cores must be positive");
  std::vector<int32_t> routes(routes_hist.size());
  for (size_t i = 0; i < routes_hist.size(); ++i) routes[i] = static_cast<int32_t>(routes_hist[i]);
  const moe_planner::CostModel cm = build_cost_model(route_buckets, thread_buckets, table_values);

  moe_planner::ExactResult res;
  {
    py::gil_scoped_release release;
    moe_planner::Workload w =
        moe_planner::prepare_workload(routes.data(), static_cast<int>(routes.size()), static_cast<int>(num_cores), cm);
    res = moe_planner::exact_optimum(w, static_cast<int>(max_active));
  }

  py::dict d;
  d["feasible"] = res.feasible;
  d["num_active"] = res.num_active;
  d["execute_ns"] = res.execute_ns;
  return d;
}

void register_moe_planner(py::module_& m) {
  py::class_<moe_planner::NativeIntervalPlanner>(m, "NativeIntervalPlanner")
      .def(py::init([](int num_cores, std::vector<int> widths, std::vector<std::vector<int>> shapes,
                       const py::dict& model_payload, int planner_threads, std::vector<int> tail_repartition_widths) {
             return std::make_unique<moe_planner::NativeIntervalPlanner>(
                 num_cores, std::move(widths), std::move(shapes), parse_interval_cost_model(model_payload),
                 planner_threads, std::move(tail_repartition_widths));
           }),
           py::arg("num_cores"), py::arg("widths"), py::arg("shapes"), py::arg("model_payload"),
           py::arg("planner_threads") = 0, py::arg("tail_repartition_widths") = std::vector<int>{})
      .def("plan", &native_interval_plan, py::arg("expert_ids"), py::arg("routes"), py::arg("dynamic_tail_pool") = true,
           py::arg("tail_pool_max_routes") = 12, py::arg("forced_tail_pool_threads") = std::nullopt,
           py::arg("bounded_tail_repartition") = std::nullopt)
      .def("_estimate_isolated", &moe_planner::NativeIntervalPlanner::EstimateIsolated)
      .def("_score_dag", &moe_planner::NativeIntervalPlanner::ScoreDag)
      .def_property_readonly("configured_workers", &moe_planner::NativeIntervalPlanner::configured_workers);

  m.def("moe_schedule_plan", &moe_schedule_plan,
        "CPU MoE schedule planner (FIXED / SORTED_TOKEN_BALANCED_1T / "
        "UNIFORM_WAVES / GREEDY_MARGINAL_GAIN / ENUMERATE_CORE_GROUPS). "
        "Returns scheduled-bridge arrays + execution-cost estimate.",
        py::arg("routes_hist"), py::arg("num_cores"), py::arg("kind"), py::arg("route_buckets") = c10::nullopt,
        py::arg("thread_buckets") = c10::nullopt, py::arg("table_values") = c10::nullopt);

  m.def("moe_schedule_plan_timed", &moe_schedule_plan_timed,
        "Time native C++ MoE planner work inside C++. The reported "
        "prepare_ns covers prepare_workload(routes -> active + dense cost "
        "rows); plan_ns covers run_planner() only; total_native_ns is their "
        "sum and excludes Python dict/list conversion overhead.",
        py::arg("routes_hist"), py::arg("num_cores"), py::arg("kind"), py::arg("warmup") = 100, py::arg("iters") = 1000,
        py::arg("route_buckets") = c10::nullopt, py::arg("thread_buckets") = c10::nullopt,
        py::arg("table_values") = c10::nullopt);

  m.def("moe_exact_optimum", &moe_exact_optimum,
        "Exact optimum (subset DP) of the wave-barrier execution objective "
        "for small active-expert counts. Returns {feasible, num_active, "
        "execute_ns}.",
        py::arg("routes_hist"), py::arg("num_cores"), py::arg("max_active") = 16,
        py::arg("route_buckets") = c10::nullopt, py::arg("thread_buckets") = c10::nullopt,
        py::arg("table_values") = c10::nullopt);
}
