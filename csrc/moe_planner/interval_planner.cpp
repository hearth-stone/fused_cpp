#include "interval_planner.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace moe_planner {
namespace {

constexpr double kCompletionTolerance = 1e-6;
constexpr int kAutoTailPoolMinHeadShapes = 2;
constexpr int kDefaultPlannerWorkers = 8;
constexpr int kAutoTailPoolThresholds[] = {1, 2, 4, 8, 12};

struct Expert {
  int expert_id = 0;
  int routes = 0;
  int original_index = 0;
};

struct Curve {
  std::vector<int> points;
  std::vector<double> values;

  bool empty() const { return points.empty(); }

  double InterpolateLinear(double x) const {
    if (points.empty()) {
      throw std::invalid_argument("cannot interpolate an empty curve");
    }
    if (x <= points.front()) {
      return values.front() * x / points.front();
    }
    if (x >= points.back()) {
      return values.back() * x / points.back();
    }
    const auto upper = std::lower_bound(points.begin(), points.end(), x);
    const size_t index = static_cast<size_t>(upper - points.begin());
    const double x0 = points[index - 1];
    const double x1 = points[index];
    const double y0 = values[index - 1];
    const double y1 = values[index];
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
  }

  double InterpolateLogRoute(int routes) const {
    if (points.empty()) {
      throw std::invalid_argument("cannot interpolate an empty curve");
    }
    routes = std::max(routes, 1);
    if (routes <= points.front()) {
      return values.front();
    }
    if (routes >= points.back()) {
      return values.back();
    }
    const auto upper = std::lower_bound(points.begin(), points.end(), routes);
    const size_t index = static_cast<size_t>(upper - points.begin());
    const double x = std::log2(static_cast<double>(routes));
    const double x0 = std::log2(static_cast<double>(points[index - 1]));
    const double x1 = std::log2(static_cast<double>(points[index]));
    const double y0 = values[index - 1];
    const double y1 = values[index];
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
  }
};

Curve MakeCurve(const std::map<int, double>& values) {
  Curve curve;
  curve.points.reserve(values.size());
  curve.values.reserve(values.size());
  for (const auto& [point, value] : values) {
    curve.points.push_back(point);
    curve.values.push_back(value);
  }
  return curve;
}

std::vector<int> ShapeSignature(std::vector<int> shape) {
  shape.erase(std::remove_if(shape.begin(), shape.end(), [](int value) { return value <= 0; }), shape.end());
  std::sort(shape.begin(), shape.end(), std::greater<int>());
  return shape;
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

int DefaultPlannerWorkers(int num_cores) {
  int hardware_threads = static_cast<int>(std::thread::hardware_concurrency());
#if defined(_OPENMP)
  hardware_threads = std::max(hardware_threads, omp_get_num_procs());
#endif
  if (hardware_threads <= 0) {
    hardware_threads = num_cores;
  }
  return std::max(1, std::min({kDefaultPlannerWorkers, num_cores, hardware_threads}));
}

bool CandidateLessByMakespan(const IntervalCandidate& left, const IntervalCandidate& right) {
  return left.makespan_ns < right.makespan_ns;
}

bool SelectionKeyLess(const IntervalCandidate& left, const IntervalCandidate& right) {
  return std::tie(left.active_working_set_bytes, left.resource_groups, left.execution_mode, left.pessimistic_ns,
                  left.makespan_ns) < std::tie(right.active_working_set_bytes, right.resource_groups,
                                               right.execution_mode, right.pessimistic_ns, right.makespan_ns);
}

IntervalCandidate SelectCandidate(std::vector<IntervalCandidate>* candidates) {
  if (candidates->empty()) {
    throw std::invalid_argument("cannot select from an empty candidate set");
  }
  std::stable_sort(candidates->begin(), candidates->end(), CandidateLessByMakespan);
  const IntervalCandidate& fastest = candidates->front();
  const double fastest_lower = fastest.makespan_ns - fastest.uncertainty_ns;
  const double fastest_upper = fastest.pessimistic_ns;
  const IntervalCandidate* selected = nullptr;
  for (const IntervalCandidate& candidate : *candidates) {
    const bool overlaps =
        candidate.makespan_ns - candidate.uncertainty_ns <= fastest_upper && candidate.pessimistic_ns >= fastest_lower;
    if (overlaps && (selected == nullptr || SelectionKeyLess(candidate, *selected))) {
      selected = &candidate;
    }
  }
  if (selected == nullptr) {
    throw std::runtime_error("candidate uncertainty selection produced no result");
  }
  return *selected;
}

int PeakActiveTasks(const std::vector<std::pair<double, double>>& intervals) {
  std::vector<std::pair<double, int>> events;
  events.reserve(intervals.size() * 2);
  for (const auto& [begin, end] : intervals) {
    events.emplace_back(begin, 1);
    events.emplace_back(end, -1);
  }
  std::sort(events.begin(), events.end());
  int active = 0;
  int peak = 0;
  for (const auto& [time, delta] : events) {
    (void)time;
    active += delta;
    peak = std::max(peak, active);
  }
  return peak;
}

}  // namespace

struct NativeIntervalPlanner::Impl {
  struct Phase {
    double duration_ns = 0.0;
    int64_t working_set_bytes = 0;
  };

  struct SimulationTask {
    int routes = 0;
    int threads = 0;
    std::vector<int> dependencies;
  };

  struct TailPoolSimulation {
    std::vector<SimulationTask> tasks;
    int peak_active = 0;
    int64_t active_working_set_bytes = 0;
    std::vector<int64_t> window_bytes_per_worker;
  };

  struct TailRepartitionAnchor {
    double median_ns = 0.0;
    double p10_ns = 0.0;
    double p90_ns = 0.0;
    int num_iters = 1;
  };

  explicit Impl(IntervalCostModelConfig model_config) : config(std::move(model_config)) {
    if (config.schema_version < 2) {
      throw std::invalid_argument("native interval planner requires a schema-v2 cost model");
    }
    if (config.profile_runs <= 0) {
      throw std::invalid_argument("profile_runs must be positive");
    }
    if (config.w13_stage_bytes <= 0 || config.w2_stage_bytes <= 0) {
      throw std::invalid_argument("full-N stage byte counts must be positive");
    }

    std::map<int, std::map<int, double>> iso_by_threads;
    std::map<int, std::map<int, double>> bulk_by_threads;
    for (const IntervalIsoEntry& entry : config.isolated) {
      iso_exact[{entry.routes, entry.threads}] = entry.nanoseconds;
      iso_by_threads[entry.threads][entry.routes] = entry.nanoseconds;
      if (entry.routes >= 12 && entry.routes % 12 == 0) {
        bulk_by_threads[entry.threads][entry.routes] = entry.nanoseconds;
      }
    }
    for (const auto& [threads, values] : iso_by_threads) {
      iso_curves[threads] = MakeCurve(values);
    }
    for (const auto& [threads, values] : bulk_by_threads) {
      bulk_iso_curves[threads] = MakeCurve(values);
    }
    for (const auto& [threads, value] : config.overheads) {
      overheads[threads] = value;
    }

    std::map<int, std::map<int, double>> derate_2d_rows;
    for (const IntervalDerateEntry& entry : config.derate_2d) {
      derate_2d_rows[entry.experts][entry.routes] = entry.value;
    }
    for (const auto& [experts, values] : derate_2d_rows) {
      derate_2d[experts] = MakeCurve(values);
    }
    for (const IntervalDerateEntry& entry : config.derate_3d) {
      derate_3d[entry.experts][entry.routes][entry.max_threads] = entry.value;
    }
    for (const auto& [experts, curve] : derate_2d) {
      derate_expert_points.push_back(experts);
      derate_route_points.insert(derate_route_points.end(), curve.points.begin(), curve.points.end());
    }
    std::sort(derate_route_points.begin(), derate_route_points.end());
    derate_route_points.erase(std::unique(derate_route_points.begin(), derate_route_points.end()),
                              derate_route_points.end());

    shape_derate = BuildShapeCurves(config.shape_derate);
    group_curves = BuildShapeCurves(config.group_curves);
    full_call_curves = BuildShapeCurves(config.full_call_curves);
    p10_curves = BuildShapeCurves(config.p10_curves);
    p90_curves = BuildShapeCurves(config.p90_curves);
    full_call_p10_curves = BuildShapeCurves(config.full_call_p10_curves);
    full_call_p90_curves = BuildShapeCurves(config.full_call_p90_curves);
    for (const IntervalTailRepartitionEntry& entry : config.tail_repartition_anchors) {
      if (entry.root_shape.empty() ||
          std::any_of(entry.root_shape.begin(), entry.root_shape.end(), [](int width) { return width <= 0; }) ||
          entry.tail_width <= 0 || entry.route_slices <= 0 || entry.routes <= 0 || entry.num_iters <= 0 ||
          entry.p10_ns < 0.0 ||
          entry.p10_ns > entry.median_ns || entry.median_ns > entry.p90_ns) {
        throw std::invalid_argument("bounded-tail calibration entry is invalid");
      }
      const auto key = std::make_tuple(entry.root_shape, entry.tail_width, entry.route_slices, entry.routes);
      const bool inserted =
          tail_repartition_anchors
              .emplace(key, TailRepartitionAnchor{entry.median_ns, entry.p10_ns, entry.p90_ns, entry.num_iters})
              .second;
      if (!inserted) {
        throw std::invalid_argument("duplicate bounded-tail calibration entry");
      }
    }

    if (config.use_formula_iso) {
      if (!config.iso_formula.enabled) {
        throw std::invalid_argument("formula iso mode requires iso_formula calibration");
      }
      formula_route_work = MakeCurve(ToMap(config.iso_formula.route_work));
      formula_measured_phi = MakeCurve(ToMap(config.iso_formula.measured_phi));
      formula_phi_correction.points = formula_measured_phi.points;
      formula_phi_correction.values.reserve(formula_measured_phi.values.size());
      for (size_t index = 0; index < formula_measured_phi.points.size(); ++index) {
        const double threads = formula_measured_phi.points[index];
        const double baseline = FormulaPhiUsl(threads);
        if (baseline <= 0.0) {
          throw std::invalid_argument("formula phi baseline must be positive");
        }
        formula_phi_correction.values.push_back(formula_measured_phi.values[index] / baseline);
      }
    }
  }

  static std::map<int, double> ToMap(const std::vector<std::pair<int, double>>& values) {
    return std::map<int, double>(values.begin(), values.end());
  }

  static std::map<std::vector<int>, Curve> BuildShapeCurves(const std::vector<IntervalShapeCurveEntry>& entries) {
    std::map<std::vector<int>, std::map<int, double>> rows;
    for (const IntervalShapeCurveEntry& entry : entries) {
      rows[ShapeSignature(entry.shape)][entry.routes] = entry.value;
    }
    std::map<std::vector<int>, Curve> result;
    for (const auto& [shape, values] : rows) {
      result[shape] = MakeCurve(values);
    }
    return result;
  }

  int M12TailCapacity(int remainder) const {
    if (remainder <= 0) {
      return 0;
    }
    if (config.exact_m) {
      return remainder;
    }
    if (remainder <= 2) {
      return remainder;
    }
    if (remainder <= 4) {
      return 4;
    }
    if (remainder <= 8) {
      return 8;
    }
    return 12;
  }

  int M12EffectiveRows(int routes) const {
    routes = std::max(routes, 0);
    return routes / 12 * 12 + M12TailCapacity(routes % 12);
  }

  double FormulaOverhead(double threads) const {
    if (threads <= 0.0) {
      throw std::invalid_argument("threads must be positive");
    }
    return std::max(config.iso_formula.o0 + config.iso_formula.o1 / threads, 0.0);
  }

  double FormulaPhiUsl(double threads) const {
    return (1.0 + config.iso_formula.alpha * (threads - 1.0) + config.iso_formula.beta * threads * (threads - 1.0)) /
           threads;
  }

  double FormulaPhiCorrection(double threads) const {
    const Curve& curve = formula_phi_correction;
    if (threads <= curve.points.front()) {
      return curve.values.front();
    }
    if (threads >= curve.points.back()) {
      return curve.values.back();
    }
    const auto upper = std::lower_bound(curve.points.begin(), curve.points.end(), threads);
    const size_t index = static_cast<size_t>(upper - curve.points.begin());
    if (curve.points[index] == threads) {
      return curve.values[index];
    }
    const double t0 = curve.points[index - 1];
    const double t1 = curve.points[index];
    return curve.values[index - 1] + (curve.values[index] - curve.values[index - 1]) * (threads - t0) / (t1 - t0);
  }

  double FormulaPhi(double threads) const { return FormulaPhiUsl(threads) * FormulaPhiCorrection(threads); }

  double FormulaTiso(int routes, int threads) const {
    return FormulaOverhead(threads) + formula_route_work.InterpolateLinear(routes) * FormulaPhi(threads);
  }

  bool SupportsFormulaOnlyWidth(int routes, int threads) const {
    return config.use_formula_iso && !formula_phi_correction.empty() &&
           threads >= formula_phi_correction.points.front() && threads <= formula_phi_correction.points.back() &&
           routes >= 36 && routes % 12 == 0;
  }

  double Overhead(int threads) const {
    if (config.use_formula_iso) {
      return FormulaOverhead(threads);
    }
    const auto overhead = overheads.find(threads);
    return overhead == overheads.end() ? 0.0 : overhead->second;
  }

  double RawIso(int routes, int threads) const {
    const auto exact = iso_exact.find({routes, threads});
    if (exact != iso_exact.end()) {
      return exact->second;
    }
    const auto curve = iso_curves.find(threads);
    if (curve == iso_curves.end()) {
      throw std::out_of_range("no isolated calibration for requested thread width");
    }
    return curve->second.InterpolateLinear(routes);
  }

  double Tiso(int routes, int threads) const {
    if (routes <= 0) {
      return 0.0;
    }
    if (iso_curves.find(threads) == iso_curves.end() && !SupportsFormulaOnlyWidth(routes, threads)) {
      throw std::out_of_range("no isolated calibration for requested thread width");
    }

    if (!config.use_formula_iso) {
      const auto exact = iso_exact.find({routes, threads});
      if (exact != iso_exact.end()) {
        return exact->second;
      }
    }

    const int blocks = routes / 12;
    const int tail = M12TailCapacity(routes % 12);
    const double overhead = Overhead(threads);
    if (blocks == 0) {
      return RawIso(tail, threads);
    }

    double bulk = 0.0;
    if (config.use_formula_iso && blocks > 2) {
      bulk = FormulaTiso(blocks * 12, threads);
    } else {
      const auto bulk_curve = bulk_iso_curves.find(threads);
      bulk = bulk_curve == bulk_iso_curves.end() ? RawIso(blocks * 12, threads)
                                                 : bulk_curve->second.InterpolateLinear(blocks * 12);
    }
    if (tail == 0) {
      return bulk;
    }
    const double tail_time = RawIso(tail, threads);
    return overhead + std::max(bulk - overhead, 0.0) + std::max(tail_time - overhead, 0.0);
  }

  int64_t OwnerStageBytes(int64_t stage_bytes, int64_t tile_bytes, int threads) const {
    if (threads <= 0) {
      throw std::invalid_argument("threads must be positive");
    }
    if (stage_bytes <= 0 || tile_bytes <= 0) {
      return 0;
    }
    const int64_t stage_tiles = (stage_bytes + tile_bytes - 1) / tile_bytes;
    const int64_t owner_tiles = (stage_tiles + threads - 1) / threads;
    return owner_tiles * tile_bytes;
  }

  int64_t TaskMaxStageBytes(int routes, int threads) const {
    (void)routes;
    (void)threads;
    return std::max(config.w13_stage_bytes, config.w2_stage_bytes);
  }

  int64_t WindowBytesPerWorker(int routes, int threads) const {
    (void)routes;
    return std::max(OwnerStageBytes(config.w13_stage_bytes, config.w13_tile_bytes, threads),
                    OwnerStageBytes(config.w2_stage_bytes, config.w2_tile_bytes, threads));
  }

  std::vector<Phase> TaskPhases(int routes, int threads) const {
    const double isolated = Tiso(routes, threads);
    const double overhead = std::min(Overhead(threads), isolated * 0.9);
    const double compute = std::max(isolated - overhead, 0.0);
    std::vector<Phase> phases;
    if (overhead > 0.0) {
      phases.push_back({overhead, 0});
    }
    const double w13_phase = compute * (2.0 / 3.0);
    if (w13_phase > 0.0) {
      phases.push_back({w13_phase, config.w13_stage_bytes});
    }
    const double w2_phase = compute / 3.0;
    if (w2_phase > 0.0) {
      phases.push_back({w2_phase, config.w2_stage_bytes});
    }
    return phases;
  }

  int NearestRoute(int routes) const {
    if (derate_route_points.empty()) {
      throw std::invalid_argument("derate calibration has no route points");
    }
    const int effective = M12EffectiveRows(routes);
    int best = derate_route_points.front();
    int best_distance = std::abs(best - effective);
    for (int point : derate_route_points) {
      const int distance = std::abs(point - effective);
      if (distance < best_distance) {
        best = point;
        best_distance = distance;
      }
    }
    return best;
  }

  std::optional<double> ShapeDerate(int routes, const std::vector<int>& shape) const {
    if (!config.use_shape_derate || shape.empty()) {
      return std::nullopt;
    }
    const std::vector<int> signature = ShapeSignature(shape);
    if (signature.size() <= 1) {
      return 1.0;
    }
    const auto curve = shape_derate.find(signature);
    if (curve == shape_derate.end()) {
      return std::nullopt;
    }
    return curve->second.InterpolateLogRoute(M12EffectiveRows(routes));
  }

  double DerateAt(int experts, int route_point, int max_threads) const {
    if (config.use_max_team_derate) {
      const auto experts_entry = derate_3d.find(experts);
      if (experts_entry != derate_3d.end()) {
        const auto route_entry = experts_entry->second.find(route_point);
        if (route_entry != experts_entry->second.end() && !route_entry->second.empty()) {
          auto best = route_entry->second.begin();
          int best_distance = std::abs(best->first - max_threads);
          for (auto current = std::next(best); current != route_entry->second.end(); ++current) {
            const int distance = std::abs(current->first - max_threads);
            if (distance < best_distance) {
              best = current;
              best_distance = distance;
            }
          }
          return best->second;
        }
      }
    }
    const auto experts_entry = derate_2d.find(experts);
    if (experts_entry == derate_2d.end()) {
      throw std::out_of_range("missing derate expert-count calibration");
    }
    const auto route =
        std::lower_bound(experts_entry->second.points.begin(), experts_entry->second.points.end(), route_point);
    if (route == experts_entry->second.points.end() || *route != route_point) {
      throw std::out_of_range("missing derate route calibration");
    }
    return experts_entry->second.values[static_cast<size_t>(route - experts_entry->second.points.begin())];
  }

  double DerateFloat(double equivalent_experts, int routes, int max_threads) const {
    if (equivalent_experts <= 1.0) {
      return 1.0;
    }
    if (derate_expert_points.empty()) {
      throw std::invalid_argument("derate calibration has no expert-count points");
    }
    const int route_point = NearestRoute(routes);
    if (equivalent_experts <= derate_expert_points.front()) {
      return DerateAt(derate_expert_points.front(), route_point, max_threads);
    }
    if (equivalent_experts >= derate_expert_points.back()) {
      return DerateAt(derate_expert_points.back(), route_point, max_threads);
    }
    const auto upper = std::lower_bound(derate_expert_points.begin(), derate_expert_points.end(), equivalent_experts);
    const int n1 = *upper;
    const int n0 = *std::prev(upper);
    const double d0 = DerateAt(n0, route_point, max_threads);
    const double d1 = DerateAt(n1, route_point, max_threads);
    return d0 + (d1 - d0) * (equivalent_experts - n0) / (n1 - n0);
  }

  double WorkingSetDerate(const std::vector<int>& active, const std::vector<int>& routes,
                          const std::vector<int>& threads, const std::vector<int64_t>& working_sets) const {
    std::vector<int> compute_active;
    for (int index : active) {
      if (working_sets[index] > 0) {
        compute_active.push_back(index);
      }
    }
    if (compute_active.size() <= 1 || config.max_stage_bytes <= 0) {
      return 1.0;
    }

    int64_t total_bytes = 0;
    int max_routes = 0;
    int max_threads = 0;
    std::vector<int> shape;
    bool homogeneous = true;
    for (int index : compute_active) {
      total_bytes += working_sets[index];
      max_routes = std::max(max_routes, routes[index]);
      max_threads = std::max(max_threads, threads[index]);
      shape.push_back(threads[index]);
      homogeneous = homogeneous && working_sets[index] == config.max_stage_bytes;
    }
    if (homogeneous) {
      const std::optional<double> exact = ShapeDerate(max_routes, shape);
      if (exact.has_value()) {
        return *exact;
      }
    }
    return DerateFloat(static_cast<double>(total_bytes) / config.max_stage_bytes, max_routes, max_threads);
  }

  double FlatDerate(int active_count, int max_routes, int max_threads, const std::vector<int>& shape) const {
    if (active_count <= 1) {
      return 1.0;
    }
    const std::optional<double> exact = ShapeDerate(max_routes, shape);
    return exact.has_value() ? *exact : DerateFloat(active_count, max_routes, max_threads);
  }

  double OverheadFraction(int routes, int threads) const {
    const double isolated = Tiso(routes, threads);
    if (isolated <= 0.0) {
      return 0.0;
    }
    return std::min(Overhead(threads) / isolated, 0.9);
  }

  static void BuildDagState(const std::vector<SimulationTask>& tasks, std::vector<int>* dependency_count,
                            std::vector<std::vector<int>>* successors, std::vector<bool>* started) {
    const int count = static_cast<int>(tasks.size());
    dependency_count->resize(count);
    successors->assign(count, {});
    started->resize(count);
    for (int task = 0; task < count; ++task) {
      (*dependency_count)[task] = static_cast<int>(tasks[task].dependencies.size());
      (*started)[task] = tasks[task].dependencies.empty();
      for (int dependency : tasks[task].dependencies) {
        if (dependency < 0 || dependency >= task) {
          throw std::invalid_argument("task dependencies must refer to earlier tasks");
        }
        (*successors)[dependency].push_back(task);
      }
    }
  }

  double StagedDagMakespan(const std::vector<SimulationTask>& tasks) const {
    const int count = static_cast<int>(tasks.size());
    if (count == 0) {
      return 0.0;
    }
    std::vector<int> routes(count);
    std::vector<int> threads(count);
    std::vector<std::vector<Phase>> phases(count);
    std::vector<int> phase_index(count, 0);
    std::vector<double> remaining(count);
    int max_events = count + 2;
    for (int task = 0; task < count; ++task) {
      routes[task] = tasks[task].routes;
      threads[task] = tasks[task].threads;
      phases[task] = TaskPhases(routes[task], threads[task]);
      if (phases[task].empty()) {
        throw std::invalid_argument("DAG tasks must have positive route counts");
      }
      remaining[task] = phases[task].front().duration_ns;
      max_events += static_cast<int>(phases[task].size());
    }

    std::vector<int> dependency_count;
    std::vector<std::vector<int>> successors;
    std::vector<bool> started;
    BuildDagState(tasks, &dependency_count, &successors, &started);
    std::vector<bool> finished(count, false);
    int finished_count = 0;
    int guard = 0;
    double wall = config.call_setup_ns;

    while (finished_count < count) {
      if (++guard > 2 * max_events) {
        throw std::runtime_error("stage-aware DAG simulation did not converge");
      }
      std::vector<int> active;
      std::vector<int64_t> working_sets(count, 0);
      for (int task = 0; task < count; ++task) {
        if (started[task] && !finished[task]) {
          active.push_back(task);
          working_sets[task] = phases[task][phase_index[task]].working_set_bytes;
        }
      }
      if (active.empty()) {
        throw std::invalid_argument("DAG deadlock (cycle or unreachable task)");
      }
      const double slowdown = WorkingSetDerate(active, routes, threads, working_sets);
      double elapsed = std::numeric_limits<double>::infinity();
      for (int task : active) {
        const double multiplier = working_sets[task] == 0 ? 1.0 : slowdown;
        elapsed = std::min(elapsed, remaining[task] * multiplier);
      }
      wall += elapsed;
      for (int task : active) {
        const double multiplier = working_sets[task] == 0 ? 1.0 : slowdown;
        remaining[task] -= elapsed / multiplier;
      }
      for (int task : active) {
        if (remaining[task] > kCompletionTolerance) {
          continue;
        }
        ++phase_index[task];
        if (phase_index[task] < static_cast<int>(phases[task].size())) {
          remaining[task] = phases[task][phase_index[task]].duration_ns;
          continue;
        }
        finished[task] = true;
        ++finished_count;
        for (int successor : successors[task]) {
          --dependency_count[successor];
          if (dependency_count[successor] == 0) {
            started[successor] = true;
          }
        }
      }
    }
    return wall;
  }

  double FlatDagMakespan(const std::vector<SimulationTask>& tasks) const {
    const int count = static_cast<int>(tasks.size());
    if (count == 0) {
      return 0.0;
    }
    std::vector<int> routes(count);
    std::vector<int> threads(count);
    std::vector<double> remaining(count);
    for (int task = 0; task < count; ++task) {
      routes[task] = tasks[task].routes;
      threads[task] = tasks[task].threads;
      remaining[task] = Tiso(routes[task], threads[task]);
    }
    std::vector<int> dependency_count;
    std::vector<std::vector<int>> successors;
    std::vector<bool> started;
    BuildDagState(tasks, &dependency_count, &successors, &started);
    std::vector<bool> finished(count, false);
    int finished_count = 0;
    int guard = 0;
    double wall = config.call_setup_ns;
    while (finished_count < count) {
      if (++guard > 2 * count + 2) {
        throw std::runtime_error("flat DAG simulation did not converge");
      }
      std::vector<int> active;
      std::vector<int> shape;
      int max_routes = 0;
      int max_threads = 0;
      for (int task = 0; task < count; ++task) {
        if (started[task] && !finished[task]) {
          active.push_back(task);
          shape.push_back(threads[task]);
          max_routes = std::max(max_routes, routes[task]);
          max_threads = std::max(max_threads, threads[task]);
        }
      }
      if (active.empty()) {
        throw std::invalid_argument("DAG deadlock (cycle or unreachable task)");
      }
      const double slowdown = FlatDerate(static_cast<int>(active.size()), max_routes, max_threads, shape);
      std::vector<double> multipliers(count, 1.0);
      double elapsed = std::numeric_limits<double>::infinity();
      for (int task : active) {
        const double fraction = OverheadFraction(routes[task], threads[task]);
        multipliers[task] = fraction + (1.0 - fraction) * slowdown;
        elapsed = std::min(elapsed, remaining[task] * multipliers[task]);
      }
      wall += elapsed;
      for (int task : active) {
        remaining[task] -= elapsed / multipliers[task];
      }
      for (int task : active) {
        if (remaining[task] > kCompletionTolerance) {
          continue;
        }
        finished[task] = true;
        ++finished_count;
        for (int successor : successors[task]) {
          --dependency_count[successor];
          if (dependency_count[successor] == 0) {
            started[successor] = true;
          }
        }
      }
    }
    return wall;
  }

  double DagMakespan(const std::vector<SimulationTask>& tasks) const {
    return config.use_stage_model && config.max_stage_bytes > 0 ? StagedDagMakespan(tasks) : FlatDagMakespan(tasks);
  }

  double ProfiledCurve(const std::map<std::vector<int>, Curve>& curves, int routes,
                       const std::vector<int>& shape) const {
    const auto curve = curves.find(ShapeSignature(shape));
    if (curve == curves.end()) {
      throw std::out_of_range("shape was not measured by the cost model");
    }
    return curve->second.InterpolateLinear(M12EffectiveRows(routes));
  }

  double RelativeUncertainty(int routes, const std::vector<int>& shape) const {
    const double median = ProfiledCurve(group_curves, routes, shape);
    const double p10 = ProfiledCurve(p10_curves, routes, shape);
    const double p90 = ProfiledCurve(p90_curves, routes, shape);
    return median <= 0.0 ? 0.0 : std::max({median - p10, p90 - median, 0.0}) / median;
  }

  double RelativeFullCallUncertainty(int routes, const std::vector<int>& shape) const {
    const double median = ProfiledCurve(full_call_curves, routes, shape);
    const double p10 = ProfiledCurve(full_call_p10_curves, routes, shape);
    const double p90 = ProfiledCurve(full_call_p90_curves, routes, shape);
    return median <= 0.0 ? 0.0 : std::max({median - p10, p90 - median, 0.0}) / median;
  }

  bool UsesFullWorkloadAnchor(const std::vector<Expert>& experts, const std::vector<int>& shape) const {
    (void)shape;
    if (!config.has_full_workload_anchors || static_cast<int>(experts.size()) != config.local_experts ||
        experts.empty()) {
      return false;
    }
    if (!std::all_of(experts.begin(), experts.end(),
                     [&](const Expert& expert) { return expert.routes == experts.front().routes; })) {
      return false;
    }
    return true;
  }

  std::optional<std::pair<double, double>> BoundedTailAnchor(
      const std::vector<Expert>& experts, const std::vector<int>& root_shape, int tail_width,
      int route_slices) const {
    if (experts.size() != root_shape.size() + 2 || experts.empty() ||
        route_slices <= 0 ||
        !std::all_of(experts.begin(), experts.end(),
                     [&](const Expert& expert) { return expert.routes == experts.front().routes; })) {
      return std::nullopt;
    }
    const int routes = M12EffectiveRows(experts.front().routes);
    if (routes % route_slices != 0) {
      return std::nullopt;
    }
    const auto anchor =
        tail_repartition_anchors.find(std::make_tuple(root_shape, tail_width, route_slices, routes));
    if (anchor == tail_repartition_anchors.end()) {
      return std::nullopt;
    }
    const TailRepartitionAnchor& value = anchor->second;
    const double spread = std::max({value.median_ns - value.p10_ns, value.p90_ns - value.median_ns, 0.0});
    return std::make_pair(value.median_ns, spread / std::sqrt(static_cast<double>(value.num_iters)));
  }

  double Uncertainty(const std::vector<Expert>& experts, const std::vector<int>& shape, double makespan,
                     bool use_full_workload_anchor) const {
    if (config.schema_version < 2) {
      return 0.0;
    }
    if (use_full_workload_anchor && UsesFullWorkloadAnchor(experts, shape)) {
      const double relative = RelativeFullCallUncertainty(experts.front().routes, shape);
      return makespan * relative / std::sqrt(config.profile_runs);
    }
    double relative = 0.0;
    for (const Expert& expert : experts) {
      relative = std::max(relative, RelativeUncertainty(expert.routes, shape));
    }
    const int waves = std::max<int>(
        1, (static_cast<int>(experts.size()) + static_cast<int>(shape.size()) - 1) / static_cast<int>(shape.size()));
    return makespan * relative / std::sqrt(static_cast<double>(waves * config.profile_runs));
  }

  IntervalCostModelConfig config;
  std::map<std::pair<int, int>, double> iso_exact;
  std::map<int, Curve> iso_curves;
  std::map<int, Curve> bulk_iso_curves;
  std::map<int, double> overheads;
  std::map<int, Curve> derate_2d;
  std::map<int, std::map<int, std::map<int, double>>> derate_3d;
  std::vector<int> derate_expert_points;
  std::vector<int> derate_route_points;
  std::map<std::vector<int>, Curve> shape_derate;
  std::map<std::vector<int>, Curve> group_curves;
  std::map<std::vector<int>, Curve> full_call_curves;
  std::map<std::vector<int>, Curve> p10_curves;
  std::map<std::vector<int>, Curve> p90_curves;
  std::map<std::vector<int>, Curve> full_call_p10_curves;
  std::map<std::vector<int>, Curve> full_call_p90_curves;
  std::map<std::tuple<std::vector<int>, int, int, int>, TailRepartitionAnchor> tail_repartition_anchors;
  Curve formula_route_work;
  Curve formula_measured_phi;
  Curve formula_phi_correction;
};

namespace {

std::vector<Expert> BuildExperts(const std::vector<int>& expert_ids, const std::vector<int>& routes) {
  if (expert_ids.size() != routes.size()) {
    throw std::invalid_argument("expert_ids and routes must have the same length");
  }
  std::vector<Expert> experts;
  experts.reserve(expert_ids.size());
  for (size_t index = 0; index < expert_ids.size(); ++index) {
    if (routes[index] > 0) {
      experts.push_back({expert_ids[index], routes[index], static_cast<int>(index)});
    }
  }
  if (experts.empty()) {
    throw std::invalid_argument("at least one active expert is required");
  }
  return experts;
}

std::vector<IntervalTask> BuildStrictTasks(const NativeIntervalPlanner::Impl& model, const std::vector<Expert>& experts,
                                           const std::vector<int>& shape) {
  const int lane_count = static_cast<int>(shape.size());
  std::vector<double> loads(lane_count, 0.0);
  std::vector<std::vector<int>> lane_experts(lane_count);
  std::vector<int> order(experts.size());
  for (size_t index = 0; index < experts.size(); ++index) {
    order[index] = static_cast<int>(index);
  }
  std::stable_sort(order.begin(), order.end(),
                   [&](int left, int right) { return experts[left].routes > experts[right].routes; });
  for (int expert_index : order) {
    int selected_lane = 0;
    double selected_finish = loads[0] + model.Tiso(experts[expert_index].routes, shape[0]);
    for (int lane = 1; lane < lane_count; ++lane) {
      const double finish = loads[lane] + model.Tiso(experts[expert_index].routes, shape[lane]);
      if (finish < selected_finish) {
        selected_lane = lane;
        selected_finish = finish;
      }
    }
    lane_experts[selected_lane].push_back(expert_index);
    loads[selected_lane] += model.Tiso(experts[expert_index].routes, shape[selected_lane]);
  }

  std::vector<IntervalTask> tasks;
  tasks.reserve(experts.size());
  int core_begin = 0;
  for (int lane = 0; lane < lane_count; ++lane) {
    int previous = -1;
    for (int expert_index : lane_experts[lane]) {
      IntervalTask task;
      task.expert_id = experts[expert_index].expert_id;
      task.routes = experts[expert_index].routes;
      task.core_begin = core_begin;
      task.threads = shape[lane];
      if (previous >= 0) {
        task.dependencies.push_back(previous);
      }
      tasks.push_back(std::move(task));
      previous = static_cast<int>(tasks.size()) - 1;
    }
    core_begin += shape[lane];
  }
  return tasks;
}

std::vector<NativeIntervalPlanner::Impl::SimulationTask> ToSimulationTasks(const std::vector<IntervalTask>& tasks) {
  std::vector<NativeIntervalPlanner::Impl::SimulationTask> result;
  result.reserve(tasks.size());
  for (const IntervalTask& task : tasks) {
    result.push_back({task.routes, task.threads, task.dependencies});
  }
  return result;
}

IntervalCandidate BuildStrictCandidate(const NativeIntervalPlanner::Impl& model, const std::vector<Expert>& experts,
                                       const std::vector<int>& shape) {
  IntervalCandidate candidate;
  candidate.shape = shape;
  candidate.tasks = BuildStrictTasks(model, experts, shape);
  if (model.UsesFullWorkloadAnchor(experts, shape)) {
    candidate.makespan_ns = model.ProfiledCurve(model.full_call_curves, experts.front().routes, shape);
  } else {
    candidate.makespan_ns = model.DagMakespan(ToSimulationTasks(candidate.tasks));
  }
  candidate.uncertainty_ns = model.Uncertainty(experts, shape, candidate.makespan_ns, true);
  candidate.pessimistic_ns = candidate.makespan_ns + candidate.uncertainty_ns;
  std::map<std::pair<int, int>, int64_t> resource_working_sets;
  std::map<std::pair<int, int>, int64_t> resource_owner_windows;
  for (const IntervalTask& task : candidate.tasks) {
    const std::pair<int, int> resource = {task.core_begin, task.threads};
    resource_working_sets[resource] =
        std::max(resource_working_sets[resource], model.TaskMaxStageBytes(task.routes, task.threads));
    resource_owner_windows[resource] =
        std::max(resource_owner_windows[resource], model.WindowBytesPerWorker(task.routes, task.threads));
  }
  candidate.resource_groups = static_cast<int>(resource_working_sets.size());
  for (const auto& [resource, bytes] : resource_working_sets) {
    (void)resource;
    candidate.active_working_set_bytes += bytes;
  }
  for (const auto& [resource, bytes] : resource_owner_windows) {
    (void)resource;
    candidate.window_bytes_per_worker.push_back(bytes);
  }
  return candidate;
}

std::optional<std::vector<IntervalTask>> BuildBoundedTailTasks(const std::vector<IntervalTask>& tasks, int num_cores,
                                                               int tail_width, int route_slices) {
  constexpr int kTailExperts = 2;
  if (num_cores <= 0 || num_cores % kTailExperts != 0 || tail_width <= 0 || route_slices <= 0 ||
      num_cores % tail_width != 0 || kTailExperts * tail_width > num_cores) {
    return std::nullopt;
  }
  const int physical_tail_tasks = kTailExperts * route_slices;
  if (route_slices > 1 && physical_tail_tasks * tail_width != num_cores) {
    return std::nullopt;
  }

  std::vector<int> successors(tasks.size(), 0);
  for (size_t task_id = 0; task_id < tasks.size(); ++task_id) {
    for (int dependency : tasks[task_id].dependencies) {
      if (dependency < 0 || dependency >= static_cast<int>(task_id)) {
        return std::nullopt;
      }
      ++successors[static_cast<size_t>(dependency)];
    }
  }

  std::vector<int> roots;
  std::vector<int> tails;
  for (size_t task_id = 0; task_id < tasks.size(); ++task_id) {
    (tasks[task_id].dependencies.empty() ? roots : tails).push_back(static_cast<int>(task_id));
  }
  if (tails.size() != kTailExperts || roots.size() + tails.size() != tasks.size()) {
    return std::nullopt;
  }
  for (int task_id : tails) {
    const IntervalTask& tail = tasks[static_cast<size_t>(task_id)];
    if (successors[static_cast<size_t>(task_id)] != 0 || tail_width <= tail.threads ||
        tail.routes % route_slices != 0) {
      return std::nullopt;
    }
  }

  const auto by_core = [&](int left, int right) {
    return std::tie(tasks[static_cast<size_t>(left)].core_begin, left) <
           std::tie(tasks[static_cast<size_t>(right)].core_begin, right);
  };
  std::stable_sort(roots.begin(), roots.end(), by_core);
  std::stable_sort(tails.begin(), tails.end(), by_core);

  std::vector<IntervalTask> rewritten;
  rewritten.reserve(roots.size() + static_cast<size_t>(physical_tail_tasks));
  for (int task_id : roots) {
    IntervalTask root = tasks[static_cast<size_t>(task_id)];
    root.dependencies.clear();
    rewritten.push_back(std::move(root));
  }
  const int partition_width = num_cores / kTailExperts;
  for (int tail_index = 0; tail_index < kTailExperts; ++tail_index) {
    for (int route_slice = 0; route_slice < route_slices; ++route_slice) {
      const int core_begin =
          route_slices == 1 ? tail_index * partition_width
                            : (tail_index * route_slices + route_slice) * tail_width;
      const int core_end = core_begin + tail_width;
      if (core_end > num_cores) {
        return std::nullopt;
      }
      IntervalTask tail = tasks[static_cast<size_t>(tails[static_cast<size_t>(tail_index)])];
      tail.routes /= route_slices;
      tail.core_begin = core_begin;
      tail.threads = tail_width;
      tail.dependencies.clear();
      for (size_t root_id = 0; root_id < roots.size(); ++root_id) {
        const IntervalTask& root = tasks[static_cast<size_t>(roots[root_id])];
        if (root.core_begin < core_end && core_begin < root.core_begin + root.threads) {
          tail.dependencies.push_back(static_cast<int>(root_id));
        }
      }
      if (tail.dependencies.empty()) {
        return std::nullopt;
      }
      rewritten.push_back(std::move(tail));
    }
  }
  return rewritten;
}

std::optional<IntervalCandidate> BuildBoundedTailCandidate(const NativeIntervalPlanner::Impl& model,
                                                           const std::vector<Expert>& experts,
                                                           const IntervalCandidate& strict_candidate, int num_cores,
                                                           int tail_width, int route_slices) {
  std::optional<std::vector<IntervalTask>> tasks =
      BuildBoundedTailTasks(strict_candidate.tasks, num_cores, tail_width, route_slices);
  if (!tasks.has_value()) {
    return std::nullopt;
  }
  try {
    constexpr int kTailExperts = 2;
    const int physical_tail_tasks = kTailExperts * route_slices;
    IntervalCandidate candidate;
    candidate.shape = strict_candidate.shape;
    candidate.tail_repartition_width = tail_width;
    candidate.tail_repartition_tasks = kTailExperts;
    candidate.tail_repartition_route_slices = route_slices;
    candidate.tasks = std::move(*tasks);
    const std::optional<std::pair<double, double>> anchor =
        model.BoundedTailAnchor(experts, candidate.shape, tail_width, route_slices);
    if (route_slices > 1 && !anchor.has_value()) {
      return std::nullopt;
    }
    if (anchor.has_value()) {
      candidate.makespan_ns = anchor->first;
      candidate.uncertainty_ns = anchor->second;
    } else {
      candidate.makespan_ns = model.DagMakespan(ToSimulationTasks(candidate.tasks));
      candidate.uncertainty_ns = model.Uncertainty(experts, candidate.shape, candidate.makespan_ns, false);
    }
    candidate.pessimistic_ns = candidate.makespan_ns + candidate.uncertainty_ns;

    int64_t root_working_set = 0;
    int64_t tail_working_set = 0;
    std::vector<int64_t> root_windows;
    std::vector<int64_t> tail_windows;
    const size_t first_tail = candidate.tasks.size() - physical_tail_tasks;
    for (size_t task_id = 0; task_id < candidate.tasks.size(); ++task_id) {
      const IntervalTask& task = candidate.tasks[task_id];
      const int64_t bytes = model.TaskMaxStageBytes(task.routes, task.threads);
      const int64_t window = model.WindowBytesPerWorker(task.routes, task.threads);
      if (task_id < first_tail) {
        root_working_set += bytes;
        root_windows.push_back(window);
      } else {
        tail_working_set += bytes;
        tail_windows.push_back(window);
      }
    }
    candidate.active_working_set_bytes = std::max(root_working_set, tail_working_set);
    candidate.window_bytes_per_worker =
        root_working_set >= tail_working_set ? std::move(root_windows) : std::move(tail_windows);
    candidate.resource_groups = std::max<int>(static_cast<int>(first_tail), physical_tail_tasks);
    return candidate;
  } catch (const std::out_of_range&) {
    return std::nullopt;
  } catch (const std::invalid_argument&) {
    return std::nullopt;
  }
}

bool TailPoolLayout(const std::vector<IntervalTask>& tasks, int num_cores, int pool_threads, int max_pooled_routes,
                    std::vector<bool>* pooled, std::vector<std::vector<int>>* resolved_dependencies) {
  if (pool_threads <= 0 || pool_threads > num_cores || num_cores % pool_threads != 0 || max_pooled_routes <= 0) {
    return false;
  }
  pooled->resize(tasks.size());
  bool any_pooled = false;
  for (size_t task = 0; task < tasks.size(); ++task) {
    (*pooled)[task] = tasks[task].routes <= max_pooled_routes;
    any_pooled = any_pooled || (*pooled)[task];
  }
  if (!any_pooled) {
    return false;
  }

  resolved_dependencies->assign(tasks.size(), {});
  for (size_t task_id = 0; task_id < tasks.size(); ++task_id) {
    if ((*pooled)[task_id]) {
      continue;
    }
    const IntervalTask& task = tasks[task_id];
    if (task.core_begin % pool_threads != 0 || task.threads % pool_threads != 0) {
      return false;
    }
    std::vector<int> frontier = task.dependencies;
    std::set<int> fixed_dependencies;
    while (!frontier.empty()) {
      const int dependency = frontier.back();
      frontier.pop_back();
      if (dependency < 0 || dependency >= static_cast<int>(task_id)) {
        return false;
      }
      if ((*pooled)[dependency]) {
        frontier.insert(frontier.end(), tasks[dependency].dependencies.begin(), tasks[dependency].dependencies.end());
      } else {
        fixed_dependencies.insert(dependency);
      }
    }
    (*resolved_dependencies)[task_id].assign(fixed_dependencies.begin(), fixed_dependencies.end());
  }
  return true;
}

std::optional<NativeIntervalPlanner::Impl::TailPoolSimulation> BuildTailPoolSimulation(
    const NativeIntervalPlanner::Impl& model, const std::vector<IntervalTask>& tasks, int num_cores, int pool_threads,
    int max_pooled_routes) {
  std::vector<bool> pooled;
  std::vector<std::vector<int>> resolved_dependencies;
  if (!TailPoolLayout(tasks, num_cores, pool_threads, max_pooled_routes, &pooled, &resolved_dependencies)) {
    return std::nullopt;
  }

  std::vector<int> fixed_task_ids;
  for (size_t task = 0; task < tasks.size(); ++task) {
    if (!pooled[task]) {
      fixed_task_ids.push_back(static_cast<int>(task));
    }
  }
  std::vector<int> fixed_sim_ids(tasks.size(), -1);
  std::vector<double> isolated_finish(tasks.size(), 0.0);
  NativeIntervalPlanner::Impl::TailPoolSimulation simulation;
  std::vector<std::pair<double, double>> intervals;
  for (int task_id : fixed_task_ids) {
    const IntervalTask& task = tasks[task_id];
    double start = 0.0;
    std::vector<int> simulation_dependencies;
    for (int dependency : resolved_dependencies[task_id]) {
      start = std::max(start, isolated_finish[dependency]);
      simulation_dependencies.push_back(fixed_sim_ids[dependency]);
    }
    const double finish = start + model.Tiso(task.routes, task.threads);
    fixed_sim_ids[task_id] = static_cast<int>(simulation.tasks.size());
    isolated_finish[task_id] = finish;
    simulation.tasks.push_back({task.routes, task.threads, std::move(simulation_dependencies)});
    intervals.emplace_back(start, finish);
  }

  const int group_count = num_cores / pool_threads;
  std::vector<std::vector<int>> group_blockers(group_count);
  for (int task_id : fixed_task_ids) {
    const IntervalTask& task = tasks[task_id];
    const int first_group = task.core_begin / pool_threads;
    for (int group = first_group; group < first_group + task.threads / pool_threads; ++group) {
      group_blockers[group].push_back(task_id);
    }
  }

  using Availability = std::pair<double, int>;
  std::priority_queue<Availability, std::vector<Availability>, std::greater<Availability>> availability;
  for (int group = 0; group < group_count; ++group) {
    double available = 0.0;
    for (int blocker : group_blockers[group]) {
      available = std::max(available, isolated_finish[blocker]);
    }
    availability.push({available, group});
  }

  std::vector<int> pooled_task_ids;
  for (size_t task = 0; task < tasks.size(); ++task) {
    if (pooled[task]) {
      pooled_task_ids.push_back(static_cast<int>(task));
    }
  }
  std::sort(pooled_task_ids.begin(), pooled_task_ids.end(), [&](int left, int right) {
    if (tasks[left].routes != tasks[right].routes) {
      return tasks[left].routes > tasks[right].routes;
    }
    return tasks[left].expert_id < tasks[right].expert_id;
  });
  std::vector<int> previous_pool_task(group_count, -1);
  for (int task_id : pooled_task_ids) {
    const auto [available_ns, group] = availability.top();
    availability.pop();
    std::vector<int> dependencies;
    if (previous_pool_task[group] < 0) {
      for (int blocker : group_blockers[group]) {
        dependencies.push_back(fixed_sim_ids[blocker]);
      }
    } else {
      dependencies.push_back(previous_pool_task[group]);
    }
    const int simulation_id = static_cast<int>(simulation.tasks.size());
    const double finish = available_ns + model.Tiso(tasks[task_id].routes, pool_threads);
    simulation.tasks.push_back({tasks[task_id].routes, pool_threads, std::move(dependencies)});
    intervals.emplace_back(available_ns, finish);
    previous_pool_task[group] = simulation_id;
    availability.push({finish, group});
  }

  simulation.peak_active = PeakActiveTasks(intervals);
  std::vector<int64_t> task_working_sets;
  task_working_sets.reserve(simulation.tasks.size());
  for (const NativeIntervalPlanner::Impl::SimulationTask& task : simulation.tasks) {
    task_working_sets.push_back(model.TaskMaxStageBytes(task.routes, task.threads));
  }
  std::sort(task_working_sets.begin(), task_working_sets.end(), std::greater<int64_t>());
  for (int index = 0; index < std::min<int>(simulation.peak_active, task_working_sets.size()); ++index) {
    simulation.active_working_set_bytes += task_working_sets[index];
  }

  std::map<std::pair<int, int>, int64_t> fixed_windows;
  for (int task_id : fixed_task_ids) {
    const IntervalTask& task = tasks[task_id];
    const std::pair<int, int> resource = {task.core_begin, task.threads};
    fixed_windows[resource] = std::max(fixed_windows[resource], model.WindowBytesPerWorker(task.routes, task.threads));
  }
  for (const auto& [resource, bytes] : fixed_windows) {
    (void)resource;
    simulation.window_bytes_per_worker.push_back(bytes);
  }
  const int active_pool_groups = std::min<int>(pooled_task_ids.size(), group_count);
  int64_t pooled_window = 0;
  for (int task_id : pooled_task_ids) {
    pooled_window = std::max(pooled_window, model.WindowBytesPerWorker(tasks[task_id].routes, pool_threads));
  }
  simulation.window_bytes_per_worker.insert(simulation.window_bytes_per_worker.end(), active_pool_groups,
                                            pooled_window);
  return simulation;
}

std::optional<IntervalCandidate> BuildTailPoolCandidate(const NativeIntervalPlanner::Impl& model,
                                                        const std::vector<Expert>& experts,
                                                        const IntervalCandidate& strict_candidate, int num_cores,
                                                        int pool_threads, int max_pooled_routes) {
  std::optional<NativeIntervalPlanner::Impl::TailPoolSimulation> simulation =
      BuildTailPoolSimulation(model, strict_candidate.tasks, num_cores, pool_threads, max_pooled_routes);
  if (!simulation.has_value()) {
    return std::nullopt;
  }
  IntervalCandidate candidate;
  candidate.shape = strict_candidate.shape;
  candidate.execution_mode = IntervalExecutionMode::kTailPool;
  candidate.tail_pool_threads = pool_threads;
  candidate.tail_pool_max_routes = max_pooled_routes;
  candidate.tail_pool_tasks = static_cast<int>(std::count_if(
      experts.begin(), experts.end(), [&](const Expert& expert) { return expert.routes <= max_pooled_routes; }));
  candidate.makespan_ns = model.DagMakespan(simulation->tasks);
  candidate.uncertainty_ns = model.Uncertainty(experts, candidate.shape, candidate.makespan_ns, false);
  candidate.pessimistic_ns = candidate.makespan_ns + candidate.uncertainty_ns;
  candidate.tasks = strict_candidate.tasks;
  candidate.active_working_set_bytes = simulation->active_working_set_bytes;
  candidate.window_bytes_per_worker = simulation->window_bytes_per_worker;
  candidate.resource_groups = simulation->peak_active;
  return candidate;
}

std::vector<int> TailPoolThresholds(const std::vector<Expert>& experts, int max_pooled_routes, bool forced) {
  if (max_pooled_routes <= 0) {
    throw std::invalid_argument("tail_pool_max_routes must be positive");
  }
  if (forced) {
    return {max_pooled_routes};
  }
  std::set<int> points;
  for (int threshold : kAutoTailPoolThresholds) {
    if (threshold <= max_pooled_routes) {
      points.insert(threshold);
    }
  }
  points.insert(max_pooled_routes);
  std::set<std::vector<int>> seen_pooled_sets;
  std::vector<int> thresholds;
  for (int threshold : points) {
    std::vector<int> pooled_set;
    for (const Expert& expert : experts) {
      if (expert.routes <= threshold) {
        pooled_set.push_back(expert.expert_id);
      }
    }
    if (!pooled_set.empty() && seen_pooled_sets.insert(pooled_set).second) {
      thresholds.push_back(threshold);
    }
  }
  return thresholds;
}

std::vector<IntervalCandidate> TailPoolHeadCandidates(std::vector<IntervalCandidate> strict_candidates) {
  std::stable_sort(strict_candidates.begin(), strict_candidates.end(), CandidateLessByMakespan);
  const IntervalCandidate& fastest = strict_candidates.front();
  const double fastest_lower = fastest.makespan_ns - fastest.uncertainty_ns;
  const double fastest_upper = fastest.pessimistic_ns;
  std::set<std::vector<int>> selected_shapes;
  for (const IntervalCandidate& candidate : strict_candidates) {
    if (candidate.makespan_ns - candidate.uncertainty_ns <= fastest_upper &&
        candidate.pessimistic_ns >= fastest_lower) {
      selected_shapes.insert(candidate.shape);
    }
  }
  for (int index = 0; index < std::min<int>(kAutoTailPoolMinHeadShapes, strict_candidates.size()); ++index) {
    selected_shapes.insert(strict_candidates[index].shape);
  }
  std::vector<IntervalCandidate> result;
  for (const IntervalCandidate& candidate : strict_candidates) {
    if (selected_shapes.find(candidate.shape) != selected_shapes.end()) {
      result.push_back(candidate);
    }
  }
  return result;
}

}  // namespace

NativeIntervalPlanner::NativeIntervalPlanner(int num_cores, std::vector<int> widths,
                                             std::vector<std::vector<int>> shapes, IntervalCostModelConfig model,
                                             int planner_threads, std::vector<int> tail_repartition_widths)
    : num_cores_(num_cores),
      widths_(std::move(widths)),
      tail_repartition_widths_(std::move(tail_repartition_widths)),
      shapes_(std::move(shapes)),
      configured_workers_(planner_threads > 0 ? planner_threads : DefaultPlannerWorkers(num_cores)),
      impl_(std::make_unique<Impl>(std::move(model))) {
  if (num_cores_ <= 0) {
    throw std::invalid_argument("num_cores must be positive");
  }
  configured_workers_ = std::max(1, std::min(configured_workers_, num_cores_));
  if (widths_.empty() || shapes_.empty()) {
    throw std::invalid_argument("native interval planner requires widths and shapes");
  }
  for (const std::vector<int>& shape : shapes_) {
    int total = 0;
    for (int width : shape) {
      total += width;
      if (std::find(widths_.begin(), widths_.end(), width) == widths_.end()) {
        throw std::invalid_argument("candidate shape contains an unsupported width");
      }
    }
    if (total != num_cores_) {
      throw std::invalid_argument("candidate shape does not cover num_cores");
    }
  }
  std::sort(tail_repartition_widths_.begin(), tail_repartition_widths_.end());
  tail_repartition_widths_.erase(std::unique(tail_repartition_widths_.begin(), tail_repartition_widths_.end()),
                                 tail_repartition_widths_.end());
  for (int width : tail_repartition_widths_) {
    if (width <= 0 || width > num_cores_ / 2 || num_cores_ % width != 0) {
      throw std::invalid_argument("tail repartition widths must be divisors no wider than half the planner domain");
    }
  }
}

NativeIntervalPlanner::~NativeIntervalPlanner() = default;

double NativeIntervalPlanner::EstimateIsolated(int routes, int threads) const { return impl_->Tiso(routes, threads); }

double NativeIntervalPlanner::ScoreDag(const std::vector<int>& routes, const std::vector<int>& threads,
                                       const std::vector<std::vector<int>>& dependencies) const {
  if (routes.size() != threads.size() || routes.size() != dependencies.size()) {
    throw std::invalid_argument("routes, threads, and dependencies must have equal lengths");
  }
  std::vector<Impl::SimulationTask> tasks;
  tasks.reserve(routes.size());
  for (size_t index = 0; index < routes.size(); ++index) {
    tasks.push_back({routes[index], threads[index], dependencies[index]});
  }
  return impl_->DagMakespan(tasks);
}

IntervalPlanResult NativeIntervalPlanner::Plan(const std::vector<int>& expert_ids, const std::vector<int>& routes,
                                               bool dynamic_tail_pool, int tail_pool_max_routes,
                                               std::optional<int> forced_tail_pool_threads,
                                               bool bounded_tail_repartition) const {
  const std::vector<Expert> experts = BuildExperts(expert_ids, routes);
  std::vector<IntervalCandidate> strict_candidates(shapes_.size());
  ParallelFor(shapes_.size(), configured_workers_,
              [&](size_t index) { strict_candidates[index] = BuildStrictCandidate(*impl_, experts, shapes_[index]); });

  std::vector<IntervalCandidate> tail_repartition_candidates;
  if (bounded_tail_repartition && !forced_tail_pool_threads.has_value() && !tail_repartition_widths_.empty()) {
    const std::vector<IntervalCandidate> heads = TailPoolHeadCandidates(strict_candidates);
    struct TailSpec {
      size_t head = 0;
      int width = 0;
      int route_slices = 1;
    };
    std::vector<TailSpec> specs;
    for (size_t head = 0; head < heads.size(); ++head) {
      for (int width : tail_repartition_widths_) {
        specs.push_back({head, width, 1});
        if (4 * width == num_cores_) {
          specs.push_back({head, width, 2});
        }
      }
    }
    std::vector<std::optional<IntervalCandidate>> evaluated(specs.size());
    ParallelFor(specs.size(), configured_workers_, [&](size_t index) {
      const TailSpec& spec = specs[index];
      evaluated[index] = BuildBoundedTailCandidate(
          *impl_, experts, heads[spec.head], num_cores_, spec.width, spec.route_slices);
    });
    for (std::optional<IntervalCandidate>& candidate : evaluated) {
      if (candidate.has_value()) {
        tail_repartition_candidates.push_back(std::move(*candidate));
      }
    }
  }

  std::vector<IntervalCandidate> dynamic_candidates;
  if (dynamic_tail_pool || forced_tail_pool_threads.has_value()) {
    const std::vector<IntervalCandidate> heads =
        forced_tail_pool_threads.has_value() ? strict_candidates : TailPoolHeadCandidates(strict_candidates);
    const std::vector<int> thresholds =
        TailPoolThresholds(experts, tail_pool_max_routes, forced_tail_pool_threads.has_value());
    std::vector<int> pool_widths;
    if (forced_tail_pool_threads.has_value()) {
      pool_widths.push_back(*forced_tail_pool_threads);
    } else {
      for (int width : widths_) {
        if (width == 1 || width == 2 || width == 4) {
          pool_widths.push_back(width);
        }
      }
    }

    struct DynamicSpec {
      size_t head = 0;
      int threshold = 0;
      int width = 0;
    };
    std::vector<DynamicSpec> specs;
    for (size_t head = 0; head < heads.size(); ++head) {
      for (int threshold : thresholds) {
        for (int width : pool_widths) {
          specs.push_back({head, threshold, width});
        }
      }
    }
    std::vector<std::optional<IntervalCandidate>> evaluated(specs.size());
    ParallelFor(specs.size(), configured_workers_, [&](size_t index) {
      const DynamicSpec& spec = specs[index];
      evaluated[index] =
          BuildTailPoolCandidate(*impl_, experts, heads[spec.head], num_cores_, spec.width, spec.threshold);
    });
    for (std::optional<IntervalCandidate>& candidate : evaluated) {
      if (candidate.has_value()) {
        dynamic_candidates.push_back(std::move(*candidate));
      }
    }
  }

  std::vector<IntervalCandidate> candidates;
  if (forced_tail_pool_threads.has_value()) {
    if (dynamic_candidates.empty()) {
      throw std::invalid_argument("no valid forced tail-pool candidate");
    }
    candidates = dynamic_candidates;
  } else {
    candidates = strict_candidates;
    if (bounded_tail_repartition) {
      candidates.insert(candidates.end(), tail_repartition_candidates.begin(), tail_repartition_candidates.end());
    }
    if (dynamic_tail_pool) {
      candidates.insert(candidates.end(), dynamic_candidates.begin(), dynamic_candidates.end());
    }
  }

  IntervalPlanResult result;
  result.strict_candidates = static_cast<int>(strict_candidates.size());
  result.dynamic_candidates = static_cast<int>(dynamic_candidates.size());
  result.tail_repartition_candidates = static_cast<int>(tail_repartition_candidates.size());
  result.configured_workers = configured_workers_;
  result.selected = SelectCandidate(&candidates);
  result.candidates = std::move(candidates);
  return result;
}

}  // namespace moe_planner
