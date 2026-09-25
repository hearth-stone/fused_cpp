// SPDX-License-Identifier: Apache-2.0
#include "quick_bindings.h"

#include <torch/extension.h>

#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "analytic_placed_dag.h"
#include "probe_event_sim.h"
#include "hot_wide_planner.h"
#include "interval_planner.h"

namespace py = pybind11;

namespace {

using moe_planner::AnalyticDagMachine;
using moe_planner::AnalyticDagPhase;
using moe_planner::AnalyticDagTasks;
using moe_planner::IntervalAssignmentOrder;
using moe_planner::NativeAnalyticPlacedDag;
using moe_planner::IntervalCandidate;
using moe_planner::IntervalPlanResult;
using moe_planner::IntervalTask;
using moe_planner::NativeHotWidePlanner;
using moe_planner::NativeProbeEventSim;
using moe_planner::NativeQuickPlanner;
using moe_planner::ProbeEventProfile;

const char* AssignmentOrderName(IntervalAssignmentOrder order) {
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

py::list TasksToPython(const std::vector<IntervalTask>& tasks) {
  py::list result;
  for (const IntervalTask& task : tasks) {
    result.append(py::make_tuple(task.expert_id, task.routes, task.core_begin, task.threads, task.dependencies));
  }
  return result;
}

py::dict CandidateToPython(const IntervalCandidate& candidate, bool include_tasks) {
  py::dict result;
  result["shape"] = candidate.shape;
  result["assignment_order"] = AssignmentOrderName(candidate.assignment_order);
  result["execution_mode"] = "strict";
  result["tail_pool_threads"] = py::none();
  result["tail_pool_max_routes"] = py::none();
  result["tail_pool_tasks"] = 0;
  result["tail_repartition_width"] = py::none();
  result["tail_repartition_tasks"] = 0;
  result["tail_repartition_route_slices"] = 1;
  result["makespan_ns"] = candidate.makespan_ns;
  result["uncertainty_ns"] = candidate.uncertainty_ns;
  result["pessimistic_ns"] = candidate.pessimistic_ns;
  if (include_tasks) {
    result["tasks"] = TasksToPython(candidate.tasks);
  }
  result["active_working_set_bytes"] = candidate.active_working_set_bytes;
  result["window_bytes_per_worker"] = candidate.window_bytes_per_worker;
  result["resource_groups"] = candidate.resource_groups;
  return result;
}

py::dict Plan(const NativeQuickPlanner& planner, const std::vector<int>& expert_ids,
              const std::vector<int>& routes, const std::vector<std::vector<double>>& costs) {
  IntervalPlanResult native_result;
  {
    py::gil_scoped_release release;
    native_result = planner.Plan(expert_ids, routes, costs);
  }
  py::dict result;
  result["selected"] = CandidateToPython(native_result.selected, true);
  py::list candidates;
  for (const IntervalCandidate& candidate : native_result.candidates) {
    candidates.append(CandidateToPython(candidate, false));
  }
  result["candidates"] = std::move(candidates);
  result["configured_workers"] = native_result.configured_workers;
  result["strict_candidates"] = native_result.strict_candidates;
  return result;
}

py::dict Assign(const NativeQuickPlanner& planner, const std::vector<int>& expert_ids,
                const std::vector<int>& routes, const std::vector<int>& shape,
                const std::vector<double>& costs) {
  IntervalCandidate candidate;
  {
    py::gil_scoped_release release;
    candidate = planner.Assign(expert_ids, routes, shape, costs);
  }
  return CandidateToPython(candidate, true);
}

py::dict PlanShared(const NativeQuickPlanner& planner, const std::vector<int>& expert_ids,
                    const std::vector<int>& routes, int shared_expert_id,
                    const std::vector<std::vector<int>>& shapes, const std::vector<int>& cost_widths,
                    const std::vector<std::vector<double>>& costs_by_width) {
  IntervalPlanResult native_result;
  {
    py::gil_scoped_release release;
    native_result = planner.PlanShared(expert_ids, routes, shared_expert_id, shapes, cost_widths, costs_by_width);
  }
  py::dict result;
  result["selected"] = CandidateToPython(native_result.selected, true);
  py::list candidates;
  for (const IntervalCandidate& candidate : native_result.candidates) {
    candidates.append(CandidateToPython(candidate, false));
  }
  result["candidates"] = std::move(candidates);
  result["configured_workers"] = native_result.configured_workers;
  result["strict_candidates"] = native_result.strict_candidates;
  return result;
}

py::dict HotWidePlan(const NativeHotWidePlanner& planner, const std::vector<int>& expert_ids,
                     const std::vector<int>& routes, const std::vector<int>& cost_widths,
                     const std::vector<std::vector<double>>& costs_by_width) {
  NativeHotWidePlanner::Plan plan;
  {
    py::gil_scoped_release release;
    plan = planner.PlanExperts(expert_ids, routes, cost_widths, costs_by_width);
  }
  py::dict result;
  result["tasks"] = TasksToPython(plan.tasks);
  result["shape"] = plan.shape;
  result["score_ns"] = plan.score_ns;
  result["templates"] = plan.templates;
  return result;
}

// (gemm, active_threads, fixed_ns, residual_scale, base_ns, working_set_bytes, gemm_demand,
//  l2_demand, llc_demand, epilogue_demand, compulsory_dram_bytes, spillable_dram_bytes,
//  dram_rate, gemm_ns, l2_ns, llc_ns, epilogue_ns, loading)
using PhaseTuple = std::tuple<bool, int, double, double, double, double, double, double, double, double, double,
                              double, double, double, double, double, double, bool>;

AnalyticDagPhase PhaseFromTuple(const PhaseTuple& value) {
  AnalyticDagPhase phase;
  std::tie(phase.gemm, phase.active_threads, phase.fixed_ns, phase.residual_scale, phase.base_ns,
           phase.working_set_bytes, phase.gemm_demand, phase.l2_demand, phase.llc_demand, phase.epilogue_demand,
           phase.compulsory_dram_bytes, phase.spillable_dram_bytes, phase.dram_rate, phase.gemm_ns, phase.l2_ns,
           phase.llc_ns, phase.epilogue_ns, phase.loading) = value;
  return phase;
}

using PlacedTasksTuple =
    std::tuple<std::vector<int64_t>, std::vector<int>, std::vector<std::vector<int>>, std::vector<std::vector<int>>>;

AnalyticDagTasks TasksFromTuple(PlacedTasksTuple value) {
  AnalyticDagTasks tasks;
  tasks.routes = std::move(std::get<0>(value));
  tasks.threads = std::move(std::get<1>(value));
  tasks.cpu_ids = std::move(std::get<2>(value));
  tasks.dependencies = std::move(std::get<3>(value));
  return tasks;
}

using AggregateTasksTuple = std::tuple<std::vector<int64_t>, std::vector<int>, std::vector<std::vector<int>>>;

AnalyticDagTasks AggregateTasksFromTuple(AggregateTasksTuple value) {
  AnalyticDagTasks tasks;
  tasks.routes = std::move(std::get<0>(value));
  tasks.threads = std::move(std::get<1>(value));
  tasks.dependencies = std::move(std::get<2>(value));
  tasks.cpu_ids.resize(tasks.routes.size());
  return tasks;
}

}  // namespace


void register_moe_quick_planner(py::module_& m) {
  py::class_<NativeHotWidePlanner>(m, "NativeHotWidePlanner", py::module_local())
      .def(py::init<int, std::vector<int>, int, std::vector<int>, int, int, std::vector<std::pair<int, double>>>(),
           py::arg("num_cores"), py::arg("domain_cores"), py::arg("bulk_width"), py::arg("wide_widths"),
           py::arg("max_wide_lanes"), py::arg("max_wide_cores"), py::arg("lane_scale"))
      .def("plan", &HotWidePlan, py::arg("expert_ids"), py::arg("routes"), py::arg("cost_widths"),
           py::arg("costs_by_width"))
      .def_property_readonly("shapes", [](const NativeHotWidePlanner& planner) { return planner.shapes(); });

  py::class_<NativeQuickPlanner>(m, "NativeQuickPlanner", py::module_local())
      .def(py::init<int, std::vector<std::vector<int>>, int64_t, std::vector<std::pair<int, int64_t>>, double, int,
                    int>(),
           py::arg("num_cores"), py::arg("shapes"), py::arg("max_stage_bytes"), py::arg("window_bytes_by_width"),
           py::arg("relative_error"), py::arg("profile_runs") = 1, py::arg("planner_threads") = 1)
      .def("plan", &Plan, py::arg("expert_ids"), py::arg("routes"), py::arg("costs"))
      .def("plan_shared", &PlanShared, py::arg("expert_ids"), py::arg("routes"), py::arg("shared_expert_id"),
           py::arg("shapes"), py::arg("cost_widths"), py::arg("costs_by_width"))
      .def("assign", &Assign, py::arg("expert_ids"), py::arg("routes"), py::arg("shape"), py::arg("costs"))
      .def_property_readonly("configured_workers", [](const NativeQuickPlanner& planner) {
        return planner.configured_workers();
      });

  py::class_<NativeAnalyticPlacedDag>(m, "NativeAnalyticPlacedDag", py::module_local())
      .def(py::init([](int cores_per_rank, double call_setup_ns, std::vector<int> cpu_domain,
                       std::vector<int> domain_sizes, std::vector<double> domain_capacity_bytes,
                       double llc_effective_fraction, std::vector<double> gemm_rate, std::vector<double> l2_rate,
                       std::vector<double> dram_rate, std::vector<double> epilogue_rate,
                       std::vector<std::vector<double>> domain_llc_rate, std::vector<double> llc_rate,
                       double llc_saturated_rate, double rank_llc_capacity_bytes, bool dram_injection,
                       double dram_injection_capacity_scale, double dram_saturated_rate,
                       std::vector<double> wide_isolated_scale, std::vector<double> wide_full_cohort_scale,
                       std::vector<double> narrow_full_cohort_correction,
                       std::vector<double> dram_multi_stream_rate, double loading_steady_dilation,
                       double merge_route_thread_ns, double merge_fixed_ns) {
             AnalyticDagMachine machine;
             machine.cores_per_rank = cores_per_rank;
             machine.call_setup_ns = call_setup_ns;
             machine.cpu_domain = std::move(cpu_domain);
             machine.domain_sizes = std::move(domain_sizes);
             machine.domain_capacity_bytes = std::move(domain_capacity_bytes);
             machine.llc_effective_fraction = llc_effective_fraction;
             machine.gemm_rate = std::move(gemm_rate);
             machine.l2_rate = std::move(l2_rate);
             machine.dram_rate = std::move(dram_rate);
             machine.epilogue_rate = std::move(epilogue_rate);
             machine.domain_llc_rate = std::move(domain_llc_rate);
             machine.llc_rate = std::move(llc_rate);
             machine.llc_saturated_rate = llc_saturated_rate;
             machine.rank_llc_capacity_bytes = rank_llc_capacity_bytes;
             machine.dram_injection = dram_injection;
             machine.dram_injection_capacity_scale = dram_injection_capacity_scale;
             machine.dram_saturated_rate = dram_saturated_rate;
             machine.wide_isolated_scale = std::move(wide_isolated_scale);
             machine.wide_full_cohort_scale = std::move(wide_full_cohort_scale);
             machine.narrow_full_cohort_correction = std::move(narrow_full_cohort_correction);
             machine.dram_multi_stream_rate = std::move(dram_multi_stream_rate);
             machine.loading_steady_dilation = loading_steady_dilation;
             machine.merge_route_thread_ns = merge_route_thread_ns;
             machine.merge_fixed_ns = merge_fixed_ns;
             return NativeAnalyticPlacedDag(std::move(machine));
           }),
           py::arg("cores_per_rank"), py::arg("call_setup_ns"), py::arg("cpu_domain"), py::arg("domain_sizes"),
           py::arg("domain_capacity_bytes"), py::arg("llc_effective_fraction"), py::arg("gemm_rate"),
           py::arg("l2_rate"), py::arg("dram_rate"), py::arg("epilogue_rate"), py::arg("domain_llc_rate"),
           py::arg("llc_rate"), py::arg("llc_saturated_rate"), py::arg("rank_llc_capacity_bytes"),
           py::arg("dram_injection"), py::arg("dram_injection_capacity_scale"), py::arg("dram_saturated_rate"),
           py::arg("wide_isolated_scale"), py::arg("wide_full_cohort_scale"),
           py::arg("narrow_full_cohort_correction"), py::arg("dram_multi_stream_rate") = std::vector<double>{},
           py::arg("loading_steady_dilation") = 0.0, py::arg("merge_route_thread_ns") = 0.0,
           py::arg("merge_fixed_ns") = 0.0)
      .def(
          "register_phases",
          [](NativeAnalyticPlacedDag& dag, int64_t routes, int threads, const std::vector<PhaseTuple>& phases) {
            std::vector<AnalyticDagPhase> converted;
            converted.reserve(phases.size());
            for (const PhaseTuple& phase : phases) {
              converted.push_back(PhaseFromTuple(phase));
            }
            dag.register_phases(routes, threads, std::move(converted));
          },
          py::arg("routes"), py::arg("threads"), py::arg("phases"))
      .def("has_phases", &NativeAnalyticPlacedDag::has_phases, py::arg("routes"), py::arg("threads"))
      .def(
          "makespan",
          [](const NativeAnalyticPlacedDag& dag, PlacedTasksTuple tasks) {
            return dag.makespan(TasksFromTuple(std::move(tasks)));
          },
          py::arg("tasks"))
      .def(
          "makespans",
          [](const NativeAnalyticPlacedDag& dag, std::vector<PlacedTasksTuple> batch, int workers) {
            std::vector<AnalyticDagTasks> converted;
            converted.reserve(batch.size());
            for (PlacedTasksTuple& tasks : batch) {
              converted.push_back(TasksFromTuple(std::move(tasks)));
            }
            return dag.makespans(converted, workers);
          },
          py::arg("batch"), py::arg("workers") = 1)
      .def(
          "makespan_aggregate",
          [](const NativeAnalyticPlacedDag& dag, AggregateTasksTuple tasks) {
            return dag.makespan_aggregate(AggregateTasksFromTuple(std::move(tasks)));
          },
          py::arg("tasks"))
      .def(
          "finish_times_aggregate",
          [](const NativeAnalyticPlacedDag& dag, AggregateTasksTuple tasks) {
            std::vector<double> finish_times;
            const double makespan = dag.makespan_aggregate(AggregateTasksFromTuple(std::move(tasks)), &finish_times);
            return py::make_tuple(makespan, finish_times);
          },
          py::arg("tasks"));

  py::class_<NativeProbeEventSim>(m, "NativeProbeEventSim", py::module_local())
      .def(py::init<int, double>(), py::arg("resolution"), py::arg("g0"))
      .def("add_table", &NativeProbeEventSim::add_table, py::arg("values"))
      .def(
          "add_profile",
          [](NativeProbeEventSim& sim, std::vector<double> phase_ns, std::vector<bool> phase_load, double overhead_ns,
             int width, double steady_load_cores, double steady_steady_cores, int ll_table, int ls_table,
             int sl_table, int ss_table, bool windowed, double r_cal, double e_cal) {
            ProbeEventProfile profile;
            profile.phase_ns = std::move(phase_ns);
            profile.phase_load.assign(phase_load.begin(), phase_load.end());
            profile.overhead_ns = overhead_ns;
            profile.width = width;
            profile.steady_load_cores = steady_load_cores;
            profile.steady_steady_cores = steady_steady_cores;
            profile.ll_table = ll_table;
            profile.ls_table = ls_table;
            profile.sl_table = sl_table;
            profile.ss_table = ss_table;
            profile.windowed = windowed;
            profile.r_cal = r_cal;
            profile.e_cal = e_cal;
            return sim.add_profile(std::move(profile));
          },
          py::arg("phase_ns"), py::arg("phase_load"), py::arg("overhead_ns"), py::arg("width"),
          py::arg("steady_load_cores"), py::arg("steady_steady_cores"), py::arg("ll_table"), py::arg("ls_table"),
          py::arg("sl_table"), py::arg("ss_table"), py::arg("windowed"), py::arg("r_cal"), py::arg("e_cal"))
      .def(
          "simulate",
          [](const NativeProbeEventSim& sim, const std::vector<int>& profiles,
             const std::vector<std::vector<int>>& dependencies) {
            auto result = sim.simulate(profiles, dependencies);
            return py::make_tuple(result.makespan_ns, result.finish_ns);
          },
          py::arg("profiles"), py::arg("dependencies"))
      .def("makespans", &NativeProbeEventSim::makespans, py::arg("profiles"), py::arg("dependencies"),
           py::arg("workers") = 1);
}
