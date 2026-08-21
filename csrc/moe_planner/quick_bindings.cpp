// SPDX-License-Identifier: Apache-2.0
#include "quick_bindings.h"

#include <torch/extension.h>

#include <pybind11/stl.h>

#include <stdexcept>
#include <utility>
#include <vector>

#include "interval_planner.h"

namespace py = pybind11;

namespace {

using moe_planner::IntervalAssignmentOrder;
using moe_planner::IntervalCandidate;
using moe_planner::IntervalPlanResult;
using moe_planner::IntervalTask;
using moe_planner::NativeQuickPlanner;

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

}  // namespace

void register_moe_quick_planner(py::module_& m) {
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
}
