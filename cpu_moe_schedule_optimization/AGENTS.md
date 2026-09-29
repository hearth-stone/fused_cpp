# CPU MoE Planner And Cost Model Guide

These rules apply under `cpu_moe_schedule_optimization/` in addition to the
repository guide.

## Required Model Synchronization

- Read `MATHEMATICAL_MODEL.md` before changing planner decisions, candidate
  shapes, pruning, resource accounting, contention, stage geometry, or
  calibration interpretation.
- Update the relevant formulas, assumptions, pruning table, validation notes,
  and changelog in the same change. A code-only model change is incomplete.
- Keep `planners/plan_schema.md` and `cost_model/profile_schema.md` synchronized
  with serialized plan or profile changes.

## Boundaries

- Separate hardware-independent formulas from machine calibration. Machine data
  may correct parameters; it must not silently redefine model semantics.
- Keep empirical profiles machine/topology/ISA/shape specific and reject
  incompatible profiles rather than extrapolating silently.
- Keep Python and native C++ planner semantics aligned: candidate construction,
  pruning, tie-breaking, uncertainty, and selected plan fields must agree.
- Planner output must fully describe runtime behavior. Do not rely on hidden
  process environment to reinterpret an otherwise identical plan.
- Do not expand the search space without documenting its size, pruning rule,
  cold-planning cost, and expected benefit.

## Calibration And Results

- Record machine identity, topology, ISA, backend N tile, page policy, shape,
  route grid, thread widths, repetitions, and source commit with calibration.
- Do not overwrite an existing calibration with data from a different machine
  or execution geometry. Create a distinct profile identity.
- Keep raw traces and large profiler output outside Git. Commit compact profiles
  only when production consumes them; otherwise commit a concise result report.

## Validation

- Use focused tests from `tests/test_moe_analytic_model.py`,
  `tests/test_moe_cost_model_v2.py`, `tests/test_moe_native_interval_planner.py`,
  `tests/test_moe_plan_v2.py`, and `tests/test_moe_stage_window_plan.py`.
- Formula changes require unit/limit checks plus holdout validation on every
  claimed machine class.
- Planner changes require deterministic plan tests, Python/native parity, cold
  planning time, predicted makespan, actual makespan where available, and regret
  against the declared oracle or measured candidate set.
- Never describe a model as converged from training/calibration fit alone;
  report holdout error, unsupported regions, and uncertainty.
