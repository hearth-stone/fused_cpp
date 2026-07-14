@../WORKSPACE.md
@AGENT.md

## Local Notes

- This subproject is an independent C++/Python extension workspace.
- Before changing SDPA or microkernel code, read the local SDPA TODO/version
  docs and keep implementation, tests, benchmarks, and docs synchronized.
- Before changing the CPU MoE scheduling model, planner decision space, cost
  model resources, or pruning rules, read
  `cpu_moe_schedule_optimization/MATHEMATICAL_MODEL.md` and update its formulas,
  pruning table, validation notes, and changelog in the same change.
