# Plan Schema

This file defines the first offline representation for CPU MoE scheduler output.
It is intentionally close to the design document:

```text
Plan -> Wave -> Team
```

The same schema can be serialized as JSON by simulators and later translated into
native C++ runtime structures.

## Plan

```json
{
  "kind": "GREEDY_MARGINAL_GAIN",
  "num_cores": 16,
  "active_experts": [
    {
      "expert_id": 7,
      "routes": 512,
      "route_bucket": 512
    }
  ],
  "waves": [
    {
      "wave_id": 0,
      "teams": [
        {
          "expert_id": 7,
          "routes": 512,
          "threads": 8,
          "estimated_time_ns": 240000
        }
      ],
      "estimated_wave_time_ns": 240000
    }
  ],
  "measured_plan_cost_ns": 150000000,
  "estimated_plan_cost_ns": 35000,
  "estimated_execute_cost_ns": 240000,
  "estimated_total_cost_ns": 275000,
  "exactness_scope": "heuristic",
  "scheduled_bridge": {
    "num_threads": 16,
    "wave_offsets": [0, 2, 3],
    "team_expert_ids": [7, 9, 13],
    "team_threads": [8, 4, 16]
  },
  "metadata": {
    "planner": "offline_simulator.py",
    "cost_metric": "median_ns",
    "planner_cost_source": "model",
    "planner_cost_model_ns": 35000
  }
}
```

## ExpertWork

```json
{
  "expert_id": 7,
  "routes": 512,
  "route_bucket": 512
}
```

Rules:

- `expert_id` is in `[0, num_experts)`.
- `routes > 0`.
- `route_bucket` is optional. It is useful when the plan is generated from a
  bucketized cost table.

## Team

```json
{
  "expert_id": 7,
  "routes": 512,
  "threads": 8,
  "estimated_time_ns": 240000
}
```

Rules:

- One team executes one expert workload once.
- `threads >= 1`.
- `estimated_time_ns = T_expert(routes, threads)`.
- First-stage models ignore `core_mask` and `numa_node`.

## Wave

```json
{
  "wave_id": 0,
  "teams": [],
  "estimated_wave_time_ns": 0
}
```

Rules:

- A wave executes all its teams in parallel.
- The sum of team threads in a wave must be `<= num_cores`.
- `estimated_wave_time_ns = max(team.estimated_time_ns for team in teams)`.
- Empty waves are invalid.

## Plan Cost

```text
estimated_execute_cost_ns = sum(wave.estimated_wave_time_ns)
estimated_total_cost_ns = estimated_plan_cost_ns + estimated_execute_cost_ns
```

`estimated_plan_cost_ns` is the planner cost used for scoring. In the offline
Python simulator this should usually come from the complexity-based planner cost
model, not Python wall time. The first complexity model uses features such as
active expert count, core count, cost-table lookup count, scan operations, sort
scale, and wave packing operations.

`measured_plan_cost_ns` records the local Python implementation time for
diagnostics. It is useful for understanding prototype overhead, but it should not
be interpreted as native runtime planner cost.

`T_dispatch` and `T_combine` are intentionally not included in the first offline
schema. If future experiments show they depend strongly on the plan shape, add
them as explicit plan-dependent fields instead of hiding them in metadata.

## Scheduled C++ Bridge

`scheduled_bridge` is the compact tensor representation accepted by the
experimental `fused_moe_bf16_tiled_scheduled` C++ entrypoint:

```json
{
  "num_threads": 8,
  "wave_offsets": [0, 4, 6],
  "team_expert_ids": [3, 8, 1, 6, 0, 5],
  "team_threads": [4, 2, 1, 1, 4, 4]
}
```

Rules:

- `num_threads` is the worker thread count passed to the C++ kernel.
- `wave_offsets` has length `num_waves + 1`.
- Teams in wave `i` are in
  `[wave_offsets[i], wave_offsets[i + 1])`.
- `team_expert_ids[j]` and `team_threads[j]` define one team.
- The sum of `team_threads` inside each wave must be `<= num_threads`.
- The first bridge implementation requires exactly one team per active expert.
  It does not yet support splitting one expert across multiple teams.
- Physical core ids are not represented yet. Current plans control logical
  thread grouping only.

The current C++ bridge accepts these arrays as `torch.int32` or another integer
CPU dtype and reconstructs the same wave/team structure natively.

## Planner Kinds

Current first-stage planner kinds:

- `FIXED_GLOBAL_THREADS`: one thread per expert, packed into waves.
- `UNIFORM_WAVES`: enumerate one uniform `threads_per_expert`.
- `GREEDY_MARGINAL_GAIN`: greedily add thread slots where the model predicts the
  largest expert-time reduction.
- `ENUMERATE_CORE_GROUPS`: enumerate integer core group shapes such as `[8]`,
  `[4, 4]`, `[4, 2, 1, 1]`, and `[1, 1, 1, 1, 1, 1, 1, 1]`; assign larger
  experts to larger slots and choose the shape with the lowest estimated
  execution cost. Current implementation caps the candidate shape list at 512
  entries for large core counts; 8-core and 16-core cases are fully enumerated.

## Exactness Scope

Allowed first-stage values:

- `baseline`: simple reference plan.
- `heuristic`: approximate planner with no strict optimality claim.
- `exact_within_enumerated_space`: exact optimum inside a finite candidate set.
- `exact_within_bucketized_space`: exact optimum after route bucketization.
- `offline_exact`: MILP/exhaustive result not intended for hot path use.
