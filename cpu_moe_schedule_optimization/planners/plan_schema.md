# Plan Schema

This file defines the first offline representation for CPU MoE scheduler output.

> **⚠ DEPRECATED — wave 调度后续不考虑。** 见 [../DEPRECATED_WAVE.md](../DEPRECATED_WAVE.md)。
> 本文件中的 `Wave` 层、`wave_offsets` scheduled bridge、以及除 `ASYNC_INTERVAL_DAG` 外的
> 所有 planner kinds 均已废弃,仅作历史参考。**保留并继续**:`ASYNC_INTERVAL_DAG` /
> async bridge、`Team`、cost model。
It is intentionally close to the design document:

```text
Plan -> Wave -> Team
Plan -> AsyncTask DAG
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
  "async_tasks": [],
  "async_bridge": null,
  "runtime_bridge": "wave_offsets",
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

> **⚠ DEPRECATED(wave,后续不考虑)** — 见 ../DEPRECATED_WAVE.md。

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

## AsyncTask

`ASYNC_INTERVAL_DAG` uses a second representation instead of global waves:

```json
{
  "task_id": 3,
  "expert_id": 12,
  "route_begin": 0,
  "route_count": 384,
  "routes": 384,
  "threads": 2,
  "core_begin": 1,
  "core_end": 3,
  "deps": [0],
  "estimated_time_ns": 9000000,
  "estimated_start_ns": 12000000,
  "estimated_finish_ns": 21000000
}
```

Rules:

- One async task executes one expert workload on a contiguous logical-core
  interval `[core_begin, core_end)`.
- `deps` contains tasks that must complete before this task can start. These
  dependencies are derived from core interval reuse, so they encode local
  split/join constraints without a global wave barrier.
- `estimated_execute_cost_ns` is the makespan:

```text
max(task.estimated_finish_ns for task in async_tasks)
```

- Async plans set `scheduled_bridge = null` because they are not representable
  as global waves. They set `runtime_bridge = "async_task_dag"` and also expose
  an `async_bridge` compact tensor representation.

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

> **⚠ DEPRECATED(wave bridge,后续不考虑)。** 下文 async bridge 保留。

`scheduled_bridge` is the compact tensor representation accepted by the
experimental `fused_moe_bf16_tiled_scheduled` C++ entrypoint:

```json
{
  "num_threads": 8,
  "thread_cpu_ids": [0, 1, 2, 3, 4, 5, 6, 7],
  "wave_offsets": [0, 4, 6],
  "team_expert_ids": [3, 8, 1, 6, 0, 5],
  "team_threads": [4, 2, 1, 1, 4, 4]
}
```

Rules:

- `num_threads` is the worker thread count passed to the C++ kernel.
- `thread_cpu_ids` maps logical worker thread id to a physical CPU id. It must
  have length `num_threads`. If omitted by a legacy caller, the runtime falls
  back to the environment/default pinning policy.
- `wave_offsets` has length `num_waves + 1`.
- Teams in wave `i` are in
  `[wave_offsets[i], wave_offsets[i + 1])`.
- `team_expert_ids[j]` and `team_threads[j]` define one team.
- The sum of `team_threads` inside each wave must be `<= num_threads`.
- The first bridge implementation requires exactly one team per active expert.
  It does not yet support splitting one expert across multiple teams.

The current C++ bridge accepts these arrays as `torch.int32` or another integer
CPU dtype and reconstructs the same wave/team structure natively.

`ASYNC_INTERVAL_DAG` uses a separate async task-DAG bridge instead of this wave
bridge.

## Async C++ Bridge

`ASYNC_INTERVAL_DAG` is translated to compact task arrays accepted by
`fused_moe_bf16_tiled_async`:

```json
{
  "num_threads": 8,
  "thread_cpu_ids": [0, 1, 2, 3, 4, 5, 6, 7],
  "task_expert_ids": [0, 3, 5],
  "task_core_begins": [0, 4, 6],
  "task_threads": [4, 2, 2],
  "task_dep_offsets": [0, 0, 1, 2],
  "task_deps": [0, 1]
}
```

Rules:

- `task_expert_ids`, `task_core_begins`, and `task_threads` have one entry per
  task.
- `task_dep_offsets` / `task_deps` are a CSR dependency list. Dependencies must
  refer to earlier task ids.
- Each task uses the logical-thread interval
  `[task_core_begins[i], task_core_begins[i] + task_threads[i])`.
- The first async bridge supports exactly one task per active expert.
- Offline simulator JSON includes this representation under `async_bridge`.

## Planner Kinds

Current first-stage planner kinds:

> **⚠ 除 `ASYNC_INTERVAL_DAG` 外的所有 kinds 均已 DEPRECATED(wave,后续不考虑)。**

- `FIXED_GLOBAL_THREADS`: one thread per expert, packed into waves.
- `SORTED_TOKEN_BALANCED_1T`: sort active experts by routed-token count and
  greedily assign each one-thread expert to the lightest logical core queue;
  the queues are then emitted as wave-aligned scheduled teams. This is the
  default strong baseline for comparing richer planners.
- `UNIFORM_WAVES`: enumerate one uniform `threads_per_expert`.
- `GREEDY_MARGINAL_GAIN`: greedily add thread slots where the model predicts the
  largest expert-time reduction.
- `ENUMERATE_CORE_GROUPS`: enumerate integer core group shapes such as `[8]`,
  `[4, 4]`, `[4, 2, 1, 1]`, and `[1, 1, 1, 1, 1, 1, 1, 1]`; assign larger
  experts to larger slots and choose the shape with the lowest estimated
  execution cost. Current implementation caps the candidate shape list at 512
  entries for large core counts; 8-core and 16-core cases are fully enumerated.
- `ASYNC_INTERVAL_DAG`: list-schedule experts onto contiguous logical-core
  intervals. Each task depends only on previous tasks that used overlapping
  cores, so unrelated intervals can advance independently. This models the
  precomputed + dynamic-triggered schedule needed to avoid global wave waits.

## Exactness Scope

Allowed first-stage values:

- `baseline`: simple reference plan.
- `heuristic`: approximate planner with no strict optimality claim.
- `exact_within_enumerated_space`: exact optimum inside a finite candidate set.
- `exact_within_bucketized_space`: exact optimum after route bucketization.
- `offline_exact`: MILP/exhaustive result not intended for hot path use.
