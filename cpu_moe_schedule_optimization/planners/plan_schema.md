# Plan Schema

This file defines the versioned representation for CPU MoE scheduler output.

The production async bridge is now **Plan V2**. Strict execution remains the
default. The ARM native executor also accepts an explicit whole-expert
`tail_pool` placement: aligned thread groups may claim pooled experts only
after their fixed work completes. Neither mode resizes a running task.

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

`T_dispatch` and the duration/contention response of `T_combine` are not
included in the planner objective. Plan V2 does expose a conservative
`early_merge` execution choice, but it may only disable overlap when predicted
expert completion times coincide; it does not score an unmodeled combine gain.

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

## Async C++ Bridge: Plan V2

`ASYNC_INTERVAL_DAG` is translated to compact task arrays accepted by
`AsyncMoEPlanV2`. The runtime validates the full plan and passes it to the
native `fused_moe_bf16_tiled_async_plan_v2` entrypoint:

```json
{
  "plan_version": 2,
  "execution_mode": "strict",
  "num_threads": 8,
  "thread_cpu_ids": [0, 1, 2, 3, 4, 5, 6, 7],
  "task_expert_ids": [0, 3, 5],
  "task_core_begins": [0, 4, 6],
  "task_threads": [4, 2, 2],
  "task_dep_offsets": [0, 0, 1, 2],
  "task_deps": [0, 1],
  "task_preferred_threads": [4, 2, 2],
  "task_min_threads": [4, 2, 2],
  "task_max_threads": [4, 2, 2],
  "task_allowed_thread_offsets": [0, 1, 2, 3],
  "task_allowed_threads": [4, 2, 2],
  "task_placement_modes": [0, 0, 0],
  "task_numa_nodes": [-1, -1, -1],
  "task_stage_ids": [0, 0, 0],
  "task_resize_points": [0, 0, 0],
  "task_range_granularities": [0, 0, 0],
  "task_w13_window_bytes": [-1, 1048576, 4194304],
  "task_w2_window_bytes": [-1, 524288, 1048576],
  "task_resize_timeout_ns": [0, 0, 0],
  "task_preferred_core_begins": [-1, -1, -1],
  "early_merge": null
}
```

Rules:

- `plan_version` must be `2`; `execution_mode` is `strict`, `tail_pool`, or
  experimental `elastic`.
- `task_expert_ids`, `task_core_begins`, and `task_threads` have one entry per
  task.
- `task_dep_offsets` / `task_deps` are a CSR dependency list. Dependencies must
  refer to earlier task ids.
- Placement `0` is fixed and uses the logical-thread interval
  `[task_core_begins[i], task_core_begins[i] + task_threads[i])`. Placement
  `1` is a tail-pool task and requires `task_core_begins[i] = -1`.
- `task_allowed_thread_offsets` / `task_allowed_threads` are a second CSR list
  containing the legal discrete widths for every task. The entries for one task
  must be positive, unique, and strictly increasing.
- `task_preferred_threads[i]` and the selected `task_threads[i]` must both be in
  the task's allowed-width list. `task_min_threads` and `task_max_threads` must
  equal the first and last entries of that list.
- `task_numa_nodes=-1` means no additional NUMA constraint and remains required
  by strict/tail-pool execution. An elastic task with a resize point must carry
  an explicit NUMA node, and every physical CPU in its preferred cohort must
  resolve to that node.
- Stage id `0` means an expert pipeline task, resize mask `0` means no legal
  resize point, resize mask `1` means W13-to-W2 only, and range granularity
  `0` means the full expert task. A positive range granularity assigns the
  next contiguous route slice of that many rows to this occurrence of the
  expert id. Repeated occurrences are consumed in task-array order; the final
  slice may be shorter, but the slices must cover the expert exactly. Mask `1`
  is accepted only in elastic mode.
- `task_resize_timeout_ns` is optional and defaults to zero. A zero timeout is a
  nonblocking acquisition: the W2 stage expands only when all extra cohort
  workers are already idle, otherwise its original team continues immediately.
  A positive timeout lets W2-ready base teams in the same planner cohort
  rendezvous until the deadline. Once a task reaches its deadline, later
  cohorts cannot borrow its original team; it falls back as soon as any already
  running finite W2 job releases those workers. The timeout therefore bounds
  new-cohort formation, not preemption of an already running W2; observed
  ready-to-assignment delay can include at most that final finite cohort job.
- `task_preferred_core_begins` is optional and defaults to `-1`. For a
  resizable elastic task, `-1` derives the aligned preferred cohort containing
  the selected team. A non-negative value explicitly selects the logical-core
  start of W2's preferred cohort. The explicit target must be aligned to
  `task_preferred_threads`, fit within `num_threads`, remain on the task's NUMA
  node, and either fully contain the selected interval or be completely
  disjoint from it. Partial overlap is rejected. Strict, tail-pool, and
  non-resizable elastic tasks require `-1`.
- `task_w13_window_bytes` and `task_w2_window_bytes` are optional per-task SVE
  stage overrides. Missing arrays are materialized as `-1`; each present array
  must have one entry per task. `-1` inherits the operator-wide
  `weight_window_bytes`, `0` selects the stage's legacy range rule, and a
  positive value selects a tile-aligned nominal packed-B byte window. Non-SVE
  backends ignore the override. A pre-V2 native extension rejects a plan with
  any non-negative override instead of silently changing its execution.
- `early_merge` is an optional plan-level tri-state. Missing or `null` retains
  the runtime team-load heuristic, `true` forces the ready-token path, and
  `false` waits for expert compute to finish before all workers merge uniform
  contiguous token ranges. `true` requires SVE direct-route W2 and is rejected
  for elastic execution. `FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE=0` remains a
  global kill switch. A Python wrapper connected to a native extension without
  this argument rejects explicit `true`/`false` instead of ignoring it.
- Strict execution permits either one full-range task per active expert or
  multiple route-slice tasks. All repeated tasks for one expert must carry the
  same positive granularity. Mixing full-range and sliced tasks, leaving a
  route gap, or extending past the available task count is rejected natively.
  Tail-pool and elastic execution still require one full-range task per expert.
  An expert becomes ready for token merge only after all of its slices finish;
  each slice owns disjoint route rows and an independent fixed team/scratch
  interval.
- Strict mode requires every placement to be fixed and ignores the legacy
  short-pool environment variables.
- Tail-pool mode requires at least one pooled task. All pooled tasks use one
  selected width that divides `num_threads`; every fixed interval is aligned
  to that width. Pooled tasks have no dependencies, and fixed tasks cannot
  depend on pooled tasks. A released group claims one whole pooled expert at a
  time from the shared queue.
- Elastic mode requires fixed placement and the fused ARM SVE backend. A
  resizable task must have `preferred_threads > task_threads`, the preferred
  width must be an integer multiple of the selected width. With a derived
  target, the selected interval must fit in
  `[floor(core_begin/preferred)*preferred, ... + preferred)`. An explicit
  disjoint target migrates only W2: the runtime first acquires every target
  worker, then releases source workers outside the target. It never preempts a
  GEMM and never copies the task-owned packed-C intermediate. Failure to
  acquire the complete target preserves the original team and kernel path.
- Offline simulator JSON includes this representation under `async_bridge`.
- The production planner currently emits a singleton allowed-width list for
  every task:

  ```text
  allowed_threads[i] = {task_threads[i]}
  preferred_threads[i] = min_threads[i] = max_threads[i] = task_threads[i]
  ```

  `AsyncMoEPlanV2` accepts a wider envelope so cached/offline plans can be
  forward-compatible. Strict/tail-pool still use `task_threads` exactly after
  a task starts; elastic may select `task_preferred_threads` only for W2.
- `upgrade_legacy_async_plan()` converts the previous fixed-width dictionary to
  this singleton-width, all-fixed representation.
- `fused_moe_bf16_tiled_async_plan()` is the public adapter. Callers
  should materialize and cache `AsyncMoEPlanV2` outside the timed operator path.
  A pre-V2 native extension can execute strict plans through the legacy entry,
  but tail-pool plans require the V2 symbol and elastic plans require the
  dedicated experimental elastic symbol.
- `IntervalPlanner.to_elastic_w2_bridge()` lowers explicit planner transitions
  such as `8:16` and `2:8` into aligned local cohorts. The bridge may restrict
  resizing to selected task ids and attach an explicit W2 target start to each
  selected task, enabling planner-specified disjoint cohorts such as
  `16-31 -> 32-63`. This is currently a benchmark/validation bridge, not an
  automatically searched production candidate.
- `IntervalPlanner` and `PlannedMoE` compare the original strict execution with
  eligible whole-expert tail-pool and bounded terminal-repartition candidates
  by default. The tail pool searches route buckets `1/2/4/8/12` at or below
  `tail_pool_max_routes=12`, aligned `1/2/4T` pool widths, and
  strict-competitive head shapes. Duplicate pooled sets are removed before it
  lowers each online list schedule to a surrogate DAG and scores it with the
  current contention model.
- A bounded terminal repartition is still a strict bridge. It is legal only
  when the selected head has exactly two second-wave tasks and both have no
  successor. On a 96-worker planner domain it searches `24/32/48T`, places the
  two tasks in disjoint half-domain intervals, and replaces their dependencies
  with every first-wave task overlapping the new interval. Metadata records
  `tail_repartition_width`, `tail_repartition_tasks`, and
  `tail_repartition_candidates`; the executor needs no new mode or handoff.
- `dynamic_tail_pool=False` with an omitted bounded-tail option requests the
  original strict baseline. Passing `bounded_tail_repartition=True` explicitly
  evaluates the bounded candidate without enabling the dynamic tail pool.
  `tail_pool_threads=T` remains a forced override, but shape selection is still
  performed against that tail-pool policy instead of rewriting an already
  selected strict shape.
- The plan cache separates strict, automatic, forced, and bounded-tail
  policies. It records the eligible-expert count at every searched route
  threshold. A bounded-tail hit reruns the cheap LPT assignment and validates
  that the cached width still applies to exactly two terminal second-wave
  tasks and remains supported by the isolated model at the new route counts;
  a topology or calibration mismatch evicts the entry and reruns cold search.
- The runtime, rather than the planner, observes actual fixed-task completion
  and assigns the next pooled task. The surrogate is therefore a selection
  model, not an exact prediction of runtime claim order.
- For schema-v2 empirical profiles, an available current extension runs cold
  candidate search in C++ and parallelizes independent candidates. Planner
  output records `planner_backend`, `planner_workers`, `strict_candidates`,
  `dynamic_candidates`, and `tail_repartition_candidates`; these are
  diagnostics and are not part of the runtime bridge identity.
  `FUSED_CPP_MOE_NATIVE_COLD_PLANNER=0` selects the Python
  reference implementation, while `FUSED_CPP_MOE_PLANNER_THREADS` controls
  native candidate workers. Cache hits rebuild the same bridge without rerunning
  either cold solver.
- Profiled kernel selection remains operator-wide. A policy-aware plan emits
  adjacent `operator_options`:

  ```json
  {
    "w13_split": false,
    "weight_window_bytes": 2097152
  }
  ```

  A positive window serializes both SVE GEMMs into tile-aligned packed-B
  ranges. It is selected only from an exact schema-v2 profile carrying that
  target and the actual W13/W2 range counts.

  An optional named `TaskStageWindowPolicy` may populate the two per-task
  arrays after task widths and placements have been selected. The current
  static policy is a deterministic lookup on `(routes, actual_task_threads)`;
  unsupported combinations emit `-1`. It does not add candidates or change
  cost-model scores, and tail-pool tasks use the selected pool width. The
  policy name is part of `PlannedMoE` cache identity and result metadata.
  `PlannedMoE` resolves the policy independently for every candidate model.
  The measured `amazon_c5_192c_tp4_f512_v1` policy is default-on only when the
  complete dual-NUMA AmazonC5192Cores TP4/F512 SVE JIT exact-M split profile
  identity and one of its 96-core rank CPU sets match. No-split and all other
  profiles inherit their operator-wide policy. Pass
  `use_default_stage_window_policy=False` for a controlled baseline.

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
