# Expert Cost Profile Schema

## Schema v2: calibration-domain- and topology-bound contention profile

Schema v2 is the active format for SVE fused-MoE scheduling profiles. A profile
is valid only for the exact implementation, sharded expert shape, and NUMA/rank
execution context recorded in the file. The only active SVE geometry is
`full_n_team_stripes`: W13 and W2 each traverse their complete packed N domain
and team width partitions N tiles among workers. Plan V2 may further divide one
worker's complete owner stripe into serial tile windows. That is an execution
order within the same full-N geometry, not a weight-range/profile variant.

Paper-facing use of empirical profiles and current rerun requirements are
indexed in [`../../docs/moe_paper_readiness.md`](../../docs/moe_paper_readiness.md).

Analytical machine calibrations may include an optional
`overheads.by_width` list of discrete `{threads, expert_fixed_ns, route_ns}`
overrides. Widths must be unique and positive and costs must be non-negative.
A missing width inherits the scalar `expert_fixed_ns` and `route_ns`; the model
never interpolates or extrapolates these overheads. The complete analytical
calibration is part of the persisted `T_iso` cache identity, so adding or
changing an override invalidates stale scalar costs automatically.

Analytical machine calibrations may also include an optional
`planner.narrow_team_contention_correction.full_cohort_correction` list of
discrete `{threads, correction}` points. This is a residual correction to the
placed event model's shared-resource dilation, not an additional physical
slowdown: a value below one is legal when the resource equations overpredict
contention. For peer occupancy fraction `q`, the applied factor is
`1 + (correction - 1) * q`, and the final phase dilation is clamped to at least
one. Widths must be positive and unique and corrections finite and positive;
missing widths use one, with no interpolation or extrapolation. The first
calibration covers only 1T/2T BF16 narrow teams in one fully occupied 80-core
mixed cohort. Historical files without the field retain their previous scores.

Historical schema-v2 files may record `w13_split`, `w13_split_chunks`,
`weight_window_bytes`, `w13_window_ranges`, or `w2_window_ranges`. They are not
active calibration inputs. A non-unit legacy range pair is rejected because its
timings describe a different loop order. Only one full-N calibration may exist
for a complete domain identity.

```json
{
  "schema_version": 2,
  "kind": "contention_derate",
  "target": {
    "host_logical_cores": 64,
    "aggregate_profiled_cores": 64,
    "cores_per_rank": 32,
    "concurrent_ranks": 2,
    "cpu_ids_by_rank": [[0, 1], [32, 33]],
    "numa_nodes": [0, 1],
    "llc_bytes_by_rank": [50331648, 50331648]
  },
  "kernel": {
    "name": "fused_moe_bf16_tiled_async",
    "backend": "sve",
    "backend_n_tile": 8,
    "parallel_axis": "N",
    "sve_implementation": "jit",
    "m_tail_policy": "xbyak_exact_m",
    "stage_geometry": "full_n_team_stripes",
    "git_available": true,
    "git_commit": "...",
    "git_worktree_dirty": true,
    "xbyak_aarch64_commit": "...",
    "source_sha256": "...",
    "extension_sha256": "..."
  },
  "parallelism": {
    "mode": "tp",
    "degree": 2,
    "global_experts": 64,
    "local_experts": 64
  },
  "expert_shape": {
    "dtype": "bf16",
    "hidden_size": 4096,
    "intermediate_size": 1024,
    "measurement_experts": 64,
    "isolated_measurement_experts": 8,
    "weight_reuse": "streaming_distinct_experts"
  },
  "working_set": {
    "w13_packed_bytes_per_expert": 16777216,
    "w2_packed_bytes_per_expert": 8388608,
    "w13_stage_bytes_per_expert": 16777216,
    "w2_stage_bytes_per_expert": 8388608,
    "max_weight_stage_bytes_per_expert": 16777216
  },
  "measurement": {
    "rank_synchronization": "socket_barrier_per_call",
    "rank_aggregation": "median_of_pairwise_max",
    "profile_scope": "concurrent_rank_pair",
    "output_buffer": "preallocated_reused_native_out",
    "assignment": "earliest_finish_lpt_using_streaming_T_iso",
    "derate": "full_call_median / LPT_isolated_baseline_makespan"
  },
  "iso_formula": {
    "version": 3,
    "kind": "separable_route_usl_calibrated",
    "equation": "O(t) + C(R) * phi_usl(t) * k_phi(t)",
    "o0": 100000.0,
    "o1": 300000.0,
    "alpha": 0.001,
    "beta": 0.0002,
    "c_pts": [[1, 200000.0], [12, 900000.0], [2040, 165000000.0]],
    "phi_pts": [[1, 1.0], [2, 0.51], [4, 0.26], [8, 0.13]],
    "thread_domain": [1, 32]
  },
  "bounded_tail_repartition": {
    "kind": "exact_layout_full_call_anchor",
    "objective": "fused_moe_full_call_ns",
    "extension_sha256": "...",
    "root_core_begins": [0, 16, 32, 48, 64, 80],
    "tail_core_begins": [0, 48],
    "entries": [{
      "root_shape": [16, 16, 16, 16, 16, 16],
      "tail_width": 24,
      "route_slices": 1,
      "routes": 1536,
      "median_ns": 9796654,
      "p10_ns": 9683675,
      "p90_ns": 9909612,
      "num_iters": 101
    }]
  },
  "isolated": [],
  "entries": []
}
```

Required v2 identity fields are:

- hardware: profiled CPU sets, NUMA nodes, LLC bytes, cores per rank, and
  concurrent rank count;
- kernel identity: backend, SVE implementation, M-tail policy,
  `full_n_team_stripes`, N-split policy, source hash, and extension binary hash;
- distributed shape: TP/EP mode and degree, global/local expert counts, H, and
  sharded F;
- calibration scope: the full local expert count, activation, dtype, SVE N tile,
  NUMA nodes, and exact physical CPU sets;
- working set: actual full packed W13/W2 stage bytes per expert;
- measurement: whether ranks were synchronized and how per-rank samples were
  reduced to global wall time.

For concurrent ranks, every timed call starts behind a cross-rank barrier. The
merged sample is `max(rank_sample_i)` for each iteration, and summary statistics
are computed from those paired maxima. Taking the maximum of two independently
computed medians is not schema-v2 compliant.

The input, schedule tensors, and native BF16 `out` tensor are allocated once per
measurement point. Warmup calls first-touch the output, and all timed calls reuse
the same storage. Allocation and page-fault cost therefore remain outside
`T_iso` and contention derates; model-runtime allocation is an end-to-end term,
not expert-compute time. Profiles that omit `measurement.output_buffer` used the
legacy allocate-per-call path and are not comparable at large total route counts.

The active generator is `profile_contention_async_dual_rank.py`. The underlying
single-rank worker is `profile_contention_async.py`; its schema-v2 output records
one rank and can also be used for single-rank targets.

New SVE profiles use `sve_implementation=jit` and
`m_tail_policy=xbyak_exact_m`. The isolated and contention route grids include
every M from 1 through 12 so the exact generated tail is measured rather than
interpolated through the former M1/M2/M4/M8 buckets. Profiles without these
fields are interpreted as `asm/static_bucketed` for history compatibility.
Catalog queries for the production JIT path must specify both fields; a mixed
catalog intentionally rejects an implementation-unspecified ambiguous query.
Static and JIT profiles remain distinct calibration domains.

`ProfileCatalog` has no legacy weight-range, range-count, or byte-window variant
API. It rejects non-full-N records and two records with the same domain identity.
Optional Plan V2 `task_w13_window_tiles/task_w2_window_tiles` remain legal
because every value covers the same complete N domain.

The TP/EP evaluator's `--sve-implementation auto` lookup first requests one
compatible `jit/xbyak_exact_m` calibration. It falls back to
`asm/static_bucketed` only when no JIT calibration matches. It never composes
calibrations across implementations. Explicit `jit` or `asm` selection is
strict and fails when that implementation has no compatible calibration.

`git_commit` and `git_worktree_dirty` may be `null` on a deployment host without
repository metadata. `source_sha256` and `extension_sha256` remain mandatory
kernel identity fields in that case.

The isolated table uses eight consecutive experts, enough to stream beyond LLC
without multiplying the slow 1T/large-M points by every local expert. The
contention table uses all local experts. For mixed-width shapes it assigns
experts with the same earliest-finish LPT rule as `IntervalPlanner`, records
`lane_task_counts`, and computes:

```text
iso_baseline_makespan = max_lanes(lane_tasks * T_iso(M, lane_threads))
full_call_derate      = full_call_median / iso_baseline_makespan
```

Schema-v2 profiles carrying `iso_formula` use it by default:

```text
T_iso(R,t) = O(t) + C(R) * phi_usl(t) * k_phi(t)
O(t)       = o0 + o1/t
phi_usl(t) = (1 + alpha(t-1) + beta*t(t-1)) / t
```

`C(R)` is the one-dimensional single-thread route calibration curve. The
`phi_pts` values define the one-dimensional correction
`k_phi(t)=phi_measured(t)/phi_usl(t)`, which is interpolated only over threads
and never over routes. The generalized-USL formula is valid only in
`thread_domain`; it is not used to extrapolate to a larger machine.
Exact M1..M12 tails and the first two complete M12 panels retain their measured
`(tail, threads)` correction because they do not share the steady-state M12
thread scaling; larger M uses the formula and composes its exact remainder from
the measured tail cost. Historical `asm/static_bucketed` profiles retain their
M1/M2/M4/M8 bucket mapping. Profiles without a serialized
`iso_formula` remain table-backed for compatibility, but can fit it at load time
with `iso_mode="formula"`. Set `iso_mode="table"` or
`FUSED_CPP_COST_MODEL_ISO_MODE=table` to run the previous two-dimensional table
as a validation baseline.

The complete `full_call_*` curve is the authoritative calibration for a uniform
full-rank workload. `makespan_ns` remains a normalized diagnostic; it must not
be multiplied by an arbitrary number of waves. The stage-aware simulator emits
one full W13 phase and one full W2 phase per task. Team width determines each
phase's tile-aligned owner stripe.

`bounded_tail_repartition` is an optional, placement-aware full-call anchor for
the terminal repartitions supported by the production planner. Unlike a
contention `shape`, its `root_shape` preserves lane order; `root_core_begins`
and `tail_core_begins` bind the physical interval layout. `route_slices`
defaults to `1`. A value greater than one means each of the two terminal
experts is divided into that many contiguous M/route slices, placed in grouped
expert order; such candidates require an exact anchor and are never estimated
by the generic DAG model. The model consumes an entry only when all active
experts have exactly the recorded route count, the strict root shape, tail
width, route-slice count, and full-N kernel geometry match the measurement.
Route interpolation and extrapolation are intentionally forbidden. Unmatched
unsliced workloads continue through the analytical stage/DAG simulator;
unmatched sliced candidates are pruned.

The anchor is separate from `T_iso`: it corrects effects that a width-only
contention signature cannot identify, such as equal-width tail teams placed on
specific core intervals or two teams simultaneously scanning one expert's
packed weights. `p10_ns`, `p90_ns`, and `num_iters` provide the candidate
uncertainty; the source result and extension hash must identify the exact
binary and measurement run that produced it.

### Explainable `T_iso` roofline shadow report

`tiso_roofline.py` emits a separate schema-v1 diagnostic with kind
`explainable_tiso_roofline_shadow`. It does not modify the active contention
profile and is not yet consumed by `ContentionCostModel`.

Each observation records the fitted steady-M12 panel time plus:

- `required_tflops = panel_flops / panel_time`;
- `required_l3_gbs = panel_kernel_bytes / panel_time`;
- arithmetic intensity and linear bulk-fit residuals.

The two required rates are equivalent interpretations of one latency sample,
not independently measured ceilings. A future active profile must serialize
pure W13/W2 compute ceilings, L3 bandwidth, copy bandwidth, and fixed/tail
residuals separately before using the roofline prediction path.

### Layered GEMM cost shadow report

`gemm_ecm.py` emits a separate schema-v2 diagnostic with kind
`sve_bf16_gemm_ecm_shadow`. It is not consumed by `ContentionCostModel`.

`model_layers` serializes the three contracts used by the shadow path:

- `algorithm`: `LogicalGemmWork`, useful FLOPs, compulsory one-pass bytes, and
  the W13-to-W2 dependency;
- `implementation`: the `KernelDemand` mapper identity and formula;
- `machine_response`: the measured-profile identity and response formula.

The legacy `kernel` section remains for report compatibility and records the
actual SVE N tile, exact M12/M8/M4/M2 BFMMLA and A/B load counts,
logical/compute/packed/store tail rows, L1 traffic, and the current shared-cache
traffic convention. Stage observations expose the following
equivalent lower-bound views of one measured M12 panel slope:

- required executed TFLOP/s and BFMMLA instruction/s;
- required balanced L1 load GB/s;
- required shared-cache GB/s;
- held-out route error from a sparse panel-slope calibration.

When a matching identity profile is supplied, it also records incremental SiLU
time per panel/output and the identity-W13 to W2 ratio. These required rates are
not independent ceilings. Active rollout requires separately measured BFMMLA,
load/private/shared-cache, and epilogue service rates.

### Full-stage owner-cache working-set shadow

`working_set_model.py` emits `full_stage_working_set_band_validation`. Its independent
calibration input is the CSV from `bench_weight_scan.cpp`; it is not a
`contention_derate` profile and must not be loaded by `ContentionCostModel`.
The serialized model records per-core private-cache bytes/ways, reserved ways,
resident scan bandwidth saturation, the derived owner-cache budget, and the
predicted expert/byte band. `profile_summary` and `holdout_summary` contain
measured regret but are validation results, not active cost anchors. This path
uses the profile's full W13/W2 stage sizes and is currently gated to at least 16
physical M12/tail panels.

New `profile_moe_stage_breakdown.py` output serializes a top-level `kernel`
object with `backend_n_tile`, `parallel_axis`,
`stage_geometry=full_n_team_stripes`, and `w13_skip_silu`. Per-row
`w13_parallel_axis` and
`w2_parallel_axis` describe M/N team partitioning; they are independent of the
full-stage geometry.

TP2 and EP2 reproduction commands for the 64-core/two-NUMA target are:

```bash
PYTHONPATH=src .venv/bin/python \
  cpu_moe_schedule_optimization/cost_model/profile_contention_async_dual_rank.py \
  --output tmp/tp2_profile_v2.json --parallel-mode tp --parallel-degree 2 \
  --hidden-size 4096 --ffn-hidden-size 1024 \
  --global-experts 64 --local-experts 64 --measurement-experts 0 \
  --isolated-measurement-experts 8 \
  --sve-implementation jit --warmup 5 --runs 20

PYTHONPATH=src .venv/bin/python \
  cpu_moe_schedule_optimization/cost_model/profile_contention_async_dual_rank.py \
  --output tmp/ep2_profile_v2.json --parallel-mode ep --parallel-degree 2 \
  --hidden-size 4096 --ffn-hidden-size 2048 \
  --global-experts 64 --local-experts 32 --measurement-experts 0 \
  --isolated-measurement-experts 8 \
  --sve-implementation jit --warmup 5 --runs 20
```

Defaults bind rank 0 to CPUs 0-31/NUMA0 and rank 1 to CPUs 32-63/NUMA1.

Generate one canonical full-N file per calibration domain. Historical
range/window sweeps remain research artifacts only; they are not valid runtime
profiles. Runtime profiles and the planner carry no weight-range, range-count,
or byte-window control. Per-task tile windows are Plan V2 execution metadata and
must not be reconstructed from those retired fields.

## Legacy schema v1

This file defines the first offline schema for the expert execution cost table:

```text
T_expert(routes, threads) -> nanoseconds
```

The table is measured per machine and per expert kernel shape. The scheduler must
not assume linear thread scaling.

### JSON Shape

```json
{
  "schema_version": 1,
  "target": {
    "machine": "apple-m-series-or-linux-aarch64-host",
    "cpu": "string",
    "num_cores": 16,
    "os": "string"
  },
  "expert_shape": {
    "dtype": "bf16",
    "hidden_size": 7168,
    "intermediate_size": 2048,
    "kernel": "fused_cpp_moe_bf16_tiled"
  },
  "route_buckets": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
  "thread_buckets": [1, 2, 3, 4, 6, 8, 12, 16],
  "metric": "median_ns",
  "entries": [
    {
      "routes": 64,
      "threads": 4,
      "median_ns": 120000,
      "p10_ns": 115000,
      "p90_ns": 132000,
      "p99_ns": 150000,
      "stddev_ns": 8000,
      "num_iters": 200
    }
  ],
  "planner_costs": [
    {
      "kind": "GREEDY_MARGINAL_GAIN",
      "active_experts": 128,
      "num_cores": 16,
      "distribution_shape": "zipf_like",
      "median_ns": 35000,
      "num_iters": 200
    }
  ]
}
```

### Required Fields

- `schema_version`: integer schema version. Start with `1`.
- `target`: hardware and OS metadata for reproducibility.
- `expert_shape`: kernel and model shape metadata. A cost table is only valid for
  matching expert kernels and tensor shapes.
- `route_buckets`: route counts measured or bucketized.
- `thread_buckets`: thread counts measured.
- `metric`: default metric used by the scheduler, usually `median_ns`.
- `entries`: measured `T_expert(routes, threads)` points.

### Entry Rules

Each `entries[]` item must contain:

- `routes`: positive integer route count.
- `threads`: positive integer thread count.
- `median_ns`: median expert execution latency in nanoseconds.
- `num_iters`: number of benchmark iterations.

Recommended optional latency fields:

- `p10_ns`
- `p90_ns`
- `p99_ns`
- `stddev_ns`

### Lookup Rules

The first simulator supports two lookup modes:

1. Exact match when `(routes, threads)` exists.
2. Nearest measured bucket fallback when exact match is absent.

Future runtime integration should prefer interpolation or explicit route
bucketization, but that belongs in the cost model implementation rather than in
planner logic.

### Planning Cost

`planner_costs` is optional in the first schema. It records measured planner
overhead:

```text
T_plan(kind, active_experts, num_cores, distribution_shape) -> nanoseconds
```

When absent, the offline simulator uses a complexity-based planner cost model
for scoring and still records local Python planner wall time as diagnostics. The
complexity coefficients should be calibrated with native planner profiling before
runtime integration.

### Lightweight Native Planner Cost Table

> **⚠ DEPRECATED（阶段 1 已删除相关工具）**：本节描述的是 wave 版 planner 开销表，其生成工具
> `build_lightweight_planner_cost.py` 与消费者 `synthetic_sweep.py` / `offline_simulator.py` 均已删除。
> 现行的 plan 开销测量见 [`../planners/bench_planner_overhead.py`](../planners/bench_planner_overhead.py)。以下仅存历史参考。

For runtime-style scoring, prefer a compact lookup table instead of the
complexity formula:

```text
T_plan(kind, active_experts, cores) -> nanoseconds
```

The table is generated from native C++ planner timings and scored with nearest
bucket lookup:

1. choose the nearest `cores` bucket;
2. choose the nearest `active_experts` bucket within that core bucket;
3. return the measured cost for `kind`.

Example:

```json
{
  "schema_version": 1,
  "kind": "native_planner_cost_table",
  "source_profile": "tmp/moe_schedule_bench/native_planner_cost_aws_8c_20260701.json",
  "metric": "total_native_median_ns",
  "lookup": "nearest_core_then_nearest_active",
  "core_buckets": [8],
  "active_buckets": [1, 2, 4, 6, 8, 16, 32, 64, 128, 192, 256],
  "planners": [
    "FIXED_GLOBAL_THREADS",
    "SORTED_TOKEN_BALANCED_1T",
    "UNIFORM_WAVES",
    "ENUMERATE_CORE_GROUPS",
    "GREEDY_MARGINAL_GAIN"
  ],
  "table": {
    "8": {
      "FIXED_GLOBAL_THREADS": {
        "1": 970,
        "256": 50370
      },
      "GREEDY_MARGINAL_GAIN": {
        "1": 2000,
        "256": 94770
      }
    }
  }
}
```

Historical, non-runnable invocation (the consumer has been deleted):

```bash
python -B cpu_moe_schedule_optimization/benchmarks/synthetic_sweep.py \
  --plan-cost-source native_table \
  --planner-cost-profile cpu_moe_schedule_optimization/cost_model/profiles/planner_native_table.json
```

The older complexity model remains available for diagnostics, but should not be
treated as the runtime planner overhead model.
