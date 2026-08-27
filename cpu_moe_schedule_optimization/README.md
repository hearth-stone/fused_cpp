# CPU MoE Schedule Optimization

This directory contains the cost models, planners, calibration tools, workload
catalog, offline oracles, and validation utilities for CPU MoE scheduling.

The active system is an async interval-DAG lowered to Plan V2. The old
barrier-after-wave simulator, AUTO selector, and wave planner family are
retired; see [DEPRECATED_WAVE.md](DEPRECATED_WAVE.md). Do not use
[DESIGN.md](DESIGN.md) or [FINDINGS.md](FINDINGS.md) as current algorithm or
performance specifications.

The paper-facing contribution scope, reusable evidence, prohibited claims, and
remaining gates are maintained in
[docs/moe_paper_readiness.md](../docs/moe_paper_readiness.md).

## Current System

Given the active expert route counts and one calibrated CPU rank, the system
selects an executable plan for the fused expert runtime:

```text
topk_ids
  -> route histogram
  -> calibrated T_iso / phase-resource model
  -> quick or full planner
  -> Plan V2 core-interval DAG
  -> fused_moe_bf16_tiled_async_plan
```

The planner-visible expert job is the complete
`gather-pack -> W13 -> SwiGLU/packC -> W2` pipeline. Different experts have no
data dependency, but their execution rates depend on the simultaneously active
compute, cache, and DRAM demand.

For one scheduling domain:

- `C` is the ordered set of logical workers mapped to physical CPU IDs;
- `M_i` is the route count of active expert `i`;
- `T_iso(M_i,t)` is its isolated time at team width `t`;
- the contention model advances active task phases to completion events;
- the objective is expert-compute makespan, with planning latency reported
  separately and included in end-to-end comparisons.

Router, communication, and final TopK merge are not yet complete physical
demand vectors in the active analytical model. Do not describe its prediction
as full layer time.

## Source Of Truth

When documents disagree, use this order:

1. current implementation and tests;
2. [MATHEMATICAL_MODEL.md](MATHEMATICAL_MODEL.md) for the problem, model, and
   pruning semantics;
3. [planners/plan_schema.md](planners/plan_schema.md) and
   [cost_model/profile_schema.md](cost_model/profile_schema.md) for serialized
   contracts;
4. [the SVE manifest](../optimizations/fused_moe_sve/manifest.yaml) for feature
   lifecycle;
5. dated result reports for measurements on their recorded binary/profile;
6. archived design and weekly documents.

## Execution Geometry

Every production W13 and W2 task covers one complete packed-N domain. Legacy
weight split flags, range counts, and byte-window ABI fields have been removed.

Plan V2 may carry a per-worker window in whole backend N tiles for each stage.
For stage `(K_s,N_s)`, tile width `nu`, team width `t`, and stored window
`w_s`:

```text
q_s                = N_s / nu
full_stripe_tiles  = ceil(q_s / t)
resolved_window    = w_s > 0 ? w_s : full_stripe_tiles
tiles_per_window   = t * resolved_window
num_windows        = ceil(q_s / tiles_per_window)
owner_window_bytes = resolved_window * K_s * nu * 2
```

The window changes only the serial visitation order of the same N tiles. It
does not change weight coverage, arithmetic, packed layout, or output
ownership. The deterministic stage-window policy runs after width selection;
window size is not a free planner search dimension.

## Cost-Model Backends

### Empirical schema v2

`cost_model/phase_model.py::ContentionCostModel` consumes one exact
calibration-domain profile. Profiles are bound to model shape, TP/EP mode and
degree, expert ownership, kernel/backend identity, backend N tile, CPU/NUMA/LLC
placement, page/execution policy, and source/binary identity.

The empirical backend provides:

- measured isolated curves;
- exact full-call anchors where available;
- stage-aware W13/W2 execution;
- active working-set and contention response;
- uncertainty intervals;
- rank-lifetime switching when compatible single/dual-rank companion profiles
  exist.

Cross-profile interpolation remains disabled. Missing or incompatible profiles
fail closed instead of silently borrowing another machine, F dimension,
parallel degree, or topology.

### Analytical backend

`cost_model/analytic_model.py::AnalyticMoeCostModel` separates:

1. logical GEMM work;
2. exact SVE kernel demand;
3. machine compute, L2, LLC, DRAM, and epilogue services;
4. setup, cold packed-B, and steady packed-B phases;
5. event-time shared-resource contention.

It does not load a route/thread latency table or measured contention-shape
table. Machine calibration is explicit and never runs at import or on the first
request.

The current analytical evidence is mixed:

- 9.18% isolated MAPE on 108 true holdout points passes the 10% gate;
- 49.57% contention P90 error fails the 15% gate;
- 8.17% maximum measured shape regret fails the 5% gate;
- the narrower tile-window selector reaches 1.10/3.45/4.40% median/P90/max
  regret on its six declared 192-core transition points.

The analytical runtime is therefore an explicit deployment capability, not a
general replacement for the empirical holdout oracle.

## Planner Modes

### Quick

`IntervalPlanner.plan_quick()` and `PlannedMoE(search_mode="quick")` evaluate
homogeneous team shapes only. Each distinct `(routes,width)` obtains one
`T_iso`; deterministic LPT assigns experts to equal-width lanes. Native C++
assignment and candidate selection are used when available. Quick emits strict
Plan V2 and does not search phase-DAG contention, temporal order, tail pools, or
bounded tail repartition.

`plan_quick_fixed()` evaluates one caller-selected homogeneous width and is the
supported deployment fallback. `plan_quick_with_shared()` is a separate bounded
mixed-width extension for one synthetic all-token shared expert.

### Full

`IntervalPlanner.plan()` and `PlannedMoE(search_mode="full")` evaluate the full
modeled candidate family:

- strict mixed-width core shapes;
- deterministic temporal lane orders;
- eligible whole-expert tail pools;
- bounded terminal repartition/route-slice candidates;
- the quick winner rescored as an explicit strict baseline for analytical full
  search.

Analytical full selects minimum expected modeled makespan. Its systematic model
error is not reduced by the number of experts or calibration runs.

Full is currently an offline performance-oracle path. The latest 80-core DSV4
run evaluated 142 strict and 327 dynamic candidates and took about 42 seconds
cold. Its selected plan measured 34.418 ms versus quick at 37.141 ms; the best
measured top-six candidate was 33.907 ms, for 1.51% selected shortlist regret.

### Deployment runtime

`src/fused_cpp/moe/planner_runtime.py::MoePlannerRuntime` uses quick search,
native C++ selection when available, and strict Plan V2. It:

- binds one analytical calibration to an ordered CPU rank;
- preserves the existing dispatcher for incompatible calls;
- disables route-plan caching by default;
- caches identity-bound `T_iso(M,t)` scalars in memory and optionally on disk;
- supports `initialize_planner(max_routes)` to precompute the dense cost grid;
- supports `FUSED_CPP_MOE_PLANNER_FIXED_THREADS` for fixed-width LPT.

The current quick path is not uniformly better than fixed 8T. In the latest
80-core nine-case operator check it improved long/short bimodal by 22.13%, but
regressed active-set 8/16 by 15.54%/11.31% after selecting an over-wide 40T
team. This is an open paper-critical gate, not a hidden limitation.

## Plan V2

The executable bridge is documented in [planners/plan_schema.md](planners/plan_schema.md).
Its active semantics include:

- explicit ordered `thread_cpu_ids`;
- fixed core intervals and CSR task dependencies;
- singleton allowed widths for current strict/tail-pool execution;
- strict or whole-expert tail-pool placement;
- optional bounded route slices for strict plans;
- optional W13/W2 per-worker tile windows;
- plan-level tri-state `early_merge`.

No running task changes width. The retired W2-resize, timed-release, weight
range, and byte-window fields are rejected.

Production planning materializes `early_merge=true` after selecting the compute
plan. That is an operational default validated on the declared m5 TP2 DSV4
capture, not a cost-model-derived cross-workload optimum; historical TP4
long/short evidence contains a known regression when early merge stays on.

## Workload Catalog

`planners/workload_catalog.py` provides deterministic controlled histograms for
`tokens=2048`, `top_k=6`, `experts=256`, and 12,288 total routes:

| Workload | Histogram |
| --- | --- |
| `moe256-uniform` | `256 x 48` |
| `moe256-active-set-{8,16,32,64,128}` | fixed total routes across the named active set |
| `moe256-tiered-hotspot` | `4 x 768 + 12 x 384 + 48 x 96` |
| `moe256-long-short-bimodal` | `5 x 2040 + 174 x 12` |
| `dsv4-real-2048-seq70` | captured top-16 counts plus a deterministic moment-matched synthetic tail |

The DSV4 artifact is a reconstructed routing summary, not a complete original
`topk_ids` trace. Paper evaluation must add complete multi-layer traces and
decode-oriented inputs.

## Current Commands

### Offline comparison

Use a profile compatible with the requested core count and geometry:

```bash
python cpu_moe_schedule_optimization/planners/simulate_schedules.py \
  <compatible-schema-v2-profile.json> \
  --preset dsv4-real-2048-seq70 \
  --cores <rank-cores> \
  --shapes
```

This runs only the model/simulator. It does not execute the native kernel and
cannot establish measured regret.

### Analytical holdout

```bash
python cpu_moe_schedule_optimization/cost_model/validate_analytic_model.py \
  <machine-calibration.json> \
  <compatible-empirical-profile.json> \
  --output <holdout-report.json>
```

Report evaluated and skipped coverage, isolated and contention error, and
measured shape regret. Do not use calibration-fit error as holdout accuracy.

### Real Plan V2 comparison

On a target SVE machine, with a profile matching the current source/binary and
measurement geometry:

```bash
PYTHONPATH=src numactl --cpunodebind=<node> --membind=<node> \
  taskset -c <cpu-list> .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_vllm_staged_schedule.py \
  --preset dsv4-real-2048-seq70 \
  --production-profile <compatible-schema-v2-profile.json> \
  --route-dtype fp32 \
  --warmup 5 \
  --runs 21
```

Use FP32 route storage for the primary numerical path unless BF16 route-storage
model quality has been validated separately.

## Focused Validation

Planner/model/schema changes normally start with:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_moe_analytic_model.py \
  tests/test_moe_cost_model_v2.py \
  tests/test_moe_native_interval_planner.py \
  tests/test_moe_plan_v2.py \
  tests/test_moe_stage_window_plan.py
```

Kernel execution must also run the relevant SVE tests on the target ISA. A
local non-SVE build is not correctness evidence for the native path.

## Document Map

- Paper thesis and evidence: [docs/moe_paper_readiness.md](../docs/moe_paper_readiness.md)
- Mathematical source of truth: [MATHEMATICAL_MODEL.md](MATHEMATICAL_MODEL.md)
- Active implementation checklist: [TODO.md](TODO.md)
- Analytical model: [cost_model/ANALYTIC_MODEL.md](cost_model/ANALYTIC_MODEL.md)
- Profile schema: [cost_model/profile_schema.md](cost_model/profile_schema.md)
- Plan V2 schema: [planners/plan_schema.md](planners/plan_schema.md)
- Offline isolated oracle: [planners/ISOLATED_CP_SAT_ORACLE.md](planners/ISOLATED_CP_SAT_ORACLE.md)
- SVE implementation and experiments: [optimizations/fused_moe_sve/README.md](../optimizations/fused_moe_sve/README.md)
- SVE feature lifecycle: [optimizations/fused_moe_sve/manifest.yaml](../optimizations/fused_moe_sve/manifest.yaml)
- vLLM integration: [docs/vllm_bf16_tiled_moe_integration.md](../docs/vllm_bf16_tiled_moe_integration.md)
- Retired wave design: [DEPRECATED_WAVE.md](DEPRECATED_WAVE.md)
- Archived wave-era design: [DESIGN.md](DESIGN.md)
- Archived wave-era results: [FINDINGS.md](FINDINGS.md)

## Evidence Discipline

- A profile is valid only for its declared calibration domain and binary.
- A dated result is not silently relabelled as current after kernel, planner, or
  geometry changes.
- Correctness precedes performance measurement.
- Performance claims name the command, machine, CPU/NUMA placement, page
  policy, model shape, route distribution, baseline, sample count, statistic,
  absolute values, and relative result.
- Planner claims include planning latency and measured regret against a declared
  oracle or measured candidate set.
- Unsupported regions, skipped validation, and negative results remain visible.
