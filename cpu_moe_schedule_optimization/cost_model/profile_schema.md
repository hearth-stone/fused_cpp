# Expert Cost Profile Schema

This file defines the first offline schema for the expert execution cost table:

```text
T_expert(routes, threads) -> nanoseconds
```

The table is measured per machine and per expert kernel shape. The scheduler must
not assume linear thread scaling.

## JSON Shape

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

## Required Fields

- `schema_version`: integer schema version. Start with `1`.
- `target`: hardware and OS metadata for reproducibility.
- `expert_shape`: kernel and model shape metadata. A cost table is only valid for
  matching expert kernels and tensor shapes.
- `route_buckets`: route counts measured or bucketized.
- `thread_buckets`: thread counts measured.
- `metric`: default metric used by the scheduler, usually `median_ns`.
- `entries`: measured `T_expert(routes, threads)` points.

## Entry Rules

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

## Lookup Rules

The first simulator supports two lookup modes:

1. Exact match when `(routes, threads)` exists.
2. Nearest measured bucket fallback when exact match is absent.

Future runtime integration should prefer interpolation or explicit route
bucketization, but that belongs in the cost model implementation rather than in
planner logic.

## Planning Cost

`planner_costs` is optional in the first schema. It records measured planner
overhead:

```text
T_plan(kind, active_experts, num_cores, distribution_shape) -> nanoseconds
```

When absent, the offline simulator uses a complexity-based planner cost model
for scoring and still records local Python planner wall time as diagnostics. The
complexity coefficients should be calibrated with native planner profiling before
runtime integration.
