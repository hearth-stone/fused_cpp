# Synthetic Benchmarks

This directory contains offline benchmark helpers for synthetic CPU MoE routing
histograms.

The goal is to compare planner behavior across controlled workload shapes before
real router dumps are available.

## Synthetic Sweep

Run the smoke suite:

```bash
python -B cpu_moe_schedule_optimization/benchmarks/synthetic_sweep.py
```

Run the larger synthetic suite:

```bash
python -B cpu_moe_schedule_optimization/benchmarks/synthetic_sweep.py \
  --case-set full \
  --cores 16
```

The summary reports:

- token count used by the case;
- CPU core count used by the case;
- active expert count;
- Gini coefficient;
- max load violation versus all-expert mean;
- best planning-aware plan;
- speedup versus `FIXED_GLOBAL_THREADS`;
- auto selector plan and regret versus the best planning-aware plan;
- best execution-only plan.

The sweep currently includes:

- `FIXED_GLOBAL_THREADS`
- `UNIFORM_WAVES`
- `GREEDY_MARGINAL_GAIN`
- `ENUMERATE_CORE_GROUPS`

The smoke suite includes two DSV4-like synthetic cases derived from the
`20260629-141213-analysis` routing summary:

- `dsv4_sparse_topk`: `tokens=2100`, `top_k=6`, about 6 active experts with
  roughly equal route counts.
- `dsv4_broad_heavytail`: `tokens=2048`, `top_k=6`, broad heavy-tail routing
  close to the observed `active≈204`, `top16≈0.56` shape.

The default sweep core count is 16. The smoke suite also includes explicit
8-core variants:

- `dsv4_sparse_topk_8c`
- `dsv4_broad_heavytail_8c`

## Auto Selector

The sweep compares the oracle best plan from all enabled planners against the
`AUTO` selector. `AUTO` is a cheap heuristic gate:

```text
routes stats -> planner subset -> score candidates -> select best
```

It is intended to avoid running expensive planners on balanced workloads while
keeping regret close to 1.0 on skewed workloads.

## Selector Stress

Run a parameter-grid stress test for the `AUTO` selector:

```bash
python -B cpu_moe_schedule_optimization/benchmarks/selector_stress.py
```

Quick smoke test:

```bash
python -B cpu_moe_schedule_optimization/benchmarks/selector_stress.py \
  --case-set quick
```

The stress test reports the worst `auto_total / oracle_total` cases across
multiple distributions and core counts. Use it to tune `select_auto_planners()`
thresholds.

## Scheduled C++ Bridge Benchmark

After building the C++ extension, run a real kernel timing from simulator plans:

```bash
PYTHONPATH=src .venv/bin/python \
  cpu_moe_schedule_optimization/benchmarks/scheduled_bridge_bench.py \
  --distribution active_subset \
  --active-experts 6 \
  --tokens 2100 \
  --top-k 6 \
  --cores 8 \
  --planner groups
```

Compare all planner-generated schedules plus the existing non-scheduled kernel:

```bash
PYTHONPATH=src .venv/bin/python \
  cpu_moe_schedule_optimization/benchmarks/scheduled_bridge_bench.py \
  --distribution lognormal \
  --lognormal-sigma 2.0 \
  --cores 8 \
  --planner all \
  --include-default
```

The benchmark converts each `Plan` to:

```text
wave_offsets
team_expert_ids
team_threads
```

and passes those tensors to `fused_moe_bf16_tiled_scheduled`.

To use a calibrated expert cost table instead of the synthetic model, first
generate one with `cost_model/profile_expert_cost.py`, then pass it here:

```bash
PYTHONPATH=src .venv/bin/python \
  cpu_moe_schedule_optimization/benchmarks/scheduled_bridge_bench.py \
  --distribution active_subset \
  --active-experts 6 \
  --tokens 2100 \
  --top-k 6 \
  --cores 8 \
  --planner groups \
  --cost-table cpu_moe_schedule_optimization/cost_model/profiles/local_dsv4_8c.json
```

The sweep uses the complexity-based planner cost model by default. For fixed
planner-cost sensitivity tests, switch to `--plan-cost-source model` and override
costs in the same style as `offline_simulator.py`:

```bash
python -B cpu_moe_schedule_optimization/benchmarks/synthetic_sweep.py \
  --plan-cost-source model \
  --planner-cost fixed=1us \
  --planner-cost uniform=5us \
  --planner-cost greedy=20us \
  --planner-cost groups=30us
```
