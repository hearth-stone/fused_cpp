> **⚠ DEPRECATED — 本文档中的 wave 调度内容后续不考虑，仅作历史参考。** 见 [../DEPRECATED_WAVE.md](../DEPRECATED_WAVE.md)。async interval-DAG + cost model 保留并继续。
>
> **阶段 1 已删除脚本**：下方 *Synthetic Sweep* / *Auto Selector* / *Selector Stress* / *Scheduled C++ Bridge Benchmark* 四节所述脚本
> （`synthetic_sweep.py`、`selector_stress.py`、`scheduled_bridge_bench.py`）**已删除**，仅存历史说明。
> 离线对比多种调度算法请改用 [`../planners/simulate_schedules.py`](../planners/simulate_schedules.py)。

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
- speedup versus `SORTED_TOKEN_BALANCED_1T`;
- auto selector plan and regret versus the best planning-aware plan;
- best execution-only plan.

The sweep currently includes:

- `FIXED_GLOBAL_THREADS`
- `SORTED_TOKEN_BALANCED_1T`
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
thread_cpu_ids
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

Structured benchmark results can be written at the same time:

```bash
PYTHONPATH=src .venv/bin/python \
  cpu_moe_schedule_optimization/benchmarks/scheduled_bridge_bench.py \
  --distribution hotspot \
  --hot-experts 4 \
  --hot-fraction 0.75 \
  --tokens 2048 \
  --top-k 6 \
  --cores 8 \
  --planner all \
  --cost-table cpu_moe_schedule_optimization/cost_model/profiles/aws_dsv4_8c_packa_sha46228bb_20260703.json \
  --output-json tmp/moe_schedule_bench/hotspot_4x75.json \
  --output-csv tmp/moe_schedule_bench/hotspot_4x75.csv
```

The JSON includes the full workload histogram, planner predictions, measured
timings, plan-shape features, metadata, and the full scheduled bridge plan. The
CSV keeps one row per planner with the fields needed for cost-model calibration.

## MoE Stage Breakdown

To inspect the scheduled MoE compute flow, enable the C++ trace collector through
`profile_moe_stage_breakdown.py`:

```bash
PYTHONPATH=src .venv/bin/python \
  cpu_moe_schedule_optimization/benchmarks/profile_moe_stage_breakdown.py \
  --routes 2048 \
  --threads 1,2,4,8 \
  --runs 5 \
  --output-json tmp/moe_schedule_bench/stage_breakdown_2048.json \
  --output-csv tmp/moe_schedule_bench/stage_breakdown_2048.csv
```

The script reports C++ traced e2e time and percentage columns for:

```text
route_build scratch_alloc gather_input w13 activation w2 scatter_route_out
compute_gap merge_routes_total output_cast other
```

`compute_gap` is the part of scheduled compute not explained by gather/GEMM/
activation/scatter. It is useful for spotting synchronization, dispatch, and
unmeasured per-wave overhead.

## Exact-Range Owner-Cache Bandwidth

`bench_weight_scan.cpp` measures the N-split packed-B ownership pattern without
GEMM instructions. Each team owns one expert stream and each worker scans a
disjoint column slice. Repeated passes model successive M12 panels:

```bash
g++ -std=c++17 -O2 -pthread \
  cpu_moe_schedule_optimization/benchmarks/bench_weight_scan.cpp \
  -o /tmp/bench_weight_scan

taskset -c 0-95 /tmp/bench_weight_scan \
  --cpu-ids 0-95 \
  --groups 1,2,3,4,5,6,7,8,9,10,11,12,16,24,32 \
  --stream-mib 16 --passes 170 --warmup 2 --runs 9 \
  --output-csv /tmp/weight_scan.csv
```

Feed the CSV and an exact-range schema-v2 profile to
`cost_model/working_set_model.py`. The model fits only resident scan bandwidth;
the fused profile remains held-out validation. See
`cost_model/WORKING_SET_MODEL_VALIDATION.md` for the formula and V3 results.

## Isolated MoE GEMM M/N Split

Use `profile_w2_gemm_split.py` to isolate either MoE GEMM shape:

```text
w13: A[M, H] x W13[H, 2F] -> C[M, 2F]
w2:  A[M, F] x W2[F, H]  -> C[M, H]
```

For the second GEMM:

```text
A[M, F] x W2[F, H] -> C[M, H]
```

The script uses the existing `bf16_linear` wrapper around refs/i8gemm and forces
the underlying BF16 GEMM split through `BF16_NEON_SPLIT=m|n`. This removes MoE
routing, activation, scatter, and scheduler overhead from the measurement:

```bash
PYTHONPATH=src .venv/bin/python \
  cpu_moe_schedule_optimization/benchmarks/profile_w2_gemm_split.py \
  --stage w2 \
  --m-values 16,32,64,128,256,512,1024,1536,2048,3072,4096,6144,8192 \
  --threads 1,2,4,8 \
  --splits m,n \
  --runs 7 \
  --output-json tmp/moe_schedule_bench/w2_gemm_split.json \
  --output-csv tmp/moe_schedule_bench/w2_gemm_split.csv
```

Switch `--stage w13` and output paths to measure the first GEMM:

```bash
PYTHONPATH=src .venv/bin/python \
  cpu_moe_schedule_optimization/benchmarks/profile_w2_gemm_split.py \
  --stage w13 \
  --m-values 16,32,64,128,256,512,1024,1536,2048,3072,4096,6144,8192 \
  --threads 1,2,4,8 \
  --splits m,n \
  --runs 7 \
  --output-json tmp/moe_schedule_bench/w13_gemm_split.json \
  --output-csv tmp/moe_schedule_bench/w13_gemm_split.csv
```

By default the script sets `BF16_NEON_CLAMP_THREADS=0`, so each requested thread
count is tested directly. Add `--keep-thread-clamp` to keep the upstream
refs/i8gemm thread-clamp behavior.

AWS 8 核 profile 已保存为（unpinned）：

```text
cpu_moe_schedule_optimization/cost_model/profiles/aws_dsv4_8c_packa_sha46228bb_20260703.json
```

The sweep uses the complexity-based planner cost model by default. For fixed
planner-cost sensitivity tests, switch to `--plan-cost-source model` and override
costs in the same style as `offline_simulator.py`:

```bash
python -B cpu_moe_schedule_optimization/benchmarks/synthetic_sweep.py \
  --cost-table cpu_moe_schedule_optimization/cost_model/profiles/aws_dsv4_8c_packa_sha46228bb_20260703.json \
  --plan-cost-source model \
  --planner-cost fixed=1us \
  --planner-cost balanced=2us \
  --planner-cost uniform=5us \
  --planner-cost greedy=20us \
  --planner-cost groups=30us
```
