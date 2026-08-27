# CPU MoE Cost-Model Benchmarks

This directory contains focused kernel and stage probes used by the CPU MoE
cost-model work. The old wave-planner `synthetic_sweep.py`,
`selector_stress.py`, and `scheduled_bridge_bench.py` tools were removed in
2026-07; their commands are not supported.

Use the current tools according to the question being measured:

| Tool | Purpose | Runtime scope |
| --- | --- | --- |
| `bench_fused_silu.py` | Fused W13 SwiGLU/pack-C versus separate activation | Complete synchronous BF16 MoE call |
| `bench_fused_silu_nsplit.py` | Historical hierarchical N-split sensitivity for fused activation | Complete synchronous BF16 MoE call |
| `bench_split_mn.py` | M-split versus N-split for isolated W13/W2 GEMM shapes | Native `team_gemm` only |
| `bench_weight_scan.cpp` | Packed-B owner-scan/cache service without GEMM | Standalone native probe |
| `profile_moe_stage_breakdown.py` | Scheduled operator stage decomposition | Scheduled BF16 MoE call |
| `profile_single_gemm_trace.py` | Per-stage W13/W2 critical-worker latency | Scheduled BF16 MoE call |
| `profile_w2_gemm_split.py` | Forced M/N split through the legacy BF16 linear wrapper | Isolated BF16 GEMM |

Planner comparisons live under `../planners/`; current SVE Plan V2 and E2E
benchmarks live under `../../optimizations/fused_moe_sve/benchmarks/`.
Paper-facing scope and evidence requirements are in
[`../../docs/moe_paper_readiness.md`](../../docs/moe_paper_readiness.md).

## Measurement Rules

- Build the current extension before target measurements.
- Pin the process and record the exact CPU list and NUMA memory node.
- Record page policy, backend, SVE vector length, model shape, route shape,
  thread count, warmups, runs, and statistic.
- Validate output before reporting candidate performance.
- Do not compare results across source/binary identities as if they were one
  A/B run.
- Keep raw traces and large dumps outside the source tree; retain concise result
  reports under `optimizations/fused_moe_sve/results/`.

## Fused Activation

Compare the synchronous unfused-activation and fused-W13 paths:

```bash
PYTHONPATH=src taskset -c 0-7 .venv/bin/python \
  cpu_moe_schedule_optimization/benchmarks/bench_fused_silu.py \
  --hidden-size 4096 \
  --ffn-hidden-size 512 \
  --num-experts 8 \
  --top-k 2 \
  --tokens 8,64,256,1024 \
  --threads 1,4,8 \
  --warmup 5 \
  --runs 20
```

This benchmark includes routing and the complete synchronous operator. It is
not interchangeable with the explicit same-GEMM unfused-pipeline comparator in
`optimizations/fused_moe_sve/features/unfused_pipeline.cpp`.

`bench_fused_silu_nsplit.py` retains a historical environment-driven N-split
comparison. It is useful for mechanism diagnosis, not for current Plan V2
planner claims.

## Isolated M/N GEMM Split

`bench_split_mn.py` measures only the native middle-layer `team_gemm`; packing,
routing, activation, scatter, and merge are outside the timed region:

```bash
PYTHONPATH=src taskset -c 0-7 .venv/bin/python \
  cpu_moe_schedule_optimization/benchmarks/bench_split_mn.py \
  --hidden-size 4096 \
  --ffn-hidden-size 512 \
  --m-values 1,2,4,8,16,32,48,64,128,256,512,1024,2048 \
  --threads 1,2,4,8 \
  --warmup 5 \
  --runs 40 \
  --output-json /tmp/moe_split_mn.json \
  --output-csv /tmp/moe_split_mn.csv
```

Use this tool only to answer isolated GEMM mapping questions. A GEMM winner is
not automatically an expert-pipeline or planner winner.

## Scheduled Stage Breakdown

Trace one active expert through the scheduled operator:

```bash
PYTHONPATH=src taskset -c 0-7 .venv/bin/python \
  cpu_moe_schedule_optimization/benchmarks/profile_moe_stage_breakdown.py \
  --routes 16,64,256,1024,2048 \
  --threads 1,2,4,8 \
  --warmup 3 \
  --runs 5 \
  --output-json /tmp/moe_stage_breakdown.json \
  --output-csv /tmp/moe_stage_breakdown.csv
```

The report separates plan materialization, route build, validation, scratch,
gather, W13, activation, W2, scatter, merge, output cast, and residual compute
gap. It is a stage-attribution tool, not a full planner benchmark.

Use `profile_single_gemm_trace.py` when only the critical-worker W13/W2 trace is
needed:

```bash
PYTHONPATH=src taskset -c 0-7 .venv/bin/python \
  cpu_moe_schedule_optimization/benchmarks/profile_single_gemm_trace.py \
  --routes 16,64,256,1024,2048 \
  --threads 1,2,4,8 \
  --warmup 3 \
  --runs 5 \
  --trace-file /tmp/moe_single_gemm_trace.log
```

## Packed-B Owner-Scan Service

`bench_weight_scan.cpp` measures the N-owner packed-B scan pattern without GEMM
instructions. Each team owns one expert stream and each worker scans a disjoint
column slice. Repeated passes model successive M panels:

```bash
g++ -std=c++17 -O2 -pthread \
  cpu_moe_schedule_optimization/benchmarks/bench_weight_scan.cpp \
  -o /tmp/bench_weight_scan

taskset -c 0-95 /tmp/bench_weight_scan \
  --cpu-ids 0-95 \
  --groups 1,2,3,4,5,6,7,8,9,10,11,12,16,24,32 \
  --stream-mib 16 \
  --passes 170 \
  --warmup 2 \
  --runs 9 \
  --output-csv /tmp/weight_scan.csv
```

This probe is diagnostic input for cache/service modeling. It does not replace
fused-kernel holdout validation.

## Legacy BF16 Linear Split Probe

`profile_w2_gemm_split.py` forces `BF16_NEON_SPLIT=m|n|auto` through the older
`bf16_linear` wrapper. It excludes MoE routing and scheduling and is retained
for historical M/N mapping investigations:

```bash
PYTHONPATH=src taskset -c 0-7 .venv/bin/python \
  cpu_moe_schedule_optimization/benchmarks/profile_w2_gemm_split.py \
  --stage w2 \
  --m-values 16,32,64,128,256,512,1024,2048 \
  --threads 1,2,4,8 \
  --splits m,n,auto \
  --warmup 3 \
  --runs 7 \
  --output-json /tmp/w2_gemm_split.json \
  --output-csv /tmp/w2_gemm_split.csv
```

Do not use this legacy linear result as evidence for the current SVE exact-M
fused expert without a matching current-runtime validation.

## Current Planner And E2E Tools

- Offline model comparison:
  `../planners/simulate_schedules.py`
- Planner overhead:
  `../planners/bench_planner_overhead.py`
- Empirical/native cold planner:
  `../planners/bench_native_cold_planner.py`
- Planned E2E scheduling:
  `../planners/bench_e2e_scheduling.py`
- Current SVE Plan V2 workload comparison:
  `../../optimizations/fused_moe_sve/benchmarks/bench_vllm_staged_schedule.py`
- Timeline capture:
  `../../optimizations/fused_moe_sve/benchmarks/capture_schedule_timeline.py`
