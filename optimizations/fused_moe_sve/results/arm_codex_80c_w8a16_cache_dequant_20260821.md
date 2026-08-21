# W8A16 cache dequantization on Arm-codex 80C

## Scope

This run checks whether the existing per-channel W8A16 cache-dequantized MoE
path beats packed BF16 on the captured DeepSeek V4 Flash routing shape. It does
not change production dispatch.

## Configuration

- Source: Git `42e12d8`, clean local worktree before the result record.
- Host: `Arm-codex-internal`, NUMA3 CPUs `240-319`, local memory binding.
- Pages: ordinary pages; this host has no configured HugeTLB pool.
- Build: MoE-only, SVE256, effective kernel flags `-O2 -std=c++17`.
- Shape: TP4, E256, T2048, TopK6, H4096, F512, clamped SwiGLU limit 10.
- Routing: captured `dsv4-real-2048-seq70`, 223 active experts, M1-M918.
- Output: FP32 direct route store followed by the existing weighted merge.
- Calibration: explicit quick calibration on the same 80 CPUs; 7.89 seconds.
- Plan: strict `10 x 8T`, LPT assignment, early merge enabled.
- Timing: BF16 and W8A16 alternate in one process; three independent processes,
  each with 7 warmups and 31 measured samples.

The quick calibration measured 7.360 TFLOP/s aggregate BFMMLA, 351.79 GB/s
DRAM service and 1.208 TB/s aggregate LLC service at 80 threads.

## Correctness

The focused W8A16 suite passed three tests on the target build. Across all
window measurements, W8A16 versus BF16 output had maximum absolute difference
`0.00012970` and relative L2 error `0.01502405`. Packed W8A16 weights, including
scales, consumed 1,631,584,256 bytes versus 3,221,225,472 BF16 bytes, or
50.651 percent.

## Window sweep

SVE256 uses a 128 KiB W13 packed-B tile and a 16 KiB W2 packed-B tile for this
shape. Positive speedup means W8A16 cache dequantization is faster.

| W13 tiles/thread | W2 tiles/thread | BF16 ms | W8 cache ms | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 8 | 37.261 | 37.176 | 1.002x |
| 2 | 16 | 37.003 | 36.184 | 1.023x |
| 2 | 32 | 35.242 | 34.502 | 1.021x |
| 4 | 8 | 37.519 | 37.682 | 0.996x |
| 4 | 16 | 37.395 | 36.543 | 1.023x |
| 4 | 32 | 35.802 | 34.398 | 1.041x |
| 8 | 8 | 38.179 | 38.786 | 0.984x |
| 8 | 16 | 38.263 | 37.635 | 1.017x |
| 8 | 32 | 36.740 | 35.377 | 1.039x |

The best absolute point is `(t=8, w13=4, w2=32, R13=2, R2=1)`, corresponding
to a 512 KiB per-thread W13 dequant window and the full 512 KiB W2 owner stripe.

## Independent repeats

| Process | BF16 ms | W8 cache ms | Speedup | Gain |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 36.050 | 34.417 | 1.0475x | 4.75% |
| 2 | 36.042 | 34.403 | 1.0476x | 4.76% |
| 3 | 35.763 | 34.347 | 1.0412x | 4.12% |

The median of process medians is 36.042 ms for BF16 and 34.403 ms for W8A16,
or 1.0476x. The planner's zero-window full-stripe geometry independently
measured 36.880 versus 35.456 ms at the median of three process medians, or
1.0402x.

## Decision

Cache dequantization has a real 4-5 percent kernel-level benefit on this
captured SVE256 workload, unlike register dequantization on long routes. The
result remains below the declared 5 percent adoption gate and still has about
1.50 percent relative L2 error, so W8A16 remains explicit and experimental.
The next decision is model-level quality validation and cost-model selection;
do not enable W8A16 globally from this result.

## Command

```bash
PYTHONPATH=.:src FUSED_CPP_MOE_SVE=1 \
FUSED_CPP_MOE_W2_DIRECT_ROUTE=1 FUSED_CPP_MOE_W2_BF16_ROUTE=0 \
FUSED_CPP_MOE_PREPACK_THREADS=80 OMP_NUM_THREADS=80 OMP_DYNAMIC=FALSE \
OMP_PROC_BIND=close OMP_PLACES=cores \
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_w8a16_plan_v2.py \
  --profile tmp/arm_codex_numa3_80c_quick_20260821.json \
  --workload cpu_moe_schedule_optimization/planners/workloads/deepseek_v4_flash_2048_seq70.json \
  --experts 256 --tokens 2048 --top-k 6 --hidden 4096 --intermediate 512 \
  --threads 80 --cpu-start 240 --tp-degree 4 --swiglu-limit 10 \
  --w8-mode cache --w13-window-sweep 4 --w2-window-sweep 32 \
  --sweep-warmup 7 --sweep-runs 31 --warmup 7 --runs 31
```
