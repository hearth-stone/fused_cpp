# AmazonECS8Cores V1 cache-derived analytical GEMM-core calibration

Date: 2026-08-02

## Scope

- Host: `AmazonECS8Cores`, CPUs `0-7` (8 Neoverse-V1 cores).
- Kernel: ARM SVE BF16 JIT M12 packed-A/packed-B K-loop, compile-time VL 256 bits.
- Widths: `1,2,4,8`, with explicit worker affinity.
- Allocation: ordinary pages; these cache-resident probes do not require HugeTLB.
- Retained run: 64/512 matrix warmup/runs, 128/32768 L1-core warmup/runs,
  and 64/2048 L2-hot warmup/runs.

## Hardware-derived geometry

Linux sysfs reports:

| Resource | Detected capacity | Probe fraction | Generated geometry | Footprint |
| --- | ---: | ---: | --- | ---: |
| L1D per core | 64 KiB | 0.625 | M12/K728/N16 full-no-store | 40,768 B A+B |
| L2 per core | 1 MiB | 0.5 | M12/K9360/N16 full-no-store | 524,160 B A+B |
| LLC | 32 MiB | 1/6 | K4096/N640 B-only | 5 MiB B |

The L1 geometry is identical to the V3 geometry because both machines expose a
64 KiB private L1D. The L2 geometry automatically halves K from 18720 to 9360
because V1 exposes a 1 MiB private L2 rather than V3's 2 MiB.

## Compute services

| Threads | Register-only BFMMLA | L1-hot GEMM core | L2-hot GEMM diagnostic |
| ---: | ---: | ---: | ---: |
| 1 | 0.331 TFLOP/s | 0.289 TFLOP/s | 0.302 TFLOP/s |
| 2 | 0.662 TFLOP/s | 0.483 TFLOP/s | 0.495 TFLOP/s |
| 4 | 1.058 TFLOP/s | 0.872 TFLOP/s | 0.890 TFLOP/s |
| 8 | 1.833 TFLOP/s | 1.621 TFLOP/s | 1.663 TFLOP/s |

The original short run produced one 8-thread L1-core outlier at 0.399 TFLOP/s:
per-call medians remained about 1.43 us, but a few long samples inflated the
timed sum. Two independent repeats measured 1.654 and 1.576 TFLOP/s. The
retained longer run measured 1.621 TFLOP/s and removes this timing artifact.

## Interpretation

The register-only loop omits A/B loads, address updates, and the production
K-loop control mix. It overstates the executable full-loop rate by 14.4% at one
thread and 13.1% at eight threads, so it remains an ISA diagnostic rather than
the planner compute resource.

The L2-hot diagnostic is 2.1--4.5% faster than the L1-hot probe. This does not
mean L2 is intrinsically faster than L1: the L2 geometry has K=9360 and
amortizes call/prologue/epilogue costs more effectively than the K=728 L1
geometry. Therefore the current L1-hot service is an appropriate conservative,
executable GEMM-core ceiling, but not a strict mathematical peak. A future
refinement can repeat the same L1-resident K chunk inside one generated call so
that the cache footprint remains bounded while fixed costs are amortized like a
production-length K loop.

The V1 all-core result also confirms that the service curve must retain measured
thread-width derating. Eight-thread L1-hot throughput is 5.60 times the
single-thread result, not 8 times, and the register-only loop shows a similar
5.54 times scaling limit. This loss exists even without lower-cache traffic and
must not be assigned to DRAM contention.

## Command

```bash
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
PYTHONPATH=src:cpu_moe_schedule_optimization/cost_model \
taskset -c 0-7 .venv/bin/python \
  cpu_moe_schedule_optimization/cost_model/profile_analytic_services.py \
  --output /tmp/analytic_services_v1_8c_hot_gemm_long_20260802.json \
  --cpu-ids 0-7 --widths 1,2,4,8 --seed 20260804 \
  --matrix-warmup 64 --matrix-runs 512 \
  --core-warmup 128 --core-runs 32768 \
  --hot-warmup 64 --hot-runs 2048
```

## Artifact

- `cpu_moe_schedule_optimization/cost_model/profiles/analytic_services_amazon_ecs_v1_8c_hot_gemm_20260802.json`
