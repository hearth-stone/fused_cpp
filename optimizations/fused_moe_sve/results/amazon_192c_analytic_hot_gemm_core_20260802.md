# AmazonC5192Cores cache-derived analytical GEMM-core calibration

Date: 2026-08-02

## Scope

- Host: `AmazonC5192Cores`, NUMA0 CPUs `0-95` (96 cores).
- Kernel: ARM SVE BF16 JIT M12 packed-A/packed-B K-loop.
- Allocation: 32 MiB HugeTLB through `/dev/hugepages-32M`.
- Widths: `1,2,4,8,16,24,32,48,64,96`, with explicit worker affinity.
- Holdout: the unchanged TP4 `H=4096`, `F=512`, E256 split-W13 profile from
  2026-08-01. No contention observation is used to fit the calibration.

## Hardware-derived probe geometry

Linux sysfs reports:

| Resource | Detected capacity | Probe fraction | Generated geometry | Footprint |
| --- | ---: | ---: | --- | ---: |
| L1D per core | 64 KiB | 0.625 | M12/K728/N16 full-no-store | 40,768 B A+B |
| L2 per core | 2 MiB | 0.5 | M12/K18720/N16 full-no-store | 1,048,320 B A+B |
| LLC per NUMA rank | 96 MiB | 1/6 | K4096/N2048 B-only | 16 MiB B |

For SVE N tile `nu=8`, packed W13 needs a physical minimum of `2*nu=16`
columns. For cache level L, the profiler computes:

```text
K_L = 8 * floor(f_L * C_L / (16 * (12 + 2*nu)))
```

The L1 probe executes production A/B loads, address/control instructions, and
BFMMLA, but omits store and epilogue. The L2 version is a diagnostic. The old
register-only BFMMLA probe remains in the artifact but is not the planner's
compute resource.

## Service curves

| Threads | Register-only | L1-hot GEMM core | L2-hot GEMM | L2 B-only | LLC B-only | DRAM B-only |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.412 T | 0.340 T | 0.335 T | 100.1 GB/s | 67.6 GB/s | 43.4 GB/s |
| 2 | 0.823 T | 0.682 T | 0.666 T | 200.2 GB/s | 75.4 GB/s | 67.4 GB/s |
| 4 | 1.652 T | 1.363 T | 1.324 T | 393.2 GB/s | 136.7 GB/s | 125.7 GB/s |
| 8 | 3.289 T | 2.712 T | 2.646 T | 799.7 GB/s | 210.1 GB/s | 181.4 GB/s |
| 16 | 6.586 T | 5.436 T | 5.260 T | 1,589.6 GB/s | 350.3 GB/s | 283.0 GB/s |
| 24 | 9.865 T | 8.154 T | 7.976 T | 2,377.6 GB/s | 411.1 GB/s | 354.2 GB/s |
| 32 | 13.006 T | 10.846 T | 10.696 T | 3,185.4 GB/s | 367.0 GB/s | 373.3 GB/s |
| 48 | 19.539 T | 16.244 T | 15.822 T | 4,775.4 GB/s | 574.9 GB/s | 355.8 GB/s |
| 64 | 26.214 T | 20.911 T | 20.130 T | 6,262.7 GB/s | 798.3 GB/s | 386.1 GB/s |
| 96 | 39.309 T | 30.377 T | 28.215 T | 9,421.5 GB/s | 1,178.0 GB/s | 396.1 GB/s |

The L1-hot curve uses 64 warmups and 4096 timed calls per width because one call
is only about 0.8 us. The 96-core register-only number overstates the executable
GEMM-core ceiling by 29.4%. The L1-hot curve includes the load/control issue mix
and is therefore the active `gemm_core_flops` service. L2/LLC/DRAM endpoint
bounds remain separate. An independent 64/96T repeat measured 20.866/30.236
TFLOP/s, within 0.5% of the retained 20.911/30.377 TFLOP/s points.

### 96-core scaling recheck

The original profiler times every approximately 0.8 us L1-hot invocation and
uses the accumulated time of the slowest worker. To distinguish a real
all-core throughput loss from per-call timing and scheduler tails, a second
probe executed 131,072 untimed kernels inside each native call and timed only
the complete batch. Five randomized repeats produced:

| Threads | L1-hot GEMM core | Linear efficiency | Register-only BFMMLA | Linear efficiency |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.345 T | 100.00% | 0.419 T | 100.00% |
| 48 | 16.524 T | 99.69% | 20.021 T | 99.53% |
| 64 | 21.168 T | 95.78% | 26.651 T | 99.36% |
| 80 | 25.820 T | 93.46% | 33.264 T | 99.21% |
| 96 | 30.438 T | 91.82% | 39.810 T | 98.95% |

The 96-thread L1-hot range was 30.406--30.447 TFLOP/s across all five repeats.
Its slowest-worker batch duration rose from about 106.1 ms at 1T to 115.6 ms
at 96T, so the loss remains when per-kernel clocks are removed. Register-only
BFMMLA remains within 1.1% of linear scaling. Therefore V3 has no material
cross-core sharing limit in the matrix execution units, but the attainable
full load/control/compute loop develops a reproducible knee between 48 and 64
active cores.

### PMU attribution of the 48--96T loss

Long in-call batches were repeated under the Neoverse-V3 PMU. Event counts were
normalized by useful FLOPs, so the comparison is not affected by the doubled
thread count. The results reject both cache-capacity loss and ordinary DVFS:

- `L1D_CACHE_REFILL / L1D_CACHE` was `0.00545%` at 48T and `0.00496%` at
  96T. `STALL_BACKEND_L2D` was zero at both widths. The 40,768-byte A+B
  footprint therefore remained resident in the private 64-KiB L1D caches.
- `CPU_CYCLES / CNT_CYCLES` was `3.300` at both widths, so the core clock
  remained 3.3 GHz. Retired instructions per FLOP were also unchanged.
- Backend stall slots per FLOP increased by `29.2%`. The V3 top-down events
  attributed the growth to CPU-side backend pressure: `STALL_BACKEND_CPUBOUND`
  increased by `41.4%` per FLOP and `STALL_BACKEND_BUSY` by `87.0%`.
- The dispatch sub-events identify the dominant queue as the vector issue
  queue: `DISPATCH_STALL_IQ_VX` increased by `41.2%` per FLOP. The load/store
  issue-queue count was about 160 times smaller at 96T, and memory-bound cycles
  remained below 1% of total cycles.
- In the register-only control, `DISPATCH_STALL_IQ_VX` per FLOP changed by less
  than `0.1%` from 48T to 96T, matching its nearly linear throughput.

A separate victim/aggressor test fixed 48 full-loop workers on CPUs 0--47. The
median victim duration over three runs was `1.6794 s` in isolation, `1.6864 s`
(`+0.42%`) with 48 register-only BFMMLA workers on CPUs 48--95, and `1.8308 s`
(`+9.01%`) with 48 additional full load/BFMMLA workers. Thus the coupling is
socket-wide and is triggered by the combined load-to-vector-compute stream, not
by private-cache misses or matrix execution alone. The observable mechanism is
vector-dispatch backpressure. The platform does not expose a counter that can
distinguish firmware dispatch/power throttling from another implementation
specific vector-issue control, so that final hardware label remains an
inference rather than a measured fact.

For modeling, the one-thread L1-hot peak remains necessary, and the aggregate
active-core derating is real at high occupancy. A single power curve is not an
ideal representation because it would spread the 64--96T loss into the nearly
linear 1--48T region; a measured piecewise active-core efficiency curve is the
appropriate follow-up.

## Holdout comparison

The same 12 isolated points (`M={12,192,2040}`, `T={1,4,16,48}`) fit only an
expert fixed term, a per-route term, and one common stage scale.

| Metric | 2026-08-01 component model | L1-hot GEMM-core model |
| --- | ---: | ---: |
| Isolated MAPE, all 120 points | 9.81% | 8.94% |
| Isolated MAPE, 108 holdout points | 10.22% | 9.18% |
| Isolated holdout P90 | 23.06% | 17.41% |
| Contention MAPE, 54 points | 17.65% | 17.59% |
| Contention P90 | 47.79% | 49.57% |
| Mean measured shape regret | 3.33% | 3.33% |
| Maximum measured shape regret | 8.17% | 8.17% |

The residual fit changed from `51.912 us + 479.10 ns/route`, scale `1.18546`,
to `51.407 us + 418.76 ns/route`, scale `1.10842`. The explicit core ceiling
improves isolated holdout accuracy and reduces the common stage correction, but
the remaining route term shows that non-GEMM operator work is still not a
dedicated resource. Ranking failures remain routes 48/192/768, which require
multi-team packed-B retention and below-NUMA LLC topology rather than a higher
matrix peak.

## Decision

Use `gemm_core_flops` as the analytical compute resource and retain
`matrix_flops`, frontend, and L1 load curves as diagnostics/legacy fallback.
Do not switch the production planner from the empirical backend. The isolated
gate now passes (`9.18% <= 10%`), but the contention and regret gates remain far
above `15%` and `5%`.

## Command

```bash
FUSED_CPP_MOE_HUGETLBFS_PATH=/dev/hugepages-32M \
PYTHONPATH=src:cpu_moe_schedule_optimization/cost_model \
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python cpu_moe_schedule_optimization/cost_model/profile_analytic_services.py \
  --output /tmp/moe_analytic_calibration/analytic_services_hot_gemm_20260802.json \
  --cpu-ids 0-95 --widths 1,2,4,8,16,24,32,48,64,96
```

## Artifacts

- `cpu_moe_schedule_optimization/cost_model/profiles/analytic_services_amazon_c5_192c_numa0_hot_gemm_20260802.json`
- `cpu_moe_schedule_optimization/cost_model/profiles/analytic_machine_amazon_c5_192c_numa0_sve_jit_hot_gemm_20260802.json`
- `cpu_moe_schedule_optimization/cost_model/profiles/analytic_holdout_amazon_c5_192c_numa0_tp4_F512_E256_splitw13_20260801.json`
- `optimizations/fused_moe_sve/results/amazon_192c_analytic_hot_gemm_fit_20260802.json`
- `optimizations/fused_moe_sve/results/amazon_192c_analytic_hot_gemm_validation_20260802.json`
