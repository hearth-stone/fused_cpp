# AmazonC5192Cores NUMA0 analytical thin calibration

Date: 2026-08-01

## Scope

- Host: `AmazonC5192Cores`, NUMA0 CPUs `0-95` (96 cores).
- Kernel: ARM SVE BF16 JIT exact-M, N-split, split-W13 enabled.
- Expert shape: TP4, `H=4096`, `F=512`, 256 local experts.
- Holdout call: 96 consecutive distinct experts; 32 MiB HugeTLB allocation.
- Isolated grid: routes `1,2,8,12,24,48,96,192,384,768,1536,2040` and
  widths `1,2,4,8,16,24,32,48,64,96`.
- Contention grid: routes `1,12,48,192,768,2040` and nine homogeneous
  96-core shapes from `96x1T` through `1x96T`.

The calibration uses no contention row. Twelve isolated points
(`M={12,192,2040}`, `T={1,4,16,48}`) fit only expert-fixed cost, per-route
cost, and one common GEMM residual scale. All remaining isolated points and all
contention points are holdout observations.

## Service anchors

| Threads | Matrix TFLOP/s | L1 GB/s | L2 GB/s | LLC GB/s | DRAM GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.413 | 128.6 | 99.5 | 67.8 | 42.6 |
| 8 | 3.300 | 1008.2 | 792.0 | 188.9 | 180.5 |
| 16 | 6.602 | 2036.6 | 1589.2 | 347.9 | 303.9 |
| 24 | 9.850 | 2588.8 | 2369.8 | 412.1 | 320.7 |
| 32 | 13.083 | 4078.1 | 3179.0 | 374.0 | 372.0 |
| 48 | 19.674 | 6031.0 | 4724.6 | 580.1 | 400.8 |
| 64 | 26.322 | 7422.5 | 6263.1 | 798.0 | 388.7 |
| 96 | 39.007 | 9437.1 | 9334.4 | 1188.7 | 395.9 |

The matrix probe is register-only M12 BFMMLA. L1/L2/LLC/DRAM probes are
B-only endpoint-to-register probes, so hierarchy bounds compose with `max`
rather than being added as independent link times. The DRAM curve reaches its
sustainable knee at about 48 threads and 396 GB/s. The LLC measurements are
non-monotonic and the compact power curve has 22.9% MAPE and 59.5% maximum
error, which remains a known topology limitation.

Independent repeated-scan PMU measurements provide packed-B private-L2 miss
anchors: 18.0% below the transition, 62.3% at the nominal 2 MiB capacity, and
86.9% at 4 MiB. These are hardware/kernel retention anchors, not routed-expert
latency points.

The thin residual fit produced:

- expert fixed cost: `51.912 us`;
- per-route cost: `479.10 ns`;
- common W13/W2 scale: `1.18546`;
- training MAPE: `6.11%` over 12 points.

## Holdout result

| Metric | Service-only first pass | Final thin calibration | Gate |
| --- | ---: | ---: | ---: |
| Isolated MAPE, all 120 points | 27.87% | 9.81% | 10% |
| Isolated MAPE, 108 true holdout points | 28.77% | 10.22% | 10% |
| Isolated holdout P90 | 54.50% | 23.06% | diagnostic |
| Contention MAPE, 54 points | 30.58% | 17.65% | diagnostic |
| Contention P90 | 49.60% | 47.79% | 15% |
| Mean measured shape regret | 9.90% | 3.33% | diagnostic |
| Maximum measured shape regret | 32.31% | 8.17% | 5% |

The cache-retention correction materially improves planner ranking, but the
calibration still fails all three production gates when the isolated metric is
computed on true holdout points.

| Routes | Predicted shape | Measured best | Selected measured/predicted | Regret |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `96x1T` | `24x4T` | `3.118 / 3.104 ms` | 1.71% |
| 12 | `96x1T` | `24x4T` | `3.204 / 3.109 ms` | 0.29% |
| 48 | `12x8T` | `6x16T` | `6.043 / 5.135 ms` | 8.17% |
| 192 | `12x8T` | `6x16T` | `13.491 / 11.639 ms` | 2.90% |
| 768 | `6x16T` | `12x8T` | `47.682 / 43.938 ms` | 6.92% |
| 2040 | `24x4T` | `24x4T` | `119.593 / 138.988 ms` | 0.00% |

The largest absolute errors remain long-route `1T/2T` shapes. The current
capacity curve turns packed-B retention into LLC/DRAM spill too aggressively
for those windows; for example, route 2040 `48x2T` is overestimated by 103%.
Conversely, routes 48 and 192 still underprice the active-window penalty enough
to prefer 8T instead of the measured 16T. A single NUMA-wide LLC power curve and
three L2 retention anchors cannot yet describe both regimes.

## Decision

Do not make the analytical backend the production default. Keep the empirical
backend as the runtime model and use this calibration for shadow diagnostics.
The next calibration revision needs an independent multi-team packed-B
retention/refill probe and topology-aware LLC service, followed by unseen mixed
distribution and stage-window validation. It must not fit a task-pair slowdown
matrix or consume contention rows as calibration inputs.

## Artifacts

- `cpu_moe_schedule_optimization/cost_model/profiles/analytic_services_amazon_c5_192c_numa0_20260801.json`
- `cpu_moe_schedule_optimization/cost_model/profiles/analytic_machine_amazon_c5_192c_numa0_sve_jit_thin_20260801.json`
- `cpu_moe_schedule_optimization/cost_model/profiles/analytic_holdout_amazon_c5_192c_numa0_tp4_F512_E256_splitw13_20260801.json`
- `optimizations/fused_moe_sve/results/amazon_192c_analytic_thin_fit_20260801.json`
- `optimizations/fused_moe_sve/results/amazon_192c_analytic_thin_validation_20260801.json`
