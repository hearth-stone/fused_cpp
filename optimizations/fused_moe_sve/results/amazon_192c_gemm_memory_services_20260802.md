# AmazonC5192Cores GEMM operand memory-service curves

Date: 2026-08-02

## Scope

- Host: `AmazonC5192Cores`, NUMA0 CPUs `0-95`.
- Kernel: production SVE BF16 JIT M12 full-no-store GEMM body.
- Geometry: `M=12`, `K=728`, `N=16` (`n_tile=8`).
- Cache geometry: 64 KiB private L1D, 2 MiB private L2, and 96 MiB LLC.
- Packed-B allocation: 32 MiB HugeTLB through `/dev/hugepages-32M`.
- Widths: `1,2,4,8,12,16,24,32,48,64,80,96`.

The geometry is derived from 62.5% of private L1D. One scan reads 17,472 B of
packed A and 23,296 B of packed B, executes 279,552 physical FLOPs, and has a
40,768 B combined footprint. The output store and fused epilogue are disabled,
so this experiment isolates the production GEMM request stream.

## States

| State | Rotating operand | Cache-hot operand | Counted stream intensity |
| --- | --- | --- | ---: |
| A-hot/B-stream | B | A | 12.00 FLOP/B |
| A-stream/B-hot | A | B | 16.00 FLOP/B |
| A+B-stream | A and B | none | 6.857 FLOP/B |

The hot operand is still loaded by the GEMM. Its bytes are excluded only from
the reported DRAM-stream denominator. Every worker owns a disjoint rotating
window. Each window is at least 8 times its private L2, and the aggregate
window is at least 4 times the NUMA LLC. There is no timed-copy reuse.

Worker processes first pack A and allocate output, stop inside the native
benchmark immediately before warmup, and are then released together by the
parent. Setup traffic and process launch skew are therefore outside the
concurrent service interval. Each table entry is the median of five repeats;
aggregate time is the slowest worker's complete timed interval.

## Results

The GB/s columns count only the designated rotating operand bytes. TFLOP/s is
the useful execution rate of the identical GEMM body.

| Cores | A-hot/B-stream GB/s | TFLOP/s | A-stream/B-hot GB/s | TFLOP/s | A+B-stream GB/s | TFLOP/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 25.63 | 0.308 | 13.08 | 0.209 | 34.82 | 0.239 |
| 2 | 44.82 | 0.538 | 25.64 | 0.410 | 62.05 | 0.425 |
| 4 | 89.88 | 1.079 | 50.23 | 0.804 | 120.48 | 0.826 |
| 8 | 169.21 | 2.031 | 93.28 | 1.492 | 207.77 | 1.425 |
| 12 | 240.95 | 2.891 | 136.22 | 2.180 | 265.86 | 1.823 |
| 16 | 292.07 | 3.505 | 174.88 | 2.798 | 319.13 | 2.188 |
| 24 | 303.72 | 3.645 | 241.88 | 3.870 | 305.32 | 2.094 |
| 32 | 311.08 | 3.733 | 290.07 | 4.641 | 306.59 | 2.102 |
| 48 | 308.15 | 3.698 | 321.18 | 5.139 | 311.38 | 2.135 |
| 64 | 322.29 | 3.868 | 309.34 | 4.950 | 320.17 | 2.195 |
| 80 | 337.91 | 4.055 | 329.36 | 5.270 | 340.98 | 2.338 |
| 96 | 364.22 | 4.371 | 354.52 | 5.672 | 363.46 | 2.492 |

At 96 cores, the three designated byte rates are all close to the measured
NUMA Copy ceiling of 375.9 GB/s: 96.9%, 94.3%, and 96.7%. This does not make
the states interchangeable. Useful GEMM throughput differs by more than 2x
between B-hot and both-stream because the kernel consumes different bytes per
FLOP and exposes different operand latency.

The A-stream/B-hot curve grows more slowly through 24 cores because each byte
of the counted A stream supports 16 FLOPs. Conversely, the both-stream state
creates the largest low-width byte demand but the lowest useful arithmetic
rate. A generic DRAM curve cannot represent both effects.

## Window convergence

The 96-core point was repeated with larger no-reuse windows:

| Private/LLC multiples | A-hot/B-stream | A-stream/B-hot | A+B-stream |
| --- | ---: | ---: | ---: |
| 8x / 4x (retained) | 364.22 GB/s | 354.52 GB/s | 363.46 GB/s |
| 8x / 8x | 364.28 GB/s | 356.40 GB/s | 363.52 GB/s |
| 16x / 16x | 368.19 GB/s | 357.02 GB/s | 360.54 GB/s |

All retained values are within 1.1% of the 16x/16x control. Smaller 2x windows
overstated the high-core A-stream/B-hot result by up to 12.6%, which is why the
profiler defaults use the larger eviction window.

## Modeling consequence

Retain these as three phase-specific GEMM service measurements:

1. Use A-hot/B-stream for a cold packed-B panel whose packed-A panel is already
   resident.
2. Use A-stream/B-hot for a steady packed-B reuse phase that advances packed A.
3. Use A+B-stream as the fully cold control and validation bound.

They should eventually calibrate request-generation efficiency against one
shared physical DRAM service, rather than become three independent capacities.
No production planner or cost-model behavior changes in this experiment.

## Four-state expert decomposition

A first offline validation applied the four cache states to one production
shape:

```text
TP4 H=4096 F=512, M=2040, 4 threads
W13: N=1024, two 4 MiB stage windows
W2:  K=512, N=4096, one 4 MiB stage window
```

For one thread in one stage window, let `P=ceil(M/12)` and let `Q` be the
number of assigned N8 tiles. With an M-panel outer loop and N-tile inner loop,
the ideal retained-window counts are:

```text
cold A / cold B = 1
hot  A / cold B = Q - 1
cold A / hot  B = P - 1
hot  A / hot  B = (P - 1) * (Q - 1)
```

The retained W13 window gives `P=170`, `Q=16`, and two repetitions. W2 gives
`P=170`, `Q=128`, and one repetition. The resulting counts and predicted
full-no-store GEMM-core time are:

| Stage | Cold/cold | Hot-A/cold-B | Cold-A/hot-B | Hot/hot | Predicted core |
| --- | ---: | ---: | ---: | ---: | ---: |
| W13 | 2 | 30 | 338 | 5,070 | 14.009 ms |
| W2 | 1 | 127 | 169 | 21,463 | 6.538 ms |
| Total | 3 | 157 | 507 | 26,533 | 20.547 ms |

The four per-wave costs at four threads were recovered from the two-N8-tile
service probes. At the K728 reference geometry they are 0.835 us cold/cold,
0.518 us hot-A/cold-B, 0.981 us cold-A/hot-B, and 0.410 us L1-hot/hot. W13
and W2 use the measured 0.422 us L2-hot/hot value: each thread revisits its B
tile only after traversing a 1 MiB B stripe, and the W13 96 KiB M12 A panel also
does not fit in the 64 KiB L1D. Costs were scaled linearly by stage K.

Three fresh-process validations rotated over 17 distinct experts, so no timed
expert reused weights. Median traced stage times were:

| Stage | Four-state prediction | Measured median | Error |
| --- | ---: | ---: | ---: |
| W13 fused SiLU/packC | 14.009 ms | 14.554 ms | -3.75% |
| W2 direct route store | 6.538 ms | 6.793 ms | -3.75% |
| GEMM stages | 20.547 ms | 21.347 ms | -3.75% |
| Full operator | core model only | 22.554 ms | n/a |

The untraced full-call median was 22.578 ms. The traced non-GEMM remainder was
about 1.23 ms, dominated by 0.65 ms gather and 0.53 ms route scatter. The 0.80
ms GEMM-stage residual is an upper bound on fused SiLU/packC, direct stores,
dispatch, and the K728-to-production-K scaling error because the four-state
probe intentionally omits stores and epilogues.

For this long-route expert, hot/hot contributes 89.7% of predicted GEMM-core
time, cold-A/hot-B contributes 9.6%, and both cold-B states together contribute
only 0.7%. Thus stage-window retention, the hot GEMM ceiling, and epilogues
matter much more than the first cold weight panel. Short-route experts have a
very different mixture and must not reuse these percentages.

## Pure-GEMM validation

The benchmark-only entrypoint was extended to traverse a complete positive
multiple of M12 with the same full-no-store JIT body. This removes SiLU, packC,
direct route stores, gather, scatter, and merge from the comparison. Bulk-M and
first-panel prefetch remained disabled, matching the default per-M12 dispatch.

For each point, four synchronized processes were pinned to CPUs `0-3`. Every
process received a disjoint packed-B expert that had not been used by any prior
timed wave. W13 measured one `K4096 x N128` stripe and counted it twice for the
two split-W13 windows; W2 measured one `K512 x N1024` stripe. Each table entry is
the median of 11 cold-weight waves. Error is `(prediction / measurement - 1)`.

| M | W13 predicted | W13 measured | Error | W2 predicted | W2 measured | Error | Total error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 0.099 ms | 0.108 ms | -8.68% | 0.048 ms | 0.052 ms | -8.06% | -8.48% |
| 24 | 0.181 ms | 0.217 ms | -16.25% | 0.086 ms | 0.104 ms | -17.10% | -16.53% |
| 48 | 0.347 ms | 0.374 ms | -7.16% | 0.163 ms | 0.182 ms | -10.19% | -8.15% |
| 192 | 1.341 ms | 1.327 ms | +1.08% | 0.625 ms | 0.615 ms | +1.55% | +1.23% |
| 768 | 5.317 ms | 5.137 ms | +3.49% | 2.470 ms | 2.365 ms | +4.44% | +3.79% |
| 2040 | 14.096 ms | 13.418 ms | +5.05% | 6.544 ms | 6.267 ms | +4.42% | +4.85% |

The total absolute error has a 6.50% median and a 16.53% maximum over the full
sweep. Restricting the result to `M>=192` gives a 3.79% median and 4.85% maximum.
The independent 21-wave M2040 run measured 19.727 ms against a 20.641 ms
prediction (+4.63%), reproducing the sweep result within 0.3%.

The state-counting model is therefore a useful long-route GEMM-core model, but
it is not yet a uniformly accurate short-route model. Three geometry effects
remain outside the four binary state labels:

1. The K728 calibration has a 40 KiB A+B footprint and fits L1D. A W13 M12/N8
   tile at K4096 is 160 KiB, so its nominal hot-A phase is served from a lower
   cache level.
2. Scaling the K728 state cost linearly to W2 K512 also scales fixed control,
   address-generation, and call costs that do not shrink with K.
3. The known M24/N>=128 throughput trough affects both production stripes and
   is not represented by multiplying independent M12 state costs.

At long routes the sign reverses because the no-reuse A-stream calibration is
more pessimistic than the recently packed, repeatedly scanned expert A. For
M2040 W13, the model assigns 1.951 ms to cold-A/hot-B transitions; the measured
residual after the modeled hot/hot and cold-B portions is only about 1.27 ms.
For W2, the modeled 6.374 ms hot/hot contribution alone exceeds the complete
6.267 ms measurement. These arithmetic checks show that the remaining error is
service-cost transfer across working-set geometries, not a state-count error.

The earlier fused-stage comparison understated this distinction: the core model
predicted 20.641 ms, pure GEMM measured about 19.7 ms, and the traced fused W13
plus W2 stages measured about 21.35 ms. Fused epilogues and stores therefore
masked the pure-core overestimate and made the same model appear to
underestimate the fused stages.

## Command

```bash
FUSED_CPP_MOE_HUGETLBFS_PATH=/dev/hugepages-32M \
PYTHONPATH=src OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
.venv/bin/python \
      cpu_moe_schedule_optimization/cost_model/profile_gemm_memory_services.py \
      --output /tmp/gemm_memory_services.json --cpu-ids 0-95 \
      --widths 1,2,4,8,12,16,24,32,48,64,80,96 \
  --warmup 4 --minimum-timed-scans 128 --repeats 5
```

Pure-GEMM validation sweep:

```bash
FUSED_CPP_MOE_HUGETLBFS_PATH=/dev/hugepages-32M \
PYTHONPATH=src OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
numactl --cpunodebind=0 --membind=0 taskset -c 0-3 \
.venv/bin/python \
  cpu_moe_schedule_optimization/cost_model/validate_gemm_four_state.py \
  --memory-services /tmp/gemm_memory_services_4c_20260802.json \
  --hot-services \
    cpu_moe_schedule_optimization/cost_model/profiles/analytic_services_amazon_c5_192c_numa0_hot_gemm_20260802.json \
  --output /tmp/pure_gemm_four_state_sweep_20260802.json \
  --cpu-ids 0,1,2,3 --m-values 12,24,48,192,768,2040 \
  --waves 11 --experts 320
```
