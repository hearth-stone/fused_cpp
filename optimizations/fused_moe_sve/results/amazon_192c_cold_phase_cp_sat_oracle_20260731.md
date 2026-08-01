# Amazon 192-Core Cold-Phase CP-SAT Oracle

Date: 2026-07-31

## Question

Estimate the scheduling headroom of mixed-width experts relative to the
current 96-core strict `12x8T` plan without treating arbitrarily many cold
experts as if all retained isolated throughput.

This is an offline model experiment. It did not execute kernels on the target
host. The solver consumed the AmazonC5192Cores profile and the repository's
deterministic workload histograms, so solver placement is host-independent.

## Model

- Profile:
  `contention_async_amazon_c5_192c_dual_numa_tp4_sve_F512_E256_splitw13_schema_v2_xbyak_exactm_20260727.json`
- Shape: TP4, H=4096, F=512, E=256, BF16 SiLU, SVE JIT exact-M.
- Rank capacity: 96 cores.
- Kernel policy: split-W13 with task policy
  `amazon_c5_192c_tp4_f512_v1`.
- Fixed baseline: the current `12x8T` LPT lane DAG. The oracle preserves its
  lane dependencies and 8T modes, but may insert model-optimal cold-resource
  waits inside each lane.
- Mixed modes: `1,2,4,8,16T`, whole-expert fixed width.
- Cold reference: `T_iso(min(M, 12), T)`.
- Cold traffic: one scan of all W13 and W2 packed-B ranges.
- DRAM ceiling: 336.4 GB/s, the measured single-NUMA STREAM Triad rate.
- Quantization: 1000 ns and 0.25 GB/s.
- A selected team remains occupied across all internal resource waits.

The default `expert` granularity fluid-aggregates all cold ranges into one
cold phase and all remaining isolated time into one steady phase. It preserves
total `T_iso` and total cold packed-B bytes. The diagnostic `range` mode keeps
the current W13/W2 window sequence.

Only cold packed-B DRAM demand is constrained. Packed-A, stores, LLC-to-L2
refills, compute/frequency contention, merge, and communication are omitted.
The result is a scheduling surrogate, not a wall-time prediction.

## Commands

The main DSV4 command was run with 15 and 60 second limits:

```bash
.venv/bin/python \
  cpu_moe_schedule_optimization/planners/cold_phase_cp_sat_oracle.py \
  cpu_moe_schedule_optimization/cost_model/profiles/contention_async_amazon_c5_192c_dual_numa_tp4_sve_F512_E256_splitw13_schema_v2_xbyak_exactm_20260727.json \
  --workload dsv4-real-2048-seq70 \
  --num-cores 96 \
  --widths 1,2,4,8,16 \
  --baseline-width 8 \
  --dram-bandwidth-gbps 336.4 \
  --phase-granularity expert \
  --time-quantum-ns 1000 \
  --max-time-s 60 \
  --workers 8
```

The tiered and bimodal cases used the same command with their workload name
and a 15 second limit. A DSV4 diagnostic replaced
`--phase-granularity expert` with `range`.

## Results

`LB-UB` is the CP-SAT best objective bound through the best feasible
incumbent. For DSV4, the table combines the strongest valid bound and
incumbent from the 15 and 60 second runs. The gain interval is computed from
both fixed and mixed optimum intervals, not from incumbent values alone.

| Workload | Fixed 12x8T LB-UB | Mixed LB-UB | Mixed gain interval | Incumbent gain |
| --- | ---: | ---: | ---: | ---: |
| Captured DSV4 | 11.214-12.500 ms | 8.345-8.782 ms | 27.7%-49.8% | 42.3% |
| Long/short bimodal | 11.368 ms exact | 6.699-6.992 ms | 62.6%-69.7% | 62.6% |
| Tiered hotspot | 6.510 ms exact | 5.409-6.266 ms | 3.9%-20.4% | 3.9% |

The best DSV4 mixed incumbent selected:

| Width | Experts |
| ---: | ---: |
| 1T | 173 |
| 2T | 7 |
| 4T | 7 |
| 8T | 34 |
| 16T | 2 |

The result is not a uniform move to narrow teams. Most short and medium jobs
use narrow teams, while wider modes remain useful for selected long jobs and
tail filling.

## Sensitivity

Ten-to-fifteen-second DSV4 runs changed the assumed DRAM ceiling:

| DRAM ceiling | Fixed incumbent | Mixed incumbent | Incumbent gain | Mixed gap |
| ---: | ---: | ---: | ---: | ---: |
| 300.0 GB/s | 12.740 ms | 9.974 ms | 27.7% | 6.2% |
| 336.4 GB/s | 12.500 ms | 8.782 ms | 42.3% | 5.0% |
| 375.9 GB/s | 12.006 ms | 8.396 ms | 43.0% | 11.1% |

The non-monotonic gain percentage is partly solver-gap variation. The stable
observation is that the absolute mixed bound depends materially on the DRAM
ceiling, so that value must remain machine calibration rather than a universal
constant.

The range-level DSV4 diagnostic found `13.152 -> 12.971 ms`, only 1.4%
incumbent improvement, but retained a 35.7% mixed solver gap and a 57.6% gain
upper bound. It is inconclusive, not evidence that the fluid result is
achievable. The range model needs stronger decomposition or a longer search.

## Interpretation

The CP-SAT bounds prove, within the fluid surrogate, that fixed `12x8T` leaves
substantial scheduling headroom when cold packed-B demand can be staggered and
short experts can run in narrow teams beside long steady phases. For captured
DSV4, the model-internal gain is at least 27.7% under the stated 336.4 GB/s
ceiling.

It does not prove a 27.7% kernel or E2E speedup. The omitted shared resources
can slow the mixed schedule, and aggregating later W13/W2 cold ranges into an
expert prefix makes the schedule easier to optimize than the exact execution.

The next useful step is to translate one mixed incumbent into a bounded
runtime candidate and measure it. Production planner changes should wait for
that validation and for a converged range-level comparison.
