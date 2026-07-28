# Amazon 192-Core Isolated CP-SAT Oracle Validation

Date: 2026-07-26

## Scope

This experiment compares the offline no-contention CP-SAT oracle with the
current fused MoE runtime on `AmazonC5192Cores`.

- Profile:
  `contention_async_amazon_c5_192c_dual_numa_tp4_sve_F512_E256_splitw13_schema_v2_xbyak_exactm_20260726.json`
- Shape: TP4, H=4096, F=512, E=256, BF16 SiLU, 2048 tokens, TopK=6.
- Oracle capacity: one 96-core rank with widths
  `1,2,4,8,16,32,48,64,96`.
- Oracle duration: profile `T_iso(routes, threads)`.
- Oracle model: whole-expert, fixed-width, non-preemptive jobs with no
  contention; 1000 ns ticks, 30 seconds, 32 CP-SAT workers.
- Runtime: split-W13 SVE JIT exact-M, current stage-window policy and automatic
  tail-pool selection.
- Runtime sampling: five warmups and 21 timed calls on each NUMA rank. Rank 0
  used CPUs `0-95`; rank 1 used `96-191`. The reported wall time is the median
  of index-paired rank maxima.

The two runtime processes execute concurrently but do not use the profiler's
per-call socket barrier. The paired-max result is therefore a full-machine
validation sample, not a replacement calibration table.

The runtime extension SHA256 was
`dabac02205bad377eea8507674de39cb8811282a5a682fdc821782040ed51063`;
the profile records
`e652d9aad3025a4836d0406110bcbdf3fdc1356aaba2bf789595359a58a92559`.
Absolute prediction errors include that identity mismatch.

## Results

`Oracle LB-UB` is the unresolved optimum interval after the solver limit.
`Actual gap` is the current dual-rank wall time relative to the unknown
isolated optimum. `Plan regret` evaluates the current plan with the same
quantized `T_iso`, so it excludes contention and operator stages outside that
surrogate.

| Workload | Oracle LB-UB ms | Solver gap | Current wall ms | Actual gap | Current-plan isolated regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform, `256x48` | 5.486-5.827 | 5.85% | 19.364 | 232.3-253.0% | 63.1-73.2% |
| Active set 8, `8x1536` | **7.930 exact** | 0% | 11.927 | 50.4% | **7.0%** |
| Active set 16, `16x768` | 5.349-6.679 | 19.91% | 11.848 | 77.4-121.5% | 6.6-33.1% |
| Active set 32, `32x384` | 5.432-6.495 | 16.37% | 10.225 | 57.4-88.2% | 2.5-22.6% |
| Active set 64, `64x192` | 5.463-6.252 | 12.62% | 10.742 | 71.8-96.6% | 10.5-26.4% |
| Active set 128, `128x96` | 5.402-5.782 | 6.57% | 10.994 | 90.1-103.5% | 16.9-25.2% |
| Tiered hotspot | 5.400-6.036 | 10.54% | 10.002 | 65.7-85.2% | 3.6-15.8% |
| Long/short bimodal | **6.1316 exact** | 0% | 10.930 | 78.3% | **0%** |
| Captured DSV4 | 5.640-6.055 | 6.85% | 15.849 | 161.7-181.0% | 17.1-25.7% |

The two exact points were rerun with 100 ns ticks. Their conservative
nearest-tick critical-path error bounds are 0.0004 ms for active-set-8 and
0.0090 ms for the 179-expert bimodal case.

## Interpretation

The no-contention oracle is not a prediction of achievable wall time. It
allows every simultaneously active expert to retain its isolated throughput.
This is particularly optimistic for uniform and captured routing, where the
solver can overlap many narrow cold-weight jobs without charging aggregate
LLC or DRAM demand. The runtime also includes TopK=6 route merge and other
operator work, while the profile isolated measurements use TopK=1 and
`skip_weighted=true`.

The useful absolute result is the decomposition:

- The current bimodal `16T` head plus `1T` tail pool reaches the exact
  fixed-duration isolated optimum. Its remaining 78.3% wall-time gap is not an
  isolated scheduling-choice gap.
- Active-set-8 is within 7.0% of its exact isolated optimum. Its 50.4% wall
  gap is again dominated by effects omitted from the oracle.
- Uniform retains a 63.1-73.2% isolated scheduling gap because the production
  candidate space does not express the oracle's unrestricted mixed-width
  schedule. That schedule is not expected to retain isolated rates under 256
  concurrent cold expert streams, so this is not evidence for adopting it.
- For the remaining workloads, solver uncertainty is still material. The
  reported regret intervals must not be collapsed to their incumbent values.

For reference, the current contention-aware model's prediction error against
the dual-rank current wall time was +1.6%, +1.0%, -17.4%, -1.7%, -15.0%,
-18.6%, -12.8%, -22.5%, and -33.6% in table order. Median absolute error was
15.0%. The profile/runtime hash mismatch and lack of a per-call dual-rank
barrier mean these values identify where a matched refresh is needed; they are
not a clean calibration verdict.

## Decision

The v1 oracle is useful as an optimistic scheduling bound and can prove that a
plan has closed the isolated scheduling gap, as in the bimodal tail-pool case.
It is not yet a useful hardware performance ceiling for high-active-set
workloads. The next bound should add aggregate matrix, LLC-to-L2, and DRAM
capacity constraints and use a calibration profile generated from the same
runtime extension.
