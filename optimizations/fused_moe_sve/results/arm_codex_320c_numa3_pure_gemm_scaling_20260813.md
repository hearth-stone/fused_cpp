# Arm-codex NUMA3 pure-GEMM scaling

Date: 2026-08-13

## Question

After introducing per-stage expert windows and machine-specific scheduling, is
the pure SVE BF16 GEMM body still a material bottleneck on the 320-core
Arm-codex host? This experiment separates three effects that end-to-end MoE
timing otherwise combines:

1. isolated scaling of one expert as its N-split team grows;
2. aggregate throughput when all 80 cores in one NUMA node are occupied; and
3. the effect of the two independent 40-core LLC domains inside that NUMA node.

## Configuration

- Host: `Arm-codex-internal`, 320 Arm cores, four NUMA nodes.
- Placement: NUMA3, CPUs `240-319`, with memory bound to NUMA3.
- LLC topology: CPUs `240-279` and `280-319` are separate 40-core, 70 MiB LLC
  domains.
- Runtime: SVE256 JIT exact-M kernel, compiled with
  `FUSED_CPP_SVE_VECTOR_BITS=256` and effective `-O2`.
- Pages: explicit 32 MiB HugeTLB packed-weight backing.
- W13: `K=4096`, `N=1024`, 8 MiB packed B per expert.
- W2: `K=512`, `N=4096`, 4 MiB packed B per expert.
- Routes: `M=12`, `216`, and `2040`.
- Measurement: prepacked A, rotating cold expert weights, and row-major FP32 C
  store. The measurement excludes gather, SiLU, packC, merge, planner, and
  worker-pool scheduling.

Therefore these numbers characterize the pure GEMM service available to the
fused expert, not complete W13/W2 stage or operator time.

## Isolated expert scaling

One `M=2040` expert was run with increasing N-split width. Efficiency is
relative to ideal scaling from the corresponding 1T point.

| Threads | W13 time | W13 | Efficiency | W2 time | W2 | Efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 188.39 ms | 90.8 GFLOP/s | 100.0% | 98.93 ms | 86.5 GFLOP/s | 100.0% |
| 2 | 95.28 ms | 179.6 GFLOP/s | 98.9% | 49.77 ms | 171.9 GFLOP/s | 99.4% |
| 4 | 47.00 ms | 364.1 GFLOP/s | 100.3% | 24.88 ms | 344.0 GFLOP/s | 99.4% |
| 8 | 23.47 ms | 729.0 GFLOP/s | 100.4% | 12.43 ms | 688.2 GFLOP/s | 99.4% |
| 16 | 11.75 ms | 1.456 TFLOP/s | 100.2% | 6.22 ms | 1.376 TFLOP/s | 99.4% |
| 32 | 6.04 ms | 2.832 TFLOP/s | 97.4% | 3.12 ms | 2.745 TFLOP/s | 99.2% |

The kernel therefore remains almost linearly scalable through 32 threads for a
single long-route expert. A poor end-to-end width decision on this host must
not be attributed to an intrinsic 16T or 32T GEMM scaling failure.

## Full NUMA throughput

For each `(M, threads)` cell, enough independent experts ran concurrently to
occupy all 80 cores. Aggregate throughput is based on the slowest group in the
wave, so the table includes cross-expert and cache-service contention.

### W13

| M | 1T | 2T | 4T | 8T | 16T | Best |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 12 | 3.963 | 4.171 | **5.132** | 2.469 | 2.362 | 20 experts x 4T |
| 216 | 4.716 | 5.213 | 6.185 | 6.277 | **6.552** | 5 experts x 16T |
| 2040 | 5.021 | 6.326 | 6.890 | **6.991** | 6.931 | 10 experts x 8T |

### W2

| M | 1T | 2T | 4T | 8T | 16T | Best |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 12 | **3.738** | 3.285 | 2.336 | 2.194 | 2.303 | 80 experts x 1T |
| 216 | 5.020 | 6.299 | 6.216 | 6.508 | **6.607** | 5 experts x 16T |
| 2040 | 6.069 | 5.808 | 5.945 | 5.885 | **6.301** | 5 experts x 16T |

Throughput entries are TFLOP/s. The optimum is stage- and route-dependent:
W13 prefers `4T/16T/8T` at `M=12/216/2040`, while W2 prefers `1T/16T/16T`.
The planner must retain distinct W13 and W2 service curves; a single
route-to-team-width rule cannot represent this host.

## Relation to the compute ceiling

The matching 80-core analytic service probe measured:

| Service | 80-core result |
| --- | ---: |
| Register-only BFMMLA | 7.353 TFLOP/s |
| L1-hot production GEMM | 6.926 TFLOP/s |
| L2-hot production GEMM | 7.055 TFLOP/s |
| LLC-resident B stream | 1,164.7 GB/s |
| Cold-B logical service probe | 379.9 GB/s |

The last row is logical packed-B bytes divided by probe time, not a measured
memory-controller byte count and therefore not a physical DRAM ceiling. A
later same-node cross-check measured 352.6-358.3 GB/s with an 8/4 GiB SVE
read-only stream. STREAM Triad reported 268.6 GB/s of algorithmic traffic; with
ordinary write-allocate stores its 24 counted bytes correspond to approximately
32 physical bytes, or `268.6 * 4 / 3 = 358.1 GB/s`. The two independent physical
traffic estimates agree. Use the cold-B probe only as a workload-specific
service rate and use approximately 355 GB/s as the sustainable NUMA3 pure-read
reference unless memory-controller counters establish a different value.

At `M=2040`, W13 reaches 6.991 TFLOP/s, effectively the measured L1-hot GEMM
ceiling. W2 reaches 6.301 TFLOP/s, about 91.0% of that ceiling. There is no
large remaining instruction-schedule gap in the W13 pure-GEMM body at the
NUMA-wide operating point; scheduler and fused-stage work dominate the next
optimization decisions.

## LLC-domain check

Repeating the saturated long-route experiment on only CPUs `240-279` gives:

| Stage | Best 40-core result | 80-core result | 40C to 80C scaling |
| --- | ---: | ---: | ---: |
| W13, M=2040 | 3.584 TFLOP/s at 8T | 6.991 TFLOP/s at 8T | 1.95x |
| W2, M=2040 | 3.328 TFLOP/s at 1T | 6.301 TFLOP/s at 16T | 1.89x |

The second LLC domain contributes nearly another full domain of throughput,
but not perfectly. Placement and width calibration should therefore be made at
LLC-domain granularity inside this NUMA node rather than treating the 80 cores
as one uniform shared-cache group.

W2 also has a reproducible saturated `2T` trough on one 40-core LLC domain:

| W2 M=2040 | 1T | 2T | 4T | 8T |
| --- | ---: | ---: | ---: | ---: |
| First sweep | 3.328 | 2.471 | 3.035 | 3.111 |
| Five-repeat recheck | 3.322 | 2.609 | 2.964 | 3.024 |

Entries are TFLOP/s. The isolated 2T kernel retains 99.4% linear efficiency,
so this trough is a concurrent cache/memory-service effect, not a defective 2T
N-split microkernel. Isolated scaling alone is insufficient for team-width
selection.

## Conclusion

The pure SVE GEMM implementation is not the main limiter for long-route experts
on Arm-codex NUMA3: one expert scales almost linearly through 32T, saturated
W13 reaches the measured production-GEMM ceiling, and using both LLC domains
scales aggregate throughput by about 1.9x. The remaining scheduling problem is
to choose stage-specific team widths while respecting LLC-domain contention.

The actionable rules from this run are:

1. Preserve separate W13 and W2 width models.
2. Model isolated scalability and saturated service independently.
3. Treat the two 40-core LLC domains as distinct placement resources.
4. Do not generalize the W2 saturated 2T trough into a kernel-level penalty.
5. Recalibrate complete fused stages before changing the production planner;
   this experiment intentionally excludes their epilogues and orchestration.

Raw captures from this run are retained under `tmp/arm_codex_gemm/`, and the
matching service-ceiling captures are under `tmp/arm_codex_calibration/`.
