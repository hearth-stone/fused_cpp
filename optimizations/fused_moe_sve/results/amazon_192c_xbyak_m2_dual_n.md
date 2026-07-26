# M1/M2 dual-N SVE JIT experiment

## Scope

- Host: `AmazonC5192Cores`, NUMA node 0, CPUs 0-95.
- Shape: `H=4096`, `F=512`, split-W13 enabled.
- Candidate: `FUSED_CPP_MOE_SVE_JIT_M2_DUAL_N=1`.
- Default behavior is unchanged; the candidate is experimental and off.

The generated M1/M2 kernel computes two adjacent N tiles in one K loop. It
shares each packed-A load and loop-control sequence across eight BFMMLA
instructions. W13 also shares packC offsets and cached SiLU constants between
the two tile epilogues.

The candidate uses dual-N only for `K>=1024`. Without this guard, the real W2
shape (`K=512,N=4096`) increased the 32-expert W2 phase from about 3.37 ms to
4.54-4.72 ms and regressed the complete expert by about 10%. W13 has
`K=4096`, so it remains eligible.

## Correctness

The exact-M static-assembly comparison passes for polynomial degrees 4, 5,
and 6:

```text
3 passed
```

## Single-core results

A pure cold-weight GEMM rotating 64 expert weights improved packed-B bandwidth
by about 0.7-1.2%, depending on M and N-range count. In the complete scheduled
expert, 61-call same-process A/B measurements gave:

| Routes | Baseline | Dual-N | Gain |
|---:|---:|---:|---:|
| 1 | 11.084 ms | 11.009 ms | +0.68% |
| 2 | 11.094 ms | 11.022 ms | +0.65% |

A scheduled trace over 32 experts attributed the change as follows:

| Stage | Baseline | Dual-N | Change |
|---|---:|---:|---:|
| W13 fused SiLU/packC | 7.3415 ms | 7.2598 ms | -1.11% |
| W2 packed | 3.3885 ms | 3.4447 ms | +1.66% |
| Scheduled compute | 11.1260 ms | 11.0462 ms | -0.72% |

The W2 code is identical after the K guard; its trace difference is run noise.

## Active-expert sweep

Each call processes 96 experts in one or more waves. Only the number of
simultaneously active one-thread experts changes. Every boundary point uses 31
same-process alternating samples.

| Active experts | M1 gain | M2 gain |
|---:|---:|---:|
| 8 | +2.30% | +2.23% |
| 12 | +0.32% | +0.11% |
| 16 | +0.53% | +0.64% |
| 24 | -5.72% | -5.40% |
| 32 | -2.15% | -1.20% |
| 96 | -2.42% | -4.98% |

Dual-N therefore is not a generally faster M1/M2 kernel. It improves
instruction efficiency at low concurrency, but reverses for some larger
expert waves. A production selector would need planner-visible concurrency and
barrier-topology conditions; M and K alone are insufficient.

## PMU attribution at 8 and 24 active experts

`perf stat` was attached after setup and warmup to exactly the pinned worker
TIDs for 500 profiled calls. Counts below are dual-N changes relative to the
ordinary JIT.

| Active experts | Cycles | Instructions | L2 refill | LL read miss | Memory stall |
|---:|---:|---:|---:|---:|---:|
| 8 | -2.42% | -10.66% | +0.68% | -21.76% | +2.87% |
| 24 | +5.23% | +13.55% | +0.41% | -63.92% | -21.58% |

At 24 experts, both last-level misses and memory-stall cycles fall sharply
while elapsed cycles and retired instructions rise. This rules out insufficient
DRAM bandwidth or an increased L2 refill volume as the cause of the 24-way
regression.

One traced 24-expert call provides the corresponding synchronization evidence:

- Ordinary W13 wave maxima were 0.67-0.74 ms; dual-N maxima were 0.82-0.91 ms.
- Ordinary W2 per-wave CV was 9.7-15.6%; dual-N CV was 16.8-22.7%.

The external schedule has a global barrier between waves. The larger lane-time
spread raises the critical-path lane time and leaves other workers spinning;
those waits explain why aggregate retired instructions can rise even though
the generated dual-N body contains less A-load and loop-control work. The
planner-visible condition is therefore not just aggregate bandwidth: expected
within-wave lane variance and barrier topology also matter.
