# Single-core unique-weight cache window

Date: 2026-07-15

## Method

The host was `AmazonC5192Cores`, with 192 Neoverse-V3 cores split across two
NUMA nodes. Each core has 2 MiB private L2; NUMA node 0 has a 96 MiB shared L3.
The boundary repeat used CPU 48 with memory bound to NUMA node 0.

The benchmark directly called the production
`moe_sve_w2_packed_bf16_m12` assembly kernel. The fixed shape was `M=120`,
`K=4096`, BF16 input/weight/output, and one thread. N varied, making packed-B
weight size:

```text
weight_bytes = K * N * sizeof(bf16) = 8192 * N
```

Each call contains ten M12 panels, so it scans the same packed B ten times
inside one GEMM. Across calls there is no packed-B reuse: two warmups and nine
measured calls consume 11 distinct weight addresses with distinct values.
After these 11 weights, allocation adds and first-touches at least 192 MiB of
additional weights. It then reads one element from every cache line in that
tail, preventing streaming-store behavior from leaving the measured weights
resident. The measured weights are therefore older than an entire
two-L3-capacity read working set before execution begins.

The benchmark raises `SIGSTOP` only after all allocation, first-touch, packed A,
and output initialization. The runner then attaches ordinary-user `perf 7.0.6`
and resumes the process, excluding initialization from all counters. Seven
events fit in the ten available PMU slots without multiplexing:

```text
cycles:u,instructions:u,l2d_cache:u,l2d_cache_refill:u,
ll_cache_rd:u,ll_cache_miss_rd:u,stall_backend_mem:u
```

Command:

```bash
.venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/run_single_core_weight_window.py \
  --m 120 --k 4096 --cpu 48 --numa-node 0 \
  --warmup 2 --runs 9 --cold-tail-mib 192 \
  --output-json /tmp/single_core_weight_window.json
```

## Private-L2 transition

`L2 refill / weight line` divides total L2D refills by the 11 invocations
and by packed-B cache lines. It includes packed A, output, and control traffic,
so it is a transition indicator rather than a literal number of B scans.

| Weight | N | GFLOP/s | L2 refill / weight line | L2 refill / L2 access |
|---:|---:|---:|---:|---:|
| 0.50 MiB | 64 | 301.6 | 2.271 | 6.06% |
| 0.75 MiB | 96 | 307.0 | 2.636 | 7.30% |
| 1.00 MiB | 128 | 305.0 | 2.569 | 7.90% |
| 1.25 MiB | 160 | 298.4 | 2.901 | 8.35% |
| 1.50 MiB | 192 | 292.4 | 3.641 | 9.67% |
| 1.75 MiB | 224 | 307.3 | 5.368 | 12.54% |
| 2.00 MiB | 256 | 319.6 | 7.160 | 15.76% |
| 2.50 MiB | 320 | 311.7 | 7.102 | 15.91% |
| 3.00 MiB | 384 | 310.8 | 7.507 | 16.89% |
| 4.00 MiB | 512 | 300.8 | 8.756 | 19.68% |

The refill transition starts between 1.5 and 1.75 MiB and is clear at the
nominal 2 MiB capacity. A packed-B window equal to the full advertised L2 does
not remain resident because the active A panel, output, code, and replacement
conflicts share that cache. The robust packed-B-only L2 window is at most about
1.5 MiB for this loop.

To separate packed-B reuse from the A, C, and control traffic included in the
raw counter, an additional `M=12` run used 202 distinct cold weights per point.
That run scans B once per call; the `M=120` run scans B ten times. Let `R1` and
`R10` be L2D refills per call for `M=12` and `M=120`, and let `L` be the number
of cache lines in packed B. Assuming non-B traffic is approximately the same
per M12 panel, the fraction of the nine repeated B scans retained by L2 is:

```text
retained_fraction = (10 * R1 - R10) / (9 * L)
```

| Weight | M12 refill / B line | M120 refill / B line | Estimated repeated-B retention |
|---:|---:|---:|---:|
| 0.50 MiB | 0.970 | 2.271 | 82.6% |
| 0.75 MiB | 0.985 | 2.636 | 80.1% |
| 1.00 MiB | 0.992 | 2.569 | 81.7% |
| 1.25 MiB | 1.065 | 2.901 | 86.1% |
| 1.50 MiB | 0.990 | 3.641 | 69.6% |
| 1.75 MiB | 0.992 | 5.368 | 50.6% |
| 2.00 MiB | 1.056 | 7.160 | 37.7% |
| 2.50 MiB | 0.994 | 7.102 | 31.5% |
| 3.00 MiB | 0.992 | 7.507 | 26.8% |
| 4.00 MiB | 0.993 | 8.756 | 13.1% |

The differential control confirms that the sharp loss begins at 1.5 MiB and
continues through 2 MiB. Therefore 1.5 MiB is an engineering upper bound before
the steep transition, not a claim of complete residency. A policy requiring at
least roughly 80% repeated-scan retention should use about 1.25 MiB instead.
The estimate still depends on per-panel non-B traffic being comparable between
the two shapes, so it is a scheduling calibration rather than an architectural
capacity measurement.

## Shared-L3 transition

The table combines the broad sweep and one independent boundary repeat after
the explicit 192 MiB cache-line read scan.
GFLOP/s and LL miss ratio are medians where a point has multiple runs. `Drop`
is relative to the 48 MiB resident point at 315.7 GFLOP/s.

| Weight | N | Repeats | GFLOP/s | Range | Drop | LL read miss ratio | Backend memory stall / cycles |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 MiB | 6144 | 2 | 315.7 | 315.7-315.7 | 0.0% | 30.4% | 0.324% |
| 56 MiB | 7168 | 1 | 313.1 | 313.1 | 0.8% | 34.4% | 0.346% |
| 64 MiB | 8192 | 2 | 309.6 | 308.7-310.5 | 1.9% | 41.3% | 0.341% |
| 72 MiB | 9216 | 1 | 303.6 | 303.6 | 3.8% | 44.7% | 0.405% |
| 80 MiB | 10240 | 2 | 293.7 | 293.6-293.8 | 7.0% | 49.2% | 0.426% |
| 88 MiB | 11264 | 1 | 284.5 | 284.5 | 9.9% | 55.4% | 0.492% |
| 96 MiB | 12288 | 2 | 284.9 | 284.6-285.3 | 9.8% | 61.4% | 0.501% |
| 104 MiB | 13312 | 1 | 288.0 | 288.0 | 8.8% | 55.8% | 0.514% |
| 112 MiB | 14336 | 2 | 284.0 | 283.6-284.4 | 10.0% | 60.7% | 0.538% |
| 128 MiB | 16384 | 2 | 283.9 | 283.9-283.9 | 10.1% | 64.9% | 0.556% |
| 160 MiB | 20480 | 1 | 282.0 | 282.0 | 10.7% | 66.3% | 0.554% |
| 192 MiB | 24576 | 1 | 281.6 | 281.6 | 10.8% | 71.3% | 0.582% |

The transition band is 64-88 MiB. At a 5% throughput-loss tolerance, 72 MiB
is the largest measured near-peak point; 80 MiB is already outside the band.
The robust recommendation is 64 MiB, exactly two-thirds of the nominal 96 MiB
L3. At 88-96 MiB, throughput has reached the nonresident plateau and the LL
read miss ratio has crossed roughly 50-60%.

This is not a frequency or instruction-count artifact. Across 48-128 MiB, the
cycle/time frequency proxy stayed around 3.29-3.31 GHz and instructions per FLOP
stayed in 0.0484-0.0486. IPC fell from about 4.64 at 48 MiB to 4.17-4.20 at 96-128
MiB, while cycles per FLOP increased from about 0.0105 to 0.0116. The PMU and
throughput transitions therefore identify exposed cache/memory latency.

## Address-translation control

The LLC boundary points were repeated with `L1D_TLB`, `L1D_TLB_REFILL`,
`L2D_TLB_REFILL`, and page-fault events. Initialization remained outside the
perf region. The OS base page is 4 KiB, but the large anonymous allocation is
almost entirely backed by 2 MiB transparent huge pages. For example, the
128 MiB-weight process had 1,703,936 KiB of `AnonHugePages` out of 1,708,964 KiB
of anonymous RSS while stopped at the profiler boundary.

| Weight | GFLOP/s | L1 DTLB refill / access | L2 refill / L1 refill | Page faults |
|---:|---:|---:|---:|---:|
| 48 MiB | 315.2 | 0.00255% | 3.85% | 5 |
| 80 MiB | 292.1 | 0.00229% | 5.51% | 5 |
| 96 MiB | 287.8 | 0.00169% | 3.75% | 5 |
| 128 MiB | 282.8 | 0.00210% | 4.47% | 5 |
| 192 MiB | 282.8 | 0.00151% | 3.68% | 5 |

The L1 DTLB miss rate is approximately 0.0015-0.0026%, and only 3.7-5.5% of
those misses continue to an L2 DTLB refill. Neither ratio has a transition at
the 64-96 MiB throughput boundary. Separate fault-type samples at 48, 96, and
192 MiB classified all five faults as minor and reported zero major faults;
the constant count does not scale with the weight. Address translation and
demand paging therefore do not explain the LLC-window performance loss under
the host's current THP configuration.

`LL_CACHE_MISS_RD` counts attributable demand-read transactions, not every
64-byte transfer, and does not include writeback traffic. Its ratio is used to
locate the transition, not to estimate exact DRAM bytes. Neoverse-V3 does not
implement the core-PMU `L3D_CACHE_REFILL` event on this host, and no L3/DRAM
uncore PMU is exposed to Linux.
