# M12 assembly K-block experiment

## Scope

- Host: `AmazonC5192Cores`, NUMA node 0, CPU 48.
- ISA: 128-bit runtime SVE BF16 (`VL=16 bytes`, `n_tile=8`).
- Private L1D: 64 KiB, four-way set associative, 64-byte cache lines.
- Compiler: `g++ -O2 -march=armv8.6-a+sve+bf16+i8mm`.
- W13-range shape: `M=12,K=4096,N=512`.
- W2 shape: `M=12,K=512,N=4096`.
- Packed B: 4 MiB for either shape; every invocation uses a distinct cold
  address followed by a guard, plus a scanned 192 MiB cold tail.
- Dense timing sweep: five warmups and 51 samples in each of five independent
  processes. Each process executes one fixed Kc, uses a distinct partial-C
  scratch per invocation, and scans packed A immediately before the timed
  kernel. The final baseline/Kc=800 check used seven independent processes
  per path.

Running one fixed Kc per process matters. Interleaving different Kc variants
changes the cache and hardware-prefetch state seen by the next variant and
moved individual medians by up to several percent. Production would select one
layout and Kc and then execute that path repeatedly, so fixed-Kc steady state is
the authoritative comparison.

This is a standalone experiment. It does not change the production fused-MoE
weight format, assembly symbols, or dispatch.

## Assembly dataflow

The baseline loop order is:

```text
Ntile -> full K -> BF16 store
```

The experimental assembly entrypoints use:

```text
Kchunk -> Ntile -> K4
```

The first K chunk zeros `z8-z31`. Intermediate chunks load and store all 24
FP32 accumulator vectors in an accumulator-native scratch layout; only the
last chunk executes the unchanged BF16 output epilogue. Scratch size is
`12 * N * sizeof(float)`. The arithmetic instruction order inside each output
element is unchanged, and an FP32 store/load is exact.

For W13 and `Kc=800`, the packed-A slice is 18.75 KiB and one packed-B N tile
is 12.5 KiB, for a 31.25 KiB active A+B window. Partial C is 24 KiB. There are
six K chunks, so the five boundaries add
`2 * 5 * 24 KiB = 240 KiB` of L1-oriented scratch traffic. The baseline and
K-block kernels both issue 6 MiB of A register loads over all 64 N tiles; the
intended difference is that K-block repeatedly serves an 18.75 KiB A slice
from L1 instead of repeatedly serving the full 96 KiB A panel from L2.

## Packed-B layouts

Two assembly addressing modes were tested:

1. `kblock_*` keeps production Ntile-major packed B. Loop interchange turns
   each 64 KiB Ntile stream into many short segments separated by 64 KiB.
2. `kblock_packed_*` uses Kchunk-major/Ntile-minor packed B. Each complete K
   chunk is contiguous, so B remains one compulsory cold stream while the A
   slice is reused.

Correctness uses patterned data and explicitly repacks B for every fixed Kc.
All 44 standalone load/K-block variants matched
`moe_sve_w2_packed_bf16_m12` bit for bit for `K=N=64`, W13, and W2. Timed
weights contain one finite BF16 value per copy, so either physical ordering is
valid without measuring an online repack. A production implementation would
prepack persistent weights into the selected format.

## Results

Keeping production packed B made every W13 K-block choice slower in the
initial 31-sample sweep:

| Variant | Median ms | Paired gain vs LD1H |
|---|---:|---:|
| `baseline_ld1h` | 0.1740 | reference |
| `kblock_256` | 0.2739 | -36.23% |
| `kblock_512` | 0.2571 | -32.13% |
| `kblock_768` | 0.2304 | -24.56% |
| `kblock_1024` | 0.2183 | -19.86% |

Kchunk-major packed B reverses the result. The following dense sweep ran one
fixed variant per process; the gain column is the median of process-indexed
throughput ratios against the isolated baseline processes:

| Variant | Process-median range | Median ms | GFLOP/s | Gain vs baseline |
|---|---:|---:|---:|---:|
| `baseline_ld1h` | 0.1773-0.1778 | 0.1777 | 283.23 | reference |
| `kblock_packed_704` | 0.1696-0.1726 | 0.1709 | 294.52 | +3.92% |
| `kblock_packed_720` | 0.1707-0.1716 | 0.1714 | 293.71 | +3.68% |
| `kblock_packed_736` | 0.1661-0.1677 | 0.1669 | 301.59 | +6.41% |
| `kblock_packed_752` | 0.1609-0.1629 | 0.1617 | 311.30 | +9.89% |
| `kblock_packed_768` | 0.1599-0.1618 | 0.1608 | 312.99 | +10.26% |
| `kblock_packed_784` | 0.1642-0.1668 | 0.1654 | 304.24 | +7.50% |
| `kblock_packed_800` | 0.1591-0.1611 | **0.1596** | **315.34** | **+11.37%** |
| `kblock_packed_816` | 0.1660-0.1673 | 0.1665 | 302.29 | +6.79% |
| `kblock_packed_832` | 0.1657-0.1667 | 0.1660 | 303.12 | +6.99% |
| `kblock_packed_848` | 0.1716-0.1740 | 0.1722 | 292.21 | +3.25% |
| `kblock_packed_864` | 0.1717-0.1726 | 0.1725 | 291.76 | +2.96% |
| `kblock_packed_880` | 0.1737-0.1755 | 0.1750 | 287.55 | +1.60% |
| `kblock_packed_896` | 0.1758-0.1779 | 0.1765 | 285.21 | +0.62% |

The final seven-process rerun used identical state controls for both paths:

| Variant | Process-median range | Median ms | Median GFLOP/s |
|---|---:|---:|---:|
| `baseline_ld1h` | 0.1769-0.1785 | 0.1779 | 282.9 |
| `kblock_packed_800` | 0.1592-0.1606 | 0.1595 | 315.5 |

The ratio of the two median throughputs is +11.5%.

The `Kc=800` result is not a favorable packed-B base-address accident. Holding
every cold B copy at each of four 4 KiB-spaced L1 set phases gave
`0.1588-0.1597 ms` for Kc=800, with a worst-color gain of 11.16%. It remained
the best candidate at every tested color. Kc=752 and Kc=768 were the next
stable choices, with worst-color gains of 9.38% and 10.13%, respectively.

## Kc rule

For one M12 N tile, the simultaneously reused packed-A slice and streamed-B
slice are:

```text
A_slice(Kc) = 2 * M * Kc bytes
B_slice(Kc) = 2 * n_tile * Kc bytes
W_tile(Kc)  = 2 * Kc * (M + n_tile) bytes
```

The measured useful envelope on this four-way 64 KiB L1D ends near half of
L1. Setting `W_tile` to `L1D/2` gives the first-order target:

```text
Kc_target ~= L1D_bytes / (4 * (M + n_tile))
          = 65536 / (4 * (12 + 8))
          = 819.2
```

The practical choice is a K4-aligned value just below this bound that also
avoids a pathological final K chunk. Kc=816 is closest to the formula but
leaves a 16-deep tail for K=4096; Kc=800 leaves a 96-deep tail and wins. The
16-step local oscillation around the envelope shows that cache-set mapping,
loop alignment, and tail shape still require a small empirical calibration.
The formula predicts the useful search interval, not the exact winner.

## PMU evidence

A profiler-gated run counted five warmups plus 51 calls, with unique scratch,
prewarmed A, and one distinct cold 4 MiB B per call:

| Variant | Median ms | Cycles/call | Instructions/call | IPC | L1 refill lines/call | L2 refill lines/call | Memory-stall/cycles |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_ld1h` | 0.1788 | 595,452 | 2,442,882 | 4.103 | 11,312 | 65,700 | 0.720% |
| `kblock_packed_752` | 0.1611 | 538,371 | 2,465,530 | 4.580 | 5,733 | 65,999 | 0.508% |
| `kblock_packed_768` | 0.1622 | 539,705 | 2,465,530 | 4.568 | 5,487 | 65,995 | 0.493% |
| `kblock_packed_800` | **0.1606** | **532,880** | 2,465,524 | **4.627** | **5,345** | 65,961 | 0.496% |
| `kblock_packed_816` | 0.1659 | 553,254 | 2,465,528 | 4.456 | 5,985 | 65,930 | 0.489% |
| `kblock_packed_832` | 0.1657 | 552,252 | 2,461,091 | 4.456 | 5,824 | 65,890 | 0.448% |

The approximately 65.9K L2 refills are invariant and match the 65,536 cache
lines in the compulsory 4 MiB B stream. K-blocking therefore does not reduce
cold-B traffic. At Kc=800 it reduces L1 refills by about 53%, cuts cycles by
10.5%, and raises IPC from 4.10 to 4.63 while adding less than 1% instructions.
Kc=816 and Kc=832 then lose cycles despite similar refill totals, which is why
the footprint formula needs the tail/set-mapping calibration described above.
Event semantics are platform-specific, so refill counts are used
comparatively rather than converted into exact traffic bytes.

The W2 control confirms that the common assembly path itself is neutral when
there is only one K chunk. `Kc=256` uses two chunks and gives a smaller stable
gain:

| Variant | Process-median range | Median of medians | Median paired gain |
|---|---:|---:|---:|
| `baseline_ld1h` | 0.1798-0.1800 ms | 0.1799 ms | reference |
| `kblock_packed_256` | 0.1745-0.1749 ms | 0.1746 ms | 3.13% |
| `kblock_packed_512` | 0.1799-0.1804 ms | 0.1801 ms | -0.10% |

## Long-route M=2040 control

The isolated M12 result makes every 4 MiB B input cold. A long expert behaves
differently: one `M=2040` call consists of 170 consecutive M12 panels using the
same B. The benchmark's `--m 2040` mode models that sequence while keeping the
experimental GEMM bodies unchanged:

- host and binding: `AmazonC5192Cores`, NUMA node 0, CPU 48, one thread;
- shape: BF16 `M=2040,K=4096,N=512`, one 4 MiB W13 range;
- each outer call uses a distinct cold B, then reuses it for all 170 panels;
- the complete 15.94 MiB packed A is scanned before timing;
- the K-block path reuses one 24 KiB partial-C scratch across the 170 panels;
- five warmups and 31 samples in each of seven independent processes.

| Variant | Process-median range | Median of medians | GFLOP/s | Throughput gain |
|---|---:|---:|---:|---:|
| `baseline_ld1h` | 25.4653-25.6891 ms | 25.5168 ms | 335.32 | reference |
| `kblock_packed_800` | 25.2835-25.3371 ms | 25.3171 ms | 337.97 | **+0.79%** |

Kc=800 saves 0.1997 ms per range, a 0.78% latency reduction. The effective
per-panel time changes from 0.1501 to 0.1489 ms, much closer than the isolated
cold-panel times of 0.1779 and 0.1595 ms. Only the first of 170 panels sees the
compulsory cold-B condition responsible for most of the isolated 11.5% gain.

A three-process sweep confirmed that a larger Kc does not recover that gain:

| Kc | Median of process medians |
|---:|---:|
| 512 | 25.3588 ms |
| 704 | 25.3290 ms |
| 752 | 25.3460 ms |
| 768 | 25.3728 ms |
| 800 | **25.3211 ms** |
| 896 | 25.5351 ms |
| 960 | 25.6465 ms |
| 1024 | 25.7652 ms |
| 1152 | 25.6334 ms |
| 1280 | 25.8726 ms |

Kc=704-800 is effectively the useful long-route plateau; Kc=800 remains the
best fixed choice among the measured candidates. This control covers the pure
single-thread BF16 GEMM body, not fused SiLU, N-split concurrency, or expert
end-to-end time.

## Reproduction

```bash
make -C optimizations/fused_moe_sve/benchmarks check-m12-kblock

for variant in \
  baseline_ld1h \
  kblock_packed_704 kblock_packed_720 kblock_packed_736 \
  kblock_packed_752 kblock_packed_768 kblock_packed_784 \
  kblock_packed_800 kblock_packed_816 kblock_packed_832 \
  kblock_packed_848 kblock_packed_864 kblock_packed_880 \
  kblock_packed_896; do
  python3 optimizations/fused_moe_sve/benchmarks/run_m12_streaming_b.py \
    --no-build --shape w13 --variants "${variant}" \
    --warmup 5 --runs 51 --repeat 5 --cpu 48 --numa-node 0 \
    --cold-tail-mib 192 --unique-scratch --prewarm-a
done

for variant in baseline_ld1h kblock_packed_800; do
  python3 optimizations/fused_moe_sve/benchmarks/run_m12_streaming_b.py \
    --no-build --shape w13 --m 2040 --variants "${variant}" \
    --warmup 5 --runs 31 --repeat 7 --cpu 48 --numa-node 0 \
    --cold-tail-mib 192 --unique-scratch --prewarm-a
done
```

Do not put all Kc candidates in one `--variants` list when measuring absolute
Kc performance. Multi-variant mode remains useful for correctness and paired
instruction-policy experiments, but it is not the fixed-layout steady state
used for this Kc decision.

## Decision

The assembly K-block implementation validates the A-residency hypothesis, but
only when packed B follows the interchanged loop order. For this M12 shape,
the reusable A+B window should be kept near but below half of private L1D;
Kc=800 is the best measured fixed choice and improves isolated throughput by
about 11.5%. For a 2040-row sequence that reuses one B across 170 panels, the
same variant improves the pure single-thread GEMM body by only 0.79%.

The required Kchunk-major weight pack, fused-W13 final-chunk epilogues, and
expert-level split-W13 tests were integrated on 2026-07-18. Production uses a
common cross-Mr L1 selector rather than this M12-only optimum. The integration
and its high-concurrency applicability limit are recorded in
[`amazon_192c_8c_production_kc.md`](amazon_192c_8c_production_kc.md). The
Ntile-major K-block variants remain only as the negative layout control.
