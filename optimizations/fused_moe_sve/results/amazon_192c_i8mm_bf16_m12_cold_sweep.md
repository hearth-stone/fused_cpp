# Upstream i8mm BF16 M12 cold-B sweep and M=24 cache-state analysis

Date: 2026-07-27

## Scope

This report characterizes the upstream
`refs/i8gemm/lib/bf16gemm_sve.S::bf16gemm_k_nld_f_m12` kernel. It does
not measure the fused MoE epilogue and does not change production dispatch.

Machine and method:

- Host: `AmazonC5192Cores`, NUMA0 CPU 48.
- Cache at CPU 48: 64 KiB 4-way L1D, 2 MiB 8-way private L2,
  96 MiB 16-way LLC, 64-byte cache lines.
- Shape: BF16 `K=2048`, FP32 output, packed A and packed B.
- Compiler: `cc -O2 -Wall -Wextra -mcpu=native`.
- Each timed GEMM uses a distinct packed-B allocation. Before timing begins,
  the benchmark scans a 192 MiB tail to evict initialization traffic.
- Each table point is the median of five independent process medians; each
  process uses 51 timed samples.
- For `M>12`, panels within one logical GEMM intentionally reuse the same B.
  This is the fixed-Mr/full-N execution order being studied.

The reproducible benchmark is:

```text
optimizations/fused_moe_sve/benchmarks/bench_i8mm_bf16_m12_cold.c
```

Build:

```bash
cc -O2 -Wall -Wextra -Wshadow -Wconversion -Wstrict-prototypes \
  -Wmissing-prototypes -Wundef -mcpu=native \
  -Irefs/i8gemm/lib \
  optimizations/fused_moe_sve/benchmarks/bench_i8mm_bf16_m12_cold.c \
  refs/i8gemm/lib/bf16gemm_sve.S \
  -o /tmp/bench_i8mm_bf16_m12_cold -lm
```

## Cold-B M/N sweep

Sweep dimensions:

- `M={12,24,48,96,192,384}`
- `N={64,128,192,256,320,384,448,512}`

Representative point command:

```bash
taskset -c 48 /tmp/bench_i8mm_bf16_m12_cold \
  point 24 2048 128 51 48
```

### Median time in microseconds

| M \ N | 64 | 128 | 192 | 256 | 320 | 384 | 448 | 512 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 11.9 | 22.7 | 34.7 | 47.5 | 58.8 | 71.7 | 83.3 | 93.5 |
| 24 | 23.5 | 66.5 | 94.1 | 115.7 | 135.0 | 159.3 | 182.6 | 203.5 |
| 48 | 42.4 | 107.2 | 143.6 | 182.7 | 228.3 | 284.0 | 334.6 | 360.6 |
| 96 | 79.5 | 175.1 | 251.4 | 335.1 | 427.8 | 528.9 | 618.6 | 664.5 |
| 192 | 156.1 | 320.3 | 475.9 | 641.7 | 817.0 | 1007.9 | 1202.6 | 1272.3 |
| 384 | 314.9 | 625.7 | 932.4 | 1250.9 | 1586.2 | 1937.8 | 2330.7 | 2578.7 |

### Throughput in GFLOP/s

| M \ N | 64 | 128 | 192 | 256 | 320 | 384 | 448 | 512 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 264.0 | 277.7 | 272.2 | 264.6 | 267.6 | 263.4 | 264.4 | 269.1 |
| 24 | 268.2 | 189.3 | 200.7 | 217.5 | 233.0 | 237.0 | 241.3 | 247.4 |
| 48 | 296.5 | 234.7 | 262.9 | 275.6 | 275.6 | 265.8 | 263.3 | 279.2 |
| 96 | 316.7 | 287.5 | 300.4 | 300.4 | 294.1 | 285.5 | 284.8 | 303.0 |
| 192 | 322.5 | 314.3 | 317.3 | 313.7 | 308.0 | 299.6 | 293.0 | 316.5 |
| 384 | 319.7 | 321.7 | 323.9 | 321.9 | 317.3 | 311.7 | 302.3 | 312.3 |

Observed regimes:

- `M=12` remains near 264-278 GFLOP/s: every call performs one cold-B pass.
- `M>=96` generally reaches 285-324 GFLOP/s as B reuse is amortized.
- `M=24,N>=128` is a repeatable trough. It is not a single-run outlier.

## M=24 panel decomposition

The `panels` mode times each M12 call separately. B modes are:

- `0`: every panel receives a different cold B.
- `1`: panels directly reuse the first panel's B.
- `2`: reuse B after explicitly reading every cache line.
- `3`: reuse B after scanning 4 MiB to evict the prior private-cache state.

Example:

```bash
taskset -c 48 /tmp/bench_i8mm_bf16_m12_cold \
  panels 2 2048 128 1 101 48
```

At `N=128`:

| Condition | Panel 0 | Panel 1 | Panel 1 GFLOP/s |
| --- | ---: | ---: | ---: |
| Directly reuse B | 26.8 us | 40.6 us | 155 |
| Use another cold B | 23.0 us | 23.0 us | 274 |
| Fully prewarm reused B | about 26.5 us | about 27 us | about 233 |
| Evict reused B first | about 23.5 us | about 25 us | about 250 |

At `N=256`, direct reuse makes panel 1 approximately 63-65 us, while a new
cold B takes approximately 45-46 us. Prewarming reduces panel 1 to
approximately 43 us; eviction reduces it to approximately 49 us.

Keeping the same packed-A panel for both calls reduces the `N=128` panel-1
time from approximately 40.6 us to 34.9 us. Switching to the next 48 KiB
packed-A panel therefore contributes 5-7 us, but does not explain most of the
trough.

## B cache-state probe

The cache mode compares a cache-line scan of untouched B, B immediately after
one M12 GEMM, and B after a complete scalar prewarm:

```bash
taskset -c 48 /tmp/bench_i8mm_bf16_m12_cold \
  cache 2048 128 101 48
```

| N | B size | Cold scan | Scan after GEMM | Fully hot scan |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 256 KiB | about 53 GB/s | about 102 GB/s | about 102 GB/s |
| 128 | 512 KiB | about 42 GB/s | about 56 GB/s | about 69 GB/s |
| 256 | 1 MiB | about 40 GB/s | about 38.5 GB/s | about 55 GB/s |
| 512 | 2 MiB | about 39 GB/s | about 42 GB/s | about 54 GB/s |

The first GEMM leaves all 256 KiB reusable at `N=64`. At `N>=128`, the state
left by the GEMM is neither fully hot nor equivalent to a clean cold stream.

## Why M=24 is the trough

For `K=2048` on a 128-bit SVE machine:

- One packed M12 A panel is `12 * 2048 * 2 = 48 KiB`.
- One eight-column B tile is `2048 * 8 * 2 = 32 KiB`.
- The instantaneous L1 footprint is 80 KiB, above the 64 KiB L1D capacity.
- With contiguous mapping, A consumes roughly three of four L1 ways and the
  current B tile roughly two ways, so L1 conflict replacement is unavoidable.

The kernel loops over N tiles and scans the full 48 KiB A panel for every tile.
The first M12 call streams B once. Above a 256 KiB B footprint, that pass does
not leave the complete B in the same state as an explicit prewarm. The second
M12 call then sees a mixed hit/miss stream. Experimentally, that mixed state is
slower than both endpoints:

- replacing it with a completely new cold B restores performance;
- explicitly making B fully hot restores performance;
- evicting the mixed state before reuse also restores most performance.

The exact split between hardware prefetch disruption, miss-queue scheduling,
and cache replacement cannot be uniquely identified with the available PMU
events. The causal claim is limited to the experimentally controlled cache
state.

Four-panel measurements explain why larger M recovers. At `N=128`, a typical
sequence is:

| M12 panel | Time |
| ---: | ---: |
| 0 | about 26.5 us |
| 1 | 37-40 us |
| 2 | 24-26 us |
| 3 | about 18.7 us |

`M=24` contains exactly the cold first pass and the pathological transition
pass, then stops. `M>=48` includes later passes after the cache state has
converged and amortizes the transition.

## Fixed-Nr/full-M validation

The follow-up compared the original fixed-Mr/full-N order with upstream
`bf16gemm_k_nld_f_nr_fullm`. Fixed-Nr holds one eight-column, 32 KiB B tile
and processes every M12 A panel before advancing N. Both paths use distinct
cold B copies, alternate measurement order, and produce bit-exact FP32 output.

```bash
taskset -c 48 /tmp/bench_i8mm_bf16_m12_cold \
  compare 24 2048 128 51 48
```

Each entry below is the median of five independent process medians.

### M=24

| N | Fixed-Mr us | Fixed-Nr us | Fixed-Nr gain | Fixed-Nr GFLOP/s |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 22.891 | 28.024 | -18.3% | 224.5 |
| 128 | 65.394 | 57.250 | +14.2% | 219.8 |
| 192 | 95.050 | 86.563 | +9.8% | 218.0 |
| 256 | 112.795 | 116.478 | -3.2% | 216.1 |
| 320 | 132.376 | 143.427 | -7.7% | 219.3 |
| 384 | 157.795 | 171.020 | -7.7% | 220.7 |
| 448 | 179.400 | 198.849 | -9.8% | 221.5 |
| 512 | 199.161 | 226.482 | -12.1% | 222.2 |

### M=48

| N | Fixed-Mr us | Fixed-Nr us | Fixed-Nr gain | Fixed-Nr GFLOP/s |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 41.501 | 46.722 | -11.2% | 269.3 |
| 128 | 107.506 | 93.677 | +14.8% | 268.6 |
| 192 | 152.477 | 141.625 | +7.7% | 266.5 |
| 256 | 186.213 | 189.040 | -1.5% | 266.3 |
| 320 | 231.122 | 234.267 | -1.3% | 268.6 |
| 384 | 279.684 | 279.921 | -0.1% | 269.7 |
| 448 | 322.512 | 326.070 | -1.1% | 270.1 |
| 512 | 348.625 | 372.592 | -6.4% | 270.2 |

Fixed-Nr removes the pathological complete-B second traversal and recovers
9.8-14.8% at `N=128/192`. It does not eliminate the M=24 throughput trough:
its M=24 throughput remains approximately 216-224 GFLOP/s. The upstream
full-M body is invoked once per eight-column N tile, so M=24 amortizes its
entry, pointer setup, and store control over only two M12 blocks. M=48
amortizes the same work over four blocks and sustains approximately
266-270 GFLOP/s.

At `N>=256`, the original fixed-Mr path has already amortized the one-time
mixed-cache transition over a longer N traversal and eventually reaches
approximately 240-256 GFLOP/s for M=24. Fixed-Nr remains near its
approximately 220 GFLOP/s M=24 control ceiling and therefore loses. At
`N=64`, the 256 KiB B matrix is fully reusable after the first panel, so
fixed-Mr is already the correct order.

## 16/32-column N-group hybrid

The hybrid keeps the upstream M12 computation body unchanged. Its outer loop
selects a 16- or 32-column B group, then runs every M12 A panel against that
group before advancing N. This differs from fixed-Nr/full-M because each M12
call amortizes its entry over two or four N tiles rather than one.

The benchmark rotates the four strategies' execution order, assigns each
timed invocation a distinct cold B copy, and verifies all outputs bit-exactly:

```bash
taskset -c 48 /tmp/bench_i8mm_bf16_m12_cold \
  groups 24 2048 128 51 48
```

Each entry is the median of five independent process medians.

### M=24 hybrid results

| N | Fixed-Mr GFLOP/s | Fixed-Nr GFLOP/s | Group16 GFLOP/s | Group32 GFLOP/s | Group32 vs fixed-Mr |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 274.8 | 223.7 | 249.5 | 266.6 | -3.0% |
| 128 | 193.2 | 214.8 | 244.8 | 264.9 | +37.1% |
| 192 | 197.9 | 217.6 | 240.4 | 261.5 | +32.1% |
| 256 | 228.8 | 216.0 | 241.7 | 260.2 | +13.7% |
| 320 | 241.6 | 221.7 | 245.5 | 261.4 | +8.2% |
| 384 | 249.4 | 222.5 | 246.2 | 259.1 | +3.9% |
| 448 | 250.0 | 222.1 | 246.2 | 261.9 | +4.8% |
| 512 | 254.5 | 220.4 | 242.8 | 258.9 | +1.7% |

Group32 removes the N-dependent M=24 trough: its throughput remains within
approximately 259-267 GFLOP/s over the entire sweep. In particular, the
`N=128` rate is only 0.7% below its `N=64` rate, whereas fixed-Mr falls by
29.7%. Group16 also avoids the severe transition but sustains only
approximately 240-250 GFLOP/s because it enters the M12 body twice as often.

### M=48 control results

| N | Fixed-Mr GFLOP/s | Group16 GFLOP/s | Group32 GFLOP/s | Group32 vs fixed-Mr |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 301.1 | 283.5 | 296.4 | -1.5% |
| 128 | 232.2 | 278.8 | 292.5 | +26.0% |
| 192 | 247.3 | 281.1 | 292.4 | +18.2% |
| 256 | 269.2 | 280.3 | 291.2 | +8.2% |
| 320 | 270.0 | 285.3 | 294.1 | +8.9% |
| 384 | 277.6 | 285.4 | 295.6 | +6.5% |
| 448 | 286.8 | 285.6 | 293.1 | +2.2% |
| 512 | 293.7 | 284.6 | 292.9 | -0.3% |

The M48 control shows the same cache-transition removal. Group32 remains near
291-296 GFLOP/s, while fixed-Mr catches up only after its one-time transition
has been amortized across a larger N traversal.

### Eight-core cross-platform check

The same `M=24, K=2048, N=128` comparison was repeated on
`AmazonECS8Cores`, pinned to CPU0. The median of five independent 51-sample
processes was 53.947 us for fixed-Mr and 57.060 us for group32, so group32
regressed by 5.5%. All outputs remained bit-exact.

The 32-column optimum is therefore machine-specific rather than a universal
kernel ordering rule. It must be selected through machine calibration or an
architecture-specific policy.

## Decision

The 32-column N-group is the best measured traversal for the cold-B
transition region and eliminates the M=24 low point without changing the
microkernel or packed layout. The original fixed-Mr remains slightly better
when the complete B matrix is already small enough to reuse (`N=64`) and is
effectively tied at `N=512`. Group16 is not a preferred operating point.

This is an experimental traversal result, not yet a production-path change.
Before integrating it into the fused W13/W2 driver, expose N-group as a
calibrated machine policy and verify that its additional outer-loop
boundaries do not interfere with split-W13 staging or expert-level N
splitting.
