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

## Implication

The trough is specific to the fixed-Mr/full-N order that completes all N tiles
for one M12 A panel before starting the next panel. A fixed-Nr/full-M order is
the relevant follow-up: hold one 32 KiB B tile and process all M12 A panels
before advancing N. That alternative is not claimed faster in this report
until it is measured with the same cold-B protocol.
