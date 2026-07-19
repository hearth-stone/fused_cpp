# M12 streaming-B load experiment

## Scope

- Host: `AmazonC5192Cores`, NUMA node 0, CPU 48.
- ISA: 128-bit runtime SVE BF16 (`VL=16 bytes`, `n_tile=8`).
- Compiler: `g++ -O2 -march=armv8.6-a+sve+bf16+i8mm`.
- Shapes: W13 range `M=12,K=4096,N=512` and W2
  `M=12,K=512,N=4096`; packed B is 4 MiB in both cases.
- Timing: five warmups and 31 samples for the sweep; five warmups and 51
  samples in each of seven independent processes for final paired tests.

The standalone baseline and candidates have the same M12 BFMMLA instruction
sequence and BF16 store epilogue. Only the B-load/prefetch sequence differs.
The production `moe_sve_w2_packed_bf16_m12` symbol is a correctness reference,
not the timed baseline. All eleven standalone variants matched it bit for bit on
both real shapes and on the small `K=N=64` build check.

Each invocation uses a different packed-B address; copy values rotate across
32 finite BF16 values. Used copies are followed by a 192 MiB cold tail scanned
before timing. Every weight copy also has a 4 KiB guard, so a software prefetch
issued near the final K iteration cannot warm the next invocation's B.

## Initial candidate sweep

Representative process medians:

| B-load policy | W13 ms | W13 gain | W2 ms | W2 gain |
|---|---:|---:|---:|---:|
| standalone `LD1H` | 0.1783 | reference | 0.1797 | reference |
| `PLDL1STRM`, 256 B | 0.1780 | +0.15% | 0.1726 | +4.15%* |
| `PLDL1STRM`, 512 B | 0.1733 | +2.91% | 0.2785 | -35.48% |
| `PLDL1STRM`, 1024 B | 0.1721 | +3.61% | 0.2404 | -25.25% |
| `PLDL1STRM`, 2048 B | 0.1625 | +9.71% | 0.2254 | -20.28% |
| `PLDL2STRM`, 512 B | 0.1818 | -1.93% | 0.1666 | +7.86%* |
| `PLDL2STRM`, 1024 B | 0.1809 | -1.40% | 0.1658 | +8.41%* |
| `PLDL2STRM`, 2048 B | 0.1823 | -2.18% | 0.1661 | +8.18%* |
| `LDNT1H` | 0.1797 | -0.77% | 0.1689 | +6.43%* |

`*` The nine-way W2 sweep disturbed the ordinary-load baseline and made these
percentages look too large. The isolated two-way result below is authoritative.

W13 benefits monotonically over the tested L1-prefetch distances. W2 is
damaged by all but the shortest L1 distance. This is consistent with its
packed-A panel being only `12 * 512 * 2 = 12 KiB`: streaming B through the
64 KiB L1D can displace reusable A. The W13 packed-A panel is
`12 * 4096 * 2 = 96 KiB`, already larger than L1, and can use the latency
overlap without sacrificing an L1-resident A panel. Cache sizes read from
CPU48 sysfs were 64 KiB L1D, 2 MiB private L2, and 96 MiB shared L3; no PMU
counter was collected for this experiment, so the displacement mechanism is
an inference from the size boundary and timing response.

The initial sweep issued two adjacent cache-line hints per K4 body under an
incorrect 256-bit-VL assumption. The runtime VL is 128 bit, so four SVE B loads
consume one 64-byte line per K4. Adjacent K4 iterations therefore repeated one
of the two hints. Follow-up variants issue exactly one hint per consumed line.

## Isolated paired results

The paired mode alternates only the standalone baseline and one candidate.
Every sample still uses a distinct cold weight.

| Shape and candidate | Baseline median range | Candidate median range | Per-process gain range | Median gain |
|---|---:|---:|---:|---:|
| W13, one `PLDL1STRM` 2048 B ahead | 0.1764-0.1779 ms | 0.1610-0.1616 ms | 9.57-10.35% | **9.99%** |
| W2, one `PLDL2STRM` 1024 B ahead | 0.1667-0.1671 ms | 0.1653-0.1659 ms | 0.48-0.93% | **0.73%** |

In a separate three/five-way control, the redundant second hint reduced W13's
median gain from 10.01% to 9.15%. W2 L1 prefetch remained about 21.04% slower
with one hint and 23.83% slower with two, so duplicate hints were not the main
reason that L1 prefetch failed on W2.

Commands:

```bash
for rep in 1 2 3 4 5 6 7; do
  numactl --cpunodebind=0 --membind=0 taskset -c 48 \
    optimizations/fused_moe_sve/benchmarks/bench_m12_streaming_b \
    --shape w13 --variants baseline_ld1h,pldl1strm_2048_x1 \
    --warmup 5 --runs 51 --cpu 48 --cold-tail-mib 192
done

for rep in 1 2 3 4 5 6 7; do
  numactl --cpunodebind=0 --membind=0 taskset -c 48 \
    optimizations/fused_moe_sve/benchmarks/bench_m12_streaming_b \
    --shape w2 --variants baseline_ld1h,pldl2strm_1024_x1 \
    --warmup 5 --runs 51 --cpu 48 --cold-tail-mib 192
done
```

## Decision

The large-K W13-like M12 path has a reproducible software-prefetch opportunity.
The one-hint 2 KiB L1 streaming prefetch raises its effective B rate from
roughly 23.6 GB/s to 26.0 GB/s without changing arithmetic or output. W2's 0.73%
median gain is too small to justify a production specialization by itself, and
`LDNT1H` is not useful here.

This remains a standalone experiment. Before production integration, the W13
prefetch body must be applied to the fused SiLU/packC entrypoint and measured
inside the complete split-W13 expert; dispatch should be gated by a large-K or
packed-A-footprint condition rather than by `M=12` alone.
