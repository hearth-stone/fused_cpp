# Small-M assembly K-block experiment

## Scope

- Host: `AmazonC5192Cores`, NUMA node 0, CPU 48.
- ISA: 128-bit runtime SVE BF16 (`VL=16 bytes`, `n_tile=8`).
- Private L1D: 64 KiB, four-way set associative, 64-byte cache lines.
- Compiler: `g++ -O2 -march=armv8.6-a+sve+bf16+i8mm`.
- Shape: `K=4096,N=512`, with `M=8,4,2,1`.
- Packed B: 4 MiB; every invocation uses a distinct cold address followed by
  a guard and a scanned 192 MiB cold tail.
- Timing: five warmups and 51 samples in each of five independent processes.
  Each process executes one fixed height/Kc, uses unique partial-C scratch per
  invocation, and scans packed A immediately before the timed kernel.

This is a standalone experiment. It does not change production fused-MoE
packing, symbols, dispatch, or defaults.

## Implementation

`msmall_kblock_kernels.S` adds one dynamic-Kc assembly entry for each
production small-M path:

- M8 preserves the production 16-accumulator, two-stage BFMMLA pipeline.
- M4 preserves its eight-accumulator pipeline.
- M2 preserves its four-accumulator pipeline.
- M1 follows production and reuses the M2 compute body while predicating the
  second output row away.

The loop order changes from:

```text
Ntile -> full K -> BF16 store
```

to:

```text
Kchunk -> Ntile -> K4
```

Packed B is Kchunk-major/Ntile-minor, so the complete 4 MiB cold-B traversal
remains contiguous. Intermediate FP32 accumulators use an accumulator-native
scratch layout. Kc is read from an experiment-only field immediately after
`gemm_params_t`; no fixed assembly symbol is needed for each sampled Kc.

M8/M4/M2/M1 all matched the corresponding production BF16 kernel bit for bit.
The check covered Kc `64,88,128,248,256` with `K=256,N=64`, including
ordinary multiple chunks, non-divisible tails, an eight-deep tail, and a
single chunk. The real W2 shape `K=512,N=4096` also passed for Kc=256 and
Kc=512.

## Physical footprint

The logical-M formula used by the M12 experiment was:

```text
W_tile(Kc) = 2 * Kc * (M + n_tile)
```

It cannot use logical M directly for the smaller production kernels. Packed A
keeps a 64-byte stride for every K4 block. M8 loads the full line; M4 and
M2/M1 load only 32 or 16 bytes, but fetching any part occupies the complete
cache line. Therefore all four heights have:

```text
A_cache_per_K = 64 bytes / 4 = 16 bytes
B_per_K        = 2 * n_tile = 16 bytes
W_tile(Kc)     = 32 * Kc bytes
Kc_half_L1     = (64 KiB / 2) / 32 = 1024
```

This predicts a common capacity boundary near Kc=1024, not the logical-M
predictions 1024/1365/1638/1820.

## Timing results

The dense isolated sweeps give:

| Height | Baseline | Best measured region | Representative best | Gain | Kc=1024 |
|---|---:|---:|---:|---:|---:|
| M8 | 0.1448 ms, 231.7 GFLOP/s | Kc=896-960 | 0.1296 ms, 258.8 GFLOP/s at Kc=896 | +11.7% | 0.1344 ms, +8.0% |
| M4 | 0.1212 ms, 138.4 GFLOP/s | Kc=384-768 | 0.1130 ms, 148.5 GFLOP/s at Kc=768 | +7.3% | 0.1167 ms, +3.9% |
| M2 | 0.1184 ms, 70.9 GFLOP/s | Kc=16-704 | 0.1092 ms, 76.8 GFLOP/s at Kc=384/704 | +8.4% | 0.1154 ms, +2.5% |
| M1 | 0.1185 ms, 35.4 GFLOP/s | Kc=16-704 | 0.1092 ms, 38.4 GFLOP/s at Kc=448 | +8.5% | 0.1154 ms, +2.6% |

M8 has a real capacity-sensitive optimum. Four fixed packed-B cache colors
preserved the same ordering:

| B color | Baseline ms | Kc=896 ms | Kc=960 ms | Kc=1024 ms |
|---:|---:|---:|---:|---:|
| 0 | 0.1451 | 0.1296 | 0.1300 | 0.1343 |
| 1 | 0.1452 | 0.1300 | 0.1302 | 0.1344 |
| 2 | 0.1460 | 0.1295 | 0.1298 | 0.1343 |
| 3 | 0.1447 | 0.1297 | 0.1298 | 0.1344 |

Kc=896 and Kc=960 use 28 and 30 KiB A+B windows. Kc=1024 reaches
exactly 32 KiB and consistently regresses, so the half-L1 rule correctly
identifies the upper envelope. The 64/128-step oscillation within that envelope
comes from chunk stride, cache-set mapping, and K-tail shape; the formula does
not predict the exact aligned winner.

M4 has a wide plateau rather than a sharp capacity point. M2/M1 are flatter
still: even Kc=16, which creates 256 K chunks, is within measurement noise of
Kc=384-704. Extra partial-C traffic and loop instructions fit in otherwise
exposed memory-latency slots.

## PMU results

The controlled profiler counted five warmups plus 51 calls:

| Height/variant | Cycles | Instructions | IPC | L1 refill lines | L2 refill lines | Memory stall |
|---|---:|---:|---:|---:|---:|---:|
| M8 baseline | 484,339 | 1,849,545 | 3.819 | 16,020 | 65,749 | 0.683% |
| M8 Kc=896 | 432,089 | 1,862,834 | 4.311 | 4,311 | 65,931 | 0.508% |
| M8 Kc=960 | 434,230 | 1,862,833 | 4.290 | 4,558 | 65,919 | 0.517% |
| M8 Kc=1024 | 447,405 | 1,859,549 | 4.156 | 5,487 | 65,853 | 0.539% |
| M4 baseline | 405,055 | 1,191,501 | 2.942 | 37,894 | 65,702 | 4.151% |
| M4 Kc=384 | 379,777 | 1,212,955 | 3.194 | 19,138 | 66,004 | 1.575% |
| M4 Kc=768 | 378,030 | 1,202,310 | 3.180 | 50,471 | 65,877 | 9.372% |
| M2 baseline | 394,216 | 862,464 | 2.188 | 65,949 | 65,735 | 15.709% |
| M2 Kc=16 | 365,299 | 1,275,698 | 3.492 | 63,644 | 65,832 | 17.961% |
| M2 Kc=384 | 366,311 | 878,794 | 2.399 | 55,608 | 65,880 | 16.935% |
| M2 Kc=704 | 366,098 | 870,712 | 2.378 | 62,582 | 65,821 | 22.975% |
| M1 baseline | 397,786 | 862,389 | 2.168 | 65,757 | 65,716 | 15.974% |
| M1 Kc=32 | 365,068 | 1,068,257 | 2.926 | 63,827 | 65,826 | 21.605% |
| M1 Kc=448 | 365,110 | 877,099 | 2.402 | 53,515 | 65,878 | 14.735% |

L2 refill is invariant at approximately 65.8K lines and matches the compulsory
4 MiB B stream. K splitting does not reduce cold-B traffic.

M8 behaves like M12: K blocking primarily removes A-side L1 refills and raises
IPC. M2/M1 are different. Their baseline L1 refill count already approaches
the complete B stream and memory-stall share is much higher. Kc=16 adds 48% M2
instructions but does not increase time because those instructions execute in
otherwise exposed load-latency slots. Capacity alone therefore cannot select
their Kc.

## Rule validation

The first-order law is valid as:

1. A physical cache-footprint calculation, using cache lines actually fetched
   rather than logical M.
2. A capacity ceiling for Kc.
3. An approximate optimum only while A residency materially limits a
   compute-heavy kernel, as it does for M12 and M8.

It is not a universal exact optimizer. For M4/M2/M1, cold-B latency dominates
before the half-L1 boundary is reached. A practical selector needs:

```text
Kc <= Kc_capacity
T(Kc) = T_cold_B + T_compute + T_chunk_overhead(K / Kc)
        + T_A_refill(Kc, kernel_height)
```

On this machine, fixed choices that stay near the upper edge of each measured
plateau are Kc=896 for M8, Kc=768 for M4, and Kc=704 for M2/M1. The latter two
avoid needless chunk instructions while remaining on the latency-hidden
plateau. These are per-M experiment conclusions. Production uses one common
Kc layout for every Mr; its cross-machine selector is documented in
[`amazon_192c_8c_production_kc.md`](amazon_192c_8c_production_kc.md).

## Reproduction

```bash
make -C optimizations/fused_moe_sve/benchmarks check-msmall-kblock

python3 optimizations/fused_moe_sve/benchmarks/run_msmall_kblock.py \
  --m 8 --variants baseline,kblock \
  --k-blocks 704,736,768,800,832,864,896,928,960,992,1024,1056 \
  --warmup 5 --runs 51 --repeat 5 --cpu 48 --numa-node 0 \
  --cold-tail-mib 192 --unique-scratch --prewarm-a

for m in 4 2 1; do
  python3 optimizations/fused_moe_sve/benchmarks/run_msmall_kblock.py \
    --m "$m" --variants baseline,kblock \
    --k-blocks 16,32,64,128,192,256,384,512,640,704,768,832,1024 \
    --warmup 5 --runs 51 --repeat 5 --cpu 48 --numa-node 0 \
    --cold-tail-mib 192 --unique-scratch --prewarm-a
done
```
