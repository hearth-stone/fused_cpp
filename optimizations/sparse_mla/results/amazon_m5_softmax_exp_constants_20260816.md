# Amazon M5 online-softmax exp packed-constant experiment

## Result

The packed-lane Estrin implementation is faster only after enough independent
work is exposed. Its best isolated configuration is unroll 8, while the
broadcast-constant Horner baseline peaks at unroll 4.

At sequence length 2048, the median of three independent session medians was:

| Mode | Horner best | Packed/Estrin best | Latency change | Throughput change |
|---|---:|---:|---:|---:|
| Isolated | U4, 0.605697 ns/element | U8, 0.571827 ns/element | -5.59% | +5.92% |
| 16 live SVE accumulators | U4, 0.613147 ns/element | U8, 0.579812 ns/element | -5.44% | +5.75% |

This is not a universal instruction-for-instruction win. Under pressure, U4
packed/Estrin is 7.71% lower-throughput than U4 Horner. At U8, however,
packed/Estrin is 12.66% higher-throughput than U8 Horner because the shorter
dependency tree and packed constants materially reduce hot-loop spill traffic.

## Accuracy

Both implementations used the same degree-5 coefficients and clamp interval.
Against `std::exp` on 1,048,570 finite inputs in `[-87, 0]`, plus six separately
checked clamp/NaN/Inf cases:

| Metric | Horner broadcast | Packed-lane Estrin |
|---|---:|---:|
| Max absolute error | 2.32458115e-6 | 2.32458115e-6 |
| Max relative error | 3.28743069e-6 | 3.28739660e-6 |
| Max FP32 ULP distance | 39 | 39 |
| RMS error | 5.33102073e-8 | 5.33600694e-8 |
| Mean absolute error | 4.05537779e-9 | 4.15222133e-9 |
| BF16 mismatches versus rounded reference | 76 | 76 |
| BF16 max absolute difference | 0.001953125 | 0.001953125 |
| Special-case semantic mismatches | 0 / 6 | 0 / 6 |

The accuracy envelope is therefore effectively unchanged for BF16 P storage,
although the different evaluation order is not FP32 bit-exact.

## Register pressure and generated code

GCC initially scalarized the lane intrinsics into separate broadcast constants.
The final benchmark keeps three packed vectors live outside the loop: one for
range reduction and two for the six polynomial coefficients. Assembly was
accepted only after it contained indexed forms such as `fmul ... vN.s[1]`,
`fmls ... vN.s[0]`, and `fmla ... vN.s[0..2]`.

The pressure wrapper forces the production-like 16 SVE FP32 accumulators to be
live across the inlined softmax with input-only empty-assembly constraints.
The following counts exclude constant loads, output stores, scalar tails, and
the common one-time post-barrier save before the synthetic scalar reduction:

| Unroll | Variant | Z accumulator slots saved across softmax | Hot-loop Q spill load/store pairs per iteration | Fixed frame plus scalable area |
|---:|---|---:|---:|---:|
| 4 | Horner | 5 | 0 | 128 B + 8 VL |
| 4 | Packed/Estrin | 6 | 0 | 128 B + 8 VL |
| 8 | Horner | 8 | 14 | 352 B + 8 VL |
| 8 | Packed/Estrin | 8 | 5 | 224 B + 8 VL |

Whole-function Z store/load instruction counts, including the common final
reduction staging, were 13/13, 14/14, 16/16, and 16/16 in table order.

Best-to-best, packed/Estrin uses deeper unrolling and therefore more spills than
Horner U4. Matched at U8, it keeps the same eight cross-softmax Z spill slots but
reduces repeated Q spill traffic by 64.3%. This distinction prevents treating
the matched-U8 register-pressure improvement as a zero-spill result.

## Method

- Host: `AmazonM5192Cores`, AArch64 Neoverse V3, native SVE width 128 bits.
- Placement: NUMA node 1, physical CPU 96.
- Compiler: GCC 15.2, `-O3 -std=c++17 -mcpu=native`.
- Data: 2048 FP32 scores uniformly generated in `[-16, 0]`; BF16 output and
  FP32 sum are both consumed.
- Timing: 200 warmups, 2,000 iterations per sample, 21 samples per session,
  three independent sessions. Tables use the median session median.
- Reproducibility command:

```sh
cd /data/fused_cpp-sparse-attn/optimizations/sparse_mla/benchmarks
make clean all assembly
numactl --physcpubind=96 --membind=1 ./softmax_exp_constants \
  --length=2048 --warmup=200 --iterations=2000 --samples=21
```

## Decision

Keep this as a Lab result rather than changing production dispatch. The
isolated throughput and BF16 accuracy gates pass, but the original
"no additional vector spills" best-to-best gate does not: U8 packed/Estrin is
faster while using more stack traffic than the best U4 Horner implementation.
The next useful step is an integrated online-softmax tile prototype that can
reuse or retire score/P registers, instead of preserving all 16 synthetic GEMM
accumulators across a length-2048 loop.
