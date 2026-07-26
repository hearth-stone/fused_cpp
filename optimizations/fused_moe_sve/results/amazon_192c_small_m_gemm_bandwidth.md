# Small-M cold-weight GEMM bandwidth

Date: 2026-07-24

## Question

This experiment isolates why the single-core SVE GEMM body does not always
reach the bandwidth of a packed-B-only stream when `M=1` and B is scanned once.
It compares the TP4 W13 shape (`K=4096, N=1024`, 8 MiB B) and W2 shape
(`K=512, N=4096`, 4 MiB B) without SiLU or other fused expert stages.

The conclusion is specific to the exact generated M1 body on
`AmazonC5192Cores`, CPU 48, NUMA node 0. It is not a STREAM bandwidth claim.

## Kernel accounting

At the compiled 128-bit SVE width, one N tile has eight columns. Each K4 step
in the physical M2 body contains:

```text
4 x LD1H B       = 64 B packed B
1 x LD1RQH A     = 16 B loaded from one 64 B packed-A line
4 x BFMMLA
```

The packed-A pointer advances by 64 B per K4. Therefore:

| Shape | N tiles | K4 steps/tile | B loads/call | A loads/call | BFMMLA/call | Packed-A address footprint |
|---|---:|---:|---:|---:|---:|---:|
| W13 `4096x1024` | 128 | 1024 | 524,288 | 131,072 | 524,288 | 64 KiB |
| W2 `512x4096` | 512 | 128 | 262,144 | 65,536 | 262,144 | 8 KiB |

CPU 48 reports a 64 KiB, four-way, 64-byte-line L1D. The W13 packed-A
address footprint therefore occupies the nominal capacity of the entire L1D;
the W2 footprint occupies one eighth of it.

## Method

The benchmark rotates 64 separately packed experts. The reuse distance is
512 MiB for W13 and 256 MiB for W2, both greater than the 96 MiB NUMA-local
L3. The measured L2 refill count is one packed-B cache line per call to within
0.2%, independently confirming that B is cold at this level.

The JIT emits benchmark-only controls with an otherwise identical loop:

| Probe | B load | A load | BFMMLA | Store |
|---|:---:|:---:|:---:|:---:|
| `control-only` | no | no | no | no |
| `a-only` | no | yes | no | no |
| `b-only` | yes | no | no | no |
| `ba-only` | yes | yes | no | no |
| `bfmmla-only` | yes | fixed zero operand | yes | no |
| `full-no-store` | yes | yes | yes | no |
| `full-with-store` | yes | yes | yes | yes |

`profile_pure_gemm_probe.sh` uses two native `SIGSTOP` boundaries. It attaches
perf after allocation, packing, output creation, and JIT generation, and
detaches after exactly 256 warmups plus 1024 measured calls but before pybind
return and Python statistics. The 200 ms attach delay was validated by three
control runs whose retired-instruction counts differed by at most four
instructions. Six events ran without multiplexing:

```text
cycles:u,instructions:u,r70:u,l1d_cache_refill:u,
l2d_cache_refill:u,stall_backend_mem:u
```

Raw `r70` is Arm `LD_SPEC`. Raw `r74` was tested but did not count BFMMLA on
this host and is not used. Every table entry below is the median of three
processes; each process reports the median of 1024 timed calls.

A final non-perf cross-check measured the ordinary production dispatch at
0.2343 ms / 35.805 GB/s and `full-with-store` at 0.2342 ms / 35.820 GB/s, a
0.04% difference. The probe call path therefore represents the production
GEMM body at the resolution of this experiment.

Example:

```bash
for mode in control-only a-only b-only ba-only bfmmla-only \
            full-no-store full-with-store; do
  optimizations/fused_moe_sve/benchmarks/profile_pure_gemm_probe.sh \
    "${mode}" "repeat_1" 4096 1024
done
```

## W13 result

| Probe | Time (ms) | B-equivalent GB/s | Instructions/call | L1D refill/call | L2D refill/call | Memory-stall/cycles |
|---|---:|---:|---:|---:|---:|---:|
| control only | 0.0404 | - | 526,366 | 0 | 0 | 0.01% |
| A only | 0.0404 | - | 657,438 | 508 | 0 | 0.01% |
| B only | 0.2064 | 40.63 | 1,050,654 | 126,024 | 131,136 | 44.20% |
| B+A only | 0.2293 | 36.59 | 1,181,726 | 162,088 | 131,464 | 38.73% |
| B+BFMMLA | 0.2169 | 38.68 | 1,575,198 | 117,056 | 131,103 | 21.81% |
| full, no store | 0.2334 | 35.94 | 1,706,014 | 121,132 | 131,227 | 14.80% |
| full with store | 0.2343 | 35.80 | 1,709,086 | 123,406 | 131,282 | 15.14% |

The appropriate W13 denominator is its own B-only rate, 40.63 GB/s. The full
GEMM reaches 88.1% of that rate. The older 40.04 GB/s marker happens to be
close, but it is not a universal single-core read ceiling.

The dynamic instruction controls exactly match the generated loop:

```text
B-only - control      = 524,288 instructions/call
A-only - control      = 131,072 instructions/call
BA-only - B-only      = 131,072 instructions/call
BFMMLA-only - B-only  = 524,544 instructions/call
```

The last value contains 524,288 BFMMLA plus 256 zero-initialization
instructions. The store epilogue adds exactly 3,072 instructions/call and
about 0.9 microseconds.

Adding A to the B-only stream increases L1D refills by 36,064 lines/call but
L2D refills by only 329 lines/call. Thus the repeated packed-A stream is
displaced from L1D but remains in L2. In the full kernel, L2D refills remain
within 0.1% of B-only and memory-stall cycles fall by 61.8%. The lower
B-equivalent bandwidth is therefore not caused by more lower-level traffic or
more exposed memory stalls.

## Fixed-A causal control

Two additional probes execute the same BA-only and full-no-store instruction
streams but source every `LD1RQH` from the same packed-A cache line. The normal
and fixed-A full binaries have the same size and differ by only six encoded
bytes: the three static `LD1RQH` sites use `[x13]` versus `[x0]`.

| Probe | Normal A | Fixed A | Change |
|---|---:|---:|---:|
| BA-only time | 0.2293 ms | 0.2083 ms | -21.0 us |
| BA-only L1D refills | 207.47 M | 160.33 M | -47.14 M |
| Full-no-store time | 0.2334 ms | 0.2169 ms | -16.5 us |
| Full-no-store instructions | 2.183697 B | 2.183697 B | 0 |

Fixed-A full-no-store equals BFMMLA-only at 0.2169 ms. Therefore the normal
W13 full-body gap over B-only decomposes as:

```text
packed-A address-stream cost: about 16.5 us
BFMMLA cost after overlap:    about 10.5 us
FP32 store epilogue:          about  0.9 us
total:                        about 27.9 us
```

This control proves the A-address contribution without naming an unmeasured
load queue, issue port, or prefetch mechanism.

## Fixed-B K sweep

Keeping B at 8 MiB while varying K changes only the packed-A footprint and the
inner/outer loop geometry. Each point is a 1024-call median.

| K | N | A footprint | BA-only minus B-only | Full minus BFMMLA-only | Full/B-only bandwidth |
|---:|---:|---:|---:|---:|---:|
| 512 | 8192 | 8 KiB | 2.9 us | 0.5 us | 94.9% |
| 1024 | 4096 | 16 KiB | 12.6 us | 8.1 us | 91.7% |
| 2048 | 2048 | 32 KiB | 15.6 us | 13.5 us | 89.9% |
| 4096 | 1024 | 64 KiB | 22.9 us | 16.5 us | 88.5% |
| 8192 | 512 | 128 KiB | 24.6 us | 16.7 us | 87.5% |

The 64 KiB point is where BA-only first adds a large L1D refill stream:
36,064 lines/call at 64 KiB and about 85,234 lines/call at 128 KiB. The timing
cost begins below that capacity boundary, so L1 capacity misses are not the
only component; the table establishes the empirical packed-A-footprint curve
without assigning the sub-capacity portion to an unmeasured hardware unit.

## W2 comparison

| Probe | Time (ms) | B-equivalent GB/s | Instructions/call | L1D refill/call | L2D refill/call | Memory-stall/cycles |
|---|---:|---:|---:|---:|---:|---:|
| B only | 0.0985 | 42.58 | 531,365 | 61,539 | 65,588 | 42.89% |
| B+A only | 0.1008 | 41.61 | 596,901 | 59,517 | 65,614 | 30.55% |
| B+BFMMLA | 0.1042 | 40.27 | 794,533 | 52,411 | 65,569 | 18.81% |
| full, no store | 0.1046 | 40.12 | 859,045 | 52,020 | 65,575 | 17.28% |
| full with store | 0.1045 | 40.14 | 871,333 | 52,143 | 65,580 | 16.41% |

W2 reaches 94.3% of its 42.58 GB/s B-only rate. Its 8 KiB packed-A footprint
adds no refill stream, and adding A to the real BFMMLA body is within about
0.4 microseconds. It is therefore incorrect to group W13 and W2 under one
claim that both fail to fill a 40.04 GB/s ceiling.

## Supported conclusions

1. W13 M1 full GEMM is 35.80 GB/s versus a same-traversal B-only ceiling of
   40.63 GB/s, or 88.1%.
2. The deficit is not additional L2/memory traffic: L2D refills are unchanged
   and memory-stall cycles are lower in the full body.
3. The W13 packed-A stream spans the full 64 KiB L1D. Holding its address fixed
   removes the entire 16.5 us A-path cost without changing instruction count.
4. BFMMLA adds the remaining approximately 10.5 us while hiding more than half
   of the B-only memory-stall cycles.
5. W2 behaves differently because its packed-A footprint is only 8 KiB; it
   reaches 94.3% of its own B-only rate.

These results cover single-core M1 and cold B. They do not directly establish
multicore DRAM saturation, M3-M12 behavior, or a production optimization. The
fixed-A variants are causal probes and intentionally do not preserve GEMM
numerics.
