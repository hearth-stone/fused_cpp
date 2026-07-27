# Amazon C8i8 AVX-512 K-loop scheduling and packed-B prefetch

Date: 2026-07-26

## Outcome

The AVX-512 Xbyak W13 and W2 kernels now have cache-key-isolated K-loop
variants:

- `baseline`: the previous schedule, including the established W13 T0
  prefetch;
- `no_prefetch`: the old single-pair schedule without software prefetch;
- `unroll2`: consume two adjacent BF16 K-pairs per loop;
- `unroll2_t0`: two-pair scheduling plus L1 packed-B prefetch;
- `unroll2_t1`: two-pair scheduling plus L2 packed-B prefetch.

Two-pair scheduling loads the next pair before completing the current pair,
halves loop-control frequency, and uses a second accumulation bank when the
exact-M register budget permits. W13 and W2 retain their existing VNNI2/N32
weights, output epilogues, Bulk-MN loops, small-M multi-N kernels, and exact
tails. Odd K-pair counts use a generated one-pair tail.

The selected C8i automatic policy is stage- and dimension-aware:

- W13 keeps `baseline` below M96, H256, or F256. It uses `unroll2_t0` for
  M96--511 and `unroll2` at M>=512, where repeated M panels already keep B
  hot.
- W2 uses `unroll2_t0` at M>=12, F>=128, and H>=256.
- unknown CPU profiles, invalid dimensions, and expert teams wider than eight
  workers retain `baseline`.

This policy improved H4096/F512 M256 by 0.74% on one core and 1.50%--1.81% on
2--8 cores. At M2048 it improved median latency by 2.78% on one core and 3.56%
on eight cores. H4096/F2048 M256 improved 1.41%/1.53% at 1T/8T. The held-out
H64/F2048 shape correctly remained on the old kernels.

## Machine and method

- host: `AmazonC8i8Cores`
- CPU: Intel Xeon 6975P-C, family 6/model 173/stepping 1
- topology: 8 physical vCPUs, one thread per core
- caches reported by `lscpu`: 48 KiB L1d and 2 MiB L2 per core, 480 MiB
  shared L3
- source baseline: local commit `183204e` plus this uncommitted change
- affinity: CPU 0 for one-thread runs; CPUs 0--7 with resident-worker pinning
  for cooperative runs
- frequency caveat: the KVM guest does not expose a fixed-frequency control,
  so variants were rotated in one process and compared by median, mean, and
  same-iteration paired ratios

`benchmarks/bench_avx512_k_loop.py` prepares one set of weights and inputs,
reuses output tensors, validates every variant against PyTorch, warms every
JIT key, and rotates timed order. Packing, allocation, reference calculation,
and JIT generation are outside timed regions.

Representative end-to-end command:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src \
FUSED_CPP_MOE_X86_ISA=avx512 \
FUSED_CPP_MOE_AVX512_IMPL=jit \
FUSED_CPP_MOE_X86_POLICY_PROFILE=c8i \
FUSED_CPP_MOE_PIN_THREADS=1 \
FUSED_CPP_MOE_PIN_THREAD_CPUS=0,1,2,3,4,5,6,7 \
taskset -c 0-7 .venv/bin/python benchmarks/bench_avx512_k_loop.py \
  --hidden 4096 --intermediate 512 --routes 4,12,48,96,256 \
  --threads 8 --variants baseline,unroll2,unroll2_t0,unroll2_t1,auto \
  --warmup 12 --runs 51
```

The standalone W13 and W2 binaries use already-packed tensors and time only
the generated kernel. Hardware counters were collected with `sudo -n perf`
because the guest has `perf_event_paranoid=4`. Each counter result is the mean
of three repetitions with 1,001 W13 or 2,001 W2 measured kernel calls.

## Generated schedule

The original inner loop consumes one packed K-pair and branches once per pair.
The new common body:

1. loads the current and next B pair into separate ZMM operands;
2. issues their A broadcasts and `VDPBF16PS` groups in an interleaved order;
3. advances A/B by two pairs and branches once;
4. executes a one-pair tail when the padded reduction contains an odd number
   of pairs.

For small exact-M kernels with enough spare ZMM registers, the two pairs feed
independent accumulator banks that are reduced before the unchanged
epilogue. M12 and wider multi-N kernels use load/compute overlap without
forcing accumulator spills. Four-pair unrolling was not generated for the
common M12 kernels because their accumulator set leaves insufficient ZMM
capacity for two more B banks without spills.

The explicit T0 distance is eight K-pairs and the T1 distance is sixteen
K-pairs. Like the legacy W13 hint, they use x86's non-faulting prefetch
semantics and may point beyond the logical end of the current packed panel.
The selected schedule is part of `KernelKey`, so forced A/B modes cannot alias
an already-generated function.

## Correctness

The final x86 test command is:

```text
PYTHONPATH=src FUSED_CPP_MOE_X86_ISA=avx512 \
  .venv/bin/python -m pytest -q \
  tests/test_moe_backend_dispatch.py tests/test_moe_avx512_bf16.py
```

Result: `289 passed, 3 skipped, 1 warning in 0.98s`.

Coverage includes all five explicit variants, C8i automatic M96/M513
boundaries, odd W13 and W2 K-pair counts, M and N tails, direct BF16, route
FP32, weighted-direct BF16, forced Bulk-MN, cache windows, and cooperative
four-worker N split. Candidate checksums were identical to baseline in the
standalone kernels; end-to-end maximum absolute error versus PyTorch was
`1.19e-7` for H4096/F512 and `2.38e-7` for H4096/F2048.

## Standalone kernels

### W13, M12/F512

GFLOP/s below includes gate/up GEMM and uses the same generated SiLU/multiply
epilogue for every row.

| K | baseline | no prefetch | unroll2 | unroll2 T0 | unroll2 T1 |
|---:|---:|---:|---:|---:|---:|
| 256 | 117.39 | 117.29 | 117.45 | 117.32 | 117.51 |
| 512 | 120.79 | 120.82 | 120.84 | 120.87 | 120.85 |
| 1024 | 120.52 | 119.22 | 121.94 | 122.35 | 122.46 |
| 4096 | 120.49 | 115.41 | 119.95 | 120.87 | 120.73 |

The isolated long-K panel demonstrates why W13 cannot simply drop its old
prefetch: at K4096, `no_prefetch` lost about 4.4%. At M4/K4096, plain
`unroll2` also fell from 107.12 to 97.43 GFLOP/s because the small-M schedule
lost prefetch coverage without enough panel reuse. These negative controls
are why automatic W13 selection starts at M96.

### W2, M12/N4096

| K | baseline GFLOP/s | unroll2 GFLOP/s | unroll2 T0 GFLOP/s | T0 gain |
|---:|---:|---:|---:|---:|
| 128 | 116.99 | 119.30 | 122.41 | +4.6% |
| 512 | 118.17 | 121.69 | 123.24 | +4.3% |
| 1024 | 115.21 | 119.76 | 120.87 | +4.9% |

W2 benefits more consistently because its many output blocks amortize the
schedule and repeatedly expose a cold packed-B stream.

## Hardware-counter evidence

### W2 M12/K512/N4096

| counter | baseline | unroll2 | unroll2 T0 | T0 change |
|---|---:|---:|---:|---:|
| cycles | 3,531,450,804 | 3,429,028,601 | 3,383,705,545 | -4.18% |
| instructions | 3,142,058,065 | 3,041,588,383 | 3,108,831,454 | -1.06% |
| branches | 107,062,305 | 73,744,977 | 73,753,637 | -31.1% |
| L1D load misses | 5,937,336 | 4,261,597 | 3,186,728 | -46.3% |

Branch count, cycles, and L1D misses were stable. T0 adds explicit prefetch
load events, but it still reduces total cycles. T1 reached similar wall time
while leaving 4.33 million L1D misses, versus 3.19 million for T0, so it is
retained only as an experiment.

### W13 M12/K4096/F512

| counter | baseline | no prefetch | unroll2 T0 | T0 change |
|---|---:|---:|---:|---:|
| cycles | 3,500,311,306 | 3,649,471,460 | 3,489,454,217 | -0.31% |
| instructions | 3,443,986,066 | 3,376,450,418 | 3,342,267,130 | -2.95% |
| branches | 131,195,340 | 131,212,056 | 97,351,628 | -25.8% |
| L1D load misses | 3,516,185 | 5,395,452 | 3,309,460 | -5.9% |
| median kernel time | 0.8529 ms | 0.8893 ms | 0.8504 ms | -0.29% |

This confirms that W13 T0 mainly protects load latency; two-pair loop control
alone is insufficient for an isolated long-K panel. At large expert M, B
panel reuse changes that balance and makes explicit prefetch unnecessary.

## End-to-end automatic policy

The following values are median speedup over forced `baseline`.

### H4096/F512

| threads | M12 | M48 | M96 | M256 |
|---:|---:|---:|---:|---:|
| 1 | 1.0004x | 1.0007x | 1.0047x | 1.0074x |
| 2 | 0.9999x | 1.0022x | 1.0125x | 1.0150x |
| 4 | 1.0122x | 1.0026x | 1.0112x | 1.0163x |
| 8 | 1.0061x | 1.0001x | 1.0147x | 1.0181x |

M12/M48 use only the selected W2 schedule or remain effectively baseline;
the larger and more stable gain begins when W13 also switches at M96.

At M2048, a final four-way rotated run gave:

| threads | baseline | auto | median speedup | explicit best |
|---:|---:|---:|---:|---:|
| 1 | 231.913 ms | 225.647 ms | 1.0278x | `unroll2` 225.640 ms |
| 8 | 31.010 ms | 29.943 ms | 1.0356x | `unroll2` 29.933 ms |

The stage-specific auto combination is within 0.04% of the best forced common
mode, while retaining W2's measured T0 behavior.

After passing H/F/team width into `PrepareJitKernels` so every selected key is
generated before worker startup, a final two-way regression check measured
M256 at 28.375→28.048 ms (1.0117x) on 1T and 3.891→3.828 ms (1.0166x) on 8T.
M2048 measured 231.388→225.635 ms (1.0255x) and 30.995→29.942 ms (1.0352x),
consistent with the four-way data above.

### Held-out shapes

| shape | threads | M96 | M256 |
|---|---:|---:|---:|
| H4096/F2048 | 1 | 1.0054x | 1.0141x |
| H4096/F2048 | 8 | 1.0074x | 1.0153x |
| H64/F2048 | 1 | 1.0000x | 0.9997x |
| H64/F2048 | 8 | 0.9987x | 0.9923x median |

H64/F2048 fails both stage guards, so auto and baseline invoke the same
kernel keys. Its sub-millisecond eight-worker distribution is bimodal: mean
and paired ratios remained within about 1%, and the apparent median movement
is scheduler noise rather than an automatic-kernel change.

## Interpretation

K scheduling and packed-B prefetch are not one global choice:

- W13 has long reductions and already depended on an L1 prefetch. It benefits
  from two-pair scheduling only after enough M-panel reuse exists.
- W2 has shorter reductions and many N blocks. Halving loop branches and
  prefetching the next B lines produces a direct, counter-visible cycle
  reduction.
- T1 does not outperform T0 on this private-L2 C8i topology.
- disabling prefetch or enabling two-pair W13 at small M is a real regression,
  not just timing noise.

The old path remains selectable, all rejected variants remain reproducible,
and the automatic policy is limited to the measured C8i CPU profile.
