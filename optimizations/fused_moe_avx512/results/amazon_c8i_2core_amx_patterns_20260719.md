# Amazon C8i 2-core AMX tile-pattern comparison

Date: 2026-07-19. Host alias: `AmazonC8i2Cores`. Hardware and software match
the environment recorded in
[`amazon_c8i_2core_amx_20260719.md`](amazon_c8i_2core_amx_20260719.md).
The tested implementation adds `2M x 2N` and `1M x 4N` schedules to the same
Xbyak AMX JIT/cache used by the existing `1M x 2N` schedule. The default
remains `m1n2`; the other schedules were selected explicitly with
`FUSED_CPP_MOE_AMX_PATTERN`.

## Correctness and the padded-intermediate fix

The new tests exercise all three schedules at M=17, 32, 33, 64, and 2048 with
H=67 and F=35, so W13 has an odd F16 block and W2 has full N32 blocks plus a
tail. They also cover two-thread N splitting, direct BF16 output, and an
invalid selector value.

This coverage exposed an older AMX row-stride defect: W13 stored F=35 into a
48-element F16-padded row while W2 consumed a 64-element K32-padded row. W2
therefore crossed row boundaries and the final route could read beyond the
allocation. The executor now allocates and strides the row-major intermediate
by W2 Kpad; the unused columns remain zero.

After the fix:

- the complete x86 test file passed 72/72 with the default schedule;
- it passed 72/72 with `m2n2` forced and 72/72 with `m1n4` forced;
- the 15 large-M/tail cases passed in 30 independent processes: 450/450;
- every timed pattern had maximum absolute error `1.1920929e-7` versus the
  PyTorch definition.

The independent MoE extension was rebuilt with:

```bash
FUSED_CPP_BUILD_MOE_ONLY=1 .venv/bin/python setup.py build_ext --inplace
```

## Method

`benchmarks/bench_amx_bf16_patterns.py` creates one input and one set of
packed weights, warms every schedule five times, and then rotates measurement
order across 21 samples per schedule. This keeps JIT startup and weight
prepacking outside timing and reduces temperature/frequency ordering bias.
Reported values are medians. FLOPs count W1, W3, and W2 as `6 * M * H * F`.

All runs used H=4096, F=512, top-k=1, `OMP_DYNAMIC=FALSE`, and pinned CPUs.
Single-core runs used core 0. Two-core runs used cores 0,1 with
`OMP_WAIT_POLICY=PASSIVE`.

```bash
OMP_NUM_THREADS=THREADS OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
  OMP_WAIT_POLICY=PASSIVE taskset -c CORES env PYTHONPATH=src \
  .venv/bin/python benchmarks/bench_amx_bf16_patterns.py \
  --tokens TOKENS --hidden 4096 --intermediate 512 \
  --experts EXPERTS --top-k 1 --routing ROUTING \
  --threads THREADS --warmup 5 --runs 21
```

## Hot expert versus M

### One core

| M | `m1n2` | `m2n2` | `m2n2` / default | `m1n4` | `m1n4` / default |
|---:|---:|---:|---:|---:|---:|
| 17 | 0.9602 ms / 222.78 GFLOP/s | 0.5378 ms / 397.76 GFLOP/s | 1.785x | 0.9615 ms / 222.48 GFLOP/s | 0.999x |
| 32 | 1.0213 ms / 394.27 GFLOP/s | 0.5781 ms / 696.52 GFLOP/s | 1.767x | 1.0007 ms / 402.39 GFLOP/s | 1.021x |
| 64 | 2.0167 ms / 399.32 GFLOP/s | 1.2204 ms / 659.87 GFLOP/s | 1.652x | 1.9803 ms / 406.66 GFLOP/s | 1.018x |
| 256 | 8.0705 ms / 399.14 GFLOP/s | 4.8926 ms / 658.39 GFLOP/s | 1.650x | 7.9195 ms / 406.75 GFLOP/s | 1.019x |
| 2048 | 75.9673 ms / 339.22 GFLOP/s | 49.2775 ms / 522.95 GFLOP/s | 1.542x | 74.7104 ms / 344.93 GFLOP/s | 1.017x |

### Two cores

| M | `m1n2` | `m2n2` | `m2n2` / default | `m1n4` | `m1n4` / default |
|---:|---:|---:|---:|---:|---:|
| 17 | 0.5100 ms / 419.47 GFLOP/s | 0.3591 ms / 595.72 GFLOP/s | 1.420x | 0.5116 ms / 418.12 GFLOP/s | 0.997x |
| 32 | 0.5734 ms / 702.24 GFLOP/s | 0.3961 ms / 1016.47 GFLOP/s | 1.447x | 0.5562 ms / 723.89 GFLOP/s | 1.031x |
| 64 | 1.0619 ms / 758.40 GFLOP/s | 0.7530 ms / 1069.53 GFLOP/s | 1.410x | 1.0396 ms / 774.64 GFLOP/s | 1.021x |
| 256 | 3.9982 ms / 805.67 GFLOP/s | 2.8447 ms / 1132.35 GFLOP/s | 1.405x | 3.9736 ms / 810.66 GFLOP/s | 1.006x |
| 2048 | 43.0266 ms / 598.93 GFLOP/s | 33.4043 ms / 771.45 GFLOP/s | 1.288x | 42.1634 ms / 611.19 GFLOP/s | 1.020x |

`m2n2` removes the M16+M1 cliff at M=17 by loading each B tile once for both
M panels. Its advantage remains material through M=2048, although unaffected
gather, tile-store epilogues, route merge, and cache bandwidth reduce the
end-to-end ratio at large M and two cores.

## Route distributions at 2048 total routes

These two-core runs use eight experts. Balanced routing gives every expert
M=256. Skewed routing sends 75% to expert 0 and distributes the rest across
the other seven. Hot routing sends all routes to expert 0 and activates the
executor's two-core N split.

| Routing / per-expert M | `m1n2` GFLOP/s | `m2n2` GFLOP/s / ratio | `m1n4` GFLOP/s / ratio |
|---|---:|---:|---:|
| balanced / 256 x 8 | 572.82 | 789.07 / 1.378x | 586.36 / 1.024x |
| skewed / 1536, 74, 73 x 6 | 401.38 | 578.11 / 1.440x | 407.68 / 1.016x |
| hot / 2048, 0 x 7 | 596.41 | 757.45 / 1.270x | 606.45 / 1.017x |

The skewed case is slower in absolute terms because the global expert queue
cannot split its M=1536 task: one worker owns that expert while the other
finishes the seven small experts. The hot special case can split N across both
workers.

## Conclusion

For this H4096/F512 fused expert, B reuse across two M panels is much more
valuable than A reuse across four N tiles. `m2n2` improves every measured route
shape by 27.0%-78.5%; `m1n4` ranges from -0.3% to +3.1%. The results justify
keeping `m1n4` as an experimental specialization and make `m2n2` the leading
candidate for a future M>=17 policy. This change deliberately leaves `m1n2`
as the default until more model dimensions and multi-socket/cache regimes are
measured.
