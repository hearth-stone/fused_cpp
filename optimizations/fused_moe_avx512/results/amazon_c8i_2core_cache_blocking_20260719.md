# Amazon C8i 2-core fused-expert cache-blocking study

Date: 2026-07-19. Host alias: `AmazonC8i2Cores`. The host is an Intel Xeon
6975P-C VM with 48 KiB L1D and 2 MiB private L2 per core and 240 MiB shared
L3. The software environment and AMX implementation are described in
[`amazon_c8i_2core_amx_20260719.md`](amazon_c8i_2core_amx_20260719.md).

## Change and method

The experiment adds optional W13 and W2 N-window traversal to the existing
AVX-512 and AMX JIT wrappers. A window processes every M panel before moving
to the next group of packed-B blocks. A zero value retains the former loop
order and serves as the baseline. Packed layouts, microkernels, output, and
the intrinsic AVX-512 fallback are unchanged.

`benchmarks/bench_x86_bf16_cache_blocks.py` creates one input and one set of
packed weights, checks every configuration against the PyTorch definition,
warms every configuration, and rotates timing order in one process. This
keeps prepack and JIT compilation outside timing and reduces ordering bias.
Reported values are medians. FLOPs count W1, W3, and W2 as
`6 * routes * H * F`. All maximum absolute errors were at most `2.3842e-7`.

Single-core runs were pinned to CPU 0. Two-core runs were pinned to CPUs 0,1
with `OMP_DYNAMIC=FALSE` and `OMP_WAIT_POLICY=PASSIVE`. AMX frequency changed
substantially as the VM warmed up: the same unblocked M=2048 `m2n2` baseline
ranged from about 52 to 66 ms across processes. Therefore only ratios from a
single rotating run are compared; absolute values from separate tables must
not be combined.

Example command:

```bash
OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE \
  taskset -c 0,1 env PYTHONPATH=src .venv/bin/python \
  benchmarks/bench_x86_bf16_cache_blocks.py \
  --backend x86_amx_bf16 --amx-pattern m1n4 \
  --configs 0:0,2:0,4:0,6:0,0:8,0:16,0:24,4:8,4:16,4:24 \
  --tokens 2048 --hidden 4096 --intermediate 512 \
  --experts 1 --top-k 1 --routing hot --threads 2
```

## Main shape: H=4096, F=512, M=2048

Each packed W13 block is 256 KiB and each W2 block is 32 KiB.

### Independent-window scan

| Backend / threads | Baseline | Best W13-only | Best W2-only |
|---|---:|---:|---:|
| AMX `m2n2`, 1 | 52.040 ms | 4 blocks: +10.0% | 16 blocks: +9.8% |
| AMX `m2n2`, 2 | 33.780 ms | 4 blocks: +8.5% | 24 blocks: +1.1% |
| AMX `m1n4`, 1 | 78.741 ms | 4 blocks: +45.8% | 16 blocks: +19.9% |
| AVX-512 JIT, 1 | 260.091 ms | no gain | 4 blocks: +2.6% |
| AVX-512 JIT, 2 | 158.050 ms | approximately neutral | 8 blocks: +3.3% |

The AVX-512 W13 wrapper already traversed B blocks outside M panels, so a
bounded W13 window adds A reloads without enough B-cache benefit. Its W2
schedule sees only a modest improvement. AMX has the opposite issue: its
panel-local schedule repeatedly streams the full packed weights, so blocking
is much more valuable.

### Joint W13/W2 scan for AMX `m1n4`

| Threads | Unblocked | W13=4, W2=16 | Best observed | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 75.235 ms | 37.686 ms | 37.626 ms at 4:32 | 2.00x |
| 2 | 42.228 ms | 27.827 ms | 27.827 ms at 4:16 | 1.52x |

W13=4/W2=16 is the common recommendation: it was best on two cores and within
0.2% of the best one-core median. The joint scan also shows that blocking both
phases matters; the separate W13-only and W2-only scans should not be combined
quantitatively because they ran at different sustained AMX frequencies.

A final independent two-configuration repeat after the complete regression
suite measured 78.064 to 41.567 ms on one core (1.88x) and 50.930 to 36.351 ms
on two cores (1.40x). The lower ratios and very different two-core absolute
times reinforce the frequency caveat, while confirming that the improvement
remains material. Across both repeat scans, use 1.88x-2.00x for one core and
1.40x-1.52x for two cores rather than treating one sample as a fixed result.

## Interaction with AMX tile pattern

Cache blocking changes the pattern ranking. With H=4096/F=512 and W13=4,
W2=16, patterns were rotated inside the same process:

| M | Threads | `m1n4` | `m2n2` | Winner |
|---:|---:|---:|---:|---|
| 17 | 1 | 0.647 ms | 0.530 ms | `m2n2` |
| 32 | 1 | 0.722 ms | 0.576 ms | `m2n2` |
| 64 | 1 | 1.136 ms | 1.107 ms | `m2n2` |
| 80 | 1 | 1.326 ms | 1.380 ms | `m1n4` |
| 128 | 1 | 1.953 ms | 2.143 ms | `m1n4` |
| 256 | 1 | 3.736 ms | 4.404 ms | `m1n4` |
| 2048 | 1 | 43.325 ms | 50.476 ms | `m1n4` |
| 64 | 2 | 0.696 ms | 0.706 ms | `m1n4` (1.4%) |
| 256 | 2 | 2.279 ms | 2.461 ms | `m1n4` |
| 2048 | 2 | 27.107 ms | 30.002 ms | `m1n4` |

Without blocking, `m2n2` beat `m1n4` by about 1.5x at M=2048 because it
reused B across two M panels. Once an L2-sized B window supplies that reuse at
the wrapper level, `m1n4`'s A reuse across four N tiles becomes valuable. The
one-core crossover was around M=80; exact-M and small-route work should retain
`m2n2`.

## Route distributions

For 2048 total routes across eight experts, two-core `m2n2` with W13=4/W2=24
improved the balanced M=256x8 case by 9.3% and the skewed
M=1536,74,73x6 case by 8.9%. Thus the benefit is not limited to the hot-expert
N-split path. With `m1n4` and W13=4/W2=16, the large hot expert reached
950.7 GFLOP/s in its same-process pattern comparison.

## Window size scales with bytes

The best W13 block count changed inversely with H, while the best W2 count
changed inversely with F:

| H / F / M | W13 bytes/block | W2 bytes/block | Recommended blocks | Joint speedup |
|---|---:|---:|---:|---:|
| 2048 / 512 / 256 | 128 KiB | 32 KiB | 8 / 16 | 1.94x |
| 4096 / 512 / 2048 | 256 KiB | 32 KiB | 4 / 16 | 2.00x (1 core), 1.52x (2 cores) |
| 4096 / 1024 / 256 | 256 KiB | 64 KiB | 4 / 8 | 2.22x |
| 8192 / 512 / 256 | 512 KiB | 32 KiB | 2 / 16 | 1.97x |

This supports byte-based targets:

```text
W13 window ~= 1 MiB  = blocks * 64 * round_up(H, 32)
W2  window ~= 512 KiB = blocks * 64 * round_up(F, 32)
```

The remaining private-L2 capacity holds A panels, AMX tile-store scratch,
intermediate data, and loop metadata. W2 was relatively flat from roughly
512 KiB to 1 MiB, but 512 KiB is the safer common target. `m1n4` should use an
even count so every iteration can take its paired-block path.

## Conclusion

Cache blocking materially improves AMX fused experts and only slightly helps
AVX-512. The strongest measured policy is byte-based blocking plus a
route-size-aware AMX pattern: use `m2n2` for small M and `m1n4` from roughly
M=80 for H=4096/F=512. These controls remain opt-in because the crossover and
ideal byte targets have only been measured on one C8i cache topology; a
production auto-policy needs more H/F shapes and machines.
