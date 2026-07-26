# Amazon C8i single-expert one-core results (2026-07-26)

## Outcome

The AMX executor now bypasses input gathering when top-k=1 routes every token
to one active expert and H is already K32-aligned. In that case the gathered
matrix would be byte-for-byte identical to the original contiguous input, so
W13 reads the input directly and the executor does not allocate its M-by-H
input scratch.

For BF16 H=4096/F=512 on one pinned C8i core, the output-reused direct-BF16
path beats KTransformers' synchronous public AMX BF16 `forward()` at every
tested M from 1 through 2048. Median speedup ranges from 2.36x to 13.25x; at
M=2048 the main sweep gives 2.89x (18.682 ms versus 53.994 ms). A final
post-rebuild repeat measured 19.559 ms, still 2.76x faster.

The generic weighted path, with only the final output preallocated, also wins
all 11 points. Its narrowest lead is M=2048 at 1.76x because it still
allocates and merges an FP32 route workspace. Weight prepack is excluded from
all latency numbers.

## Machine and software

- host alias: `AmazonC8i8Cores`
- CPU: Intel Xeon 6975P-C under KVM
- topology: 1 socket, 8 physical cores, 1 thread/core; measurements pin CPU 0
- cache: 48 KiB L1d/core, 2 MiB L2/core, 480 MiB shared L3
- ISA: AVX-512 BF16, AMX-TILE, AMX-BF16
- fused workspace base commit: `36b585c`
- KTransformers checkout: `d1a3ed8a308c`
- fused build: `FUSED_CPP_BUILD_MOE_ONLY=1`

The package warning about an unavailable main `_C` extension refers to an
unrelated x86 SDPA build issue. The independently built `fused_cpp._moe_C`
extension is loaded, and the reported backend is `x86_amx_bf16`.

## Correctness

The complete x86 MoE test file passed on the rebuilt extension:

```text
176 passed, 1 warning in 0.79s
```

The new regression covers both weighted FP32-route output and direct BF16
output for an aligned, nonzero hot expert with one and eight requested
threads. Existing unaligned-H, M-tail, pattern, and cache-window cases
exercise the gather fallback.

For E=1, M=64, H=4096, F=512:

| comparison | max abs | relative L1 |
| --- | ---: | ---: |
| fused vs PyTorch/oneDNN definition | 1.1921e-7 | 0.3688% |
| KTransformers vs PyTorch/oneDNN definition | 1.1921e-7 | 0.2953% |
| fused vs KTransformers | 1.1921e-7 | 0.3233% |

## Method

Both implementations use identical seeded BF16 input and gate/up/down
weights, E=1, top-k=1, H=4096, F=512, and hot routing. Each point uses 10
warmups and 31 timed calls in a fresh process pinned with:

```text
taskset -c 0
OMP_NUM_THREADS=1
OMP_DYNAMIC=FALSE
OMP_PROC_BIND=close
OMP_PLACES=cores
OMP_WAIT_POLICY=PASSIVE
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
```

The fused side is reproducible with:

```bash
taskset -c 0 env PYTHONPATH=src OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE \
  OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=PASSIVE \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tests/bench_moe_avx512_bf16.py \
  --backend x86_amx_bf16 --tokens 2048 --hidden 4096 \
  --intermediate 512 --experts 1 --top-k 1 --routing hot --threads 1 \
  --warmup 10 --runs 31 --skip-baseline --reuse-out --skip-weighted
```

Fused uses automatic AMX pattern/cache dispatch, a caller-reused `out`
tensor, and `skip_weighted=True`; this is valid here because softmax over the
single selected expert gives an exact route weight of one. KTransformers uses
the matching preallocated output and calls its synchronous public
`AMXBF16_MOE.forward()` directly, excluding task-queue submit/sync overhead.
Weight prepack and JIT warmup are outside the timed region. FLOP/s is
`M * 6 * H * F`.

The implementations are measured in separate processes rather than a rotated
same-process A/B because they own different native runtimes. CPU frequency was
not locked, so medians are primary. KTransformers latency advances in roughly
6 ms scheduling/tiling quanta on this setup.

## Full one-core comparison

| M | fused median ms | fused GFLOP/s | KTransformers direct ms | fused speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.452859 | 27.8 | 5.998650 | 13.25x |
| 4 | 0.459451 | 109.5 | 5.998703 | 13.06x |
| 16 | 0.499596 | 403.0 | 5.998835 | 12.01x |
| 48 | 0.768952 | 785.5 | 5.998455 | 7.80x |
| 64 | 0.915445 | 879.7 | 5.998468 | 6.55x |
| 80 | 1.089444 | 924.0 | 5.998559 | 5.51x |
| 128 | 1.535587 | 1048.9 | 5.998769 | 3.91x |
| 256 | 2.721637 | 1183.6 | 11.998650 | 4.41x |
| 512 | 5.074014 | 1269.7 | 11.998852 | 2.36x |
| 1024 | 9.672088 | 1332.2 | 23.998968 | 2.48x |
| 2048 | 18.682145 | 1379.4 | 53.993774 | 2.89x |

KTransformers prepack was about 35.8 ms per process; fused prepack was
8.3-8.7 ms. Those values are informational and excluded above.

## Direct-input contribution

The following before/after runs both reuse output and take the direct-BF16
top-1 path. They are separate-process medians, so they establish the size of
the effect without implying sub-percent precision.

| M | gathered input ms | direct input ms | improvement |
| ---: | ---: | ---: | ---: |
| 512 | 5.704781 | 5.074014 | 12.4% |
| 1024 | 11.158977 | 9.672088 | 15.4% |
| 2048 | 21.896510 | 18.682145 | 17.2% |

At M=2048 the bypass removes a 16 MiB input-scratch allocation and copy per
call. The remaining intermediate is 2 MiB.

## Generic weighted control

With arbitrary top-1 weights, fused retains its FP32 route workspace and
merge but still leads KTransformers throughout the sweep:

| M | fused weighted ms | KTransformers direct ms | fused speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.466741 | 5.998650 | 12.85x |
| 64 | 0.966101 | 5.998468 | 6.21x |
| 256 | 3.133458 | 11.998650 | 3.83x |
| 512 | 5.706182 | 11.998852 | 2.10x |
| 1024 | 10.962648 | 23.998968 | 2.19x |
| 2048 | 30.644572 | 53.993774 | 1.76x |

The next high-value single-core work is therefore persistent intermediate and
weighted-route workspace management, or a weighted top-1 W2 epilogue that can
write the caller's BF16 output directly.
