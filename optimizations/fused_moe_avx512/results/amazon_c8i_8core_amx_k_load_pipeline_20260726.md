# Amazon C8i8 AMX m1n2 True K-load software pipeline

## Decision

Keep the new `pipelined` schedule as an explicit m1n2 microkernel experiment.
Do not change automatic K-load or AMX pattern selection.

The pipeline is real rather than cosmetic: at M=2048, process-level counters
showed 1.01% fewer cycles, 0.90% fewer L1D pending-miss cycles, and 1.92% fewer
top-down memory-bound slots, despite 0.48% more instructions. Five independent
rotated runs gave median paired speedups of 0.75%, 0.66%, and 0.77% at
M=64, 512, and 2048.

That gain applies only to `m1n2`. The automatic `m2n2`/`m1n4` main patterns
were still 14%-17% faster than pipelined m1n2 on the representative shapes.
H=4096/F=512 also has no odd W13 F16 or W2 N32 block, so its automatic m1n4
path does not execute an m1n2 N tail. The measured microkernel improvement
therefore does not improve the current default full-expert path.

## Implementation

The baseline m1n2 loop already has two physical operand banks:

- bank 0: A=`TMM2`, B0=`TMM4`, B1=`TMM5`;
- bank 1: A=`TMM3`, B0=`TMM6`, B1=`TMM7`;
- W13/W2 accumulators remain in `TMM0` and `TMM1`.

Baseline executes load then dot for bank 0, followed by load then dot for bank
1. The new schedule preloads bank 0 and then repeats:

1. load the alternate bank;
2. compute the current bank;
3. advance K pointers;
4. preload the next current bank;
5. compute the alternate bank.

Odd and single K32 tails preserve the original `TDPBF16PS` accumulation
sequence. W13 and W2 use the same schedule. The mode is part of the JIT cache
key, so `baseline` and `pipelined` can alternate safely within one process.

`m2n2` and `m1n4` each use four accumulator tiles and the remaining four TMMs
for active A/B operands. They cannot hold a second complete operand bank
without spilling tile state, so the requested pipeline is deliberately reduced
to baseline for those patterns.

## Interface

`FUSED_CPP_MOE_AMX_K_LOAD_PIPELINE` accepts:

- unset, empty, `auto`, or `baseline`: established load-then-dot order;
- `pipelined`: true operand-bank ping-pong for m1n2 only.

Unknown values fail explicitly. Automatic mode remains baseline.

## Method

- Workspace: commit `a207323` plus this uncommitted pipeline change
- Machine: Amazon C8i, Intel Xeon 6975P-C, 8 physical cores/8 hardware threads
- Cache: 48 KiB L1D and 2 MiB L2 per core, shared 480 MiB L3
- Runtime: Linux 7.0.0-1006-aws, GCC 15.2.0, Python 3.12.13,
  PyTorch 2.13.0+cpu
- Build: MoE-only extension, OpenMP/Xbyak/AVX-512 BF16/AMX BF16 enabled
- Shape: one expert, top-k=1, H=4096, F=512, BF16, hot routing
- Timing: weight prepack excluded; variants rotate order in one process after
  10 per-variant warm-ups; 41 measured calls per variant
- Affinity: one-thread runs use CPU 0; multi-thread runs use explicit
  `FUSED_CPP_MOE_PIN_THREAD_CPUS` lists inside matching `taskset` masks
- Other policy: N32 weights, automatic B-load hint, resident SiLU, baseline W2
  store, per-call tile state, split-W13 enabled

Build and representative timing command:

```bash
FUSED_CPP_BUILD_MOE_ONLY=1 MAX_JOBS=8 \
  .venv/bin/python setup.py build_ext --inplace

taskset -c 0 env PYTHONPATH=src \
  OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  FUSED_CPP_MOE_W13_SPLIT_N=1 \
  .venv/bin/python benchmarks/bench_amx_bf16_patterns.py \
  --tokens 512 --hidden 4096 --intermediate 512 \
  --experts 1 --top-k 1 --routing hot --threads 1 \
  --patterns m1n2 --k-load-pipelines baseline,pipelined \
  --warmup 10 --runs 41
```

Positive speedup means pipelined is faster than baseline.

## Single-thread sweep

This is one 41-run same-process rotation across the full tested M sweep.

| M | baseline ms | pipelined ms | speedup |
|---:|---:|---:|---:|
| 16 | 0.499499 | 0.500505 | -0.20% |
| 32 | 0.685852 | 0.683008 | +0.42% |
| 64 | 1.010525 | 0.998439 | +1.21% |
| 128 | 1.627333 | 1.620841 | +0.40% |
| 512 | 5.396625 | 5.375662 | +0.39% |
| 2048 | 19.989049 | 19.842599 | +0.74% |

The effect is consistently small. M=16 is slightly negative, while the
larger working sets favor the pipeline.

## Independent-process repeats

M=64/512/2048 were each repeated in five fresh processes. Every process still
used a 41-run rotated baseline/pipelined comparison.

| M | baseline median-of-medians ms | pipeline median-of-medians ms | paired speedup range | paired speedup median |
|---:|---:|---:|---:|---:|
| 64 | 1.017619 | 1.008757 | -0.40% to +1.13% | +0.75% |
| 512 | 5.339001 | 5.303792 | +0.55% to +0.79% | +0.66% |
| 2048 | 19.771513 | 19.609132 | +0.75% to +0.89% | +0.77% |

M=512 and M=2048 reproduced the positive result in every process. M=64 had
one negative repeat and is closer to the host-noise floor.

## Multi-thread check

These are 41-run rotations with resident MoE threads explicitly pinned.

| M | threads | baseline ms | pipelined ms | speedup |
|---:|---:|---:|---:|---:|
| 512 | 2 | 2.756076 | 2.719030 | +1.36% |
| 512 | 4 | 1.439346 | 1.443266 | -0.27% |
| 512 | 8 | 0.813836 | 0.807725 | +0.76% |
| 2048 | 2 | 10.133099 | 10.030960 | +1.02% |
| 2048 | 4 | 5.102062 | 5.054724 | +0.94% |
| 2048 | 8 | 2.865817 | 2.856874 | +0.31% |

The small per-core gain generally survives N-split, but the M512/4T sign flip
again shows that it is not large enough to justify a broad automatic policy.

## Automatic-pattern comparison

The same process alternated automatic pattern selection and forced m1n2. The
two automatic K-pipeline labels resolve to the same baseline kernel because
the selected main pattern is not m1n2.

| M | automatic pattern ms | pipelined m1n2 ms | m1n2 latency penalty |
|---:|---:|---:|---:|
| 64 | 0.866741 | 1.004026 | +15.84% |
| 512 | 4.574168 | 5.287930 | +15.60% |
| 2048 | 17.039254 | 19.870464 | +16.61% |

At M=64 auto uses m2n2; at M=512/2048 it uses m1n4. The wider patterns win by
reusing loaded operands across four accumulators, which is much more valuable
than the roughly 0.7% load-overlap improvement inside m1n2.

## Counter validation

The host has `kernel.perf_event_paranoid=4`, so PMU events require
non-interactive sudo. Each mode used M=2048, one pinned core, 20 warm-ups, 101
measured calls, and five complete process repeats:

```bash
sudo -n perf stat -x, -r 5 \
  -e cycles,instructions,l1d_pend_miss.pending_cycles,l1d.replacement,topdown.memory_bound_slots \
  taskset -c 0 env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  FUSED_CPP_MOE_W13_SPLIT_N=1 \
  .venv/bin/python benchmarks/bench_amx_bf16_patterns.py \
  --tokens 2048 --hidden 4096 --intermediate 512 \
  --experts 1 --top-k 1 --routing hot --threads 1 \
  --patterns m1n2 --k-load-pipelines pipelined \
  --warmup 20 --runs 101
```

Counts include identical setup, PyTorch reference creation, JIT warm-up, and
timed calls. The long measured loop dominates both modes.

| counter | baseline | pipelined | change |
|:---|---:|---:|---:|
| cycles | 13,504,318,618 | 13,368,144,813 | -1.01% |
| instructions | 8,477,887,446 | 8,518,977,205 | +0.48% |
| L1D pending-miss cycles | 9,081,212,659 | 8,999,870,515 | -0.90% |
| L1D replacements | 1,273,286,780 | 1,277,808,666 | +0.36% |
| top-down memory-bound slots | 48,995,784,198 | 48,054,997,521 | -1.92% |

The pipeline adds loop-control work, explaining the instruction increase, but
reduces both cycles with outstanding L1D misses and memory-bound slots. That is
the intended load/compute-overlap signature.

## Correctness

The focused tests alternate baseline, pipelined, and baseline again in one
process to verify cache isolation. They cover:

- one, odd, and even K32-block counts independently in W13 and W2;
- odd hidden/intermediate dimensions and N tails;
- one and two threads;
- BF16 bit equality with baseline and tolerance against the PyTorch expert;
- explicit rejection of an unknown mode.

Results on Amazon C8i8:

```text
focused: 7 passed, 212 deselected
complete tests/test_moe_avx512_bf16.py: 219 passed
```

## Portability caveat

The result is calibrated on Xeon 6975P-C with H=4096/F=512. It establishes
that the schedule is correct and creates genuine overlap, but the sub-1%
benefit is not an ISA-wide constant. Another CPU, K length, packed-B cache
state, or AMX pattern can move or erase it. Keep the explicit mode for future
dimension-aware calibration.
