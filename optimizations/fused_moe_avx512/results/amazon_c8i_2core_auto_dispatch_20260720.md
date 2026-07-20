# Amazon C8i 2-core AMX automatic dispatch validation

Date: 2026-07-20

## Scope

This report validates the x86 fused-expert policy that no longer requires
environment variables in normal use:

- `backend="auto"` prefers AMX BF16 when CPU/OS/build support is present and
  falls back to AVX-512 BF16 otherwise;
- each AMX expert selects `m2n2` for M < 76 and `m1n4` for M >= 76;
- unset cache controls derive W13/W2 windows from 1 MiB/512 KiB packed-B byte
  targets; for H=4096/F=512 these resolve to 4/16 blocks;
- explicit backend, pattern, zero, and positive block values remain forced
  validation/reproducibility overrides.

## Machine and build

- host alias: `AmazonC8i2Cores`;
- CPU: Intel Xeon 6975P-C, 2 physical cores, one thread per core;
- cache: 48 KiB L1d and 2 MiB private L2 per core, 480 MiB reported shared L3;
- ISA: AVX-512 BF16, AMX-TILE, and AMX-BF16;
- kernel: Linux `7.0.0-1006-aws`;
- compiler: GCC 15.2.0;
- PyTorch: 2.8.0+cpu;
- build: `FUSED_CPP_BUILD_MOE_ONLY=1`, Xbyak enabled, OpenMP enabled.

All timed processes were pinned to CPU 0 or CPUs 0-1 with
`OMP_DYNAMIC=FALSE`, `OMP_PROC_BIND=close`, and `OMP_WAIT_POLICY=PASSIVE`.
Weight preparation and JIT warmup were outside timed samples.

## Correctness and fallback

The focused x86 suite completed with `95 passed, 4 skipped`. It includes:

- automatic backend metadata selecting backend 102 on this machine;
- explicit AVX-512 backend 101 remaining selectable;
- `FUSED_CPP_MOE_AMX_BF16=0` making `auto` fall back to AVX-512;
- absent and explicit-`auto` pattern/cache controls producing bit-identical
  outputs;
- one invocation containing M=75 and M=76 experts, so `m2n2` and `m1n4` JIT
  keys coexist;
- H=641/F=545 tails and enough packed blocks to cross multiple automatic
  cache windows, on both one and two threads;
- forced patterns, cache windows, tails, and invalid-control errors.

The mixed-shape test and all benchmark shapes matched the PyTorch/oneDNN
staged reference within the existing BF16 contract. Timed benchmark cases
reported maximum absolute difference `1.1920929e-7`.

Commands:

```bash
MAX_JOBS=2 FUSED_CPP_BUILD_MOE_ONLY=1 \
  .venv/bin/python setup.py build_ext --inplace

OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
  OMP_WAIT_POLICY=PASSIVE taskset -c 0,1 env PYTHONPATH=src \
  .venv/bin/python -m pytest -q \
  tests/test_moe_backend_dispatch.py tests/test_moe_avx512_bf16.py
```

## Pattern crossover

The dedicated crossover comparison rotates only `m2n2` and `m1n4` in AB/BA
order with identical packed weights. The table reports median end-to-end
milliseconds for H=4096/F=512, one hot expert, and automatic cache windows.

| M | threads | m2n2 ms | m1n4 ms | preferred |
| ---: | ---: | ---: | ---: | --- |
| 48 | 1 | 0.786 | 0.889 | m2n2 |
| 64 | 1 | 1.018 | 1.055 | m2n2 |
| 68 | 1 | 1.193 | 1.171 | noisy edge |
| 72 | 1 | 1.176 | 1.195 | noisy edge |
| 76 | 1 | 1.253 | 1.207 | m1n4 |
| 80 | 1 | 1.255 | 1.237 | m1n4 |
| 64 | 2 | 1.305 | 1.300 | tie |
| 68 | 2 | 1.451 | 1.429 | noisy edge |
| 72 | 2 | 1.464 | 1.448 | noisy edge |
| 76 | 2 | 1.536 | 1.522 | m1n4 |
| 80 | 2 | 1.631 | 1.590 | m1n4 |

Repeated scans disagreed by roughly 1%-2% at M=68/72, where exact tails make
the crossover non-monotonic. M=76 is the first tested point with a stable
one-core `m1n4` win and no two-core regression, so the policy uses that
conservative threshold. It is a C8i H=4096/F=512 calibration, not an ISA
constant.

The crossover rows use `--patterns m2n2,m1n4 --warmup 10 --runs 101`, changing
only `--tokens` and the pinned/thread settings between rows.

For a skewed 512-route call with histogram
`[384, 19, 19, 18, 18, 18, 18, 18]`, automatic per-expert selection mixes one
`m1n4` expert with seven `m2n2` experts. Its one-core median was 9.036 ms,
4.3% faster than the best global forced policy (`m1n4`, 9.427 ms). On two
cores the corresponding medians were 10.595 and 10.665 ms, a 0.7% advantage.

Representative command:

```bash
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
  OMP_WAIT_POLICY=PASSIVE taskset -c 0 env PYTHONPATH=src \
  .venv/bin/python benchmarks/bench_amx_bf16_patterns.py \
  --tokens 512 --hidden 4096 --intermediate 512 --experts 8 \
  --top-k 1 --routing skewed --threads 1 --warmup 5 --runs 31
```

## Automatic cache windows versus the old unblocked loop order

This scan rotates absent controls (`auto:auto`), explicit `0:0`, and explicit
`4:16` in one process. It therefore compares policies under the same inputs,
packed weights, JIT cache, and approximate thermal state.

| M | threads | unblocked ms | automatic ms | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 1 | 0.523 | 0.522 | 1.00x |
| 48 | 1 | 1.078 | 0.797 | 1.35x |
| 80 | 1 | 2.474 | 1.274 | 1.94x |
| 256 | 1 | 7.890 | 3.423 | 2.31x |
| 2048 | 1 | 72.851 | 34.830 | 2.09x |
| 16 | 2 | 0.379 | 0.378 | 1.00x |
| 48 | 2 | 0.705 | 0.581 | 1.21x |
| 80 | 2 | 1.332 | 0.868 | 1.53x |
| 256 | 2 | 3.944 | 2.184 | 1.81x |
| 2048 | 2 | 37.862 | 22.982 | 1.65x |

At one thread, absent controls and explicit `4:16` agreed within 0.4% for all
reported shapes. The two-thread VM showed larger run-to-run frequency and
thermal variation even though both settings resolve to the same code path;
relative automatic-versus-unblocked conclusions were stable.

Representative command:

```bash
OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
  OMP_WAIT_POLICY=PASSIVE taskset -c 0,1 env PYTHONPATH=src \
  .venv/bin/python benchmarks/bench_x86_bf16_cache_blocks.py \
  --backend x86_amx_bf16 --amx-pattern auto \
  --configs auto:auto,0:0,4:16 --tokens 2048 --hidden 4096 \
  --intermediate 512 --experts 1 --top-k 1 --routing hot \
  --threads 2 --warmup 5 --runs 21
```

## No-environment default path

Finally, all three pattern/cache variables were removed and the public
benchmark used its default `--backend auto`. The reported backend was
`x86_amx_bf16` in every run.

| M | threads | median ms | median GFLOP/s |
| ---: | ---: | ---: | ---: |
| 48 | 1 | 0.796 | 758 |
| 48 | 2 | 0.575 | 1050 |
| 2048 | 1 | 25.875 | 996 |
| 2048 | 2 | 21.497 | 1199 |

Absolute long-M throughput varies with sustained AMX load and VM frequency:
the single-policy process above is intentionally shorter than the rotating
four-pattern or three-cache-policy scans. Use the within-process rotated
comparisons for policy conclusions and these figures only as the observed
no-environment end-to-end operating points.
