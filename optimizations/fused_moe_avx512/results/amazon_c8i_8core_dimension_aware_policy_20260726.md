# Amazon C8i8 x86 BF16 dimension-aware policy calibration

Date: 2026-07-26

Machine:

- target: `AmazonC8i8Cores`
- CPU: Intel Xeon 6975P-C, CPUID family 6/model 173/stepping 1
- CPU set: `0-7`, one software worker per physical core
- worktree: `/home/ubuntu/zhangxu/fused_cpp`
- dtype: BF16

## Outcome

The production `auto` path now uses policy profile
`intel_06_ad_c8i_v1`, keyed by CPUID and model dimensions:

| Decision | C8i8 production rule |
| --- | --- |
| call ISA | AVX-512 BF16 when `max(M)*H*F <= 8192`; AMX otherwise |
| effective workers | 1 when `sum(M)*H*F <= 131072`; requested count otherwise |
| AMX pattern, `H < 4096` | `m2n2` |
| AMX pattern, `H >= 4096`, `2*F < H` | `m1n4` from M=96 |
| AMX pattern, `H >= 4096`, `2*F >= H` | `m1n4` from M=128 |
| AMX cache window | unwindowed for M<=16; otherwise 1 MiB W13 / 512 KiB W2 byte budgets |
| skew-wave target | `clamp(ceil(64*4096*512/(H*F)), 16, 256)` |

Unknown x86 CPUs use `generic_v1`, which preserves the previous M=76
pattern crossover, fixed 64-row wave target, and cache-byte formulas.
Explicit AVX-512 and AMX backends remain deterministic controls and do not
apply automatic ISA or worker-count selection.

Backend ID 104, `x86_bf16_auto`, stores one AMX-compatible K32/N32 packed
weight copy. Both selected ISA paths consume that copy, so call-level
AVX-512 selection does not require a second weight representation.

## Method

The main benchmark creates one logical input and weight set, prepares every
requested backend before timing, reuses output tensors, warms each variant,
and rotates measurement order within the same process. Prepack, JIT warmup,
reference computation, and output allocation are excluded from latency.
Every variant is checked against the PyTorch reference before timing.

Representative command:

```bash
ssh AmazonC8i8Cores \
  'env -C /home/ubuntu/zhangxu/fused_cpp \
    PYTHONPATH=src OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
    OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=PASSIVE \
    taskset -c 0-7 .venv/bin/python \
    benchmarks/bench_x86_bf16_policy.py \
    --tokens 64 --hidden 4096 --intermediate 512 \
    --experts 1 --routing hot --threads 8 --warmup 12 --runs 101'
```

The benchmark prints the detected profile and all decisions alongside
median/P90/P99/mean/stdev/best latency, GFLOP/s, correctness error, backend
ID, and prepack time. Environment overrides were used only for paired
calibration:

```text
FUSED_CPP_MOE_X86_POLICY_PROFILE=generic_v1|intel_06_ad_c8i_v1
FUSED_CPP_MOE_X86_ISA=avx512|amx
FUSED_CPP_MOE_AMX_PATTERN=m1n2|m2n2|m1n4
FUSED_CPP_MOE_X86_W13_CACHE_BLOCKS=<integer>
FUSED_CPP_MOE_X86_W2_CACHE_BLOCKS=<integer>
```

## ISA boundary

Small shapes were swept at one worker so that worker startup did not obscure
the ISA crossover.

| H/F | M | Result |
| --- | ---: | --- |
| 64/16 | 1 | AVX-512 51.5% faster than AMX |
| 64/16 | 2 | AVX-512 49.8% faster |
| 64/16 | 4 | AVX-512 32.8% faster |
| 64/16 | 8 | AVX-512 advantage narrowed to roughly 10%--19% |
| 64/16 | 16 | AMX about 9% faster |
| 128/32 | 1 | AVX-512 about 37% faster |
| 128/32 | 2 | AVX-512 about 28%--32% faster |
| 128/32 | 4 | approximately tied |
| 128/32 | 8 | AMX about 17.5% faster |

These boundaries support the compact work metric `M*H*F` and the C8i
crossover at 8192. A held-out H64/F16/M1 call gave:

| Requested workers | auto | forced AVX-512 | forced AMX | Interpretation |
| ---: | ---: | ---: | ---: | --- |
| 8 | 5.941 us | 12.965 us | 19.078 us | auto selects AVX-512 and one worker |
| 1 | 5.168 us | 5.115 us | 8.297 us | auto ISA regret versus AVX-512 is 1.04% |

For a production-size H4096/F512/M64 call, auto selected AMX:

| Workers | auto | forced AMX | auto regret |
| ---: | ---: | ---: | ---: |
| 1 | 0.920052 ms | 0.919424 ms | 0.068% |
| 8 | 0.199213 ms | 0.205049 ms | -2.85% |

The negative 8-worker regret is a favorable measurement difference, not a
claim that the identical selected AMX kernel is intrinsically faster.

## Effective worker boundary

The aggregate metric prevents many tiny experts from paying cooperative
startup/barrier cost independently.

| H/F and routes | Aggregate `sum(M)*H*F` | auto | forced AVX-512 8T | forced AMX 8T |
| --- | ---: | ---: | ---: | ---: |
| 256/64, `[1,1,1,1,1,1,1,1]` | 131072 | 17.930 us | 34.435 us | 29.661 us |

The next larger calibration point, H512/F128 with the same eight M1 experts,
is above the threshold; its forced AMX control improved from 122.643 us at
1T to 40.582 us at 8T. The production rule therefore retains requested
workers above 131072.

## AMX pattern and cache policy

Pattern sweeps showed:

- H1024/F256: `m2n2` won nearly all tested M48--2048 points.
- H4096/F512: the stable transition was near M96.
- H4096/F2048: `m1n4` became stable from M128.
- H2048/F512 was non-monotonic, so the policy conservatively keeps `m2n2`.

Held-out H4096/F512/M96 at 8T:

| Variant | Median |
| --- | ---: |
| auto (`m1n4`) | 0.210900 ms |
| forced `m1n4` | 0.209775 ms |
| forced `m2n2` | 0.216928 ms |

Auto regret was 0.54%. Exact M tails remain handled by `m1n2`.

At one worker, automatic cache windows versus explicit unwindowed traversal
gave:

| H/F | M64 speedup | M128 speedup |
| --- | ---: | ---: |
| 4096/512 | 1.20x | 2.53x |
| 4096/2048 | 1.24x | 2.47x |

M<=16 did not show a stable benefit, so C8i keeps it unwindowed.

## Dimension-scaled skew-wave target

The target comparison used forced AMX/m2n2 at 8 workers. Each profile was run
in a separate process with 31 measured samples:

| H/F | Routes | C8i target | generic target | C8i | generic | C8i speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1024/256 | `[768,128,128]` | 256 | 64 | 0.389054 ms | 0.398581 ms | 1.02x |
| 4096/2048 | `[144,24,24]` | 16 | 64 | 1.736484 ms | 2.451086 ms | 1.41x |

Example commands:

```bash
ssh AmazonC8i8Cores \
  'env -C /home/ubuntu/zhangxu/fused_cpp \
    PYTHONPATH=src OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
    OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=PASSIVE \
    FUSED_CPP_MOE_X86_POLICY_PROFILE=intel_06_ad_c8i_v1 \
    FUSED_CPP_MOE_AMX_PATTERN=m2n2 taskset -c 0-7 \
    .venv/bin/python benchmarks/bench_x86_bf16_policy.py \
    --tokens 1024 --hidden 1024 --intermediate 256 \
    --experts 3 --routing skewed --threads 8 --variants amx \
    --warmup 8 --runs 31'

ssh AmazonC8i8Cores \
  'env -C /home/ubuntu/zhangxu/fused_cpp \
    PYTHONPATH=src OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
    OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=PASSIVE \
    FUSED_CPP_MOE_X86_POLICY_PROFILE=generic_v1 \
    FUSED_CPP_MOE_AMX_PATTERN=m2n2 taskset -c 0-7 \
    .venv/bin/python benchmarks/bench_x86_bf16_policy.py \
    --tokens 1024 --hidden 1024 --intermediate 256 \
    --experts 3 --routing skewed --threads 8 --variants amx \
    --warmup 8 --runs 31'
```

The first target result is only directional because profiles were not rotated
inside one process. The 41.15% gap at H4096/F2048 is large enough to reject a
fixed 64-row target for that dimension.

## Correctness and build validation

Build:

```bash
ssh AmazonC8i8Cores \
  'env -C /home/ubuntu/zhangxu/fused_cpp \
    MAX_JOBS=8 FUSED_CPP_BUILD_MOE_ONLY=1 \
    .venv/bin/python setup.py build_ext --inplace'
```

Focused test suite:

```bash
ssh AmazonC8i8Cores \
  'env -C /home/ubuntu/zhangxu/fused_cpp \
    PYTHONPATH=src OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
    OMP_WAIT_POLICY=PASSIVE taskset -c 0-7 \
    .venv/bin/python -m pytest -q \
    tests/test_moe_backend_dispatch.py tests/test_moe_avx512_bf16.py'
```

Result: `226 passed, 3 skipped`.

Coverage added for this policy includes:

- CPUID/profile and every decision returned by the native policy probe;
- 8192 ISA and 131072 aggregate-worker boundaries;
- H/F-specific AMX pattern and cache choices;
- dimension-scaled target and skew classification;
- backend ID 104 metadata;
- the same ID 104 packed weights forced through AVX-512 and AMX, with
  bit-exact output equality;
- the same ID 104 weights falling back to AVX-512 after the AMX runtime
  kill-switch is enabled;
- existing AVX-512/AMX H/F/K/N tails and multi-thread scheduling regression.

## Interpretation limits

- Thresholds are calibrated for family 6/model 173 and are not architecture
  constants. A new CPU model must receive a new named profile and held-out
  validation.
- The N-target comparison used separate processes; only the large gap should
  be treated as decisive without a rotated rerun.
- Results are steady-state operator latency. Weight packing, first JIT,
  allocation, routing, and application-level orchestration are outside the
  timed region.
- This is a deterministic executor mapper. ISA, worker count, pattern, and
  cache choice are not new CPU MoE planner variables and do not imply a new
  schedule optimum.
