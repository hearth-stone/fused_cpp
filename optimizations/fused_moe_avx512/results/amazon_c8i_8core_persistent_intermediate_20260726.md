# Amazon C8i persistent intermediate scratch (2026-07-26)

## Outcome

The x86 executor now leases grow-only BF16 intermediate buffers from a
concurrency-safe process pool. A buffer is allocated with `at::empty` only
when the selected record's high-water mark grows. Repeated calls no longer
allocate, value-initialize, and release the complete W13-to-W2 intermediate.
AMX still clears the per-row gap between W13's F16-padded output and W2's
K32-padded input so a narrower shape cannot consume stale non-finite values.

The automatic policy enables persistent storage for AMX when the aggregate
intermediate working set is at least 256 KiB. Smaller AMX calls and AVX-512
retain transient storage because the measured benefit was neutral there.
`FUSED_CPP_MOE_X86_PERSISTENT_INTERMEDIATE=1` and `=0` force the two paths for
validation and benchmarking.

For the single-expert direct-BF16 H=4096/F=512/M=2048 case, a 101-sample
same-process rotation improved one-core median latency from 19.506 to
18.593 ms, or 4.9%. An earlier 31-sample rotation measured 18.906 to
18.428 ms, or 2.6%. The feature therefore has a repeatable positive direction
but should not be assigned sub-percent precision on an unlocked-frequency VM.

## Machine and method

- host alias: `AmazonC8i8Cores`
- CPU: Intel Xeon 6975P-C under KVM
- topology: 8 physical cores, 1 thread/core
- private cache: 48 KiB L1d and 2 MiB L2 per core
- dtype/backend: BF16, `x86_amx_bf16` unless identified as AVX-512
- shape: E=1, top-k=1, H=4096, F=512, hot routing
- output: caller-reused BF16 tensor with `skip_weighted=True`
- packing and JIT warm-up: excluded
- affinity: `taskset`, one physical CPU per requested worker
- OpenMP: fixed width, dynamic teams disabled, close/core binding, passive wait

The benchmark changes the environment override outside each timed interval and
alternates transient/persistent order in one process with identical inputs,
weights, packed weights, JIT cache, and output:

```bash
taskset -c 0 env PYTHONPATH=src OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE \
  OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=PASSIVE \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python benchmarks/bench_x86_bf16_intermediate_scratch.py \
  --backend x86_amx_bf16 --tokens 2048 --hidden 4096 \
  --intermediate 512 --threads 1 --warmup 15 --runs 101
```

## Correctness

The rebuilt MoE-only extension passed the complete x86 test file:

```text
180 passed, 1 warning in 0.78s
```

The combined backend-dispatch/x86 run passed 183 tests with four expected
architecture skips.

New coverage includes:

- AVX-512 transient versus persistent and repeated-persistent bit equality;
- AMX one- and eight-thread reuse after a wider buffer was deliberately
  polluted with NaNs, followed by a narrower F=33/K32-tail call;
- rejection of invalid environment override values.

The benchmark maximum absolute difference from the PyTorch/oneDNN definition
was at most `1.3411e-7` in the reported AMX sweep.

## One-core AMX route-count sweep

Each row is an internally rotated A/B comparison. The 256 KiB automatic
threshold corresponds to M=256 for F=512.

| M | samples | transient ms | persistent ms | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 31 | 0.453082 | 0.457844 | 0.990x |
| 64 | 31 | 0.861102 | 0.862183 | 0.999x |
| 128 | 51 | 1.534110 | 1.531221 | 1.002x |
| 256 | 51 | 2.715723 | 2.671439 | 1.017x |
| 512 | 31 | 5.053909 | 4.957560 | 1.019x |
| 1024 | 31 | 9.597710 | 9.381435 | 1.023x |
| 2048 | 101 | 19.505580 | 18.593113 | 1.049x |

The forced persistent M=1 result is why `auto` does not use the pool for small
working sets. An independent M=2048 31-sample rotation measured a smaller but
still positive 1.026x.

## M=2048 thread sweep

| threads | samples | transient ms | persistent ms | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 101 | 19.505580 | 18.593113 | 1.049x |
| 2 | 51 | 9.818894 | 9.722358 | 1.010x |
| 4 | 51 | 5.256677 | 5.017064 | 1.048x |
| 8 | 101 | 2.991907 | 2.878722 | 1.039x |

The scratch allocation and initialization occur before the worker team starts,
so removing them can remain visible as GEMM work scales across cores.

## AVX-512 control

Forced persistent storage was neutral on the much longer AVX-512 path:

| M | samples | transient ms | persistent ms | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 21 | 0.501016 | 0.492678 | 1.017x |
| 256 | 21 | 28.654611 | 28.758220 | 0.996x |
| 2048 | 21 | 233.501430 | 233.726283 | 0.999x |

Consequently the automatic policy leaves AVX-512 on its established transient
allocation path. The force-on mode remains available for other CPUs and
dimensions.

## Default-path confirmation

After the final build, the ordinary benchmark with the environment override
unset selected the automatic persistent path at M=2048 and measured
18.124 ms / 1421.9 GFLOP/s over 31 samples. A pre-change separate-process
baseline measured 18.937 ms / 1360.8 GFLOP/s. These separate-process numbers
are consistent with the rotated comparison but are not used to calculate the
feature speedup.
