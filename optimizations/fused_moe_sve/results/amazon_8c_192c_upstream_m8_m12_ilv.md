# Upstream M8/M12 ILV comparison

## Decision

Keep the current non-ILV schedules in the fused JIT path.

- Reject upstream M8 ILV. It regresses the Neoverse-V3 W13 shape by a
  reproducible 5.40% warm and 2.66% rotating-cold, and also regresses W2 by
  about 1.35%.
- Do not port upstream M12 ILV into the fused path. Its reproducible
  Neoverse-V3 gains are only 0.27-1.09%, while Neoverse-V1 is mixed. The only
  median above 1.1% is cold W2 on V1 at 1.81%, with a 1.10-2.80% five-run
  range.
- No production dispatch or fused epilogue changed. A future CPU-specific M12
  W2 experiment would need to beat the 2% threshold in the complete fused
  direct-store path before reconsideration.

## Method

- Date: 2026-07-26
- Workspace: `a131b5b` plus the benchmark and manifest changes recorded here
- Inputs: packed BF16 A/B and row-major FP32 output
- Threading: one thread, explicitly pinned to one core
- Build: `-O2 -std=c++17 -march=armv8.6-a+sve+bf16+i8mm`
- Variants in one process:
  - current Xbyak pure GEMM;
  - matching upstream non-ILV assembly;
  - upstream ILV assembly.
- Each measured sample uses one of all six permutations of the three variants.
  This gives every variant each timing position and balances pair order.
- Each table row summarizes five independent processes. Each process has 300
  samples per variant. Time is the median of the five process medians.
- `ILV speedup = non-ILV time / ILV time - 1`. The speedup column is the
  median of the five paired process speedups; the range shows all five.
- Warm uses one packed weight. Rotating-cold uses 64 distinct addresses:
  512 MiB for W13 and 256 MiB for W2.

The two target hosts were:

| Host | CPU | Core | SVE | GCC | Kernel |
|---|---|---:|---:|---|---|
| `AmazonECS8Cores` | Neoverse-V1, 8 cores | 0 | 256 bits | 12.4.0 | 6.17.0-1019-aws |
| `AmazonC5192Cores` | Neoverse-V3, 192 cores | 48, NUMA0 | 128 bits | 15.2.0 | 7.0.0-1006-aws |

## Neoverse-V1, SVE256

| Shape | State | M | non-ILV us | ILV us | non-ILV GFLOP/s | ILV GFLOP/s | ILV speedup | Five-run range |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| W13 `4096x1024` | warm | 8 | 273.201 | 273.100 | 245.64 | 245.73 | +0.046% | +0.030% to +0.080% |
| W13 `4096x1024` | warm | 12 | 341.210 | 340.521 | 295.02 | 295.62 | +0.159% | +0.117% to +0.349% |
| W13 `4096x1024` | cold | 8 | 336.659 | 341.568 | 199.34 | 196.47 | -1.353% | -1.499% to -0.976% |
| W13 `4096x1024` | cold | 12 | 404.452 | 406.139 | 248.89 | 247.85 | -0.133% | -0.720% to -0.015% |
| W2 `512x4096` | warm | 8 | 122.669 | 121.745 | 273.54 | 275.61 | +0.968% | -0.593% to +1.599% |
| W2 `512x4096` | warm | 12 | 190.494 | 193.991 | 264.22 | 259.45 | -1.329% | -4.976% to +0.967% |
| W2 `512x4096` | cold | 8 | 173.930 | 174.743 | 192.92 | 192.02 | -0.488% | -0.937% to -0.094% |
| W2 `512x4096` | cold | 12 | 227.956 | 223.052 | 220.80 | 225.65 | +1.811% | +1.099% to +2.804% |

The V1 warm W2 rows are noisier than the other cases and cross zero. They do
not provide adoption evidence.

## Neoverse-V3, SVE128

| Shape | State | M | non-ILV us | ILV us | non-ILV GFLOP/s | ILV GFLOP/s | ILV speedup | Five-run range |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| W13 `4096x1024` | warm | 8 | 234.567 | 247.651 | 286.10 | 270.98 | -5.400% | -5.468% to -5.187% |
| W13 `4096x1024` | warm | 12 | 291.357 | 289.444 | 345.50 | 347.78 | +0.663% | +0.638% to +0.684% |
| W13 `4096x1024` | cold | 8 | 289.500 | 297.601 | 231.81 | 225.50 | -2.662% | -2.890% to -2.422% |
| W13 `4096x1024` | cold | 12 | 354.898 | 353.956 | 283.64 | 284.39 | +0.266% | +0.053% to +0.631% |
| W2 `512x4096` | warm | 8 | 120.524 | 122.143 | 278.40 | 274.71 | -1.352% | -1.356% to -1.253% |
| W2 `512x4096` | warm | 12 | 156.450 | 155.238 | 321.71 | 324.22 | +0.764% | +0.667% to +0.993% |
| W2 `512x4096` | cold | 8 | 131.388 | 133.111 | 255.38 | 252.08 | -1.361% | -1.685% to -1.174% |
| W2 `512x4096` | cold | 12 | 165.628 | 163.818 | 303.88 | 307.24 | +1.086% | +0.482% to +1.370% |

## Correctness

The initial ABI smoke test used `M=8,12`, `K=N=64`, and six samples on both
SVE256 and SVE128. All outputs were bitwise equal between JIT, upstream
non-ILV, and upstream ILV.

All formal W13 and W2 measurements also reported:

```text
bitwise_equal=true
ilv_bitwise_equal=true
max_abs_diff=0
ilv_max_abs_diff=0
```

## Reproduction

V1 used:

```bash
export FUSED_CPP_MOE_SVE_VECTOR_BITS=256
export BUILD_DIR=/tmp/fused_cpp_jit_pure_gemm_ilv_256
taskset -c 0 \
  optimizations/fused_moe_sve/benchmarks/run_jit_vs_i8mm_pure_gemm.sh \
  --rows 8,12 --k 4096 --n 1024 --experts 64 \
  --warmup 64 --runs 300 --include-ilv
```

V3 used the same command with:

```bash
export FUSED_CPP_MOE_SVE_VECTOR_BITS=128
export BUILD_DIR=/tmp/fused_cpp_jit_pure_gemm_ilv_128
taskset -c 48 \
  optimizations/fused_moe_sve/benchmarks/run_jit_vs_i8mm_pure_gemm.sh \
  --rows 8,12 --k 4096 --n 1024 --experts 64 \
  --warmup 64 --runs 300 --include-ilv
```

Replace the shape with `--k 512 --n 4096` for W2, and use
`--experts 1 --warmup 32` for warm-weight measurements. Each command was
executed five times.

## Interpretation

The upstream M8 ILV schedule is not a newer universally faster M8 kernel. The
current non-ILV M8 path already overlaps the next packed-A/B loads with the
current BFMMLA block. Reordering those loads inside the arithmetic sequence is
microarchitecture sensitive and is substantially worse on V3.

M12 has one accumulator bank and no equivalent double-buffered next block, so
its ILV ordering can provide a small benefit. The measured gain is too small
and inconsistent across cache state and microarchitecture to justify copying
that schedule into every fused SiLU, packC, and W2 epilogue.
