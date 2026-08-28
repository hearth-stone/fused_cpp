# SVE strided-T mHC control postprocess

## Decision

Keep the implementation as a candidate behind the explicit SVE mHC
entrypoints. It passes SVE128 and SVE256 correctness and materially improves the
full pre stage on the primary host. Public dispatch remains on the Torch
baseline until the second SVE machine is checked.

## Implementation

The combined native call retains the projection's row-major FP32 `[T,24]`
intermediate. Control processing has two T-parallel phases:

1. Process two token rows per loop iteration. On SVE256, the four pre and four
   post logits occupy one vector and share a stable `FEXPA` plus degree-2
   sigmoid. Pre adds epsilon; post multiplies by two. The 16 combination logits
   receive their affine transform and are written as `[T,4,4]`.
2. Gather each of the 16 matrix positions across T with element stride 16. One
   batch contains eight matrices at SVE256 or four at SVE128. The 16 vectors
   remain resident through row softmax, the first column normalization, and 19
   additional row/column normalization pairs, followed by one scatter store.

The Python candidate invokes projection plus control through one pybind call.
The standalone projection and control bindings remain available for calibration
and this comparator. Weighted residual reduction and the following RMSNorm are
not part of this optimization.

## Correctness

Machine and placement:

- `Arm-codex-internal`, NUMA3, CPUs `240-247`;
- GCC 13 C++17 release extension;
- Torch bundled `libgomp` preloaded to avoid the host's dual-runtime issue;
- `OMP_NUM_THREADS=8`, `OMP_DYNAMIC=FALSE`, `OMP_PROC_BIND=close`,
  `OMP_PLACES=cores`.

SVE256 command:

```bash
PYTHONPATH=src:. LD_PRELOAD=<torch-bundled-libgomp> OMP_NUM_THREADS=8 \
OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores \
numactl --physcpubind=240-247 --membind=3 \
.venv/bin/python -m pytest -q tests/test_deepseek_v4_mhc.py
```

Result: `27 passed`.

The extension was rebuilt with `FUSED_CPP_SVE_VECTOR_BITS=128`, and the process
called `prctl(PR_SVE_SET_VL, 16)` before importing the extension. Result:
`27 passed`.

Coverage includes T tails `1/7/9/17`, T=64, K splitting, 1/2/20 Sinkhorn
iterations, repeated calls, strict BF16 boundaries, and pre/post logits from
`-100` through `100`. At T=2048/H=4096, the full benchmark reported:

| Output | Maximum absolute error | Relative L2 |
| --- | ---: | ---: |
| post mix | `1.79e-7` | `2.59e-8` |
| combination mix | `1.34e-7` | `1.05e-7` |
| normed input | `0.015625` | `1.40e-5` |

No output contained NaN or infinity.

## Performance

Configuration:

- shape `T=2048`, `C=4`, `H=4096`;
- BF16 residual and FP32 projection/control tensors;
- SVE256, NUMA3-local memory;
- CPUs `240`, `240-247`, and `240-319` for 1/8/80 threads;
- five warmups and 21 AB/BA-alternating samples;
- baseline uses the same SVE projection followed by Torch control processing;
- statistic is median wall time.

Command:

```bash
PYTHONPATH=src:. LD_PRELOAD=<torch-bundled-libgomp> \
OMP_NUM_THREADS=<threads> OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
OMP_PLACES=cores numactl --physcpubind=<cpus> --membind=3 \
.venv/bin/python \
optimizations/deepseek_v4_mhc/benchmarks/bench_sve_control_postprocess.py \
--threads <threads> --warmup 5 --runs 21
```

Control-only results:

| Threads | Torch, ms | SVE, ms | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 5.904 | 0.226 | 26.12x |
| 8 | 3.125 | 0.040 | 77.43x |
| 80 | 3.992 | 0.079 | 50.26x |

Full projection plus pre-processing results after combining projection/control
under one Python operator call:

| Threads | Projection + Torch, ms | Projection + SVE, ms | Latency reduction | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 82.519 | 77.019 | 6.67% | 1.071x |
| 8 | 24.302 | 21.124 | 13.08% | 1.150x |
| 80 | 13.559 | 9.576 | 29.37% | 1.416x |

`llvm-objdump --mattr=+sve` confirms `FEXPA`, 16 indexed SVE loads, and indexed
stores in the outlined control function. Replacing `svfloat32x4_t` tuple
helpers with direct operations on the 16 named matrix vectors reduced SVE
stack `str/ldr` instructions in that function to zero.

The host currently loads `/opt/llvm-22/.../libgomp.so.1` for the extension and a
second Torch-bundled `libgomp`. Without preloading the Torch copy,
`omp_get_num_procs()` reports one processor in an eight-CPU bound cpuset and the
projection runs at single-thread speed. Those invalid measurements are excluded.

## Remaining boundary

The candidate still materializes approximately 192 KiB of `[T,24]` projection
output at T=2048. The next decision is whether folding the control transform
into the projection epilogue saves enough to justify coupling the M7/M4 GEMM
state machines to the nonlinear control kernel.
