# x86 AVX-512 and AMX BF16 fused expert

This optimization implements the synchronous BF16 fused-SiLU expert path for
x86-64 CPUs with AVX-512 BF16 and an explicit experimental AMX backend. It
mirrors the existing ARM/SVE dataflow while using `VDPBF16PS` and a
decode-oriented M12/N32 microkernel.

The fusion boundary and M12 panel loop follow
`csrc/moe/arm/sve_bf16/kernels.S` and the ARM executor. The BF16-pair operand
layout and independent FP32 accumulators were cross-checked against oneDNN's
`src/cpu/x64/gemm/bf16/jit_avx512_core_gemm_bf16bf16f32_kern.cpp` and its copy
kernels under the ignored local `3rdparty/oneDNN` reference tree. The default
compute path now generates its own kernels with the pinned Xbyak v7.37
submodule; it does not copy oneDNN JIT code. The original intrinsic kernels
remain the correctness and runtime fallback.

## Supported contract

- dense BF16 `w13=[E, 2F, H]` and `w2=[E, H, F]` weights;
- `activation="silu"`, `fuse_silu=True`, polynomial degree 4, 5, or 6;
- arbitrary positive H/F through zero padding;
- one or two worker threads, top-k routing, weighted merge, output buffers,
  and top-1 `skip_weighted` direct BF16 output;
- no expert bias and no scheduled, async, or vLLM-staged x86 executor yet.

Backend ID 101 and N tile 32 are stored in the prepared-weight metadata.
Runtime dispatch verifies AVX-512F/BW/VL, AVX-512 BF16, and OS-enabled ZMM
state. Set `FUSED_CPP_MOE_AVX512_BF16=0` to disable the backend.

## Packed layouts and dataflow

Weights are packed once by
`prepare_fused_moe_bf16_tiled_weights(..., fuse_silu=True)`. Every packed K
unit contains two adjacent BF16 values consumed by one `VDPBF16PS`:

- W13 uses one 16-feature block at a time; each K pair stores 16 gate pairs
  followed by 16 up pairs;
- W2 uses one 32-output block at a time; each K pair stores 32 BF16 pairs;
- gathered input is grouped into logical M12 panels stored in 16-lane
  VNNI2 packed-A blocks;
- the W13 epilogue writes `SiLU(gate) * up` directly in W2's packed-A layout;
- W2 writes FP32 directly to flat route rows, then an AVX-512 merge produces
  BF16 token output. Top-1 skip-weighted execution writes BF16 directly.

Every full 12-row panel dispatches to an exact-M12 register kernel, so M=48
executes four M12 panels. A final 1-11 row panel dispatches to its own exact-M
kernel rather than computing 16 physical rows. The broader scheduled/async API
remains future work, so this backend is still experimental.

## Xbyak JIT dispatch

The process-global, mutex-protected cache specializes these dimensions:

- W13 or W2 operation and AVX-512 BF16 ISA;
- exact logical M from 1 through 12;
- W13 SiLU polynomial degree 4, 5, or 6;
- W2 valid N lanes from 1 through 32 and FP32-route or direct-BF16 output.

K remains a runtime loop so model dimensions do not multiply the cache. The
executor resolves all kernels required by the current expert route counts
before entering worker regions. Xbyak allocates writable code while generating
and `readyRE()` publishes read/execute pages before a function pointer enters
the cache. Small-M W2 kernels use two independent K accumulation sets when
register capacity permits and reduce them before the store epilogue.

`FUSED_CPP_MOE_AVX512_IMPL` controls the implementation:

- `auto` (default): use JIT when Xbyak is built and a key generates
  successfully, otherwise run the intrinsic implementation for that call;
- `jit`: require JIT and report a missing submodule or generation failure;
- `intrinsic`: bypass code generation and use the preserved implementation.

The shared cache/factory also owns the explicit AMX backend described below.
Both generated implementations target the System V x86-64 ABI; Windows builds
retain the AVX-512 intrinsic path and do not expose AMX.

Two-thread builds reuse the extension's OpenMP team and fall back to standard
threads when OpenMP is unavailable, nested, or unable to provide the requested
team size. The caller participates as thread zero in the fallback path.

## Build, test, and benchmark

On an x86 host, build only the independent MoE extension when unrelated
architecture-specific sources in the main extension are unavailable:

```bash
git submodule update --init 3rdparty/xbyak
MAX_JOBS=2 FUSED_CPP_BUILD_MOE_ONLY=1 \
  .venv/bin/python setup.py build_ext --inplace
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_moe_backend_dispatch.py tests/test_moe_avx512_bf16.py
```

If the Xbyak submodule is absent, the extension still builds with intrinsic
dispatch. `FUSED_CPP_MOE_AVX512_IMPL=jit` then fails explicitly instead of
silently claiming a JIT run.

The end-to-end benchmark excludes custom weight prepack from timed execution.
Pin the process and cap oneDNN to the same ISA for a fair AVX-512 comparison:

```bash
ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16 OMP_NUM_THREADS=2 \
  OMP_WAIT_POLICY=PASSIVE FUSED_CPP_MOE_AVX512_IMPL=jit \
  taskset -c 0,1 env PYTHONPATH=src .venv/bin/python \
  tests/bench_moe_avx512_bf16.py \
  --tokens 16 --hidden 4096 --intermediate 512 \
  --experts 8 --top-k 6 --routing balanced --threads 2
```

Without `ONEDNN_MAX_CPU_ISA`, oneDNN selects AMX on Amazon C8i. That is useful
as a machine-level ceiling but is not an AVX-512 kernel comparison. See
[`results/amazon_c8i_2core_20260718.md`](results/amazon_c8i_2core_20260718.md).

For a lower-overhead oneDNN control, `benchmarks/bench_moe_onednn_avx512.cpp`
uses public oneDNN matmul primitives with `format_tag::any`, reorders both
weights once, reports the selected implementations, and times only W13,
poly5-SiLU/multiply, and W2.

`benchmarks/bench_avx512_bf16_gemm.cpp` isolates `ComputeW2` as a single
BF16-by-BF16-to-FP32 GEMM. It reports the already-packed kernel separately
from A packing plus kernel execution, while oneDNN uses an `any` weight
descriptor and performs its one-time reorder outside timing. Use
`ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16`, `OMP_NUM_THREADS=1`, and `taskset` for
the same-ISA single-core comparison. The result report also includes oneDNN's
default AMX implementation as a separate hardware-ceiling comparison.

`benchmarks/bench_avx512_bf16_w13.cpp` isolates the fused gate/up GEMM,
polynomial SiLU-multiply, BF16 conversion, and W2 packed-A epilogue. Run the
same binary in separate `FUSED_CPP_MOE_AVX512_IMPL=jit` and `intrinsic`
processes to compare generated and fallback kernels without routing or W2.

## Experimental AMX backend

`x86_amx_bf16` activates the reserved backend ID 102 without changing the
default `auto` selection, which remains `x86_avx512_bf16`. It requires Linux,
Xbyak, AVX-512 BF16 for the vector epilogues, AMX-TILE, and AMX-BF16. Set
`FUSED_CPP_MOE_AMX_BF16=0` to hide it from runtime discovery. Linux grants
XTILEDATA state per thread, so every worker requests `ARCH_REQ_XCOMP_PERM`
before its first generated AMX call.

AMX reuses the existing K-pair/N32 packed-weight format, but its backend rounds
K to 32. Gathered input and the W13 intermediate are row-major M16 panels.
The intermediate row stride is W2's K32-padded size, which can be wider than
W13's F16-padded output (for example F=35 uses a 64-element stride); the extra
columns remain zero so W2 never reads across row boundaries.
The cache specializes exact M=1 through 16, operation, W13 polynomial degree,
W2 N tail, and output type; K remains a dynamic K32 loop. Larger M values are
M16 panels plus an exact tail.

W13 uses two FP32 accumulator tiles for gate and up, with double-buffered A/B
occupying all eight TMM registers. Since ZMM cannot read TMM state directly,
both accumulators are `TILESTORED` to a 2 KiB stack scratch before the existing
ZMM polynomial SiLU-times-up epilogue writes row-major BF16. W2 uses one or two
accumulator tiles according to N and stores through the same scratch before
route-aware FP32 or BF16 output.

`FUSED_CPP_MOE_AMX_PATTERN` selects an experimental tile-register schedule:

- `m1n2` (default) keeps one M panel and two N tiles, and double-buffers K;
- `m2n2` keeps two M panels and two N tiles, reuses each B tile across both M
  panels, and covers M32 plus exact M17-31 kernels. M1-16 tails use `m1n2`;
- `m1n4` keeps one M panel and four accumulator tiles, reuses A across two
  adjacent N32 blocks, and retains K double buffering. Odd N32/W13 blocks use
  `m1n2`.

The selector is intentionally opt-in: unset or empty remains `m1n2`, so these
experiments do not change backend auto-selection or the established AMX
schedule.

Build and select the backend explicitly:

```bash
MAX_JOBS=2 FUSED_CPP_BUILD_MOE_ONLY=1 \
  .venv/bin/python setup.py build_ext --inplace
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_moe_backend_dispatch.py tests/test_moe_avx512_bf16.py
OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE \
  taskset -c 0,1 env PYTHONPATH=src .venv/bin/python \
  tests/bench_moe_avx512_bf16.py --backend x86_amx_bf16 \
  --tokens 16 --hidden 4096 --intermediate 512 \
  --experts 8 --top-k 6 --routing balanced --threads 2
```

`benchmarks/bench_amx_bf16_gemm.cpp` compares the standalone W2 kernel with
prepacked oneDNN, while `benchmarks/bench_amx_bf16_w13.cpp` measures fused W13
through its SiLU/BF16 epilogue. Current C8i data and the M16+M1 discontinuity
are recorded in
[`results/amazon_c8i_2core_amx_20260719.md`](results/amazon_c8i_2core_amx_20260719.md).
`benchmarks/bench_amx_bf16_patterns.py` rotates `m1n2`, `m2n2`, and `m1n4`
inside one process with identical inputs and packed weights. Its hot-M sweep
through M=2048 and balanced/skewed/hot routing results are in
[`results/amazon_c8i_2core_amx_patterns_20260719.md`](results/amazon_c8i_2core_amx_patterns_20260719.md).
