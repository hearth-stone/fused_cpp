# x86 AVX-512 and AMX BF16 fused expert

This optimization implements the synchronous BF16 fused-SiLU expert path for
x86-64 CPUs with AVX-512 BF16 and AMX BF16. Runtime `auto` dispatch prefers
AMX when the CPU, OS, and build support it, then falls back to AVX-512 BF16.
It mirrors the existing ARM/SVE dataflow while using `VDPBF16PS` and a
decode-oriented M12/N32 microkernel on the AVX-512 path.

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

The shared cache/factory also owns the AMX backend and its per-expert automatic
pattern selection described below. Both generated implementations target the
System V x86-64 ABI; Windows builds retain the AVX-512 intrinsic path and do
not expose AMX.

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
  --backend x86_avx512_bf16 \
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

`benchmarks/bench_onednn_bf16_expert.cpp` is the all-oneDNN single-expert
control. Its `basic` variant executes separate BF16 gate and up matmuls,
oneDNN swish, oneDNN binary multiply, and the down matmul. Its
`postop_fused` variant executes the up matmul first, then attaches
`eltwise_swish + binary_mul(up)` to the gate matmul before the down matmul.
Thus the public oneDNN API reduces five timed primitives to three without a
custom epilogue. All three weights use `format_tag::any`; primitive creation
and one-time reorders are outside timing.

Build against the installed oneDNN prefix and run both variants in alternating
order:

```bash
g++ -std=c++17 -O3 -DNDEBUG -Wall -Wextra \
  -I/home/ubuntu/zhangxu/onednn-install/include \
  benchmarks/bench_onednn_bf16_expert.cpp \
  -L/home/ubuntu/zhangxu/onednn-install/lib \
  -Wl,-rpath,/home/ubuntu/zhangxu/onednn-install/lib \
  -ldnnl -fopenmp -o benchmarks/bench_onednn_bf16_expert

OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
  OMP_WAIT_POLICY=PASSIVE taskset -c 0,1 \
  benchmarks/bench_onednn_bf16_expert 256 4096 512 5 21
```

Unset `ONEDNN_MAX_CPU_ISA` uses AMX on C8i; setting it to
`AVX512_CORE_BF16` provides the same-ISA AVX-512 control. Correctness includes
basic-versus-fused comparison for every shape and a scalar FP32 reference for
small shapes. C8i results are in
[`results/amazon_c8i_2core_onednn_expert_20260719.md`](results/amazon_c8i_2core_onednn_expert_20260719.md).

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

## AMX backend and automatic policy

The prioritized AMX optimization backlog, acceptance checks, and rejected or
deferred design space are tracked in [`TODO.md`](TODO.md). The current active
item is the W13 vector SiLU epilogue that follows `TILESTORED`.

`auto` selects the reserved backend ID 102 (`x86_amx_bf16`) before backend ID
101 (`x86_avx512_bf16`) when AMX is available. AMX requires Linux, Xbyak,
AVX-512 BF16 for the vector epilogues, AMX-TILE, and AMX-BF16. Explicitly
requesting `x86_avx512_bf16` still forces the AVX-512 path. Setting
`FUSED_CPP_MOE_AMX_BF16=0` hides AMX from runtime discovery, so `auto` falls
back to AVX-512 BF16. Linux grants XTILEDATA state per thread, so every worker
requests `ARCH_REQ_XCOMP_PERM` before its first generated AMX call.

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

The W13 vector epilogue keeps its polynomial and exponent constants resident
in ZMM registers by default. `FUSED_CPP_MOE_AMX_SILU_EPILOGUE` is a validation
override:

- unset, empty, `auto`, or `resident` uses the bit-exact constant-resident
  implementation;
- `baseline` restores per-row constant broadcasts;
- `pipelined` additionally interleaves two independent rows by arithmetic
  stage, but did not consistently improve on `resident`;
- `rcp14` replaces division with approximate reciprocal and is not bit-exact.

On Amazon C8i, `resident` reduced a representative generated body by 28.6%
and improved a K=256 W13 microkernel by 8.4%; H=4096 full-expert latency was
neutral to 1.7% faster across the measured cases. Methodology and all retained
negative/approximate alternatives are recorded in
[`results/amazon_c8i_2core_amx_silu_20260720.md`](results/amazon_c8i_2core_amx_silu_20260720.md).

The default policy resolves a tile-register schedule independently for each
expert's routed row count:

- M < 76 uses `m2n2`; its final M1-16 tail naturally uses `m1n2`;
- M >= 76 uses `m1n4`; an odd final N32/W13 block uses `m1n2`.

`FUSED_CPP_MOE_AMX_PATTERN` is a validation and tuning override:

- unset, empty, or `auto` uses the per-expert policy above;
- `m1n2` keeps one M panel and two N tiles, and double-buffers K;
- `m2n2` keeps two M panels and two N tiles, reuses each B tile across both M
  panels, and covers M32 plus exact M17-31 kernels. M1-16 tails use `m1n2`;
- `m1n4` keeps one M panel and four accumulator tiles, reuses A across two
  adjacent N32 blocks, and retains K double buffering. Odd N32/W13 blocks use
  `m1n2`.

The M=76 crossover is currently calibrated from Amazon C8i measurements at
H=4096/F=512. Correctness is shape-independent, but performance portability
of that threshold must be revalidated for materially different x86 CPUs or
model dimensions. Automatic-policy correctness, crossover, mixed-routing, and
cache-window measurements are recorded in
[`results/amazon_c8i_2core_auto_dispatch_20260720.md`](results/amazon_c8i_2core_auto_dispatch_20260720.md).

Build and exercise the default automatic policy without dispatch environment
variables:

```bash
MAX_JOBS=2 FUSED_CPP_BUILD_MOE_ONLY=1 \
  .venv/bin/python setup.py build_ext --inplace
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_moe_backend_dispatch.py tests/test_moe_avx512_bf16.py
OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE \
  taskset -c 0,1 env PYTHONPATH=src .venv/bin/python \
  tests/bench_moe_avx512_bf16.py \
  --tokens 16 --hidden 4096 --intermediate 512 \
  --experts 8 --top-k 6 --routing balanced --threads 2
```

`benchmarks/bench_amx_bf16_gemm.cpp` compares the standalone W2 kernel with
prepacked oneDNN, while `benchmarks/bench_amx_bf16_w13.cpp` measures fused W13
through its SiLU/BF16 epilogue. Current C8i data and the M16+M1 discontinuity
are recorded in
[`results/amazon_c8i_2core_amx_20260719.md`](results/amazon_c8i_2core_amx_20260719.md).
`benchmarks/bench_amx_bf16_patterns.py` rotates `auto`, `m1n2`, `m2n2`, and
`m1n4` inside one process with identical inputs and packed weights. Its hot-M
sweep through M=2048 and balanced/skewed/hot routing results are in
[`results/amazon_c8i_2core_amx_patterns_20260719.md`](results/amazon_c8i_2core_amx_patterns_20260719.md).
Pass `--patterns auto --silu-epilogues baseline,resident,pipelined,rcp14` to
use the same rotation method for W13 epilogue A/B testing. The standalone W13
binary accepts the same comma-separated epilogue list as its seventh argument.

## Automatic N-window cache blocking

The JIT wrappers change loop order without changing kernels or packed layouts:

- `FUSED_CPP_MOE_X86_W13_CACHE_BLOCKS` is the number of adjacent W13 F16
  gate/up blocks in one window;
- `FUSED_CPP_MOE_X86_W2_CACHE_BLOCKS` is the number of adjacent W2 N32 blocks
  in one window;
- on AMX, unset, empty, or `auto` derives the block counts from approximately
  1 MiB of packed W13 and 512 KiB of packed W2 per worker;
- on AVX-512, unset, empty, or `auto` preserves the established loop order;
- zero explicitly disables blocking, while a positive integer forces that
  many blocks on either backend.

For a positive value, the wrapper processes every M panel inside the current
N window before advancing to the next window. This trades a bounded packed-B
working set in private L2 against reuse of the current A panel. It applies to
the AVX-512 JIT and all three AMX patterns; the AVX-512 intrinsic fallback
retains its original loop order. Invalid or negative values fail explicitly.

Scan several W13/W2 combinations in one process so all variants reuse the same
inputs, packed weights, JIT cache, and rotating measurement order:

```bash
OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE \
  taskset -c 0,1 env PYTHONPATH=src .venv/bin/python \
  benchmarks/bench_x86_bf16_cache_blocks.py \
  --backend x86_amx_bf16 --amx-pattern m1n4 \
  --configs auto:auto,0:0,2:0,4:0,6:0,0:8,0:16,0:24,4:8,4:16,4:24 \
  --tokens 2048 --hidden 4096 --intermediate 512 \
  --experts 1 --top-k 1 --routing hot --threads 2
```

The automatic AMX policy uses the repeatable Amazon C8i targets below rather
than one fixed block count:

```text
W13 blocks ~= 1 MiB / (64 * round_up(H, 32))
W2  blocks ~= 512 KiB / (64 * round_up(F, 32))
```

The implementation rounds down to a positive integral block count; for
`m1n4`, counts greater than one are rounded down to an even number. For
H=4096/F=512 this gives W13=4 and W2=16. With those values, `m1n4`
improved the M=2048 fused expert by 1.88x-2.00x over its unblocked schedule on
one core and 1.40x-1.52x on two cores across repeat scans. It also became
faster than cache-blocked `m2n2`
for M>=80 in the measured one-core sweep. Small M still favors `m2n2`, and
AVX-512 only gained about 2.5%-3.3% from W2 blocking while W13 blocking was
neutral or harmful, so AVX-512 retains its unblocked default. Environment
variables remain useful for reproducing the old unblocked AMX schedule
(`0`/`0`) and forced-policy experiments. Full methodology, dimension scaling,
routing results, and thermal caveats are in
[`results/amazon_c8i_2core_cache_blocking_20260719.md`](results/amazon_c8i_2core_cache_blocking_20260719.md).
