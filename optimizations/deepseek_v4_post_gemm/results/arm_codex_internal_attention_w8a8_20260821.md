# Arm-codex-internal DeepSeek V4 Attention W8A8

## Scope

- Machine: Arm-codex-internal, NUMA3 CPUs 240-319, 80 threads.
- Build: `FUSED_CPP_TARGET_CPU=armv8.6-a+sve+bf16+i8mm`, default `-O2`.
- Data: CPU BF16 activation/output, dynamic symmetric per-row A8,
  per-output-channel W8, FP32 accumulation.
- Shape: DeepSeek V4 Flash TP4 prefill with `M=2048`.
- Main Q and Indexer Q: `K=1024`, `N=8192` each.
- WO_B: `K=2048`, `N=4096`; TP collective is outside the measurement.
- Command: `PYTHONPATH=src:. OMP_NUM_THREADS=80 OMP_DYNAMIC=FALSE
  OMP_PROC_BIND=FALSE numactl --physcpubind=240-319
  --membind=3 .venv/bin/python tests/bench_deepseek_v4_attention_w8a8.py
  --m 2048 --threads 80 --warmup 5 --runs 15`.
- Statistic: median of 15 runs after 5 warmups.

## Correctness

The focused Arm suite passed all 44 selected tests. It covers checkpoint INT8
packing, BF16 source-weight preparation, exact agreement with the dynamic
quantization reference including ties-to-even rounding, shared-quantization
pair agreement, aligned-N register-scaled BF16/FP32 stores with bias, logical-N
tail fallback, `out=` behavior, and Dense/C128A/C4A projected postprocessing.
The projection error against the BF16 source-weight implementation was
0.616-0.619% relative L2 in the measured formal shapes, with maximum absolute
error 0.00073-0.00098.

## Performance

| Measurement | Generic BF16 | W8A8 | Ratio |
| --- | ---: | ---: | ---: |
| One Q projection | 10.975 ms | 6.626 ms | 1.66x |
| Two Q projections, shared-A8/A-pack pool | 26.679 ms | 10.669 ms | 2.50x |
| WO_B, TP-local | 12.361 ms | 7.120 ms | 1.74x |

The table's BF16 baseline is deliberately the generic `bf16_linear` wrapper so
both columns use the same library-level dispatch surface. It is not the
production Main-Q/Indexer-Q baseline. The specialized BF16 shared-Q post-GEMM
path has previously measured approximately 10.045 ms for both Q projections
together and remains about 6.2% faster than this 10.669 ms W8A8 pair. W8A8 now
converts INT32 accumulators, applies row/channel scales, adds optional bias, and
stores BF16 or FP32 inside the assembly epilogue; it no longer materializes the
64 MiB FP32 accumulator. Logical-N tails use final-dtype padded scratch and
crop rather than the legacy FP32 accumulator. Main-Q and Indexer-Q now share
one A pack and one dynamic M12/M8 worker pool over approximately 1 MiB packed-B
N windows. The remaining 6.2% gap requires full post-GEMM E2E validation before
changing the default.

Against the previous separate FP32-epilogue implementation on the same formal
shapes, direct scaled store changed W8A8 latency from 10.538 to 9.116 ms for one
Q projection (-13.5%), 21.898 to 13.083 ms for the shared-A8 pair (-40.3%), and
13.040 to 7.030 ms for WO_B (-46.1%).

Adding the shared packed-A pool and N windows then reduced the pair from 13.083
to 11.328 ms with M12 disabled (-13.4%). Enabling the M12 direct-scaled kernel
reduced it further to 10.669 ms (-5.8%). Relative to the original separate
FP32-epilogue pair at 21.898 ms, the combined improvement is 51.3%.

### OpenMP-only activation quantization

The BF16-to-A8 SVE quantizer and the remaining row-parallel helpers were moved
from `at::parallel_for` to the extension OpenMP runtime without changing the
rounding or scaling contract. On the same NUMA3 CPU mask with 10 warmups and 31
timed samples, the pair changed from 9.031 ms immediately before the change to
8.444 ms in the first candidate run. Five independent candidate processes
reported 8.228, 8.657, 8.022, 8.858, and 8.598 ms; their median is 8.598 ms.
This is an apparent 4.8% improvement against the immediately preceding run,
with about a 5% process-to-process range, so it is evidence of a modest gain
rather than a stable 3 ms result.

With output tensors reused through the native binding, the measured pair
changed from 8.765 to 6.712 ms. An `N=16` upper-bound probe for activation
quantization, A8 allocation, packed-A, OpenMP startup, and two minimal GEMMs was
1.235 ms after the change. A standalone single-runtime implementation executes
the complete prequantized scaled pair in about 3.205 ms. Therefore activation
quantization is not the remaining bottleneck; co-loaded OpenMP runtime and
affinity behavior plus output allocation remain the dominant integration gap.

### Torch GNU OpenMP runtime A/B

A temporary link-only candidate replaced `_C.so`'s `libgomp.so.1` dependency,
which resolves to LLVM `libomp.so` on this host, with the exact hashed GNU
runtime used by the installed Torch wheel:
`torch.libs/libgomp-947d5fa1.so.1.0.0`. No source or compiler optimization was
changed. `/proc/self/maps` then contained only that GNU runtime, and the same 44
focused Arm tests passed.

With `OMP_PROC_BIND=FALSE`, five independent 31-sample processes reported pair
medians of 4.659, 4.702, 4.725, 4.735, and 4.659 ms. Their 4.702 ms median is
45.3% lower than the 8.598 ms median from the LLVM-runtime build. One complete
candidate run measured 2.386 ms for one W8A8 Q projection, 4.652 ms for the
W8A8 pair, and 1.858 ms for W8A8 WO_B. The generic BF16 measurements in the
same process were 7.457, 14.891, and 6.947 ms respectively, showing that the
runtime mismatch affected other extension OpenMP kernels too.

`OMP_PROC_BIND=close OMP_PLACES=cores` also became functional: the reused-output
native pair measured 3.611 ms and the allocating wrapper measured 4.870 ms,
instead of timing out or using about one CPU. The temporary candidate was
removed after measurement and the remote extension and build-library outputs
were restored. Production adoption requires making Torch GNU-runtime discovery
and linking explicit in `setup.py`, with fallback behavior for non-wheel Torch
installations and Linux build validation beyond this host.

An earlier `OMP_PROC_BIND=close OMP_PLACES=cores` run reported 409.646 ms for
the pair. It is excluded: under the Python/Torch runtime that configuration
used only about one CPU (`perf stat`: 1.052 CPUs utilized), and 1T/8T single-Q
times were both about 132.2 ms. Disabling OpenMP binding while retaining the
NUMA CPU mask restored parallel execution. The production path still needs an
explicit affinity solution for the co-loaded libgomp/libomp runtimes before
adopting this backend under the repository's normal bound-thread runtime.

## Decision

Keep W8A8 explicitly selectable for `attn.wq_b`, `attn.indexer.wq_b`, and
`attn.wo_b`. Keep BF16 as the automatic/default path. A future default-dispatch
decision for Q requires resolving the dual-OpenMP affinity issue, followed by
complete post-GEMM E2E validation.
