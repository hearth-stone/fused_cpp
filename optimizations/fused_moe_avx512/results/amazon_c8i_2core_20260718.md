# Amazon C8i 2-core AVX-512 BF16 fused-expert results

Date: 2026-07-18. Host alias: `AmazonC8i2Cores`. The CPU was an Intel Xeon
6975P-C with two visible physical cores, 48 KiB L1D and 2 MiB L2 per core.
The process was pinned to one or both cores. GCC was 15.2.0; PyTorch was
2.8.0+cpu under Python 3.12.13. The standalone build used oneDNN commit
`827155c34da859cb6a5467057cc9ea0ac5116a67` (library version 3.14). Custom and
oneDNN weight preparation was completed before timing where noted.

## Correctness

```bash
MAX_JOBS=2 FUSED_CPP_BUILD_MOE_ONLY=1 \
  .venv/bin/python setup.py build_ext --inplace
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_moe_backend_dispatch.py tests/test_moe_avx512_bf16.py
```

The AVX-512 suite passed 14 cases. Coverage includes H/F/M tails, four
successive M12 panels, top-k 1 and 2, one and two threads, SiLU polynomial
degrees 4/5/6, top-1 direct output, user output buffers, invalid bias/thread
inputs, and the runtime kill switch. A separate `OMP_THREAD_LIMIT=1` run also
passed the dual-thread case through the standard-thread fallback.
The combined dispatch and x86 run passed 17 tests with four architecture
skips. Representative maximum absolute error against the PyTorch BF16
definition was at most `1.19e-7` for the benchmark's small-normal inputs.
Object-code inspection confirmed `vdpbf16ps` in the native kernel.

## AVX-512-to-AVX-512 end-to-end comparison

The baseline is the existing `fused_moe_naive` PyTorch implementation. It runs
two oneDNN matmuls plus separately materialized SiLU/multiply per expert. Both
processes were forced to `ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16`; otherwise this
CPU's oneDNN selects AMX (`brg_matmul:avx10_1_512_amx`). FLOP/s uses
`routes * 6 * H * F`, counting W1, W3, and W2 GEMMs only. Each process used 31
custom and 21 baseline samples after warmup. The reported two-core balanced
row is the middle of three process medians because this two-vCPU instance had
visible process-level jitter; its three custom medians were 171.24, 193.53,
and 207.44 GFLOP/s.

| Shape and routing | Threads | Fused AVX-512 | PyTorch/oneDNN staged AVX-512 | Median speedup |
|---|---:|---:|---:|---:|
| M=12, H=4096, F=512, E=1, top-k=1 hot | 1 | 1.4011 ms, 107.77 GFLOP/s | 2.1395 ms, 70.58 GFLOP/s | 1.53x |
| 16 tokens, 96 routes, H=4096, F=512, E=8, exactly M=12/expert | 1 | 14.7468 ms, 81.91 GFLOP/s | 21.2590 ms, 56.82 GFLOP/s | 1.44x |
| same balanced shape | 2 | 6.2416 ms, 193.53 GFLOP/s | 10.6699 ms, 113.21 GFLOP/s | 1.71x |
| M=48, H=4096, F=512, E=1, top-k=1 hot | 1 | 5.5710 ms, 108.41 GFLOP/s | 7.1809 ms, 84.11 GFLOP/s | 1.29x |
| same M=48 hot shape | 2 | 3.4653 ms, 174.29 GFLOP/s | 4.0685 ms, 148.45 GFLOP/s | 1.17x |

The middle balanced two-core process reached 207.39 GFLOP/s at its best sample.
The comparison measures the complete Python operator including routing,
gather, fused compute, route store, and merge; it is not a raw GEMM-only
microbenchmark. One-time custom prepack was about 8.5 ms for one expert and
68 ms for eight experts with the default serial prepack setting; neither value
is included in operator latency.

## Prepacked oneDNN C++ control

`benchmarks/bench_moe_onednn_avx512.cpp` constructs each oneDNN matmul with an
`any` weight descriptor, reorders W13 and W2 once into the implementation's
preferred blocked layouts, and excludes both reorders from timing. It still
materializes BF16 gate/up and gated-intermediate tensors, with the same poly5
SiLU/multiply between the two matmuls. Both primitive descriptors reported
`brg_matmul:avx512_core_bf16`.

```bash
ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16 OMP_NUM_THREADS=1 \
  taskset -c 0 benchmarks/bench_moe_onednn_avx512 12 4096 512 5 21
```

| M | Threads | Median | Best | Packed W13 + W2 |
|---:|---:|---:|---:|---:|
| 12 | 1 | 1.3169 ms, 114.66 GFLOP/s | 1.2918 ms, 116.89 GFLOP/s | 8 MiB + 4 MiB |
| 12 | 2 | 0.6756 ms, 223.50 GFLOP/s | 0.6428 ms, 234.90 GFLOP/s | 8 MiB + 4 MiB |
| 48 | 1 | 5.2097 ms, 115.93 GFLOP/s | 5.1722 ms, 116.77 GFLOP/s | 8 MiB + 4 MiB |
| 48 | 2 | 2.6145 ms, 231.01 GFLOP/s | 2.5914 ms, 233.07 GFLOP/s | 8 MiB + 4 MiB |

This is a lower-overhead compute control, not the same boundary as the Python
end-to-end table. At M=12, the custom end-to-end path is within about 6% of the
single-core prepacked oneDNN control while also doing routing, gather, direct
route placement, and merge. The custom kernel's value is fusion and operator
integration, rather than exceeding oneDNN's isolated prepacked GEMM control.

## Single-GEMM control

`benchmarks/bench_avx512_bf16_gemm.cpp` calls the custom W2
BF16-by-BF16-to-FP32 kernel directly. Custom B packing and oneDNN's `any`
weight reorder are both performed once outside timing. `kernel` also excludes
custom A packing; `pack A + kernel` includes the M12-to-physical-M16 VNNI2
packing on every iteration. Route ids are sequential, so both outputs are
contiguous. Results below are medians of 51 hot-weight samples pinned to core
0. oneDNN reported `brg_matmul:avx512_core_bf16` when capped to the same ISA.

| M | K | N | Custom kernel | Custom pack A + kernel | oneDNN AVX-512 BF16 |
|---:|---:|---:|---:|---:|---:|
| 12 | 512 | 4096 | 0.4384 ms, 114.82 GFLOP/s | 0.4397 ms, 114.46 GFLOP/s | 0.4445 ms, 113.24 GFLOP/s |
| 48 | 512 | 4096 | 1.7319 ms, 116.25 GFLOP/s | 1.7407 ms, 115.66 GFLOP/s | 1.7249 ms, 116.71 GFLOP/s |
| 12 | 4096 | 1024 | 0.8583 ms, 117.28 GFLOP/s | 0.8721 ms, 115.42 GFLOP/s | 0.8659 ms, 116.25 GFLOP/s |
| 48 | 4096 | 1024 | 3.4241 ms, 117.59 GFLOP/s | 3.4795 ms, 115.72 GFLOP/s | 3.4310 ms, 117.36 GFLOP/s |

The first geometry matches W2 for H=4096/F=512. The second has W13's combined
GEMM dimensions, but it still exercises `ComputeW2` and therefore excludes the
actual fused gate/up SiLU epilogue. Maximum absolute error versus oneDNN was
`4.47e-8`. The packed custom kernel is within -0.4% to +1.4% of prepacked
oneDNN across these shapes; including A packing keeps it within -1.4% to
+1.1%. The AVX-512 single-GEMM kernel is therefore effectively at oneDNN's
performance level on this core when M is an exact multiple of 12.

The generic final-panel path always computes 16 physical rows, so useful
performance is much lower when M is not divisible by 12:

| M | K | N | Custom kernel | oneDNN AVX-512 BF16 | Custom/oneDNN |
|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 4096 | 0.6632 ms, 6.32 GFLOP/s | 0.1411 ms, 29.72 GFLOP/s | 0.21x |
| 4 | 512 | 4096 | 0.6647 ms, 25.24 GFLOP/s | 0.1468 ms, 114.32 GFLOP/s | 0.22x |
| 8 | 512 | 4096 | 0.6727 ms, 49.88 GFLOP/s | 0.2903 ms, 115.59 GFLOP/s | 0.43x |
| 11 | 512 | 4096 | 0.6878 ms, 67.07 GFLOP/s | 0.4062 ms, 113.59 GFLOP/s | 0.59x |
| 13 | 512 | 4096 | 1.1258 ms, 48.43 GFLOP/s | 0.4788 ms, 113.89 GFLOP/s | 0.43x |

M=13 exposes the discontinuity particularly clearly: the executor runs one
efficient M12 panel and then pays for a full physical-M16 fallback to compute
the final row. Dedicated M8/M4/M2/M1 register kernels are required before the
same-ISA oneDNN parity extends to arbitrary expert route counts.

Without the ISA cap, oneDNN selected `brg_matmul:avx10_1_512_amx`:

| M | K | N | Custom AVX-512 kernel | oneDNN AMX | AMX/custom |
|---:|---:|---:|---:|---:|---:|
| 12 | 512 | 4096 | 114.86 GFLOP/s | 276.98 GFLOP/s | 2.41x |
| 48 | 512 | 4096 | 116.37 GFLOP/s | 335.47 GFLOP/s | 2.88x |
| 12 | 4096 | 1024 | 117.33 GFLOP/s | 339.10 GFLOP/s | 2.89x |
| 48 | 4096 | 1024 | 117.66 GFLOP/s | 510.53 GFLOP/s | 4.34x |

This large isolated-GEMM AMX advantage is partly hidden in the staged full
operator by routing, intermediate materialization, activation, and merge
costs, which do not receive AMX's matrix-compute speedup.

## AMX context and remaining work

With oneDNN's default AMX selection, the staged baseline remained about 10%
faster on the balanced shape: 89.14 versus 81.30 GFLOP/s on one core and
206.29 versus 186.40 GFLOP/s on two in representative 31/21-sample processes.
These numbers are context only: AMX and AVX-512 BF16 have different compute
ceilings.

Larger experts now loop over full M12 panels: M=48 reaches 108.41 GFLOP/s on
one core, within 6.5% of the isolated prepacked oneDNN AVX-512 control. The
two-core full operator reaches 174.29 GFLOP/s, leaving more scheduling and
integration overhead than the isolated 231.01 GFLOP/s control. The remaining
kernel gap is concentrated in final 1-11 row tails, which still use the
generic M-vector path; dedicated M8/M4/M2/M1 register kernels are the next
compute optimization.
