# Sparse MLA Scalable SVE 8x2VL (2026-08-14)

## Scope

This record tracks the staged replacement of the fixed NEON 8-head by 8-key
Sparse MLA kernel:

1. fixed NEON QK/PV 8x8 baseline;
2. scalable SVE BFMMLA QK 8x2VL with the existing NEON PV path;
3. scalable SVE BFMMLA QK/PV 8x2VL.

This checkpoint covers stages 1 and 2. The optimized public entrypoint and its
tensor/API contracts are unchanged. On an SVE BF16 build, the gathered sparse
head-major path uses `N=svcntb()/2`, which is 8 keys at SVL128 and 16 keys at
SVL256. The shared-dense head-major subpath and PV remain fixed 8x8 at this
checkpoint. Non-SVE builds retain the fixed NEON implementation.

Primary change class: O (production optimization). Numerical behavior remains
within the existing BF16 tolerance. The rollback boundary is the scalable QK
header and its call-site packing changes.

## Implementation

- Q remains packed as four 2-row panels for every four reduction elements.
- K is packed into four scalable B vectors per K=4 block. Each independent
  128-bit SVE segment holds one 4x2 BF16 panel.
- Sixteen FP32 SVE accumulators compute four row pairs by four column pairs.
  BFMMLA operates independently in each 128-bit segment, so the instruction
  and register topology is the same for every supported VL.
- The result epilogue unzips each BFMMLA accumulator and scatters it into an
  8-by-2VL row-major score tile. Online softmax and the materialized BF16
  probability/PV path are otherwise unchanged.
- K/V gather tiles and L2 chunks are rounded to the runtime key tile. Packing
  and computation are worker-local and use the same runtime VL.

## Method

- Host: `AmazonECS8Cores`, Neoverse-V1, CPUs 0--7, Linux AArch64.
- Build: release `_C`, GCC 12.4, `-O2 -march=armv8.6-a+sve+bf16+i8mm`,
  profiling disabled, and no fixed `-msve-vector-bits` on `sparse_mla.cpp`.
- Baseline source: fixed NEON head-major 8x8 at commit `e92e155` (measured at
  `99291fe`, whose intervening change only detects the maximum build-time VL).
- Workload: BF16 `q[2048,32,192]`, `kv[2560,1,192]`, `d_v=128`, DSV4-like
  capacity 640 (`512` compressed plus `128` recent), `context_start=0`,
  compression ratio 4, and 777,792 valid KV pairs per head.
- Source work: 15,929,180,160 FLOPs. The timed public call includes index
  validation/conversion, Q/K/V packing, QK, online softmax, PV, output
  conversion, and OpenMP scheduling.
- Inputs use seed `20260814`. Each result has 5 warmups and 21 timed samples;
  the median is reported. CPU and library thread counts are pinned to either
  CPU 0 or CPUs 0--7.
- Benchmark command:

```bash
env PYTHONPATH=src OMP_NUM_THREADS=<threads> \
  FUSED_CPP_NUM_THREADS=<threads> OMP_DYNAMIC=FALSE \
  OMP_PROC_BIND=close OMP_PLACES=cores MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  taskset -c <cores> .venv/bin/python \
  tests/bench_sparse_mla_scalable.py --threads <threads> \
  --warmup 5 --iters 21 --seed 20260814
```

For SVL128 validation, the process called `prctl(PR_SVE_SET_VL, 16)` before
importing PyTorch or `_C`; the launcher verified `PR_SVE_GET_VL == 16`. The
fixed-SVL256 `_moe_C` module was replaced only in that test process by an empty
module because Sparse MLA uses the independently built scalable `_C` module.

## Results

| Kernel / runtime VL | Threads | Median | Range | Source throughput | Versus fixed 8x8 |
|---|---:|---:|---:|---:|---:|
| fixed NEON QK/PV 8x8 | 1 | 252.176 ms | 251.803--252.595 | 63.17 GFLOP/s | baseline |
| SVE QK 8x8 + NEON PV, SVL128 | 1 | 250.760 ms | 250.357--251.348 | 63.52 GFLOP/s | 1.0056x, -0.56% latency |
| SVE QK 8x16 + NEON PV, SVL256 | 1 | 225.232 ms | 224.983--225.764 | 70.72 GFLOP/s | 1.1196x, -10.68% latency |
| fixed NEON QK/PV 8x8 | 8 | 38.508 ms | 38.210--39.283 | 413.66 GFLOP/s | baseline |
| SVE QK 8x8 + NEON PV, SVL128 | 8 | 38.313 ms | 38.175--38.567 | 415.77 GFLOP/s | 1.0051x, -0.51% latency |
| SVE QK 8x16 + NEON PV, SVL256 | 8 | 36.759 ms | 36.633--37.024 | 433.34 GFLOP/s | 1.0476x, -4.54% latency |

The SVL128 form is effectively neutral relative to fixed NEON 8x8, while
SVL256 amortizes packing and loop overhead over twice as many keys. The smaller
8-thread gain shows that QK is only one part of the full public call; gather,
softmax, the unchanged PV, and output work dilute the single-core QK gain.

The BF16 output checksum differs slightly because the QK accumulation and
online-softmax tile boundary changed: fixed 8x8 was `4257.796875` at 1 thread,
SVL128 was `4257.795410`, and SVL256 was `4257.775391`. Focused reference tests
below validate the supported numerical tolerance.

## Correctness and Code Generation

- Default SVL256, CPUs 0--3, four OpenMP threads:
  `pytest -q tests/test_sparse_mla.py` -> `24 passed` in 27.53 seconds.
- Forced SVL128 with the same affinity and thread count: the same suite ->
  `24 passed` in 26.87 seconds.
- The generated `qkt_8x2vl_bf16` reduction loop contains 16 `bfmmla`
  instructions per K=4 step. Its 16 FP32 accumulators remain in SVE registers;
  the disassembly has no stack frame or SVE register spill.

## Raw Samples

```text
fixed_1t_svl256=251.971,252.410,252.050,252.498,251.989,252.216,251.803,252.202,251.827,252.550,252.079,252.539,252.061,252.357,252.176,252.595,251.899,252.243,251.860,252.461,251.979
fixed_8t_svl256=38.210,38.508,38.293,38.521,38.325,38.531,38.298,38.547,38.279,38.526,38.284,38.552,38.303,38.530,38.248,39.283,38.647,38.557,38.282,38.562,38.274
sve_qk_1t_svl128=250.571,251.061,250.600,250.760,250.576,251.063,250.653,251.010,250.466,251.131,250.410,251.098,250.801,251.152,250.528,251.142,250.687,251.348,250.453,251.106,250.357
sve_qk_8t_svl128=38.205,38.493,38.175,38.550,38.252,38.531,38.313,38.529,38.282,38.567,38.259,38.424,38.266,38.523,38.183,38.505,38.221,38.439,38.185,38.436,38.304
sve_qk_1t_svl256=225.158,225.689,225.168,225.545,225.148,225.764,225.073,225.592,224.983,225.588,225.165,225.671,225.232,225.595,225.116,225.536,225.145,225.754,225.102,225.544,225.071
sve_qk_8t_svl256=36.739,36.993,36.653,36.927,36.712,36.955,36.737,37.024,36.633,36.951,36.711,36.940,36.759,37.009,36.707,36.963,36.733,36.948,36.705,36.991,36.741
```

## Checkpoint Decision

Adopt stage 2 on SVE BF16 builds. It meets both-VL correctness, is neutral at
SVL128, and improves the representative SVL256 workload. Keep fixed NEON 8x8
as the non-SVE fallback. The next isolated checkpoint replaces the remaining
PV path with a scalable 8x2VL BFMMLA implementation and compares all three
stages before deciding the final default.
