# Sparse MLA Shared-Prefix K/V Packing (2026-08-14)

## Decision

Enable call-level shared K/V packing for the V4 select-all prefill shape. The
detector accepts up to two fixed-start contiguous prefixes, corresponding to
the compressed cache and the causal/SWA cache, and requires both maximum
lengths to be complete runtime 2VL tiles. Query-dependent TopK, a sliding SWA
start, unsupported shapes, non-SVE builds, and non-beneficial inputs retain the
existing per-token gather-pack fallback.

Primary change class: O (production optimization). The public API, default
entrypoint, packed ABI, indices semantics, and fallback behavior are unchanged.
The rollback boundary is the commit that adds
`packing.shared_prefix_kv_2vl`.

The adoption gate is: both SVL128 and SVL256 correctness pass; the production
2048-token shape improves by at least 5% at one and eight threads; the later
query-dependent sparse shape remains within 1% of baseline. All gates pass.

## Method

- Host: `AmazonECS8Cores`, Neoverse-V1, Linux AArch64, CPUs 0--7.
- Runtime VL: native SVL256 for performance; correctness also forced SVL128
  with `prctl(PR_SVE_SET_VL, 16)` before importing the extension.
- Build: release `_C`, GCC 12.4, C++17, `-O2`, SVE BF16 enabled.
- Baseline: commit `b6842d8`, saved before changing shared-prefix packing.
- Candidate: baseline plus `packing.shared_prefix_kv_2vl`; the installed
  candidate and saved candidate shared object had the same SHA-256.
- Inputs: seed `20260814`, BF16 `q[2048,32,192]`, `d_v=128`, compression
  ratio 4, 512 compressed positions, and a 2048-token causal/SWA prefix.
  The combined KV has 2,560 rows and 2,621,952 valid pairs per head.
- Timing: public `flash_mla_sparse_fwd`, 5 warmups, 21 measured iterations,
  median reported. One thread was bound to CPU 0; eight threads to CPUs 0--7.
  OpenMP dynamic teams and dependent library threading were disabled.

Benchmark command:

```bash
OMP_NUM_THREADS=<1|8> OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
taskset -c <0|0-7> .venv/bin/python tests/bench_sparse_mla_scalable.py \
  --pattern sparse --s-q 2048 --h-q 32 --d-qk 192 --d-v 128 \
  --compressed-capacity 512 --window-size 2048 --compress-ratio 4 \
  --context-start 0 --threads <1|8> --warmup 5 --iters 21 \
  --seed 20260814
```

## Results

| Threads | Baseline median | Candidate median | Latency change | Candidate source throughput |
|---:|---:|---:|---:|---:|
| 1 | 552.018 ms | 451.861 ms | **-18.14%** | 118.84 GFLOP/s |
| 8, first session | 97.476 ms | 76.860 ms | **-21.15%** | 698.64 GFLOP/s |
| 8, reverse-order repeat | 82.702 ms | 76.929 ms | **-6.98%** | 698.01 GFLOP/s |

The eight-thread baseline changed performance state between sessions, while the
candidate remained at 76.9 ms. The conservative paired result is therefore a
6.98% latency reduction, still above the adoption gate. The one-thread samples
were stable within roughly 1.4 ms end to end.

The candidate checksum differs slightly (`-4358.31` versus `-4358.26`) because
the two logical segments form separate online-softmax chunks. Direct comparison
with the naive reference passes the existing BF16 output and statistics
tolerances at both runtime vector lengths.

## Later Sparse Guardrail

For `context_start=10000`, compressed capacity 512, and sliding window 128, the
detector rejects the query-dependent rows and uses the original gather-pack
path:

| Threads | Baseline | Candidate | Change |
|---:|---:|---:|---:|
| 1 | 279.778 ms | 279.899 ms | +0.04% |
| 8 | 41.668 ms | 41.586 ms | -0.20% |

This is within measurement noise and passes the 1% guardrail.

## Correctness

- Focused shared-prefix/head-major cases, native SVL256: `6 passed`.
- Full `tests/test_sparse_mla.py`, native SVL256, OMP=4, CPUs 0--3:
  `26 passed` in 26.70 seconds.
- The same full file at forced SVL128, OMP=4, CPUs 4--7:
  `26 passed` in 28.85 seconds.
- Coverage includes two prefix segments, negative padding, optional sink,
  output-only and exact statistics, ragged/duplicate sparse indices,
  unsupported-shape fallbacks, and positive out-of-range rejection.

## Raw Samples

```text
baseline_1t=551.504,552.256,551.383,551.757,552.018,552.139,551.613,552.178,551.355,552.014,551.784,552.028,552.524,552.205,552.158,552.386,552.172,552.603,551.706,551.981,551.823
candidate_1t=452.716,451.626,452.026,451.855,451.872,451.589,451.810,451.861,452.399,451.559,452.515,451.357,452.401,451.344,452.524,451.578,452.036,451.808,452.378,451.717,452.069
baseline_8t_repeat=82.683,82.721,82.658,82.722,82.726,82.737,82.789,82.679,82.724,82.681,82.702,82.663,82.673,82.703,82.642,82.728,82.643,82.694,82.652,82.797,82.772
candidate_8t_repeat=76.798,76.841,76.978,77.062,77.218,76.729,77.184,77.223,76.999,76.536,77.118,77.150,77.203,76.731,76.569,76.929,77.030,76.918,76.904,76.479,76.619
```
