# Sparse MLA SVE QK contiguous score store

Date: 2026-08-17

## Decision

Adopt the contiguous QK score-store epilogue for SVE VL128 and VL256. It
replaces per-accumulator indexed scatter stores with row-pair deinterleave and
contiguous stores, preserves bit-identical timed checksums, passes the complete
Sparse MLA correctness file at both vector lengths, and improves every measured
workload on both Neoverse V1 and Neoverse V3.

## Mechanism

The `8x2VL` QK kernel has four column-pair accumulators for each two-row pair.
The old epilogue independently extracted each accumulator's two rows, built
vector indexes, and issued two scatter stores. The new epilogue follows the
adopted i8gemm M12 technique:

- VL128: one `UZP .d` level joins adjacent column-pair accumulators into two
  contiguous vectors per row.
- VL256: a second `UZP .d` level groups the two 128-bit segments into low/high
  N8 vectors per row.
- Other vector lengths retain the original scatter fallback.

Scale multiplication remains before the bitwise deinterleave. QK BFMMLA,
packed layouts, score order, score-copy/max fusion, softmax, packed-P, PV,
dispatch, and public contracts are unchanged. GCC initially outlined the larger
helper; `always_inline` is required to match the reference macro. M5 assembly
inspection confirmed inlined `UZP` plus ordinary `ST1W` on the VL128 hot path.

The first literal two-level port failed the VL128 numerical check because that
layout has only one 128-bit segment. It was not benchmarked. Splitting VL128
and VL256 deinterleave depth fixed the ordering before performance collection.

## Correctness

- Amazon ECS 8C native SVL256, cores 0--3, four threads:
  `tests/test_sparse_mla.py` reported `26 passed`.
- Amazon ECS 8C forced SVL128, cores 4--7, four threads, existing `prctl`
  launcher: `tests/test_sparse_mla.py` reported `26 passed`.
- M5 native SVL128 cores 96--99: native-versus-naive dense, odd dense tail,
  shared-prefix, later sparse, output, and statistics checks passed.
- Every timed checksum matched its scatter baseline.

## M5 Neoverse V3, SVL128

NUMA1 cores 96--191, 96 threads, close/core binding, BF16
`h_q=32,d_qk=192,d_v=128`, seed 20260817, ten warmups, 21 samples, three paired
sessions with reversed order:

| Case | Contiguous session medians | Scatter session medians | Stable change |
|---|---|---|---:|
| 2048 shared-prefix | 11.041 / 11.055 / 11.057 ms | 11.392 / 11.376 / 11.389 ms | -2.93% |
| 8192 shared dense | 8.094 / 8.112 ms in stable low mode | 8.740 / 8.771 ms | about -7.4% |
| Later low-overlap sparse | 2.326 / 2.330 / 2.326 ms | 2.483 / 2.490 / 2.491 ms | -6.59% |

The third 8192 contiguous session switched between the known machine modes and
is not used for the quantified gain.

A one-thread, one-call profiling diagnostic on 2048 shared-prefix reduced
`qkt_total` from 160.175 to 125.254 ms (-21.80%). Profile total changed from
325.637 to 289.752 ms (-11.02%); softmax stayed 64.76 ms and PV stayed near
95 ms. Profiling overhead means these values are attribution evidence, not the
primary wall-time result.

## Amazon 8C Neoverse V1, SVL256

Five warmups and 21 samples for eight-thread forward/reverse order; one-thread
screening used nine samples:

| Case | 8T contiguous | 8T scatter | 8T change | 1T contiguous | 1T scatter | 1T change |
|---|---:|---:|---:|---:|---:|---:|
| 2048 shared-prefix | 68.823 / 68.812 ms | 73.050 / 73.288 ms | -5.79% / -6.11% | 381.617 ms | 423.819 ms | -9.96% |
| 8192 shared dense | 116.962 / 116.669 ms | 125.850 / 125.989 ms | -7.06% / -7.40% | 733.210 ms | 818.668 ms | -10.44% |
| Later low-overlap sparse | 35.007 / 35.015 ms | 37.428 / 37.477 ms | -6.47% / -6.57% | 220.521 ms | 240.109 ms | -8.16% |

## Interpretation

This confirms the i8gemm report's diagnosis in Sparse MLA: the backend cost of
scatter/address generation was substantial even though total score bytes are
small and cache-resident. The gain is largest at one thread and SVL256, where
the two-level deinterleave replaces the greatest number of scatter operations.
It also explains why cache-panel and B-load reuse experiments were weak: they
did not address the score-store backend bottleneck.
