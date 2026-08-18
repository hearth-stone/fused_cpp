# Sparse MLA SVE PV contiguous output update

Date: 2026-08-17

## Decision

Adopt the contiguous PV output-update epilogue for SVE VL128 and VL256. It
replaces per-accumulator indexed gather/add/scatter with row-pair deinterleave
and predicated contiguous load/add/store. Both target vector lengths pass the
full Sparse MLA correctness file, all timed checksums match, and every measured
workload improves on both Neoverse V1 and Neoverse V3.

## Mechanism

The `8x2VL` PV kernel holds four two-column accumulators for each two-row pair.
The previous epilogue independently extracted each accumulator, constructed
indexed offsets, gathered the old output, added the new partial PV result, and
scattered it back. The adopted epilogue instead:

- uses one `UZP .d` level at VL128 or two levels at VL256 to reconstruct two
  contiguous output vectors per row;
- predicates the low and high vectors independently for the final Dv tile;
- performs contiguous `LD1W + FADD + ST1W` to retain online-softmax `O += PV`;
- retains the old indexed implementation for other vector lengths.

The BFMMLA reduction body, packed-P/V formats, online correction order,
normalization, dispatch, and public numerical behavior are unchanged. Object
disassembly on both targets confirmed 16 BFMMLA instructions followed by
ordinary predicated `LD1W/ST1W` in the supported-VL hot path.

## Correctness

- Amazon ECS 8C native SVL256, cores 0--3, four threads:
  `tests/test_sparse_mla.py` reported `26 passed`.
- Amazon ECS 8C forced SVL128, cores 4--7, four threads, existing `prctl`
  launcher: `tests/test_sparse_mla.py` reported `26 passed`.
- M5 native SVL128 cores 96--99: native-versus-naive dense, odd dense tail,
  shared-prefix, later sparse, output, and statistics checks passed.
- Every timed candidate checksum matched the QK-contiguous baseline.

## M5 Neoverse V3, SVL128

NUMA1 cores 96--191, 96 threads, close/core binding, BF16
`h_q=32,d_qk=192,d_v=128`, seed 20260817, ten warmups, 21 samples, and
candidate/baseline then baseline/candidate ordering:

| Case | Contiguous session medians | Indexed session medians | Change from median-of-medians |
|---|---:|---:|---:|
| 2048 prefix workload | 1.591 / 1.591 ms | 1.648 / 1.647 ms | -3.43% |
| Query length 8192 | 8.116 / 8.113 ms | 8.323 / 8.336 ms | -2.58% |
| Later low-overlap sparse | 2.279 / 2.268 ms | 2.322 / 2.321 ms | -2.07% |

The command was `numactl --physcpubind=96-191 --membind=1` plus
`OMP_NUM_THREADS=96 OMP_PROC_BIND=close OMP_PLACES=cores`, running
`/tmp/bench_sparse_mla_scalable.py --threads 96 --warmup 10 --iters 21
--seed 20260817`. The later case added `--context-start 10000`; the long-query
case added `--s-q 8192`.

A one-thread, one-call 2048 profile on core 96 reduced `pv` from 28.027 to
22.354 ms (-20.24%) and profile total from 103.691 to 97.885 ms (-5.60%). QK
was 36.118/36.305 ms and softmax was 19.252/19.239 ms, confirming that the
change is localized to the PV epilogue.

## Amazon 8C Neoverse V1, SVL256

Cores 0--7, eight threads, close/core binding, five warmups, 21 samples, and
the same forward/reverse binary order:

| Case | Contiguous session medians | Indexed session medians | Change from median-of-medians |
|---|---:|---:|---:|
| 2048 prefix workload | 20.409 / 20.501 ms | 21.650 / 21.643 ms | -5.50% |
| Query length 8192 | 117.187 / 117.222 ms | 124.261 / 124.425 ms | -5.74% |
| Later low-overlap sparse | 32.982 / 33.050 ms | 35.015 / 35.008 ms | -5.70% |

The command used `taskset -c 0-7`, `OMP_NUM_THREADS=8`, and
`/tmp/bench_sparse_mla_scalable.py --threads 8 --warmup 5 --iters 21
--seed 20260817`, with the same per-case arguments as M5.

## Interpretation

PV had the same address-generation and scatter backend problem previously
removed from QK, plus indexed loads of the old online output. Reconstructing
whole rows makes both sides of the update contiguous. The larger gain at
SVL256 is consistent with replacing more indexed operations per kernel call;
the smaller 96-thread M5 gain reflects dilution by packing, QK, softmax, and
parallel overhead even though the isolated PV slot improves by about 20%.
