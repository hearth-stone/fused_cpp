# Sparse MLA masked-tail comparison on Amazon 192-core Arm

> Follow-up (2026-08-12): this report's decision remains the conclusion for the
> masked 8x8 compute candidates. A later, independent guarded 2D KV scheduler
> was adopted for the public BF16 path; see
> `amazon_sparse_mla_2d_fused_20260812.md`. The indexed 4x4 kernel remains the
> correctness baseline and the no-split fallback.

## Decision

Keep the public/default `indexed_4x4` dispatch unchanged. Both masked 8x8
variants reduce single-thread latency by more than 7%, but neither improves the
2048-token forward once the operator reaches its approximately 55 ms scaling
plateau at 12 or more threads. The pruned variant is the stronger candidate,
but it has not met the multi-thread adoption gate.

## Compared variants

- `indexed_4x4`: existing planner decomposition and indexed 4x4 QK/scalar PV.
- `masked_dense_8x8`: promote safe monotonic ragged tails to a packed dense 8x8
  QK tile, mask each row before softmax, and use the existing packed BF16 PV.
- `masked_dense_8x8_pruned`: use the same tile, but omit inactive 2x2 BFMMLA
  blocks and invalid BFMLAL PV coefficient updates. A causal tile executes 10
  of 16 QK subblocks and 36 of 64 PV coefficients.

Both candidates reuse packed Q across the dense and masked segments of one
head. Generic sparse rows, duplicate indices, non-monotonic runs, partial
eight-query blocks, unsupported dtypes/shapes, and unsafe end-of-cache loads
remain on the indexed fallback.

## Method

- Host: `AmazonC5192Cores`, Neoverse V3, 192 logical CPUs, two NUMA nodes.
- Placement: NUMA node 1, `--physcpubind=96-191 --membind=1`.
- Build: GCC 15.2, `-O2 -march=armv8.6-a+sve+bf16+i8mm -fopenmp`, PyTorch
  2.13 CPU extension.
- OpenMP: `OMP_PROC_BIND=close`, `OMP_PLACES=cores`; requested threads shown
  below.
- Shape: BF16 `q[2048,32,192]`, `kv[2560,1,192]`, `d_v=128`.
- Indices: DeepSeek-V4-like compressed prefix `floor((token+1)/4)` plus a
  dense causal run of `token+1`, padded with negative int32 values to top-k
  2560.
- Valid attention pairs per head: 2,621,952; source-work count:
  53,697,576,960 FLOP.
- Promoted masked-tail pairs per head: 16,896, or 0.644% of valid pairs. The
  indexed tail is therefore disproportionately expensive at one thread, while
  its saved work is too small to move the shared dense-path scaling plateau.
- Timing: 5 warmups and 21 measured calls. Variant order rotates every measured
  iteration; reported latency is the median. The timed call includes int32 to
  int64 conversion, plan construction, packing, and forward computation.
- Output is preallocated for each variant. HugeTLB was configured on the host,
  but these ordinary PyTorch allocations were not explicitly HugeTLB-backed.

Command template:

```bash
OMP_NUM_THREADS=<T> OMP_PROC_BIND=close OMP_PLACES=cores \
numactl --physcpubind=96-191 --membind=1 env PYTHONPATH=src \
.venv/bin/python tests/bench_sparse_mla_tail_variants.py \
  --workload v4-forward --threads <T> --warmup 5 --iters 21
```

## Full-forward results

| Threads | indexed 4x4 ms | masked dense 8x8 ms | dense vs base | pruned 8x8 ms | pruned vs base |
|---:|---:|---:|---:|---:|---:|
| 1 | 586.217 | 544.225 | -7.16% | 541.370 | -7.65% |
| 12 | 55.710 | 55.832 | +0.22% | 55.949 | +0.43% |
| 32 | 54.702 | 55.402 | +1.28% | 55.500 | +1.46% |
| 96 | 55.281 | 55.426 | +0.26% | 55.352 | +0.13% |

Source-work throughput at 1 thread was 91.60, 98.67, and 99.19 GFLOP/s for
indexed, dense, and pruned respectively. At 96 threads it was 971.35, 968.82,
and 970.10 GFLOP/s. The 12/32/96 distributions overlap; those differences are
not evidence of a multi-thread improvement.

## Tail-only isolation

The 2048-token result above answers whether the tail change improves the whole
operator; it does not isolate the three tail implementations. A second
benchmark therefore removes every long dense segment and fixes 1,024 complete
eight-query blocks. Each block uses independent K/V rows so the result does not
depend on repeatedly hitting one eight-row tile in cache. The timed region still
includes plan construction, packing, softmax, and output accumulation, so this
is a tail-path benchmark rather than a bare-intrinsic loop.

`tail-causal` contains one `[1,2,3,4,5,6,7,8]` tile per query block:

| Threads | indexed 4x4 ms | masked dense 8x8 ms | pruned 8x8 ms | pruned vs dense |
|---:|---:|---:|---:|---:|
| 1 | 120.638 | 32.091 | 27.946 | -12.92% |
| 12 | 10.936 | 3.361 | 2.984 | -11.22% |
| 32 | 4.668 | 1.712 | 1.559 | -8.94% |
| 96 | 2.846 | 1.622 | 1.541 | -4.99% |

`tail-v4-mix` contains one causal tile plus one compressed tile per query block;
the compressed valid lengths cycle through the four V4 phases:

| Threads | indexed 4x4 ms | masked dense 8x8 ms | pruned 8x8 ms | pruned vs dense |
|---:|---:|---:|---:|---:|
| 1 | 222.502 | 112.268 | 105.780 | -5.78% |
| 12 | 19.803 | 10.394 | 9.788 | -5.83% |
| 32 | 8.230 | 4.509 | 4.278 | -5.12% |
| 96 | 3.728 | 2.278 | 2.171 | -4.70% |

Both use BF16 `h_q=32`, `d_qk=192`, `d_v=128`, 5 warmups, 21 rotating
interleaved measurements, and the same NUMA placement as the full-forward test.
The causal 96-thread samples were bimodal, so its median is less stable than the
other rows; the V4-mix 96-thread distribution was tight. Both candidates again
had maximum BF16 output difference 0.015625 versus indexed.

These isolated results show that 2x2 QK and exact-lane PV pruning are effective:
pruned consistently beats the full dense tile. Their small difference in the
2048-token full forward is dilution: masked tails account for only 0.644% of
valid pairs there, while the shared dense path determines multi-thread latency.

Tail-only command template:

```bash
OMP_NUM_THREADS=<T> OMP_PROC_BIND=close OMP_PLACES=cores \
numactl --physcpubind=96-191 --membind=1 env PYTHONPATH=src \
.venv/bin/python tests/bench_sparse_mla_tail_variants.py \
  --workload <tail-causal|tail-v4-mix> --tail-blocks 1024 \
  --threads <T> --warmup 5 --iters 21
```

The maximum BF16 output difference versus `indexed_4x4` was 0.015625 for both
masked variants. Their checksums matched each other. Focused correctness on the
target host passed:

```text
24 passed in 0.32s
```

Coverage includes causal lengths 1 through 8, all four compressed-tail masks,
uniform residual lengths 2/4/6 after a shared dense segment, empty rows,
duplicate/non-contiguous, partial-query, and unsafe end-of-cache fallbacks,
invalid selector/dtype/index rejection, output and statistics, and the existing
sparse MLA cases.

After extracting the ISA-specific code into
`csrc/sparse_mla_tail_microkernels.{h,cpp}`, a 2-warmup/5-run sanity check gave
584.684 / 543.118 / 540.956 ms at 1 thread and 54.429 / 54.762 / 55.241 ms at
12 threads (indexed/full/pruned). This smaller check is not used as the deciding
measurement; it verifies that the source move preserved the formal conclusion.

## Optimization attempts

| Attempt | Change | Observation |
|---|---|---|
| Full masked tile | Replaced indexed tails with packed full 8x8 QK/PV. | Initial 12-thread 5-run median was 53.827 ms versus 54.629 ms, but the formal interleaved run did not reproduce a gain. |
| Initial QK 2x2 pruning | Emitted only active BFMMLA subblocks. | Initial 12-thread median was 54.552 ms; scalar block stores and a 4-lane loop erased the instruction reduction. |
| QK scheduling and Q reuse | Added two-E-block unrolling, vector 2-lane stores, and reused packed Q across dense/masked tiles. | Quick 12-thread median improved to 54.288 ms versus 54.590 ms, still within noise. |
| Exact PV pruning | Skipped invalid BFMLAL coefficient updates in the recognized masks. | Causal PV work fell from 64 to 36 coefficients; the final one-thread result improved, but the multi-thread forward remained on the shared-KV plateau. |

## Residual risk and next decision

The candidates intentionally use the packed BF16 probability path for promoted
tails, so they are tolerance-equivalent rather than bitwise-equivalent to the
indexed fp32 scalar accumulation. They are AArch64/BF16 candidates; fp32 and
unsupported shapes use or require the indexed path. Before adoption, repeat the
21-run comparison in three independent sessions and either demonstrate a
single-thread dispatch policy with no default multi-thread change or meet the
multi-thread non-regression gate. Otherwise retire the comparison binding and
specialized kernels by 2026-09-11.
