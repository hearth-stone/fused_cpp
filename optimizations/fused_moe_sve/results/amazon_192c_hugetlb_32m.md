# Explicit 32 MiB HugeTLB packed weights on AmazonC5192Cores

Date: 2026-08-01

## Configuration

- Host: `AmazonC5192Cores`, two 96-core NUMA nodes.
- HugeTLB page size: 32 MiB.
- Reservation: 160 pages on NUMA0 and 160 pages on NUMA1, 10 GiB total.
- Mount: `/dev/hugepages-32M`, `hugetlbfs pagesize=32M,mode=1777`.
- Persistence: `fused-cpp-hugepages.service` reserves and mounts the pool at boot.
- Default benchmark environment:
  `FUSED_CPP_MOE_HUGETLBFS_PATH=/dev/hugepages-32M` in `/etc/environment`.
- Scope: only reusable packed W13/W2 tensors move to HugeTLB storage. Dense
  source weights, inputs, outputs, and temporary scratch retain the normal
  allocator.

The Python wrapper treats an explicitly configured pool as strict. A missing
mount or exhausted pool raises an error instead of silently using 4 KiB pages.
Unset `FUSED_CPP_MOE_HUGETLBFS_PATH` for a named ordinary-page baseline.

## Mapping validation

The production TP4 shape `H=4096,F=512,E=256` consumed 96 pages on NUMA0:

| Tensor | Bytes | 32 MiB pages | `KernelPageSize` | `Private_Hugetlb` |
| --- | ---: | ---: | ---: | ---: |
| packed W13 | 2 GiB | 64 | 32768 kB | 2 GiB |
| packed W2 | 1 GiB | 32 | 32768 kB | 1 GiB |

Both mappings were hugetlbfs-backed and unlinked after `mmap`; the tensor owns
the live mapping. The corresponding 4 KiB baseline reported
`KernelPageSize=4 kB`, `AnonHugePages=0`, and no `hg` mapping flag.

## Correctness

```bash
PYTHONPATH=src FUSED_CPP_MOE_HUGETLBFS_PATH=/dev/hugepages-32M \
  OMP_NUM_THREADS=1 numactl --cpunodebind=0 --membind=0 taskset -c 0-31 \
  .venv/bin/python -m pytest -q \
  tests/test_fused_moe_bf16_tiled.py::test_sve_m12_silu_and_w2_bf16_route_match_legacy_for_unit_top1
```

Result: `1 passed`. The storage helper unit tests also passed on local and
target environments: `6 passed`.

## M1 one-wave comparison

The comparison used the SVE JIT exact-M split-W13 path, `H=4096,F=512,E=256`,
one M1 route and one thread per active expert, distinct rotating weights,
contiguous NUMA0 CPUs, 12 warmups, and 101 measured runs. Positive values mean
32 MiB HugeTLB was faster.

| Concurrent experts | 4 KiB median | 32 MiB median | Change |
| ---: | ---: | ---: | ---: |
| 1 | 0.3844 ms | 0.3810 ms | +0.89% |
| 4 | 0.5495 ms | 0.5331 ms | +2.98% |
| 8 | 0.7238 ms | 0.7285 ms | -0.66% |
| 12 | 0.7662 ms | 0.7574 ms | +1.16% |
| 16 | 0.8437 ms | 0.8373 ms | +0.75% |
| 24 | 1.0780 ms | 1.0786 ms | -0.06% |
| 32 | 1.4106 ms | 1.3852 ms | +1.81% |
| 48 | 1.9162 ms | 1.8943 ms | +1.15% |
| 64 | 2.4445 ms | 2.4242 ms | +0.83% |
| 80 | 2.9699 ms | 2.9408 ms | +0.98% |
| 96 | 3.3831 ms | 3.3557 ms | +0.81% |

The M1 curve does not show a large-page step change. At high concurrency the
gain is consistently around one percent; low-concurrency differences are on
the same scale as run-to-run noise. The target-host default is enabled to make
future TLB/cache experiments use a stable explicit page policy, not because
this single workload establishes a large general speedup.

## Capacity boundary

Each NUMA node has 5 GiB in the default pool. This supports one TP4
`E=256,F=512` packed-weight set (3 GiB) per node. Shapes whose packed W13+W2
exceed 5 GiB, or multiple simultaneous preparations on one node, fail
explicitly and require a deliberate page-pool rebalance or larger reservation.
