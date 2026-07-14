# Elastic SVE fused GEMM experiment

This directory contains a standalone experiment. It does not change the fused
MoE API or default dispatch.

The experiment invokes the production `moe_sve_*_m12` assembly symbols with
three scheduling variants:

- `static`: one fixed contiguous N range per worker over the full M dimension.
- `epoch_fixed`: the same mapping, with M divided into fixed 12-row-aligned
  epochs. This isolates kernel fragmentation.
- `phase_claim_fixed` / `elastic_phase`: workers claim one N lane spanning every
  consecutive M epoch with the same lane count. This preserves each core's
  packed-B stripe affinity and has no barrier until the lane count changes.
- `strict_epoch_claim`: a stress reference that synchronizes every epoch.

An N lane always spans an entire M epoch. The experiment deliberately does not
schedule individual SVE N tiles because that would rescan packed A once per tile
instead of once per participating lane.

The benchmark reports three separate costs:

- `fragmentation_over_static_pct`: splitting M into epochs without a barrier.
- `phase_claim_over_epoch_fixed_pct`: atomic lane claiming with barriers only
  at real lane-count changes.
- `strict_claim_over_epoch_fixed_pct`: the cost of synchronizing every epoch.
- `elastic_phase_over_piecewise_phase_pct`: changing the lane count after
  accounting for the fixed-low and fixed-high phase-claim costs.

Use `--epoch-rows 1020` with `M=2040` for one mid-GEMM resize point, or
`--epoch-rows 204` for a ten-epoch synchronization stress test. Input and
packed-weight buffers rotate across copies during timing.

Build and check on an SVE BF16 AArch64 machine:

```bash
make -C optimizations/fused_moe_sve/benchmarks
make -C optimizations/fused_moe_sve/benchmarks check
```

Run the default 96-core NUMA-local sweep:

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  python3 optimizations/fused_moe_sve/benchmarks/run_sweep.py
```

The measured 192-core-host NUMA0 results are recorded in
[`results/amazon_192c_numa0.md`](results/amazon_192c_numa0.md).
