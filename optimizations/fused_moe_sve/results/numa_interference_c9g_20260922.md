# What concurrent work on C9g's other NUMA node costs a node-0 measurement

## Status

Measured 2026-09-22. E10's rule transfers to this machine unchanged: compute-bound work on
the foreign node is free, memory streaming on it costs about 1%. A second result came out of
checking the baseline rather than the treatment - this machine's session-to-session
reproducibility is 0.1%, and the one session that appeared to contradict that was
contaminated by the measuring session that preceded it, through an idle guard that only
watches the 1-minute load average.

## Machine

Amazon C9g, Neoverse-V3, 192 cores in two NUMA nodes (node0 `0-95`, node1 `96-191`), 193 GB
per node, L3 192 MiB in two instances of 96 MiB, one per node, SVE-128 so the packed-B tile
is 8. Measurements run on node 0 with `numactl --physcpubind=0-95 --membind=0`.

This is not the instance that produced the 2026-09-21 C9g results; that one was
reprovisioned. Same specification, different machine, and gcc 15.2 rather than the earlier
toolchain.

## Why repeat E10 here

E10 measured this on `Arm-codex` and the rule went into
`docs/agent_benchmark_hygiene.md`. Every C9g measurement since has assumed it transfers,
which is not obvious: two nodes of 96 cores against four of 80, one 96 MiB LLC instance per
node against two 70 MiB instances per node (so 96 cores share one slice here against 40
there), one socket rather than two, and a per-domain weight footprint 2-4x `Arm-codex`'s.

## Design (frozen before collection)

`tmp/c9g_numa_20260922/design.md`. The measured job is the homogeneous full-load grid already
validated on this machine, restricted to twelve cells: widths 2, 4, 8 x routes 96, 192 x W13
window 0 (full stripe) and 1 tile, TP4 (F=512), 144 experts, 3 warmup + 31 runs, producer-hot
A, cold B, jemalloc never-purge, every session passing the benchmark's bitwise-equality gate.
Background is E10's unchanged `background.py` on node 1 only, 96 threads: `stream` is float32
`a += b` over 2 GiB buffers, `gemm` is BF16 4096^3 matmul. Three conditions - `idle`,
`n1_gemm`, `n1_stream` - two sessions each, second pass in reverse order.

Thresholds, frozen with the design and the same bands E10 used: at most 0.6% no measurable
effect, 0.6-2% measurable, above 2% material.

## The baseline had to be measured before the result could be read

The two idle sessions, first and last, differed by a median 0.68% per cell, which is the size
of the effect being measured. Four further idle sessions were then run back to back, 45 s
apart, with nothing between them:

| | per-cell spread over four sessions | session-to-session median |
| --- | --- | --- |
| four clean back-to-back sessions | median 0.49%, max 1.14% | -0.36% to +0.36%, mostly within 0.1% |

So 0.1% is this machine's session-to-session floor, comparable to `Arm-codex`'s 0.18%. The
0.68% was not a machine property. Scoring every session of the day against the median of the
four clean ones locates it:

| session | median against the clean baseline | per-cell range |
| --- | --- | --- |
| the four clean sessions | -0.03%, -0.01%, +0.11%, -0.02% | within 0.9% |
| **NUMA idle #1 (first)** | **+0.54%** | **-1.95% to +1.53%** |
| NUMA idle #6 (last) | -0.02% | within 0.3% |
| NUMA gemm, two sessions | -0.06%, +0.15% | |
| NUMA stream, two sessions | +1.06%, +0.90% | |

Idle #1 is the outlier, not idle #6. It started 27 s after an unrelated grid run of mine had
finished. The chain's `wait_idle` admitted it because the 1-minute load average had fallen to
0.39, while the 5-minute average was still 3.10. **A 1-minute load average is too fast a
filter to separate a session from the work immediately before it.**

Before this was checked, the 0.68% was read as a property of C9g and explained at length -
whether it was thermal, a within-session ramp, or physical page placement. Those analyses were
of one contaminated sample. Within-session stability was never in question: the late eight
runs of a cell differ from its early eight by 0.13% and -0.09% in the two sessions.

## Results

Each condition's two sessions against the median of the four clean idle sessions, over the
twelve cells:

| background on node 1 | session A | session B | cells slower | verdict |
| --- | --- | --- | --- | --- |
| BF16 matmul, 96 threads | -0.06% | +0.15% | 4 of 12 | no measurable effect |
| memory streaming, 96 threads | **+1.06%** | **+0.90%** | **12 of 12** (min +0.12%) | measurable |

Against `Arm-codex`: matmul +0.01% / +0.24%, one node of streaming +0.83% (same socket) /
+1.84% (other socket). The per-node streaming cost is the same to within the measurement.

## Reading

- **The rule transfers.** Builds, tests, planner searches and analysis may run on node 1
  during a node-0 measurement. Memory streaming may not.
- **The mechanism is not this machine's LLC.** C9g packs 96 cores onto one 96 MiB LLC slice
  against `Arm-codex`'s 40 on 70 MiB, and its per-domain weight footprint is 2-4x, yet the
  cross-node streaming cost is the same. The contention is on the path between nodes - memory
  controllers and interconnect - not in the last-level cache, which is a different mechanism
  from the footprint effect that governs window value on this machine.
- **The effect is now well separated from the floor.** At a 0.1% session-to-session floor, a
  1% effect with 12 of 12 cells in the same direction is unambiguous. Read against the
  contaminated baseline it had looked marginal.

## Protocol change

`wait_idle` now requires the 5-minute load average to have settled as well as the 1-minute
one, and a fixed cooldown before the first measured session of a chain. Recorded in
`docs/agent_benchmark_hygiene.md`.

## Limitations

- Twelve grid cells at one shape (TP4, F=512), not whole plans on real routing layers, which
  is what E10 used on `Arm-codex`. The cells cover both windowed and unwindowed geometries at
  the widths the planner uses.
- Two sessions per condition. The gemm result rests on both sessions landing inside the clean
  baseline's own scatter rather than on a tight confidence interval.
- Core frequency was not recorded on this machine; E10 established for `Arm-codex` that node-3
  cores hold maximum frequency under these backgrounds, and that check was not repeated here.
