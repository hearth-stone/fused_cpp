# What concurrent work on the other NUMA nodes costs a node-3 measurement (E10)

## Status

Measured, with a rule for the benchmark protocol: compute-bound work elsewhere on the
machine is free, memory-streaming work elsewhere is not. The rule is in
`docs/agent_benchmark_hygiene.md` ("Machine Sharing And Other NUMA Nodes"); this report
carries the evidence.

## Machine

`Arm-codex`, HiSilicon, 2 sockets x 160 cores, four NUMA nodes of 80 cores (node0 `0-79`,
node1 `80-159`, node2 `160-239`, node3 `240-319`), 512 GB per node, L3 560 MiB in 8
instances (70 MiB each, two per node). Nodes 2 and 3 share a socket, so node 2 is node 3's
socket neighbour and nodes 0/1 are across the inter-socket link. All project measurements
run on node 3 with `numactl --physcpubind=240-319 --membind=3`.

## Design (frozen before collection)

`tmp/numa_interference_20260920/design.md`. The measured job is the unchanged plan
benchmark - `tmp/m2_validation_20260919/group_w/bench.py` on
`tmp/search_reliability_20260920/plans_e6_r008.json`, two real route layers, 11 plans each,
22 points, 5 warmup + 31 runs, producer-hot A, cold B, jemalloc never-purge - on node 3.
Background processes run `background.py` placed on other nodes only, since the measured
plans already own all 80 cores of node 3: `stream` is float32 `a += b` over 2 GiB buffers
(266 GB/s per node with 80 threads), `gemm` is BF16 4096^3 matmul (3.91 TFLOP/s).

Two sessions per condition, the second pass in reverse condition order, 12 sessions,
23.2 min, all rc=0, every session passing the benchmark's bitwise-equality gate. Idle plan
times 26.91-28.96 ms; the two idle sessions, first and last, differ by a median 0.18%
(max 0.81%).

Thresholds frozen with the design: <= 0.6% no measurable effect (the protocol's
cross-batch bound), 0.6-2% measurable, > 2% material.

## Results

| background (other nodes only) | median | p10 | p90 | max | verdict |
| --- | --- | --- | --- | --- | --- |
| node 2 gemm (same socket) | +0.01% | -0.34% | +0.35% | +0.59% | no measurable effect |
| node 1 gemm (other socket) | +0.24% | -0.09% | +0.72% | +0.80% | no measurable effect |
| node 2 stream (same socket) | +0.83% | +0.61% | +1.08% | +1.38% | measurable |
| node 1 stream (other socket) | +1.84% | +1.40% | +2.25% | +2.28% | measurable |
| nodes 0+1+2 stream | +13.09% | +11.78% | +13.71% | +14.89% | material |

Per round, each condition's two sessions against the mean of the two idle sessions:
n2_gemm +0.13 / -0.25, n1_gemm +0.70 / -0.05, n2_stream +1.03 / +0.56, n1_stream +2.27 /
+1.29, n012_stream +12.52 / +13.38. Signs and ordering reproduce across the rounds; the
streaming magnitudes drift by about one point between them.

## Reading

- A whole foreign node running BF16 matmul at 3.9 TFLOP/s leaves node 3's plan times
  unchanged. Builds, tests, planner searches and analysis may run during a measurement.
- Memory streaming elsewhere costs 0.8-1.8% per node and 13% for three nodes, although the
  measured job's memory is entirely node-3 local.
- The cross-socket node costs consistently more than the socket neighbour, so the effect is
  not a simple "same socket is worse" picture. That ordering is unexplained.
- Frequency is not the mechanism: node-3 cores sampled under the three-node streaming
  background, with node 3 itself busy, stay at the 2900 MHz maximum. The contention is in
  the memory path shared beyond the node.
- For scale, the discarded E6 r022 sessions measured 79-100 ms against 27-28 ms (+190%),
  far beyond anything foreign nodes cause here - consistent with that foreign job holding
  cores on node 3 itself.

## Limits

One machine, one plan set, two sessions per condition. The streaming background is a
synthetic bandwidth hog, not a real workload mix, and its per-session rate was not
recorded: the background processes were killed at the end of each session before they
could write their own report, so the 266 GB/s and 3.91 TFLOP/s figures come from the smoke
runs of the same generator. The machine's load average during the streaming sessions
(76-222) confirms the background ran.

## Artifacts

`tmp/numa_interference_20260920/`: `design.md`, `decision.md`, `chain_e10.sh`,
`background.py`, `analyze_e10.py`, 12 session JSON/log files, job state.
