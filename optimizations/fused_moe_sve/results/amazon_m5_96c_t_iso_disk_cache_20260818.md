# Analytical T_iso Disk Cache on AmazonM5192Cores

## Decision

Enable the versioned scalar `T_iso(M,T)` disk cache by default for
`MoePlannerRuntime`. The cache changes only how analytical quick-planner costs
are obtained; it does not persist final plans or change scheduling semantics.

## Method

- Machine: AmazonM5192Cores NUMA0, CPUs 0-95, memory node 0.
- Shape: DSV4 TP2, H4096/F1024/E256, 2048 tokens, TopK6.
- Routing: captured `dsv4-real-2048-seq70`, 223 active experts and 28 distinct
  route counts.
- Candidate set: eight homogeneous quick-planner shapes.
- Measurement: two independent Python processes using one initially absent
  temporary cache directory. Both materialized the full Plan V2 and hashed its
  task, dependency, and stage-window tensors.

## Result

| Process | Cache | Entries | Runtime init | Plan wall | Internal search |
| --- | --- | ---: | ---: | ---: | ---: |
| First | miss then store | 224 | 2.292 ms | 21.982 ms | 19.273 ms |
| Second | disk hit | 224 | 1.930 ms | 2.636 ms | 1.242 ms |

The cross-process plan-wall reduction was 88.01%. Both plans had SHA256
`9acf031cc91045e9a8d39784bd62c518b1587e1643a1a458de61e31482a7d2d3`.
The JSON cache occupied 30,837 bytes. The remaining approximately 2.6 ms is
route counting, signature construction, native LPT selection, Plan V2
materialization, and validation; it is not repeated analytical prediction.

The cache identity includes the complete machine calibration, analytical model
schema/name and formula-source SHA256, policy dimensions and topology,
supported widths, exact-M mode, and output element size. Missing, mismatched,
corrupt, or unwritable files fall back to analytical computation. Writes merge
under a file lock and publish via atomic replacement.
