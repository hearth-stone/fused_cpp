# Policy-aware MoE planner validation

## Target and calibration

- Target: AWS AArch64, 64 logical cores, two NUMA nodes.
- Rank CPU sets: `0-31` and `32-63`; each rank has 48 MiB LLC.
- Workload: H=4096, 64 global experts, 2048 tokens, top-k=6.
- TP2 profile: F=1024, 64 local experts per rank.
- EP2 profile: F=2048, 32 local experts per rank.
- Kernel: SVE BF16, N tile 8, M12 bulk plus M1/M2/M4/M8 tails.
- Calibration: 20 timed runs per point, all local experts in contention calls,
  eight consecutive experts in isolated streaming calls.
- Source SHA256: `21853affada8848aaaea863c04e2701be2b77ba0411248ed16fcfccb40470ddf`.
- Extension SHA256: `2d5fc0411c6f5e0887e9728037fe8249d4758e789e974ed4f30ffca33a77d1da`.

The split and non-split tables use identical route/thread/shape grids. Every
sample starts both NUMA-local ranks behind a socket barrier, and each global
wall sample is the pairwise maximum of the two rank samples.

## Exhaustive validation

`validate_policy_planner.py` measured every one of the 12 core shapes under
both W13 policies, plus an independent execution of the planner-selected plan.
It used two warmups and ten timed runs for uniform and M12-aligned hotspot
routing. Regret is computed from measured medians against the fastest fixed
policy/shape; a negative raw delta from repeated-measurement noise is reported
as zero regret.

| Mode | Routing | Selected rank plans | Predicted ms | Actual ms | Best fixed ms | Regret | Prediction error |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TP2 | uniform | S:`8,8,8,8` / S:`8,8,8,8` | 42.863 | 42.730 | 43.001 | 0.00% | +0.31% |
| TP2 | hotspot | S:`16,8,8` / S:`16,8,8` | 43.306 | 43.320 | 43.058 | 0.61% | -0.03% |
| EP2 | uniform | S:`32` / S:`32` | 45.704 | 45.469 | 45.488 | 0.00% | +0.52% |
| EP2 | hotspot | S:`32` / S:`16,16` | 63.508 | 60.690 | 60.713 | 0.00% | +4.64% |

`S` means split-W13. The EP hotspot intentionally places 9216 routes on rank 0
and 3072 on rank 1. The evaluator and planner therefore choose rank-local
shapes and use the slower rank as global compute wall time.

The largest measured regret is 0.61%; the largest absolute prediction error is
4.64%. The full result, including all fixed candidates, percentile intervals,
raw regret, and rank histograms, is stored in
`cost_model/profiles/policy_planner_validation_amazon_c5_64c_20260711.json`.

## TP2 versus EP2 estimate

For uniform routing and the modeled 60 GB/s intra-pair, 20 GB/s inter-pair
topology, the layer evaluator reports:

| Mode | Compute ms | Communication ms | Total ms | Policy/shape |
| --- | ---: | ---: | ---: | --- |
| TP2 | 42.863 | 0.841 | 43.704 | split, `8,8,8,8` |
| EP2 | 45.704 | 2.519 | 48.223 | split, `32` |

These totals combine measured-profile compute predictions with the analytical
collective model; they are not an end-to-end distributed runtime measurement.

## Scope

The validation covers the exact TP2/EP2 policies, machine topology, kernel
binary, and synthetic uniform/hotspot histograms above. No target-model routing
dump is present in this repository. Cross-profile interpolation remains
disabled until a real histogram is supplied with `--routes-json` and passes the
same exhaustive regret gate.
