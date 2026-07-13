# Policy-aware MoE planner validation

## Target and calibration

- Target: AWS AArch64, 64 logical cores, two NUMA nodes.
- Rank CPU sets: `0-31` and `32-63`; each rank has 48 MiB LLC.
- Workload: H=4096, 64 global experts, 2048 tokens, top-k=6.
- TP2 profile: F=1024, 64 local experts per rank.
- EP2 profile: F=2048, 32 local experts per rank.
- Kernel: SVE BF16, N tile 8, M12 bulk plus M1/M2/M4/M8 tails.
- Calibration: five warmups and 20 timed runs per point, all local experts in
  contention calls, and eight consecutive experts in isolated streaming calls.
- Source SHA256: `21853affada8848aaaea863c04e2701be2b77ba0411248ed16fcfccb40470ddf`.
- Extension SHA256: `2d5fc0411c6f5e0887e9728037fe8249d4758e789e974ed4f30ffca33a77d1da`.

The active schema-v2 calibration consists of four exact-policy profiles:

- `contention_async_amazon_c5_64c_tp2_sve_F1024_splitw13_v2_r1_20260713.json`
- `contention_async_amazon_c5_64c_tp2_sve_F1024_nosplitw13_v2_r1_20260713.json`
- `contention_async_amazon_c5_64c_ep2_sve_F2048_splitw13_v2_r1_20260713.json`
- `contention_async_amazon_c5_64c_ep2_sve_F2048_nosplitw13_v2_r1_20260713.json`

The split and non-split tables use identical grids:

- isolated routes: `1,2,4,8,12,24,48,96,192,384,768,1536,2040`;
- contention routes: `1,2,4,8,12,24,48,192,768,2040`;
- threads: `1,2,4,8,16,32`;
- 12 exact 32-core shapes, from `(32)` through `(1 x 32)`.

Every sample starts both NUMA-local ranks behind a socket barrier. A global
sample is the pairwise maximum of the two rank samples. These profiles therefore
describe two concurrently active ranks; they are not single-rank profiles.

## Validation workloads

`validate_policy_planner.py` measures every core shape under both W13 policies,
plus an independent execution of the planner-selected plan. The 2026-07-13
result uses two warmups and ten timed runs for each candidate and contains three
workloads:

- `uniform`: equal routes per expert;
- `hotspot`: four 768-route experts, twelve 384-route experts, and 48 96-route
  experts;
- `trace`: the captured `dsv4-real-2048-seq70` routing summary.

The captured summary is from DeepSeek V4 Flash rank 0, sequence 70, layer 27.
It contains 2048 tokens, top-k=6, 256 experts, 12288 routes, and 223 active
experts. The capture retained exact counts for the top 16 experts. Its remaining
tail is deterministically reconstructed to preserve active count, total routes,
min/max, mean, and population standard deviation. For this 64-expert evaluator,
expert IDs are folded modulo 64. TP2 uses the folded global histogram on both
ranks; EP2 partitions it into 32 local experts per rank, with 6068 and 6220
routes respectively.

## Exhaustive results

Regret is computed from measured medians against the fastest fixed policy and
shape. A negative raw delta caused by repeated-measurement noise is reported as
zero regret.

| Mode | Routing | Selected rank plans | Predicted ms | Actual ms | Best fixed ms | Regret | Prediction error |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TP2 | uniform | S:`8,8,8,8` / S:`8,8,8,8` | 42.617 | 43.537 | 42.690 | 1.98% | -2.11% |
| TP2 | hotspot | S:`16,8,8` / S:`16,8,8` | 43.043 | 42.868 | 42.844 | 0.06% | +0.41% |
| TP2 | trace | S:`16,8,8` / S:`16,8,8` | 42.686 | 43.974 | 43.537 | 1.01% | -2.93% |
| EP2 | uniform | S:`32` / S:`32` | 45.485 | 45.409 | 45.420 | 0.00% | +0.17% |
| EP2 | hotspot | S:`32` / S:`32` | 63.340 | 60.671 | 59.659 | 1.70% | +4.40% |
| EP2 | trace | S:`16,16` / S:`16,16` | 45.055 | 44.694 | 44.680 | 0.03% | +0.81% |

`S` means split-W13. The maximum selected-plan regret is 1.98%. Prediction error
ranges from -2.93% to +4.40%. Separate 30-sample repeats show an approximately
1% practical noise floor, so sub-percent candidate ordering is not treated as
significant.

The complete exhaustive result, including fixed candidates, percentile
intervals, raw regret, and rank histograms, is stored in
`cost_model/profiles/policy_planner_validation_amazon_c5_64c_20260713.json`.

## EP2 hotspot diagnosis

The EP2 hotspot places `4x768 + 12x384 + 16x96 = 9216` routes on rank 0 and
`32x96 = 3072` routes on rank 1. Both ranks select split-W13 `(32)`, so each rank
runs one expert at a time and the residual is not intra-rank expert contention.

A targeted five-warmup, 30-run replay measured:

| Rank pair | Predicted ms | Actual ms |
| --- | ---: | ---: |
| mixed / cold | 63.340 | 60.603 |
| mixed / idle | 63.340 | 59.654 |
| mixed / mixed | 63.340 | 63.000 |

Homogeneous dual-rank component replays match the profile within 0.41%. The
mixed rank also matches the profile when both ranks remain active for the whole
call. In the real mixed/cold case, the cold rank finishes at about 28.5 ms and
the mixed rank then runs alone until about 60.6 ms. The current evaluator instead
predicts each rank independently with a `concurrent_ranks=2` profile and takes
their maximum, applying the dual-rank derate to the long rank's entire DAG.

The total overestimate is about 2.74 ms. Of this, about 2.40 ms, or 88%, is the
missing rank-lifetime contention release; only about 0.34 ms is the residual
between the profile and the symmetric mixed/mixed replay. Stage traces place
W13 at 67.3-67.8% of W13+W2 time, so the existing 2/3 W13 + 1/3 W2 split is not
the source of this hotspot.

There is a separate 96-route full-call residual. Route 96 is an exact isolated
point but is not a contention-route anchor, so a uniform 96-route full call is
interpolated between routes 48 and 192. This does not create the rank-0 hotspot
prediction, whose 96/384/768 isolated points are exact, but it must be separated
from the rank-lifetime fix when validating the short rank.

## TP2 versus EP2 estimate

For uniform routing and the modeled 60 GB/s intra-pair, 20 GB/s inter-pair
topology, the layer evaluator reports:

| Mode | Compute ms | Communication ms | Total ms | Policy/shape |
| --- | ---: | ---: | ---: | --- |
| TP2 | 42.617 | 0.841 | 43.458 | split, `8,8,8,8` |
| EP2 | 45.485 | 2.519 | 48.003 | split, `32` |

These totals combine exact-profile compute predictions with the analytical
collective model. They are not an end-to-end distributed runtime measurement
and do not yet include route-dependent gather/scatter or weighted top-k combine.

## Scope and remaining gates

The validation covers the exact TP2/EP2 policies, machine topology, kernel
binary, and workloads above. The captured routing file is a real summary with a
moment-matched tail, not a complete original `topk_ids` dump.

Cross-profile interpolation remains disabled. Passing the captured trace on an
exact profile does not validate interpolation across F, parallel degree, or
machine topology; those dimensions require out-of-profile measurements.

Before treating absolute EP time as calibrated, add matching single-rank
profiles and a joint multi-rank event simulation that changes rates when a rank
finishes. The acceptance gates remain:

- selected-plan absolute error <= 3%;
- top-three candidate median absolute error <= 3% and P90 <= 5%;
- measured planner regret <= 2%.
