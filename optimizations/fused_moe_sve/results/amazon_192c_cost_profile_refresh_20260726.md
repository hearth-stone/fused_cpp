# Amazon 192-Core TP4 Cost-Profile Refresh

Date: 2026-07-26

## Scope

This refresh replaces the 2026-07-20 TP4/F512 split-W13 and no-split-W13
calibration pair with measurements from the current ARM extension at commit
`da267eb`.

- Host: AmazonC5192Cores, 192 Arm cores, two 96-core NUMA nodes.
- Rank 0: CPUs `0-95`, memory node 0.
- Rank 1: CPUs `96-191`, memory node 1.
- Shape: H=4096, F=512, BF16 SiLU, TP4, 256 global/local experts.
- Weights: 256 distinct expert weights; timed calls walk consecutive experts.
- Aggregation: two synchronized ranks, median of the pairwise maximum.
- Sampling: 5 warmups and 20 measured calls per point.
- Isolated grid: routes `1..12,24,48,96,192,384,768,1536,2040`;
  threads `1,2,4,8,16,32,48,64,96`.
- Contention grid: routes `1..12,24,48,192,768,2040`; 14 shapes from
  `1x96T` through `96x1T`.

The profiler command was run twice, with `--w13-split 1` and
`--w13-split 0`:

```bash
ISO=1,2,3,4,5,6,7,8,9,10,11,12,24,48,96,192,384,768,1536,2040
CONT=1,2,3,4,5,6,7,8,9,10,11,12,24,48,192,768,2040
SHAPES='96;64,32;48x2;48,32,16;32x3;32,32,16x2;32,16x4;16x6;8x12;4x24;2x32;2x48;1x64;1x96'

.venv/bin/python \
  cpu_moe_schedule_optimization/cost_model/profile_contention_async_dual_rank.py \
  --output tmp/profile.json \
  --parallel-mode tp --parallel-degree 4 \
  --hidden-size 4096 --ffn-hidden-size 512 \
  --global-experts 256 --local-experts 256 --measurement-experts 256 \
  --isolated-measurement-experts 8 \
  --w13-split 1 --sve-implementation jit \
  --cpu-groups '0-95;96-191' --numa-nodes 0,1 \
  --isolated-route-buckets "$ISO" --contention-route-buckets "$CONT" \
  --thread-buckets 1,2,4,8,16,32,48,64,96 \
  --contention-shapes "$SHAPES" --warmup 5 --runs 20 \
  --keep-rank-profiles
```

Both profiles record source SHA
`a62e0d9425381750fdc859ed97e2a4d1cc33727ffe36dbb75c453f061a924dfb`
and extension SHA
`e652d9aad3025a4836d0406110bcbdf3fdc1356aaba2bf789595359a58a92559`.
The aggregate profile SHA256 values are:

- split-W13: `6143c27e980975471e1e3c4b1bcd59916b8d1b5f75a7a6d3983d950c3e07cc47`
- no-split-W13: `d29f32ecb2d4d934144b9aae568ddb3f990798706d391948d2838445fcfa0648`

## Old Versus New

The route, thread, shape, expert-population, pinning, and sampling grids are
unchanged. The working-set metadata now explicitly records the zero byte-window
target and derived W13/W2 range sizes.

Across all isolated points, the new/old median is +1.67% for split-W13 and
+2.99% for no-split-W13. The change is strongly team-width dependent:
1-64T points are generally within about 0-5%, while the 96T medians increase
by 73.25% and 78.74%, respectively.

Across all homogeneous full-call points, the median increase is 12.51% for
split-W13 and 12.40% for no-split-W13. This aggregate gives equal weight to
wide shapes that production rarely selects. Long-route shapes near the
production optimum moved much less:

| Policy, M=2040 | Old | New | Delta |
| --- | ---: | ---: | ---: |
| split `1x96T` | 1004.37 ms | 1208.55 ms | +20.33% |
| split `3x32T` | 463.42 ms | 510.72 ms | +10.21% |
| split `6x16T` | 368.09 ms | 382.53 ms | +3.92% |
| split `12x8T` | 335.80 ms | 339.15 ms | +1.00% |
| split `24x4T` | 332.45 ms | 334.60 ms | +0.65% |
| no-split `6x16T` | 327.46 ms | 343.23 ms | +4.82% |
| no-split `12x8T` | 323.43 ms | 329.60 ms | +1.91% |

The best homogeneous full-call point across both policies changed as follows:

| Routes | Old best | Old ms | New best | New ms | Old choice regret on new |
| ---: | --- | ---: | --- | ---: | ---: |
| 1 | no-split `24x4T` | 8.660 | no-split `64x1T` | 9.294 | 2.82% |
| 12 | no-split `24x4T` | 8.989 | split `96x1T` | 9.501 | 3.44% |
| 24 | split `6x16T` | 12.489 | split `12x8T` | 14.405 | 1.99% |
| 48 | split `6x16T` | 16.010 | no-split `12x8T` | 19.199 | 2.46% |
| 192 | no-split `6x16T` | 34.661 | no-split `6x16T` | 39.956 | 0.00% |
| 768 | no-split `6x16T` | 123.137 | split `24x4T` | 129.275 | 1.17% |
| 2040 | no-split `12x8T` | 323.427 | no-split `12x8T` | 329.601 | 0.00% |

Over all 17 contention routes, retaining the old cross-policy optimum has
3.01% median and 7.63% maximum regret under the new table. The absolute-time
refresh is therefore material, but most old scheduling decisions remain close
to the new measured optimum.

## Planner Impact

The table below runs the current `PolicyAwarePlanner` against the old pair and
the refreshed pair. These are model predictions, not new end-to-end operator
measurements.

| Workload | Old policy/shape | Old ms | New policy/shape | New ms | Old-choice regret on new |
| --- | --- | ---: | --- | ---: | ---: |
| Uniform | split `6x16T` | 16.010 | split `6x16T` | 19.672 | 0.00% |
| Active set 8 | no-split `3x32T` | 8.752 | no-split `3x32T` | 9.660 | 0.00% |
| Active set 16 | no-split `32,32,16,16` | 8.092 | no-split `32,32,16,16` | 9.021 | 0.00% |
| Active set 32 | no-split `32,32,16,16` | 8.463 | split `12x8T` | 10.095 | 4.24% |
| Active set 64 | no-split `6x16T` | 8.526 | no-split `6x16T` | 10.433 | 0.00% |
| Active set 128 | no-split `6x16T` | 10.150 | no-split `6x16T` | 12.537 | 0.00% |
| Tiered hotspot | no-split `6x16T` | 8.458 | split `12x8T` | 9.764 | 2.04% |
| Long/short bimodal | no-split `6x16T` | 9.331 | no-split `6x16T` | 9.842 | 0.00% |
| Captured DSV4 | split `12x8T` | 9.982 | split `12x8T` | 10.493 | 0.00% |

Only active-set-32 and tiered-hotspot change the selected policy/shape with a
nonzero old-choice penalty. The current pruning rules and planner candidate
space therefore remain unchanged.

## Measurement Quality

The refreshed table exposes substantially more run-to-run variance than the
2026-07-20 table:

- split full-call relative interval: median 0.82% to 6.17%;
- no-split full-call relative interval: median 0.91% to 5.94%;
- split NUMA max/min asymmetry: median 5.14%, P90 22.32%;
- no-split NUMA max/min asymmetry: median 4.46%, P90 25.98%;
- isolated 96T NUMA asymmetry: split median 24.62%, no-split median 31.25%.

Long-route contention is more stable: for M>=768, median NUMA asymmetry is
2.40% for split and 2.51% for no-split. A second reduced split-W13 run with
30 samples at M=12/192/2040 reproduced the full table's 18 overlapping
contention points with +0.47% median difference and a -4.13% to +1.92% range.
This confirms the long-route trend while leaving 96T as a noisy calibration
edge.

The serialized isolated formula is unchanged in form. On measured M>=48
points, its absolute fit error is:

| Policy | Median | P90 | Maximum |
| --- | ---: | ---: | ---: |
| split-W13 | 1.91% | 9.66% | 36.17% |
| no-split-W13 | 1.00% | 9.11% | 42.66% |

The maxima are concentrated at wide teams. The direct `phi_pts` correction and
the measured thread domain remain authoritative; coefficients must not be
interpreted as machine-independent physical constants or extrapolated beyond
96 threads.

## Decision

The active catalog now contains only the refreshed split/no-split pair. Both
profiles pass schema validation, load through `ContentionCostModel`, and are
returned atomically by `ProfileCatalog.policy_variants()`. The old pair remains
in Git history. No cost-model equation, candidate-space, or production pruning
change is made by this refresh.
