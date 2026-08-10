# Amazon 192-core: what retiring the per-thread stage window cost

## Question

The stage window was retired on 2026-08-09 in favour of `full_n_team_stripes`,
where the team width alone sets each worker's packed-B footprint. Reintroducing the
window as a `(threads, window_tiles)` pair makes the old geometry the `R = 1`
endpoint of the new one, so the two can be A/B'd inside one binary. How much does
the full stripe cost against a windowed traversal?

## Configuration

- Host: AmazonC5192Cores, NUMA0 CPUs `0-95`, memory bound to node 0.
- Shape: TP4 `H=4096`, `F=512`, BF16 fused SiLU, SVE JIT exact-M backend.
- W13 is 128 tiles of 64 KiB, W2 is 512 tiles of 8 KiB, `backend_n_tile=8`.
- Pages: explicit 32 MiB HugeTLB through `/dev/hugepages-32M`.
- Windows are set through `FUSED_CPP_MOE_STAGE_WINDOW_TILES=<w13>,<w2>` for the
  probe and through the Plan V2 fields for the end-to-end run. `0` is the full
  per-thread stripe, which is the shipped geometry.
- 3 warmups and 21 measured calls per point, medians reported.

## Harness validation

The window sweep reproduces the pre-retirement grid. Holding `w2` at 16 tiles,
which is the `0.5 MiB` per range those runs used at four threads, against
`results/data/stage_window_omega_20260807/grid_m{72,96,120}_t4.json`:

| M | ω13 tiles | measured ms | pre-retirement ms | delta |
| ---: | ---: | ---: | ---: | ---: |
| 72 | 1 | **9.826** | 9.977 | -1.52% |
| 72 | 2 | 10.023 | 10.202 | -1.76% |
| 72 | 4 | 10.451 | 10.585 | -1.27% |
| 96 | 1 | **11.519** | 11.687 | -1.44% |
| 96 | 2 | 11.666 | 11.695 | -0.24% |
| 96 | 4 | 12.317 | 12.302 | +0.12% |
| 120 | 1 | 13.913 | 14.170 | -1.82% |
| 120 | 2 | **13.683** | 13.874 | -1.37% |
| 120 | 4 | 14.238 | 14.398 | -1.11% |

All nine cells land within -1.8% to +0.1%, and the argmax reproduces cell for
cell: `M = 72` prefers one tile, `M = 96` is a tie, `M = 120` prefers two. The
harness and the override therefore agree with the measurement the retired policy
was calibrated against.

## Result 1: the full stripe costs about 2x at four threads

W13 swept with `w2` also at the full stripe, so the `w=full` column is exactly the
shipped geometry. Wall time in ms:

| M | t | w13=full | w13=1 | w13=2 | w13=4 | best | shipped penalty |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: |
| 56 | 4 | 18.700 | 10.797 | 10.731 | **10.699** | 4 | **+74.8%** |
| 72 | 4 | 21.276 | 11.795 | **11.746** | 11.969 | 2 | **+81.1%** |
| 96 | 4 | 28.691 | 14.147 | **14.041** | 14.506 | 2 | **+104.3%** |
| 120 | 4 | 31.714 | 17.053 | 16.601 | **16.598** | 4 | **+91.1%** |
| 56 | 8 | 12.935 | 10.511 | **10.462** | 10.501 | 2 | +23.6% |
| 72 | 8 | 14.049 | 11.482 | **11.381** | 11.572 | 2 | +23.5% |
| 96 | 8 | 16.420 | 13.492 | **13.230** | 13.314 | 2 | +24.1% |
| 120 | 8 | 20.482 | 16.488 | **15.656** | 15.771 | 2 | +30.8% |

The penalty tracks the stripe size: at four threads a worker owns 32 tiles, or
2 MiB, which is the whole private L2; at eight threads it owns 16 tiles. This is the
same monotone curve the pre-retirement sweeps saw, which topped out at 8 tiles and
already measured +23% there at `M = 120`. Extending the axis to the full stripe just
continues it.

## Result 2: W13 carries most of it, W2 the rest

| M | t | shipped | ω13=2, w2 full | w13 full, ω2=16 | ω13=2, ω2=16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 4 | 28.098 | 14.159 (-49.6%) | 24.587 (-12.5%) | **11.515 (-59.0%)** |
| 120 | 4 | 32.060 | 16.572 (-48.3%) | 26.468 (-17.4%) | **13.676 (-57.3%)** |
| 96 | 8 | 16.333 | 13.174 (-19.3%) | 15.449 (-5.4%) | **12.180 (-25.4%)** |
| 120 | 8 | 20.665 | 15.592 (-24.5%) | 17.872 (-13.5%) | **14.346 (-30.6%)** |

Windowing W13 alone recovers most of the gap and W2 alone recovers 5-17%, and the
two compose. Neither stage is negligible, which matches the `(t, ω13, ω2)` triple
being the parameter rather than a single window.

## Result 3: an independent end-to-end A/B agrees

The overlap probe also runs 28 concurrent long-route experts, so the same question
was asked again through Plan V2 with nothing else running: 96 experts each routed to
96 tokens, 24 teams of four threads filling 96 cores, which mirrors
`moe256-active-set-128`. Both arms use the same plan and differ only in the two
window fields.

| (ω13, ω2) tiles | median ms | vs shipped | bitwise equal |
| :--- | ---: | ---: | :--- |
| (0, 0) shipped full stripe | 13.216 | - | - |
| (2, 16) | 6.965 | **-47.3%** | yes |
| (1, 16) | **6.785** | **-48.7%** | yes |
| (2, 8) | 6.898 | -47.8% | yes |
| (4, 32) | 7.282 | -44.9% | yes |

Two harnesses that share no measurement code agree: the shipped geometry is about
2x slower than a windowed one on a four-thread configuration a live catalog preset
plans. The end-to-end gap is smaller than the probe's because the probe adds
concurrent long-route experts, raising the aggregate footprint further.

Every windowed arm is bitwise identical to the full stripe, as expected: windows
only reorder the (M panel, window) traversal.

## Reading

The mechanism is aggregate rather than per-core. At four threads the shipped
geometry keeps `96 x 2 MiB = 192 MiB` of packed B concurrently live against a 96 MiB
L3, while a two-tile window keeps `96 x 128 KiB = 12 MiB`. That the eight-thread
penalty is a third of the four-thread one, with half the stripe, fits the same
reading.

This does not contradict the earlier finding that the window is worth only
0.85-2.95% at fixed width. That comparison was between small windows, 1 to 8 tiles.
The full stripe is 32 tiles at four threads, off the end of that range.

## Interim conclusion

The gap is far above the 2% adoption bar, so the window is worth carrying as a
first-class parameter and the policy should select it. Results 4 and 5 bound where
that is true.

## Result 4: sixteen threads gains at short routes and loses at long ones

The 1/2/4/8-thread table above does not cover what the planner actually picks for
the synthetic presets, which is 16 and 32 threads. Sweeping those, W13 only:

| M | t | w=full | w=1 | w=2 | w=4 | best | full-stripe penalty |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: |
| 48 | 16 | 11.589 | **10.057** | 10.095 | 10.707 | 1 | **+15.2%** |
| 72 | 16 | 13.552 | **12.187** | 12.373 | 13.177 | 1 | **+11.2%** |
| 96 | 16 | 15.756 | 14.589 | **14.429** | 15.432 | 2 | **+9.2%** |
| 120 | 16 | 17.866 | 17.801 | **17.168** | 17.768 | 2 | +4.1% |
| 144 | 16 | **19.908** | 21.306 | 20.128 | 20.514 | full | 0% |
| 192 | 16 | **23.837** | 29.738 | 25.994 | 25.155 | full | 0% |
| 216 | 16 | **26.347** | 34.783 | 30.602 | 28.400 | full | 0% |
| 384 | 16 | **43.040** | 77.447 | 65.465 | 47.685 | full | 0% |
| 48 | 32 | 12.560 | 12.665 | 12.947 | 12.492 | (4) | 0.5% |
| 96 | 32 | **17.893** | 19.421 | 19.273 | 18.040 | full | 0% |
| 192 | 32 | 31.221 | 36.870 | 33.992 | 31.038 | (4) | 0.6% |
| 384 | 32 | **51.406** | 86.010 | 68.702 | 51.509 | full | 0% |

Two things bound the policy. At sixteen threads the gain stops at route 143: from
144 up the full stripe wins and a one-tile window costs 7-32%, rising to 80% at
route 384. That is the large-M regime, where A no longer fits and a wide window is
what amortizes its rescans, so narrowing the window is the wrong move. At
thirty-two threads the stripe is already four tiles and every window lands within
0.6%, inside the noise floor, so there is nothing to select.

## Result 5: the calibrated policy is worth 10-12% on two presets and neutral on two

The planner now reads the windows off a band table at the already-selected
`(routes, threads)`, so no search dimension is added. Both A/B arms use the same
planner-generated plan and differ only in the two window fields, 3 warmups and 21
measured calls, medians:

| preset | windowed tasks | policy ms | full stripe ms | gain | bitwise equal |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `moe256-uniform` | 256/256 | **14.944** | 16.797 | **+12.4%** | yes |
| `dsv4-real-2048-seq70` | 145/223 | **14.601** | 16.153 | **+10.6%** | yes |
| `moe256-tiered-hotspot` | 48/64 | 9.518 | 9.664 | +1.5% | yes |
| `moe256-active-set-128` | 52/128 | 12.908 | 12.824 | -0.65% | yes |

The noise floor of this harness was measured at 1.3-1.7% by A/Bing two arms whose
plans were identical, so the last two rows are neutral rather than a gain or a
regression.

`moe256-active-set-128` is worth a note: its 52 windowed tasks are at route 96 with
sixteen threads, exactly the cell result 4 measured at +9.2%, yet end to end it does
not move. Its remaining 76 tasks run at thirty-two threads, which the policy leaves
alone, so the windowed tasks are not what sets the makespan. The homogeneous probe
therefore overstates the gain for a mixed-width plan.

## Decision

Adopt. Two presets clear the 2% bar by a wide margin, none regresses beyond noise,
and every arm is bitwise identical to the full stripe. Shapes and widths the table
does not cover keep reporting `stage_geometry=full_n_team_stripes` and run exactly
as before, so profiles calibrated under that name stay valid.

## Not established

- Only `M` in 48-384 and widths 4, 8, 16, 32 were measured. Widths 1, 2, 3, 6, 12
  are carried over from the V4 table or left uncovered.
- The sixteen-thread entries were calibrated on the W13 axis only; their W2 entry is
  the full stripe because that axis was not swept there.
- The end-to-end A/B of result 3 uses a hand-built plan that packs 96 tasks onto 24
  team slots. Result 5 uses planner-generated plans.
- Whether `first_panel_prefetch` or `bulk_m`, both switched off whenever `R > 1`,
  would recover part of the windowed arm if re-enabled per window.
- Why `moe256-active-set-128` does not move despite holding a cell measured at
  +9.2% in isolation. The mixed-width reading above is inference, not measurement.

## Data

- `results/data/stage_window_tiles_20260810/m*_t*_w*.json`: the W13 sweeps of
  results 1 and 4.
- `results/data/stage_window_tiles_20260810/m*_w*.json`: the harness validation
  against the pre-retirement grid.
- `results/data/stage_window_tiles_20260810/m*_t*_*_*.json`: the two-stage
  decomposition of result 2.
