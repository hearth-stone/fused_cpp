# Why narrow lanes are under-predicted: the concurrent weight footprint against the LLC

## Status

The 2T defect - plans built from two-thread lanes measure 1.21-1.24 x their prediction while
every other family sits at 1.09 - has a mechanism. It is not specific to 2T and not a property
of the window term: **the weights a 40-core LLC domain holds live at once are 480 MiB / t when
lanes run full stripes, against 70 MiB of LLC, and the full-load dilation follows that
overflow.** A one-tile window makes the live footprint about 5 MiB whatever the lane width,
and then every width dilates the same.

This is a diagnosis. No model, table, planner default or search space changes with it; the
model revision it implies is designed and validated separately.

## What was known

- E2 and E5: 2T plans measure 1.237 and 1.281 x their v11 prediction, and the predicted order
  of 2T against 2T-free plans was wrong on 18 of 18 workloads. 2T lanes are out of the search
  space because of it.
- P7: next to 2T background lanes the dilation is higher than next to 4T lanes (factors 1.10 to
  1.44 loading, 1.53 to 1.65 steady). Applied as factors (v12) it did not fix plans.
- P8: with 2T backgrounds v11 under-predicts, and the gap grows with the background's route
  count (1.14 at M12 to 1.56 at M96). Encoding "a 2T lane counts as loading in every phase"
  (v13) reproduced the probe cells at 0.978 and still failed E5.
- Every probe from P1 to P8 ran with windows off, while real 2T lanes run windowed.

## E12: the window term is not the defect

`tmp/two_thread_mechanism_20260921/design.md`, frozen before collection. Nine layers never used
before (5, 23, 35 of the three requests), three families that differ only in window geometry:
family H (64 of 80 cores in 2T lanes), family M (2T next to unwindowed 32T/16T lanes), family
M8 (2T next to windowed 8T lanes). Two sessions, session spread median 0.30%.

| variant | measured / predicted |
| --- | --- |
| `h2t_win` / `h2t_full` | 1.033 / 1.077 |
| `mix_win` / `mix_2Tfull` | 1.134 / 1.125 |
| `mix8_win` / `mix8_2Tfull` | 1.051 / 1.057 |

All three frozen readings came out negative: the windows-off variants sit at the model's anchor
level, the window gain is predicted to within 0.04, and hand-built 2T-heavy plans are predicted
as well as any other plan. So the defect is not carried by 2T lanes as such - it is carried by
the plans the search builds out of them.

## The corpus: error tracks packing, not lane width

`tmp/model_error_meta_20260921/collect.py` recomputes the v11 prediction of every plan measured
in E3, E5, E6b, E9, E11 and E12 - 546 plans - from the bridge's own windows, with model-visible
features. Correlations with measured/predicted: window credit +0.56, utilization +0.38,
2T share +0.23, loading share -0.08.

Among the 437 tightly packed plans (modelled core utilization >= 0.88):

| subset | n | measured / predicted |
| --- | --- | --- |
| window credit 0.00-0.02 | 94 | 1.090 |
| window credit 0.10-0.25 | 97 | 1.103 |
| no 2T work | 407 | 1.093 |
| 2T share >= 0.2 | 29 | 1.218 |

So the window credit is realized - 20 points of credit cost one point of error - and the error
is carried by 2T work in tightly packed plans. The loose 2T-heavy plans of E12 (utilization
0.47-0.71) are predicted at 1.03-1.13, which is why E12 alone could not see it.

## The grid: the mechanism

The window table's own grid is the cleanest possible measurement of this state - 160 synthetic
experts of exactly M routes on homogeneous lanes, all 80 cores busy - and both the unwindowed
and the one-tile-window cells were measured for widths 2, 4, 8 and 16 in
`tmp/window_table_rebuild_20260920` and `tmp/window_table_2t_20260920`. Dividing each cell by
the model's isolated time for the same chain gives the measured dilation:

| M | t=2 full / win | t=4 full / win | t=8 full / win | t=16 full / win |
| --- | --- | --- | --- | --- |
| 12 | 1.72 / 1.76 | 1.75 / 1.79 | 1.66 / 1.69 | 1.51 / 1.55 |
| 24 | 1.77 / 1.19 | 1.61 / 1.25 | 1.44 / 1.25 | 1.40 / 1.32 |
| 48 | 1.73 / 1.06 | 1.39 / 1.07 | 1.25 / 1.08 | 1.17 / 1.14 |
| 96 | 1.57 / 1.07 | 1.26 / 1.06 | 1.15 / 1.06 | 1.10 / 1.09 |
| 192 | 1.47 / 1.10 | 1.18 / 1.06 | 1.10 / 1.05 | 1.07 / 1.07 |
| 480 | 1.25 / 1.05 | 1.15 / 1.13 | 1.09 / 1.07 | 1.07 / 1.07 |
| 720 | 1.22 / 1.04 | 1.14 / 1.13 | 1.10 / 1.09 | 1.06 / 1.06 |

Read across a row: unwindowed, the four widths differ by 0.16 to 0.56 and the narrow lane is
always worst; with a one-tile window they differ by 0.03 to 0.13 and all four sit at 1.05-1.13.
The ordering is the order of the weight footprint a 40-core LLC domain (70 MiB) holds live.
An expert's packed B is 12 MiB (8 MiB W13 + 4 MiB W2); a domain runs 40/t lanes at once, so
unwindowed it holds 480 MiB / t live, and with a one-tile W13 window each core holds 128 KiB,
which is 5 MiB per domain at every width:

| width | lanes per domain | live B per domain, full stripe | vs 70 MiB | with a 1-tile window | dilation at M=96 (full / win) |
| --- | --- | --- | --- | --- | --- |
| 2T | 20 | 240 MiB | 3.4x | 5.0 MiB (0.07x) | 1.57 / 1.07 |
| 4T | 10 | 120 MiB | 1.7x | 5.0 MiB (0.07x) | 1.26 / 1.06 |
| 8T | 5 | 60 MiB | 0.9x | 5.0 MiB (0.07x) | 1.15 / 1.06 |
| 16T | 2 | 24 MiB | 0.3x | 4.0 MiB (0.06x) | 1.10 / 1.09 |

A domain whose concurrent weights do not fit the LLC re-reads them from DRAM for every 12-row
panel, so its DRAM traffic scales with M; one whose weights fit reads them once and serves the
later panels from the LLC. The window restores that reuse, which is why the gain is largest
exactly where the overflow is largest (the table credits 2T r = 0.60-0.84, more than any other
width), and why the windowed footprint - and with it the dilation - stops depending on width.

The private L2 tells the same story less sharply (the per-core slice is 12 MiB / t against
1.25 MiB, so 4.8x at 2T down to 0.6x at 16T), but the PMU measurement of the windows says the
LLC is where it happens: a one-tile W13 window cuts DRAM reads by 27-75% while L2 refills move
by 0-9% (MATHEMATICAL_MODEL v1.103, `tmp/window_pmu_20260919`).

M = 12 is the exception that fits: one panel means there is no cross-panel reuse to lose, so
windows do not help (1.72 against 1.76 at 2T) and all widths dilate by 1.5-1.8.

The model cannot express this. Its curves are functions of the target's width and the number of
other cores in a state; two configurations with the same core counts and different live slices
read the same curve. Its error is exactly the part the slice explains: at M >= 24 unwindowed it
under-predicts by 15-45% at 2T, 6-11% at 4T, 1-6% at 8T and 0-4% at 16T, and with windows the
error falls to 1.03-1.13 at every width.

A check that this is contention and not an isolated-time bias: both the windowed and the
unwindowed cells are divided by the same `T_iso(M, t)`, so a width-dependent error in the
isolated time would appear in both. It appears only in the unwindowed cells.

## What this explains

- Why v13's "a 2T lane counts as loading in every phase" fixed small M and overshot large M
  (0.92 at M=192, 0.75 at M=720): it replaced an M-shaped effect with a flat one.
- Why P7's background-width factors did not fix plans: the width is a proxy for the slice, and
  the proxy breaks as soon as windows or mixed widths enter.
- Why searched 2T plans are worse than hand-built ones: the search packs the machine, and it is
  free to put small experts - which the table leaves unwindowed below 17 routes - on 2T lanes,
  where their slice overflows L2 by 4.8x.
- Why the stage window table gives narrow lanes the largest gains at all: the narrower the
  lane, the more lanes share a domain, and the more the window cuts from its live footprint.

## Is a fix worth it? No, for 2T

The mechanism says how to model narrow lanes; it does not say they are worth modelling. E13
(`tmp/two_thread_value_20260921/design.md`) asked what 2T could buy, on the 18 layers of E5.

Per-expert core time at full load, from the grid cells the table selects, says narrow lanes are
more core-efficient at small M: at M = 24, 4.81 core-ms at 2T against 5.13 at 4T, 5.59 at 8T
and 6.13 at 16T; at M = 96 the four widths tie within 1%; at M = 720 2T and 8T tie and 4T is 4%
worse. Giving every expert of a layer its cheapest width and dividing by 80 cores bounds any
plan's makespan:

| bound | median over the 18 layers |
| --- | --- |
| widths 2 to 16 | 27.88 ms |
| widths 4 to 16 | 28.13 ms |
| measured `l13_no2` reference | 28.25 ms |

**2T improves the bound by 0.82%, and the reference plan is already 1.3% above the bound that
includes 2T.** A perfect 2T planner could win about one point, most of which is not even
attributable to the width.

The measurement agrees, if less sharply: six hand-built variants on those layers, balanced with
the measured grid costs, are all 15-20% slower than the reference and win no layer. The 4T
control with the same shape is +15.23% against the best 2T variant's +17.53%, so most of that
gap is the hand-built shape rather than the width - the measurement rules out hand-built 2T
plans, and the bound rules out the rest.

One placement rule died here too: keeping experts of 16 routes or fewer off 2T lanes, which the
mechanism suggests because they cannot be windowed, makes the plan 5.10% *slower*. Pushing them
onto the wide lanes unbalances it, and that costs more than their LLC overflow.

So 2T stays out of the search space, and no 2T-specific model work follows. What remains worth
doing with the mechanism is general: it explains the window table's value and the model's
residual level bias, and it would tighten the model wherever a plan's lanes run unwindowed.

## What a fix would have to do

Parameterize the contention state by the live weight footprint the LLC domain carries - the sum
over its concurrent lanes of `threads x window_tiles x tile bytes`, or the full stripe when
unwindowed - against the domain's 70 MiB, instead of by lane width. The measurements to
calibrate it already exist for overflow 0.3x to 3.4x (the unwindowed grid) and for 0.07x (the
windowed grid). The revision has to be validated on fresh
layers against the current reference before it can change the model, and only then does the
question of returning 2T lanes to the search space reopen.

## Artifacts

- `tmp/two_thread_mechanism_20260921/` - E12 design, decision, generator, analyzer, sessions.
- `tmp/model_error_meta_20260921/` - the 546-plan corpus and its collector.
- `tmp/window_table_rebuild_20260920/`, `tmp/window_table_2t_20260920/` - the grids re-read here.
