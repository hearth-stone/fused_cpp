# The stage window table under the thread-major order: rebuild, 2T, and plan validation

## Status

The window table was re-measured under the thread-major window order that landed on
2026-09-20. **The registered table `ARM_CODEX_NUMA3_80C_TP4_F512_N16_V3` stays registered**:
the rebuilt table (V4) is 0.27% faster on 18 fresh layers, below the 0.3% adoption
threshold frozen before collection. The rebuild did establish that the table's skeleton is
order-independent, that its fine structure is not reproducible at the level of its own
grid, and that 2T - carried in V3 from the old order - keeps its large window gain under
the new one.

## Why it was rebuilt

The screening (`tmp/window_table_order_screen_20260920/decision.md`, 171 cells, one session
per build) fired its R1 rule: at M = 12 the fastest window differed between the orders at
t=8 (1.23%) and t=16 (1.03%). Its other two rules passed - no scale moved by more than
0.018 and the level difference was 0.288%, inside the old build's own 0.246% session
spread. Per the frozen rule any R1 flag triggers a rebuild.

## What was measured

All on the thread-major build, CPUs 240-319, membind 3, jemalloc never-purge, 160
synthetic experts of exactly M routes on homogeneous lanes, 3 warmup + 31 runs, variants
shuffled per round; benches are the 2026-09-19 files with only `WIDTHS` changed and the
analyzers are byte-identical copies, so the adoption rules are the original ones (a window
is adopted only when both sessions pick it, its median beats that width's full stripe by
at least 2% and its p90 is below zero; the W2 sweep is paired against `w13one` at 1%).

| run | grid | sessions | wall |
| --- | --- | --- | --- |
| `tmp/window_table_rebuild_20260920` | widths 4/8/16, M 12-720, five window variants; W2 sweep at W13 = 1 | 2 + 2 | 35.3 min |
| `tmp/window_table_2t_20260920` | widths 2 and 8, same two grids | 2 + 2 | 33.6 min |
| `tmp/window_table_validation_20260921` | 18 fresh layers x 4 variants | 6 | 21.5 min |

The 2T run's first attempt failed in all four sessions with `KeyError: 'w8_full'` - both
benches report every cell against the 8T full stripe as well as its own, so width 8 must be
in the grid. No data was produced; the retry used widths (2, 8) and new seeds.

## Results

### The skeleton holds, the fine structure does not reproduce

Against V3, 16 of 36 cells change their (w13, w2) choice. In the 20 cells that keep their
choice, the scale moves by at most 0.017 (8/96). In the 16 that change it the scale is a
different variant's ratio, so it moves more - a median 0.013 and at most 0.061 (8/24,
0.848 -> 0.909, where V4 leaves W2 unwindowed). The W13 window itself differs in only
three cells (4/480 1 -> 2 tiles, 16/48 2 -> 1, 8/720 full -> 1); every other change is a W2
tile count among near-tied options, six of them turning the W2 window off and three
turning it on.

The 2T run repeated the 8T column an hour after the rebuild measured it, under identical
conditions - the grid's own reproducibility control:

| M | rebuild | 2T run | scale difference |
| --- | --- | --- | --- |
| 24 | (1, 8) r 0.872 | (1, 8) r 0.826 | -0.046 |
| 48 | (1, 8) r 0.865 | (1, 8) r 0.833 | -0.032 |
| 96 | (1, 8) r 0.913 | (1, 8) r 0.893 | -0.020 |
| 144-480 | (1, 8) | (1, 8) | -0.009 to +0.002 |
| 720 | (1, 8) r 0.976 | full stripe | +0.024, choice flipped |

So the grid resolves the skeleton - one W13 tile for M in 17-500, full stripe at M = 12 and
for 16T above M = 48 - and not which W2 tile count wins. The 16 differences against V3 are
within what the grid reproduces against itself.

### M = 12 is settled

At M = 12 all of 4/8/16T choose the full stripe under the new order as well: the windows
that looked fastest in the screening do not pass the 2% adoption gate. V3's "routes 1-16
keep the full stripe" band stands, and the screening's two flags are resolved.

### 2T keeps its window gain

2T holds a one-tile W13 window at every M from 24 up with scales 0.60-0.84, now measured
under the thread-major order instead of inherited from the old one. This matters because
the production planner's default width set is every power of two (`_default_widths`) and
the reliable-width filter only applies to tail-pool candidates, so real plans do contain 2T
lanes.

### Plan-level validation on 18 fresh layers

Layers 4, 11, 18, 26, 33 and 38 of the three requests, never used in any earlier experiment.
One quick plan per layer, with the window assignment as the only difference between
variants (the generator asserts identical structure and differing windows on all 18):

| variant | median vs `win_v3` | range | layers better |
| --- | --- | --- | --- |
| `win_v4` (rebuilt table) | -0.27% | -0.79% to +0.96% | 16 of 18 |
| `win_full` (no windows) | +1.69% | +0.63% to +4.00% | 0 of 18 |
| `quick_v4` (planned with V4 scales) | -0.28% | -0.62% to +0.54% | 16 of 18 |

Session-to-session spread was a median 0.613%. The frozen rule required a median at least
0.3% better to adopt, so V4 is not adopted; the measurement lands 0.03 points inside the
tie band with a consistent sign, which is a small gain this design was not powered to
certify rather than an absence of one.

Two side results: windows are worth 1.69% on whole plans against no windows at all, which
is what the registered table buys; and planning with V4's refreshed scales lands where the
V3-planned structure does, so the scale refresh does not move planning decisions.

## Decisions

- V3 stays registered in `stage_window_policy.py`; no planner default, asset or model
  change follows from this work.
- V4 (`tmp/window_table_rebuild_20260920/table_v4.json`) is kept as the table measured
  under the thread-major order, for reference and for any future adoption attempt.
- The window table is confirmed valid under the new order, which closes the limitation the
  thread-major change carried.

## Open

- Adoption of V4 was not decided by evidence but by a threshold it missed by 0.03 points.
  Resolving it needs a design powered for a 0.3% effect - more sessions or paired rounds -
  not another grid.
- 32T is still uncalibrated. The full-load grid cannot represent it: 80 cores hold only two
  32T lanes and leave 16 cores idle, so covering 32T needs a different design, for example
  filler load on the remaining cores.
- The grid's fine structure is not reproducible at the 1% level; a table that wants to
  choose W2 tile counts needs a better-powered grid, or the choice should be acknowledged
  as a tie and fixed by a rule rather than measured per cell.
