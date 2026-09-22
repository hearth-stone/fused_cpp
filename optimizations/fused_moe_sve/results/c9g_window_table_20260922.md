# Registering C9g's stage window table

## Status

Registered. On 18 fresh layers the table's windows beat full stripes on **18 of 18**, by a
median 2.07%, against a session-to-session floor of 0.1%.

## Why this needed whole-plan evidence

The grid says windows are worth 3-4x per cell on this machine - a 2T lane at M = 96 goes from
28.0 ms on full stripes to 9.4 ms with a one-tile window. That is not the number production
sees, and this session was already caught extrapolating a grid ranking to real layers: the same
grid ranks 2T fastest at every M up to 192, yet forcing 2T lanes on real layers loses by 157%
(`c9g_width_and_panel_20260922.md`). So the table was measured the way E11 measured
`Arm-codex`'s, on whole plans.

## Design

`tmp/c9g_window_20260922/design.md`, frozen before measurement. Six layers never used on this
machine and not used by the 2T study (5, 13, 21, 29, 33, 39) x three requests = 18 layers. Two
plans each from the same quick planner and the same median-of-three calibration, differing only
in the window policy:

| plan | W13 window tiles | W2 window tiles |
| --- | --- | --- |
| `windows` | 0, 1, 2 | 0, 8, 16 |
| `full_stripe` | 0 | 0 |

`bench.py` unchanged, producer-hot A, cold B, 5 warmup + 31 runs, points shuffled per round,
two sessions, `wait_idle` before each, node 0, jemalloc never-purge.

Frozen rule: register if windows are faster on at least 14 of 18 layers with a median gain of
at least 1%; do not register below 0.5% or if more than 4 layers are slower; otherwise a
measured tie and the table stays out. The 1% floor sits five times above this machine's
0.1% session-to-session floor and its 0.49% per-cell spread
(`numa_interference_c9g_20260922.md`).

## Results

| | |
| --- | --- |
| layers where windows win | **18 of 18** |
| median gain | **2.07%** |
| best / worst layer | 7.56% / 1.24% |
| repeat error between the two sessions | 0.00% to 0.16% |

Every layer clears the 1% threshold on its own, and the smallest gain is eight times the
largest repeat error. Against `Arm-codex`, where E11 measured 1.69% on whole plans: the same
order, slightly larger here, consistent with this machine's larger per-domain footprint.

**The grid's 3-4x does not carry to whole plans.** A plan spends its time across many experts
at many route counts, most of them outside the bands where windowing is worth most, and the
makespan is set by the slowest lane rather than by the average cell.

## The table

Composed from this instance's grid by the unchanged 2026-09-19 rule. Six route bands over
17-720; routes 1-16 keep the full stripe because every window tied there, and above 720 there
is no evidence. Widths 2-32; anything else is uncalibrated and keeps the full stripe.

It also reproduces across instances. The grid was re-measured on a rebuilt machine and composed
again: 28 of 35 (width, M) cells choose the same window on TP4 and 30 of 35 on TP2, with time
scales drifting 0.001 in median and 0.029 at most. Ten of the twelve disagreements flip between
a one-tile and a two-tile window while `r` barely moves.

## Registration

`AMAZON_C9G_192C_TP4_F512_N8_V1` in `stage_window_policy.py`, scoped to
`amazon_c9g_192c_2numa_96c_sve128_tp4`. It shares its shape with the Amazon C5 table - H=4096,
F=512, backend tile 8, because a 128-bit SVE build tiles by 8 - so the two are separated only
by `machine_ids`. Before that field was added on 2026-09-21 this registration would have been
impossible: C9g would have silently inherited C5's table, and C5's machines would have
inherited C9g's.

## Limitations

- TP4 only. The TP2 table is composed (`tmp/c9g_grid_20260922/tp2_table.json`) and reproduces
  across instances at 30 of 35 cells, but no whole-plan validation was run for it, so it is not
  registered.
- The quick planner only. No reference search was run, because C9g has no event-model probe
  curves.
- The gain is measured against full stripes, which is what C9g runs today. It does not say the
  table is the best table, only that it beats having none.
