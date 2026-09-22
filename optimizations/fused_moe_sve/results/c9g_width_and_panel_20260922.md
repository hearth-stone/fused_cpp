# C9g: the window table across instances, the panel term, and whether 2T is wrongly excluded

## Status

Three experiments on the rebuilt C9g instance, all with the median-of-three calibration and
the idle guard that now watches the 5-minute load average.

1. **The window table reproduces across instances.** 28 of 35 (width, M) cells choose the same
   window on TP4 and 30 of 35 on TP2, with time-scale drift of 0.001 in median.
2. **The per-panel term stays unadopted**, and the follow-up refuted both of the readings
   frozen for it. The miss is a fixed number of microseconds, not a structure.
3. **2T is correctly excluded**, decisively: forced 2T plans lose on 18 of 18 real layers by a
   median 157%. This overturns a claim in `second_machine_c9g_20260921.md`.

## 1. The window table across instances

The 2026-09-21 table came from an instance that has since been reprovisioned. The grid was
re-measured on the new one with the identical design (widths 2-32, routes 12-720, W13 window
0-32 tiles, two sessions per shape) and composed by the same frozen 2026-09-19 rule.

| | cells choosing the same window | time-scale drift |
| --- | --- | --- |
| TP4 | 28 of 35 | median 0.001, max 0.029 |
| TP2 | 30 of 35 | median 0.001, max 0.013 |

Ten of the twelve disagreements are ties the grid cannot resolve - the choice flips between a
one-tile and a two-tile window while `r` barely moves:

| | 2026-09-21 | 2026-09-22 |
| --- | --- | --- |
| TP4 width 2, M=24 | (2, 16) r=0.610 | (1, 8) r=0.611 |
| TP4 width 4, M=24 | (2, 16) r=0.679 | (1, 8) r=0.693 |
| TP2 width 8, M=24 | (2, 8) r=0.686 | (1, 4) r=0.686 |

Two are real: TP4 width 8 at M=384 moves from an 8-tile window (r=0.971) to the full stripe,
and TP2 width 8 at M=720 from 16 tiles to 8. Both are cells where windowing was worth under 3%
to begin with.

So a window table is a property of the machine model, not of the individual instance.

## 2. The per-panel term: both frozen readings refuted

`panel_term_c9g_20260922.md` left the identity inconclusive, missing only at routes 13 - one
full M12 panel plus a one-row tail - by 2.0-4.9%. Two accounts were frozen: the miss follows
the tail shape, or it grows with the panel count. Routes 25, 26, 37 and 38 separate them.

Relative error of the identity, mean of two passes at one thread:

| | 13 (1+1) | 25 (2+1) | 37 (3+1) | 14 (1+2) | 26 (2+2) | 38 (3+2) |
| --- | --- | --- | --- | --- | --- | --- |
| TP4 | +3.0% | +1.4% | +1.1% | -0.3% | -0.5% | -0.9% |
| TP2 | +5.0% | +3.3% | +2.8% | +4.7% | +3.3% | +2.3% |

Neither reading holds. Tail-shaped fails because TP2's two-row tails miss as much as its
one-row tails. Panel-shaped fails because the error **shrinks** as panels are added rather than
growing. In absolute microseconds the pattern is plain:

| | 13 | 25 | 37 | spread |
| --- | --- | --- | --- | --- |
| TP4, one-row tails | +23.0 us | +18.2 us | +18.3 us | 4.8 us |
| TP2, one-row tails | +79.8 us | +84.5 us | +100.2 us | 20.4 us |
| TP2, two-row tails | +74.1 us | +86.2 us | +82.2 us | 12.1 us |

The identity carries a roughly **fixed absolute offset** - about 19 us on TP4, about 85 us on
TP2 - and the percentage falls only because the total grows from 760 us to 1746 us. On TP4 the
offset appears on one-row tails alone (two-row tails sit at -0.3% to -0.9%); on TP2 it appears
on both. That asymmetry is unexplained, and the frozen rule admits neither account, so the
per-panel term remains unadopted.

## 3. Is 2T wrongly excluded? No, and not marginally

`second_machine_c9g_20260921.md` reported that with table windows 2T is the fastest width at
every M up to 192, by 21% at M = 96, and concluded that "E13's closing of 2T is a statement
about `Arm-codex`, not about the kernel". That conclusion is wrong.

Building each of 18 fresh layers' plan twice, with widths `(4,8,16,32)` and `(2,4,8,16,32)`,
gives the **identical plan on all 18** - the width search never picks 2T even when allowed. So
the question is not whether to re-admit 2T but whether the model is right to reject it. Two
further plans per layer force a homogeneous width past the search: 48 lanes of 2T, and 24
lanes of 4T as the control for forcing anything at all.

| | faster than production | median | best | worst |
| --- | --- | --- | --- | --- |
| forced 2T (48 lanes) | **0 of 18** | **+157.5%** | +48.0% | +257.2% |
| forced 4T (24 lanes) | 2 of 18 | +35.7% | -1.1% | +90.4% |

Not one layer prefers 2T, and the loss is more than a doubling. The 4T control loses too, so
this is about forcing a narrow homogeneous width, not about 2T specifically - 2T is simply the
extreme of it.

### Why the grid said the opposite

The grid gives every expert the same M and loads every lane equally; that is 2T's best case.
A real layer routes unevenly - these have 203-249 active experts with a heavy head - and 96
cores cut into 48 two-thread lanes put about five experts on each, so the makespan is pinned by
the heaviest lane. The imbalance cost dominates the cache benefit, which is the same thing E13
found on `Arm-codex` when the rule "keep small experts off 2T lanes" measured 5.10% slower.

**A homogeneous full-load grid ranks widths for that workload, not for real layers.** This is
the second time today that has mattered, and the 2026-09-21 report's width-transfer claim is
withdrawn on this evidence.

## Corrections to earlier reports

- `second_machine_c9g_20260921.md`: "the best lane width does not transfer" and the inference
  that E13 is machine-specific are withdrawn. The grid ranking is real; the inference from it
  to real layers is not. E13's conclusion holds on both machines, for the same reason.

## Limitations

- All of part 2 is at one thread. The 4-16 thread region is untouched.
- Part 3 uses the quick planner only; C9g has no event-model probe curves, so no reference
  search was run. A search that could build heterogeneous lanes might place a few 2T lanes
  profitably where a homogeneous plan cannot, and that is not tested here.
- The window table is composed but not registered. Registration needs a whole-plan validation
  of the kind E11 ran on `Arm-codex`.
