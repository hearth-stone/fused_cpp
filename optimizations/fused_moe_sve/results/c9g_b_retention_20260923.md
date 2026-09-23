# C9g packed-B retention: measured, and not the multi-panel fix

## Status

The C9g analytic calibrations never ran the packed-B repeated-scan retention probe that
`ANALYTIC_MODEL.md` requires; they carried the builder defaults (effective fraction 0.75, miss
floor 0, miss 1 at 2 and 4 MiB), i.e. perfect retention below 1.5 MiB. The probe is now
measured. Its values are close to Arm-codex's and, under a rule frozen before collection, adding
them to the calibration does **not** fix the multi-panel under-prediction. Rejected; recorded.

## The error being chased

With the shipped median calibrations, isolated experts whose owner B window fits the effective
L2 (TP4 >= 8T, TP2 >= 16T) are under-predicted once they have a second M12 panel and
over-predicted below it:

| TP4, error of the shipped model | routes 12 | routes 13 | routes 25 |
| --- | --- | --- | --- |
| 1T | +1.7% | -6.4% | -13.4% |
| 8T | -3.7% | -27.5% | -21.5% |
| 16T | +1.7% | -19.3% | -16.8% |

The sign flips exactly where the model's B-reuse miss switches to zero, and the measured cost of
an added panel at 8-96T is close to B bytes at the LLC service rate. That pointed at retention.

## Probe

`bench_single_core_weight_window` (production `kernels.S`, `n_tile = svcnth()`), K=4096, one
distinct cold packed B per call, CPU 0 node 0, five independent processes each for M=12 (one
panel) and M=120 (ten panels). `miss = (L2 refills per weight line at M120 - at M12) / 9`,
against W = weight window + one 12-row A panel. `perf` was elevated with `sudo -n` for the
counting process only (`perf_event_paranoid=4`); `stall_backend_mem` was dropped from the event
set because it is not counted in short M12 runs, which the stock runner refuses.

| W (MiB) | 0.34 | 0.72 | 1.09 | 1.47 | 1.84 | 2.09 | 3.09 | 4.09 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| miss per reuse | 14.3% | 16.3% | 19.7% | 26.2% | 51.6% | 72.3% | 74.6% | 86.6% |

Fit (the model's own smoothstep form): floor 15.2%, knee 0.63 x L2, 64.5% at 2 MiB, 85.4% at
4 MiB. Arm-codex (2026-08-31): 17.4%, 0.662, 68.0%, 79.8%.

## Frozen score

Both calibrations rebuilt locally with exactly the registered procedure (median service probe,
training profile, train routes 12/192/2040, per-width overhead fit) - reproduced byte-identical
without the retention values - then with them. Scored on isolated points not used in training:
the dense sweep, the pairs round and the triplet round, 434 points per shape.

| | shipped | with retention |
| --- | --- | --- |
| TP4 mean absolute error | 7.66% | 7.84% |
| TP2 mean absolute error | 10.55% | 12.58% |
| TP2 at 8T | 8.76% | 19.39% |
| signed error, routes 13-30, >= 8T (TP4 / TP2) | -10.6% / -17.6% | -10.8% / -19.9% |

Adoption needed a 25% relative improvement on both shapes, no width worse by more than 1 point
and the knee bias halved. It fails every clause.

## What this rules out

The multi-panel under-prediction is not panel-to-panel B falling out of L2: one core re-reading
a 1.1 MiB window misses about a fifth of its lines per reuse, while the added-panel cost at
8-48T looks like the whole B at LLC rate. The cost has to come from something the model does not
charge per panel. Candidates, unattributed: a thin panel still issues the owner's whole B stripe
through L1 regardless of its rows; a width-independent residue of about 25-30 us per added panel
on TP4 at 16-48T; and, from the triplet round, an in-panel two-row increment the model prices
1.5-3x too high. Attribution comes before any model change, and it is parked behind the
one-click calibration work.

The defaults are still wrong for this machine, so the one-click calibration should run the
probe rather than inherit them; on the evidence here, doing so does not move the planner's
isolated accuracy.

Lab: `tmp/c9g_b_retention_20260923/` (design, runner, fit and score, decision).
