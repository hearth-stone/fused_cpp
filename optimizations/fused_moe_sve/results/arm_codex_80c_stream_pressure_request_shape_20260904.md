# Arm 80C fixed-count request-arrival shape

Date: 2026-09-04

## Technical summary

Request-arrival shape matters after packed-B count is fixed, but only once the
workload reaches the high-pressure regime. With eight distinct packed-B blocks,
8x1T is faster than start-aligned 8x2T by 0.113/0.077 ms in two independent
paired sessions, and its aggregate DDR queue latency is lower by 16.9/14.1
cycles. Both latency P90 and queue P90 remain negative in both sessions.

With four distinct blocks, narrowing 4x4T to 4x2T or 4x1T also lowers aggregate
queue latency, but the victim-span differences do not keep one sign outside the
noise interval. This rejects the proposed asymmetric signature in which narrow
teams hurt the victim despite a lower aggregate queue. The supported mechanism
is instead a two-dimensional injection pressure: distinct packed-B count sets
the number of independent weight streams, while per-stream team width sets how
many N-stripe requesters inject those blocks concurrently.

Do not turn the observed four-to-eight transition into a fitted count knee from
these two counts. No cost-model structure is added.

## Controlled geometry

All expert starts are held fixed inside each count group:

- Four blocks: starts 48/52/56/60, widths 4T, 2T, or 1T.
- Eight blocks: starts 48/50/52/54/56/58/60/62, widths 2T or 1T.
- The expert IDs, M=1 routes, packed-weight allocations, victim, LLC placement,
  and start positions are unchanged when width changes.
- Every session uses one long-lived process, four measured packed copies, one
  disjoint scrub copy, direct per-cell PMU reset/read, and randomized paired
  rounds.

Both sessions use 5 warmups and 31 measured rounds. All 60 PMU events per cell
have running ratio 1.0 and native overlap equals the fixed packed-B count.

## Four blocks remain below the stable shape-effect threshold

| Session | Shape | Peer threads | Paired slowdown vs isolated ms | Queue pressure vs isolated cycles |
| --- | --- | ---: | ---: | ---: |
| 20260922 | 4x4T | 16 | +0.0494 | +18.82 |
| 20260922 | 4x2T | 8 | +0.0423 | +17.06 |
| 20260922 | 4x1T | 4 | +0.0140 | +13.44 |
| 20260923 | 4x4T | 16 | +0.0512 | +20.94 |
| 20260923 | 4x2T | 8 | +0.0362 | +16.13 |
| 20260923 | 4x1T | 4 | +0.0728 | +11.12 |

Aggregate queue pressure decreases consistently as teams narrow. The direct
shape contrasts do not establish a victim-latency order:

| Narrower minus wider | Session-1 span median, P10/P90 ms | Session-2 span median, P10/P90 ms | Session-1 queue median, P10/P90 cycles | Session-2 queue median, P10/P90 cycles |
| --- | --- | --- | --- | --- |
| 4x2T - 4x4T | -0.0198 (-0.0490, +0.0180) | -0.0133 (-0.0447, +0.0257) | -1.54 (-5.56, -0.36) | -2.99 (-6.75, -0.37) |
| 4x1T - 4x2T | -0.0158 (-0.0586, +0.0134) | +0.0337 (-0.0082, +0.0664) | -3.34 (-5.51, -1.91) | -5.58 (-8.41, -3.93) |
| 4x1T - 4x4T | -0.0338 (-0.0618, +0.0026) | +0.0225 (-0.0122, +0.0512) | -5.34 (-8.07, -3.57) | -9.02 (-11.89, -6.21) |

The queue effect is real, but the resulting latency differences are too small
and session-sensitive at count four. No 4-block width correction is identified.

## Eight blocks show a stable injection-width effect

| Session | Shape | Peer threads | Paired slowdown vs isolated ms | Slowdown P10/P90 ms | Queue pressure cycles |
| --- | --- | ---: | ---: | --- | ---: |
| 20260922 | 8x2T | 16 | +0.1830 | +0.1223 / +0.2456 | +48.43 |
| 20260922 | 8x1T | 8 | +0.0639 | +0.0295 / +0.1137 | +29.28 |
| 20260923 | 8x2T | 16 | +0.1724 | +0.1357 / +0.2285 | +42.82 |
| 20260923 | 8x1T | 8 | +0.1022 | +0.0570 / +0.1433 | +28.63 |

The direct 8x1T-minus-8x2T contrasts are:

- Session 20260922: span `-0.1133 ms` with P10/P90
  `-0.1676/-0.0815 ms`; queue `-16.90 cycles` with P10/P90
  `-30.68/-7.65 cycles`.
- Session 20260923: span `-0.0775 ms` with P10/P90
  `-0.1411/-0.0247 ms`; queue `-14.06 cycles` with P10/P90
  `-20.61/-7.19 cycles`.

Both predeclared signs are stable in both sessions: fewer requesters per packed-B
block lowers aggregate DDR queue latency and lowers victim latency. Backend
stall ratio also falls by 8.1/5.2 percentage points in the two session medians;
the first session is fully separated, while the second P90 is approximately
zero. LLC miss differences are less stable and remain a guardrail only.

## Physical interpretation

One packed-B block is not one fixed-rate memory stream. A wider M=1 team reads
disjoint N stripes from the same block. It does not create additional distinct
weight blocks, but it does create more simultaneous request issuers and a higher
short-window injection rate.

The two dimensions are independently visible:

- At eight active peer threads, 8x1T produces substantially more queue pressure
  than 4x2T because it has twice as many distinct B blocks.
- At eight distinct B blocks, 8x2T produces substantially more queue pressure
  than 8x1T because it has twice as many active request issuers.
- At four blocks, the width-induced queue change is below the region where it
  produces a stable victim latency difference.

Therefore `distinct_transfer_bound_B_count` remains necessary but is not
sufficient. A future planner-visible proxy must also include request-injection
shape, minimally the active team-width distribution or total active owner
threads, and must model their interaction rather than adding them independently.

## Decision and limitations

The predeclared `narrower slower despite lower aggregate queue` signature is
rejected in every comparison. The positive result is narrower-faster-with-lower-
queue for 8x1T versus 8x2T in both sessions. No fixed-count comparison supports
an unexplained victim-asymmetric penalty.

This experiment does not identify a general formula. Only counts four and eight
were tested, and width changes necessarily change total active peer threads.
That thread change is the intended request-arrival intervention, but it does not
separate issue concurrency from other team-width effects such as stage duration.
Do not add a count knee, width multiplier, or empirical residual.

If continued, the smallest independent design is a count 4/6/8 by width 1T/2T
grid with start-aligned experts, plus per-stage overlap duration. It should test
whether a scheduler-visible injection proxy predicts the transition across an
unseen count, not fit another point on the current two-count contrast.

This locked count-6 holdout was completed later on 2026-09-04. No structural
proxy passed both sessions, so absolute-model expansion is now closed. See
`arm_codex_80c_stream_pressure_proxy_grid_20260904.md`.

## Artifacts

- Session 20260922:
  `tmp/moe_stream_pressure_shape_20260922.json`, SHA256
  `0614ae5b942a82572e649a7816ad4100e2bbe82d130dfee562bbab55da496ec8`.
- Session 20260923:
  `tmp/moe_stream_pressure_shape_20260923.json`, SHA256
  `8f209ae2310e1331db090cff342cc531e17902ca1056e47f731364a53fec3346`.
- Analysis:
  `tmp/moe_stream_pressure_request_shape.json`, SHA256
  `cf96cfaa348d2aaeec410b9ac702e2f6ce67b344074c5c4a1f8a687e177d6980`.
- Mode/bridge implementation SHA256:
  `2edaf350ea6a63c1ee819c30946e55083b91ee4bd9a311480615f71edcfb6029`.
- Request-shape analyzer SHA256:
  `da3315e3c1ecfa45f8fa87d07065ec60d2d78952048f94b007afa029e037d938`.

The paired runner, direct PMU helper, frozen calibration, extension, machine,
affinity, scrub, and event definitions are unchanged from
`arm_codex_80c_stream_pressure_paired_pmu_20260904.md`. No old route or
real-trace holdout was opened.
