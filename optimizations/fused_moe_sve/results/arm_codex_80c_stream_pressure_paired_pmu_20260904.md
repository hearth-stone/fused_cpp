# Arm 80C same-process paired stream-pressure PMU

Date: 2026-09-04

## Technical summary

The same-process, same-allocation, randomized-paired protocol removes the large
isolated-baseline drift seen in the process-per-cell experiment. DDRC queue
latency is now the clear standalone PMU feature: it is positive at every count,
has anchored LOCO MAE 0.0515 ms versus 0.0778 ms for victim LLC miss ratio, and
also wins the free-intercept sensitivity. Victim LLC miss pressure is negative
at counts 1, 2, and 4 and has three nonpositive anchored predictions.

DDRC queue latency is still not accurate enough to freeze. The measured
slowdown has a sharp transition between four and eight distinct packed-B
blocks, while a linear queue feature overpredicts counts 1--4 and underpredicts
counts 8--16. Keep queue latency as the physical source and next feature
candidate; do not add a knee or empirical residual from this session.

## Paired protocol

- One long-lived Python process holds every plan, input, output, four rotating
  measured packed-weight allocations, and one disjoint scrub allocation.
- Every round contains isolated, 1x16T, 2x8T, 4x4T, 8x2T, 16x1T, and 4x1T once
  in randomized order.
- Before every cell, the disjoint allocation touches all 18 expert W13/W2
  blocks while counters and tracing are disabled.
- Linux `perf_event_open` opens 60 events once. Every cell performs reset,
  enable, fused call, disable, and read.
- The Python control thread is pinned to CPU240 after warmup; the victim remains
  on CPU304. This keeps control syscalls off the victim core counter.
- Core PMU, all ten LLC7 L3C slices, all eight NUMA3 DDRCs, and native target
  traces are sampled in the same call.
- Main session: seed 20260920, 5 warmups, 31 randomized paired rounds.
- Independent layout repeat: seed 20260921, isolated plus 4x1T, 31 paired rounds.

All 60 events in every cell report `time_running/time_enabled=1.0`. Trace counts
are 31 per mode and overlap medians are exactly 0/1/2/4/8/16; 4x1T reports four.

## The count response has a four-to-eight-stream transition

Each delta below is the median of 31 same-round candidate-minus-isolated
differences, not a difference between independently measured medians.

| Distinct B | Cell | Absolute span ms | Paired delta ms | Delta P10 ms | Delta P90 ms | Queue pressure cycles | Queue P10 | Queue P90 | LLC miss pressure |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | isolated | 0.5833 | 0.0000 | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0 | 0.0000 |
| 1 | 1x16T | 0.5909 | +0.0082 | -0.0169 | +0.0354 | +11.1 | +9.0 | +13.8 | -0.0073 |
| 2 | 2x8T | 0.6120 | +0.0295 | +0.0037 | +0.0597 | +19.0 | +14.4 | +21.8 | -0.0133 |
| 4 | 4x4T | 0.6211 | +0.0455 | -0.0004 | +0.0693 | +21.0 | +17.9 | +22.8 | -0.0294 |
| 8 | 8x2T | 0.7821 | +0.2040 | +0.1616 | +0.2521 | +41.8 | +34.5 | +45.8 | +0.1385 |
| 16 | 16x1T | 0.7798 | +0.2042 | +0.1466 | +0.2696 | +36.0 | +30.9 | +43.7 | +0.0631 |

Queue pressure is stable and positive even where the latency delta is near the
noise floor. The victim slowdown and queue pressure both jump between four and
eight streams, then plateau. LLC miss pressure does not provide a stable low-
count signal: its P10--P90 crosses zero for counts 1, 2, 4, and 16.

This strengthens the causal attribution to DDR request queueing. It does not
prove that a hard count knee is a portable model parameter; count also changes
team width and request-arrival shape along this ladder.

## Paired leave-one-count-out comparison

The primary fit remains a nonnegative, isolated-anchored single-feature model.
No count is removed after observing its result.

| Metric | DDRC queue latency | Victim LLC miss ratio |
| --- | ---: | ---: |
| Anchored LOCO MAE | 0.0515 ms | 0.0778 ms |
| Anchored LOCO RMSE | 0.0528 ms | 0.0860 ms |
| Anchored maximum error | 0.0677 ms | 0.1220 ms |
| Nonnegative pressure at every count | yes | no |
| Nonpositive predictions | 0 | 3 |

Queue fold errors for counts 1/2/4/8/16 are
`+0.0418/+0.0598/+0.0528/-0.0352/-0.0677 ms`: the linear form is too high below
the transition and too low above it. LLC fold errors are
`-0.0201/-0.0515/-0.0967/+0.0985/-0.1220 ms` and fail both magnitude and
physical-direction checks.

A free-intercept sensitivity gives queue/LLC LOCO MAE `0.0355/0.0432 ms`, RMSE
`0.0398/0.0540 ms`, and maximum error `0.0595/0.0801 ms`. Allowing an intercept
does not reverse the winner or make either feature precise enough to freeze.

## Independent 4x1T confirms queue but exposes layout residual

| Session | Paired slowdown ms | Slowdown P10/P90 ms | Queue pressure cycles | Queue prediction error ms | LLC prediction error ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Main 7-mode session | +0.0744 | not used as repeat gate | +9.92 | -0.0307 | +0.2149 |
| Independent 4x1T repeat | +0.0878 | +0.0329 / +0.1374 | +13.50 | -0.0283 | +0.0706 |

The independent repeat has a stable positive slowdown and stable positive queue
pressure. Queue underpredicts it by about 0.028 ms in both sessions. This is a
repeatable layout/request-shape residual, not evidence to add a post-hoc
constant. The count ladder uses four 4-thread peers; the layout control uses
four 1-thread peers.

## Decision

1. Accept the measurement protocol. It removes cross-process baseline drift and
   provides complete per-cell PMU reads.
2. Reject victim LLC miss ratio as a standalone pressure feature.
3. Retain DDRC queue latency as the primary physical feature, but reject the
   current linear quantitative model.
4. Do not add a count knee, queue threshold, two-feature fit, or layout residual
   from this session.
5. Do not read the old route/real-trace holdout, replace frozen v8, or connect
   the candidate to VND/LNS.

If modeling continues, the next independent probe must hold distinct-B count
fixed while varying request-arrival shape, particularly 4x4T versus 4x1T and
8x2T versus 8x1T on a fixed core subset. Its purpose is to identify a scheduler-
visible proxy for queue pressure, since production planning cannot read future
PMU values.

This fixed-count gate was completed later on 2026-09-04. Count eight shows a
stable wider-team injection effect; count four remains below the resolvable
latency threshold. See
`arm_codex_80c_stream_pressure_request_shape_20260904.md`.

## Artifacts

- Main paired JSON:
  `tmp/moe_stream_pressure_pmu_paired_s1.json`, SHA256
  `daa18699f28384a9ea4e5ccdc0ebc6917b9712b514908906b2447d22c7894787`.
- Independent 4x1T JSON:
  `tmp/moe_stream_pressure_pmu_paired_4x1_repeat.json`, SHA256
  `93fbc337c3d8fd9b6b15efaa2d83e1a1641b0932c5fdd8eed624ce4f9a00f7bb`.
- Paired LOCO JSON:
  `tmp/moe_stream_pressure_pmu_paired_loco.json`, SHA256
  `d9cf92abab07d6f33562121bcb054c910ec5ec6d43662b958708f79f5dabd66a`.
- `linux_perf_event.py` SHA256:
  `5265c4e0c0d1709b6141ec4ad49fc97769c19432064da1565996d6505662e184`.
- paired runner SHA256:
  `842e672fc2fb8963f8a99e8f4d8ced5e713e5a78d26e6d56adf56bd760137424`.
- paired analyzer SHA256:
  `371285d84455ed9a15450f8fbba673fbdf761a2d8ef3c2bb70d17b72fb486cfa`.

Frozen calibration and extension identities remain those recorded in
`arm_codex_80c_stream_pressure_pmu_20260904.md`. No existing holdout artifact
was opened.
