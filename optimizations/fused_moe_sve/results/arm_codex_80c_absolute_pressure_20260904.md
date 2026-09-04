# Arm-codex 80C absolute gather-pressure probe

Date: 2026-09-04.

## Decision

The new count sweep is usable for joint absolute-plus-contrast fitting. Do not
fit yet against holdout. Keep frozen v8 and VND/LNS closed.

Local 1T 68-route aggressors on LLC7 produce a stable isolated-relative
slowdown that rises from one to about four streams and then saturates. Cross-LLC
aggressors on LLC6 remain a near-zero absolute effect. The same-minus-cross
contrast therefore tracks the local absolute curve, not a separate remote
penalty. A rank-wide split still looks like the local arm once four LLC7
streams are present.

## Method

- Machine: `Arm-codex-internal`, NUMA3 CPUs `240-319`, memory node 3.
- Affinity: `taskset -c 240-319 numactl --membind=3`.
- Calibration: frozen
  `analytic_machine_numa3_80c_narrow_merge_v8_20260903.json`, SHA256
  `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`.
- Extension SHA256:
  `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`.
- Probe: `bench_gather_injection_overlap.py` with
  `--aggressor-counts 0,1,2,4,8,15`.
- Fixed one-route 1T target on logical 64 / CPU304 / LLC7. Fifteen 68-route 1T
  background experts remain in every plan; unused experts are dependency-delayed
  after the target-lane tail.
- Placements: same-LLC on logical `65-79` / CPUs305-319 / LLC7; cross-LLC on
  logical `0-14` / CPUs240-254 / LLC6; split uses the pre-fixed rule
  `count//2` on LLC6 then the remainder on LLC7, only for counts 8 and 15.
- Five warmups, 31 randomized paired trace rounds, four rotating measured packed
  copies, and one dedicated disjoint 204 MiB scrub copy before every sample.
  Independent sessions used seeds `20260907` and `20260908`.

Raw ignored-workspace artifacts:

| Artifact | SHA256 | Use |
| --- | --- | --- |
| `tmp/moe_gather_absolute_pressure_fit_20260904.json` | `a9adb55381f99a1b2c2db934c692130d6ec28c07e26eadcda650070bb0b8abb1` | Session 1, fit-only |
| `tmp/moe_gather_absolute_pressure_repeat_20260904.json` | `b7d35dcda1f12e0c1afeccd72b45c2df42c9e84c895c5ae1b52ea9afcf06f625` | Session 2, validation-only |

## Data quality

Passed:

- Both sessions have 31 samples in every mode, including `isolated_*`.
- CPU mapping: logical 64 is CPU304 NUMA3 LLC7 (`280-319`); logical 0-14 are
  CPU240-254 NUMA3 LLC6 (`240-279`); logical 65-79 are CPU305-319 NUMA3 LLC7.
- Isolated controls have zero peer overlap.
- Same-LLC overlap expert count equals the requested aggressor count on every
  paired sample for `{1,2,4,8,15}`.
- After-1 modes have every active peer in `w13_fused_silu_packc` at target
  start, and none in gather.
- Head n15 reproduces the earlier gather-transition window: median 7 peers in
  `gather_pack_a` at target start, remainder overlapping later.
- Target stage times are stage envelopes. Extension and calibration hashes match
  the frozen-v8 / prior probe identity.

Expected near-zero instability, not a probe defect:

- Cross-LLC minus isolated changes sign between sessions. Remote medians stay
  within about `0.04 ms` of isolated, inside the isolated session shift of about
  `0.03 ms`.
- Isolated head/after medians are `0.555/0.537 ms` then `0.523/0.510 ms`. Local
  n15 spans stay put (`0.868/0.820` then `0.864/0.821`), so the cross-session
  isolated move inflates repeat absolute local-minus-isolated slightly but does
  not reverse locality contrast.

## Hardware result

Fit-session target-span medians, isolated-relative paired delta in parentheses:

| Count | same-LLC head | cross-LLC head | same-LLC after-1 | cross-LLC after-1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.556 | 0.556 | 0.537 | 0.537 |
| 1 | 0.699 (+0.143) | 0.563 (+0.008) | 0.682 (+0.142) | 0.563 (+0.024) |
| 2 | 0.753 (+0.195) | 0.564 (+0.005) | 0.731 (+0.191) | 0.556 (+0.013) |
| 4 | 0.846 (+0.289) | 0.566 (+0.010) | 0.814 (+0.276) | 0.562 (+0.024) |
| 8 | 0.855 (+0.295) | 0.588 (+0.029) | 0.815 (+0.276) | 0.561 (+0.022) |
| 15 | 0.868 (+0.310) | 0.599 (+0.041) | 0.819 (+0.279) | 0.573 (+0.036) |

Same-minus-cross paired contrast:

| Count | Head fit / repeat | After-1 fit / repeat | Head P10 fit / repeat |
| ---: | ---: | ---: | ---: |
| 1 | 0.136 / 0.140 | 0.117 / 0.144 | 0.107 / 0.123 |
| 2 | 0.197 / 0.218 | 0.171 / 0.218 | 0.173 / 0.189 |
| 4 | 0.279 / 0.319 | 0.254 / 0.314 | 0.239 / 0.292 |
| 8 | 0.270 / 0.320 | 0.257 / 0.309 | 0.238 / 0.291 |
| 15 | 0.267 / 0.310 | 0.254 / 0.317 | 0.252 / 0.278 |

The n15 head/after contrasts overlap the earlier two-session local-minus-remote
result of `0.244/0.248` and `0.225/0.233 ms`. Absolute local n15 spans also
match that probe (`0.878/0.826 ms` then; `0.868/0.819 ms` now). Most of the
local excess is W13: isolated-head W13 `0.369 ms` versus same-LLC n15 W13
`0.587 ms`.

Split n8/n15 is within `0.01 ms` of same-LLC n4/n15, not halfway between local
and remote. Four LLC7 streams already sit on the plateau, so mixing in LLC6
streams does not relieve the victim.

## Fitting implication

Session 1 may be used to jointly fit per-LLC-domain injection capacity and
gather offered-rate / coupling. Session 2 is locked as same-family validation.
The old route/context sweep and three real traces remain unread holdout.

A contrast-only fit is still forbidden: remote-minus-isolated is not a stable
positive series, while local-minus-isolated is. Any candidate that needs
`effective_gather_traffic_multiplier ≈ 22` or `capacity_scale = 0.787` from the
rejected schema-v11 pass should miss this count curve.

Next action completed and rejected: the session-1 joint fit is recorded in
[arm_codex_80c_absolute_pressure_joint_fit_20260904.md](./arm_codex_80c_absolute_pressure_joint_fit_20260904.md).
Do not freeze that candidate or read holdout.
