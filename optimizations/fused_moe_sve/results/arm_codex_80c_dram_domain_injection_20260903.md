# Arm-codex 80C DRAM domain-injection probe

Date: 2026-09-03.

## Decision

The missing physical resource is primarily a per-LLC-domain memory-injection
limit, not a first-order gather-versus-stream resource split. The analytical
structure is retained as calibration-optional and default-off. No parameter is
frozen because stacking it on the frozen-v8 wide/narrow residuals fails the
pre-existing route-context holdout.

## Method

- Machine: `Arm-codex-internal`, NUMA3 CPUs `240-319`, memory node 3.
- Hardware LLC topology: CPUs `240-279` are LLC6 and CPUs `280-319` are LLC7.
- Calibration: frozen
  `analytic_machine_numa3_80c_narrow_merge_v8_20260903.json`, SHA256
  `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`.
- Extension SHA256:
  `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`.
- BF16 fused expert, `H=4096`, `F=512`, full W13/W2 owner-stripe windows.
- Fixed one-route 1T target on CPU304 and fifteen 68-route 1T aggressors.
- Local modes place aggressors on CPUs305-319 in LLC7. Remote modes place
  aggressors on CPUs240-254 in LLC6. Tasks, routes, weights, and total work are
  otherwise identical.
- Head modes start the target with aggressors. After-1 modes put one identical
  one-route expert before the target; trace verifies that all 15 aggressors are
  in W13 when the target begins.
- Five warmups, 31 randomized paired trace rounds, four rotating measured
  packed copies, and one dedicated disjoint 204 MiB scrub copy before every
  sample. The synchronous scrub is untraced and outside target timing.

Raw ignored-workspace artifacts:

| Artifact | SHA256 | Use |
| --- | --- | --- |
| `tmp/moe_gather_injection_overlap_20260904.json` | `7a5f048926621f8775b15565aa3118d8d0c9189eda2f56d2e4e8d423aa7697c1` | Excluded topology mistake: both arms were inside LLC7 |
| `tmp/moe_gather_injection_overlap_cross_llc_20260904.json` | `aea4736bb6a3b97d60dbaaee8aabf63b2e3837150259f9ebf7c8f5f180ebea1c` | Cross-LLC fit session |
| `tmp/moe_gather_injection_overlap_cross_llc_repeat_20260904.json` | `f111549d33fe3cc221a9cd932c385cda6c9036d4e9cfedcd0b9b2d49485681dd` | Independent repeat |

## Hardware result

Target-span medians from the fit session:

| Mode | Target span | Target W13 | Peer phase at target start |
| --- | ---: | ---: | --- |
| isolated head | 0.675 ms | 0.452 ms | none |
| local head | 0.878 ms | 0.593 ms | 7 gather, remainder transitioning |
| remote head | 0.635 ms | 0.436 ms | 8 gather, remainder transitioning |
| isolated after-1 | 0.663 ms | 0.446 ms | none |
| local after-1 | 0.826 ms | 0.554 ms | 15 W13 |
| remote after-1 | 0.596 ms | 0.398 ms | 15 W13 |

The paired local-minus-remote contrast is stable across phase and session:

| Context | Fit median / P10 | Repeat median / P10 |
| --- | ---: | ---: |
| head, gather-to-W13 transition | +0.244 / +0.221 ms | +0.248 / +0.230 ms |
| after-1, all peers in W13 | +0.225 / +0.212 ms | +0.233 / +0.210 ms |

Remote aggressors do not produce a stable positive penalty relative to the
matched isolated control: fit head/after deltas are `-0.049/-0.065 ms`, and the
repeat head delta is `-0.012 ms`. The robust effect is therefore locality, and
it persists when gather overlap is removed. A separate gather/stream resource
may be a second-order correction, but it is not the first missing resource.

## Analytical prototype and holdout

The placement-aware allocator now has an optional domain injection ceiling:

```text
C_domain(n) = min(C_rank_curve(n), beta * C_rank_saturated / number_of_LLC_domains)
D_domain = max(1, offered_domain / C_domain)
D_task,dram = max(D_rank, max(D_domain touched by task))
```

The event explanation records domain injection active threads, offered rate,
capacity, utilization, and dilation. Missing calibration leaves the old
rank-only DRAM behavior unchanged.

With the existing gather prototype and `beta=0.76`, the predicted
local-minus-remote head/after contrasts are `+0.145/+0.235 ms`, versus measured
`+0.244/+0.225 ms`. This reduces mean absolute contrast error from about
`0.230 ms` with the rank-only model to `0.055 ms`, but the head/after mismatch
shows that one cap still does not fully model gather/stream coupling.

More importantly, stacking this term on frozen-v8 residual corrections worsens
the earlier 25-point `M={1,2,5,6,12}` five-context holdout MAPE from `21.7%` to
about `28.5%` (`effective_traffic_multiplier=0.5`, `beta=0.76`). Removing the
old wide/narrow corrections is worse. The old residuals already absorb part of
the same contention and therefore double count it.

Decision: retain the analytical resource structure and probe, do not persist
`beta`, do not modify the frozen calibration, and do not connect schema-v10 to
VND/LNS. The next calibration must jointly refit or replace wide-team pressure,
narrow-team correction, and gather pressure, using this probe for fit and the
route sweep plus three real traces as holdout.
