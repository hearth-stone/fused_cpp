# Arm 80C planner-visible queue geometry (post-hoc)

## Scope

Change class: Lab cost-model diagnostic. This replay does not change frozen v8,
Plan V2, production scoring, or the original four-proxy rejection.

The locked count `{4,6,8} x width {1T,2T}` sessions already showed that
distinct-B count, requester threads, their product, and a measured-W13-overlap
oracle fail the count-6 holdout. The remaining question was whether a
**planner-visible** geometry — live teams plus frozen isolated phase times,
never measured overlap — would have passed the same gates.

This is post-hoc on a published holdout. `accept` is forced empty even if a
proxy had passed. It cannot reopen `stop_absolute_model_expansion`.

No hardware was run. The command:

```text
PYTHONPATH=.:src .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/analyze_stream_pressure_plan_geometry_proxies.py \
  --session tmp/moe_stream_proxy_grid_20260924.json \
  --session tmp/moe_stream_proxy_grid_20260925.json \
  --calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --output tmp/moe_stream_pressure_plan_geometry_proxies_20260905.json
```

Session SHA256 values match the original proxy-grid report:
`f687afb9…284a` and `d88a49a2…3c07`. Calibration SHA256 is frozen v8
`7928ba96…aad3`.

## Predeclared new candidates

Single-feature, through-origin, same count-4/8 fit and count-6 holdout:

| Proxy | Planner-visible definition |
| --- | --- |
| `streams_x_sqrt_width` | \(n_B\sqrt{t}\) |
| `predicted_peer_w13_core_ms` | \(\sum_i t_i\,T_{iso}^{W13}(M{=}1,t_i)\) from frozen v8 |
| `predicted_peer_operator_core_ms` | same with isolated operator/gather overhead |

Plus one two-coefficient diagnostic, still planner-visible:

\[
\widehat q = a\,n_B + b\,n_{\text{threads}}
\]

Gates are unchanged: count-6 queue error \(\le\max(3\text{ cycles},15\%)\),
slowdown error \(\le 0.02\,\mathrm{ms}\), 1T/2T direction, and \(\le 20\%\)
cross-session parameter drift.

Isolated M=1 phase times from frozen v8: W13 `0.243478/0.190612 ms` and
operator `0.158471/0.094709 ms` for 1T/2T.

## Results

Queue-to-slowdown slopes remain `0.00440/0.00476 ms/cycle` (7.5% drift). Every
new proxy-to-queue map drifts 27–62%.

| Proxy | S1 holdout | S2 holdout | Slope/coeff drift | Accept |
| --- | --- | --- | ---: | --- |
| \(n_B\sqrt{t}\) | fail (Q 4.32/7.56) | fail 6x2T Q 3.32 | 27.5% | no |
| predicted W13 core-ms | fail 6x2T Q 8.57 | fail 6x2T Q 3.94 | 27.9% | no |
| predicted operator core-ms | fail (Q 6.51/5.52) | **pass** | 26.9% | no |
| \(a n_B + b n_{\text{threads}}\) | fail 6x2T Q 8.68 | **pass** | stream 33.6%, requester 62.4% | no |

Session 2 can look solvable: operator-core-ms and the two-coefficient fit both
clear that session's count-6 cells. Session 1 does not, and the coefficients
are not the same map.

The structural reason is already in the training cells. Identical 8x2T geometry
measured queue pressure `48.168` vs `29.142` cycles (ratio **1.653**). A
deterministic \(g(P)\) is the same in both sessions, so its proxy-to-queue
slope cannot be stable.

## Decision

`accepted=[]`. Keep `stop_absolute_model_expansion=true` and frozen v8. The
planner can name the two dimensions that *cause* pressure (distinct packed-B
streams and requester width), but it cannot compute the session's DDR queue
state from those dimensions on this grid.

Further plan-only algebra on these six cells is not justified. A new
planner-visible source would have to vary when the same live-team layout
produces 48 vs 29 cycles of queue; that source is not in the executable plan.

## Artifacts

- Analyzer: `optimizations/fused_moe_sve/benchmarks/analyze_stream_pressure_plan_geometry_proxies.py`
  SHA256 `691225a03d62a1520d55501e57ef51e292605d042b837020eafb592a7cbb199b`
- Tests: `tests/test_moe_stream_pressure_plan_geometry_proxies.py` (4 passed with the original proxy-grid tests, 8 together)
- Decision JSON (ignored workspace): `tmp/moe_stream_pressure_plan_geometry_proxies_20260905.json`
  SHA256 `9982242a5879bea6548908c57176c39a1541d11a89f8d7ff89f35fdf2b5ddb52`
