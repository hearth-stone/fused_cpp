# Arm 80C temporal ranking remediation and safe-decision gate

## Outcome

The temporal-order problem is closed for planner safety, but not for exact
sub-percent point ranking. A width-specific 1T expert overhead fixes the
identifiable long-lane bias, and an uncertainty-aware executable decision gate
prevents unresolved event deltas from replacing the incumbent.

Step-2 VND remains disabled. None of the current local neighbors has a robust
gain above the 2% action-resolution margin or a positive paired P10 hardware
speedup.

## Reproduction identity

- Model/calibration implementation: commit `e74e187c8be3`
- Uncertainty-aware gate: commit `d51cb0e9fc17`
- Formal run:
  `20260902T122957Z-arm_codex_internal_temporal_overhead-arm_temporal_order_overhead_validation-d51cb0e9fc17`
- Machine: `Arm-codex-internal`, NUMA node 3, CPUs `240-319`
- Workload: BF16 TP4, `H=4096`, `F=512`, `E=256`, 2048 tokens,
  TopK6 captured routes
- Backend: 256-bit Arm SVE BF16, direct-route W2
- Measurement: 5 warmups, 31 randomized paired rounds, 4 rotating packed
  weight copies
- Decision margin: 2% robust gain

Command:

```bash
.venv/bin/python optimizations/fused_moe_sve/paper_experiments/run_matrix.py \
  --machine optimizations/fused_moe_sve/paper_experiments/machines/arm_codex_internal_temporal_overhead.json \
  --suite optimizations/fused_moe_sve/paper_experiments/suites/arm_temporal_order_overhead_validation.json
```

The forced MoE build and focused correctness suite passed before measurement.

## Root cause

The original high-skew event model placed a 15-task 1T lane at `27.47 ms`,
while its runtime phase trace reached about `32.38 ms`. Moving one moderate
expert onto that lane left the point estimate below the predicted 16T critical
path but moved the real lane above it, causing up to 14.5% hardware regression.

Early merge was not the cause. The severe candidate remained slow with early
merge disabled: `38.784 ms` versus a `32.772 ms` baseline. The difference was
inside scheduled expert compute.

An independent three-state victim probe separated absolute 1T fragmentation
overhead from peer composition. It fitted the machine-local override:

```text
T_overhead(R, 1T) = 147999.019 ns + 10471.803 ns * R
```

No other width is changed. Missing widths retain the old global overhead.

## Calibration rerun

| 1T victim state | Frozen v4 prediction | Formal measured median | Error |
| --- | ---: | ---: | ---: |
| All peers delayed | 25.419 ms | 26.025 ms | -2.33% |
| Only 4x16T peers concurrent | 25.793 ms | 25.737 ms | +0.22% |
| 4x16T and 15x1T peers concurrent | 26.621 ms | 26.461 ms | +0.61% |

The formal rerun independently refitted `191051 ns + 9867 ns * R`; that drifted
fit was reported but was not used by the holdout. All frozen endpoint errors
remain below the predeclared 5% gate.

## Three-trace holdout

The decision score is:

```text
T_guard = max(T_event, (1 + relative_uncertainty) * max_lane(sum T_iso))
```

The lane term is a risk guard, not a certified hardware upper bound. A candidate
must improve `T_guard` by at least 2% to replace the incumbent.

| Trace | Full shape | Best robust gain | Decision | Measured shortlist regret | Point Spearman |
| --- | --- | ---: | --- | ---: | ---: |
| Uniformish | `10x8T` | 0.666% | Baseline | 0.242% | -0.048 |
| Median | `1x16T + 8x8T` | 0.872% | Baseline | 0.578% | -0.756 |
| High-skew | `4x16T + 16x1T` | 0.153% | Baseline | 0.516% | 0.690 |

No measured event-top candidate had a positive paired P10 speedup (`0/24`
stable improvements). Thus retaining baseline loses at most 0.58% relative to
the measured shortlist in this run and avoids every material regression.

## Interpretation

The model now captures the large, identifiable 1T serial-lane bias and improves
high-skew ranking from the original `-0.690` Spearman to `0.690`. It still
cannot reliably order plans whose predicted differences are below 1% under
heterogeneous phase interference. Median and uniformish correlations remain
poor, but their tested deltas are below the action-resolution margin and their
paired intervals overlap zero.

Therefore the defensible claim is:

- the heavy model provides an expected score and a conservative actionable
  region;
- unresolved candidates form an equivalence/measurement shortlist;
- the planner preserves the incumbent unless improvement exceeds the declared
  resolution.

It is not defensible to claim exact temporal ordering for every expert move.
Future work should use a stochastic/quantile event model or independent
measurements if sub-percent distinctions are required. A second Arm machine is
also required before treating the 1T overhead as portable.
