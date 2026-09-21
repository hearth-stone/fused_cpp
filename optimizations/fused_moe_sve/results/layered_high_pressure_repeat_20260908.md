# Frozen high-pressure model: repeat noise versus systematic error

## Result and decision

Three prospective sessions reproduce the high-pressure bias. Random variation
exists, especially at balanced48, but does not explain the persistent signed
errors of the frozen model under this protocol. Do not refit from these repeats
or dismiss the high-pressure validation failure as noise.

Foreground is real M1/1T W13, K4096/N1024; each background is an independent
M12/1T kernel. Balanced counts below are TOTAL across two LLC domains.
Error is `100 * (prediction / measured median - 1)`; positive means predicted
too slow. CV is within-session standard deviation divided by mean, not model
error or a confidence interval.

| Background placement/count | S3 / S4 / S5 median (us) | S3 / S4 / S5 model error (%) | Within-session CV range (%) |
| --- | --- | --- | --- |
| None | 304.61 / 302.03 / 304.55 | -0.59 / +0.29 / -0.56 | 1.05–1.32 |
| Same LLC, 39 | 764.90 / 766.62 / 769.01 | -0.90 / -0.78 / -1.73 | 1.27–1.87 |
| Other LLC, 39 | 421.85 / 422.68 / 428.46 | -0.18 / +0.05 / -1.59 | 1.63–2.41 |
| Balanced, 32 | 534.61 / 538.52 / 532.96 | +5.64 / +5.59 / +4.01 | 3.86–5.35 |
| Balanced, 48 | 932.06 / 910.32 / 942.71 | +13.62 / +13.37 / +13.91 | 4.59–5.96 |
| Balanced, 64 | 1674.74 / 1675.76 / 1698.91 | -3.91 / -3.96 / -5.50 | 1.59–1.60 |
| Balanced, 78 | 2054.04 / 2047.48 / 2052.46 | -16.12 / -15.44 / -15.99 | 1.52–1.84 |

Balanced48 has a central80% time width of 8.90–12.65% of its median, whereas
balanced78 has only 3.55–4.40%. Across the three session medians, CV is 1.45%
at48 and 0.14% at78. These descriptive comparisons alone are not a statistical
test: the joint time/feature bootstrap below checks the residual directly.

| Balanced count | S3 block4 error CI95 (%) | S4 | S5 |
| --- | --- | --- | --- |
| 48 | [12.63, 15.42] | [11.33, 15.03] | [12.12, 15.38] |
| 64 | [-5.53, -3.16] | [-5.10, -3.45] | [-6.51, -4.50] |
| 78 | [-16.73, -15.61] | [-16.25, -14.87] | [-16.56, -15.56] |

All three new sessions retain the same error direction for each high-pressure
condition, and their conditional block intervals exclude zero. Historical
S1/S2 corroborate this:48 was +15.47/+14.21%,64 -3.48/-5.31%,78
-14.23/-15.57%. Historical data are not counted as new prospective evidence.
Balanced32 also retains a smaller +4.01–5.64% bias; do not claim all lower
pressure placements are now accurate.

The mechanism is still unresolved. The high-pressure queue range is outside
the original fit range; extrapolating the low-pressure interaction response
is one plausible failure mode. Global DDR queue occupancy per read command
also need not equal foreground-specific service delay: aggregation across
controllers and time can hide which requests waited. These are hypotheses,
not newly identified physical causes. Variable request arrival or arbitration
can contribute to round variation, but this experiment does not isolate it.

Next, inspect the frozen residual against per-controller queue features and
foreground cache-path features in these existing raw rounds, without fitting
to this validation set. Only then design a separate training grid and an
untouched prospective holdout for a high-pressure response candidate. Keep
the current measured-feature model bounded; it is not a plan-visible predictor
and must not be exported to neighbor pruning.

## Protocol and question

Does high-pressure sharing randomness explain the failed48/64/78-background
predictions, or does the frozen model retain a systematic residual?

Three new independent hardware processes use the EXACT existing dual-LLC
binary and runner, not a rebuilt kernel or reduced grid. Full85-cell grid,
5 warmups +31 measured rounds, seeds289808/299808/309808. Foreground real
M1/1T W13 atCPU304, controller240, balanced background M12/1T up to39+39
cores on NUMA3. Same allocations per process,4-copy rotation,256MiB scrub
per LLC and5ms lead-in. All old protocol/correctness/placement/PMU checks
apply. New files are session3/4/5; original session1/2 remain historical.

The layered model, its selected interaction form, coefficients and isolated
anchors remain frozen. New sessions were collected AFTER that model was
frozen. No fitting, threshold choice or outlier removal in this experiment.
This is class E measurement/validation only; production andplanner unchanged.

## Noise analysis

Focus conditions are isolated, same39, other39, balanced32,48,64,78; the full
hardware grid is retained to avoid changing interleaved workload context.
For each focus cell/session:

- Report31-round median, mean, standard deviation, CV, P10/P90/P99 andrange.
  P10–P90 width is divided by the median. Quantiles use linear interpolation;
  with31 observations P99 is descriptive, not a precise tail guarantee.
- Recompute the frozen model from the cell's median PMU feature ratios exactly
  as in the fit/replay tool, and verify aggregation parity.
- Jointly bootstrap the SAME round's time, miss/refill ratio, queue ratio and
  DDR rate.2000 IID resamples and2000 circular block-length4 resamples retain
  time/feature correspondence; block4 is a short-dependence/copy-cycle check.
- Keep model parameters fixed throughout bootstrap. These intervals estimate
  conditional measurement variation, NOT calibration-parameter uncertainty,
  arbitrary long-range dependence or universal hardware randomness.
- Compare new-session median distributions and signed errors separately from
  the combined five sessions. Same-sign errors with all block-bootstrap
  intervals on one side of zero are evidence against purely zero-mean repeat
  noise under this protocol, not identification of a unique physical cause.

## Reproduction and identities

Remote root `/home/zhangxu/codex/fused_cpp`, host Arm-codex-internal.
Measured source/binary remain in `tmp/dual_llc_m12_20260908/`; new raw files
in `tmp/layered_high_pressure_repeat_20260908/`, mirrored locally (ignored).

```sh
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/dual_llc_m12_20260908/bench_phase_supply.py \
  --binary tmp/dual_llc_m12_20260908/phase_supply_native \
  --output tmp/layered_high_pressure_repeat_20260908/session3.jsonl \
  --dual-m12 --seed 289808
```

Repeat session4/seed299808 andsession5/seed309808. Do not overwrite original files.

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_layered_repeat.py \
  --model tmp/layered_supply_model_20260908/frozen_fit.json \
  --historical tmp/dual_llc_m12_20260908/session1.jsonl tmp/dual_llc_m12_20260908/session2.jsonl \
  --repeats tmp/layered_high_pressure_repeat_20260908/session3.jsonl \
  tmp/layered_high_pressure_repeat_20260908/session4.jsonl \
  tmp/layered_high_pressure_repeat_20260908/session5.jsonl \
  --output tmp/layered_high_pressure_repeat_20260908/report.json
```

Binary SHA256: `740117153e96fb39cd715fa40f7d4b5767694659095699c4365a0b3f4ed3e070`.
Frozen model SHA256: `f3796a0f7e815dc5c5e297b3bafdb3bb64150358daa50150f91bec6897157d86`.
Build andhardware provenance remain in `dual_llc_m12_20260908.md`; the model
definition/split andearlier error are in `layered_measured_supply_20260908.md`.

## Validation

Targeted tests42 passed across repeat analysis, layered model, dual-M12 andM12
modules. Synthetic tests distinguish bias without noise from correctly predicted
joint time/feature variation, and verify the model is unchanged. Ruff passes.
All three complete85-cell sessions passed the existing raw protocol gates;
five distinct raw identities/seeds, binary identity, median aggregation parity,
and frozen model identity were checked. Analysis is saved in
`tmp/layered_high_pressure_repeat_20260908/report.json`. Model coefficients,
native benchmark, production kernels, planner and calibration were unchanged.
