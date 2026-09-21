# Arm 80C quick versus fixed 8T after wide-team occupancy ranking

## Scope

Change class: Lab hardware confirmation of the v1.85 quick occupancy scale.
Frozen v8 SHA is unchanged. Plan V2, kernel, and ABI are unchanged.

Question: does multiplying homogeneous quick makespan by the existing
\(B_t/S_t\) occupancy term stop the documented 40T mis-rank on active-set 8/16,
and does that restore hardware versus fixed 8T?

## Protocol

- Arm-codex-internal NUMA3 `numactl --physcpubind=240-319 --membind=3`
- Calibration `7928ba96…` (frozen v8, already contains \(B_t/S_t\))
- Extension `dd554ea3…`
- 5 warmups, 31 randomized paired rounds, 4 rotating packed copies
- Gain `100 * (fixed8 / quick - 1)`; positive means quick is faster
- Bit-exact outputs before timing
- Occupancy snapshot: 118 `tokio-rt-worker` threads plus docker/devkit on
  CPUs 240–319 (not exclusive)

Focused remote tests before measurement: 13 passed.

Command:

```text
numactl --physcpubind=240-319 --membind=3 \
  .venv/bin/python optimizations/fused_moe_sve/benchmarks/bench_quick_vs_fixed8.py \
  --analytic-calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --warmup 5 --runs 31 --weight-copies 4 --seed 20260906
```

## Results

| Workload | Quick shape | Quick ms | Fixed 8T ms | Paired median | P10 / P90 |
| --- | --- | ---: | ---: | ---: | --- |
| uniform | 20x4T | 36.687 | 35.520 | -4.17% | -10.20 / +0.50 |
| **active-set-8** | **10x8T** | 32.422 | 32.471 | **-0.03%** | -2.86 / +5.34 |
| **active-set-16** | **20x4T** | 33.140 | 32.914 | **+0.30%** | -5.20 / +3.21 |
| active-set-32 | 20x4T | 33.934 | 33.327 | -1.52% | -6.26 / +0.39 |
| active-set-64 | 20x4T | 35.434 | 31.259 | **-11.90%** | -14.09 / -10.68 |
| active-set-128 | 20x4T | 34.433 | 30.674 | **-10.92%** | -12.19 / -9.74 |
| tiered-hotspot | 20x4T | 32.886 | 28.216 | **-14.25%** | -15.67 / -12.28 |
| long-short-bimodal | 5x16T | 32.524 | 42.204 | **+30.26%** | +29.04 / +35.03 |
| dsv4-real-2048-seq70 | 10x8T | 34.007 | 34.071 | +0.26% | -0.93 / +1.03 |

Historical operator-only (v1.30, same 80C catalog, then selecting 40T on
active 8/16): uniform/active8/active16 `-2.86/-15.54/-11.31%`; bimodal
`+22.13%`.

## Decision

The original 40T mis-rank is gone on the two failing workloads: active-set-8
now chooses 10x8T and is tied with fixed 8T; active-set-16 no longer chooses
40T. That specific regression is closed at both selection and hardware.

The occupancy proxy then over-narrows denser catalogs to 20x4T and opens new
stable losses of 11–14% on active-set-64/128 and tiered. Bimodal still needs
16T and still wins. Quick still does not dominate fixed 8T.

Do not claim the production quality gate is closed. Next width work must stop
the 4T over-penalty without bringing 40T back.

## Artifacts

Remote: `/home/zhangxu/codex/fused_cpp/tmp/moe_quick_vs_fixed8_20260906/`
`result.json` SHA256 `264da4fb6629ff2b2ddfad0bea916792f4849b1266b65955c5a3ac0f8b97a982`
