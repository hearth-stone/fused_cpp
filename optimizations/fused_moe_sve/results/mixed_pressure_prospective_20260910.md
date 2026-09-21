# Prospective mixture-ratio and intermediate-history validation, 2026-09-10

## Frozen hypothesis and scope

The user requested newly collected data to test whether the mixed correction generalizes to different ratios and intermediate histories. This class E diagnostic freezes all three existing model artifacts, records their hashes in `tmp/mixed_pressure_prospective_20260910/frozen_protocol.json` before collection, and performs no refit.

At total38 competitors, let f be the fraction of M2 competitors. The preregistered extension is:

```text
T_extended(h,f) = T_conditional(h,38*f,38*(1-f)) + 4*f*(1-f)*(a_stage+b_stage*h)
```

The coefficients are the previously fitted W13 `(45.073026,4.285658)` and W2 `(27.516140,2.671842)` in us. The gate preserves the original19+19 increment at f=0.5 and vanishes at pure endpoints. Symmetry in the increment is a hypothesis, not a claim that the underlying pure costs or hardware interactions are symmetric. The unchanged conditional model supplies the separate pure costs. The production wrapper that rejects unsupported mixtures is not relaxed by this diagnostic.

Grid: M1 W13/W2 × histories0/1/2/4/8 × five38-peer ratios (M2/M120:0/38,8/30,19/19,30/8,38/0) plus no-background =60 conditions. Histories2/4 and ratios8/30,30/8 are new. Repeated pure endpoints, old19+19 and no-background controls distinguish generalization error from run-to-run changes.

## Measurement method

Same original Arm-codex-internal machine and unchanged production JIT, foregroundCPU316, backgroundCPUs280–318 excluding316, NUMA3 memory, launch allowance240–319. M1 tail A/C/routes always start at row96. Prefixes are full M12 scans of the same B, advancing their A/C/routes. Backgrounds continuously execute W13 M2/M120 and cover prefixes plus the final timed tail. There is no inserted prefix/tail delay.

Background type for lane i is M2 when `(i*n_small)%38<n_small`, otherwise M120. This distributes the requested count across cores and exactly preserves alternating M2/M120 at19+19. Layout is deterministic; reverse/rotated layouts are not tested. Therefore results are conditional on this placement policy, not all possible spatial assignments.

SVE256 BF16, H4096/F512, backend N tile16, foreground1T and1T per competitor, full owner stripes `(1,0,0,1,1)`. Full W13 B8MiB/W2 B4MiB; four foreground and per-background B copies rotate. W13 degree5 fused SiLU; W2Direct degree0. Ordinary allocations, no explicit HugeTLB. Full W2 output192MiB. Each condition prepares outputs, scrubs256MiB, starts competitors and waits for first expert-call readiness, then waits5ms before prefixes. No-background uses the same5ms lead-in.

Every condition validates all logical prefix/tail outputs, untouched output regions and all competitor outputs; W13 physical padding is excluded from logical-output checks. Foreground CPU, background count/type/CPU, and coverage across the prefix-plus-tail interval are checked. Timing includes the tail call only; prefix and gap are recorded separately. No PMU/cache residency inference.

60-cell smoke precedes two independent processes with5 warmup/31 measured randomized rounds, seeds604000/604001/604002. Per-condition medians are compared with frozen predictions; p90/p99, mean/std/CV and bootstrap95% median intervals (10,000 resamples, seed604100) are retained. No outlier trimming. These are new measurements against a previously frozen formula; no parameter selection uses these results.

## Evidence and reproduction

Artifacts: `tmp/mixed_pressure_prospective_20260910/` locally and `Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/mixed_pressure_prospective_20260910/` remotely. Source snapshots, protocol/model hashes, build identity, logs and raw JSONL remain there. HEAD `c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5`; pre-existing changes preserved in `git_status_before.txt`. GCC C++17/O3/pthread, armv8.2-a+bf16+sve/SVE256 flags match previous independent targets. Production JIT SHA256 remains `1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629`.

```bash
.venv/bin/pytest -q tests/test_moe_mixed_prospective.py
ssh Arm-codex-internal 'cd /home/zhangxu/codex/fused_cpp && bash tmp/mixed_pressure_prospective_20260910/build_smoke.sh'
ssh Arm-codex-internal 'cd /home/zhangxu/codex/fused_cpp && bash tmp/mixed_pressure_prospective_20260910/run_sessions.sh'
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_mixed_pressure_prospective.py \
  --protocol tmp/mixed_pressure_prospective_20260910/frozen_protocol.json \
  --sessions tmp/mixed_pressure_prospective_20260910/session1.jsonl tmp/mixed_pressure_prospective_20260910/session2.jsonl \
  --output tmp/mixed_pressure_prospective_20260910/analysis.json
```

Use fresh output paths for reruns. No changes to production, planner baseline or frozen model parameters. Source rollback is confined to the new optional native/driver/analyzer and their records.

## Completed results

All60 smoke conditions and both60×36 formal/warmup grids completed:4,380 numerically checked conditions, including3,720 measured calls. All build/native/runner stderr files are empty. Complete unique grid/round coverage passed. Two local grid/formula tests, Ruff, clang-format and diff whitespace checks pass. Binary SHA256 `8c1e4bbb36eaf4171977d996924f87aed8dbc5a10dece5634a1d3a70905aad37`; source SHA256 `4a2edbef68e51c1667a70a867c613747a4806208fa18bde3ad6d2f4ce1dfac91`. All three frozen model hashes still match the pre-collection protocol.

Errors use both sessions' per-condition medians. Populations overlap and must not be added together.

| Population | N | Linear mixture MAE (us) | Frozen extension MAE (us) | Linear MAPE | Extension MAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| New8/30 and30/8 ratios, all histories | 40 | 45.918 | 23.793 | 7.492% | 4.150% |
| New histories2/4, all mixed ratios | 24 | 75.592 | 39.304 | 12.507% | 6.689% |
| Repeated19/19 histories0/1/8 | 12 | 47.488 | 3.810 | 7.207% | 0.605% |
| All mixed conditions | 60 | 51.563 | 21.847 | 8.290% | 3.763% |
| Pure competition controls | 40 | 15.765 | 15.765 | 3.176% | 3.176% |
| No-background controls | 20 | 20.180 | 20.180 | 9.564% | 9.564% |

The extension improves the broad mixed average, but it is not sufficiently general as a complete model. New ratios at only the old histories0/1/8 have MAE13.409us/MAPE2.416% (24 points). At new histories2/4, all24 mixed points are underestimated, MAE39.304us/MAPE6.689%. Session1/2 all-mixed MAPE is3.784%/3.742%, so the failure is repeatable. The old19+19 geometry repeats well at0.605% MAPE; its earlier low error was not evidence for unmeasured history interpolation.

The maximum mixed error is session2 W13/history4/8 M2+30 M120: actual795.330us, predicted728.400us (−66.930us,−8.42%). Its median95% interval is790.760–798.840us, far from the prediction; session1 repeats the error at792.550us actual. This is not explained by a single noisy timing sample. Maximum per-cell CV over the full grid is10.738%; p90/p99 and all median intervals remain in `analysis.json`.

## Diagnostic error separation (post hoc, not a deployable predictor)

After evaluating the frozen formula, substitute the same-session pure endpoint medians at the same history into the mixture baseline, while keeping the old correction coefficients unchanged:

```text
T_oracle_pure = (1-f)*measured_pure_M120(h) + f*measured_pure_M2(h)
                + 4*f*(1-f)*(a+b*h)
```

This uses newly observed times and is explicitly an oracle diagnostic, not prospective prediction and not a refit. `diagnose.py` and `diagnostic.json` retain the calculation.

- Pure controls at old histories0/1/8: MAE4.659us/MAPE0.812% (24 points).
- Pure controls at new histories2/4: all16 underestimate, MAE32.424us/MAPE6.721%.
- Middle-history mixed predictions after replacing pure endpoints: MAE falls39.304→16.830us and MAPE6.689→2.568%.

Thus the pure-pressure history interpolation accounts for a substantial part of the new-history error. It is not the only problem. In the worst W13/history4/8+30 case, oracle-pure prediction is737.001us versus795.330us actual, still−58.329us. Conversely, session2 W2/history2/30+8 improves from425.110us to485.286us versus489.750us actual: that case is largely explained by pure-history interpolation error. This distinction argues against correcting every error with one larger mixed coefficient.

## Decision and remaining scope

Retain the measurements as a bounded prospective reference. Do not promote the4f(1-f) extension as universally accurate, and do not alter the frozen model parameters using these validation points in this experiment. The data support further work on pure-competition history2–4 before interpreting all remaining errors as mixture effects; they also expose a ratio/history-specific mixed residual at W13/history4/8+30.

This grid varies ratios at total38 only, uses the deterministic spatial assignment described above, and covers only M1. Different total counts, layouts, other row families, real plans and physical cache mechanisms remain unvalidated. No planner baseline or production model changes were made.
