# Frozen repaired-model interior-node validation, 2026-09-10

## Question and frozen design

The user requested checking whether the repaired model still systematically underestimates between measured history nodes and mixture ratios. This class E optional Lab diagnostic freezes `HistoryMixRepair` (`full` variant) and all three underlying profiles before collecting new measurements. No fitting or parameter changes are part of this experiment; production/planner defaults remain unchanged.

Histories2/3/4/6/8 × M1 W13/W2 ×38-peer small counts0/8/12/19/26/30/38 plus no-background =80 cells. New histories3/6 lie between fitted2/4/8; new small counts12/26 lie between fitted8/19/30. The four disjoint competitive populations across two sessions are:

- Old history and old ratio controls:60 condition medians.
- New history only, ratios at fitted nodes:40.
- New ratio only, histories at fitted nodes:24.
- Both new:16.

All80 new-condition medians are independent of fitting;40 are history-only,24 ratio-only and16 both. Twenty no-background controls are separate. This population definition is fixed before inspection; do not average away errors of a particular axis. Compare mean absolute and signed errors, MAPE, maximum errors, underprediction counts and pointwise bootstrap median intervals. Intervals characterize timing samples conditional on the session; they are not simultaneous population confidence bounds or confidence intervals for the fitted model.

## Measurement protocol

Same original Arm-codex-internal host, CPU316 foreground;38 continuous1T W13 competitors on CPUs280–318 excluding316, NUMA3 memory, launch affinity240–319. Background M2 count nS is distributed by `(lane*nS)%38<nS`; all others run M120. The19+19 control retains the prior alternating layout. Only this deterministic layout is covered.

SVE256 BF16, H4096/F512, N tile16, full owner stripes `(1,0,0,1,1)`, full W13 B8MiB/W2 B4MiB. Four foreground and per-background B copies rotate. W13 degree5 fused SiLU, W2Direct degree0. Ordinary allocations without explicit HugeTLB,192MiB W2 output. M1 target A/C/routes fixed at row96; preceding full M12 panels advance through their own rows and use the target's same B. Backgrounds run through prefix and tail. Each condition prepares outputs, scrubs256MiB, waits for first background-call readiness and then5ms; no-background has the same5ms lead-in. Only tail duration is the target metric; prefix duration and actual gap are recorded.

All logical foreground prefix/tail outputs, untouched guards/routes and competitor outputs are checked; W13 packed padding is not a logical output. CPU/count/type/coverage checks run per condition. No PMU/cache-state identification. No explicit prefix/tail delay.

80-cell smoke, then two independent processes with5 warmup and31 formal randomized rounds, seeds605000/605001/605002. Analysis seed605100,10,000 bootstrap resamples per median. No outlier removal. All source/profile hashes and the planned grid are retained before collection.

## Reproduction and artifacts

Local `tmp/repair_interpolation_20260910/`; remote `Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/repair_interpolation_20260910/`. Native source/binary, driver, frozen protocol, build identity, raw JSONL and logs are retained as applicable. HEAD `c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5`, unrelated existing work preserved in before-status. Optional build uses the same GCC C++17/O3/pthread armv8.2-a+bf16+sve/SVE256 flags. Production JIT SHA256 `1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629`.

```bash
.venv/bin/pytest -q tests/test_moe_repair_interpolation.py
ssh Arm-codex-internal 'cd /home/zhangxu/codex/fused_cpp && bash tmp/repair_interpolation_20260910/build_smoke.sh'
ssh Arm-codex-internal 'cd /home/zhangxu/codex/fused_cpp && bash tmp/repair_interpolation_20260910/run_sessions.sh'
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_repair_interpolation.py \
  --protocol tmp/repair_interpolation_20260910/frozen_protocol.json \
  --sessions tmp/repair_interpolation_20260910/session1.jsonl tmp/repair_interpolation_20260910/session2.jsonl \
  --output tmp/repair_interpolation_20260910/analysis.json
```

Use fresh output paths on rerun. Source rollback consists only of the new native/driver/analyzer and their records. No commit/push or model adoption.

## Completed results

All80 smoke conditions and both80×36 grids completed:5,840 checked conditions including4,960 formal measurements. All build/native/runner stderr files are empty. Complete unique cell/round coverage and every frozen model identity passed. Native source SHA256 `c122d4521690a6567aa364ef875e04a9b1f1a679e2396c9b130ccc64669df823`, binary SHA256 `4bcf714a581d9b99fed55448c0b732ac69015f2e2f90725ebcf63d0668e31346`. The local grid/count test, Ruff, clang-format and diff whitespace checks pass.

Signed error is prediction minus measurement. Both sessions are included; the first four competitive populations are disjoint.

| Population | N | MAE (us) | MAPE | Signed mean (us) | Underpredicted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Old nodes repeated | 60 | 4.881 | 0.770% | +3.309 | 18 |
| New history only | 40 | 6.807 | 1.159% | +3.650 | 12 |
| New ratio only | 24 | 7.867 | 1.288% | -0.879 | 12 |
| Both new | 16 | 8.039 | 1.433% | +1.673 | 8 |
| All new nodes | 80 | 7.371 | 1.253% | +1.896 | 32 |
| No-background controls | 20 | 15.136 | 7.904% | -15.136 | 20 |

New-node MAPE is1.202%/1.303% in sessions1/2 respectively. Old competitive nodes have0.770% MAPE, so new nodes are somewhat less accurate but no longer exhibit the previous broad6–7% intermediate-history underprediction. Of80 new points,32 are underpredicted and48 overpredicted;22 predictions fall below the pointwise median95% interval and31 above. Neither cancellation in the signed average nor intervals including predictions establish exact unbiasedness.

## Persistent local bias

The ratio direction is repeatable across histories, stages and both sessions:

| Small/large ratio | N | Direction | MAE (us) | MAPE | Beyond median95% interval in that direction |
| --- | ---: | --- | ---: | ---: | ---: |
| 12 M2 +26 M120 |20|All overpredicted|8.078|1.450%|12|
| 26 M2 +12 M120 |20|All underpredicted|7.794|1.242%|17|

The26/12 underprediction ranges0.036%–2.525%; its maximum absolute error is15.203us. Thus local systematic underestimation remains, even though the prior large broad bias is reduced. The pattern is consistent with residual ratio curvature not represented by piecewise-linear interpolation, but type placement is also ratio-dependent under the fixed assignment policy. This experiment cannot assign causality solely to interpolation shape or cache behavior.

Worst new point is an overprediction: session2 W2/history6/12+26 is384.860us actual versus405.742us predicted (+20.882us,+5.43%). Its median95% interval is380.250–389.060us; session1 repeats the same direction at387.260us actual. A notable underprediction is session2 W13/history3/8+30:782.410us actual versus765.315us predicted (−17.095us,−2.18%), with median interval777.920–783.970us.

Maximum per-cell CV is9.460%; full p90/p99, mean/std and intervals are retained. The no-background controls are outside the competitive interpolation repair: all20 remain underpredicted, MAPE7.904%, and are neither hidden nor used to recalibrate the frozen no-background model.

## Decision

The frozen repair generalizes substantially better to the tested interior nodes than the previously rejected sparse-history extension, but does not eliminate local systematic bias. Retain it as a bounded Lab reference, without refitting on this validation set or asserting a universally unbiased response. Future correction should target the small ratio-dependent curvature and localized history6/W2 error, rather than applying a global positive offset: one new ratio is consistently high while its reverse is consistently low.

Scope remains M1 W13/W2, total38 competitors, tested history/fraction intervals and this placement policy. No other M, total-count, spatial-layout, real-plan or planner-performance claim follows. No model parameter, baseline or production default changed during this experiment.
