# Planner comparison with a bounded M1 pressure adapter, 2026-09-10

## Scope and mechanism

The user requested running the planner with the repaired model. This class M/E Lab experiment adds an optional Python adapter and comparison runner. It does not replace the production or active experimental baseline. Rollback consists of the new adapter/runner and records.

The planner models whole-expert stage phases, whereas the repaired kernel model describes individual blocks. This first integration changes only whole M1 tasks on1T: GEMM phase baseline totals for each stage are rescaled to the frozen no-background M1 history0 cost; operator/gather/setup phases remain as before. Other M and widths retain their original isolated predictions. **The history2/4 repair and large-expert M1 tails are not activated by this adapter.** This is a partial integration, not a full six-family/panel-history planner.

During placed event simulation, count same-LLC active GEMM cores excluding the victim. Classify peer tasks with M≤12 as small and M>12 as large; all active cores of wide teams contribute. Both peer stages use the same classification. This approximates arbitrary running tasks with the calibrated W13-only M2/M120 competitors and is explicitly unvalidated. For n≤38 use the existing conditional count response at history0; add `(n/38)*R_stage(history0, small_fraction)` from the repaired38-peer mixed residual. At n>38 clamp count to38 while preserving fraction. This count scaling, classification threshold, stage transfer, wide-team per-core equivalence and clamping are experimental assumptions, not measured generalization.

The phase multiplier is the resulting M1 kernel time divided by its frozen zero-pressure time. It replaces, rather than multiplies, the baseline M1 pressure multiplier to avoid double counting. Existing resource allocation remains in the simulator and may affect other tasks as the M1 timing changes. All non-M1 phase formulas remain unchanged. Target baseline h0 values are W13=306.45us and W2=186.90us. All kernel/model profiles are retained unchanged.

## Search protocol

Use the three original high-skew/median/uniformish workloads in `tmp/model_search_compare_20260909/input.json`, with the original calibration, coefficients, route/layer and extension identities. The comparison is current CorePressureModel+mean-best versus the M1 adapter+mean-best, early mergeoff, no dynamic tail pool or bounded repartition. Each full search has141 shapes and422 DAG calls. The candidate DAG identities may differ because the M1 isolated costs change task assignment; identical shapes/counts constitute the equal budget, not an identical candidate population. Cross-score the union of selected bridges with both models.

An initial check incorrectly required identical DAG sequence hashes and stopped after both high-skew searches. Their completed outputs were retained; the check was corrected to require equal shape list and DAG count, and subsequent cases resumed. No completed search was relabeled or rerun to select a favorable result. Search runners and initial failure logs are retained. Planning was run locally on the same machine for both models; reported wall times are one-run diagnostics, not a target-Arm cold-planner speed claim.

## Hardware protocol

Export each selected-plan union to the established `bench_bounded_order_extension.py` runner. Rename the baseline choice to `anchor`; deduplicate identical bridges. On Arm-codex-internal, bind to CPUs240–319 and NUMA3 memory. Use the frozen workspace profile `tmp/workspace_phase_timeline_20260907/workspace_numa3_80c.json`, fixed pretouched workspace,2048 tokens, topk6, H4096/F512,256 experts, BF16 SVE256, backend N tile16, full owner stripes. Per1T full stage B is8MiB W13/4MiB W2; wider teams use their full owner stripes with zero window tiles. Runtime allocation policy follows the retained workspace profile; no new page-policy change.

Before timing, check bitwise output equality across both plans and four weight copies. Each of two sessions has5 warmup and31 measured randomized paired rounds, four rotating weight copies and216MiB scrub. Seeds606001/606002 for high-skew,606011/606012 median,606021/606022 uniformish. Primary metric is the established traced compute-start to last-W2 completion interval, with early mergeoff; final merge/E2E is separate. No untraced control or production runtime rebuild is introduced in this experiment.

## Artifacts and reproduction

Local `tmp/planner_repaired_pressure_20260910/`; remote same relative directory under `/home/zhangxu/codex/fused_cpp/` on Arm-codex-internal. Input/profile/source identities, complete search ledgers, rankings, selected bridges, raw traces and measurement logs are retained. Existing unrelated workspace changes are preserved in `git_status_before.txt`. No commit or push.

```bash
.venv/bin/pytest -q tests/test_moe_planner_repaired_pressure.py
.venv/bin/python optimizations/fused_moe_sve/benchmarks/run_repaired_planner_search.py \
  --input tmp/planner_repaired_pressure_20260910/input.json \
  --output-dir tmp/planner_repaired_pressure_20260910/search
.venv/bin/python optimizations/fused_moe_sve/benchmarks/prepare_repaired_planner_hardware.py \
  --root tmp/planner_repaired_pressure_20260910 --case high_skew
ssh Arm-codex-internal 'cd /home/zhangxu/codex/fused_cpp && bash tmp/planner_repaired_pressure_20260910/high_skew/run.sh'
```

The search runner can resume existing own outputs; use a new directory when inputs or implementation change. Repeat hardware preparation/execution for median and uniformish with fresh destinations. Two local adapter tests and the3-shape/9-call high-skew smoke pass; wider numerical/production parity claims are out of scope. Source and calibration identity must accompany any repeated result.

## Completed search and measured effect

All six full searches completed with141 identical shape templates and422 DAG evaluations per model/case (2,532 total evaluations). Ordered-DAG identities differ because isolated M1 cost changes alter assignment. Both models use minimum predicted makespan selection; no fallback selector was introduced. High-skew selected shape remains4×16T+16×1T and LPT, but24 M1 experts change placement;13 switch between1T and16T. M1 experts assigned to1T decrease from12 to9. Median selects1×16T+8×8T reverse-odd and uniformish10×8T reverse-even; their baseline/repaired bridges are byte-identical, with no1T tasks.

Paired compute-start to last-W2 medians (ms):

| Route | Session | Baseline selected | Repaired selected | Paired delta (repair−base) | Paired reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| high_skew | 1 | 29.30121 | 28.65395 | -0.64220 | 2.193% |
| high_skew | 2 | 29.34714 | 28.62815 | -0.71134 | 2.418% |
| median | 1 | 28.33023 | 28.33023 | +0.00000 | 0.000% |
| median | 2 | 28.49462 | 28.49462 | +0.00000 | 0.000% |
| uniformish | 1 | 27.82209 | 27.82209 | +0.00000 | 0.000% |
| uniformish | 2 | 27.83001 | 27.83001 | +0.00000 | 0.000% |

High-skew paired delta95% intervals are[−0.68260,−0.62115]ms and[−0.75248,−0.70010]ms, both below zero. Baseline last completion is lane69 in all62 samples; repaired last completion is usually lane77/78 (61/62 samples), with lane69 once. This identifies a shift in the critical lane, not proof of the full physical contention mechanism. Identical-plan cases have zero difference by identity; they are not independent model-arm samples.

### Common-plan prediction accuracy

Compare both models on the same four unique plans × two sessions, eight observations:

| Model | MAE (ms) | MAPE | Mean signed error (ms) | Maximum absolute error (ms) |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.65475 | 5.766% | +1.65475 | 2.68661 |
| repaired | 1.63829 | 5.709% | +1.63829 | 2.65636 |

Absolute prediction changes little. All eight predictions still overestimate actual compute time. More importantly, cross-plan ranking remains wrong: the repaired model predicts the baseline high-skew bridge at31.13890ms and its own selected bridge at31.28451ms, preferring the baseline by0.14561ms even though hardware prefers the repaired bridge by0.64–0.71ms. The original baseline model also predicts31.17447 versus31.31476ms, the same wrong direction.

The repaired search's minimum score over its422 evaluations is31.28451ms. Therefore the baseline selected bridge's better repaired-model score is not represented in its retained candidate population. Model-dependent assignment changed the candidate pool despite equal budget. Reranking the union with the repaired objective would choose the slower baseline bridge. The observed selected-plan speedup must not be presented as proof of accurate contention ranking; it results from the changed search/assignment under this partial adapter.

### Coverage, overhead and validation

Across the repaired full search, M1 phase-response calls are181,948/110,535/76,937 for high-skew/median/uniformish. Nonzero-pressure calls explicitly counted as extrapolated are181,869/110,343/76,818; maximum same-LLC peer cores39, clamped to the38-peer endpoint. These are event-evaluation counts, not measured runtime coverage. They make clear that nearly all competing planner queries extend beyond the exact microbenchmark protocol.

Local baseline/repaired search wall times in seconds are73.86/78.22,110.45/113.32,117.09/121.49. These are one run each in a shared local Python process, with imports/model construction excluded; they are not an Arm cold-planner overhead benchmark. Future resume is guarded by input/model/source identity locks; original completed high-skew outputs were validated and retained before installing that lock.

All four unique runtime plans pass four-copy bitwise correctness. Six traced sessions validate248 formal compute calls, plus warmups and correctness calls. All six trace hashes match their recorded session identities. All hardware and summarizer stderr files are empty; the separately retained initial search stderr records only the overly strict DAG-identity check described above. Bridge roundtrip, full stripes and early-mergeoff checks passed. Two focused adapter tests and the3-shape smoke passed; Ruff and diff checks pass. No production/native-planner parity or untraced E2E speedup is claimed.

## Decision

Retain the experiment as a bounded positive selected-runtime result for high-skew (+2.19%/+2.42% paired speed), with unchanged median/uniformish plans, almost unchanged absolute prediction error and wrong relative ranking explicitly retained. Do not switch the active baseline automatically. Full panel-history integration, other small M responses, real competitor mapping, count/stage/wide-team transfer and correct common-plan ranking remain unvalidated. This run does not test the repaired history2–4 curves inside the planner.
