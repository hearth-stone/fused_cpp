# 8T M12–48 transition/tail underprediction diagnostic, 2026-09-10

Read-only detailed analysis requested by the user. Reuse median/uniformish traces and current no-team/gather16 model; no new calibration, native measurement or planner change. The user hypothesized cold/transition/warm behavior behind the M12–48 error. Evidence supports investigating that mechanism, especially W2, but does not identify physical cache residency.

## Error changes at the first repeated B scan

Signed relative error is predicted cumulative stage time divided by observed cumulative stage time minus1, over expert/session medians in each M band. These are not paired M-sweep measurements at identical execution times. Counts differ by route workload.

| M band | Median W13 | Median W2 | Uniformish W13 | Uniformish W2 |
| --- | ---: | ---: | ---: | ---: |
| M1-11 | -4.70% | -3.76% | -2.64% | +0.74% |
| M12 | +2.25% | +3.86% | -0.01% | -4.36% |
| M13-24 | -4.50% | -15.93% | -9.40% | -20.81% |
| M25-36 | -5.97% | -8.09% | -11.80% | -9.51% |
| M37-48 | -2.40% | -5.79% | -7.70% | -7.35% |
| M49-96 | -3.09% | -4.11% | -7.11% | -4.85% |
| M97+ | -4.83% | -1.26% | -5.80% | -2.07% |

M12 is approximately correct, then W2 sharply underpredicts M13–24 by15.93%/20.81%. W2 mean absolute signed shortage is24.71/31.26us per task in that band, about18–22us in M25–48 and about18–25us in larger bands. The relative error declines as M grows. This is more consistent with missing early/restart/tail cost than a universal20%8T throughput error, although time-varying competition and different experts confound that interpretation.

Within M13–48, W2 tails of1–2 rows are short by roughly38–43us on average, tails3–4 by29–34us, while exact multiples of12 are short by13–15us. The exact-multiple comparison has few observations (median8 expert/session observations, uniformish4). This suggests both a transition issue and small-tail kernel response matter; it is not sufficient to infer a pure temperature law. Different M tasks run at different times under different peers; subtracting an unrelated M12 task from M13 does not measure the M1 tail directly.

## What the current model actually assumes

The current planner does not consume the six-family history functions for8T. The integration only patched whole M1/1T history0.8T still uses the original analytical stage model.

Per-thread owner B is1MiB for W13 and512KiB for W2. Calibrated L2 capacity is1.25MiB/core with effective fraction0.75 (960KiB). The reuse footprint is owner B plus one A panel: W13 adds96KiB, W2 adds12KiB. The model consequently assigns transient B L2 miss fraction0.5 for8T/W13 and **0 for8T/W2**. From the second W2 panel onward it models no B refill to L2; only the initial full stage4MiB is compulsory. Capacity fit is not proof of actual warmed residency or service behavior.

For M13/24/25/48, modeled total W2 B refill remains4MiB, whereas W13 is12/12/16/20MiB. The code does have transient-versus-steady reuse counts, but W2's transient count multiplies zero miss fraction and therefore adds no B traffic. It is not enough to point to the presence of a transient field as evidence that this transition is modeled correctly.

`_stage_phases` makes the first full panel `cold_b` and merges all remaining panels into one `steady_b` phase. This name does not prove physically hot B, and the W13 remainder still includes modeled transient refill. However, the aggregation prevents an explicit separate history-dependent response for each full panel and the final tail.

There is also a kernel-family approximation. All panel sizes use the same M12-calibrated gemm-core service scaled by work. In8T/W2, an M13/14 tail uses2 compute rows and receives about11.96us core time; a full12-row panel receives71.74us. Small-tail pipeline/overlap efficiency is not independently calibrated here. The mapper preserves exact/rounded rows, but that alone does not establish timing equivalence. Diagnostic frontend/L1 bounds for that tail are below its core bound, so merely enabling an existing larger bound is not an obvious complete fix.

## Counterfactual refill sensitivity, not a fit

For8T/W2 M13–48 only, add f times the full B size to L2 refill per repeated panel, f in0/0.25/0.5/1, updating corresponding L2/LLC/spillable traffic and rerunning the fixed-plan event simulation. All original files/parameters remain unchanged. This intervention says nothing about whether the physical refetch source is LLC or DRAM; DRAM increment still follows modeled spill. The f=0 case reproduces the current result.

| Refill fraction per repeat | Median M13–24 W2 bias | Uniformish M13–24 W2 bias |
| ---: | ---: | ---: |
|0|−15.93%|−20.81%|
|0.25|−15.32%|−19.69%|
|0.5|−11.60%|−14.43%|
|1|+3.40%|+3.18%|

Restoring a full scan is sufficient to cross the observed missing cost in this band, which supports sensitivity to the omitted B-service path. It is not proof that every repeated panel physically misses100%, and it overshoots some larger bands/cases. Fraction0.5 barely changes M25–48 predictions: the model's max(compute,transfer) and merged remainder hide extra transfer behind aggregate compute until a threshold is crossed. This limits fixing the problem by adjusting one miss fraction alone.

The strongest current conclusion is **the8T/W2 early-repeat and tail representation is too optimistic**. Candidate mechanisms are incomplete reuse/service transition and small-tail throughput/overlap; the existing trace cannot distinguish them. W13 also has broader underprediction beyond M48, so the entire8T error should not be relabeled as one B-temperature problem.

## Next discriminating measurement

A focused8T stage benchmark should time each full panel and exact tail separately, using the actual512KiB W2 /1MiB W13 owner stripes. Compare exact tail rows1/2/4/8/12 after0/1/2/3 and a later reference history, first in isolation and then with controlled8T background teams. Keep target A/C addresses fixed across history controls and verify every worker's output. The isolated comparison separates tail-family baseline from history; the competition axis tests whether the transition changes under peers. The previous1T evidence cannot be transplanted to8T because its per-thread B working set is8 times larger.

No new hardware experiment is claimed here. Artifacts under `tmp/gather16_model_20260910/`: `eight_transition.json`, `eight_refill_ablation.json`, `analyze8_transition.py`, `ablate8_refill.py`; rerun those scripts from the repository root. Input traces, shapes, CPU240–319/NUMA3, BF16 SVE256 H4096/F512, full stripes and two31-call sessions match the parent8T diagnostic. The first script retains exact M coverage and sample counts; the second retains all three intermediate bands for all four sensitivity settings. No code tests or production changes are needed for this read-only model diagnosis.
