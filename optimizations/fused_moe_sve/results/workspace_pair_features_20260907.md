# Workspace pair-feature diagnostic

## Decision

Keep frozen v8 and the existing LNS path unchanged. Do not add a dedicated
smoothing generator or authorize pruning. Isolated offered-density differences
are useful diagnostic candidates, but neither lane-load nor a monotone density
penalty explains all retained comparisons. No model was fitted in this study.

## Evidence and reproduction

Input: all four profile-bound suites in
`tmp/workspace_baseline_confirmation_20260907/`: median, high_skew, uniformish,
and the independent smoothing confirmation. Only these eight workspace sessions
are used as timing evidence; older allocation-lifecycle measurements are excluded.
Historical template-LNS frontiers provide route counts only, with matching route
SHA and layer. Output: `tmp/workspace_pair_features_20260907/validated_analysis.json`.

```bash
PYTHONPATH=.:src .venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_workspace_pair_features.py \
  --input-dir tmp/workspace_baseline_confirmation_20260907 \
  --calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --route-frontier-dir tmp/moe_partial_order_vnd_20260904 \
  --output tmp/workspace_pair_features_20260907/reproduction.json
```

Use a fresh output filename. The analyzer verifies session completeness,
correctness flags, profile identity, full bridge hashes, route coverage and
independent lane-chain dependencies. Recomputed placed predictions match the
stored frozen scores to relative tolerance 1e-10. Source hashes are retained in
the output; current dirty model source is not modified by this diagnostic.

Baseline: `profiles/workspace_numa3_80c.json`; Arm-codex-internal, NUMA3,
CPU240–319; E256/H4096/F512, 2048 tokens, TopK6, BF16, merge on. Actual backend
Ntile16. Full W13/W2 weight bytes per expert: 8MiB/4MiB; full owner stripes,
geometry `(t,0,0,1,1)` with plan-specific widths. Fixed 2048-token pre-touched
workspace, same-process shared weight allocations per suite, four-copy rotation,
216MiB scrub, five warmups, 31 paired rounds, two independent seeds, trace off.
Real route distributions, synthetic weights/inputs; not production end-to-end.
Full original session commands and source/build identities are linked from
`arm_codex_80c_workspace_baseline_confirmation_20260907.md`.

## Definitions and sample limits

- Pair gain: median of `100*(anchor_time/candidate_time-1)` within matched rounds,
  separately for each session; not the ratio of latency medians.
- Stable direction: the faster orientation has median gain >2% and P10 >0 in
  both sessions. P10 is a round quantile, **not a confidence interval**.
- Stable inversion: stable hardware direction opposite to the frozen point-score
  direction. This is not a newly calibrated partial-order error rate.
- All unordered pairs within each suite: 34 records, 31 unique physical plan
  pairs, 15 unique plans across three route traces. Twenty-one records have a
  stable direction; after bridge-pair deduplication, twenty unique pairs remain.
- Three inversion records are only **two unique inversions**: the median pair is
  independently reproduced in the smoothing suite, with reversed orientation.
- These are deliberately selected confirmation sets. Pair observations share
  plans and sessions; 20 pairs do not constitute 20 independent experiments.
  They are not restricted to small neighbors: width-changing LNS and weak-start
  comparisons are included. No held-out accuracy, full top-K recall, false-pruning
  safety or causal DDR claim follows from these counts.

## Stable counterexamples

| Anchor → candidate | Frozen gain | Paired hardware gains | Anchor medians ms | Candidate medians ms |
| --- | ---: | --- | --- | --- |
| median old anchor → elite2ab | -3.770% | +2.568%, +2.426% | 30.640, 30.797 | 29.994, 30.106 |
| high-skew old smooth anchor → VND-greedy | +3.088% | -8.849%, -9.465% | 30.142, 30.220 | 33.094, 33.183 |

The old smooth anchor is a historical control, not the current high-skew elite.
The VND candidate improves its own greedy parent (+4.645%, +3.118%) but remains
far slower than that stronger control. Parent-relative usefulness does not imply
global superiority.

### Median: isolated load points in the wrong direction

Old anchor → elite2ab:

- Isolated maximum lane duration: 22.912 →24.198ms (+1.286ms).
- Sum of isolated thread-time: 1827.010 →1829.571 core-ms (+0.14%).
- Offered-density CV: 1.055 →0.648; domain second moment: 16230.6 →12140.3
  (-25.2%). Total modeled bytes are unchanged.
- Width histogram changes from 1×16T+8×8T to
  2×16T+4×8T+2×4T+3×2T+2×1T.
- Isolated critical lane begin changes32→16, but its final expert remains115,
  M714. Top-two isolated lane gaps are only0.036/0.046ms.

The load increase alone cannot explain hardware speedup. Reduced offered-density
variation is a compatible hypothesis, not causal proof: width, ownership and
isolated horizon also change. The extremely small top-two gap makes a hard
single-critical-lane label fragile. Actual workspace task timelines were not
collected in these trace-off sessions.

### High-skew: similar load, different execution context

Old smooth anchor → VND-greedy:

- Isolated maximum lane duration: 23.507 →23.520ms (only +0.013ms).
- Isolated core-work: 1873.033 →1880.444 core-ms (+0.40%).
- Offered-density CV: 0.512 →1.662; domain second moment:
  14511.8 →31711.9 (+118.5%).
- Width histogram changes4×16T+16×1T →5×16T.

These features expose a large context difference that maximum lane-load largely
misses. However, v8 already models some concurrency/order effects; this result
does not mean contention is entirely absent from v8.

An especially useful amplitude control is old smooth anchor → full_reference:
frozen gain +11.950%, measured -0.145%/+0.190%, despite identical isolated
lane loads/core-work and much higher CV (0.512→0.973). Thus a large modeled order
effect is unsupported by this hardware comparison. This pair is not labelled a
stable inversion because hardware does not resolve an actionable difference.

## Descriptive feature screen, not fitted accuracy

On the twenty unique stable pairs, test only the simple sign hypothesis that
the lower feature value belongs to the hardware-faster plan. Treat changes
within1e-8 in the feature's native units as ties. Duplicate pair evidence is
counted once; its second-session-group replication is retained separately.

| Feature | Direction agrees | Disagrees | Ties |
| --- | ---: | ---: | ---: |
| Isolated maximum lane time | 5 | 11 | 4 |
| Isolated summed core-work | 14 | 2 | 4 |
| Offered-density CV | 18 | 2 | 0 |
| Offered-density global second moment | 19 | 1 | 0 |
| Offered-density domain second moment | 18 | 2 | 0 |
| Peak offered rate | 17 | 2 | 1 |

Density uses the existing `density_profile(..., "gemm")` implementation: retain
non-GEMM intervals, average consecutive phases within W13/W2, and place all lanes
on their isolated clocks. Domain assignment is fractional for lanes crossing
CPU279/280. These are **unconstrained offered rates**, not measured DDR bandwidth
or actual simultaneous hardware execution. Changed horizon affects moments;
they are not pure smoothness measures across width-changing plans.

Counter-controls prevent a monotone penalty conclusion:

- high-skew elite189/61 are3–4% faster than old smooth anchor, although CV rises
  0.512→0.557/0.581 and domain moments also rise. These are the two CV exceptions.
- high-skew full/smooth have strongly different density proxies but nearly tied
  hardware times (above).
- median old anchor → explicit smooth has unchanged isolated loads but gains
  +2.586%/+2.474%; frozen prediction is only +0.031%. This contrasts with the
  overestimated high-skew order effect, so a universal order multiplier is also
  unsupported.

## Next bounded decision

1. Retain these two unique inversions, the high-skew tied full/smooth control,
   median old/smooth, and correctly ordered greedy→VND as diagnostic targets.
2. Before fitting, separate fixed-width order changes from width/ownership
   changes. Retain maximum/second-maximum load and near-critical lanes; do not
   replace them with density alone.
3. For these fixed existing bridges, collect workspace phase timelines only if
   deeper mechanism evidence is needed: compare actual W13/W2 duration changes
   and terminal lanes against isolated-clock predictions. Keep trace-on timing
   separate from trace-off winner evidence.
4. A candidate difference correction may combine density with width/context,
   but use a separate plan-family/anchor-disjoint dataset before claiming
   predictive benefit. Repeated pairs/sessions are replication, not train/test
   independence. Do not fit on these selected counterexamples and report them
   as validation. Exact DDR queue prediction is not a prerequisite.

## Validation

`PYTHONPATH=.:src .venv/bin/python -m pytest -q tests/test_moe_workspace_pair_features.py tests/test_moe_bounded_order_extension.py tests/test_moe_pressure_balanced_orders.py`
passed25 tests. New checks cover gain orientation/warmup exclusion, fractional
cross-domain lane accounting, and rejection of cross-lane dependencies.
Ruff passes. No new hardware run, calibration fitting, model formula change,
search-policy change, production adoption or commit.
