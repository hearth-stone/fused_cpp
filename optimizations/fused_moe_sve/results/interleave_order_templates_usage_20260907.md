# Opt-in interleave templates and broader order proposals

Implementation: `bench_bounded_order_extension.py freeze --proposal-set`.
This is a bounded Lab proposal interface, not a production selector or a new
VND automatic-acceptance/pruning domain. Model formulas, lane ownership, widths,
stage windows, early merge and public PlanV2 remain unchanged.

## Stage 1: explicit interleave templates

`--proposal-set interleave` generates:

- `large_small_lane_stagger`: sort tasks by routes, with adjacent lane parities
  receiving opposite ascending/descending orientations; try both parity choices.
  This reproduces the synthetic LLLHHH/HHHLLL pattern when a lane has three
  small and three large experts, and extends to arbitrary route distributions.
- `large_small_alternating`: alternate lowest/highest remaining route groups in
  blocks of one or two tasks; opposite lane parities start from opposite ends.
  Preserve original ordering within equal-route groups.

Equal-route lanes are unchanged by these templates. Routes are a proposal rank,
not a calibrated memory density. These are not explicit route-threshold/core
partition candidates and never move an expert to another lane.

## Stage 2: broader order transforms

`--proposal-set extended` adds four families:

- independently reverse a nonempty subset of eligible lanes;
- cyclically rotate one or more lanes;
- exchange head/tail jointly in at least two lanes;
- relocate an intact contiguous task block within one or more lanes.

Each stochastic family has at most 256 attempts and 24 unique proposals.
Template families naturally have at most two/four proposals. All operators
preserve complete task objects and regenerate dependencies through existing
ExecutablePlanState lowering. No inserted idle time, resize or route slicing.

## Budget and evidence boundary

Global canonical-state deduplication removes anchor/no-op and already selected
duplicates. Each family first identifies its model-best and, when possible,
maximum-order-distance representative. For new modes reserve one slot per family
before filling remaining slots. Extended mode uses model-best for both templates,
independent reversal and joint head/tail, and diversity for rotation and block
relocation (fallback to the only representative if necessary). This avoids making
every reserved slot depend on model-top rank. At most six candidates plus anchor
are frozen for hardware; templates cannot be displaced by many local proposals.

Artifacts retain `family`, `moved_experts`, `selection_policy`, full bridge/hash,
`proposal_set`, pre-cap representative count and hardware budget. Labels survive
selection, but duplicate plans may have only their first retained family label.
No guarantee that the true best plan lies in this budget.

The default remains `legacy`: the earlier three-family protocol and selected
plans are preserved. Earlier source snapshots, frozen artifacts and measured
results are not overwritten. New families are not silently attached to the old
partial-order residual calibration; no VND/LNS default or production hook changes.

## Commands

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/bench_bounded_order_extension.py freeze \
  --proposal-set interleave \
  --source-plans tmp/selection_pair_20260906/plans.json \
  --winner tmp/selection_winners_20260906/median.json \
  --calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --frontier tmp/interleave_templates_20260907/interleave.json
```

Replace `interleave` with `extended` and choose a fresh output path for stage2.
The unchanged `measure --frontier ...` interface can consume either output on
the target machine, subject to the existing identity, correctness and budget
checks. This implementation task does not launch a new hardware experiment or
claim new performance. Existing synthetic/template measurements motivate the
families but do not validate every new candidate.

Validation: focused generator/state tests, exact synthetic route patterns,
equal work/geometry preservation, determinism/deduplication, and actual frozen
median generation. Artifacts live under ignored `tmp/interleave_templates_20260907/`.

Completed checks: 16 focused tests passed. Actual median freeze produced four
plans for interleave (including anchor, after deduplication) and seven for
extended (anchor plus one representative from each of six families). Legacy
freeze reproduced every previously saved candidate bridge hash exactly. No
new hardware measurement or commit was made.
