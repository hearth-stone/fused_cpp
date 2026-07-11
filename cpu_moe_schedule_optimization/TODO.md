# MoE scheduling implementation checklist

The policy-aware TP2/EP2 scheduling update is implemented as one change set.
Cross-profile interpolation deliberately remains disabled; a real routing dump
can be supplied to `validate_policy_planner.py --routes-json` before enabling
that separate feature.

## 1. Schema-v2 profiling

- [x] Synchronize two NUMA-local ranks and aggregate pairwise wall maxima.
- [x] Use M12-aligned contention routes plus M1/M2/M4/M8 isolated tail points.
- [x] Stream distinct expert weights; use all local experts for full-call
  contention and eight experts for isolated streaming cost.
- [x] Use the runtime planner's LPT lane assignment for mixed-width shapes.

## 2. Policy-aware exact-shape cost model

- [x] Key `T_iso` and contention data by the complete profile policy: sharded F,
  split-W13 mode, kernel identity, NUMA topology, and concurrent-rank count.
- [x] Use exact measured shape data by default instead of collapsing all shapes
  with the same active-expert count into one derate.
- [x] Preserve once-per-call cost with authoritative full-call anchors rather
  than multiplying normalized group cost by the number of waves.
- [x] Make route lookup M12-aware, including the M1/M2/M4/M8 tail kernels.
- [x] Reject incompatible, ambiguous, or grid-mismatched profiles.

## 3. Joint policy and schedule planner

- [x] Search `(w13_split, core_shape)` rather than core shape alone.
- [x] Use packed working-set bytes to prune candidates, while retaining measured
  latency as the objective.
- [x] Return split policy as an explicit plan field and replace the
  process-global environment-variable control with an operator argument.
- [x] Add explicit physical CPU sets, a policy-aware cache key, full bucketed
  routing signatures, and confidence-aware tie breaking.

## 4. TP/EP layer evaluator

- [x] Select compute profiles from TP/EP degree, sharded F, local expert count,
  and machine topology instead of hard-coded TP4/EP4 files.
- [x] Evaluate global compute wall time as the maximum of rank-local plans.
- [x] Use rank-local routing histograms for EP load imbalance.
- [x] Generalize all-reduce/all-to-all communication formulas beyond the fixed
  P=4 two-pair topology.

## 5. Stage-aware working-set model

- [x] Add W13 chunk and W2 phases to the event simulation if exact-policy tables
  do not generalize adequately across F or unseen shapes.
- [x] Derive contention from the sum of active phase working sets rather than
  only the number of active expert tasks.
- [x] Add exhaustive dual-rank uniform/hotspot regret validation and an input for
  real routing histograms.

## Deferred external validation

- Cross-profile interpolation stays disabled. Enable it only after a
  target-model routing dump is supplied and passes the same regret gate; this
  is not part of the exact-profile 1-5 implementation.
