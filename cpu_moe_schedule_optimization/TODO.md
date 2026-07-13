# MoE scheduling implementation checklist

The policy-aware TP2/EP2 scheduling update is implemented as one change set.
Cross-profile interpolation deliberately remains disabled. A captured routing
summary now passes exact-profile validation, but interpolation still requires
out-of-profile measurements across F, parallel degree, and machine topology.

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

- [ ] Cross-profile interpolation stays disabled. A target-model routing dump
  now exists and passes the exact-profile regret gate, but interpolation still
  needs out-of-profile measurements across F, parallel degree, and machine
  topology before it can be enabled.

## 6. Absolute-time accuracy follow-up (2026-07-13)

The exact-profile planner currently has selected-plan error from -2.93% to
+4.40% and maximum measured regret below 2% on TP2/EP2 uniform, hotspot, and
captured-trace cases. Thirty-sample medians have an approximately 1% practical
noise floor. Stage traces show that the existing 2/3 W13 + 1/3 W2 split is
accurate: W13 accounts for 67.3-67.8% of W13+W2 time.

### P0: close the current calibration update

- [x] Commit the 2026-07-13 TP2/EP2 split/no-split profiles, the expanded
  contention route grid, and the validator noise/stage-trace modes as separate
  reviewable changes.
- [x] Update `POLICY_MODEL_VALIDATION.md` to the 2026-07-13 profiles and real
  routing validation. Remove the stale statement that no target routing dump
  is available.
- [x] Remove the untracked legacy F512 schema-v1 profiles from the active
  workspace; retain old calibration results only in git history.

### P0: locate the remaining absolute-time error

- [x] Diagnose the EP2 hotspot. Symmetric dual-rank component replays match the
  profile within 0.4%, while full peer-idle replays are 1.4-7.0% faster. The
  mixed rank measures 63.0 ms against another mixed rank, 59.7 ms against an
  idle rank, and 60.6 ms against the real 32x96 cold rank. The +4.4% hotspot
  error comes from applying the `concurrent_ranks=2` derate for the long rank's
  whole DAG after the short rank has completed, not from `T_iso`, intra-rank
  expert contention, or the 2/3 W13 + 1/3 W2 stage split.
- [ ] Generate matching single-rank companion profiles and jointly simulate all
  rank DAGs. Advance tasks with dual-rank rates while both ranks are active,
  then switch the remaining rank to the single-rank rate at the rank-completion
  event. Do not estimate EP wall time as the maximum of independently predicted
  all-dual-rank makespans.
- [ ] Treat the uniform 96-route full-call residual separately. Route 96 is an
  exact isolated point but not a contention-route anchor, so the current full
  call estimate interpolates between 48 and 192. Add targeted full-call anchors
  only if the residual remains after rank-lifetime contention is modeled.
- [ ] Add held-out heterogeneous full-call cases for the planner-relevant
  shapes `(32)`, `(16,16)`, `(16,8,8)`, and `(8,8,8,8)`. Cover mild skew,
  bimodal hot/cold, one-dominant, hotspot, and captured routing; do not fit and
  validate on the same distributions.
- [ ] If TP heterogeneous residuals remain after fixing `T_iso`, add a
  low-overhead phase timeline and model W13/W13, W13/W2, and W2/W2 overlap from
  the active route/team distribution instead of `max(routes)` alone. Keep the
  measured 2/3 + 1/3 stage split unless new data disproves it.

Acceptance gates:

- selected-plan absolute error <= 3%;
- top-three candidate median absolute error <= 3% and P90 <= 5%;
- measured planner regret <= 2%;
- all comparisons use enough repeats to resolve the approximately 1% noise
  floor.

### P1: complete the end-to-end latency model

- [ ] Keep expert compute and route-combine costs separate. For TP2 top-k=6,
  add the measured approximately 1.1 ms weighted merge plus route-dependent
  gather/scatter; retain communication as a separate topology-bound term.
- [ ] Validate TP2 versus EP2 with an actual distributed/runtime execution.
  The current evaluator combines measured compute profiles with analytical
  communication and is not an end-to-end distributed measurement.
- [ ] Generate exact schema-v2 TP4/EP4 profiles before drawing TP4-versus-EP4
  conclusions; do not extrapolate the TP2/EP2 tables across parallel degree.

### P2: scheduling and kernel follow-ups

- [ ] Measure lane-tail weighted idle loss on a larger routing corpus. Current
  TP cases have a perfect-rebalance upper bound of only 0.7-2.4%; prototype
  same-width-lane work stealing only if representative cases repeatedly exceed
  5%.
- [ ] Run model-level accuracy evaluation for the optional Taylor-poly4 SiLU
  path. Kernel/output tests are complete, but only elementwise and operator
  output error have been measured so far.

The old AUTO selector, bucketized wave-DP, and wave-based Phase 4 items in
`FINDINGS.md` remain historical only because wave scheduling is deprecated.
They are not active TODOs for the async interval-DAG planner.
