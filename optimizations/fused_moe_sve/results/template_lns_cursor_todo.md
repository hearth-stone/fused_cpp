# Cursor TODO: template-LNS ranking cost and proposal sampling

Last reviewed: 2026-09-06. This is the maintained handoff for the next offline
LNS work. Update this file in place after each completed task, including failed
variants and validation gaps. The older dated handoff and experiment reports
remain historical evidence.

## Immediate objective

Tasks 2A and 1A are closed; Task 3 is closed for its declared five-plan,
full-parent audit. Both Task 4 Lab presamplers failed the design adoption gate.
Keep shuffle-truncate N=25, selector v1 / K16, and preserved hardware elites.
Keep `incremental_min_distance_v1`; ranking is no longer the main cost.

Next: Task 6A, prepare one finite hardware-reference pool and its budget. Use
the frozen full-parent critical cross-domain d4 neighborhood (reported unique
pool size 452, to be verified). Serialize the complete pool, join existing
measurements by exact identity, and specify a complete-pool measurement protocol.
This produces a reviewable diagnostic plan; it does not launch new hardware
sessions or silently expand the existing 64+16 hardware budget.

Do not open Task 5 for either rejected presampler. Do not add a third feature
quota solely to rescue the five known hashes. We now need a less selected set
of hardware labels to distinguish sparse good plans from shortlist misses.

## Current checkpoint and evidence

- [x] Beam repair optimization passed the frozen ranking comparison:
  `equal=true`, `mismatch_count=0`. Recorded search wall was 797.01 -> 687.55 s.
- [x] Equal sampling-cap comparison completed: 1 restart with N=50 versus
  2 restarts with N=25; both selected the same `0418b884...` plan.
- [x] Second proposal seed `20261011` completed under 2 restarts / N=25. Its
  new selected-best gains versus full were -0.018% / +0.371%; the injected
  previous winner still gained +3.302% / +3.130% in the same sessions.
- [x] Source and remote artifact review completed. The reviewed analyses and
  two-restart model hashes match the reports. The old winner came from
  `lns_00_r00`, critical strategy,
  cross-domain d4, actual closure 99, predicted gain -2.303%.
- [x] Nested model-stage timers and incremental farthest-first fill implemented.
  Review reran 21 ranking tests and 20 runner/frontier tests successfully.
- [x] Review reconstructed legacy anchor signatures in memory from the two
  original frozen models: 16/16 per-start complete rankings, 8/8 pooled complete
  rankings, and both global top16 lists matched their stored originals.
- [x] Task 2A recovered-input replay matches both original frozen models
  (`equal=true`, `mismatch_count=0`), including seed-`20261011` `lns_01_r01`.
- [x] Task 1A Arm model-only regeneration matched both frozen models
  (`equal=true`). Sidecar elapsed is authoritative; no complete-model
  improvement is claimed.
- [x] Task 3 located `0418b884...` loss on seed `20261011` at sampling after
  unique emission.
- [x] Task 3 extended to the five-plan measured fast-plan set on full parent
  `lns_00_r00` / `lns_00_r01`. Artifacts `fast_plan_loss_set_20261010.json`
  `3fa830fe...` and `fast_plan_loss_set_20261011.json` `4fe4dd55...`; summary
  `fast_plan_loss_set_summary.json` `32bf1d54...`.
- [x] Task 4 specified and design-replayed Lab policy
  `structural_coverage_then_random_v1` (`275dd663...`). Not adopted; Task 5
  stays closed.
- [x] Task 4 width-histogram iteration `structural_coverage_closure_width_v1`
  (`e5a35b06...`) design-replayed and **rejected**. Seed `20261010` lost
  `0418b884...` from the sample (shuffle 19/452, hist `[[4,8],[8,4],[16,1]]`);
  no sampling-lost tracked hash was recovered. Artifact
  `task4_feature_replay.json` `f2d095cf...`.
- [x] Post-Task-4 review verified both original/fresh Arm model comparisons
  (`equal=true`, zero mismatches), full model and sidecar hashes, and the five
  plans' scored/pooled membership in the original frozen models. Current tests:
  neighborhood + presampler 26 passed; ranking + runner/frontier 45 passed.
- [ ] Task 6A finite-pool manifest, evidence join, and budget preparation.
- [ ] Task 6B complete-pool hardware diagnostic, only after its separate budget
  is explicitly authorized. Task 5 remains closed.

Source checkpoint: `db95b7e` contains beam optimization and restart comparison.
At review time, the second-seed work, timing instrumentation, incremental ranking,
replay helper, reference-plan support, tests, and documentation were uncommitted.
Inspect
`git status --short` and preserve them before editing; do not assume a clean
checkout or overwrite that work.

Read these result reports:

- [Search breakdown and beam equivalence](arm_codex_80c_lns_search_breakdown_beam_equiv_20260905.md).
- [Restart budget comparison](arm_codex_80c_lns_restart_budget_20260905.md).
- [Second proposal seed](arm_codex_80c_lns_second_proposal_seed_20260905.md).
- [Earlier independent shortlist audit](arm_codex_80c_lns_diverse_independent_median_20260905.md).

Interpretation limits to preserve:

1. Both restart arms selected the same plan. Its small relative timing changes
   across different hardware sessions do not establish a causal search-quality
   advantage for two restarts. Keep 2/N25 as the working allocation, not a
   demonstrated optimum.
2. `shortlist_s` includes partial-order diagnostics, feature construction, and
   per-start diverse ranking. It cannot all be attributed to partial order.
   Nested fields now split those shares; the historical `shortlist_s` meaning
   is unchanged.
3. `search_wall_s` sums the per-start runs. Parent-pooled ranking and subsequent
   frontier preparation occur outside those timers; initial control construction
   is separate too. The layer bench now records those stages plus serialization
   and elapsed time through the main model artifact write. The post-write values
   are in `{output}.wall.json`; the model JSON contains earlier timing values.
   Task 1A has now run both frozen 2048-token model paths on Arm. Verified
   post-write elapsed times are 622.820 / 618.783 s; timers begin in `main`
   after imports. Historical `search_wall_s` is not a complete-model baseline,
   so no complete-model speedup ratio is established.
4. The earlier K16/K32 gate applies to its measured frontier. Later 64+16
   experiments measured K16 plus an outside-K32 sample, not all K32 plans.
   Neither gate certifies recall over every enumerated or evaluated candidate.
5. The two seeds have identical full-parent canonical state, critical expert
   IDs, and frozen enumeration configuration. Together with deterministic
   critical enumeration, this identified the presampling question. Task 3 now
   records unique emission and shuffle indices beyond N=25 for the missed
   seed. The audit is scoped to the generating full parent; absence from other
   parents' scored features still does not prove non-reachability there.
6. Legacy `candidate_features` omitted `anchor_domain_assignment_signature`.
   Comparing two implementations after both load that field as an empty string
   proves equal-input fill equivalence, not necessarily equality to the stored
   original ranking. The omitted value is recoverable from the saved anchor.
7. The original reference/incremental RSS values came from successive runs in one
   process. `ru_maxrss` is a cumulative high-water mark; the incremental value
   included the earlier reference run. Task 2A added separate child-process RSS
   runs; the current `--rss-only-fill` path covers per-start ranking, not pooled
   ranking or the full model. Keep that scope explicit.
8. The five tracked plans include single-session >=2% observations as well as
   preserved elites. They are a diagnostic set, not five independently proven
   stable winners or an unbiased estimate of fast-plan density. Classify each
   plan from actual session evidence; do not infer stability from its label.
9. Losing `2ab43572...` at pooled indices 45/291 is a budget-retention miss,
   not dominance pruning. Final membership comes from pooled/global selected
   keys and the actual hardware frontier; per-start top16 is intermediate.

## Post-Task-4 analysis and next decision

The old timing and replay gaps have closed at their stated scope. The Arm
model-only path used the frozen median L4 shape and NUMA3 protocol below, with
two restarts per parent, N=25, seeds 20261010/20261011, and no hardware sessions.
Commands were the model stage of `bench_template_lns_layer.py --initial-controls
... --restarts-per-parent 2 --seed <seed>` under
`numactl --physcpubind=240-319 --membind=3`; exact artifacts are registered below.

| Recorded slice | Seed 20261010 | Seed 20261011 |
| --- | ---: | ---: |
| Main through model artifact write | 622.820 s | 618.783 s |
| Control construction | 136.110 s | 136.184 s |
| Exact event scoring | 349.051 s | 348.924 s |
| Enumeration/sample | 117.267 s | 113.320 s |
| Per-start shortlist, all work | 2.255 s | 2.245 s |
| Per-start ranking, nested in shortlist | 0.417 s | 0.418 s |
| Parent-pooled ranking | 0.804 s | 0.814 s |
| Unaccounted main-stage time | 1.116 s | 1.116 s |

The table is a selected stage breakdown, not a set of rows to sum: the ranking
row is nested. Both fresh model comparisons have zero mismatches. Further
ranking optimization has little remaining impact; the complete-model baseline
needed for a speedup ratio was not measured.

The audit identifies two different budget losses. `0418b884...` is reachable
but missed by N=25 sampling on seed 20261011. `2ab43572...` reaches exact scoring
on both seeds but is outside pooled top32 and the outside-audit sample. Increasing
only sample coverage cannot by itself fix that second loss. Preserve old elites
independently of each new shortlist so either loss cannot discard the incumbent.

The two presampler decisions are reasonable adoption decisions on this design
set. Closure bins already cover the missed plans' categories, so another first
representative adds no useful tracked state. Adding width histograms reserves
more category representatives within the same N, displacing a later plan in an
already represented category, including the successful seed's `0418b884...`.
More represented categories do not establish more fast plans. These negative
results do not prove that all structural features are useless or that no
unmeasured candidate from the rejected variants is fast.

For scale, a specific member of a fixed 452-plan pool has inclusion probability
`1 - (1 - 25/452)^2 = 10.756%` under two independent uniform N=25 draws. Taking
50 distinct plans raises that to only `50/452 = 11.062%`. This calculation is
for one exact hash under the stated assumptions, not the probability of finding
any good plan. That latter probability depends on the unknown number of good
plans and their joint retention through the downstream shortlist.

Task 6 therefore prepares a complete, narrowly bounded hardware reference pool
before another sampler design. It can reveal fast-plan density, within-category
variation, and sampled-versus-shortlisted regret. The slice is selected because
it contains known counterexamples, so it is a design diagnostic, not an
independent holdout or a global-optimality certificate.

Evidence hygiene: Task 1A model/compare/sidecar files were independently read and
SHA-verified during this review. Task 3/4 conclusions above retain the submitted
audit records and were checked against code and frozen-model membership. Their
named raw JSON files were not located in the checked local/remote temporary
paths; Task 6A must register their exact locations/full hashes or label any
re-created audit as a new artifact. Do not claim their short digest prefixes
are a new independent raw-file verification.

## Review of the ranking experiment

The implementation log reports Darwin replay over eight per-start pools and
four parent-pooled pools, two repeats per implementation:

| Proposal seed | Reported reference mean | Reported incremental mean | Scope |
| --- | ---: | ---: | --- |
| 20261010 | 139.395 s | 0.515 s | Saved-feature ranking only |
| 20261011 | 141.957 s | 0.521 s | Saved-feature ranking only |

These historical figures support a ranking optimization, not a kernel speedup or
an Arm complete-model speedup. The original replay timing JSON paths/hashes were
not included in the completion log and were not located in the repository or
`/private/tmp` during review. Preserve these as reported measurements until the
exact outputs and commands are registered; do not invent replacement identities.
The later Task 2A completion record reports recovered-input replay and separate
RSS runs; Task 1A's Arm model-only outputs are independently verified below.

Independent review read the original model files from Arm, verified SHA256, and
used the current incremental selector in local memory. For each one-iteration
run, it derived the anchor signature with
`domain_assignment_signature(domain_assignment_payload_from_mapping(run["initial_canonical_state"]))`,
then supplied that value to the corresponding candidate features.

| Seed | Per-start ranks with missing-field default | Per-start ranks with recovered anchor | Recovered pooled ranks | Recovered global top16 |
| --- | --- | --- | --- | --- |
| 20261010 | 8/8 match original | 8/8 match original | 4/4 match original | exact |
| 20261011 | 7/8 match original | 8/8 match original | 4/4 match original | exact |

The mismatching legacy start is `lns_01_r01` for seed 20261011. Its original
ranking is recoverable; replacing the comparison baseline with another replay
is unnecessary. On these two pools, recovered and historical parent-pooled
rankings also match. This does not prove that changing missing-field handling
is neutral on every future pool: the anchor signature affects the initial
coverage tie-break through `distance_from_anchor`.

That earlier review established ranking identity for the listed pools. Tasks 2A
and 1A subsequently closed artifact/PlanV2 reconstruction and Arm model-stage
timing. The present review read their outputs and did not regenerate candidates
or collect new hardware measurements.

## Frozen experiment boundary

Keep these fixed through the ranking work and as the baseline for sampling:

| Item | Value |
| --- | --- |
| Purpose | Offline hardware-assisted reference search |
| Trace | `measured_request016_case017_zh2048-018.pt`, layer 4 |
| Shape | 2048 tokens, TopK 6, 256 experts, H4096/F512, bf16, 80 threads |
| Parents | full, one-step, greedy, fixed-width; canonical dedup |
| Operators | local/cross-domain x d4/d8/d16; beams 16/32/64; 4 templates/block |
| Allocation | 2 restarts per parent, critical + random strategies, N=25/operator |
| Candidate sampling cap | 4 parents x 2 restarts x 2 strategies x 6 operators x 25 = 2400 |
| Hardware selector | `relation_agnostic_categorical_farthest_first_v1` |
| K | 16 per unique parent after pooling restarts; audit prefix 32 |
| Hardware budget | 64 selected + 16 outside-audit samples, plus preserved controls |
| Controls | Four original anchors, `0418b884...`, and `2ab43572...` |
| Hardware protocol | Arm-codex-internal NUMA3 CPUs 240-319, membind 3; one process and one weight-allocation batch per session; 4-copy rotation; 5 warmups; 31 randomized paired rounds; two independent sessions |

The 2400 figure is a sampling cap, not identical actual event-call counts.
Report sampled records, cross-start unique states, cache hits, candidate exact
calls, and additional anchor/context calls separately. Preserve the existing
benchmark cache/page protocol and capture its environment explicitly.

Do not change frozen v8, residual calibration, selector categories/quantiles/
tie-breaks, K, operator mixture, PlanV2, kernels, or runtime defaults during
Tasks 1-3. LNS automatic acceptance and dominance pruning remain disabled.
Keep original controls and hardware elites outside the new-candidate budget;
an unsuccessful search must not replace the incumbent with a slower shortlist
winner. No ALNS weights or approximate event replay are part of this handoff.

## Task 1: close the timing accounting

Status: closed for instrumentation and the two frozen Arm model regenerations.
A complete-model before/after speedup comparison is not claimed.

- [x] Split the timer in `_run_partial_order_vnd_start` into partial-order
  evidence/selection, feature construction, quantile/dedup work, and per-start
  diverse ranking. Avoid overlapping totals; label nested timers explicitly.
- [x] In `bench_template_lns_layer.py`, separately measure restart pooling,
  parent-pooled diverse ranking, outside-audit sampling, canonical/PlanV2
  frontier construction, global merge, and serialization.
- [x] Add recording for control construction and model-stage elapsed time.
  Reconcile the component sum to the total and report unaccounted overhead.
  Use a wrapper or sidecar for elapsed time through the final artifact write.
  Preserve the meaning of existing timing fields; add explicit new fields.
- [x] Implement/profile frozen feature replay, including both per-start and
  parent-pooled rankings. Separate profile replay from full model generation.
- [x] Clarify attribution in the existing reports/TODO/manifest with an added
  clarification; preserve their historical raw numbers and artifact identities.

### Task 1A: target closeout

- [x] After Task 2A, run both frozen seeds on Arm-codex-internal using the same
  parents, route, calibration, extension, stage geometry, budgets, and selector.
  Run only model generation/replay; the shell experiment runners also contain
  hardware stages and should not be launched wholesale for this check.
- [x] Validate the original ranked prefixes, full pooled ranks, outside sample,
  plan hashes, and bridges through the actual post-search assembly path.
- [x] Record all stages, the post-write sidecar, and process elapsed time from a
  wrapper if making a complete-command claim. The current timer begins in
  `main`, after imports; distinguish it from full process lifetime.
- [x] Reconcile exclusive stage totals to elapsed time and label nested timers.
  Treat the sidecar as authoritative for time through the model artifact write;
  do not mix it with the earlier model-JSON timing snapshot.
- [x] Record source/file hashes, output artifact hashes, exact commands, and
  repeatability. Measure an instrumented reference on the same target if
  reporting complete-model before/after improvement; historical `search_wall_s`
  alone is not that baseline.

Done when: a reader can explain the complete model-stage wall time and identify
the ranking share without assuming the small global-merge timer covers ranking.
Do not launch a hardware timing session for this task.

## Task 2: make farthest-first ranking equivalent and cheaper

Status: ranking core, recovered-input replay, and target integration closed on
the two frozen seeds. Keep the incremental implementation.

Code entry: `rank_lns_diverse_candidates` in
[`lns_diverse_shortlist.py`](../benchmarks/lns_diverse_shortlist.py).

The old fill recomputed every remaining candidate's distance and token reuse
against the entire selected set, taking O(n^3) work for a full ranking. The
current implementation updates those values incrementally in O(n^2).

- [x] Cache immutable categorical tuples once per candidate.
- [x] Maintain each remaining candidate's minimum distance to the selected set.
  After taking a candidate, update using only its distance to that new member.
- [x] Maintain the exact category-match reuse sum incrementally, or equivalent
  per-dimension category counts. Preserve the definition of `token_reuse`.
- [x] Preserve operator coverage, quantile coverage, canonical duplicate
  provenance, all tie-breaks, and full ranking order. Target O(n^2) work and
  O(n) auxiliary state; no need to materialize an n-by-n distance table.
- [x] Keep the current algorithm as a test reference. Cover duplicate features,
  ties, missing categories, shuffled input, and pooled restarts.
- [x] Verify both original seeds' per-start/pooled full ranks and global top16
  with recovered anchor inputs in the in-memory review above.
- [x] Close the durable comparison for top16/top32, global dedup/backfill,
  outside-audit keys, all decision features, and executable plans in Task 2A.
- [x] Extend `compare_lns_frozen_ranking.py` for pooled rankings and outside
  sample fields. Replay now recomputes outside samples and validates available
  plans; `RANKING_FEATURE_FIELDS` includes the recovered anchor signature.
- [x] Report local saved-feature replay timing before/after and separate
  per-start RSS runs. Register exact replay output locations in Task 6A.
- [x] Measure the target model path and complete wall time in Task 1A.

### Task 2A: make legacy replay a real frozen-artifact gate

- [x] Add a legacy feature recovery path bound to `anchor_state_hash`. For the
  current single-iteration runs, verify the saved initial state's canonical
  hash equals each row's anchor hash, then recover its domain signature. For
  other runs, resolve the actual iteration anchor; never assume it is initial.
- [x] Preserve original input bytes/hashes. Mark recovered fields and their
  source in replay provenance. Missing or mismatched anchors must fail clearly.
- [x] Compare the original frozen artifacts directly to recovered-input replay
  for both seeds; keep reference-versus-incremental comparison as a separate
  implementation check. Include the recovered anchor signature in comparator
  validation instead of silently ignoring this decision-relevant field.
- [x] Keep historical and recovered parent-pooled input semantics explicit.
  Both are identical on the two reviewed outputs; if another corpus changes,
  record an input/serialization correction separately from an equivalent fill
  optimization and require the corresponding new retention validation.
- [x] In `replayed_model_payload`, recompute global outside-audit keys and
  parent ownership rather than copying baseline `stratified_keys`.
- [x] Rebuild/validate selected and audit plan rows from an exact state library.
  Remove silent `if key in by_hash` truncation: an unavailable selected state
  must fail. Verify the needed pooled-row set instead of copying old rows and
  comparing those copies. Require canonical hash and PlanV2 round trips.
- [x] Make the replay gate require repeat agreement as well as reference/new
  equality and equality to the original per-start ranked keys.
- [x] Recover and register timing replay JSON files with full hashes and exact
  commands. Run fresh child processes for each fill when comparing peak RSS.
  No new hardware timing is needed for this step.

Closed: original-seed replay, recomputed outside selection, and executable
plan checks all pass with zero unexplained mismatches; metadata records recovered
fields and source identity. Tasks 1A and 3 have also completed. A whole
JSON file hash may change with timing/recovery metadata; compare normalized
decision fields without silently changing their meaning.
Keep `incremental_min_distance_v1` as the validated ranking implementation.
Do not claim a complete-model wall improvement or a memory improvement from
the current local replay evidence.

## Task 3: audit where known fast plans are lost

Status: closed for the declared five-plan set and full-parent starts. Non-full
parents were checked for scored membership only.

Code entries: `_sample_neighborhood`, `sample_template_lns_neighborhood`, and
`enumerate_template_lns_neighbors` in
[`executable_plan_neighborhood.py`](../../../cpu_moe_schedule_optimization/planners/executable_plan_neighborhood.py).

- [x] First audit `0418b884...` against full-parent critical enumeration for
  seeds 20261010 and 20261011. Reuse exact saved inputs and execution geometry.
- [x] Track full hash/provenance through: emitted -> deduplicated -> sampled ->
  event-scored -> parent top32 -> hardware top16. Record explicit loss reasons.
- [x] Capture anchor, restart, strategy, operator, target destroy, actual
  closure, width template, domain placement, and order-policy/sequence signature.
  Preserve multiple provenance records when one canonical state appears twice.
- [x] Extend the audit to the existing measured fast-plan set, not only this
  winner. Distinguish a state outside a parent's reachable neighborhood from
  one dropped by that parent's sampling budget.
- [x] Record compact hashes/features for the presampling pool; full executable
  states are needed only for tracked targets and replay-selected candidates.

Done when: every tracked state has an evidenced loss stage. If the target is
absent before sampling, inspect closure/template/repair coverage before changing
the sampler. This task needs enumeration replay, not new hardware measurement.

Measured fast-plan set (any frozen-protocol median session with absolute median
gain vs reconstructed full `>=2%`, plus the preserved elite):

All shuffle/ranking positions in this table are zero-based array indices, not
one-based ranks. Inclusion requires index < N or index < K respectively.

| Hash | Hardware role | 20261010 2-restart N=25 | 20261011 2-restart N=25 |
| --- | --- | --- | --- |
| `0418b884...` | selected-best / injected control | not lost: `lns_00_r00` critical shuffle 19/452, pooled rank 9, selected | sampling: emitted unique on full-parent critical, shuffle 373/452 and 444/452 |
| `2ab43572...` | known elite | event-scored, pooled rank 45, outside parent top32 | event-scored on `lns_00_r01` critical shuffle 16/452, pooled rank 291 |
| `7cac2afd...` | 1-restart selected / independent consensus | sampling: unique on `lns_00_r00` critical d8, shuffle 40/716; N=50 would keep it | sampling: unique, shuffle 189/716 and 604/716 |
| `1faf090a...` | independent-median session-1 `>=2%`, audit-only | sampling: unique on `lns_00_r00` random d16, shuffle 26/1016; N=50 would keep it | sampling: unique, shuffle 455/1000 and 651/1000 |
| `07355dde...` | independent-median session-1 `>=2%`, audit-only | event-scored on `lns_00_r00` critical d4, shuffle 7/355, pooled rank 123 | sampling: unique, shuffle 324/355 and 121/355 |

All five are in the full-parent reachable neighborhood on both seeds. None appear
in other parents' `candidate_features`. Non-full parents were not enumerated;
that residual does not change the generating-parent loss stages. Random strategy
never uniquely owns `0418b884...`, `2ab43572...`, `7cac2afd...`, or `07355dde...`.

## Task 4: compare structural presampling at fixed N

This is a proposal-retention change, separate from the equivalent ranking work.
Task 3 identified the loss stages. Both variants below are rejected for adoption
on this design set and remain separate from the active sampler.

- [x] Specify one versioned alternative to per-operator shuffle-and-truncate:
  structural-coverage slots plus random-exploration slots within the same N=25.
- [x] Use only cheap plan-visible features available before exact event scoring:
  actual closure, width template, domain placement, and temporal order. Do not
  require an unavailable full-event score/quantile at this stage.
- [x] Lock quotas, bins, duplicate ownership, RNG mapping, and tie-breaks before
  the deciding experiment. Keep the current sampler as the baseline.
- [x] Use the two existing seeds and measured cases as design replay. Report
  tracked fast-plan retention and structural coverage separately from hardware
  recall; unmeasured candidates have no known hardware rank.
- [x] Do not encode measured timings, previous-winner status, or specific state
  hashes into selection. Known winners are audit labels and preserved controls.
- [x] Keep the downstream selector v1 unchanged. Record a separate presampler
  policy/version/hash; changing the proposal pool can legitimately change
  quantiles and rankings even though the downstream selector rule is frozen.

Locked Lab candidate, not wired into `sample_template_lns_neighborhood`:

| Item | Value |
| --- | --- |
| Name | `structural_coverage_then_random_v1` |
| SHA256 | `275dd6636cbd00008fc2d78d9cd4b9dd6cf0e2adf4f5fa147f01fe216ba41c24` |
| Feature | `actual_closure_size` only, via existing `CLOSURE_BINS` |
| RNG | one `random.Random(seed)` shuffle per operator, same mapping as baseline |
| Coverage | first representative of each nonempty bin in that shuffle |
| Remainder | the same shuffle, skipping coverage picks, truncated at N=25 |
| Duplicate ownership | unchanged global first-hash-wins |

Design replay on full-parent starts of seeds `20261010` / `20261011`:

- Tracked-hash retention versus baseline shuffle-truncate is unchanged: no
  sampling-lost tracked hash is recovered, and no already-sampled tracked hash
  is dropped.
- Some operators gain a missing rare bin (`32_63` on cross-domain d16; `1_15`
  or `128_plus` on a few local/cross d4/d8 operators). No operator loses a
  unique-pool bin that baseline already covered.
- The sampling-lost tracked plans are later members of already-covered closure
  bins (for example `7cac2afd...` at shuffle 40/716 in cross-domain d8). Bin
  coverage cannot recover them.

Do not open Task 5 on this candidate. Keep shuffle-truncate N=25 and preserved
elites. Prepare the Task 6 reference pool before considering another feature
quota; feature coverage by itself is not evidence of search quality.

Done when: a concrete candidate presampler has a reproducible design-replay
report and a frozen, timing-free policy ready for an independent frontier.
This candidate is frozen as a Lab design-replay artifact, not ready for an
independent hardware frontier.

### Task 4 iteration: closure bin plus width histogram

Locked before the deciding replay. Not wired into
`sample_template_lns_neighborhood`. Known fast-plan hashes are audit labels
only.

| Item | Value |
| --- | --- |
| Name | `structural_coverage_closure_width_v1` |
| SHA256 | `e5a35b06f0f198c9940fa3bfc9c4972299347ce85ceefaf390d883da4075ee76` |
| Features | `actual_closure_size` via `CLOSURE_BINS`, plus `candidate_width_histogram` from `width_histogram(state)` |
| Coverage key | `(closure_bin, width_histogram)` |
| RNG | one `random.Random(seed)` shuffle per operator, same mapping as baseline and v1 |
| Coverage | first representative of each nonempty key in that shuffle |
| Remainder | the same shuffle, skipping coverage picks, truncated at N=25 |
| Duplicate ownership | unchanged global first-hash-wins |
| Forbidden | event scores, quantiles, hardware timings, state-hash allowlists, previous winners |

- [x] Design-replay both frozen seeds on `lns_00_r00`/`lns_00_r01` versus shuffle-truncate N=25 and v1.
- [x] Record adopt/reject/inconclusive. Do not open Task 5 unless tracked-hash sampling retention improves.

Decision: **reject**. Command:
`audit_lns_fast_plan_loss.py --model <frozen> --proposal-seed 20261010|20261011 --start lns_00_r00 --start lns_00_r01 --feature-replay-output ...`.
Models `9b334b78...` / `cc53d43d...`. Replay `task4_feature_replay.json` `f2d095cf...`; per-seed `be793c46...` / `fa2a6846...`. Baseline prefix-stability true. v2 vs baseline: lost `0418b884...` on `20261010`; gained none of `7cac2afd...` / `1faf090a...` / seed-`20261011` `0418b884...` / `07355dde...`. `(closure_bin, width_histogram)` key counts rose on all 24 start/strategy/operator cells, which is why the already-sampled winner was displaced. Not wired into `sample_template_lns_neighborhood`. Task 5 stays closed. Keep N=25 shuffle-truncate and injected elites.

## Task 5: independent hardware validation and decision

Status: closed to both rejected Task 4 variants. Task 6 is a separate reference
data diagnostic; it does not adopt a presampler or replace this holdout gate.

- [ ] Predeclare new proposal, sampling, and hardware seeds plus all adoption
  gates. Do not tune using the independent results after opening the frontier.
- [ ] Compare baseline and candidate under the same sampling cap and declared
  hardware allocation; count control and audit slots separately.
- [ ] Preserve the two historical elites and all original anchors in each
  session. Compare the same selected plan across both sessions, not different
  session-best hashes as though they were one repeatable winner.
- [ ] Require bit-exact output checks before timing. Report absolute medians,
  paired median gains, P10/repeatability, consensus winner, and regret versus
  both strongest original anchor and best preserved elite.
- [ ] Report full search/model/hardware wall time, actual calls, and plan counts.
- [ ] Retain outside-shortlist auditing. The 64+16 budget does not measure the
  entire top32 union; a new K16/K32 recall claim needs a predeclared complete
  K32 audit and its explicitly reported additional budget.
- [ ] Record adopt/reject/inconclusive for the presampler, then update this file,
  the mathematical model, planner TODO, manifest, and a concise result report.

A passing old-winner retention check is necessary diagnostic evidence, not proof
of a better optimizer. An inconclusive hardware result retains the baseline and
elites; it does not justify changing K or refitting calibration after the fact.

## Task 6A: prepare a finite reference pool and explicit measurement budget

This is the immediate next task. Preparation does not include launching new
hardware sessions. Existing N=25, K16, and 64+16 search budgets remain unchanged.

- [ ] Recover/register the Task 2A/3/4 raw outputs named in the completion log,
  with exact local/remote paths, full SHA256, and executable commands. Preserve
  original bytes; if recovery fails, record the gap and create any replacement
  audit under a new identity rather than reusing its historical digest.
- [ ] Freeze one pool from full parent `98a32da5...`, the original critical IDs,
  operator `critical_window_template_repartition_cross_domain_d4_b16`, and the
  same calibrated stage-window policy. Use shipped enumeration and canonical
  global duplicate ownership before filtering to this operator. Do not choose
  candidates by score, relation, known-winner status, or the current shortlist.
- [ ] Verify the reported 452 unique members and compare pool hash sets across
  the two saved full-parent critical inputs. Store the actual count/digest;
  do not force the count or insert a missing winner to make the check pass.
- [ ] Serialize every pool member's full executable state and PlanV2 bridge.
  Require valid geometry, exact canonical hash/bridge round trips, and a sorted
  pool hash list. Record all operator/closure/width/domain/order provenance.
- [ ] Add all four original controls and both historical elites as reference
  roles, with canonical deduplication and separate role accounting. Verify
  whether `0418b884...` and `2ab43572...` already belong to the pool.
- [ ] Join existing median frontier/session artifacts to this pool by full
  state hash and protocol identity. Report measured, repeated, and unmeasured
  counts. Keep same-session times/gains distinct; selected historical samples
  cannot estimate the pool's fast-plan density or supply one common oracle.
- [ ] Correct audit aggregation to use pooled/global final membership when
  pooling is enabled. Per-start top16 is an intermediate state; an injected
  elite is a control, not a successful new proposal. Derive stable versus
  single-session-positive labels from measurements, not descriptive strings.
- [ ] Produce the proposed hardware manifest, exact runner command, number of
  plans/comparisons, memory estimate, and expected wall time for two sessions.
  Report timing/correctness/setup costs separately and label estimates.
  The initial ceiling is 452 pool members + 4 controls + 2 elite roles before
  dedup (458); the verified manifest determines the actual total.
- [ ] Explicitly state the total diagnostic budget. Measuring this pool exceeds
  the existing 80 LNS-candidate hardware slots and needs separate authorization;
  it is not a change to the default search budget. Keep original inputs, v8,
  selector v1, operator definitions, and production runtime unchanged.

Done when: the complete candidate manifest, evidence join, scope, protocol, and
cost estimate are concrete and reviewable. Retain the rejected policy records;
avoid adding another sampler while preparing this dataset.

## Task 6B: complete-pool hardware diagnostic, conditional on its budget

Do not start this task merely because Task 6A completes. Run only when the user
has authorized the explicit additional diagnostic measurement budget.

- [ ] Measure every unique frozen pool member in both sessions under the
  existing NUMA3, same-process/same-allocation, 4-copy, randomized 31-paired-round
  protocol, with bit-exact output checks. No unmeasured member may be treated
  as bad. Preserve identity and record the full actual benchmark/cache protocol.
- [ ] Report same-session and consensus pool best, all preserved controls,
  absolute times, paired gains/P10, and the count/distribution of plans meeting
  the predeclared stability criterion. Keep exact-best recall and near-best
  regret separate; predeclare any indifference band before timing results.
- [ ] Compute sampler inclusion recall and conditional shortlist recall
  separately. Use measured pool-best regret and probability of retaining any
  stable fast plan, not only recovery of the five historical hashes.
- [ ] Replay baseline random sampling, the two rejected coverage rules, and
  selector v1 on this fixed pool at predeclared budgets. Report candidate and
  hardware costs; do not claim this single-operator replay validates the full
  multi-operator/four-parent search pipeline.
- [ ] Compare hardware-time spread within the same closure/width category and
  across categories. Use measured density/recall to decide whether more sampling,
  a different hardware shortlist, or new plan-visible features merit a separate
  experiment. Preserve results where the model/feature ordering is uninformative.
- [ ] Record the decision and next experiment. This known-counterexample pool
  is design evidence only; any newly tuned policy needs another independent
  frontier before adoption. There is no global-optimality claim.

## Validation and repository workflow

Follow the repository AGENTS instructions and parent skills before code work:
impact-analysis, test-selector, safe-refactor for Task 2, and code-review-gate.
Read `MATHEMATICAL_MODEL.md` before planner work and synchronize it when retention
or model semantics change. Timing/ranking work is Lab/equivalent-refactor work;
Task 4 also changes planner semantics. Apply the corresponding validation gates.

Start with the smallest relevant tests, then add integration coverage only for
the files changed:

```bash
PYTHONPATH=.:src .venv/bin/python -m pytest -q tests/test_moe_lns_diverse_shortlist.py
PYTHONPATH=.:src .venv/bin/python -m pytest -q tests/test_moe_executable_plan_neighborhood.py tests/test_moe_lns_structural_presample.py
PYTHONPATH=.:src .venv/bin/python -m pytest -q tests/test_moe_partial_order_vnd_runner.py tests/test_moe_partial_order_hardware_frontier.py
```

Use `compare_lns_frozen_ranking.py --baseline <original-frozen-model> --candidate
<recovered-input-replay-model> --output <comparison-json>` with the completed
Task 2A legacy handling. Do not substitute a regenerated reference replay for the
original baseline to make its missing-field mismatch disappear.
Run target integration before any new hardware measurement. Read
`docs/agent_remote_execution.md` and `docs/agent_benchmark_hygiene.md` before
remote work. Preserve original artifacts and use fresh output directories.
Do not add raw benchmark artifacts to source commits. Commit only when requested.

## Frozen artifact lookup

Raw results live on `Arm-codex-internal`, relative to
`/home/zhangxu/codex/fused_cpp`; these directories were absent locally at review.
Recover exact files if needed, not newly generated substitutes with old labels.
The table combines report identities with the remote checks described above.

| Remote-relative file | SHA256 |
| --- | --- |
| `tmp/moe_lns_beam_equiv_verify_20260905/ranking_compare.json` | `2479634ffcdfa0d8be10140b42a622441feebdc4635a16f1b278c46454af05a8` |
| `tmp/moe_lns_restart_budget_20260905/one_restart_model.json` | `4be4947d58ad0be764069d8f3909659a021ebd441417a2d3cd2431425dfa1078` |
| `tmp/moe_lns_restart_budget_20260905/two_restart_model.json` | `9b334b784e80c8376aa059084875a75d4f323c2a4c8f4192e60288f4590595b8` |
| `tmp/moe_lns_restart_budget_20260905/restart_budget_analysis.json` | `027ea3302286290a874f910d2498862037e8530a2baf8b4f911f0068e598a16f` |
| `tmp/moe_lns_second_seed_20260905/second_seed_model.json` | `cc53d43d5827a353ba4325c8a787e0302fcff20a0ebc3840752acfa824d3238a` |
| `tmp/moe_lns_second_seed_20260905/second_seed_frontier.json` | `fa4c83b0843f58283e42ea0a957d34684d2e56a29c8471ef02de2550ae365895` |
| `tmp/moe_lns_second_seed_20260905/second_seed_analysis.json` | `71a23bc5baf0db71671bfff1fc5005fcd02ea79b7d38e160ce36b69b5f6bafee` |

Task 1A outputs independently verified in the post-Task-4 review, on the same
remote host/root:

| Remote-relative file | SHA256 |
| --- | --- |
| `tmp/moe_lns_task1a_20260905/seed20261010_model.json` | `56efb848dc9eaf3ffae4b58d2724088f2b5684cfbc0828ec376fa74bf29b1983` |
| `tmp/moe_lns_task1a_20260905/seed20261011_model.json` | `d6dae89cc0ca2f45fbab6e2760f41951821738b94194fd339cdca3a2ad0e32c2` |
| `tmp/moe_lns_task1a_20260905/compare_20261010.json` | `95b136cca706cb13d0f428b9383bb8d52a09c1b1cde3a61c1c7122b0b989dd38` |
| `tmp/moe_lns_task1a_20260905/compare_20261011.json` | `087fe19f92360fea0f0f6001c7f3d4a6c3a860cfed4f55c1cfca3613ec16a70c` |
| `tmp/moe_lns_task1a_20260905/seed20261010_model.json.wall.json` | `bd38a72eee56f538e9b04fd16a64789bdd988f46d4881364b552cfdab37e0206` |
| `tmp/moe_lns_task1a_20260905/seed20261011_model.json.wall.json` | `a3beb05c30f85757a6aad286a09c2a409f769dff79cfdcb89bd446ad832b9860` |

Full calibration, extension, route, and downstream policy identities are in the
linked second-seed report and model artifacts; verify all before replay. The
previous selected plan is `previous_seed_selected.json` and the older elite is
`known_median_elite.json` in the second-seed directory.

| Preserved control | Full canonical state hash |
| --- | --- |
| Previous selected | `0418b88445c2f88488ca10a1eeae3aeb7b6080e806e241eacfed17ec10904bf6` |
| Older median elite | `2ab43572d14e11060e4fc07b3aa7e0c1780553233dfda61abe74f08abc51e973` |

## Completion log

| Date | Task | Status | Command/artifact and result | Remaining gap |
| --- | --- | --- | --- | --- |
| 2026-09-05 | Handoff | Ready | Source/artifact review; next tasks above are pending | No new implementation or benchmark in this handoff |
| 2026-09-05 | 1 | Implemented; Task 1A open | Nested timers and `{output}.wall.json`; original implementation run reported 20 focused tests passed. Existing aggregate timing meanings unchanged. | Complete Arm model path and timing reconciliation not measured. Earlier full-closure status is superseded by this review. |
| 2026-09-05 | 2 | Core verified; Task 2A open | Incremental fill and 21 focused tests. Original Darwin log: seed 20261010, 139.395 s / 234240 KB -> 0.515 s / 234624 KB; seed 20261011, 141.957 s / 329664 KB -> 0.521 s / 329664 KB, two repeats. Preserve these as reported values. | Timing JSON provenance missing; RSS cumulative across implementations. Original seed-11 comparison was replay vs replay, not frozen vs replay. |
| 2026-09-05 | Review | Verified | Current-tree tests: 21 passed in 0.12 s and 20 passed in 0.18 s with the two commands above. Read-only recovery from SHA-verified original models restored all 16 per-start and 8 pooled full rankings plus both global top16 lists. | Next: Task 2A -> Task 1A -> Task 3. No new hardware result or complete-model speedup was measured. |
| 2026-09-06 | 2A | Done | Recovered-anchor replay vs original frozen models. Commands: `compare_lns_frozen_ranking.py --baseline <frozen> --candidate <recovered_replay>` for seeds 20261010 (`9b334b78...`) and 20261011 (`cc53d43d...`); both `equal=true` `mismatch_count=0`. Recovered replays `62c92bf6...` / `8b5bced3...`. Ranking tests: `pytest -q tests/test_moe_lns_diverse_shortlist.py` → 25 passed. Replay JSON `ranking_replay_20261010.json` `71db5c5f...`, `ranking_replay_20261011.json` `6a43c046...`; both `equal=true`, repeat agreement true, matches original ranked keys. Child-process RSS 20261010 ref/inc 236064/235600 KB; 20261011 235472/235200 KB. Darwin recovered-anchor ranking-only means: 142.666/0.514 s and 143.048/0.527 s. | Next was Task 1A. |
| 2026-09-06 | 1A | Done | Arm-codex-internal model-only (no hardware sessions). `numactl --physcpubind=240-319 --membind=3 .venv/bin/python bench_template_lns_layer.py --initial-controls ... --restarts-per-parent 2 --seed 20261010/20261011`. Outputs `56efb848...` / `d6dae89c...`; wall sidecars `bd38a72e...` / `a3beb05c...`. Frozen compare both `equal=true`. Sidecar elapsed 622.820 s / 618.783 s; unaccounted 1.116 s; ranking fill 0.417 s nested in shortlist_s 2.255/2.245 s vs frozen artifact shortlist_s 76.93/76.24 s. Timer begins in `main` after imports. Historical `search_wall_s` is not an instrumented complete-model baseline; no complete-model improvement claimed. | Next: Task 3. |
| 2026-09-06 | 3 | 0418b884 located; set incomplete | Shipped `enumerate_template_lns_neighbors` + `_sample_neighborhood` + `sample_template_lns_neighborhood` on frozen `lns_00_r00` full parent `98a32da5...` with IntervalPlanner stage windows. Seed 20261010 critical: emitted/unique/sampled/selected (ranked index 6, operator cross-domain d4, actual closure 99). Seed 20261011 critical: emitted and unique (same operator/closure), not sampled (`target_sampled=false`); therefore absent from `candidate_features`. Loss stage = per-operator shuffle-and-truncate N=25, not pre-sampling absence. Artifact `fast_plan_loss_0418b884.json` `72aff4ec...`. | Extend the audit to the rest of the measured fast-plan set before Task 4. |
| 2026-09-06 | 3 | Fast-plan set closed | Five tracked hashes on full parent `lns_00_r00`/`lns_00_r01` for seeds 20261010/20261011. Command: `audit_lns_fast_plan_loss.py --model <frozen> --proposal-seed <seed> --start lns_00_r00 --start lns_00_r01`. Models `9b334b78...` / `cc53d43d...`. Artifacts `fast_plan_loss_set_20261010.json` `3fa830fe...`, `fast_plan_loss_set_20261011.json` `4fe4dd55...`, summary `32bf1d54...`. Tests: `pytest -q tests/test_moe_executable_plan_neighborhood.py` → 20 passed; `tests/test_moe_lns_structural_presample.py` → 4 passed. Non-full parents membership-only. | Task 4 design replay on this set. |
| 2026-09-06 | 4 | Specified; not adopted | Lab policy `structural_coverage_then_random_v1` SHA256 `275dd663...`. Same shuffle as baseline; coverage is first representative of each nonempty `CLOSURE_BINS` bin. Design replay tracked-hash retention unchanged vs N=25 shuffle-truncate; rare-bin coverage improved on a few operators; sampling-lost tracked plans are later members of already-covered bins. Not wired into `sample_template_lns_neighborhood`. | Do not open Task 5 on this candidate. |
| 2026-09-06 | 4 | Width iteration rejected | `structural_coverage_closure_width_v1` SHA256 `e5a35b06...`. Locked `(closure_bin, width_histogram)` coverage on the same shuffle as baseline. Replay on frozen `9b334b78...` / `cc53d43d...` full-parent starts: prefix-stable baseline; v2 dropped `0418b884...` on seed `20261010` (idx 19, hist `[[4,8],[8,4],[16,1]]`) and recovered no sampling-lost tracked hash. Coverage keys increased on 24/24 operator cells. Artifact `task4_feature_replay.json` `f2d095cf...`. Tests: neighborhood+presample 26 passed; ranking 25 passed. Keep N=25 shuffle-truncate. | Task 5 stays closed. |
| 2026-09-06 | Post-Task-4 review | Verified with scoped caveats | Read and SHA-verified both fresh Arm models, compare JSONs (zero mismatches), and sidecars (622.820/618.783 s); checked five-plan scored/pooled membership in the original models and both presampler policy hashes. `PYTHONPATH=.:src .venv/bin/python -m pytest -q tests/test_moe_lns_structural_presample.py tests/test_moe_executable_plan_neighborhood.py` -> 26 passed in 0.12 s; the same command with `tests/test_moe_lns_diverse_shortlist.py tests/test_moe_partial_order_vnd_runner.py tests/test_moe_partial_order_hardware_frontier.py` -> 45 passed in 0.26 s. | Task 3/4 raw locations/full digests still need registration. No new hardware measurements in this review. Next: Task 6A preparation; Task 6B requires its explicit extra budget, Task 5 remains closed. |

For each completed task, add the source revision, artifact hashes, exact commands,
test results, measured baseline/candidate values where relevant, and the next
unchecked task. Mark a task complete only when its stated gate is satisfied.
