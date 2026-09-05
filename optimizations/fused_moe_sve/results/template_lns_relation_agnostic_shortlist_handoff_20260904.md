# Template-LNS relation-agnostic shortlist handoff

## Objective

Build and validate a bounded hardware shortlist for template-level LNS that
does not use the current partial-order relation for acceptance or pruning.
Reserve shortlist capacity across operator, target destroy size, actual
lane-atomic closure size, width histogram, LLC-domain assignment, and model-score
quantile. Use the existing measured suite only as a design replay, then freeze
the selector and validate it on one new independent median frontier.

The intended outcome is not a more accurate absolute cost model. It is a
deterministic offline selection policy that keeps structurally distinct plans
and retains the measured best under a declared hardware budget.

## Current decision and why this work is next

The fixed template-LNS neighborhood is useful on all three frozen traces, but
the local-move partial-order calibration does not transfer to large closures.

| Trace | Consensus winner | Median ms, S1/S2 | Gain vs strongest anchor, S1/S2 |
| --- | --- | ---: | ---: |
| High-skew | `189d70b0...` | 31.249 / 31.297 | +3.961% / +3.428% |
| Median | `2ab43572...` | 31.381 / 31.393 | +2.671% / +2.863% |
| Uniformish | `50a7d4be...` | 31.517 / 31.698 | +2.057% / +3.250% |

Median contains two cross-session false-pruning counterexamples. Its absolute
winner `2ab43572...` was predicted at -3.770% with an upper bound of -2.001%,
yet measured +2.290/+3.102% relative to its full anchor. Uniformish contains one
model-better candidate that did not pass the strict two-session P10 gate.

The current LNS runner therefore enforces:

```text
automatic acceptance = disabled
dominance pruning     = disabled
unmeasured candidates = budget-deferred, never proven worse
```

Partial-order evidence remains in artifacts only as a diagnostic. The next
bottleneck is hardware-budget allocation, not residual-radius fitting.

## Step 0: checkpoint the current implementation before new work

The repository base is commit `055fe250a376622e7595d452d90e0947abbdd337`
plus the current uncommitted partial-order, beam, and template-LNS work. Before
implementing the new selector:

1. Run the existing code-review gate and inspect the exact staged diff.
2. Stage the current source, tests, frozen pairwise profile, mathematical model,
   TODO, manifest, and result documents.
3. Do not stage `tmp/`, benchmark dumps, caches, compiled extensions, or model
   files.
4. Commit the checkpoint as one focused commit, for example:

   ```text
   feat: add hardware-assisted template LNS search
   ```

The current checkpoint should include these source-level additions and their
dependent modified files:

- `cpu_moe_schedule_optimization/planners/executable_plan_neighborhood.py`
- `cpu_moe_schedule_optimization/planners/profiles/arm_codex_80c_pairwise_ordering_gated_v8_20260904.json`
- `optimizations/fused_moe_sve/benchmarks/bench_executable_neighborhood_audit.py`
- `optimizations/fused_moe_sve/benchmarks/bench_partial_order_beam_layer.py`
- `optimizations/fused_moe_sve/benchmarks/build_partial_order_hardware_frontier.py`
- `optimizations/fused_moe_sve/benchmarks/bench_partial_order_hardware_frontier.py`
- `optimizations/fused_moe_sve/benchmarks/analyze_partial_order_hardware_frontier.py`
- `optimizations/fused_moe_sve/benchmarks/bench_template_lns_layer.py`
- `optimizations/fused_moe_sve/benchmarks/analyze_template_lns_frontier.py`
- `optimizations/fused_moe_sve/benchmarks/analyze_template_lns_suite.py`
- the related tests, `MATHEMATICAL_MODEL.md`, `TODO.md`, manifest, and result
  reports shown by `git status --short`.

Do not assume this handoff authorizes a commit automatically. The acting agent
must still review `git diff --cached` immediately before committing.

## Frozen inputs

Do not refit v8 or the residual report during shortlist development.

| Input | SHA256 |
| --- | --- |
| Frozen v8 calibration | `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3` |
| Native extension | `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f` |
| Gated pairwise report | `a1b89cec87cc6c9cacab4477965bd060b8f41ba7eac227ef055552609268e54a` |
| Suite analysis | `8e59f93276e31d183fea6db2741af0807cbe6662c77216ac2ecc7204ae8890bb` |

The suite-level raw artifact is:

```text
tmp/moe_partial_order_vnd_20260904/template_lns_suite_analysis.json
```

Raw artifacts are intentionally not source-controlled. Confirm that they exist
in the shared workspace before beginning replay. If they are missing, recover
them by exact hash; do not regenerate neighborhoods with a new seed and call the
result the same replay.

### Existing measured cases

| Case | Model | Frontier | Session 1 | Session 2 | Analysis |
| --- | --- | --- | --- | --- | --- |
| High-skew L1 | `32c5a530...` | `ed446199...` | `64219631...` | `ae226a3a...` | `a2ad7b24...` |
| High-skew L2 | `b2e01faf...` | `5f928bcc...` | `0b30d937...` | `fb49eb36...` | `aaefffcd...` |
| Median | `fb60dfa0...` | `443c3d3b...` | `27b0b6b5...` | `8bc15fb0...` | `79f5058a...` |
| Uniformish | `85190591...` | `f53d40a1...` | `e86795d5...` | `cea46c04...` | `088d2964...` |

Use the full hashes embedded in the analysis identity fields. The shortened
values above are navigation aids, not identity checks.

## Required candidate feature record

Create a compact, versioned feature record for every candidate before any
hardware shortlist is selected. Do not serialize a full executable plan for
every generated candidate; serialize full canonical state and PlanV2 only for
the nested audit frontier.

Minimum schema:

```json
{
  "schema_version": 1,
  "state_hash": "...",
  "anchor_state_hash": "...",
  "restart": 0,
  "strategy": "critical|random",
  "operator": "critical_window_template_repartition_cross_domain_d4_b16",
  "scope": "domain_local|cross_domain",
  "target_destroy_size": 4,
  "actual_closure_size": 91,
  "actual_closure_bin": "64_127",
  "changed_core_begin": 32,
  "changed_core_end": 70,
  "candidate_width_histogram": [[1, 14], [2, 1], [8, 4], [16, 2]],
  "width_histogram_delta": [[1, 2], [8, 4], [16, -2]],
  "domain_assignment_signature": "...",
  "cross_domain_lane_count": 0,
  "predicted_gain_pct": 0.818,
  "model_score_quantile": 3
}
```

Definitions:

- `target_destroy_size` is parsed from the operator and remains 4/8/16.
- `actual_closure_size` is exactly `len(moved_experts)`. Never substitute the
  target size; the winning high-skew d4 move had a 91-expert closure.
- Use fixed closure bins `1_15`, `16_31`, `32_63`, `64_127`, and `128_plus`.
- `candidate_width_histogram` is the sorted `(width, lane_count)` tuple for the
  complete candidate state.
- `width_histogram_delta` is candidate minus anchor and omits zero entries.
- `domain_assignment_signature` must be plan-visible and deterministic. Use,
  for each LLC domain in logical order, the ordered lane widths fully contained
  in that domain, plus the ordered `(core_begin,width,domain_ids)` descriptors
  for crossing lanes. Hash the canonical JSON representation for compactness,
  but keep the unhashed representation in debug output.
- `model_score_quantile` is an equal-frequency bin in `{0,1,2,3,4}` computed
  within one `(anchor_state_hash,restart)` proposal pool. Sort by predicted gain
  and canonical hash before assigning bins so ties are deterministic. Quantiles
  are diversity labels only; they are not confidence or acceptance levels.

Hardware measurements, relation labels, residual radius, or previous winner
status must not enter these features.

## MVP selector

The MVP budget is top-16 per named start, excluding the always-retained anchor.
The selector must be deterministic and prefix-nested so top-16 is a prefix of
top-32 for the independent audit.

### Phase A: canonical deduplication

Deduplicate by executable-state canonical hash. Merge all provenance records
for duplicates. Never discard a state because its partial-order relation is
`candidate_worse`.

### Phase B: mandatory categorical coverage

Select at most one state for each available operator, in stable operator-name
order. When choosing the representative for an operator, use this lexicographic
priority:

1. an unrepresented model-score quantile;
2. an unrepresented actual-closure bin;
3. an unrepresented domain-assignment signature;
4. an unrepresented width histogram;
5. larger categorical distance from the anchor;
6. canonical state hash.

The six default operators consume at most six of the 16 slots and guarantee
coverage of local/cross scope and target d4/d8/d16 when candidates exist.

### Phase C: score-quantile coverage

For every score quantile not yet represented, add one candidate using the same
structural novelty ordering. This consumes at most five total quantile slots,
including bins already covered in Phase B. In particular, quantile 0 and
quantile 4 are mandatory when non-empty; this keeps both model-worse and
model-better spectrum candidates without treating either as authoritative.

### Phase D: farthest-first structural fill

Fill the remaining slots with the candidate maximizing its minimum categorical
distance to the selected set. Use one point for each difference in:

- operator;
- target destroy size;
- actual closure bin;
- width histogram;
- domain-assignment signature;
- score quantile.

Break ties by lower current token reuse count across selected candidates, then
canonical hash. Do not add weights learned from hardware results in the MVP.

### Phase E: cross-start deduplication and backfill

After per-start selection, deduplicate globally by canonical hash. Backfill from
the corresponding per-start ranked remainder until each start has contributed
its declared budget or has no remaining unique candidates. Record requested,
selected, duplicate, and exhausted counts separately.

## Artifact contract

Add a separate LNS shortlist block instead of overloading `partial_order`:

```json
{
  "lns_diverse_shortlist": {
    "schema_version": 1,
    "policy": "relation_agnostic_categorical_farthest_first_v1",
    "shortlist_budget": 16,
    "audit_budget": 32,
    "selected_keys": [],
    "audit_keys": [],
    "budget_deferred_keys": [],
    "coverage": {},
    "duplicate_counts": {}
  },
  "partial_order": {
    "automatic_acceptance_enabled": false,
    "dominance_pruning_enabled": false
  }
}
```

Requirements:

- `selected_keys` must be an ordered prefix of `audit_keys`.
- Every `audit_key` must have a full canonical state and PlanV2 bridge.
- Every serialized state must reproduce its canonical hash and bridge exactly.
- Anchors are stored independently and do not consume shortlist budget.
- Candidate relation remains diagnostic metadata only.
- All non-selected candidates are `budget_deferred`; there is no `dominated`
  category for template LNS.

Use new roles in the hardware frontier:

- `lns_diverse_top16`
- `lns_diverse_audit_top32`
- `anchor`

Diagnostic relation may be a separate field; do not encode it in the role.

## Implementation map

Prefer a new Lab-local module rather than adding another production planner
dependency:

```text
optimizations/fused_moe_sve/benchmarks/lns_diverse_shortlist.py
optimizations/fused_moe_sve/benchmarks/replay_lns_diverse_shortlist.py
optimizations/fused_moe_sve/benchmarks/build_lns_diverse_hardware_frontier.py
optimizations/fused_moe_sve/benchmarks/analyze_lns_diverse_hardware_frontier.py
```

Expected integration points:

- `bench_executable_neighborhood_audit.py`: expose compact candidate feature
  rows before candidate states leave memory.
- `bench_template_lns_layer.py`: accept `--shortlist-budget 16` and
  `--audit-budget 32`; invoke the new selector in diagnostic-only mode.
- Keep `pairwise_plan_ordering.py` unchanged unless a generic non-LNS API is
  genuinely needed. The relation-agnostic selector must not acquire residual
  calibration semantics.
- Update `MATHEMATICAL_MODEL.md`, TODO, manifest, tests, and changelog in the
  same change because this alters offline candidate retention.

Minimum unit tests:

1. deterministic selection under shuffled input order;
2. canonical duplicate merge and deterministic backfill;
3. every available operator represented when `K >= operator_count`;
4. quantile 0 and 4 represented when available;
5. actual closure and target destroy kept distinct;
6. top-16 is an exact prefix of top-32;
7. width/domain signatures change when the corresponding executable structure
   changes but not when unrelated metadata changes;
8. no relation value changes selected/deferred semantics;
9. no candidate is labeled dominated or automatically accepted;
10. full state/PlanV2/hash round-trip for every audit plan.

## Phase 1: replay the existing measured suite

This is a design replay, not independent validation. The old hardware frontiers
contain only the former top-16 plus up to two model-worse spectrum candidates
per start, so the candidate universe is the measured subset, not all generated
LNS states.

For each of high-skew layer 1, high-skew layer 2, median, and uniformish:

1. Reconstruct candidate features from the frozen frontier canonical states and
   comparison provenance.
2. Run the new selector at `K={8,12,16}` on the measured subset.
3. Join both hardware sessions by `(candidate_hash,anchor_hash,start,operator)`.
4. Compute these metrics separately for each session and for cross-session
   consensus:
   - absolute measured-best recall;
   - per-parent measured-best recall;
   - strict-stable candidate recall;
   - selected-best regret versus the measured subset best;
   - operator, scope, target-size, closure-bin, width-histogram,
     domain-signature, and score-quantile coverage;
   - canonical duplicate and budget-deferred counts.
5. Report partial-order relation composition only as a diagnostic cut.

Replay success gate at K=16:

- retain the absolute measured best in both sessions for all four cases;
- retain the cross-session consensus winner for all four cases;
- selected-best regret is exactly zero in both sessions;
- no candidate is classified as dominance-pruned or automatically accepted;
- deterministic output hash under at least three shuffled input orders.

If K=16 fails on the design replay, change only the structural selection rule,
record the failed variant, and repeat. Do not change LNS generation, v8,
residual radius, or hardware measurements.

## Phase 2: freeze the selector

After replay passes:

1. Freeze the selector schema, category definitions, quantile calculation,
   tie-breaking, K=16 budget, and top-32 audit construction.
2. Save a machine-readable calibration-free policy artifact containing the
   selector version and source commit SHA. It must not contain hardware timing
   values or trace-specific state hashes.
3. Record the policy SHA256 in every subsequent model, frontier, session, and
   analysis artifact.
4. Do not modify the selector after opening the independent frontier. A failure
   requires a new version and a new independent validation frontier.

## Phase 3: independent median frontier

Use median because it produced the strongest observed model-order reversal.
This is a new proposal frontier, not a replay of the old 145 measured plans.

Recommended locked inputs:

```text
trace: measured_request016_case017_zh2048-018.pt
layer: 4
proposal seed: 20261010
parents: reconstructed full, one-step, greedy, fixed-width controls
restarts per unique parent: 1
critical experts: 32
neighbors per operator: 50
destroy sizes: 4,8,16
repair beams: 16,32,64
templates per block: 4
shortlist K: 16 per parent
audit K: 32 per parent
```

Generate one deterministic ranked prefix per parent. Freeze and serialize the
union of audit top-32 states. With four unique median parents, the maximum
hardware frontier is 4 anchors plus 128 audit candidates before cross-parent
deduplication, comparable to the previous 145-plan median session.

Run two independent hardware sessions:

```text
processes: 1
NUMA: CPUs 240-319, membind 3
weight allocations: one batch per session
weight copies: 4
warmup: 5
paired randomized rounds: 31
suggested seeds: 20261011 and 20261012
calibration/refit: forbidden
```

The hardware runner measures the complete top-32 audit frontier. Analysis then
compares the nested top-16 subset with the measured top-32 oracle.

Independent success gate:

- top-16 contains the top-32 absolute measured best in both sessions;
- top-16 contains the top-32 cross-session consensus winner;
- top-16 selected-best regret is zero in both sessions;
- every parent anchor is retained and executable;
- all outputs are bit-exact;
- both sessions have identical frontier, extension, route, policy, and
  calibration hashes;
- automatic acceptance and dominance pruning remain disabled;
- no post-hoc selector change is made from the validation results.

Also report top-8, top-12, top-24, and top-32 sensitivity, but do not replace the
predeclared K=16 decision with whichever budget looks best afterward.

If top-16 fails but top-24 passes, record the budget-quality tradeoff and stop.
Do not silently raise the default budget. If top-32 itself contains no stable
candidate above the strongest anchor, the result is a neighborhood/proposal
failure for that independent seed, not a selector recall failure.

## Decision table

| Replay K16 | Independent K16 vs K32 | Decision |
| --- | --- | --- |
| Pass | Pass | Adopt selector v1 for offline LNS hardware shortlist; next optimize enumeration cost. |
| Pass | Fail, K24 passes | Keep v1 experimental; report required larger budget and do not change default. |
| Pass | Fail through K32 | Reject v1; design a new selector version and require a new independent frontier. |
| Fail | Not run | Fix structural diversity on the design replay before opening holdout. |

No outcome in this table authorizes production integration or a global
optimality claim.

## Common failure modes

- Treating target d4/d8/d16 as actual closure size. These are often very
  different.
- Using width histogram alone. It cannot distinguish domain placement or task
  order.
- Using partial-order relation as a mandatory quota. This reintroduces the
  failed model decision through the back door.
- Computing score quantiles across different anchors. Scores are anchor-relative
  and must be binned within one proposal pool.
- Selecting top-16 and measuring only top-16. That makes measured-best recall
  tautological; the independent audit must measure top-32.
- Recreating a neighborhood during hardware replay. Hardware must consume the
  serialized executable states from the frozen frontier.
- Comparing only parent-relative gains. Always compute absolute plan medians and
  cross-session consensus across all anchors and candidates.
- Counting budget-deferred candidates as false pruning or safe pruning. There is
  no dominance claim in LNS diagnostic mode.
- Tuning selector weights or bins after viewing the independent top-32 hardware
  results.
- Adding ALNS weights before shortlist recall closes. Operator adaptation is not
  the current bottleneck.

## Required completion record

The implementing agent must leave:

1. focused unit tests and exact commands/results;
2. existing measured-suite replay JSON and a concise technical report;
3. frozen selector policy artifact and SHA256;
4. independent median model, nested frontier, two hardware sessions, and final
   analysis artifacts;
5. model/hardware call counts and wall time;
6. top-K recall/regret and structural coverage at K=8/12/16/24/32;
7. explicit adopt/reject decision and remaining limitation;
8. `MATHEMATICAL_MODEL.md`, TODO, manifest, and changelog updates;
9. no raw benchmark data in the source commit unless explicitly requested.

## Further questions after the gate

Only after selector v1 passes the independent gate:

- Can candidate feature extraction and lane-closure enumeration be cached
  without changing the ranked prefix?
- Can affected-interval event replay reduce the 3,488.51-second suite planning
  cost while preserving the same top-32 audit set?
- Does a second independent trace require K greater than 16?
- Are local and cross-domain operators complementary enough to justify ALNS, or
  is the fixed mixture sufficient once shortlist diversity is explicit?
