# One bounded order-only layer around the hardware winners

Status: completed within budget. High-skew `cyclic_rotation_1` passes both
sessions; median retains its anchor. Lab-only; no production neighborhood,
calibration, width or default selection change.

## Outcome

One useful order-only candidate was found on high-skew: paired median speedup
3.887% / 2.243%, P10 3.561% / 1.491% in the two independent sessions. It was
the diversity rotation candidate, predicted **−4.478%**, not the model-best
rotation. This is an explicit order-ranking counterexample and supports keeping
non-model-top diversity slots. It does not justify refitting on this result.

Median has no passing candidate. Keep its original anchor. For high-skew retain
the passing candidate as the result of this layer, with full executable bridge
in the frozen frontier. No second layer was launched and previous winner files
were not overwritten. The result is best within this tested set, not globally
optimal. Significant worsening from other orders rules out interpreting a
negative search result as "order has no effect".

Connection recovery: the first upload/connection attempts failed and produced
no confirmed measurements. The user restored machine connectivity. Before the
new launch the isolated output directory was empty and no probe was running.
The machine had rebooted (uptime four minutes at launch), boot ID
`8da74bfd-16d5-45a6-a6ad-1e0cd3af3b19`, with THP policy `[always] madvise never`.
Startup services were visible in the preflight, so do not claim a continuously
idle host. All comparisons use freshly measured in-session anchors; do not
compare absolute times across the reboot. Extension identity is checked against
the frozen winner before any weight preparation or timing.

## Predeclared budget and selection

Anchors: the saved hardware-consensus score-best plans from median
request016/layer4 and high-skew request008/layer38. Reuse frozen v8 and the
same measured extension. Restore complete executable state and require exact
bridge round-trip plus historical anchor event-score parity within 1 ns.

Keep CPU mapping, per-lane task membership, routes, width, stage windows and
early merge unchanged. Regenerate lane-local serial dependencies after reorder.
Only three families, each with at most 24 unique proposals from at most 256
seeded attempts:

1. Independently reverse a random nonempty subset of eligible lanes.
2. Cyclically rotate a random nonempty subset of lanes by nonzero offsets.
3. Simultaneously exchange head and tail in at least two eligible lanes.

Per family retain its fastest event prediction plus the proposal maximally
distant from that winner by task-position Hamming distance (hash tie-break).
Do not require positive model gain. Deduplicate against anchor and already
retained families. Thus at most 72 model scores and six hardware candidates per
trace, plus its anchor. No extra layer, adaptive budget or margin reduction.

Hardware: NUMA3 CPUs240–319, H4096/F512, E256, 2048 tokens, TopK6, BF16 SVE
N tile8; full-owner-stripe W13/W2 windows0. Each trace runs two independent
process sessions, fixed tensor seeds, five warmups and 31 measured rounds.
Randomize the seven plans in each round; all use the same rotating weight copy.
Scrub disjoint 216 MiB before each call outside timing, retain four copies.
Check every plan's output against the anchor bitwise on all four copies first.

Positive paired speedup is `100*(anchor/candidate-1)`. For an actionable new
anchor require median >2% and P10 >0 in **both** sessions. Report all selected
candidates, including model-worse diversity controls; no post-hoc candidate
replacement. A shared round anchor is used, not a fresh immediately adjacent
anchor call for every candidate. This increases within-round drift sensitivity.

This is a known-trace development experiment, not a fresh holdout. Failure only
limits this sampled neighborhood; it does not prove order globally unimportant.

## Frozen model shortlist

Predicted speedup percent relative to each restored anchor:

| Candidate | Median | High-skew |
| --- | ---: | ---: |
| independent_reverse_0 | +0.298 | +0.042 |
| independent_reverse_1 | −4.514 | −8.463 |
| cyclic_rotation_0 | +0.452 | +0.009 |
| cyclic_rotation_1 | −2.038 | −4.478 |
| joint_head_tail_0 | +0.878 | +0.055 |
| joint_head_tail_1 | −0.406 | −10.692 |

The `_0` candidate is model-best within its family; `_1` is the distance-based
diversity candidate. These are model predictions, not hardware measurements.
Focused generator validation: four tests passed (bounded deterministic proposals,
exact per-lane work/geometry preservation, and empty order neighborhoods).

## Artifacts and commands

Actual frozen-artifact inspection also verified all 14 bridges preserve their
anchor's exact per-lane expert/window membership, width, CPU mapping and merge
setting. The candidate task ordering is the only intentional execution change.

## Median completed result

| Candidate | S1 median ms | S2 median ms | Paired gain S1 / S2 | P10 S1 / S2 |
| --- | ---: | ---: | ---: | ---: |
| anchor | 31.9790 | 32.0134 | — | — |
| independent_reverse_0 | 31.9077 | 31.9193 | +0.166% / +0.286% | −0.583% / −0.244% |
| independent_reverse_1 | 34.4291 | 34.9731 | −7.156% / −8.487% | −7.800% / −9.656% |
| cyclic_rotation_0 | 32.0853 | 32.0451 | −0.404% / −0.058% | −1.295% / −0.697% |
| cyclic_rotation_1 | 35.0314 | 35.1077 | −8.688% / −8.638% | −10.061% / −9.358% |
| joint_head_tail_0 | 32.8606 | 32.9871 | −2.537% / −2.899% | −3.373% / −3.589% |
| joint_head_tail_1 | 36.2390 | 36.6223 | −11.623% / −12.482% | −12.298% / −13.335% |

No candidate passes the two-session actionable gate. Retain the original
median anchor. Notably joint_head_tail_0 was predicted +0.878% but lost about
2.5–2.9%; model-side order ranking remains imperfect. Large negative outcomes
show order matters, not that this sampled set contains a better order.

## High-skew completed result

| Candidate | S1 median ms | S2 median ms | Paired gain S1 / S2 | P10 S1 / S2 |
| --- | ---: | ---: | ---: | ---: |
| anchor | 32.6142 | 32.4773 | — | — |
| independent_reverse_0 | 32.6227 | 32.3079 | +0.022% / +0.556% | −0.551% / −0.162% |
| independent_reverse_1 | 33.4997 | 34.0291 | −2.612% / −4.708% | −2.979% / −5.237% |
| cyclic_rotation_0 | 32.2246 | 32.1867 | +1.290% / +0.914% | +0.774% / +0.020% |
| **cyclic_rotation_1** | **31.3980** | **31.7858** | **+3.887% / +2.243%** | **+3.561% / +1.491%** |
| joint_head_tail_0 | 32.5809 | 32.4982 | +0.135% / +0.093% | −0.417% / −0.697% |
| joint_head_tail_1 | 32.8780 | 33.9682 | −0.803% / −4.340% | −1.377% / −6.276% |

The passing plan rotates 17 of 20 lanes, including all four 16T lanes, without
changing any task's lane or width. Wide-lane rotations at logical core begins
0/16/32/48 are 12/1/14/6 task positions, respectively. This is a coordinated
sequence change, not simply a universal ascending/descending route sort.

Candidate canonical state hash:
`34c08621e1120160ec29bceb4610e18bbe1e98ff04f2a6e0071bb53b3f92879a`.
Complete PlanV2 bridge hash:
`f0f81e0667fdbc26538df8288305af4ada4cc78d4982beaff22e98d4422a2cd4`.
The bridge is `plans.cyclic_rotation_1.bridge` in `high_skew_frontier.json`.

All seven plans per trace passed bitwise BF16 comparison on all four copies in
both sessions. Both trace analyses validated complete rounds and matching
frontier/extension/runner identities. The extension matches the preceding
selection experiment, SHA256
`dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`.
There is session drift, especially among high-skew reordered plans; the positive
candidate nevertheless passes the predeclared gate in each session separately.
These are descriptive development comparisons over six tested candidates, not
a multiple-testing-adjusted guarantee or a demonstrated physical mechanism.
Scrub/copy rotation does not prove every access missed LLC; no PMU or actual
per-allocation page-backing audit was performed in this MoE timing run.

## Reproduction details and retained evidence

Local ignored directory: `tmp/bounded_order_20260906/`.
Remote: `Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/bounded_order_20260906/`.
Complete bridges and hashes are in `median_frontier.json` and
`high_skew_frontier.json`. The runner is
`optimizations/fused_moe_sve/benchmarks/bench_bounded_order_extension.py`.

Freeze example from local repository root:

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/bench_bounded_order_extension.py freeze \
  --source-plans tmp/selection_pair_20260906/plans.json \
  --winner tmp/selection_winners_20260906/median.json \
  --calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --frontier tmp/bounded_order_20260906/median_frontier.json
```

For high-skew use source `tmp/selection_other_traces_20260906/high_skew_plans.json`,
winner `tmp/selection_winners_20260906/high_skew.json`, and `high_skew_frontier.json`.

Remote measurement from repository root (copied runner only, no production sync):

```bash
env OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:src \
  timeout 300 numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/bounded_order_20260906/bench_bounded_order_extension.py measure \
  --route-file bench_assets/moe_paper/dsv4_routes_pt_20260830/measured_request016_case017_zh2048-018.pt \
  --frontier tmp/bounded_order_20260906/median_frontier.json \
  --seed 20260906 --output tmp/bounded_order_20260906/median_session1.json
```

Repeat with seed20260907/session2. High-skew uses request008_case009_zh2048-010.pt
and the high-skew frontier/session names; layer is taken from the frozen frontier.

Analyze each trace after both sessions have completed:

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_bounded_order_extension.py \
  --frontier tmp/bounded_order_20260906/median_frontier.json \
  --sessions tmp/bounded_order_20260906/median_session1.json tmp/bounded_order_20260906/median_session2.json \
  --output tmp/bounded_order_20260906/median_summary.json
```

The analyzer checks identities, independent session seeds, full 36x7 cells,
finite positive times, copy pairing and correctness flags before assessing the
predeclared gate. It reports every candidate and saves raw artifact hashes.
Focused tests (generator, state invariants, paired analysis): 11 passed.
