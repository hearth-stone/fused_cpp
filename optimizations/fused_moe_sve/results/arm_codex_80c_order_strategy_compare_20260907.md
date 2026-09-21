# Interleave versus extended order proposals from latest measured anchors

Status: completed within frozen budgets. No candidate passes both-session
actionability. Extended finds a repeatable sub-threshold median improvement;
high-skew retains its latest anchor. No fitting/default acceptance changes.

## Outcome

| Trace | Strategy | Measured candidates | Both-session actionable | Best outcome including anchor |
| --- | --- | ---: | ---: | --- |
| median | interleave | 3 | 0/3 | anchor in both sessions |
| median | extended | 6 | 0/6 | block relocation, +1.716% / +2.342% paired median |
| high-skew | interleave | 4 | 0/4 | anchor in both sessions |
| high-skew | extended | 6 | 0/6 | effectively anchor; tiny S1 head/tail difference does not repeat |

Median block relocation is faster with positive P10 (1.274% / 1.694%) in both
sessions, but S1 median does not exceed the frozen 2% threshold. Preserve it as
an exploratory near-elite, not a promoted winner. Do not describe this as no
measured improvement just because the actionable count is zero. Its medians are
31.21924 / 31.00528 ms versus anchor 31.77141 / 31.72631 ms. Extended measured
twice as many candidates as interleave here; the small advantage cannot be
attributed solely to operator quality independent of hardware budget use.

High-skew anchor medians are 30.42646 / 30.39889 ms.
The closest extended candidate has paired medians +0.045% / −0.172%, P10
−0.418% / −0.638%. No meaningful repeated improvement. Template-only and broader
search both preserve the already measured good solution, rather than rediscovering
the larger gains available from older anchors.

Decision: stop this fixed order-only comparison, keep both original anchors and
retain the median block candidate/evidence separately. Do not enlarge order-only
budgets or relax the margin post hoc. A separately scoped next search would
change same-width lane task membership (swap/relocation), rather than another
unbounded permutation layer. These selected pools do not prove the full order
space is flat, and the synthetic template benefits do not transfer automatically
to these already optimized real-route anchors.

## All candidate outcomes

Paired median speedup relative to the corresponding in-session anchor:

| Median candidate | S1 | S2 |
| --- | ---: | ---: |
| lane stagger model representative | −8.508% | −7.580% |
| alternating model representative | −8.521% | −8.916% |
| alternating diversity representative | −7.471% | −7.831% |
| independent reversal | −0.113% | −0.452% |
| rotation diversity | −5.392% | −3.841% |
| joint head/tail | −1.341% | −1.693% |
| block relocation diversity | **+1.716%** | **+2.342%** |

| High-skew candidate | S1 | S2 |
| --- | ---: | ---: |
| lane stagger model representative | −5.695% | −6.445% |
| alternating model representative | −1.700% | −1.783% |
| lane stagger diversity representative | −9.982% | −9.539% |
| alternating diversity representative | −1.388% | −1.460% |
| independent reversal | −0.904% | −1.073% |
| rotation diversity | −0.879% | −1.540% |
| joint head/tail | +0.045% | −0.172% |
| block relocation diversity | −3.433% | −3.595% |

All union plans passed bitwise BF16 output equality on all four weight copies
in both sessions. Analyzer validates complete 36x8 / 36x9 paired records and
matching frontier/extension/runner identities. No PMU collection or actual
allocation-backing audit was performed; scrub/copy rotation is not proof of
all-DRAM access. Every timed warmup/sample is retained. No previous winner file
was overwritten and no next hardware layer was launched.

Median near-elite full bridge: `plans.p07.bridge` in `median_union.json`.
State hash: `3f020d959ee68a96ab0f0b1fc3a779cfa070c974147de613e505f5ef0ebdae07`;
bridge SHA256: `1299d0221edeae53953258026ad0d366437ae5c3bc0b341af94b373259b42d42`.

## Anchors and comparison boundary

Median uses its retained score-best hardware winner. High-skew uses the latest
GEMM-density `smooth_global` candidate, canonical hash `84e15dea...`, not the older
rotation or full-selector incumbent. The added anchor adapter loads exact task
metadata from the original plans, validates two measured sessions against the
new anchor frontier, requires the chosen key to pass their stable-gain gate,
and reconstructs the complete bridge with unchanged lane membership/geometry.

Each strategy has the same upper budget: six candidates plus anchor. Preserve
the existing `interleave` and `extended` selection rules, including model-best/
diversity representatives; do not invent candidates to fill unused slots. Thus
this compares generation **and representative selection**, not every generated
template or equal actual sample counts. Union both frozen lists by complete
bridge hash and measure shared plans only once. No post-measurement replacements.

| Trace | Strategy | Scored candidates | Retained candidates | Generation / scoring seconds |
| --- | --- | ---: | ---: | ---: |
| median | interleave | 5 | 3 | 0.0025 / 0.2414 |
| median | extended | 101 | 6 | 0.0286 / 4.8120 |
| high-skew | interleave | 6 | 4 | 0.0028 / 0.4576 |
| high-skew | extended | 102 | 6 | 0.0310 / 8.4675 |

These are diagnostic local macOS freeze timings, not isolated throughput
benchmarks; orchestration overlap/cache effects can affect absolute wall time.
Scored counts give the more robust work comparison. Some generated templates
are not retained by the existing per-family representative policy even when the
overall hardware cap has spare slots. Do not label retained count as the entire
possible template space.

Median union: 8 plans including anchor; high-skew union: 9. Strategy membership
and original labels survive deduplication. The measurement/analyzer default cap
remains 7; this explicitly authorized union uses `--max-plans 13`, bounded by
two independent six-candidate lists plus a shared anchor.

## Protocol and criteria

Reuse frozen v8/extension, actual captured routing, synthetic fixed-seed weights,
hidden and router probabilities. E256/H4096/F512, BF16 SVE N tile8, NUMA3 CPUs
240–319; same per-lane task membership/width, full stripes/window0, early merge
unchanged. Four packed copies, paired-by-round same copy, disjoint 216 MiB scrub
before every call outside timing, randomized union-plan order. Five warmups,
31 effective rounds, two independent processes per trace. Bitwise output equality
on every plan/copy before timing. No new kernel or production-source sync.

Report per-strategy actual candidate count, stable fraction and best measured
plan including anchor. Stable candidate requires >2% paired median speedup and
positive P10 in both sessions. Best-of-union recall is descriptive only: union
is a selected pool, not an unbiased oracle. Do not attribute a strategy win solely
to its operators if it also consumed more hardware samples. If neither strategy
improves the latest anchor, stop this order-only budget rather than claim the
entire permutation space has no useful solutions.

## Reproduction and artifacts

Local ignored directory: `tmp/order_strategy_compare_20260907/`.
Remote: `Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/order_strategy_compare_20260907/`.
Files: `<trace>_interleave.json`, `<trace>_extended.json`, `<trace>_union.json`,
two sessions and `<trace>_summary.json`. Raw source/plan/session identities are
retained in the frozen and summary artifacts.

Freeze median using the existing `--winner tmp/selection_winners_20260906/median.json`
and source `tmp/selection_pair_20260906/plans.json`. Freeze high-skew with:

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/bench_bounded_order_extension.py freeze \
  --proposal-set extended \
  --source-plans tmp/selection_other_traces_20260906/high_skew_plans.json \
  --anchor-frontier tmp/gemm_density_20260907/high_skew_frozen.json --anchor-key smooth_global \
  --anchor-sessions tmp/gemm_density_20260907/high_skew_session1.json tmp/gemm_density_20260907/high_skew_session2.json \
  --calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --frontier tmp/order_strategy_compare_20260907/high_skew_extended.json
```

Repeat freeze for `interleave`, then merge:

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/compare_order_strategies.py merge \
  --interleave tmp/order_strategy_compare_20260907/high_skew_interleave.json \
  --extended tmp/order_strategy_compare_20260907/high_skew_extended.json \
  --output tmp/order_strategy_compare_20260907/high_skew_union.json
```

Measure the union using the copied bounded runner, exact existing NUMA/environment
protocol and `--max-plans 13`, seeds20260907/20260908. Analyze:

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/compare_order_strategies.py analyze \
  --frontier tmp/order_strategy_compare_20260907/high_skew_union.json \
  --sessions tmp/order_strategy_compare_20260907/high_skew_session1.json tmp/order_strategy_compare_20260907/high_skew_session2.json \
  --output tmp/order_strategy_compare_20260907/high_skew_summary.json
```

Focused anchor/union/generator/state validation: 19 tests passed. Native output
and performance validation completed under the frozen protocol above. Ruff,
manifest parsing and diff checks pass. No commit was created.
