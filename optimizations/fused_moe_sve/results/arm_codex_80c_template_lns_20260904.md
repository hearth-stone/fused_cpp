# Arm-codex 80C template-level LNS

## Decision

Adopt the template-level LNS structure for the high-skew offline search track
and expand its consensus hardware elite. Under essentially identical model-call
and exactly identical hardware-plan budgets, the LNS escaped the closed local
beam basin, found 40 unique candidates that were stable above their parent in
both sessions, and produced a consensus plan at `31.249/31.297 ms`.

This is not yet a three-trace or production result. Median and uniformish remain
untested, and the LNS model replay is 2.35x slower than the local beam despite
using slightly fewer event calls.

The subsequent high-skew depth-2 and cross-trace continuation is recorded in
`arm_codex_80c_template_lns_suite_20260904.md`. It supersedes this first-layer
handoff for the final LNS comparator policy.

## Equal-budget result

| Search | Event calls | Model wall | Hardware plans | Stable unique candidates |
| --- | ---: | ---: | ---: | ---: |
| Closed local beam, depth 2+3 | 3,535 | 507.06 s | 106 | 3 at depth 2; 0 at depth 3 |
| Template LNS, first layer | 3,512 | 1,193.47 s | 106 | 40 |

The comparison fixes the frozen v8 analytical calibration, the gated top-16
partial-order report, the native extension identity, the high-skew route trace,
and the total number of measured plans. The LNS spends more wall time per event
because it constructs and scores larger multi-lane DAG changes.

## Search definition

The immutable parent set was the original full state `b4312068...`, the first
hardware-beam global state `4115ab99...`, and the depth-2 state `514ced0d...`.
Each parent used two independently seeded proposal restarts. Each restart:

- selected 32 event-critical experts and an equal-size random control;
- used target destroy sizes 4/8/16 with repair beams 16/32/64;
- considered domain-local and cross-domain lane closures;
- retained at most four near/far width templates per closure;
- jointly repaired width, assignment, and four deterministic order policies;
- scored at most 50 candidates per scope/size operator with the complete placed
  event model;
- used the frozen partial order only for confidently-worse pruning and top-16
  retention.

The destroy size is a target, not the final number of moved experts. Lane
topology is atomic: the winning `d4` proposal had a 91-expert closure. The
artifact records all moved expert IDs, and every candidate stores its canonical
state and PlanV2 bridge.

The six starts produced 3,500 unique candidates. Partial-order classifications
were `0 better / 2,131 worse / 1,369 incomparable`; no candidate was accepted
from the model alone. The six top-16 lists contained 96 records, 92 unique
states, all six scope/size operators, and nine width shapes.

## Two-session hardware evidence

Both sessions ran on Arm-codex-internal NUMA3 CPUs 240-319 in a new process,
with one weight allocation batch, four-copy rotation, five warmups, 31 randomized
paired rounds, and bit-exact output checks before timing.

| Metric | Session 1 | Session 2 |
| --- | ---: | ---: |
| Timed wall | 116.20 s | 116.08 s |
| Single-session stable comparisons | 50 / 108 | 44 / 108 |
| Consensus winner median | 31.249 ms | 31.297 ms |
| Preserved `514ced0d...` median | 32.487 ms | 32.401 ms |
| Original full median | 33.023 ms | 32.915 ms |

Across sessions, 42 comparison records covering 40 unique candidates passed
`median > 2%` and `P10 > 0` in both sessions. All 42 came from cross-domain
repair: 13 from target d4, 10 from d8, and 19 from d16. Domain-local repair
produced no stable winner. None of the 11 deduplicated `candidate_worse`
sentinels was a false prune.

The model-to-hardware Spearman correlation within the measured incomparable
frontier was `0.720/0.706`. It ranked the LNS shortlist usefully, but its gain
magnitude remained too conservative for automatic acceptance.

## Winning structure

The normalized cross-session winner is state `189d70b0...`. It was the absolute
fastest plan in session 1; session 2 selected `a7b38202...`, but `189d70b0...`
was only `0.0076%` slower, so the disagreement is below the measurement
resolution.

Relative to its `4115ab99...` parent, `189d70b0...` measured:

| Session | Paired median gain | P10 | Model point gain | Model lower bound |
| --- | ---: | ---: | ---: | ---: |
| 1 | +3.880% | +1.974% | +0.818% | -0.951% |
| 2 | +3.271% | +1.506% | +0.818% | -0.951% |

Using same-session absolute medians, it was `3.961/3.528%` faster than the
preserved local-beam state `514ced0d...` and `5.676/5.169%` faster than the
original full plan.

The changed core interval is 32-69 inclusive. The parent block widths
`16,16,1,1,1,1,1,1` and its 91 experts were repaired as
`8,8,8,8,1,1,1,1,1,1`. This removes the old 16-thread lane crossing the
40-core LLC boundary. The four repaired 8-thread lanes carry
`1338/1341/1332/1336` routes; the 1,341-route expert receives its own 8-thread
lane. The rest of the plan remains byte-identical.

This result supports a structural interpretation: the useful jump required a
coordinated boundary repair and task redistribution, not another one-step swap,
split, or insertion. It does not identify a new absolute cost-model term.

The backend N tile is 16. Full W13 and W2 stage sizes are 8 MiB and 4 MiB.
Every winning-plan task uses `window_tiles=0`, meaning one full owner stripe and
`R13=R2=1`; the resulting geometry is:

| Tasks | Team | W13 owner tiles/thread | W2 owner tiles/thread | `(w13_window,w2_window,R13,R2)` |
| ---: | ---: | ---: | ---: | --- |
| 164 | 1T | 64 | 256 | `(0,0,1,1)` |
| 22 | 2T | 32 | 128 | `(0,0,1,1)` |
| 15 | 8T | 8 | 32 | `(0,0,1,1)` |
| 30 | 16T | 4 | 16 | `(0,0,1,1)` |

## Limits and next gate

- Only the high-skew trace was measured. Do not generalize the operator mix or
  no-regression claim to median or uniformish.
- The nominal d4 label hides a 91-expert lane closure. Future analysis must use
  both target and actual destroy size.
- Session-best hashes differ at a sub-0.01% gap. Preserve a small consensus
  elite rather than declaring a unique hardware optimum.
- The current Python implementation is an offline method: 1,193.47 seconds for
  one layer is not suitable for production planning.

Next, expand the consensus high-skew elite once with the same frozen gate and
then run the fixed LNS mixture on median and uniformish. Do not fit new residual
radii from these frontier measurements. Consider ALNS weights only if the
winning scope/size mix is complementary across trace classes.

## Reproducibility and artifacts

- Repository base: commit `055fe250a376622e7595d452d90e0947abbdd337`
  plus the uncommitted Lab changes described here.
- Machine: Arm-codex-internal, NUMA3, CPUs 240-319, `--membind=3`.
- Existing native extension; the command passed no HugeTLB/page-policy override.
  The runner did not capture inherited shell page-policy variables, which is a
  reproducibility limitation.
- Shape: 2,048 tokens, TopK 6, 256 experts, H=4096, F=512, bf16, 80 threads.
- Route: `measured_request008_case009_zh2048-010.pt`, layer 38, SHA256
  `d6a04171b059381ac383f5ff61ce7b24e9d84a465dc58886a29d8c55d6eb6724`.
- Calibration SHA256:
  `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`.
- Extension SHA256:
  `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`.
- Runtime controls set by the hardware runner:
  `FUSED_CPP_MOE_SVE=1`, `FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE=1`,
  `FUSED_CPP_MOE_W2_BF16_ROUTE=0`, and
  `FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE=1`.

The model replay entry point was `bench_template_lns_layer.py` under
`PYTHONPATH=.:src numactl --physcpubind=240-319 --membind=3`, with the three
parent hashes, `--critical-experts 32`, `--neighbors-per-operator 50`,
`--lns-destroy-sizes 4,8,16`, `--lns-repair-beam-widths 16,32,64`,
`--lns-templates-per-block 4`, `--restarts-per-parent 2`, and seed `20260917`.
Hardware used `bench_partial_order_hardware_frontier.py` with the same NUMA
binding, `--warmup 5 --runs 31 --weight-copies 4`, and seeds
`20260921/20260922`. `analyze_template_lns_frontier.py` generated the final
cross-session decision.

| Artifact | SHA256 |
| --- | --- |
| Model replay | `32c5a530c4c75462fd06c20f9340ec6e9977eb406d05e88bc3d29b651643d784` |
| Hardware frontier | `ed446199e673fa9da58c7024e3e06ec70d25361caf1a279c6e8be53df9a5d947` |
| Hardware session 1 | `642196316e2384210e9e2ce03ed42f66cc8fe19fe5fb0ea36e5da11551b76dde` |
| Hardware session 2 | `ae226a3a6e20b5e85237d8fa2ae437d9455f776fee5f28d69494c8ae907744de` |
| Formal analysis | `a2ad7b24dc1445c8e5e3af8fab26706ba14980b0a6f6ee985cdf9170d02d3ad1` |

Raw JSON artifacts remain under `tmp/moe_partial_order_vnd_20260904/` and are
not source-controlled.
