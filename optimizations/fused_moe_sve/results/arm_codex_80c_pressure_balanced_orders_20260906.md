# Pressure-balanced order proposals: bounded hardware test

Status: completed within budget; no candidate passes the two-session >2% gate.
Lab proposals only; no cost-model fitting,
production scheduling, widths, task membership or packed-format changes.

## Outcome

No automatic anchor replacement. Median candidates all regress. High-skew
alternating routes has repeatable small positive paired medians (1.037/1.960%,
P10 0.414/1.657%), but fails the unchanged requirement that both medians exceed
2%. Keep it as an exploratory candidate, not a promoted winner. The original
winner files are unchanged and no further experiment was launched.

The smoothness proxy is not validated as a ranking objective: smooth-domain
loses on both traces; smooth-global is inconsistent on high-skew and regresses
on median; the deliberately burstier control improves modestly on high-skew.
Actual hardware pressure was not measured, so this rejects neither the general
value of compute/memory complementarity nor all better pressure estimators.
It does reject using this isolated-phase second moment as a reliable selector
from this evidence. Do not add a fitted physical correction based on these cases.

## Hypothesis and boundaries

Interleave tasks so their isolated predicted high-memory-demand phases overlap
less. Do not assume that all large-route tasks are compute-only or all small-route
tasks continuously saturate memory. The original simple stream-count/queue proxies
failed earlier validation; this test uses a phase-rate proxy only to generate
candidates and lets hardware decide whether any is useful.

For each predicted isolated phase use offered rate
`r = phase.dram_bytes(phase.isolated_spill_fraction) / phase.base_ns`.
Concatenate phases along each lane without injected delay. Sum active rates to
obtain `q(t)`. For this order-only domain, total modeled bytes and the isolated
horizon are invariant. Minimize `integral(q(t)^2)/T` (equivalently CV-squared
up to fixed mean), not the invariant total concurrency integral. This is offered
load on an isolated timeline, **not achieved bandwidth or observed queue latency**.
The byte model retains its existing isolated cache assumptions and may be wrong.

The domain variant minimizes `integral(sum_d q_d(t)^2)/T`. Attribute a lane's
offered load to its CPU-origin LLC domains, splitting cross-domain lanes by core
fraction. This is an injection-side approximation, not independent DDR capacity:
node3 memory interleaves across both SCCLs. Neither objective accounts for
contention-driven phase-time changes, hardware prefetch or actual arrival shape.

## Fixed budget

Two traces, each with anchor plus at most four unique candidates:

- `smooth_global`: greedy descent on global second moment.
- `smooth_domain`: greedy descent on sum of domain second moments.
- `bursty_control`: greedy ascent on global second moment, a directional control.
- `alternating_routes`: alternate high/low route counts within each lane;
  neighboring lane parities start at opposite ends, without using the proxy.

Each greedy objective gets 64 mutation attempts, seed20260906, starting from
the same anchor. A mutation changes 1–3 eligible lanes by rotation, reversal or
single-task insertion. No task migration, width change, phase delay or idle time.
Deduplicate complete executable states. Use all surviving candidates, without
event-score or partial-order pruning; event predictions are diagnostic only.

Median anchor remains the prior score-best. High-skew anchor is the previous
passing `cyclic_rotation_1`, not the older worse anchor. Require exact executable
bridge reconstruction from the prior frozen frontier and original task metadata.

Hardware reuses the unchanged bounded-order runner: CPUs240–319, NUMA3, E256,
H4096/F512, BF16, 2048 tokens, TopK6, SVE N tile8, full-owner-stripe windows0,
ready-token merge, same tensors/allocations per process, four packed copies,
216 MiB scrub before every cell, five warmups and 31 randomized round-paired
measurements. Two independent sessions per trace; fixed tensors, different order
seeds. All plans must be bitwise correct on every copy before timing.

Gate unchanged: candidate paired speedup median>2% and P10>0 in both sessions.
Report all candidates and original anchors. No additional candidates, refit or
second layer after seeing results. Positive performance does not by itself prove
that the intended memory-pressure mechanism caused it; no PMU collection is
included in this initial time-only screen.

## Frozen proxy changes

Global CV-squared (dimensionless), not measured pressure:

| Trace | anchor | smooth_global | smooth_domain | bursty_control | alternating_routes |
| --- | ---: | ---: | ---: | ---: | ---: |
| median | 1.376 | 0.934 | 0.685 | 2.813 | 1.631 |
| high-skew | 1.033 | 0.471 | 0.511 | 1.346 | 1.000 |

Modeled total bytes and isolated horizon are preserved within floating-point
rounding. Thus the proxy reduction is not achieved by reducing requested work
or stretching the isolated horizon. Whether it predicts benefit remains untested.

## Median measured result

| Candidate | S1 / S2 median ms | Paired speedup S1 / S2 | P10 S1 / S2 |
| --- | ---: | ---: | ---: |
| anchor | 31.8678 / 31.7557 | — | — |
| smooth_global | 34.3265 / 33.2248 | −7.134% / −4.518% | −8.040% / −5.066% |
| smooth_domain | 34.2772 / 33.7227 | −7.093% / −5.801% | −7.731% / −6.960% |
| bursty_control | 34.8141 / 34.7181 | −8.499% / −8.566% | −8.822% / −9.039% |
| alternating_routes | 35.0363 / 34.7777 | −9.077% / −8.591% | −9.355% / −9.108% |

No candidate passes. In particular, smoothing the isolated offered-load proxy
does not improve this trace despite invariant modeled total bytes/horizon.
This does not establish the actual hardware pressure was smoother; the proxy
uses isolated phase times, not the contention-shifted execution timeline.

## High-skew measured result

| Candidate | S1 / S2 median ms | Paired speedup S1 / S2 | P10 S1 / S2 |
| --- | ---: | ---: | ---: |
| anchor (prior rotation winner) | 31.3849 / 31.4879 | — | — |
| smooth_global | 31.1573 / 31.4585 | +0.777% / +0.038% | −0.241% / −0.290% |
| smooth_domain | 31.4287 / 31.6313 | −0.209% / −0.356% | −0.657% / −0.824% |
| bursty_control | 31.1493 / 31.0705 | +0.738% / +1.320% | +0.172% / +1.018% |
| alternating_routes | 31.0042 / 30.8600 | +1.037% / +1.960% | +0.414% / +1.657% |

These are paired `anchor/candidate-1` speedups, not ratios of separately
reported medians. The small positive alternating result should not be described
as zero benefit merely because it misses the actionability margin. Conversely,
it is not a >2% stable winner and is not evidence of the intended pressure cause.
Global CV-squared for alternating barely changes (1.033 to 1.000), while the
bursty control increases it to 1.346 and still improves in these sessions.

All five plans on each trace passed bitwise output comparison on all four copies
in both independent sessions. Analyses checked complete 36x5 cells (5 warmup,
31 measured), matching frontier/runner/extension identities and copy pairing.
The unchanged extension SHA256 is
`dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`.
Observed session differences are retained, not filtered. Scrub/copy rotation
does not establish all-DRAM access; no actual allocation-backing/PMU audit was
performed in this time-only screen. Sixteen- and one-thread task membership
remains fixed, so the result does not evaluate reallocating cores or tasks
between large/small classes.

## Reproduction and retained evidence

Generator: `optimizations/fused_moe_sve/benchmarks/generate_pressure_balanced_orders.py`.
Local ignored artifacts: `tmp/pressure_balanced_orders_20260906/`.
Remote: `Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/pressure_balanced_orders_20260906/`.
Final frozen inputs are `median_frozen.json` and `high_skew_frozen.json` (the
earlier `_frontier.json` files are preliminary local-only outputs before import
placement cleanup; they are not measured). Each final input records generator
identity, full executable bridges, objective metrics and accepted-move counts.

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/generate_pressure_balanced_orders.py \
  --source-plans tmp/selection_pair_20260906/plans.json \
  --anchor-frontier tmp/bounded_order_20260906/median_frontier.json --anchor-key anchor \
  --calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --output tmp/pressure_balanced_orders_20260906/median_frozen.json
```

High-skew uses `tmp/selection_other_traces_20260906/high_skew_plans.json`,
`tmp/bounded_order_20260906/high_skew_frontier.json`, key `cyclic_rotation_1`,
and output `high_skew_frozen.json`.

Reuse `bench_bounded_order_extension.py measure` with the new frozen inputs,
the corresponding request016/request008 route files and seeds20260906/20260907;
all environment/NUMA arguments match the bounded-order report. Analyze with
`analyze_bounded_order_extension.py --frontier <trace>_frozen.json --sessions
<trace>_session1.json <trace>_session2.json --output <trace>_summary.json`.

Focused generator tests: three passed (staggering reduces second moment at
constant work/horizon; order/membership invariants; invalid duration rejection).
Combined with the reused state and paired-analyzer checks: 14 tests passed.
Ruff, source-identity/invariant checks, manifest parsing and diff checks passed.
Final raw inputs/results are `<trace>_frozen.json`, `<trace>_session1.json`,
`<trace>_session2.json`, and `<trace>_summary.json`. Summary evidence fields retain
SHA256 for both raw sessions. No production change, refit or commit was made.
