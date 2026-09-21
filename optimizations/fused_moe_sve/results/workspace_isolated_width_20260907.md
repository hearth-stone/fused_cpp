# Workspace isolated-width phase audit

## Outcome

Four sessions completed; no parameters fitted. The evidence separates three
problems: isolated GEMM phase underprediction, cancellation by legacy operator
residuals, and inaccurate workload-dependent phase dilation. Large-M16T GEMMs
receive excessive modeled dilation, while selected small-M1T tasks have real
context increments that are underpredicted or absent. A uniform reduction of
all contention penalties would not fix both populations.

## Protocol and scope

Feature: `measurement.workspace_isolated_width`, Lab diagnostic only. HEAD
`c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5` plus existing dirty tree. No model,
planner, native kernel, workspace helper or production default changed.

Arm-codex-internal, remote root `/home/zhangxu/codex/fused_cpp`, NUMA3,
CPU240–319. Fixed parent profile `profiles/workspace_numa3_80c.json`, actual
Ntile16, E256/H4096/F512,2048 tokens,TopK6,BF16,merge on. W13/W2 full weight
bytes per expert8MiB/4MiB, full owner stripes `(t,0,0,1,1)`; target widths1/8/16,
remaining full-plan widths unchanged. Four weight copies,216MiB scrub before
each call,5 warmup rounds,31 paired rounds,2 independent seeds20260913/20260914.
Same allocations, weights, hidden states and routes within each session.
Diagnostic trace is enabled and cannot authorize winners/pruning.

Extension SHA: `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`.
Calibration SHA: `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`.
All four sessions match profile lifecycle, helper identity, capacity and backend.
Model source hash matches the previous pair-feature audit; frozen placed phase
predictions come from the unchanged fixed-plan timeline artifacts.

Isolation keeps the full real-route workload and all weight/output addresses:
move one target task to the head and make every other task depend on its
completion, preserving remaining dependencies. No other expert computes during
the target. The80-worker runtime remains present; this is not a one-thread
process or an absence of scheduler activity. Unmeasured reference correctness
uses ordinary output allocation; tested/timed cells use the fixed pre-touched
workspace. All plans/copies pass bit-exact output validation after NaN poisoning.

Median: expert115(M714), widths1/8/16, plus the three fixed full plans.
High-skew: experts231(M1),63(M10),241(M62),135(M1341), widths1/16,
plus the four fixed full plans. Both route layers are unchanged from the prior
phase-timeline experiment (median layer4, high-skew layer38).

All1444 calls (including correctness and warmup) have complete task/worker
traces. Every isolated call passes the background-start-after-target-W2 check,
with only one native timestamp rounding unit (1e-5ms) tolerated. Measured sample
count is1116 plan calls:18 cells×31×2.

Placement caveat: median isolated8T uses logical begin16, whereas old-anchor/
smooth M714 uses begin32 (same LLC domain, different cores). High-skew small
experts expanded to16T use begin64 rather than their full-greedy placements.
These are recorded as `exact_placement=false` and excluded from strict matched
context conclusions below. M714/16T, M1341/16T and the1T small-expert controls
have exact physical placement matches for the cited comparisons.

## Isolated stages

Values are stage-envelope medians in ms, session1/session2; the same native
gather/W13/W2 boundaries are used in isolated and full calls. W13/W2 envelopes
can include synchronization within their boundaries. Gather is not a pure DDR
service measurement. Model error is `(prediction/hardware - 1)`.

| M | Width | Gather hardware | W13 model | W13 hardware | W2 model | W2 hardware |
| ---: | ---: | --- | ---: | --- | ---: | --- |
| 714 | 1 | 2.154/2.153 | 67.497 | 71.661/71.678 | 33.748 | 34.297/34.387 |
| 714 | 8 | 0.413/0.406 | 8.537 | 8.984/8.984 | 4.268 | 4.318/4.323 |
| 714 | 16 | 0.242/0.247 | 4.285 | 4.537/4.538 | 2.143 | 2.178/2.176 |
| 1 | 1 | 0.0077/0.0076 | 0.2435 | 0.3125/0.3202 | 0.1217 | 0.1510/0.1531 |
| 1 | 16 | 0.0520/0.0647 | 0.0505 | 0.0729/0.0735 | 0.0253 | 0.0330/0.0335 |
| 10 | 1 | 0.0282/0.0289 | 0.9453 | 1.0405/1.0450 | 0.4727 | 0.5307/0.5325 |
| 10 | 16 | 0.0522/0.0597 | 0.0600 | 0.0820/0.0839 | 0.0300 | 0.0408/0.0412 |
| 62 | 1 | 0.1875/0.1943 | 5.8611 | 6.3561/6.3637 | 2.9305 | 3.0306/3.0342 |
| 62 | 16 | 0.0626/0.0762 | 0.3721 | 0.4676/0.4772 | 0.1860 | 0.2105/0.2109 |
| 1341 | 1 | 4.072/4.094 | 126.863 | 134.599/134.697 | 63.432 | 64.553/64.512 |
| 1341 | 16 | 0.373/0.384 | 8.054 | 8.442/8.436 | 4.027 | 4.069/4.071 |

Large-M W13 is underpredicted by about4.5–5.8%; W2 by about1–1.9%.
Tiny-M stage errors are larger: M1/1T W13-22–24%, M1/16T W13 about-31%.
Thus neither width alone nor a single global stage multiplier is established by
this selected grid. There is no explicit gather phase in frozen v8.

### Total accuracy masks stage cancellation

For M714/1T, model total108.870ms versus measured target envelope108.160/108.226ms
looks close. But model W13 is about4.16ms too small, while the legacy `operator`
term is7.625ms and measured gather is only2.154/2.153ms. The operator term is
not a valid gather estimate; it compensates other phase errors.

M1341/1T shows the same pattern: model total204.486ms versus203.277/203.312ms,
legacy operator14.191ms versus gather4.072/4.094ms. For M1/1T, operator0.1585ms
versus gather0.0077/0.0076ms also masks GEMM phase underprediction. These
comparisons do not claim operator residual consists solely of gather.

## Full-workload increments

Hardware increments below are medians of `full_stage - isolated_stage` within
the same session, paired round and weight copy. They are not differences of
two medians. Model increments are frozen placed-stage time minus isolated
phase prediction for the same expert/width. Tables use exact placement matches.

### Large-M16T: modeled GEMM dilation is much too large

| Task/context | Stage | Hardware isolated s1/s2 | Hardware full s1/s2 | Paired increment s1/s2 | Model increment |
| --- | --- | --- | --- | --- | ---: |
| M714 median elite | W13 | 4.537/4.538 | 4.563/4.560 | +0.0259/+0.0192 | +1.8263 |
| M714 median elite | W2 | 2.178/2.176 | 2.194/2.195 | +0.0153/+0.0190 | +0.6794 |
| M1341 high-skew anchor | W13 | 8.442/8.436 | 8.446/8.439 | +0.0057/+0.0074 | +3.6916 |
| M1341 high-skew anchor | W2 | 4.069/4.071 | 4.097/4.094 | +0.0248/+0.0175 | +1.1760 |
| M1341 high-skew full | W13 | 8.442/8.436 | 8.542/8.517 | +0.0974/+0.0862 | +3.9594 |
| M1341 high-skew full | W2 | 4.069/4.071 | 4.081/4.080 | +0.0090/+0.0074 | +2.0366 |

M1341 in greedy/VND likewise has W13 increments about0.050–0.059ms and W2
increments about0.009–0.019ms, versus modeled3.960/1.981ms. The large wide-task
GEMM mismatch is predominantly excessive context dilation in these cases,
not an isolated stage overestimate. This does not yet identify which old
pressure/capacity term is responsible.

Gather can behave differently: M1341 in full_reference rises by0.366/0.415ms
(roughly doubles), while its W13/W2 remain close to isolated. Conversely M714
elite gather falls by0.029/0.038ms. Context increments include cache, input/output
state and synchronization changes; they are not solely DDR queue penalties.

### Small-M1T: substantial real increments remain

| Task/context | Stage | Hardware isolated s1/s2 | Hardware full s1/s2 | Paired increment s1/s2 | Model increment |
| --- | --- | --- | --- | --- | ---: |
| M1 anchor | W13 | 0.3125/0.3202 | 0.8530/0.8365 | +0.5449/+0.5170 | +0.3390 |
| M1 anchor | W2 | 0.1510/0.1531 | 0.4380/0.4358 | +0.2798/+0.2750 | +0.1660 |
| M1 full | W13 | 0.3125/0.3202 | 0.8460/0.8262 | +0.5397/+0.5013 | +0.0920 |
| M1 full | W2 | 0.1510/0.1531 | 0.3956/0.3543 | +0.2374/+0.1975 | +0.0454 |
| M62 anchor | W13 | 6.3561/6.3637 | 7.2595/7.1670 | +0.9033/+0.8010 | 0 |
| M62 anchor | W2 | 3.0306/3.0342 | 3.2554/3.2219 | +0.2180/+0.1805 | 0 |
| M62 full | W13 | 6.3561/6.3637 | 6.9474/6.9430 | +0.5913/+0.5847 | 0 |
| M62 full | W2 | 3.0306/3.0342 | 4.1312/4.1555 | +1.0955/+1.1204 | 0 |

M1 W13 increases about157–176% in these contexts. M62 changes which stage
has the larger increment when task order changes. A single width-only penalty
cannot represent this stage/context distinction. M10 is a counter-control:
full_reference W13 increments only0.0252/0.0234ms and W2 about0.0033/0.0028ms;
anchor increments vary more between sessions. Small M is not by itself a
sufficient condition for a large penalty.

## Decision and remaining limits

1. Keep v8 frozen. Establish phase-specific isolated corrections before touching
   concurrency fitting; do not preserve total accuracy through operator residual
   cancellation.
2. Audit the existing placed-model dilation terms offline against these fixed
   cases. Large-M16T needs much less GEMM dilation, but small-M1T cannot have
   its remaining context cost discarded. No new physics parameter was added.
3. Preserve stage/context distinctions and the M10 low-increment control.
   Exact DDR queue prediction is not required, and these traces do not uniquely
   identify DDR, LLC, gather coupling, merge traffic or scheduler effects.
4. The sample is selected from counterexamples, not an independent holdout.
   No whole-search-domain accuracy, top-K recall, false-pruning guarantee or
   production performance improvement is claimed. Full plans are trace-on
   controls; previous trace-off winner decisions remain authoritative.

## Evidence and reproduction

Local/remote relative root: `tmp/workspace_isolated_width_20260907/`.
Local: both frozen frontiers, four session JSONs, four `*_compact.json` files,
and `summary.json` (33 isolated phase rows,57 context phase rows, with exact
placement flags). Remote: same files and four full raw `*.log.gz` traces
(about48/48/84/85MiB). Gzip is lossless; session hashes refer to uncompressed
traces checked before compression. No raw trace was discarded. Remote runner
snapshot is in `runner/`; no production mirror source overwritten.

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/prepare_workspace_isolated_width.py \
  --input-dir tmp/workspace_phase_timeline_20260907 \
  --output-dir tmp/workspace_isolated_width_20260907
```

For each trace and session, executed remotely from project root:

```bash
OMP_NUM_THREADS=80 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/workspace_isolated_width_20260907/runner/bench_bounded_order_extension.py measure \
  --repo-root /home/zhangxu/codex/fused_cpp --max-plans 13 \
  --frontier tmp/workspace_isolated_width_20260907/${trace}.json --route-file "$route" \
  --experiment-baseline tmp/workspace_isolated_width_20260907/workspace_numa3_80c.json \
  --phase-trace tmp/workspace_isolated_width_20260907/${trace}_session${session}.log \
  --output tmp/workspace_isolated_width_20260907/${trace}_session${session}.json \
  --seed $((20260912 + session))
```

`trace=median` route:
`bench_assets/moe_paper/dsv4_routes_pt_20260830/measured_request016_case017_zh2048-018.pt`;
`trace=high_skew` route:
`bench_assets/moe_paper/dsv4_routes_pt_20260830/measured_request008_case009_zh2048-010.pt`.
`session=1,2`. Each run is followed by `analyze_workspace_isolated_width.py`
with matching `--frontier --session --trace --output` (compact JSON), then
`gzip -1` of that explicit raw log. Fresh output paths are required.

```bash
PYTHONPATH=.:src .venv/bin/python optimizations/fused_moe_sve/benchmarks/summarize_workspace_isolated_width.py \
  --input-dir tmp/workspace_isolated_width_20260907 \
  --reference-dir tmp/workspace_phase_timeline_20260907 \
  --calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --output tmp/workspace_isolated_width_20260907/reproduction.json
```

17 focused tests passed, including synthetic complete-trace acceptance and
overlap rejection:
`PYTHONPATH=.:src .venv/bin/python -m pytest -q tests/test_moe_workspace_isolated_width.py tests/test_moe_fixed_route_workspace.py tests/test_moe_transfer_task_trace.py`.
Ruff and diff checks pass. Original runtime correctness passed all18 cells×4
copies in each trace's two sessions; all1444 trace calls validated. No remaining
benchmark process, no model fit, no production change, no commit.
