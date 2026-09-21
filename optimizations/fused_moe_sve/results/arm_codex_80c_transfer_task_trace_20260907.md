# Actual task timelines for same-width transfers

Geometry correction: Ntile8 below should read Ntile16, as reported by frozen v8
and the actual packed-weight object. Executed geometry and timings are unchanged.

Completed 2026-09-07: two independent traced sessions per frozen seven-plan
frontier. Existing kernel tracing only; no rebuild, new kernel hooks, model
changes or candidate regeneration. All four sessions pass zero-tolerance output
checks. All281 calls/session (29 correctness calls plus252 round calls) have
complete task phases, matching expert/worker identity and call ordering.

## Findings

Times below are medians in milliseconds relative to scheduled-compute start.
Task start is earliest worker gather start; finish is latest worker W2 finish.
This is an execution envelope, not a measured scheduler-dependency release time.
Worker barriers/publication follow W2 and are not individually timed.

### Median: neither moved lane normally determines the compute tail

| Task/plan | S1 start–end | S2 start–end |
| --- | --- | --- |
| M1205 expert85 anchor | 3.732–27.089 | 3.842–27.147 |
| M1205 expert85 swap | 5.989–29.271 | 5.873–29.119 |
| M714 expert115 anchor | 15.376–29.259 | 15.330–29.169 |
| M714 expert115 swap | 13.305–27.153 | 13.340–27.121 |

The recipient lane (logical core_begin16,8T) initially has2.938/2.797ms
of compute slack; after the swap it still has0.552/0.685ms. The donor lane
(core_begin32,8T) changes from0.785/0.808ms slack to2.698/2.694ms. Slack is the
median per-call gap to last W2, not the difference of separately aggregated medians.
It is not proof those workers are idle: ready-token merge may use them.

Anchor last-W2 expert counts across62 rounds: expert166=43, expert42=11,
expert167=8. Candidate: expert166=37, expert42=16, expert167=7, expert85=2.
Expert115 is never last. Thus the isolated critical-lane switch is normally not
an actual compute-critical-lane switch. Recipient work uses existing slack.

Global compute finish changes30.033→29.825ms /29.965→29.878ms. Other lanes
move slightly earlier too: expert166 starts23.903→23.707ms /23.887→23.724ms.
This supports a small indirect timing effect, but does not identify bandwidth,
cache, synchronization or merge as its physical cause.

### High-skew swap: recipient suffix becomes the critical lane

M1945 expert97 start–finish changes2.659–21.655→5.107–24.090ms in S1 and
2.641–21.646→5.074–24.055ms in S2. Its own execution duration is approximately
unchanged; the main change is starting about2.44ms later.

The whole recipient lane (core_begin0,16T) finishes28.514→30.987ms /
28.450→31.184ms. Its final small task, expert19(M3), is last W2 in31/31 rounds
in each session. It is the endpoint of the delayed chain, not an assertion that
this small task alone causes the slowdown.

Anchor's actual compute tail is expert63 on1T core_begin69 in62/62 rounds,
not the isolated critical16T lane. The swap converts a recipient with only
0.834/0.517ms slack into the actual critical lane.

### High-skew relocation: moving M164 to head delays the whole chain

| Task | S1 anchor start–end | S1 relocation start–end |
| --- | --- | --- |
| moved expert25 M164 | 11.879–13.708 | 0.099–4.438 |
| expert251 M1791 | 0.105–18.204 | 4.447–22.140 |
| expert163 M1 | 18.222–18.363 | 22.165–22.320 |
| expert250 M1 | 18.368–18.507 | 22.323–22.473 |
| expert172 M578 | 18.511–24.181 | 22.476–28.210 |
| expert220 M215 | 24.201–26.540 | 28.345–30.772 |
| expert124 M1 | 26.551–26.686 | 30.847–30.962 |

S2 repeats: M164 at head0.082–4.467ms versus11.872–13.673ms on donor;
M1791 starts0.080→4.476ms, recipient lane ends26.701→31.217ms.
Expert124 is last W2 in62/62 rounds. M1791 itself is not last; the whole chain
inherits the delay. This is much larger than the frozen isolated1.476ms cost of
M164. Phase/resource attribution for this context-dependent task expansion is
not established by the envelope alone.

### Donor completion buys slack, not a shorter global finish

The same donor (core_begin48,16T) carries M1239 expert171 as its terminal task.

| Plan | Donor finish S1/S2 | Donor compute slack S1/S2 | Global compute finish S1/S2 |
| --- | --- | --- | --- |
| anchor | 26.762 /26.858 | 2.551 /2.085 | 29.295 /28.925 |
| swap | 25.169 /25.240 | 5.837 /5.956 | 30.987 /31.184 |
| relocation | 25.143 /25.187 | 5.819 /6.050 | 30.962 /31.217 |

Donor improves about1.6–1.7ms but was not critical. Recipient loses its slack
and becomes critical. This directly explains why donor acceleration does not
compensate for recipient delay in these two plans.

## Instrumentation caveat

Use native MOE_CALL e2e, not Python wall time including text formatting/file I/O.
Traced paired native speedups remain positive for median swap(+0.358/+0.204%)
and negative for high-skew swap(-5.589/-3.143%) and relocation(-5.774/-3.253%).
Magnitudes differ from untraced gates; do not replace the original performance
evidence. High-skew anchor post-last-W2 scheduled remainder is0.347/1.669ms;
swap0.429/0.450ms, relocation0.502/0.505ms. Therefore compute-tail identity is
stable, but the residual end-of-call interval varies significantly by session.
It includes merge, drain, synchronization and scheduling; this trace does not
identify an individual final merge owner or measure true CPU idle residency.

## Protocol, source and artifacts

Reuse `bench_bounded_order_extension.py` from
`tmp/same_width_transfer_20260907/` and its exact frozen frontiers. Same NUMA3
CPUs240–319,4-copy rotation,216MiB scrub,randomized plans,seeds20260907/08,
5 warmups+31 effective rounds. E256,H4096,F512,2048tokens,TopK6,BF16,Ntile8,
W13/W2 per expert8MiB/4MiB,full stripes `(t,0,0,1,1)`,early merge unchanged.
Extension `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`,
v8 calibration `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`.
Existing boot/kernel/page-policy evidence is in the same-width transfer report.
No competing benchmark found at start; no PMU or allocation-backing audit.

Measurement command is the transfer report command with
`FUSED_CPP_MOE_TRACE=1` and
`FUSED_CPP_MOE_TRACE_FILE=/home/zhangxu/codex/fused_cpp/tmp/same_width_task_trace_20260907/<trace>_session<n>.log`,
and output JSON in the same new directory. Raw logs and session JSON remain on
`Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/same_width_task_trace_20260907/`.
Local same-named directory has all four session JSONs, all four full analysis
JSONs (including per-round envelopes), and median session1 raw log.
The accidentally misdirected local traced session1 was moved to this directory;
original untraced session1 was restored from remote and its original summary
SHA256 verified. Original performance evidence is intact.

Reproduce analysis per session:

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_transfer_task_trace.py \
  --frontier tmp/same_width_transfer_20260907/median_frozen.json \
  --session tmp/same_width_task_trace_20260907/median_session1.json \
  --trace tmp/same_width_task_trace_20260907/median_session1.log \
  --output tmp/same_width_task_trace_20260907/median_reanalysis1.json
```

Focused parser tests:2 passed; Ruff and diff checks pass. Analysis ran on all four
complete logs. No production or cost-model changes and no commit. Next analysis
may decompose M164 head versus donor phase durations from these same raw logs;
no further hardware experiment is needed merely to inspect those phases.
