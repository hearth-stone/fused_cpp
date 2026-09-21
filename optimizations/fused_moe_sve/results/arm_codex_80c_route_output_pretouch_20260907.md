# Route-output pretouch control

Geometry correction: Ntile8 below should read Ntile16, verified from the frozen
calibration and actual packed-weight object. Runtime/results are unchanged.

Completed2026-09-07. Pretouch almost eliminates the M164 W2 placement-dependent
expansion, with faults shifted out of execution into pretouch. However, serial
full-buffer zeroing costs9.1–9.3ms and regresses inclusive native call latency.
Do not enable per-call zeroing or fit the old W2 expansion as weight-contention.
Workspace reuse is a justified next experiment, not implemented or adopted here.

## Outcome

All numbers are31-sample medians, session1/session2, in milliseconds unless noted.

| Plan | W2 without touch | W2 with touch | Pretouch cost | Native inclusive without | Native inclusive with |
| --- | ---: | ---: | ---: | ---: | ---: |
| anchor | 0.559/0.546 | 0.537/0.536 | 9.344/9.285 | 30.336/30.254 | 38.814/38.754 |
| cross-LLC swap | 2.106/2.190 | 0.519/0.521 | 9.115/9.105 | 32.809/32.632 | 39.687/39.555 |
| cross-LLC relocation/head | 2.804/2.876 | 0.539/0.542 | 9.114/9.281 | 31.996/31.623 | 38.847/38.762 |

Paired inclusive speedup `100*(no_touch/touch-1)` is -21.916/-21.982% for
anchor, -17.239/-17.582% for swap, and -17.450/-18.448% for relocation.
Clearing is not a performance win, despite the much faster target W2.

Subtracting measured touch time per call gives diagnostic remainder medians:
anchor29.537/29.597ms, swap30.389/30.368ms, head29.544/29.499ms. This is not
workspace-reuse performance: zeroing changes cache state and placement, and
removing its duration cannot simulate a different allocation lifecycle.

## Fault accounting

Process-wide `getrusage(RUSAGE_SELF)` minor faults; these are not W2-scoped PMU
counters or a direct measurement of THP allocation events.

| Plan | No-touch whole-call minor faults | Touch-stage minor faults | Outside-touch minor faults |
| --- | ---: | ---: | ---: |
| anchor | 344/343 | 96/607 | 1/1 |
| swap | 1714/512 | 96/96 | 1/1 |
| relocation/head | 1872/675 | 96/607 | 1/1 |

Major-fault medians are0 in every cell. Fault counts vary with allocation/page
state; do not fit a fixed count or infer exact THP backing from96 versus607.
The combination of near-complete W2-gap collapse and fault relocation strongly
supports output first-write/page-establishment costs as the dominant source of
the observed head penalty. It does not uniquely separate page allocation,
zero-fill, concurrent fault handling and cache/coherence effects. Actual output
mapping alignment/backing and per-phase faults remain unmeasured.

The intervention targets route output, not weights. Prior scrub and four-copy
weight rotation did not remove this confound. Pure resource-overlap correlations
from the earlier traces should not be interpreted as causal DDR contention.

## Implementation and protocol

Lab `bench_route_out_pretouch.py` uses process-local `TorchDispatchMode` around
each forward. Both control and treatment pass through the same dispatcher;
only treatment zeroes the exact CPU FP32 `aten.empty` result of shape
`[12288,4096]` (192MiB). Each call must hit exactly one target allocation or abort.
No allocator replacement, C++ edits, extension rebuild, production environment
parser, public API/schema/default changes or workspace reuse. The mode is removed
after each call. Zeroing uses Torch intra-op1; no extra parallel pretouch policy.

Frozen high-skew request008/layer38, E256/H4096/F512,2048tokens,TopK6,BF16,
4x16T+16x1T,NUMA3 CPUs240–319/membind3,Ntile8,full stripes `(t,0,0,1,1)`;
W13/W2 bytes per expert8MiB/4MiB. Same seeded inputs and four weight allocations
as preceding experiments,216MiB scrub before each call. Three exact plans from
`tmp/same_width_transfer_20260907/high_skew_frozen.json`; no regeneration/refit.
Two independent processes, seeds20260907/08,36 rounds (5 warmups,31 measured).
Each round randomizes all3 plans x2 modes, same copy `round%4`. All3x2x4 outputs
match anchor with zero tolerance before timing. Smoke is separate, not pooled.

Native MOE_CALL e2e includes pretouch and excludes final text-log serialization;
Python wall-clock includes serialization and is saved but not used for headline
latency. W2 envelope spans earliest worker start to latest worker finish, requiring
all16 distinct worker records. Both modes trace identically. Complete logs are
retained, not deleted after per-call parsing.

Frozen extension SHA256:
`dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`.
Calibration remains v8:
`7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`.
Initial machine load0.06/0.04/0.17, no other benchmark found. Existing THP policy
`always` was not changed. Source runner hashes are in both session records;
worktree remains uncommitted with unrelated user changes preserved.

## Reproduction and evidence

Remote directory:
`Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/route_output_pretouch_20260907/`.
Contains runner, separate `smoke/`, `session1/`, `session2/`, all per-call logs
and session JSON. Local same-named directory contains session1/session.json,
session2/session.json and validated summary.json. No large logs staged.

From remote project root, for session1 (repeat with seed20260908/session2):

```bash
env OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:src \
  timeout 300 numactl --physcpubind=240-319 --membind=3 \
  .venv/bin/python tmp/route_output_pretouch_20260907/bench_route_out_pretouch.py \
  --frontier tmp/same_width_transfer_20260907/high_skew_frozen.json \
  --route-file bench_assets/moe_paper/dsv4_routes_pt_20260830/measured_request008_case009_zh2048-010.pt \
  --seed 20260907 --output-dir tmp/route_output_pretouch_20260907/session1
```

Analyze locally with `analyze_route_out_pretouch.py --sessions
tmp/route_output_pretouch_20260907/session1/session.json
tmp/route_output_pretouch_20260907/session2/session.json --output <fresh-summary>`.
Analyzer requires complete216 cells/session, warmup/copy identity, one allocation
hit, matching extension/frontier/runner, independent seeds and correct outputs.

Focused tests: `PYTHONPATH=.:src .venv/bin/python -m pytest -q
tests/test_moe_route_out_pretouch.py`:4 passed. Ruff and diff checks pass.
Target smoke and two full hardware sessions completed. No commit made.

## Decision

Reject per-call serial192MiB zeroing as an optimization. Preserve it as a bounded
Lab diagnostic reference for testing a future workspace-lifecycle candidate.
The next candidate should amortize allocation/first-touch outside repeated calls,
preserve concurrent-call ownership and full overwrite-before-read correctness,
and report both initialization cost and steady-state latency. Do not infer its
speedup by subtracting the9ms here; measure it independently if requested.
