# Fixed max-token route-output workspace

Geometry correction: Ntile8 below should read Ntile16, verified from the frozen
calibration and actual packed-weight object. Runtime/results are unchanged.

Implemented and target-validated as an explicit Lab version. One workspace owns
a fixed CPU FP32 allocation, allocated and zero-touched once at construction.
Capacity overflow is an error; automatic growth is a low-priority TODO. No
production default, Plan V2, native ABI, kernel or cost-model formula changed.

## Interface and ownership

```python
from optimizations.fused_moe_sve.benchmarks.fixed_route_workspace import FixedRouteWorkspace

# Construct after binding the intended rank and NUMA memory policy.
workspace = FixedRouteWorkspace(max_tokens=2048, top_k=6, hidden=4096)

with torch.inference_mode(), workspace.lease(tokens=2048):
    output = fused_moe_bf16_tiled_async_plan(...)  # existing synchronous forward
```

Capacity is `max_tokens * top_k * hidden * 4` bytes:192MiB here. Construction
cost was8.049/8.176ms in the two sessions, recorded separately from steady state.
The caller retains the object across calls. It may share it across sequential
layers with compatible geometry, but never concurrently. One nonblocking lock
rejects concurrent/reentrant leases, and exception unwinding always releases it.
Concurrent forwards need separate workspace objects; this is not an automatic
pool. CPU-rank affinity must match construction; the caller must maintain the
NUMA memory policy. Capacity does not shrink or grow; excess tokens are rejected
before entering the call.

Lab integration uses TorchDispatchMode to replace the exact CPU FP32 route-output
`aten.empty` request with a contiguous prefix view of existing storage. A lease
must see exactly one matching allocation. Other allocations pass through. No
per-call allocation of the192MiB route storage, pretouch or clearing occurs.
The view is valid only as scratch within the synchronous forward; callers must
not retain it or schedule asynchronous users after the lease ends. This adapter
is intentionally not a public supported API or a replacement for explicit native
workspace plumbing in a future production integration.

## Correctness and limits

Local tests cover fixed capacity, smaller prefix reuse at the same pointer,
preservation of previous contents (no hidden clearing), invalid dimensions,
missing/duplicate allocation, exception release and cross-thread concurrent
lease rejection.12 tests pass across workspace and existing pretouch test files.

On target, before every reuse correctness call the buffer is filled with NaN.
All3 plans x2 modes x4 weight copies produce exactly the same output as the
allocation baseline, in both sessions. This verifies overwrite-before-merge for
the tested strict FP32 direct-route workload despite arbitrary old contents.
No poisoning or clearing occurs in timed calls. Variable-token native workloads,
other route dtypes, other execution modes, concurrent native forwards on separate
workspaces and other machines have not been validated; smaller-prefix logic has
local unit coverage only. Zero-token leases are explicitly unsupported.

## Steady-state results

Three frozen high-skew plans, same paired protocol as pretouch, but treatment
now reuses the fixed workspace. Native traced call time includes all per-call
work and excludes initial workspace construction and final trace serialization.
Reported gains are medians of31 paired ratios, not ratios of latency medians.

| Plan | Allocate/call ms S1/S2 | Fixed workspace ms S1/S2 | Paired gain % S1/S2 | Gain P10 % S1/S2 |
| --- | ---: | ---: | ---: | ---: |
| anchor | 30.195/30.248 | 29.287/29.259 | 3.078/3.382 | 2.591/2.784 |
| cross-LLC swap | 32.877/32.853 | 30.330/30.334 | 8.310/8.278 | 7.854/7.545 |
| cross-LLC relocation | 31.737/31.921 | 29.286/29.365 | 8.224/8.719 | 7.844/8.120 |

M164 W2 span: anchor0.534/0.536→0.535/0.538ms,
swap2.177/2.143→0.522/0.522ms, relocation2.846/2.802→0.530/0.532ms.
Whole-call minor-fault medians fall from334/333,491/495,656/658 respectively
to0/0 for all three. Major-fault medians are0. Unlike per-call full zeroing,
there is no repeated9ms touch cost; upfront8ms remains part of initialization.

This establishes benefit on the tested traced workload, not a production-wide
claim. First-call latency, total lifetime cost for very few calls and untraced
production overhead require separate evaluation. Persistent buffer placement and
cache residency differ from newly allocated buffers; no exact per-phase fault or
THP-backing attribution is claimed.

## Reproduction and evidence

Machine Arm-codex-internal, NUMA3 CPUs240–319/membind3. No other benchmark found
at start (load0.01/0.07/0.37). Frozen E256/H4096/F512,2048tokens,TopK6,BF16,
4x16T+16x1T,Ntile8,full stripes `(t,0,0,1,1)`,W13/W2 bytes8MiB/4MiB,
early merge unchanged.4-copy weights,216MiB scrub,5 warmups+31 effective rounds,
3 plans x2 modes randomly interleaved per round, same copy, seeds20260907/08.
Both sessions verify one intercepted route allocation per call. No source rebuild.
Extension `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`;
frozen v8 calibration `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`.

Remote full logs, code and JSON:
`Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/fixed_route_workspace_20260907/`.
Local same-named directory: session1/session.json, session2/session.json,
summary.json. Sessions record runner/helper hashes, treatment, max_tokens,
capacity, initialization time and extension/frontier identity. Existing pretouch
artifacts/code snapshots were not overwritten.

Run from remote project root (repeat for session2 with seed20260908):

```bash
env OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:src \
  timeout 300 numactl --physcpubind=240-319 --membind=3 \
  .venv/bin/python tmp/fixed_route_workspace_20260907/bench_route_out_pretouch.py \
  --frontier tmp/same_width_transfer_20260907/high_skew_frozen.json \
  --route-file bench_assets/moe_paper/dsv4_routes_pt_20260830/measured_request008_case009_zh2048-010.pt \
  --workspace-max-tokens 2048 --seed 20260907 \
  --output-dir tmp/fixed_route_workspace_20260907/session1
```

Analyze with `analyze_route_out_pretouch.py --sessions <session1.json>
<session2.json> --output <fresh-summary.json>`. Compatibility field `touch=true`
means treatment; session-level `treatment=fixed_workspace` disambiguates it from
the original per-call pretouch experiment. Treatment/helper identity must match.

Validation:12 local tests passed; two full target correctness/performance sessions
completed; Ruff, manifest parsing and diff checks pass. No commit created.

## Follow-ups

- Low priority: capacity growth at a safe idle boundary, with explicit cap and
  initialization accounting. No automatic growth in this version.
- Production integration, if requested: explicit owned native workspace input
  or an existing lifecycle-managed pool; preserve default fallback and validate
  supported shapes/paths. Do not silently install a process-global buffer.
