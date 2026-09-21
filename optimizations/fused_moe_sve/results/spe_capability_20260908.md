# ARM SPE capability andattribution gate

Follow-up completed: [control, exact-IP and profiling-overhead audit](spe_control_audit_20260909.md).
It replaces the native_cells_v4 protocol for new measurements; this file retains
the historical failures and capability evidence.

Class E capability measurement,not aMoE performance result. Targethasarm_spe_0,
min_interval256,count_size12,ernd1,arch_inst0. Installedperf5.10.0-216 can
record task-scoped SPE. No system-wide recording orphysical-address request.

Executed onArm-codex-internal from `/home/zhangxu/codex/fused_cpp`:

```sh
numactl --physcpubind=304 --membind=3 perf record \
  -o tmp/spe_capability_20260908/task_probe.data \
  -e arm_spe_0/load_filter=1,jitter=1/ -c 1024 -- \
  .venv/bin/python -c 'a=bytearray(8388608); print(sum(a))'
```

The testprocess completed correctly(sum0); perf recorded8.905MB. It includes
Python/runtime/startup andpotential kernel-context samples,not onlybuffer reads.
No latency orthroughput inference ismade fromthiscapability workload.

`perf report -D` decodes LAT TOT,ISSUE,XLAT packets. Examples showTOT689,
ISSUE594,XLAT1 plusan unnamedcounter-index6 field91. These are rawdecoder
values,not verified nanoseconds oradditive stage costs. Preserve theunknown
field;donotname itmemory latency withoutdocumentation. `perf script` warns
CONTEXT packets absent andTID matching maybeinaccurate. Rawdump also reports
TIME_CONV unhandled inthisversion. Therefore default synthesized thread/time
reports arenot yettrustworthy forcell attribution.

Next implement bounded recording attached totheknown nativeforeground TID,
disabled duringPREP/scrub andenabled onlyaround aGO cell,withcontrol ACKs.
Useuser-mode filtering;keep rawAUX,record identity,addresses andcell boundaries.
Validate samplepacket framing,unknown fields,loss/overflow,IP/data ranges and
timestamp conversion beforecombining cells. Prefer one-cell files initially
toavoid relying onunvalidated TIME_CONV/TID synthesis. Then compare sampled
versusunprofiled timing andperiod sensitivity. No kernel/model change yet.

This isprogress towardindependent foreground-service observation,butneither
thecapability trace norinterface availability completes thephysicalmodel goal.

## Native gated capability runner

`bench_spe_cell.py` launches theoriginalfixed50 nativebinary withno core/uncore
counters,finds theunique taskwhose allowedCPU list isexactly304,andattaches
user-only perf SPE tothatTID. Recording startsdisabled;disableACK establishes
readiness. Eachcell finishesPREP/scrub beforeenableACK,thenGO,then disableACK.
Oneperf file percell avoids cross-cell timestamp assumptions. Thisstill includes
foreground synchronization andidle code aroundkernel execution; IP/data-range
filtering remains required beforekernel attribution. No claim thatgating alone
produces a purekernel trace.

Record actualPREP-to-GO gap:perfcontrol changes theoriginal5ms lead-in protocol.
Do notcompare these single capability timings withthe31-round performance data.
Initialscope isB-only isolatedand38/12,onecell each,sameprocess allocations.
Files areexclusive-create; recorder/native cleanup isbounded toowned processes.
Control-pipe unit tests andRuff gate precede target invocation.

## Native capability attempts and historical authorization blocker

Attemptnative_cells failed beforecapture:eventmodifier `/:u` wasrejected by
installedperf. Corrected to`/u` fornative_cells_v2. Thatattempt's perf logshows
"Events disabled" butrunner rejected/missed theACK; noGO cellwasrecorded.
Allowned processes exited andfailurelogs remain. This isnot aSPE model result.

Localrunner now accepts optionalNUL termination inACK replies anddistinguishes
timeout fromunexpected replybytes. This isnotyet verified as theremote failure's
cause. Threecontrol protocol tests pass locally. Sync ofthisdiagnostic revision
wasblocked byautomatic securityreview over source transfer toArm-codex-internal;
noalternate channel wasused andnative_cells_v3 wasnot launched. Explicit user
approval ofthetarget/source sync isneeded beforethat remote continuation.

Localfollow-up inspected [perf evlist control implementation](https://android.googlesource.com/kernel/common/%2B/a0497251f2b055a137d62ed065286ba999647b3c/tools/perf/util/evlist.c):
`evlist__ctlfd_ack` writes `sizeof(EVLIST_CTL_CMD_ACK_TAG)` bytes,which includes
aC string'sterminating NUL. This supports accepting NUL-terminated replies;
it doesnotprove theexact installed openEuler revision oractual failed reply.
Regression nowcovers ACK withnewline,NUL,both,andinvalid data (4 cases).
Remote sync remainsawaiting explicitapproval;automatic goalcontinuation was
not treated assuchapproval. No newremote record ormodel fit inthisfollow-up.

## Authorized continuation: native_cells_v4

The user subsequently explicitly authorized the script transfer and experiment.
Initial SSH retries timed out; a later connectivity check and SCP completed.
`native_cells_v3` passed the ACK gate and captured the isolated cell, but the
runner rejected return code -2 after its own SIGINT. The perf log reported a
completed write and `perf report --header-only` read the file successfully.
The runner now allows SIGINT termination only when it requested the stop;
other failures remain errors. This is a termination check, not a data-quality gate.

Executed from the same remote project root:

```sh
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/spe_capability_20260908/bench_spe_cell.py \
  --binary tmp/fixed_total_placement_20260908/phase_supply_native \
  --output-dir tmp/spe_capability_20260908/native_cells_v4 \
  --conditions 0 950 --probe 1 --period 1024
```

Both B-only capability cells completed in PID 780720, foreground TID 780721,
CPU 304. Native binary SHA256:
`8a92c487b0e10b6270daaae9f3039c7d480f4b22e9fd3175dfb2191241bb24cd`.
Metadata is in the output directory's `cells.json`; previous attempts remain.

| Condition | Additional PREP-to-GO delay | Recorded kernel time | perf capture size |
| --- | ---: | ---: | ---: |
| isolated | 5.572120 ms | 368.450 us | 0.024 MB |
| 38 same-LLC / 12 other-LLC backgrounds | 207.694280 ms | 1177.740 us | 0.016 MB |

These are single capability observations, **not a performance comparison**.
The control ACK substantially and unequally extends background lead-in beyond
the standard protocol. Both samples passed the load/compute-skeleton scope
checks; B-only has `numerical_checked=false`, not GEMM numerical equivalence.

Raw `perf report -D` for condition 950 identifies AUX CPU 304 / TID 780721,
with EL0 PC and TOT/ISSUE/XLAT packets. PCs include the native text (0x407230)
and an anonymous executable mapping (for example 0xfffec39e20cc), confirming
that the gated file still mixes code locations. This is not yet exact JIT
instruction attribution. TIME_CONV remains unhandled; unknown latency index 6
is retained without interpreting it as memory latency or nanoseconds.

Next: eliminate or control the asymmetric recorder-control delay, identify
the exact probe instruction range, and validate AUX loss/completeness,
sampling-period sensitivity and profiling overhead before formal comparisons.
No model parameters or production code were changed.

Local validation: `.venv/bin/pytest -q tests/test_moe_spe_cell.py` (10 passed),
targeted `ruff check` (passed), and `git diff --check` (passed). Review scope:
only Lab recorder termination, tests and this experiment note; no commit.
