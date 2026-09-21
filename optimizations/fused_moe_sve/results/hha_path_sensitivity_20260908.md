# HHA request-path sensitivity

## Completed result

Both30-cell5/31 sessions completed andpassed original native numerical,placement,
PMU andraw-reader gates. HHA receive events reproducibly distinguish placement
atnearly equal totalDDR throughput. They arenot astandalone waiting-time model.

| Background | S1/S2 foreground us | S1 group25/group27 rx_sccl Gevents/s | S2 |
| --- | --- | --- | --- |
| isolated | 306.15/302.71 | 0.0013/0.0597 | 0.0016/0.0652 |
| balanced48 | 939.12/936.20 | 1.0947/1.1095 | 1.0918/1.1057 |
| balanced50 | 1105.31/1095.64 | 1.1009/1.1093 | 1.1030/1.1146 |
| 38/12 | 1281.68/1268.60 | 0.5320/1.7045 | 0.5303/1.7040 |
| 12/38 | 807.90/762.42 | 1.6913/0.5650 | 1.6844/0.5651 |
| balanced64 | 1686.26/1685.50 | 1.1329/1.1409 | 1.1334/1.1379 |

Fixed50 DDR rates remain279–280GB/s inboth sessions. Theforeground distribution
effect survives changed uncore grouping. All measured hha_retry rawcounts are
zero acrossboth sessions,not just theirmedians. Thismeans theeventprovides no
varying explanatory feature here; it doesnotprove absence ofbackpressure or
waiting elsewhere,or validate retry-event sensitivity.

Balanced50 to64 hasonlysmall receive-rate changes butforeground time increases
~53%. HHA throughput alone cannot explain that response. Also group27'sshare
ofthetwo rx_sccl rates isabout50%,76%,25% for25/25,38/12,12/38,close tothe
foreground-side threadfractions50%,76%,24%. Thus receive share maylargely track
source distribution alreadyvisible fromplacement,not independently measured
queue occupancy orservice latency. No extra fitted feature isadopted.

## Event semantics andremaining attribution gap

The [openEuler Hip09 perf-support record](https://gitee.com/openeuler/kernel/issues/I63VF5)
defines rx_sccl asoperations received fromanother SCCL withinthesame socket,
rx_outer asfromanother socket,andrx_ops_num asallreceived operations. Therefore
do notlabel anHHA group's rx_sccl directly asitsown local-core injection or
foreground waiting. Thegroup receiving a request andtherequest's source differ.
Source mapping stillneeds one-sided load controls andmatching target semantics.

Read-only sysfs discovery foundonlycycles/dat_access/l3c_hit/l3c_ref onL3C,
andcycles/rx_data/rx_req/tx_data/tx_req onSLLC,not thetime-sum aliases listed
intheperf-support record. Availability ofbasic counters doesnotestablish access
tounexposed latency events. No undocumented rawevent wasprogrammed.

Next: validate receive-source mapping withisolated/same-only/other-only controls,
then distinguish observable throughput/distribution fromunobserved residency.
Do notfit aqueue delay fromreceive count alone. Thetheoretical small-M goal
remains open beyond thismeasurement capability result.

Class E Lab measurement,not aphysical model term. Reuse originalfixed50 binary
`tmp/fixed_total_placement_20260908/phase_supply_native` andgrid:isolated,
balanced48/50/64,38/12,12/38;real/B/AB/full/empty (30 cells).
Change onlyuncore selection:40 HHA events replace40 L3C,retain48 DDRC and
originalfive core events. Eight HHA devices eachmeasure rx_ops_num,rx_outer,
rx_sccl,hha_retry,cycles usingtarget sysfs eventcodes,not inferred rawcodes.

Event count stays88 butgroup count changes,so enabled windows andtiming
comparability mustbechecked. Rates aresum(count/enabled_ns) inGevents/s;
retry events arenot bytes,nanoseconds oradditive stalls. Receive aliases do
notbythemselves prove requester locality orforeground attribution. Look for
sensitivity acrossisolated,empty andbackground placements beforeusing anyfeature.

Same NUMA3,foreground304/controller240,M1 W13K4096/N1024 andindependentM12
backgrounds,4-copy rotation,256MiB scrub perLLC,persistent output,5ms lead-in.
Run full-PMU smoke then two5/31 sessions439808/449808 innew
`tmp/hha_path_sensitivity_20260908/` locally/remotely. No native rebuild,
production change,profile export orcalibration fit. Rawreaders validateall
events andoriginal runtime/numerical/placement gates.

Status: runner/reader extension prepared;target smoke andformal results pending.

## Validation checkpoint

Original native SHA2568a92c487b0e10b6270daaae9f3039c7d480f4b22e9fd3175dfb2191241bb24cd
verified.30-cell full-PMU/numerical smoke passed;broader relevant tests94 passed.
Smoke rx_sccl rates showplacement sensitivity:38/12 givesgroup25/27 roughly
0.527/1.704Gevents/s,12/38 gives1.692/0.561. This isnotyet aphysical source-
attribution proof. rx_ops_num remainsroughly2.2–2.4 pergroup forloaded cases;
retry iszero insmoke,which doesnotprove absence ofcontention. All raw events
mustberetained. Twoformal5/31 sessions439808/449808 started;results pending.

Use existing `analyze_fixed_total_placement.py` withthe HHA raw paths and
`--frozen-model tmp/layered_supply_model_20260908/frozen_fit.json`;shared
summary includesper-group HHA eventrates. Analysis performsno fit.

## Evidence identities

Formalcommand exited0; complete results supersede pending checkpoint.
S1 SHA256:`709a25bd7e7cd78300f779ed67e885e34d441589b17bfc6c2eea861a2e228f6b`.
S2 SHA256:`f83771be49344ef0edeb93d0bb93dc375e7438b187ac2c27bece25643a8bb2f1`.
Raw andreport.json arecomplete locally; frozen model identity checked byanalysis.
