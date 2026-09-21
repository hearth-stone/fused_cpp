# DDRC occupancy time-scale audit

## Completed result

Both85-cell5/31 sessions pass allraw andper-controller clock checks. Theunit
conversion preserves thehigh-pressure response gap:balanced64→78 foreground
timeincreases22.33/20.68%,whilecommand-rate-weighted occupancy-time proxy
increases9.10/10.18%. Awrong CPU-versus-controller clock assumption isnot a
sufficient explanation ofthelarge residual.

| Case | S1 time us/proxy ns | S2 time us/proxy ns |
| --- | --- | --- |
| isolated | 299.00/19.95 | 306.22/19.93 |
| same39 | 756.72/91.22 | 772.13/92.81 |
| other39 | 418.52/96.30 | 424.28/96.30 |
| balanced48 | 941.33/186.54 | 920.38/187.13 |
| balanced64 | 1662.92/273.88 | 1689.76/271.95 |
| balanced78 | 2034.27/298.80 | 2039.27/299.63 |

Same39 andother39 also showtheattribution gap:similar controller proxies but
verydifferent foreground times. This isnot proofthatforegroundlatency grows
byonlytheproxy ratio. It leavesatleasttwo alternatives:foreground-specific
service worsens morethanthecontroller average,oreffective request concurrency/
overlap changes,orboth. Deriving concurrency asrequests*proxy/time would be
conditional ontheunverified proxy equality,not anindependent MLP measurement.

Formalcommand exited0; allraw andreport.json complete locally. RawS1 SHA256:
`d303012ca032982017841b66dbf2b2da55e39eac9758bffb8af313d35b29dbac`;
S2:`b063574c1628561b38ea0b225e6f36b80c64281fe03b31cc995ff5004aad1a63`.
No model coefficients orproduction paths changed.

## Next direct-observation capability

Read-only discovery found `/sys/bus/event_source/devices/arm_spe_0` onthetarget,
withload_filter,event_filter,min_latency,jitter,ts_enable,pa_enable,pct_enable
formats. Installedperf is5.10.0-216.0.0.115.oe2203sp4.aarch64. Availability
doesnotyet prove usable sampling,latency decoding orcorrect timestamp alignment.
Next runa bounded foreground-only SPE capability test,excluding scrub/preparation
andbackground threads. Distinguish sampled total/load-use latency fromDDRC-only
queue delay; record sampling bias andoverhead beforeusing anylatency distribution.

Class E Lab measurement. Currentoccupancy/command ratio hascycle-per-command
eventunits,not ns. TargetDDRC identifier0x30 exposes cycles(config0x0),distinct
fromread_cmd_occupancy0x80 andread_cmd0x41. Read cycles throughsysfs alias;
do notassume CPU clock orDRAM transfer frequency.

Addone cycles event toeach16 DDRC devices:64 DDRC+40 HHA events. Same native
fixed50 binary andoriginal85-cell sourcegrid. Eventgroup count isunchanged,
butcounter reads andper-event intervals maydiffer. Fullpositive enabled/running
checks remain. Two5/31 sessions479808/489808 aftersmoke;NUMA3/workspace/copy/
scrub/lead-in unchanged. Artifacts `tmp/ddrc_clock_scale_20260908/` locally and
under `/home/zhangxu/codex/fused_cpp/` remotely.

Percontroller compute f=cycles/enabled_ns;rate-normalized occupancy/command
divided byf givesan occupancy-time proxy in ns/command. Occupancy-rate divided
bycycle-rate givescycles-weighted occupancy percycle. Neither isautomatically
foreground-specific delay orqueue depth untilsemantics andscope arevalidated.
Unequal gates cannot berepaired completely byrate normalization. Keep allraw
counts,times andoriginal frozen ratios;no newcalibration orformula adoption.

Existing HHA source data audit foundrate normalization alone changes thehigh
queue ratio onlyslightly (~324→326 at64,~355→356 at78),whileDDRC command
gates span~1.04–1.59x foreground duration inthose cells. Thismotivates aunit/
window audit,not aclaim thatnormalization fixes thetime-model residual.

Status: optionalrunner/reader extension prepared;targetclock smoke andformal
measurements pending. Production andfrozen model remain unchanged.

## Smoke checkpoint

85-cell104-uncore-event smoke passed. DDRC cycles/enabled_ns ispositive onall
controllers. Selectedsmoke ranges:isolated1.160–1.192,same391.187–1.194,
balanced781.194–1.197 cycles/ns. Shortwindow apparent rates needoverhead/window
qualification; do notassume exactphysical clock fromthese observations alone.
Twoformal5/31 sessions479808/489808 started. Localtests40 passed;Ruff and
diff checks passed. No model parameters fitted orchanged.

## Analysis andfirst-session checkpoint

`analyze_ddrc_clock.py` retains everyreal-W13 round andper-controller metrics.
Independent enabled windows areused forcommand,occupancy andcycles; global
proxy iscommand-rate-weighted mean ofper-controller ns proxies,then cellmedian.
Unit/invalid-clock tests pass;focused total42 passed. Normalization isnot a
correction forunmatched sampling windows anddoesnot validate occupancy semantics.

Firstcomplete session479808 passesraw andclock checks. Selected cellmedians:
isolated299.00us/19.95ns proxy;same39756.72us/91.22ns;other39418.52us/96.30ns;
balanced48941.33us/186.54ns;balanced641662.92us/273.88ns;balanced782034.27us/
298.80ns. At64/78,controller medianclock rates span1.1931–1.1963 cycles/ns.
Thus the~9% proxy growth versus~22% foreground growth persists aftertime-scale
conversion. This ispreliminary single-session evidence;secondsession pending.

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_ddrc_clock.py \
  --sessions tmp/ddrc_clock_scale_20260908/session1.jsonl \
  tmp/ddrc_clock_scale_20260908/session2.jsonl \
  --output tmp/ddrc_clock_scale_20260908/report.json
```
