# Byte-identical SVE supply code placement

## Result andbounded decision

Both complete52-cell,5/31 sessions pass raw-reader,code-address andbinary checks.
At25/25 and38/12,thefour same-byte copies have smallmedian ranges,not the
large changes seen when scalar load-stream organization changed. Do notadd a
uniform code-address correction for thehigh-pressure distribution residual.
This isnot universal code-placement invariance or aformal equivalence result.

| Background | B-copy median range S1/S2 % | Full-copy median range S1/S2 % |
| --- | --- | --- |
| isolated | 0.25/0.15 | 0.77/0.80 |
| 25/25 | 0.46/0.62 | 1.06/1.10 |
| 38/12 | 0.41/1.21 | 1.24/0.89 |
| 12/38 | 3.35/8.56 | 4.02/4.98 |

Range means100*(max ofthefour cellmedians/min-1),not aselected speedup or
confidence bound. Everyoffset andoriginal-JIT comparison isretained inreport.json.
At25/25/38/12,no offset-versusoffset0 comparison has a95% interval onthe
same nonzero side inboth sessions. Intervals arepointwise; thisdoesnot prove
exactzero or equivalence within1–2%.

12/38 isexplicitly unresolved: median ranges arelarger andranking changes.
ForB,S1 offsets0/32/64/96 give564.63/568.97/550.53/568.49us; S2
536.83/575.17/529.80/536.62us. FullS1=738.96/716.53/719.35/745.30us;
S2=729.44/712.30/747.78/739.84us. No selection orfit follows fromthese
variable outcomes. Low-pressure/context-specific sensitivity remains possible.

## Implication foridentification

Changing astatic load's address isnot the same intervention aschanging which
data addresses pass throughit. The latter also changes cadence/dependencies.
The large scalar rotating benefit mustnot betransferred intoa SVE code-address
term. Next priority is aSVE load-stream organization control withdocumented
load counts,per-PC strides andcompute overlap,while retaining real-kernel bridges.
These diagnostics remain outside frozen calibration anddo notcomplete thegoal.

Class E Lab only;no production ormodel changes. The preceding scalar rotation
changed instruction count/cadence. This contrast holds machine bytes fixed and
tests transfer toSVE B-only andfull-no-store loaded4 matrix probes.

`RelocatedSupply` copies self-contained Generator bodies intoeight8192-byte
slots ofone65536-byte anonymous mapping. For eachbody,entry offsets within
the slots are0/32/64/96 bytes. Internal relative branches remain valid because
thebody iscopied intact; Generator emitsno external call oraddress-relative
literal dependency. Copies arechecked withmemcmp beforeexposure,cache-synchronized
andmadeRX. The mapping lives untilall harness workers join. Actual function
addresses areincluded ineachraw cell. No executed padding isadded.

Code placement changes instruction-cache/translation location aswell asloadPC
bits; it isnot a unique prefetch intervention. Four offsets areprespecified,
notselected aftertiming. Retain everyoffset result andcompare all againstoffset0
andtheoriginal B-only/full-no-store bridges. Do notselect a bestoffset andclaim
an unbiased speedup. Code effects areseparate fromchanging data allocation.

Protocol: --relocated-supply,52 cells perround,isolated and25/25,38/12,12/38;
real/B/AB/full-no-store/empty plusfour B copies(probes67–70) andfour full copies
(71–74). Persistent workspace,4-copy data rotation,256MiB scrub perLLC,5ms
background lead-in,NUMA3 CPU304 foreground/240 controller. Sixcore TLB events
andbothDDRC groups asintheprevious control. Optionalchain allocation remains
present throughshared harness mode,so usewithin-session comparisons.

Native smoke beforetwo independent5-warmup/31-round sessions399808/409808.
Localgrid/regressions43 passed;Ruff anddiff checks passed. Target build and
hardware evidence pending. Artifacts use `tmp/relocated_sve_supply_20260908/`
locally andunder `/home/zhangxu/codex/fused_cpp/` onArm-codex-internal.
This ismechanism identification,not a completed theoretical performance model.

## Validation checkpoint

Target build and52-cell smoke passed. Allfour B clones have525432 retired
instructions inisolated smoke; allfour full clones have853112. Sampleaddresses
match8192-byte slots plus0/32/64/96 offsets; byteidentity verified beforeRX.
Two formal5/31 sessions399808/409808 started aftersmoke; results pending.
Analysis/address-geometry tests bring thefocused totalto45 passing; Ruff and
diff checks passed. Analyzer rejects changing addresses withinone session,
unexpected address geometry,missing grids andbinary/counter mismatch.

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_relocated_supply.py \
  --sessions tmp/relocated_sve_supply_20260908/session1.jsonl \
  tmp/relocated_sve_supply_20260908/session2.jsonl \
  --output tmp/relocated_sve_supply_20260908/report.json
```

Alloffsets compared tooffset0 andoriginal JIT. Intervals arepointwise,not
multiple-comparison-adjusted; donotuse theirminimum asan adoption statistic.

## Completed evidence

Formalcommand exited0; fullresults supersede thepending checkpoint.
Binary:`6cb3a395228a3f6b2c31ebb321bb933e2f4b9f334a2c76cf9fc13d1f0d6a44bf`.
S1:`362b06f702d7d6b77d27e7290c4529d632fac7d04963e730c5feba3bc66f2b08`.
S2:`74523bd85d9734874f2193c6d8690c260d264023b2f7e39ecabf659eba755027`.
Raw andreport.json arecomplete locally; native snapshot remains remote.
