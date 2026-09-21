# Fixed-chain load-PC geometry contrast

## Result

Both complete31/5 sessions pass numerical,placement,PMU andreader checks.
Rotating two chains throughone static load site strongly improves sequential
traversal,butnotrandom traversal. The effect repeats despite greater retired
instruction count intherotating variant. It identifies sensitivity toinstruction/
address-stream organization,not a unique prefetch mechanism.

| Background same/other | Order | S1 fixed/rotating us | S2 fixed/rotating us | S1/S2 pairedchange % |
| --- | --- | --- | --- | --- |
| isolated | sequential | 527.80/470.96 | 525.17/465.90 | -10.34/-11.34 |
| 25/25 | sequential | 3590.57/1314.12 | 3166.51/1308.10 | -62.80/-58.49 |
| 38/12 | sequential | 5567.13/1417.09 | 4921.75/1427.31 | -74.12/-70.87 |
| 12/38 | sequential | 1578.82/1079.97 | 1550.96/1079.62 | -30.31/-30.39 |
| isolated | random | 7208.89/7334.04 | 7247.11/7364.35 | +1.72/+1.52 |
| 25/25 | random | 17745.47/18117.94 | 17833.34/18106.09 | +1.52/+1.12 |
| 38/12 | random | 18297.13/18515.67 | 18456.90/18646.03 | +1.19/+0.49 |
| 12/38 | random | 18365.54/18538.71 | 18355.97/18609.89 | +0.86/+1.28 |

Sequential25/25 pairedchangeCI95 is[-65.17,-58.35]/[-59.66,-54.56]%;38/12
is[-74.39,-72.87]/[-72.75,-69.07]%. Random38/12 S2CI[-0.52,1.72]% includes
zero; other random cells have small positive intervals. These are matched-round
IID bootstrap intervals,not arbitrary phase orlong-dependence bounds.

Real W13 bridges retain thedistribution effect:25/25=1099.58/1093.78us,
38/12=1286.61/1295.73us,12/38=732.87/730.81us acrossS1/S2. No claim of
exact historical binary equivalence ismade.

## Counters andinterpretation boundary

S1 two-PC/rotating retired instructions are327932/786676. At25/25,refills
are131207/131220 andLLC misses129728/130245; DTLB walks bothzero. At38/12,
refills131213/131190,misses130366/129848,walks bothzero. Global DDR rates
remain near277–280GB/s. Thus thelarge sequential timing benefit doesnot require
fewer program loads,substantially fewer cache-miss events orfewer page walks.

However, theisolated fixed random2 probe nowhas~131k refills,not the~184–223k
seen inearlier binary/sessions. Theearlier extra-refill anomaly isnot stable
across thisbinary/context change. Code addresses andlayout changed; allocation
andmachine state mayalso differ. Do notidentify itscause ascode-PC aliasing
orfit aconstant fromit withoutan intervention isolating those variables.

The new control changes staticPC count,perPC address stream,register moves and
loop cadence together. Random timing ismostly slightly worse while sequential
timing improves substantially,which constrains a genericinstruction-overhead
explanation. But it still doesnot distinguish prefetch tracking,request admission,
load scheduling orother pattern-sensitive effects uniquely.

Next identification step should keep theentire instruction sequence anddependency
graph fixed while varying onlycode placement,or keep code placement fixed while
varying stream order. Audit exactmachine bytes andaddress mapping. Then test
whether thesame mechanism transfers toSVE B/AB andreal small-M kernels. No
model term isadopted fromthisscalar-pointer intervention alone.

Class E Lab experiment. Keep two dependency chains,131072 pointer loads and
8MiB line coverage fixed. Compare originaltwo static load sites withone static
site that alternates the two pointer registers. Link graph andlogical address
order areidentical within each sequential/random pair. Explicit payload remains
1MiB;not the full8MiB SVE packed-B stream.

This intervention also changes loop cadence andregister moves. It is not a
pure prefetch enable/disable experiment. Target GCC assembly mustconfirmone
staticload site,no unrolling into two sites,andno loop spills before results
areinterpreted. Extra instructions andloop-control overhead mustbereported.

Protocol: --chain-pc implies the existing six-event TLB group. Isolated and
fixed50 background25/25,38/12,12/38;real/B/AB/no-store/empty andsequential/random
single/two-chain baselines;new probes65/66 (rotating sequential/random2).
44 cells,5 warmups+31 rounds,two independent seeds379808/389808. Same NUMA3,
CPU304 foreground/240 controller,4-copy rotation andscrub. Do notchange THP
orhardware prefetch configuration. New binary,so same-session bridges control
forbuild/code-layout differences; do notpool witholder runs.

Portable repeated coverage/endpoints including rotating traversal andgrid tests
passed (37 focused tests); withanalysis protocol tests42 passed. Target build
and44-cell PMU/numerical smoke passed. GCC assembly hasone loop-body load at
0x40bae0 (`ldr x2,[x1]`),two pointer-rotation moves andloop control; no loop
spills orload-site duplication. Formal sessions379808/389808 started after
these gates; measured results pending. Artifacts: `tmp/load_pc_geometry_20260908/` locally andunder
`/home/zhangxu/codex/fused_cpp/` onArm-codex-internal. No model fitting or
production changes. This narrows request-stream sensitivity butdoesnot itself
complete the small-M theoretical model.

Analysis command afterboth sessions complete:

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_load_pc.py \
  --sessions tmp/load_pc_geometry_20260908/session1.jsonl \
  tmp/load_pc_geometry_20260908/session2.jsonl \
  --output tmp/load_pc_geometry_20260908/report.json
```

Rejects incomplete grids,wrong31/5 protocol,counter/binary mismatch andduplicate
session identity. Reports allcells pluspaired within-order rotating-minus-fixed
differences. No frozen calibration file isread ormodified bythis analysis.

## Completed evidence identity

Binary SHA256:`01b92f8fab485b48078082f63e5eb82136c39393297bcfa920c09f99e31cf783`.
RawS1:`d799dea2261f93e3dc93a8170afded5cafd75cc050578f7807170d62a9810cf6`.
RawS2:`49dbdcbbe729947ae2b910086895e9bad327a498c84746650b35d3ee8e69f2d1`.
Formal command exited0; results supersede theearlier pending checkpoint.
Complete JSON andraw sessions arelocal; actualnative sources remain remote.
