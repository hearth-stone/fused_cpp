# Memory-supply versus compute-overlap evidence audit

## Answer and confidence

Existing workspace evidence supports selective sensitivity to workload context,
but **does not establish that memory supply deteriorates for both large and small
tasks and is hidden by computation for the large task**. That causal claim
requires new phase-scoped PMU evidence. This audit performs no new hardware run,
model fit, runtime modification or calibration adoption.

What can be established now: large-M W13 overlaps substantial background work;
its unchanged runtime is not explained by all peers having finished. The model
already implements compute/transfer overlap and cold/steady decomposition, then
applies an empirical wide-team multiplier that can slow compute-dominated phases.

## Matched timing evidence

Source: `tmp/workspace_isolated_width_20260907/summary.json` and the full trace
samples in `tmp/workspace_phase_timeline_20260907/high_skew_analysis.json`.
Arm-codex-internal,NUMA3 CPU240–319,E256/H4096/F512,2048tokens,TopK6,BF16,
Ntile16,full owner stripes `(t,0,0,1,1)`,merge on,fixed pre-touched workspace,
four weight copies,216MiB scrub,31 paired rounds,two sessions. Full W13/W2
weight bytes8MiB/4MiB per expert. Exact commands/build/calibration identities are
retained in `workspace_isolated_width_20260907.md`.

- M1/1T W13: isolated0.3125/0.3202ms,full anchor0.8530/0.8365ms.
- M1341/16T W13: isolated8.4417/8.4360ms,full anchor8.4457/8.4386ms.
- Corresponding round-paired increments:+0.5449/+0.5170ms versus
  +0.0057/+0.0074ms. This is timing evidence,not direct memory-service evidence.

For each target W13/W2 interval `[a,b]`, intersect every other task's gather,
W13,W2 stage envelope with it, weight the intersection by the other task's
assigned width,and divide by `b-a`. Report the median over31 calls separately
per session. This measures allocated-thread stage-envelope overlap, **not**
actual issuing cores, memory requests or bandwidth.

| Anchor target/stage | Other assigned-thread overlap s1/s2 | Of which narrow widths<=2 s1/s2 |
| --- | --- | --- |
| M1/1T W13 | 77.955/78.126 | 14.962/14.961 |
| M1/1T W2 | 78.959/78.958 | 14.971/14.985 |
| M1341/16T W13 | 63.564/63.309 | 15.971/15.971 |
| M1341/16T W2 | 56.575/55.372 | 14.416/14.151 |

At least one peer stage overlaps100% of each target interval in this table.
M1341 W13 has nearly all64 possible peer threads represented. This rules out
the simple explanation that its entire phase runs after background completion.
It does not prove that those peers issue enough requests to reduce victim supply.

The comparison changes width and placement as well as M: M1 uses logical begin68
(CPU308), whereas M1341 uses begin32,width16 (CPU272–287),crossing the two LLC
domains at CPU279/280. Do not interpret it as a controlled M-only intervention.

## What the frozen model says, not what PMU measured

Recomputed using `AnalyticMoeCostModel.predict_expert` from frozen v8:

| W13 phase | GEMM core time | LLC service time | DRAM service time | Base phase time |
| --- | ---: | ---: | ---: | ---: |
| M1/1T cold | 0.1891ms | 0.2048ms | 0.2435ms | 0.2435ms |
| M1341/16T cold | 0.0720ms | 0.0414ms | 0.0505ms | 0.0720ms |
| M1341/16T steady | 7.9820ms | 0.7298ms | 0 under isolated spill assumptions | 7.9820ms |

The model assigns compulsory8MiB W13 weight reads to the first cold phase.
The cold phase is only about0.89% of modeled large-M W13 time. There are two
distinct explanations compatible with weak whole-stage sensitivity:

1. slower transfers remain below computation within a phase;
2. extra delay is concentrated in a short initial cold portion and amortized
   over a long reuse/steady phase,even if that delay is not hidden.

Current hardware trace boundaries do not resolve those cold/steady portions.
Model steady DRAM=0 is an assumption for isolated spill,not proof of zero actual
DRAM traffic or zero full-workload spill. Never use it to establish the mechanism
being tested.

`AnalyticPhase._duration_from_resource_times` already takes
`max(gemm_core, max(L2,LLC,DRAM))` before adding epilogue/fixed cost. The placed
path then multiplies by wide-team dilation and narrow correction. The earlier
ablation showed that removing wide-team pressure substantially reduces large-M
overprediction. Thus the priority is auditing those residual multipliers and
resource-demand assumptions,not adding a first implementation of overlap.

## Missing evidence

Workspace trace headers contain phase times and worker/expert identity,but no
phase counters. Existing `bench_stream_pressure_pmu_paired.py` resets counters
around whole native calls for a fixed small victim; its protocol also differs
(merge off,older output lifecycle). It has no matched large-M workspace phase
control. DDRC/L3C values there are shared-domain counters,not victim-specific
phase latency. The generic DDR barrier probe is not a MoE phase workload.

Consequently neither old PMU data nor `weight_bytes/stage_time` can establish
the missing causal link. The latter is a derived achieved rate,not independently
measured service capacity; using it here would be circular.

## Minimum discriminating next measurement (not run)

Use a controlled M×width grid: M1/M1341,1T/16T,on matched LLC placement,with
isolated versus an identical sustained same-domain background. Observe W13 and
W2 separately. Keep output workspace and cold-weight protocol fixed; verify
background remains active throughout the target phase. This removes the current
width/domain confounding. Full real-trace replay remains a separate context test.

Counter windows must be stage-aligned and exclude initialization,scrub and
delayed work. Collect victim-core cycles/instructions,refills and supported
memory/backend-stall events across all victim workers,plus matched-window
L3C/DDRC read traffic and queue occupancy/commands. Shared counters alone cannot
attribute a latency increase to the victim. Prefer load-latency sampling if
supported; otherwise state the weaker attribution of combined core/uncore
evidence. Split initial cold behavior from steady behavior where feasible.

- Independent service/latency evidence worsens while large-task cycles/runtime
  and unhidden memory stalls stay nearly unchanged: supports overlap or cold
  delay amortization; distinguish the two with time-resolved evidence.
- Large task has little external refill after its initial load: supports reduced
  exposure/reuse,not the claim that its supply necessarily worsened.
- Small task slows without corresponding memory evidence: examine synchronization,
  scheduler,merge or other stage costs; do not absorb everything into bandwidth.

Adding stage-scoped collection is new diagnostic instrumentation work. This
audit does not claim that experiment has been executed or that any one outcome
has already been established. Validation assessment: usable with these explicit
caveats; causal memory-supply conclusion remains unverified.
