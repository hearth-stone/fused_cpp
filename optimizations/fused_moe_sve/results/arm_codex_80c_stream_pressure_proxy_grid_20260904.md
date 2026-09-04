# Arm 80C stream-pressure proxy grid decision

Date: 2026-09-04

## Technical summary

No tested plan-visible proxy passes the locked count-6 holdout in two sessions.
The decision is to stop extending the absolute physical cost model and move
safety to partial ordering and top-K retention.

The hardware relationship itself is coherent: measured DDR queue pressure maps
to victim slowdown with slopes 0.00440/0.00476 ms per cycle, only 7.5% apart.
The failure is upstream. Distinct-B count, active requester threads, their
product, and even an oracle using measured W13 overlap cannot predict queue
pressure with stable parameters from the executable shape. Proxy-to-queue
parameter drift is 25.8--32.3%, above the predeclared 20% gate.

Do not add a count knee, width multiplier, queue threshold, or residual. Keep
frozen v8 and use conservative partial-order/top-K policy for unresolved local
differences.

## Locked experimental design

- Grid: count `{4,6,8}` by uniform width `{1T,2T}`.
- Starts: nested prefix of logical `48,50,...,62`; count 4/6 use the first 4/6
  starts of the count-8 layout.
- Fit cells: counts 4 and 8 only.
- Holdout cells: both count-6 widths, never used to fit a slope.
- Two independent sessions: seeds 20260924 and 20260925.
- Each session: one process, shared packed allocations, 5 warmups, 31 randomized
  paired rounds, per-cell direct PMU reset/read, and full disjoint scrub.
- Every PMU running ratio is 1.0.

The proxy-to-queue and queue-to-slowdown links are both fitted through the
isolated origin with nonnegative slopes. Four proxies were declared before the
run:

1. distinct packed-B count;
2. active requester threads;
3. packed-B count times requester threads;
4. packed-B count times measured peer W13 overlap core-ms, an oracle upper
   bound for a future predicted-overlap proxy.

## Predeclared gates

Every count-6 1T/2T prediction in both sessions must satisfy:

- queue error no greater than `max(3 cycles, 15% of measured queue)`;
- slowdown error no greater than `0.02 ms`;
- correct 1T-versus-2T direction;
- proxy-to-queue and queue-to-slowdown parameter drift no greater than 20%
  across sessions.

No gate was changed after reading count 6.

## Count-6 holdout is stable enough to decide

| Session | Count-6 width | Measured queue pressure cycles | Measured slowdown ms |
| --- | ---: | ---: | ---: |
| 20260924 | 1T | 16.93 | 0.0685 |
| 20260924 | 2T | 22.50 | 0.1166 |
| 20260925 | 1T | 16.29 | 0.0650 |
| 20260925 | 2T | 18.45 | 0.1103 |

Victim slowdown repeats within 0.0035/0.0062 ms for 1T/2T. Both sessions also
agree that 2T produces more queue pressure and more slowdown. The holdout is not
too noisy to reject the proxy models.

## Proxy results

Each pair below is count-6 1T/2T error. Positive means overprediction.

| Proxy | Session-1 queue error cycles | Session-1 slowdown error ms | Session-2 queue error cycles | Session-2 slowdown error ms | Proxy parameter drift | Accepted |
| --- | --- | --- | --- | --- | ---: | --- |
| Distinct-B count | +8.49 / +2.92 | +0.0434 / -0.0047 | +2.47 / +0.31 | +0.0243 / -0.0210 | 26.2% | no |
| Active requester threads | -0.53 / +10.31 | +0.0037 / +0.0278 | -4.60 / +4.93 | -0.0094 / +0.0009 | 28.7% | no |
| Count x requesters | -3.03 / +5.30 | -0.0073 / +0.0058 | -6.88 / +0.37 | -0.0202 / -0.0208 | 32.3% | no |
| Count x measured W13 overlap | -2.05 / +1.93 | -0.0030 / -0.0091 | -5.60 / -0.90 | -0.0141 / -0.0268 | 25.8% | no |

All requester-sensitive proxies predict the count-6 width direction correctly;
distinct-B count alone predicts no 1T/2T difference and fails direction. Even
for the direction-correct proxies, absolute errors determine whether a planner
can safely distinguish local 1--3% candidates.

The measured-overlap oracle is the strongest single-session result. It passes
all count-6 gates in session 20260924, but in session 20260925 it misses the 1T
queue gate by 5.60 cycles and the 2T slowdown gate by 0.0268 ms. Its parameter
drift also fails. Since measured overlap contains more information than an
execution-time planner can know exactly, a simpler structural approximation
cannot be promoted from this evidence.

## Where identifiability fails

The queue-to-slowdown slopes are 0.00440 and 0.00476 ms/cycle. Their 7.5%
relative drift passes the 20% gate. Once queue pressure is measured, its effect
on the victim is reasonably stable.

Proxy-to-queue slopes fail instead:

| Proxy | Cross-session slope drift |
| --- | ---: |
| Distinct-B count | 26.2% |
| Active requester threads | 28.7% |
| Count x requesters | 32.3% |
| Count x measured W13 overlap | 25.8% |

The hidden variable is therefore not another victim slowdown coefficient. It is
the session-dependent mapping from planned request geometry to memory-controller
queue state. Modeling that mapping would require additional machine-state and
arrival-process variables that are not available to the offline planner.

## Decision

`accepted=[]` and `stop_absolute_model_expansion=true`.

The physical investigation has achieved its useful endpoint:

- distinct packed-B count and requester/team shape explain why pressure occurs;
- DDR queue latency is the best measured intermediate variable;
- measured queue predicts victim slowdown substantially more consistently than
  plan structure predicts queue;
- no tested executable-state proxy is stable enough for a new absolute term.

Further absolute-model probes are not justified unless a new planner-visible
source of memory-controller state becomes available. The next work should:

1. keep frozen v8 as the absolute baseline;
2. use anchor-relative partial order to declare close candidates incomparable;
3. optimize top-K recall and zero false pruning rather than point MAPE;
4. replay known real-trace counterexamples without fitting them;
5. connect partial order to VND/LNS only after its safety gates pass.

## Artifacts

- Session 20260924:
  `tmp/moe_stream_proxy_grid_20260924.json`, SHA256
  `f687afb91ab79d0f3f708734d63165b3697a18e905ec01bcb2be3eace89a284a`.
- Session 20260925:
  `tmp/moe_stream_proxy_grid_20260925.json`, SHA256
  `d88a49a24e417a7bd016d1354bd7c6adbb69969fbc9fbc8cfd6cffe07f363c07`.
- Decision artifact:
  `tmp/moe_stream_pressure_proxy_grid.json`, SHA256
  `eb0e79ecdaa0361c18cdec57b2569d7ad2bfc6f9f3bd5f3cf79ff42433727a02`.
- Grid mode implementation SHA256:
  `7b921e42e0d19d28478729c4589abc67ff9e204300b863250e694cb8314077d7`.
- Proxy-grid analyzer SHA256:
  `1c7249204f7d74c976413933cdba11a1fcea905b04e16bbdb72b0882e648f8a9`.

The paired runner, direct PMU helper, frozen calibration, extension, machine,
affinity, scrub, and event definitions are unchanged from the prior paired PMU
report. No old route or real-trace holdout was opened.
