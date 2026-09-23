# Long-expert dynamic tail: bound and trace (C9g, 2026-09-23)

Question: should the planner get a candidate where the long experts of a long/short batch run
as a flat (expert, N block) queue that idle threads claim, beside a planned schedule for the
short experts? Workload moe256-long-short-bimodal (5 x 2040 + 174 x 12 routes), C9g node 0,
96 threads, H=4096 I=512, guarded TP4 calibration. Design frozen before measuring:
`tmp/c9g_dynamic_tail_20260923/design.md`.

## Sub-batch bound (bench_split.py, protocol v2, two sessions)

Medians, execution only, ms:

| session | longs alone: best / flat L2 | shorts alone: best / flat L2 | H = sum of bests | whole batch: best (flat L2) / quick | H / whole |
| --- | --- | --- | --- | --- | --- |
| 1 | full 8.88 / 10.17 | fixed 4T 6.41 / 6.97 | 15.29 | 11.03 / 11.81 | 1.39 |
| 2 | fixed 16T 8.31 / 10.16 | fixed 4T 6.42 / 6.98 | 14.73 | 11.04 / 11.84 | 1.33 |

Verdict under the frozen rule: **NO-GO** (H exceeds the whole-batch best in both sessions by
33-39%, far outside the 0.1-1.5% session repeat). The bound is a two-phase hybrid, so it pays the
per-call fixed cost twice and forbids overlap, but that cannot account for 3.7-4.3 ms.

What it shows instead:

- On each half alone the flat queue loses: planned schedules are 13-18% faster on the longs and
  8% faster on the shorts.
- The shorts stream about 2.1 GB of weights (174 experts x 12 MiB) in 6.4 ms, about 330 GB/s:
  bandwidth-bound. The longs are compute-bound. The whole batch is 3.7-4.3 ms cheaper than the
  halves run apart, so its speed comes from running the two kinds of work concurrently.

## Trace of quick's plan on the whole batch (trace_quick.py)

Quick picks six 16T lanes: five each hold one long expert plus about 17 shorts, the sixth holds
91 shorts. With the native trace on (which inflates the call from about 11.8 to 15.3 ms), every
lane ends within 0.05 ms of the others; tail idle is 2.5% of thread-time. There is no tail for
a dynamic tail to recover, so the 7% gap between quick and the flat queue is in the rate at
which the work runs, not in idle threads at the end.

## Conclusion

The earlier reading of the related-work shapes result ("the flat queue wins with its dynamic
tail", `c9g_related_work_20260923.md`) is not supported: its planned schedules are balanced and
the flat queue loses on either half alone. The candidate is not built. The open question is
what the flat queue does differently while long and short work overlap - it runs every expert's
W13 and then every expert's W2 behind a stage barrier, with all 96 threads sweeping the same
expert's N blocks at once - which a per-stage timing comparison of the two executors on this
batch can separate.

Follow-up (same day, `c9g_long_short_partition_20260923.md`): the rate difference is concurrency
between compute-bound longs and bandwidth-bound shorts, and a static disjoint core partition
beats the flat queue without a per-stage timing run.
