# Long/short batch: why the flat queue wins, and a static partition that beats it (C9g, 2026-09-23)

Workload moe256-long-short-bimodal (5 x 2040 + 174 x 12 routes), C9g node 0 (CPUs 0-95,
`--membind=0`), 96 threads, H=4096 I=512, guarded TP4 calibration
(`tmp/c9g_dynamic_tail_20260923/cal_median_tp4_guarded.json`). Follows
`c9g_dynamic_tail_bound_20260923.md`, which ruled out a dynamic tail.

## Why quick loses: its plan runs compute first and bandwidth second

- The short experts are bandwidth-bound. Alone they take 6.4-6.5 ms at every width from 1T to
  16T (about 342 GB/s for 2.19 GB of weights). A single 16T lane streams only about 130 GB/s
  (quick trace: 91 shorts in 8.7 ms), about 8 GB/s per core, so saturating DRAM needs roughly
  40 or more cores on shorts.
- The long experts are compute-bound (longs alone 8.3-8.9 ms on planned 16T teams).
- Quick's plan (six 16T lanes) gives 80 cores to the five longs and 16 to shorts. While the
  longs run, bandwidth sits mostly idle. When they end, about half the shorts remain and run in a
  bandwidth-only phase on all 96 cores (trace: lane 80 finishes its 91 shorts at 8.7 ms, the
  long lanes start their 17 shorts each after 10.5 ms).
- The flat queue mixes long and short N blocks across all 96 threads, so DRAM stays busy while
  the longs compute. That is why it loses on either half alone and still wins the whole batch.

A core-partition estimate follows directly: with about 24 cores streaming shorts and 72 computing
longs, both regions end near 11 ms. Narrower long teams leave more cores for shorts.

## Hand-built disjoint partitions (bench_split.py --extra-plans, protocol v2, two sessions)

`L<wl>S<ws>`: each long expert gets its own lane of width wl from call start; the remaining
96 - 5*wl cores run the shorts round-robin in lanes of width ws. Stage windows inherit the
operator-wide geometry, as quick's plan does on this batch. Generator:
`tmp/c9g_dynamic_tail_20260923/gen_partition.py`; runs `part_all_s{1,2}.json`. All partition
outputs are bitwise equal to the reference.

Execution medians, ms:

| arm | session 1 | session 2 | model (placed DAG) |
| --- | --- | --- | --- |
| L14S1 | **10.33** | **10.38** | 10.04 |
| L14S2 | 10.42 | 10.48 | 10.49 |
| L13S1 | 10.42 | 10.50 | 9.63 |
| L15S3 | 10.50 | 10.62 | 11.77 |
| L12S1 | 10.50 | 10.62 | 9.27 |
| L12S2 | 10.57 | 10.65 | 9.60 |
| L15S1 | 10.57 | 10.71 | 10.86 |
| L11S1 | 10.64 | 11.57 | 9.48 |
| L12S4 | 10.69 | 10.80 | 10.09 |
| L10S1 | 10.94 | 12.08 | 9.96 |
| L10S2 | 10.98 | 12.19 | 9.86 |
| flat queue (staged L2) | 11.02 | 11.00 | - |
| L16S1 | 11.04 | 11.06 | 11.46 |
| fixed 16T | 11.84 | 11.89 | - |
| quick (6 x 16T) | 11.86 | 11.87 | 13.28 |
| full (3 x 16T + 6 x 8T) | 13.12 | 15.17 | 11.06 |

- The best static partition, L14S1, beats the flat queue by 6.3% / 5.6% and quick by 12.9% /
  12.6%. Nine partitions with 12-15T long teams beat the flat queue in both sessions.
- L16S1 only ties the flat queue: with 16T long teams the 16 short cores cannot stream fast
  enough. The widths that win (12-15T) are not in the calibration's `supported_widths`
  (1, 2, 4, 8, 16, 32, 48, 96), so no current planner can emit them.
- The analytic model prices any width when allowed (last column: the model constructed with
  widths 10-15 added, scored with `dag_makespan_placed`). It ranks the partitions coarsely
  (it prefers L12S1, measured 1.7% behind L14S1) and underprices narrow long teams by up to 18%,
  but it ranks every 12-15T partition ahead of quick's plan, and its first choice still beats the
  flat queue by 3.5-4.7%. Its misranking of full's mixed plan (predicted 11.06, measured
  13.1-15.2 ms) is outside this candidate family.

## Conclusion

The flat queue's advantage on this batch is concurrency between compute-bound and
bandwidth-bound work, and a static disjoint core partition inside the existing Plan V2 strict
executor recovers it with margin. What is missing is planner support: widths outside the
calibrated set and a long/short partition candidate scored against quick's plan with the same
simulator. Next: add both as an opt-in planner candidate and measure what it selects.
