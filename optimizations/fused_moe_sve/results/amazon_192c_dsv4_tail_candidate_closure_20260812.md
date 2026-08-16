# DSV4 tail-candidate closure on AmazonC5192Cores

Date: 2026-08-12

## Question

Can end-of-run idle be reduced without weakening the selected large/medium
head, and which mechanism is responsible for any gain: short-expert team
regrouping, queue ordering, a planned suffix DAG, or residual-M route slicing?

## Method

Common configuration:

- repository base: `204f006` plus the Lab comparator described below;
- machine: AmazonC5192Cores, NUMA0 CPUs 0-95;
- memory policy: local NUMA allocation and 32 MiB HugeTLB;
- TP4 shape: `H=4096`, `F=512`, 256 experts;
- workload: captured `dsv4-real-2048-seq70`, 2048 tokens, TopK=6;
- 223 active experts, including 75 with M<=12 (`71xM1`, `3xM6`, `1xM10`);
- base plan: 48 cores x 8T for M>48 and 48 cores x 1T for M<=48;
- ARM SVE128 BF16 JIT exact-M, `-O2`, BF16 route output, identical packed
  weights and output storage;
- paired variants alternate in one process and use the median wall time.

The E218 task uses backend N tile 8 and inherits the unsplit task's exact stage
windows. Full W13/W2 packed-B stages are 8 MiB and 4 MiB. Both slices use
`(t, w13_window_tiles, w2_window_tiles, R13, R2) = (8, 2, 8, 8, 8)`, or
128 KiB W13 and 64 KiB W2 per worker per window. Thus the comparison changes
only the contiguous M slices, dependencies, and lane occupancy; it does not
change the kernel, packed layout, or stage-window policy.

Every runtime candidate was checked against the unsplit plan before timing and
produced bit-exact output. Retired candidates were temporary implementations;
their runtime, schema, API, planner, environment controls, and tests were
removed after the decision.

## Results

### Short-expert regrouping

The first candidate retained 1T work conservation while fixed work remained,
then allowed each released same-NUMA 8T cohort to choose `1/2/4/8T` before a
whole expert started. Seven warmups and 51 paired measurements gave:

| Variant | Median | Relative result |
| --- | ---: | ---: |
| Fixed large/small partition | 11.497479 ms | baseline |
| Adaptive short-tail cohort | 11.595985 ms | -0.849% |

Of 75 pooled experts, 73 still ran at 1T and only two ran at 4T. A separate
low-overhead trace measured internal/tail idle as `24.60/70.94 core-ms` for
the baseline and `34.89/44.50 core-ms` for the candidate. Tail area fell, but
internal idle grew and wall time regressed.

A narrower nonblocking candidate allowed only immediately available 2T/4T
groups. Its 51-pair medians were `11.871177 ms` fixed and `11.944337 ms`
candidate (`-0.613%`). All 75 tasks ran at 1T, so no natural regrouping
opportunity was realized.

### Short-task queue order

The existing fixed 1T queue uses LPT. Reversing it to increasing M on the same
plan and weights produced these 51-run medians:

| Order | Median |
| --- | ---: |
| Increasing M | 11.883804 ms |
| Existing LPT | 11.892050 ms |

The LPT-relative result was `-0.069%`, inside noise. There is no evidence for a
new queue policy, so the existing order remains unchanged.

### Static residual-M suffix

The positive mechanism was isolated by selecting one terminal expert before
execution and replacing it with two equal-granularity contiguous route slices.
For expert 218 (`M=197`), the owner lane was core 24 at 8T and the donor lane
was core 40 at 8T. Plan V2 materializes the slices as `M=99` and `M=98` and
publishes expert completion only after both finish.

Five independent 51-pair comparisons gave:

| Repeat | Unsplit | Static split | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 11.810890 ms | 11.683236 ms | +1.093% |
| 2 | 11.734972 ms | 11.678737 ms | +0.482% |
| 3 | 11.831427 ms | 11.690581 ms | +1.205% |
| 4 | 11.779787 ms | 11.699704 ms | +0.685% |
| 5 (clean rebuild) | 11.821501 ms | 11.682092 ms | +1.193% |

The mean relative gain is about 0.93%, below the 2% production adoption gate.
For the clean-rebuild repeat, the aligned-pair speedup median/P10/P90 was
`+1.181%/-0.221%/+3.451%`; unsplit/split P90 was
`12.037473/11.803427 ms`. Both variants contained a few approximately 13.4 ms
system outliers, so the decision uses medians and does not expand the claim.
The selection is sensitive: terminal targets E217, E208, E216, and E219 gave
`-0.147%`, `-0.190%`, `+0.046%`, and `-0.153%`, respectively.

The retained Lab comparator is reproducible with:

```bash
FUSED_CPP_MOE_HUGETLBFS_PATH=/dev/hugepages-32M \
OMP_DYNAMIC=FALSE MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/capture_schedule_timeline.py \
  --threads 96 \
  --preset dsv4-real-2048-seq70 \
  --large-small-partition 48:48:8:1 \
  --residual-m-split 218:40 --compare-residual-m-split \
  --route-dtype bf16 --warmup 7 --runs 51
```

### Planned and runtime-selected suffixes

The cost model selected E219/core40. It predicted `14.932255 -> 14.891614 ms`
(`+0.273%`), while 51-pair execution measured `11.470484 -> 11.479305 ms`
(`-0.077%`). The candidate was therefore a model mis-selection, not a usable
small gain.

A bounded runtime protocol instead waited up to 300 us for actual lane
completion, offered one unstarted terminal expert, and split it only after a
same-width donor accepted. Eleven paired medians were `11.742341 ms` unsplit
and `11.784479 ms` runtime split (`-0.358%`). The selected task varied between
runs or no split occurred. A whole-task tail-steal control regressed by 0.763%.
Thus online discovery, handoff, and unstable target selection erase the small
benefit of the favorable static geometry.

## Decision

Retire short-expert width regrouping, queue reordering, planner-generated
suffix selection, and runtime residual-M handoff. Do not retain default-off
runtime branches for them.

Keep only the explicit static residual-M Lab comparator because its geometry
is repeatedly positive for one selected target and reuses the existing strict
Plan V2 route-slice contract. It does not enter the production planner, cost
model, cache identity, schema, or default runtime because it misses the 2% gate
and target/donor selection is not robust. P0 end-of-run tail idle remains open.
