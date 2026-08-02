# Cold-phase oracle runtime validation on AmazonC5192Cores

## Scope

- Date: 2026-07-31
- Host: `AmazonC5192Cores`
- Affinity: NUMA0, logical CPUs `0-95`
- Workload: `dsv4-real-2048-seq70`, `T=2048`, `TopK=6`, `E=256`, 223 active experts
- Shape: TP4/F512, `H=4096`, `F=512`, split-W13 SVE JIT exact-M
- Baseline: production strict `12x8T`
- Candidate: cold-phase CP-SAT mixed-width incumbent lowered to contiguous strict Plan V2 teams
- Samples: 7 warmups and 51 timed runs, randomized variant order

The physical-placement pass held every oracle time interval fixed and solved only
for a contiguous core interval. It was `OPTIMAL` in `0.06587 s`. Every candidate
was checked bit-exact against the fixed plan before timing.

## Command

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  env PYTHONPATH=src .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_cold_phase_oracle_runtime.py \
  --oracle-report /tmp/dsv4_cold_oracle.json \
  --placement-input /tmp/dsv4_runtime_placement.json \
  --release-scales 0,0.25,0.5,0.75,1,1.25 \
  --warmup 7 --runs 51 \
  --output /tmp/dsv4_oracle_runtime_51runs.json
```

## Results

| plan | median | p10--p90 | aggregate | throughput vs fixed |
| --- | ---: | ---: | ---: | ---: |
| fixed `12x8T` | 14.854 ms | 14.803--14.924 ms | 10.409 TFLOP/s | baseline |
| mixed eager | 20.643 ms | 20.528--21.609 ms | 7.490 TFLOP/s | -28.05% |
| mixed release `0.25x` | 20.649 ms | 20.549--21.444 ms | 7.488 TFLOP/s | -28.07% |
| mixed release `0.50x` | 20.765 ms | 20.676--20.915 ms | 7.446 TFLOP/s | -28.47% |
| mixed release `0.75x` | 20.780 ms | 20.711--20.961 ms | 7.441 TFLOP/s | -28.52% |
| mixed release `1.00x` | 20.869 ms | 20.793--20.994 ms | 7.409 TFLOP/s | -28.82% |
| mixed release `1.25x` | 21.008 ms | 20.905--21.615 ms | 7.360 TFLOP/s | -29.30% |

The oracle predicted `12.500 ms` for fixed and `8.782 ms` for mixed. Actual
median error was therefore `+18.83%` for fixed and `+137.63%` for the fully
timed mixed plan. Instead of the predicted `42.34%` incumbent throughput gain,
mixed increased latency by `40.50%` and reduced throughput by `28.82%`.

## Interpretation

The release mechanism is not the regression: `0.25x` release differs from eager
by only `0.03%`. Delaying more monotonically worsens the result.

The rejected candidate uses
`173x1T + 7x2T + 7x4T + 34x8T + 2x16T`. The surrogate charges packed-B DRAM
traffic only during one M12 cold phase, then assumes the remaining isolated
service time is contention-free. With this many concurrent narrow experts,
later panels do not retain that isolated cache state: shared-cache-to-L2
service, capacity eviction, contention-dependent compute rate, dispatch,
scratch/gather, store, and merge are absent from the oracle constraint.

The strict timed-release lowering is useful as a falsification tool, but this
mixed-width incumbent is not a production candidate. The next oracle revision
must model lower-cache refill/service throughout each active weight window and
validate each candidate against the production contention model before runtime.

Machine-readable samples are in
`amazon_192c_cold_phase_oracle_runtime_20260731.json`.

## Root-cause follow-up (2026-08-01)

The original result proved that timed release did not recover the oracle gain,
but did not separate low occupancy from cache contention. A second run added
stage traces, active-window reconstruction, width/concurrency sweeps, and
process-wide PMU sampling over the pinned 96-worker team.

### Critical path and occupancy

| metric | fixed `12x8T` | mixed eager |
| --- | ---: | ---: |
| E2E trace time | 14.732 ms | 20.498 ms |
| `scheduled_compute` | 14.110 ms | 19.848 ms |
| merge total | 0.389 ms | 0.401 ms |
| route build | 0.089 ms | 0.128 ms |
| visible compute work | 1228.7 core-ms | 1108.3 core-ms |
| average active cores | 93.51 / 96 | 57.18 / 96 |
| visible capacity utilization | 90.71% | 58.17% |

About `5.74 ms` of the E2E difference is inside `scheduled_compute`. Mixed does
less summed visible worker time, but its fixed core rectangles leave large
holes when actual task durations differ from the oracle. Those holes cannot be
backfilled, so lower work does not become lower makespan. Merge, route setup,
and postprocessing are not material sources of the regression.

### Active cache window and PMU evidence

The host exposes a private 2 MiB L2 per core and a shared 96 MiB L3 per NUMA.
Reconstructing the actual stage intervals gives:

| plan | W13 peak / average | W2 peak / average |
| --- | ---: | ---: |
| fixed `12x8T` | 44.0 / 26.9 MiB | 36.0 / 12.4 MiB |
| mixed eager | 201.5 / 82.3 MiB | 161.5 / 44.0 MiB |
| mixed, minimum 4T | 47.5 / 21.2 MiB | 33.5 / 8.9 MiB |

The mixed peak is over twice the NUMA L3 capacity. Fifty-call PMU medians per
call show the corresponding lower-cache amplification:

| counter | fixed | mixed eager | delta |
| --- | ---: | ---: | ---: |
| L2 refill | 5.208 GiB | 6.203 GiB | +19.1% |
| last-level read | 0.324 GiB | 0.348 GiB | +7.6% |
| last-level read miss | 0.251 GiB | 0.305 GiB | +21.5% |
| instructions | 9.629 G | 13.804 G | +43.4% |

The instruction delta is not the wall-time cause. A diagnostic per-core task
list reduced mixed instructions to 8.726 G (`-36.8%`) without changing its
approximately 20.7 ms wall time. It removed scanning on otherwise idle cores
and exposed a 34.75% active memory-stall fraction. Context switches were
`436--462` per call across plans and CPU migrations were zero.

### Width and concurrency controls

Restricting the same oracle to a minimum team width isolates the balance between
active expert count, cache window, and core coverage:

| allowed minimum width | median | aggregate | throughput vs fixed |
| --- | ---: | ---: | ---: |
| fixed `12x8T` | 14.8 ms | about 10.45 TFLOP/s | baseline |
| 2T | 17.114 ms | 9.035 TFLOP/s | -13.61% |
| 4T | 15.087 ms | 10.249 TFLOP/s | -2.00% |
| 8T | 15.833 ms | 9.765 TFLOP/s | -6.38% |

Four threads is the best mixed compromise on this host: up to 24 experts can
cover all 96 cores while their reconstructed stage windows remain around the
96 MiB L3 budget. Two threads still admit too many active windows; eight
threads reduce placement and tail-packing flexibility.

A separate completion-token cap on the original 1T small tasks confirms that
cache pressure is real but cannot be fixed by serialization alone. For M<=32,
24 slots improved mixed from `20.661` to `19.316 ms`; 8 slots worsened it to
`26.692 ms` because occupancy collapsed.

### Duration mismatch

The mixed oracle's isolated durations are least accurate exactly where it uses
narrow teams heavily. For 1T tasks, median actual/model ratios were `3.23x` at
M1, `4.58x` at M28, and `1.39x` at M197. M28 reached 33 simultaneous experts.
The oracle timeline allowed at most 8 simultaneous M<=12 cold phases, while
runtime reached 20 because absolute releases are only start-time lower bounds:
when preceding tasks run longer, nominally separate waves overlap.

The concrete failure chain is therefore:

1. The cold-only model charges B traffic once and underestimates later-panel
   lower-cache service for narrow teams.
2. Narrow tasks run longer and overlap more than the oracle timeline predicts.
3. Active W13/W2 windows exceed shared L3 and refill traffic increases.
4. Fixed core rectangles cannot backfill the resulting timing holes, reducing
   average active cores from 93.51 to 57.18.

The next model must account for per-worker L2 stripe retention and the complete
active stage window, then lower a candidate to a feedback-controlled runtime or
a work-conserving placement. Static absolute release times cannot preserve a
modeled concurrency bound after duration error.
