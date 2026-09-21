# Arm-codex 80C order/width selection and DDR interleave investigation

Date: 2026-09-06.

## Scope

Change class: read-only investigation. No source, calibration, packed format,
Plan V2, kernel, or runtime default was modified. All planner numbers below are
**model-internal predictions under frozen v8**, not hardware measurements. The
only new hardware measurements are two read-only DDR probes that allocate their
own memory and do not use the MoE kernels.

Starting question: does lane-internal execution order carry usable gain, and is
the current neighborhood search covering it?

Identity used throughout:

| Item | Value |
| --- | --- |
| Calibration | `bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json` |
| Trace | `bench_assets/moe_paper/dsv4_routes_pt_20260830/measured_request016_case017_zh2048-018.pt`, layer 4 |
| Shape | 2048 tokens, TopK 6, 256 experts, H4096/F512, bf16, 80 threads |
| Active experts | 224, routes 1–1718 |
| Machine | `Arm-codex-internal`, NUMA3 CPUs 240–319 |

## Result 1: order freedom is real but concentrated in wide lanes

Lane-internal reorder does not change any lane's isolated load, so it does not
change the isolated makespan or the LPT balance. It is a pure contention-side
degree of freedom. Its model-visible size depends strongly on lane width.

Relative to each shape's own `lpt` baseline:

| shape | lpt | reverse_odd | reverse_even | per-lane parity descent | best rotation | range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 28 lanes `[8,8,8,8,2×24]` (adopted) | 38.826 ms | −0.35% | −0.45% | +0.0015% | −0.13% | 0.45% |
| 10 lanes uniform 8T | 39.679 ms | +0.38% | +0.41% | +0.49% | +0.57% | 0.57% |
| 5 lanes uniform 16T | 37.109 ms | +5.61% | +5.29% | +5.61% | **+6.91%** | 6.91% |

On the adopted 28-lane anchor the entire order neighborhood spans 0.45%, below
the 2% actionable-gain threshold. Extending the neighborhood from the shipped
3 points to a greedy search over per-lane independent reversal (a 2^28 space)
adds 0.0015%. On 16T lanes the same freedom is worth 5.6%, and rotation beats
the 3-point optimum by a further 1.3 percentage points.

This gives a structural reason for the Step-1 neighborhood-audit gate failure
recorded in `arm_codex_80c_executable_neighborhood_audit_20260902.md`: the
search anchored on a shape whose order freedom had already been eliminated.

## Result 2: shape × order is already jointly searched; the loss is in selection

`plan()` evaluates all 141 candidate shapes, and `_candidate` →
`_score_shape_with_order` → `_temporal_assignment` runs the 3-point order search
inside each shape (acceptance tolerance `1e-10` relative). Different shapes
select different orders, so the joint enumeration is real.

The makespan-optimal candidate uses a wide lane and a reversed order. Top strict
candidates by predicted makespan:

| rank | shape | makespan | uncertainty | pessimistic | working set | order |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | 9 lanes `(16,8,8,8,…)` | **34.116 ms** | 5.12 | 39.23 | 72 MiB | reverse_odd |
| 1 | 7 lanes | 34.422 | 5.16 | 39.59 | 56 MiB | reverse_odd |
| 2 | 6 lanes | 34.472 | 5.17 | 39.64 | 48 MiB | reverse_odd |
| 5 | 5 lanes 16T | 35.028 | 5.25 | 40.28 | 40 MiB | reverse_odd |

What the two selection paths return on the same candidate set:

| path | selected | predicted | working set | regret vs rank 0 |
| --- | --- | ---: | ---: | ---: |
| `_select_analytic_full` (taken; `_uses_analytic_full()` is true here) | 28 lanes, lpt | 38.826 ms | **224 MiB** | **13.80%** |
| `_select` | 3 lanes `(32,32,16)`, reverse_odd | 38.007 ms | 24 MiB | 11.40% |

Two mechanisms produce this:

1. `_select_analytic_full` contains an explicit width fallback. When the
   makespan-optimal candidate's maximum width exceeds 8, it takes
   `target_width` = the largest width below that, and re-selects among
   candidates whose maximum width is `<= target_width`. Here rank 0 has a 16T
   lane, so the search is forced into the `<= 8T` band and returns the 28-lane
   plan. Wide lanes are exactly where order freedom lives.
2. `relative_uncertainty` is a fixed `0.15`. 49 of 141 candidates fall inside
   the "overlaps the fastest" set, spanning 34.12–43.30 ms of predicted
   makespan. A 27% real spread is declared indistinguishable and the decision
   passes to tie-breaks (`active_working_set_bytes`, then `resource_groups`),
   both of which correlate strongly with width.

Note the adopted plan's working set is 224 MiB against a 93 MiB effective LLC
(140 MiB × 0.667), while the rejected rank-0 candidate fits at 72 MiB. The
fallback did not improve the working set either.

## Result 3: ablating the calibration corrections

Each variant plans independently, then every selected shape is re-scored by the
**unmodified v8** so a single judge ranks the decisions.

| variant | picked shape | own prediction | v8 re-score | order | regret |
| --- | --- | ---: | ---: | --- | ---: |
| A: v8 as shipped | 28 lanes, max 8T | 38.826 ms | 38.826 ms | lpt | 12.63% |
| B: `wide_team_pressure` emptied | 6 lanes, max 16T | 23.571 ms | **34.472 ms** | reverse_odd | 0% |
| C: `narrow_team_contention_correction` emptied | 38 lanes, max 8T | 39.140 ms | 39.047 ms | lpt | 13.27% |
| D: both emptied | 6 lanes, max 16T | 23.571 ms | 34.472 ms | reverse_odd | 0% |

Arm B is **not** an argument for removing `wide_team_pressure`. Its own
prediction is wrong by 32% (23.571 against v8's 34.472); it reaches a good shape
because over-optimism about wide teams cancels the width fallback. Without the
correction the model would escape to unmodeled 32/40/80T plans on other traces,
which is the stated purpose of that calibration.

The finding is the internal inconsistency: with one model and one candidate set,
the selection rule returns something other than its own top-ranked candidate.

This also **refutes an earlier conjecture in this investigation** that the
narrow-team discount biases selection toward many 2T lanes. Removing it makes
the planner pick *more* lanes (38) and increases regret.

## Result 4: where the correction coefficients come from

`wide_team_pressure` (`arm_codex_80c_wide_team_strict_gate_20260902.md`,
revision `6b6b4d10b9d1`):

| width | isolated `B_t` | full-cohort `S_t` |
| ---: | ---: | ---: |
| 4 | 1.000 | 1.000 |
| 8 | 1.144 | 1.303 |
| 16 | 1.251 | 1.470 |
| 32 | 1.494 | 1.619 |
| 40 | 1.424 | 1.821 |
| 80 | 2.306 | 2.306 |

Provenance facts that matter for how it should be used:

- `fit_statistic` is `median(measured_median_ns / placement_event_predicted_ns)`
  over 10 fit layers. It is a **residual level correction**, not an independent
  physical model.
- `held_out_from_final_gate: true`; the three gate traces were not used to fit.
- `unmeasured_width_policy: "no dilation"` — uncalibrated widths get 1.0, so
  cross-width comparison mixes penalised and unpenalised widths.
- `raw_width_medians` records 4T at **0.919928** (the model *over*predicts 4T by
  8%), floored to 1.0 by `dilation_floor: 1`. 4T is the only width measured to be
  overpredicted and left uncorrected, while 1T/2T overprediction is corrected
  separately by `narrow_team_contention_correction` (0.785 / 0.532).
- 32T full-cohort `1.618875` is derived, not measured directly:
  `B32 + (median_residual − B32) / (32 / (80 − 32))`.

The same gate report contains a same-class counterexample: on high-skew,
`cp_sat_06` measured 33.148 ms while predicted at 34.693 ms, and the selected
incumbent measured 34.484 ms against a 34.483 ms prediction. Both absolute
errors are inside the 5% gate (0.00% and 4.66%), but the ordering is inverted,
leaving 4.03% measured regret. **Level accuracy and ranking accuracy are
separate properties**: when the real gap between two candidates (3.9%) is
smaller than the difference in their prediction errors (4.66%), the ordering is
not determined. A per-width median residual fixes levels and cannot fix
cross-width ordering.

`narrow_team_contention_correction`
(`arm_codex_80c_narrow_lane_merge_calibration_20260903.md`): 1T `0.784898`,
2T `0.532344`, plus a 2T expert overhead of `88,799.963 + 5,909.533 × routes` ns.
Fitted on **two cores** (logical 64/65), `2×1T` versus `1×2T`, one fixed
`4×16T + 14×1T` background, three synthetic route cases. Its decisive holdout
(`arm_width_neighborhood_audit` from `1fcbc7b`) retained baseline on all three
traces and did not reproduce the prefinal merge gains; no candidate was
positive-P10 in both formal sessions. Keeping v8 is a least-bad choice, not a
passed validation for this coefficient.

## Result 5: DDR topology and interleave (new hardware measurement)

Machine structure, from read-only queries:

| Item | Value |
| --- | --- |
| CPU | Kunpeng 920, 320 logical CPUs, 4 NUMA nodes × 80 cores |
| SCCL | 8 total; **two per NUMA node**, 40 cores each |
| node3 | sccl27 (CPUs 240–279) + sccl25 (CPUs 280–319) |
| Per SCCL | 40 cores, 70 MiB L3, 4 DDRC × 2 channels = 8 channels |
| Calibration LLC domains | exactly these two SCCLs |
| DDRC PMU | v2, `identifier 0x30`, `hisi_sccl{N}_ddrc{controller}_{channel}` |
| THP | `enabled = [always]` (not madvise) |
| HugeTLB pool | `HugePages_Total: 0` — empty |

Interleave probe (`tmp/ddr_probe/probe_ddr_interleave.py`, strided reads over
2 GiB, monitoring all 16 channels of both SCCLs):

| stride | dram_ratio | effective channels (of 16) | sccl25 share |
| ---: | ---: | ---: | ---: |
| 128 B – 4 KiB | 0.64–1.19 | **16.00** | 0.501–0.509 |
| 8 KiB | 0.681 | **8.01** | 0.513 |
| 16 KiB | 0.328 | 4.01 | 0.512 |
| 32 KiB | 0.065 | 2.01 | 0.998 |

`granularity = stride × eff / 16` gives **4 KiB** consistently from the 8/16/32
KiB rows. The most reliable evidence is the 4 KiB→8 KiB transition; rows at
16 KiB and above have `dram_ratio` below 0.33 because the touched line set drops
under 8 MiB and becomes L3-resident, so only the transition location is usable.

**Interleave granularity is 4 KiB, and the scope is all 16 channels of node3**
(sccl25 and sccl27 each take ~50%).

Two consequences:

- `FUSED_CPP_PAGES=hugetlb` cannot control channel distribution. THP is already
  `always`, so packed weights are already 2 MiB pages; a 2 MiB page contains 512
  interleave blocks spread evenly over 16 channels. Enlarging pages to 32 MiB
  changes nothing about channel distribution. Separately, the HugeTLB pool is
  empty, so that setting currently falls back to `thp` silently — any earlier
  claim of having measured `hugetlb` needs re-checking.
- The existing `bench_stream_pressure_*` probes monitor only `hisi_sccl25_*`,
  i.e. **8 of the 16 channels** the memory actually interleaves across. All
  published channel-skew numbers from those probes are a half-coverage
  projection.

## Result 6: packed layout is fully phase-aligned to the interleave period

Measured from runtime addresses (18 experts, H4096/F512, `arm_sve_bf16`,
interleave period = 16 × 4 KiB = 64 KiB):

| | W13 | W2 |
| --- | --- | --- |
| base % 64 KiB | 57344 (bucket 14) | 57344 (bucket 14) |
| expert stride | 8.000 MiB | 4.000 MiB |
| **expert stride % 64 KiB** | **0** | **0** |
| distinct start buckets over 18 experts | **1** | **1** |
| worker-segment strides (1/2/4/8/16 workers) | 8/4/2/1/0.5 MiB, all % 64 KiB = 0 | 4/2/1/0.5/0.25 MiB, all % 64 KiB = 0 |
| distinct start buckets at any worker count | **1** | **1** |

Every expert and every worker segment starts on the same 4 KiB interleave
bucket. Additionally W13's tile block is `k_pad(4096) × n_tile(8) × 2 B` =
exactly 64 KiB, equal to the interleave period, so block-level progress does not
disperse the phase either; W2's block is 8 KiB and advances 2 buckets per block.

This is a structural fact derived from addresses, independent of any performance
claim.

## Refuted hypotheses

Recorded because they cost measurement effort and should not be retried as
stated.

| Hypothesis | How it failed |
| --- | --- |
| Time-weighted concurrent stream count as an order objective | Identically equal for all lane-internal reorders — lane loads and occupancy intervals are unchanged. Zero discriminating power, by algebra, not measurement. |
| Peak concurrent small-stream count as an order objective | Anti-correlated with v8 event time: lpt peaks at 25 yet is fastest; reverse_even peaks at 13 and is 0.45% slower. |
| Physical page placement causes the channel skew | Four independent weight-copy allocations in one session have near-identical skew (spread 0.0098–0.0138) versus 0.151 across sessions. Also excluded by 4 KiB granularity: any ≥64 KiB allocation covers all channels evenly regardless of page placement. |
| Phase aliasing causes aggregate channel imbalance | A/B probe: the fully phase-aligned arm still measures effective channels 16.00, identical to the staggered arm. Aggregate counters cannot see instantaneous distribution — a worker streaming 8 MiB crosses 128 full interleave periods. |
| `hugetlb` can control channel distribution | 4 KiB granularity ≪ the 2 MiB THP already in use. |
| Planner-visible stream-count geometry can predict queue pressure | Already refuted before this investigation; see `arm_codex_80c_stream_pressure_plan_geometry_proxies_20260905.md` (all proxies fail the count-6 holdout, 27–62% cross-session drift, identical 8×2T geometry produces 48.168 vs 29.142 cycles). |

## Unresolved contradiction

The originally observed skew is an **aggregate** quantity: `read_cmd` ratio
`_1/_0` of 1.311 (S1) versus 1.029 (S2) on `isolated_head`. But aggregate
coverage is provably even for a 12 MiB weight read (192 complete interleave
periods), and the interleave probe measures exactly 8.00 effective channels
within sccl25 for strided reads. Same event (`flux_rd`, 0x84, the sysfs-documented
encoding), same PMU devices, same machine — one uniform, one skewed by 1.144.

Flux composition does not explain it: `isolated_head` reads 12 MiB of weights
against roughly 26 KiB of activations, intermediates and output, and the scrub
uses a disjoint 216 MiB copy before the counters are reset, so the target
weights are cold.

Candidates not excluded: the interleave probe pins one core (304) while the grid
runs 16 cores across sccl25's 10 L3 slices, so request paths differ; prefetch
behaviour differs between strided and sequential streams; and the mapping from
the 8 DDRC PMU instances to 8 peer channels is assumed, not established
(controller ids skip 1 and 4, with no documented reason).

Progress on this requires establishing what those 8 PMU instances correspond to,
not another mechanism hypothesis. Three hypotheses have now failed on this one
observation.

## Untested, with a falsifiable prediction

Barrier-reset phase alignment. Phase alignment is an unstable equilibrium: after
a few tens of cache lines the workers' phases disperse and nothing restores
them. That is consistent with the A/B probe measuring only +2.38% for the
staggered arm (2.059 vs 2.109 ms median, 15 interleaved rounds), where each
worker streams 8 MiB uninterrupted and therefore aligns only once. That 2.38% is
**not clean evidence**: the staggered arm also moved 2% *more* DRAM bytes
(120.5 vs 118.2 MiB) while being faster, so cache behaviour differs between arms.

Real MoE execution has stage barriers (gather → W13 → W2) and `window_tiles`
boundaries, each of which restarts all workers from segment starts whose phases
are all identical. Prediction: **staggered gain rises monotonically with barrier
density.** Sweep chunk counts of 1/8/32/128/512 in the A/B probe. If the gain
does not rise, the phase hypothesis closes cleanly.

Not attempted: hardware paired measurement of 9-lane `reverse_odd` (predicted
34.116 ms) against the adopted 28-lane `lpt` (predicted 38.826 ms). This is the
only way to decide whether the 13.80% model-internal regret is real. It is two
plans, not a pool.

## Recommended next step

The width fallback in `_select_analytic_full` is the only defect in this
investigation that is confirmed, zero-cost to inspect, and directly affects
plan quality today. Read its introducing commit and rationale before proposing
any change; if the rationale is "wide-team prediction is unreliable", note that
`wide_team_pressure` already addresses the same concern at the scoring layer,
and the second, harder cut is applied on a dimension the model is known to rank
poorly.

Changing it, or changing `relative_uncertainty`, is a production default-dispatch
change and requires hardware validation plus `MATHEMATICAL_MODEL.md`
synchronisation. Do not treat the model-internal 13.80% as a measured gain.

## Artifacts

Local, outside the source tree (ignored workspace):

- `tmp/order_neighborhood_recompute/recompute.py` plus its outputs
  `result_layer4.json`, `result_uniform8T.json`, `result_uniform16T.json`
  (planner recompute runs locally on macOS against the frozen v8 calibration and
  the shipped `IntervalPlanner`; it needs no hardware)
- `tmp/ddr_probe/probe_ddr_interleave.py`, `tmp/ddr_probe/probe_phase_ab.py`
  (scripts only; their outputs live on the remote host)

Remote (`Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/ddr_probe/`), HEAD
`a379373` at query time:

- `probe_ddr_interleave.py` + `interleave_20260906.json`
- `probe_phase_ab.py` + `phase_ab_20260906.json`
- `alias.py` (packed address/stride measurement, prints only)

Probe commands:

```bash
numactl --physcpubind=304 --membind=3 .venv/bin/python \
  tmp/ddr_probe/probe_ddr_interleave.py --region-mib 2048 \
  --target-accesses 1500000 --repeats 2 --output tmp/ddr_probe/interleave_20260906.json

numactl --membind=3 .venv/bin/python tmp/ddr_probe/probe_phase_ab.py \
  --workers 16 --blocks 32 --rounds 15 --output tmp/ddr_probe/phase_ab_20260906.json

PYTHONPATH=src:. numactl --physcpubind=304 --membind=3 .venv/bin/python \
  tmp/ddr_probe/alias.py
```

No SHA256 values are registered for these outputs: they are exploratory probe
results, not frozen experiment artifacts. Re-running the probes is cheaper than
recovering them.
