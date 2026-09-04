# Arm 80C stream-pressure PMU attribution

Date: 2026-09-04

## Technical summary

The short same-LLC penalty is associated with a saturating DDR read-command
latency and victim-side LLC-miss/backend-stall response, not with saturation of
aggregate DRAM bytes or bandwidth. From isolated to 1, 4, and 16 distinct
transfer-bound packed-B peers, DDR read-command latency increased from 32.6 to
43.9, 55.8, and 57.0 cycles while the victim span increased by 0, 0.040, 0.150,
and 0.227 ms. In contrast, aggregate DRAM read traffic continued from 10.0 to
21.5, 37.0, and 108.3 MiB/call and estimated read bandwidth continued from 5.8
to 12.1, 19.0, and 42.0 GB/s.

This identifies the pressure source but does not identify a production formula.
The 4x4T and 4x1T cells have nearly identical DRAM traffic but different
victim-side LLC miss ratios and a 0.02--0.06 ms cross-session span difference.
Do not reduce the result to stream count alone, fit a residual, read holdout, or
change frozen v8.

## Joint session result

The table uses session 3 (`seed=20260917`), where CPU304 core PMU, all ten LLC7
L3C slices, all eight NUMA3 DDRC PMUs, and the native target trace were enabled
and disabled around the same 31 measured calls.

| Cell | Distinct peer B | Peer threads | Victim span ms | Delta ms | DDR read MiB/call | DDR read GB/s | DDR read latency cycles | Victim LL-read miss ratio | Victim backend-stall ratio | LLC7 L3 hit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| isolated | 0 | 0 | 0.477 | 0.000 | 10.0 | 5.8 | 32.6 | 5.3% | 70.6% | 91.7% |
| 1x16T | 1 | 16 | 0.517 | +0.040 | 21.5 | 12.1 | 43.9 | 13.3% | 71.3% | 54.2% |
| 4x4T | 4 | 16 | 0.627 | +0.150 | 37.0 | 19.0 | 55.8 | 33.8% | 74.3% | 23.4% |
| 4x1T | 4 | 4 | 0.644 | +0.167 | 37.5 | 19.0 | 49.2 | 55.9% | 75.7% | 43.4% |
| 16x1T | 16 | 16 | 0.704 | +0.227 | 108.3 | 42.0 | 57.0 | 52.7% | 75.6% | 9.2% |

`DDR read MiB/call = 32 * sum(flux_rd) / 31 / 2^20`. The 32-byte scale is the
host perf metric definition, `flux_rd * 32 / duration_time`. DDR read latency is
`sum(read_cmd_occupancy) / sum(read_cmd)` across the eight NUMA3 controllers.
L3 hit rate is `sum(l3c_hit) / sum(l3c_ref)` across LLC7 slices 0--9. Core ratios
are from system-wide CPU304 counts within the controlled interval.

## What the counters establish

1. Aggregate bytes are not the saturating variable. Between 4x1T and 16x1T,
   DRAM read traffic increased by 70.8 MiB/call and bandwidth by 22.9 GB/s, but
   the victim span increased by only 0.061 ms in session 3 and 0.014 ms in
   session 2.
2. DDR request latency has the right knee. It rose by 23.2 cycles from isolated
   to 4x4T, then only another 1.3 cycles at 16x1T. Victim backend-stall ratio
   similarly reached 74.3% at 4x4T and 75.6% at 16x1T.
3. Distinct packed-B blocks create real cache pressure. Victim LL-read miss ratio
   increased from 5.3% isolated to 13.3% with one peer block and 33.8--55.9%
   with four blocks. Aggregate LLC7 L3 hit rate fell further to 9.2% at 16x1T.
4. Stream count is not a complete quantitative variable. The 4x4T and 4x1T
   cells both read about 37 MiB/call, but the victim LL-read miss ratio was
   33.8% versus 55.9%. Team layout/request timing still changes which misses
   become victim-visible stalls.

The supported diagnosis is therefore: distinct transfer-bound packed-B fills
increase LLC misses and DDR controller queue latency seen by the short victim;
that latency/stall component saturates near four streams even though total peer
traffic continues to rise.

## Experimental design

- Machine: `Arm-codex-internal`, 320-core HiSilicon AArch64.
- Affinity: `taskset -c 240-319`, `numactl --membind=3`.
- Victim: expert 0, M=1, 1T, logical 64 / CPU304 / LLC7.
- Peer cells: isolated, 1x16T, 4x4T, 4x1T, and 16x1T on logical 48--63.
- Shape: hidden 4096, intermediate 512, BF16, fused-SiLU, Arm SVE BF16.
- Protocol: 5 warmups, 31 measured calls, four rotating measured packed copies.
- Scrub: a fifth disjoint packed copy; 18 one-route experts touch all 18 W13/W2
  blocks before every sample while PMUs and native trace are disabled.
- Unlike the earlier stream-composition probe, the PMU plan contains only the
  victim and active peers. It has no dependency-delayed post-victim background.
- `perf stat --delay=-1 --control=fifo:ctl,ack` restricts both nested perf
  sessions to the measured cell.
- Core events: cycles, instructions, stall_backend, mem_access,
  l2d_cache_refill, ll_cache_rd, ll_cache_miss_rd, dtlb_walk on CPU304.
- Uncore events: l3c_ref/l3c_hit on `hisi_sccl25_l3c0..9`; flux_rd/flux_wr and
  read_cmd/read_cmd_occupancy on all eight `hisi_sccl25_ddrc*` PMUs.

Every event reported 100% time running. Native trace overlap medians were
0/1/4/4/16 experts for the five cells.

## Robustness and limitations

| Cell | Session 2 ms | Session 3 ms | Session difference ms |
| --- | ---: | ---: | ---: |
| isolated | 0.482 | 0.477 | -0.005 |
| 1x16T | 0.519 | 0.517 | -0.002 |
| 4x4T | 0.630 | 0.627 | -0.002 |
| 4x1T | 0.691 | 0.644 | -0.048 |
| 16x1T | 0.705 | 0.704 | -0.001 |

The main knee is stable, but 4x1T is noisier than the other cells. A future
quantitative formula must include an independent repeat of 4x1T and must not
fit its current point estimate as exact.

Uncore counters cover the complete fused call rather than ending at the target
W13 timestamp. All peers are M=1 and overlap the victim, which minimizes but
does not eliminate peer tail traffic. CPU304 counters also include dispatch and
counter-control work on that CPU during the enabled call. Use ratios and
controlled contrasts; do not interpret core cycles as the native victim span.

L3C event semantics are used as relative references/hits. They are not converted
to bytes because the platform event definition does not state that one count is
one 64-byte cache line. DDR flux is converted only with perf's explicit 32-byte
metric definition.

The first split-pass session is retained only as a methodology failure check:
core and uncore were measured in different invocations and cell latency drifted.
It must not be used for counter-to-latency fitting.

## Decision and next quantitative gate

Do not add a cost-model structure from these five cells. The next model-facing
probe, if requested, should sweep distinct transfer-bound blocks at
`0/1/2/4/8/16`, repeat 4x1T, and include same/cross-LLC controls. Freeze a
single-variable candidate only if leave-one-count-out validation shows that a
queue/miss pressure feature predicts victim slowdown across both team layouts
and an independent session. Keep route sweeps and real traces as unread holdout.

This gate was executed later on 2026-09-04. Neither feature passed; DDRC queue
latency is preferred only for a paired-process follow-up. See
`arm_codex_80c_stream_pressure_count_loco_20260904.md`.

## Reproduction and artifacts

Probe:
`optimizations/fused_moe_sve/benchmarks/bench_stream_pressure_pmu.py`

Representative joint command for one cell (repeat with each mode listed in the
design section):

```bash
outer=$(mktemp -d /tmp/moe-pmu-outer.XXXXXX)
inner=$(mktemp -d /tmp/moe-pmu-inner.XXXXXX)
mkfifo "$outer/ctl" "$outer/ack" "$inner/ctl" "$inner/ack"

perf stat -x, --no-big-num -o /tmp/uncore.csv \
  --delay=-1 --control=fifo:"$outer/ctl","$outer/ack" -a \
  -e <all-hisi_sccl25-l3c-and-ddrc-events> -- \
perf stat -x, --no-big-num -o /tmp/core.csv \
  --delay=-1 --control=fifo:"$inner/ctl","$inner/ack" -a -C 304 \
  -e cycles,instructions,stall_backend,mem_access,l2d_cache_refill,ll_cache_rd,ll_cache_miss_rd,dtlb_walk -- \
taskset -c 240-319 numactl --membind=3 \
  .venv/bin/python optimizations/fused_moe_sve/benchmarks/bench_stream_pressure_pmu.py \
  --analytic-calibration \
    bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --mode four_4t_same_head --warmup 5 --runs 31 --weight-copies 4 \
  --seed 20260917 --trace /tmp/four_4t.log --output /tmp/four_4t.json \
  --perf-control "$outer/ctl" --perf-ack "$outer/ack" \
  --perf-control "$inner/ctl" --perf-ack "$inner/ack"
```

The uncore placeholder expands `l3c_ref/l3c_hit` for
`hisi_sccl25_l3c0..9`, plus `flux_rd/flux_wr/read_cmd/read_cmd_occupancy` for
`hisi_sccl25_ddrc{0_0,0_1,2_0,2_1,3_0,3_1,5_0,5_1}`.

Frozen calibration:
`bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json`
(`7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`).

Extension SHA256:
`dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`.

Local ignored artifacts:

- `tmp/moe_stream_pressure_pmu_s2_joint`; `SHA256SUMS` SHA256
  `5880cc6b2c2d1ed9077369c0709f7e6e0a05c2fb7d4e1f25f3e28cd0a85754df`.
- `tmp/moe_stream_pressure_pmu_s3_latency`; `SHA256SUMS` SHA256
  `15f4f352c2f8065fa69394a897514de3509cab41eda7c36256c8c445ed3dc0c2`.

No holdout artifact was opened.
