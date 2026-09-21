# Arm 80C proxy-grid queue repeat (2026-09-06)

## Scope

Change class: Lab diagnostic. Same frozen v8 paired-PMU grid as
`arm_codex_80c_stream_pressure_proxy_grid_20260904.md`. No formula change,
no holdout reread for adoption, no v8 replacement.

Question: is the historical 8x2T paired queue split (48.17 vs 29.14 cycles) a
one-off, or does the same live-team layout keep moving across sessions?

## Protocol

Arm-codex-internal NUMA3 `numactl --physcpubind=240-319 --membind=3`. Remote
root `/home/zhangxu/codex/fused_cpp`, commit `a3793733…`, extension
`dd554ea3…`, calibration `7928ba96…`. Page-policy environment was empty.
Five warmups, 31 randomized paired rounds, four measured copies plus disjoint
scrub, modes `isolated_head` plus `grid{4,6,8}_{1,2}t_same_head`.

Seeds `20260926/20260927/20260928` ran back-to-back at 14:27–14:28 +08.
Focused tests `test_moe_stream_pressure_pmu_paired.py`,
`test_moe_stream_pressure_pmu.py`, `test_moe_linux_perf_event.py`: 13 passed
before measurement. Every reported cell has 31 samples and PMU
`running_ratio=1.0`.

## Occupancy (not exclusive)

All three new sessions ran while NUMA3 already had other user threads, at least:

- pid `2513165` `bench_meformer_` (13 threads on CPUs in 240–319)
- pid `94000` `tokio-rt-worker` (118 threads on those CPUs)
- `devkit` Java/GC threads and `dbus` on node 3

Loadavg at first snapshot: `1.13 1.18 1.73`. Node 3 had about 264 GiB free of
515 GiB. Historical 20260924/20260925 snapshots were not recorded.

## 8x2T paired queue

Paired value is same-round 8x2T minus isolated, in DDR read-command cycles.

| Session | Isolated queue | 8x2T paired median | P10–P90 | Stdev | Slowdown ms |
| --- | ---: | ---: | --- | ---: | ---: |
| hist 20260924 | 38.09 | **48.17** | 38.8–58.2 | 7.89 | 0.226 |
| hist 20260925 | 32.24 | **29.14** | 26.1–31.3 | 2.31 | 0.153 |
| new 20260926 | 38.21 | **12.58** | 10.3–15.4 | 2.98 | 0.215 |
| new 20260927 | 35.24 | **32.19** | 25.9–42.1 | 7.69 | 0.232 |
| new 20260928 | 35.82 | **27.89** | 23.9–32.9 | 3.78 | 0.308 |

Five-session 8x2T paired medians span **12.58–48.17** (ratio 3.83). Live teams
are identical: starts `48,50,…,62`, width 2. The 48 vs 29 split is not unique;
12.58 is a third regime. Isolated queue does not rank the extra 8x2T pressure:
20260924 and 20260926 both sit near 38 isolated cycles, but paired extras are
48.17 vs 12.58.

Victim slowdown also fails to track queue one-to-one under this occupancy
(20260928 slowdown 0.308 ms at 27.89 cycles vs 20260924 0.226 ms at 48.17).

## Decision

The repeat supports the earlier claim that DDR queue is not a function of
executable plan geometry alone. It does **not** isolate a single interferer:
the node was not exclusive, and no quiet-node control was run. Do not freeze a
queue term, do not treat 12.58 as a new calibration point, and do not reopen
v8 expansion.

A quiet NUMA3 repeat is a separate authorized measurement.

## Artifacts

Remote directory:
`/home/zhangxu/codex/fused_cpp/tmp/moe_stream_proxy_grid_repeat_20260905/`

| File | SHA256 |
| --- | --- |
| `moe_stream_proxy_grid_20260926.json` | `93f97a1585374688ee5510e1362d4945c727687b957863cf2cc7061de8994fe0` |
| `moe_stream_proxy_grid_20260927.json` | `7886d54d4be94e631c849c9b15369eba1a4a0995280fb2c3923b052d1900aa7e` |
| `moe_stream_proxy_grid_20260928.json` | `b73dd5d7bdf48059edfd9776275e31a188d3d06dec2e08008eac083d09a2d963` |

Local copies of the JSON and occupancy snapshots live under
`tmp/moe_stream_proxy_grid_repeat_20260905/` (ignored workspace storage).
