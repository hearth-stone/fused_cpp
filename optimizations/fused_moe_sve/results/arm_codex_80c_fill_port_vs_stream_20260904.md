# Arm-codex 80C fill-port vs stream identification

Date: 2026-09-04.

## Decision

Do not add a structure. Overlap is valid and remote stays near zero. On LLC7
logical `48-63`, a 16-thread $M=1$ team matches **one** 1T neighbor, not
sixteen independent 1T fills. Automated `signature` is `inconclusive` only
because `many16 - wide16 = 0.146\,\mathrm{ms}` misses the predeclared
$0.20\,\mathrm{ms}$ gap.

Keep frozen v8, keep holdout unread, and keep VND/LNS closed.

## Question

Aggressor-$M$ occupancy showed the leftover same-LLC tax tracks concurrent
**transfer-bound fills**, not 68-route bytes and not core count. This probe
asks whether a 16-thread team on the same 16 LLC7 cores is **one stream** or
**sixteen fill ports**.

Victim remains expert 0, $M=1$, 1T, logical 64 / CPU304 / LLC7. Every
aggressor is also $M=1$. The 16-core team is logical `48-63` (CPU288-303,
LLC7, excluding the victim). Cross control is logical `0-15` (LLC6). Unused
experts are dependency-delayed on the opposite LLC.

## Predeclared signatures

Head isolated-relative medians, milliseconds. `add_default_off_structure` is
always false.

| Signature | Rule | Session-1 |
| --- | --- | --- |
| fill_ports | $\lvert$wide16 same $-$ many16 1T same$\rvert \le 0.10$ and many16 $\ge 0.40$ | false ($0.146 > 0.10$, many $0.160 < 0.40$) |
| one_stream | $\lvert$wide16 same $-$ one 1T same$\rvert \le 0.10$ and many16 $-$ wide16 $\ge 0.20$ | false on the gap ($0.004 \le 0.10$, $0.146 < 0.20$) |
| invalid_overlap | one/wide `peer_overlap_experts` $< 0.8$, or many16 $< 12.8$ | false (1 / 1 / 16) |
| inconclusive | neither fill_ports nor one_stream, with valid overlap | **selected** |

Remote gate held: wide cross $-0.004$, many cross $+0.070$, both abs $\le 0.08$.

## Command

```bash
taskset -c 240-319 numactl --membind=3 \
  .venv/bin/python optimizations/fused_moe_sve/benchmarks/bench_fill_port_vs_stream.py \
  --analytic-calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --seed 20260911 \
  --trace-dir /tmp/moe_fill_port_vs_stream \
  --output /tmp/moe_fill_port_vs_stream_fit_20260904.json
```

Protocol: 5 warmup, 31 randomized paired rounds, 4 measured packed copies, one
disjoint isolated scrub copy before every sample. Modes share the same
18-expert $M=1$ histogram. Wall clock on Arm-codex-internal was about 10 s
because every expert is $M=1$.

| Artifact | SHA256 |
| --- | --- |
| frozen v8 | `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3` |
| `tmp/moe_fill_port_vs_stream_fit_20260904.json` | `89b6de1517cb1d0e6e5f1920a0fb28659d86dbc38039da0bf50d51c3dde31676` |
| extension | `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f` |

CPU mapping: logical 0 is CPU240 NUMA3 LLC6 (`240-279`); logical 48 is
CPU288 LLC7; logical 64 is CPU304 LLC7 (`280-319`).

## Head results

Isolated-relative medians, milliseconds. Session seed `20260911`, 31 paired
rounds. Isolated target span / W13 is $0.506 / 0.337\,\mathrm{ms}$.

| Mode | same-LLC span | P10 / P90 | cross-LLC span | W13 same | overlap experts | W13 overlap core-ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| one 1T | 0.018 | $-0.031$ / 0.061 | — | 0.351 | 1.00 | 0.39 |
| 1×16T | 0.014 | $-0.034$ / 0.050 | $-0.004$ | 0.363 | 1.00 | 1.08 |
| 16×1T | 0.160 | 0.076 / 0.207 | 0.070 | 0.458 | 16.00 | 10.58 |

## Why this is not sixteen fill ports

1. 1×16T $M=1$ is one expert and one packed-B striped across 16 threads.
   16×1T is sixteen experts and sixteen packed-B streams. The W13 overlap
   core-ms ratio is $1.08 / 10.58 \approx 0.10$, not $\approx 1$.
2. The 16-thread tax equals the far 1T neighbor on core 48
   ($+0.014$ vs $+0.018\,\mathrm{ms}$). It does not equal 16×1T
   ($+0.160\,\mathrm{ms}$).
3. Occupancy leftover n4 $M=1$ was $+0.660\,\mathrm{ms}$ on victim-adjacent
   cores `65-79`. The same 16 independent $M=1$ streams on `48-63` are only
   $+0.160\,\mathrm{ms}$. Same-LLC as a 40-core LLC7 label is coarser than
   the leftover neighborhood.

`one_stream` is the physical direction. The predeclared $0.20\,\mathrm{ms}$
gap was sized for occupancy-scale taxes and cannot fire when 16×1T itself is
only $+0.160\,\mathrm{ms}$.

## Next action

Do not add a default-off structure from this probe. Do not read holdout.
A 16-thread team on one small expert is one packed-B stream. If the leftover
occupancy tax must be localized further, the next hardware question is the
victim-adjacent 15-core set `65-79`, not another $(\beta,\alpha_g)$ fit.
The immediate verification is stream-count composition: 4×4T vs 4×1T vs 16×1T
and `8+8+4+1` vs 4×1T vs 21×1T. See
[arm_codex_80c_stream_count_composition_20260904.md](./arm_codex_80c_stream_count_composition_20260904.md).
