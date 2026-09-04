# Arm-codex 80C stream-count composition identification

Date: 2026-09-04.

## Decision

Do not add a structure. Overlap is valid. Leftover tax tracks **concurrent
packed-B streams**, not thread count. Automated `signature` is `stream_count`.
`8+8+4+1` taxes like four 1T streams, not like 21 independent fills.

Keep frozen v8, keep holdout unread, and keep VND/LNS closed.

## Question

Fill-port showed a 16-thread $M=1$ team matches one 1T neighbor, not sixteen
1T fills. This probe asks whether leftover tax therefore tracks stream count
so that `8+8+4+1` equals four streams.

## Predeclared signatures

Head isolated-relative medians. `add_default_off_structure` is always false.
A value is closer to A than B when $|x-A| + 0.02 < |x-B|$.

| Signature | Rule | Session-1 |
| --- | --- | --- |
| stream_count | 4×4T closer to 4×1T than to 16×1T, **and** `8+8+4+1` closer to four mix-start 1T than to 21×1T | **selected** |
| thread_count | 4×4T closer to 16×1T than to 4×1T, **and** `8+8+4+1` closer to 21×1T than to four mix-start 1T | false |
| invalid_overlap | any live mode `peer_overlap_experts` $< 0.8\times$ stream count | false |
| inconclusive | neither arm, with valid overlap | false |

Remote gate was not required to name the signature. Mix cross $+0.042$ holds
the $0.08\,\mathrm{ms}$ bound; 21×1T cross is $+0.211$ and does not.

## Command

```bash
taskset -c 240-319 numactl --membind=3 \
  .venv/bin/python optimizations/fused_moe_sve/benchmarks/bench_stream_count_composition.py \
  --analytic-calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --seed 20260912 \
  --trace-dir /tmp/moe_stream_count_composition \
  --output /tmp/moe_stream_count_composition_fit_20260904.json
```

Protocol: 5 warmup, 31 randomized paired rounds, 4 measured packed copies, one
disjoint isolated scrub copy before every sample. All 23 experts are $M=1$.
Wall clock on Arm-codex-internal was about 14 s.

| Artifact | SHA256 |
| --- | --- |
| frozen v8 | `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3` |
| `tmp/moe_stream_count_composition_fit_20260904.json` | `175d50cb2aa220be3047ea1904de92927e785a9c7233e6d2af534b03200d6490` |
| extension | `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f` |

CPU mapping: logical 0 is CPU240 NUMA3 LLC6 (`240-279`); logical 43 is
CPU283 LLC7; logical 64 is CPU304 LLC7 (`280-319`). Isolated target span /
W13 is $0.609 / 0.406\,\mathrm{ms}$. Overlap equals the live stream count
on every mode.

## Head results

Isolated-relative medians, milliseconds. Session seed `20260912`, 31 paired
rounds.

Equal-thread ladder, cores `48-63`, 16 threads:

| Mode | streams | threads | same-LLC | P10 / P90 | overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1×16T | 1 | 16 | 0.003 | $-0.023$ / 0.028 | 1 |
| 2×8T | 2 | 16 | 0.035 | 0.004 / 0.058 | 2 |
| 4×4T | 4 | 16 | 0.075 | 0.031 / 0.104 | 4 |
| 4×1T | 4 | 4 | 0.070 | 0.041 / 0.115 | 4 |
| 16×1T | 16 | 16 | 0.155 | 0.120 / 0.183 | 16 |

Mix sheet, cores `43-63`:

| Mode | streams | threads | same-LLC | P10 / P90 | cross-LLC | overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8+8+4+1 | 4 | 21 | 0.110 | 0.071 / 0.150 | 0.042 | 4 |
| four mix-start 1T | 4 | 4 | 0.103 | 0.079 / 0.128 | — | 4 |
| 21×1T | 21 | 21 | 0.301 | 0.235 / 0.362 | 0.211 | 21 |

16×1T $+0.155$ replicates fill-port 16×1T $+0.160$ on the same cores.

## Why this is stream count, not thread count

1. At a fixed 16 threads, tax rises with stream count:
   $0.003 \to 0.035 \to 0.075 \to 0.155$ for $1/2/4/16$ streams.
2. 4×4T ($16$ threads) matches 4×1T ($4$ threads): $+0.075$ vs $+0.070$.
   Both are far from 16×1T $+0.155$.
3. `8+8+4+1` ($21$ threads, $4$ streams) matches four mix-start 1T:
   $+0.110$ vs $+0.103$. Both are far from 21×1T $+0.301$.
4. 21 concurrent $M=1$ streams also leak remotely ($+0.211$). Four streams
   do not ($+0.042$). Thread count of the mix is 21, but it does not create
   that rank-wide leak.

A wide team on one $M=1$ expert remains one packed-B stream. Planning
interference should count concurrent transfer-bound experts, not the sum of
team widths. This is not a formula change.

## Next action

Do not add a default-off structure from this probe. Do not read holdout.
Stream count is identified only for transfer-bound $M=1$ peers on this LLC7
slice. Occupancy already showed GEMM-bound $M=68$ streams tax less. Unified
weight-block session-1 is `layout_neutral`: concatenating W13+W2 into one
allocation does not lower the 16×1T leftover. See
`arm_codex_80c_unified_weight_block_20260904.md`.
