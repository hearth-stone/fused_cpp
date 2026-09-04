# Arm-codex 80C unified weight-block identification

Date: 2026-09-04.

## Decision

Do not add a structure. Overlap is valid. Copying one layer's W13 and W2 into
**one anonymous DRAM allocation** does not lower the 16×1T leftover versus
the current two packed tensors. Automated `signature` is `layout_neutral`.

Keep frozen v8, keep holdout unread, and keep VND/LNS closed.

## Question

Stream-count showed leftover tax tracks concurrent packed-B streams, not
thread count. Production packing already stores every expert of one matrix in
one contiguous `[E, packed]` tensor. This probe asks whether copying one
layer's W13 and W2 into **one anonymous DRAM allocation** lowers the 16×1T
leftover versus the current two packed tensors.

It cannot collapse sixteen concurrent expert streams into one stream. The
kernel still issues sixteen independent 1T experts. Arm-codex has no reserved
HugeTLB pool (`HugePages_Total: 0`); the unified owner is one ordinary
anonymous `torch.empty`.

Victim remains expert 0, $M=1$, 1T, logical 64 / CPU304 / LLC7. Every
aggressor is also $M=1$. The 16-core team is logical `48-63`. Cross control
is logical `0-15` (LLC6).

## Predeclared signatures

Head isolated-relative medians, same layout versus that layout's isolated
scrub. `add_default_off_structure` is always false.

| Signature | Rule | Session-1 |
| --- | --- | --- |
| unified_helps | many16 split $-$ many16 unified $\ge 0.08$ | false ($-0.004$) |
| layout_neutral | $\lvert$many16 split $-$ many16 unified$\rvert \le 0.04$ | **selected** ($0.004$) |
| invalid_overlap | wide overlap $< 0.8$ or many overlap $< 12.8$ on either layout | false (1 / 16) |
| inconclusive | neither arm, with valid overlap | false |

Remote many16 absolute $\le 0.08$ was reported, not required to name the
signature. Split/unified cross are $+0.088/+0.098$, both slightly above
$0.08$.

## Command

```bash
taskset -c 240-319 numactl --membind=3 \
  .venv/bin/python optimizations/fused_moe_sve/benchmarks/bench_unified_weight_block.py \
  --analytic-calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --seed 20260913 \
  --trace-dir /tmp/moe_unified_weight_block \
  --output /tmp/moe_unified_weight_block_fit_20260904.json
```

Protocol: 5 warmup, 31 randomized paired rounds, 4 measured packed copies, one
disjoint isolated scrub copy before every sample, **per layout**. Modes share
the same 18-expert $M=1$ histogram. Numerical `assert_close` across
layout×mode. Wall clock on Arm-codex-internal was about 11 s.

| Artifact | SHA256 |
| --- | --- |
| frozen v8 | `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3` |
| `tmp/moe_unified_weight_block_fit_20260904.json` | `7bd2319f49178bdc46f62abb6d7de016af2ff1ba5fe88bdd927a5be47b3ebc32` |
| extension | `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f` |

CPU mapping: logical 0 is CPU240 NUMA3 LLC6 (`240-279`); logical 48 is
CPU288 LLC7; logical 64 is CPU304 LLC7 (`280-319`). Isolated target span /
W13 is $0.466 / 0.315\,\mathrm{ms}$ (split) and $0.477 / 0.319\,\mathrm{ms}$
(unified). Unified storage is one owner (`same_storage=true`); W13 is
$144\,\mathrm{MiB}$, W2 $72\,\mathrm{MiB}$, pad $0$. Split uses two
storages.

## Head results

Isolated-relative medians, milliseconds. Session seed `20260913`, 31 paired
rounds.

| Layout | Mode | same-LLC | P10 / P90 | cross-LLC | overlap | W13 overlap core-ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| split | 1×16T | 0.041 | $0.018$ / $0.059$ | — | 1 | 1.06 |
| unified | 1×16T | 0.042 | $0.013$ / $0.066$ | — | 1 | 1.08 |
| split | 16×1T | 0.153 | $0.122$ / $0.177$ | 0.088 | 16 | 9.59 |
| unified | 16×1T | 0.157 | $0.133$ / $0.187$ | 0.098 | 16 | 9.99 |

## What this can and cannot show

1. Split already has all experts of W13 in one tensor and all experts of W2
   in another. Unified only concatenates those two matrices.
2. Sixteen 1T experts still read sixteen packed-B working sets. One storage
   owner does not make them one stream. The 16×1T leftover stays
   $+0.15\,\mathrm{ms}$ on both layouts; 1×16T stays $+0.04\,\mathrm{ms}$.
3. `layout_neutral` means the leftover is not coming from W13/W2 being two
   allocations. Putting the layer in one DRAM block does not replace counting
   concurrent transfer-bound experts.

## Next action

Do not add a structure from this probe. Do not read holdout. The leftover is
not coming from W13/W2 being two allocations. Weight-THP session-1 is
`page_neutral`: verified 2 MiB pages do not lower the 16×1T leftover versus
explicit `MADV_NOHUGEPAGE`. See `arm_codex_80c_weight_thp_20260904.md`.
