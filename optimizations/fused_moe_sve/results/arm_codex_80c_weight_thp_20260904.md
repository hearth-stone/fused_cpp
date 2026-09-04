# Arm-codex 80C weight THP identification

Date: 2026-09-04.

## Decision

Do not add a structure. Overlap is valid and pages latched. Verified 2 MiB
THP pages on one contiguous W13+W2 block do **not** lower the 16×1T leftover
versus explicit `MADV_NOHUGEPAGE`. Automated `signature` is `page_neutral`.

Keep frozen v8, keep holdout unread, and keep VND/LNS closed.

## Question

Unified-block was `layout_neutral`, but `torch.empty` can silently inherit
THP. This probe copies the **same** contiguous W13+W2 block onto
`mmap+MADV_NOHUGEPAGE` versus `mmap+MADV_HUGEPAGE` and asks whether verified
2 MiB pages lower the 16×1T leftover.

It still cannot collapse sixteen concurrent expert streams into one. Scratch
keeps the process default `FUSED_CPP_PAGES`; only the weight owner changes.
Arm-codex THP is `[always]`; HugeTLB is not reserved. The 4 KiB arm therefore
requires `MADV_NOHUGEPAGE` before first touch.

Victim remains expert 0, $M=1$, 1T, logical 64 / CPU304 / LLC7. Every
aggressor is also $M=1$. The 16-core team is logical `48-63`. Cross control
is logical `0-15` (LLC6).

## Predeclared signatures

Head isolated-relative medians, same page policy versus that policy's
isolated scrub. `add_default_off_structure` is always false.

| Signature | Rule | Session-1 |
| --- | --- | --- |
| thp_helps | many16 small $-$ many16 THP $\ge 0.08$ | false ($0.005$) |
| page_neutral | $\lvert$many16 small $-$ many16 THP$\rvert \le 0.04$ | **selected** ($0.005$) |
| thp_not_latched | 4 KiB AnonHugePages $> 2\,\mathrm{MiB}$, or THP AnonHugePages $< 0.8\times$ request | false ($0$ vs $100\%$) |
| invalid_overlap | wide overlap $< 0.8$ or many overlap $< 12.8$ on either policy | false (1 / 16) |
| inconclusive | neither leftover arm, with valid overlap and latched pages | false |

Remote many16 absolute $\le 0.08$ was reported, not required to name the
signature. Small/THP cross are $+0.145/+0.143$, both above $0.08$. Isolated
absolute spans are $0.577/0.571\,\mathrm{ms}$.

## Command

```bash
taskset -c 240-319 numactl --membind=3 \
  .venv/bin/python optimizations/fused_moe_sve/benchmarks/bench_weight_thp.py \
  --analytic-calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --seed 20260914 \
  --trace-dir /tmp/moe_weight_thp \
  --output /tmp/moe_weight_thp_fit_20260904.json
```

Protocol: 5 warmup, 31 randomized paired rounds, 4 measured mapped copies, one
disjoint isolated scrub copy before every sample, **per page policy**. Modes
share the same 18-expert $M=1$ histogram. Numerical `assert_close` across
policy×mode. Wall clock on Arm-codex-internal was about 11 s.

| Artifact | SHA256 |
| --- | --- |
| frozen v8 | `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3` |
| `tmp/moe_weight_thp_fit_20260904.json` | `da6d1883e28483619bd844b3e56017fcd3350a0abb51561e2b0a75ea4ed9c2c4` |
| extension | `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f` |

CPU mapping: logical 0 is CPU240 NUMA3 LLC6 (`240-279`); logical 48 is
CPU288 LLC7; logical 64 is CPU304 LLC7 (`280-319`). Sysfs THP enabled
`[always]`; defrag `[madvise]`. HugeTLB total $0$. Weight owner is
$216\,\mathrm{MiB}$. smaps: 4 KiB AnonHugePages $=0$; THP AnonHugePages
$=216\,\mathrm{MiB}$ ($100\%$). Both layouts `same_storage=true`.

## Head results

Isolated-relative medians, milliseconds. Session seed `20260914`, 31 paired
rounds.

| Pages | Mode | same-LLC | P10 / P90 | cross-LLC | overlap | W13 overlap core-ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4 KiB | 1×16T | 0.006 | $-0.021$ / $0.031$ | — | 1 | 1.06 |
| THP | 1×16T | 0.008 | $-0.015$ / $0.037$ | — | 1 | 1.11 |
| 4 KiB | 16×1T | 0.167 | $0.111$ / $0.199$ | 0.145 | 16 | 11.66 |
| THP | 16×1T | 0.162 | $0.130$ / $0.208$ | 0.143 | 16 | 11.54 |

Isolated absolute span / W13 is $0.577 / 0.382\,\mathrm{ms}$ (4 KiB) and
$0.571 / 0.373\,\mathrm{ms}$ (THP).

## What this can and cannot show

1. Pages actually differed: 4 KiB had zero huge pages; THP covered the whole
   $216\,\mathrm{MiB}$ owner. This is not the silent-THP confound from v0.73.
2. Sixteen 1T experts still read sixteen packed-B working sets. 2 MiB pages
   do not make them one stream.
3. Isolated span is page-neutral here ($0.006\,\mathrm{ms}$). The historical
   $+13\%$ packed-B bandwidth from 4 KiB to THP was a sequential microbench
   on another host; it does not show up as leftover or isolated span on this
   1-route 1T victim.
4. Remote 16×1T is high in this session ($+0.145$) on **both** page sizes, so
   it is not a THP effect.

## Next action

Leftover identification is closed. Do not add a structure. Do not read
holdout. Handoff:
`cost_model_leftover_identification_handoff_20260904.md`.
