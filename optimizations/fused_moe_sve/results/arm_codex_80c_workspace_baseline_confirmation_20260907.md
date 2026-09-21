# Workspace baseline, anchor confirmation, model reassessment and smoothing

All four requested steps completed in order. No production-default changes,
model fitting or new physical term. Experimental anchors are registered under
the new lifecycle; old winner artifacts remain unchanged.

## 1. Explicit lifecycle baseline

Authoritative profile: `optimizations/fused_moe_sve/profiles/workspace_numa3_80c.json`.
Usage/policy: `workspace_experiment_baseline.md` in this results directory.
New confirmations pass `--experiment-baseline <profile>`, binding extension,
calibration, workspace implementation and protocol. Runner records actual packed
Ntile16, output_lifecycle and baseline SHA256. Analysis rejects mixed session
identities and a mismatch with the profile-bound frozen frontier. Old allocation
and pre-profile workspace data keep their original labels and are not silently
merged into new pairs. Legacy CLI/production defaults are not changed.

## 2. Common-session anchor/elite confirmation

Exact prior plans only; complete-bridge dedup gives median3,high-skew6,uniformish5.
Each runs two independent sessions before any smoothing confirmation. Gate stays
>2% paired median gain and positive P10 in both sessions versus the retained
strong anchor. Among eligible improvements choose minimum mean session latency;
retain other near candidates without claiming a stable order between them.

| Trace / retained candidate | Anchor latency ms S1/S2 | Candidate ms S1/S2 | Paired gain % S1/S2 | P10 % S1/S2 |
| --- | ---: | ---: | ---: | ---: |
| median2ab43572 | 30.640/30.797 | 29.994/30.106 | 2.568/2.426 | 1.114/1.374 |
| high-skew189d70 | 30.142/30.220 | 29.193/29.042 | 3.083/4.026 | 0.581/1.986 |
| high-skew61ea6f | 30.142/30.220 | 29.211/29.134 | 3.272/3.759 | 1.059/1.699 |

Median anchor becomes2ab43572; block relocation remains a near reference.
High-skew uses189d70 as representative anchor and retains61ea6f as elite;
both beat the previous GEMM-density anchor, but no stable superiority between
these two elites is claimed. Uniformish keeps its existing anchor: best retained
LNS sample gains1.651/1.361%, P10=0.521/-0.514%, insufficient for promotion.

Weak-parent evidence is separate. High-skew greedy insertion gains4.645/3.118%,
P10=2.505/1.917%, but absolute time33.094/33.183ms is far slower than the
confirmed strong anchor. Uniformish LNS sample gains11.118/9.882% versus its weak
parent, not versus full. Retain VND for weak-start repair and LNS for elite search;
do not conflate either comparison with global optimality.

Experimental registry: `tmp/workspace_baseline_confirmation_20260907/registry.json`,
which references three full decision artifacts. They retain full executable
bridges, evidence and `partial_order_authorized=false`, `production_adopted=false`.

Confirmed bridge identities:

- median2ab43572: `1db2776f8cbfb56ef823b87222c140ae0886b05e48a65751760fb8d44c74369d`.
- high-skew189d70: `fdc523c1891b1258eef11ecc63c8e74ec8c7aa172c07400732553a168c344845`.
- uniformish retained: `86b4b7a44eb4f415eecd057d5e6a82ce3487e17511692fb769972fa26d218f8a`.

## 3. Reassess without fitting

Completed and recorded in `workspace_model_reassessment_20260907.md` before
launching step4. On these selected confirmation pools, frozen-v8 MAPE is
13.913/19.823/3.160% for median/high-skew/uniformish. Relative-gain MAE is
3.944/11.061/1.275 percentage points. These are diagnostics on different candidate
mixes, not unbiased accuracy estimates or a comparison to previous larger pools.

No new physical parameter is warranted from these aggregates. Old total residuals
must not become a bandwidth term. The existing partial-order residual report is
not authorized for workspace hard pruning/acceptance; the median confirmed elite
still contradicts its old candidate_worse label. Any future calibration must use
separate fit/evaluation data and a matching lifecycle identity.

## 4. Independent median smoothing confirmation

Only after steps2/3 completed, freeze four plans: newly confirmed2ab43572 as
anchor, old anchor, smooth candidate, block reference. Two fresh sessions use
seeds20260909/10. No candidate was regenerated, and no automatic adoption follows.

| Plan | Latency ms S1/S2 | Gain vs confirmed anchor % S1/S2 | P10 % S1/S2 |
| --- | ---: | ---: | ---: |
| confirmed2ab43572 | 30.070/29.664 | 0/0 | 0/0 |
| old anchor | 30.810/30.692 | -2.379/-2.957 | -3.896/-4.211 |
| smoothing | 29.991/29.912 | +0.034/-0.344 | -1.669/-2.331 |
| block reference | 30.319/30.205 | -0.904/-1.147 | -2.643/-2.791 |

Repaired relative to the old anchor, smoothing gains2.586/2.474%, with positive
P10=1.513/0.728%. Its old-anchor signal is confirmed in these independent sessions.
But it does not beat the stronger newly confirmed elite, so it is retained as a
near reference and is not adopted. The experimental anchor registry is unchanged
by smoothing; `smoothing_adopted=false` is explicit. Earlier inconsistent repeats
remain part of the evidence rather than being overwritten.

## Implementation and evidence

New assembler and finalizer are Lab-only. Runner/profile checks and analysis
identity checks are additive. All plans are exact copies from prior frozen
frontiers; sources/bridge hashes are checked.30 focused tests passed for baseline
mixing rejection, bridge dedup, confirmation, workspace ownership/capacity and
existing runner behavior. Ruff, YAML/profile parsing and diff checks pass.

Eight target sessions: six for anchors, then two for smoothing. Every plan x4
copies passes NaN-poisoned workspace output equality against the allocation-mode
reference.5 warmups+31 effective paired rounds,4-copy rotation,216MiB scrub,
phase tracing OFF. E256/H4096/F512,2048tokens,TopK6,BF16,Ntile16,full stripes
`(t,0,0,1,1)`,W13/W2 bytes8MiB/4MiB,early merge unchanged. NUMA3 CPUs240–319,
membind3.192MiB fixed workspace initialized/touched once per process; initialization
is recorded separately. THP policy checked as `always`, actual backing not audited.
No new PMU/fault measurement; no external competing benchmark found at start.

Frozen extension `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`,
v8 calibration `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`.
No rebuild, growth, production dependency, API or default dispatch changes.

Local directory: `tmp/workspace_baseline_confirmation_20260907/` contains three
anchor frontiers, six raw sessions, three decisions, registry, smoothing frontier,
two smoothing sessions and summary. Remote raw/code/profile copies remain at
`Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/workspace_baseline_confirmation_20260907/`.
The local final registry/decisions are the handoff source; old result directories
are untouched. No commit created.

Reproduction:

1. `assemble_workspace_confirmation.py --stage anchors --output-dir <fresh-dir>`.
2. Run bounded `measure` for each frontier with `--experiment-baseline <profile>`,
   existing NUMA/OpenMP protocol, seeds20260907/08; finalize with
   `PYTHONPATH=.:src .../finalize_workspace_confirmation.py --frontier <file>
   --sessions <s1> <s2> --output <decision>`.
3. Recompute point-error diagnostics without fitting or opening a new probe.
4. Assemble `--stage smoothing --confirmation-dir <completed-dir> --output-dir
   <fresh-dir>`; run seeds20260909/10 and use bounded analysis. Compare against
   the confirmed anchor, with old-anchor comparison reported separately.

Further work is separate: independent workspace-specific calibration design,
explicit native/production integration if requested, and low-priority capacity
growth. This turn does not authorize automatic reuse of old pruning radii.
