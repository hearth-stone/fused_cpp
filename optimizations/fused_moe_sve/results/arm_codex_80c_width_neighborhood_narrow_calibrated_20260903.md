# Arm-codex 80C narrow-calibrated width-neighborhood holdout

## Artifact identity

- Run id: `20260902T155347Z-arm_codex_internal_temporal_overhead-arm_width_neighborhood_audit-1fcbc7b130f0`
- Independent high-skew repeat: `20260902T161403Z-arm_codex_internal_temporal_overhead-arm_width_neighborhood_audit-1fcbc7b130f0`
- Source revision: `1fcbc7b130f0e5cdf1fd9160b9695c7b79076be8`
- Suite: `arm_width_neighborhood_audit`
- Machine: `Arm-codex-internal`, NUMA3 CPUs `240-319`, memory node 3
- Calibration: `analytic_machine_numa3_80c_narrow_merge_v8_20260903.json`
- Calibration SHA256: `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`
- Calibration asset-tree SHA256: `23f5f7d222029181b1541d6dab5a01df23c2e340a2bcdb9f85371887fbbc68df`
- Extension SHA256: `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`
- Local artifact: `tmp/moe_paper_runs/<run-id>/`; the same run is retained in
  the configured remote results directory.

The clean snapshot, route asset, calibration asset, ignored i8gemm dependency,
and pinned xbyak submodule all passed exact hash verification. The forced MoE
extension build completed in 5.75 s. Focused target correctness reported
`29 passed, 113 deselected`; every measured candidate also passed exact-output
comparison before timing.

## Configuration

- BF16 fused expert, `H=4096`, `F=512`, 256 experts, 2048 tokens, TopK6.
- Full W13/W2 stage sizes: 8 MiB / 4 MiB; backend N tile: 8.
- Baseline and measured candidates used full owner stripes:
  `(w13_window_tiles,w2_window_tiles,R13,R2)=(0,0,1,1)`.
- Baseline shapes: uniformish `10x8T`; median `1x16T+8x8T`;
  high-skew `4x16T+16x1T`.
- Search: critical and seeded-random sets of 32 experts, 64 candidates per
  operator, complete placed event scoring, canonical deduplication.
- Hardware: five warmups, 31 randomized paired rounds, four rotating packed
  weight copies, strict fixed whole-expert Plan V2, early merge enabled by the
  existing fixed policy, no tail pool/repartition/resize/stealing.

## Main result

| Trace | Baseline median | Stable / measured | Event-hardware Spearman | Selected | Combined regret |
| --- | ---: | ---: | ---: | --- | ---: |
| uniformish | 32.693 ms | 0 / 12 | 0.629 | baseline | 0.386% |
| median | 32.523 ms | 0 / 16 | -0.003 | baseline | 0.637% |
| high-skew | 32.503 ms | 1 / 13 | 0.049 | baseline | 0.983% |

No candidate reached the predeclared 2% robust action margin, so all three
guarded decisions retained their baseline. There is no selected-plan
regression, and every measured shortlist regret is below 1%, comfortably
inside the 5% gate.

The only within-run stable candidate is the model-ranked high-skew width winner: splitting
one 16T lane into two 8T lanes, changing the shape to
`3x16T + 2x8T + 16x1T`. Its predicted event/robust gain is `0.278%`; measured
paired median is `+1.037%`, paired P10 is `+0.334%`, P90 is `+2.576%`, and it
wins 30/31 rounds. Its prediction remains below the 2% acceptance threshold.

The independent high-skew repeat retained the same model ranking and baseline,
but did not reproduce positive P10 for that split: paired median remained
positive at `+1.271%`, while P10 changed to `-0.632%` with 27/31 wins. Combined
shortlist regret was `1.582%`. A different merge was within-run stable at
`+1.437%` median and `+0.433%` P10, but the same merge in the first formal run
had `+0.864%` median and `-0.813%` P10. No candidate therefore has positive P10
in both commit-bound sessions.

## Effect of the narrow-team calibration

In the prefinal v4 run, the two stable high-skew candidates were 1T+1T to 2T
lane merges, predicted at only `0.0197%/0.0130%` and measured at
`+1.436%/+1.248%` median with small positive P10. They did not reproduce as
stable improvements in the commit-bound run:

- the top predicted merge (`+0.140%`) measured `-0.169%` median and
  `-1.470%` P10;
- three other measured merges produced `+0.211%/+0.864%/+0.686%` medians but
  all had negative P10.

Thus the independent calibration improved the visibility of merge candidates
(best predicted merge rose from about `0.020%` to `0.140%`) but also showed
that the original 1.2%--1.4% merge result was not repeatable above the noise
floor. The top split is itself the same state that measured `-0.446%` median in
the prefinal run, `+1.037%` with positive P10 in the first formal run, and
`+1.271%` with negative P10 in the repeat. The calibration increased its
predicted gain from `0.110%` to `0.278%` but did not change the top width-state
identity. This is not closure of point ranking: median/high-skew Spearman
remains weak, and sub-2% hardware deltas are non-stationary across sessions.

## Decision

The Step-3 width neighborhood remains a useful Lab search space because several
width changes show repeated positive medians around 1%--1.5%. It does not pass
a cross-session stable-improvement gate: no candidate has positive P10 in both
formal sessions. The safe incumbent-plus-shortlist decision does pass the
no-regression and regret gates.

Do not enable deterministic width-VND. No robust prediction reaches 2%, the
apparent winners change across sessions, and point ranking remains weak. The
next search work should keep the baseline incumbent and use width operators
only inside a measured/offline shortlist or a larger LNS experiment; it should
not weaken the uncertainty/actionability gate to accept these sub-2% deltas.
