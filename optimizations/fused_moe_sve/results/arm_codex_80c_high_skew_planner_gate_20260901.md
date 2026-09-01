# Arm-codex 80-core analytical-full width gate

Date: 2026-09-01.

## Question

The Amazon 192-core high-skew experiment showed that the useful fixed-width
candidate was already present but ranked incorrectly. Does the same failure
occur on Arm-codex NUMA3, and can a bounded planner rule remove it without
regressing less-skewed captured traces?

The final run is bound to Git revision
`b2196270211b4049c2872350fce1b9188aa77de5`. The measured `_moe_C` SHA256 was
`dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`.
The exact runner id is
`20260901T080803Z-arm_codex_internal-arm_high_skew_closure-b2196270211b`.

## Configuration

- Host: `Arm-codex-internal`, 320 Arm cores, SVE256.
- Placement: NUMA3 CPUs `240-319`, memory node 3.
- Topology: two 40-core, 70 MiB LLC domains; private L2 is 1.25 MiB.
- Shape: BF16 TP4 proxy, `H=4096`, `F=512`, `E=256`, 2048 tokens, TopK6.
- Route output and merge accumulation: FP32.
- Traces: three complete 43-layer route captures; selected layers are
  uniformish request022/layer20, median request016/layer4, and high-skew
  request008/layer38.
- Calibration SHA256:
  `e0ec1cd4ede5dfbdb1ef1748807357292cbad2b1786431642a9934908e42884b`.
- Route-asset tree SHA256:
  `cb263814f56665d6ac0e6af36572b0735d45794e99c95a288b7cae895923e840`.
- Candidates: all 142 analytical strict full shapes plus explicit
  4T/8T/16T x LPT/reverse-odd/reverse-even controls.
- Measurement: 5 warmups, 31 randomized paired rounds, four rotating packed
  weight copies. Every candidate produced bitwise-identical BF16 output.
- Page policy: ordinary THP (`policy=thp`); no HugeTLB mapping was latched.
- Focused runner correctness: 95 passed, 96 deselected.
- Full cold planning: 29.39--50.63 s across the three traces. This remains an
  offline planner/autotuner, not a request-path algorithm.

The runner rendered the following command for each route:

```bash
env OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONPATH=.:src \
  numactl --physcpubind=240-319 --membind=3 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_high_skew_planner_closure.py \
  --route-file <captured-route.pt> --route-layer <layer> \
  --analytic-calibration \
  bench_assets/moe_paper/arm_codex_numa3_80c/analytic_machine_numa3_80c_topology_v2_20260813.json \
  --threads 80 --widths 4,8,16 \
  --orders lpt,reverse_odd,reverse_even \
  --warmup 5 --runs 31 --weight-copies 4 --output <result.json>
```

## Gate

The legacy analytical-full selector chooses minimum expected makespan. The
accepted rule keeps that winner unless all of the following hold:

1. its maximum team width is greater than 8;
2. a candidate capped at the next narrower calibrated width has an uncertainty
   interval overlapping the winner; and
3. at least one such candidate exists.

The gate then selects minimum expected makespan inside that one-step-narrower
set. It cannot jump directly from 32T to 8T, and it does not resurrect the
rejected minimum-working-set tie-break. Quick/request-path, explicit-shape,
empirical, and native planners are unchanged.

## Results

| Trace | Legacy full plan | Legacy | One-step plan | One-step | Paired gain median/P10/P90 | Regret vs measured set |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| high-skew | `(32,32,8,8)`, reverse-even | 42.091 ms | `(16,16,16,16,8,8)`, reverse-even | 36.525 ms | 15.27/14.30/16.00% | 0.00% |
| median | `(32,32,8,8)`, reverse-odd | 40.643 ms | `(16,16,16,16,8,8)`, reverse-odd | 34.433 ms | 17.95/17.42/18.76% | 0.26% |
| uniformish | `(16,16,8,8,8,8,8,8)`, reverse-even | 33.859 ms | `10x8T`, reverse-even | 31.987 ms | 5.69/4.23/7.29% | 0.00% |

The gate passes the declared <=5% measured-regret and <=2% held-out-regression
criteria on this three-trace corpus. The manual fixed-width/order model rank
Spearman values are 1.000/0.855/0.158 for high-skew/median/uniformish. Thus the
gate closes selection on the measured corpus but does not establish accurate
temporal-order ranking or absolute-time prediction.

## Interpretation and limits

The error is not missing candidate coverage. The expected winner beats the
one-step plan by only 0.6--2.0% in prediction, inside the calibration's 15%
systematic uncertainty, yet uses wider mixed teams and loses 5.4--22.6% in
paired execution. The current DAG still does not carry physical core intervals
into its two-LLC-domain service calculation, so this evidence supports a
conservative planner gate rather than a fitted route-pair correction.

Remaining gates are a larger multi-layer/request corpus and a second Arm
machine. The result does not close production quick planner quality or the
general analytical contention/regret target.
