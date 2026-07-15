# SVE BF16 GEMM ECM Shadow Model

## Status

`gemm_ecm.py` is a microkernel-aware shadow model. It does not replace the
active `T_iso` formula or planner scoring. Its purpose is to determine whether
the fused W13/W2 stage time can be represented by reusable machine service
rates instead of a two-dimensional route/thread latency table.

The first validation passes for steady M12 bulk on a clean 32-core slice of the
192-core Neoverse V3 host. The 64-core target host was unreachable during this
run, so the V3 results validate model structure only; they are not target
calibration data.

## Assembly-derived work

Let the SVE vector length be $V$ bytes and the BF16 N tile be
$\nu=V/2$. For a GEMM with padded dimensions $(M,K,N)$, let
$q=N/\nu$ be its number of N tiles. Each physical M panel $p$ carries four
different row counts:

- $m_l(p)$: logical rows requested by the caller;
- $m_c(p)$: rows actually computed by the assembly body;
- $m_p(p)$: rows reserved by packed-A storage;
- $m_s(p)$: rows physically written by the epilogue.

The distinction matters for tails. Logical M1 dispatches the M2 body, reserves
an eight-row packed tail, and predicates the store to one row:

$$
(m_l,m_c,m_p,m_s)=(1,2,8,1).
$$

Logical tails 9--11 are padded to one full M12 panel. For one N tile and one
K4 iteration, an $m_c$ panel executes:

$$
I_{\mathrm{BFMMLA}}^{K4}=2m_c,
\qquad
I_A^{K4}=\frac{m_c}{2},
\qquad
I_B^{K4}=4.
$$

Therefore, over all N tiles:

$$
I_{\mathrm{BFMMLA}}=\frac{m_cKq}{2},
\qquad
I_A=\frac{m_cKq}{8},
\qquad
I_B=Kq.
$$

Each BFMMLA instruction performs $2V$ FLOPs. The exact L1 load traffic is:

$$
Q_{L1,A}=16I_A,
\qquad
Q_{L1,B}=VI_B.
$$

This is different from shared-cache traffic. If sequential range $j$ owns
$q_j$ N tiles, the current conservative cache convention is:

$$
Q_{\mathrm{shared},A}
=2K\sum_p m_c(p)\sum_j\min(t,q_j),
$$

$$
Q_{\mathrm{shared},B}
=2KNP,
$$

where $P$ is the number of physical M panels. Each active N owner brings A once
per range, while all N owners together stream one complete B matrix per M
panel. Later N-tile visits to the same A panel are counted at L1, not again at
the shared-cache layer.

Consequently, splitting W13 into two sequential ranges does not change
BFMMLA count, L1 A/B bytes, total B bytes, or C bytes. It changes the temporal
weight working set, adds one range invocation, and under the conservative
convention doubles shared-cache A scans. A counter study may later show that A
survives in private cache across the range boundary; that would change only
the shared-A term, not the instruction counts.

## ECM formula

For one stage, the model evaluates:

$$
T_{\mathrm{body}}
=\max\left(
T_{\mathrm{BFMMLA}},
T_{\mathrm{frontend}},
T_{L1\text{-load}}+T_{\mathrm{private}}+T_{\mathrm{shared}}
\right),
$$

$$
T_{\mathrm{stage}}
=T_{\mathrm{fixed}}
+cT_{\mathrm{range}}
+T_{\mathrm{body}}
+T_{\mathrm{epilogue}}.
$$

The corresponding service rates are aggregate rates at the stage's active
thread width:

$$
T_{\mathrm{BFMMLA}}=\frac{F_{\mathrm{balanced}}}{P_{\mathrm{BFMMLA}}(t)},
\qquad
T_{L1\text{-load}}=\frac{Q_{L1,\mathrm{balanced}}}{B_{L1}(t)},
$$

$$
T_{\mathrm{private}}=\frac{Q_{\mathrm{private}}}{B_{\mathrm{private}}(t)},
\qquad
T_{\mathrm{shared}}=\frac{Q_{\mathrm{shared}}}{B_{\mathrm{shared}}(t)}.
$$

`balanced` inflates the busiest thread's tile count by the number of active
threads, so non-divisible N partitions retain their load-imbalance cost.
Frontend counts currently include only BFMMLA and A/B load instructions, so
that term is explicitly a lower bound until loop/address/epilogue instruction
counts or PMU data are added.

## V3 structural validation

Configuration:

- host: `AmazonC5192Cores`, Neoverse V3, NUMA0 CPUs `0-31`;
- shape: $H=4096$, $F=1024$, BF16, fused W13 SiLU, FP32 W2 store;
- kernel: SVE M12, `backend_n_tile=8`, W13 split into two N ranges;
- routes: `12,24,48,96,192,384,768,1536,2040`;
- threads: `1,2,4,8,16,32`;
- sampling: 3 warmups, 10 timed runs;
- calibration routes: `192,384,768`;
- fully held-out routes: `1536,2040`.

The same sweep was repeated with `FUSED_CPP_MOE_W13_SKIP_SILU=1` to isolate the
incremental SiLU epilogue cost. The benchmark command was:

```bash
PYTHONPATH=src OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
FUSED_CPP_MOE_W13_SPLIT_N=1 taskset -c 0-31 \
.venv/bin/python \
  cpu_moe_schedule_optimization/benchmarks/profile_moe_stage_breakdown.py \
  --hidden-size 4096 --ffn-hidden-size 1024 \
  --routes 12,24,48,96,192,384,768,1536,2040 \
  --threads 1,2,4,8,16,32 --warmup 3 --runs 10 --fuse-silu
```

### M12 panel service and held-out error

| Stage | Threads | Panel time (us) | Required TFLOP/s | Required L1 GB/s | Required shared GB/s | Held-out median | Held-out max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W13 | 1 | 617.704 | 0.326 | 67.9 | 27.5 | 0.25% | 0.28% |
| W13 | 2 | 324.393 | 0.621 | 129.3 | 53.0 | 0.99% | 1.13% |
| W13 | 4 | 175.610 | 1.146 | 238.8 | 100.2 | 0.91% | 1.02% |
| W13 | 8 | 79.073 | 2.546 | 530.4 | 232.4 | 0.48% | 0.77% |
| W13 | 16 | 41.824 | 4.814 | 1002.8 | 476.9 | 0.27% | 0.41% |
| W13 | 32 | 24.213 | 8.315 | 1732.3 | 953.8 | 1.08% | 1.68% |
| W2 | 1 | 301.692 | 0.334 | 69.5 | 28.5 | 0.46% | 0.55% |
| W2 | 2 | 152.642 | 0.659 | 137.4 | 56.6 | 0.93% | 1.05% |
| W2 | 4 | 87.990 | 1.144 | 238.3 | 98.7 | 1.20% | 1.21% |
| W2 | 8 | 38.159 | 2.638 | 549.6 | 230.1 | 0.31% | 0.39% |
| W2 | 16 | 19.207 | 5.241 | 1091.9 | 467.5 | 0.74% | 0.86% |
| W2 | 32 | 9.811 | 10.261 | 2137.6 | 955.2 | 0.92% | 1.01% |

The panel-linear form is therefore a good description of steady M12 bulk on
this host: all held-out stage errors are below 1.7%. This does not validate
M1/M2/M4/M8 startup or cross-machine rates.

The three `required` columns are equivalent views of the same observed panel
time, not three independently measured ceilings. In particular, the roughly
955 GB/s shared-cache requirement at 32 threads cannot be compared directly
with a roughly 196 GB/s DRAM/STREAM result. Treating the latter as the ECM
shared-cache service rate would overpredict the GEMM time by several times.

### Incremental SiLU cost

| Threads | Extra W13 panel time (us) | Fraction of W13 | ns/output | Identity W13 / W2 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 33.953 | 5.50% | 2.7631 | 1.903 |
| 2 | 15.388 | 4.74% | 1.2523 | 1.950 |
| 4 | 7.041 | 4.01% | 0.5730 | 1.957 |
| 8 | 3.586 | 4.53% | 0.2918 | 1.972 |
| 16 | 1.391 | 3.33% | 0.1132 | 2.107 |
| 32 | 0.938 | 3.87% | 0.0763 | 2.347 |

SiLU contributes only 3.3--5.5% of W13 and about 2.3--3.7% of W13+W2 stage
time in this sweep. Up to 16 threads, identity W13 is close to the expected 2x
W2 body work. The 2.35 ratio at 32 threads is a residual that the final model
must explain with split-range/cache/control counters rather than folding it
into a universal FLOP coefficient.

## Existing 64-core constraint check

For the target TP2 shape $H=4096,F=1024$, one complete W13+W2 M12 panel has:

$$
F_{12}=6\cdot12HF=301.99\ \mathrm{MFLOP}.
$$

The existing target profile reports 974.787 us/panel at one thread and
39.651 us/panel at 32 threads. Independently supplied pure-BFMMLA rates are
403.8 GFLOP/s and 7.5606 TFLOP/s respectively. Their matrix lower bounds are:

$$
T_{\mathrm{matrix}}(1)=747.9\ \mathrm{us},
\qquad
T_{\mathrm{matrix}}(32)=39.94\ \mathrm{us}.
$$

Thus the matrix pipe explains about 77% of one-thread panel time and essentially
all 32-thread bulk time. The 0.7% inversion at 32 threads is within
cross-run/benchmark variation and should not be interpreted as exceeding the
hardware ceiling. This check supports an ECM model: low-thread performance
needs load/frontend/epilogue terms, while high-thread bulk is already close to
the measured BFMMLA constraint.

## Next calibration required

Before planner rollout, collect on the target 64-core host:

1. pure BFMMLA throughput for every planner thread width;
2. L1 load issue and private/shared cache transfer rates with the same tile
   access pattern;
3. SiLU identity/real epilogue pairs for split and no-split W13;
4. PMU instruction and cache-refill counters to test the shared-A scan
   convention;
5. M1/M2/M4/M8 and one/two-panel startup residuals;
6. held-out shapes, especially EP $F=2048$, before replacing active `T_iso`.

Generate the report with:

```bash
.venv/bin/python \
  cpu_moe_schedule_optimization/cost_model/gemm_ecm.py \
  /path/to/silu_stage_profile.json \
  --identity-profile /path/to/identity_stage_profile.json \
  --n-tile 8 --w13-n-ranges 2
```
