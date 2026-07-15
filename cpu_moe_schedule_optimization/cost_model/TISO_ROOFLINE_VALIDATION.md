# Explainable `T_iso` Roofline Model

## Status

This is a simplified shadow baseline. It does not replace the table/formula
path used by `ContentionCostModel.T_iso`. The newer layered model is documented
in [`GEMM_ECM_VALIDATION.md`](./GEMM_ECM_VALIDATION.md); this document remains
useful for the aggregate FLOP/shared-cache view and historical comparisons.

The implementation and report CLI are in `tiso_roofline.py`.

## Kernel work

For logical route count $R$, define the physical panel multiset:

$$
\mathcal P(R)
=\{\underbrace{12,\ldots,12}_{\lfloor R/12\rfloor}\}
\cup
\begin{cases}
\varnothing,&R\bmod12=0,\\
\{\kappa(R\bmod12)\},&R\bmod12>0,
\end{cases}
$$

where $\kappa(r)$ is the number of rows actually computed by the tail body:

$$
\kappa(r)\in\{2,4,8,12\}.
$$

The exported M1 entry aliases the M2 body, so logical M1 computes two rows but
predicates its store to one row. Logical tails 9--11 execute as one M12 panel.
Packed-A storage separately rounds tails 1--8 to eight rows. The implementation
therefore tracks compute, packed, and store rows independently.

For one physical panel with $m_c$ compute rows and $m_s$ store rows, hidden
size $H$, intermediate size $F$,
and $t$ N-split partitions, let $c_{13}$ be the number of sequential W13 N
ranges: one for no-split and two for the current split-W13 policy. W13 has:

$$
\mathrm{FLOP}_{13}(m_c)=4m_cHF,
$$

$$
Q_{13}(m_c,m_s,t)
=\underbrace{4HF}_{B\text{ weights}}
+\underbrace{2c_{13}tm_cH}_{A\text{ scans}}
+\underbrace{2m_sF}_{\text{BF16 pack-C write}}.
$$

W2 has:

$$
\mathrm{FLOP}_{2}(m_c)=2m_cHF,
$$

$$
Q_2(m_c,m_s,t)
=\underbrace{2HF}_{B\text{ weights}}
+\underbrace{2tm_cF}_{A\text{ scans}}
+\underbrace{4m_sH}_{\text{FP32 down write}}.
$$

The $A$ terms scale with $t$ because every N partition scans packed A. W13 also
scans packed A again for each sequential split range. The aggregate B term does
not scale with $t$ or $c_{13}$: workers and ranges own disjoint N columns, while
their union is one complete weight pass per M panel. This is the current
kernel-level L3/cache traffic convention, not a one-pass tensor-size estimate.

The synthetic `skip_weighted=True` calibration additionally performs gather,
packed-A writes, FP32 down reads, and BF16 output writes:

$$
Q_{\mathrm{aux}}(R,t)
=2RH+2M_{\mathrm{packed}}H+4RH+2RH,
$$

where $M_{\mathrm{packed}}$ includes the eight-row packed storage block for
tails 1--8. Route-index metadata and allocator/control traffic remain in the
fixed residual rather than being assigned speculative byte counts.

## Time formula

Once compute and cache-copy ceilings have been measured independently, the
first explainable formula is:

$$
T_{\mathrm{iso}}(R,t)
=O(t)+T_{\mathrm{aux}}(R,t)
+\sum_{p\in\mathcal P(R)}
\left[T_{13}(p,t)+T_2(p,t)\right],
$$

$$
T_{13}(p,t)
=\max\left(
\frac{\mathrm{FLOP}_{13}(m_c(p))}{P_{13}(t)},
\frac{Q_{13}(m_c(p),m_s(p),t)}{B_{L3}(t)}
\right),
$$

$$
T_2(p,t)
=\max\left(
\frac{\mathrm{FLOP}_2(m_c(p))}{P_2(t)},
\frac{Q_2(m_c(p),m_s(p),t)}{B_{L3}(t)}
\right),
$$

$$
T_{\mathrm{aux}}(R,t)
=\frac{Q_{\mathrm{aux}}(R,t)}{B_{\mathrm{copy}}(t)}.
$$

$P_{13}(t)$ and $P_2(t)$ are effective fused-kernel BF16 ceilings, not ISA
marketing peaks. $B_{L3}(t)$ uses the same kernel-visible traffic convention as
the $Q$ terms. $B_{\mathrm{copy}}(t)$ is measured separately because gather and
scatter do not have the same access pattern as packed GEMM.

Splitting W13 into two N ranges does not change aggregate FLOPs, B bytes, or C
bytes, but it does add another packed-A scan in the current implementation. It
also changes temporal working set, cache residency, and the number of range
launches, so split/no-split require distinct calibrated $O(t)$ and ceiling
parameters.

## Identifiability

One observed panel duration $\Delta T(t)$ determines:

$$
P_{\mathrm{req}}(t)
=\frac{\mathrm{FLOP}_{13}+\mathrm{FLOP}_2}{\Delta T(t)},
$$

and the equivalent memory-bound interpretation:

$$
B_{\mathrm{req}}(t)
=\frac{Q_{13}+Q_2}{\Delta T(t)}.
$$

These are two views of the same latency observation. They are not two
independent fitted hardware ceilings. Infinitely many pairs
$(P(t),B_{L3}(t))$ produce the same maximum, so `T_iso` measurements alone
cannot prove whether a point is compute- or cache-bandwidth-bound.

Consequently, rollout requires independent pure-GEMM $P_{13}/P_2$ and L3
bandwidth measurements. The shadow report intentionally prints
`required_tflops` and `required_l3_gbs` instead of labeling either one as the
measured ceiling.

## Existing-profile observations

The report fits:

$$
T_{\mathrm{iso}}(R,t)
\approx I(t)+\frac{R}{12}\Delta T_{12}(t)
$$

over full M12 points with $R\ge192$. $I(t)$ is only a regression intercept; it
currently combines fixed work, cold-weight effects, and short-run ramp-up, so
it must not yet be equated with $O(t)$.

### 64-core TP2, $H=4096,F=1024$

| Threads | Required TFLOP/s | Required L3 GB/s | Bulk fit median error | Bulk fit max error |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.310 | 26.3 | 0.76% | 1.35% |
| 2 | 0.611 | 52.2 | 0.71% | 2.33% |
| 4 | 1.207 | 105.0 | 0.10% | 1.31% |
| 8 | 2.451 | 220.4 | 0.19% | 1.40% |
| 16 | 4.588 | 439.5 | 0.69% | 3.30% |
| 32 | 7.616 | 818.8 | 1.38% | 5.05% |

### 64-core EP2, $H=4096,F=2048$

| Threads | Required TFLOP/s | Required L3 GB/s | Bulk fit median error | Bulk fit max error |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.318 | 26.8 | 1.17% | 1.80% |
| 2 | 0.624 | 52.8 | 0.24% | 0.74% |
| 4 | 1.231 | 105.1 | 0.28% | 1.01% |
| 8 | 2.402 | 209.0 | 0.25% | 1.93% |
| 16 | 4.902 | 442.4 | 0.15% | 2.34% |
| 32 | 8.666 | 838.5 | 0.48% | 2.77% |

### 8-core, $H=4096,F=512$

| Threads | Required TFLOP/s | Required L3 GB/s | Bulk fit median error | Bulk fit max error |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.262 | 22.5 | 0.25% | 0.62% |
| 2 | 0.422 | 36.9 | 0.39% | 1.59% |
| 4 | 0.758 | 68.4 | 0.80% | 2.81% |
| 8 | 1.438 | 137.7 | 1.28% | 6.65% |

The near-linear steady bulk confirms that a panel service-rate formula is a
reasonable base. It does not validate tails, cold first panels, or the split
range transition.

## Remaining calibration

The scalar roofline cannot represent the instruction/cache hierarchy by itself.
Before replacing active `T_iso`, follow the ECM calibration list in
`GEMM_ECM_VALIDATION.md`; at minimum measure and serialize:

1. pure W13 and W2 effective BF16 throughput for every allowed thread width;
2. L3/cache bandwidth under the same CPU placement and panel access pattern;
3. gather/pack/scatter copy bandwidth;
4. M1/M2/M4/M8 and 1--2 panel startup residuals;
5. split/no-split range-launch and cold-residency residuals.

Reproduce a report with:

```bash
.venv/bin/python \
  cpu_moe_schedule_optimization/cost_model/tiso_roofline.py \
  cpu_moe_schedule_optimization/cost_model/profiles/<profile>.json
```
