# Arm-codex 80C small-expert context decomposition

## Evidence status

This is a direct-sync provisional Lab diagnostic, not a clean-commit paper run.
It adds no production behavior or calibration parameter.

- Machine: `Arm-codex-internal`, NUMA3 CPUs `240-319`, memory node 3.
- Calibration: frozen `analytic_machine_numa3_80c_narrow_merge_v8_20260903.json`.
- Calibration SHA256: `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`.
- Extension SHA256: `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`.
- BF16 fused expert, `H=4096`, `F=512`, 80 threads, full owner-stripe
  W13/W2 windows, one fixed 1-route 1T target on logical core 64.
- Five warmups, 31 randomized paired phase-trace rounds, four rotating measured
  packed copies. The final protocol adds one dedicated disjoint 660 MiB scrub
  copy before every sample; scrub execution is synchronous, untraced, and
  excluded from the target span. A second process repeated each protocol with
  seed `20260904`.
- All modes execute identical experts, routes, weights, and total work. A
  disabled background's first tasks depend on completion of the target lane;
  this removes that background from the target interval without changing the
  output or total work.

Raw ignored-workspace artifacts:

| Artifact | SHA256 |
| --- | --- |
| `tmp/moe_small_expert_context_20260903.json` | `a2a63d11034d20c5e5bc0493737cbf25d5eaefafd9fb60589073bbfb219e62d7` |
| `tmp/moe_small_expert_context_repeat_20260903.json` | `34db10bf0d22982c9d624811239076bdeb82eacd03deaf74242522986f4d9889` |
| `tmp/moe_small_expert_context_scrub_20260903.json` | `61c0e929ad7a575831f972bdbb2c690df243e028062cdbb10044e66d15867e34` |
| `tmp/moe_small_expert_context_scrub_repeat_20260903.json` | `90f34a8498a2a3ad1abfdae69b21ff4b9ab97f08cef1192dcc62c23bf8c07d4f` |
| `tmp/moe_small_expert_context_scrub_m2_20260903.json` | `be6b83475360c330ba4e053c90f19fc162465228eef835661b692fac569ca1a1` |
| `tmp/moe_small_expert_context_scrub_m5_20260903.json` | `277d3896e5d04bafedbf931edba38c0ff29ea6a6b9ad542ad3ff5d78d261df74` |
| `tmp/moe_small_expert_context_scrub_m6_20260903.json` | `28cf59e56d6c281df3245fcc941e19885496ec6e7df9bb4b7f3634d59e1b4376` |
| `tmp/moe_small_expert_context_scrub_m12_20260903.json` | `81c195d34902e0d9002a59f85d0c8bccfc884a6395bacb22031aa69a5c01b0dd` |

## Controlled modes

The target expert and its fixed 68-route same-lane companion are present in
every mode.

| Mode | Target position | Background active during target |
| --- | --- | --- |
| `full_head` | lane head | 4x16T plus 15x1T |
| `full_after_68` | after 68-route predecessor | 4x16T plus 15x1T |
| `wide_only_head` | lane head | 4x16T only |
| `narrow_only_head` | lane head | 15x1T only |
| `isolated_head` | lane head | none |

## Pre-scrub diagnostic results

Target expert span medians in the first/repeat sessions:

| Mode | First | Repeat | Frozen-v8 model |
| --- | ---: | ---: | ---: |
| isolated head | 0.665 ms | 0.603 ms | 0.524 ms |
| 1T background only | 0.938 ms | 0.913 ms | 0.524 ms |
| 16T background only | 1.201 ms | 1.200 ms | 0.636 ms |
| full cohort head | 1.482 ms | 1.475 ms | 0.613 ms |
| full cohort after 68 routes | 0.918 ms | 0.926 ms | 0.713 ms |

Paired median effects in the first/repeat sessions:

| Contrast | First | Repeat | Directional range |
| --- | ---: | ---: | --- |
| full head minus isolated | +0.823 ms | +0.881 ms | P10 remains +0.783/+0.821 ms |
| 16T-only minus isolated | +0.546 ms | +0.599 ms | P10 remains +0.499/+0.571 ms |
| 1T-only minus isolated | +0.272 ms | +0.311 ms | P10 remains +0.237/+0.283 ms |
| full interaction beyond additive effects | -0.011 ms | -0.037 ms | intervals include or approach zero |
| after-68 minus full head | -0.557 ms | -0.545 ms | P90 remains -0.528/-0.512 ms |

The full-cohort penalty is therefore approximately additive. In the first run,
the 16T background accounts for about two thirds of the `0.823 ms` excess and
the 1T peers account for about one third; the repeat has the same split. The
interaction term is negligible relative to either main effect.

The head-position effect is concentrated in W13. Moving the same target behind
the 68-route predecessor changes median W13 from `1.135/1.147 ms` to
`0.614/0.620 ms`, while gather stays near `0.009 ms` and W2 changes much less.
This supports a deterministic cold-W13 phase-alignment/service effect: at lane
head the target joins the initial burst of concurrent weight streams; after the
predecessor it enters after that burst has evolved. It is not consistent with
random measurement noise.

## Strict-DRAM scrub protocol and results

The final benchmark retains the four-copy measured rotation and allocates a
fifth packed copy exclusively for cache scrubbing. Before every warmup or
measured sample it runs the complete `full_head` plan on the scrub copy with
tracing disabled. The synchronous plan return is the barrier; only then is the
selected mode executed on its rotating measured copy with tracing enabled.
Thus every sample is preceded by about 660 MiB of disjoint packed-weight
traffic, more than four times the rank LLC capacity, without adding scrub time
or records to the target span.

Target span medians in the first/repeat scrub sessions:

| Mode | First | Repeat | Frozen-v8 model |
| --- | ---: | ---: | ---: |
| isolated head | 0.633 ms | 0.640 ms | 0.524 ms |
| 1T background only | 0.907 ms | 0.908 ms | 0.524 ms |
| 16T background only | 1.193 ms | 1.199 ms | 0.636 ms |
| full cohort head | 1.445 ms | 1.453 ms | 0.613 ms |
| full cohort after 68 routes | 0.921 ms | 0.924 ms | 0.713 ms |

Paired effects remain stable under scrub:

| Contrast | First | Repeat |
| --- | ---: | ---: |
| full head minus isolated | +0.811 ms | +0.819 ms |
| 16T-only minus isolated | +0.558 ms | +0.559 ms |
| 1T-only minus isolated | +0.272 ms | +0.270 ms |
| full interaction beyond additive effects | -0.022 ms | -0.022 ms |
| after-68 minus full head | -0.523 ms | -0.531 ms |

Every main-effect P10 remains positive and every after-68 P90 remains negative.
The scrub protocol therefore preserves the original decomposition while
removing copy-residency as an explanation.

An Arm SPE audit filtered samples by the printed virtual address ranges of the
measured target weights. Before explicit scrub, target W13 had 310 L3 misses
and 23 hits in the sampled high-latency loads. With scrub it had 356 misses and
6 hits; sampled W13 hit incidence fell from about 6.9% to 1.7%. Scrubbed target
W2 had 242 misses and 7 hits. SPE used its default 30-cycle minimum-latency
filter, so these are a cold-source gate rather than unbiased byte fractions.
The raw scrub audit remains on `Arm-codex-internal` as
`/tmp/perf_small_expert_scrub_addr.data`.

## Model diagnosis and decision

Frozen v8 predicts `narrow_only_head` identically to `isolated_head` at
`0.524 ms`, missing the measured 1T-peer main effect. It predicts full-head at
`0.613 ms`, below even its `wide_only_head` prediction, while scrubbed hardware
measures `1.445--1.453 ms`. The full-head model error remains about `-58%`.
The existing narrow full-cohort correction, fitted on longer target spans, does
not transfer to a one-route lane-head W13 phase.

This independently reproduces a concrete missing physical context, but one
route point cannot identify a safe new parameter. Do not modify frozen v8 yet.
The next gate is a predeclared route sweep over `M={1,2,5,6,12}` with the same
five contexts. A new term is eligible only if the W13 head/background effect is
stable by route bucket, remains separable from W2 and fixed task overhead, and
improves held-out real-trace critical-path-switch predictions.

## Route sweep and first gather-pressure model

The predeclared `M={1,2,5,6,12}` sweep used the same scrub protocol, fixed
background, five randomized paired contexts, five warmups, and 21 trace rounds
for each new route point. M1 retains the earlier 31-round session.

| Target M | Isolated | 1T-only delta | 16T-only delta | Full delta | After-68 minus full |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.633 ms | +0.272 ms | +0.558 ms | +0.811 ms | -0.523 ms |
| 2 | 0.629 ms | +0.302 ms | +0.552 ms | +0.836 ms | -0.553 ms |
| 5 | 0.966 ms | +0.113 ms | +0.392 ms | +0.621 ms | -0.530 ms |
| 6 | 0.941 ms | +0.128 ms | +0.414 ms | +0.655 ms | -0.540 ms |
| 12 | 1.909 ms | +0.022 ms | +0.167 ms | +0.283 ms | -0.202 ms |

The victim effect is therefore route-sensitive: M1/M2 behave as the same
kernel-tail bucket, M5/M6 form a second bucket, and by M12 the 1T-peer main
effect is almost gone. Stage medians localize nearly all of the excess to W13;
the target gather itself remains small. Full-head traces additionally show why
the frozen model misses the effect. For a representative call, target W13 runs
from `0.558--1.711 ms`, the first 68-route 1T peer enters W13 at `1.037 ms`, and
the first 16T peer enters W13 at `1.438 ms`. Before those transitions, their
`gather_pack_a` phases are already active. The frozen model represents those
gathers as zero-resource operator overhead and predicts the target finished
before the transitions.

An opt-in analytical prototype now represents gather as
`fixed_ns + row_ns * ceil(M/T)` and assigns effective DRAM bytes for input reads
plus packed-A writes. The old non-gather residual is redistributed into W13/W2
in proportion to their isolated stage time, preserving the old isolated total
where width residuals exist while correcting temporal placement. Missing
calibration leaves the old behavior exactly unchanged.

Using trace-derived `fixed_ns=5 us`, `row_ns=2.9 us`, and scanning only the
effective-traffic multiplier on these 25 points gives a best neighborhood near
`3x`. Mean absolute relative error falls from frozen-v8 `21.7%` to `12.6%`.
M1 full-head and 16T-only become `1.500/1.192 ms`, close to measured
`1.445/1.193 ms`. This is not a passing calibration: the same setting predicts
M1 1T-only and after-68 as `1.254/1.171 ms`, versus measured `0.907/0.921 ms`.
It also has no independent real-trace holdout yet.

The independent scrubbed M1 repeat, which was not needed to choose the traffic
multiplier, confirms both the useful and failed parts: full-head/16T-only errors
are `+3.22/-0.59%`, while 1T-only/after-68 errors remain `+38.03/+26.74%` and
isolated is `-18.15%`; five-context MAPE is `17.34%`. Thus the lower 25-point
in-sample error is not sufficient evidence to freeze the parameter.

The failed joint fit rejects one global "background bandwidth" multiplier.
The remaining split is physically narrower: domain-local, latency-bound 1T
gathers and cross-domain streaming traffic do not share the same service curve
or victim coupling. The prototype remains default-off, no new calibration is
frozen, and VND/LNS remains gated. The next independent probe must record 1T
and 16T gather duration plus DRAM/LLC-domain traffic while sweeping overlap
against a fixed cold-W13 victim.
