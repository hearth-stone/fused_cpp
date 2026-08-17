# Amazon M5 Sparse MLA post-fusion cache retune

Date: 2026-08-17

## Decision

Keep the existing automatic `Sc_l2` heuristic and do not reintroduce a token
panel. After score-copy/max fusion and direct packed-P generation, every fixed
shared-path chunk size in `{64,128,256,512}` was slower than auto on the key
2048 and 8192 cases. The temporary compile-time control was removed; no runtime
or build flag remains.

## Scope and method

Only the shared-dense and shared-prefix `sc_tile` value was overridden. QK/PV,
online-softmax, direct packed-P, scheduling, indexed sparse, and public behavior
were otherwise the step-2 checkpoint. Each value was compiled into a distinct
M5/SVL128 extension; hashes were checked to ensure the variants differed.

M5 NUMA1 cores 96--191, 96 threads, BF16 `h_q=32,d_qk=192,d_v=128`, seed
20260817. Screening used ten warmups and nine samples in forward/reverse order.
Auto and the closest fixed value, 512, then used ten warmups and 21 samples in
both orders. Commands and shapes match the preceding checkpoint records.

Direct native-versus-naive checks passed at fixed 64 and 512 for shared dense,
odd dense tails, shared-prefix, later sparse, output, and statistics. Changing
chunk boundaries changes online reduction order, so fixed-tile checksums differ
from auto while remaining within the existing correctness tolerance.

## Screening results

| Sc tile | 2048 shared-prefix forward/reverse | 8192 shared-dense representative low mode |
|---:|---:|---:|
| auto | 11.380 / 11.399 ms | 8.776 ms |
| 64 | 12.905 / 12.883 ms | 11.670 ms |
| 128 | 12.184 / 12.170 ms | 10.133 ms |
| 256 | 11.687 / 11.703 ms | 9.412--9.449 ms |
| 512 | 11.486 / 11.481 ms | 9.088 ms |

## Final auto versus 512

| Case | Auto | Fixed 512 | 512 change |
|---|---:|---:|---:|
| 2048 shared-prefix, forward | 11.385 ms | 11.484 ms | +0.87% |
| 2048 shared-prefix, reverse | 11.397 ms | 11.430 ms | +0.29% |
| 8192 shared-dense, clean low mode | 8.736--8.776 ms | 9.079--9.088 ms | about +3.6--3.9% |
| Later low-overlap sparse, forward | 2.485 ms | 2.497 ms | +0.48% |
| Later low-overlap sparse, reverse | 2.487 ms | 2.494 ms | +0.28% |

The indexed sparse path does not consume the shared-path override; its small
difference reflects binary/run noise.

## Token-panel conclusion

The earlier full token-panel `{4,8,16}` by B-block `{64,128,256}` experiment
already showed that reduced L2/bus traffic did not reduce wall time. In this
ordered sequence, the stronger dense `16xVL` two-token kernel also halved B
loads but regressed the clean 8192 comparison by 2.75%. Together with the fixed
Sc-tile results above, this rejects adjacent-token cache reuse as a useful next
dimension after inner-loop fusion. No panel implementation is restored.

## Overall sequence result

The retained implementation is score-copy/max fusion plus direct packed-P.
Against the original baseline its stable cumulative improvements are about
2.1% for 2048 shared-prefix and 4.7% for later sparse; 8192 low-mode samples
show roughly 5% improvement but the host's high/low modes prevent an
unconditional 8192 claim. PV online-output fusion, dense 16xVL multi-query, and
manual cache blocking were all retired.
