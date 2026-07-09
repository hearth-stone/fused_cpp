"""Memory-read model for 2D GEMM thread splitting.

For C[M,N] = A[M,K] @ B[K,N], splitting output tiles by M and N gives:

    read_A = Tn * M * K * bytes_A
    read_B = Tm * K * N * bytes_B

where Tm is the number of M partitions and Tn is the number of N partitions.
This counts duplicated reads without assuming cache reuse.  Use
``duplicate_penalty < 1`` to model partial cache reuse of duplicated bytes.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class GemmMemorySplit:
    tm: int
    tn: int
    active_threads: int
    m_block: int
    n_block: int
    unique_bytes: int
    duplicated_bytes: int
    effective_bytes: float
    read_a_bytes: int
    read_b_bytes: int
    total_bytes_no_cache: int

    @property
    def label(self) -> str:
        return f"{self.tm}x{self.tn}"


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _format_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    v = float(value)
    for unit in units:
        if abs(v) < 1024.0 or unit == units[-1]:
            return f"{v:.3f} {unit}"
        v /= 1024.0
    return f"{v:.3f} TiB"


def enumerate_splits(
    M: int,
    N: int,
    K: int,
    threads: int,
    *,
    bytes_a: int = 2,
    bytes_b: int = 2,
    m_tile: int = 8,
    n_tile: int = 8,
    require_all_threads: bool = True,
    duplicate_penalty: float = 1.0,
) -> list[GemmMemorySplit]:
    """Return all valid (Tm,Tn) split candidates sorted by effective bytes.

    ``duplicate_penalty`` is applied only to bytes above the unique A+B read.
    ``1.0`` means no cache-reuse discount; ``0.0`` means duplicated reads are
    free.
    """
    if min(M, N, K, threads, bytes_a, bytes_b, m_tile, n_tile) <= 0:
        raise ValueError("M, N, K, threads, bytes, and tile sizes must be positive")
    if duplicate_penalty < 0.0:
        raise ValueError("duplicate_penalty must be non-negative")

    max_tm = min(threads, _ceil_div(M, m_tile))
    max_tn = min(threads, _ceil_div(N, n_tile))
    unique_a = M * K * bytes_a
    unique_b = K * N * bytes_b
    unique = unique_a + unique_b
    out: list[GemmMemorySplit] = []
    for tm in range(1, max_tm + 1):
        for tn in range(1, max_tn + 1):
            active = tm * tn
            if active > threads:
                continue
            if require_all_threads and active != threads:
                continue
            read_a = tn * unique_a
            read_b = tm * unique_b
            total = read_a + read_b
            duplicated = total - unique
            effective = unique + duplicate_penalty * duplicated
            out.append(
                GemmMemorySplit(
                    tm=tm,
                    tn=tn,
                    active_threads=active,
                    m_block=_ceil_div(M, tm),
                    n_block=_ceil_div(N, tn),
                    unique_bytes=unique,
                    duplicated_bytes=duplicated,
                    effective_bytes=effective,
                    read_a_bytes=read_a,
                    read_b_bytes=read_b,
                    total_bytes_no_cache=total,
                )
            )
    out.sort(key=lambda s: (s.effective_bytes, -s.active_threads, s.tm, s.tn))
    return out


def best_split(*args, **kwargs) -> GemmMemorySplit:
    candidates = enumerate_splits(*args, **kwargs)
    if not candidates:
        raise ValueError("no valid split candidates")
    return candidates[0]


def _print_table(candidates: Iterable[GemmMemorySplit], limit: int) -> None:
    rows = list(candidates)
    if limit > 0:
        rows = rows[:limit]
    print(
        "split active  M_blk  N_blk  effective     no_cache      "
        "dup          A_read       B_read"
    )
    for s in rows:
        print(
            f"{s.label:>5} {s.active_threads:>6} "
            f"{s.m_block:>6} {s.n_block:>6} "
            f"{_format_bytes(s.effective_bytes):>12} "
            f"{_format_bytes(s.total_bytes_no_cache):>12} "
            f"{_format_bytes(s.duplicated_bytes):>12} "
            f"{_format_bytes(s.read_a_bytes):>12} "
            f"{_format_bytes(s.read_b_bytes):>12}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--bytes-a", type=int, default=2)
    parser.add_argument("--bytes-b", type=int, default=2)
    parser.add_argument("--m-tile", type=int, default=8)
    parser.add_argument("--n-tile", type=int, default=8)
    parser.add_argument(
        "--allow-idle",
        action="store_true",
        help="Allow Tm*Tn < threads if it minimizes memory traffic.",
    )
    parser.add_argument(
        "--duplicate-penalty",
        type=float,
        default=1.0,
        help="Cache-reuse discount for duplicated bytes: 1=no discount, 0=free.",
    )
    parser.add_argument("--top", type=int, default=16)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = enumerate_splits(
        args.M,
        args.N,
        args.K,
        args.threads,
        bytes_a=args.bytes_a,
        bytes_b=args.bytes_b,
        m_tile=args.m_tile,
        n_tile=args.n_tile,
        require_all_threads=not args.allow_idle,
        duplicate_penalty=args.duplicate_penalty,
    )
    if not candidates:
        raise SystemExit("no valid split candidates")
    if args.json:
        print(json.dumps([asdict(c) for c in candidates], indent=2))
        return
    best = candidates[0]
    print(
        f"M={args.M} N={args.N} K={args.K} threads={args.threads} "
        f"bytes_a={args.bytes_a} bytes_b={args.bytes_b} "
        f"duplicate_penalty={args.duplicate_penalty}"
    )
    print(
        f"best={best.label} active={best.active_threads} "
        f"effective={_format_bytes(best.effective_bytes)} "
        f"no_cache={_format_bytes(best.total_bytes_no_cache)} "
        f"duplicated={_format_bytes(best.duplicated_bytes)}"
    )
    _print_table(candidates, args.top)


if __name__ == "__main__":
    main()
