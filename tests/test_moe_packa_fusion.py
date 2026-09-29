# -*- coding: utf-8 -*-
"""packA fusion: fused gather + M12/M8 reorder packing.

Task 1: `gather_pack_a_reorder_m8` must produce a byte-for-byte identical
packed buffer to the two-step reference (gather tokens into a row-major,
K_pad-padded buffer, then `pack_a_reorder_m8`).
"""

from __future__ import annotations

import platform

import pytest
import torch

from fused_cpp.moe import _HAS_BF16_TILED_FUSED_MOE

pytestmark = pytest.mark.skipif(
    platform.machine() not in ("aarch64", "arm64") or not _HAS_BF16_TILED_FUSED_MOE,
    reason="BF16 tiled fused MoE backend requires AArch64",
)


def _ceil8(x: int) -> int:
    return (x + 7) // 8 * 8


def _reference_packed(input_tokens, routes, top_k, k_pad):
    """gather -> row-major [rows, k_pad] (tail zero) -> pack_a_reorder_m8."""
    from fused_cpp import _C

    rows = routes.numel()
    h = input_tokens.size(1)
    a_pad = torch.zeros(_ceil8(rows), k_pad, dtype=torch.bfloat16)
    for m in range(rows):
        token = int(routes[m].item()) // top_k
        a_pad[m, :h] = input_tokens[token, :]
    # pack_a_reorder_m8 packs total_rows=rows over ceil(rows/8) blocks; rows
    # beyond `rows` are zero-filled by the kernel itself, so pass rows as
    # total_rows on the a_pad[:rows] view padded to k_pad.
    return _C.fused_moe_test_pack_a_reorder_m8(a_pad[:rows].contiguous())


def _fused_packed(input_tokens, routes, top_k, k_pad):
    from fused_cpp import _C

    return _C.fused_moe_test_gather_pack_a_reorder_m8(input_tokens, routes, top_k, k_pad)


def _m12_main_rows(rows: int) -> int:
    full_rows = rows // 12 * 12
    return full_rows + 12 if rows - full_rows >= 9 else full_rows


def _reference_sve_hybrid_packed(input_tokens, routes, top_k, k_pad):
    rows = routes.numel()
    h = input_tokens.size(1)
    main_rows = _m12_main_rows(rows)
    physical_rows = main_rows + (_ceil8(rows - main_rows) if rows > main_rows else 0)
    gathered = torch.zeros(physical_rows, k_pad, dtype=torch.bfloat16)
    for m in range(rows):
        token = int(routes[m].item()) // top_k
        gathered[m, :h] = input_tokens[token]

    panels = []
    for begin in range(0, main_rows, 12):
        panel = gathered[begin : begin + 12]
        panels.append(panel.reshape(12, k_pad // 4, 4).permute(1, 0, 2).contiguous().flatten())
    if rows > main_rows:
        panel = gathered[main_rows : main_rows + 8]
        panels.append(panel.reshape(8, k_pad // 4, 4).permute(1, 0, 2).contiguous().flatten())
    return torch.cat(panels)


@pytest.mark.parametrize(
    "rows,k_pad,group_size",
    [
        (1, 24, 8),
        (1, 32, 8),
        (1, 64, 8),
        (1, 72, 8),
        (9, 40, 8),
        (12, 4096, 8),
        (13, 4096, 2),
        (13, 4096, 8),
        (24, 4096, 8),
        (37, 4096, 16),
        (85, 4096, 8),
        (95, 4096, 8),
    ],
)
def test_sve_hybrid_mk_work_covers_panels_with_minimum_k_chunk(rows, k_pad, group_size):
    from fused_cpp import _moe_C

    work = _moe_C.fused_moe_test_gather_pack_a_work(rows, k_pad, group_size)
    main_rows = _m12_main_rows(rows)
    panel_count = main_rows // 12 + (1 if rows > main_rows else 0)
    by_panel = {panel: [] for panel in range(panel_count)}
    active_tids = set()
    for tid, panel, panel_rows, k_begin, k_size in work:
        assert 0 <= tid < group_size
        assert panel_rows == (12 if panel < main_rows // 12 else 8)
        assert k_size > 0
        by_panel[panel].append((k_begin, k_size))
        active_tids.add(tid)

    for ranges in by_panel.values():
        ranges.sort()
        cursor = 0
        for k_begin, k_size in ranges:
            assert k_begin == cursor
            if len(ranges) > 1:
                assert k_size >= 32
                assert k_begin % 8 == 0
            cursor += k_size
        assert cursor == k_pad

    main_blocks = main_rows // 12
    if panel_count >= group_size or k_pad < 64:
        expected_active = min(panel_count, group_size)
    else:
        main_atoms = main_blocks * max(1, k_pad // 32)
        tail_atoms = min(max(1, k_pad // 32), max(1, (k_pad + 24) // 48)) if rows > main_rows else 0
        expected_active = min(group_size, main_atoms + tail_atoms)
    assert len(active_tids) == expected_active
    if panel_count >= group_size or k_pad < 64:
        assert all(len(ranges) == 1 for ranges in by_panel.values())
    if rows == 13 and group_size == 2:
        assert active_tids == {0, 1}
        assert set(panel for _, panel, _, _, _ in work) == {0, 1}
    if rows == 72 and group_size == 8:
        work_per_tid = {tid: 0 for tid in active_tids}
        for tid, _, panel_rows, _, k_size in work:
            work_per_tid[tid] += panel_rows * k_size
        assert set(work_per_tid.values()) == {rows * k_pad // group_size}


@pytest.mark.parametrize(
    "rows,h,k_pad,top_k,group_size",
    [
        (1, 20, 24, 6, 8),
        (1, 65, 72, 6, 8),
        (9, 33, 40, 2, 8),
        (12, 61, 64, 1, 8),
        (13, 130, 136, 6, 8),
        (37, 128, 128, 2, 16),
        (85, 130, 136, 6, 8),
        (95, 130, 136, 6, 8),
    ],
)
def test_sve_hybrid_mk_gather_pack_matches_serial_layout(rows, h, k_pad, top_k, group_size):
    from fused_cpp import _moe_C

    torch.manual_seed(rows * 257 + h + top_k + group_size)
    num_tokens = rows + 4
    tokens = (torch.randn(num_tokens, h) * 0.1).to(torch.bfloat16)
    routes = torch.randint(0, num_tokens * top_k, (rows,), dtype=torch.int64)
    ref = _reference_sve_hybrid_packed(tokens, routes, top_k, k_pad)
    out = _moe_C.fused_moe_test_gather_pack_a_reorder_sve_hybrid(tokens, routes, top_k, k_pad, group_size)

    assert ref.shape == out.shape
    assert torch.equal(ref.view(torch.int16), out.view(torch.int16)), (
        f"hybrid pack mismatch rows={rows} h={h} top_k={top_k} threads={group_size}"
    )


def test_sve_hybrid_mk_rejects_non_k8_padding():
    from fused_cpp import _moe_C

    with pytest.raises(RuntimeError, match="multiple of 8"):
        _moe_C.fused_moe_test_gather_pack_a_work(12, 68, 8)


@pytest.mark.parametrize("top_k", [1, 2, 6])
@pytest.mark.parametrize("h", [128, 4096, 130])  # 130 -> K-tail not mult of 4
@pytest.mark.parametrize("rows", [8, 64, 37, 1])  # 37 = 32 + 4 + 1 tail
def test_gather_pack_matches_two_step(rows, h, top_k):
    torch.manual_seed(rows * 131 + h + top_k)
    num_tokens = max(rows // top_k + 4, rows + 4)
    tokens = (torch.randn(num_tokens, h) * 0.1).to(torch.bfloat16)
    # routes are flat indices; token = flat // top_k must be a valid token.
    routes = torch.randint(0, num_tokens * top_k, (rows,), dtype=torch.int64)
    routes = torch.clamp(routes, max=num_tokens * top_k - 1)
    k_pad = _ceil8(h)

    ref = _reference_packed(tokens, routes, top_k, k_pad)
    out = _fused_packed(tokens, routes, top_k, k_pad)

    assert ref.shape == out.shape
    # bit-identical
    assert torch.equal(ref.view(torch.int16), out.view(torch.int16)), f"mismatch rows={rows} h={h} top_k={top_k}"


# ── Part 1 threaded e2e: FUSED_CPP_MOE_FUSED_PACKA on vs off ──────────────
import contextlib  # noqa: E402
import os  # noqa: E402

from fused_cpp.moe import fused_moe_bf16_tiled  # noqa: E402
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights  # noqa: E402

_NSPLIT_ENV_KEYS = (
    "FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT",
    "FUSED_CPP_MOE_N_SPLIT_GROUPS_PER_PARTITION",
    "FUSED_CPP_MOE_N_SPLIT_CORE_BASES",
    "FUSED_CPP_MOE_FUSED_PACKA",
)


@contextlib.contextmanager
def _nsplit_packa_env(packa, groups_per_partition=1, core_bases="0"):
    saved = {k: os.environ.get(k) for k in _NSPLIT_ENV_KEYS}
    os.environ["FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT"] = "1"
    os.environ["FUSED_CPP_MOE_N_SPLIT_GROUPS_PER_PARTITION"] = str(groups_per_partition)
    os.environ["FUSED_CPP_MOE_N_SPLIT_CORE_BASES"] = core_bases
    os.environ["FUSED_CPP_MOE_FUSED_PACKA"] = "1" if packa else "0"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _bf16(*shape, scale=0.2):
    return (torch.randn(*shape) * scale).to(torch.bfloat16)


@pytest.mark.parametrize("degree", [4, 5, 6])
@pytest.mark.parametrize("num_tokens", [2, 64, 130])
def test_packa_on_off_equivalence(degree, num_tokens):
    hidden, ffn, num_experts, top_k = 128, 64, 4, 2
    torch.manual_seed(degree * 17 + num_tokens)
    x = _bf16(num_tokens, hidden)
    w13 = _bf16(num_experts, 2 * ffn, hidden)
    w2 = _bf16(num_experts, hidden, ffn)
    ti = torch.tensor(
        [[(i + j) % num_experts for j in range(top_k)] for i in range(num_tokens)],
        dtype=torch.int32,
    )
    tw = torch.softmax(torch.randn(num_tokens, top_k), dim=-1)
    fused_w = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)

    def run(packa):
        with _nsplit_packa_env(packa, groups_per_partition=1, core_bases="0"):
            return fused_moe_bf16_tiled(
                x,
                fused_w,
                tw,
                ti,
                num_threads=4,
                activation="silu",
                silu_poly_degree=degree,
            )

    off = run(False)
    on = run(True)
    assert on.shape == off.shape
    # Same math (pre-packed read vs in-kernel repack): expect bit-identical.
    assert torch.equal(on.view(torch.int16), off.view(torch.int16)), (
        f"packa on/off differ: degree={degree} num_tokens={num_tokens}"
    )


# ── Task 3: packed-C store bit-identity vs row-major fused + pack ─────────
def _ceil4(x: int) -> int:
    return (x + 3) // 4 * 4


@pytest.mark.parametrize("degree", [4, 5, 6])
@pytest.mark.parametrize("h,f", [(128, 64), (256, 128), (4096, 512)])
@pytest.mark.parametrize("m", [8, 64, 37, 3])  # 37=32+4+1, 3 tail
def test_packc_matches_rowmajor_then_pack(m, h, f, degree):
    from fused_cpp import _C

    torch.manual_seed(m * 7 + h + f + degree)
    a = (torch.randn(m, h) * 0.05).to(torch.bfloat16)
    w13 = (torch.randn(2 * f, h) * 0.05).to(torch.bfloat16)

    # row-major fused reference -> [m, f]; pad to [ceil8(m), ceil4(f)] then pack.
    rm = _C.fused_moe_test_fused_w13_silu(a, w13, degree)  # [m, f]
    mp, fp = _ceil8(m), _ceil4(f)
    rm_pad = torch.zeros(mp, fp, dtype=torch.bfloat16)
    rm_pad[:m, :f] = rm
    ref = _C.fused_moe_test_pack_a_reorder_m8(rm_pad.contiguous())

    # packed-C store path.
    pk = _C.fused_moe_test_fused_w13_silu_packc(a, w13, degree)

    assert ref.shape == pk.shape, (ref.shape, pk.shape)
    assert torch.equal(ref.view(torch.int16), pk.view(torch.int16)), f"packc mismatch m={m} h={h} f={f} degree={degree}"


# ── Task 1: per-tail (rows%8) packed dispatch bit-identity vs m8-pad packc ──
@pytest.mark.parametrize("degree", [4, 5, 6])
@pytest.mark.parametrize("h,f", [(128, 64), (256, 128), (4096, 512)])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 15, 16, 37, 45])
def test_packc_tail_dispatch_matches_pad8(m, h, f, degree):
    from fused_cpp import _C

    torch.manual_seed(m * 13 + h + f + degree)
    a = (torch.randn(m, h) * 0.05).to(torch.bfloat16)
    w13 = (torch.randn(2 * f, h) * 0.05).to(torch.bfloat16)

    # m8-pad packc reference (already proven == row-major fused padded+packed).
    ref = _C.fused_moe_test_fused_w13_silu_packc(a, w13, degree)
    # per-tail dispatch (m8 full blocks + packed-read reorder-m8 tail kernels).
    tail = _C.fused_moe_test_fused_w13_silu_packc_tail(a, w13, degree)

    assert ref.shape == tail.shape, (ref.shape, tail.shape)
    assert torch.equal(ref.view(torch.int16), tail.view(torch.int16)), (
        f"tail dispatch mismatch m={m} h={h} f={f} degree={degree}"
    )


# ── Task 4: multi-expert, per-expert row counts spanning ALL tails ───────
@pytest.mark.parametrize("degree", [4, 5, 6])
@pytest.mark.parametrize("gpp", [1, 8])
def test_packa_multi_expert_all_tails(degree, gpp):
    # Route (top_k=1) so expert e receives counts[e] rows -> covers tails
    # 1..7 plus full+tail blocks. Verifies w13+w2 tail dispatch in full e2e.
    counts = [1, 2, 3, 4, 5, 6, 7, 9, 15, 16, 37]
    num_experts = len(counts)
    hidden, ffn, top_k = 4096, 512, 1
    torch.manual_seed(degree * 101 + gpp)
    ids = []
    for e, c in enumerate(counts):
        ids += [e] * c
    num_tokens = len(ids)
    x = _bf16(num_tokens, hidden)
    w13 = _bf16(num_experts, 2 * ffn, hidden)
    w2 = _bf16(num_experts, hidden, ffn)
    ti = torch.tensor([[e] for e in ids], dtype=torch.int32)
    tw = torch.ones(num_tokens, top_k)
    fused_w = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)

    def run(packa):
        with _nsplit_packa_env(packa, groups_per_partition=gpp, core_bases="0"):
            return fused_moe_bf16_tiled(
                x,
                fused_w,
                tw,
                ti,
                num_threads=8,
                activation="silu",
                silu_poly_degree=degree,
            )

    off = run(False)
    on = run(True)
    assert on.shape == off.shape
    assert torch.equal(on.view(torch.int16), off.view(torch.int16)), (
        f"multi-expert all-tails packa mismatch degree={degree} gpp={gpp}"
    )


# ── Pool: cross-shape reuse must not leak stale intermediate padding ─────
def test_packa_pool_cross_shape():
    # The scratch pool persists across calls on the calling thread and is
    # reused across differently-sized calls. Verify no stale data leaks: each
    # shape must match the packa-off fallback bit-for-bit.
    # F must be a multiple of 8 (fused path). Vary H/F/rows/experts so the
    # persistent pool grows then reuses buffers of different logical shape;
    # stale packed_a/down/intermediate from a bigger call must not leak.
    shapes = [
        (256, 4096, 512, 8, 2),  # big: dirties the pool
        (40, 512, 8, 6, 2),  # small rows + small F
        (48, 256, 16, 5, 2),  # different H/F/strides
        (37, 512, 24, 4, 1),  # non-8-multiple rows (tails) + F=24
        (300, 4096, 512, 8, 2),  # grow again, back to big
    ]
    for si, (T, H, F, E, tk) in enumerate(shapes):
        torch.manual_seed(si * 991 + T + F)
        x = _bf16(T, H)
        w13 = _bf16(E, 2 * F, H)
        w2 = _bf16(E, H, F)
        ids = torch.stack([torch.randperm(E)[:tk] for _ in range(T)]).to(torch.int32)
        tw = torch.softmax(torch.randn(T, tk), dim=-1)
        fused_w = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)

        def run(packa):
            with _nsplit_packa_env(packa, groups_per_partition=1, core_bases="0"):
                return fused_moe_bf16_tiled(x, fused_w, tw, ids, num_threads=8, activation="silu", silu_poly_degree=5)

        off = run(False)
        on = run(True)
        assert torch.equal(on.view(torch.int16), off.view(torch.int16)), (
            f"pool cross-shape leak at shape#{si} T={T} H={H} F={F}"
        )
