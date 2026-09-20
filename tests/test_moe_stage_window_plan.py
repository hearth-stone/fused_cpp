"""The `(threads, window_tiles)` -> computation pattern mapping.

The native `StageWindowPlan` and the planner-side derivation in
`cost_model/full_stage_geometry.py` must agree tile for tile, because the planner
chooses windows there and the runtime executes them here. These tests pin both
against each other and pin the `R = 1` endpoint against `full_n_team_stripes`.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import pytest

from fused_cpp.moe import _HAS_BF16_TILED_FUSED_MOE


ROOT = Path(__file__).resolve().parents[1]
COST_MODEL = ROOT / "cpu_moe_schedule_optimization" / "cost_model"
if str(COST_MODEL) not in sys.path:
    sys.path.insert(0, str(COST_MODEL))

from full_stage_geometry import full_stage_geometry  # noqa: E402

pytestmark = pytest.mark.skipif(
    platform.machine() not in ("aarch64", "arm64") or not _HAS_BF16_TILED_FUSED_MOE,
    reason="BF16 tiled fused MoE backend requires AArch64",
)

# TP4 production stages: w13 is [2F, H] with K=H, N=2F; w2 is [H, F] with K=F, N=H.
_STAGES = {"w13": (4096, 1024), "w2": (512, 4096)}
_N_TILE = 8
_WIDTHS = [1, 2, 3, 4, 6, 8, 12, 16, 32]


def _stage_window_plan(n, n_tile, threads, window_tiles=0):
    from fused_cpp import _C

    return _C.fused_moe_test_stage_window_plan(n, n_tile, threads, window_tiles)


def _split_plan_n_ranges(stage, m, k, n, group_size):
    """The N-split ranges the pre-window executor produced for a full stripe."""
    from fused_cpp import _C

    _, _, n_ranges = _C.fused_moe_test_split_plan(stage, m, k, n, group_size)
    return [tuple(item) for item in n_ranges]


@pytest.mark.parametrize("stage", sorted(_STAGES))
@pytest.mark.parametrize("threads", _WIDTHS)
def test_full_stripe_is_the_single_window_endpoint(stage, threads):
    k, n = _STAGES[stage]
    windows, range_tiles, window_tiles, ranges = _stage_window_plan(n, _N_TILE, threads)

    geometry = full_stage_geometry(k=k, n=n, n_tile=_N_TILE)
    assert windows == 1
    assert window_tiles == geometry.tiles_per_worker(threads)
    assert range_tiles == threads * window_tiles
    assert len(ranges) == 1

    # The single window's ranges tile [0, N) exactly, in order.
    cursor = 0
    for begin, cols in ranges[0]:
        assert begin == cursor
        cursor += cols
    assert cursor == n


@pytest.mark.parametrize("stage", sorted(_STAGES))
@pytest.mark.parametrize("threads", _WIDTHS)
def test_r1_matches_the_legacy_n_split_ranges(stage, threads):
    """The R=1 endpoint must reproduce `fused_moe_test_split_plan`'s N ranges."""
    k, n = _STAGES[stage]
    _, _, _, ranges = _stage_window_plan(n, _N_TILE, threads)
    legacy = _split_plan_n_ranges(stage, 128, k, n, threads)
    assert ranges[0] == legacy


@pytest.mark.parametrize("stage", sorted(_STAGES))
@pytest.mark.parametrize("threads", _WIDTHS)
def test_native_matches_the_planner_derivation(stage, threads):
    k, n = _STAGES[stage]
    geometry = full_stage_geometry(k=k, n=n, n_tile=_N_TILE)
    full = geometry.tiles_per_worker(threads)
    for window_tiles in sorted({1, 2, 3, 5, 7, 16, full}):
        if window_tiles > geometry.total_tiles:
            continue
        windows, range_tiles, resolved, ranges = _stage_window_plan(n, _N_TILE, threads, window_tiles)
        expected = geometry.window_plan(threads, window_tiles)

        assert resolved == window_tiles
        assert range_tiles == expected.range_tiles
        assert windows == expected.windows
        for index in range(windows):
            for tid in range(threads):
                want = expected.thread_range(index, tid)
                assert ranges[index][tid] == (want.begin_tile * _N_TILE, want.tiles * _N_TILE), (
                    f"{stage} t={threads} omega={window_tiles} window={index} tid={tid}"
                )


@pytest.mark.parametrize("stage", sorted(_STAGES))
@pytest.mark.parametrize("threads", [1, 2, 4, 8, 16, 32])
def test_windows_tile_the_stage_for_every_legal_window(stage, threads):
    k, n = _STAGES[stage]
    geometry = full_stage_geometry(k=k, n=n, n_tile=_N_TILE)
    for window_tiles in range(1, geometry.tiles_per_worker(threads) + 1):
        _, _, _, ranges = _stage_window_plan(n, _N_TILE, threads, window_tiles)
        spans = sorted((b, c) for window in ranges for (b, c) in window if c > 0)
        cursor = 0
        for begin, cols in spans:
            assert begin == cursor, f"{stage} t={threads} omega={window_tiles}"
            cursor += cols
        assert cursor == n, f"{stage} t={threads} omega={window_tiles}"
        # A power-of-two width divides both stages, so no window starves a worker.
        for window in ranges:
            assert all(cols > 0 for (_, cols) in window)


def test_uneven_width_keeps_every_worker_on_one_contiguous_stripe():
    """t=6, omega=7 over 128 tiles: stripes 22/22/21/21/21/21, windows inside each stripe.

    Ownership comes first, so a worker's windows are consecutive tiles of its own stripe and
    only the workers with the shorter stripe run out of work in the last pass.
    """
    n = _STAGES["w13"][1]
    windows, range_tiles, _, ranges = _stage_window_plan(n, _N_TILE, 6, 7)
    assert (windows, range_tiles) == (4, 42)
    per_thread = [[(begin // _N_TILE, cols // _N_TILE) for (begin, cols) in (window[tid] for window in ranges)
                   if cols > 0] for tid in range(6)]
    sizes = [sum(tiles for _, tiles in thread) for thread in per_thread]
    assert sizes == [22, 22, 21, 21, 21, 21]
    for thread in per_thread:  # contiguous inside the stripe, in order
        cursor = thread[0][0]
        for begin, tiles in thread:
            assert begin == cursor
            cursor += tiles
    idle = sum(1 for (_, cols) in ranges[-1] if cols == 0)
    assert idle == 4  # the four 21-tile stripes are done after three windows


def test_stage_window_plan_rejects_illegal_geometry():
    with pytest.raises(RuntimeError, match="exceed stage tiles"):
        _stage_window_plan(1024, _N_TILE, 4, 1024 // _N_TILE + 1)
    with pytest.raises(RuntimeError, match="must be tile aligned"):
        _stage_window_plan(1020, _N_TILE, 4, 1)
    with pytest.raises(RuntimeError, match="must be positive"):
        _stage_window_plan(0, _N_TILE, 4, 1)


# ── W13 execution: reordering the (M panel, window) traversal is a no-op ──


def _team_w13_packc_window(a, w13, group_size, window_tiles, use_sve, degree=5, n_tile=_N_TILE):
    from fused_cpp import _C

    return _C.fused_moe_test_team_w13_silu_packc_window(a, w13, group_size, degree, n_tile, window_tiles, use_sve)


@pytest.mark.parametrize("use_sve", [True, False])
@pytest.mark.parametrize("group_size", [1, 2, 4, 8])
@pytest.mark.parametrize("rows", [1, 7, 12, 13, 28, 72, 120])
def test_windowed_w13_is_bitwise_identical_to_the_full_stripe(rows, group_size, use_sve):
    """Every legal window must produce the same bytes as the R=1 full stripe.

    Windows only reorder the (M panel, window) traversal. Each pair is visited
    once and each worker writes disjoint output columns, and the fused W13 has no
    K blocking so nothing accumulates across windows. So this is exact equality,
    not a tolerance comparison.
    """
    import torch

    torch.manual_seed(rows * 100 + group_size)
    h, f = 256, 128
    a = torch.randn(rows, h, dtype=torch.bfloat16) * 0.05
    w13 = torch.randn(2 * f, h, dtype=torch.bfloat16) * 0.05

    reference = _team_w13_packc_window(a, w13, group_size, 0, use_sve)
    total_tiles = (2 * f) // _N_TILE
    full = -(-total_tiles // group_size)
    for window_tiles in range(1, full + 1):
        got = _team_w13_packc_window(a, w13, group_size, window_tiles, use_sve)
        assert torch.equal(got, reference), (
            f"rows={rows} t={group_size} sve={use_sve} omega={window_tiles} differs from the full stripe"
        )


@pytest.mark.parametrize("use_sve", [True, False])
def test_windowed_w13_covers_every_column_for_a_starving_width(use_sve):
    """A short tail window leaves some workers idle but must still cover the stage."""
    import torch

    torch.manual_seed(7)
    h, f = 256, 128
    a = torch.randn(24, h, dtype=torch.bfloat16) * 0.05
    w13 = torch.randn(2 * f, h, dtype=torch.bfloat16) * 0.05

    reference = _team_w13_packc_window(a, w13, 6, 0, use_sve)
    # t=6, omega=7 -> range=42 over 32 tiles, so one short window of 32 tiles.
    got = _team_w13_packc_window(a, w13, 6, 7, use_sve)
    assert torch.equal(got, reference)


def test_windowed_w13_rejects_an_oversized_window():
    import torch

    a = torch.randn(8, 256, dtype=torch.bfloat16) * 0.05
    w13 = torch.randn(256, 256, dtype=torch.bfloat16) * 0.05
    total_tiles = 256 // _N_TILE
    with pytest.raises(RuntimeError, match="exceed stage tiles"):
        _team_w13_packc_window(a, w13, 4, total_tiles + 1, True)


# ── W2 execution and the owner-scatter mirror ────────────────────────────


def _prepare_w2_packed(h, f):
    """Pack a TP-shaped expert pair and return the W2 half: K=F, N=H."""
    import torch

    from fused_cpp import _C

    torch.manual_seed(11)
    w13 = torch.randn(1, 2 * f, h, dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(1, h, f, dtype=torch.bfloat16) * 0.05
    packed = _C.fused_moe_bf16_tiled_prepare_weights(w13, w2, False, "auto")
    return packed[3], packed[4], packed[5]


def _team_w2_window(a, w2_packed, k, n, group_size, window_tiles, mode):
    from fused_cpp import _C

    return _C.fused_moe_test_team_w2_window(a, w2_packed, k, n, group_size, _N_TILE, window_tiles, mode)


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("group_size", [1, 2, 4, 8])
@pytest.mark.parametrize("rows", [1, 7, 12, 13, 28, 72])
def test_windowed_w2_is_bitwise_identical_to_the_full_stripe(rows, group_size, mode):
    import torch

    k, n = 128, 256
    packed, packed_k, packed_n = _prepare_w2_packed(n, k)
    torch.manual_seed(rows * 31 + group_size)
    a = torch.randn(rows, packed_k, dtype=torch.bfloat16) * 0.05

    reference = _team_w2_window(a, packed, packed_k, packed_n, group_size, 0, mode)
    total_tiles = packed_n // _N_TILE
    full = -(-total_tiles // group_size)
    for window_tiles in sorted({1, 2, 3, 5, full}):
        if window_tiles > total_tiles:
            continue
        got = _team_w2_window(a, packed, packed_k, packed_n, group_size, window_tiles, mode)
        assert torch.equal(got, reference), (
            f"rows={rows} t={group_size} mode={mode} omega={window_tiles} differs from the full stripe"
        )


def _scatter_ranges(n, group_size, local_tid, window_tiles, owner):
    from fused_cpp import _C

    return [
        tuple(item)
        for item in _C.fused_moe_test_w2_scatter_ranges(n, _N_TILE, group_size, local_tid, window_tiles, owner)
    ]


@pytest.mark.parametrize("group_size", [1, 2, 4, 8])
@pytest.mark.parametrize("window_tiles", [0, 1, 2, 5])
def test_owner_scatter_mirrors_the_w2_gemm_windows(group_size, window_tiles):
    """The owner path must scatter exactly the ranges its GEMM computed.

    With R > 1 a worker owns a strided set of tile runs rather than one stripe, so
    a single-stripe scatter would touch columns it never wrote. Mirroring is the
    precondition for skipping the W2-to-scatter barrier.
    """
    n = _STAGES["w2"][1]
    for tid in range(group_size):
        scatter = _scatter_ranges(n, group_size, tid, window_tiles, True)
        _, _, _, gemm = _stage_window_plan(n, _N_TILE, group_size, window_tiles)
        expected = [
            (begin, cols) for window in gemm for (idx, (begin, cols)) in enumerate(window) if idx == tid and cols > 0
        ]
        assert scatter == expected, f"t={group_size} tid={tid} omega={window_tiles}"


@pytest.mark.parametrize("group_size", [1, 2, 4, 8])
@pytest.mark.parametrize("window_tiles", [0, 1, 2, 5])
def test_owner_scatter_ranges_cover_the_stage_exactly(group_size, window_tiles):
    n = _STAGES["w2"][1]
    spans = sorted(
        item for tid in range(group_size) for item in _scatter_ranges(n, group_size, tid, window_tiles, True)
    )
    cursor = 0
    for begin, cols in spans:
        assert begin == cursor, f"t={group_size} omega={window_tiles}"
        cursor += cols
    assert cursor == n


@pytest.mark.parametrize("group_size", [1, 2, 4, 8])
def test_non_owner_scatter_ignores_the_window(group_size):
    """The non-owner path runs after a barrier, so it keeps one contiguous stripe."""
    n = _STAGES["w2"][1]
    for tid in range(group_size):
        baseline = _scatter_ranges(n, group_size, tid, 0, False)
        assert len(baseline) == 1
        for window_tiles in (1, 2, 5):
            assert _scatter_ranges(n, group_size, tid, window_tiles, False) == baseline


# ── Plan V2 carries the windows per task ─────────────────────────────────


def _windowed_plan(bridge, w13_window_tiles, w2_window_tiles):
    from fused_cpp.moe.plan import AsyncMoEPlanV2

    payload = dict(bridge)
    payload["task_w13_window_tiles"] = w13_window_tiles
    payload["task_w2_window_tiles"] = w2_window_tiles
    return AsyncMoEPlanV2.from_dict(payload)


def test_plan_v2_defaults_the_windows_to_the_full_stripe():
    from fused_cpp.moe.plan import AsyncMoEPlanV2, upgrade_legacy_async_plan

    bridge = upgrade_legacy_async_plan(
        {
            "task_expert_ids": [0, 1],
            "task_core_begins": [0, 2],
            "task_threads": [2, 2],
            "task_dep_offsets": [0, 0, 0],
            "task_deps": [],
            "thread_cpu_ids": [0, 1, 2, 3],
            "num_threads": 4,
        }
    )
    plan = AsyncMoEPlanV2.from_dict(bridge)
    assert plan.task_w13_window_tiles.tolist() == [0, 0]
    assert plan.task_w2_window_tiles.tolist() == [0, 0]

    windowed = _windowed_plan(bridge, [1, 2], [4, 8])
    assert windowed.task_w13_window_tiles.tolist() == [1, 2]
    assert windowed.task_w2_window_tiles.tolist() == [4, 8]


def test_plan_v2_rejects_a_negative_window():
    from fused_cpp.moe.plan import upgrade_legacy_async_plan

    bridge = upgrade_legacy_async_plan(
        {
            "task_expert_ids": [0, 1],
            "task_core_begins": [0, 2],
            "task_threads": [2, 2],
            "task_dep_offsets": [0, 0, 0],
            "task_deps": [],
            "thread_cpu_ids": [0, 1, 2, 3],
            "num_threads": 4,
        }
    )
    with pytest.raises(ValueError, match="task_w13_window_tiles must be non-negative"):
        _windowed_plan(bridge, [-1, 0], [0, 0])
    with pytest.raises(ValueError, match="task_w2_window_tiles must be non-negative"):
        _windowed_plan(bridge, [0, 0], [0, -3])


# ── the calibrated policy ────────────────────────────────────────────────


def _policy():
    planners = ROOT / "cpu_moe_schedule_optimization" / "planners"
    if str(planners) not in sys.path:
        sys.path.insert(0, str(planners))
    from stage_window_policy import AMAZON_C5_192C_TP4_F512_V5

    return AMAZON_C5_192C_TP4_F512_V5


def test_policy_reproduces_the_measured_choices():
    """Spot-check the table against amazon_192c_stage_window_tiles_20260810.md."""
    policy = _policy()
    # 49-95 at four threads measured one tile as best.
    assert policy.select(72, 4) == (1, 8)
    # 96-143 at four threads: one tile at the lower edge, which is the live cell.
    assert policy.select(96, 4) == (1, 16)
    assert policy.select(120, 4) == (1, 16)
    # Eight threads preferred two tiles across 56-120.
    assert policy.select(72, 8) == (2, 16)
    assert policy.select(96, 8) == (2, 8)
    assert policy.select(120, 8) == (2, 8)
    # Sixteen threads: W13 only, and only through route 143.
    assert policy.select(48, 16) == (1, 0)
    assert policy.select(72, 16) == (1, 0)
    assert policy.select(120, 16) == (2, 0)


def test_policy_leaves_large_routes_and_wide_teams_on_the_full_stripe():
    """Windowing measured worse there, so the table must not reach into it."""
    policy = _policy()
    # 144 and above at sixteen threads: the full stripe won by 7-32%.
    assert policy.select(144, 16) == (0, 0)
    assert policy.select(216, 16) == (0, 0)
    assert policy.select(384, 16) == (0, 0)
    # Thirty-two threads was inside the noise floor everywhere.
    for routes in (48, 96, 192, 384):
        assert policy.select(routes, 32) == (0, 0)


def test_policy_falls_back_to_the_full_stripe_when_uncovered():
    """An uncalibrated route or width keeps the pre-window geometry."""
    policy = _policy()
    assert policy.select(1, 4) == (0, 0)
    assert policy.select(12, 4) == (0, 0)
    assert policy.select(600, 4) == (0, 0)
    assert policy.select(2040, 4) == (0, 0)
    assert policy.select(96, 3) == (0, 0)
    assert policy.select(96, 12) == (0, 0)
    assert policy.select(96, 64) == (0, 0)


def test_policy_windows_are_legal_for_every_covered_cell():
    """Every emitted window must fit the stage and cover it exactly."""
    policy = _policy()
    w13 = full_stage_geometry(k=4096, n=1024, n_tile=_N_TILE)
    w2 = full_stage_geometry(k=512, n=4096, n_tile=_N_TILE)
    for band in policy.bands:
        for routes in (band.min_routes, band.max_routes):
            for threads in band.widths:
                w13_tiles, w2_tiles = policy.select(routes, threads)
                assert 0 <= w13_tiles <= w13.total_tiles
                assert 0 <= w2_tiles <= w2.total_tiles
                # Native accepts it, and the derived pattern covers the stage.
                _, _, _, ranges = _stage_window_plan(1024, _N_TILE, threads, w13_tiles)
                covered = sum(cols for window in ranges for (_, cols) in window)
                assert covered == 1024


def test_stage_geometry_name_reports_the_full_stripe_only_when_it_is_one():
    planners = ROOT / "cpu_moe_schedule_optimization" / "planners"
    if str(planners) not in sys.path:
        sys.path.insert(0, str(planners))
    from stage_window_policy import stage_geometry_name

    assert stage_geometry_name([0, 0], [0, 0]) == "full_n_team_stripes"
    assert stage_geometry_name([0, 1], [0, 0]) == "windowed_team_stripes"
    assert stage_geometry_name([0, 0], [4, 0]) == "windowed_team_stripes"
