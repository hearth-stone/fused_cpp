from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
COST_MODEL = ROOT / "cpu_moe_schedule_optimization" / "cost_model"
sys.path.insert(0, str(COST_MODEL))

from gemm_ecm import (  # noqa: E402
    EcmCaps,
    build_stage_report,
    gemm_ecm_work,
    kernel_panels,
    predict_ecm,
    w13_stage,
    w2_stage,
)
from gemm_cost_model import (  # noqa: E402
    ExecutionSchedule,
    LogicalGemmWork,
    MeasuredMachineProfile,
    fused_expert_work,
    predict_stage,
)
from sve_bf16_kernel_model import SveBf16KernelProfile  # noqa: E402


def test_logical_expert_work_contains_only_algorithmic_work() -> None:
    routes = 12
    hidden_size = 4096
    intermediate_size = 1024
    work = fused_expert_work(routes, hidden_size, intermediate_size)

    assert work.w13.useful_flops == 4 * routes * hidden_size * intermediate_size
    assert work.w2.useful_flops == 2 * routes * hidden_size * intermediate_size
    assert work.useful_flops == 6 * routes * hidden_size * intermediate_size
    assert work.w13.compulsory_a_read_bytes == 2 * routes * hidden_size
    assert work.w13.compulsory_b_read_bytes == 4 * hidden_size * intermediate_size
    assert work.dependencies == (("w13", "w2"),)


def test_kernel_constraints_are_applied_only_during_lowering() -> None:
    logical = LogicalGemmWork(
        name="generic",
        routes=12,
        k=10,
        n=16,
        output_columns=8,
        input_element_bytes=2,
        weight_element_bytes=2,
        output_element_bytes=2,
    )
    schedule = ExecutionSchedule(threads=1)

    with pytest.raises(ValueError, match="multiple of eight"):
        SveBf16KernelProfile(n_tile=8).lower(logical, schedule)


def test_zero_route_work_does_not_claim_weight_or_working_set_traffic() -> None:
    logical = fused_expert_work(0, 64, 32).w2
    mapping = SveBf16KernelProfile(n_tile=8).lower(
        logical,
        ExecutionSchedule(threads=1),
    )

    assert logical.useful_flops == 0
    assert logical.compulsory_b_read_bytes == 0
    assert mapping.demand.shared_cache_bytes == 0
    assert mapping.demand.transient_working_set_bytes == 0


def test_sve_mapper_emits_generic_kernel_demand() -> None:
    logical = fused_expert_work(12, 4096, 1024).w13
    schedule = ExecutionSchedule(
        threads=8,
        parallel_axis="N",
        sequential_n_ranges=2,
    )
    mapping = SveBf16KernelProfile(n_tile=8).lower(logical, schedule)

    assert mapping.demand.logical_work is logical
    assert mapping.demand.schedule is schedule
    assert mapping.demand.executed_flops == logical.useful_flops
    assert mapping.demand.range_invocations == 2
    assert mapping.demand.shared_cache_bytes == mapping.llc_bytes


def test_measured_profile_is_bound_to_implementation_and_thread_width() -> None:
    logical = fused_expert_work(12, 64, 32).w2
    mapping = SveBf16KernelProfile(n_tile=8).lower(
        logical,
        ExecutionSchedule(threads=2),
    )
    common = dict(
        machine_id="test-machine",
        threads=mapping.demand.active_threads,
        matrix_flops_per_second=1e12,
        l1_load_bytes_per_second=1e12,
        shared_cache_bytes_per_second=1e12,
    )

    with pytest.raises(ValueError, match="implementation mismatch"):
        predict_stage(
            mapping.demand,
            MeasuredMachineProfile(implementation_id="other-kernel", **common),
        )

    wrong_width = MeasuredMachineProfile(
        implementation_id=mapping.demand.implementation_id,
        **{**common, "threads": mapping.demand.active_threads + 1},
    )
    with pytest.raises(ValueError, match="thread mismatch"):
        predict_stage(mapping.demand, wrong_width)


def test_kernel_panels_separate_logical_compute_pack_and_store_rows() -> None:
    tail_expectations = {
        1: (1, 2, 8, 1, "M1-exact"),
        2: (2, 2, 8, 2, "M2-exact"),
        3: (3, 4, 8, 3, "M3-exact"),
        5: (5, 6, 8, 5, "M5-exact"),
        9: (9, 10, 12, 9, "M9-exact"),
    }

    for routes, expected in tail_expectations.items():
        panel = kernel_panels(routes)[0]
        assert (
            panel.logical_rows,
            panel.compute_rows,
            panel.packed_rows,
            panel.store_rows,
            panel.kernel,
        ) == expected

    panels = kernel_panels(25)
    assert [panel.compute_rows for panel in panels] == [12, 12, 2]

    legacy = kernel_panels(9, exact_m=False)[0]
    assert (legacy.compute_rows, legacy.store_rows, legacy.kernel) == (12, 12, "M12-padded")
    legacy_profile = SveBf16KernelProfile(n_tile=8, exact_m=False)
    assert legacy_profile.implementation_id.endswith("m12_m8_m4_m2_v1")


def test_m12_instruction_counts_match_assembly_k4_body() -> None:
    hidden_size = 4096
    intermediate_size = 1024
    stage = w13_stage(
        12,
        1,
        hidden_size,
        intermediate_size,
        n_tile=8,
        n_ranges=2,
    )
    work = gemm_ecm_work(stage)
    n_tiles = 2 * intermediate_size // stage.n_tile
    k4_blocks = hidden_size // 4

    assert work.bfmmla_instructions == 24 * k4_blocks * n_tiles
    assert work.a_load_instructions == 6 * k4_blocks * n_tiles
    assert work.b_load_instructions == 4 * k4_blocks * n_tiles
    assert work.executed_flops == 4 * 12 * hidden_size * intermediate_size
    assert work.l1_a_read_bytes == work.a_load_instructions * 16
    assert work.l1_b_read_bytes == (work.b_load_instructions * stage.vector_bytes)


def test_w13_range_split_changes_shared_a_scans_not_inner_loop_work() -> None:
    common = dict(
        routes=12,
        threads=8,
        hidden_size=4096,
        intermediate_size=1024,
        n_tile=8,
    )
    no_split = gemm_ecm_work(w13_stage(**common, n_ranges=1))
    split = gemm_ecm_work(w13_stage(**common, n_ranges=2))

    assert split.bfmmla_instructions == no_split.bfmmla_instructions
    assert split.l1_a_read_bytes == no_split.l1_a_read_bytes
    assert split.l1_b_read_bytes == no_split.l1_b_read_bytes
    assert split.llc_b_read_bytes == no_split.llc_b_read_bytes
    assert split.llc_a_read_bytes == 2 * no_split.llc_a_read_bytes
    assert split.demand.transient_working_set_bytes * 2 == no_split.demand.transient_working_set_bytes


def test_nsplit_b_traffic_is_fixed_while_a_scans_scale_with_threads() -> None:
    single = gemm_ecm_work(w2_stage(12, 1, 4096, 1024, n_tile=8))
    eight = gemm_ecm_work(w2_stage(12, 8, 4096, 1024, n_tile=8))

    assert eight.llc_b_read_bytes == single.llc_b_read_bytes
    assert eight.llc_a_read_bytes == 8 * single.llc_a_read_bytes
    assert eight.balanced_bfmmla_instructions == eight.bfmmla_instructions


def test_nsplit_imbalance_uses_busiest_thread_equivalent_work() -> None:
    work = gemm_ecm_work(w2_stage(12, 3, 64, 32, n_tile=8))

    assert work.allocation.per_thread_tiles == (3, 3, 2)
    assert work.balanced_bfmmla_instructions * 8 == (work.bfmmla_instructions * 9)
    assert work.balanced_output_elements * 8 == work.output_elements * 9


def test_logical_m1_executes_m2_flops_but_stores_one_row() -> None:
    work = gemm_ecm_work(w2_stage(1, 1, 64, 32, n_tile=8))

    assert work.logical_rows == 1
    assert work.compute_rows == 2
    assert work.packed_rows == 8
    assert work.store_rows == 1
    assert work.executed_flops == 2 * work.useful_flops
    assert work.compute_efficiency == pytest.approx(0.5)
    assert work.l1_c_write_bytes == 64 * 4


def test_ecm_prediction_uses_matrix_or_nonoverlap_critical_path() -> None:
    work = gemm_ecm_work(w2_stage(12, 1, 64, 32, n_tile=8))
    matrix_bound = predict_ecm(
        work,
        EcmCaps(
            bfmmla_flops_per_second=1e9,
            l1_load_bytes_per_second=1e15,
            llc_bytes_per_second=1e15,
            epilogue_elements_per_second=1e12,
            stage_fixed_ns=10.0,
            range_fixed_ns=5.0,
        ),
    )
    transfer_bound = predict_ecm(
        work,
        EcmCaps(
            bfmmla_flops_per_second=1e15,
            l1_load_bytes_per_second=1e9,
            llc_bytes_per_second=1e9,
        ),
    )

    assert matrix_bound.bottleneck == "matrix"
    assert matrix_bound.total_ns == pytest.approx(
        matrix_bound.fixed_ns + matrix_bound.matrix_ns + matrix_bound.epilogue_ns
    )
    assert transfer_bound.bottleneck == "load_transfer"
    assert transfer_bound.body_ns == pytest.approx(transfer_bound.l1_load_ns + transfer_bound.llc_ns)


def _synthetic_stage_profile(*, w13_panel_ns: float) -> dict:
    rows = []
    for routes in (192, 384, 768, 1536, 2040):
        panels = routes / 12
        rows.append(
            {
                "routes": routes,
                "threads": 1,
                "w13_ms": (20_000.0 + panels * w13_panel_ns) / 1e6,
                "w2_ms": (10_000.0 + panels * 300_000.0) / 1e6,
            }
        )
    return {
        "shape": {
            "activation": "silu",
            "ffn_hidden_size": 1024,
            "fuse_silu": True,
            "hidden_size": 4096,
            "skip_weighted": True,
            "top_k": 1,
        },
        "rows": rows,
    }


def test_stage_report_validates_panel_model_on_holdout_routes() -> None:
    silu = _synthetic_stage_profile(w13_panel_ns=620_000.0)
    identity = _synthetic_stage_profile(w13_panel_ns=590_000.0)
    report = build_stage_report(
        silu,
        source_profile="silu.json",
        n_tile=8,
        w13_n_ranges=2,
        identity_profile=identity,
        identity_source="identity.json",
    )

    observations = {(row["stage"], row["threads"]): row for row in report["observations"]}
    assert observations[("w13", 1)]["panel_ns"] == pytest.approx(620_000.0)
    assert observations[("w2", 1)]["panel_ns"] == pytest.approx(300_000.0)
    assert observations[("w13", 1)]["holdout_max_error"] == pytest.approx(0.0)
    assert report["silu_deltas"][0]["silu_extra_panel_ns"] == pytest.approx(30_000.0)
    assert report["schema_version"] == 2
    assert report["model_layers"]["algorithm"]["formula"] == "w_s = A_s(x)"
    assert report["model_layers"]["implementation"]["contract"] == "KernelDemand"
    assert report["model_layers"]["machine_response"]["measured"] is True
