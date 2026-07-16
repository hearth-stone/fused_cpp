"""SVE BF16 packed-A implementation mapping for the layered GEMM model."""

from __future__ import annotations

from dataclasses import dataclass

if __package__:
    from .gemm_cost_model import ExecutionSchedule, KernelDemand, LogicalGemmWork
else:
    from gemm_cost_model import ExecutionSchedule, KernelDemand, LogicalGemmWork


M_PANEL = 12
BF16_BYTES = 2
SVE_BF16_IMPLEMENTATION_ID = "sve_bf16_packed_a_m12_m8_m4_m2_v1"


@dataclass(frozen=True)
class KernelPanel:
    """One dispatched M panel and its implementation-specific row counts."""

    logical_rows: int
    compute_rows: int
    packed_rows: int
    store_rows: int
    kernel: str


def kernel_panels(routes: int) -> tuple[KernelPanel, ...]:
    """Return the exact M12/M8/M4/M2 dispatch for ``routes`` rows."""
    if routes < 0:
        raise ValueError(f"routes must be non-negative, got {routes}")

    full_panels, remainder = divmod(int(routes), M_PANEL)
    panels = [KernelPanel(12, 12, 12, 12, "M12")] * full_panels
    if remainder == 0:
        return tuple(panels)
    if remainder == 1:
        tail = KernelPanel(1, 2, 8, 1, "M2-as-M1")
    elif remainder == 2:
        tail = KernelPanel(2, 2, 8, 2, "M2")
    elif remainder <= 4:
        tail = KernelPanel(remainder, 4, 8, 4, "M4")
    elif remainder <= 8:
        tail = KernelPanel(remainder, 8, 8, 8, "M8")
    else:
        tail = KernelPanel(remainder, 12, 12, 12, "M12-padded")
    return (*panels, tail)


@dataclass(frozen=True)
class NTileAllocation:
    total_tiles: int
    range_tiles: tuple[int, ...]
    per_thread_tiles: tuple[int, ...]
    active_threads: int
    busiest_thread_tiles: int
    active_thread_range_scans: int


def allocate_n_tiles(
    n_columns: int,
    n_tile: int,
    threads: int,
    n_ranges: int,
) -> NTileAllocation:
    """Mirror the contiguous N-tile split used by each sequential range."""
    if min(n_columns, n_tile, threads, n_ranges) <= 0:
        raise ValueError("n_columns, n_tile, threads, and n_ranges must be positive")
    if n_columns % n_tile != 0:
        raise ValueError(f"n_columns must be padded to n_tile: {n_columns} % {n_tile} != 0")

    total_tiles = n_columns // n_tile
    range_base, range_extra = divmod(total_tiles, n_ranges)
    if range_extra:
        raise ValueError(f"N tiles={total_tiles} cannot form {n_ranges} equal ranges")
    range_tiles = (range_base,) * n_ranges
    if not range_tiles or range_base <= 0:
        raise ValueError(f"n_ranges={n_ranges} exceeds available N tiles={total_tiles}")

    per_thread = [0] * threads
    active_thread_range_scans = 0
    for tiles in range_tiles:
        thread_base, thread_extra = divmod(tiles, threads)
        active_thread_range_scans += min(tiles, threads)
        for thread_id in range(threads):
            per_thread[thread_id] += thread_base + (thread_id < thread_extra)

    active_threads = sum(value > 0 for value in per_thread)
    return NTileAllocation(
        total_tiles=total_tiles,
        range_tiles=range_tiles,
        per_thread_tiles=tuple(per_thread),
        active_threads=active_threads,
        busiest_thread_tiles=max(per_thread),
        active_thread_range_scans=active_thread_range_scans,
    )


@dataclass(frozen=True)
class SveBf16KernelExecution:
    """SVE-specific mapping details plus the generic physical demand."""

    logical_work: LogicalGemmWork
    schedule: ExecutionSchedule
    n_tile: int
    panels: tuple[KernelPanel, ...]
    allocation: NTileAllocation
    logical_rows: int
    compute_rows: int
    packed_rows: int
    store_rows: int
    useful_flops: int
    executed_flops: int
    bfmmla_instructions: int
    a_load_instructions: int
    b_load_instructions: int
    balanced_bfmmla_instructions: int
    balanced_key_instructions: int
    l1_a_read_bytes: int
    l1_b_read_bytes: int
    l1_c_write_bytes: int
    balanced_l1_load_bytes: int
    private_refill_bytes: int
    llc_a_read_bytes: int
    llc_b_read_bytes: int
    llc_c_write_bytes: int
    output_elements: int
    balanced_output_elements: int
    demand: KernelDemand

    @property
    def vector_bytes(self) -> int:
        return self.n_tile * BF16_BYTES

    @property
    def key_body_instructions(self) -> int:
        return self.bfmmla_instructions + self.a_load_instructions + self.b_load_instructions

    @property
    def l1_load_bytes(self) -> int:
        return self.l1_a_read_bytes + self.l1_b_read_bytes

    @property
    def llc_bytes(self) -> int:
        return self.llc_a_read_bytes + self.llc_b_read_bytes + self.llc_c_write_bytes

    @property
    def compute_efficiency(self) -> float:
        return self.demand.compute_efficiency


@dataclass(frozen=True)
class SveBf16KernelProfile:
    """Exact lowering rules for the current SVE BF16 packed-A assembly."""

    n_tile: int
    implementation_id: str = SVE_BF16_IMPLEMENTATION_ID

    def __post_init__(self) -> None:
        if self.n_tile <= 0:
            raise ValueError("SVE BF16 n_tile must be positive")
        if not self.implementation_id:
            raise ValueError("implementation_id must be non-empty")

    @property
    def vector_bytes(self) -> int:
        return self.n_tile * BF16_BYTES

    def lower(
        self,
        logical_work: LogicalGemmWork,
        schedule: ExecutionSchedule,
    ) -> SveBf16KernelExecution:
        """Apply SVE tiling, padding, N ownership, and cache-traffic rules."""
        if schedule.parallel_axis != "N":
            raise ValueError("current SVE BF16 profile supports only N-split schedules")
        if logical_work.input_element_bytes != BF16_BYTES:
            raise ValueError("current SVE BF16 profile requires BF16 input")
        if logical_work.weight_element_bytes != BF16_BYTES:
            raise ValueError("current SVE BF16 profile requires BF16 weights")
        if logical_work.k % 8 != 0:
            raise ValueError(f"SVE packed K must be a multiple of eight, got {logical_work.k}")
        if logical_work.n % self.n_tile != 0:
            raise ValueError(f"SVE packed N must be padded to n_tile: {logical_work.n} % {self.n_tile} != 0")
        if (logical_work.output_columns * self.n_tile) % logical_work.n != 0:
            raise ValueError("each SVE N tile must map to an integer number of outputs")

        panels = kernel_panels(logical_work.routes)
        allocation = allocate_n_tiles(
            logical_work.n,
            self.n_tile,
            schedule.threads,
            schedule.sequential_n_ranges,
        )
        logical_rows = sum(panel.logical_rows for panel in panels)
        compute_rows = sum(panel.compute_rows for panel in panels)
        packed_rows = sum(panel.packed_rows for panel in panels)
        store_rows = sum(panel.store_rows for panel in panels)

        # One N tile and K4 body execute 2*Mr BFMMLA, Mr/2 16-byte A
        # broadcasts, and four vector B loads.
        bfmmla_per_tile = compute_rows * logical_work.k // 2
        a_loads_per_tile = compute_rows * logical_work.k // 8
        b_loads_per_tile = len(panels) * logical_work.k
        bfmmla = bfmmla_per_tile * allocation.total_tiles
        a_loads = a_loads_per_tile * allocation.total_tiles
        b_loads = b_loads_per_tile * allocation.total_tiles

        busiest_bfmmla = bfmmla_per_tile * allocation.busiest_thread_tiles
        busiest_a_loads = a_loads_per_tile * allocation.busiest_thread_tiles
        busiest_b_loads = b_loads_per_tile * allocation.busiest_thread_tiles
        balanced_bfmmla = busiest_bfmmla * allocation.active_threads
        balanced_key = (busiest_bfmmla + busiest_a_loads + busiest_b_loads) * allocation.active_threads

        l1_a = a_loads * 16
        l1_b = b_loads * self.vector_bytes
        l1_c = store_rows * logical_work.output_columns * logical_work.output_element_bytes
        balanced_l1 = (busiest_a_loads * 16 + busiest_b_loads * self.vector_bytes) * allocation.active_threads

        # This is an implementation policy, not an algorithmic lower bound:
        # each active N owner/range refills A once, while all owners together
        # stream one full B matrix for every physical M panel.
        a_panel_bytes = compute_rows * logical_work.k * BF16_BYTES
        llc_a = a_panel_bytes * allocation.active_thread_range_scans
        llc_b = len(panels) * logical_work.k * logical_work.n * BF16_BYTES
        private_refill = llc_a + llc_b
        output_elements = store_rows * logical_work.output_columns
        output_columns_per_tile = logical_work.output_columns * self.n_tile // logical_work.n
        balanced_output_elements = (
            store_rows * output_columns_per_tile * allocation.busiest_thread_tiles * allocation.active_threads
        )

        executed_flops = bfmmla * (2 * self.vector_bytes)
        balanced_executed_flops = balanced_bfmmla * (2 * self.vector_bytes)
        key_instructions = bfmmla + a_loads + b_loads
        demand = KernelDemand(
            logical_work=logical_work,
            schedule=schedule,
            implementation_id=self.implementation_id,
            compute_kind="sve_bfmmla_bf16",
            active_threads=allocation.active_threads,
            executed_flops=executed_flops,
            balanced_executed_flops=balanced_executed_flops,
            key_body_instructions=key_instructions,
            balanced_key_body_instructions=balanced_key,
            l1_load_bytes=l1_a + l1_b,
            balanced_l1_load_bytes=balanced_l1,
            private_refill_bytes=private_refill,
            shared_cache_read_bytes=llc_a + llc_b,
            shared_cache_write_bytes=l1_c,
            epilogue_elements=output_elements,
            balanced_epilogue_elements=balanced_output_elements,
            transient_working_set_bytes=(
                logical_work.k * logical_work.n * BF16_BYTES // schedule.sequential_n_ranges
                if logical_work.routes
                else 0
            ),
            stage_invocations=1,
            range_invocations=schedule.sequential_n_ranges,
        )
        return SveBf16KernelExecution(
            logical_work=logical_work,
            schedule=schedule,
            n_tile=self.n_tile,
            panels=panels,
            allocation=allocation,
            logical_rows=logical_rows,
            compute_rows=compute_rows,
            packed_rows=packed_rows,
            store_rows=store_rows,
            useful_flops=logical_work.useful_flops,
            executed_flops=executed_flops,
            bfmmla_instructions=bfmmla,
            a_load_instructions=a_loads,
            b_load_instructions=b_loads,
            balanced_bfmmla_instructions=balanced_bfmmla,
            balanced_key_instructions=balanced_key,
            l1_a_read_bytes=l1_a,
            l1_b_read_bytes=l1_b,
            l1_c_write_bytes=l1_c,
            balanced_l1_load_bytes=balanced_l1,
            private_refill_bytes=private_refill,
            llc_a_read_bytes=llc_a,
            llc_b_read_bytes=llc_b,
            llc_c_write_bytes=l1_c,
            output_elements=output_elements,
            balanced_output_elements=balanced_output_elements,
            demand=demand,
        )
