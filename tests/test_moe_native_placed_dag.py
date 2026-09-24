from __future__ import annotations

import random
import sys
from dataclasses import replace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / "cpu_moe_schedule_optimization" / "cost_model")]

from analytic_model import (  # noqa: E402
    DramDomainInjectionCalibration,
    LlcDomainCalibration,
    NarrowTeamContentionCorrection,
    SaturatingServiceCurve,
    WideTeamPressureCalibration,
)
from test_moe_analytic_model import _calibration, _model  # noqa: E402


def _piecewise(points) -> SaturatingServiceCurve:
    return SaturatingServiceCurve(
        single_thread_rate=points[0][1],
        saturated_rate=points[-1][1],
        saturation_threads=points[-1][0],
        curve="piecewise_linear",
        points=tuple(points),
    )


def _contended_model():
    """Two LLC domains and every placed-DAG term switched on, with capacities small
    enough that LLC spill, domain dilation and DRAM injection all bind."""
    base = _calibration(cores=8)
    domain_curve = _piecewise(((1, 2e9), (2, 3e9), (4, 4e9)))
    calibration = replace(
        base,
        caches=replace(base.caches, llc_bytes_per_rank=96 * 1024, llc_effective_fraction=0.5),
        llc_bytes=_piecewise(((1, 2e9), (2, 3e9), (4, 4e9), (8, 6e9))),
        dram_bytes=_piecewise(((1, 1e9), (4, 2e9), (8, 2.5e9))),
        rank_cpu_ids=tuple(range(8)),
        llc_domains=(
            LlcDomainCalibration("0", (0, 1, 2, 3), 48 * 1024, domain_curve),
            LlcDomainCalibration("1", (4, 5, 6, 7), 48 * 1024, domain_curve),
        ),
        wide_team_pressure=WideTeamPressureCalibration(
            isolated_dilation=((4, 1.05), (8, 1.2)),
            full_cohort_dilation=((4, 1.1), (8, 1.4)),
        ),
        narrow_team_contention_correction=NarrowTeamContentionCorrection(
            full_cohort_correction=((1, 0.95), (2, 0.72)),
        ),
        dram_domain_injection=DramDomainInjectionCalibration(enabled=True, capacity_scale=0.6),
    )
    return _model(calibration)


def _native(model):
    native = model._native_placed_dag()
    if native is None:
        pytest.skip("native placed-DAG extension is not built")
    return native


def _random_plan(rng: random.Random, cores: int = 8):
    """Lanes over a random split of the cores; each lane is a dependency chain."""
    widths = []
    left = cores
    while left:
        width = rng.choice([w for w in (1, 2, 4) if w <= left])
        widths.append(width)
        left -= width
    rng.shuffle(widths)
    tasks, begin = [], 0
    for width in widths:
        cpus = tuple(range(begin, begin + width))
        previous = None
        for _ in range(rng.randint(1, 4)):
            tasks.append((rng.choice((1, 3, 12, 13, 48, 96, 300)), width, cpus, () if previous is None else (previous,)))
            previous = len(tasks) - 1
        begin += width
    return tasks


@pytest.mark.parametrize("seed", range(40))
def test_native_placed_dag_matches_python(seed: int) -> None:
    model = _contended_model()
    _native(model)
    tasks = _random_plan(random.Random(seed))

    assert model.dag_makespan_placed(tasks) == pytest.approx(model._python_dag_makespan_placed(tasks), rel=1e-12)


def test_native_batch_matches_single_scores_on_several_workers() -> None:
    model = _contended_model()
    _native(model)
    batch = [_random_plan(random.Random(seed)) for seed in range(16)]

    scores = model.dag_makespans_placed(batch, workers=4)

    assert scores == [model.dag_makespan_placed(tasks) for tasks in batch]
    assert scores == pytest.approx([model._python_dag_makespan_placed(tasks) for tasks in batch], rel=1e-12)


def test_native_placed_dag_keeps_the_python_validation() -> None:
    model = _contended_model()
    _native(model)
    with pytest.raises(ValueError, match="overlapping placed tasks"):
        model.dag_makespan_placed([(48, 2, (0, 1), ()), (48, 2, (1, 2), ())])
    with pytest.raises(ValueError, match="one unique CPU per thread"):
        model.dag_makespan_placed([(48, 2, (0, 0), ())])
    with pytest.raises(ValueError, match="earlier tasks"):
        model.dag_makespan_placed([(48, 2, (0, 1), (0,))])
    with pytest.raises(ValueError, match="outside the calibrated rank"):
        model.dag_makespan_placed([(48, 2, (0, 9), ())])
    # Ordered overlap through a transitive dependency is legal.
    chained = [(48, 2, (0, 1), ()), (12, 2, (2, 3), (0,)), (12, 4, (0, 1, 2, 3), (1,))]
    assert model.dag_makespan_placed(chained) == pytest.approx(model._python_dag_makespan_placed(chained), rel=1e-12)
    assert model.dag_makespan_placed([]) == 0.0


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("with_domains", (True, False))
def test_native_aggregate_dag_matches_python_makespan_and_finish_times(seed: int, with_domains: bool) -> None:
    model = _contended_model() if with_domains else _model()
    _native(model)
    tasks = [(routes, threads, deps) for routes, threads, _, deps in _random_plan(random.Random(seed))]

    reference_makespan, reference_finish = model._dag_result(tasks)
    assert model.dag_makespan(tasks) == pytest.approx(reference_makespan, rel=1e-12)
    assert model.dag_task_finish_times(tasks) == pytest.approx(reference_finish, rel=1e-12)


def test_native_aggregate_dag_keeps_the_python_validation() -> None:
    model = _model()
    _native(model)
    with pytest.raises(ValueError, match="must refer to earlier tasks"):
        model.dag_makespan([(48, 2, [1]), (12, 2, [])])
    # A duplicated dependency is counted and released twice, as _dag_state does.
    tasks = [(48, 2, []), (12, 2, [0, 0])]
    assert model.dag_makespan(tasks) == pytest.approx(model._dag_result(tasks)[0], rel=1e-12)
    assert model.dag_makespan([]) == 0.0
