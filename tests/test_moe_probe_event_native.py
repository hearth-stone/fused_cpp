from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "src"),
    str(ROOT / "cpu_moe_schedule_optimization" / "cost_model"),
    str(ROOT / "cpu_moe_schedule_optimization" / "planners"),
]

from probe_event_model import ProbeEventModel  # noqa: E402
from stage_window_policy import AMAZON_C9G_192C_TP4_F512_N8_V1, ARM_CODEX_NUMA3_80C_TP4_F512_N16_V4  # noqa: E402

ANALYTIC = dict(hidden_size=4096, intermediate_size=512, global_experts=256, local_experts=256, mode="tp",
                degree=4, concurrent_ranks=1, down_output_element_bytes=4)
ASSETS = {
    "arm_codex_v11": (ROOT / "bench_assets/moe_paper/arm_codex_numa3_80c_jemalloc/probe_event_v11_20260920.json",
                      ARM_CODEX_NUMA3_80C_TP4_F512_N16_V4, 80),
    "c9g_v11": (ROOT / "bench_assets/moe_paper/amazon_c9g_96c_tp4/probe_event_v11_c9g_20260924.json",
                AMAZON_C9G_192C_TP4_F512_N8_V1, 96),
}


def _scorer(name: str, windowed: bool = True):
    path, table, cores = ASSETS[name]
    if not path.exists():
        pytest.skip(f"calibration asset {path.name} is not present")
    model = ProbeEventModel.from_calibration(path, window_policy=table if windowed else None, **ANALYTIC)
    from probe_event_native import NativeProbeEventScorer

    try:
        return model, NativeProbeEventScorer(model), cores
    except RuntimeError:
        pytest.skip("native probe event simulator is not built")


def _random_plan(rng: random.Random, cores: int):
    """Lanes over the rank; each lane is a chain of experts with mixed route counts."""
    tasks, begin = [], 0
    while begin < cores:
        width = rng.choice([w for w in (2, 4, 8, 16, 32) if w <= cores - begin])
        previous = None
        for _ in range(rng.randint(1, 6)):
            routes = rng.choice((1, 5, 12, 13, 24, 48, 96, 200, 384, 700, 1500, 2040))
            tasks.append((routes, width, tuple(range(begin, begin + width)), () if previous is None else (previous,)))
            previous = len(tasks) - 1
        begin += width
    return tasks


@pytest.mark.parametrize("asset", sorted(ASSETS))
@pytest.mark.parametrize("windowed", (True, False))
@pytest.mark.parametrize("seed", range(12))
def test_native_scorer_matches_the_python_event_model(asset: str, windowed: bool, seed: int) -> None:
    model, scorer, cores = _scorer(asset, windowed)
    tasks = _random_plan(random.Random(seed), cores)

    reference = model.simulate(tasks)
    native = scorer.simulate(tasks)

    assert native["makespan_ns"] == pytest.approx(reference["makespan_ns"], rel=1e-9)
    assert native["call_ns"] == pytest.approx(reference["call_ns"], rel=1e-9)
    assert native["task_finish_ns"] == pytest.approx(reference["task_finish_ns"], rel=1e-9)


def test_native_scorer_delegates_and_keeps_python_only_paths() -> None:
    model, scorer, cores = _scorer("c9g_v11")
    tasks = _random_plan(random.Random(99), cores)

    assert scorer.T_iso(384, 4) == model.T_iso(384, 4)
    assert scorer.t_over_ns == model.t_over_ns
    events: list = []
    assert scorer.simulate(tasks, event_log=events)["makespan_ns"] == model.simulate(tasks)["makespan_ns"]
    assert events, "an event log request must run the Python model"
    with pytest.raises(ValueError, match="earlier tasks"):
        scorer.simulate([(12, 4, (0, 1, 2, 3), (0,))])
