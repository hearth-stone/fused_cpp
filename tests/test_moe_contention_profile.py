from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
COST_MODEL = ROOT / "cpu_moe_schedule_optimization" / "cost_model"
sys.path.insert(0, str(COST_MODEL))

import profile_contention_async as profile  # noqa: E402


def test_make_async_run_reuses_native_output(monkeypatch) -> None:
    """Timed profiler calls must reuse one output allocation after warmup."""
    outputs: list[torch.Tensor] = []

    def fake_fused_moe(*_args, **kwargs) -> torch.Tensor:
        output = kwargs["out"]
        outputs.append(output)
        output.fill_(len(outputs))
        return output

    monkeypatch.setattr(profile, "fused_moe_bf16_tiled_async", fake_fused_moe)
    run, groups, tasks, lane_counts = profile.make_async_run(
        packed=object(),
        hidden_size=4,
        routes=1,
        shape=[1],
        measurement_experts=2,
        num_profile_experts=2,
        cpu_ids=[0],
        w13_ranges=2,
        w2_ranges=1,
        generator=torch.Generator().manual_seed(0),
        std=0.01,
    )

    first = run()
    second = run()

    assert groups == 2
    assert tasks == 2
    assert lane_counts == [2]
    assert first.data_ptr() == second.data_ptr()
    assert outputs[0].data_ptr() == outputs[1].data_ptr()
