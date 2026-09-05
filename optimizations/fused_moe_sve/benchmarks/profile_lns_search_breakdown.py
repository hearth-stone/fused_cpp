#!/usr/bin/env python3
"""Profile template-LNS enumeration vs scoring on one frozen parent state."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(REPO_ROOT),
    str(REPO_ROOT / "src"),
    str(REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"),
    str(REPO_ROOT / "cpu_moe_schedule_optimization" / "planners"),
]

from analytic_model import AnalyticMoeCostModel  # noqa: E402
from executable_plan_neighborhood import (  # noqa: E402
    ExecutablePlanEvaluator,
    sample_template_lns_neighborhood,
)
from lns_diverse_shortlist import state_from_canonical_payload  # noqa: E402


def _peak_rss_kb() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(round(usage / 1024.0))
    return int(usage)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--analytic-calibration", type=Path, required=True)
    parser.add_argument("--critical-experts", type=int, default=32)
    parser.add_argument("--neighbors-per-operator", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.model_artifact.read_text(encoding="utf-8"))
    run = payload["runs"][args.start]
    state = state_from_canonical_payload(run["initial_canonical_state"])
    model = AnalyticMoeCostModel(
        args.analytic_calibration,
        hidden_size=int(payload["shape"]["hidden"]),
        intermediate_size=int(payload["shape"]["intermediate"]),
        global_experts=int(payload["shape"]["experts"]),
        local_experts=int(payload["shape"]["experts"]),
        mode="tp",
        degree=4,
        concurrent_ranks=1,
        down_output_element_bytes=4,
    )
    widths = tuple(model.supported_widths)
    expert_ids = list(run["iterations"][0]["critical_expert_ids"])[: args.critical_experts]
    evaluator = ExecutablePlanEvaluator(model)

    def window_selector(routes: int, threads: int) -> tuple[int, int]:
        return (0, 0)

    begin = time.perf_counter_ns()
    sampled = sample_template_lns_neighborhood(
        state,
        allowed_widths=widths,
        isolated_cost=model.T_iso,
        window_selector=window_selector,
        critical_expert_ids=expert_ids,
        per_operator=args.neighbors_per_operator,
        seed=args.seed,
    )
    sample_s = (time.perf_counter_ns() - begin) / 1.0e9
    screen_begin = time.perf_counter_ns()
    for neighbor in sampled.neighbors:
        evaluator.screen(neighbor.state)
    screen_s = (time.perf_counter_ns() - screen_begin) / 1.0e9
    exact_begin = time.perf_counter_ns()
    for neighbor in sampled.neighbors:
        evaluator.exact(neighbor.state)
    exact_s = (time.perf_counter_ns() - exact_begin) / 1.0e9
    sampled_hashes = [neighbor.state.canonical_hash() for neighbor in sampled.neighbors]
    result = {
        "start": args.start,
        "parent_state_hash": state.canonical_hash(),
        "parent_shape": list(state.shape),
        "critical_experts": len(expert_ids),
        "sampled": len(sampled.neighbors),
        "sampled_hashes_sha256": hashlib.sha256("".join(sampled_hashes).encode("utf-8")).hexdigest(),
        "proposed": sampled.proposed,
        "unique": sampled.unique,
        "duplicates": sampled.duplicates,
        "sample_s": sample_s,
        "screen_s": screen_s,
        "exact_s": exact_s,
        "exact_calls": evaluator.exact_calls,
        "breakdown": sampled.breakdown,
        "ru_maxrss_kb": _peak_rss_kb(),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
