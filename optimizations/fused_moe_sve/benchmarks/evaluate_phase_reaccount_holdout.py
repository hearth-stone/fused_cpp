#!/usr/bin/env python3
"""Evaluate one frozen phase-reaccount candidate on locked route and real-trace holdouts."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
COST_MODEL_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"
sys.path[:0] = [str(REPO_ROOT), str(COST_MODEL_DIR)]

from analytic_model import AnalyticMoeCostModel  # noqa: E402
from optimizations.fused_moe_sve.benchmarks.bench_small_expert_context import (  # noqa: E402
    MODES,
    _build_bridge,
    _model_target_span_ms,
)
from optimizations.fused_moe_sve.benchmarks.fit_phase_reaccount_calibration import (  # noqa: E402
    LOCKED_HOLDOUT_SHA256,
    STAGE_KEYS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-calibration", type=Path, required=True)
    parser.add_argument("--candidate-calibration", type=Path, required=True)
    parser.add_argument("--route-holdout", type=Path, action="append", required=True)
    parser.add_argument(
        "--real-trace",
        type=Path,
        nargs=2,
        action="append",
        metavar=("MEASUREMENT", "RESCORE"),
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_locked(path: Path) -> tuple[dict, str]:
    digest = _sha256(path)
    if digest not in LOCKED_HOLDOUT_SHA256:
        raise ValueError(f"artifact is not in the locked holdout set: {path}")
    return json.loads(path.read_text(encoding="utf-8")), digest


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _error_metrics(errors: list[float]) -> dict[str, float | int]:
    return {
        "points": len(errors),
        "mape": statistics.fmean(abs(value) for value in errors),
        "median_absolute_relative_error": statistics.median(abs(value) for value in errors),
        "p90_absolute_relative_error": _percentile([abs(value) for value in errors], 0.9),
        "max_absolute_relative_error": max(abs(value) for value in errors),
    }


def _route_model(calibration: Path, experts: int) -> AnalyticMoeCostModel:
    return AnalyticMoeCostModel(
        calibration,
        hidden_size=4096,
        intermediate_size=512,
        global_experts=experts,
        local_experts=experts,
        mode="tp",
        degree=4,
        concurrent_ranks=1,
        down_output_element_bytes=4,
    )


def _route_holdout_report(
    frozen_path: Path,
    candidate_path: Path,
    artifacts: list[tuple[dict, str, Path]],
) -> dict:
    expert_count = len(artifacts[0][0]["modes"]["full_head"].get("tasks", ())) or 55
    frozen = _route_model(frozen_path, expert_count)
    candidate = _route_model(candidate_path, expert_count)
    errors = {"frozen": [], "candidate": []}
    stage_errors = {"gather": [], "w13": [], "w2": [], "total": []}
    pair_rows = []
    topk = {"1": 0, "3": 0, "5": 0}
    rows = []
    for artifact, digest, path in artifacts:
        routes = int(artifact["shape"]["target_routes"])
        predictions = {"frozen": {}, "candidate": {}}
        for name, model in (("frozen", frozen), ("candidate", candidate)):
            for mode in MODES:
                bridge, route_map = _build_bridge(
                    model,
                    thread_cpu_ids=tuple(range(240, 320)),
                    mode=mode,
                    target_routes=routes,
                )
                predictions[name][mode] = _model_target_span_ms(model, bridge, route_map)
                observed = float(artifact["modes"][mode]["target_span"]["median_ms"])
                errors[name].append(predictions[name][mode] / observed - 1.0)
                rows.append(
                    {
                        "artifact": str(path),
                        "sha256": digest,
                        "routes": routes,
                        "mode": mode,
                        "model": name,
                        "predicted_ms": predictions[name][mode],
                        "measured_ms": observed,
                        "relative_error": predictions[name][mode] / observed - 1.0,
                    }
                )
        isolated = artifact["modes"]["isolated_head"]
        isolated_prediction = candidate.predict_expert(routes, 1)
        measured_stage = {
            "gather": float(isolated["stages"]["gather_pack_a"]["median_ms"]),
            "w13": float(isolated["stages"][STAGE_KEYS["w13"]]["median_ms"]),
            "w2": float(isolated["stages"][STAGE_KEYS["w2"]]["median_ms"]),
            "total": float(isolated["target_span"]["median_ms"]),
        }
        predicted_stage = {
            "gather": isolated_prediction.phases[0].base_ns / 1.0e6,
            "w13": isolated_prediction.w13_ns / 1.0e6,
            "w2": isolated_prediction.w2_ns / 1.0e6,
            "total": isolated_prediction.total_ns / 1.0e6,
        }
        for stage in stage_errors:
            stage_errors[stage].append(predicted_stage[stage] / measured_stage[stage] - 1.0)
        measured_order = sorted(
            MODES,
            key=lambda mode: float(artifact["modes"][mode]["target_span"]["median_ms"]),
        )
        predicted_order = sorted(MODES, key=lambda mode: predictions["candidate"][mode])
        measured_best = measured_order[0]
        for size in (1, 3, 5):
            topk[str(size)] += measured_best in predicted_order[:size]
        for left, right in itertools.combinations(MODES, 2):
            samples_left = artifact["modes"][left]["target_span"]["samples_ms"]
            samples_right = artifact["modes"][right]["target_span"]["samples_ms"]
            hardware_delta = [
                float(left_value) - float(right_value)
                for left_value, right_value in zip(samples_left, samples_right, strict=True)
            ]
            median_delta = statistics.median(hardware_delta)
            p10 = _percentile(hardware_delta, 0.1)
            p90 = _percentile(hardware_delta, 0.9)
            resolvable = p10 * p90 > 0.0
            predicted_delta = predictions["candidate"][left] - predictions["candidate"][right]
            correct = predicted_delta * median_delta > 0.0 if resolvable else None
            pair_rows.append(
                {
                    "routes": routes,
                    "left": left,
                    "right": right,
                    "measured_delta_ms": median_delta,
                    "measured_p10_ms": p10,
                    "measured_p90_ms": p90,
                    "predicted_delta_ms": predicted_delta,
                    "hardware_resolvable": resolvable,
                    "direction_correct": correct,
                }
            )
    resolvable_rows = [row for row in pair_rows if row["hardware_resolvable"]]
    false_dominance = sum(not bool(row["direction_correct"]) for row in resolvable_rows)
    return {
        "absolute_error": {name: _error_metrics(values) for name, values in errors.items()},
        "candidate_isolated_stage_error": {
            stage: _error_metrics(values)
            for stage, values in stage_errors.items()
        },
        "pairwise": {
            "pairs": len(pair_rows),
            "hardware_resolvable": len(resolvable_rows),
            "direction_correct": sum(bool(row["direction_correct"]) for row in resolvable_rows),
            "direction_accuracy": (
                statistics.fmean(bool(row["direction_correct"]) for row in resolvable_rows)
                if resolvable_rows
                else None
            ),
            "false_dominance": false_dominance,
            "rows": pair_rows,
        },
        "topk_measured_best_recall": {
            size: retained / len(artifacts)
            for size, retained in topk.items()
        },
        "rows": rows,
    }


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    begin = 0
    while begin < len(order):
        end = begin + 1
        while end < len(order) and values[order[end]] == values[order[begin]]:
            end += 1
        rank = (begin + end - 1) / 2.0
        for offset in range(begin, end):
            ranks[order[offset]] = rank
        begin = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_rank = _rank(left)
    right_rank = _rank(right)
    left_mean = statistics.fmean(left_rank)
    right_mean = statistics.fmean(right_rank)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left_rank, right_rank))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left_rank)
        * sum((y - right_mean) ** 2 for y in right_rank)
    )
    return numerator / denominator if denominator else None


def _real_trace_report(measurement_path: Path, rescore_path: Path) -> dict:
    measurement, digest = _read_locked(measurement_path)
    rescore = json.loads(rescore_path.read_text(encoding="utf-8"))
    score_by_hash = {
        str(row["state_hash"]): float(row["event_ns"]) / 1.0e6
        for strategy in rescore["strategies"].values()
        for row in strategy["scored"]
    }
    baseline_hash = str(measurement["baseline"]["state_hash"])
    score_by_hash[baseline_hash] = float(rescore["baseline"]["event_ms"])
    hardware = measurement["hardware_shortlist"]["measurement"]
    measured = [(baseline_hash, hardware["baseline"]["samples_ms"])] + [
        (str(row["state_hash"]), row["stats"]["samples_ms"])
        for row in hardware["candidates"]
    ]
    missing = sorted(state_hash for state_hash, _ in measured if state_hash not in score_by_hash)
    available = [(state_hash, samples) for state_hash, samples in measured if state_hash in score_by_hash]
    measured_medians = [statistics.median(float(value) for value in samples) for _, samples in available]
    predicted_scores = [score_by_hash[state_hash] for state_hash, _ in available]
    measured_best_hash = available[min(range(len(available)), key=measured_medians.__getitem__)][0]
    predicted_order = [
        state_hash
        for state_hash, _ in sorted(
            ((state_hash, score_by_hash[state_hash]) for state_hash, _ in available),
            key=lambda item: (item[1], item[0]),
        )
    ]
    pair_rows = []
    for (left_hash, left_samples), (right_hash, right_samples) in itertools.combinations(available, 2):
        deltas = [
            float(left) - float(right)
            for left, right in zip(left_samples, right_samples, strict=True)
        ]
        median_delta = statistics.median(deltas)
        p10 = _percentile(deltas, 0.1)
        p90 = _percentile(deltas, 0.9)
        resolvable = p10 * p90 > 0.0
        predicted_delta = score_by_hash[left_hash] - score_by_hash[right_hash]
        pair_rows.append(
            {
                "left": left_hash,
                "right": right_hash,
                "measured_delta_ms": median_delta,
                "measured_p10_ms": p10,
                "measured_p90_ms": p90,
                "predicted_delta_ms": predicted_delta,
                "hardware_resolvable": resolvable,
                "direction_correct": predicted_delta * median_delta > 0.0 if resolvable else None,
            }
        )
    resolvable = [row for row in pair_rows if row["hardware_resolvable"]]
    return {
        "measurement": str(measurement_path),
        "measurement_sha256": digest,
        "rescore": str(rescore_path),
        "rescore_sha256": _sha256(rescore_path),
        "available_states": len(available),
        "missing_state_hashes": missing,
        "spearman": _spearman(predicted_scores, measured_medians),
        "measured_best_state_hash": measured_best_hash,
        "topk_recall": {
            str(size): measured_best_hash in predicted_order[: min(size, len(predicted_order))]
            for size in (8, 16, 32)
        },
        "pairwise": {
            "pairs": len(pair_rows),
            "hardware_resolvable": len(resolvable),
            "direction_correct": sum(bool(row["direction_correct"]) for row in resolvable),
            "direction_accuracy": (
                statistics.fmean(bool(row["direction_correct"]) for row in resolvable)
                if resolvable
                else None
            ),
            "false_dominance": sum(not bool(row["direction_correct"]) for row in resolvable),
            "rows": pair_rows,
        },
    }


def main() -> int:
    args = parse_args()
    route_artifacts = [
        (*_read_locked(path), path)
        for path in args.route_holdout
    ]
    report = {
        "kind": "moe_phase_reaccount_holdout_report",
        "artifact_role": "holdout_only",
        "identity": {
            "frozen_calibration": str(args.frozen_calibration),
            "frozen_calibration_sha256": _sha256(args.frozen_calibration),
            "candidate_calibration": str(args.candidate_calibration),
            "candidate_calibration_sha256": _sha256(args.candidate_calibration),
        },
        "route_context": _route_holdout_report(
            args.frozen_calibration,
            args.candidate_calibration,
            route_artifacts,
        ),
        "real_traces": [
            _real_trace_report(measurement, rescore)
            for measurement, rescore in args.real_trace
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
