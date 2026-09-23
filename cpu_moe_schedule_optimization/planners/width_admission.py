"""Decide from measured plans which team widths quick planning may use.

Quick planning picks one homogeneous shape per workload by its model score. Given,
for a set of workloads, the measured time of every homogeneous width's plan and the
score quick ranks it by, the effect of any quick width set is known offline: the set
selects a width per workload, and that width's measured time is its outcome. This
module evaluates candidate widths against a base set under one frozen rule and
writes the verdict into a calibration payload (``planner.supported_widths`` and
``planner.unreliable_widths``) without new measurements.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

# Frozen admission rule (C9g width-error run, 2026-09-23): a width is excluded when
# it makes any workload more than SLOWER_LIMIT slower than the base set, admitted when
# it is otherwise at least WORKLOAD_GAIN faster on one workload or MEDIAN_GAIN faster
# in median, and neutral (not admitted) otherwise.
SLOWER_LIMIT = 0.01
WORKLOAD_GAIN = 0.01
MEDIAN_GAIN = 0.003

# Quick's own ordering of homogeneous candidates: model makespan, then active working
# set, then resource groups (IntervalPlanner.plan_quick).
QuickKey = tuple[float, float, float]


@dataclass(frozen=True)
class WidthVerdict:
    width: int
    verdict: str  # "admit", "exclude" or "neutral"
    changed_workloads: int
    median_change: float
    best_change: float
    worst_change: float


def select_width(scores: Mapping[int, QuickKey], widths: Sequence[int]) -> int:
    """The width quick selects among ``widths`` from its per-width ranking keys."""
    usable = [int(width) for width in widths if int(width) in scores]
    if not usable:
        raise ValueError(f"no scored width among {tuple(widths)}")
    return min(usable, key=lambda width: (tuple(scores[width]), width))


def evaluate(
    measured: Mapping[str, Mapping[int, float]],
    scores: Mapping[str, Mapping[int, QuickKey]],
    widths: Sequence[int],
) -> dict[str, float]:
    """Measured time of quick's selection with ``widths`` on every workload."""
    return {key: float(measured[key][select_width(scores[key], widths)]) for key in measured}


def judge(
    measured: Mapping[str, Mapping[int, float]],
    scores: Mapping[str, Mapping[int, QuickKey]],
    base_widths: Sequence[int],
    candidates: Sequence[int],
) -> list[WidthVerdict]:
    """Judge each candidate width added alone to ``base_widths``."""
    if set(measured) != set(scores):
        raise ValueError("measured times and scores must cover the same workloads")
    base = evaluate(measured, scores, base_widths)
    verdicts = []
    for width in candidates:
        widened = tuple(base_widths) + (int(width),)
        outcome = evaluate(measured, scores, widened)
        change = {key: outcome[key] / base[key] - 1.0 for key in base}
        worst, best = max(change.values()), min(change.values())
        median = statistics.median(change.values())
        if worst > SLOWER_LIMIT:
            verdict = "exclude"
        elif best < -WORKLOAD_GAIN or median < -MEDIAN_GAIN:
            verdict = "admit"
        else:
            verdict = "neutral"
        changed = sum(
            1 for key in base if select_width(scores[key], widened) != select_width(scores[key], base_widths)
        )
        verdicts.append(WidthVerdict(int(width), verdict, changed, median, best, worst))
    return verdicts


def admitted_set(
    measured: Mapping[str, Mapping[int, float]],
    scores: Mapping[str, Mapping[int, QuickKey]],
    base_widths: Sequence[int],
    verdicts: Sequence[WidthVerdict],
) -> tuple[int, ...]:
    """Admit greedily by median gain while the joint set stays within the slower limit."""
    base = evaluate(measured, scores, base_widths)
    joint = list(base_widths)
    for verdict in sorted((v for v in verdicts if v.verdict == "admit"), key=lambda v: (v.median_change, v.width)):
        trial = evaluate(measured, scores, tuple(joint) + (verdict.width,))
        if max(trial[key] / base[key] - 1.0 for key in base) <= SLOWER_LIMIT:
            joint.append(verdict.width)
    return tuple(sorted(set(joint) - set(base_widths)))


def apply_width_admission(
    payload: Mapping[str, object],
    *,
    planner_widths: Sequence[int],
) -> dict[str, object]:
    """Return a calibration payload whose planner searches exactly ``planner_widths``.

    Every other supported width stays runnable but is listed as unreliable. Each
    planner width must carry its own fitted overheads (``overheads.by_width``): the
    analytic model silently falls back to the common overheads for a width without
    them, which is how uncalibrated widths come to be priced optimistically.
    """
    result = copy.deepcopy(dict(payload))
    planner = dict(result["planner"])  # type: ignore[arg-type]
    supported = sorted({int(width) for width in planner["supported_widths"]})
    chosen = sorted({int(width) for width in planner_widths})
    if not chosen:
        raise ValueError("planner_widths must not be empty")
    missing = sorted(set(chosen) - set(supported))
    if missing:
        raise ValueError(f"planner widths {missing} are not supported by the calibration")
    fitted = {int(point["threads"]) for point in result.get("overheads", {}).get("by_width", ())}  # type: ignore[union-attr]
    unfitted = sorted(set(chosen) - fitted)
    if unfitted:
        raise ValueError(f"planner widths {unfitted} have no fitted per-width overheads")
    unreliable = sorted(set(supported) - set(chosen))
    planner["supported_widths"] = supported
    if unreliable:
        planner["unreliable_widths"] = unreliable
    else:
        planner.pop("unreliable_widths", None)
    result["planner"] = planner
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a calibration whose planner searches the given widths.")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--planner-widths", required=True, help="comma-separated widths, e.g. 4,6,8,16,24,32,48,96")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    widths = [int(value) for value in args.planner_widths.split(",") if value.strip()]
    payload = apply_width_admission(json.loads(args.calibration.read_text()), planner_widths=widths)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps(payload["planner"]))
    return 0


__all__ = [
    "MEDIAN_GAIN",
    "SLOWER_LIMIT",
    "WORKLOAD_GAIN",
    "WidthVerdict",
    "admitted_set",
    "apply_width_admission",
    "evaluate",
    "judge",
    "main",
    "select_width",
]


if __name__ == "__main__":
    raise SystemExit(main())
