"""Formula-generated shadow policy for exact per-thread stage windows.

The policy deliberately stays outside the production planner until its
cross-machine holdout passes. It has no route table: for an already selected
``(M, threads)`` point it derives cache-boundary candidates from the packed tile,
M-panel, and private-cache geometry, then asks :class:`AnalyticMoeCostModel` to
score those exact runtime windows.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache


FULL_STRIPE = 0
POLICY_VERSION = 6


@dataclass(frozen=True)
class AnalyticStageWindowDecision:
    routes: int
    threads: int
    cohort_threads: int
    w13_window_tiles: int
    w2_window_tiles: int
    w13_resolved_tiles: int
    w2_resolved_tiles: int
    w13_windows: int
    w2_windows: int
    w13_worker_bytes: int
    w2_worker_bytes: int


@dataclass(frozen=True)
class AnalyticStageWindowEvaluation:
    """One task-local score plus the full-rank reusable-B spill surcharge."""

    score: object
    shared_b_llc_spill_ns: float
    policy_objective_ns: float


class AnalyticStageWindowPolicy:
    """Select W13/W2 windows from cache geometry and service demand.

    Candidate generation is structural rather than empirical. It contains the
    full stripe and power-of-two owner windows at or above one L1D. The
    base objective is task-local. The deterministic pre-policy also adds only
    the *incremental* DRAM service for packed-B reuses that stop fitting in the
    shared LLC when a homogeneous wave fills the rank. Streaming A and C stores
    are transfer demand, not resident reuse sets, so they do not consume this
    window budget. Compulsory B traffic and task-local spill are already in the
    base score and are not counted again. Callers may pass ``cohort_threads``
    to use a smaller known cohort; otherwise one complete rank is the
    conservative pre-planning reference.
    """

    def __init__(self, model, *, cohort_threads: int | None = None) -> None:
        self.model = model
        requested = None if cohort_threads is None else int(cohort_threads)
        if requested is not None and (requested <= 0 or requested > model.calibration.cores_per_rank):
            raise ValueError("cohort_threads must be in [1, cores_per_rank]")
        self.cohort_threads = requested
        identity = {
            "version": POLICY_VERSION,
            "machine": model.calibration.to_dict(),
            "hidden_size": model.hidden_size,
            "intermediate_size": model.intermediate_size,
            "backend_n_tile": model.policy.backend_n_tile,
            "cohort_threads": self.cohort_threads,
            "supported_widths": model.supported_widths,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:12]
        self.name = (
            f"analytic_stage_window_v{POLICY_VERSION}:"
            f"{model.calibration.machine_id}:{digest}"
        )

    def _geometry(self, stage: str):
        if stage == "w13":
            return self.model._w13_geometry
        if stage == "w2":
            return self.model._w2_geometry
        raise ValueError(f"unsupported stage {stage!r}")

    def _resolved_cohort_threads(self, threads: int) -> int:
        if self.cohort_threads is None:
            return (self.model.calibration.cores_per_rank // threads) * threads
        complete_teams = self.cohort_threads // threads
        return max(complete_teams, 1) * threads

    @lru_cache(maxsize=4096)
    def candidate_window_tiles(self, stage: str, routes: int, threads: int) -> tuple[int, ...]:
        stage = str(stage)
        routes = int(routes)
        threads = int(threads)
        if routes <= 0 or threads <= 0:
            raise ValueError("routes and threads must be positive")
        if threads not in self.model.supported_widths:
            raise KeyError(f"unsupported analytical thread width {threads}")

        _, geometry = self.model._stage_mapping_and_geometry(stage, routes, threads)
        full_tiles = geometry.tiles_per_worker(threads)
        if routes <= 12:
            return (full_tiles,)

        tile_bytes = geometry.bytes_per_tile
        cache = self.model.calibration.caches
        minimum_tiles = max(1, math.ceil(cache.l1d_bytes_per_core / tile_bytes))
        preferred_tiles = max(minimum_tiles, 2)
        candidates = {full_tiles, min(full_tiles, preferred_tiles)}

        power = 1
        while power <= full_tiles:
            if power >= minimum_tiles:
                candidates.add(power)
            power *= 2

        legal: list[int] = []
        for candidate in sorted(candidates):
            if candidate != full_tiles and candidate < minimum_tiles:
                continue
            plan = geometry.window_plan(threads, candidate)
            if candidate != full_tiles and plan.starves_any_thread():
                continue
            legal.append(candidate)
        if full_tiles not in legal:
            legal.append(full_tiles)
        return tuple(sorted(set(legal)))

    @lru_cache(maxsize=4096)
    def stage_scores(self, stage: str, routes: int, threads: int):
        geometry = self._geometry(stage)
        full_tiles = geometry.tiles_per_worker(int(threads))
        return tuple(
            self.model.score_stage_window(
                stage,
                int(routes),
                int(threads),
                FULL_STRIPE if tiles == full_tiles else tiles,
            )
            for tiles in self.candidate_window_tiles(stage, int(routes), int(threads))
        )

    def _shared_b_llc_spill_ns(self, score) -> float:
        cohort_threads = self._resolved_cohort_threads(score.threads)
        if cohort_threads <= score.threads:
            return 0.0
        cohort = self.model.score_stage_window(
            score.stage,
            score.routes,
            score.threads,
            FULL_STRIPE if score.is_full_stripe else score.window_tiles,
            cohort_threads,
        )
        if len(score.window_demands) != len(cohort.window_demands):
            raise RuntimeError("task-local and cohort stage-window geometry disagree")

        surcharge_ns = 0.0
        for local_window, cohort_window in zip(score.window_demands, cohort.window_demands):
            local_spill = self.model._llc_miss_fraction(local_window.reusable_b_bytes)
            cohort_b_working_set = cohort_window.reusable_b_bytes * cohort.cohort_tasks
            cohort_spill = self.model._llc_miss_fraction(cohort_b_working_set)
            additional_fraction = max(cohort_spill - local_spill, 0.0)
            reusable_b_refill_bytes = max(
                cohort_window.b_l2_refill_bytes - cohort_window.compulsory_dram_bytes,
                0.0,
            )
            additional_bytes = (
                additional_fraction
                * reusable_b_refill_bytes
                * cohort.cohort_tasks
            )
            active_threads = cohort_window.active_threads * cohort.cohort_tasks
            surcharge_ns += (
                additional_bytes
                / self.model.calibration.service_rate("dram_bytes", active_threads)
                * 1e9
            )
        return surcharge_ns

    @lru_cache(maxsize=4096)
    def stage_evaluations(
        self,
        stage: str,
        routes: int,
        threads: int,
    ) -> tuple[AnalyticStageWindowEvaluation, ...]:
        evaluations = []
        for score in self.stage_scores(stage, routes, threads):
            shared_b_spill_ns = self._shared_b_llc_spill_ns(score)
            evaluations.append(
                AnalyticStageWindowEvaluation(
                    score=score,
                    shared_b_llc_spill_ns=shared_b_spill_ns,
                    policy_objective_ns=score.objective_ns + shared_b_spill_ns,
                )
            )
        return tuple(evaluations)

    def stage_evaluation(
        self,
        stage: str,
        routes: int,
        threads: int,
        window_tiles: int,
    ) -> AnalyticStageWindowEvaluation:
        geometry = self._geometry(stage)
        resolved = geometry.tiles_per_worker(int(threads)) if int(window_tiles) == FULL_STRIPE else int(window_tiles)
        for evaluation in self.stage_evaluations(stage, int(routes), int(threads)):
            if evaluation.score.window_tiles == resolved:
                return evaluation
        raise KeyError(
            f"window {window_tiles} (resolved {resolved}) is not a structural candidate "
            f"for {stage} M={routes} T={threads}"
        )

    @lru_cache(maxsize=4096)
    def _selected_stage(self, stage: str, routes: int, threads: int):
        evaluations = self.stage_evaluations(stage, routes, threads)
        scores = tuple(evaluation.score for evaluation in evaluations)
        full = next(score for score in scores if score.is_full_stripe)
        if routes <= 12:
            return scores, full

        minimum = min(
            evaluations,
            key=lambda evaluation: (
                evaluation.policy_objective_ns,
                -evaluation.score.window_tiles,
            ),
        )
        resolution_ns = self.model.calibration.relative_uncertainty * (
            minimum.score.transfer_ns + minimum.shared_b_llc_spill_ns
        )
        equivalent = tuple(
            evaluation
            for evaluation in evaluations
            if evaluation.policy_objective_ns <= minimum.policy_objective_ns + resolution_ns
        )
        tile_bytes = self._geometry(stage).bytes_per_tile
        cache = self.model.calibration.caches
        cache_midpoint_bytes = math.sqrt(
            cache.l1d_bytes_per_core * cache.effective_l2_b_reuse_bytes_per_core
        )
        preferred_bytes = max(
            cache.l1d_bytes_per_core,
            2 * tile_bytes,
            cache_midpoint_bytes,
        )
        selected = min(
            equivalent,
            key=lambda evaluation: (
                abs(math.log2(evaluation.score.owner_window_bytes / preferred_bytes)),
                evaluation.policy_objective_ns,
                -evaluation.score.window_tiles,
            ),
        )
        return scores, selected.score

    @staticmethod
    def _abi_tiles(score) -> int:
        return FULL_STRIPE if score.is_full_stripe else score.window_tiles

    @lru_cache(maxsize=4096)
    def decision(self, routes: int, threads: int) -> AnalyticStageWindowDecision:
        routes = int(routes)
        threads = int(threads)
        if routes <= 0 or threads <= 0:
            raise ValueError("routes and threads must be positive")
        _, w13 = self._selected_stage("w13", routes, threads)
        _, w2 = self._selected_stage("w2", routes, threads)
        return AnalyticStageWindowDecision(
            routes=routes,
            threads=threads,
            cohort_threads=self._resolved_cohort_threads(threads),
            w13_window_tiles=self._abi_tiles(w13),
            w2_window_tiles=self._abi_tiles(w2),
            w13_resolved_tiles=w13.window_tiles,
            w2_resolved_tiles=w2.window_tiles,
            w13_windows=w13.windows,
            w2_windows=w2.windows,
            w13_worker_bytes=w13.owner_window_bytes,
            w2_worker_bytes=w2.owner_window_bytes,
        )

    def select(self, routes: int, threads: int) -> tuple[int, int]:
        decision = self.decision(int(routes), int(threads))
        return decision.w13_window_tiles, decision.w2_window_tiles

    @staticmethod
    def _score_row(score, evaluation: AnalyticStageWindowEvaluation | None = None) -> dict:
        row = asdict(score)
        row.pop("window_demands")
        row["is_full_stripe"] = score.is_full_stripe
        if evaluation is not None:
            row["shared_b_llc_spill_ns"] = evaluation.shared_b_llc_spill_ns
            row["policy_objective_ns"] = evaluation.policy_objective_ns
        return row

    @staticmethod
    def _delta(candidate, selected) -> dict:
        return {
            "window_tiles": candidate.score.window_tiles,
            "policy_objective_ns": candidate.policy_objective_ns - selected.policy_objective_ns,
            "shared_b_llc_spill_ns": candidate.shared_b_llc_spill_ns - selected.shared_b_llc_spill_ns,
            "task_local_objective_ns": candidate.score.objective_ns - selected.score.objective_ns,
            "a_l2_refill_bytes": candidate.score.a_l2_refill_bytes - selected.score.a_l2_refill_bytes,
            "b_l2_refill_bytes": candidate.score.b_l2_refill_bytes - selected.score.b_l2_refill_bytes,
            "transfer_ns": candidate.score.transfer_ns - selected.score.transfer_ns,
            "window_overhead_ns": candidate.score.window_overhead_ns - selected.score.window_overhead_ns,
        }

    def explain(self, routes: int, threads: int) -> dict:
        routes = int(routes)
        threads = int(threads)
        decision = self.decision(routes, threads)
        payload = {
            "policy": self.name,
            "decision": asdict(decision),
            "reason": (
                "single_panel_has_no_packed_b_reuse"
                if routes <= 12
                else "uncertainty_robust_cache_service_objective"
            ),
            "candidates": {},
        }
        for stage in ("w13", "w2"):
            evaluations = self.stage_evaluations(stage, routes, threads)
            _, selected_score = self._selected_stage(stage, routes, threads)
            selected = next(item for item in evaluations if item.score == selected_score)
            minimum = min(
                evaluations,
                key=lambda item: (item.policy_objective_ns, -item.score.window_tiles),
            )
            full = next(item for item in evaluations if item.score.is_full_stripe)
            ordered = sorted(evaluations, key=lambda item: item.score.window_tiles)
            selected_index = ordered.index(selected)
            smaller = ordered[selected_index - 1] if selected_index > 0 else None
            larger = ordered[selected_index + 1] if selected_index + 1 < len(ordered) else None
            payload["candidates"][stage] = {
                "selected": self._score_row(selected.score, selected),
                "minimum": self._score_row(minimum.score, minimum),
                "full_stripe": self._score_row(full.score, full),
                "objective_resolution_ns": (
                    self.model.calibration.relative_uncertainty
                    * (minimum.score.transfer_ns + minimum.shared_b_llc_spill_ns)
                ),
                "selected_vs_full": self._delta(selected, full),
                "smaller_vs_selected": self._delta(smaller, selected) if smaller else None,
                "larger_vs_selected": self._delta(larger, selected) if larger else None,
                "rows": [self._score_row(item.score, item) for item in evaluations],
            }
        return payload

    def cost_model_entries(self) -> tuple[object, ...]:
        """The formula-generated mapping intentionally has no finite table."""
        return ()


__all__ = [
    "AnalyticStageWindowDecision",
    "AnalyticStageWindowEvaluation",
    "AnalyticStageWindowPolicy",
    "FULL_STRIPE",
    "POLICY_VERSION",
]
