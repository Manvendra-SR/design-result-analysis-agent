"""
backend/state_machine/nodes.py
================================
The five adaptive-loop nodes. Each is a method on ``AdaptiveLoopNodes``
(bound to a ``StateMachineContext``), and each follows the same contract:

    1. load everything it needs from PostgreSQL via ``StateManager``
    2. do its one job (call an agent or a tool)
    3. write the results back to PostgreSQL
    4. persist the new ``sessions.current_node``
    5. return ``{"current_node": <next node>}`` for LangGraph

No node keeps state in memory between invocations, and LangGraph state
carries nothing but ``session_id`` + ``current_node`` - so a crash between
any two nodes is recovered by re-reading ``sessions.current_node`` and
resuming (see ``executor.py``).

Node-by-node
------------
planning     -> plan_experiments(question, profile); queue configs; save the
                plan rationale                                             -> executing
executing    -> run each queued config, tagging results with the current
                cycle number; store them; clear the queue                  -> validating
validating   -> detect anomalies on successful experiments; flag them       -> analyzing
analyzing    -> pairwise t-tests between configuration groups; store them    -> recommending
recommending -> recommend_next(...); append a per-cycle history entry;
                (safety cap: force conclude at MAX_ADAPTIVE_CYCLES);
                branch on action:
                  conclude            -> mark session concluded             -> concluded
                  run_more_experiments -> queue the recommended configs      -> executing
                                          (the graph then loops back here)

Adaptive loop vs. crash recovery (see graph.py)
-----------------------------------------------
``recommending -> executing`` is a genuine conditional edge inside one
``graph.invoke()`` - the loop runs autonomously until the Recommender
concludes. The ``START`` router is separate: it only resumes a session at
its persisted ``current_node`` after a crash / on a fresh invocation.

Cycle history (see backend/models/cycle.py)
-------------------------------------------
``sessions.current_recommendation`` / ``sessions.latest_analysis`` /
``sessions.pending_configs`` are *latest-only* / ephemeral. Because the loop
now runs many cycles without a human in between, the per-cycle
recommendation + statistical analysis are also appended to
``sessions.cycle_history`` (a JSON array), and every experiment row carries
``experiments.cycle`` - together they let the API reconstruct the full
cycle-by-cycle investigation for the UI.

Requirements
------------
13.3  Planning node stores generated configs, transitions to execution
13.4  Execution node runs experiments, stores results
13.5  Validation node runs anomaly detection, updates statuses
13.6  Analysis node computes statistical comparisons
13.7  Recommendation node calls the Recommender agent
13.8  Recommendation node loops to execution or terminates
13.9  State persisted to PostgreSQL after each node
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from backend.config import MAX_ADAPTIVE_CYCLES
from backend.models.cycle import CycleHistoryEntry
from backend.models.experiment import ExperimentConfiguration, ExperimentResult
from backend.state_machine.context import StateMachineContext
from backend.state_machine.state import (
    ANALYZING,
    CONCLUDED,
    EXECUTING,
    RECOMMENDING,
    VALIDATING,
    AdaptiveLoopState,
)
from backend.tools.statistical_analyzer import (
    InsufficientDataError,
    InsufficientVarianceError,
)

logger = logging.getLogger(__name__)

_ConditionKey = Tuple[str, bool, Tuple[Tuple[str, float], ...]]


def _condition_key(config: ExperimentConfiguration) -> _ConditionKey:
    """A 'condition' is everything about a config except its random seed."""
    return (
        config.model_type,
        config.preprocessing.normalize,
        tuple(sorted(config.hyperparameters.items())),
    )


def _condition_label(config: ExperimentConfiguration) -> str:
    hp = ", ".join(f"{k}={v}" for k, v in sorted(config.hyperparameters.items()))
    parts = [config.model_type]
    if config.preprocessing.normalize:
        parts.append("normalized")
    if hp:
        parts.append(hp)
    return " | ".join(parts)


class AdaptiveLoopNodes:
    """The adaptive-loop node implementations, bound to a dependency context."""

    def __init__(self, context: StateMachineContext) -> None:
        self._ctx = context

    # ------------------------------------------------------------------
    # 1. Planning
    # ------------------------------------------------------------------
    def planning(self, state: AdaptiveLoopState) -> Dict[str, str]:
        sm = self._ctx.state_manager
        session_id = state["session_id"]
        session = sm.get_session(session_id)
        profile = sm.get_dataset(session.dataset_id)

        plan = self._ctx.planner.plan_experiments(session.research_question, profile)
        sm.save_planned_configs(session_id, plan.experiments)
        # The planner's rationale for the initial design - otherwise discarded.
        # The API attaches it to cycle 1 in the investigation history.
        sm.save_plan_explanation(session_id, plan.explanation)
        sm.update_session_node(session_id, EXECUTING)

        logger.info(
            "planning: session=%s -> %d configs queued", session_id, plan.total_count
        )
        return {"current_node": EXECUTING}

    # ------------------------------------------------------------------
    # 2. Execution
    # ------------------------------------------------------------------
    def executing(self, state: AdaptiveLoopState) -> Dict[str, str]:
        sm = self._ctx.state_manager
        session_id = state["session_id"]
        session = sm.get_session(session_id)
        profile = sm.get_dataset(session.dataset_id)

        configs = sm.load_planned_configs(session_id)
        # cycle_count is bumped by `recommending` at the END of each cycle, so
        # during cycle 1's execution it is 0, during cycle 2's it is 1, ...
        current_cycle = session.cycle_count + 1
        succeeded = failed = 0
        # Run one at a time and shrink the queue after each, so a crash
        # mid-batch resumes with only the not-yet-run configs still queued
        # (design.md's "crashed after 8/12 experiments" recovery scenario).
        for i, config in enumerate(configs):
            result = self._ctx.runner.run_experiment(config, profile, session_id=session_id)
            result.cycle = current_cycle  # which adaptive cycle produced this
            sm.store_experiment(result)
            sm.save_planned_configs(session_id, configs[i + 1 :])
            if result.status == "success":
                succeeded += 1
            else:
                failed += 1

        sm.update_session_node(session_id, VALIDATING)
        logger.info(
            "executing: session=%s cycle=%d -> %d experiments (%d ok, %d failed)",
            session_id, current_cycle, succeeded + failed, succeeded, failed,
        )
        return {"current_node": VALIDATING}

    # ------------------------------------------------------------------
    # 3. Validation (anomaly detection)
    # ------------------------------------------------------------------
    def validating(self, state: AdaptiveLoopState) -> Dict[str, str]:
        sm = self._ctx.state_manager
        session_id = state["session_id"]

        experiments = sm.query_experiments(session_id, status="success")
        try:
            anomalies = self._ctx.detector.detect_anomalies(experiments)
        except Exception as exc:  # noqa: BLE001 - graceful degradation (design.md)
            logger.warning(
                "validating: anomaly detection failed for session=%s, continuing: %s",
                session_id, exc,
            )
            anomalies = []

        # Idempotent across re-entry: skip (experiment, rule) pairs already recorded.
        already = {
            (a.experiment_id, a.rule) for a in sm.query_anomalies(session_id=session_id)
        }
        flagged = 0
        for anomaly in anomalies:
            if (anomaly.experiment_id, anomaly.rule) in already:
                continue
            sm.store_anomaly(anomaly)
            sm.update_experiment_status(anomaly.experiment_id, "anomalous")
            flagged += 1

        sm.update_session_node(session_id, ANALYZING)
        logger.info("validating: session=%s -> %d new anomaly report(s)", session_id, flagged)
        return {"current_node": ANALYZING}

    # ------------------------------------------------------------------
    # 4. Analysis (statistical comparisons)
    # ------------------------------------------------------------------
    def analyzing(self, state: AdaptiveLoopState) -> Dict[str, str]:
        sm = self._ctx.state_manager
        session_id = state["session_id"]
        session = sm.get_session(session_id)
        profile = sm.get_dataset(session.dataset_id)

        metric = "accuracy" if profile.task_type == "classification" else "val_loss"
        experiments = sm.query_experiments(session_id, status="success")
        comparisons = self._compare_conditions(experiments, metric)

        sm.save_analysis(session_id, comparisons)
        sm.update_session_node(session_id, RECOMMENDING)
        logger.info(
            "analyzing: session=%s -> %d pairwise comparison(s) on metric=%s",
            session_id, len(comparisons), metric,
        )
        return {"current_node": RECOMMENDING}

    def _compare_conditions(self, experiments: List[ExperimentResult], metric: str):
        groups: Dict[_ConditionKey, List[ExperimentResult]] = {}
        labels: Dict[_ConditionKey, str] = {}
        for exp in experiments:
            key = _condition_key(exp.config)
            groups.setdefault(key, []).append(exp)
            labels.setdefault(key, _condition_label(exp.config))

        keys = list(groups)
        out = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                try:
                    out.append(
                        self._ctx.analyzer.compare_conditions(
                            groups[keys[i]],
                            groups[keys[j]],
                            metric=metric,
                            condition_a_name=labels[keys[i]],
                            condition_b_name=labels[keys[j]],
                        )
                    )
                except (InsufficientDataError, InsufficientVarianceError):
                    # Not enough replicates / no variance for this pair yet -
                    # the recommender will see it as a knowledge gap.
                    continue
        return out

    # ------------------------------------------------------------------
    # 5. Recommendation
    # ------------------------------------------------------------------
    def recommending(self, state: AdaptiveLoopState) -> Dict[str, str]:
        sm = self._ctx.state_manager
        session_id = state["session_id"]
        session = sm.get_session(session_id)

        experiments = sm.query_experiments(session_id)
        stats = sm.load_analysis(session_id)
        anomalies = sm.query_anomalies(session_id=session_id)

        recommendation = self._ctx.recommender.recommend_next(
            session.research_question, experiments, stats, anomalies
        )
        next_cycle = session.cycle_count + 1

        # Safety cap: the graph loops `recommending -> executing` autonomously,
        # so a Recommender that never concludes would loop forever. Force a
        # conclusion at MAX_ADAPTIVE_CYCLES, and record *why* it stopped.
        if (
            recommendation.action == "run_more_experiments"
            and next_cycle >= MAX_ADAPTIVE_CYCLES
        ):
            recommendation = recommendation.model_copy(
                update={
                    "action": "conclude",
                    "recommended_experiments": [],
                    "evidence_summary": (
                        recommendation.evidence_summary
                        + f"  [Investigation stopped: reached the "
                        f"{MAX_ADAPTIVE_CYCLES}-cycle safety limit.]"
                    ),
                }
            )
            logger.warning(
                "recommending: session=%s hit the %d-cycle cap, forcing conclude",
                session_id, MAX_ADAPTIVE_CYCLES,
            )

        sm.save_recommendation(session_id, recommendation)
        # Permanent per-cycle record (current_recommendation / latest_analysis
        # are overwritten next cycle; this is what the UI reads for history).
        sm.append_cycle_history(
            session_id,
            CycleHistoryEntry(
                cycle_number=next_cycle,
                recommendation=recommendation,
                statistical_comparisons=stats,
            ),
        )

        if recommendation.action == "conclude":
            sm.update_session_node(session_id, CONCLUDED, cycle_count=next_cycle)
            sm.update_session_status(session_id, "concluded")
            logger.info(
                "recommending: session=%s -> conclude (cycle %d)", session_id, next_cycle
            )
            return {"current_node": CONCLUDED}

        sm.save_planned_configs(session_id, recommendation.recommended_experiments)
        sm.update_session_node(session_id, EXECUTING, cycle_count=next_cycle)
        logger.info(
            "recommending: session=%s -> run_more_experiments, %d config(s) queued "
            "(cycle %d complete, looping back to executing)",
            session_id, len(recommendation.recommended_experiments), next_cycle,
        )
        return {"current_node": EXECUTING}
