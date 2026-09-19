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
validating   -> RE-EVALUATE anomalies across every completed experiment and
                reconcile against the stored flags (raise / withdraw /
                re-open); sync experiment statuses                          -> analyzing
analyzing    -> pairwise comparisons between conditions + per-condition
                summaries; every skipped pair recorded with its reason       -> recommending
recommending -> recommend_next(...); append a per-cycle history entry;
                branch on the agent's action, with the MAX_ADAPTIVE_CYCLES
                cap as a separate, recorded stop:
                  conclude             -> concluded (termination: agent_concluded)
                  run_more_experiments -> executing, unless the cap is
                                          reached (termination: cycle_limit)

Cumulative evidence, per-cycle records
---------------------------------------
The Recommender deliberately reasons over *all* evidence gathered so far -
that is what makes the loop adaptive, and why its prose legitimately refers
to earlier cycles. What was missing was attribution: it received experiments
and anomalies with no cycle on them, so it could only speak timelessly
("two runs were flagged as anomalous") even in a cycle that flagged nothing.
It now receives the cycle of every experiment and anomaly, whether each
anomaly is still open, and where it is in the cycle budget. See
``agents/recommender.py`` and ``models/cycle.py``.

Anomaly lifecycle
-----------------
``validating`` re-runs detection over every completed experiment each cycle
and reconciles, rather than only appending. A flag raised against a thin
3-replicate group is withdrawn once the group grows and the value proves
ordinary, and the experiment returns to ``success`` so it counts in the
statistics again. Without this, "no unresolved anomalies" - one of the two
conditions the Recommender needs in order to conclude - was unreachable, and
every investigation ran to the safety cap.

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
from backend.models.anomaly import AnomalyReport
from backend.models.condition import ConditionKey, condition_key, condition_label
from backend.models.cycle import CycleHistoryEntry
from backend.models.experiment import ExperimentResult
from backend.models.statistics import AnalysisSnapshot, ComparisonSkip
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

#: (experiment_id, rule) - the identity of one anomaly finding, used to
#: reconcile freshly detected anomalies against the stored ones.
_AnomalyKey = Tuple[str, str]


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
        current_cycle = _current_cycle(session)
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
    # 3. Validation (anomaly detection + lifecycle reconciliation)
    # ------------------------------------------------------------------
    def validating(self, state: AdaptiveLoopState) -> Dict[str, str]:
        sm = self._ctx.state_manager
        session_id = state["session_id"]
        session = sm.get_session(session_id)
        current_cycle = _current_cycle(session)

        # Every experiment, including ones currently flagged anomalous: the
        # detector reports what holds *now*, over the full evidence.
        experiments = sm.query_experiments(session_id)
        try:
            detected = self._ctx.detector.detect_anomalies(experiments)
        except Exception as exc:  # noqa: BLE001 - graceful degradation (design.md)
            # Detection failing must not silently look like "nothing is wrong":
            # that would withdraw every open flag. Leave the stored flags exactly
            # as they are and move on.
            logger.warning(
                "validating: anomaly detection failed for session=%s, keeping "
                "existing flags unchanged: %s",
                session_id, exc,
            )
            sm.update_session_node(session_id, ANALYZING)
            return {"current_node": ANALYZING}

        raised, resolved, reopened = self._reconcile_anomalies(
            session_id, detected, current_cycle
        )
        self._sync_experiment_statuses(experiments, detected)

        sm.update_session_node(session_id, ANALYZING)
        logger.info(
            "validating: session=%s cycle=%d -> %d open flag(s) "
            "(%d newly raised, %d withdrawn, %d re-opened)",
            session_id, current_cycle, len(detected), raised, resolved, reopened,
        )
        return {"current_node": ANALYZING}

    def _reconcile_anomalies(
        self,
        session_id: str,
        detected: List[AnomalyReport],
        current_cycle: int,
    ) -> Tuple[int, int, int]:
        """Bring the stored anomaly flags in line with what currently holds.

        Returns ``(raised, resolved, reopened)``. Reports are never deleted -
        a withdrawn flag keeps its row with ``resolved_cycle`` set, so the
        history of what was flagged and when it cleared stays visible.
        """
        sm = self._ctx.state_manager
        current: Dict[_AnomalyKey, AnomalyReport] = {
            (a.experiment_id, a.rule): a for a in detected
        }
        stored: Dict[_AnomalyKey, AnomalyReport] = {
            (a.experiment_id, a.rule): a
            for a in sm.query_anomalies(session_id=session_id)
        }

        raised = resolved = reopened = 0

        for key, anomaly in current.items():
            existing = stored.get(key)
            if existing is None:
                anomaly.detected_cycle = current_cycle
                sm.store_anomaly(anomaly)
                raised += 1
            elif not existing.is_open:
                # It held again after having been withdrawn - re-open the
                # original report rather than creating a duplicate.
                sm.set_anomaly_resolution(existing.anomaly_id, None)
                reopened += 1

        for key, existing in stored.items():
            if existing.is_open and key not in current:
                sm.set_anomaly_resolution(existing.anomaly_id, current_cycle)
                resolved += 1

        return raised, resolved, reopened

    def _sync_experiment_statuses(
        self,
        experiments: List[ExperimentResult],
        detected: List[AnomalyReport],
    ) -> None:
        """Make ``experiments.status`` agree with the open flags.

        An experiment is ``anomalous`` exactly while it carries at least one
        open flag, and returns to ``success`` when its last flag is withdrawn -
        which puts it back into the statistical analysis, where it belongs.
        ``failed`` runs (no metrics) are never touched.
        """
        sm = self._ctx.state_manager
        flagged = {a.experiment_id for a in detected}
        for exp in experiments:
            if exp.status == "failed":
                continue
            desired = "anomalous" if exp.experiment_id in flagged else "success"
            if exp.status != desired:
                sm.update_experiment_status(exp.experiment_id, desired)

    # ------------------------------------------------------------------
    # 4. Analysis (statistical comparisons + per-condition summaries)
    # ------------------------------------------------------------------
    def analyzing(self, state: AdaptiveLoopState) -> Dict[str, str]:
        sm = self._ctx.state_manager
        session_id = state["session_id"]
        session = sm.get_session(session_id)
        profile = sm.get_dataset(session.dataset_id)

        metric = "accuracy" if profile.task_type == "classification" else "val_loss"
        # All experiments, not just successful ones: the analyzer itself uses
        # only successful values, but the per-condition summaries need to
        # report how many replicates were anomalous or failed.
        experiments = sm.query_experiments(session_id)
        analysis = self._analyze(experiments, metric)

        sm.save_analysis(session_id, analysis)
        sm.update_session_node(session_id, RECOMMENDING)
        logger.info(
            "analyzing: session=%s -> %d comparison(s), %d skipped, "
            "%d condition(s) on metric=%s",
            session_id,
            len(analysis.comparisons),
            len(analysis.skipped),
            len(analysis.condition_summaries),
            metric,
        )
        return {"current_node": RECOMMENDING}

    def _analyze(
        self, experiments: List[ExperimentResult], metric: str
    ) -> AnalysisSnapshot:
        """Compare every pair of conditions, recording both results and skips.

        A pair that cannot be tested is a knowledge gap, not a non-event: the
        reason is captured so the UI can state it and the Recommender can act
        on it. Previously these were caught and dropped, which made "no
        comparisons" indistinguishable from "nothing to compare".
        """
        groups: Dict[ConditionKey, List[ExperimentResult]] = {}
        labels: Dict[ConditionKey, str] = {}
        for exp in experiments:
            key = condition_key(exp.config)
            groups.setdefault(key, []).append(exp)
            labels.setdefault(key, condition_label(exp.config))

        keys = list(groups)
        comparisons = []
        skipped: List[ComparisonSkip] = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                name_a, name_b = labels[keys[i]], labels[keys[j]]
                try:
                    comparisons.append(
                        self._ctx.analyzer.compare_conditions(
                            groups[keys[i]],
                            groups[keys[j]],
                            metric=metric,
                            condition_a_name=name_a,
                            condition_b_name=name_b,
                        )
                    )
                except (InsufficientDataError, InsufficientVarianceError) as exc:
                    skipped.append(
                        ComparisonSkip(
                            condition_a_name=name_a,
                            condition_b_name=name_b,
                            metric=metric,
                            reason_code=(
                                "insufficient_variance"
                                if isinstance(exc, InsufficientVarianceError)
                                else "insufficient_data"
                            ),
                            reason=str(exc),
                        )
                    )

        summaries = []
        for key in keys:
            summary = self._ctx.analyzer.summarize_condition(
                groups[key], metric=metric, condition_name=labels[key]
            )
            if summary is not None:
                summaries.append(summary)

        return AnalysisSnapshot(
            comparisons=comparisons,
            skipped=skipped,
            condition_summaries=summaries,
        )

    # ------------------------------------------------------------------
    # 5. Recommendation
    # ------------------------------------------------------------------
    def recommending(self, state: AdaptiveLoopState) -> Dict[str, str]:
        sm = self._ctx.state_manager
        session_id = state["session_id"]
        session = sm.get_session(session_id)

        experiments = sm.query_experiments(session_id)
        analysis = sm.load_analysis(session_id)
        anomalies = sm.query_anomalies(session_id=session_id)
        # The cycle this recommendation closes.
        cycle = _current_cycle(session)

        recommendation = self._ctx.recommender.recommend_next(
            session.research_question,
            experiments,
            analysis,
            anomalies,
            current_cycle=cycle,
            max_cycles=MAX_ADAPTIVE_CYCLES,
        )

        # The agent's output is stored VERBATIM. The safety cap is a separate
        # fact about the run, recorded in `termination_reason` - it never
        # rewrites `action` over an explanation that argues for more
        # experiments, which produced a final screen contradicting itself.
        at_cap = cycle >= MAX_ADAPTIVE_CYCLES
        if recommendation.action == "conclude":
            termination = "agent_concluded"
        elif at_cap:
            termination = "cycle_limit"
            logger.warning(
                "recommending: session=%s hit the %d-cycle cap while the agent "
                "still wanted more experiments; stopping and recording why",
                session_id, MAX_ADAPTIVE_CYCLES,
            )
        else:
            termination = None

        sm.save_recommendation(session_id, recommendation)
        # Permanent per-cycle record (current_recommendation / latest_analysis
        # are overwritten next cycle; this is what the UI reads for history).
        sm.append_cycle_history(
            session_id,
            CycleHistoryEntry(
                cycle_number=cycle,
                recommendation=recommendation,
                statistical_comparisons=analysis.comparisons,
                skipped_comparisons=analysis.skipped,
                condition_summaries=analysis.condition_summaries,
                termination_reason=termination,
            ),
        )

        if termination is not None:
            sm.update_session_node(session_id, CONCLUDED, cycle_count=cycle)
            sm.update_session_status(
                session_id, "concluded", termination_reason=termination
            )
            logger.info(
                "recommending: session=%s -> stop (cycle %d, reason=%s)",
                session_id, cycle, termination,
            )
            return {"current_node": CONCLUDED}

        sm.save_planned_configs(session_id, recommendation.recommended_experiments)
        sm.update_session_node(session_id, EXECUTING, cycle_count=cycle)
        logger.info(
            "recommending: session=%s -> run_more_experiments, %d config(s) queued "
            "(cycle %d complete, looping back to executing)",
            session_id, len(recommendation.recommended_experiments), cycle,
        )
        return {"current_node": EXECUTING}


def _current_cycle(session) -> int:
    """The 1-based cycle currently in flight.

    ``sessions.cycle_count`` is bumped by ``recommending`` at the END of each
    cycle, so during cycle 1 it is 0, during cycle 2 it is 1, and so on.
    """
    return session.cycle_count + 1
