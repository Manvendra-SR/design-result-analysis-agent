"""
tests/unit/test_state_machine_nodes.py
========================================
Unit tests for the five adaptive-loop nodes (backend/state_machine/nodes.py).

Each test drives one node directly against an in-memory ``StateManager``
with a real ``AnomalyDetector`` / ``StatisticalAnalyzer`` and stubbed
planner / recommender / runner, then asserts on what landed in the
database and which node comes next.

Test cases
----------
1.  test_planning_queues_configs_and_advances
2.  test_planning_propagates_planning_error
3.  test_executing_runs_queue_stores_results_and_clears_queue
4.  test_executing_tags_experiments_with_the_current_cycle
5.  test_executing_resumes_with_only_unrun_configs
6.  test_validating_flags_anomalies_and_updates_status
7.  test_validating_is_idempotent_on_re_entry
8.  test_validating_degrades_gracefully_when_detector_raises
9.  test_analyzing_stores_pairwise_comparisons
10. test_analyzing_uses_val_loss_metric_for_regression
11. test_recommending_conclude_marks_session_concluded
12. test_recommending_run_more_queues_configs_and_bumps_cycle
13. test_recommending_passes_precomputed_stats_to_agent
14. test_recommending_records_a_cycle_history_entry
15. test_recommending_safety_cap_forces_conclude
"""

from __future__ import annotations

import pytest

from backend.agents.planner import PlanningError
from backend.models.experiment import ExperimentConfiguration, ExperimentResult
from backend.state_machine.nodes import AdaptiveLoopNodes
from tests._fakes import (
    StubRecommender,
    StubRunner,
    build_context,
    make_dataset_profile,
    run_more_then_conclude,
)


def _seed(state_manager, task_type: str = "classification") -> str:
    state_manager.create_dataset(make_dataset_profile(task_type=task_type))
    return state_manager.create_session("Does dropout help?", "ds-test")


def _state(session_id: str, node: str) -> dict:
    return {"session_id": session_id, "current_node": node}


# ---------------------------------------------------------------------------
# 1-2. planning
# ---------------------------------------------------------------------------

def test_planning_queues_configs_and_advances(state_manager) -> None:
    sid = _seed(state_manager)
    nodes = AdaptiveLoopNodes(build_context(state_manager))

    result = nodes.planning(_state(sid, "planning"))

    assert result == {"current_node": "executing"}
    assert len(state_manager.load_planned_configs(sid)) == 8
    assert state_manager.get_session(sid).current_node == "executing"


def test_planning_propagates_planning_error(state_manager) -> None:
    sid = _seed(state_manager)

    class BoomPlanner:
        def plan_experiments(self, q, p):
            raise PlanningError("not answerable with this dataset")

    nodes = AdaptiveLoopNodes(build_context(state_manager, planner=BoomPlanner()))

    with pytest.raises(PlanningError):
        nodes.planning(_state(sid, "planning"))
    # current_node not advanced -> a retry re-runs planning
    assert state_manager.get_session(sid).current_node == "planning"


# ---------------------------------------------------------------------------
# 3-4. executing
# ---------------------------------------------------------------------------

def test_executing_runs_queue_stores_results_and_clears_queue(state_manager) -> None:
    sid = _seed(state_manager)
    nodes = AdaptiveLoopNodes(build_context(state_manager))
    nodes.planning(_state(sid, "planning"))

    result = nodes.executing(_state(sid, "executing"))

    assert result == {"current_node": "validating"}
    experiments = state_manager.query_experiments(sid)
    assert len(experiments) == 8
    assert all(e.status == "success" for e in experiments)
    assert all(e.cycle == 1 for e in experiments)  # tagged with the cycle
    assert state_manager.load_planned_configs(sid) == []


def test_executing_tags_experiments_with_the_current_cycle(state_manager) -> None:
    sid = _seed(state_manager)
    nodes = AdaptiveLoopNodes(build_context(state_manager))
    nodes.planning(_state(sid, "planning"))
    nodes.executing(_state(sid, "executing"))  # cycle 1: 8 experiments

    # advance cycle_count + queue new configs, as `recommending` would for cycle 2
    state_manager.update_session_node(sid, "executing", cycle_count=1)
    state_manager.save_planned_configs(sid, [
        ExperimentConfiguration(dataset_id="ds-test", model_type="mlp",
                                hyperparameters={"dropout": 0.25}, random_seed=s)
        for s in (5, 6, 7)
    ])
    nodes.executing(_state(sid, "executing"))  # cycle 2: 3 experiments

    by_cycle: dict[int, int] = {}
    for e in state_manager.query_experiments(sid):
        by_cycle[e.cycle] = by_cycle.get(e.cycle, 0) + 1
    assert by_cycle == {1: 8, 2: 3}


def test_executing_resumes_with_only_unrun_configs(state_manager) -> None:
    sid = _seed(state_manager)
    # Simulate a crash: 2 experiments already stored, 2 configs still queued.
    profile = state_manager.get_dataset("ds-test")
    runner = StubRunner()
    ctx = build_context(state_manager, runner=runner)
    nodes = AdaptiveLoopNodes(ctx)

    all_configs = [
        ExperimentConfiguration(dataset_id="ds-test", model_type="mlp",
                                hyperparameters={"dropout": 0.0}, random_seed=s)
        for s in (1, 2, 3, 4)
    ]
    for cfg in all_configs[:2]:
        state_manager.store_experiment(runner.run_experiment(cfg, profile, session_id=sid))
    state_manager.save_planned_configs(sid, all_configs[2:])

    nodes.executing(_state(sid, "executing"))

    assert len(state_manager.query_experiments(sid)) == 4  # 2 pre-existing + 2 resumed
    assert state_manager.load_planned_configs(sid) == []


# ---------------------------------------------------------------------------
# 5-7. validating
# ---------------------------------------------------------------------------

class _NearChanceRunner:
    """Every experiment lands at ~52% accuracy on a binary task -> validation collapse."""

    def run_experiment(self, config, dataset_profile, session_id=""):
        return ExperimentResult(
            session_id=session_id, config=config, task_type="classification",
            metrics={"train_loss": 0.69, "val_loss": 0.69, "accuracy": 0.52,
                     "n_classes": 2.0, "n_val_samples": 100.0,
                     "training_time_seconds": 0.01, "initial_train_loss": 0.69},
            status="success",
        )

    def run_batch(self, configs, dataset_profile, session_id=""):
        return [self.run_experiment(c, dataset_profile, session_id=session_id) for c in configs]


def test_validating_flags_anomalies_and_updates_status(state_manager) -> None:
    sid = _seed(state_manager)
    nodes = AdaptiveLoopNodes(build_context(state_manager, runner=_NearChanceRunner()))
    nodes.planning(_state(sid, "planning"))
    nodes.executing(_state(sid, "executing"))

    result = nodes.validating(_state(sid, "validating"))

    assert result == {"current_node": "analyzing"}
    anomalies = state_manager.query_anomalies(session_id=sid)
    assert len(anomalies) == 8
    assert all(a.rule == "validation_collapse" for a in anomalies)
    assert all(e.status == "anomalous" for e in state_manager.query_experiments(sid))


def test_validating_is_idempotent_on_re_entry(state_manager) -> None:
    sid = _seed(state_manager)
    nodes = AdaptiveLoopNodes(build_context(state_manager, runner=_NearChanceRunner()))
    nodes.planning(_state(sid, "planning"))
    nodes.executing(_state(sid, "executing"))

    nodes.validating(_state(sid, "validating"))
    first = len(state_manager.query_anomalies(session_id=sid))
    nodes.validating(_state(sid, "validating"))
    second = len(state_manager.query_anomalies(session_id=sid))

    assert first == second == 8  # no duplicate anomaly rows


def test_validating_degrades_gracefully_when_detector_raises(state_manager) -> None:
    sid = _seed(state_manager)
    ctx = build_context(state_manager)
    nodes = AdaptiveLoopNodes(ctx)
    nodes.planning(_state(sid, "planning"))
    nodes.executing(_state(sid, "executing"))

    class BoomDetector:
        def detect_anomalies(self, experiments):
            raise RuntimeError("detector exploded")

    ctx.detector = BoomDetector()
    result = nodes.validating(_state(sid, "validating"))

    assert result == {"current_node": "analyzing"}  # advanced despite the failure
    assert state_manager.query_anomalies(session_id=sid) == []


# ---------------------------------------------------------------------------
# 8-9. analyzing
# ---------------------------------------------------------------------------

def test_analyzing_stores_pairwise_comparisons(state_manager) -> None:
    sid = _seed(state_manager)
    nodes = AdaptiveLoopNodes(build_context(state_manager))
    nodes.planning(_state(sid, "planning"))
    nodes.executing(_state(sid, "executing"))
    nodes.validating(_state(sid, "validating"))

    result = nodes.analyzing(_state(sid, "analyzing"))

    assert result == {"current_node": "recommending"}
    comparisons = state_manager.load_analysis(sid)
    assert len(comparisons) == 1  # 2 dropout conditions -> 1 pairwise comparison
    assert comparisons[0].metric == "accuracy"


def test_analyzing_uses_val_loss_metric_for_regression(state_manager) -> None:
    sid = _seed(state_manager, task_type="regression")
    nodes = AdaptiveLoopNodes(build_context(state_manager, task_type="regression"))
    nodes.planning(_state(sid, "planning"))
    nodes.executing(_state(sid, "executing"))
    nodes.validating(_state(sid, "validating"))

    nodes.analyzing(_state(sid, "analyzing"))

    comparisons = state_manager.load_analysis(sid)
    assert comparisons and comparisons[0].metric == "val_loss"


# ---------------------------------------------------------------------------
# 10-12. recommending
# ---------------------------------------------------------------------------

def _advance_to_recommending(ctx, sid) -> AdaptiveLoopNodes:
    nodes = AdaptiveLoopNodes(ctx)
    nodes.planning(_state(sid, "planning"))
    nodes.executing(_state(sid, "executing"))
    nodes.validating(_state(sid, "validating"))
    nodes.analyzing(_state(sid, "analyzing"))
    return nodes


def test_recommending_conclude_marks_session_concluded(state_manager) -> None:
    sid = _seed(state_manager)
    ctx = build_context(state_manager, recommender=StubRecommender())  # default = conclude
    nodes = _advance_to_recommending(ctx, sid)

    result = nodes.recommending(_state(sid, "recommending"))

    assert result == {"current_node": "concluded"}
    session = state_manager.get_session(sid)
    assert session.status == "concluded"
    assert session.current_node == "concluded"
    assert session.cycle_count == 1
    assert state_manager.get_recommendation(sid).action == "conclude"


def test_recommending_run_more_queues_configs_and_bumps_cycle(state_manager) -> None:
    sid = _seed(state_manager)
    ctx = build_context(state_manager, recommender=run_more_then_conclude())
    nodes = _advance_to_recommending(ctx, sid)

    result = nodes.recommending(_state(sid, "recommending"))

    assert result == {"current_node": "executing"}
    session = state_manager.get_session(sid)
    assert session.status == "active"
    assert session.cycle_count == 1
    assert len(state_manager.load_planned_configs(sid)) == 4


def test_recommending_passes_precomputed_stats_to_agent(state_manager) -> None:
    sid = _seed(state_manager)
    recommender = StubRecommender()
    ctx = build_context(state_manager, recommender=recommender)
    nodes = _advance_to_recommending(ctx, sid)

    nodes.recommending(_state(sid, "recommending"))

    call = recommender.received[-1]
    # The node hands the agent the StatisticalComparison objects the analysis
    # node already computed - the agent never sees raw metrics to crunch.
    from backend.models.statistics import StatisticalComparison

    assert call["statistical_results"]
    assert all(isinstance(s, StatisticalComparison) for s in call["statistical_results"])
    assert all(isinstance(e, ExperimentResult) for e in call["experiments"])


def test_recommending_records_a_cycle_history_entry(state_manager) -> None:
    sid = _seed(state_manager)
    ctx = build_context(state_manager, recommender=run_more_then_conclude())
    nodes = _advance_to_recommending(ctx, sid)

    nodes.recommending(_state(sid, "recommending"))

    history = state_manager.get_cycle_history(sid)
    assert len(history) == 1
    assert history[0].cycle_number == 1
    assert history[0].recommendation.action == "run_more_experiments"
    # the analysis this cycle was based on is retained alongside it
    assert history[0].statistical_comparisons


def test_recommending_safety_cap_forces_conclude(state_manager) -> None:
    from backend.config import MAX_ADAPTIVE_CYCLES
    from tests._fakes import always_run_more

    sid = _seed(state_manager)
    ctx = build_context(state_manager, recommender=always_run_more())
    nodes = _advance_to_recommending(ctx, sid)

    # Pretend we're already at the last allowed cycle.
    state_manager.update_session_node(sid, "recommending", cycle_count=MAX_ADAPTIVE_CYCLES - 1)

    result = nodes.recommending(_state(sid, "recommending"))

    assert result == {"current_node": "concluded"}
    session = state_manager.get_session(sid)
    assert session.status == "concluded"
    assert session.cycle_count == MAX_ADAPTIVE_CYCLES
    rec = state_manager.get_recommendation(sid)
    assert rec.action == "conclude"
    assert "safety limit" in rec.evidence_summary
