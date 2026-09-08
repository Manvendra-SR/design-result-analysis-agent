"""
tests/unit/test_state_machine_executor.py
===========================================
Integration tests for backend/state_machine/executor.py + graph.py: the
autonomous adaptive loop driven through ``execute_cycle`` against an
in-memory database, with stubbed agents / runner.

One ``execute_cycle`` call now runs the *whole* investigation - the graph
loops ``recommending -> executing`` internally until the Recommender
concludes (or the safety cap).

Test cases
----------
1.  test_execute_cycle_runs_all_cycles_to_conclusion
2.  test_execute_cycle_on_missing_session_raises_keyerror
3.  test_execute_cycle_on_concluded_session_raises_cycleerror
4.  test_resume_from_analyzing_continues_to_conclusion
5.  test_recommendation_error_leaves_current_node_unadvanced
6.  test_run_phase_is_idle_after_a_clean_run
7.  test_run_phase_is_failed_after_a_node_raises
8.  test_resume_after_failure_clears_run_phase
"""

from __future__ import annotations

import pytest

from backend.agents.recommender import RecommendationError
from backend.state_machine.executor import CycleError, execute_cycle
from tests._fakes import (
    StubRecommender,
    build_context,
    make_dataset_profile,
    run_more_then_conclude,
)


def _seed(state_manager) -> str:
    state_manager.create_dataset(make_dataset_profile())
    return state_manager.create_session("Does dropout help?", "ds-test")


def test_execute_cycle_runs_all_cycles_to_conclusion(state_manager) -> None:
    sid = _seed(state_manager)
    # run_more_then_conclude: cycle 1 -> run_more (4 configs), cycle 2 -> conclude
    ctx = build_context(state_manager, recommender=run_more_then_conclude())

    result = execute_cycle(sid, ctx)

    assert result.current_node == "concluded"
    assert result.status == "concluded"
    assert result.cycles_completed == 2
    assert result.experiments_completed == 12  # cycle 1: 8, cycle 2: 4
    assert result.recommendation.action == "conclude"

    # per-cycle history was recorded for both cycles
    history = state_manager.get_cycle_history(sid)
    assert [h.cycle_number for h in history] == [1, 2]
    assert history[0].recommendation.action == "run_more_experiments"
    assert history[1].recommendation.action == "conclude"

    # every experiment is tagged with the cycle that produced it
    by_cycle: dict[int, int] = {}
    for e in state_manager.query_experiments(sid):
        by_cycle[e.cycle] = by_cycle.get(e.cycle, 0) + 1
    assert by_cycle == {1: 8, 2: 4}

    assert state_manager.get_session(sid).plan_explanation  # planner rationale kept


def test_execute_cycle_on_missing_session_raises_keyerror(state_manager) -> None:
    ctx = build_context(state_manager)
    with pytest.raises(KeyError):
        execute_cycle("00000000-0000-0000-0000-000000000000", ctx)


def test_execute_cycle_on_concluded_session_raises_cycleerror(state_manager) -> None:
    sid = _seed(state_manager)
    ctx = build_context(state_manager, recommender=StubRecommender())  # concludes on cycle 1

    execute_cycle(sid, ctx)
    with pytest.raises(CycleError):
        execute_cycle(sid, ctx)


def test_resume_from_analyzing_continues_to_conclusion(state_manager) -> None:
    """A session that crashed mid-cycle resumes at its persisted node and
    then keeps looping to conclusion - crash recovery and the adaptive loop
    coexist."""
    sid = _seed(state_manager)

    # Warm the DB: run the real loop, then rewind the session to mid-cycle-1.
    warm_ctx = build_context(state_manager, recommender=run_more_then_conclude())
    execute_cycle(sid, warm_ctx)
    assert state_manager.get_session(sid).status == "concluded"

    # Simulate a crash during cycle 1's analysis: reset status + node, keep
    # cycle 1's experiments, drop the history so the loop re-derives it.
    state_manager.update_session_status(sid, "active")
    state_manager.update_session_node(sid, "analyzing", cycle_count=0)

    class Tracker:
        planned = executed = 0

    class TrackingRunner:
        def run_experiment(self, *a, **k):
            Tracker.executed += 1
            raise AssertionError("execution must not run when resuming from 'analyzing'")

        def run_batch(self, *a, **k):
            raise AssertionError("execution must not run when resuming from 'analyzing'")

    resume_ctx = build_context(
        state_manager, runner=TrackingRunner(), recommender=StubRecommender()
    )
    result = execute_cycle(sid, resume_ctx)

    # analysis + recommendation ran (StubRecommender concludes) -> no execution.
    assert Tracker.executed == 0
    assert result.current_node == "concluded"
    assert result.status == "concluded"


def test_recommendation_error_leaves_current_node_unadvanced(state_manager) -> None:
    sid = _seed(state_manager)

    class BoomRecommender:
        def recommend_next(self, *a, **k):
            raise RecommendationError("LLM returned an invalid action")

    ctx = build_context(state_manager, recommender=BoomRecommender())

    with pytest.raises(RecommendationError):
        execute_cycle(sid, ctx)

    # planning/executing/validating/analyzing ran and advanced; the failure
    # is at 'recommending', so that is where the session is parked for a retry.
    assert state_manager.get_session(sid).current_node == "recommending"


def test_run_phase_is_idle_after_a_clean_run(state_manager) -> None:
    sid = _seed(state_manager)
    assert state_manager.get_session(sid).run_phase == "idle"  # fresh session

    result = execute_cycle(sid, build_context(state_manager, recommender=StubRecommender()))

    assert result.run_phase == "idle"
    session = state_manager.get_session(sid)
    assert session.run_phase == "idle"
    assert session.run_error is None
    assert session.status == "concluded"


def test_run_phase_is_failed_after_a_node_raises(state_manager) -> None:
    sid = _seed(state_manager)

    class BoomRecommender:
        def recommend_next(self, *a, **k):
            raise RecommendationError("LLM returned an invalid action")

    ctx = build_context(state_manager, recommender=BoomRecommender())
    with pytest.raises(RecommendationError):
        execute_cycle(sid, ctx)

    session = state_manager.get_session(sid)
    assert session.run_phase == "failed"
    assert session.run_error and "invalid action" in session.run_error
    assert session.status == "active"  # NOT concluded - resumable
    assert session.current_node == "recommending"


def test_resume_after_failure_clears_run_phase(state_manager) -> None:
    sid = _seed(state_manager)

    class BoomThenOk:
        def __init__(self):
            self.calls = 0

        def recommend_next(self, *a, **k):
            self.calls += 1
            if self.calls == 1:
                raise RecommendationError("transient LLM hiccup")
            return StubRecommender().recommend_next(*a, **k)

    rec = BoomThenOk()
    ctx = build_context(state_manager, recommender=rec)

    with pytest.raises(RecommendationError):
        execute_cycle(sid, ctx)
    assert state_manager.get_session(sid).run_phase == "failed"

    # Resume: the second call picks up at 'recommending' and concludes.
    result = execute_cycle(sid, ctx)
    assert result.run_phase == "idle"
    session = state_manager.get_session(sid)
    assert session.run_phase == "idle"
    assert session.run_error is None
    assert session.status == "concluded"
