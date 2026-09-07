"""
tests/unit/test_state_machine_loop.py
=======================================
The autonomous adaptive loop: ``recommending -> executing`` is a real
in-graph conditional edge, so one ``execute_cycle`` runs multiple cycles
inside a single ``graph.invoke()``.

Test cases
----------
1.  test_run_more_experiments_re_enters_executing
2.  test_three_cycles_run_in_one_invocation
3.  test_conclude_reaches_end
4.  test_safety_cap_forces_conclusion
5.  test_pending_configs_flow_from_one_cycle_to_the_next
6.  test_cycle_count_equals_number_of_cycles
7.  test_experiments_from_all_cycles_remain_queryable
8.  test_crash_recovery_resumes_then_keeps_looping
9.  test_anomaly_and_analysis_still_run_each_cycle
"""

from __future__ import annotations

from backend.config import MAX_ADAPTIVE_CYCLES
from backend.state_machine.executor import execute_cycle
from backend.state_machine.nodes import AdaptiveLoopNodes
from tests._fakes import (
    StubRecommender,
    always_run_more,
    build_context,
    make_dataset_profile,
    run_more_n_times_then_conclude,
    run_more_then_conclude,
)


def _seed(state_manager) -> str:
    state_manager.create_dataset(make_dataset_profile())
    return state_manager.create_session("Does dropout help?", "ds-test")


# ---------------------------------------------------------------------------
# 1-3. The loop
# ---------------------------------------------------------------------------

def test_run_more_experiments_re_enters_executing(state_manager) -> None:
    """After `recommending` returns run_more, the executing node runs again
    within the same invocation (the runner is called for cycle 2)."""
    sid = _seed(state_manager)
    ctx = build_context(state_manager, recommender=run_more_then_conclude())

    execute_cycle(sid, ctx)

    # cycle 1 planned 8, cycle 2 ran the 4 recommended -> executing re-entered
    by_cycle: dict[int, int] = {}
    for e in state_manager.query_experiments(sid):
        by_cycle[e.cycle] = by_cycle.get(e.cycle, 0) + 1
    assert by_cycle == {1: 8, 2: 4}


def test_three_cycles_run_in_one_invocation(state_manager) -> None:
    sid = _seed(state_manager)
    ctx = build_context(state_manager, recommender=run_more_n_times_then_conclude(2))

    result = execute_cycle(sid, ctx)

    assert result.cycles_completed == 3
    assert result.status == "concluded"
    assert [h.cycle_number for h in state_manager.get_cycle_history(sid)] == [1, 2, 3]


def test_conclude_reaches_end(state_manager) -> None:
    sid = _seed(state_manager)
    ctx = build_context(state_manager, recommender=StubRecommender())  # concludes cycle 1

    result = execute_cycle(sid, ctx)

    assert result.cycles_completed == 1
    assert result.current_node == "concluded"
    assert result.recommendation.action == "conclude"


# ---------------------------------------------------------------------------
# 4. Safety cap
# ---------------------------------------------------------------------------

def test_safety_cap_forces_conclusion(state_manager) -> None:
    sid = _seed(state_manager)
    ctx = build_context(state_manager, recommender=always_run_more())

    result = execute_cycle(sid, ctx)

    assert result.status == "concluded"
    assert result.cycles_completed == MAX_ADAPTIVE_CYCLES
    assert result.recommendation.action == "conclude"
    assert "safety limit" in result.recommendation.evidence_summary
    # every forced cycle is still in the history
    assert len(state_manager.get_cycle_history(sid)) == MAX_ADAPTIVE_CYCLES


# ---------------------------------------------------------------------------
# 5-7. pending_configs / cycle_count / experiments
# ---------------------------------------------------------------------------

def test_pending_configs_flow_from_one_cycle_to_the_next(state_manager) -> None:
    sid = _seed(state_manager)
    ctx = build_context(state_manager, recommender=run_more_then_conclude())

    execute_cycle(sid, ctx)

    # cycle 2's experiments are exactly the 4 configs the recommender queued
    cycle2 = [e for e in state_manager.query_experiments(sid) if e.cycle == 2]
    assert len(cycle2) == 4
    assert {e.config.hyperparameters["dropout"] for e in cycle2} == {0.25}
    assert {e.config.random_seed for e in cycle2} == {5, 6, 7, 8}
    # the queue is empty once the loop concludes
    assert state_manager.load_planned_configs(sid) == []


def test_cycle_count_equals_number_of_cycles(state_manager) -> None:
    sid = _seed(state_manager)
    ctx = build_context(state_manager, recommender=run_more_n_times_then_conclude(3))

    result = execute_cycle(sid, ctx)

    assert result.cycles_completed == 4
    assert state_manager.get_session(sid).cycle_count == 4


def test_experiments_from_all_cycles_remain_queryable(state_manager) -> None:
    sid = _seed(state_manager)
    ctx = build_context(state_manager, recommender=run_more_n_times_then_conclude(2))

    execute_cycle(sid, ctx)

    all_exp = state_manager.query_experiments(sid)
    assert len(all_exp) == 8 + 3 + 3  # cycle 1 plan + 2 recommended rounds
    assert sorted({e.cycle for e in all_exp}) == [1, 2, 3]
    assert all(e.status == "success" for e in all_exp)


# ---------------------------------------------------------------------------
# 8. Crash recovery + loop
# ---------------------------------------------------------------------------

def test_crash_recovery_resumes_then_keeps_looping(state_manager) -> None:
    """A session parked mid-cycle (e.g. at 'validating' after a crash) resumes
    there and then continues looping autonomously to conclusion."""
    sid = _seed(state_manager)
    ctx = build_context(state_manager, recommender=run_more_then_conclude())

    # Run cycle 1's planning + executing by hand, then "crash": leave the
    # session parked at 'validating' with cycle 1's experiments in the DB.
    nodes = AdaptiveLoopNodes(ctx)
    nodes.planning({"session_id": sid, "current_node": "planning"})
    nodes.executing({"session_id": sid, "current_node": "executing"})
    assert state_manager.get_session(sid).current_node == "validating"
    assert len(state_manager.query_experiments(sid)) == 8

    # Resume: START router -> validating -> analyzing -> recommending
    #   -> (run_more) executing cycle 2 -> ... -> recommending -> conclude
    result = execute_cycle(sid, ctx)

    assert result.status == "concluded"
    assert result.cycles_completed == 2
    by_cycle: dict[int, int] = {}
    for e in state_manager.query_experiments(sid):
        by_cycle[e.cycle] = by_cycle.get(e.cycle, 0) + 1
    assert by_cycle == {1: 8, 2: 4}


# ---------------------------------------------------------------------------
# 9. Anomaly + analysis per cycle
# ---------------------------------------------------------------------------

def test_anomaly_and_analysis_still_run_each_cycle(state_manager) -> None:
    sid = _seed(state_manager)
    ctx = build_context(state_manager, recommender=run_more_n_times_then_conclude(2))

    execute_cycle(sid, ctx)

    # Each cycle's history entry carries that cycle's statistical comparisons.
    history = state_manager.get_cycle_history(sid)
    assert len(history) == 3
    assert all(h.statistical_comparisons for h in history)  # >= 1 comparison each
    # comparisons are cumulative (analysis queries all successful experiments),
    # so later cycles have at least as many as earlier ones.
    counts = [len(h.statistical_comparisons) for h in history]
    assert counts == sorted(counts)
