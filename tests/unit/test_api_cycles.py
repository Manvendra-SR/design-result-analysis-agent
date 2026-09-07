"""
tests/unit/test_api_cycles.py
===============================
Tests for GET /api/sessions/{id}/cycles - the per-cycle investigation
history the UI renders so the autonomous loop is not a black box.

Test cases
----------
1.  test_cycles_returns_one_entry_per_cycle
2.  test_each_cycle_has_experiments_stats_and_recommendation
3.  test_cycle_one_carries_the_plan_explanation
4.  test_final_cycle_recommendation_is_conclude
5.  test_continued_flag_tracks_run_more_vs_conclude
6.  test_cycles_before_any_run_is_empty
7.  test_cycles_unknown_session_404
8.  test_anomalies_are_grouped_into_their_cycle
"""

from __future__ import annotations

import pytest

from tests._fakes import (
    make_dataset_profile,
    make_test_client,
    run_more_n_times_then_conclude,
)


@pytest.fixture
def investigated(state_manager):
    """A session whose investigation has run to conclusion over 3 cycles."""
    state_manager.create_dataset(make_dataset_profile())
    client, ctx = make_test_client(
        state_manager, recommender=run_more_n_times_then_conclude(2)
    )
    sid = client.post(
        "/api/sessions",
        json={"research_question": "Does dropout help?", "dataset_id": "ds-test"},
    ).json()["session_id"]
    client.post(f"/api/sessions/{sid}/run-cycle")
    return client, sid, state_manager


def test_cycles_returns_one_entry_per_cycle(investigated) -> None:
    client, sid, _ = investigated
    resp = client.get(f"/api/sessions/{sid}/cycles")
    assert resp.status_code == 200
    body = resp.json()
    assert [c["cycle_number"] for c in body] == [1, 2, 3]


def test_each_cycle_has_experiments_stats_and_recommendation(investigated) -> None:
    client, sid, _ = investigated
    body = client.get(f"/api/sessions/{sid}/cycles").json()
    for cyc in body:
        assert cyc["experiments"], f"cycle {cyc['cycle_number']} has no experiments"
        assert cyc["recommendation"] is not None
        assert cyc["statistical_comparisons"]  # >= 1 comparison
        # experiments in this cycle really carry that cycle number
        assert {e["cycle"] for e in cyc["experiments"]} == {cyc["cycle_number"]}


def test_cycle_one_carries_the_plan_explanation(investigated) -> None:
    client, sid, _ = investigated
    body = client.get(f"/api/sessions/{sid}/cycles").json()
    assert body[0]["plan_explanation"]  # planner's initial rationale
    assert body[1]["plan_explanation"] is None
    assert body[2]["plan_explanation"] is None


def test_final_cycle_recommendation_is_conclude(investigated) -> None:
    client, sid, _ = investigated
    body = client.get(f"/api/sessions/{sid}/cycles").json()
    assert body[-1]["recommendation"]["action"] == "conclude"
    assert body[-1]["continued"] is False


def test_continued_flag_tracks_run_more_vs_conclude(investigated) -> None:
    client, sid, _ = investigated
    body = client.get(f"/api/sessions/{sid}/cycles").json()
    assert [c["continued"] for c in body] == [True, True, False]


def test_cycles_before_any_run_is_empty(state_manager) -> None:
    state_manager.create_dataset(make_dataset_profile())
    client, _ = make_test_client(state_manager)
    sid = client.post(
        "/api/sessions",
        json={"research_question": "q?", "dataset_id": "ds-test"},
    ).json()["session_id"]
    resp = client.get(f"/api/sessions/{sid}/cycles")
    assert resp.status_code == 200
    assert resp.json() == []


def test_cycles_unknown_session_404(state_manager) -> None:
    client, _ = make_test_client(state_manager)
    resp = client.get("/api/sessions/does-not-exist/cycles")
    assert resp.status_code == 404


def test_anomalies_are_grouped_into_their_cycle(state_manager) -> None:
    # Seed a session with a hand-built anomalous experiment in cycle 2.
    state_manager.create_dataset(make_dataset_profile())
    client, ctx = make_test_client(
        state_manager, recommender=run_more_n_times_then_conclude(1)
    )
    sid = client.post(
        "/api/sessions",
        json={"research_question": "q?", "dataset_id": "ds-test"},
    ).json()["session_id"]
    client.post(f"/api/sessions/{sid}/run-cycle")

    # Attach an anomaly to one of cycle 2's experiments directly.
    from backend.models.anomaly import AnomalyReport

    exps = state_manager.query_experiments(sid)
    cycle2_exp = next(e for e in exps if e.cycle == 2)
    state_manager.store_anomaly(
        AnomalyReport(
            experiment_id=cycle2_exp.experiment_id,
            rule="outlier_detection",
            explanation="hand-injected for the test",
            severity="warning",
        )
    )

    body = client.get(f"/api/sessions/{sid}/cycles").json()
    cyc1, cyc2 = body[0], body[1]
    assert cyc1["anomalies"] == []
    assert len(cyc2["anomalies"]) == 1
    assert cyc2["anomalies"][0]["experiment_id"] == cycle2_exp.experiment_id
