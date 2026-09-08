"""
tests/unit/test_api_sessions.py
=================================
Tests for the session + run-cycle endpoints (backend/api/routes/sessions.py).

The ``get_context`` dependency is overridden with a fully-stubbed
``StateMachineContext`` over in-memory SQLite, so ``POST /run-cycle`` really
drives the Phase 5 state machine end-to-end (the Phase 5 -> Phase 6
integration path) - only the LLM agents and the training runner are faked.
One ``POST /run-cycle`` runs the whole autonomous investigation.

Test cases
----------
1.  test_create_session_returns_201_and_id
2.  test_create_session_rejects_empty_question
3.  test_create_session_rejects_too_long_question
4.  test_create_session_unknown_dataset_returns_404
5.  test_list_sessions
6.  test_get_session_detail_after_investigation
7.  test_get_session_unknown_returns_404
8.  test_run_cycle_runs_whole_investigation_to_conclusion
9.  test_run_cycle_on_concluded_session_returns_409
10. test_get_recommendation_404_before_first_cycle
11. test_get_recommendation_is_the_final_one
12. test_run_cycle_propagates_planning_error_as_400
"""

from __future__ import annotations

import pytest

from backend.agents.planner import PlanningError
from tests._fakes import make_dataset_profile, make_test_client, run_more_then_conclude


@pytest.fixture
def client_ctx(state_manager):
    state_manager.create_dataset(make_dataset_profile())
    client, ctx = make_test_client(state_manager, recommender=run_more_then_conclude())
    return client, ctx, state_manager


def _new_session(client, dataset_id: str = "ds-test") -> str:
    resp = client.post(
        "/api/sessions",
        json={"research_question": "Does dropout help?", "dataset_id": dataset_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["session_id"]


# ---------------------------------------------------------------------------
# create / list / get
# ---------------------------------------------------------------------------

def test_create_session_returns_201_and_id(client_ctx) -> None:
    client, _, _ = client_ctx
    resp = client.post(
        "/api/sessions",
        json={"research_question": "Does dropout help?", "dataset_id": "ds-test"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["session_id"]
    assert "created_at" in body


def test_create_session_rejects_empty_question(client_ctx) -> None:
    client, _, _ = client_ctx
    resp = client.post(
        "/api/sessions", json={"research_question": "", "dataset_id": "ds-test"}
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "validation_error"


def test_create_session_rejects_too_long_question(client_ctx) -> None:
    client, _, _ = client_ctx
    resp = client.post(
        "/api/sessions",
        json={"research_question": "x" * 501, "dataset_id": "ds-test"},
    )
    assert resp.status_code == 400


def test_create_session_unknown_dataset_returns_404(client_ctx) -> None:
    client, _, _ = client_ctx
    resp = client.post(
        "/api/sessions",
        json={"research_question": "Does dropout help?", "dataset_id": "nope"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"] == "not_found"


def test_list_sessions(client_ctx) -> None:
    client, _, _ = client_ctx
    _new_session(client)
    _new_session(client)
    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_get_session_detail_after_investigation(client_ctx) -> None:
    client, _, _ = client_ctx
    sid = _new_session(client)
    client.post(f"/api/sessions/{sid}/run-cycle")
    resp = client.get(f"/api/sessions/{sid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["experiment_count"] == 12   # 2 cycles: 8 + 4
    assert body["current_node"] == "concluded"
    assert body["status"] == "concluded"
    assert body["cycle_count"] == 2
    assert body["plan_explanation"]  # planner rationale surfaced


def test_get_session_unknown_returns_404(client_ctx) -> None:
    client, _, _ = client_ctx
    resp = client.get("/api/sessions/does-not-exist")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# run-cycle
# ---------------------------------------------------------------------------

def test_run_cycle_runs_whole_investigation_to_conclusion(client_ctx) -> None:
    client, _, _ = client_ctx
    sid = _new_session(client)

    # a brand-new session is not running
    detail = client.get(f"/api/sessions/{sid}").json()
    assert detail["run_phase"] == "idle"
    assert detail["run_error"] is None

    resp = client.post(f"/api/sessions/{sid}/run-cycle")
    assert resp.status_code == 200
    body = resp.json()
    assert body["current_node"] == "concluded"
    assert body["status"] == "concluded"
    assert body["run_phase"] == "idle"
    assert body["cycles_completed"] == 2         # run_more_then_conclude
    assert body["experiments_completed"] == 12   # 8 + 4
    assert body["recommendation"]["action"] == "conclude"  # the FINAL one

    # after a clean finish the session is concluded and not running
    detail = client.get(f"/api/sessions/{sid}").json()
    assert detail["run_phase"] == "idle"


def test_run_cycle_on_concluded_session_returns_409(client_ctx) -> None:
    client, _, _ = client_ctx
    sid = _new_session(client)
    client.post(f"/api/sessions/{sid}/run-cycle")  # runs to conclusion
    resp = client.post(f"/api/sessions/{sid}/run-cycle")
    assert resp.status_code == 409
    assert resp.json()["error"] == "conflict"


def test_run_cycle_propagates_planning_error_as_400(state_manager) -> None:
    state_manager.create_dataset(make_dataset_profile())

    class BoomPlanner:
        def plan_experiments(self, q, p):
            raise PlanningError("not answerable with this dataset")

    client, _ = make_test_client(state_manager, planner=BoomPlanner())
    sid = _new_session(client)
    resp = client.post(f"/api/sessions/{sid}/run-cycle")
    assert resp.status_code == 400
    assert resp.json()["error"] == "planning_failed"

    # the failure is persisted: the UI can show a failed state after a refresh,
    # and the session stays resumable (status still 'active').
    detail = client.get(f"/api/sessions/{sid}").json()
    assert detail["run_phase"] == "failed"
    assert detail["run_error"] and "not answerable" in detail["run_error"]
    assert detail["status"] == "active"
    assert detail["current_node"] == "planning"


# ---------------------------------------------------------------------------
# recommendation
# ---------------------------------------------------------------------------

def test_get_recommendation_404_before_first_cycle(client_ctx) -> None:
    client, _, _ = client_ctx
    sid = _new_session(client)
    resp = client.get(f"/api/sessions/{sid}/recommendation")
    assert resp.status_code == 404


def test_get_recommendation_is_the_final_one(client_ctx) -> None:
    client, _, _ = client_ctx
    sid = _new_session(client)
    client.post(f"/api/sessions/{sid}/run-cycle")
    resp = client.get(f"/api/sessions/{sid}/recommendation")
    assert resp.status_code == 200
    # the investigation ran to conclusion, so this is the concluding rec
    assert resp.json()["action"] == "conclude"
    assert resp.json()["recommended_experiments"] == []


# ---------------------------------------------------------------------------
# DELETE /api/sessions/{id}
# ---------------------------------------------------------------------------

def test_delete_session_removes_it(client_ctx) -> None:
    client, _, _ = client_ctx
    sid = _new_session(client)
    resp = client.delete(f"/api/sessions/{sid}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["deleted"] == "session" and body["id"] == sid
    assert client.get(f"/api/sessions/{sid}").status_code == 404
    assert client.get("/api/sessions").json() == []


def test_delete_session_unknown_returns_404(client_ctx) -> None:
    client, _, _ = client_ctx
    resp = client.delete("/api/sessions/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


def test_delete_session_removes_its_experiments(client_ctx) -> None:
    client, _, sm = client_ctx
    sid = _new_session(client)
    client.post(f"/api/sessions/{sid}/run-cycle")  # generates experiments
    assert len(sm.query_experiments(sid)) > 0

    resp = client.delete(f"/api/sessions/{sid}")
    assert resp.status_code == 200
    assert resp.json()["experiments_deleted"] > 0
    assert client.get(f"/api/sessions/{sid}").status_code == 404
    # the dataset the session used is untouched
    assert client.get("/api/datasets/ds-test").status_code == 200
