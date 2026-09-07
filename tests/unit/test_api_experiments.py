"""
tests/unit/test_api_experiments.py
====================================
Tests for the experiment query endpoints (backend/api/routes/experiments.py).

Test cases
----------
1.  test_list_session_experiments
2.  test_list_session_experiments_status_filter
3.  test_list_session_experiments_rejects_unknown_status
4.  test_list_session_experiments_unknown_session_404
5.  test_get_experiment_detail
6.  test_get_experiment_unknown_404
"""

from __future__ import annotations

import pytest

from tests._fakes import make_dataset_profile, make_test_client, run_more_then_conclude


@pytest.fixture
def seeded(state_manager):
    state_manager.create_dataset(make_dataset_profile())
    client, ctx = make_test_client(state_manager, recommender=run_more_then_conclude())
    resp = client.post(
        "/api/sessions",
        json={"research_question": "Does dropout help?", "dataset_id": "ds-test"},
    )
    sid = resp.json()["session_id"]
    # One call runs the whole investigation: cycle 1 (8 exp) + cycle 2 (4 exp).
    client.post(f"/api/sessions/{sid}/run-cycle")
    return client, sid


def test_list_session_experiments(seeded) -> None:
    client, sid = seeded
    resp = client.get(f"/api/sessions/{sid}/experiments")
    assert resp.status_code == 200
    assert len(resp.json()) == 12
    assert {e["status"] for e in resp.json()} == {"success"}
    assert {e["cycle"] for e in resp.json()} == {1, 2}  # tagged per cycle


def test_list_session_experiments_status_filter(seeded) -> None:
    client, sid = seeded
    assert len(client.get(f"/api/sessions/{sid}/experiments?status=success").json()) == 12
    assert client.get(f"/api/sessions/{sid}/experiments?status=failed").json() == []


def test_list_session_experiments_rejects_unknown_status(seeded) -> None:
    client, sid = seeded
    resp = client.get(f"/api/sessions/{sid}/experiments?status=running")
    assert resp.status_code == 400
    assert resp.json()["error"] == "bad_request"


def test_list_session_experiments_unknown_session_404(seeded) -> None:
    client, _ = seeded
    resp = client.get("/api/sessions/nope/experiments")
    assert resp.status_code == 404


def test_get_experiment_detail(seeded) -> None:
    client, sid = seeded
    one = client.get(f"/api/sessions/{sid}/experiments").json()[0]
    resp = client.get(f"/api/experiments/{one['experiment_id']}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["experiment_id"] == one["experiment_id"]
    assert body["config"]["model_type"] == "mlp"
    assert body["metrics"]["accuracy"] > 0


def test_get_experiment_unknown_404(seeded) -> None:
    client, _ = seeded
    resp = client.get("/api/experiments/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
    assert resp.json()["error"] == "not_found"
