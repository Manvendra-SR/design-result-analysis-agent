"""
tests/unit/test_api_errors.py
===============================
Cross-cutting API behaviour: the ErrorResponse shape, status codes, CORS
headers, and the 500 fallback.

Test cases
----------
1.  test_error_response_shape_is_consistent
2.  test_cors_headers_present_on_response
3.  test_cors_preflight_allowed
4.  test_unhandled_exception_becomes_500
5.  test_llm_error_becomes_502
6.  test_health_endpoint
"""

from __future__ import annotations

import pytest

from backend.agents.llm_client import LLMError
from tests._fakes import make_dataset_profile, make_test_client, run_more_then_conclude


@pytest.fixture
def client(state_manager):
    state_manager.create_dataset(make_dataset_profile())
    api_client, _ = make_test_client(state_manager, recommender=run_more_then_conclude())
    return api_client


def test_error_response_shape_is_consistent(client) -> None:
    body = client.get("/api/sessions/missing").json()
    assert set(body) == {"error", "message", "details"}
    assert isinstance(body["error"], str)
    assert isinstance(body["message"], str)


def test_cors_headers_present_on_response(client) -> None:
    resp = client.get("/api/sessions", headers={"Origin": "http://localhost:5173"})
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] in ("*", "http://localhost:5173")


def test_cors_preflight_allowed(client) -> None:
    resp = client.options(
        "/api/sessions",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.status_code in (200, 204)
    assert "access-control-allow-origin" in resp.headers


def test_unhandled_exception_becomes_500(state_manager) -> None:
    class ExplodingPlanner:
        def plan_experiments(self, q, p):
            raise RuntimeError("something unexpected")

    state_manager.create_dataset(make_dataset_profile())
    api_client, _ = make_test_client(state_manager, planner=ExplodingPlanner())
    sid = api_client.post(
        "/api/sessions",
        json={"research_question": "q?", "dataset_id": "ds-test"},
    ).json()["session_id"]

    resp = api_client.post(f"/api/sessions/{sid}/run-cycle")
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "internal_error"
    assert "unexpected" in body["message"].lower()


def test_llm_error_becomes_502(state_manager) -> None:
    class UnreachablePlanner:
        def plan_experiments(self, q, p):
            raise LLMError("Could not reach Ollama at http://localhost:11434")

    state_manager.create_dataset(make_dataset_profile())
    api_client, _ = make_test_client(state_manager, planner=UnreachablePlanner())
    sid = api_client.post(
        "/api/sessions",
        json={"research_question": "q?", "dataset_id": "ds-test"},
    ).json()["session_id"]

    resp = api_client.post(f"/api/sessions/{sid}/run-cycle")
    assert resp.status_code == 502
    assert resp.json()["error"] == "llm_unavailable"


def test_health_endpoint(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
