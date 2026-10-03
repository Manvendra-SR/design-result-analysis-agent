"""Tests for the FastAPI layer: datasets, sessions, the background run, errors."""

from __future__ import annotations

import pytest

from backend.state_machine.graph import run_investigation
from backend.tools.dataset.ingestion import delete_dataset_files
from tests._fakes import FakePlanner, FakeRecommender, fake_run, make_profile, make_test_client


@pytest.fixture
def client(repo):
    return make_test_client(repo)


def _csv(rows: int = 40) -> str:
    return "\n".join(["feat_a,feat_b,label", *(f"{i * 0.5},{i % 7},{i % 2}" for i in range(rows))])


def _ingest(client, **overrides) -> dict:
    body = {"filename": "c.csv", "csv_content": _csv(), "target_column": "label", **overrides}
    return client.post("/api/datasets", json=body)


# --------------------------------------------------------------------- datasets
def test_ingest_and_read_a_dataset(client) -> None:
    resp = _ingest(client)
    assert resp.status_code == 201, resp.text
    profile = resp.json()
    assert (profile["task_type"], profile["n_classes"], profile["n_rows"]) == ("classification", 2, 40)
    assert client.get(f"/api/datasets/{profile['dataset_id']}").json()["dataset_id"] == profile["dataset_id"]
    assert len(client.get("/api/datasets").json()) == 1
    delete_dataset_files(profile["dataset_id"])


@pytest.mark.parametrize(
    "overrides, code",
    [({"target_column": "nope"}, "invalid_dataset"), ({"csv_content": _csv(rows=5)}, "invalid_dataset"),
     ({"csv_content": ""}, "validation_error")],
)
def test_bad_uploads_return_400(client, overrides, code) -> None:
    resp = _ingest(client, **overrides)
    assert resp.status_code == 400 and resp.json()["error"] == code


def test_unknown_resources_return_404_in_the_error_shape(client) -> None:
    for path in ("/api/datasets/missing", "/api/sessions/missing"):
        resp = client.get(path)
        assert resp.status_code == 404
        assert resp.json()["error"] == "not_found" and resp.json()["message"]


def test_dataset_delete_refuses_while_in_use_unless_cascading(client, repo) -> None:
    ds_id = _ingest(client).json()["dataset_id"]
    sid = repo.create_session("q", ds_id).session_id
    resp = client.delete(f"/api/datasets/{ds_id}")
    assert resp.status_code == 409 and resp.json()["error"] == "dataset_in_use"

    resp = client.delete(f"/api/datasets/{ds_id}?cascade=true")
    assert resp.status_code == 200 and resp.json()["sessions_deleted"] == 1
    assert client.get(f"/api/sessions/{sid}").status_code == 404


# --------------------------------------------------------------------- sessions
def _session(client, repo) -> str:
    repo.create_dataset(make_profile("ds-1"))
    resp = client.post("/api/sessions", json={"research_question": " Does dropout help? ", "dataset_id": "ds-1"})
    assert resp.status_code == 201, resp.text
    return resp.json()["session_id"]


def test_create_list_and_read_a_session(client, repo) -> None:
    sid = _session(client, repo)
    detail = client.get(f"/api/sessions/{sid}").json()
    assert detail["research_question"] == "Does dropout help?"
    assert (detail["status"], detail["plan"], detail["analysis"], detail["experiments"]) == ("pending", None, None, [])
    assert detail["max_rounds"] >= 1 and detail["n_seeds"] >= 1
    assert detail["created_at"].endswith("+00:00")  # explicit UTC for the browser
    assert [s["session_id"] for s in client.get("/api/sessions").json()] == [sid]


def test_create_session_for_unknown_dataset_is_404(client) -> None:
    resp = client.post("/api/sessions", json={"research_question": "q", "dataset_id": "nope"})
    assert resp.status_code == 404


def test_run_starts_the_investigation_in_the_background(repo) -> None:
    started = []
    client = make_test_client(repo, investigator=started.append)
    sid = _session(client, repo)

    resp = client.post(f"/api/sessions/{sid}/run")
    assert resp.status_code == 202 and resp.json()["status"] == "running"
    assert started == [sid]

    again = client.post(f"/api/sessions/{sid}/run")
    assert again.status_code == 409 and again.json()["error"] == "conflict"


def test_a_failed_run_restarts_cleanly_and_a_finished_one_is_final(repo) -> None:
    client = make_test_client(repo, investigator=lambda sid: repo.set_status(sid, "done"))
    sid = _session(client, repo)
    repo.set_status(sid, "failed", error="LLM timed out")

    assert client.post(f"/api/sessions/{sid}/run").status_code == 202
    detail = client.get(f"/api/sessions/{sid}").json()
    assert detail["status"] == "done" and detail["error"] is None
    assert client.post(f"/api/sessions/{sid}/run").status_code == 409


def test_detail_after_a_run_has_validation_analysis_and_a_report(repo) -> None:
    def investigate(sid):
        run_investigation(sid, repo, FakePlanner(), FakeRecommender(), run=fake_run(), max_rounds=2, n_seeds=2)

    client = make_test_client(repo, investigator=investigate)
    sid = _session(client, repo)
    client.post(f"/api/sessions/{sid}/run")
    detail = client.get(f"/api/sessions/{sid}").json()

    assert detail["status"] == "done", detail["error"]
    assert detail["analysis"]["split"] == "val"
    assert [c["label"] for c in detail["analysis"]["conditions"]] == ["dropout=0", "dropout=0.5"]
    assert detail["report"]["primary"]["verdict"] in ("better", "worse", "inconclusive")
    assert len(detail["decisions"]) == 1
    experiment = detail["experiments"][0]
    assert "val_scores" not in experiment and "test_scores" not in experiment  # per-row data stays server-side
    assert experiment["created_at"].endswith("+00:00")


def test_delete_session(client, repo) -> None:
    sid = _session(client, repo)
    resp = client.delete(f"/api/sessions/{sid}")
    assert resp.status_code == 200 and resp.json()["deleted"] == "session"
    assert client.get(f"/api/sessions/{sid}").status_code == 404


def test_unhandled_errors_become_500(repo, monkeypatch) -> None:
    client = make_test_client(repo)
    monkeypatch.setattr(repo, "list_sessions", lambda: 1 / 0)
    resp = client.get("/api/sessions")
    assert resp.status_code == 500 and resp.json()["error"] == "internal_error"
