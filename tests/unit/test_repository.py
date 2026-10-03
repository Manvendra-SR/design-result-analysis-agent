"""Tests for backend/database/repository.py against in-memory SQLite."""

from __future__ import annotations

import pytest

from backend.models.dataset import DatasetInUseError
from backend.models.investigation import Decision
from tests._fakes import dropout_plan, fake_run, make_profile


def _session(repo, dataset_id="ds-1"):
    repo.create_dataset(make_profile(dataset_id))
    return repo.create_session("Does dropout help?", dataset_id)


def test_dataset_round_trip(repo) -> None:
    repo.create_dataset(make_profile("ds-1"))
    assert repo.get_dataset("ds-1").class_distribution == {"0": 600, "1": 400}
    assert [d.dataset_id for d in repo.list_datasets()] == ["ds-1"]
    with pytest.raises(KeyError):
        repo.get_dataset("missing")


def test_dataset_in_use_needs_cascade(repo) -> None:
    session = _session(repo)
    repo.add_experiment(fake_run()(dropout_plan().get("c1").config("ds-1"), None, session.session_id, 1))
    with pytest.raises(DatasetInUseError):
        repo.delete_dataset("ds-1")
    assert repo.delete_dataset("ds-1", cascade=True) == (1, 1)
    assert repo.list_sessions() == []


def test_session_lifecycle(repo) -> None:
    session = _session(repo)
    assert session.status == "pending" and session.decisions == [] and session.plan is None
    with pytest.raises(KeyError):
        repo.create_session("q", "missing-dataset")

    repo.set_status(session.session_id, "running")
    repo.save_plan(session.session_id, dropout_plan())
    repo.append_decision(session.session_id, Decision(round=1, action="refine", parent="c1", knob="dropout", values=[0.2],
                                                          new_candidates=["c3"], rationale="a"))
    repo.append_decision(session.session_id, Decision(round=2, action="conclude", rationale="b"))

    loaded = repo.get_session(session.session_id)
    assert loaded.status == "running"
    assert loaded.plan == dropout_plan()
    assert [d.action for d in loaded.decisions] == ["refine", "conclude"]
    assert loaded.decisions[0].new_candidates == ["c3"]


def test_experiments_keep_their_per_row_scores(repo) -> None:
    session = _session(repo)
    result = fake_run()(dropout_plan().get("c2").config("ds-1", seed=2), None, session.session_id, 1)
    repo.add_experiment(result)

    [loaded] = repo.list_experiments(session.session_id)
    assert loaded.val_scores == result.val_scores and loaded.test_scores == result.test_scores
    assert loaded.config == result.config
    assert repo.list_sessions()[0].experiment_count == 1


def test_reset_and_interrupted_runs(repo) -> None:
    session = _session(repo)
    sid = session.session_id
    repo.save_plan(sid, dropout_plan())
    repo.add_experiment(fake_run()(dropout_plan().get("c1").config("ds-1"), None, sid, 1))
    repo.set_status(sid, "running")

    assert repo.fail_interrupted_runs() == 1
    assert repo.get_session(sid).status == "failed"

    repo.reset_session(sid)
    reset = repo.get_session(sid)
    assert (reset.status, reset.plan, reset.error) == ("pending", None, None)
    assert repo.list_experiments(sid) == []


def test_delete_session(repo) -> None:
    session = _session(repo)
    repo.add_experiment(fake_run()(dropout_plan().get("c1").config("ds-1"), None, session.session_id, 1))
    assert repo.delete_session(session.session_id) == 1
    with pytest.raises(KeyError):
        repo.get_session(session.session_id)
