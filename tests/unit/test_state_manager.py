"""
tests/unit/test_state_manager.py
==================================
Unit tests for backend/tools/state_manager.py.

All tests use an in-memory SQLite engine injected via StateManager(engine=...).
No live PostgreSQL connection is required.

SQLite limitations + workarounds
---------------------------------
SQLite does not support JSONB; the ORM models fall back to JSON type which
stores dicts as JSON strings.  SQLAlchemy handles serialisation/deserialisation
transparently, so the JSONB round-trip path
    ExperimentConfiguration -> model_dump() -> JSONB col -> dict -> model_validate()
is fully tested here.  PostgreSQL-specific features (index GIN, UUID native)
are verified in the Checkpoint 2.11 script against the real local database.

Test cases
----------
Session operations (Tasks 2.6, 2.10)
  1. test_create_session_returns_uuid
  2. test_get_session_returns_correct_data
  3. test_get_session_raises_on_missing
  4. test_list_sessions_empty
  5. test_list_sessions_includes_experiment_count
  6. test_update_session_status

Experiment operations (Tasks 2.7, 2.10)
  7. test_store_and_retrieve_experiment_roundtrip  (JSONB round-trip, Task 2.1 correction)
  8. test_query_experiments_by_session
  9. test_query_experiments_filter_by_status
  10. test_update_experiment_status
  11. test_get_experiment_raises_on_missing

Anomaly operations (Task 2.8, 2.10)
  12. test_store_and_retrieve_anomaly
  13. test_query_anomalies_by_experiment
  14. test_query_anomalies_by_session

Retry logic (Task 2.9)
  15. test_retry_on_transient_error
  16. test_retry_exhausted_raises
"""

import uuid
from datetime import datetime
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from backend.database.models import Base
from backend.models.anomaly import AnomalyReport
from backend.models.experiment import ExperimentConfiguration, ExperimentResult
from backend.tools.state_manager import StateManager


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SQLITE_URL = "sqlite:///:memory:"


def _make_cfg(dropout: float = 0.2, seed: int = 42) -> ExperimentConfiguration:
    return ExperimentConfiguration(
        model_type="mnist_mlp",
        hyperparameters={
            "dropout": dropout,
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        random_seed=seed,
    )


def _make_result(
    session_id: str,
    cfg: ExperimentConfiguration,
    status: str = "success",
    experiment_id: str | None = None,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id or str(uuid.uuid4()),
        session_id=session_id,
        config=cfg,
        metrics={
            "train_loss": 0.15,
            "val_loss": 0.18,
            "accuracy": 0.94,
            "training_time_seconds": 45.2,
        }
        if status != "failed"
        else None,
        status=status,  # type: ignore[arg-type]
        error="oops" if status == "failed" else None,
        timestamp=datetime.utcnow(),
    )


def _make_anomaly(experiment_id: str) -> AnomalyReport:
    return AnomalyReport(
        experiment_id=experiment_id,
        rule="outlier_detection",
        explanation="Validation val_loss of 2.3 is 3.7 std devs from mean.",
        severity="warning",
        detected_at=datetime.utcnow(),
    )


@pytest.fixture()
def sm_engine():
    """Fresh in-memory SQLite engine with all 3 tables created."""
    engine = create_engine(SQLITE_URL)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture()
def sm(sm_engine) -> Generator[StateManager, None, None]:
    """StateManager with injected SQLite engine."""
    yield StateManager(engine=sm_engine)


# ---------------------------------------------------------------------------
# 1. test_create_session_returns_uuid
# ---------------------------------------------------------------------------

def test_create_session_returns_uuid(sm: StateManager) -> None:
    """create_session() returns a non-empty UUID string."""
    sid = sm.create_session("Does dropout help?")
    assert isinstance(sid, str)
    assert len(sid) > 0
    # Verify it is a valid UUID4
    parsed = uuid.UUID(sid, version=4)
    assert str(parsed) == sid


# ---------------------------------------------------------------------------
# 2. test_get_session_returns_correct_data
# ---------------------------------------------------------------------------

def test_get_session_returns_correct_data(sm: StateManager) -> None:
    """get_session() returns the correct research_question and defaults."""
    question = "What learning rate is best?"
    sid = sm.create_session(question)

    session = sm.get_session(sid)
    assert session.session_id == sid
    assert session.research_question == question
    assert session.status == "active"
    assert session.current_node == "planning"
    assert session.cycle_count == 0


# ---------------------------------------------------------------------------
# 3. test_get_session_raises_on_missing
# ---------------------------------------------------------------------------

def test_get_session_raises_on_missing(sm: StateManager) -> None:
    """get_session() raises KeyError for an unknown session_id."""
    with pytest.raises(KeyError):
        sm.get_session("00000000-0000-0000-0000-000000000000")


# ---------------------------------------------------------------------------
# 4. test_list_sessions_empty
# ---------------------------------------------------------------------------

def test_list_sessions_empty(sm: StateManager) -> None:
    """list_sessions() returns [] when no sessions exist."""
    result = sm.list_sessions()
    assert result == []


# ---------------------------------------------------------------------------
# 5. test_list_sessions_includes_experiment_count
# ---------------------------------------------------------------------------

def test_list_sessions_includes_experiment_count(sm: StateManager) -> None:
    """list_sessions() returns correct experiment_count per session."""
    sid1 = sm.create_session("Q1")
    sid2 = sm.create_session("Q2")

    cfg = _make_cfg()
    # Store 2 experiments in sid1, 1 in sid2
    sm.store_experiment(_make_result(sid1, cfg))
    sm.store_experiment(_make_result(sid1, cfg, seed=99))
    sm.store_experiment(_make_result(sid2, cfg))

    summaries = sm.list_sessions()
    # list_sessions orders by created_at DESC → sid2 first, then sid1
    by_id = {s.session_id: s for s in summaries}

    assert by_id[sid1].experiment_count == 2
    assert by_id[sid2].experiment_count == 1
    assert by_id[sid1].research_question == "Q1"


def _make_result(session_id, cfg, status="success", experiment_id=None, seed=42):
    return ExperimentResult(
        experiment_id=experiment_id or str(uuid.uuid4()),
        session_id=session_id,
        config=cfg,
        metrics={
            "train_loss": 0.15, "val_loss": 0.18,
            "accuracy": 0.94, "training_time_seconds": 45.2,
        } if status != "failed" else None,
        status=status,
        error="oops" if status == "failed" else None,
        timestamp=datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# 6. test_update_session_status
# ---------------------------------------------------------------------------

def test_update_session_status(sm: StateManager) -> None:
    """update_session_status() persists the new status."""
    sid = sm.create_session("Q")
    sm.update_session_status(sid, "concluded")

    session = sm.get_session(sid)
    assert session.status == "concluded"


# ---------------------------------------------------------------------------
# 7. test_store_and_retrieve_experiment_roundtrip  (JSONB round-trip)
# ---------------------------------------------------------------------------

def test_store_and_retrieve_experiment_roundtrip(sm: StateManager) -> None:
    """Full JSONB round-trip: ExperimentConfiguration survives PostgreSQL storage.

    ExperimentConfiguration
      -> model_dump()          (plain Python dict)
      -> stored in config col  (JSONB / JSON)
      -> retrieved as dict
      -> ExperimentConfiguration.model_validate(dict)
      -> ExperimentResult.config == original config

    This validates that the nested Pydantic model serialises and deserialises
    without data loss through the ORM layer.
    """
    sid = sm.create_session("round-trip test")
    original_cfg = _make_cfg(dropout=0.35, seed=123)
    experiment_id = str(uuid.uuid4())

    original_result = ExperimentResult(
        experiment_id=experiment_id,
        session_id=sid,
        config=original_cfg,
        metrics={"train_loss": 0.10, "val_loss": 0.12,
                 "accuracy": 0.97, "training_time_seconds": 30.0},
        status="success",
        timestamp=datetime.utcnow(),
    )

    sm.store_experiment(original_result)
    retrieved = sm.get_experiment(experiment_id)

    # ExperimentResult fields
    assert retrieved.experiment_id == experiment_id
    assert retrieved.session_id == sid
    assert retrieved.status == "success"
    assert retrieved.error is None

    # JSONB round-trip: ExperimentConfiguration fully preserved
    assert retrieved.config.model_type == "mnist_mlp"
    assert retrieved.config.hyperparameters["dropout"] == pytest.approx(0.35)
    assert retrieved.config.hyperparameters["learning_rate"] == pytest.approx(0.001)
    assert retrieved.config.hyperparameters["batch_size"] == 32
    assert retrieved.config.random_seed == 123

    # Metrics round-trip
    assert retrieved.metrics["accuracy"] == pytest.approx(0.97)
    assert retrieved.metrics["train_loss"] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# 8. test_query_experiments_by_session
# ---------------------------------------------------------------------------

def test_query_experiments_by_session(sm: StateManager) -> None:
    """query_experiments() returns only experiments for the given session."""
    sid1 = sm.create_session("Q1")
    sid2 = sm.create_session("Q2")
    cfg = _make_cfg()

    sm.store_experiment(_make_result(sid1, cfg))
    sm.store_experiment(_make_result(sid2, cfg))

    results = sm.query_experiments(sid1)
    assert len(results) == 1
    assert results[0].session_id == sid1


# ---------------------------------------------------------------------------
# 9. test_query_experiments_filter_by_status
# ---------------------------------------------------------------------------

def test_query_experiments_filter_by_status(sm: StateManager) -> None:
    """query_experiments(status=...) filters by status correctly."""
    sid = sm.create_session("Q")
    cfg = _make_cfg()

    sm.store_experiment(_make_result(sid, cfg, status="success"))
    sm.store_experiment(_make_result(sid, cfg, status="failed"))

    successes = sm.query_experiments(sid, status="success")
    failures = sm.query_experiments(sid, status="failed")
    all_exp = sm.query_experiments(sid)

    assert len(successes) == 1
    assert len(failures) == 1
    assert len(all_exp) == 2
    assert all(e.status == "success" for e in successes)
    assert all(e.status == "failed" for e in failures)


# ---------------------------------------------------------------------------
# 10. test_update_experiment_status
# ---------------------------------------------------------------------------

def test_update_experiment_status(sm: StateManager) -> None:
    """update_experiment_status() marks an experiment as 'anomalous'."""
    sid = sm.create_session("Q")
    eid = str(uuid.uuid4())
    cfg = _make_cfg()
    sm.store_experiment(_make_result(sid, cfg, experiment_id=eid))

    sm.update_experiment_status(eid, "anomalous")
    retrieved = sm.get_experiment(eid)
    assert retrieved.status == "anomalous"


# ---------------------------------------------------------------------------
# 11. test_get_experiment_raises_on_missing
# ---------------------------------------------------------------------------

def test_get_experiment_raises_on_missing(sm: StateManager) -> None:
    """get_experiment() raises KeyError for an unknown experiment_id."""
    with pytest.raises(KeyError):
        sm.get_experiment("00000000-0000-0000-0000-000000000000")


# ---------------------------------------------------------------------------
# 12. test_store_and_retrieve_anomaly
# ---------------------------------------------------------------------------

def test_store_and_retrieve_anomaly(sm: StateManager) -> None:
    """store_anomaly() + query_anomalies() round-trip."""
    sid = sm.create_session("Q")
    eid = str(uuid.uuid4())
    cfg = _make_cfg()
    sm.store_experiment(_make_result(sid, cfg, experiment_id=eid))

    anomaly = _make_anomaly(eid)
    sm.store_anomaly(anomaly)

    retrieved = sm.query_anomalies(experiment_id=eid)
    assert len(retrieved) == 1
    a = retrieved[0]
    assert a.experiment_id == eid
    assert a.rule == "outlier_detection"
    assert a.severity == "warning"
    assert "std devs" in a.explanation


# ---------------------------------------------------------------------------
# 13. test_query_anomalies_by_experiment
# ---------------------------------------------------------------------------

def test_query_anomalies_by_experiment(sm: StateManager) -> None:
    """query_anomalies(experiment_id=...) returns only that experiment's anomalies."""
    sid = sm.create_session("Q")
    eid1 = str(uuid.uuid4())
    eid2 = str(uuid.uuid4())
    cfg = _make_cfg()
    sm.store_experiment(_make_result(sid, cfg, experiment_id=eid1))
    sm.store_experiment(_make_result(sid, cfg, experiment_id=eid2))

    sm.store_anomaly(_make_anomaly(eid1))
    sm.store_anomaly(_make_anomaly(eid2))

    result = sm.query_anomalies(experiment_id=eid1)
    assert len(result) == 1
    assert result[0].experiment_id == eid1


# ---------------------------------------------------------------------------
# 14. test_query_anomalies_by_session
# ---------------------------------------------------------------------------

def test_query_anomalies_by_session(sm: StateManager) -> None:
    """query_anomalies(session_id=...) joins experiments and returns all session anomalies."""
    sid = sm.create_session("Q")
    eid1 = str(uuid.uuid4())
    eid2 = str(uuid.uuid4())
    other_sid = sm.create_session("Other")
    other_eid = str(uuid.uuid4())

    cfg = _make_cfg()
    sm.store_experiment(_make_result(sid, cfg, experiment_id=eid1))
    sm.store_experiment(_make_result(sid, cfg, experiment_id=eid2))
    sm.store_experiment(_make_result(other_sid, cfg, experiment_id=other_eid))

    sm.store_anomaly(_make_anomaly(eid1))
    sm.store_anomaly(_make_anomaly(eid2))
    sm.store_anomaly(_make_anomaly(other_eid))  # should NOT appear in sid query

    result = sm.query_anomalies(session_id=sid)
    assert len(result) == 2
    exp_ids = {a.experiment_id for a in result}
    assert exp_ids == {eid1, eid2}


# ---------------------------------------------------------------------------
# 15. test_retry_on_transient_error
# ---------------------------------------------------------------------------

def test_retry_on_transient_error(sm: StateManager) -> None:
    """StateManager retries on OperationalError and succeeds on 3rd attempt."""
    call_count = {"n": 0}
    original_create = sm._Session

    class FakeSession:
        """Simulates a session that fails twice then works."""

        def __init__(self):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise OperationalError("transient", None, Exception("down"))
            # 3rd call: use a real session
            self._inner = original_create()

        def __enter__(self):
            return self._inner.__enter__()

        def __exit__(self, *args):
            return self._inner.__exit__(*args)

    with patch.object(sm, "_Session", side_effect=FakeSession):
        sid = sm.create_session("retry test")
    # Did not raise → retry logic succeeded
    assert call_count["n"] == 3


# ---------------------------------------------------------------------------
# 16. test_retry_exhausted_raises
# ---------------------------------------------------------------------------

def test_retry_exhausted_raises(sm: StateManager) -> None:
    """StateManager raises OperationalError after 3 failed attempts."""
    with patch.object(
        sm,
        "_Session",
        side_effect=OperationalError("down", None, Exception("down")),
    ):
        with pytest.raises(OperationalError):
            sm.create_session("should fail")
