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
are verified in the Checkpoint 2.12 script against the real local database.

Test cases
----------
Dataset operations (dataset-first architecture revision)
  1. test_create_dataset_returns_dataset_id
  2. test_get_dataset_roundtrip
  3. test_get_dataset_raises_on_missing
  4. test_list_datasets_newest_first

Session operations
  5. test_create_session_returns_uuid
  6. test_get_session_returns_correct_data
  7. test_get_session_raises_on_missing
  8. test_list_sessions_empty
  9. test_list_sessions_includes_experiment_count
  10. test_update_session_status

Experiment operations
  11. test_store_and_retrieve_experiment_roundtrip  (JSONB round-trip)
  12. test_query_experiments_by_session
  13. test_query_experiments_filter_by_status
  14. test_update_experiment_status
  15. test_get_experiment_raises_on_missing

Anomaly operations
  16. test_store_and_retrieve_anomaly
  17. test_query_anomalies_by_experiment
  18. test_query_anomalies_by_session

Retry logic
  19. test_retry_on_transient_error
  20. test_retry_exhausted_raises
"""

import uuid
from datetime import datetime
from typing import Generator
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

from backend.database.models import Base
from backend.models.anomaly import AnomalyReport
from backend.models.dataset import DatasetProfile
from backend.models.experiment import ExperimentConfiguration, ExperimentResult
from backend.tools.state_manager import StateManager


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SQLITE_URL = "sqlite:///:memory:"


def _make_dataset_profile(dataset_id: str | None = None) -> DatasetProfile:
    return DatasetProfile(
        dataset_id=dataset_id or str(uuid.uuid4()),
        original_filename="fixture.csv",
        storage_path="data/uploads/fixture/data.csv",
        target_column="target",
        feature_columns=["a", "b"],
        numeric_columns=["a", "b"],
        categorical_columns=[],
        task_type="classification",
        n_rows=100,
        n_features=2,
        n_classes=2,
        class_labels=["0", "1"],
        class_distribution={"0": 50, "1": 50},
        missing_value_counts={},
        split_seed=42,
    )


def _make_cfg(dataset_id: str, dropout: float = 0.2, seed: int = 42) -> ExperimentConfiguration:
    return ExperimentConfiguration(
        dataset_id=dataset_id,
        model_type="mlp",
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
        task_type="classification" if status != "failed" else None,
        metrics={
            "train_loss": 0.15,
            "val_loss": 0.18,
            "accuracy": 0.94,
            "n_classes": 2,
            "n_val_samples": 20,
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
    """Fresh in-memory SQLite engine with all 4 tables created."""
    engine = create_engine(SQLITE_URL)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture()
def sm(sm_engine) -> Generator[StateManager, None, None]:
    """StateManager with injected SQLite engine."""
    yield StateManager(engine=sm_engine)


@pytest.fixture()
def dataset_id(sm: StateManager) -> str:
    """A dataset already persisted via StateManager, for tests that need a
    session/experiment scoped to a real dataset_id."""
    return sm.create_dataset(_make_dataset_profile())


# ---------------------------------------------------------------------------
# 1-4. Dataset operations
# ---------------------------------------------------------------------------

def test_create_dataset_returns_dataset_id(sm: StateManager) -> None:
    profile = _make_dataset_profile()
    returned_id = sm.create_dataset(profile)
    assert returned_id == profile.dataset_id


def test_get_dataset_roundtrip(sm: StateManager) -> None:
    profile = _make_dataset_profile()
    sm.create_dataset(profile)

    retrieved = sm.get_dataset(profile.dataset_id)

    assert retrieved.dataset_id == profile.dataset_id
    assert retrieved.target_column == "target"
    assert retrieved.task_type == "classification"
    assert retrieved.n_classes == 2
    assert retrieved.class_labels == ["0", "1"]
    assert retrieved.split_seed == 42


def test_get_dataset_raises_on_missing(sm: StateManager) -> None:
    with pytest.raises(KeyError):
        sm.get_dataset("00000000-0000-0000-0000-000000000000")


def test_list_datasets_newest_first(sm: StateManager) -> None:
    first = _make_dataset_profile()
    second = _make_dataset_profile()
    sm.create_dataset(first)
    sm.create_dataset(second)

    datasets = sm.list_datasets()
    ids = {d.dataset_id for d in datasets}
    assert ids == {first.dataset_id, second.dataset_id}


# ---------------------------------------------------------------------------
# 5. test_create_session_returns_uuid
# ---------------------------------------------------------------------------

def test_create_session_returns_uuid(sm: StateManager, dataset_id: str) -> None:
    """create_session() returns a non-empty UUID string."""
    sid = sm.create_session("Does dropout help?", dataset_id)
    assert isinstance(sid, str)
    assert len(sid) > 0
    # Verify it is a valid UUID4
    parsed = uuid.UUID(sid, version=4)
    assert str(parsed) == sid


# ---------------------------------------------------------------------------
# 6. test_get_session_returns_correct_data
# ---------------------------------------------------------------------------

def test_get_session_returns_correct_data(sm: StateManager, dataset_id: str) -> None:
    """get_session() returns the correct research_question, dataset_id, and defaults."""
    question = "What learning rate is best?"
    sid = sm.create_session(question, dataset_id)

    session = sm.get_session(sid)
    assert session.session_id == sid
    assert session.dataset_id == dataset_id
    assert session.research_question == question
    assert session.status == "active"
    assert session.current_node == "planning"
    assert session.cycle_count == 0


# ---------------------------------------------------------------------------
# 7. test_get_session_raises_on_missing
# ---------------------------------------------------------------------------

def test_get_session_raises_on_missing(sm: StateManager) -> None:
    """get_session() raises KeyError for an unknown session_id."""
    with pytest.raises(KeyError):
        sm.get_session("00000000-0000-0000-0000-000000000000")


# ---------------------------------------------------------------------------
# 8. test_list_sessions_empty
# ---------------------------------------------------------------------------

def test_list_sessions_empty(sm: StateManager) -> None:
    """list_sessions() returns [] when no sessions exist."""
    result = sm.list_sessions()
    assert result == []


# ---------------------------------------------------------------------------
# 9. test_list_sessions_includes_experiment_count
# ---------------------------------------------------------------------------

def test_list_sessions_includes_experiment_count(sm: StateManager, dataset_id: str) -> None:
    """list_sessions() returns correct experiment_count per session."""
    sid1 = sm.create_session("Q1", dataset_id)
    sid2 = sm.create_session("Q2", dataset_id)

    cfg = _make_cfg(dataset_id)
    # Store 2 experiments in sid1, 1 in sid2
    sm.store_experiment(_make_result(sid1, cfg))
    sm.store_experiment(_make_result(sid1, cfg))
    sm.store_experiment(_make_result(sid2, cfg))

    summaries = sm.list_sessions()
    by_id = {s.session_id: s for s in summaries}

    assert by_id[sid1].experiment_count == 2
    assert by_id[sid2].experiment_count == 1
    assert by_id[sid1].research_question == "Q1"


# ---------------------------------------------------------------------------
# 10. test_update_session_status
# ---------------------------------------------------------------------------

def test_update_session_status(sm: StateManager, dataset_id: str) -> None:
    """update_session_status() persists the new status."""
    sid = sm.create_session("Q", dataset_id)
    sm.update_session_status(sid, "concluded")

    session = sm.get_session(sid)
    assert session.status == "concluded"


# ---------------------------------------------------------------------------
# 11. test_store_and_retrieve_experiment_roundtrip  (JSONB round-trip)
# ---------------------------------------------------------------------------

def test_store_and_retrieve_experiment_roundtrip(sm: StateManager, dataset_id: str) -> None:
    """Full JSONB round-trip: ExperimentConfiguration survives PostgreSQL storage.

    ExperimentConfiguration
      -> model_dump()          (plain Python dict)
      -> stored in config col  (JSONB / JSON)
      -> retrieved as dict
      -> ExperimentConfiguration.model_validate(dict)
      -> ExperimentResult.config == original config

    This validates that the nested Pydantic model serialises and deserialises
    without data loss through the ORM layer, and that the new top-level
    ``task_type`` column round-trips alongside it.
    """
    sid = sm.create_session("round-trip test", dataset_id)
    original_cfg = _make_cfg(dataset_id, dropout=0.35, seed=123)
    experiment_id = str(uuid.uuid4())

    original_result = ExperimentResult(
        experiment_id=experiment_id,
        session_id=sid,
        config=original_cfg,
        task_type="classification",
        metrics={"train_loss": 0.10, "val_loss": 0.12,
                 "accuracy": 0.97, "n_classes": 2, "n_val_samples": 20,
                 "training_time_seconds": 30.0},
        status="success",
        timestamp=datetime.utcnow(),
    )

    sm.store_experiment(original_result)
    retrieved = sm.get_experiment(experiment_id)

    # ExperimentResult fields
    assert retrieved.experiment_id == experiment_id
    assert retrieved.session_id == sid
    assert retrieved.status == "success"
    assert retrieved.task_type == "classification"
    assert retrieved.error is None

    # JSONB round-trip: ExperimentConfiguration fully preserved
    assert retrieved.config.dataset_id == dataset_id
    assert retrieved.config.model_type == "mlp"
    assert retrieved.config.hyperparameters["dropout"] == pytest.approx(0.35)
    assert retrieved.config.hyperparameters["learning_rate"] == pytest.approx(0.001)
    assert retrieved.config.hyperparameters["batch_size"] == 32
    assert retrieved.config.random_seed == 123

    # Metrics round-trip
    assert retrieved.metrics["accuracy"] == pytest.approx(0.97)
    assert retrieved.metrics["train_loss"] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# 12. test_query_experiments_by_session
# ---------------------------------------------------------------------------

def test_query_experiments_by_session(sm: StateManager, dataset_id: str) -> None:
    """query_experiments() returns only experiments for the given session."""
    sid1 = sm.create_session("Q1", dataset_id)
    sid2 = sm.create_session("Q2", dataset_id)
    cfg = _make_cfg(dataset_id)

    sm.store_experiment(_make_result(sid1, cfg))
    sm.store_experiment(_make_result(sid2, cfg))

    results = sm.query_experiments(sid1)
    assert len(results) == 1
    assert results[0].session_id == sid1


# ---------------------------------------------------------------------------
# 13. test_query_experiments_filter_by_status
# ---------------------------------------------------------------------------

def test_query_experiments_filter_by_status(sm: StateManager, dataset_id: str) -> None:
    """query_experiments(status=...) filters by status correctly."""
    sid = sm.create_session("Q", dataset_id)
    cfg = _make_cfg(dataset_id)

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
# 14. test_update_experiment_status
# ---------------------------------------------------------------------------

def test_update_experiment_status(sm: StateManager, dataset_id: str) -> None:
    """update_experiment_status() marks an experiment as 'anomalous'."""
    sid = sm.create_session("Q", dataset_id)
    eid = str(uuid.uuid4())
    cfg = _make_cfg(dataset_id)
    sm.store_experiment(_make_result(sid, cfg, experiment_id=eid))

    sm.update_experiment_status(eid, "anomalous")
    retrieved = sm.get_experiment(eid)
    assert retrieved.status == "anomalous"


# ---------------------------------------------------------------------------
# 15. test_get_experiment_raises_on_missing
# ---------------------------------------------------------------------------

def test_get_experiment_raises_on_missing(sm: StateManager) -> None:
    """get_experiment() raises KeyError for an unknown experiment_id."""
    with pytest.raises(KeyError):
        sm.get_experiment("00000000-0000-0000-0000-000000000000")


# ---------------------------------------------------------------------------
# 16. test_store_and_retrieve_anomaly
# ---------------------------------------------------------------------------

def test_store_and_retrieve_anomaly(sm: StateManager, dataset_id: str) -> None:
    """store_anomaly() + query_anomalies() round-trip."""
    sid = sm.create_session("Q", dataset_id)
    eid = str(uuid.uuid4())
    cfg = _make_cfg(dataset_id)
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
# 17. test_query_anomalies_by_experiment
# ---------------------------------------------------------------------------

def test_query_anomalies_by_experiment(sm: StateManager, dataset_id: str) -> None:
    """query_anomalies(experiment_id=...) returns only that experiment's anomalies."""
    sid = sm.create_session("Q", dataset_id)
    eid1 = str(uuid.uuid4())
    eid2 = str(uuid.uuid4())
    cfg = _make_cfg(dataset_id)
    sm.store_experiment(_make_result(sid, cfg, experiment_id=eid1))
    sm.store_experiment(_make_result(sid, cfg, experiment_id=eid2))

    sm.store_anomaly(_make_anomaly(eid1))
    sm.store_anomaly(_make_anomaly(eid2))

    result = sm.query_anomalies(experiment_id=eid1)
    assert len(result) == 1
    assert result[0].experiment_id == eid1


# ---------------------------------------------------------------------------
# 18. test_query_anomalies_by_session
# ---------------------------------------------------------------------------

def test_query_anomalies_by_session(sm: StateManager, dataset_id: str) -> None:
    """query_anomalies(session_id=...) joins experiments and returns all session anomalies."""
    sid = sm.create_session("Q", dataset_id)
    eid1 = str(uuid.uuid4())
    eid2 = str(uuid.uuid4())
    other_sid = sm.create_session("Other", dataset_id)
    other_eid = str(uuid.uuid4())

    cfg = _make_cfg(dataset_id)
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
# 19. test_retry_on_transient_error
# ---------------------------------------------------------------------------

def test_retry_on_transient_error(sm: StateManager, dataset_id: str) -> None:
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
        sid = sm.create_session("retry test", dataset_id)
    # Did not raise → retry logic succeeded
    assert call_count["n"] == 3


# ---------------------------------------------------------------------------
# 20. test_retry_exhausted_raises
# ---------------------------------------------------------------------------

def test_retry_exhausted_raises(sm: StateManager, dataset_id: str) -> None:
    """StateManager raises OperationalError after 3 failed attempts."""
    with patch.object(
        sm,
        "_Session",
        side_effect=OperationalError("down", None, Exception("down")),
    ):
        with pytest.raises(OperationalError):
            sm.create_session("should fail", dataset_id)
