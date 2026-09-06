"""
backend/tools/state_manager.py
================================
StateManager: PostgreSQL interface for all dataset, experiment, and session
persistence.

This is the single class through which all database reads and writes are
performed.  No other module should import from ``backend.database.models``
directly — they must go through StateManager.

Architecture
------------
- Uses SQLAlchemy ORM (``DatasetModel``, ``SessionModel``, ``ExperimentModel``,
  ``AnomalyModel``)
- Pydantic models are the external-facing types; ORM models are internal
- All public methods carry a tenacity ``@retry`` decorator (max 3 attempts,
  exponential backoff) that retries on ``OperationalError`` (connection blips)
- Engine injection via constructor kwarg allows tests to pass a pre-built
  in-memory SQLite engine without monkeypatching

Engine / Session lifecycle
--------------------------
The constructor creates a SQLAlchemy ``sessionmaker`` factory bound to the
engine.  Each public method opens a fresh session per call, commits on
success, and rolls back automatically on any exception.

JSONB round-trip
----------------
PostgreSQL stores ``ExperimentConfiguration`` and ``DatasetProfile`` as
JSONB (in ``experiments.config`` and ``datasets.profile`` respectively).
Round-trip path::

    ExperimentConfiguration / DatasetProfile
      -> model_dump()              (Python dict)
      -> stored by SQLAlchemy      (JSONB column)
      -> returned from PostgreSQL  (Python dict)
      -> Model.model_validate(dict)

This is exercised by the explicit round-trip tests in
``tests/unit/test_state_manager.py``.

Dataset-first architecture revision
-------------------------------------
``create_session`` now requires ``dataset_id`` — every session is scoped to
exactly one dataset (see ``backend/models/dataset.py``). ``create_dataset``/
``get_dataset``/``list_datasets`` were added alongside the pre-existing
session/experiment/anomaly methods; the underlying CSV file itself is never
stored here — only the ``DatasetProfile`` metadata (the file lives on disk,
written by ``backend/tools/dataset/ingestion.py``).

Requirements
------------
1.7  State_Manager persists DatasetProfile metadata
4.1  State_Manager persists experiment configurations and results
4.2  State_Manager assigns unique identifiers
4.3  State_Manager records timestamp, config JSON, metrics, status, session FK
4.4  State_Manager supports querying by session_id and status
4.5  State_Manager returns experiments ordered by timestamp
4.7  State_Manager supports atomic transactions
4.8  State_Manager retries up to 3 times with exponential backoff
5.5  State_Manager updates experiment status to 'anomalous'
8.1  State_Manager creates sessions scoped to a dataset, with unique identifiers
8.4  State_Manager persists research question, experiment count, cycle count
8.6  State_Manager lists sessions with summary statistics
"""

from __future__ import annotations

import logging
from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import backend.config  # noqa: F401  side-effect: loads .env + configures logging
from backend.database.connection import get_engine
from backend.database.models import (
    AnomalyModel,
    DatasetModel,
    ExperimentModel,
    SessionModel,
)
from backend.models.anomaly import AnomalyReport
from backend.models.dataset import DatasetProfile
from backend.models.experiment import ExperimentConfiguration, ExperimentResult
from backend.models.recommendation import SessionSummary

logger = logging.getLogger(__name__)

# Retry decorator applied to all public StateManager methods.
# Retries up to 3 times on transient connection failures, with exponential
# backoff starting at 1 second and capped at 10 seconds.
_retry_db = retry(
    retry=retry_if_exception_type(OperationalError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class StateManager:
    """PostgreSQL interface for all dataset, experiment, and session persistence.

    Parameters
    ----------
    database_url:
        SQLAlchemy-compatible connection string.  When ``None``, uses
        ``DATABASE_URL`` from ``backend.config`` (loaded from .env).
        Pass ``"sqlite:///:memory:"`` in unit tests.
    engine:
        Pre-built SQLAlchemy engine (test injection).  When provided,
        ``database_url`` is ignored.  Useful for injecting an in-memory
        SQLite engine without calling ``get_engine()``.
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        engine=None,
    ) -> None:
        if engine is not None:
            self._engine = engine
        else:
            self._engine = get_engine(database_url)

        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)
        logger.debug("StateManager initialised with engine: %s", self._engine.url)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _session_model_to_summary(
        row: SessionModel, experiment_count: int
    ) -> SessionSummary:
        return SessionSummary(
            session_id=row.session_id,
            research_question=row.research_question,
            status=row.status,
            cycle_count=row.cycle_count,
            experiment_count=experiment_count,
            created_at=row.created_at,
        )

    @staticmethod
    def _dataset_model_to_profile(row: DatasetModel) -> DatasetProfile:
        """Reconstruct a DatasetProfile from a DatasetModel ORM row.

        ``row.profile`` is the full JSONB-stored DatasetProfile dict; the
        surrounding columns (dataset_id, original_filename, storage_path,
        created_at) are also stored redundantly as their own columns for
        indexing/querying, but the profile dict is the source of truth for
        every field, so validating it directly reproduces the full model.
        """
        return DatasetProfile.model_validate(row.profile)

    @staticmethod
    def _experiment_model_to_result(row: ExperimentModel) -> ExperimentResult:
        """Convert an ExperimentModel ORM row to an ExperimentResult Pydantic model.

        JSONB round-trip: ``row.config`` is already a Python dict when
        returned by SQLAlchemy from PostgreSQL.  ``ExperimentConfiguration.
        model_validate(row.config)`` reconstructs the typed Pydantic object.
        """
        config = ExperimentConfiguration.model_validate(row.config)
        return ExperimentResult(
            experiment_id=row.experiment_id,
            session_id=row.session_id,
            config=config,
            task_type=row.task_type,  # type: ignore[arg-type]
            metrics=row.metrics,
            status=row.status,  # type: ignore[arg-type]
            error=row.error,
            timestamp=row.timestamp,
        )

    @staticmethod
    def _anomaly_model_to_report(row: AnomalyModel) -> AnomalyReport:
        return AnomalyReport(
            anomaly_id=row.anomaly_id,
            experiment_id=row.experiment_id,
            rule=row.rule,  # type: ignore[arg-type]
            explanation=row.explanation,
            severity=row.severity,  # type: ignore[arg-type]
            detected_at=row.detected_at,
        )

    # ------------------------------------------------------------------
    # Dataset operations
    # ------------------------------------------------------------------

    @_retry_db
    def create_dataset(self, profile: DatasetProfile) -> str:
        """Persist a DatasetProfile and return its dataset_id.

        The profile's own ``dataset_id`` (already assigned by
        ``ingest_csv``) is used as the primary key, so the caller's
        in-memory ``DatasetProfile`` and the persisted row always agree.

        Requirements: 1.7
        """
        row = DatasetModel(
            dataset_id=profile.dataset_id,
            original_filename=profile.original_filename,
            storage_path=profile.storage_path,
            profile=profile.model_dump(mode="json"),
        )
        with self._Session() as db:
            db.add(row)
            db.commit()
        logger.info("Dataset created: %s (%s)", profile.dataset_id, profile.original_filename)
        return profile.dataset_id

    @_retry_db
    def get_dataset(self, dataset_id: str) -> DatasetProfile:
        """Retrieve a dataset's profile by its UUID.

        Raises
        ------
        KeyError
            If no dataset with the given ``dataset_id`` exists.

        Requirements: 1.7
        """
        with self._Session() as db:
            row = db.get(DatasetModel, dataset_id)
        if row is None:
            raise KeyError(f"Dataset not found: {dataset_id!r}")
        return self._dataset_model_to_profile(row)

    @_retry_db
    def list_datasets(self) -> List[DatasetProfile]:
        """List all ingested datasets, newest first.

        Requirements: 1.7
        """
        with self._Session() as db:
            rows = db.query(DatasetModel).order_by(DatasetModel.created_at.desc()).all()
        return [self._dataset_model_to_profile(r) for r in rows]

    # ------------------------------------------------------------------
    # Session operations
    # ------------------------------------------------------------------

    @_retry_db
    def create_session(self, research_question: str, dataset_id: str) -> str:
        """Create a new research session scoped to a dataset, and return its session_id.

        Parameters
        ----------
        research_question:
            Natural language question driving the experiment loop.
        dataset_id:
            UUID of the dataset this session investigates. Every experiment
            run within this session references this same dataset.

        Returns
        -------
        str
            UUID of the newly created session.

        Requirements: 1.7, 8.1, 8.4
        """
        row = SessionModel(research_question=research_question, dataset_id=dataset_id)
        with self._Session() as db:
            db.add(row)
            db.commit()
            session_id = row.session_id
        logger.info("Session created: %s (dataset=%s)", session_id, dataset_id)
        return session_id

    @_retry_db
    def get_session(self, session_id: str) -> SessionModel:
        """Retrieve a session by its UUID.

        Parameters
        ----------
        session_id:
            UUID of the session to retrieve.

        Returns
        -------
        SessionModel
            The ORM row (detached; ``expire_on_commit=False`` keeps attrs alive).

        Raises
        ------
        KeyError
            If no session with the given ``session_id`` exists.

        Requirements: 8.3
        """
        with self._Session() as db:
            row = db.get(SessionModel, session_id)
        if row is None:
            raise KeyError(f"Session not found: {session_id!r}")
        return row

    @_retry_db
    def list_sessions(self) -> List[SessionSummary]:
        """List all sessions with their experiment counts.

        Returns sessions ordered by ``created_at`` descending (newest first).
        Experiment count is computed via a COUNT JOIN — it is NOT a database
        column.

        Returns
        -------
        List[SessionSummary]

        Requirements: 8.6
        """
        with self._Session() as db:
            rows = (
                db.query(
                    SessionModel,
                    func.count(ExperimentModel.experiment_id).label("exp_count"),
                )
                .outerjoin(
                    ExperimentModel,
                    ExperimentModel.session_id == SessionModel.session_id,
                )
                .group_by(SessionModel.session_id)
                .order_by(SessionModel.created_at.desc())
                .all()
            )

        return [
            self._session_model_to_summary(session_row, count)
            for session_row, count in rows
        ]

    @_retry_db
    def update_session_status(self, session_id: str, status: str) -> None:
        """Update the status of a session.

        Parameters
        ----------
        session_id:
            UUID of the session to update.
        status:
            New status string: ``"active"`` or ``"concluded"``.

        Raises
        ------
        KeyError
            If the session does not exist.

        Requirements: 8.4
        """
        with self._Session() as db:
            row = db.get(SessionModel, session_id)
            if row is None:
                raise KeyError(f"Session not found: {session_id!r}")
            row.status = status
            db.commit()
        logger.info("Session %s status -> %s", session_id, status)

    @_retry_db
    def update_session_node(
        self,
        session_id: str,
        current_node: str,
        cycle_count: Optional[int] = None,
        current_recommendation: Optional[str] = None,
    ) -> None:
        """Update session workflow state (used by Phase 5 LangGraph nodes).

        Parameters
        ----------
        session_id:
            UUID of the session to update.
        current_node:
            New LangGraph node name.
        cycle_count:
            If provided, overwrite the session's cycle counter.
        current_recommendation:
            If provided, replace the JSON blob recommendation.

        Requirements: 14.9
        """
        with self._Session() as db:
            row = db.get(SessionModel, session_id)
            if row is None:
                raise KeyError(f"Session not found: {session_id!r}")
            row.current_node = current_node
            if cycle_count is not None:
                row.cycle_count = cycle_count
            if current_recommendation is not None:
                row.current_recommendation = current_recommendation
            db.commit()

    # ------------------------------------------------------------------
    # Experiment operations
    # ------------------------------------------------------------------

    @_retry_db
    def store_experiment(self, experiment: ExperimentResult) -> None:
        """Persist a completed experiment to the database.

        The ``ExperimentConfiguration`` is serialised to a plain dict via
        ``model_dump()`` before storage (JSONB round-trip).

        Parameters
        ----------
        experiment:
            Completed experiment result to store.

        Requirements: 4.1, 4.2, 4.3
        """
        row = ExperimentModel(
            experiment_id=experiment.experiment_id,
            session_id=experiment.session_id,
            config=experiment.config.model_dump(mode="json"),  # JSONB round-trip
            task_type=experiment.task_type,
            metrics=experiment.metrics,
            status=experiment.status,
            error=experiment.error,
            timestamp=experiment.timestamp,
        )
        with self._Session() as db:
            db.add(row)
            db.commit()
        logger.debug(
            "Experiment stored: %s (status=%s)", experiment.experiment_id, experiment.status
        )

    @_retry_db
    def query_experiments(
        self,
        session_id: str,
        status: Optional[str] = None,
    ) -> List[ExperimentResult]:
        """Retrieve experiments for a session, optionally filtered by status.

        Results are ordered by ``timestamp`` ascending (oldest first).

        Parameters
        ----------
        session_id:
            UUID of the parent session.
        status:
            Optional filter: ``"success"``, ``"failed"``, ``"anomalous"``,
            ``"pending"``, or ``"running"``.

        Returns
        -------
        List[ExperimentResult]

        Requirements: 4.4, 4.5
        """
        with self._Session() as db:
            q = (
                db.query(ExperimentModel)
                .filter(ExperimentModel.session_id == session_id)
                .order_by(ExperimentModel.timestamp)
            )
            if status is not None:
                q = q.filter(ExperimentModel.status == status)
            rows = q.all()

        return [self._experiment_model_to_result(r) for r in rows]

    @_retry_db
    def get_experiment(self, experiment_id: str) -> ExperimentResult:
        """Retrieve a single experiment by its UUID.

        Parameters
        ----------
        experiment_id:
            UUID of the experiment.

        Returns
        -------
        ExperimentResult

        Raises
        ------
        KeyError
            If no experiment with the given ID exists.

        Requirements: 4.4
        """
        with self._Session() as db:
            row = db.get(ExperimentModel, experiment_id)
        if row is None:
            raise KeyError(f"Experiment not found: {experiment_id!r}")
        return self._experiment_model_to_result(row)

    @_retry_db
    def update_experiment_status(self, experiment_id: str, status: str) -> None:
        """Update the status of a single experiment (e.g., mark as 'anomalous').

        Parameters
        ----------
        experiment_id:
            UUID of the experiment to update.
        status:
            New status string.

        Raises
        ------
        KeyError
            If the experiment does not exist.

        Requirements: 5.5
        """
        with self._Session() as db:
            row = db.get(ExperimentModel, experiment_id)
            if row is None:
                raise KeyError(f"Experiment not found: {experiment_id!r}")
            row.status = status
            db.commit()
        logger.debug("Experiment %s status -> %s", experiment_id, status)

    # ------------------------------------------------------------------
    # Anomaly operations
    # ------------------------------------------------------------------

    @_retry_db
    def store_anomaly(self, anomaly: AnomalyReport) -> None:
        """Persist an anomaly report linked to an experiment.

        Parameters
        ----------
        anomaly:
            AnomalyReport produced by Anomaly_Detector.

        Requirements: 5.5
        """
        row = AnomalyModel(
            anomaly_id=anomaly.anomaly_id,
            experiment_id=anomaly.experiment_id,
            rule=anomaly.rule,
            explanation=anomaly.explanation,
            severity=anomaly.severity,
            detected_at=anomaly.detected_at,
        )
        with self._Session() as db:
            db.add(row)
            db.commit()
        logger.debug(
            "Anomaly stored: %s (rule=%s, experiment=%s)",
            anomaly.anomaly_id, anomaly.rule, anomaly.experiment_id,
        )

    @_retry_db
    def query_anomalies(
        self,
        experiment_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[AnomalyReport]:
        """Query anomalies by experiment or session.

        Exactly one of ``experiment_id`` or ``session_id`` should be provided.
        If both are given, ``experiment_id`` takes precedence.
        If neither is given, all anomalies are returned.

        When ``session_id`` is supplied the query joins
        ``anomalies -> experiments`` to find anomalies belonging to any
        experiment in that session.

        Parameters
        ----------
        experiment_id:
            Filter to anomalies for a specific experiment.
        session_id:
            Filter to anomalies across all experiments in a session.

        Returns
        -------
        List[AnomalyReport]

        Requirements: 5.5
        """
        with self._Session() as db:
            q = db.query(AnomalyModel)

            if experiment_id is not None:
                q = q.filter(AnomalyModel.experiment_id == experiment_id)
            elif session_id is not None:
                q = q.join(
                    ExperimentModel,
                    AnomalyModel.experiment_id == ExperimentModel.experiment_id,
                ).filter(ExperimentModel.session_id == session_id)

            rows = q.all()

        return [self._anomaly_model_to_report(r) for r in rows]
