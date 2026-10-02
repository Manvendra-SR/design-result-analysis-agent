"""
backend/database/repository.py
================================
``Repository``: every database read and write goes through here. It converts
between ORM rows and the Pydantic models, and holds no other logic.

Each method opens its own short-lived SQLAlchemy session. Pass ``engine=``
to use a different database (tests use in-memory SQLite).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from backend.database.connection import get_engine
from backend.database.models import DatasetModel, ExperimentModel, SessionModel
from backend.models.dataset import DatasetInUseError, DatasetProfile
from backend.models.experiment import ExperimentResult
from backend.models.investigation import Decision, Plan, Report, Session, SessionSummary


class Repository:
    def __init__(self, engine: Optional[Engine] = None) -> None:
        self._engine = engine
        self._factory: Optional[sessionmaker] = None

    def _db(self):
        if self._factory is None:  # connect lazily, so importing the app needs no database
            self._factory = sessionmaker(bind=self._engine or get_engine(), expire_on_commit=False)
        return self._factory()

    # ------------------------------------------------------------------ datasets
    def create_dataset(self, profile: DatasetProfile) -> None:
        with self._db() as db:
            db.add(
                DatasetModel(
                    dataset_id=profile.dataset_id,
                    original_filename=profile.original_filename,
                    storage_path=profile.storage_path,
                    profile=profile.model_dump(mode="json"),
                    created_at=profile.created_at,
                )
            )
            db.commit()

    def get_dataset(self, dataset_id: str) -> DatasetProfile:
        with self._db() as db:
            row = db.get(DatasetModel, dataset_id)
        if row is None:
            raise KeyError(f"Dataset not found: {dataset_id!r}")
        return DatasetProfile.model_validate(row.profile)

    def list_datasets(self) -> List[DatasetProfile]:
        with self._db() as db:
            rows = db.query(DatasetModel).order_by(DatasetModel.created_at.desc()).all()
        return [DatasetProfile.model_validate(r.profile) for r in rows]

    def delete_dataset(self, dataset_id: str, cascade: bool = False) -> Tuple[int, int]:
        """Delete a dataset row; returns (sessions_deleted, experiments_deleted).

        Refuses (``DatasetInUseError``) while investigations use it, unless
        ``cascade`` - deleting research history must be explicit.
        """
        with self._db() as db:
            row = db.get(DatasetModel, dataset_id)
            if row is None:
                raise KeyError(f"Dataset not found: {dataset_id!r}")
            sessions = list(row.sessions)
            if sessions and not cascade:
                raise DatasetInUseError(
                    f"Dataset is used by {len(sessions)} investigation(s). "
                    "Delete those first, or delete them together with the dataset."
                )
            experiments = sum(len(s.experiments) for s in sessions)
            for s in sessions:
                db.delete(s)
            db.delete(row)
            db.commit()
        return len(sessions), experiments

    # ------------------------------------------------------------------ sessions
    def create_session(self, research_question: str, dataset_id: str) -> Session:
        self.get_dataset(dataset_id)  # KeyError if it does not exist
        with self._db() as db:
            row = SessionModel(research_question=research_question, dataset_id=dataset_id, decisions=[])
            db.add(row)
            db.commit()
            return _to_session(row)

    def get_session(self, session_id: str) -> Session:
        with self._db() as db:
            row = db.get(SessionModel, session_id)
        if row is None:
            raise KeyError(f"Session not found: {session_id!r}")
        return _to_session(row)

    def list_sessions(self) -> List[SessionSummary]:
        with self._db() as db:
            rows = (
                db.query(SessionModel, func.count(ExperimentModel.experiment_id))
                .outerjoin(ExperimentModel)
                .group_by(SessionModel.session_id)
                .order_by(SessionModel.created_at.desc())
                .all()
            )
        return [
            SessionSummary(
                session_id=s.session_id,
                dataset_id=s.dataset_id,
                research_question=s.research_question,
                status=s.status,
                experiment_count=count,
                created_at=s.created_at,
            )
            for s, count in rows
        ]

    def delete_session(self, session_id: str) -> int:
        """Delete a session and its experiments; returns experiments deleted."""
        with self._db() as db:
            row = db.get(SessionModel, session_id)
            if row is None:
                raise KeyError(f"Session not found: {session_id!r}")
            n = len(row.experiments)
            db.delete(row)
            db.commit()
        return n

    def _update_session(self, session_id: str, **fields) -> None:
        with self._db() as db:
            row = db.get(SessionModel, session_id)
            if row is None:
                raise KeyError(f"Session not found: {session_id!r}")
            for key, value in fields.items():
                setattr(row, key, value)
            db.commit()

    def set_status(self, session_id: str, status: str, error: Optional[str] = None) -> None:
        self._update_session(session_id, status=status, error=error)

    def save_plan(self, session_id: str, plan: Plan) -> None:
        self._update_session(session_id, plan=plan.model_dump(mode="json"))

    def append_decision(self, session_id: str, decision: Decision) -> None:
        decisions = [d.model_dump(mode="json") for d in self.get_session(session_id).decisions]
        self._update_session(session_id, decisions=[*decisions, decision.model_dump(mode="json")])

    def save_report(self, session_id: str, report: Report) -> None:
        self._update_session(session_id, report=report.model_dump(mode="json"))

    def reset_session(self, session_id: str) -> None:
        """Discard a previous (failed) run's results so the investigation can start over."""
        with self._db() as db:
            db.query(ExperimentModel).filter(ExperimentModel.session_id == session_id).delete()
            db.commit()
        self._update_session(session_id, status="pending", error=None, plan=None, decisions=[], report=None)

    def fail_interrupted_runs(self) -> int:
        """Mark runs left 'running' by a server restart as failed; returns how many."""
        with self._db() as db:
            n = (
                db.query(SessionModel)
                .filter(SessionModel.status == "running")
                .update({"status": "failed", "error": "Interrupted: the server restarted during the run."})
            )
            db.commit()
        return n

    # --------------------------------------------------------------- experiments
    def add_experiment(self, result: ExperimentResult) -> None:
        with self._db() as db:
            db.add(
                ExperimentModel(
                    experiment_id=result.experiment_id,
                    session_id=result.session_id,
                    round=result.round,
                    config=result.config.model_dump(mode="json"),
                    status=result.status,
                    error=result.error,
                    metrics=result.metrics,
                    val_scores=result.val_scores,
                    test_scores=result.test_scores,
                    created_at=result.created_at,
                )
            )
            db.commit()

    def list_experiments(self, session_id: str) -> List[ExperimentResult]:
        with self._db() as db:
            rows = (
                db.query(ExperimentModel)
                .filter(ExperimentModel.session_id == session_id)
                .order_by(ExperimentModel.created_at)
                .all()
            )
        return [
            ExperimentResult(
                experiment_id=r.experiment_id,
                session_id=r.session_id,
                round=r.round,
                config=r.config,
                status=r.status,
                error=r.error,
                metrics=r.metrics,
                val_scores=r.val_scores,
                test_scores=r.test_scores,
                created_at=r.created_at,
            )
            for r in rows
        ]


def _to_session(row: SessionModel) -> Session:
    return Session(
        session_id=row.session_id,
        dataset_id=row.dataset_id,
        research_question=row.research_question,
        status=row.status,
        error=row.error,
        plan=row.plan,
        decisions=row.decisions or [],
        report=row.report,
        created_at=row.created_at,
    )
