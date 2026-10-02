"""
backend/database/models.py
===========================
SQLAlchemy ORM models for the three tables. This is the single schema
definition: ``init_db`` creates the tables from it.

    datasets     one row per ingested CSV (the file itself lives on disk)
    sessions     one investigation: question, status, plan, decisions, report
    experiments  one training run, with its per-row validation/test scores

Statistics are never stored - they are recomputed from ``experiments``.

JSON columns use JSONB on PostgreSQL and plain JSON on SQLite (tests).
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.types import JSON

UUID_TYPE = String(36).with_variant(PG_UUID(as_uuid=False), "postgresql")
JSON_TYPE = JSON().with_variant(JSONB, "postgresql")


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class DatasetModel(Base):
    __tablename__ = "datasets"

    dataset_id = Column(UUID_TYPE, primary_key=True, default=_uuid)
    original_filename = Column(Text, nullable=False)
    storage_path = Column(Text, nullable=False)
    profile = Column(JSON_TYPE, nullable=False)  # the full DatasetProfile
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    sessions = relationship("SessionModel", back_populates="dataset")


class SessionModel(Base):
    __tablename__ = "sessions"

    session_id = Column(UUID_TYPE, primary_key=True, default=_uuid)
    dataset_id = Column(UUID_TYPE, ForeignKey("datasets.dataset_id"), nullable=False, index=True)
    research_question = Column(Text, nullable=False)
    status = Column(String(16), nullable=False, default="pending")  # pending|running|done|failed
    error = Column(Text, nullable=True)
    plan = Column(JSON_TYPE, nullable=True)                   # Plan
    decisions = Column(JSON_TYPE, nullable=False, default=list)  # [Decision]
    report = Column(JSON_TYPE, nullable=True)                 # Report
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    dataset = relationship("DatasetModel", back_populates="sessions")
    experiments = relationship(
        "ExperimentModel", back_populates="session", cascade="all, delete-orphan"
    )


class ExperimentModel(Base):
    __tablename__ = "experiments"

    experiment_id = Column(UUID_TYPE, primary_key=True, default=_uuid)
    session_id = Column(
        UUID_TYPE, ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True
    )
    round = Column(Integer, nullable=False)
    config = Column(JSON_TYPE, nullable=False)       # ExperimentConfiguration
    status = Column(String(16), nullable=False)      # ok|failed
    error = Column(Text, nullable=True)
    metrics = Column(JSON_TYPE, nullable=True)
    val_scores = Column(JSON_TYPE, nullable=True)    # one score per validation row
    test_scores = Column(JSON_TYPE, nullable=True)   # one score per test row
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    session = relationship("SessionModel", back_populates="experiments")
