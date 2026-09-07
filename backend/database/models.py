"""
backend/database/models.py
===========================
SQLAlchemy ORM models mapping Python classes to the 4 database tables.

Tables
------
DatasetModel    -> datasets    (added by the dataset-first architecture revision)
SessionModel    -> sessions
ExperimentModel -> experiments
AnomalyModel    -> anomalies

Relationships
-------------
Session  *  -- 1 Dataset      (a session is scoped to exactly one dataset)
Session  1 -- * Experiment  (cascade delete)
Experiment 1 -- * Anomaly   (cascade delete)

Usage
-----
These models are used by StateManager (Phase 2) for all database reads/writes.
They should not be imported directly by business-logic layers - use
StateManager instead.

Notes on JSONB and UUID
-----------------------
JSONB and UUID columns use PostgreSQL native types on PostgreSQL, with
portable JSON and String(36) fallbacks for in-memory SQLite during unit tests.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.types import JSON

# Portable column types: native PostgreSQL types when on PG, portable types on SQLite
UUID_TYPE = String(36).with_variant(PG_UUID(as_uuid=False), "postgresql")
JSON_TYPE = JSON().with_variant(JSONB, "postgresql")


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


# ---------------------------------------------------------------------------
# DatasetModel
# ---------------------------------------------------------------------------
class DatasetModel(Base):
    """
    ORM representation of the ``datasets`` table.

    One row per user-uploaded, ingested CSV dataset. ``profile`` (JSONB)
    stores the complete ``DatasetProfile`` (task_type, columns, class info,
    split_seed, ...) - the underlying CSV file itself lives on disk under
    ``data/uploads/<dataset_id>/`` and is never stored in the database.
    """

    __tablename__ = "datasets"

    dataset_id: str = Column(
        UUID_TYPE,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="UUID primary key - unique dataset identifier",
    )
    original_filename: str = Column(
        Text, nullable=False, comment="Filename as uploaded by the user"
    )
    storage_path: str = Column(
        Text,
        nullable=False,
        comment="Path to the canonical stored CSV copy on disk",
    )
    profile: dict = Column(
        JSON_TYPE,
        nullable=False,
        comment="Full DatasetProfile as JSON (task_type, columns, class info, split_seed, ...)",
    )
    created_at: datetime = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        server_default=func.now(),
        comment="Dataset ingestion timestamp (UTC)",
    )

    # Relationship: dataset is referenced by many sessions (no cascade -
    # deleting a dataset while sessions reference it should fail loudly,
    # not silently orphan/cascade-delete research history).
    sessions = relationship("SessionModel", back_populates="dataset")

    def __repr__(self) -> str:
        return f"<DatasetModel dataset_id={self.dataset_id!r} filename={self.original_filename!r}>"


# ---------------------------------------------------------------------------
# SessionModel
# ---------------------------------------------------------------------------
class SessionModel(Base):
    """
    ORM representation of the ``sessions`` table.

    One row per research investigation, scoped to exactly one Dataset
    (``dataset_id``). Tracks the current LangGraph node (``current_node``)
    for crash recovery and stores the latest recommendation as a JSON blob
    (``current_recommendation``).
    """

    __tablename__ = "sessions"

    session_id: str = Column(
        UUID_TYPE,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="UUID primary key - unique research session identifier",
    )
    dataset_id: str = Column(
        UUID_TYPE,
        ForeignKey("datasets.dataset_id"),
        nullable=False,
        index=True,
        comment="Foreign key to datasets table - the dataset this session investigates",
    )
    research_question: str = Column(
        Text,
        nullable=False,
        comment="Natural language research question driving the experiment loop",
    )
    status: str = Column(
        String(20),
        nullable=False,
        default="active",
        comment="'active' | 'concluded'",
    )
    current_node: str = Column(
        String(20),
        nullable=False,
        default="planning",
        comment=(
            "Current LangGraph node - persisted for crash recovery. "
            "Values: 'planning' | 'executing' | 'validating' | "
            "'analyzing' | 'recommending' | 'concluded'"
        ),
    )
    current_recommendation: str | None = Column(
        Text,
        nullable=True,
        default=None,
        comment="JSON blob of the latest Recommendation; overwritten each cycle",
    )
    pending_configs: str | None = Column(
        Text,
        nullable=True,
        default=None,
        comment=(
            "Phase 5 scratch: JSON list of ExperimentConfigurations the "
            "planning/recommendation node queued for the next execution node"
        ),
    )
    latest_analysis: str | None = Column(
        Text,
        nullable=True,
        default=None,
        comment=(
            "Phase 5 scratch: JSON list of StatisticalComparisons the "
            "analysis node computed for the recommendation node"
        ),
    )
    plan_explanation: str | None = Column(
        Text,
        nullable=True,
        default=None,
        comment="Phase 5 history: the planner's rationale for the cycle-1 design",
    )
    cycle_history: str | None = Column(
        Text,
        nullable=True,
        default=None,
        comment=(
            "Phase 5 history: JSON array of CycleHistoryEntry (per-cycle "
            "recommendation + statistical comparisons), appended each cycle"
        ),
    )
    cycle_count: int = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of completed adaptive cycles",
    )
    created_at: datetime = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        server_default=func.now(),
        comment="Session creation timestamp (UTC)",
    )
    updated_at: datetime = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        server_default=func.now(),
        comment="Last update timestamp (UTC)",
    )

    # Relationship: session owns many experiments (cascade delete)
    experiments = relationship(
        "ExperimentModel",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ExperimentModel.timestamp",
    )
    # Relationship: session belongs to exactly one dataset (no cascade)
    dataset = relationship("DatasetModel", back_populates="sessions")

    def __repr__(self) -> str:
        return (
            f"<SessionModel session_id={self.session_id!r} "
            f"dataset_id={self.dataset_id!r} status={self.status!r} "
            f"current_node={self.current_node!r}>"
        )


# ---------------------------------------------------------------------------
# ExperimentModel
# ---------------------------------------------------------------------------
class ExperimentModel(Base):
    """
    ORM representation of the ``experiments`` table.

    One row per individual ML experiment run.

    ``config`` (JSONB): complete ExperimentConfiguration
        e.g. {"dataset_id": "3fa85f64-...", "model_type": "mlp",
               "hyperparameters": {"dropout": 0.2, "learning_rate": 0.001,
                                   "batch_size": 32, "hidden_size": 64, "epochs": 20},
               "preprocessing": {"normalize": False},
               "random_seed": 42}

    ``metrics`` (JSONB): final training metrics, keyed by task_type (NULL while pending)
        e.g. classification: {"train_loss": 0.15, "val_loss": 0.18,
               "accuracy": 0.94, "n_classes": 2, "n_val_samples": 120,
               "training_time_seconds": 12.5}

    ``status``: 'pending' | 'running' | 'success' | 'failed' | 'anomalous'
    """

    __tablename__ = "experiments"

    experiment_id: str = Column(
        UUID_TYPE,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="UUID primary key - unique experiment identifier",
    )
    session_id: str = Column(
        UUID_TYPE,
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to sessions table",
    )
    config: dict = Column(
        JSON_TYPE,
        nullable=False,
        comment="ExperimentConfiguration as JSON (dataset_id, model_type, hyperparameters, preprocessing, random_seed)",
    )
    task_type: str | None = Column(
        String(20),
        nullable=True,
        default=None,
        comment="'classification' | 'regression' - the dataset's task type when this experiment ran",
    )
    metrics: dict | None = Column(
        JSON_TYPE,
        nullable=True,
        default=None,
        comment="Result metrics as JSON, keyed by task_type (see ExperimentResult docstring)",
    )
    status: str = Column(
        String(20),
        nullable=False,
        default="pending",
        comment="'pending' | 'running' | 'success' | 'failed' | 'anomalous'",
    )
    error: str | None = Column(
        Text,
        nullable=True,
        default=None,
        comment="Error message when status='failed'",
    )
    cycle: int | None = Column(
        Integer,
        nullable=True,
        default=None,
        comment="1-based adaptive cycle that produced this experiment (set by the execution node)",
    )
    timestamp: datetime = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        server_default=func.now(),
        comment="Experiment run timestamp (UTC)",
    )

    # Relationships
    session = relationship("SessionModel", back_populates="experiments")
    anomalies = relationship(
        "AnomalyModel",
        back_populates="experiment",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            f"<ExperimentModel experiment_id={self.experiment_id!r} "
            f"status={self.status!r}>"
        )


# ---------------------------------------------------------------------------
# AnomalyModel
# ---------------------------------------------------------------------------
class AnomalyModel(Base):
    """
    ORM representation of the ``anomalies`` table.

    One row per detected anomaly.  Created by Anomaly_Detector (Phase 3).
    Explanations are template-generated - no LLM calls for anomaly text.

    ``rule``: detection rule that fired
        'outlier_detection' | 'loss_divergence' | 'validation_collapse'

    ``severity``: 'warning' | 'critical'
        validation_collapse -> 'critical', others -> 'warning'
    """

    __tablename__ = "anomalies"

    anomaly_id: str = Column(
        UUID_TYPE,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="UUID primary key - unique anomaly identifier",
    )
    experiment_id: str = Column(
        UUID_TYPE,
        ForeignKey("experiments.experiment_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to experiments table",
    )
    rule: str = Column(
        String(50),
        nullable=False,
        comment="'outlier_detection' | 'loss_divergence' | 'validation_collapse'",
    )
    explanation: str = Column(
        Text,
        nullable=False,
        comment="Template-generated natural language explanation (no LLM call)",
    )
    severity: str = Column(
        String(20),
        nullable=False,
        comment="'warning' | 'critical'",
    )
    detected_at: datetime = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        server_default=func.now(),
        comment="Anomaly detection timestamp (UTC)",
    )

    # Relationship
    experiment = relationship("ExperimentModel", back_populates="anomalies")

    def __repr__(self) -> str:
        return (
            f"<AnomalyModel anomaly_id={self.anomaly_id!r} "
            f"rule={self.rule!r} severity={self.severity!r}>"
        )