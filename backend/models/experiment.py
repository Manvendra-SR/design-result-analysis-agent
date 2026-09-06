"""
backend/models/experiment.py
=============================
Pydantic v2 data models for experiment configurations and results.

ExperimentConfiguration
    Complete specification for one ML experiment run against a profiled
    Dataset. Used as input to Experiment_Runner and stored as JSONB in
    PostgreSQL.

ExperimentResult
    Represents a *completed* experiment (status in success|failed|anomalous).
    Returned by Experiment_Runner, stored and retrieved by StateManager.

Notes on JSONB round-trip
--------------------------
``ExperimentConfiguration`` is serialised to a plain dict via
``config.model_dump()`` before being stored in the JSONB ``config`` column.
On retrieval from PostgreSQL the JSONB column is already a Python dict;
``ExperimentConfiguration.model_validate(row.config)`` reconstructs it.
``ExperimentResult`` uses ``ConfigDict(from_attributes=True)`` so it can also
be constructed directly from a SQLAlchemy ORM row.

Dataset-first architecture revision
-------------------------------------
``model_type`` used to be ``Literal["mnist_mlp", "synthetic_regression"]`` -
two hardcoded, dataset-specific problems. It is now ``Literal["mlp",
"linear_baseline"]``: two model families that are dispatched by the
referenced dataset's *task type*, not by a fixed dataset identity. Every
configuration now carries ``dataset_id`` (see ``backend/models/dataset.py``)
and a ``preprocessing`` choice. See DESIGN_REVIEW_CHANGES.md's "Architecture
Revision" entry for the full rationale.

Requirements
------------
2.4  Experiment_Planner_Agent specifies dataset/model/hyperparameters/seed explicitly
3.1  Experiment_Runner executes configurations against a resolved Dataset
3.5  Experiment_Runner returns metrics keyed by task_type
4.3  State_Manager persists experiment configurations and results
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.models.dataset import PreprocessingConfig


# ---------------------------------------------------------------------------
# ExperimentConfiguration
# ---------------------------------------------------------------------------
class ExperimentConfiguration(BaseModel):
    """Complete specification for one ML experiment run against a dataset.

    Supported model types
    ---------------------
    ``mlp``
        Single-hidden-layer feed-forward network (``TabularMLP``), sized to
        the referenced dataset's feature/class counts.
        Hyperparameters: hidden_size (>0), dropout (0-1), learning_rate (>0),
        batch_size (>0), epochs (>0).

    ``linear_baseline``
        Logistic or linear regression (scikit-learn), chosen by the
        dataset's task type. No tunable hyperparameters.

    Example
    -------
    >>> cfg = ExperimentConfiguration(
    ...     dataset_id="3fa85f64-5717-4562-b3fc-2c963f66afa6",
    ...     model_type="mlp",
    ...     hyperparameters={"dropout": 0.2, "learning_rate": 0.001, "batch_size": 32},
    ...     random_seed=42,
    ... )
    """

    dataset_id: str
    model_type: Literal["mlp", "linear_baseline"]
    hyperparameters: Dict[str, float] = Field(default_factory=dict)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    random_seed: int

    model_config = ConfigDict(
        protected_namespaces=(),  # 'model_type' is a domain field, not a Pydantic namespace
        json_schema_extra={
            "example": {
                "dataset_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "model_type": "mlp",
                "hyperparameters": {
                    "dropout": 0.2,
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "hidden_size": 64,
                    "epochs": 20,
                },
                "preprocessing": {"normalize": False},
                "random_seed": 42,
            }
        }
    )

    @model_validator(mode="after")
    def _validate_hyperparameters(self) -> "ExperimentConfiguration":
        """Validate hyperparameter ranges based on model_type.

        ``linear_baseline`` has no tunable hyperparameters, so it has
        nothing to validate here - any values passed are simply ignored by
        the trainer.
        """
        hp = self.hyperparameters

        if self.model_type == "mlp":
            if "dropout" in hp and not (0.0 <= hp["dropout"] <= 1.0):
                raise ValueError(f"dropout must be in [0, 1], got {hp['dropout']}")
            if "learning_rate" in hp and hp["learning_rate"] <= 0:
                raise ValueError(
                    f"learning_rate must be > 0, got {hp['learning_rate']}"
                )
            if "batch_size" in hp and hp["batch_size"] <= 0:
                raise ValueError(f"batch_size must be > 0, got {hp['batch_size']}")
            if "hidden_size" in hp and hp["hidden_size"] <= 0:
                raise ValueError(f"hidden_size must be > 0, got {hp['hidden_size']}")
            if "epochs" in hp and hp["epochs"] <= 0:
                raise ValueError(f"epochs must be > 0, got {hp['epochs']}")

        return self


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------
class ExperimentResult(BaseModel):
    """Represents a *completed* experiment stored in PostgreSQL.

    This model covers the terminal states only:
    - ``success``  : training completed normally
    - ``failed``   : training raised an exception (``error`` is populated)
    - ``anomalous``: experiment was flagged by Anomaly_Detector after completion

    The intermediate states (``pending``, ``running``) exist only in the
    database and are managed internally by StateManager; they never appear
    in this Pydantic model.

    Metrics are keyed by task_type, not model_type
    -------------------------------------------------
    ``metrics`` is ``None`` when ``status == "failed"``. Otherwise its keys
    depend on ``task_type``, not on which model produced it - this is what
    makes ``mlp`` and ``linear_baseline`` results on the same dataset
    directly comparable:

    - ``classification``: ``train_loss``, ``val_loss``, ``accuracy``,
      ``n_classes``, ``n_val_samples``, ``training_time_seconds``
      (+ ``initial_train_loss`` for ``mlp`` only - no epoch loop for
      ``linear_baseline``)
    - ``regression``: ``train_loss``, ``val_loss``, ``training_time_seconds``
      - no ``accuracy`` key at all (replaces the old "accuracy=0.0 by
      convention" approach)

    ORM round-trip
    --------------
    ``ConfigDict(from_attributes=True)`` allows constructing this model from
    a SQLAlchemy ORM row via::

        ExperimentResult.model_validate(orm_row, from_attributes=True)

    Requirements covered: 3.5, 4.3
    """

    experiment_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="UUID primary key",
    )
    session_id: str = Field(description="FK to sessions table")
    config: ExperimentConfiguration = Field(
        description="Full ExperimentConfiguration (stored as JSONB)"
    )
    task_type: Optional[Literal["classification", "regression"]] = Field(
        default=None,
        description="The dataset's task type at the time this experiment ran",
    )
    metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Training metrics, keyed by task_type; None when status='failed'",
    )
    status: Literal["success", "failed", "anomalous"] = Field(
        description="Terminal experiment status"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message when status='failed'",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Experiment run timestamp (UTC)",
    )

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "experiment_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "session_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
                "config": {
                    "dataset_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "model_type": "mlp",
                    "hyperparameters": {
                        "dropout": 0.2,
                        "learning_rate": 0.001,
                        "batch_size": 32,
                        "hidden_size": 64,
                        "epochs": 20,
                    },
                    "preprocessing": {"normalize": False},
                    "random_seed": 42,
                },
                "task_type": "classification",
                "metrics": {
                    "train_loss": 0.15,
                    "val_loss": 0.18,
                    "accuracy": 0.94,
                    "n_classes": 2,
                    "n_val_samples": 120,
                    "training_time_seconds": 12.5,
                },
                "status": "success",
                "error": None,
                "timestamp": "2024-01-15T10:30:00Z",
            }
        },
    )
