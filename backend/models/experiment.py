"""
backend/models/experiment.py
=============================
Pydantic v2 data models for experiment configurations and results.

ExperimentConfiguration
    Complete specification for one ML experiment run.
    Used as input to Phase 3 Experiment_Runner and stored as JSONB in PostgreSQL.

ExperimentResult
    Represents a *completed* experiment (status in success|failed|anomalous).
    Returned by Phase 3 Experiment_Runner, stored and retrieved by StateManager.

Notes on JSONB round-trip
--------------------------
``ExperimentConfiguration`` is serialised to a plain dict via
``config.model_dump()`` before being stored in the JSONB ``config`` column.
On retrieval from PostgreSQL the JSONB column is already a Python dict;
``ExperimentConfiguration.model_validate(row.config)`` reconstructs it.
``ExperimentResult`` uses ``ConfigDict(from_attributes=True)`` so it can also
be constructed directly from a SQLAlchemy ORM row.

Requirements
------------
1.4  Experiment_Planner_Agent specifies configurations explicitly
2.1  Experiment_Runner executes configurations
2.5  Experiment_Runner returns metrics including losses and accuracy
3.2  State_Manager persists experiment configurations and results
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# ExperimentConfiguration
# ---------------------------------------------------------------------------
class ExperimentConfiguration(BaseModel):
    """Complete specification for one ML experiment run.

    Supported model types
    ---------------------
    ``mnist_mlp``
        2-layer MLP (784->128->10) trained on MNIST.
        Hyperparameters: dropout (0-1), learning_rate (>0), batch_size (>0)

    ``synthetic_regression``
        Polynomial regression on a generated dataset.
        Hyperparameters: model_complexity (1-5), noise_level (>0)

    Example
    -------
    >>> cfg = ExperimentConfiguration(
    ...     model_type="mnist_mlp",
    ...     hyperparameters={"dropout": 0.2, "learning_rate": 0.001, "batch_size": 32},
    ...     random_seed=42,
    ... )
    """

    model_type: Literal["mnist_mlp", "synthetic_regression"]
    hyperparameters: Dict[str, float]
    random_seed: int

    model_config = ConfigDict(
        protected_namespaces=(),  # 'model_type' is a domain field, not a Pydantic namespace
        json_schema_extra={
            "example": {
                "model_type": "mnist_mlp",
                "hyperparameters": {
                    "dropout": 0.2,
                    "learning_rate": 0.001,
                    "batch_size": 32,
                },
                "random_seed": 42,
            }
        }
    )

    @model_validator(mode="after")
    def _validate_hyperparameters(self) -> "ExperimentConfiguration":
        """Validate hyperparameter ranges based on model_type."""
        hp = self.hyperparameters

        if self.model_type == "mnist_mlp":
            if "dropout" in hp:
                if not (0.0 <= hp["dropout"] <= 1.0):
                    raise ValueError(
                        f"dropout must be in [0, 1], got {hp['dropout']}"
                    )
            if "learning_rate" in hp:
                if hp["learning_rate"] <= 0:
                    raise ValueError(
                        f"learning_rate must be > 0, got {hp['learning_rate']}"
                    )
            if "batch_size" in hp:
                if hp["batch_size"] <= 0:
                    raise ValueError(
                        f"batch_size must be > 0, got {hp['batch_size']}"
                    )

        elif self.model_type == "synthetic_regression":
            if "model_complexity" in hp:
                if not (1 <= hp["model_complexity"] <= 5):
                    raise ValueError(
                        f"model_complexity must be in [1, 5], "
                        f"got {hp['model_complexity']}"
                    )
            if "noise_level" in hp:
                if hp["noise_level"] <= 0:
                    raise ValueError(
                        f"noise_level must be > 0, got {hp['noise_level']}"
                    )

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

    Metrics
    -------
    ``metrics`` is ``None`` when ``status == "failed"`` (training did not
    complete, so no metrics were recorded).  For ``success`` and ``anomalous``
    experiments the dict contains:
    - ``train_loss``             : final training loss
    - ``val_loss``               : final validation loss
    - ``accuracy``               : validation accuracy (MNIST) or 0.0 (regression)
    - ``training_time_seconds``  : wall-clock training time

    ORM round-trip
    --------------
    ``ConfigDict(from_attributes=True)`` allows constructing this model from
    a SQLAlchemy ORM row via::

        ExperimentResult.model_validate(orm_row, from_attributes=True)

    Requirements covered: 2.5, 3.2
    """

    experiment_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="UUID primary key",
    )
    session_id: str = Field(description="FK to sessions table")
    config: ExperimentConfiguration = Field(
        description="Full ExperimentConfiguration (stored as JSONB)"
    )
    metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Training metrics; None when status='failed'",
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
                    "model_type": "mnist_mlp",
                    "hyperparameters": {
                        "dropout": 0.2,
                        "learning_rate": 0.001,
                        "batch_size": 32,
                    },
                    "random_seed": 42,
                },
                "metrics": {
                    "train_loss": 0.15,
                    "val_loss": 0.18,
                    "accuracy": 0.94,
                    "training_time_seconds": 45.2,
                },
                "status": "success",
                "error": None,
                "timestamp": "2024-01-15T10:30:00Z",
            }
        },
    )
