"""
backend/models/dataset.py
===========================
Pydantic v2 data models for user-uploaded datasets.

DatasetProfile
    The record produced by ingesting a CSV file (see
    ``backend/tools/dataset/ingestion.py``). Every ``ExperimentConfiguration``
    references a dataset by ``dataset_id``; every ``Session`` is scoped to
    exactly one dataset. This is the "the system inspects/understands the
    dataset" step of the dataset-first workflow.

PreprocessingConfig
    The one preprocessing choice exposed as an experiment variable
    (``normalize``). One-hot encoding of categorical columns is *not* part
    of this model — it is a fixed, structural step applied once at
    ingestion, not a per-experiment research variable.

DatasetValidationError
    Raised by ``ingest_csv`` when the uploaded file fails validation
    (missing target column, empty file, too few rows, single-valued
    target, etc.) — see ``backend/tools/dataset/ingestion.py`` for the
    exact checks.

Task type inference is a documented heuristic, not ground truth
------------------------------------------------------------------
A non-numeric target column, or a numeric target with a small number of
distinct integer-like values, is inferred as ``classification``; everything
else is inferred as ``regression``. A caller may override this via
``ingest_csv(..., task_type_override=...)`` — ``task_type_source`` records
whether the stored ``task_type`` was inferred or supplied, so a later
UI/planner layer can confirm or correct the heuristic without re-reading
the raw file.

Requirements
------------
1.3  Infer task_type using a documented heuristic
1.4  Allow a caller to override the inferred task_type
1.5  Compute a DatasetProfile (columns, task_type, class info, ...)
1.6  Compute a fixed train/val/test split seed per dataset
1.7  Persist DatasetProfile metadata in PostgreSQL; store the file on disk
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class DatasetValidationError(ValueError):
    """Raised when an uploaded CSV fails ingestion validation.

    Carries a specific, actionable reason (missing target column, empty
    file, too few rows, single-valued target, etc.) rather than a generic
    failure message.
    """


class PreprocessingConfig(BaseModel):
    """The one preprocessing choice exposed as an experiment variable.

    One-hot encoding of categorical columns is deliberately *not* here —
    it is a fixed, structural step applied once at ingestion (see
    ``backend/tools/dataset/preprocessing.py``), not a research variable.
    ``normalize`` directly answers research questions like "does
    normalization improve training stability?": when true, a
    ``StandardScaler`` is fit on the training split only and applied
    unchanged to validation/test (no leakage).
    """

    normalize: bool = False

    model_config = ConfigDict(json_schema_extra={"example": {"normalize": False}})


class DatasetProfile(BaseModel):
    """Result of ingesting and profiling a user-uploaded CSV dataset.

    Scoped intentionally: this carries what is operationally required to
    run experiments and what is directly useful to a future LLM planner
    describing the dataset in a prompt — not a general exploratory-analysis
    profiling framework (no per-column distributions/correlations).

    ORM round-trip
    --------------
    Stored as a single JSONB column (``datasets.profile``) by StateManager,
    the same pattern used for ``ExperimentConfiguration``.
    """

    dataset_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="UUID primary key",
    )
    original_filename: str = Field(description="Filename as uploaded by the user")
    storage_path: str = Field(
        description="Path to the canonical stored copy, e.g. data/uploads/<dataset_id>/data.csv"
    )
    target_column: str = Field(description="Name of the column being predicted")
    feature_columns: List[str] = Field(
        description="All columns used as model input (everything except target_column)"
    )
    numeric_columns: List[str] = Field(
        description="Feature columns treated as numeric"
    )
    categorical_columns: List[str] = Field(
        description="Feature columns one-hot encoded at ingestion"
    )
    task_type: Literal["classification", "regression"] = Field(
        description="Inferred (or overridden) prediction task type"
    )
    task_type_source: Literal["inferred", "user_specified"] = Field(
        default="inferred",
        description="Whether task_type came from the heuristic or a caller override",
    )
    n_rows: int = Field(description="Row count after dropping rows with missing values")
    n_features: int = Field(
        description="Feature count after one-hot encoding (the model's actual input_dim)"
    )
    n_classes: Optional[int] = Field(
        default=None, description="Number of distinct classes (classification only)"
    )
    class_labels: Optional[List[str]] = Field(
        default=None,
        description=(
            "Fixed label order for the target column (classification only). "
            "Index into this list is the integer class id used consistently "
            "by both `mlp` (CrossEntropyLoss needs integer targets) and "
            "`linear_baseline`."
        ),
    )
    class_distribution: Optional[Dict[str, int]] = Field(
        default=None,
        description="Raw row count per class label (classification only)",
    )
    missing_value_counts: Dict[str, int] = Field(
        description="Missing-value count per original column, before dropping rows"
    )
    split_seed: int = Field(
        description=(
            "Fixed seed for this dataset's train/val/test split. Independent "
            "of any experiment's random_seed, so experiments differing only "
            "in seed compare model/training randomness, not different rows "
            "landing in validation."
        )
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "dataset_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "original_filename": "churn.csv",
                "storage_path": "data/uploads/3fa85f64.../data.csv",
                "target_column": "churned",
                "feature_columns": ["tenure_months", "monthly_charges", "contract_type"],
                "numeric_columns": ["tenure_months", "monthly_charges"],
                "categorical_columns": ["contract_type"],
                "task_type": "classification",
                "task_type_source": "inferred",
                "n_rows": 4200,
                "n_features": 5,
                "n_classes": 2,
                "class_labels": ["no", "yes"],
                "class_distribution": {"no": 3100, "yes": 1100},
                "missing_value_counts": {"tenure_months": 3},
                "split_seed": 42,
                "created_at": "2024-01-15T10:00:00Z",
            }
        },
    )
