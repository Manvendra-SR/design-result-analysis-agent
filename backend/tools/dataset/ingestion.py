"""
backend/tools/dataset/ingestion.py
=====================================
CSV ingestion: validates an uploaded file, infers task type, computes a
DatasetProfile, and stores a canonical copy on disk.

Validation, not guessing
-------------------------
The caller must name the target column explicitly - this module never
guesses it (e.g. "last column"). Task type IS inferred by a documented
heuristic (see ``_infer_task_type``), but a caller may override it via
``task_type_override``; ``DatasetProfile.task_type_source`` records which
happened, so a later UI/planner layer can confirm or correct the heuristic.

What this does NOT do
----------------------
No missing-value imputation (rows with any missing value are dropped), no
feature engineering, no feature selection, no class balancing. This is a
deliberate scope boundary (Requirement 1.9), not an oversight.

Requirements
------------
1.1  Validate the file before accepting it as a Dataset
1.2  Return specific, actionable errors on invalid input
1.3  Infer task_type via a documented heuristic
1.4  Allow a caller to override the inferred task_type
1.5  Compute a DatasetProfile (columns, task_type, class info, ...)
1.6  Compute a fixed split_seed per dataset
1.7  Store the dataset file on disk (metadata persistence is the caller's
     job, via StateManager.create_dataset)
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import pandas as pd

from backend.models.dataset import DatasetProfile, DatasetValidationError

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_UPLOADS_DIR = _PROJECT_ROOT / "data" / "uploads"

_MIN_ROWS = 20
_MAX_CLASSIFICATION_UNIQUE_VALUES = 20
# Fixed for every dataset - simpler than generating a per-dataset random
# seed, and just as effective: what matters is that the split doesn't
# change across experiments on the SAME dataset (see DatasetProfile.split_seed).
_DEFAULT_SPLIT_SEED = 42

_VALID_TASK_TYPES = ("classification", "regression")


def _infer_task_type(target: pd.Series) -> str:
    """Heuristic: non-numeric target, or numeric with a small number of
    distinct integer-like values, -> classification; otherwise regression.
    """
    if not pd.api.types.is_numeric_dtype(target):
        return "classification"

    unique_values = target.dropna().unique()
    if len(unique_values) == 0:
        return "regression"
    if len(unique_values) <= _MAX_CLASSIFICATION_UNIQUE_VALUES and all(
        float(v).is_integer() for v in unique_values
    ):
        return "classification"
    return "regression"


def ingest_csv(
    file_path: str,
    target_column: str,
    task_type_override: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> DatasetProfile:
    """Validate, profile, and store a copy of an uploaded CSV dataset.

    Parameters
    ----------
    file_path:
        Path to the uploaded CSV file.
    target_column:
        Name of the column to predict. Required - never guessed.
    task_type_override:
        If given, forces ``"classification"`` or ``"regression"`` instead
        of using the inferred heuristic.
    dataset_name:
        Optional display name; defaults to the source file's name.

    Returns
    -------
    DatasetProfile

    Raises
    ------
    DatasetValidationError
        On any validation failure (unreadable file, missing target column,
        too few usable rows, single-valued target, invalid override, etc.)

    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
    """
    if task_type_override is not None and task_type_override not in _VALID_TASK_TYPES:
        raise DatasetValidationError(
            f"task_type_override must be one of {_VALID_TASK_TYPES}, "
            f"got {task_type_override!r}"
        )

    source = Path(file_path)
    if not source.exists():
        raise DatasetValidationError(f"File not found: {file_path}")

    try:
        df = pd.read_csv(source)
    except Exception as exc:  # noqa: BLE001 - surfaced as a validation error
        raise DatasetValidationError(
            f"Could not parse {file_path!r} as CSV: {exc}"
        ) from exc

    if df.empty or df.shape[1] == 0:
        raise DatasetValidationError("Uploaded CSV is empty.")

    if len(df.columns) != len(set(df.columns)):
        raise DatasetValidationError("Uploaded CSV has duplicate column names.")

    if target_column not in df.columns:
        raise DatasetValidationError(
            f"Target column {target_column!r} not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Missing-value handling: drop rows with any missing value (documented
    # limitation - no imputation). Counts are captured before dropping.
    missing_value_counts = {k: int(v) for k, v in df.isna().sum().to_dict().items()}
    df = df.dropna().reset_index(drop=True)

    if len(df) < _MIN_ROWS:
        raise DatasetValidationError(
            f"Dataset has only {len(df)} usable rows after dropping rows "
            f"with missing values; at least {_MIN_ROWS} are required."
        )

    target = df[target_column]
    if target.nunique(dropna=True) < 2:
        raise DatasetValidationError(
            f"Target column {target_column!r} has fewer than 2 distinct "
            "values; there is no signal to learn."
        )

    if task_type_override is not None:
        if task_type_override == "regression" and not pd.api.types.is_numeric_dtype(target):
            raise DatasetValidationError(
                f"task_type_override='regression' requires a numeric target "
                f"column; {target_column!r} is not numeric."
            )
        task_type = task_type_override
        task_type_source = "user_specified"
    else:
        task_type = _infer_task_type(target)
        task_type_source = "inferred"

    feature_columns = [c for c in df.columns if c != target_column]
    if not feature_columns:
        raise DatasetValidationError("Dataset has no feature columns besides the target.")

    numeric_columns = [c for c in feature_columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_columns = [c for c in feature_columns if c not in numeric_columns]

    n_classes = None
    class_labels = None
    class_distribution = None
    if task_type == "classification":
        # Fixed label order (sorted by string form) for a deterministic
        # label<->index mapping used consistently by mlp and linear_baseline.
        class_labels = sorted(target.astype(str).unique().tolist())
        n_classes = len(class_labels)
        counts = target.astype(str).value_counts()
        class_distribution = {label: int(counts.get(label, 0)) for label in class_labels}

    # n_features reflects the model's actual input_dim: feature count AFTER
    # one-hot encoding, computed the same way backend/tools/dataset/dataset.py
    # will encode it later, so the two stay consistent.
    if categorical_columns:
        n_features = pd.get_dummies(
            df[feature_columns], columns=categorical_columns, drop_first=False
        ).shape[1]
    else:
        n_features = len(feature_columns)

    dataset_id = str(uuid.uuid4())
    storage_dir = _UPLOADS_DIR / dataset_id
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_path = storage_dir / "data.csv"
    # Store the cleaned (missing-value-dropped) data so later loads via
    # Dataset.load_split stay consistent with the profile computed here.
    df.to_csv(storage_path, index=False)

    return DatasetProfile(
        dataset_id=dataset_id,
        original_filename=dataset_name or source.name,
        storage_path=str(storage_path),
        target_column=target_column,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        task_type=task_type,
        task_type_source=task_type_source,
        n_rows=len(df),
        n_features=n_features,
        n_classes=n_classes,
        class_labels=class_labels,
        class_distribution=class_distribution,
        missing_value_counts=missing_value_counts,
        split_seed=_DEFAULT_SPLIT_SEED,
    )
