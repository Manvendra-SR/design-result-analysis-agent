"""
tests/unit/test_dataset_ingestion.py
=======================================
Unit tests for backend/tools/dataset/ingestion.py.

Fixtures are small, fast, in-memory CSVs built with
sklearn.datasets.make_classification/make_regression and written to a
pytest tmp_path - no MNIST download, no live dependency, matching the
existing "no live dependency in unit tests" convention.

Test cases
----------
1. test_ingest_csv_classification_profile_fields
2. test_ingest_csv_regression_profile_fields
3. test_ingest_csv_missing_target_column_raises
4. test_ingest_csv_empty_file_raises
5. test_ingest_csv_too_few_rows_raises
6. test_ingest_csv_single_valued_target_raises
7. test_ingest_csv_infers_classification_for_categorical_target
8. test_ingest_csv_task_type_override_sets_source_and_type
9. test_ingest_csv_invalid_override_raises
10. test_ingest_csv_regression_override_rejects_non_numeric_target
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from backend.models.dataset import DatasetValidationError
from backend.tools.dataset.ingestion import ingest_csv


def _classification_csv(path: Path, n_samples: int = 60) -> Path:
    x, y = make_classification(
        n_samples=n_samples, n_features=4, n_informative=3, n_redundant=0,
        n_classes=2, random_state=0,
    )
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(4)])
    df["target"] = y
    file_path = path / "classification.csv"
    df.to_csv(file_path, index=False)
    return file_path


def _regression_csv(path: Path, n_samples: int = 60) -> Path:
    x, y = make_regression(n_samples=n_samples, n_features=3, noise=1.0, random_state=0)
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(3)])
    df["target"] = y
    file_path = path / "regression.csv"
    df.to_csv(file_path, index=False)
    return file_path


# ---------------------------------------------------------------------------
# 1-2. Valid CSV -> correct profile fields
# ---------------------------------------------------------------------------

def test_ingest_csv_classification_profile_fields(tmp_path: Path) -> None:
    csv_path = _classification_csv(tmp_path)

    profile = ingest_csv(str(csv_path), target_column="target")

    assert profile.task_type == "classification"
    assert profile.task_type_source == "inferred"
    assert profile.target_column == "target"
    assert set(profile.feature_columns) == {"f0", "f1", "f2", "f3"}
    assert profile.numeric_columns == profile.feature_columns
    assert profile.categorical_columns == []
    assert profile.n_classes == 2
    assert profile.class_labels == ["0", "1"]
    assert profile.n_rows == 60
    assert profile.n_features == 4
    assert profile.split_seed == 42
    assert Path(profile.storage_path).exists()


def test_ingest_csv_regression_profile_fields(tmp_path: Path) -> None:
    csv_path = _regression_csv(tmp_path)

    profile = ingest_csv(str(csv_path), target_column="target")

    assert profile.task_type == "regression"
    assert profile.n_classes is None
    assert profile.class_labels is None
    assert profile.n_rows == 60


# ---------------------------------------------------------------------------
# 3-6. Validation failures
# ---------------------------------------------------------------------------

def test_ingest_csv_missing_target_column_raises(tmp_path: Path) -> None:
    csv_path = _classification_csv(tmp_path)

    with pytest.raises(DatasetValidationError, match="not found"):
        ingest_csv(str(csv_path), target_column="does_not_exist")


def test_ingest_csv_empty_file_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("")

    with pytest.raises(DatasetValidationError):
        ingest_csv(str(csv_path), target_column="target")


def test_ingest_csv_too_few_rows_raises(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    csv_path = tmp_path / "tiny.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(DatasetValidationError, match="rows"):
        ingest_csv(str(csv_path), target_column="target")


def test_ingest_csv_single_valued_target_raises(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": np.arange(30), "target": [1] * 30})
    csv_path = tmp_path / "constant_target.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(DatasetValidationError, match="distinct values"):
        ingest_csv(str(csv_path), target_column="target")


# ---------------------------------------------------------------------------
# 7. Categorical (string) target -> inferred classification
# ---------------------------------------------------------------------------

def test_ingest_csv_infers_classification_for_categorical_target(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "a": np.random.default_rng(0).normal(size=40),
            "target": (["yes"] * 20) + (["no"] * 20),
        }
    )
    csv_path = tmp_path / "categorical_target.csv"
    df.to_csv(csv_path, index=False)

    profile = ingest_csv(str(csv_path), target_column="target")

    assert profile.task_type == "classification"
    assert profile.task_type_source == "inferred"
    assert profile.class_labels == ["no", "yes"]
    assert profile.class_distribution == {"no": 20, "yes": 20}


# ---------------------------------------------------------------------------
# 8-10. task_type_override
# ---------------------------------------------------------------------------

def test_ingest_csv_task_type_override_sets_source_and_type(tmp_path: Path) -> None:
    # A numeric target that would normally infer as classification (few,
    # integer-like unique values) can be forced to regression instead.
    df = pd.DataFrame(
        {"a": np.arange(30, dtype=float), "target": np.arange(30, dtype=float)}
    )
    csv_path = tmp_path / "override.csv"
    df.to_csv(csv_path, index=False)

    profile = ingest_csv(str(csv_path), target_column="target", task_type_override="regression")

    assert profile.task_type == "regression"
    assert profile.task_type_source == "user_specified"


def test_ingest_csv_invalid_override_raises(tmp_path: Path) -> None:
    csv_path = _classification_csv(tmp_path)

    with pytest.raises(DatasetValidationError):
        ingest_csv(str(csv_path), target_column="target", task_type_override="not_a_real_type")


def test_ingest_csv_regression_override_rejects_non_numeric_target(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "a": np.random.default_rng(0).normal(size=30),
            "target": (["yes"] * 15) + (["no"] * 15),
        }
    )
    csv_path = tmp_path / "bad_override.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(DatasetValidationError, match="numeric"):
        ingest_csv(str(csv_path), target_column="target", task_type_override="regression")
