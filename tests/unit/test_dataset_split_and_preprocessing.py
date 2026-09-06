"""
tests/unit/test_dataset_split_and_preprocessing.py
======================================================
Unit tests for backend/tools/dataset/splitting.py, preprocessing.py, and
the Dataset facade (dataset.py).

Test cases
----------
1. test_compute_split_is_deterministic
2. test_compute_split_covers_all_rows_without_overlap
3. test_load_split_deterministic_across_calls
4. test_load_split_normalize_changes_scale_not_row_count
5. test_load_split_one_hot_schema_is_stable
6. test_normalization_fit_only_on_train_no_leakage
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.tools.dataset.ingestion import ingest_csv
from backend.tools.dataset.dataset import Dataset
from backend.tools.dataset.preprocessing import apply_normalization
from backend.tools.dataset.splitting import compute_split


# ---------------------------------------------------------------------------
# 1-2. splitting.compute_split
# ---------------------------------------------------------------------------

def test_compute_split_is_deterministic() -> None:
    first = compute_split(n_rows=100, split_seed=42)
    second = compute_split(n_rows=100, split_seed=42)

    assert list(first.train) == list(second.train)
    assert list(first.val) == list(second.val)
    assert list(first.test) == list(second.test)


def test_compute_split_covers_all_rows_without_overlap() -> None:
    split = compute_split(n_rows=100, split_seed=42)

    all_indices = np.concatenate([split.train, split.val, split.test])
    assert sorted(all_indices) == list(range(100))
    assert len(set(split.train) & set(split.val)) == 0
    assert len(set(split.train) & set(split.test)) == 0
    assert len(set(split.val) & set(split.test)) == 0
    # Roughly 70/15/15
    assert 65 <= len(split.train) <= 75
    assert 10 <= len(split.val) <= 20
    assert 10 <= len(split.test) <= 20


# ---------------------------------------------------------------------------
# Shared fixture: a small ingested classification dataset with one
# categorical column, for the Dataset-facade tests below.
# ---------------------------------------------------------------------------

def _ingest_mixed_dataset(tmp_path: Path):
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame(
        {
            "num1": rng.normal(loc=10, scale=3, size=n),
            "num2": rng.normal(loc=-5, scale=1, size=n),
            "cat1": rng.choice(["red", "green", "blue"], size=n),
            "target": rng.choice([0, 1], size=n),
        }
    )
    csv_path = tmp_path / "mixed.csv"
    df.to_csv(csv_path, index=False)
    return ingest_csv(str(csv_path), target_column="target")


# ---------------------------------------------------------------------------
# 3. Dataset.load_split is deterministic
# ---------------------------------------------------------------------------

def test_load_split_deterministic_across_calls(tmp_path: Path) -> None:
    profile = _ingest_mixed_dataset(tmp_path)
    dataset = Dataset(profile)

    first = dataset.load_split(normalize=False)
    second = dataset.load_split(normalize=False)

    np.testing.assert_array_equal(first.x_train, second.x_train)
    np.testing.assert_array_equal(first.y_train, second.y_train)
    np.testing.assert_array_equal(first.x_val, second.x_val)
    np.testing.assert_array_equal(first.x_test, second.x_test)


# ---------------------------------------------------------------------------
# 4. normalize changes scale but not row count/order
# ---------------------------------------------------------------------------

def test_load_split_normalize_changes_scale_not_row_count(tmp_path: Path) -> None:
    profile = _ingest_mixed_dataset(tmp_path)
    dataset = Dataset(profile)

    raw = dataset.load_split(normalize=False)
    normalized = dataset.load_split(normalize=True)

    assert raw.x_train.shape == normalized.x_train.shape
    assert raw.x_val.shape == normalized.x_val.shape
    np.testing.assert_array_equal(raw.y_train, normalized.y_train)

    # Normalized training features should be ~standardized.
    assert np.abs(normalized.x_train.mean(axis=0)).max() < 1e-6
    assert np.abs(normalized.x_train.std(axis=0) - 1.0).max() < 1e-6
    # Raw features should NOT already be standardized (sanity check the
    # comparison is meaningful).
    assert np.abs(raw.x_train.mean(axis=0)).max() > 1e-6


# ---------------------------------------------------------------------------
# 5. One-hot schema is stable (same column count every load)
# ---------------------------------------------------------------------------

def test_load_split_one_hot_schema_is_stable(tmp_path: Path) -> None:
    profile = _ingest_mixed_dataset(tmp_path)
    dataset = Dataset(profile)

    first = dataset.load_split(normalize=False)
    second = dataset.load_split(normalize=False)

    assert first.x_train.shape[1] == second.x_train.shape[1] == profile.n_features
    # 2 numeric + 3 one-hot categories (red/green/blue) = 5 columns
    assert profile.n_features == 5


# ---------------------------------------------------------------------------
# 6. Leakage guard: StandardScaler is fit on train only
# ---------------------------------------------------------------------------

def test_normalization_fit_only_on_train_no_leakage() -> None:
    rng = np.random.default_rng(1)
    x_train = rng.normal(loc=0.0, scale=1.0, size=(50, 2))
    # val/test drawn from a very different distribution - if the scaler
    # were (incorrectly) fit on train+val+test combined, the train split's
    # normalized mean/std would be pulled away from 0/1.
    x_val = rng.normal(loc=100.0, scale=50.0, size=(10, 2))
    x_test = rng.normal(loc=-100.0, scale=50.0, size=(10, 2))

    norm_train, norm_val, norm_test = apply_normalization(x_train, x_val, x_test)

    # Fit-on-train-only means the TRAIN split's normalized stats are exactly
    # standardized, regardless of how extreme val/test are.
    assert np.abs(norm_train.mean(axis=0)).max() < 1e-8
    assert np.abs(norm_train.std(axis=0) - 1.0).max() < 1e-8

    # And the same fitted transform (train's mean/std) applied to val/test
    # means val/test are NOT standardized to 0/1 themselves - proving the
    # scaler wasn't refit on them.
    assert np.abs(norm_val.mean(axis=0)).max() > 1.0
    assert np.abs(norm_test.mean(axis=0)).max() > 1.0
