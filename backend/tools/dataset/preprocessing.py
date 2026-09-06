"""
backend/tools/dataset/preprocessing.py
=========================================
One-hot encoding (fixed, structural) + optional StandardScaler
(fit-on-train, no leakage).

Scope boundary (Requirement 1.9): this module does exactly two things and
nothing more. No missing-value imputation, no feature engineering, no
feature selection, no dimensionality reduction, no class balancing, no
searched/automatic transformation.

Leakage
-------
One-hot encoding operates on the fixed category vocabulary present in the
dataset - a structural property of a column, not a statistic learned from
data, so it carries no train/test leakage risk. It is applied to the full
dataset *before* splitting (see backend/tools/dataset/dataset.py), which is
what keeps train/val/test one-hot columns consistent with each other.

``StandardScaler`` is the one thing here that is actually fit on data:
``apply_normalization`` fits it on ``x_train`` only and applies the same
fitted transform to val/test, never fitting on val/test or the full dataset.

Requirements
------------
1.8  Fixed one-hot encoding; normalization as an optional per-experiment choice
1.9  No imputation/feature engineering/feature selection/class balancing
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from backend.models.dataset import DatasetProfile


def encode_features(df: pd.DataFrame, profile: DatasetProfile) -> np.ndarray:
    """One-hot encode categorical columns; return a numeric feature matrix.

    Must be called on the full dataset (before splitting) so that the
    resulting column set is identical regardless of which rows later land
    in train/val/test.
    """
    features = df[profile.feature_columns]
    if profile.categorical_columns:
        features = pd.get_dummies(
            features, columns=profile.categorical_columns, drop_first=False
        )
    return features.to_numpy(dtype=float)


def encode_target(series: pd.Series, profile: DatasetProfile) -> np.ndarray:
    """Encode the target column: class index (classification) or float (regression).

    Classification uses the fixed ``profile.class_labels`` order computed
    at ingestion, so the label<->index mapping never changes across
    experiments or models.
    """
    if profile.task_type == "classification":
        label_to_index = {label: i for i, label in enumerate(profile.class_labels)}
        return series.astype(str).map(label_to_index).to_numpy(dtype=np.int64)
    return series.to_numpy(dtype=float)


def apply_normalization(
    x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a StandardScaler on x_train only; apply the same fit to val/test.

    This is the only learned preprocessing step in the pipeline. Fitting
    exclusively on the training split - never on val/test, never on the
    full dataset - is what keeps it leakage-free.
    """
    scaler = StandardScaler()
    scaler.fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_val), scaler.transform(x_test)
