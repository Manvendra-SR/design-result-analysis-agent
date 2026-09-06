"""
backend/tools/dataset/dataset.py
===================================
Dataset facade: loads the CSV referenced by a DatasetProfile and produces a
deterministic, optionally-normalized train/val/test split as numpy arrays -
what trainers consume, regardless of how or when the dataset was ingested.

This is the abstraction that lets ``train_mlp``/``train_linear_baseline``
stay dataset-source-agnostic: they call ``Dataset(profile).load_split(...)``
instead of reaching into a specific file format or loader.

Requirements
------------
3.1  Trainers consume a resolved Dataset, not a hardcoded data source
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from backend.models.dataset import DatasetProfile
from backend.tools.dataset.preprocessing import (
    apply_normalization,
    encode_features,
    encode_target,
)
from backend.tools.dataset.splitting import compute_split


class DatasetSplit(NamedTuple):
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


class Dataset:
    """Loads and splits the CSV referenced by a ``DatasetProfile``.

    ``load_split`` is deterministic given ``profile.split_seed``: the same
    profile always yields identical row assignments to train/val/test,
    regardless of how many times or in what order it is called.
    """

    def __init__(self, profile: DatasetProfile) -> None:
        self._profile = profile

    def load_split(self, normalize: bool = False) -> DatasetSplit:
        """Load, encode, and split the dataset.

        Parameters
        ----------
        normalize:
            When true, fits a ``StandardScaler`` on the training split and
            applies it to val/test as well (see
            ``backend/tools/dataset/preprocessing.py``). One-hot encoding
            of categorical columns always happens, regardless of this flag.
        """
        df = pd.read_csv(self._profile.storage_path)

        x = encode_features(df, self._profile)
        y = encode_target(df[self._profile.target_column], self._profile)

        split = compute_split(len(df), self._profile.split_seed)
        x_train, x_val, x_test = x[split.train], x[split.val], x[split.test]
        y_train, y_val, y_test = y[split.train], y[split.val], y[split.test]

        if normalize:
            x_train, x_val, x_test = apply_normalization(x_train, x_val, x_test)

        return DatasetSplit(x_train, y_train, x_val, y_val, x_test, y_test)
