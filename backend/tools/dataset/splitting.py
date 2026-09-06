"""
backend/tools/dataset/splitting.py
=====================================
Deterministic 70/15/15 train/val/test split, fixed per dataset via
``DatasetProfile.split_seed`` - independent of any experiment's
``random_seed``.

Why the split must not be reseeded per experiment
----------------------------------------------------
If the split changed with each experiment's random_seed (as the original
Phase 3 MNIST loader did), comparing results across seeds would silently
compare different train/val rows in addition to different model
initializations - confounding exactly the kind of "does dropout=0.2 beat
dropout=0.0" comparison this system exists to answer. Computing the split
once per dataset removes that confound.

Requirements
------------
1.6  Fixed train/val/test split per dataset
3.8  random_seed governs model init / training randomness only
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from sklearn.model_selection import train_test_split


class SplitIndices(NamedTuple):
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def compute_split(n_rows: int, split_seed: int) -> SplitIndices:
    """Compute a deterministic 70/15/15 split of row indices [0, n_rows)."""
    indices = np.arange(n_rows)
    train_idx, rest_idx = train_test_split(
        indices, test_size=0.30, random_state=split_seed
    )
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.50, random_state=split_seed
    )
    return SplitIndices(train=train_idx, val=val_idx, test=test_idx)
