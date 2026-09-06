"""
backend/tools/trainers.py
============================
Deterministic training routines for the two supported models: ``mlp``
(generic feed-forward network, PyTorch) and ``linear_baseline``
(logistic/linear regression, scikit-learn), both executed against a
resolved ``DatasetProfile`` rather than a hardcoded data source.

Both functions are pure with respect to ``(ExperimentConfiguration,
DatasetProfile)`` in, ``ExperimentResult`` out — they do not catch
exceptions (that is ``ExperimentRunner``'s job) and they do not touch the
database (that is ``StateManager``'s job).

Metrics contract: keyed by task_type, not model_type
-------------------------------------------------------
So that ``mlp`` and ``linear_baseline`` results on the same dataset are
directly comparable ("which of these configurations performs best?"):

- ``classification``: ``train_loss``, ``val_loss``, ``accuracy``,
  ``n_classes``, ``n_val_samples``, ``training_time_seconds``
  (+ ``initial_train_loss`` for ``mlp`` only)
- ``regression``: ``train_loss``, ``val_loss``, ``training_time_seconds``
  — no ``accuracy`` key at all (replaces the old "accuracy=0.0 by
  convention" approach; regression results never fake a classification metric)

Seed vs. split
---------------
``config.random_seed`` governs model init / batch shuffling only.
``dataset_profile.split_seed`` (fixed once per dataset — see
``backend/tools/dataset/splitting.py``) governs the train/val/test split,
so multi-seed comparisons vary training randomness only, never which rows
end up in validation.

``linear_baseline``'s "loss"
-------------------------------
scikit-learn's ``LogisticRegression``/``LinearRegression`` have no epoch
loop, so there is no natural per-epoch "loss" the way there is for ``mlp``.
For classification, log-loss (the same cross-entropy quantity
``nn.CrossEntropyLoss`` optimizes) is used via ``sklearn.metrics.log_loss``
on predicted probabilities, so ``train_loss``/``val_loss`` stay meaningful
and comparable in kind to ``mlp``'s. For regression, MSE is used, matching
``nn.MSELoss``. Neither model sets ``initial_train_loss`` — the same
convention previously used to skip the loss-divergence check for the old
synthetic-regression trainer, generalized to any single-shot fit.

Requirements
------------
3.1  mlp and linear_baseline both execute against a resolved Dataset
3.3  mlp: configurable hidden_size, dropout, learning_rate, batch_size, epochs
3.4  linear_baseline: LogisticRegression/LinearRegression, no tunable hyperparameters
3.5  Metrics keyed by task_type, not model_type
3.8  random_seed governs model init / training randomness only
"""

from __future__ import annotations

import logging
import math
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader, TensorDataset

from backend.models.dataset import DatasetProfile
from backend.models.experiment import ExperimentConfiguration, ExperimentResult
from backend.tools.dataset.dataset import Dataset
from backend.tools.models import TabularMLP

logger = logging.getLogger(__name__)


def _loss_window_endpoints(epoch_losses: List[float]) -> Tuple[float, float]:
    """Return (initial_loss, final_loss) bounding the final 20% of epochs.

    The window covers ``ceil(0.2 * n_epochs)`` epochs, and ``initial_loss``
    is the loss immediately preceding that window — generalizes to any
    epoch count (unchanged from the original Phase 3 implementation).
    """
    n = len(epoch_losses)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return epoch_losses[0], epoch_losses[0]
    window_size = max(1, math.ceil(0.2 * n))
    initial_idx = max(0, n - window_size - 1)
    return epoch_losses[initial_idx], epoch_losses[-1]


def train_mlp(
    config: ExperimentConfiguration,
    dataset_profile: DatasetProfile,
    session_id: str = "",
) -> ExperimentResult:
    """Train a ``TabularMLP`` per ``config`` against ``dataset_profile``.

    Hyperparameters read from ``config.hyperparameters`` (defaults applied
    for keys the caller omits): ``hidden_size`` (64), ``dropout`` (0.0),
    ``learning_rate`` (0.001), ``batch_size`` (32), ``epochs`` (20).

    Does not catch exceptions; ``ExperimentRunner`` turns a raised
    exception into a ``status="failed"`` result.

    Parameters
    ----------
    config:
        Must have ``model_type == "mlp"``.
    dataset_profile:
        The dataset referenced by ``config.dataset_id``, already resolved
        by the caller (this function never touches the database).
    session_id:
        Passed through into the returned ``ExperimentResult``.

    Requirements: 3.1, 3.3, 3.5, 3.8
    """
    hp = config.hyperparameters
    hidden_size = int(hp.get("hidden_size", 64))
    dropout = float(hp.get("dropout", 0.0))
    learning_rate = float(hp.get("learning_rate", 0.001))
    batch_size = int(hp.get("batch_size", 32))
    epochs = int(hp.get("epochs", 20))

    # Seed before anything random-dependent so the whole run is reproducible.
    # This governs model init / batch shuffling only — the train/val/test
    # split itself comes from dataset_profile.split_seed (see module docstring).
    torch.manual_seed(config.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = Dataset(dataset_profile).load_split(normalize=config.preprocessing.normalize)

    is_classification = dataset_profile.task_type == "classification"
    output_dim = dataset_profile.n_classes if is_classification else 1

    x_train_t = torch.tensor(split.x_train, dtype=torch.float32)
    x_val_t = torch.tensor(split.x_val, dtype=torch.float32)
    if is_classification:
        y_train_t = torch.tensor(split.y_train, dtype=torch.long)
        y_val_t = torch.tensor(split.y_val, dtype=torch.long)
    else:
        y_train_t = torch.tensor(split.y_train, dtype=torch.float32)
        y_val_t = torch.tensor(split.y_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.random_seed),
    )

    # Reseed immediately before model construction, matching the original
    # MNIST trainer's defensive pattern: keeps weight init decoupled from
    # anything upstream that might have consumed the global torch RNG.
    torch.manual_seed(config.random_seed)
    model = TabularMLP(
        input_dim=dataset_profile.n_features,
        output_dim=output_dim,
        hidden_size=hidden_size,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion: nn.Module = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()

    epoch_train_losses: List[float] = []
    val_loss = 0.0
    accuracy = 0.0

    x_val_device = x_val_t.to(device)
    y_val_device = y_val_t.to(device)

    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        epoch_train_loss = running_loss / max(n_batches, 1)
        epoch_train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val_device)
            val_loss = float(criterion(val_outputs, y_val_device).item())
            if is_classification:
                preds = val_outputs.argmax(dim=1)
                accuracy = float((preds == y_val_device).float().mean().item())

        logger.debug(
            "mlp epoch %d/%d: train_loss=%.4f val_loss=%.4f",
            epoch + 1, epochs, epoch_train_loss, val_loss,
        )

    training_time = time.perf_counter() - start
    initial_train_loss, final_train_loss = _loss_window_endpoints(epoch_train_losses)

    metrics = {
        "train_loss": final_train_loss,
        "val_loss": val_loss,
        "training_time_seconds": training_time,
        "initial_train_loss": initial_train_loss,
    }
    if is_classification:
        metrics["accuracy"] = accuracy
        metrics["n_classes"] = float(dataset_profile.n_classes)
        metrics["n_val_samples"] = float(len(split.y_val))

    logger.info(
        "mlp training complete: dataset=%s hidden_size=%d dropout=%.2f lr=%.4f "
        "batch_size=%d epochs=%d seed=%d -> val_loss=%.4f (%.1fs)",
        dataset_profile.dataset_id, hidden_size, dropout, learning_rate,
        batch_size, epochs, config.random_seed, val_loss, training_time,
    )

    return ExperimentResult(
        session_id=session_id,
        config=config,
        task_type=dataset_profile.task_type,
        metrics=metrics,
        status="success",
    )


def train_linear_baseline(
    config: ExperimentConfiguration,
    dataset_profile: DatasetProfile,
    session_id: str = "",
) -> ExperimentResult:
    """Fit a scikit-learn LogisticRegression/LinearRegression baseline.

    Dispatched by ``dataset_profile.task_type``. No tunable hyperparameters
    — ``config.hyperparameters`` is ignored. Uses the same Dataset/
    preprocessing pipeline as ``train_mlp``, so results are directly
    comparable.

    Requirements: 3.1, 3.4, 3.5
    """
    split = Dataset(dataset_profile).load_split(normalize=config.preprocessing.normalize)
    is_classification = dataset_profile.task_type == "classification"

    start = time.perf_counter()
    if is_classification:
        model = LogisticRegression(max_iter=1000, random_state=config.random_seed)
        model.fit(split.x_train, split.y_train)

        class_ids = list(range(dataset_profile.n_classes))
        train_loss = float(
            log_loss(split.y_train, model.predict_proba(split.x_train), labels=class_ids)
        )
        val_loss = float(
            log_loss(split.y_val, model.predict_proba(split.x_val), labels=class_ids)
        )
        accuracy = float(model.score(split.x_val, split.y_val))
    else:
        model = LinearRegression()
        model.fit(split.x_train, split.y_train)
        train_loss = float(np.mean((model.predict(split.x_train) - split.y_train) ** 2))
        val_loss = float(np.mean((model.predict(split.x_val) - split.y_val) ** 2))

    training_time = time.perf_counter() - start

    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "training_time_seconds": training_time,
    }
    if is_classification:
        metrics["accuracy"] = accuracy
        metrics["n_classes"] = float(dataset_profile.n_classes)
        metrics["n_val_samples"] = float(len(split.y_val))

    logger.info(
        "linear_baseline training complete: dataset=%s task_type=%s seed=%d "
        "-> val_loss=%.4f (%.4fs)",
        dataset_profile.dataset_id, dataset_profile.task_type,
        config.random_seed, val_loss, training_time,
    )

    return ExperimentResult(
        session_id=session_id,
        config=config,
        task_type=dataset_profile.task_type,
        metrics=metrics,
        status="success",
    )
