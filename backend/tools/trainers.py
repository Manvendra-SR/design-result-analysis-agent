"""
backend/tools/trainers.py
============================
Training routines for the model families in ``FAMILIES``, against any profiled dataset:

- ``linear_baseline`` : LogisticRegression / LinearRegression (scikit-learn)
- ``decision_tree``   : DecisionTreeClassifier / Regressor (scikit-learn, random_state 0)
- ``random_forest``   : RandomForestClassifier / Regressor, 200 trees (scikit-learn)
- ``mlp``             : ``TabularMLP`` (PyTorch), fixed number of epochs

All use the same fixed train/val/test split and preprocessing, and all
return the same thing, so their results are directly comparable:

- ``metrics``     : the validation metric (``accuracy`` for classification,
                    ``mse`` for regression), the same metric on the training
                    split (``train_accuracy`` / ``train_mse``, so the agent can
                    tell overfitting from underfitting), ``train_loss``,
                    ``training_time_seconds``
- ``val_scores``  : one score per validation row
- ``test_scores`` : one score per test row (sealed until the final report)

A per-row score is 1.0/0.0 (correct or not) for classification and the
squared error for regression. The statistics resample these rows.

``config.random_seed`` governs weight init, dropout and batch order (MLP) and
the bootstrap samples (random forest) only; the split is fixed per dataset
(``DatasetProfile.split_seed``). The linear model and the decision tree do not
depend on it.

The MLP trains on CUDA when a GPU is available, otherwise on CPU.
"""

from __future__ import annotations

import time
from typing import Dict, List, NamedTuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from backend.models.dataset import DatasetProfile
from backend.models.experiment import MLP_DEFAULTS, ExperimentConfiguration
from backend.tools.dataset.dataset import Dataset
from backend.tools.models import TabularMLP


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class TrainOutput(NamedTuple):
    metrics: Dict[str, float]
    val_scores: List[float]
    test_scores: List[float]


def row_scores(predictions: np.ndarray, targets: np.ndarray, task_type: str) -> np.ndarray:
    """Per-row score: correctness for classification, squared error for regression."""
    if task_type == "classification":
        return (predictions == targets).astype(float)
    return (predictions - targets).astype(float) ** 2


def _output(task_type, train_scores, val_scores, test_scores, train_loss, seconds) -> TrainOutput:
    metric = "accuracy" if task_type == "classification" else "mse"
    return TrainOutput(
        metrics={
            metric: float(np.mean(val_scores)),
            f"train_{metric}": float(np.mean(train_scores)),
            "train_loss": float(train_loss),
            "training_time_seconds": float(seconds),
        },
        val_scores=val_scores.tolist(),
        test_scores=test_scores.tolist(),
    )


def train_mlp(config: ExperimentConfiguration, profile: DatasetProfile) -> TrainOutput:
    hp = {**MLP_DEFAULTS, **config.hyperparameters}
    is_classification = profile.task_type == "classification"
    split = Dataset(profile).load_split(normalize=config.preprocessing.normalize)

    # The whole training set lives on the device and batches are taken by index,
    # so there is no per-batch host-to-GPU copy. Shuffling uses a CPU generator
    # seeded per run, so batch order is identical on CPU and GPU.
    target_dtype = torch.long if is_classification else torch.float32
    x_train = torch.tensor(split.x_train, dtype=torch.float32, device=DEVICE)
    y_train = torch.tensor(split.y_train, dtype=target_dtype, device=DEVICE)
    batch_size = int(hp["batch_size"])
    shuffle = torch.Generator().manual_seed(config.random_seed)

    torch.manual_seed(config.random_seed)  # weight init + dropout masks (CPU and CUDA)
    model = TabularMLP(
        input_dim=profile.n_features,
        output_dim=profile.n_classes if is_classification else 1,
        hidden_size=int(hp["hidden_size"]),
        dropout=float(hp["dropout"]),
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(hp["learning_rate"]))
    criterion = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()

    start = time.perf_counter()
    train_loss = float("nan")
    n = len(x_train)
    for _ in range(int(hp["epochs"])):
        model.train()
        order = torch.randperm(n, generator=shuffle).to(DEVICE)
        total = torch.zeros((), device=DEVICE)
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(idx)
        train_loss = (total / n).item()  # one device sync per epoch, not per batch
    seconds = time.perf_counter() - start

    model.eval()

    def predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = model(torch.tensor(x, dtype=torch.float32, device=DEVICE))
        return (out.argmax(dim=1) if is_classification else out).cpu().numpy()

    return _output(
        profile.task_type,
        row_scores(predict(split.x_train), split.y_train, profile.task_type),
        row_scores(predict(split.x_val), split.y_val, profile.task_type),
        row_scores(predict(split.x_test), split.y_test, profile.task_type),
        train_loss,
        seconds,
    )


def _train_sklearn(model, config: ExperimentConfiguration, profile: DatasetProfile) -> TrainOutput:
    """Fit a scikit-learn estimator and score every train/val/test row."""
    split = Dataset(profile).load_split(normalize=config.preprocessing.normalize)
    is_classification = profile.task_type == "classification"

    start = time.perf_counter()
    model.fit(split.x_train, split.y_train)
    seconds = time.perf_counter() - start

    if is_classification:
        train_loss = log_loss(split.y_train, model.predict_proba(split.x_train), labels=model.classes_)
    else:
        train_loss = float(np.mean((model.predict(split.x_train) - split.y_train) ** 2))
    return _output(
        profile.task_type,
        row_scores(model.predict(split.x_train), split.y_train, profile.task_type),
        row_scores(model.predict(split.x_val), split.y_val, profile.task_type),
        row_scores(model.predict(split.x_test), split.y_test, profile.task_type),
        train_loss,
        seconds,
    )


def _tree_params(config: ExperimentConfiguration) -> Dict:
    """The knobs set on this configuration, typed for scikit-learn (unset -> library default)."""
    hp = config.hyperparameters
    params: Dict = {k: int(hp[k]) for k in ("max_depth", "min_samples_leaf") if k in hp}
    if "max_features" in hp:
        params["max_features"] = float(hp["max_features"])
    return params


def train_linear_baseline(config: ExperimentConfiguration, profile: DatasetProfile) -> TrainOutput:
    is_classification = profile.task_type == "classification"
    model = LogisticRegression(max_iter=1000) if is_classification else LinearRegression()
    return _train_sklearn(model, config, profile)


def train_decision_tree(config: ExperimentConfiguration, profile: DatasetProfile) -> TrainOutput:
    cls = DecisionTreeClassifier if profile.task_type == "classification" else DecisionTreeRegressor
    return _train_sklearn(cls(random_state=0, **_tree_params(config)), config, profile)


def train_random_forest(config: ExperimentConfiguration, profile: DatasetProfile) -> TrainOutput:
    cls = RandomForestClassifier if profile.task_type == "classification" else RandomForestRegressor
    model = cls(n_estimators=200, n_jobs=-1, random_state=config.random_seed, **_tree_params(config))
    return _train_sklearn(model, config, profile)


TRAINERS = {
    "linear_baseline": train_linear_baseline,
    "decision_tree": train_decision_tree,
    "random_forest": train_random_forest,
    "mlp": train_mlp,
}
