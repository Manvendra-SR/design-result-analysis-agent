"""
backend/models/experiment.py
=============================
One experiment = one complete training run of one configuration.

FAMILIES
    The model families the agent may use, and for each one the knobs it
    reads (with ranges and defaults), whether its result depends on the seed,
    its default preprocessing, and which knobs the agent may refine. This is
    the search space: "a valid configuration" is a fact about this table,
    not something an LLM is asked to respect.

ExperimentConfiguration
    What to train: dataset, model type, hyperparameters, preprocessing, seed.

ExperimentResult
    What came out: ``status`` is ``ok`` or ``failed`` (crashed, or produced a
    non-finite number). There is no third state - an unusual but valid result
    is evidence and stays in the analysis.

    Besides summary ``metrics``, each run keeps a per-row score for the
    validation and test splits (1/0 correctness for classification, squared
    error for regression). Those per-row scores are what the statistics
    resample - see ``backend/tools/stats.py``. They are excluded from API
    responses (thousands of numbers per run) but stored with the run.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Literal, NamedTuple, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.models.dataset import PreprocessingConfig
from backend.models.timestamps import UTCDateTime

#: Ordered simplest first; the simplest family in a selection plan is its reference.
ModelType = Literal["linear_baseline", "decision_tree", "random_forest", "mlp"]


class Knob(NamedTuple):
    low: float
    high: float
    default: Optional[float]  # None: the library's own default (the key is left out)
    integer: bool = False
    high_open: bool = False   # range is [low, high) instead of [low, high]
    low_open: bool = False    # range is (low, high] instead of [low, high]

    def check(self, name: str, value: float) -> None:
        above = value > self.low if self.low_open else value >= self.low
        below = value < self.high if self.high_open else value <= self.high
        if not (above and below):
            lo, hi = "(" if self.low_open else "[", ")" if self.high_open else "]"
            raise ValueError(f"{name} must be in {lo}{self.low:g}, {self.high:g}{hi}, got {value:g}")
        if self.integer and value != int(value):
            raise ValueError(f"{name} must be a whole number, got {value:g}")


class Family(NamedTuple):
    knobs: Dict[str, Knob]
    seeded: bool             # False: the result does not depend on random_seed, so it runs once
    normalize: bool          # default preprocessing (trees are scale-invariant)
    refinable: Tuple[str, ...]  # knobs the agent may change in selection mode

    @property
    def defaults(self) -> Dict[str, float]:
        return {k: v.default for k, v in self.knobs.items() if v.default is not None}


FAMILIES: Dict[str, Family] = {
    "linear_baseline": Family(knobs={}, seeded=False, normalize=True, refinable=()),
    "decision_tree": Family(
        knobs={
            "max_depth": Knob(1, 30, 8, integer=True),
            "min_samples_leaf": Knob(1, 200, 1, integer=True),
        },
        seeded=False, normalize=False, refinable=("max_depth", "min_samples_leaf"),
    ),
    "random_forest": Family(
        knobs={
            "max_depth": Knob(2, 40, None, integer=True),
            "min_samples_leaf": Knob(1, 100, 1, integer=True),
            "max_features": Knob(0.1, 1.0, None),
        },
        seeded=True, normalize=False, refinable=("max_depth", "min_samples_leaf", "max_features"),
    ),
    "mlp": Family(
        knobs={
            "hidden_size": Knob(1, 512, 64, integer=True),
            "dropout": Knob(0.0, 1.0, 0.0, high_open=True),
            "learning_rate": Knob(0.0, 1.0, 0.001, low_open=True),
            "batch_size": Knob(16, 4096, 32, integer=True),  # >= 16 keeps one run's cost bounded
            "epochs": Knob(1, 50, 20, integer=True),         # <= 50 keeps one run's cost bounded
        },
        seeded=True, normalize=True, refinable=("hidden_size", "dropout", "learning_rate", "epochs"),
    ),
}

#: Hyperparameters the MLP trainer reads, with the defaults it applies.
MLP_DEFAULTS: Dict[str, float] = FAMILIES["mlp"].defaults

#: Every knob name of every family.
ALL_KNOBS: Tuple[str, ...] = tuple(dict.fromkeys(k for f in FAMILIES.values() for k in f.knobs))


class ExperimentConfiguration(BaseModel):
    """Complete specification for one training run.

    ``hyperparameters`` may only hold knobs of ``model_type``'s family, each
    within its range (``FAMILIES``); a knob left out takes its default.
    """

    dataset_id: str
    model_type: ModelType
    hyperparameters: Dict[str, float] = Field(default_factory=dict)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    random_seed: int = 0

    model_config = ConfigDict(protected_namespaces=())  # 'model_type' is a domain field

    @model_validator(mode="after")
    def _validate_hyperparameters(self) -> "ExperimentConfiguration":
        knobs = FAMILIES[self.model_type].knobs
        unknown = set(self.hyperparameters) - set(knobs)
        if unknown:
            raise ValueError(f"unknown {self.model_type} hyperparameter(s): {sorted(unknown)}")
        for name, value in self.hyperparameters.items():
            knobs[name].check(name, value)
        return self

    def key(self) -> str:
        """Identity of the configuration ignoring the seed: runs with the same key are one candidate."""
        hp = ",".join(f"{k}={v:g}" for k, v in sorted(self.hyperparameters.items()))
        return f"{self.model_type}|{hp}|normalize={self.preprocessing.normalize}"


class ExperimentResult(BaseModel):
    """A finished training run."""

    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    round: int = Field(description="1-based investigation round that ran this experiment")
    config: ExperimentConfiguration
    status: Literal["ok", "failed"]
    error: Optional[str] = None
    metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Validation metric (accuracy or mse), the same metric on the training split "
        "(train_accuracy or train_mse), train_loss, training_time_seconds",
    )
    val_scores: Optional[List[float]] = Field(default=None, exclude=True)
    test_scores: Optional[List[float]] = Field(default=None, exclude=True)
    created_at: UTCDateTime = Field(default_factory=datetime.utcnow)
