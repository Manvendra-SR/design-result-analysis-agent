"""
backend/models/experiment.py
=============================
One experiment = one complete training run of one configuration.

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
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.models.dataset import PreprocessingConfig
from backend.models.timestamps import UTCDateTime

ModelType = Literal["mlp", "linear_baseline"]

#: Hyperparameters the MLP trainer reads, with the defaults it applies.
MLP_DEFAULTS: Dict[str, float] = {
    "hidden_size": 64,
    "dropout": 0.0,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 20,
}


class ExperimentConfiguration(BaseModel):
    """Complete specification for one training run.

    ``mlp`` reads ``hidden_size``, ``dropout``, ``learning_rate``,
    ``batch_size`` and ``epochs`` (defaults in ``MLP_DEFAULTS``).
    ``linear_baseline`` (logistic / linear regression) has no hyperparameters.
    """

    dataset_id: str
    model_type: ModelType
    hyperparameters: Dict[str, float] = Field(default_factory=dict)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    random_seed: int = 0

    model_config = ConfigDict(protected_namespaces=())  # 'model_type' is a domain field

    @model_validator(mode="after")
    def _validate_hyperparameters(self) -> "ExperimentConfiguration":
        hp = self.hyperparameters
        if self.model_type == "linear_baseline":
            return self
        unknown = set(hp) - set(MLP_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown mlp hyperparameter(s): {sorted(unknown)}")
        if "dropout" in hp and not 0.0 <= hp["dropout"] < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {hp['dropout']}")
        if "learning_rate" in hp and not 0.0 < hp["learning_rate"] <= 1.0:
            raise ValueError(f"learning_rate must be in (0, 1], got {hp['learning_rate']}")
        if "hidden_size" in hp and not 1 <= hp["hidden_size"] <= 512:
            raise ValueError(f"hidden_size must be in [1, 512], got {hp['hidden_size']}")
        if "batch_size" in hp and not 1 <= hp["batch_size"] <= 4096:
            raise ValueError(f"batch_size must be in [1, 4096], got {hp['batch_size']}")
        if "epochs" in hp and not 1 <= hp["epochs"] <= 100:
            raise ValueError(f"epochs must be in [1, 100], got {hp['epochs']}")
        return self


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
        description="Validation metric (accuracy or mse), train_loss, training_time_seconds",
    )
    val_scores: Optional[List[float]] = Field(default=None, exclude=True)
    test_scores: Optional[List[float]] = Field(default=None, exclude=True)
    created_at: UTCDateTime = Field(default_factory=datetime.utcnow)
