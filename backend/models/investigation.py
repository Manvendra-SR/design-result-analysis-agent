"""
backend/models/investigation.py
=================================
Everything an investigation produces besides raw experiments.

Plan
    The experiment design: ONE factor, the levels to try, which level is the
    reference, and the configuration every level shares. Configurations are
    derived from it (``config_for``), so "vary one factor at a time" holds by
    construction rather than by asking an LLM nicely.

Decision
    What the Recommender decided after a round: explore more levels of the
    same factor, or conclude.

Analysis / ConditionSummary / Comparison
    Statistics over one evaluation split. Never stored - recomputed from the
    experiments whenever needed (``backend/tools/stats.py``).

Report
    The final answer: one comparison, best validation level vs the reference,
    evaluated once on the held-out test split.

Session / SessionSummary
    One investigation and its stored state.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.models.dataset import PreprocessingConfig
from backend.models.experiment import ExperimentConfiguration, ModelType
from backend.models.timestamps import UTCDateTime

#: A value of the factor under test: a model type, a normalize flag, or a number.
Level = Union[bool, float, str]

Factor = Literal[
    "model_type", "normalize", "hidden_size", "dropout", "learning_rate", "batch_size", "epochs"
]
HYPERPARAMETER_FACTORS = ("hidden_size", "dropout", "learning_rate", "batch_size", "epochs")


def format_value(value: Level) -> str:
    """20.0 -> '20', 0.2 -> '0.2', True -> 'true'."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


class BaseSetup(BaseModel):
    """The configuration every level shares (the factor's own key is overridden)."""

    model_type: ModelType = "mlp"
    hyperparameters: Dict[str, float] = Field(default_factory=dict)
    normalize: bool = False

    model_config = ConfigDict(protected_namespaces=())  # 'model_type' is a domain field


class Plan(BaseModel):
    factor: Factor
    levels: List[Level] = Field(description="Every level tried so far, reference included")
    reference: Level
    base: BaseSetup = Field(default_factory=BaseSetup)
    rationale: str

    @model_validator(mode="after")
    def _validate(self) -> "Plan":
        if len(self.levels) < 2:
            raise ValueError("a plan needs at least 2 levels to compare")
        if len({format_value(v) for v in self.levels}) != len(self.levels):
            raise ValueError(f"levels must be distinct, got {self.levels}")
        check_level(self.factor, self.reference)
        if self.reference not in self.levels:
            raise ValueError(f"reference {self.reference!r} is not one of the levels")
        if self.factor in HYPERPARAMETER_FACTORS and self.base.model_type != "mlp":
            raise ValueError(f"{self.factor} is an mlp hyperparameter, but the base model is linear_baseline")
        for level in self.levels:
            check_level(self.factor, level)
            self.config_for(level, dataset_id="-")  # runs ExperimentConfiguration's range checks
        return self

    def config_for(self, level: Level, dataset_id: str, seed: int = 0) -> ExperimentConfiguration:
        """The base configuration with the factor set to ``level``."""
        model_type = self.base.model_type
        hp = dict(self.base.hyperparameters)
        normalize = self.base.normalize
        if self.factor == "model_type":
            model_type = level  # type: ignore[assignment]
        elif self.factor == "normalize":
            normalize = bool(level)
        else:
            hp[self.factor] = float(level)
        if model_type == "linear_baseline":
            hp = {}
        return ExperimentConfiguration(
            dataset_id=dataset_id,
            model_type=model_type,
            hyperparameters=hp,
            preprocessing=PreprocessingConfig(normalize=normalize),
            random_seed=seed,
        )

    def level_of(self, config: ExperimentConfiguration) -> Level:
        """Which level of the factor a configuration belongs to."""
        if self.factor == "model_type":
            return config.model_type
        if self.factor == "normalize":
            return config.preprocessing.normalize
        return config.hyperparameters[self.factor]

    def label(self, level: Level) -> str:
        if self.factor == "model_type":
            return str(level)
        return f"{self.factor}={format_value(level)}"


def check_level(factor: str, level: Level) -> None:
    """Raise ValueError if ``level`` is the wrong kind of value for ``factor``."""
    if factor == "model_type":
        if level not in ("mlp", "linear_baseline"):
            raise ValueError(f"model_type level must be 'mlp' or 'linear_baseline', got {level!r}")
    elif factor == "normalize":
        if not isinstance(level, bool):
            raise ValueError(f"normalize level must be true/false, got {level!r}")
    elif isinstance(level, bool) or not isinstance(level, (int, float)):
        raise ValueError(f"{factor} level must be a number, got {level!r}")


class Decision(BaseModel):
    round: int = Field(description="The round whose results this decision was based on")
    action: Literal["explore", "conclude"]
    new_levels: List[Level] = Field(default_factory=list)
    rationale: str
    decided_by: Literal["agent", "budget"] = "agent"


# ---------------------------------------------------------------------------
# Statistics (computed, never stored on their own)
# ---------------------------------------------------------------------------
class ConditionSummary(BaseModel):
    level: Level
    label: str
    is_reference: bool
    n_ok: int
    n_failed: int
    mean: Optional[float] = Field(description="Mean metric over the condition's successful runs")
    seed_std: Optional[float] = Field(description="Spread across seeds (stability, not uncertainty)")


class Comparison(BaseModel):
    """One level vs the reference: difference in the metric, with a 95% CI."""

    level: Level
    label: str
    diff: float = Field(description="metric(level) - metric(reference)")
    ci_low: float
    ci_high: float
    verdict: Literal["better", "worse", "inconclusive"]


class Analysis(BaseModel):
    split: Literal["val", "test"]
    metric: Literal["accuracy", "mse"]
    higher_is_better: bool
    n_rows: int = Field(description="Evaluation rows each comparison resamples")
    majority_rate: Optional[float] = Field(
        default=None, description="Accuracy of always predicting the most common class"
    )
    conditions: List[ConditionSummary]
    comparisons: List[Comparison]


class Report(BaseModel):
    challenger: str = Field(description="Label of the best non-reference level on validation")
    reference: str
    metric: Literal["accuracy", "mse"]
    higher_is_better: bool
    challenger_score: float = Field(description="Test-split metric")
    reference_score: float
    comparison: Comparison = Field(description="Test-split comparison, challenger vs reference")
    majority_rate: Optional[float] = None
    n_test_rows: int
    rounds_run: int
    stopped_by: Literal["agent", "budget"]
    headline: str = Field(description="Deterministic one-sentence answer")
    interpretation: Optional[str] = Field(default=None, description="LLM-written explanation")


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
SessionStatus = Literal["pending", "running", "done", "failed"]


class Session(BaseModel):
    session_id: str
    dataset_id: str
    research_question: str
    status: SessionStatus
    error: Optional[str] = None
    plan: Optional[Plan] = None
    decisions: List[Decision] = Field(default_factory=list)
    report: Optional[Report] = None
    created_at: UTCDateTime = Field(default_factory=datetime.utcnow)


class SessionSummary(BaseModel):
    session_id: str
    dataset_id: str
    research_question: str
    status: SessionStatus
    experiment_count: int
    created_at: UTCDateTime
