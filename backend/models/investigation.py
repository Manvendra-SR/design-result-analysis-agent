"""
backend/models/investigation.py
=================================
Everything an investigation produces besides raw experiments.

Candidate
    One fully specified configuration (minus the seed) with its lineage:
    ``parent`` and the one ``knob`` changed to ``value`` relative to it. Every
    candidate after the first round is a parent with ONE knob changed, so each
    new experiment is a controlled comparison with something already measured.

Plan
    The experiment design, in one of two modes:

    effect     "Does X help?" - ONE factor, its levels, a reference level and
               the configuration every level shares (``base``). Every level is
               the reference with only the factor changed; the factor is locked.
    selection  "Which model is best?" - 2-4 model families, each at its
               registry defaults; the simplest family is the reference. Later
               rounds add one-knob children of candidates still in contention.

    Candidates are derived by code (``Plan.effect``, ``Plan.selection``,
    ``Plan.refine``), so "change one thing at a time" holds by construction
    rather than by asking an LLM nicely.

Decision
    What happened after a round: ``refine`` (1-3 new values of one knob of one
    parent) or ``conclude``, and who decided - the agent, the round budget, or
    the settled rule (every contender in one family).

Analysis / ConditionSummary / Comparison
    Statistics over one evaluation split. Never stored - recomputed from the
    experiments whenever needed (``backend/tools/stats.py``).

Report
    The final answer: the winner chosen on validation, compared once on the
    held-out test split with the reference (and, in selection mode, with the
    best candidate of another family).

Session / SessionSummary
    One investigation and its stored state.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.models.dataset import PreprocessingConfig
from backend.models.experiment import ALL_KNOBS, FAMILIES, ExperimentConfiguration, ModelType
from backend.models.timestamps import UTCDateTime

#: A value of the factor under test: a model type, a normalize flag, or a number.
Level = Union[bool, float, str]

FACTORS = ("model_type", "normalize", *ALL_KNOBS)
Factor = Literal[FACTORS]  # type: ignore[valid-type]

#: Families, simplest first.
FAMILY_ORDER: List[str] = list(FAMILIES)

#: Hard limit on candidates one refinement may add.
MAX_NEW_CANDIDATES = 3


def format_value(value: Level) -> str:
    """20.0 -> '20', 0.2 -> '0.2', True -> 'true'."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def check_level(factor: str, level: Level) -> None:
    """Raise ValueError if ``level`` is the wrong kind of value for ``factor``."""
    if factor == "model_type":
        if level not in FAMILIES:
            raise ValueError(f"model_type level must be one of {FAMILY_ORDER}, got {level!r}")
    elif factor == "normalize":
        if not isinstance(level, bool):
            raise ValueError(f"normalize level must be true/false, got {level!r}")
    elif isinstance(level, bool) or not isinstance(level, (int, float)):
        raise ValueError(f"{factor} level must be a number, got {level!r}")


class BaseSetup(BaseModel):
    """Effect mode: the configuration every level shares (the factor's own key is overridden).

    ``normalize=None`` means the family's default; hyperparameters that do not
    belong to a level's family are ignored for that level.
    """

    model_type: ModelType = "mlp"
    hyperparameters: Dict[str, float] = Field(default_factory=dict)
    normalize: Optional[bool] = None

    model_config = ConfigDict(protected_namespaces=())  # 'model_type' is a domain field


class Candidate(BaseModel):
    id: str
    label: str
    model_type: ModelType
    hyperparameters: Dict[str, float] = Field(default_factory=dict)
    normalize: bool
    parent: Optional[str] = Field(default=None, description="Candidate this one was derived from")
    knob: Optional[str] = Field(default=None, description="What differs from the parent (effect mode: the factor)")
    value: Optional[Level] = None
    round: int = 1

    model_config = ConfigDict(protected_namespaces=())

    def config(self, dataset_id: str, seed: int = 0) -> ExperimentConfiguration:
        return ExperimentConfiguration(
            dataset_id=dataset_id,
            model_type=self.model_type,
            hyperparameters=self.hyperparameters,
            preprocessing=PreprocessingConfig(normalize=self.normalize),
            random_seed=seed,
        )

    @property
    def change(self) -> Optional[str]:
        return f"{self.knob}={format_value(self.value)}" if self.knob else None


def _effect_candidate(base: BaseSetup, factor: str, level: Level, **fields) -> Candidate:
    model_type, hp, normalize = base.model_type, dict(base.hyperparameters), base.normalize
    if factor == "model_type":
        model_type = level  # type: ignore[assignment]
    elif factor == "normalize":
        normalize = bool(level)
    else:
        if factor not in FAMILIES[model_type].knobs:
            raise ValueError(f"{factor} is not a hyperparameter of {model_type}")
        hp[factor] = float(level)
    family = FAMILIES[model_type]
    return Candidate(
        label=str(level) if factor == "model_type" else f"{factor}={format_value(level)}",
        model_type=model_type,
        hyperparameters={k: v for k, v in hp.items() if k in family.knobs},
        normalize=family.normalize if normalize is None else normalize,
        knob=factor,
        value=level,
        **fields,
    )


class Plan(BaseModel):
    mode: Literal["effect", "selection"]
    candidates: List[Candidate]
    reference: str = Field(description="Id of the reference candidate")
    factor: Optional[Factor] = Field(default=None, description="Effect mode: the one factor varied")
    base: Optional[BaseSetup] = Field(default=None, description="Effect mode: the configuration levels share")
    rationale: str

    # ------------------------------------------------------------ construction
    @classmethod
    def effect(cls, factor: str, levels: Sequence[Level], reference: Level, base: BaseSetup, rationale: str) -> "Plan":
        """Every level is the base with only ``factor`` changed; the reference is the root."""
        if len(levels) < 2:
            raise ValueError("a plan needs at least 2 levels to compare")
        if len({format_value(v) for v in levels}) != len(levels):
            raise ValueError(f"levels must be distinct, got {list(levels)}")
        for level in [reference, *levels]:
            check_level(factor, level)
        if format_value(reference) not in {format_value(v) for v in levels}:
            raise ValueError(f"reference {reference!r} is not one of the levels")
        ordered = [reference, *[v for v in levels if format_value(v) != format_value(reference)]]
        candidates = [
            _effect_candidate(base, factor, level, id=f"c{i + 1}", parent=None if i == 0 else "c1")
            for i, level in enumerate(ordered)
        ]
        return cls(mode="effect", candidates=candidates, reference="c1", factor=factor, base=base,
                   rationale=rationale)

    @classmethod
    def selection(cls, families: Sequence[str], rationale: str) -> "Plan":
        """Each family at its registry defaults; the simplest family is the reference."""
        unknown = [f for f in families if f not in FAMILIES]
        if unknown:
            raise ValueError(f"unknown model families {unknown}; choose from {FAMILY_ORDER}")
        chosen = [f for f in FAMILY_ORDER if f in families]
        if len(chosen) != len(families) or len(chosen) < 2:
            raise ValueError(f"choose 2-{len(FAMILIES)} distinct families, got {list(families)}")
        candidates = [
            Candidate(id=f"c{i + 1}", label=f, model_type=f, hyperparameters=FAMILIES[f].defaults,
                      normalize=FAMILIES[f].normalize)
            for i, f in enumerate(chosen)
        ]
        return cls(mode="selection", candidates=candidates, reference="c1", rationale=rationale)

    @model_validator(mode="after")
    def _validate(self) -> "Plan":
        ids = [c.id for c in self.candidates]
        if len(self.candidates) < 2:
            raise ValueError("a plan needs at least 2 candidates to compare")
        if len(set(ids)) != len(ids):
            raise ValueError(f"candidate ids must be unique, got {ids}")
        if self.reference not in ids:
            raise ValueError(f"reference {self.reference!r} is not a candidate")
        keys = [c.config("-").key() for c in self.candidates]  # also runs the range checks
        if len(set(keys)) != len(keys):
            raise ValueError("two candidates have the same configuration")
        if self.mode == "effect":
            if self.factor is None or self.base is None:
                raise ValueError("an effect plan needs a factor and a base")
            for c in self.candidates:
                expected_parent = None if c.id == self.reference else self.reference
                if c.knob != self.factor or c.parent != expected_parent:
                    raise ValueError(f"{c.id} must be the reference with only {self.factor} changed")
        else:
            roots = [c.model_type for c in self.candidates if c.parent is None]
            if len(set(roots)) != len(roots):
                raise ValueError("each family appears once as a root")
            for c in self.candidates:
                if c.parent is None:
                    continue
                parent = self.get(c.parent)
                if parent.model_type != c.model_type or c.knob not in FAMILIES[c.model_type].refinable:
                    raise ValueError(f"{c.id} must change one refinable knob of its parent")
        return self

    # ----------------------------------------------------------------- lookups
    def get(self, candidate_id: str) -> Candidate:
        for c in self.candidates:
            if c.id == candidate_id:
                return c
        raise KeyError(f"no candidate {candidate_id!r}")

    def candidate_of(self, config: ExperimentConfiguration) -> Optional[Candidate]:
        """Which candidate a configuration (any seed) belongs to."""
        key = config.key()
        return next((c for c in self.candidates if c.config(config.dataset_id).key() == key), None)

    # -------------------------------------------------------------- refinement
    def refine(
        self,
        parent: Optional[str],
        knob: Optional[str],
        values: Sequence[Level],
        round: int,
        contenders: Optional[Sequence[str]] = None,
    ) -> List[Candidate]:
        """The new candidates for ``parent`` with ``knob`` set to each of ``values``.

        Raises ``ValueError`` for any rule a proposal breaks: effect mode keeps
        its factor and parent; selection mode only refines contenders, on a
        knob of the parent's own family; values must be in range; no
        configuration may be run twice.
        """
        if not 1 <= len(values) <= MAX_NEW_CANDIDATES:
            raise ValueError(f"refine needs 1-{MAX_NEW_CANDIDATES} values, got {len(values)}")
        if self.mode == "effect":
            if knob not in (None, self.factor):
                raise ValueError(f"this investigation varies only {self.factor}; cannot change {knob}")
            for v in values:
                check_level(self.factor, v)  # type: ignore[arg-type]
            build = lambda v, **f: _effect_candidate(self.base, self.factor, v, parent=self.reference, **f)  # noqa: E731
        else:
            if parent not in {c.id for c in self.candidates}:
                raise ValueError(f"unknown parent {parent!r}")
            if contenders is not None and parent not in contenders:
                raise ValueError(f"{parent} is not a contender; refine one of {list(contenders)} or conclude")
            source = self.get(parent)
            family = FAMILIES[source.model_type]
            if knob not in family.refinable:
                raise ValueError(f"{source.model_type} can refine only {list(family.refinable)}, not {knob!r}")
            for v in values:
                check_level(knob, v)

            def build(v, **f):
                return Candidate(
                    label=f"{source.label}, {knob}={format_value(v)}", model_type=source.model_type,
                    hyperparameters={**source.hyperparameters, knob: float(v)}, normalize=source.normalize,
                    parent=parent, knob=knob, value=float(v), **f,
                )

        seen = {c.config("-").key() for c in self.candidates}
        new: List[Candidate] = []
        for v in values:
            candidate = build(v, id=f"c{len(self.candidates) + len(new) + 1}", round=round)
            key = candidate.config("-").key()  # runs the range checks
            if key in seen:
                raise ValueError(f"{candidate.label} was already tried; propose untried values or conclude")
            seen.add(key)
            new.append(candidate)
        return new

    def with_candidates(self, new: Sequence[Candidate]) -> "Plan":
        data = self.model_dump()
        data["candidates"] += [c.model_dump() for c in new]
        return Plan.model_validate(data)


class Decision(BaseModel):
    round: int = Field(description="The round whose results this decision was based on")
    action: Literal["refine", "conclude"]
    parent: Optional[str] = None
    knob: Optional[str] = None
    values: List[Level] = Field(default_factory=list)
    new_candidates: List[str] = Field(default_factory=list, description="Ids of the candidates it added")
    rationale: str
    decided_by: Literal["agent", "budget", "settled"] = "agent"


# ---------------------------------------------------------------------------
# Statistics (computed, never stored on their own)
# ---------------------------------------------------------------------------
class ConditionSummary(BaseModel):
    id: str
    label: str
    family: str
    parent: Optional[str]
    change: Optional[str] = Field(description="What differs from the parent, e.g. 'max_depth=12'")
    is_reference: bool
    n_ok: int
    n_failed: int
    mean: Optional[float] = Field(description="Mean metric over the condition's successful runs")
    seed_std: Optional[float] = Field(description="Spread across seeds (stability, not uncertainty)")
    train_metric: Optional[float] = Field(default=None, description="The same metric on the training split")
    gap: Optional[float] = Field(
        default=None, description="How much better it scores on train than on validation (positive = overfitting)"
    )
    contender: bool = Field(default=False, description="Selection mode: not clearly worse than the leader")


class Comparison(BaseModel):
    """Candidate ``a`` vs candidate ``b``: difference in the metric, with a CI."""

    a: str
    label: str = Field(description="Label of a")
    b: str
    against: str = Field(description="Label of b")
    anchor: Literal["reference", "leader", "parent", "runner_up"]
    diff: float = Field(description="metric(a) - metric(b)")
    ci_low: float
    ci_high: float
    confidence: float = 0.95
    verdict: Literal["better", "worse", "inconclusive"]


class Analysis(BaseModel):
    split: Literal["val", "test"]
    metric: Literal["accuracy", "mse"]
    higher_is_better: bool
    mode: Literal["effect", "selection"]
    n_rows: int = Field(description="Evaluation rows each comparison resamples")
    majority_rate: Optional[float] = Field(
        default=None, description="Accuracy of always predicting the most common class"
    )
    leader: Optional[str] = Field(default=None, description="Id of the candidate with the best mean")
    contenders: List[str] = Field(default_factory=list, description="Selection mode: ids still in the race")
    conditions: List[ConditionSummary]
    comparisons: List[Comparison]


class ScoreCI(BaseModel):
    value: float
    ci_low: float
    ci_high: float


class Report(BaseModel):
    mode: Literal["effect", "selection"]
    winner: str = Field(description="Label of the best non-reference candidate on validation")
    winner_id: str
    reference: str
    runner_up: Optional[str] = Field(default=None, description="Selection: best candidate of another family")
    metric: Literal["accuracy", "mse"]
    higher_is_better: bool
    winner_score: ScoreCI = Field(description="Test-split metric of the winner")
    reference_score: ScoreCI
    primary: Comparison = Field(description="Test split: winner vs reference")
    secondary: Optional[Comparison] = Field(default=None, description="Test split: winner vs runner-up")
    confidence: float = Field(description="CI level of each test comparison (Bonferroni over the comparisons)")
    winner_val_score: float
    val_to_test_drop: float = Field(
        description="How much worse the winner scored on test than on validation (positive = selection optimism)"
    )
    effort: Dict[str, int] = Field(description="Candidates tried per family")
    candidates_tried: int
    majority_rate: Optional[float] = None
    n_test_rows: int
    rounds_run: int
    stopped_by: Literal["agent", "budget", "settled"]
    headline: str = Field(description="Deterministic answer")
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
