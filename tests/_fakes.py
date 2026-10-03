"""
tests/_fakes.py
=================
Deterministic stand-ins for the LLM and for training, so the loop and the API
can be tested without network calls or real model fitting.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np

from backend.models.dataset import DatasetProfile
from backend.models.experiment import ExperimentConfiguration, ExperimentResult
from backend.models.investigation import BaseSetup, Decision, Plan

N_ROWS = 200


def make_profile(dataset_id: str = "ds-1", task_type: str = "classification") -> DatasetProfile:
    common = dict(
        dataset_id=dataset_id,
        original_filename="fixture.csv",
        storage_path="unused.csv",
        target_column="target",
        feature_columns=["a", "b"],
        numeric_columns=["a", "b"],
        categorical_columns=[],
        n_rows=1000,
        n_features=2,
        missing_value_counts={},
        split_seed=42,
    )
    if task_type == "classification":
        return DatasetProfile(
            task_type="classification", n_classes=2, class_labels=["0", "1"],
            class_distribution={"0": 600, "1": 400}, **common,
        )
    return DatasetProfile(task_type="regression", **common)


def dropout_plan(levels=(0.0, 0.5), reference=0.0) -> Plan:
    return Plan.effect(
        factor="dropout",
        levels=list(levels),
        reference=reference,
        base=BaseSetup(hyperparameters={"hidden_size": 16, "epochs": 2}),
        rationale="Vary dropout, everything else fixed.",
    )


def selection_plan(families=("linear_baseline", "decision_tree", "random_forest", "mlp")) -> Plan:
    return Plan.selection(list(families), rationale="Compare the families at their defaults.")


def scores_with_accuracy(accuracy: float, seed: int, n: int = N_ROWS) -> List[float]:
    """Per-row 0/1 scores with (about) the given accuracy."""
    rng = np.random.default_rng(seed)
    return (rng.random(n) < accuracy).astype(float).tolist()


def fake_run(accuracy_of=lambda cfg: 0.6 + cfg.hyperparameters.get("dropout", 0.0) * 0.4,
             seed_of=lambda cfg: cfg.random_seed):
    """A ``run`` function for the graph: no training, accuracy decided by ``accuracy_of``.

    Runs with the same ``seed_of`` share their random draws, so their scores
    differ only where their accuracies differ.
    """

    def run(config: ExperimentConfiguration, profile: DatasetProfile, session_id: str, round: int):
        acc = accuracy_of(config)
        val = scores_with_accuracy(acc, seed=seed_of(config))
        test = scores_with_accuracy(acc, seed=seed_of(config) + 1000)
        return ExperimentResult(
            session_id=session_id, round=round, config=config, status="ok",
            metrics={"accuracy": float(np.mean(val)), "train_accuracy": min(1.0, acc + 0.05),
                     "train_loss": 0.5, "training_time_seconds": 0.01},
            val_scores=val, test_scores=test,
        )

    return run


class FakePlanner:
    def __init__(self, plan: Optional[Plan] = None, error: Optional[Exception] = None) -> None:
        self._plan = plan or dropout_plan()
        self._error = error
        self.calls = 0

    def plan(self, question: str, profile: DatasetProfile) -> Plan:
        self.calls += 1
        if self._error:
            raise self._error
        return self._plan


class FakeRecommender:
    """Refines with the scripted steps in order, then concludes.

    A step is a list of values (effect mode: new levels of the factor) or a
    ``(parent, knob, values)`` tuple (selection mode).
    """

    def __init__(self, refine: Optional[List[Any]] = None, interpretation: Any = "It helps.") -> None:
        self._refine = list(refine or [])
        self._interpretation = interpretation
        self.decide_calls: List[int] = []
        self.seen: List[Any] = []
        self.interpret_calls = 0

    def decide(self, question, plan, analysis, round, max_rounds) -> Decision:
        self.decide_calls.append(round)
        self.seen.append(analysis)
        if self._refine:
            step = self._refine.pop(0)
            parent, knob, values = step if isinstance(step, tuple) else (plan.reference, plan.factor, step)
            return Decision(round=round, action="refine", parent=parent, knob=knob, values=list(values),
                            rationale="Try more.")
        return Decision(round=round, action="conclude", rationale="Enough evidence.")

    def interpret(self, question, plan, report) -> str:
        self.interpret_calls += 1
        if isinstance(self._interpretation, Exception):
            raise self._interpretation
        return self._interpretation


class StubLLM:
    """Returns scripted JSON replies in order and records every request."""

    def __init__(self, *replies: Dict[str, Any]) -> None:
        self._replies = [json.dumps(r) for r in replies]
        self.requests: List[List[Dict[str, str]]] = []

    def chat_json(self, messages, schema, temperature=0.2, **_options) -> str:
        self.requests.append(messages)
        return self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]


def make_test_client(repo, investigator=None):
    """A TestClient wired to ``repo``; ``investigator`` replaces the background run."""
    from fastapi.testclient import TestClient

    from backend.api.app import create_app
    from backend.api.dependencies import get_investigator, get_repository

    app = create_app()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_investigator] = lambda: investigator or (lambda session_id: None)
    return TestClient(app, raise_server_exceptions=False)
