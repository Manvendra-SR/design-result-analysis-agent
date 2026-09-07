"""
tests/_fakes.py
=================
Deterministic stand-ins for the LLM agents and the experiment runner, shared
by the Phase 5 (state machine) and Phase 6 (API) tests.

The state machine and the API are the units under test in those suites, so
the planner / recommender / runner are faked - they have their own suites
(``test_planner.py``, ``test_recommender.py``, ``test_experiment_runner.py``,
``checkpoint_3_18.py``). The anomaly detector and statistical analyzer are
real (they are pure and fast).
"""

from __future__ import annotations

from typing import List, Optional

from backend.models.anomaly import AnomalyReport
from backend.models.dataset import DatasetProfile
from backend.models.experiment import (
    ExperimentConfiguration,
    ExperimentPlan,
    ExperimentResult,
)
from backend.models.recommendation import Recommendation
from backend.models.statistics import StatisticalComparison
from backend.state_machine.context import StateMachineContext
from backend.tools.anomaly_detector import AnomalyDetector
from backend.tools.statistical_analyzer import StatisticalAnalyzer
from backend.tools.state_manager import StateManager


def make_dataset_profile(dataset_id: str = "ds-test", task_type: str = "classification") -> DatasetProfile:
    common = dict(
        dataset_id=dataset_id,
        original_filename="fixture.csv",
        storage_path="unused-in-fakes.csv",
        target_column="target",
        feature_columns=["a", "b"],
        numeric_columns=["a", "b"],
        categorical_columns=[],
        n_rows=200,
        n_features=2,
        missing_value_counts={},
        split_seed=42,
    )
    if task_type == "classification":
        return DatasetProfile(
            task_type="classification",
            n_classes=2,
            class_labels=["0", "1"],
            class_distribution={"0": 100, "1": 100},
            **common,
        )
    return DatasetProfile(task_type="regression", **common)


#: Default seeds per condition. Four (not the minimum three) so that a
#: linearly-spaced stub metric across seeds never produces a leave-one-out
#: "outlier" - with three collinear points the extreme one sits exactly at
#: the 3-sigma boundary and floating-point noise decides. Four is clean.
DEFAULT_SEEDS = (1, 2, 3, 4)


class StubPlanner:
    """Returns a fixed 2-condition x 4-seed plan (or a scripted sequence of plans)."""

    def __init__(self, *plans: ExperimentPlan) -> None:
        self._plans = list(plans)
        self.calls = 0

    def plan_experiments(self, research_question: str, dataset_profile: DatasetProfile) -> ExperimentPlan:
        self.calls += 1
        if self._plans:
            return self._plans.pop(0) if len(self._plans) > 1 else self._plans[0]
        experiments = [
            ExperimentConfiguration(
                dataset_id=dataset_profile.dataset_id,
                model_type="mlp",
                hyperparameters={"dropout": dropout},
                random_seed=seed,
            )
            for dropout in (0.0, 0.5)
            for seed in DEFAULT_SEEDS
        ]
        return ExperimentPlan(experiments=experiments, explanation="stub: vary dropout")


class StubRunner:
    """Produces deterministic ExperimentResults: dropout>=0.25 -> better metric.

    Within a condition the metric moves linearly with ``random_seed`` (a
    small, well-separated spread) so the statistical analyzer sees real
    variance but the anomaly detector sees no outliers.
    """

    def __init__(self, task_type: str = "classification", n_val_samples: int = 200) -> None:
        self._task_type = task_type
        self._n_val = n_val_samples

    def run_experiment(self, config: ExperimentConfiguration, dataset_profile: DatasetProfile, session_id: str = "") -> ExperimentResult:
        dropout = float(config.hyperparameters.get("dropout", 0.0))
        spread = 0.002 * (config.random_seed - 1)  # 0.000, 0.002, 0.004, ...
        if self._task_type == "classification":
            accuracy = (0.78 if dropout < 0.25 else 0.90) + spread
            metrics = {
                "train_loss": 0.20,
                "val_loss": round(1.0 - accuracy, 4),
                "accuracy": round(accuracy, 4),
                "n_classes": 2.0,
                "n_val_samples": float(self._n_val),
                "training_time_seconds": 0.01,
                "initial_train_loss": 0.35,
            }
        else:
            val_loss = (0.50 if dropout < 0.25 else 0.20) - spread
            metrics = {
                "train_loss": round(val_loss - 0.05, 4),
                "val_loss": round(val_loss, 4),
                "training_time_seconds": 0.01,
                "initial_train_loss": 0.60,
            }
        return ExperimentResult(
            session_id=session_id,
            config=config,
            task_type=self._task_type,  # type: ignore[arg-type]
            metrics=metrics,
            status="success",
        )

    def run_batch(self, configs, dataset_profile, session_id: str = "") -> List[ExperimentResult]:
        return [self.run_experiment(c, dataset_profile, session_id=session_id) for c in configs]


class StubRecommender:
    """Scripted recommender.

    ``recommend_next`` returns each response in ``responses`` in turn (the
    last one repeats). Also records the arguments it was called with, so
    tests can assert it received pre-computed statistics rather than raw
    numbers to crunch.
    """

    def __init__(self, *responses: Recommendation) -> None:
        self._responses = list(responses) or [
            Recommendation(
                action="conclude",
                recommended_experiments=[],
                explanation="stub: sufficient evidence",
                evidence_summary="stub",
            )
        ]
        self.received: list[dict] = []

    def recommend_next(
        self,
        research_question: str,
        experiments: List[ExperimentResult],
        statistical_results: Optional[List[StatisticalComparison]] = None,
        anomalies: Optional[List[AnomalyReport]] = None,
    ) -> Recommendation:
        self.received.append(
            {
                "research_question": research_question,
                "experiments": experiments,
                "statistical_results": statistical_results,
                "anomalies": anomalies,
            }
        )
        return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]


def run_more_then_conclude(dataset_id: str = "ds-test") -> StubRecommender:
    """A recommender that asks for one more round, then concludes."""
    more = Recommendation(
        action="run_more_experiments",
        recommended_experiments=[
            ExperimentConfiguration(
                dataset_id=dataset_id,
                model_type="mlp",
                hyperparameters={"dropout": 0.25},
                random_seed=seed,
            )
            for seed in (5, 6, 7, 8)
        ],
        explanation="stub: explore dropout=0.25",
        evidence_summary="stub",
    )
    done = Recommendation(
        action="conclude",
        recommended_experiments=[],
        explanation="stub: conclusive",
        evidence_summary="stub",
    )
    return StubRecommender(more, done)


def _run_more(dataset_id: str, seeds=(5, 6, 7)) -> Recommendation:
    return Recommendation(
        action="run_more_experiments",
        recommended_experiments=[
            ExperimentConfiguration(
                dataset_id=dataset_id,
                model_type="mlp",
                hyperparameters={"dropout": 0.25},
                random_seed=seed,
            )
            for seed in seeds
        ],
        explanation="stub: explore dropout=0.25",
        evidence_summary="stub: not conclusive yet",
    )


def run_more_n_times_then_conclude(n: int, dataset_id: str = "ds-test") -> StubRecommender:
    """A recommender that asks for ``n`` more rounds, then concludes."""
    responses = [_run_more(dataset_id) for _ in range(n)]
    responses.append(
        Recommendation(
            action="conclude",
            recommended_experiments=[],
            explanation="stub: conclusive",
            evidence_summary="stub",
        )
    )
    return StubRecommender(*responses)


def always_run_more(dataset_id: str = "ds-test") -> StubRecommender:
    """A recommender that NEVER concludes - used to test the safety cap."""
    return StubRecommender(_run_more(dataset_id))  # single response, repeats forever


def build_context(
    state_manager: StateManager,
    *,
    planner: Optional[object] = None,
    recommender: Optional[object] = None,
    runner: Optional[object] = None,
    task_type: str = "classification",
) -> StateMachineContext:
    return StateMachineContext(
        state_manager=state_manager,
        planner=planner or StubPlanner(),
        recommender=recommender or StubRecommender(),
        runner=runner or StubRunner(task_type=task_type),
        detector=AnomalyDetector(),
        analyzer=StatisticalAnalyzer(),
    )


def make_test_client(state_manager: StateManager, **context_kwargs):
    """Build a FastAPI ``TestClient`` whose ``get_context`` is a stubbed context.

    Returns ``(client, context)``. Not used as a context manager, so the
    app's lifespan (its startup DB check) does not fire during tests.
    """
    from fastapi.testclient import TestClient

    from backend.api.app import create_app
    from backend.api.dependencies import get_context

    context = build_context(state_manager, **context_kwargs)
    app = create_app()
    app.dependency_overrides[get_context] = lambda: context
    # raise_server_exceptions=False so a deliberate 500 is asserted on the
    # response rather than re-raised into the test.
    return TestClient(app, raise_server_exceptions=False), context
