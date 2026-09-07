"""
tests/unit/test_state_manager_cycles.py
=========================================
Unit tests for the Phase 5 history accessors added to StateManager:
``experiments.cycle`` round-trip, ``save_plan_explanation``,
``append_cycle_history`` / ``get_cycle_history``.

Uses the shared ``state_manager`` fixture (in-memory SQLite).
"""

from __future__ import annotations

import uuid

import pytest

from backend.models.cycle import CycleHistoryEntry
from backend.models.experiment import ExperimentConfiguration, ExperimentResult
from backend.models.recommendation import Recommendation
from backend.models.statistics import StatisticalComparison
from tests._fakes import make_dataset_profile


@pytest.fixture
def session_id(state_manager) -> str:
    state_manager.create_dataset(make_dataset_profile())
    return state_manager.create_session("q?", "ds-test")


def _result(session_id: str, cycle: int, seed: int) -> ExperimentResult:
    return ExperimentResult(
        session_id=session_id,
        config=ExperimentConfiguration(
            dataset_id="ds-test", model_type="mlp",
            hyperparameters={"dropout": 0.1}, random_seed=seed,
        ),
        task_type="classification",
        metrics={"train_loss": 0.1, "val_loss": 0.12, "accuracy": 0.9,
                 "n_classes": 2, "n_val_samples": 100, "training_time_seconds": 1.0},
        status="success",
        cycle=cycle,
    )


def _entry(n: int, action: str = "run_more_experiments") -> CycleHistoryEntry:
    return CycleHistoryEntry(
        cycle_number=n,
        recommendation=Recommendation(
            action=action,
            recommended_experiments=[],
            explanation=f"cycle {n} reasoning",
            evidence_summary="s",
        ),
        statistical_comparisons=[
            StatisticalComparison(
                condition_a_name="a", condition_b_name="b", metric="accuracy",
                t_statistic=1.0, p_value=0.5, effect_size=0.1,
                confidence_interval=(0.0, 1.0), sample_sizes=(3, 3),
            )
        ],
    )


def test_experiment_cycle_round_trips(state_manager, session_id) -> None:
    state_manager.store_experiment(_result(session_id, cycle=1, seed=1))
    state_manager.store_experiment(_result(session_id, cycle=2, seed=2))

    got = {e.config.random_seed: e.cycle for e in state_manager.query_experiments(session_id)}
    assert got == {1: 1, 2: 2}


def test_experiment_cycle_defaults_to_none(state_manager, session_id) -> None:
    r = _result(session_id, cycle=1, seed=9)
    r.cycle = None
    state_manager.store_experiment(r)
    assert state_manager.query_experiments(session_id)[0].cycle is None


def test_save_and_read_plan_explanation(state_manager, session_id) -> None:
    assert state_manager.get_session(session_id).plan_explanation is None
    state_manager.save_plan_explanation(session_id, "vary dropout with 3 seeds each")
    assert state_manager.get_session(session_id).plan_explanation == "vary dropout with 3 seeds each"


def test_cycle_history_appends_and_orders(state_manager, session_id) -> None:
    assert state_manager.get_cycle_history(session_id) == []

    state_manager.append_cycle_history(session_id, _entry(1))
    state_manager.append_cycle_history(session_id, _entry(2))
    state_manager.append_cycle_history(session_id, _entry(3, action="conclude"))

    history = state_manager.get_cycle_history(session_id)
    assert [h.cycle_number for h in history] == [1, 2, 3]
    assert history[0].recommendation.explanation == "cycle 1 reasoning"
    assert history[2].recommendation.action == "conclude"
    assert all(len(h.statistical_comparisons) == 1 for h in history)


def test_cycle_history_missing_session_raises(state_manager) -> None:
    with pytest.raises(KeyError):
        state_manager.append_cycle_history(str(uuid.uuid4()), _entry(1))
