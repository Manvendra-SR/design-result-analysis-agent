"""Tests for the trainers and run_experiment, on a small real dataset."""

from __future__ import annotations

import numpy as np
import pytest

from backend.models.experiment import ExperimentConfiguration
from backend.tools import experiment_runner, trainers
from backend.tools.dataset.dataset import Dataset


def _cfg(profile, model_type="mlp", seed=0, **hp) -> ExperimentConfiguration:
    hp = hp or ({"hidden_size": 8, "epochs": 3} if model_type == "mlp" else {})
    return ExperimentConfiguration(dataset_id=profile.dataset_id, model_type=model_type,
                                   hyperparameters=hp, random_seed=seed)


@pytest.mark.parametrize("model_type", ["mlp", "linear_baseline", "decision_tree", "random_forest"])
def test_every_family_scores_every_validation_and_test_row(real_profile, model_type) -> None:
    split = Dataset(real_profile).load_split()
    out = trainers.TRAINERS[model_type](_cfg(real_profile, model_type), real_profile)

    assert len(out.val_scores) == len(split.y_val)
    assert len(out.test_scores) == len(split.y_test)
    assert set(out.val_scores) <= {0.0, 1.0}
    assert out.metrics["accuracy"] == pytest.approx(np.mean(out.val_scores))
    assert set(out.metrics) == {"accuracy", "train_accuracy", "train_loss", "training_time_seconds"}
    assert 0.0 <= out.metrics["train_accuracy"] <= 1.0


def test_the_baseline_learns_the_signal(real_profile) -> None:
    out = trainers.train_linear_baseline(_cfg(real_profile, "linear_baseline"), real_profile)
    assert out.metrics["accuracy"] > 0.75


def test_tree_knobs_reach_the_model(real_profile) -> None:
    stump = trainers.train_decision_tree(_cfg(real_profile, "decision_tree", max_depth=1), real_profile)
    deep = trainers.train_decision_tree(_cfg(real_profile, "decision_tree", max_depth=30), real_profile)
    assert deep.metrics["train_accuracy"] > stump.metrics["train_accuracy"]  # a deeper tree fits train better
    assert deep.metrics["train_accuracy"] == pytest.approx(1.0)


def test_the_decision_tree_ignores_the_seed(real_profile) -> None:
    a = trainers.train_decision_tree(_cfg(real_profile, "decision_tree", seed=0), real_profile)
    b = trainers.train_decision_tree(_cfg(real_profile, "decision_tree", seed=5), real_profile)
    assert a.val_scores == b.val_scores


def test_the_random_forest_is_reproducible_per_seed_and_uses_it(real_profile) -> None:
    cfg = lambda seed: _cfg(real_profile, "random_forest", seed=seed, max_depth=3, max_features=0.5)  # noqa: E731
    first = trainers.train_random_forest(cfg(1), real_profile)
    assert first.val_scores == trainers.train_random_forest(cfg(1), real_profile).val_scores
    probs = {tuple(trainers.train_random_forest(cfg(s), real_profile).val_scores) for s in range(4)}
    assert len(probs) > 1  # bootstrap samples follow the seed


def test_mlp_is_reproducible_per_seed(real_profile) -> None:
    first = trainers.train_mlp(_cfg(real_profile, seed=4), real_profile)
    again = trainers.train_mlp(_cfg(real_profile, seed=4), real_profile)
    assert first.val_scores == again.val_scores


def test_run_experiment_success(real_profile) -> None:
    result = experiment_runner.run_experiment(_cfg(real_profile, "linear_baseline"), real_profile, "s", 2)
    assert result.status == "ok" and result.round == 2 and result.error is None
    assert result.val_scores and result.test_scores


def test_a_crash_becomes_a_failed_run(real_profile, monkeypatch) -> None:
    def boom(config, profile):
        raise RuntimeError("shape mismatch")

    monkeypatch.setitem(trainers.TRAINERS, "mlp", boom)
    monkeypatch.setattr(experiment_runner, "TRAINERS", trainers.TRAINERS)
    result = experiment_runner.run_experiment(_cfg(real_profile), real_profile, "s", 1)
    assert result.status == "failed" and "shape mismatch" in result.error
    assert result.metrics is None


def test_a_non_finite_result_becomes_a_failed_run(real_profile, monkeypatch) -> None:
    def nan_loss(config, profile):
        return trainers.TrainOutput({"accuracy": 0.5, "train_loss": float("nan"),
                                     "training_time_seconds": 1.0}, [1.0], [0.0])

    monkeypatch.setitem(trainers.TRAINERS, "mlp", nan_loss)
    result = experiment_runner.run_experiment(_cfg(real_profile), real_profile, "s", 1)
    assert result.status == "failed" and "non-finite" in result.error
