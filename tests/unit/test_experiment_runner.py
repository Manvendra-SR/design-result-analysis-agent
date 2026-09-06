"""
tests/unit/test_experiment_runner.py
=======================================
Unit tests for backend/tools/experiment_runner.py.

Actual training (torch/scikit-learn) is always mocked here — these tests
verify dispatch, error handling, and retry logic only. Real end-to-end
training against a real ingested dataset is exercised by
scripts/checkpoint_3_18.py.

Test cases
----------
1. test_run_experiment_mlp_dispatches_to_mlp_trainer
2. test_run_experiment_linear_baseline_dispatches_to_baseline_trainer
3. test_run_experiment_unknown_model_type_returns_failed
4. test_experiment_configuration_rejects_invalid_hyperparameters
5. test_run_experiment_permanent_error_not_retried
6. test_run_experiment_retries_transient_error_then_succeeds
7. test_run_experiment_retries_exhausted_returns_failed
8. test_run_batch_continues_after_failure
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from backend.models.dataset import DatasetProfile
from backend.models.experiment import ExperimentConfiguration, ExperimentResult
from backend.tools.experiment_runner import ExperimentRunner


def _dataset_profile() -> DatasetProfile:
    return DatasetProfile(
        dataset_id="ds-fixture",
        original_filename="fixture.csv",
        storage_path="unused-in-these-tests.csv",
        target_column="target",
        feature_columns=["a", "b"],
        numeric_columns=["a", "b"],
        categorical_columns=[],
        task_type="classification",
        n_rows=100,
        n_features=2,
        n_classes=2,
        class_labels=["0", "1"],
        class_distribution={"0": 50, "1": 50},
        missing_value_counts={},
        split_seed=42,
    )


def _mlp_cfg(seed: int = 42) -> ExperimentConfiguration:
    return ExperimentConfiguration(
        dataset_id="ds-fixture",
        model_type="mlp",
        hyperparameters={"dropout": 0.2, "learning_rate": 0.001, "batch_size": 32},
        random_seed=seed,
    )


def _linear_baseline_cfg(seed: int = 42) -> ExperimentConfiguration:
    return ExperimentConfiguration(
        dataset_id="ds-fixture",
        model_type="linear_baseline",
        random_seed=seed,
    )


def _success_result(cfg: ExperimentConfiguration, profile: DatasetProfile, session_id: str) -> ExperimentResult:
    return ExperimentResult(
        session_id=session_id,
        config=cfg,
        task_type=profile.task_type,
        metrics={"train_loss": 0.1, "val_loss": 0.12, "accuracy": 0.95,
                 "n_classes": 2, "n_val_samples": 20, "training_time_seconds": 1.0},
        status="success",
    )


# ---------------------------------------------------------------------------
# 1 & 2. Dispatch
# ---------------------------------------------------------------------------

def test_run_experiment_mlp_dispatches_to_mlp_trainer() -> None:
    cfg = _mlp_cfg()
    profile = _dataset_profile()
    mock_trainer = MagicMock(
        side_effect=lambda c, p, session_id="": _success_result(c, p, session_id)
    )

    with patch.dict(ExperimentRunner._TRAINERS, {"mlp": mock_trainer}):
        result = ExperimentRunner().run_experiment(cfg, profile, session_id="sess-1")

    mock_trainer.assert_called_once()
    assert result.status == "success"
    assert result.session_id == "sess-1"


def test_run_experiment_linear_baseline_dispatches_to_baseline_trainer() -> None:
    cfg = _linear_baseline_cfg()
    profile = _dataset_profile()
    mock_trainer = MagicMock(
        side_effect=lambda c, p, session_id="": _success_result(c, p, session_id)
    )

    with patch.dict(ExperimentRunner._TRAINERS, {"linear_baseline": mock_trainer}):
        result = ExperimentRunner().run_experiment(cfg, profile)

    mock_trainer.assert_called_once()
    assert result.status == "success"


# ---------------------------------------------------------------------------
# 3. Unknown model_type (defensive branch - unreachable via normal validated
#    configs since model_type is a Pydantic Literal, but a config could be
#    mutated after construction, e.g. by a future caller)
# ---------------------------------------------------------------------------

def test_run_experiment_unknown_model_type_returns_failed() -> None:
    cfg = _mlp_cfg()
    cfg.model_type = "unknown_model"  # bypass Literal validation (no validate_assignment)

    result = ExperimentRunner().run_experiment(cfg, _dataset_profile())

    assert result.status == "failed"
    assert "unknown_model" in result.error


# ---------------------------------------------------------------------------
# 4. Invalid hyperparameters (Phase 2 validator, exercised from Phase 3's
#    perspective: ExperimentRunner relies on this happening before it's called)
# ---------------------------------------------------------------------------

def test_experiment_configuration_rejects_invalid_hyperparameters() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfiguration(
            dataset_id="ds-fixture",
            model_type="mlp",
            hyperparameters={"dropout": 1.5, "learning_rate": 0.001, "batch_size": 32},
            random_seed=1,
        )


# ---------------------------------------------------------------------------
# 5. Permanent error - no retry
# ---------------------------------------------------------------------------

def test_run_experiment_permanent_error_not_retried() -> None:
    cfg = _mlp_cfg()
    mock_trainer = MagicMock(side_effect=ValueError("bad hyperparameter combination"))

    with patch.dict(ExperimentRunner._TRAINERS, {"mlp": mock_trainer}):
        result = ExperimentRunner().run_experiment(cfg, _dataset_profile())

    assert result.status == "failed"
    assert "bad hyperparameter combination" in result.error
    assert mock_trainer.call_count == 1  # no retry for a permanent error


# ---------------------------------------------------------------------------
# 6. Transient error - retried, then succeeds
# ---------------------------------------------------------------------------

def test_run_experiment_retries_transient_error_then_succeeds() -> None:
    cfg = _mlp_cfg()
    profile = _dataset_profile()
    call_count = {"n": 0}

    def flaky(c, p, session_id=""):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise RuntimeError("CUDA out of memory")
        return _success_result(c, p, session_id)

    mock_trainer = MagicMock(side_effect=flaky)

    with patch.dict(ExperimentRunner._TRAINERS, {"mlp": mock_trainer}):
        result = ExperimentRunner().run_experiment(cfg, profile)

    assert result.status == "success"
    assert call_count["n"] == 3


# ---------------------------------------------------------------------------
# 7. Transient error - retries exhausted -> failed
# ---------------------------------------------------------------------------

def test_run_experiment_retries_exhausted_returns_failed() -> None:
    cfg = _mlp_cfg()
    mock_trainer = MagicMock(side_effect=RuntimeError("CUDA out of memory"))

    with patch.dict(ExperimentRunner._TRAINERS, {"mlp": mock_trainer}):
        result = ExperimentRunner().run_experiment(cfg, _dataset_profile())

    assert result.status == "failed"
    assert "out of memory" in result.error.lower()
    assert mock_trainer.call_count == 3


# ---------------------------------------------------------------------------
# 8. run_batch continues past a failure
# ---------------------------------------------------------------------------

def test_run_batch_continues_after_failure() -> None:
    profile = _dataset_profile()
    good_cfg = _mlp_cfg(seed=1)
    bad_cfg = _mlp_cfg(seed=2)

    def trainer(c, p, session_id=""):
        if c.random_seed == 2:
            raise ValueError("boom")
        return _success_result(c, p, session_id)

    mock_trainer = MagicMock(side_effect=trainer)

    with patch.dict(ExperimentRunner._TRAINERS, {"mlp": mock_trainer}):
        results = ExperimentRunner().run_batch([good_cfg, bad_cfg], profile)

    assert len(results) == 2
    assert results[0].status == "success"
    assert results[1].status == "failed"
    assert "boom" in results[1].error
