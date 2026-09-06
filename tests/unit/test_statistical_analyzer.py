"""
tests/unit/test_statistical_analyzer.py
==========================================
Unit tests for backend/tools/statistical_analyzer.py.

Test cases (task 3.12)
------------------------
1. test_compare_conditions_matches_scipy_ttest
2. test_compare_conditions_cohens_d_matches_manual_formula
3. test_compare_conditions_confidence_interval_matches_scipy
4. test_compare_conditions_insufficient_data_raises
5. test_compare_conditions_insufficient_variance_raises
6. test_compare_conditions_excludes_anomalous_and_failed
7. test_compare_conditions_underpowered_warning
8. test_compare_conditions_default_labels_are_inferred
9. test_compute_summary_statistics_matches_numpy
10. test_compute_summary_statistics_no_data_raises
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest
from scipy import stats

from backend.models.experiment import ExperimentConfiguration, ExperimentResult
from backend.tools.statistical_analyzer import (
    InsufficientDataError,
    InsufficientVarianceError,
    StatisticalAnalyzer,
)


def _cfg(dropout: float, seed: int) -> ExperimentConfiguration:
    return ExperimentConfiguration(
        dataset_id="ds-fixture",
        model_type="mlp",
        hyperparameters={"dropout": dropout, "learning_rate": 0.001, "batch_size": 32},
        random_seed=seed,
    )


def _exp(
    accuracy: float,
    dropout: float = 0.0,
    seed: int = 0,
    status: str = "success",
) -> ExperimentResult:
    return ExperimentResult(
        session_id="s",
        config=_cfg(dropout, seed),
        task_type="classification" if status != "failed" else None,
        metrics={"train_loss": 0.1, "val_loss": 0.1, "accuracy": accuracy,
                 "n_classes": 2, "n_val_samples": 100,
                 "training_time_seconds": 1.0} if status != "failed" else None,
        status=status,  # type: ignore[arg-type]
        error="boom" if status == "failed" else None,
    )


CONDITION_A_ACC = [0.90, 0.91, 0.89, 0.92, 0.88]
CONDITION_B_ACC = [0.95, 0.96, 0.94, 0.97, 0.93]


def _condition(values: List[float], dropout: float) -> List[ExperimentResult]:
    return [_exp(v, dropout=dropout, seed=i) for i, v in enumerate(values)]


# ---------------------------------------------------------------------------
# 1. Matches scipy's t-test directly
# ---------------------------------------------------------------------------

def test_compare_conditions_matches_scipy_ttest() -> None:
    cond_a = _condition(CONDITION_A_ACC, dropout=0.0)
    cond_b = _condition(CONDITION_B_ACC, dropout=0.5)

    result = StatisticalAnalyzer().compare_conditions(cond_a, cond_b)

    expected_t, expected_p = stats.ttest_ind(CONDITION_A_ACC, CONDITION_B_ACC)
    assert result.t_statistic == pytest.approx(expected_t)
    assert result.p_value == pytest.approx(expected_p)
    assert result.sample_sizes == (5, 5)
    assert result.warning is None  # n=5 per condition, not underpowered


# ---------------------------------------------------------------------------
# 2. Cohen's d matches the manual pooled-std formula
# ---------------------------------------------------------------------------

def test_compare_conditions_cohens_d_matches_manual_formula() -> None:
    cond_a = _condition(CONDITION_A_ACC, dropout=0.0)
    cond_b = _condition(CONDITION_B_ACC, dropout=0.5)

    result = StatisticalAnalyzer().compare_conditions(cond_a, cond_b)

    mean_a, mean_b = np.mean(CONDITION_A_ACC), np.mean(CONDITION_B_ACC)
    pooled_std = np.sqrt((np.var(CONDITION_A_ACC) + np.var(CONDITION_B_ACC)) / 2)
    expected_effect_size = (mean_a - mean_b) / pooled_std

    assert result.effect_size == pytest.approx(expected_effect_size)


# ---------------------------------------------------------------------------
# 3. Confidence interval matches scipy.stats.t.interval
# ---------------------------------------------------------------------------

def test_compare_conditions_confidence_interval_matches_scipy() -> None:
    cond_a = _condition(CONDITION_A_ACC, dropout=0.0)
    cond_b = _condition(CONDITION_B_ACC, dropout=0.5)

    result = StatisticalAnalyzer().compare_conditions(cond_a, cond_b)

    n_a, n_b = len(CONDITION_A_ACC), len(CONDITION_B_ACC)
    mean_a, mean_b = np.mean(CONDITION_A_ACC), np.mean(CONDITION_B_ACC)
    pooled_std = np.sqrt((np.var(CONDITION_A_ACC) + np.var(CONDITION_B_ACC)) / 2)
    se = pooled_std * np.sqrt(1 / n_a + 1 / n_b)
    expected_ci = stats.t.interval(0.95, n_a + n_b - 2, loc=mean_a - mean_b, scale=se)

    assert result.confidence_interval[0] == pytest.approx(expected_ci[0])
    assert result.confidence_interval[1] == pytest.approx(expected_ci[1])


# ---------------------------------------------------------------------------
# 4. Insufficient data (< 2 successful experiments in a condition)
# ---------------------------------------------------------------------------

def test_compare_conditions_insufficient_data_raises() -> None:
    cond_a = _condition([0.9], dropout=0.0)  # only 1 experiment
    cond_b = _condition(CONDITION_B_ACC, dropout=0.5)

    with pytest.raises(InsufficientDataError):
        StatisticalAnalyzer().compare_conditions(cond_a, cond_b)


# ---------------------------------------------------------------------------
# 5. Insufficient variance (identical values -> std ~ 0)
# ---------------------------------------------------------------------------

def test_compare_conditions_insufficient_variance_raises() -> None:
    cond_a = _condition([0.9, 0.9, 0.9], dropout=0.0)
    cond_b = _condition(CONDITION_B_ACC, dropout=0.5)

    with pytest.raises(InsufficientVarianceError):
        StatisticalAnalyzer().compare_conditions(cond_a, cond_b)


# ---------------------------------------------------------------------------
# 6. Anomalous / failed experiments excluded
# ---------------------------------------------------------------------------

def test_compare_conditions_excludes_anomalous_and_failed() -> None:
    cond_a = _condition(CONDITION_A_ACC, dropout=0.0) + [
        _exp(0.01, dropout=0.0, seed=99, status="anomalous"),
        _exp(0.0, dropout=0.0, seed=100, status="failed"),
    ]
    cond_b = _condition(CONDITION_B_ACC, dropout=0.5)

    result = StatisticalAnalyzer().compare_conditions(cond_a, cond_b)

    # Only the 5 "success" experiments in cond_a should count.
    assert result.sample_sizes == (5, 5)


# ---------------------------------------------------------------------------
# 7. Underpowered warning (n < 5 per condition)
# ---------------------------------------------------------------------------

def test_compare_conditions_underpowered_warning() -> None:
    cond_a = _condition([0.90, 0.91, 0.89], dropout=0.0)
    cond_b = _condition([0.95, 0.96, 0.94], dropout=0.5)

    result = StatisticalAnalyzer().compare_conditions(cond_a, cond_b)

    assert result.warning == "underpowered"
    assert result.sample_sizes == (3, 3)


# ---------------------------------------------------------------------------
# 8. Default condition labels are inferred from config when not supplied
# ---------------------------------------------------------------------------

def test_compare_conditions_default_labels_are_inferred() -> None:
    cond_a = _condition(CONDITION_A_ACC, dropout=0.0)
    cond_b = _condition(CONDITION_B_ACC, dropout=0.5)

    result = StatisticalAnalyzer().compare_conditions(cond_a, cond_b)

    assert "dropout=0.0" in result.condition_a_name
    assert "dropout=0.5" in result.condition_b_name

    named = StatisticalAnalyzer().compare_conditions(
        cond_a, cond_b, condition_a_name="baseline", condition_b_name="treatment"
    )
    assert named.condition_a_name == "baseline"
    assert named.condition_b_name == "treatment"


# ---------------------------------------------------------------------------
# 9. compute_summary_statistics matches numpy directly
# ---------------------------------------------------------------------------

def test_compute_summary_statistics_matches_numpy() -> None:
    experiments = _condition(CONDITION_A_ACC, dropout=0.0)

    result = StatisticalAnalyzer().compute_summary_statistics(experiments)

    assert result.mean == pytest.approx(np.mean(CONDITION_A_ACC))
    assert result.std == pytest.approx(np.std(CONDITION_A_ACC))
    assert result.min == pytest.approx(np.min(CONDITION_A_ACC))
    assert result.max == pytest.approx(np.max(CONDITION_A_ACC))
    assert result.median == pytest.approx(np.median(CONDITION_A_ACC))
    assert result.count == 5


# ---------------------------------------------------------------------------
# 10. compute_summary_statistics with zero successful experiments raises
# ---------------------------------------------------------------------------

def test_compute_summary_statistics_no_data_raises() -> None:
    experiments = [_exp(0.0, seed=1, status="failed")]

    with pytest.raises(InsufficientDataError):
        StatisticalAnalyzer().compute_summary_statistics(experiments)
