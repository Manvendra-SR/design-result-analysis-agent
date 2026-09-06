"""
tests/unit/test_anomaly_detector.py
======================================
Unit tests for backend/tools/anomaly_detector.py.

Test cases
----------
1. test_outlier_detection_flags_known_outlier
2. test_no_outliers_for_normal_experiments
3. test_outlier_grouping_ignores_random_seed_but_respects_hyperparameters
4. test_outlier_grouping_separates_different_datasets
5. test_loss_divergence_detected_when_loss_increases
6. test_no_loss_divergence_when_loss_decreases
7. test_loss_divergence_skipped_without_initial_train_loss_key
8. test_validation_collapse_detected_for_low_mnist_shaped_accuracy
9. test_validation_collapse_not_checked_for_regression
10. test_validation_collapse_binary_classification_not_misfired_by_naive_multiplier
11. test_validation_collapse_binary_classification_flags_near_chance_accuracy
12. test_severity_mapping
13. test_failed_and_already_anomalous_experiments_are_ignored
14. test_explanation_uses_correct_placeholders
"""

from __future__ import annotations

from typing import Dict, Optional

from backend.models.experiment import ExperimentConfiguration, ExperimentResult
from backend.tools.anomaly_detector import AnomalyDetector


def _mlp_cfg(
    dropout: float = 0.2, seed: int = 0, dataset_id: str = "ds-a"
) -> ExperimentConfiguration:
    return ExperimentConfiguration(
        dataset_id=dataset_id,
        model_type="mlp",
        hyperparameters={"dropout": dropout, "learning_rate": 0.001, "batch_size": 32},
        random_seed=seed,
    )


def _regression_cfg(seed: int = 0, dataset_id: str = "ds-a") -> ExperimentConfiguration:
    return ExperimentConfiguration(
        dataset_id=dataset_id,
        model_type="mlp",
        hyperparameters={"epochs": 5},
        random_seed=seed,
    )


def _exp(
    config: ExperimentConfiguration,
    val_loss: float = 0.18,
    train_loss: float = 0.15,
    accuracy: Optional[float] = 0.95,
    n_classes: Optional[int] = 10,
    n_val_samples: Optional[int] = 1000,
    initial_train_loss: Optional[float] = None,
    task_type: str = "classification",
    status: str = "success",
) -> ExperimentResult:
    metrics: Optional[Dict[str, float]] = None
    if status != "failed":
        metrics = {"train_loss": train_loss, "val_loss": val_loss, "training_time_seconds": 1.0}
        if task_type == "classification":
            metrics["accuracy"] = accuracy
            metrics["n_classes"] = n_classes
            metrics["n_val_samples"] = n_val_samples
        if initial_train_loss is not None:
            metrics["initial_train_loss"] = initial_train_loss

    return ExperimentResult(
        session_id="s",
        config=config,
        task_type=task_type if status != "failed" else None,  # type: ignore[arg-type]
        metrics=metrics,
        status=status,  # type: ignore[arg-type]
        error="boom" if status == "failed" else None,
    )


# ---------------------------------------------------------------------------
# 1-2. Outlier detection
# ---------------------------------------------------------------------------

def test_outlier_detection_flags_known_outlier() -> None:
    # Leave-one-out z-scores verified numerically: with rest=[0.18,0.19,0.17,0.20]
    # (std ~0.011), a value of 0.5 has z ~28 (flagged); none of the 4 normal
    # points trigger against their own leave-one-out baseline (max |z| ~0.73).
    cfg = _mlp_cfg(dropout=0.2)
    normal = [_exp(cfg, val_loss=v) for v in [0.18, 0.19, 0.17, 0.20]]
    outlier = _exp(_mlp_cfg(dropout=0.2, seed=99), val_loss=0.5)

    anomalies = AnomalyDetector().detect_anomalies(normal + [outlier])

    outlier_reports = [a for a in anomalies if a.rule == "outlier_detection"]
    assert len(outlier_reports) == 1
    assert outlier_reports[0].experiment_id == outlier.experiment_id
    assert outlier_reports[0].severity == "warning"


def test_no_outliers_for_normal_experiments() -> None:
    cfg = _mlp_cfg(dropout=0.2)
    experiments = [_exp(cfg, val_loss=v) for v in [0.18, 0.19, 0.17, 0.20, 0.18]]

    anomalies = AnomalyDetector().detect_anomalies(experiments)

    assert [a for a in anomalies if a.rule == "outlier_detection"] == []


def test_outlier_grouping_ignores_random_seed_but_respects_hyperparameters() -> None:
    # Same hyperparameters, different seeds -> one group (outlier detectable).
    group_dropout_2 = [
        _exp(_mlp_cfg(dropout=0.2, seed=i), val_loss=v)
        for i, v in enumerate([0.18, 0.19, 0.17, 0.20])
    ]
    # Different hyperparameters -> separate group, own (tiny) distribution;
    # with only 1 experiment, no outlier can be established for this group.
    group_dropout_5 = [_exp(_mlp_cfg(dropout=0.5, seed=0), val_loss=0.30)]

    anomalies = AnomalyDetector().detect_anomalies(group_dropout_2 + group_dropout_5)

    assert [a for a in anomalies if a.rule == "outlier_detection"] == []


def test_outlier_grouping_separates_different_datasets() -> None:
    # dataset_id is now part of the group key: identical hyperparameters on
    # two different datasets must never be pooled together. Dataset A's
    # normal cluster (4 points) plus dataset B's single point (val_loss=0.5,
    # which WOULD read as an obvious outlier if pooled with A) must not
    # produce any anomaly for B - a lone point can't establish a baseline.
    dataset_a = [
        _exp(_mlp_cfg(dropout=0.2, seed=i, dataset_id="ds-a"), val_loss=v)
        for i, v in enumerate([0.18, 0.19, 0.17, 0.20])
    ]
    dataset_b = [_exp(_mlp_cfg(dropout=0.2, seed=0, dataset_id="ds-b"), val_loss=0.5)]

    anomalies = AnomalyDetector().detect_anomalies(dataset_a + dataset_b)

    assert [a for a in anomalies if a.rule == "outlier_detection"] == []


# ---------------------------------------------------------------------------
# 5-7. Loss divergence
# ---------------------------------------------------------------------------

def test_loss_divergence_detected_when_loss_increases() -> None:
    cfg = _mlp_cfg()
    exp = _exp(cfg, train_loss=0.50, initial_train_loss=0.20)

    anomalies = AnomalyDetector().detect_anomalies([exp])

    divergence = [a for a in anomalies if a.rule == "loss_divergence"]
    assert len(divergence) == 1
    assert divergence[0].experiment_id == exp.experiment_id
    assert divergence[0].severity == "warning"


def test_no_loss_divergence_when_loss_decreases() -> None:
    cfg = _mlp_cfg()
    exp = _exp(cfg, train_loss=0.10, initial_train_loss=0.20)

    anomalies = AnomalyDetector().detect_anomalies([exp])

    assert [a for a in anomalies if a.rule == "loss_divergence"] == []


def test_loss_divergence_skipped_without_initial_train_loss_key() -> None:
    # linear_baseline (and any single-shot fit) never sets initial_train_loss.
    cfg = _regression_cfg()
    exp = _exp(cfg, train_loss=100.0, task_type="regression", accuracy=None,
               n_classes=None, n_val_samples=None)  # no initial_train_loss

    anomalies = AnomalyDetector().detect_anomalies([exp])

    assert [a for a in anomalies if a.rule == "loss_divergence"] == []


# ---------------------------------------------------------------------------
# 8-11. Validation collapse (class-count-aware z-test)
# ---------------------------------------------------------------------------

def test_validation_collapse_detected_for_low_mnist_shaped_accuracy() -> None:
    # Reproduces the original MNIST-shaped scenario: 10 classes, large n_val,
    # accuracy barely above the 10% chance baseline -> not statistically
    # distinguishable from guessing -> still flagged, same as the old flat
    # "< 20%" rule would have caught.
    cfg = _mlp_cfg()
    exp = _exp(cfg, accuracy=0.11, n_classes=10, n_val_samples=1000)

    anomalies = AnomalyDetector().detect_anomalies([exp])

    collapse = [a for a in anomalies if a.rule == "validation_collapse"]
    assert len(collapse) == 1
    assert collapse[0].experiment_id == exp.experiment_id
    assert collapse[0].severity == "critical"


def test_validation_collapse_not_checked_for_regression() -> None:
    # Regression experiments have no "accuracy" key at all - task_type gates
    # this rule, not a hardcoded model_type check.
    cfg = _regression_cfg()
    exp = _exp(cfg, task_type="regression", accuracy=None, n_classes=None, n_val_samples=None)

    anomalies = AnomalyDetector().detect_anomalies([exp])

    assert [a for a in anomalies if a.rule == "validation_collapse"] == []


def test_validation_collapse_binary_classification_not_misfired_by_naive_multiplier() -> None:
    # The rejected "2x baseline" multiplier would give threshold=1.0 for
    # n_classes=2 (baseline=50%), flagging almost any non-perfect model.
    # The z-test must NOT flag a comfortably-above-chance binary result.
    cfg = _mlp_cfg()
    exp = _exp(cfg, accuracy=0.60, n_classes=2, n_val_samples=100)

    anomalies = AnomalyDetector().detect_anomalies([exp])

    assert [a for a in anomalies if a.rule == "validation_collapse"] == []


def test_validation_collapse_binary_classification_flags_near_chance_accuracy() -> None:
    # A binary classifier barely beating chance (52% with n=100) should
    # still be flagged - the z-test is class-count-aware, not "never fires
    # for binary".
    cfg = _mlp_cfg()
    exp = _exp(cfg, accuracy=0.52, n_classes=2, n_val_samples=100)

    anomalies = AnomalyDetector().detect_anomalies([exp])

    assert len([a for a in anomalies if a.rule == "validation_collapse"]) == 1


# ---------------------------------------------------------------------------
# 12. Severity mapping
# ---------------------------------------------------------------------------

def test_severity_mapping() -> None:
    cfg = _mlp_cfg()
    low_acc = _exp(cfg, accuracy=0.05, train_loss=0.1, initial_train_loss=0.05)

    anomalies = AnomalyDetector().detect_anomalies([low_acc])

    by_rule = {a.rule: a.severity for a in anomalies}
    assert by_rule["validation_collapse"] == "critical"
    assert by_rule["loss_divergence"] == "warning"


# ---------------------------------------------------------------------------
# 13. Failed / already-anomalous experiments are ignored entirely
# ---------------------------------------------------------------------------

def test_failed_and_already_anomalous_experiments_are_ignored() -> None:
    cfg = _mlp_cfg()
    failed = _exp(cfg, status="failed")
    already_anomalous = _exp(cfg, accuracy=0.05, status="anomalous")

    anomalies = AnomalyDetector().detect_anomalies([failed, already_anomalous])

    assert anomalies == []


# ---------------------------------------------------------------------------
# 14. Explanation text uses the correct template placeholders
# ---------------------------------------------------------------------------

def test_explanation_uses_correct_placeholders() -> None:
    cfg = _mlp_cfg()
    exp = _exp(cfg, accuracy=0.11, n_classes=10, n_val_samples=1000)

    anomalies = AnomalyDetector().detect_anomalies([exp])
    collapse = next(a for a in anomalies if a.rule == "validation_collapse")

    assert "11.0%" in collapse.explanation
    assert "10.0%" in collapse.explanation  # baseline = 1/10
    assert "10-class" in collapse.explanation
    assert "not statistically distinguishable from random guessing" in collapse.explanation
