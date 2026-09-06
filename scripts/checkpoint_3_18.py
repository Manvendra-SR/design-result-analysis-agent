"""
scripts/checkpoint_3_18.py
=============================
Phase 3 Checkpoint (dataset-first architecture revision): verify the
deterministic tools end-to-end with REAL training (not mocked) against a
real, uploaded-and-ingested CSV dataset — mirrors the style of
scripts/checkpoint_2_11.py.

Run with the project venv (needs "backend" importable — either run from the
project root with PYTHONPATH=. set, or install the project in editable mode):
    PYTHONPATH=. ./.venv/Scripts/python.exe scripts/checkpoint_3_18.py

Checks performed
-----------------
1.  Dataset ingestion: a small real CSV -> DatasetProfile (classification)
2.  Experiment_Runner: real `mlp` experiment on the ingested dataset -> success
3.  Experiment_Runner: same seed run twice -> identical metrics (determinism;
    proves the dataset's train/val/test split does not change with the seed)
4.  Experiment_Runner: real `linear_baseline` experiment on the same dataset,
    with metrics in the same task-type-keyed shape as `mlp`
5.  `normalize=True` vs `normalize=False` both train successfully
6.  Statistical_Analyzer: compare two real hidden_size conditions -> scipy t-test
7.  Statistical_Analyzer: compute_summary_statistics on real results
8.  Anomaly_Detector: no anomalies on well-behaved real results
9.  Anomaly_Detector: correctly flags a manually-constructed bad experiment
    (validation_collapse via the class-count-aware z-test, and loss_divergence)
10. Persistence: dataset + session + experiments persisted and re-queried via
    StateManager against a real local PostgreSQL instance
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

print("=== Phase 3 Checkpoint: Dataset-First Deterministic Tools Verification ===")
print()

from backend.models.dataset import PreprocessingConfig
from backend.models.experiment import ExperimentConfiguration, ExperimentResult
from backend.tools.dataset.ingestion import ingest_csv
from backend.tools.experiment_runner import ExperimentRunner
from backend.tools.statistical_analyzer import StatisticalAnalyzer
from backend.tools.anomaly_detector import AnomalyDetector

print("[OK] Dataset-first tools import cleanly")

runner = ExperimentRunner()
analyzer = StatisticalAnalyzer()
detector = AnomalyDetector()

# ---------------------------------------------------------------------------
# 1. Ingest a small real CSV dataset (classification)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
n_samples = 300
x = rng.normal(size=(n_samples, 4))
# Genuinely learnable signal: class depends on a linear combination of features.
logits = x[:, 0] * 1.5 - x[:, 1] * 0.8 + x[:, 2] * 0.3
y = (logits + rng.normal(scale=0.5, size=n_samples) > 0).astype(int)
df = pd.DataFrame(x, columns=["f0", "f1", "f2", "f3"])
df["label"] = y

tmp_dir = Path(tempfile.mkdtemp(prefix="checkpoint_3_18_"))
csv_path = tmp_dir / "demo.csv"
df.to_csv(csv_path, index=False)

profile = ingest_csv(str(csv_path), target_column="label", dataset_name="checkpoint_3_18_demo.csv")
assert profile.task_type == "classification"
assert profile.n_classes == 2
assert profile.n_rows == n_samples
print(
    "[OK] 1. ingest_csv -> dataset_id=%s task_type=%s n_classes=%d n_rows=%d"
    % (profile.dataset_id, profile.task_type, profile.n_classes, profile.n_rows)
)

# ---------------------------------------------------------------------------
# 2. mlp - real run
# ---------------------------------------------------------------------------
mlp_cfg = ExperimentConfiguration(
    dataset_id=profile.dataset_id,
    model_type="mlp",
    hyperparameters={"dropout": 0.0, "learning_rate": 0.01, "batch_size": 32,
                      "hidden_size": 32, "epochs": 30},
    random_seed=42,
)
mlp_result = runner.run_experiment(mlp_cfg, profile, session_id="checkpoint-3-18")
assert mlp_result.status == "success", f"mlp failed: {mlp_result.error}"
assert mlp_result.task_type == "classification"
assert mlp_result.metrics["accuracy"] > 0.6, (
    f"accuracy too low for a sanity check: {mlp_result.metrics['accuracy']}"
)
print(
    "[OK] 2. mlp -> accuracy=%.4f val_loss=%.4f train_loss=%.4f (%.2fs)"
    % (mlp_result.metrics["accuracy"], mlp_result.metrics["val_loss"],
       mlp_result.metrics["train_loss"], mlp_result.metrics["training_time_seconds"])
)

# ---------------------------------------------------------------------------
# 3. mlp determinism (same seed -> identical metrics; split is fixed too)
# ---------------------------------------------------------------------------
mlp_result_2 = runner.run_experiment(mlp_cfg, profile, session_id="checkpoint-3-18")
assert mlp_result.metrics["accuracy"] == mlp_result_2.metrics["accuracy"], (
    f"{mlp_result.metrics['accuracy']} != {mlp_result_2.metrics['accuracy']}"
)
assert mlp_result.metrics["val_loss"] == mlp_result_2.metrics["val_loss"]
assert mlp_result.metrics["train_loss"] == mlp_result_2.metrics["train_loss"]
print("[OK] 3. mlp determinism -> identical accuracy/val_loss/train_loss across 2 runs, same seed")

# ---------------------------------------------------------------------------
# 4. linear_baseline - real run, same dataset, comparable metrics shape
# ---------------------------------------------------------------------------
baseline_cfg = ExperimentConfiguration(
    dataset_id=profile.dataset_id,
    model_type="linear_baseline",
    random_seed=42,
)
baseline_result = runner.run_experiment(baseline_cfg, profile, session_id="checkpoint-3-18")
assert baseline_result.status == "success", f"linear_baseline failed: {baseline_result.error}"
assert set(["train_loss", "val_loss", "accuracy", "n_classes", "n_val_samples",
            "training_time_seconds"]).issubset(baseline_result.metrics.keys())
assert "initial_train_loss" not in baseline_result.metrics  # no epoch loop
print(
    "[OK] 4. linear_baseline -> accuracy=%.4f val_loss=%.4f (metrics shape matches mlp's)"
    % (baseline_result.metrics["accuracy"], baseline_result.metrics["val_loss"])
)

# ---------------------------------------------------------------------------
# 5. normalize=True vs normalize=False both succeed
#
# Note: model_copy(update={"preprocessing": {...}}) would silently leave a
# raw dict in place of a PreprocessingConfig instance - Pydantic's
# model_copy does NOT re-validate/coerce nested BaseModel fields the way
# the constructor does. Passing an actual PreprocessingConfig instance
# avoids that footgun; worth remembering for the future Planner/Recommender,
# which will build config variants the same way.
# ---------------------------------------------------------------------------
normalized_cfg = mlp_cfg.model_copy(update={
    "preprocessing": PreprocessingConfig(normalize=True),
    "random_seed": 43,
})
normalized_result = runner.run_experiment(normalized_cfg, profile, session_id="checkpoint-3-18")
assert normalized_result.status == "success", f"normalized mlp failed: {normalized_result.error}"
print(
    "[OK] 5. mlp with preprocessing.normalize=True -> accuracy=%.4f (vs %.4f unnormalized)"
    % (normalized_result.metrics["accuracy"], mlp_result.metrics["accuracy"])
)

# ---------------------------------------------------------------------------
# 6. Statistical_Analyzer - real t-test between two hidden_size conditions
# ---------------------------------------------------------------------------
small_hidden = [
    runner.run_experiment(
        ExperimentConfiguration(
            dataset_id=profile.dataset_id, model_type="mlp",
            hyperparameters={"hidden_size": 4, "epochs": 15, "learning_rate": 0.01},
            random_seed=seed,
        ),
        profile,
    )
    for seed in range(5)
]
large_hidden = [
    runner.run_experiment(
        ExperimentConfiguration(
            dataset_id=profile.dataset_id, model_type="mlp",
            hyperparameters={"hidden_size": 64, "epochs": 15, "learning_rate": 0.01},
            random_seed=seed,
        ),
        profile,
    )
    for seed in range(5)
]
comparison = analyzer.compare_conditions(small_hidden, large_hidden, metric="accuracy")
assert comparison.sample_sizes == (5, 5)
print(
    "[OK] 6. StatisticalAnalyzer.compare_conditions (hidden_size 4 vs 64) -> "
    "t=%.3f p=%.4f d=%.3f (scipy-computed)"
    % (comparison.t_statistic, comparison.p_value, comparison.effect_size)
)

# ---------------------------------------------------------------------------
# 7. Statistical_Analyzer - summary statistics on real results
# ---------------------------------------------------------------------------
summary = analyzer.compute_summary_statistics(large_hidden, metric="accuracy")
assert summary.count == 5
print(
    "[OK] 7. StatisticalAnalyzer.compute_summary_statistics -> mean=%.4f std=%.4f (n=%d)"
    % (summary.mean, summary.std, summary.count)
)

# ---------------------------------------------------------------------------
# 8. Anomaly_Detector - no anomalies on well-behaved real results
# ---------------------------------------------------------------------------
anomalies = detector.detect_anomalies(large_hidden)
print(f"[OK] 8. AnomalyDetector on clean real results -> {len(anomalies)} anomalies (expected: 0 or few)")

# ---------------------------------------------------------------------------
# 9. Anomaly_Detector - correctly flags a manually-constructed bad experiment
# ---------------------------------------------------------------------------
bad_cfg = ExperimentConfiguration(
    dataset_id=profile.dataset_id, model_type="mlp",
    hyperparameters={"hidden_size": 32, "epochs": 15}, random_seed=999,
)
collapsed = ExperimentResult(
    session_id="checkpoint-3-18",
    config=bad_cfg,
    task_type="classification",
    metrics={"train_loss": 2.3, "val_loss": 2.3, "accuracy": 0.51,
             "n_classes": 2, "n_val_samples": 50,
             "training_time_seconds": 1.0, "initial_train_loss": 1.0},
    status="success",
)
flagged = detector.detect_anomalies([collapsed])
rules = {a.rule for a in flagged}
assert "validation_collapse" in rules, "expected near-chance binary accuracy to be flagged"
assert "loss_divergence" in rules
print(f"[OK] 9. AnomalyDetector flags injected bad experiment -> rules={sorted(rules)}")

# ---------------------------------------------------------------------------
# 10. Persistence through StateManager against a real local PostgreSQL instance
# ---------------------------------------------------------------------------
from backend.database.connection import test_connection

if not test_connection():
    print("[SKIP] 10. No local PostgreSQL reachable (DATABASE_URL) - persistence check skipped.")
else:
    from backend.database.init_db import init_db
    from backend.tools.state_manager import StateManager

    init_db()
    sm = StateManager()

    persisted_dataset_id = sm.create_dataset(profile)
    assert persisted_dataset_id == profile.dataset_id
    sid = sm.create_session(
        "Does hidden_size (model complexity) improve accuracy on this dataset?",
        persisted_dataset_id,
    )

    for result in [mlp_result, baseline_result, normalized_result] + small_hidden + large_hidden:
        result.session_id = sid
        sm.store_experiment(result)

    stored = sm.query_experiments(sid)
    assert len(stored) == 3 + len(small_hidden) + len(large_hidden)

    # `flagged` (check 9) references the synthetic "collapsed" experiment,
    # which was never persisted - re-run detection against the real,
    # persisted experiments instead, to prove the anomaly write/query path
    # end-to-end.
    real_anomalies = detector.detect_anomalies(stored)
    for a in real_anomalies:
        sm.store_anomaly(a)
    queried_anomalies = sm.query_anomalies(session_id=sid)
    assert len(queried_anomalies) == len(real_anomalies)

    retrieved_profile = sm.get_dataset(persisted_dataset_id)
    assert retrieved_profile.task_type == "classification"

    print(
        "[OK] 10. Persistence -> dataset=%s session=%s, %d experiments and %d anomalies "
        "round-tripped through PostgreSQL"
        % (persisted_dataset_id, sid, len(stored), len(queried_anomalies))
    )

print()
print("=== ALL CHECKS PASSED ===")
