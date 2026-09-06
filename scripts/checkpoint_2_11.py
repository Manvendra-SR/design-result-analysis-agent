import sys, uuid
from datetime import datetime

print("=== Phase 2 Checkpoint 2.12: PostgreSQL Verification (dataset-first architecture) ===")
print()

from backend.models import (
    DatasetProfile, ExperimentConfiguration, ExperimentResult,
    AnomalyReport, StatisticalComparison, SummaryStatistics,
    Recommendation, SessionSummary
)
print("[OK] All Pydantic models import cleanly from backend.models")

from backend.database.connection import test_connection
conn_ok = test_connection()
if not conn_ok:
    print("[FAIL] Cannot connect to PostgreSQL. Check DATABASE_URL in .env")
    sys.exit(1)
print("[OK] PostgreSQL connection verified")

from backend.database.init_db import init_db
init_db()
print("[OK] Schema applied (idempotent)")

from backend.tools.state_manager import StateManager
sm = StateManager()

# 1. Create a dataset (dataset-first architecture: sessions are scoped to a dataset)
profile = DatasetProfile(
    dataset_id=str(uuid.uuid4()),
    original_filename="checkpoint_2_12.csv",
    storage_path="data/uploads/checkpoint-2-12/data.csv",
    target_column="churned",
    feature_columns=["tenure_months", "monthly_charges", "contract_type"],
    numeric_columns=["tenure_months", "monthly_charges"],
    categorical_columns=["contract_type"],
    task_type="classification",
    n_rows=500,
    n_features=5,
    n_classes=2,
    class_labels=["no", "yes"],
    class_distribution={"no": 350, "yes": 150},
    missing_value_counts={"tenure_months": 3},
    split_seed=42,
)
did = sm.create_dataset(profile)
assert did == profile.dataset_id
print("[OK] 1. create_dataset -> " + did)

# 2. Retrieve dataset - full JSONB round-trip
retrieved_ds = sm.get_dataset(did)
assert retrieved_ds.task_type == "classification"
assert retrieved_ds.n_classes == 2
assert retrieved_ds.class_labels == ["no", "yes"]
assert retrieved_ds.split_seed == 42
print("[OK] 2. get_dataset (JSONB round-trip) -> task_type=" + retrieved_ds.task_type + ", n_classes=" + str(retrieved_ds.n_classes))

# 3. list_datasets includes it
all_datasets = sm.list_datasets()
assert any(d.dataset_id == did for d in all_datasets)
print("[OK] 3. list_datasets -> " + str(len(all_datasets)) + " dataset(s), including ours")

# 4. Create session scoped to the dataset
sid = sm.create_session("Does model complexity improve performance on this dataset?", did)
assert sid and len(sid) > 0
print("[OK] 4. create_session -> " + sid)

# 5. Retrieve session
session = sm.get_session(sid)
assert session.dataset_id == did
assert session.status == "active"
assert session.current_node == "planning"
assert session.cycle_count == 0
print("[OK] 5. get_session -> dataset_id=" + session.dataset_id + ", status=" + session.status)

# 6. Store experiment with JSONB config (new dataset_id/preprocessing shape)
cfg = ExperimentConfiguration(
    dataset_id=did,
    model_type="mlp",
    hyperparameters={"dropout": 0.2, "learning_rate": 0.001, "batch_size": 32,
                      "hidden_size": 64, "epochs": 20},
    random_seed=42
)
eid = str(uuid.uuid4())
result = ExperimentResult(
    experiment_id=eid,
    session_id=sid,
    config=cfg,
    task_type="classification",
    metrics={"train_loss": 0.15, "val_loss": 0.18, "accuracy": 0.94,
             "n_classes": 2, "n_val_samples": 75, "training_time_seconds": 4.2},
    status="success",
    timestamp=datetime.utcnow()
)
sm.store_experiment(result)
print("[OK] 6. store_experiment -> " + eid)

# 7. Retrieve experiment - full JSONB round-trip, including task_type column
retrieved = sm.get_experiment(eid)
assert retrieved.experiment_id == eid
assert retrieved.session_id == sid
assert retrieved.status == "success"
assert retrieved.task_type == "classification"
assert retrieved.config.dataset_id == did
assert retrieved.config.model_type == "mlp"
assert abs(retrieved.config.hyperparameters["dropout"] - 0.2) < 1e-9
assert retrieved.config.random_seed == 42
assert abs(retrieved.metrics["accuracy"] - 0.94) < 1e-9
print("[OK] 7. get_experiment (JSONB + task_type round-trip) -> model_type=" + retrieved.config.model_type + ", accuracy=" + str(retrieved.metrics["accuracy"]))

# 8. Store anomaly
aid = str(uuid.uuid4())
anomaly = AnomalyReport(
    anomaly_id=aid,
    experiment_id=eid,
    rule="outlier_detection",
    explanation="Validation val_loss of 2.3 is 3.7 std devs from mean (0.18).",
    severity="warning",
    detected_at=datetime.utcnow()
)
sm.store_anomaly(anomaly)
print("[OK] 8. store_anomaly -> " + aid)

# 9. Query anomaly by session (JOIN path)
anomalies_by_session = sm.query_anomalies(session_id=sid)
assert len(anomalies_by_session) == 1
assert anomalies_by_session[0].experiment_id == eid
assert anomalies_by_session[0].rule == "outlier_detection"
print("[OK] 9. query_anomalies(session_id=...) JOIN -> " + str(len(anomalies_by_session)) + " anomaly found")

# 10. FK relationship: second experiment (linear_baseline this time) same session
eid2 = str(uuid.uuid4())
cfg2 = ExperimentConfiguration(
    dataset_id=did,
    model_type="linear_baseline",
    random_seed=99
)
result2 = ExperimentResult(
    experiment_id=eid2, session_id=sid, config=cfg2, task_type="classification",
    metrics={"train_loss": 0.30, "val_loss": 0.33, "accuracy": 0.87,
             "n_classes": 2, "n_val_samples": 75, "training_time_seconds": 0.2},
    status="success", timestamp=datetime.utcnow()
)
sm.store_experiment(result2)
all_exp = sm.query_experiments(sid)
assert len(all_exp) == 2
print("[OK] 10. FK relationship: " + str(len(all_exp)) + " experiments (mlp + linear_baseline) linked to session " + sid[:8] + "...")

# 11. list_sessions with COUNT join
summaries = sm.list_sessions()
our_summary = next(s for s in summaries if s.session_id == sid)
assert our_summary.experiment_count == 2
print("[OK] 11. list_sessions COUNT join -> experiment_count=" + str(our_summary.experiment_count))

# 12. update experiment status (anomalous)
sm.update_experiment_status(eid, "anomalous")
updated = sm.get_experiment(eid)
assert updated.status == "anomalous"
print("[OK] 12. update_experiment_status -> " + updated.status)

# 13. update session status
sm.update_session_status(sid, "concluded")
session2 = sm.get_session(sid)
assert session2.status == "concluded"
print("[OK] 13. update_session_status -> " + session2.status)

# 14. Pydantic validator rejects invalid hyperparameters
try:
    bad = ExperimentConfiguration(
        dataset_id=did,
        model_type="mlp",
        hyperparameters={"dropout": 1.5, "learning_rate": 0.001, "batch_size": 32},
        random_seed=1
    )
    print("[FAIL] Should have raised for dropout=1.5")
    sys.exit(1)
except Exception:
    print("[OK] 14. ExperimentConfiguration validator rejects dropout=1.5")

print()
print("=== ALL 14 CHECKS PASSED ===")
print("Test dataset ID: " + did)
print("Test session ID: " + sid)
print("Inspect in pgAdmin: database=design_analytic_agent, tables: datasets, sessions, experiments, anomalies")
