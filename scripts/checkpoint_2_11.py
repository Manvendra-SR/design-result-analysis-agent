import sys, uuid
from datetime import datetime

print("=== Phase 2 Checkpoint 2.11: PostgreSQL Verification ===")
print()

from backend.models import (
    ExperimentConfiguration, ExperimentResult,
    AnomalyReport, StatisticalComparison, SummaryStatistics,
    Recommendation, SessionSummary
)
print("[OK] All 7 Pydantic models import cleanly from backend.models")

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

# 1. Create session
sid = sm.create_session("Does dropout improve MNIST classification performance?")
assert sid and len(sid) > 0
print("[OK] 1. create_session -> " + sid)

# 2. Retrieve session
session = sm.get_session(sid)
assert session.research_question == "Does dropout improve MNIST classification performance?"
assert session.status == "active"
assert session.current_node == "planning"
assert session.cycle_count == 0
print("[OK] 2. get_session -> status=" + session.status + ", node=" + session.current_node)

# 3. Store experiment with JSONB config
cfg = ExperimentConfiguration(
    model_type="mnist_mlp",
    hyperparameters={"dropout": 0.2, "learning_rate": 0.001, "batch_size": 32},
    random_seed=42
)
eid = str(uuid.uuid4())
result = ExperimentResult(
    experiment_id=eid,
    session_id=sid,
    config=cfg,
    metrics={"train_loss": 0.15, "val_loss": 0.18, "accuracy": 0.94, "training_time_seconds": 45.2},
    status="success",
    timestamp=datetime.utcnow()
)
sm.store_experiment(result)
print("[OK] 3. store_experiment -> " + eid)

# 4. Retrieve experiment - full JSONB round-trip
retrieved = sm.get_experiment(eid)
assert retrieved.experiment_id == eid
assert retrieved.session_id == sid
assert retrieved.status == "success"
assert retrieved.config.model_type == "mnist_mlp"
assert abs(retrieved.config.hyperparameters["dropout"] - 0.2) < 1e-9
assert retrieved.config.random_seed == 42
assert abs(retrieved.metrics["accuracy"] - 0.94) < 1e-9
print("[OK] 4. get_experiment (JSONB round-trip) -> model_type=" + retrieved.config.model_type + ", accuracy=" + str(retrieved.metrics["accuracy"]))

# 5. Store anomaly
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
print("[OK] 5. store_anomaly -> " + aid)

# 6. Query anomaly by session (JOIN path)
anomalies_by_session = sm.query_anomalies(session_id=sid)
assert len(anomalies_by_session) == 1
assert anomalies_by_session[0].experiment_id == eid
assert anomalies_by_session[0].rule == "outlier_detection"
print("[OK] 6. query_anomalies(session_id=...) JOIN -> " + str(len(anomalies_by_session)) + " anomaly found")

# 7. FK relationship: second experiment same session
eid2 = str(uuid.uuid4())
cfg2 = ExperimentConfiguration(
    model_type="mnist_mlp",
    hyperparameters={"dropout": 0.5, "learning_rate": 0.001, "batch_size": 32},
    random_seed=99
)
result2 = ExperimentResult(
    experiment_id=eid2, session_id=sid, config=cfg2,
    metrics={"train_loss": 0.22, "val_loss": 0.25, "accuracy": 0.91, "training_time_seconds": 44.0},
    status="success", timestamp=datetime.utcnow()
)
sm.store_experiment(result2)
all_exp = sm.query_experiments(sid)
assert len(all_exp) == 2
print("[OK] 7. FK relationship: " + str(len(all_exp)) + " experiments linked to session " + sid[:8] + "...")

# 8. list_sessions with COUNT join
summaries = sm.list_sessions()
our_summary = next(s for s in summaries if s.session_id == sid)
assert our_summary.experiment_count == 2
print("[OK] 8. list_sessions COUNT join -> experiment_count=" + str(our_summary.experiment_count))

# 9. update experiment status (anomalous)
sm.update_experiment_status(eid, "anomalous")
updated = sm.get_experiment(eid)
assert updated.status == "anomalous"
print("[OK] 9. update_experiment_status -> " + updated.status)

# 10. update session status  
sm.update_session_status(sid, "concluded")
session2 = sm.get_session(sid)
assert session2.status == "concluded"
print("[OK] 10. update_session_status -> " + session2.status)

# 11. Pydantic validator rejects invalid hyperparameters
try:
    bad = ExperimentConfiguration(
        model_type="mnist_mlp",
        hyperparameters={"dropout": 1.5, "learning_rate": 0.001, "batch_size": 32},
        random_seed=1
    )
    print("[FAIL] Should have raised for dropout=1.5")
    sys.exit(1)
except Exception:
    print("[OK] 11. ExperimentConfiguration validator rejects dropout=1.5")

print()
print("=== ALL 11 CHECKS PASSED ===")
print("Test session ID: " + sid)
print("Inspect in pgAdmin: database=design_analytic_agent, tables: sessions, experiments, anomalies")
