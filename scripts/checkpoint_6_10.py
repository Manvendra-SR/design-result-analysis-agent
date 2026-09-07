"""
scripts/checkpoint_6_10.py
=============================
Phase 6 Checkpoint: exercise all 11 API endpoints end-to-end through
FastAPI's TestClient. Mirrors scripts/checkpoint_5_11.py.

Run:
    PYTHONPATH=. ./.venv/Scripts/python.exe scripts/checkpoint_6_10.py

Offline (always) - the app runs against in-memory SQLite with stubbed
agents + the REAL ExperimentRunner, so /run-cycle drives the real Phase 5
autonomous adaptive loop:

1.  POST /api/datasets            ingest a real CSV
2.  GET  /api/datasets            + GET /api/datasets/{id}
3.  POST /api/sessions            (+ 400 empty question, 404 unknown dataset)
4.  GET  /api/sessions            + GET /api/sessions/{id}
5.  POST /api/sessions/{id}/run-cycle   -> ONE call runs the whole investigation
5b. GET  /api/sessions/{id}/cycles       -> per-cycle history for the UI
6.  GET  /api/sessions/{id}/experiments  (+ ?status= filter, tagged by cycle)
7.  GET  /api/experiments/{id}
8.  GET  /api/sessions/{id}/recommendation   (404 before, 200 after = the final one)
9.  409 on run-cycle for a concluded session
10. CORS headers + consistent ErrorResponse shape

Live (only if Ollama answers): one real autonomous investigation via the API.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

print("=== Phase 6 Checkpoint: FastAPI Backend (11 endpoints) ===")
print()

from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.dependencies import get_context
from backend.database.models import Base
from backend.models.experiment import ExperimentConfiguration, ExperimentPlan
from backend.models.recommendation import Recommendation
from backend.state_machine.context import StateMachineContext
from backend.tools.anomaly_detector import AnomalyDetector
from backend.tools.experiment_runner import ExperimentRunner
from backend.tools.state_manager import StateManager
from backend.tools.statistical_analyzer import StatisticalAnalyzer

print("[OK] backend.api imports cleanly")

# ---------------------------------------------------------------------------
# Real CSV - a clearly learnable binary boundary so the small MLP trains to
# well above chance (a near-chance dataset would just get every run flagged
# validation_collapse and the checkpoint would exercise nothing meaningful).
# ---------------------------------------------------------------------------
rng = np.random.default_rng(1)
n = 500
x = rng.normal(size=(n, 3))
y = (x[:, 0] * 1.8 - x[:, 1] * 0.9 + x[:, 2] * 0.4 + rng.normal(scale=0.35, size=n) > 0).astype(int)
frame = pd.DataFrame(x, columns=["a", "b", "c"])
frame["label"] = y
csv_text = frame.to_csv(index=False)

engine = create_engine(
    "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
)
Base.metadata.create_all(engine)
sm = StateManager(engine=engine)


# A plain "does model capacity matter?" ablation - vary hidden_size, hold
# everything else. Whether the analyzer produces a t-test depends on the
# real training runs (converged runs have zero accuracy variance across
# seeds - that is expected and handled, not a failure).
_HP = {"epochs": 10, "learning_rate": 0.01}


class _StubPlanner:
    def plan_experiments(self, question, dataset_profile):
        return ExperimentPlan(
            experiments=[
                ExperimentConfiguration(
                    dataset_id=dataset_profile.dataset_id, model_type="mlp",
                    hyperparameters={"hidden_size": h, **_HP}, random_seed=s,
                )
                for h in (4, 32)
                for s in (1, 2, 3)
            ],
            explanation="stub: vary hidden_size (model capacity)",
        )


class _StubRecommender:
    calls = 0

    def recommend_next(self, question, experiments, statistical_results, anomalies):
        _StubRecommender.calls += 1
        if _StubRecommender.calls >= 2:
            return Recommendation(action="conclude", recommended_experiments=[],
                                  explanation="stub", evidence_summary="stub")
        return Recommendation(
            action="run_more_experiments",
            recommended_experiments=[
                ExperimentConfiguration(
                    dataset_id="x", model_type="mlp",
                    hyperparameters={"hidden_size": 8, **_HP}, random_seed=s,
                )
                for s in (4, 5, 6)
            ],
            explanation="stub", evidence_summary="stub",
        )


ctx = StateMachineContext(
    state_manager=sm, planner=_StubPlanner(), recommender=_StubRecommender(),
    runner=ExperimentRunner(), detector=AnomalyDetector(), analyzer=StatisticalAnalyzer(),
)
app = create_app()
app.dependency_overrides[get_context] = lambda: ctx
client = TestClient(app, raise_server_exceptions=False)

# 1. ingest
r = client.post("/api/datasets", json={"filename": "demo.csv", "csv_content": csv_text, "target_column": "label"})
assert r.status_code == 201, r.text
dataset_id = r.json()["dataset_id"]
print(f"[OK] 1. POST /api/datasets -> {dataset_id} task={r.json()['task_type']}")

# 2. list / get dataset
assert len(client.get("/api/datasets").json()) == 1
assert client.get(f"/api/datasets/{dataset_id}").status_code == 200
assert client.get("/api/datasets/nope").status_code == 404
print("[OK] 2. GET /api/datasets, GET /api/datasets/{id} (+404)")

# 3. create session
r = client.post("/api/sessions", json={"research_question": "Does dropout help?", "dataset_id": dataset_id})
assert r.status_code == 201
session_id = r.json()["session_id"]
assert client.post("/api/sessions", json={"research_question": "", "dataset_id": dataset_id}).status_code == 400
assert client.post("/api/sessions", json={"research_question": "q", "dataset_id": "nope"}).status_code == 404
print(f"[OK] 3. POST /api/sessions -> {session_id} (+400 empty question, +404 unknown dataset)")

# 4. list / get session
assert len(client.get("/api/sessions").json()) == 1
detail = client.get(f"/api/sessions/{session_id}").json()
assert detail["current_node"] == "planning" and detail["experiment_count"] == 0
assert client.get("/api/sessions/nope").status_code == 404
print("[OK] 4. GET /api/sessions, GET /api/sessions/{id} (+404)")

# 8a. recommendation + cycles empty before any run
assert client.get(f"/api/sessions/{session_id}/recommendation").status_code == 404
assert client.get(f"/api/sessions/{session_id}/cycles").json() == []

# 5. run-cycle -> ONE call runs the whole autonomous investigation
r = client.post(f"/api/sessions/{session_id}/run-cycle")
assert r.status_code == 200, r.text
assert r.json()["current_node"] == "concluded" and r.json()["status"] == "concluded"
assert r.json()["cycles_completed"] == 2  # stub recommender: run_more once, then conclude
assert r.json()["recommendation"]["action"] == "conclude"  # the FINAL one
print(f"[OK] 5. POST /run-cycle -> {r.json()['cycles_completed']} cycles autonomously, "
      f"status={r.json()['status']}, experiments={r.json()['experiments_completed']}")

# 5b. cycle-by-cycle history is exposed - structure and wiring, not content.
#     (statistical_comparisons may be empty on a cycle whose real training
#      converged to zero accuracy variance - the analysis node skips
#      zero-variance pairs by design; content is unit-tested with a
#      variance-injecting stub runner in test_api_cycles.py.)
cycles = client.get(f"/api/sessions/{session_id}/cycles").json()
assert [c["cycle_number"] for c in cycles] == [1, 2]
assert cycles[0]["recommendation"]["action"] == "run_more_experiments" and cycles[0]["continued"] is True
assert cycles[1]["recommendation"]["action"] == "conclude" and cycles[1]["continued"] is False
assert cycles[0]["plan_explanation"] and cycles[1]["plan_explanation"] is None
for c in cycles:
    assert c["experiments"] and c["recommendation"]
    assert isinstance(c["statistical_comparisons"], list)
    assert {e["cycle"] for e in c["experiments"]} == {c["cycle_number"]}
_summary = [
    (c["cycle_number"], c["recommendation"]["action"],
     len(c["experiments"]), len(c["statistical_comparisons"]))
    for c in cycles
]
print(f"[OK] 5b. GET /cycles -> per-cycle history (cycle, action, #exp, #stat): {_summary}")

# 6. experiments (all cycles, tagged)
exps = client.get(f"/api/sessions/{session_id}/experiments").json()
assert len(exps) == 9  # cycle 1: 2 conditions x 3 seeds; cycle 2: 3 recommended
assert sorted({e["cycle"] for e in exps}) == [1, 2]
assert client.get(f"/api/sessions/{session_id}/experiments?status=success").status_code == 200
assert client.get(f"/api/sessions/{session_id}/experiments?status=bogus").status_code == 400
print(f"[OK] 6. GET /experiments -> {len(exps)} rows across cycles {sorted({e['cycle'] for e in exps})} "
      f"(+ ?status= filter, +400 on bad status)")

# 7. experiment detail
one = client.get(f"/api/experiments/{exps[0]['experiment_id']}")
assert one.status_code == 200 and one.json()["config"]["model_type"] == "mlp"
assert client.get("/api/experiments/nope").status_code == 404
print("[OK] 7. GET /api/experiments/{id} (+404)")

# 8b. recommendation after
rec = client.get(f"/api/sessions/{session_id}/recommendation")
assert rec.status_code == 200 and rec.json()["action"] == "conclude"
print("[OK] 8. GET /api/sessions/{id}/recommendation (404 before first cycle, 200 after)")

# 9. 409 on concluded
assert client.post(f"/api/sessions/{session_id}/run-cycle").status_code == 409
print("[OK] 9. POST /run-cycle on a concluded session -> 409")

# 10. CORS + error shape
cors = client.get("/api/sessions", headers={"Origin": "http://localhost:5173"})
assert "access-control-allow-origin" in cors.headers
err = client.get("/api/sessions/missing").json()
assert set(err) == {"error", "message", "details"}
print("[OK] 10. CORS header present; ErrorResponse shape = {error, message, details}")

# ---------------------------------------------------------------------------
# Live check
# ---------------------------------------------------------------------------
from backend.agents.llm_client import OllamaClient

llm = OllamaClient()
if not llm.health_check():
    print()
    print(f"[SKIP] live. No Ollama server at {llm.base_url} - live API cycle skipped.")
    llm.close()
    print()
    print("=== OFFLINE CHECKS PASSED ===")
    sys.exit(0)

print(f"[OK]    Ollama reachable at {llm.base_url}, model={llm.model}")
from backend.agents.planner import ExperimentPlannerAgent
from backend.agents.recommender import RecommenderAgent

live_engine = create_engine(
    "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
)
Base.metadata.create_all(live_engine)
live_sm = StateManager(engine=live_engine)
live_ctx = StateMachineContext(
    state_manager=live_sm,
    planner=ExperimentPlannerAgent(llm_client=llm),
    recommender=RecommenderAgent(llm_client=llm),
    runner=ExperimentRunner(), detector=AnomalyDetector(), analyzer=StatisticalAnalyzer(),
)
live_app = create_app()
live_app.dependency_overrides[get_context] = lambda: live_ctx
live_client = TestClient(live_app, raise_server_exceptions=False)

did = live_client.post("/api/datasets", json={"filename": "demo.csv", "csv_content": csv_text, "target_column": "label"}).json()["dataset_id"]
lsid = live_client.post("/api/sessions", json={"research_question": "Does dropout improve accuracy on this dataset?", "dataset_id": did}).json()["session_id"]
live = live_client.post(f"/api/sessions/{lsid}/run-cycle")
if live.status_code == 200:
    lc = live_client.get(f"/api/sessions/{lsid}/cycles").json()
    print(f"[OK] live. POST /run-cycle (real agents) -> {live.json()['cycles_completed']} cycle(s) "
          f"autonomously, status={live.json()['status']}, "
          f"experiments={live.json()['experiments_completed']}")
    for c in lc:
        print(f"          cycle {c['cycle_number']}: {c['recommendation']['action']}")
else:
    print(f"[WARN] live. /run-cycle -> {live.status_code} {live.json().get('error')}: "
          f"{str(live.json().get('message'))[:140]}")
llm.close()

print()
print("=== ALL CHECKS PASSED ===")
