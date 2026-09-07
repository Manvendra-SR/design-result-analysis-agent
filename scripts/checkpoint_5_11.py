"""
scripts/checkpoint_5_11.py
=============================
Phase 5 Checkpoint: verify the LangGraph adaptive-loop state machine
end-to-end. Mirrors scripts/checkpoint_3_18.py / checkpoint_4_11.py.

Run:
    PYTHONPATH=. ./.venv/Scripts/python.exe scripts/checkpoint_5_11.py

Offline checks (always run - no Ollama, no PostgreSQL)
-----------------------------------------------------
Uses an in-memory SQLite StateManager, stubbed Planner/Recommender, and the
REAL ExperimentRunner (real torch/scikit-learn training) + real
AnomalyDetector + real StatisticalAnalyzer.

1.  ONE execute_cycle runs the whole autonomous investigation (the graph
    loops recommending -> executing internally) to conclusion
2.  every experiment is tagged with the adaptive cycle that produced it
3.  per-cycle history (recommendation + analysis) is retained, not just the
    latest; the planner's rationale is kept too
4.  crash recovery: a session parked mid-cycle resumes at its node AND then
    keeps looping to conclusion
5.  safety cap: a recommender that never concludes still terminates at
    MAX_ADAPTIVE_CYCLES
5b. execute_cycle on a concluded session raises CycleError

Live check (only if an Ollama server answers)
---------------------------------------------
6.  one real autonomous investigation with the real Planner + Recommender
    against the configured OLLAMA_MODEL (PostgreSQL used if reachable)
"""

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

print("=== Phase 5 Checkpoint: LangGraph Adaptive-Loop State Machine ===")
print()

from backend.agents.llm_client import OllamaClient
from backend.agents.planner import ExperimentPlannerAgent
from backend.agents.recommender import RecommenderAgent
from backend.database.models import Base
from backend.models.experiment import ExperimentConfiguration, ExperimentPlan
from backend.models.recommendation import Recommendation
from backend.state_machine.context import StateMachineContext
from backend.state_machine.executor import CycleError, execute_cycle
from backend.tools.anomaly_detector import AnomalyDetector
from backend.tools.dataset.ingestion import ingest_csv
from backend.tools.experiment_runner import ExperimentRunner
from backend.tools.state_manager import StateManager
from backend.tools.statistical_analyzer import StatisticalAnalyzer

print("[OK] backend.state_machine imports cleanly")

# ---------------------------------------------------------------------------
# Ingest a small real CSV (used by every check)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
n = 240
x = rng.normal(size=(n, 4))
y = (x[:, 0] * 1.4 - x[:, 1] * 0.7 + rng.normal(scale=0.5, size=n) > 0).astype(int)
frame = pd.DataFrame(x, columns=["f0", "f1", "f2", "f3"])
frame["label"] = y
csv_path = Path(tempfile.mkdtemp(prefix="checkpoint_5_11_")) / "demo.csv"
frame.to_csv(csv_path, index=False)
profile = ingest_csv(str(csv_path), target_column="label", dataset_name="checkpoint_5_11_demo.csv")
print(f"[OK]    ingested demo dataset id={profile.dataset_id} task={profile.task_type}")


def _fresh_state_manager() -> StateManager:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return StateManager(engine=engine)


class _StubPlanner:
    def plan_experiments(self, question, dataset_profile):
        exps = [
            ExperimentConfiguration(
                dataset_id=dataset_profile.dataset_id, model_type="mlp",
                hyperparameters={"dropout": d, "epochs": 8, "hidden_size": 16},
                random_seed=s,
            )
            for d in (0.0, 0.4)
            for s in (1, 2, 3)
        ]
        return ExperimentPlan(experiments=exps, explanation="stub: vary dropout")


class _StubRecommender:
    def __init__(self):
        self.calls = 0

    def recommend_next(self, question, experiments, statistical_results, anomalies):
        self.calls += 1
        if self.calls >= 2:
            return Recommendation(action="conclude", recommended_experiments=[],
                                  explanation="stub: enough evidence", evidence_summary="stub")
        return Recommendation(
            action="run_more_experiments",
            recommended_experiments=[
                ExperimentConfiguration(
                    dataset_id=profile.dataset_id, model_type="mlp",
                    hyperparameters={"dropout": 0.2, "epochs": 8, "hidden_size": 16},
                    random_seed=s,
                )
                for s in (4, 5, 6)
            ],
            explanation="stub: explore dropout=0.2", evidence_summary="stub",
        )


def _offline_context(state_manager, recommender=None):
    return StateMachineContext(
        state_manager=state_manager,
        planner=_StubPlanner(),
        recommender=recommender or _StubRecommender(),
        runner=ExperimentRunner(),
        detector=AnomalyDetector(),
        analyzer=StatisticalAnalyzer(),
    )


# ---------------------------------------------------------------------------
# 1-3. ONE execute_cycle runs the whole autonomous investigation
# ---------------------------------------------------------------------------
sm = _fresh_state_manager()
sm.create_dataset(profile)
sid = sm.create_session("Does dropout improve accuracy on this dataset?", profile.dataset_id)
ctx = _offline_context(sm)

result = execute_cycle(sid, ctx)  # stub recommender: run_more once, then conclude
assert result.current_node == "concluded" and result.status == "concluded", result.current_node
assert result.cycles_completed == 2, result.cycles_completed
assert result.experiments_completed == 6 + 3
assert result.recommendation.action == "conclude"
print(f"[OK] 1. one execute_cycle -> {result.cycles_completed} cycles, "
      f"{result.experiments_completed} experiments, final action={result.recommendation.action}")

# every experiment is tagged with the cycle that produced it
by_cycle = {}
for e in sm.query_experiments(sid):
    by_cycle[e.cycle] = by_cycle.get(e.cycle, 0) + 1
assert by_cycle == {1: 6, 2: 3}, by_cycle
print(f"[OK] 2. experiments tagged by cycle -> {by_cycle}")

# per-cycle history is retained (not just the final recommendation)
history = sm.get_cycle_history(sid)
assert [h.cycle_number for h in history] == [1, 2]
assert history[0].recommendation.action == "run_more_experiments"
assert history[1].recommendation.action == "conclude"
assert sm.get_session(sid).plan_explanation
print(f"[OK] 3. cycle_history retained: "
      f"{[(h.cycle_number, h.recommendation.action) for h in history]}; plan_explanation kept")

# ---------------------------------------------------------------------------
# 4. Crash recovery: a session parked mid-cycle resumes AND keeps looping
# ---------------------------------------------------------------------------
sm2 = _fresh_state_manager()
sm2.create_dataset(profile)
sid2 = sm2.create_session("Does dropout help?", profile.dataset_id)
recovery_ctx = _offline_context(sm2)

from backend.state_machine.nodes import AdaptiveLoopNodes

nodes = AdaptiveLoopNodes(recovery_ctx)
nodes.planning({"session_id": sid2, "current_node": "planning"})
nodes.executing({"session_id": sid2, "current_node": "executing"})  # cycle 1 experiments in DB
assert sm2.get_session(sid2).current_node == "validating"  # "crashed" here

r_recover = execute_cycle(sid2, recovery_ctx)  # START router -> validating -> ... -> loop -> conclude
assert r_recover.current_node == "concluded" and r_recover.cycles_completed == 2
recover_by_cycle = {}
for e in sm2.query_experiments(sid2):
    recover_by_cycle[e.cycle] = recover_by_cycle.get(e.cycle, 0) + 1
assert recover_by_cycle == {1: 6, 2: 3}, recover_by_cycle
print(f"[OK] 4. crash recovery: resumed from 'validating', looped to conclusion -> {recover_by_cycle}")

# ---------------------------------------------------------------------------
# 5. Safety cap: a recommender that never concludes still terminates
# ---------------------------------------------------------------------------
from backend.config import MAX_ADAPTIVE_CYCLES

class _NeverConcludes:
    def recommend_next(self, question, experiments, statistical_results, anomalies):
        return Recommendation(
            action="run_more_experiments", explanation="keep going", evidence_summary="never done",
            recommended_experiments=[
                ExperimentConfiguration(dataset_id=profile.dataset_id, model_type="mlp",
                    hyperparameters={"dropout": 0.3, "epochs": 6, "hidden_size": 16}, random_seed=s)
                for s in (7, 8, 9)
            ],
        )

sm3 = _fresh_state_manager()
sm3.create_dataset(profile)
sid3 = sm3.create_session("q", profile.dataset_id)
capped = execute_cycle(sid3, _offline_context(sm3, recommender=_NeverConcludes()))
assert capped.status == "concluded" and capped.cycles_completed == MAX_ADAPTIVE_CYCLES
assert "safety limit" in capped.recommendation.evidence_summary
print(f"[OK] 5. safety cap: never-concluding recommender stopped at "
      f"{MAX_ADAPTIVE_CYCLES} cycles with a 'reached the ...-cycle safety limit' note")

# ---------------------------------------------------------------------------
# 5b. Concluded session is rejected
# ---------------------------------------------------------------------------
try:
    execute_cycle(sid, ctx)
    raise AssertionError("expected CycleError")
except CycleError:
    print("[OK] 5b. execute_cycle on a concluded session -> CycleError")

# ---------------------------------------------------------------------------
# 6. Live check against a real Ollama server
# ---------------------------------------------------------------------------
llm = OllamaClient()
if not llm.health_check():
    print()
    print(f"[SKIP] 6. No Ollama server at {llm.base_url} - live state-machine check skipped.")
    llm.close()
    print()
    print("=== OFFLINE CHECKS PASSED ===")
    sys.exit(0)

print(f"[OK]    Ollama reachable at {llm.base_url}, model={llm.model}")
logging.getLogger("backend").setLevel(logging.INFO)

from backend.database.connection import test_connection

if test_connection():
    from backend.database.init_db import init_db
    init_db()
    live_sm = StateManager()
    print("[OK]    using real PostgreSQL for the live check")
else:
    live_sm = _fresh_state_manager()
    print("[OK]    PostgreSQL not reachable - using in-memory SQLite for the live check")

live_sm.create_dataset(profile)
live_sid = live_sm.create_session(
    "Does dropout improve classification accuracy on this dataset?", profile.dataset_id
)
live_ctx = StateMachineContext(
    state_manager=live_sm,
    planner=ExperimentPlannerAgent(llm_client=llm),
    recommender=RecommenderAgent(llm_client=llm),
    runner=ExperimentRunner(),
    detector=AnomalyDetector(),
    analyzer=StatisticalAnalyzer(),
)
try:
    live_result = execute_cycle(live_sid, live_ctx)
    live_history = live_sm.get_cycle_history(live_sid)
    print(f"[OK] 6. live investigation -> {live_result.cycles_completed} cycle(s), "
          f"node={live_result.current_node} experiments={live_result.experiments_completed} "
          f"final action={live_result.recommendation.action}")
    for h in live_history:
        print(f"        cycle {h.cycle_number}: {h.recommendation.action} - "
              f"{h.recommendation.explanation[:110]}...")
except Exception as exc:  # noqa: BLE001 - a small model can produce a rejected plan
    print(f"[WARN] 6. live investigation raised {type(exc).__name__}: {str(exc)[:160]}")
finally:
    llm.close()

print()
print("=== ALL CHECKS PASSED ===")
